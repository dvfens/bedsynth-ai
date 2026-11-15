"""SDV synthesizer wrapper (global or per-segment models) - DESIGN.md §7.3.

:class:`BedSynthSynthesizer` wraps one SDV single-table synthesizer (``global`` mode) or one
synthesizer per *segment*. A segment is a value of one column (``segment_by="ward_id"``: one
model per ward) or a value combination of several columns (``segment_by=["ward_id",
"admission_source"]``: one model per ward and admission source). In settings.yaml
``segment_by`` may also be a ``{table: column(s)}`` mapping, resolved for the table when the
wrapper is created (tables not listed are modelled globally). The wrapper adds the
hospital-specific behaviour the raw SDV API does not provide:

* **Segmentation** - each segment gets its own model, fitted on its rows without the segment
  column(s). Columns that are constant inside a segment (AMU never books electives, emergency
  stays are never booked ahead...) are dropped before fitting, remembered, and re-inserted
  when sampling, so they are reproduced exactly. Segment-specific supports come for free: SDV
  clips numeric columns to their training range, so the elective stays of a ward x source
  model are admitted between 07:00 and 14:00 like the reference electives.
* **Constraints** - the experiment's constraint groups (``constraints: true | false |
  [groups]``, :mod:`bedsynth.synthesis.constraints`) are built per segment with the segment's
  constants and key available (band labels substitute ``elective_scheduled = 0`` on AMU and
  read the ward's bed capacity; constraints whose columns are all constant are skipped). The
  learned state of programmable constraints (e.g. the structural zeros found per segment) is
  recorded in the model card.
* **Ordered encodings** (GaussianCopula) - calendar and ward categoricals are re-encoded with
  ``rdt``'s ``OrderedUniformEncoder`` ordered by an operational statistic (months by mean
  occupancy, weekdays by mean discharges, wards by bed capacity...), and so are the merged
  outcome combinations of patient_stays' ``FixedCombinations`` (by mean length of stay). The
  Gaussian copula models *linear* dependence in latent space, so a demand-ordered encoding lets
  it capture non-monotonic seasonal/weekly effects that an arbitrary category order would
  destroy. (An order by a statistic whose category means are indistinguishable - weekdays by
  elective lists - is itself arbitrary and imposes a false gradient; elective lists are modelled
  per weekday by the ``conditional_marginals`` constraint group instead.)
* **Marginal selection** (``gc_tuned``) - candidate marginals are compared per segment on a
  held-in subsample by the Kolmogorov-Smirnov statistic: the candidate with the lowest mean KS
  becomes the segment's ``default_distribution`` and every directly modelled numerical column
  keeps the candidate with its own lowest KS (``numerical_distributions``). After the final fit a
  post-fit check samples the model once and refits columns whose marginal degenerated on the
  full segment (an MLE gamma fit of length of stay produced 80 % zero-night stays) with a robust
  family. Choices, scores, overrides and fallbacks are recorded in the model card.
* **Conditional sampling** - see below.
* **Reproducibility** - every SDV call is preceded by ``reset_sampling()`` +
  ``_set_random_state(seed)`` (numpy seeding has no effect on SDV 2.x); child seeds are
  derived deterministically from the caller's seed, so ``sample(..., seed=s)`` is repeatable.
* **Output contract** - exactly ``schema.training_columns(table)`` in canonical order,
  integer columns ``int64``, floats rounded to the schema's decimals, categoricals typed like
  the training data (``month`` stays ``int``).
* **Backward compatibility** - pickles written before multi-column segmentation and
  constraint groups (``segment_by`` a plain string, SDV built-in constraints only) load and
  sample unchanged; their cards list the groups they were actually fitted with.

Conditional sampling strategy
-----------------------------
Probing SDV 2.2 showed two facts that shape the sampler:

1. Every SDV sampling call has a fixed cost of ~0.2-0.3 s (rdt/constraint reverse
   transforms), whatever the number of rows: 6 rows cost almost as much as 5,000. Calling
   ``sample_from_conditions`` once per condition cell (e.g. 84 month x weekday cells for a
   calendar year of one ward) therefore takes ~25 s.
2. GaussianCopula's native conditioning maps a categorical condition to a *single* latent
   point inside the category's interval (the rdt transform draw), which narrows the
   conditional distribution of the other columns; rejection sampling from the unconditional
   model is exact.

Requests are first routed to segments: conditions on every segment column select one segment,
conditions on some of them select the matching segments (rows apportioned by training support),
and the remaining conditions are served inside each segment. Requests are then served by
**pooled rejection**: per segment one unconditional draw,
sized for the most demanding request (``k / p`` rows, ``p`` = training frequency of the
condition), is routed to all requests whose conditions the rows satisfy. Requests that are too
rare to pool (``k / p`` above :data:`POOL_MAX_ROWS`) or have no training support use SDV's
native ``sample_from_conditions`` (GaussianCopula conditions in latent space and can
extrapolate to unseen combinations); if SDV raises, the request falls back to an oversampled
unconditional draw filtered on the conditions. Combinations never observed together whose
values were all observed are sized by the independence estimate (product of the values'
marginal shares) and pooled like the others. ``sample_many(..., should_stop=...)`` lets the
generator stop between segments when its time budget is spent. Conditions that cannot be produced at all
(unseen category values, combinations excluded by a ``FixedCombinations`` constraint,
mismatching constants) are detected up-front and return zero rows instead of letting SDV
burn its retry budget.
"""
from __future__ import annotations

import contextlib
import copy
import json
import logging
import math
import os
import threading
import time
import warnings
import zlib
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

import numpy as np
import pandas as pd

from bedsynth import schema
from bedsynth.config import get_seed, get_settings, project_relative
from bedsynth.synthesis.constraints import (
    BUILTIN_GROUPS,
    ENFORCEMENT_NOTE,
    build_constraints,
    describe_constraints,
    fixed_combination_groups,
    learned_state,
)
from bedsynth.synthesis.constraints import constraint_groups as resolve_constraint_groups
from bedsynth.synthesis.metadata import build_metadata

LOGGER = logging.getLogger(__name__)

MODEL_TYPES: tuple[str, ...] = ("gaussian_copula", "ctgan", "tvae", "copulagan")

# --- sampling tunables -------------------------------------------------------------------
POOL_MAX_ROWS: int = 50_000
"""Largest unconditional draw used to serve conditional requests by routing (per segment and
call; ~3 s for a GaussianCopula). Pools are sized by need, so the cap only binds for rare
conditions or models with low validity."""
POOL_MARGIN: float = 1.3
"""Over-draw factor applied to ``k / p`` so pooled requests are usually filled in one call."""
NATIVE_CALLS_PER_SEGMENT: int = 8
"""Upper bound on SDV ``sample_from_conditions`` calls per segment and ``sample_many`` call
(~0.3 s each): pooling serves many cells far more cheaply, native calls are reserved for the
rarest requests (unseen combinations first)."""
SDV_MAX_TRIES_PER_BATCH: int = 20
"""``max_tries_per_batch`` for SDV conditional sampling (SDV's default of 100 can take ~20 s
before raising on an impossible condition)."""
FALLBACK_MIN_PROBABILITY: float = 0.002
"""Probability floor used to size the oversampled fallback draw."""

# --- marginal selection ------------------------------------------------------------------
DEFAULT_DISTRIBUTION_CANDIDATES: tuple[str, ...] = ("truncnorm", "beta", "gamma", "norm")
TUNE_MAX_ROWS: int = 500
"""Rows per segment used to compare candidate distributions (beta/gamma MLE fits dominate the
cost: ~2-4 s per candidate at 1,000 rows)."""
POSTFIT_CHECK_ROWS: int = 2_000
"""Rows sampled after the final fit to detect marginals that degenerated on the full segment."""
POSTFIT_KS_FLOOR: float = 0.15
"""A column is degenerate when its post-fit KS exceeds ``max(this, 2 x tuned KS + 0.05)``."""
FALLBACK_DISTRIBUTION: str = "truncnorm"
"""Robust family used for marginals that degenerated (``norm`` if truncnorm itself failed)."""
_WRAPPER_PARAM_KEYS: tuple[str, ...] = ("tune_distributions", "distribution_candidates")
_COPULA_MODELS: frozenset[str] = frozenset({"gaussian_copula"})
_TORCH_MODELS: frozenset[str] = frozenset({"ctgan", "tvae", "copulagan"})
GLOBAL_SEGMENT = "__global__"

# Ordered-encoding rules: column -> (method, statistic column). Categories present in the data
# are sorted by the statistic (ties broken by the schema's natural category order).
# Weekdays are ordered by mean discharges (Saturday lowest, Monday's post-weekend peak highest):
# ordering them by mean elective lists turned statistically indistinguishable Mon-Fri means into
# an arbitrary order along which the linear latent link imposed a false gradient (Friday lists
# -36 %, Saturday 5x). Elective lists are now modelled per weekday by the conditional_marginals
# constraint group, so the weekday order only shapes the remaining weekly patterns.
ORDERING_RULES: dict[str, dict[str, tuple[str, str | None]]] = {
    schema.WARD_DAILY: {
        "month": ("mean", "occupancy_rate"),
        "day_of_week": ("mean", "discharges"),
        "ward_id": ("bed_capacity", None),
    },
    schema.PATIENT_STAYS: {
        "admission_month": ("mean", "los_nights"),
        "admission_day_of_week": ("natural", None),
        "ward_id": ("bed_capacity", None),
    },
}
#: Statistic that orders the combinations of FixedCombinations' merged columns (ordered encodings):
#: patient_stays (separation type, destination, delay reason) by mean length of stay, so the
#: copula links the delay reason to the length of stay.
JOINT_ORDERING_STATISTIC: dict[str, str] = {schema.PATIENT_STAYS: "los_nights"}


# =========================================================================================
# Small public helpers (shared with the registry and the generator)
# =========================================================================================
def _stable_int(key: Any) -> int:
    """Map a hashable key to a stable 32-bit integer (independent of PYTHONHASHSEED)."""
    if isinstance(key, (bool, np.bool_)):
        return int(key)
    if isinstance(key, (int, np.integer)):
        return int(key) & 0xFFFFFFFF
    return zlib.crc32(repr(key).encode("utf-8"))


def derive_seed(seed: int, *keys: Any) -> int:
    """Deterministic child seed in ``[0, 2**31 - 1]`` derived from ``seed`` and ``keys``.

    Uses :class:`numpy.random.SeedSequence` so child streams are statistically independent.
    """
    entropy = [_stable_int(seed)] + [_stable_int(k) for k in keys]
    return int(np.random.SeedSequence(entropy).generate_state(1, dtype=np.uint32)[0] & 0x7FFFFFFF)


def new_seed() -> int:
    """Draw a fresh random seed (used when callers pass ``seed=None``)."""
    return int(np.random.SeedSequence().generate_state(1, dtype=np.uint32)[0] & 0x7FFFFFFF)


def largest_remainder(total: int, weights: Sequence[float]) -> list[int]:
    """Apportion ``total`` integer units proportionally to ``weights`` (Hamilton method).

    Non-finite or negative weights count as zero. Ties in the fractional remainders are broken
    by position, so the result is deterministic. Returns all zeros when every weight is zero.
    """
    w = np.asarray(list(weights), dtype=float)
    w = np.where(np.isfinite(w) & (w > 0), w, 0.0)
    if total <= 0 or w.size == 0 or w.sum() <= 0:
        return [0] * int(w.size)
    quotas = w / w.sum() * int(total)
    base = np.floor(quotas).astype(int)
    remainder = int(total) - int(base.sum())
    order = sorted(range(w.size), key=lambda i: (-(quotas[i] - base[i]), i))
    for i in order[:remainder]:
        base[i] += 1
    return [int(x) for x in base]


def to_python(value: Any) -> Any:
    """Convert a numpy/pandas scalar to a plain Python value (NaN/NA -> ``None``)."""
    if value is None:
        return None
    if isinstance(value, (np.generic,)):
        value = value.item()
    if isinstance(value, float) and math.isnan(value):
        return None
    if value is pd.NA or value is pd.NaT:
        return None
    return value


def json_safe(obj: Any) -> Any:
    """Recursively convert ``obj`` into JSON-serialisable Python types (NaN -> ``None``)."""
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [json_safe(v) for v in obj.tolist()]
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.generic, float)) or obj is None or obj is pd.NA:
        value = to_python(obj)
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value
    return obj


def empty_training_frame(table: str) -> pd.DataFrame:
    """Zero-row frame with ``schema.training_columns(table)`` and their canonical dtypes."""
    dtypes: dict[str, str] = {}
    for spec in schema.column_specs(table, ["train"]):
        if spec.sdtype == "numerical":
            dtypes[spec.name] = "int64" if spec.dtype == "int" else "float64"
        elif spec.categories and all(isinstance(v, int) for v in spec.categories):
            dtypes[spec.name] = "int64"
        else:
            dtypes[spec.name] = "object"
    return pd.DataFrame({c: pd.Series(dtype=d) for c, d in dtypes.items()})


def ks_statistic(real: pd.Series, synthetic: pd.Series) -> float:
    """Two-sample Kolmogorov-Smirnov statistic (1.0 when either sample is empty)."""
    from scipy.stats import ks_2samp

    a = pd.to_numeric(real, errors="coerce").dropna().to_numpy(dtype=float)
    b = pd.to_numeric(synthetic, errors="coerce").dropna().to_numpy(dtype=float)
    if a.size == 0 or b.size == 0:
        return 1.0
    return float(ks_2samp(a, b, method="asymp").statistic)


# =========================================================================================
# SDV plumbing
# =========================================================================================
@contextlib.contextmanager
def _quiet_sdv() -> Iterator[None]:
    """Silence SDV/rdt/copulas user warnings that are expected and handled by the wrapper.

    (Programmatic metadata, replaced transformers, partial conditional samples, training
    range notices and scipy fitting chatter.)
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, module=r"(sdv|rdt|copulas|ctgan)(\.|$)")
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"(scipy|copulas|numpy)(\.|$)")
        yield


@contextlib.contextmanager
def _seeded_global_rng(seed: int, with_torch: bool) -> Iterator[None]:
    """Seed numpy (and torch) global RNGs for a fit and restore the previous state afterwards.

    SDV does not expose a fit seed; GAN/VAE training draws from the global torch/numpy
    generators, so this makes CTGAN/TVAE training reproducible without leaking state.
    """
    np_state = np.random.get_state()
    np.random.seed(int(seed) % (2**32))
    with contextlib.ExitStack() as stack:
        if with_torch:
            import torch

            stack.enter_context(torch.random.fork_rng(devices=[]))
            torch.manual_seed(int(seed))
        try:
            yield
        finally:
            np.random.set_state(np_state)


def _synthesizer_class(model: str) -> type:
    """SDV synthesizer class for a BedSynth model type (imported lazily: torch is heavy)."""
    from sdv.single_table import (
        CopulaGANSynthesizer,
        CTGANSynthesizer,
        GaussianCopulaSynthesizer,
        TVAESynthesizer,
    )

    classes = {
        "gaussian_copula": GaussianCopulaSynthesizer,
        "ctgan": CTGANSynthesizer,
        "tvae": TVAESynthesizer,
        "copulagan": CopulaGANSynthesizer,
    }
    try:
        return classes[model]
    except KeyError as exc:
        raise ValueError(f"Unknown model type '{model}'. Expected one of {MODEL_TYPES}") from exc


def _sdv_sample(synthesizer: Any, table: str, num_rows: int) -> pd.DataFrame:
    """Unconditional SDV sample without progress bars.

    SDV 2.2's public ``sample()`` forces a tqdm bar whenever constraints exist
    (``disable=not show_progress_bar``), which bypasses ``TQDM_DISABLE`` and floods API/script
    logs; the internal sampler is called with the bar off when available.
    """
    sampler = getattr(synthesizer, "_sample_with_progress_bar", None)
    if sampler is None:  # pragma: no cover - defensive for other SDV versions
        return synthesizer.sample(num_rows)[table]
    synthesizer._validate_fit_before_sample()
    return sampler(int(num_rows), show_progress_bar=False)


def match_conditions(frame: pd.DataFrame, conditions: dict[str, Any]) -> np.ndarray:
    """Boolean mask of rows of ``frame`` that satisfy every ``column == value`` condition.

    A condition on a column missing from ``frame`` matches no row.
    """
    mask = np.ones(len(frame), dtype=bool)
    for column, value in conditions.items():
        if column not in frame.columns:
            return np.zeros(len(frame), dtype=bool)
        mask &= frame[column].eq(value).to_numpy(dtype=bool)
    return mask


# =========================================================================================
# Experiment configuration
# =========================================================================================
def segment_columns_of(setting: Any, table: str | None = None) -> tuple[str, ...]:
    """Normalise a ``segment_by`` setting to a tuple of column names (``()`` = global model).

    Args:
        setting: ``None``, one column name, a list of column names, or a ``{table: column(s)}``
            mapping (resolved for ``table``; tables missing from the mapping are global).
        table: Table used to resolve a mapping.

    Raises:
        ValueError: a mapping without ``table``, an unsupported type or duplicated columns.
    """
    if isinstance(setting, Mapping):
        if table is None:
            raise ValueError("A per-table segment_by mapping can only be resolved for a given table.")
        setting = setting.get(table)
    if setting is None or (isinstance(setting, str) and not setting.strip()):
        return ()
    if isinstance(setting, str):
        columns: tuple[str, ...] = (setting,)
    elif isinstance(setting, (list, tuple)):
        columns = tuple(str(c) for c in setting)
    else:
        raise ValueError(f"segment_by must be a column name, a list of column names or null, got {setting!r}")
    if len(set(columns)) != len(columns):
        raise ValueError(f"segment_by lists a column more than once: {list(columns)}")
    return columns


def normalise_constraints_setting(setting: Any) -> bool | list[str]:
    """``constraints`` value of an experiment: a YAML boolean stays a boolean, anything else
    (one group name, ``"all"``, ``"none"`` or a list of names) becomes the list of groups it selects.

    Raises:
        ValueError: unknown group names or an unsupported value type.
    """
    if setting is None:
        return False
    if isinstance(setting, (bool, np.bool_)):
        return bool(setting)
    return list(resolve_constraint_groups(setting))


def _segment_setting(columns: Sequence[str]) -> str | list[str] | None:
    """Canonical ``segment_by`` value: ``None``, one column name, or a list of names."""
    if not columns:
        return None
    return columns[0] if len(columns) == 1 else list(columns)


@dataclass
class ExperimentConfig:
    """One synthesis experiment = model family + configuration (settings.yaml: synthesis.experiments).

    ``segment_by`` is ``None`` (global model), one categorical column (``"ward_id"``: one SDV
    model per ward), a list of columns (``["ward_id", "admission_source"]``: one model per value
    combination) or a ``{table: column(s)}`` mapping resolved with :meth:`for_table`.
    ``constraints`` is ``True`` (every constraint group), ``False`` (none) or a list of groups of
    :data:`bedsynth.synthesis.constraints.CONSTRAINT_GROUPS`.
    """

    experiment_id: str
    model: str                                 # one of MODEL_TYPES
    segment_by: str | list[str] | dict[str, Any] | None = None   # column(s) or {table: column(s)}
    constraints: bool | list[str] = True       # True / False / list of constraint groups
    ordinal_metadata: bool = True              # declare ordered categoricals as SDV 'ordinal'
    ordered_encodings: bool = False            # data-driven OrderedUniformEncoder orders (GaussianCopula)
    params: dict[str, Any] = field(default_factory=dict)  # synthesizer constructor kwargs
    description: str = ""

    @classmethod
    def from_settings(cls, experiment_id: str, quick: bool = False, table: str | None = None) -> "ExperimentConfig":
        """Read ``synthesis.experiments[experiment_id]`` from settings.yaml.

        Args:
            experiment_id: Experiment key in settings.yaml.
            quick: Apply ``synthesis.quick_mode_params`` (e.g. 10 epochs for CTGAN/TVAE).
            table: Resolve a per-table ``segment_by`` mapping for this table.

        Raises:
            KeyError: unknown experiment. ValueError: invalid constraint groups / segment_by.
        """
        synthesis = get_settings()["synthesis"]
        try:
            raw = dict(synthesis["experiments"][experiment_id])
        except KeyError as exc:
            raise KeyError(
                f"Unknown experiment '{experiment_id}'. Known: {sorted(synthesis['experiments'])}"
            ) from exc
        params = dict(raw.get("params") or {})
        if quick:
            params.update((synthesis.get("quick_mode_params") or {}).get(raw["model"], {}))
        config = cls(
            experiment_id=experiment_id,
            model=raw["model"],
            segment_by=copy.deepcopy(raw.get("segment_by")),
            constraints=normalise_constraints_setting(raw.get("constraints", True)),
            ordinal_metadata=bool(raw.get("ordinal_metadata", True)),
            ordered_encodings=bool(raw.get("ordered_encodings", False)),
            params=params,
            description=raw.get("description", ""),
        )
        return config.for_table(table) if table is not None else config

    @property
    def constraint_groups(self) -> tuple[str, ...]:
        """Enabled constraint groups in canonical order."""
        return resolve_constraint_groups(self.constraints)

    def segment_columns(self, table: str | None = None) -> tuple[str, ...]:
        """Segment columns for ``table`` (``()`` = global model)."""
        return segment_columns_of(self.segment_by, table)

    def for_table(self, table: str) -> "ExperimentConfig":
        """Copy with ``segment_by`` resolved for ``table`` (``None``, a column or a list)."""
        return replace(self, segment_by=_segment_setting(self.segment_columns(table)), params=dict(self.params))

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly copy of the configuration."""
        return asdict(self)


# =========================================================================================
# Segment model
# =========================================================================================
@dataclass
class SegmentModel:
    """One fitted SDV synthesizer plus the bookkeeping needed to sample from it.

    Attributes:
        key: Segment value (e.g. ``"AMU"``) for one segment column, a tuple of values (e.g.
            ``("GMED", "elective")``) for several, ``None`` for a global model.
        n_rows: Training rows of the segment.
        columns: Columns modelled by SDV (segment column(s) and constant columns excluded).
        constants: Columns constant within the segment -> their value (re-inserted on sampling).
        support: Joint counts of the categorical train columns seen in training
            (columns + ``count``), used for feasibility checks and request sizing.
        synthesizer: Fitted SDV synthesizer (``None`` if every column is constant).
        constraints: Descriptions of the constraints applied to this segment (with the learned
            state of programmable constraints under ``learned``).
        fixed_combinations: Column groups of the applied FixedCombinations constraints.
    """

    key: Any
    n_rows: int
    columns: list[str]
    constants: dict[str, Any]
    support: pd.DataFrame
    synthesizer: Any = None
    constraints: list[dict[str, Any]] = field(default_factory=list)
    fixed_combinations: list[tuple[str, ...]] = field(default_factory=list)
    distribution: str | None = None
    distribution_scores: dict[str, float] = field(default_factory=dict)
    column_distributions: dict[str, str] = field(default_factory=dict)   # per-column overrides of distribution
    distribution_fallbacks: dict[str, dict[str, float]] = field(default_factory=dict)  # degenerate fits replaced
    ordered_encodings: dict[str, list] = field(default_factory=dict)
    fit_seconds: float = 0.0
    # Lazily built lookup indexes (calendar mode queries thousands of condition cells).
    _value_sets: dict[str, frozenset] = field(default_factory=dict, repr=False)
    _count_index: dict[tuple[str, ...], dict[tuple, int]] = field(default_factory=dict, repr=False)

    @property
    def label(self) -> str:
        """Printable segment name (``__global__`` for a global model, ``GMED/elective`` for a tuple key)."""
        if self.key is None:
            return GLOBAL_SEGMENT
        if isinstance(self.key, tuple):
            return "/".join(str(v) for v in self.key)
        return str(self.key)

    def values(self, column: str) -> frozenset:
        """Values of a categorical column observed in this segment's training rows."""
        if column not in self._value_sets:
            observed = self.support[column].unique() if column in self.support.columns else []
            self._value_sets[column] = frozenset(to_python(v) for v in observed)
        return self._value_sets[column]

    def count(self, conditions: dict[str, Any]) -> int:
        """Training rows of this segment matching ``conditions`` (all segment rows if empty)."""
        if not conditions:
            return int(self.n_rows)
        columns = tuple(sorted(conditions))
        index = self._count_index.get(columns)
        if index is None:
            if any(c not in self.support.columns for c in columns):
                return 0
            grouped = self.support.groupby(list(columns), dropna=False, sort=False)["count"].sum()
            index = {
                tuple(to_python(v) for v in (key if isinstance(key, tuple) else (key,))): int(n)
                for key, n in grouped.items()
            }
            self._count_index[columns] = index
        try:
            return index.get(tuple(to_python(conditions[c]) for c in columns), 0)
        except TypeError:  # unhashable condition value
            return 0

    def independence_probability(self, conditions: dict[str, Any]) -> float:
        """Share of the segment's rows matching ``conditions`` if the condition columns were
        independent (product of their marginal shares; 0 when a value was never observed).

        Used to size pooled draws for value combinations that never occurred together in training
        but whose values did (e.g. a calendar cell's occupancy band on an unseen weekday).
        """
        if not self.n_rows:
            return 0.0
        probability = 1.0
        for column, value in conditions.items():
            probability *= self.count({column: value}) / self.n_rows
        return probability

    def resolve_constants(self, conditions: dict[str, Any]) -> dict[str, Any] | None:
        """Drop conditions satisfied by constant columns; ``None`` if a constant contradicts one."""
        resolved = {}
        for column, value in conditions.items():
            if column in self.constants:
                if self.constants[column] != value:
                    return None
            else:
                resolved[column] = value
        return resolved

    def is_feasible(self, conditions: dict[str, Any]) -> bool:
        """Whether the model can produce rows matching ``conditions`` at all.

        A value never seen in training cannot be generated by SDV's categorical transformers,
        and a FixedCombinations group can only produce combinations observed in training.
        """
        if self.resolve_constants(conditions) is None:
            return False
        for column, value in conditions.items():
            try:
                if value not in self.values(column):
                    return False
            except TypeError:  # unhashable condition value
                return False
        for group in self.fixed_combinations:
            partial = {c: conditions[c] for c in group if c in conditions}
            if len(partial) >= 2 and self.count(partial) == 0:
                return False
        return True

    def to_card(self, total_rows: int, key_values: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """JSON-serialisable summary for the model card (``key_values``: segment column -> value)."""
        return json_safe({
            "segment": self.label,
            "segment_values": dict(key_values or {}),
            "rows": self.n_rows,
            "share": round(self.n_rows / total_rows, 6) if total_rows else None,
            "modelled_columns": list(self.columns),
            "constant_columns": dict(self.constants),
            "constraints": self.constraints,
            "distribution": self.distribution,
            "distribution_scores": self.distribution_scores,
            "column_distributions": dict(getattr(self, "column_distributions", None) or {}),
            "distribution_fallbacks": dict(getattr(self, "distribution_fallbacks", None) or {}),
            "ordered_encodings": self.ordered_encodings,
            "fit_seconds": round(self.fit_seconds, 3),
        })


# =========================================================================================
# The wrapper
# =========================================================================================
class BedSynthSynthesizer:
    """Wraps one global SDV single-table synthesizer, or one per segment value.

    Contract:
    * ``fit(train_df)`` uses only ``schema.training_columns(table)`` of ``train_df``.
    * ``sample(num_rows, conditions, seed)`` returns a DataFrame with exactly the training
      columns in canonical order and *up to* ``num_rows`` rows. ``conditions`` maps
      conditionable columns to a single value each (the generator expands lists). It never
      raises for infeasible conditions - it returns fewer (possibly zero) rows instead.
    * ``seed`` makes sampling reproducible (reset_sampling + _set_random_state).

    Args:
        table: ``"ward_daily"`` or ``"patient_stays"``.
        config: Experiment configuration (or an experiment id from settings.yaml); a per-table
            ``segment_by`` mapping is resolved for ``table``.
        tune_distributions: Compare candidate GaussianCopula marginals per segment. ``None``
            (default) enables it for GaussianCopula experiments that use ordered encodings
            (the ``gc_tuned`` recipe) unless ``params["tune_distributions"]`` says otherwise.
        distribution_candidates: Candidate ``default_distribution`` names (default
            :data:`DEFAULT_DISTRIBUTION_CANDIDATES`, or ``params["distribution_candidates"]``).
        seed: Fit seed (GAN/VAE training and tuning subsamples); default ``project.seed``.
    """

    def __init__(
        self,
        table: str,
        config: ExperimentConfig | str,
        *,
        tune_distributions: bool | None = None,
        distribution_candidates: Sequence[str] | None = None,
        seed: int | None = None,
    ):
        if table not in schema.TABLES:
            raise KeyError(f"Unknown table '{table}'. Known tables: {list(schema.TABLES)}")
        if isinstance(config, str):
            config = ExperimentConfig.from_settings(config, table=table)
        if config.model not in MODEL_TYPES:
            raise ValueError(f"Unknown model type '{config.model}'. Expected one of {MODEL_TYPES}")
        segment_columns = config.segment_columns(table)
        categorical = schema.categorical_columns(table)
        invalid = [c for c in segment_columns if c not in categorical]
        if invalid:
            raise ValueError(
                f"segment_by={config.segment_by!r} must name categorical train columns of '{table}': {categorical}"
            )
        self.table = table
        self.config = config.for_table(table)
        self._segment_columns: tuple[str, ...] = segment_columns
        self._constraint_groups: tuple[str, ...] = resolve_constraint_groups(config.constraints)
        params = dict(config.params or {})
        param_tune = params.pop("tune_distributions", None)
        param_candidates = params.pop("distribution_candidates", None)
        self._sdv_params: dict[str, Any] = params
        if tune_distributions is None:
            tune_distributions = (
                bool(param_tune)
                if param_tune is not None
                else config.model in _COPULA_MODELS and config.ordered_encodings
            )
        self.tune_distributions = bool(tune_distributions) and config.model in _COPULA_MODELS
        self.distribution_candidates: tuple[str, ...] = tuple(
            distribution_candidates or param_candidates or DEFAULT_DISTRIBUTION_CANDIDATES
        )
        self.seed = int(seed) if seed is not None else get_seed()
        self.training_columns: list[str] = schema.training_columns(table)
        self._categorical = schema.categorical_columns(table)
        self._segments: dict[Any, SegmentModel] = {}
        self._segment_order: list[Any] = []
        self._dtypes: dict[str, str] = {}
        self._training_rows = 0
        self._fit_seconds: float | None = None
        self._fitted_at: str | None = None
        self._versions: dict[str, str] = {}
        self._annotations: dict[str, Any] = {}
        self._lock = threading.RLock()

    # ------------------------------------------------------------------ properties
    @property
    def is_fitted(self) -> bool:
        """Whether :meth:`fit` has completed."""
        return bool(self._segments)

    @property
    def experiment_id(self) -> str:
        """Experiment id of the configuration (also the registry model id)."""
        return self.config.experiment_id

    @property
    def segment_by(self) -> str | list[str] | None:
        """Segment column(s) as configured: one column name, a list of names, or ``None`` (global)."""
        return self.config.segment_by

    @property
    def segment_columns(self) -> tuple[str, ...]:
        """Segment columns as a tuple (``()`` for a global model)."""
        return self._segment_columns

    @property
    def constraint_groups(self) -> tuple[str, ...]:
        """Constraint groups the SDV synthesizers were (or will be) fitted with."""
        return self._constraint_groups

    @property
    def segments(self) -> list[Any]:
        """Segment keys in sampling order (``[None]`` for a global model)."""
        return list(self._segment_order)

    @property
    def training_rows(self) -> int:
        """Rows used for fitting (after dropping incomplete rows)."""
        return self._training_rows

    @property
    def annotations(self) -> dict[str, Any]:
        """Training information attached with :meth:`annotate` (copy)."""
        return dict(self._annotations)

    def __repr__(self) -> str:
        state = f"fitted on {self._training_rows} rows" if self.is_fitted else "unfitted"
        mode = f"segment_by={self.segment_by}" if self._segment_columns else "global"
        return f"BedSynthSynthesizer(table={self.table!r}, experiment={self.experiment_id!r}, {mode}, {state})"

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state.pop("_lock", None)  # locks cannot be pickled
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore a pickle; pickles written before multi-column segmentation and constraint
        groups (``segment_by`` a string, SDV built-in constraints only) are upgraded in memory."""
        self.__dict__.update(state)
        self._lock = threading.RLock()
        if "_segment_columns" not in state:
            self._segment_columns = segment_columns_of(self.config.segment_by, self.table)
        if "_constraint_groups" not in state:
            self._constraint_groups = BUILTIN_GROUPS if self.config.constraints else ()

    # ------------------------------------------------------------------ segment keys
    def _key_values(self, key: Any) -> dict[str, Any]:
        """Segment key -> ``{segment column: value}`` (``{}`` for a global model)."""
        columns = self._segment_columns
        if not columns or key is None:
            return {}
        if len(columns) == 1:
            return {columns[0]: key}
        return dict(zip(columns, key))

    def _key_matches(self, key: Any, values: Mapping[str, Any]) -> bool:
        """Whether a segment key agrees with the given ``{segment column: value}`` conditions."""
        key_values = self._key_values(key)
        return all(key_values.get(column) == value for column, value in values.items())

    def _split_conditions(self, conditions: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        """``(conditions on segment columns, remaining conditions)``."""
        fixed = {c: v for c, v in conditions.items() if c in self._segment_columns}
        rest = {c: v for c, v in conditions.items() if c not in self._segment_columns}
        return fixed, rest

    # ================================================================== fitting
    def fit(self, train_df: pd.DataFrame) -> "BedSynthSynthesizer":
        """Fit the synthesizer(s) on the training columns of ``train_df``.

        Rows with missing training values are dropped (logged): the processed train split has
        none, and generated data must not contain any.
        """
        missing = [c for c in self.training_columns if c not in train_df.columns]
        if missing:
            raise ValueError(f"Training data for '{self.table}' is missing train columns {missing}")
        data = self._prepare_training_frame(train_df[self.training_columns])
        if data.empty:
            raise ValueError(f"No usable training rows for '{self.table}'.")
        with self._lock:
            started = time.perf_counter()
            self._segments, self._segment_order = {}, []
            self._dtypes = {c: str(data[c].dtype) for c in self.training_columns}
            for key, part in self._iter_segments(data):
                LOGGER.info("Fitting %s/%s segment %s (%d rows)", self.table, self.experiment_id,
                            _segment_label(key), len(part))
                self._segments[key] = self._fit_segment(key, part)
                self._segment_order.append(key)
            self._training_rows = len(data)
            self._fit_seconds = time.perf_counter() - started
            self._fitted_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
            self._versions = _library_versions()
        LOGGER.info("Fitted %r in %.1fs", self, self._fit_seconds)
        return self

    def _prepare_training_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Canonical dtypes for training: int64 counts/months, float64 metrics, object labels."""
        data = frame.copy()
        for column in self.training_columns:
            spec = schema.column_spec(self.table, column)
            if spec.sdtype == "numerical" or self._is_integer_category(column):
                data[column] = pd.to_numeric(data[column], errors="coerce")
            else:
                data[column] = data[column].astype(object).where(data[column].notna(), None)
        incomplete = data.isna().any(axis=1)
        if incomplete.any():
            LOGGER.warning("Dropping %d training rows with missing values", int(incomplete.sum()))
            data = data.loc[~incomplete]
        for column in self.training_columns:
            spec = schema.column_spec(self.table, column)
            if spec.dtype == "int" or self._is_integer_category(column):
                data[column] = data[column].round().astype("int64")
            elif spec.dtype == "float":
                data[column] = data[column].astype(float)
        return data.reset_index(drop=True)

    def _is_integer_category(self, column: str) -> bool:
        spec = schema.column_spec(self.table, column)
        return spec.sdtype in ("categorical", "ordinal") and bool(spec.categories) and all(
            isinstance(v, (int, np.integer)) and not isinstance(v, bool) for v in spec.categories
        )

    def _iter_segments(self, data: pd.DataFrame) -> Iterator[tuple[Any, pd.DataFrame]]:
        """Yield ``(segment key, rows)`` in the schema's category order.

        One segment column gives scalar keys (``"AMU"``), several give tuple keys
        (``("GMED", "elective")``) ordered lexicographically by the schema category order.
        """
        columns = self._segment_columns
        if not columns:
            yield None, data
            return
        ranks = {c: {v: i for i, v in enumerate(schema.column_spec(self.table, c).categories or [])} for c in columns}
        if len(columns) == 1:
            keys: list[Any] = [to_python(v) for v in pd.unique(data[columns[0]])]
        else:
            combos = data[list(columns)].drop_duplicates().itertuples(index=False, name=None)
            keys = [tuple(to_python(v) for v in combo) for combo in combos]

        def order(key: Any) -> tuple:
            values = key if isinstance(key, tuple) else (key,)
            return tuple((ranks[c].get(v, len(ranks[c])), str(v)) for c, v in zip(columns, values))

        for key in sorted(keys, key=order):
            part = data.loc[match_conditions(data, self._key_values(key))].reset_index(drop=True)
            if len(part) < 10:
                LOGGER.warning("Segment %s=%s has only %d training rows", "/".join(columns), _segment_label(key), len(part))
            yield key, part

    def _fit_segment(self, key: Any, part: pd.DataFrame) -> SegmentModel:
        """Fit one SDV synthesizer on a segment's rows (segment and constant columns removed)."""
        started = time.perf_counter()
        model_part = part.drop(columns=list(self._segment_columns)) if self._segment_columns else part
        constants = {
            c: to_python(model_part[c].iloc[0])
            for c in model_part.columns
            if model_part[c].nunique(dropna=False) <= 1
        }
        columns = [c for c in model_part.columns if c not in constants]
        segment = SegmentModel(
            key=key,
            n_rows=len(part),
            columns=columns,
            constants=constants,
            support=self._support_table(model_part),
        )
        if constants:
            LOGGER.info("Segment %s: constant columns re-inserted at sampling: %s", segment.label, sorted(constants))
        if not columns:
            LOGGER.warning("Segment %s: every column is constant; no SDV model fitted", segment.label)
            return segment

        context = {**constants, **self._key_values(key)}  # values of the columns this model does not learn
        if self._constraint_groups:
            segment.constraints = describe_constraints(
                self.table, columns, groups=self._constraint_groups, constants=context
            )
            segment.fixed_combinations = fixed_combination_groups(self.table, columns, self._constraint_groups)
        orders = self._transformer_orders(part, columns)
        distribution = self._configured_distribution()
        column_scores: dict[str, dict[str, float]] = {}
        # Columns of an applied SDV Inequality are renamed/replaced before the copula sees them,
        # so they cannot carry a per-column marginal (they use the default).
        replaced = {c for d in segment.constraints if d.get("type") == "inequality" for c in d.get("columns", [])}
        if self.tune_distributions:
            distribution, segment.distribution_scores, segment.column_distributions, column_scores = (
                self._select_distribution(model_part[columns], orders, segment.label, context, exclude=replaced)
            )
        synthesizer, applied, constraints = self._fit_synthesizer(
            segment, columns, distribution, orders, model_part[columns], context
        )
        if self.tune_distributions:
            fallback = self._degenerate_marginals(synthesizer, segment, model_part[columns], distribution, column_scores)
            if fallback:
                distribution = self._apply_fallbacks(segment, fallback, distribution, replaced)
                synthesizer, applied, constraints = self._fit_synthesizer(
                    segment, columns, distribution, orders, model_part[columns], context
                )
        segment.synthesizer = synthesizer
        segment.distribution = distribution
        segment.ordered_encodings = applied
        segment.constraints = _with_learned_state(segment.constraints, constraints)
        segment.fit_seconds = time.perf_counter() - started
        return segment

    def _fit_synthesizer(
        self,
        segment: SegmentModel,
        columns: list[str],
        distribution: str | None,
        orders: dict[str, list],
        data: pd.DataFrame,
        context: Mapping[str, Any],
    ) -> tuple[Any, dict[str, list], list[Any]]:
        """Create and fit the segment's SDV synthesizer (seeded); returns ``_create_synthesizer``'s triple."""
        synthesizer, applied, constraints = self._create_synthesizer(
            columns, distribution, orders, data, context,
            numerical_distributions=getattr(segment, "column_distributions", None),
        )
        with _quiet_sdv(), _seeded_global_rng(derive_seed(self.seed, "fit", segment.label),
                                              with_torch=self.config.model in _TORCH_MODELS):
            synthesizer.fit({self.table: data})
        return synthesizer, applied, constraints

    def _degenerate_marginals(
        self,
        synthesizer: Any,
        segment: SegmentModel,
        data: pd.DataFrame,
        distribution: str | None,
        column_scores: Mapping[str, Mapping[str, float]],
    ) -> dict[str, float]:
        """Numerical columns whose fitted marginal collapsed on the full segment: ``{column: KS}``.

        Marginals are selected on a subsample; an MLE fit on every row of the segment can still
        degenerate (a gamma fit of length of stay that generated 80 % zero-night stays). The fitted
        synthesizer is sampled once and a column counts as degenerate when its KS against the
        training rows exceeds ``max(POSTFIT_KS_FLOOR, 2 x its tuned KS + 0.05)``.
        """
        numeric = [c for c in data.columns if schema.column_spec(self.table, c).sdtype == "numerical"]
        if not numeric:
            return {}
        with _quiet_sdv():
            synthesizer.reset_sampling()
            synthesizer._set_random_state(derive_seed(self.seed, "postfit-check", segment.label))
            sample = _sdv_sample(synthesizer, self.table, min(len(data), POSTFIT_CHECK_ROWS))
        overrides = getattr(segment, "column_distributions", None) or {}
        degenerate = {}
        for column in numeric:
            if column not in sample.columns:
                continue
            ks = ks_statistic(data[column], sample[column])
            chosen = overrides.get(column, distribution)
            tuned = (column_scores.get(chosen) or {}).get(column)
            limit = max(POSTFIT_KS_FLOOR, 2.0 * tuned + 0.05) if tuned is not None else POSTFIT_KS_FLOOR
            if ks > limit:
                degenerate[column] = round(float(ks), 4)
        return degenerate

    def _apply_fallbacks(
        self, segment: SegmentModel, degenerate: Mapping[str, float], distribution: str | None, replaced: set[str]
    ) -> str | None:
        """Switch degenerate marginals to a robust family; returns the (possibly new) default.

        Directly modelled columns get a per-column override; a degenerate column the copula only
        sees through an SDV ``Inequality`` can only follow the default, so the default changes.
        """
        overrides = dict(getattr(segment, "column_distributions", None) or {})
        new_default = distribution
        for column, ks in degenerate.items():
            current = overrides.get(column, distribution)
            robust = FALLBACK_DISTRIBUTION if current != FALLBACK_DISTRIBUTION else "norm"
            if column in replaced:
                new_default = robust
            else:
                overrides[column] = robust
            LOGGER.warning("Segment %s: marginal '%s' of %s degenerated on the full segment (KS %.3f); refitting "
                           "with '%s'", segment.label, current, column, ks, robust)
        segment.column_distributions = {c: d for c, d in overrides.items() if d != new_default}
        segment.distribution_fallbacks = {c: {"ks": ks} for c, ks in degenerate.items()}
        return new_default

    def _support_table(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Joint counts of the categorical train columns present in ``frame``."""
        columns = [c for c in self._categorical if c in frame.columns]
        if not columns:
            return pd.DataFrame({"count": [len(frame)]})
        counts = frame.groupby(columns, dropna=False, sort=False).size().rename("count")
        return counts.reset_index()

    def _transformer_orders(self, part: pd.DataFrame, columns: list[str]) -> dict[str, list]:
        """Category orders to encode with ``OrderedUniformEncoder`` (GaussianCopula only).

        * ``ordered_encodings``: demand-ordered calendar/ward columns (:data:`ORDERING_RULES`).
        * ``ordinal_metadata``: ordinal columns (occupancy/staffing level) restricted to the
          categories observed in the segment. With SDV's default (all schema categories) an
          unseen level such as ``critical`` on a quiet ward would receive 10% of the latent
          mass, generating labels the ward never had.
        """
        if self.config.model not in _COPULA_MODELS:
            return {}
        orders: dict[str, list] = {}
        if self.config.ordered_encodings:
            orders.update(self._encoding_orders(part, columns))
        if self.config.ordinal_metadata:
            for column, categories in schema.ordered_categories(self.table).items():
                if column in columns and column not in orders:
                    observed = {to_python(v) for v in pd.unique(part[column])}
                    if observed != set(categories):
                        orders[column] = [c for c in categories if c in observed]
        return orders

    def _configured_distribution(self) -> str | None:
        if self.config.model not in _COPULA_MODELS and self.config.model != "copulagan":
            return None
        return str(self._sdv_params.get("default_distribution") or "beta")

    def _create_synthesizer(
        self,
        columns: list[str],
        distribution: str | None,
        orders: dict[str, list],
        data: pd.DataFrame,
        constants: Mapping[str, Any] | None = None,
        numerical_distributions: Mapping[str, str] | None = None,
    ) -> tuple[Any, dict[str, list], list[Any]]:
        """Instantiate an SDV synthesizer for ``columns`` with constraints and ordered encodings.

        ``constants`` are the values of the train columns this (segment) model does not learn;
        constraints that read them (band labels) substitute them. ``numerical_distributions``
        (GaussianCopula only) overrides the default marginal for individual columns.

        Returns the (unfitted) synthesizer, the ordered encodings actually applied
        (constraint-merged columns such as ``ward_id#admission_source`` are skipped) and the
        constraint objects added to it (fitted in place by SDV).
        """
        excluded = [c for c in self.training_columns if c not in columns]
        metadata = build_metadata(self.table, "training", ordinal=self.config.ordinal_metadata, exclude=excluded)
        kwargs = dict(self._sdv_params)
        if distribution is not None:
            kwargs["default_distribution"] = distribution
        if numerical_distributions and self.config.model in _COPULA_MODELS:
            kwargs["numerical_distributions"] = {c: d for c, d in numerical_distributions.items() if c in columns}
        constraints: list[Any] = []
        with _quiet_sdv():
            synthesizer = _synthesizer_class(self.config.model)(metadata, **kwargs)
            if self._constraint_groups:
                constraints = build_constraints(
                    self.table, columns, groups=self._constraint_groups, constants=constants
                )
                if constraints:
                    synthesizer.add_constraints(constraints)
            applied: dict[str, list] = {}
            if orders:
                from rdt.transformers import OrderedUniformEncoder

                synthesizer.auto_assign_transformers({self.table: data})
                current = synthesizer.get_transformers()
                updates = {c: OrderedUniformEncoder(order=list(o)) for c, o in orders.items() if c in current}
                applied = {c: list(orders[c]) for c in updates}
                if self.config.ordered_encodings:
                    for joint, labels, shown in self._joint_column_orders(constraints, data):
                        if joint in current:
                            updates[joint] = OrderedUniformEncoder(order=labels)
                            applied[joint] = shown
                if updates:
                    synthesizer.update_transformers(updates)
        return synthesizer, applied, constraints

    def _joint_column_orders(self, constraints: Sequence[Any], data: pd.DataFrame) -> list[tuple[str, list, list]]:
        """Orders for the merged columns of fitted ``FixedCombinations`` constraints.

        SDV encodes a fixed-combination group as one categorical column whose categories are
        ordered by frequency, so a linear copula cannot relate, say, the delay reason to the
        length of stay. The combinations are ordered by the mean of
        :data:`JOINT_ORDERING_STATISTIC` instead (stays without a delay reason, which are shorter,
        then the delayed ones). Relies on SDV 2.2's ``_joint_column`` / ``_ids_to_combinations``
        (pinned version); anything unexpected leaves SDV's default encoding in place.

        Returns:
            ``[(joint column, ordered category labels, readable combinations in that order)]``.
        """
        statistic = JOINT_ORDERING_STATISTIC.get(self.table)
        if statistic is None or statistic not in data.columns:
            return []
        out = []
        for constraint in constraints:
            joint = getattr(constraint, "_joint_column", None)
            combos = getattr(constraint, "_ids_to_combinations", None)
            columns = list(getattr(constraint, "column_names", None) or [])
            if not joint or not combos or not columns or any(c not in data.columns for c in columns):
                continue
            means = data.groupby(columns, sort=False)[statistic].mean()
            lookup = {tuple(to_python(v) for v in (k if isinstance(k, tuple) else (k,))): float(m) for k, m in means.items()}
            try:
                labels = sorted(combos, key=lambda label: (lookup.get(tuple(to_python(v) for v in combos[label]),
                                                                      math.inf), str(label)))
            except TypeError:  # pragma: no cover - defensive for other SDV versions
                continue
            out.append((joint, labels, ["/".join(str(v) for v in combos[label]) for label in labels]))
        return out

    def _encoding_orders(self, part: pd.DataFrame, columns: list[str]) -> dict[str, list]:
        """Data-driven category orders for the ordered-encoding columns of this segment.

        Only categories present in the segment are ordered: listing unseen categories would
        make ``OrderedUniformEncoder`` reserve 10% of the probability mass for them.
        """
        derived: pd.DataFrame | None = None
        orders: dict[str, list] = {}
        for column, (method, statistic) in ORDERING_RULES[self.table].items():
            if column not in columns or column not in part.columns:
                continue
            natural = list(schema.column_spec(self.table, column).categories or [])
            rank = {v: i for i, v in enumerate(natural)}
            present = [to_python(v) for v in pd.unique(part[column])]
            if method == "natural":
                key_of = {v: 0.0 for v in present}
            elif method == "bed_capacity":
                key_of = {v: float(schema.WARDS[v].bed_capacity) if v in schema.WARDS else 0.0 for v in present}
            else:
                if statistic in part.columns:
                    values = pd.to_numeric(part[statistic], errors="coerce")
                else:
                    if derived is None:
                        derived = schema.derive(self.table, part)
                    values = pd.to_numeric(derived[statistic], errors="coerce")
                means = values.groupby(part[column].to_numpy()).mean()
                key_of = {v: float(means.get(v, np.nan)) for v in present}
                key_of = {v: (0.0 if math.isnan(s) else s) for v, s in key_of.items()}
            orders[column] = sorted(present, key=lambda v: (round(key_of[v], 9), rank.get(v, len(rank))))
        return orders

    def _select_distribution(
        self,
        data: pd.DataFrame,
        orders: dict[str, list],
        label: str,
        constants: Mapping[str, Any] | None = None,
        exclude: Iterable[str] = (),
    ) -> tuple[str, dict[str, float], dict[str, str], dict[str, dict[str, float]]]:
        """Pick GaussianCopula marginals by KS: a default plus per-column overrides.

        Each candidate is fitted as the ``default_distribution`` on (a subsample of) the segment
        and sampled with a fixed seed. The default is the candidate with the lowest mean
        two-sample KS statistic over the numerical columns (held-in training rows; ties keep the
        configured default). A copula reproduces each column's own fitted marginal, so every
        numerical column not in ``exclude`` then gets the candidate with the lowest KS for that
        column (one family for all columns of a segment mis-shapes skewed or mostly-zero columns).

        Returns:
            ``(default distribution, mean KS per candidate, {column: distribution}, {candidate:
            {column: KS}})`` - the third mapping only lists columns whose best candidate differs
            from the default.
        """
        configured = self._configured_distribution() or "beta"
        from sdv.single_table import GaussianCopulaSynthesizer

        valid = set(GaussianCopulaSynthesizer._DISTRIBUTIONS)
        candidates: list[str] = []
        for name in [configured, *self.distribution_candidates]:
            if name not in valid:
                LOGGER.warning("Ignoring unknown distribution candidate '%s'", name)
            elif name not in candidates:
                candidates.append(name)
        numeric = [c for c in data.columns if schema.column_spec(self.table, c).sdtype == "numerical"]
        if len(candidates) < 2 or not numeric:
            return configured, {}, {}, {}
        subsample = data
        if len(data) > TUNE_MAX_ROWS:
            subsample = data.sample(n=TUNE_MAX_ROWS, random_state=derive_seed(self.seed, "tune", label))
        subsample = subsample.reset_index(drop=True)
        scores: dict[str, float] = {}
        column_scores: dict[str, dict[str, float]] = {}
        for candidate in candidates:
            try:
                synthesizer, _, _ = self._create_synthesizer(list(data.columns), candidate, orders, subsample, constants)
                with _quiet_sdv():
                    synthesizer.fit({self.table: subsample})
                    synthesizer.reset_sampling()
                    synthesizer._set_random_state(derive_seed(self.seed, "tune-sample", label))
                    synthetic = _sdv_sample(synthesizer, self.table, len(subsample))
                column_scores[candidate] = {c: ks_statistic(data[c], synthetic[c]) for c in numeric}
                scores[candidate] = round(float(np.mean(list(column_scores[candidate].values()))), 6)
            except Exception as exc:  # a candidate failing must not fail training
                LOGGER.warning("Distribution candidate '%s' failed on segment %s: %s", candidate, label, exc)
        if not scores:
            return configured, {}, {}, {}
        best = min(scores, key=lambda name: (scores[name], candidates.index(name)))
        excluded = set(exclude)
        per_column: dict[str, str] = {}
        for column in numeric:
            if column in excluded:
                continue
            choice = min(column_scores, key=lambda name: (round(column_scores[name][column], 6), name != best,
                                                         candidates.index(name)))
            if choice != best:
                per_column[column] = choice
        LOGGER.info("Segment %s: selected marginal '%s' (mean KS %s), per-column overrides %s",
                    label, best, scores, per_column)
        return best, scores, per_column, column_scores

    # ================================================================== sampling
    def sample(
        self, num_rows: int, conditions: dict[str, Any] | None = None, seed: int | None = None
    ) -> pd.DataFrame:
        """Sample up to ``num_rows`` rows, optionally conditioned on categorical values.

        Args:
            num_rows: Rows wanted (``0`` returns an empty, correctly typed frame).
            conditions: ``{column: value}`` with one value per conditionable column.
            seed: Seed for reproducible output (``None`` draws a fresh one).

        Returns:
            DataFrame with exactly ``schema.training_columns(table)`` and canonical dtypes.
        """
        return self.sample_many([(conditions, num_rows)], seed=seed)[0]

    def sample_many(
        self,
        requests: Sequence[tuple[dict[str, Any] | None, int]],
        seed: int | None = None,
        should_stop: Callable[[], bool] | None = None,
    ) -> list[pd.DataFrame]:
        """Serve several ``(conditions, num_rows)`` requests with as few SDV calls as possible.

        Requests are routed to segments, then served per segment by one pooled unconditional
        draw (rows routed to every request they satisfy) plus native SDV conditional sampling
        for requests too rare to pool. Requests should be disjoint (as produced by the
        generator): a pooled row is assigned to the first matching request only.

        Args:
            requests: ``(conditions, num_rows)`` pairs.
            seed: Seed for reproducible output (``None`` draws a fresh one).
            should_stop: Optional callback checked before every segment and native SDV call;
                once it returns ``True`` the remaining segments are skipped and the rows drawn
                so far are returned (the generator passes its time-budget check). Output is only
                reproducible from the seed when the callback never fires.

        Returns:
            One DataFrame per request (same order), each with up to ``num_rows`` rows.
        """
        self._require_fitted()
        seed = new_seed() if seed is None else int(seed)
        normalised = [(self._normalise_conditions(cond), max(0, int(k))) for cond, k in requests]
        with self._lock:
            routed: dict[Any, list[tuple[int, dict[str, Any], int]]] = {}
            for index, (conditions, k) in enumerate(normalised):
                for key, segment_conditions, n in self._route(conditions, k):
                    routed.setdefault(key, []).append((index, segment_conditions, n))
            parts: list[list[pd.DataFrame]] = [[] for _ in normalised]
            for position, key in enumerate(self._segment_order):
                if key not in routed:
                    continue
                if should_stop is not None and should_stop():
                    LOGGER.debug("sample_many stopped before segment %s (time budget spent)", _segment_label(key))
                    break
                served = self._sample_segment(self._segments[key], routed[key], derive_seed(seed, "segment", position),
                                              should_stop)
                for index, frame in served:
                    if len(frame):
                        parts[index].append(frame)
        results = []
        for index, frames in enumerate(parts):
            if not frames:
                results.append(self.empty_frame())
                continue
            frame = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
            if len(frames) > 1:  # interleave segments
                order = np.random.default_rng(derive_seed(seed, "shuffle", index)).permutation(len(frame))
                frame = frame.iloc[order]
            results.append(frame.reset_index(drop=True))
        return results

    def _route(self, conditions: dict[str, Any], k: int) -> list[tuple[Any, dict[str, Any], int]]:
        """Split a request across segments: ``[(segment key, segment conditions, rows), ...]``.

        Conditions on segment columns select the matching segments (one segment when every
        segment column is conditioned); the request's rows are apportioned to them by the
        training rows matching the remaining conditions (largest remainder). Segment columns
        are removed from the conditions passed to a segment (its model does not learn them).
        """
        if k <= 0:
            return []
        if not self._segment_columns:
            return [(None, conditions, k)]
        fixed, rest = self._split_conditions(conditions)
        segments = [self._segments[key] for key in self._segment_order if self._key_matches(key, fixed)]
        if not segments:
            return []
        if len(fixed) == len(self._segment_columns):
            return [(segments[0].key, rest, k)]
        feasible = [s.is_feasible(rest) for s in segments]
        weights = [s.count(rest) if ok else 0 for s, ok in zip(segments, feasible)]
        if sum(weights) == 0:
            # No training row matches: spread over feasible segments by size (GaussianCopula
            # can still generate unseen combinations of seen values).
            weights = [s.n_rows if ok else 0 for s, ok in zip(segments, feasible)]
        allocation = largest_remainder(k, weights)
        return [(s.key, rest, n) for s, n in zip(segments, allocation) if n > 0]

    def _sample_segment(
        self,
        segment: SegmentModel,
        requests: list[tuple[int, dict[str, Any], int]],
        seed: int,
        should_stop: Callable[[], bool] | None = None,
    ) -> list[tuple[int, pd.DataFrame]]:
        """Serve the requests routed to one segment (pooled draw + native conditional calls).

        ``should_stop`` (see :meth:`sample_many`) is checked before every native SDV call.
        """
        served: list[tuple[int, pd.DataFrame]] = []
        candidates: list[tuple[int, dict[str, Any], int, float]] = []
        for index, conditions, k in requests:
            resolved = segment.resolve_constants(conditions)
            if resolved is None or not segment.is_feasible(conditions):
                LOGGER.debug("Segment %s cannot produce %s", segment.label, conditions)
                continue
            if segment.synthesizer is None:
                if not resolved:
                    served.append((index, self._constant_frame(segment, k)))
                continue
            probability = segment.count(resolved) / segment.n_rows if resolved else 1.0
            if probability <= 0 and resolved:
                # never observed together, but every value was: the copula still produces the
                # combination, so a pooled draw sized by the independence estimate can serve it
                probability = segment.independence_probability(resolved)
            candidates.append((index, resolved, k, probability))
        # GaussianCopula can condition natively: use it for the rarest requests that a pooled
        # draw cannot serve (unseen values first), pool everything else.
        too_rare = [
            c for c in candidates if c[3] <= 0 or c[2] / c[3] > POOL_MAX_ROWS
        ] if self.config.model in _COPULA_MODELS else []
        too_rare.sort(key=lambda c: (c[3], c[0]))
        native = too_rare[:NATIVE_CALLS_PER_SEGMENT]
        native_ids = {c[0] for c in native}
        pooled = [c for c in candidates if c[0] not in native_ids]
        if pooled:
            served.extend(self._serve_pooled(segment, pooled, derive_seed(seed, "pool")))
        for position, (index, conditions, k, probability) in enumerate(native):
            if should_stop is not None and should_stop():
                break
            frame = self._sample_native(segment, conditions, k, probability, derive_seed(seed, "native", position))
            served.append((index, frame))
        return served

    def _serve_pooled(
        self, segment: SegmentModel, requests: list[tuple[int, dict[str, Any], int, float]], seed: int
    ) -> list[tuple[int, pd.DataFrame]]:
        """One unconditional draw routed to every request whose conditions the rows satisfy."""
        needs = []
        for _, conditions, k, probability in requests:
            if not conditions:
                needs.append(k)
            elif probability > 0:
                needs.append(k / probability * POOL_MARGIN)
            else:  # feasible but never observed: modest best-effort draw
                needs.append(50 * k)
        largest_request = max(k for _, _, k, _ in requests)
        # Never draw fewer rows than the largest request (an unconditional request may exceed
        # POOL_MAX_ROWS); otherwise cap the pool so rare requests cannot explode the draw.
        pool_size = max(largest_request, int(min(POOL_MAX_ROWS, math.ceil(max(needs)))))
        draw = self._draw(segment, pool_size, seed)
        available = np.ones(len(draw), dtype=bool)
        # Position index per condition-column set: routing costs O(pool) instead of
        # O(pool x requests) (calendar mode routes one pool to hundreds of cells).
        indexes: dict[tuple[str, ...], dict[tuple, np.ndarray]] = {}
        no_rows = np.array([], dtype=np.int64)
        served = []
        for index, conditions, k, _ in requests:
            if not conditions:
                rows = np.flatnonzero(available)[:k]
            else:
                columns = tuple(sorted(conditions))
                if columns not in indexes:
                    grouped = draw.groupby(list(columns), sort=False, dropna=False).indices
                    indexes[columns] = {
                        tuple(to_python(v) for v in (key if isinstance(key, tuple) else (key,))): positions
                        for key, positions in grouped.items()
                    }
                positions = indexes[columns].get(tuple(conditions[c] for c in columns), no_rows)
                rows = positions[available[positions]][:k]
            available[rows] = False
            served.append((index, draw.iloc[rows].reset_index(drop=True)))
        return served

    def _sample_native(
        self, segment: SegmentModel, conditions: dict[str, Any], k: int, probability: float, seed: int
    ) -> pd.DataFrame:
        """SDV ``sample_from_conditions``; on failure, an oversampled unconditional draw + filter."""
        from sdv.sampling import Condition

        synthesizer = segment.synthesizer
        try:
            with _quiet_sdv():
                synthesizer.reset_sampling()
                synthesizer._set_random_state(int(seed))
                raw = synthesizer.sample_from_conditions(
                    [Condition(column_values=dict(conditions), num_rows=int(k))],
                    max_tries_per_batch=SDV_MAX_TRIES_PER_BATCH,
                )[self.table]
            frame = self._finalize(segment, raw)
            return frame.loc[match_conditions(frame, conditions)].head(k).reset_index(drop=True)
        except Exception as exc:  # SDV raises ValueError / ConstraintNotMetError / SamplingError
            LOGGER.debug("Native conditional sampling failed for %s on segment %s (%s); "
                         "falling back to oversampled unconditional sampling", conditions, segment.label, exc)
        size = int(min(POOL_MAX_ROWS, math.ceil(k / max(probability, FALLBACK_MIN_PROBABILITY) * 1.5)))
        draw = self._draw(segment, max(size, k), derive_seed(seed, "fallback"))
        return draw.loc[match_conditions(draw, conditions)].head(k).reset_index(drop=True)

    def _draw(self, segment: SegmentModel, num_rows: int, seed: int) -> pd.DataFrame:
        """Seeded unconditional draw from a segment, finalised to the output contract."""
        synthesizer = segment.synthesizer
        with _quiet_sdv():
            synthesizer.reset_sampling()
            synthesizer._set_random_state(int(seed))
            raw = _sdv_sample(synthesizer, self.table, int(num_rows))
        return self._finalize(segment, raw)

    def _constant_frame(self, segment: SegmentModel, k: int) -> pd.DataFrame:
        return self._finalize(segment, pd.DataFrame(index=range(int(k))))

    def _finalize(self, segment: SegmentModel, raw: pd.DataFrame) -> pd.DataFrame:
        """Re-insert segment/constant columns, order columns and enforce canonical dtypes."""
        frame = raw.reset_index(drop=True).copy()
        for column, value in segment.constants.items():
            frame[column] = value
        for column, value in self._key_values(segment.key).items():
            frame[column] = value
        frame = frame.reindex(columns=self.training_columns)
        incomplete = frame.isna().any(axis=1)
        if incomplete.any():
            LOGGER.debug("Dropping %d sampled rows with missing values", int(incomplete.sum()))
            frame = frame.loc[~incomplete].reset_index(drop=True)
        return self._cast(frame)

    def _cast(self, frame: pd.DataFrame) -> pd.DataFrame:
        for column in self.training_columns:
            spec = schema.column_spec(self.table, column)
            if spec.sdtype == "numerical":
                values = pd.to_numeric(frame[column], errors="coerce")
                if spec.dtype == "int":
                    frame[column] = np.round(values.to_numpy(dtype=float)).astype("int64")
                else:
                    values = values.astype(float)
                    frame[column] = values.round(spec.decimals) if spec.decimals is not None else values
            elif self._dtypes.get(column, "").startswith("int") or self._is_integer_category(column):
                frame[column] = pd.to_numeric(frame[column], errors="coerce").round().astype("int64")
            else:
                frame[column] = frame[column].astype(object)
        return frame

    def empty_frame(self) -> pd.DataFrame:
        """Zero-row frame with the training columns and canonical dtypes."""
        return empty_training_frame(self.table)

    def _normalise_conditions(self, conditions: dict[str, Any] | None) -> dict[str, Any]:
        """Validate condition columns and coerce values to the training dtypes."""
        if not conditions:
            return {}
        normalised: dict[str, Any] = {}
        for column, value in conditions.items():
            if column not in self._categorical:
                raise ValueError(
                    f"Column '{column}' cannot be used as a condition for '{self.table}'. "
                    f"Conditionable columns: {self._categorical}"
                )
            if isinstance(value, (list, tuple, set, np.ndarray, pd.Series)):
                values = list(value)
                if len(values) != 1:
                    raise ValueError(
                        f"Condition on '{column}' must be a single value (got {values}); "
                        "expand value lists into separate requests."
                    )
                value = values[0]
            value = to_python(value)
            if self._is_integer_category(column):
                try:
                    value = int(value)
                except (TypeError, ValueError):
                    pass  # unknown value -> infeasible, returns zero rows
            normalised[column] = value
        return normalised

    def _require_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError(f"{self!r} is not fitted; call fit() first.")

    # ================================================================== support queries
    def _segment_views(self, conditions: dict[str, Any]) -> Iterator[tuple[SegmentModel, dict[str, Any]]]:
        """Segments matching the conditions on segment columns, with those conditions removed."""
        fixed, rest = self._split_conditions(conditions)
        for key in self._segment_order:
            if self._key_matches(key, fixed):
                yield self._segments[key], rest

    def is_feasible(self, conditions: dict[str, Any] | None) -> bool:
        """Whether at least one segment can produce rows matching ``conditions``."""
        self._require_fitted()
        normalised = self._normalise_conditions(conditions)
        return any(segment.is_feasible(cond) for segment, cond in self._segment_views(normalised))

    def condition_probability(self, conditions: dict[str, Any] | None) -> float:
        """Share of training rows matching ``conditions`` (0 when never observed)."""
        self._require_fitted()
        normalised = self._normalise_conditions(conditions)
        if not normalised:
            return 1.0
        matched = sum(segment.count(cond) for segment, cond in self._segment_views(normalised))
        return matched / self._training_rows if self._training_rows else 0.0

    def training_support(self, columns: Sequence[str]) -> pd.DataFrame:
        """Joint training counts of categorical ``columns`` (columns + ``count``).

        Lets the generator allocate rows by observed frequencies when the processed reference
        split is not available (e.g. a model shipped without its training data).
        """
        self._require_fitted()
        columns = list(columns)
        unknown = [c for c in columns if c not in self._categorical]
        if unknown:
            raise ValueError(f"training_support: {unknown} are not categorical train columns")
        frames = []
        for key in self._segment_order:
            support = self._segments[key].support.copy()
            for column, value in self._key_values(key).items():
                support[column] = value
            for column, value in self._segments[key].constants.items():
                if column not in support.columns:
                    support[column] = value
            frames.append(support)
        merged = pd.concat(frames, ignore_index=True)
        if not columns:
            return pd.DataFrame({"count": [int(merged["count"].sum())]})
        return merged.groupby(columns, dropna=False, sort=False)["count"].sum().reset_index()

    # ================================================================== model card & persistence
    def annotate(self, **info: Any) -> None:
        """Attach extra training information (shown under ``training`` in the model card).

        A ``source`` that is a file-system path (a ``Path`` or an absolute path string) is stored
        project-relative (:func:`bedsynth.config.project_relative`), so neither the pickle nor the
        card records the home directory; a descriptive source such as ``"provided DataFrame"`` is
        kept as given.
        """
        if _is_path_value(info.get("source")):
            info["source"] = project_relative(info["source"])
        self._annotations.update(json_safe(info))

    def model_card(self) -> dict[str, Any]:
        """JSON-serialisable description: table, experiment config, sdv version, rows, fit seconds,
        segments, constraints, training columns, fitted_at (ISO)."""
        segments = [
            self._segments[k].to_card(self._training_rows, self._key_values(k)) for k in self._segment_order
        ]
        first = self._segments[self._segment_order[0]] if self._segment_order else None
        synthesizer_class = (
            type(first.synthesizer).__name__ if first is not None and first.synthesizer is not None
            else _synthesizer_class_name(self.config.model)
        )
        card = {
            "table": self.table,
            "experiment_id": self.experiment_id,
            "model": self.config.model,
            "synthesizer": synthesizer_class,
            "description": self.config.description,
            "experiment": self.config.to_dict(),
            "fitted": self.is_fitted,
            "fitted_at": self._fitted_at,
            "fit_seconds": round(self._fit_seconds, 3) if self._fit_seconds is not None else None,
            "fit_seed": self.seed,
            "training_rows": self._training_rows,
            "training_columns": list(self.training_columns),
            "segment_by": self.segment_by,
            "segment_columns": list(self._segment_columns),
            "segment_label": " x ".join(self._segment_columns) or "global",
            "n_segments": len(segments),
            "segments": segments,
            "constraint_groups": list(self._constraint_groups),
            "effective_constraint_groups": self._effective_groups(segments),
            "constraints": self._applied_constraints(segments),
            "constraint_coverage": self._constraint_coverage(),
            "constraint_enforcement": ENFORCEMENT_NOTE,
            "distributions": {s["segment"]: s["distribution"] for s in segments},
            "distribution_selection": {
                "enabled": self.tune_distributions,
                "candidates": list(self.distribution_candidates) if self.tune_distributions else [],
                "metric": "two-sample KS vs held-in training rows: mean over numerical columns for the default, "
                          "per column for the overrides",
                "scores": {s["segment"]: s["distribution_scores"] for s in segments if s["distribution_scores"]},
                "column_overrides": {s["segment"]: s["column_distributions"] for s in segments
                                     if s.get("column_distributions")},
                "fallbacks": {s["segment"]: s["distribution_fallbacks"] for s in segments
                              if s.get("distribution_fallbacks")},
            },
            "ordered_encodings": {s["segment"]: s["ordered_encodings"] for s in segments if s["ordered_encodings"]},
            "conditional_sampling": {
                "strategy": "pooled rejection (exact) + native SDV conditioning for rare conditions",
                "pool_max_rows": POOL_MAX_ROWS,
                "native_max_tries_per_batch": SDV_MAX_TRIES_PER_BATCH,
            },
            "training": dict(self._annotations),
            **self._versions,
        }
        return json_safe(card)

    @staticmethod
    def _applied_constraints(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Constraints actually applied, merged over segments (learned state left in the segments).

        Each entry is a constraint description plus ``segments`` (how many segments apply it);
        constraints skipped everywhere (e.g. their anchor is a segment column) are not listed.
        """
        merged: dict[tuple, dict[str, Any]] = {}
        for segment in segments:
            for item in segment.get("constraints") or []:
                key = (item.get("group"), tuple(item.get("columns") or []), item.get("expression"))
                entry = merged.get(key)
                if entry is None:
                    entry = {k: v for k, v in item.items() if k != "learned"}
                    entry["segments"] = 0
                    merged[key] = entry
                entry["segments"] += 1
        return list(merged.values())

    def _effective_groups(self, segments: list[dict[str, Any]]) -> list[str]:
        """Configured constraint groups applied in at least one segment (canonical order)."""
        applied = {item.get("group") for s in segments for item in s.get("constraints") or []}
        return [g for g in self._constraint_groups if g in applied]

    def _constraint_coverage(self) -> list[dict[str, Any]]:
        """How each enabled constraint spec is enforced across segments.

        ``constraint_segments``: segments that apply it inside SDV; ``constant_segments``: segments
        where every column of the spec is a segment column or constant, so the segment constants
        reproduce it exactly (e.g. PS-H06 in emergency/transfer segments of a ward x source model);
        the remaining segments do not need it (nothing to enforce there) or leave it to rejection.
        """
        if not self._constraint_groups or not self._segment_order:
            return []
        from bedsynth.synthesis.constraints import CONSTRAINT_SPECS, applicable_specs

        per_segment = []
        for key in self._segment_order:
            segment = self._segments[key]
            constants = {**segment.constants, **self._key_values(key)}
            applied = applicable_specs(self.table, segment.columns, groups=self._constraint_groups,
                                       constants=constants) if segment.columns else []
            per_segment.append((constants, {id(spec) for spec, _ in applied}))
        coverage = []
        for spec in CONSTRAINT_SPECS[self.table]:
            if spec.kind not in self._constraint_groups:
                continue
            columns = list(dict.fromkeys([*([spec.anchor] if spec.anchor else []), *spec.columns]))
            in_sdv = sum(id(spec) in applied for _, applied in per_segment)
            constant = sum(id(spec) not in applied and all(c in constants for c in columns)
                           for constants, applied in per_segment)
            coverage.append({
                "group": spec.kind,
                "columns": columns,
                "enforces_rules": list(spec.rule_ids),
                "constraint_segments": in_sdv,
                "constant_segments": constant,
                "segments": len(per_segment),
            })
        return coverage

    def save(self, directory: Path | str) -> Path:
        """Write ``synthesizer.pkl`` (cloudpickle) and ``model_card.json`` into ``directory``."""
        import cloudpickle

        self._require_fitted()
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        pickle_path = directory / "synthesizer.pkl"
        tmp = pickle_path.with_name(pickle_path.name + ".tmp")
        with self._lock, open(tmp, "wb") as handle:
            cloudpickle.dump(self, handle)
        os.replace(tmp, pickle_path)
        card_path = directory / "model_card.json"
        tmp = card_path.with_name(card_path.name + ".tmp")
        tmp.write_text(json.dumps(self.model_card(), indent=2), encoding="utf-8")
        os.replace(tmp, card_path)
        LOGGER.info("Saved %r to %s", self, directory)
        return directory

    @classmethod
    def load(cls, directory: Path | str) -> "BedSynthSynthesizer":
        """Load a synthesizer saved with :meth:`save` (directory or ``synthesizer.pkl`` path)."""
        import cloudpickle

        path = Path(directory)
        if path.is_dir():
            path = path / "synthesizer.pkl"
        if not path.exists():
            raise FileNotFoundError(f"No saved synthesizer at {path}")
        with open(path, "rb") as handle:
            model = cloudpickle.load(handle)
        if not isinstance(model, cls):
            raise TypeError(f"{path} does not contain a {cls.__name__} (found {type(model).__name__})")
        saved, current = model._versions.get("sdv_version"), _library_versions().get("sdv_version")
        if saved and current and saved != current:
            LOGGER.warning("Model %s was fitted with sdv %s but sdv %s is installed", path, saved, current)
        return model


def _is_path_value(value: Any) -> bool:
    """Whether an annotation value is a file-system path (``Path``, absolute or ``~`` string)."""
    if isinstance(value, os.PathLike):
        return True
    if not isinstance(value, str) or not value:
        return False
    return os.path.isabs(value) or value.startswith(("~", "/", "\\")) or bool(PureWindowsPath(value).drive)


def _segment_label(key: Any) -> str:
    """Printable segment key (mirrors :attr:`SegmentModel.label`)."""
    if key is None:
        return GLOBAL_SEGMENT
    if isinstance(key, tuple):
        return "/".join(str(v) for v in key)
    return str(key)


def _with_learned_state(descriptions: list[dict[str, Any]], constraints: list[Any]) -> list[dict[str, Any]]:
    """Attach the fitted state of programmable constraints to their descriptions.

    ``describe_constraints`` and ``build_constraints`` resolve the same applicable specs in the
    same order, so descriptions and constraint objects pair up by position.
    """
    if len(descriptions) != len(constraints):  # pragma: no cover - defensive
        LOGGER.warning("Constraint descriptions (%d) and objects (%d) do not pair up", len(descriptions), len(constraints))
        return descriptions
    out = []
    for description, constraint in zip(descriptions, constraints):
        state = learned_state(constraint)
        out.append(json_safe({**description, "learned": state}) if state else description)
    return out


def _synthesizer_class_name(model: str) -> str:
    return {
        "gaussian_copula": "GaussianCopulaSynthesizer",
        "ctgan": "CTGANSynthesizer",
        "tvae": "TVAESynthesizer",
        "copulagan": "CopulaGANSynthesizer",
    }.get(model, model)


def _library_versions() -> dict[str, str]:
    """Versions recorded in model cards (sdv, rdt, bedsynth)."""
    import bedsynth

    versions = {"bedsynth_version": bedsynth.__version__}
    for name in ("sdv", "rdt"):
        try:
            module = __import__(name)
            versions[f"{name}_version"] = str(getattr(module, "__version__", "unknown"))
        except ImportError:  # pragma: no cover
            versions[f"{name}_version"] = "unavailable"
    return versions


__all__ = [
    "BedSynthSynthesizer",
    "DEFAULT_DISTRIBUTION_CANDIDATES",
    "ExperimentConfig",
    "MODEL_TYPES",
    "SegmentModel",
    "derive_seed",
    "empty_training_frame",
    "json_safe",
    "ks_statistic",
    "largest_remainder",
    "match_conditions",
    "new_seed",
    "segment_columns_of",
    "to_python",
]
