"""Scenario-aware synthetic data generation engine - DESIGN.md §8.

The API, scripts and evaluation all generate data exclusively through
``SyntheticDataGenerator.generate``. One call:

1. **Validates and merges the request** (§8.1) - table, scenario (must belong to the table),
   row limits, mode, dates, wards, condition columns/values and filters. Scenario conditions,
   request conditions and ``wards`` are intersected; scenario and request filters are merged
   to the tightest bounds. Invalid input raises :class:`GenerationError`; a missing trained
   model raises :class:`ModelNotFoundError`.
2. **Plans condition cells** - value lists are expanded into combinations ("cells") and rows
   are allocated with the largest-remainder method in proportion to each combination's joint
   frequency in the reference train split (§8.2). Calendar mode (§8.4) builds a
   ``wards x dates`` grid, groups it by ``(ward_id, month, day_of_week)`` and splits each
   group's days across the remaining combinations (hierarchical frequency back-off when a
   group has no reference support).
3. **Runs the rejection loop** (§8.3) - every round asks the model for each unfinished cell's
   batch in a single :meth:`BedSynthSynthesizer.sample_many` call, derives the table
   (:func:`bedsynth.schema.derive`), rejects rows violating any *hard* rule of
   :mod:`bedsynth.validation` or the cell's conditions / the merged filters, keeps up to the
   remaining quota and adapts the next batch to the observed acceptance
   (``ceil(remaining / max(acceptance, 0.01) * 1.2)``, clamped to ``max_batch_rows``).
   Two refinements keep this fast without changing its semantics: the first batch uses
   ``max(oversample_factor, 1.2 / raw_hard_valid_rate)`` when the model card records the
   model's raw validity (measured after training), and a cell's acceptance is shrunk towards
   the request-wide acceptance (30 pseudo-rows) so tiny calendar cells do not get 120x
   batches after one unlucky round. Both depend only on the request, the model and the seed,
   so results stay reproducible. Cells that cannot be produced at all are stopped early
   (futility rule).
   **Allocation strata.** Rejection does not accept all rows equally often (with SDV's
   built-in constraints only - the ``gc_segmented`` recipe - a ward-day labelled ``critical``
   passes the hard rules ~12 % of the time, a ``low`` one ~52 %; the programmable constraints
   of ``gc_tuned`` narrow but do not remove such gaps, and CTGAN/TVAE show larger ones), so
   filling one quota per condition cell would skew the output towards "easy" rows
   (e.g. 5 % instead of 19 % critical days). Each cell is therefore split into implicit
   strata (``generation.allocation_strata``: ``ward_id x occupancy_level`` for ward_daily,
   ``ward_id x admission_source x delay_reason`` for patient_stays) whose quotas follow the
   reference frequencies under the request's conditions (largest remainder; calendar mode draws
   them multinomially with back-off to coarser calendar groups). Requests with filters never
   read the real rows that match the filters (a tiny filtered subset would copy the make-up of
   individual real records into the output): their strata weights come from a seeded synthetic
   pilot draw of the model, filtered like the request (``source = "model_pilot_sample"``). The
   report's ``condition_allocation`` stays at the level of the requested combinations. A
   stratum that is abandoned, or still short after ``STRATA_PATIENCE_ROUNDS`` rounds, hands
   its quota to siblings that produce rows more easily (spare accepted rows first), so weak
   models still deliver the requested rows with a bounded bias: ``rebalanced_rows`` counts the
   delivered rows that sit in a stratum beyond its original quota (the TVD of the output's strata
   mix from the allocation is ``rebalanced_rows / rows``), ``quota_moves`` every quota transfer,
   and a warning flags requests where more than 10 % of the rows were rebalanced.
   **Time budget.** Every request carries a time budget (``GenerationRequest.time_budget_seconds``,
   default ``settings.generation.time_budget_seconds``) measured from the start of the request
   and checked before every rejection round, in sample and calendar mode alike. The work of one
   round is bounded too, so a single round cannot overrun the budget by much: the rows requested
   from the model in one round are capped at ``generation.max_round_rows`` (cell batches are
   scaled down proportionally) and at the remaining budget times the sampling throughput
   measured in the earlier rounds of the request (a conservative
   :data:`DEFAULT_ROWS_PER_SECOND` before the first round is timed); the model also stops between
   segments once the budget is spent. A request that runs out of time keeps the rows accepted so
   far (siblings' spare rows still top up short strata), reports the shortfall with a warning and
   never raises. The throughput cap only binds when less than ``max_round_rows / throughput``
   seconds remain; results of a request it shapes, or of a budget-stopped request, depend on the
   machine's speed, so only requests that finish comfortably within their budget are reproducible
   from the seed (the report counts ``throttled_rounds``).
4. **Assembles the output** - seeded shuffle (sample mode) or ``ward_id, date`` order
   (calendar mode), sequential ids, canonical columns and dtypes, and a JSON-serialisable
   report that makes every run reproducible from its ``seed``.

Rows are conditionally independent draws: day-to-day census continuity is not modelled in
calendar mode (documented limitation, DESIGN.md §13).
"""
from __future__ import annotations

import itertools
import logging
import math
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from bedsynth import scenarios, schema, validation
from bedsynth.config import get_settings
from bedsynth.data import io
from bedsynth.synthesis.models import (
    BedSynthSynthesizer,
    derive_seed,
    empty_training_frame,
    json_safe,
    largest_remainder,
    match_conditions,
    new_seed,
    to_python,
)
from bedsynth.synthesis.registry import ModelNotFoundError, ModelRegistry

LOGGER = logging.getLogger(__name__)

MODES: tuple[str, ...] = ("sample", "calendar")
CALENDAR_KEYS: tuple[str, ...] = ("ward_id", "month", "day_of_week")
ACCEPTANCE_FLOOR = 0.01          # acceptance used to size batches when nothing was accepted yet
BATCH_GROWTH = 1.2               # safety factor on the adaptive batch size (DESIGN.md §8.3)
FUTILITY_MIN_ROWS = 2_000        # a cell with no accepted row after this many sampled rows ...
FUTILITY_ROWS_PER_TARGET = 50    # ... (or 50x its quota, capped at max_batch_rows) is abandoned
FUTILITY_REQUEST_FACTOR = 5      # ... or after 5x that many requested rows the model never returned
PRIOR_PSEUDO_ROWS = 30.0         # shrinkage of a cell's acceptance towards the request-wide rate
MAX_FIRST_BATCH_FACTOR = 50.0    # cap on the prior-informed first-round oversampling factor
DEFAULT_TIME_BUDGET_SECONDS = 300.0
DEFAULT_MAX_ROUND_ROWS = 200_000  # rows requested from the model in one round (settings generation.max_round_rows)
DEFAULT_ROWS_PER_SECOND = 5_000.0  # conservative throughput assumed until the first round of a request is timed
MIN_ROUND_ROWS = 1_000           # the time-based round cap never shrinks a round below this many rows
#: Implicit allocation strata (settings ``generation.allocation_strata`` overrides; [] disables).
#: Rejection sampling accepts some strata far more often than others (with SDV built-in
#: constraints only, a ward-day labelled 'critical' passes the hard rules ~12 % of the time, a
#: 'low' one ~52 %; AMU stays twice as often as GMED stays), which would skew the generated mix towards the "easy"
#: strata. Each condition cell is therefore split into strata whose quotas follow the
#: reference frequencies (filters applied), so rejection only shapes rows *within* a stratum.
DEFAULT_ALLOCATION_STRATA: dict[str, tuple[str, ...]] = {
    schema.WARD_DAILY: ("ward_id", "occupancy_level"),
    schema.PATIENT_STAYS: ("ward_id", "admission_source", "delay_reason"),
}
MIN_STRATA_SUPPORT = 20          # calendar mode backs off to coarser reference groups below this
SPARE_ROWS_PER_CELL = 2.0        # accepted rows kept beyond a stratum's quota (x quota) for top-ups
STRATA_PATIENCE_ROUNDS = 4       # after this many rounds short strata hand their quota to more productive siblings
REBALANCE_WARNING_SHARE = 0.10   # warn when more than this share of the output was rebalanced between strata
MAX_LISTED_COMBINATIONS = 5      # infeasible condition combinations quoted in the (single) warning
#: Filtered requests take their strata weights from a synthetic *pilot* draw of the model (never
#: from the real rows that match the filters: tiny filtered subsets would leak the make-up of
#: individual real records into the output and the report).
PILOT_ROWS_PER_ROW = 20          # pilot rows drawn per requested row ...
MIN_PILOT_ROWS = 2_000           # ... at least this many
MAX_PILOT_ROWS = 20_000          # ... and at most this many
PILOT_BUDGET_SHARE = 0.25        # the pilot may use this share of the time budget (at the default throughput)
MIN_PILOT_SUPPORT = 30           # pilot rows matching a cell's conditions and the filters needed to stratify it

__all__ = [
    "GenerationError",
    "GenerationRequest",
    "GenerationResult",
    "ModelNotFoundError",
    "SyntheticDataGenerator",
    "expand_conditions",
]


class GenerationError(ValueError):
    """Invalid generation request (unknown scenario, bad conditions/filters, bad dates...)."""


# ``ModelNotFoundError`` ("No trained model is available for the requested table / model id",
# a ``LookupError``) is defined next to the registry that raises it and re-exported here, so
# ``from bedsynth.synthesis.generator import ModelNotFoundError`` keeps working.


@dataclass
class GenerationRequest:
    """One generation request (DESIGN.md §8.1); see the field comments."""

    table: str = "ward_daily"
    num_rows: int = 100                         # ignored in calendar mode
    scenario: str | None = None                 # scenario name (must belong to table)
    model_id: str | None = None                 # experiment id; None = registry default
    wards: list[str] | None = None              # restrict to wards (merged into conditions)
    conditions: dict[str, Any] | None = None    # extra conditions {column: value | [values]}
    filters: dict[str, dict] | None = None      # extra filters {column: {min,max,in,not_in}}
    seed: int | None = None
    mode: str = "sample"                        # "sample" | "calendar" (ward_daily only)
    start_date: str | None = None               # ISO date, calendar mode
    end_date: str | None = None                 # ISO date, calendar mode (inclusive)
    max_attempts: int | None = None             # None = settings.generation.max_attempts
    include_derived: bool = True                # False = key + train columns only
    time_budget_seconds: float | None = None    # None = the generator's default (settings.generation)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GenerationResult:
    """Generated rows plus a JSON-serialisable report that makes the run reproducible."""

    data: pd.DataFrame                          # generated rows, canonical column order, ids assigned
    report: dict[str, Any] = field(default_factory=dict)
    # report keys (JSON-serialisable): table, model_id, scenario, mode, seed,
    #   num_rows_requested, num_rows_generated, shortfall, attempts, sampled_rows,
    #   accepted_rows, acceptance_rate, rejected_by_rule {rule_id: count},
    #   rejected_by_filter (int), condition_allocation [ {conditions, requested, generated} ],
    #   elapsed_seconds, warnings [str]
    #   + raw_hard_valid_rate, conditions, filters, reference_source, model {...},
    #     include_derived, max_attempts, time_budget_seconds, stopped_reason (None unless the
    #     rejection loop stopped early, e.g. on the time budget), throttled_rounds (rounds scaled
    #     down by the per-round row cap), rows_per_second (measured sampling throughput),
    #     calendar {...} (calendar mode)


# =========================================================================================
# Internal planning structures
# =========================================================================================
@dataclass
class _Spec:
    """A validated, merged request."""

    table: str
    mode: str
    scenario: scenarios.Scenario | None
    num_rows: int
    conditions: dict[str, tuple]
    filters: dict[str, dict]
    seed: int
    max_attempts: int
    include_derived: bool
    time_budget_seconds: float = DEFAULT_TIME_BUDGET_SECONDS
    start_date: date | None = None
    end_date: date | None = None


@dataclass
class _Cell:
    """One condition combination with its row quota (and its dates in calendar mode)."""

    conditions: dict[str, Any]
    requested: int
    group: dict[str, Any] = field(default_factory=dict)
    dates: list[str] = field(default_factory=list)
    reference_count: int | None = None
    feasible: bool = True
    exhausted: bool = False
    batch: int = 0
    requested_from_model: int = 0
    sampled: int = 0
    passed: int = 0
    kept: int = 0
    frames: list[pd.DataFrame] = field(default_factory=list)
    parent: int = 0                  # index of the requested condition combination (report level)
    stratum: bool = False            # True for an implicit allocation stratum of its parent
    weight: float = 1.0              # reference weight among sibling strata (quota rebalancing)
    allocated: int = 0               # quota the stratum received at stratification (before rebalancing)
    spare: list[pd.DataFrame] = field(default_factory=list)   # accepted rows beyond the quota
    spare_rows: int = 0

    @property
    def remaining(self) -> int:
        """Rows still missing from the cell's quota."""
        return self.requested - self.kept

    def take_spare(self) -> int:
        """Move spare accepted rows into the quota (as far as it is open); returns rows taken."""
        taken = 0
        while self.remaining > 0 and self.spare:
            frame = self.spare.pop(0)
            take = frame.iloc[: self.remaining]
            self.frames.append(take)
            self.kept += len(take)
            taken += len(take)
            self.spare_rows -= len(frame)
            if len(take) < len(frame):
                rest = frame.iloc[len(take):]
                self.spare.insert(0, rest)
                self.spare_rows += len(rest)
        return taken

    @property
    def acceptance(self) -> float:
        """Accepted rows per row requested from the model (0 before the first round)."""
        return self.passed / self.requested_from_model if self.requested_from_model else 0.0


@dataclass
class _LoopStats:
    attempts: int = 0
    sampled_rows: int = 0
    hard_valid_rows: int = 0
    accepted_rows: int = 0
    rejected_by_filter: int = 0
    rejected_by_rule: dict[str, int] = field(default_factory=dict)
    stopped_reason: str | None = None
    first_batch_factor: float = 0.0
    acceptance_prior: float | None = None
    quota_moves: int = 0              # quota moved between strata, summed over every move (one row's quota can move twice)
    throttled_rounds: int = 0         # rounds whose batches were scaled down by the per-round row cap
    effective_rounds: float = 0.0     # rounds weighted by their batch scale (strata patience rule)
    timed_rows: int = 0               # rows requested from the model in timed rounds ...
    timed_seconds: float = 0.0        # ... and the seconds those rounds took (sampling + validation)

    @property
    def rows_per_second(self) -> float | None:
        """Sampling throughput measured over the rounds run so far (``None`` before the first)."""
        return self.timed_rows / self.timed_seconds if self.timed_seconds > 0 and self.timed_rows else None


@dataclass
class _StrataInfo:
    """How condition cells were split into implicit allocation strata (report block)."""

    columns: tuple[str, ...] = ()
    source: str | None = None
    cells: int = 0
    stratified_cells: int = 0
    pilot_rows: int | None = None     # synthetic pilot rows matching the filters (filtered requests)

    def to_dict(self, stats: _LoopStats, cells: Sequence[_Cell] = ()) -> dict[str, Any]:
        out = {
            "columns": list(self.columns),
            "source": self.source,
            "strata_cells": self.cells,
            "stratified_combinations": self.stratified_cells,
            "rebalanced_rows": rebalanced_rows(cells),
            "quota_moves": stats.quota_moves,
        }
        if self.pilot_rows is not None:
            out["pilot_rows"] = self.pilot_rows
        return out


def rebalanced_rows(cells: Sequence[_Cell]) -> int:
    """Delivered rows that sit in a stratum beyond the quota it was allocated (a true row count).

    Rebalancing moves quota from strata a model rarely produces to siblings it produces more
    easily; the rows a stratum delivers beyond its original (reference-proportional) quota are the
    rows that shifted the output mix. Without a shortfall, the total variation distance between the
    output's strata mix and the allocation is exactly ``rebalanced_rows / rows``.
    """
    total = 0
    for cell in cells:
        if cell.stratum:
            delivered = min(cell.kept, len(cell.dates)) if cell.dates else cell.kept
            total += max(0, delivered - cell.allocated)
    return total


def expand_conditions(conditions: dict[str, Sequence[Any]] | None) -> list[dict[str, Any]]:
    """Cartesian product of condition value lists: ``{a: (1, 2)} -> [{a: 1}, {a: 2}]``."""
    if not conditions:
        return [{}]
    columns = list(conditions)
    return [dict(zip(columns, values)) for values in itertools.product(*(conditions[c] for c in columns))]


# =========================================================================================
# Generator
# =========================================================================================
class SyntheticDataGenerator:
    """Loads trained models through the registry and generates validated scenario data.

    Args:
        models_dir: Registry root (default: configured ``models_dir``).
        processed_dir: Directory of the processed reference splits used for allocation
            frequencies (default: configured ``processed_dir``).
        time_budget_seconds: Default time budget of one request, used when the request does
            not set ``time_budget_seconds`` (default: ``settings.generation.time_budget_seconds``
            if present, else 300 s).

    The generator is thread-safe: caches are lock-protected and each model serialises its own
    sampling (SDV synthesizers keep mutable random state).
    """

    def __init__(
        self,
        models_dir: Path | str | None = None,
        processed_dir: Path | str | None = None,
        *,
        time_budget_seconds: float | None = None,
    ):
        self.registry = ModelRegistry(models_dir)
        self.processed_dir = Path(processed_dir) if processed_dir is not None else None
        self.settings = dict(get_settings().get("generation", {}))
        if time_budget_seconds is None:
            time_budget_seconds = float(self.settings.get("time_budget_seconds") or DEFAULT_TIME_BUDGET_SECONDS)
        self.time_budget_seconds = time_budget_seconds
        self.allocation_strata = self._configured_strata(self.settings.get("allocation_strata"))
        self._models: dict[tuple[str, str], tuple[BedSynthSynthesizer, float]] = {}
        self._reference: dict[str, tuple[float | None, pd.DataFrame | None]] = {}
        self._lock = threading.RLock()

    @staticmethod
    def _configured_strata(configured: Any) -> dict[str, tuple[str, ...]]:
        """Allocation strata per table: settings override the defaults (``[]`` disables)."""
        strata = dict(DEFAULT_ALLOCATION_STRATA)
        for table, columns in (configured or {}).items():
            if table not in schema.TABLES:
                raise ValueError(f"generation.allocation_strata: unknown table '{table}'")
            columns = tuple(columns or ())
            allowed = set(schema.conditionable_columns(table))
            unknown = [c for c in columns if c not in allowed]
            if unknown:
                raise ValueError(
                    f"generation.allocation_strata[{table}]: {unknown} are not categorical train columns "
                    f"(allowed: {sorted(allowed)})"
                )
            strata[table] = columns
        return strata

    def __repr__(self) -> str:
        return f"SyntheticDataGenerator(models_dir={str(self.registry.models_dir)!r})"

    # ------------------------------------------------------------------ models
    def available_models(self, table: str | None = None) -> list[dict[str, Any]]:
        """Registered models (summaries from the registry), optionally for one table."""
        if table is not None:
            self._check_table(table)
        return self.registry.list_models(table)

    def default_model_id(self, table: str) -> str | None:
        """Model used when a request does not name one (``None`` if nothing is trained)."""
        self._check_table(table)
        return self.registry.default_model_id(table)

    def get_model(self, table: str, model_id: str | None = None) -> tuple[str, BedSynthSynthesizer]:
        """Resolve and load (with caching) a trained model: returns ``(model_id, model)``.

        The cache is invalidated when the model file changes on disk (re-training).

        Raises:
            ModelNotFoundError: nothing trained for ``table`` / unknown ``model_id``.
        """
        self._check_table(table)
        with self._lock:
            resolved = model_id or self.registry.default_model_id(table)
            if resolved is None:
                raise ModelNotFoundError(
                    f"No trained model available for table '{table}'. Train one with "
                    f"`python scripts/02_train_synthesizers.py --tables {table}`."
                )
            try:
                path = self.registry.model_dir(table, resolved) / "synthesizer.pkl"
            except ValueError as exc:
                raise ModelNotFoundError(f"Invalid model id '{resolved}'.") from exc
            mtime = path.stat().st_mtime if path.exists() else None
            cached = self._models.get((table, resolved))
            if cached is not None and mtime is not None and cached[1] == mtime:
                return resolved, cached[0]
            model = self.registry.load(table, resolved)  # raises ModelNotFoundError
            self._models[(table, resolved)] = (model, mtime or 0.0)
            return resolved, model

    def clear_cache(self) -> None:
        """Forget loaded models and reference frequencies (e.g. after re-training)."""
        with self._lock:
            self._models.clear()
            self._reference.clear()

    def sample_raw(
        self,
        table: str,
        num_rows: int,
        model_id: str | None = None,
        seed: int | None = None,
        conditions: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        """Derived but *unfiltered* model samples (no rule rejection).

        Used by the evaluation's validity score: how often the model produces operationally
        valid rows on its own.
        """
        _, model = self.get_model(table, model_id)
        raw = model.sample(int(num_rows), conditions=conditions, seed=seed)
        return schema.derive(table, raw)

    # ------------------------------------------------------------------ generate
    def generate(self, request: GenerationRequest) -> GenerationResult:
        """Generate validated synthetic rows for ``request`` (see module docstring)."""
        started = time.perf_counter()
        spec = self._validate(request)
        model_id, model = self.get_model(spec.table, request.model_id)
        warnings_: list[str] = []
        if spec.mode == "calendar":
            cells, reference_source, calendar_info = self._plan_calendar(spec, model, model_id, warnings_)
            strata_rng = np.random.default_rng(derive_seed(spec.seed, "calendar-strata"))
            cells, strata = self._stratify(spec, model, cells, rng=strata_rng, started=started)
        else:
            cells, reference_source = self._plan_sample(spec, model, model_id, warnings_)
            calendar_info = None
            cells, strata = self._stratify(spec, model, cells, started=started)
        stats = self._run_rejection_loop(spec, model, cells, started)
        data = self._assemble(spec, cells)
        report = self._build_report(
            spec, model_id, model, cells, stats, data, reference_source, calendar_info, warnings_, started
        )
        report["allocation_strata"] = json_safe(strata.to_dict(stats, cells))
        moved = int(report["allocation_strata"]["rebalanced_rows"])
        if len(data) and moved > REBALANCE_WARNING_SHARE * len(data):
            report["warnings"].append(
                f"{moved} of {len(data)} rows ({moved / len(data):.0%}) were delivered in allocation strata beyond "
                f"their reference quota because model '{model_id}' rarely produces valid rows in the other strata: "
                "the output mix is skewed towards the strata the model produces easily."
            )
        LOGGER.info(
            "Generated %d/%d %s rows (scenario=%s, model=%s, seed=%d) in %.2fs",
            len(data), report["num_rows_requested"], spec.table,
            spec.scenario.name if spec.scenario else None, model_id, spec.seed, report["elapsed_seconds"],
        )
        return GenerationResult(data=data, report=report)

    # ================================================================== validation (§8.1)
    @staticmethod
    def _check_table(table: str) -> None:
        if table not in schema.TABLES:
            raise GenerationError(f"Unknown table '{table}'. Known tables: {list(schema.TABLES)}")

    def _validate(self, request: GenerationRequest) -> _Spec:
        table = request.table
        self._check_table(table)
        mode = request.mode or "sample"
        if mode not in MODES:
            raise GenerationError(f"Unknown mode '{mode}'. Expected one of {list(MODES)}.")
        if mode == "calendar" and table != schema.WARD_DAILY:
            raise GenerationError("Calendar mode is only available for the 'ward_daily' table.")

        scenario = None
        if request.scenario:
            try:
                scenario = scenarios.get_scenario(request.scenario)
            except KeyError as exc:
                raise GenerationError(str(exc.args[0]) if exc.args else str(exc)) from exc
            if scenario.table != table:
                raise GenerationError(
                    f"Scenario '{scenario.name}' belongs to table '{scenario.table}', not '{table}'. "
                    f"Scenarios for '{table}': {[s.name for s in scenarios.scenarios_for(table)]}"
                )

        max_rows = int(self.settings.get("max_rows", 20_000))
        num_rows = request.num_rows
        if mode == "sample":
            if isinstance(num_rows, bool) or not isinstance(num_rows, (int, np.integer)):
                raise GenerationError(f"num_rows must be an integer, got {num_rows!r}.")
            if not 1 <= int(num_rows) <= max_rows:
                raise GenerationError(f"num_rows must be between 1 and {max_rows}, got {num_rows}.")

        conditions = self._merged_conditions(table, scenario, request)
        filters = self._merged_filters(table, scenario, request.filters)

        seed = request.seed
        if seed is None:
            seed = new_seed()
        elif isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
            raise GenerationError(f"seed must be a non-negative integer, got {seed!r}.")

        max_attempts = request.max_attempts if request.max_attempts is not None else self.settings.get("max_attempts", 25)
        if isinstance(max_attempts, bool) or not isinstance(max_attempts, (int, np.integer)) or int(max_attempts) < 1:
            raise GenerationError(f"max_attempts must be a positive integer, got {max_attempts!r}.")

        spec = _Spec(
            table=table,
            mode=mode,
            scenario=scenario,
            num_rows=int(num_rows) if mode == "sample" else 0,
            conditions=conditions,
            filters=filters,
            seed=int(seed),
            max_attempts=int(max_attempts),
            include_derived=bool(request.include_derived),
            time_budget_seconds=self._time_budget(request.time_budget_seconds),
        )
        if mode == "calendar":
            spec.start_date, spec.end_date = self._parse_dates(request.start_date, request.end_date)
            wards = conditions.get("ward_id", schema.WARD_IDS)
            days = (spec.end_date - spec.start_date).days + 1
            if days * len(wards) > max_rows:
                raise GenerationError(
                    f"Calendar grid of {len(wards)} ward(s) x {days} day(s) = {days * len(wards)} rows "
                    f"exceeds the limit of {max_rows} rows."
                )
        return spec

    def _time_budget(self, requested: Any) -> float:
        """The request's time budget in seconds (``None`` = the generator's default)."""
        if requested is None:
            return float(self.time_budget_seconds)
        if isinstance(requested, bool) or not isinstance(requested, (int, float, np.integer, np.floating)):
            raise GenerationError(f"time_budget_seconds must be a positive number, got {requested!r}.")
        budget = float(requested)
        if not math.isfinite(budget) or budget <= 0:
            raise GenerationError(f"time_budget_seconds must be a positive number, got {requested!r}.")
        return budget

    def _merged_conditions(
        self, table: str, scenario: scenarios.Scenario | None, request: GenerationRequest
    ) -> dict[str, tuple]:
        """scenario.conditions ∩ request.conditions ∩ {ward_id: wards}, values normalised."""
        try:
            extra = scenarios.validate_conditions(table, request.conditions)
        except ValueError as exc:
            raise GenerationError(str(exc)) from exc
        merged = {c: self._normalise_values(table, c, v) for c, v in (scenario.conditions if scenario else {}).items()}
        extra = {c: self._normalise_values(table, c, v) for c, v in extra.items()}
        wards = list(request.wards or [])
        if isinstance(request.wards, str):
            wards = [request.wards]
        try:
            merged = scenarios.merge_conditions(merged, extra)
            if wards:
                merged = scenarios.merge_conditions(merged, {"ward_id": self._normalise_values(table, "ward_id", wards)})
        except ValueError as exc:
            raise GenerationError(f"Conditions do not overlap: {exc}") from exc
        return merged

    @staticmethod
    def _normalise_values(table: str, column: str, values: Sequence[Any]) -> tuple:
        """Coerce condition values to canonical codes and check them against the vocabulary."""
        spec = schema.column_spec(table, column)
        categories = list(spec.categories or [])
        integer = bool(categories) and all(isinstance(c, int) for c in categories)
        out: list[Any] = []
        for raw in values if isinstance(values, (list, tuple, set)) else [values]:
            value = to_python(raw)
            if integer:
                try:
                    if isinstance(value, float) and not value.is_integer():
                        raise ValueError(value)
                    value = int(value)
                except (TypeError, ValueError):
                    raise GenerationError(f"Invalid value {raw!r} for '{column}': expected one of {categories}.") from None
            elif value not in categories and value is not None:
                value = schema.normalize_category(column, value) or value
            if value not in categories:
                raise GenerationError(f"Invalid value {raw!r} for '{column}'. Allowed values: {categories}.")
            if value not in out:
                out.append(value)
        if not out:
            raise GenerationError(f"Condition on '{column}' must list at least one value.")
        return tuple(out)

    @staticmethod
    def _merged_filters(table: str, scenario: scenarios.Scenario | None, extra: dict | None) -> dict[str, dict]:
        try:
            extra = scenarios.validate_filters(table, extra)
        except ValueError as exc:
            raise GenerationError(str(exc)) from exc
        for column, spec in extra.items():
            for bound in ("min", "max"):
                if bound in spec and (isinstance(spec[bound], bool) or not isinstance(spec[bound], (int, float, np.number))):
                    raise GenerationError(f"Filter '{column}.{bound}' must be a number, got {spec[bound]!r}.")
            for key in ("in", "not_in"):
                if key in spec and not isinstance(spec[key], (list, tuple, set)):
                    raise GenerationError(f"Filter '{column}.{key}' must be a list, got {spec[key]!r}.")
        merged = scenarios.merge_filters(dict(scenario.filters) if scenario else {}, extra)
        for column, spec in merged.items():
            if "min" in spec and "max" in spec and spec["min"] > spec["max"]:
                raise GenerationError(f"Filters on '{column}' are contradictory: min {spec['min']} > max {spec['max']}.")
            if "in" in spec and len(spec["in"]) == 0:
                raise GenerationError(f"Filters on '{column}' leave no allowed value.")
        return merged

    @staticmethod
    def _parse_dates(start: str | None, end: str | None) -> tuple[date, date]:
        if not start or not end:
            raise GenerationError("Calendar mode requires start_date and end_date (YYYY-MM-DD).")
        try:
            start_d = datetime.strptime(str(start), schema.DATE_FORMAT).date()
            end_d = datetime.strptime(str(end), schema.DATE_FORMAT).date()
        except ValueError as exc:
            raise GenerationError(f"Invalid calendar dates {start!r} / {end!r}: use YYYY-MM-DD.") from exc
        if end_d < start_d:
            raise GenerationError(f"end_date {end} is before start_date {start}.")
        return start_d, end_d

    # ================================================================== allocation (§8.2)
    def _reference_full(self, table: str) -> pd.DataFrame | None:
        """Fully derived reference train split (cached by file mtime; None if unavailable)."""
        path = io.processed_path(table, "train", self.processed_dir)
        mtime = path.stat().st_mtime if path.exists() else None
        with self._lock:
            cached = self._reference.get(table)
            if cached is not None and cached[0] == mtime:
                return cached[1]
            frame = None
            if mtime is not None:
                try:
                    full = io.load_processed(table, "train", self.processed_dir)
                    missing = [c for c in schema.training_columns(table) if c not in full.columns]
                    if missing:
                        raise ValueError(f"missing train columns {missing}")
                    frame = schema.derive(table, full[schema.training_columns(table)])
                except Exception as exc:  # noqa: BLE001 - fall back to model frequencies
                    LOGGER.warning("Cannot read reference split %s: %s", path, exc)
            self._reference[table] = (mtime, frame)
            return frame

    def _reference_frame(self, table: str) -> pd.DataFrame | None:
        """Categorical train columns of the reference train split (cached by file mtime)."""
        full = self._reference_full(table)
        if full is None:
            return None
        return full[[c for c in schema.categorical_columns(table) if c in full.columns]]

    def _support(self, table: str, model: BedSynthSynthesizer, columns: list[str]) -> tuple[pd.DataFrame | None, str]:
        """Joint counts over ``columns``: reference split first, else the model's training data."""
        reference = self._reference_frame(table)
        if reference is not None and all(c in reference.columns for c in columns):
            if not columns:
                return pd.DataFrame({"count": [len(reference)]}), "reference_train_split"
            counts = reference.groupby(columns, dropna=False, sort=False).size().rename("count").reset_index()
            return counts, "reference_train_split"
        try:
            return model.training_support(columns), "model_training_data"
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("No frequency information for allocation: %s", exc)
            return None, "uniform"

    @staticmethod
    def _combo_counts(support: pd.DataFrame | None, combos: list[dict[str, Any]], base: dict[str, Any] | None = None) -> list[int]:
        if support is None:
            return [0] * len(combos)
        frame = support.loc[match_conditions(support, base)] if base else support
        counts = frame["count"].to_numpy()
        return [int(counts[match_conditions(frame, combo)].sum()) for combo in combos]

    def _plan_sample(
        self, spec: _Spec, model: BedSynthSynthesizer, model_id: str, warnings_: list[str]
    ) -> tuple[list[_Cell], str]:
        combos = expand_conditions(spec.conditions)
        feasible = [model.is_feasible(c) for c in combos]
        infeasible = [combo for combo, ok in zip(combos, feasible) if not ok]
        if infeasible:
            listed = ", ".join(str(combo) for combo in infeasible[:MAX_LISTED_COMBINATIONS])
            more = len(infeasible) - MAX_LISTED_COMBINATIONS
            warnings_.append(
                f"Model '{model_id}' cannot generate {len(infeasible)} of {len(combos)} condition combination(s) "
                f"(the value combination never occurs in its training data), so no rows were allocated to them: "
                f"{listed}" + (f" and {more} more." if more > 0 else ".")
            )
        if len(combos) == 1:
            counts, source = [None], "not_needed"
            weights = [1.0 if feasible[0] else 0.0]
        else:
            support, source = self._support(spec.table, model, list(spec.conditions))
            counts = self._combo_counts(support, combos)
            weights = [c if ok else 0 for c, ok in zip(counts, feasible)]
            if source != "reference_train_split":
                warnings_.append(
                    "Reference train split not available: rows were allocated using "
                    + ("the model's training frequencies." if source == "model_training_data" else "uniform weights.")
                )
            if sum(weights) == 0 and any(feasible):
                weights = [1.0 if ok else 0.0 for ok in feasible]
                warnings_.append("No reference support for the requested conditions: rows were allocated uniformly.")
                source = "uniform"
        allocation = largest_remainder(spec.num_rows, weights)
        cells = [
            _Cell(conditions=combo, requested=n, group=dict(combo), reference_count=count, feasible=ok, parent=i)
            for i, (combo, n, count, ok) in enumerate(zip(combos, allocation, counts, feasible))
        ]
        return cells, source

    # ================================================================== allocation strata
    def _stratify(
        self,
        spec: _Spec,
        model: BedSynthSynthesizer,
        cells: list[_Cell],
        rng: np.random.Generator | None = None,
        started: float | None = None,
    ) -> tuple[list[_Cell], _StrataInfo]:
        """Split every open cell into implicit strata with reference-proportional quotas.

        Unfiltered requests weight the strata by the joint frequencies of the free strata columns
        among the reference rows that match the cell's conditions (categorical frequencies of the
        train split, i.e. the model's training support; the model's own training counts when the
        split is unavailable). Filtered requests never read the real rows that match the filters:
        their weights come from a seeded synthetic pilot draw of the model, derived and filtered
        (:meth:`_pilot_reference`), i.e. from the model's own distribution under the filters; cells
        with fewer than :data:`MIN_PILOT_SUPPORT` matching pilot rows stay unstratified.
        Sample mode allocates quotas with the largest-remainder method; calendar mode (``rng``
        given) draws them multinomially (unbiased for the few days of one calendar group) and
        backs off to coarser groups with fewer than :data:`MIN_STRATA_SUPPORT` rows.
        Cells without support for any feasible stratum stay unstratified.
        """
        columns = tuple(self.allocation_strata.get(spec.table, ()))
        info = _StrataInfo(columns=columns)
        if not columns:
            return cells, info
        min_support = 0
        if spec.filters:
            reference = self._pilot_reference(spec, model, cells, started)
            if reference is None:
                info.source = "unavailable (no time for a pilot draw)"
                return cells, info
            info.source, info.pilot_rows, min_support = "model_pilot_sample", len(reference), MIN_PILOT_SUPPORT
        else:
            reference = self._reference_full(spec.table)
            info.source = "reference_train_split" if reference is not None else "model_training_data"
        out: list[_Cell] = []
        for cell in cells:
            free = [c for c in columns if c not in cell.conditions]
            if not free or cell.requested <= 0 or not cell.feasible:
                out.append(cell)
                continue
            strata = self._stratum_weights(model, reference, cell.conditions, free, calendar=rng is not None,
                                           min_support=min_support)
            if not strata:
                out.append(cell)
                continue
            weights = [w for _, w in strata]
            if rng is None:
                allocation = largest_remainder(cell.requested, weights)
            else:
                probabilities = np.asarray(weights, dtype=float) / float(sum(weights))
                allocation = [int(n) for n in rng.multinomial(cell.requested, probabilities)]
            dates = list(cell.dates)
            if rng is not None:
                rng.shuffle(dates)
            start = 0
            info.stratified_cells += 1
            for (values, weight), n in zip(strata, allocation):
                if n <= 0:
                    continue
                out.append(_Cell(
                    conditions={**cell.conditions, **values},
                    requested=n,
                    group=dict(cell.group),
                    dates=sorted(dates[start:start + n]) if dates else [],
                    reference_count=cell.reference_count,
                    feasible=True,
                    parent=cell.parent,
                    stratum=True,
                    weight=float(weight),
                    allocated=n,
                ))
                start += n
                info.cells += 1
        return out, info

    def _pilot_reference(
        self, spec: _Spec, model: BedSynthSynthesizer, cells: list[_Cell], started: float | None
    ) -> pd.DataFrame | None:
        """Seeded synthetic pilot draw of the model, derived and filtered by the request filters.

        One request per condition combination (calendar cells are pooled per combination without
        their month/weekday), sized in proportion to its quota: ``PILOT_ROWS_PER_ROW`` rows per
        requested row within ``[MIN_PILOT_ROWS, MAX_PILOT_ROWS]`` and at most
        :data:`PILOT_BUDGET_SHARE` of the time budget at :data:`DEFAULT_ROWS_PER_SECOND`. The pilot
        is *not* filtered by the hard rules: it describes the model's distribution before
        rejection, which the strata then protect from rejection bias. Returns ``None`` when the
        budget leaves no room for a pilot.
        """
        quotas: dict[tuple, tuple[dict[str, Any], int]] = {}
        for cell in cells:
            if cell.requested <= 0 or not cell.feasible:
                continue
            conditions = {k: v for k, v in cell.conditions.items()
                          if spec.mode != "calendar" or k not in ("month", "day_of_week")}
            key = tuple(sorted((k, str(v)) for k, v in conditions.items()))
            previous = quotas.get(key, (conditions, 0))[1]
            quotas[key] = (conditions, previous + cell.requested)
        if not quotas:
            return None
        total = sum(n for _, n in quotas.values())
        size = min(MAX_PILOT_ROWS, max(MIN_PILOT_ROWS, PILOT_ROWS_PER_ROW * total))
        size = int(min(size, spec.time_budget_seconds * PILOT_BUDGET_SHARE * DEFAULT_ROWS_PER_SECOND))
        if size < MIN_PILOT_SUPPORT:
            return None
        shares = largest_remainder(size, [n for _, n in quotas.values()])
        requests = [(conditions, n) for (conditions, _), n in zip(quotas.values(), shares) if n > 0]
        deadline = (started if started is not None else time.perf_counter()) + spec.time_budget_seconds
        frames = model.sample_many(requests, seed=derive_seed(spec.seed, "strata-pilot"),
                                   should_stop=lambda: time.perf_counter() > deadline)
        frames = [f for f in frames if len(f)]
        if not frames:
            return None
        pilot = schema.derive(spec.table, pd.concat(frames, ignore_index=True))
        return pilot.loc[scenarios.filter_mask(pilot, spec.filters).to_numpy(dtype=bool)].reset_index(drop=True)

    def _stratum_weights(
        self,
        model: BedSynthSynthesizer,
        reference: pd.DataFrame | None,
        conditions: dict[str, Any],
        free: list[str],
        calendar: bool = False,
        min_support: int = 0,
    ) -> list[tuple[dict[str, Any], float]]:
        """``[(stratum values, weight)]`` of the feasible strata of one cell (empty: no support).

        ``min_support``: rows of ``reference`` matching the cell needed at the last back-off level
        (pilot references; smaller cells stay unstratified).
        """
        levels = [conditions]
        if calendar:  # back off from (ward, month, weekday, combo) to coarser calendar groups
            levels += [
                {k: v for k, v in conditions.items() if k != "day_of_week"},
                {k: v for k, v in conditions.items() if k not in ("day_of_week", "month")},
            ]
        for depth, level in enumerate(levels):
            if reference is not None:
                subset = reference.loc[match_conditions(reference, level)]
                if subset.empty or len(subset) < min_support:
                    continue
                counts = subset.groupby(free, sort=True, dropna=True).size()
            else:
                try:
                    support = model.training_support(sorted(set(level) | set(free)))
                except Exception as exc:  # noqa: BLE001 - no frequency information: stay unstratified
                    LOGGER.debug("No training support for strata: %s", exc)
                    return []
                support = support.loc[match_conditions(support, level)]
                if support.empty:
                    continue
                counts = support.groupby(free, sort=True, dropna=True)["count"].sum()
            if calendar and depth < len(levels) - 1 and int(counts.sum()) < MIN_STRATA_SUPPORT:
                continue
            strata = []
            for key, count in counts.items():
                values = dict(zip(free, key if isinstance(key, tuple) else (key,)))
                values = {c: to_python(v) for c, v in values.items()}
                stratum = {**conditions, **values}
                # Strata found after a calendar back-off may never occur at the cell's exact
                # conditions in training (e.g. a critical day on one ward-month-weekday): they are
                # kept as long as the model can produce their values (dropping them shrank the
                # critical share of calendar years); the model serves them by pooled rejection.
                if count > 0 and model.is_feasible(stratum):
                    strata.append((values, float(count)))
            if strata:
                return strata
        return []

    def _plan_calendar(
        self, spec: _Spec, model: BedSynthSynthesizer, model_id: str, warnings_: list[str]
    ) -> tuple[list[_Cell], str, dict[str, Any]]:
        dates = pd.date_range(spec.start_date, spec.end_date, freq="D")
        wards = list(spec.conditions.get("ward_id", schema.WARD_IDS))
        grid = pd.DataFrame(
            [(w, d) for w in wards for d in dates], columns=["ward_id", "timestamp"]
        )
        grid["date"] = grid["timestamp"].dt.strftime(schema.DATE_FORMAT)
        grid["month"] = grid["timestamp"].dt.month.astype(int)
        grid["day_of_week"] = grid["timestamp"].dt.dayofweek.map(dict(enumerate(schema.DAYS_OF_WEEK)))
        total_grid = len(grid)
        for column in ("month", "day_of_week"):
            if column in spec.conditions:
                grid = grid.loc[grid[column].isin(list(spec.conditions[column]))]
        dropped = total_grid - len(grid)
        if grid.empty:
            raise GenerationError(
                "No calendar day in the requested range matches the scenario/request conditions on "
                f"{ {c: list(v) for c, v in spec.conditions.items() if c in CALENDAR_KEYS} }."
            )
        if dropped:
            warnings_.append(
                f"{dropped} of {total_grid} ward-days in the calendar range do not match the conditions on "
                "month/day_of_week and were dropped."
            )
        other = {c: v for c, v in spec.conditions.items() if c not in CALENDAR_KEYS}
        combos = expand_conditions(other)
        support, source = (None, "not_needed")
        if len(combos) > 1:
            support, source = self._support(spec.table, model, list(CALENDAR_KEYS) + list(other))
            if source != "reference_train_split":
                warnings_.append(
                    "Reference train split not available: calendar days were allocated using "
                    + ("the model's training frequencies." if source == "model_training_data" else "uniform weights.")
                )
        rng = np.random.default_rng(derive_seed(spec.seed, "calendar-dates"))
        cells: list[_Cell] = []
        backoffs = 0
        for (ward_id, month, dow), rows in grid.groupby(["ward_id", "month", "day_of_week"], sort=True):
            base = {"ward_id": ward_id, "month": int(month), "day_of_week": dow}
            feasible = [model.is_feasible({**base, **combo}) for combo in combos]
            if len(combos) == 1:
                weights = [1.0 if feasible[0] else 0.0]
            else:
                weights, used_backoff = self._calendar_weights(support, base, combos, feasible)
                backoffs += used_backoff
            group_dates = rows["date"].tolist()
            rng.shuffle(group_dates)
            allocation = largest_remainder(len(group_dates), weights)
            if sum(allocation) == 0:  # nothing feasible: keep the days as unfillable cells
                allocation = largest_remainder(len(group_dates), [1.0] * len(combos))
            start = 0
            for combo, n, ok in zip(combos, allocation, feasible):
                if n == 0:
                    continue
                cells.append(_Cell(
                    conditions={**base, **combo},
                    requested=n,
                    group={"ward_id": ward_id, **combo},
                    dates=sorted(group_dates[start:start + n]),
                    feasible=ok,
                    parent=len(cells),
                ))
                start += n
        infeasible_days = sum(c.requested for c in cells if not c.feasible)
        if infeasible_days:
            warnings_.append(
                f"{infeasible_days} calendar day(s) fall in ward/month/weekday/condition combinations that "
                f"model '{model_id}' never saw in training and cannot be generated."
            )
        if backoffs:
            LOGGER.debug("Calendar allocation used frequency back-off for %d groups", backoffs)
        info = {
            "start_date": spec.start_date.isoformat(),
            "end_date": spec.end_date.isoformat(),
            "days": len(dates),
            "wards": wards,
            "grid_rows": total_grid,
            "dropped_rows": dropped,
            "cells": len(cells),
        }
        return cells, source, info

    def _calendar_weights(
        self, support: pd.DataFrame | None, base: dict[str, Any], combos: list[dict[str, Any]], feasible: list[bool]
    ) -> tuple[list[float], int]:
        """Combination weights for one calendar group with hierarchical back-off.

        Tries (ward, month, weekday), then (ward, month), (ward) and finally the whole
        reference before falling back to uniform weights over feasible combinations.
        """
        levels = [base, {k: base[k] for k in ("ward_id", "month")}, {"ward_id": base["ward_id"]}, {}]
        for depth, level in enumerate(levels):
            counts = self._combo_counts(support, combos, level)
            weights = [float(c) if ok else 0.0 for c, ok in zip(counts, feasible)]
            if sum(weights) > 0:
                return weights, int(depth > 0)
        return [1.0 if ok else 0.0 for ok in feasible], 1

    # ================================================================== rejection loop (§8.3)
    def _run_rejection_loop(
        self, spec: _Spec, model: BedSynthSynthesizer, cells: list[_Cell], started: float
    ) -> _LoopStats:
        oversample = float(self.settings.get("oversample_factor", 1.6))
        max_batch = int(self.settings.get("max_batch_rows", 50_000))
        max_round_rows = int(self.settings.get("max_round_rows") or DEFAULT_MAX_ROUND_ROWS)
        stats = _LoopStats(acceptance_prior=self._acceptance_prior(model))
        factor = oversample
        if stats.acceptance_prior:
            factor = min(MAX_FIRST_BATCH_FACTOR, max(oversample, BATCH_GROWTH / max(stats.acceptance_prior, ACCEPTANCE_FLOOR)))
        stats.first_batch_factor = factor
        active = [c for c in cells if c.requested > 0 and c.feasible]
        for cell in active:
            cell.batch = int(min(max_batch, max(1, math.ceil(cell.requested * factor))))
        deadline = started + spec.time_budget_seconds

        def out_of_time() -> bool:
            return time.perf_counter() > deadline

        for attempt in range(spec.max_attempts):
            todo = [c for c in active if c.remaining > 0 and not c.exhausted]
            if not todo:
                break
            now = time.perf_counter()
            elapsed = now - started
            if elapsed > spec.time_budget_seconds:
                stats.stopped_reason = (
                    f"time budget of {spec.time_budget_seconds:g}s exhausted after {elapsed:.1f}s "
                    f"({stats.attempts} round(s) completed)"
                )
                break
            cap = self._round_row_cap(deadline - now, stats.rows_per_second, max_round_rows)
            scale = self._cap_round(todo, cap)
            stats.throttled_rounds += int(scale < 1.0)
            # A throttled round gives each stratum only part of its batch, so it counts as that
            # fraction of a round towards the strata patience rule.
            stats.effective_rounds += scale
            stats.attempts += 1
            round_started = time.perf_counter()
            frames = model.sample_many(
                [(c.conditions, c.batch) for c in todo], seed=derive_seed(spec.seed, "round", attempt),
                should_stop=out_of_time,
            )
            self._process_round(spec, todo, frames, stats)
            stats.timed_rows += sum(c.batch for c in todo)
            stats.timed_seconds += time.perf_counter() - round_started
            requested_rows = sum(c.requested_from_model for c in active)
            overall = sum(c.passed for c in active) / requested_rows if requested_rows else None
            for cell in todo:
                self._adapt_batch(cell, max_batch, overall)
            self._family_futility(active, max_batch)
            self._rebalance(active, stats, max_batch, overall, "exhausted")
            if stats.effective_rounds >= STRATA_PATIENCE_ROUNDS - 1e-9:
                # Strata still short after a fair chance hand their quota to more productive
                # siblings (bounds the cost of models that rarely produce a stratum).
                self._rebalance(active, stats, max_batch, overall, "patience")
        # Strata still short after the last round take their siblings' spare accepted rows.
        self._rebalance(active, stats, max_batch, None, "final")
        return stats

    @staticmethod
    def _round_row_cap(remaining_seconds: float, rows_per_second: float | None, max_round_rows: int) -> int:
        """Most rows the next round may request: ``max_round_rows`` and what fits in the remaining budget.

        The time-based part is ``remaining_seconds x throughput`` (the throughput measured in the
        request's earlier rounds, :data:`DEFAULT_ROWS_PER_SECOND` before the first one), floored at
        :data:`MIN_ROUND_ROWS` so a round always makes progress.
        """
        rate = rows_per_second if rows_per_second and rows_per_second > 0 else DEFAULT_ROWS_PER_SECOND
        by_time = max(float(MIN_ROUND_ROWS), max(remaining_seconds, 0.0) * rate)
        return max(1, int(min(float(max_round_rows), by_time)))

    @staticmethod
    def _cap_round(todo: list[_Cell], cap: int) -> float:
        """Scale the cells' batches down proportionally so the round requests at most ``cap`` rows.

        Every cell keeps a batch of at least one row. Returns the scale factor (1.0 = unchanged).
        """
        total = sum(c.batch for c in todo)
        if total <= cap:
            return 1.0
        scale = cap / total
        for cell in todo:
            cell.batch = max(1, int(cell.batch * scale))
        return scale

    def _rebalance(
        self, cells: list[_Cell], stats: _LoopStats, max_batch: int, overall: float | None, mode: str
    ) -> None:
        """Move unfilled quota of allocation strata to sibling strata of the same condition cell.

        ``mode``:

        * ``"exhausted"`` (every round) - the quota of abandoned strata goes to their open
          siblings in proportion to the reference weights;
        * ``"patience"`` (after :data:`STRATA_PATIENCE_ROUNDS` rounds) - every short stratum first
          uses its siblings' spare accepted rows, then hands the rest to siblings that accept
          rows more often (they keep sampling);
        * ``"final"`` (after the loop) - short strata only use their siblings' spare rows.

        Calendar dates move with the quota (siblings share ward, month and weekday), so every
        row keeps a matching date. Quota moves are counted in ``stats.quota_moves``; the rows they
        finally shift are reported as ``rebalanced_rows`` (:func:`rebalanced_rows`).
        """
        families: dict[int, list[_Cell]] = {}
        for cell in cells:
            if cell.stratum:
                families.setdefault(cell.parent, []).append(cell)
        for family in families.values():
            if len(family) < 2:
                continue
            for donor in family:
                if donor.remaining <= 0 or (mode == "exhausted" and not donor.exhausted):
                    continue
                if mode in ("patience", "final"):
                    self._borrow_spare(donor, family, stats)
                if mode == "final" or donor.remaining <= 0:
                    continue
                if mode == "exhausted":
                    targets = [c for c in family if c is not donor and not c.exhausted]
                else:
                    targets = [c for c in family if c is not donor and not c.exhausted
                               and c.passed > 0 and c.acceptance > donor.acceptance]
                if not targets:
                    continue
                shares = largest_remainder(donor.remaining, [max(c.weight, 1e-9) for c in targets])
                for target, share in zip(targets, shares):
                    if share <= 0:
                        continue
                    self._move_quota(donor, target, share)
                    stats.quota_moves += share
                    target.take_spare()
                    if target.remaining > 0:
                        self._adapt_batch(target, max_batch, overall)

    def _borrow_spare(self, donor: _Cell, family: list[_Cell], stats: _LoopStats) -> None:
        """Fill ``donor``'s open quota from siblings' spare rows (quota moves to those siblings)."""
        lenders = sorted((c for c in family if c is not donor and c.spare_rows > 0), key=lambda c: -c.weight)
        for lender in lenders:
            share = min(donor.remaining, lender.spare_rows)
            if share <= 0:
                break
            self._move_quota(donor, lender, share)
            stats.quota_moves += share
            lender.take_spare()

    @staticmethod
    def _family_futility(cells: list[_Cell], max_batch: int) -> None:
        """Abandon all strata of a condition cell together when none of them accepted a row.

        Applies the per-cell futility rule of :meth:`_adapt_batch` to the family's total quota,
        so splitting a hopeless request into many small strata does not multiply the rounds
        spent before giving up.
        """
        families: dict[int, list[_Cell]] = {}
        for cell in cells:
            if cell.stratum:
                families.setdefault(cell.parent, []).append(cell)
        for family in families.values():
            if any(c.passed for c in family) or all(c.exhausted for c in family):
                continue
            requested = sum(c.requested for c in family)
            futility = min(max_batch, max(FUTILITY_MIN_ROWS, FUTILITY_ROWS_PER_TARGET * requested))
            sampled = sum(c.sampled for c in family)
            asked = sum(c.requested_from_model for c in family)
            if sampled >= futility or asked >= FUTILITY_REQUEST_FACTOR * futility:
                for cell in family:
                    cell.exhausted = True

    @staticmethod
    def _move_quota(donor: _Cell, target: _Cell, rows: int) -> None:
        """Transfer ``rows`` of unfilled quota (and their calendar dates) from donor to target."""
        donor.requested -= rows
        target.requested += rows
        if donor.dates:
            moved = donor.dates[len(donor.dates) - rows:]
            donor.dates = donor.dates[: len(donor.dates) - rows]
            target.dates = sorted(target.dates + moved)

    @staticmethod
    def _acceptance_prior(model: BedSynthSynthesizer) -> float | None:
        """Raw hard-valid rate recorded in the model card after training (None if unknown)."""
        diagnostics = model.annotations.get("sample_diagnostics") or {}
        rate = diagnostics.get("raw_hard_valid_rate")
        try:
            rate = float(rate)
        except (TypeError, ValueError):
            return None
        return rate if 0.0 < rate <= 1.0 else None

    def _process_round(
        self, spec: _Spec, todo: list[_Cell], frames: list[pd.DataFrame], stats: _LoopStats
    ) -> None:
        """Validate one round of samples and hand accepted rows to their cells."""
        owners = np.concatenate([np.full(len(f), i, dtype=int) for i, f in enumerate(frames)]) if frames else np.array([], int)
        for cell, frame in zip(todo, frames):
            cell.requested_from_model += cell.batch
            cell.sampled += len(frame)
        non_empty = [f for f in frames if len(f)]
        if not non_empty:
            return
        raw = pd.concat(non_empty, ignore_index=True)
        stats.sampled_rows += len(raw)
        derived = schema.derive(spec.table, raw)
        report = validation.validate_table(derived, spec.table, include_soft=False)
        hard = pd.Series(report.hard_valid_mask).reindex(derived.index, fill_value=False).to_numpy(dtype=bool)
        for rule in report.rule_results:
            if rule.get("severity", "hard") == "hard" and int(rule.get("violations") or 0) > 0:
                stats.rejected_by_rule[rule["rule_id"]] = stats.rejected_by_rule.get(rule["rule_id"], 0) + int(rule["violations"])
        stats.hard_valid_rows += int(hard.sum())
        conditions_ok = self._conditions_ok(derived, owners, [c.conditions for c in todo])
        filters_ok = scenarios.filter_mask(derived, spec.filters).to_numpy(dtype=bool)
        accepted = hard & conditions_ok & filters_ok
        stats.rejected_by_filter += int((hard & ~(conditions_ok & filters_ok)).sum())
        stats.accepted_rows += int(accepted.sum())
        for position, cell in enumerate(todo):
            rows = np.flatnonzero(accepted & (owners == position))
            cell.passed += len(rows)
            open_quota = max(cell.remaining, 0)
            take = rows[:open_quota]
            if len(take):
                cell.frames.append(derived.iloc[take])
                cell.kept += len(take)
            if cell.stratum:  # keep a few surplus rows: sibling strata may need them later
                room = int(math.ceil(SPARE_ROWS_PER_CELL * cell.requested)) - cell.spare_rows
                extra = rows[open_quota:open_quota + max(room, 0)]
                if len(extra):
                    cell.spare.append(derived.iloc[extra])
                    cell.spare_rows += len(extra)

    @staticmethod
    def _conditions_ok(derived: pd.DataFrame, owners: np.ndarray, conditions: list[dict[str, Any]]) -> np.ndarray:
        """Per-row check that each row satisfies the conditions of the cell it was sampled for."""
        ok = np.ones(len(derived), dtype=bool)
        for column in sorted({c for cond in conditions for c in cond}):
            constrained = np.array([column in cond for cond in conditions], dtype=bool)[owners]
            expected = np.array([cond.get(column) for cond in conditions], dtype=object)[owners]
            matches = np.asarray(derived[column].to_numpy(dtype=object) == expected, dtype=bool)
            ok &= ~constrained | matches
        return ok

    @staticmethod
    def _adapt_batch(cell: _Cell, max_batch: int, overall: float | None) -> None:
        """Next batch = ceil(remaining / max(acceptance, 0.01) * 1.2), clamped; futility stop.

        ``acceptance`` is the cell's accepted / requested rows, shrunk towards the request-wide
        acceptance ``overall`` with :data:`PRIOR_PSEUDO_ROWS` pseudo-rows.
        """
        if cell.remaining <= 0:
            return
        futility = min(max_batch, max(FUTILITY_MIN_ROWS, FUTILITY_ROWS_PER_TARGET * cell.requested))
        if cell.passed == 0 and (
            cell.sampled >= futility or cell.requested_from_model >= FUTILITY_REQUEST_FACTOR * futility
        ):
            cell.exhausted = True
            return
        if overall is None:
            acceptance = cell.acceptance
        else:
            acceptance = (cell.passed + PRIOR_PSEUDO_ROWS * overall) / (cell.requested_from_model + PRIOR_PSEUDO_ROWS)
        acceptance = max(acceptance, ACCEPTANCE_FLOOR)
        cell.batch = int(min(max_batch, max(1, math.ceil(cell.remaining / acceptance * BATCH_GROWTH))))

    # ================================================================== output
    def _assemble(self, spec: _Spec, cells: list[_Cell]) -> pd.DataFrame:
        parts = []
        for cell in cells:
            if not cell.frames:
                continue
            frame = pd.concat(cell.frames, ignore_index=True)
            if spec.mode == "calendar":
                frame = frame.head(len(cell.dates)).copy()
                frame["date"] = cell.dates[: len(frame)]
            parts.append(frame)
        if parts:
            data = pd.concat(parts, ignore_index=True)
        else:
            data = schema.derive(spec.table, empty_training_frame(spec.table))
            if spec.mode == "calendar":
                data["date"] = pd.Series(dtype=object)
        if spec.mode == "calendar":
            data = data.sort_values(["ward_id", "date"], kind="mergesort").reset_index(drop=True)
        elif len(data):
            order = np.random.default_rng(derive_seed(spec.seed, "shuffle")).permutation(len(data))
            data = data.iloc[order].reset_index(drop=True)
        data = schema.assign_ids(data, spec.table)
        include_time = spec.mode == "calendar"
        if spec.include_derived:
            columns = schema.output_columns(spec.table, include_time=include_time)
        else:
            columns = schema.column_names(spec.table, ["key", "train"] + (["time"] if include_time else []))
        columns = [c for c in columns if c in data.columns]
        return schema.coerce_dtypes(schema.order_columns(data[columns], spec.table), spec.table)

    def _build_report(
        self,
        spec: _Spec,
        model_id: str,
        model: BedSynthSynthesizer,
        cells: list[_Cell],
        stats: _LoopStats,
        data: pd.DataFrame,
        reference_source: str,
        calendar_info: dict[str, Any] | None,
        warnings_: list[str],
        started: float,
    ) -> dict[str, Any]:
        requested = sum(c.requested for c in cells) if spec.mode == "calendar" else spec.num_rows
        generated = len(data)
        shortfall = max(0, requested - generated)
        acceptance = stats.accepted_rows / stats.sampled_rows if stats.sampled_rows else 0.0
        warnings_ = list(warnings_)
        # strata whose quota was handed to siblings are not a shortfall
        exhausted = [c for c in cells if c.exhausted and c.remaining > 0]
        for cell in exhausted[:5]:
            warnings_.append(
                f"Stopped sampling {cell.conditions}: no row satisfied the validation rules and "
                f"filters after {cell.sampled} sampled rows ({cell.requested_from_model} requested)."
            )
        if len(exhausted) > 5:
            warnings_.append(f"... and {len(exhausted) - 5} more condition combinations were abandoned.")
        if stats.stopped_reason:
            warnings_.append(f"Rejection loop stopped early: {stats.stopped_reason}.")
        if shortfall:
            short_cells = len({c.parent for c in cells if c.requested > c.kept})
            if stats.stopped_reason and stats.stopped_reason.startswith("time budget"):
                advice = "increase time_budget_seconds, request fewer rows or relax the conditions/filters."
            else:
                advice = (f"the scenario is rare under model '{model_id}': increase max_attempts or relax the "
                          "conditions/filters.")
            warnings_.append(
                f"Generated {generated} of {requested} requested rows after {stats.attempts} attempt(s) "
                f"({short_cells} condition combination(s) short); {advice}"
            )
        low = float(self.settings.get("low_acceptance_warning", 0.05))
        if stats.sampled_rows and acceptance < low:
            warnings_.append(
                f"Low acceptance rate ({acceptance:.1%} < {low:.0%}): most sampled rows violated hard "
                "validation rules or the scenario conditions/filters."
            )
        card = model.model_card()
        report = {
            "table": spec.table,
            "model_id": model_id,
            "model": {
                "id": model_id,
                "experiment_id": card.get("experiment_id"),
                "synthesizer": card.get("synthesizer"),
                "model": card.get("model"),
                "segment_by": card.get("segment_by"),
                "constraint_groups": card.get("constraint_groups"),
                "sdv_version": card.get("sdv_version"),
                "fitted_at": card.get("fitted_at"),
            },
            "scenario": spec.scenario.name if spec.scenario else None,
            "mode": spec.mode,
            "seed": spec.seed,
            "num_rows_requested": requested,
            "num_rows_generated": generated,
            "shortfall": shortfall,
            "attempts": stats.attempts,
            "max_attempts": spec.max_attempts,
            "sampled_rows": stats.sampled_rows,
            "accepted_rows": stats.accepted_rows,
            "acceptance_rate": round(acceptance, 6),
            "raw_hard_valid_rate": round(stats.hard_valid_rows / stats.sampled_rows, 6) if stats.sampled_rows else None,
            "rejected_by_rule": dict(sorted(stats.rejected_by_rule.items())),
            "rejected_by_filter": stats.rejected_by_filter,
            "first_batch_factor": round(stats.first_batch_factor, 4),
            "acceptance_prior": stats.acceptance_prior,
            "time_budget_seconds": spec.time_budget_seconds,
            "stopped_reason": stats.stopped_reason,
            "throttled_rounds": stats.throttled_rounds,
            "rows_per_second": round(stats.rows_per_second, 1) if stats.rows_per_second else None,
            "condition_allocation": self._allocation_report(spec, cells),
            "conditions": {c: list(v) for c, v in spec.conditions.items()},
            "filters": spec.filters,
            "reference_source": reference_source,
            "include_derived": spec.include_derived,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "warnings": warnings_,
        }
        if calendar_info is not None:
            report["calendar"] = calendar_info
        return json_safe(report)

    @staticmethod
    def _allocation_report(spec: _Spec, cells: list[_Cell]) -> list[dict[str, Any]]:
        """Per-combination allocation (calendar cells aggregated by ward + remaining conditions).

        Implicit allocation strata are summed back into the requested combination they belong
        to, so the report lists exactly the combinations of the request's conditions.
        """
        if spec.mode != "calendar":
            entries: dict[int, dict[str, Any]] = {}
            for cell in cells:
                entry = entries.get(cell.parent)
                if entry is None:
                    entries[cell.parent] = {
                        "conditions": dict(cell.group),
                        "requested": cell.requested,
                        "generated": cell.kept,
                        "reference_count": cell.reference_count,
                        "sampled": cell.sampled,
                        "accepted": cell.passed,
                        "feasible": cell.feasible,
                    }
                else:
                    entry["requested"] += cell.requested
                    entry["generated"] += cell.kept
                    entry["sampled"] += cell.sampled
                    entry["accepted"] += cell.passed
            return [entries[key] for key in sorted(entries)]
        groups: dict[tuple, dict[str, Any]] = {}
        for cell in cells:
            key = tuple(sorted((k, str(v)) for k, v in cell.group.items()))
            entry = groups.setdefault(key, {"conditions": dict(cell.group), "requested": 0, "generated": 0,
                                            "cells": 0, "sampled": 0, "accepted": 0})
            entry["requested"] += cell.requested
            entry["generated"] += min(cell.kept, len(cell.dates))
            entry["cells"] += 1
            entry["sampled"] += cell.sampled
            entry["accepted"] += cell.passed
        return list(groups.values())
