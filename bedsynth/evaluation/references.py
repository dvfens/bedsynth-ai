"""Reference points for the Week-3 scores: the real holdout and a no-learning baseline.

Absolute scores (fidelity 0.93, operational 0.86, ...) cannot be interpreted on their own:
real data does not score 1.0 against the training split (sampling noise), and a generator
that learns nothing can score well on families that reward per-group marginals. Two
*reference pseudo-models* are therefore scored with exactly the same families, inside the
same generation system (the generator's allocation strata), and shown next to the models:

``real_holdout`` - the **ceiling**: what a perfect model would deliver. Rows are real holdout
    rows (never seen by any model), drawn with the generator's allocation-strata quotas
    (seeded bootstrap to the evaluated size; scenario rows from the holdout rows inside the
    scenario). Families whose real-data test set is the holdout use disjoint real data
    instead: ML utility trains on a size-matched real-train subsample (TSTR = a second
    size-matched TRTR), the booking-quota decision is made on that train subsample and scored
    out of sample on the holdout, and privacy compares one half of the holdout with the other
    half as the "unseen" baseline. Validity is the holdout's (100 % hard-valid by cleaning).
``null_stratified_marginals`` - the **floor**: a generator without learning. Each row draws
    its allocation stratum (``ward_id x occupancy_level`` / ``ward_id x admission_source x
    delay_reason``) with the training frequencies, then every other train column
    *independently* from training rows of that stratum (within-stratum marginals kept, all
    within-row relationships destroyed). Hard-invalid rows are rejected like the generator
    does; its raw validity is the hard-valid rate of the unfiltered draws. Privacy for this
    baseline is meaningless in one direction - it copies training *values* column by column
    - and is reported as computed.

:func:`normalised_scores` maps a model's component scores to ``(model - null) / (real - null)``:
0 = no better than the no-learning baseline, 1 = as good as real data (values above 1 or
below 0 are kept, e.g. when strata copying beats real-data noise).
"""
from __future__ import annotations

import logging
import time
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from bedsynth import validation
from bedsynth.config import get_seed
from bedsynth.evaluation.scoring import evaluate_model, finite_or_none, to_jsonable
from bedsynth.evaluation.statistical import ensure_derived, subsample
from bedsynth.scenarios import get_scenario, scenario_mask
from bedsynth.schema import TABLES, WARD_DAILY, assign_ids, derive, training_columns

logger = logging.getLogger(__name__)

REAL_REFERENCE_ID = "real_holdout"
NULL_REFERENCE_ID = "null_stratified_marginals"
REFERENCE_IDS: tuple[str, ...] = (REAL_REFERENCE_ID, NULL_REFERENCE_ID)
NULL_POOL_FACTOR = 4             # raw null rows drawn per evaluated row (before hard-rule rejection)
MIN_NULL_POOL_ROWS = 20_000
MIN_NORMALISATION_GAP = 0.05     # real - null below this: the family cannot separate them, normalisation undefined

REFERENCE_DESCRIPTIONS: dict[str, str] = {
    REAL_REFERENCE_ID: "Ceiling: real holdout rows (unseen by every model) drawn with the generator's allocation-strata "
                       "quotas; ML utility and the quota decision use a size-matched real-train subsample, privacy "
                       "compares holdout halves.",
    NULL_REFERENCE_ID: "Floor: no learning - allocation stratum by training frequency, every other column drawn "
                       "independently from training rows of that stratum, hard-invalid rows rejected.",
}


# ---------------------------------------------------------------------------
# Pool generator (same allocation strata as the SyntheticDataGenerator)
# ---------------------------------------------------------------------------
def _largest_remainder(total: int, weights: Sequence[float]) -> list[int]:
    """Split ``total`` into integers proportional to ``weights`` (largest-remainder method)."""
    w = np.asarray(weights, dtype=float)
    if total <= 0 or w.size == 0 or w.sum() <= 0:
        return [0] * len(w)
    exact = w / w.sum() * total
    counts = np.floor(exact).astype(int)
    order = np.argsort(-(exact - counts), kind="stable")
    counts[order[: total - counts.sum()]] += 1
    return counts.tolist()


class PoolGenerator:
    """Stand-in for ``SyntheticDataGenerator`` that draws rows from a fixed pool.

    ``generate(request)`` honours ``table``, ``num_rows``, ``scenario`` and ``seed``: rows of
    ``pool`` inside the scenario are drawn with replacement, with per-stratum quotas that
    follow the joint frequencies of ``strata`` among the ``reference`` rows inside the scenario
    (largest remainder) - the allocation the generator applies. Strata without pool rows hand
    their quota to the other strata (counted as ``rebalanced_rows``). ``acceptance_rate`` is
    reported as given (the pool's hard-valid rate).
    """

    def __init__(self, pool: pd.DataFrame, reference: pd.DataFrame, table: str, strata: Sequence[str], *,
                 name: str, acceptance_rate: float | None = None):
        if table not in TABLES:
            raise ValueError(f"Unknown table '{table}'")
        self.table = table
        self.pool = ensure_derived(pool, table).reset_index(drop=True)
        self.reference = ensure_derived(reference, table).reset_index(drop=True)
        self.strata = [c for c in strata if c in self.pool.columns and c in self.reference.columns]
        self.name = name
        self.acceptance_rate = acceptance_rate

    def __repr__(self) -> str:
        return f"PoolGenerator(name={self.name!r}, table={self.table!r}, pool_rows={len(self.pool)})"

    def _subset(self, frame: pd.DataFrame, scenario: str | None) -> pd.DataFrame:
        if not scenario:
            return frame
        return frame.loc[scenario_mask(frame, get_scenario(scenario)).to_numpy(dtype=bool)]

    def _draw(self, pool: pd.DataFrame, reference: pd.DataFrame, n_rows: int,
              rng: np.random.Generator) -> tuple[pd.DataFrame, int]:
        """Stratified draw of ``n_rows`` pool rows; returns the rows and the rebalanced quota."""
        if not self.strata or reference.empty:
            return pool.iloc[rng.integers(0, len(pool), n_rows)], 0
        weights = reference.groupby(self.strata, sort=True).size()
        groups = pool.groupby(self.strata, sort=True).indices
        keys = [k if isinstance(k, tuple) else (k,) for k in weights.index]
        lookup = {(k if isinstance(k, tuple) else (k,)): v for k, v in groups.items()}
        quotas = _largest_remainder(n_rows, weights.to_numpy())
        available = [lookup.get(key) is not None for key in keys]
        moved = sum(q for q, ok in zip(quotas, available) if not ok)
        if moved:
            extra = _largest_remainder(moved, [w if ok else 0.0 for w, ok in zip(weights.to_numpy(), available)])
            quotas = [q + e if ok else 0 for q, e, ok in zip(quotas, extra, available)]
        parts = [pool.iloc[lookup[key][rng.integers(0, len(lookup[key]), quota)]]
                 for key, quota in zip(keys, quotas) if quota > 0 and lookup.get(key) is not None]
        if not parts:  # no stratum of the reference exists in the pool: unstratified draw
            return pool.iloc[rng.integers(0, len(pool), n_rows)], n_rows
        return pd.concat(parts), int(moved)

    def generate(self, request: Any) -> Any:
        """Draw ``request.num_rows`` rows (see class docstring); returns a ``GenerationResult``."""
        from bedsynth.synthesis.generator import GenerationError, GenerationResult

        if request.table != self.table:
            raise GenerationError(f"{self!r} generates '{self.table}' rows, not '{request.table}'.")
        start = time.perf_counter()
        seed = get_seed() if request.seed is None else int(request.seed)
        rng = np.random.default_rng(seed)
        pool = self._subset(self.pool, request.scenario)
        reference = self._subset(self.reference, request.scenario)
        n_rows = int(request.num_rows)
        warnings_: list[str] = []
        if pool.empty or n_rows <= 0:
            data, moved = self.pool.iloc[0:0], 0
            if n_rows > 0:
                warnings_.append(f"{self.name}: no pool rows inside scenario {request.scenario!r}.")
        else:
            data, moved = self._draw(pool, reference, n_rows, rng)
        data = data.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        data = assign_ids(data, self.table) if len(data) else data
        report = {
            "table": self.table, "model_id": self.name, "scenario": request.scenario, "seed": seed,
            "num_rows_requested": n_rows, "num_rows_generated": len(data), "shortfall": max(0, n_rows - len(data)),
            "sampled_rows": len(data), "accepted_rows": len(data), "acceptance_rate": self.acceptance_rate,
            "stopped_reason": None, "warnings": warnings_,
            "allocation_strata": {"columns": list(self.strata), "source": "reference_train_split",
                                  "rebalanced_rows": moved, "pool_rows": len(pool)},
            "elapsed_seconds": round(time.perf_counter() - start, 3),
        }
        return GenerationResult(data=data, report=report)


# ---------------------------------------------------------------------------
# No-learning baseline
# ---------------------------------------------------------------------------
def stratified_marginals(train: pd.DataFrame, table: str, n_rows: int, strata: Sequence[str],
                         seed: int) -> pd.DataFrame:
    """No-learning draws: stratum by training frequency, every other train column independent within it."""
    rng = np.random.default_rng(seed)
    columns = training_columns(table)
    frame = train[columns].reset_index(drop=True)
    strata = [c for c in strata if c in columns]
    if not strata:
        return derive(table, pd.DataFrame({c: frame[c].to_numpy()[rng.integers(0, len(frame), n_rows)]
                                           for c in columns}))
    groups = frame.groupby(strata, sort=True).indices
    keys = list(groups)
    sizes = np.array([len(groups[k]) for k in keys], dtype=float)
    picks = rng.choice(len(keys), size=int(n_rows), p=sizes / sizes.sum())
    parts = []
    for index in np.unique(picks):
        m, rows = int((picks == index).sum()), groups[keys[index]]
        parts.append(pd.DataFrame({c: frame[c].to_numpy()[rows[rng.integers(0, len(rows), m)]] for c in columns}))
    out = pd.concat(parts, ignore_index=True)
    return derive(table, out.sample(frac=1.0, random_state=seed).reset_index(drop=True))


def null_pool(train: pd.DataFrame, table: str, n_rows: int, strata: Sequence[str], seed: int
              ) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """``(hard-valid pool, raw draws of n_rows, raw hard-valid rate)`` of the no-learning baseline."""
    pool_rows = max(MIN_NULL_POOL_ROWS, NULL_POOL_FACTOR * int(n_rows))
    raw = stratified_marginals(train, table, pool_rows, strata, seed)
    valid = np.asarray(validation.validate_table(raw, table, include_soft=False).hard_valid_mask, dtype=bool)
    return raw.loc[valid].reset_index(drop=True), raw.head(int(n_rows)).reset_index(drop=True), float(valid.mean())


# ---------------------------------------------------------------------------
# Reference evaluation
# ---------------------------------------------------------------------------
def _request(table: str, n_rows: int, seed: int) -> Any:
    from bedsynth.synthesis.generator import GenerationRequest

    return GenerationRequest(table=table, num_rows=int(n_rows), seed=int(seed))


def _holdout_halves(holdout: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    order = np.random.default_rng(seed).permutation(len(holdout))
    half = len(holdout) // 2
    return holdout.iloc[order[:half]].reset_index(drop=True), holdout.iloc[order[half:]].reset_index(drop=True)


def evaluate_real_reference(table: str, train: pd.DataFrame, holdout: pd.DataFrame, n_rows: int, *,
                            strata: Sequence[str], seed: int | None = None, **options: Any) -> dict[str, Any]:
    """Score the real holdout like a model (see module docstring); ``options`` go to ``evaluate_model``."""
    from bedsynth.evaluation.optimization import evaluate_optimization_utility
    from bedsynth.evaluation.privacy import evaluate_privacy
    from bedsynth.evaluation.utility import evaluate_ml_utility
    from bedsynth.evaluation.validity import evaluate_validity

    seed = get_seed() if seed is None else int(seed)
    train = ensure_derived(train, table).reset_index(drop=True)
    holdout = ensure_derived(holdout, table).reset_index(drop=True)
    generator = PoolGenerator(holdout, train, table, strata, name=REAL_REFERENCE_ID, acceptance_rate=1.0)
    draws = generator.generate(_request(table, n_rows, seed)).data
    train_sample = subsample(train, int(n_rows), seed + 11)
    half_a, half_b = _holdout_halves(holdout, seed + 13)
    overrides = {
        "validity": lambda: evaluate_validity(holdout, table, basis="real_holdout"),
        "ml_utility": lambda: evaluate_ml_utility(train, holdout, train_sample, table, seed=seed),
        "privacy": lambda: evaluate_privacy(train, half_a, table, half_b, seed=seed, run_sdmetrics=False),
    }
    if table == WARD_DAILY:
        overrides["optimization"] = lambda: evaluate_optimization_utility(train, draws, holdout=holdout,
                                                                          quota_data=train_sample)
    results = evaluate_model(table, train, holdout, draws, generator=generator,
                             model_id=REAL_REFERENCE_ID, seed=seed, privacy_sdmetrics=False, forced_columns=strata,
                             model_view=False, output_validity=False, overrides=overrides, **options)
    results["reference"] = {
        "id": REAL_REFERENCE_ID, "kind": "ceiling", "description": REFERENCE_DESCRIPTIONS[REAL_REFERENCE_ID],
        "basis": {
            "statistical/operational/scenario_coverage/policy+capacity": "holdout rows, allocation-strata bootstrap",
            "validity": "real holdout split",
            "ml_utility": f"size-matched real-train subsample ({len(train_sample)} rows) as the training set",
            "booking_quota": "decision on the real-train subsample, scored out of sample on the holdout",
            "privacy": f"holdout half ({len(half_a)} rows) vs the other half as the unseen baseline",
        },
    }
    return results


def evaluate_null_reference(table: str, train: pd.DataFrame, holdout: pd.DataFrame, n_rows: int, *,
                            strata: Sequence[str], seed: int | None = None, **options: Any) -> dict[str, Any]:
    """Score the no-learning stratified-marginals baseline like a model; ``options`` go to ``evaluate_model``."""
    seed = get_seed() if seed is None else int(seed)
    train = ensure_derived(train, table).reset_index(drop=True)
    pool, raw, valid_rate = null_pool(train, table, n_rows, strata, seed + 21)
    generator = PoolGenerator(pool, train, table, strata, name=NULL_REFERENCE_ID, acceptance_rate=valid_rate)
    sample = generator.generate(_request(table, n_rows, seed)).data
    results = evaluate_model(table, train, holdout, sample, raw_synthetic=raw, generator=generator,
                             model_id=NULL_REFERENCE_ID, seed=seed, privacy_sdmetrics=False, forced_columns=strata,
                             model_view=False, output_validity=True, **options)
    results["reference"] = {
        "id": NULL_REFERENCE_ID, "kind": "floor", "description": REFERENCE_DESCRIPTIONS[NULL_REFERENCE_ID],
        "basis": {"pool_rows": len(pool), "raw_hard_valid_rate": valid_rate, "strata": list(strata)},
    }
    return results


def evaluate_references(table: str, train: pd.DataFrame, holdout: pd.DataFrame, n_rows: int, *,
                        strata: Sequence[str], seed: int | None = None, **options: Any) -> dict[str, dict[str, Any]]:
    """``{reference_id: evaluate_model-style results}`` of both reference pseudo-models.

    A reference that fails is logged and returned as ``{"model_id": id, "error": ...}`` so a
    broken reference never stops the model evaluation.
    """
    out: dict[str, dict[str, Any]] = {}
    for ref_id, func in ((REAL_REFERENCE_ID, evaluate_real_reference), (NULL_REFERENCE_ID, evaluate_null_reference)):
        start = time.perf_counter()
        try:
            out[ref_id] = func(table, train, holdout, n_rows, strata=strata, seed=seed, **options)
        except Exception as exc:  # noqa: BLE001 - references are context, never fatal
            logger.exception("Reference %s/%s failed", table, ref_id)
            out[ref_id] = {"table": table, "model_id": ref_id, "error": f"{type(exc).__name__}: {exc}"}
        logger.info("reference %s/%s evaluated in %.1f s", table, ref_id, time.perf_counter() - start)
    return out


def normalised_scores(components: Mapping[str, Any], real: Mapping[str, Any] | None,
                      null: Mapping[str, Any] | None) -> dict[str, float | None]:
    """``(model - null) / (real - null)`` per key of ``components`` (``None`` when undefined).

    ``components`` / ``real`` / ``null`` map component names (and ``composite``) to scores.
    Undefined when a score is missing or the real-null gap is below 0.05: the family then cannot
    tell real data from the no-learning baseline and the ratio would only amplify noise.
    """
    out: dict[str, float | None] = {}
    for key, value in components.items():
        m = finite_or_none(value)
        r = finite_or_none((real or {}).get(key))
        z = finite_or_none((null or {}).get(key))
        if m is None or r is None or z is None or (r - z) < MIN_NORMALISATION_GAP:
            out[key] = None
        else:
            out[key] = (m - z) / (r - z)
    return to_jsonable(out)


__all__ = [
    "NULL_REFERENCE_ID",
    "PoolGenerator",
    "REAL_REFERENCE_ID",
    "REFERENCE_DESCRIPTIONS",
    "REFERENCE_IDS",
    "evaluate_null_reference",
    "evaluate_real_reference",
    "evaluate_references",
    "normalised_scores",
    "null_pool",
    "stratified_marginals",
]
