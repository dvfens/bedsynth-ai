"""Score primitives, JSON sanitisation, composite scoring and the evaluation runner.

Conventions shared by every evaluation family (DESIGN.md §9):

* each family returns a JSON-serialisable ``dict`` with a ``name``, a one-line
  ``description`` and a top-level ``score`` in [0, 1], where 1 means the synthetic data is
  indistinguishable from (or as useful as) the real data. ``score`` is ``None`` only when
  it is not meaningful (e.g. no ML task is learnable from the real data);
* raw statistics are mapped to scores with the primitives below (complements of
  distances, relative-error and absolute-difference scores, correlation agreement), so
  every family reads the same way in the Week-3 report;
* :func:`to_jsonable` converts numpy / pandas objects to plain Python and replaces NaN and
  infinities with ``None``.

:func:`composite_score` combines the family scores with
``settings.evaluation.composite_weights``; :func:`evaluate_model` runs every family for one
trained model and is what ``scripts/03_evaluate.py`` calls.
"""
from __future__ import annotations

import logging
import math
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import numpy as np
import pandas as pd

from bedsynth.config import get_seed, section
from bedsynth.schema import WARD_DAILY

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Composite definition
# ---------------------------------------------------------------------------
DEFAULT_COMPOSITE_WEIGHTS: dict[str, float] = {
    "fidelity": 0.30,
    "validity": 0.20,
    "operational": 0.25,
    "utility": 0.15,
    "privacy": 0.10,
}

# composite component -> result families (keys of the evaluate_model dict) averaged into it
COMPOSITE_COMPONENTS: dict[str, tuple[str, ...]] = {
    "fidelity": ("statistical",),
    "validity": ("validity",),
    "operational": ("operational",),
    "utility": ("ml_utility", "optimization"),
    "privacy": ("privacy",),
}

COMPONENT_DESCRIPTIONS: dict[str, str] = {
    "fidelity": "SDV QualityReport overall score (column shapes + column pair trends).",
    "validity": "Share of raw model samples that pass every hard operational rule.",
    "operational": "Mean score of the occupancy / scheduling / patient-flow pattern checks.",
    "utility": "Mean of ML utility (train-synthetic-test-real) and optimisation utility (decision agreement).",
    "privacy": "Distance-to-closest-record based protection against copying training records.",
}


# ---------------------------------------------------------------------------
# JSON sanitisation
# ---------------------------------------------------------------------------
def _json_key(key: Any) -> str:
    """Dictionary keys must be strings in JSON; tuples become 'a|b'."""
    if isinstance(key, str):
        return key
    if isinstance(key, (bool, np.bool_)):
        return str(bool(key))
    if isinstance(key, (int, np.integer)):
        return str(int(key))
    if isinstance(key, (float, np.floating)):
        value = float(key)
        return str(int(value)) if value.is_integer() else str(value)
    if isinstance(key, tuple):
        return "|".join(_json_key(k) for k in key)
    return str(key)


def to_jsonable(obj: Any, ndigits: int | None = 6) -> Any:
    """Recursively convert ``obj`` to JSON-native types.

    numpy scalars/arrays and pandas objects become Python scalars, lists and dicts; NaN,
    infinities, ``pd.NA`` and ``NaT`` become ``None``; floats are rounded to ``ndigits``
    (``None`` keeps full precision); tuples and sets become lists.
    """
    if obj is None or isinstance(obj, str):
        return obj
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        value = float(obj)
        if not math.isfinite(value):
            return None
        return round(value, ndigits) if ndigits is not None else value
    if isinstance(obj, Mapping):
        return {_json_key(k): to_jsonable(v, ndigits) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v, ndigits) for v in obj]
    if isinstance(obj, (set, frozenset)):
        return [to_jsonable(v, ndigits) for v in sorted(obj, key=str)]
    if isinstance(obj, np.ndarray):
        return [to_jsonable(v, ndigits) for v in obj.tolist()]
    if isinstance(obj, pd.Series):
        return {_json_key(k): to_jsonable(v, ndigits) for k, v in obj.items()}
    if isinstance(obj, pd.DataFrame):
        return [to_jsonable(record, ndigits) for record in obj.to_dict(orient="records")]
    if obj is pd.NaT or obj is pd.NA:
        return None
    if isinstance(obj, (pd.Timestamp, datetime, date)):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    try:
        if pd.isna(obj):
            return None
    except (TypeError, ValueError):
        pass
    return str(obj)


# ---------------------------------------------------------------------------
# Score primitives
# ---------------------------------------------------------------------------
def finite_or_none(value: Any) -> float | None:
    """``float(value)`` if it is a finite number, else ``None``."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def clip01(value: Any) -> float | None:
    """Clip a number to [0, 1] (``None`` for missing / non-finite input)."""
    number = finite_or_none(value)
    if number is None:
        return None
    return min(1.0, max(0.0, number))


def mean_or_none(values: Iterable[Any]) -> float | None:
    """Mean of the finite values in ``values`` (``None`` if there are none)."""
    finite = [v for v in (finite_or_none(x) for x in values) if v is not None]
    return float(np.mean(finite)) if finite else None


def complement_score(distance: Any) -> float | None:
    """Score of a [0, 1] distance such as KS or total variation: ``1 - distance``."""
    number = finite_or_none(distance)
    return None if number is None else clip01(1.0 - number)


def relative_error(real: Any, synthetic: Any, floor: float = 1e-9) -> float | None:
    """``|synthetic - real| / max(|real|, floor)`` (``floor`` guards near-zero references)."""
    r, s = finite_or_none(real), finite_or_none(synthetic)
    if r is None or s is None:
        return None
    return abs(s - r) / max(abs(r), floor)


def relative_error_score(real: Any, synthetic: Any, floor: float = 1e-9) -> float | None:
    """``1 - relative error``, clipped to [0, 1]."""
    error = relative_error(real, synthetic, floor)
    return None if error is None else clip01(1.0 - error)


def abs_diff_score(real: Any, synthetic: Any, scale: float) -> float | None:
    """``1 - |synthetic - real| / scale`` clipped to [0, 1] (``scale`` = difference scored 0)."""
    r, s = finite_or_none(real), finite_or_none(synthetic)
    if r is None or s is None:
        return None
    return clip01(1.0 - abs(s - r) / scale)


def correlation_agreement(real: Any, synthetic: Any, tolerance: float = 0.10) -> dict[str, Any]:
    """Compare a real and a synthetic correlation coefficient.

    ``sign_agrees`` is True when both coefficients have the same sign or differ by at most
    ``tolerance`` (two weak correlations of opposite sign are not a disagreement).
    ``score = 0.5 * sign_agrees + 0.5 * (1 - min(1, |difference|))``. A synthetic
    coefficient that is undefined (constant column) counts as 0.
    """
    r = finite_or_none(real)
    if r is None:
        return {"real": None, "synthetic": finite_or_none(synthetic), "abs_diff": None, "sign_agrees": None, "score": None}
    s = finite_or_none(synthetic)
    s = 0.0 if s is None else s
    diff = abs(r - s)
    agrees = bool(np.sign(r) == np.sign(s) or diff <= tolerance)
    score = 0.5 * float(agrees) + 0.5 * (1.0 - min(1.0, diff))
    return {"real": r, "synthetic": s, "abs_diff": diff, "sign_agrees": agrees, "score": score}


def family_score(result: Any) -> float | None:
    """Extract a family's ``score`` from an evaluation result (``None`` if absent/failed)."""
    if not isinstance(result, Mapping):
        return None
    return clip01(result.get("score"))


def family_failed(result: Any) -> bool:
    """True when an evaluation family raised (``_run_family`` recorded an ``error``)."""
    return isinstance(result, Mapping) and bool(result.get("error")) and result.get("score") is None


# ---------------------------------------------------------------------------
# Composite score
# ---------------------------------------------------------------------------
def composite_weights() -> dict[str, float]:
    """Composite weights from ``settings.evaluation.composite_weights`` (defaults if absent)."""
    configured = (section("evaluation") or {}).get("composite_weights") or {}
    weights = dict(DEFAULT_COMPOSITE_WEIGHTS)
    weights.update({k: float(v) for k, v in configured.items() if k in DEFAULT_COMPOSITE_WEIGHTS})
    return weights


def composite_score(results: dict[str, Any], weights: dict[str, float] | None = None) -> dict[str, Any]:
    """Weighted composite of the family scores (DESIGN.md §9.9).

    ``results`` is an :func:`evaluate_model`-style dict whose values are family results
    (``statistical``, ``validity``, ``operational``, ``ml_utility``, ``optimization``,
    ``privacy``). Each component is the mean of its defined family scores
    (:data:`COMPOSITE_COMPONENTS`; a failed family counts as 0): ``utility`` is the mean of ML and
    optimisation utility on ward_daily and ML utility alone on patient_stays, where optimisation is
    undefined.

    * A family that **failed** (raised; ``error`` recorded) scores 0 - a crash must never
      raise the composite - and the composite is marked ``complete = False`` (listed in
      ``failed_families`` / ``failed_components``); model selection ignores incomplete
      composites.
    * A family that is **undefined by design** (skipped, e.g. optimisation on patient_stays,
      or no informative ML task) is left out of its component's mean; the component weights
      stay as configured. Only when a whole component is undefined (every family of it) is that
      component left out of the composite and the remaining weights renormalised
      (``effective_weights``); such components are listed in ``missing_components``.
    """
    weights = dict(weights or composite_weights())
    components: dict[str, float | None] = {}
    parts: dict[str, dict[str, float | None]] = {}
    failed_families: list[str] = []
    for component, families in COMPOSITE_COMPONENTS.items():
        family_scores: dict[str, float | None] = {}
        for family in families:
            result = results.get(family)
            if family_failed(result):
                failed_families.append(family)
                family_scores[family] = 0.0
            else:
                family_scores[family] = family_score(result)
        parts[component] = family_scores
        components[component] = mean_or_none(family_scores.values())

    available = {k: v for k, v in components.items() if v is not None and weights.get(k, 0.0) > 0}
    total_weight = sum(weights[k] for k in available)
    if total_weight > 0:
        score = sum(weights[k] * v for k, v in available.items()) / total_weight
        effective = {k: weights[k] / total_weight for k in available}
    else:
        score, effective = None, {}
    failed_components = [c for c, families in COMPOSITE_COMPONENTS.items() if set(families) & set(failed_families)]
    return to_jsonable({
        "name": "composite",
        "description": "Weighted sum of fidelity, validity, operational, utility and privacy scores "
                       "(weights from settings.evaluation.composite_weights, renormalised over the components "
                       "defined for the table; a failed family scores 0 and marks the composite incomplete).",
        "score": clip01(score),
        "complete": not failed_families,
        "components": components,
        "component_parts": parts,
        "component_descriptions": COMPONENT_DESCRIPTIONS,
        "weights": weights,
        "effective_weights": effective,
        "missing_components": [k for k, v in components.items() if v is None],
        "failed_families": failed_families,
        "failed_components": failed_components,
    })


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------
def _run_family(name: str, func: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    """Run one evaluation family; a failure is recorded instead of aborting the evaluation."""
    start = time.perf_counter()
    try:
        result = dict(func())
    except Exception as exc:  # noqa: BLE001 - one failing family must not lose the others
        logger.exception("Evaluation family '%s' failed", name)
        result = {"name": name, "score": None, "error": f"{type(exc).__name__}: {exc}"}
    result["seconds"] = round(time.perf_counter() - start, 3)
    logger.info("evaluation %-12s score=%s (%.1f s)", name, result.get("score"), result["seconds"])
    return result


def _skipped(name: str, reason: str) -> dict[str, Any]:
    return {"name": name, "score": None, "skipped": True, "reason": reason}


FAMILIES: tuple[str, ...] = ("statistical", "validity", "operational", "privacy", "ml_utility", "optimization",
                             "scenario_coverage")


def _compact_operational(result: Mapping[str, Any]) -> dict[str, Any]:
    """Operational result without the per-group detail (for the model view)."""
    keep = ("score", "n_checks", "n_passed", "pass_rate", "failed_checks", "beyond_real_noise", "system_checks",
            "group_scores", "error", "seconds")
    out = {k: result.get(k) for k in keep if k in result}
    out["checks"] = [{k: c.get(k) for k in ("name", "group", "metric", "value", "score", "passed", "strata_forced")}
                     for c in result.get("checks") or []]
    return out


def _compact_statistical(result: Mapping[str, Any]) -> dict[str, Any]:
    """Statistical result without the per-column detail (for the model view)."""
    quality = ((result.get("sdv") or {}).get("quality") or {})
    return {"score": result.get("score"), "score_source": result.get("score_source"),
            "properties": quality.get("properties"), "summary": result.get("summary"),
            "error": result.get("error"), "seconds": result.get("seconds")}


def evaluate_model(
    table: str,
    real_train: pd.DataFrame,
    real_holdout: pd.DataFrame,
    synthetic: pd.DataFrame,
    raw_synthetic: pd.DataFrame | None = None,
    generator: Any | None = None,
    model_id: str | None = None,
    include_scenarios: bool = True,
    seed: int | None = None,
    *,
    rows_per_scenario: int | None = None,
    privacy_sdmetrics: bool = True,
    forced_columns: list[str] | tuple[str, ...] | None = None,
    operational_calibration: Mapping[str, Any] | None = None,
    scenario_options: Mapping[str, Any] | None = None,
    model_view: bool = True,
    output_validity: bool = True,
    overrides: Mapping[str, Callable[[], dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Run every evaluation family for one trained model and compute the composite.

    ``synthetic``     - validated generator output (hard-invalid rows already rejected)
    ``raw_synthetic`` - derived but *unfiltered* model samples, used for the validity score
                        (how often the model produces operationally valid rows on its own);
                        falls back to ``synthetic`` when not given.
    ``generator``     - object with ``.generate(GenerationRequest) -> GenerationResult``,
                        required for scenario coverage.

    Options: ``forced_columns`` - the generator's allocation-strata columns (checks determined
    by them become system checks, see :mod:`bedsynth.evaluation.operational`);
    ``operational_calibration`` - real-vs-real noise of the operational checks at this size;
    ``scenario_options`` - keyword arguments of :func:`evaluate_scenario_coverage`
    (``time_budget_seconds``, ``max_attempts``, ``total_budget_seconds``, ``strata``);
    ``model_view`` - also score statistical fidelity and operational patterns on the raw model
    samples (``results["model_view"]``: what the model learned, before rejection and strata
    allocation); ``output_validity`` - hard/soft rule rates of the generator output
    (``results["output_validity"]``); ``overrides`` - ``{family: callable}`` replacing a family's
    computation (used by the real-data reference, see :mod:`bedsynth.evaluation.references`).

    Real-data roles: fidelity, operational patterns, policy evaluation and capacity planning
    compare against ``real_train`` (the data the model learned); ``real_holdout`` is the test
    set for ML utility, the out-of-sample set of the booking-quota regret and the privacy
    baseline. Each family is timed and isolated: an exception is recorded under that family
    (``score=None``, ``error``) and counts as 0 in an incomplete composite.
    """
    # Imported here because the family modules import the primitives defined above.
    from bedsynth.evaluation.coverage import evaluate_scenario_coverage
    from bedsynth.evaluation.operational import evaluate_operational
    from bedsynth.evaluation.optimization import evaluate_optimization_utility
    from bedsynth.evaluation.privacy import evaluate_privacy
    from bedsynth.evaluation.statistical import evaluate_statistical
    from bedsynth.evaluation.utility import evaluate_ml_utility
    from bedsynth.evaluation.validity import evaluate_validity

    seed = get_seed() if seed is None else int(seed)
    start = time.perf_counter()
    validity_frame = raw_synthetic if raw_synthetic is not None else synthetic
    validity_basis = "raw_model_samples" if raw_synthetic is not None else "generator_output"
    has_holdout = real_holdout is not None and len(real_holdout) > 0
    overrides = dict(overrides or {})

    def family(name: str, default: Callable[[], dict[str, Any]]) -> dict[str, Any]:
        return _run_family(name, overrides.get(name, default))

    results: dict[str, Any] = {
        "table": table,
        "model_id": model_id,
        "seed": seed,
        "evaluated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_rows": {
            "real_train": len(real_train),
            "real_holdout": len(real_holdout) if real_holdout is not None else 0,
            "synthetic": len(synthetic),
            "raw_synthetic": len(raw_synthetic) if raw_synthetic is not None else None,
        },
        "forced_columns": list(forced_columns or []),
    }
    results["statistical"] = family(
        "statistical", lambda: evaluate_statistical(real_train, synthetic, table, seed=seed))
    results["validity"] = family(
        "validity", lambda: evaluate_validity(validity_frame, table, basis=validity_basis))
    results["operational"] = family(
        "operational", lambda: evaluate_operational(real_train, synthetic, table, forced_columns=forced_columns,
                                                    calibration=operational_calibration))
    results["privacy"] = family(
        "privacy", lambda: evaluate_privacy(real_train, synthetic, table, real_holdout, seed=seed,
                                            run_sdmetrics=privacy_sdmetrics, disclosure_columns=forced_columns))
    if has_holdout:
        results["ml_utility"] = family(
            "ml_utility", lambda: evaluate_ml_utility(real_train, real_holdout, synthetic, table, seed=seed))
    else:
        results["ml_utility"] = _skipped("ml_utility", "No real holdout data to test on.")
    if table == WARD_DAILY:
        results["optimization"] = family(
            "optimization", lambda: evaluate_optimization_utility(real_train, synthetic,
                                                                  holdout=real_holdout if has_holdout else None))
    else:
        results["optimization"] = _skipped(
            "optimization", "Optimisation utility is defined on ward_daily (bed and booking decisions).")
    if include_scenarios and generator is not None:
        options = dict(scenario_options or {})
        results["scenario_coverage"] = family(
            "scenario_coverage",
            lambda: evaluate_scenario_coverage(generator, real_train, table, model_id=model_id,
                                               rows_per_scenario=rows_per_scenario, seed=seed, **options))
    else:
        reason = "Disabled (include_scenarios=False)." if not include_scenarios else "No generator supplied."
        results["scenario_coverage"] = _skipped("scenario_coverage", reason)
    if output_validity and raw_synthetic is not None:
        results["output_validity"] = _run_family(
            "output_validity", lambda: evaluate_validity(synthetic, table, basis="generator_output"))
    if model_view and raw_synthetic is not None:
        results["model_view"] = _model_view(table, real_train, raw_synthetic, seed, operational_calibration)

    results["composite"] = composite_score(results)
    results["timings"] = {name: results[name].get("seconds") for name in FAMILIES}
    for extra in ("output_validity", "model_view"):
        if extra in results:
            results["timings"][extra] = results[extra].get("seconds")
    results["timings"]["total"] = round(time.perf_counter() - start, 3)
    return to_jsonable(results)


def _model_view(table: str, real_train: pd.DataFrame, raw: pd.DataFrame, seed: int,
                calibration: Mapping[str, Any] | None) -> dict[str, Any]:
    """Statistical fidelity and operational patterns of the raw (unfiltered) model samples."""
    from bedsynth.evaluation.operational import evaluate_operational
    from bedsynth.evaluation.statistical import evaluate_statistical

    start = time.perf_counter()
    statistical = _run_family("model_view.statistical", lambda: evaluate_statistical(real_train, raw, table, seed=seed))
    operational = _run_family("model_view.operational",
                              lambda: evaluate_operational(real_train, raw, table, calibration=calibration))
    return {
        "name": "model_view",
        "description": "Families re-scored on raw model samples (no rejection, no allocation strata): what the "
                       "model learned on its own. The composite uses the generator output (system view).",
        "basis": "raw_model_samples",
        "n_rows": len(raw),
        "statistical": _compact_statistical(statistical),
        "operational": _compact_operational(operational),
        "seconds": round(time.perf_counter() - start, 3),
    }


__all__ = [
    "COMPOSITE_COMPONENTS",
    "DEFAULT_COMPOSITE_WEIGHTS",
    "FAMILIES",
    "abs_diff_score",
    "clip01",
    "complement_score",
    "composite_score",
    "composite_weights",
    "correlation_agreement",
    "evaluate_model",
    "family_failed",
    "family_score",
    "finite_or_none",
    "mean_or_none",
    "relative_error",
    "relative_error_score",
    "to_jsonable",
]
