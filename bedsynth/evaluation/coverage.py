"""Scenario coverage (DESIGN.md §9.8).

The generator must be able to produce every operational scenario of the catalogue
(``bedsynth.scenarios``) and the scenario data must resemble the matching real ward-days
or stays. For each scenario of a table this module:

1. measures **real support** - rows of the reference data inside the scenario
   (``scenarios.scenario_mask``);
2. **generates** ``rows_per_scenario`` rows through ``generator.generate`` (with the
   evaluation's per-request ``time_budget_seconds`` / ``max_attempts``) and records generated /
   requested, acceptance rate, seconds, the generator's ``allocation_strata`` block
   (rebalanced quota), any early stop, rounds the generator shrank to fit the time budget
   (``throttled_rounds``: such results depend on machine speed) and warnings;
3. scores **coverage** = mean of *fulfilment* (generated / requested) and *strata fidelity*
   ``1 - TVD`` between the output's mix of the generator's allocation strata
   (``ward_id x occupancy_level`` / ``ward_id x admission_source x delay_reason``) and the mix
   of the real scenario subset. The generator sets strata quotas from the training
   frequencies, so a model that cannot produce some strata only shows up through the quota
   moved to easier strata ("rebalancing"); filtered scenarios take their strata from the
   model's own filtered pilot draw, so their TVD also measures how well the model places the
   filtered rows. A strata TVD more than ``strata_tvd_flag`` (0.05) above the rounding floor
   (:func:`strata_rounding_tvd`: the TVD of the best possible allocation of that many rows,
   ~0.1 for patient_stays at 100 rows) is flagged. *Combination coverage* (share of material
   ``ward_id x occupancy_level`` /
   ``ward_id x admission_source`` combinations of the real subset present in the output) is
   still reported but, when its columns are allocation strata, it is forced by the quotas
   (``strata_forced``) and left out of the score - real samples of the same size reach only
   ~0.86-0.90 by chance;
4. scores **fidelity** with :func:`quick_quality` (synthetic subset vs real subset, the
   allocation-strata columns excluded because their mix is imposed) when both have at least
   20 rows, and compares scenario KPI means.

Scenario score = 0.5 · coverage + 0.5 · fidelity (coverage alone when fidelity cannot be
assessed; 0 when generation fails). ``score`` = mean scenario score over the evaluated
scenarios. With ``total_budget_seconds``, scenarios not started before the budget ran out
are listed as ``not_evaluated`` (excluded from the score, ``complete = False``).
"""
from __future__ import annotations

import logging
import time
from typing import Any, Sequence

import numpy as np
import pandas as pd

from bedsynth.config import get_seed, section
from bedsynth.evaluation.scoring import clip01, finite_or_none, mean_or_none, relative_error, to_jsonable
from bedsynth.evaluation.statistical import ensure_derived, quick_quality
from bedsynth.scenarios import Scenario, scenario_mask, scenarios_for
from bedsynth.schema import PATIENT_STAYS, WARD_DAILY

logger = logging.getLogger(__name__)

DEFAULT_ROWS_PER_SCENARIO = 400
MIN_FIDELITY_ROWS = 20
MATERIAL_COMBINATION_SHARE = 0.01
DEFAULT_STRATA_TVD_FLAG = 0.05

COMBINATION_COLUMNS: dict[str, tuple[str, str]] = {
    WARD_DAILY: ("ward_id", "occupancy_level"),
    PATIENT_STAYS: ("ward_id", "admission_source"),
}
#: Allocation strata of the generator (``settings.generation.allocation_strata`` overrides).
DEFAULT_STRATA_COLUMNS: dict[str, tuple[str, ...]] = {
    WARD_DAILY: ("ward_id", "occupancy_level"),
    PATIENT_STAYS: ("ward_id", "admission_source", "delay_reason"),
}
#: Derived columns that relabel a strata column (excluded from scenario fidelity with it).
STRATA_RELABELLINGS: dict[str, tuple[str, ...]] = {
    "ward_id": ("ward_type", "bed_capacity"),
}
SCENARIO_KPIS: dict[str, tuple[str, ...]] = {
    WARD_DAILY: (
        "occupancy_rate", "emergency_boarders", "avg_bed_wait_hours", "elective_scheduled",
        "elective_cancellation_rate", "delayed_discharge_rate", "bed_turnover_rate", "patients_per_nurse",
        "escalation_beds_open", "beds_closed",
    ),
    PATIENT_STAYS: (
        "los_days", "bed_wait_hours", "discharge_delay_hours", "bed_turnaround_minutes", "booking_lead_days",
        "is_delayed_discharge", "is_weekend_discharge", "discharged_before_noon",
    ),
}

METRIC_DESCRIPTIONS: dict[str, str] = {
    "real_support": "Rows (and share) of the real reference data that belong to the scenario.",
    "fulfilment": "Generated rows / requested rows (capped at 1).",
    "strata_mix_tvd": "Total variation distance between the output's allocation-strata mix and the real scenario "
                      "subset's mix (> 0 when the generator had to move quota to other strata).",
    "strata_rounding_tvd": "Smallest strata-mix TVD an output of that many rows can reach (rounding floor); the "
                           "rebalancing flag needs a TVD more than strata_tvd_flag above it.",
    "strata_fidelity": "1 - strata_mix_tvd.",
    "combination_coverage": "Share of material real combinations (>= 1% of the real scenario subset) present in the "
                            "synthetic subset (strata-forced when its columns are allocation strata).",
    "weighted_combination_coverage": "Share of real scenario rows whose combination appears in the synthetic subset.",
    "in_scenario_rate": "Share of generated rows that satisfy the scenario's conditions and filters.",
    "fidelity": "quick_quality score of the synthetic scenario subset against the real scenario subset "
                "(allocation-strata columns excluded).",
    "kpi_means": "Mean of each scenario KPI in real vs synthetic subsets, with relative difference.",
}


def strata_columns(table: str) -> tuple[str, ...]:
    """Allocation strata columns of ``table`` (``settings.generation.allocation_strata`` or the default)."""
    configured = ((section("generation") or {}).get("allocation_strata") or {})
    if table in configured:
        return tuple(configured[table] or ())
    return DEFAULT_STRATA_COLUMNS.get(table, ())


def forced_exclusions(columns: Sequence[str]) -> list[str]:
    """Strata columns plus the derived columns that relabel them."""
    out = list(columns)
    for column in columns:
        out += [c for c in STRATA_RELABELLINGS.get(column, ()) if c not in out]
    return out


def strata_mix_tvd(real: pd.DataFrame, synthetic: pd.DataFrame, columns: Sequence[str]) -> float | None:
    """TVD between the joint mix of ``columns`` in ``synthetic`` and in ``real`` (``None`` if undefined)."""
    columns = [c for c in columns if c in real.columns and c in synthetic.columns]
    if not columns or real.empty or synthetic.empty:
        return None
    p = real.groupby(columns, dropna=False, observed=True).size() / len(real)
    q = synthetic.groupby(columns, dropna=False, observed=True).size() / len(synthetic)
    index = p.index.union(q.index)
    return 0.5 * float((p.reindex(index, fill_value=0.0) - q.reindex(index, fill_value=0.0)).abs().sum())


def strata_rounding_tvd(real: pd.DataFrame, n_rows: int, columns: Sequence[str]) -> float | None:
    """Smallest strata-mix TVD any ``n_rows``-row output can reach against ``real`` (rounding floor).

    The largest-remainder allocation of ``n_rows`` to the real subset's strata - what a perfect
    generator delivers - still differs from the real shares by rounding: with ~60 patient_stays
    strata and 100 rows the floor is ~0.1, far above the 0.05 rebalancing flag. The flag is
    therefore raised only for a TVD beyond this floor plus the flag margin.
    """
    columns = [c for c in columns if c in real.columns]
    if not columns or real.empty or n_rows <= 0:
        return None
    shares = (real.groupby(columns, dropna=False, observed=True).size() / len(real)).to_numpy(dtype=float)
    exact = shares * n_rows
    counts = np.floor(exact)
    remainder = int(n_rows - counts.sum())
    if remainder > 0:
        counts[np.argsort(-(exact - counts), kind="stable")[:remainder]] += 1
    return 0.5 * float(np.abs(counts / n_rows - shares).sum())


def rebalancing_flagged(tvd: float | None, floor: float | None, margin: float) -> bool:
    """Whether a strata-mix TVD shows quota moved between strata (beyond rounding + ``margin``)."""
    return bool(tvd is not None and tvd > (floor or 0.0) + margin)


def combination_coverage(real: pd.DataFrame, synthetic: pd.DataFrame, columns: tuple[str, str],
                         min_share: float = MATERIAL_COMBINATION_SHARE) -> dict[str, Any]:
    """Coverage of the real subset's value combinations of ``columns`` by the synthetic subset."""
    if real.empty or any(c not in real.columns or c not in synthetic.columns for c in columns):
        return {"columns": list(columns), "n_real_combinations": 0, "coverage": None, "weighted_coverage": None}
    shares = real.groupby(list(columns)).size() / len(real)
    synthetic_combos = set(synthetic.groupby(list(columns)).size().index) if not synthetic.empty else set()
    material = shares[shares >= min_share]
    present = [combo in synthetic_combos for combo in material.index]
    missing = [list(combo) for combo, ok in zip(material.index, present) if not ok]
    weighted = float(sum(share for combo, share in shares.items() if combo in synthetic_combos))
    return {
        "columns": list(columns),
        "n_real_combinations": int(len(shares)),
        "n_material_combinations": int(len(material)),
        "coverage": float(sum(present) / len(present)) if present else None,
        "weighted_coverage": weighted,
        "missing_material_combinations": missing[:20],
    }


def _mean(series: pd.Series) -> float | None:
    """Mean of a numeric or boolean series (``None`` when empty)."""
    if series.empty:
        return None
    return finite_or_none(pd.to_numeric(series, errors="coerce").astype(float).mean())


def _kpi_means(real: pd.DataFrame, synthetic: pd.DataFrame, table: str) -> dict[str, dict[str, float | None]]:
    """Real vs synthetic mean of each scenario KPI present in both subsets."""
    out = {}
    for kpi in SCENARIO_KPIS[table]:
        if kpi not in real.columns or kpi not in synthetic.columns:
            continue
        r, s = _mean(real[kpi]), _mean(synthetic[kpi])
        out[kpi] = {"real": r, "synthetic": s, "relative_diff": relative_error(r, s, floor=1e-6)}
    return out


def _allocation_block(report: dict[str, Any]) -> dict[str, Any] | None:
    """The generator's allocation-strata report block (``None`` when the generator has none)."""
    block = report.get("allocation_strata")
    return dict(block) if isinstance(block, dict) else None


def evaluate_scenario(generator: Any, scenario: Scenario, reference: pd.DataFrame, table: str,
                      model_id: str | None, rows: int, seed: int, *,
                      time_budget_seconds: float | None = None, max_attempts: int | None = None,
                      strata: Sequence[str] | None = None,
                      strata_tvd_flag: float = DEFAULT_STRATA_TVD_FLAG) -> dict[str, Any]:
    """Generate one scenario and compare it with the matching real subset (see module docstring)."""
    from bedsynth.synthesis.generator import GenerationRequest

    strata = tuple(strata_columns(table) if strata is None else strata)
    real_subset = reference.loc[scenario_mask(reference, scenario)]
    entry: dict[str, Any] = {
        "name": scenario.name,
        "title": scenario.title,
        "category": scenario.category,
        "description": scenario.description,
        "status": "evaluated",
        "real_support": {"rows": len(real_subset), "share": len(real_subset) / len(reference) if len(reference) else None},
        "requested": rows,
        "seed": seed,
        "time_budget_seconds": time_budget_seconds,
        "max_attempts": max_attempts,
    }
    start = time.perf_counter()
    try:
        result = generator.generate(GenerationRequest(table=table, num_rows=rows, scenario=scenario.name,
                                                      model_id=model_id, seed=seed, max_attempts=max_attempts,
                                                      time_budget_seconds=time_budget_seconds))
    except Exception as exc:  # noqa: BLE001 - a failing scenario scores 0, the others continue
        logger.warning("Scenario '%s' failed: %s", scenario.name, exc)
        entry.update(status="failed", generated=0, seconds=round(time.perf_counter() - start, 3),
                     error=f"{type(exc).__name__}: {exc}", coverage=0.0, fidelity=None, score=0.0)
        return entry
    seconds = time.perf_counter() - start
    report = dict(result.report or {})
    synthetic = ensure_derived(result.data, table) if len(result.data) else result.data
    in_scenario = float(scenario_mask(synthetic, scenario).mean()) if len(synthetic) else None
    fulfilment = clip01(len(synthetic) / rows) if rows else None
    combo_columns = COMBINATION_COLUMNS[table]
    combos = combination_coverage(real_subset, synthetic, combo_columns)
    combos["strata_forced"] = bool(strata) and set(combo_columns) <= set(strata)
    tvd = strata_mix_tvd(real_subset, synthetic, strata) if strata else None
    rounding = strata_rounding_tvd(real_subset, len(synthetic), strata) if strata else None
    strata_fidelity = None if tvd is None else clip01(1.0 - tvd)
    parts = [fulfilment, strata_fidelity]
    if not combos["strata_forced"]:
        parts.append(combos["coverage"])
    coverage = mean_or_none(parts)

    fidelity, quality = None, None
    if len(real_subset) >= MIN_FIDELITY_ROWS and len(synthetic) >= MIN_FIDELITY_ROWS:
        quality = quick_quality(real_subset, synthetic, table, include_validity=False, seed=seed,
                                exclude_columns=forced_exclusions(strata))
        fidelity = quality["score"]
    score = coverage if fidelity is None else mean_or_none([coverage, fidelity])
    allocation = _allocation_block(report)
    entry.update(
        generated=len(synthetic),
        fulfilment=fulfilment,
        acceptance_rate=report.get("acceptance_rate"),
        seconds=round(seconds, 3),
        stopped_reason=report.get("stopped_reason"),
        throttled_rounds=report.get("throttled_rounds"),
        warnings=list(report.get("warnings") or []),
        in_scenario_rate=in_scenario,
        allocation_strata=allocation,
        strata_columns=list(strata),
        strata_mix_tvd=tvd,
        strata_rounding_tvd=rounding,
        strata_fidelity=strata_fidelity,
        rebalancing_flag=rebalancing_flagged(tvd, rounding, strata_tvd_flag),
        combinations=combos,
        coverage=coverage,
        fidelity=fidelity,
        fidelity_components=None if quality is None else quality["components"],
        kpi_means=_kpi_means(real_subset, synthetic, table),
        score=score if score is not None else 0.0,
    )
    return entry


def _not_evaluated(scenario: Scenario, rows: int, reason: str) -> dict[str, Any]:
    return {"name": scenario.name, "title": scenario.title, "category": scenario.category,
            "description": scenario.description, "status": "not_evaluated", "reason": reason,
            "requested": rows, "score": None}


def evaluate_scenario_coverage(
    generator: Any,
    reference: pd.DataFrame,
    table: str,
    model_id: str | None = None,
    rows_per_scenario: int | None = None,
    seed: int | None = None,
    *,
    scenarios: list[str] | None = None,
    time_budget_seconds: float | None = None,
    max_attempts: int | None = None,
    total_budget_seconds: float | None = None,
    strata: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Generate every scenario of ``table`` and score coverage and fidelity (see module docstring).

    ``generator`` needs ``.generate(GenerationRequest) -> GenerationResult``; ``reference`` is
    the fully derived real data used for scenario support and comparison. ``scenarios``
    restricts the run to the named scenarios. ``time_budget_seconds`` / ``max_attempts`` are
    passed to every request (``None`` = generator defaults); ``total_budget_seconds`` caps the
    whole run: each request gets at most the remaining budget and scenarios not started in
    time are reported as ``not_evaluated``. ``strata`` overrides the allocation strata columns.
    """
    start = time.perf_counter()
    seed = get_seed() if seed is None else int(seed)
    settings = section("evaluation") or {}
    if rows_per_scenario is None:
        rows_per_scenario = int(settings.get("scenario_rows") or DEFAULT_ROWS_PER_SCENARIO)
    strata_flag = float(settings.get("strata_tvd_flag") or DEFAULT_STRATA_TVD_FLAG)
    reference_df = ensure_derived(reference, table).reset_index(drop=True)
    selected = [s for s in scenarios_for(table) if scenarios is None or s.name in scenarios]
    entries = []
    for index, scenario in enumerate(selected):
        budget = time_budget_seconds
        if total_budget_seconds is not None:
            remaining = float(total_budget_seconds) - (time.perf_counter() - start)
            if remaining <= 1.0:
                entries.append(_not_evaluated(scenario, rows_per_scenario,
                                              f"coverage time budget of {total_budget_seconds:g}s exhausted"))
                continue
            budget = remaining if budget is None else min(float(budget), remaining)
        entries.append(evaluate_scenario(generator, scenario, reference_df, table, model_id, rows_per_scenario,
                                         seed + index, time_budget_seconds=budget, max_attempts=max_attempts,
                                         strata=strata, strata_tvd_flag=strata_flag))
    evaluated = [e for e in entries if e["status"] != "not_evaluated"]
    succeeded = [e for e in evaluated if "error" not in e]
    result = {
        "name": "scenario_coverage",
        "description": "Ability to generate every catalogue scenario (fulfilment, strata fidelity) "
                       "and fidelity of each scenario subset to the matching real subset.",
        "table": table,
        "model_id": model_id,
        "score": mean_or_none(e["score"] for e in evaluated),
        "rows_per_scenario": rows_per_scenario,
        "time_budget_seconds": time_budget_seconds,
        "max_attempts": max_attempts,
        "total_budget_seconds": total_budget_seconds,
        "n_scenarios": len(entries),
        "n_evaluated": len(evaluated),
        "n_succeeded": len(succeeded),
        "complete": len(evaluated) == len(entries),
        "failed_scenarios": [e["name"] for e in evaluated if "error" in e],
        "not_evaluated_scenarios": [e["name"] for e in entries if e["status"] == "not_evaluated"],
        "budget_stopped_scenarios": [e["name"] for e in succeeded if e.get("stopped_reason")],
        "throttled_scenarios": [e["name"] for e in succeeded if e.get("throttled_rounds")],
        "rebalanced_scenarios": [e["name"] for e in succeeded if e.get("rebalancing_flag")],
        "mean_fulfilment": mean_or_none(e.get("fulfilment") for e in succeeded),
        "mean_strata_fidelity": mean_or_none(e.get("strata_fidelity") for e in succeeded),
        "mean_acceptance_rate": mean_or_none(e.get("acceptance_rate") for e in succeeded),
        "mean_fidelity": mean_or_none(e.get("fidelity") for e in succeeded),
        "mean_combination_coverage": mean_or_none((e.get("combinations") or {}).get("coverage") for e in succeeded),
        "scenarios": entries,
        "metric_descriptions": METRIC_DESCRIPTIONS,
        "seconds": round(time.perf_counter() - start, 3),
    }
    return to_jsonable(result)


__all__ = [
    "combination_coverage",
    "evaluate_scenario",
    "evaluate_scenario_coverage",
    "forced_exclusions",
    "strata_columns",
    "strata_mix_tvd",
]
