"""Optimisation utility: does synthetic data lead to the same decisions? (DESIGN.md §9.7)

Hospital administrators want synthetic data to *test optimisation models*. Three small
but realistic decision problems are solved on real and on synthetic ward-days (each row
is one scenario) and the decisions are compared.

(a) **Elective booking quota (sample average approximation).** For every elective ward
    and weekday choose the integer quota ``q in [0, max_observed + 3]`` maximising the
    sample average of ``admitted(q) - λ · cancelled(q)`` with ``λ = 2``, where
    ``admitted = min(q, F)`` and ``cancelled = max(0, q - F)``, and ``F`` - the beds left
    for electives - is ``max(0, available_beds - (census_start + admissions_emergency +
    transfers_in - discharges - transfers_out))``. This is a newsvendor problem whose
    optimum is the ``1 / (1 + λ)`` quantile of ``F``; ties are broken towards the smaller
    quota. The quota chosen on synthetic data and the quota chosen on the real training
    split are both evaluated **out of sample on the real holdout split**; the relative
    regret is the holdout objective lost by the synthetic decision compared with the real
    decision. ``real_decision_regret_vs_oracle`` (what even the real-trained quota loses
    against the holdout's own optimum) is the real-vs-real floor. Without a holdout the
    regret is computed in sample on the training split (where the real quota is optimal by
    construction).
(b) **Admission-control policy evaluation.** Policies admit booked electives while the
    start-of-day census, net of the ward's expected same-day outflow (estimated from the
    real data and frozen), stays within an occupancy target τ ∈ {0.85, 0.90, 0.95, 1.00} of
    available beds - optionally also within the day-shift nurse-ratio capacity. Each policy
    is simulated on real and on synthetic days (elective admission rate, overflow share,
    mean projected occupancy, objective ``admitted - λ · elective overflow``). The score is
    the *policy-evaluation accuracy* ``1 - mean_p min(1, |obj_syn(p) - obj_real(p)| / |obj_real(p)|)``:
    can the synthetic data predict what each policy achieves? Kendall's τ of the two policy
    rankings and top-1 agreement are reported as diagnostics only: on this data the real
    ranking is the monotone order of the occupancy targets (``real_ranking_monotone``), which
    almost any data reproduces (independent columns reach τ = 0.71), so ranking agreement does
    not discriminate.
(c) **Capacity planning.** Beds needed per ward = 95th percentile of
    ``census_end + emergency_boarders``; compared by relative error.

``score`` = mean of (1 - out-of-sample relative regret), policy-evaluation accuracy and
capacity accuracy (1 - mean relative error).
"""
from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

from bedsynth.evaluation.scoring import clip01, finite_or_none, mean_or_none, to_jsonable
from bedsynth.evaluation.statistical import ensure_derived
from bedsynth.schema import WARD_DAILY, WARD_IDS, WARDS, WEEKDAYS

logger = logging.getLogger(__name__)

CANCELLATION_PENALTY = 2.0                      # λ: cost of a bed cancellation relative to one admission
QUOTA_HEADROOM = 3                              # quotas searched up to max observed bookings + 3
OCCUPANCY_TARGETS: tuple[float, ...] = (0.85, 0.90, 0.95, 1.00)
CAPACITY_QUANTILE = 0.95
MIN_SCENARIOS = 5                               # minimum ward-days per ward x weekday to optimise
OBJECTIVE_FLOOR = 0.05                          # denominator floor of the policy objective relative error
ELECTIVE_WARDS: tuple[str, ...] = tuple(w for w in WARD_IDS if WARDS[w].accepts_electives)

METRIC_DESCRIPTIONS: dict[str, str] = {
    "quota_agreement": "Share of ward x weekday groups where the synthetic-optimal quota equals the real-optimal quota.",
    "relative_regret": "Holdout objective lost by the synthetic-optimal quotas relative to the real-optimal quotas, "
                       "divided by the holdout objective of the real-optimal quotas (summed over groups; out of sample).",
    "in_sample_relative_regret": "The same regret evaluated on the training split (the real quota is optimal there).",
    "real_decision_regret_vs_oracle": "Holdout objective lost by the real-optimal quotas against the holdout optimum "
                                      "(real-vs-real floor of the regret).",
    "objective_accuracy": "1 - mean over policies of min(1, |synthetic objective - real objective| / |real objective|).",
    "kendall_tau": "Kendall rank correlation between the policy rankings on real and on synthetic data (diagnostic).",
    "top1_agreement": "Whether the best policy on synthetic data is the best policy on real data (diagnostic).",
    "kpi_mean_abs_diff": "Mean |real - synthetic| over the policy KPI table (rates and occupancy).",
    "capacity_relative_error": "Mean relative error of the per-ward 95th percentile of census_end + emergency_boarders.",
}


# ---------------------------------------------------------------------------
# (a) Elective booking quota - sample average approximation
# ---------------------------------------------------------------------------
def projected_census_without_electives(df: pd.DataFrame) -> pd.Series:
    """End-of-day census implied by the day's non-elective flows."""
    return (df["census_start"] + df["admissions_emergency"] + df["transfers_in"]
            - df["discharges"] - df["transfers_out"]).astype(float)


def free_beds_for_electives(df: pd.DataFrame) -> pd.Series:
    """Beds left for elective admissions in each scenario (ward-day), never negative."""
    return (df["available_beds"].astype(float) - projected_census_without_electives(df)).clip(lower=0.0)


def quota_objective(free_beds: np.ndarray, quotas: np.ndarray, penalty: float = CANCELLATION_PENALTY) -> np.ndarray:
    """Sample-average objective ``mean(min(q, F) - penalty · max(0, q - F))`` for each quota."""
    free = np.asarray(free_beds, dtype=float)[None, :]
    q = np.asarray(quotas, dtype=float)[:, None]
    admitted = np.minimum(q, free)
    cancelled = np.maximum(0.0, q - free)
    return (admitted - penalty * cancelled).mean(axis=1)


def optimal_quota(free_beds: np.ndarray, max_quota: int, penalty: float = CANCELLATION_PENALTY) -> tuple[int, np.ndarray]:
    """SAA-optimal integer quota in ``[0, max_quota]`` (smallest maximiser) and the objective curve."""
    quotas = np.arange(int(max_quota) + 1)
    objective = quota_objective(free_beds, quotas, penalty)
    best = int(np.flatnonzero(np.isclose(objective, objective.max()))[0])
    return best, objective


def _relative(lost: float, base: float) -> float:
    """``lost / base`` with a defined value when the base objective is not positive."""
    if base > 0:
        return lost / base
    return 0.0 if lost <= 0 else 1.0


def _quota_group(real_g: pd.DataFrame, decision_g: pd.DataFrame, holdout_g: pd.DataFrame | None,
                 penalty: float) -> dict[str, Any]:
    """Quotas of one ward x weekday from the real and the synthetic scenarios, with their objectives."""
    observed_max = max(real_g["elective_scheduled"].max(),
                       decision_g["elective_scheduled"].max() if len(decision_g) else 0)
    max_quota = int(observed_max) + QUOTA_HEADROOM
    q_real, real_curve = optimal_quota(free_beds_for_electives(real_g).to_numpy(), max_quota, penalty)
    missing = len(decision_g) < MIN_SCENARIOS
    # Without synthetic scenarios the decision falls back to booking nothing.
    q_syn = 0 if missing else optimal_quota(free_beds_for_electives(decision_g).to_numpy(), max_quota, penalty)[0]
    group = {
        "n_real": len(real_g), "n_synthetic": len(decision_g), "quota_real": q_real, "quota_synthetic": q_syn,
        "max_quota": max_quota, "missing_in_synthetic": missing,
        "objective_real_optimal": float(real_curve[q_real]),
        "objective_real_at_synthetic_quota": float(real_curve[q_syn]),
        "in_sample_regret": float(real_curve[q_real] - real_curve[q_syn]),
    }
    if holdout_g is not None and len(holdout_g) >= MIN_SCENARIOS:
        holdout_curve = quota_objective(free_beds_for_electives(holdout_g).to_numpy(), np.arange(max_quota + 1), penalty)
        group.update(
            n_holdout=len(holdout_g),
            holdout_objective_real_quota=float(holdout_curve[q_real]),
            holdout_objective_synthetic_quota=float(holdout_curve[q_syn]),
            holdout_objective_oracle=float(holdout_curve.max()),
            regret=float(holdout_curve[q_real] - holdout_curve[q_syn]),
        )
    return group


def evaluate_booking_quotas(real: pd.DataFrame, synthetic: pd.DataFrame,
                            penalty: float = CANCELLATION_PENALTY, *,
                            holdout: pd.DataFrame | None = None) -> dict[str, Any]:
    """Solve the quota problem on real and synthetic data per elective ward x weekday and compare.

    With ``holdout`` both decisions are evaluated out of sample on the holdout's scenarios
    (groups with fewer than :data:`MIN_SCENARIOS` holdout ward-days are skipped for the
    regret); without it the regret is computed in sample on ``real``.
    """
    groups = []
    for ward_id in ELECTIVE_WARDS:
        for day in WEEKDAYS:
            real_g = real.loc[(real["ward_id"] == ward_id) & (real["day_of_week"] == day)]
            if len(real_g) < MIN_SCENARIOS:
                continue
            syn_g = synthetic.loc[(synthetic["ward_id"] == ward_id) & (synthetic["day_of_week"] == day)]
            holdout_g = None if holdout is None else holdout.loc[
                (holdout["ward_id"] == ward_id) & (holdout["day_of_week"] == day)]
            groups.append({"ward_id": ward_id, "day_of_week": day, **_quota_group(real_g, syn_g, holdout_g, penalty)})
    if not groups:
        return {"name": "booking_quota_saa", "score": None, "note": "no elective ward x weekday with enough real scenarios"}
    in_sample = _relative(sum(g["in_sample_regret"] for g in groups), sum(g["objective_real_optimal"] for g in groups))
    oos = [g for g in groups if "regret" in g]
    if oos:
        real_objective = sum(g["holdout_objective_real_quota"] for g in oos)
        relative_regret = _relative(sum(g["regret"] for g in oos), real_objective)
        oracle = sum(g["holdout_objective_oracle"] for g in oos)
        floor = _relative(oracle - real_objective, oracle)
        basis = "out_of_sample_holdout"
    else:
        relative_regret, floor, basis = in_sample, None, "in_sample_training_split"
    return {
        "name": "booking_quota_saa",
        "description": "Elective booking quota per ward x weekday by sample average approximation "
                       f"(objective admitted - {penalty:g} x cancellations); synthetic and real quotas evaluated "
                       + ("out of sample on the real holdout." if oos else "on the real training split."),
        "penalty": penalty,
        "regret_basis": basis,
        "n_groups": len(groups),
        "n_groups_out_of_sample": len(oos),
        "quota_agreement": float(np.mean([g["quota_real"] == g["quota_synthetic"] for g in groups])),
        "mean_abs_quota_diff": float(np.mean([abs(g["quota_real"] - g["quota_synthetic"]) for g in groups])),
        "relative_regret": relative_regret,
        "in_sample_relative_regret": in_sample,
        "real_decision_regret_vs_oracle": floor,
        "score": clip01(1.0 - max(0.0, relative_regret)),
        "groups": groups,
    }


# ---------------------------------------------------------------------------
# (b) Admission-control policy ranking
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class AdmissionPolicy:
    """Admit booked electives while projected census stays within ``target`` x available beds."""

    target: float
    nurse_cap: bool

    @property
    def name(self) -> str:
        return f"target_{round(self.target * 100)}" + ("_nurse_cap" if self.nurse_cap else "")


POLICIES: tuple[AdmissionPolicy, ...] = tuple(
    AdmissionPolicy(target, cap) for target in OCCUPANCY_TARGETS for cap in (False, True)
)


def expected_net_outflow(reference: pd.DataFrame) -> dict[str, float]:
    """Mean same-day net outflow per ward (discharges + transfers out - emergency - transfers in)."""
    net = (reference["discharges"] + reference["transfers_out"]
           - reference["admissions_emergency"] - reference["transfers_in"]).astype(float)
    return net.groupby(reference["ward_id"]).mean().to_dict()


def simulate_policy(df: pd.DataFrame, policy: AdmissionPolicy, net_outflow: dict[str, float],
                    penalty: float = CANCELLATION_PENALTY) -> dict[str, float | None]:
    """Apply ``policy`` to every ward-day of ``df`` and return its KPIs."""
    if df.empty:
        return {"elective_admission_rate": None, "overflow_share": None, "mean_projected_occupancy": None, "objective": None}
    available = df["available_beds"].astype(float).to_numpy()
    expected_census = (df["census_start"].astype(float) - df["ward_id"].map(net_outflow).fillna(0.0)).to_numpy()
    allowed = np.floor(policy.target * available - expected_census)
    if policy.nurse_cap:
        ratio = df["ward_id"].map({w: info.nurse_ratio_day for w, info in WARDS.items()}).astype(float).to_numpy()
        allowed = np.minimum(allowed, np.floor(df["nurses_day_shift"].astype(float).to_numpy() * ratio - expected_census))
    scheduled = df["elective_scheduled"].astype(float).to_numpy()
    admitted = np.clip(np.minimum(scheduled, allowed), 0.0, None)
    projected = np.clip(projected_census_without_electives(df).to_numpy() + admitted, 0.0, None)
    elective_overflow = np.minimum(admitted, np.clip(projected - available, 0.0, None))
    total_scheduled = scheduled.sum()
    return {
        "elective_admission_rate": float(admitted.sum() / total_scheduled) if total_scheduled > 0 else None,
        "overflow_share": float(np.mean(projected > available)),
        "mean_projected_occupancy": float(np.nanmean(projected / np.where(available > 0, available, np.nan))),
        "objective": float(np.mean(admitted - penalty * elective_overflow)),
    }


def objective_accuracy(real_objectives: list[float | None], synthetic_objectives: list[float | None]) -> float | None:
    """``1 - mean min(1, |syn - real| / max(|real|, floor))`` over policies with a real objective.

    A policy the synthetic data cannot evaluate (no elective activity: objective ``None``)
    counts as a full error.
    """
    errors = []
    for real, synthetic in zip(real_objectives, synthetic_objectives):
        r, s = finite_or_none(real), finite_or_none(synthetic)
        if r is None:
            continue
        errors.append(1.0 if s is None else min(1.0, abs(s - r) / max(abs(r), OBJECTIVE_FLOOR)))
    return None if not errors else clip01(1.0 - float(np.mean(errors)))


def evaluate_policy_evaluation(real: pd.DataFrame, synthetic: pd.DataFrame,
                               penalty: float = CANCELLATION_PENALTY) -> dict[str, Any]:
    """KPI table of every policy on real and synthetic data, objective accuracy and ranking diagnostics."""
    real_e = real.loc[real["ward_id"].isin(ELECTIVE_WARDS)]
    syn_e = synthetic.loc[synthetic["ward_id"].isin(ELECTIVE_WARDS)]
    net_outflow = expected_net_outflow(real_e)
    table, real_obj, syn_obj, kpi_diffs = [], [], [], []
    for policy in POLICIES:
        real_kpi = simulate_policy(real_e, policy, net_outflow, penalty)
        syn_kpi = simulate_policy(syn_e, policy, net_outflow, penalty)
        table.append({"policy": policy.name, "target": policy.target, "nurse_cap": policy.nurse_cap,
                      "real": real_kpi, "synthetic": syn_kpi})
        real_obj.append(real_kpi["objective"])
        syn_obj.append(syn_kpi["objective"] if syn_kpi["elective_admission_rate"] is not None else None)
        for kpi in ("elective_admission_rate", "overflow_share", "mean_projected_occupancy"):
            r, s = finite_or_none(real_kpi[kpi]), finite_or_none(syn_kpi[kpi])
            if r is not None and s is not None:
                kpi_diffs.append(abs(r - s))

    tau = None
    if all(v is not None for v in real_obj + syn_obj):
        # Rounding treats near-identical objectives as ties instead of noise-driven rank flips.
        tau = finite_or_none(kendalltau(np.round(real_obj, 4), np.round(syn_obj, 4)).statistic)
    real_ranking, syn_ranking = _rank_policies(real_obj), _rank_policies(syn_obj)
    accuracy = objective_accuracy(real_obj, syn_obj)
    return {
        "name": "policy_evaluation",
        "description": "Elective admission-control policies (occupancy target, optional nurse-ratio cap) simulated "
                       "on real vs synthetic ward-days; score = accuracy of the predicted policy objectives.",
        "n_policies": len(POLICIES),
        "kpis": table,
        "objective_accuracy": accuracy,
        "best_policy": {"real": real_ranking[0], "synthetic": syn_ranking[0]},
        "top1_agreement": bool(any(v is not None for v in syn_obj) and real_ranking[0] == syn_ranking[0]),
        "ranking": {"real": real_ranking, "synthetic": syn_ranking},
        "real_ranking_monotone": real_ranking == _monotone_ranking(),
        "kendall_tau": tau,
        "kpi_mean_abs_diff": mean_or_none(kpi_diffs),
        "score": accuracy,
    }


def evaluate_policy_ranking(real: pd.DataFrame, synthetic: pd.DataFrame,
                            penalty: float = CANCELLATION_PENALTY) -> dict[str, Any]:
    """Backward-compatible alias of :func:`evaluate_policy_evaluation`."""
    return evaluate_policy_evaluation(real, synthetic, penalty)


def _monotone_ranking() -> list[str]:
    """Policies ordered by decreasing occupancy target, uncapped before nurse-capped."""
    ordered = sorted(POLICIES, key=lambda p: (-p.target, p.nurse_cap))
    return [p.name for p in ordered]


def _rank_policies(objectives: list[float | None]) -> list[str]:
    """Policy names from best to worst objective (missing objectives last, ties in definition order)."""
    keyed = [(-math.inf if value is None else value, policy.name) for value, policy in zip(objectives, POLICIES)]
    return [name for _, name in sorted(keyed, key=lambda item: -item[0])]


# ---------------------------------------------------------------------------
# (c) Capacity planning
# ---------------------------------------------------------------------------
def evaluate_capacity_planning(real: pd.DataFrame, synthetic: pd.DataFrame,
                               quantile: float = CAPACITY_QUANTILE) -> dict[str, Any]:
    """Beds needed per ward (quantile of census_end + emergency_boarders), real vs synthetic."""
    def beds_needed(d: pd.DataFrame) -> float:
        return float((d["census_end"] + d["emergency_boarders"]).astype(float).quantile(quantile))

    wards, errors = [], []
    for ward_id in WARD_IDS:
        real_w = real.loc[real["ward_id"] == ward_id]
        if len(real_w) < MIN_SCENARIOS:
            continue
        syn_w = synthetic.loc[synthetic["ward_id"] == ward_id]
        real_need = beds_needed(real_w)
        syn_need = beds_needed(syn_w) if len(syn_w) else None
        error = 1.0 if syn_need is None else abs(syn_need - real_need) / max(real_need, 1.0)
        errors.append(error)
        wards.append({"ward_id": ward_id, "beds_needed_real": real_need, "beds_needed_synthetic": syn_need,
                      "relative_error": error})
    mean_error = mean_or_none(errors)
    return {
        "name": "capacity_planning",
        "description": f"Beds needed per ward = p{round(quantile * 100)} of census_end + emergency_boarders.",
        "quantile": quantile,
        "wards": wards,
        "mean_relative_error": mean_error,
        "score": None if mean_error is None else clip01(1.0 - mean_error),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def evaluate_optimization_utility(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    *,
    penalty: float = CANCELLATION_PENALTY,
    holdout: pd.DataFrame | None = None,
    quota_data: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Decision agreement between real and synthetic ward_daily data (see module docstring).

    ``holdout`` - real ward-days unseen by the model: booking-quota decisions are evaluated on
    them out of sample (in sample on ``real`` without it). ``quota_data`` - data the
    "synthetic" quota decision is made on (default ``synthetic``); the real-data reference uses a
    training subsample here so that its decision is not made on the holdout it is scored on.
    """
    start = time.perf_counter()
    real_df = ensure_derived(real, WARD_DAILY).reset_index(drop=True)
    syn_df = ensure_derived(synthetic, WARD_DAILY).reset_index(drop=True)
    holdout_df = None if holdout is None or len(holdout) == 0 else ensure_derived(holdout, WARD_DAILY).reset_index(drop=True)
    decision_df = syn_df if quota_data is None else ensure_derived(quota_data, WARD_DAILY).reset_index(drop=True)
    quotas = evaluate_booking_quotas(real_df, decision_df, penalty, holdout=holdout_df)
    policies = evaluate_policy_evaluation(real_df, syn_df, penalty)
    capacity = evaluate_capacity_planning(real_df, syn_df)
    components = {
        "booking_quota": quotas.get("score"),
        "policy_evaluation": policies.get("score"),
        "capacity_planning": capacity.get("score"),
    }
    result = {
        "name": "optimization_utility",
        "description": "Agreement of optimisation decisions made on synthetic vs real ward-days: elective "
                       "booking quotas (out-of-sample regret on the real holdout), admission-policy evaluation "
                       "and capacity plans.",
        "table": WARD_DAILY,
        "score": mean_or_none(components.values()),
        "components": components,
        "booking_quota": quotas,
        "policy_evaluation": policies,
        "capacity_planning": capacity,
        "metric_descriptions": METRIC_DESCRIPTIONS,
        "seconds": round(time.perf_counter() - start, 3),
    }
    return to_jsonable(result)


__all__ = [
    "AdmissionPolicy",
    "POLICIES",
    "evaluate_booking_quotas",
    "evaluate_capacity_planning",
    "evaluate_optimization_utility",
    "evaluate_policy_evaluation",
    "evaluate_policy_ranking",
    "free_beds_for_electives",
    "objective_accuracy",
    "optimal_quota",
    "quota_objective",
    "simulate_policy",
]
