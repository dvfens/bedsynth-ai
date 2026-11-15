"""Operational occupancy and scheduling patterns (DESIGN.md §9.4).

Generic fidelity metrics can be high while the patterns a bed manager relies on are wrong
(e.g. the right overall LOS distribution but the wrong LOS per admission source). This
module compares those patterns directly. Every check returns::

    {name, description, group, metric, value, criterion, threshold, threshold_used, passed,
     score, real, synthetic, details, strata_forced, real_reference, within_real_noise,
     conditions_ok, conditions_waived}

* ``metric`` / ``value`` - the comparison statistic (e.g. mean KS over wards) and its value;
* ``criterion`` / ``threshold`` - the design pass rule (``value <= threshold`` plus, for some
  checks, a structural condition: every group reproduced, same correlation sign, profile
  correlation >= 0.5); ``passed`` is evaluated with ``threshold_used`` (see calibration);
* ``score`` in [0, 1] - agreement on a common scale (1 - distance for KS/TVD,
  1 - |Δ| / scale for shares, 1 - relative error for medians/means, sign agreement and
  |Δρ| for relationships, correlation and WAPE for calendar profiles);
* ``real`` / ``synthetic`` - the statistic per group (ward, season, band, ...).

Grouped checks score each group separately and average (groups weighted equally); a
group with < 10 real rows is skipped, a group the synthetic data (almost) never produces
(< 5 rows) scores 0 and fails the check. A pooled check whose statistic is defined on the
real data but undefined on the synthetic data (e.g. fewer than 5 synthetic winter rows for
the seasonal contrast) also scores 0 and fails; only a statistic that is undefined on the
real data leaves a check unscored (``passed`` and ``score`` None, outside the family score).

**Real-noise calibration** (:func:`calibrate_operational`). Even real data does not match the
training split exactly: a 2,000-row holdout sample failed ``weekend_weekday_discharge_ratio``
in 20 % of draws. The calibration re-splits the real data (train + holdout) ``reps`` times at
random into a pseudo training split and ``n`` real rows - ``n`` being the *evaluated* synthetic
size - and stores per check the real-vs-real distribution of ``value``. With a calibration, a
check passes when
``value <= threshold_used = max(threshold, noise_limit)``, where ``noise_limit`` is the larger
of the largest real value and ``mean + 3 sd`` of the real values
(``settings.evaluation.noise_sd_multiplier``), and its structural condition is waived when real
samples fail it too; ``within_real_noise`` says whether the synthetic value is inside the limit
and ``real_reference`` keeps the real mean, p95, noise limit and pass rates. A per-check
quantile is not enough: with ~17 checks, a 99th-percentile limit still fails a real sample on
some check in ~15 % of draws, while the brief asks that a real sample of the evaluated size
passes. ``real_pass_rate_cv`` (two-fold cross-validation over the draws) estimates how often a
fresh real sample passes *every* check.

**Strata-forced checks.** The generator allocates rows with the reference frequencies of its
allocation strata (``generation.allocation_strata``: ``ward_id x occupancy_level`` and
``ward_id x admission_source x delay_reason``). A check whose statistic depends only on
those columns (occupancy band mix per ward, admission-source mix per ward) then measures the
generator's copy of the training frequencies, not the model, and real data cannot match it
(sampling noise). Given ``forced_columns``, such checks are tagged ``strata_forced``, reported
as *system checks* and left out of the family score, the pass count and ``failed_checks``.

The family ``score`` is the mean score of the (non-forced) checks.
"""
from __future__ import annotations

import inspect
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd

from bedsynth.config import get_seed
from bedsynth.evaluation.scoring import (
    abs_diff_score,
    clip01,
    complement_score,
    correlation_agreement,
    finite_or_none,
    mean_or_none,
    relative_error,
    relative_error_score,
    to_jsonable,
)
from bedsynth.evaluation.statistical import ensure_derived, ks_statistic, tv_distance
from bedsynth.schema import (
    ADMISSION_SOURCES,
    DAYS_OF_WEEK,
    DELAY_REASONS,
    DISCHARGE_DESTINATIONS,
    MONTHS,
    OCCUPANCY_LEVELS,
    PATIENT_STAYS,
    SEASONS,
    WARD_DAILY,
    WARD_IDS,
    WARDS,
    WEEKDAYS,
    WEEKEND_DAYS,
)

logger = logging.getLogger(__name__)

MIN_REAL_GROUP_ROWS = 10          # groups with fewer real rows are too noisy to compare
MIN_SYNTHETIC_GROUP_ROWS = 5      # fewer synthetic rows = the model does not reproduce the group
SHARE_SCALE = 0.20                # absolute difference of a share that scores 0
OCCUPANCY_SCALE = 0.10            # absolute difference of a mean occupancy rate that scores 0
HOURS: tuple[int, ...] = tuple(range(24))
ELECTIVE_WARDS: tuple[str, ...] = tuple(w for w in WARD_IDS if WARDS[w].accepts_electives)
NON_NONE_DELAY_REASONS: tuple[str, ...] = tuple(r for r in DELAY_REASONS if r != "none")
ELECTIVE_ADMISSION_HOURS: tuple[int, int] = (6, 14)   # elective admission window 06:00-14:59
DEFAULT_CALIBRATION_REPS = 50
DEFAULT_NOISE_SD_MULTIPLIER = 3.0     # noise limit = max(largest real value, mean + 3 sd of the real values)


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------
@dataclass
class CheckOutcome:
    """Comparison result of one check (name, description and group are added on registration).

    ``threshold`` is the design threshold of ``value`` (pass: ``value <= threshold``);
    ``conditions_ok`` is the outcome of the structural part of the pass rule (every group
    reproduced, same correlation sign, ...; ``None`` = no such condition), so the runner can
    re-evaluate ``passed`` with a calibrated threshold.
    """

    metric: str
    value: float | None
    criterion: str
    passed: bool | None
    score: float | None
    real: Any = None
    synthetic: Any = None
    details: dict[str, Any] = field(default_factory=dict)
    threshold: float | None = None
    conditions_ok: bool | None = None


@dataclass
class GroupOutcome:
    """Comparison of one group (e.g. one ward) inside a grouped check."""

    real: Any
    synthetic: Any
    value: float | None
    score: float | None
    extra: dict[str, Any] = field(default_factory=dict)


CheckRunner = Callable[..., dict[str, Any]]
_CHECKS: dict[str, list[CheckRunner]] = {WARD_DAILY: [], PATIENT_STAYS: []}


def _calibrated_threshold(threshold: float | None, reference: dict[str, Any] | None) -> float | None:
    """``max(threshold, real noise limit)`` (the design threshold when no calibration is available)."""
    limit = finite_or_none((reference or {}).get("noise_limit"))
    if threshold is None or limit is None:
        return threshold
    return max(float(threshold), limit)


def _apply_threshold(outcome: CheckOutcome, threshold_used: float | None, waive_conditions: bool = False) -> bool | None:
    """Pass/fail of ``outcome`` with ``threshold_used``.

    Structural failures (missing group, opposite correlation sign, flat profile correlation) fail
    the check unless ``waive_conditions``: the calibration found that real samples of the
    evaluated size fail that condition too often for it to be meaningful.
    """
    if outcome.value is None or threshold_used is None:
        return outcome.passed
    return bool((waive_conditions or outcome.conditions_ok is not False) and outcome.value <= threshold_used)


def _conditions_waived(reference: dict[str, Any] | None) -> bool:
    """Whether the calibration waives the structural part of a check's pass rule."""
    return bool((reference or {}).get("conditions_waived"))


def _check(table: str, name: str, group: str, uses: tuple[str, ...] = ()) -> Callable:
    """Register a check for ``table``; its docstring's first line becomes the description.

    ``uses`` lists the columns that fully determine the check's statistic when it is a pure
    function of categorical columns (used to detect strata-forced checks). The registered
    runner never raises: an exception becomes a check with ``score=None`` and an ``error``
    field, so one broken check cannot hide the others.
    """

    def decorator(func: Callable[[pd.DataFrame, pd.DataFrame], CheckOutcome]) -> Callable:
        description = (inspect.getdoc(func) or name).splitlines()[0]

        def run(real: pd.DataFrame, synthetic: pd.DataFrame, *, forced: frozenset[str] = frozenset(),
                reference: dict[str, Any] | None = None) -> dict[str, Any]:
            forced_check = bool(uses) and set(uses) <= set(forced)
            base = {"name": name, "description": description, "group": group, "uses": list(uses),
                    "strata_forced": forced_check}
            try:
                outcome = func(real, synthetic)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Operational check '%s' failed: %s", name, exc)
                return {**base, "metric": None, "value": None, "criterion": None, "threshold": None,
                        "threshold_used": None, "passed": None, "score": None, "real": None, "synthetic": None,
                        "details": {}, "real_reference": None, "within_real_noise": None,
                        "error": f"{type(exc).__name__}: {exc}"}
            threshold_used = _calibrated_threshold(outcome.threshold, reference)
            waived = _conditions_waived(reference)
            limit = finite_or_none((reference or {}).get("noise_limit"))
            within = None if outcome.value is None or limit is None else bool(outcome.value <= limit)
            return {**base, "metric": outcome.metric, "value": outcome.value, "criterion": outcome.criterion,
                    "threshold": outcome.threshold, "threshold_used": threshold_used,
                    "conditions_ok": outcome.conditions_ok, "conditions_waived": waived,
                    "passed": _apply_threshold(outcome, threshold_used, waived), "passed_design": outcome.passed,
                    "score": clip01(outcome.score), "real": outcome.real, "synthetic": outcome.synthetic,
                    "details": outcome.details, "real_reference": reference, "within_real_noise": within}

        run.check_name = name  # type: ignore[attr-defined]
        run.check_uses = uses  # type: ignore[attr-defined]
        run.check_function = func  # type: ignore[attr-defined]
        _CHECKS[table].append(run)
        return func

    return decorator


# ---------------------------------------------------------------------------
# Small statistics helpers
# ---------------------------------------------------------------------------
def _ratio(numerator: float, denominator: float) -> float | None:
    """``numerator / denominator`` or ``None`` when the denominator is 0 / missing."""
    num, den = finite_or_none(numerator), finite_or_none(denominator)
    if num is None or den is None or den == 0:
        return None
    return num / den


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype(float)


def _summary(series: pd.Series) -> dict[str, float | None]:
    """n, mean, median and 90th percentile of a numeric series."""
    values = _numeric(series).dropna()
    if values.empty:
        return {"n": 0, "mean": None, "p50": None, "p90": None}
    return {"n": int(values.size), "mean": float(values.mean()), "p50": float(values.median()),
            "p90": float(values.quantile(0.9))}


def _shares(series: pd.Series, categories: tuple) -> dict[Any, float]:
    """Share of each category in ``categories`` (zeros included)."""
    counts = series.value_counts(normalize=True)
    return {c: float(counts.get(c, 0.0)) for c in categories}


def _spearman(x: pd.Series, y: pd.Series) -> float | None:
    """Spearman rank correlation (``None`` for < 3 rows or a constant input)."""
    x_values, y_values = _numeric(x), _numeric(y)
    if len(x_values) < 3 or x_values.nunique() < 2 or y_values.nunique() < 2:
        return None
    return finite_or_none(x_values.corr(y_values, method="spearman"))


def _electives(d: pd.DataFrame) -> pd.DataFrame:
    """Elective stays of a patient_stays frame."""
    return d.loc[d["admission_source"] == "elective"]


# ---------------------------------------------------------------------------
# Group evaluators: (real_group, synthetic_group | None) -> GroupOutcome
# ---------------------------------------------------------------------------
GroupEvaluator = Callable[[pd.DataFrame, pd.DataFrame | None], GroupOutcome]


def _scalar_abs(stat: Callable[[pd.DataFrame], Any], scale: float) -> GroupEvaluator:
    """Compare a scalar statistic by absolute difference (``scale`` difference scores 0)."""

    def evaluate(rg: pd.DataFrame, sg: pd.DataFrame | None) -> GroupOutcome:
        r = finite_or_none(stat(rg))
        s = finite_or_none(stat(sg)) if sg is not None else None
        diff = abs(s - r) if r is not None and s is not None else None
        return GroupOutcome(r, s, diff, abs_diff_score(r, s, scale))

    return evaluate


def _scalar_relative(stat: Callable[[pd.DataFrame], Any], floor: float) -> GroupEvaluator:
    """Compare a scalar statistic by relative error (denominator at least ``floor``)."""

    def evaluate(rg: pd.DataFrame, sg: pd.DataFrame | None) -> GroupOutcome:
        r = finite_or_none(stat(rg))
        s = finite_or_none(stat(sg)) if sg is not None else None
        return GroupOutcome(r, s, relative_error(r, s, floor), relative_error_score(r, s, floor))

    return evaluate


def _ks(column: str) -> GroupEvaluator:
    """Compare the distribution of a numeric column by the KS statistic."""

    def evaluate(rg: pd.DataFrame, sg: pd.DataFrame | None) -> GroupOutcome:
        if sg is None:
            return GroupOutcome(_summary(rg[column]), None, None, None)
        ks = ks_statistic(rg[column], sg[column])
        return GroupOutcome(_summary(rg[column]), _summary(sg[column]), ks, complement_score(ks))

    return evaluate


def _tvd(column: str, categories: tuple) -> GroupEvaluator:
    """Compare the category mix of a column by total variation distance."""

    def evaluate(rg: pd.DataFrame, sg: pd.DataFrame | None) -> GroupOutcome:
        if sg is None:
            return GroupOutcome(_shares(rg[column], categories), None, None, None)
        tvd = tv_distance(rg[column], sg[column])
        return GroupOutcome(_shares(rg[column], categories), _shares(sg[column], categories), tvd,
                            complement_score(tvd))

    return evaluate


def _distribution(column: str, floor: float) -> GroupEvaluator:
    """Median and p90 (relative error, denominator >= ``floor``) plus the KS statistic.

    Score = mean of (1 - median error, 1 - p90 error, 1 - KS); the reported value is KS.
    """

    def evaluate(rg: pd.DataFrame, sg: pd.DataFrame | None) -> GroupOutcome:
        real_summary = _summary(rg[column])
        if sg is None:
            return GroupOutcome(real_summary, None, None, None)
        syn_summary = _summary(sg[column])
        ks = ks_statistic(rg[column], sg[column])
        median_error = relative_error(real_summary["p50"], syn_summary["p50"], floor)
        p90_error = relative_error(real_summary["p90"], syn_summary["p90"], floor)
        score = mean_or_none([
            None if median_error is None else clip01(1.0 - median_error),
            None if p90_error is None else clip01(1.0 - p90_error),
            complement_score(ks),
        ])
        extra = {"ks": ks, "median_rel_error": median_error, "p90_rel_error": p90_error}
        return GroupOutcome(real_summary, syn_summary, ks, score, extra)

    return evaluate


# ---------------------------------------------------------------------------
# Outcome builders
# ---------------------------------------------------------------------------
def _by_group(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    by: str,
    evaluate: GroupEvaluator,
    order: tuple,
    *,
    metric: str,
    threshold: float,
    criterion: str,
) -> CheckOutcome:
    """Apply ``evaluate`` per value of ``by``; value/score = means over groups (missing groups score 0)."""
    real_out, syn_out, per_group, scores, values = {}, {}, {}, [], []
    skipped, missing = [], []
    for group in order:
        rg = real.loc[real[by] == group]
        if len(rg) < MIN_REAL_GROUP_ROWS:
            skipped.append(group)
            continue
        sg = synthetic.loc[synthetic[by] == group]
        if len(sg) < MIN_SYNTHETIC_GROUP_ROWS:
            missing.append(group)
            sg = None
        outcome = evaluate(rg, sg)
        group_score = 0.0 if sg is None else outcome.score
        real_out[group], syn_out[group] = outcome.real, outcome.synthetic
        per_group[group] = {"value": outcome.value, "score": group_score, **outcome.extra}
        scores.append(group_score)
        values.append(outcome.value)
    value = mean_or_none(values)
    if missing:  # a group of the real data the model (almost) never produces
        passed: bool | None = False
    else:
        passed = None if value is None else bool(value <= threshold)
    return CheckOutcome(
        metric=metric, value=value, criterion=criterion, passed=passed, score=mean_or_none(scores),
        real=real_out, synthetic=syn_out,
        details={"group_by": by, "by_group": per_group, "skipped_groups": skipped, "missing_groups": missing},
        threshold=threshold, conditions_ok=not missing,
    )


def _pooled(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    evaluate: GroupEvaluator,
    *,
    metric: str,
    threshold: float,
    criterion: str,
) -> CheckOutcome:
    """Apply ``evaluate`` to the whole frames.

    The check fails with score 0 when the pattern exists in the real data but not in the synthetic
    data: fewer than 5 synthetic rows, or a statistic that is defined on the real rows but
    undefined on the synthetic ones (e.g. fewer than 5 synthetic winter rows for the seasonal
    contrast). Only a statistic undefined on the real data leaves the check unscored.
    """
    sg = synthetic if len(synthetic) >= MIN_SYNTHETIC_GROUP_ROWS else None
    if len(real) < MIN_REAL_GROUP_ROWS:
        return CheckOutcome(metric, None, criterion, None, None, details={"note": "too few real rows"},
                            threshold=threshold)
    outcome = evaluate(real, sg)
    if sg is None:  # the pattern exists in the real data but the model does not produce it
        return CheckOutcome(metric, None, criterion, False, 0.0, outcome.real, None,
                            details={"note": f"fewer than {MIN_SYNTHETIC_GROUP_ROWS} synthetic rows"},
                            threshold=threshold, conditions_ok=False)
    if outcome.value is None and evaluate(real, real).value is not None:
        # defined on the real rows (comparing them with themselves gives a value), undefined on the
        # synthetic rows: the model does not produce what the statistic needs - same as a missing pattern
        return CheckOutcome(metric, None, criterion, False, 0.0, outcome.real, outcome.synthetic,
                            details={**outcome.extra,
                                     "note": "statistic undefined on the synthetic data (e.g. too few rows of "
                                             "a group it needs, or a zero denominator)"},
                            threshold=threshold, conditions_ok=False)
    passed = None if outcome.value is None else bool(outcome.value <= threshold)
    return CheckOutcome(metric, outcome.value, criterion, passed, outcome.score, outcome.real, outcome.synthetic,
                        details=dict(outcome.extra), threshold=threshold)


def _profile(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    by: str,
    column: str,
    order: tuple,
    *,
    wape_threshold: float = 0.10,
    corr_threshold: float = 0.5,
    within: str | None = "ward_id",
) -> CheckOutcome:
    """Calendar profile of the mean of ``column`` by ``by`` (e.g. month).

    With ``within`` (default ``ward_id``) the profile is ward-mix adjusted: the mean per
    ``(within, by)`` cell is averaged over groups with the *real* group shares as fixed weights,
    for both frames. Wards differ by an order of magnitude in volume (AMU ~18 emergency
    admissions a day, ELDC < 1), so a pooled monthly mean mostly measured how many AMU rows a
    month happened to contain: even the real holdout split failed the pooled check.
    Shape agreement = Pearson correlation of the two profiles; level agreement = weighted
    absolute percentage error (WAPE, robust to near-zero entries such as Sunday electives).
    Score = mean(max(0, r), max(0, 1 - WAPE)); r is omitted when the real profile is flat.
    """
    if within is not None and within in real.columns and within in synthetic.columns:
        weights = real[within].value_counts(normalize=True)
        r = _adjusted_profile(real, by, column, within, weights, order)
        s = _adjusted_profile(synthetic, by, column, within, weights, order)
    else:
        r = real.groupby(by)[column].mean().reindex(list(order))
        s = synthetic.groupby(by)[column].mean().reindex(list(order))
    present = r.notna()
    r, s = r[present].astype(float), s[present].astype(float).fillna(0.0)
    denominator = float(r.abs().sum())
    if denominator > 0:
        wape = float((s - r).abs().sum()) / denominator
    else:
        wape = 0.0 if float(s.abs().sum()) == 0 else 1.0
    corr = None
    if present.sum() >= 3 and r.std() > 0:
        corr = float(np.corrcoef(r, s)[0, 1]) if s.std() > 0 else 0.0
    score = mean_or_none([max(0.0, 1.0 - wape), None if corr is None else max(0.0, corr)])
    corr_ok = corr is None or corr >= corr_threshold
    passed = bool(wape <= wape_threshold and corr_ok)
    return CheckOutcome(
        metric="wape", value=wape,
        criterion=f"WAPE <= {wape_threshold:.2f} and profile correlation >= {corr_threshold:.1f}",
        passed=passed, score=score, real=r.to_dict(), synthetic=s.to_dict(),
        details={"group_by": by, "adjusted_within": within, "profile_correlation": corr},
        threshold=wape_threshold, conditions_ok=corr_ok,
    )


def _adjusted_profile(frame: pd.DataFrame, by: str, column: str, within: str, weights: pd.Series,
                      order: tuple) -> pd.Series:
    """Mean of ``column`` by ``by``, averaged over ``within`` groups with fixed ``weights``.

    A missing ``(group, by)`` cell is skipped and the weights of that ``by`` value renormalised;
    a ``by`` value without any cell is NaN.
    """
    cells = _numeric(frame[column]).groupby([frame[within], frame[by]]).mean().unstack(within)
    cells = cells.reindex(index=list(order), columns=list(weights.index))
    w = weights.to_numpy(dtype=float)
    values = cells.to_numpy(dtype=float)
    available = ~np.isnan(values)
    total = (np.where(available, values, 0.0) * w).sum(axis=1)
    mass = (available * w).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        profile = np.where(mass > 0, total / np.where(mass > 0, mass, 1.0), np.nan)
    return pd.Series(profile, index=list(order), dtype=float)


def _relationship(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    x: Callable[[pd.DataFrame], pd.Series],
    y: Callable[[pd.DataFrame], pd.Series],
    subset: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
    threshold: float = 0.15,
) -> CheckOutcome:
    """Spearman correlation of two quantities: direction (sign) and strength (|Δρ|) agreement."""
    real_s = subset(real) if subset else real
    syn_s = subset(synthetic) if subset else synthetic
    r_corr = _spearman(x(real_s), y(real_s))
    s_corr = _spearman(x(syn_s), y(syn_s)) if len(syn_s) >= MIN_SYNTHETIC_GROUP_ROWS else None
    agreement = correlation_agreement(r_corr, s_corr)
    criterion = f"same sign and |Δρ| <= {threshold:.2f}"
    if agreement["score"] is None:
        return CheckOutcome("abs_diff_spearman", None, criterion, None, None, real=r_corr, synthetic=s_corr,
                            details={"note": "real correlation undefined (constant input)"}, threshold=threshold)
    passed = bool(agreement["sign_agrees"] and agreement["abs_diff"] <= threshold)
    return CheckOutcome(
        metric="abs_diff_spearman", value=agreement["abs_diff"],
        criterion=criterion, passed=passed, score=agreement["score"],
        real=r_corr, synthetic=agreement["synthetic"],
        details={"sign_agrees": agreement["sign_agrees"], "n_real": len(real_s), "n_synthetic": len(syn_s)},
        threshold=threshold, conditions_ok=agreement["sign_agrees"],
    )


# ---------------------------------------------------------------------------
# ward_daily checks
# ---------------------------------------------------------------------------
@_check(WARD_DAILY, "occupancy_mean_by_ward", "occupancy")
def _occupancy_mean_by_ward(real: pd.DataFrame, syn: pd.DataFrame) -> CheckOutcome:
    """Mean midnight occupancy rate per ward (absolute difference; a 0.10 difference scores 0)."""
    return _by_group(real, syn, "ward_id", _scalar_abs(lambda d: _numeric(d["occupancy_rate"]).mean(), OCCUPANCY_SCALE),
                     WARD_IDS, metric="mean_abs_diff", threshold=0.03,
                     criterion="mean |Δ occupancy| <= 0.03 and every ward reproduced")


@_check(WARD_DAILY, "occupancy_distribution_by_ward", "occupancy")
def _occupancy_distribution_by_ward(real: pd.DataFrame, syn: pd.DataFrame) -> CheckOutcome:
    """Distribution of midnight occupancy rate per ward (two-sample KS statistic)."""
    return _by_group(real, syn, "ward_id", _ks("occupancy_rate"), WARD_IDS, metric="mean_ks", threshold=0.15,
                     criterion="mean KS <= 0.15 and every ward reproduced")


@_check(WARD_DAILY, "occupancy_band_mix_by_ward", "occupancy", uses=("ward_id", "occupancy_level"))
def _occupancy_band_mix_by_ward(real: pd.DataFrame, syn: pd.DataFrame) -> CheckOutcome:
    """Mix of occupancy bands (low / normal / high / critical) per ward (total variation distance)."""
    return _by_group(real, syn, "ward_id", _tvd("occupancy_level", OCCUPANCY_LEVELS), WARD_IDS,
                     metric="mean_tvd", threshold=0.10, criterion="mean TVD <= 0.10 and every ward reproduced")


@_check(WARD_DAILY, "monthly_emergency_profile", "demand")
def _monthly_emergency_profile(real: pd.DataFrame, syn: pd.DataFrame) -> CheckOutcome:
    """Monthly profile of mean emergency admissions per ward-day, ward-mix adjusted (seasonal demand shape and level)."""
    return _profile(real, syn, "month", "admissions_emergency", MONTHS)


@_check(WARD_DAILY, "seasonal_pressure_contrast", "demand")
def _seasonal_pressure_contrast(real: pd.DataFrame, syn: pd.DataFrame) -> CheckOutcome:
    """Winter minus summer share of critical-occupancy ward-days (absolute difference of the contrast).

    The allocation strata fix each ward's overall band mix but not how the bands spread over the
    year, so this check measures whether the model itself puts the pressure in winter.
    """
    def contrast(d: pd.DataFrame) -> float | None:
        winter, summer = d.loc[d["season"] == "winter"], d.loc[d["season"] == "summer"]
        if len(winter) < MIN_SYNTHETIC_GROUP_ROWS or len(summer) < MIN_SYNTHETIC_GROUP_ROWS:
            return None
        return float((winter["occupancy_level"] == "critical").mean() - (summer["occupancy_level"] == "critical").mean())
    return _pooled(real, syn, _scalar_abs(contrast, SHARE_SCALE), metric="abs_diff", threshold=0.05,
                   criterion="|Δ (winter - summer critical share)| <= 0.05")


@_check(WARD_DAILY, "zero_inflated_count_shares", "capacity")
def _zero_inflated_count_shares(real: pd.DataFrame, syn: pd.DataFrame) -> CheckOutcome:
    """Share of ward-days with closed beds, with ED boarders and (on booked days) with cancellations (mean |Δ share|).

    These counts are zero on most ward-days; a smooth marginal turns their point mass at zero
    into frequent small positive values.
    """
    def shares(d: pd.DataFrame) -> dict[str, float | None]:
        booked = d.loc[_numeric(d["elective_scheduled"]) > 0]
        return {
            "beds_closed": float((_numeric(d["beds_closed"]) > 0).mean()) if len(d) else None,
            "emergency_boarders": float((_numeric(d["emergency_boarders"]) > 0).mean()) if len(d) else None,
            "elective_cancelled_when_booked": (float((_numeric(booked["elective_cancelled"]) > 0).mean())
                                               if len(booked) >= MIN_SYNTHETIC_GROUP_ROWS else None),
        }
    criterion = "mean |Δ share| <= 0.05"
    if len(real) < MIN_REAL_GROUP_ROWS:
        return CheckOutcome("mean_abs_diff", None, criterion, None, None, details={"note": "too few real rows"},
                            threshold=0.05)
    real_shares = shares(real)
    if len(syn) < MIN_SYNTHETIC_GROUP_ROWS:
        return CheckOutcome("mean_abs_diff", None, criterion, False, 0.0, real_shares, None,
                            details={"note": f"fewer than {MIN_SYNTHETIC_GROUP_ROWS} synthetic rows"},
                            threshold=0.05, conditions_ok=False)
    syn_shares = shares(syn)
    keys = [k for k in real_shares if real_shares[k] is not None and syn_shares[k] is not None]
    value = mean_or_none([abs(syn_shares[k] - real_shares[k]) for k in keys])
    score = mean_or_none([abs_diff_score(real_shares[k], syn_shares[k], SHARE_SCALE) for k in keys])
    passed = None if value is None else bool(value <= 0.05)
    return CheckOutcome("mean_abs_diff", value, criterion, passed, score, real_shares, syn_shares,
                        details={"compared": keys}, threshold=0.05)


@_check(WARD_DAILY, "weekly_elective_profile", "scheduling")
def _weekly_elective_profile(real: pd.DataFrame, syn: pd.DataFrame) -> CheckOutcome:
    """Day-of-week profile of elective bookings on wards that accept electives (ward-mix adjusted)."""
    def elective(d: pd.DataFrame) -> pd.DataFrame:
        return d.loc[d["ward_id"].isin(ELECTIVE_WARDS)]
    return _profile(elective(real), elective(syn), "day_of_week", "elective_scheduled", DAYS_OF_WEEK)


@_check(WARD_DAILY, "relationship_bed_pressure_cancellations", "scheduling")
def _relationship_bed_pressure_cancellations(real: pd.DataFrame, syn: pd.DataFrame) -> CheckOutcome:
    """Spearman correlation of start-of-day bed pressure (census_start / available beds) with the elective cancellation rate on days with bookings."""
    def pressure(d: pd.DataFrame) -> pd.Series:
        return _numeric(d["census_start"]) / _numeric(d["available_beds"]).where(_numeric(d["available_beds"]) > 0)
    return _relationship(real, syn, pressure, lambda d: d["elective_cancellation_rate"],
                         subset=lambda d: d.loc[_numeric(d["elective_scheduled"]) > 0])


@_check(WARD_DAILY, "weekly_discharge_profile", "patient_flow")
def _weekly_discharge_profile(real: pd.DataFrame, syn: pd.DataFrame) -> CheckOutcome:
    """Day-of-week profile of mean discharges per ward-day (ward-mix adjusted)."""
    return _profile(real, syn, "day_of_week", "discharges", DAYS_OF_WEEK)


@_check(WARD_DAILY, "weekend_weekday_discharge_ratio", "patient_flow")
def _weekend_weekday_discharge_ratio(real: pd.DataFrame, syn: pd.DataFrame) -> CheckOutcome:
    """Mean weekend discharges divided by mean weekday discharges (relative error)."""
    def ratio(d: pd.DataFrame) -> float | None:
        discharges = _numeric(d["discharges"])
        return _ratio(discharges[d["day_of_week"].isin(WEEKEND_DAYS)].mean(),
                      discharges[d["day_of_week"].isin(WEEKDAYS)].mean())
    return _pooled(real, syn, _scalar_relative(ratio, floor=0.05), metric="relative_error", threshold=0.10,
                   criterion="relative error <= 0.10")


@_check(WARD_DAILY, "before_noon_discharge_share", "patient_flow")
def _ward_before_noon_share(real: pd.DataFrame, syn: pd.DataFrame) -> CheckOutcome:
    """Share of discharges completed before noon (absolute difference)."""
    def share(d: pd.DataFrame) -> float | None:
        return _ratio(_numeric(d["discharges_before_noon"]).sum(), _numeric(d["discharges"]).sum())
    return _pooled(real, syn, _scalar_abs(share, SHARE_SCALE), metric="abs_diff", threshold=0.03,
                   criterion="|Δ share| <= 0.03")


@_check(WARD_DAILY, "relationship_census_discharges", "relationships")
def _relationship_census_discharges(real: pd.DataFrame, syn: pd.DataFrame) -> CheckOutcome:
    """Spearman correlation of start-of-day census with discharges (sign agreement and |Δρ|)."""
    return _relationship(real, syn, lambda d: d["census_start"], lambda d: d["discharges"])


@_check(WARD_DAILY, "relationship_occupancy_boarders", "relationships")
def _relationship_occupancy_boarders(real: pd.DataFrame, syn: pd.DataFrame) -> CheckOutcome:
    """Spearman correlation of occupancy rate with emergency patients boarding at midnight."""
    return _relationship(real, syn, lambda d: d["occupancy_rate"], lambda d: d["emergency_boarders"])


@_check(WARD_DAILY, "relationship_delays_occupancy", "relationships")
def _relationship_delays_occupancy(real: pd.DataFrame, syn: pd.DataFrame) -> CheckOutcome:
    """Spearman correlation of delayed-discharge rate with occupancy rate."""
    return _relationship(real, syn, lambda d: d["delayed_discharge_rate"], lambda d: d["occupancy_rate"])


@_check(WARD_DAILY, "relationship_short_staffing_cancellations", "relationships")
def _relationship_short_staffing_cancellations(real: pd.DataFrame, syn: pd.DataFrame) -> CheckOutcome:
    """Spearman correlation of short staffing (0/1) with elective cancellation rate on days with bookings."""
    return _relationship(real, syn, lambda d: (d["staffing_level"] == "short").astype(float),
                         lambda d: d["elective_cancellation_rate"],
                         subset=lambda d: d.loc[_numeric(d["elective_scheduled"]) > 0])


@_check(WARD_DAILY, "relationship_escalation_occupancy", "relationships")
def _relationship_escalation_occupancy(real: pd.DataFrame, syn: pd.DataFrame) -> CheckOutcome:
    """Spearman correlation of escalation beds open with occupancy rate."""
    return _relationship(real, syn, lambda d: d["escalation_beds_open"], lambda d: d["occupancy_rate"])


@_check(WARD_DAILY, "short_staffing_share_by_season", "staffing")
def _short_staffing_share_by_season(real: pd.DataFrame, syn: pd.DataFrame) -> CheckOutcome:
    """Share of short-staffed ward-days per season (absolute difference)."""
    return _by_group(real, syn, "season", _scalar_abs(lambda d: (d["staffing_level"] == "short").mean(), SHARE_SCALE),
                     SEASONS, metric="mean_abs_diff", threshold=0.05,
                     criterion="mean |Δ share| <= 0.05 and every season reproduced")


@_check(WARD_DAILY, "boarding_prevalence_by_occupancy_band", "emergency")
def _boarding_prevalence_by_band(real: pd.DataFrame, syn: pd.DataFrame) -> CheckOutcome:
    """Share of ward-days with emergency patients boarding at midnight, per occupancy band."""
    return _by_group(real, syn, "occupancy_level",
                     _scalar_abs(lambda d: (_numeric(d["emergency_boarders"]) > 0).mean(), SHARE_SCALE),
                     OCCUPANCY_LEVELS, metric="mean_abs_diff", threshold=0.05,
                     criterion="mean |Δ share| <= 0.05 and every band reproduced")


# ---------------------------------------------------------------------------
# patient_stays checks
# ---------------------------------------------------------------------------
@_check(PATIENT_STAYS, "los_by_source", "length_of_stay")
def _los_by_source(real: pd.DataFrame, syn: pd.DataFrame) -> CheckOutcome:
    """Length of stay (days) per admission source: median and p90 (relative error) and KS."""
    return _by_group(real, syn, "admission_source", _distribution("los_days", floor=0.5), ADMISSION_SOURCES,
                     metric="mean_ks", threshold=0.10, criterion="mean KS <= 0.10 and every source reproduced")


@_check(PATIENT_STAYS, "los_by_ward", "length_of_stay")
def _los_by_ward(real: pd.DataFrame, syn: pd.DataFrame) -> CheckOutcome:
    """Length of stay (days) per ward: median and p90 (relative error) and KS."""
    return _by_group(real, syn, "ward_id", _distribution("los_days", floor=0.5), WARD_IDS,
                     metric="mean_ks", threshold=0.10, criterion="mean KS <= 0.10 and every ward reproduced")


@_check(PATIENT_STAYS, "admission_hour_profile_by_source", "patient_flow")
def _admission_hour_profile(real: pd.DataFrame, syn: pd.DataFrame) -> CheckOutcome:
    """Admission hour-of-day distribution per admission source (total variation distance)."""
    return _by_group(real, syn, "admission_source", _tvd("admission_hour", HOURS), ADMISSION_SOURCES,
                     metric="mean_tvd", threshold=0.10, criterion="mean TVD <= 0.10 and every source reproduced")


@_check(PATIENT_STAYS, "discharge_hour_profile", "patient_flow")
def _discharge_hour_profile(real: pd.DataFrame, syn: pd.DataFrame) -> CheckOutcome:
    """Ward departure hour-of-day distribution (total variation distance)."""
    return _pooled(real, syn, _tvd("discharge_hour", HOURS), metric="tvd", threshold=0.10, criterion="TVD <= 0.10")


@_check(PATIENT_STAYS, "before_noon_discharge_share", "patient_flow")
def _stay_before_noon_share(real: pd.DataFrame, syn: pd.DataFrame) -> CheckOutcome:
    """Share of ward departures before noon (absolute difference)."""
    return _pooled(real, syn, _scalar_abs(lambda d: d["discharged_before_noon"].astype(float).mean(), SHARE_SCALE),
                   metric="abs_diff", threshold=0.03, criterion="|Δ share| <= 0.03")


@_check(PATIENT_STAYS, "weekend_discharge_share", "patient_flow")
def _weekend_discharge_share(real: pd.DataFrame, syn: pd.DataFrame) -> CheckOutcome:
    """Share of ward departures on Saturday or Sunday (absolute difference)."""
    return _pooled(real, syn, _scalar_abs(lambda d: d["is_weekend_discharge"].astype(float).mean(), SHARE_SCALE),
                   metric="abs_diff", threshold=0.03, criterion="|Δ share| <= 0.03")


@_check(PATIENT_STAYS, "delay_prevalence_by_destination", "delayed_discharge")
def _delay_prevalence_by_destination(real: pd.DataFrame, syn: pd.DataFrame) -> CheckOutcome:
    """Share of delayed departures per discharge destination (absolute difference)."""
    return _by_group(real, syn, "discharge_destination",
                     _scalar_abs(lambda d: d["is_delayed_discharge"].astype(float).mean(), SHARE_SCALE),
                     DISCHARGE_DESTINATIONS, metric="mean_abs_diff", threshold=0.05,
                     criterion="mean |Δ share| <= 0.05 and every destination reproduced")


@_check(PATIENT_STAYS, "mean_delay_by_reason", "delayed_discharge")
def _mean_delay_by_reason(real: pd.DataFrame, syn: pd.DataFrame) -> CheckOutcome:
    """Mean discharge delay (hours) per delay reason (relative error)."""
    return _by_group(real, syn, "delay_reason",
                     _scalar_relative(lambda d: _numeric(d["discharge_delay_hours"]).mean(), floor=1.0),
                     NON_NONE_DELAY_REASONS, metric="mean_relative_error", threshold=0.20,
                     criterion="mean relative error <= 0.20 and every reason reproduced")


@_check(PATIENT_STAYS, "bed_turnaround_distribution", "turnover")
def _bed_turnaround_distribution(real: pd.DataFrame, syn: pd.DataFrame) -> CheckOutcome:
    """Bed turnaround minutes after departure: median and p90 (relative error) and KS."""
    return _pooled(real, syn, _distribution("bed_turnaround_minutes", floor=10.0), metric="ks", threshold=0.10,
                   criterion="KS <= 0.10")


@_check(PATIENT_STAYS, "elective_lead_time_distribution", "scheduling")
def _elective_lead_time_distribution(real: pd.DataFrame, syn: pd.DataFrame) -> CheckOutcome:
    """Booking lead time (days) of elective admissions: median and p90 (relative error) and KS."""
    return _pooled(_electives(real), _electives(syn), _distribution("booking_lead_days", floor=1.0), metric="ks",
                   threshold=0.10, criterion="KS <= 0.10")


@_check(PATIENT_STAYS, "elective_reschedule_share", "scheduling")
def _elective_reschedule_share(real: pd.DataFrame, syn: pd.DataFrame) -> CheckOutcome:
    """Share of elective admissions rescheduled at least once (relative error)."""
    return _pooled(_electives(real), _electives(syn),
                   _scalar_relative(lambda d: (_numeric(d["times_rescheduled"]) > 0).mean(), floor=0.02),
                   metric="relative_error", threshold=0.25, criterion="relative error <= 0.25")


@_check(PATIENT_STAYS, "relationship_lead_reschedules", "scheduling")
def _relationship_lead_reschedules(real: pd.DataFrame, syn: pd.DataFrame) -> CheckOutcome:
    """Spearman correlation of booking lead time with times rescheduled among elective admissions."""
    return _relationship(real, syn, lambda d: d["booking_lead_days"], lambda d: d["times_rescheduled"],
                         subset=_electives)


@_check(PATIENT_STAYS, "elective_admission_weekday_profile", "scheduling")
def _elective_admission_weekday_profile(real: pd.DataFrame, syn: pd.DataFrame) -> CheckOutcome:
    """Admission day-of-week mix of elective admissions, weekend electives included (total variation distance)."""
    outcome = _pooled(_electives(real), _electives(syn), _tvd("admission_day_of_week", DAYS_OF_WEEK),
                      metric="tvd", threshold=0.05, criterion="TVD <= 0.05")
    for key, frame in (("real", outcome.real), ("synthetic", outcome.synthetic)):
        if isinstance(frame, dict):
            outcome.details[f"weekend_share_{key}"] = float(sum(frame.get(day, 0.0) for day in WEEKEND_DAYS))
    return outcome


@_check(PATIENT_STAYS, "elective_admission_window_share", "scheduling")
def _elective_admission_window_share(real: pd.DataFrame, syn: pd.DataFrame) -> CheckOutcome:
    """Share of elective admissions in the 06:00-14:59 admission window (absolute difference)."""
    low, high = ELECTIVE_ADMISSION_HOURS
    return _pooled(_electives(real), _electives(syn),
                   _scalar_abs(lambda d: _numeric(d["admission_hour"]).between(low, high).mean(), SHARE_SCALE),
                   metric="abs_diff", threshold=0.03, criterion="|Δ share| <= 0.03")


@_check(PATIENT_STAYS, "emergency_bed_wait_distribution", "emergency")
def _emergency_bed_wait_distribution(real: pd.DataFrame, syn: pd.DataFrame) -> CheckOutcome:
    """Bed wait (hours) of emergency admissions: median and p90 (relative error) and KS."""
    def emergencies(d: pd.DataFrame) -> pd.DataFrame:
        return d.loc[d["admission_source"] == "emergency"]
    return _pooled(emergencies(real), emergencies(syn), _distribution("bed_wait_hours", floor=0.25), metric="ks",
                   threshold=0.10, criterion="KS <= 0.10")


@_check(PATIENT_STAYS, "source_mix_by_ward", "admission_mix", uses=("ward_id", "admission_source"))
def _source_mix_by_ward(real: pd.DataFrame, syn: pd.DataFrame) -> CheckOutcome:
    """Admission source mix (elective / emergency / transfer) per ward (total variation distance)."""
    return _by_group(real, syn, "ward_id", _tvd("admission_source", ADMISSION_SOURCES), WARD_IDS,
                     metric="mean_tvd", threshold=0.05, criterion="mean TVD <= 0.05 and every ward reproduced")


@_check(PATIENT_STAYS, "transfer_out_share_by_ward", "admission_mix")
def _transfer_out_share_by_ward(real: pd.DataFrame, syn: pd.DataFrame) -> CheckOutcome:
    """Share of stays ending in an internal transfer out, per ward (absolute difference)."""
    return _by_group(real, syn, "ward_id",
                     _scalar_abs(lambda d: (d["separation_type"] == "transfer_out").mean(), SHARE_SCALE),
                     WARD_IDS, metric="mean_abs_diff", threshold=0.03,
                     criterion="mean |Δ share| <= 0.03 and every ward reproduced")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def check_names(table: str) -> list[str]:
    """Names of the operational checks run for ``table`` (in report order)."""
    return [run.check_name for run in _CHECKS[table]]  # type: ignore[attr-defined]


def strata_forced_checks(table: str, forced_columns: Iterable[str]) -> list[str]:
    """Checks of ``table`` whose statistic is fully determined by ``forced_columns``."""
    forced = set(forced_columns or ())
    return [run.check_name for run in _CHECKS[table]  # type: ignore[attr-defined]
            if run.check_uses and set(run.check_uses) <= forced]  # type: ignore[attr-defined]


@dataclass(frozen=True)
class _Draw:
    """Outcome of one check on one real-vs-real draw."""

    value: float | None
    passed: bool | None
    conditions_ok: bool | None
    threshold: float | None


def _check_values(real_df: pd.DataFrame, syn_df: pd.DataFrame, table: str) -> dict[str, _Draw]:
    """``{check: outcome}`` (value, design pass, structural condition, threshold) for one pair."""
    out = {}
    for run in _CHECKS[table]:
        name = run.check_name  # type: ignore[attr-defined]
        try:
            o = run.check_function(real_df, syn_df)  # type: ignore[attr-defined]
            out[name] = _Draw(finite_or_none(o.value), o.passed, o.conditions_ok, o.threshold)
        except Exception as exc:  # noqa: BLE001 - a broken check has no reference
            logger.debug("Calibration of '%s' failed: %s", name, exc)
            out[name] = _Draw(None, None, None, None)
    return out


def calibrate_operational(
    real_train: pd.DataFrame,
    real_holdout: pd.DataFrame,
    table: str,
    n_rows: int,
    *,
    reps: int = DEFAULT_CALIBRATION_REPS,
    sd_multiplier: float = DEFAULT_NOISE_SD_MULTIPLIER,
    seed: int | None = None,
) -> dict[str, Any]:
    """Real-vs-real noise of every check at the evaluated size ``n_rows``.

    Random-split design: every repetition pools ``real_train`` and ``real_holdout``, re-splits the
    pool at random into a pseudo training split (as large as ``real_train``) and a remainder, and
    draws ``n_rows`` rows from the remainder (without replacement when it is large enough,
    otherwise a bootstrap). Each check compares the draw with the pseudo training split, so the
    distribution covers the sampling noise of *both* sides instead of being anchored on the one
    realised holdout. Per check: mean, sd, p95, max and ``noise_limit = max(max, mean +
    sd_multiplier * sd)`` of ``value``, the share of draws that pass the design rule and the share
    that satisfy its structural condition (same correlation sign, profile correlation >= 0.5,
    every group present); a structural condition failed by any real draw is noise at this size
    and is waived (``conditions_waived``). ``real_pass_rate_cv`` estimates, by two-fold
    cross-validation over the draws (limits from one half, pass rate on the other), the share of
    fresh real samples that pass *every* check. When ``n_rows`` exceeds the remainder, the
    bootstrap adds the remainder's own sampling noise, so the limits err on the lenient side.
    """
    if table not in _CHECKS:
        raise KeyError(f"Unknown table '{table}'")
    start = time.perf_counter()
    seed = get_seed() if seed is None else int(seed)
    train_df = ensure_derived(real_train, table).reset_index(drop=True)
    holdout_df = ensure_derived(real_holdout, table).reset_index(drop=True)
    if holdout_df.empty or n_rows < 1 or reps < 1:
        raise ValueError("calibrate_operational needs a non-empty holdout, n_rows >= 1 and reps >= 1")
    rng = np.random.default_rng(seed)
    pool = pd.concat([train_df, holdout_df], ignore_index=True)
    n_train, n_rows = len(train_df), int(n_rows)
    draws: dict[str, list[_Draw]] = {name: [] for name in check_names(table)}
    for _ in range(int(reps)):
        order = rng.permutation(len(pool))
        pseudo_train = pool.iloc[order[:n_train]].reset_index(drop=True)
        rest = order[n_train:]
        picks = rng.choice(rest, size=n_rows, replace=n_rows > len(rest))
        sample = pool.iloc[picks].reset_index(drop=True)
        for name, outcome in _check_values(pseudo_train, sample, table).items():
            draws[name].append(outcome)
    checks = {name: _noise_summary(outcomes, sd_multiplier) for name, outcomes in draws.items()}
    return to_jsonable({
        "name": "operational_noise_calibration",
        "description": f"Real-vs-real distribution of every operational check value: {reps} random re-splits of the "
                       f"real data, {n_rows} real rows scored against a pseudo training split.",
        "table": table,
        "n": int(n_rows),
        "reps": int(reps),
        "sd_multiplier": sd_multiplier,
        "seed": seed,
        "basis": "random re-splits of the real data (train + holdout): a pseudo training split vs n rows of the remainder",
        "real_pass_rate_cv": _cross_validated_pass_rate(draws, sd_multiplier),
        "checks": checks,
        "seconds": round(time.perf_counter() - start, 3),
    })


def _noise_summary(outcomes: list[_Draw], sd_multiplier: float) -> dict[str, Any]:
    """Mean, sd, p95, max, noise limit, design and structural pass rates of one check's real draws."""
    values = np.array([o.value for o in outcomes if o.value is not None], dtype=float)
    judged = [o.passed for o in outcomes if o.passed is not None]
    structural = [o.conditions_ok is not False for o in outcomes if o.value is not None]
    conditions_rate = float(np.mean(structural)) if structural else None
    summary: dict[str, Any] = {
        "mean": None, "sd": None, "p95": None, "noise_limit": None, "max": None,
        "pass_rate_design": float(np.mean(judged)) if judged else None,
        "conditions_pass_rate": conditions_rate,
        "conditions_waived": bool(conditions_rate is not None and conditions_rate < 1.0),
        "n_defined": int(values.size),
    }
    if values.size:
        sd = float(values.std(ddof=1)) if values.size > 1 else 0.0
        summary.update(mean=float(values.mean()), sd=sd, p95=float(np.quantile(values, 0.95)),
                       max=float(values.max()),
                       noise_limit=max(float(values.max()), float(values.mean()) + sd_multiplier * sd))
    return summary


def _cross_validated_pass_rate(draws: dict[str, list[_Draw]], sd_multiplier: float) -> float | None:
    """Share of real draws passing every check with limits calibrated on the *other* half of the draws."""
    n_draws = max((len(v) for v in draws.values()), default=0)
    if n_draws < 4:
        return None
    halves = (list(range(0, n_draws, 2)), list(range(1, n_draws, 2)))
    passes = []
    for fit, test in (halves, halves[::-1]):
        limits = {name: _noise_summary([outcomes[i] for i in fit], sd_multiplier) for name, outcomes in draws.items()}
        passes += [all(_draw_passes(outcomes[i], limits[name]) for name, outcomes in draws.items()) for i in test]
    return float(np.mean(passes))


def _draw_passes(draw: _Draw, reference: dict[str, Any]) -> bool:
    """Whether one real draw passes a check under the calibrated rule.

    A draw without a value passes unless the check failed it (``passed`` False: the sample lacks
    a group or the rows a statistic needs, as a synthetic sample would fail).
    """
    if draw.value is None or draw.threshold is None:
        return draw.passed is not False
    threshold = _calibrated_threshold(draw.threshold, reference)
    return bool((reference.get("conditions_waived") or draw.conditions_ok is not False) and draw.value <= threshold)


def evaluate_operational(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    table: str,
    *,
    forced_columns: Iterable[str] | None = None,
    calibration: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run every operational-pattern check for ``table``; ``score`` = mean (non-forced) check score.

    ``forced_columns`` - categorical columns whose joint mix the generator imposed (allocation
    strata); checks determined by them are reported as system checks (see module docstring).
    ``calibration`` - output of :func:`calibrate_operational` for the evaluated size: thresholds
    become ``max(design threshold, real noise limit)`` and every check carries its real reference.
    """
    if table not in _CHECKS:
        raise KeyError(f"Unknown table '{table}'")
    start = time.perf_counter()
    real_df = ensure_derived(real, table).reset_index(drop=True)
    syn_df = ensure_derived(synthetic, table).reset_index(drop=True)
    forced = frozenset(forced_columns or ())
    references = (calibration or {}).get("checks") or {}
    checks = [run(real_df, syn_df, forced=forced, reference=references.get(run.check_name))  # type: ignore[attr-defined]
              for run in _CHECKS[table]]

    model_checks = [c for c in checks if not c["strata_forced"]]
    system_checks = [c for c in checks if c["strata_forced"]]
    scored = [c for c in model_checks if c["score"] is not None]
    judged = [c for c in model_checks if c["passed"] is not None]
    noise_judged = [c for c in model_checks if c["within_real_noise"] is not None]
    groups: dict[str, list[float]] = {}
    for check in scored:
        groups.setdefault(check["group"], []).append(check["score"])
    result = {
        "name": "operational_patterns",
        "description": "Agreement of operational occupancy, scheduling and patient-flow patterns "
                       "between synthetic and real data (mean of check scores; strata-forced system checks excluded).",
        "table": table,
        "score": mean_or_none(c["score"] for c in scored),
        "n_checks": len(model_checks),
        "n_scored": len(scored),
        "n_passed": sum(1 for c in judged if c["passed"]),
        "pass_rate": (sum(1 for c in judged if c["passed"]) / len(judged)) if judged else None,
        "group_scores": {group: float(np.mean(values)) for group, values in groups.items()},
        "failed_checks": [c["name"] for c in judged if not c["passed"]],
        "n_within_real_noise": sum(1 for c in noise_judged if c["within_real_noise"]) if noise_judged else None,
        "beyond_real_noise": [c["name"] for c in noise_judged if not c["within_real_noise"]],
        "system_checks": [c["name"] for c in system_checks],
        "forced_columns": sorted(forced),
        "calibration": None if calibration is None else {
            k: calibration.get(k) for k in ("n", "reps", "sd_multiplier", "seed", "basis", "real_pass_rate_cv")},
        "errors": {c["name"]: c["error"] for c in checks if c.get("error")},
        "n_rows": {"real": len(real_df), "synthetic": len(syn_df)},
        "checks": checks,
        "seconds": round(time.perf_counter() - start, 3),
    }
    return to_jsonable(result)


__all__ = [
    "CheckOutcome",
    "calibrate_operational",
    "check_names",
    "evaluate_operational",
    "strata_forced_checks",
]
