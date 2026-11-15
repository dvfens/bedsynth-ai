"""Tests for the Week-3 evaluation layer (``bedsynth.evaluation``) and ``scripts/03_evaluate.py``.

Synthetic stand-ins are built from the rule-satisfying fixtures:

* identical copy           - should score (near) perfectly on fidelity / patterns / utility,
                             and be flagged by privacy;
* noise-perturbed copy     - numeric train columns get Gaussian noise (marginals shift);
* column-shuffled copy     - each train column permuted independently (marginals kept,
                             relationships and operational identities destroyed);
* independent sample       - same generator, different seed (a "well generalising" model).

The generator and registry are replaced by small fakes so the evaluation can be tested in
isolation from the synthesis layer.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from bedsynth import schema
from bedsynth.data import io
from bedsynth.data.fixtures import make_patient_stays, make_ward_daily
from bedsynth.evaluation import (
    composite_score,
    evaluate_ml_utility,
    evaluate_model,
    evaluate_operational,
    evaluate_optimization_utility,
    evaluate_privacy,
    evaluate_scenario_coverage,
    evaluate_statistical,
    evaluate_validity,
    quick_quality,
    to_jsonable,
)
from bedsynth.evaluation.coverage import rebalancing_flagged, strata_mix_tvd, strata_rounding_tvd
from bedsynth.evaluation.operational import calibrate_operational, strata_forced_checks
from bedsynth.evaluation.optimization import (
    POLICIES,
    evaluate_booking_quotas,
    free_beds_for_electives,
    objective_accuracy,
    optimal_quota,
    quota_objective,
)
from bedsynth.evaluation.references import (
    NULL_REFERENCE_ID,
    REAL_REFERENCE_ID,
    PoolGenerator,
    evaluate_references,
    normalised_scores,
    stratified_marginals,
)
from bedsynth.evaluation.scoring import (
    abs_diff_score,
    clip01,
    correlation_agreement,
    mean_or_none,
    relative_error_score,
)
from bedsynth.evaluation.statistical import RELABELLED_COLUMNS, cramers_v, ks_statistic, tv_distance
from bedsynth.evaluation.utility import classification_metrics, utility_ratios
from bedsynth.scenarios import scenario_mask, scenarios_for
from bedsynth.schema import PATIENT_STAYS, WARD_DAILY
from bedsynth.synthesis.generator import GenerationError, GenerationResult

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TABLES = (WARD_DAILY, PATIENT_STAYS)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def assert_result(result: dict[str, Any], *, score_required: bool = True) -> None:
    """Result is JSON-serialisable (no NaN / numpy), self-describing and scored in [0, 1]."""
    json.dumps(result, allow_nan=False)
    assert result.get("name") and result.get("description")
    score = result["score"]
    if score_required:
        assert score is not None
    if score is not None:
        assert 0.0 <= score <= 1.0


def perturb(df: pd.DataFrame, table: str, seed: int = 0, scale: float = 0.5) -> pd.DataFrame:
    """Noise-perturbed copy: Gaussian noise on numeric train columns, bounds kept, re-derived."""
    rng = np.random.default_rng(seed)
    out = df.copy()
    for spec in schema.column_specs(table, ["train"]):
        if spec.sdtype != "numerical":
            continue
        values = out[spec.name].astype(float)
        noisy = (values + rng.normal(0.0, scale * (values.std() + 1.0), len(values))).clip(
            lower=spec.min_value, upper=spec.max_value)
        out[spec.name] = noisy.round(spec.decimals or 0).astype("int64" if spec.dtype == "int" else float)
    return schema.derive(table, out)


def shuffle_columns(df: pd.DataFrame, table: str, seed: int = 0) -> pd.DataFrame:
    """Column-shuffled copy: each train column permuted independently, then re-derived."""
    rng = np.random.default_rng(seed)
    out = df.copy()
    for column in schema.training_columns(table):
        out[column] = rng.permutation(out[column].to_numpy())
    return schema.derive(table, out)


@pytest.fixture(scope="module")
def data() -> dict[str, dict[str, pd.DataFrame]]:
    """Small real / holdout / independent frames per table (fast, rule-satisfying)."""
    return {
        WARD_DAILY: {
            "real": make_ward_daily(n_days=60, seed=11),
            "holdout": make_ward_daily(n_days=20, seed=12),
            "other": make_ward_daily(n_days=60, seed=13),
        },
        PATIENT_STAYS: {
            "real": make_patient_stays(n=700, seed=21),
            "holdout": make_patient_stays(n=250, seed=22),
            "other": make_patient_stays(n=700, seed=23),
        },
    }


class FakeGenerator:
    """Stand-in for SyntheticDataGenerator: resamples reference rows.

    ``mode='faithful'`` honours scenario conditions/filters; ``mode='scrambled'`` ignores
    them and draws from a column-shuffled copy. ``fail`` lists scenarios that raise.
    """

    def __init__(self, reference: pd.DataFrame, table: str, mode: str = "faithful", fail: tuple[str, ...] = ()):
        self.table = table
        self.reference = reference if mode == "faithful" else shuffle_columns(reference, table, seed=5)
        self.mode = mode
        self.fail = set(fail)
        self.requests: list[Any] = []

    def generate(self, request) -> GenerationResult:
        self.requests.append(request)
        if request.scenario in self.fail:
            raise GenerationError(f"scenario {request.scenario} not supported")
        pool = self.reference
        if self.mode == "faithful" and request.scenario:
            pool = pool.loc[scenario_mask(pool, request.scenario)]
        if pool.empty:
            data = pool.iloc[0:0]
        else:
            data = pool.sample(n=request.num_rows, replace=True, random_state=request.seed)
        data = schema.assign_ids(data.reset_index(drop=True), self.table)
        report = {"acceptance_rate": 0.8, "sampled_rows": len(data), "accepted_rows": len(data), "warnings": []}
        return GenerationResult(data=data, report=report)


# ---------------------------------------------------------------------------
# Scoring primitives and JSON
# ---------------------------------------------------------------------------
def test_to_jsonable_converts_numpy_pandas_and_nan():
    payload = {
        "a": np.float64(0.5), "b": np.int64(3), "c": np.nan, "d": [np.inf, 1.0], "e": np.bool_(True),
        1: pd.Series({"x": 1.0}), "f": (1, 2), "g": pd.NA, "h": np.array([1, 2]),
    }
    out = to_jsonable(payload)
    json.dumps(out, allow_nan=False)
    assert out == {"a": 0.5, "b": 3, "c": None, "d": [None, 1.0], "e": True, "1": {"x": 1.0}, "f": [1, 2],
                   "g": None, "h": [1, 2]}


def test_score_primitives():
    assert clip01(1.7) == 1.0 and clip01(-0.2) == 0.0 and clip01(float("nan")) is None
    assert mean_or_none([None, 1.0, float("nan"), 0.0]) == 0.5 and mean_or_none([]) is None
    assert relative_error_score(10.0, 12.0) == pytest.approx(0.8)
    assert abs_diff_score(0.5, 0.55, scale=0.2) == pytest.approx(0.75)
    same = correlation_agreement(0.6, 0.5)
    assert same["sign_agrees"] and same["score"] == pytest.approx(0.5 + 0.5 * 0.9)
    flipped = correlation_agreement(0.6, -0.5)
    assert not flipped["sign_agrees"] and flipped["score"] < 0.5
    assert correlation_agreement(None, 0.3)["score"] is None


def test_statistical_primitives():
    a = pd.Series(np.arange(100.0))
    assert ks_statistic(a, a) == 0.0
    assert ks_statistic(a, a + 1000) == 1.0
    cats = pd.Series(["x", "y"] * 50)
    assert tv_distance(cats, cats) == 0.0
    assert tv_distance(cats, pd.Series(["z"] * 100)) == 1.0
    assert cramers_v(cats, cats) == pytest.approx(1.0)
    assert cramers_v(cats, pd.Series(["k"] * 100)) is None


def test_composite_score_weights_and_missing_components():
    results = {
        "statistical": {"score": 0.9}, "validity": {"score": 0.8}, "operational": {"score": 0.7},
        "ml_utility": {"score": 0.6}, "optimization": {"score": 0.4}, "privacy": {"score": 1.0},
    }
    out = composite_score(results)
    assert_result(out)
    weights = out["weights"]
    expected = (weights["fidelity"] * 0.9 + weights["validity"] * 0.8 + weights["operational"] * 0.7
                + weights["utility"] * 0.5 + weights["privacy"] * 1.0) / sum(weights.values())
    assert out["components"]["utility"] == pytest.approx(0.5)
    assert out["score"] == pytest.approx(expected, abs=1e-6)

    # Families absent by design are excluded (renormalised); a FAILED family scores 0 and the
    # composite is incomplete - a crash must never raise the score (E10).
    partial = composite_score({"statistical": {"score": 0.9}, "privacy": {"score": None, "error": "boom"}})
    expected_partial = weights["fidelity"] * 0.9 / (weights["fidelity"] + weights["privacy"])
    assert partial["score"] == pytest.approx(expected_partial)
    assert partial["complete"] is False and partial["failed_families"] == ["privacy"]
    assert partial["failed_components"] == ["privacy"]
    assert set(partial["missing_components"]) == {"validity", "operational", "utility"}
    skipped = composite_score({"statistical": {"score": 0.9}, "privacy": {"score": None, "skipped": True}})
    assert skipped["score"] == pytest.approx(0.9) and skipped["complete"] is True


def test_failed_family_never_raises_the_composite():
    healthy = {"statistical": {"score": 0.9}, "validity": {"score": 0.8}, "operational": {"score": 0.7},
               "ml_utility": {"score": 0.3}, "privacy": {"score": 1.0}}
    crashed = {**healthy, "ml_utility": {"name": "ml_utility", "score": None, "error": "ValueError: boom"}}
    assert composite_score(crashed)["score"] < composite_score(healthy)["score"]
    assert composite_score(crashed)["complete"] is False


# ---------------------------------------------------------------------------
# Statistical fidelity and quick quality
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("table", TABLES)
def test_statistical_identical_copy_beats_perturbed_and_shuffled(data, table):
    real = data[table]["real"]
    identical = evaluate_statistical(real, real.copy(), table)
    perturbed = evaluate_statistical(real, perturb(real, table), table)
    shuffled = evaluate_statistical(real, shuffle_columns(real, table), table)
    for result in (identical, perturbed, shuffled):
        assert_result(result)
        assert result["score_source"] == "sdv_quality_report"
    assert identical["score"] > 0.99
    assert perturbed["score"] < identical["score"] - 0.02
    assert shuffled["score"] < identical["score"] - 0.02
    assert identical["summary"]["spearman_mean_abs_diff"] == pytest.approx(0.0, abs=1e-9)
    assert shuffled["summary"]["spearman_mean_abs_diff"] > identical["summary"]["spearman_mean_abs_diff"]
    column = {WARD_DAILY: "census_start", PATIENT_STAYS: "bed_wait_hours"}[table]
    numeric = {c["column"]: c for c in perturbed["numeric_columns"]}
    assert numeric[column]["ks_statistic"] > 0
    assert set(numeric[column]["real"]) == {"mean", "std", "p5", "p50", "p95", "min", "max"}


def test_statistical_ignores_single_column_relabellings(data):
    real = data[WARD_DAILY]["real"]
    result = evaluate_statistical(real, perturb(real, WARD_DAILY), WARD_DAILY)
    compared = {c["column"] for c in result["numeric_columns"] + result["categorical_columns"]}
    assert not compared & set(RELABELLED_COLUMNS[WARD_DAILY])
    assert {"ward_id", "month", "day_of_week"} <= compared
    assert set(RELABELLED_COLUMNS[WARD_DAILY]) <= set(result["sdv"]["excluded_columns"])
    shapes = {s["column"] for s in result["sdv"]["quality"]["column_shapes"]}
    assert "ward_type" not in shapes and "season" not in shapes


def test_statistical_fallback_without_sdv(data):
    real = data[WARD_DAILY]["real"]
    result = evaluate_statistical(real, perturb(real, WARD_DAILY), WARD_DAILY, run_sdv=False)
    assert_result(result)
    assert result["score_source"] == "fallback_column_metrics"
    assert result["score"] == pytest.approx(result["summary"]["fallback_score"])


def test_quick_quality_is_fast_and_discriminates(data):
    real = data[WARD_DAILY]["real"]
    big_real = pd.concat([real] * 11, ignore_index=True).head(5000)
    big_syn = pd.concat([data[WARD_DAILY]["other"]] * 11, ignore_index=True).head(5000)
    start = time.perf_counter()
    result = quick_quality(big_real, big_syn, WARD_DAILY)
    assert time.perf_counter() - start < 2.0
    assert_result(result)
    assert set(result["components"]) == {"column_shapes", "column_pair_trends", "validity"}
    assert result["components"]["validity"] == pytest.approx(1.0)

    shuffled_df = shuffle_columns(real, WARD_DAILY)
    identical = quick_quality(real, real.copy(), WARD_DAILY, include_validity=False)
    shuffled = quick_quality(real, shuffled_df, WARD_DAILY, include_validity=False)
    assert identical["score"] == pytest.approx(1.0)
    # Shuffling keeps train marginals; strong pairs (capacity ~ census) are destroyed.
    assert shuffled["score"] < identical["score"] - 0.03
    assert shuffled["components"]["column_pair_trends"] < 0.95
    assert shuffled["worst_pairs"][0]["abs_diff"] > 0.5
    assert quick_quality(real, shuffled_df, WARD_DAILY)["components"]["validity"] < 0.5


# ---------------------------------------------------------------------------
# Validity
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("table", TABLES)
def test_validity_on_real_rules(data, table):
    real = data[table]["real"]
    valid = evaluate_validity(real, table, basis="raw_model_samples")
    broken = evaluate_validity(shuffle_columns(real, table), table, basis="raw_model_samples")
    assert_result(valid)
    assert_result(broken)
    assert valid["score"] == pytest.approx(1.0)
    assert broken["score"] < 0.9
    assert broken["worst_hard_rules"]
    for rule in broken["hard_rules"] + broken["soft_rules"]:
        assert rule["rule_id"] and rule["name"] and rule["description"]
        assert 0.0 <= rule["violation_rate"] <= 1.0


def test_validity_maps_the_validation_report(data):
    """Score, rule split and worst rules follow bedsynth.validation on crafted violations."""
    from bedsynth import validation

    df = data[WARD_DAILY]["real"].head(10).copy()
    df.loc[[8, 9], "avg_los_days"] = 95.0                                   # WD-H13 (hard): > 90 days
    df.loc[3, "nurses_night_shift"] = df.loc[3, "nurses_day_shift"] + 5     # WD-S01 (soft): night > day
    expected = validation.validate_table(df, WARD_DAILY)
    result = evaluate_validity(df, WARD_DAILY, basis="generator_output")
    assert_result(result)
    assert result["score"] == pytest.approx(0.8) == pytest.approx(expected.n_hard_valid / 10)
    assert result["basis"] == "generator_output" and result["n_hard_valid"] == 8
    hard_ids = [r["rule_id"] for r in result["hard_rules"]]
    assert hard_ids == [rule.rule_id for rule in validation.get_rules(WARD_DAILY, "hard")]
    assert [r["rule_id"] for r in result["hard_rules"] if r["violations"]] == ["WD-H13"]
    assert result["worst_hard_rules"] == [
        {"rule_id": "WD-H13", "name": "metric_ranges", "violation_rate": pytest.approx(0.2)}]
    soft = {r["rule_id"]: r for r in result["soft_rules"]}
    assert set(soft) == {rule.rule_id for rule in validation.get_rules(WARD_DAILY, "soft")}
    assert soft["WD-S01"]["violations"] == expected.violation_counts("soft")["WD-S01"] >= 1
    assert all("example_indices" not in r for r in result["hard_rules"] + result["soft_rules"])
    assert 0.0 <= result["soft_clean_rate"] <= 0.9
    assert result["soft_violation_rates"]["WD-S01"] == pytest.approx(soft["WD-S01"]["violation_rate"])
    assert result["hard_violation_rates"]["WD-H13"] == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# Operational patterns
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("table", TABLES)
def test_operational_identical_copy_is_perfect(data, table):
    real = data[table]["real"]
    result = evaluate_operational(real, real.copy(), table)
    assert_result(result)
    assert result["score"] == pytest.approx(1.0)
    assert result["errors"] == {}
    assert result["n_checks"] >= 12
    for check in result["checks"]:
        assert check["name"] and check["description"] and check["metric"] and check["criterion"]
        assert check["passed"] in (True, None)
        assert check["score"] is None or check["score"] == pytest.approx(1.0)


@pytest.mark.parametrize("table", TABLES)
def test_operational_degrades_for_perturbed_and_shuffled(data, table):
    real = data[table]["real"]
    perturbed = evaluate_operational(real, perturb(real, table), table)
    shuffled = evaluate_operational(real, shuffle_columns(real, table), table)
    for result in (perturbed, shuffled):
        assert_result(result)
        assert result["score"] < 0.95
        assert result["failed_checks"]


def test_operational_missing_ward_scores_zero(data):
    real = data[WARD_DAILY]["real"]
    without_icu = real.loc[real["ward_id"] != "ICU"]
    result = evaluate_operational(real, without_icu, WARD_DAILY)
    check = next(c for c in result["checks"] if c["name"] == "occupancy_mean_by_ward")
    assert check["details"]["missing_groups"] == ["ICU"]
    assert check["details"]["by_group"]["ICU"]["score"] == 0.0
    assert check["passed"] is False


def test_operational_missing_pattern_fails_pooled_check(data):
    real = data[PATIENT_STAYS]["real"]
    no_electives = real.loc[real["admission_source"] != "elective"]
    result = evaluate_operational(real, no_electives, PATIENT_STAYS)
    check = next(c for c in result["checks"] if c["name"] == "elective_lead_time_distribution")
    assert check["score"] == 0.0 and check["passed"] is False
    assert "elective_lead_time_distribution" in result["failed_checks"]


def test_calendar_profiles_are_ward_mix_adjusted(data):
    """Over-representing a high-volume ward in one month must not distort the monthly profile."""
    real = data[WARD_DAILY]["real"]
    extra_amu = real.loc[(real["ward_id"] == "AMU") & (real["month"] == real["month"].min())]
    skewed = pd.concat([real] + [extra_amu] * 4, ignore_index=True)
    checks = {c["name"]: c for c in evaluate_operational(real, skewed, WARD_DAILY)["checks"]}
    monthly = checks["monthly_emergency_profile"]
    assert monthly["details"]["adjusted_within"] == "ward_id"
    assert monthly["value"] == pytest.approx(0.0, abs=1e-9) and monthly["passed"] is True
    from bedsynth.evaluation.operational import _profile

    pooled = _profile(real, skewed, "month", "admissions_emergency", schema.MONTHS, within=None)
    assert pooled.value > 0.0   # without the adjustment the ward mix shift shows up as a profile error


def test_operational_strata_forced_checks_are_system_checks(data):
    """Checks determined by the allocation strata are reported but excluded from the score (E2)."""
    real = data[WARD_DAILY]["real"]
    shuffled = shuffle_columns(real, WARD_DAILY)
    assert strata_forced_checks(WARD_DAILY, ("ward_id", "occupancy_level")) == ["occupancy_band_mix_by_ward"]
    assert strata_forced_checks(PATIENT_STAYS, ("ward_id", "admission_source", "delay_reason")) == ["source_mix_by_ward"]
    plain = evaluate_operational(real, shuffled, WARD_DAILY)
    forced = evaluate_operational(real, shuffled, WARD_DAILY, forced_columns=("ward_id", "occupancy_level"))
    band = next(c for c in forced["checks"] if c["name"] == "occupancy_band_mix_by_ward")
    assert band["strata_forced"] and forced["system_checks"] == ["occupancy_band_mix_by_ward"]
    assert forced["n_checks"] == plain["n_checks"] - 1
    assert "occupancy_band_mix_by_ward" not in forced["failed_checks"]
    model_scores = [c["score"] for c in forced["checks"] if not c["strata_forced"] and c["score"] is not None]
    assert forced["score"] == pytest.approx(float(np.mean(model_scores)))


def test_scheduling_checks_cover_reschedules_and_admission_window(data):
    real = data[PATIENT_STAYS]["real"]
    names = {c["name"] for c in evaluate_operational(real, real.copy(), PATIENT_STAYS)["checks"]}
    assert {"elective_reschedule_share", "relationship_lead_reschedules", "elective_admission_weekday_profile",
            "elective_admission_window_share"} <= names
    late = real.copy()
    electives = late["admission_source"] == "elective"
    late.loc[electives, "admission_hour"] = 20                       # every elective admitted at 20:00
    late = schema.derive(PATIENT_STAYS, late)
    check = next(c for c in evaluate_operational(real, late, PATIENT_STAYS)["checks"]
                 if c["name"] == "elective_admission_window_share")
    assert check["passed"] is False and check["score"] < 0.5
    ward_names = {c["name"] for c in evaluate_operational(data[WARD_DAILY]["real"], data[WARD_DAILY]["real"],
                                                          WARD_DAILY)["checks"]}
    assert "relationship_bed_pressure_cancellations" in ward_names


def test_seasonal_contrast_and_zero_inflated_count_checks():
    """S10 / S7: the seasonal spread of critical days and the zero shares of mostly-zero counts are checked."""
    real = make_ward_daily(n_days=365, seed=11)
    real.loc[real["season"] == "winter", "occupancy_level"] = "critical"   # winter pressure (labels only)
    same = {c["name"]: c for c in evaluate_operational(real, real.copy(), WARD_DAILY)["checks"]}
    assert same["seasonal_pressure_contrast"]["passed"] is True and same["seasonal_pressure_contrast"]["value"] == 0
    assert same["seasonal_pressure_contrast"]["real"] > 0.5
    assert same["zero_inflated_count_shares"]["passed"] is True and same["zero_inflated_count_shares"]["value"] == 0

    flat = real.copy()                                   # the same band mix, spread evenly over the year
    flat["occupancy_level"] = flat["occupancy_level"].sample(frac=1.0, random_state=0).to_numpy()
    seasonal = next(c for c in evaluate_operational(real, flat, WARD_DAILY)["checks"]
                    if c["name"] == "seasonal_pressure_contrast")
    assert seasonal["passed"] is False and abs(seasonal["synthetic"]) < 0.2 and seasonal["score"] < 0.5

    smeared = real.copy()                                # small positive counts on every day
    smeared["beds_closed"] = 1
    smeared["emergency_boarders"] = 1
    zero = next(c for c in evaluate_operational(real, smeared, WARD_DAILY)["checks"]
                if c["name"] == "zero_inflated_count_shares")
    assert zero["passed"] is False and zero["synthetic"]["beds_closed"] == 1.0 and zero["score"] < 0.9


def test_pooled_check_fails_when_its_statistic_is_undefined_on_synthetic_data():
    """CODE-05: a statistic defined on the real data but undefined on the synthetic data (fewer than
    5 synthetic winter rows for the seasonal contrast) fails with score 0, like a missing elective
    pattern, instead of dropping out of the operational score and the pass count."""
    real = make_ward_daily(n_days=365, seed=11)
    winter = real["season"] == "winter"
    almost_no_winter = pd.concat([real.loc[~winter], real.loc[winter].head(3)], ignore_index=True)
    result = evaluate_operational(real, almost_no_winter, WARD_DAILY)
    check = next(c for c in result["checks"] if c["name"] == "seasonal_pressure_contrast")
    assert check["passed"] is False and check["score"] == 0.0 and check["value"] is None
    assert check["real"] is not None and check["synthetic"] is None
    assert "undefined on the synthetic data" in check["details"]["note"]
    assert "seasonal_pressure_contrast" in result["failed_checks"]
    scored = [c["score"] for c in result["checks"] if not c["strata_forced"] and c["score"] is not None]
    assert 0.0 in scored and result["score"] == pytest.approx(float(np.mean(scored)))
    assert result["n_scored"] == len(scored)

    # undefined on the real data as well: nothing to compare, the check stays unscored
    no_winter = real.loc[~winter].reset_index(drop=True)
    unscored = next(c for c in evaluate_operational(no_winter, almost_no_winter, WARD_DAILY)["checks"]
                    if c["name"] == "seasonal_pressure_contrast")
    assert unscored["passed"] is None and unscored["score"] is None


def _same_distribution_splits(table: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Disjoint random thirds of one fixture frame: training split, holdout, independent real sample."""
    big = make_ward_daily(n_days=360, seed=11) if table == WARD_DAILY else make_patient_stays(n=6000, seed=21)
    order = np.random.default_rng(0).permutation(len(big))
    third = len(big) // 3
    return tuple(big.iloc[order[i * third:(i + 1) * third]].reset_index(drop=True) for i in range(3))


@pytest.mark.parametrize("table", TABLES)
def test_operational_calibration_lets_real_samples_pass(table):
    """Thresholds calibrated to real-vs-real noise at the evaluated size let real samples pass (E7).

    The calibrated rule only relaxes the design rules (its failures are a subset), keeps every
    design threshold as a floor, and its cross-validated estimate of the share of real samples
    passing every check is high.
    """
    real, holdout, other = _same_distribution_splits(table)
    n_rows = 500
    sample = other.sample(n=n_rows, random_state=1)
    calibration = calibrate_operational(real, holdout, table, n_rows, reps=40, seed=3)
    assert calibration["n"] == n_rows and calibration["reps"] == 40
    assert set(calibration["checks"]) == {c["name"] for c in evaluate_operational(real, real, table)["checks"]}
    assert calibration["real_pass_rate_cv"] >= 0.7
    uncalibrated = evaluate_operational(real, sample, table)
    calibrated = evaluate_operational(real, sample, table, calibration=calibration)
    # Calibration only relaxes the design rules: its failures are a subset of the design failures.
    assert set(calibrated["failed_checks"]) <= set(uncalibrated["failed_checks"])
    assert len(calibrated["failed_checks"]) <= 1
    for check in calibrated["checks"]:
        reference = check["real_reference"]
        if check["threshold"] is not None and reference and reference["noise_limit"] is not None:
            assert check["threshold_used"] == pytest.approx(max(check["threshold"], reference["noise_limit"]))
            assert reference["noise_limit"] >= reference["max"]
            assert check["within_real_noise"] == (check["value"] <= reference["noise_limit"])
    assert calibrated["calibration"]["n"] == n_rows


def test_calibration_waives_structural_conditions_real_data_fails():
    """A pass-rule condition that real samples of the evaluated size violate is waived (E7)."""
    from bedsynth.evaluation.operational import CheckOutcome, _apply_threshold

    outcome = CheckOutcome("abs_diff_spearman", 0.08, "same sign and |Δρ| <= 0.15", False, 0.7,
                           threshold=0.15, conditions_ok=False)
    assert _apply_threshold(outcome, 0.15) is False
    assert _apply_threshold(outcome, 0.15, waive_conditions=True) is True
    assert _apply_threshold(outcome, 0.05, waive_conditions=True) is False


def test_strata_copy_null_loses_within_row_relationships(data):
    """A no-learning generator that keeps the strata marginals scores below a faithful model on
    within-row relationships (E12)."""
    real = data[WARD_DAILY]["real"]
    null = stratified_marginals(real, WARD_DAILY, len(real), ("ward_id", "occupancy_level"), seed=4)
    faithful = {c["name"]: c for c in evaluate_operational(real, real.copy(), WARD_DAILY)["checks"]}
    copied = {c["name"]: c for c in evaluate_operational(real, null, WARD_DAILY)["checks"]}
    relationships = [n for n, c in faithful.items() if c["group"] == "relationships" and c["score"] is not None]
    assert relationships
    assert any(copied[n]["score"] < faithful[n]["score"] - 0.05 for n in relationships)


def test_quick_quality_empty_synthetic(data):
    real = data[WARD_DAILY]["real"]
    result = quick_quality(real, real.iloc[0:0], WARD_DAILY)
    json.dumps(result, allow_nan=False)
    assert result["score"] is None and result["note"]


def test_quick_quality_excludes_columns(data):
    real = data[WARD_DAILY]["real"]
    result = quick_quality(real, shuffle_columns(real, WARD_DAILY), WARD_DAILY, include_validity=False,
                           exclude_columns=["ward_id", "occupancy_level"])
    assert "ward_id" not in result["column_scores"] and "occupancy_level" not in result["column_scores"]
    assert result["excluded_columns"] == ["occupancy_level", "ward_id"]


# ---------------------------------------------------------------------------
# Privacy
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("table", TABLES)
def test_privacy_flags_copy_and_accepts_independent_sample(data, table):
    real, holdout, other = data[table]["real"], data[table]["holdout"], data[table]["other"]
    copy = evaluate_privacy(real, real.copy(), table, holdout, run_sdmetrics=False)
    fresh = evaluate_privacy(real, other, table, holdout, run_sdmetrics=False)
    assert_result(copy)
    assert_result(fresh)
    assert copy["exact_matches"]["synthetic_rate"] == pytest.approx(1.0)
    assert copy["components"]["exact_match_protection"] == pytest.approx(0.0)
    assert copy["score"] < 0.35
    assert fresh["score"] > 0.8
    assert fresh["dcr"]["median_ratio"] > copy["dcr"]["median_ratio"]


def test_privacy_reports_aggregate_disclosure(data):
    """Output that copies the training strata mix exactly is flagged (E2 side note), not scored."""
    from bedsynth.evaluation.privacy import aggregate_disclosure

    real, holdout, other = data[WARD_DAILY]["real"], data[WARD_DAILY]["holdout"], data[WARD_DAILY]["other"]
    copied = aggregate_disclosure(real, real.sample(frac=1.0, random_state=1), holdout, ["ward_id", "occupancy_level"])
    assert copied["synthetic_vs_train_tvd"] == pytest.approx(0.0) and copied["mirrors_training_frequencies"] is True
    independent = aggregate_disclosure(real, other, holdout, ["ward_id", "occupancy_level"])
    assert independent["synthetic_vs_train_tvd"] > 0
    result = evaluate_privacy(real, other, WARD_DAILY, holdout, run_sdmetrics=False,
                              disclosure_columns=["ward_id", "occupancy_level"])
    assert result["aggregate_disclosure"]["columns"] == ["ward_id", "occupancy_level"]
    unscored = evaluate_privacy(real, other, WARD_DAILY, holdout, run_sdmetrics=False)
    assert unscored["aggregate_disclosure"] is None and unscored["score"] == pytest.approx(result["score"])


def test_privacy_without_holdout_and_sdmetrics_cross_check(data):
    real, holdout, other = (data[WARD_DAILY][k] for k in ("real", "holdout", "other"))
    no_holdout = evaluate_privacy(real, other, WARD_DAILY)
    assert_result(no_holdout)
    assert no_holdout["components"]["dcr_ratio_protection"] is None
    assert no_holdout["score"] == pytest.approx(no_holdout["components"]["exact_match_protection"])

    checked = evaluate_privacy(real, other, WARD_DAILY, holdout, sdmetrics_rows=60)
    assert_result(checked)
    sdm = checked["sdmetrics"]
    assert sdm["ran"] and sdm["rows"] == 60 and "error" not in sdm
    assert 0.0 <= sdm["DCRBaselineProtection"]["score"] <= 1.0
    assert 0.0 <= sdm["DCROverfittingProtection"]["score"] <= 1.0


# ---------------------------------------------------------------------------
# ML utility (TSTR)
# ---------------------------------------------------------------------------
ML_ITER = 40  # fewer boosting iterations keep the unit tests fast


def test_ml_utility_tstr_runs_on_small_ward_daily(data):
    real, holdout = data[WARD_DAILY]["real"], data[WARD_DAILY]["holdout"]
    identical = evaluate_ml_utility(real, holdout, real.copy(), WARD_DAILY, max_iter=ML_ITER)
    shuffled = evaluate_ml_utility(real, holdout, shuffle_columns(real, WARD_DAILY), WARD_DAILY, max_iter=ML_ITER)
    assert_result(identical)
    assert_result(shuffled)
    assert [t["name"] for t in identical["tasks"]] == ["high_pressure_classification", "discharges_regression"]
    assert identical["score"] == pytest.approx(1.0)  # same data + same seed -> TSTR == TRTR
    assert shuffled["score"] < identical["score"]
    regression = identical["tasks"][1]
    assert set(regression["trtr"]) == {"r2", "mae"} and regression["informative"]


def test_ml_utility_ratios_give_no_skill_classifiers_zero():
    """All-positive and random classifiers score ~0 (the old F1 ratio gave them ~0.5) (E3)."""
    rng = np.random.default_rng(0)
    y = (rng.random(4000) < 0.28).astype(int)
    informative = np.clip(0.28 + 0.35 * (y - 0.28) + rng.normal(0, 0.2, y.size), 0, 1)
    trtr = classification_metrics(y, informative)
    assert trtr["auc"] > 0.7 and trtr["average_precision"] > trtr["positive_rate"] + 0.1
    assert trtr["f1_no_skill"] == pytest.approx(2 * y.mean() / (1 + y.mean()))
    all_positive = classification_metrics(y, np.ones(y.size))
    random_scores = classification_metrics(y, rng.random(y.size))
    assert mean_or_none(utility_ratios("classification", trtr, all_positive).values()) == pytest.approx(0.0)
    assert mean_or_none(utility_ratios("classification", trtr, random_scores).values()) < 0.06
    assert mean_or_none(utility_ratios("classification", trtr, trtr).values()) == pytest.approx(1.0)
    assert utility_ratios("regression", {"r2": 0.6}, {"r2": -0.1}) == {"r2_ratio": 0.0}


def test_ml_utility_tstr_is_compared_with_a_size_matched_trtr(data):
    """A perfect generator with fewer rows than the training split scores ~1 (E4)."""
    real, holdout = data[WARD_DAILY]["real"], data[WARD_DAILY]["holdout"]
    perfect = real.sample(n=200, random_state=1).reset_index(drop=True)
    result = evaluate_ml_utility(real, holdout, perfect, WARD_DAILY, max_iter=ML_ITER)
    regression = next(t for t in result["tasks"] if t["name"] == "discharges_regression")
    assert regression["size_matched"] and regression["n_rows"]["train_real"] == 200
    assert regression["n_rows"]["train_real_full"] == len(real) and regression["trtr_full"] is not None
    unmatched = evaluate_ml_utility(real, holdout, perfect, WARD_DAILY, max_iter=ML_ITER, size_matched=False)
    assert unmatched["tasks"][1]["trtr_full"] is None and unmatched["size_matched"] is False


def _learnable_stays(n: int, seed: int) -> pd.DataFrame:
    """Stays whose LOS depends strongly on the ward (so the LOS task is learnable)."""
    df = make_patient_stays(n=n, seed=seed)
    base = {w: i for i, w in enumerate(schema.WARD_IDS)}
    df["los_nights"] = df["ward_id"].map(base) * 2 + (df.index.to_series() % 2)
    df["discharge_delay_hours"] = 0.0
    df["delay_reason"] = "none"
    return schema.derive(PATIENT_STAYS, df)


def test_ml_utility_patient_stays(data):
    real, holdout = _learnable_stays(500, 31), _learnable_stays(200, 32)
    identical = evaluate_ml_utility(real, holdout, real.copy(), PATIENT_STAYS, max_iter=ML_ITER)
    assert_result(identical)
    los = next(t for t in identical["tasks"] if t["name"] == "los_regression")
    assert los["informative"] and los["trtr"]["r2"] > 0.5
    assert identical["score"] == pytest.approx(1.0)

    shuffled = evaluate_ml_utility(real, holdout, shuffle_columns(real, PATIENT_STAYS), PATIENT_STAYS,
                                   max_iter=ML_ITER)
    assert shuffled["score"] < 0.5

    # Fixture stays have random LOS / delays: tasks are uninformative and the score is undefined.
    random_tasks = evaluate_ml_utility(data[PATIENT_STAYS]["real"], data[PATIENT_STAYS]["holdout"],
                                       data[PATIENT_STAYS]["other"], PATIENT_STAYS, max_iter=ML_ITER)
    assert_result(random_tasks, score_required=False)
    if random_tasks["score"] is None:
        assert random_tasks["n_informative_tasks"] == 0 and random_tasks["note"]


# ---------------------------------------------------------------------------
# Optimisation utility
# ---------------------------------------------------------------------------
def test_saa_quota_hand_made_example():
    free = np.array([1, 3, 3, 3, 3, 3, 3, 6, 6, 6], dtype=float)
    # Newsvendor optimum: smallest q with P(F <= q) >= 1 / (1 + λ) = 1/3  ->  q = 3.
    quota, curve = optimal_quota(free, max_quota=9, penalty=2.0)
    assert quota == 3
    assert curve[3] == pytest.approx((1 + 3 * 9) / 10 - 2.0 * 2 / 10)   # admitted 2.8 - 2 x 0.2 cancelled
    assert curve[0] == 0.0 and np.argmax(curve) == 3
    assert optimal_quota(free, 9, penalty=0.0)[0] == 6     # no cancellation cost: book up to max free beds
    assert optimal_quota(free, 9, penalty=100.0)[0] == 1   # prohibitive cost: book only certain beds
    assert quota_objective(free, np.array([3]), 2.0)[0] == pytest.approx(2.4)


def _quota_frame(free_beds: list[int], ward_id: str = "GSUR", day: str = "Mon") -> pd.DataFrame:
    n = len(free_beds)
    return pd.DataFrame({
        "ward_id": [ward_id] * n, "day_of_week": [day] * n, "elective_scheduled": [4] * n,
        "available_beds": [30] * n, "census_start": [30 - f for f in free_beds], "admissions_emergency": [0] * n,
        "transfers_in": [0] * n, "discharges": [0] * n, "transfers_out": [0] * n,
    })


def test_free_beds_and_booking_regret():
    real = _quota_frame([1, 3, 3, 3, 3, 3, 3, 6, 6, 6])
    assert free_beds_for_electives(real).tolist() == [1, 3, 3, 3, 3, 3, 3, 6, 6, 6]
    row = pd.DataFrame({"available_beds": [20], "census_start": [18], "admissions_emergency": [4],
                        "transfers_in": [1], "discharges": [5], "transfers_out": [1]})
    assert free_beds_for_electives(row).iloc[0] == 3        # 20 - (18 + 4 + 1 - 5 - 1)

    same = evaluate_booking_quotas(real, real.copy())
    assert same["quota_agreement"] == 1.0 and same["relative_regret"] == 0.0 and same["score"] == 1.0
    assert same["regret_basis"] == "in_sample_training_split"

    full = _quota_frame([0] * 10)                           # synthetic says there are never free beds
    worse = evaluate_booking_quotas(real, full)
    group = worse["groups"][0]
    assert (group["quota_real"], group["quota_synthetic"]) == (3, 0)
    assert group["in_sample_regret"] == pytest.approx(2.4)  # real objective at q=3 minus at q=0
    assert worse["relative_regret"] == pytest.approx(1.0) and worse["score"] == 0.0


def test_booking_quota_regret_is_out_of_sample_on_the_holdout():
    """Decisions from train and from synthetic data are both scored on the holdout (E8)."""
    train = _quota_frame([1, 3, 3, 3, 3, 3, 3, 6, 6, 6])
    free_holdout = [2, 2, 2, 2, 2, 3, 3, 3, 3, 3]           # the holdout has fewer free beds than train
    holdout = _quota_frame(free_holdout)
    synthetic = _quota_frame([2] * 10)                      # synthetic data suggests booking 2
    result = evaluate_booking_quotas(train, synthetic, holdout=holdout)
    group = result["groups"][0]
    assert result["regret_basis"] == "out_of_sample_holdout" and result["n_groups_out_of_sample"] == 1
    assert (group["quota_real"], group["quota_synthetic"]) == (3, 2)
    curve = quota_objective(np.array(free_holdout, dtype=float), np.arange(group["max_quota"] + 1))
    assert group["holdout_objective_real_quota"] == pytest.approx(curve[3]) == pytest.approx(1.5)
    assert group["holdout_objective_synthetic_quota"] == pytest.approx(curve[2]) == pytest.approx(2.0)
    # Out of sample the synthetic decision beats the real-trained one: negative regret, full score.
    assert result["relative_regret"] == pytest.approx(-0.5 / 1.5) and result["score"] == 1.0
    assert result["real_decision_regret_vs_oracle"] == pytest.approx(0.25)   # (oracle 2.0 - 1.5) / 2.0
    assert result["in_sample_relative_regret"] == pytest.approx((2.4 - 1.7) / 2.4)  # real quota optimal in sample


def test_optimization_utility_identical_vs_shuffled(data):
    real = data[WARD_DAILY]["real"]
    identical = evaluate_optimization_utility(real, real.copy())
    shuffled = evaluate_optimization_utility(real, shuffle_columns(real, WARD_DAILY))
    assert_result(identical)
    assert_result(shuffled)
    assert identical["score"] == pytest.approx(1.0)
    policies = identical["policy_evaluation"]
    assert policies["kendall_tau"] == pytest.approx(1.0) and policies["objective_accuracy"] == pytest.approx(1.0)
    assert len(policies["kpis"]) == len(POLICIES) == 8
    assert shuffled["score"] < identical["score"]
    assert shuffled["capacity_planning"]["mean_relative_error"] > 0
    # Component-level discrimination (E12): shuffling must cost the policy component clearly.
    assert shuffled["components"]["policy_evaluation"] < 0.9
    assert set(identical["components"]) == {"booking_quota", "policy_evaluation", "capacity_planning"}


def test_policy_objective_accuracy():
    assert objective_accuracy([1.0, 2.0], [1.0, 2.0]) == pytest.approx(1.0)
    assert objective_accuracy([1.0, 2.0], [0.5, 2.0]) == pytest.approx(0.75)
    assert objective_accuracy([1.0, 2.0], [None, None]) == pytest.approx(0.0)  # no elective activity at all
    assert objective_accuracy([None], [1.0]) is None


# ---------------------------------------------------------------------------
# Scenario coverage
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("table", TABLES)
def test_scenario_coverage_with_fake_generator(data, table):
    real = data[table]["real"]
    good = evaluate_scenario_coverage(FakeGenerator(real, table), real, table, model_id="m", rows_per_scenario=60, seed=1)
    bad = evaluate_scenario_coverage(FakeGenerator(real, table, mode="scrambled"), real, table, rows_per_scenario=60, seed=1)
    assert_result(good)
    assert_result(bad)
    assert good["n_scenarios"] == len(scenarios_for(table))
    assert good["score"] > bad["score"]
    supported = [s for s in good["scenarios"] if s["real_support"]["rows"] > 0]
    assert supported and all(s["in_scenario_rate"] == pytest.approx(1.0) for s in supported)
    for entry in good["scenarios"]:
        assert entry["name"] and entry["description"] and 0.0 <= entry["score"] <= 1.0


class SlowGenerator(FakeGenerator):
    """FakeGenerator that takes ``delay`` seconds per request."""

    def __init__(self, *args, delay: float = 0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.delay = delay

    def generate(self, request) -> GenerationResult:
        time.sleep(self.delay)
        return super().generate(request)


def test_scenario_coverage_budgets(data):
    """Per-request budgets reach the generator; scenarios beyond the total budget are listed (E11)."""
    real = data[WARD_DAILY]["real"]
    generator = SlowGenerator(real, WARD_DAILY, delay=0.6)
    result = evaluate_scenario_coverage(generator, real, WARD_DAILY, rows_per_scenario=30, seed=1,
                                        time_budget_seconds=20, max_attempts=8, total_budget_seconds=1.5)
    assert_result(result)
    assert all(r.max_attempts == 8 and r.time_budget_seconds <= 20 for r in generator.requests)
    assert generator.requests[0].time_budget_seconds == pytest.approx(1.5, abs=0.1)  # capped by the total budget
    assert result["complete"] is False and result["not_evaluated_scenarios"]
    assert result["n_evaluated"] == len(generator.requests) < result["n_scenarios"]
    skipped = next(s for s in result["scenarios"] if s["status"] == "not_evaluated")
    assert skipped["score"] is None and "budget" in skipped["reason"]


class BudgetedGenerator(FakeGenerator):
    """FakeGenerator whose reports say the request ran out of time (shortfall) after throttled rounds."""

    def generate(self, request) -> GenerationResult:
        result = super().generate(request)
        keep = max(1, len(result.data) // 2)
        report = {**result.report, "stopped_reason": "time budget of 1s exhausted after 1.2s (2 round(s) completed)",
                  "throttled_rounds": 2}
        return GenerationResult(data=result.data.head(keep), report=report)


def test_budget_stops_and_throttling_are_recorded(evaluate_script, data):
    """Shortfalls from the time budget are recorded honestly, never hidden (E11)."""
    real = data[WARD_DAILY]["real"]
    coverage = evaluate_scenario_coverage(BudgetedGenerator(real, WARD_DAILY), real, WARD_DAILY, rows_per_scenario=40,
                                          seed=1, scenarios=["baseline", "high_occupancy"])
    assert coverage["budget_stopped_scenarios"] == ["baseline", "high_occupancy"]
    assert coverage["throttled_scenarios"] == ["baseline", "high_occupancy"]
    assert all(s["fulfilment"] == pytest.approx(0.5) for s in coverage["scenarios"])
    _, summary = evaluate_script.generate_synthetic(BudgetedGenerator(real, WARD_DAILY), WARD_DAILY, "m", 100, seed=1,
                                                    max_rows_per_request=50, time_budget_seconds=1.0)
    assert summary["shortfall"] == 50 and summary["throttled_rounds"] == 4 and len(summary["stopped_reasons"]) == 2


def test_scenario_coverage_flags_rebalanced_strata(data):
    """Output whose allocation-strata mix drifts from the real subset is flagged and scores lower (E9)."""
    real = data[WARD_DAILY]["real"]
    skewed = FakeGenerator(real.loc[real["ward_id"] == "AMU"], WARD_DAILY, mode="faithful")
    good = evaluate_scenario_coverage(FakeGenerator(real, WARD_DAILY), real, WARD_DAILY, rows_per_scenario=80,
                                      seed=1, scenarios=["baseline"])
    bad = evaluate_scenario_coverage(skewed, real, WARD_DAILY, rows_per_scenario=80, seed=1, scenarios=["baseline"])
    good_entry, bad_entry = good["scenarios"][0], bad["scenarios"][0]
    assert bad_entry["strata_mix_tvd"] > 0.5 and bad_entry["rebalancing_flag"] is True
    assert good_entry["strata_mix_tvd"] < bad_entry["strata_mix_tvd"] and bad["rebalanced_scenarios"] == ["baseline"]
    assert bad_entry["score"] < good_entry["score"]
    assert good_entry["combinations"]["strata_forced"] is True        # forced by the quotas: not scored
    assert strata_mix_tvd(real, real, ("ward_id", "occupancy_level")) == pytest.approx(0.0)


def test_rebalancing_flag_allows_for_the_rounding_floor():
    """A perfect small output cannot match many strata exactly: only TVD beyond that floor is flagged."""
    real = pd.DataFrame({"s": [f"s{i}" for i in range(60) for _ in range(i % 4 + 1)]})   # 60 uneven strata
    floor = strata_rounding_tvd(real, 100, ["s"])
    assert 0.05 < floor < 0.4                                    # 100 rows cannot follow 60 shares exactly
    assert strata_rounding_tvd(real, 100_000, ["s"]) < 0.001     # the floor vanishes for large outputs
    assert strata_rounding_tvd(real, 0, ["s"]) is None and strata_rounding_tvd(real, 10, ["x"]) is None
    assert not rebalancing_flagged(floor + 0.04, floor, 0.05)    # rounding-level drift is not rebalancing
    assert rebalancing_flagged(floor + 0.06, floor, 0.05) and not rebalancing_flagged(None, floor, 0.05)


def test_scenario_coverage_records_failures(data):
    real = data[WARD_DAILY]["real"]
    generator = FakeGenerator(real, WARD_DAILY, fail=("high_occupancy",))
    result = evaluate_scenario_coverage(generator, real, WARD_DAILY, rows_per_scenario=40, seed=3,
                                        scenarios=["baseline", "high_occupancy"])
    assert_result(result)
    assert result["n_scenarios"] == 2 and result["failed_scenarios"] == ["high_occupancy"]
    failed = next(s for s in result["scenarios"] if s["name"] == "high_occupancy")
    assert failed["score"] == 0.0 and "GenerationError" in failed["error"]
    assert [r.seed for r in generator.requests] == [3, 4]


# ---------------------------------------------------------------------------
# Reference pseudo-models
# ---------------------------------------------------------------------------
def test_pool_generator_follows_the_allocation_strata(data):
    real, holdout = data[WARD_DAILY]["real"], data[WARD_DAILY]["holdout"]
    from bedsynth.synthesis.generator import GenerationRequest

    generator = PoolGenerator(holdout, real, WARD_DAILY, ("ward_id", "occupancy_level"), name="ref",
                              acceptance_rate=1.0)
    result = generator.generate(GenerationRequest(table=WARD_DAILY, num_rows=400, seed=3))
    assert len(result.data) == 400 and result.data["record_id"].is_unique
    assert strata_mix_tvd(real, result.data, ("ward_id", "occupancy_level")) < 0.06
    scenario = generator.generate(GenerationRequest(table=WARD_DAILY, num_rows=50, scenario="high_occupancy", seed=3))
    assert scenario_mask(scenario.data, "high_occupancy").all()
    assert scenario.report["allocation_strata"]["columns"] == ["ward_id", "occupancy_level"]
    again = generator.generate(GenerationRequest(table=WARD_DAILY, num_rows=400, seed=3))
    pd.testing.assert_frame_equal(result.data, again.data)              # deterministic for a seed


def test_normalised_scores():
    out = normalised_scores({"fidelity": 0.9, "validity": 0.5, "privacy": 1.0},
                            {"fidelity": 1.0, "validity": 1.0, "privacy": 1.0},
                            {"fidelity": 0.8, "validity": 0.0, "privacy": 1.0})
    assert out["fidelity"] == pytest.approx(0.5) and out["validity"] == pytest.approx(0.5)
    assert out["privacy"] is None                                       # no room between floor and ceiling


def test_evaluate_references_on_fixtures(data):
    """The real holdout scores near the ceiling, the no-learning baseline below it."""
    real, holdout = data[WARD_DAILY]["real"], data[WARD_DAILY]["holdout"]
    refs = evaluate_references(WARD_DAILY, real, holdout, 300, strata=("ward_id", "occupancy_level"), seed=5,
                               include_scenarios=False)
    assert set(refs) == {REAL_REFERENCE_ID, NULL_REFERENCE_ID}
    for ref_id, result in refs.items():
        json.dumps(result, allow_nan=False)
        assert "error" not in result, result.get("error")
        assert result["reference"]["id"] == ref_id and result["composite"]["complete"]
    real_ref, null_ref = refs[REAL_REFERENCE_ID], refs[NULL_REFERENCE_ID]
    assert real_ref["validity"]["basis"] == "real_holdout" and real_ref["validity"]["score"] == pytest.approx(1.0)
    assert null_ref["validity"]["score"] < 1.0
    assert real_ref["composite"]["score"] > null_ref["composite"]["score"]
    assert real_ref["ml_utility"]["tasks"][0]["n_rows"]["train_synthetic"] == 300   # train subsample, not holdout


# ---------------------------------------------------------------------------
# evaluate_model
# ---------------------------------------------------------------------------
def test_evaluate_model_end_to_end(data):
    real, holdout, other = (data[WARD_DAILY][k] for k in ("real", "holdout", "other"))
    generator = FakeGenerator(real, WARD_DAILY)
    result = evaluate_model(WARD_DAILY, real, holdout, other, raw_synthetic=perturb(real, WARD_DAILY),
                            generator=generator, model_id="fake", seed=7, rows_per_scenario=30,
                            privacy_sdmetrics=False, forced_columns=["ward_id", "occupancy_level"],
                            scenario_options={"time_budget_seconds": 10, "max_attempts": 3})
    json.dumps(result, allow_nan=False)
    for family in ("statistical", "validity", "operational", "privacy", "ml_utility", "optimization",
                   "scenario_coverage", "output_validity", "model_view"):
        assert family in result and "seconds" in result[family]
        assert "error" not in result[family], result[family].get("error")
    assert result["validity"]["basis"] == "raw_model_samples"
    assert result["output_validity"]["basis"] == "generator_output"
    assert result["operational"]["system_checks"] == ["occupancy_band_mix_by_ward"]
    quotas = result["optimization"]["booking_quota"]
    # The 20-day fixture holdout has < 5 ward-days per ward x weekday: the regret falls back to in sample.
    assert (quotas["regret_basis"] == "out_of_sample_holdout") == (quotas["n_groups_out_of_sample"] > 0)
    assert result["model_view"]["basis"] == "raw_model_samples"
    assert result["model_view"]["statistical"]["score"] is not None
    assert result["model_view"]["operational"]["score"] is not None
    assert all(r.max_attempts == 3 for r in generator.requests)
    composite = result["composite"]
    assert 0.0 <= composite["score"] <= 1.0 and composite["complete"] is True
    assert composite["components"]["utility"] == pytest.approx(
        np.mean([result["ml_utility"]["score"], result["optimization"]["score"]]))
    assert result["timings"]["total"] > 0


def test_evaluate_model_isolates_family_failures(monkeypatch, data):
    from bedsynth.evaluation import operational, statistical, utility

    def broken(*args, **kwargs):
        raise RuntimeError("operational exploded")

    def stub(name: str):
        return lambda *args, **kwargs: {"name": name, "description": "stub", "score": 0.5}

    monkeypatch.setattr(operational, "evaluate_operational", broken)
    monkeypatch.setattr(statistical, "evaluate_statistical", stub("statistical_fidelity"))  # keep the test fast
    monkeypatch.setattr(utility, "evaluate_ml_utility", stub("ml_utility"))
    real, holdout = data[PATIENT_STAYS]["real"], data[PATIENT_STAYS]["holdout"]
    result = evaluate_model(PATIENT_STAYS, real, holdout, data[PATIENT_STAYS]["other"], include_scenarios=False,
                            privacy_sdmetrics=False)
    json.dumps(result, allow_nan=False)
    assert result["operational"]["score"] is None and "operational exploded" in result["operational"]["error"]
    composite = result["composite"]
    assert composite["failed_families"] == ["operational"] and composite["complete"] is False
    assert composite["components"]["operational"] == 0.0               # a crash counts as 0, never as neutral
    assert "utility" not in composite["failed_components"]
    assert result["optimization"]["skipped"] and result["scenario_coverage"]["skipped"]
    assert composite["score"] is not None


# ---------------------------------------------------------------------------
# scripts/03_evaluate.py
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def evaluate_script():
    spec = importlib.util.spec_from_file_location("evaluate_script", PROJECT_ROOT / "scripts" / "03_evaluate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeModel:
    def __init__(self, reference: pd.DataFrame, table: str):
        self.reference, self.table = reference, table

    def sample(self, num_rows: int, conditions=None, seed=None) -> pd.DataFrame:
        rows = self.reference.sample(n=num_rows, replace=True, random_state=seed)
        return rows[schema.training_columns(self.table)].reset_index(drop=True)


class FakeRegistry:
    def __init__(self, models: dict[str, FakeModel], table: str):
        self.models, self.table, self.defaults = models, table, {}

    def list_models(self, table=None):
        return [{"table": self.table, "experiment_id": m} for m in self.models] if table in (None, self.table) else []

    def load(self, table, experiment_id=None):
        return self.models[experiment_id]

    def set_default(self, table, experiment_id, reason=""):
        self.defaults[table] = (experiment_id, reason)


class RoutingGenerator:
    """Routes requests to a per-model FakeGenerator (by request.model_id)."""

    def __init__(self, generators: dict[str, FakeGenerator]):
        self.generators = generators

    def generate(self, request) -> GenerationResult:
        return self.generators[request.model_id].generate(request)


def test_script_helpers(evaluate_script):
    args = evaluate_script.parse_args(["--tables", "ward_daily", "--rows", "50", "--quick", "--no-scenarios"])
    assert args.tables == ["ward_daily"] and args.rows == 50 and args.quick and args.no_scenarios
    assert evaluate_script.synthetic_row_count(7000) == 7000
    assert evaluate_script.synthetic_row_count(7000, rows=123) == 123
    assert evaluate_script.synthetic_row_count(7000, quick=True) == evaluate_script.QUICK_MAX_ROWS
    cap = evaluate_script.evaluation_settings()["max_synthetic_rows"]
    assert evaluate_script.synthetic_row_count(44882) == min(44882, cap)      # default capped (patient_stays)
    assert evaluate_script.synthetic_row_count(44882, rows=30000) == 30000    # an explicit --rows wins
    settings = evaluate_script.evaluation_settings()
    for key in ("sample_time_budget_seconds", "scenario_time_budget_seconds", "scenario_max_attempts",
                "coverage_time_budget_seconds", "noise_calibration_reps", "noise_sd_multiplier"):
        assert settings[key] is not None and settings[key] > 0
    quick = evaluate_script.evaluation_settings(quick=True)
    for key in ("sample_time_budget_seconds", "scenario_time_budget_seconds", "coverage_time_budget_seconds"):
        assert 0 < quick[key] < settings[key]                                  # --quick tightens the caps
    entries = [{"model_id": "a", "composite": 0.7}, {"model_id": "b", "composite": None}, {"model_id": "c", "composite": 0.8}]
    assert evaluate_script.select_best(entries)["model_id"] == "c"
    assert evaluate_script.select_best([{"model_id": "x", "composite": None}]) is None

    existing = {"tables": {"ward_daily": {"models": [{"model_id": "old", "composite": 0.9},
                                                     {"model_id": "gone", "composite": 0.95}]}}}
    merged = evaluate_script.merge_comparison(existing, {"ward_daily": [{"model_id": "new", "composite": 0.8}]},
                                              {"ward_daily": ["old", "new"]})
    table = merged["tables"]["ward_daily"]
    assert [e["model_id"] for e in table["models"]] == ["old", "new"]   # 'gone' is no longer registered
    assert table["best_model"] == "old"
    assert evaluate_script.select_best([{"model_id": "x", "composite": 0.9, "complete": False},
                                        {"model_id": "y", "composite": 0.5}])["model_id"] == "y"


def test_script_merge_never_selects_stale_failed_or_incomparable_entries(evaluate_script):
    """E10: a model that fails now, an entry of an older model version or of another run
    configuration can never become the registry default."""
    config = {"quick": True, "rows": 2000, "rows_per_scenario": 100, "scenarios": True, "seed": 42}
    other_config = {**config, "quick": False, "rows": 44882}
    existing = {"tables": {"patient_stays": {"models": [
        {"model_id": "gc_tuned", "composite": 0.95, "run_config": config, "model_fitted_at": "t-old"},
        {"model_id": "tvae", "composite": 0.90, "run_config": other_config, "model_fitted_at": "t1"},
        {"model_id": "ctgan", "composite": 0.40, "run_config": config, "model_fitted_at": "t1"},
    ]}}}
    new = {"patient_stays": [
        evaluate_script.failed_entry("gc_segmented", "RuntimeError: boom", config, {"fitted_at": "t1"}),
        {"model_id": "gc_baseline", "composite": 0.50, "complete": True, "status": "ok", "run_config": config,
         "model_fitted_at": "t1", "components": {"fidelity": 0.5}},
    ]}
    registered = {"patient_stays": ["gc_tuned", "tvae", "ctgan", "gc_segmented", "gc_baseline"]}
    fitted = {"patient_stays": {"gc_tuned": "t-new", "tvae": "t1", "ctgan": "t1", "gc_segmented": "t1",
                                "gc_baseline": "t1"}}
    merged = evaluate_script.merge_comparison(existing, new, registered, run_configs={"patient_stays": config},
                                              fitted=fitted)
    table = merged["tables"]["patient_stays"]
    by_id = {e["model_id"]: e for e in table["models"]}
    assert "gc_tuned" not in by_id                                       # stale: the model was retrained since
    assert by_id["tvae"]["comparable"] is False                          # other run configuration
    assert by_id["gc_segmented"]["status"] == "failed"
    assert table["best_model"] == "gc_baseline" and table["best_composite"] == 0.50
    assert "selection_rule" in merged


def test_script_reference_rows_and_normalised_scores(evaluate_script):
    refs = [{"model_id": "real_holdout", "status": "ok", "composite": 0.97,
             "components": {"fidelity": 0.97, "validity": 1.0}},
            {"model_id": "null_stratified_marginals", "status": "ok", "composite": 0.75,
             "components": {"fidelity": 0.93, "validity": 0.2}}]
    new = {"ward_daily": [{"model_id": "m", "status": "ok", "composite": 0.86,
                           "components": {"fidelity": 0.95, "validity": 0.6}}]}
    merged = evaluate_script.merge_comparison({"tables": {}}, new, {"ward_daily": ["m"]},
                                              references={"ward_daily": refs})
    table = merged["tables"]["ward_daily"]
    assert [r["model_id"] for r in table["references"]] == ["real_holdout", "null_stratified_marginals"]
    assert table["best_model"] == "m"                                     # references are never selected
    normalised = table["models"][0]["normalised"]
    assert normalised["validity"] == pytest.approx(0.5) and normalised["composite"] == pytest.approx(0.5)


def test_script_chunked_generation(evaluate_script, data):
    real = data[WARD_DAILY]["real"]
    generator = FakeGenerator(real, WARD_DAILY)
    frame, summary = evaluate_script.generate_synthetic(generator, WARD_DAILY, "m", 250, seed=5, max_rows_per_request=100,
                                                        time_budget_seconds=45.0)
    assert len(frame) == 250 and frame["record_id"].is_unique
    assert [r.num_rows for r in generator.requests] == [100, 100, 50]
    assert [r.seed for r in generator.requests] == [5, 6, 7]
    assert all(r.time_budget_seconds == 45.0 for r in generator.requests)
    assert summary["chunks"] == 3 and summary["acceptance_rate"] == pytest.approx(1.0)
    assert summary["shortfall"] == 0 and summary["stopped_reasons"] == []


class StrataReportingGenerator(FakeGenerator):
    """FakeGenerator whose reports carry an allocation-strata block (like the real generator)."""

    def generate(self, request) -> GenerationResult:
        result = super().generate(request)
        result.report["allocation_strata"] = {"columns": ["ward_id", "occupancy_level"], "source": "reference",
                                              "strata_cells": 4, "rebalanced_rows": 7}
        return result


def test_script_generation_flags_strata_drift(evaluate_script, data):
    """E9: rebalancing that skews the output mix is visible in the generation summary."""
    real = data[WARD_DAILY]["real"]
    skewed = StrataReportingGenerator(real.loc[real["ward_id"] == "ICU"], WARD_DAILY)
    _, summary = evaluate_script.generate_synthetic(skewed, WARD_DAILY, "m", 120, seed=1, reference=real,
                                                    strata_tvd_flag=0.05)
    assert summary["allocation_strata"]["rebalanced_rows"] == 7
    assert summary["strata_mix_tvd"] > 0.5 and summary["rebalancing_flag"] is True
    assert any("allocation-strata mix" in w for w in summary["warnings"])
    _, faithful = evaluate_script.generate_synthetic(StrataReportingGenerator(real, WARD_DAILY), WARD_DAILY, "m",
                                                     len(real), seed=1, reference=real, strata_tvd_flag=0.05)
    assert faithful["strata_mix_tvd"] < summary["strata_mix_tvd"]


def _script_fixtures(real: pd.DataFrame, holdout: pd.DataFrame) -> tuple[FakeRegistry, RoutingGenerator]:
    """Processed splits on disk plus a registry/generator with a faithful and a scrambled model."""
    io.write_table(real, io.processed_path(WARD_DAILY, "train"))
    io.write_table(holdout, io.processed_path(WARD_DAILY, "holdout"))
    scrambled = shuffle_columns(real, WARD_DAILY, seed=9)
    registry = FakeRegistry({"good": FakeModel(real, WARD_DAILY), "bad": FakeModel(scrambled, WARD_DAILY)}, WARD_DAILY)
    generator = RoutingGenerator({"good": FakeGenerator(real, WARD_DAILY),
                                  "bad": FakeGenerator(real, WARD_DAILY, mode="scrambled")})
    return registry, generator


def test_script_run_orchestration(evaluate_script, data, isolated_paths, monkeypatch):
    """Generation, raw samples, result files, comparison and defaults (evaluation itself stubbed)."""
    real, holdout = data[WARD_DAILY]["real"], data[WARD_DAILY]["holdout"]
    registry, generator = _script_fixtures(real, holdout)
    calls = []

    def fake_evaluate_model(table, real_train, real_holdout, synthetic, raw_synthetic=None, model_id=None, **kwargs):
        calls.append({"model_id": model_id, "synthetic": len(synthetic), "raw": len(raw_synthetic), **kwargs})
        score = quick_quality(real_train, synthetic, table, include_validity=False)["score"]
        families = {f: {"name": f, "description": "stub", "score": score}
                    for f in ("statistical", "validity", "operational", "privacy", "ml_utility", "optimization")}
        return {"table": table, "model_id": model_id, "n_rows": {"synthetic": len(synthetic)}, **families,
                "composite": composite_score(families), "timings": {"total": 0.0}}

    def fake_references(table, train, holdout, n_rows, **kwargs):
        calls.append({"model_id": "references", "n_rows": n_rows, **kwargs})
        return {ref: {"table": table, "model_id": ref, "reference": {"kind": kind},
                      "composite": {"score": score, "components": {"fidelity": score}}}
                for ref, kind, score in (("real_holdout", "ceiling", 0.99), ("null_stratified_marginals", "floor", 0.1))}

    monkeypatch.setattr(evaluate_script, "evaluate_model", fake_evaluate_model)
    monkeypatch.setattr(evaluate_script, "evaluate_references", fake_references)
    monkeypatch.setattr(evaluate_script, "calibrate_operational",
                        lambda train, holdout, table, n, **kw: {"n": n, "reps": kw["reps"], "checks": {}})
    args = evaluate_script.parse_args(["--tables", "ward_daily", "--rows", "200", "--quick", "--no-scenarios"])
    outcome = evaluate_script.run(args, registry, generator)

    model_calls = [c for c in calls if c["model_id"] != "references"]
    assert [c["model_id"] for c in model_calls] == ["good", "bad"]
    assert all(c["synthetic"] == 200 and c["raw"] == 200 for c in model_calls)
    assert all(c["include_scenarios"] is False and c["privacy_sdmetrics"] is False for c in model_calls)
    assert all(c["operational_calibration"]["n"] == 200 for c in model_calls)
    assert all(c["scenario_options"]["max_attempts"] == evaluate_script.evaluation_settings()["scenario_max_attempts"]
               for c in model_calls)
    assert [c["n_rows"] for c in calls if c["model_id"] == "references"] == [200]
    results_dir = isolated_paths["results_dir"]
    assert (results_dir / "ward_daily__good.json").exists() and (results_dir / "ward_daily__bad.json").exists()
    assert (results_dir / "ward_daily__reference__real_holdout.json").exists()
    comparison = json.loads((results_dir / "model_comparison.json").read_text(encoding="utf-8"))
    table = comparison["tables"]["ward_daily"]
    assert table["best_model"] == "good" and len(table["references"]) == 2
    assert table["run_config"]["rows"] == 200 and table["noise_calibration"]["n"] == 200
    assert all(e["run_config"]["quick"] is True and "normalised" in e for e in table["models"])
    assert registry.defaults["ward_daily"][0] == "good"
    assert outcome["defaults"] == {"ward_daily": "good"}


def test_script_records_failed_models(evaluate_script, data, isolated_paths, monkeypatch):
    """A model whose evaluation raises is stored as failed and cannot become the default (E10)."""
    real, holdout = data[WARD_DAILY]["real"], data[WARD_DAILY]["holdout"]
    registry, generator = _script_fixtures(real, holdout)

    def flaky_evaluate_model(table, real_train, real_holdout, synthetic, model_id=None, **kwargs):
        if model_id == "good":
            raise RuntimeError("evaluation exploded")
        families = {f: {"name": f, "description": "stub", "score": 0.4}
                    for f in ("statistical", "validity", "operational", "privacy", "ml_utility", "optimization")}
        return {"table": table, "model_id": model_id, **families, "composite": composite_score(families),
                "timings": {"total": 0.0}}

    monkeypatch.setattr(evaluate_script, "evaluate_model", flaky_evaluate_model)
    args = evaluate_script.parse_args(["--tables", "ward_daily", "--rows", "100", "--quick", "--no-scenarios",
                                       "--no-references"])
    monkeypatch.setattr(evaluate_script, "calibrate", lambda *a, **k: None)
    outcome = evaluate_script.run(args, registry, generator)
    table = outcome["comparison"]["tables"]["ward_daily"]
    statuses = {e["model_id"]: e["status"] for e in table["models"]}
    assert statuses == {"good": "failed", "bad": "ok"} and table["best_model"] == "bad"


# ---------------------------------------------------------------------------
# scripts/03_evaluate.py: registry metrics (card["metrics"])
# ---------------------------------------------------------------------------
def _registered(registry, cards: dict[str, str], trained: bool = False) -> None:
    """Register plain model cards ``{model_id: fitted_at}`` (an empty pickle marks a model as trained)."""
    for model_id, fitted_at in cards.items():
        if trained:
            directory = registry.model_dir(WARD_DAILY, model_id)
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "synthesizer.pkl").write_bytes(b"")
        registry.register(WARD_DAILY, model_id, {"experiment_id": model_id, "fitted_at": fitted_at,
                                                 "effective_constraint_groups": ["inequality"]})


def _comparison_entry(model_id: str, composite: float | None, **extra: Any) -> dict[str, Any]:
    entry = {"model_id": model_id, "status": "ok", "composite": composite, "complete": True,
             "components": {"fidelity": 0.9, "validity": 0.8, "operational": 0.7, "utility": 0.6, "privacy": 1.0},
             "family_scores": {"scenario_coverage": 0.75},
             "run_config": {"quick": False, "rows": 7016, "rows_per_scenario": 400, "scenarios": True, "seed": 42},
             "model_fitted_at": "t1", "evaluated_at": "2026-11-14T23:11:25+00:00",
             "results_file": f"ward_daily__{model_id}.json", "comparable": True}
    entry.update(extra)
    return entry


def test_script_sync_registry_metrics_contract(evaluate_script, isolated_paths):
    """Every comparison entry's scores are copied into its registry card (fixed contract, nothing re-scored);
    entries of retrained or unregistered models are skipped; rank 1 is the selected model."""
    from bedsynth.config import project_relative
    from bedsynth.synthesis.registry import ModelRegistry

    registry = ModelRegistry()
    _registered(registry, {"good": "t1", "bad": "t1", "old": "t1", "broken": "t1", "retrained": "t-new"})
    results_dir = isolated_paths["results_dir"]
    (results_dir / "ward_daily__good.json").write_text(json.dumps({
        "validity": {"basis": "raw_model_samples", "hard_valid_rate": 0.944, "score": 0.944},
        "operational": {"n_passed": 14, "n_scored": 17, "n_checks": 17, "score": 0.89},
        "scenario_coverage": {"score": 0.5}}), encoding="utf-8")
    comparison = {"tables": {WARD_DAILY: {"models": [
        _comparison_entry("old", 0.99, comparable=False),
        _comparison_entry("good", 0.94),
        _comparison_entry("retrained", 0.97),
        _comparison_entry("bad", 0.60, run_config={"quick": True, "rows": 2000}),
        _comparison_entry("gone", 0.50),
        evaluate_script.failed_entry("broken", "RuntimeError: boom", {"quick": False, "rows": 7016}, {"fitted_at": "t1"}),
    ]}}}

    written = evaluate_script.sync_registry_metrics(registry, comparison, results_dir)
    assert written == {WARD_DAILY: ["old", "good", "bad", "broken"]}       # retrained and gone are skipped
    metrics = {m: registry.get_card(WARD_DAILY, m).get("metrics") for m in ("good", "bad", "old", "broken", "retrained")}
    assert metrics["retrained"] is None
    good = metrics["good"]
    assert list(good) == list(evaluate_script.REGISTRY_METRIC_KEYS)
    assert good == {"composite": 0.94, "complete": True, "fidelity": 0.9, "validity": 0.8, "operational": 0.7,
                    "utility": 0.6, "privacy": 1.0, "raw_hard_valid_rate": 0.944, "operational_passed": 14,
                    "operational_scored": 17, "scenario_coverage": 0.75, "rank": 1, "comparable": True,
                    "evaluated_at": "2026-11-14T23:11:25+00:00", "run_quick": False, "rows": 7016,
                    "source": project_relative(results_dir / "model_comparison.json")}
    # no result file: those values stay null; the run configuration is the entry's own
    assert metrics["bad"]["raw_hard_valid_rate"] is None and metrics["bad"]["operational_passed"] is None
    assert metrics["bad"]["rank"] == 2 and metrics["bad"]["run_quick"] is True and metrics["bad"]["rows"] == 2000
    assert metrics["old"]["comparable"] is False and metrics["old"]["rank"] == 3        # after the eligible entries
    assert metrics["broken"]["complete"] is False and metrics["broken"]["composite"] is None
    assert metrics["broken"]["rank"] == 4 and metrics["broken"]["fidelity"] is None
    assert all(isinstance(m["rank"], int) and isinstance(m["evaluated_at"], str) for m in metrics.values() if m)
    # registries without update_card (test fakes) are left alone
    assert evaluate_script.sync_registry_metrics(FakeRegistry({}, WARD_DAILY), comparison, results_dir) == {}


def test_script_sync_registry_mode_reads_only_stored_results(evaluate_script, isolated_paths, monkeypatch, capsys):
    """--sync-registry evaluates nothing: it copies model_comparison.json into the registry cards."""
    from bedsynth.synthesis.registry import ModelRegistry

    monkeypatch.setattr(evaluate_script, "run", lambda *a, **k: pytest.fail("--sync-registry must not evaluate"))
    assert evaluate_script.main(["--sync-registry"]) == 1                  # no comparison file yet
    assert "model_comparison.json not found" in capsys.readouterr().err
    registry = ModelRegistry()
    _registered(registry, {"good": "t1"})
    (isolated_paths["results_dir"] / "model_comparison.json").write_text(
        json.dumps({"tables": {WARD_DAILY: {"models": [_comparison_entry("good", 0.9)]}}}), encoding="utf-8")
    assert evaluate_script.main(["--sync-registry", "--tables", WARD_DAILY]) == 0
    assert "registry metrics written for 1 model(s): good" in capsys.readouterr().out
    assert ModelRegistry().get_card(WARD_DAILY, "good")["metrics"]["composite"] == 0.9
    assert evaluate_script.parse_args(["--sync-registry"]).sync_registry is True


def test_script_run_writes_registry_metrics_and_effective_groups(evaluate_script, data, isolated_paths, monkeypatch):
    """run() with a real registry: defaults, then card["metrics"] for every evaluated model; the result
    files' model-card summary carries effective_constraint_groups (CODE-04)."""
    from bedsynth.synthesis.registry import ModelRegistry

    real, holdout = data[WARD_DAILY]["real"], data[WARD_DAILY]["holdout"]
    _, generator = _script_fixtures(real, holdout)
    registry = ModelRegistry()
    _registered(registry, {"good": "t1", "bad": "t2"}, trained=True)

    def fake_evaluate_model(table, real_train, real_holdout, synthetic, raw_synthetic=None, model_id=None, **kwargs):
        score = 0.9 if model_id == "good" else 0.5
        families = {f: {"name": f, "description": "stub", "score": score}
                    for f in ("statistical", "operational", "privacy", "ml_utility", "optimization")}
        families["validity"] = {"name": "validity", "description": "stub", "score": score,
                                "basis": "raw_model_samples", "hard_valid_rate": score}
        families["operational"].update(n_passed=3, n_scored=4)
        families["scenario_coverage"] = {"name": "scenario_coverage", "description": "stub", "score": 0.7}
        return {"table": table, "model_id": model_id, "n_rows": {"synthetic": len(synthetic)}, **families,
                "composite": composite_score(families), "timings": {"total": 0.0},
                "evaluated_at": "2026-11-15T00:00:00+00:00"}

    monkeypatch.setattr(evaluate_script, "evaluate_model", fake_evaluate_model)
    monkeypatch.setattr(evaluate_script, "draw_raw_samples",
                        lambda reg, table, model_id, n_rows, seed, generator=None: real.sample(n_rows, replace=True,
                                                                                             random_state=seed))
    args = evaluate_script.parse_args(["--tables", "ward_daily", "--rows", "120", "--quick", "--no-scenarios",
                                       "--no-references"])
    monkeypatch.setattr(evaluate_script, "calibrate", lambda *a, **k: None)
    outcome = evaluate_script.run(args, registry, generator)

    assert outcome["defaults"] == {"ward_daily": "good"} and ModelRegistry().default_model_id(WARD_DAILY) == "good"
    entries = {e["model_id"]: e for e in outcome["comparison"]["tables"][WARD_DAILY]["models"]}
    for model_id, rank in (("good", 1), ("bad", 2)):
        metrics = ModelRegistry().get_card(WARD_DAILY, model_id)["metrics"]
        assert metrics["rank"] == rank and metrics["composite"] == entries[model_id]["composite"]
        assert metrics["operational_passed"] == 3 and metrics["operational_scored"] == 4
        assert metrics["raw_hard_valid_rate"] == entries[model_id]["components"]["validity"]
        assert metrics["run_quick"] is True and metrics["rows"] == 120 and metrics["comparable"] is True
        assert metrics["scenario_coverage"] == 0.7
    stored = json.loads((isolated_paths["results_dir"] / "ward_daily__good.json").read_text(encoding="utf-8"))
    assert stored["model_card"]["effective_constraint_groups"] == ["inequality"]
    assert entries["good"]["model_card"]["effective_constraint_groups"] == ["inequality"]


# ---------------------------------------------------------------------------
# Week-3 report sections (bedsynth.reporting.week3, rendered by 03_evaluate)
# ---------------------------------------------------------------------------
def _report_inputs(model_ids: tuple[str, ...]) -> dict[str, Any]:
    """Minimal 03_evaluate-style results: models with cards, output validity and references."""
    def family(score: float, **extra: Any) -> dict[str, Any]:
        return {"score": score, **extra}

    checks = [{"name": "weekly_elective_profile", "group": "scheduling", "metric": "wape", "value": 0.2, "score": 0.8,
               "passed": False, "threshold": 0.1, "threshold_used": 0.1, "within_real_noise": False,
               "real_reference": {"noise_limit": 0.07}, "description": "Day-of-week profile of elective bookings."},
              {"name": "occupancy_band_mix_by_ward", "group": "occupancy", "metric": "mean_tvd", "value": 0.0,
               "score": 1.0, "passed": True, "strata_forced": True, "real_reference": {"noise_limit": 0.06}}]
    results, entries = {}, []
    for rank, model_id in enumerate(model_ids):
        score = 0.5 + 0.1 * rank
        components = {"fidelity": score, "validity": score, "operational": score, "utility": score, "privacy": 1.0}
        results[f"{WARD_DAILY}__{model_id}"] = {
            "table": WARD_DAILY, "model_id": model_id, "n_rows": {"synthetic": 100, "real_train": 100, "real_holdout": 20},
            "statistical": family(score, summary={"worst_numeric_columns": ["census_start"]}),
            "validity": family(score, worst_hard_rules=[{"rule_id": "WD-H06", "violation_rate": 0.1}]),
            "output_validity": family(1.0, soft_violation_rates={"WD-S03": 0.11}),
            "operational": family(score, n_checks=1, n_passed=0, failed_checks=["weekly_elective_profile"],
                                  system_checks=["occupancy_band_mix_by_ward"], checks=checks),
            "optimization": family(0.9, components={"booking_quota": 0.9, "policy_evaluation": 0.9, "capacity_planning": 0.9},
                                   booking_quota={"score": 0.9, "relative_regret": 0.06, "real_decision_regret_vs_oracle": 0.03,
                                                  "quota_agreement": 0.5, "regret_basis": "out_of_sample_holdout"},
                                   policy_evaluation={"score": 0.95, "objective_accuracy": 0.95, "kendall_tau": 1.0,
                                                      "real_ranking_monotone": True},
                                   capacity_planning={"score": 0.99, "mean_relative_error": 0.01}),
            "composite": {"score": score, "components": components},
            "generation": {"acceptance_rate": 0.9, "strata_mix_tvd": 0.0, "allocation_strata": {"rebalanced_rows": 0}},
            "run": {"quick": True, "rows": 100, "rows_per_scenario": 100, "scenarios": True, "seed": 42, "sdmetrics": False},
            "model_card": {"segment_label": "global" if model_id == "gc_baseline" else "ward_id", "quick": True,
                           "constraint_groups": [] if model_id == "gc_baseline" else ["inequality", "capacity"]},
        }
        entries.append({"model_id": model_id, "status": "ok", "composite": score, "components": components,
                        "results_file": f"{WARD_DAILY}__{model_id}.json"})
    references = [{"model_id": REAL_REFERENCE_ID, "kind": "ceiling", "status": "ok", "composite": 0.98,
                   "components": {c: 0.98 for c in ("fidelity", "validity", "operational", "utility", "privacy")}},
                  {"model_id": NULL_REFERENCE_ID, "kind": "floor", "status": "ok", "composite": 0.6,
                   "components": {"fidelity": 0.9, "validity": 0.2, "operational": 0.6, "utility": 0.6, "privacy": 1.0}}]
    real_ref = {"validity": {"soft_rules": [{"rule_id": "WD-S03", "description": "Boarders only above 90 %."}],
                             "soft_violation_rates": {"WD-S03": 0.0}},
                "operational": {"checks": [{**checks[0], "value": 0.05, "passed": True}]}}
    best = model_ids[-1]
    comparison = {"generated_at": "2026-01-01T00:00:00+00:00", "composite_weights": {"fidelity": 0.3},
                  "tables": {WARD_DAILY: {"models": entries, "references": references, "best_model": best,
                                          "best_composite": entries[-1]["composite"],
                                          "run_config": results[f"{WARD_DAILY}__{best}"]["run"],
                                          "noise_calibration": {"n": 100, "reps": 50, "real_pass_rate_cv": 0.96}}}}
    return {"comparison": comparison, "results": results, "references": {WARD_DAILY: {REAL_REFERENCE_ID: real_ref}}}


def test_week3_report_has_the_required_sections(isolated_paths):
    from bedsynth.reporting import render_week3_report

    text = render_week3_report(_report_inputs(("gc_baseline", "gc_segmented", "gc_tuned"))).read_text(encoding="utf-8")
    for heading in ("## 1. Run configuration", "## 2. Model comparison", "## 3. Improvements made to SDV generation",
                    "## 4. Validity: raw model samples vs delivered output", "## 5. Scheduling relationships",
                    "## 6. Optimisation-model testing", "## 7. Quality gaps", "## 8. Model details"):
        assert heading in text
    assert "Caveat - quick evaluation" in text and "Caveat - quick-trained models" in text
    assert "_real holdout (ceiling)_" in text and "_no-learning baseline (floor)_" in text
    assert "gc_baseline -> gc_segmented" in text and "gc_segmented -> gc_tuned" in text
    assert "raw hard-valid" in text and "output hard-valid" in text and "WD-S03" in text
    assert "real noise limit" in text and "weekly_elective_profile" in text
    assert "quota regret (holdout)" in text and "diagnostic only" in text
    assert "system check" in text


def test_week3_report_degrades_without_gc_segmented(isolated_paths):
    from bedsynth.reporting import render_week3_report

    text = render_week3_report(_report_inputs(("gc_baseline", "gc_tuned"))).read_text(encoding="utf-8")
    assert "gc_baseline -> gc_tuned" in text and "gc_segmented has no result" in text
    single = render_week3_report(_report_inputs(("gc_tuned",))).read_text(encoding="utf-8")
    assert "Ablation not available" in single


@pytest.mark.slow
def test_script_run_with_real_evaluation(evaluate_script, data, isolated_paths):
    """Full script run on fakes with the real evaluate_model (~15 s)."""
    real, holdout = data[WARD_DAILY]["real"], data[WARD_DAILY]["holdout"]
    registry, generator = _script_fixtures(real, holdout)
    args = evaluate_script.parse_args(["--tables", "ward_daily", "--rows", "300", "--quick"])

    outcome = evaluate_script.run(args, registry, generator)

    results_dir = isolated_paths["results_dir"]
    for model_id in ("good", "bad"):
        stored = json.loads((results_dir / f"ward_daily__{model_id}.json").read_text(encoding="utf-8"))
        assert stored["model_id"] == model_id and stored["n_rows"]["synthetic"] == 300
        assert stored["generation"]["generated"] == 300
    comparison = json.loads((results_dir / "model_comparison.json").read_text(encoding="utf-8"))
    table = comparison["tables"]["ward_daily"]
    assert table["best_model"] == "good"
    assert {e["model_id"] for e in table["models"]} == {"good", "bad"}
    assert all(set(e["components"]) == {"fidelity", "validity", "operational", "utility", "privacy"}
               for e in table["models"])
    assert {r["model_id"] for r in table["references"]} == {"real_holdout", "null_stratified_marginals"}
    assert (results_dir / "ward_daily__reference__real_holdout.json").exists()
    assert all(e.get("normalised") is not None for e in table["models"])
    assert table["noise_calibration"]["n"] == 300
    assert registry.defaults["ward_daily"][0] == "good" and "composite" in registry.defaults["ward_daily"][1]
    assert outcome["defaults"] == {"ward_daily": "good"}
    assert isinstance(args, argparse.Namespace)


@pytest.mark.slow
def test_script_run_with_the_real_engine(evaluate_script, data, isolated_paths):
    """03_evaluate end to end with the real registry, generator, evaluation and Week-3 report."""
    from bedsynth.synthesis.generator import SyntheticDataGenerator
    from bedsynth.synthesis.registry import ModelRegistry
    from bedsynth.synthesis.training import train_experiment

    real, holdout = data[WARD_DAILY]["real"], data[WARD_DAILY]["holdout"]
    io.write_table(real, io.processed_path(WARD_DAILY, "train"))
    io.write_table(holdout, io.processed_path(WARD_DAILY, "holdout"))
    train_experiment(WARD_DAILY, "gc_tuned", train_df=real, quick=True)
    args = evaluate_script.parse_args(["--tables", "ward_daily", "--rows", "300", "--quick"])

    outcome = evaluate_script.run(args, ModelRegistry(), SyntheticDataGenerator())

    results_dir = isolated_paths["results_dir"]
    stored = json.loads((results_dir / "ward_daily__gc_tuned.json").read_text(encoding="utf-8"))
    assert stored["generation"]["generated"] == 300 and stored["validity"]["basis"] == "raw_model_samples"
    assert stored["validity"]["n_rows"] == 300
    for family in ("statistical", "validity", "operational", "privacy", "ml_utility", "optimization"):
        assert "error" not in stored[family], (family, stored[family].get("error"))
    coverage = stored["scenario_coverage"]
    assert coverage["n_scenarios"] == len(scenarios_for(WARD_DAILY))
    assert coverage["n_succeeded"] == coverage["n_evaluated"] >= 1 and coverage["score"] > 0.5
    assert all(s.get("time_budget_seconds") is not None for s in coverage["scenarios"] if s["status"] == "evaluated")
    assert stored["generation"]["time_budget_seconds"] is not None
    assert stored["model_view"]["basis"] == "raw_model_samples" and stored["output_validity"]["score"] == 1.0
    composite = stored["composite"]
    assert 0.0 < composite["score"] <= 1.0 and composite["missing_components"] == [] and composite["complete"]
    assert outcome["defaults"] == {"ward_daily": "gc_tuned"}
    assert ModelRegistry().default_model_id("ward_daily") == "gc_tuned"
    report = (isolated_paths["reports_dir"] / "week3_evaluation.md").read_text(encoding="utf-8")
    assert "ward_daily / gc_tuned" in report and "Scenario coverage" in report
    assert "real holdout (ceiling)" in report and "## 7. Quality gaps" in report
