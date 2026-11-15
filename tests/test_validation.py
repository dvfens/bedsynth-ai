"""Tests for the operational rule engine (bedsynth.validation, DESIGN.md §6.4)."""
from __future__ import annotations

import json
import math
import time
from typing import Any, Callable

import numpy as np
import pandas as pd
import pytest

from bedsynth import validation
from bedsynth.schema import PATIENT_STAYS, TABLES, WARD_DAILY, WARDS
from bedsynth.validation import (
    Rule,
    ValidationReport,
    get_rule,
    get_rules,
    hard_valid_mask,
    rules_catalog,
    validate_table,
)

EXPECTED_HARD_RULES: dict[str, dict[str, str]] = {
    WARD_DAILY: {
        "WD-H01": "valid_categories",
        "WD-H02": "non_negative_integer_counts",
        "WD-H03": "cancellations_within_bookings",
        "WD-H04": "noon_discharges_within_discharges",
        "WD-H05": "delayed_within_census",
        "WD-H06": "census_within_available_beds",
        "WD-H07": "census_start_within_physical_beds",
        "WD-H08": "capacity_limits",
        "WD-H09": "staffing_present",
        "WD-H10": "occupancy_level_consistent",
        "WD-H11": "staffing_level_consistent",
        "WD-H12": "ward_admission_profile",
        "WD-H13": "metric_ranges",
        "WD-H14": "wait_definition",
        "WD-H15": "closures_only_free_beds",
    },
    PATIENT_STAYS: {
        "PS-H01": "valid_categories",
        "PS-H02": "hours_in_range",
        "PS-H03": "non_negative_values",
        "PS-H04": "positive_length_of_stay",
        "PS-H05": "elective_booking_lead",
        "PS-H06": "non_elective_unbooked",
        "PS-H07": "delay_reason_consistent",
        "PS-H08": "delay_within_stay",
        "PS-H09": "separation_destination_consistent",
        "PS-H10": "delay_reason_allowed",
        "PS-H11": "ward_admission_profile",
        "PS-H12": "metric_ranges",
        "PS-H13": "delay_threshold",
        "PS-H14": "rebooking_lead",
    },
}
EXPECTED_SOFT_RULES: dict[str, list[str]] = {
    WARD_DAILY: [f"WD-S0{i}" for i in range(1, 7)],
    PATIENT_STAYS: [f"PS-S0{i}" for i in range(1, 6)],
}


def _fixture(table: str, ward_daily_df: pd.DataFrame, patient_stays_df: pd.DataFrame) -> pd.DataFrame:
    return (ward_daily_df if table == WARD_DAILY else patient_stays_df).copy()


def _with_changes(df: pd.DataFrame, row: Any, changes: dict[str, Any]) -> pd.DataFrame:
    """Copy of ``df`` with ``changes`` written into one row (columns become object dtype)."""
    out = df.copy()
    for column, value in changes.items():
        out[column] = out[column].astype(object)
        out.at[row, column] = value
    return out


def _first_row(df: pd.DataFrame, where: Callable[[pd.DataFrame], pd.Series] | None) -> Any:
    rows = df.index if where is None else df.index[where(df).to_numpy()]
    assert len(rows), "fixture has no row matching the case filter"
    return rows[0]


# ---------------------------------------------------------------------------
# Catalogue
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("table", TABLES)
def test_hard_rule_ids_and_names_match_the_design(table):
    hard = get_rules(table, "hard")
    assert {r.rule_id: r.name for r in hard} == EXPECTED_HARD_RULES[table]
    assert [r.rule_id for r in hard] == sorted(EXPECTED_HARD_RULES[table])


@pytest.mark.parametrize("table", TABLES)
def test_soft_rules_are_catalogued(table):
    soft = get_rules(table, "soft")
    assert [r.rule_id for r in soft] == EXPECTED_SOFT_RULES[table]
    assert all(r.severity == "soft" and r.table == table for r in soft)
    assert len({r.name for r in get_rules(table)}) == len(get_rules(table))  # names unique per table


@pytest.mark.parametrize("table", TABLES)
def test_rules_catalog_is_json_serialisable(table):
    catalog = rules_catalog(table)
    assert json.loads(json.dumps(catalog)) == catalog
    assert len(catalog) == len(get_rules(table))
    for entry in catalog:
        assert set(entry) == {"rule_id", "table", "severity", "name", "description", "columns"}
        assert entry["description"].endswith(".") and entry["columns"]


def test_get_rules_filters_and_rejects_bad_arguments():
    assert len(get_rules(WARD_DAILY)) == len(get_rules(WARD_DAILY, "hard")) + len(get_rules(WARD_DAILY, "soft"))
    assert get_rule("PS-H09").name == "separation_destination_consistent"
    with pytest.raises(KeyError):
        get_rules("unknown_table")
    with pytest.raises(ValueError):
        get_rules(WARD_DAILY, "medium")
    with pytest.raises(KeyError):
        get_rule("XX-H01")


def test_rule_check_returns_boolean_series_aligned_to_index(ward_daily_df):
    df = ward_daily_df.head(10).set_axis(range(100, 110))
    result = get_rule("WD-H03").check(df)
    assert isinstance(result, pd.Series) and result.dtype == bool
    assert result.index.equals(df.index) and result.all()


# ---------------------------------------------------------------------------
# Hard rules: pass on fixtures, catch hand-made violations
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("table", TABLES)
def test_every_hard_rule_passes_on_fixture_data(table, ward_daily_df, patient_stays_df):
    report = validate_table(_fixture(table, ward_daily_df, patient_stays_df), table)
    assert report.all_hard_passed
    assert report.violation_counts("hard") == dict.fromkeys(EXPECTED_HARD_RULES[table], 0)
    assert hard_valid_mask(_fixture(table, ward_daily_df, patient_stays_df), table).all()


def _elective_ward(df: pd.DataFrame) -> pd.Series:
    return df["ward_id"].isin(["GSUR", "ORTH", "CARD"]) & (df["elective_scheduled"] >= 1)


# (rule_id, row filter, changes) - each change set breaks the named hard rule on one row.
WARD_DAILY_VIOLATIONS: list[tuple[str, Any, dict[str, Any]]] = [
    ("WD-H01", None, {"occupancy_level": "busy"}),
    ("WD-H01", None, {"ward_id": "XYZ"}),
    ("WD-H01", None, {"month": 13}),
    ("WD-H02", None, {"emergency_boarders": -1}),
    ("WD-H02", None, {"admissions_elective": 2.5}),
    ("WD-H03", _elective_ward, {"elective_cancelled": 99}),
    ("WD-H04", lambda d: d["discharges"] >= 1, {"discharges_before_noon": 99}),
    ("WD-H05", None, {"delayed_discharges": 999}),
    ("WD-H06", None, {"census_end": 999}),
    ("WD-H06", None, {"census_end": -1}),
    ("WD-H07", lambda d: d["ward_id"] == "GMED", {"census_start": 40}),
    ("WD-H08", lambda d: d["ward_id"] == "ICU", {"escalation_beds_open": 3}),
    ("WD-H08", None, {"beds_closed": -1}),
    ("WD-H08", None, {"available_beds": 0}),
    ("WD-H09", None, {"nurses_night_shift": 0}),
    ("WD-H10", lambda d: d["occupancy_level"] != "low", {"occupancy_level": "low"}),
    ("WD-H11", lambda d: d["staffing_level"] != "surplus", {"staffing_level": "surplus"}),
    ("WD-H12", lambda d: d["ward_id"] == "AMU", {"elective_scheduled": 1}),
    ("WD-H12", lambda d: d["ward_id"] == "PAED", {"transfers_in": 1}),
    ("WD-H13", None, {"avg_bed_turnaround_min": 5.0}),
    ("WD-H13", None, {"avg_los_days": 120.0}),
    ("WD-H13", None, {"avg_bed_wait_hours": 100.0}),
    ("WD-H14", lambda d: d["admissions_emergency"] >= 1, {"admissions_emergency": 0, "avg_bed_wait_hours": 2.5}),
    # an occupied bed was closed: GMED's 32 patients fit its physical beds (WD-H07) but not the available ones
    ("WD-H15", lambda d: d["ward_id"].eq("GMED") & (d["beds_closed"] >= 1) & (d["escalation_beds_open"] == 0),
     {"census_start": 32}),
]

PATIENT_STAYS_VIOLATIONS: list[tuple[str, Any, dict[str, Any]]] = [
    ("PS-H01", None, {"discharge_destination": "the moon"}),
    ("PS-H01", None, {"admission_month": 0}),
    ("PS-H02", None, {"admission_hour": 24}),
    ("PS-H02", None, {"discharge_hour": 7.5}),
    ("PS-H03", None, {"bed_wait_hours": -0.5}),
    ("PS-H03", None, {"times_rescheduled": 1.5}),
    ("PS-H03", None, {"bed_turnaround_minutes": None}),
    ("PS-H04", None, {"los_hours": 0}),
    ("PS-H05", lambda d: d["admission_source"] == "elective", {"booking_lead_days": 0}),
    ("PS-H06", lambda d: d["admission_source"] == "emergency", {"booking_lead_days": 3}),
    ("PS-H06", lambda d: d["admission_source"] == "transfer", {"times_rescheduled": 1}),
    ("PS-H07", lambda d: d["delay_reason"] == "none", {"discharge_delay_hours": 5.0}),
    ("PS-H07", lambda d: d["delay_reason"] != "none", {"discharge_delay_hours": 0.0}),
    ("PS-H08", lambda d: d["delay_reason"] != "none", {"discharge_delay_hours": 10_000.0}),
    ("PS-H09", lambda d: d["separation_type"] == "transfer_out", {"discharge_destination": "home"}),
    ("PS-H09", lambda d: d["separation_type"] == "discharge", {"discharge_destination": "internal_ward"}),
    ("PS-H10", lambda d: d["discharge_destination"] == "home", {"delay_reason": "awaiting_care_package"}),
    ("PS-H11", lambda d: d["ward_id"] == "AMU", {"admission_source": "elective"}),
    ("PS-H11", lambda d: d["ward_id"] == "PAED", {"admission_source": "transfer"}),
    ("PS-H12", None, {"bed_turnaround_minutes": 5}),
    ("PS-H12", None, {"bed_turnaround_minutes": 2000}),
    ("PS-H12", None, {"los_nights": 200}),
    ("PS-H12", lambda d: d["admission_source"] == "elective", {"booking_lead_days": 400}),
    ("PS-H12", lambda d: d["admission_source"] == "elective", {"times_rescheduled": 11}),
    ("PS-H12", None, {"bed_wait_hours": 120.0}),
    ("PS-H13", lambda d: (d["delay_reason"] != "none") & (d["los_hours"] >= 1), {"discharge_delay_hours": 0.7}),
    ("PS-H13", lambda d: (d["delay_reason"] != "none") & (d["los_hours"] >= 4), {"discharge_delay_hours": 3.9}),
    ("PS-H14", lambda d: (d["admission_source"] == "elective") & (d["booking_lead_days"] < 300),
     {"times_rescheduled": 1, "booking_lead_days": 7}),
    ("PS-H14", lambda d: (d["admission_source"] == "elective") & (d["booking_lead_days"] < 300),
     {"times_rescheduled": 3, "booking_lead_days": 21}),
]


@pytest.mark.parametrize(
    ("table", "rule_id", "where", "changes"),
    [(WARD_DAILY, *case) for case in WARD_DAILY_VIOLATIONS] + [(PATIENT_STAYS, *case) for case in PATIENT_STAYS_VIOLATIONS],
    ids=[f"{c[0]}-{next(iter(c[2]))}" for c in WARD_DAILY_VIOLATIONS + PATIENT_STAYS_VIOLATIONS],
)
def test_hard_rule_catches_hand_made_violation(table, rule_id, where, changes, ward_daily_df, patient_stays_df):
    base = _fixture(table, ward_daily_df, patient_stays_df)
    row = _first_row(base, where)
    df = _with_changes(base, row, changes)
    report = validate_table(df, table)
    result = next(r for r in report.rule_results if r["rule_id"] == rule_id)
    assert result["violations"] == 1
    assert result["example_indices"] == [row]
    assert not report.hard_valid_mask[row]
    assert report.hard_valid_mask.drop(index=row).all()


def test_every_hard_rule_has_a_violation_case():
    covered = {case[0] for case in WARD_DAILY_VIOLATIONS + PATIENT_STAYS_VIOLATIONS}
    assert covered == set(EXPECTED_HARD_RULES[WARD_DAILY]) | set(EXPECTED_HARD_RULES[PATIENT_STAYS])


def _rule_passes(rule_id: str, df: pd.DataFrame) -> list[bool]:
    return get_rule(rule_id).check(df).tolist()


def test_delay_threshold_and_rebooking_boundaries():
    """D1 regression: delays are 0 or >= 4 h; a stay rescheduled r times was booked >= 1 + 7r days ahead."""
    delays = pd.DataFrame({"discharge_delay_hours": [0.0, 4.0, 4.1, 3.99, 0.1, np.nan, -4.0]})
    assert _rule_passes("PS-H13", delays) == [True, True, True, False, False, False, False]
    stays = pd.DataFrame({
        "admission_source": ["elective", "elective", "elective", "elective", "emergency", "elective"],
        "booking_lead_days": [1, 8, 7, 22, 0, np.nan],
        "times_rescheduled": [0, 1, 1, 3, 0, 0],
    })
    assert _rule_passes("PS-H14", stays) == [True, True, False, True, True, False]


def test_closed_beds_must_be_free_beds():
    """D1 regression: census_start may use escalation beds (WD-H07) but never a closed bed (WD-H15)."""
    days = pd.DataFrame({
        "census_start": [30, 30, 31, 33],
        "bed_capacity": [32, 32, 32, 32],
        "escalation_beds_open": [0, 0, 0, 2],
        "available_beds": [32, 30, 30, 33],
    })
    assert _rule_passes("WD-H07", days) == [True, True, True, True]
    assert _rule_passes("WD-H15", days) == [True, True, False, True]


@pytest.mark.parametrize("value", [True, False, np.bool_(True)])
def test_booleans_are_never_valid_category_codes(ward_daily_df, value):
    """A3 regression: True == 1 must not pass as month 1 (nor False as an unknown month)."""
    df = _with_changes(ward_daily_df.head(5), 0, {"month": value})
    report = validate_table(df, WARD_DAILY)
    assert report.violation_counts()["WD-H01"] == 1 and not report.hard_valid_mask.iloc[0]
    as_bool_column = ward_daily_df.head(3).assign(month=True)
    assert validate_table(as_bool_column, WARD_DAILY).violation_counts()["WD-H01"] == 3


# ---------------------------------------------------------------------------
# Soft rules
# ---------------------------------------------------------------------------
SOFT_VIOLATIONS: list[tuple[str, str, Any, dict[str, Any]]] = [
    (WARD_DAILY, "WD-S01", None, {"nurses_night_shift": 99}),
    (WARD_DAILY, "WD-S02", lambda d: d["ward_id"] == "GMED", {"patients_per_nurse": 15.0}),
    (WARD_DAILY, "WD-S03", lambda d: d["emergency_boarders"] == 0, {"emergency_boarders": 2, "occupancy_rate": 0.5}),
    (WARD_DAILY, "WD-S04", lambda d: d["escalation_beds_open"] == 0, {"escalation_beds_open": 1, "occupancy_rate": 0.6}),
    (WARD_DAILY, "WD-S05", lambda d: d["is_weekend"] & d["ward_id"].eq("GSUR"), {"elective_scheduled": 3}),
    (WARD_DAILY, "WD-S06", lambda d: d["discharges"] >= 5, {"discharges_before_noon": 5, "discharges": 5}),
    (PATIENT_STAYS, "PS-S01", lambda d: d["admission_source"] == "elective", {"admission_hour": 3}),
    (PATIENT_STAYS, "PS-S02", None, {"discharge_hour": 2}),
    (PATIENT_STAYS, "PS-S03", lambda d: d["admission_source"] == "elective", {"bed_wait_hours": 6.5}),
    (PATIENT_STAYS, "PS-S04", lambda d: d["ward_id"] == "PAED",
     {"separation_type": "transfer_out", "discharge_destination": "internal_ward"}),
    (PATIENT_STAYS, "PS-S05", lambda d: d["discharge_destination"] == "home", {"discharge_delay_hours": 80.0}),
]


@pytest.mark.parametrize(("table", "rule_id", "where", "changes"), SOFT_VIOLATIONS, ids=[c[1] for c in SOFT_VIOLATIONS])
def test_soft_rule_flags_plausibility_issue(table, rule_id, where, changes, ward_daily_df, patient_stays_df):
    base = _fixture(table, ward_daily_df, patient_stays_df)
    clean = validate_table(base, table).violation_counts("soft")[rule_id]
    row = _first_row(base, where)
    report = validate_table(_with_changes(base, row, changes), table)
    result = next(r for r in report.rule_results if r["rule_id"] == rule_id)
    assert result["severity"] == "soft"
    assert result["violations"] == clean + 1
    assert report.hard_valid_mask.drop(index=row).all()  # soft findings never reject other rows


def test_soft_rules_never_affect_the_hard_valid_mask(ward_daily_df):
    df = _with_changes(ward_daily_df, ward_daily_df.index[0], {"nurses_night_shift": 99})
    report = validate_table(df, WARD_DAILY)
    assert report.violation_counts("soft")["WD-S01"] >= 1
    assert report.all_hard_passed


def test_include_soft_false_evaluates_only_hard_rules(patient_stays_df):
    report = validate_table(patient_stays_df, PATIENT_STAYS, include_soft=False)
    assert {r["severity"] for r in report.rule_results} == {"hard"}
    assert len(report.rule_results) == len(EXPECTED_HARD_RULES[PATIENT_STAYS])


# ---------------------------------------------------------------------------
# Robustness: NaN, unknown categories, wrong dtypes, absent columns
# ---------------------------------------------------------------------------
def test_nan_unknown_and_wrong_dtypes_fail_without_raising(ward_daily_df):
    df = ward_daily_df.head(50).copy()
    broken = {
        0: {"census_start": "abc"},           # unparseable number
        1: {"ward_id": None},                  # missing category
        2: {"month": "thirteen"},              # wrong dtype
        3: {"day_of_week": 5},                 # number where a label is expected
        4: {"occupancy_level": np.nan},
        5: {"nurses_day_shift": pd.NA},
        6: {"avg_los_days": float("inf")},
        7: {"staffing_level": ["short"]},      # unhashable cell
    }
    for row, changes in broken.items():
        df = _with_changes(df, row, changes)
    report = validate_table(df, WARD_DAILY)
    assert not report.hard_valid_mask.iloc[:8].any()
    assert report.hard_valid_mask.iloc[8:].all()
    assert report.violation_counts()["WD-H01"] == 5   # rows 1, 2, 3, 4, 7


def test_string_typed_frame_is_coerced_not_rejected(patient_stays_df):
    as_text = patient_stays_df.head(200).astype(str)
    report = validate_table(as_text, PATIENT_STAYS)
    assert report.all_hard_passed


def test_absent_columns_fail_the_rules_that_read_them(ward_daily_df):
    df = ward_daily_df.drop(columns=["census_end"])
    report = validate_table(df, WARD_DAILY)
    counts = report.violation_counts()
    assert counts["WD-H02"] == counts["WD-H06"] == counts["WD-H10"] == len(df)
    assert counts["WD-H03"] == 0 and counts["WD-H11"] == 0
    assert not report.hard_valid_mask.any()


def test_broken_check_counts_every_row_as_violation(ward_daily_df, monkeypatch):
    def explode(df: pd.DataFrame) -> pd.Series:
        raise RuntimeError("boom")

    broken = Rule("WD-H99", WARD_DAILY, "hard", "broken", "A check that raises.", ("ward_id",), explode)
    monkeypatch.setitem(validation._RULES, WARD_DAILY, validation._RULES[WARD_DAILY] + (broken,))
    report = validate_table(ward_daily_df.head(20), WARD_DAILY)
    assert report.violation_counts()["WD-H99"] == 20
    assert not report.hard_valid_mask.any()


def test_empty_frame_and_non_dataframe_input(ward_daily_df):
    report = validate_table(ward_daily_df.iloc[0:0], WARD_DAILY)
    assert report.n_rows == 0 and report.all_hard_passed and report.hard_violation_rate == 0.0
    assert all(r["violations"] == 0 and r["violation_rate"] == 0.0 for r in report.rule_results)
    with pytest.raises(TypeError):
        validate_table([{"ward_id": "AMU"}], WARD_DAILY)


def test_duplicate_index_labels_are_supported(patient_stays_df):
    df = pd.concat([patient_stays_df.head(30), patient_stays_df.head(30)])  # labels 0..29 twice
    df = _with_changes(df.reset_index(drop=True), 31, {"los_hours": 0}).set_axis(list(range(30)) * 2)
    report = validate_table(df, PATIENT_STAYS)
    assert report.n_hard_valid == 59
    assert next(r for r in report.rule_results if r["rule_id"] == "PS-H04")["example_indices"] == [1]


# ---------------------------------------------------------------------------
# Report contents
# ---------------------------------------------------------------------------
def test_report_to_dict_is_json_serialisable_with_capped_examples(ward_daily_df):
    df = ward_daily_df.copy()
    df["nurses_night_shift"] = 0                      # every row violates WD-H09
    report = validate_table(df, WARD_DAILY)
    payload = report.to_dict()
    assert json.loads(json.dumps(payload, allow_nan=False)) == payload
    result = next(r for r in payload["rule_results"] if r["rule_id"] == "WD-H09")
    assert result["violations"] == len(df) and result["violation_rate"] == 1.0
    assert result["example_indices"] == list(df.index[: validation.MAX_EXAMPLES])
    assert payload["hard_violations"]["WD-H09"] == len(df)
    assert payload["n_hard_valid"] == 0 and payload["all_hard_passed"] is False
    assert payload["rules_checked"] == {"hard": 15, "soft": 6}


def test_example_indices_use_positions_for_non_integer_index(patient_stays_df):
    df = _with_changes(patient_stays_df.head(10), 4, {"los_hours": 0}).set_index("stay_id")
    report = validate_table(df, PATIENT_STAYS)
    assert next(r for r in report.rule_results if r["rule_id"] == "PS-H04")["example_indices"] == [4]


def test_hard_valid_mask_matches_report_and_filters_frames(patient_stays_df):
    df = _with_changes(patient_stays_df.head(100), 7, {"admission_hour": 30})
    mask = hard_valid_mask(df, PATIENT_STAYS)
    assert mask.equals(validate_table(df, PATIENT_STAYS, include_soft=False).hard_valid_mask)
    assert len(df[mask]) == 99 and 7 not in df[mask].index


# ---------------------------------------------------------------------------
# Performance (runs inside the generator's rejection-sampling loop)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("table", TABLES)
def test_validating_50k_rows_takes_under_two_seconds(table, ward_daily_df, patient_stays_df):
    base = _fixture(table, ward_daily_df, patient_stays_df)
    big = pd.concat([base] * math.ceil(50_000 / len(base)), ignore_index=True).iloc[:50_000]
    timings = []
    for _ in range(2):
        start = time.perf_counter()
        report = validate_table(big, table)
        timings.append(time.perf_counter() - start)
    assert report.n_rows == 50_000 and report.all_hard_passed
    assert min(timings) < 2.0, f"validate_table took {min(timings):.2f} s for 50k rows"
    start = time.perf_counter()
    assert hard_valid_mask(big, table).all()
    assert time.perf_counter() - start < 2.0


def test_report_dataclass_properties():
    mask = pd.Series([True, False, True, True])
    report = ValidationReport(table=WARD_DAILY, n_rows=4, hard_valid_mask=mask)
    assert report.n_hard_valid == 3 and not report.all_hard_passed
    assert report.hard_violation_rate == pytest.approx(0.25)
    assert WARDS["ICU"].max_escalation_beds == 2  # WD-H08 case above relies on this
