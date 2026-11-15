"""Tests for the reference-data simulator and the dirty-data injection (DESIGN.md §6.1-6.2).

The hard operational rules of §6.4 are implemented here directly with pandas (the shared
``bedsynth.validation`` module is developed independently), so these tests pin down the
simulator contract on their own: every clean row, after the standard derivation, must pass
every hard rule.
"""
from __future__ import annotations

import json
import math
from typing import Callable

import numpy as np
import pandas as pd
import pytest

from bedsynth.config import get_settings
from bedsynth.data import io
from bedsynth.data.quality_issues import (
    ALT_DATE_FORMAT,
    ALT_DATETIME_FORMAT,
    CATEGORY_VARIANT_COLUMNS,
    INVALID_VALUES,
    LOG_COLUMNS,
    category_variant,
    inject_quality_issues,
)
from bedsynth.data.simulator import SimulationResult, calibration_summary, generate_source_data, simulate
from bedsynth.schema import (
    ADMISSION_SOURCES,
    ALLOWED_DELAY_REASONS,
    CATEGORY_ALIASES,
    DATE_FORMAT,
    DATETIME_FORMAT,
    DAYS_OF_WEEK,
    DELAY_REASONS,
    DELAY_THRESHOLD_HOURS,
    DISCHARGE_DESTINATIONS,
    OCCUPANCY_LEVELS,
    PATIENT_STAYS,
    RAW_COLUMNS,
    REBOOK_MIN_DAYS,
    SEPARATION_TYPES,
    STAFFING_LEVELS,
    WARD_DAILY,
    WARD_IDS,
    WARDS,
    add_calendar_features,
    add_stay_time_features,
    derive_patient_stays,
    derive_ward_daily,
    integer_columns,
    normalize_category,
    occupancy_levels,
    staffing_levels,
    training_columns,
)

START, END, SEED = "2023-01-01", "2023-03-31", 11       # 90 winter days: pressure is visible


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def run() -> SimulationResult:
    return simulate(START, END, seed=SEED)


@pytest.fixture(scope="module")
def ward_daily(run: SimulationResult) -> pd.DataFrame:
    return derive_ward_daily(add_calendar_features(run.ward_daily))


@pytest.fixture(scope="module")
def stays(run: SimulationResult) -> pd.DataFrame:
    return derive_patient_stays(add_stay_time_features(run.patient_stays))


@pytest.fixture(scope="module")
def quality_config() -> dict:
    return get_settings()["simulation"]["data_quality"]


@pytest.fixture(scope="module")
def injected(run: SimulationResult, quality_config: dict) -> dict:
    """Dirty copies + logs of both tables, and pristine copies to detect mutation."""
    out = {}
    rng = np.random.default_rng(123)
    for table, frame in ((WARD_DAILY, run.ward_daily), (PATIENT_STAYS, run.patient_stays)):
        pristine = frame.copy(deep=True)
        dirty, log = inject_quality_issues(frame, table, quality_config[table], rng)
        out[table] = {"clean": frame, "pristine": pristine, "dirty": dirty, "log": log}
    return out


# ---------------------------------------------------------------------------
# Hard rules of DESIGN.md §6.4 (True = row passes)
# ---------------------------------------------------------------------------
def _integral_non_negative(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    ok = pd.Series(True, index=df.index)
    for column in columns:
        values = pd.to_numeric(df[column], errors="coerce")
        ok &= values.notna() & (values == values.round()) & (values >= 0)
    return ok


def _between(series: pd.Series, low: float, high: float) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return values.between(low, high).fillna(False)


def _ward_profile_ok(d: pd.DataFrame) -> pd.Series:
    takes_electives = d["ward_id"].map({w: i.accepts_electives for w, i in WARDS.items()})
    takes_transfers = d["ward_id"].map({w: i.accepts_transfers for w, i in WARDS.items()})
    no_electives = (d["elective_scheduled"] == 0) & (d["admissions_elective"] == 0)
    return (takes_electives | no_electives) & (takes_transfers | (d["transfers_in"] == 0))


WARD_DAILY_RULES: dict[str, Callable[[pd.DataFrame], pd.Series]] = {
    "WD-H01 valid_categories": lambda d: (
        d["ward_id"].isin(WARD_IDS) & d["month"].isin(range(1, 13)) & d["day_of_week"].isin(DAYS_OF_WEEK)
        & d["staffing_level"].isin(STAFFING_LEVELS) & d["occupancy_level"].isin(OCCUPANCY_LEVELS)
    ),
    "WD-H02 non_negative_integer_counts": lambda d: _integral_non_negative(d, integer_columns(WARD_DAILY, ("train", "derived"))),
    "WD-H03 cancellations_within_bookings": lambda d: d["elective_cancelled"] <= d["elective_scheduled"],
    "WD-H04 noon_discharges_within_discharges": lambda d: d["discharges_before_noon"] <= d["discharges"],
    "WD-H05 delayed_within_census": lambda d: d["delayed_discharges"] <= d["census_start"],
    "WD-H06 census_within_available_beds": lambda d: (d["census_end"] >= 0) & (d["census_end"] <= d["available_beds"]),
    "WD-H07 census_start_within_physical_beds": lambda d: d["census_start"] <= d["bed_capacity"] + d["escalation_beds_open"],
    "WD-H08 capacity_limits": lambda d: (
        (d["escalation_beds_open"] <= d["ward_id"].map({w: i.max_escalation_beds for w, i in WARDS.items()}))
        & (d["beds_closed"] >= 0) & (d["available_beds"] >= 1)
    ),
    "WD-H09 staffing_present": lambda d: (d["nurses_day_shift"] >= 1) & (d["nurses_night_shift"] >= 1),
    "WD-H10 occupancy_level_consistent": lambda d: d["occupancy_level"] == occupancy_levels(d["census_end"], d["available_beds"]),
    "WD-H11 staffing_level_consistent": lambda d: d["staffing_level"] == staffing_levels(d["nurses_day_shift"], d["ward_id"]),
    "WD-H12 ward_admission_profile": _ward_profile_ok,
    "WD-H13 metric_ranges": lambda d: (
        _between(d["avg_bed_wait_hours"], 0, 96) & _between(d["avg_bed_turnaround_min"], 10, 1440)
        & _between(d["avg_los_days"], 0.04, 90)
    ),
    "WD-H14 wait_definition": lambda d: (d["admissions_emergency"] > 0) | (d["avg_bed_wait_hours"] == 0),
    "WD-H15 closures_only_free_beds": lambda d: d["census_start"] <= d["available_beds"],
}

PATIENT_STAYS_RULES: dict[str, Callable[[pd.DataFrame], pd.Series]] = {
    "PS-H01 valid_categories": lambda s: (
        s["ward_id"].isin(WARD_IDS) & s["admission_source"].isin(ADMISSION_SOURCES)
        & s["admission_month"].isin(range(1, 13)) & s["admission_day_of_week"].isin(DAYS_OF_WEEK)
        & s["separation_type"].isin(SEPARATION_TYPES) & s["discharge_destination"].isin(DISCHARGE_DESTINATIONS)
        & s["delay_reason"].isin(DELAY_REASONS)
    ),
    "PS-H02 hours_in_range": lambda s: (
        _integral_non_negative(s, ["admission_hour", "discharge_hour"])
        & (s["admission_hour"] <= 23) & (s["discharge_hour"] <= 23)
    ),
    "PS-H03 non_negative_values": lambda s: (
        _integral_non_negative(s, integer_columns(PATIENT_STAYS, ("train",)))
        & (s["bed_wait_hours"] >= 0) & (s["discharge_delay_hours"] >= 0)
        & s[training_columns(PATIENT_STAYS)].notna().all(axis=1)
    ),
    "PS-H04 positive_length_of_stay": lambda s: s["los_hours"] >= 1,
    "PS-H05 elective_booking_lead": lambda s: (s["admission_source"] != "elective") | (s["booking_lead_days"] >= 1),
    "PS-H06 non_elective_unbooked": lambda s: (
        (s["admission_source"] == "elective") | ((s["booking_lead_days"] == 0) & (s["times_rescheduled"] == 0))
    ),
    "PS-H07 delay_reason_consistent": lambda s: (s["delay_reason"] == "none") == (s["discharge_delay_hours"] == 0),
    "PS-H08 delay_within_stay": lambda s: s["discharge_delay_hours"] <= s["los_hours"],
    "PS-H09 separation_destination_consistent": lambda s: (
        (s["separation_type"] == "transfer_out") == (s["discharge_destination"] == "internal_ward")
    ),
    "PS-H10 delay_reason_allowed": lambda s: pd.Series(
        [r in ALLOWED_DELAY_REASONS.get(d, ()) for r, d in zip(s["delay_reason"], s["discharge_destination"])],
        index=s.index,
    ),
    "PS-H11 ward_admission_profile": lambda s: pd.Series(
        [src in WARDS[w].admission_sources for w, src in zip(s["ward_id"], s["admission_source"])], index=s.index,
    ),
    "PS-H12 metric_ranges": lambda s: (
        _between(s["bed_turnaround_minutes"], 10, 1440) & (s["bed_wait_hours"] <= 96) & (s["los_nights"] <= 180)
        & (s["booking_lead_days"] <= 365) & (s["times_rescheduled"] <= 10)
    ),
    "PS-H13 delay_threshold": lambda s: (
        (s["discharge_delay_hours"] == 0) | (s["discharge_delay_hours"] >= DELAY_THRESHOLD_HOURS)
    ),
    "PS-H14 rebooking_lead": lambda s: (
        (s["admission_source"] != "elective") | (s["booking_lead_days"] >= 1 + REBOOK_MIN_DAYS * s["times_rescheduled"])
    ),
}


# ---------------------------------------------------------------------------
# Shape and format of the clean extract
# ---------------------------------------------------------------------------
def test_columns_match_raw_schema(run: SimulationResult) -> None:
    assert list(run.ward_daily.columns) == list(RAW_COLUMNS[WARD_DAILY])
    assert list(run.patient_stays.columns) == list(RAW_COLUMNS[PATIENT_STAYS])
    for frame in (run.ward_daily, run.patient_stays):
        assert not frame.isna().any().any(), "clean output must not contain missing values"
        ints = [c for c in frame.columns if frame[c].dtype.kind in "iu"]
        assert all(frame[c].dtype == np.int64 for c in ints)


def test_ward_daily_grid_and_order(run: SimulationResult) -> None:
    wd = run.ward_daily
    expected_days = pd.date_range(START, END, freq="D").strftime(DATE_FORMAT)
    assert len(wd) == len(expected_days) * len(WARD_IDS)
    assert set(wd["date"]) == set(expected_days)
    assert not wd.duplicated(["date", "ward_id"]).any()
    assert wd.equals(wd.sort_values(["date", "ward_id"], kind="stable").reset_index(drop=True))
    pd.to_datetime(wd["date"], format=DATE_FORMAT)       # raises if any date is malformed
    assert (wd["bed_capacity"] == wd["ward_id"].map({w: i.bed_capacity for w, i in WARDS.items()})).all()


def test_stays_window_ids_and_order(run: SimulationResult) -> None:
    ps = run.patient_stays
    admitted = pd.to_datetime(ps["admission_datetime"], format=DATETIME_FORMAT)
    departed = pd.to_datetime(ps["discharge_datetime"], format=DATETIME_FORMAT)
    assert admitted.min() >= pd.Timestamp(START)
    assert admitted.max() < pd.Timestamp(END) + pd.Timedelta(days=1)
    assert admitted.is_monotonic_increasing
    assert (departed > admitted).all()
    assert ps["stay_id"].is_unique
    assert ps["stay_id"].str.fullmatch(r"ST-\d{7}").all()
    assert ps["stay_id"].iloc[0] == "ST-0000001"


# ---------------------------------------------------------------------------
# Hard-rule validity (100 % of rows after derivation)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("rule", list(WARD_DAILY_RULES))
def test_ward_daily_hard_rules(ward_daily: pd.DataFrame, rule: str) -> None:
    passed = WARD_DAILY_RULES[rule](ward_daily).fillna(False).astype(bool)
    assert passed.all(), f"{rule}: {int((~passed).sum())} violating rows, e.g.\n{ward_daily[~passed].head(3).T}"


@pytest.mark.parametrize("rule", list(PATIENT_STAYS_RULES))
def test_patient_stays_hard_rules(stays: pd.DataFrame, rule: str) -> None:
    passed = PATIENT_STAYS_RULES[rule](stays).fillna(False).astype(bool)
    assert passed.all(), f"{rule}: {int((~passed).sum())} violating rows, e.g.\n{stays[~passed].head(3).T}"


def test_shared_rule_catalogue_accepts_every_simulated_row(ward_daily: pd.DataFrame, stays: pd.DataFrame) -> None:
    """The mirror above and bedsynth.validation agree: every hard rule holds on the reference data."""
    from bedsynth.validation import get_rules, validate_table

    for table, frame, mirror in ((WARD_DAILY, ward_daily, WARD_DAILY_RULES), (PATIENT_STAYS, stays, PATIENT_STAYS_RULES)):
        report = validate_table(frame, table, include_soft=False)
        assert report.all_hard_passed, {k: v for k, v in report.violation_counts().items() if v}
        assert {key.split()[0] for key in mirror} == {rule.rule_id for rule in get_rules(table, "hard")}


# ---------------------------------------------------------------------------
# Operational coherence
# ---------------------------------------------------------------------------
def test_flow_balance_matches_derivation(run: SimulationResult, ward_daily: pd.DataFrame) -> None:
    """The simulator's census_end equals the flow-balance derivation, and admissions add up."""
    assert (run.ward_daily["census_end"] == ward_daily["census_end"]).all()
    raw = run.ward_daily
    assert (raw["admissions_elective"] == raw["elective_scheduled"] - raw["elective_cancelled"]).all()


def test_census_continuity(run: SimulationResult) -> None:
    for ward_id, rows in run.ward_daily.groupby("ward_id"):
        rows = rows.sort_values("date")
        assert (rows["census_end"].to_numpy()[:-1] == rows["census_start"].to_numpy()[1:]).all(), ward_id


def test_cross_table_admissions(run: SimulationResult) -> None:
    """ward_daily admissions per ward/day/source equal the stays admitted that day."""
    ps = run.patient_stays.assign(date=run.patient_stays["admission_datetime"].str[:10])
    counts = ps.groupby(["date", "ward_id", "admission_source"]).size().unstack(fill_value=0)
    counts = counts.reindex(columns=list(ADMISSION_SOURCES), fill_value=0)
    wd = run.ward_daily.set_index(["date", "ward_id"])
    joined = wd.join(counts, how="left").fillna(0)
    assert (joined["admissions_elective"] == joined["elective"]).all()
    assert (joined["admissions_emergency"] == joined["emergency"]).all()
    assert (joined["transfers_in"] == joined["transfer"]).all()


def test_daily_bed_wait_matches_stays(run: SimulationResult) -> None:
    """avg_bed_wait_hours is the mean wait of that day's emergency admissions."""
    ps = run.patient_stays
    emergency = ps[ps["admission_source"] == "emergency"].assign(date=ps["admission_datetime"].str[:10])
    means = emergency.groupby(["date", "ward_id"])["bed_wait_hours"].mean()
    wd = run.ward_daily.set_index(["date", "ward_id"])["avg_bed_wait_hours"]
    aligned = wd.loc[means.index]
    assert np.allclose(aligned.to_numpy(), means.to_numpy(), atol=0.011)


def test_transfers_out_match_transfers_in(run: SimulationResult) -> None:
    """Every internal transfer leaves one ward at the minute it enters another."""
    daily = run.ward_daily.groupby("date")[["transfers_in", "transfers_out"]].sum()
    assert (daily["transfers_in"] == daily["transfers_out"]).all()
    ps = run.patient_stays
    assert run.ward_daily["transfers_in"].sum() == (ps["admission_source"] == "transfer").sum()


def test_realism_signals(run: SimulationResult, stays: pd.DataFrame) -> None:
    """Visible ward differences and operational patterns (DESIGN.md §6.1)."""
    los = stays.groupby("ward_id")["los_days"].mean()
    assert los["AMU"] < 2.5 < los["ELDC"], "AMU turns over fast, elderly care stays long"
    care = stays[stays["discharge_destination"] == "care_facility"]
    home = stays[stays["discharge_destination"] == "home"]
    assert care["is_delayed_discharge"].mean() > home["is_delayed_discharge"].mean()
    electives = stays[stays["admission_source"] == "elective"]
    assert electives["admission_hour"].between(7, 14).all()
    assert (stays.loc[stays["ward_id"] == "PAED", "separation_type"] == "discharge").all()
    wd = run.ward_daily
    weekend = pd.to_datetime(wd["date"]).dt.dayofweek >= 5
    assert wd.loc[weekend, "discharges"].mean() < wd.loc[~weekend, "discharges"].mean()
    assert wd.loc[weekend, "elective_scheduled"].mean() < wd.loc[~weekend, "elective_scheduled"].mean()
    assert (wd["escalation_beds_open"] > 0).any() and (wd["beds_closed"] > 0).any()
    assert set(wd["staffing_level"]) == set(STAFFING_LEVELS)
    assert set(wd["occupancy_level"]) == set(OCCUPANCY_LEVELS)


# ---------------------------------------------------------------------------
# Realism regressions (review findings D2, D5, D6) on one simulated year
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def year() -> SimulationResult:
    return simulate("2023-01-01", "2023-12-31", seed=SEED)


def test_bed_waits_have_no_point_masses(year: SimulationResult) -> None:
    """D2: no single bed-wait value dominates a ward x source group (it was 3.00 h for 54 %)."""
    stays = year.patient_stays
    assert (stays["bed_wait_hours"] == 3.0).mean() < 0.01
    for (ward_id, source), waits in stays.groupby(["ward_id", "admission_source"])["bed_wait_hours"]:
        if len(waits) >= 200:
            assert waits.value_counts(normalize=True).iloc[0] < 0.06, (ward_id, source)
    check = next(c for c in year.calibration["checks"] if c["name"] == "bed_wait_point_mass")
    assert check["passed"] is True


def test_weekly_discharge_profile_has_no_weekend_backlog_spike(year: SimulationResult) -> None:
    """D5: Sunday is quieter than Saturday and Monday carries no 1.4x backlog spike."""
    wd = year.ward_daily
    per_day = wd.groupby(pd.to_datetime(wd["date"]).dt.dayofweek)["discharges"].mean()
    assert per_day[6] < per_day[5] < per_day[1:5].min()
    assert per_day[0] / per_day[1:5].mean() < 1.25
    checks = {c["name"]: c for c in year.calibration["checks"]}
    assert checks["monday_to_weekday_discharge_ratio"]["passed"] and checks["sunday_to_saturday_discharge_ratio"]["passed"]


def test_delayed_discharges_concentrate_on_medical_wards(year: SimulationResult) -> None:
    """D6: children rarely wait for social care; elderly and medical wards carry the delays."""
    wd = year.ward_daily
    rate = (wd["delayed_discharges"] / wd["census_start"].where(wd["census_start"] > 0)).fillna(0.0)
    by_ward = rate.groupby(wd["ward_id"]).mean()
    assert by_ward["PAED"] < 0.06 and by_ward["PAED"] < by_ward["GMED"] / 2 < by_ward["ELDC"]
    stays = year.patient_stays
    delayed_share = (stays["discharge_delay_hours"] > 0).groupby(stays["ward_id"]).mean()
    assert delayed_share["PAED"] < 0.12 < delayed_share["GMED"] < delayed_share["ELDC"]


def test_truncated_lognormal_has_no_mass_on_its_bounds() -> None:
    from bedsynth.data.simulator import _truncated_lognormal

    values = _truncated_lognormal(np.random.default_rng(0), 1.4, 0.45, 0.5, 3.0, 20_000)
    assert values.min() >= 0.5 and values.max() <= 3.0
    assert (values == 0.5).sum() == 0 and (values == 3.0).sum() == 0


def test_weekend_planner_deferrals_land_on_weekdays() -> None:
    """D5: deferred weekend reviews go to Mon-Wed, deferred services to Monday, pushes to Friday."""
    import random

    from bedsynth.data.simulator import MINUTES_PER_DAY, _Calendar, _DeparturePlanner, _weekend_setting

    sim = get_settings()["simulation"]
    sim["weekend_discharge_capacity"] = {"Sat": 0.0, "Sun": 0.0}      # always defer
    sim["friday_discharge_push"] = {"Sat": 1.0, "Sun": 1.0}           # always push when possible
    cal = _Calendar.build(pd.Timestamp("2023-01-02"), pd.Timestamp("2023-03-31"), 0, 10)   # day 0 = Monday
    planner = _DeparturePlanner(sim, cal)
    rng = random.Random(1)
    saturday, sunday = 5 * MINUTES_PER_DAY + 600, 6 * MINUTES_PER_DAY + 600
    for ready in (saturday, sunday):
        reviewed = planner._defer_weekend_review(ready, rng)
        assert planner.dow(reviewed // MINUTES_PER_DAY) in (0, 1, 2) and reviewed > ready
        departed, deferred = planner._defer_weekend_service(ready, rng)
        assert deferred and planner.dow(departed // MINUTES_PER_DAY) == 0
        pushed = planner.friday_push(admit=0, ready=ready, rng=rng)
        assert planner.dow(pushed // MINUTES_PER_DAY) == 4 and pushed < ready
    assert planner.friday_push(admit=4 * MINUTES_PER_DAY, ready=saturday, rng=rng) == saturday  # < 1 day in hospital
    assert _weekend_setting({"weekend_discharge_capacity": 0.55}, "weekend_discharge_capacity", 0.6) == (0.55, 0.55)
    assert _weekend_setting({}, "friday_discharge_push", 0.0) == (0.0, 0.0)


# ---------------------------------------------------------------------------
# Determinism and API behaviour
# ---------------------------------------------------------------------------
def test_seeded_determinism(run: SimulationResult) -> None:
    again = simulate(START, END, seed=SEED)
    pd.testing.assert_frame_equal(again.ward_daily, run.ward_daily)
    pd.testing.assert_frame_equal(again.patient_stays, run.patient_stays)
    other = simulate(START, END, seed=SEED + 1)
    assert not other.patient_stays.equals(run.patient_stays)


def test_settings_section_or_full_settings_are_equivalent() -> None:
    settings = get_settings()
    a = simulate("2023-05-01", "2023-05-20", seed=3, settings=settings)
    b = simulate("2023-05-01", "2023-05-20", seed=3, settings=settings["simulation"])
    pd.testing.assert_frame_equal(a.ward_daily, b.ward_daily)


def test_invalid_period_raises() -> None:
    with pytest.raises(ValueError):
        simulate("2023-02-01", "2023-01-01", seed=1)


def test_calibration_summary_is_json_serialisable(run: SimulationResult) -> None:
    summary = run.calibration
    json.dumps(summary)                      # raises on numpy scalars / NaN-unsafe objects
    names = {c["name"] for c in summary["checks"]}
    assert {"occupancy_AMU", "occupancy_mix_critical", "winter_boarding_days", "elective_cancellation_rate",
            "short_staffing_share", "emergency_bed_wait_p90_winter_hours"} <= names
    assert summary["targets_evaluated"] == sum(c["passed"] is not None for c in summary["checks"])
    assert summary == calibration_summary(run.ward_daily, run.patient_stays)
    json.dumps(run.diagnostics)


# ---------------------------------------------------------------------------
# Dirty-data injection
# ---------------------------------------------------------------------------
EXPECTED_ISSUES = {
    WARD_DAILY: {"category_variant", "alt_date_format", "missing_value", "invalid_value", "duplicate_row"},
    PATIENT_STAYS: {"category_variant", "alt_date_format", "missing_value", "invalid_value",
                    "timestamp_swap", "duplicate_row"},
}


def _text(value: object) -> str | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, (float, np.floating)):
        return repr(float(value))
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return str(int(value))
    return str(value)


def _row_keys(clean: pd.DataFrame, table: str) -> list[str]:
    if table == PATIENT_STAYS:
        return clean["stay_id"].tolist()
    return (clean["ward_id"] + "|" + clean["date"]).tolist()


def _deduplicate(dirty: pd.DataFrame, table: str) -> pd.DataFrame:
    if table == PATIENT_STAYS:
        return dirty.drop_duplicates(subset=["stay_id"], keep="first").reset_index(drop=True)
    return dirty.drop_duplicates(keep="first").reset_index(drop=True)


@pytest.mark.parametrize("table", [WARD_DAILY, PATIENT_STAYS])
def test_injection_produces_every_issue_type(injected: dict, table: str) -> None:
    log = injected[table]["log"]
    assert list(log.columns) == list(LOG_COLUMNS)
    assert (log["table"] == table).all()
    assert EXPECTED_ISSUES[table] <= set(log["issue_type"])
    invalid_columns = set(log.loc[log["issue_type"] == "invalid_value", "column"])
    assert invalid_columns == {column for column, _ in INVALID_VALUES[table]}


@pytest.mark.parametrize("table", [WARD_DAILY, PATIENT_STAYS])
def test_injection_log_matches_changed_cells(injected: dict, table: str) -> None:
    clean, dirty, log = injected[table]["clean"], injected[table]["dirty"], injected[table]["log"]
    assert list(dirty.columns) == list(clean.columns)
    n_duplicates = int((log["issue_type"] == "duplicate_row").sum())
    assert len(dirty) == len(clean) + n_duplicates

    deduped = _deduplicate(dirty, table)
    assert len(deduped) == len(clean)
    keys = _row_keys(clean, table)
    changed: dict[tuple[str, str], str | None] = {}
    for column in clean.columns:
        before, after = clean[column].map(_text), deduped[column].map(_text)
        for i in np.flatnonzero((before != after).to_numpy()):
            changed[(keys[i], column)] = after.iloc[i]

    logged: dict[tuple[str, str], str | None] = {}
    for entry in log[log["issue_type"] != "duplicate_row"].itertuples(index=False):
        if entry.issue_type == "timestamp_swap":
            first, second = entry.column.split("|")
            new_first, new_second = entry.injected_value.split("|")
            logged[(entry.row_key, first)], logged[(entry.row_key, second)] = new_first, new_second
        else:
            logged[(entry.row_key, entry.column)] = entry.injected_value
    assert changed == logged
    per_type = log["issue_type"].value_counts()
    assert per_type.sum() == len(log)
    assert len(log) - n_duplicates + int(per_type.get("timestamp_swap", 0)) == len(changed)


@pytest.mark.parametrize("table", [WARD_DAILY, PATIENT_STAYS])
def test_injected_values_are_recoverable(injected: dict, table: str) -> None:
    log = injected[table]["log"]
    variants = log[log["issue_type"] == "category_variant"]
    assert set(variants["column"]) == set(CATEGORY_VARIANT_COLUMNS[table])
    for entry in variants.itertuples(index=False):
        assert entry.injected_value != entry.original_value
        assert normalize_category(entry.column, entry.injected_value) == entry.original_value
    formats = log[log["issue_type"] == "alt_date_format"]
    for entry in formats.itertuples(index=False):
        source, alt = (DATE_FORMAT, ALT_DATE_FORMAT) if table == WARD_DAILY else (DATETIME_FORMAT, ALT_DATETIME_FORMAT)
        assert pd.to_datetime(entry.injected_value, format=alt) == pd.to_datetime(entry.original_value, format=source)
    sentinels = dict(INVALID_VALUES[table])
    for entry in log[log["issue_type"] == "invalid_value"].itertuples(index=False):
        assert float(entry.injected_value) == float(sentinels[entry.column])
    missing = log[log["issue_type"] == "missing_value"]
    assert missing["injected_value"].isna().all()


def test_injection_leaves_ground_truth_untouched(injected: dict) -> None:
    for table in (WARD_DAILY, PATIENT_STAYS):
        pd.testing.assert_frame_equal(injected[table]["clean"], injected[table]["pristine"])


def test_injection_is_deterministic(run: SimulationResult, quality_config: dict) -> None:
    a = inject_quality_issues(run.patient_stays, PATIENT_STAYS, quality_config[PATIENT_STAYS], np.random.default_rng(5))
    b = inject_quality_issues(run.patient_stays, PATIENT_STAYS, quality_config, np.random.default_rng(5))
    pd.testing.assert_frame_equal(a[0], b[0])
    pd.testing.assert_frame_equal(a[1], b[1])


@pytest.mark.parametrize("column", sorted(set(CATEGORY_VARIANT_COLUMNS[WARD_DAILY]) | set(CATEGORY_VARIANT_COLUMNS[PATIENT_STAYS])))
def test_category_variants_normalise_back(column: str) -> None:
    rng = np.random.default_rng(0)
    for canonical in sorted(set(CATEGORY_ALIASES[column].values())):
        for _ in range(5):
            variant = category_variant(column, canonical, rng)
            assert variant != canonical
            assert normalize_category(column, variant) == canonical


# ---------------------------------------------------------------------------
# File generation
# ---------------------------------------------------------------------------
def test_generate_source_data_writes_files(isolated_paths: dict) -> None:
    raw_dir = isolated_paths["raw_dir"]
    manifest = generate_source_data(seed=9, start_date="2023-06-01", end_date="2023-07-10")
    for table in (WARD_DAILY, PATIENT_STAYS):
        assert io.raw_path(table, raw_dir).exists()
        assert io.ground_truth_path(table, raw_dir).exists()
    log = io.read_csv_raw(raw_dir / "_ground_truth" / "injection_log.csv")
    on_disk = json.loads((raw_dir / "simulation_manifest.json").read_text(encoding="utf-8"))
    assert on_disk == manifest
    assert manifest["seed"] == 9
    assert manifest["period"] == {"start_date": "2023-06-01", "end_date": "2023-07-10", "days": 40,
                                  "warmup_days": manifest["period"]["warmup_days"],
                                  "cooldown_days": manifest["period"]["cooldown_days"]}
    assert manifest["data_quality_injection"]["total_issues"] == len(log)
    for table in (WARD_DAILY, PATIENT_STAYS):
        truth = io.read_csv_raw(io.ground_truth_path(table, raw_dir))
        raw = io.read_csv_raw(io.raw_path(table, raw_dir))
        assert list(truth.columns) == list(RAW_COLUMNS[table])
        assert list(raw.columns) == list(RAW_COLUMNS[table])
        assert manifest["row_counts"][table] == {"clean": len(truth), "raw": len(raw)}
        counts = log.loc[log["table"] == table, "issue_type"].value_counts().to_dict()
        assert manifest["data_quality_injection"]["issues_by_table"][table] == counts
    expected = simulate("2023-06-01", "2023-07-10", seed=9)
    truth = io.read_table(io.ground_truth_path(PATIENT_STAYS, raw_dir), PATIENT_STAYS)
    assert truth["stay_id"].tolist() == expected.patient_stays["stay_id"].tolist()
    assert truth["bed_wait_hours"].tolist() == expected.patient_stays["bed_wait_hours"].tolist()


def test_generate_source_data_without_injection(tmp_path) -> None:
    manifest = generate_source_data(raw_dir=tmp_path, seed=4, inject_issues=False,
                                    start_date="2023-09-01", end_date="2023-09-21")
    assert manifest["data_quality_injection"]["total_issues"] == 0
    for table in (WARD_DAILY, PATIENT_STAYS):
        raw = io.read_csv_raw(io.raw_path(table, tmp_path))
        truth = io.read_csv_raw(io.ground_truth_path(table, tmp_path))
        pd.testing.assert_frame_equal(raw, truth)


# ---------------------------------------------------------------------------
# Full reference period (calibration regression; a few seconds)
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_full_period_meets_calibration_targets() -> None:
    result = simulate(seed=42)
    summary = result.calibration
    assert result.diagnostics["runtime_seconds"] < 120
    assert summary["period"]["days"] == 1096
    failed = summary["failed_targets"]
    assert summary["targets_passed"] >= summary["targets_evaluated"] - 2, f"calibration regressed: {failed}"
