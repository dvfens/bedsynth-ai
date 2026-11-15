"""Tests for the Week-1 cleaning pipeline (bedsynth.data.preprocessing, DESIGN.md §6.3).

Dirty raw inputs are built by hand from the rule-satisfying fixtures: every issue type of
DESIGN.md §6.2 is injected a few times, with a matching ground truth and injection log, so the
tests can check that each issue is detected, repaired and accounted for in the report. The
script test runs the real simulator (one quarter) and the real Week-1 report renderer.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import numpy as np
import pandas as pd
import pytest

from bedsynth.config import PATH_ENV_VARS, project_relative, resolve_project_path
from bedsynth.data import preprocessing as prep
from bedsynth.data.fixtures import make_patient_stays, make_ward_daily
from bedsynth.data.io import ground_truth_path, processed_path, raw_path, read_csv_raw, read_table
from bedsynth.schema import (
    CATEGORY_ALIASES,
    PATIENT_STAYS,
    RAW_COLUMNS,
    TABLES,
    WARD_DAILY,
    WARD_IDS,
    column_names,
    training_columns,
)
from bedsynth.validation import validate_table

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "01_prepare_data.py"
N_DAYS = 30


# ---------------------------------------------------------------------------
# Hand-made dirty extracts
# ---------------------------------------------------------------------------
def raw_ward_daily(n_days: int = N_DAYS, seed: int = 21) -> pd.DataFrame:
    """Clean ward_daily in raw-extract format (ward-major, consecutive days)."""
    return make_ward_daily(n_days=n_days, seed=seed)[list(RAW_COLUMNS[WARD_DAILY])].reset_index(drop=True)


def raw_patient_stays(n: int = 300, seed: int = 22) -> pd.DataFrame:
    """Clean patient_stays in raw-extract format (with timestamps)."""
    return make_patient_stays(n=n, seed=seed, with_times=True)[list(RAW_COLUMNS[PATIENT_STAYS])].reset_index(drop=True)


def spelling_variant(column: str, canonical: str, rng: np.random.Generator) -> str:
    """A non-canonical spelling of ``canonical``: an alias plus case and whitespace noise."""
    aliases = sorted(a for a, value in CATEGORY_ALIASES[column].items() if value == canonical and a != canonical)
    alias = str(rng.choice(aliases)) if aliases else canonical.lower()
    alias = (alias.upper(), alias.title(), alias)[int(rng.integers(3))]
    return f"  {alias}" if alias == canonical or rng.random() < 0.5 else f"{alias} "


@dataclass
class DirtyTable:
    table: str
    truth: pd.DataFrame
    raw: pd.DataFrame
    log: list[dict[str, Any]] = field(default_factory=list)
    marked: dict[str, list[str]] = field(default_factory=dict)   # injection name -> row keys


class Injector:
    """Inject issues into distinct rows of a clean table and log them like the simulator."""

    def __init__(self, truth: pd.DataFrame, table: str, seed: int) -> None:
        self.extract = DirtyTable(table, truth.reset_index(drop=True), truth.reset_index(drop=True).astype(object))
        self.rng = np.random.default_rng(seed)
        self.free = [int(p) for p in self.rng.permutation(len(truth))]

    def key(self, pos: int) -> str:
        row = self.extract.truth.loc[pos]
        return f"{row['ward_id']}|{row['date']}" if self.extract.table == WARD_DAILY else str(row["stay_id"])

    def take(self, n: int, where: np.ndarray | None = None) -> list[int]:
        chosen = [p for p in self.free if where is None or where[p]][:n]
        assert len(chosen) == n, "not enough eligible rows for the injection"
        self.free = [p for p in self.free if p not in chosen]
        return chosen

    def log(self, pos: int, column: str, issue: str, original: Any, injected: Any, name: str) -> None:
        self.extract.log.append({
            "table": self.extract.table, "row_key": self.key(pos), "column": column, "issue_type": issue,
            "original_value": original, "injected_value": "" if injected is None else injected,
        })
        self.extract.marked.setdefault(name, []).append(self.key(pos))

    def corrupt(self, name: str, issue: str, column: str, rows: list[int], make: Callable[[Any], Any]) -> None:
        for pos in rows:
            original = self.extract.truth.at[pos, column]
            injected = make(original)
            self.extract.raw.at[pos, column] = injected
            self.log(pos, column, issue, original, injected, name)

    def swap_timestamps(self, rows: list[int]) -> None:
        raw = self.extract.raw
        for pos in rows:
            admitted, left = raw.at[pos, "admission_datetime"], raw.at[pos, "discharge_datetime"]
            raw.at[pos, "admission_datetime"], raw.at[pos, "discharge_datetime"] = left, admitted
            self.log(pos, "admission_datetime|discharge_datetime", "swapped_timestamps",
                     f"{admitted} / {left}", f"{left} / {admitted}", "swapped_timestamps")

    def duplicate(self, rows: list[int], transform: Callable[[pd.DataFrame], pd.DataFrame] | None = None) -> None:
        copies = self.extract.raw.loc[rows].copy()
        copies = transform(copies) if transform else copies
        self.extract.raw = pd.concat([self.extract.raw, copies], ignore_index=True)
        for pos in rows:
            self.log(pos, "", "duplicate", "", "", "duplicate")


def dirty_ward_daily(seed: int = 21) -> DirtyTable:
    injector = Injector(raw_ward_daily(seed=seed), WARD_DAILY, seed)
    truth, rng = injector.extract.truth, injector.rng
    for column, n in (("ward_id", 6), ("staffing_level", 4), ("occupancy_level", 4)):
        injector.corrupt(f"variant_{column}", "category_variant", column, injector.take(n),
                         lambda v, c=column: spelling_variant(c, v, rng))
    injector.corrupt("day_first_date", "alt_date_format", "date", injector.take(5),
                     lambda v: pd.Timestamp(v).strftime("%d/%m/%Y"))
    for column in ("nurses_night_shift", "avg_bed_turnaround_min", "staffing_level", "avg_los_days"):
        injector.corrupt(f"missing_{column}", "missing_value", column, injector.take(3), lambda v: None)
    has_previous_day = (np.arange(len(truth)) % N_DAYS) != 0
    for column, sentinel in (("census_start", -1), ("discharges", 999), ("nurses_day_shift", 0),
                             ("avg_bed_turnaround_min", 0), ("beds_closed", -2)):
        where = has_previous_day if column == "census_start" else None
        injector.corrupt(f"invalid_{column}", "invalid_value", column, injector.take(3, where),
                         lambda v, s=sentinel: s)
    duplicated = injector.take(4)
    injector.duplicate(duplicated[:3])
    injector.duplicate(duplicated[3:], lambda d: d.assign(date=pd.to_datetime(d["date"]).dt.strftime("%d/%m/%Y")))
    junk = injector.extract.raw.loc[injector.take(1)].assign(date="not a date")   # cannot be dated: dropped
    injector.extract.raw = pd.concat([injector.extract.raw, junk], ignore_index=True)
    return injector.extract


def dirty_patient_stays(seed: int = 22) -> DirtyTable:
    injector = Injector(raw_patient_stays(seed=seed), PATIENT_STAYS, seed)
    truth, rng = injector.extract.truth, injector.rng
    for column, n in (("ward_id", 5), ("admission_source", 5), ("separation_type", 4),
                      ("discharge_destination", 4), ("delay_reason", 4)):
        injector.corrupt(f"variant_{column}", "category_variant", column, injector.take(n),
                         lambda v, c=column: spelling_variant(c, v, rng))
    for column in ("admission_datetime", "discharge_datetime"):
        injector.corrupt(f"day_first_{column}", "alt_date_format", column, injector.take(3),
                         lambda v: pd.Timestamp(v).strftime("%d/%m/%Y %H:%M"))
    delayed = (truth["discharge_delay_hours"].astype(float) > 0).to_numpy()
    transfer_out = (truth["separation_type"] == "transfer_out").to_numpy()
    elective = (truth["admission_source"] == "elective").to_numpy()
    missing = lambda v: None  # noqa: E731
    injector.corrupt("missing_reason_delayed", "missing_value", "delay_reason", injector.take(2, delayed), missing)
    injector.corrupt("missing_reason_on_time", "missing_value", "delay_reason", injector.take(2, ~delayed), missing)
    injector.corrupt("missing_destination_transfer", "missing_value", "discharge_destination",
                     injector.take(1, transfer_out), missing)
    injector.corrupt("missing_destination_discharge", "missing_value", "discharge_destination",
                     injector.take(2, ~transfer_out & delayed), missing)
    for column in ("bed_turnaround_minutes", "bed_wait_hours"):
        injector.corrupt(f"missing_{column}", "missing_value", column, injector.take(3), missing)
    injector.corrupt("invalid_wait", "invalid_value", "bed_wait_hours", injector.take(2), lambda v: -1)
    injector.corrupt("invalid_turnaround", "invalid_value", "bed_turnaround_minutes", injector.take(2), lambda v: 99999)
    injector.corrupt("invalid_lead", "invalid_value", "booking_lead_days", injector.take(2, elective), lambda v: -5)
    injector.swap_timestamps(injector.take(3))
    injector.duplicate(injector.take(3))
    return injector.extract


def write_dirty_extract(raw_dir: Path, *extracts: DirtyTable) -> None:
    """Write raw files, ground truth and the injection log exactly where the simulator would."""
    log_rows: list[dict[str, Any]] = []
    for extract in extracts:
        for path, frame in ((raw_path(extract.table, raw_dir), extract.raw),
                            (ground_truth_path(extract.table, raw_dir), extract.truth)):
            path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_csv(path, index=False)
        log_rows += extract.log
    log_path = Path(raw_dir) / "_ground_truth" / prep.INJECTION_LOG_FILENAME
    pd.DataFrame(log_rows, columns=list(prep.INJECTION_LOG_COLUMNS)).to_csv(log_path, index=False)


def roundtrip(frame: pd.DataFrame, tmp_path: Path, name: str = "raw.csv") -> pd.DataFrame:
    """Write and re-read a raw frame the way the pipeline reads extracts."""
    path = tmp_path / name
    frame.to_csv(path, index=False)
    return read_csv_raw(path)


# ---------------------------------------------------------------------------
# Step 2 - categories
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("column", "raw", "expected"),
    [
        ("ward_id", "  Gen Med ", "GMED"),
        ("ward_id", "general   medicine", "GMED"),      # internal whitespace run
        ("ward_id", "ICU", "ICU"),
        ("separation_type", "Transfer_Out", "transfer_out"),
        ("delay_reason", "N/A", "none"),
        ("discharge_destination", "Nursing Home", "care_facility"),
        ("occupancy_level", "FULL", "critical"),
        ("staffing_level", "Understaffed", "short"),
        ("ward_id", "Mars Ward", None),
        ("ward_id", np.nan, None),
    ],
)
def test_canonical_category(column, raw, expected):
    assert prep.canonical_category(column, raw) == expected


def test_standardize_categories_counts_variants_and_unknowns():
    df = pd.DataFrame({
        "ward_id": ["AMU", " amu", "Cardiology", "Atlantis", None],
        "admission_source": ["emergency", "A&E", "planned", "emergency", "xfer"],
        "separation_type": ["discharge"] * 5,
        "discharge_destination": ["home"] * 5,
        "delay_reason": ["none", "n/a", "NA", "Pharmacy", "none"],
        "stay_id": [" ST-1", "ST-2", "ST-3", "ST-4", "ST-5"],
    })
    out, report = prep.standardize_categories(df, PATIENT_STAYS)
    assert out["ward_id"].tolist()[:3] == ["AMU", "AMU", "CARD"] and out["ward_id"].isna().sum() == 2
    assert out["admission_source"].tolist() == ["emergency", "emergency", "elective", "emergency", "transfer"]
    assert out["delay_reason"].tolist() == ["none", "none", "none", "awaiting_medication", "none"]
    assert out.at[0, "stay_id"] == "ST-1"
    ward = report["columns"]["ward_id"]
    assert (ward["canonical"], ward["standardised"], ward["unknown_to_missing"], ward["missing"]) == (1, 2, 1, 1)
    assert ward["unknown_values"] == ["Atlantis"]
    assert report["issues"]["category_variant"] == {"ward_id": 2, "admission_source": 3, "delay_reason": 3}
    assert report["issues"]["unknown_category"] == {"ward_id": 1}


# ---------------------------------------------------------------------------
# Step 3 - timestamps
# ---------------------------------------------------------------------------
def test_parse_timestamp_series_iso_first_then_day_first():
    series = pd.Series(["2023-01-05", "05/01/2023", " 2023-02-01 ", "2023-13-45", None, "31/02/2023"])
    parsed, how = prep.parse_timestamp_series(series, "date")
    assert how.tolist() == ["iso", "day_first", "iso", "unparseable", "missing", "unparseable"]
    assert parsed.iloc[1] == pd.Timestamp("2023-01-05")      # day-first, not month-first
    datetimes = pd.Series(["2023-01-05 10:30", "05/01/2023 10:30", "2023-01-05 10:30:00", "2023-01-05"])
    parsed, how = prep.parse_timestamp_series(datetimes, "datetime")
    assert how.tolist() == ["iso", "day_first", "iso", "unparseable"]
    assert parsed.iloc[:3].nunique() == 1


def test_parse_timestamps_drops_undatable_rows_and_adds_calendar_features():
    df = raw_ward_daily().head(4).astype(object)
    df.loc[1, "date"] = pd.Timestamp(df.loc[1, "date"]).strftime("%d/%m/%Y")
    df.loc[2, "date"] = "yesterday"
    out, report = prep.parse_timestamps(df, WARD_DAILY)
    assert len(out) == 3 and report["rows_dropped"] == 1
    assert out["date"].tolist() == [df.loc[0, "date"], raw_ward_daily().loc[1, "date"], df.loc[3, "date"]]
    assert {"month", "day_of_week"} <= set(out.columns)
    assert report["columns"]["date"]["day_first"] == 1 and report["columns"]["date"]["unparseable_examples"] == ["yesterday"]


# ---------------------------------------------------------------------------
# Step 4 - duplicates
# ---------------------------------------------------------------------------
def test_remove_duplicates_exact_then_key_keeping_first():
    df = raw_ward_daily().head(5).astype(object)
    modified_copy = df.loc[[1]].assign(discharges=99)                 # same key, different values
    no_ward = df.loc[[2]].assign(ward_id=np.nan, date="2030-01-01")
    frame = pd.concat([df, df.loc[[0, 3]], modified_copy, no_ward], ignore_index=True)
    out, report = prep.remove_duplicates(frame, WARD_DAILY)
    assert (report["exact_duplicates"], report["key_duplicates"], report["missing_key"]) == (2, 1, 1)
    assert len(out) == 5 and out.equals(df.reset_index(drop=True))
    assert report["issues"] == {"exact_duplicate": {"*": 2}, "key_duplicate": {"*": 1}, "missing_key": {"*": 1}}


def test_stays_without_source_id_are_kept():
    df = raw_patient_stays().head(4).astype(object)
    df.loc[2, "stay_id"] = np.nan
    df.loc[3, "stay_id"] = np.nan
    df.loc[3, "bed_wait_hours"] = 1.23
    out, report = prep.remove_duplicates(df, PATIENT_STAYS)
    assert len(out) == 4 and report["rows_without_source_key_kept"] == 2


# ---------------------------------------------------------------------------
# Step 5 - invalid values
# ---------------------------------------------------------------------------
def test_flag_invalid_values_ward_daily():
    df, _ = prep.parse_timestamps(raw_ward_daily().head(N_DAYS * 2), WARD_DAILY)
    df = df.astype({"census_start": object})
    sentinels = {0: ("census_start", -1), 1: ("discharges", 999), 2: ("nurses_day_shift", 0),
                 3: ("avg_bed_turnaround_min", 0), 4: ("beds_closed", -2), 5: ("census_start", "n/a"),
                 6: ("transfers_in", 1.5), 7: ("avg_los_days", 400)}
    for row, (column, value) in sentinels.items():
        df[column] = df[column].astype(object)
        df.at[row, column] = value
    original = df.copy()
    out, report = prep.flag_invalid_values(df, WARD_DAILY)
    for row, (column, _) in sentinels.items():
        assert np.isnan(out.at[row, column]), (row, column)
    untouched = [c for c in prep.raw_numeric_columns(WARD_DAILY)]
    rest = out.drop(index=list(sentinels))[untouched].astype(float)
    assert rest.equals(original.drop(index=list(sentinels))[untouched].astype(float))
    assert report["columns"]["census_start"] == {"non_numeric": 1, "below_minimum": 1}
    assert report["columns"]["discharges"] == {"above_maximum": 1}
    assert report["columns"]["transfers_in"] == {"non_integer": 1}
    assert report["cells_invalidated"] == 8


def test_physical_bounds_are_ward_specific():
    frame = pd.DataFrame({"ward_id": ["ICU", "GMED", "UNKNOWN"]})
    low, high = prep.physical_bounds(frame, WARD_DAILY, "census_start")
    assert low.tolist() == [0, 0, 0] and high.tolist()[:2] == [14, 36] and np.isinf(high[2])
    _, high = prep.physical_bounds(frame, WARD_DAILY, "escalation_beds_open")
    assert high.tolist()[:2] == [2, 4]


def test_flag_invalid_values_stays_swaps_timestamps_and_checks_relations():
    df = raw_patient_stays().head(40).astype(object)
    swapped_admission = df.loc[0, "admission_datetime"]
    df.loc[0, ["admission_datetime", "discharge_datetime"]] = [df.loc[0, "discharge_datetime"], swapped_admission]
    elective = df.index[df["admission_source"] == "elective"][0]
    emergency = df.index[df["admission_source"] == "emergency"][0]
    df.loc[elective, "booking_lead_days"] = 0          # electives are booked >= 1 day ahead
    df.loc[emergency, "times_rescheduled"] = 2         # non-electives are never rescheduled
    df.loc[5, "discharge_delay_hours"] = 10_000.0      # longer than the stay
    df.loc[6, "bed_turnaround_minutes"] = 99999
    out, report = prep.flag_invalid_values(df, PATIENT_STAYS)
    assert report["timestamp_swaps_repaired"] == 1 and out.loc[0, "admission_datetime"] == swapped_admission
    assert np.isnan(out.loc[elective, "booking_lead_days"]) and np.isnan(out.loc[emergency, "times_rescheduled"])
    assert np.isnan(out.loc[5, "discharge_delay_hours"]) and np.isnan(out.loc[6, "bed_turnaround_minutes"])
    assert report["issues"]["timestamp_swap"] == {"admission_datetime|discharge_datetime": 1}
    assert report["columns"]["bed_turnaround_minutes"] == {"above_maximum": 1}


# ---------------------------------------------------------------------------
# Step 6 - repairs
# ---------------------------------------------------------------------------
def _prepared_ward_daily(truth: pd.DataFrame) -> pd.DataFrame:
    frame, _ = prep.parse_timestamps(truth, WARD_DAILY)
    frame, _ = prep.flag_invalid_values(frame, WARD_DAILY)
    return frame


def test_repair_ward_daily_recovers_exact_values_from_continuity_and_flow_identity():
    truth = raw_ward_daily()
    dirty = truth.copy().astype({"census_start": float, "discharges": float, "elective_cancelled": float})
    gmed = truth.index[truth["ward_id"] == "GMED"]
    dirty.loc[gmed[5], "census_start"] = np.nan            # continuity: census_end of the day before
    dirty.loc[gmed[0], "census_start"] = np.nan            # first day: flow identity
    dirty.loc[gmed[9], "discharges"] = np.nan              # flow identity
    dirty.loc[gmed[12], "elective_cancelled"] = np.nan     # elective identity
    repaired, report = prep.repair_ward_daily(_prepared_ward_daily(dirty))
    merged = repaired.merge(truth, on=["ward_id", "date"], suffixes=("", "_true"))
    for column in ("census_start", "discharges", "elective_cancelled"):
        assert (merged[column] == merged[f"{column}_true"]).all(), column
    methods = report["by_column"]
    assert methods["census_start"] == {"temporal_continuity": 1, "flow_identity": 1}
    assert methods["discharges"] == {"flow_identity": 1}
    assert methods["elective_cancelled"] == {"elective_identity": 1}
    assert report["approximate_cells"] == 0 and report["unrepaired"] == {}


def test_repair_ward_daily_interpolates_metrics_and_recomputes_labels():
    truth = raw_ward_daily()
    dirty = truth.copy().astype(object)
    rows = truth.index[truth["ward_id"] == "CARD"]
    dirty.loc[rows[3], "avg_los_days"] = np.nan
    dirty.loc[rows[4], "avg_bed_turnaround_min"] = np.nan
    dirty.loc[rows[0], "avg_bed_turnaround_min"] = np.nan        # series edge
    dirty.loc[rows[6], "staffing_level"] = np.nan
    dirty.loc[rows[7], "nurses_night_shift"] = np.nan
    wrong_label = "low" if truth.loc[rows[8], "occupancy_level"] != "low" else "high"
    dirty.loc[rows[8], "occupancy_level"] = wrong_label
    repaired, report = prep.repair_ward_daily(_prepared_ward_daily(dirty))
    card = repaired[repaired["ward_id"] == "CARD"].reset_index(drop=True)
    neighbours = truth.loc[[rows[2], rows[4]], "avg_los_days"].astype(float)
    assert card.loc[3, "avg_los_days"] == pytest.approx(neighbours.mean(), abs=0.01)
    assert card.loc[0, "avg_bed_turnaround_min"] == truth.loc[rows[1], "avg_bed_turnaround_min"]
    assert card.loc[6, "staffing_level"] == truth.loc[rows[6], "staffing_level"]
    assert card.loc[8, "occupancy_level"] == truth.loc[rows[8], "occupancy_level"]
    assert card.loc[7, "nurses_night_shift"] >= 1
    labels = report["label_recomputation"]
    assert labels["staffing_level"]["raw_missing"] == 1 and labels["occupancy_level"]["disagreed"] == 1
    assert report["by_column"]["avg_los_days"] == {"within_ward_interpolation": 1}
    assert report["by_column"]["nurses_night_shift"] == {"ward_staffing_band_median": 1}


def test_repair_ward_daily_capacity_state_keeps_row_valid_and_band_consistent():
    truth = raw_ward_daily()
    dirty = truth.copy().astype({"beds_closed": float, "escalation_beds_open": float})
    targets = list(truth.index[:: N_DAYS // 2][:10])
    dirty.loc[targets[:5], "beds_closed"] = np.nan
    dirty.loc[targets[5:], "escalation_beds_open"] = np.nan
    repaired, report = prep.repair_ward_daily(_prepared_ward_daily(dirty))
    derived = prep.derive_table(repaired, WARD_DAILY)
    assert validate_table(derived, WARD_DAILY).all_hard_passed
    merged = derived.merge(truth[["ward_id", "date", "occupancy_level"]], on=["ward_id", "date"], suffixes=("", "_raw"))
    assert (merged["occupancy_level"] == merged["occupancy_level_raw"]).all()   # recorded bands reproduced
    assert sum(report["by_column"]["beds_closed"].values()) == 5


def test_repair_patient_stays_follows_operational_rules():
    truth = raw_patient_stays()
    dirty = truth.copy().astype(object)
    delayed = truth.index[truth["discharge_delay_hours"] > 0]
    on_time = truth.index[truth["discharge_delay_hours"] == 0]
    transfer = truth.index[truth["separation_type"] == "transfer_out"][0]
    elective = truth.index[truth["admission_source"] == "elective"][:2]
    emergency = truth.index[truth["admission_source"] == "emergency"][0]
    dirty.loc[[delayed[0], on_time[0]], "delay_reason"] = np.nan
    dirty.loc[[transfer, delayed[1]], "discharge_destination"] = np.nan
    dirty.loc[elective[0], "booking_lead_days"] = np.nan
    dirty.loc[elective[1], "times_rescheduled"] = np.nan
    dirty.loc[emergency, "booking_lead_days"] = np.nan
    dirty.loc[on_time[1:4], "bed_wait_hours"] = np.nan
    dirty.loc[on_time[4:6], "bed_turnaround_minutes"] = np.nan
    dirty.loc[on_time[6], "separation_type"] = np.nan
    repaired, report = prep.repair_patient_stays(dirty)
    assert repaired.loc[on_time[0], "delay_reason"] == "none"
    assert repaired.loc[delayed[0], "delay_reason"] != "none"
    assert repaired.loc[transfer, "discharge_destination"] == "internal_ward"
    assert repaired.loc[elective[0], "booking_lead_days"] >= 1
    assert repaired.loc[emergency, "booking_lead_days"] == 0
    assert repaired.loc[on_time[6], "separation_type"] == truth.loc[on_time[6], "separation_type"]
    assert report["unrepaired"] == {}
    derived = prep.derive_table(repaired, PATIENT_STAYS)
    assert validate_table(derived, PATIENT_STAYS).all_hard_passed
    methods = report["by_column"]
    assert methods["delay_reason"]["definitional_none"] == 1
    assert methods["discharge_destination"]["separation_rule"] == 1
    assert methods["booking_lead_days"] == {"ward_elective_median": 1, "non_elective_zero": 1}
    assert set(methods["bed_turnaround_minutes"]) <= {"ward_departure_period_hot_deck", "ward_hot_deck", "global_hot_deck"}
    assert set(methods["bed_wait_hours"]) <= {"ward_source_hot_deck", "ward_hot_deck", "source_hot_deck", "global_hot_deck"}


def test_hot_deck_imputation_draws_observed_donors_without_point_masses():
    """D8 regression: imputed continuous values come from observed rows of the same group and vary."""
    frame = pd.DataFrame({
        "ward_id": ["AMU"] * 200 + ["ICU"] * 200,
        "value": np.r_[np.linspace(1.0, 2.0, 200), np.linspace(10.0, 20.0, 200)],
    })
    missing = np.zeros(len(frame), dtype=bool)
    missing[::4] = True                                             # 100 gaps, 50 per ward
    observed = frame["value"].copy()
    frame.loc[missing, "value"] = np.nan
    levels = ((("ward_id",), "ward_hot_deck"), ((), "global_hot_deck"))
    log = prep._RepairLog(len(frame))
    assert prep._impute_group_hot_deck(frame, "value", levels, log, decimals=3) == 100
    filled = frame.loc[missing]
    donors = set(observed[~missing].round(3))
    assert set(filled["value"]) <= donors                          # only observed values are used
    assert filled.loc[filled["ward_id"] == "AMU", "value"].between(1.0, 2.0).all()
    assert filled.loc[filled["ward_id"] == "ICU", "value"].between(10.0, 20.0).all()
    assert filled["value"].value_counts().iloc[0] <= 5            # no single imputed value dominates
    again = frame.copy()
    again.loc[missing, "value"] = np.nan
    prep._impute_group_hot_deck(again, "value", levels, prep._RepairLog(len(again)), decimals=3)
    pd.testing.assert_series_equal(again["value"], frame["value"])  # deterministic


def test_rule_aware_imputation_of_booking_lead_and_delays():
    """D1 interplay: imputed leads respect 1 + 7 x reschedules and imputed delays are >= 4 h."""
    truth = raw_patient_stays()
    dirty = truth.copy().astype(object)
    rescheduled = truth.index[(truth["admission_source"] == "elective") & (truth["times_rescheduled"] >= 1)]
    delayed = truth.index[truth["discharge_delay_hours"] > 0]
    assert len(rescheduled) and len(delayed) >= 3
    dirty.loc[rescheduled, "booking_lead_days"] = np.nan
    dirty.loc[delayed[:3], "discharge_delay_hours"] = np.nan
    repaired, _ = prep.repair_patient_stays(dirty)
    lead = repaired.loc[rescheduled, "booking_lead_days"].astype(float)
    assert (lead >= 1 + 7 * truth.loc[rescheduled, "times_rescheduled"]).all()
    assert (repaired.loc[delayed[:3], "discharge_delay_hours"].astype(float) >= 4.0).all()
    derived = prep.derive_table(repaired, PATIENT_STAYS)
    assert validate_table(derived, PATIENT_STAYS).all_hard_passed


# ---------------------------------------------------------------------------
# Steps 1-8 on clean input: nothing changes, nothing is dropped
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("table", TABLES)
def test_cleaning_clean_input_is_lossless(table, tmp_path):
    truth = raw_ward_daily() if table == WARD_DAILY else raw_patient_stays()
    reference = (make_ward_daily(n_days=N_DAYS, seed=21) if table == WARD_DAILY
                 else make_patient_stays(n=300, seed=22, with_times=True))
    clean, report = prep.clean_table(roundtrip(truth, tmp_path), table)
    assert len(clean) == len(truth) and report["rows_dropped"]["total"] == 0
    assert report["repairs"]["cells_repaired"] == 0 and report["issues_detected"] == {}
    key = list(prep.SOURCE_KEYS[table])
    merged = clean.merge(reference, on=key, suffixes=("", "_ref"))
    for column in training_columns(table):
        if column not in key:
            assert (merged[column].astype(str) == merged[f"{column}_ref"].astype(str)).all(), column
    assert report["validation"]["final"]["hard_valid_rate"] == 1.0


def test_missing_raw_columns_raise():
    with pytest.raises(ValueError, match="missing columns"):
        prep.clean_ward_daily(raw_ward_daily().drop(columns=["census_end"]))


# ---------------------------------------------------------------------------
# Steps 9-10 - checks, ids and split
# ---------------------------------------------------------------------------
def test_ward_daily_continuity_detects_breaks(ward_daily_df):
    report = prep.ward_daily_continuity(ward_daily_df)
    assert report["continuity_rate"] == 1.0 and report["missing_ward_days"] == 0
    broken = ward_daily_df.copy()
    broken.loc[broken.index[5], "census_start"] += 1
    assert prep.ward_daily_continuity(broken)["breaks"] == 1


def test_cross_table_admissions_match_their_ward_days():
    stays = make_patient_stays(n=200, seed=3, with_times=True)
    stays["date"] = stays["admission_datetime"].str[:10]
    counts = stays.groupby(["ward_id", "date", "admission_source"]).size().unstack(fill_value=0)
    days = counts.reset_index().rename(columns={"elective": "admissions_elective",
                                                "emergency": "admissions_emergency", "transfer": "transfers_in"})
    report = prep.cross_table_admissions(days, stays)
    assert report["all_sources_match_rate"] == 1.0 and report["stays_without_ward_day"] == 0
    days.loc[0, "admissions_emergency"] += 2
    report = prep.cross_table_admissions(days, stays)
    assert report["by_source"]["emergency"]["match_rate"] < 1.0


def test_assign_ids_chronologically(patient_stays_df):
    stays = make_patient_stays(n=50, seed=8, with_times=True).sample(frac=1.0, random_state=0)
    ordered = prep.assign_ids_chronologically(stays, PATIENT_STAYS)
    assert ordered["stay_id"].tolist() == [f"ST-{i:07d}" for i in range(1, 51)]
    assert ordered["admission_datetime"].is_monotonic_increasing


def test_split_train_holdout_is_stratified_disjoint_and_deterministic(ward_daily_df):
    train, holdout = prep.split_train_holdout(ward_daily_df, WARD_DAILY, fraction=0.2, seed=5)
    assert len(train) + len(holdout) == len(ward_daily_df)
    assert set(train["record_id"]).isdisjoint(holdout["record_id"])
    shares = holdout["ward_id"].value_counts() / ward_daily_df["ward_id"].value_counts()
    assert shares.round(2).eq(0.2).all() and set(shares.index) == set(WARD_IDS)
    assert train["record_id"].is_monotonic_increasing and holdout["record_id"].is_monotonic_increasing
    again, _ = prep.split_train_holdout(ward_daily_df, WARD_DAILY, fraction=0.2, seed=5)
    other, _ = prep.split_train_holdout(ward_daily_df, WARD_DAILY, fraction=0.2, seed=6)
    assert train.equals(again) and not train.equals(other)
    with pytest.raises(ValueError):
        prep.split_train_holdout(ward_daily_df, WARD_DAILY, fraction=1.0)


def test_split_keeps_tiny_strata_in_train():
    df = pd.DataFrame({"ward_id": ["AMU", "ICU", "ICU"], "x": [1, 2, 3]})
    train, holdout = prep.split_train_holdout(df, WARD_DAILY, fraction=0.5, seed=0)
    assert train["ward_id"].tolist().count("AMU") == 1 and len(holdout) == 1


# ---------------------------------------------------------------------------
# Step 12 - cleaning accuracy and JSON safety
# ---------------------------------------------------------------------------
def test_cleaning_accuracy_perfect_and_degraded(tmp_path):
    truth = raw_patient_stays()
    clean, _ = prep.clean_patient_stays(roundtrip(truth, tmp_path))
    perfect = prep.cleaning_accuracy(clean, truth, PATIENT_STAYS)
    assert perfect["cell_agreement"] == 1.0 and perfect["row_recall"] == 1.0 and perfect["row_precision"] == 1.0
    degraded = clean.drop(index=[0, 1]).copy()
    degraded.loc[2, "bed_wait_hours"] = 99.0
    accuracy = prep.cleaning_accuracy(degraded, truth, PATIENT_STAYS)
    assert accuracy["row_recall"] == pytest.approx(298 / 300, abs=1e-6)
    assert accuracy["column_agreement"]["bed_wait_hours"] == pytest.approx(297 / 298, abs=1e-6)
    assert accuracy["least_accurate_columns"][0][0] == "bed_wait_hours"


def test_injected_issue_outcomes_handle_log_variants(tmp_path):
    truth = raw_patient_stays()
    clean, _ = prep.clean_patient_stays(roundtrip(truth, tmp_path))
    clean = clean[clean["stay_id"] != truth.at[1, "stay_id"]].copy()            # row dropped by cleaning
    clean.loc[clean["stay_id"] == truth.at[2, "stay_id"], "bed_wait_hours"] = -1.0  # defect left in place
    clean.loc[clean["stay_id"] == truth.at[3, "stay_id"], "bed_wait_hours"] = 0.5   # plausible imputation
    log = pd.DataFrame([
        {"row_key": truth.at[0, "stay_id"], "column": "*", "issue_type": "duplicate_row"},
        {"row_key": truth.at[1, "stay_id"], "column": "ward_id", "issue_type": "category_variant"},
        {"row_key": truth.at[2, "stay_id"], "column": "bed_wait_hours", "issue_type": "invalid_value",
         "injected_value": "-1"},
        {"row_key": truth.at[3, "stay_id"], "column": "bed_wait_hours", "issue_type": "missing_value"},
        {"row_key": truth.at[4, "stay_id"], "column": "admission_datetime, discharge_datetime",
         "issue_type": "timestamp_swap"},
        {"row_key": "ST-9999999", "column": "ward_id", "issue_type": "category_variant"},
    ]).assign(table=PATIENT_STAYS, original_value="")
    outcomes = prep.cleaning_accuracy(clean, truth, PATIENT_STAYS, log)["injected_issues"]
    assert outcomes["status_counts"] == {"exact": 2, "repaired": 1, "row_dropped": 1, "unresolved": 1,
                                         "not_evaluable": 1}
    assert outcomes["evaluable"] == 5 and outcomes["resolved_rate"] == pytest.approx(0.6)
    assert outcomes["by_issue_type"]["invalid_value"]["unresolved"] == 1


def test_ward_daily_log_keys_are_normalised(tmp_path):
    truth = raw_ward_daily()
    clean, _ = prep.clean_ward_daily(roundtrip(truth, tmp_path))
    row = truth.index[(truth["ward_id"] == "GMED") & (truth["date"] == "2023-01-05")][0]
    log = pd.DataFrame([{"table": WARD_DAILY, "row_key": " gen med |05/01/2023", "column": "discharges",
                         "issue_type": "invalid_value", "original_value": truth.at[row, "discharges"],
                         "injected_value": 999}])
    outcomes = prep.cleaning_accuracy(clean, truth, WARD_DAILY, log)["injected_issues"]
    assert outcomes["status_counts"]["exact"] == 1 and outcomes["status_counts"]["not_evaluable"] == 0


def test_dataset_checks_tolerate_empty_tables(ward_daily_df, patient_stays_df):
    stays = make_patient_stays(n=5, seed=1, with_times=True).iloc[0:0]
    report = prep.cross_table_admissions(ward_daily_df, stays)
    assert report["stays_in_window"] == 0 and report["by_source"]["emergency"]["stays_total"] == 0
    assert prep.ward_daily_continuity(ward_daily_df.iloc[0:0])["continuity_rate"] is None
    assert prep.cross_table_admissions(ward_daily_df.iloc[0:0], stays)["all_sources_match_rate"] is None


def test_to_jsonable_converts_numpy_nan_and_paths():
    payload = {"a": np.int64(3), "b": np.float32(0.5), "c": float("nan"), "d": [np.bool_(True), pd.NA],
               ("x", 1): Path("p"), "t": pd.Timestamp("2024-01-01"), "s": {2, 1}, "n": np.array([1.0, np.inf])}
    converted = prep.to_jsonable(payload)
    assert json.loads(json.dumps(converted, allow_nan=False)) == {
        "a": 3, "b": 0.5, "c": None, "d": [True, None], "x|1": "p", "t": "2024-01-01T00:00:00",
        "s": [1, 2], "n": [1.0, None],
    }


def test_prepare_datasets_requires_raw_files(isolated_paths):
    with pytest.raises(FileNotFoundError, match="01_prepare_data"):
        prep.prepare_datasets()


# ---------------------------------------------------------------------------
# End to end on a small dirty extract (isolated temp directories)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    """Run ``prepare_datasets`` once on the dirty extract with every path redirected to tmp."""
    root = tmp_path_factory.mktemp("week1")
    with pytest.MonkeyPatch.context() as patch:
        dirs = {}
        for key, env_var in PATH_ENV_VARS.items():
            dirs[key] = root / key
            dirs[key].mkdir()
            patch.setenv(env_var, str(dirs[key]))
        extracts = {WARD_DAILY: dirty_ward_daily(), PATIENT_STAYS: dirty_patient_stays()}
        write_dirty_extract(dirs["raw_dir"], *extracts.values())
        report = prep.prepare_datasets(seed=123)
        yield SimpleNamespace(report=report, dirs=dirs, extracts=extracts)


def test_e2e_writes_every_output(prepared):
    processed = prepared.dirs["processed_dir"]
    for table in TABLES:
        for split in ("clean", "train", "holdout"):
            frame = read_csv_raw(processed_path(table, split, processed))
            assert list(frame.columns) == column_names(table)
    catalog = json.loads((processed / prep.SCHEMA_CATALOG_FILENAME).read_text(encoding="utf-8"))
    assert {r["rule_id"] for r in catalog["tables"][WARD_DAILY]["rules"]} >= {"WD-H01", "WD-S06"}
    metadata = json.loads((processed / prep.METADATA_FILENAME).read_text(encoding="utf-8"))
    assert set(metadata["tables"]) == set(TABLES)
    assert set(prepared.report["outputs"]) == {f"{t}_{s}" for t in TABLES for s in ("clean", "train", "holdout")} | {
        "metadata", "schema_catalog", "data_quality_report"} | {f"metadata_{t}" for t in TABLES}
    assert not any(resolve_project_path(p).is_relative_to(PROJECT_ROOT / "data")
                   for p in prepared.report["outputs"].values())


def test_e2e_report_stores_locations_without_the_home_directory(prepared):
    """GAP-04: raw_dir, processed_dir, raw_file and outputs are stored with ``project_relative`` (no home
    directory prefix) and resolve back to the files that were written."""
    report = prepared.report
    assert report["raw_dir"] == project_relative(prepared.dirs["raw_dir"])
    assert report["processed_dir"] == project_relative(prepared.dirs["processed_dir"])
    locations = [report["raw_dir"], report["processed_dir"], *report["outputs"].values(),
                 *(report["tables"][table]["raw_file"] for table in TABLES)]
    home = Path.home().as_posix().lower()
    for value in locations:
        assert value == project_relative(value) and not value.lower().startswith(home), value
        assert "\\" not in value and resolve_project_path(value).exists(), value
    assert resolve_project_path(report["outputs"]["data_quality_report"]).resolve() == (
        prepared.dirs["processed_dir"] / prep.REPORT_FILENAME).resolve()
    saved = (prepared.dirs["processed_dir"] / prep.REPORT_FILENAME).read_text(encoding="utf-8").replace("\\\\", "\\")
    assert str(Path.home()).lower() not in saved.lower()


def test_e2e_week1_report_documents_attributes_scenarios_metadata_and_rules(prepared, tmp_path):
    """D4 regression: the Week-1 report carries the data dictionary, scenario mapping, SDV metadata,
    constraints and rule catalogue, and separates 'no longer defective' from 'exactly recovered'."""
    from bedsynth.reporting.week1 import render_week1_report

    text = render_week1_report(prepared.report, reports_dir=tmp_path / "r", figures_dir=tmp_path / "f").read_text(
        encoding="utf-8")
    for heading in ("### 1.1 Attribute dictionary", "### 1.2 Required scenarios and their reference support",
                    "### Rule catalogue", "## 9. SDV readiness: metadata and constraints"):
        assert heading in text, heading
    assert "| census_start | train | int / numerical | patients |" in text
    assert "| Varying staffing or scheduling conditions | `staffing_surplus` | ward_daily |" in text
    assert "| PS-H13 | hard | delay threshold |" in text and "metadata_patient_stays.json" in text
    assert "exactly recovered" in text and "injected issues no longer defective" in text
    before = prepared.report["tables"][WARD_DAILY]["validation"]["before"]
    assert "normalised" in before["basis"]


def test_e2e_sdv_metadata_matches_the_shipped_csvs(prepared):
    """D3 regression: every shipped metadata file validates against the CSV it describes."""
    from sdv.metadata import Metadata
    from sdv.single_table import GaussianCopulaSynthesizer

    processed = prepared.dirs["processed_dir"]
    project = Metadata.load_from_json(str(processed / prep.METADATA_FILENAME))
    for split in ("clean", "train", "holdout"):                        # key + time + train + derived
        project.validate_data({table: read_csv_raw(processed_path(table, split, processed)) for table in TABLES})
    for table in TABLES:
        train = read_csv_raw(processed_path(table, "train", processed))
        single = Metadata.load_from_json(str(processed / prep.table_metadata_filename(table)))
        assert list(single.tables) == [table]
        single.validate_data({table: train[training_columns(table)]})  # exactly what synthesizers fit
        GaussianCopulaSynthesizer(single)                              # single-table metadata is usable
    catalog = json.loads((processed / prep.SCHEMA_CATALOG_FILENAME).read_text(encoding="utf-8"))["tables"]
    for table in TABLES:
        entry = catalog[table]
        assert entry["sdv_metadata"]["training"] == f"metadata_{table}.json"
        assert entry["sdv_metadata"]["fit_columns"] == training_columns(table)
        assert entry["constraints"] and all("enforces_rules" in item for item in entry["constraints"])
        assert set(entry["rejection_enforced_rules"]) == {r["rule_id"] for r in entry["rules"] if r["severity"] == "hard"}


def test_e2e_report_is_json_serialisable_and_saved(prepared):
    text = json.dumps(prepared.report, allow_nan=False)
    saved = (prepared.dirs["processed_dir"] / prep.REPORT_FILENAME).read_text(encoding="utf-8")
    assert json.loads(saved) == json.loads(text)
    for table in TABLES:
        section = prepared.report["tables"][table]
        assert {"issues_detected", "issues_injected", "rows_dropped", "repairs", "validation",
                "cleaning_accuracy", "split", "final_rows"} <= set(section)
    assert set(prepared.report["consistency_checks"]) == {"ward_daily_continuity", "cross_table_admissions"}


def test_e2e_clean_data_is_complete_valid_and_chronological(prepared):
    processed = prepared.dirs["processed_dir"]
    for table in TABLES:
        clean = read_table(processed_path(table, "clean", processed), table)
        assert len(clean) == len(prepared.extracts[table].truth) == prepared.report["tables"][table]["final_rows"]
        assert validate_table(clean, table).all_hard_passed
        key = "record_id" if table == WARD_DAILY else "stay_id"
        assert clean[key].is_unique and clean[key].is_monotonic_increasing
        time_column = "date" if table == WARD_DAILY else "admission_datetime"
        assert clean[time_column].is_monotonic_increasing


def test_e2e_every_injected_issue_is_resolved(prepared):
    for table in TABLES:
        accuracy = prepared.report["tables"][table]["cleaning_accuracy"]
        outcomes = accuracy["injected_issues"]
        assert outcomes["logged"] == len(prepared.extracts[table].log)
        assert outcomes["resolved_rate"] == 1.0
        assert outcomes["status_counts"]["unresolved"] == outcomes["status_counts"]["row_dropped"] == 0
        assert accuracy["row_recall"] == accuracy["row_precision"] == 1.0
        assert accuracy["cell_agreement"] > 0.99
        by_type = outcomes["by_issue_type"]
        for issue in ("category_variant", "alt_date_format", "duplicate"):
            assert by_type[issue]["exact_recovery_rate"] == 1.0, (table, issue)
    assert prepared.report["tables"][PATIENT_STAYS]["cleaning_accuracy"]["injected_issues"][
        "by_issue_type"]["swapped_timestamps"]["exact_recovery_rate"] == 1.0


def test_e2e_flow_repairs_recover_true_counts(prepared):
    extract = prepared.extracts[WARD_DAILY]
    clean = read_table(processed_path(WARD_DAILY, "clean", prepared.dirs["processed_dir"]), WARD_DAILY)
    clean.index = clean["ward_id"] + "|" + clean["date"]
    truth = extract.truth.set_index(extract.truth["ward_id"] + "|" + extract.truth["date"])
    for name, column in (("invalid_census_start", "census_start"), ("invalid_discharges", "discharges"),
                         ("missing_staffing_level", "staffing_level"), ("variant_ward_id", "ward_id")):
        keys = extract.marked[name]
        assert clean.loc[keys, column].astype(str).tolist() == truth.loc[keys, column].astype(str).tolist(), name
    continuity = prepared.report["consistency_checks"]["ward_daily_continuity"]
    assert continuity["continuity_rate"] == 1.0 and continuity["missing_ward_days"] == 0


def test_e2e_report_accounts_for_detected_issues_and_drops(prepared):
    wd = prepared.report["tables"][WARD_DAILY]
    detected = wd["issues_detected"]
    assert detected["category_variant"] == {"ward_id": 6, "staffing_level": 4, "occupancy_level": 4}
    assert detected["alt_date_format"] == {"date": 6}             # 5 injected + 1 day-first duplicate copy
    assert detected["invalid_value"] == {"census_start": 3, "discharges": 3, "nurses_day_shift": 3,
                                         "avg_bed_turnaround_min": 3, "beds_closed": 3}
    assert detected["missing_value"]["staffing_level"] == 3
    assert wd["rows_dropped"] == {"unparseable_timestamp": 1, "exact_duplicate": 4, "key_duplicate": 0,
                                  "missing_key": 0, "hard_rule_violation": 0, "total": 5}
    assert wd["validation"]["before"]["hard_violations"]["WD-H01"] > 0
    assert sum(wd["validation"]["final"]["hard_violations"].values()) == 0
    methods = wd["repairs"]["by_method"]
    for method in ("temporal_continuity", "flow_identity", "within_ward_interpolation", "recomputed_from_counts"):
        assert methods.get(method, 0) >= 1, method
    assert wd["issues_injected"]["invalid_value"]["census_start"] == 3

    ps = prepared.report["tables"][PATIENT_STAYS]
    assert ps["issues_detected"]["timestamp_swap"] == {"admission_datetime|discharge_datetime": 3}
    assert ps["issues_detected"]["invalid_value"] == {"bed_wait_hours": 2, "bed_turnaround_minutes": 2,
                                                      "booking_lead_days": 2}
    assert ps["rows_dropped"]["exact_duplicate"] == 3 and ps["rows_dropped"]["total"] == 3
    summary = prepared.report["summary"][PATIENT_STAYS]
    assert summary["hard_valid_rate_final"] == 1.0 and summary["hard_valid_rate_raw"] < 1.0


def test_e2e_split_is_stratified_by_ward(prepared):
    processed = prepared.dirs["processed_dir"]
    for table in TABLES:
        key = "record_id" if table == WARD_DAILY else "stay_id"
        clean = read_csv_raw(processed_path(table, "clean", processed))
        train = read_csv_raw(processed_path(table, "train", processed))
        holdout = read_csv_raw(processed_path(table, "holdout", processed))
        assert set(train[key]).isdisjoint(holdout[key])
        assert set(train[key]) | set(holdout[key]) == set(clean[key])
        totals = clean["ward_id"].value_counts()
        shares = holdout["ward_id"].value_counts().reindex(totals.index, fill_value=0) / totals
        assert ((shares - 0.2).abs() <= 1.0 / totals + 1e-9).all(), shares.to_dict()
        split = prepared.report["tables"][table]["split"]
        assert (split["train_rows"], split["holdout_rows"]) == (len(train), len(holdout))
        assert split["stratify_by"] == "ward_id" and set(split["by_stratum"]) == set(totals.index)


# ---------------------------------------------------------------------------
# scripts/01_prepare_data.py
# ---------------------------------------------------------------------------
def load_script() -> types.ModuleType:
    spec = importlib.util.spec_from_file_location("prepare_data_script", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_script_imports_without_side_effects_and_parses_flags(isolated_paths):
    module = load_script()
    assert not any(isolated_paths["processed_dir"].iterdir())
    args = module.build_parser().parse_args(["--resimulate", "--no-inject", "--seed", "5"])
    assert (args.resimulate, args.no_inject, args.seed, args.no_report) == (True, True, 5, False)


def test_script_reuses_existing_raw_data(isolated_paths, monkeypatch, capsys):
    module = load_script()
    write_dirty_extract(isolated_paths["raw_dir"], dirty_ward_daily(), dirty_patient_stays())
    fake = types.ModuleType("bedsynth.data.simulator")
    fake.generate_source_data = lambda **kwargs: pytest.fail("simulator must not run when raw data exists")
    monkeypatch.setitem(sys.modules, "bedsynth.data.simulator", fake)
    assert module.ensure_raw_data(isolated_paths["raw_dir"], inject=False) is None
    assert "only applies when the extract is (re)generated" in capsys.readouterr().out


@pytest.fixture()
def short_simulation_settings(tmp_path, monkeypatch):
    """settings.yaml copy whose simulation covers one winter quarter (BEDSYNTH_SETTINGS)."""
    import yaml

    from bedsynth.config import DEFAULT_SETTINGS_PATH

    settings = yaml.safe_load(DEFAULT_SETTINGS_PATH.read_text(encoding="utf-8"))
    settings["simulation"].update(start_date="2023-01-01", end_date="2023-03-31", warmup_days=30, cooldown_days=30)
    path = tmp_path / "settings_short.yaml"
    path.write_text(yaml.safe_dump(settings, sort_keys=False), encoding="utf-8")
    monkeypatch.setenv("BEDSYNTH_SETTINGS", str(path))
    return path


def test_script_simulates_when_raw_missing_then_prepares(isolated_paths, short_simulation_settings, capsys):
    """The real simulator (one quarter, no injection) feeds the real pipeline and Week-1 report."""
    module = load_script()

    assert module.main(["--no-inject", "--seed", "9"]) == 0

    raw_dir, processed = isolated_paths["raw_dir"], isolated_paths["processed_dir"]
    manifest = json.loads((raw_dir / "simulation_manifest.json").read_text(encoding="utf-8"))
    assert manifest["seed"] == 9 and manifest["period"]["start_date"] == "2023-01-01"
    assert manifest["data_quality_injection"]["enabled"] is False
    for table in TABLES:
        assert raw_path(table, raw_dir).exists() and ground_truth_path(table, raw_dir).exists()
        assert processed_path(table, "train", processed).exists()
    report = json.loads((processed / "data_quality_report.json").read_text(encoding="utf-8"))
    assert report["seed"] == 9 and report["ground_truth_available"] == {WARD_DAILY: True, PATIENT_STAYS: True}
    for table in TABLES:
        head = report["summary"][table]
        # no injected defects: nothing dropped, every row valid and identical to the ground truth
        assert head["rows_dropped"] == 0 and head["hard_valid_rate_final"] == 1.0
        assert head["raw_rows"] == head["final_rows"] == manifest["row_counts"][table]["clean"]
        assert head["cell_agreement_with_truth"] == 1.0
    assert report["summary"][WARD_DAILY]["raw_rows"] == 90 * len(WARD_IDS)
    assert report["consistency_checks"]["ward_daily_continuity"]["continuity_rate"] == 1.0
    assert report["consistency_checks"]["cross_table_admissions"]["all_sources_match_rate"] == 1.0
    week1 = isolated_paths["reports_dir"] / "week1_data_preparation.md"
    text = week1.read_text(encoding="utf-8")
    assert "Week 1: data requirements and dataset preparation" in text and "Calibration:" in text
    assert list(isolated_paths["figures_dir"].glob("week1_*.png"))
    out = capsys.readouterr().out
    assert "Simulating the source extract" in out and "Week 1 data preparation (seed 9" in out
    assert "Week-1 report written" in out and "cell agreement" in out


def test_script_report_hook_is_optional(monkeypatch, capsys):
    module = load_script()
    monkeypatch.setitem(sys.modules, "bedsynth.reporting", None)   # import fails like a missing package
    assert module.render_week1_report({"seed": 1}) is None
    monkeypatch.setitem(sys.modules, "bedsynth.reporting", types.ModuleType("bedsynth.reporting"))
    assert module.render_week1_report({"seed": 1}) is None
    out = capsys.readouterr().out
    assert "not available" in out and "exposes none of" in out
