"""Week-1 cleaning pipeline: raw hospital extract -> clean, validated, split datasets.

Implements DESIGN.md §6.3. Every step is a small helper that can be used and tested on
its own; :func:`prepare_datasets` chains them for both tables.

====  ====================================================  =========================================
step  purpose                                               helper(s)
====  ====================================================  =========================================
1     profile the raw extract as delivered                  :func:`profile_raw`
2     standardise text and categories (unknown -> missing)  :func:`standardize_categories`
3     parse dates: ISO first, then day-first                :func:`parse_timestamps`
4     drop exact duplicates, then key duplicates            :func:`remove_duplicates`
5     coerce numerics; impossible values -> missing         :func:`flag_invalid_values`
6     repair / impute, most principled method first         :func:`repair_ward_daily`,
                                                            :func:`repair_patient_stays`
7     derive calendar/time features and derived columns     :func:`derive_table`
8     validate; drop rows that are still hard-invalid       :func:`drop_hard_invalid`
9     dataset checks                                        :func:`ward_daily_continuity`,
                                                            :func:`cross_table_admissions`
10    chronological ids; stratified train/holdout split     :func:`assign_ids_chronologically`,
                                                            :func:`split_train_holdout`
11    write CSVs, SDV metadata, schema catalogue, report    :func:`prepare_datasets`
12    cleaning accuracy against the simulator ground truth  :func:`cleaning_accuracy`
====  ====================================================  =========================================

:func:`clean_ward_daily` and :func:`clean_patient_stays` bundle steps 1-8 for one table and
return ``(clean_df, report)``; cleaned rows keep their *source* key (``ward_id`` + ``date`` /
the raw ``stay_id``) so they can be scored against the ground truth before step 10 assigns
the final chronological identifiers.

Every report is JSON-serialisable (see :func:`to_jsonable`) and detailed enough to write the
Week-1 data-preparation report from it: issue counts per type and column, rule violations
before/after cleaning, repairs by method, rows dropped by reason, split sizes, dataset
consistency checks and cleaning accuracy.
"""
from __future__ import annotations

import json
import logging
import math
import re
import time
import zlib
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

from bedsynth.config import get_path, get_seed, project_relative, section
from bedsynth.data.io import ground_truth_path, processed_path, raw_path, read_csv_raw, write_table
from bedsynth.schema import (
    ADMISSION_SOURCES,
    ALLOWED_DELAY_REASONS,
    DATE_FORMAT,
    DATETIME_FORMAT,
    DELAY_THRESHOLD_HOURS,
    DISCHARGE_DESTINATIONS,
    PATIENT_STAYS,
    RAW_COLUMNS,
    TABLES,
    WARD_DAILY,
    WARDS,
    WardInfo,
    add_calendar_features,
    add_stay_time_features,
    assign_ids,
    coerce_dtypes,
    column_names,
    column_spec,
    column_specs,
    derive,
    derive_patient_stays,
    derive_ward_daily,
    min_rebooked_lead_days,
    normalize_category,
    numerical_columns,
    occupancy_level_for,
    occupancy_levels,
    schema_as_dict,
    staffing_levels,
)
from bedsynth.validation import ValidationReport, rules_catalog, validate_table

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ROW_LEVEL = "*"                                  # column label for row-level issues (duplicates)
REPORT_FILENAME = "data_quality_report.json"
METADATA_FILENAME = "metadata.json"                    # SDV metadata of the CSV files (both tables)
TABLE_METADATA_FILENAME = "metadata_{table}.json"      # single-table SDV metadata the synthesizers fit
SCHEMA_CATALOG_FILENAME = "schema_catalog.json"
INJECTION_LOG_FILENAME = "injection_log.csv"
INJECTION_LOG_COLUMNS: tuple[str, ...] = (
    "table", "row_key", "column", "issue_type", "original_value", "injected_value",
)

SOURCE_KEYS: dict[str, tuple[str, ...]] = {WARD_DAILY: ("ward_id", "date"), PATIENT_STAYS: ("stay_id",)}
CHRONOLOGICAL_ORDER: dict[str, tuple[str, ...]] = {
    WARD_DAILY: ("date", "ward_id"),
    PATIENT_STAYS: ("admission_datetime", "discharge_datetime", "ward_id", "stay_id"),
}
# Accepted timestamp spellings (DESIGN.md §6.3 step 3): ISO first, then day-first.
TIMESTAMP_FORMATS: dict[str, dict[str, tuple[str, ...]]] = {
    "date": {"iso": ("%Y-%m-%d",), "day_first": ("%d/%m/%Y",)},
    "datetime": {
        "iso": ("%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S"),
        "day_first": ("%d/%m/%Y %H:%M", "%d/%m/%Y %H:%M:%S"),
    },
}
CANONICAL_TIMESTAMP_FORMAT: dict[str, str] = {"date": DATE_FORMAT, "datetime": DATETIME_FORMAT}

FLOW_CAP_FACTOR = 3        # a daily flow above 3x the ward's physical beds is a sentinel, not data
MAX_REPAIR_ROUNDS = 3      # continuity / identity repairs are iterated until nothing changes
NIGHT_HOURS: frozenset[int] = frozenset(range(20, 24)) | frozenset(range(0, 7))  # slower turnaround

_WD_FLOW_SIGNS: dict[str, int] = {   # census_end = sum(sign * term)
    "census_start": 1, "admissions_elective": 1, "admissions_emergency": 1,
    "transfers_in": 1, "transfers_out": -1, "discharges": -1,
}
_WD_CENSUS_LIKE = ("census_start", "census_end", "delayed_discharges")
_WD_FLOW_COUNTS = (
    "elective_scheduled", "elective_cancelled", "admissions_elective", "admissions_emergency",
    "transfers_in", "transfers_out", "discharges", "discharges_before_noon", "emergency_boarders",
)
_WD_TRAILING_METRICS = ("avg_bed_turnaround_min", "avg_los_days")
_WD_CAPACITY_STATE = ("escalation_beds_open", "beds_closed")
_WD_ORDERED_PAIRS = (   # (smaller, larger) count pairs kept consistent after approximate imputation
    ("elective_cancelled", "elective_scheduled"),
    ("discharges_before_noon", "discharges"),
    ("delayed_discharges", "census_start"),
)

_MEDIAN_LEVELS_WARD_DOW = (
    (("ward_id", "day_of_week"), "ward_dow_median"),
    (("ward_id",), "ward_median"),
    ((), "global_median"),
)
_MEDIAN_LEVELS_NURSES = (
    (("ward_id", "staffing_level"), "ward_staffing_band_median"),
) + _MEDIAN_LEVELS_WARD_DOW
# Continuous stay metrics are imputed by seeded hot-deck (a random observed donor of the same
# group): one group median per cell would put an artificial point mass in the data that the
# synthesizers then learn as a real mode (review finding D8). Counts keep their robust medians.
_HOT_DECK_LEVELS_WAIT = (
    (("ward_id", "admission_source"), "ward_source_hot_deck"),
    (("ward_id",), "ward_hot_deck"),
    (("admission_source",), "source_hot_deck"),
    ((), "global_hot_deck"),
)
_HOT_DECK_LEVELS_TURNAROUND = (
    (("ward_id", "_departure_period"), "ward_departure_period_hot_deck"),
    (("ward_id",), "ward_hot_deck"),
    ((), "global_hot_deck"),
)
_HOT_DECK_LEVELS_DELAY = (
    (("discharge_destination", "delay_reason"), "destination_reason_hot_deck"),
    (("delay_reason",), "reason_hot_deck"),
    ((), "global_delay_hot_deck"),
)
_HOT_DECK_SEED = 6203                    # imputation is deterministic: one stream per column
_MEDIAN_LEVELS_ELECTIVE = (
    (("ward_id",), "ward_elective_median"),
    ((), "elective_median"),
)

_STATUS_ORDER = ("exact", "repaired", "row_dropped", "unresolved", "not_evaluable")
_ISO_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")


# ---------------------------------------------------------------------------
# Column groups and small utilities
# ---------------------------------------------------------------------------
def raw_category_columns(table: str) -> list[str]:
    """Categorical/ordinal columns of the raw extract (standardised in step 2)."""
    return [c for c in RAW_COLUMNS[table] if column_spec(table, c).sdtype in ("categorical", "ordinal")]


def raw_numeric_columns(table: str) -> list[str]:
    """Numerical columns of the raw extract (coerced and range-checked in step 5)."""
    return [c for c in RAW_COLUMNS[table] if column_spec(table, c).sdtype == "numerical"]


def raw_time_columns(table: str) -> list[str]:
    """Date/datetime columns of the raw extract (parsed in step 3)."""
    return [c for c in RAW_COLUMNS[table] if column_spec(table, c).role == "time"]


def to_jsonable(obj: Any) -> Any:
    """Recursively convert to JSON-safe Python: numpy -> python, NaN/inf -> None, paths -> str."""
    if obj is None or isinstance(obj, str):
        return obj
    if obj is pd.NA or obj is pd.NaT:
        return None
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        value = float(obj)
        return value if math.isfinite(value) else None
    if isinstance(obj, dict):
        return {_json_key(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (set, frozenset)):
        return [to_jsonable(v) for v in sorted(obj, key=str)]
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, (np.ndarray, pd.Series, pd.Index)):
        return [to_jsonable(v) for v in obj.tolist()]
    if isinstance(obj, (pd.Timestamp, datetime, date)):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def _json_key(key: Any) -> str:
    """JSON object keys must be strings; tuples become ``'a|b'``."""
    if isinstance(key, tuple):
        return "|".join(str(k) for k in key)
    return str(key)


def _count(mask: Any) -> int:
    """Number of True values as a Python int."""
    return int(np.count_nonzero(np.asarray(mask, dtype=bool)))


def _rate(numerator: float, denominator: float) -> float | None:
    """Rounded ratio, or None when the denominator is zero (undefined)."""
    return round(float(numerator) / float(denominator), 6) if denominator else None


def _add_issue(issues: dict[str, dict[str, int]], issue_type: str, column: str, count: int) -> None:
    """Accumulate ``count`` into ``issues[issue_type][column]`` (zero counts are skipped)."""
    if count:
        bucket = issues.setdefault(issue_type, {})
        bucket[column] = bucket.get(column, 0) + int(count)


def _merge_issues(*parts: dict[str, dict[str, int]]) -> dict[str, dict[str, int]]:
    """Sum several ``{issue_type: {column: count}}`` mappings."""
    merged: dict[str, dict[str, int]] = {}
    for part in parts:
        for issue_type, per_column in (part or {}).items():
            for column, count in per_column.items():
                _add_issue(merged, issue_type, column, count)
    return merged


def _round_values(values: np.ndarray, decimals: int | None) -> np.ndarray:
    """Round half away from zero for counts (decimals=0), numpy rounding otherwise."""
    if decimals is None:
        return values
    if decimals == 0:
        return np.sign(values) * np.floor(np.abs(values) + 0.5)
    return np.round(values, decimals)


def _ward_values(attribute: str) -> dict[str, Any]:
    """``{ward_id: WardInfo.<attribute>}`` for every catalogued ward."""
    return {ward_id: getattr(info, attribute) for ward_id, info in WARDS.items()}


def _ward_lookup(frame: pd.DataFrame, attribute: str) -> np.ndarray:
    """Ward catalogue attribute per row as float (NaN for unknown wards)."""
    return pd.to_numeric(frame["ward_id"].map(_ward_values(attribute)), errors="coerce").to_numpy(dtype=float)


def source_keys(frame: pd.DataFrame, table: str) -> pd.Series:
    """Source key of every row: ``'<ward_id>|<date>'`` for ward_daily, the stay_id for stays."""
    if table == WARD_DAILY:
        return frame["ward_id"].astype(str) + "|" + frame["date"].astype(str)
    return frame["stay_id"].astype(str)


def _require_raw_columns(raw_df: pd.DataFrame, table: str) -> None:
    """Fail fast on a malformed extract (a missing column is a schema error, not dirty data)."""
    missing = [c for c in RAW_COLUMNS[table] if c not in raw_df.columns]
    if missing:
        raise ValueError(
            f"Raw {table} extract is missing columns {missing}; expected schema.RAW_COLUMNS['{table}']."
        )


class _RepairLog:
    """Counts filled/corrected cells per column and method; remembers approximate fills."""

    def __init__(self, n_rows: int) -> None:
        """Start an empty log for a frame of ``n_rows`` rows."""
        self.n_rows = n_rows
        self.counts: dict[str, dict[str, int]] = {}
        self.method_kind: dict[str, str] = {}
        self.adjustments: dict[str, int] = {}
        self._approximate: dict[str, np.ndarray] = {}

    def record(self, column: str, method: str, mask: np.ndarray, approximate: bool = False) -> None:
        """Log that ``mask`` cells of ``column`` were filled/corrected by ``method``."""
        n = _count(mask)
        if not n:
            return
        per_column = self.counts.setdefault(column, {})
        per_column[method] = per_column.get(method, 0) + n
        self.method_kind[method] = "approximate" if approximate else "exact"
        if approximate:
            current = self._approximate.setdefault(column, np.zeros(self.n_rows, dtype=bool))
            current |= np.asarray(mask, dtype=bool)

    def adjust(self, column: str, mask: np.ndarray) -> None:
        """Log approximate values nudged to satisfy a relational constraint."""
        if _count(mask):
            self.adjustments[column] = self.adjustments.get(column, 0) + _count(mask)

    def approximate(self, column: str) -> np.ndarray:
        """Mask of ``column`` cells filled by an approximate (statistical) method."""
        return self._approximate.get(column, np.zeros(self.n_rows, dtype=bool))

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe summary: totals, exact vs approximate cells, counts by column and method."""
        by_method: dict[str, int] = {}
        for per_column in self.counts.values():
            for method, n in per_column.items():
                by_method[method] = by_method.get(method, 0) + n
        exact = sum(n for m, n in by_method.items() if self.method_kind.get(m) == "exact")
        return {
            "cells_repaired": sum(by_method.values()),
            "exact_cells": exact,
            "approximate_cells": sum(by_method.values()) - exact,
            "by_column": self.counts,
            "by_method": dict(sorted(by_method.items(), key=lambda kv: (-kv[1], kv[0]))),
            "method_kind": dict(sorted(self.method_kind.items())),
            "constraint_adjustments": self.adjustments,
        }


def _fill(frame: pd.DataFrame, column: str, values: Any, mask: np.ndarray, log: _RepairLog,
          method: str, approximate: bool = False) -> int:
    """Write ``values`` into the numeric ``column`` where ``mask`` and log the repair."""
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return 0
    current = frame[column].to_numpy(dtype=float, copy=True)
    current[mask] = np.broadcast_to(np.asarray(values, dtype=float), current.shape)[mask]
    frame[column] = current
    log.record(column, method, mask, approximate)
    return _count(mask)


def _fill_labels(frame: pd.DataFrame, column: str, values: Any, mask: np.ndarray, log: _RepairLog,
                 method: str, approximate: bool = False) -> int:
    """Categorical counterpart of :func:`_fill`."""
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return 0
    current = frame[column].to_numpy(dtype=object, copy=True)
    if isinstance(values, np.ndarray):
        current[mask] = values[mask]
    else:
        current[mask] = values
    frame[column] = current
    log.record(column, method, mask, approximate)
    return _count(mask)


def _rank_values(series: pd.Series, vocabulary: Sequence[str]) -> list[str]:
    """Distinct values by descending frequency (ties in vocabulary order)."""
    counts = series.value_counts()
    order = {value: i for i, value in enumerate(vocabulary)}
    return sorted(counts.index, key=lambda v: (-int(counts[v]), order.get(v, len(order)), str(v)))


# ---------------------------------------------------------------------------
# Step 1 - profiling
# ---------------------------------------------------------------------------
def profile_raw(raw_df: pd.DataFrame, table: str) -> dict[str, Any]:
    """Step 1: describe the raw extract exactly as delivered (nothing is changed).

    Rows, exact duplicates, missing values per column, category spellings (canonical /
    variant / unknown), timestamp formats, non-numeric cells and the rule violations of a
    provisional derivation of the raw rows (the "before" picture of the cleaning; spellings and
    date formats are normalised first, so it counts operational defects, not parsing gaps).
    """
    expected = list(RAW_COLUMNS[table])
    present = [c for c in expected if c in raw_df.columns]
    return {
        "rows": int(len(raw_df)),
        "columns": int(raw_df.shape[1]),
        "missing_columns": [c for c in expected if c not in raw_df.columns],
        "unexpected_columns": [str(c) for c in raw_df.columns if c not in expected],
        "exact_duplicate_rows": int(raw_df.duplicated().sum()),
        "missing_values": {c: int(raw_df[c].isna().sum()) for c in present},
        "categories": {
            c: _category_profile(raw_df[c], c) for c in raw_category_columns(table) if c in raw_df.columns
        },
        "timestamps": {
            c: _timestamp_profile(raw_df[c], column_spec(table, c).dtype)
            for c in raw_time_columns(table) if c in raw_df.columns
        },
        "non_numeric_values": {
            c: _non_numeric_count(raw_df[c]) for c in raw_numeric_columns(table) if c in raw_df.columns
        },
        "provisional_validation": _provisional_validation(raw_df, table),
    }


def _category_profile(series: pd.Series, column: str) -> dict[str, Any]:
    """Counts of canonical, variant (mappable), unknown and missing spellings."""
    counts = series.value_counts(dropna=True)
    mapping = {value: canonical_category(column, value) for value in counts.index}
    canonical = sum(int(n) for v, n in counts.items() if mapping[v] == v)
    unknown_values = [v for v in counts.index if mapping[v] is None]
    return {
        "canonical": canonical,
        "variant": int(counts.sum()) - canonical - sum(int(counts[v]) for v in unknown_values),
        "unknown": sum(int(counts[v]) for v in unknown_values),
        "missing": int(series.isna().sum()),
        "distinct_spellings": int(len(counts)),
        "unknown_examples": sorted(str(v) for v in unknown_values)[:10],
    }


def _timestamp_profile(series: pd.Series, kind: str) -> dict[str, Any]:
    """Counts of ISO, day-first, unparseable and missing timestamps."""
    _, how = parse_timestamp_series(series, kind)
    unparseable = series[how == "unparseable"]
    return {
        **{label: _count(how == label) for label in ("iso", "day_first", "unparseable", "missing")},
        "unparseable_examples": sorted({str(v) for v in unparseable})[:10],
    }


def _non_numeric_count(series: pd.Series) -> int:
    """Present cells that cannot be read as numbers."""
    return _count(series.notna() & pd.to_numeric(series, errors="coerce").isna())


PROVISIONAL_VALIDATION_BASIS = (
    "every raw row (duplicates included) after category spellings and day-first dates were normalised; "
    "no value was repaired, imputed or dropped"
)


def _provisional_validation(raw_df: pd.DataFrame, table: str) -> dict[str, Any]:
    """Validate the raw rows before any repair: the "before" picture of the cleaning.

    Category aliases are mapped to canonical codes and both accepted date formats are parsed
    first (unknown spellings and unparseable dates become missing), so the counts measure the
    operational defects of the extract - impossible values, gaps, broken identities - rather
    than spellings the validator cannot read. Numbers are coerced; nothing is imputed.
    """
    try:
        frame = raw_df[[c for c in RAW_COLUMNS[table] if c in raw_df.columns]].reset_index(drop=True)
        frame, _ = standardize_categories(frame, table)
        for column in raw_time_columns(table):
            if column in frame.columns:
                kind = column_spec(table, column).dtype
                frame[column] = format_timestamps(parse_timestamp_series(frame[column], kind)[0], kind)
        for column in raw_numeric_columns(table):
            if column in frame.columns:
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame = add_calendar_features(frame) if table == WARD_DAILY else add_stay_time_features(frame)
        report = validate_table(derive(table, frame), table)
    except (KeyError, ValueError, TypeError) as exc:
        logger.warning("Provisional validation of raw %s failed: %s", table, exc)
        return {"error": f"{type(exc).__name__}: {exc}"}
    return {**validation_summary(report), "basis": PROVISIONAL_VALIDATION_BASIS}


def validation_summary(report: ValidationReport) -> dict[str, Any]:
    """Compact, JSON-safe view of a validation report (violations for every rule)."""
    return {
        "rows": int(report.n_rows),
        "hard_valid_rows": report.n_hard_valid,
        "hard_valid_rate": _rate(report.n_hard_valid, report.n_rows),
        "hard_violations": report.violation_counts("hard"),
        "soft_violations": report.violation_counts("soft"),
    }


# ---------------------------------------------------------------------------
# Step 2 - strings and categories
# ---------------------------------------------------------------------------
def canonical_category(column: str, value: Any) -> Any:
    """Canonical code of a raw category spelling (None when unknown or missing).

    Delegates to :func:`schema.normalize_category`, which handles aliases, case, surrounding
    whitespace and collapsed internal whitespace / underscores (shared with the injector,
    the generator and the API).
    """
    return normalize_category(column, value)


def _strip_text(series: pd.Series) -> pd.Series:
    """Trim surrounding whitespace of text cells; empty strings become missing."""
    stripped = series.map(lambda v: v.strip() if isinstance(v, str) else v)
    return stripped.where(stripped != "", np.nan)


def standardize_categories(df: pd.DataFrame, table: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Step 2: trim text cells and map every category spelling to its canonical code.

    Unknown spellings become missing (and are imputed or dropped later). Mapping is done
    once per distinct raw value, so the cost does not grow with the number of rows.
    """
    out = df.copy()
    issues: dict[str, dict[str, int]] = {}
    columns: dict[str, Any] = {}
    for column in [*raw_time_columns(table), *SOURCE_KEYS[table]]:
        if column in out.columns and column not in raw_category_columns(table):
            out[column] = _strip_text(out[column])
    for column in raw_category_columns(table):
        if column not in out.columns:
            continue
        series = out[column]
        mapping = {value: canonical_category(column, value) for value in series.dropna().unique()}
        standardised = series.map(mapping)
        present = series.notna().to_numpy()
        known = standardised.notna().to_numpy()
        changed = present & known & (standardised.astype(object) != series.astype(object)).to_numpy()
        unknown = present & ~known
        columns[column] = {
            "canonical": _count(present & known & ~changed),
            "standardised": _count(changed),
            "unknown_to_missing": _count(unknown),
            "missing": _count(~present),
            "unknown_values": sorted({str(v) for v in series[unknown]})[:10],
        }
        _add_issue(issues, "category_variant", column, _count(changed))
        _add_issue(issues, "unknown_category", column, _count(unknown))
        out[column] = standardised.astype(object).where(known, np.nan)
    return out, {"columns": columns, "issues": issues}


# ---------------------------------------------------------------------------
# Step 3 - timestamps
# ---------------------------------------------------------------------------
def parse_timestamp_series(series: pd.Series, kind: str = "date") -> tuple[pd.Series, pd.Series]:
    """Parse dates (``kind='date'``) or datetimes (``'datetime'``): ISO first, then day-first.

    Returns ``(timestamps, how)`` aligned to ``series``: ``how`` is ``'iso'``, ``'day_first'``,
    ``'unparseable'`` or ``'missing'`` per cell; unparsed cells are NaT.
    """
    formats = TIMESTAMP_FORMATS[kind]
    values = series.map(lambda v: v.strip() if isinstance(v, str) else v).to_numpy(dtype=object)
    present = ~pd.isna(values) & (values != "")
    parsed = np.full(len(values), np.datetime64("NaT"), dtype="datetime64[ns]")
    how = np.where(present, "unparseable", "missing").astype(object)
    for label in ("iso", "day_first"):
        for fmt in formats[label]:
            todo = np.flatnonzero(present & np.isnat(parsed))
            if not todo.size:
                break
            attempt = pd.to_datetime(pd.Series(values[todo]).astype(str), format=fmt, errors="coerce")
            attempt = attempt.to_numpy(dtype="datetime64[ns]")
            hit = ~np.isnat(attempt)
            parsed[todo[hit]] = attempt[hit]
            how[todo[hit]] = label
    return pd.Series(parsed, index=series.index), pd.Series(how, index=series.index, dtype=object)


def format_timestamps(parsed: pd.Series, kind: str = "date") -> pd.Series:
    """Canonical ISO text (``YYYY-MM-DD`` / ``YYYY-MM-DD HH:MM``) of parsed timestamps; NaT -> NaN.

    Vectorised with ``numpy.datetime_as_string`` (``Series.dt.strftime`` formats element by
    element and dominates the runtime on 50k-row extracts).
    """
    values = parsed.to_numpy(dtype="datetime64[ns]")
    text = np.datetime_as_string(values, unit="D" if kind == "date" else "m").astype(object)
    if kind == "datetime":
        text = np.array([t.replace("T", " ") for t in text], dtype=object)
    text[np.isnat(values)] = np.nan
    return pd.Series(text, index=parsed.index, dtype=object)


def parse_timestamps(df: pd.DataFrame, table: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Step 3: rewrite every timestamp in canonical ISO form; drop rows that cannot be dated.

    ward_daily rows also receive their calendar features (month, day_of_week) here because
    step 6 imputes within ward x day-of-week groups.
    """
    out = df.reset_index(drop=True).copy()
    keep = np.ones(len(out), dtype=bool)
    columns: dict[str, Any] = {}
    issues: dict[str, dict[str, int]] = {}
    for column in raw_time_columns(table):
        kind = column_spec(table, column).dtype
        parsed, how = parse_timestamp_series(out[column], kind)
        counts = {label: _count(how == label) for label in ("iso", "day_first", "unparseable", "missing")}
        counts["unparseable_examples"] = sorted({str(v) for v in out.loc[how == "unparseable", column]})[:10]
        columns[column] = counts
        _add_issue(issues, "alt_date_format", column, counts["day_first"])
        _add_issue(issues, "unparseable_date", column, counts["unparseable"])
        out[column] = format_timestamps(parsed, kind)
        keep &= parsed.notna().to_numpy()
    out = out[keep].reset_index(drop=True)
    if table == WARD_DAILY:
        out = add_calendar_features(out)
    return out, {"columns": columns, "rows_dropped": _count(~keep), "issues": issues}


# ---------------------------------------------------------------------------
# Step 4 - duplicates
# ---------------------------------------------------------------------------
def remove_duplicates(df: pd.DataFrame, table: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Step 4: drop exact duplicate rows, then rows repeating a key (keep the first).

    Keys: ``ward_id + date`` (ward-days without either are dropped - they cannot be placed in
    a ward series) and ``stay_id`` (stays without a source id are kept: ids are reassigned
    in step 10).
    """
    keys = list(SOURCE_KEYS[table])
    exact = df.duplicated(keep="first").to_numpy()
    frame = df[~exact].reset_index(drop=True)
    key_missing = frame[keys].isna().any(axis=1).to_numpy()
    missing_key_dropped = key_missing if table == WARD_DAILY else np.zeros(len(frame), dtype=bool)
    key_duplicate = frame.duplicated(subset=keys, keep="first").to_numpy() & ~key_missing
    drop = missing_key_dropped | key_duplicate
    report = {
        "exact_duplicates": _count(exact),
        "key_duplicates": _count(key_duplicate),
        "missing_key": _count(missing_key_dropped),
        "rows_without_source_key_kept": _count(key_missing & ~missing_key_dropped),
        "rows_dropped": _count(exact) + _count(drop),
        "key_duplicate_examples": source_keys(frame[key_duplicate], table).head(5).tolist(),
        "issues": {},
    }
    _add_issue(report["issues"], "exact_duplicate", ROW_LEVEL, report["exact_duplicates"])
    _add_issue(report["issues"], "key_duplicate", ROW_LEVEL, report["key_duplicates"])
    _add_issue(report["issues"], "missing_key", ROW_LEVEL, report["missing_key"])
    return frame[~drop].reset_index(drop=True), report


# ---------------------------------------------------------------------------
# Step 5 - numeric coercion and impossible values
# ---------------------------------------------------------------------------
def _ward_daily_cap(column: str, info: WardInfo) -> float | None:
    """Ward-specific physical maximum of a ward_daily count (None = schema bound only)."""
    if column == "escalation_beds_open":
        return info.max_escalation_beds
    if column == "beds_closed":
        return info.max_physical_beds - 1
    if column in ("nurses_day_shift", "nurses_night_shift"):
        return max(info.max_physical_beds, 2 * info.required_nurses_day)
    if column in _WD_CENSUS_LIKE:
        return info.max_physical_beds
    if column in _WD_FLOW_COUNTS:
        return FLOW_CAP_FACTOR * info.max_physical_beds
    return None


def physical_bounds(frame: pd.DataFrame, table: str, column: str) -> tuple[np.ndarray, np.ndarray]:
    """Per-row physically possible ``[low, high]`` of a numeric column.

    Schema bounds everywhere; ward_daily counts are additionally capped by the ward's bed
    catalogue (e.g. census <= base + escalation beds, flows <= 3x physical beds).
    """
    spec = column_spec(table, column)
    n_rows = len(frame)
    low = np.full(n_rows, -np.inf if spec.min_value is None else float(spec.min_value))
    high = np.full(n_rows, np.inf if spec.max_value is None else float(spec.max_value))
    if table == WARD_DAILY and "ward_id" in frame.columns:
        caps = {w: _ward_daily_cap(column, info) for w, info in WARDS.items()}
        if any(cap is not None for cap in caps.values()):
            ward_caps = pd.to_numeric(frame["ward_id"].map(caps), errors="coerce").to_numpy(dtype=float)
            high = np.where(np.isnan(ward_caps), high, np.minimum(high, ward_caps))
    return low, high


def flag_invalid_values(df: pd.DataFrame, table: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Step 5: coerce numerics; physically impossible values become missing (counted).

    Impossible = non-numeric, non-finite, fractional counts, below the physical minimum
    (negative counts, zero nurses, zero turnaround) or above the physical maximum (999-style
    sentinels). For stays, swapped admission/discharge timestamps are swapped back and values
    that contradict the admission source or the length of stay are invalidated.
    """
    out = df.copy()
    columns: dict[str, dict[str, int]] = {}
    issues: dict[str, dict[str, int]] = {}
    for column in raw_numeric_columns(table):
        reasons = _invalidate_numeric(out, table, column)
        if reasons:
            columns[column] = reasons
            _add_issue(issues, "invalid_value", column, sum(reasons.values()))
    report: dict[str, Any] = {"columns": columns}
    if table == WARD_DAILY:
        mismatch = _check_catalogue_capacity(out)
        if mismatch:
            columns.setdefault("bed_capacity", {})["catalogue_mismatch"] = mismatch
            _add_issue(issues, "invalid_value", "bed_capacity", mismatch)
    else:
        swaps = repair_timestamp_swaps(out)
        report["timestamp_swaps_repaired"] = swaps
        _add_issue(issues, "timestamp_swap", "admission_datetime|discharge_datetime", swaps)
        for column, n in _invalidate_stay_relations(out).items():
            columns.setdefault(column, {})["inconsistent"] = n
            _add_issue(issues, "invalid_value", column, n)
    report["cells_invalidated"] = sum(
        n for column, reasons in columns.items() for reason, n in reasons.items() if reason != "catalogue_mismatch"
    )
    report["issues"] = issues
    return out, report


def _invalidate_numeric(frame: pd.DataFrame, table: str, column: str) -> dict[str, int]:
    """Coerce one column to float in place, blanking impossible values; counts per reason."""
    raw = frame[column]
    values = pd.to_numeric(raw, errors="coerce").astype(float).to_numpy(copy=True)
    non_numeric = raw.notna().to_numpy() & np.isnan(values)
    non_finite = np.isinf(values)
    finite = np.isfinite(values)
    non_integer = np.zeros(len(values), dtype=bool)
    if column_spec(table, column).dtype == "int":
        non_integer = finite & (np.floor(values) != values)
    low, high = physical_bounds(frame, table, column)
    below = finite & (values < low)
    above = finite & (values > high)
    values[non_finite | non_integer | below | above] = np.nan
    frame[column] = values
    reasons = {
        "non_numeric": _count(non_numeric),
        "non_finite": _count(non_finite),
        "non_integer": _count(non_integer),
        "below_minimum": _count(below),
        "above_maximum": _count(above),
    }
    return {reason: n for reason, n in reasons.items() if n}


def _check_catalogue_capacity(frame: pd.DataFrame) -> int:
    """Count raw bed_capacity values that disagree with the ward catalogue (fixed in step 6)."""
    catalogue = _ward_lookup(frame, "bed_capacity")
    raw = frame["bed_capacity"].to_numpy(dtype=float)
    return _count(~np.isnan(raw) & ~np.isnan(catalogue) & (raw != catalogue))


def repair_timestamp_swaps(frame: pd.DataFrame) -> int:
    """Swap admission/discharge timestamps back where discharge precedes admission (in place)."""
    admission = pd.to_datetime(frame["admission_datetime"], format=DATETIME_FORMAT, errors="coerce")
    discharge = pd.to_datetime(frame["discharge_datetime"], format=DATETIME_FORMAT, errors="coerce")
    swapped = (discharge < admission).to_numpy()
    if swapped.any():
        adm = frame["admission_datetime"].to_numpy(dtype=object)
        dis = frame["discharge_datetime"].to_numpy(dtype=object)
        frame["admission_datetime"] = np.where(swapped, dis, adm)
        frame["discharge_datetime"] = np.where(swapped, adm, dis)
    return _count(swapped)


def stay_los_hours(frame: pd.DataFrame) -> np.ndarray:
    """Hour-resolution length of stay from the ISO timestamps (NaN where undefined)."""
    times = add_stay_time_features(frame[["admission_datetime", "discharge_datetime"]])
    hours = times["los_nights"] * 24 + times["discharge_hour"] - times["admission_hour"]
    return hours.astype("float64").to_numpy()


def _invalidate_stay_relations(frame: pd.DataFrame) -> dict[str, int]:
    """Invalidate stay values that contradict the admission source or the stay length."""
    source = frame["admission_source"]
    elective = (source == "elective").to_numpy()
    non_elective = source.isin(["emergency", "transfer"]).to_numpy()
    lead = frame["booking_lead_days"].to_numpy(dtype=float, copy=True)
    rescheduled = frame["times_rescheduled"].to_numpy(dtype=float, copy=True)
    delay = frame["discharge_delay_hours"].to_numpy(dtype=float, copy=True)
    bad_lead = (elective & (lead < 1)) | (non_elective & ~np.isnan(lead) & (lead != 0))
    bad_rescheduled = non_elective & ~np.isnan(rescheduled) & (rescheduled != 0)
    bad_delay = delay > stay_los_hours(frame) + 1e-9
    for column, values, mask in (
        ("booking_lead_days", lead, bad_lead),
        ("times_rescheduled", rescheduled, bad_rescheduled),
        ("discharge_delay_hours", delay, bad_delay),
    ):
        values[mask] = np.nan
        frame[column] = values
    counts = {
        "booking_lead_days": _count(bad_lead),
        "times_rescheduled": _count(bad_rescheduled),
        "discharge_delay_hours": _count(bad_delay),
    }
    return {column: n for column, n in counts.items() if n}


# ---------------------------------------------------------------------------
# Step 6 - shared imputation helpers
# ---------------------------------------------------------------------------
def _impute_group_median(
    frame: pd.DataFrame,
    column: str,
    levels: Sequence[tuple[tuple[str, ...], str]],
    log: _RepairLog,
    *,
    target: np.ndarray | None = None,
    pool: np.ndarray | None = None,
    decimals: int | None = None,
    lower: Any = None,
    upper: Any = None,
) -> int:
    """Fill missing ``column`` values with group medians, trying ``levels`` in order.

    ``levels`` pairs grouping keys with the method name logged for that level (empty keys =
    overall median). Medians use observed values of rows in ``pool`` only; only missing
    rows in ``target`` are filled. Filled values are rounded and clipped to [lower, upper].
    """
    values = frame[column].to_numpy(dtype=float, copy=True)
    todo = np.isnan(values) if target is None else (np.asarray(target, dtype=bool) & np.isnan(values))
    if not todo.any():
        return 0
    in_pool = np.ones(len(values), dtype=bool) if pool is None else np.asarray(pool, dtype=bool)
    observed = pd.Series(np.where(in_pool, values, np.nan), index=frame.index)
    filled_total = 0
    for keys, method in levels:
        if not todo.any():
            break
        if keys:
            medians = observed.groupby([frame[k] for k in keys], dropna=True).transform("median")
            medians = medians.to_numpy(dtype=float)
        else:
            medians = np.full(len(values), observed.median())
        fill = todo & ~np.isnan(medians)
        if not fill.any():
            continue
        candidate = _round_values(medians, decimals)
        if lower is not None:
            candidate = np.fmax(candidate, lower)
        if upper is not None:
            candidate = np.fmin(candidate, upper)
        values[fill] = candidate[fill]
        log.record(column, method, fill, approximate=True)
        todo &= ~fill
        filled_total += _count(fill)
    frame[column] = values
    return filled_total


def _impute_group_hot_deck(
    frame: pd.DataFrame,
    column: str,
    levels: Sequence[tuple[tuple[str, ...], str]],
    log: _RepairLog,
    *,
    target: np.ndarray | None = None,
    pool: np.ndarray | None = None,
    decimals: int | None = None,
    lower: Any = None,
    upper: Any = None,
) -> int:
    """Fill missing ``column`` values with a random observed donor of the same group.

    Seeded hot-deck imputation (one deterministic stream per column): every missing row in
    ``target`` draws a value from the observed rows in ``pool`` that share its grouping keys,
    trying ``levels`` in order (empty keys = any observed row). Unlike a group median this keeps
    the within-group spread and adds no point mass. Values are rounded and clipped to
    [lower, upper] (scalars or per-row arrays).
    """
    values = frame[column].to_numpy(dtype=float, copy=True)
    todo = np.isnan(values) if target is None else (np.asarray(target, dtype=bool) & np.isnan(values))
    if not todo.any():
        return 0
    in_pool = np.ones(len(values), dtype=bool) if pool is None else np.asarray(pool, dtype=bool)
    donor_rows = in_pool & ~np.isnan(values)
    rng = np.random.default_rng([_HOT_DECK_SEED, zlib.crc32(column.encode("utf-8"))])
    filled_total = 0
    for keys, method in levels:
        if not todo.any():
            break
        if keys:
            group = frame.groupby(list(keys), dropna=True, sort=True).ngroup().to_numpy()
        else:
            group = np.zeros(len(values), dtype=np.int64)
        fill = np.zeros(len(values), dtype=bool)
        for group_id in np.unique(group[todo & (group >= 0)]):
            donors = values[donor_rows & (group == group_id)]
            if donors.size == 0:
                continue
            rows = np.flatnonzero(todo & (group == group_id))
            values[rows] = rng.choice(donors, size=rows.size, replace=True)
            fill[rows] = True
        if not fill.any():
            continue
        candidate = _round_values(values, decimals)
        if lower is not None:
            candidate = np.fmax(candidate, lower)
        if upper is not None:
            candidate = np.fmin(candidate, upper)
        values[fill] = candidate[fill]
        log.record(column, method, fill, approximate=True)
        todo &= ~fill
        filled_total += _count(fill)
    frame[column] = values
    return filled_total


def _group_mode(frame: pd.DataFrame, column: str, keys: Sequence[str]) -> np.ndarray:
    """Most frequent observed ``column`` value within each key group, per row (None if none)."""
    keys = list(keys)
    observed = frame[[*keys, column]].dropna()
    if observed.empty:
        return np.full(len(frame), None, dtype=object)
    counts = observed.groupby([*keys, column]).size().reset_index(name="_n")
    counts = counts.sort_values(["_n", column], ascending=[False, True], kind="mergesort")
    best = counts.drop_duplicates(subset=keys, keep="first").drop(columns="_n")
    merged = frame[keys].merge(best, on=keys, how="left")
    return merged[column].astype(object).where(merged[column].notna(), None).to_numpy(dtype=object)


# ---------------------------------------------------------------------------
# Step 6a - ward_daily repairs
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class _Neighbours:
    """Same-ward previous/next calendar day of every row (frame sorted by ward_id, date)."""

    has_prev: np.ndarray
    has_next: np.ndarray

    @classmethod
    def from_frame(cls, frame: pd.DataFrame) -> "_Neighbours":
        """Adjacency of rows that are consecutive calendar days of the same ward."""
        n_rows = len(frame)
        has_prev = np.zeros(n_rows, dtype=bool)
        if n_rows > 1:
            wards = frame["ward_id"].to_numpy(dtype=object)
            dates = pd.to_datetime(frame["date"], format=DATE_FORMAT, errors="coerce").to_numpy()
            has_prev[1:] = (wards[1:] == wards[:-1]) & ((dates[1:] - dates[:-1]) == np.timedelta64(1, "D"))
        has_next = np.zeros(n_rows, dtype=bool)
        has_next[:-1] = has_prev[1:]
        return cls(has_prev, has_next)

    def previous(self, values: np.ndarray) -> np.ndarray:
        """Value on the same ward's previous day (NaN when that day is absent)."""
        shifted = np.full(len(values), np.nan)
        shifted[1:] = values[:-1]
        return np.where(self.has_prev, shifted, np.nan)

    def following(self, values: np.ndarray) -> np.ndarray:
        """Value on the same ward's next day (NaN when that day is absent)."""
        shifted = np.full(len(values), np.nan)
        shifted[:-1] = values[1:]
        return np.where(self.has_next, shifted, np.nan)


def repair_ward_daily(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Step 6 for ward_daily: fill every missing train value, most principled method first.

    1. ``bed_capacity`` from the ward catalogue; structural zeros (no electives/transfers on
       wards that take none).
    2. Temporal continuity ``census_start(d) = census_end(d-1)``; elective identity
       ``admissions_elective = scheduled - cancelled``; flow identity solved for a single
       missing term, anchored on the raw ``census_end`` (or the next day's census_start).
       Iterated until nothing changes.
    3. Capacity state (escalation/closed beds) from adjacent days, constrained to keep the
       census within the available beds and to reproduce the recorded occupancy band.
    4. 7-day trailing metrics interpolated in time within each ward series; bed wait = 0
       when there were no emergency admissions.
    5. Day nurses by ward x declared staffing band; ``staffing_level`` recomputed from
       counts; night nurses by ward x staffing band.
    6. Remaining gaps -> ward x day-of-week median (rounded for counts), then ward, then
       overall median; approximate values are nudged to respect ordered count pairs.
    7. ``occupancy_level`` recomputed from census_end / available_beds.

    Raw-label disagreements are reported. The raw derived columns (census_end,
    admissions_elective) serve only as repair evidence: step 7 recomputes them.
    """
    out = df.copy()
    if "day_of_week" not in out.columns or "month" not in out.columns:
        out = add_calendar_features(out)
    out = out.sort_values(["ward_id", "date"], kind="mergesort").reset_index(drop=True)
    for column in raw_numeric_columns(WARD_DAILY):
        out[column] = pd.to_numeric(out[column], errors="coerce").astype(float)
    log = _RepairLog(len(out))
    neighbours = _Neighbours.from_frame(out)
    raw_staffing = out["staffing_level"].copy()
    raw_occupancy = out["occupancy_level"].copy()

    _apply_catalogue_capacity(out, log)
    _fill_structural_zeros(out, log)
    for _ in range(MAX_REPAIR_ROUNDS):
        progress = _fill_census_start_from_continuity(out, neighbours, log)
        progress += _solve_elective_identity(out, log)
        progress += _solve_flow_identity(out, neighbours, log)
        if not progress:
            break
    flow_mismatches = _count_flow_identity_mismatches(out)
    _repair_capacity_state(out, neighbours, raw_occupancy, log)
    _interpolate_trailing_metrics(out, log)
    _apply_wait_definition(out, log)
    _impute_group_median(out, "nurses_day_shift", _MEDIAN_LEVELS_NURSES, log, decimals=0,
                         lower=1, upper=physical_bounds(out, WARD_DAILY, "nurses_day_shift")[1])
    staffing_labels = _recompute_staffing_level(out, raw_staffing, log)
    _impute_remaining_ward_daily(out, log)
    _clamp_approximate_counts(out, log)
    occupancy_labels = _recompute_occupancy_level(out, raw_occupancy, log)

    train_numeric = numerical_columns(WARD_DAILY, ("train",))
    unrepaired = {c: int(out[c].isna().sum()) for c in train_numeric if out[c].isna().any()}
    out = out.sort_values(["date", "ward_id"], kind="mergesort").reset_index(drop=True)
    report = {
        **log.to_dict(),
        "label_recomputation": {"staffing_level": staffing_labels, "occupancy_level": occupancy_labels},
        "flow_identity_mismatches": flow_mismatches,
        "unrepaired": unrepaired,
    }
    return out, report


def _apply_catalogue_capacity(out: pd.DataFrame, log: _RepairLog) -> None:
    """bed_capacity is a catalogue attribute: overwrite missing or disagreeing raw values."""
    catalogue = _ward_lookup(out, "bed_capacity")
    raw = out["bed_capacity"].to_numpy(dtype=float)
    changed = ~np.isnan(catalogue) & (np.isnan(raw) | (raw != catalogue))
    out["bed_capacity"] = catalogue
    log.record("bed_capacity", "ward_catalogue", changed)


def _fill_structural_zeros(out: pd.DataFrame, log: _RepairLog) -> None:
    """Wards that take no electives / transfers have structurally zero counts."""
    no_electives = _ward_lookup(out, "accepts_electives") == 0
    no_transfers = _ward_lookup(out, "accepts_transfers") == 0
    for column in ("elective_scheduled", "elective_cancelled"):
        _fill(out, column, 0.0, out[column].isna().to_numpy() & no_electives, log, "structural_zero")
    evidence = out["admissions_elective"].to_numpy(dtype=float, copy=True)
    evidence[np.isnan(evidence) & no_electives] = 0.0
    out["admissions_elective"] = evidence
    _fill(out, "transfers_in", 0.0, out["transfers_in"].isna().to_numpy() & no_transfers, log, "structural_zero")


def _within_bounds(out: pd.DataFrame, column: str, values: np.ndarray) -> np.ndarray:
    """Solved values are accepted only when whole and physically possible."""
    low, high = physical_bounds(out, WARD_DAILY, column)
    return np.isfinite(values) & (np.floor(values) == values) & (values >= low) & (values <= high)


def _fill_census_start_from_continuity(out: pd.DataFrame, neighbours: _Neighbours, log: _RepairLog) -> int:
    """census_start(d) = census_end(d-1) of the same ward (exact)."""
    previous_end = neighbours.previous(out["census_end"].to_numpy(dtype=float))
    mask = out["census_start"].isna().to_numpy() & _within_bounds(out, "census_start", previous_end)
    return _fill(out, "census_start", previous_end, mask, log, "temporal_continuity")


def _census_end_anchor(out: pd.DataFrame, neighbours: _Neighbours) -> np.ndarray:
    """Known end-of-day census: the raw census_end, else the next day's census_start."""
    anchor = out["census_end"].to_numpy(dtype=float)
    next_start = neighbours.following(out["census_start"].to_numpy(dtype=float))
    return np.where(np.isnan(anchor), next_start, anchor)


def _solve_elective_identity(out: pd.DataFrame, log: _RepairLog) -> int:
    """admissions_elective = scheduled - cancelled: recover one missing side (exact)."""
    scheduled = out["elective_scheduled"].to_numpy(dtype=float)
    cancelled = out["elective_cancelled"].to_numpy(dtype=float)
    admitted = out["admissions_elective"].to_numpy(dtype=float)
    solved_scheduled = admitted + cancelled
    mask = np.isnan(scheduled) & _within_bounds(out, "elective_scheduled", solved_scheduled)
    filled = _fill(out, "elective_scheduled", solved_scheduled, mask, log, "elective_identity")
    solved_cancelled = scheduled - admitted
    mask = (np.isnan(cancelled) & _within_bounds(out, "elective_cancelled", solved_cancelled)
            & (solved_cancelled <= scheduled))
    filled += _fill(out, "elective_cancelled", solved_cancelled, mask, log, "elective_identity")
    scheduled = out["elective_scheduled"].to_numpy(dtype=float)
    cancelled = out["elective_cancelled"].to_numpy(dtype=float)
    out["admissions_elective"] = np.where(np.isnan(admitted), scheduled - cancelled, admitted)
    return filled


def _flow_terms(out: pd.DataFrame) -> dict[str, np.ndarray]:
    """Terms of the flow identity (elective admissions from scheduled - cancelled when known)."""
    terms = {name: out[name].to_numpy(dtype=float) for name in _WD_FLOW_SIGNS}
    scheduled = out["elective_scheduled"].to_numpy(dtype=float)
    cancelled = out["elective_cancelled"].to_numpy(dtype=float)
    both = ~np.isnan(scheduled) & ~np.isnan(cancelled)
    terms["admissions_elective"] = np.where(both, scheduled - cancelled, terms["admissions_elective"])
    return terms


def _solve_flow_identity(out: pd.DataFrame, neighbours: _Neighbours, log: _RepairLog) -> int:
    """Solve census_end = start + admissions + transfers_in - transfers_out - discharges
    for the single missing term of a row."""
    anchor = _census_end_anchor(out, neighbours)
    terms = _flow_terms(out)
    n_missing = np.sum([np.isnan(v) for v in terms.values()], axis=0)
    eligible = (n_missing == 1) & ~np.isnan(anchor)
    progress = 0
    for name, sign in _WD_FLOW_SIGNS.items():
        mask = eligible & np.isnan(terms[name])
        if not mask.any():
            continue
        known = sum(s * np.nan_to_num(terms[other]) for other, s in _WD_FLOW_SIGNS.items() if other != name)
        solved = (anchor - known) * sign
        ok = mask & _within_bounds(out, name, solved)
        if name == "admissions_elective":   # evidence only; the next round solves its parts
            out[name] = np.where(ok, solved, out[name].to_numpy(dtype=float))
            progress += _count(ok)
        else:
            progress += _fill(out, name, solved, ok, log, "flow_identity")
    return progress


def _count_flow_identity_mismatches(out: pd.DataFrame) -> int:
    """Rows whose fully known flows disagree with the raw census_end (undetected corruption)."""
    terms = _flow_terms(out)
    predicted = sum(sign * terms[name] for name, sign in _WD_FLOW_SIGNS.items())
    raw_end = out["census_end"].to_numpy(dtype=float)
    return _count(~np.isnan(predicted) & ~np.isnan(raw_end) & (predicted != raw_end))


def _census_end_estimate(out: pd.DataFrame, neighbours: _Neighbours) -> np.ndarray:
    """Best known end-of-day census: anchor, else the flow identity when all terms are known."""
    anchor = _census_end_anchor(out, neighbours)
    terms = _flow_terms(out)
    predicted = sum(sign * terms[name] for name, sign in _WD_FLOW_SIGNS.items())
    return np.where(np.isnan(anchor), predicted, anchor)


def _repair_capacity_state(out: pd.DataFrame, neighbours: _Neighbours, raw_occupancy: pd.Series,
                           log: _RepairLog) -> None:
    """Fill escalation_beds_open / beds_closed from adjacent days, subject to constraints."""
    census_end = _census_end_estimate(out, neighbours)
    labels = raw_occupancy.to_numpy(dtype=object)
    for column in _WD_CAPACITY_STATE:
        values = out[column].to_numpy(dtype=float, copy=True)
        missing = np.flatnonzero(np.isnan(values))
        if not missing.size:
            continue
        previous, following = neighbours.previous(values), neighbours.following(values)
        medians = _round_values(
            pd.Series(values).groupby([out["ward_id"], out["day_of_week"]]).transform("median").to_numpy(dtype=float), 0
        )
        methods: dict[str, np.ndarray] = {}
        for pos in missing:
            preferred = [(previous[pos], "temporal_neighbour"), (following[pos], "temporal_neighbour"),
                         (medians[pos], "ward_dow_median")]
            value, method = _choose_capacity_value(out, pos, column, preferred, census_end[pos], labels[pos])
            if method is not None:
                values[pos] = value
                methods.setdefault(method, np.zeros(len(values), dtype=bool))[pos] = True
        out[column] = values
        for method, mask in methods.items():
            log.record(column, method, mask, approximate=True)


def _choose_capacity_value(out: pd.DataFrame, pos: int, column: str, preferred: list[tuple[float, str]],
                           census_end: float, label: Any) -> tuple[float | None, str | None]:
    """First candidate that keeps the ward-day feasible and reproduces its occupancy band."""
    info = WARDS.get(out.at[pos, "ward_id"])
    if info is None:
        return None, None
    upper = info.max_escalation_beds if column == "escalation_beds_open" else info.max_physical_beds - 1
    candidates = [(float(v), m) for v, m in preferred if not np.isnan(v) and 0 <= v <= upper]
    centre = candidates[0][0] if candidates else 0.0
    candidates += [(float(v), "constraint_search")
                   for v in sorted(range(upper + 1), key=lambda v: (abs(v - centre), v))]
    fallback: tuple[float | None, str | None] = (None, None)
    seen: set[float] = set()
    for value, method in candidates:
        if value in seen:
            continue
        seen.add(value)
        feasible, band_ok = _capacity_feasible(out, pos, column, value, info, census_end, label)
        if feasible and band_ok:
            return value, method
        if feasible and fallback[1] is None:
            fallback = (value, method)
    return fallback


def _capacity_feasible(out: pd.DataFrame, pos: int, column: str, value: float, info: WardInfo,
                       census_end: float, label: Any) -> tuple[bool, bool]:
    """(hard constraints hold, occupancy band reproduced) for one candidate value."""
    escalation = out.at[pos, "escalation_beds_open"]
    closed = out.at[pos, "beds_closed"]
    escalation = value if column == "escalation_beds_open" else (0.0 if pd.isna(escalation) else escalation)
    closed = value if column == "beds_closed" else (0.0 if pd.isna(closed) else closed)
    available = info.bed_capacity + escalation - closed
    census_start = out.at[pos, "census_start"]
    if available < 1 or escalation > info.max_escalation_beds:
        return False, False
    if not pd.isna(census_start) and census_start > info.bed_capacity + escalation:
        return False, False
    if not np.isnan(census_end) and (census_end > available or census_end < 0):
        return False, False
    if not isinstance(label, str) or np.isnan(census_end):
        return True, True
    return True, occupancy_level_for(census_end, available) == label


def _interpolate_trailing_metrics(out: pd.DataFrame, log: _RepairLog) -> None:
    """Time-interpolate the 7-day trailing means within each ward's series."""
    dates = pd.to_datetime(out["date"], format=DATE_FORMAT).to_numpy()
    groups = out.groupby("ward_id", sort=False).indices
    for column in _WD_TRAILING_METRICS:
        values = out[column].to_numpy(dtype=float)
        if not np.isnan(values).any():
            continue
        filled = values.copy()
        for positions in groups.values():
            series = pd.Series(values[positions], index=pd.DatetimeIndex(dates[positions]))
            if series.notna().any() and series.isna().any():
                filled[positions] = series.interpolate(method="time", limit_direction="both").to_numpy()
        decimals = column_spec(WARD_DAILY, column).decimals
        mask = np.isnan(values) & ~np.isnan(filled)
        _fill(out, column, _round_values(filled, decimals), mask, log, "within_ward_interpolation", approximate=True)


def _apply_wait_definition(out: pd.DataFrame, log: _RepairLog) -> None:
    """The mean bed wait of zero emergency admissions is defined as 0."""
    emergencies = out["admissions_emergency"].to_numpy(dtype=float)
    wait = out["avg_bed_wait_hours"].to_numpy(dtype=float)
    mask = (emergencies == 0) & (np.isnan(wait) | (wait != 0))
    _fill(out, "avg_bed_wait_hours", 0.0, mask, log, "definitional_zero")


def _relabel(out: pd.DataFrame, column: str, expected: pd.Series, raw: pd.Series, log: _RepairLog) -> dict[str, int]:
    """Replace a band label by the value recomputed from counts; report raw disagreements."""
    expected_known = expected.notna().to_numpy()
    raw_known = raw.notna().to_numpy()
    same = (raw.astype(object).to_numpy() == expected.astype(object).to_numpy())
    filled = ~raw_known & expected_known
    corrected = raw_known & expected_known & ~same
    log.record(column, "recomputed_from_counts", filled)
    log.record(column, "corrected_from_counts", corrected)
    out[column] = expected.astype(object).where(expected.notna(), np.nan).to_numpy(dtype=object)
    return {
        "raw_labels": _count(raw_known),
        "raw_missing": _count(~raw_known),
        "agreed": _count(raw_known & expected_known & same),
        "disagreed": _count(corrected),
        "undetermined": _count(~expected_known),
    }


def _recompute_staffing_level(out: pd.DataFrame, raw_staffing: pd.Series, log: _RepairLog) -> dict[str, int]:
    """staffing_level from day-shift nurses vs the ward requirement."""
    expected = staffing_levels(out["nurses_day_shift"], out["ward_id"])
    return _relabel(out, "staffing_level", expected, raw_staffing, log)


def _recompute_occupancy_level(out: pd.DataFrame, raw_occupancy: pd.Series, log: _RepairLog) -> dict[str, int]:
    """occupancy_level from the (repaired) census_end / available_beds."""
    derived = derive_ward_daily(out)
    expected = occupancy_levels(derived["census_end"], derived["available_beds"])
    return _relabel(out, "occupancy_level", pd.Series(expected.to_numpy(dtype=object)), raw_occupancy, log)


def _impute_remaining_ward_daily(out: pd.DataFrame, log: _RepairLog) -> None:
    """Night nurses by staffing band, then every remaining gap by ward x day-of-week median."""
    for column in numerical_columns(WARD_DAILY, ("train",)):
        if not out[column].isna().any():
            continue
        spec = column_spec(WARD_DAILY, column)
        low, high = physical_bounds(out, WARD_DAILY, column)
        levels = _MEDIAN_LEVELS_NURSES if column == "nurses_night_shift" else _MEDIAN_LEVELS_WARD_DOW
        _impute_group_median(out, column, levels, log, decimals=0 if spec.dtype == "int" else spec.decimals,
                             lower=low, upper=high)


def _clamp_approximate_counts(out: pd.DataFrame, log: _RepairLog) -> None:
    """Nudge approximate (median) values so ordered count pairs stay consistent."""
    for smaller, larger in _WD_ORDERED_PAIRS:
        small = out[smaller].to_numpy(dtype=float, copy=True)
        large = out[larger].to_numpy(dtype=float, copy=True)
        violated = small > large
        fix_small = violated & log.approximate(smaller)
        fix_large = violated & ~fix_small & log.approximate(larger)
        small[fix_small] = large[fix_small]
        large[fix_large] = small[fix_large]
        out[smaller], out[larger] = small, large
        log.adjust(smaller, fix_small)
        log.adjust(larger, fix_large)
    start = out["census_start"].to_numpy(dtype=float, copy=True)
    physical = out["bed_capacity"].to_numpy(dtype=float) + out["escalation_beds_open"].to_numpy(dtype=float)
    fix = (start > physical) & log.approximate("census_start")
    start[fix] = physical[fix]
    out["census_start"] = start
    log.adjust("census_start", fix)


# ---------------------------------------------------------------------------
# Step 6b - patient_stays repairs
# ---------------------------------------------------------------------------
def repair_patient_stays(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Step 6 for patient_stays: fill every missing value, most principled method first.

    * ``separation_type`` from the destination (internal_ward <=> transfer_out), else ward mode.
    * ``discharge_destination``: internal_ward for transfers out, else the ward's most common
      discharge destination that is compatible with the recorded delay reason.
    * ``discharge_delay_hours``: 0 when the reason is 'none', else a random delayed stay of the
      same destination x reason (hot-deck; at least the 4 h threshold, capped at the stay).
    * ``delay_reason``: 'none' when the delay is 0, else the destination's most common
      non-none reason.
    * ``booking_lead_days`` / ``times_rescheduled``: 0 for non-electives, else the ward's
      elective median (lead >= 1 + 7 x reschedules, rule PS-H14).
    * ``bed_wait_hours``: ward x admission-source hot-deck; ``bed_turnaround_minutes``: ward x
      departure period (night 20:00-07:00 vs day) hot-deck - turnaround depends on when the bed
      is vacated, not on how the patient arrived.
    """
    out = df.reset_index(drop=True).copy()
    for column in raw_numeric_columns(PATIENT_STAYS):
        out[column] = pd.to_numeric(out[column], errors="coerce").astype(float)
    log = _RepairLog(len(out))
    los_hours = stay_los_hours(out)
    discharge_hour = add_stay_time_features(out[["admission_datetime", "discharge_datetime"]])["discharge_hour"]
    night = discharge_hour.isin(list(NIGHT_HOURS)).to_numpy()
    out["_departure_period"] = np.where(discharge_hour.notna().to_numpy(), np.where(night, "night", "day"), None)

    _repair_separation_type(out, log)
    _repair_destination(out, log)
    _repair_delay_hours(out, los_hours, log)
    _repair_delay_reason(out, log)
    _repair_booking_fields(out, log)
    _impute_group_hot_deck(out, "bed_wait_hours", _HOT_DECK_LEVELS_WAIT, log, decimals=2, lower=0.0,
                           upper=column_spec(PATIENT_STAYS, "bed_wait_hours").max_value)
    turnaround = column_spec(PATIENT_STAYS, "bed_turnaround_minutes")
    _impute_group_hot_deck(out, "bed_turnaround_minutes", _HOT_DECK_LEVELS_TURNAROUND, log, decimals=0,
                           lower=turnaround.min_value, upper=turnaround.max_value)

    out = out.drop(columns="_departure_period")
    check = [*raw_numeric_columns(PATIENT_STAYS), *raw_category_columns(PATIENT_STAYS)]
    unrepaired = {c: int(out[c].isna().sum()) for c in check if out[c].isna().any()}
    return out, {**log.to_dict(), "unrepaired": unrepaired}


def _repair_separation_type(out: pd.DataFrame, log: _RepairLog) -> None:
    """Missing separation_type: implied by the destination (internal_ward), else ward mode."""
    missing = out["separation_type"].isna().to_numpy()
    if not missing.any():
        return
    destination = out["discharge_destination"].to_numpy(dtype=object)
    inferred = np.where(destination == "internal_ward", "transfer_out", "discharge").astype(object)
    from_destination = missing & ~pd.isna(destination)
    _fill_labels(out, "separation_type", inferred, from_destination, log, "destination_rule")
    remaining = missing & ~from_destination
    if remaining.any():
        modes = _group_mode(out, "separation_type", ("ward_id",))
        _fill_labels(out, "separation_type", modes, remaining & ~pd.isna(modes), log, "ward_mode", approximate=True)


def _destination_rankings(out: pd.DataFrame) -> tuple[dict[str, list[str]], list[str]]:
    """Discharge destinations ranked by frequency per ward (and overall), transfers excluded."""
    home_like = [d for d in DISCHARGE_DESTINATIONS if d != "internal_ward"]
    observed = out[(out["separation_type"] == "discharge") & out["discharge_destination"].isin(home_like)]
    overall = _rank_values(observed["discharge_destination"], DISCHARGE_DESTINATIONS)
    overall += [d for d in home_like if d not in overall]
    per_ward = {
        str(ward): _rank_values(group["discharge_destination"], DISCHARGE_DESTINATIONS)
        for ward, group in observed.groupby("ward_id")
    }
    return per_ward, overall


def _pick_destination(ranking: list[str], reason: Any) -> str:
    """Most common destination for which the recorded delay reason is meaningful."""
    if isinstance(reason, str) and reason != "none":
        for destination in ranking:
            if reason in ALLOWED_DELAY_REASONS[destination]:
                return destination
    return ranking[0]


def _repair_destination(out: pd.DataFrame, log: _RepairLog) -> None:
    """Missing destination: internal_ward for transfers out, else a reason-compatible ward mode."""
    missing = out["discharge_destination"].isna().to_numpy()
    if not missing.any():
        return
    separation = out["separation_type"].to_numpy(dtype=object)
    _fill_labels(out, "discharge_destination", "internal_ward", missing & (separation == "transfer_out"),
                 log, "separation_rule")
    discharges = missing & (separation == "discharge")
    if not discharges.any():
        return
    per_ward, overall = _destination_rankings(out)
    wards = out["ward_id"].to_numpy(dtype=object)
    reasons = out["delay_reason"].to_numpy(dtype=object)
    chosen = np.full(len(out), None, dtype=object)
    for pos in np.flatnonzero(discharges):
        ranking = per_ward.get(wards[pos], [])
        chosen[pos] = _pick_destination(ranking + [d for d in overall if d not in ranking], reasons[pos])
    _fill_labels(out, "discharge_destination", chosen, discharges, log, "ward_mode", approximate=True)


def _repair_delay_hours(out: pd.DataFrame, los_hours: np.ndarray, log: _RepairLog) -> None:
    """Missing delay: 0 when no reason, else a destination x reason donor (>= 4 h, <= the stay)."""
    delay = out["discharge_delay_hours"].to_numpy(dtype=float)
    missing = np.isnan(delay)
    if not missing.any():
        return
    reason = out["delay_reason"]
    no_delay = missing & (reason == "none").to_numpy()
    _fill(out, "discharge_delay_hours", 0.0, no_delay, log, "definitional_zero")
    _fill(out, "discharge_delay_hours", 0.0, missing & reason.isna().to_numpy(), log, "assumed_no_delay",
          approximate=True)
    delayed = missing & reason.notna().to_numpy() & (reason != "none").to_numpy()
    _impute_group_hot_deck(out, "discharge_delay_hours", _HOT_DECK_LEVELS_DELAY, log, target=delayed,
                           pool=delay > 0, decimals=1, lower=DELAY_THRESHOLD_HOURS, upper=los_hours)


def _reason_choices(out: pd.DataFrame) -> dict[str, tuple[str, str]]:
    """Per destination: (most common observed allowed non-none reason, method name)."""
    observed = out[out["delay_reason"].notna() & (out["delay_reason"] != "none")]
    choices: dict[str, tuple[str, str]] = {}
    for destination, allowed in ALLOWED_DELAY_REASONS.items():
        reasons = observed.loc[observed["discharge_destination"] == destination, "delay_reason"]
        reasons = reasons[reasons.isin(allowed)]
        if len(reasons):
            choices[destination] = (_rank_values(reasons, allowed)[0], "destination_mode")
        else:
            choices[destination] = (next(r for r in allowed if r != "none"), "allowed_default")
    return choices


def _repair_delay_reason(out: pd.DataFrame, log: _RepairLog) -> None:
    """Missing reason: 'none' for undelayed stays, else the destination's most common reason."""
    missing = out["delay_reason"].isna().to_numpy()
    if not missing.any():
        return
    delay = out["discharge_delay_hours"].to_numpy(dtype=float)
    _fill_labels(out, "delay_reason", "none", missing & (delay == 0), log, "definitional_none")
    delayed = missing & (delay > 0)
    if not delayed.any():
        return
    choices = _reason_choices(out)
    destinations = out["discharge_destination"].to_numpy(dtype=object)
    chosen = np.full(len(out), None, dtype=object)
    by_method: dict[str, np.ndarray] = {}
    for pos in np.flatnonzero(delayed):
        if destinations[pos] in choices:
            chosen[pos], method = choices[destinations[pos]]
            by_method.setdefault(method, np.zeros(len(out), dtype=bool))[pos] = True
    for method, mask in by_method.items():
        _fill_labels(out, "delay_reason", chosen, mask, log, method, approximate=True)


def _repair_booking_fields(out: pd.DataFrame, log: _RepairLog) -> None:
    """Lead time / reschedules: 0 for non-electives, else the ward's elective median."""
    source = out["admission_source"]
    elective = (source == "elective").to_numpy()
    non_elective = source.isin(["emergency", "transfer"]).to_numpy()
    rescheduled = out["times_rescheduled"].to_numpy(dtype=float)
    min_lead = np.where(np.isnan(rescheduled), 1.0, min_rebooked_lead_days(np.nan_to_num(rescheduled)))
    for column, lower in (("booking_lead_days", min_lead), ("times_rescheduled", 0.0)):
        missing = out[column].isna().to_numpy()
        _fill(out, column, 0.0, missing & non_elective, log, "non_elective_zero")
        _impute_group_median(out, column, _MEDIAN_LEVELS_ELECTIVE, log, target=missing & elective, pool=elective,
                             decimals=0, lower=lower, upper=column_spec(PATIENT_STAYS, column).max_value)


# ---------------------------------------------------------------------------
# Steps 7-8 - derivation and validation
# ---------------------------------------------------------------------------
def derive_table(df: pd.DataFrame, table: str) -> pd.DataFrame:
    """Step 7: calendar/time features + every derived column, canonical columns and dtypes."""
    if table == WARD_DAILY:
        frame = derive_ward_daily(add_calendar_features(df))
    else:
        frame = derive_patient_stays(add_stay_time_features(df))
    frame = frame[[c for c in column_names(table) if c in frame.columns]]
    return coerce_dtypes(frame, table)


def drop_hard_invalid(df: pd.DataFrame, table: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Step 8: validate the repaired table and drop rows that still break a hard rule.

    A dropped row is counted under every hard rule it violates (``rows_dropped_by_rule``);
    ``dropped_examples`` lists up to five source keys per rule.
    """
    frame = df.reset_index(drop=True)
    before = validate_table(frame, table)
    keep = before.hard_valid_mask.to_numpy()
    clean = frame[keep].reset_index(drop=True)
    after = validate_table(clean, table)
    keys = source_keys(frame, table) if set(SOURCE_KEYS[table]) <= set(frame.columns) else None
    examples = {
        r["rule_id"]: ([keys.iloc[i] for i in r["example_indices"]] if keys is not None else r["example_indices"])
        for r in before.rule_results if r["severity"] == "hard" and r["violations"]
    }
    report = {
        "after_repair": validation_summary(before),
        "final": validation_summary(after),
        "rows_dropped": _count(~keep),
        "rows_dropped_by_rule": {rule: n for rule, n in before.violation_counts("hard").items() if n},
        "dropped_examples": examples,
    }
    return clean, report


# ---------------------------------------------------------------------------
# Steps 1-8 per table
# ---------------------------------------------------------------------------
def clean_ward_daily(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Steps 1-8 for ward_daily: raw extract -> clean, fully derived, hard-valid ward-days.

    Rows keep their source key (ward_id + date) and are sorted by date, ward_id; record ids
    are assigned in step 10 (:func:`assign_ids_chronologically`).
    """
    return _clean_table(raw_df, WARD_DAILY, repair_ward_daily)


def clean_patient_stays(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Steps 1-8 for patient_stays: raw extract -> clean, fully derived, hard-valid stays.

    Rows keep their source ``stay_id`` and are sorted by admission time; stay ids are
    reassigned chronologically in step 10.
    """
    return _clean_table(raw_df, PATIENT_STAYS, repair_patient_stays)


def clean_table(raw_df: pd.DataFrame, table: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Dispatch to :func:`clean_ward_daily` or :func:`clean_patient_stays`."""
    if table == WARD_DAILY:
        return clean_ward_daily(raw_df)
    if table == PATIENT_STAYS:
        return clean_patient_stays(raw_df)
    raise KeyError(f"Unknown table '{table}'. Known tables: {list(TABLES)}")


def _clean_table(raw_df: pd.DataFrame, table: str,
                 repair: Callable[[pd.DataFrame], tuple[pd.DataFrame, dict[str, Any]]]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Steps 1-8 for one table; ``repair`` is the table-specific step 6."""
    started = time.perf_counter()
    _require_raw_columns(raw_df, table)
    profile = profile_raw(raw_df, table)
    frame = raw_df[list(RAW_COLUMNS[table])].reset_index(drop=True)
    frame, standardisation = standardize_categories(frame, table)
    frame, timestamps = parse_timestamps(frame, table)
    frame, duplicates = remove_duplicates(frame, table)
    frame, invalid = flag_invalid_values(frame, table)
    frame, repairs = repair(frame)
    frame = derive_table(frame, table)
    frame, validation = drop_hard_invalid(frame, table)
    frame = sort_chronologically(frame, table)

    rows_dropped = {
        "unparseable_timestamp": timestamps["rows_dropped"],
        "exact_duplicate": duplicates["exact_duplicates"],
        "key_duplicate": duplicates["key_duplicates"],
        "missing_key": duplicates["missing_key"],
        "hard_rule_violation": validation["rows_dropped"],
    }
    rows_dropped["total"] = sum(rows_dropped.values())
    report = {
        "table": table,
        "raw_rows": int(len(raw_df)),
        "clean_rows": int(len(frame)),
        "retention_rate": _rate(len(frame), len(raw_df)),
        "issues_detected": _detected_issues(profile, standardisation, timestamps, duplicates, invalid, repairs),
        "rows_dropped": rows_dropped,
        "repairs": repairs,
        "validation": {"before": profile["provisional_validation"], **validation},
        "profile": profile,
        "standardisation": standardisation,
        "timestamps": timestamps,
        "duplicates": duplicates,
        "invalid_values": invalid,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    logger.info("Cleaned %s: %d raw rows -> %d clean rows (%d dropped)",
                table, len(raw_df), len(frame), rows_dropped["total"])
    return frame, report


def _detected_issues(profile: dict, standardisation: dict, timestamps: dict, duplicates: dict,
                     invalid: dict, repairs: dict) -> dict[str, dict[str, int]]:
    """Issue counts per issue type and column, as detected by the pipeline."""
    missing: dict[str, dict[str, int]] = {}
    for column, n in profile["missing_values"].items():
        _add_issue(missing, "missing_value", column, n)
    labels: dict[str, dict[str, int]] = {}
    for column, stats in repairs.get("label_recomputation", {}).items():
        _add_issue(labels, "inconsistent_label", column, stats.get("disagreed", 0))
    return _merge_issues(missing, standardisation["issues"], timestamps["issues"], duplicates["issues"],
                         invalid["issues"], labels)


# ---------------------------------------------------------------------------
# Step 9 - dataset checks
# ---------------------------------------------------------------------------
def ward_daily_continuity(ward_daily: pd.DataFrame) -> dict[str, Any]:
    """Step 9a: share of consecutive ward-days with ``census_start(d) == census_end(d-1)``."""
    frame = ward_daily.sort_values(["ward_id", "date"], kind="mergesort").reset_index(drop=True)
    neighbours = _Neighbours.from_frame(frame)
    previous_end = neighbours.previous(frame["census_end"].to_numpy(dtype=float))
    continuous = neighbours.has_prev & (frame["census_start"].to_numpy(dtype=float) == previous_end)
    by_ward = {}
    for ward, positions in frame.groupby("ward_id", sort=True).indices.items():
        pairs = _count(neighbours.has_prev[positions])
        by_ward[str(ward)] = {
            "pairs": pairs,
            "continuous": _count(continuous[positions]),
            "continuity_rate": _rate(_count(continuous[positions]), pairs),
        }
    dates = pd.to_datetime(frame["date"], format=DATE_FORMAT, errors="coerce")
    n_days = int((dates.max() - dates.min()).days) + 1 if len(frame) else 0
    expected = n_days * frame["ward_id"].nunique()
    pairs = _count(neighbours.has_prev)
    return {
        "consecutive_pairs": pairs,
        "continuous_pairs": _count(continuous),
        "continuity_rate": _rate(_count(continuous), pairs),
        "breaks": pairs - _count(continuous),
        "date_range": [dates.min(), dates.max()] if len(frame) else [None, None],
        "expected_ward_days": expected,
        "missing_ward_days": max(expected - len(frame), 0),
        "by_ward": by_ward,
    }


_ADMISSION_COUNT_COLUMNS: dict[str, str] = {
    "elective": "admissions_elective",
    "emergency": "admissions_emergency",
    "transfer": "transfers_in",
}


def cross_table_admissions(ward_daily: pd.DataFrame, patient_stays: pd.DataFrame) -> dict[str, Any]:
    """Step 9b: stays admitted per ward / day / source vs the ward_daily admission counts.

    Only stays admitted inside the ward_daily date window are compared. Rows dropped during
    cleaning on either side show up as mismatches, so the match rate also measures how much
    the cleaning preserved the link between the two tables.
    """
    days = ward_daily[["ward_id", "date", *_ADMISSION_COUNT_COLUMNS.values()]].copy()
    stays = patient_stays[["ward_id", "admission_source"]].copy()
    stays["date"] = patient_stays["admission_datetime"].astype(str).str[:10]
    if days.empty:
        return {"ward_days": 0, "stays_in_window": 0, "by_source": {}, "all_sources_match_rate": None}
    in_window = ((stays["date"] >= days["date"].min()) & (stays["date"] <= days["date"].max())).to_numpy()
    window = stays[in_window]
    counts = (window.groupby(["ward_id", "date", "admission_source"]).size()
              .unstack(fill_value=0).reindex(columns=list(ADMISSION_SOURCES), fill_value=0).reset_index())
    merged = days.merge(counts, on=["ward_id", "date"], how="left")
    merged[list(ADMISSION_SOURCES)] = merged[list(ADMISSION_SOURCES)].fillna(0)
    by_source: dict[str, Any] = {}
    all_match = np.ones(len(merged), dtype=bool)
    for source, column in _ADMISSION_COUNT_COLUMNS.items():
        expected = merged[column].to_numpy(dtype=float)
        observed = merged[source].to_numpy(dtype=float)
        equal = expected == observed
        all_match &= equal
        by_source[source] = {
            "ward_daily_column": column,
            "ward_daily_total": int(np.nansum(expected)),
            "stays_total": int(observed.sum()),
            "match_rate": _rate(_count(equal), len(merged)),
            "mean_abs_diff": round(float(np.nanmean(np.abs(expected - observed))), 4),
        }
    day_keys = set(zip(days["ward_id"], days["date"]))
    unmatched = sum(1 for key in zip(window["ward_id"], window["date"]) if key not in day_keys)
    return {
        "ward_days": int(len(merged)),
        "stays_in_window": int(len(window)),
        "stays_outside_window": int((~in_window).sum()),
        "stays_without_ward_day": int(unmatched),
        "by_source": by_source,
        "all_sources_match_rate": _rate(_count(all_match), len(merged)),
    }


# ---------------------------------------------------------------------------
# Step 10 - identifiers and train/holdout split
# ---------------------------------------------------------------------------
def sort_chronologically(df: pd.DataFrame, table: str) -> pd.DataFrame:
    """Sort by date, ward (ward_daily) or admission/discharge time (stays); stable."""
    order = [c for c in CHRONOLOGICAL_ORDER[table] if c in df.columns]
    return df.sort_values(order, kind="mergesort", na_position="last").reset_index(drop=True)


def assign_ids_chronologically(df: pd.DataFrame, table: str) -> pd.DataFrame:
    """Step 10a: sort chronologically and assign sequential ids (WD-000001 / ST-0000001)."""
    return assign_ids(sort_chronologically(df, table), table)


def split_train_holdout(
    df: pd.DataFrame,
    table: str,
    fraction: float | None = None,
    seed: int | None = None,
    stratify_by: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Step 10b: seeded random train/holdout split, stratified by ``stratify_by``.

    Defaults come from ``settings.preprocessing`` (holdout_fraction, stratify_by) and the
    project seed. Each stratum sends ``round(fraction * n)`` rows to the holdout (keeping at
    least one row in train); both splits preserve the input order.
    """
    if table not in TABLES:
        raise KeyError(f"Unknown table '{table}'. Known tables: {list(TABLES)}")
    settings = section("preprocessing")
    fraction = float(settings.get("holdout_fraction", 0.2) if fraction is None else fraction)
    if not 0.0 <= fraction < 1.0:
        raise ValueError(f"holdout fraction must be in [0, 1), got {fraction}")
    stratify_by = stratify_by if stratify_by is not None else settings.get("stratify_by")
    rng = np.random.default_rng(get_seed() if seed is None else int(seed))
    if stratify_by and stratify_by in df.columns:
        groups = df.groupby(stratify_by, sort=False, dropna=False).indices
        strata = [np.sort(groups[key]) for key in sorted(groups, key=str)]
    else:
        if stratify_by:
            logger.warning("Column '%s' not in %s; splitting without stratification.", stratify_by, table)
        strata = [np.arange(len(df))]
    holdout = np.zeros(len(df), dtype=bool)
    for positions in strata:
        n_rows = len(positions)
        k = min(int(math.floor(fraction * n_rows + 0.5)), max(n_rows - 1, 0))
        if k:
            holdout[rng.choice(positions, size=k, replace=False)] = True
    return df[~holdout].reset_index(drop=True), df[holdout].reset_index(drop=True)


def split_summary(train: pd.DataFrame, holdout: pd.DataFrame, stratify_by: str | None,
                  fraction: float, seed: int) -> dict[str, Any]:
    """Split sizes overall and per stratum (JSON-safe)."""
    by_stratum: dict[str, Any] = {}
    if stratify_by and stratify_by in train.columns:
        train_counts = train[stratify_by].value_counts(dropna=False)
        holdout_counts = holdout[stratify_by].value_counts(dropna=False)
        for value in sorted(set(train_counts.index) | set(holdout_counts.index), key=str):
            n_train, n_holdout = int(train_counts.get(value, 0)), int(holdout_counts.get(value, 0))
            by_stratum[str(value)] = {"train": n_train, "holdout": n_holdout,
                                      "holdout_share": _rate(n_holdout, n_train + n_holdout)}
    return {
        "train_rows": int(len(train)),
        "holdout_rows": int(len(holdout)),
        "holdout_fraction": fraction,
        "realised_holdout_share": _rate(len(holdout), len(train) + len(holdout)),
        "stratify_by": stratify_by,
        "seed": int(seed),
        "by_stratum": by_stratum,
    }


# ---------------------------------------------------------------------------
# Step 12 - cleaning accuracy against the ground truth
# ---------------------------------------------------------------------------
def load_injection_log(raw_dir: Path | str | None = None) -> pd.DataFrame | None:
    """The simulator's dirty-data injection log (``_ground_truth/injection_log.csv``), or None."""
    path = Path(raw_dir if raw_dir is not None else get_path("raw_dir")) / "_ground_truth" / INJECTION_LOG_FILENAME
    if not path.exists():
        return None
    log_df = read_csv_raw(path)
    missing = [c for c in INJECTION_LOG_COLUMNS if c not in log_df.columns]
    if missing:
        logger.warning("Injection log %s lacks columns %s; ignoring it.", path, missing)
        return None
    return log_df


def injected_issue_counts(injection_log: pd.DataFrame | None, table: str) -> dict[str, dict[str, int]] | None:
    """Logged injected issues of ``table`` per issue type and column (None without a log)."""
    if injection_log is None:
        return None
    entries = injection_log[injection_log["table"].astype(str).str.strip() == table]
    issues: dict[str, dict[str, int]] = {}
    for (issue_type, column), n in entries.groupby(["issue_type", "column"], dropna=False).size().items():
        _add_issue(issues, str(issue_type), ROW_LEVEL if pd.isna(column) else str(column), int(n))
    return issues


def _comparable(frame: pd.DataFrame, table: str) -> pd.DataFrame:
    """Raw columns normalised for comparison, indexed by source key (first row per key)."""
    out = pd.DataFrame(index=frame.index)
    for column in RAW_COLUMNS[table]:
        if column not in frame.columns:
            continue
        spec = column_spec(table, column)
        if spec.role == "time":
            parsed, _ = parse_timestamp_series(frame[column], spec.dtype)
            out[column] = format_timestamps(parsed, spec.dtype)
        elif spec.sdtype == "numerical":
            values = pd.to_numeric(frame[column], errors="coerce").astype(float).to_numpy()
            out[column] = _round_values(values, 0 if spec.dtype == "int" else spec.decimals)
        elif spec.sdtype in ("categorical", "ordinal"):
            mapping = {v: canonical_category(column, v) for v in frame[column].dropna().unique()}
            out[column] = frame[column].map(mapping).astype(object)
        else:
            out[column] = _strip_text(frame[column].astype(object))
    out.index = source_keys(out, table)
    return out[~out.index.duplicated(keep="first")]


def _equal_cells(a: pd.Series, b: pd.Series, numeric: bool) -> np.ndarray:
    """Element-wise equality (numbers within 1e-6; two missing values count as equal)."""
    if numeric:
        x, y = a.to_numpy(dtype=float), b.to_numpy(dtype=float)
        return np.isclose(x, y, rtol=0.0, atol=1e-6) | (np.isnan(x) & np.isnan(y))
    x, y = a.to_numpy(dtype=object), b.to_numpy(dtype=object)
    return (x == y) | (pd.isna(x) & pd.isna(y))


def cleaning_accuracy(
    clean_df: pd.DataFrame,
    truth_df: pd.DataFrame,
    table: str,
    injection_log: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Step 12: agreement of a cleaned table with the simulator's clean ground truth.

    Rows are aligned on the source key (``ward_id + date`` / ``stay_id``). Reports row
    recall/precision, per-raw-column agreement and overall cell agreement; with the
    injection log, also the outcome of every logged issue: ``exact`` (original value
    recovered / duplicate removed), ``repaired`` (defect gone, plausible imputed value),
    ``row_dropped``, ``unresolved`` or ``not_evaluable``.
    """
    clean = _comparable(clean_df, table)
    truth = _comparable(truth_df, table)
    common = truth.index.intersection(clean.index)
    keys = set(SOURCE_KEYS[table])
    columns = [c for c in RAW_COLUMNS[table] if c not in keys and c in clean.columns and c in truth.columns]
    per_column: dict[str, float | None] = {}
    equal_cells = 0
    clean_aligned, truth_aligned = clean.loc[common], truth.loc[common]
    for column in columns:
        numeric = column_spec(table, column).sdtype == "numerical"
        equal = _equal_cells(clean_aligned[column], truth_aligned[column], numeric)
        per_column[column] = _rate(_count(equal), len(common))
        equal_cells += _count(equal)
    result: dict[str, Any] = {
        "truth_rows": int(len(truth)),
        "clean_rows": int(len(clean)),
        "aligned_rows": int(len(common)),
        "row_recall": _rate(len(common), len(truth)),
        "row_precision": _rate(len(common), len(clean)),
        "cell_agreement": _rate(equal_cells, len(common) * len(columns)),
        "column_agreement": per_column,
        "least_accurate_columns": sorted(
            ((c, r) for c, r in per_column.items() if r is not None and r < 1.0), key=lambda cr: cr[1]
        )[:5],
        "injected_issues": None,
    }
    if injection_log is not None:
        clean_key_counts = source_keys(clean_df, table).value_counts()
        result["injected_issues"] = _injected_issue_outcomes(clean, truth, clean_key_counts, injection_log, table)
    return result


def _normalise_row_key(row_key: Any, table: str) -> str | None:
    """Log row key in canonical source-key form (ward alias / day-first date tolerated)."""
    if pd.isna(row_key):
        return None
    text = str(row_key).strip()
    if table == PATIENT_STAYS:
        return text or None
    ward_part, _, date_part = text.partition("|")
    ward_id = canonical_category("ward_id", ward_part) or ward_part.strip()
    date_part = date_part.strip()
    if _ISO_DATE.fullmatch(date_part):   # fast path: the documented "<ward_id>|<YYYY-MM-DD>" form
        return f"{ward_id}|{date_part}"
    parsed, _ = parse_timestamp_series(pd.Series([date_part]), "date")
    return f"{ward_id}|{parsed.iloc[0]:%Y-%m-%d}" if parsed.notna().iloc[0] else None


def _logged_columns(value: Any, table: str) -> list[str]:
    """Raw columns named in a log ``column`` cell ('a', 'a|b', 'a,b', ...); [] = row-level."""
    if pd.isna(value):
        return []
    tokens = re.split(r"[\s,;|+/&]+", str(value).strip())
    return [t for t in tokens if t in RAW_COLUMNS[table]]


def _comparable_value(value: Any, column: str, table: str) -> Any:
    """One logged value normalised like :func:`_comparable` (None when missing/unparseable)."""
    spec = column_spec(table, column)
    if pd.isna(value):
        return None
    if spec.sdtype == "numerical":
        number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        return None if pd.isna(number) else float(_round_values(np.array([number], dtype=float),
                                                                 0 if spec.dtype == "int" else spec.decimals)[0])
    if spec.role == "time":
        parsed, _ = parse_timestamp_series(pd.Series([value]), spec.dtype)
        return format_timestamps(parsed, spec.dtype).iloc[0] if parsed.notna().iloc[0] else None
    return str(value).strip()


def _same(a: Any, b: Any) -> bool:
    """Equality of two normalised scalars; missing values never match."""
    if a is None or b is None or (isinstance(a, float) and math.isnan(a)) or (isinstance(b, float) and math.isnan(b)):
        return False
    if isinstance(a, (int, float, np.number)) and isinstance(b, (int, float, np.number)):
        return abs(float(a) - float(b)) <= 1e-6
    return a == b


def _cell_status(clean_value: Any, true_value: Any, injected: Any, column: str, table: str) -> str:
    """'exact', 'repaired' or 'unresolved' (missing, or still the injected value)."""
    if pd.isna(clean_value):
        return "unresolved"
    numeric = column_spec(table, column).sdtype == "numerical"
    clean_cmp = float(clean_value) if numeric else clean_value
    true_cmp = None if pd.isna(true_value) else (float(true_value) if numeric else true_value)
    if _same(clean_cmp, true_cmp):
        return "exact"
    if _same(clean_cmp, _comparable_value(injected, column, table)):
        return "unresolved"
    return "repaired"


def _injected_issue_outcomes(clean: pd.DataFrame, truth: pd.DataFrame, clean_key_counts: pd.Series,
                             injection_log: pd.DataFrame, table: str) -> dict[str, Any]:
    """Classify every logged injected issue of ``table`` by what the cleaning did to it."""
    entries = injection_log[injection_log["table"].astype(str).str.strip() == table]
    per_type: dict[str, dict[str, int]] = {}
    for entry in entries.itertuples(index=False):
        issue_type = str(entry.issue_type)
        key = _normalise_row_key(entry.row_key, table)
        columns = [] if "dup" in issue_type.lower() else _logged_columns(entry.column, table)
        if key is None:
            status = "not_evaluable"
        elif not columns:   # row-level issue (duplicates): exactly one copy must survive
            copies = int(clean_key_counts.get(key, 0))
            status = "exact" if copies == 1 else ("row_dropped" if copies == 0 else "unresolved")
        elif key not in truth.index:
            status = "not_evaluable"
        elif key not in clean.index:
            status = "row_dropped"
        else:
            statuses = [
                _cell_status(clean.at[key, c], truth.at[key, c], entry.injected_value, c, table) for c in columns
            ]
            status = max(statuses, key=("exact", "repaired", "unresolved").index)
        bucket = per_type.setdefault(issue_type, dict.fromkeys(_STATUS_ORDER, 0))
        bucket[status] += 1
    totals = dict.fromkeys(_STATUS_ORDER, 0)
    for bucket in per_type.values():
        for status, n in bucket.items():
            totals[status] += n
    return {
        "logged": int(len(entries)),
        **_outcome_rates(totals),
        "status_counts": totals,
        "by_issue_type": {t: {**counts, **_outcome_rates(counts)} for t, counts in sorted(per_type.items())},
    }


def _outcome_rates(counts: dict[str, int]) -> dict[str, float | None]:
    """Outcome shares over the evaluable issues (``not_evaluable`` excluded)."""
    evaluable = sum(counts.values()) - counts["not_evaluable"]
    return {
        "evaluable": evaluable,
        "resolved_rate": _rate(counts["exact"] + counts["repaired"], evaluable),
        "exact_recovery_rate": _rate(counts["exact"], evaluable),
        "row_dropped_rate": _rate(counts["row_dropped"], evaluable),
        "unresolved_rate": _rate(counts["unresolved"], evaluable),
    }


# ---------------------------------------------------------------------------
# Step 11 - orchestration and outputs
# ---------------------------------------------------------------------------
def _write_json(path: Path, payload: Any) -> Path:
    """Write ``payload`` as strict JSON (NaN rejected), UTF-8, indented."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), indent=2, allow_nan=False), encoding="utf-8")
    return path


def table_metadata_filename(table: str) -> str:
    """File name of the single-table SDV metadata of ``table`` (``metadata_<table>.json``)."""
    return TABLE_METADATA_FILENAME.format(table=table)


def _save_sdv_metadata(metadata: Any, path: Path) -> Path:
    if path.exists():
        path.unlink()                    # SDV refuses to overwrite an existing file
    metadata.save_to_json(str(path))
    return path


def write_metadata(processed_dir: Path) -> dict[str, str]:
    """Write the SDV metadata files of the prepared datasets (DESIGN.md §6.3 step 11).

    * ``metadata.json`` - both tables exactly as the ``{table}_{clean,train,holdout}.csv`` files
      store them: key (primary key), time columns (datetime with their format), training and
      derived columns. ``Metadata.validate_data`` accepts each CSV against it.
    * ``metadata_<table>.json`` - single-table metadata of the *training* columns only, which is
      what every BedSynth synthesizer fits (``GaussianCopulaSynthesizer(metadata)`` etc.); select
      ``schema.training_columns(table)`` from a CSV before fitting. Keys, timestamps and derived
      columns are never learned (derived columns are recomputed by ``schema.derive``).

    Returns ``{output name: path}``.
    """
    from bedsynth.synthesis.metadata import add_table_metadata, build_metadata  # imports SDV (slow)
    from sdv.metadata import Metadata

    project = Metadata()
    for table in TABLES:
        add_table_metadata(project, table, columns="full")
        for spec in column_specs(table, ("time",)):
            date_format = DATE_FORMAT if spec.dtype == "date" else DATETIME_FORMAT
            project.add_column(spec.name, table_name=table, sdtype="datetime", datetime_format=date_format)
    project.validate()
    outputs = {"metadata": str(_save_sdv_metadata(project, processed_dir / METADATA_FILENAME))}
    for table in TABLES:
        path = processed_dir / table_metadata_filename(table)
        outputs[f"metadata_{table}"] = str(_save_sdv_metadata(build_metadata(table, columns="training"), path))
    return outputs


def _constraint_catalog(table: str) -> list[dict[str, Any]] | None:
    """SDV constraints the synthesizers use (``None`` when the synthesis stack is unavailable)."""
    try:
        from bedsynth.synthesis.constraints import describe_constraints
    except Exception as exc:  # noqa: BLE001 - the catalogue is documentation; never fail Week 1 on it
        logger.warning("Constraint catalogue unavailable: %s", exc)
        return None
    try:
        return to_jsonable(describe_constraints(table))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not describe the constraints of %s: %s", table, exc)
        return None


def write_schema_catalog(path: Path) -> Path:
    """Schema, SDV metadata files, constraints and operational rules of every table (JSON).

    Per table: the schema catalogue (columns with roles, types, units, bounds), which metadata
    file to fit with and which columns it covers, the SDV constraints applied during training
    and the hard rules enforced at generation time by rejection sampling.
    """
    catalog = {}
    for table in TABLES:
        rules = rules_catalog(table)
        catalog[table] = {
            **schema_as_dict(table),
            "sdv_metadata": {
                "csv_files": METADATA_FILENAME,
                "training": table_metadata_filename(table),
                "fit_columns": column_names(table, ["train"]),
                "never_learned": column_names(table, ["key", "time", "derived"]),
            },
            "constraints": _constraint_catalog(table),
            "rejection_enforced_rules": [rule["rule_id"] for rule in rules if rule["severity"] == "hard"],
            "rules": rules,
        }
    return _write_json(path, {"tables": catalog})


def _accuracy_if_available(clean: pd.DataFrame, table: str, raw_dir: Path,
                           injection_log: pd.DataFrame | None) -> dict[str, Any] | None:
    """Step 12 when the simulator's ground truth is present, else None."""
    truth_file = ground_truth_path(table, raw_dir)
    if not truth_file.exists():
        logger.info("No ground truth for %s at %s; skipping cleaning accuracy.", table, truth_file)
        return None
    return cleaning_accuracy(clean, read_csv_raw(truth_file), table, injection_log)


def _headline(report: dict[str, Any]) -> dict[str, Any]:
    """Week-1 headline numbers of one table report."""
    validation = report["validation"]
    accuracy = report.get("cleaning_accuracy") or {}
    issues = accuracy.get("injected_issues") or {}
    return {
        "raw_rows": report["raw_rows"],
        "final_rows": report.get("final_rows", report["clean_rows"]),
        "retention_rate": report["retention_rate"],
        "issues_detected": sum(sum(cols.values()) for cols in report["issues_detected"].values()),
        "cells_repaired": report["repairs"]["cells_repaired"],
        "rows_dropped": report["rows_dropped"]["total"],
        "hard_valid_rate_raw": validation["before"].get("hard_valid_rate"),
        "hard_valid_rate_after_repair": validation["after_repair"]["hard_valid_rate"],
        "hard_valid_rate_final": validation["final"]["hard_valid_rate"],
        "cell_agreement_with_truth": accuracy.get("cell_agreement"),
        "injected_issues_resolved_rate": issues.get("resolved_rate"),
        "injected_issues_exact_recovery_rate": issues.get("exact_recovery_rate"),
        "train_rows": report.get("split", {}).get("train_rows"),
        "holdout_rows": report.get("split", {}).get("holdout_rows"),
    }


def prepare_datasets(
    raw_dir: Path | str | None = None,
    processed_dir: Path | str | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """Run the Week-1 pipeline for both tables and return the data-quality report.

    Reads ``{table}_raw.csv`` from ``raw_dir``, cleans each table (steps 1-8), runs the
    dataset checks (9), assigns chronological ids and splits train/holdout (10), writes
    ``{table}_{clean,train,holdout}.csv``, ``metadata.json`` (the CSVs), ``metadata_<table>.json``
    (the training columns, see :func:`write_metadata`), ``schema_catalog.json`` and
    ``data_quality_report.json`` to ``processed_dir`` (11) and scores the cleaning against the
    ground truth when the simulator left one (12). ``seed`` (default: project seed) drives
    the split.

    The locations recorded in the report (``raw_dir``, ``processed_dir``, ``tables.<table>.raw_file``
    and ``outputs``) are written with :func:`bedsynth.config.project_relative` - relative to the
    project root, ``~/...`` outside it - so the report never contains the home directory;
    :func:`bedsynth.config.resolve_project_path` turns them back into paths.
    """
    started = time.perf_counter()
    raw_dir = Path(raw_dir) if raw_dir is not None else get_path("raw_dir")
    processed_dir = Path(processed_dir) if processed_dir is not None else get_path("processed_dir")
    processed_dir.mkdir(parents=True, exist_ok=True)
    seed = get_seed() if seed is None else int(seed)
    settings = section("preprocessing")
    fraction = float(settings.get("holdout_fraction", 0.2))
    stratify_by = settings.get("stratify_by", "ward_id")
    injection_log = load_injection_log(raw_dir)

    cleaned: dict[str, pd.DataFrame] = {}
    tables: dict[str, dict[str, Any]] = {}
    for table in TABLES:
        source = raw_path(table, raw_dir)
        if not source.exists():
            raise FileNotFoundError(
                f"Raw extract not found at {source}. Run `python scripts/01_prepare_data.py` "
                "(it simulates the source data when raw files are missing)."
            )
        logger.info("Cleaning %s (%s)", table, source)
        frame, report = clean_table(read_csv_raw(source), table)
        report["raw_file"] = project_relative(source)
        report["issues_injected"] = injected_issue_counts(injection_log, table)
        report["cleaning_accuracy"] = _accuracy_if_available(frame, table, raw_dir, injection_log)
        cleaned[table], tables[table] = frame, report

    consistency = {
        "ward_daily_continuity": ward_daily_continuity(cleaned[WARD_DAILY]),
        "cross_table_admissions": cross_table_admissions(cleaned[WARD_DAILY], cleaned[PATIENT_STAYS]),
    }

    outputs: dict[str, str] = {}
    for table in TABLES:
        final = assign_ids_chronologically(cleaned[table], table)
        train, holdout = split_train_holdout(final, table, fraction, seed, stratify_by)
        tables[table]["final_rows"] = int(len(final))
        tables[table]["split"] = split_summary(train, holdout, stratify_by, fraction, seed)
        for split, frame in (("clean", final), ("train", train), ("holdout", holdout)):
            outputs[f"{table}_{split}"] = project_relative(
                write_table(frame, processed_path(table, split, processed_dir)))
    outputs.update({name: project_relative(path) for name, path in write_metadata(processed_dir).items()})
    outputs["schema_catalog"] = project_relative(write_schema_catalog(processed_dir / SCHEMA_CATALOG_FILENAME))
    report_path = processed_dir / REPORT_FILENAME
    outputs["data_quality_report"] = project_relative(report_path)

    report = to_jsonable({
        "pipeline": "bedsynth.data.preprocessing.prepare_datasets",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": seed,
        "settings": {"holdout_fraction": fraction, "stratify_by": stratify_by},
        "raw_dir": project_relative(raw_dir),
        "processed_dir": project_relative(processed_dir),
        "ground_truth_available": {t: tables[t]["cleaning_accuracy"] is not None for t in TABLES},
        "injection_log_available": injection_log is not None,
        "summary": {t: _headline(tables[t]) for t in TABLES},
        "consistency_checks": consistency,
        "tables": tables,
        "outputs": outputs,
        "elapsed_seconds": round(time.perf_counter() - started, 2),
    })
    _write_json(report_path, report)
    logger.info("Week-1 preparation finished in %.1f s; report at %s", report["elapsed_seconds"], report_path)
    return report


__all__ = [
    "assign_ids_chronologically",
    "canonical_category",
    "clean_patient_stays",
    "clean_table",
    "clean_ward_daily",
    "cleaning_accuracy",
    "cross_table_admissions",
    "derive_table",
    "drop_hard_invalid",
    "flag_invalid_values",
    "injected_issue_counts",
    "load_injection_log",
    "parse_timestamp_series",
    "parse_timestamps",
    "physical_bounds",
    "prepare_datasets",
    "profile_raw",
    "remove_duplicates",
    "repair_patient_stays",
    "repair_timestamp_swaps",
    "repair_ward_daily",
    "sort_chronologically",
    "source_keys",
    "split_summary",
    "split_train_holdout",
    "standardize_categories",
    "stay_los_hours",
    "table_metadata_filename",
    "to_jsonable",
    "validation_summary",
    "ward_daily_continuity",
    "write_metadata",
    "write_schema_catalog",
]
