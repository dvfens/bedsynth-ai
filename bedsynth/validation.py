"""Operational rule validation for source and synthetic hospital operations data.

Rules are evaluated on a *fully derived* table (``schema.derive(table, df)`` already applied).

* hard rules - a violating row is operationally impossible/inconsistent; the generator
  rejects it (rejection sampling) and the cleaning step repairs or drops it.
* soft rules - plausibility warnings; reported, never rejected.

The catalogue (rule ids, names, columns) is specified in DESIGN.md §6.4 and served to API
clients through :func:`rules_catalog`.

Missing-value policy
--------------------
Rule checks never raise on NaN, unknown categories, wrong dtypes or absent columns. A row
passes a rule only when every value the rule reads is present, parseable and (for
categorical columns) in its vocabulary, *and* the condition holds; anything else is a
violation. Implications ``A => B`` therefore also fail when ``A`` cannot be evaluated.

Performance
-----------
:func:`validate_table` runs inside the generator's rejection-sampling loop (DESIGN.md §8.3),
so every check is vectorised over numpy arrays - no row loops and no index alignment. The
full catalogue validates 50k rows in well under a second.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import pandas as pd

from bedsynth.schema import (
    ADMISSION_SOURCES,
    ALLOWED_DELAY_REASONS,
    DAYS_OF_WEEK,
    DELAY_REASONS,
    DELAY_THRESHOLD_HOURS,
    DISCHARGE_DESTINATIONS,
    OCCUPANCY_LEVELS,
    OCCUPANCY_THRESHOLDS,
    PATIENT_STAYS,
    SEPARATION_TYPES,
    STAFFING_LEVELS,
    TABLES,
    WARD_DAILY,
    WARD_IDS,
    WARDS,
    WEEKEND_DAYS,
    categorical_columns,
    column_spec,
    column_specs,
    integer_columns,
    min_rebooked_lead_days,
    occupancy_levels,
    staffing_levels,
)

logger = logging.getLogger(__name__)

SEVERITIES: tuple[str, ...] = ("hard", "soft")
MAX_EXAMPLES = 5          # example_indices reported per rule
_EPS = 1e-9               # tolerance for comparisons involving rounded float columns


# ---------------------------------------------------------------------------
# Public data structures
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Rule:
    """One operational rule: metadata plus a vectorised check over a derived table."""

    rule_id: str                                   # e.g. "WD-H03" (H = hard, S = soft)
    table: str                                     # "ward_daily" | "patient_stays"
    severity: str                                  # "hard" | "soft"
    name: str                                      # short snake_case name
    description: str                               # one sentence, operational wording
    columns: tuple[str, ...]                       # columns the rule reads
    check: Callable[[pd.DataFrame], pd.Series]     # boolean Series aligned to df.index, True = passes

    def to_dict(self) -> dict[str, Any]:
        """Catalogue entry (everything except the check callable), JSON-serialisable."""
        return {
            "rule_id": self.rule_id,
            "table": self.table,
            "severity": self.severity,
            "name": self.name,
            "description": self.description,
            "columns": list(self.columns),
        }


@dataclass
class ValidationReport:
    """Outcome of :func:`validate_table`: per-rule results and the hard-valid row mask."""

    table: str
    n_rows: int
    hard_valid_mask: pd.Series                      # True = row passes every hard rule
    rule_results: list[dict[str, Any]] = field(default_factory=list)
    # each rule_results item: {rule_id, name, severity, description, columns,
    #                          violations (int), violation_rate (float), example_indices (list[int], <= 5)}
    # example_indices are index labels for integer-indexed frames, otherwise row positions.

    @property
    def n_hard_valid(self) -> int:
        """Rows that pass every hard rule."""
        return int(self.hard_valid_mask.sum())

    @property
    def all_hard_passed(self) -> bool:
        """True when no row violates a hard rule (also for an empty table)."""
        return self.n_hard_valid == self.n_rows

    @property
    def hard_violation_rate(self) -> float:
        """Share of rows violating at least one hard rule (0.0 for an empty table)."""
        return 0.0 if self.n_rows == 0 else 1.0 - self.n_hard_valid / self.n_rows

    def violation_counts(self, severity: str | None = "hard") -> dict[str, int]:
        """``{rule_id: violations}`` for one severity (``None`` = every evaluated rule)."""
        return {
            r["rule_id"]: int(r["violations"])
            for r in self.rule_results
            if severity is None or r["severity"] == severity
        }

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable summary (plain Python types only, no pandas objects)."""
        n_rows = int(self.n_rows)
        return {
            "table": self.table,
            "n_rows": n_rows,
            "n_hard_valid": self.n_hard_valid,
            "hard_valid_rate": round(1.0 - float(self.hard_violation_rate), 6),
            "hard_violation_rate": round(float(self.hard_violation_rate), 6),
            "all_hard_passed": bool(self.all_hard_passed),
            "rules_checked": {
                severity: sum(1 for r in self.rule_results if r["severity"] == severity)
                for severity in SEVERITIES
            },
            "hard_violations": self.violation_counts("hard"),
            "soft_violations": self.violation_counts("soft"),
            "rule_results": [_plain_result(r) for r in self.rule_results],
        }


# ---------------------------------------------------------------------------
# Vectorised array helpers (never raise; missing/invalid -> NaN / False)
# ---------------------------------------------------------------------------
def _column(df: pd.DataFrame, column: str) -> pd.Series | None:
    """The column as a Series (first one if labels are duplicated), or None if absent."""
    if column not in df.columns:
        return None
    data = df[column]
    if isinstance(data, pd.DataFrame):
        data = data.iloc[:, 0]
    return data


def _to_float(value: Any) -> float:
    """``float(value)``, or NaN when the value is not numeric."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _numeric(df: pd.DataFrame, column: str) -> np.ndarray:
    """Column as float64; absent columns and unparseable values become NaN."""
    series = _column(df, column)
    if series is None:
        return np.full(len(df), np.nan)
    try:
        return np.asarray(pd.to_numeric(series, errors="coerce").astype("float64"), dtype=np.float64)
    except (TypeError, ValueError):
        return np.fromiter((_to_float(v) for v in series), dtype=np.float64, count=len(series))


def _codes(df: pd.DataFrame, column: str, vocabulary: Sequence[Any]) -> np.ndarray:
    """Position of each value in ``vocabulary`` (-1 for missing/unknown/absent)."""
    series = _column(df, column)
    if series is None:
        return np.full(len(df), -1, dtype=np.int64)
    try:
        return np.asarray(pd.Categorical(series, categories=list(vocabulary)).codes, dtype=np.int64)
    except (TypeError, ValueError):  # unhashable or otherwise odd cell values
        index = {value: i for i, value in enumerate(vocabulary)}

        def lookup(value: Any) -> int:
            try:
                return index.get(value, -1)
            except TypeError:
                return -1

        return np.fromiter((lookup(v) for v in series), dtype=np.int64, count=len(series))


def _isin(df: pd.DataFrame, column: str, vocabulary: Iterable[Any]) -> np.ndarray:
    """True where the column's value is one of ``vocabulary`` (missing/absent -> False)."""
    return _codes(df, column, tuple(vocabulary)) >= 0


def _bool_cells(df: pd.DataFrame, column: str) -> np.ndarray:
    """True where a cell holds a boolean (``True == 1`` must never pass as a category code)."""
    series = _column(df, column)
    if series is None:
        return np.zeros(len(df), dtype=bool)
    if pd.api.types.is_bool_dtype(series.dtype):
        return np.ones(len(series), dtype=bool)
    if series.dtype != object:
        return np.zeros(len(series), dtype=bool)
    return np.fromiter((isinstance(v, (bool, np.bool_)) for v in series), dtype=bool, count=len(series))


def _whole(values: np.ndarray) -> np.ndarray:
    """Finite whole numbers."""
    return np.isfinite(values) & (np.floor(values) == values)


def _within(values: np.ndarray, low: float | None = None, high: float | None = None) -> np.ndarray:
    """True where ``low <= value <= high`` (open bounds when None; NaN fails)."""
    ok = ~np.isnan(values)
    if low is not None:
        ok &= values >= low
    if high is not None:
        ok &= values <= high
    return ok


def _known(*arrays: np.ndarray) -> np.ndarray:
    """True where none of the numeric arrays is NaN."""
    ok = np.ones(len(arrays[0]), dtype=bool)
    for values in arrays:
        ok &= ~np.isnan(values)
    return ok


def _ward_attribute(df: pd.DataFrame, attribute: str) -> np.ndarray:
    """Per-row ward catalogue attribute as float (NaN for unknown wards)."""
    lookup = np.array([float(getattr(WARDS[w], attribute)) for w in WARD_IDS] + [np.nan])
    return lookup[_codes(df, "ward_id", WARD_IDS)]  # code -1 indexes the trailing NaN


def _canonical_wards(df: pd.DataFrame) -> pd.Series:
    """ward_id as canonical codes or None (safe input for schema helpers)."""
    lookup = np.array(list(WARD_IDS) + [None], dtype=object)
    return pd.Series(lookup[_codes(df, "ward_id", WARD_IDS)], dtype=object)


def _pair_allowed(
    df: pd.DataFrame,
    first: tuple[str, Sequence[str]],
    second: tuple[str, Sequence[str]],
    allowed: Callable[[str, str], bool],
) -> np.ndarray:
    """True where the (first, second) category pair is allowed (unknown values -> False)."""
    (col_a, vocab_a), (col_b, vocab_b) = first, second
    matrix = np.zeros((len(vocab_a) + 1, len(vocab_b) + 1), dtype=bool)  # last row/col = unknown
    for i, a in enumerate(vocab_a):
        for j, b in enumerate(vocab_b):
            matrix[i, j] = allowed(a, b)
    return matrix[_codes(df, col_a, vocab_a), _codes(df, col_b, vocab_b)]


def _label_matches(df: pd.DataFrame, column: str, expected: np.ndarray, vocabulary: Sequence[str]) -> np.ndarray:
    """True where ``df[column]`` equals the expected label (undefined expectations fail)."""
    observed = _codes(df, column, vocabulary)
    ok = np.zeros(len(df), dtype=bool)
    for code, label in enumerate(vocabulary):
        ok |= (observed == code) & (expected == label)
    return ok


# ---------------------------------------------------------------------------
# Shared checks
# ---------------------------------------------------------------------------
def _vocabulary_check(df: pd.DataFrame, table: str) -> np.ndarray:
    """Every categorical train column holds a value from its vocabulary."""
    ok = np.ones(len(df), dtype=bool)
    for spec in column_specs(table, ("train",)):
        if spec.sdtype not in ("categorical", "ordinal") or not spec.categories:
            continue
        if all(isinstance(c, (int, np.integer)) for c in spec.categories):   # month columns
            ok &= np.isin(_numeric(df, spec.name), np.asarray(spec.categories, dtype=float))
            ok &= ~_bool_cells(df, spec.name)
        else:
            ok &= _isin(df, spec.name, spec.categories)
    return ok


def _range_check(df: pd.DataFrame, bounds: dict[str, tuple[float | None, float | None]]) -> np.ndarray:
    """Every column of ``bounds`` lies within its ``(low, high)`` range."""
    ok = np.ones(len(df), dtype=bool)
    for column, (low, high) in bounds.items():
        ok &= _within(_numeric(df, column), low, high)
    return ok


def _spec_bounds(table: str, columns: Iterable[str]) -> dict[str, tuple[float | None, float | None]]:
    """Physical [min, max] bounds of columns as declared in the schema."""
    return {c: (column_spec(table, c).min_value, column_spec(table, c).max_value) for c in columns}


# ---------------------------------------------------------------------------
# ward_daily checks (DESIGN.md §6.4)
# ---------------------------------------------------------------------------
_WD_COUNT_COLUMNS: tuple[str, ...] = tuple(integer_columns(WARD_DAILY, ("train", "derived")))
_WD_METRIC_BOUNDS = _spec_bounds(WARD_DAILY, ("avg_bed_wait_hours", "avg_bed_turnaround_min", "avg_los_days"))


def _wd_counts(df: pd.DataFrame) -> np.ndarray:
    ok = np.ones(len(df), dtype=bool)
    for column in _WD_COUNT_COLUMNS:
        values = _numeric(df, column)
        ok &= _whole(values) & (values >= 0)
    return ok


def _wd_cancellations(df: pd.DataFrame) -> np.ndarray:
    return _numeric(df, "elective_cancelled") <= _numeric(df, "elective_scheduled")


def _wd_noon_discharges(df: pd.DataFrame) -> np.ndarray:
    return _numeric(df, "discharges_before_noon") <= _numeric(df, "discharges")


def _wd_delayed(df: pd.DataFrame) -> np.ndarray:
    return _numeric(df, "delayed_discharges") <= _numeric(df, "census_start")


def _wd_census_end(df: pd.DataFrame) -> np.ndarray:
    census_end = _numeric(df, "census_end")
    return (census_end >= 0) & (census_end <= _numeric(df, "available_beds"))


def _wd_census_start(df: pd.DataFrame) -> np.ndarray:
    physical = _numeric(df, "bed_capacity") + _numeric(df, "escalation_beds_open")
    return _numeric(df, "census_start") <= physical


def _wd_closures_free_beds(df: pd.DataFrame) -> np.ndarray:
    """Only free beds are closed, so the patients present at 00:00 fit in the available beds."""
    return _numeric(df, "census_start") <= _numeric(df, "available_beds")


def _wd_capacity(df: pd.DataFrame) -> np.ndarray:
    return (
        (_numeric(df, "escalation_beds_open") <= _ward_attribute(df, "max_escalation_beds"))
        & (_numeric(df, "beds_closed") >= 0)
        & (_numeric(df, "available_beds") >= 1)
    )


def _wd_staffing_present(df: pd.DataFrame) -> np.ndarray:
    return (_numeric(df, "nurses_day_shift") >= 1) & (_numeric(df, "nurses_night_shift") >= 1)


def _wd_occupancy_level(df: pd.DataFrame) -> np.ndarray:
    expected = occupancy_levels(
        pd.Series(_numeric(df, "census_end")), pd.Series(_numeric(df, "available_beds"))
    ).to_numpy(dtype=object)
    return _label_matches(df, "occupancy_level", expected, OCCUPANCY_LEVELS)


def _wd_staffing_level(df: pd.DataFrame) -> np.ndarray:
    expected = staffing_levels(pd.Series(_numeric(df, "nurses_day_shift")), _canonical_wards(df))
    return _label_matches(df, "staffing_level", expected.to_numpy(dtype=object), STAFFING_LEVELS)


def _wd_admission_profile(df: pd.DataFrame) -> np.ndarray:
    takes_electives = _ward_attribute(df, "accepts_electives")
    takes_transfers = _ward_attribute(df, "accepts_transfers")
    scheduled = _numeric(df, "elective_scheduled")
    cancelled = _numeric(df, "elective_cancelled")
    transfers_in = _numeric(df, "transfers_in")
    known = _known(takes_electives, takes_transfers, scheduled, cancelled, transfers_in)
    no_electives = (scheduled == 0) & (cancelled == 0)
    return known & ((takes_electives == 1) | no_electives) & ((takes_transfers == 1) | (transfers_in == 0))


def _wd_metric_ranges(df: pd.DataFrame) -> np.ndarray:
    return _range_check(df, _WD_METRIC_BOUNDS)


def _wd_wait_definition(df: pd.DataFrame) -> np.ndarray:
    emergencies = _numeric(df, "admissions_emergency")
    wait = _numeric(df, "avg_bed_wait_hours")
    return _known(emergencies, wait) & ((emergencies != 0) | (wait == 0))


def _wd_night_within_day(df: pd.DataFrame) -> np.ndarray:
    return _numeric(df, "nurses_night_shift") <= _numeric(df, "nurses_day_shift")


def _wd_nurse_workload(df: pd.DataFrame) -> np.ndarray:
    return _numeric(df, "patients_per_nurse") <= 2.0 * _ward_attribute(df, "nurse_ratio_day") + _EPS


def _wd_boarding_pressure(df: pd.DataFrame) -> np.ndarray:
    boarders = _numeric(df, "emergency_boarders")
    occupancy = _numeric(df, "occupancy_rate")
    full = occupancy >= OCCUPANCY_THRESHOLDS["high"] - _EPS
    return _known(boarders, occupancy) & ((boarders <= 0) | full)


def _wd_escalation_pressure(df: pd.DataFrame) -> np.ndarray:
    escalation = _numeric(df, "escalation_beds_open")
    occupancy = _numeric(df, "occupancy_rate")
    return _known(escalation, occupancy) & ((escalation <= 0) | (occupancy >= 0.80 - _EPS))


def _wd_weekend_electives(df: pd.DataFrame) -> np.ndarray:
    scheduled = _numeric(df, "elective_scheduled")
    weekend = _isin(df, "day_of_week", WEEKEND_DAYS)
    return _isin(df, "day_of_week", DAYS_OF_WEEK) & ~np.isnan(scheduled) & (~weekend | (scheduled <= 2))


def _wd_noon_share(df: pd.DataFrame) -> np.ndarray:
    discharges = _numeric(df, "discharges")
    noon = _numeric(df, "discharges_before_noon")
    return _known(discharges, noon) & ((discharges < 4) | (noon <= 0.8 * discharges + _EPS))


# ---------------------------------------------------------------------------
# patient_stays checks (DESIGN.md §6.4)
# ---------------------------------------------------------------------------
_PS_COUNT_COLUMNS: tuple[str, ...] = ("booking_lead_days", "times_rescheduled", "los_nights", "bed_turnaround_minutes")
_PS_HOUR_VALUED_COLUMNS: tuple[str, ...] = ("bed_wait_hours", "discharge_delay_hours")
_PS_METRIC_BOUNDS: dict[str, tuple[float | None, float | None]] = {
    "bed_turnaround_minutes": _spec_bounds(PATIENT_STAYS, ["bed_turnaround_minutes"])["bed_turnaround_minutes"],
    "bed_wait_hours": (None, column_spec(PATIENT_STAYS, "bed_wait_hours").max_value),
    "los_nights": (None, column_spec(PATIENT_STAYS, "los_nights").max_value),
    "booking_lead_days": (None, column_spec(PATIENT_STAYS, "booking_lead_days").max_value),
    "times_rescheduled": (None, column_spec(PATIENT_STAYS, "times_rescheduled").max_value),
}
_COMPLEX_DESTINATIONS: tuple[str, ...] = ("care_facility", "home_with_support")


def _ps_hours(df: pd.DataFrame) -> np.ndarray:
    ok = np.ones(len(df), dtype=bool)
    for column in ("admission_hour", "discharge_hour"):
        hours = _numeric(df, column)
        ok &= _whole(hours) & (hours >= 0) & (hours <= 23)
    return ok


def _ps_non_negative(df: pd.DataFrame) -> np.ndarray:
    ok = np.ones(len(df), dtype=bool)
    for column in _PS_COUNT_COLUMNS:
        values = _numeric(df, column)
        ok &= _whole(values) & (values >= 0)
    for column in _PS_HOUR_VALUED_COLUMNS:
        values = _numeric(df, column)
        ok &= np.isfinite(values) & (values >= 0)
    return ok


def _ps_positive_los(df: pd.DataFrame) -> np.ndarray:
    return _numeric(df, "los_hours") >= 1


def _ps_elective_lead(df: pd.DataFrame) -> np.ndarray:
    lead = _numeric(df, "booking_lead_days")
    elective = _isin(df, "admission_source", ("elective",))
    return _isin(df, "admission_source", ADMISSION_SOURCES) & ~np.isnan(lead) & (~elective | (lead >= 1))


def _ps_non_elective_unbooked(df: pd.DataFrame) -> np.ndarray:
    lead = _numeric(df, "booking_lead_days")
    rescheduled = _numeric(df, "times_rescheduled")
    elective = _isin(df, "admission_source", ("elective",))
    unbooked = (lead == 0) & (rescheduled == 0)
    return _isin(df, "admission_source", ADMISSION_SOURCES) & _known(lead, rescheduled) & (elective | unbooked)


def _ps_delay_reason_consistent(df: pd.DataFrame) -> np.ndarray:
    delay = _numeric(df, "discharge_delay_hours")
    no_reason = _isin(df, "delay_reason", ("none",))
    return _isin(df, "delay_reason", DELAY_REASONS) & ~np.isnan(delay) & (no_reason == (delay == 0))


def _ps_delay_within_stay(df: pd.DataFrame) -> np.ndarray:
    return _numeric(df, "discharge_delay_hours") <= _numeric(df, "los_hours") + _EPS


def _ps_delay_threshold(df: pd.DataFrame) -> np.ndarray:
    """A recorded delay is zero or at least the delay threshold (shorter gaps are not delays)."""
    delay = _numeric(df, "discharge_delay_hours")
    return (delay == 0) | (delay >= DELAY_THRESHOLD_HOURS - _EPS)


def _ps_rebooking_lead(df: pd.DataFrame) -> np.ndarray:
    """A postponed elective was rebooked >= 7 days later each time: lead >= 1 + 7 x reschedules."""
    lead = _numeric(df, "booking_lead_days")
    rescheduled = _numeric(df, "times_rescheduled")
    elective = _isin(df, "admission_source", ("elective",))
    known = _isin(df, "admission_source", ADMISSION_SOURCES) & _known(lead, rescheduled)
    return known & (~elective | (lead >= min_rebooked_lead_days(rescheduled)))


def _ps_separation_destination(df: pd.DataFrame) -> np.ndarray:
    return _pair_allowed(
        df,
        ("separation_type", SEPARATION_TYPES),
        ("discharge_destination", DISCHARGE_DESTINATIONS),
        lambda separation, destination: (separation == "transfer_out") == (destination == "internal_ward"),
    )


def _ps_delay_reason_allowed(df: pd.DataFrame) -> np.ndarray:
    return _pair_allowed(
        df,
        ("discharge_destination", DISCHARGE_DESTINATIONS),
        ("delay_reason", DELAY_REASONS),
        lambda destination, reason: reason in ALLOWED_DELAY_REASONS[destination],
    )


def _ps_ward_admission_profile(df: pd.DataFrame) -> np.ndarray:
    return _pair_allowed(
        df,
        ("ward_id", WARD_IDS),
        ("admission_source", ADMISSION_SOURCES),
        lambda ward_id, source: source in WARDS[ward_id].admission_sources,
    )


def _ps_metric_ranges(df: pd.DataFrame) -> np.ndarray:
    return _range_check(df, _PS_METRIC_BOUNDS)


def _ps_elective_admission_hours(df: pd.DataFrame) -> np.ndarray:
    hour = _numeric(df, "admission_hour")
    elective = _isin(df, "admission_source", ("elective",))
    in_window = (hour >= 6) & (hour <= 14)
    return _isin(df, "admission_source", ADMISSION_SOURCES) & ~np.isnan(hour) & (~elective | in_window)


def _ps_daytime_departures(df: pd.DataFrame) -> np.ndarray:
    return _within(_numeric(df, "discharge_hour"), 7, 22)


def _ps_short_elective_wait(df: pd.DataFrame) -> np.ndarray:
    wait = _numeric(df, "bed_wait_hours")
    elective = _isin(df, "admission_source", ("elective",))
    return _isin(df, "admission_source", ADMISSION_SOURCES) & ~np.isnan(wait) & (~elective | (wait <= 4))


def _ps_no_paediatric_transfers(df: pd.DataFrame) -> np.ndarray:
    paediatric = _isin(df, "ward_id", ("PAED",))
    transfer_out = _isin(df, "separation_type", ("transfer_out",))
    known = _isin(df, "ward_id", WARD_IDS) & _isin(df, "separation_type", SEPARATION_TYPES)
    return known & ~(paediatric & transfer_out)


def _ps_long_delays(df: pd.DataFrame) -> np.ndarray:
    delay = _numeric(df, "discharge_delay_hours")
    complex_destination = _isin(df, "discharge_destination", _COMPLEX_DESTINATIONS)
    known = _isin(df, "discharge_destination", DISCHARGE_DESTINATIONS) & ~np.isnan(delay)
    return known & ((delay <= 72) | complex_destination)


# ---------------------------------------------------------------------------
# Rule catalogue
# ---------------------------------------------------------------------------
def _as_check(func: Callable[[pd.DataFrame], np.ndarray], doc: str) -> Callable[[pd.DataFrame], pd.Series]:
    """Wrap an array-returning check into the ``Rule.check`` contract (Series on df.index)."""

    def check(df: pd.DataFrame) -> pd.Series:
        return pd.Series(np.asarray(func(df), dtype=bool), index=df.index)

    target = getattr(func, "func", func)  # unwrap functools.partial
    check.__name__ = getattr(target, "__name__", "check")
    check.__doc__ = doc
    return check


def _rule(
    rule_id: str,
    name: str,
    description: str,
    columns: Iterable[str],
    func: Callable[[pd.DataFrame], np.ndarray],
) -> Rule:
    """Build a catalogue entry; table and severity follow from the id (``WD-H03`` = ward_daily, hard)."""
    table = WARD_DAILY if rule_id.startswith("WD-") else PATIENT_STAYS
    severity = "hard" if rule_id[3] == "H" else "soft"
    check = _as_check(func, f"{rule_id} {name}: {description} True = the row passes.")
    return Rule(rule_id, table, severity, name, description, tuple(columns), check)


_WD_FLOW = ("census_start", "admissions_elective", "admissions_emergency", "transfers_in", "transfers_out", "discharges")

_WARD_DAILY_RULES: tuple[Rule, ...] = (
    _rule("WD-H01", "valid_categories",
          "ward_id, month, day_of_week, staffing_level and occupancy_level take values from their vocabularies.",
          categorical_columns(WARD_DAILY), partial(_vocabulary_check, table=WARD_DAILY)),
    _rule("WD-H02", "non_negative_integer_counts",
          "Every bed, nurse and patient count (learned and derived) is present, a whole number and not negative.",
          _WD_COUNT_COLUMNS, _wd_counts),
    _rule("WD-H03", "cancellations_within_bookings",
          "A ward cannot cancel more elective admissions than were booked for the day.",
          ("elective_cancelled", "elective_scheduled"), _wd_cancellations),
    _rule("WD-H04", "noon_discharges_within_discharges",
          "Discharges completed before noon cannot exceed the day's discharges.",
          ("discharges_before_noon", "discharges"), _wd_noon_discharges),
    _rule("WD-H05", "delayed_within_census",
          "Patients medically fit but awaiting discharge at midnight cannot exceed the midnight census.",
          ("delayed_discharges", "census_start"), _wd_delayed),
    _rule("WD-H06", "census_within_available_beds",
          "The end-of-day census is non-negative and fits within the beds available that day.",
          ("census_end", "available_beds"), _wd_census_end),
    _rule("WD-H07", "census_start_within_physical_beds",
          "The start-of-day census fits within the base beds plus the escalation beds open.",
          ("census_start", "bed_capacity", "escalation_beds_open"), _wd_census_start),
    _rule("WD-H08", "capacity_limits",
          "Escalation beds respect the ward's surge limit, closed beds are not negative and at least one bed is available.",
          ("ward_id", "escalation_beds_open", "beds_closed", "available_beds"), _wd_capacity),
    _rule("WD-H09", "staffing_present",
          "At least one registered nurse is rostered on both the day and the night shift.",
          ("nurses_day_shift", "nurses_night_shift"), _wd_staffing_present),
    _rule("WD-H10", "occupancy_level_consistent",
          "The occupancy band matches the midnight occupancy census_end / available_beds.",
          ("occupancy_level", "census_end", "available_beds"), _wd_occupancy_level),
    _rule("WD-H11", "staffing_level_consistent",
          "The staffing band matches the day-shift nurses against the ward's nurse requirement.",
          ("staffing_level", "nurses_day_shift", "ward_id"), _wd_staffing_level),
    _rule("WD-H12", "ward_admission_profile",
          "Wards that take no elective admissions book none, and wards that take no transfers receive none.",
          ("ward_id", "elective_scheduled", "elective_cancelled", "transfers_in"), _wd_admission_profile),
    _rule("WD-H13", "metric_ranges",
          "Average bed wait (0-96 h), bed turnaround (10-1440 min) and length of stay (0.04-90 days) are physically plausible.",
          tuple(_WD_METRIC_BOUNDS), _wd_metric_ranges),
    _rule("WD-H14", "wait_definition",
          "With no emergency admissions the day's average bed wait is defined as zero.",
          ("admissions_emergency", "avg_bed_wait_hours"), _wd_wait_definition),
    _rule("WD-H15", "closures_only_free_beds",
          "Only free beds are closed, so the start-of-day census fits within the beds available that day.",
          ("census_start", "available_beds"), _wd_closures_free_beds),
    _rule("WD-S01", "night_staffing_within_day",
          "Night-shift nurses do not outnumber day-shift nurses.",
          ("nurses_night_shift", "nurses_day_shift"), _wd_night_within_day),
    _rule("WD-S02", "nurse_workload_plausible",
          "Patients per day-shift nurse stay within twice the ward's target nurse ratio.",
          ("patients_per_nurse", "ward_id"), _wd_nurse_workload),
    _rule("WD-S03", "boarding_under_pressure",
          "Emergency patients board in ED overnight only when the ward is at least 90% occupied.",
          ("emergency_boarders", "occupancy_rate"), _wd_boarding_pressure),
    _rule("WD-S04", "escalation_under_pressure",
          "Escalation beds are open only when the ward is at least 80% occupied.",
          ("escalation_beds_open", "occupancy_rate"), _wd_escalation_pressure),
    _rule("WD-S05", "few_weekend_electives",
          "At most two elective admissions are booked on a Saturday or Sunday.",
          ("day_of_week", "elective_scheduled"), _wd_weekend_electives),
    _rule("WD-S06", "noon_discharge_share_plausible",
          "On days with four or more discharges, at most 80% are completed before noon.",
          ("discharges", "discharges_before_noon"), _wd_noon_share),
)

_PATIENT_STAYS_RULES: tuple[Rule, ...] = (
    _rule("PS-H01", "valid_categories",
          "All categorical stay attributes take values from their vocabularies.",
          categorical_columns(PATIENT_STAYS), partial(_vocabulary_check, table=PATIENT_STAYS)),
    _rule("PS-H02", "hours_in_range",
          "Admission and departure hours are whole hours between 0 and 23.",
          ("admission_hour", "discharge_hour"), _ps_hours),
    _rule("PS-H03", "non_negative_values",
          "Counts are whole and not negative, bed wait and discharge delay hours are not negative, and none is missing.",
          _PS_COUNT_COLUMNS + _PS_HOUR_VALUED_COLUMNS, _ps_non_negative),
    _rule("PS-H04", "positive_length_of_stay",
          "Every stay lasts at least one hour.",
          ("los_hours",), _ps_positive_los),
    _rule("PS-H05", "elective_booking_lead",
          "Elective admissions are booked at least one day in advance.",
          ("admission_source", "booking_lead_days"), _ps_elective_lead),
    _rule("PS-H06", "non_elective_unbooked",
          "Emergency and transfer admissions have no booking lead time and were never rescheduled.",
          ("admission_source", "booking_lead_days", "times_rescheduled"), _ps_non_elective_unbooked),
    _rule("PS-H07", "delay_reason_consistent",
          "A delay reason is recorded exactly when the departure was delayed ('none' if and only if zero delay hours).",
          ("delay_reason", "discharge_delay_hours"), _ps_delay_reason_consistent),
    _rule("PS-H08", "delay_within_stay",
          "The discharge delay cannot be longer than the stay itself.",
          ("discharge_delay_hours", "los_hours"), _ps_delay_within_stay),
    _rule("PS-H09", "separation_destination_consistent",
          "Stays that end in an internal transfer go to another ward, and only those do.",
          ("separation_type", "discharge_destination"), _ps_separation_destination),
    _rule("PS-H10", "delay_reason_allowed",
          "The delay reason is operationally meaningful for the discharge destination.",
          ("delay_reason", "discharge_destination"), _ps_delay_reason_allowed),
    _rule("PS-H11", "ward_admission_profile",
          "The admission source is one the ward accepts.",
          ("ward_id", "admission_source"), _ps_ward_admission_profile),
    _rule("PS-H12", "metric_ranges",
          "Bed turnaround is 10-1440 min, bed wait at most 96 h, stays at most 180 nights, booking lead at most 365 days and reschedules at most 10.",
          tuple(_PS_METRIC_BOUNDS), _ps_metric_ranges),
    _rule("PS-H13", "delay_threshold",
          f"A departure counts as delayed only when it is more than {DELAY_THRESHOLD_HOURS:g} hours after the "
          f"patient was ready, so recorded delays are either zero or at least {DELAY_THRESHOLD_HOURS:g} hours.",
          ("discharge_delay_hours",), _ps_delay_threshold),
    _rule("PS-H14", "rebooking_lead",
          "Postponed electives are rebooked at least seven days later, so the booking lead is at least "
          "1 + 7 x times_rescheduled days.",
          ("admission_source", "booking_lead_days", "times_rescheduled"), _ps_rebooking_lead),
    _rule("PS-S01", "elective_admission_hours",
          "Elective patients are admitted between 06:00 and 14:59.",
          ("admission_source", "admission_hour"), _ps_elective_admission_hours),
    _rule("PS-S02", "daytime_departures",
          "Patients leave the ward between 07:00 and 22:59.",
          ("discharge_hour",), _ps_daytime_departures),
    _rule("PS-S03", "short_elective_wait",
          "Elective patients wait at most four hours for their bed.",
          ("admission_source", "bed_wait_hours"), _ps_short_elective_wait),
    _rule("PS-S04", "no_paediatric_transfers_out",
          "Paediatric stays do not end in an internal transfer.",
          ("ward_id", "separation_type"), _ps_no_paediatric_transfers),
    _rule("PS-S05", "long_delays_for_complex_discharges",
          "Discharge delays beyond 72 hours occur only for care-facility or supported-home discharges.",
          ("discharge_delay_hours", "discharge_destination"), _ps_long_delays),
)

_RULES: dict[str, tuple[Rule, ...]] = {WARD_DAILY: _WARD_DAILY_RULES, PATIENT_STAYS: _PATIENT_STAYS_RULES}
_RULES_BY_ID: dict[str, Rule] = {rule.rule_id: rule for rules in _RULES.values() for rule in rules}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def get_rules(table: str, severity: str | None = None) -> list[Rule]:
    """Rules of ``table`` in catalogue order, optionally restricted to ``'hard'`` or ``'soft'``."""
    if table not in _RULES:
        raise KeyError(f"Unknown table '{table}'. Known tables: {list(TABLES)}")
    if severity is not None and severity not in SEVERITIES:
        raise ValueError(f"Unknown severity '{severity}'. Expected one of {SEVERITIES} or None.")
    return [rule for rule in _RULES[table] if severity is None or rule.severity == severity]


def get_rule(rule_id: str) -> Rule:
    """Look up one rule by id (e.g. ``'WD-H03'``)."""
    try:
        return _RULES_BY_ID[rule_id]
    except KeyError as exc:
        raise KeyError(f"Unknown rule '{rule_id}'. Known rules: {sorted(_RULES_BY_ID)}") from exc


def rules_catalog(table: str) -> list[dict[str, Any]]:
    """JSON-serialisable rule catalogue of ``table`` (hard rules first, then soft)."""
    return [rule.to_dict() for rule in get_rules(table)]


def validate_table(df: pd.DataFrame, table: str, include_soft: bool = True) -> ValidationReport:
    """Evaluate the rule catalogue of ``table`` on a fully derived frame.

    Returns per-rule violation counts/rates (with up to five example rows each) and the
    mask of rows that pass every hard rule. Never raises on bad cell values.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"validate_table expects a pandas DataFrame, got {type(df).__name__}")
    rules = get_rules(table) if include_soft else get_rules(table, "hard")
    n_rows = len(df)
    absent = sorted({c for rule in rules for c in rule.columns if c not in df.columns})
    if absent and n_rows:
        logger.warning("validate_table(%s): columns %s are absent; rules reading them fail.", table, absent)

    hard_ok = np.ones(n_rows, dtype=bool)
    results: list[dict[str, Any]] = []
    for rule in rules:
        passed = _evaluate(rule, df)
        if rule.severity == "hard":
            hard_ok &= passed
        failed_positions = np.flatnonzero(~passed)
        violations = int(failed_positions.size)
        results.append({
            "rule_id": rule.rule_id,
            "name": rule.name,
            "severity": rule.severity,
            "description": rule.description,
            "columns": list(rule.columns),
            "violations": violations,
            "violation_rate": round(violations / n_rows, 6) if n_rows else 0.0,
            "example_indices": _example_indices(df.index, failed_positions[:MAX_EXAMPLES]),
        })
    mask = pd.Series(hard_ok, index=df.index, name="hard_valid")
    return ValidationReport(table=table, n_rows=n_rows, hard_valid_mask=mask, rule_results=results)


def hard_valid_mask(df: pd.DataFrame, table: str) -> pd.Series:
    """Boolean Series aligned to ``df.index``: True where the row passes every hard rule."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"hard_valid_mask expects a pandas DataFrame, got {type(df).__name__}")
    ok = np.ones(len(df), dtype=bool)
    for rule in get_rules(table, "hard"):
        ok &= _evaluate(rule, df)
    return pd.Series(ok, index=df.index, name="hard_valid")


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------
def _evaluate(rule: Rule, df: pd.DataFrame) -> np.ndarray:
    """Run one rule check defensively: any failure marks every row as violating."""
    n_rows = len(df)
    try:
        result = rule.check(df)
        series = result if isinstance(result, pd.Series) else pd.Series(np.asarray(result))
        if series.dtype != bool:
            series = series.astype(object).where(series.notna(), False).astype(bool)
        values = series.to_numpy(dtype=bool)
        if values.shape != (n_rows,):
            raise ValueError(f"check returned {values.shape[0]} values for {n_rows} rows")
        return values
    except Exception as exc:  # noqa: BLE001 - a broken check must never break validation
        logger.warning("Rule %s could not be evaluated (%s); all %d rows count as violations.",
                       rule.rule_id, exc, n_rows)
        return np.zeros(n_rows, dtype=bool)


def _example_indices(index: pd.Index, positions: np.ndarray) -> list[int]:
    """Index labels of the example rows when the index is integer-typed, else positions."""
    if pd.api.types.is_integer_dtype(index.dtype):
        return [int(index[p]) for p in positions]
    return [int(p) for p in positions]


def _plain_result(result: dict[str, Any]) -> dict[str, Any]:
    """Copy of one rule result with numpy scalars converted to Python types."""
    plain: dict[str, Any] = {}
    for key, value in result.items():
        if isinstance(value, np.integer):
            value = int(value)
        elif isinstance(value, np.floating):
            value = float(value)
        elif isinstance(value, (list, tuple)):
            value = [int(v) if isinstance(v, np.integer) else v for v in value]
        plain[key] = value
    return plain


__all__ = [
    "MAX_EXAMPLES",
    "SEVERITIES",
    "Rule",
    "ValidationReport",
    "get_rule",
    "get_rules",
    "hard_valid_mask",
    "rules_catalog",
    "validate_table",
]
