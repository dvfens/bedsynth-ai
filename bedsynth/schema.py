"""BedSynth AI schema - the single source of truth for tables, columns, wards and derivations.

Every layer (simulation, cleaning, SDV metadata, synthesis, validation, evaluation, API)
imports its table/column definitions from here so the whole pipeline stays consistent.

Tables
------
ward_daily     one row per ward per calendar day (capacity, staffing, patient flow,
               emergency demand, scheduling, turnover, occupancy)
patient_stays  one row per inpatient ward stay (bed episode): admission / discharge
               timing, scheduling, length of stay, delays and bed turnaround.
               Strictly non-clinical: no diagnoses, procedures or personal identifiers.

Column roles
------------
key      identifier assigned by the pipeline (never learned by SDV)
time     calendar timestamps: present in source data (and ward_daily calendar-mode output)
         but not learned - SDV cannot learn seasonality from raw timestamps, so calendar
         context is encoded explicitly in train columns (month, day_of_week, ...)
train    attributes learned by the SDV synthesizer
derived  attributes recomputed deterministically from train columns by ``derive()``;
         they are never sampled, which guarantees identities such as the flow balance
         census_end = census_start + admissions + transfers_in - transfers_out - discharges
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Tables & calendar vocabularies
# ---------------------------------------------------------------------------
WARD_DAILY = "ward_daily"
PATIENT_STAYS = "patient_stays"
TABLES: tuple[str, ...] = (WARD_DAILY, PATIENT_STAYS)

DAYS_OF_WEEK: tuple[str, ...] = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
WEEKDAYS: tuple[str, ...] = DAYS_OF_WEEK[:5]
WEEKEND_DAYS: tuple[str, ...] = ("Sat", "Sun")
MONTHS: tuple[int, ...] = tuple(range(1, 13))
SEASONS: tuple[str, ...] = ("winter", "spring", "summer", "autumn")
SEASON_BY_MONTH: dict[int, str] = {
    12: "winter", 1: "winter", 2: "winter",
    3: "spring", 4: "spring", 5: "spring",
    6: "summer", 7: "summer", 8: "summer",
    9: "autumn", 10: "autumn", 11: "autumn",
}
SEASON_MONTHS: dict[str, tuple[int, ...]] = {
    season: tuple(m for m in (12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11) if SEASON_BY_MONTH[m] == season)
    for season in SEASONS
}

DATE_FORMAT = "%Y-%m-%d"
DATETIME_FORMAT = "%Y-%m-%d %H:%M"

# ---------------------------------------------------------------------------
# Operational vocabularies
# ---------------------------------------------------------------------------
OCCUPANCY_LEVELS: tuple[str, ...] = ("low", "normal", "high", "critical")
# Midnight occupancy bands (lower bound inclusive):
#   low < 0.75 <= normal < 0.90 <= high < 0.97 <= critical
OCCUPANCY_THRESHOLDS: dict[str, float] = {"normal": 0.75, "high": 0.90, "critical": 0.97}
STAFFING_LEVELS: tuple[str, ...] = ("short", "adequate", "surplus")

ADMISSION_SOURCES: tuple[str, ...] = ("elective", "emergency", "transfer")
SEPARATION_TYPES: tuple[str, ...] = ("discharge", "transfer_out")
DISCHARGE_DESTINATIONS: tuple[str, ...] = (
    "home", "home_with_support", "care_facility", "other_hospital", "internal_ward",
)
DELAY_REASONS: tuple[str, ...] = (
    "none",
    "awaiting_transport",
    "awaiting_medication",
    "awaiting_documentation",
    "awaiting_care_package",
    "awaiting_care_facility_bed",
    "awaiting_downstream_bed",
)
# A departure counts as a delayed discharge only when it happens more than four hours after
# the patient was ready to leave; shorter gaps are routine (porters, paperwork) and are
# recorded as 0 h with reason 'none'. Recorded at 0.1 h resolution, the smallest delay is 4.0 h.
DELAY_THRESHOLD_HOURS: float = 4.0
# A postponed elective is rebooked at least this many days after the cancelled date, so a stay
# rescheduled r times was booked at least 1 + REBOOK_MIN_DAYS * r days before admission.
REBOOK_MIN_DAYS: int = 7
# Which delay reasons are operationally meaningful for each destination.
ALLOWED_DELAY_REASONS: dict[str, tuple[str, ...]] = {
    "home": ("none", "awaiting_transport", "awaiting_medication", "awaiting_documentation"),
    "home_with_support": ("none", "awaiting_care_package", "awaiting_medication", "awaiting_transport"),
    "care_facility": ("none", "awaiting_care_facility_bed", "awaiting_transport", "awaiting_documentation"),
    "other_hospital": ("none", "awaiting_transport", "awaiting_documentation"),
    "internal_ward": ("none", "awaiting_downstream_bed"),
}


# ---------------------------------------------------------------------------
# Ward catalogue (a 194-bed reference general hospital)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class WardInfo:
    ward_id: str
    name: str
    ward_type: str
    bed_capacity: int              # funded base beds
    max_escalation_beds: int       # surge beds that can be opened under pressure
    nurse_ratio_day: float         # target patients per registered nurse, day shift
    nurse_ratio_night: float       # target patients per registered nurse, night shift
    admission_sources: tuple[str, ...]

    @property
    def required_nurses_day(self) -> int:
        return math.ceil(self.bed_capacity / self.nurse_ratio_day)

    @property
    def required_nurses_night(self) -> int:
        return math.ceil(self.bed_capacity / self.nurse_ratio_night)

    @property
    def accepts_electives(self) -> bool:
        return "elective" in self.admission_sources

    @property
    def accepts_transfers(self) -> bool:
        return "transfer" in self.admission_sources

    @property
    def max_physical_beds(self) -> int:
        return self.bed_capacity + self.max_escalation_beds

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["admission_sources"] = list(self.admission_sources)
        data.update(
            required_nurses_day=self.required_nurses_day,
            required_nurses_night=self.required_nurses_night,
            accepts_electives=self.accepts_electives,
            accepts_transfers=self.accepts_transfers,
        )
        return data


WARDS: dict[str, WardInfo] = {
    w.ward_id: w
    for w in (
        WardInfo("AMU", "Acute Medical Unit", "acute", 28, 6, 5.0, 7.0, ("emergency",)),
        WardInfo("GMED", "General Medicine", "medical", 32, 4, 7.0, 9.0, ("elective", "emergency", "transfer")),
        WardInfo("ELDC", "Elderly Care", "medical", 30, 4, 7.0, 9.0, ("emergency", "transfer")),
        WardInfo("CARD", "Cardiology", "medical", 20, 2, 6.0, 8.0, ("elective", "emergency", "transfer")),
        WardInfo("GSUR", "General Surgery", "surgical", 30, 2, 6.0, 8.0, ("elective", "emergency", "transfer")),
        WardInfo("ORTH", "Orthopaedics", "surgical", 24, 2, 6.0, 8.0, ("elective", "emergency", "transfer")),
        WardInfo("ICU", "Intensive Care Unit", "critical_care", 12, 2, 1.5, 2.0, ("elective", "emergency", "transfer")),
        WardInfo("PAED", "Paediatrics", "paediatric", 18, 2, 4.0, 5.0, ("elective", "emergency")),
    )
}
WARD_IDS: tuple[str, ...] = tuple(WARDS)
WARD_TYPES: tuple[str, ...] = ("acute", "medical", "surgical", "critical_care", "paediatric")
TOTAL_BEDS: int = sum(w.bed_capacity for w in WARDS.values())


def ward(ward_id: str) -> WardInfo:
    try:
        return WARDS[ward_id]
    except KeyError as exc:
        raise KeyError(f"Unknown ward_id '{ward_id}'. Known wards: {list(WARD_IDS)}") from exc


# ---------------------------------------------------------------------------
# Banding helpers (shared by simulator, cleaning, validation and generator)
# ---------------------------------------------------------------------------
def occupancy_level_for(census: float, available_beds: float) -> str | None:
    """Occupancy band for one ward-day, computed from counts (avoids float-rounding edge cases)."""
    if available_beds is None or census is None or pd.isna(available_beds) or pd.isna(census) or available_beds <= 0:
        return None
    rate = round(float(census) / float(available_beds), 6)
    if rate >= OCCUPANCY_THRESHOLDS["critical"]:
        return "critical"
    if rate >= OCCUPANCY_THRESHOLDS["high"]:
        return "high"
    if rate >= OCCUPANCY_THRESHOLDS["normal"]:
        return "normal"
    return "low"


def occupancy_levels(census: pd.Series, available_beds: pd.Series) -> pd.Series:
    """Vectorised :func:`occupancy_level_for` (NaN where undefined)."""
    avail = pd.to_numeric(available_beds, errors="coerce").astype(float)
    cen = pd.to_numeric(census, errors="coerce").astype(float)
    rate = (cen / avail.where(avail > 0)).round(6)
    out = pd.Series(np.nan, index=rate.index, dtype=object)
    out[rate < OCCUPANCY_THRESHOLDS["normal"]] = "low"
    out[(rate >= OCCUPANCY_THRESHOLDS["normal"]) & (rate < OCCUPANCY_THRESHOLDS["high"])] = "normal"
    out[(rate >= OCCUPANCY_THRESHOLDS["high"]) & (rate < OCCUPANCY_THRESHOLDS["critical"])] = "high"
    out[rate >= OCCUPANCY_THRESHOLDS["critical"]] = "critical"
    return out


def min_rebooked_lead_days(times_rescheduled: Any) -> Any:
    """Smallest possible booking lead (days) of an elective postponed ``times_rescheduled`` times.

    Works on scalars, numpy arrays and pandas Series (``1 + REBOOK_MIN_DAYS * r``).
    """
    return 1 + REBOOK_MIN_DAYS * times_rescheduled


def staffing_slack(required_day: int) -> int:
    """Extra nurses above the requirement that still count as 'adequate'."""
    return max(1, round(0.15 * required_day))


def staffing_level_for(nurses_day: float, ward_id: str) -> str | None:
    """short: below requirement; adequate: requirement .. requirement+slack; surplus: above."""
    if ward_id not in WARDS or nurses_day is None or pd.isna(nurses_day):
        return None
    required = WARDS[ward_id].required_nurses_day
    if nurses_day < required:
        return "short"
    if nurses_day <= required + staffing_slack(required):
        return "adequate"
    return "surplus"


def staffing_levels(nurses_day: pd.Series, ward_ids: pd.Series) -> pd.Series:
    """Vectorised :func:`staffing_level_for` (NaN where undefined)."""
    required = ward_ids.map({w: info.required_nurses_day for w, info in WARDS.items()}).astype(float)
    slack = required.map(lambda r: staffing_slack(int(r)) if pd.notna(r) else np.nan)
    nurses = pd.to_numeric(nurses_day, errors="coerce").astype(float)
    out = pd.Series(np.nan, index=nurses.index, dtype=object)
    valid = required.notna() & nurses.notna()
    out[valid & (nurses < required)] = "short"
    out[valid & (nurses >= required) & (nurses <= required + slack)] = "adequate"
    out[valid & (nurses > required + slack)] = "surplus"
    return out


# ---------------------------------------------------------------------------
# Column catalogue
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ColumnSpec:
    name: str
    dtype: str                    # int | float | str | bool | date | datetime
    sdtype: str                   # id | categorical | ordinal | numerical | boolean | datetime
    role: str                     # key | time | train | derived
    description: str
    unit: str | None = None
    categories: tuple | None = None   # allowed values (ordered for ordinal columns)
    min_value: float | None = None    # hard physical lower bound (inclusive)
    max_value: float | None = None    # hard physical upper bound (inclusive); ward-specific limits live in validation
    decimals: int | None = None       # rounding applied to float columns

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if self.categories is not None:
            data["categories"] = list(self.categories)
        return data


def _c(name, dtype, sdtype, role, description, **kw) -> ColumnSpec:
    return ColumnSpec(name=name, dtype=dtype, sdtype=sdtype, role=role, description=description, **kw)


WARD_DAILY_COLUMNS: tuple[ColumnSpec, ...] = (
    _c("record_id", "str", "id", "key", "Record identifier, e.g. WD-000001."),
    _c("date", "date", "datetime", "time", "Calendar date of the ward-day (YYYY-MM-DD). Source data and calendar-mode output only."),
    _c("ward_id", "str", "categorical", "train", "Ward code.", categories=WARD_IDS),
    _c("ward_type", "str", "categorical", "derived", "Ward service type (from ward catalogue).", categories=WARD_TYPES),
    _c("month", "int", "categorical", "train", "Calendar month 1-12.", categories=MONTHS),
    _c("season", "str", "categorical", "derived", "Meteorological season derived from month.", categories=SEASONS),
    _c("day_of_week", "str", "categorical", "train", "Day of week (Mon..Sun).", categories=DAYS_OF_WEEK),
    _c("is_weekend", "bool", "boolean", "derived", "Saturday or Sunday."),
    _c("bed_capacity", "int", "numerical", "derived", "Funded base beds of the ward (catalogue).", unit="beds", min_value=1, decimals=0),
    _c("escalation_beds_open", "int", "numerical", "train", "Temporary surge beds opened that day.", unit="beds", min_value=0, decimals=0),
    _c("beds_closed", "int", "numerical", "train", "Beds unavailable (staffing, infection control, maintenance); only free beds are closed.", unit="beds", min_value=0, decimals=0),
    _c("available_beds", "int", "numerical", "derived", "bed_capacity + escalation_beds_open - beds_closed.", unit="beds", min_value=1, decimals=0),
    _c("staffing_level", "str", "ordinal", "train", "Declared nurse staffing status (consistent with nurses_day_shift vs requirement).", categories=STAFFING_LEVELS),
    _c("nurses_day_shift", "int", "numerical", "train", "Registered nurses rostered on the day shift.", unit="nurses", min_value=1, decimals=0),
    _c("nurses_night_shift", "int", "numerical", "train", "Registered nurses rostered on the night shift.", unit="nurses", min_value=1, decimals=0),
    _c("census_start", "int", "numerical", "train", "Inpatients in the ward at 00:00.", unit="patients", min_value=0, decimals=0),
    _c("elective_scheduled", "int", "numerical", "train", "Elective (planned) admissions booked for the day.", unit="patients", min_value=0, decimals=0),
    _c("elective_cancelled", "int", "numerical", "train", "Booked electives cancelled on the day (bed shortage, staffing, patient).", unit="patients", min_value=0, decimals=0),
    _c("admissions_elective", "int", "numerical", "derived", "elective_scheduled - elective_cancelled.", unit="patients", min_value=0, decimals=0),
    _c("admissions_emergency", "int", "numerical", "train", "Emergency admissions into the ward.", unit="patients", min_value=0, decimals=0),
    _c("transfers_in", "int", "numerical", "train", "Patients transferred in from other wards.", unit="patients", min_value=0, decimals=0),
    _c("transfers_out", "int", "numerical", "train", "Patients transferred out to other wards.", unit="patients", min_value=0, decimals=0),
    _c("discharges", "int", "numerical", "train", "Patients discharged from hospital.", unit="patients", min_value=0, decimals=0),
    _c("discharges_before_noon", "int", "numerical", "train", "Discharges completed before 12:00.", unit="patients", min_value=0, decimals=0),
    _c("census_end", "int", "numerical", "derived", "Inpatients at 24:00 (flow balance).", unit="patients", min_value=0, decimals=0),
    _c("delayed_discharges", "int", "numerical", "train", "Patients at 00:00 medically fit for discharge but still occupying a bed.", unit="patients", min_value=0, decimals=0),
    _c("emergency_boarders", "int", "numerical", "train", "Emergency patients waiting in ED for a bed in this ward at 24:00.", unit="patients", min_value=0, decimals=0),
    _c("avg_bed_wait_hours", "float", "numerical", "train", "Mean bed-request-to-admission wait of the day's emergency admissions (0 if none).", unit="hours", min_value=0, max_value=96, decimals=2),
    _c("avg_bed_turnaround_min", "float", "numerical", "train", "7-day rolling mean minutes from patient departure to bed ready.", unit="minutes", min_value=10, max_value=1440, decimals=1),
    _c("avg_los_days", "float", "numerical", "train", "7-day rolling mean length of stay of patients leaving the ward.", unit="days", min_value=0.04, max_value=90, decimals=2),
    _c("occupancy_rate", "float", "numerical", "derived", "census_end / available_beds (midnight occupancy).", unit="ratio", min_value=0, max_value=1, decimals=4),
    _c("occupancy_level", "str", "ordinal", "train", "Occupancy band: low <0.75 <= normal <0.90 <= high <0.97 <= critical.", categories=OCCUPANCY_LEVELS),
    _c("bed_turnover_rate", "float", "numerical", "derived", "(discharges + transfers_out) / available_beds.", unit="ratio", min_value=0, decimals=4),
    _c("patients_per_nurse", "float", "numerical", "derived", "census_end / nurses_day_shift.", unit="ratio", min_value=0, decimals=2),
    _c("delayed_discharge_rate", "float", "numerical", "derived", "delayed_discharges / census_start (0 if empty).", unit="ratio", min_value=0, max_value=1, decimals=4),
    _c("elective_cancellation_rate", "float", "numerical", "derived", "elective_cancelled / elective_scheduled (0 if none booked).", unit="ratio", min_value=0, max_value=1, decimals=4),
)

PATIENT_STAYS_COLUMNS: tuple[ColumnSpec, ...] = (
    _c("stay_id", "str", "id", "key", "Stay identifier, e.g. ST-0000001."),
    _c("admission_datetime", "datetime", "datetime", "time", "Admission timestamp (YYYY-MM-DD HH:MM). Source data only."),
    _c("discharge_datetime", "datetime", "datetime", "time", "Ward departure timestamp (YYYY-MM-DD HH:MM). Source data only."),
    _c("ward_id", "str", "categorical", "train", "Ward code.", categories=WARD_IDS),
    _c("ward_type", "str", "categorical", "derived", "Ward service type.", categories=WARD_TYPES),
    _c("admission_source", "str", "categorical", "train", "elective (planned), emergency (via ED) or internal transfer.", categories=ADMISSION_SOURCES),
    _c("admission_month", "int", "categorical", "train", "Admission month 1-12.", categories=MONTHS),
    _c("admission_season", "str", "categorical", "derived", "Season of admission.", categories=SEASONS),
    _c("admission_day_of_week", "str", "categorical", "train", "Admission day of week.", categories=DAYS_OF_WEEK),
    _c("admission_hour", "int", "numerical", "train", "Admission hour 0-23.", unit="hour", min_value=0, max_value=23, decimals=0),
    _c("booking_lead_days", "int", "numerical", "train", "Days between booking and admission (electives, at least 1 + 7 x times_rescheduled; 0 otherwise).", unit="days", min_value=0, max_value=365, decimals=0),
    _c("times_rescheduled", "int", "numerical", "train", "Times the elective admission was postponed (0 for non-electives).", unit="count", min_value=0, max_value=10, decimals=0),
    _c("bed_wait_hours", "float", "numerical", "train", "Hours from bed request to bed allocation.", unit="hours", min_value=0, max_value=96, decimals=2),
    _c("los_nights", "int", "numerical", "train", "Number of midnights spent in the ward.", unit="nights", min_value=0, max_value=180, decimals=0),
    _c("discharge_day_of_week", "str", "categorical", "derived", "Day of week of ward departure.", categories=DAYS_OF_WEEK),
    _c("discharge_hour", "int", "numerical", "train", "Ward departure hour 0-23.", unit="hour", min_value=0, max_value=23, decimals=0),
    _c("los_hours", "int", "numerical", "derived", "Hour-resolution length of stay: los_nights*24 + discharge_hour - admission_hour.", unit="hours", min_value=1, decimals=0),
    _c("los_days", "float", "numerical", "derived", "los_hours / 24.", unit="days", min_value=0, decimals=3),
    _c("separation_type", "str", "categorical", "train", "Stay ended by hospital discharge or internal transfer out.", categories=SEPARATION_TYPES),
    _c("discharge_destination", "str", "categorical", "train", "Destination after leaving the ward.", categories=DISCHARGE_DESTINATIONS),
    _c("delay_reason", "str", "categorical", "train", "Reason the departure was delayed after the patient was ready ('none' if not delayed).", categories=DELAY_REASONS),
    _c("discharge_delay_hours", "float", "numerical", "train", "Hours between ready-to-leave and actual departure; 0 unless the gap exceeded the 4 h delay threshold (so either 0 or >= 4).", unit="hours", min_value=0, decimals=1),
    _c("is_delayed_discharge", "bool", "boolean", "derived", "discharge_delay_hours > 0 (a delay of 4 h or more; see PS-H13)."),
    _c("discharged_before_noon", "bool", "boolean", "derived", "discharge_hour < 12."),
    _c("is_weekend_discharge", "bool", "boolean", "derived", "Departure on Saturday or Sunday."),
    _c("bed_turnaround_minutes", "int", "numerical", "train", "Minutes from departure until the bed was cleaned and ready.", unit="minutes", min_value=10, max_value=1440, decimals=0),
)

COLUMNS: dict[str, tuple[ColumnSpec, ...]] = {
    WARD_DAILY: WARD_DAILY_COLUMNS,
    PATIENT_STAYS: PATIENT_STAYS_COLUMNS,
}

# Columns present in the raw hospital extract (before preprocessing derives the rest).
RAW_COLUMNS: dict[str, tuple[str, ...]] = {
    WARD_DAILY: (
        "date", "ward_id", "bed_capacity", "escalation_beds_open", "beds_closed", "staffing_level",
        "nurses_day_shift", "nurses_night_shift", "census_start", "elective_scheduled", "elective_cancelled",
        "admissions_elective", "admissions_emergency", "transfers_in", "transfers_out", "discharges",
        "discharges_before_noon", "census_end", "delayed_discharges", "emergency_boarders",
        "avg_bed_wait_hours", "avg_bed_turnaround_min", "avg_los_days", "occupancy_level",
    ),
    PATIENT_STAYS: (
        "stay_id", "ward_id", "admission_source", "admission_datetime", "discharge_datetime",
        "booking_lead_days", "times_rescheduled", "bed_wait_hours", "separation_type",
        "discharge_destination", "delay_reason", "discharge_delay_hours", "bed_turnaround_minutes",
    ),
}

ID_PREFIX: dict[str, str] = {WARD_DAILY: "WD-", PATIENT_STAYS: "ST-"}
ID_WIDTH: dict[str, int] = {WARD_DAILY: 6, PATIENT_STAYS: 7}


# ---------------------------------------------------------------------------
# Catalogue accessors
# ---------------------------------------------------------------------------
def _check_table(table: str) -> None:
    if table not in COLUMNS:
        raise KeyError(f"Unknown table '{table}'. Known tables: {list(TABLES)}")


def column_specs(table: str, roles: Iterable[str] | None = None) -> list[ColumnSpec]:
    _check_table(table)
    roles = set(roles) if roles is not None else None
    return [c for c in COLUMNS[table] if roles is None or c.role in roles]


def column_spec(table: str, name: str) -> ColumnSpec:
    for spec in column_specs(table):
        if spec.name == name:
            return spec
    raise KeyError(f"Column '{name}' is not part of table '{table}'.")


def column_names(table: str, roles: Iterable[str] | None = None) -> list[str]:
    return [c.name for c in column_specs(table, roles)]


def key_column(table: str) -> str:
    return column_names(table, ["key"])[0]


def time_columns(table: str) -> list[str]:
    return column_names(table, ["time"])


def training_columns(table: str) -> list[str]:
    """Columns learned by SDV, in canonical order."""
    return column_names(table, ["train"])


def derived_columns(table: str) -> list[str]:
    return column_names(table, ["derived"])


def output_columns(table: str, include_time: bool = False) -> list[str]:
    """Columns of a generated dataset (key + optional time + train + derived) in canonical order."""
    roles = {"key", "train", "derived"} | ({"time"} if include_time else set())
    return column_names(table, roles)


def categorical_columns(table: str, roles: Iterable[str] = ("train",)) -> list[str]:
    return [c.name for c in column_specs(table, roles) if c.sdtype in ("categorical", "ordinal")]


def numerical_columns(table: str, roles: Iterable[str] = ("train",)) -> list[str]:
    return [c.name for c in column_specs(table, roles) if c.sdtype == "numerical"]


def boolean_columns(table: str, roles: Iterable[str] = ("derived",)) -> list[str]:
    return [c.name for c in column_specs(table, roles) if c.sdtype == "boolean"]


def integer_columns(table: str, roles: Iterable[str] | None = None) -> list[str]:
    return [c.name for c in column_specs(table, roles) if c.dtype == "int" and c.sdtype == "numerical"]


def conditionable_columns(table: str) -> list[str]:
    """Train columns that may be used in SDV conditional sampling (categorical/ordinal)."""
    return categorical_columns(table, ("train",))


def ordered_categories(table: str) -> dict[str, list]:
    """Categorical columns with a natural order (ordinal sdtype)."""
    return {c.name: list(c.categories) for c in column_specs(table) if c.sdtype == "ordinal" and c.categories}


def order_columns(df: pd.DataFrame, table: str) -> pd.DataFrame:
    """Reorder to canonical order; columns not in the schema are kept at the end."""
    canonical = [c for c in column_names(table) if c in df.columns]
    extras = [c for c in df.columns if c not in canonical]
    return df[canonical + extras]


def schema_as_dict(table: str) -> dict[str, Any]:
    """JSON-serialisable description of a table (used by API and reports)."""
    _check_table(table)
    return {
        "table": table,
        "key": key_column(table),
        "columns": [c.to_dict() for c in COLUMNS[table]],
        "training_columns": training_columns(table),
        "derived_columns": derived_columns(table),
        "time_columns": time_columns(table),
        "raw_columns": list(RAW_COLUMNS[table]),
        "wards": [w.to_dict() for w in WARDS.values()],
    }


# ---------------------------------------------------------------------------
# Category normalisation (shared by dirty-data injection and cleaning)
# Keys are lower-cased, stripped variants; values are canonical codes.
# ---------------------------------------------------------------------------
CATEGORY_ALIASES: dict[str, dict[str, str]] = {
    "ward_id": {
        "amu": "AMU", "a.m.u.": "AMU", "a.m.u": "AMU", "acute medical unit": "AMU", "acute medicine": "AMU",
        "gmed": "GMED", "general medicine": "GMED", "gen med": "GMED", "g.med": "GMED", "gen. medicine": "GMED",
        "eldc": "ELDC", "elderly care": "ELDC", "care of the elderly": "ELDC", "geriatrics": "ELDC", "coe": "ELDC",
        "card": "CARD", "cardiology": "CARD", "cardio": "CARD", "ccu": "CARD",
        "gsur": "GSUR", "general surgery": "GSUR", "gen surg": "GSUR", "surgery": "GSUR",
        "orth": "ORTH", "orthopaedics": "ORTH", "orthopedics": "ORTH", "ortho": "ORTH", "t&o": "ORTH",
        "icu": "ICU", "intensive care unit": "ICU", "itu": "ICU", "critical care": "ICU", "i.c.u.": "ICU",
        "paed": "PAED", "paediatrics": "PAED", "pediatrics": "PAED", "children's ward": "PAED", "peds": "PAED",
    },
    "admission_source": {
        "elective": "elective", "planned": "elective", "elective (planned)": "elective", "el": "elective", "booked": "elective",
        "emergency": "emergency", "a&e": "emergency", "ed": "emergency", "emerg": "emergency", "non-elective": "emergency",
        "transfer": "transfer", "internal transfer": "transfer", "xfer": "transfer", "ward transfer": "transfer",
    },
    "staffing_level": {
        "short": "short", "understaffed": "short", "below establishment": "short", "short-staffed": "short",
        "adequate": "adequate", "safe": "adequate", "ok": "adequate", "at establishment": "adequate",
        "surplus": "surplus", "overstaffed": "surplus", "above establishment": "surplus",
    },
    "occupancy_level": {
        "low": "low", "normal": "normal", "moderate": "normal", "high": "high",
        "critical": "critical", "very high": "critical", "full": "critical",
    },
    "separation_type": {
        "discharge": "discharge", "discharged": "discharge", "dc": "discharge",
        "transfer_out": "transfer_out", "transfer out": "transfer_out", "transferred": "transfer_out", "xfer out": "transfer_out",
    },
    "discharge_destination": {
        "home": "home", "usual residence": "home", "own home": "home",
        "home_with_support": "home_with_support", "home with support": "home_with_support", "home with care package": "home_with_support",
        "care_facility": "care_facility", "care home": "care_facility", "nursing home": "care_facility", "residential care": "care_facility",
        "other_hospital": "other_hospital", "other hospital": "other_hospital", "external transfer": "other_hospital",
        "internal_ward": "internal_ward", "internal ward": "internal_ward", "another ward": "internal_ward",
    },
    "delay_reason": {
        "none": "none", "no delay": "none", "n/a": "none", "na": "none",
        "awaiting_transport": "awaiting_transport", "awaiting transport": "awaiting_transport", "transport": "awaiting_transport",
        "awaiting_medication": "awaiting_medication", "awaiting medication": "awaiting_medication", "pharmacy": "awaiting_medication", "awaiting tto": "awaiting_medication",
        "awaiting_documentation": "awaiting_documentation", "awaiting documentation": "awaiting_documentation", "paperwork": "awaiting_documentation",
        "awaiting_care_package": "awaiting_care_package", "awaiting care package": "awaiting_care_package", "package of care": "awaiting_care_package",
        "awaiting_care_facility_bed": "awaiting_care_facility_bed", "awaiting care home bed": "awaiting_care_facility_bed", "awaiting placement": "awaiting_care_facility_bed",
        "awaiting_downstream_bed": "awaiting_downstream_bed", "awaiting downstream bed": "awaiting_downstream_bed", "awaiting ward bed": "awaiting_downstream_bed",
    },
    "day_of_week": {
        "mon": "Mon", "monday": "Mon", "tue": "Tue", "tues": "Tue", "tuesday": "Tue",
        "wed": "Wed", "wednesday": "Wed", "thu": "Thu", "thur": "Thu", "thurs": "Thu", "thursday": "Thu",
        "fri": "Fri", "friday": "Fri", "sat": "Sat", "saturday": "Sat", "sun": "Sun", "sunday": "Sun",
    },
}
CATEGORY_ALIASES["admission_day_of_week"] = CATEGORY_ALIASES["day_of_week"]


def normalize_category(column: str, value: Any) -> Any:
    """Map a raw categorical value to its canonical code (None if unknown or missing).

    Matching is case-insensitive and ignores surrounding whitespace. When the stripped,
    lower-cased spelling is not a known alias, it is retried with internal whitespace runs
    and underscores collapsed (``"general   medicine"`` -> ``GMED``, ``"Transfer_Out"`` ->
    ``transfer_out``), so the dirty-data injector, the cleaning pipeline, the generator and
    the API all share one normalisation.
    """
    if value is None or (isinstance(value, float) and math.isnan(value)) or value is pd.NA:
        return None
    aliases = CATEGORY_ALIASES.get(column)
    if aliases is None:
        return value
    text = str(value).strip().lower()
    canonical = aliases.get(text)
    if canonical is None:
        compact = " ".join(text.replace("_", " ").split())
        canonical = aliases.get(compact) or aliases.get(compact.replace(" ", "_"))
    return canonical


def normalize_category_series(series: pd.Series, column: str) -> pd.Series:
    if column not in CATEGORY_ALIASES:
        return series
    return series.map(lambda v: normalize_category(column, v))


# ---------------------------------------------------------------------------
# Calendar feature helpers (source data and calendar-mode generation)
# ---------------------------------------------------------------------------
def add_calendar_features(df: pd.DataFrame, date_column: str = "date") -> pd.DataFrame:
    """Add month and day_of_week to ward_daily rows from an ISO ``date`` column."""
    out = df.copy()
    dates = pd.to_datetime(out[date_column], format=DATE_FORMAT, errors="coerce")
    out["month"] = dates.dt.month.astype("Int64")
    out["day_of_week"] = dates.dt.dayofweek.map(dict(enumerate(DAYS_OF_WEEK)))
    return out


def add_stay_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive the learnable timing columns of patient_stays from its timestamps."""
    out = df.copy()
    adm = pd.to_datetime(out["admission_datetime"], format=DATETIME_FORMAT, errors="coerce")
    dis = pd.to_datetime(out["discharge_datetime"], format=DATETIME_FORMAT, errors="coerce")
    out["admission_month"] = adm.dt.month.astype("Int64")
    out["admission_day_of_week"] = adm.dt.dayofweek.map(dict(enumerate(DAYS_OF_WEEK)))
    out["admission_hour"] = adm.dt.hour.astype("Int64")
    out["los_nights"] = (dis.dt.normalize() - adm.dt.normalize()).dt.days.astype("Int64")
    out["discharge_hour"] = dis.dt.hour.astype("Int64")
    return out


# ---------------------------------------------------------------------------
# Derivations
# ---------------------------------------------------------------------------
def _ward_map(field: str) -> dict[str, Any]:
    return {w: getattr(info, field) for w, info in WARDS.items()}


def _safe_ratio(num: pd.Series, den: pd.Series, decimals: int, zero_value: float | None = 0.0) -> pd.Series:
    num = pd.to_numeric(num, errors="coerce").astype(float)
    den = pd.to_numeric(den, errors="coerce").astype(float)
    ratio = num / den.where(den > 0)
    if zero_value is not None:
        ratio = ratio.where(~((den == 0) & num.notna()), zero_value)
    return ratio.round(decimals)


def derive_ward_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Recompute every derived ward_daily column from the train columns.

    Also applies one definitional normalisation: ``avg_bed_wait_hours`` is 0 when there
    were no emergency admissions (the mean of an empty set is defined as 0).
    """
    missing = [c for c in training_columns(WARD_DAILY) if c not in df.columns]
    if missing:
        raise KeyError(f"derive_ward_daily: missing train columns {missing}")
    out = df.copy()
    month = pd.to_numeric(out["month"], errors="coerce")
    out["ward_type"] = out["ward_id"].map(_ward_map("ward_type"))
    out["season"] = month.map(SEASON_BY_MONTH)
    out["is_weekend"] = out["day_of_week"].isin(WEEKEND_DAYS)
    out["bed_capacity"] = out["ward_id"].map(_ward_map("bed_capacity"))
    out["available_beds"] = out["bed_capacity"] + out["escalation_beds_open"] - out["beds_closed"]
    out["admissions_elective"] = out["elective_scheduled"] - out["elective_cancelled"]
    out["census_end"] = (
        out["census_start"]
        + out["admissions_elective"]
        + out["admissions_emergency"]
        + out["transfers_in"]
        - out["transfers_out"]
        - out["discharges"]
    )
    out["occupancy_rate"] = _safe_ratio(out["census_end"], out["available_beds"], 4, zero_value=None)
    out["bed_turnover_rate"] = _safe_ratio(out["discharges"] + out["transfers_out"], out["available_beds"], 4, zero_value=None)
    out["patients_per_nurse"] = _safe_ratio(out["census_end"], out["nurses_day_shift"], 2, zero_value=None)
    out["delayed_discharge_rate"] = _safe_ratio(out["delayed_discharges"], out["census_start"], 4)
    out["elective_cancellation_rate"] = _safe_ratio(out["elective_cancelled"], out["elective_scheduled"], 4)
    no_emergency = pd.to_numeric(out["admissions_emergency"], errors="coerce") == 0
    out.loc[no_emergency, "avg_bed_wait_hours"] = 0.0
    return order_columns(out, WARD_DAILY)


def derive_patient_stays(df: pd.DataFrame) -> pd.DataFrame:
    """Recompute every derived patient_stays column from the train columns."""
    missing = [c for c in training_columns(PATIENT_STAYS) if c not in df.columns]
    if missing:
        raise KeyError(f"derive_patient_stays: missing train columns {missing}")
    out = df.copy()
    month = pd.to_numeric(out["admission_month"], errors="coerce")
    out["ward_type"] = out["ward_id"].map(_ward_map("ward_type"))
    out["admission_season"] = month.map(SEASON_BY_MONTH)
    dow_index = out["admission_day_of_week"].map({d: i for i, d in enumerate(DAYS_OF_WEEK)})
    los_nights = pd.to_numeric(out["los_nights"], errors="coerce")
    out["discharge_day_of_week"] = ((dow_index + los_nights) % 7).map(dict(enumerate(DAYS_OF_WEEK)))
    out["los_hours"] = los_nights * 24 + pd.to_numeric(out["discharge_hour"], errors="coerce") - pd.to_numeric(out["admission_hour"], errors="coerce")
    out["los_days"] = (out["los_hours"] / 24.0).round(3)
    out["is_delayed_discharge"] = pd.to_numeric(out["discharge_delay_hours"], errors="coerce").fillna(0) > 0
    out["discharged_before_noon"] = pd.to_numeric(out["discharge_hour"], errors="coerce") < 12
    out["is_weekend_discharge"] = out["discharge_day_of_week"].isin(WEEKEND_DAYS)
    return order_columns(out, PATIENT_STAYS)


def derive(table: str, df: pd.DataFrame) -> pd.DataFrame:
    _check_table(table)
    return derive_ward_daily(df) if table == WARD_DAILY else derive_patient_stays(df)


def assign_ids(df: pd.DataFrame, table: str, start: int = 1) -> pd.DataFrame:
    """(Re)assign sequential identifiers (WD-000001 / ST-0000001) in the key column."""
    out = df.copy()
    prefix, width = ID_PREFIX[table], ID_WIDTH[table]
    out[key_column(table)] = [f"{prefix}{i:0{width}d}" for i in range(start, start + len(out))]
    return order_columns(out, table)


def coerce_dtypes(df: pd.DataFrame, table: str) -> pd.DataFrame:
    """Cast schema columns to their canonical dtypes (ints use nullable Int64 only if NaN present)."""
    out = df.copy()
    for spec in column_specs(table):
        if spec.name not in out.columns:
            continue
        col = out[spec.name]
        if spec.dtype == "int" or (spec.name in ("month", "admission_month")):
            numeric = pd.to_numeric(col, errors="coerce")
            if numeric.isna().any():
                out[spec.name] = numeric.round().astype("Int64")
            else:
                out[spec.name] = numeric.round().astype("int64")
        elif spec.dtype == "float":
            numeric = pd.to_numeric(col, errors="coerce").astype(float)
            out[spec.name] = numeric.round(spec.decimals) if spec.decimals is not None else numeric
        elif spec.dtype == "bool":
            if col.dtype != bool:
                mapped = col.map(lambda v: v if isinstance(v, (bool, np.bool_)) else (str(v).strip().lower() in ("true", "1", "yes")) if pd.notna(v) else np.nan)
                out[spec.name] = mapped.astype("boolean") if mapped.isna().any() else mapped.astype(bool)
        elif spec.dtype in ("str",):
            out[spec.name] = col.astype(object).where(col.notna(), None)
    return out
