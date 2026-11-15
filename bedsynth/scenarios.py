"""Operational scenario catalogue.

A scenario selects a region of the learned joint distribution:

* ``conditions`` - values of categorical/ordinal *train* columns the generated rows must take.
  Lists are expanded into value combinations and rows are allocated in proportion to how often
  each combination occurs in the reference (real) data. The model wrapper
  (``bedsynth.synthesis.models``) serves them by segment routing (conditions on segment columns
  select the segment models) and exact pooled rejection from unconditional segment draws; SDV's
  native conditional sampling (``sample_from_conditions``) is used only by GaussianCopula models,
  for the rarest combinations that a pooled draw cannot serve.
* ``filters``    - acceptance criteria evaluated on the *derived* table after sampling
  (rejection sampling), e.g. ``{"emergency_boarders": {"min": 1}}``.

Filter grammar (all bounds inclusive): ``{"min": x}``, ``{"max": y}``, ``{"in": [...]}``,
``{"not_in": [...]}`` - keys may be combined.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from bedsynth.schema import (
    DELAY_REASONS,
    DELAY_THRESHOLD_HOURS,
    PATIENT_STAYS,
    WARD_DAILY,
    WARD_IDS,
    WARDS,
    WEEKDAYS,
    WEEKEND_DAYS,
    column_names,
    conditionable_columns,
)

SCENARIO_CATEGORIES: dict[str, str] = {
    "baseline": "Representative mix of everyday operating conditions",
    "occupancy": "Normal, high and low bed occupancy",
    "seasonal": "Seasonal demand",
    "patient_flow": "Admission and discharge patterns",
    "emergency": "Emergency bed requirements",
    "utilization": "Ward utilisation and capacity",
    "scheduling": "Scheduled procedures and elective scheduling conditions",
    "turnover": "Bed turnover",
    "delayed_discharge": "Delayed discharge",
    "staffing": "Varying staffing conditions",
}


@dataclass(frozen=True)
class Scenario:
    name: str
    table: str
    title: str
    category: str
    description: str
    conditions: dict[str, tuple] = field(default_factory=dict)
    filters: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "table": self.table,
            "title": self.title,
            "category": self.category,
            "category_description": SCENARIO_CATEGORIES.get(self.category, ""),
            "description": self.description,
            "conditions": {k: list(v) for k, v in self.conditions.items()},
            "filters": {k: dict(v) for k, v in self.filters.items()},
        }


_S = Scenario
_NON_NONE_DELAYS = tuple(r for r in DELAY_REASONS if r != "none")
# Inpatient wards: every ward except the acute assessment unit and paediatrics, whose short stays
# make a high daily turnover the norm rather than an operating condition.
_INPATIENT_WARDS = tuple(w for w in WARD_IDS if WARDS[w].ward_type not in ("acute", "paediatric"))

SCENARIOS: dict[str, Scenario] = {
    s.name: s
    for s in (
        # ------------------------------------------------------------- ward_daily
        _S("baseline", WARD_DAILY, "Baseline operations", "baseline",
           "Unconditional sample reproducing the real mix of wards, seasons and pressures."),
        _S("normal_occupancy", WARD_DAILY, "Normal occupancy", "occupancy",
           "Ward-days running at 75-90% midnight occupancy.",
           conditions={"occupancy_level": ("normal",)}),
        _S("high_occupancy", WARD_DAILY, "High occupancy", "occupancy",
           "Ward-days at or above 90% midnight occupancy (high or critical band).",
           conditions={"occupancy_level": ("high", "critical")}),
        _S("low_occupancy", WARD_DAILY, "Low occupancy", "occupancy",
           "Ward-days below 75% midnight occupancy.",
           conditions={"occupancy_level": ("low",)}),
        _S("winter_demand", WARD_DAILY, "Winter seasonal demand", "seasonal",
           "December-February operations (respiratory season, emergency surge).",
           conditions={"month": (12, 1, 2)}),
        _S("summer_demand", WARD_DAILY, "Summer seasonal demand", "seasonal",
           "June-August operations (lower emergency demand, reduced August electives).",
           conditions={"month": (6, 7, 8)}),
        _S("emergency_pressure", WARD_DAILY, "Emergency bed pressure", "emergency",
           "High-occupancy days with emergency patients still waiting for a bed at midnight.",
           conditions={"occupancy_level": ("high", "critical")},
           filters={"emergency_boarders": {"min": 1}}),
        _S("escalation_capacity", WARD_DAILY, "Escalation beds open under pressure", "utilization",
           "High- or critical-occupancy days on which temporary surge (escalation) beds were open "
           "(escalation is decided hospital-wide, so on its own it would mostly select winter hub-ward days).",
           conditions={"occupancy_level": ("high", "critical")},
           filters={"escalation_beds_open": {"min": 1}}),
        _S("reduced_capacity", WARD_DAILY, "Reduced bed capacity", "utilization",
           "Days with beds closed (staffing, infection control or maintenance).",
           filters={"beds_closed": {"min": 1}}),
        _S("scheduled_procedures", WARD_DAILY, "Scheduled procedure days", "scheduling",
           "Weekday elective lists on surgical and cardiology wards (2+ bookings).",
           conditions={"ward_id": ("GSUR", "ORTH", "CARD"), "day_of_week": WEEKDAYS},
           filters={"elective_scheduled": {"min": 2}}),
        _S("elective_cancellations", WARD_DAILY, "Elective cancellations", "scheduling",
           "Days on which at least one booked elective admission was cancelled.",
           filters={"elective_cancelled": {"min": 1}}),
        _S("high_turnover", WARD_DAILY, "High bed turnover (inpatient wards)", "turnover",
           "Inpatient-ward days on which at least 30% of available beds were vacated (discharges + transfers "
           "out). The acute assessment and children's wards, where rapid turnover is the norm every day, are "
           "excluded so the scenario selects busy days rather than ward types.",
           conditions={"ward_id": _INPATIENT_WARDS},
           filters={"bed_turnover_rate": {"min": 0.30}}),
        _S("delayed_discharge_pressure", WARD_DAILY, "Delayed discharge pressure", "delayed_discharge",
           "Days on which 15%+ of inpatients at midnight were medically fit but awaiting discharge.",
           filters={"delayed_discharge_rate": {"min": 0.15}}),
        _S("staffing_shortage", WARD_DAILY, "Staffing shortage", "staffing",
           "Days with nurse staffing below the ward requirement.",
           conditions={"staffing_level": ("short",)}),
        _S("staffing_surplus", WARD_DAILY, "Staffing surplus", "staffing",
           "Days with more day-shift nurses than the requirement plus its slack (extra bank or agency cover), "
           "the opposite end of the staffing range to staffing_shortage.",
           conditions={"staffing_level": ("surplus",)}),
        _S("weekend_operations", WARD_DAILY, "Weekend operations", "patient_flow",
           "Saturday/Sunday operations (few electives, reduced discharges).",
           conditions={"day_of_week": WEEKEND_DAYS}),
        # ---------------------------------------------------------- patient_stays
        _S("stays_baseline", PATIENT_STAYS, "Baseline stays", "baseline",
           "Unconditional sample of ward stays reproducing the real admission mix."),
        _S("emergency_admissions", PATIENT_STAYS, "Emergency admissions", "patient_flow",
           "Stays admitted through the emergency department.",
           conditions={"admission_source": ("emergency",)}),
        _S("elective_admissions", PATIENT_STAYS, "Elective (scheduled) admissions", "scheduling",
           "Planned admissions for scheduled procedures.",
           conditions={"admission_source": ("elective",)}),
        _S("rescheduled_electives", PATIENT_STAYS, "Rescheduled electives", "scheduling",
           "Elective admissions that were postponed at least once.",
           conditions={"admission_source": ("elective",)},
           filters={"times_rescheduled": {"min": 1}}),
        _S("delayed_discharge_stays", PATIENT_STAYS, "Delayed discharges", "delayed_discharge",
           f"Stays whose departure was delayed by {DELAY_THRESHOLD_HOURS:g}+ hours after the patient was ready to leave.",
           conditions={"delay_reason": _NON_NONE_DELAYS},
           filters={"discharge_delay_hours": {"min": DELAY_THRESHOLD_HOURS}}),
        _S("long_stay_patients", PATIENT_STAYS, "Long-stay patients", "utilization",
           "Stays of 7 or more nights ('stranded' patients).",
           filters={"los_nights": {"min": 7}}),
        _S("weekend_discharges", PATIENT_STAYS, "Weekend discharges", "patient_flow",
           "Stays ending on a Saturday or Sunday.",
           filters={"discharge_day_of_week": {"in": list(WEEKEND_DAYS)}}),
        _S("winter_admissions", PATIENT_STAYS, "Winter admissions", "seasonal",
           "Stays admitted December-February.",
           conditions={"admission_month": (12, 1, 2)}),
        _S("slow_bed_turnaround", PATIENT_STAYS, "Slow bed turnaround", "turnover",
           "Stays after which the bed took 3+ hours to be ready (deep clean, night/weekend).",
           filters={"bed_turnaround_minutes": {"min": 180}}),
        _S("long_bed_waits", PATIENT_STAYS, "Long emergency bed waits", "emergency",
           "Emergency admissions that waited 4+ hours for a bed.",
           conditions={"admission_source": ("emergency",)},
           filters={"bed_wait_hours": {"min": 4}}),
    )
}


def get_scenario(name: str) -> Scenario:
    try:
        return SCENARIOS[name]
    except KeyError as exc:
        raise KeyError(f"Unknown scenario '{name}'. Known scenarios: {sorted(SCENARIOS)}") from exc


def scenarios_for(table: str) -> list[Scenario]:
    return [s for s in SCENARIOS.values() if s.table == table]


def baseline_scenario(table: str) -> Scenario:
    return SCENARIOS["baseline" if table == WARD_DAILY else "stays_baseline"]


def validate_conditions(table: str, conditions: dict[str, Any] | None) -> dict[str, tuple]:
    """Normalise user conditions to ``{column: tuple(values)}`` and check columns are conditionable."""
    if not conditions:
        return {}
    allowed = set(conditionable_columns(table))
    normalised: dict[str, tuple] = {}
    for column, values in conditions.items():
        if column not in allowed:
            raise ValueError(
                f"Column '{column}' cannot be used as a condition for table '{table}'. "
                f"Conditionable columns: {sorted(allowed)}"
            )
        if isinstance(values, (list, tuple, set)):
            vals = tuple(values)
        else:
            vals = (values,)
        if not vals:
            raise ValueError(f"Condition for '{column}' must list at least one value.")
        normalised[column] = vals
    return normalised


def validate_filters(table: str, filters: dict[str, dict] | None) -> dict[str, dict]:
    if not filters:
        return {}
    allowed = set(column_names(table, ["train", "derived"]))
    out: dict[str, dict] = {}
    for column, spec in filters.items():
        if column not in allowed:
            raise ValueError(f"Column '{column}' cannot be filtered for table '{table}'.")
        if not isinstance(spec, dict) or not spec:
            raise ValueError(f"Filter for '{column}' must be a non-empty object with min/max/in/not_in.")
        unknown = set(spec) - {"min", "max", "in", "not_in"}
        if unknown:
            raise ValueError(f"Unknown filter keys {sorted(unknown)} for '{column}'.")
        out[column] = dict(spec)
    return out


def merge_conditions(base: dict[str, tuple], extra: dict[str, tuple]) -> dict[str, tuple]:
    """Intersect two condition dicts (a column present in both keeps only common values)."""
    merged = dict(base)
    for column, values in extra.items():
        if column in merged:
            common = tuple(v for v in merged[column] if v in set(values))
            if not common:
                raise ValueError(f"Conditions on '{column}' do not overlap: {merged[column]} vs {values}")
            merged[column] = common
        else:
            merged[column] = tuple(values)
    return merged


def merge_filters(base: dict[str, dict], extra: dict[str, dict]) -> dict[str, dict]:
    merged = {k: dict(v) for k, v in base.items()}
    for column, spec in extra.items():
        target = merged.setdefault(column, {})
        if "min" in spec:
            target["min"] = max(spec["min"], target.get("min", spec["min"]))
        if "max" in spec:
            target["max"] = min(spec["max"], target.get("max", spec["max"]))
        if "in" in spec:
            target["in"] = [v for v in spec["in"] if "in" not in target or v in target["in"]]
        if "not_in" in spec:
            target["not_in"] = sorted(set(target.get("not_in", [])) | set(spec["not_in"]), key=str)
    return merged


def condition_mask(df: pd.DataFrame, conditions: dict[str, tuple] | None) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    for column, values in (conditions or {}).items():
        if column not in df.columns:
            return pd.Series(False, index=df.index)
        mask &= df[column].isin(list(values))
    return mask


def filter_mask(df: pd.DataFrame, filters: dict[str, dict] | None) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    for column, spec in (filters or {}).items():
        if column not in df.columns:
            return pd.Series(False, index=df.index)
        col = df[column]
        if "min" in spec or "max" in spec:
            numeric = pd.to_numeric(col, errors="coerce")
            if "min" in spec:
                mask &= numeric >= spec["min"]
            if "max" in spec:
                mask &= numeric <= spec["max"]
        if "in" in spec:
            mask &= col.isin(list(spec["in"]))
        if "not_in" in spec:
            mask &= ~col.isin(list(spec["not_in"]))
    return mask.fillna(False).astype(bool)


def scenario_mask(df: pd.DataFrame, scenario: Scenario | str) -> pd.Series:
    """Rows of a fully derived table that belong to a scenario (used for real-data support)."""
    if isinstance(scenario, str):
        scenario = get_scenario(scenario)
    return condition_mask(df, scenario.conditions) & filter_mask(df, scenario.filters)


def scenario_catalog(table: str | None = None) -> list[dict[str, Any]]:
    return [s.to_dict() for s in SCENARIOS.values() if table is None or s.table == table]


__all__ = [
    "SCENARIOS",
    "SCENARIO_CATEGORIES",
    "Scenario",
    "baseline_scenario",
    "condition_mask",
    "filter_mask",
    "get_scenario",
    "merge_conditions",
    "merge_filters",
    "scenario_catalog",
    "scenario_mask",
    "scenarios_for",
    "validate_conditions",
    "validate_filters",
]
