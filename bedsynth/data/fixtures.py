"""Small, fast, rule-satisfying datasets for unit tests.

These tables satisfy every *hard* operational rule and have the exact schema of the
cleaned data, but they are NOT realistic - realistic reference data comes from
``bedsynth.data.simulator``. Use these to test modules in isolation (seconds, not minutes).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from bedsynth.schema import (
    ALLOWED_DELAY_REASONS,
    DATE_FORMAT,
    DATETIME_FORMAT,
    DAYS_OF_WEEK,
    DELAY_THRESHOLD_HOURS,
    PATIENT_STAYS,
    WARD_DAILY,
    WARD_IDS,
    WARDS,
    WEEKEND_DAYS,
    assign_ids,
    derive_patient_stays,
    derive_ward_daily,
    min_rebooked_lead_days,
    occupancy_level_for,
    staffing_level_for,
    staffing_slack,
)


def make_ward_daily(
    n_days: int = 90,
    wards: list[str] | None = None,
    seed: int = 0,
    start_date: str = "2023-01-01",
    with_dates: bool = True,
) -> pd.DataFrame:
    """Return ``n_days`` consecutive ward-days per ward, fully derived, with ids."""
    rng = np.random.default_rng(seed)
    ward_ids = list(wards or WARD_IDS)
    dates = pd.date_range(start_date, periods=n_days, freq="D")
    rows: list[dict] = []
    for ward_id in ward_ids:
        info = WARDS[ward_id]
        census = int(round(info.bed_capacity * rng.uniform(0.7, 0.9)))
        for day in dates:
            dow = DAYS_OF_WEEK[day.dayofweek]
            esc = int(rng.integers(1, info.max_escalation_beds + 1)) if rng.random() < 0.15 else 0
            esc = min(max(esc, census - info.bed_capacity), info.max_escalation_beds)
            physical = info.bed_capacity + esc
            census_start = min(census, physical)
            closed = int(rng.integers(1, 3)) if rng.random() < 0.1 else 0
            closed = max(0, min(closed, physical - 1, physical - census_start))   # only free beds close
            available = physical - closed

            required = info.required_nurses_day
            draw = rng.random()
            if draw < 0.15:
                nurses_day = max(1, required - int(rng.integers(1, max(2, required // 3 + 1))))
            elif draw < 0.93:
                nurses_day = required + int(rng.integers(0, staffing_slack(required) + 1))
            else:
                nurses_day = required + staffing_slack(required) + int(rng.integers(1, 3))
            nurses_night = max(1, info.required_nurses_night + int(rng.integers(-1, 2)))

            delayed = int(rng.binomial(census_start, 0.08))
            scheduled = int(rng.poisson(2.5)) if info.accepts_electives and dow not in WEEKEND_DAYS else 0
            cancelled = int(rng.binomial(scheduled, 0.1))
            adm_el = scheduled - cancelled
            adm_em = int(rng.poisson(max(0.5, info.bed_capacity / 6)))
            tr_in = int(rng.poisson(1.0)) if info.accepts_transfers else 0
            inflow = adm_el + adm_em + tr_in

            target_end = min(max(int(round(available * rng.uniform(0.6, 1.0))), 0), available)
            out_total = min(max(census_start + inflow - target_end, 0), census_start + inflow)
            tr_out = int(rng.binomial(out_total, 0.15)) if ward_id != "PAED" else 0
            discharges = out_total - tr_out
            census_end = census_start + inflow - out_total
            noon = int(rng.binomial(discharges, 0.25))
            boarders = int(rng.poisson(1.5)) if census_end >= available and rng.random() < 0.5 else 0

            rows.append({
                "date": day.strftime(DATE_FORMAT),
                "ward_id": ward_id,
                "month": int(day.month),
                "day_of_week": dow,
                "escalation_beds_open": esc,
                "beds_closed": closed,
                "staffing_level": staffing_level_for(nurses_day, ward_id),
                "nurses_day_shift": nurses_day,
                "nurses_night_shift": nurses_night,
                "census_start": census_start,
                "elective_scheduled": scheduled,
                "elective_cancelled": cancelled,
                "admissions_emergency": adm_em,
                "transfers_in": tr_in,
                "transfers_out": tr_out,
                "discharges": discharges,
                "discharges_before_noon": noon,
                "delayed_discharges": delayed,
                "emergency_boarders": boarders,
                "avg_bed_wait_hours": round(float(rng.gamma(2.0, 1.5)), 2) if adm_em > 0 else 0.0,
                "avg_bed_turnaround_min": round(float(rng.uniform(45, 150)), 1),
                "avg_los_days": round(float(rng.uniform(1, 10)), 2),
                "occupancy_level": occupancy_level_for(census_end, available),
            })
            census = census_end
    df = derive_ward_daily(pd.DataFrame(rows))
    if not with_dates:
        df = df.drop(columns=["date"])
    return assign_ids(df, WARD_DAILY)


def _date_for(rng: np.random.Generator, year: int, month: int, dow: str) -> pd.Timestamp:
    days = pd.date_range(f"{year}-{month:02d}-01", periods=pd.Period(f"{year}-{month:02d}").days_in_month, freq="D")
    matching = [d for d in days if DAYS_OF_WEEK[d.dayofweek] == dow]
    return matching[int(rng.integers(0, len(matching)))]


def make_patient_stays(n: int = 1000, seed: int = 0, with_times: bool = False, year: int = 2023) -> pd.DataFrame:
    """Return ``n`` rule-satisfying ward stays, fully derived, with ids."""
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    non_home = ["home", "home_with_support", "care_facility", "other_hospital"]
    for _ in range(n):
        ward_id = str(rng.choice(WARD_IDS))
        info = WARDS[ward_id]
        source = str(rng.choice(info.admission_sources))
        month = int(rng.integers(1, 13))
        dow = str(rng.choice(DAYS_OF_WEEK))
        if source == "elective":
            adm_hour = int(rng.integers(7, 11))
            lead = int(rng.integers(1, 120))
            resched = int(rng.poisson(0.3))
            lead = max(lead, int(min_rebooked_lead_days(resched)))   # each rebooking is >= 7 days later
            wait = round(float(rng.uniform(0, 1.5)), 2)
        else:
            adm_hour = int(rng.integers(0, 24))
            lead, resched = 0, 0
            wait = round(float(rng.gamma(1.5, 2.0)), 2)
        los_nights = int(rng.geometric(0.25)) - 1
        dis_hour = int(rng.integers(8, 21))
        if los_nights == 0 and dis_hour <= adm_hour:
            if adm_hour < 23:
                dis_hour = int(rng.integers(adm_hour + 1, 24))
            else:
                los_nights = 1
        los_hours = los_nights * 24 + dis_hour - adm_hour
        transfer = ward_id != "PAED" and rng.random() < (0.45 if ward_id == "AMU" else 0.08)
        separation = "transfer_out" if transfer else "discharge"
        destination = "internal_ward" if transfer else str(rng.choice(non_home, p=[0.7, 0.15, 0.1, 0.05]))
        options = [r for r in ALLOWED_DELAY_REASONS[destination] if r != "none"]
        reason = "none" if rng.random() < 0.7 else str(rng.choice(options))
        delay = 0.0
        if reason != "none":   # delays are >= DELAY_THRESHOLD_HOURS and never longer than the stay
            delay = round(float(min(los_hours, rng.uniform(DELAY_THRESHOLD_HOURS, 48))), 1)
            if delay < DELAY_THRESHOLD_HOURS:
                reason, delay = "none", 0.0
        row = {
            "ward_id": ward_id,
            "admission_source": source,
            "admission_month": month,
            "admission_day_of_week": dow,
            "admission_hour": adm_hour,
            "booking_lead_days": lead,
            "times_rescheduled": resched,
            "bed_wait_hours": wait,
            "los_nights": los_nights,
            "discharge_hour": dis_hour,
            "separation_type": separation,
            "discharge_destination": destination,
            "delay_reason": reason,
            "discharge_delay_hours": delay,
            "bed_turnaround_minutes": int(rng.integers(30, 240)),
        }
        if with_times:
            day = _date_for(rng, year, month, dow)
            admit = day + pd.Timedelta(hours=adm_hour, minutes=int(rng.integers(0, 60)))
            leave = day.normalize() + pd.Timedelta(days=los_nights, hours=dis_hour, minutes=int(rng.integers(0, 60)))
            row["admission_datetime"] = admit.strftime(DATETIME_FORMAT)
            row["discharge_datetime"] = leave.strftime(DATETIME_FORMAT)
        rows.append(row)
    return assign_ids(derive_patient_stays(pd.DataFrame(rows)), PATIENT_STAYS)


def make_table(table: str, n: int = 500, seed: int = 0) -> pd.DataFrame:
    """Convenience: ~n rows of either table."""
    if table == WARD_DAILY:
        return make_ward_daily(n_days=max(1, n // len(WARD_IDS)), seed=seed)
    return make_patient_stays(n=n, seed=seed)
