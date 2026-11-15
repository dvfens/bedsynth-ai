"""Discrete-event simulation of the reference hospital (Week-1 source data, DESIGN.md §6.1).

No real hospital extract can be shared - that is the problem BedSynth AI addresses - so a
seeded, patient-level discrete-event simulation of the 194-bed reference hospital produces
the "real" operational data that every later stage (cleaning, SDV training, evaluation) is
judged against. The simulation therefore has to be *operationally coherent*, not just
plausible column by column.

Model overview
--------------
* **Time** runs in integer minutes from the simulation origin (``start_date`` minus
  ``warmup_days``). A ``heapq`` event queue drives bed requests, admissions, clinical
  readiness, departures and bed turnarounds plus the daily operational cycle: the 00:00
  capacity/staffing review, the 07:00 elective bed meeting and the 09:00/14:00 board rounds.
* **Emergency demand** is a non-homogeneous Poisson process per ward (monthly seasonality,
  weekday profile, annual growth, winter flu surges, paediatric respiratory boost) with an
  hourly ED arrival profile and a 0.5-3 h decision-to-admit lag (truncated lognormal).
* **Electives** are booked weeks ahead (lognormal lead time), reviewed at the 07:00 bed
  meeting (projected free beds after reserving 1-2 for emergencies, fewer on short-staffed
  days, ~3 % patient cancellations), cancelled on the day if not bedded by 15:00 and
  rebooked into the least-loaded weekday 7-28 days later (at most 3 times).
* **Bed allocation**: a patient is bedded only if ``occupied + beds_in_turnaround <
  available_beds``; otherwise they wait in the ward's queue (electives first, then FIFO).
  Hence the midnight census never exceeds the available beds. Patients waiting 3-4.5 h (the
  bed manager acts on each 3-hour breach within 90 minutes) may be bedded as outliers on
  another general medical ward with spare beds; ED queues beyond a quarter of a ward's beds
  trigger ambulance diversion of half the new arrivals.
* **Length of stay**: lognormal clinical LOS (calibrated so that the *total* mean stay,
  including departure delays, matches ``los_mean_days``), readiness confirmed on ward
  rounds, destination- and ward-specific discharge delays (children and planned surgical
  patients rarely wait), the weekend discharge gap (separate Saturday/Sunday capacities,
  deferred reviews worked through on Monday-Wednesday, services resuming on Monday, a
  Friday pre-weekend push) and the discharge-hour profile. Internal transfers wait for a
  downstream bed (exit block).
  Discharging is occupancy-elastic, as observed in real hospitals: under bed pressure board
  rounds bring readiness forward and delays are expedited; on quiet wards patients may stay
  until the next ward round.
* **Capacity**: escalation beds open when hospital medical occupancy exceeds 95 % for two
  mornings and close (only when free) after three mornings below 88 %; beds close (only when
  free) for short staffing, infection outbreaks and maintenance.
* **Staffing**: a sticky daily Markov chain per ward (short ~12 % of days, ~22 % in winter)
  that drives bed closures, elective capacity and bed-turnaround times.

The ward_daily table is assembled *after* the run from the stay timestamps and the daily
operational log, so the flow balance, census continuity and the cross-table admission
counts hold by construction; internal invariants are re-checked before returning.
"""
from __future__ import annotations

import copy
import heapq
import json
import logging
import math
import random
import time
from bisect import bisect_right
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from itertools import accumulate, count
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from bedsynth.config import get_path, get_seed, get_settings
from bedsynth.data.io import ground_truth_path, raw_path, write_table
from bedsynth.schema import (
    DATE_FORMAT,
    DATETIME_FORMAT,
    DAYS_OF_WEEK,
    DELAY_THRESHOLD_HOURS,
    PATIENT_STAYS,
    RAW_COLUMNS,
    REBOOK_MIN_DAYS,
    WARD_DAILY,
    WARD_IDS,
    WARDS,
    WardInfo,
    occupancy_levels,
    staffing_levels,
    staffing_slack,
)

logger = logging.getLogger(__name__)

__all__ = [
    "SimulationResult",
    "calibration_summary",
    "generate_source_data",
    "simulate",
]

# ---------------------------------------------------------------------------
# Time units and integer codes used in the hot event loop
# ---------------------------------------------------------------------------
MINUTES_PER_HOUR = 60
MINUTES_PER_DAY = 24 * MINUTES_PER_HOUR
DELAY_THRESHOLD_MIN = int(round(DELAY_THRESHOLD_HOURS * MINUTES_PER_HOUR))   # departures > 4 h after readiness are "delayed"

_ELECTIVE, _EMERGENCY, _TRANSFER = 0, 1, 2
_SOURCE_CODES: dict[str, int] = {"elective": _ELECTIVE, "emergency": _EMERGENCY, "transfer": _TRANSFER}
_SOURCE_NAMES: tuple[str, ...] = ("elective", "emergency", "transfer")

# Stay lifecycle states.
_WAITING, _BEDDED, _DEPARTED, _CANCELLED = 0, 1, 2, 3

# Event kinds double as tie-break priorities for events at the same minute: the midnight
# review must observe the census before anything else happens at 00:00.
(
    _EV_DAY_START,
    _EV_BED_MEETING,
    _EV_BOARD_ROUND,
    _EV_TURNAROUND_END,
    _EV_DEPARTURE,
    _EV_EMERGENCY_ARRIVAL,
    _EV_JOIN_QUEUE,
    _EV_READY,
    _EV_ELECTIVE_TIMEOUT,
    _EV_OVERFLOW_CHECK,
) = range(10)

_WINTER_MONTHS: frozenset[int] = frozenset({12, 1, 2})
_CALENDAR_MARGIN_DAYS = 400       # calendar look-ups beyond the horizon (late departures)

# ---------------------------------------------------------------------------
# Internal model parameters (behavioural assumptions that are not tuning knobs in
# settings.yaml). Values follow DESIGN.md §6.1; each is documented where it is used.
# ---------------------------------------------------------------------------
# Discharge destination mix for stays that end in a hospital discharge.
_DISCHARGE_DESTINATIONS: dict[str, dict[str, float]] = {
    "AMU": {"home": 0.84, "home_with_support": 0.09, "care_facility": 0.03, "other_hospital": 0.04},
    "GMED": {"home": 0.66, "home_with_support": 0.18, "care_facility": 0.11, "other_hospital": 0.05},
    "ELDC": {"home": 0.40, "home_with_support": 0.27, "care_facility": 0.29, "other_hospital": 0.04},
    "CARD": {"home": 0.85, "home_with_support": 0.08, "care_facility": 0.03, "other_hospital": 0.04},
    "GSUR": {"home": 0.88, "home_with_support": 0.06, "care_facility": 0.03, "other_hospital": 0.03},
    "ORTH": {"home": 0.76, "home_with_support": 0.12, "care_facility": 0.08, "other_hospital": 0.04},
    "ICU": {"home": 0.55, "home_with_support": 0.15, "care_facility": 0.05, "other_hospital": 0.25},
    "PAED": {"home": 0.97, "other_hospital": 0.03},
}
_DEFAULT_DESTINATIONS: dict[str, float] = {
    "home": 0.80, "home_with_support": 0.10, "care_facility": 0.05, "other_hospital": 0.05,
}

# Probability of a service delay (> 4 h) after readiness, and its lognormal length
# (median hours, sigma, min hours, max hours), by discharge destination.
_DELAY_PROBABILITY: dict[str, float] = {
    "home": 0.25, "home_with_support": 0.50, "care_facility": 0.80, "other_hospital": 0.50,
}
_DELAY_LENGTH_HOURS: dict[str, tuple[float, float, float, float]] = {
    "home": (5.5, 0.40, 4.25, 30.0),
    "home_with_support": (22.0, 0.70, 4.25, 240.0),
    "care_facility": (72.0, 0.55, 24.0, 240.0),      # 1-10 days
    "other_hospital": (11.0, 0.55, 4.25, 72.0),
}
_WINTER_SOCIAL_CARE_FACTOR = 1.25                    # longer package/placement waits Dec-Feb
_ELECTIVE_DELAY_FACTOR = 0.5                         # planned admissions have planned discharges
# Ward multipliers of the service-delay probability: delayed discharges concentrate on medical
# and elderly wards (social care); children and planned surgical/cardiology patients rarely wait.
_WARD_DELAY_FACTOR: dict[str, float] = {"PAED": 0.2, "GSUR": 0.7, "ORTH": 0.8, "CARD": 0.8}
_SOCIAL_CARE_DESTINATIONS: frozenset[str] = frozenset({"home_with_support", "care_facility"})

# Delay reasons (all within schema.ALLOWED_DELAY_REASONS[destination]).
_SHORT_DELAY_REASONS: dict[str, dict[str, float]] = {
    "home": {"awaiting_transport": 0.40, "awaiting_medication": 0.45, "awaiting_documentation": 0.15},
    "home_with_support": {"awaiting_care_package": 0.55, "awaiting_medication": 0.25, "awaiting_transport": 0.20},
    "care_facility": {"awaiting_care_facility_bed": 0.50, "awaiting_transport": 0.30, "awaiting_documentation": 0.20},
    "other_hospital": {"awaiting_transport": 0.65, "awaiting_documentation": 0.35},
}
_LONG_DELAY_REASONS: dict[str, dict[str, float]] = {          # delays of 24 h or more
    "home": {"awaiting_transport": 0.30, "awaiting_medication": 0.30, "awaiting_documentation": 0.40},
    "home_with_support": {"awaiting_care_package": 0.90, "awaiting_medication": 0.05, "awaiting_transport": 0.05},
    "care_facility": {"awaiting_care_facility_bed": 0.90, "awaiting_transport": 0.03, "awaiting_documentation": 0.07},
    "other_hospital": {"awaiting_transport": 0.50, "awaiting_documentation": 0.50},
}
_WEEKEND_DELAY_REASONS: dict[str, dict[str, float]] = {       # weekend service gaps
    "home": {"awaiting_documentation": 0.40, "awaiting_medication": 0.35, "awaiting_transport": 0.25},
    "home_with_support": {"awaiting_care_package": 0.80, "awaiting_medication": 0.10, "awaiting_transport": 0.10},
    "care_facility": {"awaiting_care_facility_bed": 0.60, "awaiting_documentation": 0.25, "awaiting_transport": 0.15},
    "other_hospital": {"awaiting_transport": 0.60, "awaiting_documentation": 0.40},
}
_LONG_DELAY_MIN = 24 * MINUTES_PER_HOUR
_DOWNSTREAM_DELAY_REASON = "awaiting_downstream_bed"

# Length of stay. Clinical LOS (admission -> ready to leave) is lognormal; electives are tighter.
_CLINICAL_LOS_SIGMA: dict[int, float] = {_ELECTIVE: 0.50, _EMERGENCY: 0.80, _TRANSFER: 0.75}
_MIN_CLINICAL_LOS_MIN = 120          # >= 2 h keeps los_hours >= 1 and delay <= los_hours
_MAX_CLINICAL_LOS_MIN = 120 * MINUTES_PER_DAY
_WARD_ROUND_START_MIN = 8 * MINUTES_PER_HOUR   # readiness is confirmed on ward rounds:
_WARD_ROUND_END_MIN = 17 * MINUTES_PER_HOUR    # in hours 08:00-16:59 it stands, otherwise
_WARD_ROUND_SPREAD_MIN = 5 * MINUTES_PER_HOUR  # it moves to the next round (08:00-12:59)
_LOS_CALIBRATION_SAMPLES = 1500
_LOS_CALIBRATION_ITERATIONS = 3
_LOS_CALIBRATION_SEED = 20220101     # fixed: model parameters depend on settings only
_ASSUMED_TRANSFER_WAIT_MIN = 90      # expected downstream wait used in the LOS calibration

# Access to a bed.
_DECISION_TO_ADMIT_HOURS = (1.4, 0.45, 0.5, 3.0)   # ED: lognormal median, sigma, truncation range
                                                   # (draws outside are redrawn: no point masses)
_ELECTIVE_CHECKIN_HOURS = (0.0, 1.5, 0.3)          # triangular (low, high, mode)
_TRANSFER_HANDOVER_HOURS = (0.25, 1.0)             # uniform porter/handover time
_ELECTIVE_ARRIVAL_HOURS: dict[int, float] = {7: 0.40, 8: 0.30, 9: 0.20, 10: 0.10}
_ELECTIVE_CUTOFF_MIN = 15 * MINUTES_PER_HOUR - 1   # electives unbedded by 14:59 are cancelled
_ED_DIVERSION_HOURS = 48                           # safety valve: divert to another hospital
_TRANSFER_ABANDON_HOURS = 60                       # safety valve: discharge instead of transfer
_DIVERT_QUEUE_SHARE = 0.25                         # ambulance divert once the ED queue for a
_DIVERT_PROB = 0.5                                 # ward reaches 25 % of its beds (half diverted)

# Elective scheduling.
_ELECTIVE_LEAD_MEDIAN_DAYS: dict[str, float] = {"ORTH": 60.0}
_ELECTIVE_LEAD_DEFAULT_MEDIAN = 30.0
_ELECTIVE_LEAD_SIGMA = 0.6
_ELECTIVE_LEAD_MAX_DAYS = 270                      # + up to 3 rebookings stays <= 365
_PATIENT_CANCELLATION_PROB = 0.03
_SHORT_STAFF_ELECTIVE_FACTOR = 0.80                # ~20 % fewer electives bedded on short-staffed days
_REBOOK_DAYS = (REBOOK_MIN_DAYS, 28)                 # validation rule PS-H14 relies on the minimum
_MAX_RESCHEDULES = 3
# Bed meeting projection (relative to 07:00): departures already planned before 15:00 and
# patients due to be ready by 14:00 count fully, patients awaiting a transfer count 0.5.
# One bed is reserved for emergencies, two when patients are already waiting for a bed.
_PLANNED_DEPARTURE_CUTOFF_MIN = 8 * MINUTES_PER_HOUR
_EXPECTED_READY_CUTOFF_MIN = 7 * MINUTES_PER_HOUR
_EXPECTED_READY_WEIGHT = 1.0
_AWAITING_TRANSFER_WEIGHT = 0.5
_EMERGENCY_RESERVE_BEDS = (1, 2)     # (normal, when the ward already has a queue)
_ELECTIVE_RISK_BEDS = 2              # bookings accepted beyond the projection (bed manager's
                                     # judgement; the 15:00 cut-off is the safety net)

# Occupancy-elastic discharging (well documented in practice): under bed pressure the 09:00
# and 14:00 board rounds bring forward patients due to be ready within a day and discharge
# delays are expedited; on a quiet ward a patient may simply stay until the next ward round.
_BOARD_ROUNDS_MIN: tuple[int, ...] = (9 * MINUTES_PER_HOUR, 14 * MINUTES_PER_HOUR)
_PULL_FORWARD_HEADROOM: tuple[int, ...] = (2, 0)   # morning round aims for two free beds; the
                                                    # afternoon round only clears queues, except
_EVENING_TAKE_HEADROOM = 2                          # on wards without elective lists (AMU, ELDC),
                                                    # which keep one bed free for the evening take
_PULL_FORWARD_WINDOW_MIN = 24 * MINUTES_PER_HOUR
_PULL_FORWARD_PROB = 0.6
_PRESSURE_DELAY_FACTOR = 0.6         # service-delay probability multiplier under pressure
_PRESSURE_SOCIAL_CARE_FACTOR = 0.6   # "discharge to assess": shorter package/placement waits
_QUIET_OCCUPANCY = 0.92              # below this (and nobody waiting) there is no pressure to discharge: a
_QUIET_STRETCH_PROB = 0.6            # ready non-elective patient may stay until the next ward round
_ALWAYS_QUIET_OCCUPANCY = 0.75       # wards planned at or below this are quiet nearly always

# Medical outliers: an emergency/transfer patient who has waited 3 h since the bed request
# may be bedded on another general medical ward that has more than two free beds and nobody
# waiting (bed managers pool the medical bed base to meet the ED 4-hour standard). The bed
# manager acts on each breach within the following 90 minutes (a per-patient threshold drawn
# uniformly), so outlier waits are spread over 3-4.5 h instead of piling up at exactly 3.00 h.
# Specialty and surgical wards are not used as outlier wards.
_OVERFLOW_AFTER_MIN = 3 * MINUTES_PER_HOUR
_OVERFLOW_JITTER_MIN = 90
_OUTLIER_RESERVE_BEDS = 2            # beds an outlier ward keeps free for its own patients
_OVERFLOW_WARDS: dict[str, tuple[str, ...]] = {
    "AMU": ("GMED", "ELDC"),
    "GMED": ("ELDC",),
    "ELDC": ("GMED",),
    "CARD": ("GMED",),
}

# Bed turnaround (departure -> bed ready).
_TURNAROUND_MEDIAN_MIN = 70.0
_TURNAROUND_SIGMA = 0.45
_TURNAROUND_NIGHT_FACTOR = 1.6       # departures 20:00-06:59
_TURNAROUND_WEEKEND_FACTOR = 1.25
_TURNAROUND_SHORT_STAFF_FACTOR = 1.3
_DEEP_CLEAN_PROB = 0.05
_DEEP_CLEAN_RANGE_MIN = (180.0, 360.0)
_TURNAROUND_CLIP_MIN = (15, 900)
_INITIAL_TURNAROUND_MIN = 75.0

# Nurse staffing: sticky daily Markov chain whose stationary mix varies by month.
_SHORT_STAFFING_BY_MONTH: dict[int, float] = {12: 0.22, 1: 0.22, 2: 0.22, 3: 0.12, 11: 0.12}
_SHORT_STAFFING_DEFAULT = 0.077      # -> ~12 % of days over a year
_SURPLUS_STAFFING_PROB = 0.06
_STAFFING_PERSISTENCE = 0.45         # probability the previous day's state carries over
_SHORT_STAFF_BED_CLOSURES = (0.35, 0.40, 0.25)   # P(0, 1, 2 beds closed) on short days

# Bed closures (only free beds are ever closed).
_OUTBREAK_WARDS: tuple[str, ...] = ("AMU", "GMED", "ELDC", "CARD", "PAED")
_OUTBREAKS_PER_YEAR = 8.0
_OUTBREAK_MONTH_WEIGHT: dict[int, float] = {
    12: 3.0, 1: 3.0, 2: 2.5, 3: 1.5, 11: 1.5, 4: 1.0, 10: 1.0, 5: 0.6, 9: 0.6, 6: 0.4, 7: 0.4, 8: 0.4,
}
_OUTBREAK_BEDS = (3, 6)
_OUTBREAK_DAYS = (3, 7)
_MAINTENANCE_MONTHS: tuple[int, ...] = (6, 7, 8, 9)
_MAINTENANCE_PER_YEAR = 3.0
_MAINTENANCE_BEDS = (2, 4)
_MAINTENANCE_DAYS = (2, 5)

# Escalation (surge) beds.
_ESCALATION_HUB_WARDS: tuple[str, ...] = ("AMU", "GMED", "ELDC")
_MEDICAL_PRESSURE_WARDS: tuple[str, ...] = ("AMU", "GMED", "ELDC", "CARD")
_ESCALATION_OPEN_THRESHOLD = 0.95
_ESCALATION_OPEN_MORNINGS = 2
_ESCALATION_CLOSE_THRESHOLD = 0.88
_ESCALATION_CLOSE_MORNINGS = 3
_ESCALATION_HUB_STEP = 2
_ESCALATION_OTHER_MAX = 2

# Demand modifiers.
_FLU_SENSITIVITY: dict[str, float] = {
    "AMU": 1.0, "GMED": 1.0, "ELDC": 1.0, "CARD": 0.6, "GSUR": 0.2, "ORTH": 0.2, "ICU": 0.8, "PAED": 1.0,
}
_WEEKEND_HOUR_FLATTENING = 0.6       # weekend discharge-hour profile = weights ** 0.6
# Weekend discharge gap (DESIGN.md §6.1): a share of the gap comes from missing senior reviews
# (readiness confirmed on a later weekday), the rest from closed services (pharmacy, transport;
# the departure slips to Monday and is recorded as a delay). Reviews deferred over the weekend
# are worked through on Monday-Wednesday, and patients due to be ready at the weekend may be
# discharged on Friday afternoon instead (pre-weekend discharge planning).
_WEEKEND_REVIEW_SHARE = 0.6
_WEEKEND_BACKLOG_SPREAD: tuple[float, ...] = (0.50, 0.30, 0.20)    # Mon, Tue, Wed
_FRIDAY_PUSH_ROUND_MIN = 14 * MINUTES_PER_HOUR                     # Friday afternoon board round
_FRIDAY_PUSH_SPREAD_MIN = 3 * MINUTES_PER_HOUR
_FRIDAY_PUSH_MIN_STAY_MIN = MINUTES_PER_DAY                        # only after a day in hospital
_INITIAL_CENSUS_SHARE = 0.80


# ---------------------------------------------------------------------------
# Public result type
# ---------------------------------------------------------------------------
@dataclass
class SimulationResult:
    """Output of :func:`simulate`.

    Attributes:
        ward_daily: ``RAW_COLUMNS['ward_daily']``, one clean row per ward per day of the
            window, sorted by ``date, ward_id``.
        patient_stays: ``RAW_COLUMNS['patient_stays']``, one clean row per ward stay admitted
            in the window, sorted by ``admission_datetime``.
        calibration: :func:`calibration_summary` of the two tables (DESIGN.md §6.1 targets).
        diagnostics: run statistics (seed, period, runtime, event counts, safety valves, ...).
    """

    ward_daily: pd.DataFrame
    patient_stays: pd.DataFrame
    calibration: dict
    diagnostics: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
class _Choice:
    """Fast weighted choice from a fixed table, driven by a ``random.Random`` stream."""

    __slots__ = ("values", "cum", "total")

    def __init__(self, values: Iterable[Any], weights: Iterable[float]) -> None:
        self.values = tuple(values)
        self.cum = list(accumulate(float(w) for w in weights))
        if len(self.cum) != len(self.values) or not self.cum or self.cum[-1] <= 0:
            raise ValueError("_Choice needs one positive-total weight per value")
        self.total = self.cum[-1]

    @classmethod
    def from_mapping(cls, mapping: Mapping[Any, float]) -> "_Choice":
        return cls(mapping.keys(), mapping.values())

    def draw(self, rng: random.Random) -> Any:
        return self.values[bisect_right(self.cum, rng.random() * self.total)]


def _simulation_settings(settings: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return a copy of the ``simulation`` settings section.

    ``settings`` may be the full settings dictionary, the ``simulation`` section itself or
    ``None`` (read from ``config/settings.yaml``).
    """
    if settings is None:
        return get_settings()["simulation"]
    if isinstance(settings.get("simulation"), Mapping):
        return copy.deepcopy(dict(settings["simulation"]))
    return copy.deepcopy(dict(settings))


def _month_lookup(mapping: Mapping[Any, float] | None, default: float = 1.0) -> np.ndarray:
    """Array indexed by month (1-12) from a mapping keyed by int or str months."""
    table = np.full(13, default, dtype=float)
    for key, value in (mapping or {}).items():
        table[int(key)] = float(value)
    return table


def _dow_lookup(mapping: Mapping[str, float] | None, default: float = 1.0) -> np.ndarray:
    """Array indexed by weekday (0=Mon) from a mapping keyed by 'Mon'..'Sun'."""
    mapping = mapping or {}
    return np.array([float(mapping.get(day, default)) for day in DAYS_OF_WEEK])


def _ceil_div(values: np.ndarray, divisor: int) -> np.ndarray:
    return -((-values) // divisor)


def _align_to_ward_round(t: int, rng: random.Random) -> int:
    """Readiness is confirmed on ward rounds: keep in-hours times, else move to the next round."""
    day, minute = divmod(t, MINUTES_PER_DAY)
    if _WARD_ROUND_START_MIN <= minute < _WARD_ROUND_END_MIN:
        return t
    if minute >= _WARD_ROUND_END_MIN:
        day += 1
    return day * MINUTES_PER_DAY + _WARD_ROUND_START_MIN + rng.randrange(_WARD_ROUND_SPREAD_MIN)


def _lognormal_minutes(rng: random.Random, mean_min: float, sigma: float) -> int:
    """Lognormal draw with the given *mean* (minutes), clipped to the clinical LOS range."""
    mu = math.log(mean_min) - 0.5 * sigma * sigma
    value = rng.lognormvariate(mu, sigma)
    return int(min(max(value, _MIN_CLINICAL_LOS_MIN), _MAX_CLINICAL_LOS_MIN))


# ---------------------------------------------------------------------------
# Calendar of a run
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class _Calendar:
    """Day indexing of one run: day 0 is the first warm-up day (the simulation origin)."""

    start: pd.Timestamp
    origin: pd.Timestamp
    window_start: int          # first output day index
    window_end: int            # last output day index (inclusive)
    horizon: int               # days with a daily operational cycle (warm-up + window + cool-down)
    months: np.ndarray         # calendar month per day index (horizon + margin)
    dows: np.ndarray           # weekday per day index, 0 = Monday
    years: np.ndarray          # years since start_date (negative during warm-up)

    @classmethod
    def build(cls, start: pd.Timestamp, end: pd.Timestamp, warmup_days: int, cooldown_days: int) -> "_Calendar":
        origin = start - pd.Timedelta(days=int(warmup_days))
        window_start = int(warmup_days)
        window_end = window_start + int((end - start).days)
        # At least one day after the window so the last midnight (boarders) is observed.
        horizon = window_end + 1 + max(int(cooldown_days), 1)
        dates = pd.date_range(origin, periods=horizon + _CALENDAR_MARGIN_DAYS, freq="D")
        return cls(
            start=start,
            origin=origin,
            window_start=window_start,
            window_end=window_end,
            horizon=horizon,
            months=dates.month.to_numpy(),
            dows=dates.dayofweek.to_numpy(),
            years=(dates - start).days.to_numpy() / 365.25,
        )

    @property
    def origin_dow(self) -> int:
        return int(self.origin.dayofweek)

    def month(self, day: int) -> int:
        return int(self.months[min(max(day, 0), len(self.months) - 1)])

    def window_days(self) -> np.ndarray:
        return np.arange(self.window_start, self.window_end + 1)


# ---------------------------------------------------------------------------
# Exogenous schedules (vectorised, generated before the event loop)
# ---------------------------------------------------------------------------
def _flu_surge_profile(cal: _Calendar, sim: Mapping[str, Any], rng: np.random.Generator) -> tuple[np.ndarray, list[dict]]:
    """Extra emergency-demand fraction per day from winter flu surge episodes.

    Each winter (December of year y to February of y+1) has a surge with probability
    ``probability_per_winter``; it starts between 1 December and 31 January, lasts
    ``duration_days_range`` days and follows a half-sine bump peaking at ``extra``.
    """
    cfg = sim.get("winter_flu_surge") or {}
    probability = float(cfg.get("probability_per_winter", 0.0))
    low, high = (float(v) for v in cfg.get("extra_multiplier_range", (0.0, 0.0)))
    dmin, dmax = (int(v) for v in cfg.get("duration_days_range", (21, 49)))
    extra = np.zeros(cal.horizon, dtype=float)
    episodes: list[dict] = []
    last_day = cal.origin + pd.Timedelta(days=cal.horizon)
    for year in range(cal.origin.year - 1, last_day.year + 1):
        occurs = rng.random() < probability
        offset = int(rng.integers(0, 62))
        duration = int(rng.integers(dmin, dmax + 1))
        peak = float(rng.uniform(low, high))
        if not occurs:
            continue
        first = (pd.Timestamp(year=year, month=12, day=1) - cal.origin).days + offset
        days = np.arange(duration)
        index = first + days
        inside = (index >= 0) & (index < cal.horizon)
        if not inside.any():
            continue
        extra[index[inside]] += peak * np.sin(np.pi * (days[inside] + 0.5) / duration)
        episodes.append({
            "start": (cal.origin + pd.Timedelta(days=int(first))).strftime(DATE_FORMAT),
            "duration_days": duration,
            "peak_extra_demand": round(peak, 3),
        })
    return extra, episodes


def _emergency_rate_matrix(cal: _Calendar, sim: Mapping[str, Any], flu_extra: np.ndarray) -> np.ndarray:
    """Expected emergency bed requests per ward (rows, WARD_IDS order) and day (columns)."""
    n_days = cal.horizon
    months, dows, years = cal.months[:n_days], cal.dows[:n_days], cal.years[:n_days]
    season = _month_lookup(sim.get("emergency_seasonality"))[months]
    weekday = _dow_lookup(sim.get("emergency_dow"))[dows]
    growth = (1.0 + float(sim.get("annual_demand_growth", 0.0))) ** years
    paed_boost = _month_lookup(sim.get("paediatric_winter_boost"))[months]
    common = season * weekday * growth
    rates = np.zeros((len(WARD_IDS), n_days), dtype=float)
    for w, ward_id in enumerate(WARD_IDS):
        base = float(sim["ward_params"].get(ward_id, {}).get("emergency_per_day", 0.0))
        surge = 1.0 + flu_extra * _FLU_SENSITIVITY.get(ward_id, 0.5)
        rate = base * common * surge
        if WARDS[ward_id].ward_type == "paediatric":
            rate = rate * paed_boost
        rates[w] = rate
    return rates


def _truncated_lognormal(rng: np.random.Generator, median: float, sigma: float, low: float, high: float,
                         size: int) -> np.ndarray:
    """Lognormal draws restricted to ``[low, high]`` by redrawing (clipping would pile mass on the bounds)."""
    values = rng.lognormal(np.log(median), sigma, size=size)
    outside = (values < low) | (values > high)
    while outside.any():
        values[outside] = rng.lognormal(np.log(median), sigma, size=int(outside.sum()))
        outside = (values < low) | (values > high)
    return values


def _ward_streams(seed_sequence: np.random.SeedSequence) -> list[np.random.Generator]:
    """One independent generator per ward (common random numbers: changing one ward's
    parameters does not reshuffle the draws of the other wards)."""
    return [np.random.default_rng(child) for child in seed_sequence.spawn(len(WARD_IDS))]


def _emergency_arrivals(cal: _Calendar, sim: Mapping[str, Any], rates: np.ndarray,
                        streams: Sequence[np.random.Generator]) -> list[tuple[int, int, int]]:
    """Draw ED bed requests: ``(join_minute, ward_index, request_minute)`` per patient.

    ``request`` is the ED bed request (hour from ``emergency_hour_weights``); ``join`` adds
    the decision-to-admit / handover lag after which the patient can be bedded.
    """
    weights = np.asarray(sim["emergency_hour_weights"], dtype=float)
    p_hour = weights / weights.sum()
    median, sigma, low, high = _DECISION_TO_ADMIT_HOURS
    arrivals: list[tuple[int, int, int]] = []
    for w in range(rates.shape[0]):
        rng = streams[w]
        counts = rng.poisson(rates[w])
        total = int(counts.sum())
        if total == 0:
            continue
        days = np.repeat(np.arange(rates.shape[1]), counts)
        hours = rng.choice(24, size=total, p=p_hour)
        minutes = rng.integers(0, 60, size=total)
        request = days * MINUTES_PER_DAY + hours * MINUTES_PER_HOUR + minutes
        lag = _truncated_lognormal(rng, median, sigma, low, high, total)
        join = request + np.rint(lag * MINUTES_PER_HOUR).astype(np.int64)
        arrivals.extend(zip(join.tolist(), [w] * total, request.tolist()))
    return arrivals


def _staffing_schedule(cal: _Calendar, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Daily nurse rosters per ward: ``(state, nurses_day, nurses_night)`` arrays.

    The state (0 short, 1 adequate, 2 surplus) follows a sticky Markov chain: with
    probability ``_STAFFING_PERSISTENCE`` yesterday's state carries over, otherwise it is
    redrawn from the month's stationary mix. Day-shift nurse numbers are then drawn so
    that ``schema.staffing_level_for`` reproduces the state exactly.
    """
    n_wards, n_days = len(WARD_IDS), cal.horizon
    months = cal.months[:n_days]
    p_short = np.array([_SHORT_STAFFING_BY_MONTH.get(int(m), _SHORT_STAFFING_DEFAULT) for m in months])
    state = np.ones((n_wards, n_days), dtype=np.int64)
    day_nurses = np.zeros((n_wards, n_days), dtype=np.int64)
    night_nurses = np.zeros((n_wards, n_days), dtype=np.int64)
    for w, ward_id in enumerate(WARD_IDS):
        keep, draw = rng.random(n_days), rng.random(n_days)
        current = 1
        row = state[w]
        for d in range(n_days):
            if d == 0 or keep[d] >= _STAFFING_PERSISTENCE:
                x = draw[d]
                current = 0 if x < p_short[d] else (2 if x < p_short[d] + _SURPLUS_STAFFING_PROB else 1)
            row[d] = current

        info = WARDS[ward_id]
        required = info.required_nurses_day
        slack = staffing_slack(required)
        u, v = rng.random(n_days), rng.random(n_days)
        short = np.maximum(1, required - np.where(u < 0.75, 1, 2))
        adequate = required + np.floor(u * (slack + 1)).astype(np.int64)
        surplus = required + slack + np.where(u < 0.75, 1, 2)
        day = np.select([row == 0, row == 1], [short, adequate], surplus)
        night = info.required_nurses_night - (row == 0).astype(np.int64) + ((row == 2) & (v < 0.5)).astype(np.int64)
        day_nurses[w] = day
        night_nurses[w] = np.maximum(1, np.minimum(night, day))
        # Wards whose requirement cannot go lower than one nurse can never be "short".
        row[(row == 0) & (day >= required)] = 1
    return state, day_nurses, night_nurses


def _closure_schedule(cal: _Calendar, staffing_state: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    """Beds each ward *wants* closed per day (short staffing, outbreaks, maintenance).

    The event loop closes at most the beds that are free at midnight, so outbreak bays
    close progressively as patients leave.
    """
    n_wards, n_days = staffing_state.shape
    months = cal.months[:n_days]
    desired = np.zeros((n_wards, n_days), dtype=np.int64)

    staffing_beds = rng.choice(len(_SHORT_STAFF_BED_CLOSURES), size=(n_wards, n_days), p=_SHORT_STAFF_BED_CLOSURES)
    desired += np.where(staffing_state == 0, staffing_beds, 0)

    beds = np.array([WARDS[w].bed_capacity for w in WARD_IDS], dtype=float)
    month_weight = np.array([_OUTBREAK_MONTH_WEIGHT[int(m)] for m in months])
    p_outbreak = _OUTBREAKS_PER_YEAR / 365.25 * month_weight / np.mean(list(_OUTBREAK_MONTH_WEIGHT.values()))
    outbreak_wards = np.array([WARD_IDS.index(w) for w in _OUTBREAK_WARDS])
    outbreak_p = beds[outbreak_wards] / beds[outbreak_wards].sum()
    n_outbreaks = 0
    for start in np.flatnonzero(rng.random(n_days) < p_outbreak):
        w = int(rng.choice(outbreak_wards, p=outbreak_p))
        size = int(rng.integers(_OUTBREAK_BEDS[0], _OUTBREAK_BEDS[1] + 1))
        days = int(rng.integers(_OUTBREAK_DAYS[0], _OUTBREAK_DAYS[1] + 1))
        desired[w, start:start + days] += size
        n_outbreaks += 1

    in_season = np.isin(months, _MAINTENANCE_MONTHS)
    season_days = 30.5 * len(_MAINTENANCE_MONTHS)
    p_maintenance = np.where(in_season, _MAINTENANCE_PER_YEAR / season_days, 0.0)
    n_maintenance = 0
    for start in np.flatnonzero(rng.random(n_days) < p_maintenance):
        w = int(rng.choice(n_wards, p=beds / beds.sum()))
        size = int(rng.integers(_MAINTENANCE_BEDS[0], _MAINTENANCE_BEDS[1] + 1))
        days = int(rng.integers(_MAINTENANCE_DAYS[0], _MAINTENANCE_DAYS[1] + 1))
        desired[w, start:start + days] += size
        n_maintenance += 1
    return desired, {"outbreak_episodes": n_outbreaks, "maintenance_episodes": n_maintenance}


def _elective_bookings(cal: _Calendar, sim: Mapping[str, Any],
                       streams: Sequence[np.random.Generator]) -> list[dict[int, list[list[int]]]]:
    """Initial elective waiting list: per ward, ``{admission_day: [[booking_day, times_rescheduled], ...]}``."""
    dow_mult = _dow_lookup(sim.get("elective_dow"))
    month_mult = _month_lookup(sim.get("elective_month"))
    n_days = cal.horizon
    bookings: list[dict[int, list[list[int]]]] = []
    for ward_id, rng in zip(WARD_IDS, streams):
        book: dict[int, list[list[int]]] = {}
        rate = float(sim["ward_params"].get(ward_id, {}).get("elective_per_weekday", 0.0))
        if WARDS[ward_id].accepts_electives and rate > 0:
            lam = rate * dow_mult[cal.dows[:n_days]] * month_mult[cal.months[:n_days]]
            per_day = rng.poisson(lam)
            median = _ELECTIVE_LEAD_MEDIAN_DAYS.get(ward_id, _ELECTIVE_LEAD_DEFAULT_MEDIAN)
            leads = np.clip(
                np.rint(rng.lognormal(np.log(median), _ELECTIVE_LEAD_SIGMA, int(per_day.sum()))),
                1, _ELECTIVE_LEAD_MAX_DAYS,
            ).astype(np.int64)
            position = 0
            for day in np.flatnonzero(per_day):
                k = int(per_day[day])
                book[int(day)] = [[int(day - lead), 0] for lead in leads[position:position + k]]
                position += k
        bookings.append(book)
    return bookings


# ---------------------------------------------------------------------------
# Departure model: readiness -> destination, departure time and delay reason
# ---------------------------------------------------------------------------
def _weekend_setting(sim: Mapping[str, Any], key: str, default: float) -> tuple[float, float]:
    """``(Saturday, Sunday)`` values of a setting given as one number or ``{Sat: x, Sun: y}``."""
    raw = sim.get(key, default)
    if isinstance(raw, Mapping):
        values = (float(raw.get("Sat", default)), float(raw.get("Sun", raw.get("Sat", default))))
    else:
        values = (float(raw), float(raw))
    return tuple(min(max(v, 0.0), 1.0) for v in values)  # type: ignore[return-value]


class _DeparturePlanner:
    """Turns a ready-to-leave time into a discharge plan.

    Steps for a hospital discharge (internal transfers are handled by the bed queues):

    1. destination drawn from the ward's discharge mix;
    2. weekend senior-review gap: readiness on Sat/Sun may only be confirmed on a later weekday
       (Monday-Wednesday, ``_WEEKEND_BACKLOG_SPREAD``: the weekend backlog takes a few days to
       clear; not a recorded delay);
    3. service delay with destination-specific probability (scaled by the ward factor, reduced
       when the ward is under bed pressure) and lognormal length, after which the departure hour
       follows ``discharge_hour_weights`` (at/after that hour, else the next day); otherwise a
       prompt departure 15 min-4 h after readiness, hour-weighted;
    4. weekend service gap: a weekend departure may slip to Monday (recorded delay);
    5. a departure more than 4 h after readiness gets a delay reason allowed for the
       destination, otherwise ``'none'``.

    ``weekend_discharge_capacity`` (one number or ``{Sat: x, Sun: y}``) is the probability that a
    patient ready on that day also leaves at the weekend: steps 2 and 4 together keep that share,
    ``_WEEKEND_REVIEW_SHARE`` of the gap coming from step 2. Deferrals always land on weekdays,
    so Saturday's backlog never inflates Sunday. Before all this, :meth:`friday_push` may bring
    a readiness that falls on the weekend forward to Friday afternoon
    (``friday_discharge_push``, pre-weekend discharge planning).
    """

    def __init__(self, sim: Mapping[str, Any], cal: _Calendar) -> None:
        weekday = [float(w) for w in sim["discharge_hour_weights"]]
        weekend = [w ** _WEEKEND_HOUR_FLATTENING for w in weekday]
        self._hours = (_Choice(range(24), weekday), _Choice(range(24), weekend))
        self._hour_weights = (weekday, weekend)
        self._hour_max = (max(weekday), max(weekend))
        capacity = _weekend_setting(sim, "weekend_discharge_capacity", 0.6)
        push = _weekend_setting(sim, "friday_discharge_push", 0.0)
        # Indexed by weekday (0 = Monday): zero on weekdays.
        self._review_defer_p = [0.0] * 5 + [1.0 - c ** _WEEKEND_REVIEW_SHARE for c in capacity]
        self._service_defer_p = [0.0] * 5 + [1.0 - c ** (1.0 - _WEEKEND_REVIEW_SHARE) for c in capacity]
        self._push_p = [0.0] * 5 + list(push)
        self._backlog = _Choice(range(len(_WEEKEND_BACKLOG_SPREAD)), _WEEKEND_BACKLOG_SPREAD)
        self._ward_delay_factor = [_WARD_DELAY_FACTOR.get(ward_id, 1.0) for ward_id in WARD_IDS]
        self._origin_dow = cal.origin_dow
        self._cal = cal
        self._destinations = [
            _Choice.from_mapping(_DISCHARGE_DESTINATIONS.get(ward_id, _DEFAULT_DESTINATIONS)) for ward_id in WARD_IDS
        ]
        self._short_reasons = {k: _Choice.from_mapping(v) for k, v in _SHORT_DELAY_REASONS.items()}
        self._long_reasons = {k: _Choice.from_mapping(v) for k, v in _LONG_DELAY_REASONS.items()}
        self._weekend_reasons = {k: _Choice.from_mapping(v) for k, v in _WEEKEND_DELAY_REASONS.items()}

    # -- calendar helpers ------------------------------------------------------
    def dow(self, day: int) -> int:
        """Weekday of a day index (0 = Monday)."""
        return (self._origin_dow + day) % 7

    def is_weekend(self, day: int) -> bool:
        return self.dow(day) >= 5

    def _slot_on(self, day: int, rng: random.Random) -> int:
        """A departure minute on ``day`` with the hour drawn from the day-type profile."""
        hour = self._hours[self.is_weekend(day)].draw(rng)
        return day * MINUTES_PER_DAY + hour * MINUTES_PER_HOUR + rng.randrange(MINUTES_PER_HOUR)

    def _slot_at_or_after(self, earliest: int, rng: random.Random) -> int:
        """Departure hour from the profile, at/after ``earliest`` that day, else the next day."""
        day, minute = divmod(earliest, MINUTES_PER_DAY)
        hour = self._hours[self.is_weekend(day)].draw(rng)
        e_hour, e_min = divmod(minute, MINUTES_PER_HOUR)
        if hour > e_hour:
            return day * MINUTES_PER_DAY + hour * MINUTES_PER_HOUR + rng.randrange(MINUTES_PER_HOUR)
        if hour == e_hour:
            return earliest + rng.randrange(MINUTES_PER_HOUR - e_min)
        return self._slot_on(day + 1, rng)

    def _prompt_departure(self, ready: int, rng: random.Random) -> int:
        """15 min-4 h after readiness, accepted in proportion to the discharge-hour profile."""
        candidate = ready + 15
        for _ in range(25):
            candidate = ready + 15 + rng.randrange(DELAY_THRESHOLD_MIN - 14)
            day, minute = divmod(candidate, MINUTES_PER_DAY)
            weekend = self.is_weekend(day)
            if rng.random() * self._hour_max[weekend] < self._hour_weights[weekend][minute // MINUTES_PER_HOUR]:
                break
        return candidate

    def _service_delay(self, destination: str, ready: int, rng: random.Random, pressure: bool = False) -> int:
        median, sigma, low, high = _DELAY_LENGTH_HOURS[destination]
        hours = rng.lognormvariate(math.log(median), sigma)
        if destination in _SOCIAL_CARE_DESTINATIONS:
            if self._cal.month(ready // MINUTES_PER_DAY) in _WINTER_MONTHS:
                hours *= _WINTER_SOCIAL_CARE_FACTOR
            if pressure:
                hours *= _PRESSURE_SOCIAL_CARE_FACTOR
        return int(round(min(max(hours, low), high) * MINUTES_PER_HOUR))

    def _next_monday(self, day: int) -> int:
        """Day index of the Monday after weekend day ``day``."""
        return day + 7 - self.dow(day)

    def _defer_weekend_review(self, ready: int, rng: random.Random) -> int:
        """A weekend readiness may only be confirmed on a Monday-Wednesday ward round."""
        day = ready // MINUTES_PER_DAY
        if rng.random() >= self._review_defer_p[self.dow(day)]:
            return ready
        review_day = self._next_monday(day) + self._backlog.draw(rng)
        return review_day * MINUTES_PER_DAY + _WARD_ROUND_START_MIN + rng.randrange(_WARD_ROUND_SPREAD_MIN)

    def _defer_weekend_service(self, depart: int, rng: random.Random) -> tuple[int, bool]:
        """A weekend departure may wait for Monday's services (pharmacy, transport, paperwork)."""
        day = depart // MINUTES_PER_DAY
        if rng.random() >= self._service_defer_p[self.dow(day)]:
            return depart, False
        return self._slot_on(self._next_monday(day), rng), True

    def _reason(self, destination: str, gap: int, service_delay: bool, weekend_deferred: bool, rng: random.Random) -> str:
        if gap <= DELAY_THRESHOLD_MIN:
            return "none"
        if weekend_deferred and not service_delay:
            return self._weekend_reasons[destination].draw(rng)
        table = self._long_reasons if gap >= _LONG_DELAY_MIN else self._short_reasons
        return table[destination].draw(rng)

    # -- public ----------------------------------------------------------------
    def friday_push(self, admit: int, ready: int, rng: random.Random) -> int:
        """Pre-weekend discharge planning: a readiness due on Sat/Sun may move to Friday afternoon.

        Only patients who will have spent at least a day in hospital by then are brought forward.
        """
        day = ready // MINUTES_PER_DAY
        p_push = self._push_p[self.dow(day)]
        if p_push <= 0.0 or rng.random() >= p_push:
            return ready
        friday = day - (self.dow(day) - 4)
        pushed = friday * MINUTES_PER_DAY + _FRIDAY_PUSH_ROUND_MIN + rng.randrange(_FRIDAY_PUSH_SPREAD_MIN)
        if pushed < admit + _FRIDAY_PUSH_MIN_STAY_MIN or pushed >= ready:
            return ready
        return pushed

    def plan_discharge(self, ward: int, ready: int, rng: random.Random, pressure: bool = False,
                       elective: bool = False) -> tuple[str, int, int, str]:
        """Return ``(destination, effective_ready, departure, delay_reason)`` (minutes).

        ``pressure`` (the ward has patients waiting for a bed) expedites discharges: the
        service-delay probability is multiplied by ``_PRESSURE_DELAY_FACTOR`` and social-care
        waits by ``_PRESSURE_SOCIAL_CARE_FACTOR``. Electives (``elective``) have planned
        discharges: their delay probability is multiplied by ``_ELECTIVE_DELAY_FACTOR``. The
        ward factor (``_WARD_DELAY_FACTOR``) applies to every stay of the ward.
        """
        destination = self._destinations[ward].draw(rng)
        ready = self._defer_weekend_review(ready, rng)
        p_delay = _DELAY_PROBABILITY[destination] * self._ward_delay_factor[ward]
        if pressure:
            p_delay *= _PRESSURE_DELAY_FACTOR
        if elective:
            p_delay *= _ELECTIVE_DELAY_FACTOR
        service_delay = rng.random() < p_delay
        if service_delay:
            delay = self._service_delay(destination, ready, rng, pressure)
            depart = self._slot_at_or_after(ready + delay, rng)
        else:
            depart = self._prompt_departure(ready, rng)
        depart, deferred = self._defer_weekend_service(depart, rng)
        reason = self._reason(destination, depart - ready, service_delay, deferred, rng)
        return destination, ready, depart, reason

    def plan_after_failed_transfer(self, ward: int, ready: int, now: int, rng: random.Random) -> tuple[str, int, str]:
        """Discharge a patient whose downstream transfer was abandoned: ``(destination, departure, reason)``."""
        destination = self._destinations[ward].draw(rng)
        depart = self._slot_at_or_after(now + 30 + rng.randrange(150), rng)
        reason = self._reason(destination, depart - ready, True, False, rng)
        return destination, depart, reason


# ---------------------------------------------------------------------------
# Clinical LOS calibration
# ---------------------------------------------------------------------------
def _target_los_days(ward_params: Mapping[str, Any], source: str) -> float:
    """Target mean total LOS (days) for a ward/source (falls back to the ward's other sources)."""
    targets = {k: float(v) for k, v in (ward_params.get("los_mean_days") or {}).items()}
    if source in targets:
        return targets[source]
    if "emergency" in targets:
        return targets["emergency"]
    return float(np.mean(list(targets.values()))) if targets else 3.0


def _calibrate_clinical_los(sim: Mapping[str, Any], cal: _Calendar, planner: _DeparturePlanner) -> dict[tuple[int, int], float]:
    """Mean clinical LOS (minutes) per (ward, source) so that the mean *total* stay hits the target.

    Total stay = clinical LOS + ward-round alignment + departure gap / delays / weekend
    effects. That overhead is estimated by a small Monte Carlo of the departure model (fixed
    seed, so the parameters depend on the settings only) and subtracted from the target;
    a few fixed-point iterations handle its dependence on the LOS itself. Wards planned to
    run well below the quiet-ward threshold (target occupancy <= ``_ALWAYS_QUIET_OCCUPANCY``,
    i.e. paediatrics) rarely face bed pressure, so the next-round stretch is included for
    them. Downstream bed waits of transfers and the remaining occupancy-elastic effects are
    endogenous; the residual is reported by :func:`calibration_summary`.
    """
    hour_weights = list(sim["emergency_hour_weights"])
    ed_hours = _Choice(range(24), hour_weights)
    elective_hours = _Choice.from_mapping(_ELECTIVE_ARRIVAL_HOURS)
    means: dict[tuple[int, int], float] = {}
    for w, ward_id in enumerate(WARD_IDS):
        params = sim["ward_params"].get(ward_id, {})
        p_transfer = float(params.get("transfer_out_prob", 0.0)) if params.get("transfer_targets") else 0.0
        always_quiet = float(params.get("target_occupancy", 1.0)) <= _ALWAYS_QUIET_OCCUPANCY
        for source_name in WARDS[ward_id].admission_sources:
            source = _SOURCE_CODES[source_name]
            sigma = _CLINICAL_LOS_SIGMA[source]
            p_stretch = _QUIET_STRETCH_PROB if always_quiet and source != _ELECTIVE else 0.0
            target = _target_los_days(params, source_name) * MINUTES_PER_DAY
            mean = target
            for _ in range(_LOS_CALIBRATION_ITERATIONS):
                rng = random.Random(_LOS_CALIBRATION_SEED + 10 * w + source)
                overhead = 0.0
                for _ in range(_LOS_CALIBRATION_SAMPLES):
                    day = cal.window_start + rng.randrange(364)
                    if source == _ELECTIVE:
                        hour = elective_hours.draw(rng)
                    elif source == _EMERGENCY:
                        hour = ed_hours.draw(rng)
                    else:
                        hour = rng.randrange(24)
                    admit = day * MINUTES_PER_DAY + hour * MINUTES_PER_HOUR + rng.randrange(MINUTES_PER_HOUR)
                    clinical = _lognormal_minutes(rng, mean, sigma)
                    ready = planner.friday_push(admit, _align_to_ward_round(admit + clinical, rng), rng)
                    if rng.random() < p_transfer:
                        depart = ready + 37 + _ASSUMED_TRANSFER_WAIT_MIN
                    else:
                        if rng.random() < p_stretch:
                            next_day = ready // MINUTES_PER_DAY + 1
                            ready = next_day * MINUTES_PER_DAY + _WARD_ROUND_START_MIN + rng.randrange(_WARD_ROUND_SPREAD_MIN)
                        depart = planner.plan_discharge(w, ready, rng, elective=source == _ELECTIVE)[2]
                    overhead += depart - admit - clinical
                mean = max(target - overhead / _LOS_CALIBRATION_SAMPLES, 0.25 * target)
            means[(w, source)] = mean
    return means


# ---------------------------------------------------------------------------
# Event-loop state
# ---------------------------------------------------------------------------
class _Stay:
    """One ward stay (bed episode); minutes are relative to the simulation origin."""

    __slots__ = (
        "sid", "ward", "source", "request", "admit", "ready", "depart", "destination", "reason",
        "turnaround", "booking_day", "rescheduled", "parent", "state", "transfer_out", "stretched",
        "overflow_after",
    )

    def __init__(self, sid: int, ward: int, source: int, request: int, booking_day: int = 0,
                 rescheduled: int = 0, parent: "_Stay | None" = None) -> None:
        self.sid = sid
        self.ward = ward
        self.source = source
        self.request = request
        self.admit: int | None = None
        self.ready: int | None = None
        self.depart: int | None = None
        self.destination: str | None = None
        self.reason: str | None = None
        self.turnaround = 0
        self.booking_day = booking_day
        self.rescheduled = rescheduled
        self.parent = parent
        self.state = _WAITING
        self.transfer_out = False
        self.stretched = False     # stayed to the next ward round because the ward was quiet
        self.overflow_after = _OVERFLOW_AFTER_MIN   # minutes after the request before outlier placement

    @property
    def awaiting_plan(self) -> bool:
        """In a bed and not yet ready-processed (no discharge plan, no transfer request)."""
        return self.state == _BEDDED and self.depart is None and not self.transfer_out


class _Ward:
    """Mutable bed state of one ward."""

    __slots__ = (
        "index", "info", "occupied", "cleaning", "available", "escalation", "closed",
        "elective_queue", "queue", "waiting", "emergency_waiting", "patients", "short_staffed",
        "high_streak", "low_streak", "escalation_cap", "is_hub", "divert_threshold",
    )

    def __init__(self, index: int, info: WardInfo) -> None:
        self.index = index
        self.info = info
        self.occupied = 0                     # patients in a bed
        self.cleaning = 0                     # vacated beds in turnaround
        self.available = info.bed_capacity    # bed_capacity + escalation - closed (today)
        self.escalation = 0
        self.closed = 0
        self.elective_queue: deque[_Stay] = deque()
        self.queue: deque[_Stay] = deque()   # emergencies + transfers, FIFO (lazy deletion)
        self.waiting = 0                      # live entries in ``queue``
        self.emergency_waiting = 0            # live emergency entries (ED boarders)
        self.patients: set[_Stay] = set()
        self.short_staffed = False
        self.high_streak = 0
        self.low_streak = 0
        self.is_hub = info.ward_id in _ESCALATION_HUB_WARDS
        self.escalation_cap = info.max_escalation_beds if self.is_hub else min(_ESCALATION_OTHER_MAX, info.max_escalation_beds)
        self.divert_threshold = max(3, round(_DIVERT_QUEUE_SHARE * info.bed_capacity))

    @property
    def in_use(self) -> int:
        return self.occupied + self.cleaning

    @property
    def under_pressure(self) -> bool:
        """Patients are waiting for one of this ward's beds."""
        return self.waiting > 0 or bool(self.elective_queue)

    @property
    def reserve_beds(self) -> int:
        """Beds held back for emergencies (bed meeting, outlier placement)."""
        return _EMERGENCY_RESERVE_BEDS[1] if self.waiting > 0 else _EMERGENCY_RESERVE_BEDS[0]

    def has_spare_bed(self) -> bool:
        """A bed is free beyond the outlier reserve (used for outlier placement)."""
        return self.available - self.in_use > _OUTLIER_RESERVE_BEDS


def _update_streaks(high: int, low: int, pressure: float) -> tuple[int, int]:
    high = high + 1 if pressure > _ESCALATION_OPEN_THRESHOLD else 0
    low = low + 1 if pressure < _ESCALATION_CLOSE_THRESHOLD else 0
    return high, low


class _HospitalModel:
    """Discrete-event model of the whole hospital for one run (see module docstring)."""

    def __init__(self, sim: Mapping[str, Any], cal: _Calendar, seed: int) -> None:
        self.sim = sim
        self.cal = cal
        self.seed = seed
        streams = np.random.SeedSequence(seed).spawn(6)
        s_arrivals, s_bookings, s_staffing, s_closures, s_flu, s_process = streams
        self.rng = random.Random(int(s_process.generate_state(1)[0]))
        self._origin_dow = cal.origin_dow

        self.planner = _DeparturePlanner(sim, cal)
        self.clinical_los_mean = _calibrate_clinical_los(sim, cal, planner=self.planner)
        self.wards = [_Ward(i, WARDS[ward_id]) for i, ward_id in enumerate(WARD_IDS)]
        self._elective_wards = [w for w in self.wards if w.info.accepts_electives]
        self._medical_wards = [w for w in self.wards if w.info.ward_id in _MEDICAL_PRESSURE_WARDS]
        self._medical_high = 0
        self._medical_low = 0
        self._transfer_routes = self._build_transfer_routes()
        by_id = {w.info.ward_id: w for w in self.wards}
        self._overflow_targets: list[list[_Ward]] = [
            [by_id[t] for t in _OVERFLOW_WARDS.get(w.info.ward_id, ()) if t in by_id] for w in self.wards
        ]
        self._overflow_sources: list[list[_Ward]] = [
            [s for s in self.wards if w in self._overflow_targets[s.index]] for w in self.wards
        ]
        self._elective_hours = _Choice.from_mapping(_ELECTIVE_ARRIVAL_HOURS)
        self._turnaround_mu = math.log(_TURNAROUND_MEDIAN_MIN)

        # Exogenous schedules.
        self.flu_extra, self.flu_episodes = _flu_surge_profile(cal, sim, np.random.default_rng(s_flu))
        rates = _emergency_rate_matrix(cal, sim, self.flu_extra)
        self.staffing_state, self.nurses_day, self.nurses_night = _staffing_schedule(cal, np.random.default_rng(s_staffing))
        self.desired_closed, self.closure_stats = _closure_schedule(cal, self.staffing_state, np.random.default_rng(s_closures))
        self.bookings = _elective_bookings(cal, sim, _ward_streams(s_bookings))

        # Daily operational log (rows: wards in WARD_IDS order, columns: day index).
        shape = (len(WARD_IDS), cal.horizon)
        self.log_escalation = np.zeros(shape, dtype=np.int64)
        self.log_closed = np.zeros(shape, dtype=np.int64)
        self.log_scheduled = np.zeros(shape, dtype=np.int64)
        self.log_cancelled = np.zeros(shape, dtype=np.int64)
        self.log_boarders = np.zeros(shape, dtype=np.int64)
        self.log_census = np.zeros(shape, dtype=np.int64)
        self.counters = {
            "overflow_admissions": 0, "readiness_brought_forward": 0, "discharge_on_next_round": 0,
            "ambulance_diverts": 0, "ed_diversions": 0, "abandoned_transfers": 0, "late_elective_cancellations": 0,
            "electives_removed_after_max_reschedules": 0, "electives_rebooked_beyond_horizon": 0,
        }

        self.stays: list[_Stay] = []
        self._seq = count()
        self._heap: list[tuple[int, int, int, Any]] = []
        self._schedule_exogenous_events(_emergency_arrivals(cal, sim, rates, _ward_streams(s_arrivals)))
        self._seed_initial_census()
        self.n_events = 0

    # -- setup -------------------------------------------------------------------
    def _build_transfer_routes(self) -> list[tuple[float, _Choice | None]]:
        routes: list[tuple[float, _Choice | None]] = []
        for ward_id in WARD_IDS:
            params = self.sim["ward_params"].get(ward_id, {})
            targets = {
                WARD_IDS.index(t): float(p)
                for t, p in (params.get("transfer_targets") or {}).items()
                if t in WARDS and t != ward_id and WARDS[t].accepts_transfers and float(p) > 0
            }
            probability = float(params.get("transfer_out_prob", 0.0))
            routes.append((probability, _Choice.from_mapping(targets)) if targets and probability > 0 else (0.0, None))
        return routes

    def _new_stay(self, ward: int, source: int, request: int, booking_day: int = 0,
                  rescheduled: int = 0, parent: _Stay | None = None) -> _Stay:
        stay = _Stay(len(self.stays), ward, source, request, booking_day, rescheduled, parent)
        self.stays.append(stay)
        return stay

    def _push(self, t: int, kind: int, payload: Any) -> None:
        heapq.heappush(self._heap, (t, kind, next(self._seq), payload))

    def _schedule_exogenous_events(self, arrivals: list[tuple[int, int, int]]) -> None:
        events: list[tuple[int, int, int, Any]] = []
        for day in range(self.cal.horizon):
            midnight = day * MINUTES_PER_DAY
            events.append((midnight, _EV_DAY_START, next(self._seq), day))
            if self._elective_wards:
                events.append((midnight + 7 * MINUTES_PER_HOUR, _EV_BED_MEETING, next(self._seq), day))
            for headroom, board_round in zip(_PULL_FORWARD_HEADROOM, _BOARD_ROUNDS_MIN):
                events.append((midnight + board_round, _EV_BOARD_ROUND, next(self._seq), headroom))
        for join, ward, request in arrivals:
            events.append((join, _EV_EMERGENCY_ARRIVAL, next(self._seq), (ward, request)))
        heapq.heapify(events)
        self._heap = events

    def _seed_initial_census(self) -> None:
        """Start ~80 % full with emergency patients part-way through their stay."""
        rng = self.rng
        for ward in self.wards:
            for _ in range(int(round(_INITIAL_CENSUS_SHARE * ward.info.bed_capacity))):
                total = self._clinical_los(ward.index, _EMERGENCY)
                elapsed = int(rng.random() * total)
                stay = self._new_stay(ward.index, _EMERGENCY, request=-elapsed - MINUTES_PER_HOUR - 1)
                stay.admit = -elapsed - 1
                stay.state = _BEDDED
                ward.occupied += 1
                ward.patients.add(stay)
                stay.ready = _align_to_ward_round(max(total - elapsed, MINUTES_PER_HOUR), rng)
                self._push(stay.ready, _EV_READY, stay)

    # -- main loop -----------------------------------------------------------------
    def run(self) -> None:
        """Process events until every patient has left (no arrivals after the horizon)."""
        handlers = (
            self._on_day_start, self._on_bed_meeting, self._on_board_round, self._on_turnaround_end,
            self._on_departure, self._on_emergency_arrival, self._on_join_queue, self._on_ready,
            self._on_elective_timeout, self._on_overflow_check,
        )
        heap, pop = self._heap, heapq.heappop
        n_events = 0
        while heap:
            t, kind, _, payload = pop(heap)
            handlers[kind](t, payload)
            n_events += 1
        self.n_events = n_events

    # -- daily cycle ---------------------------------------------------------------
    def _on_day_start(self, t: int, day: int) -> None:
        for ward in self.wards:
            if day > 0:
                self.log_boarders[ward.index, day - 1] = ward.emergency_waiting
            self.log_census[ward.index, day] = ward.occupied
        self._apply_safety_valves(t)
        open_hubs, close_hubs = self._medical_escalation_decision()
        for ward in self.wards:
            if ward.is_hub:
                opening, closing = (_ESCALATION_HUB_STEP if open_hubs else 0), close_hubs
            else:
                opening, closing = self._own_escalation_decision(ward)
            self._set_capacity(ward, day, opening, closing)
            ward.short_staffed = bool(self.staffing_state[ward.index, day] == 0)
            self.log_escalation[ward.index, day] = ward.escalation
            self.log_closed[ward.index, day] = ward.closed
            self._admit_waiting(ward, t)

    def _medical_escalation_decision(self) -> tuple[bool, bool]:
        """Hospital medical occupancy (census over available beds of the acute/medical wards)."""
        census = sum(w.occupied for w in self._medical_wards)
        beds = sum(w.available for w in self._medical_wards)
        self._medical_high, self._medical_low = _update_streaks(self._medical_high, self._medical_low, census / max(beds, 1))
        return self._medical_high >= _ESCALATION_OPEN_MORNINGS, self._medical_low >= _ESCALATION_CLOSE_MORNINGS

    def _own_escalation_decision(self, ward: _Ward) -> tuple[int, bool]:
        """Wards outside the medical hub flex (max 2 surge beds) on their own bed demand.

        Demand counts patients already waiting for the ward, as specialty units (ICU,
        cardiology, surgery) open a surge bed when patients are queuing for them.
        """
        pressure = (ward.occupied + ward.waiting) / max(ward.available, 1)
        ward.high_streak, ward.low_streak = _update_streaks(ward.high_streak, ward.low_streak, pressure)
        opening = 1 if ward.high_streak >= _ESCALATION_OPEN_MORNINGS else 0
        return opening, ward.low_streak >= _ESCALATION_CLOSE_MORNINGS

    def _set_capacity(self, ward: _Ward, day: int, opening: int, closing: bool) -> None:
        """Apply escalation and closures for the new day, touching only *free* beds.

        Afterwards ``available >= occupied + cleaning``, so the census can never exceed the
        available beds and ``census_start <= bed_capacity + escalation_beds_open``.
        """
        in_use = ward.in_use
        escalation = ward.escalation
        if opening > 0:
            escalation = min(ward.escalation_cap, escalation + opening)
        elif closing and escalation > 0:
            free_physical = ward.info.bed_capacity + escalation - in_use
            escalation -= min(escalation, max(0, free_physical))
        physical = ward.info.bed_capacity + escalation
        desired = int(self.desired_closed[ward.index, day])
        ward.escalation = escalation
        ward.closed = max(0, min(desired, physical - in_use, physical - 1))
        ward.available = physical - ward.closed

    def _apply_safety_valves(self, t: int) -> None:
        """Divert ED patients waiting >= 48 h; abandon transfers waiting >= 60 h (rare)."""
        divert_before = t - _ED_DIVERSION_HOURS * MINUTES_PER_HOUR
        abandon_before = t - _TRANSFER_ABANDON_HOURS * MINUTES_PER_HOUR
        for ward in self.wards:
            if not ward.waiting:
                continue
            for stay in ward.queue:
                if stay.state != _WAITING:
                    continue
                if stay.source == _EMERGENCY and stay.request <= divert_before:
                    stay.state = _CANCELLED
                    ward.waiting -= 1
                    ward.emergency_waiting -= 1
                    self.counters["ed_diversions"] += 1
                elif stay.source == _TRANSFER and stay.request <= abandon_before:
                    stay.state = _CANCELLED
                    ward.waiting -= 1
                    self.counters["abandoned_transfers"] += 1
                    self._discharge_instead_of_transfer(stay.parent, t)

    def _discharge_instead_of_transfer(self, stay: _Stay, t: int) -> None:
        destination, depart, reason = self.planner.plan_after_failed_transfer(stay.ward, stay.ready, t, self.rng)
        stay.transfer_out = False
        stay.destination, stay.depart, stay.reason = destination, depart, reason
        self._push(depart, _EV_DEPARTURE, stay)

    # -- electives -----------------------------------------------------------------
    def _on_bed_meeting(self, t: int, day: int) -> None:
        """07:00 bed meeting: confirm or cancel today's elective bookings."""
        rng = self.rng
        for ward in self._elective_wards:
            bookings = self.bookings[ward.index].pop(day, None)
            if not bookings:
                continue
            self.log_scheduled[ward.index, day] = len(bookings)
            bookings.sort(key=lambda booking: -booking[1])     # protect the most-postponed
            accepted: list[list[int]] = []
            cancelled: list[list[int]] = []
            for booking in bookings:
                (cancelled if rng.random() < _PATIENT_CANCELLATION_PROB else accepted).append(booking)
            capacity = self._elective_capacity(ward, t)
            if ward.short_staffed:
                capacity = int(capacity * _SHORT_STAFF_ELECTIVE_FACTOR)
            if len(accepted) > capacity:
                cancelled.extend(accepted[capacity:])
                accepted = accepted[:capacity]
            self.log_cancelled[ward.index, day] += len(cancelled)
            for booking in cancelled:
                self._rebook(ward, day, booking)
            cutoff = day * MINUTES_PER_DAY + _ELECTIVE_CUTOFF_MIN
            for booking_day, rescheduled in accepted:
                arrival = day * MINUTES_PER_DAY + self._elective_hours.draw(rng) * MINUTES_PER_HOUR + rng.randrange(MINUTES_PER_HOUR)
                stay = self._new_stay(ward.index, _ELECTIVE, arrival, booking_day, rescheduled)
                checkin = int(round(rng.triangular(*_ELECTIVE_CHECKIN_HOURS) * MINUTES_PER_HOUR))
                self._push(arrival + checkin, _EV_JOIN_QUEUE, stay)
                self._push(cutoff, _EV_ELECTIVE_TIMEOUT, stay)

    def _elective_capacity(self, ward: _Ward, t: int) -> int:
        """Electives that can be bedded today.

        Projected free beds = beds not occupied (beds being cleaned will be ready) + expected
        departures by early afternoon - beds reserved for emergencies (1-2), plus a small risk
        allowance (``_ELECTIVE_RISK_BEDS``); electives still unbedded at 14:59 are cancelled.
        """
        free_now = ward.available - ward.occupied
        planned_cutoff = t + _PLANNED_DEPARTURE_CUTOFF_MIN
        ready_cutoff = t + _EXPECTED_READY_CUTOFF_MIN
        expected = 0.0
        for patient in ward.patients:
            if patient.depart is not None:
                if patient.depart <= planned_cutoff:
                    expected += 1.0
            elif patient.transfer_out:
                expected += _AWAITING_TRANSFER_WEIGHT
            elif patient.ready is not None and patient.ready <= ready_cutoff:
                expected += _EXPECTED_READY_WEIGHT
        # Queued emergencies/transfers are not subtracted: booked electives are bedded first.
        return max(0, int(free_now + expected - ward.reserve_beds)) + _ELECTIVE_RISK_BEDS

    # -- occupancy-elastic discharging -------------------------------------------------
    def _on_board_round(self, t: int, headroom: int) -> None:
        """Under bed pressure, bring forward patients due to be ready within a day.

        ``headroom`` is the number of free beds the round aims to keep beyond the queue.
        """
        rng = self.rng
        for ward in self.wards:
            free = ward.available - ward.in_use
            target = headroom if ward.info.accepts_electives else max(headroom, _EVENING_TAKE_HEADROOM)
            shortfall = ward.waiting + len(ward.elective_queue) + target - free
            if shortfall <= 0:
                continue
            horizon = t + _PULL_FORWARD_WINDOW_MIN
            candidates = sorted(
                (s for s in ward.patients if s.awaiting_plan and t < s.ready <= horizon),
                key=lambda s: (s.ready, s.sid),
            )
            pulled = 0
            for stay in candidates:
                if pulled >= shortfall:
                    break
                if rng.random() >= _PULL_FORWARD_PROB:
                    continue
                new_ready = max(t + rng.randrange(3 * MINUTES_PER_HOUR), stay.admit + _MIN_CLINICAL_LOS_MIN)
                if new_ready < stay.ready:
                    stay.ready = new_ready
                    self._push(new_ready, _EV_READY, stay)   # the old READY event becomes stale
                    pulled += 1
            self.counters["readiness_brought_forward"] += pulled

    def _rebook(self, ward: _Ward, day: int, booking: list[int]) -> None:
        """Rebook a cancelled elective 7-28 days later (at most 3 times).

        The patient goes into the next available list slot: the weekday in the window with
        the fewest bookings so far (ties broken at random).
        """
        booking_day, rescheduled = booking
        if rescheduled >= _MAX_RESCHEDULES:
            self.counters["electives_removed_after_max_reschedules"] += 1
            return
        book = self.bookings[ward.index]
        first, last = day + _REBOOK_DAYS[0], min(day + _REBOOK_DAYS[1], self.cal.horizon - 1)
        candidates = [d for d in range(first, last + 1) if (self._origin_dow + d) % 7 < 5]
        if not candidates:
            self.counters["electives_rebooked_beyond_horizon"] += 1
            return
        fewest = min(len(book.get(d, ())) for d in candidates)
        options = [d for d in candidates if len(book.get(d, ())) == fewest]
        new_day = options[self.rng.randrange(len(options))]
        book.setdefault(new_day, []).append([booking_day, rescheduled + 1])

    def _on_elective_timeout(self, t: int, stay: _Stay) -> None:
        if stay.state != _WAITING:
            return
        stay.state = _CANCELLED
        day = stay.request // MINUTES_PER_DAY
        self.log_cancelled[stay.ward, day] += 1
        self.counters["late_elective_cancellations"] += 1
        self._rebook(self.wards[stay.ward], day, [stay.booking_day, stay.rescheduled])

    # -- patient flow --------------------------------------------------------------
    def _on_emergency_arrival(self, t: int, payload: tuple[int, int]) -> None:
        ward_index, request = payload
        ward = self.wards[ward_index]
        if ward.emergency_waiting >= ward.divert_threshold and self.rng.random() < _DIVERT_PROB:
            self.counters["ambulance_diverts"] += 1     # sent to a neighbouring hospital
            return
        self._enqueue(ward, self._new_stay(ward_index, _EMERGENCY, request), t)

    def _on_join_queue(self, t: int, stay: _Stay) -> None:
        if stay.state == _WAITING:
            self._enqueue(self.wards[stay.ward], stay, t)

    def _enqueue(self, ward: _Ward, stay: _Stay, t: int) -> None:
        if stay.source == _ELECTIVE:
            ward.elective_queue.append(stay)
        else:
            ward.queue.append(stay)
            ward.waiting += 1
            if stay.source == _EMERGENCY:
                ward.emergency_waiting += 1
        self._admit_waiting(ward, t)
        if stay.state == _WAITING and stay.source != _ELECTIVE and self._overflow_targets[ward.index]:
            stay.overflow_after = _OVERFLOW_AFTER_MIN + self.rng.randrange(_OVERFLOW_JITTER_MIN + 1)
            self._push(max(t, stay.request + stay.overflow_after), _EV_OVERFLOW_CHECK, ward.index)

    def _admit_waiting(self, ward: _Ward, t: int) -> None:
        """Bed waiting patients while ``occupied + cleaning < available``.

        The ward's own queue comes first; beds spare beyond the outlier reserve then take
        eligible outliers from the queues of wards that overflow into this one.
        """
        while ward.occupied + ward.cleaning < ward.available:
            stay = self._pop_next(ward)
            if stay is None:
                break
            self._admit(ward, stay, t)
        for source in self._overflow_sources[ward.index]:
            if not ward.has_spare_bed():
                return
            if source.waiting:
                self._take_outliers(source, ward, t)

    def _overflow_eligible(self, stay: _Stay, target: _Ward, t: int) -> bool:
        return (
            stay.state == _WAITING
            and stay.source != _ELECTIVE
            and t - stay.request >= stay.overflow_after
            and (stay.parent is None or stay.parent.ward != target.index)
        )

    def _take_outliers(self, source: _Ward, target: _Ward, t: int) -> None:
        """Move long-waiting patients from ``source``'s queue into free beds of ``target``."""
        for stay in source.queue:
            if not target.has_spare_bed():
                return
            if stay.state != _WAITING:
                continue
            if t - stay.request < _OVERFLOW_AFTER_MIN:
                return                               # queue is in request order (approximately)
            if t - stay.request < stay.overflow_after:
                continue                             # this patient's breach is not acted on yet
            if self._overflow_eligible(stay, target, t):
                self._overflow(source, stay, target, t)

    def _on_overflow_check(self, t: int, ward_index: int) -> None:
        """A queued patient reached the outlier threshold: look for a free bed elsewhere."""
        source = self.wards[ward_index]
        for target in sorted(self._overflow_targets[ward_index], key=lambda w: w.in_use - w.available):
            if source.waiting == 0:
                return
            if target.has_spare_bed() and not target.under_pressure:
                self._take_outliers(source, target, t)

    def _overflow(self, source: _Ward, stay: _Stay, target: _Ward, t: int) -> None:
        source.waiting -= 1
        if stay.source == _EMERGENCY:
            source.emergency_waiting -= 1
        stay.ward = target.index                     # becomes a stay on the receiving ward
        self.counters["overflow_admissions"] += 1
        self._admit(target, stay, t)

    @staticmethod
    def _pop_next(ward: _Ward) -> _Stay | None:
        """Next live patient: booked electives first, then emergencies/transfers FIFO."""
        while ward.elective_queue:
            stay = ward.elective_queue.popleft()
            if stay.state == _WAITING:
                return stay
        while ward.queue:
            stay = ward.queue.popleft()
            if stay.state == _WAITING:
                ward.waiting -= 1
                if stay.source == _EMERGENCY:
                    ward.emergency_waiting -= 1
                return stay
        return None

    def _admit(self, ward: _Ward, stay: _Stay, t: int) -> None:
        stay.admit = t
        stay.state = _BEDDED
        ward.occupied += 1
        ward.patients.add(stay)
        if stay.source == _TRANSFER:
            self._depart(stay.parent, t)          # leaves the sending ward at the same minute
        ready = _align_to_ward_round(t + self._clinical_los(ward.index, stay.source), self.rng)
        stay.ready = self.planner.friday_push(t, ready, self.rng)
        self._push(stay.ready, _EV_READY, stay)

    def _clinical_los(self, ward: int, source: int) -> int:
        return _lognormal_minutes(self.rng, self.clinical_los_mean[(ward, source)], _CLINICAL_LOS_SIGMA[source])

    def _on_ready(self, t: int, stay: _Stay) -> None:
        """Clinically ready: request a downstream bed or plan the discharge."""
        if not stay.awaiting_plan or stay.ready != t:
            return                                    # stale: readiness was brought forward
        rng = self.rng
        ward = self.wards[stay.ward]
        probability, targets = self._transfer_routes[stay.ward]
        if not stay.stretched and targets is not None and rng.random() < probability:
            stay.transfer_out = True
            stay.destination = "internal_ward"
            child = self._new_stay(targets.draw(rng), _TRANSFER, t, parent=stay)
            handover = int(round(rng.uniform(*_TRANSFER_HANDOVER_HOURS) * MINUTES_PER_HOUR))
            self._push(t + handover, _EV_JOIN_QUEUE, child)
            return
        quiet = not ward.under_pressure and ward.occupied < _QUIET_OCCUPANCY * ward.available
        if quiet and not stay.stretched and stay.source != _ELECTIVE and rng.random() < _QUIET_STRETCH_PROB:
            stay.stretched = True                     # no pressure: discharge on the next round
            self.counters["discharge_on_next_round"] += 1
            next_day = t // MINUTES_PER_DAY + 1
            stay.ready = next_day * MINUTES_PER_DAY + _WARD_ROUND_START_MIN + rng.randrange(_WARD_ROUND_SPREAD_MIN)
            self._push(stay.ready, _EV_READY, stay)
            return
        destination, ready, depart, reason = self.planner.plan_discharge(
            stay.ward, t, rng, pressure=ward.under_pressure, elective=stay.source == _ELECTIVE,
        )
        stay.destination, stay.ready, stay.depart, stay.reason = destination, ready, depart, reason
        self._push(depart, _EV_DEPARTURE, stay)

    def _on_departure(self, t: int, stay: _Stay) -> None:
        self._depart(stay, t)

    def _depart(self, stay: _Stay, t: int) -> None:
        ward = self.wards[stay.ward]
        ward.occupied -= 1
        ward.patients.discard(stay)
        stay.depart = t
        stay.state = _DEPARTED
        if stay.transfer_out:
            stay.reason = _DOWNSTREAM_DELAY_REASON if t - stay.ready > DELAY_THRESHOLD_MIN else "none"
        stay.turnaround = self._turnaround(ward, t)
        ward.cleaning += 1
        self._push(t + stay.turnaround, _EV_TURNAROUND_END, ward.index)

    def _on_turnaround_end(self, t: int, ward_index: int) -> None:
        ward = self.wards[ward_index]
        ward.cleaning -= 1
        self._admit_waiting(ward, t)

    def _turnaround(self, ward: _Ward, t: int) -> int:
        """Minutes to clean and prepare the vacated bed."""
        rng = self.rng
        if rng.random() < _DEEP_CLEAN_PROB:
            minutes = rng.uniform(*_DEEP_CLEAN_RANGE_MIN)
        else:
            minutes = rng.lognormvariate(self._turnaround_mu, _TURNAROUND_SIGMA)
        day, minute = divmod(t, MINUTES_PER_DAY)
        hour = minute // MINUTES_PER_HOUR
        if hour >= 20 or hour < 7:
            minutes *= _TURNAROUND_NIGHT_FACTOR
        if (self._origin_dow + day) % 7 >= 5:
            minutes *= _TURNAROUND_WEEKEND_FACTOR
        if ward.short_staffed:
            minutes *= _TURNAROUND_SHORT_STAFF_FACTOR
        low, high = _TURNAROUND_CLIP_MIN
        return int(round(min(max(minutes, low), high)))


# ---------------------------------------------------------------------------
# Table assembly (post-run): stays -> patient_stays, stays + daily log -> ward_daily
# ---------------------------------------------------------------------------
def _stay_arrays(stays: Sequence[_Stay]) -> dict[str, np.ndarray]:
    """Column arrays of every admitted stay (all of them have left by the end of the run)."""
    admitted = [s for s in stays if s.admit is not None]
    unfinished = sum(1 for s in admitted if s.depart is None)
    if unfinished:
        raise RuntimeError(f"Simulation ended with {unfinished} patients still in a bed")
    n = len(admitted)

    def ints(attr: str) -> np.ndarray:
        return np.fromiter((getattr(s, attr) for s in admitted), dtype=np.int64, count=n)

    arrays = {name: ints(name) for name in ("sid", "ward", "source", "request", "admit", "ready", "depart",
                                              "turnaround", "booking_day", "rescheduled")}
    arrays["transfer_out"] = np.fromiter((s.transfer_out for s in admitted), dtype=bool, count=n)
    arrays["destination"] = np.array([s.destination for s in admitted], dtype=object)
    arrays["reason"] = np.array([s.reason for s in admitted], dtype=object)
    return arrays


def _ward_day_counts(ward: np.ndarray, day: np.ndarray, n_wards: int, n_days: int,
                     mask: np.ndarray | None = None, weights: np.ndarray | None = None) -> np.ndarray:
    """``[ward, day]`` matrix of counts (or weighted sums) of events."""
    keep = (day >= 0) & (day < n_days)
    if mask is not None:
        keep &= mask
    flat = ward[keep] * n_days + day[keep]
    w = None if weights is None else weights[keep]
    return np.bincount(flat, weights=w, minlength=n_wards * n_days).reshape(n_wards, n_days)


def _interval_counts(ward: np.ndarray, first: np.ndarray, stop: np.ndarray, n_wards: int, n_days: int) -> np.ndarray:
    """Per ward and day, how many intervals ``[first, stop)`` (day indices) cover the day."""
    first = np.clip(first, 0, n_days)
    stop = np.clip(stop, 0, n_days)
    valid = first < stop
    width = n_days + 1
    diff = np.bincount(ward[valid] * width + first[valid], minlength=n_wards * width)
    diff = diff - np.bincount(ward[valid] * width + stop[valid], minlength=n_wards * width)
    return np.cumsum(diff.reshape(n_wards, width), axis=1)[:, :n_days]


def _trailing_mean(total: np.ndarray, n: np.ndarray, window: int, initial: np.ndarray) -> np.ndarray:
    """Trailing ``window``-day mean per ward; carried forward on days without events."""
    n_days = total.shape[1]
    zero = np.zeros((total.shape[0], 1))
    cs_total = np.concatenate([zero, np.cumsum(total, axis=1)], axis=1)
    cs_n = np.concatenate([zero, np.cumsum(n, axis=1)], axis=1)
    idx = np.arange(n_days)
    low = np.maximum(idx - window + 1, 0)
    window_total = cs_total[:, idx + 1] - cs_total[:, low]
    window_n = cs_n[:, idx + 1] - cs_n[:, low]
    mean = np.where(window_n > 0, window_total / np.maximum(window_n, 1), np.nan)
    filled = pd.DataFrame(mean).ffill(axis=1)
    for row, value in enumerate(initial):
        filled.iloc[row] = filled.iloc[row].fillna(value)
    return filled.to_numpy()


def _assemble_tables(model: _HospitalModel) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the clean raw-format tables and verify the model's internal invariants."""
    cal, sim = model.cal, model.sim
    a = _stay_arrays(model.stays)
    n_wards = len(WARD_IDS)
    n_days = int(a["depart"].max() // MINUTES_PER_DAY) + 2
    admit_day = a["admit"] // MINUTES_PER_DAY
    depart_day = a["depart"] // MINUTES_PER_DAY
    ward = a["ward"]

    # Census at 00:00 of day d: admitted before d and still present at d (depart >= 00:00).
    present_from = admit_day + 1
    present_stop = depart_day + 1
    census = _interval_counts(ward, present_from, present_stop, n_wards, n_days)
    ready_plus_4h = _ceil_div(a["ready"] + DELAY_THRESHOLD_MIN, MINUTES_PER_DAY)
    delayed = _interval_counts(ward, np.maximum(present_from, ready_plus_4h), present_stop, n_wards, n_days)

    def by_admit(mask: np.ndarray | None = None, weights: np.ndarray | None = None) -> np.ndarray:
        return _ward_day_counts(ward, admit_day, n_wards, n_days, mask, weights)

    def by_depart(mask: np.ndarray | None = None, weights: np.ndarray | None = None) -> np.ndarray:
        return _ward_day_counts(ward, depart_day, n_wards, n_days, mask, weights)

    source = a["source"]
    adm_elective = by_admit(source == _ELECTIVE)
    adm_emergency = by_admit(source == _EMERGENCY)
    transfers_in = by_admit(source == _TRANSFER)
    wait_hours = (a["admit"] - a["request"]) / MINUTES_PER_HOUR
    emergency_wait_sum = by_admit(source == _EMERGENCY, wait_hours)
    discharged = ~a["transfer_out"]
    discharges = by_depart(discharged)
    transfers_out = by_depart(a["transfer_out"])
    before_noon = by_depart(discharged & (a["depart"] % MINUTES_PER_DAY < 12 * MINUTES_PER_HOUR))
    departures = by_depart()
    turnaround_sum = by_depart(weights=a["turnaround"].astype(float))
    los_days = (a["depart"] - a["admit"]) / MINUTES_PER_DAY
    los_sum = by_depart(weights=los_days)
    los_initial = np.array([_target_los_days(sim["ward_params"].get(w, {}), "emergency") for w in WARD_IDS])
    avg_turnaround = _trailing_mean(turnaround_sum, departures, 7, np.full(n_wards, _INITIAL_TURNAROUND_MIN))
    avg_los = _trailing_mean(los_sum, departures, 7, los_initial)

    _check_invariants(model, a, census, adm_elective)

    # ---- ward_daily (window days only) ----
    days = cal.window_days()
    grid_w = np.repeat(np.arange(n_wards), len(days))
    grid_d = np.tile(days, n_wards)
    capacity = np.array([WARDS[w].bed_capacity for w in WARD_IDS])[grid_w]
    escalation = model.log_escalation[grid_w, grid_d]
    closed = model.log_closed[grid_w, grid_d]
    n_emergency = adm_emergency[grid_w, grid_d]
    mean_wait = np.where(n_emergency > 0, emergency_wait_sum[grid_w, grid_d] / np.maximum(n_emergency, 1), 0.0)
    ward_ids = pd.Series(np.array(WARD_IDS, dtype=object)[grid_w])
    census_end = census[grid_w, grid_d + 1]
    nurses_day = model.nurses_day[grid_w, grid_d]
    dates = (cal.origin + pd.to_timedelta(grid_d, unit="D")).strftime(DATE_FORMAT)
    ward_daily = pd.DataFrame({
        "date": np.asarray(dates, dtype=object),
        "ward_id": ward_ids.to_numpy(),
        "bed_capacity": capacity,
        "escalation_beds_open": escalation,
        "beds_closed": closed,
        "staffing_level": staffing_levels(pd.Series(nurses_day), ward_ids).to_numpy(),
        "nurses_day_shift": nurses_day,
        "nurses_night_shift": model.nurses_night[grid_w, grid_d],
        "census_start": census[grid_w, grid_d],
        "elective_scheduled": model.log_scheduled[grid_w, grid_d],
        "elective_cancelled": model.log_cancelled[grid_w, grid_d],
        "admissions_elective": adm_elective[grid_w, grid_d],
        "admissions_emergency": n_emergency,
        "transfers_in": transfers_in[grid_w, grid_d],
        "transfers_out": transfers_out[grid_w, grid_d],
        "discharges": discharges[grid_w, grid_d],
        "discharges_before_noon": before_noon[grid_w, grid_d],
        "census_end": census_end,
        "delayed_discharges": delayed[grid_w, grid_d],
        "emergency_boarders": model.log_boarders[grid_w, grid_d],
        "avg_bed_wait_hours": np.round(mean_wait, 2),
        "avg_bed_turnaround_min": np.round(avg_turnaround[grid_w, grid_d], 1),
        "avg_los_days": np.round(avg_los[grid_w, grid_d], 2),
        "occupancy_level": occupancy_levels(
            pd.Series(census_end), pd.Series(capacity + escalation - closed)
        ).to_numpy(),
    })
    int_columns = [c for c in RAW_COLUMNS[WARD_DAILY] if ward_daily[c].dtype.kind in "iu"]
    ward_daily[int_columns] = ward_daily[int_columns].astype(np.int64)
    ward_daily = ward_daily.sort_values(["date", "ward_id"], kind="stable").reset_index(drop=True)
    ward_daily = ward_daily[list(RAW_COLUMNS[WARD_DAILY])]

    # ---- patient_stays (admitted in the window) ----
    window_lo = cal.window_start * MINUTES_PER_DAY
    window_hi = (cal.window_end + 1) * MINUTES_PER_DAY
    keep = np.flatnonzero((a["admit"] >= window_lo) & (a["admit"] < window_hi))
    keep = keep[np.lexsort((a["sid"][keep], a["ward"][keep], a["admit"][keep]))]
    s = {k: v[keep] for k, v in a.items()}
    elective = s["source"] == _ELECTIVE
    gap = s["depart"] - s["ready"]

    def stamp(minutes: np.ndarray) -> np.ndarray:
        return np.asarray((cal.origin + pd.to_timedelta(minutes, unit="m")).strftime(DATETIME_FORMAT), dtype=object)

    patient_stays = pd.DataFrame({
        "stay_id": [f"ST-{i:07d}" for i in range(1, len(keep) + 1)],
        "ward_id": np.array(WARD_IDS, dtype=object)[s["ward"]],
        "admission_source": np.array(_SOURCE_NAMES, dtype=object)[s["source"]],
        "admission_datetime": stamp(s["admit"]),
        "discharge_datetime": stamp(s["depart"]),
        "booking_lead_days": np.where(elective, s["admit"] // MINUTES_PER_DAY - s["booking_day"], 0).astype(np.int64),
        "times_rescheduled": np.where(elective, s["rescheduled"], 0).astype(np.int64),
        "bed_wait_hours": np.round((s["admit"] - s["request"]) / MINUTES_PER_HOUR, 2),
        "separation_type": np.where(s["transfer_out"], "transfer_out", "discharge").astype(object),
        "discharge_destination": s["destination"],
        "delay_reason": s["reason"],
        "discharge_delay_hours": np.where(gap > DELAY_THRESHOLD_MIN, np.round(gap / MINUTES_PER_HOUR, 1), 0.0),
        "bed_turnaround_minutes": s["turnaround"].astype(np.int64),
    })
    return ward_daily, patient_stays[list(RAW_COLUMNS[PATIENT_STAYS])]


def _check_invariants(model: _HospitalModel, a: dict[str, np.ndarray], census: np.ndarray, adm_elective: np.ndarray) -> None:
    """Re-check the model's bookkeeping; any failure is a simulator bug, not bad luck."""
    horizon = model.cal.horizon
    if not np.array_equal(model.log_census, census[:, :horizon]):
        raise RuntimeError("Internal error: event-loop census differs from the stay-level census")
    booked = model.log_scheduled - model.log_cancelled
    if not np.array_equal(booked, adm_elective[:, :horizon]):
        raise RuntimeError("Internal error: elective bookings minus cancellations differ from elective admissions")
    capacity = np.array([WARDS[w].bed_capacity for w in WARD_IDS])[:, None]
    available = capacity + model.log_escalation - model.log_closed
    if (census[:, 1:horizon + 1] > available).any() or (available < 1).any():
        raise RuntimeError("Internal error: midnight census exceeds the available beds")
    if (census[:, :horizon] > capacity + model.log_escalation).any():
        raise RuntimeError("Internal error: census_start exceeds bed_capacity + escalation_beds_open")
    if (census[:, :horizon] > available).any():
        raise RuntimeError("Internal error: an occupied bed was closed (census_start exceeds the available beds)")
    delayed = (a["depart"] - a["ready"]) > DELAY_THRESHOLD_MIN
    if not np.array_equal(delayed, a["reason"] != "none"):
        raise RuntimeError("Internal error: delay reasons inconsistent with departure delays")
    elective = a["source"] == _ELECTIVE
    lead = a["admit"][elective] // MINUTES_PER_DAY - a["booking_day"][elective]
    if (lead < 1 + REBOOK_MIN_DAYS * a["rescheduled"][elective]).any():
        raise RuntimeError("Internal error: a rebooked elective has a booking lead below 1 + 7 x reschedules")


# ---------------------------------------------------------------------------
# Calibration summary
# ---------------------------------------------------------------------------
_OCCUPANCY_TOLERANCE = 0.04
_OCCUPANCY_MIX_TARGETS: dict[str, tuple[float, float]] = {
    "low": (0.10, 0.25), "normal": (0.25, 0.45), "high": (0.25, 0.45), "critical": (0.05, 0.20),
}
_BOARDING_WARDS: tuple[str, ...] = ("AMU", "GMED", "ELDC")
_BOARDING_TARGET = (0.05, 0.20)
_SURGICAL_WARDS: tuple[str, ...] = tuple(w for w in WARD_IDS if WARDS[w].ward_type == "surgical")
_DELAYED_RATE_TARGETS: dict[str, tuple[float, float]] = {
    "ELDC": (0.10, 0.25), "GMED": (0.05, 0.15), "surgical": (0.02, 0.08),
}
_BEFORE_NOON_TARGET = (0.15, 0.30)
_WEEKEND_RATIO_TARGET = (0.50, 0.70)
_CANCELLATION_TARGET = (0.03, 0.10)
_SHORT_STAFFING_TARGET = (0.10, 0.20)
_WAIT_MEDIAN_TARGET = (1.0, 3.0)
_WAIT_P90_WINTER_TARGET = (4.0, 12.0)
_LOS_RELATIVE_TOLERANCE = 0.15
# Realism targets added after review (Week 3): no artificial weekly discharge pattern, no point
# mass in the bed waits and few delayed discharges on the children's ward.
_MONDAY_RATIO_TARGET = (0.85, 1.25)        # Monday discharges / mean Tue-Fri (was 1.41)
_SUNDAY_RATIO_TARGET = (0.60, 1.00)        # Sunday discharges / Saturday (was 1.43)
_WAIT_POINT_MASS_TARGET = (0.0, 0.05)      # largest share of one bed-wait value (was 0.54)
_WAIT_POINT_MASS_MIN_STAYS = 200           # ward x source groups with at least this many stays
_PAED_DELAYED_RATE_TARGET = (0.0, 0.06)    # mean daily delayed-discharge rate on PAED (was 0.11)


def _num(value: Any, digits: int = 4) -> float | None:
    """Plain rounded float (``None`` for missing/NaN)."""
    if value is None:
        return None
    value = float(value)
    return None if math.isnan(value) or math.isinf(value) else round(value, digits)


def _ratio(numerator: float, denominator: float) -> float | None:
    return None if denominator in (0, 0.0) else _num(numerator / denominator)


def _range_check(name: str, value: float | None, bounds: tuple[float, float], description: str) -> dict[str, Any]:
    passed = None if value is None else bool(bounds[0] <= value <= bounds[1])
    return {"name": name, "description": description, "value": value, "target": list(bounds), "passed": passed}


def _wait_point_mass(patient_stays: pd.DataFrame) -> float | None:
    """Largest share any single ``bed_wait_hours`` value holds within a ward x source group."""
    shares = [
        float(group.value_counts(normalize=True).iloc[0])
        for _, group in patient_stays.groupby(["ward_id", "admission_source"])["bed_wait_hours"]
        if len(group) >= _WAIT_POINT_MASS_MIN_STAYS
    ]
    return _num(max(shares)) if shares else None


def calibration_summary(ward_daily: pd.DataFrame, patient_stays: pd.DataFrame,
                        settings: Mapping[str, Any] | None = None) -> dict:
    """Compare a clean raw-format extract with the calibration targets of DESIGN.md §6.1.

    Args:
        ward_daily: clean ``RAW_COLUMNS['ward_daily']`` table (e.g. ``SimulationResult.ward_daily``).
        patient_stays: clean ``RAW_COLUMNS['patient_stays']`` table.
        settings: full settings or the ``simulation`` section (for ward targets); defaults
            to ``config/settings.yaml``.

    Returns:
        JSON-serialisable dict: ``checks`` (one entry per target with ``value``, ``target``
        and ``passed``; ``passed`` is ``None`` when the period cannot evaluate it, e.g. no
        winter days), counts of evaluated/passed targets and supporting breakdowns.
    """
    sim = _simulation_settings(settings)
    wd = ward_daily
    dates = pd.to_datetime(wd["date"], format=DATE_FORMAT)
    month = dates.dt.month
    winter = month.isin(sorted(_WINTER_MONTHS))
    weekend = dates.dt.dayofweek >= 5
    available = wd["bed_capacity"] + wd["escalation_beds_open"] - wd["beds_closed"]
    occupancy = wd["census_end"] / available
    checks: list[dict[str, Any]] = []

    # Occupancy by ward.
    occupancy_by_ward: dict[str, Any] = {}
    for ward_id in WARD_IDS:
        rows = wd["ward_id"] == ward_id
        target = float(sim["ward_params"].get(ward_id, {}).get("target_occupancy", np.nan))
        mean = _num(occupancy[rows].mean()) if rows.any() else None
        occupancy_by_ward[ward_id] = {
            "mean": mean, "target": target,
            "winter_mean": _num(occupancy[rows & winter].mean()) if (rows & winter).any() else None,
            "summer_mean": _num(occupancy[rows & month.isin([6, 7, 8])].mean()) if (rows & month.isin([6, 7, 8])).any() else None,
        }
        passed = None if mean is None or math.isnan(target) else bool(abs(mean - target) <= _OCCUPANCY_TOLERANCE)
        checks.append({
            "name": f"occupancy_{ward_id}", "description": f"{ward_id} mean midnight occupancy within ±{_OCCUPANCY_TOLERANCE} of target",
            "value": mean, "target": [round(target - _OCCUPANCY_TOLERANCE, 4), round(target + _OCCUPANCY_TOLERANCE, 4)],
            "passed": passed,
        })

    # Occupancy band mix.
    mix = wd["occupancy_level"].value_counts(normalize=True)
    mix_shares = {level: _num(mix.get(level, 0.0)) for level in _OCCUPANCY_MIX_TARGETS}
    for level, bounds in _OCCUPANCY_MIX_TARGETS.items():
        checks.append(_range_check(f"occupancy_mix_{level}", mix_shares[level], bounds,
                                   f"share of ward-days in the '{level}' occupancy band"))

    # ED boarding on winter days (AMU/GMED/ELDC).
    boarding_rows = wd["ward_id"].isin(_BOARDING_WARDS) & winter
    boarding = {w: _num((wd.loc[winter & (wd["ward_id"] == w), "emergency_boarders"] > 0).mean())
                for w in _BOARDING_WARDS if (winter & (wd["ward_id"] == w)).any()}
    pooled_boarding = _num((wd.loc[boarding_rows, "emergency_boarders"] > 0).mean()) if boarding_rows.any() else None
    checks.append(_range_check("winter_boarding_days", pooled_boarding, _BOARDING_TARGET,
                               "share of AMU/GMED/ELDC winter ward-days with ED boarders at midnight"))

    # Delayed discharge rate (delayed_discharges / census_start).
    daily_delay_rate = (wd["delayed_discharges"] / wd["census_start"].where(wd["census_start"] > 0)).fillna(0.0)
    delayed_rates = {w: _num(daily_delay_rate[wd["ward_id"] == w].mean()) for w in WARD_IDS if (wd["ward_id"] == w).any()}
    surgical_rows = wd["ward_id"].isin(_SURGICAL_WARDS)
    groups = {"ELDC": wd["ward_id"] == "ELDC", "GMED": wd["ward_id"] == "GMED", "surgical": surgical_rows}
    for group, rows in groups.items():
        value = _num(daily_delay_rate[rows].mean()) if rows.any() else None
        checks.append(_range_check(f"delayed_discharge_rate_{group}", value, _DELAYED_RATE_TARGETS[group],
                                   f"mean daily delayed-discharge rate, {group}"))
    paed_rows = wd["ward_id"] == "PAED"
    paed_rate = _num(daily_delay_rate[paed_rows].mean()) if paed_rows.any() else None
    checks.append(_range_check("delayed_discharge_rate_PAED", paed_rate, _PAED_DELAYED_RATE_TARGET,
                               "mean daily delayed-discharge rate, paediatrics (children rarely wait for social care)"))

    # Discharge timing.
    before_noon = _ratio(wd["discharges_before_noon"].sum(), wd["discharges"].sum())
    checks.append(_range_check("discharges_before_noon_share", before_noon, _BEFORE_NOON_TARGET,
                               "share of discharges completed before 12:00"))
    daily_discharges = wd.groupby("date", sort=False)["discharges"].sum()
    is_weekend_date = pd.to_datetime(daily_discharges.index, format=DATE_FORMAT).dayofweek >= 5
    weekend_ratio = None
    if is_weekend_date.any() and (~is_weekend_date).any():
        weekend_ratio = _ratio(daily_discharges[is_weekend_date].mean(), daily_discharges[~is_weekend_date].mean())
    checks.append(_range_check("weekend_to_weekday_discharge_ratio", weekend_ratio, _WEEKEND_RATIO_TARGET,
                               "mean daily discharges on Sat/Sun relative to Mon-Fri"))
    weekday_means = wd.groupby(dates.dt.dayofweek)["discharges"].mean()
    monday_ratio = sunday_ratio = None
    if set(range(7)) <= set(weekday_means.index):
        monday_ratio = _ratio(weekday_means[0], weekday_means[1:5].mean())
        sunday_ratio = _ratio(weekday_means[6], weekday_means[5])
    checks.append(_range_check("monday_to_weekday_discharge_ratio", monday_ratio, _MONDAY_RATIO_TARGET,
                               "mean Monday discharges relative to Tue-Fri (no artificial weekend-backlog spike)"))
    checks.append(_range_check("sunday_to_saturday_discharge_ratio", sunday_ratio, _SUNDAY_RATIO_TARGET,
                               "mean Sunday discharges relative to Saturday (Sunday is the quietest day)"))

    # Elective cancellations.
    cancellation = _ratio(wd["elective_cancelled"].sum(), wd["elective_scheduled"].sum())
    cancellation_winter = _ratio(wd.loc[winter, "elective_cancelled"].sum(), wd.loc[winter, "elective_scheduled"].sum())
    cancellation_other = _ratio(wd.loc[~winter, "elective_cancelled"].sum(), wd.loc[~winter, "elective_scheduled"].sum())
    checks.append(_range_check("elective_cancellation_rate", cancellation, _CANCELLATION_TARGET,
                               "cancelled / scheduled elective bookings"))
    higher_in_winter = None
    if cancellation_winter is not None and cancellation_other is not None:
        higher_in_winter = bool(cancellation_winter > cancellation_other)
    checks.append({"name": "elective_cancellation_higher_in_winter", "description": "winter cancellation rate exceeds the rest of the year",
                   "value": cancellation_winter, "target": "> non-winter rate", "passed": higher_in_winter})

    # Staffing.
    short = wd["staffing_level"] == "short"
    short_share = _num(short.mean()) if len(wd) else None
    checks.append(_range_check("short_staffing_share", short_share, _SHORT_STAFFING_TARGET,
                               "share of ward-days staffed below requirement"))

    # Emergency bed waits.
    ps = patient_stays
    admitted = pd.to_datetime(ps["admission_datetime"], format=DATETIME_FORMAT)
    discharged = pd.to_datetime(ps["discharge_datetime"], format=DATETIME_FORMAT)
    emergency = ps["admission_source"] == "emergency"
    waits = ps.loc[emergency, "bed_wait_hours"]
    winter_waits = ps.loc[emergency & admitted.dt.month.isin(sorted(_WINTER_MONTHS)), "bed_wait_hours"]
    wait_median = _num(waits.median()) if len(waits) else None
    wait_p90_winter = _num(winter_waits.quantile(0.9)) if len(winter_waits) else None
    checks.append(_range_check("emergency_bed_wait_median_hours", wait_median, _WAIT_MEDIAN_TARGET,
                               "median emergency bed wait (hours)"))
    checks.append(_range_check("emergency_bed_wait_p90_winter_hours", wait_p90_winter, _WAIT_P90_WINTER_TARGET,
                               "90th percentile emergency bed wait in winter (hours)"))
    checks.append(_range_check("bed_wait_point_mass", _wait_point_mass(ps), _WAIT_POINT_MASS_TARGET,
                               "largest share of one bed-wait value within a ward x admission source "
                               f"(groups of {_WAIT_POINT_MASS_MIN_STAYS}+ stays; no artificial spike)"))

    # Supporting breakdowns (not pass/fail targets).
    los = (discharged - admitted).dt.total_seconds() / 86400.0
    los_by_ward: dict[str, Any] = {}
    for ward_id in WARD_IDS:
        targets = sim["ward_params"].get(ward_id, {}).get("los_mean_days") or {}
        entry = {}
        for source_name in ("elective", "emergency", "transfer"):
            rows = (ps["ward_id"] == ward_id) & (ps["admission_source"] == source_name)
            if not rows.any():
                continue
            mean = _num(los[rows].mean(), 3)
            target = targets.get(source_name)
            entry[source_name] = {
                "mean_days": mean, "target_days": target, "stays": int(rows.sum()),
                "within_15pct": None if target is None else bool(abs(mean - float(target)) <= _LOS_RELATIVE_TOLERANCE * float(target)),
            }
        los_by_ward[ward_id] = entry
    delayed_stay = ps["discharge_delay_hours"] > 0
    delay_by_destination = {d: _num(delayed_stay[ps["discharge_destination"] == d].mean())
                            for d in sorted(ps["discharge_destination"].dropna().unique())}
    transfer_share = {w: _num((ps.loc[ps["ward_id"] == w, "separation_type"] == "transfer_out").mean())
                      for w in WARD_IDS if (ps["ward_id"] == w).any()}

    evaluated = [c for c in checks if c["passed"] is not None]
    n_passed = sum(1 for c in evaluated if c["passed"])
    return {
        "period": {"start_date": str(wd["date"].min()) if len(wd) else None,
                   "end_date": str(wd["date"].max()) if len(wd) else None,
                   "days": int(wd["date"].nunique())},
        "rows": {WARD_DAILY: int(len(wd)), PATIENT_STAYS: int(len(ps))},
        "targets_evaluated": len(evaluated),
        "targets_passed": n_passed,
        "all_targets_passed": bool(evaluated) and n_passed == len(evaluated),
        "failed_targets": [c["name"] for c in evaluated if not c["passed"]],
        "checks": checks,
        "occupancy_by_ward": occupancy_by_ward,
        "occupancy_level_mix": mix_shares,
        "winter_boarding_by_ward": boarding,
        "delayed_discharge_rate_by_ward": delayed_rates,
        "elective_cancellation": {"overall": cancellation, "winter": cancellation_winter, "non_winter": cancellation_other},
        "short_staffing": {"overall": short_share, "winter": _num(short[winter].mean()) if winter.any() else None,
                           "non_winter": _num(short[~winter].mean()) if (~winter).any() else None},
        "emergency_bed_wait_hours": {
            "median": wait_median, "p90": _num(waits.quantile(0.9)) if len(waits) else None,
            "winter_median": _num(winter_waits.median()) if len(winter_waits) else None, "winter_p90": wait_p90_winter,
        },
        "weekend_discharges": {"weekend_to_weekday_ratio": weekend_ratio,
                               "weekend_day_share_of_discharges": _ratio(wd.loc[weekend, "discharges"].sum(), wd["discharges"].sum())},
        "length_of_stay_by_ward_source": los_by_ward,
        "delayed_stay_share_by_destination": delay_by_destination,
        "transfer_out_share_by_ward": transfer_share,
        "escalation_ward_day_share": _num((wd["escalation_beds_open"] > 0).mean()) if len(wd) else None,
        "beds_closed_ward_day_share": _num((wd["beds_closed"] > 0).mean()) if len(wd) else None,
        "admission_source_mix": {k: _num(v) for k, v in ps["admission_source"].value_counts(normalize=True).items()},
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def _jsonable(value: Any) -> Any:
    """Recursively convert to JSON-safe Python types (numpy -> python, NaN -> None)."""
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.tolist()]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return None if math.isnan(value) or math.isinf(value) else value
    if isinstance(value, (pd.Timestamp, datetime, Path)):
        return str(value)
    return value


def _diagnostics(model: _HospitalModel, seconds: float) -> dict[str, Any]:
    cal = model.cal
    los_means = {
        WARD_IDS[w]: {} for w in range(len(WARD_IDS))
    }
    for (w, source), minutes in model.clinical_los_mean.items():
        los_means[WARD_IDS[w]][_SOURCE_NAMES[source]] = round(minutes / MINUTES_PER_DAY, 3)
    return _jsonable({
        "seed": model.seed,
        "start_date": cal.start.strftime(DATE_FORMAT),
        "end_date": (cal.origin + pd.Timedelta(days=cal.window_end)).strftime(DATE_FORMAT),
        "warmup_days": cal.window_start,
        "simulated_days_with_daily_cycle": cal.horizon,
        "runtime_seconds": round(seconds, 2),
        "events_processed": model.n_events,
        "stays_simulated": sum(1 for s in model.stays if s.admit is not None),
        "counters": model.counters,
        "closure_episodes": model.closure_stats,
        "flu_surges": model.flu_episodes,
        "clinical_los_mean_days": los_means,
    })


def simulate(start_date: str | None = None, end_date: str | None = None, seed: int | None = None,
             settings: dict | None = None) -> SimulationResult:
    """Run the hospital simulation and return the clean reference extract.

    Args:
        start_date: first output day (``YYYY-MM-DD``); default ``settings.simulation.start_date``.
        end_date: last output day (inclusive); default ``settings.simulation.end_date``.
        seed: random seed (default ``settings.project.seed``). Same seed + settings -> identical output.
        settings: full settings dict or its ``simulation`` section (default: settings.yaml).

    Returns:
        :class:`SimulationResult` with both tables in ``RAW_COLUMNS`` format (clean canonical
        values, ISO date strings), the calibration summary and run diagnostics.
    """
    sim = _simulation_settings(settings)
    start = pd.Timestamp(start_date or sim["start_date"]).normalize()
    end = pd.Timestamp(end_date or sim["end_date"]).normalize()
    if end < start:
        raise ValueError(f"end_date {end.date()} is before start_date {start.date()}")
    seed = get_seed() if seed is None else int(seed)
    started = time.perf_counter()
    cal = _Calendar.build(start, end, int(sim.get("warmup_days", 60)), int(sim.get("cooldown_days", 90)))
    model = _HospitalModel(sim, cal, seed)
    model.run()
    ward_daily, patient_stays = _assemble_tables(model)
    seconds = time.perf_counter() - started
    calibration = calibration_summary(ward_daily, patient_stays, settings=sim)
    diagnostics = _diagnostics(model, seconds)
    logger.info(
        "Simulated %s..%s (seed %d) in %.1f s: %d ward-days, %d stays, %d/%d calibration targets met",
        start.date(), end.date(), seed, seconds, len(ward_daily), len(patient_stays),
        calibration["targets_passed"], calibration["targets_evaluated"],
    )
    return SimulationResult(ward_daily, patient_stays, calibration, diagnostics)


def generate_source_data(raw_dir: Path | None = None, seed: int | None = None, inject_issues: bool = True, *,
                         start_date: str | None = None, end_date: str | None = None,
                         settings: dict | None = None) -> dict:
    """Simulate the reference extract and write the Week-1 raw files.

    Files (under ``raw_dir``, default ``get_path('raw_dir')``):
    ``{table}_raw.csv`` (dirty extract), ``_ground_truth/{table}_true.csv`` (clean),
    ``_ground_truth/injection_log.csv`` and ``simulation_manifest.json``.

    Args:
        raw_dir: output directory.
        seed: simulation seed (default ``settings.project.seed``); the injection stream is
            derived from it, so the dirty files are reproducible too.
        inject_issues: apply :func:`bedsynth.data.quality_issues.inject_quality_issues`
            (also requires ``settings.simulation.data_quality.enabled``); otherwise the raw
            files are clean copies and the log is empty.
        start_date, end_date, settings: forwarded to :func:`simulate`.

    Returns:
        The manifest (also written to ``simulation_manifest.json``).
    """
    from bedsynth import __version__
    from bedsynth.data.quality_issues import LOG_COLUMNS, inject_quality_issues

    started = time.perf_counter()
    sim = _simulation_settings(settings)
    seed = get_seed() if seed is None else int(seed)
    raw_dir = Path(raw_dir) if raw_dir is not None else get_path("raw_dir")
    raw_dir.mkdir(parents=True, exist_ok=True)

    result = simulate(start_date, end_date, seed, settings=sim)
    clean = {WARD_DAILY: result.ward_daily, PATIENT_STAYS: result.patient_stays}
    quality = sim.get("data_quality") or {}
    inject = bool(inject_issues and quality.get("enabled", True))
    rng = np.random.default_rng([seed, 6201])
    dirty: dict[str, pd.DataFrame] = {}
    logs: list[pd.DataFrame] = []
    for table, frame in clean.items():
        if inject:
            dirty[table], log = inject_quality_issues(frame, table, quality.get(table) or {}, rng)
            logs.append(log)
        else:
            dirty[table] = frame.copy()
    log = pd.concat(logs, ignore_index=True) if logs else pd.DataFrame(columns=list(LOG_COLUMNS))

    files: dict[str, str] = {}
    for table in clean:
        files[f"{table}_raw"] = str(write_table(dirty[table], raw_path(table, raw_dir)).relative_to(raw_dir).as_posix())
        files[f"{table}_true"] = str(write_table(clean[table], ground_truth_path(table, raw_dir)).relative_to(raw_dir).as_posix())
    log_path = write_table(log, raw_dir / "_ground_truth" / "injection_log.csv")
    files["injection_log"] = log_path.relative_to(raw_dir).as_posix()
    manifest_path = raw_dir / "simulation_manifest.json"
    files["manifest"] = manifest_path.name

    issue_counts = {
        table: {str(k): int(v) for k, v in log.loc[log["table"] == table, "issue_type"].value_counts().sort_index().items()}
        for table in clean
    }
    manifest = _jsonable({
        "project": "BedSynth AI",
        "generator": "bedsynth.data.simulator.generate_source_data",
        "version": __version__,
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "seed": seed,
        "period": {
            "start_date": result.diagnostics["start_date"],
            "end_date": result.diagnostics["end_date"],
            "days": int(result.ward_daily["date"].nunique()),
            "warmup_days": int(sim.get("warmup_days", 60)),
            "cooldown_days": int(sim.get("cooldown_days", 90)),
        },
        "row_counts": {table: {"clean": int(len(clean[table])), "raw": int(len(dirty[table]))} for table in clean},
        "data_quality_injection": {
            "enabled": inject,
            "total_issues": int(len(log)),
            "issues_by_table": issue_counts,
        },
        "files": files,
        "runtime_seconds": {"simulation": result.diagnostics["runtime_seconds"],
                            "total": round(time.perf_counter() - started, 2)},
        "simulation_diagnostics": result.diagnostics,
        "calibration": result.calibration,
        "settings": sim,
    })
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info("Wrote raw source data to %s (%d injected issues)", raw_dir, len(log))
    return manifest
