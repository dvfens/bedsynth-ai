"""SDV constraint-augmented generation (CAG) for the BedSynth tables - DESIGN.md §7.2.

Division of labour
------------------
BedSynth guarantees the operational validity of every *returned* record with three
complementary mechanisms (DESIGN.md §6.4, §7.2, §8.3):

1. **Constraints inside the SDV synthesizer (this module)** - the model can only produce rows
   that satisfy them. They are organised in six *groups* that an experiment switches on with
   ``synthesis.experiments.<id>.constraints`` (``true`` = every group, ``false`` = none, or a
   list of group names), which makes their effect measurable as an ablation:

   ``inequality`` (SDV built-in :class:`~sdv.cag.Inequality`, ward_daily)
       ``elective_cancelled <= elective_scheduled``, ``discharges_before_noon <= discharges``,
       ``delayed_discharges <= census_start`` (WD-H03/H04/H05). SDV models ``log(high - low + 1)``
       instead of the high column.
   ``fixed_combinations`` (SDV built-in :class:`~sdv.cag.FixedCombinations`, patient_stays)
       ``(separation_type, discharge_destination, delay_reason)`` and ``(ward_id, admission_source)``
       only take combinations observed in the reference data (PS-H09/H10/H11).
   ``regime_labels`` (programmable, :class:`BandLabelConstraint`, ward_daily)
       ``occupancy_level`` and ``staffing_level`` are *recomputed* in ``reverse_transform`` from the
       counts they summarise - ``occupancy_levels(census_end, available_beds)`` with
       ``census_end``/``available_beds`` derived exactly like :func:`bedsynth.schema.derive_ward_daily`,
       and ``staffing_levels(nurses_day_shift, ward)`` - so WD-H10/WD-H11 hold by construction. The
       transform is the identity: the labels stay in the model as latent regime variables (they
       carry the correlation with demand, staffing and boarding), and conditioning on them stays
       efficient.
   ``structural_zeros`` (programmable, :class:`StructuralZerosConstraint`, both tables)
       For declared ``category -> numeric columns`` pairs, ``fit`` learns the category values whose
       numeric columns are *always exactly 0* in the training rows (at least
       :data:`STRUCTURAL_ZERO_MIN_SUPPORT` rows) and ``reverse_transform`` forces those zeros:
       no electives on Sundays, no electives/transfers on wards that take none (WD-H12, global
       models), no booking lead or reschedules for emergencies/transfers (PS-H06). A Gaussian
       copula cannot represent such point masses: it smears them into small positive values.
       (Delay hours are deliberately *not* zeroed by such a repair when no reason is recorded:
       overwriting a jointly sampled delay kept the length of stay drawn with it and inflated the
       LOS of non-delayed stays by ~17 %; ``conditional_marginals`` models the delay per reason.)
   ``capacity`` (programmable, :class:`EscalationCoverConstraint`, ward_daily)
       ``escalation_beds_open`` is re-parameterised as the *cover* the day needs above the base
       beds - ``max(0, max(census_start, census_end) - beds + beds_closed)``, derived like
       :func:`bedsynth.schema.derive_ward_daily` (beds are only closed when free) - plus a learned
       ``extra >= 0`` (the model learns ``extra``; most reference ward-days have none). The census
       then fits the beds available at both ends of the day - WD-H06 ``census_end <=
       available_beds``, WD-H15 ``census_start <= available_beds`` and therefore WD-H07 - by
       construction whenever the cover fits within the ward's observed escalation range (the rest
       is left to rejection). Measured on the reference data (quick models, 5,000 raw rows), the
       cover *improved* the column shapes of gc_tuned (mean 0.945 -> 0.948; occupancy_rate
       0.895 -> 0.956, available_beds 0.943 -> 0.949 without vs with the cover) and its
       occupancy-band mix, while the capacity failures fell from 0.106 / 0.083 / 0.105
       (WD-H06 / H07 / H15, built-in constraints only) to 0.041 / 0 / 0.016 - a
       re-parameterisation, not a distortion.
   ``conditional_marginals`` (programmable, :class:`ConditionalQuantileConstraint`, both tables)
       The *distributional transform*: within groups of rows (e.g. per delay reason, per ward and
       weekday) each declared numeric column is replaced by a standard-normal score of its
       randomised empirical CDF, and mapped back with the group's empirical quantile function.
       Each group's marginal is reproduced exactly - point masses included - while the copula
       keeps modelling how the scores depend on the other columns. Declared for delay hours per
       delay reason (exactly 0 without a reason, the reason's own range otherwise: PS-H07 and
       PS-H13 by construction, and the reason keeps predicting the delay length), elective lists
       and cancellations per ward and weekday (flat Monday-Friday lists instead of the false
       weekday gradient a linear latent link imposes along any weekday encoding; it restores
       ``cancelled <= scheduled`` itself and replaces that ``Inequality``, WD-H03), ED boarders
       and the mean bed wait per ward x occupancy band and closed beds per ward x staffing band
       (``band_groups``: the bands are recomputed from the counts like the band labels, never read
       from the sampled label), the escalation beds beyond the census need per ward, day-shift
       nurses per ward x month (winter short staffing), length of stay per admission weekday (the
       discharge weekday is admission weekday + nights mod 7, which a linear copula cannot learn:
       weekend discharges came out at 29 % instead of 22 %) and the admission/discharge hours, bed
       waits and turnaround of patient_stays per ward x admission source (exact hour-of-day
       profiles and right-skewed waits). Measured on the quick gc_tuned patient_stays model (A/B on
       the reference data, 2,000-row quick evaluation, only these two specs changed): weekend
       discharges 29.2 % -> 23.0 % (reference 22.4 %), composite 0.930 -> 0.964, operational checks
       passed 11/16 -> 16/16, raw hard-valid rate 0.941 -> 0.959. The ward_daily band and month
       groups replaced per-ward marginals of boarders and closed beds and a parametric nurse
       marginal after an A/B on the reference data (full gc_tuned, one evaluation run, only the
       specs changed): Spearman (occupancy, boarders) 0.10 -> 0.44 (reference 0.44), boarding
       share on critical days 0.16 -> 0.46 (reference 0.44) and none on low/normal days (was
       7-9 %), short-staffed share 0.17 -> 0.21 in winter and 0.15-0.16 -> 0.08-0.10 in the other
       seasons (reference 0.24 and 0.08-0.10), closed beds on 56 % of short-staffed days
       (reference 59 %, was 13 %), mean bed wait on critical days 3.0 -> 4.9 h (reference 4.4 h);
       operational checks 14/17 -> 17/17, composite 0.942 -> 0.958.

2. **Derivation** (:func:`bedsynth.schema.derive`) recomputes every derived column, which
   guarantees identities such as the flow balance
   ``census_end = census_start + admissions + transfers_in - transfers_out - discharges``.

3. **Rejection sampling** in :mod:`bedsynth.synthesis.generator` removes rows that still break a
   *hard* rule of :mod:`bedsynth.validation`: ward-days whose census needs more escalation beds
   than the ward ever opened (WD-H06/WD-H15 remainder); same-day stays leaving before they
   arrive (PS-H04); delays longer than the stay (PS-H08); reschedules without the 7-day
   rebooking lead (PS-H14); delay rules for rare reasons that share a pooled marginal (PS-H07,
   PS-H13). Each affects a few % of raw rows or less, and the candidate by-construction fixes
   (re-parameterising same-day departure hours, clipping delays to the stay) would pile mass up
   at a threshold or mix two meanings into one modelled column, i.e. distort the learned
   marginals - rejection keeps the learned distribution intact at a small sampling cost.

Ordering inside SDV
-------------------
SDV 2.2 applies constraint transforms in list order and ``reverse_transform`` in **reverse**
order, after its data processor has rounded and clipped the numeric columns. ``Inequality``
replaces its high column by a ``low#high`` diff column and ``FixedCombinations`` merges its
columns into one ``a#b#c`` column, so programmable constraints that read or overwrite those
columns are listed **first** (:data:`CONSTRAINT_SPECS` order = ``add_constraints`` order): their
``reverse_transform`` then runs last, on fully restored columns. Among them the reverse order is
conditional marginals -> structural zeros -> escalation cover -> band-grouped ED marginals -> band
labels: the zeros and the cover read the restored counts, and the ED boarders / bed waits (listed
right after the band labels) are restored from the occupancy band of the final escalation beds -
the band the occupancy label then gets. Within the conditional marginals the day-shift nurses are
listed after the closed beds, so the closed beds are restored from the staffing band of the
restored nurses. (The escalation cover and the band-grouped marginals read the original census and
flow columns in ``transform`` because they precede the other conditional marginals and the
inequalities in the transform chain; the ward_daily conditional marginals then transform the
cover's ``extra`` beds.)

Segmented models
----------------
Segments drop columns that are constant within the segment (e.g. AMU never books electives)
and the segment column(s). :func:`build_constraints` therefore receives the modelled columns and
the segment's ``constants`` (dropped columns + segment key values): an inequality needs both
columns modelled, a fixed-combination group keeps its modelled columns when at least two remain,
a band label or the escalation cover must be modelled and its inputs modelled or constant
(constants are substituted, e.g. ``elective_scheduled = 0`` on AMU), a structural-zero pair needs
its category column and at least one numeric column modelled, a band-grouped conditional marginal
needs the inputs of its bands modelled or constant. Constraints whose columns are all
constant are skipped - the constants already reproduce them exactly.
"""
from __future__ import annotations

import logging
import zlib
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from sdv.cag import SingleTableProgrammableConstraint

from bedsynth import schema
from bedsynth.schema import PATIENT_STAYS, TABLES, WARD_DAILY, training_columns

LOGGER = logging.getLogger(__name__)

# --- constraint groups -------------------------------------------------------------------
INEQUALITY = "inequality"
FIXED_COMBINATIONS = "fixed_combinations"
REGIME_LABELS = "regime_labels"
STRUCTURAL_ZEROS = "structural_zeros"
CAPACITY = "capacity"
CONDITIONAL_MARGINALS = "conditional_marginals"
CONSTRAINT_GROUPS: tuple[str, ...] = (
    INEQUALITY, FIXED_COMBINATIONS, REGIME_LABELS, STRUCTURAL_ZEROS, CAPACITY, CONDITIONAL_MARGINALS,
)
"""Every constraint group, in canonical order (``constraints: true`` enables all of them)."""
BUILTIN_GROUPS: tuple[str, ...] = (INEQUALITY, FIXED_COMBINATIONS)
"""Groups implemented with SDV's built-in CAG classes."""
PROGRAMMABLE_GROUPS: tuple[str, ...] = (REGIME_LABELS, STRUCTURAL_ZEROS, CAPACITY, CONDITIONAL_MARGINALS)
"""Groups implemented as ``sdv.cag.SingleTableProgrammableConstraint`` subclasses (this module)."""

STRUCTURAL_ZERO_MIN_SUPPORT = 30
"""Training rows a category value needs before its all-zero columns are treated as structural."""
CONDITIONAL_MIN_SUPPORT = 30
"""Training rows a group needs for its own empirical marginal (smaller groups share a pooled one)."""
_PIT_EPSILON = 1e-6
"""Uniform scores are kept inside ``[eps, 1 - eps]`` before the normal quantile function."""

OCCUPANCY_INPUTS: tuple[str, ...] = (
    "ward_id", "escalation_beds_open", "beds_closed", "census_start", "elective_scheduled",
    "elective_cancelled", "admissions_emergency", "transfers_in", "transfers_out", "discharges",
)
"""Train columns that determine ``census_end`` and ``available_beds`` (schema.derive_ward_daily)."""
STAFFING_INPUTS: tuple[str, ...] = ("ward_id", "nurses_day_shift")
"""Train columns that determine the staffing band (schema.staffing_levels)."""
BAND_LABEL_INPUTS: dict[str, tuple[str, ...]] = {
    "occupancy_level": OCCUPANCY_INPUTS,
    "staffing_level": STAFFING_INPUTS,
}
CAPACITY_INPUTS: tuple[str, ...] = OCCUPANCY_INPUTS
"""Train columns that determine the escalation cover (census and capacity of the ward-day)."""
ESCALATION_COLUMN = "escalation_beds_open"

_PROGRAMMABLE_BASE = "sdv.cag.SingleTableProgrammableConstraint"


def constraint_groups(setting: Any) -> tuple[str, ...]:
    """Normalise an experiment's ``constraints`` setting to an ordered tuple of groups.

    Args:
        setting: ``True`` (every group), ``False``/``None`` (none), one group name, or a list
            of group names (``"all"`` / ``"none"`` are accepted as names too).

    Raises:
        ValueError: unknown group names or an unsupported value type.
    """
    if setting is None or isinstance(setting, (bool, np.bool_)):
        return CONSTRAINT_GROUPS if bool(setting) else ()
    if isinstance(setting, str):
        setting = [setting]
    if isinstance(setting, (dict, bytes)) or not isinstance(setting, Iterable):
        raise ValueError(f"constraints must be true, false or a list of groups {list(CONSTRAINT_GROUPS)}, got {setting!r}")
    setting = list(setting)
    names = {str(item).strip().lower() for item in setting}
    if "all" in names:
        return CONSTRAINT_GROUPS
    names.discard("none")
    unknown = sorted(names - set(CONSTRAINT_GROUPS))
    if unknown:
        raise ValueError(f"Unknown constraint group(s) {unknown}. Known groups: {list(CONSTRAINT_GROUPS)}")
    return tuple(group for group in CONSTRAINT_GROUPS if group in names)


@dataclass(frozen=True)
class ConstraintSpec:
    """Declarative description of one constraint.

    Attributes:
        kind: Constraint group (``inequality``, ``fixed_combinations``, ``regime_labels``,
            ``structural_zeros`` or ``capacity``).
        columns: ``inequality``: ``(low, high)``; ``fixed_combinations``: the tied columns;
            ``regime_labels``: ``(label,)``; ``structural_zeros``: the numeric columns;
            ``capacity``: ``("escalation_beds_open",)`` (the re-parameterised column).
        description: Operational wording used in reports and the API.
        rule_ids: Validation rules (``bedsynth.validation``) the constraint enforces (fully or,
            as stated in the description, partly).
        anchor: Column that must be modelled for the constraint to apply at all
            (fixed combinations: optional; structural zeros: the category column).
        inputs: Columns read by a band label or the escalation cover (modelled, or constant
            within the segment).
        group_by: ``conditional_marginals``: categorical columns whose value combinations define
            the groups with their own marginal (columns constant within a segment drop out).
        bounds: ``conditional_marginals``: ``(low, high)`` pairs whose ordering the constraint
            restores; it then replaces SDV's ``Inequality`` for the pair.
        band_groups: ``conditional_marginals``: band labels (keys of :data:`BAND_LABEL_INPUTS`)
            that also define the groups, *recomputed from the counts* like
            :class:`BandLabelConstraint` (never read from the sampled label); their inputs must be
            modelled or constant.
    """

    kind: str
    columns: tuple[str, ...]
    description: str
    rule_ids: tuple[str, ...] = ()
    anchor: str | None = None
    inputs: tuple[str, ...] = ()
    group_by: tuple[str, ...] = ()
    bounds: tuple[tuple[str, str], ...] = ()
    band_groups: tuple[str, ...] = ()

    @property
    def programmable(self) -> bool:
        """Whether the constraint is a BedSynth programmable constraint (not an SDV built-in)."""
        return self.kind in PROGRAMMABLE_GROUPS

    def applicable_columns(
        self, present: Iterable[str], constants: Mapping[str, Any] | None = None
    ) -> tuple[str, ...] | None:
        """Columns of this constraint to use when only ``present`` columns are modelled.

        Args:
            present: Columns modelled by the synthesizer.
            constants: Values of the train columns the (segment) model does not learn.

        Returns:
            The effective columns, or ``None`` when the constraint cannot (or need not) be applied.
        """
        present = set(present)
        constants = constants or {}
        if self.anchor is not None and self.anchor not in present:
            return None
        if self.kind == INEQUALITY:
            return self.columns if all(c in present for c in self.columns) else None
        if self.kind == FIXED_COMBINATIONS:
            kept = tuple(c for c in self.columns if c in present)
            return kept if len(kept) >= 2 else None
        if self.kind in (REGIME_LABELS, CAPACITY):
            if not all(c in present for c in self.columns):
                return None
            if not all(c in present or c in constants for c in self.inputs):
                return None
            return self.columns
        if self.kind in (STRUCTURAL_ZEROS, CONDITIONAL_MARGINALS):
            band_inputs = {c for band in self.band_groups for c in BAND_LABEL_INPUTS[band]}
            if not all(c in present or c in constants for c in band_inputs):
                return None
            kept = tuple(c for c in self.columns if c in present)
            return kept or None
        raise ValueError(f"Unsupported constraint kind '{self.kind}'")


CONSTRAINT_SPECS: dict[str, tuple[ConstraintSpec, ...]] = {
    # Order = SDV add_constraints order (see "Ordering inside SDV" above).
    WARD_DAILY: (
        ConstraintSpec(
            REGIME_LABELS,
            ("occupancy_level",),
            "The occupancy band is recomputed from the flow-balance census_end and the beds "
            "available that day (schema.occupancy_levels), so it always matches the counts.",
            ("WD-H10",),
            inputs=OCCUPANCY_INPUTS,
        ),
        ConstraintSpec(
            REGIME_LABELS,
            ("staffing_level",),
            "The staffing band is recomputed from the day-shift nurses against the ward's nurse "
            "requirement (schema.staffing_levels).",
            ("WD-H11",),
            inputs=STAFFING_INPUTS,
        ),
        ConstraintSpec(
            CONDITIONAL_MARGINALS,
            ("emergency_boarders", "avg_bed_wait_hours"),
            "ED boarders at midnight and the day's mean bed wait follow their empirical distribution "
            "per ward and occupancy band, the band recomputed from the final counts like the "
            "occupancy label: boarders and long waits appear on the full ward-days that have them in "
            "the reference data (no boarders on low or normal days, so WD-S03 holds there).",
            (),
            group_by=("ward_id",),
            band_groups=("occupancy_level",),
        ),
        ConstraintSpec(
            CAPACITY,
            (ESCALATION_COLUMN,),
            "Escalation beds are modelled as the cover the start and end census need above the base "
            "beds net of closed beds, plus a learned extra >= 0, so the census fits the beds available "
            "at both ends of the day (WD-H06, WD-H15, hence WD-H07) whenever that cover is within the "
            "ward's observed escalation range.",
            ("WD-H06", "WD-H07", "WD-H15"),
            inputs=CAPACITY_INPUTS,
        ),
        ConstraintSpec(
            STRUCTURAL_ZEROS,
            ("elective_scheduled", "elective_cancelled", "transfers_in"),
            "Wards that never book electives or receive transfers in the reference data produce "
            "exact zeros (learned per ward).",
            ("WD-H12",),
            anchor="ward_id",
        ),
        ConstraintSpec(
            STRUCTURAL_ZEROS,
            ("elective_scheduled", "elective_cancelled"),
            "Days of the week without elective lists in the reference data (Sunday) produce exact "
            "zeros (learned).",
            (),
            anchor="day_of_week",
        ),
        ConstraintSpec(
            CONDITIONAL_MARGINALS,
            ("elective_scheduled", "elective_cancelled"),
            "Elective lists follow the empirical distribution of their day of the week (and ward): "
            "flat Monday-Friday lists, rare Saturday lists and none on Sundays are reproduced exactly "
            "instead of the false weekday gradient a linear copula imposes along the weekday "
            "encoding; cancellations never exceed the list.",
            ("WD-H03",),
            group_by=("ward_id", "day_of_week"),
            bounds=(("elective_cancelled", "elective_scheduled"),),
        ),
        ConstraintSpec(
            CONDITIONAL_MARGINALS,
            ("beds_closed",),
            "Closed beds follow their empirical distribution per ward and staffing band, the band "
            "recomputed from the day-shift nurses: beds are closed mainly on short-staffed days, and "
            "zero days keep their real share instead of a smeared parametric marginal.",
            (),
            group_by=("ward_id",),
            band_groups=("staffing_level",),
        ),
        ConstraintSpec(
            CONDITIONAL_MARGINALS,
            (ESCALATION_COLUMN,),
            "The escalation beds opened beyond the census need (mostly zero) follow their empirical "
            "distribution per ward, so zero days keep their real share.",
            (),
            group_by=("ward_id",),
        ),
        ConstraintSpec(
            CONDITIONAL_MARGINALS,
            ("nurses_day_shift",),
            "Day-shift nurses follow their empirical distribution per ward and month: the winter "
            "short staffing sits in the lower tail of a few-valued roster, which neither a smooth "
            "parametric marginal nor a linear link to the month encoding reproduces.",
            (),
            group_by=("ward_id", "month"),
        ),
        ConstraintSpec(
            INEQUALITY,
            ("elective_cancelled", "elective_scheduled"),
            "Cancelled electives cannot exceed the electives booked for the day.",
            ("WD-H03",),
        ),
        ConstraintSpec(
            INEQUALITY,
            ("discharges_before_noon", "discharges"),
            "Discharges before noon are a subset of the day's discharges.",
            ("WD-H04",),
        ),
        ConstraintSpec(
            INEQUALITY,
            ("delayed_discharges", "census_start"),
            "Delayed (medically fit) patients are a subset of the midnight census.",
            ("WD-H05",),
        ),
    ),
    PATIENT_STAYS: (
        ConstraintSpec(
            STRUCTURAL_ZEROS,
            ("booking_lead_days", "times_rescheduled"),
            "Emergency and transfer admissions are never booked ahead or rescheduled: exact zeros "
            "(learned per admission source).",
            ("PS-H06",),
            anchor="admission_source",
        ),
        ConstraintSpec(
            CONDITIONAL_MARGINALS,
            ("discharge_delay_hours",),
            "Delay hours follow the empirical distribution of their delay reason: exactly 0 without a "
            "reason, the reason's own range otherwise (e.g. days for a care-facility bed, hours for "
            "transport), so the reason keeps predicting the delay length.",
            ("PS-H07", "PS-H13"),
            group_by=("delay_reason",),
        ),
        ConstraintSpec(
            CONDITIONAL_MARGINALS,
            ("los_nights",),
            "Length of stay follows the empirical distribution of its admission weekday (and ward x "
            "source): the discharge weekday is admission weekday + nights (mod 7), a modular relation a "
            "linear copula cannot learn, so without it discharges spread evenly over the week and there "
            "are too many weekend discharges.",
            (),
            group_by=("admission_day_of_week",),
        ),
        ConstraintSpec(
            CONDITIONAL_MARGINALS,
            ("admission_hour", "bed_wait_hours", "discharge_hour", "bed_turnaround_minutes"),
            "Admission and discharge hours, bed waits and bed turnaround follow their empirical "
            "distribution per ward x admission source: hour-of-day profiles and right-skewed waits are "
            "reproduced exactly instead of a smoothed parametric marginal.",
            (),
            group_by=("ward_id", "admission_source"),
        ),
        ConstraintSpec(
            FIXED_COMBINATIONS,
            ("separation_type", "discharge_destination", "delay_reason"),
            "Separation type, destination and delay reason only appear in combinations seen in "
            "the reference data (transfer out <=> internal ward; reasons allowed per destination).",
            ("PS-H09", "PS-H10"),
        ),
        ConstraintSpec(
            FIXED_COMBINATIONS,
            ("ward_id", "admission_source"),
            "Each ward only admits through the sources it accepts (e.g. AMU: emergencies only).",
            ("PS-H11",),
            anchor="ward_id",
        ),
    ),
}


# =========================================================================================
# Programmable constraints
# =========================================================================================
def _plain(value: Any) -> Any:
    """numpy scalar -> Python scalar (JSON-friendly constraint parameters and learned state)."""
    return value.item() if isinstance(value, np.generic) else value


def _metadata_columns(metadata: Any) -> set[str] | None:
    """Column names of a single-table SDV ``Metadata`` (``None`` if they cannot be read)."""
    try:
        tables = list(metadata.tables.values())
        return set(tables[0].columns) if len(tables) == 1 else None
    except (AttributeError, IndexError):  # pragma: no cover - defensive for other SDV versions
        return None


def _missing_inputs(inputs: Iterable[str], columns: Iterable[str], constants: Mapping[str, Any]) -> list[str]:
    """Inputs that are neither modelled columns nor segment constants."""
    available = set(columns) | set(constants)
    return [c for c in inputs if c not in available]


def _input_column(data: pd.DataFrame, column: str, constants: Mapping[str, Any]) -> pd.Series:
    """A modelled column, or the segment constant broadcast to ``data.index``."""
    if column in data.columns:
        return data[column]
    return pd.Series([constants[column]] * len(data), index=data.index, dtype=object)


def _derived_ward_days(data: pd.DataFrame, constants: Mapping[str, Any]) -> pd.DataFrame:
    """:func:`bedsynth.schema.derive_ward_daily` of the modelled columns plus segment constants.

    Columns that are neither modelled nor constant get neutral placeholders; they never
    influence ``bed_capacity``, ``available_beds`` or ``census_end`` (the callers check that the
    inputs of those values are available).
    """
    frame = {}
    for column in training_columns(WARD_DAILY):
        if column in data.columns:
            frame[column] = data[column].to_numpy()
        elif column in constants:
            value = constants[column]
            frame[column] = np.full(len(data), value, dtype=object if isinstance(value, str) else None)
        else:
            frame[column] = np.zeros(len(data), dtype=float)
    return schema.derive_ward_daily(pd.DataFrame(frame, index=data.index))


def band_labels(data: pd.DataFrame, label_column: str, constants: Mapping[str, Any] | None = None) -> pd.Series:
    """Band (``occupancy_level`` / ``staffing_level``) implied by the counts of every row.

    ``occupancy_levels(census_end, available_beds)`` with both values derived like
    :func:`bedsynth.schema.derive_ward_daily`, or ``staffing_levels(nurses_day_shift, ward_id)``;
    inputs that are not modelled are read from ``constants``. NaN where the band is undefined or
    an input is missing; aligned to ``data.index``.
    """
    constants = constants or {}
    if _missing_inputs(BAND_LABEL_INPUTS[label_column], data.columns, constants):
        return pd.Series(np.nan, index=data.index, dtype=object)
    if label_column == "staffing_level":
        nurses = _input_column(data, "nurses_day_shift", constants)
        wards = _input_column(data, "ward_id", constants).astype(object)
        return schema.staffing_levels(nurses, wards)
    derived = _derived_ward_days(data, constants)
    return schema.occupancy_levels(derived["census_end"], derived["available_beds"])


class BandLabelConstraint(SingleTableProgrammableConstraint):
    """Recompute a ward_daily band label from the counts it summarises (``regime_labels`` group).

    * ``occupancy_level = schema.occupancy_levels(census_end, available_beds)`` where both values
      come from :func:`bedsynth.schema.derive_ward_daily` (flow balance, bed capacity of the ward);
    * ``staffing_level = schema.staffing_levels(nurses_day_shift, ward_id)``.

    ``transform`` is the identity, so the label stays in the model as a latent regime variable;
    ``reverse_transform`` overwrites it with the recomputed band wherever the band is defined
    (rows without a single available bed keep the sampled label: they violate WD-H08 and are left
    to the generator's rejection step, so ``is_valid`` accepts them).

    Args:
        label_column: ``"occupancy_level"`` or ``"staffing_level"``.
        constants: Values of input columns the (segment) model does not learn - the segment key
            (``ward_id``) and columns that are constant within the segment.
    """

    def __init__(self, label_column: str, constants: dict[str, Any] | None = None):
        if label_column not in BAND_LABEL_INPUTS:
            raise ValueError(f"label_column must be one of {sorted(BAND_LABEL_INPUTS)}, got {label_column!r}")
        self.label_column = label_column
        self.constants = {str(k): _plain(v) for k, v in (constants or {}).items()}

    @property
    def inputs(self) -> tuple[str, ...]:
        """Train columns the label is computed from."""
        return BAND_LABEL_INPUTS[self.label_column]

    def _missing_inputs(self, columns: Iterable[str]) -> list[str]:
        return _missing_inputs(self.inputs, columns, self.constants)

    def validate(self, metadata: Any) -> None:
        """The label and every input that is not a constant must be modelled columns."""
        columns = _metadata_columns(metadata)
        if columns is None:
            return
        missing = ([] if self.label_column in columns else [self.label_column]) + self._missing_inputs(columns)
        if missing:
            raise ValueError(
                f"BandLabelConstraint({self.label_column}) needs the label and its inputs; "
                f"missing from the metadata (and not constant): {missing}"
            )

    def fit(self, data: pd.DataFrame, metadata: Any) -> None:
        """Nothing is learned; records how many training labels already match (diagnostic)."""
        missing = self._missing_inputs(data.columns)
        if missing:
            raise ValueError(f"BandLabelConstraint({self.label_column}): inputs {missing} are neither modelled nor constant")
        expected = self.expected_labels(data)
        defined = expected.notna()
        matches = data[self.label_column].astype(object).eq(expected) | ~defined
        self.training_rows = int(len(data))
        self.training_consistency = float(matches.mean()) if len(data) else None

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Identity: the label stays in the model as a latent regime variable."""
        return data

    def get_updated_metadata(self, metadata: Any) -> Any:
        """The transform does not change the columns."""
        return metadata

    def reverse_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Overwrite the label with the band recomputed from the (restored) counts."""
        expected = self.expected_labels(data)
        defined = expected.notna().to_numpy(dtype=bool)
        if not defined.any():
            return data
        out = data.copy()
        labels = out[self.label_column].astype(object).to_numpy(copy=True)
        labels[defined] = expected.to_numpy(dtype=object)[defined]
        out[self.label_column] = labels
        return out

    def is_valid(self, data: pd.DataFrame) -> pd.Series:
        """True where the label equals the recomputed band (or the band is undefined)."""
        expected = self.expected_labels(data)
        observed = data[self.label_column].astype(object)
        return (expected.isna() | observed.eq(expected)).astype(bool)

    def expected_labels(self, data: pd.DataFrame) -> pd.Series:
        """Band implied by the counts of every row (NaN where undefined), aligned to ``data.index``."""
        return band_labels(data, self.label_column, self.constants)

    def learned(self) -> dict[str, Any]:
        """JSON-serialisable fitted state for model cards."""
        return {
            "label_column": self.label_column,
            "constants": dict(self.constants),
            "training_label_consistency": getattr(self, "training_consistency", None),
        }


class EscalationCoverConstraint(SingleTableProgrammableConstraint):
    """Escalation beds = cover needed by the census + learned extra (``capacity`` group, ward_daily).

    The *cover* of a ward-day is the number of escalation beds its census needs above the base
    beds, net of closed beds (beds are only closed when free):
    ``max(0, max(census_start, census_end) - bed_capacity + beds_closed)`` with ``census_end`` and
    ``bed_capacity`` from :func:`bedsynth.schema.derive_ward_daily`. In the reference data
    ``escalation_beds_open >= cover`` always holds (WD-H06/WD-H15), so

    * ``transform`` replaces ``escalation_beds_open`` by ``extra = escalation_beds_open - cover``
      (``>= 0``; the model learns how many beds are opened beyond the need),
    * ``reverse_transform`` sets ``escalation_beds_open = min(cover + extra, ward maximum)`` where
      the ward maximum is the largest value seen in training (keeps WD-H08).

    ``census_start <= available_beds`` (WD-H15, which implies WD-H07) and ``census_end <=
    available_beds`` (WD-H06) then hold for every row whose cover fits within the ward maximum;
    the others (a census the
    ward never accommodated) are left to the generator's rejection step, so ``is_valid`` only
    checks ``escalation_beds_open >= min(cover, ward maximum)``. Frames without the escalation
    column or the census inputs (e.g. SDV condition frames) pass through unchanged.

    Args:
        constants: Values of input columns the (segment) model does not learn - the segment key
            (``ward_id``) and columns that are constant within the segment.
    """

    def __init__(self, constants: dict[str, Any] | None = None):
        self.constants = {str(k): _plain(v) for k, v in (constants or {}).items()}
        self.max_escalation: dict[str, int] = {}

    def _usable(self, columns: Iterable[str]) -> bool:
        return ESCALATION_COLUMN in set(columns) and not _missing_inputs(CAPACITY_INPUTS, columns, self.constants)

    def validate(self, metadata: Any) -> None:
        """The escalation column must be modelled and every other input modelled or constant."""
        columns = _metadata_columns(metadata)
        if columns is not None and not self._usable(columns):
            raise ValueError(
                "EscalationCoverConstraint needs escalation_beds_open modelled and its inputs "
                f"{list(CAPACITY_INPUTS)} modelled or constant; metadata columns: {sorted(columns)}"
            )

    def fit(self, data: pd.DataFrame, metadata: Any) -> None:
        """Learn the largest escalation per ward and record how often the cover was needed."""
        if not self._usable(data.columns):
            raise ValueError(
                f"EscalationCoverConstraint: inputs {_missing_inputs(CAPACITY_INPUTS, data.columns, self.constants)} "
                "are neither modelled nor constant"
            )
        escalation = pd.to_numeric(data[ESCALATION_COLUMN], errors="coerce")
        wards = _input_column(data, "ward_id", self.constants).astype(object).to_numpy()
        self.max_escalation = {str(w): int(v) for w, v in escalation.groupby(wards).max().items() if pd.notna(v)}
        cover = self.cover(data)
        self.training_rows = int(len(data))
        self.training_cover_share = float((cover > 0).mean()) if len(data) else None
        self.training_consistency = float((escalation >= cover).mean()) if len(data) else None
        if self.training_consistency is not None and self.training_consistency < 1.0:
            LOGGER.warning("EscalationCoverConstraint: %.1f%% of training rows open fewer escalation beds than "
                           "their census needs (their extra beds are clipped at 0)", 100 * (1 - self.training_consistency))

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """``escalation_beds_open -> escalation_beds_open - cover`` (clipped at 0)."""
        if not self._usable(data.columns):
            return data
        out = data.copy()
        escalation = pd.to_numeric(data[ESCALATION_COLUMN], errors="coerce")
        out[ESCALATION_COLUMN] = (escalation - self.cover(data)).clip(lower=0).astype(data[ESCALATION_COLUMN].dtype)
        return out

    def get_updated_metadata(self, metadata: Any) -> Any:
        """The column keeps its name and numerical sdtype (it holds the extra beds while modelled)."""
        return metadata

    def reverse_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """``escalation_beds_open = min(cover + extra, ward maximum)``."""
        if not self._usable(data.columns):
            return data
        out = data.copy()
        extra = pd.to_numeric(data[ESCALATION_COLUMN], errors="coerce").clip(lower=0)
        escalation = np.minimum((self.cover(data) + extra).to_numpy(dtype=float), self._ward_maximum(data))
        out[ESCALATION_COLUMN] = np.round(escalation).astype(data[ESCALATION_COLUMN].dtype)
        return out

    def is_valid(self, data: pd.DataFrame) -> pd.Series:
        """True where the escalation beds cover the census (as far as the ward maximum allows)."""
        if not self._usable(data.columns):
            return pd.Series(True, index=data.index)
        escalation = pd.to_numeric(data[ESCALATION_COLUMN], errors="coerce").to_numpy(dtype=float)
        needed = np.minimum(self.cover(data).to_numpy(dtype=float), self._ward_maximum(data))
        return pd.Series(escalation >= needed, index=data.index)

    def cover(self, data: pd.DataFrame) -> pd.Series:
        """Escalation beds each row's census needs above the base beds net of closed beds (0 where undefined)."""
        derived = _derived_ward_days(data, self.constants)
        census = np.maximum(pd.to_numeric(derived["census_start"], errors="coerce"),
                            pd.to_numeric(derived["census_end"], errors="coerce"))
        needed = census - pd.to_numeric(derived["bed_capacity"], errors="coerce") + pd.to_numeric(
            derived["beds_closed"], errors="coerce")
        return needed.clip(lower=0).fillna(0).astype(float)

    def _ward_maximum(self, data: pd.DataFrame) -> np.ndarray:
        """Per-row largest escalation seen in training for the row's ward (inf if unknown)."""
        wards = _input_column(data, "ward_id", self.constants).astype(object)
        return wards.map(lambda w: float(self.max_escalation.get(str(w), np.inf))).to_numpy(dtype=float)

    def learned(self) -> dict[str, Any]:
        """JSON-serialisable fitted state for model cards."""
        return {
            "max_escalation": dict(self.max_escalation),
            "constants": dict(self.constants),
            "training_cover_share": getattr(self, "training_cover_share", None),
            "training_consistency": getattr(self, "training_consistency", None),
        }


class StructuralZerosConstraint(SingleTableProgrammableConstraint):
    """Learned structural zeros: ``category value -> numeric columns are exactly 0`` (``structural_zeros``).

    ``fit`` records, per numeric column, the values of ``category_column`` whose training rows
    (at least ``min_support`` of them) are all exactly zero; ``reverse_transform`` sets those
    columns to 0 for every sampled row with such a category value. ``linked_pairs`` lists
    ``(low, high)`` inequality pairs among the numeric columns: whenever ``high`` is forced to
    zero, ``low`` is forced too, so ``low <= high`` keeps holding.

    Args:
        category_column: Categorical column whose values carry the structural zeros.
        numeric_columns: Numeric columns that may be structurally zero.
        min_support: Minimum training rows of a category value (guards against learning
            accidental zeros from a handful of rows).
        linked_pairs: ``(low, high)`` column pairs of SDV ``Inequality`` constraints.
    """

    def __init__(
        self,
        category_column: str,
        numeric_columns: Sequence[str],
        min_support: int = STRUCTURAL_ZERO_MIN_SUPPORT,
        linked_pairs: Sequence[Sequence[str]] = (),
    ):
        if not numeric_columns:
            raise ValueError("StructuralZerosConstraint needs at least one numeric column")
        self.category_column = str(category_column)
        self.numeric_columns = [str(c) for c in numeric_columns]
        self.min_support = int(min_support)
        self.linked_pairs = [[str(low), str(high)] for low, high in linked_pairs]
        self.zero_categories: dict[str, list[Any]] = {}

    def validate(self, metadata: Any) -> None:
        """The category column and every numeric column must be modelled columns."""
        columns = _metadata_columns(metadata)
        if columns is None:
            return
        missing = [c for c in [self.category_column, *self.numeric_columns] if c not in columns]
        if missing:
            raise ValueError(f"StructuralZerosConstraint: columns {missing} are not in the metadata")

    def fit(self, data: pd.DataFrame, metadata: Any) -> None:
        """Learn, per numeric column, the frequent category values whose rows are all zero."""
        categories = data[self.category_column]
        support = categories.value_counts(dropna=True)
        frequent = {_plain(v) for v, n in support.items() if n >= self.min_support}
        learned: dict[str, list[Any]] = {}
        for column in self.numeric_columns:
            zero = pd.to_numeric(data[column], errors="coerce").eq(0)
            all_zero = zero.groupby(categories.to_numpy(), dropna=True).all()
            values = {_plain(v) for v, flag in all_zero.items() if bool(flag) and _plain(v) in frequent}
            learned[column] = sorted(values, key=str)
        for low, high in self.linked_pairs:  # zeroing the high side must zero the low side too
            if high in learned and low in learned:
                learned[low] = sorted(set(learned[low]) | set(learned[high]), key=str)
        self.zero_categories = learned
        LOGGER.debug("Structural zeros learned for %s: %s", self.category_column, learned)

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Identity: the model still sees the zero-inflated columns."""
        return data

    def get_updated_metadata(self, metadata: Any) -> Any:
        """The transform does not change the columns."""
        return metadata

    def reverse_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Force the learned structural zeros on the sampled rows."""
        rules = [(c, v) for c, v in self.zero_categories.items() if v and c in data.columns]
        if not rules:
            return data
        out = data.copy()
        categories = out[self.category_column]
        for column, values in rules:
            mask = categories.isin(values).to_numpy(dtype=bool)
            if mask.any():
                out.loc[mask, column] = 0
        return out

    def is_valid(self, data: pd.DataFrame) -> pd.Series:
        """True where no learned structural zero is violated."""
        ok = pd.Series(True, index=data.index)
        for column, values in self.zero_categories.items():
            if values and column in data.columns:
                nonzero = pd.to_numeric(data[column], errors="coerce").ne(0)
                ok &= ~(data[self.category_column].isin(values) & nonzero)
        return ok

    def learned(self) -> dict[str, Any]:
        """JSON-serialisable fitted state for model cards."""
        return {
            "category_column": self.category_column,
            "zero_categories": {c: list(v) for c, v in self.zero_categories.items()},
            "min_support": self.min_support,
        }


_POOLED = "__pooled__"
_POOLED_NONZERO = "__pooled_nonzero__"


def _min_zero_support(min_support: int) -> int:
    """Rows an all-zero group needs to keep its own (all-zero) marginal below ``min_support``."""
    return max(5, int(min_support) // 3)


def _empirical_table(values: pd.Series) -> dict[str, list[float]]:
    """Sorted distinct values and their cumulative shares (``{"values": [...], "cdf": [...]}``)."""
    counts = pd.to_numeric(values, errors="coerce").dropna().value_counts().sort_index()
    if counts.empty:
        return {"values": [0.0], "cdf": [1.0]}
    cdf = (counts.cumsum() / counts.sum()).to_numpy(dtype=float)
    cdf[-1] = 1.0
    return {"values": [float(v) for v in counts.index], "cdf": [float(c) for c in cdf]}


class ConditionalQuantileConstraint(SingleTableProgrammableConstraint):
    """Exact within-group marginals for numeric columns (``conditional_marginals`` group).

    A Gaussian copula fits one parametric marginal per column and links columns *linearly* in
    latent space. It therefore cannot represent point masses (a count that is 0 on 90 % of days),
    marginals whose shape depends on a category (delay hours per delay reason) or non-monotone
    category effects (elective lists flat Monday-Friday, rare on Saturdays, none on Sundays: along
    any weekday encoding the linear latent link turns the flat weekdays into a false gradient).
    This constraint applies the *distributional transform* within groups of rows that share the
    values of ``group_columns``:

    * ``fit`` stores, per group and column, the empirical distribution (distinct values and their
      cumulative shares). Groups with fewer than ``min_support`` rows share a pooled distribution
      (pooled over the groups that are not all zero when the column has any, else over every
      row), except small groups whose rows (at least a third of ``min_support``) are all exactly
      zero: they keep their structural zero.
    * ``transform`` replaces each value ``x`` by the standard-normal score ``Phi^-1(U)``, with ``U``
      drawn uniformly between ``F_g(x-)`` and ``F_g(x)`` (seeded, so repeated calls agree; ties at
      point masses are spread over their probability). Every column becomes N(0, 1) within every
      group, i.e. independent of the group columns.
    * ``reverse_transform`` maps a sampled score back with the group's empirical quantile function
      ``F_g^-1(Phi(z))``, which reproduces each group's marginal exactly (point masses included),
      then restores the ``(low, high)`` orderings of ``bounds`` with ``low = min(low, high)``.

    ``band_groups`` add a band label recomputed from each row's counts (:func:`band_labels`) to the
    group key - in ``fit`` and ``transform`` from the training counts, in ``reverse_transform`` from
    the restored counts - so a ward-day is restored with the marginal of the band it ends up in
    (e.g. ED boarders only on full wards). Such a constraint must be listed where its
    ``reverse_transform`` sees the final values of the band inputs (see "Ordering inside SDV").

    The copula still models how the scores depend on every other column (e.g. how long the delay
    is relative to the stay within a reason). Only values observed in a group's training rows are
    generated. ``group_columns`` that are constant within a segment are left out by
    :func:`build_constraints`; frames without the numeric columns (SDV condition frames) pass
    through unchanged.

    Args:
        columns: Numeric columns to transform.
        group_columns: Categorical columns whose value combinations define the groups (``[]`` =
            one group).
        bounds: ``(low, high)`` pairs of ``columns`` whose ordering is restored after sampling.
        min_support: Minimum training rows of a group for its own distribution.
        seed: Seed of the uniform draws used by ``transform``.
        band_groups: Band labels (``occupancy_level`` / ``staffing_level``) that also define the
            groups, recomputed from the counts of each row (:func:`band_labels`) in ``fit``,
            ``transform`` and ``reverse_transform`` - never read from a sampled label column.
        constants: Values of band inputs the (segment) model does not learn.
    """

    band_groups: tuple[str, ...] = ()       # class defaults keep older pickles loadable
    constants: Mapping[str, Any] = {}

    def __init__(
        self,
        columns: Sequence[str],
        group_columns: Sequence[str] = (),
        bounds: Sequence[Sequence[str]] = (),
        min_support: int = CONDITIONAL_MIN_SUPPORT,
        seed: int = 0,
        band_groups: Sequence[str] = (),
        constants: Mapping[str, Any] | None = None,
    ):
        if not columns:
            raise ValueError("ConditionalQuantileConstraint needs at least one numeric column")
        self.columns = [str(c) for c in columns]
        self.group_columns = [str(c) for c in group_columns]
        self.bounds = [[str(low), str(high)] for low, high in bounds]
        unknown = sorted({c for pair in self.bounds for c in pair} - set(self.columns))
        if unknown:
            raise ValueError(f"ConditionalQuantileConstraint bounds use columns it does not transform: {unknown}")
        bands = tuple(str(b) for b in band_groups)
        if any(b not in BAND_LABEL_INPUTS for b in bands):
            raise ValueError(f"band_groups must be among {sorted(BAND_LABEL_INPUTS)}, got {list(bands)}")
        self.band_groups = bands
        self.constants = {str(k): _plain(v) for k, v in (constants or {}).items()}
        self.min_support = int(min_support)
        self.seed = int(seed)
        self.tables: dict[str, dict[Any, dict[str, list[float]]]] = {}
        self.integer_columns: list[str] = []

    # ------------------------------------------------------------------ helpers
    def _groups(self, data: pd.DataFrame) -> list[tuple[tuple, np.ndarray]]:
        """``[(group key, row positions), ...]``; keys are tuples of plain group-column values
        followed by the recomputed bands."""
        if not len(data):
            return []
        if not self.group_columns and not self.band_groups:
            return [((), np.arange(len(data)))]
        frame = data[self.group_columns].astype(object).reset_index(drop=True)
        names = list(self.group_columns)
        for band in self.band_groups:
            name = f"__band__{band}"
            frame[name] = band_labels(data, band, self.constants).astype(object).to_numpy()
            names.append(name)
        indices = frame.groupby(names, sort=False, dropna=False).indices
        out = []
        for key, rows in indices.items():
            values = key if isinstance(key, tuple) else (key,)
            out.append((tuple(_plain(v) for v in values), np.asarray(rows, dtype=np.int64)))
        return out

    def _table(self, column: str, key: Any) -> dict[str, list[float]]:
        tables = self.tables[column]
        return tables.get(key) or tables.get(_POOLED_NONZERO) or tables[_POOLED]

    def _applies_to(self, data: pd.DataFrame) -> bool:
        if not all(c in data.columns for c in [*self.columns, *self.group_columns]):
            return False
        return not any(_missing_inputs(BAND_LABEL_INPUTS[band], data.columns, self.constants)
                       for band in self.band_groups)

    # ------------------------------------------------------------------ SDV API
    def validate(self, metadata: Any) -> None:
        """The numeric and group columns must be modelled columns."""
        columns = _metadata_columns(metadata)
        if columns is None:
            return
        missing = [c for c in [*self.columns, *self.group_columns] if c not in columns]
        if missing:
            raise ValueError(f"ConditionalQuantileConstraint: columns {missing} are not in the metadata")

    def fit(self, data: pd.DataFrame, metadata: Any) -> None:
        """Learn the empirical distribution of every column per group (and the pooled fallbacks)."""
        groups = self._groups(data)
        self.integer_columns = [c for c in self.columns if pd.api.types.is_integer_dtype(data[c])]
        self.tables = {}
        for column in self.columns:
            values = pd.to_numeric(data[column], errors="coerce").reset_index(drop=True)
            tables: dict[Any, dict[str, list[float]]] = {_POOLED: _empirical_table(values)}
            nonzero = [rows for _, rows in groups if values.iloc[rows].ne(0).any()]
            if nonzero and len(nonzero) < len(groups):
                pooled = values.iloc[np.concatenate(nonzero)]
                if len(pooled) >= self.min_support:
                    tables[_POOLED_NONZERO] = _empirical_table(pooled)
            for key, rows in groups:
                part = values.iloc[rows]
                # A small group keeps its own table only when all its rows are exact zeros (a
                # structural zero: e.g. no Saturday lists on a ward); others share a pooled one.
                if len(rows) >= self.min_support or (len(rows) >= _min_zero_support(self.min_support)
                                                     and part.eq(0).all()):
                    tables[key] = _empirical_table(part)
            self.tables[column] = tables
        self.training_rows = int(len(data))
        self.group_sizes = {"/".join(str(v) for v in key) or "all": int(len(rows)) for key, rows in groups}

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Randomised probability integral transform within groups -> standard-normal scores."""
        if not self._applies_to(data) or not self.tables:
            return data
        from scipy.special import ndtri

        out = data.copy()
        jitter = np.random.default_rng(self.seed).random((len(data), len(self.columns)))
        positions = self._groups(data)
        for j, column in enumerate(self.columns):
            values = pd.to_numeric(data[column], errors="coerce").to_numpy(dtype=float)
            scores = np.zeros(len(data), dtype=float)
            for key, rows in positions:
                table = self._table(column, key)
                support, cdf = np.asarray(table["values"]), np.asarray(table["cdf"])
                lower_cdf = np.concatenate([[0.0], cdf[:-1]])
                index = np.searchsorted(support, values[rows], side="left")
                inside = np.clip(index, 0, len(support) - 1)
                exact = (index < len(support)) & np.isclose(support[inside], values[rows])
                below = np.where(index > 0, cdf[np.clip(index - 1, 0, len(cdf) - 1)], 0.0)
                low = np.where(exact, lower_cdf[inside], below)
                high = np.where(exact, cdf[inside], below)
                scores[rows] = low + jitter[rows, j] * (high - low)
            out[column] = ndtri(np.clip(scores, _PIT_EPSILON, 1.0 - _PIT_EPSILON))
        return out

    def get_updated_metadata(self, metadata: Any) -> Any:
        """The columns keep their names but hold unrounded scores (no decimal places or ranges)."""
        from sdv.metadata import Metadata

        content = metadata.to_dict()
        for table in content.get("tables", {}).values():
            for column in self.columns:
                if column in table.get("columns", {}):
                    table["columns"][column] = {"sdtype": "numerical"}
        return Metadata.load_from_dict(content)

    def reverse_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Scores -> values with the group's empirical quantile function; restore ``bounds``."""
        if not self._applies_to(data) or not self.tables:
            return data
        from scipy.special import ndtr

        out = data.copy()
        positions = self._groups(data)
        for column in self.columns:
            probabilities = ndtr(pd.to_numeric(data[column], errors="coerce").fillna(0.0).to_numpy(dtype=float))
            restored = np.zeros(len(data), dtype=float)
            for key, rows in positions:
                table = self._table(column, key)
                support, cdf = np.asarray(table["values"]), np.asarray(table["cdf"])
                index = np.clip(np.searchsorted(cdf, probabilities[rows], side="left"), 0, len(support) - 1)
                restored[rows] = support[index]
            out[column] = np.round(restored).astype("int64") if column in self.integer_columns else restored
        for low, high in self.bounds:
            out[low] = np.minimum(out[low].to_numpy(), out[high].to_numpy())
        return out

    def is_valid(self, data: pd.DataFrame) -> pd.Series:
        """True where every ``(low, high)`` bound holds (always after ``reverse_transform``)."""
        ok = pd.Series(True, index=data.index)
        for low, high in self.bounds:
            if low in data.columns and high in data.columns:
                ok &= ~(pd.to_numeric(data[low], errors="coerce") > pd.to_numeric(data[high], errors="coerce"))
        return ok

    def learned(self) -> dict[str, Any]:
        """JSON-serialisable fitted state for model cards (group sizes and zero shares, no values)."""
        summary: dict[str, Any] = {}
        for column, tables in self.tables.items():
            groups = {}
            for key, table in tables.items():
                label = key if isinstance(key, str) else "/".join(str(v) for v in key) or "all"
                values, cdf = table["values"], table["cdf"]
                zero_share = 0.0
                if 0.0 in values:
                    i = values.index(0.0)
                    zero_share = cdf[i] - (cdf[i - 1] if i else 0.0)
                groups[label] = {"distinct_values": len(values), "zero_share": round(float(zero_share), 4)}
            summary[column] = groups
        return {
            "group_columns": list(self.group_columns),
            **({"band_groups": list(self.band_groups)} if self.band_groups else {}),
            "bounds": [list(pair) for pair in self.bounds],
            "min_support": self.min_support,
            "group_sizes": dict(getattr(self, "group_sizes", {})),
            "marginals": summary,
        }


# =========================================================================================
# Spec resolution and SDV objects
# =========================================================================================
def _check_table(table: str) -> None:
    if table not in TABLES:
        raise KeyError(f"Unknown table '{table}'. Known tables: {list(TABLES)}")


def applicable_specs(
    table: str,
    columns: Iterable[str] | None = None,
    *,
    groups: Iterable[str] | None = CONSTRAINT_GROUPS,
    constants: Mapping[str, Any] | None = None,
) -> list[tuple[ConstraintSpec, tuple[str, ...]]]:
    """Constraint specs applicable to the modelled ``columns`` with their effective columns.

    Args:
        table: BedSynth table name.
        columns: Columns actually modelled by the synthesizer (default: all train columns).
        groups: Constraint groups to consider (default: every group; ``None`` = none).
        constants: Values of train columns the (segment) model does not learn.

    Returns:
        ``[(spec, effective_columns), ...]`` in declaration (= SDV application) order.
    """
    _check_table(table)
    enabled = set(constraint_groups(groups))
    present = list(columns) if columns is not None else training_columns(table)
    out: list[tuple[ConstraintSpec, tuple[str, ...]]] = []
    for spec in CONSTRAINT_SPECS[table]:
        if spec.kind not in enabled:
            continue
        effective = spec.applicable_columns(present, constants)
        if effective is not None:
            out.append((spec, effective))
    # A conditional-marginals constraint that transforms both columns of an inequality restores
    # their ordering itself (SDV's Inequality cannot read its standard-normal scores).
    bounded = {tuple(pair) for spec, effective in out if spec.kind == CONDITIONAL_MARGINALS
               for pair in spec.bounds if set(pair) <= set(effective)}
    return [(spec, effective) for spec, effective in out
            if not (spec.kind == INEQUALITY and tuple(effective) in bounded)]


def _effective_bounds(spec: ConstraintSpec, columns: Sequence[str]) -> list[tuple[str, str]]:
    """``bounds`` of a conditional-marginals spec whose columns are both transformed."""
    return [pair for pair in spec.bounds if set(pair) <= set(columns)]


def _inequality_pairs(table: str, columns: Sequence[str]) -> list[tuple[str, str]]:
    """``(low, high)`` inequality pairs whose columns are both in ``columns``."""
    return [
        (spec.columns[0], spec.columns[1])
        for spec in CONSTRAINT_SPECS[table]
        if spec.kind == INEQUALITY and set(spec.columns) <= set(columns)
    ]


def _to_sdv(table: str, spec: ConstraintSpec, columns: tuple[str, ...], constants: Mapping[str, Any]) -> Any:
    """Instantiate a fresh (unfitted) constraint object."""
    from sdv.cag import FixedCombinations, Inequality

    if spec.kind == INEQUALITY:
        low, high = columns
        return Inequality(low_column_name=low, high_column_name=high, strict_boundaries=False)
    if spec.kind == FIXED_COMBINATIONS:
        return FixedCombinations(column_names=list(columns))
    if spec.kind == REGIME_LABELS:
        label = columns[0]
        needed = {c: constants[c] for c in BAND_LABEL_INPUTS[label] if c in constants}
        return BandLabelConstraint(label_column=label, constants=needed)
    if spec.kind == CAPACITY:
        return EscalationCoverConstraint(constants={c: constants[c] for c in CAPACITY_INPUTS if c in constants})
    if spec.kind == STRUCTURAL_ZEROS:
        return StructuralZerosConstraint(
            category_column=spec.anchor,
            numeric_columns=list(columns),
            min_support=STRUCTURAL_ZERO_MIN_SUPPORT,
            linked_pairs=_inequality_pairs(table, columns),
        )
    if spec.kind == CONDITIONAL_MARGINALS:
        band_inputs = {c for band in spec.band_groups for c in BAND_LABEL_INPUTS[band]}
        return ConditionalQuantileConstraint(
            columns=list(columns),
            group_columns=_modelled_group_columns(spec, constants),
            bounds=_effective_bounds(spec, columns),
            min_support=CONDITIONAL_MIN_SUPPORT,
            seed=zlib.crc32("|".join(columns).encode("utf-8")),
            band_groups=spec.band_groups,
            constants={c: constants[c] for c in sorted(band_inputs) if c in constants},
        )
    raise ValueError(f"Unsupported constraint kind '{spec.kind}'")


def _modelled_group_columns(spec: ConstraintSpec, constants: Mapping[str, Any]) -> list[str]:
    """Group columns of a conditional-marginals spec that the (segment) model learns."""
    return [c for c in spec.group_by if c not in constants]


def build_constraints(
    table: str,
    columns: list[str],
    *,
    groups: Iterable[str] | None = CONSTRAINT_GROUPS,
    constants: Mapping[str, Any] | None = None,
) -> list:
    """Constraint objects for ``table`` that apply to the modelled ``columns``.

    New, unfitted objects are returned on every call (constraints keep fitted state, so they
    must never be shared between synthesizers). The list order is the SDV ``add_constraints``
    order: programmable constraints first (see the module docstring).

    Args:
        table: ``"ward_daily"`` or ``"patient_stays"``.
        columns: Columns modelled by the synthesizer (segments drop constant columns and the
            segment column(s)).
        groups: Constraint groups to build (default: every group).
        constants: Values of the train columns the (segment) model does not learn (segment key
            values and constant columns); band labels substitute them for missing inputs.

    Returns:
        List of constraint instances to pass to ``synthesizer.add_constraints``.
    """
    constants = dict(constants or {})
    return [
        _to_sdv(table, spec, effective, constants)
        for spec, effective in applicable_specs(table, columns, groups=groups, constants=constants)
    ]


def _implementation(spec: ConstraintSpec) -> str:
    if spec.kind == INEQUALITY:
        return "sdv.cag.Inequality"
    if spec.kind == FIXED_COMBINATIONS:
        return "sdv.cag.FixedCombinations"
    classes = {
        REGIME_LABELS: BandLabelConstraint,
        STRUCTURAL_ZEROS: StructuralZerosConstraint,
        CAPACITY: EscalationCoverConstraint,
        CONDITIONAL_MARGINALS: ConditionalQuantileConstraint,
    }
    return f"{__name__}.{classes[spec.kind].__name__}"


def _expression(spec: ConstraintSpec, columns: tuple[str, ...], constants: Mapping[str, Any] | None = None) -> str:
    if spec.kind == INEQUALITY:
        return f"{columns[0]} <= {columns[1]}"
    if spec.kind == FIXED_COMBINATIONS:
        return f"({', '.join(columns)}) in observed combinations"
    if spec.kind == REGIME_LABELS:
        if columns[0] == "occupancy_level":
            return "occupancy_level = occupancy_levels(census_end, available_beds)"
        return "staffing_level = staffing_levels(nurses_day_shift, ward_id)"
    if spec.kind == CAPACITY:
        return ("escalation_beds_open = min(max(0, max(census_start, census_end) - bed_capacity + beds_closed) "
                "+ learned extra, ward maximum)")
    if spec.kind == CONDITIONAL_MARGINALS:
        group_columns = _modelled_group_columns(spec, constants or {})
        group_columns += [f"{band.replace('_level', '')} band (recomputed)" for band in spec.band_groups]
        groups = " x ".join(group_columns) if group_columns else "segment"
        text = f"{', '.join(columns)} ~ empirical marginal per {groups} (distributional transform)"
        bounds = _effective_bounds(spec, columns)
        if bounds:
            text += "; " + ", ".join(f"{low} <= {high}" for low, high in bounds)
        return text
    return f"{spec.anchor} in learned zero set => {' = '.join(columns)} = 0"


def describe_constraints(
    table: str,
    columns: list[str] | None = None,
    *,
    groups: Iterable[str] | None = CONSTRAINT_GROUPS,
    constants: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """JSON-serialisable description of the constraints (for model cards, reports and the API).

    Args:
        table: BedSynth table name.
        columns: Optional modelled columns; when given only applicable constraints are listed,
            with their effective (possibly reduced) column lists.
        groups: Constraint groups to describe (default: every group).
        constants: Segment constants (see :func:`build_constraints`).
    """
    specs = applicable_specs(table, columns, groups=groups, constants=constants)
    described = []
    for spec, effective in specs:
        item: dict[str, Any] = {
            "type": spec.kind,
            "group": spec.kind,
            "sdv_class": _PROGRAMMABLE_BASE if spec.programmable else _implementation(spec),
            "implementation": _implementation(spec),
            "programmable": spec.programmable,
            "columns": ([spec.anchor] if spec.kind == STRUCTURAL_ZEROS else []) + list(effective),
            "expression": _expression(spec, effective, constants),
            "description": spec.description,
            "enforces_rules": list(spec.rule_ids),
        }
        if spec.inputs:
            item["inputs"] = list(spec.inputs)
        if spec.kind == CONDITIONAL_MARGINALS:
            item["group_by"] = _modelled_group_columns(spec, constants or {})
            if spec.band_groups:
                item["band_groups"] = list(spec.band_groups)
            item["bounds"] = [list(pair) for pair in _effective_bounds(spec, effective)]
        described.append(item)
    return described


def fixed_combination_groups(
    table: str, columns: list[str], groups: Iterable[str] | None = CONSTRAINT_GROUPS
) -> list[tuple[str, ...]]:
    """Column groups of the applicable FixedCombinations constraints.

    Used to detect infeasible conditions early: a fixed-combination group can only produce
    value combinations seen during training.
    """
    return [
        eff for spec, eff in applicable_specs(table, columns, groups=groups) if spec.kind == FIXED_COMBINATIONS
    ]


def learned_state(constraint: Any) -> dict[str, Any]:
    """Fitted state of a BedSynth programmable constraint (``{}`` for SDV built-ins)."""
    learned = getattr(constraint, "learned", None)
    return learned() if callable(learned) else {}


ENFORCEMENT_NOTE = (
    "Constraints inside the SDV synthesizer (groups: inequality, fixed_combinations, regime_labels, "
    "structural_zeros, capacity, conditional_marginals) enforce ordering inequalities, observed category "
    "combinations, band labels recomputed from their counts, learned structural zeros, escalation beds that "
    "cover the census and exact per-group marginals (delay hours per delay reason, elective lists per ward "
    "and weekday, ED boarders and bed waits per ward and occupancy band, closed beds per ward and staffing "
    "band, day-shift nurses per ward and month, escalation beds beyond the census need per ward, length of "
    "stay per admission weekday, hour, wait and turnaround marginals per ward x admission source) by "
    "construction; derived columns are recomputed by schema.derive (flow balance and ratios); the remaining "
    "hard-rule failures (a census beyond the ward's observed escalation range, same-day stay hours, delays "
    "longer than the stay, rebooking leads) are removed by the generator's rejection sampling with "
    "bedsynth.validation."
)

__all__ = [
    "BAND_LABEL_INPUTS",
    "BUILTIN_GROUPS",
    "BandLabelConstraint",
    "CAPACITY",
    "CAPACITY_INPUTS",
    "CONDITIONAL_MARGINALS",
    "CONDITIONAL_MIN_SUPPORT",
    "CONSTRAINT_GROUPS",
    "CONSTRAINT_SPECS",
    "ENFORCEMENT_NOTE",
    "EscalationCoverConstraint",
    "FIXED_COMBINATIONS",
    "INEQUALITY",
    "PROGRAMMABLE_GROUPS",
    "REGIME_LABELS",
    "STRUCTURAL_ZEROS",
    "STRUCTURAL_ZERO_MIN_SUPPORT",
    "ConditionalQuantileConstraint",
    "ConstraintSpec",
    "StructuralZerosConstraint",
    "applicable_specs",
    "band_labels",
    "build_constraints",
    "constraint_groups",
    "describe_constraints",
    "fixed_combination_groups",
    "learned_state",
]
