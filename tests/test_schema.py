"""Tests for the shared schema contract (``bedsynth.schema``) and scenario helpers."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bedsynth import schema
from bedsynth.data.fixtures import make_patient_stays, make_ward_daily
from bedsynth.scenarios import SCENARIOS, merge_conditions, merge_filters, scenario_mask, scenarios_for


@pytest.mark.parametrize(
    ("column", "raw", "expected"),
    [
        ("ward_id", "AMU", "AMU"),
        ("ward_id", "  Gen Med ", "GMED"),
        ("ward_id", "general   medicine", "GMED"),          # internal whitespace run
        ("ward_id", "Children's  Ward", "PAED"),
        ("separation_type", "Transfer_Out", "transfer_out"),  # underscore spelling
        ("separation_type", "transfer   out", "transfer_out"),
        ("delay_reason", "awaiting_care package", "awaiting_care_package"),
        ("discharge_destination", "HOME_WITH_SUPPORT", "home_with_support"),
        ("delay_reason", "N/A", "none"),
        ("day_of_week", "monday", "Mon"),
        ("admission_day_of_week", " Thurs ", "Thu"),
        ("ward_id", "Mars Ward", None),
        ("ward_id", None, None),
        ("ward_id", float("nan"), None),
        ("ward_id", pd.NA, None),
        ("month", 7, 7),                                     # no aliases: returned unchanged
    ],
)
def test_normalize_category(column, raw, expected):
    assert schema.normalize_category(column, raw) == expected


def test_every_alias_and_its_noisy_variants_normalise():
    for column, aliases in schema.CATEGORY_ALIASES.items():
        for alias, canonical in aliases.items():
            assert schema.normalize_category(column, alias) == canonical
            assert schema.normalize_category(column, f"  {alias.upper()} ") == canonical
            assert schema.normalize_category(column, alias.replace(" ", "  ")) == canonical


def test_normalize_category_series_keeps_index():
    series = pd.Series(["icu", "Paediatrics", "??"], index=[10, 11, 12])
    out = schema.normalize_category_series(series, "ward_id")
    assert out.tolist() == ["ICU", "PAED", None] and list(out.index) == [10, 11, 12]


def test_bands_match_scalar_helpers():
    census = pd.Series([0, 20, 21, 26, 27, 28, 30])
    available = pd.Series([28] * 7)
    vectorised = schema.occupancy_levels(census, available).tolist()
    assert vectorised == [schema.occupancy_level_for(c, 28) for c in census]
    assert vectorised == ["low", "low", "normal", "high", "high", "critical", "critical"]
    nurses = pd.Series([1, 5, 6, 7, 8])
    wards = pd.Series(["AMU"] * 5)
    assert schema.staffing_levels(nurses, wards).tolist() == [schema.staffing_level_for(n, "AMU") for n in nurses]
    assert schema.staffing_level_for(6, "AMU") == "adequate" and schema.staffing_level_for(5, "AMU") == "short"


def test_derive_identities_on_fixtures():
    wd = make_ward_daily(n_days=10, seed=3)
    flow = (wd["census_start"] + wd["admissions_elective"] + wd["admissions_emergency"] + wd["transfers_in"]
            - wd["transfers_out"] - wd["discharges"])
    assert (flow == wd["census_end"]).all()
    assert (wd["available_beds"] == wd["bed_capacity"] + wd["escalation_beds_open"] - wd["beds_closed"]).all()
    ps = make_patient_stays(n=200, seed=4)
    assert (ps["los_hours"] == ps["los_nights"] * 24 + ps["discharge_hour"] - ps["admission_hour"]).all()
    assert (ps["is_delayed_discharge"] == (ps["discharge_delay_hours"] > 0)).all()
    assert list(schema.derive(schema.WARD_DAILY, wd[schema.training_columns(schema.WARD_DAILY)]).columns) == \
        schema.column_names(schema.WARD_DAILY, ["train", "derived"])


def test_catalogue_contract():
    assert len(schema.training_columns(schema.WARD_DAILY)) == 22
    assert len(schema.training_columns(schema.PATIENT_STAYS)) == 15
    assert schema.TOTAL_BEDS == 194 and len(SCENARIOS) == 26
    assert len(scenarios_for(schema.WARD_DAILY)) == 16 and len(scenarios_for(schema.PATIENT_STAYS)) == 10
    ids = schema.assign_ids(make_patient_stays(n=3, seed=1), schema.PATIENT_STAYS)["stay_id"].tolist()
    assert ids == ["ST-0000001", "ST-0000002", "ST-0000003"]


def test_scenario_helpers():
    assert merge_conditions({"month": (12, 1, 2)}, {"month": (1, 5)}) == {"month": (1,)}
    with pytest.raises(ValueError):
        merge_conditions({"month": (12,)}, {"month": (6,)})
    merged = merge_filters({"los_nights": {"min": 7}}, {"los_nights": {"min": 3, "max": 20}})
    assert merged == {"los_nights": {"min": 7, "max": 20}}
    wd = make_ward_daily(n_days=20, seed=5)
    mask = scenario_mask(wd, "high_occupancy")
    assert mask.dtype == bool and (wd.loc[mask, "occupancy_rate"] >= 0.9).all()
    assert np.array_equal(mask.to_numpy(), wd["occupancy_level"].isin(["high", "critical"]).to_numpy())


def test_review_constants_are_consistent():
    """D1: the delay threshold and rebooking gap are shared by schema, scenarios and rules."""
    assert schema.DELAY_THRESHOLD_HOURS == 4.0 and schema.REBOOK_MIN_DAYS == 7
    assert schema.min_rebooked_lead_days(0) == 1 and schema.min_rebooked_lead_days(3) == 22
    assert np.array_equal(schema.min_rebooked_lead_days(np.array([0, 1, 2])), np.array([1, 8, 15]))
    assert SCENARIOS["delayed_discharge_stays"].filters == {"discharge_delay_hours": {"min": schema.DELAY_THRESHOLD_HOURS}}


def test_scenarios_select_operating_conditions_not_ward_types():
    """D7: staffing varies both ways; turnover and escalation scenarios are pressure conditions."""
    staffing = {s.name: s.conditions.get("staffing_level") for s in scenarios_for(schema.WARD_DAILY) if s.category == "staffing"}
    assert staffing == {"staffing_shortage": ("short",), "staffing_surplus": ("surplus",)}
    turnover = SCENARIOS["high_turnover"]
    assert set(turnover.conditions["ward_id"]) == set(schema.WARD_IDS) - {"AMU", "PAED"}
    assert SCENARIOS["escalation_capacity"].conditions == {"occupancy_level": ("high", "critical")}
    wd = make_ward_daily(n_days=60, seed=7)
    assert not wd.loc[scenario_mask(wd, "high_turnover"), "ward_id"].isin(["AMU", "PAED"]).any()
    surplus = scenario_mask(wd, "staffing_surplus")
    assert surplus.any() and (wd.loc[surplus, "staffing_level"] == "surplus").all()
