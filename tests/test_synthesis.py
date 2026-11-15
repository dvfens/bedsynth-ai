"""Tests for the SDV synthesis layer: constraints, model wrapper, registry and training.

Models are fitted once per module on small rule-satisfying fixtures (seconds, not minutes);
CTGAN/TVAE fits (1 epoch) are marked ``slow``.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import cloudpickle
import numpy as np
import pandas as pd
import pytest

from bedsynth import schema, validation
from bedsynth.config import get_path, get_settings, project_relative, resolve_project_path
from bedsynth.data.fixtures import make_patient_stays, make_ward_daily
from bedsynth.synthesis import constraints as cons
from bedsynth.synthesis.models import (
    BedSynthSynthesizer,
    ExperimentConfig,
    derive_seed,
    json_safe,
    largest_remainder,
    segment_columns_of,
)
from bedsynth.synthesis.registry import ModelNotFoundError, ModelRegistry
from bedsynth.synthesis.training import (
    max_training_rows,
    sample_diagnostics,
    subsample_training_rows,
    train_all,
    train_experiment,
)

WD, PS = schema.WARD_DAILY, schema.PATIENT_STAYS
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ELECTIVE_ZERO_COLUMNS = ("elective_scheduled", "elective_cancelled", "transfers_in")


def _config(experiment_id: str, model: str = "gaussian_copula", **kwargs) -> ExperimentConfig:
    params = kwargs.pop("params", {"default_distribution": "truncnorm"})
    return ExperimentConfig(experiment_id=experiment_id, model=model, params=params, **kwargs)


def _violation_rates(frame: pd.DataFrame, table: str) -> dict[str, float]:
    report = validation.validate_table(schema.derive(table, frame), table, include_soft=False)
    return {r["rule_id"]: r["violation_rate"] for r in report.rule_results}


# =========================================================================================
# Module-scoped fitted models
# =========================================================================================
@pytest.fixture(scope="module")
def wd_global(ward_daily_df) -> BedSynthSynthesizer:
    """Global GaussianCopula with every constraint group and ordered encodings (ward_daily)."""
    config = _config("wd_global", constraints=True, ordinal_metadata=True, ordered_encodings=True)
    return BedSynthSynthesizer(WD, config, tune_distributions=False).fit(ward_daily_df)


@pytest.fixture(scope="module")
def wd_global_builtin(ward_daily_df) -> BedSynthSynthesizer:
    """Same global model with the SDV built-in constraints only (ablation reference)."""
    config = _config("wd_builtin", constraints=list(cons.BUILTIN_GROUPS), ordinal_metadata=True,
                     ordered_encodings=True)
    return BedSynthSynthesizer(WD, config, tune_distributions=False).fit(ward_daily_df)


@pytest.fixture(scope="module")
def wd_segmented(ward_daily_df) -> BedSynthSynthesizer:
    """gc_segmented recipe (per ward, SDV built-in constraints) on four wards, no marginal tuning."""
    subset = ward_daily_df[ward_daily_df["ward_id"].isin(["AMU", "GMED", "ICU", "PAED"])]
    return BedSynthSynthesizer(WD, "gc_segmented", tune_distributions=False).fit(subset)


@pytest.fixture(scope="module")
def wd_tuned(ward_daily_df) -> BedSynthSynthesizer:
    """gc_tuned recipe for ward_daily (per ward, every constraint group) on four wards."""
    subset = ward_daily_df[ward_daily_df["ward_id"].isin(["AMU", "GMED", "ICU", "PAED"])]
    return BedSynthSynthesizer(WD, "gc_tuned", tune_distributions=False).fit(subset)


@pytest.fixture(scope="module")
def ps_global(patient_stays_df) -> BedSynthSynthesizer:
    config = _config("ps_global", constraints=True, ordinal_metadata=True, ordered_encodings=True)
    return BedSynthSynthesizer(PS, config, tune_distributions=False).fit(patient_stays_df)


@pytest.fixture(scope="module")
def ps_subset(patient_stays_df) -> pd.DataFrame:
    return patient_stays_df[patient_stays_df["ward_id"].isin(["AMU", "GSUR", "PAED"])].reset_index(drop=True)


@pytest.fixture(scope="module")
def ps_segmented(ps_subset) -> BedSynthSynthesizer:
    """gc_tuned recipe for patient_stays: one model per ward x admission source (3 wards)."""
    return BedSynthSynthesizer(PS, "gc_tuned", tune_distributions=False).fit(ps_subset)


# =========================================================================================
# Constraint specs and groups
# =========================================================================================
@pytest.mark.parametrize(
    "setting, expected",
    [
        (True, cons.CONSTRAINT_GROUPS),
        (False, ()),
        (None, ()),
        ("all", cons.CONSTRAINT_GROUPS),
        ("regime_labels", ("regime_labels",)),
        (["structural_zeros", "inequality"], ("inequality", "structural_zeros")),
        (["none"], ()),
        (np.bool_(True), cons.CONSTRAINT_GROUPS),
    ],
)
def test_constraint_groups_normalisation(setting, expected):
    assert cons.constraint_groups(setting) == expected


def test_constraint_groups_reject_unknown_values():
    with pytest.raises(ValueError, match="Unknown constraint group"):
        cons.constraint_groups(["inequality", "bogus"])
    with pytest.raises(ValueError, match="constraints must be"):
        cons.constraint_groups(3)


def test_ward_daily_constraints_order_and_groups():
    built = cons.build_constraints(WD, schema.training_columns(WD))
    # programmable constraints first: their reverse_transform runs last, on restored columns; the
    # elective conditional marginals restore cancelled <= scheduled and replace that Inequality.
    # The ED marginals grouped by the recomputed occupancy band precede the escalation cover (their
    # reverse transform reads the final counts); the nurses follow the closed beds (whose staffing
    # band reads the restored nurses).
    assert [type(c).__name__ for c in built] == (
        ["BandLabelConstraint"] * 2 + ["ConditionalQuantileConstraint", "EscalationCoverConstraint"]
        + ["StructuralZerosConstraint"] * 2 + ["ConditionalQuantileConstraint"] * 4 + ["Inequality"] * 2
    )
    assert [c.label_column for c in built[:2]] == ["occupancy_level", "staffing_level"]
    ed, electives, closed, escalation, nurses = built[2], built[6], built[7], built[8], built[9]
    assert ed.columns == ["emergency_boarders", "avg_bed_wait_hours"]
    assert ed.group_columns == ["ward_id"] and ed.band_groups == ("occupancy_level",)
    assert electives.columns == ["elective_scheduled", "elective_cancelled"]
    assert electives.group_columns == ["ward_id", "day_of_week"]
    assert electives.bounds == [["elective_cancelled", "elective_scheduled"]]
    assert closed.columns == ["beds_closed"] and closed.band_groups == ("staffing_level",)
    assert escalation.columns == ["escalation_beds_open"] and escalation.group_columns == ["ward_id"]
    assert nurses.columns == ["nurses_day_shift"] and nurses.group_columns == ["ward_id", "month"]
    assert escalation.band_groups == () and nurses.band_groups == ()
    assert [(c.low_column_name if hasattr(c, "low_column_name") else c._low_column_name) for c in built[10:]] == [
        "discharges_before_noon", "delayed_discharges"]
    builtin = cons.build_constraints(WD, schema.training_columns(WD), groups=cons.BUILTIN_GROUPS)
    assert [type(c).__name__ for c in builtin] == ["Inequality"] * 3
    described = cons.describe_constraints(WD, groups=[cons.INEQUALITY])
    assert [d["expression"] for d in described] == [
        "elective_cancelled <= elective_scheduled",
        "discharges_before_noon <= discharges",
        "delayed_discharges <= census_start",
    ]
    assert described[0]["enforces_rules"] == ["WD-H03"]
    full = cons.describe_constraints(WD)
    labels = [d for d in full if d["group"] == cons.REGIME_LABELS]
    assert [d["enforces_rules"] for d in labels] == [["WD-H10"], ["WD-H11"]]
    assert labels[0]["programmable"] is True and labels[0]["sdv_class"] == "sdv.cag.SingleTableProgrammableConstraint"
    assert labels[0]["implementation"].endswith("BandLabelConstraint") and "census_start" in labels[0]["inputs"]
    zeros = [d for d in full if d["group"] == cons.STRUCTURAL_ZEROS]
    assert [d["columns"][0] for d in zeros] == ["ward_id", "day_of_week"]
    capacity = next(d for d in full if d["group"] == cons.CAPACITY)
    assert capacity["columns"] == ["escalation_beds_open"]
    assert capacity["enforces_rules"] == ["WD-H06", "WD-H07", "WD-H15"]
    marginals = [d for d in full if d["group"] == cons.CONDITIONAL_MARGINALS]
    assert [d["group_by"] for d in marginals] == [
        ["ward_id"], ["ward_id", "day_of_week"], ["ward_id"], ["ward_id"], ["ward_id", "month"]]
    assert [d.get("band_groups") for d in marginals] == [["occupancy_level"], None, ["staffing_level"], None, None]
    assert "occupancy band (recomputed)" in marginals[0]["expression"]
    assert marginals[1]["bounds"] == [["elective_cancelled", "elective_scheduled"]]
    assert marginals[1]["enforces_rules"] == ["WD-H03"] and marginals[1]["programmable"] is True
    # the replaced inequality is not listed next to the constraint that replaces it
    assert "elective_cancelled <= elective_scheduled" not in [d["expression"] for d in full if d["group"] == "inequality"]
    assert cons.describe_constraints(WD, groups=None) == [] and cons.describe_constraints(WD, groups=False) == []
    json.dumps(full)


def test_patient_stays_constraint_order():
    built = cons.build_constraints(PS, schema.training_columns(PS))
    assert [type(c).__name__ for c in built] == [
        "StructuralZerosConstraint", *["ConditionalQuantileConstraint"] * 3, "FixedCombinations", "FixedCombinations"]
    assert built[0].category_column == "admission_source"
    # S2: no structural-zero *repair* of delay hours; they are modelled per delay reason instead
    assert built[1].columns == ["discharge_delay_hours"] and built[1].group_columns == ["delay_reason"]
    # length of stay per admission weekday (discharge weekday = admission weekday + nights, mod 7) and
    # exact hour / wait / turnaround marginals per ward x source
    assert built[2].columns == ["los_nights"] and built[2].group_columns == ["admission_day_of_week"]
    assert built[3].group_columns == ["ward_id", "admission_source"] and "bed_wait_hours" in built[3].columns
    assert [d["enforces_rules"] for d in cons.describe_constraints(PS, groups=[cons.STRUCTURAL_ZEROS])] == [["PS-H06"]]
    assert [d["enforces_rules"] for d in cons.describe_constraints(PS, groups=[cons.CONDITIONAL_MARGINALS])] == [
        ["PS-H07", "PS-H13"], [], []]
    # in a ward x source segment the constant group columns drop out: one exact marginal per segment
    segment = [c for c in schema.training_columns(PS) if c not in ("ward_id", "admission_source")]
    in_segment = cons.build_constraints(PS, segment, groups=[cons.CONDITIONAL_MARGINALS],
                                        constants={"ward_id": "GMED", "admission_source": "emergency"})
    assert [c.group_columns for c in in_segment] == [["delay_reason"], ["admission_day_of_week"], []]


def test_los_per_admission_weekday_reproduces_the_discharge_weekday_pattern(patient_stays_df):
    """The discharge weekday is admission weekday + nights (mod 7): a linear copula spreads it evenly.

    In this training frame no stay leaves on a Sunday. Modelling length of stay per admission
    weekday (``conditional_marginals``) keeps it that way; without the group a model sends about a
    seventh of the stays home on Sundays.
    """
    frame = patient_stays_df[patient_stays_df["ward_id"].isin(["GSUR", "GMED"])].reset_index(drop=True).copy()
    dow = frame["admission_day_of_week"].map({d: i for i, d in enumerate(schema.DAYS_OF_WEEK)})
    sunday = (dow + frame["los_nights"]) % 7 == 6
    frame.loc[sunday, "los_nights"] += 1                      # move every Sunday departure to Monday
    frame.loc[sunday, "discharge_delay_hours"] = frame.loc[sunday, "discharge_delay_hours"].clip(upper=24.0)

    def sunday_share(groups: list[str]) -> float:
        config = _config(f"ps_los_{len(groups)}", constraints=groups, ordinal_metadata=True, ordered_encodings=True)
        model = BedSynthSynthesizer(PS, config, tune_distributions=False).fit(frame)
        derived = schema.derive(PS, model.sample(2000, seed=5))
        return float((derived["discharge_day_of_week"] == "Sun").mean())

    assert sunday_share([cons.CONDITIONAL_MARGINALS]) < 0.01
    assert sunday_share([]) > 0.05


def test_constraints_skip_missing_columns():
    columns = [c for c in schema.training_columns(WD) if c not in ("elective_scheduled", "elective_cancelled")]
    assert len(cons.build_constraints(WD, columns, groups=[cons.INEQUALITY])) == 2
    assert cons.build_constraints(WD, ["ward_id", "month"]) == []
    # without the dropped columns' constants the occupancy band cannot be computed: skipped
    assert [c.label_column for c in cons.build_constraints(WD, columns, groups=[cons.REGIME_LABELS])] == [
        "staffing_level"
    ]


def test_segment_constants_feed_band_labels():
    # an AMU-like segment: ward_id is the segment key, electives/transfers in are constant zeros
    columns = [c for c in schema.training_columns(WD) if c not in ("ward_id",) + ELECTIVE_ZERO_COLUMNS]
    constants = {"ward_id": "AMU", "elective_scheduled": 0, "elective_cancelled": 0, "transfers_in": 0}
    built = cons.build_constraints(WD, columns, constants=constants)
    assert [type(c).__name__ for c in built] == (
        ["BandLabelConstraint"] * 2 + ["ConditionalQuantileConstraint", "EscalationCoverConstraint"]
        + ["ConditionalQuantileConstraint"] * 3 + ["Inequality"] * 2
    )
    assert built[0].constants == constants and built[1].constants == {"ward_id": "AMU"}
    assert built[2].constants == constants and built[3].constants == constants  # occupancy band / cover inputs
    assert built[2].group_columns == [] and built[2].band_groups == ("occupancy_level",)
    assert built[4].constants == {"ward_id": "AMU"} and built[4].band_groups == ("staffing_level",)
    assert built[5].group_columns == []  # the ward is the segment key: one group per segment
    assert built[6].group_columns == ["month"]
    described = cons.describe_constraints(WD, columns, constants=constants)
    assert [d["group"] for d in described] == (
        ["regime_labels"] * 2 + ["conditional_marginals", "capacity"] + ["conditional_marginals"] * 3
        + ["inequality"] * 2)
    assert "per segment" in described[5]["expression"]
    assert "per occupancy band (recomputed)" in described[2]["expression"]
    # a ward that never opens escalation beds has a constant escalation column: no cover constraint
    no_escalation = [c for c in columns if c != "escalation_beds_open"]
    assert cons.build_constraints(WD, no_escalation, groups=[cons.CAPACITY],
                                  constants={**constants, "escalation_beds_open": 0}) == []
    # a constant label column is not modelled: nothing to recompute
    no_label = [c for c in columns if c != "staffing_level"]
    assert [c.label_column for c in cons.build_constraints(WD, no_label, groups=[cons.REGIME_LABELS],
                                                           constants=constants)] == ["occupancy_level"]


def test_fixed_combinations_reduced_for_segments():
    groups = [cons.FIXED_COMBINATIONS]
    full = cons.describe_constraints(PS, schema.training_columns(PS), groups=groups)
    assert [d["columns"] for d in full] == [
        ["separation_type", "discharge_destination", "delay_reason"],
        ["ward_id", "admission_source"],
    ]
    without_ward = [c for c in schema.training_columns(PS) if c != "ward_id"]
    assert [d["columns"] for d in cons.describe_constraints(PS, without_ward, groups=groups)] == [
        ["separation_type", "discharge_destination", "delay_reason"]
    ]
    # a constant separation_type (e.g. PAED) keeps the remaining pair
    reduced = [c for c in without_ward if c != "separation_type"]
    assert cons.fixed_combination_groups(PS, reduced) == [("discharge_destination", "delay_reason")]
    assert cons.fixed_combination_groups(PS, reduced, groups=[cons.INEQUALITY]) == []
    assert cons.build_constraints(PS, ["delay_reason", "admission_source"], groups=groups) == []


def test_build_constraints_returns_fresh_objects():
    first = cons.build_constraints(PS, schema.training_columns(PS))
    second = cons.build_constraints(PS, schema.training_columns(PS))
    assert all(a is not b for a, b in zip(first, second))


# =========================================================================================
# Programmable constraints (unit level)
# =========================================================================================
def _scrambled_labels(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["occupancy_level"] = np.roll(out["occupancy_level"].to_numpy(), 3)
    out["staffing_level"] = np.roll(out["staffing_level"].to_numpy(), 5)
    return out


@pytest.mark.parametrize("label", ["occupancy_level", "staffing_level"])
def test_band_label_constraint_global_mode(label):
    data = make_ward_daily(n_days=20, wards=["AMU", "GMED", "PAED"], seed=3, with_dates=False)
    train = data[schema.training_columns(WD)]
    constraint = cons.BandLabelConstraint(label)
    constraint.fit(train, metadata=None)
    assert constraint.training_consistency == 1.0
    assert constraint.transform(train) is train
    assert constraint.get_updated_metadata("metadata") == "metadata"
    scrambled = _scrambled_labels(train)
    assert not constraint.is_valid(scrambled).all()
    restored = constraint.reverse_transform(scrambled)
    assert restored[label].tolist() == train[label].tolist()
    assert constraint.is_valid(restored).all()
    other = "staffing_level" if label == "occupancy_level" else "occupancy_level"
    assert restored[other].tolist() == scrambled[other].tolist()  # only its own label is touched
    params = constraint.get_constraint_dict()["parameters"]
    assert params == {"label_column": label, "constants": {}}


def test_band_label_constraint_with_segment_constants():
    data = make_ward_daily(n_days=30, wards=["AMU"], seed=4, with_dates=False)
    expected = data["occupancy_level"].tolist()
    dropped = ("ward_id",) + ELECTIVE_ZERO_COLUMNS
    segment = data[[c for c in schema.training_columns(WD) if c not in dropped]]
    constraint = cons.BandLabelConstraint(
        "occupancy_level", constants={"ward_id": "AMU", "elective_scheduled": np.int64(0),
                                      "elective_cancelled": 0, "transfers_in": 0})
    assert constraint.constants["elective_scheduled"] == 0 and type(constraint.constants["elective_scheduled"]) is int
    constraint.fit(segment, metadata=None)
    restored = constraint.reverse_transform(_scrambled_labels(segment))
    assert restored["occupancy_level"].tolist() == expected
    with pytest.raises(ValueError, match="neither modelled nor constant"):
        cons.BandLabelConstraint("occupancy_level", constants={"ward_id": "AMU"}).fit(segment, metadata=None)
    with pytest.raises(ValueError, match="label_column"):
        cons.BandLabelConstraint("season")


def test_band_label_constraint_leaves_undefined_rows_to_rejection():
    data = make_ward_daily(n_days=10, wards=["ICU"], seed=5, with_dates=False)[schema.training_columns(WD)]
    data = data.copy()
    data.loc[0, "beds_closed"] = 40  # no bed available: the band is undefined (WD-H08 violation)
    constraint = cons.BandLabelConstraint("occupancy_level")
    scrambled = data.copy()
    scrambled.loc[0, "occupancy_level"] = "low" if data.loc[0, "occupancy_level"] != "low" else "high"
    restored = constraint.reverse_transform(scrambled)
    assert restored.loc[0, "occupancy_level"] == scrambled.loc[0, "occupancy_level"]
    assert constraint.is_valid(restored).all()


def test_escalation_cover_constraint_round_trip_and_repair():
    data = make_ward_daily(n_days=60, wards=["AMU", "GMED"], seed=6, with_dates=False)[schema.training_columns(WD)]
    data = data.reset_index(drop=True)
    constraint = cons.EscalationCoverConstraint()
    constraint.fit(data, metadata=None)
    assert constraint.training_consistency == 1.0 and set(constraint.max_escalation) == {"AMU", "GMED"}
    transformed = constraint.transform(data)
    extra = transformed["escalation_beds_open"]
    assert extra.dtype == np.int64 and (extra >= 0).all() and (extra <= data["escalation_beds_open"]).all()
    # valid reference rows are reproduced exactly: escalation = cover + extra
    pd.testing.assert_frame_equal(constraint.reverse_transform(transformed), data)

    # a sampled AMU row whose census exceeds the base beds gets the escalation beds it needs
    sampled = transformed.copy()
    amu = sampled.index[sampled["ward_id"] == "AMU"][0]
    sampled.loc[amu, ["census_start", "escalation_beds_open", "beds_closed", "delayed_discharges"]] = [29, 0, 0, 0]
    sampled.loc[amu, ["discharges", "discharges_before_noon"]] = [10, 0]
    assert constraint.cover(sampled).loc[amu] >= 1 and not constraint.is_valid(sampled).loc[amu]
    repaired = constraint.reverse_transform(sampled)
    assert repaired.loc[amu, "escalation_beds_open"] >= 1 and constraint.is_valid(repaired).all()
    rates = _violation_rates(repaired, WD)
    assert rates["WD-H06"] == rates["WD-H07"] == rates["WD-H08"] == rates["WD-H15"] == 0.0
    # closed beds are covered too: closures only take free beds (WD-H15)
    sampled.loc[amu, "beds_closed"] = 1
    assert constraint.cover(sampled).loc[amu] == constraint.cover(repaired).loc[amu] + 1
    # a census beyond the ward's observed escalation range is capped and left to rejection
    sampled.loc[amu, "census_start"] = 28 + constraint.max_escalation["AMU"] + 3
    capped = constraint.reverse_transform(sampled)
    assert capped.loc[amu, "escalation_beds_open"] == constraint.max_escalation["AMU"]
    assert constraint.is_valid(capped).loc[amu]
    assert _violation_rates(capped, WD)["WD-H07"] > 0

    # segment mode: the ward and the constant electives / transfers come from the constants
    segment = data.loc[data["ward_id"] == "AMU", [c for c in data.columns if c not in ("ward_id",) + ELECTIVE_ZERO_COLUMNS]]
    constants = {"ward_id": "AMU", "elective_scheduled": 0, "elective_cancelled": 0, "transfers_in": 0}
    seg = cons.EscalationCoverConstraint(constants=constants)
    seg.fit(segment, metadata=None)
    assert seg.max_escalation == {"AMU": constraint.max_escalation["AMU"]}
    pd.testing.assert_frame_equal(seg.reverse_transform(seg.transform(segment)), segment)
    # frames without the census inputs (e.g. SDV condition frames) pass through unchanged
    condition = pd.DataFrame({"occupancy_level": ["high"]})
    assert seg.transform(condition) is condition and seg.reverse_transform(condition) is condition
    assert set(seg.get_constraint_dict()["parameters"]) == {"constants"}
    json.dumps(seg.learned())
    with pytest.raises(ValueError, match="neither modelled nor constant"):
        cons.EscalationCoverConstraint().fit(segment, metadata=None)


def test_structural_zeros_constraint_learns_frequent_zero_categories():
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({
        "cat": ["a"] * 40 + ["b"] * 40 + ["c"] * 5 + ["d"] * 40,
        "high": [0] * 40 + list(rng.integers(0, 4, 40)) + [0] * 5 + [0] * 40,
        "low": [0] * 40 + [0] * 40 + [0] * 5 + list(rng.integers(1, 3, 40)),
    })
    constraint = cons.StructuralZerosConstraint("cat", ["high", "low"], min_support=30, linked_pairs=[("low", "high")])
    constraint.fit(frame, metadata=None)
    # 'c' has too few rows; 'd' is forced on the low column because its high column is zeroed
    assert constraint.zero_categories == {"high": ["a", "d"], "low": ["a", "b", "d"]}
    sampled = pd.DataFrame({"cat": ["a", "b", "c", "d"], "high": [2, 3, 1, 2], "low": [1, 1, 1, 1]},
                           index=[10, 11, 12, 13])
    assert constraint.is_valid(sampled).tolist() == [False, False, True, False]
    fixed = constraint.reverse_transform(sampled)
    assert fixed["high"].tolist() == [0, 3, 1, 0] and fixed["low"].tolist() == [0, 0, 1, 0]
    assert list(fixed.index) == [10, 11, 12, 13] and constraint.is_valid(fixed).all()
    assert (fixed["low"] <= fixed["high"]).all()
    assert constraint.transform(sampled) is sampled
    assert constraint.learned()["zero_categories"]["high"] == ["a", "d"]
    json.dumps(constraint.learned())
    assert set(constraint.get_constraint_dict()["parameters"]) == {
        "category_column", "numeric_columns", "min_support", "linked_pairs"}
    with pytest.raises(ValueError):
        cons.StructuralZerosConstraint("cat", [])


def _grouped_counts(seed: int = 0) -> pd.DataFrame:
    """'zero' rows are always 0, 'flat' rows uniform 0-4, 'heavy' rows mostly 0 with a long tail,
    'rare' has too few rows for its own marginal; ``low <= high`` holds in every row."""
    rng = np.random.default_rng(seed)
    groups = ["zero"] * 300 + ["flat"] * 300 + ["heavy"] * 300 + ["rare"] * 10
    high = np.concatenate([np.zeros(300), rng.integers(0, 5, 300), rng.choice([0] * 9 + [6], 300) * rng.integers(1, 4, 300),
                           rng.integers(1, 3, 10)]).astype(int)
    low = np.minimum(high, rng.integers(0, 2, len(high)))
    return pd.DataFrame({"g": groups, "high": high, "low": low, "other": rng.normal(size=len(high))})


def test_conditional_quantile_constraint_transform_and_exact_marginals():
    from scipy.stats import norm

    data = _grouped_counts()
    constraint = cons.ConditionalQuantileConstraint(["high", "low"], group_columns=["g"], bounds=[("low", "high")],
                                                    min_support=30, seed=3)
    constraint.fit(data, metadata=None)
    assert set(constraint.tables["high"]) >= {("zero",), ("flat",), ("heavy",)}
    assert ("rare",) not in constraint.tables["high"]          # too few rows: pooled non-zero fallback
    scores = constraint.transform(data)
    pd.testing.assert_frame_equal(scores, constraint.transform(data))  # seeded: repeated calls agree
    assert scores["other"].equals(data["other"]) and scores["g"].equals(data["g"])
    for group in ("zero", "flat", "heavy"):   # standard-normal scores within every group
        part = scores.loc[data["g"] == group, "high"]
        assert abs(part.mean()) < 0.15 and 0.85 < part.std() < 1.15
    # the scores map back to the training values exactly
    restored = constraint.reverse_transform(scores)
    assert restored["high"].tolist() == data["high"].tolist() and restored["low"].tolist() == data["low"].tolist()
    assert restored["high"].dtype == np.int64

    # fresh standard-normal scores reproduce each group's marginal (point masses included)
    rng = np.random.default_rng(9)
    sampled = pd.DataFrame({"g": np.repeat(["zero", "flat", "heavy", "rare"], 4000),
                            "high": rng.normal(size=16000), "low": rng.normal(size=16000), "other": 0.0})
    out = constraint.reverse_transform(sampled)
    assert (out.loc[out["g"] == "zero", "high"] == 0).all()
    for group in ("flat", "heavy"):
        expected = data.loc[data["g"] == group, "high"].value_counts(normalize=True).sort_index()
        observed = out.loc[out["g"] == group, "high"].value_counts(normalize=True).reindex(expected.index, fill_value=0)
        assert (observed - expected).abs().max() < 0.03
    # the rare group borrows the marginal of the groups that are not all zero (never the zero group's)
    rare = out.loc[out["g"] == "rare", "high"]
    pooled_zero_share = data.loc[data["g"].isin(["flat", "heavy"]), "high"].eq(0).mean()
    assert abs(rare.eq(0).mean() - pooled_zero_share) < 0.05
    assert (out["low"] <= out["high"]).all() and constraint.is_valid(out).all()
    assert not constraint.is_valid(pd.DataFrame({"g": ["flat"], "high": [1], "low": [2]})).any()
    # condition frames without the numeric columns pass through untouched
    condition = pd.DataFrame({"g": ["flat"]})
    assert constraint.transform(condition) is condition and constraint.reverse_transform(condition) is condition
    assert norm.cdf(scores["high"]).min() > 0


def test_conditional_quantile_constraint_metadata_state_and_errors():
    from sdv.metadata import Metadata

    data = _grouped_counts(1)
    constraint = cons.ConditionalQuantileConstraint(["high"], group_columns=["g"])
    constraint.fit(data, metadata=None)
    metadata = Metadata.load_from_dict({"tables": {"t": {"columns": {
        "g": {"sdtype": "categorical"}, "high": {"sdtype": "numerical", "decimal_places": 0},
        "low": {"sdtype": "numerical", "decimal_places": 0}, "other": {"sdtype": "numerical"}}}}})
    updated = constraint.get_updated_metadata(metadata).to_dict()["tables"]["t"]["columns"]
    # scores must not be rounded to whole numbers by SDV's formatter
    assert updated["high"] == {"sdtype": "numerical"} and updated["low"] == {"sdtype": "numerical", "decimal_places": 0}
    learned = constraint.learned()
    json.dumps(learned)
    assert learned["group_columns"] == ["g"] and learned["marginals"]["high"]["zero"]["zero_share"] == 1.0
    assert learned["group_sizes"]["rare"] == 10
    assert set(constraint.get_constraint_dict()["parameters"]) == {"columns", "group_columns", "bounds",
                                                                   "min_support", "seed", "band_groups",
                                                                   "constants"}
    pooled = cons.ConditionalQuantileConstraint(["high"])   # no group columns: one marginal
    pooled.fit(data, metadata=None)
    assert set(pooled.tables["high"]) >= {()}
    # a small group whose rows are all exact zeros keeps its structural zero (e.g. no Saturday lists)
    tiny = pd.concat([data, pd.DataFrame({"g": ["tiny"] * 12, "high": [0] * 12, "low": [0] * 12, "other": 0.0})],
                     ignore_index=True)
    small = cons.ConditionalQuantileConstraint(["high"], group_columns=["g"], min_support=30)
    small.fit(tiny, metadata=None)
    assert ("tiny",) in small.tables["high"] and ("rare",) not in small.tables["high"]
    draws = small.reverse_transform(pd.DataFrame({"g": ["tiny"] * 200, "high": np.linspace(-3, 3, 200)}))
    assert (draws["high"] == 0).all()
    with pytest.raises(ValueError, match="at least one"):
        cons.ConditionalQuantileConstraint([])
    with pytest.raises(ValueError, match="does not transform"):
        cons.ConditionalQuantileConstraint(["high"], bounds=[("low", "high")])


# =========================================================================================
# Helpers and configuration
# =========================================================================================
def test_largest_remainder_properties():
    assert largest_remainder(10, [1, 1, 1]) == [4, 3, 3]
    assert sum(largest_remainder(1000, [0.2, 0.3, 0.5])) == 1000
    assert largest_remainder(7, [0, 0]) == [0, 0]
    assert largest_remainder(5, [1, np.nan, -2, 3]) == [1, 0, 0, 4]
    assert largest_remainder(0, [1, 2]) == [0, 0]


def test_derive_seed_is_deterministic_and_spreads():
    assert derive_seed(42, "round", 1) == derive_seed(42, "round", 1)
    seeds = {derive_seed(42, "round", i) for i in range(50)}
    assert len(seeds) == 50
    assert all(0 <= s < 2**31 for s in seeds)
    assert derive_seed(42, "a") != derive_seed(43, "a")


def test_json_safe_converts_numpy_and_nan():
    payload = json_safe({"a": np.int64(3), "b": np.float32(1.5), "c": float("nan"), "d": (np.bool_(True), None)})
    assert payload == {"a": 3, "b": 1.5, "c": None, "d": [True, None]}
    json.dumps(payload)


def test_experiment_config_from_settings_quick():
    config = ExperimentConfig.from_settings("ctgan", quick=True)
    assert config.model == "ctgan" and config.params["epochs"] == 10
    with pytest.raises(KeyError):
        ExperimentConfig.from_settings("nope")


def test_ablation_experiments_in_settings():
    experiments = list(get_settings()["synthesis"]["experiments"])
    assert experiments.index("gc_baseline") < experiments.index("gc_segmented") < experiments.index("gc_tuned")
    baseline = ExperimentConfig.from_settings("gc_baseline", table=WD)
    assert baseline.constraint_groups == () and baseline.segment_by is None
    segmented = ExperimentConfig.from_settings("gc_segmented", table=PS)
    assert segmented.constraint_groups == cons.BUILTIN_GROUPS and segmented.segment_by == "ward_id"
    assert segmented.constraints == ["inequality", "fixed_combinations"]
    tuned_wd = ExperimentConfig.from_settings("gc_tuned", table=WD)
    tuned_ps = ExperimentConfig.from_settings("gc_tuned", table=PS)
    assert tuned_wd.segment_by == "ward_id" and tuned_ps.segment_by == ["ward_id", "admission_source"]
    assert tuned_ps.constraint_groups == cons.CONSTRAINT_GROUPS
    for experiment_id in ("ctgan", "tvae"):  # GAN/VAE inherit every group through constraints: true
        assert ExperimentConfig.from_settings(experiment_id).constraint_groups == cons.CONSTRAINT_GROUPS
    unresolved = ExperimentConfig.from_settings("gc_tuned")
    assert isinstance(unresolved.segment_by, dict)
    assert unresolved.segment_columns(PS) == ("ward_id", "admission_source")
    model = BedSynthSynthesizer(PS, unresolved)
    assert model.segment_columns == ("ward_id", "admission_source") and model.config.segment_by == ["ward_id", "admission_source"]
    assert BedSynthSynthesizer(WD, "gc_tuned").segment_by == "ward_id"
    json.dumps(unresolved.to_dict())


def test_experiment_settings_with_unknown_constraint_group(monkeypatch):
    settings = {"synthesis": {"experiments": {"bad": {"model": "gaussian_copula", "constraints": ["typo"]}}}}
    monkeypatch.setattr("bedsynth.synthesis.models.get_settings", lambda: settings)
    with pytest.raises(ValueError, match="Unknown constraint group"):
        ExperimentConfig.from_settings("bad")


@pytest.mark.parametrize(
    "setting, stored, groups",
    [
        ("none", [], ()),                                        # used to enable every group
        ("regime_labels", ["regime_labels"], ("regime_labels",)),  # used to enable every group
        ("all", list(cons.CONSTRAINT_GROUPS), cons.CONSTRAINT_GROUPS),
        (["capacity", "inequality"], ["inequality", "capacity"], ("inequality", "capacity")),
        (True, True, cons.CONSTRAINT_GROUPS),
        (False, False, ()),
        (None, False, ()),
    ],
)
def test_experiment_constraints_setting_strings(monkeypatch, setting, stored, groups):
    settings = {"synthesis": {"experiments": {"x": {"model": "gaussian_copula", "constraints": setting}}}}
    monkeypatch.setattr("bedsynth.synthesis.models.get_settings", lambda: settings)
    config = ExperimentConfig.from_settings("x")
    assert config.constraints == stored and config.constraint_groups == groups
    assert BedSynthSynthesizer(WD, config).constraint_groups == groups


def test_segment_columns_of():
    assert segment_columns_of(None) == () and segment_columns_of("") == ()
    assert segment_columns_of("ward_id") == ("ward_id",)
    assert segment_columns_of(["ward_id", "admission_source"]) == ("ward_id", "admission_source")
    assert segment_columns_of({PS: ["ward_id"]}, WD) == ()
    with pytest.raises(ValueError, match="given table"):
        segment_columns_of({WD: "ward_id"})
    with pytest.raises(ValueError, match="more than once"):
        segment_columns_of(["ward_id", "ward_id"])
    with pytest.raises(ValueError, match="column name"):
        segment_columns_of(3)


def test_invalid_wrapper_arguments():
    with pytest.raises(KeyError):
        BedSynthSynthesizer("bogus", "gc_baseline")
    with pytest.raises(ValueError):
        BedSynthSynthesizer(WD, _config("x", model="bogus"))
    with pytest.raises(ValueError):
        BedSynthSynthesizer(WD, _config("x", segment_by="census_start"))
    with pytest.raises(ValueError, match="categorical train columns"):
        BedSynthSynthesizer(WD, _config("x", segment_by=["ward_id", "census_start"]))
    with pytest.raises(ValueError, match="Unknown constraint group"):
        BedSynthSynthesizer(WD, _config("x", constraints=["bogus"]))
    with pytest.raises(RuntimeError):
        BedSynthSynthesizer(WD, "gc_baseline").sample(5)


# =========================================================================================
# Global GaussianCopula
# =========================================================================================
def _assert_output_contract(frame: pd.DataFrame, table: str) -> None:
    assert list(frame.columns) == schema.training_columns(table)
    assert not frame.isna().any().any()
    for spec in schema.column_specs(table, ["train"]):
        dtype = frame[spec.name].dtype
        if spec.sdtype == "numerical" and spec.dtype == "int":
            assert dtype == np.int64, spec.name
        elif spec.sdtype == "numerical":
            assert dtype == np.float64, spec.name
            assert np.allclose(frame[spec.name], frame[spec.name].round(spec.decimals)), spec.name
        elif spec.name in ("month", "admission_month"):
            assert dtype == np.int64, spec.name
        else:
            assert dtype == object, spec.name
        if spec.categories:
            assert set(frame[spec.name]).issubset(set(spec.categories)), spec.name


def test_global_sample_output_contract(wd_global):
    out = wd_global.sample(300, seed=1)
    assert len(out) == 300
    _assert_output_contract(out, WD)
    assert (out["elective_cancelled"] <= out["elective_scheduled"]).all()
    assert (out["discharges_before_noon"] <= out["discharges"]).all()
    assert (out["delayed_discharges"] <= out["census_start"]).all()


def test_programmable_constraints_raise_raw_validity(wd_global, wd_global_builtin, ward_daily_df):
    sample = schema.derive(WD, wd_global.sample(1500, seed=2))
    with_labels = _violation_rates(sample, WD)
    builtin_only = _violation_rates(wd_global_builtin.sample(1500, seed=2), WD)
    # band labels recomputed from their counts and learned structural zeros hold by construction
    for rule in ("WD-H08", "WD-H10", "WD-H11", "WD-H12"):
        assert with_labels[rule] == 0.0, rule
    assert builtin_only["WD-H10"] > 0.05 and builtin_only["WD-H11"] > 0.05
    # the escalation cover fixes every census the ward's escalation range can hold; a global copula
    # can still propose e.g. an ICU census far above 12 beds: capped at the ward maximum, left to rejection
    short = sample["census_start"] > sample["bed_capacity"] + sample["escalation_beds_open"]
    ward_maximum = sample.loc[short, "ward_id"].map(ward_daily_df.groupby("ward_id")["escalation_beds_open"].max())
    assert (sample.loc[short, "escalation_beds_open"] == ward_maximum).all()
    assert with_labels["WD-H07"] < builtin_only["WD-H07"]
    out = wd_global.sample(1500, seed=3)
    assert (out.loc[out["day_of_week"].isin(schema.WEEKEND_DAYS), "elective_scheduled"] == 0).all()
    card = wd_global.model_card()
    assert card["constraint_groups"] == list(cons.CONSTRAINT_GROUPS)
    zeros = [c for c in card["segments"][0]["constraints"] if c["group"] == cons.STRUCTURAL_ZEROS]
    learned = {c["columns"][0]: c["learned"]["zero_categories"] for c in zeros}
    assert set(learned["ward_id"]["elective_scheduled"]) == {"AMU", "ELDC"}
    assert set(learned["ward_id"]["transfers_in"]) == {"AMU", "PAED"}
    assert set(learned["day_of_week"]["elective_scheduled"]) == {"Sat", "Sun"}
    json.dumps(card)


def test_seeded_reproducibility(wd_global):
    first = wd_global.sample(200, seed=7)
    pd.testing.assert_frame_equal(first, wd_global.sample(200, seed=7))
    assert not first.equals(wd_global.sample(200, seed=8))


def test_conditions_honoured(wd_global):
    normal = wd_global.sample(150, {"occupancy_level": "normal"}, seed=3)
    assert len(normal) > 0 and (normal["occupancy_level"] == "normal").all()
    derived = schema.derive(WD, normal)
    assert ((derived["occupancy_rate"] >= 0.75) & (derived["occupancy_rate"] < 0.90)).all()
    cell = wd_global.sample(40, {"ward_id": "ICU", "day_of_week": "Mon", "month": 2}, seed=3)
    assert len(cell) > 0
    assert (cell["ward_id"] == "ICU").all() and (cell["day_of_week"] == "Mon").all() and (cell["month"] == 2).all()
    # month given as a string is coerced to the training dtype
    as_text = wd_global.sample(20, {"month": "3"}, seed=4)
    assert len(as_text) > 0 and (as_text["month"] == 3).all()


def test_infeasible_conditions_return_zero_rows(wd_global):
    # the fixture covers January-April only: September was never observed
    for conditions in ({"month": 9}, {"ward_id": "XYZ"}, {"staffing_level": "unknown"}):
        out = wd_global.sample(25, conditions, seed=1)
        assert len(out) == 0
        assert list(out.columns) == schema.training_columns(WD)
    assert not wd_global.is_feasible({"month": 9})
    assert wd_global.is_feasible({"month": 2, "ward_id": "AMU"})


def test_invalid_condition_arguments(wd_global):
    with pytest.raises(ValueError, match="cannot be used as a condition"):
        wd_global.sample(5, {"census_start": 3})
    with pytest.raises(ValueError, match="single value"):
        wd_global.sample(5, {"month": [1, 2]})
    empty = wd_global.sample(0, seed=1)
    assert empty.empty and list(empty.columns) == schema.training_columns(WD)
    assert empty["month"].dtype == np.int64


def test_condition_probability_and_support(wd_global, ward_daily_df):
    assert wd_global.condition_probability(None) == 1.0
    expected = (ward_daily_df["ward_id"] == "AMU").mean()
    assert wd_global.condition_probability({"ward_id": "AMU"}) == pytest.approx(expected)
    support = wd_global.training_support(["ward_id", "occupancy_level"])
    assert support["count"].sum() == len(ward_daily_df)
    observed = ward_daily_df.groupby(["ward_id", "occupancy_level"]).size()
    merged = support.set_index(["ward_id", "occupancy_level"])["count"]
    assert merged.sort_index().to_dict() == observed.sort_index().to_dict()


def test_ordered_encodings_applied(wd_global, ward_daily_df):
    from rdt.transformers import OrderedUniformEncoder

    segment = wd_global._segments[None]
    month_order = segment.ordered_encodings["month"]
    occupancy = schema.derive(WD, ward_daily_df).groupby("month")["occupancy_rate"].mean()
    assert month_order == list(occupancy.sort_values(kind="mergesort").index)
    assert segment.ordered_encodings["ward_id"][0] == "ICU"  # smallest ward first
    transformers = segment.synthesizer.get_transformers()
    assert isinstance(transformers["month"], OrderedUniformEncoder)
    assert list(transformers["month"].order) == month_order


# =========================================================================================
# Segmented models (one segment column)
# =========================================================================================
def test_segmented_model_card(wd_segmented):
    card = wd_segmented.model_card()
    json.dumps(card)
    assert card["segment_by"] == "ward_id" and card["n_segments"] == 4
    assert card["segment_columns"] == ["ward_id"] and card["segment_label"] == "ward_id"
    assert card["constraint_groups"] == list(cons.BUILTIN_GROUPS)
    assert {s["segment"] for s in card["segments"]} == {"AMU", "GMED", "ICU", "PAED"}
    for key in ("table", "experiment", "sdv_version", "training_rows", "fit_seconds", "segments",
                "constraints", "training_columns", "fitted_at", "distributions"):
        assert key in card
    amu = next(s for s in card["segments"] if s["segment"] == "AMU")
    assert amu["segment_values"] == {"ward_id": "AMU"}
    assert set(ELECTIVE_ZERO_COLUMNS) <= set(amu["constant_columns"])
    # the elective inequality references dropped constant columns, so it is skipped
    assert all("elective_scheduled" not in c["columns"] for c in amu["constraints"])
    assert len(amu["constraints"]) == 2
    assert "month" in amu["ordered_encodings"] and "day_of_week" in amu["ordered_encodings"]


def test_segmented_programmable_constraints_use_segment_constants(wd_tuned):
    card = wd_tuned.model_card()
    amu = next(s for s in card["segments"] if s["segment"] == "AMU")
    assert [c["group"] for c in amu["constraints"]] == (
        ["regime_labels"] * 2 + ["conditional_marginals", "capacity"] + ["conditional_marginals"] * 3
        + ["inequality"] * 2)
    occupancy = amu["constraints"][0]
    assert occupancy["learned"]["constants"] == {"ward_id": "AMU", "elective_scheduled": 0,
                                                 "elective_cancelled": 0, "transfers_in": 0}
    assert occupancy["learned"]["training_label_consistency"] == 1.0
    assert amu["constraints"][3]["learned"]["training_consistency"] == 1.0
    rates = _violation_rates(wd_tuned.sample(1200, seed=4), WD)
    for rule in ("WD-H08", "WD-H10", "WD-H11", "WD-H12"):
        assert rates[rule] == 0.0, rule
    for rule in ("WD-H07", "WD-H15"):  # by construction within the ward's observed escalation range
        assert rates[rule] < 0.005, rule
    amu_rows = wd_tuned.sample(100, {"ward_id": "AMU"}, seed=4)
    assert len(amu_rows) == 100 and (amu_rows[list(ELECTIVE_ZERO_COLUMNS)] == 0).all().all()


def test_conditional_marginals_keep_weekday_lists_and_zero_shares(wd_tuned, ward_daily_df):
    """S3/S7: elective lists per weekday and mostly-zero counts follow the training data."""
    subset = ward_daily_df[ward_daily_df["ward_id"].isin(["AMU", "GMED", "ICU", "PAED"])]
    out = wd_tuned.sample(4000, seed=12)
    assert (out.loc[out["day_of_week"].isin(schema.WEEKEND_DAYS), "elective_scheduled"] == 0).all()
    electives = out[out["ward_id"] != "AMU"]
    weekday = electives[electives["day_of_week"].isin(schema.WEEKDAYS)].groupby("day_of_week")["elective_scheduled"].mean()
    assert len(weekday) == 5 and weekday.std() / weekday.mean() < 0.2   # flat Mon-Fri lists, no gradient
    reference = subset[subset["ward_id"] != "AMU"]
    assert abs(electives["elective_scheduled"].mean() - reference["elective_scheduled"].mean()) < 0.25
    for column, tolerance in (("beds_closed", 0.05), ("emergency_boarders", 0.03), ("elective_cancelled", 0.05)):
        assert abs((out[column] > 0).mean() - (subset[column] > 0).mean()) < tolerance, column
    assert (out["elective_cancelled"] <= out["elective_scheduled"]).all()
    card = wd_tuned.model_card()
    gmed = next(s for s in card["segments"] if s["segment"] == "GMED")
    marginals = [c for c in gmed["constraints"] if c["group"] == cons.CONDITIONAL_MARGINALS]
    assert [c["group_by"] for c in marginals] == [[], ["day_of_week"], [], [], ["month"]]
    assert marginals[1]["learned"]["marginals"]["elective_scheduled"]["Sun"]["zero_share"] == 1.0


def test_band_grouped_marginals_follow_the_recomputed_occupancy_band(wd_tuned, ward_daily_df):
    """ED boarders are restored per ward x occupancy band, the band recomputed from the final counts.

    In the fixture boarders only appear on full ward-days; every band whose training rows (>= 10)
    never board must come out without boarders, and the band the marginal used is the band the
    occupancy label gets (so WD-S03, boarding only at >= 90 % occupancy, holds on those bands).
    """
    train = schema.derive(WD, ward_daily_df[ward_daily_df["ward_id"].isin(["GMED", "ICU", "PAED"])])
    out = schema.derive(WD, wd_tuned.sample(3000, seed=21))
    out = out[out["ward_id"].isin(["GMED", "ICU", "PAED"])]
    groups = train.groupby(["ward_id", "occupancy_level"])["emergency_boarders"]
    never = [key for key, rows in groups if len(rows) >= 10 and (rows == 0).all()]
    assert len(never) >= 9
    for ward, band in never:
        rows = out[(out["ward_id"] == ward) & (out["occupancy_level"] == band)]
        assert (rows["emergency_boarders"] == 0).all(), (ward, band)
    assert (out["occupancy_level"] == cons.band_labels(out, "occupancy_level")).all()
    card = wd_tuned.model_card()
    gmed = next(s for s in card["segments"] if s["segment"] == "GMED")
    ed = next(c for c in gmed["constraints"] if "emergency_boarders" in c["columns"])
    assert ed["band_groups"] == ["occupancy_level"] and ed["learned"]["band_groups"] == ["occupancy_level"]


def test_band_groups_restore_values_from_the_band_of_the_restored_counts(ward_daily_df):
    """A band-grouped marginal maps each score back with the quantile function of the band that the
    counts give at restore time, so changing the counts moves the row to another group."""
    rows = ward_daily_df[ward_daily_df["ward_id"] == "GMED"].reset_index(drop=True)
    gmed = rows.drop(columns=["ward_id"])
    critical = schema.derive(WD, rows)["occupancy_level"].eq("critical").to_numpy()
    gmed["emergency_boarders"] = np.where(critical, 3, 0)  # board only when full
    constraint = cons.ConditionalQuantileConstraint(
        ["emergency_boarders"], band_groups=["occupancy_level"], constants={"ward_id": "GMED"}, min_support=1)
    constraint.fit(gmed, None)
    scores = constraint.transform(gmed)
    assert scores["emergency_boarders"].between(-6, 6).all()
    restored = constraint.reverse_transform(scores)
    assert (restored["emergency_boarders"] == gmed["emergency_boarders"]).all()
    # empty the ward by the end of the day: every row is now 'low', whose training rows never board
    emptied = scores.copy()
    emptied[["census_start", "admissions_emergency", "transfers_in", "transfers_out", "discharges",
             "elective_scheduled", "elective_cancelled"]] = 0
    assert (constraint.reverse_transform(emptied)["emergency_boarders"] == 0).all()
    # frames without the band inputs (SDV condition frames) pass through unchanged
    condition = pd.DataFrame({"emergency_boarders": [1.0]})
    assert constraint.transform(condition).equals(condition)
    # constraints pickled before band groups existed still work (class defaults)
    legacy = cons.ConditionalQuantileConstraint(["emergency_boarders"], min_support=1)
    legacy.fit(gmed, None)
    del legacy.__dict__["band_groups"], legacy.__dict__["constants"]
    restored = legacy.reverse_transform(legacy.transform(gmed))
    assert (restored["emergency_boarders"] == gmed["emergency_boarders"]).all()
    with pytest.raises(ValueError, match="band_groups"):
        cons.ConditionalQuantileConstraint(["emergency_boarders"], band_groups=["season"])


def test_closed_beds_follow_the_staffing_band_of_the_restored_nurses(ward_daily_df):
    """Closed beds are grouped by the staffing band recomputed from the day-shift nurses (DESIGN.md
    §7.2). The nurse marginal (per ward x month) is listed after them, so its reverse transform runs
    first and the band is read from the restored nurses; patient_stays specs use no band groups."""
    marginals = [s for s in cons.CONSTRAINT_SPECS[WD] if s.kind == cons.CONDITIONAL_MARGINALS]
    closed = next(i for i, s in enumerate(marginals) if s.columns == ("beds_closed",))
    nurses = next(i for i, s in enumerate(marginals) if s.columns == ("nurses_day_shift",))
    assert marginals[closed].band_groups == ("staffing_level",) and closed < nurses
    assert marginals[nurses].group_by == ("ward_id", "month") and not marginals[nurses].band_groups
    assert not any(s.band_groups for s in cons.CONSTRAINT_SPECS[PS])
    # a band-grouped spec needs the inputs of its band modelled or constant
    columns = schema.training_columns(WD)
    assert marginals[closed].applicable_columns([c for c in columns if c != "nurses_day_shift"]) is None
    assert marginals[closed].applicable_columns([c for c in columns if c != "ward_id"],
                                                {"ward_id": "GMED"}) == ("beds_closed",)

    rows = ward_daily_df[ward_daily_df["ward_id"] == "GMED"].reset_index(drop=True)
    gmed = rows.drop(columns=["ward_id"])
    band = cons.band_labels(gmed, "staffing_level", {"ward_id": "GMED"})
    assert band.equals(schema.staffing_levels(rows["nurses_day_shift"], rows["ward_id"]))
    short = band.eq("short").to_numpy()
    assert 0 < short.sum() < len(short) and band.eq("adequate").any()
    gmed["beds_closed"] = np.where(short, 2, 0)             # beds are closed only on short-staffed days
    constraint = cons.ConditionalQuantileConstraint(
        ["beds_closed"], band_groups=["staffing_level"], constants={"ward_id": "GMED"}, min_support=1)
    constraint.fit(gmed, None)
    scores = constraint.transform(gmed)
    assert (constraint.reverse_transform(scores)["beds_closed"] == gmed["beds_closed"]).all()
    # the same scores restored with other (restored) nurse counts follow the band of those counts
    adequate, understaffed = scores.copy(), scores.copy()
    adequate["nurses_day_shift"] = int(rows.loc[band.eq("adequate"), "nurses_day_shift"].min())
    understaffed["nurses_day_shift"] = int(rows.loc[short, "nurses_day_shift"].max())
    assert (constraint.reverse_transform(adequate)["beds_closed"] == 0).all()
    assert (constraint.reverse_transform(understaffed)["beds_closed"] == 2).all()
    learned = constraint.learned()
    assert learned["band_groups"] == ["staffing_level"]
    assert learned["marginals"]["beds_closed"]["adequate"]["zero_share"] == 1.0
    assert learned["marginals"]["beds_closed"]["short"]["zero_share"] == 0.0


def test_conditional_marginals_model_delay_hours_per_reason(patient_stays_df):
    """S2/S5: delay hours follow their reason (exact zeros without one) without inflating stays."""
    data = patient_stays_df.copy()
    scale = {"awaiting_documentation": 3.0, "awaiting_transport": 0.4, "awaiting_medication": 1.0}
    factor = data["delay_reason"].map(scale).fillna(1.0)
    delayed = data["delay_reason"] != "none"
    data.loc[delayed, "discharge_delay_hours"] = (data.loc[delayed, "discharge_delay_hours"] * factor[delayed]).clip(4.0).round(1)
    data["los_nights"] = np.maximum(data["los_nights"], np.ceil(data["discharge_delay_hours"] / 24).astype(int) + 1)
    config = _config("ps_reason", constraints=True, ordinal_metadata=True, ordered_encodings=True)
    model = BedSynthSynthesizer(PS, config, tune_distributions=False).fit(data)
    raw = model.sample(6000, seed=21)
    assert (raw.loc[raw["delay_reason"] == "none", "discharge_delay_hours"] == 0).all()
    assert (raw.loc[raw["delay_reason"] != "none", "discharge_delay_hours"] >= 4.0).all()   # PS-H07/H13 by construction
    rates = _violation_rates(raw, PS)
    assert rates["PS-H07"] == 0.0 and rates["PS-H13"] == 0.0
    for reason in scale:
        expected = data.loc[data["delay_reason"] == reason, "discharge_delay_hours"].mean()
        observed = raw.loc[raw["delay_reason"] == reason, "discharge_delay_hours"].mean()
        assert abs(observed / expected - 1) < 0.15, reason
    # no repair overwrites a jointly sampled delay: the delayed share is the model's own, and
    # delayed stays stay longer than the others, as in the training data
    assert abs((raw["delay_reason"] != "none").mean() - delayed.mean()) < 0.05
    stays = schema.derive(PS, raw)
    assert stays.loc[stays["discharge_delay_hours"] > 0, "los_nights"].mean() > \
        stays.loc[stays["discharge_delay_hours"] == 0, "los_nights"].mean()
    # without the conditional marginals the copula smears the zero mass: most rows get delay hours
    groups = [g for g in cons.CONSTRAINT_GROUPS if g != cons.CONDITIONAL_MARGINALS]
    smeared = BedSynthSynthesizer(PS, _config("ps_smeared", constraints=groups, ordinal_metadata=True,
                                              ordered_encodings=True), tune_distributions=False).fit(data)
    assert _violation_rates(smeared.sample(2000, seed=21), PS)["PS-H07"] > 0.3


def test_programmable_constraints_survive_save_load(wd_tuned, tmp_path):
    loaded = BedSynthSynthesizer.load(wd_tuned.save(tmp_path / "tuned"))
    pd.testing.assert_frame_equal(loaded.sample(150, seed=6), wd_tuned.sample(150, seed=6))
    assert loaded.constraint_groups == cons.CONSTRAINT_GROUPS
    synthesizer = loaded._segments["GMED"].synthesizer
    names = [type(c).__name__ for c in synthesizer.get_constraints()]
    assert names.count("BandLabelConstraint") == 2
    json.dumps([c.get_constraint_dict() for c in synthesizer.get_constraints()])


def test_segment_routing_and_constants(wd_segmented):
    amu = wd_segmented.sample(120, {"ward_id": "AMU"}, seed=5)
    assert len(amu) == 120 and (amu["ward_id"] == "AMU").all()
    assert (amu["elective_scheduled"] == 0).all() and (amu["transfers_in"] == 0).all()
    _assert_output_contract(amu, WD)
    assert len(wd_segmented.sample(10, {"ward_id": "CARD"}, seed=5)) == 0  # ward not trained


def test_unconditional_allocation_follows_training_shares(wd_segmented):
    out = wd_segmented.sample(400, seed=9)
    assert out["ward_id"].value_counts().to_dict() == {w: 100 for w in ("AMU", "GMED", "ICU", "PAED")}
    # rows from different segments are interleaved, not blocked by ward
    assert out["ward_id"].iloc[:20].nunique() > 1


def test_conditions_without_ward_follow_segment_support(wd_segmented, ward_daily_df):
    subset = ward_daily_df[ward_daily_df["ward_id"].isin(["AMU", "GMED", "ICU", "PAED"])]
    supported = set(subset.loc[subset["occupancy_level"] == "critical", "ward_id"])
    out = wd_segmented.sample(150, {"occupancy_level": "critical"}, seed=2)
    assert len(out) > 0 and (out["occupancy_level"] == "critical").all()
    assert set(out["ward_id"]) <= supported


def test_unseen_ordinal_levels_are_not_generated(wd_segmented, ward_daily_df):
    for segment in wd_segmented._segments.values():
        seen = set(ward_daily_df.loc[ward_daily_df["ward_id"] == segment.key, "occupancy_level"])
        order = segment.ordered_encodings.get("occupancy_level")
        if seen != set(schema.OCCUPANCY_LEVELS):
            assert order is not None and set(order) == seen
        out = wd_segmented.sample(100, {"ward_id": segment.key}, seed=1)
        assert set(out["occupancy_level"]) <= seen


def test_sample_many_serves_disjoint_cells(wd_segmented):
    requests = [({"ward_id": "GMED", "month": m, "day_of_week": d}, 3) for m in (1, 2) for d in schema.DAYS_OF_WEEK]
    frames = wd_segmented.sample_many(requests, seed=11)
    assert len(frames) == len(requests)
    for (conditions, k), frame in zip(requests, frames):
        assert len(frame) <= k
        for column, value in conditions.items():
            assert (frame[column] == value).all()
    assert sum(len(f) for f in frames) >= 0.8 * sum(k for _, k in requests)
    again = wd_segmented.sample_many(requests, seed=11)
    for a, b in zip(frames, again):
        pd.testing.assert_frame_equal(a, b)


def test_save_load_round_trip(wd_segmented, isolated_paths):
    directory = wd_segmented.save(isolated_paths["models_dir"] / WD / "seg")
    assert (directory / "synthesizer.pkl").exists() and (directory / "model_card.json").exists()
    card = json.loads((directory / "model_card.json").read_text(encoding="utf-8"))
    assert card["experiment_id"] == "gc_segmented" and card["n_segments"] == 4
    loaded = BedSynthSynthesizer.load(directory)
    assert loaded.is_fitted and loaded.segments == wd_segmented.segments
    pd.testing.assert_frame_equal(loaded.sample(200, seed=4), wd_segmented.sample(200, seed=4))
    with pytest.raises(FileNotFoundError):
        BedSynthSynthesizer.load(isolated_paths["models_dir"] / "missing")


def test_pickles_without_segment_columns_or_groups_still_load(wd_segmented):
    """Models pickled before multi-column segmentation / constraint groups are upgraded on load."""
    legacy = cloudpickle.loads(cloudpickle.dumps(wd_segmented))
    del legacy.__dict__["_segment_columns"], legacy.__dict__["_constraint_groups"]
    legacy.config.constraints = True  # old cards stored a plain boolean (built-in constraints only)
    restored = cloudpickle.loads(cloudpickle.dumps(legacy))
    assert restored.segment_columns == ("ward_id",) and restored.segment_by == "ward_id"
    assert restored.constraint_groups == cons.BUILTIN_GROUPS
    pd.testing.assert_frame_equal(restored.sample(120, seed=3), wd_segmented.sample(120, seed=3))
    assert restored.condition_probability({"ward_id": "AMU"}) == pytest.approx(0.25)
    assert [c["group"] for c in restored.model_card()["constraints"]] == ["inequality"] * 3


@pytest.mark.integration
@pytest.mark.parametrize("table", [WD, PS])
def test_real_registry_models_load_and_sample(table):
    """The project's trained models (old or new pickle format) load and sample with this code."""
    registry = ModelRegistry(get_path("models_dir", create=False))
    trained = registry.trained_models(table) if registry.path.is_file() else []
    if not trained:
        pytest.skip("no trained models (run scripts/02_train_synthesizers.py)")
    for model_id in trained:
        model = registry.load(table, model_id)
        out = model.sample(50, seed=1)
        assert list(out.columns) == schema.training_columns(table)
        json.dumps(model.model_card())


def test_distribution_tuning_records_choice(ward_daily_df):
    subset = ward_daily_df[ward_daily_df["ward_id"].isin(["GMED", "ORTH"])]
    model = BedSynthSynthesizer(WD, "gc_tuned", distribution_candidates=("truncnorm", "norm")).fit(subset)
    card = model.model_card()
    assert card["distribution_selection"]["enabled"] is True
    for segment in card["segments"]:
        assert set(segment["distribution_scores"]) == {"truncnorm", "norm"}
        best = min(segment["distribution_scores"], key=segment["distribution_scores"].get)
        assert segment["distribution"] == best
        assert all(0.0 <= v <= 1.0 for v in segment["distribution_scores"].values())
        # per-column overrides (S6/S7): only other candidates, only directly modelled numeric columns
        overrides = segment["column_distributions"]
        assert set(overrides.values()) <= {"truncnorm", "norm"} - {segment["distribution"]}
        assert set(overrides) <= set(segment["modelled_columns"])
        assert not set(overrides) & {"discharges_before_noon", "discharges", "delayed_discharges", "census_start"}
    selection = card["distribution_selection"]
    assert set(selection["column_overrides"]) <= {s["segment"] for s in card["segments"]}


def test_per_column_marginals_reach_sdv_and_quick_mode_keeps_all_candidates(ward_daily_df, isolated_paths):
    subset = ward_daily_df[ward_daily_df["ward_id"] == "GMED"]
    model = BedSynthSynthesizer(WD, "gc_tuned", distribution_candidates=("truncnorm", "norm", "gamma")).fit(subset)
    segment = model._segments[model.segments[0]]
    applied = segment.synthesizer.numerical_distributions
    assert applied == {c: d for c, d in segment.column_distributions.items() if c in segment.columns}
    # quick mode only shortens GAN/VAE epochs: GaussianCopula tuning keeps the experiment's candidates
    card = train_experiment(WD, "gc_tuned", train_df=subset, quick=True)
    configured = get_settings()["synthesis"]["experiments"]["gc_tuned"]["params"]["distribution_candidates"]
    assert card["distribution_selection"]["candidates"] == list(configured)


class _CollapsingSynthesizer:
    """Stands in for a fitted SDV synthesizer whose ``los_nights`` marginal collapsed to zero."""

    def __init__(self, data: pd.DataFrame):
        self.data = data

    def reset_sampling(self) -> None:
        pass

    def _set_random_state(self, seed: int) -> None:
        pass

    def _validate_fit_before_sample(self) -> None:
        pass

    def _sample_with_progress_bar(self, num_rows: int, show_progress_bar: bool = False) -> pd.DataFrame:
        sample = self.data.sample(n=num_rows, replace=True, random_state=0).reset_index(drop=True)
        sample["los_nights"] = 0
        return sample


def test_post_fit_check_replaces_degenerate_marginals(patient_stays_df):
    """A marginal that fitted well on the tuning subsample but collapsed on the full segment
    (a gamma fit of length of stay gave 80 % zero-night stays) is detected and refitted."""
    from bedsynth.synthesis.models import FALLBACK_DISTRIBUTION, SegmentModel

    data = patient_stays_df[schema.training_columns(PS)].drop(columns=["ward_id", "admission_source"])
    model = BedSynthSynthesizer(PS, "gc_tuned")
    segment = SegmentModel(key=("GMED", "emergency"), n_rows=len(data), columns=list(data.columns), constants={},
                           support=pd.DataFrame({"count": [len(data)]}), column_distributions={"los_nights": "gamma"})
    tuned = {"gamma": {"los_nights": 0.05, "bed_wait_hours": 0.04}, "beta": {"los_nights": 0.06}}
    degenerate = model._degenerate_marginals(_CollapsingSynthesizer(data), segment, data, "beta", tuned)
    assert set(degenerate) == {"los_nights"} and degenerate["los_nights"] > 0.5
    default = model._apply_fallbacks(segment, degenerate, "beta", replaced=set())
    assert default == "beta" and segment.column_distributions == {"los_nights": FALLBACK_DISTRIBUTION}
    assert segment.distribution_fallbacks["los_nights"]["ks"] == degenerate["los_nights"]
    # a column the copula only sees through an SDV Inequality can only follow the default
    assert model._apply_fallbacks(segment, {"discharges": 0.9}, "gamma", replaced={"discharges"}) == FALLBACK_DISTRIBUTION


def test_gc_baseline_has_no_constraints_or_tuning(ward_daily_df):
    model = BedSynthSynthesizer(WD, "gc_baseline")
    assert model.tune_distributions is False and model.constraint_groups == ()
    model.fit(ward_daily_df[ward_daily_df["ward_id"].isin(["AMU", "CARD"])])
    card = model.model_card()
    assert card["constraints"] == [] and card["segments"][0]["constraints"] == []
    assert card["distributions"] == {"__global__": "beta"}
    assert card["ordered_encodings"] == {}
    assert card["segment_by"] is None and card["segment_label"] == "global"


# =========================================================================================
# Multi-column segments: ward x admission source (patient_stays gc_tuned)
# =========================================================================================
def test_multi_column_segments_and_card(ps_segmented, ps_subset):
    assert ps_segmented.segment_columns == ("ward_id", "admission_source")
    assert ps_segmented.segments == [
        ("AMU", "emergency"), ("GSUR", "elective"), ("GSUR", "emergency"), ("GSUR", "transfer"),
        ("PAED", "elective"), ("PAED", "emergency"),
    ]
    card = ps_segmented.model_card()
    json.dumps(card)
    assert card["segment_by"] == ["ward_id", "admission_source"]
    assert card["segment_label"] == "ward_id x admission_source" and card["n_segments"] == 6
    segment = next(s for s in card["segments"] if s["segment"] == "GSUR/emergency")
    assert segment["segment_values"] == {"ward_id": "GSUR", "admission_source": "emergency"}
    # emergencies are never booked: constant zeros, no structural-zero constraint needed
    assert segment["constant_columns"]["booking_lead_days"] == 0
    assert segment["constant_columns"]["times_rescheduled"] == 0
    groups = [c["group"] for c in segment["constraints"]]
    # delay hours per reason, LOS per admission weekday, exact hour/wait/turnaround marginals, outcome triple
    assert groups == ["conditional_marginals"] * 3 + ["fixed_combinations"]
    learned = segment["constraints"][0]["learned"]
    assert learned["group_columns"] == ["delay_reason"]
    assert learned["marginals"]["discharge_delay_hours"]["none"]["zero_share"] == 1.0
    counts = ps_subset.groupby(["ward_id", "admission_source"]).size()
    assert {tuple(s["segment"].split("/")): s["rows"] for s in card["segments"]} == counts.to_dict()


def test_model_card_lists_only_applied_constraints(ps_segmented, wd_tuned):
    """S8: card constraints = what the segments apply; rules covered by segment constants are reported."""
    card = ps_segmented.model_card()
    applied = {(c["group"], tuple(c["columns"])): c["segments"] for c in card["constraints"]}
    # PS-H06 structural zeros and (ward_id, admission_source) combinations are never applied in a
    # ward x source model: the segment columns and constants reproduce them
    assert ("structural_zeros", ("admission_source", "booking_lead_days", "times_rescheduled")) not in applied
    assert ("fixed_combinations", ("ward_id", "admission_source")) not in applied
    assert applied[("conditional_marginals", ("discharge_delay_hours",))] == 6
    assert all("learned" not in c for c in card["constraints"])
    assert card["constraint_groups"] == list(cons.CONSTRAINT_GROUPS)
    assert card["effective_constraint_groups"] == ["fixed_combinations", "conditional_marginals"]
    coverage = {tuple(c["enforces_rules"]): c for c in card["constraint_coverage"]}
    unbooked = coverage[("PS-H06",)]
    assert unbooked["constraint_segments"] == 0 and unbooked["segments"] == 6
    assert unbooked["constant_segments"] == 4      # every emergency / transfer segment: constant zeros
    assert coverage[("PS-H11",)]["constant_segments"] == 6
    json.dumps(card)
    wd_card = wd_tuned.model_card()
    labels = [c for c in wd_card["constraints"] if c["group"] == "regime_labels"]
    assert {c["segments"] for c in labels} == {4}
    assert "inequality" in wd_card["effective_constraint_groups"]


def test_multi_column_routing(ps_segmented, ps_subset):
    route = ps_segmented._route
    assert route({"ward_id": "GSUR", "admission_source": "elective", "delay_reason": "none"}, 10) == [
        (("GSUR", "elective"), {"delay_reason": "none"}, 10)
    ]
    assert route({"ward_id": "AMU", "admission_source": "elective"}, 5) == []  # AMU takes no electives
    electives = route({"admission_source": "elective"}, 100)
    counts = ps_subset.loc[ps_subset["admission_source"] == "elective", "ward_id"].value_counts()
    assert [key for key, _, _ in electives] == [("GSUR", "elective"), ("PAED", "elective")]
    assert [n for _, _, n in electives] == largest_remainder(100, [counts["GSUR"], counts["PAED"]])
    assert all(conditions == {} for _, conditions, _ in electives)
    by_ward = route({"ward_id": "GSUR", "delay_reason": "none"}, 60)
    assert {key for key, _, _ in by_ward} <= {("GSUR", "elective"), ("GSUR", "emergency"), ("GSUR", "transfer")}
    assert sum(n for _, _, n in by_ward) == 60 and all(c == {"delay_reason": "none"} for _, c, _ in by_ward)


def test_multi_column_conditions_and_allocation(ps_segmented, ps_subset):
    electives = ps_segmented.sample(150, {"admission_source": "elective"}, seed=2)
    assert len(electives) == 150 and (electives["admission_source"] == "elective").all()
    assert set(electives["ward_id"]) <= {"GSUR", "PAED"} and (electives["booking_lead_days"] >= 1).all()
    # each segment keeps its own training range: elective admission hours stay within the electives' range
    observed = ps_subset.loc[ps_subset["admission_source"] == "elective", "admission_hour"]
    assert electives["admission_hour"].between(observed.min(), observed.max()).all()
    emergencies = ps_segmented.sample(150, {"admission_source": "emergency", "ward_id": "PAED"}, seed=2)
    assert len(emergencies) == 150 and (emergencies[["booking_lead_days", "times_rescheduled"]] == 0).all().all()
    out = ps_segmented.sample(600, seed=8)
    _assert_output_contract(out, PS)
    counts = ps_subset.groupby(["ward_id", "admission_source"]).size()
    expected = dict(zip(counts.index, largest_remainder(600, counts.to_numpy())))
    assert out.groupby(["ward_id", "admission_source"]).size().to_dict() == expected
    assert _violation_rates(out, PS)["PS-H06"] == 0.0


def test_multi_column_support_queries(ps_segmented, ps_subset):
    share = (ps_subset["admission_source"] == "elective").mean()
    assert ps_segmented.condition_probability({"admission_source": "elective"}) == pytest.approx(share)
    gsur_none = ((ps_subset["ward_id"] == "GSUR") & (ps_subset["delay_reason"] == "none")).mean()
    assert ps_segmented.condition_probability({"ward_id": "GSUR", "delay_reason": "none"}) == pytest.approx(gsur_none)
    assert not ps_segmented.is_feasible({"ward_id": "AMU", "admission_source": "elective"})
    assert ps_segmented.is_feasible({"ward_id": "GSUR", "admission_source": "transfer"})
    support = ps_segmented.training_support(["ward_id", "admission_source", "delay_reason"])
    observed = ps_subset.groupby(["ward_id", "admission_source", "delay_reason"]).size()
    merged = support.set_index(["ward_id", "admission_source", "delay_reason"])["count"]
    assert merged.sort_index().to_dict() == observed.sort_index().to_dict()


def test_multi_column_save_load_round_trip(ps_segmented, tmp_path):
    loaded = BedSynthSynthesizer.load(ps_segmented.save(tmp_path / "ps"))
    assert loaded.segments == ps_segmented.segments and loaded.segment_columns == ps_segmented.segment_columns
    pd.testing.assert_frame_equal(loaded.sample(200, seed=12), ps_segmented.sample(200, seed=12))


# =========================================================================================
# patient_stays (global)
# =========================================================================================
def test_patient_stays_fixed_combinations(ps_global, patient_stays_df):
    out = ps_global.sample(400, seed=2)
    _assert_output_contract(out, PS)
    groups = [["separation_type", "discharge_destination", "delay_reason"], ["ward_id", "admission_source"]]
    for group in groups:
        seen = set(map(tuple, patient_stays_df[group].drop_duplicates().to_numpy()))
        assert set(map(tuple, out[group].drop_duplicates().to_numpy())) <= seen
    # combinations excluded by a FixedCombinations constraint are rejected up-front
    assert len(ps_global.sample(50, {"ward_id": "AMU", "admission_source": "elective"}, seed=1)) == 0
    transfers = ps_global.sample(60, {"separation_type": "transfer_out"}, seed=1)
    assert len(transfers) > 0 and (transfers["discharge_destination"] == "internal_ward").all()


def test_patient_stays_structural_zeros_in_global_model(ps_global):
    out = ps_global.sample(800, seed=5)
    unbooked = out["admission_source"] != "elective"
    assert (out.loc[unbooked, ["booking_lead_days", "times_rescheduled"]] == 0).all().all()
    assert (out.loc[out["delay_reason"] == "none", "discharge_delay_hours"] == 0).all()
    rates = _violation_rates(out, PS)
    assert rates["PS-H06"] == 0.0


# =========================================================================================
# Registry
# =========================================================================================
def test_registry_lifecycle(wd_global, wd_segmented, isolated_paths):
    registry = ModelRegistry()
    assert registry.models_dir == isolated_paths["models_dir"]
    assert registry.default_model_id(WD) is None and registry.list_models() == []
    with pytest.raises(ModelNotFoundError):
        registry.load(WD)

    for experiment_id, model in (("gc_baseline", wd_global), ("gc_tuned", wd_segmented)):
        model.save(registry.model_dir(WD, experiment_id))
        registry.register(WD, experiment_id, model.model_card())
    listed = registry.list_models(WD)
    assert [m["model_id"] for m in listed] == ["gc_baseline", "gc_tuned"]
    assert all(m["trained"] for m in listed)
    assert listed[1]["segment_columns"] == ["ward_id"] and listed[1]["constraint_groups"] == list(cons.BUILTIN_GROUPS)
    json.dumps(listed)
    # no explicit default yet: settings.synthesis.default_experiment (gc_tuned) is used
    assert registry.default_model_id(WD) == "gc_tuned"
    registry.set_default(WD, "gc_baseline", reason="best composite score")
    assert registry.default_model_id(WD) == "gc_baseline"
    assert registry.to_dict()[WD]["default_reason"] == "best composite score"
    assert isinstance(registry.load(WD), BedSynthSynthesizer)
    assert registry.load(WD, "gc_tuned").segment_by == "ward_id"
    registry.update_card(WD, "gc_tuned", metrics={"composite": 0.8})
    assert registry.get_card(WD, "gc_tuned")["metrics"] == {"composite": 0.8}

    with pytest.raises(ModelNotFoundError):
        registry.set_default(WD, "ctgan")
    with pytest.raises(ModelNotFoundError):
        registry.load(WD, "ctgan")
    with pytest.raises(KeyError):
        registry.model_dir("bogus", "x")
    with pytest.raises(ValueError):
        registry.model_dir(WD, "../escape")
    assert registry.default_model_id(PS) is None

    # a model whose files disappeared is no longer "trained"
    (registry.model_dir(WD, "gc_baseline") / "synthesizer.pkl").unlink()
    assert registry.default_model_id(WD) == "gc_tuned"
    assert ModelRegistry().to_dict()[WD]["trained"] == ["gc_tuned"]


@pytest.mark.parametrize("bad_id", ["D:evil", "C:\\models", "../escape", "a/b", "a\\b", ".hidden", "", "x" * 65,
                                    "gc tuned", None])
def test_registry_rejects_ids_that_could_leave_models_dir(tmp_path, bad_id):
    registry = ModelRegistry(tmp_path / "models")
    with pytest.raises(ValueError, match="Invalid experiment id"):
        registry.model_dir(WD, bad_id)
    assert registry.is_trained(WD, bad_id) is False
    for good_id in ("gc_tuned", "ctgan", "probe-1", "A" * 64):
        assert registry.model_dir(WD, good_id) == tmp_path / "models" / WD / good_id


def test_registry_tolerates_corrupt_file(isolated_paths):
    registry = ModelRegistry()
    registry.path.write_text("{not json", encoding="utf-8")
    assert registry.list_models() == [] and registry.default_model_id(PS) is None


def test_project_relative_paths_never_record_the_home_directory(tmp_path):
    """GAP-04: stored paths are project-relative inside the project and ``~/...`` elsewhere under home."""
    inside = PROJECT_ROOT / "models" / "ward_daily" / "gc_tuned"
    assert project_relative(inside) == project_relative(str(inside)) == "models/ward_daily/gc_tuned"
    assert project_relative(PROJECT_ROOT) == "."
    assert project_relative("data/processed/x.csv") == "data/processed/x.csv"           # already relative: kept
    assert project_relative(project_relative(inside)) == "models/ward_daily/gc_tuned"    # idempotent
    assert project_relative(Path.home() / "elsewhere" / "train.csv") == "~/elsewhere/train.csv"
    assert resolve_project_path("models/ward_daily/gc_tuned") == inside
    assert resolve_project_path("~/elsewhere/train.csv") == Path.home() / "elsewhere" / "train.csv"
    assert resolve_project_path(project_relative(tmp_path)).resolve() == tmp_path.resolve()
    assert resolve_project_path(str(tmp_path)) == tmp_path                              # absolute values unchanged


def test_pickles_cards_and_registry_store_project_relative_paths(wd_global, isolated_paths):
    """GAP-04: the training source in the pickle and card, and the registry path, never contain the home
    directory; the registry path is informational (loading uses model_dir())."""
    model = cloudpickle.loads(cloudpickle.dumps(wd_global))            # private copy of the module fixture
    source = PROJECT_ROOT / "data" / "processed" / "ward_daily_train.csv"
    model.annotate(source=str(source))
    assert model.annotations["source"] == "data/processed/ward_daily_train.csv"
    model.annotate(source="provided DataFrame")                       # a description is kept as given
    assert model.annotations["source"] == "provided DataFrame"
    model.annotate(source=Path.home() / "elsewhere" / "train.csv")
    assert model.annotations["source"] == "~/elsewhere/train.csv"
    model.annotate(source=source)
    registry = ModelRegistry()
    directory = model.save(registry.model_dir(WD, "gc_tuned"))
    registry.register(WD, "gc_tuned", model.model_card())

    home = {str(Path.home()).lower(), Path.home().as_posix().lower()}
    stored = [(directory / "synthesizer.pkl").read_bytes().decode("latin-1"),
              (directory / "model_card.json").read_text(encoding="utf-8").replace("\\\\", "\\"),
              registry.path.read_text(encoding="utf-8").replace("\\\\", "\\")]
    assert not any(marker in text.lower() for text in stored for marker in home)
    card = registry.get_card(WD, "gc_tuned")
    assert card["training"]["source"] == "data/processed/ward_daily_train.csv"
    assert card["path"] == project_relative(registry.model_dir(WD, "gc_tuned")) == registry.list_models(WD)[0]["path"]
    assert resolve_project_path(card["path"]) == registry.model_dir(WD, "gc_tuned")
    data = json.loads(registry.path.read_text(encoding="utf-8"))
    data[WD]["models"]["gc_tuned"]["path"] = "somewhere/else"          # stale or moved: loading ignores it
    registry.path.write_text(json.dumps(data), encoding="utf-8")
    assert registry.load(WD, "gc_tuned").annotations["source"] == "data/processed/ward_daily_train.csv"


# =========================================================================================
# Training
# =========================================================================================
def test_subsample_training_rows_is_stratified_and_seeded(patient_stays_df):
    out = subsample_training_rows(patient_stays_df, 300, seed=1)
    assert len(out) == 300
    shares = patient_stays_df["ward_id"].value_counts(normalize=True)
    assert (out["ward_id"].value_counts(normalize=True) - shares).abs().max() < 0.01
    assert out.index.is_monotonic_increasing
    pd.testing.assert_frame_equal(out, subsample_training_rows(patient_stays_df, 300, seed=1))
    assert subsample_training_rows(patient_stays_df, None, seed=1) is patient_stays_df


def test_max_training_rows_only_for_gan_vae():
    assert max_training_rows(PS, "ctgan") == 20000
    assert max_training_rows(WD, "tvae") is None
    assert max_training_rows(PS, "gaussian_copula") is None


def test_sample_diagnostics_report_hard_and_soft_rates(wd_tuned):
    diagnostics = sample_diagnostics(wd_tuned, num_rows=400, seed=1)
    assert diagnostics["rows"] == 400 and 0.0 < diagnostics["raw_hard_valid_rate"] <= 1.0
    assert all(rule.startswith("WD-H") for rule in diagnostics["hard_violation_rates"])
    assert all(rule.startswith("WD-S") for rule in diagnostics["soft_violation_rates"])
    assert "WD-H10" not in diagnostics["hard_violation_rates"]
    json.dumps(diagnostics)


def test_train_experiment_saves_and_registers(ward_daily_df, isolated_paths):
    subset = ward_daily_df[ward_daily_df["ward_id"].isin(["AMU", "ORTH"])]
    card = train_experiment(WD, "gc_baseline", train_df=subset)
    registry = ModelRegistry()
    assert resolve_project_path(card["path"]) == registry.model_dir(WD, "gc_baseline")
    assert card["path"] == registry.get_card(WD, "gc_baseline")["path"] == project_relative(registry.model_dir(WD, "gc_baseline"))
    assert card["training"]["source"] == "provided DataFrame"
    assert registry.is_trained(WD, "gc_baseline")
    assert card["training_rows"] == len(subset) and card["training"]["input_rows"] == len(subset)
    assert card["training"]["constraint_groups"] == [] and card["training"]["segment_columns"] == []
    diagnostics = card["training"]["sample_diagnostics"]
    assert diagnostics["rows"] > 0 and 0.0 < diagnostics["raw_hard_valid_rate"] <= 1.0
    assert registry.get_card(WD, "gc_baseline")["experiment_id"] == "gc_baseline"
    assert registry.default_model_id(WD) == "gc_baseline"


def test_train_experiment_resolves_per_table_segmentation(patient_stays_df, isolated_paths):
    subset = patient_stays_df[patient_stays_df["ward_id"].isin(["AMU", "PAED"])]
    card = train_experiment(PS, "gc_tuned", train_df=subset, quick=True)
    assert card["segment_by"] == ["ward_id", "admission_source"] and card["n_segments"] == 3
    assert card["training"]["segment_columns"] == ["ward_id", "admission_source"]
    assert card["constraint_groups"] == list(cons.CONSTRAINT_GROUPS)
    loaded = ModelRegistry().load(PS, "gc_tuned")
    assert loaded.segments == [("AMU", "emergency"), ("PAED", "elective"), ("PAED", "emergency")]


def test_train_all_reports_failures_and_continues(isolated_paths):
    results = train_all(tables=[WD], experiments=["gc_baseline"])  # no processed data in isolated dirs
    assert results[0]["status"] == "failed" and "FileNotFoundError" in results[0]["error"]
    with pytest.raises(FileNotFoundError):
        train_all(tables=[WD], experiments=["gc_baseline"], continue_on_error=False)


def test_training_script_cli(isolated_paths, capsys):
    spec = importlib.util.spec_from_file_location("train_script", PROJECT_ROOT / "scripts" / "02_train_synthesizers.py")
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    assert script.main(["--tables", WD, "--experiments", "gc_baseline", "--log-level", "WARNING"]) == 1
    assert "01_prepare_data.py" in capsys.readouterr().err
    table = script.format_cards([
        {"table": WD, "experiment_id": "gc_tuned", "model": "gaussian_copula", "training_rows": 960,
         "n_segments": 8, "fit_seconds": 12.3, "distributions": {"AMU": "beta", "GMED": "norm", "ICU": "beta"},
         "constraint_groups": list(cons.CONSTRAINT_GROUPS),
         "training": {"sample_diagnostics": {"raw_hard_valid_rate": 0.42}}},
        {"table": PS, "experiment_id": "gc_segmented", "model": "gaussian_copula", "n_segments": 8,
         "constraint_groups": ["inequality", "fixed_combinations"]},
        {"table": PS, "experiment_id": "ctgan", "status": "failed", "error": "boom"},
    ])
    assert "beta x2, norm x1" in table and "42.0" in table and "FAILED patient_stays/ctgan: boom" in table
    assert f"all ({len(cons.CONSTRAINT_GROUPS)})" in table and "inequality+fixed_combinations" in table


# =========================================================================================
# CTGAN / TVAE (1 epoch) - slow
# =========================================================================================
@pytest.mark.slow
@pytest.mark.parametrize("model_type", ["ctgan", "tvae"])
def test_gan_vae_fit_and_sample(model_type):
    data = make_patient_stays(n=300, seed=4)
    config = _config(f"{model_type}_test", model=model_type, constraints=True, ordinal_metadata=True,
                     params={"epochs": 1, "batch_size": 100, "enable_gpu": False, "verbose": False})
    model = BedSynthSynthesizer(PS, config).fit(data)
    out = model.sample(120, seed=1)
    _assert_output_contract(out, PS)
    pd.testing.assert_frame_equal(out, model.sample(120, seed=1))
    seen = set(map(tuple, data[["ward_id", "admission_source"]].drop_duplicates().to_numpy()))
    assert set(map(tuple, out[["ward_id", "admission_source"]].to_numpy())) <= seen
    assert (out.loc[out["admission_source"] != "elective", "booking_lead_days"] == 0).all()
    conditional = model.sample(30, {"admission_source": "emergency"}, seed=2)
    assert (conditional["admission_source"] == "emergency").all()
    assert len(model.sample(10, {"ward_id": "AMU", "admission_source": "elective"}, seed=2)) == 0
    card = model.model_card()
    assert card["synthesizer"] == {"ctgan": "CTGANSynthesizer", "tvae": "TVAESynthesizer"}[model_type]
    assert card["distributions"] == {"__global__": None}


@pytest.mark.slow
def test_ctgan_ward_daily_global(tmp_path):
    data = make_ward_daily(n_days=30, seed=5)
    config = _config("ctgan_wd", model="ctgan", constraints=True, ordinal_metadata=True,
                     params={"epochs": 1, "batch_size": 100, "enable_gpu": False, "verbose": False})
    model = BedSynthSynthesizer(WD, config).fit(data)
    out = model.sample(80, seed=3)
    _assert_output_contract(out, WD)
    assert (out["elective_cancelled"] <= out["elective_scheduled"]).all()
    rates = _violation_rates(out, WD)
    assert rates["WD-H10"] == 0.0 and rates["WD-H11"] == 0.0
    loaded = BedSynthSynthesizer.load(model.save(tmp_path / "ctgan"))
    pd.testing.assert_frame_equal(loaded.sample(50, seed=4), model.sample(50, seed=4))
