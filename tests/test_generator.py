"""Tests for the scenario generation engine (``bedsynth.synthesis.generator``).

Models are fitted once per module on the rule-satisfying fixtures (January-April 2023) and
registered in a temporary models directory. The engine validates every round with the real
``bedsynth.validation`` rule engine.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import pandas as pd
import pytest

from bedsynth import schema, validation
from bedsynth.data.fixtures import make_ward_daily
from bedsynth.synthesis import generator as generator_module
from bedsynth.synthesis.constraints import CONSTRAINT_GROUPS
from bedsynth.synthesis.generator import (
    GenerationError,
    GenerationRequest,
    GenerationResult,
    ModelNotFoundError,
    SyntheticDataGenerator,
    expand_conditions,
)
from bedsynth.synthesis.models import BedSynthSynthesizer, ExperimentConfig
from bedsynth.synthesis.registry import ModelRegistry
from bedsynth.synthesis.training import sample_diagnostics

WD, PS = schema.WARD_DAILY, schema.PATIENT_STAYS

REPORT_KEYS = {
    "table", "model_id", "scenario", "mode", "seed", "num_rows_requested", "num_rows_generated",
    "shortfall", "attempts", "sampled_rows", "accepted_rows", "acceptance_rate", "raw_hard_valid_rate",
    "rejected_by_rule", "rejected_by_filter", "condition_allocation", "elapsed_seconds", "warnings",
    "conditions", "filters", "model", "reference_source", "allocation_strata", "time_budget_seconds",
    "stopped_reason",
}


def _hard_valid(df: pd.DataFrame, table: str) -> bool:
    return bool(validation.validate_table(df, table, include_soft=False).hard_valid_mask.all())


# =========================================================================================
# Trained models (module scope)
# =========================================================================================
def _register(registry: ModelRegistry, table: str, model: BedSynthSynthesizer) -> None:
    model.annotate(sample_diagnostics=sample_diagnostics(model, num_rows=600))
    model.save(registry.model_dir(table, model.experiment_id))
    registry.register(table, model.experiment_id, model.model_card())


@pytest.fixture(scope="module")
def dirs(tmp_path_factory, ward_daily_df, patient_stays_df):
    """Models dir with ward_daily 'gc_global' (default) + 'gc_tuned' (per ward) and patient_stays
    'gc_global' (default) + 'gc_tuned' (per ward x admission source)."""
    models_dir = tmp_path_factory.mktemp("models")
    processed_dir = tmp_path_factory.mktemp("processed")  # empty: allocation falls back to model data
    registry = ModelRegistry(models_dir)
    global_config = ExperimentConfig("gc_global", "gaussian_copula", None, True, True, True,
                                     {"default_distribution": "truncnorm"})
    _register(registry, WD, BedSynthSynthesizer(WD, global_config, tune_distributions=False).fit(ward_daily_df))
    _register(registry, WD, BedSynthSynthesizer(WD, "gc_tuned", tune_distributions=False).fit(ward_daily_df))
    _register(registry, PS, BedSynthSynthesizer(PS, global_config, tune_distributions=False).fit(patient_stays_df))
    _register(registry, PS, BedSynthSynthesizer(PS, "gc_tuned", tune_distributions=False).fit(patient_stays_df))
    registry.set_default(WD, "gc_global", reason="fast global model for tests")
    registry.set_default(PS, "gc_global", reason="fast global model for tests")
    return {"models": models_dir, "processed": processed_dir}


@pytest.fixture(scope="module")
def generator(dirs) -> SyntheticDataGenerator:
    return SyntheticDataGenerator(models_dir=dirs["models"], processed_dir=dirs["processed"])


def _generate(generator: SyntheticDataGenerator, **kwargs) -> GenerationResult:
    kwargs.setdefault("seed", 123)
    return generator.generate(GenerationRequest(**kwargs))


# =========================================================================================
# Sample mode
# =========================================================================================
def test_baseline_generation_contract(generator):
    result = _generate(generator, table=WD, num_rows=250)
    data, report = result.data, result.report
    assert len(data) == 250
    assert list(data.columns) == schema.output_columns(WD)
    assert data["record_id"].iloc[0] == "WD-000001" and data["record_id"].is_unique
    assert not data.isna().any().any()
    assert _hard_valid(data, WD)
    assert REPORT_KEYS <= set(report)
    json.dumps(report)
    assert report["num_rows_generated"] == 250 and report["shortfall"] == 0
    assert report["model_id"] == "gc_global" and report["scenario"] is None and report["seed"] == 123
    assert 0 < report["acceptance_rate"] <= 1 and report["accepted_rows"] >= 250
    assert report["sampled_rows"] >= report["accepted_rows"]
    assert report["condition_allocation"] == [
        {"conditions": {}, "requested": 250, "generated": 250, "reference_count": None,
         "sampled": report["sampled_rows"], "accepted": report["accepted_rows"], "feasible": True}
    ]
    assert all(isinstance(v, int) for v in report["rejected_by_rule"].values())
    # derived identities hold in the output
    flow = (data["census_start"] + data["admissions_elective"] + data["admissions_emergency"]
            + data["transfers_in"] - data["transfers_out"] - data["discharges"])
    assert (flow == data["census_end"]).all()


def test_seeded_determinism(generator):
    first = _generate(generator, table=WD, num_rows=120, scenario="high_occupancy", seed=7)
    second = _generate(generator, table=WD, num_rows=120, scenario="high_occupancy", seed=7)
    pd.testing.assert_frame_equal(first.data, second.data)
    assert first.report["sampled_rows"] == second.report["sampled_rows"]
    other = _generate(generator, table=WD, num_rows=120, scenario="high_occupancy", seed=8)
    assert not first.data.equals(other.data)


def test_seed_none_is_drawn_and_reported(generator):
    result = generator.generate(GenerationRequest(table=WD, num_rows=60, seed=None))
    seed = result.report["seed"]
    assert isinstance(seed, int) and seed >= 0
    replay = generator.generate(GenerationRequest(table=WD, num_rows=60, seed=seed))
    pd.testing.assert_frame_equal(result.data, replay.data)


def test_high_occupancy_scenario(generator):
    result = _generate(generator, table=WD, num_rows=200, scenario="high_occupancy")
    data, report = result.data, result.report
    assert len(data) == 200 and set(data["occupancy_level"]) <= {"high", "critical"}
    assert (data["occupancy_rate"] >= 0.90).all()
    assert _hard_valid(data, WD)
    allocation = report["condition_allocation"]
    assert [a["conditions"] for a in allocation] == [{"occupancy_level": "high"}, {"occupancy_level": "critical"}]
    assert sum(a["requested"] for a in allocation) == 200
    assert all(a["reference_count"] > 0 for a in allocation)
    assert report["reference_source"] == "model_training_data"
    assert any("Reference train split not available" in w for w in report["warnings"])
    assert report["conditions"] == {"occupancy_level": ["high", "critical"]}


def test_filter_scenarios_are_satisfied(generator):
    escalation = _generate(generator, table=WD, num_rows=80, scenario="escalation_capacity").data
    assert len(escalation) == 80 and (escalation["escalation_beds_open"] >= 1).all()
    turnover = _generate(generator, table=WD, num_rows=80, scenario="high_turnover").data
    assert len(turnover) == 80 and (turnover["bed_turnover_rate"] >= 0.30).all()


def test_scheduled_procedures_conditions_and_filters(generator):
    result = _generate(generator, table=WD, num_rows=150, scenario="scheduled_procedures")
    data = result.data
    assert len(data) == 150
    assert set(data["ward_id"]) <= {"GSUR", "ORTH", "CARD"}
    assert set(data["day_of_week"]) <= set(schema.WEEKDAYS)
    assert (data["elective_scheduled"] >= 2).all()
    assert len(result.report["condition_allocation"]) == 3 * 5
    assert result.report["filters"] == {"elective_scheduled": {"min": 2}}


def test_request_merges_wards_conditions_and_filters(generator):
    result = _generate(
        generator, table=WD, num_rows=60, scenario="winter_demand", wards=["GMED"],
        conditions={"day_of_week": ["Mon", "monday"]}, filters={"discharges": {"min": 1}},
    )
    data, report = result.data, result.report
    assert len(data) == 60
    assert (data["ward_id"] == "GMED").all() and (data["day_of_week"] == "Mon").all()
    assert set(data["month"]) <= {1, 2} and (data["discharges"] >= 1).all()
    assert report["conditions"] == {"month": [12, 1, 2], "day_of_week": ["Mon"], "ward_id": ["GMED"]}
    # December never occurs in the January-April fixture: no rows allocated, with a warning
    december = next(a for a in report["condition_allocation"] if a["conditions"]["month"] == 12)
    assert december["requested"] == 0 and december["feasible"] is False
    assert any("never occurs" in w for w in report["warnings"])


def test_segmented_model_generation(generator):
    result = _generate(generator, table=WD, num_rows=160, model_id="gc_tuned", scenario="staffing_shortage")
    data = result.data
    assert len(data) == 160 and (data["staffing_level"] == "short").all()
    assert result.report["model_id"] == "gc_tuned" and result.report["model"]["segment_by"] == "ward_id"
    amu = _generate(generator, table=WD, num_rows=40, model_id="gc_tuned", wards=["AMU"]).data
    assert (amu["elective_scheduled"] == 0).all() and (amu["transfers_in"] == 0).all()


def test_patient_stays_scenarios(generator):
    delayed = _generate(generator, table=PS, num_rows=120, scenario="delayed_discharge_stays").data
    assert len(delayed) == 120
    assert (delayed["delay_reason"] != "none").all() and (delayed["discharge_delay_hours"] >= 0.1).all()
    assert list(delayed.columns) == schema.output_columns(PS)
    assert delayed["stay_id"].iloc[0] == "ST-0000001"
    assert _hard_valid(delayed, PS)
    electives = _generate(generator, table=PS, num_rows=80, scenario="rescheduled_electives").data
    assert (electives["admission_source"] == "elective").all() and (electives["times_rescheduled"] >= 1).all()
    weekend = _generate(generator, table=PS, num_rows=80, scenario="weekend_discharges").data
    assert set(weekend["discharge_day_of_week"]) <= {"Sat", "Sun"}


def test_include_derived_false(generator):
    data = _generate(generator, table=PS, num_rows=30, include_derived=False).data
    assert list(data.columns) == [schema.key_column(PS)] + schema.training_columns(PS)


def test_infeasible_conditions_and_impossible_filters(generator):
    unseen = _generate(generator, table=WD, num_rows=20, conditions={"month": 9})
    assert unseen.data.empty and unseen.report["shortfall"] == 20
    assert list(unseen.data.columns) == schema.output_columns(WD)
    assert any("never occurs" in w for w in unseen.report["warnings"])

    started = time.perf_counter()
    impossible = _generate(generator, table=WD, num_rows=20, filters={"census_start": {"min": 10_000}})
    assert time.perf_counter() - started < 30
    report = impossible.report
    assert impossible.data.empty and report["shortfall"] == 20 and report["accepted_rows"] == 0
    assert report["rejected_by_filter"] > 0
    assert any("Stopped sampling" in w for w in report["warnings"])
    assert any("Low acceptance" in w for w in report["warnings"])
    assert any("Generated 0 of 20" in w for w in report["warnings"])


# =========================================================================================
# Calendar mode
# =========================================================================================
def test_calendar_mode_grid(generator):
    result = _generate(generator, table=WD, mode="calendar", wards=["GMED", "AMU"],
                       start_date="2023-01-01", end_date="2023-01-31", model_id="gc_tuned")
    data, report = result.data, result.report
    assert len(data) == 62 and report["num_rows_requested"] == 62 and report["shortfall"] == 0
    assert list(data.columns) == schema.output_columns(WD, include_time=True)
    assert not data.duplicated(["ward_id", "date"]).any()
    assert list(zip(data["ward_id"], data["date"])) == sorted(zip(data["ward_id"], data["date"]))
    dates = pd.to_datetime(data["date"])
    assert (dates.dt.month == data["month"]).all()
    assert (dates.dt.dayofweek.map(dict(enumerate(schema.DAYS_OF_WEEK))) == data["day_of_week"]).all()
    assert set(data.loc[data["ward_id"] == "GMED", "date"]) == {f"2023-01-{d:02d}" for d in range(1, 32)}
    assert _hard_valid(data, WD)
    assert report["calendar"]["grid_rows"] == 62 and report["calendar"]["dropped_rows"] == 0
    assert {a["conditions"]["ward_id"] for a in report["condition_allocation"]} == {"GMED", "AMU"}


def test_calendar_mode_with_scenario_conditions(generator):
    weekend = _generate(generator, table=WD, mode="calendar", scenario="weekend_operations", wards=["GMED"],
                        start_date="2023-01-01", end_date="2023-01-31")
    data, report = weekend.data, weekend.report
    assert len(data) == 9 and set(data["day_of_week"]) == {"Sat", "Sun"}
    assert report["calendar"]["dropped_rows"] == 22
    assert any("were dropped" in w for w in report["warnings"])

    high = _generate(generator, table=WD, mode="calendar", scenario="high_occupancy", wards=["ICU", "ORTH"],
                     start_date="2023-02-01", end_date="2023-02-14")
    assert len(high.data) == high.report["num_rows_requested"] - high.report["shortfall"]
    assert set(high.data["occupancy_level"]) <= {"high", "critical"}
    keys = list(zip(high.data["ward_id"], high.data["date"]))
    assert keys == sorted(keys)


def test_calendar_mode_errors(generator):
    base = dict(table=WD, mode="calendar", wards=["GMED"])
    with pytest.raises(GenerationError, match="only available"):
        _generate(generator, table=PS, mode="calendar", start_date="2023-01-01", end_date="2023-01-02")
    with pytest.raises(GenerationError, match="requires start_date"):
        _generate(generator, **base, start_date="2023-01-01")
    with pytest.raises(GenerationError, match="YYYY-MM-DD"):
        _generate(generator, **base, start_date="01/01/2023", end_date="2023-01-02")
    with pytest.raises(GenerationError, match="before start_date"):
        _generate(generator, **base, start_date="2023-02-01", end_date="2023-01-01")
    with pytest.raises(GenerationError, match="No calendar day"):
        _generate(generator, **base, scenario="summer_demand", start_date="2023-01-01", end_date="2023-01-31")
    with pytest.raises(GenerationError, match="exceeds the limit"):
        _generate(generator, table=WD, mode="calendar", start_date="2000-01-01", end_date="2010-12-31")


# =========================================================================================
# Request validation and model resolution
# =========================================================================================
@pytest.mark.parametrize(
    "kwargs, message",
    [
        (dict(table="bogus"), "Unknown table"),
        (dict(scenario="no_such_scenario"), "Unknown scenario"),
        (dict(scenario="stays_baseline"), "belongs to table 'patient_stays'"),
        (dict(num_rows=0), "between 1 and"),
        (dict(num_rows=10**6), "between 1 and"),
        (dict(num_rows=2.5), "integer"),
        (dict(mode="stream"), "Unknown mode"),
        (dict(wards=["XYZ"]), "Invalid value 'XYZ'"),
        (dict(conditions={"census_start": [3]}), "cannot be used as a condition"),
        (dict(conditions={"occupancy_level": ["extreme"]}), "Invalid value 'extreme'"),
        (dict(conditions={"month": ["june"]}), "Invalid value 'june'"),
        (dict(conditions={"month": [2.5]}), "Invalid value 2.5"),
        (dict(conditions={"month": [13]}), "Invalid value 13"),
        (dict(scenario="high_occupancy", conditions={"occupancy_level": "low"}), "do not overlap"),
        (dict(filters={"not_a_column": {"min": 1}}), "cannot be filtered"),
        (dict(filters={"discharges": {"above": 1}}), "Unknown filter keys"),
        (dict(filters={"discharges": {"min": "one"}}), "must be a number"),
        (dict(filters={"discharges": {"min": 5, "max": 2}}), "contradictory"),
        (dict(seed=-1), "seed must be"),
        (dict(max_attempts=0), "max_attempts"),
    ],
)
def test_invalid_requests_raise_generation_error(generator, kwargs, message):
    request = dict(table=WD, num_rows=10)
    request.update(kwargs)
    with pytest.raises(GenerationError, match=message):
        _generate(generator, **request)


def test_generation_error_is_value_error():
    assert issubclass(GenerationError, ValueError)
    assert issubclass(ModelNotFoundError, LookupError)


def test_missing_models(generator, tmp_path):
    empty = SyntheticDataGenerator(models_dir=tmp_path / "no_models", processed_dir=tmp_path)
    with pytest.raises(ModelNotFoundError, match="02_train_synthesizers"):
        empty.generate(GenerationRequest(table=WD, num_rows=5))
    assert empty.default_model_id(WD) is None and empty.available_models() == []
    with pytest.raises(ModelNotFoundError):
        _generate(generator, table=WD, num_rows=5, model_id="ctgan")
    with pytest.raises(ModelNotFoundError):
        _generate(generator, table=PS, num_rows=5, model_id="../gc_global")


def test_available_models_and_defaults(generator):
    models = generator.available_models(WD)
    assert {m["model_id"] for m in models} == {"gc_global", "gc_tuned"}
    assert next(m for m in models if m["model_id"] == "gc_global")["is_default"] is True
    assert generator.default_model_id(WD) == "gc_global"
    assert generator.default_model_id(PS) == "gc_global"
    assert {m["table"] for m in generator.available_models()} == {WD, PS}
    with pytest.raises(GenerationError):
        generator.default_model_id("bogus")


def test_model_cache_reloads_after_retraining(generator, dirs):
    model_id, first = generator.get_model(WD, "gc_global")
    assert generator.get_model(WD)[1] is first and model_id == "gc_global"
    path = ModelRegistry(dirs["models"]).model_dir(WD, "gc_global") / "synthesizer.pkl"
    stat = path.stat()
    os.utime(path, (stat.st_atime, stat.st_mtime + 10))
    assert generator.get_model(WD, "gc_global")[1] is not first
    generator.clear_cache()
    assert generator.get_model(WD, "gc_global")[1] is not first


def test_sample_raw_returns_unfiltered_derived_rows(generator):
    raw = generator.sample_raw(WD, 300, seed=3)
    assert len(raw) == 300 and set(schema.derived_columns(WD)) <= set(raw.columns)
    report = validation.validate_table(raw, WD, include_soft=False)
    assert report.n_hard_valid < len(raw)  # unfiltered: some rows violate hard rules
    pd.testing.assert_frame_equal(raw, generator.sample_raw(WD, 300, seed=3))


def test_expand_conditions():
    assert expand_conditions(None) == [{}]
    assert expand_conditions({"a": (1, 2), "b": ("x",)}) == [{"a": 1, "b": "x"}, {"a": 2, "b": "x"}]


def test_generation_request_round_trip():
    request = GenerationRequest(table=PS, num_rows=5, scenario="stays_baseline", seed=1)
    assert request.to_dict()["scenario"] == "stays_baseline"
    assert GenerationRequest(**request.to_dict()) == request


def test_acceptance_prior_sizes_first_batch(generator):
    _, model = generator.get_model(WD, "gc_global")
    prior = model.annotations["sample_diagnostics"]["raw_hard_valid_rate"]
    report = _generate(generator, table=WD, num_rows=50).report
    assert report["acceptance_prior"] == pytest.approx(prior)
    oversample = generator.settings["oversample_factor"]
    assert report["first_batch_factor"] == pytest.approx(min(12.0, max(oversample, 1.2 / prior)), rel=1e-3)


def test_reference_split_is_used_when_available(dirs, ward_daily_df, tmp_path):
    processed = tmp_path / "processed"
    processed.mkdir()
    reference = ward_daily_df.copy()
    reference.loc[reference["occupancy_level"] == "critical", "occupancy_level"] = "high"  # no critical days
    reference.to_csv(processed / "ward_daily_train.csv", index=False)
    engine = SyntheticDataGenerator(models_dir=dirs["models"], processed_dir=processed)
    report = engine.generate(GenerationRequest(table=WD, num_rows=40, scenario="high_occupancy", seed=1)).report
    assert report["reference_source"] == "reference_train_split"
    allocation = {a["conditions"]["occupancy_level"]: a["requested"] for a in report["condition_allocation"]}
    assert allocation == {"high": 40, "critical": 0}
    assert not any("Reference train split not available" in w for w in report["warnings"])


def test_report_is_json_serialisable_for_every_ward_daily_scenario(generator):
    from bedsynth.scenarios import scenarios_for

    for scenario in scenarios_for(WD):
        report = _generate(generator, table=WD, num_rows=12, scenario=scenario.name, max_attempts=4).report
        json.dumps(report)
        assert report["scenario"] == scenario.name
        assert report["num_rows_generated"] + report["shortfall"] == 12
        assert np.isfinite(report["elapsed_seconds"])


# =========================================================================================
# Allocation strata (rejection-bias correction)
# =========================================================================================
def test_allocation_strata_follow_the_reference_mix(dirs, ward_daily_df, tmp_path):
    processed = tmp_path / "processed"
    processed.mkdir()
    ward_daily_df.to_csv(processed / "ward_daily_train.csv", index=False)
    engine = SyntheticDataGenerator(models_dir=dirs["models"], processed_dir=processed)
    result = engine.generate(GenerationRequest(table=WD, num_rows=400, seed=3))
    data, report = result.data, result.report
    assert len(data) == 400 and _hard_valid(data, WD)
    strata = report["allocation_strata"]
    assert strata["columns"] == ["ward_id", "occupancy_level"]
    assert strata["source"] == "reference_train_split"
    assert strata["strata_cells"] > 8 and strata["stratified_combinations"] == 1
    # the allocation report stays at the level of the requested combinations
    assert [a["conditions"] for a in report["condition_allocation"]] == [{}]
    assert report["condition_allocation"][0]["requested"] == 400
    assert report["condition_allocation"][0]["generated"] == 400
    keys = ["ward_id", "occupancy_level"]
    expected = ward_daily_df.groupby(keys).size() / len(ward_daily_df)
    observed = (data.groupby(keys).size() / len(data)).reindex(expected.index, fill_value=0.0)
    tvd = 0.5 * float((observed - expected).abs().sum())
    assert tvd <= 0.05 + strata["rebalanced_rows"] / 400


def test_allocation_strata_respect_conditions_and_filters(dirs, ward_daily_df, tmp_path):
    processed = tmp_path / "processed"
    processed.mkdir()
    ward_daily_df.to_csv(processed / "ward_daily_train.csv", index=False)
    engine = SyntheticDataGenerator(models_dir=dirs["models"], processed_dir=processed)
    result = engine.generate(GenerationRequest(table=WD, num_rows=120, scenario="high_occupancy", seed=4,
                                               wards=["GMED", "CARD"], filters={"discharges": {"min": 1}}))
    data = result.data
    assert len(data) == 120 and set(data["ward_id"]) <= {"GMED", "CARD"}
    assert set(data["occupancy_level"]) <= {"high", "critical"} and (data["discharges"] >= 1).all()
    allocation = result.report["condition_allocation"]
    assert len(allocation) == 4 and sum(a["requested"] for a in allocation) == 120


def test_allocation_strata_settings(dirs):
    engine = SyntheticDataGenerator(models_dir=dirs["models"], processed_dir=dirs["processed"])
    assert engine.allocation_strata[WD] == ("ward_id", "occupancy_level")
    engine.allocation_strata[WD] = ()
    report = engine.generate(GenerationRequest(table=WD, num_rows=50, seed=1)).report
    assert report["allocation_strata"]["columns"] == [] and report["allocation_strata"]["strata_cells"] == 0
    assert SyntheticDataGenerator._configured_strata({PS: []})[PS] == ()
    with pytest.raises(ValueError, match="not categorical train columns"):
        SyntheticDataGenerator._configured_strata({WD: ["census_start"]})
    with pytest.raises(ValueError, match="unknown table"):
        SyntheticDataGenerator._configured_strata({"beds": ["ward_id"]})


def test_filtered_requests_take_strata_from_a_synthetic_pilot(generator):
    # numeric filters: strata weights come from a seeded pilot draw of the model, filtered like the request
    first = _generate(generator, table=WD, num_rows=30, scenario="escalation_capacity")
    strata = first.report["allocation_strata"]
    assert strata["source"] == "model_pilot_sample" and strata["pilot_rows"] > 0 and strata["strata_cells"] > 0
    again = _generate(generator, table=WD, num_rows=30, scenario="escalation_capacity")
    pd.testing.assert_frame_equal(first.data, again.data)                 # the pilot is seeded
    assert len(first.data) == 30 and _hard_valid(first.data, WD)


def test_filtered_strata_never_read_the_real_rows(dirs, patient_stays_df, tmp_path, monkeypatch):
    """S4: a filter isolating a few real stays must not copy their make-up into the output/report."""
    processed = tmp_path / "processed"
    processed.mkdir()
    patient_stays_df.to_csv(processed / "patient_stays_train.csv", index=False)
    engine = SyntheticDataGenerator(models_dir=dirs["models"], processed_dir=processed)
    filters = {"los_nights": {"min": 6}, "discharge_delay_hours": {"min": 30}}
    real = schema.derive(PS, patient_stays_df[schema.training_columns(PS)])
    assert 0 < int(generator_module.scenarios.filter_mask(real, filters).sum()) < 40   # a tiny real subset

    def no_real_rows(*args, **kwargs):
        raise AssertionError("filtered strata must not read the reference split")

    monkeypatch.setattr(engine, "_reference_full", no_real_rows)
    result = engine.generate(GenerationRequest(table=PS, num_rows=40, model_id="gc_tuned", seed=4, filters=filters,
                                               time_budget_seconds=60))
    strata = result.report["allocation_strata"]
    assert strata["source"] in ("model_pilot_sample", "unavailable (no time for a pilot draw)")
    assert _hard_valid(result.data, PS) and (result.data["los_nights"] >= 6).all()


def test_calendar_strata_keep_backoff_bands_unseen_in_the_exact_cell(generator):
    """S11: a band found at the (ward, month) back-off level stays a stratum even when the model
    never saw it on that exact ward-month-weekday (dropping it under-represented critical days)."""
    _, model = generator.get_model(WD, "gc_tuned")
    unseen = next(
        ({"ward_id": "GMED", "month": month, "day_of_week": day, "occupancy_level": band}
         for month in (1, 2, 3, 4) for day in schema.DAYS_OF_WEEK for band in ("low", "normal", "high", "critical")
         if model.is_feasible({"ward_id": "GMED", "occupancy_level": band})
         and model.condition_probability({"ward_id": "GMED", "month": month, "day_of_week": day,
                                          "occupancy_level": band}) == 0),
        None,
    )
    assert unseen is not None
    cell = {k: unseen[k] for k in ("ward_id", "month", "day_of_week")}
    other = next(d for d in schema.DAYS_OF_WEEK if d != cell["day_of_week"])
    reference = pd.DataFrame({
        "ward_id": "GMED", "month": cell["month"],
        "day_of_week": [cell["day_of_week"]] * 5 + [other] * 25,          # exact cell too small: back off
        "occupancy_level": ["normal"] * 5 + [unseen["occupancy_level"]] * 10 + ["normal"] * 15,
    })
    strata = generator._stratum_weights(model, reference, cell, ["occupancy_level"], calendar=True)
    assert {"occupancy_level": unseen["occupancy_level"]} in [values for values, _ in strata]
    # the model serves such an unseen combination from its pooled draw; sample_many returns *up to*
    # the requested rows (one pooled draw can come back a row short - the generator's rejection
    # loop tops it up), so the property checked is that the unseen band is served at all
    rows = model.sample_many([(unseen, 5)], seed=3)[0]
    assert 0 < len(rows) <= 5 and (rows["occupancy_level"] == unseen["occupancy_level"]).all()


def test_calendar_year_keeps_flat_weekday_lists(generator):
    """S3: calendar mode reproduces the training weekday profile of elective lists (no gradient)."""
    result = _generate(generator, table=WD, model_id="gc_tuned", mode="calendar", wards=["GMED", "ICU", "PAED"],
                       start_date="2023-01-02", end_date="2023-04-30", seed=17)
    data = result.data
    assert (data.loc[data["day_of_week"].isin(schema.WEEKEND_DAYS), "elective_scheduled"] == 0).all()
    weekday = data[data["day_of_week"].isin(schema.WEEKDAYS)].groupby("day_of_week")["elective_scheduled"].mean()
    assert weekday.std() / weekday.mean() < 0.25 and _hard_valid(data, WD)


def test_rebalance_moves_quota_dates_and_spare_rows_between_strata(tmp_path):
    from bedsynth.synthesis.generator import _Cell, _LoopStats

    engine = SyntheticDataGenerator(models_dir=tmp_path / "models", processed_dir=tmp_path / "processed")
    rows = make_ward_daily(n_days=7, wards=["AMU"], with_dates=False)
    hard = _Cell(conditions={"occupancy_level": "critical"}, requested=3, parent=0, stratum=True, weight=1.0,
                 dates=["2023-01-02", "2023-01-09", "2023-01-16"])
    easy = _Cell(conditions={"occupancy_level": "low"}, requested=2, parent=0, stratum=True, weight=3.0,
                 dates=["2023-01-23", "2023-01-30"])
    easy.frames, easy.kept, easy.passed, easy.requested_from_model = [rows.iloc[:2]], 2, 4, 10
    easy.spare, easy.spare_rows = [rows.iloc[2:4]], 2
    hard.exhausted, hard.sampled, hard.requested_from_model = True, 2000, 2000
    stats = _LoopStats()

    engine._rebalance([hard, easy], stats, max_batch=1000, overall=0.4, mode="exhausted")
    assert (hard.requested, hard.dates) == (0, [])
    assert easy.requested == 5 and len(easy.dates) == 5 and easy.kept == 4 and easy.spare_rows == 0
    assert easy.remaining == 1 and easy.batch >= 1 and stats.quota_moves == 3

    short = _Cell(conditions={"occupancy_level": "high"}, requested=2, parent=1, stratum=True, weight=1.0)
    donor = _Cell(conditions={"occupancy_level": "normal"}, requested=1, parent=1, stratum=True, weight=1.0)
    donor.frames, donor.kept, donor.spare, donor.spare_rows = [rows.iloc[4:5]], 1, [rows.iloc[5:7]], 2
    engine._rebalance([short, donor], stats, max_batch=1000, overall=None, mode="final")
    assert short.requested == 0 and donor.requested == 3 and donor.kept == 3 and donor.remaining == 0


def test_rebalanced_rows_count_delivered_rows_beyond_the_original_quota():
    """E9: the report counts rows that shifted strata, not quota moves (a quota can move twice)."""
    from bedsynth.synthesis.generator import _Cell, rebalanced_rows

    a = _Cell(conditions={"occupancy_level": "critical"}, requested=0, stratum=True, allocated=4)
    b = _Cell(conditions={"occupancy_level": "high"}, requested=0, stratum=True, allocated=3)   # passed its quota on
    c = _Cell(conditions={"occupancy_level": "low"}, requested=10, stratum=True, allocated=3)
    c.kept = 10                                    # 4 + 3 rows moved here (a -> b -> c counts once per row)
    plain = _Cell(conditions={}, requested=5)
    plain.kept = 5                                 # unstratified cells never count
    assert rebalanced_rows([a, b, c, plain]) == 7
    calendar = _Cell(conditions={"occupancy_level": "low"}, requested=3, stratum=True, allocated=1,
                     dates=["2025-01-06", "2025-01-13", "2025-01-20"])
    calendar.kept = 3                              # calendar rows are delivered with the dates they carry
    assert rebalanced_rows([calendar]) == 2


# =========================================================================================
# Per-request time budget
# =========================================================================================
class _FakeClock:
    """Stands in for the generator module's ``time``: advances only when the model samples."""

    def __init__(self) -> None:
        self.now = 1_000.0

    def perf_counter(self) -> float:
        return self.now


def _slow_partial_rounds(monkeypatch, generator: SyntheticDataGenerator, table: str, model_id: str,
                         clock: _FakeClock, seconds_per_round: float = 100.0) -> None:
    """Every sampling round 'takes' ``seconds_per_round`` and returns a tenth of the rows."""
    _, model = generator.get_model(table, model_id)
    original = model.sample_many

    def sample_many(requests, seed=None, **kwargs):
        frames = original(requests, seed=seed, **kwargs)
        clock.now += seconds_per_round
        return [frame.head(max(1, len(frame) // 10)) for frame in frames]

    monkeypatch.setattr(model, "sample_many", sample_many)


def test_time_budget_default_and_validation(generator, dirs):
    report = _generate(generator, table=WD, num_rows=30).report
    assert report["time_budget_seconds"] == generator.time_budget_seconds == 300.0
    assert report["stopped_reason"] is None
    custom = SyntheticDataGenerator(models_dir=dirs["models"], processed_dir=dirs["processed"], time_budget_seconds=45)
    assert custom.generate(GenerationRequest(table=WD, num_rows=10, seed=1)).report["time_budget_seconds"] == 45.0
    assert _generate(generator, table=WD, num_rows=10, time_budget_seconds=12).report["time_budget_seconds"] == 12.0
    for bad in (0, -5, float("nan"), float("inf"), "10", True):
        with pytest.raises(GenerationError, match="time_budget_seconds"):
            _generate(generator, table=WD, num_rows=5, time_budget_seconds=bad)


@pytest.mark.parametrize("mode", ["sample", "calendar"])
def test_time_budget_stops_after_the_current_round(generator, monkeypatch, mode):
    clock = _FakeClock()
    monkeypatch.setattr(generator_module, "time", clock)
    _slow_partial_rounds(monkeypatch, generator, WD, "gc_global", clock)
    kwargs = dict(table=WD, model_id="gc_global", time_budget_seconds=50, seed=21)
    if mode == "calendar":
        kwargs.update(mode="calendar", wards=["GMED", "ICU"], start_date="2023-01-01", end_date="2023-02-28")
    else:
        kwargs.update(num_rows=300)
    result = generator.generate(GenerationRequest(**kwargs))  # never raises on an exhausted budget
    data, report = result.data, result.report
    json.dumps(report)
    assert report["attempts"] == 1 and report["time_budget_seconds"] == 50.0
    assert report["stopped_reason"].startswith("time budget of 50s exhausted")
    assert 0 < report["num_rows_generated"] == len(data) < report["num_rows_requested"]
    assert report["shortfall"] == report["num_rows_requested"] - len(data)
    assert any("increase time_budget_seconds" in w for w in report["warnings"])
    assert any("stopped early" in w for w in report["warnings"])
    assert _hard_valid(data, WD)
    if mode == "calendar":
        assert report["num_rows_requested"] == 118 and not data.duplicated(["ward_id", "date"]).any()
        assert (pd.to_datetime(data["date"]).dt.month == data["month"]).all()


def test_round_row_cap_and_proportional_batch_scaling():
    from bedsynth.synthesis.generator import DEFAULT_ROWS_PER_SECOND, MIN_ROUND_ROWS, _Cell

    cap = SyntheticDataGenerator._round_row_cap
    assert cap(1_000.0, 50_000.0, 200_000) == 200_000                  # max_round_rows binds
    assert cap(2.0, None, 200_000) == int(2.0 * DEFAULT_ROWS_PER_SECOND)  # default throughput before round 1
    assert cap(3.0, 20_000.0, 200_000) == 60_000                       # measured throughput x remaining budget
    assert cap(-4.0, 20_000.0, 200_000) == MIN_ROUND_ROWS              # spent budget: progress floor
    cells = [_Cell(conditions={}, requested=10, batch=batch) for batch in (60_000, 30_000, 10_000, 1)]
    assert SyntheticDataGenerator._cap_round(cells, 200_000) == 1.0 and cells[0].batch == 60_000
    scale = SyntheticDataGenerator._cap_round(cells, 10_000)
    assert scale == pytest.approx(10_000 / 100_001)
    assert [c.batch for c in cells] == [5_999, 2_999, 999, 1]          # proportional, at least one row each
    assert sum(c.batch for c in cells) <= 10_000


def test_time_budget_bounds_rounds_on_the_real_clock(generator):
    # A rare filter over ~20 condition cells: without the per-round cap the second round would ask
    # the model for up to 50,000 rows per cell (about a million rows) and overrun the budget >10x.
    budget = 2.0
    started = time.perf_counter()
    result = _generate(generator, table=PS, num_rows=20_000, model_id="gc_global", seed=8,
                       conditions={"ward_id": list(schema.WARD_IDS),
                                   "admission_source": list(schema.ADMISSION_SOURCES)},
                       filters={"bed_wait_hours": {"min": 9.5}}, time_budget_seconds=budget)
    elapsed = time.perf_counter() - started
    report = result.report
    assert elapsed < 2 * budget + 2.0, f"{elapsed:.1f}s for a {budget}s budget: {report['warnings']}"
    assert report["stopped_reason"].startswith("time budget")
    assert report["throttled_rounds"] >= 1 and report["rows_per_second"] > 0
    assert report["shortfall"] == 20_000 - len(result.data) > 0
    assert _hard_valid(result.data, PS) and (result.data["bed_wait_hours"] >= 9.5).all()


def test_sample_many_should_stop_skips_remaining_segments(generator):
    _, model = generator.get_model(PS, "gc_tuned")
    requests = [({"ward_id": "GMED"}, 30), ({"ward_id": "CARD"}, 30)]
    calls = []

    def stop_after_first_segment() -> bool:
        calls.append(1)
        return len(calls) > 1

    frames = model.sample_many(requests, seed=4, should_stop=stop_after_first_segment)
    assert [len(f) > 0 for f in frames].count(True) == 1    # the second ward was skipped
    complete = model.sample_many(requests, seed=4)
    assert all(len(f) == 30 for f in complete)


def test_budget_exhausted_before_the_first_round_returns_no_rows(generator):
    result = _generate(generator, table=WD, num_rows=40, time_budget_seconds=1e-9)
    assert result.data.empty and list(result.data.columns) == schema.output_columns(WD)
    report = result.report
    assert report["attempts"] == 0 and report["sampled_rows"] == 0 and report["shortfall"] == 40
    assert report["stopped_reason"].startswith("time budget")


# =========================================================================================
# Multi-column segments (patient_stays gc_tuned: ward x admission source)
# =========================================================================================
def test_ward_by_source_model_with_reference_strata(dirs, patient_stays_df, tmp_path):
    processed = tmp_path / "processed"
    processed.mkdir()
    patient_stays_df.to_csv(processed / "patient_stays_train.csv", index=False)
    engine = SyntheticDataGenerator(models_dir=dirs["models"], processed_dir=processed)
    result = engine.generate(GenerationRequest(table=PS, num_rows=400, model_id="gc_tuned", seed=6))
    data, report = result.data, result.report
    assert len(data) == 400 and _hard_valid(data, PS)
    assert report["model"]["segment_by"] == ["ward_id", "admission_source"]
    assert report["model"]["constraint_groups"] == list(CONSTRAINT_GROUPS)
    strata = report["allocation_strata"]
    assert strata["columns"] == ["ward_id", "admission_source", "delay_reason"]
    assert strata["source"] == "reference_train_split" and strata["strata_cells"] > 20
    keys = ["ward_id", "admission_source"]
    expected = patient_stays_df.groupby(keys).size() / len(patient_stays_df)
    observed = (data.groupby(keys).size() / len(data)).reindex(expected.index, fill_value=0.0)
    assert 0.5 * float((observed - expected).abs().sum()) <= 0.03 + strata["rebalanced_rows"] / 400

    electives = engine.generate(GenerationRequest(table=PS, num_rows=150, model_id="gc_tuned",
                                                  scenario="elective_admissions", seed=7)).data
    assert len(electives) == 150 and (electives["admission_source"] == "elective").all()
    hours = patient_stays_df.loc[patient_stays_df["admission_source"] == "elective", "admission_hour"]
    assert electives["admission_hour"].between(hours.min(), hours.max()).all()


def test_ward_by_source_model_without_reference(generator):
    # no reference split: strata weights come from the model's training support (tuple segment keys)
    result = _generate(generator, table=PS, num_rows=200, model_id="gc_tuned", scenario="emergency_admissions")
    data, report = result.data, result.report
    assert len(data) == 200 and (data["admission_source"] == "emergency").all()
    assert (data[["booking_lead_days", "times_rescheduled"]] == 0).all().all()
    assert report["allocation_strata"]["source"] == "model_training_data"
    assert report["allocation_strata"]["strata_cells"] > 5
    assert report["condition_allocation"][0]["requested"] == 200
    # both segment columns conditioned: one segment serves the request
    gsur = _generate(generator, table=PS, num_rows=60, model_id="gc_tuned", scenario="elective_admissions",
                     wards=["GSUR"]).data
    assert len(gsur) == 60 and (gsur["ward_id"] == "GSUR").all() and (gsur["booking_lead_days"] >= 1).all()
    # a combination no segment covers is reported, not raised
    amu = _generate(generator, table=PS, num_rows=20, model_id="gc_tuned", scenario="elective_admissions",
                    wards=["AMU"])
    assert amu.data.empty and any("never occurs" in w for w in amu.report["warnings"])
