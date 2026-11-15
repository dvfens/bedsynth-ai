"""Tests of the BedSynth AI FastAPI service (docs/DESIGN.md section 10).

Most tests replace the generation engine with :class:`FakeGenerator` (rule-satisfying fixture
rows and a plausible generation report) through ``create_app(generator=...)`` so they run in
milliseconds; validation, the rule catalogue and quick quality are always the real
``bedsynth.validation`` / ``bedsynth.evaluation`` implementations. The ``slow`` test trains a
real model on fixture data and drives the API through the real ``SyntheticDataGenerator``; the
``integration`` test runs against the project's own trained models when they exist.
"""
from __future__ import annotations

import csv
import importlib.util
import io
import json
import math
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import warnings
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Any, Iterator

import numpy as np
import pandas as pd
import pytest

warnings.filterwarnings("ignore", message=r".*httpx.*")  # starlette's TestClient prefers httpx2

from fastapi.testclient import TestClient  # noqa: E402

from bedsynth import schema  # noqa: E402
from bedsynth.api.main import create_app  # noqa: E402
from bedsynth.api.schemas import MAX_ROWS, VALIDATE_EXAMPLES  # noqa: E402
from bedsynth.api.service import (  # noqa: E402
    ENGINE_LOADING_DETAIL,
    ENGINE_NOT_LOADED_DETAIL,
    DatasetNotFoundError,
    DatasetStore,
    frame_to_records,
    prepare_records,
    to_jsonable,
)
from bedsynth.config import get_path  # noqa: E402
from bedsynth.data import io as data_io  # noqa: E402
from bedsynth.data.fixtures import make_patient_stays, make_ward_daily  # noqa: E402
from bedsynth.scenarios import (  # noqa: E402
    SCENARIOS,
    baseline_scenario,
    condition_mask,
    filter_mask,
    merge_conditions,
    merge_filters,
    validate_conditions,
    validate_filters,
)
from bedsynth.synthesis.generator import (  # noqa: E402
    GenerationError,
    GenerationRequest,
    GenerationResult,
    ModelNotFoundError,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VALID_WARD_DAY: dict[str, Any] = VALIDATE_EXAMPLES["ward_days"]["value"]["records"][0]
GENERATE = "/api/v1/generate"


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------
FAKE_MODEL_CARDS: dict[str, list[dict[str, Any]]] = {
    "ward_daily": [
        {
            "experiment_id": "gc_tuned",
            "table": "ward_daily",
            "experiment": {"experiment_id": "gc_tuned", "model": "gaussian_copula", "segment_by": "ward_id",
                           "description": "GaussianCopula segmented per ward"},
            "sdv_version": "2.2.0",
            "training_rows": 2240,
            "fit_seconds": 2.5,
            "fitted_at": "2026-11-01T09:00:00+00:00",
            "metrics": {"composite": 0.86},
        },
        {
            "experiment_id": "ctgan",
            "table": "ward_daily",
            "experiment": {"experiment_id": "ctgan", "model": "ctgan"},
            "sdv_version": "2.2.0",
            "training_rows": 2240,
            "fit_seconds": 61.0,
            "fitted_at": "2026-11-01T09:05:00+00:00",
        },
    ],
    "patient_stays": [
        {
            "experiment_id": "gc_tuned",
            "table": "patient_stays",
            "experiment": {"experiment_id": "gc_tuned", "model": "gaussian_copula", "segment_by": "ward_id"},
            "sdv_version": "2.2.0",
            "training_rows": 1900,
            "fit_seconds": 1.8,
            "fitted_at": "2026-11-01T09:10:00+00:00",
        },
    ],
}


@lru_cache(maxsize=None)
def _pool(table: str) -> pd.DataFrame:
    """Rule-satisfying rows the fake engine samples from (built once per session)."""
    if table == schema.WARD_DAILY:
        return make_ward_daily(n_days=365, seed=21, with_dates=False)   # a full year: every scenario has rows
    return make_patient_stays(n=2400, seed=22)


class FakeGenerator:
    """Duck-typed ``SyntheticDataGenerator`` serving fixture rows with a plausible report."""

    def __init__(self, trained: tuple[str, ...] = schema.TABLES) -> None:
        self.trained = tuple(trained)
        self.requests: list[GenerationRequest] = []

    def available_models(self, table: str | None = None) -> list[dict[str, Any]]:
        tables = [table] if table else list(schema.TABLES)
        return [dict(card) for name in tables if name in self.trained for card in FAKE_MODEL_CARDS[name]]

    def default_model_id(self, table: str) -> str | None:
        return "gc_tuned" if table in self.trained else None

    def generate(self, request: GenerationRequest) -> GenerationResult:
        self.requests.append(request)
        table = request.table
        if table not in self.trained:
            raise ModelNotFoundError(f"No trained model for table '{table}'.")
        model_id = request.model_id or "gc_tuned"
        if model_id not in {card["experiment_id"] for card in FAKE_MODEL_CARDS[table]}:
            raise ModelNotFoundError(f"Model '{model_id}' is not trained for '{table}'.")
        scenario = SCENARIOS[request.scenario] if request.scenario else baseline_scenario(table)
        if scenario.table != table:
            raise GenerationError(f"Scenario '{scenario.name}' belongs to '{scenario.table}'.")
        try:
            conditions = merge_conditions(dict(scenario.conditions), validate_conditions(table, request.conditions))
            if request.wards:
                conditions = merge_conditions(conditions, {"ward_id": tuple(request.wards)})
            filters = merge_filters(dict(scenario.filters), validate_filters(table, request.filters))
        except ValueError as exc:
            raise GenerationError(str(exc)) from exc

        seed = 1234 if request.seed is None else request.seed
        pool = _pool(table)
        candidates = pool[condition_mask(pool, conditions) & filter_mask(pool, filters)]
        data = self._calendar(request, candidates, seed) if request.mode == "calendar" else self._sample(table, request.num_rows, candidates, seed)
        if not request.include_derived:
            data = data[[c for c in schema.column_names(table, ["key", "time", "train"]) if c in data.columns]]

        n_rows = len(data)
        requested = n_rows if request.mode == "calendar" else request.num_rows
        report = {
            "table": table, "model_id": model_id, "scenario": scenario.name, "mode": request.mode, "seed": seed,
            "num_rows_requested": requested, "num_rows_generated": n_rows, "shortfall": max(0, requested - n_rows),
            "attempts": 1, "sampled_rows": math.ceil(n_rows * 1.25), "accepted_rows": n_rows,
            "acceptance_rate": 0.8 if n_rows else 0.0, "raw_hard_valid_rate": np.float64(0.93),
            "rejected_by_rule": {"WD-H06" if table == "ward_daily" else "PS-H08": np.int64(math.ceil(n_rows * 0.07))},
            "rejected_by_filter": math.ceil(n_rows * 0.13),
            "condition_allocation": [{"conditions": {k: list(v) for k, v in conditions.items()}, "requested": requested, "generated": n_rows}],
            "elapsed_seconds": 0.01,
            "warnings": [] if n_rows >= requested else [f"Shortfall: generated {n_rows} of {requested} rows."],
        }
        return GenerationResult(data=data, report=report)

    @staticmethod
    def _sample(table: str, num_rows: int, candidates: pd.DataFrame, seed: int) -> pd.DataFrame:
        if candidates.empty:
            return candidates.reset_index(drop=True)
        rows = candidates.sample(n=num_rows, replace=len(candidates) < num_rows, random_state=seed)
        return schema.assign_ids(rows.reset_index(drop=True), table)

    @staticmethod
    def _calendar(request: GenerationRequest, candidates: pd.DataFrame, seed: int) -> pd.DataFrame:
        dates = pd.date_range(request.start_date, request.end_date, freq="D")
        frames = []
        for offset, ward in enumerate(request.wards or schema.WARD_IDS):
            rows = candidates[candidates["ward_id"] == ward]
            if rows.empty:
                continue
            picked = rows.sample(n=len(dates), replace=True, random_state=seed + offset).reset_index(drop=True)
            picked["date"] = dates.strftime(schema.DATE_FORMAT)
            picked["month"] = dates.month.astype(int)
            picked["day_of_week"] = [schema.DAYS_OF_WEEK[day] for day in dates.dayofweek]
            frames.append(picked[[c for c in picked.columns if c not in schema.derived_columns("ward_daily")]])
        data = schema.derive("ward_daily", pd.concat(frames, ignore_index=True))
        return schema.assign_ids(data.sort_values(["ward_id", "date"]).reset_index(drop=True), "ward_daily")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture()
def fake_generator() -> FakeGenerator:
    return FakeGenerator()


@pytest.fixture()
def client(isolated_paths: dict[str, Path], fake_generator: FakeGenerator) -> Iterator[TestClient]:
    with TestClient(create_app(generator=fake_generator)) as test_client:
        yield test_client


@pytest.fixture()
def reference_data(isolated_paths: dict[str, Path]) -> Path:
    """Fixture train splits in the isolated processed dir (the API's reference data)."""
    data_io.write_table(make_ward_daily(n_days=60, seed=5), data_io.processed_path("ward_daily", "train"))
    data_io.write_table(make_patient_stays(n=600, seed=6, with_times=True), data_io.processed_path("patient_stays", "train"))
    return isolated_paths["processed_dir"]


def _generate(client: TestClient, **payload: Any) -> dict[str, Any]:
    response = client.post(GENERATE, json=payload)
    assert response.status_code == 200, response.text
    return response.json()


def _load_script(path: Path, name: str) -> ModuleType:
    """Import a script by path (registered in sys.modules first, as dataclasses require)."""
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Service endpoints
# ---------------------------------------------------------------------------
def test_root_lists_links_and_every_endpoint(client: TestClient) -> None:
    body = client.get("/").json()
    assert body["links"]["docs"] == "/docs" and body["links"]["ui"] == "/ui"
    endpoints = {(item["method"], item["path"]) for item in body["endpoints"]}
    expected = {
        ("GET", "/health"), ("GET", "/ui"), ("GET", "/api/v1/tables"), ("GET", "/api/v1/tables/{table}/schema"),
        ("GET", "/api/v1/scenarios"), ("GET", "/api/v1/scenarios/{name}"), ("GET", "/api/v1/models"),
        ("POST", "/api/v1/generate"), ("POST", "/api/v1/validate"), ("POST", "/api/v1/evaluate"),
        ("GET", "/api/v1/datasets"), ("GET", "/api/v1/datasets/{dataset_id}"),
        ("GET", "/api/v1/datasets/{dataset_id}/download"), ("DELETE", "/api/v1/datasets/{dataset_id}"),
    }
    assert expected <= endpoints


def test_health_reports_models_and_reference_data(client: TestClient, reference_data: Path) -> None:
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["engine"] == {"available": True, "loaded": True, "state": "ready", "detail": None}
    assert body["models"]["ward_daily"] == {"trained": ["gc_tuned", "ctgan"], "default": "gc_tuned"}
    assert body["trained_tables"] == ["ward_daily", "patient_stays"]
    assert body["reference_data"] == {"ward_daily": True, "patient_stays": True}
    assert body["sdv_version"] and body["messages"] == []


def test_health_is_degraded_when_a_table_is_untrained(isolated_paths: dict[str, Path]) -> None:
    with TestClient(create_app(generator=FakeGenerator(trained=("ward_daily",)))) as test_client:
        health = test_client.get("/health").json()
        response = test_client.post(GENERATE, json={"table": "patient_stays", "num_rows": 10})
    assert health["status"] == "degraded" and health["trained_tables"] == ["ward_daily"]
    assert any("patient_stays" in message and "02_train_synthesizers" in message for message in health["messages"])
    assert response.status_code == 503
    assert response.json()["error_type"] == "model_unavailable"
    assert "02_train_synthesizers" in response.json()["detail"]


def test_unavailable_engine_degrades_gracefully(isolated_paths: dict[str, Path]) -> None:
    """/health never builds the engine: the failure shows once a request has tried to (fixed message only)."""
    def broken_engine() -> Any:
        raise RuntimeError(r"model store unreachable at C:\Users\someone\models")

    with TestClient(create_app(generator_factory=broken_engine)) as test_client:
        before = test_client.get("/health").json()
        generate = test_client.post(GENERATE, json={"table": "ward_daily", "num_rows": 5})
        health = test_client.get("/health").json()
        models = test_client.get("/api/v1/models")
        tables = test_client.get("/api/v1/tables")
    assert before["engine"] == {"available": True, "loaded": False, "state": "not_loaded", "detail": ENGINE_NOT_LOADED_DETAIL}
    fixed = "The generation engine could not be initialised (RuntimeError); see the server log."
    assert generate.status_code == 503 and generate.json() == {"detail": fixed, "error_type": "engine_unavailable"}
    assert health["status"] == "degraded"
    assert health["engine"] == {"available": False, "loaded": False, "state": "failed", "detail": fixed}
    assert health["messages"][0] == fixed
    assert models.status_code == 503 and models.json()["detail"] == fixed
    assert tables.status_code == 200 and tables.json()["tables"][0]["default_model"] is None
    for text in (generate.text, json.dumps(health), models.text, tables.text):
        assert "someone" not in text and "unreachable" not in text


def test_module_level_app_initialises_the_engine_lazily() -> None:
    from bedsynth.api import main

    assert main.app.state.service.engine_loaded is False
    assert main.app.state.service.store.base_dir.name == "api"


def _write_registry(
    models_dir: Path,
    models: dict[str, list[str]],
    defaults: dict[str, str] | None = None,
    without_file: tuple[tuple[str, str], ...] = (),
) -> None:
    """A registry.json in the layout of bedsynth.synthesis.registry, with placeholder model files."""
    registry: dict[str, Any] = {}
    for table, model_ids in models.items():
        registry[table] = {
            "default": (defaults or {}).get(table),
            "default_reason": "",
            "models": {model_id: {"experiment_id": model_id, "table": table, "synthesizer": "GaussianCopulaSynthesizer"}
                       for model_id in model_ids},
        }
        for model_id in model_ids:
            if (table, model_id) not in without_file:
                (models_dir / table / model_id).mkdir(parents=True, exist_ok=True)
                (models_dir / table / model_id / "synthesizer.pkl").write_bytes(b"placeholder")
    (models_dir / "registry.json").write_text(json.dumps(registry), encoding="utf-8")


def _wait_for_engine(test_client: TestClient, timeout: float = 10.0) -> dict[str, Any]:
    """Poll /health until the engine has left the 'loading' state."""
    deadline = time.monotonic() + timeout
    while True:
        body = test_client.get("/health").json()
        if body["engine"]["state"] != "loading" or time.monotonic() > deadline:
            return body
        time.sleep(0.02)


def test_health_answers_at_once_while_the_engine_loads(isolated_paths: dict[str, Path]) -> None:
    """/health never waits for the start-up preload: it says 'loading' and lists the registry's models."""
    _write_registry(isolated_paths["models_dir"], {"ward_daily": ["gc_tuned", "ctgan"], "patient_stays": ["gc_tuned"]},
                    defaults={"ward_daily": "gc_tuned"})
    release = threading.Event()
    factory_calls: list[float] = []

    def slow_engine() -> FakeGenerator:      # stands in for the SDV import of the real engine
        factory_calls.append(time.perf_counter())
        assert release.wait(10), "the test did not release the engine"
        return FakeGenerator()

    with TestClient(create_app(generator_factory=slow_engine, preload_engine=True)) as test_client:
        started = time.perf_counter()
        loading = test_client.get("/health")
        elapsed = time.perf_counter() - started
        release.set()
        ready = _wait_for_engine(test_client)
    assert loading.status_code == 200 and elapsed < 1.0
    body = loading.json()
    assert body["status"] == "degraded"
    assert body["engine"] == {"available": True, "loaded": False, "state": "loading", "detail": ENGINE_LOADING_DETAIL}
    assert body["models"] == {"ward_daily": {"trained": ["ctgan", "gc_tuned"], "default": "gc_tuned"},
                              "patient_stays": {"trained": ["gc_tuned"], "default": "gc_tuned"}}
    assert body["trained_tables"] == ["ward_daily", "patient_stays"] and ENGINE_LOADING_DETAIL in body["messages"]
    assert ready["status"] == "ok" and ready["engine"] == {"available": True, "loaded": True, "state": "ready", "detail": None}
    assert ready["models"]["ward_daily"] == {"trained": ["gc_tuned", "ctgan"], "default": "gc_tuned"}  # listed by the engine
    assert len(factory_calls) == 1


def test_health_reports_the_engine_state_at_call_time(isolated_paths: dict[str, Path]) -> None:
    """engine.loaded is false until a request creates the engine (no preload); /health never creates it."""
    _write_registry(isolated_paths["models_dir"], {"ward_daily": ["gc_tuned"], "patient_stays": ["gc_tuned"]})
    with TestClient(create_app(generator_factory=FakeGenerator)) as test_client:
        first = test_client.get("/health").json()
        second = test_client.get("/health").json()
        assert test_client.app.state.service.engine_loaded is False
        assert test_client.get("/api/v1/models").status_code == 200       # this request creates the engine
        after = test_client.get("/health").json()
    for body in (first, second):
        assert body["engine"] == {"available": True, "loaded": False, "state": "not_loaded", "detail": ENGINE_NOT_LOADED_DETAIL}
        assert body["status"] == "ok" and body["trained_tables"] == ["ward_daily", "patient_stays"]
    assert after["engine"] == {"available": True, "loaded": True, "state": "ready", "detail": None}


def test_failed_preload_is_reported_without_exception_text(isolated_paths: dict[str, Path]) -> None:
    def broken_engine() -> Any:
        raise PermissionError(13, "Permission denied", r"C:\Users\someone\models\registry.json")

    with TestClient(create_app(generator_factory=broken_engine, preload_engine=True)) as test_client:
        body = _wait_for_engine(test_client)
    fixed = "The generation engine could not be initialised (PermissionError); see the server log."
    assert body["status"] == "degraded"
    assert body["engine"] == {"available": False, "loaded": False, "state": "failed", "detail": fixed}
    assert "someone" not in json.dumps(body) and "Permission denied" not in json.dumps(body)


def test_registry_file_status_mirrors_the_model_registry(isolated_paths: dict[str, Path]) -> None:
    """Without the engine, trained models and defaults follow the ModelRegistry rules exactly."""
    from bedsynth.api.service import BedSynthService
    from bedsynth.synthesis.registry import ModelRegistry

    models_dir = isolated_paths["models_dir"]
    _write_registry(models_dir, {"ward_daily": ["tvae", "gc_tuned", "ctgan", "../escape"], "patient_stays": ["tvae", "ctgan"]},
                    defaults={"ward_daily": "tvae", "patient_stays": "gc_tuned"}, without_file=(("ward_daily", "tvae"),))
    service = BedSynthService(generator_factory=FakeGenerator)
    trained, defaults, error = service._registry_file_status()
    registry = ModelRegistry(models_dir)
    for table in schema.TABLES:
        assert trained[table] == sorted(registry.trained_models(table))
        assert defaults[table] == registry.default_model_id(table)
    assert error is None and service.engine_loaded is False
    assert trained == {"ward_daily": ["ctgan", "gc_tuned"], "patient_stays": ["ctgan", "tvae"]}  # tvae has no file

    (models_dir / "registry.json").write_text("{not json", encoding="utf-8")
    with TestClient(create_app(generator_factory=FakeGenerator)) as test_client:
        corrupt = test_client.get("/health").json()
    assert corrupt["status"] == "degraded" and corrupt["trained_tables"] == []
    assert "The model registry could not be read (JSONDecodeError); see the server log." in corrupt["messages"]


def test_health_without_the_engine_does_not_import_sdv(isolated_paths: dict[str, Path]) -> None:
    """The registry file is read with json only: answering /health never imports SDV or torch."""
    _write_registry(isolated_paths["models_dir"], {"ward_daily": ["gc_tuned"]})
    code = (
        "import sys, warnings\n"
        "warnings.filterwarnings('ignore')\n"
        "from fastapi.testclient import TestClient\n"
        "from bedsynth.api.main import create_app\n"
        "with TestClient(create_app()) as client:\n"
        "    body = client.get('/health').json()\n"
        "assert body['models']['ward_daily']['trained'] == ['gc_tuned'], body\n"
        "assert body['engine']['state'] == 'not_loaded', body\n"
        "heavy = sorted({name.split('.')[0] for name in sys.modules} & {'sdv', 'torch', 'copulas', 'ctgan', 'rdt'})\n"
        "print('HEAVY=' + ','.join(heavy))\n"
    )
    result = subprocess.run([sys.executable, "-c", code], cwd=PROJECT_ROOT, env=dict(os.environ),
                            capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stderr[-2000:]
    assert result.stdout.strip().splitlines()[-1] == "HEAVY=", result.stdout


def test_health_hides_model_listing_errors(isolated_paths: dict[str, Path]) -> None:
    class BrokenCatalogue(FakeGenerator):
        def available_models(self, table: str | None = None) -> list[dict[str, Any]]:
            raise TypeError("registry exploded at /home/someone/models/registry.json")

    with TestClient(create_app(generator=BrokenCatalogue())) as test_client:
        body = test_client.get("/health").json()
    fixed = "Could not list trained models (TypeError); see the server log."
    assert body["status"] == "degraded" and body["messages"][0] == fixed
    assert body["engine"] == {"available": False, "loaded": True, "state": "ready", "detail": fixed}
    assert "someone" not in json.dumps(body) and "exploded" not in json.dumps(body)


# ---------------------------------------------------------------------------
# Catalogue
# ---------------------------------------------------------------------------
def test_tables_and_schema(client: TestClient) -> None:
    tables = {table["name"]: table for table in client.get("/api/v1/tables").json()["tables"]}
    assert set(tables) == set(schema.TABLES)
    ward_daily = tables["ward_daily"]
    assert ward_daily["n_training_columns"] == 22 and ward_daily["default_model"] == "gc_tuned"
    assert ward_daily["supports_calendar_mode"] and not tables["patient_stays"]["supports_calendar_mode"]
    assert tables["patient_stays"]["n_training_columns"] == 15 and tables["patient_stays"]["n_scenarios"] == 10

    detail = client.get("/api/v1/tables/patient_stays/schema").json()
    assert detail["key"] == "stay_id" and len(detail["wards"]) == len(schema.WARD_IDS)
    assert [column["name"] for column in detail["columns"]] == schema.column_names("patient_stays")
    assert detail["rules_available"] and any(rule["rule_id"].startswith("PS-H") for rule in detail["rules"])
    assert "long_bed_waits" in detail["scenarios"]


@pytest.mark.parametrize("path", [
    "/api/v1/tables/beds/schema", "/api/v1/scenarios?table=beds", "/api/v1/models?table=beds", "/api/v1/datasets?table=beds",
])
def test_unknown_table_returns_404(client: TestClient, path: str) -> None:
    response = client.get(path)
    assert response.status_code == 404
    assert response.json()["error_type"] == "table_not_found"


def test_scenario_catalogue_and_reference_support(client: TestClient, reference_data: Path) -> None:
    everything = client.get("/api/v1/scenarios").json()
    assert everything["count"] == len(SCENARIOS) == 26
    assert all(item["support"] is None for item in everything["scenarios"])

    stays = client.get("/api/v1/scenarios", params={"table": "patient_stays", "include_support": True}).json()
    assert {item["table"] for item in stays["scenarios"]} == {"patient_stays"}
    assert all(item["support"]["reference_rows"] == 600 for item in stays["scenarios"])

    one = client.get("/api/v1/scenarios/High_Occupancy").json()
    assert one["name"] == "high_occupancy" and one["conditions"] == {"occupancy_level": ["high", "critical"]}
    assert 0 < one["support"]["matching_rows"] < one["support"]["reference_rows"]

    missing = client.get("/api/v1/scenarios/icu_meltdown")
    assert missing.status_code == 404 and missing.json()["error_type"] == "scenario_not_found"


def test_models_listing(client: TestClient) -> None:
    tables = client.get("/api/v1/models").json()["tables"]
    ward_daily = tables["ward_daily"]
    assert ward_daily["default_model"] == "gc_tuned"
    gc_tuned = next(model for model in ward_daily["models"] if model["model_id"] == "gc_tuned")
    assert gc_tuned["is_default"] and gc_tuned["synthesizer"] == "gaussian_copula"
    assert gc_tuned["segment_by"] == "ward_id" and gc_tuned["metrics"] == {"composite": 0.86}
    assert gc_tuned["training_rows"] == 2240 and gc_tuned["card"]["fit_seconds"] == 2.5
    only_stays = client.get("/api/v1/models", params={"table": "patient_stays"}).json()["tables"]
    assert list(only_stays) == ["patient_stays"]


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
def test_generate_returns_records_reports_and_persists(
    client: TestClient, fake_generator: FakeGenerator, isolated_paths: dict[str, Path]
) -> None:
    body = _generate(client, table="ward_daily", num_rows=40, scenario="high_occupancy", seed=7)
    assert body["n_rows"] == len(body["records"]) == 40
    assert DatasetStore.is_valid_id(body["dataset_id"]) and body["persisted"] is True
    assert body["model"] == {"id": "gc_tuned", "synthesizer": "gaussian_copula", "sdv_version": "2.2.0",
                             "fitted_at": "2026-11-01T09:00:00+00:00"}
    assert body["scenario"]["name"] == "high_occupancy" and body["scenario"]["category"] == "occupancy"
    assert body["generation"]["seed"] == 7 and body["generation"]["raw_hard_valid_rate"] == 0.93
    assert body["validation"]["all_hard_rules_passed"] is True and body["quality"] is None
    assert {record["occupancy_level"] for record in body["records"]} <= {"high", "critical"}
    roles = {column["name"]: column["role"] for column in body["columns"]}
    assert roles["record_id"] == "key" and roles["ward_id"] == "train" and roles["occupancy_rate"] == "derived"
    assert body["links"]["download_csv"].endswith(f"{body['dataset_id']}/download?format=csv")
    json.dumps(body, allow_nan=False)  # strictly valid JSON (no NaN)

    stored = isolated_paths["synthetic_dir"] / "api" / body["dataset_id"]
    assert (stored / "data.csv").is_file() and (stored / "metadata.json").is_file()
    sent = fake_generator.requests[-1]
    assert (sent.table, sent.scenario, sent.num_rows, sent.seed, sent.mode) == ("ward_daily", "high_occupancy", 40, 7, "sample")


def test_generate_normalises_wards_conditions_and_filters(client: TestClient, fake_generator: FakeGenerator) -> None:
    body = _generate(
        client, table="ward_daily", num_rows=25, seed=3,
        wards=["amu", "Intensive Care Unit", "AMU"],
        conditions={"day_of_week": ["monday", "Tue"], "month": ["1", 2]},
        filters={"emergency_boarders": {"max": 3}},
    )
    sent = fake_generator.requests[-1]
    assert sent.wards == ["AMU", "ICU"]
    assert sent.conditions == {"day_of_week": ["Mon", "Tue"], "month": [1, 2]}
    assert sent.filters == {"emergency_boarders": {"max": 3}}
    assert body["request"]["wards"] == ["AMU", "ICU"] and body["request"]["validate"] is True
    assert {record["ward_id"] for record in body["records"]} <= {"AMU", "ICU"}
    assert {record["day_of_week"] for record in body["records"]} <= {"Mon", "Tue"}


def test_generate_is_reproducible_for_a_seed(client: TestClient) -> None:
    payload = {"table": "patient_stays", "num_rows": 30, "scenario": "emergency_admissions", "seed": 11, "persist": False}
    first, second = _generate(client, **payload), _generate(client, **payload)
    assert first["records"] == second["records"]
    assert first["dataset_id"] != second["dataset_id"]


def test_generate_csv_carries_bedsynth_headers(client: TestClient) -> None:
    response = client.post(GENERATE, json={
        "table": "patient_stays", "num_rows": 30, "scenario": "delayed_discharge_stays", "format": "csv", "seed": 5,
    })
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/csv")
    assert response.headers["x-bedsynth-rows"] == "30"
    assert response.headers["x-bedsynth-model"] == "gc_tuned"
    assert response.headers["x-bedsynth-scenario"] == "delayed_discharge_stays"
    assert response.headers["x-bedsynth-hard-rules-passed"] == "true"
    dataset_id = response.headers["x-bedsynth-dataset-id"]
    assert DatasetStore.is_valid_id(dataset_id)
    assert f'filename="{dataset_id}.csv"' in response.headers["content-disposition"]
    rows = list(csv.DictReader(io.StringIO(response.text)))
    assert len(rows) == 30 and all(row["delay_reason"] != "none" for row in rows)
    stored = client.get(f"/api/v1/datasets/{dataset_id}").json()
    assert stored["validation"]["all_hard_rules_passed"] and stored["request"]["format"] == "csv"


def test_generate_calendar_mode(client: TestClient) -> None:
    body = _generate(client, table="ward_daily", mode="calendar", wards=["AMU", "ICU"],
                     start_date="2025-01-30", end_date="2025-02-05", seed=1)
    frame = pd.DataFrame(body["records"])
    assert body["n_rows"] == 14 and frame.groupby("ward_id").size().to_dict() == {"AMU": 7, "ICU": 7}
    assert (frame["date"].min(), frame["date"].max()) == ("2025-01-30", "2025-02-05")
    assert set(frame["month"]) == {1, 2}
    assert [column["name"] for column in body["columns"]][:3] == ["record_id", "date", "ward_id"]
    assert body["request"]["start_date"] == "2025-01-30"


def test_generate_without_derived_columns_still_validates(client: TestClient) -> None:
    body = _generate(client, table="ward_daily", num_rows=20, include_derived=False, seed=2)
    names = {column["name"] for column in body["columns"]}
    assert "occupancy_rate" not in names and "census_start" in names
    assert body["validation"]["all_hard_rules_passed"]


def test_generate_without_persisting(client: TestClient) -> None:
    body = _generate(client, table="ward_daily", num_rows=5, persist=False, seed=4)
    assert body["persisted"] is False and body["links"] == {}
    assert client.get(f"/api/v1/datasets/{body['dataset_id']}").status_code == 404


def test_generate_with_evaluate_uses_the_scenario_reference_subset(client: TestClient, reference_data: Path) -> None:
    body = _generate(client, table="ward_daily", num_rows=60, scenario="high_occupancy", evaluate=True, seed=2)
    quality = body["quality"]
    assert 0.0 <= quality["score"] <= 1.0
    assert quality["reference"]["subset"] == "matching" and quality["reference"]["synthetic_rows_compared"] == 60
    assert quality["reference"]["reference_rows"] < 480  # 60 days x 8 wards, filtered to the scenario
    assert "score" in quality["metrics"]


def test_generate_evaluate_without_reference_data_warns(client: TestClient) -> None:
    body = _generate(client, table="ward_daily", num_rows=10, evaluate=True, seed=2)
    assert body["quality"] is None
    assert any("Reference data" in warning and "01_prepare_data" in warning for warning in body["warnings"])


@pytest.mark.parametrize("payload, fragment", [
    ({"table": "ward_daily", "num_rows": 0}, "num_rows"),
    ({"table": "ward_daily", "num_rows": MAX_ROWS + 1}, "num_rows"),
    ({"table": "beds"}, "table"),
    ({"table": "ward_daily", "wards": ["XYZ"]}, "Unknown ward"),
    ({"table": "ward_daily", "nrows": 5}, "nrows"),
    ({"table": "ward_daily", "model_id": "../../etc"}, "model_id"),
    ({"table": "ward_daily", "seed": -1}, "seed"),
    ({"table": "ward_daily", "filters": {"census_start": {}}}, "at least one"),
    ({"table": "ward_daily", "filters": {"census_start": {"min": 5, "max": 1}}}, "greater than"),
    ({"table": "patient_stays", "mode": "calendar", "start_date": "2025-01-01", "end_date": "2025-01-02"}, "ward_daily"),
    ({"table": "ward_daily", "mode": "calendar", "start_date": "2025-01-01"}, "start_date and end_date"),
    ({"table": "ward_daily", "mode": "calendar", "start_date": "2025-02-01", "end_date": "2025-01-01"}, "on or after"),
    ({"table": "ward_daily", "mode": "calendar", "start_date": "2020-01-01", "end_date": "2025-12-31"}, "maximum"),
    ({"table": "ward_daily", "start_date": "2025-01-01"}, "calendar mode"),
])
def test_generate_rejects_malformed_requests_with_422(client: TestClient, payload: dict[str, Any], fragment: str) -> None:
    response = client.post(GENERATE, json=payload)
    assert response.status_code == 422, response.text
    body = response.json()
    assert body["error_type"] == "validation_error" and fragment in body["detail"]
    assert body["errors"] and set(body["errors"][0]) == {"loc", "msg", "type"}


@pytest.mark.parametrize("payload, status, error_type", [
    ({"table": "ward_daily", "scenario": "icu_meltdown"}, 404, "scenario_not_found"),
    ({"table": "ward_daily", "scenario": "long_bed_waits"}, 400, "scenario_table_mismatch"),
    ({"table": "ward_daily", "model_id": "tvae"}, 404, "model_not_found"),
    ({"table": "ward_daily", "conditions": {"occupancy_level": "overflowing"}}, 400, "invalid_conditions"),
    ({"table": "ward_daily", "conditions": {"census_start": 3}}, 400, "invalid_conditions"),
    ({"table": "ward_daily", "filters": {"no_such_column": {"min": 1}}}, 400, "invalid_filters"),
    ({"table": "patient_stays", "filters": {"discharge_day_of_week": {"in": ["Funday"]}}}, 400, "invalid_filters"),
    ({"table": "ward_daily", "scenario": "scheduled_procedures", "wards": ["AMU"]}, 400, "generation_error"),
])
def test_generate_semantic_errors(client: TestClient, payload: dict[str, Any], status: int, error_type: str) -> None:
    response = client.post(GENERATE, json=payload)
    assert response.status_code == status, response.text
    assert response.json()["error_type"] == error_type and response.json()["detail"]


def test_generate_maps_unexpected_engine_failures_to_500(isolated_paths: dict[str, Path]) -> None:
    class ExplodingGenerator(FakeGenerator):
        def generate(self, request: GenerationRequest) -> GenerationResult:
            raise RuntimeError("boom")

    with TestClient(create_app(generator=ExplodingGenerator())) as test_client:
        response = test_client.post(GENERATE, json={"table": "ward_daily", "num_rows": 5})
    assert response.status_code == 500
    assert response.json()["error_type"] == "internal_error" and "boom" not in response.json()["detail"]


# ---------------------------------------------------------------------------
# Validation and evaluation of client records
# ---------------------------------------------------------------------------
def test_validate_normalises_aliases_derives_and_flags(client: TestClient) -> None:
    response = client.post("/api/v1/validate", json=VALIDATE_EXAMPLES["ward_days"]["value"])
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["n_rows"] == 2 and body["n_invalid_rows"] == 1 and body["invalid_rows"] == [1]
    assert body["normalisation"]["values_normalised"] == {"ward_id": 1, "day_of_week": 1, "staffing_level": 1}
    assert "census_end" in body["normalisation"]["derived_computed"]
    wd_h03 = next(rule for rule in body["validation"]["rules"] if rule["rule_id"] == "WD-H03")
    assert not wd_h03["passed"] and wd_h03["example_rows"] == [1]
    assert body["records"][1]["ward_id"] == "GMED" and body["records"][1]["day_of_week"] == "Mon"
    assert body["records"][0]["census_end"] == 30 and body["records"][0]["occupancy_level"] == "high"


def test_validate_reports_mismatches_unknown_values_and_columns(client: TestClient) -> None:
    record = dict(VALID_WARD_DAY, census_end=99, ward_id="MARS", notes="free text", census_start="thirty")
    body = client.post("/api/v1/validate", json={"table": "ward_daily", "records": [VALID_WARD_DAY, record]}).json()
    normalisation = body["normalisation"]
    assert normalisation["unknown_values"] == {"ward_id": 1}
    assert normalisation["invalid_numbers"] == {"census_start": 1}
    assert normalisation["unknown_columns"] == ["notes"]
    assert body["invalid_rows"] == [1] and len(body["warnings"]) == 3
    assert body["records"] is None

    consistent = client.post("/api/v1/validate", json={"table": "ward_daily", "records": [dict(VALID_WARD_DAY, census_end=31)]}).json()
    assert consistent["normalisation"]["derived_mismatches"] == {"census_end": 1}
    assert any("census_end (1)" in warning for warning in consistent["warnings"])


def test_validate_requires_every_training_column(client: TestClient) -> None:
    response = client.post("/api/v1/validate", json={"table": "patient_stays", "records": [{"ward_id": "AMU"}]})
    assert response.status_code == 400
    assert response.json()["error_type"] == "invalid_records" and "admission_source" in response.json()["detail"]


def test_validate_patient_stay_example_is_valid(client: TestClient) -> None:
    body = client.post("/api/v1/validate", json=VALIDATE_EXAMPLES["stay"]["value"]).json()
    assert body["validation"]["all_hard_rules_passed"] and body["n_invalid_rows"] == 0


def test_evaluate_stored_dataset_and_client_records(client: TestClient, reference_data: Path) -> None:
    generated = _generate(client, table="patient_stays", num_rows=80, scenario="emergency_admissions", seed=4)
    by_dataset = client.post("/api/v1/evaluate", json={"dataset_id": generated["dataset_id"]})
    assert by_dataset.status_code == 200, by_dataset.text
    body = by_dataset.json()
    assert body["source"] == {"type": "dataset", "dataset_id": generated["dataset_id"], "n_rows": 80}
    assert body["quality"]["reference"]["subset"] == "matching" and 0.0 <= body["quality"]["score"] <= 1.0
    assert body["validation"]["all_hard_rules_passed"] and body["normalisation"] is None

    by_records = client.post("/api/v1/evaluate", json={
        "table": "patient_stays", "records": generated["records"][:40], "reference_split": "train", "include_validation": False,
    }).json()
    assert by_records["source"]["type"] == "records" and by_records["quality"]["reference"]["subset"] == "full"
    assert by_records["validation"] is None and by_records["normalisation"]["derived_computed"] == []


def test_evaluate_errors(client: TestClient, reference_data: Path) -> None:
    records = [VALID_WARD_DAY] * 20
    for payload in ({}, {"table": "ward_daily"}, {"records": records},
                    {"table": "ward_daily", "records": records[:5]},
                    {"dataset_id": "ds-0123456789ab", "table": "ward_daily", "records": records}):
        assert client.post("/api/v1/evaluate", json=payload).status_code == 422, payload
    missing = client.post("/api/v1/evaluate", json={"dataset_id": "ds-0123456789ab"})
    assert missing.status_code == 404 and missing.json()["error_type"] == "dataset_not_found"
    dataset_id = _generate(client, table="ward_daily", num_rows=20, seed=1)["dataset_id"]
    mismatch = client.post("/api/v1/evaluate", json={"dataset_id": dataset_id, "table": "patient_stays"})
    assert mismatch.status_code == 400 and mismatch.json()["error_type"] == "table_mismatch"


def test_evaluate_without_reference_data_returns_503(client: TestClient) -> None:
    response = client.post("/api/v1/evaluate", json={"table": "ward_daily", "records": [VALID_WARD_DAY] * 20})
    assert response.status_code == 503
    assert response.json()["error_type"] == "reference_data_unavailable"


# ---------------------------------------------------------------------------
# Persisted datasets
# ---------------------------------------------------------------------------
def test_dataset_lifecycle(client: TestClient) -> None:
    created = _generate(client, table="ward_daily", num_rows=12, scenario="winter_demand", seed=9, evaluate=False)
    dataset_id = created["dataset_id"]
    other = _generate(client, table="patient_stays", num_rows=6, seed=1)

    listing = client.get("/api/v1/datasets").json()
    assert listing["total"] == 2 and {item["dataset_id"] for item in listing["datasets"]} == {dataset_id, other["dataset_id"]}
    only_wards = client.get("/api/v1/datasets", params={"table": "ward_daily", "limit": 5}).json()
    summary = only_wards["datasets"][0]
    assert only_wards["total"] == 1 and summary["dataset_id"] == dataset_id
    assert (summary["n_rows"], summary["seed"], summary["scenario"], summary["model_id"]) == (12, 9, "winter_demand", "gc_tuned")
    assert summary["all_hard_rules_passed"] is True and summary["size_bytes"] > 0

    detail = client.get(f"/api/v1/datasets/{dataset_id}").json()
    assert "records" not in detail and detail["generation"]["seed"] == 9 and detail["size_bytes"] > 0

    as_csv = client.get(f"/api/v1/datasets/{dataset_id}/download")
    assert as_csv.headers["content-type"].startswith("text/csv")
    assert as_csv.headers["x-bedsynth-dataset-id"] == dataset_id and as_csv.headers["x-bedsynth-rows"] == "12"
    assert as_csv.headers["x-bedsynth-scenario"] == "winter_demand" and "attachment" in as_csv.headers["content-disposition"]
    assert len(list(csv.DictReader(io.StringIO(as_csv.text)))) == 12

    as_json = client.get(f"/api/v1/datasets/{dataset_id}/download", params={"format": "json"})
    assert as_json.json()["records"] == created["records"]  # lossless CSV round trip
    assert client.get(f"/api/v1/datasets/{dataset_id}/download", params={"format": "xml"}).status_code == 422

    assert client.delete(f"/api/v1/datasets/{dataset_id}").json() == {"dataset_id": dataset_id, "deleted": True}
    for response in (client.get(f"/api/v1/datasets/{dataset_id}"), client.delete(f"/api/v1/datasets/{dataset_id}")):
        assert response.status_code == 404 and response.json()["error_type"] == "dataset_not_found"
    assert client.get("/api/v1/datasets").json()["total"] == 1


@pytest.mark.parametrize("dataset_id", ["ds-zzzzzzzzzzzz", "ds-0123456789ab", "DS-0123456789AB", "ds-0123", "..", "..%2F..%2Fconfig"])
def test_dataset_endpoints_reject_unknown_and_malformed_ids(client: TestClient, dataset_id: str) -> None:
    response = client.get(f"/api/v1/datasets/{dataset_id}")
    assert response.status_code == 404
    assert response.json()["error_type"] in {"dataset_not_found", "not_found"}


def test_dataset_store_is_path_traversal_safe(tmp_path: Path) -> None:
    store = DatasetStore(tmp_path / "store")
    (tmp_path / "secret").mkdir()
    (tmp_path / "secret" / "metadata.json").write_text("{}", encoding="utf-8")
    for bad in ("../secret", "..\\secret", "ds-../../secret", "ds-0123456789abc", "", "ds-xyz", None):
        with pytest.raises(DatasetNotFoundError):
            store.get(bad)  # type: ignore[arg-type]
        with pytest.raises(DatasetNotFoundError):
            store.delete(bad)  # type: ignore[arg-type]
    assert (tmp_path / "secret" / "metadata.json").is_file()


def test_dataset_store_round_trip(tmp_path: Path) -> None:
    store = DatasetStore(tmp_path / "store")
    assert store.list_datasets() == [] and store.count() == 0
    frame = make_patient_stays(n=5, seed=1)
    dataset_id = store.new_id()
    store.save(frame, {"dataset_id": dataset_id, "table": "patient_stays", "created_at": "2026-01-01T00:00:00+00:00",
                       "n_rows": 5, "generation": {"seed": np.int64(3)}})
    assert store.exists(dataset_id) and store.count() == 1
    pd.testing.assert_frame_equal(store.load(dataset_id), frame, check_dtype=False)
    assert store.get(dataset_id)["generation"] == {"seed": 3}
    assert [item["dataset_id"] for item in store.list_datasets()] == [dataset_id]
    assert not any(path.name.startswith(".") for path in (tmp_path / "store").iterdir())
    with pytest.raises(FileExistsError):
        store.save(frame, {"dataset_id": dataset_id, "table": "patient_stays"})
    store.delete(dataset_id)
    assert store.count() == 0 and not any((tmp_path / "store").iterdir())


def test_default_store_lives_under_the_synthetic_dir(isolated_paths: dict[str, Path]) -> None:
    store = DatasetStore()
    assert store.base_dir == get_path("synthetic_dir", create=False) / "api"
    assert not store.base_dir.exists()  # resolving the store never creates directories


# ---------------------------------------------------------------------------
# JSON helpers and record preparation
# ---------------------------------------------------------------------------
def test_frame_to_records_converts_numpy_and_missing_values() -> None:
    frame = pd.DataFrame({
        "i": np.array([1, 2], dtype="int64"), "f": [1.5, np.nan], "b": np.array([True, False]),
        "s": ["x", None], "n": pd.array([1, None], dtype="Int64"), "inf": [np.inf, 0.25],
    })
    records = frame_to_records(frame)
    assert records == [
        {"i": 1, "f": 1.5, "b": True, "s": "x", "n": 1, "inf": None},
        {"i": 2, "f": None, "b": False, "s": None, "n": None, "inf": 0.25},
    ]
    assert type(records[0]["i"]) is int and type(records[0]["b"]) is bool and type(records[1]["inf"]) is float
    json.dumps(records, allow_nan=False)
    assert frame_to_records(pd.DataFrame()) == []


def test_to_jsonable_handles_nested_numpy_and_pandas_values() -> None:
    value = {
        "a": np.int64(3), "b": [np.float32(0.5), float("nan")], "c": {np.str_("k"): np.bool_(True)},
        "d": pd.Timestamp("2025-01-01"), "e": (1, 2), "f": np.array([1, 2]), "g": pd.NA, "h": {3, 1},
    }
    assert to_jsonable(value) == {"a": 3, "b": [0.5, None], "c": {"k": True}, "d": "2025-01-01T00:00:00",
                                  "e": [1, 2], "f": [1, 2], "g": None, "h": [1, 3]}


def test_prepare_records_fills_missing_derived_values_only() -> None:
    partial = dict(VALID_WARD_DAY, occupancy_rate=None, census_end=30)
    prepared = prepare_records("ward_daily", [partial])
    row = prepared.frame.iloc[0]
    assert row["occupancy_rate"] == pytest.approx(30 / 32) and row["census_end"] == 30
    assert "occupancy_rate" not in prepared.normalisation["derived_computed"]
    assert prepared.normalisation["derived_mismatches"] == {} and prepared.warnings == []
    assert list(prepared.frame.columns[:3]) == ["ward_id", "ward_type", "month"]


# ---------------------------------------------------------------------------
# UI, OpenAPI, CORS and error envelope
# ---------------------------------------------------------------------------
def test_ui_is_served_and_self_contained(client: TestClient) -> None:
    response = client.get("/ui")
    assert response.status_code == 200 and response.headers["content-type"].startswith("text/html")
    html = response.text
    assert "<title>BedSynth AI" in html and 'const API = "/api/v1"' in html
    assert all(f"${{API}}/{endpoint}" in html for endpoint in ("generate", "scenarios", "models", "datasets", "evaluate"))
    assert re.search(r"""(src|href)\s*=\s*["']https?://""", html) is None, "the UI must not load external assets"
    assert "@import" not in html and "<script src" not in html


def test_openapi_documents_examples_tags_limits_and_errors(client: TestClient) -> None:
    spec = client.get("/openapi.json").json()
    operation = spec["paths"]["/api/v1/generate"]["post"]
    examples = operation["requestBody"]["content"]["application/json"]["examples"]
    families = {SCENARIOS[item["value"]["scenario"]].category for item in examples.values() if item["value"].get("scenario")}
    assert len(examples) >= 8 and len(families) >= 6
    assert any(item["value"].get("mode") == "calendar" for item in examples.values())
    assert {"400", "404", "422", "503"} <= set(operation["responses"])
    assert "text/csv" in operation["responses"]["200"]["content"]
    assert {tag["name"] for tag in spec["tags"]} == {"service", "catalogue", "generation", "quality", "datasets"}
    properties = spec["components"]["schemas"]["GenerateRequest"]["properties"]
    assert properties["num_rows"]["maximum"] == MAX_ROWS and "validate" in properties
    assert properties["num_rows"]["description"]
    download = spec["paths"]["/api/v1/datasets/{dataset_id}/download"]["get"]["responses"]["200"]["content"]
    assert {"text/csv", "application/json"} <= set(download)


def test_cors_allows_local_demo_pages(client: TestClient) -> None:
    preflight = client.options(GENERATE, headers={
        "Origin": "http://localhost:5173", "Access-Control-Request-Method": "POST", "Access-Control-Request-Headers": "content-type",
    })
    assert preflight.status_code == 200
    assert preflight.headers["access-control-allow-origin"] in {"*", "http://localhost:5173"}
    response = client.post(GENERATE, json={"table": "ward_daily", "num_rows": 3, "format": "csv"}, headers={"Origin": "http://localhost:5173"})
    exposed = response.headers["access-control-expose-headers"]
    assert "X-BedSynth-Dataset-Id" in exposed and "X-BedSynth-Scenario" in exposed


def test_uniform_error_envelope_and_timing_header(client: TestClient) -> None:
    assert client.get("/no/such/route").json() == {"detail": "Not Found", "error_type": "not_found"}
    not_allowed = client.put("/api/v1/tables")
    assert not_allowed.status_code == 405 and not_allowed.json()["error_type"] == "method_not_allowed"
    assert float(client.get("/health").headers["x-process-time-ms"]) >= 0.0


# ---------------------------------------------------------------------------
# Demo materials
# ---------------------------------------------------------------------------
def test_api_demo_script_writes_evidence(client: TestClient, reference_data: Path, tmp_path: Path) -> None:
    demo = _load_script(PROJECT_ROOT / "scripts" / "05_api_demo.py", "bedsynth_api_demo_script")
    out_dir = tmp_path / "evidence"
    summary = demo.run_demo(client, out_dir, rows=40, seed=42)
    failures = [(step["name"], step["checks"], step["status_code"]) for step in summary["steps"] if step["outcome"] == "FAIL"]
    assert summary["failed"] == 0, failures
    assert summary["skipped"] == 0 and summary["passed"] >= 30
    evidence = sorted(out_dir.glob("*.json"))
    assert len(evidence) == summary["total"]
    first = json.loads(evidence[0].read_text(encoding="utf-8"))
    assert first["request"]["path"] == "/" and first["passed"] is True
    assert list(out_dir.glob("*.csv")), "CSV responses are saved next to their evidence"
    markdown = (out_dir / "summary.md").read_text(encoding="utf-8")
    assert "API demo evidence" in markdown and f"{summary['passed']} passed, 0 failed" in markdown


def test_api_demo_main_writes_summary_and_week4_report(
    isolated_paths: dict[str, Path], reference_data: Path, tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    demo = _load_script(PROJECT_ROOT / "scripts" / "05_api_demo.py", "bedsynth_api_demo_script_full")
    monkeypatch.setattr(demo, "load_generator", lambda: (FakeGenerator(), None))
    out_dir = tmp_path / "evidence"
    assert demo.main(["--rows", "40", "--out-dir", str(out_dir)]) == 0
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["failed"] == 0 and summary["passed"] >= 30
    report = (isolated_paths["reports_dir"] / "week4_api_system.md").read_text(encoding="utf-8")
    assert "Week 4: API and system evaluation" in report and f"**{summary['passed']} passed, 0 failed" in report
    assert "Week-4 report written" in capsys.readouterr().out
    # A6: one persisted dataset per scenario is kept (with its metadata) and indexed in the report
    coverage = summary["scenario_coverage"]
    assert len(coverage) == len(SCENARIOS) and all(item["outcome"] == "ok" for item in coverage)
    store = isolated_paths["synthetic_dir"] / "api_demo"
    assert summary["datasets_dir"] == str(store)
    for item in coverage:
        assert (store / item["dataset_id"] / "data.csv").is_file() and (store / item["dataset_id"] / "metadata.json").is_file()
    assert "## 3. Scenario coverage through the API" in report and "| winter_admissions | patient_stays |" in report
    # the exact wording belongs to bedsynth/reporting/week4.py ("No fast-suite results stored yet: ...")
    assert "## 5. Testing evidence" in report and "results stored yet" in report
    # a second run starts from an empty demo store instead of accumulating datasets
    assert demo.main(["--rows", "40", "--scenario-rows", "0", "--out-dir", str(out_dir), "--no-report"]) == 0
    remaining = {child.name for child in store.iterdir() if child.name.startswith("ds-")}
    assert remaining and not remaining & {item["dataset_id"] for item in coverage}


def test_week4_report_summarises_junit_testing_evidence(isolated_paths: dict[str, Path]) -> None:
    """A6: the Week-4 report carries the stored test results (pytest JUnit XML)."""
    from bedsynth.reporting.week4 import JUNIT_FILE, load_junit_summary, render_week4_report

    junit = isolated_paths["results_dir"] / JUNIT_FILE
    junit.write_text(
        '<?xml version="1.0" encoding="utf-8"?><testsuites><testsuite name="pytest" tests="4" time="12.5" '
        'timestamp="2026-11-15T00:00:00"><testcase classname="tests.test_api" name="test_a" time="1"/>'
        '<testcase classname="tests.test_api" name="test_b" time="1"><failure message="boom"/></testcase>'
        '<testcase classname="tests.test_schema" name="test_c" time="1"/>'
        '<testcase classname="tests.test_schema" name="test_d" time="1"><skipped message="slow"/></testcase>'
        "</testsuite></testsuites>", encoding="utf-8")
    summary = load_junit_summary(junit)
    assert summary["totals"] == {"tests": 4, "passed": 2, "failed": 1, "errors": 0, "skipped": 1}
    assert summary["by_file"]["test_api.py"] == {"passed": 1, "failed": 1, "errors": 0, "skipped": 0}
    assert summary["failures"] == ["test_api.py::test_b"]
    text = render_week4_report().read_text(encoding="utf-8")
    assert "**2 passed, 1 failed, 0 errors, 1 skipped** of 4 tests" in text and "`test_api.py::test_b`" in text


def test_api_demo_script_explains_missing_models(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    demo = _load_script(PROJECT_ROOT / "scripts" / "05_api_demo.py", "bedsynth_api_demo_script_main")
    monkeypatch.setattr(demo, "load_generator", lambda: (None, demo.NO_MODELS_MESSAGE))
    assert demo.main([]) == 2
    assert "02_train_synthesizers" in capsys.readouterr().err


def test_demo_client_walkthrough(client: TestClient, reference_data: Path, tmp_path: Path) -> None:
    walkthrough = _load_script(PROJECT_ROOT / "demo" / "demo_client.py", "bedsynth_demo_client")
    lines: list[str] = []
    result = walkthrough.run_walkthrough(client, rows=40, out_dir=tmp_path / "out", echo=lambda *parts: lines.append(" ".join(map(str, parts))))
    assert result["ok"] and len(result["datasets"]) == 2
    assert list((tmp_path / "out").glob("ds-*.csv"))
    assert any("invalid rows: [5]" in line for line in lines)
    assert client.get("/api/v1/datasets").json()["total"] == 0  # cleaned up


def test_serve_script_parses_arguments() -> None:
    serve = _load_script(PROJECT_ROOT / "scripts" / "04_serve_api.py", "bedsynth_serve_script")
    args = serve.parse_args(["--port", "8123", "--reload", "--no-preload"])
    assert (args.host, args.port, args.reload, args.no_preload) == ("127.0.0.1", 8123, True, True)
    assert serve.display_url("0.0.0.0", 8123) == "http://127.0.0.1:8123"
    assert serve.log_config("debug")["loggers"]["bedsynth"]["level"] == "DEBUG"


# ---------------------------------------------------------------------------
# Review regressions: robustness, limits, verdicts and security (findings A1-A11)
# ---------------------------------------------------------------------------
def _settings_with(**api_overrides: Any) -> dict[str, Any]:
    from bedsynth.config import get_settings

    settings = get_settings()
    settings["api"] = {**settings["api"], **api_overrides}
    return settings


def test_conditions_listing_every_value_are_dropped(client: TestClient, fake_generator: FakeGenerator) -> None:
    """A1: a column whose values cover its vocabulary means 'no condition' and is not expanded."""
    every = {column: list(schema.column_spec("patient_stays", column).categories)
             for column in schema.conditionable_columns("patient_stays")}
    body = _generate(client, table="patient_stays", num_rows=10, conditions=every, seed=3, persist=False)
    assert fake_generator.requests[-1].conditions is None
    assert sum("lists every value" in warning for warning in body["warnings"]) == len(every)
    assert body["request"]["conditions"] is None and body["n_rows"] == 10


def test_too_many_condition_combinations_are_rejected_before_sampling(
    client: TestClient, fake_generator: FakeGenerator
) -> None:
    """A1: the product of value counts (after merging scenario and wards) is capped."""
    conditions = {"month": list(range(1, 12)), "day_of_week": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat"],
                  "ward_id": list(schema.WARD_IDS[:7]), "staffing_level": ["short", "adequate"],
                  "occupancy_level": ["low", "normal", "high"]}
    response = client.post(GENERATE, json={"table": "ward_daily", "num_rows": 10, "conditions": conditions})
    assert response.status_code == 400 and response.json()["error_type"] == "too_many_combinations"
    assert "2772" in response.json()["detail"] and fake_generator.requests == []
    # the scenario intersects occupancy_level to ['high']: 924 combinations are within the cap
    ok = client.post(GENERATE, json={"table": "ward_daily", "num_rows": 10, "conditions": conditions,
                                     "scenario": "high_occupancy", "persist": False})
    assert ok.status_code == 200, ok.text


def test_engine_reports_are_bounded(isolated_paths: dict[str, Path]) -> None:
    """A1: thousands of engine warnings / allocation entries never reach the client or the store."""
    class ChattyGenerator(FakeGenerator):
        def generate(self, request: GenerationRequest) -> GenerationResult:
            result = super().generate(request)
            result.report["warnings"] = [f"combination {i} is infeasible" for i in range(5000)]
            result.report["condition_allocation"] = [{"conditions": {"i": i}, "requested": 0, "generated": 0}
                                                     for i in range(3000)]
            return result

    with TestClient(create_app(generator=ChattyGenerator())) as test_client:
        body = _generate(test_client, table="ward_daily", num_rows=5, seed=1)
    assert len(body["warnings"]) <= 26 and body["warnings"][-1].endswith("more warnings omitted.")
    assert len(body["generation"]["condition_allocation"]) == 200
    assert body["generation"]["condition_allocation_total"] == 3000


def test_generation_time_budget_and_max_attempts_reach_the_engine(
    client: TestClient, fake_generator: FakeGenerator
) -> None:
    """A2/A7: the API passes a bounded time budget (settings default) and max_attempts to the engine."""
    body = _generate(client, table="ward_daily", num_rows=5, seed=1, persist=False)
    sent = fake_generator.requests[-1]
    assert sent.time_budget_seconds == 60.0 and sent.max_attempts is None
    assert body["request"]["time_budget_seconds"] == 60.0
    _generate(client, table="ward_daily", num_rows=5, seed=1, persist=False, time_budget_seconds=7.5, max_attempts=40)
    sent = fake_generator.requests[-1]
    assert (sent.time_budget_seconds, sent.max_attempts) == (7.5, 40)
    for bad in ({"time_budget_seconds": 1000}, {"time_budget_seconds": 0}, {"max_attempts": 0},
                {"max_attempts": 101}, {"max_attempts": True}):
        response = client.post(GENERATE, json={"table": "ward_daily", **bad})
        assert response.status_code == 422, bad


class _BlockingGenerator(FakeGenerator):
    """Holds every generation until ``release`` is set (simulates a slow model)."""

    def __init__(self) -> None:
        super().__init__()
        import threading

        self.started = threading.Event()
        self.release = threading.Event()

    def generate(self, request: GenerationRequest) -> GenerationResult:
        self.started.set()
        assert self.release.wait(10), "test did not release the generator"
        return super().generate(request)


def test_busy_engine_answers_503_and_health_stays_responsive(isolated_paths: dict[str, Path]) -> None:
    """A2: bounded admission - excess generations get 503 server_busy + Retry-After; /health is unaffected."""
    import threading
    import time

    generator = _BlockingGenerator()
    settings = _settings_with(max_pending_generations=1, generation_wait_seconds=0.2)
    results: dict[str, Any] = {}
    with TestClient(create_app(generator=generator, settings=settings)) as test_client:
        def first() -> None:
            results["first"] = test_client.post(GENERATE, json={"table": "ward_daily", "num_rows": 3, "seed": 1})

        worker = threading.Thread(target=first)
        worker.start()
        assert generator.started.wait(5)
        busy = test_client.post(GENERATE, json={"table": "ward_daily", "num_rows": 3, "seed": 2})
        started = time.perf_counter()
        health = test_client.get("/health")
        health_seconds = time.perf_counter() - started
        generator.release.set()
        worker.join(10)
    assert busy.status_code == 503 and busy.json()["error_type"] == "server_busy"
    assert int(busy.headers["retry-after"]) >= 1
    assert health.status_code == 200 and health_seconds < 5
    assert results["first"].status_code == 200


def test_queued_generation_gives_up_after_the_wait_limit(isolated_paths: dict[str, Path]) -> None:
    """A2: a request queued behind a running one waits at most generation_wait_seconds."""
    import threading

    generator = _BlockingGenerator()
    settings = _settings_with(max_pending_generations=2, generation_wait_seconds=0.3)
    with TestClient(create_app(generator=generator, settings=settings)) as test_client:
        worker = threading.Thread(target=lambda: test_client.post(GENERATE, json={"table": "ward_daily", "num_rows": 3}))
        worker.start()
        assert generator.started.wait(5)
        queued = test_client.post(GENERATE, json={"table": "ward_daily", "num_rows": 3})
        generator.release.set()
        worker.join(10)
    assert queued.status_code == 503 and "stayed busy" in queued.json()["detail"]


def _validate(client: TestClient, table: str, records: list[dict[str, Any]], **extra: Any) -> dict[str, Any]:
    response = client.post("/api/v1/validate", json={"table": table, "records": records, **extra})
    assert response.status_code == 200, response.text
    return response.json()


def _failed_hard_rules(body: dict[str, Any]) -> set[str]:
    return {rule["rule_id"] for rule in body["validation"]["rules"] if rule["severity"] == "hard" and not rule["passed"]}


@pytest.mark.parametrize("changes", [
    {"occupancy_rate": 5.0}, {"occupancy_rate": -3}, {"census_end": 31}, {"ward_type": "surgical"},
    {"is_weekend": True}, {"season": "summer"}, {"available_beds": 40},
])
def test_inconsistent_derived_values_make_a_ward_day_hard_invalid(client: TestClient, changes: dict[str, Any]) -> None:
    """A3: provided derived values that contradict their definition fail the hard WD-D01 check."""
    body = _validate(client, "ward_daily", [VALID_WARD_DAY, dict(VALID_WARD_DAY, **changes)], return_records=True)
    assert body["n_invalid_rows"] == 1 and body["invalid_rows"] == [1]
    assert body["validation"]["all_hard_rules_passed"] is False
    assert _failed_hard_rules(body) == {"WD-D01"}            # the rules themselves ran on recomputed values
    column = next(iter(changes))
    assert body["normalisation"]["derived_mismatch_rows"] == {column: [1]}
    assert body["records"][1][column] == changes[column]     # the client's value is echoed back unchanged


def test_validation_runs_on_recomputed_stay_values(client: TestClient) -> None:
    """A3: a wrong provided los_hours no longer causes a false PS-H08 positive; it fails PS-D01 instead."""
    stay = VALIDATE_EXAMPLES["stay"]["value"]["records"][0]
    body = _validate(client, "patient_stays", [dict(stay, delay_reason="awaiting_transport",
                                                      discharge_delay_hours=20.0, los_hours=1)])
    assert _failed_hard_rules(body) == {"PS-D01"} and body["normalisation"]["derived_mismatches"] == {"los_hours": 1}
    lying_flag = _validate(client, "patient_stays", [dict(stay, delay_reason="awaiting_transport",
                                                            discharge_delay_hours=20.0, is_delayed_discharge=False)])
    assert _failed_hard_rules(lying_flag) == {"PS-D01"}
    short_delay = _validate(client, "patient_stays", [dict(stay, delay_reason="awaiting_transport", discharge_delay_hours=0.7)])
    assert _failed_hard_rules(short_delay) == {"PS-H13"}      # D1: delays are 0 or >= 4 h


@pytest.mark.parametrize("changes, rule", [({"month": True}, "WD-H01"), ({"census_start": True}, "WD-H02")])
def test_booleans_are_not_category_codes_or_counts(client: TestClient, changes: dict[str, Any], rule: str) -> None:
    """A3: JSON true is neither month 1 nor one patient."""
    body = _validate(client, "ward_daily", [dict(VALID_WARD_DAY, **changes)])
    assert body["n_invalid_rows"] == 1 and rule in _failed_hard_rules(body)


@pytest.mark.parametrize("filters", [
    {"ward_id": {"min": 1}}, {"season": {"min": 0}}, {"occupancy_level": {"max": 2}}, {"is_weekend": {"min": 0}},
    {"census_start": {"in": ["abc"]}}, {"is_weekend": {"in": ["perhaps"]}}, {"occupancy_rate": {"min": 1.5}},
    {"census_start": {"max": -1}}, {"occupancy_level": {"not_in": ["low", "normal", "high", "critical"]}},
    {"is_weekend": {"in": [True], "not_in": [True]}},
])
def test_impossible_or_mistyped_filters_return_400(client: TestClient, fake_generator: FakeGenerator,
                                                   filters: dict[str, Any]) -> None:
    """A4: filters that cannot match any row are rejected before sampling."""
    response = client.post(GENERATE, json={"table": "ward_daily", "num_rows": 5, "filters": filters})
    assert response.status_code == 400, response.text
    assert response.json()["error_type"] == "invalid_filters" and fake_generator.requests == []


def test_filter_values_are_coerced_to_the_column_type(client: TestClient, fake_generator: FakeGenerator) -> None:
    """A4: 'true' becomes a boolean and numeric strings become numbers before the engine sees them."""
    _generate(client, table="ward_daily", num_rows=5, seed=1, persist=False,
              filters={"is_weekend": {"in": ["true"]}, "census_start": {"in": ["20", 21.0]}, "month": {"min": 1, "max": 6}})
    assert fake_generator.requests[-1].filters == {
        "is_weekend": {"in": [True]}, "census_start": {"in": [20, 21]}, "month": {"min": 1, "max": 6},
    }


@pytest.mark.parametrize("raw_bound", ["NaN", "Infinity", "true"])
def test_non_finite_or_boolean_filter_bounds_return_422(client: TestClient, raw_bound: str) -> None:
    """A4: NaN/inf/boolean bounds are schema errors (Python's json accepts NaN literals)."""
    body = '{"table": "ward_daily", "filters": {"census_start": {"min": %s}}}' % raw_bound
    response = client.post(GENERATE, content=body, headers={"Content-Type": "application/json"})
    assert response.status_code == 422 and response.json()["error_type"] == "validation_error"


def test_requests_no_row_can_satisfy_return_422_and_store_nothing(client: TestClient) -> None:
    """A4: an empty generation is an error (no persisted empty dataset 'passing' every rule)."""
    response = client.post(GENERATE, json={"table": "ward_daily", "num_rows": 20, "filters": {"census_start": {"min": 1000}}})
    assert response.status_code == 422 and response.json()["error_type"] == "no_rows_generated"
    assert "no dataset was stored" in response.json()["detail"]
    assert client.get("/api/v1/datasets").json()["total"] == 0


def test_zero_row_validation_summary_has_no_verdict() -> None:
    """A4: all_hard_rules_passed is null (not true) when there is nothing to judge."""
    from bedsynth.api.service import summarise_validation
    from bedsynth.validation import validate_table

    empty = make_ward_daily(n_days=1, seed=0).iloc[0:0]
    summary = summarise_validation(validate_table(empty, "ward_daily"))
    assert summary["n_rows"] == 0 and summary["all_hard_rules_passed"] is None and summary["hard_valid_rate"] is None


def _saved_store(tmp_path: Path, n: int = 1) -> tuple[DatasetStore, list[str]]:
    store = DatasetStore(tmp_path / "store")
    ids = []
    for _ in range(n):
        dataset_id = store.new_id()
        store.save(make_patient_stays(n=5, seed=1), {"dataset_id": dataset_id, "table": "patient_stays", "n_rows": 5})
        ids.append(dataset_id)
    return store, ids


def test_concurrent_deletes_succeed_exactly_once(tmp_path: Path) -> None:
    """A5: racing deletes of one dataset give one success and clean 404s, never other errors."""
    from concurrent.futures import ThreadPoolExecutor

    store, ids = _saved_store(tmp_path, n=5)

    def delete(dataset_id: str) -> str:
        try:
            store.delete(dataset_id)
            return "deleted"
        except DatasetNotFoundError:
            return "not_found"

    with ThreadPoolExecutor(max_workers=8) as pool:
        outcomes = list(pool.map(delete, [dataset_id for dataset_id in ids for _ in range(8)]))
    assert outcomes.count("deleted") == len(ids) and outcomes.count("not_found") == 7 * len(ids)
    assert store.count() == 0


def test_delete_of_a_locked_dataset_is_a_conflict(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A5: a dataset whose files stay locked (Windows: open by another process) answers 409, not 500."""
    from bedsynth.api import service as service_module

    store, (dataset_id,) = _saved_store(tmp_path)
    monkeypatch.setattr(DatasetStore, "DELETE_RETRY_SECONDS", 0.0)

    def locked(source: Any, target: Any) -> None:
        raise PermissionError(13, "The process cannot access the file because it is being used by another process")

    monkeypatch.setattr(service_module.os, "replace", locked)
    with pytest.raises(service_module.ConflictError) as raised:
        store.delete(dataset_id)
    assert raised.value.status_code == 409 and raised.value.error_type == "dataset_busy"
    monkeypatch.undo()
    assert store.exists(dataset_id)


def test_downloads_release_the_file_so_delete_works(client: TestClient) -> None:
    """A5: CSV downloads are read under the store lock (no open handle survives the response)."""
    dataset_id = _generate(client, table="ward_daily", num_rows=6, seed=2)["dataset_id"]
    for fmt in ("csv", "json", "csv"):
        assert client.get(f"/api/v1/datasets/{dataset_id}/download", params={"format": fmt}).status_code == 200
    assert client.delete(f"/api/v1/datasets/{dataset_id}").status_code == 200


def test_corrupt_dataset_metadata_is_a_handled_error(client: TestClient, isolated_paths: dict[str, Path]) -> None:
    """A5: unreadable metadata gives a JSON 'dataset_corrupt' error (not an unhandled 500) and can be deleted."""
    dataset_id = _generate(client, table="ward_daily", num_rows=6, seed=2)["dataset_id"]
    (isolated_paths["synthetic_dir"] / "api" / dataset_id / "metadata.json").write_text("{not json", encoding="utf-8")
    response = client.get(f"/api/v1/datasets/{dataset_id}")
    assert response.status_code == 500 and response.json()["error_type"] == "dataset_corrupt"
    assert client.get("/api/v1/datasets").json()["total"] == 0
    assert client.delete(f"/api/v1/datasets/{dataset_id}").status_code == 200


def test_model_info_reports_multi_column_segments_and_constraint_groups() -> None:
    """Cards of ward x source models list their segment columns (not null) and constraint groups."""
    from bedsynth.api.schemas import ModelInfo
    from bedsynth.api.service import model_info

    card = {"experiment_id": "gc_tuned", "table": "patient_stays", "model": "gaussian_copula",
            "segment_by": ["ward_id", "admission_source"], "segment_label": "ward_id x admission_source",
            "constraint_groups": ["fixed_combinations", "structural_zeros", "conditional_marginals"],
            "effective_constraint_groups": ["fixed_combinations", "conditional_marginals"],
            "experiment": {"segment_by": ["ward_id", "admission_source"]}}
    info = ModelInfo(**model_info(card, "patient_stays", "gc_tuned")).model_dump()
    assert info["segment_by"] == ["ward_id", "admission_source"] and info["is_default"]
    assert info["segment_label"] == "ward_id x admission_source"
    assert info["effective_constraint_groups"] == ["fixed_combinations", "conditional_marginals"]

    # an unresolved per-table setting in a nested experiment config is resolved for the table
    nested = {"experiment_id": "gc_tuned", "experiment": {"segment_by": {"ward_daily": "ward_id",
                                                                         "patient_stays": ["ward_id", "admission_source"]}}}
    assert model_info(nested, "ward_daily", None)["segment_by"] == "ward_id"
    assert model_info(nested, "patient_stays", None)["segment_label"] == "ward_id x admission_source"
    # a global model: top-level null wins over anything nested
    glob = {"experiment_id": "ctgan", "segment_by": None, "segment_label": "global",
            "experiment": {"segment_by": "ward_id"}}
    assert model_info(glob, "ward_daily", None)["segment_by"] is None
    assert model_info(glob, "ward_daily", None)["segment_label"] == "global"


def test_model_cards_and_errors_never_leak_server_paths(isolated_paths: dict[str, Path]) -> None:
    """A8: nested card paths are reduced to file names and missing-model errors use a fixed message."""
    secret = r"C:\Users\someone\project\data\processed\ward_daily_train.csv"

    class LeakyGenerator(FakeGenerator):
        def available_models(self, table: str | None = None) -> list[dict[str, Any]]:
            cards = super().available_models(table)
            for card in cards:
                card["training"] = {"source": secret, "rows": 10}
                card["path"] = "/home/someone/models/x/synthesizer.pkl"
            return cards

        def generate(self, request: GenerationRequest) -> GenerationResult:
            raise ModelNotFoundError(r"no synthesizer.pkl under C:\Users\someone\models\ward_daily\gc_tuned")

    with TestClient(create_app(generator=LeakyGenerator())) as test_client:
        models = test_client.get("/api/v1/models").text
        missing = test_client.post(GENERATE, json={"table": "ward_daily", "num_rows": 5, "model_id": "gc_tuned"})
    assert "someone" not in models and "ward_daily_train.csv" in models
    assert missing.status_code == 503 and missing.json()["error_type"] == "model_unavailable"
    assert "someone" not in missing.text and "gc_tuned" in missing.json()["detail"]


def test_scrub_paths_removes_absolute_paths() -> None:
    from bedsynth.api.service import redact_paths, scrub_paths

    text = r"failed for C:\Users\x\models\a.pkl and /home/x/data/b.csv; see GET /api/v1/models"
    assert scrub_paths(text) == "failed for <server path> and <server path>; see GET /api/v1/models"
    assert redact_paths({"a": {"source": "/srv/x/train.csv", "path": "/srv/x"}, "b": ["D:/m/c.pkl"]}) == {
        "a": {"source": "train.csv"}, "b": ["c.pkl"]}
    # home-relative paths (bedsynth.config.project_relative outside the project) are server paths too;
    # project-relative ones are kept
    home = r"no file at ~/AppData/Local/Temp/run/models/a.pkl, ~/data/b.pkl or ~\models\c.pkl (approx. 3~/s)"
    assert scrub_paths(home) == "no file at <server path>, <server path> or <server path> (approx. 3~/s)"
    assert redact_paths({"source": "~/AppData/Local/Temp/run/raw/ward_daily_raw.csv",
                         "train": "data/processed/ward_daily_train.csv"}) == {
        "source": "ward_daily_raw.csv", "train": "data/processed/ward_daily_train.csv"}
    # user names with spaces and MSYS / Git Bash drive paths are scrubbed completely
    spaced = (r"cannot open 'C:\Users\John Smith\proj\x.csv', C:\Users\Mary Ann Lee\y.csv and "
              "/c/Users/John Smith/proj/z.pkl now")
    assert scrub_paths(spaced) == "cannot open '<server path>', <server path> and <server path> now"
    assert scrub_paths(r"C:\a\b.csv and C:\c\d.csv") == "<server path> and <server path>"
    assert scrub_paths("w/o rows/day; see GET /api/v1/models?table=x") == "w/o rows/day; see GET /api/v1/models?table=x"


@pytest.mark.parametrize("dates", [("1850-01-01", "1850-01-02"), ("9999-12-01", "9999-12-31")])
def test_calendar_dates_outside_the_supported_range_return_422(client: TestClient, dates: tuple[str, str]) -> None:
    """A8: no pandas-internal overflow messages for far-away calendar dates."""
    response = client.post(GENERATE, json={"table": "ward_daily", "mode": "calendar",
                                           "start_date": dates[0], "end_date": dates[1]})
    assert response.status_code == 422 and "between 1900-01-01 and 2200-12-31" in response.json()["detail"]


def test_cors_rejects_foreign_websites_but_allows_local_pages(client: TestClient) -> None:
    """A9: CORS is limited to pages served from this machine by default."""
    evil = client.options("/api/v1/datasets/ds-0123456789ab", headers={
        "Origin": "https://evil.example", "Access-Control-Request-Method": "DELETE"})
    assert "access-control-allow-origin" not in evil.headers
    listing = client.get("/api/v1/datasets", headers={"Origin": "https://evil.example"})
    assert "access-control-allow-origin" not in listing.headers
    local = client.options("/api/v1/datasets/ds-0123456789ab", headers={
        "Origin": "http://127.0.0.1:8000", "Access-Control-Request-Method": "DELETE"})
    assert local.headers["access-control-allow-origin"] == "http://127.0.0.1:8000"


def test_unhandled_errors_are_json_with_cors_headers(isolated_paths: dict[str, Path]) -> None:
    """A9: an unexpected exception still returns the uniform body with CORS headers."""
    class BrokenCatalogue(FakeGenerator):
        def available_models(self, table: str | None = None) -> list[dict[str, Any]]:
            raise TypeError("registry exploded")

    with TestClient(create_app(generator=BrokenCatalogue())) as test_client:
        response = test_client.get("/api/v1/models", headers={"Origin": "http://localhost:5173"})
    assert response.status_code == 500
    assert response.json() == {"detail": "Internal server error; see the server log.", "error_type": "internal_error"}
    assert response.headers["access-control-allow-origin"] == "http://localhost:5173"


def test_quality_needs_a_minimum_number_of_rows(client: TestClient, reference_data: Path) -> None:
    """A10: no KS/TVD score from a handful of rows, for generations and stored datasets alike."""
    small = _generate(client, table="ward_daily", num_rows=5, evaluate=True, seed=3)
    assert small["quality"] is None and any("at least 20" in warning for warning in small["warnings"])
    stored = client.post("/api/v1/evaluate", json={"dataset_id": small["dataset_id"]})
    assert stored.status_code == 400 and stored.json()["error_type"] == "too_few_rows"


@pytest.mark.parametrize("payload", [{"num_rows": True}, {"num_rows": "7"}, {"num_rows": 7.0}, {"seed": True}, {"seed": "3"}])
def test_num_rows_and_seed_are_strict_integers(client: TestClient, payload: dict[str, Any]) -> None:
    """A11: booleans and numeric strings are not silently coerced."""
    response = client.post(GENERATE, json={"table": "ward_daily", **payload})
    assert response.status_code == 422, response.text


def test_csv_generation_carries_the_quality_score_header(client: TestClient, reference_data: Path) -> None:
    """A11: with format=csv the computed quality is visible (X-BedSynth-Quality-Score), not discarded."""
    response = client.post(GENERATE, json={"table": "ward_daily", "num_rows": 40, "evaluate": True, "format": "csv",
                                           "persist": False, "seed": 4})
    assert response.status_code == 200
    assert 0.0 <= float(response.headers["x-bedsynth-quality-score"]) <= 1.0


# ---------------------------------------------------------------------------
# Live-test review regressions (API documentation round)
# ---------------------------------------------------------------------------
SECRET_PATH = r"C:\Users\someone\secret\file.csv"


def test_internal_errors_never_echo_exception_text_or_paths(
    client: TestClient, reference_data: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Validation, derivation and quality failures answer with fixed messages (traceback in the server log)."""
    from bedsynth import schema as schema_module
    from bedsynth import validation as validation_module
    from bedsynth.api import service as service_module

    def raise_with_path(exc_type: type[Exception]) -> Any:
        def broken(*args: Any, **kwargs: Any) -> Any:
            raise exc_type(f"cannot read {SECRET_PATH}")
        return broken

    responses = []
    monkeypatch.setattr(validation_module, "validate_table", raise_with_path(ValueError))
    validate = client.post("/api/v1/validate", json={"table": "ward_daily", "records": [VALID_WARD_DAY]})
    generated = client.post(GENERATE, json={"table": "ward_daily", "num_rows": 5, "seed": 1, "persist": False})
    assert validate.status_code == 500
    assert validate.json() == {"detail": "Validation failed (ValueError); see the server log.", "error_type": "validation_failed"}
    assert generated.status_code == 200
    assert "Validation skipped: Validation failed (ValueError); see the server log." in generated.json()["warnings"]
    responses += [validate, generated]

    records = {"table": "ward_daily", "records": [VALID_WARD_DAY] * 20, "include_validation": False}
    monkeypatch.setattr(service_module, "run_quick_quality", raise_with_path(KeyError))
    bad_input = client.post("/api/v1/evaluate", json=records)
    assert bad_input.status_code == 400 and bad_input.json()["error_type"] == "evaluation_failed"
    assert bad_input.json()["detail"].startswith("Quality evaluation failed (KeyError); check that the records match")
    monkeypatch.setattr(service_module, "run_quick_quality", raise_with_path(RuntimeError))
    crash = client.post("/api/v1/evaluate", json=records)
    assert crash.status_code == 500
    assert crash.json() == {"detail": "Quality evaluation failed (RuntimeError); see the server log.", "error_type": "evaluation_failed"}
    responses += [bad_input, crash]

    monkeypatch.setattr(schema_module, "derive", raise_with_path(TypeError))
    underivable = client.post("/api/v1/validate", json={"table": "ward_daily", "records": [VALID_WARD_DAY]})
    assert underivable.status_code == 400 and underivable.json()["error_type"] == "invalid_records"
    assert underivable.json()["detail"].startswith("Could not compute the derived columns of these records (TypeError)")
    responses.append(underivable)
    for response in responses:
        assert "someone" not in response.text and "cannot read" not in response.text


def test_stored_csv_is_byte_identical_to_the_generated_csv(client: TestClient, isolated_paths: dict[str, Path]) -> None:
    """Stored datasets use LF line endings like /generate's CSV body (CRLF on Windows before)."""
    payload = {"table": "ward_daily", "num_rows": 50, "scenario": "high_occupancy", "seed": 7}
    as_json = _generate(client, **payload)                                   # persisted
    as_csv = client.post(GENERATE, json={**payload, "format": "csv"})        # persisted as well
    assert as_csv.status_code == 200
    body = as_csv.content
    assert b"\r" not in body and body.count(b"\n") == 51
    for dataset_id in (as_json["dataset_id"], as_csv.headers["x-bedsynth-dataset-id"]):
        download = client.get(f"/api/v1/datasets/{dataset_id}/download")
        assert download.status_code == 200 and download.content == body
        assert (isolated_paths["synthetic_dir"] / "api" / dataset_id / "data.csv").read_bytes() == body
    assert client.get(f"/api/v1/datasets/{as_json['dataset_id']}/download", params={"format": "json"}).json()["records"] \
        == as_json["records"]


def test_write_table_keeps_the_platform_line_ending_by_default(tmp_path: Path) -> None:
    frame = pd.DataFrame({"a": [1, 2]})
    default = data_io.write_table(frame, tmp_path / "default.csv").read_bytes()
    unix = data_io.write_table(frame, tmp_path / "lf.csv", lineterminator="\n").read_bytes()
    assert default == "a{0}1{0}2{0}".format(os.linesep).encode() and unix == b"a\n1\n2\n"


def test_malformed_json_names_the_character_offset(client: TestClient) -> None:
    text = '{"table": "ward_daily", "num_rows": 5,}'
    with pytest.raises(json.JSONDecodeError) as raised:
        json.loads(text)
    response = client.post(GENERATE, content=text, headers={"Content-Type": "application/json"})
    assert response.status_code == 422
    body = response.json()
    assert body["error_type"] == "validation_error" and body["errors"][0]["type"] == "json_invalid"
    assert body["errors"][0]["loc"] == ["body", raised.value.pos]
    assert body["detail"] == f"body (character {raised.value.pos}): JSON decode error"


@pytest.mark.parametrize("payload, field", [
    ({"table": "ward_daily", "run_validation": False}, "run_validation"),
    ({"table": "ward_daily", "filters": {"day_of_week": {"in_": ["Mon"]}}}, "in_"),
])
def test_request_bodies_reject_internal_field_names(client: TestClient, fake_generator: FakeGenerator,
                                                    payload: dict[str, Any], field: str) -> None:
    """Only the public names are accepted ('validate', 'in'), not the attribute names behind them."""
    response = client.post(GENERATE, json=payload)
    assert response.status_code == 422, response.text
    assert response.json()["errors"][0]["type"] == "extra_forbidden" and field in response.json()["detail"]
    assert fake_generator.requests == []


def test_request_bodies_accept_the_public_aliases(client: TestClient, fake_generator: FakeGenerator) -> None:
    from bedsynth.api.schemas import FilterSpec, GenerateRequest

    body = _generate(client, table="ward_daily", num_rows=5, seed=1, persist=False, validate=False,
                     filters={"day_of_week": {"in": ["monday", "Tue"]}})
    assert body["validation"] is None and body["request"]["validate"] is False and "run_validation" not in body["request"]
    assert fake_generator.requests[-1].filters == {"day_of_week": {"in": ["Mon", "Tue"]}}
    assert GenerateRequest.model_validate({"validate": False}).run_validation is False
    assert FilterSpec.model_validate({"in": [1]}).to_engine() == {"in": [1]}


def test_openapi_wording_matches_the_data_and_fields(client: TestClient) -> None:
    spec = client.get("/openapi.json").json()
    description = spec["info"]["description"]
    assert "privacy-safe" not in description
    assert "no real records; privacy assessed with distance-to-closest-record and disclosure checks" in " ".join(description.split())
    schemas = spec["components"]["schemas"]
    assert "GaussianCopulaSynthesizer" in schemas["ModelSummary"]["properties"]["synthesizer"]["description"]
    assert "GaussianCopulaSynthesizer" in schemas["ModelInfo"]["properties"]["synthesizer"]["description"]
    metrics = schemas["ModelInfo"]["properties"]["metrics"]
    assert "rank (1 = best comparable model of its table)" in metrics["description"]
    assert metrics["examples"][0]["source"] == "reports/results/model_comparison.json"
    assert schemas["EngineStatus"]["properties"]["state"]["enum"] == ["not_loaded", "loading", "ready", "failed"]
    assert "state" in schemas["EngineStatus"]["required"]


def test_model_info_passes_registry_metrics_through_unchanged() -> None:
    """GET /api/v1/models returns card['metrics'] as scripts/03_evaluate.py stored it (null when absent)."""
    from bedsynth.api.schemas import MODEL_METRICS_EXAMPLE, ModelInfo
    from bedsynth.api.service import model_info

    metrics = dict(MODEL_METRICS_EXAMPLE, fidelity=None, rank=3, comparable=False, run_quick=True)
    card = {"experiment_id": "ctgan", "table": "ward_daily", "synthesizer": "CTGANSynthesizer", "metrics": metrics,
            "path": r"C:\Users\someone\models\ward_daily\ctgan"}
    info = model_info(card, "ward_daily", "gc_tuned")
    assert info["metrics"] == metrics and list(info["metrics"]) == list(metrics)
    assert ModelInfo(**info).model_dump(mode="json")["metrics"] == metrics
    assert info["synthesizer"] == "CTGANSynthesizer" and info["is_default"] is False and "path" not in info["card"]
    # only the 'metrics' key counts: older 'evaluation' / 'scores' blocks are not reported as metrics
    assert model_info({"experiment_id": "tvae", "evaluation": {"composite": 0.7}, "scores": {"x": 1}},
                      "ward_daily", None)["metrics"] is None
    assert model_info({"experiment_id": "tvae", "metrics": None}, "ward_daily", None)["metrics"] is None


def test_models_endpoint_returns_the_card_metrics(isolated_paths: dict[str, Path]) -> None:
    from bedsynth.api.schemas import MODEL_METRICS_EXAMPLE

    class EvaluatedGenerator(FakeGenerator):
        def available_models(self, table: str | None = None) -> list[dict[str, Any]]:
            cards = super().available_models(table)
            for card in cards:
                card["metrics"] = dict(MODEL_METRICS_EXAMPLE, rank=1 if card["experiment_id"] == "gc_tuned" else 2)
            return cards

    with TestClient(create_app(generator=EvaluatedGenerator())) as test_client:
        models = test_client.get("/api/v1/models", params={"table": "ward_daily"}).json()["tables"]["ward_daily"]["models"]
    assert {model["model_id"]: model["metrics"]["rank"] for model in models} == {"gc_tuned": 1, "ctgan": 2}
    assert all(model["metrics"]["composite"] == MODEL_METRICS_EXAMPLE["composite"] for model in models)


def test_ui_offers_bash_and_powershell_commands_and_short_labels(client: TestClient) -> None:
    html = client.get("/ui").text
    assert "Copy as curl (bash)" in html and "Copy for PowerShell" in html
    assert "Invoke-RestMethod -Method Post -Uri" in html and "-ContentType 'application/json' -Body $body" in html
    assert re.search(r"\.stat \.sub \{[^}]*overflow-wrap: anywhere", html)
    assert 'replace(/Synthesizer$/, "")' in html                          # 'GaussianCopula' in the Model tile
    assert "m.metrics.composite" in html                                  # composite next to each model
    assert "engine.state" in html and "scheduleHealth(HEALTH_LOADING_POLL_MS)" in html


def test_ui_inline_script_parses(tmp_path: Path) -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    html = (PROJECT_ROOT / "bedsynth" / "api" / "static" / "index.html").read_text(encoding="utf-8")
    scripts = re.findall(r"<script>(.*?)</script>", html, flags=re.S)
    assert len(scripts) == 1
    script = tmp_path / "ui.js"
    script.write_text(scripts[0], encoding="utf-8")
    result = subprocess.run([node, "--check", str(script)], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr


def test_serve_script_describes_what_reload_watches(capsys: pytest.CaptureFixture[str]) -> None:
    serve = _load_script(PROJECT_ROOT / "scripts" / "04_serve_api.py", "bedsynth_serve_script_help")
    with pytest.raises(SystemExit):
        serve.parse_args(["--help"])
    text = " ".join(capsys.readouterr().out.split())
    assert "Python file under bedsynth/ changes" in text and "config/settings.yaml are not watched" in text
    assert "or config/ change" not in text


# ---------------------------------------------------------------------------
# The real engine
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_api_with_a_real_model_trained_on_fixture_data(isolated_paths: dict[str, Path]) -> None:
    """End to end through SyntheticDataGenerator with a segmented GaussianCopula fitted on fixture
    rows in the isolated models dir (DESIGN.md section 12; ~30 s including the SDV import)."""
    reference = make_ward_daily(n_days=60, seed=5)
    data_io.write_table(reference, data_io.processed_path("ward_daily", "train"))
    from bedsynth.synthesis.generator import SyntheticDataGenerator
    from bedsynth.synthesis.training import train_experiment

    train_experiment("ward_daily", "gc_tuned", train_df=reference, quick=True)
    generator = SyntheticDataGenerator()

    with TestClient(create_app(generator=generator)) as test_client:
        health = test_client.get("/health").json()
        assert health["models"]["ward_daily"] == {"trained": ["gc_tuned"], "default": "gc_tuned"}
        assert health["status"] == "degraded" and health["trained_tables"] == ["ward_daily"]
        model = test_client.get("/api/v1/models", params={"table": "ward_daily"}).json()["tables"]["ward_daily"]["models"][0]
        assert model["model_id"] == "gc_tuned" and model["segment_by"] == "ward_id" and model["is_default"]
        assert "path" not in model["card"], "server paths must not leak through the API"

        payload = {"table": "ward_daily", "num_rows": 50, "scenario": "high_occupancy", "seed": 42, "evaluate": True}
        body = _generate(test_client, **payload)
        assert body["n_rows"] == 50 and body["validation"]["all_hard_rules_passed"]
        assert {record["occupancy_level"] for record in body["records"]} <= {"high", "critical"}
        assert body["model"]["id"] == "gc_tuned" and body["model"]["synthesizer"] and body["model"]["sdv_version"]
        assert body["generation"]["seed"] == 42 and 0.0 < body["generation"]["acceptance_rate"] <= 1.0
        assert 0.0 <= body["quality"]["score"] <= 1.0 and body["quality"]["reference"]["subset"] == "matching"
        assert _generate(test_client, **payload, persist=False)["records"] == body["records"]

        as_csv = test_client.post(GENERATE, json={**payload, "format": "csv", "evaluate": False})
        assert as_csv.headers["x-bedsynth-model"] == "gc_tuned" and as_csv.headers["x-bedsynth-rows"] == "50"
        calendar = _generate(test_client, table="ward_daily", mode="calendar", wards=["AMU"],
                             start_date="2025-01-01", end_date="2025-01-07", seed=1)
        assert all(record["ward_id"] == "AMU" and record["date"].startswith("2025-01-0") for record in calendar["records"])
        untrained = test_client.post(GENERATE, json={"table": "patient_stays", "num_rows": 5})
        assert untrained.status_code == 503 and untrained.json()["error_type"] == "model_unavailable"


@pytest.mark.integration
def test_generate_with_trained_models(tmp_path: Path) -> None:
    """The project's own trained models (skipped until scripts/02_train_synthesizers.py has run)."""
    if not (get_path("models_dir", create=False) / "registry.json").is_file():
        pytest.skip("no trained models (run scripts/02_train_synthesizers.py)")
    from bedsynth.synthesis.generator import SyntheticDataGenerator

    generator = SyntheticDataGenerator()
    trained = [
        table for table in schema.TABLES
        if any(entry.get("trained", True) for entry in generator.available_models(table))
    ]
    if not trained:
        pytest.skip("no trained models in the registry")

    with TestClient(create_app(generator=generator, store_dir=tmp_path / "datasets")) as test_client:
        for table in trained:
            scenario = "high_occupancy" if table == "ward_daily" else "emergency_admissions"
            payload = {"table": table, "num_rows": 60, "scenario": scenario, "seed": 42, "evaluate": True}
            first = test_client.post(GENERATE, json=payload)
            assert first.status_code == 200, first.text
            body = first.json()
            assert body["n_rows"] == 60 and body["validation"]["all_hard_rules_passed"]
            assert body["model"]["id"] and body["generation"]["seed"] == 42
            repeat = test_client.post(GENERATE, json={**payload, "persist": False}).json()
            assert repeat["records"] == body["records"], "the same seed must reproduce the same records"
            stored = test_client.get(f"/api/v1/datasets/{body['dataset_id']}/download", params={"format": "json"}).json()
            assert stored["n_rows"] == 60
