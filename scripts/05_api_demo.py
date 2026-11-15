"""Exercise every BedSynth AI API endpoint in-process and record the evidence (Week 4).

Usage::

    python scripts/05_api_demo.py [--rows 300] [--scenario-rows 100] [--seed 42]
                                  [--out-dir reports/api_demo] [--store-dir DIR] [--run-tests]
                                  [--run-slow-tests]

The FastAPI app runs in-process (``fastapi.testclient``) against the *real* trained models.
Every call is written to ``reports/api_demo/NN_<step>.json`` (request, status, timing,
selected headers, response body with records truncated, and the checks applied), CSV
responses are saved next to them, and ``summary.md`` / ``summary.json`` tabulate the run.

Besides one call per endpoint, the demo generates one persisted dataset per scenario of the
catalogue (``--scenario-rows`` rows each, with a quality score when reference data exists) and
indexes them in the summary: the Week-4 "generated datasets" deliverable. Datasets are kept in
``<synthetic_dir>/api_demo`` (one ``data.csv`` + ``metadata.json`` per dataset; the folder is
reset at the start of every run) unless ``--store-dir`` names another directory. The demo also
deletes one dataset through the API to show the lifecycle.

``--run-tests`` first runs the fast test suite (``pytest -m "not slow"``) and stores its JUnit
XML as ``<results_dir>/pytest_junit.xml``, which the Week-4 report summarises as testing
evidence. ``--run-slow-tests`` (independent of ``--run-tests``) runs the slow suite
(``pytest -m slow``: real CTGAN/TVAE fits, the full simulator, real-engine API and evaluation
runs; several minutes) into ``<results_dir>/pytest_junit_slow.xml``. pytest records the
machine's host name in the XML; it is removed after each run (nothing else in the file
changes), and the outcome of each suite is recorded under ``test_runs`` in ``summary.json`` and
as a table in ``summary.md``.

Exit codes: 0 every check passed, 1 at least one check failed, 2 nothing to demo yet (no
trained models / engine unavailable).
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import getpass
import io
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence
from xml.etree import ElementTree

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Starlette warns that its TestClient prefers httpx2; the bundled httpx works fine.
warnings.filterwarnings("ignore", message=r".*httpx.*")

from bedsynth import __version__  # noqa: E402
from bedsynth.config import get_path, project_relative  # noqa: E402
from bedsynth.scenarios import SCENARIOS  # noqa: E402
from bedsynth.schema import PATIENT_STAYS, TABLES, WARD_DAILY  # noqa: E402

API = "/api/v1"
EVIDENCE_GLOBS: tuple[str, ...] = ("[0-9][0-9]_*.json", "[0-9][0-9]_*.csv", "summary.md", "summary.json")
DEMO_STORE_NAME = "api_demo"                 # <synthetic_dir>/api_demo keeps the demo datasets
JUNIT_FILE = "pytest_junit.xml"              # <results_dir>/pytest_junit.xml (read by the Week-4 report)
SLOW_JUNIT_FILE = "pytest_junit_slow.xml"    # <results_dir>/pytest_junit_slow.xml (--run-slow-tests)
#: ``hostname="..."`` attribute pytest writes on every ``<testsuite>`` element of its JUnit XML.
_JUNIT_HOSTNAME_ATTRIBUTE = re.compile(r"""(<testsuite\b[^<>]*?)\s+hostname=(?:"[^"]*"|'[^']*')""")
DATASET_DIR_PATTERN = re.compile(r"^ds-[0-9a-f]{12}$")
MAX_RECORDS_IN_EVIDENCE = 5
MAX_TEXT_IN_EVIDENCE = 2000
EVIDENCE_HEADERS: tuple[str, ...] = ("content-type", "content-disposition", "x-process-time-ms")

NO_MODELS_MESSAGE = """No trained models were found, so there is nothing to demo yet.

Prepare the data and train the synthesizers first:
    python scripts/01_prepare_data.py
    python scripts/02_train_synthesizers.py --quick
(or run everything with: python scripts/run_pipeline.py --quick)
then re-run: python scripts/05_api_demo.py"""

Check = tuple[str, Callable[[Any, Any], bool]]


# ---------------------------------------------------------------------------
# Evidence recording
# ---------------------------------------------------------------------------
@dataclass
class Step:
    """One API call (or a skipped one) of the demo."""

    index: int
    name: str
    method: str
    path: str
    expected_status: int = 200
    status_code: int | None = None
    elapsed_ms: float = 0.0
    checks: list[tuple[str, bool]] = field(default_factory=list)
    note: str = ""
    skipped: bool = False
    evidence: str | None = None

    @property
    def passed(self) -> bool:
        if self.skipped:
            return True
        return self.status_code == self.expected_status and all(ok for _, ok in self.checks)

    @property
    def outcome(self) -> str:
        return "skipped" if self.skipped else ("ok" if self.passed else "FAIL")


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:60]


def _truncate(body: Any) -> Any:
    """Keep evidence files readable: at most a few records per list of records."""
    if isinstance(body, dict):
        out = {}
        for key, value in body.items():
            if key == "records" and isinstance(value, list) and len(value) > MAX_RECORDS_IN_EVIDENCE:
                out[key] = value[:MAX_RECORDS_IN_EVIDENCE]
                out["records_truncated"] = {"shown": MAX_RECORDS_IN_EVIDENCE, "total": len(value)}
            else:
                out[key] = _truncate(value)
        return out
    if isinstance(body, list):
        return [_truncate(item) for item in body]
    return body


class DemoRunner:
    """Calls endpoints through a client, applies checks and writes one evidence file per call."""

    def __init__(self, client: Any, out_dir: Path) -> None:
        self.client = client
        self.out_dir = out_dir
        self.steps: list[Step] = []
        self.highlights: list[str] = []
        out_dir.mkdir(parents=True, exist_ok=True)
        for pattern in EVIDENCE_GLOBS:  # only files this script writes
            for stale in out_dir.glob(pattern):
                stale.unlink()

    def call(
        self,
        name: str,
        method: str,
        path: str,
        *,
        json_body: Any = None,
        params: dict[str, Any] | None = None,
        expected: int = 200,
        checks: Sequence[Check] = (),
        note: str = "",
    ) -> tuple[Step, Any, Any]:
        """Perform one request; returns (step, parsed body, raw response)."""
        step = Step(index=len(self.steps) + 1, name=name, method=method, path=path, expected_status=expected, note=note)
        started = time.perf_counter()
        response = self.client.request(method, path, json=json_body, params=params)
        step.elapsed_ms = round((time.perf_counter() - started) * 1000.0, 1)
        step.status_code = response.status_code
        body = self._parse(response)
        if response.status_code == expected:
            for label, check in checks:
                try:
                    ok = bool(check(body, response))
                except Exception:  # a failing check must not abort the demo
                    ok = False
                step.checks.append((label, ok))
        self._write(step, json_body, params, response, body)
        self.steps.append(step)
        return step, body, response

    def skip(self, name: str, method: str, path: str, reason: str) -> Step:
        step = Step(index=len(self.steps) + 1, name=name, method=method, path=path, skipped=True, note=reason)
        self.steps.append(step)
        return step

    @staticmethod
    def _parse(response: Any) -> Any:
        content_type = response.headers.get("content-type", "")
        if "json" in content_type:
            try:
                return response.json()
            except ValueError:
                return response.text
        return response.text

    def _write(self, step: Step, json_body: Any, params: Any, response: Any, body: Any) -> None:
        stem = f"{step.index:02d}_{_slug(step.name)}"
        headers = {
            key: value
            for key, value in response.headers.items()
            if key.lower() in EVIDENCE_HEADERS or key.lower().startswith("x-bedsynth-")
        }
        content_type = response.headers.get("content-type", "")
        if isinstance(body, str):
            if "text/csv" in content_type:
                (self.out_dir / f"{stem}.csv").write_text(body, encoding="utf-8")
                headers["saved_as"] = f"{stem}.csv"
            shown: Any = body if len(body) <= MAX_TEXT_IN_EVIDENCE else body[:MAX_TEXT_IN_EVIDENCE] + "\n... (truncated)"
        else:
            shown = _truncate(body)
        evidence = {
            "step": step.index,
            "name": step.name,
            "note": step.note,
            "request": {"method": step.method, "path": step.path, "params": params, "json": _truncate(json_body)},
            "response": {
                "status_code": step.status_code,
                "expected_status": step.expected_status,
                "elapsed_ms": step.elapsed_ms,
                "headers": headers,
                "body": shown,
            },
            "checks": [{"check": label, "passed": ok} for label, ok in step.checks],
            "passed": step.passed,
        }
        path = self.out_dir / f"{stem}.json"
        path.write_text(json.dumps(evidence, indent=2, default=str), encoding="utf-8")
        step.evidence = path.name


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------
def _all_hard_rules_passed(body: Any, _: Any) -> bool:
    return bool(body["validation"]["all_hard_rules_passed"])


def _records_satisfy(predicate: Callable[[dict[str, Any]], bool]) -> Callable[[Any, Any], bool]:
    return lambda body, _: bool(body["records"]) and all(predicate(record) for record in body["records"])


def _csv_rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(text)))


def _score_in_unit_interval(body: Any, _: Any) -> bool:
    score = body["quality"]["score"]
    return isinstance(score, (int, float)) and 0.0 <= score <= 1.0


def _scenario_rows_match(scenario: Any) -> Callable[[Any, Any], bool]:
    """Every generated record lies inside the scenario (conditions and filters)."""
    def check(body: Any, _: Any) -> bool:
        import pandas as pd

        from bedsynth.scenarios import scenario_mask

        frame = pd.DataFrame(body["records"])
        return bool(len(frame)) and bool(scenario_mask(frame, scenario).all())

    return check


def scenario_coverage(runner: "DemoRunner", trained: Sequence[str], reference: dict[str, Any],
                      rows: int, seed: int) -> list[dict[str, Any]]:
    """Generate and persist one dataset per catalogue scenario; returns the dataset index."""
    index: list[dict[str, Any]] = []
    for offset, scenario in enumerate(SCENARIOS.values()):
        if scenario.table not in trained:
            runner.skip(f"scenario {scenario.name}", "POST", f"{API}/generate", f"no trained model for {scenario.table}")
            continue
        evaluate = bool(reference.get(scenario.table)) and rows >= 20
        payload = {"table": scenario.table, "num_rows": rows, "scenario": scenario.name, "seed": seed + 100 + offset,
                   "evaluate": evaluate}
        step, body, _ = runner.call(
            f"scenario {scenario.name}", "POST", f"{API}/generate", json_body=payload,
            checks=[("requested rows delivered", lambda b, _: b["n_rows"] == rows),
                    ("all hard rules passed", _all_hard_rules_passed),
                    ("every row inside the scenario", _scenario_rows_match(scenario))],
        )
        body = body if isinstance(body, dict) else {}
        generation = body.get("generation") or {}
        index.append({
            "scenario": scenario.name, "table": scenario.table, "category": scenario.category,
            "requested_rows": rows, "n_rows": body.get("n_rows"), "dataset_id": body.get("dataset_id"),
            "persisted": body.get("persisted"), "model": (body.get("model") or {}).get("id"), "seed": payload["seed"],
            "acceptance_rate": generation.get("acceptance_rate"), "elapsed_seconds": generation.get("elapsed_seconds"),
            "all_hard_rules_passed": (body.get("validation") or {}).get("all_hard_rules_passed"),
            "quality_score": (body.get("quality") or {}).get("score"), "outcome": step.outcome, "evidence": step.evidence,
        })
    return index


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
def _generate_table_demos(runner: DemoRunner, table: str, rows: int, seed: int, reference: bool) -> dict[str, Any]:
    """Generation steps for one table; returns the bodies needed by later steps."""
    results: dict[str, Any] = {}
    quality_checks: list[Check] = [("quality score in [0, 1]", _score_in_unit_interval)] if reference else []
    common_checks: list[Check] = [
        ("rows generated", lambda body, _: body["n_rows"] > 0),
        ("all hard rules passed", _all_hard_rules_passed),
    ]
    if table == WARD_DAILY:
        _, baseline, _ = runner.call(
            "generate ward_daily baseline (json, evaluate)", "POST", f"{API}/generate",
            json_body={"table": WARD_DAILY, "num_rows": rows, "scenario": "baseline", "seed": seed, "evaluate": True},
            checks=[("requested rows delivered", lambda body, _: body["n_rows"] == rows), *common_checks, *quality_checks],
        )
        results["primary"] = baseline
        runner.call(
            "generate ward_daily high_occupancy (evaluate)", "POST", f"{API}/generate",
            json_body={"table": WARD_DAILY, "num_rows": rows, "scenario": "high_occupancy", "seed": seed + 1, "evaluate": True},
            checks=[
                *common_checks,
                ("only high/critical occupancy", _records_satisfy(lambda r: r["occupancy_level"] in ("high", "critical"))),
                *quality_checks,
            ],
        )
        runner.call(
            "generate ward_daily emergency_pressure (csv)", "POST", f"{API}/generate",
            json_body={"table": WARD_DAILY, "num_rows": min(rows, 150), "scenario": "emergency_pressure", "seed": seed + 2, "format": "csv"},
            checks=[
                ("text/csv response", lambda _, response: response.headers["content-type"].startswith("text/csv")),
                ("X-BedSynth headers present", lambda _, response: all(
                    response.headers.get(name) for name in ("x-bedsynth-dataset-id", "x-bedsynth-rows", "x-bedsynth-model", "x-bedsynth-scenario"))),
                ("rows header matches body", lambda body, response: int(response.headers["x-bedsynth-rows"]) == len(_csv_rows(body))),
                ("every row has emergency boarders", lambda body, _: all(int(row["emergency_boarders"]) >= 1 for row in _csv_rows(body))),
            ],
        )
        _, calendar, _ = runner.call(
            "generate ward_daily calendar January (AMU, ICU)", "POST", f"{API}/generate",
            json_body={"table": WARD_DAILY, "mode": "calendar", "wards": ["AMU", "ICU"], "start_date": "2025-01-01",
                       "end_date": "2025-01-31", "seed": seed + 3},
            checks=[
                ("one row per ward per day", lambda body, _: body["n_rows"] == 62),
                ("dates cover the period", _records_satisfy(lambda r: "2025-01-01" <= str(r.get("date")) <= "2025-01-31")),
                *common_checks[1:],
            ],
        )
        results["calendar"] = calendar
    else:
        _, delayed, _ = runner.call(
            "generate patient_stays delayed_discharge_stays (evaluate)", "POST", f"{API}/generate",
            json_body={"table": PATIENT_STAYS, "num_rows": rows, "scenario": "delayed_discharge_stays", "seed": seed + 4, "evaluate": True},
            checks=[*common_checks, ("every stay is delayed", _records_satisfy(lambda r: r["delay_reason"] != "none")), *quality_checks],
        )
        results["primary"] = delayed
        runner.call(
            "generate patient_stays long_bed_waits", "POST", f"{API}/generate",
            json_body={"table": PATIENT_STAYS, "num_rows": min(rows, 200), "scenario": "long_bed_waits", "seed": seed + 5},
            checks=[
                *common_checks,
                ("emergency admissions waiting 4+ h", _records_satisfy(
                    lambda r: r["admission_source"] == "emergency" and r["bed_wait_hours"] >= 4)),
            ],
        )
    return results


def run_demo(client: Any, out_dir: Path, rows: int = 300, seed: int = 42, scenario_rows: int = 100,
             datasets_dir: Path | None = None, test_runs: Sequence[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Call every endpoint through ``client``, write the evidence to ``out_dir`` and summarise.

    ``scenario_rows`` > 0 also generates one persisted dataset per scenario (see
    :func:`scenario_coverage`); 0 skips that part. ``datasets_dir`` (where the app stores
    datasets) is only recorded in the summary, as are ``test_runs`` (outcomes of
    :func:`run_test_suite`, stored under ``test_runs``; an empty list when no suite ran).
    """
    runner = DemoRunner(client, out_dir)

    # --- service & catalogue -------------------------------------------------
    runner.call("service info", "GET", "/", checks=[("links to /docs and /ui", lambda b, _: b["links"]["ui"] == "/ui" and b["links"]["docs"] == "/docs")])
    _, health, _ = runner.call(
        "health", "GET", "/health", checks=[("at least one trained table", lambda b, _: len(b["trained_tables"]) >= 1)]
    )
    health = health if isinstance(health, dict) else {}
    trained = [t for t in TABLES if t in health.get("trained_tables", [])]
    reference = dict(health.get("reference_data", {}))
    runner.call("demo UI", "GET", "/ui", checks=[
        ("HTML page", lambda _, r: r.headers["content-type"].startswith("text/html")),
        ("no external assets", lambda b, _: re.search(r"""(src|href)\s*=\s*["']https?://""", b) is None),
    ])
    runner.call("openapi schema", "GET", "/openapi.json", checks=[
        ("generate has >= 5 examples", lambda b, _: len(
            b["paths"][f"{API}/generate"]["post"]["requestBody"]["content"]["application/json"]["examples"]) >= 5),
    ])
    runner.call("list tables", "GET", f"{API}/tables", checks=[("two tables", lambda b, _: len(b["tables"]) == 2)])
    for table, n_train in ((WARD_DAILY, 22), (PATIENT_STAYS, 15)):
        runner.call(f"schema {table}", "GET", f"{API}/tables/{table}/schema", checks=[
            (f"{n_train} training columns", lambda b, _, n=n_train: len(b["training_columns"]) == n),
            ("rule catalogue served", lambda b, _: b["rules_available"] and len(b["rules"]) > 0),
        ])
    runner.call("scenario catalogue (with reference support)", "GET", f"{API}/scenarios", params={"include_support": "true"},
                checks=[(f"{len(SCENARIOS)} scenarios", lambda b, _: b["count"] == len(SCENARIOS))])
    runner.call("scenarios for patient_stays", "GET", f"{API}/scenarios", params={"table": PATIENT_STAYS},
                checks=[("only patient_stays", lambda b, _: {s["table"] for s in b["scenarios"]} == {PATIENT_STAYS})])
    runner.call("scenario high_occupancy", "GET", f"{API}/scenarios/high_occupancy",
                checks=[("conditions on occupancy_level", lambda b, _: "occupancy_level" in b["conditions"])])
    runner.call("trained models", "GET", f"{API}/models",
                checks=[("default model per trained table", lambda b, _: all(b["tables"][t]["default_model"] for t in trained))])

    # --- generation ----------------------------------------------------------
    generated: dict[str, dict[str, Any]] = {}
    for table in TABLES:
        if table in trained:
            generated[table] = _generate_table_demos(runner, table, rows, seed, bool(reference.get(table)))
        else:
            runner.skip(f"generate {table}", "POST", f"{API}/generate", f"no trained model for {table}")

    primary_table = next((t for t in TABLES if generated.get(t, {}).get("primary")), None)
    primary = generated.get(primary_table, {}).get("primary") if primary_table else None
    repro_rows = min(rows, 100)
    if primary_table:
        payload = {"table": primary_table, "num_rows": repro_rows, "seed": seed + 99, "persist": False}
        _, first, _ = runner.call("reproducibility run A (seed fixed)", "POST", f"{API}/generate", json_body=payload)
        runner.call("reproducibility run B (same seed)", "POST", f"{API}/generate", json_body=payload, checks=[
            ("identical records for the same seed", lambda b, _: isinstance(first, dict) and b["records"] == first.get("records")),
        ])

    # --- validation & evaluation ----------------------------------------------
    if primary and primary.get("records"):
        records = [dict(r) for r in primary["records"][:50]]
        broken = dict(records[0])
        if primary_table == WARD_DAILY:
            broken.update(ward_id="gen med" if broken["ward_id"] == "GMED" else broken["ward_id"].lower(),
                          elective_cancelled=int(broken["elective_scheduled"]) + 3)
        else:
            broken.update(ward_id=broken["ward_id"].lower(), los_nights=-1)
        runner.call("validate records (+1 corrupted)", "POST", f"{API}/validate",
                    json_body={"table": primary_table, "records": [*records, broken]},
                    checks=[("exactly the corrupted row is invalid", lambda b, _: b["invalid_rows"] == [len(records)]),
                            ("ward alias normalised", lambda b, _: b["normalisation"]["values_normalised"].get("ward_id", 0) >= 1)])
        if primary.get("persisted") and reference.get(primary_table):
            runner.call("evaluate stored dataset", "POST", f"{API}/evaluate", json_body={"dataset_id": primary["dataset_id"]},
                        checks=[("quality score in [0, 1]", _score_in_unit_interval)])
            runner.call("evaluate client records", "POST", f"{API}/evaluate",
                        json_body={"table": primary_table, "records": primary["records"][:100]},
                        checks=[("quality score in [0, 1]", _score_in_unit_interval)])
        else:
            runner.skip("evaluate", "POST", f"{API}/evaluate", "reference data or persisted dataset unavailable")

    # --- datasets --------------------------------------------------------------
    runner.call("list datasets", "GET", f"{API}/datasets", checks=[("datasets were persisted", lambda b, _: b["total"] >= 1)])
    if primary and primary.get("persisted"):
        dataset_id = primary["dataset_id"]
        base = f"{API}/datasets/{dataset_id}"
        runner.call("dataset metadata", "GET", base, checks=[("same row count", lambda b, _: b["n_rows"] == primary["n_rows"])])
        runner.call("download dataset (csv)", "GET", f"{base}/download", params={"format": "csv"}, checks=[
            ("all rows downloaded", lambda b, _: len(_csv_rows(b)) == primary["n_rows"]),
            ("X-BedSynth-Dataset-Id header", lambda _, r: r.headers.get("x-bedsynth-dataset-id") == dataset_id),
        ])
        runner.call("download dataset (json)", "GET", f"{base}/download", params={"format": "json"},
                    checks=[("records match the generation", lambda b, _: b["records"] == primary["records"])])
        runner.call("delete dataset", "DELETE", base, checks=[("deleted", lambda b, _: b["deleted"] is True)])
        runner.call("deleted dataset is gone (404)", "GET", base, expected=404,
                    checks=[("dataset_not_found error", lambda b, _: b["error_type"] == "dataset_not_found")])

    # --- error handling ----------------------------------------------------------
    runner.call("invalid request (422)", "POST", f"{API}/generate", json_body={"table": WARD_DAILY, "num_rows": 0}, expected=422,
                checks=[("uniform error body", lambda b, _: b["error_type"] == "validation_error" and "num_rows" in b["detail"])])
    runner.call("unknown scenario (404)", "POST", f"{API}/generate", json_body={"table": WARD_DAILY, "scenario": "icu_meltdown"},
                expected=404, checks=[("scenario_not_found", lambda b, _: b["error_type"] == "scenario_not_found")])
    runner.call("scenario of another table (400)", "POST", f"{API}/generate",
                json_body={"table": WARD_DAILY, "scenario": "long_bed_waits"}, expected=400,
                checks=[("scenario_table_mismatch", lambda b, _: b["error_type"] == "scenario_table_mismatch")])
    runner.call("impossible filter (400)", "POST", f"{API}/generate",
                json_body={"table": WARD_DAILY, "filters": {"occupancy_rate": {"min": 1.5}}}, expected=400,
                checks=[("invalid_filters", lambda b, _: b["error_type"] == "invalid_filters")])

    # --- one persisted dataset per scenario (Week-4 deliverable) -------------------
    coverage = scenario_coverage(runner, trained, reference, scenario_rows, seed) if scenario_rows > 0 else []

    summary = _summarise(runner, health, generated, rows, seed)
    summary["scenario_coverage"] = coverage
    summary["scenario_rows"] = scenario_rows
    summary["datasets_dir"] = display_path(datasets_dir) if datasets_dir is not None else None
    summary["test_runs"] = [dict(item) for item in test_runs or []]
    (out_dir / "summary.md").write_text(_summary_markdown(summary, runner), encoding="utf-8")
    return summary


def _generation_highlight(name: str, body: Any) -> str | None:
    if not isinstance(body, dict) or "generation" not in body:
        return None
    generation = body["generation"]
    quality = (body.get("quality") or {}).get("score")
    rate = generation.get("acceptance_rate")
    parts = [
        f"{name}: {body['n_rows']} rows",
        f"acceptance {rate:.1%}" if isinstance(rate, (int, float)) else None,
        f"{generation.get('elapsed_seconds')} s" if generation.get("elapsed_seconds") is not None else None,
        "all hard rules passed" if (body.get("validation") or {}).get("all_hard_rules_passed") else None,
        f"quality {quality:.3f}" if isinstance(quality, (int, float)) else None,
        f"model {body['model']['id']}" if body.get("model") else None,
    ]
    return ", ".join(part for part in parts if part)


def _summarise(runner: DemoRunner, health: dict[str, Any], generated: dict[str, dict[str, Any]], rows: int, seed: int) -> dict[str, Any]:
    steps = runner.steps
    highlights = []
    for table, bodies in generated.items():
        for label, body in bodies.items():
            line = _generation_highlight(f"{table} {label}", body)
            if line:
                highlights.append(line)
    return {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "bedsynth_version": __version__,
        "sdv_version": health.get("sdv_version"),
        "python": platform.python_version(),
        "rows": rows,
        "seed": seed,
        "models": health.get("models", {}),
        "reference_data": health.get("reference_data", {}),
        "total": len(steps),
        "passed": sum(1 for s in steps if s.passed and not s.skipped),
        "failed": sum(1 for s in steps if not s.passed),
        "skipped": sum(1 for s in steps if s.skipped),
        "highlights": highlights,
        "steps": [
            {"index": s.index, "name": s.name, "method": s.method, "path": s.path, "status_code": s.status_code,
             "expected_status": s.expected_status, "elapsed_ms": s.elapsed_ms, "outcome": s.outcome,
             "checks": [{"check": label, "passed": ok} for label, ok in s.checks], "note": s.note, "evidence": s.evidence}
            for s in steps
        ],
    }


def _summary_markdown(summary: dict[str, Any], runner: DemoRunner) -> str:
    lines = [
        "# BedSynth AI - API demo evidence",
        "",
        f"Generated {summary['generated_at']} by `scripts/05_api_demo.py` (BedSynth {summary['bedsynth_version']}, "
        f"SDV {summary['sdv_version']}, Python {summary['python']}). Every call ran in-process through "
        "`fastapi.testclient` against the trained models; each row links to the JSON evidence of the call.",
        "",
        f"**Result: {summary['passed']} passed, {summary['failed']} failed, {summary['skipped']} skipped "
        f"({summary['total']} steps; {summary['rows']} rows per generation, base seed {summary['seed']}).**",
        "",
        "## Trained models",
        "",
        "| Table | Default model | Trained models | Reference data |",
        "|---|---|---|---|",
    ]
    for table, info in summary["models"].items():
        trained = ", ".join(info.get("trained") or []) or "none"
        lines.append(f"| {table} | {info.get('default') or '-'} | {trained} | {'yes' if summary['reference_data'].get(table) else 'no'} |")
    if summary["highlights"]:
        lines += ["", "## Generation highlights", ""] + [f"- {line}" for line in summary["highlights"]]
    coverage = summary.get("scenario_coverage") or []
    if coverage:
        stored = sum(1 for item in coverage if item.get("persisted"))
        lines += ["", "## Scenario coverage (one persisted dataset per scenario)", "",
                  f"{stored} of {len(coverage)} scenario datasets were stored"
                  + (f" in `{summary['datasets_dir']}`" if summary.get("datasets_dir") else "") + ".", "",
                  "| Scenario | Table | Rows | Acceptance | Hard rules | Quality | Dataset | Outcome |",
                  "|---|---|---|---|---|---|---|---|"]
        for item in coverage:
            rate = item.get("acceptance_rate")
            quality = item.get("quality_score")
            lines.append(
                f"| {item['scenario']} | {item['table']} | {item.get('n_rows')} | "
                f"{f'{rate:.1%}' if isinstance(rate, (int, float)) else '-'} | "
                f"{'passed' if item.get('all_hard_rules_passed') else '-'} | "
                f"{f'{quality:.3f}' if isinstance(quality, (int, float)) else '-'} | {item.get('dataset_id') or '-'} | "
                f"{item.get('outcome')} |")
    lines += _test_runs_markdown(summary.get("test_runs") or [])
    lines += ["", "## Calls", "", "| # | Step | Request | Status | Time (ms) | Checks | Outcome | Evidence |", "|---|---|---|---|---|---|---|---|"]
    for step in runner.steps:
        checks = f"{sum(ok for _, ok in step.checks)}/{len(step.checks)}" if step.checks else "-"
        status = "-" if step.status_code is None else f"{step.status_code} (expected {step.expected_status})"
        evidence = f"[{step.evidence}]({step.evidence})" if step.evidence else step.note
        lines.append(f"| {step.index} | {step.name} | `{step.method} {step.path}` | {status} | {step.elapsed_ms} | {checks} | {step.outcome} | {evidence} |")
    failures = [(step, label) for step in runner.steps for label, ok in step.checks if not ok]
    failures += [(step, f"status {step.status_code} != {step.expected_status}") for step in runner.steps
                 if not step.skipped and step.status_code != step.expected_status]
    if failures:
        lines += ["", "## Failed checks", ""] + [f"- step {step.index} ({step.name}): {label}" for step, label in failures]
    return "\n".join(lines) + "\n"


def _test_runs_markdown(test_runs: Sequence[dict[str, Any]]) -> list[str]:
    """``## Test suites`` table of the suites run with ``--run-tests`` / ``--run-slow-tests``."""
    if not test_runs:
        return []
    lines = ["", "## Test suites", "",
             "| Suite | Command | Result | Tests | Passed | Failed | Errors | Skipped | Time (s) | JUnit XML |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for item in test_runs:
        totals = item.get("totals") or {}
        result = "ok" if item.get("ok") else "exit code {}".format(item.get("exit_code"))
        counts = " | ".join(str(totals.get(key, "-")) for key in ("tests", "passed", "failures", "errors", "skipped"))
        lines.append(f"| {item.get('suite')} | `{item.get('command')}` | {result} | {counts} | "
                     f"{totals.get('seconds', '-')} | {item.get('junit_xml')} |")
    return lines


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def load_generator() -> tuple[Any | None, str | None]:
    """The real generation engine, or (None, message) when there is nothing to demo."""
    try:
        from bedsynth.synthesis.generator import SyntheticDataGenerator

        generator = SyntheticDataGenerator()
        trained = {table: generator.available_models(table) for table in TABLES}
    except NotImplementedError:
        return None, "The generation engine (bedsynth/synthesis/generator.py) is not implemented yet."
    except Exception as exc:  # report any engine problem as a clear message
        return None, f"Could not load the generation engine: {type(exc).__name__}: {exc}"
    if not any(trained.values()):
        return None, NO_MODELS_MESSAGE
    return generator, None


def write_summary_json(summary: dict[str, Any], out_dir: Path) -> Path:
    """Machine-readable copy of the run summary (read by the Week-4 report)."""
    path = out_dir / "summary.json"
    path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return path


def render_week4_report(summary: dict[str, Any], out_dir: Path) -> Path | None:
    """Optional hook: ``bedsynth.reporting.render_week4_report`` (failures only print a warning)."""
    try:
        from bedsynth.reporting import render_week4_report as renderer

        path = renderer(summary, demo_dir=out_dir)
    except Exception as exc:  # noqa: BLE001 - the demo evidence is already written
        print(f"Week-4 report skipped ({type(exc).__name__}: {exc})", file=sys.stderr)
        return None
    print(f"Week-4 report written to {path}")
    return path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exercise every BedSynth AI API endpoint and save the evidence.")
    parser.add_argument("--rows", type=int, default=300, help="Rows per generation (default: 300).")
    parser.add_argument("--scenario-rows", type=int, default=100,
                        help="Rows of the persisted dataset generated per scenario (default: 100; 0 skips them).")
    parser.add_argument("--seed", type=int, default=42, help="Base seed (default: 42).")
    parser.add_argument("--out-dir", type=Path, default=None, help="Evidence directory (default: reports/api_demo).")
    parser.add_argument("--store-dir", type=Path, default=None,
                        help="Directory of the demo datasets (default: <synthetic_dir>/api_demo, reset on every run).")
    parser.add_argument("--run-tests", action="store_true",
                        help="Run the fast test suite first and store its JUnit XML for the Week-4 report.")
    parser.add_argument("--run-slow-tests", action="store_true",
                        help=f"Run the slow test suite (pytest -m slow, several minutes) first and store its JUnit XML "
                             f"as <results_dir>/{SLOW_JUNIT_FILE} (independent of --run-tests).")
    parser.add_argument("--no-report", action="store_true", help="Do not render the Week-4 report (bedsynth.reporting).")
    return parser.parse_args(argv)


def display_path(path: Path | str) -> str:
    """``path`` relative to the project root when it lies inside it (no user names in the evidence)."""
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def reset_demo_store(store_dir: Path) -> None:
    """Remove the datasets of a previous demo run from the dedicated demo store."""
    if not store_dir.is_dir():
        return
    for child in store_dir.iterdir():
        if child.is_dir() and (DATASET_DIR_PATTERN.match(child.name) or child.name.startswith((".staging-", ".deleting-"))):
            shutil.rmtree(child, ignore_errors=True)


def strip_junit_hostname(path: Path) -> bool:
    """Remove the ``hostname`` attribute pytest writes on ``<testsuite>`` (the rest stays byte-identical).

    Returns whether the file changed. The JUnit XML is submitted as testing evidence, and the host
    name identifies the machine it ran on. The file is read as bytes, so line endings are kept.
    """
    try:
        text = path.read_bytes().decode("utf-8")
    except (OSError, UnicodeDecodeError):
        return False
    stripped = _JUNIT_HOSTNAME_ATTRIBUTE.sub(r"\1", text)
    if stripped == text:
        return False
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(stripped.encode("utf-8"))
    os.replace(tmp, path)
    return True


def _machine_identifier_patterns() -> list[tuple[re.Pattern[str], str]]:
    """Patterns for this machine's home directory, user name (in pytest temp paths) and host name.

    The identifiers are read at run time and never written anywhere. The home directory is matched
    with native, forward-slash, doubled-backslash (``repr`` of a Windows path) and Git Bash
    (``/c/...``) separators, longest variant first.
    """
    patterns: list[tuple[re.Pattern[str], str]] = []
    home = str(Path.home())
    if len(home) > 3:
        variants = {home, home.replace("\\", "/"), home.replace("\\", "\\\\")}
        drive = re.match(r"^([A-Za-z]):[\\/](.*)$", home)
        if drive:
            variants.add("/" + drive.group(1).lower() + "/" + drive.group(2).replace("\\", "/"))
        for variant in sorted(variants, key=len, reverse=True):
            patterns.append((re.compile(re.escape(variant), re.IGNORECASE), "~"))
    try:
        user = getpass.getuser()
    except Exception:  # noqa: BLE001 - no user name available: nothing to redact
        user = ""
    if len(user) >= 2:
        patterns.append((re.compile(r"pytest-of-" + re.escape(user), re.IGNORECASE), "pytest-of-USER"))
    host = platform.node()
    if len(host) >= 3:
        patterns.append((re.compile(r"(?<![\w-])" + re.escape(host) + r"(?![\w-])", re.IGNORECASE), "<host>"))
    return patterns


def scrub_junit_identifiers(path: Path) -> int:
    """Replace this machine's home directory, user name and host name anywhere in a JUnit XML file.

    Tracebacks of failing tests carry pytest's temporary paths (under the home directory, in a
    ``pytest-of-<user>`` folder); they become ``~`` and ``pytest-of-USER``, and a host name becomes
    ``<host>``. Line endings are kept. Returns the number of replacements (0 when the file is clean
    or unreadable).
    """
    try:
        text = path.read_bytes().decode("utf-8")
    except (OSError, UnicodeDecodeError):
        return 0
    count = 0
    for pattern, replacement in _machine_identifier_patterns():
        text, replaced = pattern.subn(replacement, text)
        count += replaced
    if count:
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_bytes(text.encode("utf-8"))
        os.replace(tmp, path)
    return count


def junit_totals(path: Path) -> dict[str, Any] | None:
    """Tests, passed, failures, errors, skipped and seconds of a JUnit XML file (``None`` if unreadable)."""
    try:
        root = ElementTree.parse(path).getroot()
    except (OSError, ElementTree.ParseError):
        return None
    suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
    totals: dict[str, Any] = {key: sum(int(suite.get(key) or 0) for suite in suites)
                              for key in ("tests", "failures", "errors", "skipped")}
    totals["passed"] = totals["tests"] - totals["failures"] - totals["errors"] - totals["skipped"]
    totals["seconds"] = round(sum(float(suite.get("time") or 0.0) for suite in suites), 1)
    return totals


def run_test_suite(results_dir: Path, *, slow: bool = False) -> dict[str, Any]:
    """Run the fast (``pytest -m "not slow"``) or the slow (``-m slow``) suite and keep its JUnit XML.

    The XML (``pytest_junit.xml`` / ``pytest_junit_slow.xml`` in ``results_dir``) is testing
    evidence for the Week-4 report; its host name is removed (:func:`strip_junit_hostname`) and
    any home-directory path, user name or host name in failure tracebacks is redacted
    (:func:`scrub_junit_identifiers`). Returns the outcome recorded under ``test_runs`` in the demo
    summary (paths project-relative).
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    junit = results_dir / (SLOW_JUNIT_FILE if slow else JUNIT_FILE)
    marker = "slow" if slow else "not slow"
    command = [sys.executable, "-m", "pytest", "-q", "-m", marker, f"--junitxml={junit}"]
    started = time.perf_counter()
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
    seconds = round(time.perf_counter() - started, 1)
    hostname_removed = strip_junit_hostname(junit) if junit.is_file() else False
    identifiers_redacted = scrub_junit_identifiers(junit) if junit.is_file() else 0
    tail = completed.stdout.strip().splitlines()[-1:] if completed.stdout else []
    shown = project_relative(junit)
    print(f"{'Slow' if slow else 'Fast'} test suite finished with exit code {completed.returncode} in {seconds:.0f} s "
          f"({tail[0] if tail else 'no output'}); JUnit XML at {shown}")
    return {
        "suite": "slow" if slow else "fast",
        "command": f'python -m pytest -q -m "{marker}" --junitxml={shown}',
        "exit_code": completed.returncode,
        "ok": completed.returncode == 0,
        "junit_xml": shown,
        "hostname_removed": hostname_removed,
        "identifiers_redacted": identifiers_redacted,
        "totals": junit_totals(junit) if junit.is_file() else None,
        "seconds": seconds,
        "finished_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    generator, problem = load_generator()
    if generator is None:
        print(problem, file=sys.stderr)
        return 2

    from fastapi.testclient import TestClient

    from bedsynth.api.main import create_app

    test_runs = []
    if args.run_tests:
        test_runs.append(run_test_suite(get_path("results_dir")))
    if args.run_slow_tests:
        test_runs.append(run_test_suite(get_path("results_dir"), slow=True))
    out_dir = args.out_dir or get_path("reports_dir") / "api_demo"
    store_dir = args.store_dir or get_path("synthetic_dir") / DEMO_STORE_NAME
    if args.store_dir is None:
        reset_demo_store(store_dir)
    with TestClient(create_app(generator=generator, store_dir=store_dir)) as client:
        summary = run_demo(client, out_dir, rows=args.rows, seed=args.seed, scenario_rows=args.scenario_rows,
                           datasets_dir=store_dir, test_runs=test_runs)

    for step in summary["steps"]:
        print(f"{step['index']:>2}. {step['outcome']:<7} {step['method']:<6} {step['path']:<45} {step['name']}")
    print(f"\n{summary['passed']} passed, {summary['failed']} failed, {summary['skipped']} skipped. "
          f"Evidence written to {out_dir} (summary.md).")
    write_summary_json(summary, out_dir)
    if not args.no_report:
        render_week4_report(summary, out_dir)
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
