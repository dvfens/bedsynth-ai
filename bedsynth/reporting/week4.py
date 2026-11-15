"""Week-4 report: final API and system evaluation (DESIGN.md §10-§11).

:func:`render_week4_report` combines the API demo evidence of ``scripts/05_api_demo.py``
(its summary dict, or ``reports/api_demo/summary.json``: endpoint calls and one persisted
dataset per scenario), the model comparison of Week 3 and the testing evidence (JUnit XML of
the fast suite, ``<results_dir>/pytest_junit.xml``, written by ``05_api_demo.py --run-tests``,
and of the slow suite, ``<results_dir>/pytest_junit_slow.xml``, written by ``05_api_demo.py
--run-slow-tests``) into ``reports/week4_api_system.md``.

The generator documentation in section 1 is generated, not typed: the request parameters and
the response fields come from the pydantic models of :mod:`bedsynth.api.schemas` (their JSON
schema; importing them does not load SDV), the start command and default port from
``scripts/04_serve_api.py``, and the example request and response from the demo evidence.
Missing optional inputs (no demo evidence, no JUnit file, no comparison) degrade to a note.
"""
from __future__ import annotations

import importlib.util
import json
import logging
import xml.etree.ElementTree as ElementTree
from pathlib import Path
from typing import Any

from bedsynth import schema
from bedsynth.config import PROJECT_ROOT, get_path, get_settings
from bedsynth.reporting._common import (
    display_path,
    draw_figure,
    figure_line,
    fmt_int,
    fmt_num,
    fmt_pct,
    is_number,
    md_table,
    output_dirs,
    project_ref,
    to_utc,
    utc_now,
    write_markdown,
)
from bedsynth.scenarios import SCENARIOS

logger = logging.getLogger(__name__)

REPORT_NAME = "week4_api_system.md"
DEMO_SUMMARY_FILE = "summary.json"
JUNIT_FILE = "pytest_junit.xml"
JUNIT_SLOW_FILE = "pytest_junit_slow.xml"
TEST_COMMAND = 'python -m pytest -q -m "not slow" --junitxml=reports/results/pytest_junit.xml'
SLOW_TEST_COMMAND = "python scripts/05_api_demo.py --run-slow-tests"
SLOW_PYTEST_COMMAND = "python -m pytest -q -m slow --junitxml=reports/results/pytest_junit_slow.xml"
SERVE_SCRIPT = "scripts/04_serve_api.py"
MAX_FAILURES_LISTED = 10
EXAMPLE_MAX_FIELDS = 8
"""Fields kept per nested object in the trimmed example response."""
#: Fields shown first in the trimmed example (the most informative ones of the generation report).
EXAMPLE_KEYS: dict[str, tuple[str, ...]] = {
    "generation": ("seed", "num_rows_requested", "num_rows_generated", "shortfall", "attempts", "sampled_rows",
                   "accepted_rows", "acceptance_rate", "rejected_by_rule"),
    "validation": ("n_rows", "n_hard_valid", "hard_valid_rate", "all_hard_rules_passed", "hard_rules", "soft_rules"),
}

#: Documentation and demo materials listed in section 1.5 (shown when the file exists).
DEMO_MATERIALS: tuple[tuple[str, str], ...] = (
    ("docs/API.md", "API reference: every endpoint, parameters, responses, errors and limits"),
    ("docs/DEMO_SCRIPT.md", "live demo runbook with timings, commands and a fallback plan"),
    ("demo/requests.http", "ready-made requests for the VS Code REST Client"),
    ("demo/demo_client.py", "Python client that walks through every endpoint of a running server"),
    ("scripts/05_api_demo.py", "the in-process demo run behind sections 2-3 (evidence in the demo folder)"),
)

#: Operations of the service (method, path) as listed by ``GET /``; the summary counts the list stored in the
#: demo evidence and falls back to this one when that evidence is missing.
API_ENDPOINTS: tuple[tuple[str, str], ...] = (
    ("GET", "/"), ("GET", "/health"), ("GET", "/ui"), ("GET", "/api/v1/tables"),
    ("GET", "/api/v1/tables/{table}/schema"), ("GET", "/api/v1/scenarios"), ("GET", "/api/v1/scenarios/{name}"),
    ("GET", "/api/v1/models"), ("POST", "/api/v1/generate"), ("POST", "/api/v1/validate"), ("POST", "/api/v1/evaluate"),
    ("GET", "/api/v1/datasets"), ("GET", "/api/v1/datasets/{dataset_id}"), ("DELETE", "/api/v1/datasets/{dataset_id}"),
    ("GET", "/api/v1/datasets/{dataset_id}/download"),
)

#: Response fields whose pydantic model carries no description.
RESPONSE_FALLBACK: dict[str, str] = {
    "table": "Table generated.",
    "warnings": "Warnings of the generation and validation steps.",
}

COVERAGE_COLUMNS_NOTE = (
    "*acceptance* = accepted / sampled rows of that request; *hard rules* = every delivered row passed all hard "
    "rules; *quality* = the API's quick quality score (column shapes, pair trends and validity) against the "
    "reference data (the matching real scenario subset when it is large enough)."
)


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Cannot read %s: %s", path, exc)
        return None


def load_junit_summary(path: Path) -> dict[str, Any] | None:
    """Totals, per-file counts and failures of a pytest JUnit XML file (``None`` if unreadable)."""
    if not path.is_file():
        return None
    try:
        root = ElementTree.parse(path).getroot()
    except (OSError, ElementTree.ParseError) as exc:
        logger.warning("Cannot read %s: %s", path, exc)
        return None
    suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
    by_file: dict[str, dict[str, int]] = {}
    failures: list[str] = []
    totals = {"tests": 0, "passed": 0, "failed": 0, "errors": 0, "skipped": 0}
    duration = 0.0
    for suite in suites:
        duration += float(suite.get("time") or 0.0)
        for case in suite.iter("testcase"):
            module = (case.get("classname") or "unknown").split(".")
            name = next((part for part in module if part.startswith("test_")), module[-1]) + ".py"
            outcome = "passed"
            if case.find("failure") is not None:
                outcome = "failed"
            elif case.find("error") is not None:
                outcome = "errors"
            elif case.find("skipped") is not None:
                outcome = "skipped"
            counts = by_file.setdefault(name, {"passed": 0, "failed": 0, "errors": 0, "skipped": 0})
            counts[outcome] += 1
            totals[outcome] += 1
            totals["tests"] += 1
            if outcome in ("failed", "errors") and len(failures) < MAX_FAILURES_LISTED:
                failures.append(f"{name}::{case.get('name')}")
    return {"totals": totals, "by_file": dict(sorted(by_file.items())), "failures": failures,
            "duration_seconds": round(duration, 1), "timestamp": suites[0].get("timestamp") if suites else None}


def _junit_lines(label: str, command: str, junit: dict[str, Any]) -> list[str]:
    """Totals sentence and per-file table of one JUnit summary (timestamp in UTC)."""
    totals = junit["totals"]
    lines = [f"{label} (`{command}`, {to_utc(junit.get('timestamp')) or 'time unknown'}): "
             f"**{totals['passed']} passed, {totals['failed']} failed, {totals['errors']} errors, "
             f"{totals['skipped']} skipped** of {totals['tests']} tests in {junit['duration_seconds']} s.", ""]
    rows = [[name, counts["passed"], counts["failed"] + counts["errors"], counts["skipped"]]
            for name, counts in junit["by_file"].items()]
    lines += md_table(["test file", "passed", "failed", "skipped"], rows) + [""]
    if junit["failures"]:
        lines += ["Failing tests: " + ", ".join(f"`{name}`" for name in junit["failures"]), ""]
    return lines


def _testing_section(junit: dict[str, Any] | None, slow: dict[str, Any] | None = None) -> list[str]:
    lines = ["## 5. Testing evidence", ""]
    if not junit:
        lines += [f"_No fast-suite results stored yet: run `{TEST_COMMAND}` (or `python scripts/05_api_demo.py "
                  "--run-tests`) and re-render this report._", ""]
    else:
        lines += _junit_lines("Fast test suite", 'pytest -m "not slow"', junit)
        lines += ["The API tests drive every endpoint through `fastapi.testclient` (contract, limits, error envelope, "
                  "concurrency and security regressions).", ""]
    if slow:
        lines += _junit_lines("Slow test suite", "pytest -m slow", slow)
        lines += ["The slow tests fit real SDV models (CTGAN and TVAE included) and run the simulator calibration, "
                  "the API and the evaluation on real engines end to end.", ""]
    else:
        lines += ["The slow tests (marker `slow`: real SDV fits, simulator calibration, real-engine API and evaluation "
                  f"runs) have no stored result yet: run `{SLOW_TEST_COMMAND}` (or `{SLOW_PYTEST_COMMAND}`), which "
                  f"writes `reports/results/{JUNIT_SLOW_FILE}`, and re-render this report.", ""]
    return lines


def _coverage_section(summary: dict[str, Any] | None) -> list[str]:
    lines = ["## 3. Scenario coverage through the API", ""]
    coverage = (summary or {}).get("scenario_coverage") or []
    if not coverage:
        return lines + ["_The API demo has not generated the per-scenario datasets yet "
                        "(`python scripts/05_api_demo.py`)._", ""]
    stored = sum(1 for item in coverage if item.get("persisted"))
    ok = sum(1 for item in coverage if item.get("outcome") == "ok")
    where = summary.get("datasets_dir")
    lines += [
        f"The demo generated one dataset per scenario ({summary.get('scenario_rows')} rows each): {ok} of "
        f"{len(coverage)} met every check (requested rows, all hard rules, every row inside the scenario) and "
        f"{stored} were stored" + (f" in `{display_path(where)}`" if where else "") + " as `data.csv` + "
        "`metadata.json` (request, generation report, validation, quality). The same folder also keeps the datasets "
        "that the other generation steps of section 2 persisted and the demo did not delete.",
        "",
        COVERAGE_COLUMNS_NOTE,
        "",
    ]
    rows = []
    for item in coverage:
        rate, quality = item.get("acceptance_rate"), item.get("quality_score")
        rows.append([item.get("scenario"), item.get("table"), item.get("category"), fmt_int(item.get("n_rows")),
                     fmt_pct(rate), "yes" if item.get("all_hard_rules_passed") else "-",
                     fmt_num(quality) if is_number(quality) else "-",
                     item.get("dataset_id") or "-", item.get("outcome")])
    lines += md_table(["scenario", "table", "category", "rows", "acceptance", "hard rules", "quality", "dataset",
                       "outcome"], rows)
    return lines + [""]


# ---------------------------------------------------------------------------
# Section 1: the generator (start command, request, response, example, materials)
# ---------------------------------------------------------------------------
def _serve_defaults() -> tuple[str | None, int | None]:
    """Default host and port of ``scripts/04_serve_api.py`` (read from its argument parser)."""
    path = PROJECT_ROOT / SERVE_SCRIPT
    try:
        spec = importlib.util.spec_from_file_location("bedsynth_serve_api_script", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # top level only defines the CLI; uvicorn is imported in main()
        args = module.parse_args([])
        return str(args.host), int(args.port)
    except Exception as exc:  # noqa: BLE001 - documentation only
        logger.info("Cannot read the defaults of %s: %s", SERVE_SCRIPT, type(exc).__name__)
        return None, None


def _usage_section() -> list[str]:
    host, port = _serve_defaults()
    base = f"http://{host}:{port}" if host and port else None
    binding = f" (binds {host}:{port} by default)" if base else ""
    lines = ["### 1.1 Running the generator", "",
             f"From the project root, `python {SERVE_SCRIPT}` starts the service with uvicorn{binding}; "
             f"`--port <port>` serves it on another port when {port or 'the default port'} is already in use, "
             "`--host 0.0.0.0` listens on every interface and `--reload` restarts it when the code changes. "
             "`uvicorn bedsynth.api.main:app --port <port>` runs the same application."]
    if base:
        lines.append(f"Then open {base}/ui for the offline demo UI and {base}/docs for the interactive OpenAPI "
                     "documentation (Swagger UI, loaded from a CDN); /health reports the status, the trained models "
                     "and the reference data.")
    lines += ["Without trained models the service still starts: /health reports `degraded` and generation answers "
              "503 with the command to run.", ""]
    return lines


def _ref_name(ref: str) -> str:
    return ref.rsplit("/", 1)[-1]


def _type_text(prop: dict[str, Any]) -> str:
    """Compact type of a JSON-schema property: type, allowed values and limits."""
    options = [o for o in (prop.get("anyOf") or [prop]) if o.get("type") != "null"]
    texts = []
    for option in options:
        if "enum" in option:
            texts.append(" / ".join(f"'{value}'" for value in option["enum"]))
            continue
        kind = "object" if "$ref" in option else option.get("type") or "any"
        if kind == "array":
            text = f"list of {_type_text(option.get('items') or {})}"
            low, high = option.get("minItems"), option.get("maxItems")
            if low is not None or high is not None:
                text += f" ({low if low is not None else 0}-{high if high is not None else 'any'} items)"
        else:
            text = kind + (f" ({option['format']})" if option.get("format") else "")
            low, high, above = option.get("minimum"), option.get("maximum"), option.get("exclusiveMinimum")
            if low is not None and high is not None:
                text += f", {low}..{high}"
            elif above is not None and high is not None:
                text += f", > {above} and <= {high}"
            elif high is not None:
                text += f", <= {high}"
            if option.get("maxLength") is not None:
                text += f", at most {option['maxLength']} characters"
        texts.append(text)
    return " or ".join(texts) or "any"


def _default_text(prop: dict[str, Any], required: bool) -> str:
    if required:
        return "required"
    if "default" not in prop or prop["default"] is None:
        return "-"
    value = prop["default"]
    return f"`{value if isinstance(value, str) else json.dumps(value)}`"


def _request_section() -> list[str]:
    """Parameters of ``POST /api/v1/generate`` from the pydantic request model's JSON schema."""
    try:
        from bedsynth.api.schemas import GenerateRequest
    except Exception as exc:  # noqa: BLE001 - documentation only
        logger.info("bedsynth.api.schemas unavailable: %s", type(exc).__name__)
        return []
    request_schema = GenerateRequest.model_json_schema()
    properties = request_schema.get("properties") or {}
    required = set(request_schema.get("required") or [])
    rows = [[f"`{name}`", _type_text(prop), _default_text(prop, name in required), prop.get("description") or ""]
            for name, prop in properties.items()]
    count = (f"{len(rows)} fields, all optional" if not required
             else f"{len(rows)} fields ({len(required & set(properties))} required)")
    strict = "; unknown fields are rejected with 422" if request_schema.get("additionalProperties") is False else ""
    return (["### 1.2 Generation request (`POST /api/v1/generate`)", "",
             f"JSON body with {count} (read from the request model `bedsynth.api.schemas.GenerateRequest`{strict}). "
             "`-` = no default: omit the field or send null.", ""]
            + md_table(["field", "type", "default", "meaning"], rows)
            + ["", "`validate` only switches the validation report on or off: rejection sampling against every hard "
               "rule always runs.", ""])


def _response_field_text(prop: dict[str, Any], defs: dict[str, Any], name: str) -> str:
    """Description of one response field, completed from the referenced model (its wording and fields)."""
    text = (prop.get("description") or "").rstrip(".")
    options = [o for o in (prop.get("anyOf") or [prop]) if o.get("type") != "null"]
    nullable = len(options) < len(prop.get("anyOf") or [prop])
    option = options[0] if options else {}
    is_list = option.get("type") == "array"
    ref = option.get("$ref") or (option.get("items") or {}).get("$ref")
    if ref:
        model = defs.get(_ref_name(ref)) or {}
        if not text:
            wording = (model.get("description") or "").rstrip(".")
            text = (f"list of: {wording[:1].lower()}{wording[1:]}" if is_list else wording) if wording else ""
        fields = ", ".join(model.get("properties") or {})
        if fields:
            text = f"{text} - fields {fields}" if text else f"fields {fields}"
    text = text or RESPONSE_FALLBACK.get(name, "").rstrip(".")
    if nullable and "null" not in text:
        text = f"{text}; may be null" if text else "may be null"
    return (text[:1].upper() + text[1:] + ".") if text else "-"


def _response_section(summary: dict[str, Any] | None, demo_path: Path) -> list[str]:
    """Fields of the JSON generation response and the headers of the CSV response."""
    try:
        from bedsynth.api.schemas import GenerateResponse
    except Exception:  # noqa: BLE001 - documentation only
        return []
    response_schema = GenerateResponse.model_json_schema()
    defs = response_schema.get("$defs") or {}
    rows = [[f"`{name}`", _response_field_text(prop, defs, name)]
            for name, prop in (response_schema.get("properties") or {}).items()]
    lines = ["### 1.3 Generation response", "",
             "With `format: \"json\"` the response carries the records together with their metadata (from the "
             "response model `bedsynth.api.schemas.GenerateResponse`):", ""]
    lines += md_table(["field", "content"], rows) + [""]
    headers = _csv_headers(summary, demo_path)
    if headers:
        lines += ["With `format: \"csv\"` the body is `text/csv` and the dataset is described by response headers "
                  f"(as recorded by the demo): {', '.join(f'`{h}`' for h in headers)}; the full report is stored with "
                  "the dataset (`GET /api/v1/datasets/{id}`) when it is persisted.", ""]
    return lines


def _header_name(name: str) -> str:
    return "-".join("BedSynth" if part.lower() == "bedsynth" else part.capitalize() for part in name.split("-"))


def _csv_headers(summary: dict[str, Any] | None, demo_path: Path) -> list[str]:
    """``X-BedSynth-*`` headers of the first CSV generation in the demo evidence."""
    for step in (summary or {}).get("steps") or []:
        if step.get("path") != "/api/v1/generate" or step.get("outcome") != "ok":
            continue
        evidence = _load_evidence(demo_path, step)
        headers = ((evidence or {}).get("response") or {}).get("headers") or {}
        names = [_header_name(name) for name in headers if name.lower().startswith("x-bedsynth-")]
        if names:
            return names
    return []


def _load_evidence(demo_path: Path, step: dict[str, Any]) -> dict[str, Any] | None:
    name = str(step.get("evidence") or "")
    if not name.endswith(".json") or "/" in name or "\\" in name:
        return None
    return _load_json(demo_path / name)


def _trim(value: Any, depth: int = 0, key: str = "") -> Any:
    """Shortened copy of a JSON value for the example (nested objects and lists cut down)."""
    if isinstance(value, dict):
        nested = any(isinstance(v, (dict, list)) for v in value.values())
        if depth >= 2 and (nested or len(value) > EXAMPLE_MAX_FIELDS):
            return f"{{... {len(value)} fields}}"
        preferred = [k for k in EXAMPLE_KEYS.get(key, ()) if k in value]
        order = preferred + [k for k in value if k not in preferred]
        keep = order if not depth else order[:max(EXAMPLE_MAX_FIELDS, len(preferred))]
        trimmed = {name: _trim(value[name], depth + 1, name) for name in keep}
        if len(order) > len(keep):
            trimmed["..."] = f"{len(order) - len(keep)} more fields"
        return trimmed
    if isinstance(value, list):
        if not value:
            return []
        if depth >= 2:
            return f"[... {len(value)} items]"
        first = _trim(value[0], depth + 1)
        return [first, f"... {len(value) - 1} more"] if len(value) > 1 else [first]
    if isinstance(value, str) and len(value) > 90:
        return value[:87] + "..."
    return value


def _example_evidence(summary: dict[str, Any] | None, demo_path: Path) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """First successful JSON generation of the demo with its evidence file (request and response body)."""
    for step in (summary or {}).get("steps") or []:
        if step.get("path") != "/api/v1/generate" or step.get("outcome") != "ok":
            continue
        evidence = _load_evidence(demo_path, step) or {}
        request = (evidence.get("request") or {}).get("json")
        body = (evidence.get("response") or {}).get("body")
        if isinstance(request, dict) and request and isinstance(body, dict) and body:
            return step, evidence
    return None


def _example_section(summary: dict[str, Any] | None, demo_path: Path) -> list[str]:
    """One real request/response pair from the demo evidence, trimmed."""
    lines = ["### 1.4 Example request and response", ""]
    found = _example_evidence(summary, demo_path)
    if found is None:
        return lines + ["_No generation evidence yet: run `python scripts/05_api_demo.py` and re-render this report._", ""]
    step, evidence = found
    request, response = evidence["request"], evidence["response"]
    body = dict(response["body"])
    body.pop("records_truncated", None)  # bookkeeping of the demo's evidence file, not part of the API response
    records = body.get("records") or []
    total = int(body["n_rows"]) if is_number(body.get("n_rows")) else len(records)
    trimmed = _trim(body)
    if records:
        trimmed["records"] = [_trim(records[0], 1)] + ([f"... {fmt_int(total - 1)} more records"] if total > 1 else [])
    generation, checked = body.get("generation") or {}, body.get("validation") or {}
    lines += [f"Demo step {step.get('index')} ({step.get('name')}; evidence "
              f"`{display_path(demo_path / str(step.get('evidence')))}`) sent `{request.get('method', 'POST')} "
              f"{step.get('path')}` with this body:", "",
              "```json", json.dumps(request["json"], indent=2, ensure_ascii=False), "```", "",
              f"The response (status {response.get('status_code')}) carried {fmt_int(body.get('n_rows'))} records "
              f"(seed {generation.get('seed')}, acceptance {fmt_pct(generation.get('acceptance_rate'))}, all hard rules "
              f"passed: {'yes' if checked.get('all_hard_rules_passed') else 'no'}); trimmed below (nested objects "
              "shortened, first record only):", "",
              "```json", json.dumps(trimmed, indent=2, ensure_ascii=False), "```", ""]
    return lines


def _materials_section(report_dir: Path, demo_path: Path) -> list[str]:
    rows = [f"- {project_ref(path, report_dir)}: {text}" for path, text in DEMO_MATERIALS
            if (PROJECT_ROOT / path).exists()]
    lines = ["### 1.5 Documentation and demo materials", ""]
    lines += rows or ["_No documentation files found._"]
    lines += [f"- Demo evidence: `{display_path(demo_path)}/` (one JSON file per call, `summary.md`) and the offline UI "
              "at /ui.", ""]
    return lines


def _latency_figure(steps: list[dict[str, Any]], path: Path) -> Path | None:
    timed = [s for s in steps if s.get("outcome") != "skipped" and isinstance(s.get("elapsed_ms"), (int, float))]

    def draw(fig: Any) -> None:
        ax = fig.add_subplot(1, 1, 1)
        names = [f"{s['index']}. {s['name']}"[:48] for s in timed]
        colours = ["#009E73" if s.get("outcome") == "ok" else "#D55E00" for s in timed]
        ax.barh(range(len(timed)), [s["elapsed_ms"] / 1000.0 for s in timed], color=colours)
        ax.set_yticks(range(len(timed)))
        ax.set_yticklabels(names, fontsize=7)
        ax.invert_yaxis()
        ax.set_xlabel("response time (s, in-process)")
        ax.set_title("API demo calls (green = all checks passed)")
        ax.grid(axis="x", alpha=0.3)

    return draw_figure(path, draw, size=(8.5, 0.28 * max(len(timed), 6) + 1.2))


def _endpoint_counts(summary: dict[str, Any] | None, demo_path: Path) -> tuple[int, int]:
    """(operations, paths) the service lists at ``GET /``: from the demo evidence, else :data:`API_ENDPOINTS`."""
    for step in (summary or {}).get("steps") or []:
        if step.get("method") == "GET" and step.get("path") == "/" and step.get("evidence"):
            evidence = _load_json(demo_path / str(step["evidence"])) or {}
            endpoints = ((evidence.get("response") or {}).get("body") or {}).get("endpoints")
            if isinstance(endpoints, list) and endpoints and all(isinstance(e, dict) for e in endpoints):
                return len(endpoints), len({e.get("path") for e in endpoints})
    return len(API_ENDPOINTS), len({path for _, path in API_ENDPOINTS})


def _summary_lines(summary: dict[str, Any] | None, comparison: dict[str, Any] | None, junit: dict[str, Any] | None,
                   slow: dict[str, Any] | None, endpoints: tuple[int, int]) -> list[str]:
    """Headline facts of the Week-4 deliverable, computed from the evidence files (placed under the title)."""
    n_endpoints, n_paths = endpoints
    lines = ["## Summary", "",
             f"- FastAPI generator with {n_endpoints} endpoints on {n_paths} paths (section 1): parameterised generation (scenario, "
             "conditions, filters, wards, seed, sample or calendar mode, JSON or CSV), validation of every record "
             "against the operational rules, metadata with every response, persisted datasets and an offline demo UI; "
             "how to run it, the request parameters, the response fields and an example are in sections 1.1-1.4."]
    if summary:
        coverage = summary.get("scenario_coverage") or []
        text = f"- End-to-end demo: {summary.get('passed')} of {summary.get('total')} API calls passed their checks"
        if coverage:
            text += (f"; {sum(1 for c in coverage if c.get('outcome') == 'ok')} of {len(coverage)} scenarios returned "
                     "validated datasets")
        lines.append(text + " (sections 2-3).")
    defaults = []
    for table in schema.TABLES:
        info = ((comparison or {}).get("tables") or {}).get(table) or {}
        best = next((e for e in info.get("models") or [] if e.get("model_id") == info.get("best_model")), None)
        if best:
            defaults.append(f"{table} `{best.get('model_id')}` (composite {fmt_num(best.get('composite'))})")
    if defaults:
        lines.append(f"- Default models behind the API: {'; '.join(defaults)} (section 4).")
    tests = []
    for label, data in (("fast", junit), ("slow", slow)):
        totals = (data or {}).get("totals") or {}
        if totals:
            tests.append(f"{totals.get('passed')} of {totals.get('tests')} {label} tests passed")
    if tests:
        lines.append(f"- Tests: {'; '.join(tests)}" + ("" if slow else " (no stored slow-suite result)") + " (section 5).")
    return lines + [""]


def render_week4_report(
    demo_summary: dict[str, Any] | None = None,
    reports_dir: Path | str | None = None,
    figures_dir: Path | str | None = None,
    demo_dir: Path | str | None = None,
) -> Path:
    """Write the Week-4 API and system report.

    Args:
        demo_summary: Summary dict returned by ``run_demo`` in ``scripts/05_api_demo.py``;
            default: ``<demo_dir>/summary.json``.
        reports_dir, figures_dir: Output directories (default: configured directories).
        demo_dir: Directory of the demo evidence (default ``reports/api_demo``).

    Returns:
        Path of ``week4_api_system.md``.
    """
    report_dir, figure_dir = output_dirs(reports_dir, figures_dir)
    demo_path = Path(demo_dir) if demo_dir is not None else report_dir / "api_demo"
    summary = demo_summary or _load_json(demo_path / DEMO_SUMMARY_FILE)
    results_dir = get_path("results_dir", create=False)
    comparison = _load_json(results_dir / "model_comparison.json")
    junit = load_junit_summary(results_dir / JUNIT_FILE)
    slow = load_junit_summary(results_dir / JUNIT_SLOW_FILE)
    endpoint_rows = [
        ["GET", "/", "service info and links"], ["GET", "/health", "status, versions, trained models, reference data"],
        ["GET", "/ui", "single-page demo UI (no CDN)"], ["GET", "/api/v1/tables", "tables, column counts, default model"],
        ["GET", "/api/v1/tables/{table}/schema", "schema and rule catalogue"],
        ["GET", "/api/v1/scenarios[/{name}]", f"scenario catalogue ({len(SCENARIOS)} scenarios)"],
        ["GET", "/api/v1/models", "trained models, metrics, defaults"],
        ["POST", "/api/v1/generate", "generate validated synthetic data (sample or calendar mode)"],
        ["POST", "/api/v1/validate", "validate client records"], ["POST", "/api/v1/evaluate", "quick quality + validity"],
        ["GET/DELETE", "/api/v1/datasets[/{id}[/download]]", "persisted generations"],
    ]

    lines = [
        "# BedSynth AI - Week 4: API and system evaluation",
        "",
        f"Generated {utc_now()}.",
        "",
        *_summary_lines(summary, comparison, junit, slow, _endpoint_counts(summary, demo_path)),
        "## 1. Service",
        "",
        "The FastAPI service (`bedsynth.api`, `python scripts/04_serve_api.py`) exposes the generation engine: "
        "scenario-aware generation (JSON or CSV with `X-BedSynth-*` headers), record validation against the "
        "operational rules, quick quality evaluation against the reference data, persisted datasets and an offline "
        "demo UI at `/ui` (OpenAPI docs at `/docs`).",
        "",
    ]
    lines += md_table(["method", "path", "purpose"], endpoint_rows) + [""]
    lines += _usage_section()
    lines += _request_section()
    lines += _response_section(summary, demo_path)
    lines += _example_section(summary, demo_path)
    lines += _materials_section(report_dir, demo_path)

    lines += ["## 2. End-to-end API demo", ""]
    if not summary:
        lines += ["_No API demo evidence yet: run `python scripts/05_api_demo.py` after training the models._", ""]
    else:
        lines += [
            f"`scripts/05_api_demo.py` exercised every endpoint in-process against the trained models "
            f"({to_utc(summary.get('generated_at')) or '-'}, SDV {summary.get('sdv_version')}, {summary.get('rows')} rows "
            f"per generation, base seed {summary.get('seed')}): **{summary.get('passed')} passed, {summary.get('failed')} "
            f"failed, {summary.get('skipped')} skipped** of {summary.get('total')} steps. Evidence: "
            f"`{display_path(demo_path)}/` (one JSON file per call plus `summary.md`).",
            "",
        ]
        model_rows = [[t, (info or {}).get("default") or "-", ", ".join((info or {}).get("trained") or []) or "-"]
                      for t, info in (summary.get("models") or {}).items()]
        if model_rows:
            lines += md_table(["table", "default model", "trained models"], model_rows) + [""]
        if summary.get("highlights"):
            lines += ["**Generation highlights**", ""] + [f"- {h}" for h in summary["highlights"]] + [""]
        steps = summary.get("steps") or []
        rows = []
        for step in steps:
            checks = step.get("checks") or []
            passed = sum(1 for c in checks if c.get("passed"))
            rows.append([step.get("index"), step.get("name"), f"`{step.get('method')} {step.get('path')}`",
                         step.get("status_code") if step.get("status_code") is not None else "-",
                         fmt_num(step.get("elapsed_ms"), 1), f"{passed}/{len(checks)}" if checks else "-",
                         step.get("outcome")])
        lines += md_table(["#", "step", "request", "status", "ms", "checks", "outcome"], rows) + [""]
        figure = _latency_figure(steps, figure_dir / "week4_api_latency.png")
        lines += figure_line(figure, report_dir, "Response times of the API demo calls")

    lines += _coverage_section(summary)
    lines += ["## 4. Models behind the API", ""]
    if comparison:
        rows = []
        for table in schema.TABLES:
            info = (comparison.get("tables") or {}).get(table) or {}
            for entry in info.get("models") or []:
                rows.append([table, entry.get("model_id") + (" (default)" if entry.get("model_id") == info.get("best_model") else ""),
                             fmt_num(entry.get("composite")),
                             *[fmt_num((entry.get("components") or {}).get(c))
                               for c in ("fidelity", "validity", "operational", "utility", "privacy")]])
        lines += md_table(["table", "model", "composite", "fidelity", "validity", "operational", "utility", "privacy"], rows)
        lines += ["", f"Details: {project_ref('reports/week3_evaluation.md', report_dir)}.", ""]
    else:
        lines += ["_No model comparison yet: run `python scripts/03_evaluate.py`._", ""]

    lines += _testing_section(junit, slow)
    api = get_settings().get("api") or {}
    budget, max_budget = api.get("time_budget_seconds"), api.get("max_time_budget_seconds")
    combinations, pending = api.get("max_condition_combinations"), api.get("max_pending_generations")
    lines += [
        "## 6. Guarantees and limitations",
        "",
        "- Every generated record passes all hard operational rules (rejection sampling); the response reports the "
        "acceptance rate, rejected rows per rule, the seed and any shortfall. A request no row can satisfy returns "
        "422 `no_rows_generated` and stores nothing.",
        "- The same request with the same seed returns identical records (seeded SDV sampling) when it finishes "
        "within its time budget (`time_budget_seconds`"
        + (f", default {float(budget):g} s" if is_number(budget) else "")
        + (f", at most {float(max_budget):g} s" if is_number(max_budget) else "") + ").",
        "- Requests are bounded: at most "
        + (f"{fmt_int(combinations)} condition combinations" if is_number(combinations)
           else "`api.max_condition_combinations` condition combinations")
        + " (400 `too_many_combinations`), a per-request time budget, and at most "
        + (f"{fmt_int(pending)} generations" if is_number(pending) else "`api.max_pending_generations` generations")
        + " running or queued (503 `server_busy` with `Retry-After` when full).",
        "- `/validate` recomputes every derived column: provided derived values that contradict their definition "
        "make the row hard-invalid (`WD-D01` / `PS-D01`).",
        "- CORS admits pages served from this machine only (`settings.api.cors_origin_regex`); the API has no "
        "authentication and is meant for local or trusted networks.",
        "- Calendar-mode rows are conditionally independent draws: day-to-day census continuity is not modelled.",
        "- Tables are synthesised independently (no linkage of stays to ward-days); the reference data is simulated.",
        "- `/docs` and `/redoc` load Swagger UI / ReDoc assets from a CDN; `/ui` works fully offline.",
        "",
    ]
    return write_markdown(report_dir / REPORT_NAME, lines)


__all__ = ["JUNIT_FILE", "REPORT_NAME", "load_junit_summary", "render_week4_report"]
