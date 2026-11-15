"""Service layer of the BedSynth AI API.

:class:`BedSynthService` owns every interaction between the HTTP layer and the BedSynth
engine; :mod:`bedsynth.api.main` only maps HTTP requests to service calls and service errors
to JSON error responses. The service reaches the engine exclusively through public contracts:

* :mod:`bedsynth.synthesis.generator` - ``SyntheticDataGenerator`` / ``GenerationRequest`` and
  the ``GenerationError`` / ``ModelNotFoundError`` exceptions,
* :class:`bedsynth.synthesis.registry.ModelRegistry` - default-model reasons and model cards,
* :mod:`bedsynth.validation` - ``validate_table`` and ``rules_catalog``,
* :func:`bedsynth.evaluation.quick_quality`,
* :mod:`bedsynth.schema` and :mod:`bedsynth.scenarios` - catalogues, derivations and masks.

Engine modules are imported lazily because they pull in SDV and torch (around ten seconds on a
cold start): importing the API is fast and never requires trained models. ``GET /health``
never creates the engine or waits for it: until the engine is loaded it reads the trained
models straight from ``registry.json`` (without importing SDV) and reports the engine state
(``not_loaded``, ``loading``, ``ready`` or ``failed``) as it is at the time of the call.
Everything the service returns is JSON-safe (numpy scalars become Python numbers, NaN becomes
``None``), and error messages sent to clients never contain exception text or server paths
(the full traceback goes to the server log).

Generated datasets are persisted by :class:`DatasetStore` as
``<synthetic_dir>/api/<dataset_id>/{data.csv, metadata.json}``; the CSV uses LF line endings,
like the CSV body of ``/generate``.

Robustness limits (``settings.api``): requests may expand into at most
``max_condition_combinations`` condition cells, every generation runs within a time budget
(``time_budget_seconds``, at most ``max_time_budget_seconds``), generations run one at a time
(SDV sampling swaps process-wide RNG state) behind a bounded queue (``max_pending_generations``,
``generation_wait_seconds``; overflow answers 503 ``server_busy``), and requests no row can
satisfy are rejected before (impossible filters, 400) or after sampling (422 ``no_rows_generated``).
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import math
import os
import re
import secrets
import shutil
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from importlib import metadata as importlib_metadata
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

import numpy as np
import pandas as pd

from bedsynth import __version__, schema, validation
from bedsynth.api.schemas import (
    MIN_EVALUATE_ROWS,
    MODEL_ID_PATTERN,
    EvaluateRequest,
    GenerateRequest,
    ValidateRequest,
    canonical_category,
    months_in_period,
)
from bedsynth.config import get_path, get_settings
from bedsynth.data import io
from bedsynth.scenarios import (
    SCENARIO_CATEGORIES,
    SCENARIOS,
    Scenario,
    condition_mask,
    filter_mask,
    merge_conditions,
    merge_filters,
    scenario_mask,
    scenarios_for,
    validate_conditions,
    validate_filters,
)

logger = logging.getLogger(__name__)

API_PREFIX = "/api/v1"
DATASET_ID_PATTERN = re.compile(r"^ds-[0-9a-f]{12}$")
MIN_REFERENCE_SUBSET_ROWS = 20          # same threshold as the scenario-coverage evaluation
MAX_INVALID_ROWS_LISTED = 100
FLOAT_MATCH_TOLERANCE = 0.006           # provided derived ratios may be rounded to 2 decimals

TRAIN_HINT = (
    "Train models with `python scripts/02_train_synthesizers.py` "
    "(or `python scripts/run_pipeline.py --quick`); the running API picks them up automatically."
)
PREPARE_HINT = "Prepare the reference data with `python scripts/01_prepare_data.py`."

#: Engine states reported by ``GET /health`` (``engine.state``).
ENGINE_NOT_LOADED, ENGINE_LOADING, ENGINE_READY, ENGINE_FAILED = "not_loaded", "loading", "ready", "failed"
ENGINE_LOADING_DETAIL = (
    "Engine loading: the generation engine (SDV) is starting in the background; "
    "requests that need it wait until it is ready."
)
ENGINE_NOT_LOADED_DETAIL = "Engine not loaded yet: it loads on the first request that needs it."

#: Registry layout of :mod:`bedsynth.synthesis.registry` (``<models_dir>/registry.json`` and
#: ``<models_dir>/<table>/<model_id>/synthesizer.pkl``). Repeated here because importing that
#: module imports SDV, and ``/health`` must answer while the engine is still loading.
REGISTRY_FILE = "registry.json"
MODEL_FILE = "synthesizer.pkl"
_MODEL_ID_RE = re.compile(MODEL_ID_PATTERN)

#: Response headers attached to every CSV download (also exposed through CORS).
CSV_HEADER_NAMES: tuple[str, ...] = (
    "X-BedSynth-Dataset-Id",
    "X-BedSynth-Table",
    "X-BedSynth-Rows",
    "X-BedSynth-Model",
    "X-BedSynth-Scenario",
    "X-BedSynth-Seed",
    "X-BedSynth-Persisted",
    "X-BedSynth-Hard-Rules-Passed",
    "X-BedSynth-Quality-Score",
)

TABLE_INFO: dict[str, dict[str, str]] = {
    schema.WARD_DAILY: {
        "title": "Ward-day operations",
        "grain": "One row per ward per calendar day",
        "description": (
            "Daily capacity, nurse staffing, patient flow, emergency demand, elective scheduling, "
            "bed turnover and midnight occupancy for each of the eight wards of a 194-bed hospital."
        ),
    },
    schema.PATIENT_STAYS: {
        "title": "Patient ward stays",
        "grain": "One row per inpatient ward stay (bed episode)",
        "description": (
            "Admission and discharge timing, elective booking and rescheduling, bed waits, length of "
            "stay, discharge delays and bed turnaround. Strictly non-clinical: no diagnoses, "
            "procedures or personal identifiers."
        ),
    },
}


# ---------------------------------------------------------------------------
# Errors (mapped to HTTP responses by bedsynth.api.main)
# ---------------------------------------------------------------------------
class ServiceError(Exception):
    """Base error rendered by the API as ``{"detail": ..., "error_type": ...}``."""

    status_code: int = 500
    error_type: str = "internal_error"

    def __init__(
        self, detail: str, *, error_type: str | None = None, headers: Mapping[str, str] | None = None
    ) -> None:
        super().__init__(detail)
        self.detail = detail
        self.headers: dict[str, str] = dict(headers or {})
        if error_type is not None:
            self.error_type = error_type


class BadRequestError(ServiceError):
    """The request is well-formed but cannot be served as asked (HTTP 400)."""

    status_code = 400
    error_type = "invalid_request"


class NotFoundError(ServiceError):
    """An unknown table, scenario, model or dataset was referenced (HTTP 404)."""

    status_code = 404
    error_type = "not_found"


class DatasetNotFoundError(NotFoundError):
    """No persisted dataset with this id exists (or the id is malformed)."""

    error_type = "dataset_not_found"


class ServiceUnavailableError(ServiceError):
    """A dependency is not ready: no trained model, no reference data, engine missing (HTTP 503)."""

    status_code = 503
    error_type = "service_unavailable"


class ServiceBusyError(ServiceUnavailableError):
    """Too many generations are running or queued; the client should retry later (HTTP 503)."""

    error_type = "server_busy"


class UnprocessableError(ServiceError):
    """The request is valid but cannot be satisfied, e.g. no row meets its filters (HTTP 422)."""

    status_code = 422
    error_type = "unprocessable_request"


class ConflictError(ServiceError):
    """The resource is in use and cannot be changed right now (HTTP 409)."""

    status_code = 409
    error_type = "conflict"


def _engine_not_implemented() -> ServiceUnavailableError:
    return ServiceUnavailableError(
        "The generation engine (bedsynth.synthesis.generator) is not available in this build.",
        error_type="engine_unavailable",
    )


def _message(exc: BaseException) -> str:
    """Readable exception message (KeyError's str() adds quotes)."""
    if isinstance(exc, KeyError) and exc.args:
        return str(exc.args[0])
    return str(exc) or type(exc).__name__


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------
def to_jsonable(value: Any) -> Any:
    """Recursively convert ``value`` into JSON-serialisable Python objects.

    numpy scalars/arrays become ``int``/``float``/``bool``/lists, NaN/inf/NA/NaT become ``None``,
    timestamps and dates become ISO strings, enums their value, paths strings, mappings plain
    dicts with string keys, and other iterables lists. Unknown objects fall back to ``str``.
    """
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, (str, bool)):
        return value
    if isinstance(value, (int, np.integer)) and not isinstance(value, np.bool_):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (pd.Timestamp, dt.datetime, dt.date)):
        return value.isoformat()
    if isinstance(value, np.datetime64):
        return None if np.isnat(value) else pd.Timestamp(value).isoformat()
    if isinstance(value, Enum):
        return to_jsonable(value.value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.DataFrame):
        return frame_to_records(value)
    if isinstance(value, (pd.Series, pd.Index, np.ndarray)):
        return [to_jsonable(item) for item in value.tolist()]
    if isinstance(value, (set, frozenset)):
        return [to_jsonable(item) for item in sorted(value, key=str)]
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return str(value)


def _jsonable_scalar(value: Any) -> Any:
    """Fast path of :func:`to_jsonable` for the cells of a record."""
    if value is None or isinstance(value, (str, bool)):
        return value
    if type(value) is int:
        return value
    if type(value) is float:
        return value if math.isfinite(value) else None
    return to_jsonable(value)


def frame_to_records(frame: pd.DataFrame | None) -> list[dict[str, Any]]:
    """Convert a DataFrame into JSON-safe records (NaN/NA/NaT -> ``None``, numpy -> Python)."""
    if frame is None or frame.empty:
        return []
    objects = frame.astype(object).where(frame.notna(), None)
    columns = [str(column) for column in objects.columns]
    return [
        {column: _jsonable_scalar(cell) for column, cell in zip(columns, row)}
        for row in objects.itertuples(index=False, name=None)
    ]


def utc_now_iso() -> str:
    """Current UTC time as an ISO 8601 string (second precision)."""
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def package_version(name: str) -> str | None:
    """Installed version of a distribution without importing it (None if not installed)."""
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return None


def _dedupe(items: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(str(item) for item in items if item))


MAX_RESPONSE_WARNINGS = 25            # warnings kept per response (the rest are summarised)
MAX_ALLOCATION_ENTRIES = 200          # condition_allocation entries kept in a stored report


def _cap_warnings(warnings: Sequence[str], limit: int = MAX_RESPONSE_WARNINGS) -> list[str]:
    """At most ``limit`` warnings plus one line counting the omitted ones."""
    warnings = list(warnings)
    if len(warnings) <= limit:
        return warnings
    return warnings[:limit] + [f"... {len(warnings) - limit} more warnings omitted."]


def _compact_report(report: dict[str, Any]) -> dict[str, Any]:
    """Bound the size of an engine report (deduplicated, capped warnings and allocation list)."""
    if isinstance(report.get("warnings"), list):
        report["warnings"] = _cap_warnings(_dedupe(report["warnings"]))
    allocation = report.get("condition_allocation")
    if isinstance(allocation, list) and len(allocation) > MAX_ALLOCATION_ENTRIES:
        report["condition_allocation"] = allocation[:MAX_ALLOCATION_ENTRIES]
        report["condition_allocation_total"] = len(allocation)
    return report


def _format_counts(counts: Mapping[str, int]) -> str:
    return ", ".join(f"{column} ({count})" for column, count in counts.items())


# ---------------------------------------------------------------------------
# Lazy access to the engine (heavy imports)
# ---------------------------------------------------------------------------
def _engine_module() -> ModuleType:
    """Import :mod:`bedsynth.synthesis.generator` on first use (it imports SDV/torch)."""
    from bedsynth.synthesis import generator

    return generator


def _default_generator_factory(
    models_dir: Path | str | None, processed_dir: Path | str | None
) -> Callable[[], Any]:
    def factory() -> Any:
        engine = _engine_module()
        return engine.SyntheticDataGenerator(models_dir=models_dir, processed_dir=processed_dir)

    return factory


def run_quick_quality(real: pd.DataFrame, synthetic: pd.DataFrame, table: str) -> dict[str, Any]:
    """Call :func:`bedsynth.evaluation.quick_quality` (imported lazily: heavy evaluation stack)."""
    from bedsynth import evaluation

    return evaluation.quick_quality(real, synthetic, table)


_PATH_STOP = r"\s'\"`,;()<>|"
#: Continuation of a path over spaces inside a component (``C:\Users\John Smith\x.csv``): words that are
#: followed by a separator. A continuation word never starts with ``~`` and never contains ``:``, so a
#: second path after "and" / "or" is not merged into the first.
_SPACED_COMPONENTS = rf"(?:(?: [^{_PATH_STOP}\\/:~][^{_PATH_STOP}\\/:]*)+[\\/][^{_PATH_STOP}]*)*"
_WINDOWS_PATH = re.compile(rf"(?:[A-Za-z]:[\\/]|\\\\)[^{_PATH_STOP}]+" + _SPACED_COMPONENTS)
#: POSIX locations, including MSYS / Git Bash drive paths (``/c/Users/...``).
_POSIX_PATH = re.compile(
    rf"(?<![\w.])/(?:[A-Za-z]|home|Users|tmp|var|opt|usr|mnt|srv|root|data|app|private)/[^{_PATH_STOP}]*"
    + _SPACED_COMPONENTS
)
#: Home-relative paths (``~/...``), the form bedsynth.config.project_relative stores outside the project.
_HOME_PATH = re.compile(rf"(?<![\w.~])~[\\/][^{_PATH_STOP}]*" + _SPACED_COMPONENTS)


def scrub_paths(text: str) -> str:
    """Replace server file-system paths (absolute or ``~/...``) in a message by ``<server path>``."""
    for pattern in (_WINDOWS_PATH, _HOME_PATH, _POSIX_PATH):
        text = pattern.sub("<server path>", text)
    return text


def _model_unavailable(table: str | None, model_id: str | None) -> ServiceUnavailableError:
    """Fixed 503 message for a missing model (registry messages contain server paths)."""
    where = f" for table '{table}'" if table else ""
    if model_id:
        detail = f"Model '{model_id}'{where} is not trained or its files are missing."
    else:
        detail = f"No trained model is available{where}."
    return ServiceUnavailableError(f"{detail} {TRAIN_HINT}", error_type="model_unavailable")


@contextmanager
def map_engine_errors(engine: ModuleType, table: str | None = None, model_id: str | None = None) -> Iterator[None]:
    """Translate generation-engine exceptions into :class:`ServiceError` subclasses.

    Messages that reach the client never contain server file-system paths.
    """
    try:
        yield
    except ServiceError:
        raise
    except engine.ModelNotFoundError as exc:
        logger.info("Model unavailable: %s", _message(exc))
        raise _model_unavailable(table, model_id) from exc
    except engine.GenerationError as exc:
        raise BadRequestError(scrub_paths(_message(exc)), error_type="generation_error") from exc
    except NotImplementedError as exc:
        raise _engine_not_implemented() from exc
    except (ValueError, KeyError) as exc:
        raise BadRequestError(scrub_paths(_message(exc)), error_type="invalid_request") from exc
    except FileNotFoundError as exc:
        logger.exception("A file required by the generation engine is missing")
        raise ServiceUnavailableError(
            f"A model file required by the generation engine is missing. {TRAIN_HINT}",
            error_type="model_unavailable",
        ) from exc
    except Exception as exc:
        logger.exception("Unexpected generation failure")
        raise ServiceError(
            f"Generation failed unexpectedly ({type(exc).__name__}); see the server log.",
            error_type="internal_error",
        ) from exc


# ---------------------------------------------------------------------------
# Model cards
# ---------------------------------------------------------------------------
_NESTED_CARD_KEYS: tuple[str, ...] = ("experiment", "config", "experiment_config")
#: Card fields never exposed through the API, at any nesting level (server file-system locations).
_PRIVATE_CARD_KEYS: frozenset[str] = frozenset({"path", "model_path", "models_dir", "processed_dir", "directory"})
_ABSOLUTE_PATH_VALUE = re.compile(r"^(?:[A-Za-z]:[\\/]|\\\\|/[^/\s]+/|~[\\/])")


def redact_paths(value: Any) -> Any:
    """Copy of a model card without server paths.

    Keys of :data:`_PRIVATE_CARD_KEYS` are dropped at every level and any other string that is an
    absolute or home-relative (``~/...``) file-system path (e.g. ``card.training.source``) is
    reduced to its file name. Project-relative values (``data/processed/...``) are kept.
    """
    if isinstance(value, Mapping):
        return {key: redact_paths(item) for key, item in value.items() if str(key) not in _PRIVATE_CARD_KEYS}
    if isinstance(value, (list, tuple)):
        return [redact_paths(item) for item in value]
    if isinstance(value, str) and _ABSOLUTE_PATH_VALUE.match(value):
        return re.split(r"[\\/]", value.rstrip("\\/"))[-1]
    return value


def _card_value(card: Mapping[str, Any], keys: Sequence[str], kinds: tuple[type, ...]) -> Any:
    """First value of ``keys`` (of an accepted type) in a model card or its nested config."""
    containers = [card, *(card.get(key) for key in _NESTED_CARD_KEYS)]
    for container in containers:
        if not isinstance(container, Mapping):
            continue
        for key in keys:
            value = container.get(key)
            if isinstance(value, kinds) and not isinstance(value, bool) and value != "":
                return value
    return None


def _segment_by(card: Mapping[str, Any], table: str) -> str | list[str] | None:
    """Segment column(s) of a model card: a column name, a list of columns or ``None`` (global).

    The resolved top-level ``segment_by`` wins; an experiment config nested in the card may hold
    the per-table settings mapping (``{table: column(s)}``), which is resolved for ``table``.
    """
    for container in (card, *(card.get(key) for key in _NESTED_CARD_KEYS)):
        if not isinstance(container, Mapping) or "segment_by" not in container:
            continue
        value = container.get("segment_by")
        if isinstance(value, Mapping):
            value = value.get(table)
        if isinstance(value, str) and value:
            return value
        if isinstance(value, (list, tuple)) and value and all(isinstance(item, str) for item in value):
            return [str(item) for item in value]
        return None
    return None


def _string_list(value: Any) -> list[str] | None:
    """``value`` as a list of strings (``None`` unless it is a list/tuple of strings)."""
    if isinstance(value, (list, tuple)) and all(isinstance(item, str) for item in value):
        return [str(item) for item in value]
    return None


def model_id_of(card: Mapping[str, Any]) -> str | None:
    """Experiment id of a model card / registry entry."""
    value = _card_value(card, ("experiment_id", "model_id", "id"), (str,))
    return str(value) if value is not None else None


def model_info(card: Mapping[str, Any], table: str, default_id: str | None) -> dict[str, Any]:
    """Normalise a model card (whatever its exact layout) into the ``ModelInfo`` shape.

    ``metrics`` is the card's ``metrics`` mapping as written by ``scripts/03_evaluate.py``
    (composite, components, rank, ...), passed through as stored; ``None`` when the card has none.
    """
    model_id = model_id_of(card) or "unknown"
    training_rows = _card_value(card, ("training_rows", "n_training_rows", "train_rows", "rows"), (int, float))
    fit_seconds = _card_value(card, ("fit_seconds", "fit_time_seconds", "training_seconds"), (int, float))
    metrics = to_jsonable(card["metrics"]) if isinstance(card.get("metrics"), Mapping) else None
    table = str(card.get("table") or table)
    segment_by = _segment_by(card, table)
    segment_label = _card_value(card, ("segment_label",), (str,))
    if segment_label is None and segment_by is not None:
        segment_label = segment_by if isinstance(segment_by, str) else " x ".join(segment_by)
    return {
        "model_id": model_id,
        "table": table,
        "synthesizer": _card_value(card, ("synthesizer", "model_type", "model", "synthesizer_class"), (str,)),
        "description": _card_value(card, ("description",), (str,)),
        "sdv_version": _card_value(card, ("sdv_version",), (str,)),
        "fitted_at": _card_value(card, ("fitted_at",), (str,)),
        "training_rows": int(training_rows) if training_rows is not None else None,
        "fit_seconds": float(fit_seconds) if fit_seconds is not None else None,
        "segment_by": segment_by,
        "segment_label": segment_label,
        "constraint_groups": _string_list(card.get("constraint_groups")),
        "effective_constraint_groups": _string_list(card.get("effective_constraint_groups")),
        "is_default": model_id == default_id,
        "metrics": redact_paths(metrics),
        "card": redact_paths(to_jsonable(dict(card))),
    }


def model_summary(card: Mapping[str, Any] | None, model_id: str | None) -> dict[str, Any]:
    """``{id, synthesizer, sdv_version, fitted_at}`` of the model that produced a dataset."""
    info = model_info(card, "", None) if card else {}
    return {
        "id": model_id or info.get("model_id"),
        "synthesizer": info.get("synthesizer"),
        "sdv_version": info.get("sdv_version"),
        "fitted_at": info.get("fitted_at"),
    }


# ---------------------------------------------------------------------------
# Validation summaries
# ---------------------------------------------------------------------------
def _int_list(values: Iterable[Any] | None, limit: int = 5) -> list[int]:
    out: list[int] = []
    for value in values or []:
        try:
            out.append(int(value))
        except (TypeError, ValueError):
            continue
        if len(out) >= limit:
            break
    return out


def _rule_result(item: Mapping[str, Any]) -> dict[str, Any]:
    violations = int(item.get("violations") or 0)
    rate = item.get("violation_rate")
    rate = float(rate) if isinstance(rate, (int, float, np.number)) and math.isfinite(float(rate)) else 0.0
    return {
        "rule_id": str(item.get("rule_id", "")),
        "name": str(item.get("name", "")),
        "severity": str(item.get("severity", "hard")),
        "description": str(item.get("description", "")),
        "columns": [str(column) for column in item.get("columns") or []],
        "violations": violations,
        "violation_rate": round(rate, 6),
        "passed": violations == 0,
        "example_rows": _int_list(item.get("example_indices")),
    }


def _rule_group(rules: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    failed = sum(1 for rule in rules if not rule["passed"])
    return {
        "checked": len(rules),
        "passed": len(rules) - failed,
        "failed": failed,
        "violations": sum(int(rule["violations"]) for rule in rules),
    }


def hard_invalid_positions(report: Any, n_rows: int, extra_invalid: np.ndarray | None = None) -> np.ndarray:
    """0-based positions of rows failing a hard rule (or flagged by ``extra_invalid``)."""
    mask = getattr(report, "hard_valid_mask", None)
    invalid = np.zeros(n_rows, dtype=bool)
    if isinstance(mask, pd.Series) and len(mask) == n_rows:
        invalid |= ~mask.fillna(False).astype(bool).to_numpy()
    if extra_invalid is not None and len(extra_invalid) == n_rows:
        invalid |= np.asarray(extra_invalid, dtype=bool)
    return np.flatnonzero(invalid)


def summarise_validation(
    report: Any,
    extra_rules: Sequence[Mapping[str, Any]] = (),
    extra_invalid: np.ndarray | None = None,
) -> dict[str, Any]:
    """JSON summary of a :class:`bedsynth.validation.ValidationReport`.

    Built from the report's documented attributes (``n_rows``, ``hard_valid_mask``,
    ``rule_results``) so it does not depend on ``to_dict``'s layout. ``extra_rules`` are
    API-level checks in :func:`_rule_result` shape (e.g. the derived-value consistency of client
    records) and ``extra_invalid`` flags the rows they reject. ``all_hard_rules_passed`` is
    ``None`` when there is no row to judge.
    """
    rules = [_rule_result(item) for item in getattr(report, "rule_results", None) or []]
    hard_rules = [rule for rule in rules if rule["severity"] == "hard"] + [dict(rule) for rule in extra_rules]
    soft_rules = [rule for rule in rules if rule["severity"] != "hard"]
    n_rows = int(getattr(report, "n_rows", 0) or 0)
    if isinstance(getattr(report, "hard_valid_mask", None), pd.Series) or extra_invalid is not None:
        n_valid = n_rows - len(hard_invalid_positions(report, n_rows, extra_invalid))
    else:
        n_valid = int(getattr(report, "n_hard_valid", n_rows))
    return {
        "n_rows": n_rows,
        "n_hard_valid": n_valid,
        "hard_valid_rate": round(n_valid / n_rows, 6) if n_rows else None,
        "all_hard_rules_passed": (n_valid == n_rows) if n_rows else None,
        "hard_rules": _rule_group(hard_rules),
        "soft_rules": _rule_group(soft_rules),
        "rules": hard_rules + soft_rules,
    }


def describe_columns(table: str, columns: Iterable[Any]) -> list[dict[str, Any]]:
    """Schema description ``{name, dtype, role, unit, description}`` of output columns."""
    specs = {spec.name: spec for spec in schema.column_specs(table)}
    described: list[dict[str, Any]] = []
    for name in columns:
        spec = specs.get(str(name))
        if spec is None:
            described.append({"name": str(name), "dtype": "unknown", "role": "extra", "unit": None, "description": ""})
        else:
            described.append(
                {"name": spec.name, "dtype": spec.dtype, "role": spec.role, "unit": spec.unit, "description": spec.description}
            )
    return described


def derived_view(table: str, frame: pd.DataFrame) -> pd.DataFrame:
    """``frame`` with every derived column present (derived when missing), index reset."""
    frame = frame.reset_index(drop=True)
    if all(column in frame.columns for column in schema.derived_columns(table)):
        return frame
    if not all(column in frame.columns for column in schema.training_columns(table)):
        return frame
    return schema.derive(table, frame)


def subsample(frame: pd.DataFrame, max_rows: int | None, seed: int = 0) -> pd.DataFrame:
    """Seeded random subsample of at most ``max_rows`` rows (the frame itself when smaller)."""
    if max_rows is None or len(frame) <= max_rows:
        return frame
    return frame.sample(n=int(max_rows), random_state=int(seed) % (2**32))


def dataset_headers(metadata: Mapping[str, Any]) -> dict[str, str]:
    """``X-BedSynth-*`` response headers describing a (generated or stored) dataset."""
    generation = metadata.get("generation") or {}
    validation_block = metadata.get("validation") or {}
    seed = generation.get("seed")
    headers = {
        "X-BedSynth-Dataset-Id": str(metadata.get("dataset_id") or ""),
        "X-BedSynth-Table": str(metadata.get("table") or ""),
        "X-BedSynth-Rows": str(metadata.get("n_rows") or 0),
        "X-BedSynth-Model": str((metadata.get("model") or {}).get("id") or ""),
        "X-BedSynth-Scenario": str((metadata.get("scenario") or {}).get("name") or ""),
        "X-BedSynth-Seed": "" if seed is None else str(seed),
        "X-BedSynth-Persisted": "true" if metadata.get("persisted") else "false",
    }
    if validation_block:
        passed = validation_block.get("all_hard_rules_passed")
        headers["X-BedSynth-Hard-Rules-Passed"] = "unknown" if passed is None else ("true" if passed else "false")
    score = (metadata.get("quality") or {}).get("score")
    if isinstance(score, (int, float)) and not isinstance(score, bool):
        headers["X-BedSynth-Quality-Score"] = f"{float(score):.4f}"
    return headers


# ---------------------------------------------------------------------------
# Client records (validate / evaluate)
# ---------------------------------------------------------------------------
_TRUE_STRINGS = frozenset({"true", "1", "yes", "y", "t"})
_FALSE_STRINGS = frozenset({"false", "0", "no", "n", "f"})


def _to_bool(value: Any) -> bool | None:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value is None or value is pd.NA or (isinstance(value, float) and math.isnan(value)):
        return None
    text = str(value).strip().lower()
    if text in _TRUE_STRINGS:
        return True
    if text in _FALSE_STRINGS:
        return False
    return None


#: API-level hard check of client records: provided derived values must match their definitions.
DERIVED_CHECK_IDS: dict[str, str] = {schema.WARD_DAILY: "WD-D01", schema.PATIENT_STAYS: "PS-D01"}
DERIVED_CHECK_NAME = "derived_values_consistent"


@dataclass
class PreparedRecords:
    """Client records ready for validation/evaluation, plus what was changed on the way.

    ``frame`` holds the (normalised) training columns with every derived column *recomputed*
    from them - the rules and the quality score always see definitional values. ``records`` is
    what the client gets back: provided derived values are kept as given and missing ones are
    filled in. ``derived_mismatch`` flags the rows whose provided derived values disagree with
    their definitions; :meth:`derived_check` turns it into a hard rule result.
    """

    frame: pd.DataFrame
    normalisation: dict[str, Any]
    warnings: list[str] = field(default_factory=list)
    records: pd.DataFrame | None = None
    derived_mismatch: np.ndarray | None = None
    table: str | None = None

    @property
    def output(self) -> pd.DataFrame:
        """Records as returned to the client (provided derived values kept)."""
        return self.records if self.records is not None else self.frame

    def derived_check(self) -> dict[str, Any]:
        """The ``derived_values_consistent`` hard check in the shape of a rule result."""
        n_rows = len(self.frame)
        mismatch = self.derived_mismatch if self.derived_mismatch is not None else np.zeros(n_rows, dtype=bool)
        positions = np.flatnonzero(mismatch)
        table = self.table or schema.WARD_DAILY
        computed = set(self.normalisation.get("derived_computed") or [])
        provided = [column for column in schema.derived_columns(table) if column not in computed]
        return {
            "rule_id": DERIVED_CHECK_IDS.get(table, "API-D01"),
            "name": DERIVED_CHECK_NAME,
            "severity": "hard",
            "description": (
                "Provided derived values match their schema definitions (flow balance, available beds, "
                "occupancy rate, calendar labels, length of stay, ...). API check on client records."
            ),
            "columns": provided,
            "violations": int(positions.size),
            "violation_rate": round(positions.size / n_rows, 6) if n_rows else 0.0,
            "passed": positions.size == 0,
            "example_rows": [int(position) for position in positions[:5]],
        }


def _normalise_categories(frame: pd.DataFrame, table: str) -> tuple[dict[str, int], dict[str, int]]:
    """Map category aliases to canonical codes in place.

    Unknown values (booleans included: ``true`` is never month 1) are kept as provided so they
    fail ``valid_categories``; returns the counts of normalised and unknown values per column.
    """
    normalised: dict[str, int] = {}
    unknown: dict[str, int] = {}
    for spec in schema.column_specs(table, roles=("train",)):
        if spec.sdtype not in ("categorical", "ordinal") or spec.name not in frame.columns:
            continue
        original = frame[spec.name]
        canonical = original.map(lambda value, c=spec.name, cats=spec.categories: canonical_category(c, value, cats))
        present = original.notna()
        changed = present & canonical.notna() & (original.astype(str) != canonical.astype(str))
        invalid = present & canonical.isna()
        frame[spec.name] = canonical.where(canonical.notna(), original)
        if changed.any():
            normalised[spec.name] = int(changed.sum())
        if invalid.any():
            unknown[spec.name] = int(invalid.sum())
    return normalised, unknown


def _is_bool(value: Any) -> bool:
    return isinstance(value, (bool, np.bool_))


def _coerce_numeric_columns(frame: pd.DataFrame, table: str) -> dict[str, int]:
    """Coerce numerical columns in place (no rounding, so non-integral counts still fail rules).

    Booleans are not numbers here (``true`` would otherwise count as 1 patient): like any other
    non-numeric value they become missing and are reported in ``invalid_numbers``.
    """
    invalid: dict[str, int] = {}
    for spec in schema.column_specs(table, roles=("train", "derived")):
        if spec.sdtype != "numerical" or spec.name not in frame.columns:
            continue
        original = frame[spec.name]
        booleans = original.map(_is_bool).astype(bool)
        numeric = pd.to_numeric(original.astype(object).where(~booleans, None), errors="coerce").astype(float)
        bad = original.notna() & numeric.isna()
        if bad.any():
            invalid[spec.name] = int(bad.sum())
        if spec.dtype == "int" and numeric.notna().all() and bool((numeric % 1 == 0).all()):
            numeric = numeric.astype("int64")
        frame[spec.name] = numeric
    for name in schema.boolean_columns(table, roles=("derived",)):
        if name in frame.columns:
            frame[name] = frame[name].map(_to_bool)
    return invalid


def _differs(provided: pd.Series, expected: pd.Series, spec: schema.ColumnSpec) -> pd.Series:
    """Rows where a provided derived value disagrees with its definition (both present)."""
    both = provided.notna() & expected.notna()
    if spec.dtype in ("int", "float"):
        tolerance = FLOAT_MATCH_TOLERANCE if spec.dtype == "float" else 1e-9
        gap = (pd.to_numeric(provided, errors="coerce") - pd.to_numeric(expected, errors="coerce")).abs()
        return (both & (gap > tolerance)).fillna(False).astype(bool)
    if spec.dtype == "bool":
        return (both & (provided.map(_to_bool) != expected.map(_to_bool))).astype(bool)
    return (both & (provided.astype(str) != expected.astype(str))).astype(bool)


@dataclass
class _DerivedCheck:
    """Outcome of comparing provided derived values with their recomputed definitions."""

    validation_frame: pd.DataFrame
    returned_frame: pd.DataFrame
    computed: list[str]
    mismatches: dict[str, int]
    mismatch_rows: dict[str, list[int]]
    row_mismatch: np.ndarray


def _check_derived(frame: pd.DataFrame, table: str) -> _DerivedCheck:
    """Recompute every derived column and compare it with the values the client provided.

    The validation frame keeps the client's training columns but takes every derived column
    from :func:`bedsynth.schema.derive` (``derive`` also normalises one training column,
    ``avg_bed_wait_hours``; that change is deliberately *not* copied, so WD-H14 can still fail).
    The returned frame keeps provided derived values and fills missing ones.
    """
    try:
        expected_frame = schema.derive(table, frame)
    except Exception as exc:  # derive() is strict about dtypes; report instead of failing with 500
        logger.warning("Could not derive the columns of client %s records", table, exc_info=True)
        raise BadRequestError(
            f"Could not compute the derived columns of these records ({type(exc).__name__}); check the "
            f"column types against GET {API_PREFIX}/tables/{table}/schema.",
            error_type="invalid_records",
        ) from exc
    validation_frame = frame.copy()
    returned = frame.copy()
    computed: list[str] = []
    mismatches: dict[str, int] = {}
    mismatch_rows: dict[str, list[int]] = {}
    row_mismatch = np.zeros(len(frame), dtype=bool)
    for spec in schema.column_specs(table, roles=("derived",)):
        expected = expected_frame[spec.name]
        validation_frame[spec.name] = expected
        if spec.name not in frame.columns:
            returned[spec.name] = expected
            computed.append(spec.name)
            continue
        provided = frame[spec.name]
        differs = _differs(provided, expected, spec).to_numpy(dtype=bool)
        if differs.any():
            positions = np.flatnonzero(differs)
            mismatches[spec.name] = int(positions.size)
            mismatch_rows[spec.name] = [int(position) for position in positions[:MAX_INVALID_ROWS_LISTED]]
            row_mismatch |= differs
        returned[spec.name] = provided.where(provided.notna(), expected)
    return _DerivedCheck(validation_frame, returned, computed, mismatches, mismatch_rows, row_mismatch)


def prepare_records(table: str, records: Sequence[Mapping[str, Any]]) -> PreparedRecords:
    """Turn client records into a fully derived frame ready for the validation rules.

    Steps: check every training column is present, normalise category aliases, coerce numeric
    and boolean columns, recompute every derived column (flagging provided values that disagree
    with their definitions) and restore the canonical column order.
    """
    frame = pd.DataFrame.from_records([dict(record) for record in records])
    missing = [column for column in schema.training_columns(table) if column not in frame.columns]
    if missing:
        raise BadRequestError(
            f"Records for '{table}' are missing required training columns: {', '.join(missing)}. "
            f"See GET {API_PREFIX}/tables/{table}/schema.",
            error_type="invalid_records",
        )
    known = set(schema.column_names(table))
    unknown_columns = [str(column) for column in frame.columns if column not in known]
    normalised, unknown_values = _normalise_categories(frame, table)
    invalid_numbers = _coerce_numeric_columns(frame, table)
    check = _check_derived(frame, table)

    warnings: list[str] = []
    if unknown_columns:
        warnings.append(f"Columns not in the {table} schema were ignored by the rules: {', '.join(unknown_columns)}.")
    if unknown_values:
        warnings.append(f"Values outside the column vocabulary (they fail validation): {_format_counts(unknown_values)}.")
    if invalid_numbers:
        warnings.append(f"Non-numeric values treated as missing: {_format_counts(invalid_numbers)}.")
    if check.mismatches:
        warnings.append(
            "Provided derived values disagree with their schema definition, so those rows fail the hard "
            f"'{DERIVED_CHECK_NAME}' check (the rules ran on the recomputed values): {_format_counts(check.mismatches)}."
        )
    normalisation = {
        "values_normalised": normalised,
        "unknown_values": unknown_values,
        "invalid_numbers": invalid_numbers,
        "unknown_columns": unknown_columns,
        "derived_computed": check.computed,
        "derived_mismatches": check.mismatches,
        "derived_mismatch_rows": check.mismatch_rows,
    }
    return PreparedRecords(
        frame=schema.order_columns(check.validation_frame, table).reset_index(drop=True),
        normalisation=normalisation,
        warnings=warnings,
        records=schema.order_columns(check.returned_frame, table).reset_index(drop=True),
        derived_mismatch=check.row_mismatch,
        table=table,
    )


# ---------------------------------------------------------------------------
# Generation filters
# ---------------------------------------------------------------------------
def _filter_error(message: str) -> BadRequestError:
    return BadRequestError(message, error_type="invalid_filters")


def _is_numeric_column(spec: schema.ColumnSpec) -> bool:
    """Numerical columns and integer-coded categoricals (month) accept ``min``/``max``."""
    if spec.sdtype == "numerical":
        return True
    return bool(spec.categories) and all(isinstance(c, int) and not isinstance(c, bool) for c in spec.categories)


def _filter_values(column: str, key: str, values: Sequence[Any], spec: schema.ColumnSpec) -> list[Any]:
    """``in``/``not_in`` values coerced to the column type (400 when a value cannot match)."""
    out: list[Any] = []
    for value in values:
        if spec.categories:
            code = canonical_category(column, value, spec.categories)
            if code is None:
                raise _filter_error(
                    f"Unknown value {value!r} in filter '{column}.{key}'. Allowed values: {list(spec.categories)}."
                )
            out.append(code)
        elif spec.dtype == "bool":
            flag = _to_bool(value)
            if flag is None:
                raise _filter_error(f"Filter '{column}.{key}' expects true or false, got {value!r}.")
            out.append(flag)
        elif spec.sdtype == "numerical":
            number = None if _is_bool(value) else pd.to_numeric(value, errors="coerce")
            if number is None or not np.isfinite(number):
                raise _filter_error(f"Filter '{column}.{key}' expects numbers, got {value!r}.")
            number = float(number)
            out.append(int(number) if spec.dtype == "int" and number.is_integer() else number)
        else:
            out.append(value)
    return out


def _normalise_filter(table: str, column: str, spec: dict[str, Any]) -> dict[str, Any]:
    """Type-check one filter against the column it reads (see ``BedSynthService.normalise_filters``)."""
    column_spec = schema.column_spec(table, column)
    low, high = spec.get("min"), spec.get("max")
    if (low is not None or high is not None) and not _is_numeric_column(column_spec):
        raise _filter_error(
            f"Filter '{column}' cannot use min/max: the column is {column_spec.sdtype}. Use 'in' or 'not_in'."
        )
    for name, bound in (("min", low), ("max", high)):
        if bound is not None and (_is_bool(bound) or not isinstance(bound, (int, float)) or not math.isfinite(bound)):
            raise _filter_error(f"Filter '{column}.{name}' must be a finite number, got {bound!r}.")
    if low is not None and column_spec.max_value is not None and low > column_spec.max_value:
        raise _filter_error(
            f"Filter '{column}.min' ({low:g}) is above the largest possible value "
            f"({column_spec.max_value:g}); no row can match."
        )
    if high is not None and column_spec.min_value is not None and high < column_spec.min_value:
        raise _filter_error(
            f"Filter '{column}.max' ({high:g}) is below the smallest possible value "
            f"({column_spec.min_value:g}); no row can match."
        )
    for key in ("in", "not_in"):
        if key in spec:
            spec[key] = _filter_values(column, key, spec[key], column_spec)
    if column_spec.categories:
        vocabulary: list[Any] | None = list(column_spec.categories)
    else:
        vocabulary = [False, True] if column_spec.dtype == "bool" else None
    if vocabulary is not None:
        allowed = [value for value in vocabulary if "in" not in spec or value in spec["in"]]
        allowed = [value for value in allowed if value not in spec.get("not_in", [])]
        if not allowed:
            raise _filter_error(f"The filter on '{column}' excludes every value; no row can match.")
    return spec


# ---------------------------------------------------------------------------
# Reference subsets (quality is measured against comparable real rows)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ReferenceSubset:
    """Selects the reference rows comparable with a generated dataset.

    A conditional scenario (e.g. ``high_occupancy``) deliberately shifts the distribution, so its
    output is compared with the reference rows of the same scenario/conditions/filters rather
    than with the whole reference table.
    """

    scenario: Scenario | None = None
    conditions: Mapping[str, Sequence[Any]] = field(default_factory=dict)
    filters: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)

    @property
    def is_full(self) -> bool:
        """True when no restriction applies (the whole reference table is comparable)."""
        scenario_restricts = bool(self.scenario and (self.scenario.conditions or self.scenario.filters))
        return not (scenario_restricts or self.conditions or self.filters)

    def mask(self, frame: pd.DataFrame) -> pd.Series:
        """Boolean mask of ``frame`` rows inside the subset (raises ValueError if disjoint)."""
        conditions: dict[str, tuple] = dict(self.scenario.conditions) if self.scenario else {}
        filters: dict[str, dict] = {k: dict(v) for k, v in self.scenario.filters.items()} if self.scenario else {}
        conditions = merge_conditions(conditions, {k: tuple(v) for k, v in self.conditions.items()})
        filters = merge_filters(filters, {k: dict(v) for k, v in self.filters.items()})
        return condition_mask(frame, conditions) & filter_mask(frame, filters)


def build_reference_subset(
    table: str,
    scenario: Scenario | None,
    conditions: Mapping[str, Sequence[Any]] | None = None,
    filters: Mapping[str, Mapping[str, Any]] | None = None,
    wards: Sequence[str] | None = None,
    months: Sequence[int] | None = None,
) -> ReferenceSubset:
    """Reference subset for a request: scenario + extra conditions + wards (+ calendar months)."""
    extra: dict[str, tuple] = {column: tuple(values) for column, values in (conditions or {}).items()}
    try:
        if wards:
            extra = merge_conditions(extra, {"ward_id": tuple(wards)})
        if months and table == schema.WARD_DAILY:
            extra = merge_conditions(extra, {"month": tuple(months)})
    except ValueError:
        logger.debug("Wards/months do not overlap the request conditions; using the conditions only")
    return ReferenceSubset(scenario=scenario, conditions=extra, filters=dict(filters or {}))


# ---------------------------------------------------------------------------
# Dataset store
# ---------------------------------------------------------------------------
class DatasetStore:
    """File-system store of generated datasets: ``<base_dir>/<dataset_id>/{data.csv, metadata.json}``.

    Identifiers are ``ds-`` followed by 12 lowercase hex characters. Every method validates the
    id against that pattern *before* touching the file system and checks that the resolved path
    stays directly inside ``base_dir``, so ids such as ``../../etc`` can never escape the store.
    Writes are staged in a hidden directory and moved into place atomically; deletes are moved
    aside first, so listings never observe half-written or half-deleted datasets.

    Concurrency: every read and mutation of one store runs under a re-entrant lock, and downloads
    read the stored file completely inside it (:meth:`read_data`) instead of streaming an open
    handle, so a concurrent delete can never hit a file that is still open (on Windows a rename
    of an open file fails). Concurrent deletes of one dataset therefore resolve to exactly one
    success and ``DatasetNotFoundError`` for the others; a delete that still meets a locked file
    (e.g. opened by another program) retries briefly and then raises :class:`ConflictError`.
    """

    DATA_FILE = "data.csv"
    METADATA_FILE = "metadata.json"
    DELETE_RETRIES = 5
    DELETE_RETRY_SECONDS = 0.05

    def __init__(self, base_dir: Path | str | None = None) -> None:
        self._base_dir = Path(base_dir) if base_dir is not None else None
        self._lock = threading.RLock()

    @property
    def base_dir(self) -> Path:
        """Store directory (default ``<synthetic_dir>/api``); resolved lazily, never created on read."""
        if self._base_dir is not None:
            return self._base_dir
        return get_path("synthetic_dir", create=False) / "api"

    @staticmethod
    def new_id() -> str:
        """A fresh random dataset id, e.g. ``ds-3f9c0a1b2c4d``."""
        return f"ds-{secrets.token_hex(6)}"

    @staticmethod
    def is_valid_id(dataset_id: Any) -> bool:
        return isinstance(dataset_id, str) and DATASET_ID_PATTERN.fullmatch(dataset_id) is not None

    def _dataset_dir(self, dataset_id: str) -> Path:
        if not self.is_valid_id(dataset_id):
            raise DatasetNotFoundError(f"Dataset '{dataset_id}' not found (ids look like 'ds-0123456789ab').")
        base = self.base_dir.resolve()
        path = (base / dataset_id).resolve()
        if path.parent != base:
            raise DatasetNotFoundError(f"Dataset '{dataset_id}' not found.")
        return path

    def _existing_dir(self, dataset_id: str) -> Path:
        path = self._dataset_dir(dataset_id)
        if not (path / self.METADATA_FILE).is_file():
            raise DatasetNotFoundError(f"Dataset '{dataset_id}' not found.")
        return path

    def exists(self, dataset_id: str) -> bool:
        try:
            self._existing_dir(dataset_id)
        except DatasetNotFoundError:
            return False
        return True

    def save(self, data: pd.DataFrame, metadata: Mapping[str, Any]) -> dict[str, Any]:
        """Persist ``data`` (CSV) and ``metadata`` (JSON) under ``metadata['dataset_id']``."""
        dataset_id = str(metadata.get("dataset_id"))
        target = self._dataset_dir(dataset_id)
        stored = to_jsonable(dict(metadata))
        with self._lock:
            if target.exists():
                raise FileExistsError(f"Dataset '{dataset_id}' already exists.")
            target.parent.mkdir(parents=True, exist_ok=True)
            staging = target.parent / f".staging-{dataset_id}-{secrets.token_hex(4)}"
            staging.mkdir()
            try:
                # LF line endings on every platform: the stored file (and its download) is then
                # byte-identical to the CSV body /generate returns for the same records.
                io.write_table(data, staging / self.DATA_FILE, lineterminator="\n")
                (staging / self.METADATA_FILE).write_text(json.dumps(stored, indent=2), encoding="utf-8")
                os.replace(staging, target)
            except BaseException:
                shutil.rmtree(staging, ignore_errors=True)
                raise
        logger.info("Stored dataset %s (%s rows) in %s", dataset_id, len(data), target)
        return stored

    def _read_metadata(self, directory: Path) -> dict[str, Any]:
        metadata = json.loads((directory / self.METADATA_FILE).read_text(encoding="utf-8"))
        if not isinstance(metadata, dict):
            raise ValueError("metadata.json does not hold a JSON object")
        data_file = directory / self.DATA_FILE
        metadata["size_bytes"] = data_file.stat().st_size if data_file.is_file() else 0
        return metadata

    def _metadata_of(self, dataset_id: str) -> dict[str, Any]:
        """Metadata of an existing dataset; unreadable files raise a ``dataset_corrupt`` error."""
        directory = self._existing_dir(dataset_id)
        try:
            return self._read_metadata(directory)
        except FileNotFoundError as exc:     # deleted between the existence check and the read
            raise DatasetNotFoundError(f"Dataset '{dataset_id}' not found.") from exc
        except (OSError, ValueError) as exc:
            logger.warning("Dataset %s has unreadable metadata: %s", dataset_id, exc)
            raise ServiceError(
                f"Dataset '{dataset_id}' is unreadable (corrupt or locked metadata); delete it and generate it again.",
                error_type="dataset_corrupt",
            ) from exc

    def list_datasets(self, table: str | None = None) -> list[dict[str, Any]]:
        """Metadata of every readable stored dataset (optionally of one table), newest first."""
        base = self.base_dir
        if not base.is_dir():
            return []
        items: list[dict[str, Any]] = []
        with self._lock:
            for child in base.iterdir():
                if not child.is_dir() or not self.is_valid_id(child.name):
                    continue
                try:
                    metadata = self._read_metadata(child)
                except (OSError, ValueError) as exc:
                    logger.warning("Skipping unreadable dataset %s: %s", child.name, exc)
                    continue
                if table is None or metadata.get("table") == table:
                    items.append(metadata)
        items.sort(key=lambda item: (str(item.get("created_at") or ""), str(item.get("dataset_id"))), reverse=True)
        return items

    def count(self) -> int:
        """Number of stored dataset directories (unreadable ones included)."""
        base = self.base_dir
        if not base.is_dir():
            return 0
        with self._lock:
            return sum(1 for child in base.iterdir() if child.is_dir() and self.is_valid_id(child.name))

    def get(self, dataset_id: str) -> dict[str, Any]:
        """Stored metadata of a dataset (raises :class:`DatasetNotFoundError`)."""
        with self._lock:
            return self._metadata_of(dataset_id)

    def data_path(self, dataset_id: str) -> Path:
        """Path of the stored CSV file (raises :class:`DatasetNotFoundError`).

        Prefer :meth:`read_data` for downloads: a path handed to a streaming response keeps the
        file open after the lock is released.
        """
        with self._lock:
            path = self._existing_dir(dataset_id) / self.DATA_FILE
            if not path.is_file():
                raise DatasetNotFoundError(f"Dataset '{dataset_id}' has no data file.")
            return path

    def read_data(self, dataset_id: str) -> tuple[bytes, dict[str, Any]]:
        """``(CSV bytes, metadata)`` of a dataset, read consistently under the store lock."""
        with self._lock:
            metadata = self._metadata_of(dataset_id)
            try:
                return self.data_path(dataset_id).read_bytes(), metadata
            except FileNotFoundError as exc:
                raise DatasetNotFoundError(f"Dataset '{dataset_id}' has no data file.") from exc

    def load(self, dataset_id: str, table: str | None = None) -> pd.DataFrame:
        """Stored records with canonical dtypes."""
        with self._lock:
            table = table or str(self.get(dataset_id).get("table"))
            try:
                return io.read_table(self.data_path(dataset_id), table)
            except FileNotFoundError as exc:
                raise DatasetNotFoundError(f"Dataset '{dataset_id}' has no data file.") from exc

    def delete(self, dataset_id: str) -> None:
        """Remove a dataset.

        Raises :class:`DatasetNotFoundError` when it does not exist (also when a concurrent request
        deleted it first) and :class:`ConflictError` when its files stay locked by another process.
        """
        with self._lock:
            path = self._existing_dir(dataset_id)
            trash = path.parent / f".deleting-{dataset_id}-{secrets.token_hex(4)}"
            for attempt in range(self.DELETE_RETRIES):
                try:
                    os.replace(path, trash)
                    break
                except FileNotFoundError as exc:
                    raise DatasetNotFoundError(f"Dataset '{dataset_id}' not found.") from exc
                except PermissionError as exc:
                    if attempt == self.DELETE_RETRIES - 1:
                        logger.warning("Dataset %s is locked and cannot be deleted: %s", dataset_id, exc)
                        raise ConflictError(
                            f"Dataset '{dataset_id}' is in use (a download may still be running); try again shortly.",
                            error_type="dataset_busy",
                        ) from exc
                    time.sleep(self.DELETE_RETRY_SECONDS * (attempt + 1))
        shutil.rmtree(trash, ignore_errors=True)
        logger.info("Deleted dataset %s", dataset_id)


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------
@dataclass
class GenerationOutcome:
    """Result of :meth:`BedSynthService.generate`: the JSON payload and the generated frame."""

    response: dict[str, Any]
    data: pd.DataFrame

    def csv_text(self) -> str:
        return self.data.to_csv(index=False, lineterminator="\n")

    def headers(self) -> dict[str, str]:
        return dataset_headers(self.response)


class BedSynthService:
    """Application service behind the REST API (one instance per app; thread-safe).

    Parameters
    ----------
    generator:
        A ready generation engine (anything with ``available_models``, ``default_model_id`` and
        ``generate``). When omitted, :class:`bedsynth.synthesis.generator.SyntheticDataGenerator`
        is created lazily on first use (``generator_factory`` overrides how).
    store_dir / store:
        Where persisted datasets live (default ``<synthetic_dir>/api``).
    registry:
        Optional :class:`~bedsynth.synthesis.registry.ModelRegistry` used to enrich model listings
        (default reasons, metrics).
    models_dir / processed_dir:
        Passed to the default generator; ``processed_dir`` also locates the reference data.
    """

    def __init__(
        self,
        generator: Any | None = None,
        *,
        store_dir: Path | str | None = None,
        store: DatasetStore | None = None,
        registry: Any | None = None,
        models_dir: Path | str | None = None,
        processed_dir: Path | str | None = None,
        generator_factory: Callable[[], Any] | None = None,
        settings: Mapping[str, Any] | None = None,
    ) -> None:
        settings = dict(settings or get_settings())
        api_settings = settings.get("api") or {}
        project = settings.get("project") or {}
        self.name: str = str(project.get("name", "BedSynth AI"))
        self.title: str = str(api_settings.get("title", self.name))
        self.max_rows: int = int(api_settings.get("max_rows", 10_000))
        self.evaluate_max_rows: int = int(api_settings.get("evaluate_max_rows", 5000))
        self.persist_enabled: bool = bool(api_settings.get("persist_datasets", True))
        self.max_condition_combinations: int = int(api_settings.get("max_condition_combinations", 1000))
        self.time_budget_seconds: float = float(api_settings.get("time_budget_seconds", 60))
        self.max_pending_generations: int = max(1, int(api_settings.get("max_pending_generations", 4)))
        self.generation_wait_seconds: float = float(api_settings.get("generation_wait_seconds", 60))
        self.store = store or DatasetStore(store_dir)
        self._generator = generator
        self._owns_generator = generator is None and generator_factory is None
        self._generator_factory = generator_factory or _default_generator_factory(models_dir, processed_dir)
        self._registry = registry
        self._models_dir = models_dir
        self._processed_dir = Path(processed_dir) if processed_dir is not None else None
        default_experiments = (settings.get("synthesis") or {}).get("default_experiment")
        self._default_experiments: dict[str, Any] = dict(default_experiments or {})
        # Engine life cycle as seen by /health (which never waits for it): ``_init_lock`` is held
        # while the engine is being created, ``_preloading`` is set from the moment the start-up
        # preload is scheduled until it has finished, and ``_engine_error`` keeps the (client-safe)
        # message of the last failed initialisation until an attempt succeeds.
        self._init_lock = threading.Lock()
        self._preloading = threading.Event()
        self._engine_error: str | None = None
        self._registry_file_lock = threading.Lock()
        self._registry_file_cache: tuple[tuple[str, int, int], dict[str, dict[str, Any]]] | None = None
        self._sdv_version: tuple[str | None] | None = None
        # SDV sampling swaps the process-wide numpy/torch RNG state (copulas, ctgan), so concurrent
        # draws - even from different models - would break seeded reproducibility: generations run
        # one at a time. Each holds the engine for at most its time budget, and admission control
        # (``_pending``) bounds the queue so waiting requests never exhaust the server's thread pool.
        self._generate_lock = threading.Lock()
        self._pending = threading.BoundedSemaphore(self.max_pending_generations)
        self._reference_lock = threading.Lock()
        self._reference_cache: dict[tuple[str, str, str], tuple[int, pd.DataFrame]] = {}

    # ------------------------------------------------------------------ engine
    @property
    def engine_loaded(self) -> bool:
        """True once the generation engine has been created."""
        return self._generator is not None

    @property
    def engine_state(self) -> str:
        """``not_loaded``, ``loading``, ``ready`` or ``failed`` (never waits for the engine)."""
        return self._engine_snapshot()[1]

    def _engine_snapshot(self) -> tuple[Any, str, str | None]:
        """``(generator or None, state, client-safe error)`` at the time of the call, without waiting.

        The checks are ordered so that a concurrent initialisation is never misreported: an
        attempt records its outcome (generator or error) before it releases ``_init_lock``, and the
        preload flag is cleared only after that.
        """
        generator = self._generator
        if generator is not None:
            return generator, ENGINE_READY, None
        if self._preloading.is_set() or self._init_lock.locked():
            return None, ENGINE_LOADING, None
        generator = self._generator
        if generator is not None:
            return generator, ENGINE_READY, None
        error = self._engine_error
        return (None, ENGINE_FAILED, error) if error is not None else (None, ENGINE_NOT_LOADED, None)

    def get_generator(self) -> Any:
        """The generation engine, created on first use (raises 503 errors when unavailable).

        Concurrent callers wait for one initialisation. A failure is logged with its traceback;
        clients get a fixed message naming only the exception type, and the next call retries.
        """
        if self._generator is not None:
            return self._generator
        with self._init_lock:
            if self._generator is None:
                started = time.perf_counter()
                try:
                    generator = self._generator_factory()
                except NotImplementedError as exc:
                    error: ServiceError = _engine_not_implemented()
                    self._engine_error = error.detail
                    raise error from exc
                except Exception as exc:
                    logger.exception("Could not initialise the generation engine")
                    error = ServiceUnavailableError(
                        f"The generation engine could not be initialised ({type(exc).__name__}); see the server log.",
                        error_type="engine_unavailable",
                    )
                    self._engine_error = error.detail
                    raise error from exc
                self._engine_error = None
                self._generator = generator
                logger.info("Generation engine %s ready in %.1f s", type(generator).__name__, time.perf_counter() - started)
        return self._generator

    def start_preload(self) -> threading.Thread:
        """Load the engine in a daemon thread (start-up); ``/health`` reports ``loading`` meanwhile."""
        self._preloading.set()
        thread = threading.Thread(target=self.preload, name="bedsynth-engine-preload", daemon=True)
        try:
            thread.start()
        except BaseException:
            self._preloading.clear()
            raise
        return thread

    def preload(self) -> None:
        """Initialise the engine eagerly (called at start-up so the first request is fast)."""
        try:
            generator = self.get_generator()
            for table in schema.TABLES:
                self._model_entries(generator, table)
        except ServiceError as exc:
            logger.warning("Engine preload skipped: %s", exc.detail)
        except Exception:  # noqa: BLE001 - a background thread must log, not die silently
            logger.exception("Engine preload failed")
        finally:
            self._preloading.clear()

    def _model_entries(self, generator: Any, table: str) -> list[dict[str, Any]]:
        """Trained models of ``table`` (registered entries whose files are missing are skipped)."""
        try:
            entries = generator.available_models(table)
        except NotImplementedError as exc:
            raise _engine_not_implemented() from exc
        normalised: list[dict[str, Any]] = []
        for entry in entries or []:
            if isinstance(entry, Mapping):
                if entry.get("trained") is not False:
                    normalised.append(dict(entry))
            elif entry is not None:
                normalised.append({"model_id": str(entry)})
        return normalised

    def _default_model_id(self, generator: Any, table: str) -> str | None:
        try:
            value = generator.default_model_id(table)
        except NotImplementedError as exc:
            raise _engine_not_implemented() from exc
        return str(value) if value else None

    def _model_status(self) -> tuple[dict[str, list[str]], dict[str, str | None], str | None]:
        """Trained ids and default per table from the engine (created, or waited for, when needed),
        plus an error detail when the engine is unavailable. Used by ``/api/v1/tables``."""
        try:
            generator = self.get_generator()
        except ServiceError as exc:
            return {table: [] for table in schema.TABLES}, {table: None for table in schema.TABLES}, exc.detail
        return self._engine_model_status(generator)

    def _engine_model_status(
        self, generator: Any
    ) -> tuple[dict[str, list[str]], dict[str, str | None], str | None]:
        """Trained ids and default per table as listed by a ready engine (never raises)."""
        trained: dict[str, list[str]] = {table: [] for table in schema.TABLES}
        defaults: dict[str, str | None] = {table: None for table in schema.TABLES}
        try:
            for table in schema.TABLES:
                entries = self._model_entries(generator, table)
                trained[table] = [mid for mid in (model_id_of(entry) for entry in entries) if mid]
                defaults[table] = self._default_model_id(generator, table) if entries else None
        except ServiceError as exc:
            return trained, defaults, exc.detail
        except Exception as exc:  # a broken registry must not take /health down
            logger.exception("Could not list trained models")
            return trained, defaults, f"Could not list trained models ({type(exc).__name__}); see the server log."
        return trained, defaults, None

    def _models_root(self) -> Path:
        """Models directory of the engine: explicit ``models_dir``, the injected registry's, or the configured one."""
        if self._models_dir is not None:
            return Path(self._models_dir)
        registry_dir = getattr(self._registry, "models_dir", None)
        if isinstance(registry_dir, (str, Path)):
            return Path(registry_dir)
        return get_path("models_dir", create=False)

    def _registry_file(self) -> tuple[dict[str, dict[str, Any]], str | None]:
        """``registry.json`` reduced to ``{table: {"default": id | None, "models": [ids]}}``.

        Read with :mod:`json` only (no SDV import) and cached until the file changes. A missing
        file means nothing is registered; an unreadable one yields an empty view plus a
        client-safe message (the details go to the server log).
        """
        path = self._models_root() / REGISTRY_FILE
        try:
            stat = path.stat()
        except FileNotFoundError:
            return {}, None
        except OSError as exc:
            logger.warning("Model registry %s is not accessible: %s", path, exc)
            return {}, f"The model registry could not be read ({type(exc).__name__}); see the server log."
        key = (str(path), stat.st_mtime_ns, stat.st_size)
        with self._registry_file_lock:
            cached = self._registry_file_cache
            if cached is not None and cached[0] == key:
                return cached[1], None
        try:
            raw = json.loads(path.read_text(encoding="utf-8")) or {}
        except (OSError, ValueError) as exc:
            logger.warning("Model registry %s is not readable: %s", path, exc)
            return {}, f"The model registry could not be read ({type(exc).__name__}); see the server log."
        view: dict[str, dict[str, Any]] = {}
        for table in schema.TABLES:
            entry = raw.get(table) if isinstance(raw, Mapping) else None
            entry = entry if isinstance(entry, Mapping) else {}
            models = entry.get("models")
            default = entry.get("default")
            view[table] = {
                "default": default if isinstance(default, str) else None,
                "models": [str(model_id) for model_id in models] if isinstance(models, Mapping) else [],
            }
        with self._registry_file_lock:
            self._registry_file_cache = (key, view)
        return view, None

    def _registry_file_status(self) -> tuple[dict[str, list[str]], dict[str, str | None], str | None]:
        """Trained ids and default per table from ``registry.json``, without the engine.

        Mirrors :class:`bedsynth.synthesis.registry.ModelRegistry`: a model is trained when it is
        registered and its ``synthesizer.pkl`` exists (ids sorted, as the engine lists them); the
        default is the registry default when trained, else
        ``settings.synthesis.default_experiment``, else the first trained id.
        """
        view, error = self._registry_file()
        root = self._models_root()
        trained: dict[str, list[str]] = {}
        defaults: dict[str, str | None] = {}
        for table in schema.TABLES:
            entry = view.get(table) or {}
            ids = sorted(
                model_id for model_id in entry.get("models", [])
                if _MODEL_ID_RE.fullmatch(model_id) and (root / table / model_id / MODEL_FILE).exists()
            )
            configured = self._default_experiments.get(table)
            if not ids:
                default = None
            elif entry.get("default") in ids:
                default = entry["default"]
            elif configured in ids:
                default = str(configured)
            else:
                default = ids[0]
            trained[table], defaults[table] = ids, default
        return trained, defaults, error

    def _registry_view(self) -> dict[str, Any]:
        """Registry contents (``registry.json`` layout) when a registry is reachable, else ``{}``."""
        registry = self._registry
        if registry is None and self._generator is not None:
            registry = getattr(self._generator, "registry", None)
        if registry is None and self._owns_generator:
            try:
                from bedsynth.synthesis.registry import ModelRegistry

                registry = self._registry = ModelRegistry(self._models_dir)
            except Exception as exc:
                logger.debug("Model registry unavailable: %s", exc)
                return {}
        if registry is None:
            return {}
        try:
            view = registry.to_dict()
        except Exception as exc:
            logger.debug("Model registry not readable: %s", exc)
            return {}
        return to_jsonable(view) if isinstance(view, Mapping) else {}

    def _model_card(self, generator: Any, table: str, model_id: str | None) -> dict[str, Any] | None:
        """Model card of ``model_id`` (registry card merged with the generator's entry)."""
        if not model_id:
            return None
        try:
            entries = self._model_entries(generator, table)
        except ServiceError:
            entries = []
        entry = next((item for item in entries if model_id_of(item) == model_id), None)
        registry_table = self._registry_view().get(table) or {}
        registry_models = registry_table.get("models") if isinstance(registry_table, Mapping) else None
        registry_card = registry_models.get(model_id) if isinstance(registry_models, Mapping) else None
        if entry is None and not isinstance(registry_card, Mapping):
            return None
        return {**(registry_card if isinstance(registry_card, Mapping) else {}), **(entry or {})}

    # --------------------------------------------------------------- catalogue
    def check_table(self, table: str) -> str:
        """Return ``table`` or raise a 404 error listing the known tables."""
        if table not in schema.TABLES:
            raise NotFoundError(
                f"Unknown table '{table}'. Known tables: {', '.join(schema.TABLES)}.", error_type="table_not_found"
            )
        return table

    def resolve_scenario(self, name: str | None, table: str | None = None) -> Scenario | None:
        """Scenario by name (404 if unknown; 400 if it belongs to another table)."""
        if not name:
            return None
        scenario = SCENARIOS.get(name)
        if scenario is None:
            raise NotFoundError(
                f"Unknown scenario '{name}'. See GET {API_PREFIX}/scenarios for the catalogue.",
                error_type="scenario_not_found",
            )
        if table is not None and scenario.table != table:
            raise BadRequestError(
                f"Scenario '{name}' belongs to table '{scenario.table}', not '{table}'.",
                error_type="scenario_table_mismatch",
            )
        return scenario

    def service_info(self, endpoints: Sequence[Mapping[str, str]] = ()) -> dict[str, Any]:
        """Payload of ``GET /``."""
        return {
            "name": self.name,
            "title": self.title,
            "version": __version__,
            "description": (
                "Synthetic, non-clinical hospital operations data (ward-day bed occupancy and patient "
                "stays) generated on demand with SDV models, scenario conditioning and rule validation."
            ),
            "links": {
                "docs": "/docs",
                "redoc": "/redoc",
                "openapi": "/openapi.json",
                "ui": "/ui",
                "health": "/health",
                "tables": f"{API_PREFIX}/tables",
                "scenarios": f"{API_PREFIX}/scenarios",
                "models": f"{API_PREFIX}/models",
                "generate": f"{API_PREFIX}/generate",
                "datasets": f"{API_PREFIX}/datasets",
            },
            "endpoints": [dict(item) for item in endpoints],
        }

    def _installed_sdv_version(self) -> str | None:
        """Installed SDV version, read once from the package metadata (SDV is not imported)."""
        if self._sdv_version is None:
            self._sdv_version = (package_version("sdv"),)
        return self._sdv_version[0]

    def health(self) -> dict[str, Any]:
        """Payload of ``GET /health``: cheap, never raises, never creates or waits for the engine.

        ``engine.loaded`` and ``engine.state`` describe the engine at the time of the call. Once it
        is ready, the trained models are listed by the engine; before that (``not_loaded``,
        ``loading``, ``failed``) they are read from ``registry.json`` directly. ``status`` is
        ``ok`` when every table has a trained model and the engine is ready or loads lazily on
        first use; it is ``degraded`` while the engine is loading at start-up, after a failed
        initialisation, or when a table has no trained model (``messages`` say what to do).
        """
        generator, state, init_error = self._engine_snapshot()
        registry_error: str | None = None
        listing_error: str | None = None
        if generator is not None:
            trained, defaults, listing_error = self._engine_model_status(generator)
        else:
            trained, defaults, registry_error = self._registry_file_status()
        engine_error = listing_error or init_error
        trained_tables = [table for table in schema.TABLES if trained[table]]
        reference = {table: self.reference_available(table) for table in schema.TABLES}
        if engine_error:
            detail: str | None = engine_error
        elif state == ENGINE_LOADING:
            detail = ENGINE_LOADING_DETAIL
        elif state == ENGINE_NOT_LOADED:
            detail = ENGINE_NOT_LOADED_DETAIL
        else:
            detail = None

        messages: list[str] = []
        if engine_error:
            messages.append(engine_error)
        else:
            if state == ENGINE_LOADING:
                messages.append(ENGINE_LOADING_DETAIL)
            messages.extend(f"No trained model for '{table}'. {TRAIN_HINT}" for table in schema.TABLES if not trained[table])
        if registry_error:
            messages.append(registry_error)
        messages.extend(
            f"Reference data for '{table}' is missing (needed by evaluate and scenario support). {PREPARE_HINT}"
            for table, available in reference.items()
            if not available
        )
        try:
            dataset_count = self.store.count()
        except OSError as exc:
            logger.warning("Dataset store not readable: %s", exc)
            dataset_count = 0
            messages.append(f"The dataset store could not be read ({type(exc).__name__}); see the server log.")
        healthy = (
            engine_error is None
            and state in (ENGINE_READY, ENGINE_NOT_LOADED)
            and len(trained_tables) == len(schema.TABLES)
        )
        return {
            "status": "ok" if healthy else "degraded",
            "version": __version__,
            "sdv_version": self._installed_sdv_version(),
            "engine": {"available": engine_error is None, "loaded": generator is not None, "state": state, "detail": detail},
            "models": {table: {"trained": trained[table], "default": defaults[table]} for table in schema.TABLES},
            "trained_tables": trained_tables,
            "reference_data": reference,
            "datasets": {"count": dataset_count, "persistence_enabled": self.persist_enabled},
            "messages": messages,
            "checked_at": utc_now_iso(),
        }

    def list_tables(self) -> dict[str, Any]:
        """Payload of ``GET /api/v1/tables``."""
        trained, defaults, _ = self._model_status()
        tables = []
        for table in schema.TABLES:
            info = TABLE_INFO[table]
            tables.append({
                "name": table,
                "title": info["title"],
                "description": info["description"],
                "grain": info["grain"],
                "key_column": schema.key_column(table),
                "n_columns": len(schema.output_columns(table)),
                "n_training_columns": len(schema.training_columns(table)),
                "n_derived_columns": len(schema.derived_columns(table)),
                "n_scenarios": len(scenarios_for(table)),
                "conditionable_columns": schema.conditionable_columns(table),
                "supports_calendar_mode": table == schema.WARD_DAILY,
                "default_model": defaults[table],
                "trained_models": trained[table],
                "reference_data_available": self.reference_available(table),
                "links": {
                    "schema": f"{API_PREFIX}/tables/{table}/schema",
                    "scenarios": f"{API_PREFIX}/scenarios?table={table}",
                    "models": f"{API_PREFIX}/models?table={table}",
                },
            })
        return {"tables": tables}

    def _rules_catalog(self, table: str) -> tuple[list[dict[str, Any]], bool]:
        try:
            return [to_jsonable(dict(rule)) for rule in validation.rules_catalog(table)], True
        except NotImplementedError:
            return [], False
        except Exception:
            logger.exception("Could not load the rule catalogue for %s", table)
            return [], False

    def table_schema(self, table: str) -> dict[str, Any]:
        """Payload of ``GET /api/v1/tables/{table}/schema``: schema catalogue + rules."""
        self.check_table(table)
        payload = schema.schema_as_dict(table)
        rules, rules_available = self._rules_catalog(table)
        payload.update(
            conditionable_columns=schema.conditionable_columns(table),
            filterable_columns=schema.column_names(table, ["train", "derived"]),
            scenarios=[scenario.name for scenario in scenarios_for(table)],
            rules=rules,
            rules_available=rules_available,
        )
        return to_jsonable(payload)

    def list_scenarios(self, table: str | None = None, include_support: bool = False) -> dict[str, Any]:
        """Payload of ``GET /api/v1/scenarios`` (optionally with reference-data support)."""
        if table is not None:
            self.check_table(table)
        items = []
        for scenario in SCENARIOS.values():
            if table is not None and scenario.table != table:
                continue
            item = scenario.to_dict()
            if include_support:
                item["support"] = self._scenario_support(scenario)
            items.append(item)
        return to_jsonable({"count": len(items), "categories": dict(SCENARIO_CATEGORIES), "scenarios": items})

    def get_scenario(self, name: str) -> dict[str, Any]:
        """Payload of ``GET /api/v1/scenarios/{name}`` (with reference-data support)."""
        scenario = self.resolve_scenario(name.strip().lower())
        if scenario is None:
            raise NotFoundError("A scenario name is required.", error_type="scenario_not_found")
        item = scenario.to_dict()
        item["support"] = self._scenario_support(scenario)
        return to_jsonable(item)

    def _scenario_support(self, scenario: Scenario) -> dict[str, Any] | None:
        try:
            reference = self.load_reference(scenario.table, "train")
        except ServiceError:
            return None
        matching = int(scenario_mask(reference, scenario).sum())
        total = len(reference)
        return {
            "split": "train",
            "reference_rows": total,
            "matching_rows": matching,
            "share": round(matching / total, 4) if total else None,
        }

    def list_models(self, table: str | None = None) -> dict[str, Any]:
        """Payload of ``GET /api/v1/models``: trained models, defaults and registry metrics."""
        tables = [self.check_table(table)] if table is not None else list(schema.TABLES)
        generator = self.get_generator()
        registry = self._registry_view()
        payload: dict[str, Any] = {}
        for name in tables:
            entries = self._model_entries(generator, name)
            default = self._default_model_id(generator, name) if entries else None
            registry_table = registry.get(name) if isinstance(registry.get(name), Mapping) else {}
            registry_models = registry_table.get("models") if isinstance(registry_table.get("models"), Mapping) else {}
            models = []
            for entry in entries:
                registry_card = registry_models.get(model_id_of(entry) or "")
                card = {**(registry_card if isinstance(registry_card, Mapping) else {}), **entry}
                models.append(model_info(card, name, default))
            payload[name] = {
                "default_model": default,
                "default_reason": registry_table.get("default_reason") or None,
                "models": models,
            }
        return {"tables": payload}

    # --------------------------------------------------------- reference data
    def reference_path(self, table: str, split: str = "train") -> Path:
        processed_dir = self._processed_dir or get_path("processed_dir", create=False)
        return io.processed_path(table, split, processed_dir)

    def reference_available(self, table: str, split: str = "train") -> bool:
        try:
            return self.reference_path(table, split).is_file()
        except (KeyError, ValueError, OSError):
            return False

    def load_reference(self, table: str, split: str = "train") -> pd.DataFrame:
        """Processed reference split (cached until the file changes). Callers must not mutate it."""
        path = self.reference_path(table, split)
        if not path.is_file():
            logger.info("Reference data not found at %s", path)
            raise ServiceUnavailableError(
                f"Reference data for '{table}' ({split} split) is not available. {PREPARE_HINT}",
                error_type="reference_data_unavailable",
            )
        stamp = path.stat().st_mtime_ns
        key = (table, split, str(path))
        with self._reference_lock:
            cached = self._reference_cache.get(key)
            if cached is not None and cached[0] == stamp:
                return cached[1]
        frame = derived_view(table, io.read_table(path, table))
        with self._reference_lock:
            self._reference_cache[key] = (stamp, frame)
        return frame

    # ------------------------------------------------------ validation/quality
    def run_validation(self, frame: pd.DataFrame, table: str, include_soft: bool = True) -> Any:
        """:func:`bedsynth.validation.validate_table` with errors mapped to service errors."""
        try:
            return validation.validate_table(frame.reset_index(drop=True), table, include_soft=include_soft)
        except NotImplementedError as exc:
            raise ServiceUnavailableError(
                "Operational rule validation (bedsynth.validation) is not available in this build.",
                error_type="validation_unavailable",
            ) from exc
        except Exception as exc:
            logger.exception("Validation failed")
            raise ServiceError(
                f"Validation failed ({type(exc).__name__}); see the server log.", error_type="validation_failed"
            ) from exc

    def _validation_block(
        self, frame: pd.DataFrame, table: str, warnings: list[str], prepared: PreparedRecords | None = None
    ) -> dict[str, Any] | None:
        """Validation summary for a response; failures become warnings instead of errors.

        With ``prepared`` (client records) the derived-value consistency check joins the hard rules.
        """
        try:
            report = self.run_validation(frame, table, include_soft=True)
        except ServiceError as exc:
            warnings.append(f"Validation skipped: {exc.detail}")
            return None
        if prepared is not None:
            summary = summarise_validation(report, [prepared.derived_check()], prepared.derived_mismatch)
        else:
            summary = summarise_validation(report)
        if summary["all_hard_rules_passed"] is False:
            failed = [rule["rule_id"] for rule in summary["rules"] if rule["severity"] == "hard" and not rule["passed"]]
            warnings.append(
                f"{summary['n_rows'] - summary['n_hard_valid']} rows violate hard rules ({', '.join(failed)})."
            )
        return summary

    def compute_quality(
        self,
        synthetic: pd.DataFrame,
        table: str,
        *,
        split: str = "train",
        subset: ReferenceSubset | None = None,
        max_rows: int | None = None,
        seed: int = 0,
        warnings: list[str] | None = None,
    ) -> dict[str, Any]:
        """Quick quality of ``synthetic`` versus the (subset of the) reference split.

        Raises a 400 ``too_few_rows`` error below :data:`MIN_EVALUATE_ROWS` synthetic rows: KS/TVD
        statistics of a handful of rows are meaningless.
        """
        warnings = warnings if warnings is not None else []
        reference = self.load_reference(table, split)
        if len(synthetic) < MIN_EVALUATE_ROWS:
            raise BadRequestError(
                f"Quality needs at least {MIN_EVALUATE_ROWS} synthetic rows to be meaningful; got {len(synthetic)}.",
                error_type="too_few_rows",
            )
        real, kind = reference, "full"
        if subset is not None and not subset.is_full:
            try:
                mask = subset.mask(reference)
            except (ValueError, KeyError) as exc:
                logger.debug("Reference subset not applicable: %s", exc)
                mask = None
            if mask is not None:
                matching = int(mask.sum())
                if matching >= MIN_REFERENCE_SUBSET_ROWS:
                    real, kind = reference[mask], "matching"
                else:
                    warnings.append(
                        f"Only {matching} reference rows match the scenario/conditions (minimum "
                        f"{MIN_REFERENCE_SUBSET_ROWS}); quality was measured against the full {split} split."
                    )
        limit = max_rows or self.evaluate_max_rows
        real_sample = subsample(real, limit, seed)
        synthetic_sample = subsample(synthetic, limit, seed)
        columns = [
            column
            for column in schema.column_names(table, ["train", "derived"])
            if column in real_sample.columns and column in synthetic_sample.columns
        ]
        try:
            metrics = run_quick_quality(
                real_sample[columns].reset_index(drop=True), synthetic_sample[columns].reset_index(drop=True), table
            )
        except NotImplementedError as exc:
            raise ServiceUnavailableError(
                "Quality evaluation (bedsynth.evaluation.quick_quality) is not available in this build.",
                error_type="evaluation_unavailable",
            ) from exc
        except (ValueError, KeyError) as exc:
            logger.warning("Quality evaluation of %s records failed", table, exc_info=True)
            raise BadRequestError(
                f"Quality evaluation failed ({type(exc).__name__}); check that the records match the table "
                f"schema (GET {API_PREFIX}/tables/{table}/schema). Details are in the server log.",
                error_type="evaluation_failed",
            ) from exc
        except Exception as exc:
            logger.exception("Quality evaluation failed")
            raise ServiceError(
                f"Quality evaluation failed ({type(exc).__name__}); see the server log.", error_type="evaluation_failed"
            ) from exc
        metrics = to_jsonable(metrics) if isinstance(metrics, Mapping) else {"value": to_jsonable(metrics)}
        score = metrics.get("score")
        return {
            "score": round(float(score), 4) if isinstance(score, (int, float)) and not isinstance(score, bool) else None,
            "reference": {
                "split": split,
                "subset": kind,
                "reference_rows": int(len(real)),
                "rows_compared": int(len(real_sample)),
                "synthetic_rows_compared": int(len(synthetic_sample)),
            },
            "metrics": metrics,
        }

    # --------------------------------------------------------------- generate
    def normalise_conditions(
        self, table: str, conditions: Mapping[str, Any] | None, warnings: list[str] | None = None
    ) -> dict[str, list[Any]]:
        """Validate condition columns and map values to canonical codes (400 on unknown values).

        A column whose values cover its whole vocabulary means the same as no condition, so it is
        dropped (with a warning appended to ``warnings``) instead of being expanded into one
        value combination per category.
        """
        if not conditions:
            return {}
        try:
            validated = validate_conditions(table, dict(conditions))
        except ValueError as exc:
            raise BadRequestError(str(exc), error_type="invalid_conditions") from exc
        normalised: dict[str, list[Any]] = {}
        for column, values in validated.items():
            categories = schema.column_spec(table, column).categories
            canonical: list[Any] = []
            for value in values:
                code = canonical_category(column, value, categories)
                if code is None:
                    raise BadRequestError(
                        f"Unknown value {value!r} for condition '{column}'. Allowed values: {list(categories or [])}.",
                        error_type="invalid_conditions",
                    )
                if code not in canonical:
                    canonical.append(code)
            if categories and set(canonical) == set(categories):
                if warnings is not None:
                    warnings.append(f"Condition on '{column}' lists every value; it was dropped (same as no condition).")
                continue
            normalised[column] = canonical
        return normalised

    def condition_combinations(
        self,
        table: str,
        scenario: Scenario | None,
        conditions: Mapping[str, Sequence[Any]],
        wards: Sequence[str] | None,
    ) -> int | None:
        """Number of value combinations the engine will plan (scenario ∩ conditions ∩ wards).

        Returns ``None`` when the conditions contradict each other (the engine reports that).
        """
        merged: dict[str, tuple] = {column: tuple(values) for column, values in (scenario.conditions if scenario else {}).items()}
        try:
            merged = merge_conditions(merged, {column: tuple(values) for column, values in conditions.items()})
            if wards:
                merged = merge_conditions(merged, {"ward_id": tuple(wards)})
        except ValueError:
            return None
        return math.prod(len(values) for values in merged.values())

    def check_condition_combinations(
        self,
        table: str,
        scenario: Scenario | None,
        conditions: Mapping[str, Sequence[Any]],
        wards: Sequence[str] | None,
    ) -> None:
        """400 ``too_many_combinations`` when the request would expand into too many condition cells."""
        combinations = self.condition_combinations(table, scenario, conditions, wards)
        if combinations is not None and combinations > self.max_condition_combinations:
            raise BadRequestError(
                f"The conditions expand into {combinations} value combinations; the maximum is "
                f"{self.max_condition_combinations} (settings.api.max_condition_combinations). "
                "List fewer values per column, or drop columns you do not need to control.",
                error_type="too_many_combinations",
            )

    def normalise_filters(self, table: str, filters: Mapping[str, Any] | None) -> dict[str, dict[str, Any]]:
        """Validate filters and coerce their values to the column type (400 on impossible filters).

        ``min``/``max`` apply to numeric columns only and must overlap the column's physical
        range; ``in``/``not_in`` values are mapped to canonical categories, booleans or numbers;
        a ``not_in`` list may not exclude every value of a categorical or boolean column.
        """
        if not filters:
            return {}
        raw = {
            column: spec.to_engine() if hasattr(spec, "to_engine") else dict(spec)
            for column, spec in filters.items()
        }
        try:
            validated = validate_filters(table, raw)
        except ValueError as exc:
            raise BadRequestError(str(exc), error_type="invalid_filters") from exc
        return {column: _normalise_filter(table, column, dict(spec)) for column, spec in validated.items()}

    def _check_model_id(self, generator: Any, table: str, model_id: str | None) -> None:
        """404 when ``model_id`` is not among the trained models (other models do exist)."""
        if not model_id:
            return
        try:
            entries = self._model_entries(generator, table)
        except ServiceError:
            return  # let the engine report the problem
        trained = [mid for mid in (model_id_of(entry) for entry in entries) if mid]
        if trained and model_id not in trained:
            raise NotFoundError(
                f"Model '{model_id}' is not trained for table '{table}'. Trained models: {', '.join(trained)}.",
                error_type="model_not_found",
            )

    def _generation_quality(
        self,
        data: pd.DataFrame,
        request: GenerateRequest,
        scenario: Scenario | None,
        conditions: Mapping[str, Sequence[Any]],
        filters: Mapping[str, Mapping[str, Any]],
        seed: int,
        warnings: list[str],
    ) -> dict[str, Any] | None:
        subset = build_reference_subset(
            request.table, scenario, conditions, filters, request.wards, request.calendar_months
        )
        try:
            return self.compute_quality(data, request.table, subset=subset, seed=seed, warnings=warnings)
        except ServiceError as exc:
            warnings.append(f"Quality evaluation skipped: {exc.detail}")
            return None

    def generate(self, request: GenerateRequest) -> GenerationOutcome:
        """Generate, validate, optionally evaluate and persist a synthetic dataset.

        Steps: resolve the scenario and normalise conditions/filters (400/404 early, including a
        cap on the number of condition combinations), run the engine within its time budget,
        reject an empty result (422 ``no_rows_generated``: nothing is stored), validate and score
        the output, describe it, store it, and build the response.
        """
        table = request.table
        scenario = self.resolve_scenario(request.scenario, table)
        request_warnings: list[str] = []
        conditions = self.normalise_conditions(table, request.conditions, request_warnings)
        self.check_condition_combinations(table, scenario, conditions, request.wards)
        filters = self.normalise_filters(table, request.filters)
        generator = self.get_generator()
        self._check_model_id(generator, table, request.model_id)

        data, report = self._run_engine(generator, request, scenario, conditions, filters)
        warnings = request_warnings + [str(item) for item in report.get("warnings") or []]
        if data.empty:
            raise self._no_rows_error(warnings)
        analysis = derived_view(table, data)
        seed = report.get("seed") if isinstance(report.get("seed"), int) else (request.seed or 0)
        validation_summary = self._validation_block(analysis, table, warnings) if request.run_validation else None
        quality = (
            self._generation_quality(analysis, request, scenario, conditions, filters, seed, warnings)
            if request.evaluate
            else None
        )

        echo = request.model_dump(mode="json", by_alias=True)
        echo.update(
            scenario=scenario.name if scenario else None,
            conditions=conditions or None,
            filters=filters or None,
            time_budget_seconds=self._time_budget(request),
        )
        persisted = bool(request.persist and self.persist_enabled)
        if request.persist and not self.persist_enabled:
            warnings.append("Dataset persistence is disabled on this server (settings.api.persist_datasets).")
        dataset_id = self.store.new_id()
        metadata: dict[str, Any] = {
            "dataset_id": dataset_id,
            "persisted": persisted,
            "created_at": utc_now_iso(),
            "table": table,
            "n_rows": int(len(data)),
            "model": self._model_block(generator, table, report, request.model_id),
            "scenario": self._scenario_block(scenario or SCENARIOS.get(str(report.get("scenario") or ""))),
            "request": echo,
            "generation": report,
            "validation": validation_summary,
            "quality": quality,
            "columns": describe_columns(table, data.columns),
            "warnings": _cap_warnings(_dedupe(warnings)),
            "links": self.dataset_links(dataset_id) if persisted else {},
        }
        if persisted:
            self._persist(data, metadata)
        response = {**metadata, "records": frame_to_records(data) if request.format == "json" else []}
        return GenerationOutcome(response=response, data=data)

    @staticmethod
    def _no_rows_error(warnings: Sequence[str]) -> UnprocessableError:
        """422 for a request whose conditions/filters no generated row could satisfy."""
        reasons = " ".join(_dedupe(warnings)[:3])
        return UnprocessableError(
            "No rows could be generated for this request, so no dataset was stored. "
            "Relax the scenario, conditions or filters (or raise time_budget_seconds / max_attempts)."
            + (f" Engine report: {reasons}" if reasons else ""),
            error_type="no_rows_generated",
        )

    def _time_budget(self, request: GenerateRequest) -> float:
        """The request's time budget (seconds): its own value or ``settings.api.time_budget_seconds``."""
        return float(request.time_budget_seconds or self.time_budget_seconds)

    def _run_engine(
        self,
        generator: Any,
        request: GenerateRequest,
        scenario: Scenario | None,
        conditions: Mapping[str, Sequence[Any]],
        filters: Mapping[str, Mapping[str, Any]],
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Call ``generator.generate`` (admission-controlled, serialised, errors mapped).

        Returns ``(rows, JSON-safe report)``. At most ``max_pending_generations`` requests run or
        wait at a time and a waiting request gives up after ``generation_wait_seconds``; both
        answer 503 ``server_busy`` with a ``Retry-After`` header.
        """
        engine = _engine_module()
        engine_request = engine.GenerationRequest(
            table=request.table,
            num_rows=request.num_rows,
            scenario=scenario.name if scenario else None,
            model_id=request.model_id,
            wards=list(request.wards) if request.wards else None,
            conditions=dict(conditions) or None,
            filters=dict(filters) or None,
            seed=request.seed,
            mode=request.mode,
            start_date=request.start_date.isoformat() if request.start_date else None,
            end_date=request.end_date.isoformat() if request.end_date else None,
            max_attempts=request.max_attempts,
            include_derived=request.include_derived,
            time_budget_seconds=self._time_budget(request),
        )
        logger.info(
            "Generating table=%s scenario=%s rows=%s mode=%s model=%s seed=%s budget=%ss",
            request.table, engine_request.scenario, request.num_rows, request.mode, request.model_id, request.seed,
            engine_request.time_budget_seconds,
        )
        retry_after = {"Retry-After": str(max(1, math.ceil(self.time_budget_seconds)))}
        if not self._pending.acquire(blocking=False):
            raise ServiceBusyError(
                f"{self.max_pending_generations} generation requests are already running or queued; retry shortly.",
                headers=retry_after,
            )
        try:
            if not self._generate_lock.acquire(timeout=self.generation_wait_seconds):
                raise ServiceBusyError(
                    f"The generation engine stayed busy for {self.generation_wait_seconds:g} s; retry shortly.",
                    headers=retry_after,
                )
            try:
                with map_engine_errors(engine, request.table, request.model_id):
                    result = generator.generate(engine_request)
            finally:
                self._generate_lock.release()
        finally:
            self._pending.release()
        data = getattr(result, "data", None)
        data = (data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)).reset_index(drop=True)
        return data, _compact_report(redact_paths(to_jsonable(dict(getattr(result, "report", None) or {}))))

    def _model_block(self, generator: Any, table: str, report: Mapping[str, Any], requested_id: str | None) -> dict[str, Any]:
        """``{id, synthesizer, sdv_version, fitted_at}``, preferring the card of the model that sampled."""
        model_id = report.get("model_id") or requested_id
        if not model_id:
            try:
                model_id = self._default_model_id(generator, table)
            except ServiceError:
                model_id = None
        reported = report.get("model")
        card = reported if isinstance(reported, Mapping) else self._model_card(generator, table, model_id)
        return model_summary(card, model_id)

    @staticmethod
    def _scenario_block(scenario: Scenario | None) -> dict[str, Any] | None:
        if scenario is None:
            return None
        return {"name": scenario.name, "title": scenario.title, "description": scenario.description, "category": scenario.category}

    def _persist(self, data: pd.DataFrame, metadata: dict[str, Any]) -> None:
        """Store a dataset; on I/O failure keep serving the response but flag it as not persisted."""
        try:
            self.store.save(data, metadata)
        except OSError as exc:
            logger.exception("Could not persist dataset %s", metadata.get("dataset_id"))
            metadata.update(persisted=False, links={})
            metadata["warnings"] = _dedupe([*metadata["warnings"], f"The dataset could not be stored ({type(exc).__name__})."])

    # ------------------------------------------------------- validate/evaluate
    def validate_records(self, request: ValidateRequest) -> dict[str, Any]:
        """Payload of ``POST /api/v1/validate``.

        The rules run on the recomputed derived values; rows whose *provided* derived values
        disagree with their definitions additionally fail the hard ``derived_values_consistent``
        check, so the verdict (``n_invalid_rows``, ``invalid_rows``, ``all_hard_rules_passed``)
        covers both.
        """
        prepared = prepare_records(request.table, request.records)
        report = self.run_validation(prepared.frame, request.table, include_soft=request.include_soft)
        summary = summarise_validation(report, [prepared.derived_check()], prepared.derived_mismatch)
        invalid_positions = hard_invalid_positions(report, len(prepared.frame), prepared.derived_mismatch)
        return {
            "table": request.table,
            "n_rows": int(len(prepared.frame)),
            "validation": summary,
            "normalisation": prepared.normalisation,
            "n_invalid_rows": int(len(invalid_positions)),
            "invalid_rows": [int(position) for position in invalid_positions[:MAX_INVALID_ROWS_LISTED]],
            "warnings": _dedupe(prepared.warnings),
            "records": frame_to_records(prepared.output) if request.return_records else None,
        }

    def _subset_from_metadata(self, metadata: Mapping[str, Any]) -> ReferenceSubset:
        """Reference subset matching how a stored dataset was generated."""
        request = metadata.get("request") or {}
        table = str(metadata.get("table"))
        months: list[int] = []
        if request.get("mode") == "calendar":
            try:
                months = months_in_period(
                    dt.date.fromisoformat(str(request.get("start_date"))), dt.date.fromisoformat(str(request.get("end_date")))
                )
            except ValueError:
                months = []
        return build_reference_subset(
            table,
            SCENARIOS.get(str(request.get("scenario") or "")),
            request.get("conditions") or {},
            request.get("filters") or {},
            request.get("wards"),
            months,
        )

    def evaluate(self, request: EvaluateRequest) -> dict[str, Any]:
        """Payload of ``POST /api/v1/evaluate`` (client records or a stored dataset)."""
        warnings: list[str] = []
        normalisation: dict[str, Any] | None = None
        prepared: PreparedRecords | None = None
        if request.dataset_id is not None:
            metadata = self.store.get(request.dataset_id)
            table = str(metadata.get("table"))
            if request.table is not None and request.table != table:
                raise BadRequestError(
                    f"Dataset '{request.dataset_id}' contains '{table}' records, not '{request.table}'.",
                    error_type="table_mismatch",
                )
            frame = derived_view(table, self.store.load(request.dataset_id, table))
            subset: ReferenceSubset | None = self._subset_from_metadata(metadata)
            source = {"type": "dataset", "dataset_id": request.dataset_id, "n_rows": int(len(frame))}
        else:
            table = str(request.table)
            prepared = prepare_records(table, request.records or [])
            frame, normalisation = prepared.frame, prepared.normalisation
            warnings.extend(prepared.warnings)
            scenario = self.resolve_scenario(request.scenario, table)
            subset = ReferenceSubset(scenario=scenario) if scenario else None
            source = {"type": "records", "dataset_id": None, "n_rows": int(len(frame))}
        quality = self.compute_quality(
            frame, table, split=request.reference_split, subset=subset, max_rows=request.max_rows,
            seed=request.seed, warnings=warnings,
        )
        validation_summary = (
            self._validation_block(frame, table, warnings, prepared) if request.include_validation else None
        )
        return {
            "table": table,
            "source": source,
            "quality": quality,
            "validation": validation_summary,
            "normalisation": normalisation,
            "warnings": _dedupe(warnings),
        }

    # --------------------------------------------------------------- datasets
    @staticmethod
    def dataset_links(dataset_id: str) -> dict[str, str]:
        base = f"{API_PREFIX}/datasets/{dataset_id}"
        return {
            "self": base,
            "download_csv": f"{base}/download?format=csv",
            "download_json": f"{base}/download?format=json",
        }

    def _dataset_summary(self, metadata: Mapping[str, Any]) -> dict[str, Any]:
        dataset_id = str(metadata.get("dataset_id"))
        generation = metadata.get("generation") or {}
        validation_block = metadata.get("validation") or {}
        quality = metadata.get("quality") or {}
        seed = generation.get("seed")
        return {
            "dataset_id": dataset_id,
            "created_at": metadata.get("created_at"),
            "table": metadata.get("table"),
            "scenario": (metadata.get("scenario") or {}).get("name"),
            "model_id": (metadata.get("model") or {}).get("id"),
            "mode": (metadata.get("request") or {}).get("mode"),
            "n_rows": metadata.get("n_rows"),
            "seed": seed if isinstance(seed, int) else None,
            "all_hard_rules_passed": validation_block.get("all_hard_rules_passed") if validation_block else None,
            "quality_score": quality.get("score") if quality else None,
            "size_bytes": int(metadata.get("size_bytes") or 0),
            "links": self.dataset_links(dataset_id),
        }

    def list_datasets(self, table: str | None = None, limit: int = 50) -> dict[str, Any]:
        """Payload of ``GET /api/v1/datasets`` (newest first)."""
        if table is not None:
            self.check_table(table)
        items = self.store.list_datasets(table)
        shown = items[: max(0, int(limit))]
        return {"total": len(items), "count": len(shown), "datasets": [self._dataset_summary(item) for item in shown]}

    def get_dataset(self, dataset_id: str) -> dict[str, Any]:
        """Payload of ``GET /api/v1/datasets/{id}`` (stored metadata)."""
        metadata = self.store.get(dataset_id)
        metadata["links"] = self.dataset_links(dataset_id)
        return metadata

    def dataset_csv(self, dataset_id: str) -> tuple[bytes, dict[str, str]]:
        """Stored CSV content and its ``X-BedSynth-*`` headers.

        The file is read completely under the store lock (datasets are at most a few MB), so no
        open handle outlives the request and a concurrent delete cannot fail on it.
        """
        content, metadata = self.store.read_data(dataset_id)
        return content, dataset_headers(metadata)

    def dataset_records(self, dataset_id: str) -> dict[str, Any]:
        """Payload of the JSON download of a stored dataset."""
        metadata = self.store.get(dataset_id)
        table = str(metadata.get("table"))
        frame = self.store.load(dataset_id, table)
        return {
            "dataset_id": dataset_id,
            "table": table,
            "created_at": metadata.get("created_at"),
            "n_rows": int(len(frame)),
            "columns": describe_columns(table, frame.columns),
            "records": frame_to_records(frame),
        }

    def delete_dataset(self, dataset_id: str) -> dict[str, Any]:
        """Payload of ``DELETE /api/v1/datasets/{id}``."""
        self.store.delete(dataset_id)
        return {"dataset_id": dataset_id, "deleted": True}


__all__ = [
    "API_PREFIX",
    "BadRequestError",
    "BedSynthService",
    "CSV_HEADER_NAMES",
    "ConflictError",
    "DERIVED_CHECK_IDS",
    "DatasetNotFoundError",
    "DatasetStore",
    "GenerationOutcome",
    "NotFoundError",
    "PreparedRecords",
    "ReferenceSubset",
    "ServiceBusyError",
    "ServiceError",
    "ServiceUnavailableError",
    "UnprocessableError",
    "build_reference_subset",
    "dataset_headers",
    "describe_columns",
    "frame_to_records",
    "hard_invalid_positions",
    "map_engine_errors",
    "model_info",
    "prepare_records",
    "redact_paths",
    "scrub_paths",
    "summarise_validation",
    "to_jsonable",
]
