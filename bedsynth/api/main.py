"""FastAPI application of BedSynth AI (Week 4 - docs/DESIGN.md section 10).

Run it with ``uvicorn bedsynth.api.main:app --port 8000`` or ``python scripts/04_serve_api.py``.

:func:`create_app` builds an isolated application around a
:class:`~bedsynth.api.service.BedSynthService`; tests inject a fake generator and a temporary
dataset store through it. The module-level ``app`` creates its generation engine lazily, so
importing this module needs neither trained models nor the SDV stack; when the server starts,
the engine loads in a background thread (``BEDSYNTH_API_PRELOAD=0`` defers it to the first
request that needs it). ``/health`` never waits for the engine: while it loads, ``/health``
answers at once with ``status`` 'degraded' and ``engine.state`` 'loading'.

This module is a thin HTTP adapter: request parsing and documentation (pydantic models in
:mod:`bedsynth.api.schemas`), routing, CORS, and the mapping of service errors to uniform JSON
error bodies ``{"detail": ..., "error_type": ...}``.
"""
from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any, AsyncIterator, Awaitable, Callable, Mapping, Sequence

from fastapi import APIRouter, Body, Depends, FastAPI, Query, Request
from fastapi import Path as PathParam
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.base import BaseHTTPMiddleware

from bedsynth import __version__
from bedsynth.api.schemas import (
    EVALUATE_EXAMPLES,
    GENERATE_EXAMPLES,
    VALIDATE_EXAMPLES,
    DatasetDetail,
    DatasetListResponse,
    DatasetRecords,
    DeleteResponse,
    DownloadFormat,
    ErrorResponse,
    EvaluateRequest,
    EvaluateResponse,
    GenerateRequest,
    GenerateResponse,
    HealthResponse,
    ModelsResponse,
    ScenarioInfo,
    ScenarioListResponse,
    ServiceInfo,
    TableSchemaResponse,
    TablesResponse,
    ValidateRequest,
    ValidateResponse,
)
from bedsynth.api.service import API_PREFIX, CSV_HEADER_NAMES, BedSynthService, NotFoundError, ServiceError
from bedsynth.config import get_settings

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent / "static"
UI_FILE = STATIC_DIR / "index.html"
#: Browser origins allowed by default: pages served from this machine on any port.
DEFAULT_CORS_ORIGIN_REGEX = r"https?://(localhost|127\.0\.0\.1|\[::1\])(:\d+)?"

API_DESCRIPTION = """
**BedSynth AI** generates realistic synthetic hospital operations data (no real records; privacy
assessed with distance-to-closest-record and disclosure checks) - ward-day bed occupancy and
patient ward stays - from SDV models trained on a reference extract.

* **Scenarios** - 26 named operating conditions (high occupancy, winter demand, emergency
  pressure, elective cancellations, delayed discharges, staffing shortage, ...) select a region of
  the learned distribution through conditional sampling and rejection filters.
* **Validation** - every generated row satisfies the hard operational rules (flow balance,
  capacity limits, staffing consistency, ...); soft-rule plausibility is reported per rule.
* **Evaluation** - an optional quick quality score against the matching reference data.
* **Datasets** - generations are persisted and can be listed, downloaded (CSV/JSON) and deleted.

Start with `GET /api/v1/scenarios`, then `POST /api/v1/generate`, or open the demo UI at
[/ui](/ui). Errors always return `{"detail": ..., "error_type": ...}`: 400 invalid request
(including impossible filters and more than `settings.api.max_condition_combinations` condition
combinations), 404 unknown table (in a URL), scenario, model or dataset, 409 dataset in use,
422 schema errors (including an unknown table in a request body) and requests no generated row
could satisfy (`no_rows_generated`), 503 no trained model, missing reference data or a busy
engine (`server_busy`, with `Retry-After`). Each generation runs within a time budget
(`time_budget_seconds`). All data is synthetic and non-clinical.
"""

TAGS_METADATA: list[dict[str, str]] = [
    {"name": "service", "description": "Service information, health and the interactive demo UI."},
    {"name": "catalogue", "description": "Tables, column schemas, operational rules, scenarios and trained models."},
    {"name": "generation", "description": "Generate validated synthetic records, optionally for a scenario."},
    {"name": "quality", "description": "Validate or evaluate client records and stored datasets."},
    {"name": "datasets", "description": "Persisted generations: list, inspect, download (CSV/JSON) and delete."},
]

_ERROR_DESCRIPTIONS: dict[int, str] = {
    400: "Invalid request (e.g. unknown condition value, impossible filter, too many condition combinations, "
         "scenario of another table, generation error).",
    404: "Unknown table (in the URL), scenario, model or dataset.",
    409: "The dataset is in use (e.g. a download is running) and cannot be deleted right now.",
    422: "The request does not match the schema (types, ranges, unknown fields, unknown table in the body), "
         "or no generated row could satisfy it (error_type 'no_rows_generated').",
    503: "Not ready or busy: no trained model, missing reference data, unavailable engine component, or too "
         "many generations in progress (error_type 'server_busy', see Retry-After).",
}

_HTTP_ERROR_TYPES: dict[int, str] = {404: "not_found", 405: "method_not_allowed", 413: "payload_too_large"}


def _errors(*codes: int) -> dict[int | str, dict[str, Any]]:
    """OpenAPI ``responses`` entries documenting the uniform error body."""
    return {code: {"model": ErrorResponse, "description": _ERROR_DESCRIPTIONS[code]} for code in codes}


def get_service(request: Request) -> BedSynthService:
    """Dependency: the service bound to the application handling the request."""
    return request.app.state.service


ServiceDep = Annotated[BedSynthService, Depends(get_service)]
DatasetIdParam = Annotated[str, PathParam(description="Dataset id returned by /generate, e.g. 'ds-0123456789ab'.")]
TableQuery = Annotated[str | None, Query(description="Restrict to one table: ward_daily or patient_stays.")]


# ---------------------------------------------------------------------------
# Service routes
# ---------------------------------------------------------------------------
service_router = APIRouter(tags=["service"])


def _endpoint_list(app: FastAPI) -> list[dict[str, str]]:
    """Documented endpoints, read from the (cached) OpenAPI schema so the list never drifts."""
    endpoints: list[dict[str, str]] = []
    for path, operations in app.openapi().get("paths", {}).items():
        for method, operation in operations.items():
            endpoints.append({"method": method.upper(), "path": path, "summary": str(operation.get("summary", ""))})
    return endpoints


@service_router.get("/", response_model=ServiceInfo, summary="Service information and links")
def root(request: Request, service: ServiceDep) -> dict[str, Any]:
    """Name, version, links to the docs and the demo UI, and the list of endpoints."""
    return service.service_info(_endpoint_list(request.app))


@service_router.get("/health", response_model=HealthResponse, summary="Health, trained models and reference data")
def health(service: ServiceDep) -> dict[str, Any]:
    """Liveness plus readiness details. Always 200 and answers at once: it never loads the generation
    engine or waits for it. `status` is 'degraded' while a table has no trained model, while the engine
    is still loading at start-up (`engine.state` 'loading') or after it failed to initialise."""
    return service.health()


@service_router.get(
    "/ui",
    response_class=HTMLResponse,
    summary="Interactive demo UI",
    responses={200: {"content": {"text/html": {}}, "description": "Single-page demo (no external assets)."}},
)
def ui() -> HTMLResponse:
    """Single-page demo to pick a scenario, generate, inspect validation/quality and manage datasets."""
    if not UI_FILE.is_file():
        raise NotFoundError("The demo UI is not installed (bedsynth/api/static/index.html is missing).")
    return HTMLResponse(UI_FILE.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# API v1
# ---------------------------------------------------------------------------
api_router = APIRouter(prefix=API_PREFIX)


@api_router.get("/tables", tags=["catalogue"], response_model=TablesResponse, summary="List tables")
def list_tables(service: ServiceDep) -> dict[str, Any]:
    """Tables with descriptions, column counts, scenario counts and their default model."""
    return service.list_tables()


@api_router.get(
    "/tables/{table}/schema",
    tags=["catalogue"],
    response_model=TableSchemaResponse,
    responses=_errors(404),
    summary="Column schema and operational rules of a table",
)
def table_schema(
    table: Annotated[str, PathParam(description="ward_daily or patient_stays.")], service: ServiceDep
) -> dict[str, Any]:
    """Every column (role, dtype, unit, vocabulary, bounds), the ward catalogue and the rule catalogue."""
    return service.table_schema(table)


@api_router.get(
    "/scenarios",
    tags=["catalogue"],
    response_model=ScenarioListResponse,
    responses=_errors(404),
    summary="Scenario catalogue",
)
def list_scenarios(
    service: ServiceDep,
    table: TableQuery = None,
    include_support: Annotated[
        bool, Query(description="Add how many reference rows match each scenario (needs the processed data).")
    ] = False,
) -> dict[str, Any]:
    """Named operating conditions with their SDV conditions and rejection filters."""
    return service.list_scenarios(table, include_support=include_support)


@api_router.get(
    "/scenarios/{name}",
    tags=["catalogue"],
    response_model=ScenarioInfo,
    responses=_errors(404),
    summary="One scenario (with reference-data support)",
)
def get_scenario(
    name: Annotated[str, PathParam(description="Scenario name, e.g. 'high_occupancy'.")], service: ServiceDep
) -> dict[str, Any]:
    """A scenario and how many reference rows match it (when the reference data is available)."""
    return service.get_scenario(name)


@api_router.get(
    "/models",
    tags=["catalogue"],
    response_model=ModelsResponse,
    responses=_errors(404, 503),
    summary="Trained models and defaults",
)
def list_models(service: ServiceDep, table: TableQuery = None) -> dict[str, Any]:
    """Registry view: trained experiments per table, their model cards, metrics and the default."""
    return service.list_models(table)


@api_router.post(
    "/generate",
    tags=["generation"],
    response_model=GenerateResponse,
    responses={
        200: {
            "description": "Generated dataset: JSON (format=json) or CSV with X-BedSynth-* headers (format=csv).",
            "content": {"text/csv": {"schema": {"type": "string"}}},
        },
        **_errors(400, 404, 422, 503),
    },
    summary="Generate synthetic records",
)
def generate(
    body: Annotated[GenerateRequest, Body(openapi_examples=GENERATE_EXAMPLES)], service: ServiceDep
) -> Any:
    """Generate rule-valid synthetic rows for a table, scenario, wards and conditions.

    The response echoes the request and carries the generation report (seed, acceptance rate,
    rejections per rule), the rule validation, the optional quality score, column descriptions,
    warnings and the records. With `format=csv` the records are returned as `text/csv`; the
    `X-BedSynth-Dataset-Id`, `-Rows`, `-Model` and `-Scenario` headers identify the dataset.
    """
    outcome = service.generate(body)
    if body.format == "csv":
        headers = outcome.headers()
        headers["Content-Disposition"] = f'attachment; filename="{outcome.response["dataset_id"]}.csv"'
        return Response(content=outcome.csv_text(), media_type="text/csv; charset=utf-8", headers=headers)
    return outcome.response


@api_router.post(
    "/validate",
    tags=["quality"],
    response_model=ValidateResponse,
    responses=_errors(400, 422, 503),
    summary="Validate client records against the operational rules",
)
def validate_records(
    body: Annotated[ValidateRequest, Body(openapi_examples=VALIDATE_EXAMPLES)], service: ServiceDep
) -> dict[str, Any]:
    """Normalise category aliases, derive missing derived columns and run the hard and soft rules."""
    return service.validate_records(body)


@api_router.post(
    "/evaluate",
    tags=["quality"],
    response_model=EvaluateResponse,
    responses=_errors(400, 404, 422, 503),
    summary="Quick quality of client records or a stored dataset",
)
def evaluate(
    body: Annotated[EvaluateRequest, Body(openapi_examples=EVALUATE_EXAMPLES)], service: ServiceDep
) -> dict[str, Any]:
    """Quick statistical quality (and rule validity) versus the reference data, or the subset of it that
    matches the scenario/conditions the records were generated for."""
    return service.evaluate(body)


@api_router.get(
    "/datasets",
    tags=["datasets"],
    response_model=DatasetListResponse,
    responses=_errors(404),
    summary="List persisted datasets",
)
def list_datasets(
    service: ServiceDep,
    table: TableQuery = None,
    limit: Annotated[int, Query(ge=1, le=500, description="Maximum datasets to return (newest first).")] = 50,
) -> dict[str, Any]:
    """Persisted generations, newest first."""
    return service.list_datasets(table, limit=limit)


@api_router.get(
    "/datasets/{dataset_id}",
    tags=["datasets"],
    response_model=DatasetDetail,
    responses=_errors(404),
    summary="Metadata of a persisted dataset",
)
def get_dataset(dataset_id: DatasetIdParam, service: ServiceDep) -> dict[str, Any]:
    """Everything stored with a dataset except its records (request, report, validation, quality)."""
    return service.get_dataset(dataset_id)


@api_router.get(
    "/datasets/{dataset_id}/download",
    tags=["datasets"],
    response_class=Response,
    response_model=None,
    responses={
        200: {
            "model": DatasetRecords,
            "description": "The stored records as CSV (with X-BedSynth-* headers) or JSON.",
            "content": {"text/csv": {"schema": {"type": "string"}}},
        },
        **_errors(404, 422),
    },
    summary="Download a persisted dataset",
)
def download_dataset(
    dataset_id: DatasetIdParam,
    service: ServiceDep,
    fmt: Annotated[DownloadFormat, Query(alias="format", description="csv (default) or json.")] = "csv",
) -> Response:
    """Download the stored records as an attachment."""
    if fmt == "csv":
        content, headers = service.dataset_csv(dataset_id)
        headers["Content-Disposition"] = f'attachment; filename="{dataset_id}.csv"'
        return Response(content=content, media_type="text/csv; charset=utf-8", headers=headers)
    payload = DatasetRecords.model_validate(service.dataset_records(dataset_id)).model_dump(mode="json")
    return JSONResponse(payload, headers={"Content-Disposition": f'attachment; filename="{dataset_id}.json"'})


@api_router.delete(
    "/datasets/{dataset_id}",
    tags=["datasets"],
    response_model=DeleteResponse,
    responses=_errors(404, 409),
    summary="Delete a persisted dataset",
)
def delete_dataset(dataset_id: DatasetIdParam, service: ServiceDep) -> dict[str, Any]:
    """Remove the dataset's CSV and metadata."""
    return service.delete_dataset(dataset_id)


# ---------------------------------------------------------------------------
# Error handling and middleware
# ---------------------------------------------------------------------------
def _error_response(
    status_code: int,
    detail: str,
    error_type: str,
    errors: list[dict[str, Any]] | None = None,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    body: dict[str, Any] = {"detail": detail, "error_type": error_type}
    if errors is not None:
        body["errors"] = errors
    return JSONResponse(status_code=status_code, content=body, headers=headers)


async def _service_error_handler(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, ServiceError)
    if exc.status_code >= 500:
        logger.warning("%s %s failed (%s): %s", request.method, request.url.path, exc.error_type, exc.detail)
    return _error_response(exc.status_code, exc.detail, exc.error_type, headers=exc.headers or None)


def _format_loc(loc: Sequence[Any], error_type: str = "") -> str:
    """Readable location of a validation error: ``num_rows``, ``filters.census_start.min``, ...

    For malformed JSON (``json_invalid``) pydantic puts the character offset of the problem in
    ``loc`` (``['body', 23]``); it is rendered as ``body (character 23)`` so it does not read
    like a field name.
    """
    if error_type == "json_invalid":
        offsets = [part for part in loc if isinstance(part, int) and not isinstance(part, bool)]
        return f"body (character {offsets[0]})" if offsets else "body"
    parts = [str(part) for part in loc if part != "body"]
    return ".".join(parts) or "body"


async def _validation_error_handler(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, RequestValidationError)
    errors = [
        {
            "loc": [part if isinstance(part, int) else str(part) for part in error.get("loc", ())],
            "msg": str(error.get("msg", "")).removeprefix("Value error, "),
            "type": str(error.get("type", "")),
        }
        for error in exc.errors()
    ]
    detail = "; ".join(f"{_format_loc(error['loc'], error['type'])}: {error['msg']}" for error in errors) or "Invalid request."
    return _error_response(422, detail, "validation_error", errors)


async def _http_error_handler(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, StarletteHTTPException)
    detail = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
    error_type = _HTTP_ERROR_TYPES.get(exc.status_code, "http_error")
    return _error_response(exc.status_code, detail, error_type, headers=getattr(exc, "headers", None))


async def _unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error("Unhandled error on %s %s", request.method, request.url.path, exc_info=exc)
    return _error_response(500, "Internal server error; see the server log.", "internal_error")


async def _timing_middleware(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
    """Add ``X-Process-Time-Ms`` to every response (handy when comparing scenarios/models).

    It also turns exceptions that no handler mapped into the uniform 500 body *inside* the CORS
    middleware: the catch-all handler of Starlette runs outside it, so its responses would
    carry no CORS headers and a browser client could not even read the error.
    """
    started = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception as exc:  # noqa: BLE001 - last line of defence, logged with the traceback
        response = await _unhandled_error_handler(request, exc)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.1f}"
    logger.debug("%s %s -> %s in %.1f ms", request.method, request.url.path, response.status_code, elapsed_ms)
    return response


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    service: BedSynthService = app.state.service
    logger.info("%s %s started; datasets are stored in %s", service.title, __version__, service.store.base_dir)
    if app.state.preload_engine:
        service.start_preload()  # background thread; /health reports engine.state 'loading' meanwhile
    yield


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------
def create_app(
    generator: Any | None = None,
    store_dir: Path | str | None = None,
    *,
    registry: Any | None = None,
    models_dir: Path | str | None = None,
    processed_dir: Path | str | None = None,
    generator_factory: Callable[[], Any] | None = None,
    cors_origins: Sequence[str] | None = None,
    preload_engine: bool = False,
    settings: Mapping[str, Any] | None = None,
) -> FastAPI:
    """Build the BedSynth AI FastAPI application.

    Parameters
    ----------
    generator:
        Generation engine to use (``SyntheticDataGenerator`` or a test double). When omitted the
        real engine is created lazily on the first request that needs it.
    store_dir:
        Directory of persisted datasets (default ``<synthetic_dir>/api``).
    registry, models_dir, processed_dir, generator_factory:
        Optional overrides forwarded to :class:`~bedsynth.api.service.BedSynthService`.
    cors_origins:
        Allowed CORS origins (default ``settings.api.cors_origins``). Browser pages on localhost /
        127.0.0.1 (any port) are always allowed through ``settings.api.cors_origin_regex``; other
        websites are not, so a page a user happens to visit cannot list, delete or mass-generate
        datasets on a locally running API. List ``"*"`` explicitly to allow every origin.
    preload_engine:
        Start loading the engine in a background thread when the app starts (``/health`` keeps
        answering at once and reports ``engine.state`` 'loading' until it is ready).
    settings:
        Full settings dictionary (default ``config/settings.yaml``); its ``api`` section sets the
        limits, time budget, admission control and CORS policy (tests override it here).
    """
    settings = dict(settings) if settings is not None else get_settings()
    api_settings = settings.get("api") or {}
    service = BedSynthService(
        generator,
        store_dir=store_dir,
        registry=registry,
        models_dir=models_dir,
        processed_dir=processed_dir,
        generator_factory=generator_factory,
        settings=settings,
    )
    app = FastAPI(
        title=service.title,
        version=__version__,
        description=API_DESCRIPTION,
        openapi_tags=TAGS_METADATA,
        lifespan=_lifespan,
    )
    app.state.service = service
    app.state.preload_engine = preload_engine

    origins = list(cors_origins if cors_origins is not None else api_settings.get("cors_origins") or [])
    origin_regex = api_settings.get("cors_origin_regex", DEFAULT_CORS_ORIGIN_REGEX) or None
    app.add_middleware(BaseHTTPMiddleware, dispatch=_timing_middleware)
    app.add_middleware(
        CORSMiddleware,  # added last = outermost, so CORS headers are present on error responses too
        allow_origins=origins,
        allow_origin_regex=origin_regex,
        allow_credentials=False,
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=[*CSV_HEADER_NAMES, "Content-Disposition", "X-Process-Time-Ms", "Retry-After"],
    )
    app.add_exception_handler(ServiceError, _service_error_handler)
    app.add_exception_handler(RequestValidationError, _validation_error_handler)
    app.add_exception_handler(StarletteHTTPException, _http_error_handler)
    app.add_exception_handler(Exception, _unhandled_error_handler)
    app.include_router(service_router)
    app.include_router(api_router)
    return app


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    return default if value is None else value.strip().lower() in {"1", "true", "yes", "on"}


#: Application used by ``uvicorn bedsynth.api.main:app``. Importing it never loads the engine. When
#: the server starts, a background thread imports SDV and creates the engine (``BEDSYNTH_API_PRELOAD=0``
#: defers this to the first request that needs it: /generate, /api/v1/models or /api/v1/tables, which
#: wait until the engine is ready). /health never creates the engine or waits for it: until the engine
#: is ready it reads the trained models from registry.json without importing SDV, and while the
#: engine is loading it answers at once with status 'degraded' and engine.state 'loading'.
app = create_app(preload_engine=_env_flag("BEDSYNTH_API_PRELOAD", default=True))

__all__ = ["app", "create_app", "get_service"]
