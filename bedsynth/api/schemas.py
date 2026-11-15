"""Pydantic v2 request/response models of the BedSynth AI REST API.

The models document the API contract in the generated OpenAPI schema (``/docs``): every
field carries a description, request bodies carry examples covering several scenario
families, and numeric limits come from ``config/settings.yaml`` (``api`` section).

Responses are produced by :mod:`bedsynth.api.service` as plain JSON-safe dictionaries; the
models below validate and document them when FastAPI serialises the response.
"""
from __future__ import annotations

import datetime as dt
import math
from typing import Any, Iterable, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator, model_validator

from bedsynth.config import section
from bedsynth.schema import CATEGORY_ALIASES, DAYS_OF_WEEK, WARD_IDS, normalize_category

# ---------------------------------------------------------------------------
# Limits (settings.yaml -> api) and shared constrained types
# ---------------------------------------------------------------------------
_API_SETTINGS: dict[str, Any] = section("api") or {}

MAX_ROWS: int = int(_API_SETTINGS.get("max_rows", 10_000))
MAX_CALENDAR_DAYS: int = int(_API_SETTINGS.get("max_calendar_days", 1100))
EVALUATE_MAX_ROWS: int = int(_API_SETTINGS.get("evaluate_max_rows", 5000))
#: Default and maximum time budget (seconds) of one generation request (the engine stops
#: sampling when it runs out and reports the shortfall, so no request holds the engine longer).
TIME_BUDGET_SECONDS: float = float(_API_SETTINGS.get("time_budget_seconds", 60))
MAX_TIME_BUDGET_SECONDS: float = float(_API_SETTINGS.get("max_time_budget_seconds", 120))
MAX_ATTEMPTS: int = int(_API_SETTINGS.get("max_attempts", 100))
MAX_SEED: int = 2**32 - 1
MIN_EVALUATE_ROWS: int = 20
#: Calendar dates the engine can represent (pandas nanosecond timestamps end in 2262).
MIN_DATE: dt.date = dt.date(1900, 1, 1)
MAX_DATE: dt.date = dt.date(2200, 12, 31)

#: Experiment ids are used to build model paths, so only a safe character set is accepted.
MODEL_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_\-]{0,63}$"
DATASET_ID_REGEX = r"^ds-[0-9a-f]{12}$"

TableName = Literal["ward_daily", "patient_stays"]
GenerationMode = Literal["sample", "calendar"]
OutputFormat = Literal["json", "csv"]
DownloadFormat = Literal["csv", "json"]
ReferenceSplit = Literal["train", "holdout"]

ConditionScalar = StrictInt | str
ConditionValue = ConditionScalar | list[ConditionScalar]
FilterValue = bool | int | float | str


def canonical_category(column: str, value: Any, categories: Iterable[Any] | None) -> Any | None:
    """Return the canonical vocabulary value for ``value`` or ``None`` when it is unknown.

    Accepts exact values, the aliases of :data:`bedsynth.schema.CATEGORY_ALIASES`
    (e.g. ``"icu"`` / ``"Intensive Care Unit"`` -> ``"ICU"``, ``"monday"`` -> ``"Mon"``),
    case-insensitive matches and numeric strings for integer-coded columns such as ``month``.
    Booleans and missing values are never valid categories.
    """
    if value is None or isinstance(value, bool) or (isinstance(value, float) and math.isnan(value)):
        return None
    allowed = list(categories or [])
    if not allowed:
        return value
    if all(isinstance(item, int) and not isinstance(item, bool) for item in allowed):
        try:
            number = float(str(value).strip())
        except ValueError:
            return None
        if not number.is_integer() or int(number) not in allowed:
            return None
        return int(number)
    if value in allowed:
        return value
    alias_column = column
    if column not in CATEGORY_ALIASES and tuple(allowed) == DAYS_OF_WEEK:
        alias_column = "day_of_week"  # e.g. discharge_day_of_week shares the weekday aliases
    alias = normalize_category(alias_column, value)
    if alias in allowed:
        return alias
    lookup = {str(item).lower(): item for item in allowed}
    return lookup.get(str(value).strip().lower())


def months_in_period(start: dt.date | None, end: dt.date | None) -> list[int]:
    """Distinct calendar months (in order of appearance) covered by ``start..end`` inclusive."""
    if start is None or end is None or end < start:
        return []
    months: list[int] = []
    day = start.replace(day=1)
    while day <= end and len(months) < 12:
        months.append(day.month)
        day = (day + dt.timedelta(days=32)).replace(day=1)
    return months


class APIModel(BaseModel):
    """Base model: allows field names such as ``model_id`` and population by alias or name."""

    model_config = ConfigDict(protected_namespaces=(), validate_by_name=True, validate_by_alias=True)


class RequestModel(APIModel):
    """Base for request bodies: only the public field names are accepted.

    Unknown fields are rejected so typos surface as 422 errors, and a field with a public alias
    accepts the alias only (``validate``, ``in``), never the internal attribute name
    (``run_validation``, ``in_``).
    """

    model_config = ConfigDict(protected_namespaces=(), extra="forbid", validate_by_name=False, validate_by_alias=True)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------
class ErrorDetail(APIModel):
    """One field-level problem of a 422 response."""

    loc: list[str | int] = Field(description="Location of the problem, e.g. ['body', 'num_rows'].")
    msg: str = Field(description="Human-readable message.")
    type: str = Field(description="Machine-readable pydantic error type.")


class ErrorResponse(APIModel):
    """Uniform error body returned for every 4xx/5xx response."""

    detail: str = Field(description="Human-readable explanation of the error.")
    error_type: str = Field(
        description="Stable machine-readable category, e.g. 'validation_error', 'scenario_not_found', "
        "'generation_error', 'model_unavailable'."
    )
    errors: list[ErrorDetail] | None = Field(None, description="Field-level problems (422 responses only).")

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {"detail": "Unknown scenario 'icu_meltdown'.", "error_type": "scenario_not_found"},
            ]
        }
    )


# ---------------------------------------------------------------------------
# Generation request
# ---------------------------------------------------------------------------
class FilterSpec(RequestModel):
    """Acceptance criteria applied to the *derived* table after sampling (bounds are inclusive)."""

    min: int | float | None = Field(None, description="Inclusive lower bound (numeric columns, finite).")
    max: int | float | None = Field(None, description="Inclusive upper bound (numeric columns, finite).")
    in_: list[FilterValue] | None = Field(
        None, alias="in", min_length=1, description="Accepted values (categorical or boolean columns)."
    )
    not_in: list[FilterValue] | None = Field(None, min_length=1, description="Rejected values.")

    @field_validator("min", "max", mode="before")
    @classmethod
    def _finite_number(cls, value: Any) -> Any:
        """Bounds are finite numbers: booleans, NaN and infinity are rejected (422)."""
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("Filter bounds must be numbers.")
        if not math.isfinite(float(value)):
            raise ValueError("Filter bounds must be finite numbers (no NaN or infinity).")
        return value

    @model_validator(mode="after")
    def _check_bounds(self) -> "FilterSpec":
        if self.min is None and self.max is None and self.in_ is None and self.not_in is None:
            raise ValueError("A filter needs at least one of 'min', 'max', 'in' or 'not_in'.")
        if self.min is not None and self.max is not None and self.min > self.max:
            raise ValueError(f"Filter 'min' ({self.min}) is greater than 'max' ({self.max}).")
        return self

    def to_engine(self) -> dict[str, Any]:
        """Filter grammar of :mod:`bedsynth.scenarios` (``{"min", "max", "in", "not_in"}``)."""
        return self.model_dump(by_alias=True, exclude_none=True)


class GenerateRequest(RequestModel):
    """Body of ``POST /api/v1/generate``."""

    table: TableName = Field("ward_daily", description="Table to synthesise.")
    num_rows: int = Field(
        100,
        ge=1,
        le=MAX_ROWS,
        strict=True,
        description=f"Rows to generate in sample mode (1..{MAX_ROWS}, a JSON integer). Ignored in calendar "
        "mode, where the output has one row per ward per day.",
    )
    scenario: str | None = Field(
        None,
        max_length=64,
        description="Scenario name from GET /api/v1/scenarios; it must belong to `table`. "
        "Omit for an unconditional (baseline) sample.",
        examples=["high_occupancy"],
    )
    model_id: str | None = Field(
        None,
        pattern=MODEL_ID_PATTERN,
        description="Trained experiment id (GET /api/v1/models). Omit to use the registry default.",
        examples=["gc_tuned"],
    )
    wards: list[str] | None = Field(
        None,
        min_length=1,
        max_length=len(WARD_IDS),
        description="Restrict generation to these wards. Codes or common aliases are accepted "
        f"(e.g. 'icu', 'General Medicine'). Known wards: {', '.join(WARD_IDS)}.",
        examples=[["AMU", "GMED"]],
    )
    conditions: dict[str, ConditionValue] | None = Field(
        None,
        description="Extra conditions on categorical training columns, intersected with the "
        "scenario's conditions: {column: value | [values]}. Aliases are normalised "
        "(e.g. 'monday' -> 'Mon'). A column listing every value is dropped (same as no condition); "
        "after merging with the scenario and `wards`, the product of the value counts may not exceed "
        "settings.api.max_condition_combinations (default 1000; 400 too_many_combinations).",
        examples=[{"occupancy_level": ["high", "critical"], "month": [12, 1, 2]}],
    )
    filters: dict[str, FilterSpec] | None = Field(
        None,
        description="Extra acceptance filters on training or derived columns, merged with the "
        "scenario's filters (bounds are tightened, value lists intersected). min/max apply to numeric "
        "columns only and must overlap the column's range; in/not_in values are coerced to the column type. "
        "Filters no row can satisfy are rejected with 400 invalid_filters.",
        examples=[{"emergency_boarders": {"min": 1}}],
    )
    seed: int | None = Field(
        None,
        ge=0,
        le=MAX_SEED,
        strict=True,
        description="Random seed (a JSON integer). The same seed, model and request reproduce the same "
        "records when the request finishes within its time budget; when omitted a seed is drawn and "
        "reported in `generation.seed`.",
    )
    max_attempts: int | None = Field(
        None,
        ge=1,
        le=MAX_ATTEMPTS,
        strict=True,
        description=f"Maximum rejection-sampling rounds (1..{MAX_ATTEMPTS}); omit for the engine default "
        "(settings.generation.max_attempts). Raise it for rare scenarios.",
    )
    time_budget_seconds: float | None = Field(
        None,
        gt=0,
        le=MAX_TIME_BUDGET_SECONDS,
        allow_inf_nan=False,
        strict=True,
        description=f"Time budget of the request in seconds (default {TIME_BUDGET_SECONDS:g}, maximum "
        f"{MAX_TIME_BUDGET_SECONDS:g}). Sampling stops when it runs out and the shortfall is reported.",
    )
    mode: GenerationMode = Field(
        "sample",
        description="'sample' draws `num_rows` rows; 'calendar' (ward_daily only) produces one row "
        "per ward per date between `start_date` and `end_date`.",
    )
    start_date: dt.date | None = Field(None, description="First date (ISO) of the calendar (calendar mode).")
    end_date: dt.date | None = Field(None, description="Last date (ISO, inclusive) of the calendar (calendar mode).")
    include_derived: bool = Field(
        True, description="Include derived columns (occupancy_rate, census_end, los_hours, ...) in the output."
    )
    run_validation: bool = Field(
        True,
        alias="validate",
        description="Run the operational rule checks (hard + soft) on the output and report them.",
    )
    evaluate: bool = Field(
        False,
        description="Compute a quick quality score against the reference data (matching scenario "
        "subset when large enough). Adds roughly a second.",
    )
    format: OutputFormat = Field(
        "json",
        description="'json' returns records and reports; 'csv' returns text/csv with X-BedSynth-* "
        "headers (reports are stored with the dataset when persisted).",
    )
    persist: bool = Field(
        True, description="Store the dataset (CSV + metadata) for later download via /api/v1/datasets."
    )

    @field_validator("scenario")
    @classmethod
    def _normalise_scenario(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip().lower()
        return value or None

    @field_validator("start_date", "end_date")
    @classmethod
    def _date_in_supported_range(cls, value: dt.date | None) -> dt.date | None:
        if value is not None and not MIN_DATE <= value <= MAX_DATE:
            raise ValueError(f"Calendar dates must lie between {MIN_DATE.isoformat()} and {MAX_DATE.isoformat()}.")
        return value

    @field_validator("wards")
    @classmethod
    def _normalise_wards(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        wards: list[str] = []
        for raw in value:
            ward = canonical_category("ward_id", raw, WARD_IDS)
            if ward is None:
                raise ValueError(f"Unknown ward '{raw}'. Known wards: {', '.join(WARD_IDS)}.")
            if ward not in wards:
                wards.append(ward)
        return wards

    @model_validator(mode="after")
    def _check_mode(self) -> "GenerateRequest":
        if self.mode == "calendar":
            if self.table != "ward_daily":
                raise ValueError("Calendar mode is only available for the ward_daily table.")
            if self.start_date is None or self.end_date is None:
                raise ValueError("Calendar mode requires both start_date and end_date (ISO dates).")
            if self.end_date < self.start_date:
                raise ValueError("end_date must be on or after start_date.")
            days = self.calendar_days
            if days > MAX_CALENDAR_DAYS:
                raise ValueError(f"The calendar spans {days} days; the maximum is {MAX_CALENDAR_DAYS}.")
            rows = days * len(self.wards or WARD_IDS)
            if rows > MAX_ROWS:
                raise ValueError(
                    f"Calendar mode would generate {rows} rows ({len(self.wards or WARD_IDS)} wards x "
                    f"{days} days); the maximum is {MAX_ROWS}. Shorten the period or select fewer wards."
                )
        elif self.start_date is not None or self.end_date is not None:
            raise ValueError("start_date and end_date are only used in calendar mode (set mode='calendar').")
        return self

    @property
    def calendar_days(self) -> int:
        """Number of calendar days requested (0 outside calendar mode)."""
        if self.start_date is None or self.end_date is None:
            return 0
        return (self.end_date - self.start_date).days + 1

    @property
    def calendar_months(self) -> list[int]:
        """Distinct months covered by the calendar period (empty outside calendar mode)."""
        return months_in_period(self.start_date, self.end_date)


# ---------------------------------------------------------------------------
# Shared response fragments
# ---------------------------------------------------------------------------
class ModelSummary(APIModel):
    """The trained model that produced a dataset."""

    id: str | None = Field(description="Experiment id, e.g. 'gc_tuned'.")
    synthesizer: str | None = Field(
        None, description="SDV synthesizer class, e.g. 'GaussianCopulaSynthesizer', 'CTGANSynthesizer'."
    )
    sdv_version: str | None = Field(None, description="SDV version the model was fitted with.")
    fitted_at: str | None = Field(None, description="ISO timestamp of the fit.")


class ScenarioSummary(APIModel):
    """Scenario that shaped a generation."""

    name: str
    title: str
    description: str
    category: str | None = None


class ColumnInfo(APIModel):
    """Description of one output column (from the schema catalogue)."""

    name: str
    dtype: str = Field(description="int | float | str | bool | date | datetime")
    role: str = Field(description="key | time | train (learned by SDV) | derived (recomputed) | extra")
    unit: str | None = None
    description: str = ""


class RuleResult(APIModel):
    """Outcome of one operational rule."""

    rule_id: str = Field(description="Rule identifier, e.g. 'WD-H03' (H = hard, S = soft).")
    name: str
    severity: str = Field(description="'hard' (operationally impossible) or 'soft' (implausible).")
    description: str = ""
    columns: list[str] = Field(default_factory=list)
    violations: int = Field(description="Rows violating the rule.")
    violation_rate: float = Field(description="Violating rows / rows checked.")
    passed: bool
    example_rows: list[int] = Field(
        default_factory=list, description="Up to five 0-based positions of violating rows in `records`."
    )


class RuleGroupSummary(APIModel):
    """Aggregate over the hard or soft rules."""

    checked: int
    passed: int
    failed: int
    violations: int = Field(description="Total violations summed over the rules of this group.")


class ValidationSummary(APIModel):
    """Operational rule validation of a set of records (hard rules = impossible rows)."""

    n_rows: int
    n_hard_valid: int = Field(description="Rows passing every hard rule.")
    hard_valid_rate: float | None
    all_hard_rules_passed: bool | None = Field(
        description="True when every row passes every hard rule; null when there are no rows to judge."
    )
    hard_rules: RuleGroupSummary
    soft_rules: RuleGroupSummary
    rules: list[RuleResult]


class ReferenceInfo(APIModel):
    """Which reference ("real") data a quality score was computed against."""

    split: str = Field(description="Processed split used as reference: 'train' or 'holdout'.")
    subset: Literal["matching", "full"] = Field(
        description="'matching' when only the reference rows matching the request's scenario, conditions, "
        "filters, wards or calendar months were used (at least 20 rows); 'full' otherwise."
    )
    reference_rows: int = Field(description="Reference rows available in the chosen subset.")
    rows_compared: int = Field(description="Reference rows compared (after subsampling).")
    synthetic_rows_compared: int = Field(description="Synthetic rows compared (after subsampling).")


class QualityResult(APIModel):
    """Quick statistical quality (column shapes, pair trends, validity) versus reference data."""

    score: float | None = Field(description="Overall quick-quality score in [0, 1].")
    reference: ReferenceInfo
    metrics: dict[str, Any] = Field(description="Full output of bedsynth.evaluation.quick_quality.")


# ---------------------------------------------------------------------------
# Generation response & persisted datasets
# ---------------------------------------------------------------------------
class GenerationMetadata(APIModel):
    """Everything known about a generated dataset except its records."""

    dataset_id: str = Field(description="Dataset identifier ('ds-' + 12 hex characters).")
    persisted: bool = Field(description="True when the dataset was stored and can be downloaded later.")
    created_at: str = Field(description="UTC creation timestamp (ISO 8601).")
    table: str
    n_rows: int = Field(description="Number of generated records.")
    model: ModelSummary
    scenario: ScenarioSummary | None
    request: dict[str, Any] = Field(description="Normalised echo of the request.")
    generation: dict[str, Any] = Field(
        description="Generation report: seed, attempts, sampled/accepted rows, acceptance_rate, "
        "rejections per rule and filter, condition allocation, elapsed_seconds, warnings."
    )
    validation: ValidationSummary | None = Field(None, description="Rule validation (null when validate=false).")
    quality: QualityResult | None = Field(None, description="Quick quality (only when evaluate=true).")
    columns: list[ColumnInfo]
    warnings: list[str] = Field(default_factory=list)
    links: dict[str, str] = Field(default_factory=dict, description="Related endpoints for persisted datasets.")


class GenerateResponse(GenerationMetadata):
    """Response of ``POST /api/v1/generate`` with ``format=json``."""

    records: list[dict[str, Any]] = Field(description="Generated rows (NaN encoded as null).")


class DatasetDetail(GenerationMetadata):
    """Stored metadata of a persisted dataset."""

    size_bytes: int = Field(0, description="Size of the stored CSV file.")


class DatasetSummary(APIModel):
    """One row of the persisted-dataset listing."""

    dataset_id: str
    created_at: str | None
    table: str | None
    scenario: str | None
    model_id: str | None
    mode: str | None
    n_rows: int | None
    seed: int | None
    all_hard_rules_passed: bool | None
    quality_score: float | None
    size_bytes: int
    links: dict[str, str]


class DatasetListResponse(APIModel):
    """Persisted datasets, newest first."""

    total: int = Field(description="Datasets matching the filter.")
    count: int = Field(description="Datasets returned (bounded by `limit`).")
    datasets: list[DatasetSummary]


class DatasetRecords(APIModel):
    """JSON download of a persisted dataset."""

    dataset_id: str
    table: str
    created_at: str | None
    n_rows: int
    columns: list[ColumnInfo]
    records: list[dict[str, Any]]


class DeleteResponse(APIModel):
    """Result of deleting a dataset."""

    dataset_id: str
    deleted: bool


# ---------------------------------------------------------------------------
# Validation & evaluation of client records
# ---------------------------------------------------------------------------
class ValidateRequest(RequestModel):
    """Body of ``POST /api/v1/validate``."""

    table: TableName = Field(description="Table the records belong to.")
    records: list[dict[str, Any]] = Field(
        min_length=1,
        max_length=MAX_ROWS,
        description="Rows to validate. Every training column is required; derived columns are computed when "
        "missing. Provided derived values are checked against their definitions (a disagreement makes the row "
        "hard-invalid) and the operational rules always run on the recomputed values.",
    )
    include_soft: bool = Field(True, description="Also evaluate soft (plausibility) rules.")
    return_records: bool = Field(
        False, description="Return the normalised records with derived columns completed."
    )


class NormalisationSummary(APIModel):
    """What the API changed or found while preparing client records."""

    values_normalised: dict[str, int] = Field(
        default_factory=dict, description="Category values mapped to canonical codes, per column (e.g. 'amu' -> 'AMU')."
    )
    unknown_values: dict[str, int] = Field(
        default_factory=dict, description="Values outside the column vocabulary, per column (kept; they fail validation)."
    )
    invalid_numbers: dict[str, int] = Field(
        default_factory=dict, description="Non-numeric values in numeric columns, per column (treated as missing)."
    )
    unknown_columns: list[str] = Field(default_factory=list, description="Columns not in the schema (ignored by the rules).")
    derived_computed: list[str] = Field(default_factory=list, description="Derived columns computed by the API.")
    derived_mismatches: dict[str, int] = Field(
        default_factory=dict,
        description="Provided derived values that disagree with their definition (e.g. census_end flow balance), per "
        "column. Such rows fail the hard 'derived_values_consistent' check; the rules run on the recomputed values.",
    )
    derived_mismatch_rows: dict[str, list[int]] = Field(
        default_factory=dict,
        description="0-based positions (first 100 per column) of the records whose provided derived value disagrees.",
    )


class ValidateResponse(APIModel):
    """Response of ``POST /api/v1/validate``."""

    table: str
    n_rows: int
    validation: ValidationSummary
    normalisation: NormalisationSummary
    n_invalid_rows: int = Field(description="Rows failing at least one hard rule.")
    invalid_rows: list[int] = Field(description="0-based positions of hard-invalid rows (first 100).")
    warnings: list[str] = Field(default_factory=list)
    records: list[dict[str, Any]] | None = Field(None, description="Normalised records (return_records=true).")


class EvaluateRequest(RequestModel):
    """Body of ``POST /api/v1/evaluate``: client records *or* a stored dataset."""

    table: TableName | None = Field(None, description="Table of `records` (required with records; optional with dataset_id).")
    records: list[dict[str, Any]] | None = Field(
        None, min_length=MIN_EVALUATE_ROWS, max_length=MAX_ROWS, description="Synthetic rows to evaluate."
    )
    dataset_id: str | None = Field(None, pattern=DATASET_ID_REGEX, description="A persisted dataset to evaluate instead of records.")
    scenario: str | None = Field(
        None,
        max_length=64,
        description="Compare against the reference rows of this scenario (records only; stored "
        "datasets use the scenario and conditions they were generated with).",
    )
    reference_split: ReferenceSplit = Field("train", description="Reference split: 'train' (what the models learned) or 'holdout'.")
    max_rows: int = Field(
        EVALUATE_MAX_ROWS,
        ge=MIN_EVALUATE_ROWS,
        le=EVALUATE_MAX_ROWS,
        description="Subsample both sides to at most this many rows (keeps evaluation fast).",
    )
    seed: int = Field(0, ge=0, le=MAX_SEED, description="Seed for subsampling.")
    include_validation: bool = Field(True, description="Also run the operational rule validation.")

    @field_validator("scenario")
    @classmethod
    def _normalise_scenario(cls, value: str | None) -> str | None:
        return (value.strip().lower() or None) if value is not None else None

    @model_validator(mode="after")
    def _check_source(self) -> "EvaluateRequest":
        if (self.records is None) == (self.dataset_id is None):
            raise ValueError("Provide exactly one of 'records' or 'dataset_id'.")
        if self.records is not None and self.table is None:
            raise ValueError("'table' is required when evaluating records.")
        if self.dataset_id is not None and self.scenario is not None:
            raise ValueError("'scenario' only applies to records; stored datasets use their own scenario.")
        return self


class EvaluationSource(APIModel):
    """What was evaluated."""

    type: Literal["records", "dataset"]
    dataset_id: str | None = None
    n_rows: int


class EvaluateResponse(APIModel):
    """Response of ``POST /api/v1/evaluate``."""

    table: str
    source: EvaluationSource
    quality: QualityResult
    validation: ValidationSummary | None = None
    normalisation: NormalisationSummary | None = None
    warnings: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Service, catalogue and model responses
# ---------------------------------------------------------------------------
class EndpointInfo(APIModel):
    method: str
    path: str
    summary: str


class ServiceInfo(APIModel):
    """Response of ``GET /``."""

    name: str
    title: str
    version: str
    description: str
    links: dict[str, str]
    endpoints: list[EndpointInfo]


EngineState = Literal["not_loaded", "loading", "ready", "failed"]


class EngineStatus(APIModel):
    available: bool = Field(
        description="False when the generation engine failed to initialise or cannot list its models; true while "
        "it is loading or not loaded yet."
    )
    loaded: bool = Field(
        description="True when the engine had been initialised at the time of the call (it loads in a background "
        "thread at start-up, or on the first request that needs it when preloading is off)."
    )
    state: EngineState = Field(
        description="'ready' (initialised), 'loading' (being initialised; requests that need the engine wait), "
        "'not_loaded' (loads on the first request that needs it) or 'failed' (the last initialisation failed; "
        "the next request that needs the engine retries)."
    )
    detail: str | None = Field(
        None, description="Why the engine is not ready (loading, not loaded yet, or the initialisation error type)."
    )


class TableModelsStatus(APIModel):
    trained: list[str] = Field(description="Trained experiment ids.")
    default: str | None = Field(description="Model used when a request omits model_id.")


class DatasetStoreStatus(APIModel):
    count: int
    persistence_enabled: bool


class HealthResponse(APIModel):
    """Response of ``GET /health``: answered at once, without loading or waiting for the engine."""

    status: Literal["ok", "degraded"] = Field(
        description="'ok' when every table has a trained model and the engine is ready (or loads on first use); "
        "'degraded' while a table has no trained model, while the engine is still loading at start-up "
        "(engine.state 'loading') or after it failed to initialise (see messages)."
    )
    version: str
    sdv_version: str | None
    engine: EngineStatus
    models: dict[str, TableModelsStatus] = Field(
        description="Trained models per table: listed by the engine once it is ready, read from the model "
        "registry file (registry.json) before that."
    )
    trained_tables: list[str]
    reference_data: dict[str, bool] = Field(description="Whether the processed train split exists, per table.")
    datasets: DatasetStoreStatus
    messages: list[str] = Field(default_factory=list, description="What to run to fix a degraded status.")
    checked_at: str


class TableInfo(APIModel):
    """One entry of ``GET /api/v1/tables``."""

    name: str
    title: str
    description: str
    grain: str
    key_column: str
    n_columns: int = Field(description="Columns of a generated dataset (key + training + derived).")
    n_training_columns: int
    n_derived_columns: int
    n_scenarios: int
    conditionable_columns: list[str]
    supports_calendar_mode: bool
    default_model: str | None
    trained_models: list[str]
    reference_data_available: bool
    links: dict[str, str]


class TablesResponse(APIModel):
    tables: list[TableInfo]


class ColumnSchema(APIModel):
    name: str
    dtype: str
    sdtype: str
    role: str
    description: str
    unit: str | None = None
    categories: list[Any] | None = None
    min_value: float | None = None
    max_value: float | None = None
    decimals: int | None = None


class WardSchema(APIModel):
    ward_id: str
    name: str
    ward_type: str
    bed_capacity: int
    max_escalation_beds: int
    nurse_ratio_day: float
    nurse_ratio_night: float
    admission_sources: list[str]
    required_nurses_day: int
    required_nurses_night: int
    accepts_electives: bool
    accepts_transfers: bool


class RuleInfo(APIModel):
    rule_id: str
    table: str
    severity: str
    name: str
    description: str
    columns: list[str] = Field(default_factory=list)


class TableSchemaResponse(APIModel):
    """Response of ``GET /api/v1/tables/{table}/schema``."""

    table: str
    key: str
    columns: list[ColumnSchema]
    training_columns: list[str]
    derived_columns: list[str]
    time_columns: list[str]
    raw_columns: list[str]
    wards: list[WardSchema]
    conditionable_columns: list[str]
    filterable_columns: list[str]
    scenarios: list[str]
    rules: list[RuleInfo]
    rules_available: bool = Field(description="False while the rule catalogue cannot be loaded.")


class ScenarioSupport(APIModel):
    """How common a scenario is in the reference data."""

    split: str
    reference_rows: int
    matching_rows: int
    share: float | None


class ScenarioInfo(APIModel):
    name: str
    table: str
    title: str
    category: str
    category_description: str
    description: str
    conditions: dict[str, list[Any]]
    filters: dict[str, dict[str, Any]]
    support: ScenarioSupport | None = None


class ScenarioListResponse(APIModel):
    count: int
    categories: dict[str, str]
    scenarios: list[ScenarioInfo]


#: Illustrative ``ModelInfo.metrics`` of ward_daily gc_tuned with the band-grouped spec, from its first
#: evaluation on 2026-11-15 (the live values are in the registry card; scenario coverage and the
#: timestamp differ slightly between evaluation runs).
MODEL_METRICS_EXAMPLE: dict[str, Any] = {
    "composite": 0.958271, "complete": True, "fidelity": 0.968472, "validity": 0.937856, "operational": 0.945605,
    "utility": 0.960379, "privacy": 0.997, "raw_hard_valid_rate": 0.937856, "operational_passed": 17,
    "operational_scored": 17, "scenario_coverage": 0.949279, "rank": 1, "comparable": True,
    "evaluated_at": "2026-11-15T03:15:33+00:00", "run_quick": False, "rows": 7016,
    "source": "reports/results/model_comparison.json",
}


class ModelInfo(APIModel):
    """A trained model as listed by the registry."""

    model_id: str
    table: str
    synthesizer: str | None = Field(
        None, description="SDV synthesizer class, e.g. 'GaussianCopulaSynthesizer', 'CTGANSynthesizer', 'TVAESynthesizer'."
    )
    description: str | None = None
    sdv_version: str | None = None
    fitted_at: str | None = None
    training_rows: int | None = None
    fit_seconds: float | None = None
    segment_by: str | list[str] | None = Field(
        None, description="Segment column(s): one SDV model per value (combination); null = one global model.")
    segment_label: str | None = Field(
        None, description="Readable segmentation, e.g. 'ward_id', 'ward_id x admission_source' or 'global'.")
    constraint_groups: list[str] | None = Field(
        None, description="Constraint groups the experiment enables (synthesis.experiments.<id>.constraints).")
    effective_constraint_groups: list[str] | None = Field(
        None, description="Enabled constraint groups that were applied inside SDV in at least one segment.")
    is_default: bool
    metrics: dict[str, Any] | None = Field(
        None,
        description="Results of the last model comparison (scripts/03_evaluate.py), stored in the model card and "
        "returned as stored; null until the model has been evaluated. Keys: composite (overall score in [0, 1]), "
        "complete (every score family available), fidelity, validity, operational, utility and privacy "
        "(component scores in [0, 1]), raw_hard_valid_rate (share of raw samples passing every hard rule), "
        "operational_passed / operational_scored (operational checks passed / scored), scenario_coverage, "
        "rank (1 = best comparable model of its table), comparable (evaluated under the same settings as the "
        "other models of its table), evaluated_at (ISO timestamp), run_quick (true for a --quick run), rows "
        "(synthetic rows evaluated) and source (the results file, relative to the project). Scores are null "
        "when not computed.",
        examples=[MODEL_METRICS_EXAMPLE],
    )
    card: dict[str, Any] = Field(default_factory=dict, description="Full model card.")


class TableModels(APIModel):
    default_model: str | None
    default_reason: str | None = None
    models: list[ModelInfo]


class ModelsResponse(APIModel):
    tables: dict[str, TableModels]


# ---------------------------------------------------------------------------
# OpenAPI examples
# ---------------------------------------------------------------------------
_VALID_WARD_DAY: dict[str, Any] = {
    "ward_id": "GMED", "month": 1, "day_of_week": "Mon", "escalation_beds_open": 0, "beds_closed": 0,
    "staffing_level": "adequate", "nurses_day_shift": 5, "nurses_night_shift": 4, "census_start": 30,
    "elective_scheduled": 2, "elective_cancelled": 0, "admissions_emergency": 4, "transfers_in": 1,
    "transfers_out": 1, "discharges": 6, "discharges_before_noon": 1, "delayed_discharges": 2,
    "emergency_boarders": 0, "avg_bed_wait_hours": 2.5, "avg_bed_turnaround_min": 80.0,
    "avg_los_days": 5.8, "occupancy_level": "high",
}
_INVALID_WARD_DAY: dict[str, Any] = {
    **_VALID_WARD_DAY,
    "ward_id": "gen med", "day_of_week": "monday", "staffing_level": "safe",
    "elective_scheduled": 1, "elective_cancelled": 3,
}
_VALID_STAY: dict[str, Any] = {
    "ward_id": "ORTH", "admission_source": "elective", "admission_month": 3, "admission_day_of_week": "Tue",
    "admission_hour": 8, "booking_lead_days": 45, "times_rescheduled": 1, "bed_wait_hours": 0.5,
    "los_nights": 3, "discharge_hour": 11, "separation_type": "discharge", "discharge_destination": "home",
    "delay_reason": "none", "discharge_delay_hours": 0.0, "bed_turnaround_minutes": 75,
}

GENERATE_EXAMPLES: dict[str, dict[str, Any]] = {
    "baseline": {
        "summary": "Baseline ward-days",
        "description": "Unconditional sample reproducing the real mix of wards, seasons and pressures.",
        "value": {"table": "ward_daily", "num_rows": 200, "scenario": "baseline", "seed": 42},
    },
    "high_occupancy": {
        "summary": "Occupancy: high-occupancy days with a quality check",
        "description": "Ward-days at or above 90% midnight occupancy, evaluated against the matching reference days.",
        "value": {"table": "ward_daily", "num_rows": 500, "scenario": "high_occupancy", "seed": 7, "evaluate": True},
    },
    "winter_medical": {
        "summary": "Seasonal: winter demand on medical wards",
        "value": {"table": "ward_daily", "num_rows": 300, "scenario": "winter_demand",
                  "wards": ["AMU", "GMED", "ELDC"], "seed": 11},
    },
    "emergency_pressure_csv": {
        "summary": "Emergency: bed pressure with boarders, as CSV",
        "description": "Returns text/csv with X-BedSynth-Dataset-Id / -Rows / -Model / -Scenario headers.",
        "value": {"table": "ward_daily", "num_rows": 250, "scenario": "emergency_pressure", "format": "csv", "seed": 3},
    },
    "scheduled_procedures": {
        "summary": "Scheduling: elective lists with cancellations",
        "value": {"table": "ward_daily", "num_rows": 300, "scenario": "scheduled_procedures",
                  "filters": {"elective_cancelled": {"min": 1}}, "seed": 5},
    },
    "staffing_shortage": {
        "summary": "Staffing: short-staffed days in winter",
        "value": {"table": "ward_daily", "num_rows": 200, "scenario": "staffing_shortage",
                  "conditions": {"month": [12, 1, 2]}, "seed": 13},
    },
    "calendar_january": {
        "summary": "Calendar mode: January for two wards",
        "description": "One row per ward per day with a `date` column (rows are independent draws).",
        "value": {"table": "ward_daily", "mode": "calendar", "wards": ["AMU", "ICU"],
                  "start_date": "2025-01-01", "end_date": "2025-01-31", "seed": 21},
    },
    "delayed_discharges": {
        "summary": "Delayed discharge: stays awaiting care packages or placements",
        "value": {"table": "patient_stays", "num_rows": 400, "scenario": "delayed_discharge_stays",
                  "conditions": {"discharge_destination": ["care_facility", "home_with_support"]}, "seed": 9},
    },
    "long_bed_waits": {
        "summary": "Emergency: long bed waits for emergency admissions",
        "value": {"table": "patient_stays", "num_rows": 300, "scenario": "long_bed_waits",
                  "evaluate": True, "seed": 17},
    },
    "custom_conditions": {
        "summary": "Custom: ICU pressure without a named scenario",
        "description": "Conditions and filters can be combined freely (and with a scenario).",
        "value": {"table": "ward_daily", "num_rows": 150,
                  "conditions": {"ward_id": "ICU", "occupancy_level": ["high", "critical"]},
                  "filters": {"emergency_boarders": {"min": 1}}, "include_derived": True, "seed": 99},
    },
}

VALIDATE_EXAMPLES: dict[str, dict[str, Any]] = {
    "ward_days": {
        "summary": "Ward-days: one valid, one with aliases and an impossible cancellation count",
        "value": {"table": "ward_daily", "records": [_VALID_WARD_DAY, _INVALID_WARD_DAY], "return_records": True},
    },
    "stay": {
        "summary": "A rescheduled elective orthopaedic stay",
        "value": {"table": "patient_stays", "records": [_VALID_STAY]},
    },
}

EVALUATE_EXAMPLES: dict[str, dict[str, Any]] = {
    "stored_dataset": {
        "summary": "Evaluate a persisted dataset",
        "description": "Use a dataset_id returned by /api/v1/generate.",
        "value": {"dataset_id": "ds-0123456789ab", "reference_split": "train"},
    },
    "client_records": {
        "summary": "Evaluate client records against the high-occupancy reference subset",
        "description": "At least 20 records are required; this example repeats one row for brevity.",
        "value": {"table": "ward_daily", "scenario": "high_occupancy", "records": [_VALID_WARD_DAY] * MIN_EVALUATE_ROWS},
    },
}

__all__ = [
    "DATASET_ID_REGEX",
    "EVALUATE_EXAMPLES",
    "EVALUATE_MAX_ROWS",
    "GENERATE_EXAMPLES",
    "MAX_ATTEMPTS",
    "MAX_CALENDAR_DAYS",
    "MAX_DATE",
    "MAX_ROWS",
    "MAX_TIME_BUDGET_SECONDS",
    "MIN_DATE",
    "MIN_EVALUATE_ROWS",
    "MODEL_ID_PATTERN",
    "TIME_BUDGET_SECONDS",
    "VALIDATE_EXAMPLES",
    "canonical_category",
    "months_in_period",
]
