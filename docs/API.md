# BedSynth AI REST API reference

This is the reference for the BedSynth AI REST API: the FastAPI service in `bedsynth/api/` that generates,
validates, evaluates and stores synthetic hospital-operations data. It is written for two readers: the
examiners who want to see what the Week-4 generator does, and a technical reader who wants to call it.
`README.md` ("Using the API") is the short version; `docs/DESIGN.md` section 10 explains the design.

All example responses were captured in-process (`fastapi.testclient`) against the final models (the
full run of 2026-11-15; the examples were re-checked after that run). They are trimmed
(`// ...` marks omitted content), and identifiers, timestamps and numbers (acceptance rates, quality
scores, timings) are **illustrative**. For model-quality results, see the README "Results" section
and `reports/week3_evaluation.md`.

## Contents

1. [Overview](#1-overview)
2. [Endpoint reference](#2-endpoint-reference)
3. [Generation in depth](#3-generation-in-depth)
4. [Validating and evaluating client records](#4-validating-and-evaluating-client-records)
5. [Dataset lifecycle and concurrency](#5-dataset-lifecycle-and-concurrency)
6. [Error model](#6-error-model)
7. [Limits](#7-limits)
8. [Scenario catalogue](#8-scenario-catalogue)
9. [Client examples](#9-client-examples)

---

## 1. Overview

The service wraps the trained SDV synthesizers and the generation engine
(`bedsynth/synthesis/generator.py`) in a small JSON API. A client picks a table, and optionally a
scenario, wards, conditions and filters. It gets back synthetic records that pass every hard operational
rule, along with metadata: a generation report, the hard/soft rule validation, an optional quick quality
score and column descriptions. Generations can be stored, then listed, downloaded as CSV or JSON, and
deleted. Records from elsewhere can be checked against the same rules (`/validate`) or scored
against the reference data (`/evaluate`). All data is synthetic and non-clinical.

| Table | Grain | Generated columns |
|---|---|---|
| `ward_daily` | one row per ward per day (a *ward-day*) | 35: `record_id`, 22 learned (`train`) and 12 derived columns; calendar mode adds `date` (36) |
| `patient_stays` | one row per inpatient ward stay (a *stay*) | 24: `stay_id`, 15 learned and 8 derived columns |

### 1.1 Base URL, versioning and content types

| Item | Value |
|---|---|
| Base URL | `http://127.0.0.1:8000` (the default of `scripts/04_serve_api.py`) |
| Versioning | data endpoints live under `/api/v1`; the service version (`1.0.0`) is returned by `GET /` and `GET /health` |
| Unversioned paths | `/`, `/health`, `/ui`, `/docs`, `/redoc`, `/openapi.json` |
| Request bodies | `application/json`. Unknown fields are rejected (422), and so are internal attribute names: only the public names are accepted (`validate`, not `run_validation`; the filter key `in`, not `in_`). A body sent with another content type (for example `text/plain`, or curl's default `application/x-www-form-urlencoded` when `-H "Content-Type: application/json"` is left out) or with no `Content-Type` header is not parsed as JSON and fails with 422 |
| Responses | `application/json`; `text/csv; charset=utf-8` for CSV generations and downloads; `text/html` for `/ui`, `/docs` and `/redoc` |
| Errors | always `{"detail": ..., "error_type": ...}` (see [section 6](#6-error-model)) |
| Authentication | none |

**No authentication.** BedSynth AI is a local tool for development, teaching and demos. It has no
authentication and no rate limit apart from the generation queue. Keep the default `--host 127.0.0.1`:
`--host 0.0.0.0` exposes every endpoint, including `DELETE`, to the network.

**CORS.** By default the only browser pages that may call the API are those served from `localhost`,
`127.0.0.1` or `[::1]`, on any port (`api.cors_origin_regex`). Origins listed in `api.cors_origins`
(empty by default) are allowed too, and `["*"]` allows every website. Credentials are never allowed.
Allowed methods are `GET`, `POST`, `DELETE` and `OPTIONS`, any request header is accepted, and these
response headers are exposed to scripts: the nine `X-BedSynth-*` headers, `Content-Disposition`,
`X-Process-Time-Ms` and `Retry-After`.
Error responses carry CORS headers too. A preflight from a disallowed origin gets a 400 response from the
CORS layer with the plain-text body `Disallowed CORS origin`, not the JSON error body (a preflight that asks
for a method outside those four, such as `PUT`, gets `Disallowed CORS method` in the same way). A simple `GET`
from a disallowed origin still runs, but its response has no `Access-Control-Allow-Origin` header, so
the browser hides it from the page.

**Common response headers**

| Header | Present on | Meaning |
|---|---|---|
| `X-Process-Time-Ms` | every response produced by the application (CORS preflights excepted) | server processing time in milliseconds |
| `X-BedSynth-*` | CSV generations and CSV downloads | dataset identity and verdicts ([section 3.5](#35-csv-output-and-the-x-bedsynth-headers)) |
| `Content-Disposition` | CSV generations and all downloads | `attachment; filename="<dataset_id>.csv"` (or `.json`) |
| `Retry-After` | 503 `server_busy` | seconds to wait before retrying (60 by default) |
| `Allow` | 405 responses | the methods the path supports |

### 1.2 Starting the service

Run commands from the project root with the virtual-environment interpreter (shown as `python`, as in
the README):

```bash
python scripts/04_serve_api.py                        # http://127.0.0.1:8000
python scripts/04_serve_api.py --port 8001            # if port 8000 is already taken
python scripts/04_serve_api.py --port 8080 --reload   # restart automatically when a .py file in bedsynth/ changes
uvicorn bedsynth.api.main:app --port 8000             # the same application without the wrapper
```

If another program already listens on port 8000, uvicorn logs `[Errno 10048] error while attempting
to bind on address ('127.0.0.1', 8000)` (Windows) and exits, while that program keeps answering at the
address; start BedSynth on another port and use that port in the URLs below.

| `04_serve_api.py` option | Default | Effect |
|---|---|---|
| `--host` | `127.0.0.1` | interface to bind |
| `--port` | `8000` | port to listen on |
| `--reload` | off | restart when a `.py` file in `bedsynth/` changes (development). uvicorn's reloader watches only `*.py` files, so an edit to `config/settings.yaml` still needs a manual restart |
| `--log-level` | `info` | `critical`, `error`, `warning`, `info` or `debug`; the `bedsynth` logger follows it |
| `--no-preload` | off | load the generation engine on the first request that needs it instead of at start-up (sets `BEDSYNTH_API_PRELOAD=0`); `/health` then reports `engine.state` `"not_loaded"` until that request |

What happens at start-up:

* **Engine preload.** A background thread imports the generation engine (SDV and torch) when the server
  starts: about 5-12 s on the development machine (5 s in an earlier check with warm file caches,
  10.4 s and 11.6 s in two checks on the final code), longer on a cold start. `/health` never waits
  for it: meanwhile it answers at once with `"status": "degraded"` and
  `"engine": {"state": "loading"}` ([section 2.2](#22-get-health-health-and-readiness)). Requests that
  need the engine (`/api/v1/tables`, `/api/v1/models`, `/api/v1/generate`) wait for it, so the first of
  them can be slow.
* **Models.** Each trained model is loaded the first time it is used, and reloaded when its file changes.
  The registry file is re-read whenever it is consulted, so models trained while the server runs are
  picked up without a restart.
* **Settings.** Limits come from `config/settings.yaml`, or from the file named by `BEDSYNTH_SETTINGS`,
  and are read at start-up, so restart the server after changing them. Directories can be redirected
  with the `BEDSYNTH_*_DIR` variables listed in the README. For example, `BEDSYNTH_SYNTHETIC_DIR` moves
  the dataset store and `BEDSYNTH_MODELS_DIR` points at another set of models.
* **Missing models or data.** Without trained models the server still starts: `/health` reports
  `"status": "degraded"` and puts the command to run in `messages`, and `/generate` answers 503
  `model_unavailable`. Without the processed reference data (`data/processed/`):
  * `/evaluate` answers 503 `reference_data_unavailable`;
  * scenario `support` is `null`;
  * generation still works: allocation falls back to the model's training frequencies (with a warning
    when the request has several condition combinations), and `evaluate: true` adds a warning instead
    of a score.

### 1.3 Interactive documentation and the demo UI

| URL | What it is | Network use |
|---|---|---|
| `/docs` | Swagger UI. Every endpoint can be tried; `POST /api/v1/generate` offers ten example bodies (baseline, high occupancy, winter demand, CSV output, calendar mode, delayed discharges, ...) | loads `swagger-ui-dist@5` from `cdn.jsdelivr.net` and its favicon from `fastapi.tiangolo.com` |
| `/redoc` | ReDoc rendering of the same schema | loads `redoc@2` from `cdn.jsdelivr.net`, fonts from `fonts.googleapis.com` and its favicon from `fastapi.tiangolo.com` |
| `/openapi.json` | OpenAPI 3.1 schema (field descriptions, limits, examples) | served locally |
| `/ui` | single-page demo UI (`bedsynth/api/static/index.html`) | fully offline: one HTML file, no external assets |

The demo UI calls `GET /health`, `GET /api/v1/scenarios?include_support=true`,
`GET /api/v1/tables/ward_daily/schema`, `GET /api/v1/tables` and `GET /api/v1/models` to build its form.
Its status pill reads *Starting · loading the engine…* while `/health` reports `engine.state`
`"loading"` and polls `/health` every 1.5 s until the engine is ready (it then fetches what could not be
listed meanwhile); when the API cannot be reached it shows *API unreachable · retrying* and tries again
after 2, 5, 10, 20 and then every 30 s, so the page recovers on its own once the server is up. The
model picker shows each model's composite score from `GET /api/v1/models`. The UI then sends
`POST /api/v1/generate` with these fields: `table`, `scenario`, `num_rows` (or `"mode": "calendar"` with
`start_date` and `end_date`), `wards`, `seed`, `model_id`, `evaluate`, `validate`, `include_derived` and
`persist`. It does not send `conditions`, `filters`, `time_budget_seconds` or `max_attempts`; use
`/docs` or a client for those. Its dataset panel uses `GET /api/v1/datasets?limit=100`,
`GET /api/v1/datasets/{id}`, both download formats, `POST /api/v1/evaluate` with `{"dataset_id": ...}` and
`DELETE`. Two buttons copy the current request: *Copy as curl (bash)* (single-quoted body, for Git Bash,
Linux and macOS) and *Copy for PowerShell* (the body in a single-quoted here-string, sent with
`Invoke-RestMethod`; works in Windows PowerShell 5.1 and PowerShell 7, see
[section 9.2](#92-windows-powershell-51)). The UI gives up on a generation after 240 s, on `/health`
after 10 s (and checks again) and on any other call after 30 s.

### 1.4 Endpoint summary

| Method | Path | Purpose | Success |
|---|---|---|---|
| GET | `/` | service information, links and the endpoint list | 200 JSON |
| GET | `/health` | health, engine state, trained models, reference data, dataset count (answers at once) | 200 JSON (always) |
| GET | `/ui` | interactive demo UI | 200 HTML |
| GET | `/api/v1/tables` | the two tables with column counts, scenarios and default model | 200 JSON |
| GET | `/api/v1/tables/{table}/schema` | columns, ward catalogue and operational rule catalogue | 200 JSON |
| GET | `/api/v1/scenarios` | scenario catalogue, optionally with reference support | 200 JSON |
| GET | `/api/v1/scenarios/{name}` | one scenario with reference support | 200 JSON |
| GET | `/api/v1/models` | trained models with their evaluation metrics, model cards and default per table | 200 JSON |
| POST | `/api/v1/generate` | generate validated synthetic records | 200 JSON or CSV |
| POST | `/api/v1/validate` | check client records against the hard and soft rules | 200 JSON |
| POST | `/api/v1/evaluate` | quick quality of client records or a stored dataset | 200 JSON |
| GET | `/api/v1/datasets` | list stored datasets, newest first | 200 JSON |
| GET | `/api/v1/datasets/{dataset_id}` | stored metadata of one dataset | 200 JSON |
| GET | `/api/v1/datasets/{dataset_id}/download` | stored records as CSV or JSON | 200 CSV or JSON |
| DELETE | `/api/v1/datasets/{dataset_id}` | delete a stored dataset | 200 JSON |

---

## 2. Endpoint reference

### 2.1 `GET /`: service information

Returns the service name, version, a description, links to the main resources and one entry per
documented endpoint. The endpoint list is read from the OpenAPI schema, so it never drifts from the
code.

| Field | Type | Meaning |
|---|---|---|
| `name`, `title`, `version`, `description` | string | service identity (`version` is the package version) |
| `links` | object | `docs`, `redoc`, `openapi`, `ui`, `health`, `tables`, `scenarios`, `models`, `generate`, `datasets` |
| `endpoints` | array | `{method, path, summary}` for each of the 15 endpoints in [section 1.4](#14-endpoint-summary) |

```bash
curl -s http://127.0.0.1:8000/
```

```jsonc
{
  "name": "BedSynth AI",
  "title": "BedSynth AI - Synthetic Hospital Operations Data Generator",
  "version": "1.0.0",
  "description": "Synthetic, non-clinical hospital operations data (ward-day bed occupancy and patient stays) generated on demand with SDV models, scenario conditioning and rule validation.",
  "links": {"docs": "/docs", "redoc": "/redoc", "openapi": "/openapi.json", "ui": "/ui", "health": "/health",
            "tables": "/api/v1/tables", "scenarios": "/api/v1/scenarios", "models": "/api/v1/models",
            "generate": "/api/v1/generate", "datasets": "/api/v1/datasets"},
  "endpoints": [
    {"method": "GET", "path": "/", "summary": "Service information and links"},
    {"method": "GET", "path": "/health", "summary": "Health, trained models and reference data"},
    {"method": "GET", "path": "/ui", "summary": "Interactive demo UI"}
    // ... 12 more, in path order (/api/v1/tables ... /api/v1/datasets/{dataset_id}/download)
  ]
}
```

### 2.2 `GET /health`: health and readiness

Always answers 200, and at once: `/health` never creates the generation engine or waits for it.
`status` is `"ok"` when the engine is ready (or loads on first use, with preloading off), there is no
engine or model-listing error and both tables have a trained model; otherwise it is `"degraded"` and
`messages` says why and what to run. It is `"degraded"` in particular for the few seconds after a server
start while the engine loads. Missing reference data adds a message but does not make the status
degraded.

| Field | Type | Meaning |
|---|---|---|
| `status` | `"ok"` or `"degraded"` | overall readiness |
| `version`, `sdv_version` | string | package version and installed SDV version (read from the package metadata; SDV is not imported) |
| `engine` | object | `available`, `loaded`, `state`, `detail`: see the table below |
| `models` | object | per table: `trained` (experiment ids whose files exist, sorted) and `default` (used when a request omits `model_id`). Listed by the engine once it is ready; before that read from `<models_dir>/registry.json` with plain JSON parsing (no SDV import): a model counts as trained when it is registered and its `synthesizer.pkl` exists, and the default is the registry default if trained, else `synthesis.default_experiment`, else the first trained id - the same rules the engine applies |
| `trained_tables` | array | tables with at least one trained model |
| `reference_data` | object | per table: whether the processed train split exists (needed by `/evaluate`, scenario support and quality scores) |
| `datasets` | object | `count` (stored dataset folders, unreadable ones included; 0 when the store cannot be read) and `persistence_enabled` (`api.persist_datasets`) |
| `messages` | array | what to fix: the engine state while loading or after a failure, the command to run for each missing model or reference split, an unreadable registry or dataset store |
| `checked_at` | string | UTC timestamp |

| `engine.state` | `available` | `loaded` | `detail` (also the first message, except `not_loaded`) | `status` |
|---|---|---|---|---|
| `"loading"` (start-up preload running) | `true` | `false` | `Engine loading: the generation engine (SDV) is starting in the background; requests that need it wait until it is ready.` | `"degraded"` |
| `"not_loaded"` (preloading off, before the first request that needs the engine) | `true` | `false` | `Engine not loaded yet: it loads on the first request that needs it.` | `"ok"` if both tables have a trained model |
| `"ready"` | `true` | `true` | `null` | `"ok"` if both tables have a trained model |
| `"failed"` (the last initialisation failed; the next request that needs the engine retries) | `false` | `false` | `The generation engine could not be initialised (<ExceptionType>); see the server log.` | `"degraded"` |

`loaded` describes the engine at the time of the call. If a ready engine cannot list its models, the
response has `engine` `{"available": false, "loaded": true, "state": "ready"}`, status `"degraded"`
and the message `Could not list trained models (<ExceptionType>); see the server log.`. Two more
messages can appear: `The model registry could not be read (<ExceptionType>); see the server log.`
(unreadable `registry.json` before the engine is ready) and `The dataset store could not be read
(<ExceptionType>); see the server log.`. The exception text and the traceback go to the server log
only.

```bash
curl -s http://127.0.0.1:8000/health
```

```jsonc
{
  "status": "ok",
  "version": "1.0.0",
  "sdv_version": "2.2.0",
  "engine": {"available": true, "loaded": true, "state": "ready", "detail": null},
  "models": {
    "ward_daily": {"trained": ["ctgan", "gc_baseline", "gc_segmented", "gc_tuned", "tvae"], "default": "gc_tuned"},
    "patient_stays": {"trained": ["ctgan", "gc_baseline", "gc_segmented", "gc_tuned", "tvae"], "default": "gc_tuned"}
  },
  "trained_tables": ["ward_daily", "patient_stays"],
  "reference_data": {"ward_daily": true, "patient_stays": true},
  "datasets": {"count": 0, "persistence_enabled": true},
  "messages": [],
  "checked_at": "2026-11-15T10:48:48+00:00"
}
```

The same call made right after the server started, while the engine loaded (it answered in 0.03 s in
process; the engine was ready about 6 s later):

```jsonc
{
  "status": "degraded",
  "engine": {"available": true, "loaded": false, "state": "loading",
             "detail": "Engine loading: the generation engine (SDV) is starting in the background; requests that need it wait until it is ready."},
  "models": {"ward_daily": {"trained": ["ctgan", "gc_baseline", "gc_segmented", "gc_tuned", "tvae"], "default": "gc_tuned"},
             "patient_stays": {"trained": ["ctgan", "gc_baseline", "gc_segmented", "gc_tuned", "tvae"], "default": "gc_tuned"}},
  "trained_tables": ["ward_daily", "patient_stays"],
  "messages": ["Engine loading: the generation engine (SDV) is starting in the background; requests that need it wait until it is ready."]
  // ... version, sdv_version, reference_data, datasets, checked_at
}
```

On a server without models or reference data (trimmed):

```jsonc
{
  "status": "degraded",
  "models": {"ward_daily": {"trained": [], "default": null}, "patient_stays": {"trained": [], "default": null}},
  "trained_tables": [],
  "reference_data": {"ward_daily": false, "patient_stays": false},
  "messages": [
    "No trained model for 'ward_daily'. Train models with `python scripts/02_train_synthesizers.py` (or `python scripts/run_pipeline.py --quick`); the running API picks them up automatically.",
    // ... the same for patient_stays
    "Reference data for 'ward_daily' is missing (needed by evaluate and scenario support). Prepare the reference data with `python scripts/01_prepare_data.py`.",
    // ...
  ]
  // ...
}
```

### 2.3 `GET /ui`: demo UI

Returns the single-page demo as `text/html` (see [section 1.3](#13-interactive-documentation-and-the-demo-ui)).
If `bedsynth/api/static/index.html` is missing it returns 404 `not_found`.

### 2.4 `GET /api/v1/tables`: tables

Always answers 200. While the engine is still loading after a server start, the request waits for
it. If the generation engine cannot be initialised, `default_model` is `null` and `trained_models` is
empty; `/health` gives the reason.

| Field (per item of `tables`) | Meaning |
|---|---|
| `name`, `title`, `description`, `grain` | table identity |
| `key_column` | `record_id` or `stay_id` |
| `n_columns`, `n_training_columns`, `n_derived_columns` | columns of a generated dataset (key + learned + derived) |
| `n_scenarios` | 16 for `ward_daily`, 10 for `patient_stays` |
| `conditionable_columns` | columns usable in `conditions` |
| `supports_calendar_mode` | `true` for `ward_daily` only |
| `default_model`, `trained_models` | registry view |
| `reference_data_available` | whether the processed train split exists |
| `links` | `schema`, `scenarios`, `models` URLs for the table |

```jsonc
{
  "tables": [
    {
      "name": "ward_daily",
      "title": "Ward-day operations",
      "description": "Daily capacity, nurse staffing, patient flow, emergency demand, elective scheduling, bed turnover and midnight occupancy for each of the eight wards of a 194-bed hospital.",
      "grain": "One row per ward per calendar day",
      "key_column": "record_id",
      "n_columns": 35, "n_training_columns": 22, "n_derived_columns": 12, "n_scenarios": 16,
      "conditionable_columns": ["ward_id", "month", "day_of_week", "staffing_level", "occupancy_level"],
      "supports_calendar_mode": true,
      "default_model": "gc_tuned",
      "trained_models": ["ctgan", "gc_baseline", "gc_segmented", "gc_tuned", "tvae"],
      "reference_data_available": true,
      "links": {"schema": "/api/v1/tables/ward_daily/schema", "scenarios": "/api/v1/scenarios?table=ward_daily",
                "models": "/api/v1/models?table=ward_daily"}
    }
    // ... patient_stays: 24 columns (15 learned, 8 derived), 10 scenarios, conditionable columns
    //     ward_id, admission_source, admission_month, admission_day_of_week, separation_type,
    //     discharge_destination, delay_reason; supports_calendar_mode false
  ]
}
```

### 2.5 `GET /api/v1/tables/{table}/schema`: columns and rules

Path parameter `table`: `ward_daily` or `patient_stays`. Any other value returns 404 `table_not_found`.
`bedsynth/schema.py` is the source of truth for this payload.

| Field | Meaning |
|---|---|
| `table`, `key` | table and key column |
| `columns` | every column: `name`, `dtype` (`int`, `float`, `str`, `bool`, `date`, `datetime`), `sdtype`, `role` (`key`, `time`, `train` = learned by SDV, `derived` = recomputed), `description`, `unit`, `categories` (vocabulary), `min_value`/`max_value` (physical bounds), `decimals` |
| `training_columns`, `derived_columns`, `time_columns`, `raw_columns` | column groups (time columns exist in the source data; generated data has none, apart from `date` in calendar mode) |
| `wards` | the eight wards: `ward_id`, `name`, `ward_type`, `bed_capacity`, `max_escalation_beds`, nurse ratios, `admission_sources`, required nurses, `accepts_electives`, `accepts_transfers` |
| `conditionable_columns` | categorical learned columns usable in `conditions` |
| `filterable_columns` | learned and derived columns usable in `filters` (34 for `ward_daily`, 23 for `patient_stays`) |
| `scenarios` | scenario names of the table |
| `rules` | rule catalogue: `rule_id`, `table`, `severity` (`hard` or `soft`), `name`, `description`, `columns` |
| `rules_available` | `false` only if the rule catalogue cannot be loaded |

```bash
curl -s http://127.0.0.1:8000/api/v1/tables/ward_daily/schema
```

```jsonc
{
  "table": "ward_daily",
  "key": "record_id",
  "columns": [
    {"name": "record_id", "dtype": "str", "sdtype": "id", "role": "key", "description": "Record identifier, e.g. WD-000001.",
     "unit": null, "categories": null, "min_value": null, "max_value": null, "decimals": null},
    {"name": "ward_id", "dtype": "str", "sdtype": "categorical", "role": "train", "description": "Ward code.",
     "unit": null, "categories": ["AMU", "GMED", "ELDC", "CARD", "GSUR", "ORTH", "ICU", "PAED"],
     "min_value": null, "max_value": null, "decimals": null}
    // ... 34 more
  ],
  "training_columns": ["ward_id", "month", "day_of_week", "escalation_beds_open", "beds_closed", "staffing_level", /* ... */ "occupancy_level"],
  "derived_columns": ["ward_type", "season", "is_weekend", "bed_capacity", "available_beds", "admissions_elective", "census_end",
                      "occupancy_rate", "bed_turnover_rate", "patients_per_nurse", "delayed_discharge_rate", "elective_cancellation_rate"],
  "wards": [
    {"ward_id": "AMU", "name": "Acute Medical Unit", "ward_type": "acute", "bed_capacity": 28, "max_escalation_beds": 6,
     "nurse_ratio_day": 5.0, "nurse_ratio_night": 7.0, "admission_sources": ["emergency"], "required_nurses_day": 6,
     "required_nurses_night": 4, "accepts_electives": false, "accepts_transfers": false}
    // ... 7 more
  ],
  "rules": [
    {"rule_id": "WD-H01", "table": "ward_daily", "severity": "hard", "name": "valid_categories",
     "description": "ward_id, month, day_of_week, staffing_level and occupancy_level take values from their vocabularies.",
     "columns": ["ward_id", "month", "day_of_week", "staffing_level", "occupancy_level"]}
    // ... WD-H02 to WD-H15, WD-S01 to WD-S06
  ],
  "rules_available": true
  // ... time_columns, raw_columns, conditionable_columns, filterable_columns, scenarios
}
```

### 2.6 `GET /api/v1/scenarios`: scenario catalogue

| Query parameter | Type | Default | Meaning |
|---|---|---|---|
| `table` | string | all tables | `ward_daily` or `patient_stays`; any other value returns 404 `table_not_found` |
| `include_support` | boolean | `false` | add how many reference rows match each scenario |

| Field | Meaning |
|---|---|
| `count` | scenarios returned |
| `categories` | category key mapped to its description (for example `"utilization": "Ward utilisation and capacity"`) |
| `scenarios[]` | `name`, `table`, `title`, `category`, `category_description`, `description`, `conditions` (`{column: [values]}`, values the generated rows must take; served by segment routing and pooled rejection, [section 3.2](#32-scenario-wards-conditions-and-filters)), `filters` (`{column: {min, max, in, not_in}}`, applied by rejection), `support` |
| `support` | `{split: "train", reference_rows, matching_rows, share}`: reference train rows that satisfy the scenario's conditions and filters. `null` unless `include_support=true` is given (and when the reference data is missing) |

For example, `GET /api/v1/scenarios?table=ward_daily&include_support=true` returns
`"support": {"split": "train", "reference_rows": 7016, "matching_rows": 394, "share": 0.0562}` for
`elective_cancellations`, the rarest scenario; without `include_support` every `support` is `null`.

[Section 8](#8-scenario-catalogue) summarises all 26 scenarios.

### 2.7 `GET /api/v1/scenarios/{name}`: one scenario

The name is trimmed and matched case-insensitively (`HIGH_OCCUPANCY` works). The response is one
catalogue item and always includes `support` (`null` when the reference data is missing). An unknown name
returns 404 `scenario_not_found`.

```bash
curl -s http://127.0.0.1:8000/api/v1/scenarios/high_occupancy
```

```json
{
  "name": "high_occupancy",
  "table": "ward_daily",
  "title": "High occupancy",
  "category": "occupancy",
  "category_description": "Normal, high and low bed occupancy",
  "description": "Ward-days at or above 90% midnight occupancy (high or critical band).",
  "conditions": {"occupancy_level": ["high", "critical"]},
  "filters": {},
  "support": {"split": "train", "reference_rows": 7016, "matching_rows": 3296, "share": 0.4698}
}
```

### 2.8 `GET /api/v1/models`: trained models

Optional query parameter `table` (an unknown table returns 404 `table_not_found`). While the engine is
still loading after a server start, the request waits for it. Returns 503 `engine_unavailable` if the
generation engine cannot be initialised (`"The generation engine could not be initialised
(<ExceptionType>); see the server log."`). With no trained models it returns 200 with empty lists.

For each table the response gives `default_model`, `default_reason` and `models[]`. The default is the
registry default (set by `scripts/03_evaluate.py`); if that is not trained, it is
`synthesis.default_experiment` from the settings; failing that, the first trained id in alphabetical
order.

| Field (per model) | Meaning |
|---|---|
| `model_id`, `table`, `synthesizer`, `description` | experiment identity; `synthesizer` is the SDV synthesizer class (`GaussianCopulaSynthesizer`, `CTGANSynthesizer`, `TVAESynthesizer`) |
| `sdv_version`, `fitted_at`, `training_rows`, `fit_seconds` | fit facts |
| `segment_by`, `segment_label` | segmentation: a column, a list of columns, or `null` for one global model (`"global"`); for example `"ward_id x admission_source"` |
| `constraint_groups`, `effective_constraint_groups` | constraint groups the experiment enables, and those applied inside SDV in at least one segment |
| `is_default` | `true` for the table's default |
| `metrics` | the results of the last model comparison, written into the model card by `scripts/03_evaluate.py` and returned as stored; `null` until the model has been evaluated (or when it was retrained after its evaluation). Keys below |
| `card` | the full model card with server file paths removed |

| `metrics` key | Meaning |
|---|---|
| `composite` | overall score in [0, 1] (weighted sum of the five components) |
| `complete` | every score family available (incomplete composites are never selected) |
| `fidelity`, `validity`, `operational`, `utility`, `privacy` | component scores in [0, 1] |
| `raw_hard_valid_rate` | share of raw, unfiltered model samples that pass every hard rule |
| `operational_passed`, `operational_scored` | operational checks passed and scored |
| `scenario_coverage` | scenario-coverage family score |
| `rank` | 1 = best comparable model of its table (the registry default) |
| `comparable` | evaluated under the same run configuration as the table's other models |
| `evaluated_at`, `run_quick`, `rows` | when, whether in `--quick` mode, and on how many synthetic rows the model was evaluated |
| `source` | the results file, relative to the project (`reports/results/model_comparison.json`) |

```jsonc
{
  "tables": {
    "ward_daily": {
      "default_model": "gc_tuned",
      "default_reason": "Best composite score 0.958 among 5 comparable evaluated model(s) (scripts/03_evaluate.py, 2026-11-15T05:31:06+00:00).",
      "models": [
        // ... ctgan, gc_baseline, gc_segmented (models are listed in id order)
        {
          "model_id": "gc_tuned", "table": "ward_daily", "synthesizer": "GaussianCopulaSynthesizer",
          "description": "GaussianCopula segmented per ward (ward_daily) and per ward x admission source (patient_stays), ...",
          "sdv_version": "2.2.0", "fitted_at": "2026-11-15T05:01:30+00:00", "training_rows": 7016, "fit_seconds": 33.886,
          "segment_by": "ward_id", "segment_label": "ward_id",
          "constraint_groups": ["inequality", "fixed_combinations", "regime_labels", "structural_zeros", "capacity", "conditional_marginals"],
          "effective_constraint_groups": ["inequality", "regime_labels", "structural_zeros", "capacity", "conditional_marginals"],
          "is_default": true,
          "metrics": {"composite": 0.958271, "complete": true, "fidelity": 0.968472, "validity": 0.937856,
                      "operational": 0.945605, "utility": 0.960379, "privacy": 0.997, "raw_hard_valid_rate": 0.937856,
                      "operational_passed": 17, "operational_scored": 17, "scenario_coverage": 0.949286, "rank": 1,
                      "comparable": true, "evaluated_at": "2026-11-15T05:22:19+00:00", "run_quick": false, "rows": 7016,
                      "source": "reports/results/model_comparison.json"},
          "card": { /* experiment, segments, constraints, distributions, training diagnostics, metrics, ... */ }
        }
        // ... tvae
      ]
    }
    // ... patient_stays
  }
}
```

### 2.9 `POST /api/v1/generate`: generate synthetic records

Generates synthetic records for one table. Every returned row passes all hard rules of the table (15
for ward-days, 14 for stays). Section 3 explains the semantics in depth.

**Request body** (every field is optional)

| Field | Type | Default | Validation and limits | Meaning |
|---|---|---|---|---|
| `table` | `"ward_daily"` or `"patient_stays"` | `"ward_daily"` | other values: 422 | table to synthesise |
| `num_rows` | integer | `100` | 1 to 10,000 (`api.max_rows`); must be a JSON integer (`"100"` or `100.0` gives 422) | rows in sample mode; ignored (but still validated) in calendar mode |
| `scenario` | string or null | `null` | at most 64 characters; trimmed and lower-cased (blank means none); unknown: 404 `scenario_not_found`; other table: 400 `scenario_table_mismatch` | named scenario ([section 8](#8-scenario-catalogue)) |
| `model_id` | string or null | registry default | pattern `^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$` (422); not trained while others are: 404 `model_not_found`; nothing trained: 503 `model_unavailable` | experiment id from `GET /api/v1/models` |
| `wards` | array of strings or null | all wards | 1 to 8 items; codes or aliases (`"icu"`, `"General Medicine"`); unknown ward: 422; duplicates removed | restrict generation to these wards |
| `conditions` | object or null | `null` | `{column: value or [values]}` (integers or strings) on conditionable columns; bad column or value: 400 `invalid_conditions`; more than 1,000 combinations after merging: 400 `too_many_combinations` | categorical conditions ([section 3.2](#32-scenario-wards-conditions-and-filters)) |
| `filters` | object or null | `null` | `{column: {min, max, in, not_in}}` on learned or derived columns; see [section 3.2](#32-scenario-wards-conditions-and-filters) for 400/422 cases | acceptance filters (inclusive bounds) |
| `seed` | integer or null | drawn at random | 0 to 4,294,967,295; JSON integer | random seed ([section 3.3](#33-reproducibility-and-seeds)) |
| `max_attempts` | integer or null | 25 (`generation.max_attempts`) | 1 to 100 (`api.max_attempts`); JSON integer | maximum rejection-sampling rounds |
| `time_budget_seconds` | number or null | 60 (`api.time_budget_seconds`) | greater than 0, at most 120 (`api.max_time_budget_seconds`), finite | sampling time budget ([section 3.6](#36-time-budget-and-max_attempts)) |
| `mode` | `"sample"` or `"calendar"` | `"sample"` | calendar mode for `ward_daily` only (422) | [section 3.1](#31-sample-mode-and-calendar-mode) |
| `start_date`, `end_date` | ISO date or null | `null` | calendar mode only (422 otherwise); both required; `end_date` on or after `start_date`; between 1900-01-01 and 2200-12-31; at most 1,100 days (`api.max_calendar_days`); wards x days at most 10,000 | calendar period, inclusive |
| `include_derived` | boolean | `true` | | include derived columns in the output |
| `validate` | boolean | `true` | | run the hard and soft rules on the output and report them |
| `evaluate` | boolean | `false` | | add a quick quality score against the matching reference rows (adds roughly a second) |
| `format` | `"json"` or `"csv"` | `"json"` | | response format ([section 3.5](#35-csv-output-and-the-x-bedsynth-headers)) |
| `persist` | boolean | `true` | | store the dataset for `/api/v1/datasets` |

**Response** (`format: "json"`)

| Field | Meaning |
|---|---|
| `dataset_id` | `ds-` followed by 12 hex characters; always assigned, even when the dataset is not stored |
| `persisted` | `true` when the dataset was stored and can be fetched later |
| `created_at` | UTC timestamp (ISO 8601, seconds) |
| `table`, `n_rows` | table and rows returned. `n_rows` can be lower than `num_rows` (a shortfall, [section 3.7](#37-shortfall-empty-results-and-warnings)) |
| `model` | `{id, synthesizer, sdv_version, fitted_at}` of the model that sampled |
| `scenario` | `{name, title, description, category}`, or `null` |
| `request` | normalised echo of the request: canonical scenario name, the request's own conditions and filters after normalisation, the effective `time_budget_seconds`, and `seed` as sent (`null` when drawn) |
| `generation` | generation report ([section 3.8](#38-generation-report)) |
| `validation` | validation summary ([section 3.9](#39-validation-summary)), or `null` when `validate` is false |
| `quality` | quick quality ([section 3.10](#310-quality-block)), or `null` |
| `columns` | `{name, dtype, role, unit, description}` for each output column, in output order |
| `warnings` | request, engine, validation and quality warnings, deduplicated; at most 25 plus one line counting the rest |
| `links` | `{self, download_csv, download_json}` when persisted, otherwise `{}` |
| `records` | the rows (missing values as `null`, booleans as JSON booleans) |

**Status codes:** 200; 400 `invalid_conditions`, `too_many_combinations`, `invalid_filters`,
`scenario_table_mismatch`, `generation_error`, `invalid_request`; 404 `scenario_not_found`,
`model_not_found`; 422 `validation_error`, `no_rows_generated`; 503 `model_unavailable`,
`engine_unavailable`, `server_busy`; 500 `internal_error`.

**Example:** 200 high-occupancy ward-days for two wards, reproducible from the seed, with a quality score
(the README example).

```bash
curl -s -X POST http://127.0.0.1:8000/api/v1/generate -H "Content-Type: application/json" \
     -d '{"table": "ward_daily", "num_rows": 200, "scenario": "high_occupancy", "wards": ["AMU", "GMED"], "seed": 42, "evaluate": true}'
```

```jsonc
{
  "dataset_id": "ds-dbd4a91c4059",
  "persisted": true,
  "created_at": "2026-11-15T10:48:48+00:00",
  "table": "ward_daily",
  "n_rows": 200,
  "model": {"id": "gc_tuned", "synthesizer": "GaussianCopulaSynthesizer", "sdv_version": "2.2.0", "fitted_at": "2026-11-15T05:01:30+00:00"},
  "scenario": {"name": "high_occupancy", "title": "High occupancy",
               "description": "Ward-days at or above 90% midnight occupancy (high or critical band).", "category": "occupancy"},
  "request": {"table": "ward_daily", "num_rows": 200, "scenario": "high_occupancy", "model_id": null, "wards": ["AMU", "GMED"],
              "conditions": null, "filters": null, "seed": 42, "max_attempts": null, "time_budget_seconds": 60.0,
              "mode": "sample", "start_date": null, "end_date": null, "include_derived": true, "validate": true,
              "evaluate": true, "format": "json", "persist": true},
  "generation": {
    "table": "ward_daily", "model_id": "gc_tuned",
    "model": {"id": "gc_tuned", "experiment_id": "gc_tuned", "synthesizer": "GaussianCopulaSynthesizer", "model": "gaussian_copula",
              "segment_by": "ward_id", "constraint_groups": ["inequality", "fixed_combinations", "regime_labels", "structural_zeros",
              "capacity", "conditional_marginals"], "sdv_version": "2.2.0", "fitted_at": "2026-11-15T05:01:30+00:00"},
    "scenario": "high_occupancy", "mode": "sample", "seed": 42,
    "num_rows_requested": 200, "num_rows_generated": 200, "shortfall": 0,
    "attempts": 1, "max_attempts": 25, "sampled_rows": 319, "accepted_rows": 292,
    "acceptance_rate": 0.915361, "raw_hard_valid_rate": 0.915361,
    "rejected_by_rule": {"WD-H06": 26, "WD-H15": 6}, "rejected_by_filter": 0,
    "first_batch_factor": 1.6, "acceptance_prior": 0.9305,
    "time_budget_seconds": 60.0, "stopped_reason": null, "throttled_rounds": 0, "rows_per_second": 1205.7,
    "condition_allocation": [
      {"conditions": {"occupancy_level": "high", "ward_id": "AMU"}, "requested": 42, "generated": 42,
       "reference_count": 209, "sampled": 68, "accepted": 68, "feasible": true},
      {"conditions": {"occupancy_level": "high", "ward_id": "GMED"}, "requested": 75, "generated": 75,
       "reference_count": 378, "sampled": 118, "accepted": 118, "feasible": true}
      // ... critical/AMU (50 rows), critical/GMED (33 rows)
    ],
    "conditions": {"occupancy_level": ["high", "critical"], "ward_id": ["AMU", "GMED"]},
    "filters": {},
    "reference_source": "reference_train_split",
    "include_derived": true, "elapsed_seconds": 0.288, "warnings": [],
    "allocation_strata": {"columns": ["ward_id", "occupancy_level"], "source": "reference_train_split",
                          "strata_cells": 0, "stratified_combinations": 0, "rebalanced_rows": 0, "quota_moves": 0}
  },
  "validation": {
    "n_rows": 200, "n_hard_valid": 200, "hard_valid_rate": 1.0, "all_hard_rules_passed": true,
    "hard_rules": {"checked": 15, "passed": 15, "failed": 0, "violations": 0},
    "soft_rules": {"checked": 6, "passed": 5, "failed": 1, "violations": 1},
    "rules": [
      {"rule_id": "WD-H01", "name": "valid_categories", "severity": "hard",
       "description": "ward_id, month, day_of_week, staffing_level and occupancy_level take values from their vocabularies.",
       "columns": ["ward_id", "month", "day_of_week", "staffing_level", "occupancy_level"],
       "violations": 0, "violation_rate": 0.0, "passed": true, "example_rows": []}
      // ... 20 more rules (here one soft warning: WD-S01, night nurses above day nurses, on row 82)
    ]
  },
  "quality": {
    "score": 0.9738,
    "reference": {"split": "train", "subset": "matching", "reference_rows": 1003, "rows_compared": 1003, "synthetic_rows_compared": 200},
    "metrics": {"name": "quick_quality", "score": 0.973838,
                "components": {"column_shapes": 0.950875, "column_pair_trends": 0.970639, "validity": 1.0},
                "worst_columns": [{"column": "month", "score": 0.858863} /* ... */]
                // ... description, table, validity_note, column_scores, worst_pairs, excluded_columns,
                //     n_rows, metric_descriptions, seconds
    }
  },
  "columns": [
    {"name": "record_id", "dtype": "str", "role": "key", "unit": null, "description": "Record identifier, e.g. WD-000001."},
    {"name": "ward_id", "dtype": "str", "role": "train", "unit": null, "description": "Ward code."}
    // ... 33 more
  ],
  "warnings": [],
  "links": {"self": "/api/v1/datasets/ds-dbd4a91c4059",
            "download_csv": "/api/v1/datasets/ds-dbd4a91c4059/download?format=csv",
            "download_json": "/api/v1/datasets/ds-dbd4a91c4059/download?format=json"},
  "records": [
    {"record_id": "WD-000001", "ward_id": "AMU", "ward_type": "acute", "month": 4, "season": "spring", "day_of_week": "Fri",
     "is_weekend": false, "bed_capacity": 28, "escalation_beds_open": 1, "beds_closed": 1, "available_beds": 28,
     "staffing_level": "short", "nurses_day_shift": 5, "nurses_night_shift": 3, "census_start": 23,
     "elective_scheduled": 0, "elective_cancelled": 0, "admissions_elective": 0, "admissions_emergency": 20,
     "transfers_in": 0, "transfers_out": 4, "discharges": 11, "discharges_before_noon": 4, "census_end": 28,
     "delayed_discharges": 3, "emergency_boarders": 2, "avg_bed_wait_hours": 1.59, "avg_bed_turnaround_min": 104.6,
     "avg_los_days": 1.37, "occupancy_rate": 1.0, "occupancy_level": "critical", "bed_turnover_rate": 0.5357,
     "patients_per_nurse": 5.6, "delayed_discharge_rate": 0.1304, "elective_cancellation_rate": 0.0}
    // ... 199 more
  ]
}
```

A patient-stays record has this shape:

```json
{"stay_id": "ST-0000001", "ward_id": "PAED", "ward_type": "paediatric", "admission_source": "emergency", "admission_month": 3,
 "admission_season": "spring", "admission_day_of_week": "Thu", "admission_hour": 18, "booking_lead_days": 0,
 "times_rescheduled": 0, "bed_wait_hours": 0.9, "los_nights": 2, "discharge_day_of_week": "Sat", "discharge_hour": 14,
 "los_hours": 44, "los_days": 1.833, "separation_type": "discharge", "discharge_destination": "home",
 "delay_reason": "none", "discharge_delay_hours": 0.0, "is_delayed_discharge": false,
 "discharged_before_noon": false, "is_weekend_discharge": true, "bed_turnaround_minutes": 61}
```

### 2.10 `POST /api/v1/validate`: validate client records

Checks records against the hard and soft rules. It is useful for data from another generator, an
optimisation model's output or a hand-made test case. [Section 4.1](#41-the-validate-pipeline) describes
the pipeline.

| Request field | Type | Default | Validation | Meaning |
|---|---|---|---|---|
| `table` | `"ward_daily"` or `"patient_stays"` | required | | table of the records |
| `records` | array of objects | required | 1 to 10,000 items (`api.max_rows`); every learned column is required (400 `invalid_records` lists the missing ones) | rows to check; derived columns are optional |
| `include_soft` | boolean | `true` | | also evaluate the soft rules |
| `return_records` | boolean | `false` | | return the normalised records with derived columns completed |

| Response field | Meaning |
|---|---|
| `table`, `n_rows` | as requested |
| `validation` | validation summary ([section 3.9](#39-validation-summary)). The hard rules include the API check `WD-D01` or `PS-D01` (`derived_values_consistent`), so `hard_rules.checked` is 16 for ward-days and 15 for stays |
| `normalisation` | `values_normalised` (aliases mapped, per column), `unknown_values` (outside the vocabulary; kept, they fail `WD-H01`/`PS-H01`), `invalid_numbers` (non-numeric values treated as missing), `unknown_columns` (ignored), `derived_computed` (derived columns the API filled in), `derived_mismatches` and `derived_mismatch_rows` (provided derived values that disagree with their definition: count and the first 100 positions per column) |
| `n_invalid_rows`, `invalid_rows` | rows failing at least one hard rule, and their 0-based positions (first 100) |
| `warnings` | one line per kind of problem found |
| `records` | normalised records when `return_records` is true, otherwise `null` |

**Status codes:** 200; 400 `invalid_records` (missing learned columns, or derived columns that cannot
be computed: "Could not compute the derived columns of these records (<ExceptionType>); check the
column types against GET /api/v1/tables/{table}/schema."); 422 `validation_error` (for example an
empty `records` list); 500 `validation_failed` ("Validation failed (<ExceptionType>); see the server
log."); 503 `validation_unavailable`.

**Example:** the first record is valid. The second uses aliases (`"gen med"`, `"monday"`, `"safe"`) and
cancels three electives when only one was booked (this is the `ward_days` example body in `/docs`).

```jsonc
// request: {"table": "ward_daily", "return_records": true, "records": [{...valid ward-day...}, {...same ward-day with
//           "ward_id": "gen med", "day_of_week": "monday", "staffing_level": "safe", "elective_scheduled": 1, "elective_cancelled": 3}]}
{
  "table": "ward_daily",
  "n_rows": 2,
  "validation": {
    "n_rows": 2, "n_hard_valid": 1, "hard_valid_rate": 0.5, "all_hard_rules_passed": false,
    "hard_rules": {"checked": 16, "passed": 13, "failed": 3, "violations": 3},
    "soft_rules": {"checked": 6, "passed": 6, "failed": 0, "violations": 0},
    "rules": [
      {"rule_id": "WD-H03", "name": "cancellations_within_bookings", "severity": "hard",
       "description": "A ward cannot cancel more elective admissions than were booked for the day.",
       "columns": ["elective_cancelled", "elective_scheduled"], "violations": 1, "violation_rate": 0.5,
       "passed": false, "example_rows": [1]}
      // ... WD-H02 (admissions_elective = 1 - 3 is negative) and WD-H10 (the band no longer matches the census) also fail
    ]
  },
  "normalisation": {
    "values_normalised": {"ward_id": 1, "day_of_week": 1, "staffing_level": 1},
    "unknown_values": {}, "invalid_numbers": {}, "unknown_columns": [],
    "derived_computed": ["ward_type", "season", "is_weekend", "bed_capacity", "available_beds", "admissions_elective", "census_end",
                         "occupancy_rate", "bed_turnover_rate", "patients_per_nurse", "delayed_discharge_rate", "elective_cancellation_rate"],
    "derived_mismatches": {}, "derived_mismatch_rows": {}
  },
  "n_invalid_rows": 1,
  "invalid_rows": [1],
  "warnings": [],
  "records": [ /* ... both rows with canonical codes ("GMED", "Mon", "adequate") and all derived columns */ ]
}
```

### 2.11 `POST /api/v1/evaluate`: quick quality of records or a stored dataset

Scores synthetic rows against the reference ("real") data and, by default, validates them.
[Section 4.2](#42-the-evaluate-endpoint) explains which reference rows are compared.

| Request field | Type | Default | Validation | Meaning |
|---|---|---|---|---|
| `records` | array of objects or null | `null` | 20 to 10,000 items | rows to evaluate (every learned column required) |
| `dataset_id` | string or null | `null` | pattern `^ds-[0-9a-f]{12}$` | a stored dataset to evaluate instead |
| `table` | table name or null | `null` | required with `records`; with `dataset_id` it must match the dataset (400 `table_mismatch`) | table of the rows |
| `scenario` | string or null | `null` | records only (422 with `dataset_id`); at most 64 characters | compare with the reference rows of this scenario |
| `reference_split` | `"train"` or `"holdout"` | `"train"` | | reference split: the data the models learned from, or the held-out 20 % |
| `max_rows` | integer | 5,000 | 20 to 5,000 (`api.evaluate_max_rows`) | subsample both sides to at most this many rows |
| `seed` | integer | `0` | 0 to 4,294,967,295 | subsampling seed |
| `include_validation` | boolean | `true` | | also run the rule validation |

Exactly one of `records` and `dataset_id` must be given (422 otherwise).

| Response field | Meaning |
|---|---|
| `table` | table evaluated |
| `source` | `{type: "records" or "dataset", dataset_id, n_rows}` |
| `quality` | quality block ([section 3.10](#310-quality-block)) |
| `validation` | validation summary, or `null` when `include_validation` is false. For records it includes `WD-D01`/`PS-D01` |
| `normalisation` | as for `/validate` (records only; `null` for datasets) |
| `warnings` | for example, the rows that violate hard rules (`"20 rows violate hard rules (WD-H02, WD-H03, WD-H10)."`), falling back to the full reference split, or problems with the records |

**Status codes:** 200; 400 `too_few_rows`, `table_mismatch`, `invalid_records`,
`scenario_table_mismatch`, `evaluation_failed` (the quality computation rejected the records:
"Quality evaluation failed (<ExceptionType>); check that the records match the table schema (GET
/api/v1/tables/{table}/schema). Details are in the server log."); 404 `dataset_not_found`,
`scenario_not_found`; 422 `validation_error`; 500 `dataset_corrupt`, `evaluation_failed` (unexpected
failure: "Quality evaluation failed (<ExceptionType>); see the server log."); 503
`reference_data_unavailable`, `evaluation_unavailable`.

```bash
curl -s -X POST http://127.0.0.1:8000/api/v1/evaluate -H "Content-Type: application/json" -d '{"dataset_id": "ds-dbd4a91c4059"}'
```

```jsonc
{
  "table": "ward_daily",
  "source": {"type": "dataset", "dataset_id": "ds-dbd4a91c4059", "n_rows": 200},
  "quality": {
    "score": 0.9738,
    "reference": {"split": "train", "subset": "matching", "reference_rows": 1003, "rows_compared": 1003, "synthetic_rows_compared": 200},
    "metrics": { /* quick_quality output, see section 3.10 */ }
  },
  "validation": {"n_rows": 200, "n_hard_valid": 200, "hard_valid_rate": 1.0, "all_hard_rules_passed": true,
                 "hard_rules": {"checked": 15, "passed": 15, "failed": 0, "violations": 0},
                 "soft_rules": {"checked": 6, "passed": 5, "failed": 1, "violations": 1}, "rules": [ /* ... */ ]},
  "normalisation": null,
  "warnings": []
}
```

### 2.12 `GET /api/v1/datasets`: list stored datasets

| Query parameter | Type | Default | Meaning |
|---|---|---|---|
| `table` | string | all | `ward_daily` or `patient_stays` (404 `table_not_found` otherwise) |
| `limit` | integer | 50 | 1 to 500 (422 otherwise); newest datasets first |

The response has `total` (datasets matching the filter), `count` (datasets returned) and `datasets[]`.
Each summary has `dataset_id`, `created_at`, `table`, `scenario`, `model_id`, `mode`, `n_rows`, `seed`,
`all_hard_rules_passed`, `quality_score` (`null` without `evaluate`), `size_bytes` (stored CSV) and
`links`. Datasets whose metadata cannot be read are left out of the listing.

```jsonc
{
  "total": 4,
  "count": 4,
  "datasets": [
    {"dataset_id": "ds-a8addad76b28", "created_at": "2026-11-15T10:48:53+00:00", "table": "ward_daily",
     "scenario": "emergency_pressure", "model_id": "gc_tuned", "mode": "sample", "n_rows": 100, "seed": 3,
     "all_hard_rules_passed": true, "quality_score": 0.9278, "size_bytes": 14445,
     "links": {"self": "/api/v1/datasets/ds-a8addad76b28",
               "download_csv": "/api/v1/datasets/ds-a8addad76b28/download?format=csv",
               "download_json": "/api/v1/datasets/ds-a8addad76b28/download?format=json"}}
    // ... 3 more (a calendar dataset has "mode": "calendar" and, without evaluate, "quality_score": null)
  ]
}
```

### 2.13 `GET /api/v1/datasets/{dataset_id}`: dataset metadata

Returns everything stored with the dataset except its records. This is the JSON generation response
without `records`, plus `size_bytes` and `links`. It is the only place where the full report of a CSV
generation can be read. Returns 404 `dataset_not_found` if the dataset does not exist or the id is
malformed, and 500 `dataset_corrupt` if its metadata file cannot be read.

### 2.14 `GET /api/v1/datasets/{dataset_id}/download`: download records

| Query parameter | Values | Default |
|---|---|---|
| `format` | `csv` or `json` (422 otherwise) | `csv` |

* **CSV:** `text/csv; charset=utf-8` with the `X-BedSynth-*` headers and
  `Content-Disposition: attachment; filename="<dataset_id>.csv"`. The body is the stored file. It is
  written with LF line endings on every platform, so it is byte-identical to the CSV body that
  `/generate` returns for the same records (checked: the download of the section 2.9 example and the
  CSV body of the same request were the same 28,356 bytes).
* **JSON:** `{dataset_id, table, created_at, n_rows, columns, records}` with
  `Content-Disposition: attachment; filename="<dataset_id>.json"`.

Errors: 404 `dataset_not_found`, 422 `validation_error`, 500 `dataset_corrupt`.

```text
HTTP/1.1 200 OK
content-type: text/csv; charset=utf-8
content-disposition: attachment; filename="ds-dbd4a91c4059.csv"
x-bedsynth-dataset-id: ds-dbd4a91c4059
x-bedsynth-table: ward_daily
x-bedsynth-rows: 200
x-bedsynth-model: gc_tuned
x-bedsynth-scenario: high_occupancy
x-bedsynth-seed: 42
x-bedsynth-persisted: true
x-bedsynth-hard-rules-passed: true
x-bedsynth-quality-score: 0.9738

record_id,ward_id,ward_type,month,season,day_of_week,is_weekend,bed_capacity,escalation_beds_open,...
```

### 2.15 `DELETE /api/v1/datasets/{dataset_id}`: delete a dataset

Removes the dataset's CSV and metadata and returns `{"dataset_id": "ds-...", "deleted": true}`.
Deleting it again returns 404 `dataset_not_found`. If the files are locked (on Windows, for example,
when another program has the CSV open), the API retries briefly and then returns 409 `dataset_busy`.

---

## 3. Generation in depth

### 3.1 Sample mode and calendar mode

| | `"sample"` (default) | `"calendar"` (`ward_daily` only) |
|---|---|---|
| Rows | `num_rows` | one per ward per date from `start_date` to `end_date` inclusive (`wards` or all eight wards) |
| Extra column | none | `date` (ISO), placed after `record_id` |
| Row order | seeded shuffle | sorted by `ward_id`, then `date` |
| `month`, `day_of_week` | sampled | fixed by each date |
| Allocation | rows split across condition combinations in proportion to their reference frequency | days grouped by (ward, month, weekday); remaining conditions split each group's days, backing off to coarser groups when a group has no reference support |
| Report | `condition_allocation` per requested combination | `condition_allocation` per ward (plus remaining conditions), and a `calendar` block |

In calendar mode, a scenario or condition on `month` or `day_of_week` removes the non-matching days and
adds a warning. For example, `weekend_operations` over January 2025 for two wards keeps 16 of the 62
ward-days ("46 of 62 ward-days in the calendar range do not match the conditions on month/day_of_week
and were dropped."). A range with no matching day returns 400 `generation_error`. Rows are independent
draws: the `census_end` of one day is not carried into the next day's `census_start`.

```bash
curl -s -X POST http://127.0.0.1:8000/api/v1/generate -H "Content-Type: application/json" \
     -d '{"table": "ward_daily", "mode": "calendar", "wards": ["ICU"], "start_date": "2025-01-01", "end_date": "2025-01-14", "seed": 21}'
```

```jsonc
{
  "n_rows": 14,
  "generation": {
    "mode": "calendar", "num_rows_requested": 14, "num_rows_generated": 14, "shortfall": 0,
    "reference_source": "not_needed",
    "condition_allocation": [{"conditions": {"ward_id": "ICU"}, "requested": 14, "generated": 14, "cells": 12, "sampled": 30, "accepted": 26}],
    "calendar": {"start_date": "2025-01-01", "end_date": "2025-01-14", "days": 14, "wards": ["ICU"],
                 "grid_rows": 14, "dropped_rows": 0, "cells": 7}
    // ...
  },
  "records": [
    {"record_id": "WD-000001", "date": "2025-01-01", "ward_id": "ICU", "month": 1, "day_of_week": "Wed", "census_start": 5, "census_end": 7, /* ... */ "occupancy_level": "low"},
    {"record_id": "WD-000002", "date": "2025-01-02", "ward_id": "ICU", "month": 1, "day_of_week": "Thu", "census_start": 11, "census_end": 10, /* ... */ "occupancy_level": "normal"}
    // ... 12 more
  ]
  // ...
}
```

The first day closes with 7 patients and the second opens with 11: calendar rows are not chained.

In the request echo, `num_rows` keeps its default (100) in calendar mode. The planned row count is
`generation.num_rows_requested`.

### 3.2 Scenario, wards, conditions and filters

| Input | Columns | How it is applied | How it combines |
|---|---|---|---|
| `scenario` | its own conditions and filters ([section 8](#8-scenario-catalogue)) | as below | the starting point |
| `wards` | `ward_id` | as a condition | intersected with the scenario's and the request's `ward_id` values |
| `conditions` | categorical learned columns (`conditionable_columns`) | conditions on the segment columns pick the segment models; the rest are served by exact pooled rejection from one unconditional draw per segment (GaussianCopula's native conditional sampling only for the rarest combinations, which a pooled draw of at most 50,000 rows cannot serve or which have no training support); every accepted row is checked to match the combination it was drawn for | intersected per column with the scenario's conditions; columns are combined as a Cartesian product |
| `filters` | any learned or derived column (`filterable_columns`) | rejection after sampling, on the derived table | merged with the scenario's filters: the highest `min`, the lowest `max`, `in` lists intersected, `not_in` lists united |

**Conditions.**

* **Values.** A single value or a list. Values are normalised: aliases and case are mapped
  (`"monday"` becomes `"Mon"`, `"icu"` becomes `"ICU"`) and numeric strings become integers for `month`
  and `admission_month` (`"12"` becomes `12`; a JSON float such as `12.0` is a 422).
* **Errors.** An unknown value, or a column that is not conditionable, returns 400 `invalid_conditions`.
* **Redundant columns.** A column whose list covers its whole vocabulary means "no condition" and is
  dropped with a warning ("Condition on 'staffing_level' lists every value; it was dropped (same as no
  condition).").
* **Combinations.** After merging with the scenario and `wards`, the product of the value counts may not
  exceed 1,000 (`api.max_condition_combinations`), or the request fails with 400 `too_many_combinations`.
  The `too_many_combinations` example in `demo/requests.http` expands into 11 x 6 x 7 x 2 x 3 = 2,772
  combinations.
* **Contradictions.** If the intersection leaves a column with no value (for example `high_occupancy`
  with `{"occupancy_level": "low"}`, or `wards: ["AMU"]` with `{"ward_id": "ICU"}`), the request fails
  with 400 `generation_error` ("Conditions do not overlap ...").
* **Allocation.** Each combination is a *condition cell*. Rows are allocated to cells in proportion to
  each combination's frequency in the reference train split (largest-remainder method; reported as
  `reference_count`). A combination the model never saw in training gets no rows: its
  `condition_allocation` entry has `"feasible": false` and a warning lists it. For example,
  `delayed_discharge_stays` with two discharge destinations expands into 12 combinations, of which 6
  are infeasible.

**Filters.** A filter is `{"min": x, "max": y, "in": [...], "not_in": [...]}` with any non-empty subset
of the keys; bounds are inclusive. The API checks filters before any sampling:

| Problem | Result |
|---|---|
| column not filterable (`{"foo": {"min": 1}}`) | 400 `invalid_filters` |
| `min`/`max` on a non-numeric column (`month` and `admission_month` are allowed) | 400 `invalid_filters` |
| bound outside the column's physical range (`{"occupancy_rate": {"min": 1.5}}`) | 400 `invalid_filters` |
| `in`/`not_in` value that does not exist (`{"day_of_week": {"in": ["Funday"]}}`) | 400 `invalid_filters` |
| filter that excludes every value of a categorical or boolean column | 400 `invalid_filters` |
| `min` greater than `max`, empty filter `{}`, unknown key (`"gt"`), NaN, infinite or boolean bound | 422 `validation_error` |

`in`/`not_in` values are converted to the column type: aliases become canonical codes
(`{"discharge_day_of_week": {"in": ["saturday", "Sunday"]}}` becomes `["Sat", "Sun"]`), `"true"` becomes
a boolean and numeric strings become numbers. Filters that are possible one by one but impossible
together (for example `occupancy_rate` at least 0.99 and `census_end` at most 0) pass these checks.
They end in 422 `no_rows_generated` after sampling.

**Echo and effective values.** `request.conditions` and `request.filters` echo only the request's own
values, normalised. `generation.conditions` and `generation.filters` show the effective merged values.
For example, `scheduled_procedures` plus `{"elective_cancelled": {"min": 1}}` echoes the one request
filter, while the report shows both `elective_scheduled >= 2` (from the scenario) and
`elective_cancelled >= 1`.

**Ward codes.**

| Code | Ward | Beds | Admission sources |
|---|---|---|---|
| `AMU` | Acute Medical Unit | 28 | emergency |
| `GMED` | General Medicine | 32 | elective, emergency, transfer |
| `ELDC` | Elderly Care | 30 | emergency, transfer |
| `CARD` | Cardiology | 20 | elective, emergency, transfer |
| `GSUR` | General Surgery | 30 | elective, emergency, transfer |
| `ORTH` | Orthopaedics | 24 | elective, emergency, transfer |
| `ICU` | Intensive Care Unit | 12 | elective, emergency, transfer |
| `PAED` | Paediatrics | 18 | elective, emergency |

### 3.3 Reproducibility and seeds

* **Same seed, same records.** The same `seed`, model and request return the same records, provided the
  request finishes within its time budget. In a check, two runs of
  `{"table": "ward_daily", "num_rows": 50, "scenario": "winter_demand", "seed": 7}` returned identical
  records. Only `dataset_id` (always new) and `created_at` (the time of the run) change.
* **Seed omitted.** When `seed` is omitted, a seed is drawn and reported in `generation.seed` (and the
  `X-BedSynth-Seed` header); `request.seed` stays `null`. Send that seed back to reproduce the run.
* **Pin the model.** Pass `model_id` explicitly to keep a seed reproducible over time:
  `scripts/03_evaluate.py` can change the registry default, and retraining changes the model itself.
* **Machine speed.** A run is reproducible only if it was not shaped by the time budget. If
  `generation.stopped_reason` is set or `generation.throttled_rounds` is above 0, the result depends on
  the machine's speed.
* **Why generations run one at a time.** SDV sampling swaps the process-wide random state, so the
  service runs generations one at a time to keep seeds meaningful ([section 5](#5-dataset-lifecycle-and-concurrency)).

### 3.4 Output options: `include_derived`, `validate`, `evaluate`, `persist`

| Option | `true` | `false` |
|---|---|---|
| `include_derived` (default `true`) | key, learned and derived columns (`date` too in calendar mode) | key and learned columns only (23 columns for ward-days, plus `date` in calendar mode). Validation and quality still run on recomputed derived values |
| `validate` (default `true`) | `validation` block; in CSV, the `X-BedSynth-Hard-Rules-Passed` header | `validation: null`, no hard-rule header. Rows still pass every hard rule: rejection sampling does not depend on this flag |
| `evaluate` (default `false`) | `quality` block against the matching reference rows; in CSV, `X-BedSynth-Quality-Score`. With fewer than 20 rows the score is skipped with the warning "Quality evaluation skipped: Quality needs at least 20 synthetic rows to be meaningful; got 10." | `quality: null` |
| `persist` (default `true`) | dataset stored, `links` filled | `persisted: false`, `links: {}`; the `dataset_id` is reported but cannot be fetched |

If `api.persist_datasets` is `false`, nothing is stored and the response warns "Dataset persistence is
disabled on this server (settings.api.persist_datasets).". If writing fails, the response is still
returned with `persisted: false` and the warning "The dataset could not be stored (...)".

### 3.5 CSV output and the X-BedSynth headers

With `"format": "csv"`, the response body is the records as `text/csv; charset=utf-8` (LF line endings;
booleans written as `True`/`False`), with `Content-Disposition: attachment; filename="<dataset_id>.csv"`.
The report, validation and quality are not in the body. They are still computed, and they are stored
with the dataset when it is persisted: read them with `GET /api/v1/datasets/{dataset_id}`. With
`persist: false` and CSV output, only the headers survive.

| Header | Value | Present |
|---|---|---|
| `X-BedSynth-Dataset-Id` | `ds-...` | always |
| `X-BedSynth-Table` | table | always |
| `X-BedSynth-Rows` | rows returned; compare with `num_rows` to detect a shortfall | always |
| `X-BedSynth-Model` | experiment id | always |
| `X-BedSynth-Scenario` | scenario name, or empty | always |
| `X-BedSynth-Seed` | effective seed | always |
| `X-BedSynth-Persisted` | `true` or `false` | always |
| `X-BedSynth-Hard-Rules-Passed` | `true`, `false` or `unknown` (no rows) | when `validate` is true |
| `X-BedSynth-Quality-Score` | score with four decimals | when a quality score was computed |

The same headers accompany `GET /api/v1/datasets/{dataset_id}/download?format=csv`.

```bash
curl -s -D - -o emergency_pressure.csv -X POST http://127.0.0.1:8000/api/v1/generate -H "Content-Type: application/json" \
     -d '{"table": "ward_daily", "num_rows": 100, "scenario": "emergency_pressure", "format": "csv", "seed": 3, "evaluate": true}'
```

```text
HTTP/1.1 200 OK
content-type: text/csv; charset=utf-8
content-disposition: attachment; filename="ds-a8addad76b28.csv"
x-bedsynth-dataset-id: ds-a8addad76b28
x-bedsynth-table: ward_daily
x-bedsynth-rows: 100
x-bedsynth-model: gc_tuned
x-bedsynth-scenario: emergency_pressure
x-bedsynth-seed: 3
x-bedsynth-persisted: true
x-bedsynth-hard-rules-passed: true
x-bedsynth-quality-score: 0.9278
x-process-time-ms: 4538.6
```

### 3.6 Time budget and `max_attempts`

Generation is rejection sampling. In each round the model is asked for a batch of rows per condition
cell. The rows are derived and checked against every hard rule, their cell's conditions and the filters,
and accepted rows fill the cell's quota. The next batch adapts to the acceptance observed so far.

* **`time_budget_seconds`** (default 60 s, at most 120 s) bounds sampling.
  * **What counts.** The budget runs from the start of the engine call and is checked before every
    round. For filtered requests, the strata pilot draw is sized to fit in about a quarter of it. It
    does not cover waiting in the generation queue, validation, quality or storage, so the response can
    take longer than the budget.
  * **Bounded rounds.** One round cannot overrun the budget by much: the rows requested in a round are
    capped at `generation.max_round_rows` (200,000) and at what the remaining time allows at the
    throughput measured so far. Rounds scaled down this way are counted in `throttled_rounds`.
  * **Running out.** When the budget runs out the loop stops and keeps the rows accepted so far.
    `stopped_reason` is set and the response warns, for example "Rejection loop stopped early: time
    budget of 1s exhausted after 1.1s (1 round(s) completed)."
* **`max_attempts`** (default 25 from `generation.max_attempts`, at most 100) is the maximum number of
  rounds. Raise it for rare scenarios. Scenario `support.share` shows how rare a scenario is in the
  reference data; for example, `rescheduled_electives` matches about 1 % of reference stays.
* **Futility rule.** The engine gives up on a combination that has produced no acceptable row after
  `max(2,000, 50 x its quota)` sampled rows (capped by `generation.max_batch_rows`) and says so in a
  warning ("Stopped sampling {...}: no row satisfied the validation rules and filters after N sampled
  rows ...").

### 3.7 Shortfall, empty results and warnings

A request that cannot deliver every row is not an error. The response is a normal 200 in which:

* `n_rows`, `generation.num_rows_generated` and `X-BedSynth-Rows` are lower than requested;
* `generation.shortfall` is above 0;
* a warning explains why and what to change.

The advice depends on the cause:

```jsonc
// {"table": "ward_daily", "num_rows": 300, "scenario": "emergency_pressure", "max_attempts": 1, "seed": 4}
"warnings": ["Generated 120 of 300 requested rows after 1 attempt(s) (2 condition combination(s) short); the scenario is rare under model 'gc_tuned': increase max_attempts or relax the conditions/filters."]

// {"table": "patient_stays", "num_rows": 300, "scenario": "long_bed_waits", "time_budget_seconds": 1, "seed": 17}
"warnings": ["Rejection loop stopped early: time budget of 1s exhausted after 1.1s (3 round(s) completed).",
             "Generated 282 of 300 requested rows after 3 attempt(s) (1 condition combination(s) short); increase time_budget_seconds, request fewer rows or relax the conditions/filters."]
```

If **no** row can be generated, the API returns 422 `no_rows_generated`, stores nothing, and quotes up to
three of the request's warnings. For example, AMU never books electives, so
`{"table": "ward_daily", "num_rows": 50, "wards": ["AMU"], "filters": {"elective_scheduled": {"min": 1}}}`
gives up after about half a second:

```json
{
  "detail": "No rows could be generated for this request, so no dataset was stored. Relax the scenario, conditions or filters (or raise time_budget_seconds / max_attempts). Engine report: Stopped sampling {'ward_id': 'AMU'}: no row satisfied the validation rules and filters after 6080 sampled rows (6080 requested). Generated 0 of 50 requested rows after 2 attempt(s) (1 condition combination(s) short); the scenario is rare under model 'gc_tuned': increase max_attempts or relax the conditions/filters. Low acceptance rate (0.0% < 5%): most sampled rows violated hard validation rules or the scenario conditions/filters.",
  "error_type": "no_rows_generated"
}
```

With `format: "csv"` the warnings are not in the response. Check `X-BedSynth-Rows`, and read the stored
metadata if the dataset was persisted.

| Warning (abridged) | Meaning |
|---|---|
| `Generated X of Y requested rows after N attempt(s) ...` | shortfall; the advice depends on the cause (time budget or rare scenario) |
| `Rejection loop stopped early: time budget of ...` | the time budget ran out (`stopped_reason`) |
| `Low acceptance rate (x% < 5%) ...` | fewer than `generation.low_acceptance_warning` (5 %) of sampled rows were accepted |
| `Model '...' cannot generate k of n condition combination(s) ...` | combinations absent from the model's training data received no rows |
| `Stopped sampling {...}: no row satisfied ...` | futility rule ([section 3.6](#36-time-budget-and-max_attempts)) |
| `N of M ward-days in the calendar range do not match ...` | calendar days dropped by month or weekday conditions |
| `N calendar day(s) fall in ... combinations that model '...' never saw ...` | calendar days that cannot be generated |
| `N of M rows (x%) were delivered in allocation strata beyond their reference quota ...` | more than 10 % of rows were rebalanced between strata ([section 3.8](#38-generation-report)) |
| `Condition on '...' lists every value; it was dropped ...` | redundant condition removed |
| `Reference train split not available: rows were allocated using ...` (calendar mode: `... calendar days were allocated using ...`) | allocation used the model's training frequencies or uniform weights |
| `No reference support for the requested conditions: rows were allocated uniformly.` | none of the combinations occurs in the reference data |
| `Quality evaluation skipped: ...` | `evaluate` was requested but no score could be computed (fewer than 20 rows, reference data missing, ...) |
| `Only N reference rows match the scenario/conditions (minimum 20); quality was measured against the full <split> split.` | matching subset too small |
| `Validation skipped: ...` | the rule validation failed; the rows were still filtered by the hard rules during sampling |
| `Dataset persistence is disabled on this server ...` / `The dataset could not be stored (...)` | not persisted |
| `... N more warnings omitted.` | more than 25 warnings |

### 3.8 Generation report

`generation` is the engine's report, made JSON-safe with server paths removed. It is stored with every
persisted dataset.

| Field | Meaning |
|---|---|
| `table`, `mode`, `scenario` | as resolved (`scenario` is the name, or `null`) |
| `model_id`, `model` | the experiment that sampled: `id`, `experiment_id`, `synthesizer`, `model`, `segment_by`, `constraint_groups`, `sdv_version`, `fitted_at` |
| `seed` | effective seed (drawn when the request had none) |
| `num_rows_requested`, `num_rows_generated`, `shortfall` | planned rows (calendar mode: the ward-days left after dropping), rows delivered, and the difference |
| `attempts`, `max_attempts` | rejection rounds run, and the limit |
| `sampled_rows` | rows drawn from the model over all rounds |
| `accepted_rows` | sampled rows that passed every hard rule, their cell's conditions and the filters (surplus rows beyond a quota included) |
| `acceptance_rate` | `accepted_rows / sampled_rows` |
| `raw_hard_valid_rate` | share of sampled rows passing every hard rule, before conditions and filters |
| `rejected_by_rule` | `{rule_id: violations}` among sampled rows (hard rules only). A row that breaks two rules counts under both, so the sum can exceed the number of rejected rows |
| `rejected_by_filter` | hard-valid rows rejected because they missed their cell's conditions or the filters |
| `first_batch_factor`, `acceptance_prior` | oversampling of the first round, and the model's raw hard-valid rate recorded after training that sized it |
| `time_budget_seconds`, `stopped_reason`, `throttled_rounds`, `rows_per_second` | effective budget; why the loop stopped early (`null` unless the budget ran out); rounds scaled down to fit the round cap or the remaining time; measured sampling throughput |
| `elapsed_seconds` | engine time for the request |
| `conditions`, `filters` | effective merged conditions and filters ([section 3.2](#32-scenario-wards-conditions-and-filters)) |
| `reference_source` | where the combination weights came from: `reference_train_split`, `model_training_data`, `uniform`, or `not_needed` (a single combination) |
| `condition_allocation` | sample mode: one entry per requested combination with `conditions`, `requested`, `generated`, `reference_count`, `sampled`, `accepted` and `feasible`. Calendar mode: one entry per ward (and remaining conditions) with `requested`, `generated`, `cells`, `sampled` and `accepted`. At most 200 entries are kept; when there are more, `condition_allocation_total` gives the full count |
| `allocation_strata` | see below |
| `calendar` | calendar mode only: `start_date`, `end_date`, `days`, `wards`, `grid_rows`, `dropped_rows`, `cells` |
| `include_derived` | as requested |
| `warnings` | the engine's warnings (the top-level `warnings` adds request, validation and quality warnings) |

**`allocation_strata`.** Rejection sampling accepts some kinds of rows more easily than others: a calm
ward-day passes the rules more often than a critical one. Filling one quota per condition cell would
therefore skew the output towards the easy rows. To prevent this, each cell is split into implicit
*strata*:

* `ward_id` x `occupancy_level` for ward-days and `ward_id` x `admission_source` x `delay_reason` for
  stays (`generation.allocation_strata`; columns already fixed by the conditions are not stratified);
* each stratum's quota follows the reference frequencies;
* requests with filters never read the matching real rows: their strata weights come from a seeded
  synthetic pilot draw of the model.

| Field | Meaning |
|---|---|
| `columns` | strata columns of the table |
| `source` | `reference_train_split`, `model_training_data`, `model_pilot_sample` (filtered requests), `unavailable (no time for a pilot draw)`, or `null` when stratification is disabled (`[]` in the settings) |
| `strata_cells` | strata created (0 when the conditions fix every strata column) |
| `stratified_combinations` | condition combinations that were split into strata |
| `rebalanced_rows` | delivered rows that sit in a stratum beyond its original quota. Without a shortfall, `rebalanced_rows / rows` is the total variation distance between the output's strata mix and the reference allocation. A warning appears above 10 % |
| `quota_moves` | quota moved between sibling strata (one row's quota can move twice) |
| `pilot_rows` | filtered requests only: synthetic pilot rows that satisfied the filters |

### 3.9 Validation summary

The same structure is used by `/generate`, `/validate` and `/evaluate`.

| Field | Meaning |
|---|---|
| `n_rows`, `n_hard_valid`, `hard_valid_rate` | rows, rows passing every hard rule, and their share (`null` for 0 rows) |
| `all_hard_rules_passed` | `true` when every row passes every hard rule; `null` when there are no rows |
| `hard_rules`, `soft_rules` | `{checked, passed, failed, violations}` over the rules of each severity |
| `rules[]` | `rule_id`, `name`, `severity`, `description`, `columns`, `violations` (rows violating the rule), `violation_rate`, `passed`, `example_rows` (up to five 0-based positions in `records`) |

Hard rules describe operationally impossible rows. Generated data passes all of them by construction.
Soft rules describe implausible but possible rows; they are reported, never enforced. For example, in
an unconditional sample of 120 stays (seed 11) two stays failed `PS-S03 short_elective_wait` (an
elective patient waited more than four hours for a bed), with `"example_rows": [17, 49]`.
[Section 4.3](#43-rule-identifiers) lists the rule ids.

### 3.10 Quality block

`quality` (from `/generate` with `evaluate: true`, and from `/evaluate`) is a fast fidelity estimate,
not the full Week-3 evaluation.

| Field | Meaning |
|---|---|
| `score` | mean of the available components, rounded to four decimals, in [0, 1] |
| `reference.split` | `train` (always, for `/generate`) or the requested split (`/evaluate`) |
| `reference.subset` | `matching` when only the comparable reference rows were used (at least 20 rows). Otherwise `full`: either nothing restricts the comparison (no scenario conditions or filters, no conditions, filters or wards, no calendar months), or fewer than 20 reference rows match, which adds a warning. Comparable means the same scenario, conditions, filters, wards and calendar months for generations and stored datasets, and the same scenario for `/evaluate` with records |
| `reference.reference_rows`, `rows_compared`, `synthetic_rows_compared` | rows in the chosen subset, and rows compared after subsampling (at most `api.evaluate_max_rows` = 5,000 per side, seeded) |
| `metrics` | output of `bedsynth.evaluation.quick_quality`: `name`, `description`, `table`, `score` (unrounded), `components` (`column_shapes` = mean 1 - KS or 1 - TVD per column, `column_pair_trends` = 1 - mean absolute Spearman difference / 2, `validity` = hard-valid share), `validity_note`, `column_scores`, `worst_columns`, `worst_pairs`, `excluded_columns`, `n_rows`, `metric_descriptions`, `seconds` |

---

## 4. Validating and evaluating client records

### 4.1 The `/validate` pipeline

1. **Required columns.** Every learned column of the table must be present (400 `invalid_records` lists
   the missing ones; `GET /api/v1/tables/{table}/schema` lists them). Columns outside the schema are
   ignored and reported.
2. **Category normalisation.** Aliases, case and spacing are mapped to canonical codes (`"gen med"` to
   `GMED`, `"monday"` to `Mon`, `"safe"` to `adequate`). Unknown values, booleans included, are kept, so
   they fail `WD-H01`/`PS-H01`.
3. **Numeric coercion.** Non-numeric values in numeric columns become missing and are reported in
   `invalid_numbers`. Booleans are not numbers: `true` is never one patient.
4. **Derived values.** Every derived column (flow balance `census_end`, `available_beds`,
   `occupancy_rate`, calendar labels, `los_hours`, ...) is recomputed from the learned columns. A
   provided derived value that disagrees with its definition (tolerance 0.006 for float columns,
   exact otherwise) is recorded in `derived_mismatches`. The row then fails the API's hard check `WD-D01` (ward
   days) or `PS-D01` (stays), `derived_values_consistent`. If the derived columns cannot be computed at
   all, the request fails with 400 `invalid_records`, naming only the exception type and pointing to the
   table schema (the details go to the server log).
5. **Rules.** All rules run on the recomputed derived values, so a wrong provided value cannot hide or
   cause a rule failure. The returned records keep the client's derived values as given and fill in the
   missing ones.

Example: a correct ward-day sent with `"census_end": 31` when the flow balance gives 30.

```jsonc
{
  "n_invalid_rows": 1,
  "invalid_rows": [0],
  "validation": {"hard_rules": {"checked": 16, "passed": 15, "failed": 1, "violations": 1}, /* ... */
                 "rules": [ /* ... */ {"rule_id": "WD-D01", "name": "derived_values_consistent", "severity": "hard", "violations": 1, "passed": false, "example_rows": [0] /* ... */}]},
  "normalisation": {"derived_mismatches": {"census_end": 1}, "derived_mismatch_rows": {"census_end": [0]}, /* ... */},
  "warnings": ["Provided derived values disagree with their schema definition, so those rows fail the hard 'derived_values_consistent' check (the rules ran on the recomputed values): census_end (1)."]
}
```

### 4.2 The `/evaluate` endpoint

* **Stored dataset (`dataset_id`).** The reference subset is rebuilt from the stored request: scenario,
  conditions, filters, wards and, in calendar mode, the months of the period. Evaluating a dataset
  against the train split therefore gives the score computed at generation time; it is identical when
  neither side has more than 5,000 rows, so no subsampling is needed.
* **Records.** Only `scenario` narrows the reference rows. Wards and other conditions are not inferred
  from the records.
* **Subset size.** If fewer than 20 reference rows match, the full split is used and a warning says so.
* **Row count.** Quality needs at least 20 synthetic rows. Records are checked by the schema (422 below
  20); a stored dataset with fewer rows returns 400 `too_few_rows` ("Quality needs at least 20
  synthetic rows to be meaningful; got 10."). `/generate` with `evaluate: true` turns the same
  condition into a warning.
* **Split choice.** `reference_split: "holdout"` compares with the 20 % of reference rows the models
  never saw; `"train"` compares with what they learned.
* **Validation.** `include_validation: true` adds the validation summary. For records it includes
  `WD-D01`/`PS-D01` and the `normalisation` block.

### 4.3 Rule identifiers

Hard rules (`H`) reject rows during generation and appear in `rejected_by_rule`. Soft rules (`S`) are
only reported. `GET /api/v1/tables/{table}/schema` returns each rule's description and columns.

| ward_daily | Name | patient_stays | Name |
|---|---|---|---|
| `WD-H01` | valid_categories | `PS-H01` | valid_categories |
| `WD-H02` | non_negative_integer_counts | `PS-H02` | hours_in_range |
| `WD-H03` | cancellations_within_bookings | `PS-H03` | non_negative_values |
| `WD-H04` | noon_discharges_within_discharges | `PS-H04` | positive_length_of_stay |
| `WD-H05` | delayed_within_census | `PS-H05` | elective_booking_lead |
| `WD-H06` | census_within_available_beds | `PS-H06` | non_elective_unbooked |
| `WD-H07` | census_start_within_physical_beds | `PS-H07` | delay_reason_consistent |
| `WD-H08` | capacity_limits | `PS-H08` | delay_within_stay |
| `WD-H09` | staffing_present | `PS-H09` | separation_destination_consistent |
| `WD-H10` | occupancy_level_consistent | `PS-H10` | delay_reason_allowed |
| `WD-H11` | staffing_level_consistent | `PS-H11` | ward_admission_profile |
| `WD-H12` | ward_admission_profile | `PS-H12` | metric_ranges |
| `WD-H13` | metric_ranges | `PS-H13` | delay_threshold |
| `WD-H14` | wait_definition | `PS-H14` | rebooking_lead |
| `WD-H15` | closures_only_free_beds | `PS-S01` | elective_admission_hours |
| `WD-S01` | night_staffing_within_day | `PS-S02` | daytime_departures |
| `WD-S02` | nurse_workload_plausible | `PS-S03` | short_elective_wait |
| `WD-S03` | boarding_under_pressure | `PS-S04` | no_paediatric_transfers_out |
| `WD-S04` | escalation_under_pressure | `PS-S05` | long_delays_for_complex_discharges |
| `WD-S05` | few_weekend_electives | `PS-D01` | derived_values_consistent (API check, client records only) |
| `WD-S06` | noon_discharge_share_plausible | | |
| `WD-D01` | derived_values_consistent (API check, client records only) | | |

---

## 5. Dataset lifecycle and concurrency

```text
POST /generate (persist: true) --> GET /datasets --> GET /datasets/{id} --> GET /datasets/{id}/download --> DELETE /datasets/{id}
        |                                                                   POST /evaluate {"dataset_id": ...}
        +-- 422 no_rows_generated: nothing stored
```

* **Storage.** Each dataset is a folder `<synthetic_dir>/api/<dataset_id>/` holding `data.csv` and
  `metadata.json` (by default `data/synthetic/api/`; set `BEDSYNTH_SYNTHETIC_DIR` to move it). The
  folder is written to a hidden staging directory and moved into place in one step, so listings never
  show half-written datasets. `data.csv` has LF line endings on every platform, like the CSV body of
  `/generate`.
* **Identifiers.** Ids are `ds-` plus 12 random lowercase hex characters. They are checked against that
  pattern before the file system is touched: a malformed id in a path returns 404 `dataset_not_found`
  ("ids look like 'ds-0123456789ab'"), and in the `/evaluate` body it is a 422.
* **What is stored.** The metadata is the full JSON generation response without `records`, including the
  request echo, the generation report, validation, quality, columns and warnings.
* **Listing.** Newest first (by `created_at`, then id), filtered by `table`, at most `limit` items.
  `total` counts every match. Unreadable datasets are skipped, although `/health` still counts them;
  fetching one gives 500 `dataset_corrupt`, and it can still be deleted.
* **Deleting.** A deleted dataset's folder is first moved aside and then removed, so a concurrent
  listing never sees half of it.

**Concurrency.**

| Situation | Behaviour |
|---|---|
| Several generation requests | Generations run one at a time behind a service-wide lock, because SDV sampling swaps the process-wide random state. At most `api.max_pending_generations` (4) may be running or queued; further requests get 503 `server_busy` at once. A queued request gives up after `api.generation_wait_seconds` (60 s) with 503 `server_busy` ("The generation engine stayed busy for 60 s; retry shortly."). Both carry `Retry-After: 60` |
| Other endpoints during a generation | not queued: `/health`, the catalogue, `/validate`, `/evaluate` and the dataset endpoints keep answering |
| Store reads and writes | every read and change of the store runs under one re-entrant lock |
| Download during a delete | downloads read the whole stored file inside the lock (no file handle outlives the request), so a delete cannot hit an open download |
| Two deletes of one dataset | exactly one 200; the others get 404 `dataset_not_found` |
| Files locked by another program | the delete is retried five times (about half a second in total), then 409 `dataset_busy`; retry later |

Example 503 response (captured by occupying all four admission slots in-process):

```text
HTTP/1.1 503 Service Unavailable
retry-after: 60
content-type: application/json

{"detail": "4 generation requests are already running or queued; retry shortly.", "error_type": "server_busy"}
```

---

## 6. Error model

### 6.1 The error body

Every error produced by the application has the same JSON body:

| Field | Type | Meaning |
|---|---|---|
| `detail` | string | human-readable explanation, often with the fix. Messages of unexpected failures name only the exception type (for example `"The generation engine could not be initialised (RuntimeError); see the server log."`); the exception text and traceback go to the server log. Engine messages that are passed on are scrubbed of server file paths |
| `error_type` | string | stable machine-readable category (table below); branch on this, not on `detail` |
| `errors` | array | 422 `validation_error` only: `{loc, msg, type}` per problem (`loc` such as `["body", "num_rows"]` or `["query", "limit"]`) |

```json
{
  "detail": "num_rows: Input should be greater than or equal to 1; wards: Unknown ward 'MARS'. Known wards: AMU, GMED, ELDC, CARD, GSUR, ORTH, ICU, PAED.",
  "error_type": "validation_error",
  "errors": [
    {"loc": ["body", "num_rows"], "msg": "Input should be greater than or equal to 1", "type": "greater_than_equal"},
    {"loc": ["body", "wards"], "msg": "Unknown ward 'MARS'. Known wards: AMU, GMED, ELDC, CARD, GSUR, ORTH, ICU, PAED.", "type": "value_error"}
  ]
}
```

Malformed JSON is also a 422 `validation_error`. Its `detail` names the character offset of the problem
and its `errors` entry has `"type": "json_invalid"` with the offset in `loc`. For the body
`{"table": "ward_daily", "num_rows": 5,}` (a trailing comma):

```json
{"detail": "body (character 37): JSON decode error", "error_type": "validation_error",
 "errors": [{"loc": ["body", 37], "msg": "JSON decode error", "type": "json_invalid"}]}
```

An internal attribute name is an unknown field: `{"table": "ward_daily", "run_validation": false}` gives
`"run_validation: Extra inputs are not permitted"` (`"type": "extra_forbidden"`); send `validate`
instead (and `in`, not `in_`, inside a filter). The only non-JSON error is the CORS layer's plain-text
400 for a disallowed preflight ([section 1.1](#11-base-url-versioning-and-content-types)).

### 6.2 Error catalogue

| Status | `error_type` | Endpoints | Cause | Remedy |
|---|---|---|---|---|
| 400 | `invalid_conditions` | generate | condition on a non-conditionable column, or an unknown value | use `conditionable_columns` and `categories` from the table schema |
| 400 | `too_many_combinations` | generate | the merged conditions expand into more than 1,000 combinations (`api.max_condition_combinations`) | list fewer values, or drop columns you do not need to control |
| 400 | `invalid_filters` | generate | unknown column, `min`/`max` on a non-numeric column, bound outside the physical range, unknown `in`/`not_in` value, filter excluding every value | see [section 3.2](#32-scenario-wards-conditions-and-filters) |
| 400 | `scenario_table_mismatch` | generate, evaluate | the scenario belongs to the other table | use the scenario's table (shown in the catalogue) |
| 400 | `generation_error` | generate | the engine rejected the request: conditions that do not overlap (scenario, conditions and wards), no calendar day matching the month/weekday conditions, ... | follow the detail; relax or align the conditions |
| 400 | `invalid_request` | generate | another invalid value reported by the engine | follow the detail |
| 400 | `invalid_records` | validate, evaluate (records) | learned columns missing, or derived columns cannot be computed (the message names the exception type only) | send every learned column with valid types (`GET /api/v1/tables/{table}/schema`) |
| 400 | `too_few_rows` | evaluate (`dataset_id`) | the stored dataset has fewer than 20 rows | evaluate a larger dataset |
| 400 | `table_mismatch` | evaluate | `table` differs from the stored dataset's table | omit `table` or correct it |
| 400 | `evaluation_failed` | evaluate | the quality computation rejected the data (a `ValueError` or `KeyError`; the message names the exception type and points to the table schema, details in the server log) | check the records' columns and values against the table schema |
| 404 | `not_found` | any | unknown path; `/ui` without its HTML file | check the URL |
| 404 | `table_not_found` | table schema, `?table=` on scenarios, models and datasets | unknown table name in the URL | `ward_daily` or `patient_stays` (an unknown table in a request body is a 422) |
| 404 | `scenario_not_found` | scenario, generate, evaluate | unknown scenario name | `GET /api/v1/scenarios` |
| 404 | `model_not_found` | generate | `model_id` is not trained for the table while other models are | `GET /api/v1/models`, or omit `model_id` |
| 404 | `dataset_not_found` | datasets, evaluate | no such dataset (deleted, never stored, or malformed id) | `GET /api/v1/datasets` |
| 405 | `method_not_allowed` | any | wrong HTTP method (the `Allow` header lists the right one) | use the documented method |
| 409 | `dataset_busy` | delete | the dataset's files are locked by another process | retry shortly |
| 422 | `validation_error` | any with a body or query | schema violation: types, ranges, unknown fields (internal attribute names such as `run_validation` or `in_` included), unknown table in the body, cross-field rules (calendar dates, evaluate sources), malformed JSON (`body (character N): JSON decode error`), a missing or non-JSON content type | read `errors` |
| 422 | `no_rows_generated` | generate | no generated row satisfied the request; nothing was stored | relax the scenario, conditions or filters, or raise `time_budget_seconds`/`max_attempts` |
| 500 | `internal_error` | any | unexpected server error, logged with its traceback | report it with the server log |
| 500 | `dataset_corrupt` | dataset metadata, download, evaluate | the stored metadata cannot be read | delete the dataset and generate it again |
| 500 | `validation_failed` | validate | unexpected failure inside the rule validation ("Validation failed (<ExceptionType>); see the server log.") | report it with the server log |
| 500 | `evaluation_failed` | evaluate | unexpected failure inside the quality computation ("Quality evaluation failed (<ExceptionType>); see the server log.") | report it with the server log |
| 503 | `model_unavailable` | generate | no trained model for the table, or its files are missing (also a named `model_id` when nothing is trained) | run `python scripts/02_train_synthesizers.py`; no restart needed |
| 503 | `engine_unavailable` | models, generate | the generation engine cannot be initialised, for example a broken installation ("The generation engine could not be initialised (<ExceptionType>); see the server log."); the next request retries | check the server log and the environment |
| 503 | `reference_data_unavailable` | evaluate | the processed reference split is missing | run `python scripts/01_prepare_data.py` |
| 503 | `validation_unavailable` (validate), `evaluation_unavailable` (evaluate) | validate, evaluate | the rule or quality module is not available in this build (defensive guard) | reinstall the project |
| 503 | `server_busy` | generate | four generations are running or queued, or the wait exceeded 60 s; `Retry-After` is set | retry after `Retry-After` seconds |

Inside `/generate`, a failing validation or quality computation does not fail the request: it becomes a
"Validation skipped: ..." or "Quality evaluation skipped: ..." warning. The application also maps any
other framework HTTP error to `error_type` `http_error`, and 413 to `payload_too_large`; BedSynth itself
raises neither.

---

## 7. Limits

All limits are read from `config/settings.yaml` at start-up, except those marked "constant" (defined in
`bedsynth/api/schemas.py`, `bedsynth/api/main.py`, `bedsynth/api/service.py` or
`bedsynth/validation.py`).

| Limit | Value | Settings key | When exceeded |
|---|---|---|---|
| Rows per sample-mode generation | 1 to 10,000 | `api.max_rows` | 422 |
| Calendar period | at most 1,100 days | `api.max_calendar_days` | 422 |
| Calendar rows (wards x days) | at most 10,000 | `api.max_rows` | 422 (8 wards x 1,100 days = 8,800, so the day limit applies first by default) |
| Calendar dates | 1900-01-01 to 2200-12-31 | constant | 422 |
| Condition combinations per request | at most 1,000 | `api.max_condition_combinations` | 400 `too_many_combinations` |
| Time budget of a generation | default 60 s, maximum 120 s | `api.time_budget_seconds`, `api.max_time_budget_seconds` | 422 above the maximum; the default applies when omitted |
| Rejection rounds | default 25, maximum 100 | `generation.max_attempts`, `api.max_attempts` | 422 above the maximum |
| Generations running or queued | 4 | `api.max_pending_generations` | 503 `server_busy` |
| Wait in the generation queue | 60 s | `api.generation_wait_seconds` | 503 `server_busy` |
| `Retry-After` on `server_busy` | 60 s (`api.time_budget_seconds`, rounded up) | `api.time_budget_seconds` | |
| Rows requested from the model per round | 200,000 | `generation.max_round_rows` | the round is scaled down (`throttled_rounds`) |
| Low-acceptance warning | below 5 % accepted | `generation.low_acceptance_warning` | warning |
| Allocation strata columns | `ward_id` x `occupancy_level`; `ward_id` x `admission_source` x `delay_reason` | `generation.allocation_strata` | |
| Records per `/validate` request | 1 to 10,000 | `api.max_rows` | 422 |
| Records per `/evaluate` request | 20 to 10,000 | `api.max_rows`; minimum constant | 422 |
| Quality subsample per side | at most 5,000 (`max_rows`, 20 to 5,000) | `api.evaluate_max_rows` | 422 for `max_rows` out of range |
| Minimum rows for a quality score | 20 synthetic rows | constant | 400 `too_few_rows` (evaluate) or a warning (generate) |
| Minimum matching reference rows | 20 | constant | the full split is used, with a warning |
| Dataset listing | `limit` 1 to 500, default 50 | constant | 422 |
| Seed | 0 to 4,294,967,295 | constant | 422 |
| Scenario name | at most 64 characters | constant | 422 |
| `model_id` | `^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$` | constant | 422 |
| Wards per request | 1 to 8 | constant (the ward catalogue) | 422 |
| Warnings per generation response | 25 plus one summary line | constant | extra warnings counted |
| `condition_allocation` entries in a report | 200 plus `condition_allocation_total` | constant | truncated |
| Positions listed in `invalid_rows` / `derived_mismatch_rows` | first 100 | constant | truncated |
| `example_rows` per rule | 5 | constant | truncated |
| Dataset persistence | on | `api.persist_datasets` | off: nothing stored, with a warning |
| Extra CORS origins | none | `api.cors_origins` | |
| Local CORS origins | localhost, 127.0.0.1 and [::1] on any port | `api.cors_origin_regex` | |

---

## 8. Scenario catalogue

Scenarios are defined in `bedsynth/scenarios.py` and served by `GET /api/v1/scenarios`. *Conditions*
fix values of categorical training columns (served by segment routing and pooled rejection,
[section 3.2](#32-scenario-wards-conditions-and-filters)); *filters* are applied by rejection after
sampling and derivation. Scenarios with filters therefore usually accept fewer rows per round.
`GET /api/v1/scenarios?include_support=true` shows how common each scenario is in the reference data.

**`ward_daily`** (16 scenarios)

| Scenario | Category | Selects |
|---|---|---|
| `baseline` | `baseline` | unconditional sample (the real mix of wards, seasons and pressures) |
| `normal_occupancy` | `occupancy` | `occupancy_level` = normal (75-90 % midnight occupancy) |
| `high_occupancy` | `occupancy` | `occupancy_level` in {high, critical} (at least 90 %) |
| `low_occupancy` | `occupancy` | `occupancy_level` = low (below 75 %) |
| `winter_demand` | `seasonal` | `month` in {12, 1, 2} |
| `summer_demand` | `seasonal` | `month` in {6, 7, 8} |
| `emergency_pressure` | `emergency` | `occupancy_level` in {high, critical}; filter `emergency_boarders` at least 1 |
| `escalation_capacity` | `utilization` | `occupancy_level` in {high, critical}; filter `escalation_beds_open` at least 1 |
| `reduced_capacity` | `utilization` | filter `beds_closed` at least 1 |
| `scheduled_procedures` | `scheduling` | `ward_id` in {GSUR, ORTH, CARD}, `day_of_week` Mon-Fri; filter `elective_scheduled` at least 2 |
| `elective_cancellations` | `scheduling` | filter `elective_cancelled` at least 1 |
| `high_turnover` | `turnover` | inpatient wards (`ward_id` in {GMED, ELDC, CARD, GSUR, ORTH, ICU}); filter `bed_turnover_rate` at least 0.30 |
| `delayed_discharge_pressure` | `delayed_discharge` | filter `delayed_discharge_rate` at least 0.15 |
| `staffing_shortage` | `staffing` | `staffing_level` = short |
| `staffing_surplus` | `staffing` | `staffing_level` = surplus |
| `weekend_operations` | `patient_flow` | `day_of_week` in {Sat, Sun} |

**`patient_stays`** (10 scenarios)

| Scenario | Category | Selects |
|---|---|---|
| `stays_baseline` | `baseline` | unconditional sample (the real admission mix) |
| `emergency_admissions` | `patient_flow` | `admission_source` = emergency |
| `elective_admissions` | `scheduling` | `admission_source` = elective |
| `rescheduled_electives` | `scheduling` | `admission_source` = elective; filter `times_rescheduled` at least 1 |
| `delayed_discharge_stays` | `delayed_discharge` | `delay_reason` any of the six reasons other than none; filter `discharge_delay_hours` at least 4 |
| `long_stay_patients` | `utilization` | filter `los_nights` at least 7 |
| `weekend_discharges` | `patient_flow` | filter `discharge_day_of_week` in {Sat, Sun} |
| `winter_admissions` | `seasonal` | `admission_month` in {12, 1, 2} |
| `slow_bed_turnaround` | `turnover` | filter `bed_turnaround_minutes` at least 180 |
| `long_bed_waits` | `emergency` | `admission_source` = emergency; filter `bed_wait_hours` at least 4 |

Category keys are API values and keep the code's spelling (`utilization`). Their descriptions
("Ward utilisation and capacity", ...) are in the `categories` object of the catalogue response.

---

## 9. Client examples

The examples assume the server runs at `http://127.0.0.1:8000`. Generation requests can take up to the
time budget (at most 120 s) plus a wait in the queue (up to 60 s), so set client timeouts of at least
240 s. The demo UI uses 240 s and `demo/demo_client.py` uses 300 s.

### 9.1 curl (bash, Git Bash, macOS, Linux)

```bash
BASE=http://127.0.0.1:8000

curl -s "$BASE/health"
curl -s "$BASE/api/v1/scenarios?table=patient_stays"

# JSON generation (pipe through jq if it is installed: | jq '.generation | {seed, acceptance_rate, shortfall}')
curl -s -X POST "$BASE/api/v1/generate" -H "Content-Type: application/json" \
     -d '{"table": "ward_daily", "num_rows": 200, "scenario": "winter_demand", "wards": ["AMU", "GMED", "ELDC"], "seed": 11}'

# Conditions and filters without a named scenario
curl -s -X POST "$BASE/api/v1/generate" -H "Content-Type: application/json" \
     -d '{"table": "ward_daily", "num_rows": 150, "conditions": {"ward_id": "ICU", "occupancy_level": ["high", "critical"]}, "filters": {"emergency_boarders": {"min": 1}}, "seed": 99}'

# CSV: save the body, print the headers
curl -s -D - -o stays.csv -X POST "$BASE/api/v1/generate" -H "Content-Type: application/json" \
     -d '{"table": "patient_stays", "num_rows": 300, "scenario": "delayed_discharge_stays", "format": "csv", "seed": 9}'

# Validate one record (missing derived columns are computed)
curl -s -X POST "$BASE/api/v1/validate" -H "Content-Type: application/json" \
     -d '{"table": "patient_stays", "records": [{"ward_id": "ORTH", "admission_source": "elective", "admission_month": 3, "admission_day_of_week": "Tue", "admission_hour": 8, "booking_lead_days": 45, "times_rescheduled": 1, "bed_wait_hours": 0.5, "los_nights": 3, "discharge_hour": 11, "separation_type": "discharge", "discharge_destination": "home", "delay_reason": "none", "discharge_delay_hours": 0.0, "bed_turnaround_minutes": 75}]}'

# Dataset lifecycle (replace the id with one returned by /generate)
ID=ds-0123456789ab
curl -s "$BASE/api/v1/datasets?limit=10"
curl -s "$BASE/api/v1/datasets/$ID"
curl -s -OJ "$BASE/api/v1/datasets/$ID/download?format=csv"      # saves <id>.csv (name from Content-Disposition)
curl -s -X POST "$BASE/api/v1/evaluate" -H "Content-Type: application/json" -d "{\"dataset_id\": \"$ID\"}"
curl -s -X DELETE "$BASE/api/v1/datasets/$ID"
```

### 9.2 Windows PowerShell 5.1

In Windows PowerShell 5.1, `curl` is an alias of `Invoke-WebRequest`, so call the real program as
`curl.exe` (part of Windows 10 and 11). PowerShell 5.1 also strips the double quotes inside an
argument passed to a native program: `'{"table": "ward_daily"}'` reaches curl as `{table: ward_daily}`,
and the API answers 422 with `"detail": "body (character 1): JSON decode error"`. Either send the body
through stdin or a file, or escape the inner quotes. The escaping is for Windows PowerShell 5.1 only:
PowerShell 7.3 and later pass inner quotes through unchanged, so there the backslashes would reach the
API and break the JSON; write the plain JSON instead.

```powershell
$base = "http://127.0.0.1:8000"

# Body in a single-quoted here-string ('@ must start its line), sent with Invoke-RestMethod - the form
# the demo UI's "Copy for PowerShell" button produces; works in Windows PowerShell 5.1 and PowerShell 7
$body = @'
{"table": "ward_daily", "num_rows": 100, "scenario": "high_occupancy", "seed": 42}
'@
Invoke-RestMethod -Method Post -Uri "$base/api/v1/generate" -ContentType 'application/json' -Body $body

# The same body through stdin with the real curl (no quoting problem either)
$body | curl.exe -s -X POST "$base/api/v1/generate" -H "Content-Type: application/json" --data-binary "@-"

# Inline body: escape every inner double quote with a backslash (Windows PowerShell 5.1 only)
curl.exe -s -X POST "$base/api/v1/generate" -H "Content-Type: application/json" `
         -d '{\"table\": \"ward_daily\", \"num_rows\": 100, \"scenario\": \"high_occupancy\", \"seed\": 42}'

# Body in a file: ASCII encoding writes no byte-order mark; quote "@file", since @name means splatting in PowerShell
'{"table": "ward_daily", "num_rows": 100, "scenario": "emergency_pressure", "format": "csv", "seed": 3}' |
    Out-File -Encoding ascii body.json
curl.exe -s -D - -o emergency.csv -X POST "$base/api/v1/generate" -H "Content-Type: application/json" --data-binary "@body.json"
```

The native cmdlets avoid quoting problems altogether. Always pass `-Depth 10` to `ConvertTo-Json`: at its
default depth of 2 it turns anything nested deeper into text. A filter such as
`filters = @{ discharge_day_of_week = @{ in = @("Sat", "Sun") } }` would be sent as `"in": "Sat Sun"`
(and a nested hashtable as `"System.Collections.Hashtable"`), which the API rejects with 422.

```powershell
$base = "http://127.0.0.1:8000"
$body = @{
    table      = "ward_daily"
    num_rows   = 200
    scenario   = "staffing_shortage"
    conditions = @{ month = @(12, 1, 2) }
    seed       = 13
    evaluate   = $true
} | ConvertTo-Json -Depth 10

try {
    $result = Invoke-RestMethod -Method Post -Uri "$base/api/v1/generate" -ContentType "application/json" -Body $body
    $result.generation | Select-Object seed, num_rows_generated, shortfall, acceptance_rate
    $result.warnings
    $result.records | Export-Csv -NoTypeInformation staffing_shortage.csv
} catch {
    $_.ErrorDetails.Message    # the JSON error body: {"detail": ..., "error_type": ...}
}

# CSV with headers: -OutFile saves the body, -PassThru keeps the response for its headers
$csvBody = @{ table = "patient_stays"; num_rows = 100; scenario = "long_bed_waits"; format = "csv"; seed = 17 } | ConvertTo-Json
$response = Invoke-WebRequest -UseBasicParsing -Method Post -Uri "$base/api/v1/generate" -ContentType "application/json" `
                              -Body $csvBody -OutFile long_bed_waits.csv -PassThru
$response.Headers["X-BedSynth-Dataset-Id"]; $response.Headers["X-BedSynth-Rows"]

# Lifecycle
$id = $response.Headers["X-BedSynth-Dataset-Id"]
Invoke-RestMethod -Uri "$base/api/v1/datasets/$id" | Select-Object dataset_id, n_rows, warnings
Invoke-WebRequest -UseBasicParsing -Uri "$base/api/v1/datasets/$id/download?format=json" -OutFile "$id.json"
Invoke-RestMethod -Method Delete -Uri "$base/api/v1/datasets/$id"
```

### 9.3 Python (httpx)

`httpx` is already in `requirements.txt`.

```python
import io
import time

import httpx
import pandas as pd

BASE_URL = "http://127.0.0.1:8000"


def post_json(client: httpx.Client, path: str, payload: dict, attempts: int = 3) -> httpx.Response:
    """POST a JSON body, waiting and retrying while the generation queue is full (503 server_busy)."""
    for _ in range(attempts):
        response = client.post(path, json=payload)
        if response.status_code == 503 and response.json().get("error_type") == "server_busy":
            time.sleep(int(response.headers.get("Retry-After", "10")))
            continue
        break
    if response.is_error:
        error = response.json()
        raise RuntimeError(f"{response.status_code} {error['error_type']}: {error['detail']}")
    return response


with httpx.Client(base_url=BASE_URL, timeout=300) as client:
    # JSON: records plus report, validation and quality
    body = post_json(client, "/api/v1/generate", {
        "table": "ward_daily", "num_rows": 200, "scenario": "high_occupancy",
        "wards": ["AMU", "GMED"], "seed": 42, "evaluate": True,
    }).json()
    report = body["generation"]
    print(body["dataset_id"], body["n_rows"], report["seed"], report["acceptance_rate"], report["shortfall"])
    print("hard rules passed:", body["validation"]["all_hard_rules_passed"], "quality:", body["quality"]["score"])
    for warning in body["warnings"]:
        print("warning:", warning)
    ward_days = pd.DataFrame(body["records"])

    # CSV: records as text/csv, identity in the X-BedSynth-* headers
    response = post_json(client, "/api/v1/generate", {
        "table": "patient_stays", "num_rows": 300, "scenario": "delayed_discharge_stays", "seed": 9, "format": "csv",
    })
    stays = pd.read_csv(io.StringIO(response.text))
    dataset_id = response.headers["X-BedSynth-Dataset-Id"]
    if int(response.headers["X-BedSynth-Rows"]) < 300:
        print("shortfall:", client.get(f"/api/v1/datasets/{dataset_id}").json()["warnings"])

    # Validate records produced elsewhere (for example by an optimisation model)
    check = post_json(client, "/api/v1/validate", {"table": "ward_daily", "records": body["records"][:10]}).json()
    print("invalid rows:", check["invalid_rows"])

    # Evaluate the stored dataset against the held-out reference split, then delete it
    evaluation = post_json(client, "/api/v1/evaluate", {"dataset_id": dataset_id, "reference_split": "holdout"}).json()
    print("quality vs holdout:", evaluation["quality"]["score"], evaluation["quality"]["reference"]["subset"])
    client.delete(f"/api/v1/datasets/{dataset_id}").raise_for_status()
```

### 9.4 More examples and evidence

* **`demo/requests.http`**: one request for every endpoint, written for the VS Code REST Client
  extension. Named requests reuse dataset ids. It includes the error cases (422, 404, 400
  `invalid_filters`, 400 `too_many_combinations`) and the `WD-D01` check.
* **`demo/demo_client.py`**: an httpx walkthrough of the whole API against a running server. It covers
  health, the catalogue, JSON and CSV generation, validation of a deliberately broken record, evaluation,
  listing, download and clean-up:
  `python demo/demo_client.py [--base-url URL] [--rows N] [--seed N] [--out-dir DIR] [--keep] [--timeout S]`.
* **`scripts/05_api_demo.py`**: exercises every endpoint in-process against the real models. It writes
  one evidence file per call, plus `summary.md`/`summary.json`, to `reports/api_demo/` and generates one
  stored dataset per scenario. `--run-tests` and `--run-slow-tests` first run the fast and the slow test
  suite and store their JUnit XML (`reports/results/pytest_junit.xml`, `pytest_junit_slow.xml`).
* **`/docs`**: interactive examples for every request body.
