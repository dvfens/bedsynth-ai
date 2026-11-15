# BedSynth AI

Synthetic hospital bed-occupancy and scheduling data on demand. BedSynth AI learns the joint
distribution of **non-clinical** hospital operations data with [SDV](https://sdv.dev) and generates
realistic, operationally valid synthetic records - including targeted scenarios such as winter
pressure, escalation, delayed discharges or staffing shortages - through a FastAPI service that
validates every record and returns it with metadata.

The full specification is [`docs/DESIGN.md`](docs/DESIGN.md); the other documents are listed under
[Documentation](#documentation).

## What is in the box

| Stage | Module / script | Output |
|---|---|---|
| Reference data | `bedsynth/data/simulator.py` - discrete-event simulation of a 194-bed hospital (2022-2024), plus dirty-data injection | `data/raw/` |
| Week 1 - preparation | `scripts/01_prepare_data.py` (`bedsynth/data/preprocessing.py`, `bedsynth/validation.py`) | `data/processed/` (clean/train/holdout CSVs, SDV metadata, schema catalogue) + `reports/week1_data_preparation.md` |
| Week 2 - SDV pipeline | `scripts/02_train_synthesizers.py` (`bedsynth/synthesis/`) | `models/` + `registry.json` + `reports/week2_synthesis_pipeline.md` |
| Week 3 - evaluation | `scripts/03_evaluate.py` (`bedsynth/evaluation/`) | `reports/results/*.json` + `reports/week3_evaluation.md`, best model per table becomes the default |
| Week 4 - API | `scripts/04_serve_api.py` (`bedsynth/api/`), `scripts/05_api_demo.py`, `demo/` | REST API + demo UI, `reports/api_demo/`, `data/synthetic/api_demo/` + `reports/week4_api_system.md` |
| Submission | `scripts/package_submission.py` | `dist/BedSynth_AI_submission.zip` (see [Packaging a submission](#packaging-a-submission)) |

Two tables are modelled (`bedsynth/schema.py` is the source of truth):

* **ward_daily** - one row per ward per day: capacity, escalation and closed beds, nurse staffing,
  census, elective bookings and cancellations, emergency admissions, transfers, discharges,
  delayed discharges, ED boarders, bed waits, turnaround, length of stay and the occupancy band.
* **patient_stays** - one row per ward stay: admission source and timing, booking lead time and
  reschedules, bed wait, length of stay, discharge timing, destination, delay reason and delay
  hours, bed turnaround.

26 scenarios (`bedsynth/scenarios.py`) cover baseline operations, normal/high/low occupancy,
winter/summer demand, emergency pressure and long bed waits, escalation and reduced capacity,
scheduled procedures, cancellations and rescheduled electives, high/slow bed turnover, delayed
discharges, staffing shortage and surplus, weekend operations and long-stay patients.

## Setup

Python 3.13 with the pinned packages of `requirements.txt` (SDV 2.2, sdmetrics 0.33, rdt 2.1,
pandas 2.3, scikit-learn 1.9, FastAPI 0.162, ...). The project expects a virtual environment in
`.venv`:

```bash
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt     # Windows
.venv/bin/python -m pip install -r requirements.txt             # Linux / macOS
```

Run every command from the project root with the virtual-environment interpreter
(`.venv\Scripts\python.exe` on Windows, `.venv/bin/python` elsewhere; shown as `python` below).

## Quick start

The submission already contains the trained models, the evaluation results, the reports and the API
demo evidence of the final full run, so the service can be started straight away:

```bash
python scripts/04_serve_api.py              # then open http://127.0.0.1:8000/ui or /docs
```

**Re-running the pipeline overwrites the submitted results.** `run_pipeline.py` and the step scripts
below rewrite `data/processed/`, `models/` (with the registry defaults), `reports/` (results, weekly
reports, API demo evidence, JUnit files) and `data/synthetic/api_demo/`; after a `--quick` run they
hold quick-mode output that no longer matches the [Results](#results), `docs/DESIGN.md` or the slide
deck. To rehearse a run, point every output folder elsewhere first. The steps inherit the variables,
and the raw extract is re-simulated because the new raw folder is empty (bash below; the PowerShell
form is in `docs/DEMO_SCRIPT.md` section 6). A server started from that shell would serve the
scratch models, so use a separate terminal:

```bash
for d in RAW PROCESSED SYNTHETIC MODELS REPORTS FIGURES RESULTS; do
  export BEDSYNTH_${d}_DIR="/tmp/bedsynth-rerun/$(echo $d | tr A-Z a-z)"; done
python scripts/run_pipeline.py --quick      # prepare -> train -> evaluate -> API demo (about 16 min)
```

`--quick` trains CTGAN/TVAE for 10 epochs (GaussianCopula models keep their full marginal search)
and evaluates on at most 2,000 rows per model, 100 rows per scenario, with tighter time caps
(measured on the development machine in the integration round, before the final full run: prepare
11 s, train 266 s, evaluate 596 s, API demo 86 s; before the improvement round added the time caps,
the quick evaluation took 868 s). Without it, training uses `settings.yaml`
(200 CTGAN/TVAE epochs) and the evaluation uses the training-split size (7,016 ward-days;
patient_stays capped at 20,000 rows), 400 rows per scenario and the sdmetrics cross-check. The final
full run on the same machine (2026-11-15, `python scripts/run_pipeline.py --run-tests --run-slow-tests`)
took 2,245.5 s (37.4 min): prepare 10.7 s, train 914.7 s (15.0 min of fits for the 10 models),
evaluate 927.8 s and the demo step 392.3 s, of which the fast and slow test suites took 165 s and
149 s (JUnit times 161.9 s and 145.8 s, in `reports/results/`) and the 58 API calls 70.7 s.
`run_pipeline.py --steps train evaluate`, `--resimulate`, `--seed N`, `--dry-run`, and `--run-tests` /
`--run-slow-tests` (passed to the demo step, see [Tests](#tests)) are available.

Step by step:

```bash
python scripts/01_prepare_data.py            # simulate (if data/raw is missing), clean, validate, split
python scripts/02_train_synthesizers.py --quick
python scripts/03_evaluate.py --quick        # writes reports/results and sets the registry defaults
python scripts/05_api_demo.py --run-tests    # fast tests + every endpoint in-process; evidence + JUnit XML
```

## Using the API

Start the service with `python scripts/04_serve_api.py` (or `uvicorn bedsynth.api.main:app --port 8000`).
The generation engine loads in a background thread at start-up (about 5-12 s). Meanwhile `/health`
answers at once with status `degraded` and `"engine": {"state": "loading"}`, and requests that need the
engine (`/api/v1/generate`, `/api/v1/models`, `/api/v1/tables`) wait for it. Without trained models
the service still starts: `/health` reports `degraded` and generation answers 503 with the command
to run. If port 8000 is already taken (uvicorn logs `[Errno 10048]` and exits), serve on another
port: `python scripts/04_serve_api.py --port 8001`.

| Endpoint | Purpose |
|---|---|
| `GET /health`, `GET /`, `GET /ui`, `GET /docs` | status (engine state, trained models, reference data), links, offline demo UI, OpenAPI docs (`/docs` loads Swagger UI from a CDN) |
| `GET /api/v1/tables`, `GET /api/v1/tables/{table}/schema` | tables, columns, operational rules |
| `GET /api/v1/scenarios`, `GET /api/v1/scenarios/{name}` | scenario catalogue; `GET /api/v1/scenarios?include_support=true` adds each scenario's reference support (matching train rows), which the single-scenario endpoint always includes |
| `GET /api/v1/models` | trained models with their Week-3 metrics (composite, components, rank, raw hard-valid rate, operational checks passed, scenario coverage), segmentation, constraint groups and the default per table |
| `POST /api/v1/generate` | generate validated data (JSON or CSV, sample or calendar mode) |
| `POST /api/v1/validate`, `POST /api/v1/evaluate` | validate records / quick quality vs reference data |
| `GET/DELETE /api/v1/datasets...` | persisted generations |

Example - 200 high-occupancy ward-days for two wards, reproducible with a seed (bash, Git Bash,
macOS, Linux):

```bash
curl -X POST http://127.0.0.1:8000/api/v1/generate -H "Content-Type: application/json" \
     -d '{"table": "ward_daily", "num_rows": 200, "scenario": "high_occupancy", "wards": ["AMU", "GMED"], "seed": 42, "evaluate": true}'
```

The same request in Windows PowerShell 5.1, where `curl` is an alias of `Invoke-WebRequest` and an
inline JSON argument loses its double quotes (put the body in a here-string; `'@` must start its line):

```powershell
$body = @'
{"table": "ward_daily", "num_rows": 200, "scenario": "high_occupancy", "wards": ["AMU", "GMED"], "seed": 42, "evaluate": true}
'@
$r = Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/api/v1/generate -ContentType application/json -Body $body
$r.generation | Select-Object seed, acceptance_rate, shortfall
# or with the real curl, sending the body through stdin:
$body | curl.exe -s -X POST http://127.0.0.1:8000/api/v1/generate -H "Content-Type: application/json" --data-binary "@-"
```

The demo UI's *Copy as curl (bash)* and *Copy for PowerShell* buttons produce the bash command and the
`Invoke-RestMethod` form for the request on screen. Calendar mode returns one row per ward per day:
`{"table": "ward_daily", "mode": "calendar", "wards": ["ICU"], "start_date": "2025-01-01", "end_date": "2025-03-31"}`.
A JSON response includes the generation report (seed, acceptance rate, rejected rows per rule,
allocation, warnings), the validation summary and - with `"evaluate": true` - a quick quality score
against the reference data. `"format": "csv"` returns only `text/csv` plus `X-BedSynth-*` headers
(dataset id, rows, seed, model, hard-rule verdict, quality score); the full report of a persisted
dataset is at `GET /api/v1/datasets/{id}`, and its stored CSV is byte-identical to the CSV body
(LF line endings).

Optional request fields `time_budget_seconds` (default 60, at most 120) and `max_attempts` (1-100)
bound a generation; a request that runs out of time returns the rows accepted so far with a
shortfall warning. Limits that keep the service responsive: at most 1,000 condition combinations per
request (400 `too_many_combinations`), at most 4 generations running or queued (503 `server_busy`
with `Retry-After`), filters are type- and range-checked (400 `invalid_filters`), a request no row can
satisfy returns 422 `no_rows_generated` and stores nothing. `/validate` recomputes derived columns
and flags provided derived values that disagree (`WD-D01` / `PS-D01`). Unexpected failures (engine
start-up, model listing, validation, quality evaluation) are reported with the exception type only,
the traceback stays in the server log, and engine messages are scrubbed of server paths. CORS admits
pages served from this machine only (set `api.cors_origins` to open it).

More examples: `demo/requests.http` (VS Code REST Client) and `python demo/demo_client.py` (an
httpx walkthrough of every endpoint against a running server).

### Documentation

* [`docs/API.md`](docs/API.md) - API reference: every endpoint, parameter, response field, error and limit.
* [`docs/DEMO_SCRIPT.md`](docs/DEMO_SCRIPT.md) - live demo runbook with timings, commands and a fallback plan.
* [`docs/QA_PREP.md`](docs/QA_PREP.md) - likely examiner questions with short answers.
* [`docs/DELIVERABLES.md`](docs/DELIVERABLES.md) - every brief item mapped to its evidence.
* [`docs/TEAM_PLAN.md`](docs/TEAM_PLAN.md) - work split, presentation plan and contribution log.
* [`docs/presentation/`](docs/presentation/) - the slide deck `BedSynth_AI.pptx`, built from the saved
  results by `node docs/presentation/build_deck.js`.

## Bring your own facility extract

The pipeline runs on a real extract with the same schema. Put `ward_daily_raw.csv` and
`patient_stays_raw.csv` in `data/raw/` (or point `BEDSYNTH_RAW_DIR` at their folder) and run
`python scripts/01_prepare_data.py`: it simulates only when these files are missing or `--resimulate`
is given. The required raw columns are `bedsynth/schema.py` `RAW_COLUMNS`:

* `ward_daily_raw.csv`: `date, ward_id, bed_capacity, escalation_beds_open, beds_closed,
  staffing_level, nurses_day_shift, nurses_night_shift, census_start, elective_scheduled,
  elective_cancelled, admissions_elective, admissions_emergency, transfers_in, transfers_out,
  discharges, discharges_before_noon, census_end, delayed_discharges, emergency_boarders,
  avg_bed_wait_hours, avg_bed_turnaround_min, avg_los_days, occupancy_level`;
* `patient_stays_raw.csv`: `stay_id, ward_id, admission_source, admission_datetime,
  discharge_datetime, booking_lead_days, times_rescheduled, bed_wait_hours, separation_type,
  discharge_destination, delay_reason, discharge_delay_hours, bed_turnaround_minutes`.

Dates may be ISO or day-first (`DD/MM/YYYY[ HH:MM]`), category spellings are mapped through
`schema.CATEGORY_ALIASES`, and the ward catalogue `WARDS` in `bedsynth/schema.py` (beds, escalation
limits, nurse ratios, admission sources) must describe the facility. Without the simulator's files,
`01_prepare_data.py` skips what depends on them: no cleaning accuracy (cell agreement, injected
issues recovered), because there is no ground truth in `data/raw/_ground_truth/`; no
detected-versus-injected comparison, because there is no injection log; and no calibration table,
because there is no `simulation_manifest.json` (the Week-1 report then says the extract was provided
externally). Then retrain, re-evaluate and restart the API, since a changed schema module is not
reloaded by a running server.

## How validity is guaranteed

1. **Constraints inside the SDV synthesizers** (`bedsynth/synthesis/constraints.py`, six groups):
   SDV built-ins (count inequalities, observed category combinations) plus BedSynth programmable
   constraints - occupancy/staffing bands recomputed from the counts, learned structural zeros (no
   electives on Sundays, no booking lead for emergencies), escalation beds that cover the census, and
   exact per-group marginals (delay hours per delay reason, elective lists per ward and weekday, ED
   boarders and the mean bed wait per ward and occupancy band, closed beds per ward and staffing band,
   day-shift nurses per ward and month, escalation beds beyond the census need per ward, length of
   stay per admission weekday, hour profiles and waits per ward and admission source) - so the default
   `gc_tuned` models produce 93.8 % (ward-days) / 96.0 % (stays) hard-valid rows before any
   rejection in the Week-3 evaluation, up from 31.5 % / 24.3 % for `gc_segmented`, the pre-Week-3
   recipe without programmable constraints (`reports/week3_evaluation.md` section 3);
2. derived columns recomputed by `schema.derive` (flow balance, occupancy rate, LOS hours ...);
3. rejection sampling against the hard operational rules of `bedsynth/validation.py`, with
   allocation strata so that rejection does not skew the occupancy / ward / delay mix (filtered
   requests take their strata from a synthetic pilot draw, never from the matching real rows).

Every returned record passes all 15 ward-day or 14 stay hard rules; soft (plausibility) rules are
reported.

## Configuration

`config/settings.yaml` holds every tunable (simulation, preprocessing, experiments, generation,
evaluation caps and weights, API limits). Synthesis experiments take `segment_by` (a column, a list
of columns or a per-table mapping) and `constraints` (`true`, `false`, one group name or a list of
groups); the ablation `gc_baseline` -> `gc_segmented` -> `gc_tuned` isolates segmentation/encodings/
marginals and the programmable constraints. `generation.time_budget_seconds` and
`generation.max_round_rows` bound one request. Directories can be redirected with
`BEDSYNTH_RAW_DIR`, `BEDSYNTH_PROCESSED_DIR`, `BEDSYNTH_SYNTHETIC_DIR`, `BEDSYNTH_MODELS_DIR`,
`BEDSYNTH_REPORTS_DIR`, `BEDSYNTH_FIGURES_DIR`, `BEDSYNTH_RESULTS_DIR`; `BEDSYNTH_SETTINGS` points
at another settings file.

## Results

Final full-mode run (2026-11-15, `run_pipeline.py --run-tests --run-slow-tests` without `--quick`;
the models, registry, results, reports, API demo evidence and JUnit files all come from this one
run): CTGAN/TVAE trained for 200 epochs; every model evaluated on 7,016 generated ward-days or
20,000 stays, 400 rows per scenario, with the sdmetrics cross-check and evaluation seed 42. The
composite weighs fidelity 0.30, raw validity 0.20, operational patterns 0.25, utility 0.15 and
privacy 0.10; the real holdout (ceiling) and a no-learning baseline (floor) are scored the same way,
and the operational checks use pass thresholds calibrated to real-data noise at the evaluated sample
size.

| table / metric | real holdout | no-learning floor | gc_tuned (default) | gc_segmented | ctgan | tvae | gc_baseline |
|---|---|---|---|---|---|---|---|
| ward_daily composite | 0.976 | 0.758 | **0.958** | 0.805 | 0.758 | 0.832 | 0.563 |
| ward_daily raw hard-valid | 100 % | 20.5 % | 93.8 % | 31.5 % | 51.5 % | 77.0 % | 3.4 % |
| ward_daily operational checks passed | 17/17 | 13/17 | 17/17 | 12/17 | 9/17 | 9/17 | 4/17 |
| ward_daily scenario coverage | 0.977 | 0.961 | 0.949 | 0.943 | 0.892 | 0.895 | 0.806 |
| patient_stays composite | 0.987 | 0.927 | **0.965** | 0.698 | 0.789 | 0.749 | 0.390 |
| patient_stays raw hard-valid | 100 % | 80.0 % | 96.0 % | 24.3 % | 83.3 % | 87.3 % | 0.6 % |
| patient_stays operational checks passed | 16/16 | 13/16 | 16/16 | 4/16 | 7/16 | 2/16 | 2/16 |
| patient_stays scenario coverage | 0.977 | 0.976 | 0.962 | 0.896 | 0.918 | 0.763 | 0.478 |

`gc_tuned` is the registry default for both tables (the best complete, comparable composite) and the
only model that passes every scored operational check. On ward_daily TVAE comes second (0.832), ahead
of `gc_segmented` (0.805), and CTGAN (0.7585) sits just above the no-learning floor (0.7576); on
patient_stays CTGAN comes second (0.789) and every model except `gc_tuned` scores below the floor
(0.927). `gc_tuned` generates all 16 ward_daily and 10 patient_stays scenarios, and the delivered
output of every model is 100 % hard-valid (patient_stays `gc_baseline` delivered only 1,978 of its
20,000 rows within its time budget). Through the API, all 26 scenarios returned validated 100-row
datasets (Week-4 report section 3) and seeded requests were reproducible (section 2). CTGAN and
TVAE ran in one fixed configuration without hyperparameter tuning (200 epochs, batch size 500, CPU,
one global model per table; the patient_stays fits used 20,000 of the 45,002 training rows), while the
GaussianCopula recipe was improved step by step, so the comparison shows which recipe worked best
under this budget, not an upper bound for the deep models.

The ward_daily constraint specs of `gc_tuned` (ED boarders and the mean bed wait per ward and
occupancy band, closed beds per ward and staffing band, day-shift nurses per ward and month) were
adopted after an A/B test against the previous specs; its evidence is kept in
`reports/experiments/ward_daily_band_groups_ab.json` (`docs/DESIGN.md` section 7.2). Across
evaluation seeds 42, 7, 11 and 2024 the adopted spec scored 0.955-0.959 (mean 0.957) against
0.941-0.942 (mean 0.942) for the previous spec and passed 16-17 of 17 operational checks against
13-14, at a raw hard-valid rate of 93.2-93.8 % against 93.8-94.4 %. The rejected variant x4 passed
all 17 checks on the three extra seeds, where the adopted x7 failed one check on two of them; x7 was
kept because it alone met the pre-set adoption bar on the official seed, and it scored a higher
utility than x4 on all four seeds. The specs were compared on the same evaluation whose utility and
privacy families use the real holdout, so the adopted spec's utility and privacy figures carry some
selection optimism. CTGAN and TVAE use the same
ward_daily constraint specs, so they were trained with the band-grouped specs too: on ward_daily
TVAE scores higher and CTGAN lower than in the 2026-11-14 run with the previous specs (DESIGN
section 7.2).

Remaining gaps, as the reports show them (`reports/week3_evaluation.md` sections 5-7; `docs/DESIGN.md`
section 13 quantifies them):

* ward_daily: in the A/B re-scores the adopted spec failed one borderline check on seed 7
  (`relationship_bed_pressure_cancellations`) and one on seed 11 (`weekend_weekday_discharge_ratio`),
  and the re-scores reused one trained model, so training-seed variation is not covered (DESIGN
  section 7.2); in the final run `seasonal_pressure_contrast` (0.063 against a threshold of 0.074)
  and `weekend_weekday_discharge_ratio` (0.131 against 0.136) pass narrowly (Week-3 section 5); seven
  filtered scenarios place rows on other wards or bands than the real subset, `emergency_pressure`
  most (strata TVD 0.506; Week-3 sections 7-8).
* patient_stays: fidelity is slightly below the no-learning floor (0.977 against 0.981; Week-3
  section 2), three checks pass but lie beyond real-data noise (length of stay by admission source,
  delay prevalence by destination, mean delay by reason; Week-3 section 8), the worst column shapes
  are the length-of-stay columns, six filtered scenarios drift in their strata mix (section 7), and
  care-facility delays are too short (DESIGN section 13).
* Optimisation testing: `gc_tuned`'s optimisation utility on ward_daily (0.955) is above the floor
  (0.921) and level with the ceiling (0.953), but its capacity-planning error (2.0 %) is above the
  floor's (1.4 %) and the decision problems are stylised (Week-3 section 6); on the three extra
  evaluation seeds of the A/B test the adopted spec's optimisation score (0.945-0.958) was not always
  above the previous spec's (0.928-0.952).

## Tests

```bash
python -m pytest -m "not slow"     # 633 tests, about 2.7 minutes
python -m pytest -m slow           # 7 tests, about 2.5 minutes: CTGAN/TVAE fits, full simulator run, real-engine API and evaluation runs
```

`python scripts/05_api_demo.py --run-tests` runs the fast suite and `--run-slow-tests` the slow suite
before the API demo and stores their JUnit XML (`reports/results/pytest_junit.xml`,
`reports/results/pytest_junit_slow.xml`), which the Week-4 report summarises. The host name attribute
pytest writes is removed, and any home-directory path, user name or host name in a failure traceback
is redacted (counted as `identifiers_redacted` under `test_runs` in `reports/api_demo/summary.json`).
`python scripts/run_pipeline.py --run-tests --run-slow-tests` passes both options to its demo step.
The stored evidence of the final run (2026-11-15) records 633 of 633 fast tests passed in 161.9 s and
7 of 7 slow tests in 145.8 s (JUnit times). Before that run, one generator test that required exactly
5 rows from a pooled draw documented to return up to 5 rows was relaxed to that contract
(`tests/test_generator.py::test_calendar_strata_keep_backoff_bands_unseen_in_the_exact_cell`). The
`integration` tests load and sample the project's own trained models, directly and through the API,
once the pipeline has run.

## Packaging a submission

```bash
python scripts/package_submission.py --dry-run     # list the files, their total size and the scan result; writes nothing
python scripts/package_submission.py               # writes dist/BedSynth_AI_submission.zip
```

The zip holds one `BedSynth_AI/` folder with `README.md`, `requirements.txt`, `pyproject.toml`,
`bedsynth/`, `scripts/`, `config/`, `tests/`, `demo/`, `docs/` (without
`docs/presentation/node_modules`), `reports/`, `data/processed/` (the SDV-ready dataset), `models/` and
`data/synthetic/api_demo/` (the API demo's datasets) - including the build outputs `.gitignore` keeps
out of a repository. `.venv`, `__pycache__`, `.pytest_cache`, `dist/` and compiled `*.pyc` files are
never included, and `data/raw/` only with `--include-raw`. Before anything is written, every included
file is scanned for the current user's home-directory path, the user name inside a pytest temporary
path and the machine's host name (pickles and other binary files, and the members of `.pptx`/`.zip`
containers, included); any hit aborts the build and lists the file (`--allow-leaks` writes the zip
anyway). A dry run of the final project found none of them.

## Deploying to Vercel

A live copy of the API and demo UI runs on Vercel at **https://bedsynth-ai.vercel.app**
(UI at [`/ui`](https://bedsynth-ai.vercel.app/ui), interactive docs at
[`/docs`](https://bedsynth-ai.vercel.app/docs)). Vercel runs the FastAPI app as one Python
serverless function that also serves the UI, so the frontend and backend deploy together.

```bash
python scripts/build_vercel_bundle.py              # writes dist/vercel (needs models/ and data/processed/)
cd dist/vercel && npx vercel deploy --prod         # first time: npx vercel link --project bedsynth-ai
```

The repository cannot be deployed from GitHub as it is: `models/` and `data/` are not committed, and
PyTorch (needed only by CTGAN/TVAE) is far larger than Vercel's function size limit. The build script
therefore copies the package, settings, the GaussianCopula models, the processed reference data and
the evaluation results into `dist/vercel`, vendors SDV without its PyTorch dependencies, and writes
the entry point (`api/index.py`) and `vercel.json`. Differences from a local run:

* only the GaussianCopula models are served (`gc_tuned` is the default of both tables anyway);
  requesting `ctgan` or `tvae` returns 404;
* public limits are tighter: at most 2,000 rows (or a 400-day calendar) per request and a 45 s
  default time budget;
* generated datasets are kept in the function's `/tmp`, so saved datasets disappear when Vercel
  recycles the instance;
* the first request after an idle period takes about 10-20 s while SDV and the model load.

## Limitations

The reference data is simulated (a real deployment retrains on a facility extract with the same
schema, see [Bring your own facility extract](#bring-your-own-facility-extract)); rows are
conditionally independent; the two tables are synthesised independently; the generator reproduces
the training frequencies of its allocation strata exactly. For testing optimisation models this
means:

* calendar days are drawn independently, so one day's `census_end` is not the next day's
  `census_start`: a multi-day bed-planning or scheduling model gets realistic days with the right
  seasonal and weekly mix, but no census continuity (no carried-over backlog, no multi-day closure or
  escalation episodes);
* stays are not linked to ward-days, so a model that needs both views of one hospital (for example a
  patient-level bed allocator checked against ward capacity) cannot take them from one generation;
* optimisation testing is defined for ward_daily only (booking quotas, admission-control policies,
  capacity planning); stay-level usefulness rests on two train-on-synthetic, test-on-real ML tasks.

Future work: SDV's `PARSynthesizer` for ward-day sequences with census continuity; a calendar-mode
continuity post-pass that carries `census_end` into the next day's `census_start` and re-derives and
re-validates the rows; a stay-level decision test (for example elective bed booking by predicted
length of stay, scored out of sample like the booking quotas). Measured remaining gaps of the default
models are listed in `docs/DESIGN.md` section 13.
