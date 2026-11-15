# BedSynth AI - live demo runbook

Runbook for the final presentation: a **10-12 minute live demo** (section 3) and a **3-minute
short version** (section 4), with a pre-demo checklist, a fallback plan and an optional pipeline
demo. Every API request below was checked against the service: in-process with
`fastapi.testclient`, and against a real uvicorn server from Git Bash (`curl`) and Windows
PowerShell 5.1 (`curl.exe`). Values under **Expect** are **illustrative**: they were captured
in-process on 2026-11-15 against the final models of that day's full run (steps 3, 5 and 8 re-run
with exactly the requests below after the run), and a live run may differ slightly in timings. The
final metrics are in the README's Results section and `reports/week3_evaluation.md`. Do not quote
numbers from memory during the demo: point to those two places.

Companion document for the questions afterwards: [`docs/QA_PREP.md`](QA_PREP.md).

## 1. Roles (team of four)

| role | during the demo | owns these Q&A sections |
|---|---|---|
| Narrator | opening, links every step to the brief, wrap-up | data preparation; ethics, privacy and limitations |
| UI driver | drives `/ui` in the browser | SDV pipeline |
| Terminal driver | runs the curl commands | FastAPI; testing and documentation |
| Timekeeper / fallback | watches the clock, keeps the evidence pack open (section 5) | quality evaluation; occupancy, scheduling and scenario coverage |

## 2. Pre-demo checklist

### 2.1 The day before

- [ ] Trained models and reference data exist: `models/registry.json`,
      `data/processed/ward_daily_train.csv`, `data/processed/patient_stays_train.csv`. If they
      are missing, run the pipeline (section 6) the day before - never on the day.
- [ ] Fallback evidence exists: `reports/api_demo/summary.md`, the per-call files
      `reports/api_demo/NN_*.json` next to it, and `data/synthetic/api_demo/` (one folder per
      generated dataset; the 26 scenario datasets are indexed in `summary.md`).
- [ ] Rehearse the full sequence once with a stopwatch; delete the rehearsal datasets afterwards
      with the **Delete** buttons of the UI's *Saved datasets* panel. Do not touch
      `data/synthetic/api_demo/` (fallback evidence).

### 2.2 T-15 minutes: environment and server

1. Open two terminals at the project root: one for the **server**, one for the **client**
   (Windows PowerShell or Git Bash).
2. Check the interpreter: `.venv\Scripts\python.exe --version` must print Python 3.13.x.
3. Start the server in the server terminal and leave it running:

   ```powershell
   .venv\Scripts\python.exe scripts\04_serve_api.py
   ```

   (Git Bash: `.venv/Scripts/python.exe scripts/04_serve_api.py`.) The banner prints the demo UI,
   API docs and health URLs. The engine preloads in a background thread (about 5-12 s); wait for
   the log line `Generation engine SyntheticDataGenerator ready in ... s`. Meanwhile `/health`
   answers at once with `"status": "degraded"` and `"engine": {"state": "loading"}`, the UI pill
   reads *Starting · loading the engine…* and switches to *Ready* by itself, and a generation
   request that arrives earlier simply waits for the engine. If port 8000 is taken, the banner
   still prints but uvicorn then logs
   `[Errno 10048] error while attempting to bind on address ('127.0.0.1', 8000)` and exits, and
   whatever else holds the port answers in the browser: restart with `--port 8001` and change
   `8000` everywhere below.
4. Close heavy applications, mute notifications, set the browser zoom to about 125 % for the
   projector.

### 2.3 Browser tabs

| tab | check |
|---|---|
| `http://127.0.0.1:8000/ui` | the status pill (top right) reads **Ready · 2/2 tables trained** in green (*Starting · loading the engine…* for the first seconds after a server start); hovering shows the version and the SDV version |
| `http://127.0.0.1:8000/docs` | Swagger UI loads from a CDN (internet needed). If it stays blank, skip it: `/ui`, the terminal and `demo/requests.http` cover everything |
| optional | the README's Results section, `reports/week3_evaluation.md`, `reports/api_demo/summary.md` (VS Code Markdown preview) |

What the UI offers (`bedsynth/api/static/index.html`, fully offline): table and scenario pickers
(scenarios grouped by the brief's categories, with an info box showing the SDV conditions as blue
tags, the rejection filters as amber tags and the scenario's share of the reference data), *Sample
rows* / *Calendar* mode with dates, rows, seed (↻ picks a random one), model (registry default or
any trained model, each shown with its Week-3 composite score), ward chips (none = all wards), and the options *Evaluate quality*, *Validate
operational rules*, *Include derived columns*, *Save the dataset on the server*. **Generate** (or
Ctrl+Enter) shows stat tiles and four tabs: **Records** (first 100 rows; blue headers are derived
columns, hover a header for its definition), **Validation** (every hard and soft rule plus a bar
chart of rows rejected per rule during generation), **Quality** (gauge, component bars, least
similar columns, full metrics JSON) and **Report** (rows allocated per condition combination,
generation report, normalised request). The expandable *Request body (POST /api/v1/generate)*
shows the exact JSON the UI sends, with *Copy as curl (bash)* and *Copy for PowerShell* buttons
that copy it as a command for Git Bash or for Windows PowerShell. The *Saved datasets* panel lists persisted datasets with
**Open**, **CSV**, **JSON**, **Evaluate** and **Delete** actions. The UI does not expose
conditions, filters or `/validate`: show those from the terminal or `/docs`.

### 2.4 Client terminal: helpers (paste once, from the project root)

Git Bash (Linux/macOS: `PY="$PWD/.venv/bin/python"`):

```bash
PY="$PWD/.venv/Scripts/python.exe"
API=http://127.0.0.1:8000/api/v1
post() { curl -s -X POST "$API/$1" -H "Content-Type: application/json" -d "$2"; }
show() { "$PY" -c "
import json, sys
r = json.load(sys.stdin)
if 'error_type' in r: sys.exit(print(r['error_type'], '-', r['detail']))
g, v, q = r['generation'], r.get('validation') or {}, r.get('quality')
print('dataset', r['dataset_id'], '|', r['n_rows'], 'rows | model', r['model']['id'], '| seed', g['seed'])
print('acceptance', g['acceptance_rate'], '| rejected by rule', {k: n for k, n in g['rejected_by_rule'].items() if n})
print('hard rules passed:', v.get('all_hard_rules_passed'), '| soft rules flagged:', (v.get('soft_rules') or {}).get('failed'))
if q: print('quality', q['score'], 'vs', q['reference']['subset'], 'reference subset,', q['reference']['rows_compared'], 'real rows')
for w in r.get('warnings', []): print('warning:', w)
"; }
err() { curl -s -w "  [HTTP %{http_code}]\n" -X POST "$API/generate" -H "Content-Type: application/json" -d "$1"; }
mkdir -p /tmp/bedsynth-demo && cd /tmp/bedsynth-demo
```

Windows PowerShell (5.1 or 7):

```powershell
$API = "http://127.0.0.1:8000/api/v1"
function Post-Api($path, $json) { $json | curl.exe -s -X POST "$API/$path" -H "Content-Type: application/json" --data-binary "@-" }
function Show-Gen($r) {
  if ($r.error_type) { return "$($r.error_type) - $($r.detail)" }
  "dataset $($r.dataset_id) | $($r.n_rows) rows | model $($r.model.id) | seed $($r.generation.seed)"
  "acceptance $($r.generation.acceptance_rate) | hard rules passed: $($r.validation.all_hard_rules_passed) | soft rules flagged: $($r.validation.soft_rules.failed)"
  if ($r.quality) { "quality $($r.quality.score) vs $($r.quality.reference.subset) reference subset, $($r.quality.reference.rows_compared) real rows" }
  $r.warnings
}
function Post-Status($json) { $json | curl.exe -s -w "  [HTTP %{http_code}]\n" -X POST "$API/generate" -H "Content-Type: application/json" --data-binary "@-" }
New-Item -ItemType Directory -Force "$env:TEMP\bedsynth-demo" | Out-Null; Set-Location "$env:TEMP\bedsynth-demo"
```

The helpers send the JSON body through stdin (`--data-binary "@-"`) because Windows PowerShell
5.1 strips the double quotes of an inline JSON argument (the API then answers 422 with
`body (character 1): JSON decode error`). Downloads land in the temporary folder, not in the repository. Alternative client: open
`demo/requests.http` in VS Code with the REST Client extension (it targets port 8000 and has
equivalent requests for most steps: generations for most scenario families, calendar mode, the
same broken ward-day for `/validate`, `/evaluate`, the dataset lifecycle and most of the error
cases).

### 2.5 T-5 minutes: warm-up

The first generation for each table loads that table's model file and the reference data. Run one
small, non-persisted request per table so the demo requests are fast:

```bash
post generate '{"table": "ward_daily", "num_rows": 20, "seed": 1, "persist": false}' | show
post generate '{"table": "patient_stays", "num_rows": 20, "seed": 1, "persist": false}' | show
```

```powershell
Show-Gen (Post-Api generate '{"table": "ward_daily", "num_rows": 20, "seed": 1, "persist": false}' | ConvertFrom-Json)
Show-Gen (Post-Api generate '{"table": "patient_stays", "num_rows": 20, "seed": 1, "persist": false}' | ConvertFrom-Json)
```

Expect `hard rules passed: True` for both; nothing is stored (`"persist": false`).

## 3. Full demo (10-12 minutes)

| time | step | grading area | where |
|---|---|---|---|
| 0:00-1:00 | 1. Opening and health | documentation & presentation; FastAPI | UI + terminal |
| 1:00-2:00 | 2. Scenario catalogue | occupancy, scheduling & scenario coverage | UI + terminal |
| 2:00-4:00 | 3. Generate high-occupancy ward-days with a quality check | SDV pipeline; quality evaluation | UI |
| 4:00-5:00 | 4. Reproducibility with a seed | SDV pipeline | terminal |
| 5:00-6:00 | 5. Calendar mode | occupancy & scheduling | UI |
| 6:00-7:00 | 6. CSV export with metadata headers | FastAPI | terminal |
| 7:00-8:15 | 7. `/validate` catches a broken record | quality / validation; FastAPI | terminal or `/docs` |
| 8:15-9:00 | 8. `/evaluate` a stored dataset | quality evaluation | UI (+ terminal) |
| 9:00-9:45 | 9. Datasets: list, download, delete | FastAPI | UI + terminal |
| 9:45-10:45 | 10. Error handling | FastAPI | terminal |
| 10:45-11:30 | 11. Wrap-up | documentation, testing & presentation | `/docs`, evidence pack |
| 11:30-12:00 | buffer | | |

### Step 1 - Opening and health (0:00-1:00)

**Do.** Show the `/ui` header pill, then in the terminal:

```bash
curl -s http://127.0.0.1:8000/health | "$PY" -m json.tool
```

```powershell
$h = curl.exe -s http://127.0.0.1:8000/health | ConvertFrom-Json
$h | Format-List status, version, sdv_version, trained_tables, reference_data
foreach ($t in 'ward_daily', 'patient_stays') { "${t}: models $($h.models.$t.trained -join ', ') | default $($h.models.$t.default)" }
```

**Say.** "BedSynth AI generates synthetic, non-clinical hospital operations data - ward-days and
patient stays - for testing bed-management and scheduling optimisation models without sharing
real facility data. The reference data is a three-year simulation of a 194-bed, eight-ward
hospital with deliberately injected data-quality problems, which we cleaned, validated and turned
into SDV metadata (Week 1). On top we trained SDV synthesizers (Week 2), evaluated them (Week 3)
and serve the best model per table through this FastAPI service (Week 4)."

**Expect.** `"status": "ok"` and `"engine": {"available": true, "loaded": true, "state": "ready",
"detail": null}`, five trained models per table (`ctgan`, `gc_baseline`, `gc_segmented`,
`gc_tuned`, `tvae`); `"default"` is the registry default chosen by the Week-3 evaluation
(`gc_tuned` for both tables); `"reference_data"` true for both tables. Within the first seconds after
a server start the status is `"degraded"` with `"state": "loading"` - wait and repeat the call.

**Fallback.** `reports/api_demo/02_health.json`.

### Step 2 - Scenario catalogue (1:00-2:00)

**Do.** Open the *Scenario* drop-down: the scenarios are grouped by the brief's categories.
Select **High occupancy** and point at the info box. Optionally show the catalogue in the
terminal:

```bash
curl -s "$API/scenarios/high_occupancy"; echo
```

```powershell
(curl.exe -s "$API/scenarios?table=ward_daily&include_support=true" | ConvertFrom-Json).scenarios | Select-Object name, category, @{n='matching';e={$_.support.matching_rows}}, @{n='share';e={$_.support.share}} | Format-Table -AutoSize
```

**Say.** "26 named scenarios - 16 for ward-days, 10 for stays - cover every family in the brief:
normal, high and low occupancy, seasonal demand, admission and discharge patterns, emergency
pressure, ward utilisation, scheduled procedures, bed turnover, delayed discharge and staffing.
A scenario has *conditions* on categorical columns (blue tags) - the engine routes them to the
per-ward SDV models and keeps only the rows of their draws that match - and *filters*, which are
applied by rejection after the derived columns are computed (amber tags). The support line shows
how much of the reference data the scenario covers."

**Expect.** `high_occupancy` = condition `occupancy_level in [high, critical]`, no filter, support
about half of the 7,016 reference ward-days; `emergency_pressure` adds the filter
`emergency_boarders >= 1`.

**Fallback.** `reports/api_demo/08_scenario_catalogue_with_reference_support.json`,
`10_scenario_high_occupancy.json`.

### Step 3 - Generate high-occupancy ward-days with a quality check (2:00-4:00)

**Do (UI).** Table *Ward-day operations (ward_daily)*, scenario **High occupancy**, *Sample rows*,
Rows **200**, Seed **42**, Model *Registry default*, no ward selected, tick **Evaluate quality
against reference data**, press **Generate**. Expand *Request body* to show the JSON the UI sent.
Then walk through the tiles and tabs:

1. Tiles: Rows, Acceptance rate (kept of sampled), Generation time, Seed (*reuse it to
   reproduce*), Model, Hard rules (*All passed*), Quality (*vs matching subset*).
2. **Records**: only `high` / `critical` occupancy levels; blue headers are derived columns
   (`census_end`, `occupancy_rate`, ...) recomputed from the learned ones.
3. **Validation**: all 15 hard rules pass; the bar chart shows the rows the model proposed that
   were rejected during generation, and by which rule.
4. **Quality**: gauge plus column shapes, column pair trends and validity; compared against the
   reference days that are themselves high or critical (*matching subset*).
5. **Report**: rows allocated per condition combination (high vs critical in the reference
   proportion), the generation report (seed, attempts, `allocation_strata`, time budget) and the
   normalised request.

Terminal equivalent (if the UI misbehaves):

```bash
post generate '{"table": "ward_daily", "num_rows": 200, "scenario": "high_occupancy", "seed": 42, "evaluate": true}' | show
```

```powershell
$r = Post-Api generate '{"table": "ward_daily", "num_rows": 200, "scenario": "high_occupancy", "seed": 42, "evaluate": true}' | ConvertFrom-Json
Show-Gen $r
```

**Say.** "The default model (`gc_tuned` - the Week-3 winner) is one Gaussian copula per ward
with five constraint groups - every group that applies to ward-days, including our four
programmable SDV constraints. The scenario condition is served from those per-ward models by
exact rejection on the occupancy band; every row then goes through derivation and rejection
against the 15 hard operational rules - so the
acceptance rate is below 100 %, and the Validation tab shows which rules the rejected rows broke
(here the capacity rules). Allocation strata keep the ward and occupancy-band mix at the
reference proportions, so rejection does not skew the output towards easy rows (any quota moved
between strata is reported as `rebalanced_rows`). The response carries the records plus their
metadata: model, seed, generation report, rule-by-rule validation and a quick quality score."

**Expect (illustrative).** 200 rows; acceptance about 90 % (286 of 317 sampled rows passed every
check over two rounds and the quotas kept 200; the others broke WD-H06 / WD-H15, census above the
available beds); *All passed* with 15 hard rules (the tile's sub-line counts the soft warnings: 2 in
our run, WD-S01 and WD-S02 on one row each); quality about 0.976 against the 3,296 matching
reference ward-days; report allocation 119 high / 81 critical (set by the reference frequencies,
not by the model), no rows rebalanced between strata; about 2 s.

**Fallback.** `reports/api_demo/13_generate_ward_daily_high_occupancy_evaluate.json`.

### Step 4 - Reproducibility with a seed (4:00-5:00)

**Do.** Run the same request twice as CSV without persisting it and compare the files, then
change the seed:

```bash
for run in 1 2; do post generate '{"table": "ward_daily", "num_rows": 200, "scenario": "high_occupancy", "seed": 42, "format": "csv", "persist": false}' > run$run.csv; done
cmp run1.csv run2.csv && echo "identical"; sha256sum run1.csv run2.csv
post generate '{"table": "ward_daily", "num_rows": 200, "scenario": "high_occupancy", "seed": 7, "format": "csv", "persist": false}' > run3.csv
cmp -s run1.csv run3.csv || echo "seed 7 gives different records"
```

```powershell
$body = '{"table": "ward_daily", "num_rows": 200, "scenario": "high_occupancy", "seed": 42, "format": "csv", "persist": false}'
$body | curl.exe -s -X POST "$API/generate" -H "Content-Type: application/json" --data-binary "@-" -o run1.csv
$body | curl.exe -s -X POST "$API/generate" -H "Content-Type: application/json" --data-binary "@-" -o run2.csv
Get-FileHash run1.csv, run2.csv | Format-Table Hash, Path
$body.Replace('"seed": 42', '"seed": 7') | curl.exe -s -X POST "$API/generate" -H "Content-Type: application/json" --data-binary "@-" -o run3.csv
(Get-FileHash run1.csv).Hash -eq (Get-FileHash run3.csv).Hash
```

(In PowerShell use `curl.exe -o`, not `>`: Windows PowerShell 5.1 re-encodes redirected output.)
UI alternative: press **Generate** again with seed 42 - same acceptance rate, same quality score
and the same first rows (only the timing and the dataset id change).

**Say.** "Every SDV call is seeded, so the same seed, model and request return identical records -
here byte-identical CSV files; a different seed gives different data. Without a seed the API
draws one and reports it in the response, so any dataset can be regenerated. This holds for
requests that finish within their time budget."

**Expect.** `identical` and two equal hashes; the last line prints
`seed 7 gives different records` (bash) or `False` (PowerShell).

**Fallback.** `reports/api_demo/18_reproducibility_run_a_seed_fixed.json` and
`19_reproducibility_run_b_same_seed.json` (check *identical records for the same seed: passed*).

### Step 5 - Calendar mode (5:00-6:00)

**Do (UI).** Scenario *No scenario (custom / unconditional)*, Mode **Calendar**, start
**2025-01-01**, end **2025-01-31** (the defaults), ward chips **AMU** and **ICU**, seed **21**,
untick *Evaluate quality*. The hint reads *One row per ward per day: 2 wards × 31 days = 62
rows* and the Rows field is disabled. **Generate**; in **Records**, show the `date` column, then
compare one day's `census_end` with the next day's `census_start`.

Terminal variant (first rows only):

```bash
post generate '{"table": "ward_daily", "mode": "calendar", "wards": ["AMU", "ICU"], "start_date": "2025-01-01", "end_date": "2025-01-31", "seed": 21, "format": "csv", "persist": false}' | cut -d, -f2,3,5,7,16,25 | head -6
```

```powershell
Post-Api generate '{"table": "ward_daily", "mode": "calendar", "wards": ["AMU", "ICU"], "start_date": "2025-01-01", "end_date": "2025-01-31", "seed": 21, "format": "csv", "persist": false}' | ConvertFrom-Csv | Select-Object -First 5 date, ward_id, month, day_of_week, census_start, census_end | Format-Table
```

**Say.** "Calendar mode gives one row per ward per day for a date range - the shape a scheduling
model consumes. Month and weekday follow the dates, so seasonal and weekly patterns land on the
right days. Limitation: rows are conditionally independent draws, so day-to-day census
continuity is not modelled - one day's closing census does not carry over to the next day's
opening census."

**Expect (illustrative).** 62 rows sorted by ward then date, all 15 hard rules passed (in our run
the soft rule WD-S04, escalation beds below 80 % occupancy, flagged 7 rows), under 1 s;
consecutive days do not chain - for example AMU closed 1 January with 26 patients and opened
2 January with 27.

Optional (scenario plus calendar): the body below keeps the nine February days and warns that
the 10 March days do not match the scenario's months and were dropped (send it with
`post generate '...' | show` or `Show-Gen (Post-Api generate '...' | ConvertFrom-Json)`, or
paste it into `/docs`).

```json
{"table": "ward_daily", "mode": "calendar", "scenario": "winter_demand", "wards": ["GMED"], "start_date": "2025-02-20", "end_date": "2025-03-10", "seed": 5}
```

**Fallback.** `reports/api_demo/15_generate_ward_daily_calendar_january_amu_icu.json`.

### Step 6 - CSV export with metadata headers (6:00-7:00)

**Do.**

```bash
curl -s -D - -o stays.csv -X POST "$API/generate" -H "Content-Type: application/json" -d '{"table": "patient_stays", "num_rows": 100, "scenario": "delayed_discharge_stays", "seed": 9, "format": "csv"}'
head -3 stays.csv
```

```powershell
'{"table": "patient_stays", "num_rows": 100, "scenario": "delayed_discharge_stays", "seed": 9, "format": "csv"}' | curl.exe -s -D - -o stays.csv -X POST "$API/generate" -H "Content-Type: application/json" --data-binary "@-"
Get-Content stays.csv -TotalCount 3
```

**Say.** "With `format: csv` the body is plain CSV for spreadsheets or optimisation tools, and the
metadata travels in `X-BedSynth-*` headers: dataset id, table, rows, model, scenario, seed,
persisted flag and the hard-rule verdict - plus the quality score when `evaluate` is set. The full
generation report stays retrievable at `GET /api/v1/datasets/{id}`."

**Expect.** `HTTP/1.1 200 OK`, `content-type: text/csv; charset=utf-8`, headers
`x-bedsynth-dataset-id`, `-table: patient_stays`, `-rows: 100`, `-model`,
`-scenario: delayed_discharge_stays`, `-seed: 9`, `-persisted: true`,
`-hard-rules-passed: true`, `x-process-time-ms`; the file has a header line and 100 stays, each
with a delay reason other than `none` and at least 4 delay hours; about 2 s.

**Fallback.** `reports/api_demo/14_generate_ward_daily_emergency_pressure_csv.json` (headers) and
its `.csv`.

### Step 7 - `/validate` catches a deliberately broken record (7:00-8:15)

**Do.** Send one valid ward-day and one broken copy: aliases (`gen med`, `monday`, `safe`) and
3 cancellations for 1 booked elective.

```bash
post validate '{"table": "ward_daily", "records": [
 {"ward_id": "GMED", "month": 1, "day_of_week": "Mon", "escalation_beds_open": 0, "beds_closed": 0, "staffing_level": "adequate", "nurses_day_shift": 5, "nurses_night_shift": 4, "census_start": 30, "elective_scheduled": 2, "elective_cancelled": 0, "admissions_emergency": 4, "transfers_in": 1, "transfers_out": 1, "discharges": 6, "discharges_before_noon": 1, "delayed_discharges": 2, "emergency_boarders": 0, "avg_bed_wait_hours": 2.5, "avg_bed_turnaround_min": 80.0, "avg_los_days": 5.8, "occupancy_level": "high"},
 {"ward_id": "gen med", "month": 1, "day_of_week": "monday", "escalation_beds_open": 0, "beds_closed": 0, "staffing_level": "safe", "nurses_day_shift": 5, "nurses_night_shift": 4, "census_start": 30, "elective_scheduled": 1, "elective_cancelled": 3, "admissions_emergency": 4, "transfers_in": 1, "transfers_out": 1, "discharges": 6, "discharges_before_noon": 1, "delayed_discharges": 2, "emergency_boarders": 0, "avg_bed_wait_hours": 2.5, "avg_bed_turnaround_min": 80.0, "avg_los_days": 5.8, "occupancy_level": "high"}
]}' | "$PY" -c "
import json, sys
r = json.load(sys.stdin)
print('rows', r['n_rows'], '| invalid rows', r['invalid_rows'], '| aliases normalised', r['normalisation']['values_normalised'])
for rule in r['validation']['rules']:
    if not rule['passed']: print(' ', rule['rule_id'], rule['name'], '-', rule['description'])
"
```

```powershell
$records = @'
{"table": "ward_daily", "records": [
 {"ward_id": "GMED", "month": 1, "day_of_week": "Mon", "escalation_beds_open": 0, "beds_closed": 0, "staffing_level": "adequate", "nurses_day_shift": 5, "nurses_night_shift": 4, "census_start": 30, "elective_scheduled": 2, "elective_cancelled": 0, "admissions_emergency": 4, "transfers_in": 1, "transfers_out": 1, "discharges": 6, "discharges_before_noon": 1, "delayed_discharges": 2, "emergency_boarders": 0, "avg_bed_wait_hours": 2.5, "avg_bed_turnaround_min": 80.0, "avg_los_days": 5.8, "occupancy_level": "high"},
 {"ward_id": "gen med", "month": 1, "day_of_week": "monday", "escalation_beds_open": 0, "beds_closed": 0, "staffing_level": "safe", "nurses_day_shift": 5, "nurses_night_shift": 4, "census_start": 30, "elective_scheduled": 1, "elective_cancelled": 3, "admissions_emergency": 4, "transfers_in": 1, "transfers_out": 1, "discharges": 6, "discharges_before_noon": 1, "delayed_discharges": 2, "emergency_boarders": 0, "avg_bed_wait_hours": 2.5, "avg_bed_turnaround_min": 80.0, "avg_los_days": 5.8, "occupancy_level": "high"}
]}
'@
$v = Post-Api validate $records | ConvertFrom-Json
$v | Format-List n_rows, n_invalid_rows, invalid_rows
$v.normalisation.values_normalised
$v.validation.rules | Where-Object { -not $_.passed } | Format-Table rule_id, name, violations
```

Zero-typing alternative: in `/docs`, open `POST /api/v1/validate`, **Try it out**, pick the example
*Ward-days: one valid, one with aliases and an impossible cancellation count*, **Execute** - it is
the same request. (`demo/requests.http` has it too.)

**Say.** "`/validate` checks anyone's records - for example data from another generator - against
the same rules. It normalises aliases, recomputes every derived column and runs the 15 hard and 6
soft ward-day rules. One impossible number cascades: three cancellations for one booking breaks
`cancellations_within_bookings`, makes the derived elective admissions negative, and moves the
recomputed midnight census out of the stated occupancy band. Only row 1 is flagged; row 0 passes."

**Expect.** `invalid rows [1]`; aliases normalised
`{ward_id: 1, day_of_week: 1, staffing_level: 1}`; failing rules **WD-H02**
`non_negative_integer_counts`, **WD-H03** `cancellations_within_bookings`, **WD-H10**
`occupancy_level_consistent`; the hard-rule group counts 16 checks (the 15 rules plus the API's
derived-value check WD-D01).

Optional extra (derived value that lies): post the valid record alone with `"census_end": 31`
added - it fails **WD-D01** `derived_values_consistent`, because the flow balance gives 30.

**Fallback.** `reports/api_demo/20_validate_records_1_corrupted.json` (50 generated records plus
1 corrupted copy; only row 50 is invalid).

### Step 8 - `/evaluate` a stored dataset (8:15-9:00)

**Do (UI).** In *Saved datasets*, click **Evaluate** on the `high_occupancy` row from step 3: the
dataset opens and the **Quality** tab shows the result. Terminal alternative:

```bash
ID=$(curl -s "$API/datasets?table=ward_daily" | "$PY" -c "import json,sys; print(next(d['dataset_id'] for d in json.load(sys.stdin)['datasets'] if d['scenario'] == 'high_occupancy'))")
post evaluate "{\"dataset_id\": \"$ID\"}" | "$PY" -c "import json,sys; r=json.load(sys.stdin); q=r['quality']; print('quality', q['score'], '|', q['reference']['subset'], 'subset of the', q['reference']['split'], 'split,', q['reference']['rows_compared'], 'real rows | hard rules passed:', r['validation']['all_hard_rules_passed'])"
```

```powershell
$ID = ((curl.exe -s "$API/datasets?table=ward_daily" | ConvertFrom-Json).datasets | Where-Object scenario -eq 'high_occupancy' | Select-Object -First 1).dataset_id
$e = Post-Api evaluate ('{"dataset_id": "' + $ID + '"}') | ConvertFrom-Json
$e.quality.score; $e.quality.reference | Format-List; $e.validation.all_hard_rules_passed
```

Add `"reference_split": "holdout"` to compare against real rows that no model has seen.

**Say.** "The quick quality score combines column shapes (1 - KS / 1 - TVD), Spearman pair trends
and hard-rule validity against the reference rows that match how the dataset was generated -
here the high-occupancy reference days. It is a fast sanity check; the rigorous Week-3 evaluation
adds the SDV QualityReport, calibrated operational checks, ML and optimisation utility, privacy,
scenario coverage, and a real-data ceiling and no-learning floor for every model."

**Expect (illustrative).** Quality about 0.976 (the same score as at generation time),
`subset: matching`, `split: train`, 3,296 real rows; with the holdout about 0.972 against the 842
matching holdout rows; hard rules passed `True`.

**Fallback.** `reports/api_demo/21_evaluate_stored_dataset.json`, `22_evaluate_client_records.json`.

### Step 9 - Datasets: list, download, delete (9:00-9:45)

**Do (UI).** Show the *Saved datasets* table (newest first: table, scenario, model, rows, hard-rule
verdict, quality, size) and click **CSV** or **JSON** on one row; then **Delete** the calendar
dataset of step 5 (confirm the dialog; the list refreshes itself). Terminal variant (`ID` /
`$ID` comes from step 8; if you used the UI there, run the first line of step 8's terminal block
first):

```bash
curl -s "$API/datasets?limit=5" | "$PY" -c "
import json, sys
r = json.load(sys.stdin)
print(r['total'], 'stored datasets, newest first:')
for d in r['datasets']: print(' ', d['dataset_id'], d['table'], d['scenario'], d['n_rows'], 'rows | seed', d['seed'], '| hard rules passed', d['all_hard_rules_passed'], '| quality', d['quality_score'])
"
curl -s -D - -o "$ID.csv" "$API/datasets/$ID/download?format=csv" | grep -i x-bedsynth
curl -s -o "$ID.json" "$API/datasets/$ID/download?format=json" && ls -l "$ID".*
```

```powershell
(curl.exe -s "$API/datasets?limit=5" | ConvertFrom-Json).datasets | Format-Table dataset_id, table, scenario, n_rows, seed, all_hard_rules_passed, quality_score
curl.exe -s -o "$ID.csv" "$API/datasets/$ID/download?format=csv"
curl.exe -s -o "$ID.json" "$API/datasets/$ID/download?format=json"
Get-ChildItem "$ID.*" | Format-Table Name, Length
```

Deleting from the terminal (use a dataset you no longer need, then show the 404):
`curl -s -X DELETE "$API/datasets/<id>"` followed by `curl -s "$API/datasets/<id>"`
(`curl.exe` in PowerShell).

**Say.** "By default every generation is persisted as CSV plus `metadata.json`, so a test
dataset can be listed, re-downloaded as CSV or JSON, re-evaluated or deleted later. Dataset ids
are validated before any file access, writes are atomic, and model cards and engine error
messages are scrubbed of server paths."

**Expect.** The list shows this session's persisted datasets - step 3's high-occupancy
ward-days, step 6's stays and, until you delete it, step 5's calendar from the UI - each with
`hard rules passed True`; the CSV download carries the same `X-BedSynth-*` headers as step 6
(plus the quality score when the dataset was generated with `evaluate`); delete returns
`{"dataset_id": ..., "deleted": true}` and the next GET answers 404 `dataset_not_found`.

**Fallback.** `reports/api_demo/23_list_datasets.json` to `28_deleted_dataset_is_gone_404.json`.

### Step 10 - Error handling (9:45-10:45)

**Do.**

```bash
err '{"table": "ward_daily", "num_rows": 50, "filters": {"occupancy_rate": {"min": 1.5}}}'
err '{"table": "ward_daily", "scenario": "icu_meltdown"}'
err '{"table": "ward_daily", "scenario": "long_bed_waits"}'
err '{"table": "ward_daily", "num_rows": 0, "wards": ["MARS"]}'
err '{"table": "ward_daily", "num_rows": 20, "wards": ["AMU"], "filters": {"elective_scheduled": {"min": 1}}}'
```

```powershell
Post-Status '{"table": "ward_daily", "num_rows": 50, "filters": {"occupancy_rate": {"min": 1.5}}}'
Post-Status '{"table": "ward_daily", "scenario": "icu_meltdown"}'
Post-Status '{"table": "ward_daily", "scenario": "long_bed_waits"}'
Post-Status '{"table": "ward_daily", "num_rows": 0, "wards": ["MARS"]}'
Post-Status '{"table": "ward_daily", "num_rows": 20, "wards": ["AMU"], "filters": {"elective_scheduled": {"min": 1}}}'
```

**Say.** "Every error has the same body - `detail` plus a stable `error_type`. An impossible
filter is rejected before any sampling (occupancy can never exceed 100 %); unknown scenarios
give 404; a scenario of the other table gives 400; schema errors give 422 with one message per
field. The last request is well-formed but no row can satisfy it - AMU never books electives - so
the engine gives up quickly with 422 `no_rows_generated` and stores nothing."

**Expect.**

| request | status | `error_type` | detail (abridged) |
|---|---|---|---|
| `occupancy_rate.min` 1.5 | 400 | `invalid_filters` | above the largest possible value (1); no row can match |
| scenario `icu_meltdown` | 404 | `scenario_not_found` | see GET /api/v1/scenarios for the catalogue |
| `long_bed_waits` on ward_daily | 400 | `scenario_table_mismatch` | belongs to table 'patient_stays' |
| `num_rows` 0, ward `MARS` | 422 | `validation_error` | num_rows >= 1; unknown ward 'MARS', known wards listed |
| AMU with electives | 422 | `no_rows_generated` | no dataset was stored; relax the scenario, conditions or filters (about half a second) |

Optional: the 2,772-combination request in the *Limits and guarantees* section of
`demo/requests.http` returns 400 `too_many_combinations` (limit 1,000).

**Fallback.** `reports/api_demo/29_invalid_request_422.json` to `32_impossible_filter_400.json`.

### Step 11 - Wrap-up (10:45-11:30)

**Do.** Show `/docs` (endpoint groups and the ten request examples of `POST /api/v1/generate`),
then `reports/api_demo/summary.md` (58 checked steps, one persisted dataset per scenario).

**Say.** "Everything you saw is documented: the README (setup, quick start, API usage, validity
guarantees, results), `docs/DESIGN.md` (the full specification), the four weekly reports, and the
OpenAPI page with examples, plus the API reference and this runbook in `docs/`. Testing evidence:
the fast and the slow pytest suites - their JUnit summaries are in the Week-4 report - and
`scripts/05_api_demo.py`, which exercises every endpoint with the real models and stores the
evidence. The model comparison, with the real-data ceiling and the no-learning floor, is in the
README's Results section and the Week-3 report. Happy to take questions."

## 4. Short version (3 minutes)

| time | action | say |
|---|---|---|
| 0:00-0:20 | `/ui`: point at the *Ready · 2/2 tables trained* pill | "SDV models for ward-days and stays, served by FastAPI; all data synthetic and non-clinical." |
| 0:20-1:30 | Step 3 in the UI (High occupancy, 200 rows, seed 42, Evaluate): tiles, **Validation**, **Quality** | "Scenario rows drawn from the per-ward SDV models; every row passes the 15 hard rules - rejected rows per rule are shown; quick quality against the matching reference days." |
| 1:30-2:15 | Step 7 in the terminal (or the `/docs` example) | "One impossible cancellation count breaks three hard rules, and only that row is flagged." |
| 2:15-2:40 | Step 10, first request only (`occupancy_rate.min` 1.5) | "Impossible requests are refused before sampling, with a uniform error body." |
| 2:40-3:00 | *Saved datasets*: click **CSV** | "Every dataset is stored with its metadata and downloadable; the same seed reproduces it." |

## 5. Fallback plan

| symptom | action |
|---|---|
| `04_serve_api.py` fails: port in use (`[Errno 10048] ... bind` after the banner) | `--port 8001`, then use `http://127.0.0.1:8001` in the browser and the `API` variable |
| pill shows *Starting · loading the engine…* | normal for the first 5-12 s after a server start: the UI polls `/health` and switches to *Ready* by itself; generation requests sent meanwhile wait for the engine |
| pill shows *API unreachable · retrying* | the server is not listening (yet), or on another port: the UI retries by itself (after 2, 5, 10, 20, then every 30 s); check the server terminal for the `Generation engine ... ready` line or a bind error |
| pill shows *Engine unavailable* | the engine failed to initialise; `/health` names the exception type and the server log has the traceback: switch to the evidence pack below |
| pill shows *Degraded* or generation answers 503 `model_unavailable` | models are missing or unreadable: switch to the evidence pack below (do not train on stage) |
| `/docs` or `/redoc` blank | no internet for the CDN assets: carry on with `/ui`, the terminal or `demo/requests.http` |
| PowerShell answers 422 `body (character 1): JSON decode error` | an inline JSON argument lost its quotes: use the `Post-Api` helper (stdin), the UI's *Copy for PowerShell* command, or Git Bash |
| `curl` in PowerShell prints a red `Invoke-WebRequest` error, or stops at a `Uri:` prompt (`-s` binds to `-SessionVariable`) | press Ctrl+C if prompted and type `curl.exe` - in Windows PowerShell 5.1 `curl` is an alias of `Invoke-WebRequest` |
| a generation is slow or answers 503 `server_busy` | wait for the `Retry-After` seconds; at most 4 generations run or queue at a time. Lower `num_rows` if needed |
| a shortfall warning appears | the request ran out of time (default budget 60 s) or of rejection rounds; follow the warning's advice: raise `time_budget_seconds` (at most 120) or `max_attempts` (at most 100), request fewer rows, or relax the filters |
| the UI breaks | continue with the terminal; every UI step has a terminal command |
| nothing works live | walk through the evidence pack below |

**Evidence pack** (pre-generated by `scripts/05_api_demo.py` against the trained models; every file
holds the request, status, timing, headers, the response body with records truncated, and the
checks applied):

| demo step | evidence in `reports/api_demo/` |
|---|---|
| overview | `summary.md` (step table and one dataset per scenario), `summary.json` |
| 1 health | `02_health.json` |
| 2 catalogue | `08_scenario_catalogue_with_reference_support.json`, `10_scenario_high_occupancy.json`, `11_trained_models.json` |
| 3 generate + quality | `12_generate_ward_daily_baseline_json_evaluate.json`, `13_generate_ward_daily_high_occupancy_evaluate.json`, `16_generate_patient_stays_delayed_discharge_stays_evaluate.json` |
| 4 reproducibility | `18_reproducibility_run_a_seed_fixed.json`, `19_reproducibility_run_b_same_seed.json` |
| 5 calendar | `15_generate_ward_daily_calendar_january_amu_icu.json` |
| 6 CSV export | `14_generate_ward_daily_emergency_pressure_csv.json` and `.csv` |
| 7 validate | `20_validate_records_1_corrupted.json` |
| 8 evaluate | `21_evaluate_stored_dataset.json`, `22_evaluate_client_records.json` |
| 9 datasets | `23_list_datasets.json` to `28_deleted_dataset_is_gone_404.json` |
| 10 errors | `29_invalid_request_422.json` to `32_impossible_filter_400.json` |
| scenario coverage | `33_scenario_baseline.json` to `58_scenario_long_bed_waits.json` |

The generated datasets themselves are in `data/synthetic/api_demo/<dataset id>/` (`data.csv` and
`metadata.json`); `summary.md` maps every scenario to its dataset id. The ids change each time
`05_api_demo.py` runs (it resets that folder), so read them from `summary.md` on the day. Testing
evidence: `reports/week4_api_system.md` section 5 (from `reports/results/pytest_junit.xml` and
`reports/results/pytest_junit_slow.xml`).

If the server cannot start but Python works, the whole API can still be exercised in-process
without touching the evidence pack:

```powershell
.venv\Scripts\python.exe scripts\05_api_demo.py --scenario-rows 0 --no-report --out-dir "$env:TEMP\bedsynth-live\api_demo" --store-dir "$env:TEMP\bedsynth-live\store"
```

It prints one line per step with its outcome and writes a fresh `summary.md` to the `--out-dir`.

## 6. Optional pipeline demo (for "how are the models trained?")

The full pipeline is one command - `scripts/run_pipeline.py` runs prepare -> train -> evaluate ->
API demo, each step in a fresh interpreter, and stops at the first failure:

```powershell
.venv\Scripts\python.exe scripts\run_pipeline.py --quick --dry-run    # prints the four steps (script names and options), runs nothing
.venv\Scripts\python.exe scripts\run_pipeline.py --quick              # about 16 min on the development machine (integration round)
```

The dry run prints script names only (`python.exe 01_prepare_data.py`), not paste-ready paths: the
step scripts are in `scripts/`. README timings for `--quick`, measured on the development machine in
the integration round before the final full run: prepare 11 s, train 266 s, evaluate 596 s, API demo
86 s. `--quick` trains CTGAN/TVAE for 10 epochs (the GaussianCopula models keep
their full marginal search) and evaluates at most 2,000 rows per model; without it, training uses
the 200 epochs of `config/settings.yaml` and the full evaluation - the final full run (2026-11-15)
took 914.7 s to train, 927.8 s to evaluate and 392.3 s for the API demo with both test suites
(37.4 min in total). Too long to run live - show the dry run
and, if there is time, train two models live into a scratch folder (on the development machine
the gc_baseline fit takes seconds and the gc_tuned fit about half a minute for ward_daily; allow
about a minute in total):

```powershell
.venv\Scripts\python.exe scripts\02_train_synthesizers.py --quick --tables ward_daily --experiments gc_baseline gc_tuned --models-dir "$env:TEMP\bedsynth-live\models" --no-report
```

It prints the model-card table: segments, constraint groups, chosen marginals, raw hard-valid
percentage and fit time per model - the ablation in miniature (README Results and the Week-2/3
reports have the full comparison).

**Warning.** Run without redirection, `run_pipeline.py` (quick or not) rewrites `data/processed/`
and overwrites `models/`, the registry and its defaults, `reports/` and `data/synthetic/api_demo/`,
that is, the final results you are presenting. To rehearse the whole pipeline safely, point every output folder elsewhere
first (the steps inherit the variables; data is re-simulated because the scratch raw folder is
empty). Use a separate terminal and close it afterwards - a server started from a terminal that
still has these variables would serve the scratch models:

```powershell
$out = "$env:TEMP\bedsynth-rerun"
foreach ($name in 'RAW', 'PROCESSED', 'SYNTHETIC', 'MODELS', 'REPORTS', 'FIGURES', 'RESULTS') {
    Set-Item "Env:BEDSYNTH_$($name)_DIR" "$out\$($name.ToLower())"
}
.venv\Scripts\python.exe scripts\run_pipeline.py --quick
```

## 7. Windows gotchas (quick reference)

- Windows PowerShell 5.1: `curl` is `Invoke-WebRequest`; always type `curl.exe`.
- Inline JSON in PowerShell 5.1 loses its double quotes: send bodies through stdin
  (`Post-Api`), or use Git Bash. The UI's *Copy as curl (bash)* button produces Git Bash syntax;
  *Copy for PowerShell* produces a here-string body plus `Invoke-RestMethod`, which works in
  PowerShell 5.1 and 7.
- Save responses with `curl.exe -o file`, not `>` (PowerShell 5.1 re-encodes redirected output).
- In Git Bash, a bare `python` may open the Microsoft Store alias: use the `$PY` variable
  (`.venv/Scripts/python.exe`).
- The server terminal must stay open; stop the server with Ctrl+C after the presentation.
