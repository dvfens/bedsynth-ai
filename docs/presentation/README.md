# BedSynth AI - final presentation

`BedSynth_AI.pptx` is the team's final presentation: 20 editable 16:9 slides with speaker
notes on every slide. It is **generated** by `build_deck.js` from the project's saved results,
so it never has to be edited by hand to refresh numbers: rerun one command after a new pipeline
run and every figure, chart and table follows.

Every number on a slide and in the speaker notes is read or computed from the input files listed
below; nothing is typed into the script. Charts that show numbers are native PowerPoint charts
(right-click, then *Edit Data* in PowerPoint), and tables are native tables.

**Before presenting, open the deck once in PowerPoint and page through it.** PowerPoint lays out
chart labels itself, so check the chart labels (slides 5, 6, 11-16 and 18) and the text boxes
next to them. The delivered deck was checked slide by slide in an HTML preview of the generated
file (text overflow, off-slide text, chart data), not in PowerPoint itself: PowerPoint
automation was not available on the build machine.

## Build

Requires Node.js 18 or later (developed with Node 22) and npm. Install the pinned dependencies
once (`pptxgenjs` 4.0.1 and `jszip` 3.10.2, locked in `package-lock.json`):

```bash
cd docs/presentation
npm ci
```

Then, from the project root:

```bash
node docs/presentation/build_deck.js
```

This reads `reports/`, `models/` and `data/processed/` and writes
`docs/presentation/BedSynth_AI.pptx`. The script prints `Wrote ...: 20 slides`, and adds
`(quick-run results)` when the inputs come from a `--quick` run; in that case the result slides
also carry a *Quick-run results (...)* footnote. The delivered deck is built from the final full
run, so neither appears.

Rebuilding overwrites the deck, so wording or layout changes belong in `build_deck.js`, not in
PowerPoint.

### Options

| Option | Default | Meaning |
|---|---|---|
| `--root DIR` | two levels up from the script (the project root) | base for the three input defaults |
| `--reports-dir DIR` | `<root>/reports` | evaluation results, pytest JUnit XML, API demo evidence |
| `--models-dir DIR` | `<root>/models` | `registry.json` and the model cards |
| `--processed-dir DIR` | `<root>/data/processed` | data-quality report, schema catalogue, clean ward_daily table |
| `--out FILE` | `docs/presentation/BedSynth_AI.pptx` | output deck |
| `--ui-screenshot FILE` | `docs/presentation/assets/ui_screenshot.png` | screenshot of the demo UI (optional) |
| `-h`, `--help` | | usage |

Relative paths are resolved against the current directory. Example (build from another copy
of the results into a scratch file):

```bash
node docs/presentation/build_deck.js --reports-dir path/to/reports --models-dir path/to/models --out path/to/deck.pptx
```

## Rebuild after a new run

1. Run the pipeline steps that change the inputs: `scripts/02_train_synthesizers.py` (models),
   `scripts/03_evaluate.py` (results, registry defaults) and
   `scripts/05_api_demo.py --run-tests --run-slow-tests` (API demo evidence and both JUnit files),
   or `scripts/run_pipeline.py --run-tests --run-slow-tests` for all of them.
2. Re-capture the UI screenshot if the models changed (next section).
3. Rebuild: `node docs/presentation/build_deck.js`. Check that it prints `20 slides` without
   `(quick-run results)`.
4. Open the deck in PowerPoint and page through it (chart labels, slide 14 and 19 text boxes).

Wording that depends on the results is chosen from the data, not typed: the title of the results
slide names the best model, and its notes name the runner-up per table and say where CTGAN and
TVAE stand against the no-learning floor; the operational-patterns slide lists failing checks (or,
if none fail, the checks that pass but lie beyond real-data noise), the utility box shows gaps
below the no-learning floor (or the decision results when there are none), the limitations slide
lists the remaining measured gaps, and the testing slide counts failures in red while its notes
name any failing test.

The A/B test record `reports/experiments/ward_daily_band_groups_ab.json` is not written by the
pipeline. The notes of slide 13 quote its four-seed comparison of the band-grouped ward_daily spec
with the previous spec (composite range and mean, operational checks passed, raw hard-valid rate),
and the notes of slide 19 say which checks the default fails on the extra seeds. The build uses the
file only while it describes the current default model: its control must be the ward_daily
default, its official evaluation seed must be the seed of `model_comparison.json`, and the adopted
spec's composite on that seed must equal the default's composite (within 0.0005). Otherwise, or
when the file is missing or cannot be read, the build prints a note and the notes fall back to the
single-seed caveat. After retraining with a different ward_daily spec, update or remove the file.

## Inputs

All inputs are required unless marked optional; if a file or a field the deck needs is missing,
the build stops with `build_deck: ERROR: missing input (...)` naming the file (or the missing
field) and writes no deck.

| Input | Used for |
|---|---|
| `reports/results/model_comparison.json` | composite weights, composite and components per model, ceiling and floor references, best model, run configuration (quick or full), noise calibration |
| `reports/results/<table>__<model>.json` and `<table>__reference__*.json` (the files `model_comparison.json` lists) | operational checks (failing and beyond real-data noise), raw validity per rule, scenario coverage, privacy, ML and optimisation utility |
| `reports/results/pytest_junit.xml` | fast-suite test counts per module, failures (named in the notes), errors, skipped, run time |
| `reports/results/pytest_junit_slow.xml` (optional) | slow-suite result on the testing slide and the closing slide (`pytest -m slow`, written by `scripts/05_api_demo.py --run-slow-tests`) |
| `reports/experiments/ward_daily_band_groups_ab.json` (optional) | the four-seed A/B evidence in the notes of slides 13 and 19 (previous and adopted ward_daily spec per evaluation seed: composite, raw hard-valid rate, operational checks passed, checks failing on the extra seeds); used only while it describes the current default model, see *Rebuild after a new run* |
| `reports/api_demo/summary.json` | demo steps and outcomes, scenario runs |
| `reports/api_demo/<step>.json` (found through `summary.json`) | service information (endpoint list), scenario catalogue with reference support (families, conditions, filters, matching training rows), the high-occupancy generate request and response, calendar-mode example |
| `models/registry.json` and `models/<table>/<model>/model_card.json` | default model per table, synthesizer, segments, constraint groups (and band-grouped marginals), epochs, fit time, SDV version |
| `data/processed/data_quality_report.json` | rows, injected and detected issues (totals and the per-type charts on slide 6), repairs, hard-valid rates, cell agreement with the ground truth, split, date range |
| `data/processed/schema_catalog.json` | wards and beds, columns, hard and soft rules, constraint groups, occupancy bands |
| `data/processed/ward_daily_clean.csv` | the seasonal chart on slide 5: mean midnight occupancy per ward and calendar month (the same data as `reports/figures/week1_occupancy_by_month.png`) |
| `docs/presentation/assets/ui_screenshot.png` (optional) | the demo UI on the closing slide; without it the closing slide shows the team only |

## Slides, speakers and timing

The speaker split and the timing follow the presentation plan in `docs/TEAM_PLAN.md` (section 5),
a 20-minute slot: **about 17 minutes of slides plus about 3 minutes of live demo**, then
questions. Each content slide shows its suggested speaker in the top-right corner; every slide
repeats it, with a target time, in the first line of its speaker notes; and the deck has one
PowerPoint section per speaking part.

| Slides | Content | Speaker | Target time |
|---|---|---|---|
| 1-4 | Title, problem and goal, why synthetic data, architecture | Member A | 2 min |
| 5-7 | The two tables, Week 1 data preparation, the 26 scenarios mapped to the brief families with their reference support | Member A | 3 min |
| 8-10 | Week 2 SDV pipeline (five experiments; why two single-table models rather than HMA or PAR), how a programmable constraint works, the generation engine | Member B | 4 min |
| 11-16 | Week 3 evaluation framework, composite results with ceiling and floor, the ablation (with the four-seed A/B test of the band-grouped ward_daily spec in the notes), occupancy and scheduling patterns, scenario coverage, privacy and utility | Member C | 4 min |
| 17-18 | Week 4 FastAPI (endpoints, one request and its response metadata), testing evidence | Member D | 2 min |
| after 18 | Live demo (`docs/DEMO_SCRIPT.md` section 4, the short version; fallback in section 5) | Member D | about 3 min |
| 19-20 | Limitations and future work, closing | Member C, then all for questions | 2 min |

The per-slide target times add up to exactly 17 minutes. The notes hold about 2,400 spoken words
(about 2,600 with the speaker and time lines), which is about 16 minutes at 150 words per minute;
the difference leaves time for slide changes and for pointing at the charts. When rehearsing:

- time each part against the table above rather than slide by slide;
- the notes are prompts, not a script to read: keep each slide's message and its numbers, and
  shorten the wording if a part runs long;
- if the live demo has to be skipped, the closing notes say when to offer it instead, and
  `reports/api_demo/summary.md` is the fallback evidence;
- for a shorter slot, hide slide 3 (*Why synthetic data?*, which partly repeats slide 2) with
  *Hide Slide* rather than dropping evaluation slides.

## Refreshing the UI screenshot

`assets/ui_screenshot.png` (1600 x 1000 px) shows the demo UI with the final models after one
`ward_daily` request: scenario *High occupancy*, 200 rows, seed 42, *Evaluate quality against
reference data* ticked (dark colour scheme, which suits the dark closing slide). The deck keeps
the image's aspect ratio, so another size works, but 1600 x 1000 matches the layout.

To replace it after a new run:

1. Start the API: `python scripts/04_serve_api.py` (add `--port 8001` if port 8000 is taken)
   and wait until the health pill in `/ui` reads *Ready* (the engine loads in the background).
2. Open `http://127.0.0.1:8000/ui` (or the port you chose) in a browser window of 1600 x 1000
   px with a dark colour scheme, choose the table, scenario, rows and seed above, tick
   *Evaluate quality against reference data* and press *Generate*.
3. When the results are shown, check that no result tile is cut off, take a screenshot of the
   page area (1600 x 1000) and save it as `docs/presentation/assets/ui_screenshot.png`.
4. The request stores a dataset on the server (*Save the dataset on the server* is ticked by
   default). Delete it afterwards with the delete button in the *Saved datasets* list or
   `DELETE /api/v1/datasets/{dataset_id}`, then stop the server.
5. Rebuild the deck.

The delivered screenshot was captured this way with headless Microsoft Edge driven over the
DevTools protocol (window 1600 x 1000, `prefers-color-scheme: dark`) and the stored dataset was
deleted through the API afterwards. Headless Edge can also capture the page as it first loads
(form only, no results):

```bash
msedge --headless=new --screenshot=docs/presentation/assets/ui_screenshot.png --window-size=1600,1000 http://127.0.0.1:8000/ui
```

## Export a PDF

Open `BedSynth_AI.pptx` in PowerPoint and use **File > Export > Create PDF/XPS Document** (or
**File > Save As** and choose PDF). To include the speaker notes, open **Options** in the
export dialog and set *Publish what* to *Notes pages*.

From PowerShell on Windows, with PowerPoint installed and activated (open PowerPoint once
interactively first: COM automation fails while a first-run, licence or Office update dialog is
pending):

```powershell
$pp = New-Object -ComObject PowerPoint.Application
$deck = $pp.Presentations.Open("$PWD\docs\presentation\BedSynth_AI.pptx", -1, 0, 0)   # read-only, no window
$deck.SaveAs("$PWD\docs\presentation\BedSynth_AI.pdf", 32)                         # 32 = ppSaveAsPDF
$deck.Close(); $pp.Quit()   # leave out $pp.Quit() if PowerPoint was already open
```

If `New-Object` fails with `CO_E_SERVER_EXEC_FAILURE` (0x80080005), PowerPoint could not start
at all: open it once by hand (a pending Office update, sign-in or licence dialog blocks
automation), or restart Windows if a Click-to-Run update is still being applied.

## Design notes

- Theme *BedSynth Ward*: Cambria headings, Calibri body text, Consolas for code; deep-teal
  palette with an amber accent (theme colours, so *Design > Variants > Colors* restyles the
  shapes and text; chart colours are fixed).
- Two layouts: `BS_CONTENT` (title placeholder, footer, slide number) and `BS_TITLE_DARK`
  (title and closing slides).
- Body text is at least 14 pt; chart labels 11-13 pt; captions and footnotes 10-12 pt.
- Slides 5 and 6 draw native charts from `data/processed/` instead of embedding the Week 1
  PNG figures, whose axis labels and legends would shrink to about 4-6 pt at slide size.
- After building, the script writes the theme colours into the package and removes pptxgenjs
  artefacts PowerPoint does not need (an undeclared chart axis id, content-type entries for
  slide masters that do not exist, folder entries in the ZIP).
- The document properties name the author as *BedSynth AI team*; the package contains no file
  paths.
- *Member A-D* are placeholders, as in `docs/TEAM_PLAN.md`. To put the team's names in the deck,
  edit the `MEMBERS` map at the top of `build_deck.js` and rebuild: the names replace
  *Member A-D* on the slides, in the speaker notes and in the section names (the round badges
  keep the letters). Short names fit the speaker badge best; check the badges after the rebuild.
