# BedSynth AI - examiner question bank

Forty-one likely questions for the Q&A after the demo, grouped by the brief's grading areas. Each
answer is short enough to say in under a minute and names the file to open if the examiner wants
to see it (**Open**). Section owners are listed in [`docs/DEMO_SCRIPT.md`](DEMO_SCRIPT.md),
section 1.

**Numbers.** Do not quote composite scores, raw-validity percentages, operational pass counts or
coverage scores from memory: the final values come from the full (non-quick) run and are in the
README's **Results** section and `reports/week3_evaluation.md` (summary at the top, model comparison
in section 2, ablation, A/B summary and improvement log in section 3, validity in section 4, occupancy and
scheduling checks in section 5, optimisation testing in section 6, quality gaps in section 7); the
A/B test of the ward-day specs and its seed re-scores are in `docs/DESIGN.md` section 7.2 and
`reports/experiments/ward_daily_band_groups_ab.json`. Re-read those sections before the
presentation.

| area (weight) | questions |
|---|---|
| A. Data preparation (15 %) | A1-A5 |
| B. SDV pipeline (25 %) | B1-B11 |
| C. Quality evaluation (20 %) | C1-C8 |
| D. Occupancy, scheduling and scenario coverage (15 %) | D1-D4 |
| E. FastAPI (15 %) | E1-E5 |
| F. Testing and documentation (10 %) | F1-F3 |
| G. Ethics, privacy and limitations | G1-G5 |

## A. Data preparation (15 %)

### A1. Why did you use simulated reference data instead of a real hospital extract?

Real operational data cannot be shared across facilities for privacy and competitive reasons -
that is the problem the brief describes - so no real extract was available to us. A seeded
discrete-event simulation of a 194-bed, eight-ward hospital (2022-2024) produces an
operationally coherent reference extract: seasonal emergency arrivals, elective bookings,
cancellations and rebookings, bed allocation that never exceeds the available beds, lengths of
stay, discharge delays, bed turnaround, staffing states, closures and escalation beds, checked
against 27 realism targets. Because the simulator also keeps the clean ground truth, we could
inject realistic data-quality problems and measure how well the cleaning recovers them, which a
real extract would not allow. Every downstream metric is relative to this reference - a stated
limitation.

*Open:* `bedsynth/data/simulator.py`, `docs/DESIGN.md` section 6.1, `reports/week1_data_preparation.md` section 2.

### A2. How would you switch to a real facility extract?

Place `ward_daily_raw.csv` and `patient_stays_raw.csv` with the raw columns of
`schema.RAW_COLUMNS` in `data/raw/` (or point `BEDSYNTH_RAW_DIR` at them) and run
`scripts/01_prepare_data.py`: it simulates only when the raw files are missing or `--resimulate`
is given, so it cleans, validates, splits and writes the SDV metadata for the real extract instead.
Adapt the ward catalogue `WARDS` in `bedsynth/schema.py` (beds, escalation limits, nurse ratios,
admission sources) and, if the facility uses other codes, the category aliases - the rules and
scenarios read them. Then retrain (`02_train_synthesizers.py`), re-evaluate (`03_evaluate.py`) and
restart the API, because the schema module changed (a plain retrain is picked up by a running API
without a restart). The cleaning-accuracy figures are then unavailable (they need the simulator's
ground truth) and the privacy checks become substantive instead of methodological.

*Open:* `bedsynth/schema.py` (`RAW_COLUMNS`, `WARDS`), `scripts/01_prepare_data.py`.

### A3. Which data-quality problems did you handle, and how do you know the cleaning works?

The injector adds category variants and aliases, a second date format, missing values, exact and
key duplicates (some re-entered with an alias), impossible values such as a negative census or
999 discharges, and swapped admission/discharge timestamps, and it logs every injected cell. The
pipeline standardises categories, parses both date formats, removes duplicates, nulls impossible
values, repairs from the most principled source first (census continuity and the flow identity
for ward-days, then group-wise imputation: ward x weekday medians for ward-days, seeded hot-deck
donors from comparable stays for waits, turnaround and delay hours), recomputes the derived
columns and drops rows that are still hard-invalid. Because the ground truth is kept, the Week-1 report
measures per-column agreement and the share of injected issues that are no longer defective
versus exactly recovered.

*Open:* `bedsynth/data/quality_issues.py`, `bedsynth/data/preprocessing.py`, `reports/week1_data_preparation.md` sections 3-5.

### A4. What are the two tables, and which columns does SDV actually learn?

`ward_daily` has one row per ward per day (22 learned columns: ward, month and weekday,
escalation and closed beds, nurse staffing, opening census, admissions, transfers, discharges,
electives and cancellations, delayed discharges, boarders, waits, turnaround, length of stay,
occupancy band); `patient_stays` has one row per ward stay (15 learned columns: ward, admission
source, month, weekday and hour, booking lead, reschedules, bed wait, nights, discharge hour,
separation type, destination, delay reason and hours, turnaround). Every column has a role - key, time, train
(learned) or derived - and the 12 and 8 derived columns (`census_end`, `occupancy_rate`,
`los_hours`, ...) are recomputed by `schema.derive`, so identities such as the flow balance hold
exactly. Timestamps are not learned (SDV cannot learn seasonality from them): calendar context
is explicit as month and weekday, and calendar mode attaches dates. Both tables are strictly
non-clinical: no diagnoses, procedures, ages or personal identifiers (the only keys are
pipeline-assigned record numbers).

*Open:* `bedsynth/schema.py`, `docs/DESIGN.md` section 5.

### A5. What makes the dataset "SDV-ready"?

The Week-1 step writes clean, train and holdout CSVs plus SDV metadata: `metadata.json`
describes both tables as stored (keys, datetime formats, train and derived columns) and
`metadata_<table>.json` is the single-table metadata of the training columns the synthesizers
fit. `schema_catalog.json` adds the schema, the constraint catalogue and the rule catalogue, and
`data_quality_report.json` the cleaning results. The split is 80/20, stratified by ward; the
holdout is never used for training and serves as the test set for ML utility, the privacy
baseline and the real-data ceiling.

*Open:* `bedsynth/synthesis/metadata.py`, `data/processed/`, `reports/week1_data_preparation.md` section 9.

## B. SDV pipeline (25 %)

### B1. Which synthesizers did you train, and why is GaussianCopula the default rather than CTGAN or TVAE?

Five experiments per table, all with the same constraint machinery and the same evaluation: the
GaussianCopula ablation `gc_baseline` -> `gc_segmented` -> `gc_tuned`, plus CTGAN and TVAE as
global models with every constraint group (200 epochs). The default is not hand-picked:
`03_evaluate.py` makes the best complete, comparable composite per table the registry default -
`gc_tuned` for both tables in the final run, where it is also the only model that passes every scored
operational check (README Results; Week-3 sections 2 and 5). Structurally it suits this data:
per-segment copulas fit in about half a minute on a CPU and sample fast, GaussianCopula-only features
(demand-ordered encodings, per-column tuned marginals) capture the seasonal and weekly effects, and
the programmable constraints make most raw samples valid. The deep models' weak spots are the
capacity rules on `ward_daily` (a global model proposes census values outside a ward's range) and
the elective booking-lead rules on `patient_stays`, so their output needs more rejection rounds
(compare the raw hard-valid row of README Results), and they lag on the operational patterns and ML
utility (Week-3 sections 2, 5 and 8). With 200 epochs they keep the strata mix with little
rebalancing (Week-3 section 4). In the final run TVAE is the runner-up on `ward_daily` and CTGAN on
`patient_stays`, but CTGAN only just beats the no-learning floor on `ward_daily` and both deep models
score below the floor on `patient_stays`. Caveat: CTGAN and TVAE ran in one fixed configuration
(200 epochs, batch size 500, CPU, no hyperparameter tuning; the `patient_stays` fits used a
20,000-row subsample), while the GaussianCopula recipe was improved step by step, so the comparison
shows which recipe worked best under our budget, not the best the deep models could do.

*Open:* `config/settings.yaml` (`synthesis.experiments`), `scripts/03_evaluate.py`, `reports/week3_evaluation.md` section 2.

### B2. What does the SDV metadata do in your pipeline?

`bedsynth/synthesis/metadata.py` builds SDV 2.2's `Metadata` from the schema for every fit: it
declares each training column's sdtype - numerical with its decimal places, categorical, ordinal
for the ordered occupancy and staffing bands - so SDV chooses suitable transformers (an
order-preserving encoder for the bands) and validates the data. Segment models get metadata
without their segment and constant columns. A separate evaluation metadata (train plus derived
columns, minus one-column relabellings such as `ward_type` or `season`) feeds the SDV
QualityReport, so a relabelled copy of a learned column does not count it twice and inflate
fidelity. The two-table metadata declares no relationship because the tables are synthesised
independently: SDV's multi-table synthesizer needs a single-column key, while a stay spans several
ward-days and the natural link is ward x date (`docs/DESIGN.md` section 1.1).

*Open:* `bedsynth/synthesis/metadata.py`, `docs/DESIGN.md` sections 7.1 and 9.

### B3. Which constraints do you use - built-in and programmable?

Six groups, each switchable per experiment so its effect is measurable: two SDV built-ins -
`Inequality` (ward-days: cancellations within bookings, noon discharges within discharges,
delayed patients within the census) and `FixedCombinations` (stays: only observed
separation/destination/delay-reason and ward/source combinations) - and four BedSynth subclasses
of SDV's `SingleTableProgrammableConstraint`. `regime_labels` (ward-days) recomputes the occupancy
and staffing bands from the counts; `structural_zeros` (both tables) learns category values whose
counts are always zero (no Sunday electives, no booking lead for emergencies) and forces them;
`capacity` (ward-days) re-parameterises escalation beds as the cover the census needs;
`conditional_marginals` (both tables) maps a numeric column to a normal score of its within-group
empirical distribution and back, reproducing each group's marginal exactly (delay hours per
reason, elective lists per ward and weekday, ED boarders and the mean bed wait per ward and
occupancy band, closed beds per ward and staffing band - both bands recomputed from the counts -
day-shift nurses per ward and month, length of stay per admission weekday, hours and waits per
ward and source). The band-grouped ward-day specs were adopted after an A/B test that fixed the
three checks the previous spec failed (`docs/DESIGN.md` section 7.2; see C8). Rules whose
by-construction fix would distort the learned
distributions - delays longer than the stay, same-day stays leaving before they arrive, the 7-day
rebooking gap - stay with rejection sampling.

*Open:* `bedsynth/synthesis/constraints.py` (module docstring, `CONSTRAINT_SPECS`), `reports/week2_synthesis_pipeline.md` section 3.

### B4. Explain the transform / reverse-transform ordering subtlety.

SDV 2.2 applies constraint transforms in `add_constraints` order and reverse-transforms in the
reverse order, after its data processor has already rounded and clipped the numeric columns.
`Inequality` replaces its high column by a `low#high` difference column and `FixedCombinations`
merges its columns into one, so our programmable constraints - which read or overwrite those
columns - are listed first: their `reverse_transform` then runs last, on fully restored columns,
in the order conditional marginals -> structural zeros -> escalation cover -> band-grouped ED
marginals -> band labels (ED boarders and the occupancy label must both see the band of the final
escalation beds; likewise closed beds are restored after the day-shift nurses, from the staffing
band of the restored nurse counts). The second trap: after each reverse
transform SDV silently drops rows whose `is_valid` is false, which would hide invalid samples, so
our constraints return true for rows they deliberately leave to rejection - so the raw-validity
numbers are not inflated.

*Open:* `bedsynth/synthesis/constraints.py` ("Ordering inside SDV"), `docs/DESIGN.md` sections 4 and 7.2.

### B5. Why segment the models, and how are requests routed to segments?

`gc_tuned` fits one GaussianCopula per ward for `ward_daily` (8 segments) and per ward and
admission source for `patient_stays` (20 segments), each on its own rows without the segment
columns. Wards differ structurally (12 to 32 beds, different admission sources) and a single
copula with linear latent dependence would average them; inside a segment, constant columns are
dropped and re-inserted exactly (AMU never books electives), and SDV's clipping to the training
range gives segment-specific supports for free (elective admission hours, for instance).
Conditions on the segment columns select segments; otherwise rows are apportioned to the
matching segments by their training rows that meet the remaining conditions (largest remainder).

*Open:* `bedsynth/synthesis/models.py` (module docstring, `BedSynthSynthesizer`), `docs/DESIGN.md` section 7.3.

### B6. What did each step of the ablation add?

`gc_baseline` is one global copula with default encodings, beta marginals and no constraints
(the Week-2 baseline). `gc_segmented` adds per-ward segments, the SDV built-in constraints,
ordinal metadata, demand-ordered encodings (`OrderedUniformEncoder` orders months by mean
occupancy and weekdays by mean discharges on `ward_daily`, and months and the merged
separation/destination/delay-reason column by mean length of stay on `patient_stays`, so a linear
copula can capture non-monotonic seasonal and weekly effects; the ward ordering by bed capacity
applies only to global models, and no trained model uses it) and per-column marginals chosen by KS among
truncnorm, beta, gamma and norm, with a post-fit check that refits degenerate columns.
`gc_tuned` adds the four programmable constraint groups and ward x source segments for stays. A
group only takes effect where its columns are modelled, so the model cards list the applied
groups: five on `ward_daily` (`FixedCombinations` is a stays group) and, on `patient_stays`,
`FixedCombinations` plus the conditional marginals (the ward x source segments already make the
structural zeros constant). The Week-3 report tabulates the measured effect of each step on raw
validity, fidelity, operational patterns and utility.

*Open:* `config/settings.yaml` (`synthesis.experiments`), `reports/week3_evaluation.md` section 3.

### B7. How do scenarios work - is it really conditional sampling?

A scenario has conditions on categorical training columns and filters on any training or derived
column; the request's conditions and wards are intersected with them. Condition lists expand into
value combinations whose rows are allocated by joint frequency in the reference train split. Each
combination is served conditionally inside the segment models: conditions on segment columns pick
the segment, the rest are served by exact pooled rejection from one unconditional draw per
segment (combinations never seen together are sized by an independence estimate and pooled too),
and GaussianCopula's native `sample_from_conditions` (at most 8 calls per segment and sampling
round) only for the rarest combinations, which a pooled draw cannot serve (more than 50,000 rows
needed, or no training support) - native conditioning maps a categorical value to a single
latent point, which narrows the conditional distribution, and every SDV call has a fixed cost of
a few tenths of a second. Filters such as `emergency_boarders >= 1` are applied by rejection
after derivation.

*Open:* `bedsynth/scenarios.py`, `bedsynth/synthesis/models.py` ("Conditional sampling strategy"), `bedsynth/synthesis/generator.py`.

### B8. How do you guarantee 100 % hard validity, and why does rejection sampling not skew the mix?

Three layers: constraints inside SDV make most raw rows valid, `schema.derive` recomputes the
derived columns, and the generator rejects every row that breaks any of the 15 ward-day or 14
stay hard rules (or the request's conditions and filters); the output is then validated again and
reported rule by rule. Rejection alone would be biased because strata fail at different rates -
with SDV's built-in constraints only, critical-occupancy days passed the hard rules far less often
than low-occupancy days - so each condition cell is split into allocation strata (ward x
occupancy band; ward x source x delay reason) with quotas from the reference frequencies, and
rejection only shapes rows within a stratum. Filtered requests take their strata weights from a
seeded synthetic pilot draw, never from the real rows that match the filter. A stratum still short
after four rounds hands its quota to siblings; the response reports `rebalanced_rows` and warns
when more than 10 % of the rows were rebalanced.

*Open:* `bedsynth/synthesis/generator.py` (module docstring), `bedsynth/validation.py`, `docs/DESIGN.md` section 8.3.

### B9. Is generation reproducible?

Yes, within a stated limit. Every SDV call is preceded by `reset_sampling()` and
`_set_random_state(seed)` with child seeds derived deterministically (numpy seeding has no effect
in SDV 2.x), and the API serialises generations because copulas and CTGAN swap the process-wide
random state. The same seed, model and request give identical records - our checks produced
byte-identical CSV files across requests and server restarts - and without a seed the API draws
one and reports it. Results of a request stopped by its time budget (default 60 s) or throttled
to fit it depend on machine speed, which the report flags (`stopped_reason`, `throttled_rounds`).

*Open:* `bedsynth/synthesis/models.py`, `bedsynth/synthesis/generator.py`, `bedsynth/api/service.py` (`BedSynthService`).

### B10. What are the limits of calendar mode?

Calendar mode (`ward_daily` only) builds a wards x dates grid, groups it by ward, month and
weekday, allocates each group's days to condition combinations and strata, and returns one row per
ward per day with a `date` column. Rows are conditionally independent draws: day-to-day census
continuity is not modelled, so one day's `census_end` is generally not the next day's
`census_start`, and multi-day episodes (closures, escalation runs) are not reproduced. Scenario
conditions on ward, month or weekday intersect the grid - non-matching days are dropped with a
warning - and a request is limited to 1,100 days and 10,000 rows. Use it for calendar-shaped test
inputs with the right seasonal and weekly mix, not for time-series behaviour: a multi-day
bed-planning model gets no census continuity. The fixes would be SDV's sequential `PARSynthesizer`,
at the cost of rebuilding the per-row constraint and scenario machinery, or a post-pass that carries
each day's closing census into the next day and re-validates (`docs/DESIGN.md` sections 1.1 and 13).

*Open:* `bedsynth/synthesis/generator.py` (calendar planning), `docs/DESIGN.md` sections 8.4 and 13.

### B11. How are trained models stored, selected and served?

Each experiment is saved as `models/<table>/<experiment>/synthesizer.pkl` with a
`model_card.json` (segments, applied constraints with their learned state, chosen marginals, raw
hard-valid rate, fit time) and registered in `models/registry.json`. `03_evaluate.py` sets each
table's default to the best complete, comparable and current composite and records the reason.
The registry re-reads `registry.json` on every call and writes it atomically, so a running API
picks up retrained models and new defaults without a restart; experiment ids must match a safe
pattern, so a request cannot point outside the models folder.

*Open:* `bedsynth/synthesis/registry.py`, `bedsynth/synthesis/training.py`, `GET /api/v1/models`.

## C. Quality evaluation (20 %)

### C1. How do you evaluate the synthetic data?

Every trained model generates validated data through the same engine as the API and is scored on
statistical fidelity (SDV QualityReport plus KS, TV distance and correlation differences), raw
validity (share of unfiltered samples passing every hard rule), operational patterns (calibrated
occupancy and scheduling checks), utility (ML train-on-synthetic-test-on-real and optimisation
decisions), privacy (distance to closest record and exact matches) and scenario coverage. The
families are computed on the delivered output (system view); fidelity and operational patterns
are also computed on raw model samples (model view), so the model and the generation system can
be told apart. Two reference pseudo-models - the real holdout and a no-learning baseline - are
scored the same way.

*Open:* `bedsynth/evaluation/` (`scoring.evaluate_model`), `scripts/03_evaluate.py`, `reports/week3_evaluation.md`.

### C2. How is the composite built, and how do you pick the default model?

The composite is a weighted sum of fidelity 0.30, raw validity 0.20, operational patterns 0.25,
utility 0.15 (mean of ML and optimisation utility) and privacy 0.10, set in
`settings.evaluation.composite_weights`; scenario coverage is reported next to it but not
weighted. A family that fails scores 0 and marks the composite incomplete; a family undefined by
design is left out (on `patient_stays`, which has no optimisation problem, utility is the ML
utility alone), and the weights are renormalised only if a whole component is undefined. Only
complete, comparable (same run configuration) and current entries are eligible, and the best per
table becomes the registry default. The weights favour realism because the use case is testing
optimisation models; every family is shown separately, so other weightings can be checked.

*Open:* `config/settings.yaml` (`evaluation`), `bedsynth/evaluation/scoring.py` (`composite_score`), `scripts/03_evaluate.py`.

### C3. What are the real-holdout ceiling and the no-learning floor, and why do you need them?

The ceiling scores real holdout rows - never used to train any model - drawn with the same
allocation-strata quotas: what a perfect model would get, which is not 1.0 because real data has
sampling noise too. The floor draws each row's stratum with the training frequencies and every
other column independently from training rows of that stratum, so it keeps per-stratum marginals
but destroys every within-row relationship. Both go through exactly the same families, and every
model also gets normalised scores, (model - floor) / (ceiling - floor). Absolute scores are read
between these two rows: a family on which the floor already scores close to the ceiling (the report
then shows `-` for the normalised score) cannot discriminate models.

*Open:* `bedsynth/evaluation/references.py`, `reports/week3_evaluation.md` section 2.

### C4. How did you set the pass thresholds of the operational checks?

By calibrating them to real-data noise at the evaluated sample size: the real data (train plus
holdout) is re-split 50 times (30 in quick mode) into a pseudo-training split and n real rows, and
each check records its real-versus-real distribution. A check passes when its value is at most
max(design threshold, noise limit), the noise limit being the larger of the largest real value and
the mean plus 3 standard deviations; a structural condition that real samples also fail is waived.
A per-check 99th percentile was rejected: with about 17 checks a real sample would still fail
some check in roughly 15 % of draws. Checks fully determined by the allocation strata (band mix and
source mix per ward) are reported as system checks and left out of the score.

*Open:* `bedsynth/evaluation/operational.py` (module docstring, `calibrate_operational`), `config/settings.yaml`.

### C5. What is the difference between raw validity and the 100 % you deliver?

Raw validity is the share of unfiltered model samples that pass every hard rule - what the model
produces on its own; it is the composite's validity component and the clearest measure of what
the constraints achieved (see the ablation). Delivered output is 100 % hard-valid by construction,
because rejection discards invalid rows; the generator's acceptance rate (accepted / sampled) also
counts rows rejected for leaving the requested stratum or conditions, so it can differ from the raw
rate. The report shows both, together with model-view versus system-view fidelity and operational
scores, so it is visible what rejection and the strata add on top of the model.

*Open:* `bedsynth/evaluation/validity.py`, `reports/week3_evaluation.md` sections 3-4.

### C6. What is TSTR, and why "size-matched" and "skill ratios"?

Train on synthetic, test on real: a gradient-boosting model is trained on synthetic data and
another on real training data, and both are tested on the real holdout. Tasks: end-of-day high
pressure and the number of discharges from start-of-day features (`ward_daily`); log length of
stay from admission features and delayed discharge from admission features plus destination
(`patient_stays`). The real-data model is trained on a subsample of the same size as the
synthetic sample, so the ratio measures data quality rather than training-set size, and the ratios
are skill ratios (R², AUC - 0.5, average-precision lift over the positive rate) clipped to [0, 1],
so a classifier without skill scores 0. Tasks on which even the real-data model has no skill are
excluded.

*Open:* `bedsynth/evaluation/utility.py`.

### C7. The UI shows a quality score for every dataset - is that your evaluation result?

No: that is the API's quick quality, the mean of column shapes (1 - KS / 1 - TVD), Spearman pair
trends and hard-rule validity, computed in a fraction of a second against the reference rows that
match the request (the full split when fewer than 20 rows match). Its validity component is
always 1.0 for delivered data, so it is optimistic and meant as a per-dataset sanity check. The
model comparison is the Week-3 pipeline - SDV QualityReport, calibrated operational checks,
utility, privacy and coverage, against the ceiling and the floor - reported in README Results and
the Week-3 report.

*Open:* `bedsynth/evaluation/statistical.py` (`quick_quality`), `bedsynth/api/service.py` (`compute_quality`).

### C8. Is the ward-day improvement real or seed noise?

The composite gain and the extra passed checks hold on all four evaluation seeds we tried; some
details do not. The band-grouped specs won an A/B test of seven variants against the previous spec
(same training split, one evaluation run with seed 42, a pre-set adoption bar; `docs/DESIGN.md`
section 7.2). Re-scored on evaluation seeds 7, 11 and 2024, the adopted spec scored 0.955-0.959
against 0.941-0.942 and passed 16-17 of 17 checks against 13-14: a gain of about 0.015, roughly four
times the largest seed-to-seed range (0.004). Limits: on seeds 7 and 11 it failed one borderline
check each (bed pressure versus cancellations; the weekend/weekday discharge ratio); its raw
hard-valid rate is 0.4-1.0 points lower on each seed (mean 0.7), which rejection absorbs; utility,
optimisation and privacy move more between seeds than the composite (the optimisation gain did not
appear on seeds 7 and 2024); the re-scores reused the same trained models, so training-seed
variation is untested; and the specs were chosen on the same evaluation whose utility and privacy
families use the holdout, so those figures carry some selection optimism. If asked why not x4: it
passed all 17 checks on every seed, but in the official run its privacy fell below the bar and its
utility below the previous spec's, while x7 alone met the bar and had the higher utility on all four
seeds; on the extra seeds the two are within seed-to-seed variation. The final full run retrained
the adopted spec and reproduced its composite exactly (0.958).

*Open:* `reports/experiments/ward_daily_band_groups_ab.json` (`summary`, `robustness_screens`), `docs/DESIGN.md` section 7.2.

## D. Occupancy, scheduling and scenario coverage (15 %)

### D1. How do your scenarios cover the brief?

26 scenarios (16 `ward_daily`, 10 `patient_stays`) in ten categories: baseline; normal, high and
low occupancy; seasonal (winter and summer demand, winter admissions); admission and discharge
patterns (weekend operations, emergency admissions, weekend discharges); emergency bed
requirements (emergency pressure, long bed waits); ward utilisation (escalation beds under
pressure, reduced capacity, long-stay patients); scheduled procedures (procedure days, elective
admissions, cancellations, rescheduled electives); bed turnover (high turnover, slow turnaround);
delayed discharge (ward pressure, delayed stays); staffing (shortage and surplus). Their reference
support is listed by `GET /api/v1/scenarios?include_support=true` and in the Week-1 report.
Scenarios select operating conditions rather than ward types - high turnover, for example, has a
ward condition that leaves out AMU and paediatrics, where rapid turnover is the daily norm.

*Open:* `bedsynth/scenarios.py`, `reports/week1_data_preparation.md` section 1.2.

### D2. How do you measure scenario coverage, and what about rare scenarios?

For each scenario the evaluation generates a sample (400 rows, 100 in quick mode) under a
per-scenario time budget and scores coverage - the mean of fulfilment (generated / requested) and
strata fidelity (1 - TVD between the output's strata mix and the real scenario subset's) - plus
fidelity, the quick quality against the real subset; the scenario score averages the two.
Scenarios not started within the overall budget are marked `not_evaluated` (family incomplete),
and a strata TVD more than 0.05 above the rounding floor is flagged. Rare filtered scenarios
(emergency pressure, reduced capacity, elective cancellations) have low acceptance because most
sampled rows fall outside the filter, so they need more rounds but remain exact; their strata
come from the model's own pilot draw, so a flag there means the model places those rows on other
wards or bands than the real data. Separately, the API demo stores one checked dataset per
scenario.

*Open:* `bedsynth/evaluation/coverage.py`, `reports/week3_evaluation.md` sections 2 (coverage column) and 7 (flags), `reports/api_demo/summary.md`.

### D3. Which occupancy and scheduling patterns do you check?

For `ward_daily`: occupancy mean, distribution and band mix per ward, the monthly emergency
profile, the seasonal pressure contrast (winter minus summer share of critical days), weekly
elective and discharge profiles, the weekend/weekday discharge ratio, before-noon discharges, bed
pressure versus cancellations, flow relationships (census and discharges, occupancy and boarders,
delays and occupancy, short staffing and cancellations, escalation and occupancy), short staffing
by season, boarding by occupancy band and zero-inflated counts. For `patient_stays`: length of
stay by source and ward, admission- and discharge-hour profiles, before-noon and weekend
discharges, delay prevalence by destination and mean delay by reason, turnaround, elective lead
time, reschedules and the lead-reschedule link, the elective weekday profile and admission
window, emergency bed waits, and source mix and transfer-out share by ward. All use the calibrated thresholds, and the
Week-3 report compares the scheduling relationships real versus synthetic.

*Open:* `bedsynth/evaluation/operational.py`, `reports/week3_evaluation.md` section 5.

### D4. Is the data useful for testing optimisation models?

We test it directly on `ward_daily` with three decision problems solved on real and on synthetic
ward-days: an elective booking quota per ward and weekday (a newsvendor by sample average
approximation, λ = 2) whose synthetic and real choices are both scored out of sample on the real
holdout, with a real-versus-real regret floor; admission-control policies (occupancy targets 0.85
to 1.00, with or without a nurse-ratio cap), scored by how accurately synthetic data predicts each
policy's objective (ranking agreement is only a diagnostic, because the real ranking is simply the
order of the targets); and capacity planning (95th percentile of beds needed per ward). In the final
run `gc_tuned`'s optimisation score is above the no-learning floor and level with the real-holdout
ceiling: its quotas lose less on the holdout than the real-versus-real regret floor and its policy
accuracy beats both references (Week-3 section 6). With the previous ward-day spec `gc_tuned`
scored no better than the floor on this evaluation seed, but on the three extra seeds of the A/B
re-scores the previous spec's optimisation score was sometimes as high as the adopted one's, so the
improvement depends on the evaluation sample (`docs/DESIGN.md` section 7.2; C8). Caveats: its
capacity-planning error is still larger than the floor's, exact quotas often differ from the real
ones even when the objective is close, the decision problems are stylised and defined for
`ward_daily` only (stays rely on two ML tasks), and the floor and ceiling were scored on one
evaluation seed. So the data suits functional, stress and scenario testing of optimisation code, and
policy conclusions should be confirmed on real data.

*Open:* `bedsynth/evaluation/optimization.py`, `reports/week3_evaluation.md` section 6.

## E. FastAPI (15 %)

### E1. What happens when I call `POST /api/v1/generate`, and what comes back?

Pydantic validates the body (strict integers, unknown fields rejected, calendar rules), then the
service resolves the scenario, normalises condition aliases, caps the condition combinations at
1,000, type- and range-checks the filters and checks the model id - all before any sampling. The
engine then runs under admission control and a time budget; the output is validated against every
hard and soft rule, optionally scored for quick quality, persisted as CSV plus metadata and
returned. The JSON response carries the dataset id, the model (id, synthesizer, SDV version, fit
timestamp), the scenario, the normalised request, the generation report (seed, attempts, sampled
and accepted rows, acceptance rate, rejections per rule and filter, allocation per combination and
strata, time budget, warnings), the rule-by-rule validation, quality, column descriptions and the
records; `"format": "csv"` returns the CSV with `X-BedSynth-*` headers instead.

*Open:* `bedsynth/api/schemas.py` (`GenerateRequest`), `bedsynth/api/service.py` (`BedSynthService.generate`), `/docs`.

### E2. What limits protect the service, and how do you handle concurrent requests?

Requests are bounded: 1 to 10,000 rows, calendar mode at most 1,100 days and 10,000 rows,
`time_budget_seconds` 60 by default and at most 120, `max_attempts` at most 100, at most 1,000
condition combinations, and `/evaluate` needs at least 20 rows and subsamples to at most 5,000.
Generations are deliberately serialised by one lock, because copulas and CTGAN swap the
process-wide numpy/torch random state while sampling and parallel draws would break seeded
reproducibility; admission control allows at most 4 generations running or queued, a queued request
gives up after 60 s, and overflow answers 503 `server_busy` with `Retry-After`, so `/health` and
the catalogue stay responsive. A scenario request typically takes from half a second to a few
seconds; a finer lock around the SDV sampling calls only, or several worker processes, would allow
parallel generation (future work).

*Open:* `config/settings.yaml` (`api`), `bedsynth/api/service.py` (`BedSynthService`, `_run_engine`).

### E3. How are errors reported?

Every error returns `{"detail", "error_type"}`, plus a field list for schema errors: 400 for
invalid conditions or filters, too many combinations, a scenario of the other table, a generation
error or too few rows; 404 for an unknown table (in the URL), scenario, model or dataset; 409 when
a dataset is in use; 422 for schema errors and for `no_rows_generated` (nothing is stored); 503
for missing models (the message gives the training command), a busy engine or an unavailable
component.
Impossible requests are refused before sampling - for example `occupancy_rate >= 1.5`. An
unhandled exception returns a generic 500 body; unexpected failures of the engine start-up,
validation or quality evaluation name only the exception type (the text and traceback stay in the
server log), and engine messages that are passed on are scrubbed of server paths. Malformed JSON
gets a 422 naming the character offset. Error responses also carry the CORS headers, so a browser
client on another local port can read them (the bundled UI is served from the API itself).

*Open:* `bedsynth/api/main.py` (exception handlers), `bedsynth/api/service.py` (error classes), `docs/DESIGN.md` section 10.

### E4. What does `/validate` do beyond running the rules?

It requires every training column, maps category aliases to canonical codes (`gen med` -> GMED,
`monday` -> Mon), keeps unknown values so they fail the category rule, treats non-numeric values
in numeric columns as missing, and recomputes every derived column. A provided derived value that
disagrees with its definition (for example a `census_end` that breaks the flow balance) makes the
row fail the API's hard check WD-D01 / PS-D01 `derived_values_consistent`, while the rules always
run on the recomputed values. The response lists every rule with its violations and example rows,
the invalid row positions and a normalisation summary, and can return the normalised records.

*Open:* `bedsynth/api/service.py` (`prepare_records`, `validate_records`), `demo/requests.http`.

### E5. How are datasets stored, and is the API safe to expose?

Each generation is stored as `data/synthetic/api/<dataset id>/data.csv` plus `metadata.json`
(request, report, validation, quality). Ids are `ds-` plus 12 hex characters, validated before
any file access and resolved strictly inside the store; writes are staged and moved into place
atomically, deletes are moved aside first, and a store lock serialises reads and mutations so a
delete cannot hit an open download on Windows. Model cards are redacted and engine error messages
scrubbed of server paths, and CORS admits only pages served from the same machine. There is
no authentication, so the API is meant for local or trusted networks; a deployment would add
authentication and HTTPS.

*Open:* `bedsynth/api/service.py` (`DatasetStore`, `redact_paths`, `scrub_paths`), `bedsynth/api/main.py` (`create_app`).

## F. Testing and documentation (10 %)

### F1. What is your testing strategy?

A pytest suite with ten test files, one per component (API, evaluation, generator,
preprocessing, reporting, schema, simulator, synthesis, validation) plus the submission packager -
633 fast tests (about 2.7 minutes) and 7 slow ones (about 2.5 minutes; README, Tests) - using fast
rule-satisfying fixtures and redirected paths. In the stored evidence of the final run all 633 fast
and 7 slow tests pass (Week-4 section 5). Be ready for one detail: before that run one
generator test that required exactly five rows from a pooled draw documented to return *up to* the
requested rows was relaxed to that documented contract (README, Tests).
Tests always use the real validation, evaluation and reporting code; fakes are used only
for speed or to inject failures (for example the API's fake generator). API tests drive every endpoint through
`fastapi.testclient` (contract, limits, error envelope, concurrency, security), integration tests
run against the real pipeline outputs, and the slow tests fit real CTGAN/TVAE models, check the
simulator calibration and run the real engine and `03_evaluate.py` end to end. Many tests pin bugs
we found: no-skill classifiers must score 0, the time budget must hold on the real clock,
filtered requests must never read the matching real rows, registry ids must not leave the models
folder.

*Open:* `tests/`, `docs/DESIGN.md` section 12, `reports/week4_api_system.md` section 5.

### F2. What evidence shows the system works end to end?

`scripts/05_api_demo.py` calls every endpoint in-process against the trained models - 58 steps,
including one persisted dataset per scenario - checks each response, and stores request, response
and checks per call in `reports/api_demo/` with a `summary.md`. With `--run-tests` and
`--run-slow-tests` it first runs the fast and the slow suite and stores their JUnit XML (host name
removed; home-directory paths, user names and host names in failure tracebacks redacted), which the
Week-4 report summarises. `demo/demo_client.py`
runs a similar walkthrough over HTTP against a live server, and the demo you just saw repeated the
key calls live.

*Open:* `reports/api_demo/summary.md`, `reports/week4_api_system.md`, `demo/demo_client.py`.

### F3. How can someone reproduce your results, and where is it all documented?

One command: `python scripts/run_pipeline.py` (the final full run, with both test suites, took 37
minutes on the development machine - about 15 to train, 15.5 to evaluate and 6.5 for the API demo
and the tests; `--quick` took about 16 minutes in the integration round), with
every tunable in `config/settings.yaml`, seeds for simulation, split, training, evaluation and
generation, and `BEDSYNTH_*_DIR` variables to redirect all outputs - without them a re-run overwrites
the submitted models, results and evidence (README, Quick start); `--run-tests` /
`--run-slow-tests` refresh the testing evidence too. The README covers setup, quick start, API
usage, bringing your own extract, validity guarantees, configuration, results, tests, packaging and
limitations; `docs/DESIGN.md` is the full specification (with the design rationale in section 1.1);
`docs/API.md` is the API reference; the four weekly reports are rendered from the results; `/docs`
documents every endpoint with request examples; `demo/requests.http` holds ready-made requests; and
`scripts/package_submission.py` builds the submission zip after scanning it for machine identifiers.

*Open:* `README.md`, `docs/DESIGN.md`, `docs/DELIVERABLES.md`, `scripts/run_pipeline.py`, `config/settings.yaml`.

## G. Ethics, privacy and limitations

### G1. Is the synthetic data private, and how do you measure it?

Privacy is assessed methodologically, because the reference data is simulated; the same checks
would run on a real extract. The privacy family compares synthetic rows with training rows: the
exact-match rate against the natural duplicate rate of real holdout rows, distance to closest
record (synthetic-to-train versus holdout-to-train: median ratio and share closer than the
holdout's 5th percentile) and an overfitting-protection component, with sdmetrics' DCR metrics as
an unscored cross-check in full runs. For the default models, sdmetrics' `DCRBaselineProtection` is
only 0.245 (ward-days) and 0.232 (stays): it measures the distance to the real rows against uniformly
random rows, so data that follows the real distribution scores low even when nothing is copied
(`DCROverfittingProtection` is 0.938 and 1.000; Week-3 section 8, `docs/DESIGN.md` section 13). An
unscored `aggregate_disclosure` block covers what
record-level checks cannot see: the generator reproduces the training split's exact frequencies of
its strata columns (for example the share of critical ICU days), which matters for competitive
sensitivity. There is no formal differential-privacy guarantee.

*Open:* `bedsynth/evaluation/privacy.py`, `docs/DESIGN.md` sections 9 and 13.

### G2. Could the generator leak real records?

Records are sampled from fitted distributions, not copied, but two mechanisms reproduce training
values by design: `conditional_marginals` only generates values observed in a group (exact
marginals), and the allocation strata reproduce exact training frequencies - which is why the
distance-to-closest-record and aggregate-disclosure checks exist. Filtered requests never read the
real rows that match the filter: their strata weights come from a synthetic pilot draw, because a
tiny filtered subset would otherwise copy the make-up of a handful of real records into the output
and the report. The fitted model files store per-group empirical distributions of training values
(the model cards do not), so on a real extract the models themselves must be treated as
confidential.

*Open:* `bedsynth/synthesis/constraints.py` (`ConditionalQuantileConstraint`), `bedsynth/synthesis/generator.py` (pilot draw).

### G3. Which ethical issues did you consider?

Scope: the data is strictly non-clinical - no diagnoses, procedures, ages or personal identifiers -
and meant for testing operations and optimisation software, not for clinical or individual
decisions. Labelling: the UI footer states that all records are synthetic and non-clinical, and
every API response and stored dataset carries its generation metadata (model, seed, scenario), so
synthetic data is not mistaken for real evidence. Bias: the generator reproduces the reference
patterns, including any inequities in a real facility's operations, so policy conclusions must be
validated on real data.
Governance: on a real extract, models and outputs need access control and a disclosure review
(see G2).

*Open:* `bedsynth/schema.py`, `bedsynth/api/static/index.html` (footer).

### G4. What are the main limitations and known quality gaps?

By design: the reference data is simulated; rows are conditionally independent (no day-to-day
continuity in calendar mode); the two tables are synthesised independently, so stays are not
linked to ward-days; the strata reproduce exact training frequencies with no option to perturb
them; generations are serialised; and GaussianCopula's native conditioning, used only for the
rarest condition combinations, is narrower than the true conditional distribution. The
measured gaps of the default models are listed in Week-3 sections 5-7 and quantified in
`docs/DESIGN.md` section 13. In the final run: on `ward_daily` every scored check passes on the
official evaluation seed (16-17 of 17 on the four seeds of the A/B re-scores, C8), and some pass
narrowly - the winter-summer contrast of critical days is flattened and weekend discharges are
relatively too high; on `patient_stays` fidelity is slightly below the no-learning floor, three
checks pass but lie beyond real-data noise (length of stay by source, delay prevalence by
destination, mean delay by reason), the length-of-stay columns have the worst shapes and
care-facility delays are too short; and on both tables filtered
scenarios place rows on other wards, bands or delay strata than the real subset (Week-3 section 7).
The boarder-occupancy link and the seasonal short-staffing share that the previous ward-day spec
failed pass with the band-grouped specs (`docs/DESIGN.md` section 7.2).

*Open:* `README.md` (Limitations), `docs/DESIGN.md` section 13, `reports/week3_evaluation.md` section 7.

### G5. What would you do next?

Model temporal dynamics - a sequential synthesizer such as SDV's PAR, or a post-pass that chains
each day's opening census to the previous day's close - and find a way to link the tables (a
multi-table model would first need a key that fits stays spanning several ward-days). Add a
stay-level decision test, so optimisation testing covers `patient_stays` too. Add privacy controls
before touching a real extract (noise on the strata weights, differentially private synthesizers, a
disclosure review), then retrain and recalibrate on that extract with its own ward catalogue. Close
the measured gaps - the flattened seasonal contrast, the weaker link between bed pressure and
elective cancellations, care-facility delays - and check the ward-day result across training seeds
(the four evaluation seeds re-scored one trained model). Finally, allow parallel generation with a
finer lock or worker processes, and add authentication for a deployment.

*Open:* `docs/DESIGN.md` section 13, `README.md` (Limitations).
