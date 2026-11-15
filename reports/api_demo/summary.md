# BedSynth AI - API demo evidence

Generated 2026-11-15T05:37:39+00:00 by `scripts/05_api_demo.py` (BedSynth 1.0.0, SDV 2.2.0, Python 3.13.4). Every call ran in-process through `fastapi.testclient` against the trained models; each row links to the JSON evidence of the call.

**Result: 58 passed, 0 failed, 0 skipped (58 steps; 300 rows per generation, base seed 42).**

## Trained models

| Table | Default model | Trained models | Reference data |
|---|---|---|---|
| ward_daily | gc_tuned | ctgan, gc_baseline, gc_segmented, gc_tuned, tvae | yes |
| patient_stays | gc_tuned | ctgan, gc_baseline, gc_segmented, gc_tuned, tvae | yes |

## Generation highlights

- ward_daily primary: 300 rows, acceptance 95.3%, 1.259 s, all hard rules passed, quality 0.979, model gc_tuned
- ward_daily calendar: 62 rows, acceptance 94.1%, 0.569 s, all hard rules passed, model gc_tuned
- patient_stays primary: 300 rows, acceptance 88.8%, 2.724 s, all hard rules passed, quality 0.972, model gc_tuned

## Scenario coverage (one persisted dataset per scenario)

26 of 26 scenario datasets were stored in `data/synthetic/api_demo`.

| Scenario | Table | Rows | Acceptance | Hard rules | Quality | Dataset | Outcome |
|---|---|---|---|---|---|---|---|
| baseline | ward_daily | 100 | 94.7% | passed | 0.971 | ds-dc8e9f37d3f6 | ok |
| normal_occupancy | ward_daily | 100 | 98.7% | passed | 0.969 | ds-b98194437b9d | ok |
| high_occupancy | ward_daily | 100 | 86.5% | passed | 0.967 | ds-42c61363bab8 | ok |
| low_occupancy | ward_daily | 100 | 100.0% | passed | 0.966 | ds-43a0a2d3e834 | ok |
| winter_demand | ward_daily | 100 | 88.6% | passed | 0.971 | ds-6771654a6af5 | ok |
| summer_demand | ward_daily | 100 | 98.9% | passed | 0.972 | ds-c314554720d1 | ok |
| emergency_pressure | ward_daily | 100 | 8.2% | passed | 0.925 | ds-11be79af12c9 | ok |
| escalation_capacity | ward_daily | 100 | 49.4% | passed | 0.959 | ds-32d6b158943d | ok |
| reduced_capacity | ward_daily | 100 | 7.1% | passed | 0.958 | ds-08b234a568b0 | ok |
| scheduled_procedures | ward_daily | 100 | 76.1% | passed | 0.958 | ds-d35d2686155d | ok |
| elective_cancellations | ward_daily | 100 | 6.4% | passed | 0.937 | ds-a17cb83ecdb8 | ok |
| high_turnover | ward_daily | 100 | 7.2% | passed | 0.920 | ds-284af7ca3936 | ok |
| delayed_discharge_pressure | ward_daily | 100 | 15.4% | passed | 0.964 | ds-732f9fed47b0 | ok |
| staffing_shortage | ward_daily | 100 | 83.8% | passed | 0.964 | ds-d24532bf1c02 | ok |
| staffing_surplus | ward_daily | 100 | 96.5% | passed | 0.971 | ds-f3ea90f07ab4 | ok |
| weekend_operations | ward_daily | 100 | 93.4% | passed | 0.973 | ds-af7d27070015 | ok |
| stays_baseline | patient_stays | 100 | 96.0% | passed | 0.968 | ds-042ba72ab2bf | ok |
| emergency_admissions | patient_stays | 100 | 92.3% | passed | 0.963 | ds-7c4b5dcad70e | ok |
| elective_admissions | patient_stays | 100 | 97.6% | passed | 0.966 | ds-e18368273d4b | ok |
| rescheduled_electives | patient_stays | 100 | 13.0% | passed | 0.957 | ds-71c8d26f99f9 | ok |
| delayed_discharge_stays | patient_stays | 100 | 91.3% | passed | 0.965 | ds-28e42f678d13 | ok |
| long_stay_patients | patient_stays | 100 | 23.6% | passed | 0.961 | ds-c0fa6e536439 | ok |
| weekend_discharges | patient_stays | 100 | 22.9% | passed | 0.969 | ds-b097bcb39fda | ok |
| winter_admissions | patient_stays | 100 | 94.9% | passed | 0.965 | ds-e413fc95cd59 | ok |
| slow_bed_turnaround | patient_stays | 100 | 8.0% | passed | 0.968 | ds-5082b934adda | ok |
| long_bed_waits | patient_stays | 100 | 13.3% | passed | 0.964 | ds-29fd764f5bff | ok |

## Test suites

| Suite | Command | Result | Tests | Passed | Failed | Errors | Skipped | Time (s) | JUnit XML |
|---|---|---|---|---|---|---|---|---|---|
| fast | `python -m pytest -q -m "not slow" --junitxml=reports/results/pytest_junit.xml` | ok | 633 | 633 | 0 | 0 | 0 | 161.9 | reports/results/pytest_junit.xml |
| slow | `python -m pytest -q -m "slow" --junitxml=reports/results/pytest_junit_slow.xml` | ok | 7 | 7 | 0 | 0 | 0 | 145.8 | reports/results/pytest_junit_slow.xml |

## Calls

| # | Step | Request | Status | Time (ms) | Checks | Outcome | Evidence |
|---|---|---|---|---|---|---|---|
| 1 | service info | `GET /` | 200 (expected 200) | 105.3 | 1/1 | ok | [01_service_info.json](01_service_info.json) |
| 2 | health | `GET /health` | 200 (expected 200) | 32.3 | 1/1 | ok | [02_health.json](02_health.json) |
| 3 | demo UI | `GET /ui` | 200 (expected 200) | 3.7 | 2/2 | ok | [03_demo_ui.json](03_demo_ui.json) |
| 4 | openapi schema | `GET /openapi.json` | 200 (expected 200) | 2.5 | 1/1 | ok | [04_openapi_schema.json](04_openapi_schema.json) |
| 5 | list tables | `GET /api/v1/tables` | 200 (expected 200) | 30.1 | 1/1 | ok | [05_list_tables.json](05_list_tables.json) |
| 6 | schema ward_daily | `GET /api/v1/tables/ward_daily/schema` | 200 (expected 200) | 3.4 | 2/2 | ok | [06_schema_ward_daily.json](06_schema_ward_daily.json) |
| 7 | schema patient_stays | `GET /api/v1/tables/patient_stays/schema` | 200 (expected 200) | 2.8 | 2/2 | ok | [07_schema_patient_stays.json](07_schema_patient_stays.json) |
| 8 | scenario catalogue (with reference support) | `GET /api/v1/scenarios` | 200 (expected 200) | 199.8 | 1/1 | ok | [08_scenario_catalogue_with_reference_support.json](08_scenario_catalogue_with_reference_support.json) |
| 9 | scenarios for patient_stays | `GET /api/v1/scenarios` | 200 (expected 200) | 2.8 | 1/1 | ok | [09_scenarios_for_patient_stays.json](09_scenarios_for_patient_stays.json) |
| 10 | scenario high_occupancy | `GET /api/v1/scenarios/high_occupancy` | 200 (expected 200) | 3.6 | 1/1 | ok | [10_scenario_high_occupancy.json](10_scenario_high_occupancy.json) |
| 11 | trained models | `GET /api/v1/models` | 200 (expected 200) | 84.5 | 1/1 | ok | [11_trained_models.json](11_trained_models.json) |
| 12 | generate ward_daily baseline (json, evaluate) | `POST /api/v1/generate` | 200 (expected 200) | 1368.2 | 4/4 | ok | [12_generate_ward_daily_baseline_json_evaluate.json](12_generate_ward_daily_baseline_json_evaluate.json) |
| 13 | generate ward_daily high_occupancy (evaluate) | `POST /api/v1/generate` | 200 (expected 200) | 1523.1 | 4/4 | ok | [13_generate_ward_daily_high_occupancy_evaluate.json](13_generate_ward_daily_high_occupancy_evaluate.json) |
| 14 | generate ward_daily emergency_pressure (csv) | `POST /api/v1/generate` | 200 (expected 200) | 4649.4 | 4/4 | ok | [14_generate_ward_daily_emergency_pressure_csv.json](14_generate_ward_daily_emergency_pressure_csv.json) |
| 15 | generate ward_daily calendar January (AMU, ICU) | `POST /api/v1/generate` | 200 (expected 200) | 590.2 | 3/3 | ok | [15_generate_ward_daily_calendar_january_amu_icu.json](15_generate_ward_daily_calendar_january_amu_icu.json) |
| 16 | generate patient_stays delayed_discharge_stays (evaluate) | `POST /api/v1/generate` | 200 (expected 200) | 2781.4 | 4/4 | ok | [16_generate_patient_stays_delayed_discharge_stays_evaluate.json](16_generate_patient_stays_delayed_discharge_stays_evaluate.json) |
| 17 | generate patient_stays long_bed_waits | `POST /api/v1/generate` | 200 (expected 200) | 1240.0 | 3/3 | ok | [17_generate_patient_stays_long_bed_waits.json](17_generate_patient_stays_long_bed_waits.json) |
| 18 | reproducibility run A (seed fixed) | `POST /api/v1/generate` | 200 (expected 200) | 1071.6 | - | ok | [18_reproducibility_run_a_seed_fixed.json](18_reproducibility_run_a_seed_fixed.json) |
| 19 | reproducibility run B (same seed) | `POST /api/v1/generate` | 200 (expected 200) | 1055.3 | 1/1 | ok | [19_reproducibility_run_b_same_seed.json](19_reproducibility_run_b_same_seed.json) |
| 20 | validate records (+1 corrupted) | `POST /api/v1/validate` | 200 (expected 200) | 46.9 | 2/2 | ok | [20_validate_records_1_corrupted.json](20_validate_records_1_corrupted.json) |
| 21 | evaluate stored dataset | `POST /api/v1/evaluate` | 200 (expected 200) | 87.7 | 1/1 | ok | [21_evaluate_stored_dataset.json](21_evaluate_stored_dataset.json) |
| 22 | evaluate client records | `POST /api/v1/evaluate` | 200 (expected 200) | 88.1 | 1/1 | ok | [22_evaluate_client_records.json](22_evaluate_client_records.json) |
| 23 | list datasets | `GET /api/v1/datasets` | 200 (expected 200) | 105.8 | 1/1 | ok | [23_list_datasets.json](23_list_datasets.json) |
| 24 | dataset metadata | `GET /api/v1/datasets/ds-dbd35a8375b2` | 200 (expected 200) | 7.4 | 1/1 | ok | [24_dataset_metadata.json](24_dataset_metadata.json) |
| 25 | download dataset (csv) | `GET /api/v1/datasets/ds-dbd35a8375b2/download` | 200 (expected 200) | 4.8 | 2/2 | ok | [25_download_dataset_csv.json](25_download_dataset_csv.json) |
| 26 | download dataset (json) | `GET /api/v1/datasets/ds-dbd35a8375b2/download` | 200 (expected 200) | 20.0 | 1/1 | ok | [26_download_dataset_json.json](26_download_dataset_json.json) |
| 27 | delete dataset | `DELETE /api/v1/datasets/ds-dbd35a8375b2` | 200 (expected 200) | 5.0 | 1/1 | ok | [27_delete_dataset.json](27_delete_dataset.json) |
| 28 | deleted dataset is gone (404) | `GET /api/v1/datasets/ds-dbd35a8375b2` | 404 (expected 404) | 2.9 | 1/1 | ok | [28_deleted_dataset_is_gone_404.json](28_deleted_dataset_is_gone_404.json) |
| 29 | invalid request (422) | `POST /api/v1/generate` | 422 (expected 422) | 2.5 | 1/1 | ok | [29_invalid_request_422.json](29_invalid_request_422.json) |
| 30 | unknown scenario (404) | `POST /api/v1/generate` | 404 (expected 404) | 2.6 | 1/1 | ok | [30_unknown_scenario_404.json](30_unknown_scenario_404.json) |
| 31 | scenario of another table (400) | `POST /api/v1/generate` | 400 (expected 400) | 2.5 | 1/1 | ok | [31_scenario_of_another_table_400.json](31_scenario_of_another_table_400.json) |
| 32 | impossible filter (400) | `POST /api/v1/generate` | 400 (expected 400) | 2.4 | 1/1 | ok | [32_impossible_filter_400.json](32_impossible_filter_400.json) |
| 33 | scenario baseline | `POST /api/v1/generate` | 200 (expected 200) | 1301.4 | 3/3 | ok | [33_scenario_baseline.json](33_scenario_baseline.json) |
| 34 | scenario normal_occupancy | `POST /api/v1/generate` | 200 (expected 200) | 1090.3 | 3/3 | ok | [34_scenario_normal_occupancy.json](34_scenario_normal_occupancy.json) |
| 35 | scenario high_occupancy | `POST /api/v1/generate` | 200 (expected 200) | 1439.3 | 3/3 | ok | [35_scenario_high_occupancy.json](35_scenario_high_occupancy.json) |
| 36 | scenario low_occupancy | `POST /api/v1/generate` | 200 (expected 200) | 1067.7 | 3/3 | ok | [36_scenario_low_occupancy.json](36_scenario_low_occupancy.json) |
| 37 | scenario winter_demand | `POST /api/v1/generate` | 200 (expected 200) | 2447.6 | 3/3 | ok | [37_scenario_winter_demand.json](37_scenario_winter_demand.json) |
| 38 | scenario summer_demand | `POST /api/v1/generate` | 200 (expected 200) | 2071.7 | 3/3 | ok | [38_scenario_summer_demand.json](38_scenario_summer_demand.json) |
| 39 | scenario emergency_pressure | `POST /api/v1/generate` | 200 (expected 200) | 4371.9 | 3/3 | ok | [39_scenario_emergency_pressure.json](39_scenario_emergency_pressure.json) |
| 40 | scenario escalation_capacity | `POST /api/v1/generate` | 200 (expected 200) | 3283.4 | 3/3 | ok | [40_scenario_escalation_capacity.json](40_scenario_escalation_capacity.json) |
| 41 | scenario reduced_capacity | `POST /api/v1/generate` | 200 (expected 200) | 4791.6 | 3/3 | ok | [41_scenario_reduced_capacity.json](41_scenario_reduced_capacity.json) |
| 42 | scenario scheduled_procedures | `POST /api/v1/generate` | 200 (expected 200) | 1487.4 | 3/3 | ok | [42_scenario_scheduled_procedures.json](42_scenario_scheduled_procedures.json) |
| 43 | scenario elective_cancellations | `POST /api/v1/generate` | 200 (expected 200) | 4044.1 | 3/3 | ok | [43_scenario_elective_cancellations.json](43_scenario_elective_cancellations.json) |
| 44 | scenario high_turnover | `POST /api/v1/generate` | 200 (expected 200) | 3518.0 | 3/3 | ok | [44_scenario_high_turnover.json](44_scenario_high_turnover.json) |
| 45 | scenario delayed_discharge_pressure | `POST /api/v1/generate` | 200 (expected 200) | 4435.5 | 3/3 | ok | [45_scenario_delayed_discharge_pressure.json](45_scenario_delayed_discharge_pressure.json) |
| 46 | scenario staffing_shortage | `POST /api/v1/generate` | 200 (expected 200) | 1649.6 | 3/3 | ok | [46_scenario_staffing_shortage.json](46_scenario_staffing_shortage.json) |
| 47 | scenario staffing_surplus | `POST /api/v1/generate` | 200 (expected 200) | 1337.4 | 3/3 | ok | [47_scenario_staffing_surplus.json](47_scenario_staffing_surplus.json) |
| 48 | scenario weekend_operations | `POST /api/v1/generate` | 200 (expected 200) | 1915.6 | 3/3 | ok | [48_scenario_weekend_operations.json](48_scenario_weekend_operations.json) |
| 49 | scenario stays_baseline | `POST /api/v1/generate` | 200 (expected 200) | 1347.0 | 3/3 | ok | [49_scenario_stays_baseline.json](49_scenario_stays_baseline.json) |
| 50 | scenario emergency_admissions | `POST /api/v1/generate` | 200 (expected 200) | 759.5 | 3/3 | ok | [50_scenario_emergency_admissions.json](50_scenario_emergency_admissions.json) |
| 51 | scenario elective_admissions | `POST /api/v1/generate` | 200 (expected 200) | 552.7 | 3/3 | ok | [51_scenario_elective_admissions.json](51_scenario_elective_admissions.json) |
| 52 | scenario rescheduled_electives | `POST /api/v1/generate` | 200 (expected 200) | 831.9 | 3/3 | ok | [52_scenario_rescheduled_electives.json](52_scenario_rescheduled_electives.json) |
| 53 | scenario delayed_discharge_stays | `POST /api/v1/generate` | 200 (expected 200) | 1946.6 | 3/3 | ok | [53_scenario_delayed_discharge_stays.json](53_scenario_delayed_discharge_stays.json) |
| 54 | scenario long_stay_patients | `POST /api/v1/generate` | 200 (expected 200) | 2593.9 | 3/3 | ok | [54_scenario_long_stay_patients.json](54_scenario_long_stay_patients.json) |
| 55 | scenario weekend_discharges | `POST /api/v1/generate` | 200 (expected 200) | 2606.9 | 3/3 | ok | [55_scenario_weekend_discharges.json](55_scenario_weekend_discharges.json) |
| 56 | scenario winter_admissions | `POST /api/v1/generate` | 200 (expected 200) | 930.5 | 3/3 | ok | [56_scenario_winter_admissions.json](56_scenario_winter_admissions.json) |
| 57 | scenario slow_bed_turnaround | `POST /api/v1/generate` | 200 (expected 200) | 2300.2 | 3/3 | ok | [57_scenario_slow_bed_turnaround.json](57_scenario_slow_bed_turnaround.json) |
| 58 | scenario long_bed_waits | `POST /api/v1/generate` | 200 (expected 200) | 1441.2 | 3/3 | ok | [58_scenario_long_bed_waits.json](58_scenario_long_bed_waits.json) |
