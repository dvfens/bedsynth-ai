"""Tests for the weekly report renderers (``bedsynth.reporting``) and ``scripts/run_pipeline.py``.

Week 1 and Week 2 are rendered from real pipeline outputs on fixture data (``prepare_datasets``
and a GaussianCopula trained through ``scripts/02_train_synthesizers.py``); Week 3 and Week 4
from small hand-made result dicts and evidence files, which keeps these tests fast. All outputs
go to the isolated temporary directories.
"""
from __future__ import annotations

import copy
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from bedsynth.config import get_settings
from bedsynth.data import io
from bedsynth.data.fixtures import make_patient_stays, make_ward_daily
from bedsynth.reporting import (
    _common,
    load_results,
    registry_cards,
    render_week1_report,
    render_week2_report,
    render_week3_report,
    render_week4_report,
    week1,
    week3,
)
from bedsynth.schema import PATIENT_STAYS, RAW_COLUMNS, WARD_DAILY

PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMAGE_LINK = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")
TABLE_SEPARATOR = re.compile(r"^\|(---\|)+$")


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(f"script_{name}", PROJECT_ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _assert_images_resolve(report: Path) -> list[str]:
    """Every Markdown image link of ``report`` points at an existing PNG; returns the links."""
    links = IMAGE_LINK.findall(report.read_text(encoding="utf-8"))
    for link in links:
        assert (report.parent / link).is_file(), link
    return links


def _assert_tables_consistent(text: str) -> None:
    """Every Markdown table row (outside code blocks) has as many cells as its header."""
    in_code, header = False, None
    for line in text.splitlines():
        if line.startswith("```"):
            in_code = not in_code
            continue
        if in_code or not line.startswith("|"):
            header = None
            continue
        cells = len(re.split(r"(?<!\\)\|", line.strip()[1:-1]))
        if header is None:
            header = cells
        assert cells == header, line


def _assert_no_location(text: str, *paths: Path) -> None:
    """No absolute location of the test machine leaks into a report."""
    for path in (*paths, Path.home()):
        for form in (str(path), path.as_posix()):
            assert form not in text, form


def _headings(text: str) -> list[str]:
    return [line for line in text.splitlines() if line.startswith("#")]


# ---------------------------------------------------------------------------
# Shared helpers (_common)
# ---------------------------------------------------------------------------
def test_display_path_never_prints_absolute_locations(tmp_path):
    inside = PROJECT_ROOT / "data" / "processed" / "ward_daily_train.csv"
    assert _common.display_path(inside) == "data/processed/ward_daily_train.csv"
    assert _common.display_path(str(PROJECT_ROOT / "reports")) == "reports"
    assert _common.display_path(tmp_path / "elsewhere" / "file.csv") == "file.csv"
    assert _common.display_path(r"C:\Users\someone\project\data\raw") == "raw"
    assert _common.display_path("/home/someone/project/data/raw") == "raw"
    assert _common.display_path("data\\processed\\metadata.json") == "data/processed/metadata.json"
    assert _common.display_path("../outside/secret.json") == "secret.json"
    # home-relative values (bedsynth.config.project_relative outside the project) are outside locations too
    assert _common.display_path("~/AppData/Local/Temp/run/raw") == "raw"
    assert _common.display_path("~\\models\\x.pkl") == "x.pkl" and _common.display_path("~") == "~"
    assert _common.display_path(None) == "-"


def test_to_utc_and_project_references():
    assert _common.to_utc("2026-11-15T04:51:41.707436+05:30") == "2026-11-14T23:21:41+00:00"
    assert _common.to_utc("2026-01-01T00:00:00+00:00") == "2026-01-01T00:00:00+00:00"
    assert _common.to_utc(None) is None and _common.to_utc("not a date") == "not a date"
    assert _common.to_utc("2026-01-01T12:00:00").endswith("+00:00")  # naive = local time
    # a report inside the project links the file relative to its own folder; elsewhere: code text only
    assert _common.project_ref("docs/DESIGN.md", PROJECT_ROOT / "reports") == "[`docs/DESIGN.md`](../docs/DESIGN.md)"
    assert _common.project_ref("docs/DESIGN.md", Path.home()) == "`docs/DESIGN.md`"
    assert _common.project_ref("docs/no_such_file.md", PROJECT_ROOT / "reports") == "`docs/no_such_file.md`"


# ---------------------------------------------------------------------------
# Week 1
# ---------------------------------------------------------------------------
def _prepare_fixture_data(isolated_paths) -> dict:
    from bedsynth.data.preprocessing import prepare_datasets

    raw_dir = isolated_paths["raw_dir"]
    make_ward_daily(n_days=30, seed=1)[list(RAW_COLUMNS[WARD_DAILY])].to_csv(io.raw_path(WARD_DAILY, raw_dir), index=False)
    stays = make_patient_stays(n=300, seed=2, with_times=True)[list(RAW_COLUMNS[PATIENT_STAYS])]
    stays.to_csv(io.raw_path(PATIENT_STAYS, raw_dir), index=False)
    return prepare_datasets()


WEEK1_FIGURES = {"week1_issues_detected_vs_injected.png", "week1_occupancy_by_month.png",
                 "week1_occupancy_band_mix.png", "week1_los_by_source.png"}


def test_week1_report_from_prepared_fixture_data(isolated_paths, tmp_path):
    report = _prepare_fixture_data(isolated_paths)

    path = render_week1_report(report)

    assert path == isolated_paths["reports_dir"] / "week1_data_preparation.md"
    text = path.read_text(encoding="utf-8")
    headings = _headings(text)
    assert headings[1] == "## Summary"
    for heading in ("## 1. Data requirements", "## 3. Cleaning pipeline results", "## 6. Operational rule validation",
                    "## 7. Dataset consistency", "## 8. Reference data patterns", "## 10. Output files"):
        assert heading in headings
    assert "No simulation manifest found" in text and "| AMU | Acute Medical Unit |" in text
    assert "| metric | ward_daily | patient_stays |" in text and "| train / holdout rows |" in text
    assert "pooled rejection from unconditional SDV draws" in text and "passed to SDV conditional sampling" not in text
    assert "SDV constraints declared for this table" in text
    from bedsynth.scenarios import SCENARIOS

    by_table = [sum(1 for s in SCENARIOS.values() if s.table == t) for t in (WARD_DAILY, PATIENT_STAYS)]
    assert (f"{len(SCENARIOS)} named scenarios ({by_table[0]} ward_daily, {by_table[1]} patient_stays) cover 9 of the 9 "
            "scenario families of the brief") in text
    assert "`ward_daily_train.csv`" in text and "| input | raw ward_daily extract" in text
    assert {Path(link).name for link in _assert_images_resolve(path)} == WEEK1_FIGURES
    _assert_tables_consistent(text)
    _assert_no_location(text, tmp_path)

    again = render_week1_report()  # no argument: reads data_quality_report.json from processed_dir
    assert "## Summary" in again.read_text(encoding="utf-8")


def test_week1_resolves_project_relative_and_stale_paths(isolated_paths, tmp_path, monkeypatch):
    report = _prepare_fixture_data(isolated_paths)

    # project-relative locations (as stored by newer pipelines) resolve against the project root,
    # even when the configured processed_dir points somewhere else
    relative = copy.deepcopy(report)
    relative["raw_dir"], relative["processed_dir"] = "raw_dir", "processed_dir"
    relative["outputs"] = {key: f"processed_dir/{Path(value).name}" for key, value in report["outputs"].items()}
    for table in (WARD_DAILY, PATIENT_STAYS):
        relative["tables"][table]["raw_file"] = f"raw_dir/{table}_raw.csv"
    monkeypatch.setattr(week1, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(_common, "PROJECT_ROOT", tmp_path)
    monkeypatch.setenv("BEDSYNTH_PROCESSED_DIR", str(tmp_path / "missing_processed"))
    text = render_week1_report(relative).read_text(encoding="utf-8")
    assert "`processed_dir/ward_daily_train.csv`" in text and "`raw_dir/ward_daily_raw.csv`" in text
    assert "writes the SDV-ready dataset to `processed_dir/`" in text
    assert "_The cleaned data was not found" not in text and "### 1.2" in text
    _assert_no_location(text, tmp_path)

    # stale absolute locations fall back to the configured directories: no section disappears
    monkeypatch.setenv("BEDSYNTH_PROCESSED_DIR", str(isolated_paths["processed_dir"]))
    stale = copy.deepcopy(report)
    moved = tmp_path / "moved_elsewhere"
    stale["raw_dir"], stale["processed_dir"] = str(moved / "raw"), str(moved / "processed")
    stale["outputs"] = {key: str(moved / "processed" / Path(value).name) for key, value in report["outputs"].items()}
    path = render_week1_report(stale)
    text = path.read_text(encoding="utf-8")
    assert {Path(link).name for link in _assert_images_resolve(path)} == WEEK1_FIGURES
    assert "`processed_dir/metadata_ward_daily.json`" in text
    assert "moved_elsewhere" not in text
    _assert_no_location(text, tmp_path)


def test_week1_expands_home_relative_locations(isolated_paths, tmp_path, monkeypatch):
    """Locations stored as ``~/...`` (bedsynth.config.project_relative under the home directory, outside the
    project) resolve against the home directory, not the project root, and only their file names are printed."""
    report = _prepare_fixture_data(isolated_paths)
    monkeypatch.setenv("HOME", str(tmp_path))            # tmp_path stands in for the home directory
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    tilde = copy.deepcopy(report)
    tilde["raw_dir"], tilde["processed_dir"] = "~/raw_dir", "~/processed_dir"
    tilde["outputs"] = {key: f"~/processed_dir/{Path(value).name}" for key, value in report["outputs"].items()}
    for table in (WARD_DAILY, PATIENT_STAYS):
        tilde["tables"][table]["raw_file"] = f"~/raw_dir/{table}_raw.csv"
    # the configured directories point elsewhere, so only the recorded '~/...' locations can be used
    monkeypatch.setenv("BEDSYNTH_RAW_DIR", str(tmp_path / "missing_raw"))
    monkeypatch.setenv("BEDSYNTH_PROCESSED_DIR", str(tmp_path / "missing_processed"))
    assert week1._stored_candidate("~/processed_dir") == tmp_path / "processed_dir"
    path = render_week1_report(tilde)
    text = path.read_text(encoding="utf-8")
    assert "_The cleaned data was not found" not in text
    assert {Path(link).name for link in _assert_images_resolve(path)} == WEEK1_FIGURES
    assert "`ward_daily_train.csv`" in text and "`ward_daily_raw.csv`" in text and "~/" not in text
    _assert_no_location(text, tmp_path)


def test_week1_report_without_quality_report(isolated_paths):
    with pytest.raises(FileNotFoundError, match="01_prepare_data.py"):
        render_week1_report()


def test_week1_band_mix_legend_sits_below_the_axes(tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    captured = {}
    original = plt.Figure.savefig

    def spy(fig, *args, **kwargs):
        ax = fig.axes[0]
        legend = ax.get_legend()
        fig.canvas.draw()
        captured["legend_top"] = legend.get_window_extent().y1
        captured["axes_bottom"] = ax.get_window_extent().y0
        return original(fig, *args, **kwargs)

    plt.Figure.savefig = spy
    try:
        week1._band_mix_figure(make_ward_daily(n_days=20, seed=4), tmp_path / "mix.png")
    finally:
        plt.Figure.savefig = original
    assert captured["legend_top"] < captured["axes_bottom"]


# ---------------------------------------------------------------------------
# Week 2 (through the training CLI)
# ---------------------------------------------------------------------------
def test_training_cli_writes_the_week2_report(isolated_paths, capsys):
    io.write_table(make_ward_daily(n_days=40, seed=3), io.processed_path(WARD_DAILY, "train"))
    script = _load_script("02_train_synthesizers")

    assert script.main(["--tables", WARD_DAILY, "--experiments", "gc_baseline", "--log-level", "WARNING"]) == 0

    report = isolated_paths["reports_dir"] / "week2_synthesis_pipeline.md"
    assert "Week-2 report written" in capsys.readouterr().out
    text = report.read_text(encoding="utf-8")
    assert "| ward_daily | gc_baseline (registry default) | GaussianCopulaSynthesizer |" in text
    assert "sdv.cag.FixedCombinations" in text and "## 2. Experiments (settings.yaml)" in text
    assert _headings(text)[1] == "## Summary" and "- 1 model trained (1 experiment x 1 table" in text
    assert "registry default when this report was rendered: `gc_baseline`" in text
    assert "Raw validity is measured on 2,000 unfiltered model samples" in text
    assert "Friday lists -36 %" not in text and "DESIGN.md §7.2-7.3" in text
    assert "- `gc_tuned`: " in text  # experiment descriptions are listed below the table, not in a cell
    _assert_images_resolve(report)
    _assert_tables_consistent(text)

    cards = registry_cards() + [{"table": PATIENT_STAYS, "experiment_id": "ctgan", "status": "failed", "error": "boom"}]
    again = render_week2_report(cards).read_text(encoding="utf-8")
    assert "## 7. Failed experiments" in again and "- patient_stays/ctgan: boom" in again
    assert "1 experiment(s) failed in this training run (section 7)" in again


def _card(experiment_id: str, raw: float, soft: dict[str, float], default: bool = False) -> dict:
    return {"table": WARD_DAILY, "experiment_id": experiment_id, "status": "trained", "is_default": default,
            "synthesizer": "GaussianCopulaSynthesizer", "training_rows": 100, "n_segments": 1, "fit_seconds": 30.0,
            "constraint_groups": [], "training": {"quick": False, "sample_diagnostics": {
                "rows": 500, "raw_hard_valid_rate": raw, "hard_violation_rates": {"WD-H10": 1 - raw},
                "soft_violation_rates": soft}}}


def test_week2_orders_models_by_ablation_and_formats_small_rates(isolated_paths):
    cards = [_card("tvae", 0.7, {}), _card("gc_tuned", 0.9, {"WD-S03": 0.012, "WD-S01": 0.0003}, default=True),
             _card("gc_baseline", 0.1, {"WD-S04": 0.2})]

    text = render_week2_report(cards).read_text(encoding="utf-8")

    models_section = text.split("## 4. Trained models", 1)[1].split("\n## ", 1)[0]
    rows = [line for line in models_section.splitlines() if line.startswith("| ward_daily | ")]
    assert [row.split(" | ")[1].split(" ")[0] for row in rows] == ["gc_baseline", "gc_tuned", "tvae"]
    assert "WD-S03 1.2 %" in text and "WD-S01" not in text  # rates below 0.05 % are not listed as '0 %'
    assert "gc_baseline 10.0 % -> gc_tuned 90.0 %" in text and "on 500 unfiltered model samples" in text
    assert "- 3 models trained (3 experiments x 1 table); total fit time 1.5 min (section 4)." in text
    _assert_tables_consistent(text)


def test_week2_report_without_models(isolated_paths):
    text = render_week2_report().read_text(encoding="utf-8")
    assert "No trained models yet" in text and "## Summary" not in text


# ---------------------------------------------------------------------------
# Week 3
# ---------------------------------------------------------------------------
def _model_result(model_id: str, score: float) -> dict:
    components = {"fidelity": score, "validity": score / 2, "operational": score, "utility": score, "privacy": 1.0}
    return {
        "table": WARD_DAILY, "model_id": model_id,
        "n_rows": {"real_train": 100, "real_holdout": 25, "synthetic": 50},
        "statistical": {"score": score, "score_source": "sdv_quality_report",
                        "sdv": {"quality": {"properties": {"Column Shapes": 0.9, "Column Pair Trends": 0.8}}},
                        "summary": {"mean_ks_statistic": 0.1, "worst_numeric_columns": ["census_start"]}},
        "validity": {"score": 0.4, "worst_hard_rules": [{"rule_id": "WD-H10", "name": "occupancy_level_consistent",
                                                          "violation_rate": 0.5}]},
        "operational": {"score": score, "n_checks": 2, "n_passed": 1, "failed_checks": ["weekly_elective_profile"],
                        "checks": [{"name": "occupancy_mean_by_ward", "group": "occupancy", "metric": "mean_abs_diff",
                                    "value": 0.01, "score": 0.9, "passed": True, "threshold_used": 0.03},
                                   {"name": "weekly_elective_profile", "group": "scheduling", "metric": "wape",
                                    "value": 0.3, "score": 0.6, "passed": False, "threshold_used": 0.1}]},
        "ml_utility": {"score": 0.8, "tasks": [{"name": "discharges_regression", "trtr": {"r2": 0.7}, "tstr": {"r2": 0.6},
                                               "score": 0.86, "informative": True}]},
        "optimization": {"score": 0.9, "components": {"booking_quota": 0.8, "policy_ranking": 1.0, "capacity_planning": 0.9},
                         "booking_quota": {"score": 0.8, "quota_agreement": 0.5, "relative_regret": 0.2},
                         "policy_ranking": {"score": 1.0, "kendall_tau": 1.0},
                         "capacity_planning": {"score": 0.9, "mean_relative_error": 0.1}},
        "privacy": {"score": 1.0, "components": {"exact_match_protection": 1.0},
                    "exact_matches": {"synthetic_rate": 0.0, "holdout_rate": 0.0}, "dcr": {"median_ratio": 1.1}},
        "scenario_coverage": {"score": 0.9, "n_scenarios": 1, "n_succeeded": 1, "mean_fulfilment": 1.0, "mean_fidelity": 0.9,
                              "rows_per_scenario": 50,
                              "scenarios": [{"name": "baseline", "category": "baseline", "real_support": {"rows": 100},
                                             "requested": 50, "generated": 50, "acceptance_rate": 0.3,
                                             "strata_mix_tvd": 0.02, "strata_rounding_tvd": 0.015,
                                             "combinations": {"coverage": 1.0}, "coverage": 1.0, "fidelity": 0.9,
                                             "score": 0.95, "seconds": 0.4}]},
        "composite": {"score": score, "components": components},
        "generation": {"acceptance_rate": 0.3},
    }, {"model_id": model_id, "composite": score, "components": components, "seconds": 1.0,
        "results_file": f"{WARD_DAILY}__{model_id}.json"}


def test_week3_report_from_results_and_from_files(isolated_paths):
    best, best_entry = _model_result("gc_tuned", 0.8)
    other, other_entry = _model_result("ctgan", 0.6)
    comparison = {"generated_at": "2026-01-01T00:00:00+00:00",
                  "composite_weights": {"fidelity": 0.3, "validity": 0.2, "operational": 0.25, "utility": 0.15, "privacy": 0.1},
                  "tables": {WARD_DAILY: {"models": [best_entry, other_entry], "best_model": "gc_tuned", "best_composite": 0.8}}}
    results = {"comparison": comparison, "results": {f"{WARD_DAILY}__gc_tuned": best, f"{WARD_DAILY}__ctgan": other}}

    path = render_week3_report(results)

    text = path.read_text(encoding="utf-8")
    assert path.name == "week3_evaluation.md"
    assert "| gc_tuned **(best)** | 0.800 |" in text and "### ward_daily / gc_tuned (selected default)" in text
    assert "### ward_daily / ctgan" in text and "weekly_elective_profile" in text
    assert "Kendall tau 1.000" in text and "| baseline | baseline | 100 | 50/50 |" in text
    assert "the model with the best complete composite" in text  # no stored selection rule
    assert len(_assert_images_resolve(path)) == 2
    _assert_tables_consistent(text)

    results_dir = isolated_paths["results_dir"]
    (results_dir / "model_comparison.json").write_text(json.dumps(comparison), encoding="utf-8")
    for key, value in results["results"].items():
        (results_dir / f"{key}.json").write_text(json.dumps(value), encoding="utf-8")
    assert set(load_results()["results"]) == {f"{WARD_DAILY}__gc_tuned", f"{WARD_DAILY}__ctgan"}
    assert "selected default" in render_week3_report().read_text(encoding="utf-8")


def _ablation_results() -> dict:
    """Three ablation steps, two references, quick run configuration and undefined checks."""
    entries, results = [], {}
    cards = {
        "gc_baseline": {"segment_label": "global", "constraint_groups": [], "effective_constraint_groups": [],
                        "raw_hard_valid_rate": 0.05, "fitted_at": "2026-01-01T00:00:01+00:00"},
        "gc_segmented": {"segment_label": "ward_id", "constraint_groups": ["inequality", "fixed_combinations"],
                         "raw_hard_valid_rate": 0.3, "fitted_at": "2026-01-01T00:00:02+00:00"},
        "gc_tuned": {"segment_label": "ward_id", "raw_hard_valid_rate": 0.9, "fitted_at": "2026-01-01T00:00:03+00:00",
                     "constraint_groups": ["inequality", "fixed_combinations", "regime_labels", "capacity"]},
    }
    for model_id, score in (("gc_baseline", 0.5), ("gc_segmented", 0.7), ("gc_tuned", 0.85)):
        result, entry = _model_result(model_id, score)
        result["model_card"] = cards[model_id]
        result["generation"] = {"acceptance_rate": 0.4, "strata_mix_tvd": 0.12, "strata_rounding_tvd": 0.01,
                                "rebalancing_flag": True}
        entries.append(entry)
        results[f"{WARD_DAILY}__{model_id}"] = result
    tuned = results[f"{WARD_DAILY}__gc_tuned"]
    tuned["optimization"]["policy_ranking"]["real_ranking_monotone"] = True
    tuned["operational"]["checks"].append({"name": "seasonal_pressure_contrast", "group": "demand", "metric": "abs_diff",
                                           "value": None, "passed": None, "score": None, "threshold_used": 0.1})
    baseline = results[f"{WARD_DAILY}__gc_baseline"]
    baseline["operational"]["checks"][1].update(value=None, details={"note": "fewer than 5 synthetic rows"})
    references, ref_entries = {}, []
    for ref_id, score in (("real_holdout", 0.95), ("null_stratified_marginals", 0.6)):
        reference, ref_entry = _model_result(ref_id, score)
        reference["optimization"]["policy_ranking"]["kendall_tau"] = 1.0 if ref_id == "real_holdout" else 0.5
        references[ref_id] = reference
        ref_entry.pop("results_file")
        ref_entries.append({**ref_entry, "kind": "reference", "status": "ok"})
    run_config = {"quick": True, "rows": 50, "rows_per_scenario": 5, "scenarios": True, "seed": 1, "sdmetrics": False}
    comparison = {"generated_at": "2026-01-01T00:00:00+00:00", "composite_weights": {"fidelity": 1.0},
                  "selection_rule": "highest composite among complete entries",
                  "tables": {WARD_DAILY: {"models": entries, "references": ref_entries, "best_model": "gc_tuned",
                                          "best_composite": 0.85, "run_config": run_config}}}
    return {"comparison": comparison, "results": results, "references": {WARD_DAILY: references}}


def test_week3_summary_patterns_ablation_and_notes(isolated_paths, monkeypatch):
    registry = {"gc_segmented": {"effective_constraint_groups": ["inequality"], "fitted_at": "2026-01-01T00:00:02+00:00"}}
    monkeypatch.setattr(week3, "_registry_card", lambda table, model_id: registry.get(model_id))

    text = render_week3_report(_ablation_results()).read_text(encoding="utf-8")
    headings = _headings(text)

    assert headings[1] == "## Summary" and headings.index("## Summary") < headings.index("## 1. Run configuration")
    assert "selected `gc_tuned`, composite 0.850 (no-learning floor 0.600, real-holdout ceiling 0.950)" in text
    assert "composite ward_daily gc_baseline 0.500 -> gc_segmented 0.700 -> gc_tuned 0.850" in text
    assert "the model with the highest composite among complete entries" in text
    assert "**Caveat - quick evaluation:** 50 rows per model and 5 rows per scenario, no sdmetrics" in text
    # applied constraint groups: result file, registry card, declared fallback (fixed_combinations is not a ward_daily group)
    assert "| gc_baseline | global | none | 5.0 % |" in text
    assert "| gc_segmented | ward_id | inequality | 30.0 % |" in text
    assert "| gc_tuned | ward_id | inequality, regime_labels, capacity \\* | 90.0 % |" in text
    assert "listed are the groups it enables that are declared for the table" in text
    assert "### Improvement log" in text and "WD-H15 (only free beds are closed" in text and "PS-H14 (" in text
    # occupancy/flow and scheduling checks with pass thresholds; undefined values explained
    assert "## 5. Scheduling relationships and occupancy patterns" in headings
    assert "**Occupancy, capacity and patient-flow checks**" in text and "**Scheduling checks**" in text
    assert "| check | metric | pass threshold | real noise limit | real holdout | floor |" in text
    assert "n/a (statistic undefined) not scored" in text and "n/a (fewer than 5 synthetic rows) FAIL" in text
    assert "Checks undefined on this model's output (not scored): `seasonal_pressure_contrast`" in text
    assert "even the no-learning floor reaches tau 0.500" in text and "tau ~0.7" not in text
    assert "| rounding floor |" in text and "0.120 (flagged) | 0.010 |" in text
    assert "## 8. Model details (appendix)" in headings and "| pass threshold | real noise limit |" in text
    assert "- Glossary: **KS** = " in text and "**TSTR / TRTR**" in text
    _assert_tables_consistent(text)


def test_week3_applied_groups_without_registry_card(isolated_paths, monkeypatch):
    monkeypatch.setattr(week3, "_registry_card", lambda table, model_id: None)
    results = _ablation_results()
    results["results"][f"{WARD_DAILY}__gc_tuned"]["model_card"]["effective_constraint_groups"] = ["regime_labels"]

    text = render_week3_report(results).read_text(encoding="utf-8")

    assert "| gc_tuned | ward_id | regime_labels | 90.0 % |" in text
    assert "| gc_segmented | ward_id | inequality \\* |" in text  # declared fallback, marked


def test_week3_report_requires_results(isolated_paths):
    with pytest.raises(FileNotFoundError, match="03_evaluate.py"):
        render_week3_report()


# ---------------------------------------------------------------------------
# Week 4
# ---------------------------------------------------------------------------
def _demo_summary() -> dict:
    return {
        "generated_at": "2026-01-01T00:00:00+00:00", "sdv_version": "2.2.0", "rows": 50, "seed": 42,
        "passed": 2, "failed": 0, "skipped": 1, "total": 3,
        "models": {WARD_DAILY: {"default": "gc_tuned", "trained": ["gc_tuned"]}},
        "highlights": ["ward_daily primary: 50 rows, acceptance 30.0%"],
        "steps": [
            {"index": 1, "name": "service info", "method": "GET", "path": "/", "status_code": 200, "elapsed_ms": 3.0,
             "outcome": "ok", "checks": [{"check": "links", "passed": True}]},
            {"index": 2, "name": "generate", "method": "POST", "path": "/api/v1/generate", "status_code": 200,
             "elapsed_ms": 900.0, "outcome": "ok", "checks": [], "evidence": "02_generate.json"},
            {"index": 3, "name": "evaluate", "method": "POST", "path": "/api/v1/evaluate", "status_code": None,
             "elapsed_ms": 0.0, "outcome": "skipped", "checks": []},
        ],
        "scenario_coverage": [{"scenario": "baseline", "table": WARD_DAILY, "category": "baseline", "n_rows": 10,
                               "acceptance_rate": 0.5, "all_hard_rules_passed": True, "quality_score": 0.9,
                               "dataset_id": "ds-000000000000", "outcome": "ok", "persisted": True}],
        "scenario_rows": 10, "datasets_dir": "data/synthetic/api_demo",
    }


def _write_evidence(demo_dir: Path) -> None:
    demo_dir.mkdir(parents=True, exist_ok=True)
    records = [{"record_id": f"WD-{i:06d}", "ward_id": "AMU", "census_start": 20 + i} for i in range(2)]
    body = {"dataset_id": "ds-000000000001", "persisted": True, "table": WARD_DAILY, "n_rows": 50,
            "generation": {"seed": 42, "acceptance_rate": 0.3, "rejected_by_rule": {"WD-H06": 3}, "warnings": []},
            "validation": {"all_hard_rules_passed": True, "rules": [{"rule_id": "WD-H01"}] * 3},
            "records": records, "records_truncated": {"shown": 2, "total": 50}}
    evidence = {"request": {"method": "POST", "path": "/api/v1/generate", "json": {"table": WARD_DAILY, "num_rows": 50}},
                "response": {"status_code": 200, "headers": {"x-bedsynth-rows": "50", "x-bedsynth-hard-rules-passed": "true"},
                             "body": body}}
    (demo_dir / "02_generate.json").write_text(json.dumps(evidence), encoding="utf-8")


JUNIT = ('<?xml version="1.0" encoding="utf-8"?><testsuites><testsuite name="pytest" tests="{n}" time="{t}" '
         'timestamp="2026-01-01T05:30:00.123456+05:30" hostname="build-host">{cases}</testsuite></testsuites>')


def _write_junit(path: Path, names: list[str], seconds: float) -> None:
    cases = "".join(f'<testcase classname="tests.test_api" name="{name}" time="0.1" />' for name in names)
    path.write_text(JUNIT.format(n=len(names), t=seconds, cases=cases), encoding="utf-8")


def test_week4_report_from_demo_summary(isolated_paths, tmp_path):
    summary = _demo_summary()
    path = render_week4_report(summary)
    text = path.read_text(encoding="utf-8")
    assert "**2 passed, 0 failed, 1 skipped** of 3 steps" in text and "| ward_daily | gc_tuned | gc_tuned |" in text
    assert "No model comparison yet" in text
    assert "_No generation evidence yet" in text  # the step's evidence file does not exist
    assert len(_assert_images_resolve(path)) == 1

    demo_dir = isolated_paths["reports_dir"] / "api_demo"
    _write_evidence(demo_dir)
    (demo_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    text = render_week4_report().read_text(encoding="utf-8")
    assert "**2 passed" in text and "| scenario | table | category | rows | acceptance |" in text
    assert "| baseline | ward_daily | baseline | 10 | 50.0 % | yes |" in text
    assert "Demo step 2 (generate; evidence `02_generate.json`) sent `POST /api/v1/generate`" in text
    assert '"num_rows": 50' in text and '"... 49 more records"' in text and "records_truncated" not in text
    assert "`X-BedSynth-Rows`, `X-BedSynth-Hard-Rules-Passed`" in text
    assert "evidence in the demo folder" in text and "Demo evidence: `api_demo/`" in text
    _assert_tables_consistent(text)
    _assert_no_location(text, tmp_path)


def test_week4_documents_the_generator_from_the_schemas(isolated_paths):
    from bedsynth.api.schemas import GenerateRequest

    text = render_week4_report().read_text(encoding="utf-8")
    headings = _headings(text)
    assert headings[1] == "## Summary"
    for heading in ("### 1.1 Running the generator", "### 1.2 Generation request (`POST /api/v1/generate`)",
                    "### 1.3 Generation response", "### 1.4 Example request and response",
                    "### 1.5 Documentation and demo materials"):
        assert heading in headings
    assert "`python scripts/04_serve_api.py`" in text and "`--port <port>`" in text and "/ui" in text and "/docs" in text
    assert f"JSON body with {len(GenerateRequest.model_fields)} fields" in text
    assert "| `validate` | boolean | `true` |" in text and "| `num_rows` | integer, 1.." in text
    assert "| `table` | 'ward_daily' / 'patient_stays' | `ward_daily` |" in text
    assert "| `dataset_id` |" in text and "| `records` |" in text and "| `model` | The trained model" in text
    assert "`docs/API.md`" in text and "`demo/requests.http`" in text and "`demo/demo_client.py`" in text
    api = get_settings()["api"]
    assert f"at most {api['max_condition_combinations']:,} condition combinations" in text
    assert f"default {float(api['time_budget_seconds']):g} s" in text
    _assert_tables_consistent(text)


def test_week4_testing_evidence_in_utc_with_optional_slow_suite(isolated_paths):
    results_dir = isolated_paths["results_dir"]
    _write_junit(results_dir / "pytest_junit.xml", ["test_a", "test_b"], 12.5)

    text = render_week4_report().read_text(encoding="utf-8")
    assert "Fast test suite (`pytest -m \"not slow\"`, 2026-01-01T00:00:00+00:00): **2 passed" in text
    assert "+05:30" not in text and "build-host" not in text
    assert "python scripts/05_api_demo.py --run-slow-tests" in text and "Slow test suite" not in text
    assert "- Tests: 2 of 2 fast tests passed (no stored slow-suite result) (section 5)." in text

    _write_junit(results_dir / "pytest_junit_slow.xml", ["test_slow"], 300.0)
    text = render_week4_report().read_text(encoding="utf-8")
    assert "Slow test suite (`pytest -m slow`, 2026-01-01T00:00:00+00:00): **1 passed" in text
    assert "- Tests: 2 of 2 fast tests passed; 1 of 1 slow tests passed (section 5)." in text
    _assert_tables_consistent(text)


def test_week4_report_without_demo(isolated_paths):
    text = render_week4_report().read_text(encoding="utf-8")
    assert "No API demo evidence yet" in text and "_No generation evidence yet" in text
    assert "No fast-suite results stored yet" in text


def test_api_schemas_import_does_not_load_sdv():
    code = "import sys; import bedsynth.api.schemas; print('sdv' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "False"


# ---------------------------------------------------------------------------
# scripts/run_pipeline.py
# ---------------------------------------------------------------------------
def test_run_pipeline_commands_and_dry_run(capsys):
    pipeline = _load_script("run_pipeline")
    args = pipeline.parse_args(["--quick", "--seed", "7", "--steps", "evaluate", "train", "--dry-run"])
    commands = pipeline.build_commands(args)
    assert [step for step, _ in commands] == ["train", "evaluate"]   # pipeline order, not argument order
    train, evaluate = (argv for _, argv in commands)
    assert train[1].endswith("02_train_synthesizers.py") and train[2:] == ["--quick"]
    assert evaluate[1].endswith("03_evaluate.py") and evaluate[2:] == ["--quick", "--seed", "7"]
    full = pipeline.build_commands(pipeline.parse_args(["--resimulate"]))
    assert [step for step, _ in full] == ["prepare", "train", "evaluate", "demo"] and full[0][1][2:] == ["--resimulate"]
    assert pipeline.main(["--dry-run", "--steps", "prepare"]) == 0
    out = capsys.readouterr().out
    assert "=== prepare: " in out and "01_prepare_data.py" in out and "finished" not in out
