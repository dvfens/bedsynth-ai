"""Week-3 report: synthetic data quality and scheduling evaluation (DESIGN.md §9, §11).

:func:`render_week3_report` renders the results of ``scripts/03_evaluate.py``
(``{"comparison": <model_comparison>, "results": {"<table>__<model_id>": <evaluate_model>},
"references": {"<table>": {"<reference_id>": <evaluate_model>}}}``) as
``reports/week3_evaluation.md`` with comparison and scenario-coverage figures. Called with no
argument it reads ``reports/results/model_comparison.json``, the per-model result files and the
reference result files.

Sections: a summary computed from the results, run configuration (with quick-mode caveats),
model comparison next to the real-holdout ceiling and the no-learning floor (plus normalised
scores), the improvements made to SDV generation (the ablation ``gc_baseline -> gc_segmented ->
gc_tuned``, the ward_daily A/B test read from ``reports/experiments/ward_daily_band_groups_ab.json``
when that file describes the current ward_daily default, and an improvement log with DESIGN.md
cross-references), raw vs delivered validity (with
soft-rule rates against real data), occupancy, patient-flow and scheduling patterns per model,
optimisation-model testing, quality gaps, per-model details (appendix) and notes with a glossary.
Every section degrades gracefully when a result family, a reference or an ablation step is missing.

The constraint groups a model applies come from its result file (``model_card.
effective_constraint_groups``), else from its registry card (JSON only, no pickle is loaded), else
from the groups declared for the table that the model enables.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

from bedsynth import schema, validation
from bedsynth.config import get_path, get_settings
from bedsynth.reporting._common import (
    PALETTE,
    draw_figure,
    figure_line,
    fmt_int,
    fmt_num,
    fmt_pct,
    grouped_bars,
    is_number,
    md_table,
    output_dirs,
    utc_now,
    write_markdown,
)

logger = logging.getLogger(__name__)

REPORT_NAME = "week3_evaluation.md"
COMPARISON_FILE = "model_comparison.json"
COMPONENTS: tuple[str, ...] = ("fidelity", "validity", "operational", "utility", "privacy")
REAL_REFERENCE = "real_holdout"
NULL_REFERENCE = "null_stratified_marginals"
REFERENCE_LABELS: dict[str, str] = {REAL_REFERENCE: "real holdout (ceiling)",
                                    NULL_REFERENCE: "no-learning baseline (floor)"}
ABLATION: tuple[str, ...] = ("gc_baseline", "gc_segmented", "gc_tuned")
ABLATION_STEPS: dict[str, str] = {
    "gc_baseline": "Week-2 baseline: one global GaussianCopula, default encodings, no constraints",
    "gc_segmented": "+ per-ward segments, SDV built-in constraints, ordinal metadata / ordered encodings, tuned marginals",
    "gc_tuned": "+ programmable constraints (band labels, learned structural zeros, escalation cover, exact "
                "per-group marginals) and ward x source segments for patient_stays",
}
SCHEDULING_EXTRA: tuple[str, ...] = ("relationship_short_staffing_cancellations",)
SOFT_GAP_MARGIN = 0.02            # output soft-rule rate this far above the real rate is a quality gap
LOW_NORMALISED = 0.25             # normalised score at or below this: close to the no-learning floor
#: Components compared with the no-learning floor (privacy is ~1 for every generator that does not copy rows).
FLOOR_COMPONENTS: tuple[str, ...] = ("fidelity", "validity", "operational", "utility")
#: Hard rules added after the Week-3 data review (their wording is read from ``bedsynth.validation``).
REVIEW_RULES: tuple[str, ...] = ("WD-H15", "PS-H13", "PS-H14")

#: Improvement log of section 3: (title, what changed, where it is documented). Static text, no
#: measurements: the measured effects are the ablation above and DESIGN.md. Add one entry per
#: adopted change.
IMPROVEMENT_LOG: tuple[tuple[str, str, str], ...] = (
    ("Ablation recipe",
     "`gc_baseline` -> `gc_segmented` -> `gc_tuned`, one improvement block per step, measured in the tables above",
     "DESIGN.md §7.5; Week-2 report sections 2 and 5"),
    ("Constraint specs adopted after A/B tests",
     "patient_stays: length of stay per admission weekday, and exact admission/discharge-hour, bed-wait and "
     "turnaround marginals per ward x admission source (`conditional_marginals`)",
     "DESIGN.md §7.2"),
    ("Band-grouped constraint specs adopted after an A/B test",
     "ward_daily: ED boarders and the mean bed wait per ward x occupancy band and closed beds per ward x staffing "
     "band, both bands recomputed from the counts (`band_groups`, never read from the sampled labels), and day-shift "
     "nurses per ward x month replace the per-ward marginals of the mostly-zero counts and the parametric nurse "
     "marginal (`conditional_marginals`); the variants, the adoption bar and re-scores on three further evaluation "
     "seeds are kept as evidence",
     "DESIGN.md §7.2; reports/experiments/ward_daily_band_groups_ab.json"),
    ("Constraint spec rejected",
     "ward_daily: discharges per ward x weekday and nurses per ward; it fixed the weekend/weekday discharge ratio "
     "but weakened the seasonal pressure contrast, so it was not adopted",
     "DESIGN.md §7.2"),
    ("Overwriting repair removed",
     "an earlier `structural_zeros` spec forced delay hours to 0 when no delay reason was sampled; it kept the "
     "length of stay drawn with the discarded delay and inflated the stays of non-delayed patients, so delay hours "
     "are now modelled per delay reason by `conditional_marginals`",
     "DESIGN.md §7.2"),
    ("Encodings and marginals",
     "weekdays ordered by mean discharges instead of mean elective lists (which imposed a false weekday gradient; "
     "elective lists are now modelled per ward x weekday); marginals chosen by KS per segment and per column, with "
     "a post-fit check that refits degenerate fits",
     "DESIGN.md §7.3"),
    ("Rules added after the data review",
     "{review_rules}: the simulator guaranteed these invariants but no rule checked them, so earlier "
     "\"100 % hard-valid\" output could break them",
     "DESIGN.md §6.4"),
    ("Simulator realism fixes",
     "weekend discharge model with separate Saturday and Sunday capacities and a Friday push, outlier bedding "
     "timing, truncated decision-to-admit lag and ward delay factors, with realism targets added to the "
     "calibration; hot-deck imputation instead of group medians in the cleaning step (no artificial modes)",
     "DESIGN.md §6.1, §6.3"),
    ("Allocation strata and the synthetic pilot draw",
     "rejection accepts some strata more easily than others; quotas per ward x band (ward-days) and ward x "
     "source x delay reason (stays) keep the training mix, and filtered requests take their strata from a "
     "seeded synthetic pilot draw, never from the real rows that match the filters",
     "DESIGN.md §8.3; strata TVD and rebalanced rows in section 4"),
    ("Evaluation improvements",
     "operational thresholds calibrated to real-data noise at the evaluated size; size-matched train-on-real "
     "baselines and skill ratios for ML utility; booking-quota regret scored out of sample on the real holdout; "
     "the real-holdout ceiling and the no-learning floor scored with the same families",
     "DESIGN.md §9 items 4, 6, 7 and 10; sections 1-2 and 6"),
)

#: Evidence of the ward_daily A/B test (relative to the configured reports directory; kept as JSON and
#: summarised in section 3 when it describes the current ward_daily default).
AB_TEST_FILE = Path("experiments") / "ward_daily_band_groups_ab.json"
AB_COMPOSITE_TOLERANCE = 5e-4
"""Largest difference between the adopted spec's official composite and the default's composite."""
#: Synthesizer types of the deep models (``synthesis.experiments.<id>.model``) and their display names.
DEEP_MODELS: dict[str, str] = {"ctgan": "CTGAN", "tvae": "TVAE"}

#: Abbreviations used in the report (section 9).
GLOSSARY: tuple[tuple[str, str], ...] = (
    ("KS", "Kolmogorov-Smirnov statistic: the largest gap between two cumulative distributions (0 = identical)"),
    ("TVD", "total variation distance between two category mixes (0 = identical, 1 = disjoint)"),
    ("WAPE", "weighted absolute percentage error of a calendar profile"),
    ("TSTR / TRTR", "train on synthetic / train on real data, both tested on the real holdout"),
    ("DCR", "distance to the closest training record"),
    ("AUC / AP", "area under the ROC curve / average precision"),
    ("Kendall tau", "rank agreement of two orderings (1 = same order)"),
    ("allocation strata", "the ward x occupancy band (ward-days) or ward x admission source x delay reason (stays) "
                          "cells whose quotas the generator fills"),
    ("system check", "an operational check fully determined by those quotas (reported, not scored)"),
)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_results(results_dir: Path | str | None = None) -> dict[str, Any]:
    """``{"comparison", "results", "references"}`` from the files written by ``03_evaluate.py``."""
    directory = Path(results_dir) if results_dir is not None else get_path("results_dir", create=False)
    comparison_path = directory / COMPARISON_FILE
    if not comparison_path.is_file():
        raise FileNotFoundError(f"{comparison_path} not found: run `python scripts/03_evaluate.py` first.")
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    results: dict[str, Any] = {}
    references: dict[str, dict[str, Any]] = {}
    for table, info in (comparison.get("tables") or {}).items():
        for entry in info.get("models") or []:
            if entry.get("status", "ok") != "ok":  # failed in its run: no result file
                continue
            path = directory / (entry.get("results_file") or f"{table}__{entry.get('model_id')}.json")
            if path.is_file():
                results[f"{table}__{entry.get('model_id')}"] = json.loads(path.read_text(encoding="utf-8"))
        for entry in info.get("references") or []:
            path = directory / (entry.get("results_file") or f"{table}__reference__{entry.get('model_id')}.json")
            if path.is_file():
                references.setdefault(table, {})[entry.get("model_id")] = json.loads(path.read_text(encoding="utf-8"))
    return {"comparison": comparison, "results": results, "references": references}


# ---------------------------------------------------------------------------
# Small accessors
# ---------------------------------------------------------------------------
def _family(result: dict[str, Any] | None, name: str) -> dict[str, Any]:
    return dict((result or {}).get(name) or {})


def _check_map(result: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    return {c.get("name"): c for c in _family(result, "operational").get("checks") or []}


def _label(model_id: str | None) -> str:
    return REFERENCE_LABELS.get(model_id or "", model_id or "-")


def _span(values: list[float], fmt: Any) -> str:
    """``min-max`` of ``values`` formatted with ``fmt`` (one value when both ends format alike)."""
    low, high = fmt(min(values)), fmt(max(values))
    return low if low == high else f"{low}-{high}"


def load_ab_evidence(comparison: dict[str, Any], path: Path | None = None) -> dict[str, Any] | None:
    """Seed evidence of the ward_daily A/B test, checked against the current ward_daily default.

    Reads :data:`AB_TEST_FILE` in the configured reports directory (or ``path``). Returns ``None``
    when the file is missing or unreadable, ``{"stale": <reason>}`` when it does not describe the
    current default (its control is not the default model, its official evaluation seed is not this
    evaluation's seed, or the adopted spec's composite on that seed differs from the default's by more
    than :data:`AB_COMPOSITE_TOLERANCE`), else the facts section 3 and the summary quote. Ranges and
    means are computed from the per-seed lists of the file, not read from its stored summaries.
    """
    source = path if path is not None else get_path("reports_dir", create=False) / AB_TEST_FILE
    try:
        ab = json.loads(Path(source).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    table = schema.WARD_DAILY
    if not isinstance(ab, dict) or (ab.get("table") or table) != table:
        return None
    summary = ab.get("summary") or {}
    previous, adopted = summary.get("previous_spec") or {}, summary.get("adopted_spec") or {}
    seeds = adopted.get("evaluation_seeds")
    n = len(seeds) if isinstance(seeds, list) else 0

    def numbers(values: Any) -> bool:
        return isinstance(values, list) and len(values) == n and all(is_number(v) for v in values)

    if n < 2 or not numbers(seeds) or previous.get("evaluation_seeds") != seeds:
        return {"stale": "the two specs were not scored on the same evaluation seeds"}
    for spec in (previous, adopted):
        if not all(numbers(spec.get(key)) for key in ("composite", "raw_hard_valid", "operational_checks_passed")):
            return {"stale": "the per-seed results are incomplete"}
    scored = adopted.get("operational_checks_scored")
    if not is_number(scored) or previous.get("operational_checks_scored") != scored:
        return {"stale": "the two specs were scored on different checks"}
    info = (comparison.get("tables") or {}).get(table) or {}
    official = (ab.get("official_run") or {}).get("evaluation_seed")
    run_seed = _run_config(info, {}, table).get("seed")
    if official not in seeds or official != run_seed:
        return {"stale": f"its official evaluation seed ({official}) is not this evaluation's seed ({run_seed})"}
    variants = [v for v in (ab.get("official_run") or {}).get("variants") or [] if isinstance(v, dict)]
    control = next((v for v in variants if v.get("role") == "control"), {})
    winner = next((v for v in variants if v.get("role") == "adopted"), {})
    best = info.get("best_model")
    if control.get("variant") != best:
        return {"stale": f"its control ({control.get('variant')}) is not the default model ({best})"}
    k = seeds.index(official)
    entry = next((e for e in info.get("models") or [] if e.get("model_id") == best), {})
    if not is_number(entry.get("composite")) or abs(entry["composite"] - adopted["composite"][k]) > AB_COMPOSITE_TOLERANCE:
        return {"stale": f"the adopted spec scored {adopted['composite'][k]} on seed {official}, the default model "
                         f"{entry.get('composite')}"}
    by_seed = (ab.get("robustness_screens") or {}).get("by_seed") or {}
    extra = [i for i in range(n) if i != k]

    def row(seed: Any, variant: Any) -> dict[str, Any]:
        rows = by_seed.get(str(seed)) if isinstance(by_seed.get(str(seed)), list) else []
        return next((r for r in rows if isinstance(r, dict) and r.get("variant") == variant), {})

    failing = [(seeds[i], row(seeds[i], winner.get("variant")).get("failed_checks") or [])
               for i in extra if adopted["operational_checks_passed"][i] < scored]
    adopted_rows = [winner] + [row(seeds[i], winner.get("variant")) for i in extra]
    others = []
    for variant in variants:
        name = variant.get("variant")
        if variant.get("role") in ("control", "adopted"):
            continue
        rescored = [row(seeds[i], name) for i in extra]
        if not all(rescored):
            continue
        clean = sum(1 for r in rescored if not r.get("failed_checks"))
        pairs = list(zip(adopted_rows, [variant] + rescored))
        higher = {key: sum(1 for a, o in pairs if is_number(a.get(key)) and is_number(o.get(key)) and a[key] > o[key])
                  for key in ("utility", "composite")}
        compared = sum(1 for a, o in pairs if is_number(a.get("utility")) and is_number(o.get("utility")))
        others.append({"variant": name, "all_checks_seeds": clean, "decision": variant.get("decision"),
                       "higher": higher, "compared": compared})
    drops = [100.0 * (p - a) for p, a in zip(previous["raw_hard_valid"], adopted["raw_hard_valid"])]
    return {
        "file": AB_TEST_FILE.as_posix(), "seeds": seeds, "official_seed": official, "scored": scored,
        "rows": (ab.get("official_run") or {}).get("rows"),
        "n_variants": sum(1 for v in variants if v.get("role") != "control"),
        "adopted": winner.get("variant"), "previous": previous, "adopted_spec": adopted,
        "drops": drops, "failing": failing, "others": others, "n_extra": len(extra),
    }


def _ab_lines(ab: dict[str, Any] | None) -> list[str]:
    """Section 3 block: the ward_daily A/B test behind the band-grouped specs (from its evidence file)."""
    if not ab:
        return []
    if "stale" in ab:
        return ["### ward_daily: A/B test of the band-grouped specs", "",
                f"_The A/B evidence file (`reports/{AB_TEST_FILE.as_posix()}`) does not describe the current ward_daily "
                f"default ({ab['stale']}), so it is not summarised here; see DESIGN.md §7.2._", ""]
    where = f"`reports/{ab['file']}`"
    previous, adopted, scored = ab["previous"], ab["adopted_spec"], ab["scored"]
    seeds = ab["seeds"]
    extra_seeds = [s for s in seeds if s != ab["official_seed"]]
    seed_text = ", ".join(str(s) for s in extra_seeds[:-1]) + (f" and {extra_seeds[-1]}" if extra_seeds else "")

    def spec_row(label: str, spec: dict[str, Any]) -> list[str]:
        composite = spec["composite"]
        return [label, f"{_span(composite, fmt_num)} (mean {fmt_num(sum(composite) / len(composite))})",
                f"{_span(spec['operational_checks_passed'], lambda v: str(int(v)))} of {int(scored)}",
                _span(spec["raw_hard_valid"], lambda v: f"{100.0 * v:.1f}") + " %"]

    lines = ["### ward_daily: A/B test of the band-grouped specs", "",
             f"The ward_daily `gc_tuned` spec above was adopted after an A/B test (evidence {where}; DESIGN.md §7.2): "
             f"{ab['n_variants']} spec variants and the previous spec (the control) were trained on the same split and "
             f"scored in one evaluation (seed {ab['official_seed']}"
             + (f", {fmt_int(ab['rows'])} rows per model" if is_number(ab.get("rows")) else "")
             + f"), and `{ab['adopted']}` was the only variant that met the pre-set adoption bar. The previous and the "
             f"adopted spec were then re-scored with evaluation seeds {seed_text} (the same trained models on new "
             f"evaluation samples); ranges over the {len(seeds)} seeds:", ""]
    lines += md_table(["spec", "composite", "operational checks passed", "raw hard-valid"],
                      [spec_row("previous (control)", previous), spec_row(f"adopted (`{ab['adopted']}`)", adopted)])
    lines.append("")
    drops = ab["drops"]
    if ab["failing"]:
        failed = "; ".join(f"seed {seed}: " + (", ".join(f"`{c}`" for c in checks) or "one check")
                           for seed, checks in ab["failing"])
        lines.append(f"- On {len(ab['failing'])} of the {ab['n_extra']} extra seeds the adopted spec fails a check "
                     f"that passes in this evaluation ({failed}).")
    else:
        lines.append(f"- The adopted spec passes every scored check on all {ab['n_extra']} extra seeds.")
    if drops and min(drops) > 0:
        lines.append(f"- Its raw hard-valid rate is {_span(drops, lambda v: f'{v:.1f}')} points lower than the previous "
                     f"spec's on each seed (mean {sum(drops) / len(drops):.1f}); rejection sampling still delivers only "
                     "hard-valid rows.")
    for other in ab["others"]:
        decision = str(other.get("decision") or "not adopted")
        decision = (f"rejected in the official run: {decision[len('rejected: '):]}" if decision.startswith("rejected: ")
                    else f"official run: {decision}")
        compared = other.get("compared") or 0
        versus = (f"; the adopted spec scored the higher utility on {other['higher']['utility']} and the higher "
                  f"composite on {other['higher']['composite']} of the {compared} seeds" if compared else "")
        lines.append(f"- `{other['variant']}` ({decision}) was re-scored too and passed every check on "
                     f"{other['all_checks_seeds']} of the {ab['n_extra']} extra seeds{versus}.")
    lines += ["- The specs were compared on this evaluation, whose utility and privacy families use the real holdout, so "
              "the adopted spec's utility and privacy figures carry some selection optimism; training-seed variation is "
              "not covered.", ""]
    return lines


def _ab_checks_text(ab: dict[str, Any] | None) -> str:
    """Summary addition: the adopted spec's check counts on the A/B test's evaluation seeds."""
    if not ab or "stale" in ab:
        return ""
    passed = ab["adopted_spec"]["operational_checks_passed"]
    seeds = [str(s) for s in ab["seeds"]]
    return (f" ({_span(passed, lambda v: str(int(v)))} of {int(ab['scored'])} on evaluation seeds "
            f"{', '.join(seeds[:-1])} and {seeds[-1]} of the A/B test, section 3)")


def _deep_model_note(comparison: dict[str, Any], model_results: dict[str, Any]) -> str | None:
    """Notes caveat: the deep models ran in one fixed, untuned configuration (read from the settings and cards)."""
    experiments = (get_settings().get("synthesis") or {}).get("experiments") or {}
    deep: list[str] = []
    subsampled: list[str] = []
    for table in schema.TABLES:
        info = (comparison.get("tables") or {}).get(table) or {}
        entries = [e for e in info.get("models") or [] if e.get("status", "ok") == "ok"]
        cards = _cards(info, model_results, table)
        rows = {mid: card.get("training_rows") for mid, card in cards.items() if is_number(card.get("training_rows"))}
        for entry in entries:
            model_id = entry.get("model_id")
            if (experiments.get(model_id) or {}).get("model") not in DEEP_MODELS:
                continue
            if model_id not in deep:
                deep.append(model_id)
            if rows.get(model_id) is not None and rows and rows[model_id] < max(rows.values()):
                text = f"{table} fitted on {fmt_int(rows[model_id])} of {fmt_int(max(rows.values()))} training rows"
                if text not in subsampled:
                    subsampled.append(text)
    if not deep:
        return None
    configs = [experiments.get(m) or {} for m in deep]
    names = sorted({DEEP_MODELS[c.get("model")] for c in configs})
    parts = []
    for key, label in (("epochs", "epochs"), ("batch_size", "batch size")):
        values = sorted({(c.get("params") or {}).get(key) for c in configs} - {None})
        if len(values) == 1:
            parts.append(f"{values[0]} {label}" if key == "epochs" else f"{label} {values[0]}")
    if all((c.get("params") or {}).get("enable_gpu") is False for c in configs):
        parts.append("CPU")
    if all(c.get("segment_by") is None for c in configs):
        parts.append("one global model per table" + (" without ordered encodings"
                                                     if all(c.get("ordered_encodings") is False for c in configs) else ""))
    setup = ", ".join(parts + subsampled)
    return (f"- {' and '.join(names)} ran in one fixed configuration without hyperparameter tuning (`config/settings.yaml`"
            + (f": {setup}" if setup else "") + "), while the GaussianCopula recipe was improved step by step and "
            "A/B-tested (section 3): the comparison shows which recipe worked best under this budget, not an upper bound "
            "for the deep models.")


def _sdmetrics_note(comparison: dict[str, Any], model_results: dict[str, Any]) -> str | None:
    """Notes line interpreting the sdmetrics privacy cross-check (only when a result carries it)."""
    def baseline(result: dict[str, Any] | None) -> Any:
        return (((result or {}).get("privacy") or {}).get("sdmetrics") or {}).get("DCRBaselineProtection", {}).get("score")

    if not any(is_number(baseline(r)) for r in model_results.values()):
        return None
    selected = []
    for table in schema.TABLES:
        best = ((comparison.get("tables") or {}).get(table) or {}).get("best_model")
        score = baseline(model_results.get(f"{table}__{best}"))
        if is_number(score):
            selected.append(f"{fmt_num(score)} for {table} `{best}`")
    return ("- The sdmetrics privacy cross-check (section 8) is not scored. `DCRBaselineProtection` compares the synthetic "
            "rows' median distance to the real rows with that of uniformly random rows, so data that follows the real "
            "distribution scores well below 1 even when no row is copied"
            + (f" ({', '.join(selected)})" if selected else "")
            + "; `DCROverfittingProtection` is 1 when synthetic rows are no closer to the training rows than to holdout "
            "rows. The scored distance checks compare synthetic rows with real holdout rows instead.")


def _passed(check: dict[str, Any] | None) -> str:
    if not check or check.get("passed") is None:
        return "-"
    return "pass" if check["passed"] else "FAIL"


def _ordered_models(info: dict[str, Any]) -> list[dict[str, Any]]:
    """Model entries: the selected default first, then by composite."""
    best = info.get("best_model")
    return sorted(info.get("models") or [], key=lambda e: (e.get("model_id") != best,
                                                         -(e.get("composite") if is_number(e.get("composite")) else -1.0)))


def _entry_flags(entry: dict[str, Any]) -> str:
    flags = []
    if entry.get("status", "ok") != "ok":
        flags.append(str(entry.get("status")))
    elif entry.get("complete") is False:
        flags.append("incomplete")
    if entry.get("comparable") is False:
        flags.append("other run configuration")
    return f" _({', '.join(flags)})_" if flags else ""


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def _comparison_figure(table: str, entries: list[dict[str, Any]], path: Path) -> Path | None:
    # references get their readable label on two lines, so neighbouring labels never run into each other
    models = [_label(e.get("model_id")).replace(" (", "\n(") if e.get("kind") else str(e.get("model_id"))
              for e in entries]

    def draw(fig: Any) -> None:
        ax = fig.add_subplot(1, 1, 1)
        series = {"composite": [e.get("composite") for e in entries]}
        for component in COMPONENTS:
            series[component] = [(e.get("components") or {}).get(component) for e in entries]
        grouped_bars(ax, models, series, ylabel="score")
        ax.tick_params(axis="x", labelsize=7)
        ax.set_title(f"{table}: composite and component scores (references first)")

    return draw_figure(path, draw, size=(max(9.0, 1.4 * len(entries)), 4.4))


def _coverage_figure(table: str, model_id: str, scenarios: list[dict[str, Any]], path: Path) -> Path | None:
    def draw(fig: Any) -> None:
        ax = fig.add_subplot(1, 1, 1)
        names = [s.get("name", "?") for s in scenarios]
        y = list(range(len(names)))
        coverage = [s.get("coverage") if is_number(s.get("coverage")) else 0.0 for s in scenarios]
        fidelity = [s.get("fidelity") if is_number(s.get("fidelity")) else 0.0 for s in scenarios]
        ax.barh([v - 0.2 for v in y], coverage, height=0.4, label="coverage", color=PALETTE[0])
        ax.barh([v + 0.2 for v in y], fidelity, height=0.4, label="fidelity", color=PALETTE[1])
        ax.set_yticks(y)
        ax.set_yticklabels(names, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlim(0, 1)
        ax.grid(axis="x", alpha=0.3)
        # above the bars (between title and axes): inside the axes the legend covered the last bars
        ax.legend(fontsize=8, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
        ax.set_title(f"{table} / {model_id}: scenario coverage and fidelity", pad=22)

    return draw_figure(path, draw, size=(8.0, 0.35 * max(len(scenarios), 4) + 1.2))


# ---------------------------------------------------------------------------
# 1. Run configuration
# ---------------------------------------------------------------------------
def _run_config(info: dict[str, Any], model_results: dict[str, Any], table: str) -> dict[str, Any]:
    if info.get("run_config"):
        return dict(info["run_config"])
    for entry in info.get("models") or []:
        run = (model_results.get(f"{table}__{entry.get('model_id')}") or {}).get("run")
        if run:
            return dict(run)
    return {}


def _cards(info: dict[str, Any], model_results: dict[str, Any], table: str) -> dict[str, dict[str, Any]]:
    cards = {}
    for entry in info.get("models") or []:
        model_id = entry.get("model_id")
        card = (model_results.get(f"{table}__{model_id}") or {}).get("model_card") or entry.get("model_card") or {}
        cards[model_id] = card
    return cards


def _configured_epochs() -> dict[str, Any]:
    """Experiment id -> epochs configured for a full run (``synthesis.experiments.<id>.params.epochs``)."""
    experiments = (get_settings().get("synthesis") or {}).get("experiments") or {}
    return {eid: (cfg.get("params") or {}).get("epochs") for eid, cfg in experiments.items()
            if (cfg.get("params") or {}).get("epochs") is not None}


def _method_paragraph(comparison: dict[str, Any]) -> str:
    """How models are scored and selected (opens section 1)."""
    weights = comparison.get("composite_weights") or {}
    return ("Every trained model generates validated data through the scenario engine and is scored on statistical "
            "fidelity (SDV QualityReport), operational validity (raw samples), operational patterns, utility (ML TSTR "
            "and optimisation decisions) and privacy. The composite is a weighted sum "
            f"({', '.join(f'{k} {v:.2f}' for k, v in weights.items()) or 'settings.evaluation.composite_weights'}); "
            "per table, the registry default used by the API is the model with the "
            f"{comparison.get('selection_rule') or 'best complete composite'}. Two reference pseudo-models scored "
            "with the same families anchor the scale: real holdout rows (ceiling) and a no-learning baseline that "
            "keeps only per-stratum marginals (floor).")


RUN_START_SLACK_SECONDS = 5.0
"""Tolerance when deciding whether an entry was evaluated by the run that wrote the comparison."""


def _parse_time(value: Any) -> datetime | None:
    """ISO timestamp -> aware datetime (UTC assumed without an offset); None when missing or invalid."""
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _run_time_line(comparison: dict[str, Any], references: dict[str, Any] | None = None) -> str | None:
    """Wall time of the evaluation run that wrote the comparison.

    ``run_seconds`` covers only that run. When the comparison also keeps model entries evaluated by
    earlier runs (for example after one retrained model was re-evaluated), the line names what the
    latest run evaluated, so its wall time is not read as the time of the whole comparison.
    """
    seconds = comparison.get("run_seconds")
    if not is_number(seconds):
        return None
    minutes = fmt_num(float(seconds) / 60.0, 1)
    plain = f"- Wall time of the evaluation run: {minutes} min."
    finished = _parse_time(comparison.get("generated_at"))
    if finished is None:
        return plain
    started = finished - timedelta(seconds=float(seconds) + RUN_START_SLACK_SECONDS)
    latest: list[str] = []
    earlier: list[dict[str, Any]] = []
    ref_tables: list[str] = []
    for table in schema.TABLES:
        info = (comparison.get("tables") or {}).get(table) or {}
        for entry in info.get("models") or []:
            when = _parse_time(entry.get("evaluated_at"))
            if when is None:
                continue
            if when >= started:
                latest.append(f"{table} `{entry.get('model_id')}`")
            else:
                earlier.append(entry)
        ref_times = [_parse_time((r or {}).get("evaluated_at")) for r in ((references or {}).get(table) or {}).values()]
        if any(t is not None and t >= started for t in ref_times):
            ref_tables.append(table)
    if not (latest and earlier):
        return plain
    refs = f" (with the {' and '.join(ref_tables)} references)" if ref_tables else ""
    same = (" with the same run configuration" if all(e.get("comparable") is not False for e in earlier) else "")
    return (f"- Wall time of the latest evaluation run: {minutes} min. That run evaluated {', '.join(latest)}{refs}; "
            f"the other {fmt_int(len(earlier))} model entries were kept from earlier runs{same} (per-model "
            "evaluation seconds: column *eval s* in section 2).")


def _run_configuration_section(comparison: dict[str, Any], model_results: dict[str, Any],
                               references: dict[str, Any] | None = None) -> list[str]:
    lines = ["## 1. Run configuration", "", _method_paragraph(comparison), ""]
    quick_configs: list[dict[str, Any]] = []
    quick_models: list[str] = []
    for table in schema.TABLES:
        info = (comparison.get("tables") or {}).get(table)
        if not info:
            continue
        config = _run_config(info, model_results, table)
        if not config:
            lines.append(f"- **{table}**: run configuration not recorded (results from an older evaluation run).")
            continue
        if config.get("quick"):
            quick_configs.append(config)
        scenarios = (f"{fmt_int(config.get('rows_per_scenario'))} rows per scenario" if config.get("scenarios")
                     else "scenario coverage disabled")
        budgets = ""
        if config.get("scenario_time_budget_seconds") is not None:
            budgets = (f"; time budgets: {fmt_num(config.get('sample_time_budget_seconds'), 0)} s per sample request, "
                       f"{fmt_num(config.get('scenario_time_budget_seconds'), 0)} s per scenario request "
                       f"(at most {fmt_int(config.get('scenario_max_attempts'))} rejection rounds), "
                       f"{fmt_num(config.get('coverage_time_budget_seconds'), 0)} s for all scenarios of a model")
        lines.append(f"- **{table}**: {fmt_int(config.get('rows'))} generated rows per model "
                     f"({'QUICK evaluation' if config.get('quick') else 'full evaluation'}), {scenarios}, "
                     f"sdmetrics cross-check {'on' if config.get('sdmetrics') else 'off'}, seed {config.get('seed', '-')}"
                     f"{budgets}.")
        calibration = info.get("noise_calibration") or {}
        if calibration:
            lines.append(f"  Operational thresholds calibrated on {fmt_int(calibration.get('reps'))} random re-splits of "
                         f"the real data at n = {fmt_int(calibration.get('n'))}; cross-validated share of real samples "
                         f"that pass every check: {fmt_pct(calibration.get('real_pass_rate_cv'), 0)}.")
        configured = _configured_epochs()
        for model_id, card in _cards(info, model_results, table).items():
            if card.get("quick"):
                epochs = ""
                if card.get("epochs"):
                    full = configured.get(model_id)
                    epochs = f" ({card['epochs']} epochs" + (f" instead of {full}" if full else "") + ")"
                quick_models.append(f"{table}/{model_id}{epochs}")
    run_time = _run_time_line(comparison, references)
    if run_time:
        lines.append(run_time)
    if quick_configs:
        rows = sorted({fmt_int(c.get("rows")) for c in quick_configs})
        per_scenario = sorted({fmt_int(c.get("rows_per_scenario")) for c in quick_configs})
        sdmetrics = "sdmetrics on" if any(c.get("sdmetrics") for c in quick_configs) else "no sdmetrics"
        lines.append(f"- **Caveat - quick evaluation:** {' / '.join(rows)} rows per model and {' / '.join(per_scenario)} "
                     f"rows per scenario, {sdmetrics}; sampling noise is larger than in a full run (the calibrated "
                     "thresholds account for it).")
    if quick_models:
        lines.append("- **Caveat - quick-trained models** (`02_train_synthesizers.py --quick`): " + ", ".join(quick_models)
                     + ". Quick mode only shortens training of the deep models: CTGAN/TVAE run fewer epochs than "
                     "configured for a full run in `config/settings.yaml`, so the GAN/VAE ranking is provisional "
                     "(GaussianCopula models keep their full marginal search).")
    lines.append("")
    return lines


# ---------------------------------------------------------------------------
# 2. Model comparison
# ---------------------------------------------------------------------------
def _comparison_row(entry: dict[str, Any], result: dict[str, Any], label: str) -> list[str]:
    generation = result.get("generation") or entry.get("generation") or {}
    coverage = _family(result, "scenario_coverage").get("score")
    if coverage is None:
        coverage = (entry.get("family_scores") or {}).get("scenario_coverage")
    components = entry.get("components") or {}
    return [label, fmt_num(entry.get("composite")), *[fmt_num(components.get(c)) for c in COMPONENTS],
            fmt_num(coverage), fmt_pct(generation.get("acceptance_rate")), fmt_num(entry.get("seconds"), 1)]


def _fmt_normalised(value: Any) -> str:
    """Normalised score for display (values below -1 shown as ``< -1``)."""
    if is_number(value) and value < -1.0:
        return "< -1"
    return fmt_num(value)


def _reference_entry(info: dict[str, Any], ref_id: str) -> dict[str, Any]:
    return next((r for r in info.get("references") or [] if r.get("model_id") == ref_id
                 and r.get("status", "ok") == "ok"), {})


def _floor_warning(info: dict[str, Any]) -> list[str]:
    """Explicit statement when the no-learning floor matches or beats the selected model."""
    floor = _reference_entry(info, NULL_REFERENCE)
    best = next((e for e in info.get("models") or [] if e.get("model_id") == info.get("best_model")), {})
    if not floor or not best or not is_number(floor.get("composite")) or not is_number(best.get("composite")):
        return []
    lines = []
    if floor["composite"] >= best["composite"]:
        lines.append(f"**The no-learning floor ({fmt_num(floor['composite'])}) scores at least as high as the selected "
                     f"model {best.get('model_id')} ({fmt_num(best['composite'])}).** The floor resamples real values "
                     "column by column inside each allocation stratum, so it reproduces per-stratum marginals exactly "
                     "but no within-row relationship; it is a diagnostic, not a candidate (it publishes real values and "
                     "needs the same rejection). See the quality gaps (section 7).")
    beaten = [c for c in FLOOR_COMPONENTS if is_number((floor.get("components") or {}).get(c))
              and is_number((best.get("components") or {}).get(c)) and floor["components"][c] > best["components"][c]]
    if beaten:
        lines.append(f"Components where the floor beats {best.get('model_id')}: "
                     + ", ".join(f"{c} ({fmt_num(floor['components'][c])} vs {fmt_num(best['components'][c])})"
                                 for c in beaten) + ".")
    return lines + [""] if lines else []


def _comparison_section(table: str, info: dict[str, Any], results: dict[str, Any], figure: Path | None,
                        report_dir: Path) -> list[str]:
    references = [r for r in info.get("references") or [] if r.get("status", "ok") == "ok"]
    rows = [_comparison_row(r, {}, f"_{_label(r.get('model_id'))}_") for r in references]
    for entry in _ordered_models(info):
        result = results.get(f"{table}__{entry.get('model_id')}") or {}
        marker = " **(best)**" if entry.get("model_id") == info.get("best_model") else ""
        rows.append(_comparison_row(entry, result, f"{entry.get('model_id')}{marker}{_entry_flags(entry)}"))
    lines = [f"### {table}", ""]
    lines += md_table(["model", "composite", *COMPONENTS, "scenario coverage", "generator acceptance", "eval s"], rows)
    lines.append("")
    normalised = [e for e in _ordered_models(info) if e.get("normalised")]
    if normalised and references:
        lines += ["Normalised scores, (model - floor) / (ceiling - floor): 0 = no better than the no-learning "
                  "baseline, 1 = as good as real holdout data; `-` where the two references are less than 0.05 apart "
                  "(the family cannot tell real data from the no-learning baseline), `< -1` far below the floor.", ""]
        lines += md_table(["model", "composite", *COMPONENTS],
                          [[e.get("model_id"), _fmt_normalised(e["normalised"].get("composite")),
                            *[_fmt_normalised(e["normalised"].get(c)) for c in COMPONENTS]] for e in normalised])
        lines.append("")
    lines += _floor_warning(info)
    failed = [e for e in info.get("models") or [] if e.get("status", "ok") != "ok"]
    for entry in failed:
        lines.append(f"- {entry.get('model_id')}: evaluation failed in this run ({entry.get('error', 'no detail')}); "
                     "it is never selected as the default.")
    if failed:
        lines.append("")
    lines += figure_line(figure, report_dir, f"{table}: model comparison")
    return lines


# ---------------------------------------------------------------------------
# 3. Improvements made to SDV generation (ablation)
# ---------------------------------------------------------------------------
def _component_deltas(before: dict[str, Any], after: dict[str, Any]) -> str:
    parts = []
    for component in COMPONENTS:
        a, b = (before.get("components") or {}).get(component), (after.get("components") or {}).get(component)
        if is_number(a) and is_number(b) and abs(b - a) >= 0.02:
            parts.append(f"{component} {fmt_num(a)} -> {fmt_num(b)}")
    return ", ".join(parts) or "no component moved by 0.02 or more"


def _declared_groups(table: str) -> set[str] | None:
    """Constraint groups declared for ``table`` in ``CONSTRAINT_SPECS`` (``None`` if unavailable)."""
    try:  # importing the constraints module loads SDV (a few seconds)
        from bedsynth.synthesis.constraints import CONSTRAINT_SPECS
    except Exception:  # noqa: BLE001 - fall back to the enabled groups
        return None
    return {spec.kind for spec in CONSTRAINT_SPECS.get(table, ())}


def _registry_card(table: str, model_id: str | None) -> dict[str, Any] | None:
    """Registry model card (JSON only, no pickle is loaded); ``None`` when not registered."""
    if not model_id:
        return None
    try:
        from bedsynth.synthesis.registry import ModelRegistry

        return ModelRegistry().get_card(table, model_id)
    except Exception:  # noqa: BLE001 - optional source
        return None


def _applied_groups(table: str, entry: dict[str, Any], result: dict[str, Any]) -> tuple[list[str] | None, bool]:
    """Constraint groups a model applies inside SDV and whether they are only the declared fallback.

    Order of the sources: the result file's ``model_card.effective_constraint_groups``, the registry
    card's (when it is the evaluated fit), then the enabled groups declared for the table.
    """
    card = result.get("model_card") or entry.get("model_card") or {}
    if isinstance(card.get("effective_constraint_groups"), list):
        return list(card["effective_constraint_groups"]), False
    registry = _registry_card(table, entry.get("model_id"))
    if registry and isinstance(registry.get("effective_constraint_groups"), list):
        evaluated, registered = card.get("fitted_at") or entry.get("model_fitted_at"), registry.get("fitted_at")
        if not evaluated or not registered or evaluated == registered:
            return list(registry["effective_constraint_groups"]), False
    enabled = card.get("constraint_groups")
    if not isinstance(enabled, list):
        return None, False
    declared = _declared_groups(table)
    return [g for g in enabled if declared is None or g in declared], True


def _ablation_table(table: str, info: dict[str, Any], results: dict[str, Any]) -> list[str]:
    entries = {e.get("model_id"): e for e in info.get("models") or [] if e.get("status", "ok") == "ok"}
    steps = [m for m in ABLATION if m in entries]
    lines = [f"### {table}", ""]
    if len(steps) < 2:
        lines += [f"_Ablation not available: {', '.join(steps) or 'none'} of {', '.join(ABLATION)} evaluated._", ""]
        return lines
    missing = [m for m in ABLATION if m not in entries]
    rows, previous, fallback = [], None, False
    for model_id in steps:
        entry, result = entries[model_id], results.get(f"{table}__{model_id}") or {}
        card = result.get("model_card") or entry.get("model_card") or {}
        groups, declared_only = _applied_groups(table, entry, result)
        fallback = fallback or declared_only
        groups_text = (", ".join(groups) if groups else "none") if groups is not None else "-"
        components = entry.get("components") or {}
        delta = None if previous is None else (entry.get("composite") or 0.0) - (previous.get("composite") or 0.0)
        rows.append([model_id, card.get("segment_label") or "-", groups_text + (" \\*" if declared_only else ""),
                     fmt_pct(card.get("raw_hard_valid_rate")),
                     fmt_pct(_family(result, "validity").get("score", components.get("validity"))),
                     fmt_num(entry.get("composite")), fmt_num(components.get("fidelity")),
                     fmt_num(components.get("operational")), fmt_num(components.get("utility")),
                     "-" if delta is None else f"{delta:+.3f}"])
        previous = entry
    lines += md_table(["model", "segmentation", "constraint groups applied in SDV", "raw hard-valid after training",
                       "raw hard-valid (this evaluation)", "composite", "fidelity", "operational", "utility",
                       "delta composite"], rows)
    lines.append("")
    if fallback:
        lines += ["\\* The result file and the registry do not record which groups this fit applied: listed are the "
                  "groups it enables that are declared for the table.", ""]
    for before, after in zip(steps, steps[1:]):
        lines.append(f"- **{before} -> {after}** ({ABLATION_STEPS[after]}): composite "
                     f"{fmt_num(entries[before].get('composite'))} -> {fmt_num(entries[after].get('composite'))}; "
                     f"{_component_deltas(entries[before], entries[after])}.")
    if missing:
        lines.append(f"- {', '.join(missing)} has no result in this comparison: the deltas skip that step.")
    lines.append("")
    return lines


def _improvement_log() -> list[str]:
    """Static log of the improvements behind the selected models, with DESIGN.md cross-references."""
    rules = {rule["rule_id"]: rule for table in schema.TABLES for rule in validation.rules_catalog(table)}

    def wording(rule_id: str) -> str:
        text = str(rules[rule_id].get("description") or rules[rule_id].get("name") or "").rstrip(".")
        return f"{rule_id} ({text[:1].lower() + text[1:]})" if text else rule_id

    review = "; ".join(wording(rule_id) if rule_id in rules else rule_id for rule_id in REVIEW_RULES)
    lines = ["### Improvement log", "",
             "What changed on the way to the selected models and where each change is documented (DESIGN.md is the "
             "specification; the measured effects are the ablation above and the A/B results in DESIGN.md):", ""]
    for number, (title, text, where) in enumerate(IMPROVEMENT_LOG, start=1):
        lines.append(f"{number}. **{title}** - {text.format(review_rules=review)} ({where}).")
    return lines + [""]


def _ablation_section(comparison: dict[str, Any], results: dict[str, Any],
                      ab: dict[str, Any] | None = None) -> list[str]:
    lines = ["## 3. Improvements made to SDV generation", "",
             "The ablation trains the same GaussianCopula family with one improvement block added per step "
             "(settings `synthesis.experiments`), so the deltas below are the measured effect of each block on "
             "the same evaluation:", ""]
    lines += [f"- `{model_id}` - {text}." for model_id, text in ABLATION_STEPS.items()]
    lines += ["", "*Raw hard-valid after training* is the Week-2 diagnostic (unfiltered samples drawn right after "
              "training); *raw hard-valid (this evaluation)* re-measures it on this evaluation's own unfiltered sample "
              "(size in section 1), so the two differ by sampling noise.", ""]
    for table in schema.TABLES:
        info = (comparison.get("tables") or {}).get(table)
        if info:
            lines += _ablation_table(table, info, results)
    lines += [
        "Generation-level improvements shared by every model (not part of the ablation): rejection sampling "
        "against every hard rule (the delivered data is 100 % hard-valid), allocation strata that keep the "
        "ward x band / ward x source x delay mix of the training split despite unequal rejection rates, "
        "acceptance-informed batch sizes and per-request time budgets.",
        "",
    ]
    return lines + _ab_lines(ab) + _improvement_log()


# ---------------------------------------------------------------------------
# 4. Validity: raw model samples vs delivered output
# ---------------------------------------------------------------------------
def _top_hard_rules(result: dict[str, Any]) -> str:
    worst = _family(result, "validity").get("worst_hard_rules") or []
    return ", ".join(f"{r.get('rule_id')} {fmt_pct(r.get('violation_rate'))}" for r in worst) or "none"


def _soft_rule_rows(table: str, info: dict[str, Any], results: dict[str, Any],
                    references: dict[str, Any]) -> tuple[list[str], list[list[str]]]:
    real = references.get(REAL_REFERENCE) or {}
    rules = _family(real, "validity").get("soft_rules") or []
    models = [e.get("model_id") for e in _ordered_models(info) if f"{table}__{e.get('model_id')}" in results]
    if not rules:
        first = next((results[f"{table}__{m}"] for m in models if _family(results[f"{table}__{m}"], "output_validity")), {})
        rules = _family(first, "output_validity").get("soft_rules") or []
    real_rates = _family(real, "validity").get("soft_violation_rates") or {}
    rows = []
    for rule in rules:
        rule_id = rule.get("rule_id")
        rates = []
        for model_id in models:
            output = _family(results[f"{table}__{model_id}"], "output_validity")
            rates.append(fmt_pct((output.get("soft_violation_rates") or {}).get(rule_id)))
        rows.append([rule_id, rule.get("description") or rule.get("name") or "-", fmt_pct(real_rates.get(rule_id)), *rates])
    return ["rule", "description", "real holdout", *[f"{m} output" for m in models]], rows


def _strata_flag() -> float:
    """Margin above the rounding floor at which a strata TVD is flagged (``evaluation.strata_tvd_flag``)."""
    value = (get_settings().get("evaluation") or {}).get("strata_tvd_flag")
    return float(value) if is_number(value) else 0.05


def _validity_section(comparison: dict[str, Any], results: dict[str, Any], references: dict[str, Any]) -> list[str]:
    lines = ["## 4. Validity: raw model samples vs delivered output", "",
             "- **Raw hard-valid rate** (composite component `validity`): share of *unfiltered* model samples that "
             "pass every hard operational rule - what the model produces on its own.",
             "- **Output hard-valid rate**: share of the *delivered* generator output passing every hard rule; 100 % "
             "by construction, because rejection sampling discards every invalid row.",
             "- **Generator acceptance**: accepted / sampled rows during generation. It is measured on the "
             "generator's conditional per-stratum requests, not on the unconditional raw sample, and also counts "
             "rows rejected only for leaving the requested stratum, so it can sit above or below the raw rate.",
             "- **Strata TVD**: distance between the output's allocation-strata mix and the training split's; more "
             f"than {fmt_num(_strata_flag(), 2)} above the **rounding floor** (the TVD of the best possible allocation "
             "of that many rows) the generator had to move quota to strata the model produces more easily (flagged). "
             "**Rebalanced rows** is the generator's count of delivered rows that sit in a stratum beyond its "
             "reference quota (without a shortfall, rebalanced rows / rows is the TVD from the allocation; the TVD "
             "column also includes rounding and the chunking of large samples).", ""]
    for table in schema.TABLES:
        info = (comparison.get("tables") or {}).get(table)
        if not info:
            continue
        rows = []
        for entry in _ordered_models(info):
            result = results.get(f"{table}__{entry.get('model_id')}") or {}
            if not result:
                continue
            generation = result.get("generation") or {}
            strata = generation.get("allocation_strata") or {}
            tvd = generation.get("strata_mix_tvd")
            rows.append([entry.get("model_id"), fmt_pct(_family(result, "validity").get("score")),
                         fmt_pct(_family(result, "output_validity").get("score", 1.0 if result else None)),
                         fmt_pct(generation.get("acceptance_rate")),
                         fmt_num(tvd) + (" (flagged)" if generation.get("rebalancing_flag") else ""),
                         fmt_num(generation.get("strata_rounding_tvd")),
                         fmt_int(strata.get("rebalanced_rows")), _top_hard_rules(result)])
        lines += [f"### {table}", ""]
        lines += md_table(["model", "raw hard-valid", "output hard-valid", "generator acceptance", "strata TVD",
                           "rounding floor", "rebalanced rows", "most violated hard rules (raw)"], rows)
        lines.append("")
        header, soft_rows = _soft_rule_rows(table, info, results, references.get(table) or {})
        if soft_rows:
            lines += ["Soft (plausibility) rules - violation rate of the delivered output vs real holdout data:", ""]
            lines += md_table(header, soft_rows)
            lines.append("")
    return lines


# ---------------------------------------------------------------------------
# 5. Scheduling relationships and occupancy patterns
# ---------------------------------------------------------------------------
def _check_note(check: dict[str, Any]) -> str:
    return (check.get("details") or {}).get("note") or "statistic undefined"


def _check_value(check: dict[str, Any]) -> str:
    """A check's value, or ``n/a (<reason>)`` when it could not be computed."""
    return fmt_num(check.get("value")) if check.get("value") is not None else f"n/a ({_check_note(check)})"


def _undefined_checks(operational: dict[str, Any]) -> list[dict[str, Any]]:
    """Checks left out of the score because their statistic is undefined (``passed`` is None)."""
    return [c for c in operational.get("checks") or [] if c.get("passed") is None and not c.get("strata_forced")]


def _check_cell(check: dict[str, Any] | None) -> str:
    """Model cell of a pattern table: value and verdict, or why the value is undefined."""
    if not check:
        return "-"
    verdict = "not scored" if check.get("passed") is None else _passed(check)
    return f"{_check_value(check)} {verdict}"


def _section_checks(results: Iterable[dict[str, Any]], scheduling: bool) -> list[str]:
    """Scored check names of the scheduling group (``scheduling=True``) or of every other group."""
    names: list[str] = []
    for result in results:
        for check in _family(result, "operational").get("checks") or []:
            name = check.get("name")
            is_scheduling = check.get("group") == "scheduling" or name in SCHEDULING_EXTRA
            if is_scheduling == scheduling and not check.get("strata_forced") and name not in names:
                names.append(name)
    return names


def _patterns_section(comparison: dict[str, Any], results: dict[str, Any], references: dict[str, Any]) -> list[str]:
    """Occupancy, capacity and patient-flow checks and scheduling checks, per model and reference."""
    lines = ["## 5. Scheduling relationships and occupancy patterns", "",
             "Operational checks per model (lower values are better) with pass/FAIL against the pass threshold, "
             "which is calibrated to real-data noise at the evaluated size (the larger of the design threshold and "
             "the real-data noise limit), next to the noise limit and both references. System checks forced by the "
             "allocation strata are left out (section 9); `n/a` marks a statistic that could not be computed on the "
             "model's output, with the reason.", ""]
    for table in schema.TABLES:
        info = (comparison.get("tables") or {}).get(table)
        if not info:
            continue
        models = [e.get("model_id") for e in _ordered_models(info) if f"{table}__{e.get('model_id')}" in results]
        refs = references.get(table) or {}
        sources = [results[f"{table}__{m}"] for m in models] + list(refs.values())
        checks = {m: _check_map(results[f"{table}__{m}"]) for m in models}
        real_checks, null_checks = _check_map(refs.get(REAL_REFERENCE)), _check_map(refs.get(NULL_REFERENCE))
        blocks = [(label, _section_checks(sources, scheduling))
                  for label, scheduling in (("Occupancy, capacity and patient-flow checks", False), ("Scheduling checks", True))]
        if not any(names for _, names in blocks):
            continue
        lines += [f"### {table}", ""]
        for label, names in blocks:
            if not names:
                continue
            rows = []
            for name in names:
                any_check = next((c[name] for c in checks.values() if name in c), {}) or real_checks.get(name) or {}
                rows.append([name, any_check.get("metric") or "-", fmt_num(any_check.get("threshold_used")),
                             fmt_num((any_check.get("real_reference") or {}).get("noise_limit")),
                             fmt_num((real_checks.get(name) or {}).get("value")),
                             fmt_num((null_checks.get(name) or {}).get("value")),
                             *[_check_cell(checks[m].get(name)) for m in models]])
            lines += [f"**{label}**", ""]
            lines += md_table(["check", "metric", "pass threshold", "real noise limit", "real holdout", "floor", *models],
                              rows)
            lines.append("")
            descriptions = {n: (next((c[n] for c in checks.values() if n in c), {}) or real_checks.get(n) or {})
                            .get("description") for n in names}
            lines += [f"- `{n}`: {d}" for n, d in descriptions.items() if d]
            lines.append("")
    return lines


# ---------------------------------------------------------------------------
# 6. Optimisation-model testing
# ---------------------------------------------------------------------------
def _policy_block(optimisation: dict[str, Any]) -> dict[str, Any]:
    return dict(optimisation.get("policy_evaluation") or optimisation.get("policy_ranking") or {})


def _optimisation_row(label: str, result: dict[str, Any]) -> list[str] | None:
    optimisation = _family(result, "optimization")
    if not optimisation.get("components"):
        return None
    quota, policy = optimisation.get("booking_quota") or {}, _policy_block(optimisation)
    capacity = optimisation.get("capacity_planning") or {}
    return [label, fmt_num(optimisation.get("score")), fmt_num(quota.get("score")), fmt_pct(quota.get("relative_regret")),
            fmt_pct(quota.get("real_decision_regret_vs_oracle")), fmt_pct(quota.get("quota_agreement"), 0),
            fmt_num(policy.get("objective_accuracy", policy.get("score"))), fmt_num(policy.get("kendall_tau")),
            fmt_num(capacity.get("score")), fmt_pct(capacity.get("mean_relative_error"))]


def _ranking_note(refs: dict[str, Any]) -> str:
    """Why Kendall tau is only a diagnostic, with the no-learning floor's own tau as evidence."""
    floor_tau = _policy_block(_family(refs.get(NULL_REFERENCE), "optimization")).get("kendall_tau")
    return ("- The real policy ranking is the monotone order of the occupancy targets, so ranking agreement "
            "(Kendall tau) is reported as a diagnostic only"
            + (f": even the no-learning floor reaches tau {fmt_num(floor_tau)}." if is_number(floor_tau)
               else ": almost any data reproduces it."))


def _optimisation_section(comparison: dict[str, Any], results: dict[str, Any], references: dict[str, Any]) -> list[str]:
    info = (comparison.get("tables") or {}).get(schema.WARD_DAILY)
    lines = ["## 6. Optimisation-model testing", ""]
    if not info:
        return lines + ["_No ward_daily results._", ""]
    refs = references.get(schema.WARD_DAILY) or {}
    rows = [row for ref_id in (REAL_REFERENCE, NULL_REFERENCE)
            if (row := _optimisation_row(f"_{_label(ref_id)}_", refs.get(ref_id) or {})) is not None]
    for entry in _ordered_models(info):
        row = _optimisation_row(entry.get("model_id"), results.get(f"{schema.WARD_DAILY}__{entry.get('model_id')}") or {})
        if row is not None:
            rows.append(row)
    if not rows:
        return lines + ["_No optimisation results._", ""]
    lines += ["Three decision problems are solved on synthetic ward-days and compared with the same decisions on "
              "real data (DESIGN.md §9.7): elective booking quotas (newsvendor / sample average approximation; the "
              "synthetic and the real-trained quotas are both scored **out of sample on the real holdout**), "
              "admission-control policy evaluation (can the data predict each policy's objective?) and capacity "
              "planning (p95 beds needed per ward).", ""]
    lines += md_table(["data", "score", "quota score", "quota regret (holdout)", "real-vs-real regret floor",
                       "quota agreement", "policy objective accuracy", "Kendall tau (diagnostic)", "capacity score",
                       "capacity error"], rows)
    lines.append("")
    best = results.get(f"{schema.WARD_DAILY}__{info.get('best_model')}") or {}
    policy = _policy_block(_family(best, "optimization"))
    if policy.get("real_ranking_monotone"):
        lines.append(_ranking_note(refs))
    lines.append("- Regret below the real-vs-real floor means the synthetic decision is as good as one taken on real "
                 "data of the same size; a negative regret means it did better on the holdout than the real-trained "
                 "decision (sampling noise).")
    lines.append("")
    return lines


# ---------------------------------------------------------------------------
# 7. Quality gaps
# ---------------------------------------------------------------------------
def _gap_lines(entry: dict[str, Any], result: dict[str, Any], references: dict[str, Any]) -> list[str]:
    """Quality-gap bullets of one model (failed checks, soft rules, floor, columns, scenarios, generation)."""
    lines = []
    operational = _family(result, "operational")
    failed = [c for c in operational.get("checks") or [] if c.get("passed") is False and not c.get("strata_forced")]
    if failed:
        parts = [f"`{c['name']}` {_check_value(c)} (threshold {fmt_num(c.get('threshold_used'))}"
                 + (", beyond real noise" if c.get("within_real_noise") is False else "") + ")" for c in failed]
        lines.append(f"- Failed operational checks: {'; '.join(parts)}.")
    undefined = _undefined_checks(operational)
    if undefined:
        lines.append("- Checks undefined on this model's output (not scored): "
                     + "; ".join(f"`{c['name']}` ({_check_note(c)})" for c in undefined) + ".")
    real_rates = _family(references.get(REAL_REFERENCE), "validity").get("soft_violation_rates") or {}
    output_rates = _family(result, "output_validity").get("soft_violation_rates") or {}
    soft_gaps = [f"{rule} {fmt_pct(rate)} vs {fmt_pct(real_rates.get(rule))} real"
                 for rule, rate in output_rates.items()
                 if is_number(rate) and is_number(real_rates.get(rule)) and rate > real_rates[rule] + SOFT_GAP_MARGIN]
    if soft_gaps:
        lines.append(f"- Soft rules violated more often than in real data: {', '.join(soft_gaps)}.")
    floor_components = (references.get(NULL_REFERENCE) or {}).get("composite", {}).get("components") or {}
    model_components = entry.get("components") or {}
    normalised = entry.get("normalised") or {}
    weak = [c for c in FLOOR_COMPONENTS if is_number(floor_components.get(c)) and is_number(model_components.get(c))
            and (model_components[c] <= floor_components[c]
                 or (is_number(normalised.get(c)) and normalised[c] <= LOW_NORMALISED))]
    if weak:
        lines.append("- At or near the no-learning floor on: " + ", ".join(
            f"{c} ({fmt_num(model_components[c])} vs floor {fmt_num(floor_components[c])})" for c in weak) + ".")
    summary = _family(result, "statistical").get("summary") or {}
    if summary.get("worst_numeric_columns"):
        lines.append(f"- Worst column shapes (KS): {', '.join(summary['worst_numeric_columns'][:5])}.")
    pairs = ((_family(result, "statistical").get("sdv") or {}).get("quality") or {}).get("worst_column_pairs") or []
    if pairs:
        lines.append("- Worst column pairs (SDV): " + ", ".join(
            f"{' x '.join(p.get('columns') or [])} {fmt_num(p.get('score'))}" for p in pairs[:3]) + ".")
    view = result.get("model_view") or {}
    if view:
        stat_raw = (view.get("statistical") or {}).get("score")
        op_raw = (view.get("operational") or {}).get("score")
        lines.append(f"- Model vs system view: fidelity {fmt_num(stat_raw)} on raw samples vs "
                     f"{fmt_num(_family(result, 'statistical').get('score'))} delivered; operational {fmt_num(op_raw)} raw vs "
                     f"{fmt_num(operational.get('score'))} delivered (the difference is what rejection and allocation strata add).")
    coverage = _family(result, "scenario_coverage")
    issues = []
    for key, text in (("rebalanced_scenarios", "strata mix off the real scenario subset beyond rounding (quota moved "
                                               "to easier strata, or the model places filtered rows on other strata)"),
                      ("budget_stopped_scenarios", "stopped by the time budget"),
                      ("throttled_scenarios", "rounds shrunk to fit the time budget (speed-dependent)"),
                      ("not_evaluated_scenarios", "not evaluated (coverage budget)"), ("failed_scenarios", "failed")):
        if coverage.get(key):
            issues.append(f"{text}: {', '.join(coverage[key])}")
    low = [s.get("name") for s in coverage.get("scenarios") or [] if is_number(s.get("fidelity")) and s["fidelity"] < 0.85]
    if low:
        issues.append(f"scenario fidelity below 0.85: {', '.join(low)}")
    if issues:
        lines.append(f"- Scenarios - {'; '.join(issues)}.")
    generation = result.get("generation") or {}
    if generation.get("stopped_reasons") or generation.get("shortfall"):
        lines.append(f"- The main sample was cut by the time budget: {fmt_int(generation.get('shortfall'))} rows short "
                     f"({'; '.join(generation.get('stopped_reasons') or [])}).")
    elif generation.get("throttled_rounds"):
        lines.append(f"- {fmt_int(generation['throttled_rounds'])} generation round(s) were shrunk to fit the time "
                     "budget: the sample depends on machine speed.")
    if generation.get("rebalancing_flag"):
        lines.append(f"- The main sample's strata mix drifted from the training split (TVD "
                     f"{fmt_num(generation.get('strata_mix_tvd'))}): the generator moved quota to easier strata.")
    return lines or ["- No gap beyond real-data noise was detected by the evaluation families."]


def _quality_gaps_section(comparison: dict[str, Any], results: dict[str, Any], references: dict[str, Any]) -> list[str]:
    lines = ["## 7. Quality gaps", "",
             "Gaps of each table's selected model (failed checks use the thresholds calibrated to real-data noise; "
             f"soft-rule gaps are rates more than {SOFT_GAP_MARGIN * 100:g} percentage points above the real "
             "holdout's).", ""]
    for table in schema.TABLES:
        info = (comparison.get("tables") or {}).get(table)
        if not info or not info.get("best_model"):
            continue
        best = info["best_model"]
        entry = next((e for e in info.get("models") or [] if e.get("model_id") == best), {})
        result = results.get(f"{table}__{best}")
        if not result:
            continue
        lines += [f"### {table} / {best}", ""] + _gap_lines(entry, result, references.get(table) or {}) + [""]
        others = [e.get("model_id") for e in _ordered_models(info)
                  if e.get("model_id") != best and e.get("status", "ok") == "ok"]
        summaries = [_failed_summary(model_id, results.get(f"{table}__{model_id}")) for model_id in others]
        if summaries:
            lines += ["Other models: " + "; ".join(summaries) + ".", ""]
    return lines


def _failed_summary(model_id: str, result: dict[str, Any] | None) -> str:
    """``"<model> failed k of n checks"`` for the quality-gap overview."""
    operational = _family(result, "operational")
    failed = operational.get("failed_checks") or []
    text = f"{model_id} failed {len(failed)} of {fmt_int(operational.get('n_checks'))} operational checks"
    generation = (result or {}).get("generation") or {}
    if is_number(generation.get("shortfall")) and generation["shortfall"] > 0:
        text += (f" (delivered {fmt_int(generation.get('generated'))} of {fmt_int(generation.get('requested'))} "
                 "rows within its time budget)")
    return text


# ---------------------------------------------------------------------------
# 8. Model details
# ---------------------------------------------------------------------------
def _statistical_lines(result: dict[str, Any]) -> list[str]:
    stat = result.get("statistical") or {}
    summary = stat.get("summary") or {}
    properties = ((stat.get("sdv") or {}).get("quality") or {}).get("properties") or {}
    return [
        f"- **Statistical fidelity** {fmt_num(stat.get('score'))} ({stat.get('score_source', '-')}): "
        f"column shapes {fmt_num(properties.get('Column Shapes', summary.get('column_shapes_score')))}, "
        f"pair trends {fmt_num(properties.get('Column Pair Trends', summary.get('pair_trends_score')))}; "
        f"mean KS {fmt_num(summary.get('mean_ks_statistic'))}, mean TVD {fmt_num(summary.get('mean_tv_distance'))}, "
        f"mean |dSpearman| {fmt_num(summary.get('spearman_mean_abs_diff'))}; "
        f"worst numeric columns: {', '.join(summary.get('worst_numeric_columns') or []) or '-'}.",
    ]


def _validity_lines(result: dict[str, Any]) -> list[str]:
    validity = result.get("validity") or {}
    worst = ", ".join(f"{r.get('rule_id')} {r.get('name')} ({fmt_pct(r.get('violation_rate'))})"
                      for r in validity.get("worst_hard_rules") or []) or "none"
    output = result.get("output_validity") or {}
    soft = f"; delivered output soft-clean rate {fmt_pct(output.get('soft_clean_rate'))}" if output else ""
    return [f"- **Validity** (raw model samples, before rejection) {fmt_pct(validity.get('score'))} hard-valid; "
            f"most violated: {worst}. Generator output is 100 % hard-valid by construction{soft}."]


def _utility_lines(result: dict[str, Any]) -> list[str]:
    ml = result.get("ml_utility") or {}
    lines = [f"- **ML utility (TSTR vs size-matched TRTR)** {fmt_num(ml.get('score'))}:"]
    for task in ml.get("tasks") or []:
        trtr = ", ".join(f"{k} {fmt_num(v)}" for k, v in (task.get("trtr") or {}).items()
                         if k in ("auc", "average_precision", "f1", "r2", "mae"))
        tstr = ", ".join(f"{k} {fmt_num(v)}" for k, v in (task.get("tstr") or {}).items()
                         if k in ("auc", "average_precision", "f1", "r2", "mae"))
        ratios = ", ".join(f"{k} {fmt_num(v)}" for k, v in (task.get("ratios") or {}).items())
        flag = "" if task.get("informative") else " (not informative, excluded)"
        lines.append(f"  - {task.get('name')}: TRTR {trtr or '-'}; TSTR {tstr or '-'}; "
                     f"{('ratios ' + ratios + '; ') if ratios else ''}score {fmt_num(task.get('score'))}{flag}")
    optimisation = result.get("optimization") or {}
    if optimisation.get("components"):
        quota = optimisation.get("booking_quota") or {}
        policy = _policy_block(optimisation)
        capacity = optimisation.get("capacity_planning") or {}
        label = "policy evaluation" if "policy_evaluation" in optimisation else "policy ranking"
        lines.append(
            f"- **Optimisation utility** {fmt_num(optimisation.get('score'))}: booking quotas "
            f"{fmt_num(quota.get('score'))} (quota agreement {fmt_pct(quota.get('quota_agreement'))}, relative regret "
            f"{fmt_pct(quota.get('relative_regret'))} {('out of sample' if quota.get('regret_basis') == 'out_of_sample_holdout' else 'on real data')}); "
            f"{label} {fmt_num(policy.get('score'))} (Kendall tau {fmt_num(policy.get('kendall_tau'))}); "
            f"capacity planning {fmt_num(capacity.get('score'))} (mean relative error {fmt_pct(capacity.get('mean_relative_error'))}).")
    return lines


def _privacy_lines(result: dict[str, Any]) -> list[str]:
    privacy = result.get("privacy") or {}
    components = ", ".join(f"{k.replace('_', ' ')} {fmt_num(v)}" for k, v in (privacy.get("components") or {}).items())
    exact = privacy.get("exact_matches") or {}
    dcr = privacy.get("dcr") or {}
    disclosure = privacy.get("aggregate_disclosure") or {}
    sdmetrics = privacy.get("sdmetrics") or {}
    cross = ""
    if sdmetrics.get("ran"):
        baseline = sdmetrics.get("DCRBaselineProtection") or {}
        overfit = sdmetrics.get("DCROverfittingProtection") or {}
        medians = baseline.get("median_DCR_to_real_data") or {}
        parts = []
        if is_number(baseline.get("score")):
            parts.append(f"DCRBaselineProtection {fmt_num(baseline['score'])}"
                         + (f" (median DCR {fmt_num(medians['synthetic_data'])} vs "
                            f"{fmt_num(medians['random_data_baseline'])} for random rows)"
                            if is_number(medians.get("synthetic_data")) and is_number(medians.get("random_data_baseline"))
                            else ""))
        if is_number(overfit.get("score")):
            parts.append(f"DCROverfittingProtection {fmt_num(overfit['score'])}")
        if parts:
            rows = f", {fmt_int(sdmetrics['rows'])} rows per side" if is_number(sdmetrics.get("rows")) else ""
            cross = f" sdmetrics cross-check (not scored{rows}): {', '.join(parts)}."
    aggregate = ""
    if disclosure.get("synthetic_vs_train_tvd") is not None:
        aggregate = (f" Aggregate disclosure (not scored): the {' x '.join(disclosure.get('columns') or [])} mix is "
                     f"{fmt_num(disclosure.get('synthetic_vs_train_tvd'))} TVD from the training split vs "
                     f"{fmt_num(disclosure.get('holdout_vs_train_tvd'))} for real holdout data"
                     + (" - the output mirrors the training frequencies." if disclosure.get("mirrors_training_frequencies")
                        else "."))
    return [f"- **Privacy** {fmt_num(privacy.get('score'))} ({components}); exact copies of training rows "
            f"{fmt_pct(exact.get('synthetic_rate'), 2)} (holdout baseline {fmt_pct(exact.get('holdout_rate'), 2)}), "
            f"median DCR ratio synthetic/holdout {fmt_num(dcr.get('median_ratio'))}.{cross}{aggregate}"]


def _operational_lines(result: dict[str, Any]) -> list[str]:
    operational = result.get("operational") or {}
    system = operational.get("system_checks") or []
    lines = [f"- **Operational patterns** {fmt_num(operational.get('score'))}: "
             f"{fmt_int(operational.get('n_passed'))} of {fmt_int(operational.get('n_checks'))} checks passed"
             + (f"; system checks (forced by the allocation strata, not scored): {', '.join(system)}" if system else "")
             + ".", ""]
    rows = []
    for check in operational.get("checks") or []:
        limit = (check.get("real_reference") or {}).get("noise_limit")
        if check.get("strata_forced"):
            note = "system check"
        elif check.get("passed") is None:
            note = f"undefined, not scored ({_check_note(check)})"
        elif check.get("value") is None:
            note = _check_note(check)
        else:
            note = "within real noise" if check.get("within_real_noise") else ""
        rows.append([check.get("name"), check.get("group"), check.get("metric"), fmt_num(check.get("value")),
                     fmt_num(check.get("threshold_used")), fmt_num(limit), fmt_num(check.get("score")),
                     _passed(check).replace("FAIL", "no").replace("pass", "yes"), note])
    if rows:
        lines += md_table(["check", "group", "metric", "value", "pass threshold", "real noise limit", "score", "passed",
                           "note"], rows)
    return lines


def _coverage_lines(result: dict[str, Any]) -> list[str]:
    coverage = result.get("scenario_coverage") or {}
    if coverage.get("skipped") or not coverage.get("scenarios"):
        return [f"- **Scenario coverage**: {coverage.get('reason') or 'not evaluated'}."]
    not_evaluated = coverage.get("not_evaluated_scenarios") or []
    lines = [f"- **Scenario coverage** {fmt_num(coverage.get('score'))}: {fmt_int(coverage.get('n_succeeded'))} of "
             f"{fmt_int(coverage.get('n_scenarios'))} scenarios generated, mean fulfilment "
             f"{fmt_pct(coverage.get('mean_fulfilment'))}, mean strata fidelity {fmt_num(coverage.get('mean_strata_fidelity'))}, "
             f"mean fidelity {fmt_num(coverage.get('mean_fidelity'))} ({fmt_int(coverage.get('rows_per_scenario'))} rows "
             f"requested per scenario)" + (f"; not evaluated (budget): {', '.join(not_evaluated)}" if not_evaluated else "")
             + ".", ""]
    rows = []
    for item in coverage.get("scenarios") or []:
        if item.get("status") == "not_evaluated":
            rows.append([item.get("name"), item.get("category"), "-", f"-/{fmt_int(item.get('requested'))}", "-", "-", "-",
                         "-", "-", "-", "budget"])
            continue
        rows.append([item.get("name"), item.get("category"), fmt_int((item.get("real_support") or {}).get("rows")),
                     f"{fmt_int(item.get('generated'))}/{fmt_int(item.get('requested'))}",
                     fmt_pct(item.get("acceptance_rate")),
                     fmt_num(item.get("strata_mix_tvd")) + (" (flagged)" if item.get("rebalancing_flag") else ""),
                     fmt_num(item.get("strata_rounding_tvd")),
                     fmt_num((item.get("combinations") or {}).get("coverage")),
                     fmt_num(item.get("fidelity")), fmt_num(item.get("score")), fmt_num(item.get("seconds"), 1)])
    lines += md_table(["scenario", "category", "real rows", "generated", "acceptance", "strata TVD", "rounding floor",
                       "combination coverage", "fidelity", "score", "s"], rows)
    lines += ["", "A strata TVD is flagged when it exceeds the rounding floor (the TVD of the best possible allocation "
              f"of that many rows) by more than {fmt_num(_strata_flag(), 2)} (section 4)."]
    return lines


def _model_detail(table: str, model_id: str, result: dict[str, Any], figure: Path | None, report_dir: Path,
                  full: bool) -> list[str]:
    composite = result.get("composite") or {}
    n_rows = result.get("n_rows") or {}
    lines = [f"### {table} / {model_id}" + (" (selected default)" if full else ""), "",
             f"Composite {fmt_num(composite.get('score'))}; evaluated on {fmt_int(n_rows.get('synthetic'))} "
             f"generated rows vs {fmt_int(n_rows.get('real_train'))} real training rows "
             f"(holdout {fmt_int(n_rows.get('real_holdout'))}).", ""]
    if composite.get("failed_families"):
        lines += [f"_Incomplete composite: {', '.join(composite['failed_families'])} failed and count as 0._", ""]
    lines += _statistical_lines(result) + _validity_lines(result) + _utility_lines(result) + _privacy_lines(result)
    if full:
        lines += _operational_lines(result) + [""] + _coverage_lines(result) + [""]
        lines += figure_line(figure, report_dir, f"{table} / {model_id}: scenario coverage and fidelity")
    else:
        operational = result.get("operational") or {}
        coverage = result.get("scenario_coverage") or {}
        undefined = [c.get("name") for c in _undefined_checks(operational)]
        lines.append(f"- **Operational patterns** {fmt_num(operational.get('score'))} "
                     f"({fmt_int(operational.get('n_passed'))}/{fmt_int(operational.get('n_checks'))} checks passed); "
                     f"failed: {', '.join(operational.get('failed_checks') or []) or 'none'}"
                     + (f"; undefined (not scored): {', '.join(undefined)}" if undefined else "") + ".")
        lines.append(f"- **Scenario coverage** {fmt_num(coverage.get('score'))} "
                     f"(mean fulfilment {fmt_pct(coverage.get('mean_fulfilment'))}, "
                     f"mean fidelity {fmt_num(coverage.get('mean_fidelity'))}).")
        lines.append("")
    return lines


def _details_section(comparison: dict[str, Any], model_results: dict[str, Any], figure_dir: Path,
                     report_dir: Path) -> list[str]:
    details: list[str] = ["## 8. Model details (appendix)", "",
                          "Every evaluation family per model: the selected default in full (operational checks and "
                          "per-scenario coverage), the other models in short form.", ""]
    for table in schema.TABLES:
        info = (comparison.get("tables") or {}).get(table)
        if not info:
            continue
        best = info.get("best_model")
        ordered = sorted(info.get("models") or [], key=lambda e: (e.get("model_id") != best, e.get("model_id")))
        for entry in ordered:
            model_id = entry.get("model_id")
            result = model_results.get(f"{table}__{model_id}")
            if not result:
                continue
            full = model_id == best
            coverage_fig = None
            if full and (result.get("scenario_coverage") or {}).get("scenarios"):
                scenarios = [s for s in result["scenario_coverage"]["scenarios"] if s.get("status") != "not_evaluated"]
                coverage_fig = _coverage_figure(table, model_id, scenarios,
                                                figure_dir / f"week3_{table}_{model_id}_scenarios.png")
            details += _model_detail(table, model_id, result, coverage_fig, report_dir, full)
    return details


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
NOTES: tuple[str, ...] = (
    "- Validity is measured on raw, unfiltered model samples: it shows how often a model produces operationally "
    "consistent rows on its own. The generator rejects every hard-invalid row, so returned data always passes all "
    "hard rules.",
    "- Strata-forced checks (occupancy-band mix per ward, admission-source mix per ward) and combination coverage "
    "reproduce the generator's allocation quotas, not the model; they are reported as system checks and excluded "
    "from the scores. The references are drawn through the same allocation, so model and reference rows compare "
    "like with like.",
    "- ML utility compares train-on-synthetic with a train-on-real model fitted on the same number of rows and "
    "scores skill (AUC above 0.5, average precision above the positive rate, R²): a classifier without skill scores 0.",
    "- The reference data is simulated, so privacy is assessed methodologically (the same checks apply to a real "
    "facility extract). Record-level checks (exact copies, distance to closest record) cannot see aggregate "
    "disclosure: the allocation strata reproduce the training split's joint ward x band / ward x source x delay "
    "frequencies, which the privacy section reports separately (not scored).",
    "- Calendar-mode rows are conditionally independent draws: day-to-day census continuity is not modelled "
    "(DESIGN.md §13).",
    "- Glossary: " + "; ".join(f"**{term}** = {meaning}" for term, meaning in GLOSSARY) + ".",
)


def _summary_section(comparison: dict[str, Any], model_results: dict[str, Any],
                     references: dict[str, Any], ab: dict[str, Any] | None = None) -> list[str]:
    """Headline results per table, computed from the comparison and the result files."""
    lines = ["## Summary", ""]
    ablation_parts = []
    for table in schema.TABLES:
        info = (comparison.get("tables") or {}).get(table) or {}
        best = info.get("best_model")
        entry = next((e for e in info.get("models") or [] if e.get("model_id") == best), None)
        if not entry:
            continue
        result = model_results.get(f"{table}__{best}") or {}
        real, floor = _reference_entry(info, REAL_REFERENCE), _reference_entry(info, NULL_REFERENCE)
        operational, coverage = _family(result, "operational"), _family(result, "scenario_coverage")
        config = _run_config(info, model_results, table)
        text = (f"- **{table}** ({fmt_int(config.get('rows'))} rows per model, "
                f"{'quick' if config.get('quick') else 'full'} evaluation): selected `{best}`, composite "
                f"{fmt_num(entry.get('composite'))}")
        if real or floor:
            text += (f" (no-learning floor {fmt_num(floor.get('composite'))}, real-holdout ceiling "
                     f"{fmt_num(real.get('composite'))})")
        validity = _family(result, "validity").get("score", (entry.get("components") or {}).get("validity"))
        text += f"; raw hard-valid {fmt_pct(validity)} before rejection, 100 % delivered"
        if operational:
            failed = [c.get("name") for c in operational.get("checks") or []
                      if c.get("passed") is False and not c.get("strata_forced")]
            text += (f"; {fmt_int(operational.get('n_passed'))} of {fmt_int(operational.get('n_checks'))} scored "
                     "operational checks passed" + (f" (failed: {', '.join(failed)})" if failed else "")
                     + (_ab_checks_text(ab) if table == schema.WARD_DAILY else ""))
        if coverage.get("n_scenarios"):
            text += (f"; {fmt_int(coverage.get('n_succeeded'))} of {fmt_int(coverage.get('n_scenarios'))} scenarios "
                     f"generated (coverage score {fmt_num(coverage.get('score'))})")
        optimisation = _family(result, "optimization")
        if optimisation.get("components"):
            refs = references.get(table) or {}
            floor_score = _family(refs.get(NULL_REFERENCE), "optimization").get("score")
            real_score = _family(refs.get(REAL_REFERENCE), "optimization").get("score")
            text += f"; optimisation utility {fmt_num(optimisation.get('score'))}"
            if is_number(floor_score) or is_number(real_score):
                text += f" (floor {fmt_num(floor_score)}, ceiling {fmt_num(real_score)})"
        lines.append(text + ".")
        steps = {e.get("model_id"): e for e in info.get("models") or [] if e.get("status", "ok") == "ok"}
        ablation = [m for m in ABLATION if m in steps]
        if len(ablation) > 1:
            ablation_parts.append(f"{table} " + " -> ".join(f"{m} {fmt_num(steps[m].get('composite'))}" for m in ablation))
    if len(lines) == 2:
        return []
    if ablation_parts:
        lines.append("- **Improvements made to SDV generation** (section 3): composite " + "; ".join(ablation_parts) + ".")
    lines.append("- Sections 4-6 compare validity, the occupancy, patient-flow and scheduling patterns and the "
                 "optimisation decisions of every model; section 7 lists the remaining gaps of the selected models.")
    return lines + [""]


def render_week3_report(
    results: dict[str, Any] | None = None,
    reports_dir: Path | str | None = None,
    figures_dir: Path | str | None = None,
) -> Path:
    """Write the Week-3 evaluation report.

    Args:
        results: ``{"comparison": <model_comparison dict>, "results": {"<table>__<model_id>":
            <evaluate_model dict>}, "references": {...}}`` as passed by ``scripts/03_evaluate.py``;
            ``None`` reads the files in ``reports/results``. Models and references missing from
            ``results`` are read from their result files when available.
        reports_dir, figures_dir: Output directories (default: configured directories).

    Returns:
        Path of ``week3_evaluation.md``.
    """
    if results is None or not results.get("comparison"):
        results = load_results()
    comparison = results.get("comparison") or {}
    model_results = dict(results.get("results") or {})
    references = {t: dict(v) for t, v in (results.get("references") or {}).items()}
    try:  # models evaluated in earlier runs are kept in the comparison: load their files too
        stored = load_results()
        for key, value in stored["results"].items():
            model_results.setdefault(key, value)
        for table, refs in stored["references"].items():
            for ref_id, value in refs.items():
                references.setdefault(table, {}).setdefault(ref_id, value)
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        pass
    report_dir, figure_dir = output_dirs(reports_dir, figures_dir)
    ab = load_ab_evidence(comparison)
    lines = [
        "# BedSynth AI - Week 3: quality and scheduling evaluation",
        "",
        f"Generated {utc_now()} from `scripts/03_evaluate.py` results (comparison of {comparison.get('generated_at', '-')}).",
        "",
    ]
    lines += _summary_section(comparison, model_results, references, ab)
    lines += _run_configuration_section(comparison, model_results, references)
    lines += ["## 2. Model comparison", ""]
    has_tables = False
    for table in schema.TABLES:
        info = (comparison.get("tables") or {}).get(table)
        if not info:
            continue
        has_tables = True
        refs = [r for r in info.get("references") or [] if r.get("status", "ok") == "ok"]
        entries = refs + _ordered_models(info)
        figure = _comparison_figure(table, entries, figure_dir / f"week3_{table}_model_comparison.png")
        lines += _comparison_section(table, info, model_results, figure, report_dir)
    if not has_tables:
        lines += ["_No evaluation results available: run `python scripts/03_evaluate.py`._", ""]
    lines += _ablation_section(comparison, model_results, ab)
    lines += _validity_section(comparison, model_results, references)
    lines += _patterns_section(comparison, model_results, references)
    lines += _optimisation_section(comparison, model_results, references)
    lines += _quality_gaps_section(comparison, model_results, references)
    lines += _details_section(comparison, model_results, figure_dir, report_dir)
    extra_notes = [note for note in (_deep_model_note(comparison, model_results),
                                     _sdmetrics_note(comparison, model_results)) if note]
    notes = [*NOTES[:-1], *extra_notes, NOTES[-1]]  # the glossary stays last
    lines += ["## 9. Notes", "", *notes, ""]
    return write_markdown(report_dir / REPORT_NAME, lines)


__all__ = ["REPORT_NAME", "load_results", "render_week3_report"]
