"""Week-2 report: the SDV synthesis pipeline (DESIGN.md §7, §11).

:func:`render_week2_report` documents the experiments of ``settings.synthesis`` (segmentation per
table, constraint groups), the constraints inside the SDV synthesizers (SDV built-ins and BedSynth
programmable constraints with the rules they enforce), the trained models (model cards from the
registry or from :func:`bedsynth.synthesis.training.train_all`) and the constraint ablation
(raw hard-valid rate of ``gc_baseline -> gc_segmented -> gc_tuned``) in
``reports/week2_synthesis_pipeline.md``.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Sequence

from bedsynth import schema
from bedsynth.config import get_settings
from bedsynth.reporting._common import (
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

REPORT_NAME = "week2_synthesis_pipeline.md"
ABLATION_ORDER: tuple[str, ...] = ("gc_baseline", "gc_segmented", "gc_tuned")
"""Experiments of the constraint/segmentation ablation, in the order they build on each other."""
MAX_SEGMENT_ROWS = 24
"""Segments listed per segmented model (ward x source models have about 20)."""
MIN_SOFT_RATE = 0.0005
"""Rule violation rates below this (0.0 % at one decimal) are left out of the 'most violated' lists."""
DEFAULT_MARKER = " (registry default)"


def _ablation_key(card: dict[str, Any]) -> tuple[int, int, str]:
    """Sort key: table order, then the ablation order, then the other experiments by id."""
    table, experiment = card.get("table"), card.get("experiment_id") or ""
    return (schema.TABLES.index(table) if table in schema.TABLES else len(schema.TABLES),
            ABLATION_ORDER.index(experiment) if experiment in ABLATION_ORDER else len(ABLATION_ORDER), experiment)


def registry_cards(models_dir: Path | str | None = None) -> list[dict[str, Any]]:
    """Model cards of every trained model in the registry."""
    from bedsynth.synthesis.registry import ModelRegistry

    registry = ModelRegistry(models_dir)
    data = registry.to_dict()
    cards = []
    for table in schema.TABLES:
        entry = data.get(table) or {}
        for model_id, card in (entry.get("models") or {}).items():
            if model_id in (entry.get("trained") or []):
                cards.append({**card, "table": table, "experiment_id": model_id,
                              "is_default": model_id == entry.get("effective_default")})
    return cards


def _diagnostics(card: dict[str, Any]) -> dict[str, Any]:
    return (card.get("training") or {}).get("sample_diagnostics") or {}


def _raw_validity(card: dict[str, Any]) -> float | None:
    value = _diagnostics(card).get("raw_hard_valid_rate")
    return float(value) if is_number(value) else None


def _top_violations(card: dict[str, Any], k: int = 3) -> str:
    rates = {rule: rate for rule, rate in (_diagnostics(card).get("hard_violation_rates") or {}).items()
             if is_number(rate) and rate >= MIN_SOFT_RATE}
    top = sorted(rates.items(), key=lambda kv: -kv[1])[:k]
    return ", ".join(f"{rule} {fmt_pct(rate, 1)}" for rule, rate in top) or "-"


def _marginals(card: dict[str, Any]) -> str:
    """Default marginal per segment, plus per-column overrides and degenerate-fit fallbacks."""
    chosen = [d for d in (card.get("distributions") or {}).values() if d]
    if not chosen:
        return "-"
    counts: dict[str, int] = {}
    for name in chosen:
        counts[name] = counts.get(name, 0) + 1
    text = ", ".join(f"{name} x{n}" if len(chosen) > 1 else name for name, n in sorted(counts.items()))
    selection = card.get("distribution_selection") or {}
    overrides = sum(len(v or {}) for v in (selection.get("column_overrides") or {}).values())
    fallbacks = sum(len(v or {}) for v in (selection.get("fallbacks") or {}).values())
    extras = ([f"{overrides} column overrides"] if overrides else []) + ([f"{fallbacks} fallbacks"] if fallbacks else [])
    return f"{text} ({', '.join(extras)})" if extras else text


def _segment_text(setting: Any) -> str:
    """``segment_by`` setting as text: 'global', 'ward_id', 'ward_id x admission_source' or per table."""
    if setting is None or setting == "" or setting == []:
        return "global"
    if isinstance(setting, str):
        return setting
    if isinstance(setting, (list, tuple)):
        return " x ".join(str(c) for c in setting)
    if isinstance(setting, dict):
        return "; ".join(f"{table}: {_segment_text(value)}" for table, value in setting.items())
    return str(setting)


def _groups_text(setting: Any) -> str:
    """Constraint groups of an experiment setting or card as text."""
    from bedsynth.synthesis.constraints import CONSTRAINT_GROUPS, constraint_groups

    try:
        groups = constraint_groups(setting)
    except ValueError:
        return str(setting)
    if not groups:
        return "none"
    if set(groups) == set(CONSTRAINT_GROUPS):
        return f"all ({', '.join(groups)})"
    return ", ".join(groups)


def _card_groups(card: dict[str, Any], applied: bool = False) -> list[str] | None:
    """Constraint groups a model enables (``applied=True``: the groups its segments actually apply)."""
    if applied and isinstance(card.get("effective_constraint_groups"), list):
        return card["effective_constraint_groups"]
    groups = card.get("constraint_groups")
    if isinstance(groups, list):
        return groups
    experiment = card.get("experiment") or {}
    if "constraints" in experiment:  # cards written before constraint groups: built-ins only
        return ["inequality", "fixed_combinations"] if experiment.get("constraints") else []
    return None


def _validity_figure(cards: Sequence[dict[str, Any]], path: Path) -> Path | None:
    experiments = sorted({c["experiment_id"] for c in cards},
                         key=lambda e: (ABLATION_ORDER.index(e) if e in ABLATION_ORDER else len(ABLATION_ORDER), e))
    tables = [t for t in schema.TABLES if any(c["table"] == t for c in cards)]
    lookup = {(c["table"], c["experiment_id"]): _raw_validity(c) for c in cards}

    def draw(fig: Any) -> None:
        ax = fig.add_subplot(1, 1, 1)
        series = {t: [lookup.get((t, e)) for e in experiments] for t in tables}
        grouped_bars(ax, experiments, series, ylabel="raw hard-valid rate")
        ax.set_title("Share of unfiltered model samples that pass every hard rule")

    return draw_figure(path, draw, size=(8.0, 4.0))


def _experiment_section(synthesis: dict[str, Any]) -> list[str]:
    experiments = synthesis.get("experiments") or {}
    rows = []
    for experiment_id, cfg in experiments.items():
        params = {k: v for k, v in (cfg.get("params") or {}).items()}
        rows.append([experiment_id, cfg.get("model"), _segment_text(cfg.get("segment_by")),
                     _groups_text(cfg.get("constraints", True)), "yes" if cfg.get("ordinal_metadata") else "no",
                     "yes" if cfg.get("ordered_encodings") else "no",
                     ", ".join(f"{k}={v}" for k, v in params.items())])
    lines = md_table(["experiment", "model", "segments", "constraint groups", "ordinal metadata",
                      "ordered encodings", "params"], rows)
    descriptions = [f"- `{experiment_id}`: {str(cfg['description']).rstrip('.')}." for experiment_id, cfg in experiments.items()
                    if cfg.get("description")]
    return lines + ([""] + descriptions if descriptions else [])


def _constraint_section() -> list[str]:
    from bedsynth.synthesis.constraints import ENFORCEMENT_NOTE, describe_constraints

    lines = [
        ENFORCEMENT_NOTE,
        "",
        "Programmable constraints are `sdv.cag.SingleTableProgrammableConstraint` subclasses in "
        "`bedsynth/synthesis/constraints.py`. They are listed before the SDV built-ins when added to a "
        "synthesizer, so their `reverse_transform` runs last on fully restored columns; per segment they "
        "receive the segment's constants (e.g. `elective_scheduled = 0` on AMU) and skip columns that are "
        "constant there.",
        "",
        "`conditional_marginals` (`ConditionalQuantileConstraint`) applies the distributional transform "
        "within groups: each declared column is replaced by a standard-normal score of its randomised "
        "empirical CDF in its group and mapped back with the group's empirical quantile function, so each "
        "group's marginal is reproduced exactly (point masses included) while the copula models the "
        "dependence of the scores on the other columns. It addresses limits of a linear Gaussian copula: "
        "delay hours per delay reason (exactly 0 without a reason, the reason's own range otherwise); elective "
        "lists per ward and weekday (ordering weekdays by mean electives imposed a false weekday gradient); "
        "mostly-zero counts, which smooth marginals made non-zero far too often - ED boarders and the mean bed "
        "wait per ward and occupancy band, closed beds per ward and staffing band (both bands recomputed from "
        "the counts, so boarders appear on full ward-days and closures follow short staffing) and escalation "
        "beds beyond the census need per ward; day-shift nurses per ward and month (the winter short staffing "
        "sits in the lower tail of a few-valued roster); length of stay per admission weekday (the discharge "
        "weekday is admission weekday + nights mod 7, which a linear copula cannot learn); and exact "
        "admission/discharge-hour, bed-wait and turnaround marginals per ward x admission source (smoothed "
        "parametric hour profiles and waits failed the operational checks). An earlier repair that forced "
        "delay hours to 0 when no reason was sampled raised raw validity but kept the length of stay drawn "
        "with the discarded delay, inflating the stays of non-delayed patients; it was replaced by this "
        "group. Repairs are only kept when they "
        "re-parameterise the data instead of overwriting a jointly sampled value. The development A/B "
        "measurements behind these choices are in DESIGN.md §7.2-7.3.",
        "",
    ]
    rows = []
    for table in schema.TABLES:
        for item in describe_constraints(table):
            implementation = item.get("implementation") or item["sdv_class"]
            if item.get("programmable"):
                implementation = f"{implementation.rsplit('.', 1)[-1]} (programmable)"
            rows.append([table, item["group"], implementation, item.get("expression") or ", ".join(item["columns"]),
                         ", ".join(item["enforces_rules"]) or "-", item["description"]])
    lines += md_table(["table", "group", "implementation", "expression", "rules enforced", "description"], rows)
    return lines


def _ablation_section(trained: Sequence[dict[str, Any]]) -> list[str]:
    """Raw validity of the ablation experiments (and every other trained model) per table."""
    rows = []
    for table in schema.TABLES:
        cards = [c for c in trained if c.get("table") == table]
        cards.sort(key=_ablation_key)
        for card in cards:
            soft = {rule: rate for rule, rate in (_diagnostics(card).get("soft_violation_rates") or {}).items()
                    if is_number(rate) and rate >= MIN_SOFT_RATE}
            top_soft = ", ".join(f"{r} {fmt_pct(v, 1)}" for r, v in sorted(soft.items(), key=lambda kv: -kv[1])[:2]) or "-"
            groups, applied = _card_groups(card), _card_groups(card, applied=True)
            rows.append([table, card["experiment_id"], _segment_text(card.get("segment_columns", card.get("segment_by"))),
                         _groups_text(groups) if groups is not None else "-",
                         _groups_text(applied) if applied is not None and applied != groups else "same",
                         fmt_pct(_raw_validity(card)), _top_violations(card), top_soft])
    if not rows:
        return []
    return [
        "## 5. Constraint ablation (raw validity of unfiltered samples)",
        "",
        "`gc_baseline` is one global GaussianCopula with default encodings and no constraints. `gc_segmented` "
        "adds one model per ward, ordinal metadata, demand-ordered encodings (ward_daily: months by mean "
        "occupancy, weekdays by mean discharges; patient_stays: admission months by mean length of stay, weekdays "
        "in calendar order; the ward order by bed capacity only applies to global models) and marginals selected "
        "by KS (a default per segment plus per-column "
        "overrides, with a post-fit check that replaces fits that degenerated on the full segment), with "
        "SDV's built-in constraints only. `gc_tuned` adds the programmable groups (band labels, structural "
        "zeros, escalation cover, conditional marginals), orders the patient_stays outcome combinations "
        "(separation, destination, delay reason) by mean length of stay and segments patient_stays per ward "
        "x admission source. 'applied' lists the groups the segments actually use when it differs from the "
        "enabled ones (in a ward x source model the segment columns already reproduce PS-H06 and PS-H11).",
        "",
        "The rate is measured on unfiltered samples right after training (`training.sample_diagnostics`). "
        "The generator rejects the remaining invalid rows, so a higher rate means fewer rejection rounds and "
        "less rebalancing of allocation strata - but raw validity is a diagnostic, not a goal: a constraint "
        "that raises it by overwriting sampled values can bias other columns (section 3).",
        "",
        *md_table(["table", "experiment", "segments", "constraint groups", "applied", "raw hard-valid",
                   "most violated hard rules", "most violated soft rules"], rows),
        "",
        *_coverage_lines(trained),
    ]


def _coverage_lines(trained: Sequence[dict[str, Any]]) -> list[str]:
    """How the gc_tuned models enforce each rule: inside SDV, by segment constants, or by rejection."""
    rows = []
    for card in trained:
        if card.get("experiment_id") != "gc_tuned":
            continue
        for item in card.get("constraint_coverage") or []:
            total = item.get("segments") or 0
            rows.append([card["table"], item.get("group"), ", ".join(item.get("columns") or []),
                         ", ".join(item.get("enforces_rules") or []) or "-",
                         f"{item.get('constraint_segments', 0)} / {total}",
                         f"{item.get('constant_segments', 0)} / {total}"])
    if not rows:
        return []
    return [
        "**Constraint coverage of gc_tuned** - segments that apply a constraint inside SDV, and segments where "
        "the segment columns or constant columns reproduce it exactly; elsewhere it is not needed or the "
        "generator's rejection step enforces the rule:",
        "",
        *md_table(["table", "group", "columns", "rules", "applied in SDV", "by segment constants"], rows),
        "",
    ]


def _segments_section(trained: Sequence[dict[str, Any]]) -> list[str]:
    segmented = sorted((c for c in trained if c.get("segment_by")), key=_ablation_key)
    if not segmented:
        return []
    lines: list[str] = [
        "## 6. Segments of the segmented models",
        "",
        f"Per segment (at most {MAX_SEGMENT_ROWS} listed per model): training rows, marginal family, modelled "
        "columns, constraints applied inside SDV, constant columns dropped before fitting and re-inserted when "
        "sampling, and the learned structural zeros and conditional-marginal groups.",
        "",
    ]
    for card in segmented:
        seg_rows = []
        for segment in (card.get("segments") or [])[:MAX_SEGMENT_ROWS]:
            constants = segment.get("constant_columns") or {}
            zeros = []
            for constraint in segment.get("constraints") or []:
                learned = (constraint.get("learned") or {}).get("zero_categories") or {}
                for column, values in learned.items():
                    if values:
                        zeros.append(f"{column}=0 if {constraint['columns'][0]} in {{{', '.join(map(str, values))}}}")
                if constraint.get("group") == "conditional_marginals":
                    # band_groups: bands recomputed from each row's counts (named as in the expression)
                    parts = list(constraint.get("group_by") or [])
                    parts += [f"{str(band).replace('_level', '')} band (recomputed)"
                              for band in constraint.get("band_groups") or []]
                    groups = " x ".join(parts) or "segment"
                    zeros.append(f"{', '.join(constraint.get('columns') or [])} per {groups}")
            seg_rows.append([segment.get("segment"), fmt_int(segment.get("rows")), segment.get("distribution") or "-",
                             len(segment.get("modelled_columns") or []), len(segment.get("constraints") or []),
                             ", ".join(f"{k}={v}" for k, v in constants.items()) or "-", "; ".join(zeros) or "-"])
        extra = len(card.get("segments") or []) - MAX_SEGMENT_ROWS
        lines += [f"**{card['table']} / {card['experiment_id']}** segments "
                  f"({_segment_text(card.get('segment_columns', card.get('segment_by')))}):", ""]
        lines += md_table(["segment", "rows", "marginal", "modelled columns", "constraints",
                           "constant columns (re-inserted)", "structural zeros / conditional marginals"], seg_rows)
        if extra > 0:
            lines.append(f"\n_... and {extra} more segments (see the model card)._")
        lines.append("")
    return lines


def _count(n: int, noun: str) -> str:
    return f"{n} {noun}" + ("" if n == 1 else "s")


def _summary_lines(trained: Sequence[dict[str, Any]], failed: Sequence[dict[str, Any]]) -> list[str]:
    """Headline facts of the Week-2 pipeline, computed from the model cards (placed under the title)."""
    if not trained and not failed:
        return []
    lines = ["## Summary", ""]
    if trained:
        quick = any((c.get("training") or {}).get("quick") for c in trained)
        fit_seconds = [float(c["fit_seconds"]) for c in trained if is_number(c.get("fit_seconds"))]
        experiments = {c["experiment_id"] for c in trained}
        tables = {c.get("table") for c in trained}
        lines.append(f"- {_count(len(trained), 'model')} trained ({_count(len(experiments), 'experiment')} x "
                     f"{_count(len(tables), 'table')}{', quick mode' if quick else ''}); total fit time "
                     f"{fmt_num(sum(fit_seconds) / 60.0, 1)} min (section 4).")
    for table in schema.TABLES:
        cards = {c["experiment_id"]: c for c in trained if c.get("table") == table}
        if not cards:
            continue
        parts = []
        steps = [m for m in ABLATION_ORDER if m in cards]
        if steps:
            parts.append("raw hard-valid rate of unfiltered samples "
                         + " -> ".join(f"{m} {fmt_pct(_raw_validity(cards[m]))}" for m in steps) + " (section 5)")
        default = next((m for m, card in cards.items() if card.get("is_default")), None)
        if default:
            applied = _card_groups(cards[default], applied=True)
            parts.append(f"registry default when this report was rendered: `{default}`"
                         + (f", applying {', '.join(applied)} inside SDV" if applied else ""))
        lines.append(f"- **{table}**: " + "; ".join(parts) + ".")
    if trained:
        lines.append("- Delivered data is 100 % hard-valid whatever a model's raw rate: the generator rejects every row "
                     "that breaks a hard rule.")
    if failed:
        lines.append(f"- {len(failed)} experiment(s) failed in this training run (section 7).")
    return lines + [""]


def _raw_validity_note(trained: Sequence[dict[str, Any]]) -> str:
    """Where the raw hard-valid rates of this report come from (and why Week 3 differs)."""
    rows = max((int(_diagnostics(c).get("rows") or 0) for c in trained), default=0)
    sample = f"{fmt_int(rows)} unfiltered model samples" if rows else "unfiltered model samples"
    return (f"Raw validity is measured on {sample} drawn right after training (`training.sample_diagnostics`); the "
            "Week-3 evaluation re-measures it on its own sample (size in its section 1), so the two reports can "
            "differ by sampling noise.")


def render_week2_report(
    cards: Sequence[dict[str, Any]] | None = None,
    reports_dir: Path | str | None = None,
    figures_dir: Path | str | None = None,
    models_dir: Path | str | None = None,
) -> Path:
    """Write the Week-2 pipeline report.

    Args:
        cards: Model cards (e.g. the list returned by ``train_all``; failed entries are listed
            as failures). Default: every trained model in the registry.
        reports_dir, figures_dir: Output directories (default: configured directories).
        models_dir: Registry root used when ``cards`` is None.

    Returns:
        Path of ``week2_synthesis_pipeline.md``.
    """
    report_dir, figure_dir = output_dirs(reports_dir, figures_dir)
    all_cards = list(cards) if cards is not None else registry_cards(models_dir)
    trained = [c for c in all_cards if c.get("status", "trained") == "trained" and c.get("experiment_id")]
    failed = [c for c in all_cards if c.get("status") == "failed"]
    synthesis = get_settings().get("synthesis") or {}

    lines = [
        "# BedSynth AI - Week 2: SDV synthesis pipeline",
        "",
        f"Generated {utc_now()}. Models are fitted by `scripts/02_train_synthesizers.py` "
        "(`bedsynth.synthesis.training`) on the processed train splits and registered in `models/registry.json`.",
        "",
        *_summary_lines(trained, failed),
        "## 1. Pipeline",
        "",
        "1. **Metadata** (`synthesis/metadata.py`): SDV 2.x `Metadata` built from `bedsynth.schema` "
        "(train columns only; occupancy and staffing levels as `ordinal`).",
        "2. **Constraints** (`synthesis/constraints.py`): constraint groups applied inside the SDV synthesizer - "
        "SDV built-ins (`Inequality`, `FixedCombinations`) and BedSynth programmable constraints (band labels "
        "recomputed from counts, learned structural zeros, escalation beds that cover the census, exact "
        "per-group marginals by the distributional transform).",
        "3. **Model wrapper** (`synthesis/models.py`): global or segmented SDV synthesizers (one per ward, or per "
        "ward x admission source), constant columns dropped and re-inserted per segment, demand-ordered "
        "`OrderedUniformEncoder`s (including the merged patient_stays outcome combinations, ordered by mean "
        "length of stay), marginal selection by KS per segment and per column with a post-fit check for "
        "degenerate fits, seeded reproducible sampling, pooled rejection for conditional requests.",
        "4. **Registry** (`synthesis/registry.py`): model cards, default model per table.",
        "5. **Generation engine** (`synthesis/generator.py`): scenario conditions and filters, derivation, "
        "rule-based rejection sampling with allocation strata (filtered requests take their strata from a "
        "synthetic pilot draw, never from the real rows that match the filters), calendar mode, per-request "
        "time budget with per-round row caps.",
        "",
        "## 2. Experiments (settings.yaml)",
        "",
    ]
    lines += _experiment_section(synthesis)
    lines += ["", "## 3. Constraints and enforcement", ""]
    lines += _constraint_section()

    lines += ["", "## 4. Trained models", ""]
    if not trained:
        lines += ["_No trained models yet: run `python scripts/02_train_synthesizers.py`._", ""]
    else:
        model_rows = []
        for card in sorted(trained, key=_ablation_key):
            training = card.get("training") or {}
            model_rows.append([
                card["table"], card["experiment_id"] + (DEFAULT_MARKER if card.get("is_default") else ""),
                card.get("synthesizer") or card.get("model") or "-", fmt_int(card.get("training_rows")),
                fmt_int(card.get("n_segments")), _marginals(card), fmt_num(card.get("fit_seconds"), 1),
                "yes" if training.get("quick") else "no", fmt_pct(_raw_validity(card)), _top_violations(card),
            ])
        lines += md_table(["table", "experiment", "synthesizer", "training rows", "segments", "marginals", "fit s",
                           "quick", "raw hard-valid", "most violated hard rules (raw samples)"], model_rows)
        lines += ["", f"'{DEFAULT_MARKER.strip()}' marks the default when this report was rendered; "
                  "`scripts/03_evaluate.py` re-selects it from the Week-3 composite."]
        figure = _validity_figure(trained, figure_dir / "week2_raw_validity.png")
        lines += [""] + figure_line(figure, report_dir, "Raw (unfiltered) hard-rule validity per model")
        lines += [
            _raw_validity_note(trained) + " The generator rejects "
            "rows that violate a hard rule, so generated data is always 100 % hard-valid. Low raw validity costs "
            "more than sampling time: rejection accepts some strata more easily than others, which would skew the "
            "output mix without the generator's allocation strata (rebalancing remains a bounded bias), and rows "
            "rejected for one rule shift other columns (rejecting short delayed stays raises the mean length of "
            "stay). Raising raw validity only helps when it does not bias other columns, which is why repairs "
            "that overwrite sampled values are avoided (section 3).",
            "",
        ]
        lines += _ablation_section(trained)
        lines += _segments_section(trained)
    if failed:
        lines += ["## 7. Failed experiments", ""] + [f"- {c.get('table')}/{c.get('experiment_id')}: {c.get('error')}"
                                                     for c in failed]
    return write_markdown(report_dir / REPORT_NAME, lines)


__all__ = ["ABLATION_ORDER", "REPORT_NAME", "registry_cards", "render_week2_report"]
