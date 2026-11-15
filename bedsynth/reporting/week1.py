"""Week-1 report: data requirements and dataset preparation (DESIGN.md §5-§6, §11).

:func:`render_week1_report` turns the data-quality report of
:func:`bedsynth.data.preprocessing.prepare_datasets` (plus the simulator manifest and the
cleaned data when they are on disk) into ``reports/week1_data_preparation.md`` with figures in
``reports/figures``. Called without an argument it reads
``<processed_dir>/data_quality_report.json``.

The directories and files recorded in the data-quality report (``raw_dir``, ``processed_dir``,
``tables.<table>.raw_file``, ``outputs``) may be absolute, home-relative (``~/...``) or
project-relative: relative ones are resolved against :data:`bedsynth.config.PROJECT_ROOT`,
``~/...`` against the home directory, and a recorded location that does not exist
falls back to the configured ``raw_dir`` / ``processed_dir``, so the manifest-, CSV- and
figure-based sections do not disappear when the stored paths go stale. The report itself prints
only project-relative paths.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path, PureWindowsPath
from typing import Any

import numpy as np
import pandas as pd

from bedsynth import schema, validation
from bedsynth.config import HOME_PREFIX, PROJECT_ROOT, get_path
from bedsynth.reporting._common import (
    PALETTE,
    display_path,
    draw_figure,
    figure_line,
    fmt_int,
    fmt_num,
    fmt_pct,
    grouped_bars,
    is_inside_project,
    md_table,
    output_dirs,
    path_name,
    to_utc,
    utc_now,
    write_markdown,
)

logger = logging.getLogger(__name__)

REPORT_NAME = "week1_data_preparation.md"
QUALITY_REPORT_NAME = "data_quality_report.json"
MANIFEST_NAME = "simulation_manifest.json"
SCHEMA_CATALOG_NAME = "schema_catalog.json"
MAX_LISTED_CATEGORIES = 8

#: Scenario families required by the project brief -> scenario categories of the catalogue.
BRIEF_SCENARIO_FAMILIES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Normal, high and low occupancy", ("occupancy",)),
    ("Seasonal demand", ("seasonal",)),
    ("Admission and discharge patterns", ("patient_flow",)),
    ("Emergency bed requirements", ("emergency",)),
    ("Ward utilisation", ("utilization",)),
    ("Scheduled procedures", ("scheduling",)),
    ("Bed turnover", ("turnover",)),
    ("Delayed discharge", ("delayed_discharge",)),
    ("Varying staffing or scheduling conditions", ("staffing", "scheduling")),
    ("Representative everyday mix (reference for all comparisons)", ("baseline",)),
)
BASELINE_CATEGORY = "baseline"

#: Text of section 1.2: how scenario conditions and filters are met (bedsynth/synthesis/models.py
#: ``sample_many``: pooled rejection from unconditional draws, native conditioning only as fallback).
SCENARIO_INTRO = (
    "Each scenario family of the brief maps to named scenarios (`bedsynth.scenarios`). *Conditions* (values of "
    "categorical training columns) select the matching segment models and are met exactly by pooled rejection "
    "from unconditional SDV draws, with SDV's native conditional sampling only for combinations too rare to pool "
    "(DESIGN.md §7.3); *filters* (bounds on any column, derived ones included) are enforced by rejection sampling. "
    "*Train rows* is the number of reference rows of the train split inside the scenario (what the synthesizers "
    "learn it from)."
)

#: File names ``prepare_datasets`` writes to ``processed_dir`` (used when ``outputs`` lacks a key).
DEFAULT_OUTPUT_NAMES: dict[str, str] = {
    **{f"{table}_{split}": f"{table}_{split}.csv" for table in schema.TABLES for split in ("train", "holdout", "clean")},
    **{f"metadata_{table}": f"metadata_{table}.json" for table in schema.TABLES},
    "metadata": "metadata.json",
    "schema_catalog": SCHEMA_CATALOG_NAME,
    "data_quality_report": QUALITY_REPORT_NAME,
}


# ---------------------------------------------------------------------------
# Locations recorded in the data-quality report
# ---------------------------------------------------------------------------
def _stored_candidate(value: Any) -> Path | None:
    """A path recorded in the report: absolute as recorded, ``~/...`` (the form
    :func:`bedsynth.config.project_relative` stores outside the project) under the home directory,
    relative against the project root.

    Absolute paths of another operating system (a Windows drive path read on POSIX) cannot be
    used and give ``None``; the callers then fall back to the configured directories.
    """
    if value is None or str(value).strip() == "":
        return None
    text = str(value).strip()
    if text == HOME_PREFIX or text.startswith((HOME_PREFIX + "/", HOME_PREFIX + "\\")):
        try:
            return Path(HOME_PREFIX).expanduser().joinpath(*[part for part in re.split(r"[\\/]", text[1:]) if part])
        except RuntimeError:  # no home directory on this machine
            return None
    if Path(text).is_absolute():
        return Path(text)
    if PureWindowsPath(text).drive or text.startswith(("/", "\\")):
        return None
    return PROJECT_ROOT.joinpath(*[part for part in re.split(r"[\\/]", text) if part])


def _resolve_dir(value: Any, key: str) -> Path | None:
    """Directory recorded in the report if it exists, else the configured ``key`` directory."""
    candidate = _stored_candidate(value)
    if candidate is not None and candidate.is_dir():
        return candidate
    try:
        configured = get_path(key, create=False)
    except KeyError:
        return None
    return configured if configured.is_dir() else None


def _resolve_file(value: Any, directory: Path | None, default_name: str | None = None) -> Path | None:
    """File recorded in the report if it exists, else the file of that name in ``directory``."""
    candidate = _stored_candidate(value)
    if candidate is not None and candidate.is_file():
        return candidate
    name = path_name(value) or default_name
    if directory is not None and name and (directory / name).is_file():
        return directory / name
    return None


def _output_paths(report: dict[str, Any], processed_dir: Path | None) -> dict[str, str]:
    """Output key -> project-relative path of the files ``prepare_datasets`` wrote (existing or recorded)."""
    outputs = dict(report.get("outputs") or {})
    keys = list(outputs) + [key for key in DEFAULT_OUTPUT_NAMES if key not in outputs]
    shown: dict[str, str] = {}
    for key in keys:
        resolved = _resolve_file(outputs.get(key), processed_dir, DEFAULT_OUTPUT_NAMES.get(key))
        if resolved is not None:
            shown[key] = display_path(resolved)
        elif outputs.get(key):
            shown[key] = display_path(outputs[key])
    return shown


def _load_quality_report() -> dict[str, Any]:
    path = get_path("processed_dir", create=False) / QUALITY_REPORT_NAME
    if not path.is_file():
        raise FileNotFoundError(f"{QUALITY_REPORT_NAME} not found: run `python scripts/01_prepare_data.py` first.")
    return json.loads(path.read_text(encoding="utf-8"))


def _load_manifest(raw_dir: Path | None) -> dict[str, Any] | None:
    if raw_dir is None:
        return None
    path = raw_dir / MANIFEST_NAME
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Cannot read %s: %s", path.name, exc)
        return None


def _load_clean(processed_dir: Path | None, table: str, split: str = "clean") -> pd.DataFrame | None:
    if processed_dir is None:
        return None
    from bedsynth.data import io

    try:
        return io.load_processed(table, split, processed_dir)
    except Exception as exc:  # noqa: BLE001 - figures from data are optional
        logger.info("%s %s data not available for the report: %s", split.capitalize(), table, type(exc).__name__)
        return None


def _load_schema_catalog(processed_dir: Path | None) -> dict[str, Any] | None:
    """``schema_catalog.json`` written by ``prepare_datasets`` (metadata files, constraints, rules)."""
    if processed_dir is None:
        return None
    path = processed_dir / SCHEMA_CATALOG_NAME
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("tables")
    except (OSError, json.JSONDecodeError, AttributeError) as exc:
        logger.warning("Cannot read %s: %s", path.name, exc)
        return None


#: Detected issue types of the cleaning pipeline that correspond to the injector's ``duplicate_row``.
_DUPLICATE_KINDS: tuple[str, ...] = ("exact_duplicate", "key_duplicate")


def _detected_issues(issues: dict[str, dict[str, int]] | None) -> dict[str, dict[str, int]]:
    """Detected issues keyed like the injection log (duplicates merged into ``duplicate_row``)."""
    merged: dict[str, dict[str, int]] = {}
    for kind, columns in (issues or {}).items():
        target = "duplicate_row" if kind in _DUPLICATE_KINDS else kind
        bucket = merged.setdefault(target, {})
        for column, count in columns.items():
            bucket[column] = bucket.get(column, 0) + int(count)
    return merged


def _issue_totals(issues: dict[str, dict[str, int]] | None) -> dict[str, int]:
    return {kind: int(sum(columns.values())) for kind, columns in (issues or {}).items()}


def _hard_rule_counts() -> dict[str, int]:
    return {table: sum(1 for rule in validation.rules_catalog(table) if rule["severity"] == "hard")
            for table in schema.TABLES}


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def _issues_figure(report: dict[str, Any], path: Path) -> Path | None:
    tables = [t for t in schema.TABLES if t in report.get("tables", {})]

    def draw(fig: Any) -> None:
        axes = fig.subplots(1, len(tables), squeeze=False)[0]
        for ax, table in zip(axes, tables):
            info = report["tables"][table]
            injected = _issue_totals(info.get("issues_injected"))
            detected = _issue_totals(_detected_issues(info.get("issues_detected")))
            kinds = sorted(set(injected) | set(detected))
            series = {"injected": [injected.get(k, 0) for k in kinds], "detected": [detected.get(k, 0) for k in kinds]}
            if not injected:
                series.pop("injected")
            grouped_bars(ax, [k.replace("_", "\n") for k in kinds], series, ylabel="issues", ylim=None)
            ax.set_title(table)
            ax.tick_params(axis="x", labelsize=7)

    return draw_figure(path, draw, size=(11.0, 4.2))


def _occupancy_figure(ward_daily: pd.DataFrame, path: Path) -> Path | None:
    def draw(fig: Any) -> None:
        ax = fig.add_subplot(1, 1, 1)
        derived = schema.derive(schema.WARD_DAILY, ward_daily)
        monthly = derived.groupby(["ward_id", "month"])["occupancy_rate"].mean().unstack("ward_id")
        for i, ward in enumerate(w for w in schema.WARD_IDS if w in monthly.columns):
            ax.plot(monthly.index, monthly[ward], marker="o", ms=3, label=ward, color=PALETTE[i % len(PALETTE)])
        ax.set_xticks(range(1, 13))
        ax.set_xlabel("month")
        ax.set_ylabel("mean midnight occupancy")
        ax.set_title("Seasonal occupancy profile by ward (reference data)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, ncol=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.14))

    return draw_figure(path, draw, size=(8.0, 4.8))


def _band_mix_figure(ward_daily: pd.DataFrame, path: Path) -> Path | None:
    def draw(fig: Any) -> None:
        ax = fig.add_subplot(1, 1, 1)
        shares = (ward_daily.groupby("ward_id")["occupancy_level"].value_counts(normalize=True)
                  .unstack(fill_value=0.0).reindex(columns=list(schema.OCCUPANCY_LEVELS), fill_value=0.0))
        wards = [w for w in schema.WARD_IDS if w in shares.index]
        bottom = np.zeros(len(wards))
        for i, level in enumerate(schema.OCCUPANCY_LEVELS):
            values = shares.loc[wards, level].to_numpy(dtype=float)
            ax.bar(wards, values, bottom=bottom, label=level, color=PALETTE[i % len(PALETTE)])
            bottom += values
        ax.set_ylim(0, 1)
        ax.set_ylabel("share of ward-days")
        ax.set_title("Occupancy band mix by ward (reference data)")
        # below the axes: inside them the top band's swatch sits on bars of the same colour
        ax.legend(fontsize=8, ncol=len(schema.OCCUPANCY_LEVELS), frameon=False, loc="upper center",
                  bbox_to_anchor=(0.5, -0.08), title="occupancy band", title_fontsize=8)

    return draw_figure(path, draw)


def _los_figure(stays: pd.DataFrame, path: Path) -> Path | None:
    def draw(fig: Any) -> None:
        ax = fig.add_subplot(1, 1, 1)
        derived = schema.derive(schema.PATIENT_STAYS, stays)
        for i, source in enumerate(schema.ADMISSION_SOURCES):
            values = derived.loc[derived["admission_source"] == source, "los_days"].dropna()
            if len(values):
                ax.hist(values.clip(upper=30), bins=60, density=True, histtype="step", linewidth=1.6,
                        label=f"{source} (median {values.median():.1f} d)", color=PALETTE[i % len(PALETTE)])
        ax.set_xlabel("length of stay (days, clipped at 30)")
        ax.set_ylabel("density")
        ax.set_title("Length of stay by admission source (reference data)")
        ax.legend(fontsize=8, frameon=False)

    return draw_figure(path, draw)


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------
def _brief_families() -> list[tuple[str, tuple[str, ...]]]:
    """Scenario families of the brief (the baseline reference mix is not one of them)."""
    return [(family, categories) for family, categories in BRIEF_SCENARIO_FAMILIES if BASELINE_CATEGORY not in categories]


def _summary_section(report: dict[str, Any], manifest: dict[str, Any] | None, support: dict[str, dict[str, int]],
                     files: dict[str, str]) -> list[str]:
    """Headline facts of the Week-1 deliverable, computed from the artefacts (placed under the title)."""
    from bedsynth.scenarios import SCENARIOS

    head = report.get("summary") or {}
    tables = [t for t in schema.TABLES if t in head]
    hard = _hard_rule_counts()
    lines = ["## Summary", ""]
    if manifest:
        period = manifest.get("period") or {}
        calibration = manifest.get("calibration") or {}
        text = (f"- **Reference data:** a seeded discrete-event simulation of a {schema.TOTAL_BEDS}-bed, "
                f"{len(schema.WARDS)}-ward hospital, {period.get('start_date')} .. {period.get('end_date')}")
        if calibration.get("targets_evaluated"):
            text += (f"; {fmt_int(calibration.get('targets_passed'))} of {fmt_int(calibration.get('targets_evaluated'))} "
                     "calibration targets met")
        lines.append(text + " (section 2).")
    else:
        lines.append("- **Reference data:** an externally provided raw extract (no simulation manifest found; section 2).")
    for table in tables:
        info = head[table]
        parts = [f"{fmt_int(info.get('raw_rows'))} raw rows -> {fmt_int(info.get('final_rows'))} prepared rows "
                 f"({fmt_pct(info.get('retention_rate'))} kept)",
                 f"hard-valid rate {fmt_pct(info.get('hard_valid_rate_raw'))} -> {fmt_pct(info.get('hard_valid_rate_final'))} "
                 f"against {hard.get(table)} hard rules"]
        resolved, exact = info.get("injected_issues_resolved_rate"), _exact_recovery(report, table, info)
        if resolved is not None or exact is not None:
            parts.append(f"injected issues no longer defective {fmt_pct(resolved)}, exactly recovered {fmt_pct(exact)}")
        lines.append(f"- **{table}:** " + "; ".join(parts) + " (sections 3-6).")
    train = [f"`{files[f'{t}_train']}` ({fmt_int(head[t].get('train_rows'))} rows)" for t in tables if files.get(f"{t}_train")]
    if train:
        metadata = [f"`{files[f'metadata_{t}']}`" for t in tables if files.get(f"metadata_{t}")]
        holdout = " / ".join(fmt_int(head[t].get("holdout_rows")) for t in tables)
        lines.append("- **SDV-ready dataset:** " + " and ".join(train)
                     + (f", described by the single-table SDV metadata {' and '.join(metadata)}" if metadata else "")
                     + f"; the holdout splits ({holdout} rows) are never used for training and serve the Week-3 "
                     "evaluation (sections 9-10).")
    by_table = {t: sum(1 for s in SCENARIOS.values() if s.table == t) for t in schema.TABLES}
    families = _brief_families()
    covered = [f for f, categories in families if any(s.category in categories for s in SCENARIOS.values())]
    text = (f"- **Scenarios:** {len(SCENARIOS)} named scenarios ("
            + ", ".join(f"{by_table[t]} {t}" for t in schema.TABLES)
            + f") cover {len(covered)} of the {len(families)} scenario families of the brief, plus a baseline")
    counts = [(count, name) for table_support in support.values() for name, count in table_support.items()]
    if counts:
        smallest = min(counts)
        text += f"; the smallest, `{smallest[1]}`, has {fmt_int(smallest[0])} train rows"
    lines += [text + " (section 1.2).", ""]
    return lines


def _requirements_section() -> list[str]:
    lines = [
        "## 1. Data requirements",
        "",
        "Hospital administrators need realistic operational data to test bed-occupancy and scheduling "
        "optimisation models, but real extracts cannot be shared between facilities. BedSynth AI therefore "
        "works with two strictly **non-clinical** tables (no diagnoses, procedures, ages or identifiers):",
        "",
    ]
    rows = []
    for table in schema.TABLES:
        rows.append([
            table,
            "one row per ward per day" if table == schema.WARD_DAILY else "one row per inpatient ward stay",
            len(schema.training_columns(table)),
            len(schema.derived_columns(table)),
            ", ".join(schema.time_columns(table)),
        ])
    lines += md_table(["table", "grain", "train columns", "derived columns", "time columns"], rows)
    lines += [
        "",
        "Derived columns (flow balance, occupancy rate, ratios, LOS hours...) are recomputed with "
        "`schema.derive`, never learned, so identities such as `census_end = census_start + admissions + "
        "transfers_in - transfers_out - discharges` always hold. The reference hospital has "
        f"{schema.TOTAL_BEDS} funded beds:",
        "",
    ]
    ward_rows = [
        [w.ward_id, w.name, w.ward_type, w.bed_capacity, w.max_escalation_beds, w.required_nurses_day,
         ", ".join(w.admission_sources)]
        for w in schema.WARDS.values()
    ]
    lines += md_table(["ward", "name", "type", "beds", "max escalation", "day nurses required", "admission sources"],
                      ward_rows)
    return lines + [""]


def _value_domain(spec: schema.ColumnSpec) -> str:
    """Allowed values of a column: its vocabulary (shortened) or its physical bounds."""
    if spec.categories:
        values = [str(v) for v in spec.categories]
        if len(values) > MAX_LISTED_CATEGORIES:
            values = values[:MAX_LISTED_CATEGORIES] + [f"... ({len(spec.categories)} values)"]
        return ", ".join(values)
    if spec.dtype == "bool":
        return "true / false"
    if spec.min_value is not None or spec.max_value is not None:
        low = "" if spec.min_value is None else fmt_num(spec.min_value, 4)
        high = "" if spec.max_value is None else fmt_num(spec.max_value, 4)
        return f"{low} .. {high}".strip()
    return ""


def _attribute_dictionary_section() -> list[str]:
    """Every column: role, types, unit, allowed values and meaning (from the schema catalogue)."""
    lines = [
        "### 1.1 Attribute dictionary",
        "",
        "Roles: `key` identifier assigned by the pipeline; `time` calendar timestamp (kept in the files, "
        "never learned); `train` learned by the SDV synthesizers; `derived` recomputed from the train columns "
        "by `schema.derive` (never learned). `sdtype` is the SDV type used in the metadata.",
        "",
    ]
    for table in schema.TABLES:
        rows = [
            [spec.name, spec.role, f"{spec.dtype} / {spec.sdtype}", spec.unit or "", _value_domain(spec), spec.description]
            for spec in schema.column_specs(table)
        ]
        lines += [f"**{table}** ({len(rows)} columns)", ""]
        lines += md_table(["column", "role", "dtype / sdtype", "unit", "allowed values", "description"], rows) + [""]
    return lines


def _scenario_support(table: str, frame: pd.DataFrame | None) -> dict[str, int]:
    if frame is None or not len(frame):
        return {}
    from bedsynth.scenarios import scenario_mask, scenarios_for

    try:
        derived = schema.derive(table, frame) if not set(schema.derived_columns(table)) <= set(frame.columns) else frame
        return {s.name: int(scenario_mask(derived, s).sum()) for s in scenarios_for(table)}
    except Exception as exc:  # noqa: BLE001 - support counts are informative only
        logger.info("Scenario support for %s not available: %s", table, exc)
        return {}


def _scenario_requirements_section(train: dict[str, pd.DataFrame | None], support: dict[str, dict[str, int]]) -> list[str]:
    """Brief scenario families -> catalogue scenarios, with their support in the train split."""
    from bedsynth.scenarios import SCENARIOS

    rows_total = {table: (len(frame) if frame is not None else None) for table, frame in train.items()}
    lines = ["### 1.2 Required scenarios and their reference support", "", SCENARIO_INTRO, ""]
    rows = []
    for family, categories in BRIEF_SCENARIO_FAMILIES:
        for scenario in SCENARIOS.values():
            if scenario.category not in categories:
                continue
            count = support.get(scenario.table, {}).get(scenario.name)
            total = rows_total.get(scenario.table)
            share = f" ({fmt_pct(count / total)})" if count is not None and total else ""
            definition = "; ".join(
                [f"{c} in {list(v)}" for c, v in scenario.conditions.items()]
                + [f"{c} {k} {v}" for c, spec in scenario.filters.items() for k, v in spec.items()]
            ) or "unconditional"
            rows.append([family, f"`{scenario.name}`", scenario.table, definition,
                         f"{fmt_int(count)}{share}" if count is not None else "-"])
    lines += md_table(["brief requirement", "scenario", "table", "definition", "train rows"], rows)
    return lines + [""]


def _source_section(manifest: dict[str, Any] | None) -> list[str]:
    lines = ["## 2. Reference source data (simulation)", ""]
    if manifest is None:
        return lines + ["_No simulation manifest found: the raw extract was provided externally._", ""]
    period = manifest.get("period") or {}
    counts = manifest.get("row_counts") or {}
    injection = manifest.get("data_quality_injection") or {}
    calibration = manifest.get("calibration") or {}
    lines += [
        "No real extract can be shared, so a seeded discrete-event simulation of the hospital "
        "(`bedsynth.data.simulator`) produced the reference extract "
        f"for {period.get('start_date')} .. {period.get('end_date')} ({fmt_int(period.get('days'))} days, "
        f"seed {manifest.get('seed')}), with {period.get('warmup_days')} warm-up and "
        f"{period.get('cooldown_days')} cool-down days.",
        "",
    ]
    lines += md_table(
        ["table", "clean rows (ground truth)", "raw rows (with injected issues)", "injected issues"],
        [[t, fmt_int((counts.get(t) or {}).get("clean")), fmt_int((counts.get(t) or {}).get("raw")),
          fmt_int(sum((injection.get("issues_by_table") or {}).get(t, {}).values()))] for t in schema.TABLES],
    )
    checks = calibration.get("checks") or []
    if checks:
        lines += [
            "",
            f"**Calibration:** {calibration.get('targets_passed')} of {calibration.get('targets_evaluated')} "
            "operational targets met (DESIGN.md §6.1).",
            "",
        ]
        rows = []
        for check in checks:
            target = check.get("target")
            target_text = f"{target[0]} - {target[1]}" if isinstance(target, list) and len(target) == 2 else str(target)
            rows.append([check.get("description") or check.get("name"), fmt_num(check.get("value"), 4), target_text,
                         "yes" if check.get("passed") else "**no**"])
        lines += md_table(["target", "value", "range", "met"], rows)
    return lines + [""]


def _exact_recovery(report: dict[str, Any], table: str, head: dict[str, Any]) -> Any:
    """Share of injected issues restored to their true value (older reports lack the headline field)."""
    if head.get("injected_issues_exact_recovery_rate") is not None:
        return head["injected_issues_exact_recovery_rate"]
    accuracy = ((report.get("tables") or {}).get(table) or {}).get("cleaning_accuracy") or {}
    return (accuracy.get("injected_issues") or {}).get("exact_recovery_rate")


#: Rows of the cleaning-results table: label and how to read the value from a summary entry.
_PIPELINE_METRICS: tuple[tuple[str, Any], ...] = (
    ("raw rows", lambda report, table, head: fmt_int(head.get("raw_rows"))),
    ("final rows", lambda report, table, head: fmt_int(head.get("final_rows"))),
    ("rows kept", lambda report, table, head: fmt_pct(head.get("retention_rate"))),
    ("issues detected", lambda report, table, head: fmt_int(head.get("issues_detected"))),
    ("cells repaired", lambda report, table, head: fmt_int(head.get("cells_repaired"))),
    ("rows dropped", lambda report, table, head: fmt_int(head.get("rows_dropped"))),
    ("hard-valid rate, raw -> final", lambda report, table, head:
        f"{fmt_pct(head.get('hard_valid_rate_raw'))} -> {fmt_pct(head.get('hard_valid_rate_final'))}"),
    ("cell agreement with ground truth", lambda report, table, head: fmt_pct(head.get("cell_agreement_with_truth"), 2)),
    ("injected issues no longer defective", lambda report, table, head: fmt_pct(head.get("injected_issues_resolved_rate"))),
    ("injected issues exactly recovered", lambda report, table, head: fmt_pct(_exact_recovery(report, table, head))),
    ("train / holdout rows", lambda report, table, head:
        f"{fmt_int(head.get('train_rows'))} / {fmt_int(head.get('holdout_rows'))}"),
)


def _pipeline_section(report: dict[str, Any]) -> list[str]:
    lines = [
        "## 3. Cleaning pipeline results",
        "",
        "`bedsynth.data.preprocessing.prepare_datasets` profiles the raw extract, standardises categories, "
        "parses ISO and day-first dates, removes duplicates, flags physically impossible values, repairs and "
        "imputes (most principled method first), derives, validates against the hard operational rules, "
        "checks dataset consistency and writes stratified train / holdout splits.",
        "",
    ]
    heads = report.get("summary") or {}
    tables = list(heads)
    if tables:
        rows = [[label, *[read(report, table, heads[table] or {}) for table in tables]] for label, read in _PIPELINE_METRICS]
        lines += md_table(["metric", *tables], rows)
    else:
        lines.append("_The data-quality report has no summary._")
    lines += [
        "",
        "*Hard-valid raw* counts every raw row after spellings and date formats were normalised but before any "
        "repair, so it measures operational defects rather than parsing gaps. An injected issue is *no longer "
        "defective* when the cleaned value is present, valid and differs from the injected one (it may be an "
        "imputed estimate); *exactly recovered* counts only cells restored to their true value - the stricter "
        "measure for imputed gaps.",
    ]
    return lines + [""]


def _issues_section(report: dict[str, Any], figure: Path | None, report_dir: Path) -> list[str]:
    lines = ["## 4. Data-quality issues: detected vs injected", ""]
    for table, info in (report.get("tables") or {}).items():
        detected = _detected_issues(info.get("issues_detected"))
        injected = info.get("issues_injected") or {}
        rows = []
        for kind in sorted(set(detected) | set(injected)):
            columns = sorted(set(detected.get(kind, {})) | set(injected.get(kind, {})))
            for column in columns:
                rows.append([kind, column, fmt_int(detected.get(kind, {}).get(column, 0)),
                             fmt_int(injected.get(kind, {}).get(column)) if injected else "-"])
        lines += [f"**{table}**", ""] + md_table(["issue type", "column", "detected", "injected"], rows) + [""]
    lines += [
        "Duplicates are counted per removed copy (`*` = whole row; exact and key duplicates combined). Detected "
        "counts can exceed injected ones because a duplicated row repeats the defects of its original until the "
        "copy is removed.",
        "",
    ]
    return lines + figure_line(figure, report_dir, "Issues injected by the simulator vs detected by the cleaning pipeline")


def _repairs_section(report: dict[str, Any]) -> list[str]:
    lines = ["## 5. Repairs and imputation", ""]
    for table, info in (report.get("tables") or {}).items():
        repairs = info.get("repairs") or {}
        kinds = repairs.get("method_kind") or {}
        rows = []
        for column, methods in sorted((repairs.get("by_column") or {}).items()):
            for method, count in sorted(methods.items(), key=lambda kv: -kv[1]):
                rows.append([column, method.replace("_", " "), kinds.get(method, "-"), fmt_int(count)])
        lines += [f"**{table}**: {fmt_int(repairs.get('cells_repaired'))} cells repaired "
                  f"({fmt_int(repairs.get('exact_cells'))} exactly recoverable, "
                  f"{fmt_int(repairs.get('approximate_cells'))} imputed)", ""]
        lines += md_table(["column", "method", "kind", "cells"], rows) if rows else ["_No repairs needed._"]
        lines.append("")
    return lines


def _validation_section(report: dict[str, Any]) -> list[str]:
    lines = ["## 6. Operational rule validation", "",
             "Hard rules (DESIGN.md §6.4) must hold on every row of the prepared data; soft rules are plausibility "
             "warnings and are only reported.", ""]
    for table, info in (report.get("tables") or {}).items():
        validation_info = info.get("validation") or {}
        before = (validation_info.get("before") or {}).get("hard_violations") or {}
        final = validation_info.get("final") or {}
        rows = [[rule, fmt_int(before.get(rule)), fmt_int(count)]
                for rule, count in sorted((final.get("hard_violations") or {}).items())]
        lines += [f"**{table}** - final hard-valid rate {fmt_pct(final.get('hard_valid_rate'))} "
                  f"({fmt_int(final.get('hard_valid_rows'))} of {fmt_int(final.get('rows'))} rows)", ""]
        lines += md_table(["hard rule", "violations in raw extract (formats normalised, before repair)",
                           "violations after cleaning"], rows)
        soft = final.get("soft_violations") or {}
        if soft:
            lines += ["", "Soft-rule flags after cleaning: " + ", ".join(f"{k} {fmt_int(v)}" for k, v in sorted(soft.items()))]
        lines.append("")
    return lines + _rule_catalogue_lines()


def _rule_catalogue_lines() -> list[str]:
    """The hard and soft rule catalogue with names and descriptions (DESIGN.md §6.4)."""
    lines = ["### Rule catalogue", "",
             "Hard rules reject a generated row (rejection sampling) and drop a raw row the cleaning cannot repair; "
             "soft rules are plausibility flags.", ""]
    for table in schema.TABLES:
        rows = [[rule["rule_id"], rule["severity"], rule["name"].replace("_", " "), rule["description"]]
                for rule in validation.rules_catalog(table)]
        lines += [f"**{table}**", ""] + md_table(["rule", "severity", "name", "description"], rows) + [""]
    return lines


def _consistency_section(report: dict[str, Any]) -> list[str]:
    checks = report.get("consistency_checks") or {}
    continuity = checks.get("ward_daily_continuity") or {}
    admissions = checks.get("cross_table_admissions") or {}
    return [
        "## 7. Dataset consistency",
        "",
        f"- Census continuity (`census_start(d) = census_end(d-1)` per ward): {fmt_pct(continuity.get('continuity_rate'))} "
        f"of {fmt_int(continuity.get('consecutive_pairs'))} consecutive ward-day pairs, "
        f"{fmt_int(continuity.get('missing_ward_days'))} missing ward-days.",
        f"- Cross-table admissions (stays admitted per ward/day/source vs ward_daily counts): "
        f"{fmt_pct(admissions.get('all_sources_match_rate'))} of ward-days match on every source.",
        "",
    ]


def _sdv_readiness_section(catalog: dict[str, Any] | None, files: dict[str, str]) -> list[str]:
    """SDV metadata files, sdtypes of the learned columns and the constraints declared per table."""
    lines = [
        "## 9. SDV readiness: metadata and constraints",
        "",
        "`metadata.json` describes both prepared CSV files exactly (key, timestamps, train and derived columns) "
        "and validates against them with `Metadata.validate_data`. `metadata_<table>.json` is the single-table "
        "metadata of the train columns - what every synthesizer fits: select `schema.training_columns(table)` "
        "from a CSV, then e.g. `GaussianCopulaSynthesizer(Metadata.load_from_json(...))`.",
        "",
    ]
    rows = []
    for table in schema.TABLES:
        specs = schema.column_specs(table, ("train",))
        sdtypes: dict[str, int] = {}
        for spec in specs:
            sdtypes[spec.sdtype] = sdtypes.get(spec.sdtype, 0) + 1
        file_name = files.get(f"metadata_{table}") or DEFAULT_OUTPUT_NAMES[f"metadata_{table}"]
        rows.append([table, f"`{file_name}`", len(specs), ", ".join(f"{k} {v}" for k, v in sorted(sdtypes.items())),
                     ", ".join(schema.column_names(table, ["key", "time"])), len(schema.derived_columns(table))])
    lines += md_table(["table", "training metadata", "learned columns", "sdtypes", "key / time (not learned)",
                       "derived (recomputed)"], rows) + [""]
    if not catalog:
        return lines + ["_The constraint catalogue is written to `schema_catalog.json` by `prepare_datasets`._", ""]
    for table in schema.TABLES:
        constraints = (catalog.get(table) or {}).get("constraints") or []
        if not constraints:
            continue
        rows = [[item.get("type"), item.get("implementation") or item.get("sdv_class"),
                 item.get("expression") or ", ".join(item.get("columns") or []),
                 ", ".join(item.get("enforces_rules") or []) or "-"] for item in constraints]
        lines += [f"**{table}** - {len(constraints)} SDV constraints declared for this table (each experiment applies "
                  "the groups it enables; per-model coverage is in the Week-2 report)", ""]
        lines += md_table(["constraint", "implementation", "expression / columns", "enforces rules"], rows) + [""]
        enforced = (catalog.get(table) or {}).get("rejection_enforced_rules") or []
        if enforced:
            lines += [f"All {len(enforced)} hard rules ({', '.join(enforced)}) are additionally enforced at "
                      "generation time by rejection sampling.", ""]
    return lines


def _figures_section(figures: list[tuple[Path | None, str]], report_dir: Path) -> list[str]:
    lines = ["## 8. Reference data patterns", "",
             "These operational patterns are what the synthesizers must reproduce (Week 3 evaluation).", ""]
    if not figures:
        return lines + ["_The cleaned data was not found in the processed-data directory: run "
                        "`python scripts/01_prepare_data.py` and re-render this report._", ""]
    for path, caption in figures:
        lines += figure_line(path, report_dir, caption)
    return lines


def _output_description(key: str, report: dict[str, Any]) -> str:
    """What one output file of ``prepare_datasets`` contains (row counts from the summary)."""
    heads = report.get("summary") or {}
    for table in schema.TABLES:
        head = heads.get(table) or {}
        if key == f"{table}_train":
            return f"SDV-ready train split ({fmt_int(head.get('train_rows'))} rows): what the synthesizers fit"
        if key == f"{table}_holdout":
            return (f"holdout split ({fmt_int(head.get('holdout_rows'))} rows), never used for training "
                    "(Week-3 evaluation)")
        if key == f"{table}_clean":
            return f"every prepared row ({fmt_int(head.get('final_rows'))}) before the split"
        if key == f"metadata_{table}":
            return f"SDV single-table metadata of the {len(schema.training_columns(table))} learned columns"
    return {
        "metadata": "SDV metadata of both CSV files (keys, timestamps, train and derived columns)",
        "schema_catalog": "schema catalogue: columns, SDV constraints and rules per table",
        "data_quality_report": "profiling, cleaning, validation and split results (the input of this report)",
    }.get(key, "-")


def _output_section(report: dict[str, Any], files: dict[str, str], raw_dir: Path | None,
                    processed_dir: Path | None) -> list[str]:
    """Where the inputs and the SDV-ready outputs live (project-relative paths)."""
    inside = processed_dir is not None and is_inside_project(processed_dir)
    where = (f"`{display_path(processed_dir)}/`, paths relative to the project root" if inside
             else "the configured processed-data directory (`paths.processed_dir` or `BEDSYNTH_PROCESSED_DIR`)")
    lines = ["## 10. Output files", "",
             f"`prepare_datasets` reads the raw extract and writes the SDV-ready dataset to {where}:", ""]
    heads = report.get("summary") or {}
    rows = []
    for table in schema.TABLES:
        recorded = ((report.get("tables") or {}).get(table) or {}).get("raw_file")
        raw = _resolve_file(recorded, raw_dir, f"{table}_raw.csv")
        if raw is not None or recorded:
            rows.append([f"`{display_path(raw or recorded)}`", "input",
                         f"raw {table} extract ({fmt_int((heads.get(table) or {}).get('raw_rows'))} rows)"])
    if raw_dir is not None and (raw_dir / MANIFEST_NAME).is_file():
        rows.append([f"`{display_path(raw_dir / MANIFEST_NAME)}`", "input",
                     "simulation manifest: seed, period, row counts, injected issues, calibration"])
    order = [f"{table}_{part}" for table in schema.TABLES for part in ("train", "holdout", "clean")]
    order += [f"metadata_{table}" for table in schema.TABLES] + ["metadata", "schema_catalog", "data_quality_report"]
    for key in [k for k in order if k in files] + sorted(k for k in files if k not in order):
        rows.append([f"`{files[key]}`", "output", _output_description(key, report)])
    if not rows:
        return lines + ["_No output files recorded._", ""]
    return lines + md_table(["file", "role", "content"], rows) + [""]


def render_week1_report(
    quality_report: dict[str, Any] | None = None,
    reports_dir: Path | str | None = None,
    figures_dir: Path | str | None = None,
) -> Path:
    """Write the Week-1 Markdown report (and figures) from a ``prepare_datasets`` report.

    Args:
        quality_report: The dict returned by ``prepare_datasets`` (also stored as
            ``data/processed/data_quality_report.json``); ``None`` reads that file from the
            configured ``processed_dir``.
        reports_dir, figures_dir: Output directories (default: configured ``reports_dir`` and
            ``figures_dir``).

    Returns:
        Path of ``week1_data_preparation.md``.
    """
    if quality_report is None:
        quality_report = _load_quality_report()
    report_dir, figure_dir = output_dirs(reports_dir, figures_dir)
    raw_dir = _resolve_dir(quality_report.get("raw_dir"), "raw_dir")
    processed_dir = _resolve_dir(quality_report.get("processed_dir"), "processed_dir")
    manifest = _load_manifest(raw_dir)
    files = _output_paths(quality_report, processed_dir)
    issues_fig = _issues_figure(quality_report, figure_dir / "week1_issues_detected_vs_injected.png")

    data_figures: list[tuple[Path | None, str]] = []
    ward_daily = _load_clean(processed_dir, schema.WARD_DAILY)
    if ward_daily is not None and len(ward_daily):
        data_figures.append((_occupancy_figure(ward_daily, figure_dir / "week1_occupancy_by_month.png"),
                             "Mean midnight occupancy by month and ward"))
        data_figures.append((_band_mix_figure(ward_daily, figure_dir / "week1_occupancy_band_mix.png"),
                             "Occupancy band mix by ward"))
    stays = _load_clean(processed_dir, schema.PATIENT_STAYS)
    if stays is not None and len(stays):
        data_figures.append((_los_figure(stays, figure_dir / "week1_los_by_source.png"),
                             "Length-of-stay distribution by admission source"))

    train = {table: _load_clean(processed_dir, table, "train") for table in schema.TABLES}
    support = {table: _scenario_support(table, train.get(table)) for table in schema.TABLES}
    lines = [
        "# BedSynth AI - Week 1: data requirements and dataset preparation",
        "",
        f"Generated {utc_now()} from `{QUALITY_REPORT_NAME}` (pipeline run "
        f"{to_utc(quality_report.get('generated_at')) or '-'}, seed {quality_report.get('seed')}, "
        f"{quality_report.get('elapsed_seconds')} s).",
        "",
    ]
    lines += _summary_section(quality_report, manifest, support, files)
    lines += _requirements_section()
    lines += _attribute_dictionary_section()
    lines += _scenario_requirements_section(train, support)
    lines += _source_section(manifest)
    lines += _pipeline_section(quality_report)
    lines += _issues_section(quality_report, issues_fig, report_dir)
    lines += _repairs_section(quality_report)
    lines += _validation_section(quality_report)
    lines += _consistency_section(quality_report)
    lines += _figures_section(data_figures, report_dir)
    lines += _sdv_readiness_section(_load_schema_catalog(processed_dir), files)
    lines += _output_section(quality_report, files, raw_dir, processed_dir)
    return write_markdown(report_dir / REPORT_NAME, lines)


__all__ = ["REPORT_NAME", "render_week1_report"]
