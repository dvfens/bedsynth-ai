"""Week 1 - prepare the reference datasets (DESIGN.md §6, §11).

1. Simulate the hospital source extract with ``bedsynth.data.simulator.generate_source_data``
   - only when the raw files are missing, or when ``--resimulate`` is given.
2. Clean, repair, validate and split it with ``bedsynth.data.preprocessing.prepare_datasets``
   (writes data/processed/*.csv, metadata.json, schema_catalog.json, data_quality_report.json).
3. Print a concise summary of the data-quality report.
4. Optional hook: render the Week-1 report with ``bedsynth.reporting`` when that package exists.

Usage (from the project root)::

    .venv\\Scripts\\python.exe scripts\\01_prepare_data.py
    .venv\\Scripts\\python.exe scripts\\01_prepare_data.py --resimulate --seed 7
    .venv\\Scripts\\python.exe scripts\\01_prepare_data.py --resimulate --no-inject

Importing this module has no side effects: ``bedsynth`` is imported inside the functions so
the script also runs without the package being installed.
"""
from __future__ import annotations

import argparse
import importlib
import inspect
import logging
import sys
import time
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Renderer names looked up in ``bedsynth.reporting`` by the optional Week-1 report hook.
WEEK1_RENDERERS: tuple[str, ...] = ("render_week1_report", "write_week1_report", "build_week1_report")


def build_parser() -> argparse.ArgumentParser:
    """Command-line interface of the Week-1 preparation step."""
    parser = argparse.ArgumentParser(
        description="Simulate (if needed), clean, validate and split the BedSynth reference data.",
    )
    parser.add_argument("--resimulate", action="store_true",
                        help="regenerate the raw source extract even if it already exists")
    parser.add_argument("--no-inject", action="store_true",
                        help="when (re)simulating, skip dirty-data injection (raw extract = ground truth)")
    parser.add_argument("--seed", type=int, default=None,
                        help="random seed for simulation and the train/holdout split (default: settings.yaml)")
    parser.add_argument("--no-report", action="store_true",
                        help="do not call the optional bedsynth.reporting Week-1 report hook")
    return parser


def _bootstrap_import_path() -> None:
    """Make ``import bedsynth`` work when the script is run from a source checkout."""
    root = str(PROJECT_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def missing_raw_files(raw_dir: Path) -> list[Path]:
    """Raw extract files (``{table}_raw.csv``) that do not exist yet."""
    from bedsynth.data.io import raw_path
    from bedsynth.schema import TABLES

    return [path for path in (raw_path(table, raw_dir) for table in TABLES) if not path.exists()]


def ensure_raw_data(raw_dir: Path, resimulate: bool = False, inject: bool = True,
                    seed: int | None = None) -> dict[str, Any] | None:
    """Run the simulator when raw files are missing (or on request); return its manifest.

    Returns None when the existing raw extract is reused.
    """
    missing = missing_raw_files(raw_dir)
    if not resimulate and not missing:
        print(f"Using existing raw extract in {raw_dir}")
        if not inject:
            print("  note: --no-inject only applies when the extract is (re)generated; add --resimulate")
        return None
    reason = "--resimulate requested" if resimulate else "missing " + ", ".join(p.name for p in missing)
    print(f"Simulating the source extract ({reason}; dirty-data injection {'on' if inject else 'off'}) ...")
    from bedsynth.data.simulator import generate_source_data

    started = time.perf_counter()
    manifest = generate_source_data(raw_dir=raw_dir, seed=seed, inject_issues=inject)
    print(f"  simulation finished in {time.perf_counter() - started:.1f} s")
    return manifest


def _pct(value: float | None) -> str:
    """Format a share as a percentage ('n/a' when undefined)."""
    return "n/a" if value is None else f"{100.0 * value:.1f} %"


def format_summary(report: dict[str, Any]) -> str:
    """Concise, human-readable summary of a ``prepare_datasets`` report."""
    lines = [f"BedSynth AI - Week 1 data preparation (seed {report['seed']}, {report['elapsed_seconds']} s)"]
    for table, head in report["summary"].items():
        dropped = report["tables"][table]["rows_dropped"]
        reasons = ", ".join(f"{k.replace('_', ' ')} {v}" for k, v in dropped.items() if k != "total" and v)
        lines += [
            f"  {table}",
            f"    rows       {head['raw_rows']:,} raw -> {head['final_rows']:,} final "
            f"({_pct(head['retention_rate'])} kept; dropped {head['rows_dropped']:,}{': ' + reasons if reasons else ''})",
            f"    quality    {head['issues_detected']:,} issues detected, {head['cells_repaired']:,} cells repaired; "
            f"hard-valid {_pct(head['hard_valid_rate_raw'])} raw -> {_pct(head['hard_valid_rate_final'])} final",
            f"    split      {head['train_rows']:,} train / {head['holdout_rows']:,} holdout",
        ]
        if head["cell_agreement_with_truth"] is not None:
            lines.append(
                f"    accuracy   {_pct(head['cell_agreement_with_truth'])} cell agreement with ground truth; "
                f"{_pct(head['injected_issues_resolved_rate'])} of injected issues no longer defective, "
                f"{_pct(head.get('injected_issues_exact_recovery_rate'))} exactly recovered"
            )
    checks = report["consistency_checks"]
    lines.append(
        f"  checks       census continuity {_pct(checks['ward_daily_continuity']['continuity_rate'])}; "
        f"cross-table admissions match {_pct(checks['cross_table_admissions']['all_sources_match_rate'])}"
    )
    lines.append(f"  outputs      {report['processed_dir']} (report: {Path(report['outputs']['data_quality_report']).name})")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# OPTIONAL HOOK - Week-1 report rendering.
# bedsynth.reporting is built separately. When it is importable and exposes one of
# WEEK1_RENDERERS, that callable receives the data-quality report dict (if it accepts a
# positional argument) and should return the path of the rendered report. Data
# preparation has already succeeded at this point, so failures here only print a warning.
# ---------------------------------------------------------------------------
def render_week1_report(report: dict[str, Any]) -> Path | None:
    """Render the Week-1 report via ``bedsynth.reporting`` if available (else skip)."""
    try:
        reporting = importlib.import_module("bedsynth.reporting")
    except ModuleNotFoundError as exc:
        detail = "not available" if exc.name == "bedsynth.reporting" else f"import failed ({exc})"
        print(f"Week-1 report: bedsynth.reporting {detail} - skipped.")
        return None
    renderer = next((getattr(reporting, n) for n in WEEK1_RENDERERS if callable(getattr(reporting, n, None))), None)
    if renderer is None:
        print(f"Week-1 report: bedsynth.reporting exposes none of {list(WEEK1_RENDERERS)} - skipped.")
        return None
    try:
        positional = [p for p in inspect.signature(renderer).parameters.values()
                      if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD, p.VAR_POSITIONAL)]
        result = renderer(report) if positional else renderer()
    except Exception as exc:  # noqa: BLE001 - optional step; the prepared data stays valid
        print(f"Week-1 report: {renderer.__name__} failed ({type(exc).__name__}: {exc}) - skipped.")
        return None
    if result:
        print(f"Week-1 report written to {result}")
        return Path(result)
    return None


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point; returns a process exit code."""
    args = build_parser().parse_args(argv)
    _bootstrap_import_path()
    # Progress messages from bedsynth only; third-party INFO chatter (e.g. SDV's metadata
    # banner) stays hidden because those loggers inherit the root WARNING level.
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("bedsynth").setLevel(logging.INFO)

    import bedsynth  # noqa: F401  (sets TQDM_DISABLE before SDV is imported)
    from bedsynth.config import get_path
    from bedsynth.data.preprocessing import prepare_datasets

    raw_dir = get_path("raw_dir")
    try:
        ensure_raw_data(raw_dir, resimulate=args.resimulate, inject=not args.no_inject, seed=args.seed)
    except ImportError as exc:
        print(f"error: cannot simulate the source extract - {exc}", file=sys.stderr)
        return 1
    report = prepare_datasets(raw_dir=raw_dir, seed=args.seed)
    print()
    print(format_summary(report))
    if not args.no_report:
        render_week1_report(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
