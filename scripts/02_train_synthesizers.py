"""Weeks 2-3 - train the SDV synthesis experiments and register them (DESIGN.md §7.5, §11).

For every requested ``table x experiment`` (settings.yaml ``synthesis.experiments``) the
script fits a :class:`bedsynth.synthesis.models.BedSynthSynthesizer` on the processed train
split, saves ``models/<table>/<experiment>/{synthesizer.pkl, model_card.json}``, registers the
model card in ``models/registry.json`` and prints a summary table of the model cards.

Usage (from the project root)::

    .venv\\Scripts\\python.exe scripts\\02_train_synthesizers.py
    .venv\\Scripts\\python.exe scripts\\02_train_synthesizers.py --tables ward_daily --experiments gc_baseline gc_tuned
    .venv\\Scripts\\python.exe scripts\\02_train_synthesizers.py --experiments gc_baseline gc_segmented gc_tuned  # ablation
    .venv\\Scripts\\python.exe scripts\\02_train_synthesizers.py --quick        # 10 epochs for CTGAN/TVAE

The summary table lists each model's constraint groups and its raw hard-valid rate (share of
unfiltered model samples that pass every hard rule), so the ablation
``gc_baseline -> gc_segmented -> gc_tuned`` is visible right after training.

The processed splits must exist (``scripts/01_prepare_data.py``). Exit code 1 when any
experiment failed. Importing this module has no side effects: ``bedsynth`` is imported inside
the functions so the script also runs without the package being installed.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Third-party loggers that are very chatty at INFO level during SDV fitting.
NOISY_LOGGERS: tuple[str, ...] = (
    "sdv", "rdt", "copulas", "ctgan", "SingleTableSynthesizer", "MultiTableSynthesizer", "sdmetrics",
)


def _bootstrap() -> None:
    """Make ``bedsynth`` importable from a source checkout and silence progress bars."""
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    import bedsynth  # noqa: F401  (sets TQDM_DISABLE before SDV is imported)


def build_parser() -> argparse.ArgumentParser:
    """Command-line interface of the training step."""
    _bootstrap()
    from bedsynth.config import get_settings
    from bedsynth.schema import TABLES

    experiments = list(get_settings()["synthesis"]["experiments"])
    parser = argparse.ArgumentParser(
        description="Train BedSynth SDV synthesizers and register them in the model registry.",
    )
    parser.add_argument("--tables", nargs="+", choices=list(TABLES), default=list(TABLES),
                        help="Tables to train (default: all).")
    parser.add_argument("--experiments", nargs="+", choices=experiments, default=experiments,
                        help="Experiments from settings.yaml synthesis.experiments (default: all).")
    parser.add_argument("--quick", action="store_true",
                        help="Apply synthesis.quick_mode_params (e.g. 10 epochs for CTGAN/TVAE).")
    parser.add_argument("--models-dir", type=Path, default=None,
                        help="Override the models directory (default: settings paths.models_dir).")
    parser.add_argument("--stop-on-error", action="store_true",
                        help="Abort on the first failed experiment instead of continuing.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--no-report", action="store_true",
                        help="Do not render the Week-2 pipeline report (bedsynth.reporting).")
    return parser


def configure_logging(level: str) -> None:
    """Root logging at ``level``; SDV/rdt/copulas loggers at WARNING (they are chatty)."""
    logging.basicConfig(level=getattr(logging, level), format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    for name in NOISY_LOGGERS:
        logging.getLogger(name).setLevel(max(logging.WARNING, getattr(logging, level)))


def _distribution_summary(card: dict[str, Any]) -> str:
    """'truncnorm x5, beta x3' for segmented models, the single choice otherwise."""
    chosen = [d for d in (card.get("distributions") or {}).values() if d]
    if not chosen:
        return "-"
    counts: dict[str, int] = {}
    for name in chosen:
        counts[name] = counts.get(name, 0) + 1
    if len(chosen) == 1:
        return chosen[0]
    return ", ".join(f"{name} x{n}" for name, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


def _constraint_summary(card: dict[str, Any]) -> str:
    """'all (6)', 'none' or the enabled constraint groups joined with '+'.

    When the model applies fewer groups than it enables (e.g. a ward x source model, whose
    segment columns make some constraints unnecessary), the applied groups follow in brackets.
    """
    groups = card.get("constraint_groups")
    if not isinstance(groups, list):
        return "-"
    if not groups:
        return "none"
    _bootstrap()
    from bedsynth.synthesis.constraints import CONSTRAINT_GROUPS

    text = f"all ({len(groups)})" if set(groups) == set(CONSTRAINT_GROUPS) else "+".join(str(g) for g in groups)
    applied = card.get("effective_constraint_groups")
    if isinstance(applied, list) and set(applied) != set(groups):
        text += f" [{len(applied)} applied]"
    return text


def format_cards(cards: Sequence[dict[str, Any]]) -> str:
    """Plain-text table of model cards (one line per experiment)."""
    header = ("table", "experiment", "model", "status", "rows", "segments", "constraints", "marginals",
              "valid%", "fit_s")
    rows = []
    for card in cards:
        diagnostics = (card.get("training") or {}).get("sample_diagnostics") or {}
        valid = diagnostics.get("raw_hard_valid_rate")
        rows.append((
            str(card.get("table", "")),
            str(card.get("experiment_id", "")),
            str(card.get("model", "-")),
            str(card.get("status", "trained")),
            str(card.get("training_rows", "-")),
            str(card.get("n_segments", "-")),
            _constraint_summary(card),
            _distribution_summary(card) if card.get("status", "trained") == "trained" else "-",
            f"{100 * valid:.1f}" if isinstance(valid, (int, float)) else "-",
            f"{card['fit_seconds']:.1f}" if isinstance(card.get("fit_seconds"), (int, float)) else "-",
        ))
    widths = [max(len(h), *(len(r[i]) for r in rows)) if rows else len(h) for i, h in enumerate(header)]
    lines = ["  ".join(h.ljust(w) for h, w in zip(header, widths)), "  ".join("-" * w for w in widths)]
    lines += ["  ".join(v.ljust(w) for v, w in zip(row, widths)) for row in rows]
    for card in cards:
        if card.get("status") == "failed":
            lines.append(f"FAILED {card.get('table')}/{card.get('experiment_id')}: {card.get('error')}")
    return "\n".join(lines)


def render_week2_report(cards: Sequence[dict[str, Any]], models_dir: Path | None = None) -> Path | None:
    """Optional hook: write ``reports/week2_synthesis_pipeline.md`` from the registry.

    The report covers every trained model of the registry (not only this run's), so a partial
    re-training keeps the report complete; failures of this run are listed as well. Training
    has already succeeded at this point, so a reporting failure only prints a warning.
    """
    try:
        from bedsynth.reporting import render_week2_report as renderer
        from bedsynth.reporting.week2 import registry_cards

        failed = [card for card in cards if card.get("status") == "failed"]
        path = renderer(registry_cards(models_dir) + failed)
    except Exception as exc:  # noqa: BLE001 - optional step
        print(f"Week-2 report skipped ({type(exc).__name__}: {exc})")
        return None
    print(f"Week-2 report written to {path}")
    return path


def main(argv: Sequence[str] | None = None) -> int:
    """Train the requested experiments; returns the process exit code."""
    args = build_parser().parse_args(argv)
    configure_logging(args.log_level)
    from bedsynth.data.io import processed_path
    from bedsynth.synthesis.registry import ModelRegistry
    from bedsynth.synthesis.training import train_all

    missing = [t for t in args.tables if not processed_path(t, "train").exists()]
    if missing:
        print(f"Processed train split missing for {missing}: run `python scripts/01_prepare_data.py` first.",
              file=sys.stderr)
        return 1

    started = time.perf_counter()
    print(f"Training {len(args.tables)} table(s) x {len(args.experiments)} experiment(s)"
          f"{' (quick mode)' if args.quick else ''} ...")
    cards = train_all(
        tables=args.tables,
        experiments=args.experiments,
        quick=args.quick,
        models_dir=args.models_dir,
        continue_on_error=not args.stop_on_error,
    )
    registry = ModelRegistry(args.models_dir)
    print()
    print(format_cards(cards))
    print()
    for table in args.tables:
        print(f"Default model for {table}: {registry.default_model_id(table)}")
    print(f"Registry: {registry.path}")
    print(f"Total time: {time.perf_counter() - started:.1f}s")
    if not args.no_report:
        render_week2_report(cards, args.models_dir)
    return 1 if any(card.get("status") == "failed" for card in cards) else 0


if __name__ == "__main__":
    sys.exit(main())
