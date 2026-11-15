"""Run the whole BedSynth AI pipeline end to end (DESIGN.md §11).

Steps (each runs its script in a fresh interpreter, stopping at the first failure):

==========  ===================================  ===============================================
step        script                               produces
==========  ===================================  ===============================================
prepare     ``01_prepare_data.py``               data/raw (simulated if missing), data/processed,
                                                 reports/week1_data_preparation.md
train       ``02_train_synthesizers.py``         models/ + registry.json, reports/week2_*.md
evaluate    ``03_evaluate.py``                   reports/results/*.json, registry defaults,
                                                 reports/week3_evaluation.md
demo        ``05_api_demo.py``                   reports/api_demo/, reports/week4_api_system.md
==========  ===================================  ===============================================

Usage (from the project root)::

    .venv\\Scripts\\python.exe scripts\\run_pipeline.py --quick          # fast end-to-end run (~16 min)
    .venv\\Scripts\\python.exe scripts\\run_pipeline.py                  # full training and evaluation
    .venv\\Scripts\\python.exe scripts\\run_pipeline.py --steps train evaluate --quick
    .venv\\Scripts\\python.exe scripts\\run_pipeline.py --steps demo --run-tests --run-slow-tests

``--quick`` trains CTGAN/TVAE for 10 epochs (``synthesis.quick_mode_params``) and evaluates
on at most 2,000 rows per model. ``--run-tests`` / ``--run-slow-tests`` (off by default) are
passed to the demo step, which then first runs the fast / slow test suite and refreshes the
JUnit evidence (``reports/results/pytest_junit.xml`` / ``pytest_junit_slow.xml``); ``--dry-run``
prints every command with its options. Serve the API afterwards with ``scripts/04_serve_api.py``.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

STEPS: dict[str, str] = {
    "prepare": "01_prepare_data.py",
    "train": "02_train_synthesizers.py",
    "evaluate": "03_evaluate.py",
    "demo": "05_api_demo.py",
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the BedSynth AI pipeline (prepare -> train -> evaluate -> demo).")
    parser.add_argument("--steps", nargs="+", choices=list(STEPS), default=list(STEPS),
                        help="steps to run, in pipeline order (default: all)")
    parser.add_argument("--quick", action="store_true", help="quick training (10 GAN/VAE epochs) and quick evaluation")
    parser.add_argument("--resimulate", action="store_true", help="regenerate the raw source extract")
    parser.add_argument("--seed", type=int, default=None, help="seed for simulation, split and evaluation")
    parser.add_argument("--run-tests", action="store_true",
                        help="demo step: run the fast test suite first (refreshes reports/results/pytest_junit.xml)")
    parser.add_argument("--run-slow-tests", action="store_true",
                        help="demo step: run the slow test suite first (several minutes; refreshes "
                             "reports/results/pytest_junit_slow.xml)")
    parser.add_argument("--dry-run", action="store_true", help="print the commands without running them")
    return parser.parse_args(argv)


def build_commands(args: argparse.Namespace) -> list[tuple[str, list[str]]]:
    """``[(step, argv)]`` for the selected steps, in pipeline order."""
    commands: list[tuple[str, list[str]]] = []
    for step in STEPS:
        if step not in args.steps:
            continue
        argv = [sys.executable, str(SCRIPTS_DIR / STEPS[step])]
        if step == "prepare":
            argv += ["--resimulate"] if args.resimulate else []
            argv += ["--seed", str(args.seed)] if args.seed is not None else []
        elif step == "train":
            argv += ["--quick"] if args.quick else []
        elif step == "evaluate":
            argv += ["--quick"] if args.quick else []
            argv += ["--seed", str(args.seed)] if args.seed is not None else []
        elif step == "demo":
            argv += ["--seed", str(args.seed)] if args.seed is not None else []
            argv += ["--run-tests"] if getattr(args, "run_tests", False) else []
            argv += ["--run-slow-tests"] if getattr(args, "run_slow_tests", False) else []
        commands.append((step, argv))
    return commands


def run(commands: list[tuple[str, list[str]]], dry_run: bool = False) -> tuple[int, dict[str, float]]:
    """Run the commands in order; returns ``(exit code, {step: seconds})``."""
    timings: dict[str, float] = {}
    for step, argv in commands:
        print(f"\n=== {step}: {' '.join(Path(a).name if i < 2 else a for i, a in enumerate(argv))}", flush=True)
        if dry_run:
            continue
        started = time.perf_counter()
        completed = subprocess.run(argv, cwd=PROJECT_ROOT, check=False)
        timings[step] = round(time.perf_counter() - started, 1)
        print(f"=== {step} finished in {timings[step]} s (exit code {completed.returncode})", flush=True)
        if completed.returncode != 0:
            print(f"Pipeline stopped: step '{step}' failed.", file=sys.stderr)
            return completed.returncode, timings
    return 0, timings


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    started = time.perf_counter()
    if (args.run_tests or args.run_slow_tests) and "demo" not in args.steps:
        print("Note: --run-tests / --run-slow-tests only apply to the demo step, which is not selected.", flush=True)
    code, timings = run(build_commands(args), dry_run=args.dry_run)
    if timings:
        print("\nStep timings: " + ", ".join(f"{step} {seconds} s" for step, seconds in timings.items())
              + f"; total {time.perf_counter() - started:.1f} s")
    if code == 0 and not args.dry_run:
        print("Done. Serve the API with: python scripts/04_serve_api.py  (then open /ui or /docs)")
    return code


if __name__ == "__main__":
    sys.exit(main())
