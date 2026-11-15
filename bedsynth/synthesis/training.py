"""Training entry points - DESIGN.md §7.5.

``train_experiment`` fits one experiment (settings.yaml ``synthesis.experiments``) on the
processed train split (or a given frame), saves it under the model registry layout and
registers its model card. GAN/VAE experiments are trained on a seeded, ward-stratified
subsample of at most ``synthesis.max_training_rows[table]`` rows (their cost grows with the
number of rows; GaussianCopula always uses every row). ``quick=True`` applies
``synthesis.quick_mode_params`` (e.g. 10 epochs) for fast end-to-end runs and CI; it does not
change GaussianCopula marginal selection (the evaluated quick models use the full candidate list).

``train_experiment`` resolves per-table settings (``segment_by: {ward_daily: ward_id,
patient_stays: [ward_id, admission_source]}``) for the table being trained and records the
segment columns and constraint groups in the card's ``training`` block.

After fitting, :func:`sample_diagnostics` draws an unfiltered sample and records how often the
model produces operationally valid rows on its own (``raw_hard_valid_rate``, the violation
rate of every hard rule and of every soft rule). The value documents the model in its card
(the Week-2 report's constraint ablation reads it) and lets the generator size its first
rejection-sampling batch without a pilot round.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import pandas as pd

from bedsynth import schema, validation
from bedsynth.config import get_seed, get_settings, project_relative
from bedsynth.data import io
from bedsynth.schema import TABLES
from bedsynth.synthesis.models import BedSynthSynthesizer, ExperimentConfig, derive_seed, largest_remainder
from bedsynth.synthesis.registry import ModelRegistry

LOGGER = logging.getLogger(__name__)

SUBSAMPLED_MODELS: frozenset[str] = frozenset({"ctgan", "tvae", "copulagan"})
"""Model families trained on at most ``synthesis.max_training_rows`` rows."""
DIAGNOSTIC_ROWS = 2_000
"""Unfiltered rows drawn after fitting to measure the model's raw validity."""


def max_training_rows(table: str, model: str) -> int | None:
    """Row cap for ``model`` on ``table`` (``None`` = use every row)."""
    if model not in SUBSAMPLED_MODELS:
        return None
    caps = get_settings().get("synthesis", {}).get("max_training_rows") or {}
    cap = caps.get(table)
    return int(cap) if cap else None


def subsample_training_rows(
    df: pd.DataFrame, max_rows: int | None, seed: int, stratify_by: str | None = "ward_id"
) -> pd.DataFrame:
    """Seeded subsample of at most ``max_rows`` rows that preserves the strata shares.

    Rows are apportioned to strata (wards) with the largest-remainder method and drawn
    without replacement; the original row order is kept.
    """
    if max_rows is None or len(df) <= max_rows:
        return df
    if stratify_by is None or stratify_by not in df.columns:
        return df.sample(n=max_rows, random_state=derive_seed(seed, "subsample")).sort_index()
    counts = df[stratify_by].value_counts(sort=False)
    allocation = largest_remainder(max_rows, counts.to_numpy())
    parts = [
        df.loc[df[stratify_by] == key].sample(n=n, random_state=derive_seed(seed, "subsample", key))
        for key, n in zip(counts.index, allocation)
        if n > 0
    ]
    return pd.concat(parts).sort_index()


def sample_diagnostics(model: BedSynthSynthesizer, num_rows: int = DIAGNOSTIC_ROWS, seed: int | None = None) -> dict[str, Any]:
    """Raw validity of a fitted model: rule pass rates of an unfiltered, derived sample.

    Returns a JSON-serialisable dict (``rows, raw_hard_valid_rate, hard_violation_rates,
    soft_violation_rates, seconds``); never raises (failures are reported under ``error``).
    """
    started = time.perf_counter()
    try:
        raw = model.sample(num_rows, seed=derive_seed(seed if seed is not None else get_seed(), "diagnostics"))
        derived = schema.derive(model.table, raw)
        report = validation.validate_table(derived, model.table, include_soft=True)
        rows = len(derived)

        def rates(severity: str) -> dict[str, float]:
            return {
                r["rule_id"]: round(float(r["violation_rate"]), 6)
                for r in report.rule_results
                if r.get("severity") == severity and r.get("violations")
            }

        return {
            "rows": rows,
            "raw_hard_valid_rate": round(report.n_hard_valid / rows, 6) if rows else None,
            "hard_violation_rates": rates("hard"),
            "soft_violation_rates": rates("soft"),
            "seconds": round(time.perf_counter() - started, 3),
        }
    except Exception as exc:  # noqa: BLE001 - diagnostics must not fail training
        LOGGER.warning("Sample diagnostics failed for %r: %s", model, exc)
        return {"rows": 0, "raw_hard_valid_rate": None, "error": f"{type(exc).__name__}: {exc}"}


def train_experiment(
    table: str,
    experiment_id: str,
    train_df: pd.DataFrame | None = None,
    quick: bool = False,
    models_dir: Path | str | None = None,
    source: str | None = None,
) -> dict[str, Any]:
    """Fit one experiment on ``train_df`` (default: processed train split), save it under the
    registry and return its model card.

    ``source`` optionally describes where ``train_df`` came from (recorded in the model card as
    ``training.source``; a file path is stored project-relative, see
    :func:`bedsynth.config.project_relative`). The returned card's ``path`` (the model
    directory) is project-relative too.
    """
    if table not in TABLES:
        raise KeyError(f"Unknown table '{table}'. Known tables: {list(TABLES)}")
    config = ExperimentConfig.from_settings(experiment_id, quick=quick, table=table)
    source = source or "provided DataFrame"
    if train_df is None:
        train_df = io.load_processed(table, "train")
        source = project_relative(io.processed_path(table, "train"))
    seed = get_seed()
    cap = max_training_rows(table, config.model)
    data = subsample_training_rows(train_df, cap, seed)
    if len(data) < len(train_df):
        LOGGER.info("%s/%s: training on a stratified subsample of %d of %d rows",
                    table, experiment_id, len(data), len(train_df))

    started = time.perf_counter()
    # Quick mode only shortens GAN/VAE training (quick_mode_params): GaussianCopula marginal
    # selection keeps the experiment's full candidate list, because the quick models are the ones
    # evaluated and shipped and symmetric-only marginals mis-shape skewed columns.
    model = BedSynthSynthesizer(table, config, seed=seed).fit(data)
    model.annotate(
        source=source,
        quick=quick,
        input_rows=len(train_df),
        training_rows=len(data),
        max_training_rows=cap,
        subsampled=len(data) < len(train_df),
        sdv_params=config.params,
        segment_columns=list(model.segment_columns),
        constraint_groups=list(model.constraint_groups),
    )
    model.annotate(sample_diagnostics=sample_diagnostics(model, seed=seed))
    registry = ModelRegistry(models_dir)
    directory = model.save(registry.model_dir(table, experiment_id))
    card = model.model_card()
    registry.register(table, experiment_id, card)
    LOGGER.info("Trained %s/%s in %.1fs -> %s", table, experiment_id, time.perf_counter() - started, directory)
    return {**card, "path": project_relative(directory)}


def train_all(
    tables: list[str] | None = None,
    experiments: list[str] | None = None,
    quick: bool = False,
    models_dir: Path | str | None = None,
    continue_on_error: bool = True,
) -> list[dict[str, Any]]:
    """Train every requested ``table x experiment`` combination.

    Args:
        tables: Tables to train (default: all).
        experiments: Experiment ids (default: every experiment in settings.yaml).
        quick: Apply ``synthesis.quick_mode_params``.
        models_dir: Registry root (default: configured models dir).
        continue_on_error: Log a failed experiment and continue (its entry then has
            ``status="failed"`` and ``error``); otherwise re-raise.

    Returns:
        One item per experiment: the model card plus ``status`` (``"trained"``/``"failed"``).
    """
    tables = list(tables or TABLES)
    experiments = list(experiments or get_settings()["synthesis"]["experiments"])
    results: list[dict[str, Any]] = []
    for table in tables:
        train_df: pd.DataFrame | None = None
        for experiment_id in experiments:
            try:
                if train_df is None:
                    train_df = io.load_processed(table, "train")
                card = train_experiment(
                    table, experiment_id, train_df=train_df, quick=quick, models_dir=models_dir,
                    source=project_relative(io.processed_path(table, "train")),
                )
                results.append({**card, "status": "trained"})
            except Exception as exc:  # noqa: BLE001 - report and continue with the next experiment
                if not continue_on_error:
                    raise
                LOGGER.exception("Training %s/%s failed", table, experiment_id)
                results.append({
                    "table": table,
                    "experiment_id": experiment_id,
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                })
    return results


__all__ = ["max_training_rows", "sample_diagnostics", "subsample_training_rows", "train_all", "train_experiment"]
