"""Week-3 evaluation: generate data with every trained model, evaluate it, select defaults.

Usage (from the project root)::

    .venv/Scripts/python.exe scripts/03_evaluate.py                       # every table and model
    .venv/Scripts/python.exe scripts/03_evaluate.py --tables ward_daily --models gc_tuned ctgan
    .venv/Scripts/python.exe scripts/03_evaluate.py --quick               # fast end-to-end check
    .venv/Scripts/python.exe scripts/03_evaluate.py --sync-registry       # only copy stored scores to the registry

For each table:

1. load the real train / holdout splits (``io.load_processed``) and fix the evaluated size
   N = ``--rows``, else ``settings.evaluation.synthetic_rows``, else the size of the training
   split, capped at ``settings.evaluation.max_synthetic_rows`` (20,000: patient_stays has
   45,002 training rows; ML utility compares TSTR with a TRTR trained on as many real rows,
   so the cap does not penalise any model);
2. calibrate the real-vs-real noise of the operational checks at size N
   (``evaluation.calibrate_operational``: random re-splits of the pooled real data, train +
   holdout, into a pseudo training split and N real rows of the remainder), so that a real
   sample of the evaluated size passes the checks;
3. score two reference pseudo-models with the same families (``evaluation.references``):
   ``real_holdout`` (ceiling: unseen real rows inside the same generation system) and
   ``null_stratified_marginals`` (floor: no learning);
4. for each trained model in the registry:

   a. generate N validated rows through ``SyntheticDataGenerator`` in chunks of
      ``settings.generation.max_rows``, each request with ``evaluation.sample_time_budget_seconds``;
      the summary keeps shortfall, early stops, the allocation-strata block and the TVD of
      the output's strata mix against the training split (flagged above
      ``evaluation.strata_tvd_flag``: the generator moved quota to strata the model produces
      more easily);
   b. draw N raw, *unfiltered* model samples (``SyntheticDataGenerator.sample_raw``) for the
      validity score and the model view;
   c. run ``bedsynth.evaluation.evaluate_model`` - scenario requests carry
      ``evaluation.scenario_time_budget_seconds`` / ``scenario_max_attempts`` and the whole
      coverage run ``evaluation.coverage_time_budget_seconds`` (scenarios not started in time
      are reported as not evaluated) - and write ``reports/results/<table>__<model_id>.json``.

Then ``reports/results/model_comparison.json`` (models, references, normalised scores, run
configuration) is written and each table's registry default is set to the best composite
among entries that are **complete** (no failed family), **comparable** (same run
configuration as this run) and **current** (evaluated on the model version in the registry).
A model that fails in this run is recorded as failed instead of keeping an older entry.
``--quick`` caps N at 2,000 rows, uses 100 rows per scenario, skips the slow sdmetrics
privacy cross-check and applies the tighter ``evaluation.quick_overrides`` time caps.

Finally every model with an entry in ``model_comparison.json`` gets ``metrics`` in its registry
card (``registry.update_card``; :data:`REGISTRY_METRIC_KEYS`): composite, complete, the five
components, the raw hard-valid rate, operational checks passed / scored, scenario coverage,
``rank`` (1 = best comparable model of its table), comparable, evaluated_at, run_quick, rows
and ``source`` (the comparison file). Every value is copied from the comparison entry and the
model's result file - nothing is re-scored - and an entry evaluated on an older version of the
model (card ``fitted_at`` differs from the entry's ``model_fitted_at``) is skipped.
``--sync-registry`` runs only this step from the stored files, without generating or
evaluating anything.

Runtime (development machine, 5 models x 2 tables): ``--quick`` took 596-603 s on the final
quick-trained models of the integration run (868 s for the models before the Week-3 improvement
round, which had no per-request time budgets); its time goes mostly to the 26 scenario requests
per model (1-5 s each even at 100 rows: the generator's per-segment sampling calls) and to the weak
models, whose requests run into the time caps. A full run of earlier quick-trained models took
18.5 min (23.5 min with the previous script). The caps bound one model x table at about 90 s
generation + ~35 s for the families + 150 s scenario coverage, i.e. about 50 min for 5 x 2 even if
every model hit every cap. The wall time of each run is stored in ``model_comparison.json``
(``run_seconds``) and shown in the Week-3 report.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import bedsynth  # noqa: E402,F401  - sets TQDM_DISABLE before SDV is imported
import pandas as pd  # noqa: E402

from bedsynth.config import get_path, get_seed, project_relative, section  # noqa: E402
from bedsynth.data import io  # noqa: E402
from bedsynth.evaluation import evaluate_model, to_jsonable  # noqa: E402
from bedsynth.evaluation.coverage import (  # noqa: E402
    rebalancing_flagged,
    strata_columns,
    strata_mix_tvd,
    strata_rounding_tvd,
)
from bedsynth.evaluation.operational import calibrate_operational  # noqa: E402
from bedsynth.evaluation.references import (  # noqa: E402
    NULL_REFERENCE_ID,
    REAL_REFERENCE_ID,
    REFERENCE_DESCRIPTIONS,
    evaluate_references,
    normalised_scores,
)
from bedsynth.evaluation.scoring import FAMILIES, composite_weights  # noqa: E402
from bedsynth.schema import TABLES, assign_ids, derive  # noqa: E402

logger = logging.getLogger("bedsynth.scripts.evaluate")

QUICK_MAX_ROWS = 2_000
QUICK_SCENARIO_ROWS = 100
DEFAULT_MAX_ROWS_PER_REQUEST = 20_000
COMPARISON_FILE = "model_comparison.json"
#: Defaults of the runtime and calibration settings (``settings.evaluation`` overrides them).
EVALUATION_DEFAULTS: dict[str, Any] = {
    "max_synthetic_rows": 20_000,
    "sample_time_budget_seconds": 90.0,
    "scenario_time_budget_seconds": 20.0,
    "scenario_max_attempts": 8,
    "coverage_time_budget_seconds": 150.0,
    "noise_calibration_reps": 50,
    "noise_sd_multiplier": 3.0,
    "strata_tvd_flag": 0.05,
    "references": True,
}
#: Tighter caps for ``--quick`` (``settings.evaluation.quick_overrides`` replaces them): with the full
#: caps the weak models' scenario rejection rounds dominate, whatever the row counts.
QUICK_OVERRIDES: dict[str, Any] = {
    "sample_time_budget_seconds": 45.0,
    "scenario_time_budget_seconds": 10.0,
    "coverage_time_budget_seconds": 60.0,
    "noise_calibration_reps": 30,
}
#: Run-configuration keys that must match for two comparison entries to be ranked together.
COMPARABLE_KEYS: tuple[str, ...] = ("quick", "rows", "rows_per_scenario", "scenarios", "seed")
#: Keys of ``card["metrics"]`` written to the registry (the contract the API reads).
REGISTRY_METRIC_KEYS: tuple[str, ...] = (
    "composite", "complete", "fidelity", "validity", "operational", "utility", "privacy", "raw_hard_valid_rate",
    "operational_passed", "operational_scored", "scenario_coverage", "rank", "comparable", "evaluated_at",
    "run_quick", "rows", "source",
)
#: Composite components copied into the registry metrics (``entry["components"]``).
METRIC_COMPONENTS: tuple[str, ...] = ("fidelity", "validity", "operational", "utility", "privacy")


# ---------------------------------------------------------------------------
# Command line and settings
# ---------------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate every trained BedSynth model and select the best per table.")
    parser.add_argument("--tables", nargs="+", choices=TABLES, default=list(TABLES), help="tables to evaluate")
    parser.add_argument("--models", nargs="+", default=None, help="experiment ids to evaluate (default: all trained)")
    parser.add_argument("--rows", type=int, default=None,
                        help="synthetic rows per model (default: training-split size, capped at "
                             "settings.evaluation.max_synthetic_rows)")
    parser.add_argument("--no-scenarios", action="store_true", help="skip the scenario coverage evaluation")
    parser.add_argument("--no-references", action="store_true",
                        help="skip the real-holdout / no-learning reference pseudo-models")
    parser.add_argument("--quick", action="store_true",
                        help=f"fast run: <= {QUICK_MAX_ROWS} rows, {QUICK_SCENARIO_ROWS} rows per scenario, no sdmetrics")
    parser.add_argument("--seed", type=int, default=None, help="evaluation seed (default: settings.project.seed)")
    parser.add_argument("--sync-registry", action="store_true",
                        help=f"only copy the stored scores ({COMPARISON_FILE} and the per-model result files) into "
                             "the registry model cards as card.metrics; nothing is generated or evaluated "
                             "(--tables limits the tables)")
    args = parser.parse_args(argv)
    if args.rows is not None and args.rows < 1:
        parser.error("--rows must be a positive integer")
    return args


def evaluation_settings(quick: bool = False) -> dict[str, Any]:
    """``settings.evaluation`` merged over :data:`EVALUATION_DEFAULTS` (``null`` keeps the default).

    With ``quick`` the ``quick_overrides`` (default :data:`QUICK_OVERRIDES`) replace the time caps.
    """
    configured = section("evaluation") or {}
    merged = dict(EVALUATION_DEFAULTS)
    merged.update({k: v for k, v in configured.items() if v is not None or k not in EVALUATION_DEFAULTS})
    if quick:
        overrides = configured.get("quick_overrides") or QUICK_OVERRIDES
        merged.update({k: v for k, v in overrides.items() if v is not None})
    return merged


def run_config(args: argparse.Namespace, n_rows: int, seed: int, settings: dict[str, Any]) -> dict[str, Any]:
    """Configuration of this run for one table (stored in every comparison entry)."""
    return {
        "quick": bool(args.quick),
        "rows": int(n_rows),
        "rows_per_scenario": None if args.no_scenarios else (
            QUICK_SCENARIO_ROWS if args.quick else int(settings.get("scenario_rows") or 400)),
        "scenarios": not args.no_scenarios,
        "seed": int(seed),
        "sdmetrics": not args.quick,
        "sample_time_budget_seconds": settings.get("sample_time_budget_seconds"),
        "scenario_time_budget_seconds": settings.get("scenario_time_budget_seconds"),
        "scenario_max_attempts": settings.get("scenario_max_attempts"),
        "coverage_time_budget_seconds": settings.get("coverage_time_budget_seconds"),
    }


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------
def synthetic_row_count(train_rows: int, rows: int | None = None, quick: bool = False) -> int:
    """Rows to generate: ``--rows``, else settings.evaluation.synthetic_rows, else the train size.

    The default (not an explicit ``--rows``) is capped at ``settings.evaluation.max_synthetic_rows``;
    ``quick`` caps at :data:`QUICK_MAX_ROWS`.
    """
    settings = evaluation_settings()
    if rows:
        n_rows = int(rows)
    else:
        n_rows = int(settings.get("synthetic_rows") or train_rows)
        cap = settings.get("max_synthetic_rows")
        if cap:
            n_rows = min(n_rows, int(cap))
    return min(n_rows, QUICK_MAX_ROWS) if quick else n_rows


def trained_model_ids(registry: Any, table: str) -> list[str]:
    """Experiment ids of the trained models of ``table`` in registry order (duplicates removed).

    Registered models whose artefacts are missing (``trained: false``) are skipped.
    """
    ids = []
    for card in registry.list_models(table):
        model_id = card.get("experiment_id") or card.get("model_id") or card.get("id")
        if model_id and card.get("table", table) == table and card.get("trained", True) is not False:
            ids.append(str(model_id))
    return list(dict.fromkeys(ids))


def model_card_summary(registry: Any, table: str, model_id: str) -> dict[str, Any]:
    """Model-card fields the report needs (empty when the registry has no full card)."""
    try:
        card = registry.get_card(table, model_id)
    except Exception:  # noqa: BLE001 - fakes and older registries have no full cards
        card = next((c for c in registry.list_models(table)
                     if (c.get("experiment_id") or c.get("model_id")) == model_id), {})
    training = card.get("training") or {}
    diagnostics = training.get("sample_diagnostics") or {}
    params = training.get("sdv_params") or {}
    return {
        "experiment_id": card.get("experiment_id", model_id),
        "model": card.get("model"),
        "description": card.get("description"),
        "segment_label": card.get("segment_label") or card.get("segment_by") or "global",
        "constraint_groups": card.get("constraint_groups"),
        "effective_constraint_groups": card.get("effective_constraint_groups"),
        "fitted_at": card.get("fitted_at"),
        "fit_seconds": card.get("fit_seconds"),
        "training_rows": card.get("training_rows"),
        "quick": training.get("quick"),
        "epochs": params.get("epochs"),
        "raw_hard_valid_rate": diagnostics.get("raw_hard_valid_rate"),
        "hard_violation_rates": diagnostics.get("hard_violation_rates"),
        "soft_violation_rates": diagnostics.get("soft_violation_rates"),
    }


def _strata_summary(reports: list[dict[str, Any]]) -> dict[str, Any] | None:
    blocks = [r.get("allocation_strata") for r in reports if isinstance(r.get("allocation_strata"), dict)]
    if not blocks:
        return None
    return {"columns": blocks[0].get("columns"), "source": blocks[0].get("source"),
            "strata_cells": sum(b.get("strata_cells") or 0 for b in blocks),
            "rebalanced_rows": sum(b.get("rebalanced_rows") or 0 for b in blocks),
            "quota_moves": sum(b.get("quota_moves") or 0 for b in blocks)}


def generate_synthetic(
    generator: Any, table: str, model_id: str, n_rows: int, seed: int, max_rows_per_request: int | None = None,
    *, time_budget_seconds: float | None = None, reference: pd.DataFrame | None = None,
    strata_tvd_flag: float | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Generate ``n_rows`` validated rows in chunks (each request <= settings.generation.max_rows).

    Every request carries ``time_budget_seconds`` (``None`` = the generator default). The
    summary records shortfall, early stops and the allocation-strata block; with
    ``reference`` (the training split) it adds the TVD between the output's strata mix and the
    reference mix (``strata_mix_tvd``) and a warning above ``strata_tvd_flag``.
    """
    from bedsynth.synthesis.generator import GenerationRequest

    if max_rows_per_request is None:
        max_rows_per_request = int((section("generation") or {}).get("max_rows") or DEFAULT_MAX_ROWS_PER_REQUEST)
    if strata_tvd_flag is None:
        strata_tvd_flag = float(evaluation_settings()["strata_tvd_flag"])
    frames, reports, remaining, chunk = [], [], n_rows, 0
    start = time.perf_counter()
    while remaining > 0:
        size = min(remaining, max_rows_per_request)
        request = GenerationRequest(table=table, num_rows=size, model_id=model_id, seed=seed + chunk,
                                    time_budget_seconds=time_budget_seconds)
        result = generator.generate(request)
        frames.append(result.data)
        reports.append(dict(result.report or {}))
        remaining -= size
        chunk += 1
    data = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if len(data):
        data = assign_ids(data, table)
    sampled = sum(r.get("sampled_rows") or 0 for r in reports)
    accepted = sum(r.get("accepted_rows") or 0 for r in reports)
    strata = _strata_summary(reports)
    warnings_ = [w for r in reports for w in (r.get("warnings") or [])]
    tvd = rounding = None
    if reference is not None and strata and len(data):
        tvd = strata_mix_tvd(reference, data, strata.get("columns") or [])
        rounding = strata_rounding_tvd(reference, len(data), strata.get("columns") or [])
        if rebalancing_flagged(tvd, rounding, strata_tvd_flag):
            warnings_.append(f"Output allocation-strata mix differs from the training split (TVD {tvd:.3f}, more "
                             f"than {strata_tvd_flag:g} above the rounding floor {rounding or 0.0:.3f}): the "
                             f"generator moved quota to strata model '{model_id}' produces more easily.")
    summary = {
        "requested": n_rows,
        "generated": len(data),
        "shortfall": max(0, n_rows - len(data)),
        "chunks": chunk,
        "acceptance_rate": (accepted / sampled) if sampled else None,
        "time_budget_seconds": time_budget_seconds,
        "stopped_reasons": [r["stopped_reason"] for r in reports if r.get("stopped_reason")],
        "throttled_rounds": sum(int(r.get("throttled_rounds") or 0) for r in reports),
        "allocation_strata": strata,
        "strata_mix_tvd": tvd,
        "strata_rounding_tvd": rounding,
        "rebalancing_flag": rebalancing_flagged(tvd, rounding, strata_tvd_flag),
        "seconds": round(time.perf_counter() - start, 3),
        "warnings": warnings_,
    }
    return data, summary


def draw_raw_samples(
    registry: Any, table: str, model_id: str, n_rows: int, seed: int, generator: Any | None = None
) -> pd.DataFrame:
    """Unfiltered model samples with derived columns (no rule-based rejection).

    Uses ``generator.sample_raw`` when the generator provides it (the model is then loaded
    once and shared with generation through the generator's cache), else loads the model
    from the registry.
    """
    sample_raw = getattr(generator, "sample_raw", None)
    if callable(sample_raw):
        return sample_raw(table, n_rows, model_id=model_id, seed=seed)
    model = registry.load(table, model_id)
    return derive(table, model.sample(n_rows, seed=seed))


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def scenario_options(settings: dict[str, Any], strata: list[str]) -> dict[str, Any]:
    """Keyword arguments of the scenario-coverage evaluation (time budgets, attempts, strata)."""
    return {
        "time_budget_seconds": settings.get("scenario_time_budget_seconds"),
        "max_attempts": settings.get("scenario_max_attempts"),
        "total_budget_seconds": settings.get("coverage_time_budget_seconds"),
        "strata": strata,
    }


def calibrate(train: pd.DataFrame, holdout: pd.DataFrame, table: str, n_rows: int, seed: int,
              settings: dict[str, Any]) -> dict[str, Any] | None:
    """Real-vs-real noise of the operational checks at ``n_rows`` (``None`` if it fails or is disabled)."""
    reps = int(settings.get("noise_calibration_reps") or 0)
    if reps <= 0 or holdout is None or len(holdout) == 0:
        return None
    try:
        return calibrate_operational(train, holdout, table, n_rows, reps=reps,
                                     sd_multiplier=float(settings.get("noise_sd_multiplier") or 3.0), seed=seed)
    except Exception:  # noqa: BLE001 - the evaluation runs with the design thresholds instead
        logger.exception("Operational noise calibration failed for %s", table)
        return None


def evaluate_one(
    table: str,
    model_id: str,
    train: pd.DataFrame,
    holdout: pd.DataFrame,
    registry: Any,
    generator: Any,
    args: argparse.Namespace,
    seed: int,
    *,
    calibration: dict[str, Any] | None = None,
    settings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate synthetic data for one model and run every evaluation family on it."""
    settings = settings or evaluation_settings(args.quick)
    strata = list(strata_columns(table))
    n_rows = synthetic_row_count(len(train), args.rows, args.quick)
    synthetic, generation = generate_synthetic(
        generator, table, model_id, n_rows, seed, time_budget_seconds=settings.get("sample_time_budget_seconds"),
        reference=train, strata_tvd_flag=float(settings.get("strata_tvd_flag") or 0.05))
    raw = draw_raw_samples(registry, table, model_id, n_rows, seed, generator=generator)
    results = evaluate_model(
        table, train, holdout, synthetic, raw_synthetic=raw, generator=generator, model_id=model_id,
        include_scenarios=not args.no_scenarios, seed=seed,
        rows_per_scenario=QUICK_SCENARIO_ROWS if args.quick else None,
        privacy_sdmetrics=not args.quick,
        forced_columns=(generation.get("allocation_strata") or {}).get("columns") or strata,
        operational_calibration=calibration,
        scenario_options=scenario_options(settings, strata),
    )
    results["generation"] = generation
    results["run"] = run_config(args, n_rows, seed, settings)
    results["model_card"] = model_card_summary(registry, table, model_id)
    return results


def comparison_entry(results: dict[str, Any]) -> dict[str, Any]:
    """Model-comparison row: composite, components, family scores, run configuration and card."""
    composite = results.get("composite") or {}
    generation = results.get("generation") or {}
    card = results.get("model_card") or {}
    return {
        "model_id": results.get("model_id"),
        "status": "ok",
        "composite": composite.get("score"),
        "complete": composite.get("complete", True),
        "components": composite.get("components"),
        "missing_components": composite.get("missing_components"),
        "failed_families": composite.get("failed_families") or [],
        "family_scores": {family: (results.get(family) or {}).get("score") for family in FAMILIES},
        "model_view": {k: ((results.get("model_view") or {}).get(k) or {}).get("score")
                       for k in ("statistical", "operational")},
        "output_validity": (results.get("output_validity") or {}).get("score"),
        "generation": {k: generation.get(k) for k in ("generated", "shortfall", "acceptance_rate", "strata_mix_tvd",
                                                       "rebalancing_flag", "stopped_reasons", "throttled_rounds",
                                                       "seconds")},
        "n_rows": results.get("n_rows"),
        "run_config": results.get("run"),
        "model_fitted_at": card.get("fitted_at"),
        "model_card": card,
        "evaluated_at": results.get("evaluated_at"),
        "seconds": (results.get("timings") or {}).get("total"),
        "results_file": f"{results.get('table')}__{results.get('model_id')}.json",
    }


def failed_entry(model_id: str, error: str, config: dict[str, Any] | None,
                 card: dict[str, Any] | None) -> dict[str, Any]:
    """Comparison row of a model whose evaluation raised in this run (never selected)."""
    return {
        "model_id": model_id, "status": "failed", "error": error, "composite": None, "complete": False,
        "components": None, "run_config": config, "model_fitted_at": (card or {}).get("fitted_at"),
        "model_card": card, "evaluated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "results_file": None,
    }


def reference_entry(table: str, results: dict[str, Any]) -> dict[str, Any]:
    """Comparison row of a reference pseudo-model."""
    ref_id = results.get("model_id")
    composite = results.get("composite") or {}
    return {
        "model_id": ref_id,
        "kind": (results.get("reference") or {}).get("kind"),
        "description": REFERENCE_DESCRIPTIONS.get(ref_id),
        "status": "failed" if results.get("error") else "ok",
        "error": results.get("error"),
        "composite": composite.get("score"),
        "components": composite.get("components"),
        "family_scores": {family: (results.get(family) or {}).get("score") for family in FAMILIES},
        "run_config": results.get("run"),
        "results_file": None if results.get("error") else f"{table}__reference__{ref_id}.json",
    }


def eligible(entry: dict[str, Any]) -> bool:
    """Whether an entry may be selected: scored, complete, comparable and not failed or stale."""
    return (entry.get("composite") is not None and entry.get("status", "ok") == "ok"
            and entry.get("complete", True) is not False and entry.get("comparable", True) is not False)


def select_best(entries: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Eligible entry with the highest composite score (first in order on ties; None if none)."""
    scored = [e for e in entries if eligible(e)]
    return max(scored, key=lambda e: e["composite"]) if scored else None


def load_comparison(path: Path) -> dict[str, Any]:
    """Existing comparison file (empty structure if absent or unreadable)."""
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Ignoring unreadable %s: %s", path, exc)
    return {"tables": {}}


def _comparable(config: dict[str, Any] | None, current: dict[str, Any] | None) -> bool:
    if current is None:
        return True
    if not config:
        return False
    return all(config.get(k) == current.get(k) for k in COMPARABLE_KEYS)


def _attach_normalised(entries: list[dict[str, Any]], references: list[dict[str, Any]]) -> None:
    """Add ``(model - null) / (real - null)`` scores to every model entry (in place)."""
    by_id = {r["model_id"]: r for r in references
             if r.get("status", "ok") == "ok" and r.get("comparable", True) is not False}
    real, null = by_id.get(REAL_REFERENCE_ID), by_id.get(NULL_REFERENCE_ID)
    if real is None or null is None:
        return
    real_scores = {**(real.get("components") or {}), "composite": real.get("composite")}
    null_scores = {**(null.get("components") or {}), "composite": null.get("composite")}
    for entry in entries:
        if entry.get("status") == "ok" and entry.get("components"):
            scores = {**entry["components"], "composite": entry.get("composite")}
            entry["normalised"] = normalised_scores(scores, real_scores, null_scores)


def merge_comparison(
    existing: dict[str, Any],
    new_entries: dict[str, list[dict[str, Any]]],
    registered: dict[str, list[str]],
    *,
    run_configs: dict[str, dict[str, Any]] | None = None,
    fitted: dict[str, dict[str, Any]] | None = None,
    references: dict[str, list[dict[str, Any]]] | None = None,
    calibrations: dict[str, dict[str, Any] | None] | None = None,
) -> dict[str, Any]:
    """Merge this run's entries into the comparison and recompute the best model per table.

    Earlier entries of models that are still registered are kept, but an entry evaluated on an
    older version of the model (``model_fitted_at`` differs from ``fitted``) is dropped as
    stale, and an entry from a run with another configuration (``run_configs``) is kept with
    ``comparable = False`` and never selected. This run's entries - including ``failed`` ones -
    replace earlier entries of the same model. ``references`` replace the table's reference
    rows; every model entry gets normalised scores against them.
    """
    tables = dict(existing.get("tables") or {})
    for table, entries in new_entries.items():
        current_config = (run_configs or {}).get(table)
        current_fit = (fitted or {}).get(table) or {}
        previous = {e["model_id"]: dict(e) for e in (tables.get(table) or {}).get("models", [])}
        kept: dict[str, dict[str, Any]] = {}
        for model_id, entry in previous.items():
            if model_id not in registered.get(table, [model_id]):
                continue
            if current_fit.get(model_id) and entry.get("model_fitted_at") != current_fit.get(model_id):
                logger.info("Dropping stale comparison entry %s/%s (model retrained)", table, model_id)
                continue
            entry["comparable"] = _comparable(entry.get("run_config"), current_config)
            kept[model_id] = entry
        for entry in entries:
            kept[entry["model_id"]] = {**entry, "comparable": True}
        models = sorted(kept.values(), key=lambda e: -(e["composite"] if e.get("composite") is not None else -1.0))
        table_refs = (references or {}).get(table)
        if table_refs is None:  # no references this run: keep the earlier ones, flagged by configuration
            table_refs = [{**r, "comparable": _comparable(r.get("run_config"), current_config)}
                          for r in (tables.get(table) or {}).get("references") or []]
        _attach_normalised(models, table_refs)
        best = select_best(models)
        block = {
            "models": models,
            "references": table_refs,
            "best_model": best["model_id"] if best else None,
            "best_composite": best["composite"] if best else None,
            "run_config": current_config,
        }
        calibration = (calibrations or {}).get(table)
        if calibration is not None:
            block["noise_calibration"] = {k: calibration.get(k) for k in
                                          ("n", "reps", "sd_multiplier", "seed", "basis", "real_pass_rate_cv")}
        tables[table] = block
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "description": "Composite score and components per trained model, next to the real-holdout ceiling and "
                       "the no-learning floor; the best complete, comparable composite per table becomes the "
                       "registry default.",
        "composite_weights": composite_weights(),
        "selection_rule": "highest composite among entries with status ok, complete composite (no failed family), "
                          "the same run configuration as the latest run and the current model version",
        "tables": tables,
    }


def update_registry_defaults(registry: Any, comparison: dict[str, Any], tables: list[str]) -> dict[str, str]:
    """Set each evaluated table's registry default to its best-composite model."""
    updated = {}
    for table in tables:
        info = comparison["tables"].get(table) or {}
        best, score = info.get("best_model"), info.get("best_composite")
        if not best:
            continue
        eligible_count = sum(1 for e in info.get("models") or [] if eligible(e))
        reason = (f"Best composite score {score:.3f} among {eligible_count} comparable evaluated model(s) "
                  f"(scripts/03_evaluate.py, {comparison['generated_at']}).")
        registry.set_default(table, best, reason=reason)
        updated[table] = best
    return updated


# ---------------------------------------------------------------------------
# Registry metrics (card["metrics"], copied from the stored results)
# ---------------------------------------------------------------------------
def _float_or_none(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return int(value)


def metric_ranks(entries: list[dict[str, Any]]) -> dict[str, int]:
    """``{model_id: rank}`` of one table's comparison entries (1 = best comparable model).

    Eligible entries (scored, complete, comparable, status ok - the ones :func:`select_best`
    chooses from) come first by composite, ties in stored order, so rank 1 is the registry
    default; the other entries follow by composite (unscored last).
    """
    def order(group: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return sorted(group, key=lambda e: -e["composite"] if e.get("composite") is not None else float("inf"))

    ranked = order([e for e in entries if eligible(e)]) + order([e for e in entries if not eligible(e)])
    return {str(entry.get("model_id")): rank for rank, entry in enumerate(ranked, start=1)}


def registry_metrics(entry: dict[str, Any], result: dict[str, Any] | None, *, rank: int, source: str) -> dict[str, Any]:
    """``card["metrics"]`` of one model from its comparison entry and result file (no re-scoring).

    Keys (:data:`REGISTRY_METRIC_KEYS`): ``composite``, ``complete``, the components ``fidelity``,
    ``validity``, ``operational``, ``utility`` and ``privacy`` (``entry["components"]``),
    ``raw_hard_valid_rate`` (validity of the raw model samples, result file), ``operational_passed``
    / ``operational_scored`` (operational checks passed / scored, result file),
    ``scenario_coverage`` (family score), ``rank``, ``comparable``, ``evaluated_at``, ``run_quick``
    and ``rows`` (the run configuration: evaluated size N) and ``source``. A value the stored files
    do not hold is ``None``.
    """
    result = result if isinstance(result, dict) else {}
    components = entry.get("components") or {}
    run_config = entry.get("run_config") or result.get("run") or {}
    validity = result.get("validity") or {}
    operational = result.get("operational") or {}
    coverage = (entry.get("family_scores") or {}).get("scenario_coverage")
    if coverage is None:
        coverage = (result.get("scenario_coverage") or {}).get("score")
    metrics: dict[str, Any] = {
        "composite": _float_or_none(entry.get("composite")),
        "complete": entry.get("status", "ok") == "ok" and entry.get("complete", True) is not False,
    }
    metrics.update({name: _float_or_none(components.get(name)) for name in METRIC_COMPONENTS})
    metrics.update({
        "raw_hard_valid_rate": (_float_or_none(validity.get("hard_valid_rate"))
                                if validity.get("basis") == "raw_model_samples" else None),
        "operational_passed": _int_or_none(operational.get("n_passed")),
        "operational_scored": _int_or_none(operational.get("n_scored")),
        "scenario_coverage": _float_or_none(coverage),
        "rank": int(rank),
        "comparable": entry.get("comparable", True) is not False,
        "evaluated_at": str(entry.get("evaluated_at") or result.get("evaluated_at") or ""),
        "run_quick": bool(run_config.get("quick", False)),
        "rows": _int_or_none(run_config.get("rows")),
        "source": source,
    })
    return metrics


def _read_result_file(results_dir: Path, name: Any) -> dict[str, Any] | None:
    """A per-model result file of the comparison (``None`` when absent or unreadable)."""
    if not name:
        return None
    path = results_dir / str(name)
    if not path.is_file():
        logger.warning("Result file %s is missing; its registry metrics stay null", path.name)
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Cannot read %s (%s); its registry metrics stay null", path.name, exc)
        return None


def sync_registry_metrics(registry: Any, comparison: dict[str, Any] | None, results_dir: Path,
                          tables: list[str] | None = None) -> dict[str, list[str]]:
    """Write ``card["metrics"]`` (:func:`registry_metrics`) for every model in ``comparison``.

    Covers every table of the comparison (``tables`` restricts it). Skipped: registries without
    ``update_card`` (test fakes), models that are no longer registered and entries evaluated on
    an older version of the model (the card's ``fitted_at`` differs from ``model_fitted_at``).
    Ranks are computed over the entries that are written. Returns ``{table: [model ids written]}``.
    """
    update_card = getattr(registry, "update_card", None)
    get_card = getattr(registry, "get_card", None)
    if not callable(update_card) or not callable(get_card) or not comparison:
        return {}
    source = project_relative(results_dir / COMPARISON_FILE)
    written: dict[str, list[str]] = {}
    for table, block in (comparison.get("tables") or {}).items():
        if tables is not None and table not in tables:
            continue
        current = []
        for entry in (block or {}).get("models") or []:
            model_id = str(entry.get("model_id") or "")
            if not model_id:
                continue
            try:
                card = get_card(table, model_id)
            except Exception as exc:  # noqa: BLE001 - unregistered model / unknown table
                logger.info("No registry metrics for %s/%s: %s", table, model_id, exc)
                continue
            fitted, evaluated = card.get("fitted_at"), entry.get("model_fitted_at")
            if fitted and evaluated and fitted != evaluated:
                logger.warning("No registry metrics for %s/%s: the model was retrained after its evaluation "
                               "(fitted_at %s, evaluated model %s)", table, model_id, fitted, evaluated)
                continue
            current.append(entry)
        ranks = metric_ranks(current)
        for entry in current:
            model_id = str(entry["model_id"])
            result = _read_result_file(results_dir, entry.get("results_file"))
            update_card(table, model_id, metrics=registry_metrics(entry, result, rank=ranks[model_id], source=source))
            written.setdefault(table, []).append(model_id)
    return written


def sync_registry_only(args: argparse.Namespace, registry: Any) -> int:
    """``--sync-registry``: copy the stored scores into the registry cards (nothing is evaluated)."""
    results_dir = get_path("results_dir", create=False)
    path = results_dir / COMPARISON_FILE
    if not path.is_file():
        print(f"{COMPARISON_FILE} not found in {project_relative(results_dir)}: run scripts/03_evaluate.py first.",
              file=sys.stderr)
        return 1
    written = sync_registry_metrics(registry, load_comparison(path), results_dir, tables=list(args.tables))
    for table, model_ids in written.items():
        print(f"[{table}] registry metrics written for {len(model_ids)} model(s): {', '.join(model_ids)}")
    if not written:
        print("No registry card was updated (no current, registered model in the comparison).", file=sys.stderr)
        return 1
    return 0


def write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), indent=2, allow_nan=False), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Report rendering (bedsynth.reporting)
# ---------------------------------------------------------------------------
def render_reports(results: dict[str, Any]) -> None:
    """Render Week-3 figures and the Markdown report from this run's results.

    ``run`` calls it only when the ``bedsynth.reporting`` package exists. ``results`` =
    {"comparison": <model_comparison dict>, "results": {"<table>__<model_id>": <evaluate_model
    dict>}, "references": {"<table>": {"<reference_id>": <evaluate_model dict>}}} is passed to
    ``bedsynth.reporting.render_week3_report``; a rendering failure is logged and never affects
    the evaluation outputs.
    """
    import bedsynth.reporting as reporting  # noqa: PLC0415 - optional package

    renderer = getattr(reporting, "render_week3_report", None)
    if renderer is None:
        logger.info("bedsynth.reporting has no render_week3_report yet - skipping report rendering.")
        return
    try:
        path = renderer(results)
    except Exception:  # noqa: BLE001 - the evaluation results are already written
        logger.exception("Week-3 report rendering failed (results and registry defaults are unaffected)")
        return
    print(f"Week-3 report written to {path}")


def reporting_available() -> bool:
    return importlib.util.find_spec("bedsynth.reporting") is not None


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def _evaluate_references(table: str, train: pd.DataFrame, holdout: pd.DataFrame, n_rows: int, seed: int,
                         args: argparse.Namespace, settings: dict[str, Any], calibration: dict[str, Any] | None,
                         config: dict[str, Any], results_dir: Path) -> dict[str, dict[str, Any]]:
    """Score and store the reference pseudo-models of ``table`` (empty when disabled)."""
    if args.no_references or not settings.get("references", True) or holdout is None or len(holdout) == 0:
        return {}
    strata = list(strata_columns(table))
    start = time.perf_counter()
    refs = evaluate_references(
        table, train, holdout, n_rows, strata=strata, seed=seed, include_scenarios=not args.no_scenarios,
        rows_per_scenario=QUICK_SCENARIO_ROWS if args.quick else None, operational_calibration=calibration,
        scenario_options={"strata": strata})
    for ref_id, result in refs.items():
        result["run"] = config
        if not result.get("error"):
            write_json(results_dir / f"{table}__reference__{ref_id}.json", result)
    summary = ", ".join(f"{ref_id}={(r.get('composite') or {}).get('score')}" for ref_id, r in refs.items())
    print(f"[{table}] references ({time.perf_counter() - start:.1f} s): {summary}")
    return refs


def run(args: argparse.Namespace, registry: Any, generator: Any) -> dict[str, Any]:
    """Evaluate the selected tables/models, write result files and update registry defaults."""
    seed = get_seed() if args.seed is None else args.seed
    settings = evaluation_settings(args.quick)
    results_dir = get_path("results_dir")
    all_results: dict[str, dict[str, Any]] = {}
    all_references: dict[str, dict[str, dict[str, Any]]] = {}
    new_entries: dict[str, list[dict[str, Any]]] = {}
    registered: dict[str, list[str]] = {}
    configs: dict[str, dict[str, Any]] = {}
    fitted: dict[str, dict[str, Any]] = {}
    reference_rows: dict[str, list[dict[str, Any]]] = {}
    calibrations: dict[str, dict[str, Any] | None] = {}
    run_start = time.perf_counter()

    for table in args.tables:
        try:
            train, holdout = io.load_processed(table, "train"), io.load_processed(table, "holdout")
        except FileNotFoundError as exc:
            print(f"[{table}] skipped: {exc}")
            continue
        model_ids = trained_model_ids(registry, table)
        registered[table] = model_ids
        selected = [m for m in model_ids if args.models is None or m in args.models]
        if not selected:
            print(f"[{table}] no trained models to evaluate (trained: {model_ids or 'none'}).")
            continue
        train, holdout = derive(table, train), derive(table, holdout)
        n_rows = synthetic_row_count(len(train), args.rows, args.quick)
        config = configs[table] = run_config(args, n_rows, seed, settings)
        fitted[table] = {m: model_card_summary(registry, table, m).get("fitted_at") for m in model_ids}
        start = time.perf_counter()
        calibration = calibrations[table] = calibrate(train, holdout, table, n_rows, seed, settings)
        print(f"[{table}] operational noise calibration at n={n_rows}: "
              f"{'done' if calibration else 'unavailable'} ({time.perf_counter() - start:.1f} s)")
        refs = _evaluate_references(table, train, holdout, n_rows, seed, args, settings, calibration, config,
                                    results_dir)
        if refs:
            all_references[table] = refs
            reference_rows[table] = [reference_entry(table, r) for r in refs.values()]
        for model_id in selected:
            print(f"[{table}] evaluating {model_id} on {len(train)} train / {len(holdout)} holdout rows (n={n_rows}) ...")
            start = time.perf_counter()
            try:
                results = evaluate_one(table, model_id, train, holdout, registry, generator, args, seed,
                                       calibration=calibration, settings=settings)
            except Exception as exc:  # noqa: BLE001 - keep evaluating the other models
                logger.exception("Evaluation of %s/%s failed", table, model_id)
                print(f"[{table}] {model_id} FAILED: {type(exc).__name__}: {exc}")
                new_entries.setdefault(table, []).append(failed_entry(
                    model_id, f"{type(exc).__name__}: {exc}", config, model_card_summary(registry, table, model_id)))
                continue
            path = write_json(results_dir / f"{table}__{model_id}.json", results)
            all_results[f"{table}__{model_id}"] = results
            entry = comparison_entry(results)
            new_entries.setdefault(table, []).append(entry)
            composite = entry["composite"]
            print(f"[{table}] {model_id}: composite={composite if composite is None else round(composite, 3)} "
                  f"({time.perf_counter() - start:.1f} s) -> {path}")

    if not new_entries:
        print("Nothing was evaluated - model comparison and registry defaults left unchanged.")
        return {"comparison": None, "results": all_results, "references": all_references, "defaults": {}}
    comparison = merge_comparison(load_comparison(results_dir / COMPARISON_FILE), new_entries, registered,
                                  run_configs=configs, fitted=fitted, references=reference_rows,
                                  calibrations=calibrations)
    comparison["run_seconds"] = round(time.perf_counter() - run_start, 1)
    write_json(results_dir / COMPARISON_FILE, comparison)
    defaults = update_registry_defaults(registry, comparison, list(new_entries))
    synced = sync_registry_metrics(registry, comparison, results_dir)
    _print_summary(comparison, defaults)
    for table, model_ids in synced.items():
        print(f"registry metrics for {table}: {', '.join(model_ids)}")
    if reporting_available():
        render_reports({"comparison": comparison, "results": all_results, "references": all_references})
    return {"comparison": comparison, "results": all_results, "references": all_references, "defaults": defaults}


def _print_summary(comparison: dict[str, Any], defaults: dict[str, str]) -> None:
    for table, info in comparison["tables"].items():
        print(f"\n{table}: best = {info['best_model']} (composite {info['best_composite']})")
        for entry in list(info.get("references") or []) + list(info["models"]):
            components = ", ".join(f"{k}={'-' if v is None else f'{v:.3f}'}" for k, v in (entry.get("components") or {}).items())
            composite = "-" if entry.get("composite") is None else f"{entry['composite']:.3f}"
            flags = [] if entry.get("status", "ok") == "ok" else [entry["status"]]
            if entry.get("comparable") is False:
                flags.append("other run configuration")
            if entry.get("complete") is False and entry.get("status", "ok") == "ok":
                flags.append("incomplete")
            label = f"{entry['model_id']:<26} composite={composite}  [{components}]"
            print(f"  {label}" + (f"  ({'; '.join(flags)})" if flags else ""))
    for table, model_id in defaults.items():
        print(f"registry default for {table} -> {model_id}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    from bedsynth.synthesis.registry import ModelRegistry

    if args.sync_registry:
        return sync_registry_only(args, ModelRegistry())
    from bedsynth.synthesis.generator import SyntheticDataGenerator

    outcome = run(args, ModelRegistry(), SyntheticDataGenerator())
    return 0 if outcome["results"] else 1


if __name__ == "__main__":
    sys.exit(main())
