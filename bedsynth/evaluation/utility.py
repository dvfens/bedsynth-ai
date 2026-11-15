"""Machine-learning utility: train on synthetic, test on real (DESIGN.md §9.6).

For each operational prediction task a gradient-boosting model
(``HistGradientBoostingClassifier`` / ``Regressor``) is trained twice - on the real
training split (TRTR) and on the synthetic data (TSTR) - and both are tested on the real
holdout split. Synthetic data is useful when TSTR performance approaches TRTR.

Tasks
-----
ward_daily
    (a) end-of-day high pressure (``occupancy_level`` in {high, critical}) from start-of-day
        features only; (b) number of discharges from the same features.
patient_stays
    (c) ``log1p(los_hours)`` from admission features; (d) delayed discharge from admission
        features plus the discharge destination.

Utility ratios (clipped to [0, 1]) are *skill* ratios, so a model without skill scores 0:

* regression - R² ratio ``TSTR R² / TRTR R²`` (R² of a constant prediction is <= 0);
* classification - ``(AUC - 0.5)`` ratio and the average-precision lift ratio
  ``(AP_tstr - p) / (AP_trtr - p)`` with ``p`` the positive rate of the test set (the AP of a
  classifier without skill). F1 at a 0.5 threshold is reported but not scored: TRTR is
  conservative at 0.5 on the imbalanced tasks, so an all-positive classifier used to reach an
  F1 ratio of 1.0 (``f1_no_skill = 2p / (1 + p)`` is reported for comparison).

A ratio is only defined when the real-data model is better than chance (R² >= 0.05,
AUC >= 0.55, AP lift >= 0.02); otherwise the task carries no signal to preserve and is
excluded. ``score`` = mean of the task scores (each the mean of its defined ratios), ``None``
when no task is informative.

**Size-matched TRTR.** TSTR trains on ``n`` synthetic rows; the TRTR model it is compared
with is trained on a seeded real-train subsample of the *same* size (``trtr``), so the ratio
measures data quality rather than training-set size (a perfect 2,000-row generator scored only
~0.83 on patient_stays against a TRTR trained on all 45,002 training rows). The full-size TRTR is
reported as context (``trtr_full``) when the sizes differ.
"""
from __future__ import annotations

import logging
import time
import warnings
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import average_precision_score, f1_score, mean_absolute_error, r2_score, roc_auc_score

from bedsynth.config import get_seed
from bedsynth.evaluation.scoring import clip01, mean_or_none, to_jsonable
from bedsynth.evaluation.statistical import ensure_derived, subsample
from bedsynth.schema import PATIENT_STAYS, WARD_DAILY, column_spec

logger = logging.getLogger(__name__)

MIN_R2 = 0.05            # real-data R² below this: nothing to learn
MIN_AUC = 0.55           # real-data AUC below this: nothing to learn
MIN_AP_LIFT = 0.02       # real-data average precision must beat the positive rate by this much
MAX_TRAIN_ROWS = 50_000
MIN_TRAIN_ROWS = 20
DEFAULT_MAX_ITER = 100   # boosting iterations (sklearn default)

START_OF_DAY_FEATURES: tuple[str, ...] = (
    "ward_id", "month", "day_of_week", "staffing_level", "nurses_day_shift", "escalation_beds_open",
    "beds_closed", "census_start", "delayed_discharges", "elective_scheduled",
)
ADMISSION_FEATURES: tuple[str, ...] = (
    "ward_id", "admission_source", "admission_month", "admission_day_of_week", "admission_hour",
    "booking_lead_days", "times_rescheduled", "bed_wait_hours",
)

METRIC_DESCRIPTIONS: dict[str, str] = {
    "trtr": "Train-real-test-real: model trained on a seeded real-train subsample of the same size as the "
            "synthetic training set, tested on the real holdout (the reference of every ratio).",
    "trtr_full": "Train-real-test-real on the full real training split (context; reported when its size differs).",
    "tstr": "Train-synthetic-test-real: model trained on synthetic data, tested on the real holdout.",
    "r2_ratio": "TSTR R² / TRTR R² (clipped to [0, 1]).",
    "auc_ratio": "(TSTR AUC - 0.5) / (TRTR AUC - 0.5) (clipped to [0, 1]); 0 for a classifier without skill.",
    "ap_lift_ratio": "(TSTR AP - p) / (TRTR AP - p), p = positive rate of the test set (clipped to [0, 1]); "
                     "0 for a classifier without skill.",
    "f1": "F1 of the positive class at a 0.5 probability threshold (reported, not scored).",
    "f1_no_skill": "F1 of the all-positive classifier, 2p / (1 + p): an F1 at or below it shows no skill.",
}


@dataclass(frozen=True)
class MLTask:
    """One prediction task used to measure ML utility."""

    name: str
    table: str
    kind: str                                   # "classification" | "regression"
    target: str                                 # human-readable target definition
    features: tuple[str, ...]
    description: str
    target_fn: Callable[[pd.DataFrame], pd.Series]


ML_TASKS: dict[str, tuple[MLTask, ...]] = {
    WARD_DAILY: (
        MLTask("high_pressure_classification", WARD_DAILY, "classification",
               "occupancy_level in {high, critical}", START_OF_DAY_FEATURES,
               "Predict end-of-day high bed pressure from start-of-day information.",
               lambda d: d["occupancy_level"].isin(["high", "critical"]).astype(int)),
        MLTask("discharges_regression", WARD_DAILY, "regression", "discharges", START_OF_DAY_FEATURES,
               "Predict the number of discharges from start-of-day information.",
               lambda d: pd.to_numeric(d["discharges"], errors="coerce").astype(float)),
    ),
    PATIENT_STAYS: (
        MLTask("los_regression", PATIENT_STAYS, "regression", "log1p(los_hours)", ADMISSION_FEATURES,
               "Predict length of stay (log hours) from admission information.",
               lambda d: np.log1p(pd.to_numeric(d["los_hours"], errors="coerce").astype(float).clip(lower=0))),
        MLTask("delayed_discharge_classification", PATIENT_STAYS, "classification", "is_delayed_discharge",
               ADMISSION_FEATURES + ("discharge_destination",),
               "Predict whether the departure will be delayed from admission information and destination.",
               lambda d: d["is_delayed_discharge"].astype(bool).astype(int)),
    ),
}


# ---------------------------------------------------------------------------
# Modelling helpers
# ---------------------------------------------------------------------------
def encode_features(df: pd.DataFrame, table: str, features: tuple[str, ...]) -> tuple[pd.DataFrame, list[bool]]:
    """Numeric feature matrix: categoricals as codes of the fixed schema vocabulary (unknown -> NaN).

    Using the schema vocabulary keeps codes identical between real and synthetic frames.
    Returns the matrix and the categorical-feature mask for HistGradientBoosting.
    """
    columns, mask = {}, []
    for feature in features:
        spec = column_spec(table, feature)
        if spec.sdtype in ("categorical", "ordinal"):
            codes = pd.Categorical(df[feature], categories=list(spec.categories)).codes.astype(float)
            codes[codes < 0] = np.nan
            columns[feature] = codes
            mask.append(True)
        else:
            columns[feature] = pd.to_numeric(df[feature], errors="coerce").astype(float).to_numpy()
            mask.append(False)
    return pd.DataFrame(columns, index=df.index), mask


def _fit_predict(task: MLTask, x_train: pd.DataFrame, y_train: pd.Series, x_test: pd.DataFrame,
                 mask: list[bool], seed: int, max_iter: int) -> np.ndarray:
    """Fit on (x_train, y_train); return predicted P(y=1) (classification) or predictions."""
    params = {"categorical_features": mask, "random_state": seed, "max_iter": max_iter}
    if task.kind == "classification":
        classes = np.unique(y_train)
        if classes.size < 2:  # a single-class training set can only predict that class
            return np.full(len(x_test), float(classes[0]) if classes.size else 0.0)
        model = HistGradientBoostingClassifier(**params)
        model.fit(x_train, y_train)
        return model.predict_proba(x_test)[:, 1]
    model = HistGradientBoostingRegressor(**params)
    model.fit(x_train, y_train)
    return model.predict(x_test)


def classification_metrics(y_true: pd.Series | np.ndarray, probability: np.ndarray) -> dict[str, float | None]:
    """AUC, average precision, positive rate and F1 at 0.5 of predicted probabilities ``probability``.

    AUC and AP are ``None`` when ``y_true`` has a single class. ``f1_no_skill`` is the F1 of the
    all-positive classifier (the best F1 attainable without skill).
    """
    y = np.asarray(y_true, dtype=int)
    p = float(y.mean()) if y.size else None
    two_classes = np.unique(y).size == 2
    auc = float(roc_auc_score(y, probability)) if two_classes else None
    ap = float(average_precision_score(y, probability)) if two_classes else None
    f1 = float(f1_score(y, (np.asarray(probability) >= 0.5).astype(int), zero_division=0.0))
    f1_no_skill = (2.0 * p / (1.0 + p)) if p is not None else None
    return {"auc": auc, "average_precision": ap, "positive_rate": p, "f1": f1, "f1_no_skill": f1_no_skill}


def _metrics(task: MLTask, y_true: pd.Series, prediction: np.ndarray) -> dict[str, float | None]:
    if task.kind == "classification":
        return classification_metrics(y_true, prediction)
    return {"r2": float(r2_score(y_true, prediction)), "mae": float(mean_absolute_error(y_true, prediction))}


def utility_ratios(kind: str, trtr: dict[str, Any], tstr: dict[str, Any]) -> dict[str, float | None]:
    """Skill ratios of TSTR relative to TRTR (``None`` where TRTR itself has no skill).

    Classification: ``auc_ratio`` and ``ap_lift_ratio`` - both 0 for a classifier without
    skill (constant or random scores: AUC 0.5, AP = positive rate). Regression: ``r2_ratio``.
    """
    if kind == "regression":
        informative = trtr.get("r2") is not None and trtr["r2"] >= MIN_R2
        return {"r2_ratio": clip01(tstr["r2"] / trtr["r2"]) if informative and tstr.get("r2") is not None else None}
    auc_ok = trtr.get("auc") is not None and tstr.get("auc") is not None and trtr["auc"] >= MIN_AUC
    p = trtr.get("positive_rate")
    trtr_ap, tstr_ap = trtr.get("average_precision"), tstr.get("average_precision")
    ap_ok = auc_ok and None not in (p, trtr_ap, tstr_ap) and trtr_ap - p >= MIN_AP_LIFT
    return {
        "auc_ratio": clip01((tstr["auc"] - 0.5) / (trtr["auc"] - 0.5)) if auc_ok else None,
        "ap_lift_ratio": clip01((tstr_ap - p) / (trtr_ap - p)) if ap_ok else None,
    }


def _ratios(task: MLTask, trtr: dict[str, Any], tstr: dict[str, Any]) -> dict[str, float | None]:
    """Utility ratios of one task (see :func:`utility_ratios`)."""
    return utility_ratios(task.kind, trtr, tstr)


def _prepare(df: pd.DataFrame, task: MLTask) -> tuple[pd.DataFrame, pd.Series, list[bool]]:
    """Feature matrix and target of ``df`` (rows with a missing target dropped)."""
    target = task.target_fn(df)
    keep = target.notna().to_numpy()
    x, mask = encode_features(df.loc[keep], task.table, task.features)
    return x, target[keep], mask


def evaluate_task(task: MLTask, real_train: pd.DataFrame, real_holdout: pd.DataFrame, synthetic: pd.DataFrame,
                  seed: int, max_iter: int = DEFAULT_MAX_ITER, *, size_matched: bool = True) -> dict[str, Any]:
    """TRTR vs TSTR for one task, tested on ``real_holdout``.

    With ``size_matched`` (default) the TRTR reference is trained on a seeded subsample of
    ``real_train`` with as many rows as the synthetic training set, and the full-size TRTR is
    reported as ``trtr_full`` when the sizes differ.
    """
    start = time.perf_counter()
    x_test, y_test, mask = _prepare(real_holdout, task)
    x_full, y_full, _ = _prepare(real_train, task)
    x_syn, y_syn, _ = _prepare(synthetic, task)
    if min(len(x_full), len(x_syn)) < MIN_TRAIN_ROWS or len(x_test) < MIN_TRAIN_ROWS:
        raise ValueError(f"Too few rows for task {task.name}: real={len(x_full)} synthetic={len(x_syn)} test={len(x_test)}")
    matched = size_matched and len(x_full) > len(x_syn)
    if matched:
        keep = np.sort(np.random.default_rng(seed + 17).choice(len(x_full), size=len(x_syn), replace=False))
        x_real, y_real = x_full.iloc[keep], y_full.iloc[keep]
    else:
        x_real, y_real = x_full, y_full

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trtr = _metrics(task, y_test, _fit_predict(task, x_real, y_real, x_test, mask, seed, max_iter))
        tstr = _metrics(task, y_test, _fit_predict(task, x_syn, y_syn, x_test, mask, seed, max_iter))
        trtr_full = (_metrics(task, y_test, _fit_predict(task, x_full, y_full, x_test, mask, seed, max_iter))
                     if matched else None)
    ratios = _ratios(task, trtr, tstr)
    result = {
        "name": task.name,
        "description": task.description,
        "kind": task.kind,
        "target": task.target,
        "features": list(task.features),
        "n_rows": {"train_real": len(x_real), "train_real_full": len(x_full), "train_synthetic": len(x_syn),
                   "test_real": len(x_test)},
        "size_matched": matched,
        "trtr": trtr,
        "trtr_full": trtr_full,
        "tstr": tstr,
        "ratios": ratios,
        "informative": any(v is not None for v in ratios.values()),
        "score": mean_or_none(ratios.values()),
        "seconds": round(time.perf_counter() - start, 3),
    }
    if task.kind == "classification":
        result["positive_rate"] = {"train_real": float(y_full.mean()), "train_synthetic": float(y_syn.mean()),
                                   "test_real": float(y_test.mean())}
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def evaluate_ml_utility(
    real_train: pd.DataFrame,
    real_holdout: pd.DataFrame,
    synthetic: pd.DataFrame,
    table: str,
    *,
    seed: int | None = None,
    max_train_rows: int = MAX_TRAIN_ROWS,
    max_iter: int = DEFAULT_MAX_ITER,
    size_matched: bool = True,
) -> dict[str, Any]:
    """Train-synthetic-test-real utility of ``synthetic`` for the operational tasks of ``table``.

    ``max_train_rows`` caps both training sets (seeded subsample); ``max_iter`` is the number
    of boosting iterations of every model (sklearn's default of 100); ``size_matched`` trains
    the TRTR reference on as many real rows as there are synthetic rows (see module docstring).
    """
    if table not in ML_TASKS:
        raise KeyError(f"Unknown table '{table}'")
    start = time.perf_counter()
    seed = get_seed() if seed is None else int(seed)
    train = subsample(ensure_derived(real_train, table).reset_index(drop=True), max_train_rows, seed)
    syn = subsample(ensure_derived(synthetic, table).reset_index(drop=True), max_train_rows, seed + 1)
    holdout = ensure_derived(real_holdout, table).reset_index(drop=True)

    tasks = []
    for task in ML_TASKS[table]:
        try:
            tasks.append(evaluate_task(task, train, holdout, syn, seed, max_iter, size_matched=size_matched))
        except Exception as exc:  # noqa: BLE001 - report the failure, keep the other task
            logger.warning("ML utility task '%s' failed: %s", task.name, exc)
            tasks.append({"name": task.name, "description": task.description, "kind": task.kind,
                          "score": None, "informative": False, "error": f"{type(exc).__name__}: {exc}"})
    informative = [t for t in tasks if t.get("informative")]
    result = {
        "name": "ml_utility",
        "description": "Train-on-synthetic / test-on-real skill relative to train-on-real with the same number "
                       "of training rows (HistGradientBoosting, real holdout as test set).",
        "table": table,
        "score": mean_or_none(t["score"] for t in informative),
        "n_tasks": len(tasks),
        "n_informative_tasks": len(informative),
        "size_matched": bool(size_matched),
        "n_rows": {"train_real": len(train), "train_synthetic": len(syn), "test_real": len(holdout)},
        "tasks": tasks,
        "metric_descriptions": METRIC_DESCRIPTIONS,
        "seconds": round(time.perf_counter() - start, 3),
    }
    if not informative:
        result["note"] = "No task is learnable from the real training data (TRTR at chance level); utility undefined."
    return to_jsonable(result)


__all__ = [
    "ML_TASKS",
    "MLTask",
    "classification_metrics",
    "encode_features",
    "evaluate_ml_utility",
    "evaluate_task",
    "utility_ratios",
]
