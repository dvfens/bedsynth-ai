"""Operational validity of synthetic samples (DESIGN.md §9.2).

The generator rejects hard-invalid rows, so its *output* is valid by construction. What
distinguishes models is how often they produce valid rows **on their own**, so the
Week-3 validity score is computed on raw, unfiltered model samples (``schema.derive``
applied, no rejection). The score is the hard-valid rate: the share of rows that pass
every hard rule of ``bedsynth.validation`` (§6.4). Per-rule violation rates for hard and
soft rules are reported so the Week-3 report can show *which* operational identities a
model struggles with. ``evaluate_model`` also runs this function on the generator output
(``basis="generator_output"``: hard-valid by construction, but the soft-rule rates show the
plausibility of the delivered data) and the evaluation script on the real holdout (the
reference soft-rule rates).
"""
from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
import pandas as pd

from bedsynth import validation
from bedsynth.evaluation.scoring import clip01, to_jsonable
from bedsynth.evaluation.statistical import ensure_derived

logger = logging.getLogger(__name__)

RULE_FIELDS = ("rule_id", "name", "severity", "description", "columns", "violations", "violation_rate")

METRIC_DESCRIPTIONS: dict[str, str] = {
    "hard_valid_rate": "Share of rows that pass every hard operational rule (the validity score).",
    "hard_violation_rate": "Share of rows that violate at least one hard rule.",
    "violation_rate": "Per rule: share of rows violating that rule.",
    "soft_clean_rate": "Share of rows that pass every soft (plausibility) rule.",
}


def _rule_summary(rule_result: dict[str, Any]) -> dict[str, Any]:
    """Keep the report-relevant fields of one rule result (row indices are dropped)."""
    summary = {key: rule_result.get(key) for key in RULE_FIELDS}
    summary["columns"] = list(summary.get("columns") or [])
    return summary


def _soft_clean_rate(df: pd.DataFrame, table: str) -> float | None:
    """Share of rows passing all soft rules (``None`` if the soft rules cannot be evaluated)."""
    try:
        rules = validation.get_rules(table, severity="soft")
    except NotImplementedError:
        return None
    if not rules or len(df) == 0:
        return None
    passes = np.ones(len(df), dtype=bool)
    for rule in rules:
        try:
            passes &= rule.check(df).reindex(df.index).fillna(False).to_numpy(dtype=bool)
        except Exception as exc:  # noqa: BLE001 - a broken soft rule must not hide the others
            logger.warning("Soft rule %s could not be evaluated: %s", rule.rule_id, exc)
            return None
    return float(passes.mean())


def evaluate_validity(synthetic: pd.DataFrame, table: str, *, basis: str = "unspecified") -> dict[str, Any]:
    """Hard/soft rule validity of ``synthetic`` (derived columns are recomputed if missing).

    ``basis`` documents what was validated (``raw_model_samples`` for unfiltered model
    samples, ``generator_output`` for rejection-filtered data). ``score`` = hard-valid rate
    (``None`` for an empty frame). Raises ``NotImplementedError`` while
    ``bedsynth.validation`` is unavailable.
    """
    start = time.perf_counter()
    df = ensure_derived(synthetic, table).reset_index(drop=True)
    report = validation.validate_table(df, table, include_soft=True)
    rules = [_rule_summary(r) for r in report.rule_results]
    hard = [r for r in rules if r["severity"] == "hard"]
    soft = [r for r in rules if r["severity"] == "soft"]
    n_rows = int(report.n_rows)
    hard_valid_rate = report.n_hard_valid / n_rows if n_rows else None
    worst = sorted((r for r in hard if (r["violations"] or 0) > 0), key=lambda r: -(r["violation_rate"] or 0.0))

    result = {
        "name": "validity",
        "description": "Share of synthetic rows that satisfy every hard operational rule, with per-rule violation rates.",
        "table": table,
        "basis": basis,
        "n_rows": n_rows,
        "score": clip01(hard_valid_rate),
        "hard_valid_rate": hard_valid_rate,
        "n_hard_valid": int(report.n_hard_valid),
        "hard_violation_rate": None if hard_valid_rate is None else 1.0 - hard_valid_rate,
        "worst_hard_rules": [
            {"rule_id": r["rule_id"], "name": r["name"], "violation_rate": r["violation_rate"]} for r in worst[:3]
        ],
        "hard_rules": hard,
        "soft_rules": soft,
        "hard_violation_rates": {r["rule_id"]: r["violation_rate"] for r in hard},
        "soft_violation_rates": {r["rule_id"]: r["violation_rate"] for r in soft},
        "soft_clean_rate": _soft_clean_rate(df, table),
        "metric_descriptions": METRIC_DESCRIPTIONS,
        "seconds": round(time.perf_counter() - start, 3),
    }
    if n_rows == 0:
        result["note"] = "No rows to validate."
    return to_jsonable(result)


__all__ = ["evaluate_validity"]
