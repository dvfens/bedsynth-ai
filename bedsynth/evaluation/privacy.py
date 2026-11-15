"""Privacy of synthetic records (DESIGN.md §9.5).

The reference data is simulated, so privacy is assessed *methodologically*: the same
checks would be run on a facility's real extract. The question is whether the model
copies training records instead of generalising from them.

* **Exact matches** - share of synthetic rows identical to a training row (train columns),
  compared with the natural duplicate rate of real holdout rows.
* **Distance to closest record (DCR)** - rows are encoded with min-max scaled numerics
  (clipped to [0, 1]) and one-hot categoricals weighted 0.5, and compared with the L1
  (Manhattan) metric divided by the number of columns. This equals sdmetrics' DCR
  definition (mean per-column distance; numeric |Δ| / range, categorical mismatch = 1)
  but runs on ``sklearn.neighbors.NearestNeighbors`` in seconds instead of minutes.
  Synthetic→train distances are compared with holdout→train distances: a model that
  generalises produces synthetic rows no closer to the training data than unseen real
  rows are.
* **sdmetrics cross-check** - ``DCRBaselineProtection`` and ``DCROverfittingProtection``
  on a small seeded subsample (their pairwise implementation is quadratic: ~10 s per
  metric at 1,000 rows on the development machine). They are reported, not scored, so the
  score does not depend on whether the time budget allowed them.

``score`` = mean of the four protection components (each in [0, 1], 1 = no sign of
copying): exact-match protection, DCR median ratio, close-record protection and
overfitting protection (see ``METRIC_DESCRIPTIONS``).

**Aggregate disclosure** (reported, not scored). The generator allocates rows with the training
split's joint frequencies of its allocation strata, so the output can reveal those frequencies
(e.g. the share of critical ICU days) more precisely than any independent real sample would -
a competitive-sensitivity concern of the brief that record-level DCR cannot see. With
``disclosure_columns`` the result compares the TVD between the synthetic and the training mix
of those columns with the TVD between the holdout and the training mix (the real sampling
noise): ``disclosure_ratio`` well below 1 means the output mirrors the training frequencies.
"""
from __future__ import annotations

import logging
import time
import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from bedsynth.config import get_seed, section
from bedsynth.evaluation.scoring import clip01, mean_or_none, to_jsonable
from bedsynth.evaluation.statistical import ensure_derived, subsample
from bedsynth.schema import column_specs, training_columns

logger = logging.getLogger(__name__)

DEFAULT_PRIVACY_MAX_ROWS = 5_000
SDMETRICS_MAX_ROWS = 1_000          # sdmetrics DCR is O(n^2) in pandas: keep the cross-check within budget
MAX_REFERENCE_ROWS = 50_000         # cap on training rows used as the nearest-neighbour reference
CLOSE_RECORD_QUANTILE = 0.05        # synthetic rows closer than the holdout 5th percentile are "close"

METRIC_DESCRIPTIONS: dict[str, str] = {
    "exact_match_rate": "Share of synthetic rows identical to a training row on all train columns.",
    "holdout_exact_match_rate": "Share of real holdout rows identical to a training row (natural duplicate baseline).",
    "dcr": "Distance to closest training record: mean per-column distance (numeric |Δ|/range, category mismatch = 1).",
    "dcr_median_ratio": "Median synthetic DCR divided by median holdout DCR (>= 1: synthetic rows are no closer than unseen real rows).",
    "close_record_share": "Share of synthetic rows closer to the training data than the 5th percentile of holdout DCR (expected 0.05).",
    "closer_to_training_share": "Share of synthetic rows closer to a training row than to a holdout row, "
                                "with equally sized reference sets (expected 0.5).",
    "exact_match_protection": "1 - excess exact-match rate over the holdout baseline.",
    "dcr_ratio_protection": "min(1, DCR median ratio).",
    "close_record_protection": "1 - excess close-record share over the expected 0.05 (rescaled to [0, 1]).",
    "overfitting_protection": "min(1, 2 * (1 - closer-to-training share)) - the DCROverfittingProtection formula.",
    "sdmetrics_DCRBaselineProtection": "sdmetrics: median synthetic DCR relative to random data (cross-check, not scored).",
    "sdmetrics_DCROverfittingProtection": "sdmetrics: closer-to-training share converted to a score (cross-check, not scored).",
    "aggregate_disclosure": "TVD of the synthetic vs training joint mix of the generator's allocation strata, next to "
                            "the holdout vs training TVD (real noise); a ratio well below 1 means exact training "
                            "frequencies are reproduced (reported, not scored).",
}


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------
class RecordEncoder:
    """Encode train columns so that ``L1 distance / n_columns`` is the sdmetrics DCR.

    Numerics are min-max scaled on the training range and clipped to [0, 1] (a constant
    training column becomes a 0/1 "differs from the constant" flag); categoricals are
    one-hot encoded over the schema vocabulary with weight 0.5, so a category mismatch
    contributes exactly 1. Missing numerics are imputed with the training median.
    """

    def __init__(self, table: str):
        self.table = table
        self.columns = training_columns(table)
        specs = {s.name: s for s in column_specs(table, ["train"])}
        self.numeric = [c for c in self.columns if specs[c].sdtype == "numerical"]
        self.categories = {
            c: list(specs[c].categories or ()) for c in self.columns if specs[c].sdtype in ("categorical", "ordinal")
        }
        self.minimum: dict[str, float] = {}
        self.span: dict[str, float] = {}
        self.median: dict[str, float] = {}

    def fit(self, reference: pd.DataFrame) -> "RecordEncoder":
        for column in self.numeric:
            values = pd.to_numeric(reference[column], errors="coerce").astype(float)
            self.minimum[column] = float(values.min())
            self.span[column] = float(values.max() - values.min())
            self.median[column] = float(values.median())
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        blocks = []
        for column in self.numeric:
            values = pd.to_numeric(df[column], errors="coerce").astype(float).fillna(self.median[column]).to_numpy()
            if self.span[column] > 0:
                blocks.append(np.clip((values - self.minimum[column]) / self.span[column], 0.0, 1.0))
            else:
                blocks.append((values != self.minimum[column]).astype(float))
        numeric = np.column_stack(blocks) if blocks else np.empty((len(df), 0))
        one_hot = []
        for column, cats in self.categories.items():
            values = df[column].astype(object).where(df[column].notna(), None).to_numpy()
            matches = values[:, None] == np.array(cats, dtype=object)[None, :]
            one_hot.append(0.5 * matches.astype(float))
        return np.hstack([numeric, *one_hot]) / len(self.columns)


def _nearest_distance(reference: np.ndarray, query: np.ndarray) -> np.ndarray:
    """L1 distance of every query row to its nearest reference row."""
    model = NearestNeighbors(n_neighbors=1, metric="manhattan", algorithm="brute").fit(reference)
    distances, _ = model.kneighbors(query)
    return distances[:, 0]


# ---------------------------------------------------------------------------
# Exact matches
# ---------------------------------------------------------------------------
def _row_hashes(df: pd.DataFrame, table: str) -> pd.Series:
    """Hash of each row's train columns after canonicalising dtypes (ints, rounded floats, strings)."""
    canonical = pd.DataFrame(index=df.index)
    for spec in column_specs(table, ["train"]):
        column = df[spec.name]
        if spec.sdtype == "numerical":
            decimals = spec.decimals if spec.decimals is not None else 6
            canonical[spec.name] = pd.to_numeric(column, errors="coerce").astype(float).round(decimals)
        else:
            canonical[spec.name] = column.astype(str)
    return pd.util.hash_pandas_object(canonical, index=False)


def exact_match_rate(query: pd.DataFrame, reference: pd.DataFrame, table: str) -> float | None:
    """Share of ``query`` rows whose train columns exactly match some ``reference`` row."""
    if len(query) == 0:
        return None
    reference_hashes = set(_row_hashes(reference, table).to_numpy().tolist())
    return float(_row_hashes(query, table).isin(reference_hashes).mean())


# ---------------------------------------------------------------------------
# sdmetrics cross-check
# ---------------------------------------------------------------------------
def _sdmetrics_cross_check(
    train: pd.DataFrame, synthetic: pd.DataFrame, holdout: pd.DataFrame, table: str, rows: int, seed: int
) -> dict[str, Any]:
    """Run sdmetrics DCR metrics on seeded subsamples of ``rows`` rows (errors captured)."""
    from sdmetrics.single_table import DCRBaselineProtection, DCROverfittingProtection

    from bedsynth.synthesis.metadata import build_metadata

    columns = training_columns(table)
    # sdmetrics' DCR ignores 'ordinal' columns, so declare ordered categoricals as categorical.
    metadata = build_metadata(table, "training", ordinal=False).to_dict()
    n = int(min(rows, len(train), len(synthetic), len(holdout)))
    train_s = subsample(train[columns], n, seed)
    syn_s = subsample(synthetic[columns], n, seed + 1)
    hold_s = subsample(holdout[columns], n, seed + 2)
    out: dict[str, Any] = {"rows": n}
    previous_seed = getattr(DCRBaselineProtection, "_seed", None)
    try:
        DCRBaselineProtection._seed = seed  # makes the random-data baseline reproducible
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            start = time.perf_counter()
            baseline = DCRBaselineProtection.compute_breakdown(train_s, syn_s, metadata, table)
            out["DCRBaselineProtection"] = {**baseline, "seconds": round(time.perf_counter() - start, 3)}
            start = time.perf_counter()
            overfit = DCROverfittingProtection.compute_breakdown(train_s, syn_s, hold_s, metadata, table)
            out["DCROverfittingProtection"] = {**overfit, "seconds": round(time.perf_counter() - start, 3)}
    except Exception as exc:  # noqa: BLE001 - the cross-check is optional
        logger.warning("sdmetrics DCR cross-check failed: %s", exc)
        out["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        DCRBaselineProtection._seed = previous_seed
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def _joint_tvd(reference: pd.DataFrame, other: pd.DataFrame, columns: list[str]) -> float | None:
    """TVD between the joint frequencies of ``columns`` in two frames (``None`` if undefined)."""
    if not columns or reference.empty or other.empty:
        return None
    p = reference.groupby(columns, dropna=False, observed=True).size() / len(reference)
    q = other.groupby(columns, dropna=False, observed=True).size() / len(other)
    index = p.index.union(q.index)
    return 0.5 * float((p.reindex(index, fill_value=0.0) - q.reindex(index, fill_value=0.0)).abs().sum())


def aggregate_disclosure(train: pd.DataFrame, synthetic: pd.DataFrame, holdout: pd.DataFrame | None,
                         columns: list[str] | tuple[str, ...]) -> dict[str, Any]:
    """How closely the synthetic joint mix of ``columns`` mirrors the training mix (see module docstring)."""
    usable = [c for c in columns if c in train.columns and c in synthetic.columns]
    synthetic_tvd = _joint_tvd(train, synthetic, usable)
    holdout_tvd = _joint_tvd(train, holdout, usable) if holdout is not None else None
    ratio = (synthetic_tvd / holdout_tvd) if synthetic_tvd is not None and holdout_tvd else None
    return {"columns": usable, "synthetic_vs_train_tvd": synthetic_tvd, "holdout_vs_train_tvd": holdout_tvd,
            "disclosure_ratio": ratio,
            "mirrors_training_frequencies": None if ratio is None else bool(ratio < 0.5)}


def _distance_summary(values: np.ndarray) -> dict[str, float | None]:
    if values.size == 0:
        return {"median": None, "mean": None, "p5": None, "min": None}
    return {"median": float(np.median(values)), "mean": float(values.mean()),
            "p5": float(np.percentile(values, 5)), "min": float(values.min())}


def evaluate_privacy(
    real_train: pd.DataFrame,
    synthetic: pd.DataFrame,
    table: str,
    real_holdout: pd.DataFrame | None = None,
    *,
    max_rows: int | None = None,
    seed: int | None = None,
    run_sdmetrics: bool = True,
    sdmetrics_rows: int | None = None,
    disclosure_columns: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Privacy protection of ``synthetic`` relative to the training data.

    ``max_rows`` caps the synthetic and holdout query sets (default
    ``settings.evaluation.privacy_max_rows``). ``sdmetrics_rows`` caps the quadratic sdmetrics
    DCR cross-check (default ``settings.evaluation.sdmetrics_privacy_rows``, else
    :data:`SDMETRICS_MAX_ROWS`). Without ``real_holdout`` only the exact-match component is
    available (the DCR baselines need unseen real rows). ``disclosure_columns`` (the generator's
    allocation strata) adds the unscored ``aggregate_disclosure`` block.
    """
    start = time.perf_counter()
    seed = get_seed() if seed is None else int(seed)
    evaluation_settings = section("evaluation") or {}
    if max_rows is None:
        max_rows = int(evaluation_settings.get("privacy_max_rows") or DEFAULT_PRIVACY_MAX_ROWS)
    if sdmetrics_rows is None:
        configured = evaluation_settings.get("sdmetrics_privacy_rows")
        sdmetrics_rows = SDMETRICS_MAX_ROWS if configured is None else int(configured)
    columns = training_columns(table)
    train = ensure_derived(real_train, table)[columns].reset_index(drop=True)
    syn = subsample(ensure_derived(synthetic, table)[columns].reset_index(drop=True), max_rows, seed)
    holdout = None
    if real_holdout is not None and len(real_holdout) > 0:
        holdout = subsample(ensure_derived(real_holdout, table)[columns].reset_index(drop=True), max_rows, seed + 1)

    syn_exact = exact_match_rate(syn, train, table)
    holdout_exact = exact_match_rate(holdout, train, table) if holdout is not None else None
    baseline_exact = holdout_exact or 0.0
    exact_protection = None
    if syn_exact is not None:
        exact_protection = clip01(1.0 - max(0.0, syn_exact - baseline_exact) / max(1.0 - baseline_exact, 1e-9))

    components: dict[str, float | None] = {
        "exact_match_protection": exact_protection,
        "dcr_ratio_protection": None,
        "close_record_protection": None,
        "overfitting_protection": None,
    }
    dcr: dict[str, Any] = {"note": "requires real holdout data"}
    if holdout is not None and len(syn) > 0:
        reference = subsample(train, MAX_REFERENCE_ROWS, seed + 2)
        encoder = RecordEncoder(table).fit(reference)
        reference_x, syn_x, hold_x = encoder.transform(reference), encoder.transform(syn), encoder.transform(holdout)
        syn_dcr = _nearest_distance(reference_x, syn_x)
        hold_dcr = _nearest_distance(reference_x, hold_x)
        syn_median, hold_median = float(np.median(syn_dcr)), float(np.median(hold_dcr))
        ratio = (syn_median / hold_median) if hold_median > 0 else (1.0 if syn_median == 0 else None)
        threshold = float(np.quantile(hold_dcr, CLOSE_RECORD_QUANTILE))
        close_share = float(np.mean(syn_dcr < threshold))

        # Overfitting test with equally sized train / holdout references (as DCROverfittingProtection).
        train_equal = subsample(train, len(holdout), seed + 3)
        to_train = _nearest_distance(encoder.transform(train_equal), syn_x)
        to_holdout = _nearest_distance(hold_x, syn_x)
        closer_to_training = float(np.mean(to_train < to_holdout))

        components.update(
            dcr_ratio_protection=clip01(ratio),
            close_record_protection=clip01(
                1.0 - max(0.0, close_share - CLOSE_RECORD_QUANTILE) / (1.0 - CLOSE_RECORD_QUANTILE)),
            overfitting_protection=clip01(2.0 * (1.0 - closer_to_training)),
        )
        dcr = {
            "metric": "L1 / n_columns (sdmetrics DCR definition)",
            "reference_rows": len(reference),
            "synthetic_to_train": _distance_summary(syn_dcr),
            "holdout_to_train": _distance_summary(hold_dcr),
            "median_ratio": ratio,
            "close_record_threshold": threshold,
            "close_record_share": close_share,
            "closer_to_training_share": closer_to_training,
            "equal_reference_rows": len(train_equal),
        }

    sdmetrics_result: dict[str, Any] = {"ran": False}
    if run_sdmetrics and holdout is not None and len(syn) > 0 and sdmetrics_rows > 0:
        sdmetrics_result = {"ran": True, **_sdmetrics_cross_check(train, syn, holdout, table, sdmetrics_rows, seed)}
    elif not run_sdmetrics:
        sdmetrics_result["reason"] = "disabled"
    else:
        sdmetrics_result["reason"] = "requires real holdout data"

    result = {
        "name": "privacy",
        "description": "Protection against copying training records: exact matches and distance to closest "
                       "record compared with unseen real (holdout) rows.",
        "table": table,
        "score": mean_or_none(components.values()),
        "components": components,
        "n_rows": {"train": len(train), "synthetic_used": len(syn), "holdout_used": 0 if holdout is None else len(holdout)},
        "exact_matches": {"synthetic_rate": syn_exact, "holdout_rate": holdout_exact},
        "dcr": dcr,
        "sdmetrics": sdmetrics_result,
        "aggregate_disclosure": None if not disclosure_columns else aggregate_disclosure(
            ensure_derived(real_train, table), ensure_derived(synthetic, table),
            None if real_holdout is None or len(real_holdout) == 0 else ensure_derived(real_holdout, table),
            list(disclosure_columns)),
        "note": "Reference data is simulated: privacy is assessed methodologically (same checks apply to a real extract).",
        "metric_descriptions": METRIC_DESCRIPTIONS,
        "seconds": round(time.perf_counter() - start, 3),
    }
    return to_jsonable(result)


__all__ = ["RecordEncoder", "aggregate_disclosure", "evaluate_privacy", "exact_match_rate"]
