"""Statistical fidelity (DESIGN.md §9.1) and the fast quick-quality check (§9.3).

``evaluate_statistical`` is the thorough Week-3 assessment. Its ``score`` is SDV's
QualityReport overall score (column shapes + column pair trends) computed on a seeded
subsample of ``settings.evaluation.sdv_quality_max_rows`` rows; the DiagnosticReport and a
set of transparent statistics the report can cite are added alongside:

* numeric columns - KS statistic, normalised Wasserstein distance, range coverage and
  mean / std / p5 / p50 / p95 of real and synthetic values;
* categorical columns - total variation distance (TVD), category coverage, unseen
  categories and both distributions;
* pairs - mean |Δ| of the Pearson and Spearman correlation matrices and of Cramér's V
  for categorical pairs, each with the five worst pairs.

If SDV fails, the score falls back to the mean of the column-shape and pair-trend scores
computed here (``score_source`` says which one was used).

``quick_quality`` is a numpy/scipy-only approximation (< 2 s for 5k rows) used by the API
and for scenario subsets: column shapes (1 - KS / 1 - TVD), pair trends
(1 - mean |Δ Spearman| / 2) and, optionally, the hard-rule validity rate.

The primitives (``ks_statistic``, ``tv_distance``, ``cramers_v`` ...) are shared with the
operational-pattern checks.

**Compared columns.** Train + derived columns, *except* derived columns that merely relabel a
single categorical train column (``ward_type`` and ``bed_capacity`` from ``ward_id``,
``season`` from ``month``, ``is_weekend`` from ``day_of_week``, ``admission_season`` from
``admission_month``): their shapes and their pairs with the source column are reproduced
by construction, so including them double-counts the source column and inflates both SDV
properties (:data:`RELABELLED_COLUMNS`).
"""
from __future__ import annotations

import logging
import time
import warnings
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

from bedsynth.config import get_seed, section
from bedsynth.evaluation.scoring import clip01, complement_score, mean_or_none, to_jsonable
from bedsynth.schema import (
    PATIENT_STAYS,
    WARD_DAILY,
    column_names,
    column_specs,
    derive,
    derived_columns,
    training_columns,
)

logger = logging.getLogger(__name__)

MISSING_CATEGORY = "<missing>"
DEFAULT_SDV_MAX_ROWS = 10_000
DEFAULT_QUICK_MAX_ROWS = 20_000

#: Derived columns that are a deterministic function of ONE categorical train column (a
#: relabelling or coarsening). They are left out of the fidelity metrics: their shape and their
#: pair with the source column are reproduced by construction.
RELABELLED_COLUMNS: dict[str, tuple[str, ...]] = {
    WARD_DAILY: ("ward_type", "bed_capacity", "season", "is_weekend"),
    PATIENT_STAYS: ("ward_type", "admission_season"),
}

METRIC_DESCRIPTIONS: dict[str, str] = {
    "sdv_quality_score": "SDV QualityReport overall score: mean of Column Shapes and Column Pair Trends (1 = identical).",
    "sdv_column_shapes": "SDV Column Shapes: mean of KSComplement (numeric) and TVComplement (categorical) per column.",
    "sdv_column_pair_trends": "SDV Column Pair Trends: mean CorrelationSimilarity / ContingencySimilarity over column pairs.",
    "sdv_diagnostic_score": "SDV DiagnosticReport: data validity (boundary / category adherence) and table structure.",
    "ks_statistic": "Two-sample Kolmogorov-Smirnov statistic: max distance between the real and synthetic CDFs (0 = identical).",
    "wasserstein_normalized": "Earth mover's distance between real and synthetic values divided by the real value range.",
    "range_coverage": "Share of the real [min, max] range covered by the synthetic values.",
    "tv_distance": "Total variation distance between real and synthetic category frequencies (0 = identical).",
    "category_coverage": "Share of real categories that appear in the synthetic data.",
    "correlation_mean_abs_diff": "Mean absolute difference between real and synthetic correlation coefficients over numeric column pairs.",
    "cramers_v_mean_abs_diff": "Mean absolute difference between real and synthetic Cramér's V over categorical column pairs.",
    "column_shapes_score": "Mean of 1 - KS (numeric) and 1 - TVD (categorical) over all columns.",
    "pair_trends_score": "1 - mean |Δ Spearman| / 2 over numeric column pairs.",
    "validity": "Share of synthetic rows passing every hard operational rule (bedsynth.validation).",
}


# ---------------------------------------------------------------------------
# Frame preparation
# ---------------------------------------------------------------------------
def evaluation_columns(table: str) -> list[str]:
    """Columns compared by the evaluation: train + derived columns in canonical order, minus
    the single-column relabellings of :data:`RELABELLED_COLUMNS`."""
    relabelled = set(RELABELLED_COLUMNS.get(table, ()))
    return [c for c in column_names(table, ["train", "derived"]) if c not in relabelled]


def column_kinds(table: str, columns: Iterable[str]) -> tuple[list[str], list[str]]:
    """Split ``columns`` into (numeric, categorical) using the schema sdtypes."""
    wanted = set(columns)
    numeric, categorical = [], []
    for spec in column_specs(table, ["train", "derived"]):
        if spec.name not in wanted:
            continue
        if spec.sdtype == "numerical":
            numeric.append(spec.name)
        elif spec.sdtype in ("categorical", "ordinal", "boolean"):
            categorical.append(spec.name)
    return numeric, categorical


def ensure_derived(df: pd.DataFrame, table: str) -> pd.DataFrame:
    """Return ``df`` with every derived column, recomputing them when any is missing."""
    if all(c in df.columns for c in derived_columns(table)):
        return df
    missing_train = [c for c in training_columns(table) if c not in df.columns]
    if missing_train:
        raise KeyError(f"Cannot evaluate {table}: missing train columns {missing_train}")
    return derive(table, df)


def prepare_frame(df: pd.DataFrame, table: str) -> pd.DataFrame:
    """Derived copy of ``df`` restricted to the evaluation columns, with a fresh RangeIndex."""
    full = ensure_derived(df, table)
    return full[evaluation_columns(table)].reset_index(drop=True)


def subsample(df: pd.DataFrame, max_rows: int | None, seed: int) -> pd.DataFrame:
    """Seeded random subsample of at most ``max_rows`` rows (unchanged if already small)."""
    if max_rows is None or len(df) <= max_rows:
        return df
    return df.sample(n=int(max_rows), random_state=seed).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Univariate primitives
# ---------------------------------------------------------------------------
def numeric_values(series: pd.Series) -> np.ndarray:
    """Finite float values of a series (non-numeric and NaN dropped)."""
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float, na_value=np.nan)
    return values[np.isfinite(values)]


def ks_statistic(real: pd.Series | np.ndarray, synthetic: pd.Series | np.ndarray) -> float | None:
    """Two-sample Kolmogorov-Smirnov statistic (max |F_real - F_synthetic|), exact and fast."""
    r = np.sort(numeric_values(pd.Series(real)))
    s = np.sort(numeric_values(pd.Series(synthetic)))
    if r.size == 0 or s.size == 0:
        return None
    grid = np.concatenate([r, s])
    cdf_r = np.searchsorted(r, grid, side="right") / r.size
    cdf_s = np.searchsorted(s, grid, side="right") / s.size
    return float(np.max(np.abs(cdf_r - cdf_s)))


def category_frequencies(series: pd.Series) -> pd.Series:
    """Relative frequency of each category (missing values counted as their own category)."""
    values = series.astype(object).where(series.notna(), MISSING_CATEGORY)
    return values.value_counts(normalize=True, sort=False)


def tv_distance(real: pd.Series, synthetic: pd.Series) -> float | None:
    """Total variation distance ``0.5 * sum |p - q|`` between two categorical distributions."""
    if len(real) == 0 or len(synthetic) == 0:
        return None
    p, q = category_frequencies(real), category_frequencies(synthetic)
    index = p.index.union(q.index)
    return 0.5 * float((p.reindex(index, fill_value=0.0) - q.reindex(index, fill_value=0.0)).abs().sum())


def normalized_wasserstein(real: pd.Series, synthetic: pd.Series) -> float | None:
    """Wasserstein-1 distance divided by the real value range (scale-free, 0 = identical)."""
    r, s = numeric_values(real), numeric_values(synthetic)
    if r.size == 0 or s.size == 0:
        return None
    span = float(r.max() - r.min())
    scale = span if span > 0 else max(abs(float(r.mean())), 1.0)
    return float(wasserstein_distance(r, s)) / scale


def range_coverage(real: pd.Series, synthetic: pd.Series) -> float | None:
    """Share of the real [min, max] range covered by the synthetic [min, max] range."""
    r, s = numeric_values(real), numeric_values(synthetic)
    if r.size == 0 or s.size == 0:
        return None
    r_min, r_max, s_min, s_max = r.min(), r.max(), s.min(), s.max()
    span = r_max - r_min
    if span <= 0:
        return 1.0 if s_min <= r_min <= s_max else 0.0
    uncovered = max(s_min - r_min, 0.0) + max(r_max - s_max, 0.0)
    return clip01(1.0 - uncovered / span)


def category_coverage(real: pd.Series, synthetic: pd.Series) -> float | None:
    """Share of the categories observed in ``real`` that also occur in ``synthetic``."""
    real_categories = set(real.dropna().unique())
    if not real_categories:
        return None
    return len(real_categories & set(synthetic.dropna().unique())) / len(real_categories)


def describe_numeric(series: pd.Series) -> dict[str, float | None]:
    """Mean, std, 5th/50th/95th percentiles, min and max of a numeric series."""
    values = numeric_values(series)
    if values.size == 0:
        return {k: None for k in ("mean", "std", "p5", "p50", "p95", "min", "max")}
    p5, p50, p95 = np.percentile(values, [5, 50, 95])
    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
        "p5": float(p5),
        "p50": float(p50),
        "p95": float(p95),
        "min": float(values.min()),
        "max": float(values.max()),
    }


# ---------------------------------------------------------------------------
# Pairwise primitives
# ---------------------------------------------------------------------------
def correlation_matrix(df: pd.DataFrame, columns: list[str], method: str = "pearson") -> np.ndarray:
    """Pearson or Spearman correlation matrix (rows with missing values dropped).

    Spearman is Pearson on average ranks. Entries involving a constant column are NaN.
    """
    if method not in ("pearson", "spearman"):
        raise ValueError(f"Unknown correlation method '{method}'")
    data = df[columns].apply(pd.to_numeric, errors="coerce").astype(float).dropna()
    if len(data) < 3:
        return np.full((len(columns), len(columns)), np.nan)
    if method == "spearman":
        data = data.rank(method="average")
    values = data.to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        centred = values - values.mean(axis=0)
        norms = np.sqrt((centred ** 2).sum(axis=0))
        corr = (centred.T @ centred) / np.outer(norms, norms)
    corr[~np.isfinite(corr)] = np.nan
    return np.clip(corr, -1.0, 1.0)


def correlation_difference(
    real: pd.DataFrame, synthetic: pd.DataFrame, columns: list[str], method: str = "pearson", top_k: int = 5
) -> dict[str, Any]:
    """Mean |Δ| between real and synthetic correlation matrices, plus the ``top_k`` worst pairs.

    Pairs whose real correlation is undefined (constant real column) are skipped; an
    undefined synthetic correlation counts as 0.
    """
    columns = [c for c in columns if c in real.columns and c in synthetic.columns]
    if len(columns) < 2:
        return {"method": method, "n_pairs": 0, "mean_abs_diff": None, "max_abs_diff": None, "worst_pairs": []}
    rc = correlation_matrix(real, columns, method)
    sc = correlation_matrix(synthetic, columns, method)
    pairs = []
    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            if np.isnan(rc[i, j]):
                continue
            s_value = 0.0 if np.isnan(sc[i, j]) else float(sc[i, j])
            pairs.append((columns[i], columns[j], float(rc[i, j]), s_value, abs(float(rc[i, j]) - s_value)))
    if not pairs:
        return {"method": method, "n_pairs": 0, "mean_abs_diff": None, "max_abs_diff": None, "worst_pairs": []}
    diffs = np.array([p[4] for p in pairs])
    worst = sorted(pairs, key=lambda p: p[4], reverse=True)[:top_k]
    return {
        "method": method,
        "n_pairs": len(pairs),
        "mean_abs_diff": float(diffs.mean()),
        "max_abs_diff": float(diffs.max()),
        "worst_pairs": [
            {"columns": [a, b], "real": r, "synthetic": s, "abs_diff": d} for a, b, r, s, d in worst
        ],
    }


def cramers_v(x: pd.Series, y: pd.Series) -> float | None:
    """Cramér's V association between two categorical series (``None`` if either is constant)."""
    if len(x) == 0:
        return None
    x_codes, x_levels = pd.factorize(x.astype(object).where(x.notna(), MISSING_CATEGORY))
    y_codes, y_levels = pd.factorize(y.astype(object).where(y.notna(), MISSING_CATEGORY))
    n_x, n_y = len(x_levels), len(y_levels)
    if n_x < 2 or n_y < 2:
        return None
    observed = np.bincount(x_codes * n_y + y_codes, minlength=n_x * n_y).reshape(n_x, n_y).astype(float)
    n = observed.sum()
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0)) / n
    chi2 = float(((observed - expected) ** 2 / expected).sum())
    return float(np.sqrt(chi2 / (n * (min(n_x, n_y) - 1))))


def cramers_v_difference(
    real: pd.DataFrame, synthetic: pd.DataFrame, columns: list[str], top_k: int = 5
) -> dict[str, Any]:
    """Mean |Δ Cramér's V| over categorical pairs (real-constant pairs skipped) + worst pairs."""
    columns = [c for c in columns if c in real.columns and c in synthetic.columns]
    pairs = []
    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            a, b = columns[i], columns[j]
            r_value = cramers_v(real[a], real[b])
            if r_value is None:
                continue
            s_value = cramers_v(synthetic[a], synthetic[b])
            s_value = 0.0 if s_value is None else s_value
            pairs.append((a, b, r_value, s_value, abs(r_value - s_value)))
    if not pairs:
        return {"n_pairs": 0, "mean_abs_diff": None, "max_abs_diff": None, "worst_pairs": []}
    diffs = np.array([p[4] for p in pairs])
    worst = sorted(pairs, key=lambda p: p[4], reverse=True)[:top_k]
    return {
        "n_pairs": len(pairs),
        "mean_abs_diff": float(diffs.mean()),
        "max_abs_diff": float(diffs.max()),
        "worst_pairs": [
            {"columns": [a, b], "real": r, "synthetic": s, "abs_diff": d} for a, b, r, s, d in worst
        ],
    }


# ---------------------------------------------------------------------------
# Column summaries
# ---------------------------------------------------------------------------
def numeric_column_summary(column: str, real: pd.Series, synthetic: pd.Series) -> dict[str, Any]:
    """Distribution comparison of one numeric column."""
    ks = ks_statistic(real, synthetic)
    return {
        "column": column,
        "ks_statistic": ks,
        "ks_complement": complement_score(ks),
        "wasserstein_normalized": normalized_wasserstein(real, synthetic),
        "range_coverage": range_coverage(real, synthetic),
        "real": describe_numeric(real),
        "synthetic": describe_numeric(synthetic),
    }


def categorical_column_summary(column: str, real: pd.Series, synthetic: pd.Series) -> dict[str, Any]:
    """Distribution comparison of one categorical / boolean column."""
    tvd = tv_distance(real, synthetic)
    real_categories = set(real.dropna().unique())
    unseen = sorted({v for v in synthetic.dropna().unique() if v not in real_categories}, key=str)
    return {
        "column": column,
        "tv_distance": tvd,
        "tv_complement": complement_score(tvd),
        "category_coverage": category_coverage(real, synthetic),
        "unseen_categories": unseen,
        "real_distribution": category_frequencies(real).sort_index(key=lambda idx: idx.map(str)).to_dict(),
        "synthetic_distribution": category_frequencies(synthetic).sort_index(key=lambda idx: idx.map(str)).to_dict(),
    }


def column_shape_scores(real: pd.DataFrame, synthetic: pd.DataFrame, table: str) -> dict[str, float | None]:
    """Per-column shape score: 1 - KS (numeric) or 1 - TVD (categorical)."""
    numeric, categorical = column_kinds(table, [c for c in real.columns if c in synthetic.columns])
    scores: dict[str, float | None] = {}
    for column in numeric:
        scores[column] = complement_score(ks_statistic(real[column], synthetic[column]))
    for column in categorical:
        scores[column] = complement_score(tv_distance(real[column], synthetic[column]))
    return scores


# ---------------------------------------------------------------------------
# SDV reports
# ---------------------------------------------------------------------------
def _sdv_reports(real: pd.DataFrame, synthetic: pd.DataFrame, table: str, max_rows: int, seed: int) -> dict[str, Any]:
    """SDV QualityReport + DiagnosticReport on seeded subsamples (errors captured, never raised)."""
    from sdv.evaluation import evaluate_quality, run_diagnostic

    from bedsynth.synthesis.metadata import build_metadata

    out: dict[str, Any] = {"rows_used": {"real": min(len(real), max_rows), "synthetic": min(len(synthetic), max_rows)}}
    real_s, syn_s = subsample(real, max_rows, seed), subsample(synthetic, max_rows, seed + 1)
    excluded = [c for c in column_names(table, ["train", "derived"]) if c not in real.columns]
    metadata = build_metadata(table, "evaluation", exclude=excluded)
    out["excluded_columns"] = excluded
    start = time.perf_counter()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            report = evaluate_quality(real_s, syn_s, metadata, verbose=False)
        properties = report.get_properties()
        shapes = report.get_details("Column Shapes")
        pairs = report.get_details("Column Pair Trends")
        worst_pairs = pairs.dropna(subset=["Score"]).sort_values("Score").head(10)
        out["quality"] = {
            "score": float(report.get_score()),
            "properties": dict(zip(properties["Property"], properties["Score"])),
            "column_shapes": [
                {"column": row.Column, "metric": row.Metric, "score": row.Score} for row in shapes.itertuples()
            ],
            "worst_column_pairs": [
                {"columns": [row["Column 1"], row["Column 2"]], "metric": row["Metric"], "score": row["Score"]}
                for _, row in worst_pairs.iterrows()
            ],
            "seconds": round(time.perf_counter() - start, 3),
        }
    except Exception as exc:  # noqa: BLE001 - fall back to the in-house column metrics
        logger.warning("SDV QualityReport failed for %s: %s", table, exc)
        out["quality"] = {"score": None, "error": f"{type(exc).__name__}: {exc}"}

    start = time.perf_counter()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            diagnostic = run_diagnostic(real_s, syn_s, metadata, verbose=False)
        properties = diagnostic.get_properties()
        out["diagnostic"] = {
            "score": float(diagnostic.get_score()),
            "properties": dict(zip(properties["Property"], properties["Score"])),
            "seconds": round(time.perf_counter() - start, 3),
        }
    except Exception as exc:  # noqa: BLE001
        logger.warning("SDV DiagnosticReport failed for %s: %s", table, exc)
        out["diagnostic"] = {"score": None, "error": f"{type(exc).__name__}: {exc}"}
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def evaluate_statistical(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    table: str,
    *,
    max_rows: int | None = None,
    seed: int | None = None,
    run_sdv: bool = True,
    top_k: int = 5,
) -> dict[str, Any]:
    """Statistical fidelity of ``synthetic`` against ``real`` (fully derived tables).

    ``max_rows`` caps the SDV QualityReport subsample (default
    ``settings.evaluation.sdv_quality_max_rows``); the in-house statistics use all rows.
    ``score`` is the SDV overall quality score, or the in-house fallback if SDV fails or
    ``run_sdv`` is False.
    """
    start = time.perf_counter()
    seed = get_seed() if seed is None else int(seed)
    if max_rows is None:
        max_rows = int((section("evaluation") or {}).get("sdv_quality_max_rows") or DEFAULT_SDV_MAX_ROWS)
    real_df, syn_df = prepare_frame(real, table), prepare_frame(synthetic, table)
    numeric, categorical = column_kinds(table, real_df.columns)

    numeric_summaries = [numeric_column_summary(c, real_df[c], syn_df[c]) for c in numeric]
    categorical_summaries = [categorical_column_summary(c, real_df[c], syn_df[c]) for c in categorical]
    correlations = {
        method: correlation_difference(real_df, syn_df, numeric, method=method, top_k=top_k)
        for method in ("pearson", "spearman")
    }
    association = cramers_v_difference(real_df, syn_df, categorical, top_k=top_k)

    shapes_score = mean_or_none(
        [s["ks_complement"] for s in numeric_summaries] + [s["tv_complement"] for s in categorical_summaries]
    )
    spearman_diff = correlations["spearman"]["mean_abs_diff"]
    pair_trends_score = mean_or_none([
        None if spearman_diff is None else 1.0 - spearman_diff / 2.0,
        None if association["mean_abs_diff"] is None else 1.0 - association["mean_abs_diff"],
    ])
    fallback_score = mean_or_none([shapes_score, pair_trends_score])

    sdv = _sdv_reports(real_df, syn_df, table, max_rows, seed) if run_sdv else {"skipped": True}
    sdv_score = (sdv.get("quality") or {}).get("score")
    score, source = (sdv_score, "sdv_quality_report") if sdv_score is not None else (fallback_score, "fallback_column_metrics")

    result = {
        "name": "statistical_fidelity",
        "description": "Distributional similarity of synthetic and real data: SDV QualityReport plus per-column "
                       "and pairwise statistics.",
        "table": table,
        "score": clip01(score),
        "score_source": source,
        "n_rows": {"real": len(real_df), "synthetic": len(syn_df)},
        "sdv": sdv,
        "summary": {
            "column_shapes_score": shapes_score,
            "pair_trends_score": pair_trends_score,
            "fallback_score": fallback_score,
            "mean_ks_statistic": mean_or_none(s["ks_statistic"] for s in numeric_summaries),
            "mean_wasserstein_normalized": mean_or_none(s["wasserstein_normalized"] for s in numeric_summaries),
            "mean_range_coverage": mean_or_none(s["range_coverage"] for s in numeric_summaries),
            "mean_tv_distance": mean_or_none(s["tv_distance"] for s in categorical_summaries),
            "mean_category_coverage": mean_or_none(s["category_coverage"] for s in categorical_summaries),
            "pearson_mean_abs_diff": correlations["pearson"]["mean_abs_diff"],
            "spearman_mean_abs_diff": spearman_diff,
            "cramers_v_mean_abs_diff": association["mean_abs_diff"],
            "worst_numeric_columns": [
                s["column"] for s in sorted(numeric_summaries, key=lambda s: -(s["ks_statistic"] or 0.0))[:top_k]
            ],
            "worst_categorical_columns": [
                s["column"] for s in sorted(categorical_summaries, key=lambda s: -(s["tv_distance"] or 0.0))[:top_k]
            ],
        },
        "numeric_columns": numeric_summaries,
        "categorical_columns": categorical_summaries,
        "correlations": correlations,
        "cramers_v": association,
        "metric_descriptions": METRIC_DESCRIPTIONS,
        "seconds": round(time.perf_counter() - start, 3),
    }
    return to_jsonable(result)


def _validity_rate(synthetic: pd.DataFrame, table: str) -> tuple[float | None, str | None]:
    """Hard-rule pass rate of ``synthetic`` (``None`` + note when validation is unavailable)."""
    from bedsynth import validation

    if len(synthetic) == 0:
        return None, "no rows"
    try:
        report = validation.validate_table(synthetic, table, include_soft=False)
    except NotImplementedError:
        return None, "bedsynth.validation is not implemented yet"
    return report.n_hard_valid / max(report.n_rows, 1), None


def quick_quality(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    table: str,
    *,
    include_validity: bool = True,
    max_rows: int | None = DEFAULT_QUICK_MAX_ROWS,
    seed: int | None = None,
    exclude_columns: Iterable[str] = (),
) -> dict[str, Any]:
    """Fast quality estimate (numpy/scipy only; < 2 s for 5k rows).

    Components: column shapes (mean of 1 - KS / 1 - TVD), column pair trends
    (1 - mean |Δ Spearman| / 2) and, if ``include_validity``, the hard-rule validity rate of
    ``synthetic``. ``score`` is the mean of the available components. ``exclude_columns`` are
    left out of the shape and pair metrics (e.g. columns whose mix the generator imposes).
    """
    start = time.perf_counter()
    seed = get_seed() if seed is None else int(seed)
    excluded = set(exclude_columns)
    columns = [c for c in evaluation_columns(table) if c not in excluded]
    real_df = subsample(ensure_derived(real, table)[columns].reset_index(drop=True), max_rows, seed)
    syn_full = ensure_derived(synthetic, table)
    syn_df = subsample(syn_full[columns].reset_index(drop=True), max_rows, seed + 1)
    if real_df.empty or syn_df.empty:
        return to_jsonable({
            "name": "quick_quality", "description": "Fast fidelity estimate (not computed: empty input).",
            "table": table, "score": None, "components": {}, "note": "real or synthetic data is empty",
            "n_rows": {"real": len(real_df), "synthetic": len(syn_df)},
        })
    numeric, _ = column_kinds(table, real_df.columns)

    column_scores = column_shape_scores(real_df, syn_df, table)
    shapes = mean_or_none(column_scores.values())
    spearman = correlation_difference(real_df, syn_df, numeric, method="spearman", top_k=3)
    trends = None if spearman["mean_abs_diff"] is None else clip01(1.0 - spearman["mean_abs_diff"] / 2.0)
    validity, validity_note = (None, "not requested")
    if include_validity:
        validity, validity_note = _validity_rate(syn_full, table)

    components = {"column_shapes": shapes, "column_pair_trends": trends, "validity": validity}
    worst = sorted((c for c in column_scores.items() if c[1] is not None), key=lambda item: item[1])[:5]
    result = {
        "name": "quick_quality",
        "description": "Fast fidelity estimate: column shapes (1-KS / 1-TVD), Spearman pair trends and hard-rule validity.",
        "table": table,
        "score": mean_or_none(components.values()),
        "components": components,
        "validity_note": validity_note,
        "column_scores": column_scores,
        "worst_columns": [{"column": c, "score": s} for c, s in worst],
        "worst_pairs": spearman["worst_pairs"],
        "excluded_columns": sorted(excluded),
        "n_rows": {"real": len(real_df), "synthetic": len(syn_df)},
        "metric_descriptions": {
            "column_shapes": METRIC_DESCRIPTIONS["column_shapes_score"],
            "column_pair_trends": METRIC_DESCRIPTIONS["pair_trends_score"],
            "validity": METRIC_DESCRIPTIONS["validity"],
        },
        "seconds": round(time.perf_counter() - start, 3),
    }
    return to_jsonable(result)


__all__ = [
    "METRIC_DESCRIPTIONS",
    "RELABELLED_COLUMNS",
    "category_coverage",
    "category_frequencies",
    "column_kinds",
    "column_shape_scores",
    "correlation_difference",
    "correlation_matrix",
    "cramers_v",
    "cramers_v_difference",
    "describe_numeric",
    "ensure_derived",
    "evaluate_statistical",
    "evaluation_columns",
    "ks_statistic",
    "normalized_wasserstein",
    "numeric_values",
    "prepare_frame",
    "quick_quality",
    "range_coverage",
    "subsample",
    "tv_distance",
]
