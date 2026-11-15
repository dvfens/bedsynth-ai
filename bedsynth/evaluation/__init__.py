"""Evaluation layer (Week 3): fidelity, validity, operational patterns, utility, privacy, coverage.

Every function returns a JSON-serialisable dict (plain floats/ints/str/lists/dicts, no NaN -
``None`` instead) with a ``name``, a one-line ``description`` and a top-level ``"score"`` in
[0, 1] where a score is meaningful (``None`` otherwise). ``real``/``synthetic`` inputs are
fully derived tables (``schema.derive`` applied; derived columns are recomputed if
missing) with the same columns; key/time columns are ignored. See docs/DESIGN.md §9.

Submodules
----------
statistical   SDV QualityReport/Diagnostic + KS, Wasserstein, TVD, correlations, Cramér's V;
              ``quick_quality`` (numpy/scipy only) for the API and scenario subsets
validity      hard/soft rule pass rates of raw model samples (``bedsynth.validation``)
operational   occupancy, scheduling and patient-flow pattern checks
privacy       exact matches and distance to closest record vs a real holdout baseline
utility       ML utility: train-synthetic-test-real with HistGradientBoosting (size-matched TRTR)
optimization  decision agreement: SAA booking quotas (out-of-sample regret), policy evaluation,
              capacity planning
coverage      scenario catalogue coverage and per-scenario fidelity
references    real-holdout ceiling and no-learning floor scored with the same families
scoring       score primitives, JSON sanitisation, composite score and ``evaluate_model``
"""
from __future__ import annotations

from bedsynth.evaluation.coverage import evaluate_scenario_coverage
from bedsynth.evaluation.operational import calibrate_operational, evaluate_operational
from bedsynth.evaluation.optimization import evaluate_optimization_utility
from bedsynth.evaluation.privacy import evaluate_privacy
from bedsynth.evaluation.references import evaluate_references, normalised_scores
from bedsynth.evaluation.scoring import composite_score, evaluate_model, to_jsonable
from bedsynth.evaluation.statistical import evaluate_statistical, quick_quality
from bedsynth.evaluation.utility import evaluate_ml_utility
from bedsynth.evaluation.validity import evaluate_validity

__all__ = [
    "calibrate_operational",
    "composite_score",
    "evaluate_ml_utility",
    "evaluate_model",
    "evaluate_operational",
    "evaluate_optimization_utility",
    "evaluate_privacy",
    "evaluate_references",
    "evaluate_scenario_coverage",
    "evaluate_statistical",
    "evaluate_validity",
    "normalised_scores",
    "quick_quality",
    "to_jsonable",
]
