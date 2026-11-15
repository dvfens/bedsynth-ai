"""Weekly Markdown reports and PNG figures (DESIGN.md §11).

=========================  ==============================================  =====================================
renderer                   input                                           output (``reports/``)
=========================  ==============================================  =====================================
``render_week1_report``    ``prepare_datasets`` data-quality report         ``week1_data_preparation.md``
                           (default: ``data_quality_report.json``)
``render_week2_report``    model cards (default: the registry)              ``week2_synthesis_pipeline.md``
``render_week3_report``    ``03_evaluate.py`` results (default: files)      ``week3_evaluation.md``
``render_week4_report``    ``05_api_demo.py`` summary (default: files)      ``week4_api_system.md``
=========================  ==============================================  =====================================

Figures are written to ``reports/figures`` (configured ``figures_dir``). The pipeline scripts
call these renderers through optional hooks after their main work has succeeded; each
renderer returns the path of the Markdown file it wrote. Called without arguments, every
renderer rebuilds its report from the saved artefacts (no training, evaluation or demo run).

Every report opens with a summary computed from the artefacts; numbers are never typed into the
text, and paths are printed project-relative (``_common.display_path``).
"""
from __future__ import annotations

from bedsynth.reporting.week1 import render_week1_report
from bedsynth.reporting.week2 import registry_cards, render_week2_report
from bedsynth.reporting.week3 import load_results, render_week3_report
from bedsynth.reporting.week4 import render_week4_report

__all__ = [
    "load_results",
    "registry_cards",
    "render_week1_report",
    "render_week2_report",
    "render_week3_report",
    "render_week4_report",
]
