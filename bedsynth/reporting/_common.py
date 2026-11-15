"""Shared helpers of the report renderers: formatting, Markdown tables, figures, paths.

matplotlib is imported lazily (and switched to the non-interactive ``Agg`` backend) the first
time a figure is drawn, so importing :mod:`bedsynth.reporting` stays cheap and never opens a
window. A figure that cannot be drawn is logged and skipped: the Markdown report is always
written.

Reports never print absolute paths: :func:`display_path` turns a path into a project-relative
POSIX path (or the bare file name outside the project), so a report contains no user name,
home directory or machine-specific location.
"""
from __future__ import annotations

import logging
import math
import os
import re
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath
from typing import Any, Callable, Iterable, Sequence

from bedsynth.config import PROJECT_ROOT, get_path

logger = logging.getLogger(__name__)

#: Colour cycle used by every figure (colour-blind friendly, Okabe-Ito based).
PALETTE: tuple[str, ...] = ("#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00", "#F0E442", "#000000")


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------
def is_number(value: Any) -> bool:
    """True for finite int/float values (bools excluded)."""
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def fmt_num(value: Any, digits: int = 3) -> str:
    """Number with ``digits`` decimals ('-' when undefined)."""
    if not is_number(value):
        return "-"
    if isinstance(value, int):
        return f"{value:,}"
    return f"{float(value):,.{digits}f}"


def fmt_int(value: Any) -> str:
    """Integer with thousands separators ('-' when undefined)."""
    return f"{int(value):,}" if is_number(value) else "-"


def fmt_pct(value: Any, digits: int = 1) -> str:
    """Share as a percentage ('-' when undefined)."""
    return f"{100.0 * float(value):.{digits}f} %" if is_number(value) else "-"


def md_escape(value: Any) -> str:
    """Text safe to place inside a Markdown table cell."""
    return str(value).replace("|", "\\|").replace("\n", " ")


def md_table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> list[str]:
    """GitHub-flavoured Markdown table (list of lines)."""
    lines = ["| " + " | ".join(md_escape(h) for h in headers) + " |",
             "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(md_escape(cell) for cell in row) + " |")
    return lines


def utc_now() -> str:
    """Current UTC time, ISO 8601 (seconds)."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def to_utc(timestamp: Any) -> str | None:
    """ISO timestamp converted to UTC with seconds precision (naive values are local time).

    Unparseable values are returned unchanged as text, missing ones as ``None``.
    """
    if timestamp is None or timestamp == "":
        return None
    try:
        moment = datetime.fromisoformat(str(timestamp))
    except (TypeError, ValueError):
        return str(timestamp)
    if moment.tzinfo is None:
        moment = moment.astimezone()
    return moment.astimezone(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
def output_dirs(reports_dir: Path | str | None = None, figures_dir: Path | str | None = None) -> tuple[Path, Path]:
    """Report and figure directories (configured ``reports_dir`` / ``figures_dir`` by default)."""
    reports = Path(reports_dir) if reports_dir is not None else get_path("reports_dir")
    figures = Path(figures_dir) if figures_dir is not None else get_path("figures_dir")
    reports.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    return reports, figures


def path_name(value: Any) -> str:
    """Last component of a path given as text (Windows or POSIX separators)."""
    text = str(value or "").rstrip("\\/")
    return re.split(r"[\\/]", text)[-1] if text else ""


def display_path(value: Path | str | None) -> str:
    """Path as printed in a report: project-relative POSIX path, else the bare file name.

    Absolute paths inside :data:`bedsynth.config.PROJECT_ROOT` become relative to it; paths
    outside it, home-relative paths (``~/...``, what :func:`bedsynth.config.project_relative`
    stores outside the project), absolute paths of another operating system (a Windows drive path
    read on POSIX) and relative paths that climb out of the project are reduced to their last
    component, so no report ever contains a user name, home directory or machine-specific location.
    ``None`` gives ``'-'``.
    """
    if value is None or str(value) == "":
        return "-"
    text = str(value)
    if text == "~" or text.startswith(("~/", "~\\")):
        return path_name(text)
    if Path(text).is_absolute():
        try:
            return Path(text).resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
        except (ValueError, OSError):
            return path_name(text)
    if PureWindowsPath(text).drive or text.startswith(("/", "\\")):
        return path_name(text)
    parts = [part for part in re.split(r"[\\/]", text) if part not in ("", ".")]
    if not parts or ".." in parts:
        return path_name(text)
    return "/".join(parts)


def is_inside_project(path: Path | str) -> bool:
    """Whether ``path`` lies inside the project root."""
    try:
        Path(path).resolve().relative_to(PROJECT_ROOT.resolve())
        return True
    except (ValueError, OSError):
        return False


def relative_link(target: Path, base_dir: Path) -> str:
    """Link to ``target`` relative to ``base_dir`` (forward slashes).

    Across Windows drives no relative link exists; the project-relative path (or the file name)
    is returned instead of an absolute path.
    """
    try:
        return Path(os.path.relpath(target, base_dir)).as_posix()
    except ValueError:  # different drives on Windows
        return display_path(target)


def project_ref(relative_path: str, report_dir: Path) -> str:
    """A project file as Markdown: a relative link when the report lives inside the project.

    ``relative_path`` is project-relative (e.g. ``docs/API.md``). The link target is computed
    from ``report_dir``, so it resolves from wherever the report is written inside the project;
    reports written elsewhere (tests, scratch renders) and missing files get code text only.
    """
    target = PROJECT_ROOT / relative_path
    if target.exists() and is_inside_project(report_dir):
        return f"[`{relative_path}`]({relative_link(target, report_dir)})"
    return f"`{relative_path}`"


def write_markdown(path: Path, lines: Sequence[str]) -> Path:
    """Write a Markdown report (UTF-8, trailing newline)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    logger.info("Report written to %s", path)
    return path


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def _pyplot() -> Any:
    import matplotlib

    if matplotlib.get_backend().lower() != "agg":
        matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def draw_figure(path: Path, draw: Callable[[Any], Any], *, size: tuple[float, float] = (8.0, 4.5)) -> Path | None:
    """Create a figure, let ``draw(fig)`` fill it, save it as PNG; ``None`` if drawing failed."""
    try:
        plt = _pyplot()
    except Exception as exc:  # noqa: BLE001 - figures are optional
        logger.warning("matplotlib unavailable (%s); figure %s skipped", exc, path.name)
        return None
    fig = plt.figure(figsize=size, dpi=110)
    try:
        draw(fig)
        fig.tight_layout()
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=110)
        return path
    except Exception as exc:  # noqa: BLE001 - a broken figure must not break the report
        logger.warning("Figure %s could not be drawn: %s", path.name, exc)
        return None
    finally:
        plt.close(fig)


def grouped_bars(ax: Any, categories: Sequence[str], series: dict[str, Sequence[float | None]],
                 *, ylabel: str = "", ylim: tuple[float, float] | None = (0.0, 1.0)) -> None:
    """Grouped bar chart: one group per category, one bar per series (None drawn as 0)."""
    n = max(1, len(series))
    width = 0.8 / n
    positions = range(len(categories))
    for i, (label, values) in enumerate(series.items()):
        heights = [float(v) if is_number(v) else 0.0 for v in values]
        offsets = [p - 0.4 + width * (i + 0.5) for p in positions]
        ax.bar(offsets, heights, width=width, label=label, color=PALETTE[i % len(PALETTE)])
    ax.set_xticks(list(positions))
    ax.set_xticklabels(categories, rotation=0)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.3)
    # below the axes, so bars reaching the top (e.g. privacy = 1.0) are never hidden
    ax.legend(fontsize=8, ncol=min(n, 6), frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12))


def figure_line(path: Path | None, report_dir: Path, caption: str) -> list[str]:
    """Markdown image line (with caption) or a note when the figure is missing."""
    if path is None:
        return [f"_Figure not available: {caption}_", ""]
    return [f"![{caption}]({relative_link(path, report_dir)})", "", f"*{caption}*", ""]


__all__ = [
    "PALETTE",
    "display_path",
    "draw_figure",
    "figure_line",
    "fmt_int",
    "fmt_num",
    "fmt_pct",
    "grouped_bars",
    "is_inside_project",
    "is_number",
    "md_table",
    "output_dirs",
    "path_name",
    "project_ref",
    "relative_link",
    "to_utc",
    "utc_now",
    "write_markdown",
]
