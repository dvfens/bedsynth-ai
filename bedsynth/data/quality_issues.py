"""Dirty-data injection for the simulated raw extract (DESIGN.md §6.2).

Real hospital extracts are messy: wards are typed in several ways, dates arrive in local
formats, cells are blank, uploads are duplicated and sentinel values slip in. To make the
Week-1 cleaning step meaningful (and measurable), :func:`inject_quality_issues` applies
such defects to a *copy* of the clean simulator output and logs every single change, so
the cleaned data can be scored against the untouched ground truth.

Issue types (``ISSUE_TYPES``)
-----------------------------
``category_variant``     alias spelling / case / whitespace variant of a canonical category
                         (always normalisable with ``schema.normalize_category``)
``alt_date_format``      day-first local format: ward_daily ``date`` as ``DD/MM/YYYY``,
                         patient_stays timestamps as ``DD/MM/YYYY HH:MM``
``missing_value``        cell blanked in one of the configured ``missing_columns``
``invalid_value``        impossible value: ward_daily ``census_start=-1``, ``discharges=999``,
                         ``nurses_day_shift=0``, ``avg_bed_turnaround_min=0``, ``beds_closed=-2``;
                         patient_stays ``bed_wait_hours=-1``, ``bed_turnaround_minutes=99999``,
                         ``booking_lead_days=-5``
``timestamp_swap``       admission and discharge timestamps swapped (discharge before admission)
``duplicate_row``        ward_daily: exact copy of a row; patient_stays: a second record with the
                         same ``stay_id`` (half exact copies, half re-entered with one categorical
                         field spelled differently). Copies are inserted *after* the original,
                         so "keep first" de-duplication restores the ground truth.

Log format (``LOG_COLUMNS``): one row per injected issue with ``table``, ``row_key``
(``stay_id`` or ``"<ward_id>|<YYYY-MM-DD>"`` of the clean row), ``column`` (empty for
duplicates; ``"admission_datetime|discharge_datetime"`` for swaps), ``issue_type``,
``original_value`` and ``injected_value`` (as text; empty when missing). A cell is never
modified twice, so every logged cell differs from the ground truth by exactly one issue.
Each configured issue type is injected at least once when its rate is positive, which
keeps small test extracts representative.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Mapping

import numpy as np
import pandas as pd

from bedsynth.schema import (
    CATEGORY_ALIASES,
    DATE_FORMAT,
    DATETIME_FORMAT,
    PATIENT_STAYS,
    TABLES,
    WARD_DAILY,
    key_column,
)

__all__ = [
    "CATEGORY_VARIANT_COLUMNS",
    "INVALID_VALUES",
    "ISSUE_TYPES",
    "LOG_COLUMNS",
    "category_variant",
    "empty_log",
    "inject_quality_issues",
]

LOG_COLUMNS: tuple[str, ...] = ("table", "row_key", "column", "issue_type", "original_value", "injected_value")
ISSUE_TYPES: tuple[str, ...] = (
    "category_variant",
    "alt_date_format",
    "missing_value",
    "invalid_value",
    "timestamp_swap",
    "duplicate_row",
)

CATEGORY_VARIANT_COLUMNS: dict[str, tuple[str, ...]] = {
    WARD_DAILY: ("ward_id", "staffing_level", "occupancy_level"),
    PATIENT_STAYS: ("ward_id", "admission_source", "separation_type", "discharge_destination", "delay_reason"),
}
DATE_COLUMNS: dict[str, tuple[str, ...]] = {
    WARD_DAILY: ("date",),
    PATIENT_STAYS: ("admission_datetime", "discharge_datetime"),
}
ALT_DATE_FORMAT = "%d/%m/%Y"
ALT_DATETIME_FORMAT = "%d/%m/%Y %H:%M"
INVALID_VALUES: dict[str, tuple[tuple[str, Any], ...]] = {
    WARD_DAILY: (
        ("census_start", -1),
        ("discharges", 999),
        ("nurses_day_shift", 0),
        ("avg_bed_turnaround_min", 0.0),
        ("beds_closed", -2),
    ),
    PATIENT_STAYS: (
        ("bed_wait_hours", -1.0),
        ("bed_turnaround_minutes", 99999),
        ("booking_lead_days", -5),
    ),
}
SWAP_COLUMNS: tuple[str, str] = ("admission_datetime", "discharge_datetime")
_RE_ENTERED_DUPLICATE_SHARE = 0.5          # patient_stays duplicates re-typed with an alias


def empty_log() -> pd.DataFrame:
    """An injection log with no rows (used when injection is disabled)."""
    return pd.DataFrame(columns=list(LOG_COLUMNS))


def _as_text(value: Any) -> str | None:
    """Log representation of a cell value (``None`` for missing)."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, (np.floating, float)):
        return repr(float(value))
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return str(int(value))
    return str(value)


def _issue_count(rate: float, n_rows: int, minimum: int = 1) -> int:
    """Number of cells/rows to corrupt: ``round(rate * n)``, at least ``minimum`` if rate > 0."""
    if rate is None or rate <= 0 or n_rows <= 0:
        return 0
    return int(min(n_rows, max(minimum, round(float(rate) * n_rows))))


def category_variant(column: str, canonical: str, rng: np.random.Generator) -> str:
    """A spelling variant of ``canonical`` that ``schema.normalize_category`` maps back to it.

    Picks one of the column's aliases (e.g. ``'gen med'`` for ``GMED``) and adds case and
    whitespace noise; never returns the canonical string itself.
    """
    aliases = [alias for alias, target in CATEGORY_ALIASES.get(column, {}).items() if target == canonical]
    if not aliases:
        aliases = [str(canonical).lower()]
    for _ in range(12):
        text = aliases[int(rng.integers(len(aliases)))]
        style = int(rng.integers(4))
        text = (text, text.upper(), text.title(), text.capitalize())[style]
        if rng.random() < 0.3:
            pad = " " * int(rng.integers(1, 3))
            text = pad + text if rng.random() < 0.5 else text + pad
        if text != canonical:
            return text
    return f" {canonical} "


class _Injector:
    """Applies issues to a copy of one clean table while keeping an exact log."""

    def __init__(self, clean: pd.DataFrame, table: str, rng: np.random.Generator) -> None:
        self.table = table
        self.clean = clean.reset_index(drop=True)
        self.dirty = self.clean.copy()
        self.rng = rng
        self.n = len(self.clean)
        self.keys = self._row_keys()
        self.used: dict[str, np.ndarray] = {c: np.zeros(self.n, dtype=bool) for c in self.clean.columns}
        self.log: list[dict[str, Any]] = []

    def _row_keys(self) -> list[str]:
        if self.table == PATIENT_STAYS:
            return self.clean[key_column(PATIENT_STAYS)].astype(str).tolist()
        return (self.clean["ward_id"].astype(str) + "|" + self.clean["date"].astype(str)).tolist()

    def pick(self, columns: tuple[str, ...], k: int) -> np.ndarray:
        """Choose up to ``k`` rows whose cells in ``columns`` are untouched, and reserve them."""
        free = np.ones(self.n, dtype=bool)
        for column in columns:
            free &= ~self.used[column]
        candidates = np.flatnonzero(free)
        k = min(k, len(candidates))
        if k <= 0:
            return np.array([], dtype=np.int64)
        rows = np.sort(self.rng.choice(candidates, size=k, replace=False))
        for column in columns:
            self.used[column][rows] = True
        return rows

    def record(self, row: int, column: str | None, issue_type: str, original: Any, injected: Any) -> None:
        self.log.append({
            "table": self.table,
            "row_key": self.keys[row],
            "column": column,
            "issue_type": issue_type,
            "original_value": _as_text(original),
            "injected_value": _as_text(injected),
        })

    def set_cell(self, row: int, column: str, value: Any, issue_type: str) -> None:
        if self.dirty[column].dtype != object:
            self.dirty[column] = self.dirty[column].astype(object)
        original = self.clean.at[row, column]
        self.dirty.at[row, column] = value
        self.record(row, column, issue_type, original, value)

    # -- issue families ---------------------------------------------------------
    def invalid_values(self, rate: float) -> None:
        """Sentinels/negative values plus (patient_stays) swapped timestamps."""
        kinds: list[tuple[str, ...]] = [(column,) for column, _ in INVALID_VALUES[self.table] if column in self.clean]
        if self.table == PATIENT_STAYS and all(c in self.clean for c in SWAP_COLUMNS):
            kinds.append(SWAP_COLUMNS)
        if not kinds:
            return
        total = _issue_count(rate, self.n, minimum=len(kinds))
        sentinels = dict(INVALID_VALUES[self.table])
        for i, columns in enumerate(kinds):
            share = total // len(kinds) + (1 if i < total % len(kinds) else 0)
            for row in self.pick(columns, share):
                if columns == SWAP_COLUMNS:
                    self.swap_timestamps(int(row))
                else:
                    self.set_cell(int(row), columns[0], sentinels[columns[0]], "invalid_value")

    def swap_timestamps(self, row: int) -> None:
        first, second = SWAP_COLUMNS
        for column in SWAP_COLUMNS:
            if self.dirty[column].dtype != object:
                self.dirty[column] = self.dirty[column].astype(object)
        a, b = self.clean.at[row, first], self.clean.at[row, second]
        self.dirty.at[row, first], self.dirty.at[row, second] = b, a
        self.record(row, f"{first}|{second}", "timestamp_swap", f"{a}|{b}", f"{b}|{a}")

    def missing_values(self, rate: float, columns: list[str]) -> None:
        for column in columns:
            if column not in self.clean:
                continue
            for row in self.pick((column,), _issue_count(rate, self.n)):
                self.set_cell(int(row), column, np.nan, "missing_value")

    def date_formats(self, rate: float) -> None:
        """Rewrite dates/timestamps in the day-first local format (per column)."""
        source, target = (DATE_FORMAT, ALT_DATE_FORMAT) if self.table == WARD_DAILY else (DATETIME_FORMAT, ALT_DATETIME_FORMAT)
        for column in DATE_COLUMNS[self.table]:
            if column not in self.clean:
                continue
            for row in self.pick((column,), _issue_count(rate, self.n)):
                value = self.clean.at[int(row), column]
                self.set_cell(int(row), column, datetime.strptime(str(value), source).strftime(target), "alt_date_format")

    def category_variants(self, rate: float) -> None:
        for column in CATEGORY_VARIANT_COLUMNS[self.table]:
            if column not in self.clean:
                continue
            for row in self.pick((column,), _issue_count(rate, self.n)):
                canonical = self.clean.at[int(row), column]
                self.set_cell(int(row), column, category_variant(column, str(canonical), self.rng), "category_variant")

    def duplicates(self, rate: float) -> None:
        """Insert copies of random rows somewhere after their original."""
        k = _issue_count(rate, self.n)
        if k == 0:
            return
        sources = np.sort(self.rng.choice(self.n, size=k, replace=False))
        copies: list[pd.Series] = []
        positions: list[float] = []
        for src in sources:
            src = int(src)
            row = self.dirty.iloc[src].copy()
            note = "exact copy"
            if self.table == PATIENT_STAYS and self.rng.random() < _RE_ENTERED_DUPLICATE_SHARE:
                untouched = [c for c in CATEGORY_VARIANT_COLUMNS[self.table] if c in row.index and not self.used[c][src]]
                if untouched:
                    column = untouched[int(self.rng.integers(len(untouched)))]
                    variant = category_variant(column, str(self.clean.at[src, column]), self.rng)
                    row[column] = variant
                    note = f"re-entered copy ({column}={variant!r})"
            copies.append(row)
            positions.append(float(self.rng.uniform(src + 0.5, self.n)))
            self.record(src, None, "duplicate_row", None, note)
        originals = self.dirty.assign(_position=np.arange(self.n, dtype=float))
        extra = pd.DataFrame(copies).astype(self.dirty.dtypes.to_dict()).assign(_position=positions)
        combined = pd.concat([originals, extra], ignore_index=True)
        self.dirty = combined.sort_values("_position", kind="stable").drop(columns="_position").reset_index(drop=True)

    def log_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.log, columns=list(LOG_COLUMNS)) if self.log else empty_log()


def _table_config(config: Mapping[str, Any] | None, table: str) -> dict[str, Any]:
    """Accept the per-table section or the whole ``data_quality`` section."""
    config = dict(config or {})
    if isinstance(config.get(table), Mapping):
        return dict(config[table])
    return config


def inject_quality_issues(df: pd.DataFrame, table: str, config: Mapping[str, Any] | None,
                          rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Inject realistic data-quality issues into a copy of a clean raw-format table.

    Args:
        df: clean table in ``schema.RAW_COLUMNS[table]`` format (not modified).
        table: ``'ward_daily'`` or ``'patient_stays'``.
        config: rates for this table (``settings.simulation.data_quality.<table>``:
            ``category_variant_rate``, ``alt_date_format_rate`` / ``alt_datetime_format_rate``,
            ``missing_rate`` + ``missing_columns``, ``duplicate_rate``, ``invalid_value_rate``);
            the whole ``data_quality`` section is accepted too.
        rng: numpy generator (the result is deterministic given its state).

    Returns:
        ``(dirty_df, log_df)``: the dirty table (same columns; duplicates inserted after
        their originals; modified columns become ``object`` dtype) and the injection log
        with ``LOG_COLUMNS``.
    """
    if table not in TABLES:
        raise KeyError(f"Unknown table '{table}'. Known tables: {list(TABLES)}")
    cfg = _table_config(config, table)
    injector = _Injector(df, table, rng)
    # Rarest issues first so they always find untouched cells; duplicates last (they copy
    # the already-dirty rows).
    injector.invalid_values(float(cfg.get("invalid_value_rate", 0.0)))
    injector.missing_values(float(cfg.get("missing_rate", 0.0)), list(cfg.get("missing_columns") or []))
    date_rate_key = "alt_date_format_rate" if table == WARD_DAILY else "alt_datetime_format_rate"
    injector.date_formats(float(cfg.get(date_rate_key, 0.0)))
    injector.category_variants(float(cfg.get("category_variant_rate", 0.0)))
    injector.duplicates(float(cfg.get("duplicate_rate", 0.0)))
    return injector.dirty, injector.log_frame()
