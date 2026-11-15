"""CSV I/O helpers with schema-aware dtype coercion and canonical file locations."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from bedsynth.config import get_path
from bedsynth.schema import TABLES, coerce_dtypes, order_columns

SPLITS: tuple[str, ...] = ("clean", "train", "holdout")

# Raw extracts contain alias strings such as "n/a" or "NA" that are legitimate category
# variants, so only empty cells are treated as missing when reading.
_READ_KW = dict(keep_default_na=False, na_values=[""], low_memory=False)


def _check(table: str) -> None:
    if table not in TABLES:
        raise KeyError(f"Unknown table '{table}'. Known tables: {list(TABLES)}")


def raw_path(table: str, raw_dir: Path | None = None) -> Path:
    _check(table)
    return Path(raw_dir or get_path("raw_dir")) / f"{table}_raw.csv"


def ground_truth_path(table: str, raw_dir: Path | None = None) -> Path:
    """Clean simulator output before dirty-data injection (used to score the cleaning step)."""
    _check(table)
    return Path(raw_dir or get_path("raw_dir")) / "_ground_truth" / f"{table}_true.csv"


def processed_path(table: str, split: str = "train", processed_dir: Path | None = None) -> Path:
    _check(table)
    if split not in SPLITS:
        raise ValueError(f"Unknown split '{split}'. Expected one of {SPLITS}")
    return Path(processed_dir or get_path("processed_dir")) / f"{table}_{split}.csv"


def read_csv_raw(path: Path | str) -> pd.DataFrame:
    """Read a CSV without any dtype coercion (all alias strings preserved)."""
    return pd.read_csv(path, **_READ_KW)


def read_table(path: Path | str, table: str) -> pd.DataFrame:
    """Read a CSV and coerce schema columns to canonical dtypes."""
    _check(table)
    df = pd.read_csv(path, **_READ_KW)
    return order_columns(coerce_dtypes(df, table), table)


def write_table(df: pd.DataFrame, path: Path | str, *, lineterminator: str | None = None) -> Path:
    """Write ``df`` as CSV (no index), creating the parent directory.

    ``lineterminator`` defaults to pandas' default, the platform line ending (CRLF on
    Windows), which the prepared data in ``data/processed`` uses. The API's dataset store
    passes ``"\\n"`` so a stored dataset is byte-identical to the CSV body of ``/generate``.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, lineterminator=lineterminator)
    return path


def load_processed(table: str, split: str = "train", processed_dir: Path | None = None) -> pd.DataFrame:
    path = processed_path(table, split, processed_dir)
    if not path.exists():
        raise FileNotFoundError(
            f"Processed data not found at {path}. Run `python scripts/01_prepare_data.py` first."
        )
    return read_table(path, table)
