"""SDV metadata built from the BedSynth schema (SDV 2.x unified ``Metadata``).

Three column sets are supported:

* ``training``   - train columns only (what the synthesizer learns)
* ``evaluation`` - train + derived columns (what quality reports compare)
* ``full``       - key + train + derived columns, key set as primary key (documentation)

``ordinal=True`` declares naturally ordered categoricals (occupancy_level, staffing_level)
with SDV's ``ordinal`` sdtype so they are encoded with an order-preserving transformer.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

from sdv.metadata import Metadata

from bedsynth.schema import ID_PREFIX, ID_WIDTH, TABLES, column_specs, key_column

COLUMN_SETS = ("training", "evaluation", "full")


def _roles(columns: str) -> set[str]:
    if columns == "training":
        return {"train"}
    if columns == "evaluation":
        return {"train", "derived"}
    if columns == "full":
        return {"key", "train", "derived"}
    raise ValueError(f"Unknown column set '{columns}'. Expected one of {COLUMN_SETS}")


def add_table_metadata(
    metadata: Metadata,
    table: str,
    columns: str = "training",
    ordinal: bool = True,
    exclude: Iterable[str] = (),
) -> Metadata:
    """Add ``table`` (restricted to a column set) to an existing SDV ``Metadata`` object."""
    excluded = set(exclude)
    metadata.add_table(table)
    for spec in column_specs(table, _roles(columns)):
        if spec.name in excluded:
            continue
        sdtype = spec.sdtype
        kwargs: dict = {}
        if sdtype == "ordinal":
            if ordinal and spec.categories:
                kwargs["range_values"] = list(spec.categories)
            else:
                sdtype = "categorical"
        elif sdtype == "numerical" and spec.decimals is not None:
            kwargs["decimal_places"] = int(spec.decimals)
        elif sdtype == "id":
            kwargs["regex_format"] = f"{ID_PREFIX[table]}[0-9]{{{ID_WIDTH[table]}}}"
        metadata.add_column(spec.name, table_name=table, sdtype=sdtype, **kwargs)
    if columns == "full":
        metadata.set_primary_key(key_column(table), table_name=table)
    return metadata


def build_metadata(
    table: str,
    columns: str = "training",
    ordinal: bool = True,
    exclude: Iterable[str] = (),
) -> Metadata:
    """Single-table SDV metadata for ``table``."""
    metadata = Metadata()
    add_table_metadata(metadata, table, columns=columns, ordinal=ordinal, exclude=exclude)
    metadata.validate()
    return metadata


def build_project_metadata(columns: str = "full", ordinal: bool = True) -> Metadata:
    """Metadata describing both BedSynth tables (no relationships: they are synthesised independently)."""
    metadata = Metadata()
    for table in TABLES:
        add_table_metadata(metadata, table, columns=columns, ordinal=ordinal)
    metadata.validate()
    return metadata


def save_metadata(path: Path | str, columns: str = "full", ordinal: bool = True) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    build_project_metadata(columns=columns, ordinal=ordinal).save_to_json(str(path))
    return path


def metadata_dict(table: str, columns: str = "training", ordinal: bool = True) -> dict:
    return build_metadata(table, columns=columns, ordinal=ordinal).to_dict()
