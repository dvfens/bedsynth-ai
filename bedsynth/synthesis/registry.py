"""Trained-model registry (``<models_dir>/registry.json``) - DESIGN.md §7.4.

Layout::

    models/
      registry.json
      ward_daily/<experiment_id>/synthesizer.pkl
      ward_daily/<experiment_id>/model_card.json
      patient_stays/<experiment_id>/...

registry.json::

    {"ward_daily": {"default": "gc_tuned", "default_reason": "...",
                    "models": {"gc_tuned": {<model card>}, ...}},
     "patient_stays": {...}}

The registry is stateless: every call re-reads ``registry.json``, so a training script, the
evaluation script and a running API always agree. The file is not small - each card carries the
per-segment constraint and marginal details, about 750 KB for the 10 trained models - and every
call pays for parsing it (``list_models`` and ``to_dict`` parse it more than once). Writes are
atomic (temporary file + ``os.replace``) so a reader never sees a half-written file.

A card's ``path`` (the model directory) is stored project-relative
(:func:`bedsynth.config.project_relative`) and is informational only: loading always uses
:meth:`ModelRegistry.model_dir`, i.e. ``models_dir/<table>/<experiment_id>``.
"""
from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from bedsynth.config import get_path, get_settings, project_relative
from bedsynth.schema import TABLES
from bedsynth.synthesis.models import BedSynthSynthesizer, json_safe

LOGGER = logging.getLogger(__name__)

REGISTRY_FILE = "registry.json"
PICKLE_FILE = "synthesizer.pkl"
CARD_FILE = "model_card.json"
EXPERIMENT_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_\-]{0,63}$"
"""Valid experiment / model ids (same pattern as the API's ``MODEL_ID_PATTERN``)."""
_EXPERIMENT_ID_RE = re.compile(EXPERIMENT_ID_PATTERN)

# Card fields surfaced by list_models() (the full card stays available via get_card()).
_SUMMARY_FIELDS: tuple[str, ...] = (
    "model", "synthesizer", "description", "segment_by", "segment_columns", "segment_label", "n_segments",
    "constraint_groups", "effective_constraint_groups", "training_rows", "fit_seconds", "fitted_at", "sdv_version",
    "distributions", "metrics",
)


class ModelNotFoundError(LookupError):
    """No trained model is available for the requested table / model id."""


def _train_hint(table: str, experiment_id: str | None = None) -> str:
    experiments = f" --experiments {experiment_id}" if experiment_id else ""
    return f"Train it with `python scripts/02_train_synthesizers.py --tables {table}{experiments}`."


class ModelRegistry:
    """Index of trained synthesizers, their model cards and the per-table default model.

    Args:
        models_dir: Root directory of the models (default: ``get_path("models_dir")``, which
            honours the ``BEDSYNTH_MODELS_DIR`` environment variable).
    """

    def __init__(self, models_dir: Path | str | None = None):
        self.models_dir = Path(models_dir) if models_dir is not None else get_path("models_dir")
        self.path = self.models_dir / REGISTRY_FILE

    def __repr__(self) -> str:
        return f"ModelRegistry({str(self.models_dir)!r})"

    # ------------------------------------------------------------------ file handling
    @staticmethod
    def _check_table(table: str) -> None:
        if table not in TABLES:
            raise KeyError(f"Unknown table '{table}'. Known tables: {list(TABLES)}")

    @staticmethod
    def _empty_entry() -> dict[str, Any]:
        return {"default": None, "default_reason": "", "models": {}}

    def _read(self) -> dict[str, Any]:
        """Registry content with an entry for every table (missing/corrupt file -> empty)."""
        data: dict[str, Any] = {}
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text(encoding="utf-8")) or {}
            except (OSError, json.JSONDecodeError) as exc:
                LOGGER.error("Cannot read model registry %s (%s); treating it as empty", self.path, exc)
                data = {}
        for table in TABLES:
            entry = data.get(table) or {}
            data[table] = {**self._empty_entry(), **entry}
            data[table]["models"] = dict(data[table].get("models") or {})
        return data

    def _write(self, data: dict[str, Any]) -> None:
        self.models_dir.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text(json.dumps(json_safe(data), indent=2), encoding="utf-8")
        os.replace(tmp, self.path)

    # ------------------------------------------------------------------ paths
    def model_dir(self, table: str, experiment_id: str) -> Path:
        """Directory holding ``synthesizer.pkl`` and ``model_card.json`` of one experiment.

        Raises:
            ValueError: ``experiment_id`` is not a plain id (:data:`EXPERIMENT_ID_PATTERN`, the same
                pattern the API enforces: letters, digits, ``_`` and ``-`` only), so it can never
                point outside ``models_dir`` (``..``, separators or Windows drive prefixes such as
                ``D:evil``).
        """
        self._check_table(table)
        if not isinstance(experiment_id, str) or not _EXPERIMENT_ID_RE.fullmatch(experiment_id):
            raise ValueError(f"Invalid experiment id {experiment_id!r}: expected {EXPERIMENT_ID_PATTERN}")
        return self.models_dir / table / experiment_id

    def is_trained(self, table: str, experiment_id: str) -> bool:
        """Whether the experiment's synthesizer file exists on disk."""
        try:
            return (self.model_dir(table, experiment_id) / PICKLE_FILE).exists()
        except ValueError:
            return False

    # ------------------------------------------------------------------ mutation
    def register(self, table: str, experiment_id: str, model_card: dict[str, Any]) -> None:
        """Add or replace the model card of ``experiment_id`` (call after saving the model)."""
        self._check_table(table)
        data = self._read()
        card = json_safe(dict(model_card))
        card.setdefault("table", table)
        card.setdefault("experiment_id", experiment_id)
        card["registered_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        card["path"] = project_relative(self.model_dir(table, experiment_id))  # informational only
        data[table]["models"][experiment_id] = card
        self._write(data)
        LOGGER.info("Registered %s/%s in %s", table, experiment_id, self.path)

    def update_card(self, table: str, experiment_id: str, **fields: Any) -> None:
        """Merge extra fields (e.g. evaluation ``metrics``) into a registered model card."""
        self._check_table(table)
        data = self._read()
        models = data[table]["models"]
        if experiment_id not in models:
            raise ModelNotFoundError(f"Model '{experiment_id}' is not registered for '{table}'.")
        models[experiment_id].update(json_safe(fields))
        self._write(data)

    def unregister(self, table: str, experiment_id: str) -> None:
        """Remove a model card (files on disk are left untouched)."""
        self._check_table(table)
        data = self._read()
        data[table]["models"].pop(experiment_id, None)
        if data[table]["default"] == experiment_id:
            data[table]["default"], data[table]["default_reason"] = None, ""
        self._write(data)

    def set_default(self, table: str, experiment_id: str, reason: str = "") -> None:
        """Make ``experiment_id`` the default model of ``table`` (must be registered and on disk)."""
        self._check_table(table)
        data = self._read()
        if experiment_id not in data[table]["models"] or not self.is_trained(table, experiment_id):
            raise ModelNotFoundError(
                f"Cannot make '{experiment_id}' the default for '{table}': it is not trained. "
                + _train_hint(table, experiment_id)
            )
        data[table]["default"] = experiment_id
        data[table]["default_reason"] = reason
        data[table]["default_set_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self._write(data)
        LOGGER.info("Default model for %s set to %s (%s)", table, experiment_id, reason or "no reason given")

    # ------------------------------------------------------------------ queries
    def trained_models(self, table: str) -> list[str]:
        """Registered experiment ids of ``table`` whose synthesizer file exists."""
        self._check_table(table)
        return [m for m in self._read()[table]["models"] if self.is_trained(table, m)]

    def default_model_id(self, table: str) -> str | None:
        """Registry default, else settings.synthesis.default_experiment if trained, else any trained model."""
        self._check_table(table)
        entry = self._read()[table]
        trained = [m for m in entry["models"] if self.is_trained(table, m)]
        if entry.get("default") in trained:
            return entry["default"]
        configured = (get_settings().get("synthesis", {}).get("default_experiment") or {}).get(table)
        if configured in trained:
            return configured
        return sorted(trained)[0] if trained else None

    def get_card(self, table: str, experiment_id: str | None = None) -> dict[str, Any]:
        """Full model card of a registered model (default model when ``experiment_id`` is None)."""
        model_id = experiment_id or self.default_model_id(table)
        cards = self._read()[table]["models"]
        if model_id is None:
            raise ModelNotFoundError(f"No trained model available for table '{table}'. " + _train_hint(table))
        if model_id not in cards:
            raise ModelNotFoundError(
                f"No registered model '{model_id}' for table '{table}'. " + _train_hint(table, model_id)
            )
        return dict(cards[model_id])

    def list_models(self, table: str | None = None) -> list[dict[str, Any]]:
        """Summaries of the registered models (one dict per model, JSON-serialisable).

        Each item has ``table, model_id, experiment_id, is_default, trained, path`` (the model
        directory, project-relative) plus the main model-card fields.
        """
        tables = [table] if table is not None else list(TABLES)
        data = self._read()
        items: list[dict[str, Any]] = []
        for name in tables:
            self._check_table(name)
            default = self.default_model_id(name)
            for experiment_id, card in sorted(data[name]["models"].items()):
                item = {
                    "table": name,
                    "model_id": experiment_id,
                    "experiment_id": experiment_id,
                    "is_default": experiment_id == default,
                    "trained": self.is_trained(name, experiment_id),
                    "path": project_relative(self.model_dir(name, experiment_id)),
                }
                item.update({k: card.get(k) for k in _SUMMARY_FIELDS if k in card})
                items.append(item)
        return json_safe(items)

    def load(self, table: str, experiment_id: str | None = None) -> BedSynthSynthesizer:
        """Load a trained synthesizer (default model when ``experiment_id`` is None).

        Raises:
            ModelNotFoundError: no trained model for the table / unknown experiment id.
        """
        self._check_table(table)
        model_id = experiment_id or self.default_model_id(table)
        if model_id is None:
            raise ModelNotFoundError(f"No trained model available for table '{table}'. " + _train_hint(table))
        if not self.is_trained(table, model_id):
            raise ModelNotFoundError(
                f"Model '{model_id}' for table '{table}' is not trained (no {PICKLE_FILE} under "
                f"{self.model_dir(table, model_id)}). " + _train_hint(table, model_id)
            )
        model = BedSynthSynthesizer.load(self.model_dir(table, model_id))
        if model.table != table:
            raise ModelNotFoundError(f"Model '{model_id}' was trained for '{model.table}', not '{table}'.")
        return model

    def to_dict(self) -> dict[str, Any]:
        """Whole registry (JSON-serialisable) with the effective default of every table."""
        data = self._read()
        for table in TABLES:
            data[table]["effective_default"] = self.default_model_id(table)
            data[table]["trained"] = self.trained_models(table)
        return json_safe(data)


__all__ = ["EXPERIMENT_ID_PATTERN", "ModelNotFoundError", "ModelRegistry"]
