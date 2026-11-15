"""Configuration loading and project paths.

All modules obtain settings and directories through this module so that tests (and
deployments) can redirect every path with environment variables.

Paths written into artefacts (data-quality report, model cards, registry, pickled training
annotations, evidence files) go through :func:`project_relative`, so no artefact records the
machine's home directory or user name; readers turn a stored value back into a usable path with
:func:`resolve_project_path`.
"""
from __future__ import annotations

import copy
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SETTINGS_PATH = PROJECT_ROOT / "config" / "settings.yaml"
HOME_PREFIX = "~"
"""Stands for the home directory in stored paths outside the project (``~/...``)."""

# settings.paths key -> environment variable that overrides it
PATH_ENV_VARS = {
    "raw_dir": "BEDSYNTH_RAW_DIR",
    "processed_dir": "BEDSYNTH_PROCESSED_DIR",
    "synthetic_dir": "BEDSYNTH_SYNTHETIC_DIR",
    "models_dir": "BEDSYNTH_MODELS_DIR",
    "reports_dir": "BEDSYNTH_REPORTS_DIR",
    "figures_dir": "BEDSYNTH_FIGURES_DIR",
    "results_dir": "BEDSYNTH_RESULTS_DIR",
}


@lru_cache(maxsize=1)
def _load_settings(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def get_settings() -> dict[str, Any]:
    """Return a deep copy of the settings (callers may mutate their copy freely)."""
    path = os.environ.get("BEDSYNTH_SETTINGS", str(DEFAULT_SETTINGS_PATH))
    return copy.deepcopy(_load_settings(path))


def get_path(key: str, create: bool = True) -> Path:
    """Resolve a configured directory (``settings.paths.<key>``), honouring env overrides."""
    env_var = PATH_ENV_VARS.get(key)
    if env_var and os.environ.get(env_var):
        path = Path(os.environ[env_var])
    else:
        paths = get_settings()["paths"]
        if key not in paths:
            raise KeyError(f"Unknown path key '{key}'. Known keys: {sorted(paths)}")
        path = PROJECT_ROOT / paths[key]
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def get_seed() -> int:
    return int(get_settings()["project"]["seed"])


def section(name: str) -> dict[str, Any]:
    """Return one top-level settings section (deep copy)."""
    return get_settings().get(name, {})


def _home_directory() -> Path | None:
    """The user's home directory, resolved (``None`` when it cannot be determined)."""
    try:
        return Path.home().resolve()
    except (RuntimeError, OSError, KeyError):
        return None


def project_relative(path: Path | str | os.PathLike[str]) -> str:
    """Form of ``path`` to store in an artefact (POSIX separators, never the home directory).

    * inside :data:`PROJECT_ROOT`: relative to it, e.g. ``data/processed/ward_daily_train.csv``
      (the project root itself is ``.``);
    * elsewhere under the home directory: the home prefix replaced by ``~``, e.g.
      ``~/AppData/Local/Temp/run/raw``;
    * any other absolute path: unchanged apart from POSIX separators.

    A relative ``path`` is taken to be project-relative already (the convention of
    ``settings.paths`` and of every stored value) and is only normalised, so the function is
    idempotent. :func:`resolve_project_path` turns a stored value back into a usable path.
    """
    text = os.fspath(path)
    if not text:
        return text
    candidate = Path(text).expanduser()
    if not candidate.is_absolute():
        return candidate.as_posix()
    try:
        resolved = candidate.resolve(strict=False)
    except (OSError, RuntimeError):
        resolved = Path(os.path.abspath(candidate))
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        pass
    home = _home_directory()
    if home is not None:
        try:
            inside_home = resolved.relative_to(home).as_posix()
        except ValueError:
            pass
        else:
            return HOME_PREFIX if inside_home == "." else f"{HOME_PREFIX}/{inside_home}"
    return resolved.as_posix()


def resolve_project_path(value: Path | str | os.PathLike[str]) -> Path:
    """Usable path for a value written by :func:`project_relative` (or an older absolute value).

    ``~/...`` expands to the home directory, a relative value is resolved against
    :data:`PROJECT_ROOT` and an absolute value is returned unchanged.
    """
    path = Path(os.fspath(value)).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path
