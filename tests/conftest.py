"""Shared pytest fixtures.

Fixture data comes from ``bedsynth.data.fixtures`` (fast, rule-satisfying, not realistic).
``isolated_paths`` redirects every configured directory into a temp folder so tests never
touch the real data/, models/ or reports/ folders.
"""
from __future__ import annotations

import os

os.environ.setdefault("TQDM_DISABLE", "1")

import pytest  # noqa: E402

from bedsynth.data.fixtures import make_patient_stays, make_ward_daily  # noqa: E402


@pytest.fixture(scope="session")
def ward_daily_df():
    return make_ward_daily(n_days=120, seed=11)


@pytest.fixture(scope="session")
def patient_stays_df():
    return make_patient_stays(n=1500, seed=12)


@pytest.fixture()
def isolated_paths(tmp_path, monkeypatch):
    """Point every BEDSYNTH_* directory env var at a fresh temp directory."""
    from bedsynth.config import PATH_ENV_VARS

    dirs = {}
    for key, env_var in PATH_ENV_VARS.items():
        directory = tmp_path / key
        directory.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv(env_var, str(directory))
        dirs[key] = directory
    return dirs
