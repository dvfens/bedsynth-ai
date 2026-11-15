"""BedSynth AI REST API (Week 4): FastAPI service, dataset store and demo UI.

* :mod:`bedsynth.api.main`    - ``create_app`` factory and the module-level ``app``
  (``uvicorn bedsynth.api.main:app``)
* :mod:`bedsynth.api.schemas` - pydantic request/response models (OpenAPI contract)
* :mod:`bedsynth.api.service` - service layer, dataset store and JSON helpers
* ``static/index.html``       - single-page demo served at ``/ui``

``create_app`` and ``app`` are resolved lazily so importing the package has no side effects.
"""
from __future__ import annotations

from typing import Any

__all__ = ["app", "create_app"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from bedsynth.api import main

        return getattr(main, name)
    raise AttributeError(f"module 'bedsynth.api' has no attribute '{name}'")
