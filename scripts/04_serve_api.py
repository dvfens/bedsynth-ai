"""Serve the BedSynth AI REST API with uvicorn (Week 4).

Usage::

    python scripts/04_serve_api.py                      # http://127.0.0.1:8000
    python scripts/04_serve_api.py --port 8080 --reload # restart when a Python file in bedsynth/ changes
    python scripts/04_serve_api.py --host 0.0.0.0       # listen on every interface

Then open ``/ui`` for the interactive demo and ``/docs`` for the OpenAPI documentation.

The generation engine (SDV models) is loaded in a background thread at start-up so the first
request is fast; disable this with ``--no-preload``. ``/health`` answers at once in the meantime
(``status`` 'degraded', ``engine.state`` 'loading'). Without trained models the service still
starts: ``/health`` reports ``degraded`` and generation answers 503 with the command to run.

``--reload`` uses uvicorn's file watcher, which watches Python files only (``*.py`` under
``bedsynth/``); changes to ``config/settings.yaml`` need a manual restart.

Equivalent to ``uvicorn bedsynth.api.main:app --port 8000``.
"""
from __future__ import annotations

import argparse
import copy
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

APP_PATH = "bedsynth.api.main:app"
LOG_LEVELS = ("critical", "error", "warning", "info", "debug")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Command-line options."""
    parser = argparse.ArgumentParser(description="Serve the BedSynth AI API with uvicorn.")
    parser.add_argument("--host", default="127.0.0.1", help="Interface to bind (default: 127.0.0.1).")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on (default: 8000).")
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Restart automatically when a Python file under bedsynth/ changes (development). "
        "Changes to config/settings.yaml are not watched: restart the server by hand.",
    )
    parser.add_argument("--log-level", default="info", choices=LOG_LEVELS, help="Log level (default: info).")
    parser.add_argument("--no-preload", action="store_true", help="Load the generation engine on the first request instead of at start-up.")
    return parser.parse_args(argv)


def log_config(level: str) -> dict[str, Any]:
    """uvicorn's logging configuration extended with the ``bedsynth`` logger (applied in every worker)."""
    from uvicorn.config import LOGGING_CONFIG

    config = copy.deepcopy(LOGGING_CONFIG)
    config["loggers"]["bedsynth"] = {"handlers": ["default"], "level": level.upper(), "propagate": False}
    return config


def display_url(host: str, port: int) -> str:
    """Browser-friendly base URL for the bound address."""
    shown = "127.0.0.1" if host in ("0.0.0.0", "::", "") else host
    return f"http://{shown}:{port}"


def main(argv: list[str] | None = None) -> int:
    """Start uvicorn (blocks until the server stops)."""
    args = parse_args(argv)
    import uvicorn

    if args.no_preload:
        os.environ["BEDSYNTH_API_PRELOAD"] = "0"
    else:
        os.environ.setdefault("BEDSYNTH_API_PRELOAD", "1")

    url = display_url(args.host, args.port)
    banner = (
        "BedSynth AI API\n"
        f"  demo UI : {url}/ui\n"
        f"  API docs: {url}/docs\n"
        f"  health  : {url}/health\n"
        "Press Ctrl+C to stop.\n"
    )
    print(banner, flush=True)  # flush: uvicorn logs to stderr, keep the banner first

    uvicorn.run(
        APP_PATH,
        host=args.host,
        port=args.port,
        reload=args.reload,
        # uvicorn's reloaders watch *.py files only, so config/ (settings.yaml) is not listed.
        reload_dirs=[str(ROOT / "bedsynth")] if args.reload else None,
        log_level=args.log_level,
        log_config=log_config(args.log_level),
        app_dir=str(ROOT),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
