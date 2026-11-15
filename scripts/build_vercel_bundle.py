"""Build a self-contained folder that deploys the BedSynth API and demo UI to Vercel.

Vercel runs the FastAPI app as one Python serverless function. Two things stop the repository
from deploying as it is:

* the trained models and processed reference data are not in git (``models/`` and ``data/`` are
  re-created by the pipeline), so a deployment from GitHub would have nothing to serve;
* ``requirements.txt`` installs PyTorch for CTGAN/TVAE, which is far larger than Vercel's
  function size limit.

This script copies what the API needs into ``dist/vercel`` (the package, settings, GaussianCopula
models, processed data and evaluation results), writes a requirements file without PyTorch, a
Vercel entry point and ``vercel.json``, and tightens the public request limits. CTGAN and TVAE
need PyTorch, so the deployed API serves the GaussianCopula models only (the default model of
both tables is ``gc_tuned``); run the API locally for the deep-learning models.

Usage::

    python scripts/build_vercel_bundle.py            # builds dist/vercel
    cd dist/vercel && npx vercel deploy --prod      # deploys it

Run the pipeline first so that ``models/`` and ``data/processed/`` exist.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

#: Models that need PyTorch, which the serverless bundle leaves out.
TORCH_MODELS = frozenset({"ctgan", "tvae"})

#: Runtime dependencies of the API without PyTorch. SDV itself is not listed: installing it from
#: PyPI pulls in ctgan, deepecho and torch, so the bundle vendors the (pure-Python) SDV package from
#: the build environment instead; SDV only imports CTGAN/TVAE when they are available.
REQUIREMENTS = """\
# BedSynth AI on Vercel - the API's runtime dependencies without PyTorch (no CTGAN/TVAE).
# SDV 2.2.0 is vendored in the bundle (see scripts/build_vercel_bundle.py).
rdt==2.1.0
copulas==0.15.0
sdmetrics==0.33.0
pandas==2.3.3
numpy==2.5.4
scipy==1.18.1
scikit-learn==1.9.1
fastapi==0.162.7
pydantic==2.14.3
pyyaml==6.0.3
boto3==1.44.3
cloudpickle==3.1.2
graphviz==0.21
tqdm==4.70.1
platformdirs==4.14.5
"""

#: SDV version the models were trained with (the vendored copy must match it).
SDV_VERSION = "2.2.0"

#: Vercel entry point: point writable paths at /tmp, then expose the FastAPI app.
ENTRYPOINT = '''\
"""Vercel entry point for the BedSynth AI API (built by scripts/build_vercel_bundle.py)."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# The deployment's file system is read-only apart from /tmp: generated datasets and any scratch
# output go there (they last only as long as the function instance).
for _var, _sub in {
    "BEDSYNTH_SYNTHETIC_DIR": "synthetic",
    "BEDSYNTH_RAW_DIR": "raw",
    "BEDSYNTH_FIGURES_DIR": "figures",
}.items():
    os.environ.setdefault(_var, f"/tmp/bedsynth/{_sub}")
os.environ.setdefault("BEDSYNTH_SETTINGS", str(ROOT / "config" / "settings.yaml"))
# Serverless instances should not start a background loader thread: the engine loads on the first
# request that needs it (/health still answers at once).
os.environ.setdefault("BEDSYNTH_API_PRELOAD", "0")

from urllib.parse import parse_qsl, urlencode  # noqa: E402

from bedsynth.api.main import app as bedsynth_app  # noqa: E402  (after the environment is set)

PATH_PARAM = "__path"


async def app(scope, receive, send):
    """Restore the request path Vercel's catch-all rewrite moved into ``?__path=`` (see vercel.json)."""
    if scope["type"] in ("http", "websocket"):
        pairs = parse_qsl(scope.get("query_string", b"").decode("latin-1"), keep_blank_values=True)
        original = [value for key, value in pairs if key == PATH_PARAM]
        if original:
            path = "/" + original[-1].lstrip("/")
            rest = urlencode([(key, value) for key, value in pairs if key != PATH_PARAM])
            scope = dict(scope, path=path, raw_path=path.encode("utf-8"), query_string=rest.encode("latin-1"))
    await bedsynth_app(scope, receive, send)


__all__ = ["app"]
'''

#: Limits for a public serverless deployment (the local defaults allow much larger requests).
PUBLIC_API_LIMITS = {
    "max_rows": 2000,
    "max_calendar_days": 400,
    "evaluate_max_rows": 2000,
    "time_budget_seconds": 45,
    "max_time_budget_seconds": 60,
    "max_attempts": 50,
    "max_pending_generations": 2,
    "generation_wait_seconds": 30,
}

VERCEL_CONFIG = {
    "$schema": "https://openapi.vercel.sh/vercel.json",
    "functions": {
        "api/index.py": {
            "maxDuration": 90,
            "memory": 2048,
            "includeFiles": "{bedsynth,sdv,sdv-*.dist-info,config,models,data,reports}/**",
            "excludeFiles": "{**/__pycache__/**,**/*.pyc}",
        }
    },
    # Vercel hands the function the rewritten path (/api/index), so the original one travels in a
    # query parameter that api/index.py moves back into the request before FastAPI routes it.
    "rewrites": [{"source": "/(.*)", "destination": "/api/index?__path=/$1"}],
}


def _copy_tree(src: Path, dst: Path) -> None:
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))


def _filtered_registry(registry_path: Path) -> dict:
    """registry.json without the PyTorch models (the bundle does not ship them)."""
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    for table, entry in registry.items():
        entry["models"] = {k: v for k, v in entry.get("models", {}).items() if k not in TORCH_MODELS}
        if entry.get("default") in TORCH_MODELS:
            raise SystemExit(f"The default model of {table} needs PyTorch, which the Vercel bundle leaves out.")
    return registry


def _vendor_sdv(out_dir: Path) -> None:
    """Copy the installed SDV package (and its metadata) into the bundle root."""
    import importlib.metadata

    dist = importlib.metadata.distribution("sdv")
    if dist.version != SDV_VERSION:
        raise SystemExit(f"SDV {dist.version} is installed but the models need SDV {SDV_VERSION}.")
    package_dir = Path(dist.locate_file("sdv"))
    dist_info = Path(dist.locate_file(f"sdv-{dist.version}.dist-info"))
    _copy_tree(package_dir, out_dir / "sdv")
    _copy_tree(dist_info, out_dir / dist_info.name)


def build(out_dir: Path) -> Path:
    models_dir = PROJECT_ROOT / "models"
    processed_dir = PROJECT_ROOT / "data" / "processed"
    for required in (models_dir / "registry.json", processed_dir):
        if not required.exists():
            raise SystemExit(f"{required.relative_to(PROJECT_ROOT)} is missing - run scripts/run_pipeline.py first.")

    # rebuild from scratch, but keep the Vercel project link (.vercel/) so redeploys update the same project
    out_dir.mkdir(parents=True, exist_ok=True)
    for child in out_dir.iterdir():
        if child.name == ".vercel":
            continue
        shutil.rmtree(child) if child.is_dir() else child.unlink()

    _copy_tree(PROJECT_ROOT / "bedsynth", out_dir / "bedsynth")
    _vendor_sdv(out_dir)

    settings = yaml.safe_load((PROJECT_ROOT / "config" / "settings.yaml").read_text(encoding="utf-8"))
    settings.setdefault("api", {}).update(PUBLIC_API_LIMITS)
    (out_dir / "config").mkdir()
    (out_dir / "config" / "settings.yaml").write_text(
        "# Generated by scripts/build_vercel_bundle.py from config/settings.yaml (public API limits)\n"
        + yaml.safe_dump(settings, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    (out_dir / "models").mkdir()
    (out_dir / "models" / "registry.json").write_text(
        json.dumps(_filtered_registry(models_dir / "registry.json"), indent=2) + "\n", encoding="utf-8"
    )
    for table_dir in sorted(p for p in models_dir.iterdir() if p.is_dir()):
        for model_dir in sorted(p for p in table_dir.iterdir() if p.is_dir()):
            if model_dir.name not in TORCH_MODELS:
                _copy_tree(model_dir, out_dir / "models" / table_dir.name / model_dir.name)

    _copy_tree(processed_dir, out_dir / "data" / "processed")
    results_out = out_dir / "reports" / "results"
    results_out.mkdir(parents=True)
    for result in sorted((PROJECT_ROOT / "reports" / "results").glob("*.json")):
        shutil.copy2(result, results_out / result.name)

    (out_dir / "api").mkdir()
    (out_dir / "api" / "index.py").write_text(ENTRYPOINT, encoding="utf-8")
    (out_dir / "requirements.txt").write_text(REQUIREMENTS, encoding="utf-8")
    (out_dir / ".python-version").write_text("3.12\n", encoding="utf-8")
    (out_dir / "vercel.json").write_text(json.dumps(VERCEL_CONFIG, indent=2) + "\n", encoding="utf-8")
    (out_dir / ".vercelignore").write_text("**/__pycache__\n**/*.pyc\n.vercel\n.env*\n", encoding="utf-8")
    return out_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "dist" / "vercel", help="output folder")
    args = parser.parse_args(argv)
    out_dir = build(args.out)
    size_mb = sum(p.stat().st_size for p in out_dir.rglob("*") if p.is_file()) / 1e6
    print(f"Vercel bundle written to {out_dir.relative_to(PROJECT_ROOT) if out_dir.is_relative_to(PROJECT_ROOT) else out_dir} ({size_mb:.1f} MB)")
    print("Deploy with: cd dist/vercel && npx vercel deploy --prod")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
