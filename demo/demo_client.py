"""Walk through the BedSynth AI REST API with httpx.

Start the server first, then run the walkthrough from another terminal::

    python scripts/04_serve_api.py
    python demo/demo_client.py
    python demo/demo_client.py --base-url http://127.0.0.1:8000 --rows 300 --out-dir demo/output --keep

The walkthrough checks the service health, browses the table/scenario/model catalogue,
generates high-occupancy ward-days (JSON, with a quality score) and delayed-discharge stays
(CSV), validates records (including a deliberately corrupted one), evaluates a stored dataset,
lists and downloads the persisted datasets and finally deletes what it created (unless
``--keep``). It only speaks HTTP, so it works against any running BedSynth AI server.
"""
from __future__ import annotations

import argparse
import functools
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import httpx

DEFAULT_BASE_URL = "http://127.0.0.1:8000"
API = "/api/v1"
Echo = Callable[..., None]


class WalkthroughError(RuntimeError):
    """An API call returned an error; the message carries the API's error type and detail."""


def _ensure_ok(response: httpx.Response) -> httpx.Response:
    if response.status_code >= 400:
        try:
            body = response.json()
            message = f"{body.get('error_type')}: {body.get('detail')}"
        except ValueError:
            message = response.text[:300]
        raise WalkthroughError(f"{response.request.method} {response.request.url.path} -> {response.status_code} {message}")
    return response


def _get(client: httpx.Client, path: str, **params: Any) -> Any:
    return _ensure_ok(client.get(path, params=params or None)).json()


def _post(client: httpx.Client, path: str, payload: dict[str, Any]) -> Any:
    return _ensure_ok(client.post(path, json=payload)).json()


def _section(echo: Echo, title: str) -> None:
    echo(f"\n== {title} " + "=" * max(0, 66 - len(title)))


def _describe_generation(echo: Echo, body: dict[str, Any]) -> None:
    generation = body["generation"]
    validation = body.get("validation") or {}
    quality = body.get("quality") or {}
    rate = generation.get("acceptance_rate")
    rate_text = f"{rate:.1%}" if isinstance(rate, (int, float)) else "n/a"
    echo(f"dataset {body['dataset_id']}  rows={body['n_rows']}  model={body['model']['id']}  seed={generation.get('seed')}")
    echo(f"acceptance rate={rate_text}  generation time={generation.get('elapsed_seconds')} s")
    if validation:
        echo(f"hard rules passed: {validation['all_hard_rules_passed']}  "
             f"(soft-rule warnings: {validation['soft_rules']['failed']}/{validation['soft_rules']['checked']})")
    if quality:
        echo(f"quality score {quality['score']} vs {quality['reference']['subset']} reference "
             f"({quality['reference']['rows_compared']} real rows)")
    for warning in body.get("warnings", []):
        echo(f"warning: {warning}")


def run_walkthrough(
    client: httpx.Client,
    rows: int = 200,
    seed: int = 42,
    out_dir: Path | None = None,
    keep: bool = False,
    echo: Echo = print,
) -> dict[str, Any]:
    """Run the walkthrough against ``client`` (whose base URL points at the API)."""
    created: list[str] = []
    result: dict[str, Any] = {"ok": False, "datasets": created}

    _section(echo, "Service")
    health = _get(client, "/health")
    echo(f"status={health['status']}  version={health['version']}  sdv={health['sdv_version']}")
    for message in health.get("messages", []):
        echo(f"note: {message}")
    trained = health["trained_tables"]
    if not trained:
        raise WalkthroughError("No trained models yet: run python scripts/02_train_synthesizers.py (the running API picks them up).")

    _section(echo, "Catalogue")
    for table in _get(client, f"{API}/tables")["tables"]:
        echo(f"{table['name']:<14} {table['n_training_columns']} learned + {table['n_derived_columns']} derived columns, "
             f"{table['n_scenarios']} scenarios, default model: {table['default_model']}")
    catalogue = _get(client, f"{API}/scenarios")
    by_category = Counter(scenario["category"] for scenario in catalogue["scenarios"])
    echo(f"{catalogue['count']} scenarios: " + ", ".join(f"{name} ({count})" for name, count in sorted(by_category.items())))
    models = _get(client, f"{API}/models")["tables"]
    for table, info in models.items():
        echo(f"{table:<14} models: {', '.join(m['model_id'] for m in info['models']) or 'none'} (default {info['default_model']})")

    first_json: dict[str, Any] | None = None
    if "ward_daily" in trained:
        _section(echo, "Generate: high-occupancy ward-days (JSON + quality)")
        body = _post(client, f"{API}/generate", {
            "table": "ward_daily", "num_rows": rows, "scenario": "high_occupancy", "seed": seed, "evaluate": True,
        })
        _describe_generation(echo, body)
        levels = Counter(record["occupancy_level"] for record in body["records"])
        echo("occupancy levels: " + ", ".join(f"{level}={count}" for level, count in sorted(levels.items())))
        if body["persisted"]:
            created.append(body["dataset_id"])
        first_json = body

    if "patient_stays" in trained:
        _section(echo, "Generate: delayed-discharge stays (CSV)")
        response = _ensure_ok(client.post(f"{API}/generate", json={
            "table": "patient_stays", "num_rows": rows, "scenario": "delayed_discharge_stays", "seed": seed, "format": "csv",
        }))
        dataset_id = response.headers["X-BedSynth-Dataset-Id"]
        echo(f"dataset {dataset_id}  rows={response.headers['X-BedSynth-Rows']}  model={response.headers['X-BedSynth-Model']}  "
             f"scenario={response.headers['X-BedSynth-Scenario']}")
        if response.headers.get("X-BedSynth-Persisted") == "true":
            created.append(dataset_id)
        if out_dir is not None:
            out_dir.mkdir(parents=True, exist_ok=True)
            target = out_dir / f"{dataset_id}.csv"
            target.write_text(response.text, encoding="utf-8")
            echo(f"saved {target}")

    if first_json is not None and first_json["records"]:
        _section(echo, "Validate client records")
        records = [dict(record) for record in first_json["records"][:5]]
        broken = dict(records[0], ward_id=records[0]["ward_id"].lower(), elective_cancelled=records[0]["elective_scheduled"] + 2)
        report = _post(client, f"{API}/validate", {"table": "ward_daily", "records": [*records, broken]})
        echo(f"{report['n_rows']} records, invalid rows: {report['invalid_rows']} "
             f"(aliases normalised: {report['normalisation']['values_normalised']})")
        for rule in report["validation"]["rules"]:
            if not rule["passed"]:
                echo(f"  {rule['rule_id']} {rule['name']}: {rule['violations']} violation(s) - {rule['description']}")

    if created:
        _section(echo, "Evaluate a stored dataset")
        try:
            evaluation = _post(client, f"{API}/evaluate", {"dataset_id": created[0]})
            echo(f"{created[0]}: quality {evaluation['quality']['score']} "
                 f"({evaluation['quality']['reference']['subset']} reference subset)")
        except WalkthroughError as exc:  # e.g. reference data not prepared on the server
            echo(f"evaluation unavailable: {exc}")

    _section(echo, "Persisted datasets")
    listing = _get(client, f"{API}/datasets", limit=10)
    echo(f"{listing['total']} stored dataset(s); newest:")
    for item in listing["datasets"][:5]:
        echo(f"  {item['dataset_id']}  {item['table']:<14} {item['scenario'] or '(no scenario)':<26} rows={item['n_rows']}")
    if created:
        download = _get(client, f"{API}/datasets/{created[0]}/download", format="json")
        echo(f"downloaded {created[0]} as JSON: {download['n_rows']} records, {len(download['columns'])} columns")

    if not keep:
        _section(echo, "Clean up")
        for dataset_id in created:
            _ensure_ok(client.delete(f"{API}/datasets/{dataset_id}"))
            echo(f"deleted {dataset_id}")

    result["ok"] = True
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Walk through the BedSynth AI REST API.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help=f"API base URL (default: {DEFAULT_BASE_URL}).")
    parser.add_argument("--rows", type=int, default=200, help="Rows per generation (default: 200).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    parser.add_argument("--out-dir", type=Path, default=None, help="Save the generated CSV here.")
    parser.add_argument("--keep", action="store_true", help="Keep the datasets created by the walkthrough.")
    parser.add_argument("--timeout", type=float, default=300.0, help="Request timeout in seconds (default: 300).")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    echo = functools.partial(print, flush=True)  # keep progress lines ordered with error messages (stderr)
    try:
        with httpx.Client(base_url=args.base_url, timeout=args.timeout) as client:
            run_walkthrough(client, rows=args.rows, seed=args.seed, out_dir=args.out_dir, keep=args.keep, echo=echo)
    except httpx.ConnectError:
        print(f"Cannot reach the BedSynth AI API at {args.base_url}. Start it with: python scripts/04_serve_api.py",
              file=sys.stderr)
        return 2
    except WalkthroughError as exc:
        print(f"Walkthrough stopped: {exc}", file=sys.stderr)
        return 1
    print("\nWalkthrough complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
