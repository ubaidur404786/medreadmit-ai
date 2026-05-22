"""Latency benchmark for the MedReadmit prediction API.

The API must already be running before this script is invoked:

    uvicorn src.api.main:app --host 127.0.0.1 --port 8000

Then run:

    python scripts/benchmark_api.py [--base-url http://127.0.0.1:8000]

Results are printed as a Markdown table and written to reports/latency_baseline.md.
"""

from __future__ import annotations

import argparse
import math
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

# ---------------------------------------------------------------------------
# Path setup — allow running from repo root without installing the package
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from src.data.load import load_raw  # noqa: E402
from src.data.make_target import build_target  # noqa: E402
from src.data.split import patient_grouped_split  # noqa: E402
from src.features.build_features import build_features  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_N_RECORDS = 200
_N_WARMUP = 10
_BATCH_SIZES = [1, 10, 50, 100]
_NON_SCHEMA_COLS = {"patient_nbr", "readmitted_30d", "encounter_id", "weight", "payer_code"}
_REPORTS_DIR = _REPO_ROOT / "reports"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_json_safe(v: Any) -> Any:
    """float NaN → None, numpy scalar → Python native, everything else unchanged."""
    try:
        import pandas as pd

        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(v, "item"):
        return v.item()
    return v


# Required fields in PatientEncounter — records missing any of these are skipped.
_REQUIRED_FIELDS = {
    "race",
    "gender",
    "age",
    "admission_type_id",
    "discharge_disposition_id",
    "admission_source_id",
    "time_in_hospital",
    "num_lab_procedures",
    "num_medications",
    "number_diagnoses",
}


def load_test_records(n: int) -> list[dict[str, Any]]:
    """Return the first *n* valid rows from the held-out test split as JSON-safe dicts.

    Rows where any required PatientEncounter field is null are skipped so that
    every returned record passes Pydantic validation without modification.
    Mirrors the sample_record fixture pattern in tests/test_api.py.
    """
    print(f"Loading pipeline data to build {n} test records …", flush=True)
    labelled = build_target(load_raw()).reset_index(drop=True)
    X, y, groups = build_features(labelled)
    _, _, X_test, _, _, _ = patient_grouped_split(X, y, groups)

    records: list[dict[str, Any]] = []
    scanned = 0
    for pos in range(len(X_test)):
        test_idx = X_test.index[pos]
        raw_row = labelled.loc[test_idx].to_dict()
        record = {k: _to_json_safe(v) for k, v in raw_row.items() if k not in _NON_SCHEMA_COLS}

        # Skip rows where a required field is null — they would fail Pydantic validation.
        if any(record.get(f) is None for f in _REQUIRED_FIELDS):
            scanned += 1
            continue

        records.append(record)
        scanned += 1
        if len(records) == n:
            break

    if len(records) < n:
        sys.exit(
            f"ERROR: Only {len(records)} valid records found in test split "
            f"(scanned {scanned} rows); need {n}."
        )

    print(f"  Loaded {len(records)} records (scanned {scanned} test-split rows).", flush=True)
    return records


def check_health(client: httpx.Client, base_url: str) -> None:
    """GET /health — abort with a clear message if the API is not reachable or not ok."""
    try:
        resp = client.get(f"{base_url}/health", timeout=5.0)
    except httpx.ConnectError:
        sys.exit(
            f"\nERROR: Cannot connect to {base_url}.\n"
            "Start the API first:\n"
            "  uvicorn src.api.main:app --host 127.0.0.1 --port 8000\n"
        )

    if resp.status_code != 200:
        sys.exit(f"\nERROR: /health returned {resp.status_code}:\n{resp.text}\n")

    body = resp.json()
    if body.get("status") != "ok":
        sys.exit(f"\nERROR: API status is {body.get('status')!r}, not 'ok'.\n{body}\n")

    print(
        f"  API healthy — model_version={body['model_version']}  "
        f"n_features={body['n_features']}",
        flush=True,
    )


def _fmt(ms: float) -> str:
    return f"{ms:.2f}"


# ---------------------------------------------------------------------------
# Benchmark routines
# ---------------------------------------------------------------------------


def benchmark_single(
    client: httpx.Client,
    base_url: str,
    records: list[dict[str, Any]],
) -> dict[str, float]:
    """Run N warmup + N timed sequential /predict calls; return latency stats."""
    url = f"{base_url}/predict"

    print(f"\nWarmup ({_N_WARMUP} requests) …", flush=True)
    for rec in records[:_N_WARMUP]:
        resp = client.post(url, json=rec, timeout=30.0)
        if resp.status_code != 200:
            sys.exit(f"Warmup request failed ({resp.status_code}): {resp.text}")
    print("  Done.", flush=True)

    print(f"Benchmarking /predict ({_N_RECORDS} requests) …", flush=True)
    latencies: list[float] = []
    for i, rec in enumerate(records):
        t0 = time.perf_counter()
        resp = client.post(url, json=rec, timeout=30.0)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        if resp.status_code != 200:
            sys.exit(f"Request {i} failed ({resp.status_code}): {resp.text}")
        latencies.append(elapsed_ms)

        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{_N_RECORDS} done …", flush=True)

    sorted_lat = sorted(latencies)
    n = len(sorted_lat)

    def percentile(p: float) -> float:
        idx = (p / 100) * (n - 1)
        lo, hi = int(idx), min(int(idx) + 1, n - 1)
        return sorted_lat[lo] + (idx - lo) * (sorted_lat[hi] - sorted_lat[lo])

    return {
        "p50": percentile(50),
        "p95": percentile(95),
        "p99": percentile(99),
        "mean": statistics.mean(latencies),
        "stdev": statistics.stdev(latencies),
    }


def benchmark_batch(
    client: httpx.Client,
    base_url: str,
    records: list[dict[str, Any]],
    batch_sizes: list[int],
) -> list[dict[str, float]]:
    """For each batch size, send one /predict/batch request and report latency."""
    url = f"{base_url}/predict/batch"
    results: list[dict[str, float]] = []

    print(f"\nBenchmarking /predict/batch (sizes {batch_sizes}) …", flush=True)
    for size in batch_sizes:
        batch_records = records[:size]

        t0 = time.perf_counter()
        resp = client.post(url, json={"encounters": batch_records}, timeout=60.0)
        total_ms = (time.perf_counter() - t0) * 1000

        if resp.status_code != 200:
            sys.exit(f"Batch size {size} failed ({resp.status_code}): {resp.text}")

        body = resp.json()
        # Use server-reported mean_latency_per_record_ms as the authoritative value;
        # fall back to client-side measurement if the field is absent.
        server_mean = body.get("mean_latency_per_record_ms", total_ms / size)

        results.append(
            {
                "batch_size": float(size),
                "mean_per_record_ms": server_mean,
                "total_ms": total_ms,
            }
        )
        print(
            f"  size={size:>3}  total={total_ms:6.1f} ms  "
            f"per-record={server_mean:6.2f} ms",
            flush=True,
        )

    return results


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------


def build_report(
    single_stats: dict[str, float],
    batch_results: list[dict[str, float]],
    timestamp: str,
) -> str:
    lines: list[str] = []

    lines.append(f"# API Latency Baseline — {timestamp}")
    lines.append("")
    lines.append(
        f"Hardware: {platform.platform()}, "
        f"Python {sys.version.split()[0]}"
    )
    lines.append("")

    lines.append("## Single prediction (/predict)")
    lines.append("| Metric | Value (ms) |")
    lines.append("|---|---|")
    for metric in ("p50", "p95", "p99", "mean", "stdev"):
        lines.append(f"| {metric} | {_fmt(single_stats[metric])} |")
    lines.append("")

    lines.append("## Batch prediction (/predict/batch)")
    lines.append("| Batch size | Mean per-record (ms) | Total batch (ms) |")
    lines.append("|---|---|---|")
    for r in batch_results:
        lines.append(
            f"| {int(r['batch_size'])} "
            f"| {_fmt(r['mean_per_record_ms'])} "
            f"| {_fmt(r['total_ms'])} |"
        )
    lines.append("")

    lines.append("## Notes")
    lines.append(f"- N={_N_RECORDS} single requests, {_N_WARMUP} warmup")
    lines.append("- Sequential, single client, localhost")
    lines.append("- Includes SHAP top-5 explanation")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000",
        help="Base URL of the running API (default: http://127.0.0.1:8000)",
    )
    args = parser.parse_args()
    base_url = args.base_url.rstrip("/")

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    print(f"\n=== MedReadmit API Latency Benchmark ===")
    print(f"Target: {base_url}")
    print(f"Time:   {timestamp}\n")

    records = load_test_records(_N_RECORDS)

    with httpx.Client() as client:
        print("Checking API health …", flush=True)
        check_health(client, base_url)

        single_stats = benchmark_single(client, base_url, records)
        batch_results = benchmark_batch(client, base_url, records, _BATCH_SIZES)

    report = build_report(single_stats, batch_results, timestamp)

    print("\n" + "=" * 60)
    print(report)
    print("=" * 60)

    _REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _REPORTS_DIR / "latency_baseline.md"
    out_path.write_text(report, encoding="utf-8")
    print(f"\nReport written to {out_path}")


if __name__ == "__main__":
    main()
