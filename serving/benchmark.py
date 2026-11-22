"""Phase 4 empirical profiling.

API mode   : N sequential HTTP requests to a running server after warm-up. Reports p50, p95, p99,
             mean, max, error count and throughput. Writes reports/benchmark_api.json.
Batch mode : vectorised model scoring at several batch sizes (median of repeats), plus an
             end-to-end CLI run on the test partition. Writes reports/benchmark_batch.json.

    uvicorn serving.app:app --port 8000        # terminal 1
    python -m serving.benchmark --mode api --requests 1000
    python -m serving.benchmark --mode batch
"""
from __future__ import annotations

import argparse
import json
import statistics
import time

import httpx
import numpy as np
import pandas as pd

from serving.inference import FraudScorer
from serving.predict_batch import run as run_batch
from src import config
from src.config import CUSTOMER_COL, ID_COL, TIME_COL
from src.data_pipeline import load_transactions, temporal_split

API_FIELDS = [ID_COL, TIME_COL, CUSTOMER_COL, "amount", "merchant_category", "channel",
              "is_international", "device_new", "txn_count_1h", "txn_count_24h",
              "account_age_days", "amount_mean_30d", "seconds_since_last_txn"]


def export_test_partition() -> pd.DataFrame:
    """Test rows with the same columns the upstream feature service would send."""
    _, _, test = temporal_split(load_transactions())
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = test[API_FIELDS].copy()
    out.to_csv(config.PROCESSED_DIR / "test_transactions.csv", index=False)
    return out


def _payloads(df: pd.DataFrame, n: int) -> list[dict]:
    sample = df.sample(n=min(n, len(df)), random_state=0)
    recs = []
    for row in sample.to_dict(orient="records"):
        clean = {}
        for k, v in row.items():
            if isinstance(v, float) and np.isnan(v):
                clean[k] = None
            elif k == TIME_COL:
                clean[k] = pd.Timestamp(v).isoformat()
            elif k in ("is_international", "device_new", "txn_count_1h", "txn_count_24h"):
                clean[k] = int(v)
            elif k == "account_age_days" and v is not None:
                clean[k] = int(v)
            else:
                clean[k] = v
        recs.append(clean)
    return recs


def bench_api(url: str, n: int, warmup: int) -> dict:
    df = export_test_partition()
    payloads = _payloads(df, n + warmup)
    lat, errors, scorers = [], 0, {}
    with httpx.Client(base_url=url, timeout=10.0) as client:
        for i, body in enumerate(payloads):
            t0 = time.perf_counter()
            ok = False
            try:
                r = client.post("/predict", json=body)
                ok = r.status_code == 200
                if ok:
                    s = r.json()["scorer"]
                    scorers[s] = scorers.get(s, 0) + 1
            except httpx.HTTPError:
                ok = False
            dt = (time.perf_counter() - t0) * 1000
            if i >= warmup:
                lat.append(dt)
                errors += 0 if ok else 1
    total = sum(lat) / 1000
    return {
        "mode": "api_sequential", "url": url, "requests": len(lat), "warmup_requests": warmup,
        "errors": errors, "error_rate": round(errors / len(lat), 6),
        "latency_ms": {"p50": round(float(np.percentile(lat, 50)), 2),
                       "p95": round(float(np.percentile(lat, 95)), 2),
                       "p99": round(float(np.percentile(lat, 99)), 2),
                       "mean": round(statistics.fmean(lat), 2), "max": round(max(lat), 2)},
        "throughput_rps": round(len(lat) / total, 1),
        "scorers_seen": scorers,
    }


def bench_batch(sizes=(1_000, 10_000, 50_000), repeats: int = 3) -> dict:
    scorer = FraudScorer().load()
    df = export_test_partition()
    rows = []
    for size in sizes:
        part = pd.concat([df] * (size // len(df) + 1), ignore_index=True).iloc[:size]
        part = part.assign(**{TIME_COL: pd.to_datetime(part[TIME_COL])})
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            scorer.score(part)
            times.append(time.perf_counter() - t0)
        med = statistics.median(times)
        rows.append({"batch_size": size, "median_seconds": round(med, 4),
                     "records_per_second": round(size / med, 1)})
    e2e = run_batch(str(config.PROCESSED_DIR / "test_transactions.csv"),
                    str(config.PROCESSED_DIR / "test_scored.csv"), scorer=scorer)
    return {"vectorised_scoring": rows, "end_to_end_cli": e2e}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["api", "batch"], required=True)
    p.add_argument("--url", default="http://127.0.0.1:8000")
    p.add_argument("--requests", type=int, default=1000)
    p.add_argument("--warmup", type=int, default=50)
    args = p.parse_args(argv)
    config.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    if args.mode == "api":
        res = bench_api(args.url, args.requests, args.warmup)
        out = config.REPORT_DIR / "benchmark_api.json"
    else:
        res = bench_batch()
        out = config.REPORT_DIR / "benchmark_batch.json"
    out.write_text(json.dumps(res, indent=2, default=str))
    print(json.dumps(res, indent=2, default=str))
    print(f"Written to {out}")


if __name__ == "__main__":
    main()
