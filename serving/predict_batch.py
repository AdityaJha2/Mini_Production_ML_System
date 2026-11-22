"""Batch scoring CLI: validated, vectorised, chunked, with a logged run status.

    python -m serving.predict_batch --input data/processed/test_transactions.csv \
        --output outputs/scored_transactions.csv --log-file logs/batch.log

Invalid rows are never dropped silently: they appear in the output with status=invalid and the
reason. Valid rows are scored in one vectorised call per chunk. Each run logs its status, row
counts and throughput, to stdout and to the log file.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd

from serving.inference import FraudScorer
from serving.schemas import validate_frame
from src.config import ID_COL

log = logging.getLogger("predict_batch")


def run(input_path: str, output_path: str, chunk_size: int = 50_000, scorer: FraudScorer | None = None) -> dict:
    t0 = time.perf_counter()
    scorer = scorer or FraudScorer().load()
    log.info("Reading %s", input_path)
    df = pd.read_csv(input_path)
    if ID_COL not in df.columns:
        raise SystemExit(f"Input must contain a '{ID_COL}' column")
    n_total = len(df)
    parts = []
    for start in range(0, n_total, chunk_size):
        chunk = df.iloc[start:start + chunk_size].copy()
        valid, errs = validate_frame(chunk)
        out = pd.DataFrame({
            ID_COL: chunk[ID_COL].values,
            "fraud_probability": pd.NA, "fraud_flag": pd.NA, "decision": "",
            "scorer": "", "status": "invalid", "error": errs.values,
        }, index=chunk.index)
        if valid.any():
            good = chunk[valid]
            proba, scorer_name, warns = scorer.score(good)
            flags = scorer.flags(proba, scorer_name)
            out.loc[valid, "fraud_probability"] = proba.round(6)
            out.loc[valid, "fraud_flag"] = flags
            out.loc[valid, "decision"] = scorer.decisions(flags, scorer_name)
            out.loc[valid, "scorer"] = scorer_name
            out.loc[valid, "status"] = "ok"
            out.loc[valid, "error"] = ["; ".join(w) for w in warns]
        parts.append(out)
        log.info("rows %d-%d: valid=%d invalid=%d", start, start + len(chunk) - 1,
                 int(valid.sum()), int((~valid).sum()))
    result = pd.concat(parts, ignore_index=True)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    elapsed = time.perf_counter() - t0
    summary = {
        "status": "completed", "input_rows": n_total,
        "scored": int((result["status"] == "ok").sum()),
        "invalid": int((result["status"] == "invalid").sum()),
        "flagged": int(pd.to_numeric(result["fraud_flag"], errors="coerce").fillna(0).sum()),
        "elapsed_seconds": round(elapsed, 3),
        "records_per_second": round(n_total / max(elapsed, 1e-9), 1),
        "output": output_path,
    }
    log.info("Batch summary: %s", summary)
    return summary


def main(argv=None):
    p = argparse.ArgumentParser(description="Batch card-fraud scoring")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--log-file", default="logs/batch.log")
    p.add_argument("--chunk-size", type=int, default=50_000)
    args = p.parse_args(argv)
    Path(args.log_file).parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(args.log_file)])
    try:
        run(args.input, args.output, args.chunk_size)
        return 0
    except Exception as exc:
        log.exception("Batch run failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
