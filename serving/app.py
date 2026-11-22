"""Online inference path: FastAPI service.

Endpoints: GET /health, POST /predict, POST /predict_batch, GET /metrics
Run:  uvicorn serving.app:app --host 127.0.0.1 --port 8000
"""
from __future__ import annotations

import logging
import threading
import time
from collections import deque
from contextlib import asynccontextmanager

import numpy as np
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from serving.inference import FraudScorer
from serving.schemas import BatchRequest, BatchResponse, FraudResponse, HealthResponse, TransactionRequest
from src import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("api")

scorer = FraudScorer()
_lock = threading.Lock()
COUNTERS = {"requests": 0, "errors": 0, "predictions": 0, "fallbacks": 0, "slow": 0,
            "flagged": 0, "decision_approve": 0, "decision_step_up": 0, "decision_review": 0}
RECENT_MS: deque = deque(maxlen=2000)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        scorer.load()
    except Exception as exc:  # service still starts: /health reports degraded, fallbacks answer
        log.error("Artifacts could not be loaded: %s", exc)
    yield


app = FastAPI(title="Card Fraud Detection API", version="1.0.0",
              description="Real-time fraud scoring with schema validation, fallbacks and latency budget.",
              lifespan=lifespan)


def _bump(key, n=1):
    with _lock:
        COUNTERS[key] += n


@app.middleware("http")
async def count_requests(request: Request, call_next):
    _bump("requests")
    response = await call_next(request)
    if response.status_code >= 500:
        _bump("errors")
    return response


@app.get("/health", response_model=HealthResponse)
def health():
    return scorer.health()


@app.get("/metrics")
def metrics():
    """Operational counters for the monitoring plan."""
    with _lock:
        snap = dict(COUNTERS)
        recent = np.array(RECENT_MS) if RECENT_MS else np.array([0.0])
    req = max(snap["requests"], 1)
    pred = max(snap["predictions"], 1)
    return {**snap,
            "error_rate": round(snap["errors"] / req, 6),
            "fallback_rate": round(snap["fallbacks"] / pred, 6),
            "slow_rate": round(snap["slow"] / pred, 6),
            "scoring_p95_ms": round(float(np.percentile(recent, 95)), 3),
            "latency_budget_ms": config.LATENCY_BUDGET_MS}


def _score(records: list[TransactionRequest]):
    t0 = time.perf_counter()
    df = pd.DataFrame([r.model_dump() for r in records])
    proba, scorer_name, warns = scorer.score(df)
    flags = scorer.flags(proba, scorer_name)
    decisions = scorer.decisions(flags, scorer_name)
    ms = (time.perf_counter() - t0) * 1000
    thr = scorer.heuristic_threshold if scorer_name == "fallback_heuristic" else scorer.threshold
    preds = []
    for i, tid in enumerate(df["transaction_id"]):
        preds.append(FraudResponse(
            transaction_id=tid,
            fraud_probability=float(min(max(proba[i], 0.0), 1.0)),
            fraud_flag=int(flags[i]),
            decision=str(decisions[i]),
            threshold=thr,
            scorer=scorer_name,
            model_version=scorer.model_version,
            warnings=warns[i],
        ))
    with _lock:
        RECENT_MS.append(ms / len(preds))
    _bump("predictions", len(preds))
    _bump("flagged", int(flags.sum()))
    if scorer_name != "model":
        _bump("fallbacks", len(preds))
    if ms > config.LATENCY_BUDGET_MS:
        _bump("slow", len(preds))
        log.warning("scoring exceeded latency budget: %.2f ms for %d records", ms, len(preds))
    for d in decisions:
        _bump({"approve": "decision_approve", "step_up_verification": "decision_step_up",
               "flag_for_review": "decision_review"}[str(d)])
    return preds, ms


@app.post("/predict", response_model=FraudResponse)
def predict(record: TransactionRequest):
    preds, ms = _score([record])
    log.info("predict txn=%s scorer=%s decision=%s ms=%.2f", record.transaction_id,
             preds[0].scorer, preds[0].decision, ms)
    return preds[0]


@app.post("/predict_batch", response_model=BatchResponse)
def predict_batch(req: BatchRequest):
    preds, ms = _score(req.records)
    log.info("predict_batch n=%d ms=%.2f", len(preds), ms)
    return BatchResponse(count=len(preds), predictions=preds, latency_ms=round(ms, 3))


@app.exception_handler(Exception)
async def unhandled(request: Request, exc: Exception):
    """The API never returns an HTML stack trace."""
    log.exception("Unhandled error on %s", request.url.path)
    return JSONResponse(status_code=503, content={
        "error": "scoring_unavailable",
        "detail": "The fraud service could not process this request. Retry shortly."})
