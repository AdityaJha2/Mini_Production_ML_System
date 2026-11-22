"""API contract, fallback behaviour and batch isolation, exercised end-to-end."""
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from serving import app as app_module
from serving.app import app
from serving.inference import FraudScorer
from serving.predict_batch import run as run_batch

PAYLOAD = {
    "transaction_id": "T-TEST-1", "timestamp": "2025-05-03T23:10:00", "customer_id": "C00007",
    "amount": 480.0, "merchant_category": "luxury_goods", "channel": "online", "is_international": 1,
    "device_new": 1, "txn_count_1h": 3, "txn_count_24h": 5, "account_age_days": 30,
    "amount_mean_30d": 40.0, "seconds_since_last_txn": 120.0,
}


def test_health_reports_model_loaded():
    with TestClient(app) as client:
        r = client.get("/health")
        assert r.status_code == 200 and r.json()["model_loaded"] is True


def test_predict_returns_probability_flag_and_decision():
    with TestClient(app) as client:
        r = client.post("/predict", json=PAYLOAD)
        assert r.status_code == 200
        body = r.json()
        assert 0.0 <= body["fraud_probability"] <= 1.0
        assert body["fraud_flag"] in (0, 1)
        assert body["decision"] in ("approve", "step_up_verification", "flag_for_review")
        assert body["scorer"] == "model"


def test_predict_rejects_malformed_payload_with_422():
    with TestClient(app) as client:
        assert client.post("/predict", json={**PAYLOAD, "amount": "lots"}).status_code == 422


def test_model_outage_falls_back_to_rule_engine():
    with TestClient(app) as client:
        app_module.scorer.force_model_failure = True
        try:
            r = client.post("/predict", json=PAYLOAD)
            assert r.status_code == 200 and r.json()["scorer"] == "fallback_heuristic"
        finally:
            app_module.scorer.force_model_failure = False


def test_total_outage_fails_safe_to_manual_review():
    with TestClient(app) as client:
        app_module.scorer.force_model_failure = True
        app_module.scorer.force_heuristic_failure = True
        try:
            r = client.post("/predict", json=PAYLOAD)
            assert r.status_code == 200
            assert r.json()["scorer"] == "fallback_prior"
            assert r.json()["decision"] == "flag_for_review"
        finally:
            app_module.scorer.force_model_failure = False
            app_module.scorer.force_heuristic_failure = False


def test_batch_endpoint_counts_match():
    with TestClient(app) as client:
        recs = [{**PAYLOAD, "transaction_id": f"B{i}"} for i in range(25)]
        r = client.post("/predict_batch", json={"records": recs})
        assert r.status_code == 200 and r.json()["count"] == 25


def test_batch_cli_isolates_invalid_rows(tmp_path):
    rows = pd.DataFrame([
        {**PAYLOAD, "transaction_id": "OK1"},
        {**PAYLOAD, "transaction_id": "BAD1", "amount": -10},
        {**PAYLOAD, "transaction_id": "OK2"},
    ])
    inp, out = tmp_path / "in.csv", tmp_path / "out.csv"
    rows.to_csv(inp, index=False)
    summary = run_batch(str(inp), str(out), scorer=FraudScorer().load())
    res = pd.read_csv(out)
    assert summary["scored"] == 2 and summary["invalid"] == 1
    assert res.loc[res.transaction_id == "BAD1", "status"].item() == "invalid"
    probs = res.loc[res.status == "ok", "fraud_probability"].to_numpy()
    assert np.all((probs >= 0) & (probs <= 1))
