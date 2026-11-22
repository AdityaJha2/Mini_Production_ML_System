"""Phase 4 robustness and edge-case evaluation.

Runs a fixed scenario suite through the real FastAPI application and records the HTTP status,
the scorer used and warnings. Also injects scorer outages to exercise the fallback tiers.
Output: reports/robustness.json
"""
from __future__ import annotations

import json

from fastapi.testclient import TestClient

from serving import app as app_module
from serving.app import app
from src import config

BASE = {
    "transaction_id": "T-ROBUST-1", "timestamp": "2025-05-03T14:22:10", "customer_id": "C00042",
    "amount": 129.99, "merchant_category": "electronics", "channel": "online",
    "is_international": 0, "device_new": 1, "txn_count_1h": 1, "txn_count_24h": 2,
    "account_age_days": 400, "amount_mean_30d": 60.0, "seconds_since_last_txn": 5400.0,
}

SCENARIOS = [
    ("valid_record", {}, "200 / model", "baseline request"),
    ("missing_account_age", {"account_age_days": None}, "200 / model, warning", "imputed with median"),
    ("new_card_no_history", {"amount_mean_30d": None, "seconds_since_last_txn": None},
     "200 / model, warnings", "first 30 days of a card: history features missing"),
    ("unseen_merchant_category", {"merchant_category": "crypto_exchange"}, "200 / model, warning",
     "category never seen in training: all-zero encoding"),
    ("missing_merchant_category", {"merchant_category": None}, "200 / model, warning", "treated as unknown"),
    ("extreme_amount_plausible", {"amount": 750000.0}, "200 / model, warning",
     "valid but above training 99.9th percentile"),
    ("negative_amount", {"amount": -5}, "422", "impossible value"),
    ("amount_over_limit", {"amount": 2_000_000}, "422", "outside plausible domain"),
    ("malformed_timestamp", {"timestamp": "not-a-date"}, "422", "type violation"),
    ("flag_not_binary", {"is_international": 2}, "422", "range violation"),
    ("negative_velocity", {"txn_count_1h": -1}, "422", "range violation"),
    ("missing_transaction_id", {"transaction_id": None}, "422", "required field null"),
    ("unknown_extra_field", {"card_number": "4111111111111111"}, "422", "extra fields forbidden (PII guard)"),
]


def run_suite() -> dict:
    client = TestClient(app)
    results = []
    with client:
        for name, override, expected, note in SCENARIOS:
            r = client.post("/predict", json={**BASE, **override})
            entry = {"scenario": name, "expected": expected, "status_code": r.status_code, "note": note}
            if r.status_code == 200:
                j = r.json()
                entry.update({"scorer": j["scorer"], "decision": j["decision"],
                              "fraud_probability": round(j["fraud_probability"], 4), "warnings": j["warnings"]})
            else:
                entry["validation_errors"] = [str(e["loc"][-1]) for e in r.json().get("detail", [])][:5]
            results.append(entry)

        scorer = app_module.scorer
        scorer.force_model_failure = True
        r = client.post("/predict", json=BASE).json()
        results.append({"scenario": "model_outage", "expected": "200 / fallback_heuristic",
                        "status_code": 200, "scorer": r["scorer"], "decision": r["decision"],
                        "fraud_probability": round(r["fraud_probability"], 4),
                        "note": "ML scorer raises; rule engine answers"})
        scorer.force_heuristic_failure = True
        r2 = client.post("/predict", json=BASE)
        results.append({"scenario": "model_and_rule_outage", "expected": "200 / fallback_prior, flag_for_review",
                        "status_code": r2.status_code, "scorer": r2.json()["scorer"],
                        "decision": r2.json()["decision"], "note": "fail-safe: send to manual review"})
        scorer.force_model_failure = scorer.force_heuristic_failure = False

        r3 = client.post("/predict_batch", json={"records": [BASE, {**BASE, "amount": -1}]})
        results.append({"scenario": "batch_with_one_invalid_record", "expected": "422 (whole request rejected)",
                        "status_code": r3.status_code,
                        "note": "online batch is all-or-nothing; the CLI isolates bad rows instead"})

    summary = {"scenarios": results}
    config.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (config.REPORT_DIR / "robustness.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary


if __name__ == "__main__":
    for row in run_suite()["scenarios"]:
        print(row["scenario"], row["status_code"], row.get("scorer", ""), row.get("decision", ""),
              row.get("warnings", ""))
