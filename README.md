# Card Fraud Detection: Mini Production ML System (Option B)

**Student:** Aditya Jha  |  **Roll no.:** 2025em1200044  |  **Course:** ML Model Engineering, WILP

An end-to-end system for real-time card-transaction fraud detection: leak-free temporal
preprocessing, cost-sensitive class-weighted models, serialized artifacts, a validated FastAPI
online service, a batch CLI, latency and throughput profiling, robustness tests, PSI/KS drift
monitoring and a four-tier fallback.

Repository: `https://github.com/AdityaJha2/Mini_Production_ML_System`

## 1. Quickstart (Python 3.12)
```bash
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
make data        # generate the dataset        -> data/raw/transactions.csv
make train       # fit models, select champion -> artifacts/
make evaluate    # test metrics, plots, drift  -> reports/
make test        # 24 unit and integration tests
make robustness  # edge-case suite             -> reports/robustness.json
```
Or run everything: `make all`.

## 2. Online API
```bash
uvicorn serving.app:app --host 127.0.0.1 --port 8000
curl -s http://127.0.0.1:8000/health
curl -s -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d @examples/one_request.json
curl -s -X POST http://127.0.0.1:8000/predict_batch -H "Content-Type: application/json" -d @examples/batch_request.json
curl -s http://127.0.0.1:8000/metrics
```
Interactive docs: http://127.0.0.1:8000/docs

The request carries the transaction plus history features that the upstream feature service
computes (`txn_count_1h`, `txn_count_24h`, `amount_mean_30d`, `seconds_since_last_txn`). Malformed
requests return HTTP 422 naming the field. Scoring failures return HTTP 200 from a fallback tier,
and the response says which tier answered.

Response fields: `fraud_probability`, `fraud_flag` (0/1), `decision`
(`approve`, `step_up_verification`, `flag_for_review`), `threshold`, `scorer`, `model_version`,
`warnings`.

## 3. Batch CLI
```bash
python -m serving.predict_batch --input data/processed/test_transactions.csv \
    --output outputs/scored_transactions.csv --log-file logs/batch.log
```
Output columns: `transaction_id, fraud_probability, fraud_flag, decision, scorer, status, error`.
Invalid rows are kept, with `status=invalid` and the reason. Each run logs its status and
records/second.

## 4. Benchmarks
```bash
uvicorn serving.app:app --port 8000 &          # start the server first
python -m serving.benchmark --mode api --requests 1000   # p50 / p95 / p99 -> reports/benchmark_api.json
python -m serving.benchmark --mode batch                 # records/sec      -> reports/benchmark_batch.json
python -m serving.benchmark --mode api --requests 2000 --concurrency 8   # load test -> reports/benchmark_api_c8.json
```

## 5. Layout
```
fraud-detection-ml-system/
├── README.md, requirements.txt, Makefile, Dockerfile, pytest.ini, .gitignore
├── data/            raw and processed data (generated; raw CSV is git-ignored)
├── artifacts/       serialized pipeline, champion, challenger, baseline, metadata, drift reference
├── examples/        sample request payloads for curl
├── docs/            ARCHITECTURE.md (decisions, data flow), GITHUB_SETUP.md
├── src/
│   ├── config.py           paths, split days, feature schema, business costs
│   ├── generate_data.py    reproducible transaction generator (compromised-card bursts)
│   ├── features.py         causal velocity features and row-level features
│   ├── data_pipeline.py    ingestion, validation, temporal split, leak-free preprocessing
│   ├── baselines.py        prior and heuristic rule (also the fallback scorer)
│   ├── train.py            class-weighted candidates, champion selection, cost threshold, export
│   ├── evaluate.py         metrics, cost-based threshold, PR curves, confusion matrix, test run
│   └── monitoring.py       PSI, KS, alert policy, system alert rules
├── serving/
│   ├── schemas.py          Pydantic contracts + vectorised batch validator (shared bounds)
│   ├── inference.py        scorer with model / rule / prior tiers and data warnings
│   ├── app.py              FastAPI: /health /predict /predict_batch /metrics
│   ├── predict_batch.py    batch scoring CLI
│   ├── benchmark.py        latency percentiles and throughput
│   └── robustness.py       edge cases and failure injection
├── tests/                  schema, leakage/causality, inference and fallback tests
└── report/                 builds the consolidated PDF from the outputs
```

## 6. Design summary
| Requirement (handout) | Where |
|---|---|
| Business objective, inputs, target, serving paradigm | Report §1; `src/config.py` |
| Temporal split, leakage safeguards | `src/data_pipeline.py`, `tests/test_pipeline.py` |
| Interpretable baseline | `src/baselines.py` (prior, rule); logistic regression |
| Imbalance handling without oversampling | class weights in `src/train.py`; cost threshold in `src/evaluate.py` |
| Metrics justified for fraud | Report §2; `src/evaluate.py` (PR-AUC, Recall@Precision, cost, F1-macro) |
| Serialized preprocessing and models | `artifacts/` (joblib + JSON metadata) |
| REST API with /health, /predict, schema validation | `serving/app.py`, `serving/schemas.py` |
| Batch CLI, vectorised, logged, record IDs | `serving/predict_batch.py` |
| Latency p50/p95/p99, throughput | `serving/benchmark.py`, `reports/benchmark_*.json` |
| Robustness | `serving/robustness.py`, `reports/robustness.json` |
| Drift monitoring and alert thresholds | `src/monitoring.py`, `reports/drift_monthly.csv` |
| Fallback under degradation | `serving/inference.py` (model, rule, fail-safe review) |

## 7. Data note
The dataset is generated by `src/generate_data.py` (about 300k transactions, 180 days, about
0.6% fraud). It includes compromised-card bursts, missing values and a deterministic seed.
To use a public dataset (Kaggle, IEEE-CIS or similar), place a CSV with the same columns at
`data/raw/transactions.csv`. Then compute the history features as described in `src/features.py`.

## 8. Docker (optional)
```bash
docker build -t fraud-api .
docker run -p 8000:8000 fraud-api
```
