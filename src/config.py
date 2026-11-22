"""Single source of truth: paths, temporal cut-offs, feature schema, business costs."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_PATH = DATA_DIR / "raw" / "transactions.csv"
PROCESSED_DIR = DATA_DIR / "processed"
ARTIFACT_DIR = ROOT / "artifacts"
REPORT_DIR = ROOT / "reports"
LOG_DIR = ROOT / "logs"

PIPELINE_PATH = ARTIFACT_DIR / "preprocessing_pipeline.joblib"
MODEL_PATH = ARTIFACT_DIR / "fraud_model.joblib"              # champion
CHALLENGER_PATH = ARTIFACT_DIR / "challenger_model.joblib"
BASELINE_PATH = ARTIFACT_DIR / "baseline_logistic.joblib"
METADATA_PATH = ARTIFACT_DIR / "model_metadata.json"
REFERENCE_STATS_PATH = ARTIFACT_DIR / "reference_feature_stats.json"

RANDOM_STATE = 42
START_DATE = "2025-01-01"
N_DAYS = 180
# Temporal split on day index (0 = 2025-01-01)
TRAIN_END_DAY = 119   # train: days 0-119  (2025-01-01 .. 2025-04-30)
VALID_END_DAY = 149   # validation: days 120-149 ; test: days 150-179

TARGET = "is_fraud"
ID_COL = "transaction_id"
TIME_COL = "timestamp"
CUSTOMER_COL = "customer_id"

# Columns present in the raw transaction feed (what the upstream payment system sends)
RAW_COLUMNS = [ID_COL, TIME_COL, CUSTOMER_COL, "amount", "merchant_category", "channel",
               "is_international", "device_new", "account_age_days"]
# Velocity / history features. Computed causally (past-only) at training time, and supplied by the
# upstream feature service in the online request.
VELOCITY_COLUMNS = ["txn_count_1h", "txn_count_24h", "amount_mean_30d", "seconds_since_last_txn"]
REQUIRED_REQUEST_FIELDS = [ID_COL, TIME_COL, CUSTOMER_COL, "amount", "merchant_category", "channel",
                           "is_international", "device_new", "txn_count_1h", "txn_count_24h"]
OPTIONAL_REQUEST_FIELDS = ["account_age_days", "amount_mean_30d", "seconds_since_last_txn"]

# Model input features (after derive_features)
MODEL_NUMERIC = ["amount_log", "is_night", "hour_sin", "hour_cos", "day_of_week",
                 "is_international", "device_new", "account_age_days",
                 "txn_count_1h", "txn_count_24h", "amount_ratio_log", "seconds_since_last_log"]
MODEL_CATEGORICAL = ["merchant_category", "channel"]
MODEL_FEATURES = MODEL_NUMERIC + MODEL_CATEGORICAL

CATEGORY_VALUES = {
    "merchant_category": ["grocery", "fuel", "dining", "utilities", "travel", "electronics",
                          "luxury_goods", "gift_cards", "online_gaming", "other"],
    "channel": ["online", "pos", "atm"],
}
HIGH_RISK_CATEGORIES = ["online_gaming", "gift_cards", "luxury_goods", "electronics"]

# Business cost assumptions (currency units) used for cost-sensitive threshold selection
COST_FALSE_NEGATIVE = 150.0   # average loss from a missed fraud (chargeback + fees)
COST_FALSE_POSITIVE = 15.0    # cost of step-up verification / manual review of a legitimate customer
TARGET_PRECISION = 0.50       # operating point for Recall@Precision reporting
HEURISTIC_THRESHOLD = 0.30    # cut-off for the rule-based fallback scorer
LATENCY_BUDGET_MS = 50.0      # per-request scoring budget; slower requests are logged and counted
