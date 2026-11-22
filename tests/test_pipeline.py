"""Phase 1 leakage safeguards: causal velocity features, temporal split, fit-on-train-only."""
import numpy as np
import pandas as pd

from src.config import CUSTOMER_COL, MODEL_FEATURES, TARGET, TIME_COL
from src.data_pipeline import fit_preprocessor, load_transactions, temporal_split, xy
from src.features import add_velocity_features, derive_features


def _toy():
    t = pd.to_datetime(["2025-01-01 10:00", "2025-01-01 10:30", "2025-01-01 12:00", "2025-01-03 09:00"])
    return pd.DataFrame({
        "transaction_id": ["a", "b", "c", "d"], TIME_COL: t, CUSTOMER_COL: ["X", "X", "X", "X"],
        "amount": [10.0, 30.0, 50.0, 70.0], "merchant_category": "grocery", "channel": "pos",
        "is_international": 0, "device_new": 0, "account_age_days": 100.0, TARGET: [0, 0, 0, 0],
    })


def test_velocity_counts_only_past_transactions_of_same_card():
    out = add_velocity_features(_toy()).set_index("transaction_id")
    assert out.loc["a", "txn_count_1h"] == 0               # no earlier transaction
    assert out.loc["b", "txn_count_1h"] == 1               # 'a' is 30 min earlier
    assert out.loc["c", "txn_count_1h"] == 0               # 'b' is 90 min earlier
    assert out.loc["c", "txn_count_24h"] == 2              # 'a' and 'b' are within 24 h
    assert np.isclose(out.loc["c", "amount_mean_30d"], 20.0)  # mean of 10 and 30, not including 50
    assert out.loc["d", "txn_count_24h"] == 0              # 2 days later: outside 24 h window


def test_future_rows_do_not_change_past_features():
    base = add_velocity_features(_toy()).set_index("transaction_id")
    changed = _toy()
    changed.loc[3, "amount"] = 9999.0                       # alter the FUTURE transaction 'd'
    changed.loc[2, "amount"] = 1.0                          # and 'c'
    after = add_velocity_features(changed).set_index("transaction_id")
    for tid in ["a", "b"]:
        pd.testing.assert_series_equal(base.loc[tid, ["txn_count_1h", "txn_count_24h", "amount_mean_30d",
                                                      "seconds_since_last_txn"]],
                                       after.loc[tid, ["txn_count_1h", "txn_count_24h", "amount_mean_30d",
                                                       "seconds_since_last_txn"]])


def test_temporal_split_has_no_time_overlap():
    tr, va, te = temporal_split(load_transactions())
    assert pd.to_datetime(tr[TIME_COL]).max() < pd.to_datetime(va[TIME_COL]).min()
    assert pd.to_datetime(va[TIME_COL]).max() < pd.to_datetime(te[TIME_COL]).min()
    assert len(tr) and len(va) and len(te)


def test_preprocessor_fitted_on_train_only():
    """The scaler mean for a numeric feature must equal the TRAIN mean, not the full-data mean."""
    raw = load_transactions()
    tr, va, te = temporal_split(raw)
    pre = fit_preprocessor(tr)
    scale = pre.named_transformers_["num"].named_steps["scale"]
    idx = MODEL_FEATURES.index("amount_log")
    train_mean = derive_features(tr)["amount_log"].mean()
    full_mean = derive_features(raw)["amount_log"].mean()
    assert np.isclose(scale.mean_[idx], train_mean, rtol=1e-9)
    assert not np.isclose(train_mean, full_mean, rtol=1e-9)


def test_unseen_category_encodes_all_zero():
    tr, va, _ = temporal_split(load_transactions())
    pre = fit_preprocessor(tr)
    probe = va.head(3).copy()
    probe["merchant_category"] = "crypto_exchange"
    X_probe, _ = xy(probe, pre)
    cols = [i for i, n in enumerate(pre.get_feature_names_out()) if n.startswith("merchant_category_")]
    assert np.all(X_probe[:, cols] == 0)


def test_fraud_label_is_binary_and_rare():
    raw = load_transactions()
    assert set(raw[TARGET].unique()) <= {0, 1}
    assert raw[TARGET].mean() < 0.01
