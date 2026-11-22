"""Validation rules: the online (Pydantic) and batch (vectorised) paths must agree."""
import pandas as pd
import pytest
from pydantic import ValidationError

from serving.schemas import TransactionRequest, validate_frame

VALID = {
    "transaction_id": "T1", "timestamp": "2025-05-03T14:22:10", "customer_id": "C1", "amount": 42.5,
    "merchant_category": "grocery", "channel": "pos", "is_international": 0, "device_new": 0,
    "txn_count_1h": 0, "txn_count_24h": 1, "account_age_days": None, "amount_mean_30d": None,
    "seconds_since_last_txn": None,
}


def test_valid_request_accepted_with_optional_history_missing():
    req = TransactionRequest(**VALID)
    assert req.account_age_days is None and req.amount > 0


@pytest.mark.parametrize("field,value", [
    ("amount", -5), ("amount", 2_000_000), ("timestamp", "not-a-date"), ("is_international", 2),
    ("txn_count_1h", -1), ("transaction_id", None), ("customer_id", ""),
])
def test_invalid_values_rejected(field, value):
    with pytest.raises(ValidationError):
        TransactionRequest(**{**VALID, field: value})


def test_extra_field_rejected_pii_guard():
    with pytest.raises(ValidationError):
        TransactionRequest(**{**VALID, "card_number": "4111111111111111"})


def test_batch_validator_agrees_with_online_rules():
    df = pd.DataFrame([VALID,
                       {**VALID, "transaction_id": "T2", "amount": -1},
                       {**VALID, "transaction_id": "T3", "timestamp": "bad"}])
    valid, errors = validate_frame(df)
    assert valid.tolist() == [True, False, False]
    assert "amount outside" in errors.iloc[1] or "amount must be positive" in errors.iloc[1]
    assert "not a valid datetime" in errors.iloc[2]


def test_batch_validator_flags_missing_required_column():
    df = pd.DataFrame([{k: v for k, v in VALID.items() if k != "channel"}])
    valid, errors = validate_frame(df)
    assert not valid.iloc[0] and "channel column missing" in errors.iloc[0]
