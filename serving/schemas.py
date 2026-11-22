"""Input and output contracts for the online API and the batch CLI.

One BOUNDS table is the source of truth for both paths:
* Pydantic models (online)  -> malformed payloads are rejected with HTTP 422 naming the field.
* validate_frame (batch)    -> invalid rows are marked with a reason; the rest of the file is scored.

Policy
------
REJECT : wrong type, missing required field, unknown extra field, impossible value
         (negative amount, out-of-range flags, malformed timestamp).
ACCEPT : missing optional history fields (imputed, flagged), merchant categories never seen in
         training (encoded as all-zero, flagged), amounts above the training 99.9th percentile
         (scored, flagged).
"""
from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from src.config import CUSTOMER_COL, ID_COL, TIME_COL

BOUNDS = {
    "amount": (0.01, 1_000_000.0),
    "txn_count_1h": (0, 10_000),
    "txn_count_24h": (0, 100_000),
    "account_age_days": (0, 40_000),
    "amount_mean_30d": (0.0, 1_000_000.0),
    "seconds_since_last_txn": (0.0, 31_536_000.0),
    "is_international": (0, 1),
    "device_new": (0, 1),
}
INT_FIELDS = {"txn_count_1h", "txn_count_24h", "account_age_days", "is_international", "device_new"}
BATCH_REQUIRED = [ID_COL, TIME_COL, CUSTOMER_COL, "amount", "channel", "is_international",
                  "device_new", "txn_count_1h", "txn_count_24h"]


class TransactionRequest(BaseModel):
    """One card transaction plus the history features supplied by the upstream feature service."""
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    transaction_id: str = Field(..., min_length=1, max_length=64, examples=["T00012345"])
    timestamp: datetime = Field(..., examples=["2025-05-03T14:22:10"])
    customer_id: str = Field(..., min_length=1, max_length=64, examples=["C00042"])
    amount: float = Field(..., gt=0, le=1_000_000, examples=[129.99])
    merchant_category: Optional[str] = Field(None, max_length=40, examples=["electronics"])
    channel: str = Field(..., min_length=1, max_length=20, examples=["online"])
    is_international: int = Field(..., ge=0, le=1)
    device_new: int = Field(..., ge=0, le=1)
    txn_count_1h: int = Field(..., ge=0, le=10_000)
    txn_count_24h: int = Field(..., ge=0, le=100_000)
    account_age_days: Optional[int] = Field(None, ge=0, le=40_000)
    amount_mean_30d: Optional[float] = Field(None, ge=0, le=1_000_000)
    seconds_since_last_txn: Optional[float] = Field(None, ge=0, le=31_536_000)


class FraudResponse(BaseModel):
    model_config = {"protected_namespaces": ()}
    transaction_id: str
    fraud_probability: float = Field(..., ge=0.0, le=1.0)
    fraud_flag: Literal[0, 1]
    decision: Literal["approve", "step_up_verification", "flag_for_review"]
    threshold: float
    scorer: Literal["model", "fallback_heuristic", "fallback_prior"]
    model_version: str
    warnings: list[str] = []


class BatchRequest(BaseModel):
    records: list[TransactionRequest] = Field(..., min_length=1, max_length=1000)


class BatchResponse(BaseModel):
    count: int
    predictions: list[FraudResponse]
    latency_ms: float


class HealthResponse(BaseModel):
    model_config = {"protected_namespaces": ()}
    status: Literal["ok", "degraded"]
    model_loaded: bool
    fallback_available: bool
    model_version: str


def validate_frame(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Vectorised equivalent of TransactionRequest. Returns (valid_mask, error_messages)."""
    n = len(df)
    errors = pd.Series([[] for _ in range(n)], index=df.index, dtype=object)

    def flag(mask, msg):
        for i in df.index[mask]:
            errors.at[i].append(msg)

    for col in BATCH_REQUIRED:
        if col not in df.columns:
            flag(pd.Series(True, index=df.index), f"{col} column missing")
        else:
            flag(df[col].isna(), f"{col} is required")

    if ID_COL in df.columns:
        flag(df[ID_COL].notna() & (df[ID_COL].astype(str).str.len() > 64), f"{ID_COL} longer than 64 characters")
    if TIME_COL in df.columns:
        ts = pd.to_datetime(df[TIME_COL], errors="coerce")
        flag(df[TIME_COL].notna() & ts.isna(), f"{TIME_COL} is not a valid datetime")
    for col, (lo, hi) in BOUNDS.items():
        if col not in df.columns:
            continue
        raw = df[col]
        num = pd.to_numeric(raw, errors="coerce")
        flag(raw.notna() & num.isna(), f"{col} is not numeric")
        if col == "amount":
            flag(num.notna() & (num <= 0), "amount must be positive")
        flag(num.notna() & ((num < lo) | (num > hi)), f"{col} outside [{lo}, {hi}]")
        if col in INT_FIELDS:
            flag(num.notna() & (num % 1 != 0), f"{col} must be an integer")
    valid = errors.apply(len) == 0
    return valid, errors.apply(lambda e: "; ".join(e))
