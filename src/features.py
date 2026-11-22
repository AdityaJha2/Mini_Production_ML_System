"""Feature engineering shared by training, the API and the batch CLI.

Two kinds of features
---------------------
1. Velocity / history features (txn_count_1h, txn_count_24h, amount_mean_30d,
   seconds_since_last_txn). At training time they are computed CAUSALLY: each row only sees
   earlier transactions of the same card, never later ones (see add_velocity_features and
   tests/test_pipeline.py). At serving time the upstream feature service supplies them.
2. Row-level features (derive_features): hour of day as sin/cos, night flag, log amount,
   amount relative to the card's 30-day mean, log time since the previous transaction.
   These are deterministic functions of one row, so they are identical in training and serving.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import CUSTOMER_COL, TIME_COL

HOUR = 3600
DAY = 86400


def _col(df: pd.DataFrame, name: str) -> pd.Series:
    """Return a column, or an all-NaN column when the caller did not send it."""
    if name in df.columns:
        return pd.to_numeric(df[name], errors="coerce")
    return pd.Series(np.nan, index=df.index, dtype=float)


def add_velocity_features(df: pd.DataFrame) -> pd.DataFrame:
    """Causal per-card history features. Row i uses only rows j < i of the same card."""
    out = df.copy()
    ts_all = pd.to_datetime(out[TIME_COL])
    order = np.lexsort((ts_all.values.astype("datetime64[ns]").astype(np.int64),
                        out[CUSTOMER_COL].astype(str).values))
    work = out.iloc[order]
    res = {k: np.full(len(out), np.nan) for k in
           ["txn_count_1h", "txn_count_24h", "amount_mean_30d", "seconds_since_last_txn"]}
    pos_of = {idx: p for p, idx in enumerate(out.index)}
    groups = work.groupby(CUSTOMER_COL, sort=False).indices
    for _, local in groups.items():
        rows = work.index[local]
        t = pd.to_datetime(work[TIME_COL].iloc[local]).values.astype("datetime64[s]").astype(np.int64)
        a = work["amount"].iloc[local].to_numpy(dtype=float)
        i = np.arange(len(t))
        cs = np.concatenate([[0.0], np.cumsum(a)])
        lo1 = np.searchsorted(t, t - HOUR, side="left")
        lo24 = np.searchsorted(t, t - DAY, side="left")
        lo30 = np.searchsorted(t, t - 30 * DAY, side="left")
        c1, c24, c30 = i - lo1, i - lo24, i - lo30
        s30 = cs[i] - cs[lo30]
        mean30 = np.where(c30 > 0, s30 / np.maximum(c30, 1), np.nan)
        gap = np.concatenate([[np.nan], np.diff(t).astype(float)])
        for name, vals in [("txn_count_1h", c1.astype(float)), ("txn_count_24h", c24.astype(float)),
                           ("amount_mean_30d", mean30), ("seconds_since_last_txn", gap)]:
            for k, idx in enumerate(rows):
                res[name][pos_of[idx]] = vals[k]
    for name, vals in res.items():
        out[name] = vals
    return out


def derive_features(df: pd.DataFrame) -> pd.DataFrame:
    """Row-level transforms. Works on training frames and on single-request frames."""
    out = df.copy()
    ts = pd.to_datetime(out[TIME_COL], errors="coerce")
    hour = ts.dt.hour + ts.dt.minute / 60.0
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    out["is_night"] = (hour < 6).astype(float).where(ts.notna())
    out["day_of_week"] = ts.dt.dayofweek.astype(float)
    amount = _col(out, "amount")
    out["amount_log"] = np.log1p(amount.clip(lower=0))
    mean30 = _col(out, "amount_mean_30d")
    ratio = amount / mean30.where(mean30 > 0)
    out["amount_ratio_log"] = np.log1p(ratio.clip(lower=0))
    out["seconds_since_last_log"] = np.log1p(_col(out, "seconds_since_last_txn").clip(lower=0))
    for c in ["is_international", "device_new", "account_age_days", "txn_count_1h", "txn_count_24h"]:
        out[c] = _col(out, c)
    return out
