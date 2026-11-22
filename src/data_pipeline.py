"""Phase 1 & 2 data handling: ingestion, validation, temporal split and leak-free preprocessing.

Leakage safeguards
------------------
1. Velocity features are computed from past transactions only (features.add_velocity_features).
2. The split is by time (day index), never random. Train < validation < test, asserted.
3. The ColumnTransformer (imputers, scaler, encoder) is fitted on TRAIN rows only; validation,
   test and inference rows are transformed with the fitted object.
4. Unseen categories at inference are encoded as all zeros (handle_unknown='ignore').
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config
from src.config import (ID_COL, MODEL_CATEGORICAL, MODEL_FEATURES, MODEL_NUMERIC, RAW_PATH, RAW_COLUMNS,
                        START_DATE, TARGET, TIME_COL, TRAIN_END_DAY, VALID_END_DAY)
from src.features import add_velocity_features, derive_features

log = logging.getLogger(__name__)


def load_transactions(path=RAW_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in RAW_COLUMNS + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"Raw transaction file is missing columns: {missing}")
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="raise")
    if df[ID_COL].duplicated().any():
        raise ValueError("Duplicate transaction_id values found")
    df[TARGET] = df[TARGET].astype(int)
    df = df.sort_values([TIME_COL, ID_COL]).reset_index(drop=True)
    df = add_velocity_features(df)
    log.info("Loaded %d transactions (fraud rate %.4f)", len(df), df[TARGET].mean())
    return df


def day_index(df: pd.DataFrame) -> pd.Series:
    return (pd.to_datetime(df[TIME_COL]).dt.normalize() - pd.Timestamp(START_DATE)).dt.days


def temporal_split(df: pd.DataFrame):
    day = day_index(df)
    train = df[day <= TRAIN_END_DAY].copy()
    valid = df[(day > TRAIN_END_DAY) & (day <= VALID_END_DAY)].copy()
    test = df[day > VALID_END_DAY].copy()
    if min(len(train), len(valid), len(test)) == 0:
        raise ValueError("Temporal split produced an empty partition")
    assert pd.to_datetime(train[TIME_COL]).max() < pd.to_datetime(valid[TIME_COL]).min()
    assert pd.to_datetime(valid[TIME_COL]).max() < pd.to_datetime(test[TIME_COL]).min()
    return train, valid, test


def build_preprocessor() -> ColumnTransformer:
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="unknown")),
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", numeric, MODEL_NUMERIC),
        ("cat", categorical, MODEL_CATEGORICAL),
    ], remainder="drop", verbose_feature_names_out=False)


def fit_preprocessor(train: pd.DataFrame) -> ColumnTransformer:
    pre = build_preprocessor()
    pre.fit(derive_features(train)[MODEL_FEATURES])
    return pre


def xy(df: pd.DataFrame, pre: ColumnTransformer):
    feats = derive_features(df)
    X = pre.transform(feats[MODEL_FEATURES])
    y = df[TARGET].to_numpy() if TARGET in df.columns else None
    return X, y


def describe_split(train, valid, test) -> list[dict]:
    rows = []
    for name, part in [("train", train), ("validation", valid), ("test", test)]:
        rows.append({
            "partition": name,
            "dates": f"{pd.to_datetime(part[TIME_COL]).min():%Y-%m-%d} to {pd.to_datetime(part[TIME_COL]).max():%Y-%m-%d}",
            "rows": int(len(part)),
            "cards": int(part["customer_id"].nunique()),
            "frauds": int(part[TARGET].sum()),
            "fraud_rate": round(float(part[TARGET].mean()), 5),
        })
    return rows


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    raw = load_transactions()
    tr, va, te = temporal_split(raw)
    for r in describe_split(tr, va, te):
        print(r)
    pre = fit_preprocessor(tr)
    X, _ = xy(tr, pre)
    print("transformed features:", X.shape[1], list(pre.get_feature_names_out())[:6], "...")
