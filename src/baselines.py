"""Interpretable baselines.

* PriorBaseline: predicts the training fraud rate for everyone (lower bound).
* heuristic_score: business rule built from fraud-risk signals. Also the FALLBACK scorer used by
  the API when the ML model cannot score a request.
"""
import numpy as np
import pandas as pd

from src.config import HIGH_RISK_CATEGORIES
from src.features import derive_features


class PriorBaseline:
    def fit(self, y):
        self.prior_ = float(np.mean(y))
        return self

    def predict_proba(self, n: int) -> np.ndarray:
        return np.full(n, self.prior_)


def heuristic_score(df: pd.DataFrame) -> np.ndarray:
    """Rule score in [0, 1]. Requires the raw transaction fields; derived fields are computed here."""
    d = derive_features(df)
    amount = pd.to_numeric(d["amount"], errors="coerce").fillna(0).to_numpy()
    intl = d["is_international"].fillna(0).to_numpy() > 0
    new_dev = d["device_new"].fillna(0).to_numpy() > 0
    night = d["is_night"].fillna(0).to_numpy() > 0
    risky_cat = d["merchant_category"].isin(HIGH_RISK_CATEGORIES).to_numpy()
    new_account = (d["account_age_days"].fillna(9999).to_numpy() < 90)

    score = np.full(len(d), 0.02)
    risky = intl | new_dev | night | risky_cat | new_account
    score[risky] = 0.10
    score[risky & (amount >= 200)] = 0.30
    score[(intl & new_dev) | ((intl | new_dev) & (amount >= 500))] = 0.60
    return score
