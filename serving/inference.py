"""Scoring core shared by the REST API and the batch CLI.

Four-tier degradation (Phase 4 fallback and reliability)
    1. model                : champion ML pipeline. Flag if probability >= cost-optimal threshold.
    2. fallback_heuristic   : fraud-risk rule. Flag if rule score >= HEURISTIC_THRESHOLD.
    3. fallback_prior       : model and rule both unavailable. Fail-safe: decision flag_for_review.
Each prediction names the tier that produced it.
"""
from __future__ import annotations

import json
import logging
import time

import joblib
import numpy as np
import pandas as pd

from src import config
from src.baselines import heuristic_score
from src.features import derive_features

log = logging.getLogger("inference")


class FraudScorer:
    def __init__(self):
        self.pipeline = None
        self.model = None
        self.metadata: dict = {}
        self.threshold = 0.5
        self.heuristic_threshold = config.HEURISTIC_THRESHOLD
        self.prior = 0.006
        self.amount_p999 = None
        self.model_version = "unloaded"
        self.force_model_failure = False      # test / robustness hook: simulate a model outage
        self.force_heuristic_failure = False  # test / robustness hook: simulate a rule-engine outage

    def load(self):
        t0 = time.perf_counter()
        self.pipeline = joblib.load(config.PIPELINE_PATH)
        self.model = joblib.load(config.MODEL_PATH)
        with open(config.METADATA_PATH) as f:
            self.metadata = json.load(f)
        self.threshold = float(self.metadata["decision_threshold"])
        self.heuristic_threshold = float(self.metadata["heuristic_threshold"])
        self.amount_p999 = float(self.metadata["amount_p999_train"])
        self.prior = float(self.metadata["train_fraud_rate"])
        self.model_version = self.metadata["model_version"]
        log.info("Loaded %s (threshold %.6f) in %.3fs", self.model_version, self.threshold,
                 time.perf_counter() - t0)
        return self

    @property
    def model_loaded(self) -> bool:
        return self.model is not None and self.pipeline is not None

    def _model_proba(self, df: pd.DataFrame) -> np.ndarray:
        if self.force_model_failure or not self.model_loaded:
            raise RuntimeError("model scoring unavailable")
        X = self.pipeline.transform(derive_features(df)[config.MODEL_FEATURES])
        return self.model.predict_proba(X)[:, 1]

    def score(self, df: pd.DataFrame):
        """Returns (probability, scorer_name, per-row warnings) for a validated DataFrame."""
        warnings = self.data_warnings(df)
        try:
            return self._model_proba(df), "model", warnings
        except Exception as exc:
            log.warning("Model scoring failed (%s); using rule fallback", exc)
        try:
            if self.force_heuristic_failure:
                raise RuntimeError("rule engine unavailable")
            return heuristic_score(df), "fallback_heuristic", warnings
        except Exception as exc:
            log.error("Rule fallback failed (%s); using prior (fail-safe review)", exc)
            return np.full(len(df), self.prior), "fallback_prior", warnings

    def flags(self, proba: np.ndarray, scorer: str) -> np.ndarray:
        if scorer == "fallback_prior":
            return np.zeros(len(proba), dtype=int)  # prior carries no per-transaction signal
        thr = self.heuristic_threshold if scorer == "fallback_heuristic" else self.threshold
        return (proba >= thr).astype(int)

    @staticmethod
    def decisions(flags: np.ndarray, scorer: str) -> np.ndarray:
        if scorer == "fallback_prior":
            return np.array(["flag_for_review"] * len(flags), dtype=object)
        return np.where(flags == 1, "step_up_verification", "approve").astype(object)

    def data_warnings(self, df: pd.DataFrame) -> list[list[str]]:
        out = [[] for _ in range(len(df))]
        known = set(config.CATEGORY_VALUES["merchant_category"])
        if "merchant_category" in df.columns:
            cat = df["merchant_category"]
            for i in np.where(cat.isna().to_numpy())[0]:
                out[i].append("merchant_category missing; treated as unknown")
            for i in np.where((cat.notna() & ~cat.isin(list(known))).to_numpy())[0]:
                out[i].append("unseen merchant_category; encoded as all-zero")
        if "channel" in df.columns:
            for i in np.where((~df["channel"].isin(config.CATEGORY_VALUES["channel"])).to_numpy())[0]:
                out[i].append("unseen channel; encoded as all-zero")
        for col, msg in [
            ("account_age_days", "account_age_days missing; imputed with training median"),
            ("amount_mean_30d", "no 30-day history (new card); ratio feature imputed"),
            ("seconds_since_last_txn", "no previous transaction; gap feature imputed"),
        ]:
            if col in df.columns:
                for i in np.where(df[col].isna().to_numpy())[0]:
                    out[i].append(msg)
        if self.amount_p999 is not None and "amount" in df.columns:
            for i in np.where((df["amount"] > self.amount_p999).to_numpy())[0]:
                out[i].append("amount above training 99.9th percentile")
        return out

    def health(self) -> dict:
        return {"status": "ok" if self.model_loaded else "degraded", "model_loaded": self.model_loaded,
                "fallback_available": True, "model_version": self.model_version}
