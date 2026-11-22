"""Operational drift monitoring.

Reference distributions are captured from the TRAINING partition when the model is trained.
For each raw input feature we compute:
* PSI (Population Stability Index) over 10 quantile bins (categoricals: category shares);
* Kolmogorov-Smirnov statistic and p-value (numeric features).

Alert bands: PSI < 0.10 OK; 0.10-0.25 WARNING; >= 0.25 or KS p < 0.01 ALERT.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy import stats

from src.config import CATEGORY_VALUES

NUMERIC_MONITORED = ["amount", "account_age_days", "txn_count_1h", "txn_count_24h",
                     "amount_mean_30d", "seconds_since_last_txn"]
CATEGORICAL_MONITORED = ["merchant_category", "channel"]
PSI_WARN, PSI_ALERT, KS_P_ALERT = 0.10, 0.25, 0.01
EPS, N_BINS, SAMPLE = 1e-6, 10, 2000


def _psi(expected, actual) -> float:
    e = np.clip(np.asarray(expected, float), EPS, None)
    a = np.clip(np.asarray(actual, float), EPS, None)
    e, a = e / e.sum(), a / a.sum()
    return float(np.sum((a - e) * np.log(a / e)))


def build_reference_stats(reference: pd.DataFrame, seed: int = 0) -> dict:
    out = {"numeric": {}, "categorical": {}, "amount_p999": float(reference["amount"].quantile(0.999))}
    for col in NUMERIC_MONITORED:
        x = pd.to_numeric(reference[col], errors="coerce")
        valid = x.dropna().to_numpy()
        edges = np.unique(np.quantile(valid, np.linspace(0, 1, N_BINS + 1)))
        edges[0], edges[-1] = -np.inf, np.inf
        counts, _ = np.histogram(valid, bins=edges)
        rng = np.random.default_rng(seed)
        sample = rng.choice(valid, size=min(SAMPLE, len(valid)), replace=False)
        out["numeric"][col] = {
            "edges": [None if not np.isfinite(e) else float(e) for e in edges],
            "proportions": (counts / counts.sum()).tolist(),
            "missing_rate": float(x.isna().mean()),
            "sample": sample.round(4).tolist(),
        }
    for col in CATEGORICAL_MONITORED:
        cats = CATEGORY_VALUES[col]
        props = reference[col].value_counts(normalize=True).reindex(cats, fill_value=0.0)
        out["categorical"][col] = {"categories": cats, "proportions": props.tolist(),
                                   "missing_rate": float(reference[col].isna().mean())}
    return out


def save_reference(d: dict, path):
    with open(path, "w") as f:
        json.dump(d, f)


def load_reference(path) -> dict:
    with open(path) as f:
        return json.load(f)


KS_STAT_MIN = 0.05  # require a practical effect size, not only statistical significance


def _status(psi, p, ks_stat=None):
    """ALERT if PSI is large, or the KS test is significant AND the shift is practically sized.
    With ~50k rows per window, KS p-values are significant for trivially small shifts."""
    ks_material = ks_stat is not None and ks_stat >= KS_STAT_MIN
    if psi >= PSI_ALERT or (p is not None and p < KS_P_ALERT and ks_material):
        return "ALERT"
    if psi >= PSI_WARN:
        return "WARNING"
    return "OK"


def drift_report(reference: dict, current: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col, ref in reference["numeric"].items():
        x = pd.to_numeric(current[col], errors="coerce")
        valid = x.dropna().to_numpy()
        edges = np.array([-np.inf if e is None else e for e in ref["edges"]])
        edges[0], edges[-1] = -np.inf, np.inf
        if len(valid) == 0:
            rows.append({"feature": col, "type": "numeric", "psi": np.nan, "ks_stat": np.nan, "ks_p": np.nan,
                         "missing_rate_ref": ref["missing_rate"], "missing_rate_cur": 1.0, "status": "ALERT"})
            continue
        counts, _ = np.histogram(valid, bins=edges)
        psi = _psi(ref["proportions"], counts / max(counts.sum(), 1))
        ks = stats.ks_2samp(np.array(ref["sample"]), valid)
        rows.append({"feature": col, "type": "numeric", "psi": round(psi, 4),
                     "ks_stat": round(float(ks.statistic), 4), "ks_p": float(f"{ks.pvalue:.3g}"),
                     "missing_rate_ref": round(ref["missing_rate"], 4),
                     "missing_rate_cur": round(float(x.isna().mean()), 4),
                     "status": _status(psi, float(ks.pvalue), float(ks.statistic))})
    for col, ref in reference["categorical"].items():
        cur = current[col].value_counts(normalize=True).reindex(ref["categories"], fill_value=0.0).to_numpy()
        psi = _psi(ref["proportions"], cur)
        rows.append({"feature": col, "type": "categorical", "psi": round(psi, 4), "ks_stat": np.nan,
                     "ks_p": np.nan, "missing_rate_ref": round(ref["missing_rate"], 4),
                     "missing_rate_cur": round(float(current[col].isna().mean()), 4),
                     "status": _status(psi, None)})
    return pd.DataFrame(rows)


def system_alerts(error_rate: float, p95_ms: float, fallback_rate: float, slow_rate: float) -> list[str]:
    alerts = []
    if error_rate > 0.01:
        alerts.append(f"5xx error rate {error_rate:.2%} above 1% (page on-call)")
    if p95_ms > 100:
        alerts.append(f"p95 latency {p95_ms:.1f} ms above 100 ms SLO")
    if fallback_rate > 0.05:
        alerts.append(f"fallback rate {fallback_rate:.2%} above 5% (model scoring degraded)")
    if slow_rate > 0.02:
        alerts.append(f"over-budget requests {slow_rate:.2%} above 2% (scoring latency budget)")
    return alerts
