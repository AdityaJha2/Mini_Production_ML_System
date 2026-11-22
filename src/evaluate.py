"""Metrics, cost-sensitive threshold selection, plots and the test-set evaluation entry point.

Metric choice (justified in the report)
* PR-AUC: primary ranking metric. Fraud is ~0.6% of rows, so ROC-AUC is near-saturated by the
  huge negative class; PR-AUC measures how well the top of the ranked list captures fraud.
* Recall at Precision >= 0.50: operating metric. Of the transactions we send to step-up
  verification, at least half must be real fraud, and we report how much fraud that catches.
* Expected cost per 1,000 transactions: uses the business cost of a missed fraud and of a false
  alarm. This is the metric the threshold is chosen to minimise.
* F1-macro and F1-positive: class-balanced and minority-class views of the confusion matrix.
* Brier score: calibration of the probabilities returned to callers.
"""
from __future__ import annotations

import json

import numpy as np
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score, roc_auc_score)

from src.config import COST_FALSE_NEGATIVE, COST_FALSE_POSITIVE, TARGET_PRECISION


def select_threshold_cost(y, proba, fn_cost=COST_FALSE_NEGATIVE, fp_cost=COST_FALSE_POSITIVE) -> dict:
    """Threshold minimising total cost on the given (validation) data."""
    y = np.asarray(y).astype(int)
    grid = np.unique(np.quantile(proba, np.linspace(0, 1, 2001)))
    best = None
    for t in grid:
        pred = proba >= t
        fp = int(np.sum(pred & (y == 0)))
        fn = int(np.sum(~pred & (y == 1)))
        cost = fn * fn_cost + fp * fp_cost
        if best is None or cost < best["cost"]:
            best = {"threshold": float(t), "cost": float(cost), "fp": fp, "fn": fn}
    best["policy"] = f"minimise cost: {fn_cost:g} per missed fraud, {fp_cost:g} per false alarm"
    return best


def select_threshold_recall_at_precision(y, proba, target: float = TARGET_PRECISION):
    """Lowest threshold (highest recall) whose precision >= target. None if unreachable."""
    precision, recall, thresholds = precision_recall_curve(y, proba)
    p, r = precision[:-1], recall[:-1]
    ok = np.where(p >= target)[0]
    if len(ok) == 0:
        return None
    idx = ok[np.argmax(r[ok])]
    return {"threshold": float(thresholds[idx]), "precision": float(p[idx]), "recall": float(r[idx])}


def compute_metrics(y_true, proba, threshold: float) -> dict:
    y = np.asarray(y_true).astype(int)
    p = np.asarray(proba, dtype=float)
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    rap = select_threshold_recall_at_precision(y, p)
    cost_per_1k = (fn * COST_FALSE_NEGATIVE + fp * COST_FALSE_POSITIVE) / len(y) * 1000
    return {
        "threshold": round(float(threshold), 6),
        "pr_auc": round(float(average_precision_score(y, p)), 4),
        "roc_auc": round(float(roc_auc_score(y, p)), 4),
        "precision": round(float(precision_score(y, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y, pred, zero_division=0)), 4),
        "f1_positive": round(float(f1_score(y, pred, zero_division=0)), 4),
        "f1_macro": round(float(f1_score(y, pred, average="macro", zero_division=0)), 4),
        "brier": round(float(brier_score_loss(y, p)), 6),
        "recall_at_precision_0_5": round(float(rap["recall"]), 4) if rap else None,
        "flag_rate": round(float(pred.mean()), 5),
        "expected_cost_per_1000_txn": round(float(cost_per_1k), 2),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def plot_pr_curves(curves: dict, path, title="Precision-recall curves (test set)"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for name, (y, p) in curves.items():
        pr, rc, _ = precision_recall_curve(y, p)
        ax.plot(rc, pr, label=f"{name} (PR-AUC={average_precision_score(y, p):.3f})")
    prev = float(np.mean(next(iter(curves.values()))[0]))
    ax.axhline(prev, color="grey", ls="--", lw=1, label=f"No-skill ({prev:.4f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.set_yscale("log")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_confusion(cm: dict, path, title="Confusion matrix (test set)"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    mat = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    im = ax.imshow(mat, cmap="Blues")
    for (i, j), v in np.ndenumerate(mat):
        ax.text(j, i, f"{v:,}", ha="center", va="center",
                color="white" if mat[i, j] > mat.max() / 2 else "black")
    ax.set_xticks([0, 1], ["Pred: legit", "Pred: fraud"])
    ax.set_yticks([0, 1], ["True: legit", "True: fraud"])
    ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    import joblib
    import pandas as pd

    from src import config
    from src.baselines import PriorBaseline, heuristic_score
    from src.data_pipeline import load_transactions, temporal_split, xy
    from src.monitoring import drift_report, load_reference

    config.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    meta = json.loads(config.METADATA_PATH.read_text())
    raw = load_transactions()
    train, valid, test = temporal_split(raw)
    pre = joblib.load(config.PIPELINE_PATH)
    X_va, y_va = xy(valid, pre)
    X_te, y_te = xy(test, pre)

    champ_name = max(meta["champion_selection"], key=meta["champion_selection"].get)
    challenger_name = meta["challenger"]
    models = {
        "logistic_regression_balanced": joblib.load(config.BASELINE_PATH),
        "hist_gradient_boosting_balanced": joblib.load(config.CHALLENGER_PATH),
    }
    assert challenger_name != champ_name

    table, curves = {}, {}
    table["Prior (dummy)"] = {"role": "lower bound",
                              **compute_metrics(y_te, PriorBaseline().fit(train[config.TARGET]).predict_proba(len(y_te)), 0.5)}
    h_te = heuristic_score(test)
    table["Heuristic rule"] = {"role": "rule baseline / fallback",
                               **compute_metrics(y_te, h_te, config.HEURISTIC_THRESHOLD)}
    curves["Heuristic rule"] = (y_te, h_te)
    for key, mdl in models.items():
        p_va = mdl.predict_proba(X_va)[:, 1]
        p_te = mdl.predict_proba(X_te)[:, 1]
        if key == champ_name:
            thr = meta["decision_threshold"]
            # The champion threshold must be reproducible from the validation data alone.
            assert abs(select_threshold_cost(y_va, p_va)["threshold"] - thr) < 1e-9, "threshold drift"
        else:
            thr = select_threshold_cost(y_va, p_va)["threshold"]
        role = "champion" if key == champ_name else "challenger"
        table[key] = {"role": role, **compute_metrics(y_te, p_te, thr)}
        curves[key] = (y_te, p_te)

    with open(config.REPORT_DIR / "metrics_test.json", "w") as f:
        json.dump(table, f, indent=2)
    plot_pr_curves(curves, config.REPORT_DIR / "pr_curves_test.png")
    plot_confusion(table[champ_name]["confusion_matrix"], config.REPORT_DIR / "confusion_test.png")

    ref = load_reference(config.REFERENCE_STATS_PATH)
    both = pd.concat([valid, test]).copy()
    both["_month"] = pd.to_datetime(both["timestamp"]).dt.to_period("M").astype(str)
    monthly = []
    for month, part in both.groupby("_month"):
        rep = drift_report(ref, part)
        rep.insert(0, "month", month)
        monthly.append(rep)
    monthly = pd.concat(monthly, ignore_index=True)
    monthly.to_csv(config.REPORT_DIR / "drift_monthly.csv", index=False)
    print(pd.DataFrame(table).T[["role", "threshold", "pr_auc", "roc_auc", "precision", "recall",
                                 "recall_at_precision_0_5", "f1_macro", "expected_cost_per_1000_txn"]].to_string())
    print("\nDrift, test window vs training reference:")
    print(drift_report(ref, test).to_string(index=False))


if __name__ == "__main__":
    main()
