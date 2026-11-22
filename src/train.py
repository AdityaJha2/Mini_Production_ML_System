"""Phase 2: fit baselines and models, select the champion, tune the threshold, export artifacts.

1. Temporal split (train / validation / test by day).
2. Preprocessing fitted on TRAIN only.
3. Candidates, all class-weighted (no oversampling):
     - balanced logistic regression (interpretable linear model)
     - HistGradientBoosting grid (8 configurations)
4. Champion = highest VALIDATION PR-AUC. The other model is kept as challenger.
5. Threshold = cost-minimising cut-off on VALIDATION (fraud miss cost vs false-alarm cost).
6. Serialize pipeline, models, reference drift statistics and metadata. The TEST set is not used.
"""
from __future__ import annotations

import itertools
import json
import logging
import platform
import time
from datetime import datetime, timezone

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

from src import config
from src.baselines import PriorBaseline, heuristic_score
from src.data_pipeline import describe_split, fit_preprocessor, load_transactions, temporal_split, xy
from src.evaluate import compute_metrics, select_threshold_cost
from src.monitoring import build_reference_stats, save_reference

log = logging.getLogger("train")
PARAM_GRID = {"learning_rate": [0.05, 0.1], "max_leaf_nodes": [15, 31], "l2_regularization": [0.0, 1.0]}
MAX_ITER = 300
MODEL_VERSION = "fraud-v1"


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config.ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    raw = load_transactions()
    train, valid, test = temporal_split(raw)
    log.info("Split sizes: train=%d valid=%d test=%d", len(train), len(valid), len(test))

    pre = fit_preprocessor(train)
    X_tr, y_tr = xy(train, pre)
    X_va, y_va = xy(valid, pre)
    log.info("Train fraud rate %.5f; transformed dimension %d", y_tr.mean(), X_tr.shape[1])

    prior = PriorBaseline().fit(y_tr)
    heur_va = heuristic_score(valid)
    log.info("Prior valid PR-AUC %.4f | heuristic valid PR-AUC %.4f",
             average_precision_score(y_va, prior.predict_proba(len(y_va))),
             average_precision_score(y_va, heur_va))

    logit = LogisticRegression(class_weight="balanced", max_iter=3000, C=1.0, random_state=config.RANDOM_STATE)
    logit.fit(X_tr, y_tr)
    logit_va = logit.predict_proba(X_va)[:, 1]
    log.info("Logistic (balanced) valid PR-AUC %.4f", average_precision_score(y_va, logit_va))

    best, search = None, []
    for lr, leaves, l2 in itertools.product(*PARAM_GRID.values()):
        params = {"learning_rate": lr, "max_leaf_nodes": leaves, "l2_regularization": l2}
        m = HistGradientBoostingClassifier(max_iter=MAX_ITER, early_stopping=False, class_weight="balanced",
                                           random_state=config.RANDOM_STATE, **params)
        m.fit(X_tr, y_tr)
        p = m.predict_proba(X_va)[:, 1]
        pr = float(average_precision_score(y_va, p))
        search.append({**params, "valid_pr_auc": round(pr, 4)})
        log.info("HGB %s -> valid PR-AUC %.4f", params, pr)
        if best is None or pr > best["pr_auc"]:
            best = {"model": m, "params": params, "pr_auc": pr, "proba_va": p}

    candidates = {
        "logistic_regression_balanced": {"model": logit, "pr_auc": float(average_precision_score(y_va, logit_va)),
                                         "proba_va": logit_va, "params": {"C": 1.0}},
        "hist_gradient_boosting_balanced": {"model": best["model"], "pr_auc": best["pr_auc"],
                                            "proba_va": best["proba_va"], "params": best["params"]},
    }
    champion_name = max(candidates, key=lambda k: candidates[k]["pr_auc"])
    challenger_name = next(k for k in candidates if k != champion_name)
    champ = candidates[champion_name]
    log.info("Champion: %s (valid PR-AUC %.4f)", champion_name, champ["pr_auc"])

    thr = select_threshold_cost(y_va, champ["proba_va"])
    val_metrics = compute_metrics(y_va, champ["proba_va"], thr["threshold"])
    log.info("Cost-optimal threshold %.6f (%s) | validation %s", thr["threshold"], thr["policy"],
             {k: val_metrics[k] for k in ["pr_auc", "precision", "recall", "expected_cost_per_1000_txn"]})

    joblib.dump(pre, config.PIPELINE_PATH)
    joblib.dump(champ["model"], config.MODEL_PATH)
    joblib.dump(candidates[challenger_name]["model"], config.CHALLENGER_PATH)
    joblib.dump(logit, config.BASELINE_PATH)
    save_reference(build_reference_stats(train), config.REFERENCE_STATS_PATH)

    ref = build_reference_stats(train)
    meta = {
        "model_version": f"{MODEL_VERSION}-{champion_name}",
        "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "champion_selection": {k: round(v["pr_auc"], 4) for k, v in candidates.items()},
        "champion": champion_name,
        "challenger": challenger_name,
        "hyperparameters": champ["params"],
        "hyperparameter_search": search,
        "decision_threshold": thr["threshold"],
        "threshold_policy": thr["policy"],
        "cost_assumptions": {"false_negative": config.COST_FALSE_NEGATIVE,
                             "false_positive": config.COST_FALSE_POSITIVE},
        "heuristic_threshold": config.HEURISTIC_THRESHOLD,
        "amount_p999_train": ref["amount_p999"],
        "features": config.MODEL_FEATURES,
        "transformed_feature_names": list(pre.get_feature_names_out()),
        "splits": describe_split(train, valid, test),
        "train_fraud_rate": round(float(y_tr.mean()), 6),
        "validation_metrics": val_metrics,
        "environment": {"python": platform.python_version(), "sklearn": sklearn.__version__,
                        "numpy": np.__version__},
        "training_seconds": round(time.perf_counter() - t0, 2),
    }
    with open(config.METADATA_PATH, "w") as f:
        json.dump(meta, f, indent=2, default=str)
    log.info("Artifacts written to %s (%.1fs)", config.ARTIFACT_DIR, time.perf_counter() - t0)


if __name__ == "__main__":
    main()
