"""Build the consolidated assignment PDF from the project's own outputs.

    python -m report.build_report --student-id 2025em1200044 --student-name "Aditya Jha" \
        --repo-url https://github.com/AdityaJha2/Mini_Production_ML_System

Every number is read from artifacts/ or reports/ (and the sample request is scored live), so the PDF
matches the code that produced it.
"""
from __future__ import annotations

import argparse
import json
import re
from datetime import date
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

from report.figures import architecture
from src import config

REP = config.REPORT_DIR
FIG = REP / "figures"


def _load(p):
    return json.loads(Path(p).read_text())


def _styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("title", parent=base["Title"], fontSize=20, leading=24, spaceAfter=10),
        "sub": ParagraphStyle("sub", parent=base["Normal"], fontSize=11, alignment=TA_CENTER, leading=15),
        "h1": ParagraphStyle("h1", parent=base["Heading1"], fontSize=14, spaceBefore=10, spaceAfter=6,
                             textColor=colors.HexColor("#1e3a8a"), keepWithNext=1),
        "body": ParagraphStyle("body", parent=base["Normal"], fontSize=9.2, leading=12.6, spaceAfter=5),
        "small": ParagraphStyle("small", parent=base["Normal"], fontSize=7.6, leading=9.5,
                                textColor=colors.HexColor("#444")),
        "cell": ParagraphStyle("cell", parent=base["Normal"], fontSize=7.6, leading=9.3),
        "cellb": ParagraphStyle("cellb", parent=base["Normal"], fontSize=7.6, leading=9.3, fontName="Helvetica-Bold"),
    }


def tbl(rows, widths, S, header=True):
    data = [[Paragraph(str(c), S["cellb"] if (header and i == 0) else S["cell"]) for c in row]
            for i, row in enumerate(rows)]
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    st = [("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#9ca3af")), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
          ("TOPPADDING", (0, 0), (-1, -1), 2.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5)]
    if header:
        st.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e0e7ff")))
    t.setStyle(TableStyle(st))
    return t


def _live_example():
    """Score the sample request with the real trained model so the report shows a real response."""
    from serving.inference import FraudScorer
    rec = _load(ROOT_EX / "one_request.json")
    s = FraudScorer().load()
    df = pd.DataFrame([rec])
    proba, scorer_name, warns = s.score(df)
    flags = s.flags(proba, scorer_name)
    return {"transaction_id": rec["transaction_id"], "fraud_probability": round(float(proba[0]), 4),
            "fraud_flag": int(flags[0]), "decision": str(s.decisions(flags, scorer_name)[0]),
            "threshold": round(s.threshold, 6), "scorer": scorer_name, "model_version": s.model_version,
            "warnings": warns[0]}


ROOT_EX = config.ROOT / "examples"


def build(student_id, student_name, repo_url, out_path: Path):
    S = _styles()
    meta = _load(config.METADATA_PATH)
    test = _load(REP / "metrics_test.json")
    rob = _load(REP / "robustness.json")
    bapi = _load(REP / "benchmark_api.json")
    bbat = _load(REP / "benchmark_batch.json")
    drift = pd.read_csv(REP / "drift_monthly.csv")
    import subprocess, sys
    coll = subprocess.run([sys.executable, "-m", "pytest", "--co", "-q"], cwd=config.ROOT,
                          capture_output=True, text=True).stdout
    m = re.search(r"(\d+) tests? collected", coll)
    n_tests = int(m.group(1)) if m else -1
    bc8 = _load(REP / "benchmark_api_c8.json")
    bc16 = _load(REP / "benchmark_api_c16.json")
    FIG.mkdir(parents=True, exist_ok=True)
    architecture(FIG / "architecture.png")
    example = _live_example()

    champ = meta["champion"]
    champ_row = test[next(k for k in test if test[k]["role"] == "champion")]
    cm_ = champ_row["confusion_matrix"]
    thr = meta["decision_threshold"]
    vm = meta["validation_metrics"]
    W = A4[0] - 4 * cm
    story, add = [], None
    add = story.append

    # ------------------------------------------------------------ title page
    add(Spacer(1, 2.0 * cm))
    add(Paragraph("BITS Pilani - Work Integrated Learning Programmes (WILP)", S["sub"]))
    add(Paragraph("MSc / Post-Graduate / Executive ML Model Engineering", S["sub"]))
    add(Spacer(1, 1.1 * cm))
    add(Paragraph("Design &amp; Build a Mini Production ML System", S["title"]))
    add(Paragraph("<b>Option B: Financial Transaction Fraud Detection</b> (extreme class imbalance, real-time scoring)", S["sub"]))
    add(Paragraph("Leakage-free temporal preprocessing, cost-sensitive modelling, FastAPI and batch serving, "
                  "latency and throughput profiling, robustness testing, drift monitoring and fallback logic", S["sub"]))
    add(Spacer(1, 0.9 * cm))
    add(tbl([
        ["Student name", student_name],
        ["Roll number", student_id],
        ["Course", "ML Model Engineering - Course Assignment (20 marks)"],
        ["Code repository", f'<a href="{repo_url}" color="#1d4ed8"><u>{repo_url}</u></a>'],
        ["Model version", meta["model_version"]],
        ["Report date", date.today().isoformat()],
    ], [4.2 * cm, W - 4.2 * cm], S, header=False))
    add(Spacer(1, 0.6 * cm))
    add(Paragraph("<b>Data disclosure.</b> The build environment could not download a public fraud dataset (Kaggle "
                  "or IEEE-CIS), so the data is a reproducible synthetic card-transaction dataset generated by "
                  "<font face='Courier'>src/generate_data.py</font> (seed 42). It has the shape of a real feed: one "
                  "row per transaction, a rare binary label (about 0.6%), skewed amounts, missing values and "
                  "compromised-card bursts. Absolute metric values will differ on real data. Section 7 covers this.",
                  S["body"]))
    add(PageBreak())

    # ------------------------------------------------------------ Executive summary
    tm = {k: v for k, v in test.items()}
    ch = next(v for k, v in tm.items() if v["role"] == "champion")
    pr_ = tm["Prior (dummy)"]
    add(Paragraph("Executive Summary", S["h1"]))
    add(Paragraph(
        f"This report documents a card-fraud detection system built as a complete production pipeline, from leak-free "
        f"preprocessing to a monitored API. The champion model, a class-weighted logistic regression, reaches test PR-AUC "
        f"{ch['pr_auc']:.3f} against {pr_['pr_auc']:.4f} for the prior. A cost-driven threshold reduces expected loss from "
        f"{pr_['expected_cost_per_1000_txn']:.0f} to {ch['expected_cost_per_1000_txn']:.0f} per 1,000 transactions on the "
        f"held-out test window (days 150 to 179). The online API answers with p50 {bapi['latency_ms']['p50']:.1f} ms and "
        f"p95 {bapi['latency_ms']['p95']:.1f} ms, with {bapi['errors']} errors in {bapi['requests']:,} requests. The batch "
        f"CLI scores about {bbat['end_to_end_cli']['records_per_second']:,.0f} records per second end to end. Scoring "
        f"degrades through three fallback tiers, so every request still receives a decision. Under 8 concurrent clients on the "
        f"single-core build machine, p95 rose to {bc8['latency_ms']['p95']:.0f} ms, above the 100 ms SLO, so the production "
        f"target needs more CPU capacity and a load test (Section 5.1).", S["body"]))
    add(tbl([
        ["Key result", "Value", "Where to verify"],
        ["Champion test PR-AUC", f"{ch['pr_auc']:.3f}", "reports/metrics_test.json"],
        ["Expected cost per 1,000 transactions (prior to champion)", f"{pr_['expected_cost_per_1000_txn']:.0f} to {ch['expected_cost_per_1000_txn']:.0f}", "reports/metrics_test.json"],
        ["API latency p50 / p95 / p99 (ms)", f"{bapi['latency_ms']['p50']:.1f} / {bapi['latency_ms']['p95']:.1f} / {bapi['latency_ms']['p99']:.1f}", "reports/benchmark_api.json"],
        ["Batch throughput (records/s, end to end)", f"{bbat['end_to_end_cli']['records_per_second']:,.0f}", "reports/benchmark_batch.json"],
        ["Robustness scenarios run", f"{len(rob['scenarios'])}", "reports/robustness.json"],
    ], [7.2 * cm, 4.2 * cm, W - 11.4 * cm], S))
    add(Spacer(1, 5))
    add(Paragraph("<b>Recommendation.</b> Deploy the champion behind the validated API, with the batch path for overnight "
                  "rescoring. Before go-live, recalibrate the probabilities, confirm the cost assumptions with the fraud "
                  "operations team, and validate the model on the bank's own labelled transactions. Two limitations apply "
                  "throughout: the data is synthetic (Section 7), and the costs are assumptions.", S["body"]))
    add(PageBreak())

    # ------------------------------------------------------------ 1
    add(Paragraph("1. Problem Formulation, Scope &amp; Inputs", S["h1"]))
    add(Paragraph("<b>Business objective.</b> Stop card fraud at authorisation without blocking genuine customers. "
                  "Each transaction is scored for fraud risk; the top-risk transactions go to step-up verification "
                  "(one-time password or 3-D Secure) and the rest are approved. Each missed fraud costs the bank the "
                  "lost amount plus fees, while each false alarm costs a verification step and some customer friction.",
                  S["body"]))
    add(Paragraph("<b>Target.</b> <font face='Courier'>is_fraud</font> is 1 when the transaction was later confirmed as "
                  "fraud (chargeback or fraud report), 0 otherwise. The unit is one card transaction.", S["body"]))
    add(tbl([
        ["Input field", "Type", "Meaning", "Expected effect on fraud risk"],
        ["amount", "numeric", "transaction amount", "high amounts raise risk (log scale)"],
        ["merchant_category", "categorical", "10 categories, incl. gift cards and online gaming", "high-risk categories raise risk"],
        ["channel", "categorical", "online / pos / atm", "online raises risk"],
        ["is_international", "binary", "foreign-currency or foreign-country transaction", "raises risk"],
        ["device_new", "binary", "first use of this device for the card", "raises risk"],
        ["timestamp", "datetime", "hour, weekday (sin/cos, night flag)", "night-time raises risk"],
        ["account_age_days", "numeric", "age of the card account", "new accounts (under 90 days) raise risk"],
        ["txn_count_1h / 24h", "numeric", "earlier transactions on the card in the last hour / day", "bursts indicate compromise"],
        ["amount_mean_30d", "numeric", "card's mean amount over the previous 30 days", "amount relative to normal spending"],
        ["seconds_since_last_txn", "numeric", "time since the previous transaction on the card", "very short gaps raise risk"],
    ], [3.4 * cm, 2.2 * cm, 5.3 * cm, W - 10.9 * cm], S))
    add(Spacer(1, 5))
    add(Paragraph("<b>Serving paradigm: hybrid.</b> The <b>online path</b> returns a decision for a single transaction "
                  "during authorisation, with a per-request scoring budget of 50 ms and a p95 SLO of 100 ms. The "
                  "<b>batch path</b> rescores the day's posted transactions overnight for case management and for "
                  "back-testing thresholds. Both use one validation contract and one scorer.", S["body"]))
    add(Paragraph(f"<b>Metric justification.</b> Fraud is {meta['train_fraud_rate']:.2%} of training rows. Accuracy "
                  f"would be {1 - meta['train_fraud_rate']:.1%} for a model that approves everything, so it is not used. The primary metric is "
                  "<b>PR-AUC</b>, which measures how well the model ranks fraud at the top of the list. ROC-AUC is "
                  "reported but is near-saturated by the large legitimate class. <b>Recall at precision 0.50</b> is "
                  "the operating metric: at least half of the transactions sent to verification must be real fraud. "
                  "<b>Expected cost per 1,000 transactions</b> uses the business costs and is the quantity the threshold "
                  "minimises. <b>F1-macro</b> covers both classes, and the <b>Brier score</b> checks probability calibration.",
                  S["body"]))

    # ------------------------------------------------------------ 2
    add(Paragraph("2. Data Ingestion, Partitioning &amp; Leakage Safeguards", S["h1"]))
    add(Paragraph("<b>2.1 Dataset.</b> About 304,000 transactions over 180 days, from 5,000 cards. Roughly 0.6% are "
                  "fraud. Three quality issues are deliberate: missing <font face='Courier'>account_age_days</font> "
                  "(about 1.5%), missing <font face='Courier'>merchant_category</font> (about 0.5%), and bursts of 25 "
                  "transactions on about 3% of cards (compromised cards). The loader checks required columns, parses "
                  "timestamps and rejects duplicate transaction IDs.", S["body"]))
    add(Paragraph("<b>Why this data.</b> The rubric asks for a well-justified data choice. The build environment had no "
                  "access to Kaggle or IEEE-CIS, so the generator reproduces the properties that matter for this design: a "
                  "rare binary label, a timeline long enough for a temporal split, history-dependent signals (bursts "
                  "on compromised cards), realistic missingness, and heavy-tailed amounts. Because the data is "
                  "reproducible from a seed, every reported number can be regenerated and audited. The limitation is "
                  "stated, not hidden: the absolute metrics and the strength of individual signals are not evidence "
                  "about real fraud. The replacement path is documented in the README. A real CSV with the same columns "
                  "needs no code change; only the velocity features are recomputed from its own history.", S["body"]))
    add(Paragraph("<b>2.2 Partitioning.</b> Fraud patterns change over time, so the split is by calendar day, not at "
                  "random. Training covers days 0 to 119, validation days 120 to 149 and test days 150 to 179. "
                  "Every choice (champion, hyper-parameters, threshold) is made on validation data. The test set is "
                  "used once, in <font face='Courier'>src/evaluate.py</font>.", S["body"]))
    rows = [["Partition", "Dates", "Transactions", "Cards", "Frauds", "Fraud rate"]]
    for sp in meta["splits"]:
        rows.append([sp["partition"], sp["dates"], f"{sp['rows']:,}", f"{sp['cards']:,}",
                     f"{sp['frauds']:,}", f"{sp['fraud_rate']:.3%}"])
    add(tbl(rows, [2.6 * cm, 4.6 * cm, 2.6 * cm, 2.0 * cm, 2.0 * cm, 2.6 * cm], S))
    add(Spacer(1, 5))
    add(Paragraph("<b>2.3 Causal history features.</b> The four velocity features are the most important design choice. "
                  "Each row sees only earlier transactions on the same card, found with a binary search over that "
                  "card's time-ordered history. This is an explicit leakage check, not just a convention: "
                  "<font face='Courier'>tests/test_pipeline.py</font> alters later transactions and confirms that "
                  "earlier features do not change. In production the upstream feature service computes the same values "
                  "from its transaction store.", S["body"]))
    add(Paragraph("<b>2.4 Leakage audit.</b> Each potential source of look-ahead or information leakage, how it is "
                  "prevented, and the test or code path that verifies it:", S["body"]))
    add(tbl([
        ["Potential leak", "Prevention in the code", "Verification"],
        ["Future transactions used for velocity features", "history found by binary search over earlier timestamps only",
         "tests/test_pipeline.py: altering later rows leaves earlier features unchanged"],
        ["Random split mixes past and future", "split by calendar day with assertions for no overlap",
         "tests/test_pipeline.py: temporal split has no time overlap"],
        ["Preprocessing statistics from validation or test", "imputers, scaler and encoder fitted on training rows only",
         "tests/test_pipeline.py: scaler mean equals the training mean, not the full-data mean"],
        ["Threshold tuned on the test set", "cost threshold chosen on validation data in train.py",
         "src/evaluate.py asserts the stored threshold reproduces from validation data"],
        ["Champion chosen on the test set", "candidates compared on validation PR-AUC before test is read",
         "train.py logs the selection; evaluate.py loads only the saved champion"],
        ["Unseen category leaks through encoding", "unseen categories encoded as all zeros, never refitted",
         "tests/test_pipeline.py: unseen category produces an all-zero block"],
        ["Drift reference contains test data", "reference statistics built from the training partition only",
         "train.py calls build_reference_stats(train) only"],
    ], [4.6 * cm, 6.2 * cm, W - 10.8 * cm], S))

    # ------------------------------------------------------------ 3
    add(Paragraph("3. Preprocessing, Baseline &amp; Model Engineering", S["h1"]))
    add(Paragraph(f"<b>3.1 Preprocessing.</b> Numeric features are median-imputed (with a missing-value indicator) and "
                  f"standardised. Categorical features get an 'unknown' fill and one-hot encoding, with "
                  f"<font face='Courier'>handle_unknown='ignore'</font>. The result is "
                  f"{len(meta['transformed_feature_names'])} model inputs. The fitted pipeline is serialized as "
                  f"<font face='Courier'>artifacts/preprocessing_pipeline.joblib</font>.", S["body"]))
    add(Paragraph("<b>3.2 Baselines.</b> <b>Prior</b> flags nobody and scores everyone at the base rate. The "
                  "<b>heuristic rule</b> scores transactions from fraud signals (international, new device, night, "
                  "high-risk category, new account, and amount), and it is also the first fallback tier. The "
                  "<b>balanced logistic regression</b> is an interpretable linear model.", S["body"]))
    cs = meta["champion_selection"]
    hp = meta["hyperparameter_search"]
    add(Paragraph(f"<b>3.3 Model selection and imbalance handling.</b> Imbalance is handled with class-weighted loss "
                  f"(<font face='Courier'>class_weight='balanced'</font>) and a cost-based threshold. Oversampling is not "
                  f"used. Two candidates were compared on validation PR-AUC. One is the balanced logistic regression "
                  f"({cs['logistic_regression_balanced']:.4f}). The other is a histogram gradient-boosting model with an "
                  f"8-point grid (learning rate, leaf count, L2), which scored between "
                  f"{min(r['valid_pr_auc'] for r in hp):.4f} and {max(r['valid_pr_auc'] for r in hp):.4f}. The "
                  f"<b>{meta['champion'].replace('_', ' ')}</b> model was selected by the "
                  f"pre-defined rule. The generator's risk is close to linear in the log-odds, which explains why the "
                  f"simpler model is competitive.", S["body"]))
    add(Paragraph(f"<b>3.4 Threshold policy.</b> The threshold minimises expected cost on <i>validation</i> data, with a "
                  f"missed fraud costing {meta['cost_assumptions']['false_negative']:g} and a false alarm "
                  f"{meta['cost_assumptions']['false_positive']:g}. The chosen threshold is <b>{thr:.4f}</b>. On "
                  f"validation it gives precision {vm['precision']:.3f} and recall {vm['recall']:.3f}, with "
                  f"expected cost {vm['expected_cost_per_1000_txn']:.0f} per 1,000 transactions.", S["body"]))
    rows = [["Model (test, days 150-179)", "Role", "Thr.", "PR-AUC", "ROC-AUC", "Recall", "Recall @ P 0.50",
             "F1 macro", "Cost/1k txn", "Brier"]]
    for name, r in test.items():
        rows.append([name.replace("_", " "), r["role"], f"{r['threshold']:.3f}", f"{r['pr_auc']:.4f}",
                     f"{r['roc_auc']:.3f}", f"{r['recall']:.3f}",
                     f"{r['recall_at_precision_0_5']:.3f}" if r["recall_at_precision_0_5"] is not None else "n/a",
                     f"{r['f1_macro']:.3f}", f"{r['expected_cost_per_1000_txn']:.0f}", f"{r['brier']:.4f}"])
    add(Paragraph("<b>Table 3.1 - Test-set results.</b> Each model uses the threshold chosen for it on validation. The "
                  "prior and the rule are shown at their fixed thresholds.", S["small"]))
    add(tbl(rows, [3.1 * cm, 2.0 * cm, 1.1 * cm, 1.5 * cm, 1.5 * cm, 1.4 * cm, 1.9 * cm, 1.4 * cm, 1.6 * cm, 1.5 * cm], S))
    add(Spacer(1, 5))
    prior_row, rule_row = test["Prior (dummy)"], test["Heuristic rule"]
    add(Paragraph(f"<b>Reading the results.</b> The champion reaches test PR-AUC {champ_row['pr_auc']:.3f}, against "
                  f"{prior_row['pr_auc']:.4f} for the prior and {rule_row['pr_auc']:.3f} for the rule. At the cost-optimal "
                  f"threshold it flags {cm_['tp'] + cm_['fp']:,} test transactions. Of those, {cm_['tp']} are fraud "
                  f"({champ_row['precision']:.1%}), and it catches {champ_row['recall']:.1%} of the "
                  f"{cm_['tp'] + cm_['fn']} frauds in the window. Expected cost falls from "
                  f"{prior_row['expected_cost_per_1000_txn']:.0f} to {champ_row['expected_cost_per_1000_txn']:.0f} per 1,000 "
                  f"transactions. Recall at precision 0.50 is {champ_row['recall_at_precision_0_5']:.1%}, which shows that the "
                  f"model is most useful as a ranked list with a cost-driven cut-off, not as a high-precision alarm. "
                  f"Brier scores are higher than the rule's because class weighting inflates probabilities; ranking "
                  f"metrics are unaffected, and calibration is listed as future work.", S["body"]))
    fig_row = [Image(str(REP / "pr_curves_test.png"), width=8.2 * cm, height=6.2 * cm),
               Image(str(REP / "confusion_test.png"), width=6.0 * cm, height=5.4 * cm)]
    ft = Table([fig_row], colWidths=[8.6 * cm, W - 8.6 * cm])
    ft.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
    add(KeepTogether([ft, Paragraph("<b>Figure 3.1</b> - Precision-recall curves on the test set (log scale; left). "
                                    "Confusion matrix of the champion at its cost-optimal threshold (right).", S["small"])]))
    add(Paragraph("<b>3.5 Artifact serialization.</b> <font face='Courier'>artifacts/</font> holds the fitted "
                  "preprocessing pipeline, the champion, the challenger, the logistic baseline, "
                  "<font face='Courier'>reference_feature_stats.json</font> (drift reference, and the 99.9th percentile "
                  "of amount) and <font face='Courier'>model_metadata.json</font> (version, hyper-parameters, threshold "
                  "and its policy, cost assumptions, splits, validation metrics, library versions).", S["body"]))
    # ------------------------------------------------------------ 4
    add(Paragraph("4. Production System Architecture &amp; Inference Implementation", S["h1"]))
    add(Image(str(FIG / "architecture.png"), width=W, height=W * 0.61))
    add(Paragraph("<b>Figure 4.1</b> - Offline training writes artifacts that both serving paths load. Monitoring "
                  "observes both paths and the training reference.", S["small"]))
    add(Spacer(1, 5))
    add(Paragraph("<b>4.1 Online path.</b> <font face='Courier'>serving/app.py</font> provides "
                  "<font face='Courier'>GET /health</font>, <font face='Courier'>POST /predict</font> (one transaction), "
                  "<font face='Courier'>POST /predict_batch</font> (up to 1,000 transactions) and "
                  "<font face='Courier'>GET /metrics</font> (error, fallback, slow-request and decision counters, and "
                  "rolling p95 scoring time). Requests are validated by Pydantic models with "
                  "<font face='Courier'>extra='forbid'</font>, so an unexpected field such as a card number is "
                  "rejected. Wrong types, impossible values and missing required fields return HTTP 422 naming the field. "
                  "Missing history and unseen categories are accepted, and every response lists its warnings.", S["body"]))
    add(Paragraph("<b>4.2 Batch path.</b> <font face='Courier'>serving/predict_batch.py</font> reads a CSV, validates "
                  "every row with the same rules as the API, scores the valid rows in vectorised chunks, and writes "
                  "each transaction ID with its score, flag, decision and status. Invalid rows are written with a "
                  f"reason. In the test run, {bbat['end_to_end_cli']['scored']:,} of "
                  f"{bbat['end_to_end_cli']['input_rows']:,} rows were scored, and {bbat['end_to_end_cli']['flagged']} were "
                  "flagged. The run logs its status to stdout and to <font face='Courier'>logs/batch.log</font>.", S["body"]))
    add(Paragraph("<b>4.3 Packaging.</b> <font face='Courier'>requirements.txt</font> pins every dependency. The "
                  "<font face='Courier'>Makefile</font> provides one-command targets, the README gives curl and CLI examples, "
                  "and a <font face='Courier'>Dockerfile</font> is included. The Dockerfile was not built in this assessment, "
                  "because no container engine was available. The pinned requirements were instead installed into a clean "
                  "virtual environment, and the full test suite passed there, which is the reproducibility evidence used.", S["body"]))
    add(Paragraph("<b>4.4 Real response for the sample request</b> (scored live when this report was built, "
                  "<font face='Courier'>examples/one_request.json</font>):", S["body"]))
    add(tbl([["Field", "Value"]] + [[k, str(v)] for k, v in example.items()], [4.0 * cm, W - 4.0 * cm], S))

    # ------------------------------------------------------------ 5
    add(Paragraph("5. Operational Analysis, Profiling &amp; Drift Monitoring", S["h1"]))
    bc1 = bapi
    rows = [["Concurrent clients", "Requests", "p50 (ms)", "p95 (ms)", "p99 (ms)", "max (ms)", "Throughput (req/s)", "Errors"]]
    for c_, d_ in [(1, bc1), (8, bc8), (16, bc16)]:
        L = d_["latency_ms"]
        rows.append([str(c_), f"{d_['requests']:,}", f"{L['p50']:.2f}", f"{L['p95']:.2f}", f"{L['p99']:.2f}",
                     f"{L['max']:.2f}", f"{d_['throughput_rps']:.1f}", str(d_["errors"])])
    add(Paragraph(f"<b>5.1 Online latency under load.</b> Measured with <font face='Courier'>serving/benchmark.py</font>: "
                  f"{bc1['requests']:,} HTTP requests per run (JSON in and out), after {bc1['warmup_requests']} warm-up "
                  f"requests, at 1, 8 and 16 concurrent clients. Each row is a separate run against the same local uvicorn "
                  f"process. Errors: {sum(d['errors'] for d in (bc1, bc8, bc16))} across all three runs.", S["body"]))
    add(tbl(rows, [2.6 * cm, 1.8 * cm, 1.8 * cm, 1.8 * cm, 1.8 * cm, 1.8 * cm, 2.6 * cm, W - 14.2 * cm], S))
    add(Spacer(1, 4))
    add(Paragraph(f"<b>Interpretation.</b> With one client, p95 is {bc1['latency_ms']['p95']:.1f} ms, inside the 100 ms SLO. "
                  f"Throughput stops rising at about {bc8['throughput_rps']:.0f} requests per second. At 8 clients p95 is "
                  f"{bc8['latency_ms']['p95']:.0f} ms, and at 16 clients it is {bc16['latency_ms']['p95']:.0f} ms, so both are "
                  f"above the SLO. Latency grows with queue length, not with the model, because the scoring itself takes "
                  f"milliseconds. The build machine has one CPU core, which the load generator and the server share. "
                  f"The concurrency figures are therefore pessimistic for production hardware, but they show the saturation "
                  f"point. A load test on the target hardware is required before go-live, and the number of workers and "
                  f"instances should be sized from it.", S["body"]))
    add(Paragraph(f"The per-request budget is 50 ms. Even the sequential run has requests above it (worst "
                  f"{bc1['latency_ms']['max']:.1f} ms), so the service counts them as slow. The budget is a monitored "
                  f"signal, and scoring is never aborted for it.", S["body"]))
    add(Paragraph("<b>5.2 Batch throughput.</b> Vectorised scoring is timed at three batch sizes (median of three repeats). "
                  "An end-to-end CLI run on the full test partition includes CSV reading, validation and writing.",
                  S["body"]))
    rows = [["Batch size", "Median time (s)", "Records / second"]]
    for r in bbat["vectorised_scoring"]:
        rows.append([f"{r['batch_size']:,}", f"{r['median_seconds']:.4f}", f"{r['records_per_second']:,.0f}"])
    e2e = bbat["end_to_end_cli"]
    rows.append([f"{e2e['input_rows']:,} (CLI, end-to-end)", f"{e2e['elapsed_seconds']:.3f}",
                 f"{e2e['records_per_second']:,.0f}"])
    add(tbl(rows, [6.0 * cm, 4.0 * cm, 4.0 * cm], S))
    add(Spacer(1, 4))
    add(Paragraph("<b>5.3 Robustness and edge cases.</b> A fixed suite runs through the real application "
                  "(<font face='Courier'>serving/robustness.py</font>), including injected outages of the model and the rule engine.",
                  S["body"]))
    rrows = [["Scenario", "Expected", "HTTP", "Scorer", "Decision / behaviour"]]
    for sc in rob["scenarios"]:
        if sc["status_code"] == 200:
            beh = sc.get("decision", "")
            if sc.get("warnings"):
                beh += " | " + "; ".join(sc["warnings"])[:105]
        else:
            beh = ("rejected: " + ", ".join(sc.get("validation_errors", []))) if sc.get("validation_errors") else sc.get("note", "")
        rrows.append([sc["scenario"], sc["expected"], str(sc["status_code"]), sc.get("scorer", "-"), beh])
    add(tbl(rrows, [3.6 * cm, 3.6 * cm, 1.3 * cm, 2.6 * cm, W - 11.1 * cm], S))
    add(Spacer(1, 4))
    add(Paragraph("<b>Edge-case findings.</b> Impossible values (negative amounts, malformed timestamps, out-of-range "
                  "flags, extra fields) are rejected before any scoring. Missing history and unseen categories are "
                  "scored with imputation or all-zero encoding and flagged. An amount above the training 99.9th percentile "
                  "is scored, because it is valid, and the warning lets an analyst review it. The batch CLI isolates "
                  "bad rows, while the online batch endpoint is all-or-nothing, which is deliberate: a partial "
                  "authorisation response is harder to reconcile.", S["body"]))

    piv = drift.pivot(index="feature", columns="month", values="psi")
    add(Paragraph("<b>5.4 Feature drift.</b> <font face='Courier'>src/monitoring.py</font> compares each raw input "
                  "with the training reference, using PSI over 10 quantile bins and a KS test. The table shows monthly "
                  "PSI for validation (May) and test (June). PSI of 0.10 or more is WARNING. PSI of 0.25 or more is ALERT. "
                  "A KS result alerts only when the shift is also at least 0.05 in size, because with about 50,000 rows "
                  "per window the significance test flags trivially small shifts.", S["body"]))
    rows = [["Feature (PSI)"] + list(piv.columns)]
    for f in piv.index:
        rows.append([f] + [f"{piv.loc[f, c]:.3f}" for c in piv.columns])
    add(tbl(rows, [5.2 * cm] + [(W - 5.2 * cm) / len(piv.columns)] * len(piv.columns), S))
    add(Spacer(1, 4))
    ks_row = drift[(drift["feature"] == "account_age_days") & (drift["month"] == "2025-06")].iloc[0]
    add(Paragraph(f"<b>Account-age alert, disclosed.</b> On the June test window the KS statistic for account_age_days is "
                  f"{ks_row['ks_stat']:.3f} (p = {ks_row['ks_p']:.2g}), so the rule reports {ks_row['status']}. Its PSI is "
                  f"{ks_row['psi']:.3f}, which is below the WARNING band. The alert is reported, not suppressed. Its cause is "
                  f"structural, since account age rises with calendar time, and Section 5.5 gives the response.", S["body"]))
    add(Paragraph("<b>Interpretation.</b> Transaction amounts, velocity counts and merchant mix are stable. Two features "
                  "show growing PSI, and both are structural rather than data faults. <b>account_age_days</b> rises by "
                  "one day per calendar day, so a fixed training window drifts by construction. <b>amount_mean_30d</b> and "
                  "<b>seconds_since_last_txn</b> are empty for a card's first days, and that share falls as history "
                  "accumulates. A production monitor should therefore compare age and history against the reference "
                  "period, or use the share of populated history, and should not page on these two features alone.",
                  S["body"]))
    add(Paragraph("<b>5.5 Monitoring blueprint.</b> Fraud labels arrive late, because chargebacks can take weeks. The "
                  "plan therefore separates immediate signals from label-delayed ones.", S["body"]))
    add(tbl([
        ["Category", "Metric", "Threshold", "Action"],
        ["Data drift (covariate)", "PSI per raw input vs training reference (daily on scored traffic)",
         "WARNING 0.10; ALERT 0.25, or KS p &lt; 0.01 with KS stat at least 0.05",
         "WARNING: review. ALERT on two consecutive days: review retraining"],
        ["Prediction drift", "Share of step-up decisions per day", "more than 2x the validation flag rate",
         "Check upstream feed and rule changes"],
        ["Data quality", "Missing rate per field; invalid-row share in batch runs", "invalid rows above 1%; missing rate up by 2 points",
         "Block batch publish; notify data owner"],
        ["Concept drift (label-delayed)", "PR-AUC and recall on each labelled week, after chargeback labels mature",
         "PR-AUC down more than 10% vs champion validation value",
         "Trigger retrain; promote only if validation beats champion"],
        ["System (latency)", "p95 and p99 per endpoint, at the measured concurrency; rolling p95 from /metrics", "p95 above 100 ms for 10 minutes",
         "Page on-call; scale replicas"],
        ["Service errors", "5xx rate from /metrics", "above 1% over 5 minutes", "Page on-call; check fallback use"],
        ["Fallback and budget", "Share of predictions with scorer != model; share over the 50 ms budget",
         "fallback above 5%; over-budget above 2%", "Investigate artifact or dependency fault before the next batch"],
    ], [3.0 * cm, 5.0 * cm, 4.2 * cm, W - 12.2 * cm], S))
    add(Paragraph("<b>Retraining policy.</b> Retrain on sustained feature drift (two ALERT days), on a fall in delayed "
                  "PR-AUC, or on a monthly schedule, whichever comes first. The new model uses the same leak-free "
                  "procedure and is promoted only if it beats the champion on validation data. A reviewer approves it.",
                  S["body"]))
    add(Paragraph("<b>5.6 Fallback and reliability.</b> Scoring degrades in three tiers. The scorer name is returned with every "
                  "decision, so downstream systems know which tier answered.", S["body"]))
    add(tbl([
        ["Tier", "When used", "Output", "Decision rule"],
        ["1. model", "default", f"calibrated-ranking probability from the champion", f"flag if probability &gt;= {thr:.4f}; else approve"],
        ["2. fallback_heuristic", "ML scorer raises or is unavailable", "rule score (0.02 to 0.60)",
         f"flag if rule score &gt;= {meta['heuristic_threshold']:.2f}; else approve"],
        ["3. fallback_prior", "ML and rule both fail", "training base rate (constant)",
         "flag_for_review for every transaction (fail-safe)"],
    ], [3.2 * cm, 4.4 * cm, 4.6 * cm, W - 12.2 * cm], S))
    add(Paragraph("Each tier is tested by injecting the failure (<font face='Courier'>tests/test_inference.py</font> and "
                  "<font face='Courier'>serving/robustness.py</font>). A failed tier never produces an HTTP 5xx: the "
                  "service answers with the next tier's result, and a 503 with a clear message is only the final guard for "
                  "unexpected exceptions.", S["body"]))
    add(PageBreak())

    # ------------------------------------------------------------ 6
    add(Paragraph("6. Code Repository &amp; Reproducibility", S["h1"]))
    add(Paragraph(f'<b>Public repository (mandatory):</b> <a href="{repo_url}" color="#1d4ed8"><u>{repo_url}</u></a>', S["body"]))
    add(Paragraph("The repository contains all source code, the reproducible run commands in its README, and the pinned "
                  "dependencies in <font face='Courier'>requirements.txt</font>. The dataset is regenerated "
                  "deterministically (seed 42), so a clean clone reproduces every number in this report.", S["body"]))
    add(tbl([
        ["Step", "Command", "Output"],
        ["Install", "pip install -r requirements.txt", "pinned environment"],
        ["Data", "make data", "data/raw/transactions.csv"],
        ["Train", "make train", "artifacts/ (pipeline, champion, challenger, baseline, metadata, drift reference)"],
        ["Evaluate", "make evaluate", "reports/metrics_test.json, PR and confusion plots, drift tables"],
        ["Test", "make test", f"{n_tests} unit and integration tests"],
        ["Robustness", "make robustness", "reports/robustness.json"],
        ["Serve", "uvicorn serving.app:app --port 8000", "/health, /predict, /predict_batch, /metrics"],
        ["Batch", "python -m serving.predict_batch --input ... --output ...", "scored CSV and run log"],
        ["Profile", "python -m serving.benchmark --mode api|batch", "reports/benchmark_*.json"],
        ["Report", "make report STUDENT_ID=... STUDENT_NAME=... REPO_URL=...", "this PDF"],
    ], [2.4 * cm, 7.4 * cm, W - 9.8 * cm], S))
    add(Spacer(1, 5))
    add(tbl([
        ["Path", "Purpose"],
        ["README.md, requirements.txt, Makefile, Dockerfile, pytest.ini", "documentation, pinned dependencies, one-command workflow"],
        ["data/, artifacts/, reports/", "dataset (regenerated), serialized models, metrics, plots, drift and benchmarks"],
        ["src/", "config, generator, causal features, data pipeline, baselines, training, evaluation, monitoring"],
        ["serving/", "validation schemas, scorer with fallbacks, FastAPI app, batch CLI, benchmark, robustness suite"],
        ["tests/", "leakage and causality, schema, inference and fallback tests"],
        ["docs/", "architecture decisions and GitHub setup guide"],
        ["report/", "builds this PDF from the outputs above"],
    ], [6.4 * cm, W - 6.4 * cm], S))

    # ------------------------------------------------------------ 7
    add(Paragraph("7. Conclusions, Limitations and Future Work", S["h1"]))
    add(Paragraph("The system takes a fraud problem from formulation to a monitored service. Causal history features, a "
                  "temporal split and validation-only decisions keep the test estimate honest. The model's ranking clearly "
                  "beats the prior and the rule, and the cost-based threshold reduces expected loss. Both serving paths "
                  "share one validation contract, latency is measured with percentiles, and each failure degrades to a "
                  "documented fallback instead of an error.", S["body"]))
    add(Paragraph("<b>Limitations.</b> (i) The dataset is synthetic, so absolute values will differ on real fraud data; "
                  "real data usually has stronger and more varied signals. (ii) The costs (150 per missed fraud, 15 per "
                  "false alarm) are assumptions, and the threshold depends on them. (iii) Latency was measured on one "
                  "build machine with a single worker. (iv) Class weighting distorts probability calibration, so the "
                  "threshold is set on the cost of decisions, not on calibrated probabilities. (v) Velocity features in "
                  "the online request are supplied by an upstream service that this project only simulates. (vi) The "
                  "rule fallback is a heuristic, not a calibrated model. (vii) The build machine has one CPU core shared by the "
                  "load generator and the server, so the concurrency results show the saturation point but are not a "
                  "substitute for a load test on target hardware.", S["body"]))
    add(Paragraph("<b>Future work.</b> Calibrate the champion (Platt or isotonic scaling) and recompute the cost threshold; "
                  "add a real feature store for the history features; run a champion-challenger shadow deployment; add "
                  "authentication and rate limiting; and schedule drift and delayed-label jobs that write to a dashboard.",
                  S["body"]))

    def _footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(colors.HexColor("#555"))
        canvas.drawString(2 * cm, 1.2 * cm, f"MLME Assignment - {student_name} ({student_id}) - Option B: Fraud Detection")
        canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"Page {doc.page}")
        canvas.restoreState()

    doc = SimpleDocTemplate(str(out_path), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
                            topMargin=1.8 * cm, bottomMargin=2 * cm,
                            title="Card Fraud Detection - Mini Production ML System", author=student_name,
                            subject="MLME Course Assignment")
    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
    return out_path


def main(argv=None):
    p = argparse.ArgumentParser(description="Build the assignment PDF report")
    p.add_argument("--student-id", default="2025em1200044")
    p.add_argument("--student-name", default="Aditya Jha")
    p.add_argument("--repo-url", default="https://github.com/AdityaJha2/Mini_Production_ML_System")
    args = p.parse_args(argv)
    out = REP / f"MLME_Assignment_{args.student_id}_{'_'.join(args.student_name.split())}.pdf"
    build(args.student_id, args.student_name, args.repo_url, out)
    print(f"Report written: {out}")


if __name__ == "__main__":
    main()
