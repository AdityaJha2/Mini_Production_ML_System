"""Architecture diagram for the report, drawn programmatically."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def _box(ax, x, y, w, h, text, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.02", fc=color, ec="#333", lw=1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=7.6)


def _arrow(ax, x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=10, lw=0.9, color="#333"))


def architecture(path):
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.25, 1.0)
    ax.axis("off")
    blue, green, orange, grey = "#dbeafe", "#dcfce7", "#ffedd5", "#f3f4f6"

    ax.text(0.01, 0.94, "OFFLINE: training and release", fontsize=8.5, weight="bold")
    _box(ax, 0.01, 0.74, 0.17, 0.14, "Transactions\ncsv (300k rows)", grey)
    _box(ax, 0.22, 0.74, 0.17, 0.14, "Causal velocity\nfeatures + split", blue)
    _box(ax, 0.43, 0.74, 0.17, 0.14, "Preprocessor\n(fit on train)", blue)
    _box(ax, 0.64, 0.74, 0.17, 0.14, "Train, select\nchampion, threshold", blue)
    _box(ax, 0.85, 0.74, 0.14, 0.14, "artifacts/\njoblib + JSON", green)
    for x1, x2 in [(0.18, 0.22), (0.39, 0.43), (0.60, 0.64), (0.81, 0.85)]:
        _arrow(ax, x1, 0.81, x2, 0.81)

    ax.text(0.01, 0.56, "ONLINE: authorisation request", fontsize=8.5, weight="bold")
    _box(ax, 0.01, 0.36, 0.16, 0.12, "Payment system\nJSON + history", grey)
    _box(ax, 0.22, 0.36, 0.20, 0.12, "FastAPI + Pydantic\n422 on bad input", orange)
    _box(ax, 0.47, 0.36, 0.22, 0.12, "FraudScorer\nmodel, rule, fail-safe", orange)
    _box(ax, 0.74, 0.36, 0.25, 0.12, "Decision: approve,\nstep-up, review", grey)
    for x1, x2 in [(0.17, 0.22), (0.42, 0.47), (0.69, 0.74)]:
        _arrow(ax, x1, 0.42, x2, 0.42)

    ax.text(0.01, 0.27, "BATCH: scheduled scoring", fontsize=8.5, weight="bold")
    _box(ax, 0.01, 0.07, 0.16, 0.12, "Input CSV\n(unseen data)", grey)
    _box(ax, 0.22, 0.07, 0.20, 0.12, "Row validation\ninvalid -> status", orange)
    _box(ax, 0.47, 0.07, 0.22, 0.12, "Vectorised scoring\n(chunked)", orange)
    _box(ax, 0.74, 0.07, 0.25, 0.12, "Output CSV + run log\n(records/sec)", grey)
    for x1, x2 in [(0.17, 0.22), (0.42, 0.47), (0.69, 0.74)]:
        _arrow(ax, x1, 0.13, x2, 0.13)

    ax.add_patch(FancyArrowPatch((0.925, 0.74), (0.58, 0.485), arrowstyle="-|>", mutation_scale=9,
                                 lw=0.8, ls="--", color="#2563eb"))
    ax.text(0.80, 0.585, "scorer loads artifacts at startup", fontsize=6.5, color="#2563eb", ha="center")
    ax.text(0.58, 0.245, "batch CLI loads the same artifacts", fontsize=6.5, color="#2563eb", ha="center")

    _box(ax, 0.01, -0.22, 0.98, 0.16,
         "MONITORING\n"
         "PSI and KS drift vs training reference   |   p95 latency, error and fallback rates\n"
         "label-delayed PR-AUC (concept drift)   ->   alert or retrain", "#fef9c3")
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
