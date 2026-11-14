"""Model evaluation on the held-out TEST set: metrics, plots and a simple fairness check."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss, confusion_matrix,
                             f1_score, precision_recall_curve, precision_score, recall_score, roc_auc_score,
                             roc_curve)

from src.config import FIGURES_DIR, PALETTE as C

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#B8C0CC", "axes.labelcolor": C["ink"], "xtick.color": C["ink"], "ytick.color": C["ink"],
    "font.size": 10.5, "axes.titlesize": 12.5, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.grid": True, "grid.color": "#E6EAF0", "grid.linewidth": 0.8, "axes.axisbelow": True,
    "figure.dpi": 110, "savefig.dpi": 170, "savefig.bbox": "tight",
})


def save_fig(fig, name: str, footer: str | None = None) -> str:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    if footer:
        fig.text(0.01, -0.02, footer, fontsize=8.5, color=C["neutral"], ha="left", va="top")
    path = FIGURES_DIR / name
    fig.savefig(path)
    plt.close(fig)
    return str(path)


def metrics_at(y, p, threshold: float) -> dict:
    pred = (p >= threshold).astype(int)
    return {
        "threshold": round(float(threshold), 3),
        "roc_auc": round(float(roc_auc_score(y, p)), 4),
        "pr_auc": round(float(average_precision_score(y, p)), 4),
        "brier": round(float(brier_score_loss(y, p)), 4),
        "accuracy": round(float(accuracy_score(y, pred)), 4),
        "precision": round(float(precision_score(y, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y, pred)), 4),
        "f1": round(float(f1_score(y, pred)), 4),
        "confusion": confusion_matrix(y, pred).tolist(),
    }


def decile_table(y, p) -> pd.DataFrame:
    """Customers sorted from riskiest to safest, cut into 10 equal groups."""
    d = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p)}).sort_values("p", ascending=False).reset_index(drop=True)
    d["decile"] = pd.qcut(d.index, 10, labels=range(1, 11)).astype(int)
    t = d.groupby("decile").agg(customers=("y", "size"), churners=("y", "sum"), churn_rate=("y", "mean"), avg_prob=("p", "mean"))
    t["cum_capture"] = t["churners"].cumsum() / t["churners"].sum()
    t["lift"] = t["churn_rate"] / d["y"].mean()
    return t.reset_index()


def group_metrics(df_test: pd.DataFrame, y, p, threshold: float, column: str) -> pd.DataFrame:
    """Same metrics per group (fairness / robustness check)."""
    rows = []
    for g, idx in df_test.groupby(column).groups.items():
        m = df_test.index.get_indexer(idx)
        yy, pp = np.asarray(y)[m], np.asarray(p)[m]
        rows.append({"group": f"{column} = {g}", "n": len(m), "churn_rate": round(float(yy.mean()), 3),
                     "roc_auc": round(float(roc_auc_score(yy, pp)), 3) if len(set(yy)) > 1 else None,
                     "recall": round(float(recall_score(yy, pp >= threshold)), 3),
                     "flagged_share": round(float((pp >= threshold).mean()), 3)})
    return pd.DataFrame(rows)


def make_plots(y, p, threshold: float, model_name: str) -> list[str]:
    paths = []
    auc = roc_auc_score(y, p)
    ap = average_precision_score(y, p)
    base = float(np.mean(y))

    # 1. ROC + PR
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.4))
    fpr, tpr, _ = roc_curve(y, p)
    a1.plot(fpr, tpr, color=C["primary"], lw=2.4)
    a1.plot([0, 1], [0, 1], ls="--", color=C["neutral"], lw=1)
    a1.set_xlabel("False positive rate")
    a1.set_ylabel("True positive rate")
    a1.set_title(f"ROC-AUC {auc:.2f}: ranks churners above stayers")
    pr, rc, _ = precision_recall_curve(y, p)
    a2.plot(rc, pr, color=C["accent"], lw=2.4)
    a2.axhline(base, ls="--", color=C["neutral"], lw=1)
    a2.text(0.02, base + 0.015, f"base rate {base:.0%}", color=C["neutral"], fontsize=9)
    a2.set_xlabel("Recall")
    a2.set_ylabel("Precision")
    a2.set_title(f"PR-AUC {ap:.2f} vs {base:.2f} for guessing")
    paths.append(save_fig(fig, "07_roc_pr.png", f"{model_name}, evaluated once on the 20% hold-out set."))

    # 2. Calibration
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    frac, mean_p = calibration_curve(y, p, n_bins=10, strategy="quantile")
    ax.plot([0, 1], [0, 1], ls="--", color=C["neutral"], lw=1, label="perfect")
    ax.plot(mean_p, frac, marker="o", color=C["primary"], lw=2.2, label=model_name)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.set_xlabel("Predicted churn probability")
    ax.set_ylabel("Actual churn rate")
    ax.set_title("Predicted probabilities can be read as risks")
    ax.legend(frameon=False)
    paths.append(save_fig(fig, "08_calibration.png", "Customers grouped into 10 equal bins by predicted probability."))

    # 3. Gain / lift by decile
    t = decile_table(y, p)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.4))
    a1.bar(t["decile"], t["churn_rate"], color=[C["bad"] if d <= 2 else C["light"] for d in t["decile"]], width=0.7)
    a1.axhline(base, ls="--", color=C["neutral"], lw=1)
    a1.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    a1.set_xticks(range(1, 11))
    a1.set_xlabel("Risk decile (1 = riskiest 10% of customers)")
    a1.set_ylabel("Actual churn rate")
    a1.set_title(f"Riskiest decile churns at {t.iloc[0]['churn_rate']:.0%} ({t.iloc[0]['lift']:.1f}x average)")
    a2.plot(np.r_[0, t["decile"] * 10], np.r_[0, t["cum_capture"] * 100], marker="o", color=C["primary"], lw=2.4)
    a2.plot([0, 100], [0, 100], ls="--", color=C["neutral"], lw=1)
    a2.xaxis.set_major_formatter(mtick.PercentFormatter())
    a2.yaxis.set_major_formatter(mtick.PercentFormatter())
    a2.set_xlabel("Customers contacted, riskiest first")
    a2.set_ylabel("Churners reached")
    a2.set_title(f"Contacting the top 30% reaches {t.iloc[2]['cum_capture']:.0%} of churners")
    paths.append(save_fig(fig, "09_lift_gain.png", "Hold-out set. Dashed line = contacting customers at random."))

    # 4. Confusion matrix
    cm = confusion_matrix(y, (p >= threshold).astype(int))
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ax.imshow(cm, cmap="Blues")
    ax.grid(False)
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, f"{v}", ha="center", va="center", fontsize=14, fontweight="bold",
                color="white" if v > cm.max() / 2 else C["ink"])
    ax.set_xticks([0, 1], ["Stays", "Churns"])
    ax.set_yticks([0, 1], ["Stayed", "Churned"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    rec = cm[1, 1] / cm[1].sum()
    ax.set_title(f"At threshold {threshold:.2f}: {rec:.0%} of churners caught")
    paths.append(save_fig(fig, "10_confusion.png", "Threshold chosen to maximise back-tested campaign profit (see business.py)."))
    return paths
