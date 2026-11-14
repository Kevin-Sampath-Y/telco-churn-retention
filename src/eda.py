"""
Exploratory analysis: where does churn concentrate, and is each pattern statistically real?

Chart titles are generated from the data (action titles) so they stay correct on any re-run.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
from scipy import stats

from src.config import CATEGORICAL, PALETTE as C, TARGET
from src.evaluate import save_fig  # also applies the shared plot style


def cramers_v(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    """Association between a category and churn: (Cramer's V in 0..1, chi-square p-value)."""
    table = pd.crosstab(x, y)
    chi2, p, _, _ = stats.chi2_contingency(table)
    n = table.to_numpy().sum()
    v = np.sqrt(chi2 / (n * (min(table.shape) - 1)))
    return float(v), float(p)


def categorical_association(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in CATEGORICAL:
        v, p = cramers_v(df[c], df[TARGET])
        rates = df.groupby(c)[TARGET].mean()
        rows.append({"feature": c, "cramers_v": round(v, 3), "p_value": p,
                     "highest_churn_group": rates.idxmax(), "highest_rate": round(float(rates.max()), 3),
                     "lowest_churn_group": rates.idxmin(), "lowest_rate": round(float(rates.min()), 3)})
    return pd.DataFrame(rows).sort_values("cramers_v", ascending=False).reset_index(drop=True)


def numeric_tests(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in ["tenure", "MonthlyCharges", "TotalCharges"]:
        a, b = df.loc[df[TARGET] == 1, c], df.loc[df[TARGET] == 0, c]
        u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        rows.append({"feature": c, "median_churned": round(float(a.median()), 1), "median_stayed": round(float(b.median()), 1),
                     "rank_biserial": round(float(2 * u / (len(a) * len(b)) - 1), 3), "p_value": float(p)})
    return pd.DataFrame(rows)


def churn_rate_by(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Churn rate with a 95% Wilson confidence interval per group."""
    g = df.groupby(col)[TARGET].agg(["sum", "count"])
    lo, hi = [], []
    for s, n in zip(g["sum"], g["count"]):
        ci = stats.binomtest(int(s), int(n)).proportion_ci(confidence_level=0.95, method="wilson")
        lo.append(ci.low)
        hi.append(ci.high)
    g["rate"] = g["sum"] / g["count"]
    g["ci_low"], g["ci_high"] = lo, hi
    return g.reset_index().rename(columns={col: "group"})


def _bar_with_ci(ax, t: pd.DataFrame, color=None):
    t = t.sort_values("rate")
    ax.barh(t["group"].astype(str), t["rate"], color=color or C["primary"],
            xerr=[t["rate"] - t["ci_low"], t["ci_high"] - t["rate"]], error_kw={"ecolor": C["neutral"], "capsize": 3})
    for i, (r, n, hi) in enumerate(zip(t["rate"], t["count"], t["ci_high"])):
        ax.text(hi + 0.015, i, f"{r:.0%}  (n={n:,})", va="center", fontsize=8.5)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.set_xlim(0, 0.95)
    ax.grid(axis="y", visible=False)


def make_plots(df: pd.DataFrame) -> list[str]:
    paths, base = [], df[TARGET].mean()

    # 1. contract + internet + payment
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.9), gridspec_kw={"wspace": 0.75})
    for ax, col, title in zip(axes, ["Contract", "InternetService", "PaymentMethod"],
                              ["Contract type", "Internet service", "Payment method"]):
        t = churn_rate_by(df, col)
        _bar_with_ci(ax, t, C["primary"])
        ax.axvline(base, ls="--", color=C["accent"], lw=1.2)
        ax.set_title(title)
    m2m = churn_rate_by(df, "Contract").set_index("group").loc["Month-to-month", "rate"]
    two = churn_rate_by(df, "Contract").set_index("group").loc["Two year", "rate"]
    fig.suptitle(f"Month-to-month customers churn at {m2m:.0%}, two-year customers at {two:.0%}",
                 x=0.01, ha="left", fontsize=13, fontweight="bold", y=1.04)
    paths.append(save_fig(fig, "01_churn_by_segment.png",
                          f"Bars show churn rate with 95% confidence intervals; dashed line = overall churn {base:.0%}."))

    # 2. tenure
    d = df.copy()
    d["tenure_band"] = pd.cut(d["tenure"], [-1, 6, 12, 24, 48, 72], labels=["0-6", "7-12", "13-24", "25-48", "49-72"])
    t = churn_rate_by(d, "tenure_band")
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    ax.bar(t["group"].astype(str), t["rate"], color=[C["bad"] if i == 0 else C["light"] for i in range(len(t))], width=0.65,
           yerr=[t["rate"] - t["ci_low"], t["ci_high"] - t["rate"]], error_kw={"ecolor": C["neutral"], "capsize": 3})
    for i, r in enumerate(t["rate"]):
        ax.text(i, r + 0.04, f"{r:.0%}", ha="center", fontweight="bold")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.set_xlabel("Months as a customer")
    ax.set_ylabel("Churn rate")
    ax.grid(axis="x", visible=False)
    ax.set_title(f"Risk is front-loaded: {t.iloc[0]['rate']:.0%} of customers in their first 6 months churn vs {t.iloc[-1]['rate']:.0%} after 4 years")
    paths.append(save_fig(fig, "02_churn_by_tenure.png", "Tenure bands in months; error bars = 95% confidence interval."))

    # 3. charges
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    for lab, col in [(0, C["light"]), (1, C["bad"])]:
        ax.hist(df.loc[df[TARGET] == lab, "MonthlyCharges"], bins=30, alpha=0.75, color=col, density=True,
                label="Churned" if lab else "Stayed")
    ax.set_xlabel("Monthly charges ($)")
    ax.set_ylabel("Density")
    med1, med0 = df.loc[df[TARGET] == 1, "MonthlyCharges"].median(), df.loc[df[TARGET] == 0, "MonthlyCharges"].median()
    ax.set_title(f"Churners pay more: median \\${med1:.0f}/month vs \\${med0:.0f} for customers who stay")
    ax.legend(frameon=False)
    paths.append(save_fig(fig, "03_monthly_charges.png", "Normalised histograms (each group sums to 1)."))

    # 4. strength of association
    a = categorical_association(df).head(10).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.barh(a["feature"], a["cramers_v"], color=C["primary"])
    ax.set_xlabel("Association with churn (Cramer's V, 0 = none, 1 = perfect)")
    ax.set_title(f"'{a.iloc[-1]['feature']}' has the strongest link to churn of all categorical features")
    ax.grid(axis="y", visible=False)
    paths.append(save_fig(fig, "04_feature_association.png", "All shown associations have chi-square p < 0.001 unless noted in reports/eda_tests.csv."))
    return paths
