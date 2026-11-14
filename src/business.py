"""
From probabilities to money: who should the retention team call, and is it worth it?

A churn probability alone does not tell you what to do. The decision depends on
  * what the customer is worth  (monthly charge x margin x months we expect to keep them), and
  * what an intervention costs and how often it works.

IMPORTANT: the three assumptions below (margin, offer cost, save rate) are NOT in the dataset.
They are explicit, adjustable inputs. The save rate in particular should come from an A/B test
before anyone trusts the money figures; the ranking of customers does not depend on it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Assumptions:
    horizon_months: int = 12      # value of a retained customer = this many months of margin
    gross_margin: float = 0.50    # share of revenue that is profit
    offer_cost: float = 30.0      # cost of contacting + incentivising ONE targeted customer ($)
    save_rate: float = 0.30       # share of would-be churners the offer actually keeps


def customer_value(monthly_charges, a: Assumptions):
    """Profit we keep if this customer is retained for the horizon."""
    return np.asarray(monthly_charges, dtype=float) * a.gross_margin * a.horizon_months


def expected_benefit(prob, monthly_charges, a: Assumptions):
    """Expected net profit of targeting a customer: P(churn) x P(save) x value - cost."""
    return np.asarray(prob, dtype=float) * a.save_rate * customer_value(monthly_charges, a) - a.offer_cost


def realized_benefit(churned, monthly_charges, a: Assumptions):
    """Back-test version: uses the customer's ACTUAL outcome instead of the predicted probability."""
    return np.asarray(churned, dtype=float) * a.save_rate * customer_value(monthly_charges, a) - a.offer_cost


def break_even_probability(monthly_charges, a: Assumptions):
    """Churn probability above which targeting a customer pays off (cost / (save_rate x value))."""
    return a.offer_cost / (a.save_rate * customer_value(monthly_charges, a))


def campaign_curve(scored: pd.DataFrame, a: Assumptions, prob_col: str = "churn_probability") -> pd.DataFrame:
    """Total back-tested profit if we contact the top-k customers, ranked by expected benefit."""
    d = scored.copy()
    d["exp_benefit"] = expected_benefit(d[prob_col], d["MonthlyCharges"], a)
    d["real_benefit"] = realized_benefit(d["Churn"], d["MonthlyCharges"], a)
    d = d.sort_values("exp_benefit", ascending=False).reset_index(drop=True)
    d["k"] = np.arange(1, len(d) + 1)
    d["cum_profit"] = d["real_benefit"].cumsum()
    return d


def strategy_comparison(scored: pd.DataFrame, a: Assumptions, prob_col: str = "churn_probability") -> pd.DataFrame:
    """Back-test four ways of spending the retention budget on the same customers."""
    n = len(scored)
    real = realized_benefit(scored["Churn"], scored["MonthlyCharges"], a)
    exp = expected_benefit(scored[prob_col], scored["MonthlyCharges"], a)
    model_mask = exp > 0
    k = int(model_mask.sum())
    by_prob = scored[prob_col].rank(ascending=False, method="first") <= k       # same k, ranked by probability only
    rows = [
        ("Do nothing", 0, 0.0),
        ("Contact everyone", n, float(real.sum())),
        ("Contact a random group of the same size", k, float(real.mean() * k)),
        ("Contact top by churn probability (same size)", k, float(real[by_prob].sum())),
        ("Model: contact only where expected benefit > 0", k, float(real[model_mask].sum())),
    ]
    return pd.DataFrame(rows, columns=["strategy", "customers_contacted", "net_profit"])


def best_probability_threshold(prob, churned, monthly, a: Assumptions) -> float:
    """Single probability cut-off that maximises back-tested profit (used as the headline operating point)."""
    prob, churned, monthly = map(np.asarray, (prob, churned, monthly))
    real = realized_benefit(churned, monthly, a)
    grid = np.linspace(0.05, 0.95, 91)
    profits = [real[prob >= t].sum() for t in grid]
    return float(grid[int(np.argmax(profits))])


# ---------------------------------------------------------------- rule-based next best action
_ACTIONS = [
    (lambda r: r["Contract"] == "Month-to-month",
     "Offer a discounted 1-year contract: month-to-month customers churn far more than contracted ones."),
    (lambda r: r["InternetService"] == "Fiber optic" and r["TechSupport"] != "Yes",
     "Offer free tech support for 3 months: fibre customers without it churn most."),
    (lambda r: r["PaymentMethod"] == "Electronic check",
     "Nudge to automatic payment (small bill credit): electronic-check payers churn more."),
    (lambda r: r["tenure"] <= 6,
     "Run an onboarding call in the first 6 months: early tenure is the riskiest period."),
    (lambda r: r["n_addons"] == 0 and r["InternetService"] != "No",
     "Bundle a security or backup add-on: customers with no add-ons are less attached."),
]


def recommend_actions(row: pd.Series, limit: int = 3) -> list[str]:
    out = [text for cond, text in _ACTIONS if cond(row)]
    return out[:limit] or ["No specific risk pattern: keep the customer on the standard loyalty programme."]
