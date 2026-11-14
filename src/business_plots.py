"""Charts for the retention-campaign back-test."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np

from src import business
from src.config import PALETTE as C
from src.evaluate import save_fig


def make_plots(test_scored, a: business.Assumptions) -> list[str]:
    paths = []
    curve = business.campaign_curve(test_scored, a)
    k_star = int(curve["cum_profit"].idxmax()) + 1
    peak = float(curve["cum_profit"].max())
    all_profit = float(curve["real_benefit"].sum())
    rng = np.random.default_rng(42)
    rand = np.cumsum(rng.permutation(curve["real_benefit"].to_numpy()))

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    ax.plot(curve["k"], curve["cum_profit"], color=C["primary"], lw=2.6, label="Contact riskiest-value customers first (model)")
    ax.plot(curve["k"], rand, color=C["neutral"], lw=1.6, ls="--", label="Contact customers in random order")
    ax.axhline(0, color=C["ink"], lw=0.8)
    ax.scatter([k_star], [peak], color=C["accent"], s=60, zorder=5)
    ax.annotate(f"Best: contact {k_star:,} customers\nnet profit \\${peak:,.0f}", (k_star, peak), xytext=(15, -45),
                textcoords="offset points", fontsize=9.5, arrowprops=dict(arrowstyle="-", color=C["neutral"]))
    ax.yaxis.set_major_formatter(mtick.StrMethodFormatter("${x:,.0f}"))
    ax.set_xlabel("Customers contacted (ranked by expected benefit)")
    ax.set_ylabel("Back-tested net profit")
    ax.set_title(f"Targeting by model earns \\${peak:,.0f}; contacting everyone earns \\${all_profit:,.0f}")
    ax.legend(frameon=False, loc="lower right")
    paths.append(save_fig(fig, "13_campaign_profit.png",
                          f"Hold-out customers. Assumptions: {a.horizon_months}-month horizon, {a.gross_margin:.0%} margin, "
                          f"\\${a.offer_cost:.0f}/contact, {a.save_rate:.0%} save rate (adjustable; not in the data)."))
    return paths
