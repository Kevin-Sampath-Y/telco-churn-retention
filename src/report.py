"""Write reports/metrics.json + reports/summary.md and refresh the README results block."""
from __future__ import annotations

import json
import re

import pandas as pd

from src.config import REPORTS_DIR, ROOT


def build_text(meta: dict, m: dict, deciles: pd.DataFrame, strat: pd.DataFrame, a, drivers: pd.DataFrame,
               assoc: pd.DataFrame, synthetic: bool) -> str:
    banner = ("> **Demo data.** Figures below come from a *synthetic* dataset with the Telco schema. "
              "Run `python -m src.download_data` then `python run_pipeline.py` for real results.\n\n"
              if synthetic else
              "> Figures below were computed on the public **IBM Telco Customer Churn** dataset (7,043 customers).\n\n")
    comp = pd.DataFrame(meta["comparison"])
    best = strat.set_index("strategy")
    model_profit = best.loc["Model: contact only where expected benefit > 0", "net_profit"]
    all_profit = best.loc["Contact everyone", "net_profit"]
    rand_profit = best.loc["Contact a random group of the same size", "net_profit"]
    n_model = int(best.loc["Model: contact only where expected benefit > 0", "customers_contacted"])
    top = deciles.iloc[0]
    top3 = deciles.iloc[2]

    lines = [banner, "### Model comparison (5-fold CV on training data, then hold-out test)\n",
             "| Model | CV ROC-AUC | Test ROC-AUC | Test PR-AUC | Test Brier |", "|---|---|---|---|---|"]
    for _, r in comp.iterrows():
        mark = " **(selected)**" if r["model"] == meta["selected_model"] else ""
        lines.append(f"| {r['model']}{mark} | {r['cv_roc_auc']:.3f} | {r['test_roc_auc']:.3f} | {r['test_pr_auc']:.3f} | {r['test_brier']:.3f} |")
    lines += ["", "### Key findings\n",
              f"1. **The model ranks churners well.** Hold-out ROC-AUC {m['roc_auc']:.2f} and PR-AUC {m['pr_auc']:.2f} "
              f"(guessing gives {meta['base_rate']:.2f}). Probabilities are well calibrated (Brier {m['brier']:.3f}).",
              f"2. **A small group holds most of the risk.** The riskiest 10% of customers churn at {top['churn_rate']:.0%} "
              f"({top['lift']:.1f}x the average); contacting the riskiest 30% reaches {top3['cum_capture']:.0%} of churners.",
              f"3. **Top drivers.** " + ", ".join(f"*{r['feature']}*" for _, r in drivers.head(3).iterrows()) +
              " cause the biggest drop in accuracy when shuffled. This is association, not proof of causation.",
              f"4. **Segments.** Churn is highest for *{assoc.iloc[0]['feature']} = {assoc.iloc[0]['highest_churn_group']}* "
              f"({assoc.iloc[0]['highest_rate']:.0%}) and lowest for *{assoc.iloc[0]['lowest_churn_group']}* ({assoc.iloc[0]['lowest_rate']:.0%}).",
              f"5. **Business impact (back-test, assumptions below).** Targeting the {n_model:,} customers whose expected benefit is positive "
              f"gives ${model_profit:,.0f} net profit vs ${all_profit:,.0f} for contacting everyone and ${rand_profit:,.0f} for a same-size random group.",
              "", f"*Assumptions: {a.horizon_months}-month horizon, {a.gross_margin:.0%} margin, ${a.offer_cost:.0f} per contact, "
              f"{a.save_rate:.0%} of would-be churners saved. These are NOT in the data; validate the save rate with an A/B test.*",
              "", "### Decision at the operating threshold\n",
              f"| Threshold | Precision | Recall | F1 |", "|---|---|---|---|",
              f"| {m['threshold']:.2f} | {m['precision']:.2f} | {m['recall']:.2f} | {m['f1']:.2f} |", ""]
    return "\n".join(lines) + "\n"


def write(text: str, metrics: dict) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "summary.md").write_text(text, encoding="utf-8")
    (REPORTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float), encoding="utf-8")
    readme = ROOT / "README.md"
    if readme.exists():
        body = readme.read_text(encoding="utf-8")
        new = re.sub(r"(<!-- RESULTS:START -->).*?(<!-- RESULTS:END -->)",
                     lambda mo: f"{mo.group(1)}\n{text}\n{mo.group(2)}", body, flags=re.S)
        readme.write_text(new, encoding="utf-8")
