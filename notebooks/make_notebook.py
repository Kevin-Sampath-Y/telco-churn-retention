"""Builds notebooks/01_churn_analysis.ipynb (run from the project root: python notebooks/make_notebook.py)."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = [
md("""# Telco customer churn: analysis walkthrough

**Question:** which customers are about to leave, why, and is it worth spending money to keep them?

This notebook explains the reasoning. The reusable code lives in `src/`; run `python run_pipeline.py` to reproduce everything.

1. Data and cleaning  2. Exploratory analysis  3. Modelling  4. Evaluation  5. Explainability  6. Business value  7. Limitations"""),
code("""import sys, warnings
sys.path.insert(0, "..")
warnings.filterwarnings("ignore")
import joblib, json
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Image, display
from src import config, data_prep, eda, evaluate, business, train
raw = data_prep.load_raw(config.RAW_FILE)
raw.head()"""),
md("## 1. Data and cleaning\nOne row per customer; `Churn` is the outcome. Two things to check before anything else: duplicates and the text column `TotalCharges`."),
code("""print(raw.shape, "| duplicate ids:", raw.customerID.duplicated().sum())
blank = pd.to_numeric(raw.TotalCharges, errors="coerce").isna()
print("Blank TotalCharges rows:", blank.sum(), "| their tenure values:", raw.loc[blank, "tenure"].unique())"""),
md("All 11 blank rows have `tenure == 0`: brand-new customers who have not been billed yet. The honest value is **0**, not the average, so `clean()` sets 0 and raises an error if a blank ever appears for a customer with tenure above 0."),
code("""df = data_prep.prepare()
print(df.shape, "churn rate: %.1f%%" % (100 * df[config.TARGET].mean()))
df[["tenure", "MonthlyCharges", "TotalCharges", "n_addons", "avg_monthly_charge", "price_change", "autopay"]].describe(include="all").T.head(8)"""),
md("## 2. Exploratory analysis\nChurn is **26.5%** overall, so a model that always predicts staying is 73.5% accurate and useless. That is why we use ROC-AUC and PR-AUC instead of accuracy."),
code("""eda.categorical_association(df).head(8)"""),
code("""display(Image("../reports/figures/01_churn_by_segment.png"))
display(Image("../reports/figures/02_churn_by_tenure.png"))"""),
md("Contract type, tenure and fibre internet dominate. The numeric tests confirm churners pay more per month and have much shorter tenure."),
code("""eda.numeric_tests(df)"""),
md("## 3. Modelling\nA fair comparison needs a baseline and a protocol that never lets the test set influence a choice. Here is the minimal version (no tuning) before the full tuned run in `src/train.py`."),
code("""from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from src.features import build_preprocessor
X_tr, X_te, y_tr, y_te = train.split(df)
cv = StratifiedKFold(5, shuffle=True, random_state=config.RANDOM_STATE)
for name, m in [("Logistic regression", LogisticRegression(max_iter=2000)),
                ("XGBoost (defaults)", XGBClassifier(eval_metric="logloss", n_jobs=1, random_state=42))]:
    s = cross_val_score(Pipeline([("prep", build_preprocessor()), ("model", m)]), X_tr, y_tr, cv=cv, scoring="roc_auc")
    print(f"{name:<22} CV ROC-AUC {s.mean():.3f} +/- {s.std():.3f}")"""),
md("A plain logistic regression is already within about 0.01 AUC of tuned boosting. **That is a real finding:** the signal is mostly simple (contract, tenure, services), so a simple model is a legitimate choice if explainability matters more than the last percent. The pipeline still tunes all candidates and picks by cross-validation only."),
code("""meta = json.loads(config.META_FILE.read_text())
pd.DataFrame(meta["comparison"]).drop(columns="best_params")"""),
md("## 4. Evaluation on the untouched test set"),
code("""model = joblib.load(config.MODEL_FILE)
p = model.predict_proba(X_te)[:, 1]
m = evaluate.metrics_at(y_te, p, meta["operating_threshold"])
print({k: v for k, v in m.items() if k != "confusion"})
display(Image("../reports/figures/09_lift_gain.png"))"""),
code("""display(Image("../reports/figures/08_calibration.png"))"""),
md("The probabilities are close to the diagonal, so a prediction of 40% really means roughly 4 in 10 such customers leave. That matters because the business calculation below multiplies these probabilities by money."),
md("## 5. Explainability"),
code("""display(Image("../reports/figures/12_shap_summary.png"))"""),
md("Month-to-month contracts, short tenure, fibre internet, no online security or tech support and electronic-check payment push risk **up**. These are associations in observational data, not proven causes."),
md("## 6. From prediction to decision\nA churn probability is not an action. The value of contacting a customer is `P(churn) x save rate x customer value - cost`. The margin, cost and save rate are **assumptions** (not in the data); the app lets you change them."),
code("""a = business.Assumptions()
scored = pd.read_csv(config.SCORED_FILE)
test_scored = scored[scored.split == "test"]
business.strategy_comparison(test_scored, a)"""),
code("""display(Image("../reports/figures/13_campaign_profit.png"))"""),
md("""## 7. Limitations and next steps
* One snapshot of one telecom sample: no time dimension, so we predict *who is at risk*, not *when*.
* The model finds who will leave, not who can be **persuaded** to stay. An **uplift model** or an **A/B test** is the right next step, and it is how the 30% save rate should be measured.
* `gender` is excluded on purpose; group metrics (`reports/fairness_check.csv`) should be monitored before any real use.
* Next project step: a time-based split and survival analysis (time to churn)."""),
]
nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, "notebooks/01_churn_analysis.ipynb")
print("written")
