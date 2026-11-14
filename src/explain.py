"""
Explainability: which features drive churn, globally and for one customer.

  * Permutation importance: model-agnostic, measured on the hold-out set in ORIGINAL features.
  * SHAP: how much each feature pushed THIS prediction up or down (exact for tree/linear models).
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression

from src.config import PALETTE as C, RANDOM_STATE
from src.evaluate import save_fig
from src.features import pretty_names


def _parts(pipe):
    return pipe.named_steps["prep"], pipe.named_steps["model"]


def permutation_table(pipe, X_test: pd.DataFrame, y_test, n_repeats: int = 10) -> pd.DataFrame:
    r = permutation_importance(pipe, X_test, y_test, scoring="roc_auc", n_repeats=n_repeats,
                               random_state=RANDOM_STATE, n_jobs=1)
    t = pd.DataFrame({"feature": X_test.columns, "auc_drop": r.importances_mean, "std": r.importances_std})
    return t.sort_values("auc_drop", ascending=False).reset_index(drop=True)


def plot_permutation(table: pd.DataFrame, top: int = 12) -> str:
    t = table.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(t["feature"], t["auc_drop"], xerr=t["std"], color=C["primary"], error_kw={"ecolor": C["neutral"]})
    ax.set_xlabel("Drop in ROC-AUC when the feature is shuffled")
    ax.set_title(f"'{table.iloc[0]['feature']}' matters most: shuffling it costs the model the most accuracy")
    ax.grid(axis="y", visible=False)
    return save_fig(fig, "11_permutation_importance.png", "Hold-out set, 10 shuffles per feature. Bars show +/- 1 std.")


def _explainer(pipe, background: pd.DataFrame):
    prep, model = _parts(pipe)
    bg = prep.transform(background)
    if isinstance(model, LogisticRegression):
        return shap.LinearExplainer(model, bg)
    return shap.TreeExplainer(model)


def shap_values(pipe, X: pd.DataFrame, background: pd.DataFrame):
    """Return (values [n, n_features], base_value, feature_names, transformed matrix)."""
    prep, _ = _parts(pipe)
    Xt = prep.transform(X)
    ex = _explainer(pipe, background.sample(min(200, len(background)), random_state=RANDOM_STATE))
    vals = ex.shap_values(Xt)
    if isinstance(vals, list):           # older shap returns one array per class
        vals = vals[1]
    base = np.ravel(ex.expected_value)[-1]
    return np.asarray(vals), float(base), pretty_names(prep.get_feature_names_out()), Xt


def plot_shap_summary(pipe, X_test: pd.DataFrame, background: pd.DataFrame, max_rows: int = 1000) -> str:
    X = X_test.sample(min(max_rows, len(X_test)), random_state=RANDOM_STATE)
    vals, _, names, Xt = shap_values(pipe, X, background)
    plt.figure(figsize=(8.5, 6))
    shap.summary_plot(vals, Xt, feature_names=names, max_display=14, show=False, plot_size=None)
    fig = plt.gcf()
    plt.title("What pushes customers towards churning (red = high value)", loc="left", fontsize=12, fontweight="bold")
    return save_fig(fig, "12_shap_summary.png", "SHAP values on a hold-out sample. Right of centre = raises churn risk.")


def top_drivers(pipe, row: pd.DataFrame, background: pd.DataFrame, k: int = 6) -> pd.DataFrame:
    """The k features that moved this one customer's prediction the most (signed, readable labels)."""
    vals, base, names, Xt = shap_values(pipe, row, background)
    labels = []
    for n, x in zip(names, Xt[0]):
        if " = " in n:                                   # one-hot column: say whether it applies
            labels.append(n if x >= 0.5 else n.replace(" = ", " is not "))
        else:                                            # numeric column: show the real value
            v = row.iloc[0][n]
            labels.append(f"{n} = {v:,.1f}" if isinstance(v, (int, float, np.integer, np.floating)) else n)
    d = pd.DataFrame({"feature": labels, "impact": vals[0]})
    d = d.reindex(d["impact"].abs().sort_values(ascending=False).index).head(k)
    d.attrs["base_value"] = base
    return d.reset_index(drop=True)
