"""
Train, compare and select the churn model.

Protocol (this ordering is what makes the final numbers trustworthy)
  1. Hold out 20% of customers as a TEST set (stratified) and do not touch it again until the end.
  2. On the remaining 80%, tune each candidate model with 5-fold stratified cross-validation.
  3. Select the model with the best cross-validated ROC-AUC (selection never sees the test set).
  4. Refit the winner on all training data and score the test set ONCE.
  5. Create out-of-fold predictions for every customer (used by the app's campaign back-test).
"""
from __future__ import annotations

import json
import platform
import time

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from scipy.stats import loguniform, randint, uniform
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV, StratifiedKFold,
                                     cross_val_predict, train_test_split)
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.config import (CV_FOLDS, FEATURES, ID_COL, META_FILE, MODEL_FILE, MODELS_DIR, RANDOM_STATE,
                        SCORED_FILE, TARGET, TEST_SIZE)
from src.features import build_preprocessor


def split(df: pd.DataFrame):
    X, y = df[FEATURES], df[TARGET]
    return train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)


def _pipe(model) -> Pipeline:
    return Pipeline([("prep", build_preprocessor()), ("model", model)])


def candidates(fast: bool = False) -> dict:
    """name -> (pipeline, search object factory). `fast` shrinks the search for tests."""
    n_iter = 4 if fast else 25
    cv = StratifiedKFold(CV_FOLDS if not fast else 3, shuffle=True, random_state=RANDOM_STATE)
    common = dict(scoring="roc_auc", cv=cv, n_jobs=-1, refit=True)
    return {
        "Baseline (predict base rate)": (_pipe(DummyClassifier(strategy="prior")), None),
        "Logistic regression": (
            _pipe(LogisticRegression(max_iter=2000)),
            GridSearchCV(_pipe(LogisticRegression(max_iter=2000)),
                         {"model__C": [0.01, 0.03, 0.1, 0.3, 1, 3, 10]}, **common)),
        "Random forest": (
            _pipe(RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)),
            RandomizedSearchCV(_pipe(RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)),
                               {"model__n_estimators": [200, 400], "model__max_depth": [4, 6, 8, 12],
                                "model__min_samples_leaf": [5, 10, 20, 40]},
                               n_iter=min(n_iter, 8), random_state=RANDOM_STATE, **common)),
        "XGBoost": (
            _pipe(XGBClassifier(random_state=RANDOM_STATE, eval_metric="logloss", n_jobs=1)),
            RandomizedSearchCV(_pipe(XGBClassifier(random_state=RANDOM_STATE, eval_metric="logloss", n_jobs=1)),
                               {"model__n_estimators": randint(80, 400), "model__max_depth": randint(2, 6),
                                "model__learning_rate": loguniform(0.02, 0.2),
                                "model__subsample": uniform(0.6, 0.4), "model__colsample_bytree": uniform(0.5, 0.5),
                                "model__min_child_weight": randint(1, 12), "model__reg_lambda": loguniform(0.5, 20)},
                               n_iter=n_iter, random_state=RANDOM_STATE, **common)),
    }


def _test_metrics(pipe, X_te, y_te) -> dict:
    p = pipe.predict_proba(X_te)[:, 1]
    return {"roc_auc": float(roc_auc_score(y_te, p)),
            "pr_auc": float(average_precision_score(y_te, p)),
            "brier": float(brier_score_loss(y_te, p))}


def train(df: pd.DataFrame, fast: bool = False, verbose: bool = True) -> dict:
    X_tr, X_te, y_tr, y_te = split(df)
    comparison, fitted = [], {}
    for name, (base, search) in candidates(fast).items():
        t0 = time.time()
        if search is None:
            est = base.fit(X_tr, y_tr)
            cv_auc, params = 0.5, {}
        else:
            search.fit(X_tr, y_tr)
            est, cv_auc = search.best_estimator_, float(search.best_score_)
            params = {k.replace("model__", ""): (float(v) if isinstance(v, (float, np.floating)) else
                                                 int(v) if isinstance(v, (int, np.integer)) else v)
                      for k, v in search.best_params_.items()}
        fitted[name] = est
        m = _test_metrics(est, X_te, y_te)
        comparison.append({"model": name, "cv_roc_auc": round(cv_auc, 4), **{f"test_{k}": round(v, 4) for k, v in m.items()},
                           "best_params": params, "seconds": round(time.time() - t0, 1)})
        if verbose:
            print(f"  {name:<30} CV AUC {cv_auc:.4f} | test AUC {m['roc_auc']:.4f} PR-AUC {m['pr_auc']:.4f} ({time.time() - t0:.0f}s)")

    # Selection uses cross-validated score only (never the test set).
    best_name = max((c for c in comparison if c["model"] != "Baseline (predict base rate)"),
                    key=lambda c: c["cv_roc_auc"])["model"]
    final = fitted[best_name]

    # Out-of-fold probabilities for EVERY customer: honest scores for the app's back-test.
    cv = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    from sklearn.base import clone
    oof = cross_val_predict(clone(final), df[FEATURES], df[TARGET], cv=cv, method="predict_proba", n_jobs=1)[:, 1]
    scored = df[[ID_COL, *FEATURES, TARGET]].copy()
    scored["churn_probability"] = oof
    scored["split"] = np.where(df.index.isin(X_te.index), "test", "train")

    meta = {
        "selected_model": best_name,
        "comparison": comparison,
        "n_train": int(len(X_tr)), "n_test": int(len(X_te)),
        "base_rate": round(float(df[TARGET].mean()), 4),
        "features": FEATURES,
        "versions": {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
                     "xgboost": xgboost.__version__, "pandas": pd.__version__, "numpy": np.__version__},
    }
    return {"model": final, "meta": meta, "scored": scored,
            "X_train": X_tr, "X_test": X_te, "y_train": y_tr, "y_test": y_te}


def save(result: dict) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    SCORED_FILE.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(result["model"], MODEL_FILE)
    META_FILE.write_text(json.dumps(result["meta"], indent=2, default=float))
    result["scored"].to_csv(SCORED_FILE, index=False)
