"""
Preprocessing as a scikit-learn ColumnTransformer.

Keeping preprocessing INSIDE the model pipeline matters: during cross-validation the
scaler and encoder are re-fitted on each training fold only, so nothing from the
validation fold leaks into training.
"""
from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import CATEGORICAL, NUMERIC


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC),
            # handle_unknown="ignore": a category never seen in training becomes all zeros
            # instead of crashing the app; drop="if_binary" avoids a redundant Yes/No pair.
            ("cat", OneHotEncoder(handle_unknown="ignore", drop="if_binary", sparse_output=False), CATEGORICAL),
        ],
        verbose_feature_names_out=False,
    )


def pretty_names(names) -> list[str]:
    """OneHotEncoder names like 'Contract_Month-to-month' -> 'Contract = Month-to-month'."""
    out = []
    for n in names:
        if "_" in n and n.split("_")[0] in CATEGORICAL:
            head, tail = n.split("_", 1)
            out.append(f"{head} = {tail}")
        else:
            out.append(n)
    return out
