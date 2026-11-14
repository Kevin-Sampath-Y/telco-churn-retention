"""
Load, validate and clean the raw Telco table; add engineered features.

Design rules
  * Validation fails loudly (ValueError) instead of letting bad data flow downstream.
  * Cleaning decisions are documented here, next to the code that makes them.
  * Engineered features only use information available at prediction time (no leakage).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import ADDON_SERVICES, EXPECTED_RAW_COLUMNS, ID_COL, RAW_FILE, TARGET


def load_raw(path: Path | str = RAW_FILE) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python -m src.download_data` (or `python run_pipeline.py --synthetic`).")
    return pd.read_csv(path)


def validate_raw(df: pd.DataFrame) -> None:
    """Structural checks on the raw file. Raises ValueError with a clear message."""
    missing = [c for c in EXPECTED_RAW_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Raw data is missing columns: {missing}")
    if df[ID_COL].duplicated().any():
        raise ValueError("customerID is not unique")
    if set(df[TARGET].dropna().unique()) - {"Yes", "No"}:
        raise ValueError("Churn must contain only 'Yes' / 'No'")
    if (pd.to_numeric(df["tenure"], errors="coerce") < 0).any():
        raise ValueError("tenure has negative values")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Return a tidy copy: correct types, no missing values, binary target (1 = churned)."""
    validate_raw(df)
    d = df.copy()
    for c in d.select_dtypes(include=["object", "string"]).columns:
        d[c] = d[c].astype(str).str.strip()

    # TotalCharges arrives as text; 11 rows are blank. All of them have tenure == 0, i.e. customers
    # who signed up this month and have not been billed yet, so 0 is the truthful value.
    total = pd.to_numeric(d["TotalCharges"], errors="coerce")
    blank = total.isna()
    if (d.loc[blank, "tenure"] != 0).any():
        raise ValueError("TotalCharges is missing for customers with tenure > 0: investigate before imputing")
    d["TotalCharges"] = total.fillna(0.0)

    d["SeniorCitizen"] = d["SeniorCitizen"].map({0: "No", 1: "Yes", "0": "No", "1": "Yes"})
    d[TARGET] = (d[TARGET] == "Yes").astype(int)

    if d.drop(columns=[ID_COL]).isna().any().any():
        raise ValueError("Unexpected missing values after cleaning")
    return d


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineered features, computable from a single customer's record."""
    d = df.copy()
    d["n_addons"] = (d[ADDON_SERVICES] == "Yes").sum(axis=1)
    tenure = d["tenure"].replace(0, np.nan)
    d["avg_monthly_charge"] = (d["TotalCharges"] / tenure).fillna(d["MonthlyCharges"])
    # positive = pays more per month now than the historical average (price rise or upgrades)
    d["price_change"] = d["MonthlyCharges"] - d["avg_monthly_charge"]
    d["autopay"] = np.where(d["PaymentMethod"].str.contains("automatic", case=False), "Yes", "No")
    return d


def prepare(path: Path | str = RAW_FILE) -> pd.DataFrame:
    """Raw file -> model-ready table (cleaned + engineered)."""
    return add_features(clean(load_raw(path)))
