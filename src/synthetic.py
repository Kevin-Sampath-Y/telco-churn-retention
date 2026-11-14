"""
Small synthetic generator with the Telco schema.

Used by the test-suite and by `python run_pipeline.py --synthetic` so the project
runs with no internet. Figures from synthetic data are NOT real results.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def generate(n: int = 3000, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    contract = rng.choice(["Month-to-month", "One year", "Two year"], n, p=[.55, .21, .24])
    internet = rng.choice(["DSL", "Fiber optic", "No"], n, p=[.34, .44, .22])
    tenure = np.where(contract == "Month-to-month", rng.integers(0, 40, n), rng.integers(10, 73, n))
    pay = rng.choice(["Electronic check", "Mailed check", "Bank transfer (automatic)",
                      "Credit card (automatic)"], n, p=[.34, .23, .22, .21])

    def yn(p):
        return np.where(rng.random(n) < p, "Yes", "No")

    has_net = internet != "No"
    addons = {c: np.where(has_net, yn(.4), "No internet service")
              for c in ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
                        "TechSupport", "StreamingTV", "StreamingMovies"]}
    phone = yn(.9)
    base = np.where(internet == "Fiber optic", 75, np.where(internet == "DSL", 52, 22))
    monthly = np.round(base + rng.normal(0, 8, n) + 6 * (np.array(list(addons.values())) == "Yes").sum(0) * .5, 2)
    monthly = np.clip(monthly, 18.25, 119)
    total = np.round(monthly * tenure * rng.uniform(.9, 1.05, n), 2)

    logit = (-1.1 + 1.5 * (contract == "Month-to-month") - 1.2 * (contract == "Two year")
             + 0.8 * (internet == "Fiber optic") + 0.6 * (pay == "Electronic check")
             - 0.035 * tenure + 0.008 * (monthly - 60) - 0.5 * (addons["TechSupport"] == "Yes")
             + rng.normal(0, 0.5, n))
    churn = rng.random(n) < 1 / (1 + np.exp(-logit))

    df = pd.DataFrame({
        "customerID": [f"{i:04d}-SYNTH" for i in range(n)],
        "gender": rng.choice(["Male", "Female"], n),
        "SeniorCitizen": (rng.random(n) < .16).astype(int),
        "Partner": yn(.48), "Dependents": yn(.3), "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": np.where(phone == "No", "No phone service", yn(.45)),
        "InternetService": internet, **addons,
        "Contract": contract, "PaperlessBilling": yn(.6), "PaymentMethod": pay,
        "MonthlyCharges": monthly, "TotalCharges": total.astype(str),
        "Churn": np.where(churn, "Yes", "No"),
    })
    df.loc[df["tenure"] == 0, "TotalCharges"] = " "      # same quirk as the real file
    return df
