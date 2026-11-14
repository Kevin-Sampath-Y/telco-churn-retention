"""
Central configuration: paths, column groups, modelling settings and palette.

Every other module imports from here, so a name or a number is defined once.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

RAW_FILE = RAW_DIR / "Telco-Customer-Churn.csv"
SCORED_FILE = PROCESSED_DIR / "scored_customers.csv"
MODEL_FILE = MODELS_DIR / "churn_model.joblib"
META_FILE = MODELS_DIR / "model_meta.json"
SYNTHETIC_MARKER = RAW_DIR / "_SYNTHETIC_DATA.txt"

# Public copy of the IBM sample dataset (same file as the Kaggle "Telco Customer Churn").
DATA_URL = ("https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
            "master/data/Telco-Customer-Churn.csv")

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

ID_COL = "customerID"
TARGET = "Churn"

ADDON_SERVICES = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
                  "TechSupport", "StreamingTV", "StreamingMovies"]

# gender is deliberately NOT a model feature (protected attribute, no business reason to use it).
NUMERIC = ["tenure", "MonthlyCharges", "TotalCharges", "n_addons", "avg_monthly_charge", "price_change"]
CATEGORICAL = ["SeniorCitizen", "Partner", "Dependents", "PhoneService", "MultipleLines",
               "InternetService", *ADDON_SERVICES, "Contract", "PaperlessBilling",
               "PaymentMethod", "autopay"]
FEATURES = NUMERIC + CATEGORICAL

EXPECTED_RAW_COLUMNS = [
    "customerID", "gender", "SeniorCitizen", "Partner", "Dependents", "tenure", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling",
    "PaymentMethod", "MonthlyCharges", "TotalCharges", "Churn",
]

PALETTE = {"primary": "#1F4E79", "accent": "#F28E2B", "good": "#2E8B57", "bad": "#C0392B",
           "neutral": "#6B7686", "ink": "#1B2430", "light": "#A9C1DB"}
