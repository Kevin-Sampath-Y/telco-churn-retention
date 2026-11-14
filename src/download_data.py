"""
Download the public Telco churn CSV into data/raw/.

    python -m src.download_data

If your network blocks the download, get the file from Kaggle instead
(https://www.kaggle.com/datasets/blastchar/telco-customer-churn) and save it as
data/raw/Telco-Customer-Churn.csv.
"""
from __future__ import annotations

import urllib.request

from src.config import DATA_URL, RAW_DIR, RAW_FILE


def download(force: bool = False) -> str:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if RAW_FILE.exists() and not force:
        print(f"Already present: {RAW_FILE}")
        return str(RAW_FILE)
    print(f"Downloading {DATA_URL}")
    with urllib.request.urlopen(DATA_URL, timeout=60) as resp:
        RAW_FILE.write_bytes(resp.read())
    print(f"Saved {RAW_FILE} ({RAW_FILE.stat().st_size / 1e6:.2f} MB)")
    return str(RAW_FILE)


if __name__ == "__main__":
    download()
