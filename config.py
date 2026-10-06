from pathlib import Path

BASE = Path(__file__).parent
RAW_DIR = BASE / "data" / "RAW"
PROCESSED_DIR = BASE / "data" / "processed"
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
# Canonical data layout, the ONE place these paths are defined: raw -> dataset -> features.
RAW_KAGGLE = RAW_DIR / "kaggle"  # 5,884 csv, Kaggle, 1962-2020
RAW_YF = RAW_DIR / "yfinance"  # 2020+ csv from src/extend_fetch.py (+ skipped.txt: tickers yfinance cannot serve)
DATASET = PROCESSED_DIR / "dataset_1962_2026.parquet"  # cleaned union of both raw dirs (src/build_dataset.py)
FEATURES = PROCESSED_DIR / "features.parquet"  # 25 features, horizon-agnostic (src/features.py)
