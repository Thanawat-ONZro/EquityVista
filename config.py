import os
from pathlib import Path

BASE = Path(__file__).parent
RAW_DIR = BASE / "data" / "RAW"
PROCESSED_DIR = BASE / "data" / "processed"
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR = BASE / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

TICKER = "PTT.BK"          # configurable; user will finalize the liquid ticker
START = "2016-01-01"       # 5-10y history per proposal
END = "2024-12-31"
SEED = 42
WINDOW = 60                # LSTM sequence length (60 trading days)
TEST_SPLIT = 0.8           # 80/20 train/test