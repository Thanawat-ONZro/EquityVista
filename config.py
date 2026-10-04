from pathlib import Path

BASE = Path(__file__).parent
RAW_DIR = BASE / "data" / "RAW"
PROCESSED_DIR = BASE / "data" / "processed"
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)