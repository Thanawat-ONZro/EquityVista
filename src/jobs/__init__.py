"""Training Hub job modules: `python -m src.jobs.<name> [--run-dir DIR] [--params JSON]`.

PARAMS = default params per job type (= the "Standard" preset). Kept import-free (the GUI reads it; no lightgbm here).
Use ints (not bools) for flags: the New Job form casts text with type(default).
All train_* jobs share: features, start_year (first training year), horizon (forecast days), frac (share of rows), universe (1 = keep only rows with in_universe, see src/clean.universe_flag), seed.
"""
from config import DATASET, FEATURES

FEATURES_FILE = FEATURES.name
TRAIN = ("train_lgbm", "train_logistic", "train_rf")
ROWS = 0.0532  # share of the ~18.8M training rows every model is trained on = ~1M rows: validation rank-IC stops rising there
PARAMS = {   # the train_* values were all chosen on VALIDATION (REPORT/figures/experiments.json), never on the test split
    "data_extend": {"start": "2020-01-01", "limit": 0, "workers": 6},
    "build_features": {"source": DATASET.stem, "leakage_test": 0, "start_year": 1962},
    "train_lgbm": {"features": FEATURES_FILE, "start_year": 1962, "horizon": 21, "rounds": 400, "lr": 0.05, "num_leaves": 63,
                   "min_data": 2000, "feature_fraction": 0.8, "frac": ROWS, "universe": 0, "trials": 0, "max_minutes": 30, "seed": 42},
    "train_logistic": {"features": FEATURES_FILE, "start_year": 1962, "horizon": 21, "frac": ROWS, "universe": 0, "C": 1e-5, "seed": 42},
    "train_rf": {"features": FEATURES_FILE, "start_year": 1962, "horizon": 21, "frac": ROWS, "universe": 0, "n_estimators": 100,
                 "max_depth": 12, "min_leaf": 200, "max_features": 0.33, "seed": 42},
    "evaluate": {},
}
