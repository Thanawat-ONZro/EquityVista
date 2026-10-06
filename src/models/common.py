"""Shared by every model type: the 25 features, data loading for one horizon, row subsampling.

Features (all scale-free): 13 per-row features + their 12 same-date cross-sectional percentile ranks.
No ticker/date, no adj_* price/volume levels. Same features, same split, same metrics for ALL models.
"""
import numpy as np
import polars as pl

from config import PROCESSED_DIR
from src.features import add_targets
from src.split import ROOT, make_splits

RANKED = ["ret1", "ret5", "ret20", "ret60", "vol20", "vol60",
          "close_ma20", "close_ma50", "rsi14", "macd_n", "vol_z", "hl_range"]
FEATURES = RANKED + ["mkt_ret"] + [f"r_{c}" for c in RANKED]
MODEL_DIR = ROOT / "models"


def rows(ids):
    return ids["row"].to_numpy() if isinstance(ids, pl.DataFrame) else np.asarray(ids)


def X(feat, ids):
    return feat[rows(ids), FEATURES].cast(pl.Float32).to_numpy()


def load(features: str, horizon: int, start_year: int = 1962, universe: int = 0):
    """features parquet (name under data/processed, or an absolute path) -> (feat, splits) for `horizon`; train rows start at `start_year`.
    universe=1 keeps only rows flagged `in_universe` (trailing-only liquidity filter); targets are built BEFORE the filter."""
    lf = pl.scan_parquet(PROCESSED_DIR / features).select(
        "ticker", "date", "adj_close", pl.col(FEATURES).cast(pl.Float32), *(["in_universe"] if universe else []))
    feat = add_targets(lf.collect(), horizon).drop("adj_close")
    if universe:
        feat = feat.filter("in_universe").drop("in_universe")
    return feat, make_splits(feat, horizon, start_year)


def sub(part, frac: float, rng):
    """Random row fraction of an (ids, y) split (for quick runs / RAM)."""
    ids, y = part
    if frac >= 1:
        return part
    keep = rng.random(ids.height) < frac
    return ids.filter(pl.Series(keep)), y[keep]
