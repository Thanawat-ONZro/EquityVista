"""Scale-free, leakage-safe feature builder for the US multi-ticker model.

Features at day t use only data <= t (per-ticker rolling/ewm are causal; the
cross-sectional ranks and mkt_ret only look at the same date t).
"""
import time
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "processed" / "us_market.parquet"
OUT = ROOT / "data" / "processed" / "features.parquet"

WARMUP = 60  # first 60 rows per ticker dropped (ret60 needs 61 prices)
RANKED = ["ret1", "ret5", "ret20", "ret60", "vol20", "vol60",
          "close_ma20", "close_ma50", "rsi14", "macd_n", "vol_z", "hl_range"]


def _g(e):
    return e.over("ticker")


def add_features(df: pl.DataFrame) -> pl.DataFrame:
    df = df.select("ticker", "date", "adj_open", "adj_high", "adj_low",
                   "adj_close", "adj_volume").sort("ticker", "date")
    c = pl.col("adj_close")
    lv = (pl.col("adj_volume") + 1).log()
    df = df.with_columns(ret1=_g((c / c.shift(1)).log()))
    r1 = pl.col("ret1")
    delta = _g(c.diff())
    df = df.with_columns(
        ret5=_g(r1.rolling_sum(5)), ret20=_g(r1.rolling_sum(20)), ret60=_g(r1.rolling_sum(60)),
        vol20=_g(r1.rolling_std(20)), vol60=_g(r1.rolling_std(60)),
        close_ma20=c / _g(c.rolling_mean(20)), close_ma50=c / _g(c.rolling_mean(50)),
        macd_n=(_g(c.ewm_mean(span=12, adjust=False)) - _g(c.ewm_mean(span=26, adjust=False))) / c,
        vol_z=(lv - _g(lv.rolling_mean(20))) / (_g(lv.rolling_std(20)) + 1e-9),
        hl_range=(pl.col("adj_high") - pl.col("adj_low")) / c,
        _up=_g(delta.clip(lower_bound=0).ewm_mean(alpha=1 / 14, adjust=False)),   # Wilder smoothing
        _dn=_g((-delta).clip(lower_bound=0).ewm_mean(alpha=1 / 14, adjust=False)),
    ).with_columns(rsi14=100 - 100 / (1 + pl.col("_up") / (pl.col("_dn") + 1e-12)))
    # market return uses every ticker/row (incl. warmup) of that date only
    df = df.with_columns(mkt_ret=pl.col("ret1").mean().over("date"))
    # drop per-ticker warmup, then non-finite rows (zero prices etc.)
    df = df.filter(pl.int_range(pl.len()).over("ticker") >= WARMUP)
    feats = RANKED + ["mkt_ret"]
    df = df.with_columns(pl.col(feats).fill_nan(None)).drop_nulls(feats)
    df = df.filter(pl.all_horizontal(pl.col(feats).is_finite()))
    # cross-sectional pct rank per date (0..1)
    df = df.with_columns(
        (pl.col(f).rank("average").over("date") / pl.len().over("date")).alias(f"r_{f}")
        for f in RANKED)
    return df.drop("_up", "_dn")


def add_targets(df: pl.DataFrame) -> pl.DataFrame:
    """y = next-day log return; y5 = next-5-day cumulative log return (t+1..t+5).
    Rows with null y (last row per ticker) are dropped; y5 may be null at the tail."""
    df = df.sort("ticker", "date")
    df = df.with_columns(y=_g(pl.col("ret1").shift(-1)), y5=_g(pl.col("ret5").shift(-5)))
    return df.drop_nulls("y")


def test_no_lookahead(df: pl.DataFrame, n=10, n_tickers=50, seed=0, tol=1e-9):
    """df: raw OHLCV frame. Truncating all rows after a cut date must leave every
    feature at/before the cut unchanged."""
    rng = np.random.default_rng(seed)
    tk = rng.choice(df["ticker"].unique().sort().to_numpy(), n_tickers, replace=False)
    df = df.filter(pl.col("ticker").is_in(list(tk)))
    full = add_features(df)
    cols = [c for c in full.columns if c not in ("ticker", "date")]
    dates = full["date"].unique().sort().to_numpy()
    for cut in rng.choice(dates, n, replace=False):
        part = add_features(df.filter(pl.col("date") <= cut))
        ref = full.filter(pl.col("date") <= cut)
        assert part.shape == ref.shape, (cut, part.shape, ref.shape)
        a = ref.sort("ticker", "date").select(cols).to_numpy()
        b = part.sort("ticker", "date").select(cols).to_numpy()
        assert np.abs(a - b).max() <= tol, f"lookahead at cut {cut}: {np.abs(a - b).max()}"
    return True


if __name__ == "__main__":
    t0 = time.time()
    raw = pl.read_parquet(SRC)
    print("leakage test passed:", test_no_lookahead(raw))
    out = add_targets(add_features(raw))
    out.write_parquet(OUT)
    import pandas as pd
    chk = pd.read_parquet(OUT)
    print("pandas read ok:", chk.shape)
    print(out.shape, out.columns)
    print(out.head(3))
    print(f"{time.time() - t0:.1f}s")
