"""Global-calendar train/val/test splitter with held-out (unseen) tickers, for ONE forecast horizon H.

make_splits(feat, horizon, start_year) -> {"train", "val", "test_seen", "test_unseen"}, each a tuple (ids, y)
  feat : DataFrame with ticker, date and y = forward log return over `horizon` trading days
         (features.add_targets), sorted ticker,date.
  ids  : pl.DataFrame[row, ticker, date]; `row` = row index into `feat`.
  y    : np.ndarray (n,) float64, the H-day forward log return.

Split rules (one fixed calendar for ALL tickers, never per-ticker %, never data-driven):
  train : start_year <= year, date <= TRAIN_END (2021-12-31) minus `gap` trading days
  val   : VAL_START (2022-01-01) <= date <= VAL_END (2023-12-31) minus `gap` trading days
  test  : TEST_START (2024-01-01) <= date <= end of data
  EMBARGO: gap = max(horizon, 60 trading days). The cut moves BACK from each boundary, so the last train date is
  >= gap trading days before the first val date (same val -> test): no label window (t..t+H) of an earlier split
  reaches the next one. `_check_embargo` asserts it on the actual rows. Trading days = unique dates in `feat`.
  Held-out tickers (holdout_frac of the unique tickers, seeded) appear ONLY in test_unseen.
"""
from pathlib import Path

import numpy as np
import polars as pl

from config import FEATURES, PROCESSED_DIR

ROOT = Path(__file__).resolve().parent.parent
FEATS = FEATURES
HOLDOUT_TXT = PROCESSED_DIR / "holdout_tickers.txt"

TRAIN_END, VAL_START, VAL_END, TEST_START = "2021-12-31", "2022-01-01", "2023-12-31", "2024-01-01"
GAP = 60
START_YEAR = 1962
NON_FEATURES = {"ticker", "date", "y", "in_universe"}


def feature_cols(feat: pl.DataFrame) -> list[str]:
    return [c for c in feat.columns if c not in NON_FEATURES]


def _cut(cal: list, bound, gap: int):
    """Last calendar date <= bound, moved back `gap` trading days."""
    i = max((i for i, d in enumerate(cal) if d <= bound), default=None)
    if i is None or i < gap:
        raise ValueError(f"data does not reach {bound} (+{gap} trading days of embargo)")
    return cal[i - gap]


def _check_embargo(cal: list, out: dict, gap: int) -> dict:
    """Trading-day distance train->val and val->test (seen tickers' rows) must be >= gap."""
    pos = {d: i for i, d in enumerate(cal)}
    first = lambda k: pos[out[k][0]["date"].min()]  # noqa: E731
    last = lambda k: pos[out[k][0]["date"].max()]  # noqa: E731
    gaps = {"train_val": first("val") - last("train"), "val_test": first("test_seen") - last("val")}
    assert all(g >= gap for g in gaps.values()), f"embargo {gaps} < {gap}"
    return gaps


def make_splits(feat: pl.DataFrame, horizon=21, start_year=START_YEAR, gap=GAP, holdout_frac=0.2, seed=42):
    gap = max(int(horizon), int(gap))
    cal = feat["date"].unique().sort().to_list()
    to_date = lambda s: pl.Series([s]).str.to_date()[0]  # noqa: E731
    train_cut = _cut(cal, to_date(TRAIN_END), gap)   # train: year >= start_year, date <= train_cut
    val_lo = to_date(VAL_START)
    val_cut = _cut(cal, to_date(VAL_END), gap)       # val:   val_lo <= date <= val_cut
    t_lo = to_date(TEST_START)                       # test:  date >= t_lo (end of data)

    tickers = sorted(feat["ticker"].unique().to_list())
    rng = np.random.default_rng(seed)
    hold = sorted(rng.choice(tickers, size=round(holdout_frac * len(tickers)), replace=False).tolist())
    HOLDOUT_TXT.write_text("\n".join(hold) + "\n")

    f = feat.with_row_index("row")
    d, held = pl.col("date"), pl.col("ticker").is_in(hold)
    masks = {
        "train": (d.dt.year() >= int(start_year)) & (d <= train_cut) & ~held,
        "val": (d >= val_lo) & (d <= val_cut) & ~held,
        "test_seen": (d >= t_lo) & ~held,
        "test_unseen": (d >= t_lo) & held,
    }
    out = {}
    for name, m in masks.items():
        s = f.filter(m)
        out[name] = (s.select("row", "ticker", "date"), s["y"].to_numpy())
    _check_embargo(cal, out, gap)
    return out


if __name__ == "__main__":
    feat = pl.read_parquet(FEATS)
    from src.features import add_targets
    sp = make_splits(add_targets(feat, 21), 21)
    for k, (ids, y) in sp.items():
        print(f"{k:12s} rows={ids.height:>10,} tickers={ids['ticker'].n_unique():>5,} "
              f"dates {ids['date'].min()} .. {ids['date'].max()}")
    print("OK: embargo >= gap, no date overlap")
