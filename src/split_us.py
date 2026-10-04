"""Global-calendar train/val/test splitter with held-out (unseen) tickers.

Contract (consumed by the models in Step 4)
-------------------------------------------
make_splits(feat) -> {"train", "val", "test_seen", "test_unseen"}, each a tuple
    (ids, y)
  ids : pl.DataFrame[row, ticker, date]; `row` = row index into `feat`
        (sorted ticker,date as in features.parquet), so the models can pull
        features with feat[row, FEATURES] and build 60-step windows on the fly
        from per-ticker arrays. No windows are materialised here.
  y   : np.ndarray (n, 2) float64, columns [y (t+1 log ret), y5 (t+1..t+5 cum)].
        y5 is NaN on the last 4 rows of each ticker (kept, not dropped; mask
        with np.isnan when training/scoring the 5-day target).

Split rules (one calendar for ALL tickers, never per-ticker %):
  train : date <= TRAIN_END minus `gap` trading days
  val   : TRAIN_END < date <= val_end minus `gap` trading days
  test  : test_start <= date <= TEST_END
  Buffers are carved from the END of train and of val (the later period's start
  stays fixed), so no 60-step window / 5-day label of an earlier split reaches
  into the next one. `gap` counts trading days of the global calendar (unique
  dates in `feat`).
  Held-out tickers (holdout_frac of the unique tickers, seeded) appear ONLY in
  test_unseen; the rest of the tickers feed train/val/test_seen.
"""
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parent.parent
FEATS = ROOT / "data" / "processed" / "features.parquet"
HOLDOUT_TXT = ROOT / "data" / "processed" / "holdout_tickers.txt"

TRAIN_END = "2015-12-31"
TEST_END = "2020-04-01"
TARGETS = ["y", "y5"]
NON_FEATURES = {"ticker", "date", *TARGETS}


def feature_cols(feat: pl.DataFrame) -> list[str]:
    return [c for c in feat.columns if c not in NON_FEATURES]


def _cut(cal: list, bound, gap: int):
    """Last calendar date <= bound, moved back `gap` trading days."""
    i = max(i for i, d in enumerate(cal) if d <= bound)
    return cal[i - gap]


def make_splits(feat: pl.DataFrame, val_end="2017-12-31", test_start="2018-01-01",
                gap=60, holdout_frac=0.2, seed=42):
    cal = feat["date"].unique().sort().to_list()
    to_date = lambda s: pl.Series([s]).str.to_date()[0]
    train_cut = _cut(cal, to_date(TRAIN_END), gap)   # train: date <= train_cut
    val_lo = to_date(TRAIN_END)                      # val:   val_lo < date <= val_cut
    val_cut = _cut(cal, to_date(val_end), gap)
    t_lo, t_hi = to_date(test_start), to_date(TEST_END)
    assert val_cut < t_lo and train_cut < val_lo

    tickers = sorted(feat["ticker"].unique().to_list())
    rng = np.random.default_rng(seed)
    hold = sorted(rng.choice(tickers, size=round(holdout_frac * len(tickers)), replace=False).tolist())
    HOLDOUT_TXT.write_text("\n".join(hold) + "\n")

    f = feat.with_row_index("row")
    d, held = pl.col("date"), pl.col("ticker").is_in(hold)
    masks = {
        "train": (d <= train_cut) & ~held,
        "val": (d > val_lo) & (d <= val_cut) & ~held,
        "test_seen": (d >= t_lo) & (d <= t_hi) & ~held,
        "test_unseen": (d >= t_lo) & (d <= t_hi) & held,
    }
    out = {}
    for name, m in masks.items():
        s = f.filter(m)
        out[name] = (s.select("row", "ticker", "date"), s.select(TARGETS).to_numpy())
    return out


# Baselines (predicted next-day return y_hat; scored in evaluate step, same rows as test):
#   zero-return : y_hat = 0                      (direction: always "up" / class of 0 => none)
#   majority    : direction = most frequent sign(y) in TRAIN, applied to every row
#   reversal    : y_hat = -ret1 (yesterday's loser is today's winner)
def baselines(feat: pl.DataFrame, ids: pl.DataFrame, y_train: np.ndarray) -> dict:
    n = ids.height
    majority_up = float(np.mean(y_train[:, 0] > 0)) >= 0.5
    ret1 = feat["ret1"].gather(ids["row"]).to_numpy()
    return {"zero": np.zeros(n), "majority_up": np.full(n, majority_up),
            "reversal": -ret1}


if __name__ == "__main__":
    feat = pl.read_parquet(FEATS)
    sp = make_splits(feat)
    for k, (ids, y) in sp.items():
        print(f"{k:12s} rows={ids.height:>10,} tickers={ids['ticker'].n_unique():>5,} "
              f"dates {ids['date'].min()} .. {ids['date'].max()}  y.shape={y.shape}")
    r = {k: (v[0]["date"].min(), v[0]["date"].max()) for k, v in sp.items()}
    assert r["train"][1] < r["val"][0] and r["val"][1] < r["test_seen"][0], "date overlap"
    tr_t, va_t = set(sp["train"][0]["ticker"]), set(sp["val"][0]["ticker"])
    un_t = set(sp["test_unseen"][0]["ticker"])
    assert not (un_t & (tr_t | va_t)), "holdout leaked into train/val"
    assert not (set(sp["test_seen"][0]["ticker"]) & un_t)
    rows = [set(v[0]["row"]) for v in sp.values()]
    assert sum(len(s) for s in rows) == len(set().union(*rows)), "row overlap"
    print("holdout tickers:", len(HOLDOUT_TXT.read_text().split()), "->", HOLDOUT_TXT)
    print("OK: no date overlap train<val<test, no ticker/row leakage")
