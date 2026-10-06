"""Uniform cleaning rules (src/clean.py): one tiny fixture per rule + no-look-ahead of the per-date statistics."""
from datetime import date, timedelta

import numpy as np
import polars as pl

from src import clean as K
from src.features import CAPPED, add_features

D0 = date(2019, 1, 1)
SCHEMA = {"ticker": pl.String, "date": pl.Date, "open": pl.Float64, "high": pl.Float64, "low": pl.Float64,
          "close": pl.Float64, "adj": pl.Float64, "vol": pl.Float64, "src": pl.Int8}


def series(ticker, n=12, start=D0, px=100.0, src=0, adj_mult=1.0):
    """n calm days (+-0.1 %/day, valid OHLC) for one ticker."""
    return [dict(ticker=ticker, date=start + timedelta(days=i), open=px, high=px * 1.01, low=px * 0.99,
                 close=px * (1 + 0.001 * (i % 2)), adj=px * (1 + 0.001 * (i % 2)) * adj_mult, vol=1000.0, src=src) for i in range(n)]


def run(rows, excluded=(), min_rows=3):
    out, log, extra = K.clean(pl.DataFrame(rows, schema=SCHEMA), set(excluded), min_rows)
    return out, {s["name"].split(" (")[0]: s["rows"] for s in log.steps}, log, extra


def removed(log, group):
    return sum(s["rows"] for s in log.steps if s["group"] == group)


def test_clean_rows_pass_untouched():
    out, _, log, _ = run(series("AAA"))
    assert out.height == 12 and sum(s["rows"] for s in log.steps) == 0


def test_errors_impossible_rows_dropped_and_counted():
    rows = series("AAA")
    rows[1].update(high=90.0)                      # high < low and < open/close
    rows[2].update(low=105.0)                      # low > min(open, close)
    rows[3].update(low=-1.0)                       # non-positive price (a USED field; a bad open alone is kept, see below)
    rows[4].update(vol=0.0)                        # zero volume
    rows[5].update(open=20.9, high=7000.0, low=20.4, close=6700.0, adj=6700.0)  # KTB: two price scales; its close jump is a bad print
    out, _, log, _ = run(rows)
    assert out.height == 6 and removed(log, "ERRORS") == 6 and removed(log, "MISSING") == 0   # rows 1-4, plus 5 and 6 (jump out / back)
    assert {1, 2, 3, 4, 5, 6}.isdisjoint((out["date"] - D0).dt.total_days())


def test_bad_print_return_dropped_one_pass():
    rows = series("AAA")
    rows[6].update(open=200.0, high=202.0, low=198.0, close=200.0, adj=200.0)   # +69 % log move, valid OHLC
    out, _, log, _ = run(rows)
    assert 6 not in set((out["date"] - D0).dt.total_days()) and removed(log, "ERRORS") >= 1


def test_duplicate_ticker_date_keeps_last():
    rows = series("AAA")
    rows.append({**rows[3], "close": 100.5, "adj": 100.5, "high": 102.0})
    out, _, log, _ = run(rows)
    assert out.height == 12 and removed(log, "DUPLICATES") == 1
    assert out.filter(pl.col("date") == D0 + timedelta(days=3))["adj_close"][0] == 100.5


def test_missing_row_dropped_not_imputed():
    rows = series("AAA")
    rows[4].update(close=None)
    rows[7].update(vol=None)
    out, _, log, _ = run(rows)
    assert out.height == 10 and removed(log, "MISSING") == 2 and out["adj_close"].null_count() == 0


def test_instrument_and_min_history():
    rows = series("AAA") + series("AAA.W") + series("TINY", n=2)
    out, _, log, extra = run(rows, excluded={"AAA.W"})
    assert set(out["ticker"]) == {"AAA"}
    assert removed(log, "INSTRUMENT") == 12 and removed(log, "MIN HISTORY") == 2 and extra["short"]["ticker"].unique().to_list() == ["TINY"]


def test_seam_rebase_mismatch_flag_and_overlap():
    base = series("AAA", n=10, start=D0, adj_mult=1.0)                       # old basis
    new = series("AAA", n=10, start=D0 + timedelta(days=6), adj_mult=0.5, src=1)    # new basis: every adj close is half (dividends since)
    new[2].update(adj=new[2]["adj"] * 1.2)                                    # one date whose local ratio disagrees (>1 %)
    out, _, log, extra = run(base + new)
    seam = extra["seam"]["table"]
    assert abs(seam["r"][0] - 0.5) < 1e-9 and seam["n_flagged"][0] == 1
    assert removed(log, "DUPLICATES") == 3                                    # 4 overlap dates - 1 flagged = 3 base copies
    assert out["date"].is_duplicated().sum() == 0
    flagged_day = new[2]["date"]
    assert flagged_day not in set(out["date"])                                # the mismatching date is an error, on both sides
    first = out.sort("date").row(0, named=True)                               # base row rebased onto the new basis
    assert abs(first["adj_close"] - 50.0) < 1e-9


def test_outliers_flagged_and_capped_never_deleted():
    rng = np.random.default_rng(0)
    n = 200
    df = pl.DataFrame({"date": [D0] * n, "ticker": [f"T{i}" for i in range(n)], "x": rng.normal(0, 0.01, n)})
    df = df.with_columns(pl.when(pl.col("ticker") == "T0").then(5.0).otherwise(pl.col("x")).alias("x"))
    f = df.select(**K.flags("x"))
    assert f["mad"][0] and f["iqr"][0] and f["mad"].sum() < 20                # the planted outlier is found, calm rows mostly not
    w = K.winsorize(df, ["x"])
    assert w.height == n and w["x"].max() < 0.1 and w["x"].max() <= df["x"].quantile(K.WINSOR[1], "linear") + 1e-12


def _panel():
    rng = np.random.default_rng(1)
    dates = [D0 + timedelta(days=i) for i in range(90)]
    rows = []
    for t in range(40):
        px = 50 * np.exp(np.cumsum(rng.normal(0, 0.02, len(dates))))
        rows += [{"ticker": f"T{t}", "date": d, "adj_open": p, "adj_high": p * 1.02, "adj_low": p * 0.98, "adj_close": p,
                  "adj_volume": float(rng.uniform(1e4, 1e6))} for d, p in zip(dates, px)]
    return pl.DataFrame(rows)


def _stats(df):
    """Every per-date statistic of the cleaning layer, as plain columns."""
    df = K.add_universe(df.sort("ticker", "date")).with_columns(x=K.ret1_expr())
    df = K.winsorize(df, ["x", "adj_volume"]).with_columns(m=K.flags("adj_close")["mad"], i=K.flags("adj_close")["iqr"])
    return df.sort("ticker", "date").select("ticker", "date", "in_universe", "x", "adj_volume", "m", "i")


def test_per_date_statistics_have_no_lookahead():
    df = _panel()
    cut = D0 + timedelta(days=70)
    full = _stats(df).filter(pl.col("date") <= cut)
    assert _stats(df.filter(pl.col("date") <= cut)).equals(full, null_equal=True)        # truncating the future changes nothing
    fut = df.with_columns(pl.when(pl.col("date") > cut).then(pl.col("adj_volume") * 1e6).otherwise(pl.col("adj_volume")).alias("adj_volume"))
    assert _stats(fut).filter(pl.col("date") <= cut).equals(full, null_equal=True)       # rewriting it changes nothing either


def test_features_cap_but_keep_every_row(monkeypatch):
    f = add_features(_panel())
    monkeypatch.setattr(K, "winsorize", lambda df, cols, *a: df)
    raw = add_features(_panel())
    monkeypatch.undo()
    assert {"in_universe", *CAPPED}.issubset(f.columns) and f.height == raw.height   # capped, never deleted
    m, mr = (x.group_by("date").agg(hi=pl.col("ret1").max(), lo=pl.col("ret1").min()).sort("date") for x in (f, raw))
    assert (m["hi"] <= mr["hi"] + 1e-12).all() and (m["lo"] >= mr["lo"] - 1e-12).all()
    assert (m["hi"] < mr["hi"] - 1e-9).any() or (m["lo"] > mr["lo"] + 1e-9).any()


def test_seam_identity_break_splits_ticker_instead_of_deleting():
    base = series("AAA", n=10, start=D0)
    new = series("AAA", n=10, start=D0 + timedelta(days=6), src=1)
    for i, r in enumerate(new[:4]):                                           # overlap ratios 1, 1.3, 1, 1.3: not one security
        r.update(adj=r["adj"] * (1 + 0.3 * (i % 2)))
    out, _, log, extra = run(base + new)
    assert extra["seam"]["split"] == ["AAA"] and set(out["ticker"]) == {"AAA", "AAA~2"}
    assert out.height == 20 and removed(log, "DUPLICATES") == 0               # every row survives


def test_missing_open_alone_does_not_drop_the_row():
    rows = series("AAA")
    rows[3].update(open=0.0)                       # 1970s-80s coding of "no open"; high/low/close/adj/volume are fine
    rows[4].update(open=-1.0, close=0.0, adj=0.0)  # a used price is still impossible -> dropped
    out, _, log, _ = run(rows)
    assert out.height == 11 and removed(log, "MISSING") == 0 and removed(log, "ERRORS") == 1
    day3 = out.filter(pl.col("date") == D0 + timedelta(days=3)).row(0, named=True)
    assert day3["adj_open"] is None and day3["adj_close"] > 0   # the open is unknown, not a fake 0
    assert out.filter(pl.col("date") == D0 + timedelta(days=5))["adj_open"][0] > 0


def test_crash_day_with_wide_consistent_range_survives_and_is_capped():
    rows = series("AAA")
    rows[6].update(open=100.0, high=130.0, low=30.0, close=95.0, adj=95.0)   # high/low = 4.3, internally consistent
    rows[8].update(open=100.0, high=90.0, low=30.0, close=95.0, adj=95.0)    # impossible: high < open, close
    out, _, log, _ = run(rows)
    assert out.height == 11 and removed(log, "ERRORS") == 1
    kept = out.filter(pl.col("date") == D0 + timedelta(days=6)).row(0, named=True)
    assert kept["adj_high"] / kept["adj_low"] > 4
    panel = _panel()                                                           # 40 tickers x 90 days, ranges ~4 %
    wide = pl.col("ticker").eq("T0") & (pl.col("date") == D0 + timedelta(days=80))
    wild = panel.with_columns(pl.when(wide).then(pl.col("adj_close") * 4.3).otherwise(pl.col("adj_high")).alias("adj_high"))
    f = add_features(wild).filter(pl.col("date") == D0 + timedelta(days=80))
    raw_range = 4.3 - 0.98                                                     # (high - low) / close of the planted row
    assert f.filter(pl.col("ticker") == "T0").height == 1
    assert f["hl_range"].max() < raw_range * 0.7 and f["hl_range"].max() > 0.03  # the row is kept; its feature is capped
