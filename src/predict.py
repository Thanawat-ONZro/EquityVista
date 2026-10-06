"""Production H-day forecast for ANY US ticker (no per-ticker training or state).

Model = the active run for that horizon (models/active.json, see src/active.py); every model type is a classifier.
Output: P(up) = the chance that the price is higher after H trading days (H = 21 = one month), and direction = UP if P(up) >= 0.5.
The model says nothing about the SIZE of the move, so there is no expected return, target price or price range.

13 of the 25 features are SAME-DATE cross-sectional (mkt_ret and the r_* percentile ranks), so a lone ticker
has no cross-section. The ranks (and the outlier caps) come from the run's model/ref_quantiles.json: quantiles of
each feature over the last year of the training features file, i.e. the same universe the model was trained on.
mkt_ret is still the mean of a fixed basket of liquid large caps (REFERENCE): an approximation.

EDUCATIONAL ONLY, NOT FINANCIAL ADVICE. Measured edge is small.
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import polars as pl
import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import active  # noqa: E402
from src.features import CAPPED, add_features  # noqa: E402
from src.models.common import FEATURES  # noqa: E402

REFERENCE = ("AAPL MSFT AMZN GOOGL META NVDA TSLA JPM JNJ V WMT PG XOM UNH HD MA BAC KO PEP CVX "
             "ABBV MRK PFE COST CSCO TMO ADBE CRM NKE MCD INTC T VZ ORCL ABT DIS WFC CAT IBM HON").split()
DISCLAIMER = "Educational, not financial advice."
_models = {}  # path -> (fitted pipeline, run metrics, reference quantiles); keyed by file so an active-model switch reloads itself


class NoModelError(ValueError):
    """No active model for the requested horizon."""

    def __init__(self, horizon):
        super().__init__(f"no active model for the {horizon}-day horizon")
        self.horizon = horizon


def _load(horizon):
    p = active.path(horizon)
    if p is None:
        raise NoModelError(horizon)
    if p not in _models:
        run = p.parent.parent
        try:
            m = json.loads((run / "metrics.json").read_text())
            ref = json.loads((run / "model" / "ref_quantiles.json").read_text())
        except (OSError, KeyError, ValueError) as e:
            raise ValueError(f"run {run.name} is incomplete, retrain it (python train_all.py): {e}") from e
        _models.clear()
        _models[p] = (joblib.load(p), m, ref)
    return _models[p]


def _fetch(ticker):
    syms = list(dict.fromkeys([ticker] + REFERENCE))
    raw = yf.download(syms, period="2y", auto_adjust=True, progress=False, group_by="ticker", threads=True)
    frames = []
    for s in syms:
        if s not in raw.columns.get_level_values(0):
            continue
        d = raw[s].dropna(subset=["Close"])
        d = d[d["Volume"] > 0]
        if d.empty:
            continue
        frames.append(pl.DataFrame({
            "ticker": s,
            "date": pl.Series(np.asarray(d.index, dtype="datetime64[D]")).cast(pl.Date),
            "adj_open": d["Open"].to_numpy(float),
            "adj_high": d["High"].to_numpy(float), "adj_low": d["Low"].to_numpy(float),
            "adj_close": d["Close"].to_numpy(float), "adj_volume": d["Volume"].to_numpy(float)}))
    if not frames or ticker not in {f["ticker"][0] for f in frames}:
        raise ValueError(f"no price data for {ticker!r} (bad symbol or delisted)")
    return pl.concat(frames)


def predict_ticker(ticker: str, horizon: int = 21, hide: int = 0) -> dict:
    """H-trading-day forecast for `ticker`. Raises NoModelError if no active model for `horizon`,
    ValueError if data is missing/too short. hide=N (backtest): the last N trading days of the ticker are dropped
    from EVERY symbol BEFORE the features are built, so the forecast starts at that earlier cutoff date and nothing
    after it (not even the same-date cross-sectional ranks) is used."""
    ticker, horizon, hide = ticker.upper().strip(), int(horizon), int(hide)
    model, m, ref = _load(horizon)  # fail fast, before the download
    raw = _fetch(ticker)
    if hide:
        days = raw.filter(pl.col("ticker") == ticker)["date"].unique().sort()
        if len(days) <= hide + 100:
            raise ValueError(f"{ticker}: not enough history to hide {hide} days")
        raw = raw.filter(pl.col("date") <= days[-hide - 1])
    feat = add_features(raw, cap=False)
    mine = feat.filter(pl.col("ticker") == ticker).sort("date")
    if mine.is_empty():
        raise ValueError(f"{ticker}: not enough history for the features")
    last = mine.row(-1, named=True)
    for f, q in ref.items():  # rank + cap against the training universe, not the download basket
        last[f"r_{f}"] = float(np.interp(last[f], q, np.linspace(0, 1, len(q))))
        if f in CAPPED:
            last[f] = float(np.clip(last[f], q[1], q[-2]))  # the training features are capped at the 1 % / 99 % percentiles
    x = np.array([[last[f] for f in FEATURES]], np.float32)
    p_up = float(model.predict_proba(x)[0, 1] if hasattr(model, "predict_proba") else model.predict(x)[0])  # else: LightGBM Booster
    px = float(last["adj_close"])
    target = pd.bdate_range(last["date"], periods=horizon + 1)[-1].date()  # weekends skipped; holidays are not modelled
    return {
        "ticker": ticker, "horizon": horizon, "date": last["date"], "target_date": target, "last_close": px,
        "direction": "UP" if p_up >= 0.5 else "DOWN",
        "confidence": p_up,                                       # P(up)
        "calibrated": bool(m.get("test_calibrated", 1)),          # False: P(up) failed the calibration check on test
        "hidden": hide, "run_id": active.run_id(horizon),
        "stale_days": (pd.Timestamp.today().normalize() - pd.Timestamp(last["date"])).days,
    }


def show(r: dict) -> None:
    print(f"\n=== {r['ticker']} | last close {r['last_close']:.2f} on {r['date']} -> {r['horizon']} trading days "
          f"(~{r['target_date']}) ===\n"
          f"  {r['direction']:4s} P(up) = {r['confidence']:.0%}{'' if r['calibrated'] else ' (not calibrated)'}\n  {DISCLAIMER}")
    if r["stale_days"] > 5:
        print(f"  WARNING: last bar is {r['stale_days']} days old")


if __name__ == "__main__":
    args = sys.argv[1:]
    horizon = int(args.pop(args.index("--days") + 1)) if "--days" in args else 21
    if "--days" in args:
        args.remove("--days")
    failed = 0
    for t in args or ["AAPL", "MSFT"]:
        try:
            show(predict_ticker(t, horizon))
        except Exception as e:     # keep going; non-zero exit if any requested ticker failed
            failed += 1
            print(f"\n=== {t} === ERROR: {e}", file=sys.stderr)
    sys.exit(1 if failed else 0)
