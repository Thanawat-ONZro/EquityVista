"""Production next-day prediction for ANY US ticker (no per-ticker training or state).

Models are fixed (models/lgbm_y.txt, models/cnn_lstm.keras). The features are scale-free,
but 13 of the 25 LGBM features are SAME-DATE cross-sectional (mkt_ret and the r_* percentile ranks),
so a lone ticker has no cross-section. We therefore download the requested ticker together with a
fixed basket of liquid large caps (REFERENCE) and run features.add_features on the whole panel.
The ranks are relative to that ~40-name basket, not the 2,447-name training universe: an
approximation (see README limitations).

EDUCATIONAL ONLY, NOT FINANCIAL ADVICE. Measured edge on 2018-2020 data is ~1-3 pp direction accuracy
and it does not survive trading costs (see data/processed/eval_summary.txt).
"""
import json
import os
import sys
from datetime import timedelta
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl
import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.features import add_features  # noqa: E402
from src.models import cnn_lstm_model as cnn  # noqa: E402
from src.models.lgbm_model import FEATURES as LGBM_FEATURES  # noqa: E402
from src.models.lgbm_model import MODEL_DIR, predict_lgbm  # noqa: E402
from src.split import FEATS, HOLDOUT_TXT, make_splits  # noqa: E402

REFERENCE = ("AAPL MSFT AMZN GOOGL META NVDA TSLA JPM JNJ V WMT PG XOM UNH HD MA BAC KO PEP CVX "
             "ABBV MRK PFE COST CSCO TMO ADBE CRM NKE MCD INTC T VZ ORCL ABT DIS WFC CAT IBM HON").split()
SCALER = MODEL_DIR / "cnn_lstm_scaler.npz"
DISCLAIMER = "Educational, not financial advice."
_models = {}


def _booster():
    if "lgbm" not in _models:
        m = lgb.Booster(model_file=str(MODEL_DIR / "lgbm_y.txt"))
        m.best_iteration = m.current_iteration()    # saved with num_iteration=best -> use all trees
        _models["lgbm"] = m
    return _models["lgbm"]


def _cnn():
    if "cnn" not in _models:
        import keras
        _models["cnn"] = keras.saving.load_model(cnn.MODEL_PATH, compile=False)
    return _models["cnn"]


def _cnn_scaler():
    """Global TRAIN-only mean/std the CNN was trained with; computed once from features.parquet, cached."""
    if not SCALER.exists():
        feat = pl.read_parquet(FEATS, columns=["ticker", "date"] + cnn.FEATURES + ["y", "y5"])
        tr = make_splits(feat)["train"][0]["row"].to_numpy()
        X = feat.select(cnn.FEATURES).to_numpy().astype("float32")[tr]
        np.savez(SCALER, mu=X.mean(0), sd=X.std(0) + 1e-9)
    z = np.load(SCALER)
    return z["mu"], z["sd"]


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


def predict_ticker(ticker: str, n_days: int = 60) -> dict:
    """Next-trading-day log-return predictions for `ticker` from both fixed models.

    n_days = CNN-LSTM window length (the saved model needs 60). Raises ValueError if data is missing/too short.
    """
    ticker = ticker.upper().strip()
    if n_days != cnn.W:
        raise ValueError(f"n_days must be {cnn.W} (the saved CNN-LSTM window)")
    feat = add_features(_fetch(ticker))
    mine = feat.filter(pl.col("ticker") == ticker).sort("date")
    if mine.height < n_days:
        raise ValueError(f"{ticker}: only {mine.height} usable feature rows, need {n_days}")
    last = mine.row(-1, named=True)

    # LGBM: scale-free features of the last row (same column order as training: imported FEATURES)
    assert set(LGBM_FEATURES) <= set(mine.columns)
    lgbm_ret = float(predict_lgbm(_booster(), np.array([mine.height - 1]), mine)[0])

    # CNN-LSTM: last 60 rows, standardised with the TRAIN stats
    mu, sd = _cnn_scaler()
    win = np.clip((mine.tail(n_days).select(cnn.FEATURES).to_numpy().astype("float32") - mu) / sd, -8, 8)
    cnn_ret = float(_cnn().predict(win[None], verbose=0)[1].ravel()[0] / cnn.RET_SCALE)

    px = mine.tail(1)["adj_close"][0]
    nxt = last["date"] + timedelta(days=1)
    while nxt.weekday() >= 5:                      # skip weekends; exchange holidays are not modelled
        nxt += timedelta(days=1)
    conf = float(min(1.0, abs(lgbm_ret) / (last["vol20"] + 1e-9)))   # |pred| / own 20d vol, heuristic
    # Model-driven (val-residual) interval, NOT a calibrated VaR. None if interval.json is missing.
    try:
        rs = json.loads((MODEL_DIR / "interval.json").read_text())["y"]
        half = 1.96 * rs
        iv_log = [lgbm_ret - half, lgbm_ret + half]
        iv = [float(px * np.exp(v)) for v in iv_log]
    except (OSError, KeyError, ValueError):
        iv_log = iv = None
    return {
        "interval_95": iv, "interval_95_log": iv_log,
        "ticker": ticker, "date": last["date"], "last_close": float(px), "next_date": nxt,
        "next_day_return_pred": lgbm_ret, "direction": "UP" if lgbm_ret >= 0 else "DOWN",
        "confidence": conf, "price_prev": float(px), "price_pred": float(px * np.exp(lgbm_ret)),
        "model": "LightGBM",
        "cnn_lstm": {"next_day_return_pred": cnn_ret, "direction": "UP" if cnn_ret >= 0 else "DOWN",
                     "price_pred": float(px * np.exp(cnn_ret))},
        "stale_days": (pd.Timestamp.today().normalize() - pd.Timestamp(last["date"])).days,
    }


def show(r: dict) -> None:
    c = r["cnn_lstm"]
    print(f"\n=== {r['ticker']} | last close {r['last_close']:.2f} on {r['date']} -> next session {r['next_date']} ===\n"
          f"  LightGBM : {r['next_day_return_pred']:+.3%}  {r['direction']:4s} "
          f"(price view {r['price_pred']:.2f}, |pred|/vol20 = {r['confidence']:.2f})\n"
          f"  CNN-LSTM : {c['next_day_return_pred']:+.3%}  {c['direction']:4s} (price view {c['price_pred']:.2f})\n"
          f"  Measured: LGBM ~51.5% direction acc., no edge after costs; CNN-LSTM no edge. {DISCLAIMER}")
    if r["stale_days"] > 5:
        print(f"  WARNING: last bar is {r['stale_days']} days old")


if __name__ == "__main__":
    tickers = sys.argv[1:] or ["AAPL", "MSFT"]
    extra = HOLDOUT_TXT.read_text().split()[:3] if not sys.argv[1:] and HOLDOUT_TXT.exists() else []
    failed = 0
    for t in tickers + extra:      # held-out symbols are best-effort (many are delisted since 2020)
        try:
            show(predict_ticker(t))
        except Exception as e:     # keep going; non-zero exit only if a requested ticker failed
            failed += t in tickers
            print(f"\n=== {t} === ERROR: {e}", file=sys.stderr)
    sys.exit(1 if failed else 0)
