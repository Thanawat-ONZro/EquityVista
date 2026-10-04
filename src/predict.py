import sys
import warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

import config as cfg
from src.fetch import fetch_data
from src.features import add_indicators
from src.ensemble import aggregate

SEQ_FEATS = ["Close", "Return", "MA20", "MA50", "RSI", "MACD"]
RF_FEATS = SEQ_FEATS + ["Signal"]


def predict_next(ticker=cfg.TICKER, model_dir=cfg.MODELS_DIR):
    d = Path(model_dir)
    df = add_indicators(fetch_data(ticker, cfg.START, cfg.END))

    # ARIMA: frozen train-fit params re-applied to full history; log1p in, expm1 out
    meta = joblib.load(d / "arima.pkl")
    assert meta["transform"] == "log1p"
    res = meta["model"].apply(np.log1p(df["Close"].values))
    fc = res.get_forecast(1)
    arima_next = np.expm1(fc.predicted_mean)
    arima_ci = np.expm1(fc.conf_int(alpha=0.05))  # shape (1, 2)

    # LSTM: saved train-only scalers, last WINDOW days -> THB
    fx, fy = joblib.load(d / "scaler_feat.pkl"), joblib.load(d / "scaler_target.pkl")
    win = df[SEQ_FEATS].values[-cfg.WINDOW:]
    win_s = fx.transform(win).reshape(1, cfg.WINDOW, -1)
    lstm_s = load_model(d / "lstm.keras").predict(win_s, verbose=0)
    lstm_next = fy.inverse_transform(lstm_s).ravel()

    # RF: today's feature row -> next-day direction probs
    rf = joblib.load(d / "rf.pkl")
    probs = rf.predict_proba(df[RF_FEATS].values[-1:])

    out = aggregate(probs, rf.classes_, lstm_next, None, arima_next, arima_ci)
    out["next_date"] = (df.index[-1] + pd.offsets.BDay(1)).date()
    out["last_close"] = float(df["Close"].iloc[-1])
    out["last_date"] = df.index[-1].date()
    return out


if __name__ == "__main__":
    r = predict_next()
    lo, hi = r["interval"]
    print("=" * 52)
    print(f"  NEXT-DAY FORECAST — {cfg.TICKER}")
    print("=" * 52)
    print(f"  Date                : {r['next_date']}")
    print(f"  Last close          : {r['last_close']:.2f} THB ({r['last_date']})")
    print(f"  Predicted price     : {r['price_pred']:.2f} THB  (0.6 ARIMA + 0.4 LSTM)")
    print(f"  Predicted direction : {r['direction'].upper()}")
    print(f"  Confidence          : {r['confidence'] * 100:.1f} %  (RF class probability)")
    print(f"  95% interval        : [{lo:.2f}, {hi:.2f}] THB  (ARIMA, 1-day ahead)")
    print("=" * 52)
    print("  Educational project — not financial advice.")
