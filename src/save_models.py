import sys
import warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

import config as cfg
from src.fetch import fetch_data
from src.features import add_indicators, make_sequences
from src.models.arima import fit_predict
from src.models.lstm import train_lstm
from src.models.rf_classifier import make_direction, train_rf
from src.evaluate import regression_metrics, clf_metrics

RF_FEATS = ["Close", "Return", "MA20", "MA50", "RSI", "MACD", "Signal"]


def save_all(ticker=cfg.TICKER, model_dir=cfg.MODELS_DIR):
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(cfg.SEED); tf.random.set_seed(cfg.SEED)

    df = add_indicators(fetch_data(ticker, cfg.START, cfg.END))
    n = len(df)

    # ARIMA on log1p(Close); metrics = rolling one-step with frozen params (results.ipynb cell 3)
    log_close = np.log1p(df["Close"])
    ar = fit_predict(log_close, cfg.TEST_SPLIT, order=(5, 1, 0))
    cut_a = int(n * cfg.TEST_SPLIT)
    pr = ar["model"].apply(log_close.values).get_prediction(start=cut_a)
    arima_pred = np.expm1(pr.predicted_mean)

    # LSTM: scalers fit on TRAIN windows only (cell 5), epochs=80 (cell 6)
    X, y = make_sequences(df, cfg.WINDOW)
    cut_s = int(len(X) * cfg.TEST_SPLIT)
    X_tr, X_te, y_tr, y_te = X[:cut_s], X[cut_s:], y[:cut_s], y[cut_s:]
    fx = MinMaxScaler().fit(X_tr.reshape(-1, X.shape[2]))
    fy = MinMaxScaler().fit(y_tr.reshape(-1, 1))
    sx = lambda a: fx.transform(a.reshape(-1, a.shape[2])).reshape(a.shape)
    y_tr_s = fy.transform(y_tr.reshape(-1, 1)).ravel()
    y_te_s = fy.transform(y_te.reshape(-1, 1)).ravel()
    lstm = train_lstm(sx(X_tr), y_tr_s, sx(X_te), y_te_s, epochs=80)
    lstm_pred = fy.inverse_transform(lstm["test_pred"].reshape(-1, 1)).ravel()

    # RF: next-day shifted label, no leak (cell 8)
    X_all = df[RF_FEATS].values
    y_next = make_direction(df["Close"].values)[1:]
    rf = train_rf(X_all[:-1], y_next, test_size=1 - cfg.TEST_SPLIT, seed=cfg.SEED)

    # persist
    paths = {k: model_dir / v for k, v in {
        "lstm": "lstm.keras", "scaler_feat": "scaler_feat.pkl",
        "scaler_target": "scaler_target.pkl", "rf": "rf.pkl", "arima": "arima.pkl"}.items()}
    lstm["model"].save(paths["lstm"])
    joblib.dump(fx, paths["scaler_feat"])
    joblib.dump(fy, paths["scaler_target"])
    joblib.dump(rf["model"], paths["rf"])
    ar["model"].remove_data()
    joblib.dump({"model": ar["model"], "transform": "log1p", "order": (5, 1, 0),
                 "split": cfg.TEST_SPLIT}, paths["arima"])

    return {
        "ticker": ticker,
        "artifacts": {k: str(v) for k, v in paths.items()},
        "metrics": {
            "arima": regression_metrics(df["Close"].values[cut_a:], arima_pred),
            "lstm": regression_metrics(y_te, lstm_pred),
            "rf": clf_metrics(rf["y_test"], rf["y_pred"]),
        },
    }


if __name__ == "__main__":
    import pprint
    pprint.pprint(save_all())
