import numpy as np


def aggregate(rf_probs, rf_classes, lstm_pred, lstm_truth, arima_fc, arima_conf, weights=None):
    probs = np.asarray(rf_probs)
    if probs.ndim > 1:
        probs = probs[-1]

    idx = {int(c): i for i, c in enumerate(rf_classes)}
    up = float(probs[idx[1]]) if 1 in idx else 0.0
    down = float(probs[idx[2]]) if 2 in idx else 0.0
    flat = float(probs[idx[0]]) if 0 in idx else 0.0

    prob_list = [up, down, flat]
    conf = float(max(prob_list))
    best_idx = int(np.argmax(prob_list))
    direction = {0: "up", 1: "down", 2: "sideway"}[best_idx]

    w = weights if weights is not None else {"arima": 0.6, "lstm": 0.4}
    arima_val = float(np.asarray(arima_fc).ravel()[-1])
    lstm_val = float(np.asarray(lstm_pred).ravel()[-1])
    price_pred = float(w["arima"] * arima_val + w["lstm"] * lstm_val)

    if arima_conf is not None:
        conf_arr = np.asarray(arima_conf).ravel()
        lo = float(conf_arr[-2])
        hi = float(conf_arr[-1])
    else:
        lo, hi = price_pred, price_pred

    return {
        "direction": direction,
        "confidence": conf,
        "price_pred": price_pred,
        "interval": [lo, hi],
    }
