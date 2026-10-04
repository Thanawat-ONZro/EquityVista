import numpy as np
from src.ensemble import aggregate


def test_aggregate_output_keys():
    r = aggregate(
        rf_probs=np.array([0.6, 0.3, 0.1]),
        rf_classes=np.array([1, 2, 0]),
        lstm_pred=np.array([100.0, 101.0]),
        lstm_truth=np.array([99.0, 100.0]),
        arima_fc=100.5,
        arima_conf=[99.0, 102.0],
    )
    assert set(r.keys()) == {"direction", "confidence", "price_pred", "interval"}
    assert r["direction"] in {"up", "down", "sideway"}
    assert isinstance(r["confidence"], float)
    assert isinstance(r["price_pred"], float)
    assert len(r["interval"]) == 2


def test_aggregate_direction_up():
    r = aggregate(
        rf_probs=np.array([0.8, 0.1, 0.1]),
        rf_classes=np.array([1, 2, 0]),
        lstm_pred=np.array([100.0]),
        lstm_truth=np.array([99.0]),
        arima_fc=100.0,
        arima_conf=[99.0, 101.0],
    )
    assert r["direction"] == "up"
    assert r["confidence"] == 0.8
