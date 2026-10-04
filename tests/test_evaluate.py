import math
import pytest
from src.evaluate import regression_metrics, clf_metrics


def test_regression_metrics_known():
    y_true = [100, 101, 102]
    y_pred = [100, 102, 103]
    m = regression_metrics(y_true, y_pred)
    # Differences: [0, -1, -1] -> MAE = 2/3, RMSE = sqrt(2/3)
    assert m["MAE"] == pytest.approx(2 / 3)
    assert m["RMSE"] == pytest.approx(math.sqrt(2 / 3))
    assert m["MAPE"] > 0

    # Additional case matching MAE == 1.0 and RMSE == sqrt(2)
    m2 = regression_metrics([100, 100], [100, 102])
    assert m2["MAE"] == pytest.approx(1.0)
    assert m2["RMSE"] == pytest.approx(math.sqrt(2))
    assert m2["MAPE"] > 0


def test_clf_metrics_perfect():
    y_true = [1, 1, 2, 2]
    y_pred = [1, 1, 2, 2]
    res = clf_metrics(y_true, y_pred)
    assert res["accuracy"] == 1.0
    assert res["confusion_matrix"] == [[2, 0], [0, 2]]


def test_clf_metrics_single_class_pred():
    # Test safe behavior when predictions contain only one class
    y_true = [1, 1, 2, 2]
    y_pred = [1, 1, 1, 1]
    res = clf_metrics(y_true, y_pred)
    assert res["accuracy"] == 0.5
    assert res["confusion_matrix"] == [[2, 0], [2, 0]]
