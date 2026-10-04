import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, confusion_matrix


def regression_metrics(y_true, y_pred):
    y_t = np.asarray(y_true)
    y_p = np.asarray(y_pred)
    mae = float(mean_absolute_error(y_t, y_p))
    rmse = float(np.sqrt(mean_squared_error(y_t, y_p)))
    mape = float(np.mean(np.abs((y_t - y_p) / (y_t + 1e-9))) * 100)
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}


def clf_metrics(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
