import numpy as np
from statsmodels.tsa.arima.model import ARIMA as ARIMA_MODEL


def fit_predict(series, split=0.8, order=(5, 1, 0)):
    # Raw prices used as ARIMA(5,1,0) fits stably without convergence warnings or invalid interval bounds.
    n = len(series)
    cut = int(n * split)
    train = series.iloc[:cut].values
    test = series.iloc[cut:].values

    model = ARIMA_MODEL(train, order=order).fit()
    forecast_res = model.get_forecast(len(test))
    fc = forecast_res.predicted_mean
    conf = forecast_res.conf_int(alpha=0.05)

    return {
        "train_pred": model.fittedvalues,
        "test_pred": fc,
        "test_conf": conf,
        "next": fc[-1] if len(fc) else None,
        "next_interval": conf[-1] if len(conf) else None,
        "model": model,
    }
