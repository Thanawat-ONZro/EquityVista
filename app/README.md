# EquityVista desktop app

Single-view dark dashboard: ticker bar, candlestick chart (lightweight-charts) with the predicted
next-day price as a dashed "Forecast" line, and a results panel. Uses `src/predict.py`.

## Install
```
pip install -r requirements.txt   # includes PySide6 and lightweight-charts
```

## Run
From the repo root (needs `models/lgbm_y.txt` and `models/cnn_lstm.keras`; without them the chart still works):
```
python app/main.py
```
Type a US ticker, press Predict. The first prediction is slow (TensorFlow load + 2y download).

*Educational, not financial advice; measured: no edge after costs.*
