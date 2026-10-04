# EquityVista — Next-Day Stock Forecasting with an ARIMA + LSTM + RandomForest Ensemble

**Problem:** predict the next trading day's **direction** (up / down / flat), **closing price** and a **95% interval** for a Thai stock (PTT.BK) using an AI/ML ensemble, and evaluate it honestly.

## Approach

Three complementary models, merged by a small aggregator:

| Model | Type | Role | Output |
| --- | --- | --- | --- |
| ARIMA(5,1,0) | Statistical | Linear trend / autocorrelation | Next-day price + 95% CI |
| LSTM(50)→Dropout(0.2)→LSTM(50)→Dropout(0.2)→Dense(1) | Deep learning | Non-linear momentum (60-day windows) | Next-day price |
| RandomForest (200 trees) | Machine learning | 3-class direction (up / down / flat) | Class probabilities |
| **Ensemble** | Aggregator | Combines the above | Direction + confidence + price + 95% interval |

**Ensemble rules:** direction and confidence come from the RandomForest class probabilities; price = `0.6 × ARIMA + 0.4 × LSTM`; the interval is ARIMA's 1-day 95% CI.

**Data:** PTT.BK from Yahoo Finance (`yfinance`), 2016-01-01 → 2024-12-31, 2186 trading days. **80/20 chronological split, no shuffling.**

### Data-leak discipline
- **Scalers fit on the training set only** (separate feature and target `MinMaxScaler`s), predictions inverse-transformed before metrics.
- **RF label is the *next-day* direction.** Using same-day direction leaks the answer (accuracy 1.000); the correct shifted label gives 0.366.
- **ARIMA is scored with rolling one-step-ahead forecasts.** A single 428-step forecast produced absurdly wide intervals; ARIMA uses a `log1p` transform (inverted with `expm1`), and the next-day interval comes from a full-history refit.
- LSTM trains up to 80 epochs with `EarlyStopping` (stopped around epoch 32).

## Results

Test window, PTT.BK. **Units:** MAE and RMSE in THB; MAPE in %; accuracy = 3-class direction accuracy (RF/Ensemble only). `NaN` = not applicable to that model.

| Model | MAE (THB) | RMSE (THB) | MAPE (%) | Direction acc. |
| --- | --- | --- | --- | --- |
| ARIMA | 0.2125 | 0.3038 | 0.7436 | NaN |
| LSTM | 0.3078 | 0.4158 | 1.0755 | NaN |
| RF | NaN | NaN | NaN | 0.3388 |
| Ensemble | 0.2379 | 0.3244 | 0.8321 | 0.3388 |
| *Naive (previous close, reference)* | 0.2067 | 0.3035 | 0.7236 | — |

> **Next-day forecast for PTT.BK (after 2024-12-31):** direction **DOWN**, confidence **48%**, price **28.38 THB**, 95% interval **[27.52, 29.33]**.

### Honest interpretation
- Price error of the ensemble is **about the same as the naive "tomorrow = today" baseline**, i.e. the stock behaves close to a random walk at a 1-day horizon.
- Direction accuracy (0.3388) is **below the majority-class baseline**, so the direction signal is not useful here.
- This is a school project. The result is reported as measured, not tuned to look good. **It is not financial advice.**

## Repository structure

```
EquityVista/
├── config.py            # ticker, dates, seed, LSTM window, train/test split
├── requirements.txt     # dependencies (requirements.lock.txt = pinned versions)
├── src/
│   ├── fetch.py         # download PTT.BK from Yahoo Finance -> data/RAW
│   ├── features.py      # technical indicators, labels, chronological split
│   ├── evaluate.py      # MAE / RMSE / MAPE, accuracy, confusion matrix
│   ├── ensemble.py      # direction + confidence + price + 95% interval
│   └── models/          # arima.py, lstm.py, rf_classifier.py
├── notebooks/
│   ├── EDA.ipynb        # data exploration
│   └── results.ipynb    # trains models, comparison table, next-day forecast
├── tests/               # pytest suite
├── data/                # RAW/ and processed/ (generated)
└── app/                 # PLANNED (Phase 2) - not implemented yet
```

## How to run

Requires Python ≥ 3.11 (git-bash syntax; on cmd use `.venv\Scripts\activate`).

```bash
python -m venv .venv
source .venv/Scripts/activate
pip install -r requirements.txt

python src/fetch.py                      # download data

jupyter notebook notebooks/EDA.ipynb     # explore
jupyter notebook notebooks/results.ipynb # reproduce results

# or run headless:
jupyter nbconvert --to notebook --execute --inplace notebooks/EDA.ipynb notebooks/results.ipynb
```

## Testing

```bash
pytest tests/
```

7 tests (features, evaluation metrics, ensemble) — all pass.

## Future work — Phase 2 (planned, not implemented)

A desktop app (**PySide6**) in `app/` that loads the trained ensemble and shows live next-day forecasts. Further modelling ideas: more features (market index, news sentiment), walk-forward validation, and multiple tickers.
