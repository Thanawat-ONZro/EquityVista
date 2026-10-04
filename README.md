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

## Task 12: Large multi-ticker US forecaster

A single fixed model scored across thousands of US stocks (the Phase-1 PTT.BK pipeline above is unchanged).

**Framing:** a *cross-sectional ranker / daily scanner*, **not a price oracle**. It predicts each ticker's next-day log return from scale-free features and is judged per trading day (rank-IC, AUC, long-short decile spread), never on pooled rows.

**Data:** Kaggle US stock-market CSVs, **CC0** licence. 24.2M raw rows → **11.06M rows, 2,447 tickers** after ETL (penny/illiquid/ETF filtering; `data/processed/etl_report.txt`). Dates 1962 → **2020-04-01**.

**Models:** LightGBM (primary; 25 scale-free features, `y` and `y5` heads) and a CNN-LSTM comparison arm (60-day windows, CPU-trained on a 300-ticker subsample). Global-calendar split (train ≤ 2015, val 2016-17, test 2018-20, 60-day gaps) with 20% of tickers held out as *unseen*.

**Results** (test 2018-2020, 565 days, mean over days with 95% t-CI; net = after 20 bps round-trip; full tables in `data/processed/eval_us_summary.txt`):

| 1-day, split | Model | Mean daily IC | Direction acc. − majority (pp) | Net decile spread (bps) |
| --- | --- | --- | --- | --- |
| seen | LGBM | +0.0131 [+0.0048, +0.0213] | +2.78 [+0.41, +5.15] | −14.82 [−24.82, −4.81] |
| seen | CNN-LSTM | −0.0028 [−0.0150, +0.0094] | +1.26 [−1.20, +3.72] | −19.90 [−33.63, −6.17] |
| seen | 1d-reversal | +0.0010 [−0.0088, +0.0108] | +1.20 [−0.88, +3.29] | −30.03 [−44.16, −15.90] |
| unseen | LGBM | +0.0122 [+0.0033, +0.0210] | +2.90 [+0.48, +5.31] | −12.70 [−23.59, −1.82] |
| unseen | CNN-LSTM | −0.0074 [−0.0199, +0.0051] | +1.26 [−1.27, +3.79] | −23.92 [−39.60, −8.23] |
| unseen | 1d-reversal | +0.0026 [−0.0076, +0.0127] | +1.19 [−0.92, +3.31] | −24.89 [−39.63, −10.14] |

5-day horizon (LGBM `y5` head): IC +0.0264 [+0.0040, +0.0489] on seen tickers (positive), +0.0191 [−0.0060, +0.0443] on unseen (not distinguishable from 0); net spread negative in both.

**Verdict:** no model beats the baselines after costs. LightGBM has a small, statistically positive IC and a few points of direction accuracy over the majority baseline, but every net decile spread is negative (**not tradable**). The CNN-LSTM shows no edge and does not beat the LightGBM baseline.

**Limitations**
- **Survivorship bias:** the universe is what the dump lists; delisted names are under-represented.
- **Data ends 2020-04-01**: the models are stale for today's market and were never refit on recent data.
- **Live ranks are approximate:** `mkt_ret` and the `r_*` ranks need a cross-section, so `predict_us.py` uses a ~40-stock large-cap reference basket instead of the 2,447-name training universe.
- **CPU-only:** the CNN-LSTM saw a 300-ticker subsample for 3 epochs; a bigger run might change its result.
- **No edge after costs**; CIs are slightly optimistic (daily IC is mildly autocorrelated); the 20 bps cost is a flat assumption. **Educational, not financial advice.**

**Run it**
```bash
python -m src.predict_us                 # AAPL, MSFT (+ a few held-out symbols, best effort)
python -m src.predict_us NVDA KO         # any tickers; exits non-zero if a requested ticker fails
jupyter notebook notebooks/us_pipeline.ipynb   # reproduces the pipeline from saved artifacts
```
Needs internet (`yfinance`). The first CNN-LSTM prediction caches the TRAIN-only scaler to `models/cnn_lstm_us_scaler.npz`.

**Phase 2 (desktop app):** the planned PySide6 app should call `predict_us.predict_ticker()` and show it as a scanner, with the limitations above visible.

## Future work — Phase 2 (planned, not implemented)

A desktop app (**PySide6**) in `app/` that loads the trained ensemble and shows live next-day forecasts. Further modelling ideas: more features (market index, news sentiment), walk-forward validation, and multiple tickers.
