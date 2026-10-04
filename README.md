# EquityVista — Stock Trend Forecasting

An AI/ML stock trend forecaster that predicts **next-day direction** (up / down / sideway)
with a **confidence score** and a **95% prediction interval**, using an **ensemble of
ARIMA + LSTM + RandomForest** trained on real Yahoo Finance data.

## Problem Statement

Predicting stock direction is inherently noisy. A single model is unreliable, so this
project combines three complementary models and merges their outputs into one decision:

- **ARIMA** (statsmodels) — captures linear trend / seasonality.
- **LSTM** (TensorFlow/Keras) — captures non-linear momentum from technical indicators.
- **RandomForest** (scikit-learn) — classifies the up/down direction of the next day.

A weighted aggregator combines their outputs into a final **direction + confidence + 95% range**.

## Ensemble Approach

| Model      | Role                              | Output                          |
| ---------- | --------------------------------- | ------------------------------- |
| ARIMA      | Linear trend / seasonality        | Next-day price + 95% interval   |
| LSTM       | Non-linear momentum               | Next-day price                  |
| RandomForest | Direction classification          | Up / Down / Sideway prob.       |
| **Ensemble** | Weighted vote of the above       | Direction + confidence + 95% range |

Evaluation uses standard metrics: MAE / RMSE / MAPE and accuracy / confusion matrix.

## Setup

Requires Python >= 3.11. Create and activate a local virtual environment (git-bash syntax):

```bash
cd EquityVista
python -m venv .venv
source .venv/Scripts/activate   # on cmd  use: .venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run

Phase-1 pipeline lives in `src/` (fetch → features → models → ensemble → evaluate) and is
driven from the Jupyter notebooks under `notebooks/`:

```bash
jupyter notebook notebooks/EDA.ipynb
```

Run the test suite:

```bash
python -m pytest -v
```

## Status

**Phase 1** — training / evaluation pipeline (scaffolded; fetch, features, models,
ensemble, evaluation, and notebooks are the upcoming tasks). **Phase 2** (desktop GUI app
to load the trained ensemble and show live forecasts) is out of scope until Phase 1 models
are validated.