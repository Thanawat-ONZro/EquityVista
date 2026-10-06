# EquityVista — will a US stock be higher next month? (direction classification)

Educational ML project (course 240-318 Artificial Intelligence and Machine Learning): for every US stock, predict whether its price
will be **higher one month from now** (21 trading days ≈ 30 calendar days) from 25 scale-free price/volume features, with three
classifiers compared on the same base, and judge them honestly against the "always UP" baseline. *Not financial advice.*

## What it does

1. **Data:** Kaggle US stock CSVs (1962-2020, CC0) + yfinance (2020-2026). Columns: `Date, Open, High, Low, Close, Adj Close, Volume`.
   One uniform cleaning standard (9 rules) over the whole window: 29,503,886 raw rows → **27,627,816 rows, 5,122 tickers**
   (`data/processed/dataset_1962_2026.parquet`). Outliers are capped (per-date 1 %/99 %), not deleted.
2. **Features:** 25 per ticker-day (13 per-row + 12 same-date percentile ranks), no prices/ticker/date as input (`src/features.py`).
3. **Target:** up = `ln(close[t+21] / close[t]) > 0` (`src/features.add_targets`).
4. **Split (fixed calendar):** train ..2021, validation 2022-2023, test 2024 → end of data; **60-trading-day embargo**;
   20 % of tickers held out as "unseen" (`src/split.py`). Every choice is made on validation; the test split chooses nothing.
5. **Models, all classifiers that output P(up)** (same target, features, split, training size and metrics): Logistic Regression,
   Random Forest, LightGBM. Training size (~1M rows), feature set and hyper-parameters were chosen by validation experiments
   (`../REPORT/figures/experiments.json`).
6. **Metrics** (`src/metrics.py`): confusion matrix, accuracy vs "always UP", precision, recall, F1, balanced accuracy, AUC,
   rank-IC per day with a 95 % interval, calibration (ECE); each also on the unseen tickers.
7. **Desktop app** (PySide6, `app/`): one Predict page: UP/DOWN with the model's chance, the active run's test metrics next to the other
   model types, and a backtest with reveal. The active run = the one with the best **validation** rank-IC (`models/active.json`).

## Results

`python train_all.py demo --seeds 5` (5 seeds per model, 36 min in total). Test = 2024-01-02 → 2026-09-03, 1,686,875 rows; mean ± sd over the seeds:

| Model | accuracy % | precision % | recall % | F1 % | AUC | rank-IC validation | rank-IC test | rank-IC unseen tickers | ECE |
|---|---|---|---|---|---|---|---|---|---|
| Baseline: always UP | 53.0% | 53.0% | 100.0 | 69.3% | 0.500 | – | – | – | – |
| LightGBM (used by the app) | 54.7 ± 0.1 | 54.7 ± 0.3 | 84.0 ± 3.4 | 66.2 ± 0.8 | 0.551 ± 0.002 | 0.083 ± 0.001 | 0.064 ± 0.001 | 0.069 ± 0.001 | 0.014 ± 0.001 |
| Random Forest | 54.9 ± 0.0 | 54.9 ± 0.1 | 83.2 ± 0.5 | 66.2 ± 0.1 | 0.553 ± 0.000 | 0.083 ± 0.000 | 0.065 ± 0.000 | 0.070 ± 0.001 | 0.014 ± 0.001 |
| Logistic Regression | 54.7 ± 0.0 | 54.7 ± 0.0 | 84.1 ± 0.4 | 66.3 ± 0.1 | 0.554 ± 0.000 | 0.078 ± 0.000 | 0.062 ± 0.000 | 0.067 ± 0.000 | 0.011 ± 0.000 |

Seed-42 run, 95 % interval from the daily values of the test split:

| Model | test rank-IC | accuracy above always-UP, points per day | train time |
|---|---|---|---|
| LightGBM | +0.065 [+0.054, +0.075] | +1.57 [+1.10, +2.04] | 83 s |
| Random Forest | +0.065 [+0.055, +0.076] | +1.73 [+1.29, +2.17] | 319 s |
| Logistic Regression | +0.062 [+0.050, +0.073] | +1.56 [+1.15, +1.97] | 24 s |

Every model beats "always UP" by 1.7-1.8 points of pooled accuracy with an interval above zero, every P(up) passes the
calibration check, and the three models are within noise of each other: the ceiling comes from the features, not from the model.
Full report: `../REPORT/EquityVista_Report.pdf`.

**Limitations:** survivorship bias (1,751 of the 5,112 tickers have no post-2020 data, so validation/test hold only survivors);
one test regime (2024-2026), one split (no walk-forward); price/volume features only; live predictions take `mkt_ret` from a
40-stock large-cap basket (`src/predict.py::REFERENCE`).

## Repository structure

```
config.py                 the ONE place for data paths: RAW_KAGGLE, RAW_YF, DATASET, FEATURES
src/
  extend_fetch.py         yfinance -> data/RAW/yfinance/*.csv
  clean.py                ONE cleaning standard (+ outlier flags, winsorize, universe flag)
  build_dataset.py        RAW union -> data/processed/dataset_1962_2026.parquet (+ reports/clean_report.txt)
  eda.py                  validation report (reports/eda_report.txt)
  features.py             25 features + target + no-look-ahead test
  split.py                calendar split, embargo, unseen tickers
  metrics.py              confusion matrix, accuracy/precision/recall/F1, AUC, rank-IC with intervals, calibration, threshold
  active.py               models/active.json (active run per horizon)
  predict.py              forecast for any ticker (CLI and app)
  models/                 common.py (features list, loader), lgbm_model.py
  jobs/                   one CLI module per job: data_extend, build_features, train_{logistic,rf,lgbm}, evaluate
app/                      PySide6 Predict app (see app/README.md)
train_all.py              one command: features -> 3 models x N seeds -> compare -> activate the best on validation
tests/                    pytest suite (54 tests)
```
Generated, git-ignored: `data/RAW/`, `data/processed/`, `runs/`, `reports/`, `models/*`.

## How to reproduce

Needs Python ≥ 3.11 and internet (Kaggle files are not in the repo; yfinance for 2020+).

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install polars pyarrow joblib PySide6 lightweight-charts psutil
```

1. **Raw data** (once): put the Kaggle CSVs in `data/RAW/kaggle/` and `symbols_valid_meta.csv` in `data/RAW/`.
2. **Fetch 2020-2026 + clean:** `python -m src.jobs.data_extend`
3. **Features:** `python -m src.jobs.build_features --params '{"leakage_test": 1}'`
4. **Train, compare, activate:** `python train_all.py demo --seeds 5` (one seed: about 7 min). Defaults of every job: `src/jobs/__init__.py`.
5. **App:** `python app/main.py`   CLI: `python -m src.predict NVDA KO`
6. **Tests:** `python -m pytest tests -q` → 54 passed.

## Documents (outside git, in `../REPORT/`)

| File | Content |
|---|---|
| `EquityVista_Report.docx/.pdf` | the course report (Thai) |
| `Mini_Project_Proposal.docx/.pdf` | the topic proposal |
| `RESULTS.md` | results of the current runs (section B) and the history before them |
| `_build/` | scripts that rebuild the figures, numbers and both documents |
