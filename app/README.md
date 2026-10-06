# EquityVista desktop app

One page: **Predict** (candlestick chart + H-day forecast panel, backtest with reveal). It uses the active run
(`models/active.json`: the run with the best validation rank-IC) through `src/predict.py` and shows that run's test metrics next to the other model types.

The app trains nothing. Training runs from the command line, from the repo root:
```
python train_all.py demo --seeds 5  # ~36 min (one seed: ~7 min): trains the 3 classifiers, compares, activates the best on validation
python train_all.py demo --fetch  # first refresh the 2020+ prices from yfinance and rebuild the dataset + features
```
Then start (or restart) the app:
```
python app/main.py
```
Predict -> ticker + days (21 trading days = 1 month, about 30 calendar days; a horizon needs its own trained model) -> **Predict**.

The former Training Hub pages (Dashboard / Jobs / Models / Settings) are archived in `../REPORT/archive/app_training_hub/`.

*Educational, not financial advice; measured: under 2 points of direction accuracy above always-UP.*
