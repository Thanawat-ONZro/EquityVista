import numpy as np
import pandas as pd

def add_indicators(df):
    x = df.copy()
    x["Return"] = x["Close"].pct_change()
    x["MA20"] = x["Close"].rolling(20).mean()
    x["MA50"] = x["Close"].rolling(50).mean()
    delta = x["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    x["RSI"] = 100 - (100 / (1 + rs))
    ema12 = x["Close"].ewm(span=12, adjust=False).mean()
    ema26 = x["Close"].ewm(span=26, adjust=False).mean()
    x["MACD"] = ema12 - ema26
    x["Signal"] = x["MACD"].ewm(span=9, adjust=False).mean()
    return x.dropna()

def make_sequences(df, window):
    feats = df[["Close", "Return", "MA20", "MA50", "RSI", "MACD"]].values
    X, y = [], []
    for i in range(window, len(feats)):
        X.append(feats[i - window : i])
        y.append(feats[i, 0])  # next-day Close
    return np.array(X), np.array(y)
