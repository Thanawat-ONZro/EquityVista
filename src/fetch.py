import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yfinance as yf
import pandas as pd
from config import RAW_DIR, TICKER, START, END

def fetch_data(ticker, start, end, force=False):
    out = RAW_DIR / f"{ticker}.csv"
    if out.exists() and not force:
        return pd.read_csv(out, index_col=0, parse_dates=True)
    df = yf.download(ticker, start=start, end=end, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    df.to_csv(out)
    return df

if __name__ == "__main__":
    df = fetch_data(TICKER, START, END, force=True)
    print(f"Downloaded {len(df)} rows for {TICKER}")
    print(f"Columns: {list(df.columns)}")
    print(df.tail())
