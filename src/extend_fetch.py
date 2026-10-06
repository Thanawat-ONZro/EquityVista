"""Fetch raw daily OHLCV from yfinance for the tickers in data/processed/dataset_1962_2026.parquet -> data/RAW/yfinance/<SYMBOL>.csv.

Usage: python src/extend_fetch.py [--start 2020-01-01] [--workers 6] [--out data/RAW/yfinance]

CSV columns match data/RAW/kaggle: Date,Open,High,Low,Close,Adj Close,Volume (auto_adjust=False, unmodified).
Resume: a ticker is skipped if <SYMBOL>.csv exists or it is listed in skipped.txt (no data after retry).
Failed tickers (exceptions) are in neither, so a re-run retries them.
"""
import argparse
import sys
import time
from pathlib import Path

import polars as pl
import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATASET, RAW_YF  # noqa: E402

COLS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]


def yf_symbol(s: str) -> str:
    return s.replace(".", "-")  # Nasdaq "BRK.B" -> Yahoo "BRK-B"; file keeps the base name


def clean(df):
    """Single-ticker frame -> CSV-ready frame (Date index), or None if no real rows."""
    if df is None or df.empty:
        return None
    df = df[COLS].dropna(how="all")
    if df.empty:
        return None
    df.index = df.index.tz_localize(None) if df.index.tz is not None else df.index
    df.index = df.index.strftime("%Y-%m-%d")
    df.index.name = "Date"
    return df


def fetch_batch(batch: list[str], start: str, workers: int) -> dict:
    """Bulk download (yfinance threads internally); returns {symbol: frame-or-None}."""
    syms = [yf_symbol(s) for s in batch]
    raw = yf.download(
        syms, start=start, auto_adjust=False, group_by="ticker",
        threads=workers, progress=False, actions=False,
    )
    out = {}
    for s, y in zip(batch, syms):
        try:
            out[s] = clean(raw if len(syms) == 1 else raw[y])
        except KeyError:
            out[s] = None
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", default="2020-01-01")
    ap.add_argument("--workers", type=int, default=6, help="yfinance download threads per batch")
    ap.add_argument("--out", type=Path, default=RAW_YF)
    ap.add_argument("--batch", type=int, default=50, help="tickers per bulk request")
    ap.add_argument("--pause", type=float, default=1.0, help="seconds between batches (rate-limit)")
    ap.add_argument("--base", type=Path, default=DATASET)
    ap.add_argument("--limit", type=int, default=0, help="only first N tickers (testing)")
    a = ap.parse_args()

    tickers = pl.scan_parquet(a.base).select("ticker").unique(maintain_order=True).collect()["ticker"].to_list()
    if a.limit:
        tickers = tickers[: a.limit]
    a.out.mkdir(parents=True, exist_ok=True)
    skip_file = a.out / "skipped.txt"
    skipped = set(skip_file.read_text().split()) if skip_file.exists() else set()
    todo = [t for t in tickers if t not in skipped and not (a.out / f"{t}.csv").exists()]
    print(f"{len(tickers)} tickers in base; {len(tickers) - len(todo)} already done; {len(todo)} to fetch")

    t0, fetched, failed = time.time(), 0, []
    new_skipped = []
    for i in range(0, len(todo), a.batch):
        batch = todo[i : i + a.batch]
        try:
            res = fetch_batch(batch, a.start, a.workers)
        except Exception as e:  # whole batch failed (network/rate limit) -> retry on next run
            print(f"  batch {i // a.batch} failed: {e!r}")
            failed += batch
            time.sleep(a.pause * 5)
            continue
        for s, df in res.items():
            if df is None:  # bulk can return empty on throttling: confirm once per ticker before skipping
                try:
                    df = clean(yf.Ticker(yf_symbol(s)).history(start=a.start, auto_adjust=False, actions=False))
                except Exception:
                    failed.append(s)
                    continue
            if df is None:
                new_skipped.append(s)
                with skip_file.open("a") as f:
                    f.write(s + "\n")
            else:
                df.to_csv(a.out / f"{s}.csv")
                fetched += 1
        done = i + len(batch)
        print(f"  {done}/{len(todo)}  fetched={fetched} skipped={len(new_skipped)} failed={len(failed)}  {time.time() - t0:.0f}s")
        time.sleep(a.pause)

    rows = sum(max(sum(1 for _ in p.open("rb")) - 1, 0) for p in a.out.glob("*.csv"))
    n_skip_total = len(skipped) + len(new_skipped)
    lines = [
        "EquityVista extend_fetch report",
        f"window: {a.start} -> today   out: {a.out}",
        f"base tickers: {len(tickers)}",
        f"fetched this run: {fetched}   skipped (no data) this run: {len(new_skipped)}   failed (retry on re-run): {len(failed)}",
        f"skipped total (skipped.txt): {n_skip_total}   CSV files total: {len(list(a.out.glob('*.csv')))}",
        f"total rows in CSVs: {rows:,}",
        f"failed: {' '.join(failed)}" if failed else "failed: none",
        f"runtime: {time.time() - t0:.0f}s",
    ]
    (a.out / "extend_report.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
