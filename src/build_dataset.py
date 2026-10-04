"""ETL: data/RAW/stocks/*.csv -> data/processed/us_market.parquet (+ etl_report.txt).

Usage: python -m src.build_dataset [--force]
ETFs are deliberately left out of the main parquet (optional market features later).
"""
import argparse
import sys
import time
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import PROCESSED_DIR, RAW_DIR  # noqa: E402

OUT = PROCESSED_DIR / "us_market.parquet"
REPORT = PROCESSED_DIR / "etl_report.txt"
MIN_ROWS, MIN_PRICE, MIN_VOL, MAX_ABS_RET = 500, 5.0, 100_000, 0.5
# Warrants / rights / unit bundles in the Security Name. "Common Units" (MLPs) are real equities -> kept.
NON_EQUITY = r"(?i)\bwarrants?\b|\brights?\b|\bunits?,? each\b|\bunits? consisting\b"


def excluded_symbols() -> tuple[set[str], int]:
    meta = pl.read_csv(RAW_DIR / "symbols_valid_meta.csv", infer_schema_length=0)
    bad = meta.filter(
        (pl.col("ETF") != "N") | (pl.col("Test Issue") != "N") | pl.col("Security Name").str.contains(NON_EQUITY)
    )
    return set(bad["Symbol"]), bad.height


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    if OUT.exists() and not ap.parse_args().force:
        print(f"{OUT} exists ({OUT.stat().st_size / 1e6:.0f} MB); use --force to rebuild.")
        return

    t0 = time.time()
    drops: dict[str, int] = {}
    # ponytail: ~2.5 GB of flat floats collected in RAM at once (25 GB box); chunk by ticker group if RAM shrinks.
    raw = (
        pl.scan_csv(
            str(RAW_DIR / "stocks" / "*.csv"),
            include_file_paths="path",
            infer_schema=False,
        )
        .select(
            pl.col("path").str.replace(r"^.*[\\/]", "").str.replace(r"\.csv$", "").alias("ticker"),
            pl.col("Date").str.to_date("%Y-%m-%d", strict=False).alias("date"),
            pl.col("Open").cast(pl.Float64, strict=False),
            pl.col("High").cast(pl.Float64, strict=False),
            pl.col("Low").cast(pl.Float64, strict=False),
            pl.col("Close").cast(pl.Float64, strict=False),
            pl.col("Adj Close").cast(pl.Float64, strict=False).alias("adj"),
            pl.col("Volume").cast(pl.Float64, strict=False).alias("vol"),
        )
        .filter(pl.col("date").is_not_null())
        .collect()
    )
    rows0, tick0 = raw.height, raw["ticker"].n_unique()
    print(f"loaded {rows0:,} rows / {tick0:,} tickers in {time.time() - t0:.0f}s")

    # --- instrument exclusion (meta) ---
    ex, n_meta_bad = excluded_symbols()
    df = raw.filter(~pl.col("ticker").is_in(list(ex)))
    drops["instruments excluded via meta (warrant/right/unit/test/ETF)"] = rows0 - df.height
    tick_meta = df["ticker"].n_unique()
    del raw

    # --- row cleaning, sequential: each row is counted under the first rule it breaks ---
    def rule(name: str, keep: pl.Expr) -> None:
        nonlocal df
        n = df.height
        df = df.filter(keep)
        drops[name] = n - df.height

    rule("Open <= 0 or null", pl.col("Open") > 0)
    rule("Adj Close null/<=0 or Close <= 0", pl.col("adj").is_not_null() & (pl.col("adj") > 0) & (pl.col("Close") > 0))
    rule("zero/null Volume", pl.col("vol") > 0)
    rule("High/Low null", pl.col("High").is_not_null() & pl.col("Low").is_not_null())

    # adjust, sort, then 1-day-return filter on adj_close (single pass)
    f = pl.col("adj") / pl.col("Close")
    df = df.select(
        "ticker", "date",
        (pl.col("Open") * f).alias("adj_open"),
        (pl.col("High") * f).alias("adj_high"),
        (pl.col("Low") * f).alias("adj_low"),
        pl.col("adj").alias("adj_close"),
        # volume scales inversely: split 2:1 -> price halves, shares double
        (pl.col("vol") / f).alias("adj_volume"),
        pl.col("Close").alias("raw_close"),
    ).sort("ticker", "date")
    df = df.unique(["ticker", "date"], keep="last", maintain_order=True)  # duplicate dates, if any
    rule("|1d return adj_close| >= 50%", ((pl.col("adj_close") / pl.col("adj_close").shift(1).over("ticker") - 1).abs() < MAX_ABS_RET).fill_null(True))

    # --- ticker-level screens ---
    st = df.group_by("ticker").agg(
        pl.len().alias("n"), pl.col("adj_close").median().alias("mp"), pl.col("adj_volume").median().alias("mv")
    )
    short = st.filter(pl.col("n") < MIN_ROWS)["ticker"]
    penny = st.filter((pl.col("n") >= MIN_ROWS) & ((pl.col("mp") < MIN_PRICE) | (pl.col("mv") < MIN_VOL)))["ticker"]
    n = df.height
    df = df.filter(~pl.col("ticker").is_in(short.to_list()))
    drops[f"tickers < {MIN_ROWS} rows ({len(short):,} tickers)"] = n - df.height
    n = df.height
    df = df.filter(~pl.col("ticker").is_in(penny.to_list()))
    drops[f"penny screen: median adj close < ${MIN_PRICE:.0f} or median adj vol < {MIN_VOL // 1000}k ({len(penny):,} tickers)"] = n - df.height

    df = df.sort("ticker", "date")
    df.write_parquet(OUT, compression="zstd")

    # --- report ---
    chk = pl.read_parquet(OUT)
    lines = [
        "EquityVista ETL data-quality report",
        "=" * 60,
        f"rows before (raw stocks/*.csv): {rows0:,}   tickers: {tick0:,}",
        f"rows after  (us_market.parquet): {chk.height:,}   tickers: {chk['ticker'].n_unique():,}",
        f"tickers after meta exclusion: {tick_meta:,} ({n_meta_bad} meta symbols flagged ETF/test/warrant/right/unit)",
        f"date range: {chk['date'].min()} -> {chk['date'].max()}",
        "",
        "Rows dropped per rule (sequential, first rule hit):",
        *[f"  {v:>12,}  {k}" for k, v in drops.items()],
        f"  {rows0 - chk.height:>12,}  TOTAL (check: {sum(drops.values()):,})",
        "",
        "Notes:",
        "  - Prices adjusted by Adj Close / Close; volume divided by same factor (split-consistent shares).",
        "  - Penny/liquidity screen is a per-ticker whole-history median: an ETL-level surrogate. Strict per-date",
        "    trailing 60d dollar-volume/min-price universe rules (no look-ahead) are applied in the features step.",
        "  - SURVIVORSHIP BIAS: snapshot ends ~2020-04 and contains almost only tickers alive then; delisted names are",
        "    missing. Not fixable here; disclose and always compare against same-period baselines.",
        "  - Return filter is one pass (a bad print and its snap-back are both dropped; no iteration).",
        "  - ETFs are excluded from the main parquet.",
        "",
        "Head:",
        str(chk.head(5)),
        f"\nparquet: {OUT} ({OUT.stat().st_size / 1e6:.0f} MB)   runtime: {time.time() - t0:.0f}s",
    ]
    txt = "\n".join(lines)
    REPORT.write_text(txt, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
