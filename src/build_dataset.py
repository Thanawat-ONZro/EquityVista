"""ETL: data/RAW/kaggle/*.csv + data/RAW/yfinance/*.csv -> data/processed/dataset_1962_2026.parquet (+ reports/clean_report.txt).

Usage: python -m src.build_dataset [--force]
ONE rule set (src/clean.py) is applied to the whole 1962-2026 union; RAW is read-only; the single output parquet is rebuilt from RAW every run.
ETFs are left out (instrument rule). Delisted / penny / collapsed tickers are deliberately KEPT (survivorship bias).
"""
import argparse
import sys
import time
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import BASE, DATASET, RAW_DIR, RAW_KAGGLE, RAW_YF  # noqa: E402
from src import clean as K  # noqa: E402
from src.clean import _by_year  # noqa: E402

OUT = DATASET
REPORTS = BASE / "reports"
CRISIS = (2008, 2020)


def load_raw(stocks: Path = RAW_KAGGLE, ext: Path = RAW_YF) -> pl.DataFrame:
    """Raw CSV union, same schema Date,Open,High,Low,Close,Adj Close,Volume. Header-only extended files are skipped."""
    def scan(files: list[str], src: int) -> pl.LazyFrame:
        return pl.scan_csv(files, include_file_paths="path", infer_schema=False).select(
            pl.col("path").str.replace(r"^.*[\\/]", "").str.replace(r"\.csv$", "").alias("ticker"),
            pl.col("Date").str.to_date("%Y-%m-%d", strict=False).alias("date"),
            *[pl.col(a).cast(pl.Float64, strict=False).alias(b) for a, b in
              [("Open", "open"), ("High", "high"), ("Low", "low"), ("Close", "close"), ("Adj Close", "adj"), ("Volume", "vol")]],
            pl.lit(src, dtype=pl.Int8).alias("src"))

    parts = [scan([str(p) for p in sorted(stocks.glob("*.csv"))], 0)]
    ext_files = [str(p) for p in sorted(ext.glob("*.csv")) if p.stat().st_size > 40]
    if ext_files:  # ponytail: ~2.5 GB collected in RAM at once; chunk by ticker group if RAM shrinks.
        parts.append(scan(ext_files, 1))
    return pl.concat(parts).collect()


def decade(y: int) -> str:
    return f"{y // 10 * 10}s"


def by_decade(by_year: dict) -> dict:
    out: dict = {}
    for y, n in by_year.items():
        out[decade(y)] = out.get(decade(y), 0) + n
    return dict(sorted(out.items()))


def outlier_section(df: pl.DataFrame) -> list[str]:
    """Flag (MAD, IQR) and cap (percentile) per date on daily log return and intraday range; report what deleting would cost."""
    p = df.select("date", "ticker", ret1=K.ret1_expr(), hl=K.hl_expr()).drop_nulls("ret1")
    n = p.height
    L = [f"Cross-sectional, per date, on {n:,} rows (daily log return `ret1`, intraday range `hl_range` = (high-low)/close).",
         f"Flag A: |x - median| > {K.MAD_K:g} * 1.4826 * MAD (robust z; MAD ignores the tails, unlike mean/std).",
         f"Flag B: Tukey IQR rule, outside [Q1 - {K.IQR_K:g}*IQR, Q3 + {K.IQR_K:g}*IQR].",
         f"Cap: features are winsorized at the same date's {K.WINSOR[0]:.0%} / {K.WINSOR[1]:.0%} percentiles (src/clean.winsorize, applied in the features step).",
         "Decision: KEEP + CAP, never delete. Flagged rows already passed every error rule (valid OHLC, plausible move), so they are real prints that carry",
         "information; capping only limits their pull on the squared-error loss of the linear models. Detection is cross-sectional: a market-wide crash day moves the",
         "date's median with it, so 2008/2020 are NOT over-flagged - what is flagged are idiosyncratic extremes (small caps, gaps, news).",
         "", f"{'':10}{'flag MAD':>12}{'flag IQR':>12}{'MAD or IQR':>12}{'capped (p01/p99)':>18}{'rows if deleted':>17}  tickers hit"]
    crisis_share = {}
    for col in ("ret1", "hl"):
        f = K.flags(col)
        r = p.select("date", "ticker", mad=f["mad"], iqr=f["iqr"], w=f["winsor"]).with_columns(any_=pl.col("mad") | pl.col("iqr"))
        c = r.select(pl.col("mad").sum(), pl.col("iqr").sum(), pl.col("any_").sum(), pl.col("w").sum()).row(0)
        hit = r.filter("any_")
        L.append(f"{col:10}{c[0]:>12,}{c[1]:>12,}{c[2]:>12,}{c[3]:>18,}{c[2]:>17,}  {hit['ticker'].n_unique():,}")
        yr = hit.group_by(pl.col("date").dt.year().alias("y")).len().join(
            p.group_by(pl.col("date").dt.year().alias("y")).len().rename({"len": "n"}), on="y").with_columns(share=pl.col("len") / pl.col("n"))
        crisis_share[col] = (yr.filter(pl.col("y").is_in(CRISIS))["share"].mean(), c[2] / n, yr.sort("share", descending=True).head(5))
    L += ["", "Share of rows flagged (MAD or IQR): crisis years vs all years:"]
    for col, (cr, al, top) in crisis_share.items():
        L.append(f"  {col:5} {CRISIS[0]}/{CRISIS[1]} mean {cr:.1%}  vs  all years {al:.1%};  top-5 years: " +
                 ", ".join(f"{y} ({s:.0%})" for y, s in zip(top['y'], top['share'])))
    return L


def universe_section(df: pl.DataFrame) -> list[str]:
    u = K.add_universe(df.select("date", "ticker", "adj_close", "adj_volume")).rename({"in_universe": "u"}).with_columns(dec=(pl.col("date").dt.year() // 10 * 10))
    t = u.group_by("dec").agg(rows=pl.len(), out=(~pl.col("u")).sum()).sort("dec")
    L = [f"Filter (OFF by default; features column `in_universe`, train jobs param `universe`=1): trailing {K.UNIVERSE_WINDOW}-day median dollar volume",
         f"(adj_close*adj_volume) >= the same date's own {K.UNIVERSE_Q:.0%} percentile across tickers. Only data <= t. Rows it would remove per decade",
         "(includes the first 59 rows of every ticker, which cannot be assessed):"]
    L += [f"  {d}s: {o:>10,} of {n:>10,}  ({o / n:.1%})" for d, n, o in zip(t["dec"], t["rows"], t["out"])]
    L.append(f"  total: {t['out'].sum():,} of {t['rows'].sum():,}  ({t['out'].sum() / t['rows'].sum():.1%})")
    return L


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    if OUT.exists() and not ap.parse_args().force:
        print(f"{OUT} exists ({OUT.stat().st_size / 1e6:.0f} MB); use --force to rebuild.")
        return
    REPORTS.mkdir(exist_ok=True)
    t0 = time.time()
    raw = load_raw()
    print(f"loaded {raw.height:,} rows / {raw['ticker'].n_unique():,} tickers in {time.time() - t0:.0f}s", flush=True)
    meta = pl.read_csv(RAW_DIR / "symbols_valid_meta.csv", infer_schema_length=0)
    # --- INSPECT ---
    nulls = {c: raw[c].null_count() for c in ["date", *K.FIELDS]}
    nz = {"open <= 0": (raw["open"] <= 0).sum(), "volume == 0": (raw["vol"] == 0).sum()}
    o, h, lo, c = (pl.col(x) for x in ("open", "high", "low", "close"))
    used_ok = pl.all_horizontal(*[pl.col(f) > 0 for f in K.USED_PRICES])
    consistent = (h >= lo) & (h >= c) & (lo <= c) & (~(o > 0) | ((h >= o) & (lo <= o)))
    eligible = raw.filter(pl.col("date").is_not_null() & ~pl.col("ticker").is_in(list(K.excluded_symbols(meta))))  # rows that reached those rules
    saved_open = eligible.filter((o <= 0) & used_ok & pl.col("vol").is_not_null())                                # old rule dropped these for the open alone
    saved_range = eligible.filter(used_ok & (o > 0) & (pl.col("vol") > 0) & consistent & (h / lo >= 3))  # old range rule's rows (1,453 before)
    src_n = raw.group_by("src").agg(rows=pl.len(), tickers=pl.col("ticker").n_unique(), a=pl.col("date").min(), b=pl.col("date").max()).sort("src")
    n_dup = raw.height - raw.unique(["ticker", "date", "src"]).height
    n_ov = raw.filter(pl.col("src") == 1).join(raw.filter(pl.col("src") == 0), on=["ticker", "date"]).height
    df, log, extra = K.clean(raw, K.excluded_symbols(meta))
    del raw
    df.write_parquet(OUT, compression="zstd")
    chk = pl.read_parquet(OUT)

    # --- per-year before/after + per-rule per-year CSVs ---
    years = sorted(log.start["by_year"])
    after = chk.group_by(pl.col("date").dt.year().alias("y")).agg(rows=pl.len(), tickers=pl.col("ticker").n_unique()).sort("y")
    rows_after = dict(zip(after["y"], after["rows"]))
    long = [{"year": y, "rule": s["name"], "group": s["group"], "rows_removed": n} for s in log.steps for y, n in s["by_year"].items()]
    pl.DataFrame(long).sort("year", "group").write_csv(REPORTS / "clean_by_year.csv")
    extra["seam"]["table"].write_csv(REPORTS / "seam_report.csv")

    S = log.steps
    L = ["EquityVista data-cleaning report - ONE uniform rule set over 1962-2026 (src/clean.py, src/build_dataset.py)",
         "Course frame (ch03): Inspect -> Clean (missing, duplicates, errors, outliers) -> Transform -> Validate. Rules: ../REPORT/DATA_RULES.md.",
         "=" * 100, "", "1. INSPECT (raw union, nothing removed yet)", "-" * 100]
    for r in src_n.iter_rows(named=True):
        L.append(f"  source {'kaggle' if r['src'] == 0 else 'yfinance':20} rows {r['rows']:>11,}  tickers {r['tickers']:>6,}  {r['a']} -> {r['b']}")
    L += [f"  union rows {log.start['rows']:,}  tickers {log.start['tickers']:,}",
          "  nulls per column: " + ", ".join(f"{k}={v:,}" for k, v in nulls.items()),
          "  zero/negative:    " + ", ".join(f"{k}={v:,}" for k, v in nz.items()),
          f"  duplicate (ticker,date) inside one source: {n_dup:,};  dates present in both sources (the 2020 overlap): {n_ov:,}",
          "", "2. CLEAN - rows removed per rule (sequential: a row is counted under the first rule it breaks)", "-" * 100,
          f"  {'rule':92}{'rows removed':>14}{'share':>8}{'rows left':>13}{'tickers left':>14}", f"  {'(start)':92}{'':>14}{'':>8}{log.start['rows']:>13,}{log.start['tickers']:>14,}"]
    for s in S:
        L.append(f"  {s['group'] + ': ' + s['name']:92.92}{s['rows']:>14,}{s['rows'] / log.start['rows']:>8.2%}{s['rows_after']:>13,}{s['tickers_after']:>14,}")
    tot = sum(s["rows"] for s in S)
    L += [f"  {'TOTAL removed':92}{tot:>14,}{tot / log.start['rows']:>8.2%}{chk.height:>13,}{chk['ticker'].n_unique():>14,}",
          f"  check: start {log.start['rows']:,} - removed {tot:,} = {log.start['rows'] - tot:,}  (parquet has {chk.height:,}) -> {'OK' if log.start['rows'] - tot == chk.height else 'MISMATCH'}",
          f"  tickers: {log.start['tickers']:,} -> {chk['ticker'].n_unique():,}; every ticker lost = excluded instrument or < {K.MIN_ROWS} clean rows. No penny/delisted screen exists.",
          "", "  MISSING per decade (share of the raw rows of that decade; decision: DROP - few rows, no pattern to impute; median-imputing a price/volume would invent data):"]
    miss = next(s for s in S if s["group"] == "MISSING")
    md, td = by_decade(miss["by_year"]), by_decade(log.start["by_year"])
    L += [f"    {d}: {md.get(d, 0):>9,} of {td[d]:>10,}  ({md.get(d, 0) / td[d]:.2%})" for d in td]
    L += ["", "  ERRORS per decade (rows removed):"]
    for s in S:
        if s["group"] == "ERRORS":
            L.append(f"    {s['name'][:80]:80} " + "  ".join(f"{d}:{n:,}" for d, n in by_decade(s["by_year"]).items()))
    n_open, n_open_vol = saved_open.height, saved_open.filter(pl.col("vol") > 0).height
    hl3 = chk.filter(pl.col("adj_high") / pl.col("adj_low") > 3).group_by(pl.col("date").dt.year().alias("y")).len()
    L += ["", "  RULE CHANGES vs the first version (both were unjustified; measured on the raw union, before the later rules):",
          f"    (a) non-positive price now checks only high/low/close/adj close (the open feeds no feature). Rows with open <= 0 and valid other prices, kept instead of dropped: "
          f"{n_open:,} ({n_open_vol:,} with volume > 0; the rest still fall to the zero-volume rule). Their adj_open is stored as null. Per decade: "
          + ", ".join(f"{d}:{n:,}" for d, n in by_decade(_by_year(saved_open)).items()),
          f"    (b) the session range high/low >= 3 rule is gone; only internally impossible bars are dropped. Rows the old rule dropped that are internally consistent and now stay: {saved_range.height:,}"
          f" (2008/2009/2020: {sum(_by_year(saved_range).get(y, 0) for y in (2008, 2009, 2020)):,}). They are capped by the per-date winsorization of hl_range in the features step. Per year: "
          + ", ".join(f"{y}:{n:,}" for y, n in sorted(_by_year(saved_range).items())),
          f"    high/low > 3 in the cleaned parquet (kept rows), per year: " + (", ".join(f"{y}:{n:,}" for y, n in sorted(zip(hl3['y'], hl3['len']))) or "none") + f"  (total {hl3['len'].sum():,})"]
    sm = extra["seam"]
    t = sm["table"]
    L += ["", "  SEAM (yfinance adj close is as-of-download; Kaggle's as-of-2020): r = median(new/base adj close over overlap dates), per ticker; base rows x r.",
          f"    tickers with an overlap: {t.height:,};  rebased (r != 1): {sm['rebased']:,};  tickers in both sources with NO overlap (left unscaled, seam jump possible): {len(sm['no_overlap'])}",
          f"    tickers where some per-date ratio disagrees with r by > {K.SEAM_TOL:.0%}: {t.filter(pl.col('n_flagged') > 0).height:,}  -> those dates are errors (rule above, dropped on both sides). Full list: reports/seam_report.csv. Worst 10:"]
    L += [f"      {r['ticker']:8} r={r['r']:.4f} overlap={r['n_overlap']:>3} flagged={r['n_flagged']:>3} max_dev={r['max_dev']:.1%}" for r in t.head(10).iter_rows(named=True)]
    L += [f"    IDENTITY BREAKS: {len(sm['split'])} tickers where > {K.SEAM_SPLIT:.0%} of the overlap dates disagree (the symbol was reused by another company). Nothing deleted:",
          "    the yfinance rows are kept as a separate ticker '<SYM>~2' and the old rows are not rebased. " + ", ".join(sm["split"])]
    sh = extra["short"]
    kept = chk.group_by("ticker").agg(n=pl.len(), dv=(pl.col("adj_close") * pl.col("adj_volume")).median())
    shg = sh.group_by("ticker").agg(n=pl.len(), dv=(pl.col("adj_close") * pl.col("adj_volume")).median())
    L += ["", f"  MIN HISTORY: {shg.height:,} tickers removed ({sh.height:,} rows): median {shg['n'].median():.0f} clean rows each (kept tickers: {kept['n'].median():,.0f}),",
          f"    median daily dollar volume ${shg['dv'].median():,.0f} (kept: ${kept['dv'].median():,.0f}) -> tiny / short-lived names that cannot yield labelled rows after the 60d warm-up + 21d label.",
          "", "3. OUTLIERS (flag + cap, NOT deleted)", "-" * 100, *outlier_section(chk),
          "", "4. UNIVERSE FILTER (training-step option, default OFF)", "-" * 100, *universe_section(chk),
          "", "5. TRANSFORM: z-score (StandardScaler) for Linear/Logistic only, fit on the TRAIN split only (src/jobs/train_*.py); trees are scale-invariant.",
          "", "6. PER-YEAR COUNTS (rows / tickers before -> after)", "-" * 100,
          f"  {'year':6}{'rows before':>13}{'rows after':>12}{'removed':>11}{'tickers before':>16}{'tickers after':>15}"]
    tk = dict(zip(after["y"], after["tickers"]))
    L += [f"  {y:<6}{log.start['by_year'][y]:>13,}{rows_after.get(y, 0):>12,}{log.start['by_year'][y] - rows_after.get(y, 0):>11,}{log.start['tk_by_year'][y]:>16,}{tk.get(y, 0):>15,}" for y in years]
    L += ["", "Per-rule per-year counts: reports/clean_by_year.csv", f"parquet: {OUT} ({OUT.stat().st_size / 1e6:.0f} MB)   runtime: {time.time() - t0:.0f}s"]
    txt = "\n".join(L)
    (REPORTS / "clean_report.txt").write_text(txt, encoding="utf-8")
    pl.DataFrame({"year": years, "rows_before": [log.start["by_year"][y] for y in years], "rows_after": [rows_after.get(y, 0) for y in years],
                  "tickers_before": [log.start["tk_by_year"][y] for y in years],
                  "tickers_after": [tk.get(y, 0) for y in years]}).write_csv(REPORTS / "clean_year_counts.csv")
    print(txt)


if __name__ == "__main__":
    main()
