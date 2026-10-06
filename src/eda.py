"""VALIDATE + EDA (course ch03: distribution, centre & spread, relationships, correlation) -> reports/eda_report.txt.

Usage: python -m src.eda      (after src.build_dataset and the build_features job)
"""
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import BASE, DATASET  # noqa: E402
from config import FEATURES as FEATURES_FILE  # noqa: E402
from src.models.common import FEATURES  # noqa: E402

REPORTS = BASE / "reports"
CLEAN = DATASET
FEATS = FEATURES_FILE


def stats(df: pl.DataFrame, col: str, by: str | None = None) -> pl.DataFrame:
    g = pl.col(col)
    agg = [pl.len().alias("n"), g.mean().alias("mean"), g.median().alias("median"), g.std().alias("std"), g.skew().alias("skew"),
           g.kurtosis().alias("exc_kurt"), g.quantile(0.01).alias("p01"), g.quantile(0.99).alias("p99")]
    return (df.group_by(by).agg(agg).sort(by) if by else df.select(agg)).with_columns(gap=(pl.col("mean") - pl.col("median")) / pl.col("std"))


def fmt_stats(t: pl.DataFrame, label: str) -> list[str]:
    L = [f"  {label:8}{'rows':>12}{'mean':>11}{'median':>11}{'std':>10}{'(mean-med)/std':>16}{'skew':>9}{'ex.kurt':>9}{'p01':>10}{'p99':>10}"]
    for r in t.iter_rows(named=True):
        L.append(f"  {str(r.get(label.strip(), 'all')):8}{r['n']:>12,}{r['mean']:>11.5f}{r['median']:>11.5f}{r['std']:>10.5f}{r['gap']:>16.3f}{r['skew']:>9.2f}{r['exc_kurt']:>9.1f}{r['p01']:>10.4f}{r['p99']:>10.4f}")
    return L


def main() -> None:
    L = ["EquityVista - VALIDATE + EDA report (src/eda.py).  Cleaning rules and counts: reports/clean_report.txt, ../REPORT/DATA_RULES.md",
         "=" * 110]
    yc = pl.read_csv(REPORTS / "clean_year_counts.csv")
    lf = pl.scan_parquet(CLEAN)
    # ---- 1. per-year rows / tickers before -> after (-> features) ----
    ft = pl.scan_parquet(FEATS).group_by(pl.col("date").dt.year().alias("year")).agg(f_rows=pl.len(), f_tickers=pl.col("ticker").n_unique()).collect()
    t = yc.join(ft, on="year", how="left").sort("year")
    L += ["", "1. PER-YEAR COUNTS  (before = raw union; after = cleaned parquet; features = after the 60-row warm-up per ticker, finite features)", "-" * 110,
          f"  {'year':6}{'rows before':>13}{'rows after':>12}{'tickers before':>16}{'tickers after':>15}{'feat rows':>12}{'feat tickers':>14}"]
    L += [f"  {r['year']:<6}{r['rows_before']:>13,}{r['rows_after']:>12,}{r['tickers_before']:>16,}{r['tickers_after']:>15,}{r['f_rows'] or 0:>12,}{r['f_tickers'] or 0:>14,}"
          for r in t.iter_rows(named=True)]
    L.append(f"  {'total':6}{t['rows_before'].sum():>13,}{t['rows_after'].sum():>12,}{'':>16}{'':>15}{t['f_rows'].sum():>12,}")
    # ---- 2. centre & spread ----
    d = lf.select("ticker", "date", ret=(pl.col("adj_close") / pl.col("adj_close").shift(1).over("ticker")).log(),
                  hl=(pl.col("adj_high") - pl.col("adj_low")) / pl.col("adj_close")).drop_nulls("ret").with_columns(
        decade=(pl.col("date").dt.year() // 10 * 10)).collect()
    L += ["", "2. CENTRE & SPREAD of the CLEANED data (course: mean vs median - if mean != median the distribution is skewed; heavy tails -> prefer median/robust stats)", "-" * 110]
    for col, name in (("ret", "daily log return"), ("hl", "intraday range (high-low)/close")):
        L += [f" {name}: all years"] + fmt_stats(stats(d, col), "all") + [f" {name}: by decade"] + fmt_stats(stats(d, col, "decade"), "decade")
    f = pl.scan_parquet(FEATS).select("ret1", "hl_range").collect()
    L += ["", " After winsorization (feature values in features.parquet: ret1 = log return, hl_range) - the mean moves toward the median:"]
    L += fmt_stats(stats(f, "ret1"), "all") + fmt_stats(stats(f, "hl_range"), "all")
    L += ["", " Reading: (mean-median)/std measures skew in std units.",
          " - Daily log return: median 0 (many unchanged closes), mean ~0.0001 -> the centre is symmetric, but excess kurtosis in the thousands: a few moves dominate the std.",
          "   Winsorizing at the date's 1%/99% cuts it from ~9,000 to ~8 and std from 0.054 to 0.028 (loss of linear models is no longer decided by a handful of rows).",
          " - Intraday range: mean 0.034 > median 0.024 (gap 0.27 std, skew > 5) -> right-skewed; the MEAN overstates a typical day, so use median/MAD/IQR to judge outliers",
          "   and cap the tail (clean_report.txt section 3) instead of deleting it."]
    del d, f
    # ---- 3. correlation ----
    s = pl.scan_parquet(FEATS).select(FEATURES).collect().sample(2_000_000, seed=0)
    C = np.corrcoef(s.cast(pl.Float64).to_numpy(), rowvar=False)
    L += ["", f"3. CORRELATION of the {len(FEATURES)} model features (Pearson, random sample of {s.height:,} feature rows). Columns are numbered like the rows.", "-" * 110]
    L.append(" " * 14 + "".join(f"{i:>5}" for i in range(len(FEATURES))))
    L += [f"  {i:>2} {n:11}" + "".join(f"{C[i, j]:>5.2f}" for j in range(len(FEATURES))) for i, n in enumerate(FEATURES)]
    iu = np.triu_indices(len(FEATURES), 1)
    top = sorted(zip(np.abs(C[iu]), C[iu], iu[0], iu[1]), reverse=True)[:8]
    L += ["", " Strongest pairs: " + "; ".join(f"{FEATURES[i]}~{FEATURES[j]} {c:+.2f}" for _, c, i, j in top),
          " Reading: each raw feature is highly correlated with its own cross-sectional rank (r_*) and the vol/ma/momentum families overlap -> redundant for linear models",
          " (regularised) and harmless for trees. Nothing is dropped here: the course treats correlation as EDA, the models use all 25."]
    # ---- 4. remaining high/low > 2 and > 3 by year ----
    c = lf.sort("ticker", "date").with_columns(hl=pl.col("adj_high") / pl.col("adj_low"), y=pl.col("date").dt.year(),
                                               r=(pl.col("adj_close") / pl.col("adj_close").shift(1).over("ticker")).log().abs()).group_by("y").agg(
        rows=pl.len(), gt2=(pl.col("hl") > 2).sum(), gt3=(pl.col("hl") > 3).sum(), jump=(pl.col("r") >= 0.5).sum()).sort("y").collect()
    g = pl.scan_parquet(FEATS).with_columns(hl=pl.col("adj_high") / pl.col("adj_low"), y=pl.col("date").dt.year()).group_by("y").agg(
        gt2=(pl.col("hl") > 2).sum(), gt3=(pl.col("hl") > 3).sum()).sort("y").collect()
    tol = 1e-9  # adjusted columns are products of floats: allow rounding noise, not real contradictions
    bad = lf.select(((pl.col("adj_high") < pl.col("adj_low") * (1 - tol)) | (pl.col("adj_high") < pl.max_horizontal("adj_open", "adj_close") * (1 - tol))
                     | (pl.col("adj_low") > pl.min_horizontal("adj_open", "adj_close") * (1 + tol))).sum().alias("n"),
                    (pl.min_horizontal("adj_open", "adj_high", "adj_low", "adj_close", "adj_volume") <= 0).sum().alias("m")).collect().row(0)
    L += ["", "4. REMAINING EXTREMES (cleaned parquet / features parquet)  - high/low > 2 and > 3 by year", "-" * 110,
          f"  OHLC contradictions left: {bad[0]:,};  non-positive price/volume left: {bad[1]:,}  (both must be 0)",
          f"  {'year':6}{'rows':>12}{'hl>2 clean':>12}{'hl>3 clean':>12}{'hl>2 feat':>12}{'hl>3 feat':>12}{'|lnret|>=.5':>13}"]
    gm = dict(zip(g["y"], zip(g["gt2"], g["gt3"])))
    L += [f"  {y:<6}{n:>12,}{a:>12,}{b:>12,}{gm.get(y, (0, 0))[0]:>12,}{gm.get(y, (0, 0))[1]:>12,}{j:>13,}" for y, n, a, b, j in zip(c["y"], c["rows"], c["gt2"], c["gt3"], c["jump"])]
    L += [f"  {'total':6}{c['rows'].sum():>12,}{c['gt2'].sum():>12,}{c['gt3'].sum():>12,}{g['gt2'].sum():>12,}{g['gt3'].sum():>12,}{c['jump'].sum():>13,}",
          " |lnret|>=.5 = rows whose return vs the previous SURVIVING row is still >= 50 % in log terms: persistent level shifts after the single-pass return rule dropped",
          " the first shifted row (iterating would delete the whole post-shift series). They are kept, and their features are winsorized."]
    txt = "\n".join(L)
    (REPORTS / "eda_report.txt").write_text(txt, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
