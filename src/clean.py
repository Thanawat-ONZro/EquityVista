"""ONE uniform cleaning standard for the whole 1962-2026 window (course ch03: Inspect -> Clean -> Transform -> Validate).

`clean(raw, excluded)` takes the union of data/RAW/kaggle + data/RAW/yfinance (columns: ticker, date, open, high, low,
close, adj, vol, src; src 0 = Kaggle, 1 = yfinance extension) and applies every rule below to EVERY row,
whatever its era. Rules are sequential: a row is counted under the first rule it breaks. Nothing is edited in place:
bad rows are dropped, outliers are kept (see `winsorize`), and every step is counted in the returned log.

Two principles: (1) a row is dropped only if a field that is USED is impossible (non-positive high/low/close/adj close/volume,
or an internally contradictory bar); a missing open (<= 0) is kept and stored as null adj_open; (2) extreme but self-consistent
rows are kept and capped by the per-date winsorization, never deleted.
Order (and why): INSTRUMENT -> MISSING (a null makes every later comparison undefined) -> ERRORS -> DUPLICATES ->
SEAM -> RETURN ERRORS (needs rebased, sorted prices) -> MIN HISTORY (needs the final row count per ticker).
"""
import polars as pl

MIN_ROWS = 500          # a ticker needs 60d warm-up + 21d label + data to learn from
MAX_ABS_LOGRET = 0.5    # |ln(adj_close_t / adj_close_t-1)| >= 0.5: single bad print (course: error, not signal)
SEAM_TOL = 0.03         # per-date new/base adj_close ratio may differ from the ticker's median ratio by 3 % (dividend-method noise is < 3 %)
SEAM_SPLIT = 0.5        # > 50 % of a ticker's overlap dates disagree -> two different securities share the symbol (B, GEN, TEN...)
MAD_K = 5.0             # outlier flag: |x - median| > K * 1.4826 * MAD, per date
IQR_K = 1.5             # outlier flag: Tukey fences, per date
WINSOR = (0.01, 0.99)   # per-date percentile bounds the FEATURES are capped at
UNIVERSE_Q, UNIVERSE_WINDOW = 0.05, 60
# Warrants / rights / unit bundles in the Security Name. "Common Units" (MLPs) are real equities -> kept.
NON_EQUITY = r"(?i)\bwarrants?\b|\brights?\b|\bunits?,? each\b|\bunits? consisting\b"
FIELDS = ["open", "high", "low", "close", "adj", "vol"]
USED_PRICES = ["high", "low", "close", "adj"]  # volume has its own (zero volume) rule below; the open is not used by any feature
OUT_COLS = ["ticker", "date", "adj_open", "adj_high", "adj_low", "adj_close", "adj_volume", "raw_close"]


def excluded_symbols(meta: pl.DataFrame) -> set[str]:
    """Meta symbols that are not common stock: ETF, test issue, warrant/right/unit by name."""
    bad = meta.filter((pl.col("ETF") != "N") | (pl.col("Test Issue") != "N") | pl.col("Security Name").str.contains(NON_EQUITY))
    return set(bad["Symbol"])


class Log:
    """Sequential rule counts: rows removed, rows/tickers left, and the removed rows per year."""

    def __init__(self, df: pl.DataFrame):
        self.steps: list[dict] = []
        self.start = {"rows": df.height, "tickers": df["ticker"].n_unique(), "by_year": _by_year(df),
                      "tk_by_year": dict(zip(*df.select(pl.col("date").dt.year().alias("y"), "ticker").drop_nulls("y").group_by("y").agg(
                          n=pl.col("ticker").n_unique()).to_dict(as_series=False).values()))}

    def add(self, name: str, group: str, removed: pl.DataFrame, left: pl.DataFrame):
        self.steps.append({"name": name, "group": group, "rows": removed.height, "tickers_removed_rows": removed["ticker"].n_unique(),
                           "rows_after": left.height, "tickers_after": left["ticker"].n_unique(),
                           "by_year": _by_year(removed)})


def _by_year(df: pl.DataFrame) -> dict[int, int]:
    d = df.select(pl.col("date").dt.year().alias("y")).drop_nulls().group_by("y").len()
    return dict(zip(d["y"], d["len"]))


def clean(raw: pl.DataFrame, excluded: set[str], min_rows: int = MIN_ROWS) -> tuple[pl.DataFrame, Log, dict]:
    """-> (clean frame with OUT_COLS sorted by ticker,date, Log, extras{seam, short_tickers})."""
    log = Log(raw)
    df = raw

    def drop(name: str, group: str, bad: pl.Expr) -> None:
        nonlocal df
        bad = bad.fill_null(False)
        log.add(name, group, df.filter(bad), df.filter(~bad))
        df = df.filter(~bad)

    o, h, lo, c, a = (pl.col(x) for x in ("open", "high", "low", "close", "adj"))
    drop("instrument not common stock (ETF/test/warrant/right/unit)", "INSTRUMENT", pl.col("ticker").is_in(list(excluded)))
    drop("missing date or any of open/high/low/close/adj close/volume", "MISSING",
         pl.any_horizontal(pl.col("date").is_null(), *[pl.col(f).is_null() for f in FIELDS]))
    # The open feeds no feature (features.py uses high, low, close, adj close, volume), so a missing/zero open (the 1970s-80s
    # coding of "no open") must not cost the row: only the fields that are used are checked; the open is only checked when present.
    drop("non-positive price (high/low/close/adj close <= 0)", "ERRORS", pl.any_horizontal(*[pl.col(f) <= 0 for f in USED_PRICES]))
    drop("zero volume", "ERRORS", pl.col("vol") <= 0)
    # Internally impossible rows only. An extreme but self-consistent range (crash day, wide quote) is KEPT: the per-date
    # winsorization of hl_range (features step) caps its influence; deleting it would also delete genuine 2008/2009/2020 days.
    has_open = o > 0
    drop("OHLC contradiction (high<low, high<max(open,close), low>min(open,close); open only if > 0)", "ERRORS",
         (h < lo) | (h < c) | (lo > c) | (has_open & ((h < o) | (lo > o))))
    drop("duplicate (ticker,date) inside one source, keep last", "DUPLICATES",
         ~pl.struct("ticker", "date", "src").is_last_distinct())

    # --- SEAM: yfinance Adj Close is as-of-download, Kaggle's is as-of-2020 -> rebase the old rows onto the new basis ---
    base, new = df.filter(pl.col("src") == 0), df.filter(pl.col("src") == 1)
    ov = base.select("ticker", "date", pl.col("adj").alias("b")).join(new.select("ticker", "date", pl.col("adj").alias("n")), on=["ticker", "date"])
    ov = ov.with_columns(q=pl.col("n") / pl.col("b")).with_columns(r=pl.col("q").median().over("ticker"))
    ov = ov.with_columns(bad=(pl.col("q") / pl.col("r") - 1).abs() > SEAM_TOL)
    seam = ov.group_by("ticker").agg(r=pl.col("r").first(), n_overlap=pl.len(), n_flagged=pl.col("bad").sum(),
                                     max_dev=(pl.col("q") / pl.col("r") - 1).abs().max()).sort("n_flagged", "max_dev", descending=True)
    split = seam.filter(pl.col("n_flagged") > SEAM_SPLIT * pl.col("n_overlap"))["ticker"]
    flagged = ov.filter(pl.col("bad") & ~pl.col("ticker").is_in(split.implode())).select("ticker", "date")
    both_src = set(base["ticker"]) & set(new["ticker"])
    seam_info = {"table": seam.with_columns(split=pl.col("ticker").is_in(split.implode())), "no_overlap": sorted(both_src - set(seam["ticker"])),
                 "rebased": seam.filter((pl.col("r") - 1).abs() > 1e-4).height - split.len(), "split": sorted(split)}
    # identity break: the yfinance rows become their own ticker "<SYM>~2" (nothing deleted, nothing rebased across two companies)
    df = df.with_columns(ticker=pl.when((pl.col("src") == 1) & pl.col("ticker").is_in(split.implode())).then(pl.col("ticker") + "~2").otherwise(pl.col("ticker")))
    seam = seam.with_columns(r=pl.when(pl.col("ticker").is_in(split.implode())).then(1.0).otherwise(pl.col("r")))
    df = df.join(flagged.with_columns(_f=pl.lit(True)), on=["ticker", "date"], how="left")
    drop(f"seam mismatch: per-date new/base adj close ratio differs >{SEAM_TOL:.0%} from the ticker's median ratio", "ERRORS", pl.col("_f"))
    df = df.drop("_f").join(seam.select("ticker", "r"), on="ticker", how="left")
    df = df.with_columns(r=pl.when(pl.col("src") == 0).then(pl.col("r").fill_null(1.0)).otherwise(1.0))
    keys = df.filter(pl.col("src") == 1).select("ticker", "date", _ov=pl.lit(True))
    df = df.join(keys, on=["ticker", "date"], how="left")
    drop("duplicate (ticker,date) across sources (overlap 2020), keep newest", "DUPLICATES", (pl.col("src") == 0) & pl.col("_ov"))
    f = pl.col("adj") / pl.col("close")
    df = df.select(
        "ticker", "date",
        (pl.when(pl.col("open") > 0).then(pl.col("open") * f * pl.col("r"))).alias("adj_open"), (pl.col("high") * f * pl.col("r")).alias("adj_high"),
        (pl.col("low") * f * pl.col("r")).alias("adj_low"), (pl.col("adj") * pl.col("r")).alias("adj_close"),
        (pl.col("vol") / f / pl.col("r")).alias("adj_volume"),  # split 2:1 -> price halves, shares double
        pl.col("close").alias("raw_close"),
    ).sort("ticker", "date")
    drop(f"|1-day log return of adj close| >= {MAX_ABS_LOGRET:g} (single pass)", "ERRORS",
         (pl.col("adj_close") / pl.col("adj_close").shift(1).over("ticker")).log().abs() >= MAX_ABS_LOGRET)

    n = df.group_by("ticker").len()
    short = n.filter(pl.col("len") < min_rows)["ticker"]
    short_rows = df.filter(pl.col("ticker").is_in(short.implode()))
    drop(f"ticker with < {min_rows} clean rows ({short.len():,} tickers)", "MIN HISTORY", pl.col("ticker").is_in(short.implode()))
    return df.select(OUT_COLS), log, {"seam": seam_info, "short": short_rows}


# ---------------------------------------------------------------- OUTLIERS: detect + cap, never delete
def ret1_expr() -> pl.Expr:
    return (pl.col("adj_close") / pl.col("adj_close").shift(1).over("ticker")).log()


def hl_expr() -> pl.Expr:
    return (pl.col("adj_high") - pl.col("adj_low")) / pl.col("adj_close")


def flags(col: str, k: float = MAD_K, iqr_k: float = IQR_K) -> dict[str, pl.Expr]:
    """Per-date cross-sectional outlier flags with robust statistics (uses only that date's values)."""
    x = pl.col(col)
    med = x.median().over("date")
    mad = (x - med).abs().median().over("date")
    q1, q3 = x.quantile(0.25, "linear").over("date"), x.quantile(0.75, "linear").over("date")
    return {"mad": ((x - med).abs() > k * 1.4826 * mad) & (mad > 0),
            "iqr": (x < q1 - iqr_k * (q3 - q1)) | (x > q3 + iqr_k * (q3 - q1)),
            "winsor": (x < x.quantile(WINSOR[0], "linear").over("date")) | (x > x.quantile(WINSOR[1], "linear").over("date"))}


def winsorize(df: pl.DataFrame, cols: list[str], lo: float = WINSOR[0], hi: float = WINSOR[1]) -> pl.DataFrame:
    """Cap each column at its same-date cross-sectional [lo, hi] percentiles (no look-ahead: only date t is used)."""
    return df.with_columns(pl.col(c).clip(pl.col(c).quantile(lo, "linear").over("date"), pl.col(c).quantile(hi, "linear").over("date")) for c in cols)


# ---------------------------------------------------------------- UNIVERSE (training step, default OFF)
def add_universe(df: pl.DataFrame) -> pl.DataFrame:
    """Adds bool `in_universe` = tradable at t? trailing-60d MEDIAN dollar volume >= the same date's own 5th percentile
    across tickers. Dollar volume = adj_close*adj_volume = real price*real shares (the adjustment factor cancels), so it is
    comparable over time; the percentile is era-relative (always cuts the thinnest ~5 % of that day's market, in every
    decade) and uses only information up to t. Tickers with < 60 rows of history are out (cannot be assessed). df sorted by ticker,date."""
    dv = (pl.col("adj_close") * pl.col("adj_volume")).rolling_median(UNIVERSE_WINDOW).over("ticker")
    df = df.with_columns(_dv=dv)
    return df.with_columns(in_universe=(pl.col("_dv") >= pl.col("_dv").quantile(UNIVERSE_Q, "linear").over("date")).fill_null(False)).drop("_dv")
