"""Model-agnostic test metrics, identical for every model type (so runs are directly comparable).

Every model is a classifier: p = its P(up) = the chance that the price is higher after H trading days; y = the H-day
forward log return (up = y > 0). One "day" = one trading date: a daily metric is computed across the tickers of that
date, then summarised over the days. With ~650 test days the mean of a daily metric is noisy, so every daily metric also
gets a 95 % confidence interval (two-sided Student-t on the daily series: mean +- t * std / sqrt(days)); read "CI
includes 0" as "no proven edge". Caveat: an H-day label overlaps its neighbours, so consecutive daily values are
correlated and the plain interval is somewhat too narrow; the *_lo_nov / *_hi_nov versions use only every H-th day
(non-overlapping windows) and are wider.

evaluate():
  ic, rank_ic     mean over days of the cross-sectional Pearson / Spearman corr(p, y): does the model put the stocks
                  of one day in the right order? (rank_ic is less sensitive to outliers)
  diracc          share of rows (y != 0) where the call at P(up) > 0.5 matches the sign of y
  up_rate         share of rows (y != 0) with y > 0 = the accuracy of a model that always says UP (the baseline)
  diracc_day      direction accuracy computed per day, then averaged over days (diracc pools all rows instead)
  diracc_vs_up    per day (direction accuracy - share of up rows that day), averaged: how much better than "always UP"
                  the model is on the SAME days. > 0 with a CI above 0 = a proven edge over the baseline.
  <m>_lo, <m>_hi  95 % t-interval of the daily mean for m in ic, rank_ic, diracc_day, diracc_vs_up
  <m>_lo_nov, <m>_hi_nov  the same interval on every H-th day only, for ic, rank_ic, diracc_vs_up
  n_days          number of days behind the daily means (days with >= MIN_N tickers)
classify() (stored per split as <split>_<name>):
  tp, fp, fn, tn  confusion matrix of the UP class at the threshold 0.5: the model says UP when P(up) > 0.5, reality is
                  UP when y > 0. tp = said UP and it rose, fp = said UP and it did not, fn / tn likewise for DOWN.
  accuracy        (tp + tn) / all rows. precision = tp / (tp + fp): of the UP calls, the share that rose.
  recall          tp / (tp + fn): of the rows that rose, the share the model called UP. f1 = harmonic mean of the two.
                  A model that says UP for everything has recall 1 and precision = the up-rate: compare with that.
  specificity     tn / (tn + fp): of the rows that fell, the share the model called DOWN.
  balanced_accuracy  (recall + specificity) / 2. "Always UP" and "always DOWN" both score exactly 0.5.
  up_calls        share of rows the model called UP.
  t_<name>        the same nine numbers at the run's own threshold (best_threshold on VALIDATION) instead of 0.5.
  auc             area under the ROC curve over all rows: chance that a random up row scores above a random down row
                  (0.5 = coin flip, 1 = perfect). Does not depend on the threshold.
  roc             (metrics.json only) 101 points {"fpr": [...], "tpr": [...]} of the ROC curve at fpr = 0, 0.01 .. 1.
  brier           mean squared error of P(up) against the outcome (1 if y > 0 else 0); lower is better.
  brier_ref       Brier score of the lazy forecast "P(up) = the validation up-rate, always".
  brier_skill     1 - brier / brier_ref: > 0 = beats the lazy forecast, <= 0 = adds nothing.
  ece             expected calibration error: BINS bins of equal size sorted by P(up), weighted mean of
                  |average stated P(up) - observed up share|. 0.02 = off by 2 percentage points on average.
  calibrated      1.0 if ece <= ECE_OK (the P(up) can be read as a probability), else 0.0.
  reliability     (metrics.json only) the reliability table: one row per bin with n, mean_pred, observed_up.
"""
import numpy as np
import polars as pl
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve

MIN_N = 20      # min tickers on a day for that day to count
ECE_OK = 0.02   # "well calibrated": stated P(up) is within 2 percentage points of the observed share, on average
BINS = 10
THRESHOLDS = np.round(np.arange(0.30, 0.701, 0.01), 2)  # candidates for best_threshold


def _frame(pred, y, dates) -> pl.DataFrame:
    df = pl.DataFrame({"d": dates, "p": np.asarray(pred, float), "y": np.asarray(y, float)})
    return df.filter(pl.col("p").is_finite() & pl.col("y").is_finite())


def t_ci(x, level: float = 0.95):
    """(mean, lo, hi, n) of a series: t-interval on the mean, NaN days dropped. n < 3 -> NaN."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 3:
        return float("nan"), float("nan"), float("nan"), n
    m = float(x.mean())
    h = float(stats.t.ppf((1 + level) / 2, n - 1) * x.std(ddof=1) / np.sqrt(n))
    return m, m - h, m + h, n


def _day_table(df: pl.DataFrame) -> pl.DataFrame:
    """One row per day (>= MIN_N tickers): ic, rank_ic, diracc_day, up_day, diracc_vs_up, sorted by date. p is centred (P(up) - 0.5)."""
    p, y = pl.col("p"), pl.col("y")
    nz = y != 0
    return (df.with_columns(n=pl.len().over("d")).filter(pl.col("n") >= MIN_N)
            .group_by("d").agg(ic=pl.corr("p", "y"), rank_ic=pl.corr("p", "y", method="spearman"),
                               diracc_day=((p.sign() == y.sign()) & nz).sum() / nz.sum(), up_day=((y > 0) & nz).sum() / nz.sum())
            .sort("d").with_columns(diracc_vs_up=pl.col("diracc_day") - pl.col("up_day")))


def daily_ic(p, y, dates) -> float:
    return t_ci(_day_table(_frame(p, y, dates))["ic"].to_numpy())[0]


def evaluate(p, y, dates, horizon: int) -> dict:
    df = _frame(np.asarray(p, float) - 0.5, y, dates)
    if df.is_empty():
        return {k: float("nan") for k in ("ic", "rank_ic", "diracc", "up_rate")}
    nz = df.filter(pl.col("y") != 0)
    t = _day_table(df)
    H = max(1, int(horizon))
    day = {k: t[k].to_numpy() for k in ("ic", "rank_ic", "diracc_day", "diracc_vs_up")}
    out = {"diracc": float((nz["p"].sign() == nz["y"].sign()).mean()), "up_rate": float((nz["y"] > 0).mean())}
    for k, x in day.items():
        out[k], out[f"{k}_lo"], out[f"{k}_hi"], _ = t_ci(x)
    for k in ("ic", "rank_ic", "diracc_vs_up"):  # same interval on every H-th day only
        _, out[f"{k}_lo_nov"], out[f"{k}_hi_nov"], _ = t_ci(day[k][::H])
    out["n_days"] = t.height
    return out


def confusion(p, up, thr: float = 0.5) -> dict:
    """Confusion matrix of the UP class and the numbers read off it; the model says UP when P(up) > thr."""
    said, up = np.asarray(p, float) > thr, np.asarray(up) > 0
    tp, fp = float((said & up).sum()), float((said & ~up).sum())
    fn, tn = float((~said & up).sum()), float((~said & ~up).sum())
    prec, rec, spec = tp / max(tp + fp, 1), tp / max(tp + fn, 1), tn / max(tn + fp, 1)
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "accuracy": (tp + tn) / max(len(up), 1), "precision": prec, "recall": rec,
            "f1": 2 * prec * rec / max(prec + rec, 1e-12), "specificity": spec, "balanced_accuracy": (rec + spec) / 2,
            "up_calls": float(said.mean()) if len(up) else float("nan")}


def best_threshold(p, y) -> tuple[float, list]:
    """Threshold in THRESHOLDS with the best balanced accuracy (ties -> the one nearest 0.5), and the whole curve.
    Balanced accuracy, not accuracy or F1: those two are maximised by calling everything UP whenever most rows rise."""
    up = np.asarray(y, float) > 0
    curve = [{"thr": float(t), **{k: confusion(p, up, t)[k] for k in ("balanced_accuracy", "accuracy", "f1", "up_calls")}}
             for t in THRESHOLDS]
    return max(curve, key=lambda r: (round(r["balanced_accuracy"], 6), -abs(r["thr"] - 0.5)))["thr"], curve


def classify(p, y, ref_rate: float, thr: float = 0.5) -> tuple[dict, dict]:
    """p = P(up), y = forward return, ref_rate = validation up-rate for the lazy Brier reference, thr = the run's own
    threshold (the t_* numbers). -> (scalars, {"roc": ..., "reliability": ...})."""
    p, y = np.asarray(p, float), np.asarray(y, float)
    ok = np.isfinite(p) & np.isfinite(y)
    p, up = p[ok], (y[ok] > 0).astype(float)
    if len(up) == 0 or up.min() == up.max():
        keys = list(confusion([0.0], [1]))
        return {k: float("nan") for k in ("auc", "brier", "brier_ref", "brier_skill", "ece", "calibrated", *keys, *(f"t_{k}" for k in keys))}, {}
    fpr, tpr, _ = roc_curve(up, p)
    grid = np.linspace(0, 1, 101)
    brier, ref = float(np.mean((p - up) ** 2)), float(np.mean((ref_rate - up) ** 2))
    rows = [{"n": len(i), "mean_pred": float(p[i].mean()), "observed_up": float(up[i].mean())}
            for i in np.array_split(np.argsort(p, kind="stable"), BINS) if len(i)]
    ece = sum(r["n"] * abs(r["mean_pred"] - r["observed_up"]) for r in rows) / len(up)
    return ({"auc": float(roc_auc_score(up, p)), "brier": brier, "brier_ref": ref, "brier_skill": 1 - brier / ref,
             "ece": float(ece), "calibrated": float(ece <= ECE_OK), **confusion(p, up),
             **{f"t_{k}": v for k, v in confusion(p, up, thr).items()}},
            {"roc": {"fpr": grid.tolist(), "tpr": np.interp(grid, fpr, tpr).tolist()}, "reliability": rows})
