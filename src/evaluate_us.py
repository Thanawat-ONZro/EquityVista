"""Honest per-DAY evaluation of the US models vs baselines (plan Task 12, rules 4 & 7).

Unit of observation = one trading day (cross-section of tickers), never a pooled row.
Per day: Pearson corr, rank-IC (Spearman), AUC of direction, direction accuracy.
Headline = mean over days with a day-level 95% CI (two-sided Student-t on the daily series,
scipy.stats.t; chosen over a block bootstrap for simplicity -- daily IC is only mildly
autocorrelated, so treat the CI as slightly optimistic).

Horizons: h=1 (target y, daily rebalance) and h=5 (target y5). 5-day targets overlap on
consecutive dates, so for h=5 the day series is thinned to every 5th date (non-overlapping)
before the mean/CI -- otherwise the t-stat is inflated.

Decile spread: per day, mean ACTUAL y of the top decile of predicted return minus the bottom
decile (long-short, equal weight). Cost: 2*cost_bps (round trip) is subtracted once per
rebalance, i.e. every day at h=1 (harsh: full daily turnover assumed) and every 5 days at h=5.
Days with constant predictions (zero / majority baselines) have no ranking -> spread NaN.

Rule 4: every direction accuracy is shown next to the majority baseline (train-majority sign
applied to every row) scored on the SAME days/rows; the delta is a paired day-level difference.

All models/baselines are scored on the identical row universe (rows where the CNN-LSTM's
60-step window stays in one ticker).
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.split_us import FEATS, ROOT  # noqa: E402

OUT = ROOT / "data" / "processed"
METRICS = ("corr", "ic", "auc", "diracc")
MIN_N = 20          # min tickers per day to score that day
ZERO = 1e-8         # |y| below this counts as "flat" and is excluded from direction accuracy
MODELS = ["LGBM", "CNN-LSTM", "zero-return", "majority", "1d-reversal"]


def _days(dates):
    order = np.argsort(dates, kind="stable")
    u, start = np.unique(dates[order], return_index=True)
    return order, u, np.append(start, len(order))


def day_metrics(y_true, y_pred, dates, metrics=METRICS):
    """One row per date: n_tickers + requested metrics (NaN where undefined, e.g. constant pred)."""
    y, p, d = (np.asarray(a) for a in (y_true, y_pred, dates))
    ok = ~(np.isnan(y) | np.isnan(p))
    y, p, d = y[ok], p[ok], d[ok]
    order, u, b = _days(d)
    y, p = y[order], p[order]
    rows = []
    for i, day in enumerate(u):
        yy, pp = y[b[i]:b[i + 1]], p[b[i]:b[i + 1]]
        n = len(yy)
        if n < MIN_N:
            continue
        r = {"date": day, "n_tickers": n}
        const = pp.std() == 0
        if "corr" in metrics:
            r["corr"] = np.nan if const or yy.std() == 0 else np.corrcoef(yy, pp)[0, 1]
        if "ic" in metrics:
            r["ic"] = np.nan if const or yy.std() == 0 else np.corrcoef(stats.rankdata(yy), stats.rankdata(pp))[0, 1]
        if "auc" in metrics:
            pos = yy > 0
            n1, n0 = pos.sum(), n - pos.sum()
            r["auc"] = np.nan if n1 == 0 or n0 == 0 else \
                (stats.rankdata(pp)[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
        if "diracc" in metrics:
            nz = np.abs(yy) > ZERO            # pred >= 0 is "up" (so a 0 forecast = always up)
            r["diracc"] = np.mean((pp[nz] >= 0) == (yy[nz] > 0)) if nz.any() else np.nan
        rows.append(r)
    return pd.DataFrame(rows)


def ci(s):
    """(mean, lo, hi, n_days) of a day series: 95% t-interval on the mean, NaN days dropped."""
    s = np.asarray(s, dtype=float)
    s = s[~np.isnan(s)]
    n = len(s)
    if n < 3:
        return (np.nan,) * 3 + (n,)
    m, se = s.mean(), s.std(ddof=1) / np.sqrt(n)
    if se == 0:
        return m, m, m, n
    lo, hi = stats.t.interval(0.95, n - 1, loc=m, scale=se)
    return m, lo, hi, n


def summarize_day(dm):
    """DataFrame indexed by metric: mean, lo, hi, n_days  ->  'mean daily IC = X (95% CI [a,b])'."""
    cols = [c for c in dm.columns if c not in ("date", "n_tickers")]
    return pd.DataFrame({c: ci(dm[c]) for c in cols}, index=["mean", "lo", "hi", "n_days"]).T


def decile_spread(y_true, y_pred, dates, cost_bps=10):
    """Per-day long-short decile spread in bps: gross and net of 2*cost_bps round trip."""
    y, p, d = (np.asarray(a) for a in (y_true, y_pred, dates))
    ok = ~(np.isnan(y) | np.isnan(p))
    y, p, d = y[ok], p[ok], d[ok]
    order, u, b = _days(d)
    y, p = y[order], p[order]
    rows = []
    for i, day in enumerate(u):
        yy, pp = y[b[i]:b[i + 1]], p[b[i]:b[i + 1]]
        if len(yy) < MIN_N or pp.std() == 0:
            continue
        k = len(yy) // 10
        idx = np.argsort(pp, kind="stable")
        g = (yy[idx[-k:]].mean() - yy[idx[:k]].mean()) * 1e4
        rows.append({"date": day, "gross_bps": g, "net_bps": g - 2 * cost_bps})
    return pd.DataFrame(rows, columns=["date", "gross_bps", "net_bps"])


def compare(preds, splits, dates, target=0, cost_bps=10, step=1):
    """preds[model][split] -> array aligned with splits[split] = (ids, y). Returns long DataFrame.

    (`feat` is not needed here: baselines are built by the caller; deviation from the brief.)
    step=5 for the 5-day target (non-overlapping dates)."""
    out = []
    for sp in ("test_seen", "test_unseen"):
        y, dt = splits[sp][1][:, target], dates[sp]
        per = {}
        for name, by in preds.items():
            dm = day_metrics(y, by[sp], dt).iloc[::step]
            ds = decile_spread(y, by[sp], dt, cost_bps)
            per[name] = (dm, ds.iloc[::step])
        maj = per["majority"][0].set_index("date")["diracc"]
        for name, (dm, ds) in per.items():
            rec = {"ic": ci(dm["ic"]), "corr": ci(dm["corr"]), "auc": ci(dm["auc"]),
                   "diracc": ci(dm["diracc"]), "gross_bps": ci(ds["gross_bps"]),
                   "net_bps": ci(ds["net_bps"]),
                   "diracc_vs_majority": ci(dm.set_index("date")["diracc"] - maj)}
            for metric, (m, lo, hi, n) in rec.items():
                out.append(dict(horizon=1 if target == 0 else 5, split=sp, model=name,
                                metric=metric, mean=m, lo=lo, hi=hi, n_days=n))
        # same-period base rate of "up" days (context for rule 4)
        nz = np.abs(y) > ZERO
        out.append(dict(horizon=1 if target == 0 else 5, split=sp, model="(base rate)",
                        metric="up_fraction", mean=float(np.mean(y[nz & ~np.isnan(y)] > 0)),
                        lo=np.nan, hi=np.nan, n_days=np.nan))
    return pd.DataFrame(out)


def _f(r, scale=1.0, dec=4):
    return f"{r['mean'] * scale:+.{dec}f} [{r['lo'] * scale:+.{dec}f},{r['hi'] * scale:+.{dec}f}]"


def _verdict(r, what):
    if r["n_days"] < 3 or np.isnan(r["mean"]):
        return f"{what}: undefined (constant prediction)"
    if r["lo"] <= 0 <= r["hi"]:
        return f"{what}: {r['mean']:+.4f}, CI includes 0 -> not distinguishable from zero edge"
    return f"{what}: {r['mean']:+.4f}, CI excludes 0 -> " + ("POSITIVE" if r["mean"] > 0 else "NEGATIVE (inverse signal, worse than no edge)")


def render(res):
    lines = []
    for h in sorted(res.horizon.unique()):
        for sp in ("test_seen", "test_unseen"):
            r = res[(res.horizon == h) & (res.split == sp)]
            up = r[r.metric == "up_fraction"]["mean"].iloc[0]
            nd = int(r[r.metric == "ic"]["n_days"].max())
            lines += ["", f"=== horizon {h}d | {sp.upper()} | {nd} days | same-period up-fraction {up:.4f} ===",
                      f"{'model':12s} {'mean daily IC [95% CI]':26s} {'AUC':26s} {'diracc':26s} "
                      f"{'diracc - majority(paired)':27s} {'gross bps':24s} {'net bps (after '+str(2*COST)+'bps)':26s}"]
            for m in MODELS:
                g = lambda k: r[(r.model == m) & (r.metric == k)].iloc[0]
                if r[r.model == m].empty:
                    continue
                lines.append(f"{m:12s} {_f(g('ic')):26s} {_f(g('auc')):26s} {_f(g('diracc'), 100, 2):26s} "
                             f"{_f(g('diracc_vs_majority'), 100, 2):27s} {_f(g('gross_bps'), 1, 2):24s} "
                             f"{_f(g('net_bps'), 1, 2):26s}")
        lines.append("")
        for sp in ("test_seen", "test_unseen"):
            r = res[(res.horizon == h) & (res.split == sp)]
            for m in ("LGBM", "CNN-LSTM", "1d-reversal"):
                if r[r.model == m].empty:
                    continue
                g = lambda k: r[(r.model == m) & (r.metric == k)].iloc[0]
                lines.append(f"[{h}d {sp} {m}]")
                lines.append("   " + _verdict(g("ic"), "mean daily IC"))
                d, nb = g("diracc_vs_majority"), g("net_bps")
                lines.append("   " + (f"direction acc vs majority: {d['mean']*100:+.2f}pp, CI "
                             f"{'includes 0 -> no edge over majority' if d['lo'] <= 0 <= d['hi'] else 'excludes 0'}"))
                lines.append("   " + (f"net decile spread {nb['mean']:+.2f} bps/period "
                             + ("<= 0 after costs: NOT tradable" if nb["hi"] <= 0 or nb["mean"] <= 0 else "> 0 after costs")
                             + f" (95% CI [{nb['lo']:+.2f},{nb['hi']:+.2f}])"))
    return "\n".join(lines)


COST = 10


def _selfcheck():
    rng = np.random.default_rng(0)
    d = np.repeat(np.arange(60), 200)
    y = rng.normal(size=len(d)) * 0.02
    sig = y + rng.normal(size=len(d)) * 0.02
    assert ci(day_metrics(y, sig, d)["ic"])[1] > 0.3          # planted signal -> IC clearly > 0
    noise = day_metrics(y, rng.normal(size=len(d)), d)
    m, lo, hi, _ = ci(noise["ic"])
    assert lo < 0 < hi                                        # pure noise -> CI covers 0
    assert decile_spread(y, sig, d)["gross_bps"].mean() > 0
    assert decile_spread(y, np.zeros_like(y), d).empty        # constant pred -> no ranking


def main():
    _selfcheck()
    from src.models_us import cnn_lstm_model as cnn
    from src.models_us.lgbm_model import MODEL_DIR, predict_lgbm
    import lightgbm as lgb

    t0 = time.time()
    X, tick, sp = cnn.load()
    feat = pl.read_parquet(FEATS)
    print(f"loaded in {time.time() - t0:.0f}s")
    m_cnn = cnn.keras.models.load_model(cnn.MODEL_PATH)
    boosters = [lgb.Booster(model_file=str(MODEL_DIR / f"lgbm_us_{n}.txt")) for n in ("y", "y5")]

    splits, dates, rows_of = {}, {}, {}
    for k in ("test_seen", "test_unseen"):
        ids, y = sp[k]
        r = ids["row"].to_numpy().astype(np.int64)
        ok = (r >= cnn.W - 1) & (tick[np.maximum(r - (cnn.W - 1), 0)] == tick[r])
        rows_of[k], splits[k], dates[k] = r[ok], (ids.filter(pl.Series(ok)), y[ok]), ids["date"].to_numpy()[ok]
        print(f"{k}: {ok.sum():,} rows ({ids.height - ok.sum():,} dropped: window not in one ticker)")

    y_tr = sp["train"][1]
    res = []
    for target, step, col in ((0, 1, "ret1"), (1, 5, "ret5")):
        yt = y_tr[:, target]
        maj = 1.0 if np.mean(yt[~np.isnan(yt)] > 0) >= 0.5 else -1.0
        print(f"h={1 if target == 0 else 5}: train-majority direction = {'up' if maj > 0 else 'down'}")
        preds = {n: {} for n in MODELS}
        for k in splits:
            r = rows_of[k]
            preds["LGBM"][k] = predict_lgbm(boosters[target], r, feat)
            if target == 0:
                t = time.time()
                preds["CNN-LSTM"][k] = cnn.predict(m_cnn, X, r)
                print(f"  CNN {k} predicted in {time.time() - t:.0f}s")
            preds["zero-return"][k] = np.zeros(len(r))
            preds["majority"][k] = np.full(len(r), maj)
            preds["1d-reversal"][k] = -feat[col].gather(r).to_numpy()   # h=5: -ret5
        if target == 1:
            del preds["CNN-LSTM"]            # CNN-LSTM has no 5-day head
        res.append(compare(preds, splits, dates, target, COST, step))
    res = pd.concat(res, ignore_index=True)
    res.to_csv(OUT / "eval_us_results.csv", index=False)
    txt = ("EquityVista US evaluation -- per-day metrics, mean over days with 95% t-CI.\n"
           f"Net spread = gross long-short decile spread - {2 * COST} bps round trip per rebalance.\n"
           "diracc compared with the train-majority baseline on the same days (rule 4).\n"
           "h=5 table: LGBM y5 head; '1d-reversal' row there = -ret5; CNN-LSTM has no 5d head.\n"
           + render(res))
    (OUT / "eval_us_summary.txt").write_text(txt, encoding="utf-8")
    print(txt)
    print(f"\nsaved {OUT / 'eval_us_results.csv'}\nsaved {OUT / 'eval_us_summary.txt'}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
