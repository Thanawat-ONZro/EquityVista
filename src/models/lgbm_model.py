"""LightGBM primary model (plan rule 4: boosted trees first) for the US panel.

Features (25, all scale-free -> rule 2): 13 per-row features + their 12 same-date
cross-sectional percentile ranks. No ticker/date, no adj_* price/volume levels
(adj_close is a level; close_ma20/50 are ratios and fine). mkt_ret is the
same-date cross-sectional mean return (known at close t, no lookahead).

Two heads, same features: target 0 = y (t+1 log ret), target 1 = y5 (t+1..t+5,
NaN tail rows masked). The 5-day head is cheap and feeds the decile spread.
Rows are never shuffled; early stopping uses the later-in-time val split.
"""
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.split import FEATS, ROOT, make_splits  # noqa: E402

RANKED = ["ret1", "ret5", "ret20", "ret60", "vol20", "vol60",
          "close_ma20", "close_ma50", "rsi14", "macd_n", "vol_z", "hl_range"]
FEATURES = RANKED + ["mkt_ret"] + [f"r_{c}" for c in RANKED]
MODEL_DIR = ROOT / "models"


def _rows(ids):
    return ids["row"].to_numpy() if isinstance(ids, pl.DataFrame) else np.asarray(ids)


def _X(feat, rows):
    return feat[rows, FEATURES].cast(pl.Float32).to_numpy()


def train_lgbm(train, val, feat, objective="regression", seed=42, target=0,
               rounds=500, patience=30):
    """train/val = (ids, y) from make_splits. target 0 -> y, 1 -> y5. Returns (Booster, info)."""
    data = []
    for ids, y in (train, val):
        yt = y[:, target]
        ok = ~np.isnan(yt)
        data.append((_X(feat, _rows(ids)[ok]), yt[ok]))
    (Xt, yt), (Xv, yv) = data
    params = dict(objective=objective, learning_rate=0.05, num_leaves=63, min_data_in_leaf=2000,
                  feature_fraction=0.8, bagging_fraction=0.7, bagging_freq=1, lambda_l2=10.0,
                  seed=seed, verbose=-1)
    if objective == "huber":
        params["alpha"] = 0.05  # log-return scale
    dt = lgb.Dataset(Xt, yt, feature_name=FEATURES, free_raw_data=True)
    dv = lgb.Dataset(Xv, yv, reference=dt)
    model = lgb.train(params, dt, rounds, valid_sets=[dv],
                      callbacks=[lgb.early_stopping(patience, verbose=False)])
    metric = next(iter(model.best_score["valid_0"].items()))
    imp = dict(sorted(zip(FEATURES, model.feature_importance("gain").round(1)),
                      key=lambda kv: -kv[1]))
    return model, {"feature_importance": imp, "n_rounds": model.best_iteration,
                   "val_metric": {metric[0]: metric[1]}}


def predict_lgbm(model, rows, feat):
    """rows: DataFrame with `row` col or int array -> predicted return per row."""
    return model.predict(_X(feat, _rows(rows)), num_iteration=model.best_iteration)


INTERVAL_JSON = MODEL_DIR / "interval.json"


def resid_std(model, val, feat, target=0):
    """std(actual - pred) on the val split (NaN targets masked): basis of the 95% interval."""
    ids, y = val
    yt = y[:, target]
    ok = ~np.isnan(yt)
    return float(np.std(yt[ok] - predict_lgbm(model, _rows(ids)[ok], feat)))


def save_resid_std(name, value):
    d = json.loads(INTERVAL_JSON.read_text()) if INTERVAL_JSON.exists() else {}
    d[name] = value
    INTERVAL_JSON.write_text(json.dumps(d))


def score(pred, y):
    ok = ~np.isnan(y)
    p, t = pred[ok], y[ok]
    nz = t != 0
    return {"corr": float(np.corrcoef(p, t)[0, 1]), "mean_abs_y": float(np.abs(t).mean()),
            "diracc": float((np.sign(p[nz]) == np.sign(t[nz])).mean())}


if __name__ == "__main__":
    t0 = time.time()
    feat = pl.read_parquet(FEATS)
    sp = make_splits(feat)
    print(f"load+split {time.time() - t0:.0f}s; {len(FEATURES)} features")
    MODEL_DIR.mkdir(exist_ok=True)
    for target, name in ((0, "y"), (1, "y5")):
        t1 = time.time()
        model, info = train_lgbm(sp["train"], sp["val"], feat, target=target)
        secs = time.time() - t1
        path = MODEL_DIR / f"lgbm_{name}.txt"
        model.save_model(str(path), num_iteration=model.best_iteration)
        info["resid_std_y" if target == 0 else "resid_std_y5"] = rs = resid_std(model, sp["val"], feat, target)
        save_resid_std(name, rs)
        print(f"  val resid_std={rs:.5f}")
        print(f"\n[{name}] train {secs:.0f}s rounds={info['n_rounds']} val={info['val_metric']} "
              f"file={path.name} {path.stat().st_size / 1e6:.1f}MB")
        print("  top gain:", list(info["feature_importance"].items())[:6])
        for split in ("val", "test_seen", "test_unseen"):
            ids, y = sp[split]
            print(f"  {split:12s}", score(predict_lgbm(model, ids, feat), y[:, target]))
