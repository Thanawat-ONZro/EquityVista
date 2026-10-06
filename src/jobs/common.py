"""Steps every training job shares, so all model types use the same split, features and metrics protocol."""
import json

import joblib
import numpy as np
import polars as pl

from src import active, metrics
from src.jobs import proto
from src.models import common as C

REF_DAYS = 250  # ~1 year of trading dates behind the live reference quantiles


def load(p):
    """-> (feat, splits, horizon, rng) for the job's params (features, horizon, start_year, seed)."""
    H = int(p["horizon"])
    proto.progress("load", 0)
    feat, sp = C.load(p["features"], H, int(p.get("start_year", 1962)), int(p.get("universe", 0)))
    proto.metric("horizon", H)
    return feat, sp, H, np.random.default_rng(int(p["seed"]))


def report(p_up, sp, feat, H, run_dir, model):
    """Save the model (anything with predict_proba, or a LightGBM Booster) and score val / test(seen) / unseen.
    `p_up(ids) -> P(up)`. Metrics: <val|test|unseen>_<name>, names in src/metrics.py. The run's own threshold is
    chosen on VALIDATION (so the val_t_* numbers are in-sample; judge the threshold on test / unseen)."""
    out = run_dir / active.REL
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out)
    # Reference cross-section for live forecasts (a lone ticker has none): 101 quantiles of each ranked feature
    # over the last REF_DAYS dates of the features file. predict.py ranks and caps the live values against these.
    recent = feat.filter(pl.col("date") >= feat["date"].unique().sort()[-REF_DAYS:][0])
    (out.parent / "ref_quantiles.json").write_text(json.dumps(
        {f: np.quantile(recent[f].to_numpy(), np.linspace(0, 1, 101)).tolist() for f in C.RANKED}))
    yv = sp["val"][1]
    pv = p_up(sp["val"][0])
    thr, curve = metrics.best_threshold(pv, yv)
    proto.metric("threshold", thr)
    proto.data("val_threshold_curve", curve)
    for split, prefix in (("val", "val"), ("test_seen", "test"), ("test_unseen", "unseen")):
        ids, y = sp[split]
        p = pv if split == "val" else p_up(ids)
        c, tables = metrics.classify(p, y, float(np.mean(yv > 0)), thr)
        for k, v in {**metrics.evaluate(p, y, ids["date"], H), **c}.items():
            proto.metric(f"{prefix}_{k}", v)
        for k, v in tables.items():
            proto.data(f"{prefix}_{k}", v)
    return {"model": active.REL}
