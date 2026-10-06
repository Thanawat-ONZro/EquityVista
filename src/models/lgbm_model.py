"""LightGBM (boosted decision trees = an extension of the course's tree ensembles).

A binary classifier: label = 1 if the H-day forward return is > 0, output = P(up). Rows are never shuffled; the number
of trees and the hyper-parameter search both use the later-in-time validation split (never the test split).
Trees: every fit runs to its round cap, then `pick_rounds` keeps the checkpoint with the best validation IC. (Early
stopping on the validation loss was dropped: on these noisy labels the loss bottoms out after a handful of trees, long
before the ranking stops improving, so it produced near-constant models.)
Search: a small random search over lr / num_leaves / min_data / feature_fraction, scored by validation IC (mean daily
cross-sectional corr of P(up) with the real return), bounded by `trials` and `max_minutes`.
"""
import time

import lightgbm as lgb
import numpy as np

from src import metrics
from src.models.common import FEATURES, X

BASE = dict(objective="binary", bagging_fraction=0.7, bagging_freq=1,
            lambda_l2=10.0, verbose=-1, feature_pre_filter=False)  # pre_filter off: min_data can change per trial
SEARCH_ROUNDS = 300  # rounds per search trial
GRID = (25, 50, 100, 200, 400, 800, 1600, 3200, 6400)  # tree counts compared by validation IC


def prep(train, val, feat):
    """(ids, y) splits -> (train Dataset, val Dataset), label = y > 0. Binned once, reused by every trial."""
    (it, yt), (iv, yv) = train, val
    dt = lgb.Dataset(X(feat, it), (yt > 0).astype(np.float32), feature_name=FEATURES, free_raw_data=False,
                     params={"feature_pre_filter": False})
    dv = lgb.Dataset(X(feat, iv), (yv > 0).astype(np.float32), reference=dt, free_raw_data=False)
    return dt, dv


def _params(lr, num_leaves, min_data, feature_fraction, seed, n_train):
    return {**BASE, "learning_rate": lr, "num_leaves": int(num_leaves), "feature_fraction": feature_fraction,
            "min_data_in_leaf": int(min(min_data, max(20, n_train // 50))), "seed": seed}  # clamp: tiny quick runs


def fit(dt, dv, params, rounds, callbacks=()):
    return lgb.train(params, dt, rounds, valid_sets=[dt, dv], valid_names=["train", "val"], callbacks=list(callbacks))


def pick_rounds(model, Xv, yv, dates) -> float:
    """Sets model.best_iteration to the GRID checkpoint (or the last round) with the best validation IC; returns that IC.
    yv = the forward RETURN (not the 0/1 label). ponytail: a coarse grid, not every round; a finer grid costs one predict per extra point."""
    n = model.current_iteration()
    ics = {k: metrics.daily_ic(model.predict(Xv, num_iteration=k), yv, dates) for k in [*(k for k in GRID if k < n), n]}
    model.best_iteration = max(ics, key=lambda k: ics[k] if np.isfinite(ics[k]) else -1.0)
    return ics[model.best_iteration]


def trim(model):
    """A booster holding only the trees `pick_rounds` kept, so a plain model.predict(X) is the final model."""
    return lgb.Booster(model_str=model.model_to_string(num_iteration=model.best_iteration))


def search(dt, dv, val, feat, base, trials, max_minutes, seed, n_train, on_trial=None):
    """Best {lr, num_leaves, min_data, feature_fraction} by validation IC. `base` = the job's own params
    (always tried first, so the search can only improve on them). Returns (best dict, results list)."""
    iv, yv = val
    dates = iv["date"]
    Xv = X(feat, iv)
    t0, results = time.time(), []

    def score(c):
        m = fit(dt, dv, _params(c["lr"], c["num_leaves"], c["min_data"], c["feature_fraction"], seed, n_train), SEARCH_ROUNDS)
        ic = pick_rounds(m, Xv, yv, dates)
        results.append({**c, "val_ic": ic, "rounds": m.best_iteration})
        if on_trial:
            on_trial(len(results), trials, ic)

    rng = np.random.default_rng(seed)
    score(base)  # the baseline counts as trial 1
    while len(results) < trials and time.time() - t0 < max_minutes * 60:
        score({"lr": float(np.exp(rng.uniform(np.log(0.02), np.log(0.15)))),
               "num_leaves": int(np.exp(rng.uniform(np.log(15), np.log(255)))),
               "min_data": int(np.exp(rng.uniform(np.log(200), np.log(5000)))),
               "feature_fraction": float(rng.uniform(0.5, 1.0))})
    best = max(results, key=lambda r: r["val_ic"] if np.isfinite(r["val_ic"]) else -1)
    return {k: best[k] for k in ("lr", "num_leaves", "min_data", "feature_fraction")}, results
