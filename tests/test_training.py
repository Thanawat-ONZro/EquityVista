"""Backend: split embargo, shared metrics, every train job on a small synthetic market, predict output."""
import json

import numpy as np
import polars as pl
import pytest

from src import active, features as F, metrics, predict as P, split as S
from src.jobs import PARAMS, common, proto, train_lgbm, train_logistic, train_rf
from src.models import common as C, lgbm_model as L


def synth_raw(n_tickers=30, start="2018-01-02", end="2025-06-30", seed=0):
    rng = np.random.default_rng(seed)
    dates = pl.date_range(pl.date(*map(int, start.split("-"))), pl.date(*map(int, end.split("-"))), "1d", eager=True)
    dates = dates.filter(dates.dt.weekday() < 6)
    frames = []
    for i in range(n_tickers):
        r = rng.normal(0.0003, 0.015, len(dates))
        c = 50 * np.exp(np.cumsum(r))
        frames.append(pl.DataFrame({"ticker": f"T{i:02d}", "date": dates, "adj_open": c * 0.999, "adj_high": c * 1.01,
                                    "adj_low": c * 0.99, "adj_close": c, "adj_volume": rng.uniform(1e5, 1e6, len(dates))}))
    return pl.concat(frames)


@pytest.fixture(scope="module")
def features_file(tmp_path_factory):
    p = tmp_path_factory.mktemp("feat") / "features_synth.parquet"
    F.add_features(synth_raw()).write_parquet(p)
    return str(p)


@pytest.fixture(autouse=True)
def tmp_holdout(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "HOLDOUT_TXT", tmp_path / "holdout.txt")


@pytest.mark.parametrize("H", [5, 21, 63])
def test_embargo_at_least_horizon(features_file, H):
    feat = F.add_targets(pl.read_parquet(features_file), H)
    sp = S.make_splits(feat, H)
    cal = feat["date"].unique().sort().to_list()
    pos = {d: i for i, d in enumerate(cal)}
    gap = max(H, S.GAP)
    last_train, first_val = pos[sp["train"][0]["date"].max()], pos[sp["val"][0]["date"].min()]
    last_val, first_test = pos[sp["val"][0]["date"].max()], pos[sp["test_seen"][0]["date"].min()]
    assert first_val - last_train >= gap and first_test - last_val >= gap  # label windows never overlap
    to_d = lambda x: pl.Series([x]).str.to_date()[0]  # noqa: E731
    assert sp["train"][0]["date"].max() <= to_d(S.TRAIN_END) and sp["val"][0]["date"].min() >= to_d(S.VAL_START)
    assert sp["val"][0]["date"].max() <= to_d(S.VAL_END) and sp["test_seen"][0]["date"].min() >= to_d(S.TEST_START)
    assert sp["test_seen"][0]["date"].max() == cal[-1]  # test runs to the end of the data
    assert set(sp["test_unseen"][0]["ticker"]).isdisjoint(sp["train"][0]["ticker"])
    with pytest.raises(AssertionError):  # the checker itself fires on a too-small embargo
        S._check_embargo(cal, sp, gap + 10_000)


def test_start_year_limits_train_rows_only(features_file):
    feat = F.add_targets(pl.read_parquet(features_file), 5)
    sp = S.make_splits(feat, 5, start_year=2020)
    assert sp["train"][0]["date"].min().year == 2020 and sp["val"][0]["date"].min().year == 2022
    ids, _ = common.load({**PARAMS["train_logistic"], "features": features_file, "horizon": 5, "start_year": 2021})[1]["train"]
    assert ids["date"].min().year == 2021  # the job param reaches the split


def test_target_is_forward_h_day_return():
    raw = synth_raw(2, "2018-01-02", "2018-12-31")
    df = F.add_targets(raw, 5).filter(pl.col("ticker") == "T00").sort("date")
    full = raw.filter(pl.col("ticker") == "T00").sort("date")["adj_close"].to_numpy()
    assert np.isclose(df["y"][0], np.log(full[5] / full[0]))
    assert df.height == len(full) - 5


def test_metrics_perfect_and_random():
    rng = np.random.default_rng(1)
    dates = np.repeat(np.arange(60), 50)
    y = rng.normal(size=3000)
    d = pl.Series(dates).cast(pl.Date)
    good = metrics.evaluate(y + 0.5, y, d, horizon=5)  # p - 0.5 = the return itself
    assert good["ic"] > 0.99 and good["rank_ic"] > 0.99 and good["diracc"] == 1.0
    rand = metrics.evaluate(rng.normal(size=3000) + 0.5, y, d, horizon=5)
    assert abs(rand["ic"]) < 0.1 and not any("spread" in k for k in good)


KEYS = ("test_ic", "test_rank_ic", "test_diracc", "test_accuracy", "test_f1", "test_t_balanced_accuracy", "threshold", "horizon")
FAST = {"horizon": 5, "frac": 1.0, "seed": 1}


def _run(mod, name, tmp_path, **over):
    proto._metrics.clear()
    p = {**PARAMS[name], **FAST, **over}
    art = mod.run(p, tmp_path)
    return dict(proto._metrics), art


@pytest.mark.parametrize("mod,name,over", [
    (train_lgbm, "train_lgbm", {"rounds": 40, "trials": 3, "max_minutes": 1}),
    (train_logistic, "train_logistic", {}),
    (train_rf, "train_rf", {"n_estimators": 6, "max_depth": 5, "min_leaf": 100}),
])
def test_train_jobs_report_comparable_metrics(mod, name, over, features_file, tmp_path):
    m, art = _run(mod, name, tmp_path, features=features_file, **over)
    assert all(k in m and np.isfinite(m[k]) for k in KEYS), m
    assert m["horizon"] == 5 and (tmp_path / art["model"]).exists()


def test_lgbm_search(features_file, tmp_path):
    m, _ = _run(train_lgbm, "train_lgbm", tmp_path, features=features_file, rounds=40, trials=3, max_minutes=1)
    assert m["trials_run"] >= 2 and "resid_std" not in m
    assert sorted(f.name for f in (tmp_path / "model").iterdir()) == ["model.pkl", "ref_quantiles.json"]
    assert len(proto._metrics["search_trials"]) == m["trials_run"]


def test_logistic_run_and_predict(features_file, tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    rid = "r1"
    m, art = _run(train_logistic, "train_logistic", runs / rid, features=features_file)
    assert art["model"] == active.REL  # the file Predict loads
    assert len(json.loads((runs / rid / "model" / "ref_quantiles.json").read_text())) == 12
    (runs / rid / "metrics.json").write_text(json.dumps({k: v for k, v in m.items() if isinstance(v, float)}))
    monkeypatch.setattr(active, "RUNS", runs)
    monkeypatch.setattr(active, "FILE", tmp_path / "models" / "active.json")
    with pytest.raises(P.NoModelError):
        P.predict_ticker("T00", 5)
    active.set_active(5, rid)
    raw = synth_raw(40, "2019-01-02", "2020-12-31")
    monkeypatch.setattr(P, "_fetch", lambda t: raw)
    r = P.predict_ticker("T00", 5)
    assert r["horizon"] == 5 and 0 <= r["confidence"] <= 1 and r["direction"] == ("UP" if r["confidence"] >= 0.5 else "DOWN")
    # Backtest: nothing after the cutoff may influence the forecast
    r0 = P.predict_ticker("T00", 5, hide=60)
    days = raw.filter(pl.col("ticker") == "T00")["date"].unique().sort()
    cut = days[-61]
    assert r0["date"] == cut and r0["hidden"] == 60
    px = ["adj_open", "adj_high", "adj_low", "adj_close", "adj_volume"]
    scrambled = raw.with_columns(pl.when(pl.col("date") > cut).then(pl.col(c) * 3).otherwise(pl.col(c)).alias(c) for c in px)
    monkeypatch.setattr(P, "_fetch", lambda t: scrambled)
    assert P.predict_ticker("T00", 5, hide=60)["confidence"] == r0["confidence"]  # future prices ignored
    monkeypatch.setattr(P, "_fetch", lambda t: raw.filter(pl.col("date") <= cut))
    r1 = P.predict_ticker("T00", 5)  # same as simply never having the data
    assert r1["date"] == cut and r1["confidence"] == r0["confidence"]
    monkeypatch.setattr(P, "_fetch", lambda t: raw)
    with pytest.raises(P.NoModelError):
        P.predict_ticker("T00", 21)
    P._models.clear()
    (runs / rid / "model" / "ref_quantiles.json").unlink()  # a run without the live reference cannot be used: clear error, no guess
    with pytest.raises(ValueError, match="incomplete"):
        P.predict_ticker("T00", 5)


def test_lgbm_keeps_best_validation_ic_checkpoint(features_file, tmp_path):
    m, _ = _run(train_lgbm, "train_lgbm", tmp_path, features=features_file, rounds=120, trials=0, lr=0.3)
    assert m["n_rounds"] in (25, 50, 100, 120)  # a GRID checkpoint or the cap, chosen by validation IC (no L2 early stopping)
    import joblib
    assert joblib.load(tmp_path / active.REL).num_trees() == m["n_rounds"]  # only those trees are saved
    assert C.FEATURES and len(C.FEATURES) == 25


def test_metrics_ci_delta_and_calibration():
    rng = np.random.default_rng(2)
    d = pl.Series(np.repeat(np.arange(120), 50)).cast(pl.Date)
    y = rng.normal(0.001, 0.02, 6000)
    m = metrics.evaluate(0.5 + y + rng.normal(0, 0.02, 6000), y, d, horizon=5)  # a real (noisy) signal
    assert m["ic_lo"] < m["ic"] < m["ic_hi"] and m["ic_lo"] > 0 and m["n_days"] == 120
    assert m["ic_hi_nov"] - m["ic_lo_nov"] > m["ic_hi"] - m["ic_lo"]          # every-5th-day interval is wider
    assert m["diracc_vs_up_lo"] > 0
    nope = metrics.evaluate(0.5 + rng.normal(size=6000), y, d, horizon=5)
    assert nope["ic_lo"] < 0 < nope["ic_hi"]                                   # no signal -> CI includes 0
    mid, lo, hi = metrics.t_ci(rng.normal(size=400))[:3]
    assert lo < mid < hi
    up = (y > 0).astype(float)
    good, tab = metrics.classify(1 / (1 + np.exp(-y / 0.01)), y, up.mean())
    assert good["auc"] > 0.99 and good["brier_skill"] > 0 and len(tab["roc"]["fpr"]) == 101 and len(tab["reliability"]) == 10
    assert good["precision"] == good["recall"] == good["f1"] == good["accuracy"] == 1.0 and good["fp"] == good["fn"] == 0
    always, _ = metrics.classify(np.full(len(y), 0.6), y, up.mean(), thr=0.7)                 # says UP for everything at 0.5
    assert always["recall"] == 1.0 and np.isclose(always["precision"], up.mean()) and always["tn"] == 0
    assert always["tp"] + always["fp"] + always["fn"] + always["tn"] == len(y)
    assert always["balanced_accuracy"] == 0.5 and always["up_calls"] == 1.0 and always["t_up_calls"] == 0.0  # thr 0.7: all DOWN
    bad, _ = metrics.classify(np.full(len(y), 0.95), y, up.mean())                            # over-confident constant
    assert bad["calibrated"] == 0.0 and bad["brier_skill"] < 0 and bad["auc"] == 0.5
    # the threshold is chosen by balanced accuracy: a model whose P(up) is shifted up by 0.1 gets its cut moved to 0.6
    p = np.clip(0.6 + y * 5, 0, 1)
    thr, curve = metrics.best_threshold(p, y)
    assert abs(thr - 0.6) < 0.011 and len(curve) == len(metrics.THRESHOLDS)
    assert metrics.confusion(p, up, thr)["balanced_accuracy"] > metrics.confusion(p, up)["balanced_accuracy"]


def test_train_job_stores_ci_roc_and_calibration(features_file, tmp_path):
    m, _ = _run(train_lgbm, "train_lgbm", tmp_path, features=features_file, rounds=20, trials=0)
    for k in ("test_ic_lo", "test_ic_hi", "test_diracc_vs_up", "test_diracc_vs_up_lo", "test_t_recall", "test_auc",
              "test_brier", "test_brier_skill", "test_ece", "test_calibrated", "unseen_auc", "val_auc"):
        assert k in m and np.isfinite(m[k]), k
    assert len(proto._metrics["test_roc"]["tpr"]) == 101 and len(proto._metrics["test_reliability"]) == 10
