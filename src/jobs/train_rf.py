"""Job: Random Forest classifier on direction (y > 0 = up), output = P(up) (course: bagged decision trees).
params: features, horizon, frac (share of train rows; also capped at MAX_ROWS so it fits RAM), n_estimators,
max_depth, min_leaf (min rows per leaf), max_features (share of features tried per split), seed.
Trees are grown in chunks (warm_start) so progress is visible. All cores are used (n_jobs=-1)."""
from sklearn.ensemble import RandomForestClassifier

from src.jobs import common, proto

MAX_ROWS = 3_000_000  # 25 float32 features x 3M rows = 300 MB; trees stay small thanks to min_leaf


def run(p, run_dir):
    feat, sp, H, rng = common.load(p)
    ids, y = common.C.sub(sp["train"], min(float(p["frac"]), MAX_ROWS / max(len(sp["train"][1]), 1)), rng)
    print(f"RandomForest on {len(y):,} rows")
    n = int(p["n_estimators"])
    X = common.C.X(feat, ids)
    m = RandomForestClassifier(n_estimators=1, max_depth=int(p["max_depth"]), min_samples_leaf=int(p["min_leaf"]),
                               max_features=float(p["max_features"]), n_jobs=-1, random_state=int(p["seed"]), warm_start=True)
    step = max(1, n // 20)
    for k in [*range(step, n, step), n]:
        m.set_params(n_estimators=k)
        m.fit(X, y > 0)
        proto.progress("train", k / n * 100, epoch=k, epochs=n)
    del X
    return common.report(lambda i: m.predict_proba(common.C.X(feat, i))[:, 1], sp, feat, H, run_dir, m)


if __name__ == "__main__":
    proto.main("train_rf", run)
