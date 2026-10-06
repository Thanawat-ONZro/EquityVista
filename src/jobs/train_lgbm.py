"""Job: LightGBM classifier for ONE horizon, output = P(up). params: features, horizon (days), rounds (max trees; the count
kept is the checkpoint with the best IC on the time-based validation split), lr, num_leaves, min_data, feature_fraction,
frac (share of rows), trials (hyper-parameter search budget, 0 = off), max_minutes (search time limit), seed."""
from src.jobs import common, proto
from src.models import lgbm_model as L


def run(p, run_dir):
    feat, sp, H, rng = common.load(p)
    frac, rounds, trials = float(p["frac"]), int(p["rounds"]), int(p["trials"])
    train, val = common.C.sub(sp["train"], frac, rng), sp["val"]
    n_train = len(train[1])
    proto.progress("prepare", 2)
    dt, dv = L.prep(train, val, feat)
    best = {"lr": float(p["lr"]), "num_leaves": int(p["num_leaves"]), "min_data": int(p["min_data"]),
            "feature_fraction": float(p["feature_fraction"])}
    base_pct = 0.0
    if trials > 1:
        base_pct = 40.0
        best, res = L.search(dt, dv, val, feat, best, trials, float(p["max_minutes"]), int(p["seed"]), n_train,
                             on_trial=lambda i, n, ic: proto.progress("search", base_pct * i / n))
        proto.metric("trials_run", len(res))
        proto.metric("search_best_val_ic", max(r["val_ic"] for r in res))
        proto.data("search_trials", res)
        print("best params:", best)
    every = max(1, rounds // 60)

    def cb(env):
        i = env.iteration + 1
        if i % every == 0 or i == env.end_iteration:
            r = {name: v for name, _m, v, _h in env.evaluation_result_list}
            proto.progress("train", base_pct + (100 - base_pct) * i / rounds, epoch=i, epochs=rounds,
                           loss=r.get("train"), val_loss=r.get("val"))

    proto.progress("train", base_pct)
    model = L.fit(dt, dv, L._params(best["lr"], best["num_leaves"], best["min_data"], best["feature_fraction"],
                                    int(p["seed"]), n_train), rounds, callbacks=[cb])
    L.pick_rounds(model, common.C.X(feat, val[0]), val[1], val[0]["date"])
    model = L.trim(model)
    proto.metric("n_rounds", model.num_trees())
    for k, v in best.items():
        proto.metric(f"param_{k}", v)
    return common.report(lambda ids: model.predict(common.C.X(feat, ids)), sp, feat, H, run_dir, model)


if __name__ == "__main__":
    proto.main("train_lgbm", run)
