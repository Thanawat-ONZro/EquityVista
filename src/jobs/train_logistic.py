"""Job: Logistic Regression on direction (y > 0 = up), output = P(up).
params: features, horizon, frac, C (inverse regularisation strength), seed."""
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.jobs import common, proto


def run(p, run_dir):
    feat, sp, H, rng = common.load(p)
    ids, y = common.C.sub(sp["train"], float(p["frac"]), rng)
    proto.progress("train", 30)
    m = make_pipeline(StandardScaler(), LogisticRegression(C=float(p["C"]), max_iter=300))
    m.fit(common.C.X(feat, ids), y > 0)
    return common.report(lambda i: m.predict_proba(common.C.X(feat, i))[:, 1], sp, feat, H, run_dir, m)


if __name__ == "__main__":
    proto.main("train_logistic", run)
