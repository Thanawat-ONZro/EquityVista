"""Job: compare every finished training run (same split / features / metrics) -> table in the log and
<run_dir>/comparison.csv. Reads runs/*/job.json; trains nothing."""
import csv
import json

from src.jobs import TRAIN, proto

COLS = ("val_rank_ic", "test_rank_ic", "test_accuracy", "test_up_rate", "test_f1", "test_auc", "unseen_rank_ic")


def run(p, run_dir):
    proto.progress("compare", 0)
    rows = []
    for f in sorted(proto.ROOT.glob("runs/*/job.json")):
        try:
            j = json.loads(f.read_text())
        except (OSError, ValueError):
            continue
        if j.get("type") in TRAIN and j.get("status") == "done":
            m = j["metrics"]
            rows.append({"run": j["id"], "model": j["type"][6:], "horizon": int(m.get("horizon", 0)),
                         **{c: m.get(c) for c in COLS}})
    rows.sort(key=lambda r: (r["horizon"], -(r["val_rank_ic"] or -9)))
    with (run_dir / "comparison.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, ["run", "model", "horizon", *COLS])
        w.writeheader()
        w.writerows(rows)
    print(f"{'model':9s} {'H':>3s} " + " ".join(f"{c[5:] if c.startswith('test_') else c:>12s}" for c in COLS))
    for r in rows:
        print(f"{r['model']:9s} {r['horizon']:>3d} " + " ".join(f"{(r[c] if r[c] is not None else float('nan')):>12.4f}" for c in COLS))
    proto.metric("runs_compared", len(rows))
    return {"comparison": "comparison.csv"}


if __name__ == "__main__":
    proto.main("evaluate", run)
