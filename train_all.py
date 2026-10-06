"""Train everything the Predict app needs, in one chain: [fetch + clean] -> features -> Logistic, Random Forest,
LightGBM (all classifiers: P(up)) -> evaluate; then the run with the best VALIDATION rank-IC becomes the model the app
uses (models/active.json). The test split never chooses anything.

    python train_all.py [quick|demo] [--seeds N] [--fetch] [--no-features]

  quick  ~3 min    1 % of the training rows, few trees: only checks that everything works
  demo   ~6 min per seed   the settings of src/jobs/PARAMS: ~1M training rows (5.3 %), chosen on validation
                   (REPORT/figures/experiments.json: more rows, other settings and other feature sets did not help)
  --seeds N      repeat the three models with N different seeds (default 1); the seed changes which rows are sampled
                 and the randomness inside the tree models, so the spread over seeds shows how stable a result is
  --fetch        first download the 2020+ prices (yfinance) and rebuild the cleaned dataset
  --no-features  reuse data/processed/features.parquet as it is (saves ~2 min when the dataset did not change)

Runs land in runs/<id>/ exactly as the app expects (job.json, metrics.json, model/). Restart the app afterwards.
Jobs are separate processes run one after the other by app/core/job_manager.py; a failed step cancels the rest.
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from PySide6.QtCore import QCoreApplication, QTimer  # noqa: E402

from app.core.job_manager import JobManager  # noqa: E402
from src.jobs import PARAMS, TRAIN  # noqa: E402

PRESETS = {  # overrides of src/jobs/PARAMS per job type
    "quick": {"train_logistic": {"frac": 0.01}, "train_rf": {"frac": 0.01, "n_estimators": 20},
              "train_lgbm": {"frac": 0.01, "rounds": 100}},
    "demo": {t: {} for t in TRAIN},
}


def main() -> int:
    args = sys.argv[1:]
    preset = next((a for a in args if a in PRESETS), "demo")
    seeds = int(args[args.index("--seeds") + 1]) if "--seeds" in args else 1
    specs = [("data_extend", None, None)] if "--fetch" in args else []
    if "--no-features" not in args or "--fetch" in args:
        specs.append(("build_features", {"start_year": 1962}, None))
    first = len(specs)
    specs += [(t, {**over, "seed": PARAMS[t]["seed"] + s}, None) for s in range(seeds) for t, over in PRESETS[preset].items()]
    specs.append(("evaluate", None, None))

    app = QCoreApplication([])
    mgr = JobManager()
    ids = mgr.submit_chain(specs)
    print(f"preset {preset}, {seeds} seed(s): queued {len(ids)} jobs", flush=True)
    seen, t0 = {}, time.time()

    def tick():
        for i in ids:
            j = mgr.jobs[i]
            key = (j["status"], j["stage"], int(j["pct"] // 25))
            if seen.get(i) != key:
                seen[i] = key
                print(f"{time.time() - t0:6.0f}s  {i:38s} {j['status']:10s} {j['stage']:9s} {j['pct']:3.0f}%  {j['error'] or ''}", flush=True)
        if not mgr.live_count():
            app.quit()

    timer = QTimer()
    timer.timeout.connect(tick)
    timer.start(3000)
    app.aboutToQuit.connect(mgr.shutdown)
    app.exec()
    bad = [i for i in ids if mgr.jobs[i]["status"] != "done"]
    trained = [i for i in ids[first:-1] if i not in bad]
    if trained:
        best = max(trained, key=lambda i: mgr.jobs[i]["metrics"].get("val_rank_ic", -9))
        mgr.activate(best)
        print(f"active model: {best} (validation rank-IC {mgr.jobs[best]['metrics'].get('val_rank_ic', float('nan')):.4f})", flush=True)
    if ids[-1] not in bad:
        print((ROOT / "runs" / ids[-1] / "log.txt").read_text(encoding="utf-8"))
    print(f"done in {(time.time() - t0) / 60:.1f} min" + (f"; NOT done: {bad}" if bad else ""), flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
