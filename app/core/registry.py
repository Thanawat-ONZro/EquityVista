"""Run store: runs/<run_id>/{job.json, log.txt, metrics.json, model/...} (json files only, gitignored).
The registry is just a scan of runs/. Jobs still queued/running when the app died come back as `interrupted`."""
import json
import shutil
import time
from pathlib import Path

from src import active

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "runs"  # tests monkeypatch this; every function reads it at call time
LIVE = ("queued", "running")


def run_dir(rid: str) -> Path:
    return RUNS / rid


def new_id(type_: str) -> str:
    base = f"{time.strftime('%Y%m%d-%H%M%S')}-{type_}"
    rid, n = base, 1
    while run_dir(rid).exists():
        n += 1
        rid = f"{base}-{n}"
    return rid


def save(job: dict) -> None:
    d = run_dir(job["id"])
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / "job.json.tmp"
    tmp.write_text(json.dumps(job))
    tmp.replace(d / "job.json")  # atomic: a crash never leaves half a job.json


def load_all(restore: bool = True) -> dict:
    """restore=False: read-only (the Predict app): live jobs of another process are left alone."""
    jobs = {}
    for p in sorted(RUNS.glob("*/job.json")):
        try:
            job = json.loads(p.read_text())
        except (OSError, ValueError):
            continue
        if restore and job["status"] in LIVE:
            job["status"], job["finished"] = "interrupted", time.time()
            save(job)
        jobs[job["id"]] = job
    return jobs


MODEL_NAME = {"train_lgbm": "LightGBM", "train_logistic": "Logistic", "train_rf": "Random Forest"}
KEY = "test_rank_ic"  # the one metric shown next to a model id (rank-IC: not driven by a few extreme returns)


def horizon_name(days) -> str:
    days = int(days)
    names = {5: "1 week", 10: "2 weeks", 21: "1 month", 63: "3 months", 126: "6 months", 252: "1 year"}
    return names.get(days, f"{days} days")


def when(days) -> str:
    """'in 1 month (21 trading days)': the calendar name first, the market's day count second."""
    days = int(days)
    name = horizon_name(days)
    return f"in {days} trading days" if name.endswith("days") else f"in {name} ({days} trading days)"


def horizon(job: dict):
    return job.get("params", {}).get("horizon")


def can_activate(job: dict) -> bool:
    """Any finished training run: every model type is a classifier that outputs P(up) and is saved the same way."""
    return job.get("type") in MODEL_NAME and job["status"] == "done" and (run_dir(job["id"]) / active.REL).exists()


def activate(job: dict) -> int:
    """Make a finished run the model Predict uses for its horizon (writes models/active.json only)."""
    if not can_activate(job):
        raise ValueError(f"{job['id']}: no activatable model (needs a finished training run)")
    active.set_active(horizon(job), job["id"])
    return int(horizon(job))


def summary(jobs: dict) -> list:
    """One line per active horizon: '21d · <run id> · test_ic 0.0123'."""
    out = []
    for h, rid in sorted(active.read()["h"].items(), key=lambda kv: int(kv[0])):
        m = jobs.get(rid, {}).get("metrics", {}).get(KEY)
        out.append(f"{h}d · {rid}" + (f" · rank-IC {m:.3g}" if m is not None else ""))
    return out


def delete(rid: str) -> None:
    shutil.rmtree(run_dir(rid), ignore_errors=True)


def read_log(rid: str, max_bytes: int = 200_000) -> str:
    try:
        with (run_dir(rid) / "log.txt").open("rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - max_bytes))
            return f.read().decode("utf-8", "replace")
    except OSError:
        return ""
