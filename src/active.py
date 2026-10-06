"""models/active.json = which training run Predict uses PER HORIZON: {"h": {"21": run_id}, "prev": {"21": run_id}}.
No fallback model: a horizon without an active run has no prediction ("train this horizon").
Stdlib only (the GUI, predict.py and tests all import it)."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILE = ROOT / "models" / "active.json"  # tests monkeypatch FILE and RUNS
RUNS = ROOT / "runs"
REL = "model/model.pkl"  # inside runs/<id>/; every model type is saved under this name (joblib)


def read() -> dict:
    """-> {"h": {horizon(str): run_id}, "prev": {...}}; the pre-horizon format (lgbm/cnn keys) reads as empty."""
    try:
        d = json.loads(FILE.read_text())
    except (OSError, ValueError):
        d = {}
    return {k: {str(h): r for h, r in (d.get(k) or {}).items() if r and str(h).isdigit()} for k in ("h", "prev")}


def write(d: dict) -> None:
    FILE.parent.mkdir(exist_ok=True)
    tmp = FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(d, indent=2))
    tmp.replace(FILE)  # atomic: Predict never reads a half-written file


def set_active(horizon, rid: str) -> None:
    d, h = read(), str(int(horizon))
    if d["h"].get(h) != rid:
        if h in d["h"]:
            d["prev"][h] = d["h"][h]
        d["h"][h] = rid
        write(d)


def rollback(horizon) -> None:
    """Swap with the previous model of that horizon (a second rollback goes forward again)."""
    d, h = read(), str(int(horizon))
    if h in d["prev"]:
        d["h"][h], d["prev"][h] = d["prev"][h], d["h"].get(h)
        d["prev"] = {k: v for k, v in d["prev"].items() if v}
        write(d)


def run_id(horizon) -> str | None:
    return read()["h"].get(str(int(horizon)))


def path(horizon) -> Path | None:
    """Model file of the active run for `horizon`; None if there is none (or its files are gone)."""
    rid = run_id(horizon)
    p = RUNS / rid / REL if rid else None
    return p if p and p.exists() else None
