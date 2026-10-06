"""Job protocol: child prints JSON lines prefixed `@@EV ` on stdout, everything else is plain log text.

  {"type":"progress","stage":"train","pct":42.0,"epoch":2,"epochs":3,"loss":0.01,"val_loss":0.02}
  {"type":"metric","name":"ic","value":0.012}   {"type":"done","artifacts":{...}}   {"type":"error","msg":"..."}

`main(name, run)` is the shared CLI entry: parses --run-dir/--params, runs `run(params, run_dir)` (returns an
artifacts dict), writes <run_dir>/metrics.json, emits done/error and sets the exit code.
"""
import argparse
import json
import sys
import time
import traceback
from pathlib import Path

from src.jobs import PARAMS

PREFIX = "@@EV "
ROOT = Path(__file__).resolve().parents[2]
_metrics = {}


def emit(type_, **kw):
    print(PREFIX + json.dumps({"type": type_, **kw}), flush=True)


def progress(stage, pct, **kw):
    emit("progress", stage=stage, pct=round(float(pct), 1), **kw)


def metric(name, value):
    value = float(value)
    _metrics[name] = value
    emit("metric", name=name, value=value)


def data(name, value):
    """Non-scalar result (curve points, tables): goes to metrics.json only; the live event stream carries scalars."""
    _metrics[name] = value


def main(name, run, defaults=None):
    ap = argparse.ArgumentParser(prog=f"python -m src.jobs.{name}")
    ap.add_argument("--run-dir", type=Path, default=None)
    ap.add_argument("--params", default="{}", help="JSON overrides of the defaults")
    a = ap.parse_args()
    params = {**(PARAMS[name] if defaults is None else defaults), **json.loads(a.params)}
    run_dir = a.run_dir or ROOT / "runs" / f"cli-{name}-{int(time.time())}"
    run_dir.mkdir(parents=True, exist_ok=True)
    try:
        artifacts = run(params, run_dir) or {}
    except Exception as e:  # noqa: BLE001 - the manager reads the error event; the traceback goes to the log
        traceback.print_exc()
        emit("error", msg=str(e) or type(e).__name__)
        sys.exit(1)
    (run_dir / "metrics.json").write_text(json.dumps(_metrics, indent=2))
    emit("done", artifacts=artifacts)
