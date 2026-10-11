"""Plain-language texts for the Predict panel (pure functions, no Qt) + the model-comparison data."""
import json
import math

from app.core import registry

# key -> (short helper under the number, longer text for the "?" tooltip); {when} = registry.when(horizon)
HELP = {
    "Direction": ("More likely {dir} {when}",
                  "UP/DOWN is the side the model thinks is more likely {when}. It is a guess from past patterns, not a promise."),
    "Confidence": ("Chance the price is {dir} {when}: {n}%",
                   "The model's probability for the side shown above; 50% = coin flip. It is shown only when, on test data, "
                   "the stated chances matched what really happened (within 2 points on average)."),
    "Last close": ("Latest closing price", "The latest closing price the model started from."),
    "Today": ("Change vs the day before", "Live move of the last bar on the chart. It is not part of the forecast."),
    "Horizon": ("{h} trading days ≈ {cal} calendar days",
                "The market is closed on weekends and holidays, so one calendar month has about 21 trading days."),
}


def helper(key: str, pred: dict | None = None, horizon: int = 21) -> str:
    """Short line under a card's number; works without a prediction (placeholders)."""
    txt = HELP[key][0]
    if pred and key == "Confidence" and not pred.get("calibrated", True):
        return "Hidden: not calibrated on test data"
    d = ("higher" if pred["direction"] == "UP" else "lower") if pred else "higher or lower"
    n = f"{max(pred['confidence'], 1 - pred['confidence']) * 100:.0f}" if pred else "N"
    h = int(pred["horizon"] if pred else horizon)
    return txt.format(dir=d, when=registry.when(h), h=h, cal=round(h * 365 / 252), n=n)


def tip(key: str, horizon: int = 21) -> str:
    return HELP[key][1].format(when=registry.when(horizon))


def _num(v):
    return v if isinstance(v, (int, float)) and math.isfinite(v) else None


def read_metrics(rid: str) -> dict:
    try:
        return json.loads((registry.run_dir(rid) / "metrics.json").read_text())
    except (OSError, ValueError):
        return {}


def comparison(jobs: dict, horizon: int, active_rid: str | None) -> list[dict]:
    """One finished run of each model type at `horizon`: the one with the highest VALIDATION rank-IC (ties -> latest; the
    test split never picks a run); the active run stands in for its own type. -> [{model, run, ic, acc, base, active}] in registry.MODEL_NAME order, only types that have a run."""
    best = {}
    for rid, j in sorted(jobs.items()):
        t = j.get("type")
        if t not in registry.MODEL_NAME or j.get("status") != "done" or horizon != registry.horizon(j):
            continue
        m = read_metrics(rid)
        ic, acc = _num(m.get("test_rank_ic")), _num(m.get("test_diracc"))
        if ic is None and acc is None:
            continue
        cand = (rid == active_rid, _num(m.get("val_rank_ic")) or -9, rid)
        if t not in best or cand > best[t][0]:
            best[t] = (cand, {"model": registry.MODEL_NAME[t], "run": rid, "ic": ic, "acc": acc,
                              "base": _num(m.get("test_up_rate")), "active": rid == active_rid})
    return [best[t][1] for t in registry.MODEL_NAME if t in best]
