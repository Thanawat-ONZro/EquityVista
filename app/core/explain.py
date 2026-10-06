"""Plain-language texts and verdicts for the Predict panel (pure functions, no Qt) + the model-comparison data."""
import json
import math

from app.core import registry
from src import active

# key -> (short helper under the number, longer text for the "?" tooltip); {when} = registry.when(horizon)
HELP = {
    "Direction": ("Model gives the price more than a 50% chance to be {dir} {when}",
                  "UP/DOWN is the side the model thinks is more likely {when}. It is a guess from past "
                  "patterns, not a promise; check 'How good is this model?' below to see how often it was right."),
    "Confidence": ("The model's chance that the price is {dir} {when}: {n}%; 50% = coin flip",
                   "The model's probability for the side shown above. It is shown only when, on test data, the stated "
                   "chances matched what really happened (within 2 points on average)."),
    "Last close": ("Price at the end of the last trading day",
                   "The latest closing price the model started from."),
    "Today": ("Price change of the latest session vs the day before",
              "Live move of the last bar on the chart. It is not part of the forecast."),
    "Horizon": ("How far ahead the forecast looks: {h} trading days = about {cal} calendar days",
                "The market is closed on weekends and holidays, so one calendar month has about 21 trading days. "
                "'1 month' here means 21 trading days, which is about 30 days on the calendar."),
}
QUALITY_HELP = {
    "acc": "Share of past cases where the model got UP/DOWN right, compared with simply always guessing UP.",
    "ic": "How well the model puts stocks in the right order each day (rank correlation of its score with the real return). Rough rule: below 0.01 = none, "
          "0.01-0.03 = weak, above 0.03 = useful. Not proven = the 95% interval of the test result includes zero.",
    "unseen": "Same ranking skill (rank-IC) on stocks the model never saw in training.",
}


def helper(key: str, pred: dict | None = None, horizon: int = 21) -> str:
    """Short line under a card's number; works without a prediction (placeholders)."""
    txt = HELP[key][0]
    if pred and key == "Confidence" and not pred.get("calibrated", True):
        return "Hidden: on test data this model's stated chance of a rise was off by more than 2 points (not calibrated)"
    d = ("higher" if pred["direction"] == "UP" else "lower") if pred else "higher or lower"
    n = f"{max(pred['confidence'], 1 - pred['confidence']) * 100:.0f}" if pred else "N"
    h = int(pred["horizon"] if pred else horizon)
    return txt.format(dir=d, when=registry.when(h), h=h, cal=round(h * 365 / 252), n=n)


def tip(key: str, horizon: int = 21) -> str:
    return HELP[key][1].format(when=registry.when(horizon))


# ---- verdicts ----
NONE, WEAK, USEFUL = 0, 1, 2
IC_LEVELS = ("none", "weak", "useful")


def ic_level(ic) -> int:
    """Rough rule of thumb for the rank information coefficient (day-by-day rank correlation of score and real return):
    < 0.01 none, 0.01-0.03 weak, > 0.03 useful. Negative = worse than useless."""
    if ic is None or not math.isfinite(ic) or ic < 0.01:
        return NONE
    return WEAK if ic <= 0.03 else USEFUL


def ic_verdict(ic, lo=None) -> str:
    """Chip text for an IC: its level name, or 'not proven' when a skill level is reached but the lower end of the
    95 % interval (`lo`, if the run stored one) is not above zero."""
    level = ic_level(ic)
    return "not proven" if level and lo is not None and math.isfinite(lo) and lo <= 0 else IC_LEVELS[level]


def beats(acc, base, eps: float = 0.005) -> str:
    """Direction accuracy vs 'always guess UP' (base): 'beats' / 'equal' (within eps) / 'worse' / 'n/a'."""
    if acc is None or base is None or not (math.isfinite(acc) and math.isfinite(base)):
        return "n/a"
    return "beats" if acc > base + eps else "worse" if acc < base - eps else "equal"


def _num(v):
    return v if isinstance(v, (int, float)) and math.isfinite(v) else None


def read_metrics(rid: str) -> dict:
    try:
        return json.loads((registry.run_dir(rid) / "metrics.json").read_text())
    except (OSError, ValueError):
        return {}


def quality(rid: str, jobs: dict) -> dict | None:
    """Everything the 'How good is this model?' block shows for the active run (None if it has no metrics)."""
    m = read_metrics(rid)
    if not m:
        return None
    p = jobs.get(rid, {}).get("params", {})
    acc, base, ic, unseen = (_num(m.get(k)) for k in ("test_diracc", "test_up_rate", "test_rank_ic", "unseen_rank_ic"))
    return {"run": rid, "model": registry.MODEL_NAME.get(jobs.get(rid, {}).get("type", "train_logistic"), "Logistic"),
            "horizon": int(m.get("horizon") or p.get("horizon") or 0), "start_year": p.get("start_year"),
            "acc": acc, "base": base, "acc_verdict": beats(acc, base),
            "ic": ic, "ic_level": ic_level(ic), "ic_verdict": ic_verdict(ic, _num(m.get("test_rank_ic_lo"))),
            "unseen_ic": unseen, "unseen_level": ic_level(unseen), "unseen_verdict": ic_verdict(unseen, _num(m.get("unseen_rank_ic_lo")))}


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
