"""App UI state (runs/settings.json, gitignored): e.g. the Predict panel width. Users never see its format."""
import json

from app.core import registry

FILE = lambda: registry.RUNS / "settings.json"  # noqa: E731  (RUNS is monkeypatched in tests)


def load() -> dict:
    try:
        return json.loads(FILE().read_text())
    except (OSError, ValueError):
        return {}


def ui_get(key: str, default=None):
    return load().get("ui", {}).get(key, default)


def ui_set(key: str, value) -> None:
    cfg = load()
    cfg["ui"] = {**cfg.get("ui", {}), key: value}
    FILE().parent.mkdir(exist_ok=True)
    FILE().write_text(json.dumps(cfg, indent=2))
