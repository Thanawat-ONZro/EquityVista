import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.pages import predict as _predict  # noqa: E402,F401  (QtWebEngine must be imported before the QApplication exists)
from PySide6.QtWidgets import QApplication  # noqa: E402

import pytest  # noqa: E402

from app.core import registry  # noqa: E402
from src import active  # noqa: E402
from src.split import FEATS  # noqa: E402


@pytest.fixture(autouse=True)
def _isolated_stores(tmp_path_factory, monkeypatch):
    """No test touches the real runs/ or models/active.json (a stray queued job once ran against the real data)."""
    d = tmp_path_factory.mktemp("stores")
    monkeypatch.setattr(registry, "RUNS", d / "runs")
    monkeypatch.setattr(active, "RUNS", d / "runs")
    monkeypatch.setattr(active, "FILE", d / "models" / "active.json")


@pytest.fixture(scope="session", autouse=True)
def _features_untouched():
    """The real features parquet is read-only for tests: fail loudly if a test rewrote it."""
    before = (FEATS.stat().st_mtime_ns, FEATS.stat().st_size) if FEATS.exists() else None
    yield
    assert before == ((FEATS.stat().st_mtime_ns, FEATS.stat().st_size) if FEATS.exists() else None), "a test modified the real features file"


_app = QApplication.instance() or QApplication([])  # one GUI app for the whole run (QCoreApplication-only tests reuse it)
