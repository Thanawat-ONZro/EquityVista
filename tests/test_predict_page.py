"""Predict page regression: opening it (autostart) must never block the GUI thread, with or without an active model."""
import time

import numpy as np
import pandas as pd
import pytest
from PySide6.QtCore import QEventLoop, QTimer

from app.core.job_manager import JobManager
from app.pages import predict as pp
from src import active

SLOW = 1.0  # seconds a (fake) network / model call takes on the worker thread


def _bars(*_):
    time.sleep(SLOW)
    t = pd.bdate_range("2025-01-01", periods=300).astype("datetime64[ns]")
    c = 100 + np.cumsum(np.ones(300))
    return pd.DataFrame({"time": t, "open": c, "high": c + 1, "low": c - 1, "close": c, "volume": 1.0})


def _pred(t, h, hide=0):
    time.sleep(SLOW)
    return {"ticker": t, "horizon": h, "date": "2025-12-01", "last_close": 100.0, "direction": "UP", "expected_return": 0.02,
            "confidence": 0.55, "price_pred": 102.0, "range_95": [90.0, 114.0], "range_95_ret": [-0.1, 0.14], "stale_days": 0}


def _spin(page, seconds):
    """Run the event loop; return the longest gap between two 20 ms timer ticks (= longest GUI-thread stall)."""
    loop, gaps, last = QEventLoop(), [], [time.perf_counter()]

    def tick():
        now = time.perf_counter()
        gaps.append(now - last[0])
        last[0] = now
        if (page.run_key and page.btn.isEnabled()) or now - t0 > seconds:
            loop.quit()
    t0 = time.perf_counter()
    timer = QTimer(interval=20)
    timer.timeout.connect(tick)
    timer.start()
    loop.exec()
    timer.stop()
    return max(gaps)


@pytest.fixture
def page(monkeypatch):
    monkeypatch.setattr(pp, "fetch_ohlcv", _bars)
    monkeypatch.setattr(pp, "run_predict", _pred)
    p = pp.PredictPage(JobManager())
    p.resize(1000, 600)
    p.show()
    yield p
    p.hide()


def test_opens_without_model_and_gui_stays_responsive(page):
    t0 = time.perf_counter()
    page.autostart()
    assert time.perf_counter() - t0 < 0.2  # returns at once: the work is queued, not run here
    assert page.status.toolTip().startswith("Loading AAPL")
    assert _spin(page, 5) < 0.5  # a 1 s fetch on the worker never stalls the event loop
    assert page.status.toolTip() == pp.NO_MODEL and page.btn.isEnabled() and page.n_bars == 300


def test_opens_with_model_and_gui_stays_responsive(page, monkeypatch):
    monkeypatch.setattr(active, "path", lambda h: __file__)  # "an active model exists"
    page.autostart()
    assert _spin(page, 6) < 0.5
    assert page.status.toolTip() == "Done" and page.pred["direction"] == "UP" and page.btn.isEnabled()


def test_failing_worker_still_releases_the_page(page, monkeypatch):
    def boom(*_):
        raise SystemExit("lib called exit")  # even this must come back to the UI as an error
    monkeypatch.setattr(pp, "fetch_ohlcv", boom)
    page.autostart()
    assert _spin(page, 5) < 0.5
    assert page.btn.isEnabled() and page.status.toolTip().startswith("Error")


def test_one_forecast_after_repredict_and_ticker_switch(page, monkeypatch):
    monkeypatch.setattr(pp, "fetch_ohlcv", lambda *_: _bars())
    monkeypatch.setattr(pp, "run_predict", lambda t, h, hide=0: _pred(t, h, hide))
    monkeypatch.setattr(active, "path", lambda h: __file__)
    for t in ("AAA", "AAA", "BBB"):  # re-predict the same ticker, then switch
        page.ticker.setText(t)
        page.last_key = None
        page.start()
        assert page.forecast == []  # wiped the moment a run starts
        _spin(page, 6)
        assert len(page.forecast) == 1 and len(page.chart._lines) == 1  # one reference line with the marker, never two forecasts


def test_stale_ticker_result_is_dropped(page):
    page.run_key = ("BBB", 21, False)
    page.on_pred({**_pred("AAA", 21), "ticker": "AAA"})
    assert page.pred is None
