"""Forecast panel: plain-language helpers, comparison data from fake run dirs, splitter width persistence."""
import json

import pytest

from app.core import explain, registry, settings
from app.core.job_manager import JobManager
from app.pages import predict as pp


def test_helper_texts_with_and_without_prediction():
    assert "higher or lower" in explain.helper("Direction", None, 21) and "in 1 month (21 trading days)" in explain.helper("Direction")
    assert "21 trading days ≈ 30 calendar days" in explain.helper("Horizon") and registry.when(7) == "in 7 trading days"
    pred = {"direction": "DOWN", "confidence": 0.554, "horizon": 63}
    assert "lower in 3 months (63 trading days)" in explain.helper("Direction", pred)
    assert "lower in 3 months (63 trading days): 55%" in explain.helper("Confidence", pred)  # P(up) 0.554 with DOWN: the helper speaks for the side shown
    assert "higher in 3 months (63 trading days): 55%" in explain.helper("Confidence", {**pred, "direction": "UP"})
    assert "not calibrated" in explain.helper("Confidence", {**pred, "calibrated": False})
    assert all(explain.helper(k) and explain.tip(k) for k in explain.HELP)


def _run(rid, type_, ic, acc=0.53, h=21, status="done", up=0.54, val=None):
    d = registry.run_dir(rid)
    d.mkdir(parents=True)
    (d / "metrics.json").write_text(json.dumps({"horizon": h, "test_rank_ic": ic, "test_diracc": acc, "test_up_rate": up,
                                                "val_rank_ic": ic if val is None else val, "unseen_rank_ic": 0.02}))
    return rid, {"id": rid, "type": type_, "status": status, "params": {"horizon": h, "start_year": 1962}}


def test_comparison_best_per_type_same_horizon():
    jobs = dict([_run("a-lgbm", "train_lgbm", 0.01), _run("b-lgbm", "train_lgbm", 0.03), _run("c-log", "train_logistic", 0.02),
                 _run("d-rf63", "train_rf", 0.09, h=63), _run("e-log", "train_logistic", 0.5, status="failed")])
    rows = explain.comparison(jobs, 21, "a-lgbm")
    assert [(r["model"], r["run"], r["active"]) for r in rows] == [("Logistic", "c-log", False), ("LightGBM", "a-lgbm", True)] \
        or [(r["model"], r["run"], r["active"]) for r in rows] == [("LightGBM", "a-lgbm", True), ("Logistic", "c-log", False)]
    assert explain.comparison(jobs, 21, None)[0]["run"] in ("b-lgbm", "c-log")
    assert [r["model"] for r in explain.comparison(jobs, 63, None)] == ["Random Forest"]
    assert explain.comparison({}, 21, None) == []
    two = dict([_run("x-rf", "train_rf", 0.09, val=0.01), _run("y-rf", "train_rf", 0.02, val=0.05)])
    assert explain.comparison(two, 21, None)[0]["run"] == "y-rf"  # picked by VALIDATION, not by the better test number


def test_settings_ui_roundtrip(tmp_path):
    settings.ui_set("w", 411)
    assert settings.ui_get("w") == 411 and settings.ui_get("nope", 7) == 7
    settings.ui_set("v", 1)  # another key keeps the first
    assert settings.ui_get("w") == 411


def test_clamp_width():
    assert pp.clamp_width(10) == pp.PANEL_MIN and pp.clamp_width(9999) == pp.PANEL_MAX and pp.clamp_width("x") == pp.PANEL_W


def test_panel_width_persists_and_placeholders_show():
    page = pp.PredictPage(JobManager())
    page.resize(1200, 700)
    page.show()
    assert page.helplbl["Direction"].text() and all(l.text() for l in page.helplbl.values())  # no pred: explained placeholders
    assert "Train the other models" in page.cmp_note.text()
    page.splitter.setSizes([800, 400])
    page._moved()
    page.save_width()
    assert settings.ui_get(pp.PANEL_KEY) == page.splitter.sizes()[1] == 400
    page2 = pp.PredictPage(JobManager())
    page2.resize(1200, 700)
    page2.show()
    assert abs(page2.splitter.sizes()[1] - 400) <= 2
