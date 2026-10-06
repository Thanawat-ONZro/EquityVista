import json
import shutil

import pytest

from app.core import registry
from src import active


def test_save_load_delete_and_interrupted(tmp_path, monkeypatch):
    monkeypatch.setattr(registry, "RUNS", tmp_path)
    a, b = registry.new_id("x"), "zz-live"
    registry.save({"id": a, "status": "done", "finished": 1})
    registry.save({"id": b, "status": "running", "finished": None})
    (tmp_path / "junk").mkdir()  # dir without job.json is ignored
    jobs = registry.load_all()
    assert jobs[a]["status"] == "done" and jobs[b]["status"] == "interrupted"
    assert json.loads((tmp_path / b / "job.json").read_text())["status"] == "interrupted"  # persisted
    registry.delete(a)
    assert a not in registry.load_all()


def test_readonly_load_leaves_live_jobs_alone(tmp_path, monkeypatch):
    monkeypatch.setattr(registry, "RUNS", tmp_path)
    registry.save({"id": "x", "status": "running"})
    assert registry.load_all(restore=False)["x"]["status"] == "running"  # the Predict app never rewrites a job
    assert registry.load_all(restore=False)["x"]["status"] == "running"
    assert registry.load_all()["x"]["status"] == "interrupted"


def test_new_id_unique(tmp_path, monkeypatch):
    monkeypatch.setattr(registry, "RUNS", tmp_path)
    a = registry.new_id("x")
    registry.save({"id": a, "status": "done"})
    assert registry.new_id("x") != a


def _model_run(tmp_path, rid, h=21, type_="train_logistic", status="done"):
    job = {"id": rid, "type": type_, "status": status, "params": {"horizon": h}}
    registry.save(job)
    (tmp_path / rid / active.REL).parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / rid / active.REL).write_text("model")
    return job


def _isolate(tmp_path, monkeypatch):
    monkeypatch.setattr(registry, "RUNS", tmp_path)
    monkeypatch.setattr(active, "RUNS", tmp_path)
    monkeypatch.setattr(active, "FILE", tmp_path / "models" / "active.json")


def test_activate_rollback_and_path_per_horizon(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    a, b, c = _model_run(tmp_path, "a"), _model_run(tmp_path, "b"), _model_run(tmp_path, "c", h=5)
    assert active.read()["h"] == {} and active.path(21) is None  # no model, no fallback
    registry.activate(a)
    registry.activate(b)
    registry.activate(c)
    assert active.read()["h"] == {"21": "b", "5": "c"} and active.path(21) == tmp_path / "b" / active.REL
    active.rollback(21)
    assert active.run_id(21) == "a" and active.run_id(5) == "c"  # other horizon untouched
    active.rollback(21)  # swaps back
    assert active.run_id(21) == "b"
    shutil.rmtree(tmp_path / "b")  # active run's files vanished -> None, never a crash
    assert active.path(21) is None
    assert registry.summary({}) == ["5d · c", "21d · b"]


def test_old_active_json_format_reads_as_empty(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    active.FILE.parent.mkdir()
    active.FILE.write_text('{"lgbm": "x", "cnn": "legacy", "prev": {"lgbm": "legacy"}}')
    assert active.read() == {"h": {}, "prev": {}}


def test_activate_rejects_unfinished_or_wrong_type(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    for job in (_model_run(tmp_path, "r", status="failed"), {"id": "e", "type": "evaluate", "status": "done"}):
        with pytest.raises(ValueError):
            registry.activate(job)
    assert not active.FILE.exists()
    registry.activate(_model_run(tmp_path, "rf", type_="train_rf"))  # every model type is a classifier: any can be deployed
    assert active.run_id(21) == "rf"
