import time

import pytest
from PySide6.QtCore import QCoreApplication

from app.core import registry
from app.core.job_manager import JobManager
from src.split import FEATS

FAKE = "tests.fake_job"


@pytest.fixture
def mgr(tmp_path, monkeypatch):
    monkeypatch.setattr(registry, "RUNS", tmp_path)
    app = QCoreApplication.instance() or QCoreApplication([])
    m = JobManager(max_jobs=2)
    yield m
    m.shutdown()
    app.processEvents()


def wait(cond, timeout=30):
    end = time.time() + timeout
    while not cond():
        assert time.time() < end, "timeout"
        QCoreApplication.processEvents()
        time.sleep(0.01)


def test_success(mgr):
    rid = mgr.submit("fake", {"steps": 3}, module=FAKE)
    wait(lambda: mgr.jobs[rid]["status"] == "done")
    j = mgr.jobs[rid]
    assert j["pct"] == 100 and j["metrics"] == {"ic": 0.5} and j["artifacts"] == {"model": "model/x"}
    assert len(j["history"]) == 3
    assert "step 2" in registry.read_log(rid)
    assert (registry.run_dir(rid) / "metrics.json").exists()


def test_failure_keeps_error_and_traceback(mgr):
    rid = mgr.submit("fake", {"steps": 4, "fail": 1}, module=FAKE)
    wait(lambda: mgr.jobs[rid]["status"] == "failed")
    assert mgr.jobs[rid]["error"] == "boom"
    assert "RuntimeError" in registry.read_log(rid)


def test_cancel_running_and_queued(mgr):
    mgr.max_jobs = 1
    a = mgr.submit("fake", {"steps": 100, "delay": 1}, module=FAKE)
    b = mgr.submit("fake", {"steps": 1}, module=FAKE)
    wait(lambda: mgr.jobs[a]["pct"] > 0)
    assert mgr.jobs[b]["status"] == "queued"
    mgr.cancel(b)
    assert mgr.jobs[b]["status"] == "cancelled"
    mgr.cancel(a)
    wait(lambda: mgr.jobs[a]["status"] == "cancelled")
    assert mgr.live_count() == 0


def test_training_jobs_run_one_at_a_time(mgr):
    a = mgr.submit("train_rf", {"steps": 100, "delay": 1}, module=FAKE)
    b = mgr.submit("train_logistic", {"steps": 1}, module=FAKE)
    c = mgr.submit("fake", {"steps": 1}, module=FAKE)  # a data job is not blocked by a training one
    wait(lambda: mgr.jobs[c]["status"] == "done")
    assert mgr.jobs[a]["status"] == "running" and mgr.jobs[b]["status"] == "queued"
    mgr.cancel(a)
    wait(lambda: mgr.jobs[b]["status"] == "done")


def test_retry_and_delete(mgr):
    rid = mgr.submit("fake", {"steps": 1}, module=FAKE)
    wait(lambda: mgr.jobs[rid]["status"] == "done")
    rid2 = mgr.retry(rid)
    assert rid2 != rid and mgr.jobs[rid2]["params"] == mgr.jobs[rid]["params"]
    wait(lambda: mgr.jobs[rid2]["status"] == "done")
    mgr.delete(rid)
    assert rid not in mgr.jobs and not registry.run_dir(rid).exists()


def test_restore_marks_interrupted(mgr, tmp_path):
    registry.save({"id": "old", "status": "running", "finished": None})
    assert JobManager().jobs["old"]["status"] == "interrupted"


@pytest.mark.skipif(not FEATS.exists(), reason=f"{FEATS} missing")
def test_real_tiny_train_lgbm(mgr):
    rid = mgr.submit("train_lgbm", {"frac": 0.02, "rounds": 20, "trials": 0, "horizon": 21})
    wait(lambda: mgr.jobs[rid]["status"] not in registry.LIVE, timeout=900)
    j = mgr.jobs[rid]
    assert j["status"] == "done", (j["error"], registry.read_log(rid)[-2000:])
    assert (registry.run_dir(rid) / j["artifacts"]["model"]).exists()
    assert {"test_ic", "test_rank_ic", "test_diracc", "test_f1", "threshold"} <= set(j["metrics"]) and j["history"]


def test_chain_runs_in_order(mgr):
    ids = mgr.submit_chain([("fake", {"steps": 2}, FAKE)] * 3)
    wait(lambda: mgr.live_count() == 0)
    js = [mgr.jobs[i] for i in ids]
    assert [j["status"] for j in js] == ["done"] * 3
    assert js[0]["finished"] <= js[1]["started"] and js[1]["finished"] <= js[2]["started"]  # never overlapped


def test_chain_stops_on_failure(mgr):
    done = []
    mgr.finished.connect(done.append)
    ids = mgr.submit_chain([("fake", {"steps": 1}, FAKE), ("fake", {"steps": 4, "fail": 1}, FAKE),
                            ("fake", {"steps": 1}, FAKE), ("fake", {"steps": 1}, FAKE)])
    wait(lambda: mgr.live_count() == 0)
    assert [mgr.jobs[i]["status"] for i in ids] == ["done", "failed", "cancelled", "cancelled"]
    assert ids[1] in mgr.jobs[ids[2]]["error"] and mgr.jobs[ids[3]]["started"] is None
    assert done == ids[:2]  # toast signal: done + failed only, not the cancelled tail
