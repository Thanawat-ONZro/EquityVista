"""FIFO job queue over QProcess: each job is `python -m <module>` in its own OS process (never train in the GUI
process). Protocol lines (`@@EV {json}`) update progress/metrics; everything else is appended to runs/<id>/log.txt.
Data jobs run in parallel up to `max_jobs`; training jobs (all cores + RAM) run one at a time."""
import json
import sys
import time

import psutil
from PySide6.QtCore import QObject, QProcess, QProcessEnvironment, Signal

from app.core import registry
from src import active
from src.jobs import PARAMS, TRAIN

PREFIX = "@@EV "
HEAVY = set(TRAIN)
PIPELINE = ("data_extend", "build_features", "train_lgbm", "evaluate")
HISTORY_CAP = 5000  # loss/metric points kept per job


class JobManager(QObject):
    added = Signal(str)
    changed = Signal(str)
    removed = Signal(str)
    log = Signal(str, str)  # (run_id, text line)
    finished = Signal(str)  # job ended as done / failed (toasts)
    active_changed = Signal()  # models/active.json was rewritten (Predict hot-reloads)

    def __init__(self, max_jobs=2, parent=None, restore=True):
        super().__init__(parent)
        self.max_jobs = max_jobs
        self.jobs = registry.load_all(restore)
        self._procs, self._buf, self._cancel = {}, {}, set()

    # ---- public API ----
    def submit(self, type_, params=None, module=None, after=None) -> str:
        rid = registry.new_id(type_)
        self.jobs[rid] = {
            "id": rid, "type": type_, "module": module or f"src.jobs.{type_}",
            "params": {**PARAMS.get(type_, {}), **(params or {})}, "status": "queued",
            "after": after, "created": time.time(), "started": None, "finished": None, "pct": 0.0, "stage": "",
            "epoch": None, "epochs": None, "loss": None, "val_loss": None,
            "metrics": {}, "artifacts": {}, "error": None, "history": []}
        registry.save(self.jobs[rid])
        self.added.emit(rid)
        self._pump()
        return rid

    def submit_chain(self, specs) -> list:
        """specs = [(type, params, module)]: each job waits for the previous one; if one doesn't end `done`,
        every later job is cancelled (see _pump)."""
        ids, prev = [], None
        for type_, params, module in specs:
            prev = self.submit(type_, params, module, after=prev)
            ids.append(prev)
        return ids

    def pipeline(self) -> list:
        return self.submit_chain([(t, None, None) for t in PIPELINE])

    def activate(self, rid):
        """Make a done training run the model Predict uses. Raises ValueError if it has no activatable model."""
        registry.activate(self.jobs[rid])
        self.active_changed.emit()

    def rollback(self, horizon):
        active.rollback(horizon)
        self.active_changed.emit()

    def cancel(self, rid):
        job = self.jobs[rid]
        if job["status"] == "queued":
            self._end(job, "cancelled")
        elif rid in self._procs:
            self._cancel.add(rid)
            self._kill_tree(self._procs[rid])

    def retry(self, rid) -> str:
        j = self.jobs[rid]
        return self.submit(j["type"], j["params"], j["module"])

    def delete(self, rid):
        if self.jobs[rid]["status"] in registry.LIVE:
            self.cancel(rid)
            return  # still shutting down: the user deletes it once it has ended
        del self.jobs[rid]
        registry.delete(rid)
        self.removed.emit(rid)

    def live_count(self) -> int:
        return sum(j["status"] in registry.LIVE for j in self.jobs.values())

    def shutdown(self):
        """App closing: no orphaned trainers; whatever was live is `interrupted` (as after a crash)."""
        for rid, proc in list(self._procs.items()):
            proc.finished.disconnect()
            self._kill_tree(proc)
            self._end(self.jobs[rid], "interrupted")
        self._procs.clear()

    # ---- internals ----
    @staticmethod
    def _kill_tree(proc):
        pid = proc.processId()
        try:
            if not pid:
                raise psutil.NoSuchProcess(0)
            root = psutil.Process(pid)
            for c in root.children(recursive=True):
                c.kill()
            root.kill()
        except psutil.NoSuchProcess:
            pass
        except psutil.Error:
            proc.kill()

    def _pump(self):
        for job in [j for j in self.jobs.values() if j["status"] == "queued"]:
            if job["status"] != "queued":  # ended by a nested _pump (dependency cascade)
                continue
            if job.get("after"):
                dep = self.jobs.get(job["after"], {}).get("status", "deleted")
                if dep in registry.LIVE:
                    continue
                if dep != "done":
                    job["error"] = f"stopped: {job['after']} {dep}"
                    self._end(job, "cancelled")
                    continue
            if len(self._procs) >= self.max_jobs:
                break
            if job["type"] in HEAVY and any(self.jobs[r]["type"] in HEAVY for r in self._procs):
                continue
            self._start(job)

    def _start(self, job):
        rid = job["id"]
        p = QProcess(self)
        p.setWorkingDirectory(str(registry.ROOT))
        p.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)  # tracebacks land in the log
        env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONUNBUFFERED", "1")
        env.insert("PYTHONIOENCODING", "utf-8")
        p.setProcessEnvironment(env)
        p.readyReadStandardOutput.connect(lambda rid=rid: self._read(rid))
        p.finished.connect(lambda code, _st, rid=rid: self._finished(rid, code))
        p.errorOccurred.connect(lambda err, rid=rid: self._start_failed(rid, err))
        self._procs[rid], self._buf[rid] = p, b""
        job.update(status="running", started=time.time())
        registry.save(job)
        self.changed.emit(rid)
        p.start(sys.executable, ["-u", "-m", job["module"], "--run-dir", str(registry.run_dir(rid)),
                                 "--params", json.dumps(job["params"])])

    def _start_failed(self, rid, err):
        if err == QProcess.ProcessError.FailedToStart and rid in self._procs:
            self._procs.pop(rid).deleteLater()
            job = self.jobs[rid]
            job["error"] = "failed to start python"
            self._end(job, "failed")

    def _read(self, rid):
        p = self._procs.get(rid)
        if p is None:
            return
        *lines, self._buf[rid] = (self._buf[rid] + bytes(p.readAllStandardOutput().data())).split(b"\n")
        for raw in lines:
            self._line(self.jobs[rid], raw.decode("utf-8", "replace").rstrip("\r"))

    def _line(self, job, text):
        if text.startswith(PREFIX):
            try:
                return self._event(job, json.loads(text[len(PREFIX):]))
            except (ValueError, KeyError, TypeError):
                pass  # malformed protocol line: keep it visible in the log
        with (registry.run_dir(job["id"]) / "log.txt").open("a", encoding="utf-8") as f:
            f.write(text + "\n")
        self.log.emit(job["id"], text)

    def _event(self, job, ev):
        t = ev["type"]
        if t == "progress":
            job.update(stage=ev["stage"], pct=ev["pct"])
            for k in ("epoch", "epochs", "loss", "val_loss"):
                if ev.get(k) is not None:
                    job[k] = ev[k]
            if ev.get("loss") is not None or ev.get("val_loss") is not None:
                job["history"].append({k: ev.get(k) for k in ("pct", "epoch", "loss", "val_loss")})
                del job["history"][:-HISTORY_CAP]
        elif t == "metric":
            job["metrics"][ev["name"]] = ev["value"]
        elif t == "done":
            job["artifacts"] = ev.get("artifacts", {})
        elif t == "error":
            job["error"] = ev["msg"]
            self._line(job, f"ERROR: {ev['msg']}")
        self.changed.emit(job["id"])

    def _finished(self, rid, code):
        self._read(rid)
        job = self.jobs[rid]
        tail = self._buf.pop(rid, b"")
        if tail:  # last line without newline
            self._line(job, tail.decode("utf-8", "replace"))
        self._procs.pop(rid).deleteLater()
        if rid in self._cancel:
            self._cancel.discard(rid)
            self._end(job, "cancelled")
        elif code == 0:
            job["pct"] = 100.0
            self._end(job, "done")
        else:
            job["error"] = job["error"] or f"exit code {code}"
            self._end(job, "failed")

    def _end(self, job, status):
        job.update(status=status, finished=time.time())
        registry.save(job)
        self.changed.emit(job["id"])
        if status in ("done", "failed"):
            self.finished.emit(job["id"])
        self._pump()
