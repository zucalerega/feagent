"""Background job runner for the UI server.

Every mesh / solve / geometry job runs in its own child process
(``python -m feacore.ui.worker <json>``): gmsh is process-global and not thread-safe, and a
crashing or hanging solve must not take the server down. Server threads only supervise.

Two lanes, each with its own worker count:
  quick   solves (POST /api/solve) and geometry previews   (FEACORE_UI_QUICK_WORKERS, default 1)
  study   convergence studies                              (FEACORE_UI_STUDY_WORKERS, default 1)
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from feacore.runstore import RunStore, new_id, now_iso


@dataclass
class Job:
    kind: str                       # solve | study | geometry
    payload: dict
    lane: str
    timeout: float
    log: Path | None = None
    on_exit: Callable[["Job"], None] | None = None
    id: str = field(default_factory=lambda: new_id("job"))
    state: str = "queued"           # queued | running | done | failed | timeout
    created: float = field(default_factory=time.time)
    started: float | None = None
    finished: float | None = None
    pid: int | None = None
    returncode: int | None = None
    future: Future = field(default_factory=Future)

    def summary(self) -> dict:
        return {"id": self.id, "kind": self.kind, "lane": self.lane, "state": self.state, "pid": self.pid,
                "returncode": self.returncode, "created": self.created, "started": self.started,
                "finished": self.finished,
                "target": self.payload.get("run_id") or self.payload.get("study_id") or self.payload.get("problem_id")}


class JobRunner:
    def __init__(self, store: RunStore, quick_workers: int | None = None, study_workers: int | None = None,
                 solve_timeout: float | None = None, study_timeout: float | None = None):
        self.store = store
        self.workers = {
            "quick": int(quick_workers or os.environ.get("FEACORE_UI_QUICK_WORKERS", 1)),
            "study": int(study_workers or os.environ.get("FEACORE_UI_STUDY_WORKERS", 1)),
        }
        self.timeouts = {
            "solve": float(solve_timeout or os.environ.get("FEACORE_UI_SOLVE_TIMEOUT", 1800)),
            "study": float(study_timeout or os.environ.get("FEACORE_UI_STUDY_TIMEOUT", 4 * 3600)),
            "geometry": 300.0,
        }
        self.queues = {lane: queue.Queue() for lane in self.workers}
        self.jobs: dict[str, Job] = {}
        self._procs: dict[str, subprocess.Popen] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._inflight: dict[str, Job] = {}   # dedupe key -> job (geometry previews)

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> None:
        with self._lock:
            if self._threads:
                return
            for lane, n in self.workers.items():
                for k in range(max(1, n)):
                    t = threading.Thread(target=self._loop, args=(lane,), name=f"feacore-{lane}-{k}", daemon=True)
                    t.start()
                    self._threads.append(t)

    def stop(self) -> None:
        self._stop.set()
        with self._lock:
            procs = list(self._procs.values())
        for p in procs:
            if p.poll() is None:
                p.terminate()
        for p in procs:
            try:
                p.wait(5)
            except subprocess.TimeoutExpired:
                p.kill()

    # ------------------------------------------------------------------ submit
    def submit(self, kind: str, payload: dict, log: Path | None = None,
               on_exit: Callable[[Job], None] | None = None, dedupe: str | None = None) -> Job:
        lane = "study" if kind == "study" else "quick"
        with self._lock:
            if dedupe and dedupe in self._inflight and not self._inflight[dedupe].future.done():
                return self._inflight[dedupe]
            job = Job(kind=kind, payload={"kind": kind, **payload}, lane=lane,
                      timeout=self.timeouts.get(kind, 1800.0), log=log, on_exit=on_exit)
            self.jobs[job.id] = job
            if dedupe:
                self._inflight[dedupe] = job
            # keep the job table small
            if len(self.jobs) > 500:
                for jid in [j.id for j in self.jobs.values() if j.future.done()][:100]:
                    self.jobs.pop(jid, None)
        self.start()
        self.queues[lane].put(job)
        return job

    def status(self) -> dict:
        with self._lock:
            jobs = list(self.jobs.values())
        return {
            "workers": self.workers,
            "queued": {lane: q.qsize() for lane, q in self.queues.items()},
            "running": [j.summary() for j in jobs if j.state == "running"],
            "recent": [j.summary() for j in sorted(jobs, key=lambda j: -j.created)[:10]],
        }

    def queue_position(self, job: Job) -> int | None:
        if job.state != "queued":
            return None
        with self.queues[job.lane].mutex:
            items = list(self.queues[job.lane].queue)
        return items.index(job) + 1 if job in items else None

    # ------------------------------------------------------------------ worker loop
    def _loop(self, lane: str) -> None:
        q = self.queues[lane]
        while not self._stop.is_set():
            try:
                job = q.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                self._run(job)
            except Exception as e:  # noqa: BLE001 - never kill the lane thread
                job.state = "failed"
                if not job.future.done():
                    job.future.set_exception(e)

    def _run(self, job: Job) -> None:
        cmd = [sys.executable, "-m", "feacore.ui.worker", json.dumps(job.payload)]
        job.started = time.time()
        job.state = "running"
        logf = open(job.log, "ab") if job.log else subprocess.DEVNULL
        try:
            if job.log:
                logf.write(f"\n=== {now_iso()} {job.kind} {job.id}\n".encode())
                logf.flush()
            p = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, env=os.environ.copy())
            job.pid = p.pid
            with self._lock:
                self._procs[job.id] = p
            try:
                job.returncode = p.wait(timeout=job.timeout)
                job.state = "done" if job.returncode == 0 else "failed"
            except subprocess.TimeoutExpired:
                p.terminate()
                try:
                    p.wait(10)
                except subprocess.TimeoutExpired:
                    p.kill()
                    p.wait()
                job.returncode = p.returncode
                job.state = "timeout"
        finally:
            if job.log:
                logf.close()
            with self._lock:
                self._procs.pop(job.id, None)
            job.finished = time.time()
        if job.on_exit is not None:
            try:
                job.on_exit(job)
            except Exception:  # noqa: BLE001
                pass
        if not job.future.done():
            job.future.set_result(job)


def finalize_run(store: RunStore) -> Callable[[Job], None]:
    """on_exit hook: a solve job whose record is still queued/running died without reporting."""
    def hook(job: Job) -> None:
        rid = job.payload["run_id"]
        rec = store.get_run(rid)
        if rec.get("status") in ("queued", "running"):
            store.update_run(rid, status="failed", finished_at=now_iso(),
                             error=crash_message(job))
    return hook


def finalize_study(store: RunStore) -> Callable[[Job], None]:
    def hook(job: Job) -> None:
        sid = job.payload["study_id"]
        rec = store.get_study(sid)
        if rec.get("status") in ("queued", "running"):
            prog = rec.get("progress") or {}
            prog.update(stage="failed", message=crash_message(job))
            store.update_study(sid, status="failed", finished_at=now_iso(), error=crash_message(job),
                               progress=prog)
    return hook


def crash_message(job: Job) -> str:
    if job.state == "timeout":
        return f"worker timed out after {job.timeout:.0f} s"
    rc = job.returncode
    if rc is not None and rc < 0:
        return f"worker crashed (signal {-rc}); see worker.log"
    return f"worker exited with code {rc} without writing a result; see worker.log"
