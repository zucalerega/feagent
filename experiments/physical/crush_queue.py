"""A simple crush-run queue: N slots, each running experiments/physical/crush.py jobs one after another; plus a
pull loop that copies the results back to another machine. A plain foreground process (start it with nohup);
nothing is installed, so it does not survive a reboot.

    python experiments/physical/crush_queue.py run JOBS.json --meshes MESH_ROOT --runs RUNS_ROOT
        [--slots 3] [--threads 4] [--mem-max 12G] [--min-avail-gib 20] [--ccx-bin DIR]
    python experiments/physical/crush_queue.py sync --remote <spark-host>:RUNS_ROOT --local RUNS_ROOT [--every 600]

JOBS.json: [{"system": "...", "part": "abc_00000003", "mesh": "<dir under MESH_ROOT holding mesh.msh>"}, ...].
It is re-read whenever a slot is free, so jobs can be appended while the queue runs. Every job writes
RUNS_ROOT/nonlinear/<system>/<part>/<run-id>/ (crush.py); its state is in RUNS_ROOT/queue/<system>__<part>.json
(started / finished / returncode / run dir). A job that has a state file is not started again (delete the state
file to rerun it). When every job has finished, RUNS_ROOT/queue/ALL_DONE is written and `run` exits.

Each job runs as `systemd-run --user --scope -q -p MemoryMax=<mem-max>` (when systemd-run exists; --mem-max ''
turns it off) and starts only while MemAvailable >= min-avail-gib + mem-max. `sync` rsyncs
RUNS_ROOT/nonlinear/ (without the ccx_work restart dirs) every --every seconds, never deleting anything locally,
and stops after the sync that sees ALL_DONE. The queue state goes to <local>/../queue_<host>/.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
lock = threading.Lock()


def mem_available_gib() -> float:
    for ln in Path("/proc/meminfo").read_text().splitlines():
        if ln.startswith("MemAvailable:"):
            return int(ln.split()[1]) / 2**20
    return float("inf")


def _gib(s: str) -> float:
    return float(s[:-1]) * {"G": 1, "M": 1 / 1024, "T": 1024}[s[-1].upper()] if s else 0.0


def log(qdir: Path, msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(qdir / "queue.log", "a") as f:
        f.write(line + "\n")


def next_job(a, qdir: Path):
    with lock:
        for j in json.loads(Path(a.jobs).read_text()):
            st = qdir / f"{j['system']}__{j['part']}.json"
            if not st.exists():
                st.write_text(json.dumps({**j, "state": "claimed", "claimed": time.strftime("%Y-%m-%dT%H:%M:%S%z")}))
                return j, st
    return None, None


def slot(a, k: int, qdir: Path):
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    if a.ccx_bin:
        env["PATH"] = str(Path(a.ccx_bin).expanduser()) + os.pathsep + env["PATH"]
    while True:
        j, st = next_job(a, qdir)
        if j is None:
            return
        need = a.min_avail_gib + _gib(a.mem_max)
        while mem_available_gib() < need:
            log(qdir, f"slot {k}: waiting for memory ({mem_available_gib():.0f} < {need:.0f} GiB) before {j}")
            time.sleep(60)
        run_id = time.strftime("run-%Y%m%d-%H%M") + f"-s{k}"
        out = Path(a.runs) / "nonlinear" / j["system"] / j["part"]
        cmd = [sys.executable, str(HERE / "crush.py"), "--system", j["system"], "--part", j["part"],
               "--mesh-dir", str(Path(a.meshes) / j["mesh"]), "--out", str(out), "--run-id", run_id,
               "--threads", str(a.threads)]
        if a.mem_max and shutil.which("systemd-run"):
            cmd = ["systemd-run", "--user", "--scope", "-q", "-p", f"MemoryMax={a.mem_max}", *cmd]
        rec = {**j, "state": "running", "slot": k, "run_dir": str(out / run_id), "host": os.uname().nodename,
               "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "threads": a.threads, "mem_max": a.mem_max}
        st.write_text(json.dumps(rec, indent=1))
        log(qdir, f"slot {k}: start {j['system']}/{j['part']} -> {out / run_id}")
        with open(qdir / f"{j['system']}__{j['part']}.out", "a") as fo:
            rc = subprocess.call(cmd, stdout=fo, stderr=subprocess.STDOUT, env=env)
        rec.update(state="finished" if rc == 0 else "failed", returncode=rc,
                   finished=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
        st.write_text(json.dumps(rec, indent=1))
        log(qdir, f"slot {k}: {rec['state']} (rc {rc}) {j['system']}/{j['part']}")


def run(a):
    qdir = Path(a.runs) / "queue"
    qdir.mkdir(parents=True, exist_ok=True)
    (qdir / "ALL_DONE").unlink(missing_ok=True)
    log(qdir, f"queue start: {a.slots} slots x {a.threads} threads, MemoryMax {a.mem_max or 'off'}, jobs {a.jobs}")
    ts = [threading.Thread(target=slot, args=(a, k, qdir)) for k in range(a.slots)]
    for t in ts:
        t.start()
        time.sleep(5)
    for t in ts:
        t.join()
    (qdir / "ALL_DONE").write_text(time.strftime("%Y-%m-%dT%H:%M:%S%z\n"))
    log(qdir, "queue done")


def sync(a):
    local = Path(a.local)
    local.mkdir(parents=True, exist_ok=True)
    host, root = a.remote.split(":", 1)
    while True:
        done = subprocess.call(["ssh", "-o", "BatchMode=yes", host, f"test -e {root}/queue/ALL_DONE"]) == 0
        subprocess.call(["rsync", "-a", "--exclude", "ccx_work/", "-e", "ssh -o BatchMode=yes",
                         f"{host}:{root}/nonlinear/", f"{local}/"])
        subprocess.call(["rsync", "-a", "-e", "ssh -o BatchMode=yes", f"{host}:{root}/queue/", f"{local.parent}/queue_{host}/"])
        print(time.strftime("%H:%M:%S"), "synced", "(all done)" if done else "", flush=True)
        if done:
            return
        time.sleep(a.every)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("jobs"), r.add_argument("--meshes", required=True), r.add_argument("--runs", required=True)
    r.add_argument("--slots", type=int, default=3), r.add_argument("--threads", type=int, default=4)
    r.add_argument("--mem-max", default="12G"), r.add_argument("--min-avail-gib", type=float, default=20.0)
    r.add_argument("--ccx-bin", default=None)
    s = sub.add_parser("sync")
    s.add_argument("--remote", required=True), s.add_argument("--local", required=True)
    s.add_argument("--every", type=int, default=600)
    a = ap.parse_args()
    run(a) if a.cmd == "run" else sync(a)


if __name__ == "__main__":
    main()
