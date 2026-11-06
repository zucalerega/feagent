"""Harness-vs-gold benchmark over the imported gold set, in parallel on the Spark.

    run_all.py [--shakedown N] [--mesh-workers 12] [--max-solves 6] [--parts P ...]

MESH pool (CPU, threads driving processes): per part, gold_case.py mesh2 (auto_spec + ids groups for every face
its sets use + feagent check, hint-driven fixes) then gold_case.py cases (adapter -> case.json per set + group
mapping check). One mesh per part serves all its BC sets (the mesh does not depend on the BCs).
SOLVE scheduler (cuDSS): admits a case only if est = 28 KB/DOF * DOFs + 2 GB fits in
MemAvailable - 20 GiB - sum(est of running solves); at most --max-solves at once; each solve runs in its own
`systemd-run --user --scope -p MemoryMax=1.5*est`. A global watchdog stops every running solve and pauses
admission while MemAvailable < 12 GiB. Per-case timeout 15 min (mesh stage per part: 30 min, two harness
meshes at most 15 min each). Resumable: finished cases (status.json) and meshed parts are skipped.
"""
from __future__ import annotations

import argparse, json, os, queue, random, subprocess, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HOME = Path.home()
GOLD = HOME / "feagent-mh" / "benchmarks" / "gold"
RUNS = HOME / "gold-runs"
PY = str(HOME / "micromamba/envs/feagent/bin/python")
SCRIPT = str(HOME / "gold-catalog/scripts/gold_case.py")
ENV = {"PYTHONPATH": f"{HOME}/feagent-mh/meshing-harness/src:{HOME}/feagent-mh", "MPLBACKEND": "Agg",
       "FEACORE_GOLD_DIR": str(GOLD)}
GiB = 1 << 30
FINAL = {"ok", "mesh_fail", "gate_fail", "solve_fail", "timeout"}
LOCK = threading.Lock()


def log(msg: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    with LOCK:
        with open(RUNS / "driver.log", "a") as f:
            f.write(line + "\n")
    print(line, flush=True)


def mem_available() -> int:
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    return 0


class Scope:
    """A child process in its own transient user scope (stoppable as a unit)."""
    n = 0

    def __init__(self, tag: str, args: list[str], mem_max_gb: float, out: Path):
        with LOCK:
            Scope.n += 1
            self.unit = f"goldrun-{tag}-{os.getpid()}-{Scope.n}"
        self.out = out
        cmd = ["systemd-run", "--user", "--scope", "-q", f"--unit={self.unit}", "-p", f"MemoryMax={max(1, int(mem_max_gb * 1024))}M",
               "env"] + [f"{k}={v}" for k, v in ENV.items()] + [PY, SCRIPT] + args
        self.f = open(out, "a")
        self.t0 = time.time()
        self.p = subprocess.Popen(cmd, stdout=self.f, stderr=subprocess.STDOUT, cwd=str(RUNS))
        self.killed = None

    def stop(self, why: str) -> None:
        self.killed = why
        subprocess.run(["systemctl", "--user", "stop", f"{self.unit}.scope"], capture_output=True)

    def wait(self, timeout: float) -> int | None:
        try:
            rc = self.p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self.stop("timeout")
            try:
                self.p.wait(timeout=60)
            except subprocess.TimeoutExpired:
                self.p.kill()
            rc = None
        self.f.close()
        self.seconds = time.time() - self.t0
        return rc


class Runner:
    def __init__(self, a):
        self.a = a
        self.solve_q: list[dict] = []
        self.running: dict[str, tuple[Scope, dict]] = {}
        self.paused = False
        self.mesh_done = False
        self.cv = threading.Condition()
        self.stop_all = False
        self.gpu_samples: list[tuple[float, int, int]] = []   # (t, util %, running solves)
        self.counts = {"queued": 0, "done": 0}
        self.ceiling = (mem_available() - 20 * GiB) / 1e9

    # ------------------------------------------------------------------ watchdog + GPU sampler
    def watchdog(self):
        while not self.stop_all:
            av = mem_available()
            if av < 12 * GiB:
                with self.cv:
                    victims = list(self.running.values())
                    self.paused = True
                for sc, item in victims:
                    if sc is not None:
                        sc.stop("watchdog")
                log(f"WATCHDOG MemAvailable {av / GiB:.1f} GiB < 12 GiB: stopped {len(victims)} solves "
                    f"({', '.join(i['set'] for _, i in victims)}); admission paused")
                with open(RUNS / "watchdog.log", "a") as f:
                    f.write(f"{time.strftime('%H:%M:%S')} avail {av / GiB:.1f} GiB, stopped {[i['set'] for _, i in victims]}\n")
            elif self.paused and av > 30 * GiB:
                with self.cv:
                    self.paused = False
                    self.cv.notify_all()
                log(f"watchdog: MemAvailable {av / GiB:.1f} GiB, admission resumed")
            time.sleep(1)

    def gpu_sampler(self):
        while not self.stop_all:
            try:
                o = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                                   capture_output=True, text=True, timeout=10).stdout.strip()
                self.gpu_samples.append((time.time(), int(o.split()[0]), len(self.running)))
            except Exception:  # noqa: BLE001
                pass
            time.sleep(5)

    # ------------------------------------------------------------------ mesh pool
    def mesh_part(self, part: str, sets: list[str]):
        work = RUNS / part
        work.mkdir(parents=True, exist_ok=True)
        ms = work / "mesh_status.json"
        st = json.loads(ms.read_text()) if ms.is_file() else None
        if st is None or st.get("status") not in ("ok", "gate_fail", "mesh_fail", "timeout"):
            sc = Scope(f"mesh-{part}", ["mesh2", part, str(work), "--max-dofs", str(int((self.ceiling - 2.0) * 1e9 / 28e3))], 24, work / "mesh_stage.out")
            rc = sc.wait(1800)
            st = json.loads(ms.read_text()) if ms.is_file() else {"part": part}
            if rc is None:
                st["status"] = "timeout"; st["error"] = "mesh stage > 30 min"
            elif "status" not in st:
                st["status"] = "mesh_fail"; st["error"] = f"mesh2 rc={rc}, no mesh_status"
            st["mesh_stage_wall_s"] = round(sc.seconds, 1)
            ms.write_text(json.dumps(st, indent=1))
            log(f"mesh {part}: {st['status']} {st.get('failed_gates', '')} fixes={st.get('fixes', [])} "
                f"nodes={st.get('mesh', {}).get('n_nodes')} {sc.seconds:.0f}s")
        if st["status"] in ("mesh_fail", "timeout"):
            for s in sets:
                self.finish(s, part, {"status": st["status"], "stage": "mesh", "error": st.get("error"),
                                      "mesh": self.mesh_summary(st)})
            return
        n_dofs = 3 * int(st.get("mesh", {}).get("n_nodes") or 0)
        est = 28e3 * n_dofs / 1e9 + 2.0
        if est > self.ceiling:
            for s in sets:
                self.finish(s, part, {"status": "solve_fail", "stage": "admission", "dofs_est": n_dofs,
                                      "error": f"mesh too large to solve: {n_dofs:,} DOFs, est {est:.0f} GB > budget ceiling {self.ceiling:.0f} GB",
                                      "mesh": self.mesh_summary(st)})
            return
        todo = [s for s in sets if not (work / s / "case.json").is_file()]
        if todo:
            sc = Scope(f"cases-{part}", ["cases", part, str(work), "--sets", *todo], 24, work / "cases_stage.out")
            sc.wait(900)
        n_dofs = 3 * int(st.get("mesh", {}).get("n_nodes") or 0)
        for s in sets:
            if not (work / s / "case.json").is_file():
                err = (work / s / "case_error.txt").read_text() if (work / s / "case_error.txt").is_file() else "no case.json"
                self.finish(s, part, {"status": "solve_fail", "stage": "case", "error": err, "mesh": self.mesh_summary(st)})
                continue
            item = {"set": s, "part": part, "dofs": n_dofs, "est_gb": 28e3 * n_dofs / 1e9 + 2.0, "mesh_st": st,
                    "t_queued": time.time(), "tries": 0}
            with self.cv:
                self.solve_q.append(item)
                self.cv.notify_all()

    @staticmethod
    def mesh_summary(st: dict) -> dict:
        return {k: st.get(k) for k in ("status", "failed_gates", "warn_gates", "fixes", "mesh", "mesh_stage_s",
                                       "mesh_stage_wall_s", "gates", "peak_rss_gb")}

    # ------------------------------------------------------------------ solve scheduler
    def solve_loop(self):
        ceiling = self.ceiling
        while True:
            too_big, pick = [], None
            with self.cv:
                while True:
                    if self.mesh_done and not self.solve_q and not self.running:
                        return
                    too_big = [i for i in self.solve_q if i["est_gb"] > ceiling]
                    for i in too_big:
                        self.solve_q.remove(i)
                    if too_big:
                        break
                    if not self.paused and self.solve_q and len(self.running) < self.a.max_solves:
                        budget = (mem_available() - 20 * GiB) / 1e9 - sum(i["est_gb"] for _, i in self.running.values())
                        pick = next((i for i in self.solve_q if i["est_gb"] <= budget), None)
                        if pick is not None:
                            self.solve_q.remove(pick)
                            self.running[pick["set"]] = (None, pick)      # reserve its memory before starting
                            break
                    self.cv.wait(timeout=2)
            for i in too_big:
                self.finish(i["set"], i["part"], {"status": "solve_fail", "stage": "admission",
                                                  "error": f"est {i['est_gb']:.1f} GB > budget ceiling {ceiling:.1f} GB",
                                                  "dofs_est": i["dofs"], "mesh": self.mesh_summary(i["mesh_st"])})
            if pick is None:
                continue
            item = pick
            d = RUNS / item["part"] / item["set"]
            item["tries"] += 1
            item["t_start"] = time.time()
            sc = Scope(f"solve-{item['set']}", ["solve", item["set"], str(RUNS / item["part"])], 1.5 * item["est_gb"],
                       d / "solve.out")
            with self.cv:
                self.running[item["set"]] = (sc, item)
            threading.Thread(target=self.solve_wait, args=(sc, item), daemon=True).start()

    def solve_wait(self, sc: Scope, item: dict):
        rc = sc.wait(900)
        requeue = False
        try:
            requeue = self._solve_done(sc, item, rc)
        finally:
            with self.cv:
                self.running.pop(item["set"], None)
                if requeue:
                    self.solve_q.append(item)
                self.cv.notify_all()

    def _solve_done(self, sc: Scope, item: dict, rc):
        d = RUNS / item["part"] / item["set"]
        if sc.killed == "watchdog" and item["tries"] < 2:
            log(f"solve {item['set']}: stopped by watchdog, requeued")
            return True
        res = json.loads((d / "result.json").read_text()) if (d / "result.json").is_file() and rc is not None else None
        mapping = json.loads((d / "mapping.json").read_text()) if (d / "mapping.json").is_file() else {}
        rec = {"stage": "solve", "solve_wall_s": round(sc.seconds, 2), "queue_wait_s": round(item["t_start"] - item["t_queued"], 1),
               "est_gb": round(item["est_gb"], 2), "mem_max_gb": round(1.5 * item["est_gb"], 2), "solve_rc": rc,
               "mapping_ok": all(v.get("ok") for v in mapping.values()) if mapping else None,
               "mesh": self.mesh_summary(item["mesh_st"])}
        if rc is None:
            rec["status"] = "timeout"; rec["error"] = f"solve killed: {sc.killed}"
        elif res is None or res.get("status") != "ok":
            rec["status"] = "solve_fail"
            tail = (d / "solve.out").read_text()[-1500:] if (d / "solve.out").is_file() else ""
            rec["error"] = (res or {}).get("error") or f"rc={rc} {sc.killed or ''} {tail}"
        else:
            rec["status"] = "ok" if item["mesh_st"].get("status") == "ok" else "gate_fail"
        if res is not None:
            rec["result"] = {k: res.get(k) for k in ("n_dofs", "n_nodes", "n_cells", "solver_wall_s", "total_s", "linear_solver",
                                                     "peak_rss_gb", "gold_dofs", "compare", "warnings", "error")}
        self.finish(item["set"], item["part"], rec)
        return False

    def finish(self, sid: str, part: str, rec: dict):
        d = RUNS / part / sid
        d.mkdir(parents=True, exist_ok=True)
        rec.update(set=sid, part=part, finished=time.strftime("%Y-%m-%dT%H:%M:%S"))
        (d / "status.json").write_text(json.dumps(rec, indent=1))
        with self.cv:
            self.counts["done"] += 1
            n = self.counts["done"]
        cmp = rec.get("result", {}).get("compare") or []
        inb = sum(c["in_band"] for c in cmp)
        log(f"[{n}/{self.counts['queued']}] {sid}: {rec['status']} dofs={rec.get('result', {}).get('n_dofs')} "
            f"in-band {inb}/{len(cmp)} {rec.get('error', '')[:160] if rec.get('error') else ''}")

    # ------------------------------------------------------------------ main
    def run(self, work: dict[str, list[str]]):
        t0 = time.time()
        self.counts["queued"] = sum(len(v) for v in work.values())
        log(f"start: {len(work)} parts, {self.counts['queued']} cases; MemAvailable {mem_available() / GiB:.1f} GiB")
        threading.Thread(target=self.watchdog, daemon=True).start()
        threading.Thread(target=self.gpu_sampler, daemon=True).start()
        sched = threading.Thread(target=self.solve_loop, daemon=True)
        sched.start()
        with ThreadPoolExecutor(self.a.mesh_workers) as ex:
            futs = [ex.submit(self.mesh_part, p, s) for p, s in work.items()]
            for f in futs:
                try:
                    f.result()
                except Exception as e:  # noqa: BLE001
                    log(f"mesh worker error: {type(e).__name__}: {e}")
        with self.cv:
            self.mesh_done = True
            self.cv.notify_all()
        sched.join()
        self.stop_all = True
        wall = time.time() - t0
        busy = [u for _, u, n in self.gpu_samples if n > 0]
        summary = {"wall_s": round(wall, 1), "cases": self.counts["done"], "parts": len(work),
                   "cases_per_min": round(self.counts["done"] / (wall / 60), 2) if wall > 0 else None,
                   "gpu_util_mean_all": round(sum(u for _, u, _ in self.gpu_samples) / len(self.gpu_samples), 1) if self.gpu_samples else None,
                   "gpu_util_mean_while_solving": round(sum(busy) / len(busy), 1) if busy else None,
                   "gpu_samples": len(self.gpu_samples), "mesh_workers": self.a.mesh_workers, "max_solves": self.a.max_solves}
        runs = json.loads((RUNS / "runs.json").read_text()) if (RUNS / "runs.json").is_file() else []
        runs.append({**summary, "label": self.a.label, "started": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(t0))})
        (RUNS / "runs.json").write_text(json.dumps(runs, indent=1))
        log(f"done: {json.dumps(summary)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shakedown", type=int, default=0, help="only N cases: N random parts, one set each")
    ap.add_argument("--mesh-workers", type=int, default=12)
    ap.add_argument("--max-solves", type=int, default=6)
    ap.add_argument("--parts", nargs="*")
    ap.add_argument("--label", default="full")
    a = ap.parse_args()
    RUNS.mkdir(exist_ok=True)
    idx = json.loads((GOLD / "index.json").read_text())
    by_part: dict[str, list[str]] = {}
    for sid, row in sorted(idx["sets"].items()):
        by_part.setdefault(row["part"], []).append(sid)
    if a.parts:
        by_part = {p: by_part[p] for p in a.parts}
    if a.shakedown:
        rng = random.Random(7)
        parts = sorted(by_part)
        rng.shuffle(parts)
        by_part = {p: [rng.choice(by_part[p])] for p in parts[:a.shakedown]}
    work = {}
    for p, sets in by_part.items():
        todo = []
        for s in sets:
            st = RUNS / p / s / "status.json"
            if st.is_file() and json.loads(st.read_text()).get("status") in FINAL:
                continue
            todo.append(s)
        if todo:
            work[p] = todo
    # big parts first so the long meshes start early
    work = dict(sorted(work.items(), key=lambda kv: -idx["parts"].get(kv[0], {}).get("faces", 0)))
    Runner(a).run(work)


if __name__ == "__main__":
    main()
