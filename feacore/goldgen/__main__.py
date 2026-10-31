"""Gold-benchmark generator CLI.

  python -m feacore.goldgen screen STEP_DIR --out parts.jsonl [--workers 8]
  python -m feacore.goldgen mesh parts.jsonl --out gold/ [--parts 100] [--levels 3] [--h0-frac 0.125]
         [--workers 12] [--threads 1] [--shard 0/4]
  python -m feacore.goldgen run parts.jsonl STEP_DIR --out gold/ [--parts 3] [--bcs 10] [--levels 3]
         [--mesh-workers 12] [--linear-solver cudss] [--shard 0/4]

``screen`` accepts / rejects every STEP file (one process per file: gmsh is process-global).
``mesh`` meshes the selected parts in parallel, one process per part (``--threads`` gmsh threads
each), every level written as out/_meshes/<part>/L<k>.msh plus an L<k>.json sidecar; existing
levels are reused, so meshing can run on CPU nodes ahead of the GPU.
``run`` solves and certifies each BC set (feacore.goldgen.run). With ``--mesh-workers`` > 1 it meshes
all selected parts in a process pool and solves each part as soon as its meshes are ready, so the
GPU works while gmsh does. Re-running skips BC sets whose private/reference.json exists.
``--shard i/n`` keeps every n-th selected part starting at i: n machines split the same list
deterministically.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, as_completed, wait
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path


def _screen_one(path: str):
    from feacore.goldgen.parts import Rejected, screen

    pid = "abc_" + Path(path).name.split("_")[0]
    try:
        return {"ok": True, "path": path, "part": json.loads(screen(path, pid).to_json())}
    except Rejected as e:
        return {"ok": False, "path": path, "id": pid, "reason": str(e)[:200]}
    except Exception as e:  # noqa: BLE001 - broken CAD must not stop the batch
        return {"ok": False, "path": path, "id": pid, "reason": f"{type(e).__name__}: {str(e)[:180]}"}


SCREEN_TIMEOUT = 120   # s per STEP file: some ABC files send the CAD kernel into an endless loop


def _screen_isolated(path: str) -> dict:
    """Screen one file in its own interpreter: a file that crashes (segfault / double free) or
    hangs the CAD kernel is rejected without touching the others."""
    pid = "abc_" + Path(path).name.split("_")[0]
    code = ("import json, sys; from feacore.goldgen.__main__ import _screen_one; "
            "print('@@ROW@@' + json.dumps(_screen_one(sys.argv[1])))")
    try:
        p = subprocess.run([sys.executable, "-W", "ignore", "-c", code, path], capture_output=True,
                           text=True, timeout=SCREEN_TIMEOUT, errors="replace")
    except subprocess.TimeoutExpired:
        return {"ok": False, "path": path, "id": pid, "reason": f"CAD kernel timeout (> {SCREEN_TIMEOUT} s)"}
    rows = [ln for ln in p.stdout.splitlines() if "@@ROW@@" in ln]
    if p.returncode or not rows:
        return {"ok": False, "path": path, "id": pid, "reason": f"crashed the CAD kernel (exit {p.returncode})"}
    return json.loads(rows[-1].split('@@ROW@@', 1)[1])


def cmd_screen(a) -> int:
    files = sorted(str(p) for p in Path(a.step_dir).rglob("*.step"))
    with ThreadPoolExecutor(a.workers) as ex:              # threads only wait on the subprocesses
        rows = list(ex.map(_screen_isolated, files))
    with open(a.out, "w") as f:                            # input order: the same file on every machine
        for r in rows:
            f.write(json.dumps(r) + "\n")
    n_ok = sum(r["ok"] for r in rows)
    n_bad = sum(not r["ok"] and r["reason"].startswith(("crashed", "CAD kernel timeout")) for r in rows)
    print(f"screened {len(files)}: {n_ok} accepted, {n_bad} crashed / timed out -> {a.out}")
    return 0


def _select(a) -> list[tuple[dict, list]]:
    """(screen row, BC sets) for the parts this invocation handles: accepted, with at least
    --min-bcs distinct BC sets, the first --parts of them, then this machine's --shard."""
    from feacore.goldgen.bcgen import generate
    from feacore.goldgen.parts import Part

    out = []
    for row in map(json.loads, open(a.parts)):
        if len(out) >= a.parts_n:
            break
        if not row["ok"]:
            continue
        sets = generate(Part.from_json(json.dumps(row["part"])), a.bcs)
        if len(sets) < a.min_bcs:
            print(f"[{row['part']['id']}] skipped: only {len(sets)} distinct BC sets (< {a.min_bcs})", flush=True)
            continue
        out.append((row, sets))
    i, n = (int(x) for x in a.shard.split("/"))
    return out[i::n]


def _mesh_part(job) -> list[dict]:
    """Every level of one part (worker process). A complete cached sequence (all sidecars) is
    reused; otherwise all levels are (re)made with one scheme (feacore.goldgen.mesh.mesh_part)."""
    from feacore.goldgen.mesh import fit_h0, mesh_part
    from feacore.goldgen.parts import Part

    row, out, levels, start, h0_frac, threads, max_cells = job
    part = Part.from_json(json.dumps(row["part"]))
    d = Path(out) / "_meshes" / part.id
    ks = list(range(start, start + levels))
    sides = [d / f"L{k}.json" for k in ks]
    if all(p.exists() and p.with_suffix(".msh").exists() for p in sides):
        return [json.loads(p.read_text()) for p in sides]
    if max_cells:
        h0_frac = fit_h0(row["path"], part, h0_frac, max(ks), max_cells, d, threads=threads)
    metas = [{"part": part.id, "mesh_level": k, **m}
             for k, m in zip(ks, mesh_part(row["path"], part, ks, d, threads=threads, h0_frac=h0_frac))]
    for p, m in zip(sides, metas):
        p.write_text(json.dumps(m))
    return metas


def _mesh_jobs(a, selected):
    return [(row, a.out, a.levels, a.start_level, a.h0_frac, a.threads, a.max_cells) for row, _ in selected]


def cmd_mesh(a) -> int:
    selected = _select(a)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    with ProcessPoolExecutor(a.workers) as ex, open(Path(a.out) / "meshes.jsonl", "a") as log:
        futs = {ex.submit(_mesh_part, j): j[0]["part"]["id"] for j in _mesh_jobs(a, selected)}
        for f in as_completed(futs):
            try:
                metas = f.result()
            except Exception as e:  # noqa: BLE001 - one unmeshable part must not stop the batch
                print(f"[{futs[f]}] mesh FAILED: {type(e).__name__}: {str(e)[:200]}", flush=True)
                continue
            for m in metas:
                log.write(json.dumps(m) + "\n")
            print(f"[{futs[f]}] " + ", ".join(f"L{m['mesh_level']} {m['cells']} cells {m['seconds']:.0f}s"
                                              for m in metas), flush=True)
    print(f"meshed {len(selected)} parts in {time.perf_counter() - t0:.0f}s with {a.workers} workers")
    return 0


# solve-process memory on the H100 (cuDSS): ~7 GB peak RSS for ~1.1M dofs (assembly, factorisation,
# stress recovery and diagnostics in one process), once the rigid-mode SVD stopped building an n x n
# matrix; raised automatically when a solve measures more (CPU SuperLU needs ~70). tet10: ~4.6 dofs/cell
MEM_GB_PER_MDOF = 8.0
MEM_GB_BASE = 1.5
DOFS_PER_CELL = 4.6


def _host_memory_gb() -> float:
    return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9


def _gpu_sampler(path: Path, stop: threading.Event, every: float = 10.0) -> None:
    """GPU utilisation / memory every ``every`` s to gpu.jsonl (is the GPU being fed?)."""
    import subprocess

    with open(path, "a") as f:
        while not stop.wait(every):
            try:
                q = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                                    "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
                util, mem = (float(x) for x in q.stdout.split("\n")[0].split(","))
            except Exception:  # noqa: BLE001 - no GPU / no nvidia-smi: stop sampling
                return
            f.write(json.dumps({"t": time.time(), "gpu_util": util, "gpu_mem_mib": mem}) + "\n")
            f.flush()


def cmd_run(a) -> int:
    from feacore.goldgen.run import solve_job

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    selected = _select(a)
    summaries = open(out / "summary.jsonl", "a")
    meshlog = open(out / "meshes.jsonl", "a")
    todo = {row["part"]["id"]: (row, [s for s in sets if not (out / s.id / "private" / "reference.json").exists()])
            for row, sets in selected}
    todo = {k: v for k, v in todo.items() if v[1]}
    jobs = [j for j in _mesh_jobs(a, selected) if j[0]["part"]["id"] in todo]
    stop = threading.Event()
    threading.Thread(target=_gpu_sampler, args=(out / "gpu.jsonl", stop), daemon=True).start()
    t_start = time.perf_counter()
    # two pools: gmsh meshes parts on CPU cores while solve workers keep the GPU busy; each part's
    # BC sets enter the solve pool the moment its meshes are ready
    if a.gpu_slots:                                 # inherited by the solve workers (forked below)
        os.environ["FEACORE_GPU_SLOTS"] = str(a.gpu_slots)
    limit_gb = float(os.environ.get("GMN_MEMORY_LIMIT_BYTES", 0)) / 1e9 or _host_memory_gb()
    budget = 0.8 * limit_gb - 1.5 * a.mesh_workers               # what the solve workers may use together
    print(f"memory: limit {limit_gb:.0f} GB, solve budget {budget:.0f} GB", flush=True)
    ready: list = []                                              # (mdofs, tries, job) waiting for memory
    running: dict = {}                                            # solve future -> (mdofs, tries, job, est_gb)
    gb_per_mdof = [MEM_GB_PER_MDOF]                               # raised whenever a solve measures more
    solve_ex = ProcessPoolExecutor(max(1, a.solve_workers), max_tasks_per_child=1)

    def est(mdofs: float) -> float:
        return MEM_GB_BASE + gb_per_mdof[0] * mdofs

    def launch():
        used = sum(r[3] for r in running.values())
        ready.sort(key=lambda r: -r[0])                           # biggest first while there is room
        for item in list(ready):
            if len(running) >= a.solve_workers:
                break
            e = est(item[0])
            if used + e <= budget or not running:                 # always admit one, however large
                ready.remove(item)
                running[solve_ex.submit(solve_job, item[2])] = (*item, e)
                used += e

    with ProcessPoolExecutor(max(1, a.mesh_workers)) as mesh_ex:
        mesh_futs = {mesh_ex.submit(_mesh_part, j): j[0]["part"]["id"] for j in jobs}
        pending_mesh = set(mesh_futs)
        while pending_mesh or running or ready:
            launch()
            done, _ = wait(pending_mesh | set(running), return_when=FIRST_COMPLETED)
            for f in done:
                if f in mesh_futs:
                    pending_mesh.discard(f)
                    pid = mesh_futs[f]
                    row, sets = todo[pid]
                    try:
                        meshes = f.result()
                    except Exception as e:  # noqa: BLE001
                        print(f"[{pid}] mesh FAILED: {type(e).__name__}: {str(e)[:200]}", flush=True)
                        continue
                    for m in meshes:
                        meshlog.write(json.dumps({k: m.get(k) for k in (
                            "part", "mesh_level", "h", "h0_frac", "cells", "seconds", "scheme", "nested", "alg3d",
                            "folded_before_fix", "min_sj", "min_linear_quality", "failed_attempts",
                            "fallback_reason")}) + "\n")
                        print(f"[{pid}] L{m['mesh_level']}: {m['cells']} cells, {m['seconds']:.1f}s, {m['scheme']}"
                              f"{'' if m.get('nested', True) else ' remesh'}", flush=True)
                    meshlog.flush()
                    mdofs = DOFS_PER_CELL * max(m["cells"] for m in meshes) / 1e6
                    part_json = json.dumps(row["part"])
                    ready += [(mdofs, 0, (row["path"], part_json, s, meshes, str(out), a.linear_solver)) for s in sets]
                    continue
                if f not in running:                       # a stale future of a pool already rebuilt
                    continue
                mdofs, tries, job, e_gb = running.pop(f)
                try:
                    summ = f.result()
                except BrokenProcessPool:
                    # a worker was killed (out of memory): every in-flight job is lost with the pool.
                    # Rebuild it and retry them; a job that keeps killing workers is given up.
                    lost = [(mdofs, tries, job)] + [r[:3] for r in running.values()]
                    gb_per_mdof[0] *= 1.5                         # it took more memory than we thought
                    running.clear()
                    solve_ex.shutdown(wait=False, cancel_futures=True)
                    solve_ex = ProcessPoolExecutor(max(1, a.solve_workers), max_tasks_per_child=1)
                    for e_, t_, j_ in lost:
                        if t_ + 1 > 2:
                            summ = {"id": j_[2].id, "result": "error", "error": "solve worker killed twice (memory)"}
                            summaries.write(json.dumps(summ) + "\n")
                            print(f"[{j_[2].id}] error: solve worker killed twice (memory)", flush=True)
                        else:
                            ready.append((e_, t_ + 1, j_))
                    print(f"solve pool rebuilt after a worker died; {len(lost)} jobs requeued", flush=True)
                    continue
                rss = summ.get("worker_maxrss_gb") or 0.0       # a worker's lifetime peak: an upper bound
                if mdofs > 0.05 and rss > est(mdofs):
                    gb_per_mdof[0] = 1.1 * (rss - MEM_GB_BASE) / mdofs
                summaries.write(json.dumps(summ) + "\n")
                summaries.flush()
                print(f"[{summ['id']}] {summ.get('result')} gold={summ.get('gold')} singular={summ.get('singular')} "
                      f"dofs={summ.get('dofs')} t={summ['seconds_total']}s rss={rss}GB est={e_gb:.0f}GB "
                      f"(now {gb_per_mdof[0]:.0f} GB/Mdof)", flush=True)
    solve_ex.shutdown()
    stop.set()
    print(f"run done in {time.perf_counter() - t_start:.0f}s", flush=True)
    return 0


def _common(p, parts_default: int) -> None:
    p.add_argument("parts")
    p.add_argument("--out", required=True)
    p.add_argument("--parts", dest="parts_n", type=int, default=parts_default)
    p.add_argument("--bcs", type=int, default=10)
    p.add_argument("--min-bcs", type=int, default=5, help="skip parts with fewer distinct BC sets")
    p.add_argument("--levels", type=int, default=3)
    p.add_argument("--start-level", type=int, default=0)
    p.add_argument("--h0-frac", type=float, default=1 / 8, help="base mesh size / bbox diagonal")
    p.add_argument("--threads", type=int, default=1, help="gmsh threads per part")
    p.add_argument("--max-cells", type=int, default=250_000,
                   help="cap on the finest level's cells (base size enlarged per part); 0 = no cap")
    p.add_argument("--shard", default="0/1", help="i/n: this machine's slice of the selected parts")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m feacore.goldgen")
    sub = ap.add_subparsers(required=True)
    s = sub.add_parser("screen")
    s.add_argument("step_dir")
    s.add_argument("--out", required=True)
    s.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    s.set_defaults(func=cmd_screen)
    m = sub.add_parser("mesh")
    _common(m, parts_default=100)
    m.add_argument("--workers", type=int, default=int(float(os.environ.get("GMN_CPU_LIMIT", os.cpu_count() or 4))))
    m.set_defaults(func=cmd_mesh)
    r = sub.add_parser("run")
    _common(r, parts_default=3)
    r.add_argument("step_dir", help="unused; STEP paths come from the screen file (kept for compatibility)")
    r.add_argument("--mesh-workers", type=int, default=1, help="parallel gmsh processes (CPU)")
    r.add_argument("--solve-workers", type=int, default=1, help="parallel solve processes sharing the GPU")
    r.add_argument("--gpu-slots", type=int, default=2,
                   help="max concurrent GPU factorisations across solve workers (0 = no limit)")
    r.add_argument("--linear-solver", default=None)
    r.set_defaults(func=cmd_run)
    a = ap.parse_args(argv)
    return a.func(a)


if __name__ == "__main__":
    sys.exit(main())
