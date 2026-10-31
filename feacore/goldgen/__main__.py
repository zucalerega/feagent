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
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
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


def cmd_screen(a) -> int:
    files = sorted(str(p) for p in Path(a.step_dir).rglob("*.step"))
    n_ok = 0
    with ProcessPoolExecutor(a.workers) as ex, open(a.out, "w") as f:
        for row in ex.map(_screen_one, files):
            f.write(json.dumps(row) + "\n")
            n_ok += row["ok"]
    print(f"screened {len(files)}: {n_ok} accepted -> {a.out}")
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
    from feacore.goldgen.mesh import mesh_part
    from feacore.goldgen.parts import Part

    row, out, levels, start, h0_frac, threads = job
    part = Part.from_json(json.dumps(row["part"]))
    d = Path(out) / "_meshes" / part.id
    ks = list(range(start, start + levels))
    sides = [d / f"L{k}.json" for k in ks]
    if all(p.exists() and p.with_suffix(".msh").exists() for p in sides):
        return [json.loads(p.read_text()) for p in sides]
    metas = [{"part": part.id, "mesh_level": k, **m}
             for k, m in zip(ks, mesh_part(row["path"], part, ks, d, threads=threads, h0_frac=h0_frac))]
    for p, m in zip(sides, metas):
        p.write_text(json.dumps(m))
    return metas


def _mesh_jobs(a, selected):
    return [(row, a.out, a.levels, a.start_level, a.h0_frac, a.threads) for row, _ in selected]


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


def cmd_run(a) -> int:
    from feacore.goldgen.parts import Part
    from feacore.goldgen.run import run_bcset

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    selected = _select(a)
    log = open(out / "solves.jsonl", "a")
    summaries = open(out / "summary.jsonl", "a")
    todo = {row["part"]["id"]: (row, [s for s in sets if not (out / s.id / "private" / "reference.json").exists()])
            for row, sets in selected}
    todo = {k: v for k, v in todo.items() if v[1]}
    jobs = [j for j in _mesh_jobs(a, selected) if j[0]["part"]["id"] in todo]
    with ProcessPoolExecutor(max(1, a.mesh_workers)) as ex:
        futs = {ex.submit(_mesh_part, j): j[0]["part"]["id"] for j in jobs}
        for f in as_completed(futs):                      # solve each part as soon as it is meshed
            pid = futs[f]
            row, sets = todo[pid]
            try:
                meshes = f.result()
            except Exception as e:  # noqa: BLE001
                print(f"[{pid}] mesh FAILED: {type(e).__name__}: {str(e)[:200]}", flush=True)
                continue
            for m in meshes:
                log.write(json.dumps({k: m.get(k) for k in ("part", "mesh_level", "h", "cells", "seconds", "scheme",
                                                             "folded_before_fix", "failed_attempts", "fallback_reason")}) + "\n")
                print(f"[{pid}] L{m['mesh_level']}: {m['cells']} cells, {m['seconds']:.1f}s, {m['scheme']}"
                      f" (folded before fix: {m['folded_before_fix']})", flush=True)
            log.flush()
            part = Part.from_json(json.dumps(row["part"]))
            for s in sets:
                t0 = time.perf_counter()
                try:
                    summ = run_bcset(row["path"], part, s, meshes, out, a.linear_solver, log)
                except Exception as e:  # noqa: BLE001 - one bad BC set must not stop the part
                    summ = {"id": s.id, "result": "error", "error": f"{type(e).__name__}: {str(e)[:300]}"}
                summ["seconds_total"] = round(time.perf_counter() - t0, 1)
                summaries.write(json.dumps(summ) + "\n")
                summaries.flush()
                print(f"[{s.id}] {summ.get('result')} gold={summ.get('gold')} singular={summ.get('singular')} "
                      f"dofs={summ.get('dofs')} t={summ['seconds_total']}s", flush=True)
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
    r.add_argument("--mesh-workers", type=int, default=1)
    r.add_argument("--linear-solver", default=None)
    r.set_defaults(func=cmd_run)
    a = ap.parse_args(argv)
    return a.func(a)


if __name__ == "__main__":
    sys.exit(main())
