"""Gold-benchmark generator CLI.

  python -m feacore.goldgen screen STEP_DIR --out parts.jsonl [--workers 8]
  python -m feacore.goldgen run parts.jsonl STEP_DIR --out gold/ [--parts 3] [--bcs 10]
         [--levels 4] [--start-level 0] [--linear-solver auto] [--threads 0]

``screen`` accepts / rejects every STEP file (one process per file: gmsh is process-global).
``run`` meshes each accepted part once per level, then solves and certifies each BC set
(see feacore.goldgen.run). Re-running skips BC sets whose private/reference.json exists.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
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


def cmd_run(a) -> int:
    from feacore.goldgen.bcgen import generate
    from feacore.goldgen.mesh import mesh_level
    from feacore.goldgen.parts import Part
    from feacore.goldgen.run import run_bcset

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = [r for r in map(json.loads, open(a.parts)) if r["ok"]]
    log = open(out / "solves.jsonl", "a")
    summaries = open(out / "summary.jsonl", "a")
    done_parts = 0
    for row in rows:
        if done_parts >= a.parts_n:
            break
        part = Part.from_json(json.dumps(row["part"]))
        step = row["path"]
        sets = generate(part, a.bcs)
        if len(sets) < a.min_bcs:
            print(f"[{part.id}] skipped: only {len(sets)} distinct BC sets (< {a.min_bcs})", flush=True)
            continue
        todo = [s for s in sets if not (out / s.id / "private" / "reference.json").exists()]
        done_parts += 1
        if not todo:
            continue
        meshes = []
        for k in range(a.start_level, a.start_level + a.levels):
            m = mesh_level(step, part, k, out / "_meshes" / part.id / f"L{k}.msh", threads=a.threads, h0_frac=a.h0_frac)
            meshes.append(m)
            log.write(json.dumps({"part": part.id, "mesh_level": k, **{x: m[x] for x in ("h", "cells", "seconds")}}) + "\n")
            log.flush()
            print(f"[{part.id}] L{k}: {m['cells']} cells, {m['seconds']:.1f}s", flush=True)
        for s in todo:
            t0 = time.perf_counter()
            try:
                summ = run_bcset(step, part, s, meshes, out, a.linear_solver, log)
            except Exception as e:  # noqa: BLE001 - one bad BC set must not stop the part
                summ = {"id": s.id, "result": "error", "error": f"{type(e).__name__}: {str(e)[:300]}"}
            summ["seconds_total"] = round(time.perf_counter() - t0, 1)
            summaries.write(json.dumps(summ) + "\n")
            summaries.flush()
            print(f"[{s.id}] {summ.get('result')} gold={summ.get('gold')} singular={summ.get('singular')} "
                  f"dofs={summ.get('dofs')} t={summ['seconds_total']}s", flush=True)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m feacore.goldgen")
    sub = ap.add_subparsers(required=True)
    s = sub.add_parser("screen")
    s.add_argument("step_dir")
    s.add_argument("--out", required=True)
    s.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    s.set_defaults(func=cmd_screen)
    r = sub.add_parser("run")
    r.add_argument("parts")
    r.add_argument("step_dir")
    r.add_argument("--out", required=True)
    r.add_argument("--parts", dest="parts_n", type=int, default=3)
    r.add_argument("--bcs", type=int, default=10)
    r.add_argument("--min-bcs", type=int, default=5, help="skip parts with fewer distinct BC sets")
    r.add_argument("--levels", type=int, default=3)
    r.add_argument("--h0-frac", type=float, default=1 / 8, help="base mesh size / bbox diagonal")
    r.add_argument("--start-level", type=int, default=0)
    r.add_argument("--linear-solver", default=None)
    r.add_argument("--threads", type=int, default=0)
    r.set_defaults(func=cmd_run)
    a = ap.parse_args(argv)
    return a.func(a)


if __name__ == "__main__":
    sys.exit(main())
