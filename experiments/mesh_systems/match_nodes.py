#!/usr/bin/env python
"""Build one system's nonlinear mesh of a part at a target node count by tuning only that system's own size knob.

    python experiments/mesh_systems/match_nodes.py SYSTEM OUT_ROOT [--part abc_00000065] [--target 40000]
        [--tol 0.02] [--max-tries 8]

Every system keeps its own refinement logic (its build.py, unchanged defaults); only the knob below moves:

| system | knob (CLI of its build.py) | default |
|---|---|---|
| harness_auto | `--global` (global size) | 6 mm |
| harness_expert | `--global` | 12 mm |
| ace_sizing | `--h` (global h; `--node-cap` raised to the band's top) | 3.5 mm |
| gold_nested | `--h0-frac` (base size / diagonal; the selected level L1 is the base split once) | fitted to 40k |
| netgen | `--maxh` | 3.0 mm |
| harness_hex | `--thin-factor` (tet-stage thin-wall size / wall thickness, before the hex split; its base size h is still chosen by its own candidate search for a valid hex mesh) | 1.4 |
| harness_neck | `--neck-size` (element size at the ring wall; the rest is harness_expert) | 1.6 mm |

Search: secant steps on log(nodes) vs log(knob), starting from a slope of -3 (smaller knob = more nodes for
every knob above), until the node count (as the solver builds the model) is within target x (1 +- tol). Every
try is a separate OUT_ROOT/<system>/p65_nonlinear_n40k_tries/try_NN dir; the accepted one is copied to
OUT_ROOT/<system>/p65_nonlinear_n40k (never overwritten). The tries
are logged in match.json there. Stops after --max-tries; then the closest try is reported (not copied).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
KNOBS = {"harness_auto": ("--global", 0.006, []), "harness_expert": ("--global", 0.012, []),
         "ace_sizing": ("--h", 0.0035, None), "gold_nested": ("--h0-frac", None, []),
         "netgen": ("--maxh", 0.0030, []), "harness_hex": ("--thin-factor", 1.4, []),
         "harness_neck": ("--neck-size", 0.0016, [])}
PARTS = {"abc_00000065": "y", "abc_00000003": "z"}


def count_nodes(d: Path) -> tuple[int, int]:
    sys.path.insert(0, str(ROOT))
    from feacore.solver._ccx_mesh import build_model, read_gmsh
    gm = read_gmsh(d / "mesh.msh")
    order = 2 if any(ct == "tetra10" for ct, _ in gm.groups["domain"]) else 1
    m = build_model(gm, 3, order)
    return m.n_nodes, m.n_elems


def gold_frac(d: Path) -> float | None:
    """Base size fraction a gold_nested build used (its report / log)."""
    for f in (d / "report.json", d / "levels.json"):
        if f.exists():
            t = json.loads(f.read_text())
            for k in ("h0_frac", "base_frac", "frac"):
                v = t.get(k) or (t.get("knobs") or {}).get(k)
                if v:
                    return float(v)
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("system", choices=sorted(KNOBS)), ap.add_argument("out_root")
    ap.add_argument("--part", default="abc_00000065", choices=sorted(PARTS))
    ap.add_argument("--target", type=int, default=40_000)
    ap.add_argument("--tol", type=float, default=0.02)
    ap.add_argument("--max-tries", type=int, default=8)
    ap.add_argument("--start", type=float, default=None, help="first knob value (default: the system's default)")
    a = ap.parse_args()
    flag, x0, _ = KNOBS[a.system]
    step = ROOT / "benchmarks" / "gold" / f"{a.part}_bc0" / "public" / "geometry.step"
    tag = f"p{int(a.part.split('_')[1])}_nonlinear_n{a.target // 1000}k"
    root = Path(a.out_root) / a.system / f"{tag}_tries"
    fin = Path(a.out_root) / a.system / tag
    if fin.exists():
        raise SystemExit(f"{fin} exists: never overwrite a matched mesh")
    root.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "LIBGL_ALWAYS_SOFTWARE": "1", "GALLIUM_DRIVER": "llvmpipe",
           "__GLX_VENDOR_LIBRARY_NAME": "mesa", "OMP_NUM_THREADS": "4"}
    if a.system == "netgen":
        env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(Path(a.out_root) / "netgen" / "site"),
                                                          env.get("PYTHONPATH", "")]))
    lo, hi = a.target * (1 - a.tol), a.target * (1 + a.tol)
    x = a.start or x0
    tries, slope = [], -3.0
    for i in range(a.max_tries):
        d = root / f"try_{i:02d}"
        if d.exists():
            shutil.rmtree(d)                  # this script's own scratch try dirs
        cmd = [sys.executable, str(HERE / a.system / "build.py"), str(step), PARTS[a.part], str(d),
               "--purpose", "nonlinear"]
        if x is not None:
            cmd += [flag, f"{x:.6g}"]
        if a.system == "ace_sizing":
            cmd += ["--node-cap", str(int(hi))]
        t0 = time.time()
        p = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=ROOT)
        ok = (d / "mesh.msh").exists()
        n, ne = count_nodes(d) if ok else (None, None)
        if x is None:
            x = gold_frac(d)
        rec = {"try": i, "knob": flag, "value": x, "nodes": n, "elements": ne, "rc": p.returncode,
               "build_s": round(time.time() - t0, 1), "dir": str(d), "stderr_tail": p.stderr[-400:] if not ok else ""}
        tries.append(rec)
        print(json.dumps(rec), flush=True)
        (root / "match.json").write_text(json.dumps({"system": a.system, "target": a.target, "tol": a.tol,
                                                     "tries": tries}, indent=1))
        if n is None:
            return 2
        if lo <= n <= hi:
            shutil.copytree(d, fin)
            print(f"{a.system}: {n} nodes with {flag} {x:.6g} -> {fin}", flush=True)
            return 0
        good = [t for t in tries if t["nodes"] and t["value"]]
        if len(good) >= 2:
            (x1, n1), (x2, n2) = [(t["value"], t["nodes"]) for t in good[-2:]]
            if x1 != x2 and n1 != n2:
                slope = max(-6.0, min(-0.5, (math.log(n2) - math.log(n1)) / (math.log(x2) - math.log(x1))))
        if x is None:
            return 3
        x = x * math.exp((math.log(a.target) - math.log(n)) / slope)
    best = min((t for t in tries if t["nodes"]), key=lambda t: abs(t["nodes"] - a.target))
    print(f"{a.system}: no try within the band; closest {best['nodes']} at {flag} {best['value']}", flush=True)
    return 1


if __name__ == "__main__":
    sys.exit(main())
