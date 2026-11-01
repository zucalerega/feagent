"""Reference study of supersonic_blunt_wedge_2d (feacore/problems/supersonic/blunt_wedge.py).

Solves the problem's structured reference mesh at h0 / k for each k in --levels with the fixed solver
(openfoam backend, rhoCentralFoam), every level started from the previous one's solution (mapFields),
and reports per level the QoIs, the stagnation pressure against the Rayleigh pitot pressure and the
bow shock standoff (along the line from the stagnation point against the freestream) against
Billig's cylinder-wedge correlation; then Richardson extrapolation / GCI of every QoI
(feacore.verification.gci.analyze). --alpha 0 is the symmetric validation case.

    python scripts/supersonic_blunt_study.py --alpha 10 --levels 0.5,1,2,4 --average 30 --out /tmp/sbw/study_a10
    python scripts/supersonic_blunt_study.py --alpha 0 --levels 0.5,1,2,4 --out /tmp/sbw/study_a0

At 10 deg a small limit cycle of the scheme (behind the sonic line) keeps the windward flank pressure
moving by ~0.1 % peak to peak: --average N continues each level N flow times and takes the QoIs' time
mean, so the extrapolation sees the mean flow (the band is reported). --resume reuses existing case
directories (e.g. to average a level solved earlier).

Writes <out>/study.json and prints a markdown table (docs/benchmarks_cfd.md, "Supersonic blunted wedge").
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np

from feacore.problems.supersonic.blunt_wedge import SupersonicBluntWedge2D, billig_standoff, standoff
from feacore.solver import _foam_compressible as fc
from feacore.solver import _foam_io as fio
from feacore.solver import solve
from feacore.solver._foam_env import require_openfoam
from feacore.solver._foam_fv import FvMesh
from feacore.solver._foam_mesh import convert
from feacore.verification.gci import analyze


def _snapshot(case, p: SupersonicBluntWedge2D, cdir: Path, fm, poly, mesh, tdir: Path):
    g = p.params
    r = fc.CompressibleResult(case, fm, cdir, g["gamma"], g["gas_constant"], 0.0, g["p_inf"], [])
    r._load(poly, tdir, mesh)
    return r


def average(case, p: SupersonicBluntWedge2D, cdir: Path, n_ft: int, work: Path) -> dict:
    """Continue the kept case ``cdir`` by ``n_ft`` flow times, one at a time, and evaluate the QoIs after
    each: their time mean (the reference over a limit cycle) and band (min, max). Also returns the
    last snapshot result (for the standoff)."""
    fe = require_openfoam()
    fm = convert(case.mesh_path(), 2, 1.0, work / "_avg.msh")
    poly = fio.read_poly(cdir)
    nb = fio.read_list(cdir / "constant" / "polyMesh" / "neighbour").astype(np.int64)
    mesh = FvMesh(poly["points"], poly["faces"], poly["owner"], nb, poly["patches"])
    e = np.array(p.u_inf + [0.0]) / np.linalg.norm(p.u_inf)
    speed = float(np.linalg.norm(p.u_inf))
    t_ref = float(np.ptp(mesh.C @ e)) / speed                 # as the backend defines one flow time
    opts = {**fc.DEFAULTS, **(case.analysis.options or {})}
    samples, res = [], None
    for _ in range(n_ft):
        t0 = float(fc._times(cdir)[-1].name)
        fc._write_control(cdir, t0 + t_ref, 1e-9, t_ref, float(opts["max_co"]))
        code, _, tail, reason = fc._run_solver(fe, cdir, None)
        if code != 0 or reason == "diverged":
            raise RuntimeError("rhoCentralFoam failed while averaging:\n" + "".join(tail[-20:]))
        times = fc._times(cdir)
        for old in times[:-2]:
            shutil.rmtree(old)
        res = _snapshot(case, p, cdir, fm, poly, mesh, times[-1])
        samples.append(res.evaluate_all())
    names = list(samples[0])
    return {"flow_times": n_ft, "mean": {n: float(np.mean([s[n] for s in samples])) for n in names},
            "min": {n: float(min(s[n] for s in samples)) for n in names},
            "max": {n: float(max(s[n] for s in samples)) for n in names}, "samples": samples, "result": res}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--alpha", type=float, default=10.0)
    ap.add_argument("--mach", type=float, default=3.0)
    ap.add_argument("--levels", default="0.5,1,2,4", help="refinement factors k (h = h0 / k), coarse to fine")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--chunk", type=float, default=5.0, help="flow times between steadiness checks")
    ap.add_argument("--no-map", action="store_true", help="start every level from the freestream")
    ap.add_argument("--average", type=int, default=0,
                    help="after each solve, continue this many flow times and take the QoIs' time mean "
                         "(over a limit cycle) as the level's value")
    ap.add_argument("--max-flow-times", type=float, default=None, help="cap of each solve (before averaging)")
    ap.add_argument("--resume", action="store_true", help="reuse an existing case_<k> dir instead of solving")
    args = ap.parse_args(argv)

    p = SupersonicBluntWedge2D(id=f"supersonic_blunt_wedge_2d_a{args.alpha:g}", alpha=args.alpha, mach=args.mach)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    h0 = p.params["h0"]
    rows, prev = [], None
    for k in [float(x) for x in args.levels.split(",")]:
        h = h0 / k
        tag = f"k{k:g}"
        mesh = p.mesh(out / f"{tag}.msh", h)
        cdir = out / f"case_{tag}"
        case = p.case(mesh)
        t = time.time()
        if args.resume and cdir.exists() and fc._times(cdir):
            info = None                                       # solved by an earlier run of this script
            print(f"k={k:g}: resuming {cdir} at t = {fc._times(cdir)[-1].name}", flush=True)
        else:
            if cdir.exists():
                raise SystemExit(f"{cdir} exists: use a fresh --out (or --resume)")
            extra = {"init_from": str(prev)} if prev is not None and not args.no_map else {}
            if args.max_flow_times is not None:
                extra["max_flow_times"] = args.max_flow_times
            res = solve(case, backend="openfoam", case_dir=str(cdir), chunk=args.chunk, **extra)
            info = res.info
        band = None
        if args.average > 0:
            avg = average(case, p, cdir, args.average, out)
            q, res = avg["mean"], avg["result"]
            band = {"flow_times": args.average, "min": avg["min"], "max": avg["max"]}
        else:
            q = res.evaluate_all()
        wall = time.time() - t
        row = {"k": k, "h": h, "cells": res.fm.n_cells, "time_steps": info.time_steps if info else None,
               "flow_times": info.flow_times if info else None, "steady": info.steady if info else "resumed",
               "converged": info.converged if info else None, "wall_s": round(wall, 1), "qois": q, "band": band,
               "p_stag_over_pitot": q["p_stag"] / p.pitot(), **standoff(res, p),
               "warnings": [w for w in (info.warnings if info else []) if not w.startswith("info: OpenFOAM")]}
        rows.append(row)
        prev = cdir
        spread = ("; band " + ", ".join(f"{n} {(band['max'][n] - band['min'][n]) / max(abs(q[n]), 1e-30):.1e}"
                                        for n in q) if band else "")
        print(f"k={k:g}: {row['cells']} cells, flow times {row['flow_times']}, steady={row['steady'] or 'NO'}, "
              f"{wall:.0f} s; " + ", ".join(f"{n}={v:.7g}" for n, v in q.items())
              + f"; p_stag/pitot={row['p_stag_over_pitot']:.5f}; standoff/R={row.get('standoff_over_R', float('nan')):.4f}"
              + spread, flush=True)
        (out / "study.json").write_text(json.dumps({"alpha": args.alpha, "mach": args.mach, "rows": rows}, indent=1))

    hs = [r["h"] for r in rows]
    gci = {}
    for name in rows[0]["qois"]:
        phi = [r["qois"][name] for r in rows]
        try:
            gci[name] = analyze(hs, phi)
        except Exception as e:  # noqa: BLE001
            gci[name] = {"error": str(e)}
    sd = [r.get("standoff_over_R", float("nan")) for r in rows]
    try:
        gci["standoff_over_R"] = analyze(hs, sd)
    except Exception as e:  # noqa: BLE001
        gci["standoff_over_R"] = {"error": str(e)}
    summary = {"alpha": args.alpha, "mach": args.mach, "rows": rows, "gci": gci, "pitot": p.pitot(),
               "billig_standoff_over_R": billig_standoff(args.mach)}
    (out / "study.json").write_text(json.dumps(summary, indent=1, default=str))

    print(f"\npitot p02 = {p.pitot():.6g} Pa; Billig standoff / R = {billig_standoff(args.mach):.4f}")
    names = list(rows[0]["qois"]) + ["standoff_over_R"]
    print("| level | cells | flow times | " + " | ".join(names) + " |")
    print("|---" * (len(names) + 3) + "|")
    for r in rows:
        vals = [r["qois"][n] for n in names[:-1]] + [r.get("standoff_over_R", float("nan"))]
        print(f"| h0 / {r['k']:g} | {r['cells']} | {r['flow_times']:.0f} | " + " | ".join(f"{v:.6g}" for v in vals) + " |")
    ext = []
    for n in names:
        g = gci[n].get("gci", {}) if isinstance(gci[n], dict) else {}
        num = {key: (float("nan") if g.get(key) is None else float(g[key])) for key in ("phi_ext", "p", "gci_fine")}
        ext.append(f"{num['phi_ext']:.6g} (p {num['p']:.2f}, GCI {100 * num['gci_fine']:.2f} %)")
    print("| extrapolated | | | " + " | ".join(ext) + " |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
