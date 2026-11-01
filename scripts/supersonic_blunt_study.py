"""Reference study of supersonic_blunt_wedge_2d (feacore/problems/supersonic/blunt_wedge.py).

Solves the problem's structured reference mesh at h0 / k for each k in --levels with the fixed solver
(openfoam backend, rhoCentralFoam), every level started from the previous one's solution (mapFields),
and reports per level the QoIs, the stagnation pressure against the Rayleigh pitot pressure and the
bow shock standoff (along the line from the stagnation point against the freestream) against
Billig's cylinder-wedge correlation; then Richardson extrapolation / GCI of every QoI
(feacore.verification.gci.analyze). --alpha 0 is the symmetric validation case.

    python scripts/supersonic_blunt_study.py --alpha 10 --levels 0.5,1,2,4 --out /tmp/sbw/study_a10
    python scripts/supersonic_blunt_study.py --alpha 0 --levels 0.5,1,2,4 --out /tmp/sbw/study_a0

Writes <out>/study.json and prints a markdown table (docs/benchmarks_cfd.md, "Supersonic blunted wedge").
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


from feacore.problems.supersonic.blunt_wedge import SupersonicBluntWedge2D, billig_standoff, standoff
from feacore.solver import solve
from feacore.verification.gci import analyze


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--alpha", type=float, default=10.0)
    ap.add_argument("--mach", type=float, default=3.0)
    ap.add_argument("--levels", default="0.5,1,2,4", help="refinement factors k (h = h0 / k), coarse to fine")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--chunk", type=float, default=5.0, help="flow times between steadiness checks")
    ap.add_argument("--no-map", action="store_true", help="start every level from the freestream")
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
        if cdir.exists():
            raise SystemExit(f"{cdir} exists: use a fresh --out")
        t = time.time()
        res = solve(p.case(mesh), backend="openfoam", case_dir=str(cdir), chunk=args.chunk,
                    **({"init_from": str(prev)} if prev is not None and not args.no_map else {}))
        wall = time.time() - t
        q = res.evaluate_all()
        info = res.info
        row = {"k": k, "h": h, "cells": info.n_cells, "time_steps": info.time_steps, "flow_times": info.flow_times,
               "steady": info.steady, "converged": info.converged, "wall_s": round(wall, 1), "qois": q,
               "p_stag_over_pitot": q["p_stag"] / p.pitot(), **standoff(res, p),
               "warnings": [w for w in info.warnings if not w.startswith("info: OpenFOAM")]}
        rows.append(row)
        prev = cdir
        print(f"k={k:g}: {info.n_cells} cells, {info.time_steps} steps, {info.flow_times:.0f} flow times, "
              f"steady={info.steady or 'NO'}, {wall:.0f} s; " + ", ".join(f"{n}={v:.6g}" for n, v in q.items())
              + f"; p_stag/pitot={row['p_stag_over_pitot']:.5f}; standoff/R={row.get('standoff_over_R', float('nan')):.4f}",
              flush=True)
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
