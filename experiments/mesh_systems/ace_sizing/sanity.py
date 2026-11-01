"""Linear sanity solve (feacore/skfem, CPU, PARDISO): fixed bottom, 100 N compression on top along the axis."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from feacore.case import BC, Analysis, Case, Material, QoI
from feacore.solver import solve

MAT = Path(__file__).resolve().parents[2] / "material" / "alpha_fast_dogbone1.json"


def sanity(mesh, axis, order=2, force=100.0):
    m = json.loads(MAT.read_text())
    f = [0.0, 0.0, 0.0]; f["xyz".index(axis)] = -force
    case = Case(mesh=str(mesh), analysis=Analysis(physics="elasticity", formulation="3d", order=order),
                materials={"material": Material(E=m["E"] * 1e6, nu=m["nu"])},
                bcs=[BC("fixed", group="bottom"), BC("force", group="top", value=f)],
                qois=[QoI("reaction_z", "reaction", group="bottom", component="xyz".index(axis)),
                      QoI("mean_u_top", "mean", field="u_mag", group="top"),
                      QoI("max_von_mises", "max", field="von_mises"),
                      QoI("strain_energy", "strain_energy")])
    res = solve(case, backend="skfem", linear_solver="pardiso")
    q = res.evaluate_all()
    out = {"reaction_N": q["reaction_z"], "top_mean_u_m": q["mean_u_top"], "peak_vm_Pa": q["max_von_mises"],
           "strain_energy_J": q["strain_energy"], "dofs": int(res.info.n_dofs), "solve_s": float(res.info.wall_time)}
    return out, res


def hotspots(res, k=2, min_sep=0.0):
    pts, vm = res.nodal("von_mises")
    vm = np.asarray(vm).ravel(); pts = np.asarray(pts)
    out = []
    for i in np.argsort(-vm):
        if all(np.linalg.norm(pts[i] - p) > min_sep for p, _ in out):
            out.append((pts[i], float(vm[i])))
        if len(out) == k: break
    return [([float(c) for c in p], v) for p, v in out]
