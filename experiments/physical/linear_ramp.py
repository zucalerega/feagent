"""Linear-elastic platen compression of one contract mesh (feacore/skfem, CPU, PARDISO), solved once at F_REF.

    python experiments/physical/linear_ramp.py --system ID --part abc_00000003 --mesh-dir DIR --out DIR
        [--force 100] [--threads 4] [--material experiments/material/alpha_fast_dogbone1.json]

DIR (experiments/mesh_systems/CONTRACT.md) holds mesh.msh with groups bottom / top (dim 2) and domain (dim 3), metres.
Bottom fixed; total force F_REF along -axis on top, spread per area. Material: the dogbone JSON (E [MPa] x 1e6, nu).
The problem is linear, so every field scales with F / F_REF: the ramp video and F_full are built from this one solve.

Writes OUT/result.vtu (point fields displacement [m], von_mises [Pa], ...) and OUT/summary.json:
reaction, strain energy U, compliance 2U/F (work-equivalent platen travel at F_REF; the comparison metric),
top-face mean displacement along the axis and mean |u|, peak von Mises (singular at the platen edges, mesh dependent)
and the 99.9th-percentile volume-weighted von Mises (robust), DOFs, solve time."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

AX = {"x": 0, "y": 1, "z": 2}
PARTS = {"abc_00000003": "z", "abc_00000065": "y"}
ROBUST_Q = 99.9


def _tet_volumes(p, c):
    a, b, cc, d = (p[c[:, i]] for i in range(4))
    return np.abs(np.einsum("ij,ij->i", b - a, np.cross(cc - a, d - a))) / 6.0


def volume_weighted_vm(vtu: Path, q: float = ROBUST_Q) -> dict:
    """Cell mean von Mises (mean of the cell's nodal values) weighted by the straight-sided cell volume."""
    import meshio
    m = meshio.read(vtu)
    vm = np.asarray(m.point_data["von_mises"], float)
    vals, w = [], []
    for blk in m.cells:
        if blk.type.startswith("tetra"):
            c = blk.data
            vals.append(vm[c].mean(axis=1)); w.append(_tet_volumes(m.points, c[:, :4]))
    v, w = np.concatenate(vals), np.concatenate(w)
    o = np.argsort(v); cw = np.cumsum(w[o]); cw /= cw[-1]
    pct = lambda x: float(v[o][min(np.searchsorted(cw, x / 100.0), len(v) - 1)])  # noqa: E731
    return {f"p{q:g}": pct(q), "p99": pct(99.0), "p95": pct(95.0), "volume_m3": float(w.sum()),
            "cell_mean_max": float(v.max())}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--system", required=True)
    ap.add_argument("--part", required=True, choices=sorted(PARTS))
    ap.add_argument("--mesh-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--force", type=float, default=100.0)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--material", default=str(REPO / "experiments" / "material" / "alpha_fast_dogbone1.json"))
    a = ap.parse_args(argv)
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[k] = str(a.threads)
    from feacore.case import BC, Analysis, Case, Material, QoI
    from feacore.solver import solve
    import meshsys

    md, out = Path(a.mesh_dir).resolve(), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ax = PARTS[a.part]; k = AX[ax]
    mat = json.loads(Path(a.material).read_text())
    E, nu = float(mat["E"]) * 1e6, float(mat["nu"])
    f = [0.0, 0.0, 0.0]; f[k] = -a.force
    case = Case(mesh=str(md / "mesh.msh"), analysis=Analysis(physics="elasticity", formulation="3d", order=2),
                materials={"resin": Material(E=E, nu=nu)},
                bcs=[BC("fixed", group="bottom"), BC("force", group="top", value=f)],
                qois=[QoI("strain_energy", "strain_energy"), QoI("max_von_mises", "max", field="von_mises"),
                      QoI("reaction", "reaction", group="bottom", component=k),
                      QoI("mean_u_mag_top", "mean", field="u_mag", group="top"),
                      QoI("mean_u_axis_top", "mean", field=f"u{ax}", group="top"),
                      QoI("max_u_mag", "max", field="u_mag")])
    case.save(out / "case.json")
    t = time.time()
    res = solve(case, "skfem", linear_solver="pardiso")
    solve_s = time.time() - t
    q = res.evaluate_all()
    res.write_vtu(out / "result.vtu")
    rob = volume_weighted_vm(out / "result.vtu")
    rep = meshsys.report(md) if (md / "report.json").exists() else {}
    U = float(q["strain_energy"])
    s = {"system": a.system, "part": a.part, "axis": ax, "mesh_dir": str(md), "force_N": a.force,
         "material": mat.get("name"), "E_Pa": E, "nu": nu, "threads": a.threads,
         "n_dofs": res.info.n_dofs, "n_nodes": res.info.n_nodes, "n_cells": res.info.n_cells,
         "cell_type": res.info.cell_type, "order": res.info.order, "solve_s": round(solve_s, 2),
         "solver_wall_s": round(res.info.wall_time, 2), "warnings": res.info.warnings,
         "reaction_N": float(q["reaction"]), "strain_energy_J": U,
         "compliance_2U_over_F_m": 2.0 * U / a.force, "compliance_m_per_N": 2.0 * U / a.force ** 2,
         "stiffness_N_per_m": a.force ** 2 / (2.0 * U),
         "mean_u_axis_top_m": float(q["mean_u_axis_top"]), "mean_u_mag_top_m": float(q["mean_u_mag_top"]),
         "max_u_mag_m": float(q["max_u_mag"]), "peak_von_mises_Pa": float(q["max_von_mises"]),
         "robust_von_mises_Pa": rob[f"p{ROBUST_Q:g}"], "robust_quantile": ROBUST_Q, "von_mises_volume_weighted": rob,
         "mesh_report": rep, "qois": {kk: float(v) for kk, v in q.items()}}
    (out / "summary.json").write_text(json.dumps(s, indent=1))
    print(json.dumps({kk: s[kk] for kk in ("system", "part", "n_dofs", "reaction_N", "compliance_2U_over_F_m",
                                            "mean_u_axis_top_m", "peak_von_mises_Pa", "robust_von_mises_Pa", "solve_s")}))


if __name__ == "__main__":
    main()
