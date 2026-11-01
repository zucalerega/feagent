#!/usr/bin/env python
"""Sanity solve of a LINEAR (tet10) mesh with feacore/skfem on the CPU.

    sanity_solve.py OUT_DIR AXIS [--material experiments/material/alpha_fast_dogbone1.json]

Fixed `bottom`, 100 N compressive force on `top` along -AXIS; E [MPa] x 1e6 -> Pa. Prints/writes JSON.
"""
import argparse, json, os, sys
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
from pathlib import Path
from feacore.case import BC, Analysis, Case, Material, QoI
from feacore.solver import solve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir"); ap.add_argument("axis", choices="xyz")
    ap.add_argument("--material", default="experiments/material/alpha_fast_dogbone1.json")
    ap.add_argument("--force", type=float, default=100.0)
    a = ap.parse_args()
    d = Path(a.out_dir)
    mat = json.loads(Path(a.material).read_text())
    ax = "xyz".index(a.axis)
    f = [0.0, 0.0, 0.0]; f[ax] = -a.force
    case = Case(
        mesh=str((d / "mesh.msh").resolve()),
        analysis=Analysis(physics="elasticity", formulation="3d", order=2),
        materials={"material": Material(E=mat["E"] * 1e6, nu=mat["nu"])},
        bcs=[BC("fixed", group="bottom"), BC("force", group="top", value=f)],
        qois=[QoI("strain_energy", "strain_energy"), QoI("max_von_mises", "max", field="von_mises"),
              QoI("reaction", "reaction", group="bottom", component=ax),
              QoI("mean_u_top", "mean", field="u_mag", group="top"),
              QoI("mean_uax_top", "mean", field=f"u{a.axis}", group="top")])
    res = solve(case, "skfem", linear_solver="pardiso")
    q = res.evaluate_all()
    out = {"reaction_N": q["reaction"], "top_mean_u_m": q["mean_u_top"], "top_mean_u_axis_m": q["mean_uax_top"], "max_von_mises_Pa": q["max_von_mises"],
           "strain_energy_J": q["strain_energy"], "n_dofs": res.info.n_dofs, "cell_type": res.info.cell_type,
           "wall_s": round(res.info.wall_time, 2),
           "stiffness_N_per_m": a.force / abs(q["mean_uax_top"])}
    (d / "sanity_solve.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out))

main()
