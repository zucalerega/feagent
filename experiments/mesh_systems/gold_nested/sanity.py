#!/usr/bin/env python
"""Linear sanity solve of a gold_nested mesh (feacore/skfem, CPU, PARDISO), or the CalculiX smoke check.

    sanity.py MESH.msh AXIS [--solve] [--ccx MESH.inp]

Fixed `bottom`, 100 N compressive total force on `top` along the axis (toward the bottom), Alpha Fast E/nu from
experiments/material/alpha_fast_dogbone1.json (MPa -> Pa). Prints one JSON line.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
import time
from pathlib import Path

MAT = Path(__file__).resolve().parents[2] / "material" / "alpha_fast_dogbone1.json"
CCX = Path.home() / "micromamba-local/envs/ccx/bin/ccx"


def solve(msh: str, axis: str, force: float = 100.0) -> dict:
    from feacore.case import BC, Analysis, Case, Material, QoI
    from feacore.solver import solve as fe_solve

    m = json.loads(MAT.read_text())
    f = [0.0, 0.0, 0.0]
    f["xyz".index(axis)] = -force
    case = Case(mesh=str(Path(msh).resolve()), analysis=Analysis(physics="elasticity", formulation="3d", order=2),
                materials={"material": Material(E=m["E"] * 1e6, nu=m["nu"])},
                bcs=[BC("fixed", group="bottom"), BC("force", group="top", value=f)],
                qois=[QoI("strain_energy", "strain_energy"), QoI("max_von_mises", "max", field="von_mises"),
                      QoI("mean_u_top", "mean", field="u_mag", group="top")]
                + [QoI(f"reaction_bottom_{a}", "reaction", group="bottom", component=i) for i, a in enumerate("xyz")],
                meta={"source": "gold_nested sanity"})
    case.analysis.options = {"linear_solver": "pardiso"}
    t = time.perf_counter()
    res = fe_solve(case, backend="skfem")
    out = {"solve_s": round(time.perf_counter() - t, 1)}
    out["qois"] = res.evaluate_all()
    return out


def ccx_smoke(inp: str, axis: str, disp: float = 1e-5, E: float = 1.8798e9, nu: float = 0.35) -> dict:
    """Append a linear static step to mesh.inp: bottom fixed, top displaced by -disp along the axis."""
    ax = "xyz".index(axis) + 1
    deck = Path(inp).read_text()
    # drop surface-element blocks (the group ELSETs bottom/top hold triangles); keep tets in `domain`
    deck += f"""
*MATERIAL, NAME=RESIN
*ELASTIC
{E}, {nu}
*SOLID SECTION, ELSET=domain, MATERIAL=RESIN
*BOUNDARY
bottom, 1, 3, 0.
*BOUNDARY
top, {ax}, {ax}, {-disp}
*STEP
*STATIC
*NODE FILE
U
*EL FILE
S
*END STEP
"""
    with tempfile.TemporaryDirectory() as d:
        job = Path(d) / "job.inp"
        job.write_text(deck)
        t = time.perf_counter()
        p = subprocess.run([str(CCX), "job"], cwd=d, capture_output=True, text=True, timeout=1800)
        tail = p.stdout.strip().splitlines()[-3:]
        ok = (Path(d) / "job.frd").is_file() and "Job finished" in p.stdout and "*ERROR" not in p.stdout
        return {"ccx_ok": bool(ok), "rc": p.returncode, "s": round(time.perf_counter() - t, 1), "tail": tail,
                "errors": [l for l in p.stdout.splitlines() if "ERROR" in l][:3]}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mesh"); ap.add_argument("axis")
    ap.add_argument("--ccx", action="store_true")
    a = ap.parse_args()
    print(json.dumps(ccx_smoke(a.mesh, a.axis) if a.ccx else solve(a.mesh, a.axis)))
