"""Reproduce GB10 Host's slab benchmark using the existing FEAgent heat solver.

Run from the repository root: .venv/bin/python experiments/sentilab_thermal/run.py
No network access, remote model, or third-party service is needed.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/feagent-matplotlib")

import meshio
import numpy as np
import scipy
import skfem
from skfem import MeshTet

from feacore.case import Analysis, BC, Case, Load, Material, QoI
from feacore.solver import solve

OUT = ROOT / "evaluation_runs/sentilab_thermal"
L, WIDTH, K, POWER, H, AMBIENT = .001, .020, 150., 10., 1000., 25.
SOURCE = POWER / (WIDTH * WIDTH * L)


def exact(z, mode):
    offset = 0. if mode == "A" else POWER / (H * WIDTH**2)
    return AMBIENT + offset + SOURCE * z * (2 * L - z) / (2 * K)


def mesh(path, n):
    """Full 3D tensor grid split into tetrahedra, with named boundary faces.

    n subdivisions along z; 2n along each in-plane direction. Refinement
    halves every edge; fixed aspect ratio is disclosed in the report.
    """
    m = MeshTet.init_tensor(np.linspace(0, WIDTH, 2*n+1),
                            np.linspace(0, WIDTH, 2*n+1),
                            np.linspace(0, L, n+1))
    faces = m.facets[:, m.boundary_facets()].T
    z = m.p[2, faces].mean(axis=1)
    tags = np.where(np.isclose(z, 0, atol=1e-14), 2,
                    np.where(np.isclose(z, L, atol=1e-14), 3, 4))
    meshio.write(path, meshio.Mesh(
        points=m.p.T, cells=[("tetra", m.t.T), ("triangle", faces)],
        cell_data={"gmsh:physical": [np.ones(m.nelements, dtype=int), tags],
                   "gmsh:geometrical": [np.ones(m.nelements, dtype=int), tags]},
        field_data={"domain": [1, 3], "bottom": [2, 2], "top": [3, 2], "sides": [4, 2]}),
        file_format="gmsh22", binary=False)


def run_case(meshpath, n, mode, order):
    name = f"{mode}_P{order}_nz{n}"
    bc = BC("temperature", group="bottom", value=AMBIENT) if mode == "A" else BC(
        "convection", group="bottom", h=H, T_inf=AMBIENT)
    case = Case(mesh=meshpath.name,
        analysis=Analysis(physics="heat", formulation="3d", order=order,
                          options={"linear_solver": "scipy"}),
        materials={"slab": Material(k=K)}, bcs=[bc],
        loads=[Load("source", SOURCE)],
        qois=[QoI("peak_C", "max", field="T"), QoI("mean_C", "mean", field="T"),
              QoI("bottom_mean_C", "mean", field="T", group="bottom")],
        meta={"origin": "GB10 Host, Senti room reply #57170, 2026-10-31T20:54:14Z",
              "units": "SI; temperature in Celsius", "adiabatic": "top and sides (natural zero flux)"})
    case.save(OUT / f"{name}.json")
    case.base_dir = str(OUT)
    result = solve(case, backend="skfem")
    values = result.evaluate_all()
    md = result.model
    fb = md.facet_basis(md.facets_of("bottom"))
    boundary_T = np.asarray(fb.interpolate(result.x))
    gradient = np.asarray(fb.interpolate(result.x).grad)
    # Raw FE gradient, not the globally recovered flux. Outward normal
    # on the bottom is -z: (-k grad T).n = +k dT/dz.
    raw_outward = -K * np.sum(gradient * np.asarray(fb.normals), axis=0)
    flux_out = float(np.sum(raw_outward * fb.dx))
    if mode == "A":
        conservative_out = -result.evaluate(QoI("bottom_reaction", "reaction", group="bottom"))
    else:
        conservative_out = float(np.sum(H * (boundary_T-AMBIENT) * fb.dx))
    free = np.setdiff1d(np.arange(len(result.x)), result.dirichlet_dofs)
    prescribed = np.zeros_like(result.x)
    prescribed[result.dirichlet_dofs] = result.x[result.dirichlet_dofs]
    rhs = (result.f - result.K @ prescribed)[free]
    residual = float(np.linalg.norm((result.K @ result.x-result.f)[free]) / np.linalg.norm(rhs))
    # Domain L2 field error using solver quadrature (not nodal sampling).
    zq = np.asarray(md.sb.global_coordinates())[2]
    field_error = np.asarray(md.sb.interpolate(result.x)) - exact(zq, mode)
    volume = float(np.sum(md.sb.dx))
    l2 = float(np.sqrt(np.sum(field_error**2 * md.sb.dx) / volume))
    coordinates, temperatures = result.nodal("T")
    nodal_error = float(np.max(np.abs(temperatures - exact(coordinates[:, 2], mode))))
    z = np.linspace(0, L, 101)
    pts = np.column_stack([np.full_like(z, WIDTH/2), np.full_like(z, WIDTH/2), z])
    profile = result.probe("T", pts)
    np.savetxt(OUT / f"{name}_profile.csv", np.column_stack([z, profile, exact(z, mode)]),
               delimiter=",", header="z_m,T_FEA_C,T_exact_C", comments="")
    result.write_vtu(OUT / f"{name}.vtu")
    solver_line = next(w for w in result.info.warnings if "rel_residual=" in w)
    linear_time = float(re.search(r"t=([\d.e+\-]+)s", solver_line)[1])
    row = dict(case=mode, order=order, nx=2*n, ny=2*n, nz=n,
               cells=result.info.n_cells, dofs=result.info.n_dofs,
               peak_C=values["peak_C"], exact_peak_C=float(exact(L, mode)),
               mean_C=values["mean_C"], exact_mean_C=float(exact(0, mode)+SOURCE*L**2/(3*K)),
               bottom_mean_C=values["bottom_mean_C"],
               L2_temperature_error_K=l2, max_nodal_error_K=nodal_error,
               bottom_outward_conservative_W=conservative_out,
               bottom_outward_raw_gradient_W=flux_out,
               balance_relative_error=abs(conservative_out-POWER)/POWER,
               free_equation_relative_residual=residual,
               linear_solve_s_reported=linear_time, assembly_solve_recovery_s=result.info.wall_time,
               volume_m3=volume, warnings=result.info.warnings)
    if abs(volume-WIDTH**2*L) > 1e-15 or residual > 1e-8 or row["balance_relative_error"] > 1e-8:
        raise RuntimeError(f"Geometry, residual, or energy check failed: {name}: {row}")
    if order == 2 and (l2 > 1e-7 or abs(flux_out-POWER) > 1e-5):
        raise RuntimeError(f"Quadratic patch check failed: {name}: {row}")
    print(f"{name}: {row['dofs']} DOFs, peak={row['peak_C']:.9f} C, "
          f"L2={l2:.3g} K, outward={conservative_out:.9f} W, raw={flux_out:.6f} W", flush=True)
    return row, (z, profile)


def main():
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT,
                        help="Output directory; default is the archived benchmark directory")
    OUT = parser.parse_args().out.resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    rows, profiles = [], {}
    for n in (2, 4, 8):
        path = OUT / f"slab_nz{n}.msh"
        mesh(path, n)
        for order in (1, 2):
            for mode in ("A", "B"):
                row, profile = run_case(path, n, mode, order)
                rows.append(row)
                profiles[mode, order, n] = profile
    for mode in ("A", "B"):
        errors = [r["L2_temperature_error_K"] for r in rows if r["case"] == mode and r["order"] == 1]
        if not all(a/b > 2.5 for a, b in zip(errors, errors[1:])):
            raise RuntimeError(f"P1 temperature L2 error did not converge: {mode}, {errors}")
    environment = {"python": sys.version, "platform": platform.platform(),
                   "numpy": np.__version__, "scipy": scipy.__version__, "skfem": skfem.__version__,
                   "meshio": meshio.__version__, "git_commit": subprocess.check_output(
                       ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()}
    (OUT / "results.json").write_text(json.dumps(dict(environment=environment, results=rows), indent=2))
    fields = [k for k in rows[0] if k != "warnings"]
    with (OUT / "results.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    for ax, mode in zip(axes[:2], ("A", "B")):
        z = np.linspace(0, L, 201)
        ax.plot(z*1000, exact(z, mode), "k-", label="Exact")
        for order in (1, 2):
            zp, tp = profiles[mode, order, 2]
            ax.plot(zp[::10]*1000, tp[::10], "o" if order == 1 else "+", label=f"P{order}, coarse")
        ax.set(xlabel="z (mm)", ylabel="Temperature (C)", title=f"Case {mode}")
        ax.legend()
    for mode in ("A", "B"):
        rr = [r for r in rows if r["case"] == mode and r["order"] == 1]
        axes[2].loglog([r["nz"] for r in rr], [r["L2_temperature_error_K"] for r in rr], "o-", label=f"{mode}, P1")
    axes[2].set(xlabel="Elements through thickness", ylabel="RMS field error (K)", title="Mesh convergence")
    axes[2].legend()
    fig.savefig(OUT / "verification.png", dpi=180)
    plt.close(fig)
    table = "\n".join(f"|{r['case']}|P{r['order']}|{r['nx']} x {r['ny']} x {r['nz']}|{r['dofs']}|"
                        f"{r['peak_C']:.9f}|{r['mean_C']:.9f}|{r['L2_temperature_error_K']:.3g}|"
                        f"{r['bottom_outward_conservative_W']:.9f}|{r['bottom_outward_raw_gradient_W']:.6f}|"
                        f"{r['free_equation_relative_residual']:.2g}|{r['assembly_solve_recovery_s']:.3f}|" for r in rows)
    report = f"""# FEAgent independent slab verification for SentiLab

Completed locally with the existing FEAgent `feacore.solver.solve`, scikit-fem 3D heat backend,
SciPy SuperLU direct solve. No remote AI model was used. No core solver code was changed.
Specification: GB10 Host's reply to message #57170, 2026-10-31 20:54:14 UTC.

## Model and exact answer

Full 20 x 20 x 1 mm slab; all inputs converted to SI. Conductivity 150 W/(m K),
uniform source 25,000,000 W/m3 (10 W total). Top and sides have natural zero flux.
Case A: bottom fixed at 25 C. Case B: bottom convection h=1000 W/(m2 K), ambient 25 C.
With L=0.001 m, T(z)=T_bottom+q*z*(2*L-z)/(2*k). Case A T_bottom=25 C;
case B T_bottom=25+10/(1000*0.02**2)=50 C.
Exact peaks: A 25.083333333 C, B 50.083333333 C.
Exact volume means: A 25.055555556 C, B 50.055555556 C. Outward bottom heat: 10 W.

## Results

|Case|Basis|Grid subdivisions|DOFs|Peak C|Mean C|RMS field error K|Conservative bottom W|Raw gradient bottom W|Relative residual|Assembly+solve+recovery s|
|---|---|---|---|---|---|---|---|---|---|---|
{table}

Grid cells are split into 6 affine tetrahedra. Each refinement halves all edge lengths;
in-plane grid spacing is 10 times through-thickness spacing. P1 and P2 share the same
straight geometry. P2 adds edge DOFs. Meshes, case JSON, field VTU, 101-point central
z-profiles, CSV results and software versions are included beside this report.

## Checks and interpretation

All 12 solves pass volume, free-equation residual and global heat-balance checks.
P1 RMS field error decreases by more than 2.5x on each refinement (approximately second
order). P2 represents the exact quadratic profile, so its remaining errors are numerical
roundoff; P2 is a patch check, not an estimate of convergence for complicated chip packages.
Peak is the maximum FE nodal temperature; mean and RMS error use volume quadrature.

Conservative bottom heat uses negative Dirichlet reaction for A and the integrated
h*(T-T_ambient) Robin law for B. Independently, raw heat flux integrates
(-k*grad(T)).n using element gradients on the bottom, without flux recovery. The coarse
P1 gradient need not give 10 W even when conservative heat balance closes; it converges
under refinement. P2 gives the exact flux to numerical precision.

Residual is ||(K*T-f)_free|| / ||(f-K*T_prescribed)_free||. Timing in the table includes
assembly, main solve and flux recovery; `linear_solve_s_reported` in CSV is the main
factorization/solve time parsed from the solver diagnostic (rounded there). Mesh generation,
postprocessing and output are excluded. Timings are single local runs, not performance claims.

This verifies uniform-source units, flux sign and convection mapping on this slab.
It does not validate chiplet contact resistances, multilayer packages, measured material
properties, nonlinear heat transfer, or physical measurements. Next useful collaboration:
reuse this benchmark in the receiving team's solver, then agree on one shared package case.
"""
    (OUT / "REPORT.md").write_text(report)
    print(f"PASS: all 12 solves and refinement checks. Report: {OUT / 'REPORT.md'}")


if __name__ == "__main__":
    main()
