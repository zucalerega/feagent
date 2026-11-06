"""Verification of the 4 wingbox GPU solves (equilibrium, box-beam theory, GCI, von Mises peak).

Reads RUN/<case>/gpu/{result.json,result.vtu,watchdog.txt} for the cases c2, c1, fine, xf.
Writes RUN/verify/verification.json and RUN/summary.json. CPU only.

    CUDA_VISIBLE_DEVICES="" uv run python experiments/wingbox/verify/verify_all.py
"""
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pyvista as pv

HERE = Path(__file__).resolve().parent
RUN = HERE.parent
sys.path.insert(0, str(HERE))
import beam_theory  # noqa: E402

from feacore.verification.gci import _observed_order, analyze, gci3  # noqa: E402

CASES = ["assembly_c2", "assembly_c1", "assembly", "assembly_xf"]   # coarse -> fine
P = 20000.0
EQ_TOL = 1e-6
P_THEORY = {"mean_uz_tip": 3.0, "mean_uz_tip_ends": 3.0, "strain_energy": 4.0}  # tet10 smooth-solution rates


def load(case):
    d = RUN / case / os.environ.get("SOLVE_DIR", "gpu")   # "gpu" = Spark cuDSS run; run_cpu.sh uses "cpu"
    res = json.loads((d / "result.json").read_text())
    wd = (d / "watchdog.txt").read_text()
    kv = dict(re.findall(r"(\w+)=([-\w.]+)", wd))
    lin = next(w for w in res["info"]["warnings"] if "linear_solver=" in w)
    t_lin = float(re.search(r"t=([\d.]+)s", lin).group(1))
    return res, kv, lin, t_lin, d / "result.vtu"


def probe(grid, pts, fields):
    s = pv.PolyData(np.asarray(pts, float)).sample(grid)
    ok = np.asarray(s["vtkValidPointMask"]).astype(bool)
    return {f: np.where(ok, np.asarray(s[f]), np.nan).tolist() for f in fields}


def stress_study(vtu):
    g = pv.read(vtu)
    X = np.asarray(g.points)
    vm = np.asarray(g["von_mises"])
    out = {}
    i = int(np.argmax(vm))
    out["vm_max"] = float(vm[i])
    out["vm_max_at"] = X[i].round(3).tolist()
    # the 20 highest nodes: are they all at the clamped root?
    top = np.argsort(vm)[-20:][::-1]
    out["top20_y_max_mm"] = float(X[top, 1].max())
    out["top20_at"] = X[top[:5]].round(2).tolist()
    for lo, hi, key in [(0.0, 50.0, "root_y0_50"), (100.0, 1900.0, "y100_1900"),
                        (300.0, 1700.0, "y300_1700")]:
        m = (X[:, 1] >= lo) & (X[:, 1] <= hi)
        j = np.flatnonzero(m)[np.argmax(vm[m])]
        out[f"vm_max_{key}"] = float(vm[j])
        out[f"vm_max_{key}_at"] = X[j].round(2).tolist()
    # skin bending stress sigma_yy at the upper-skin mid-plane (z = 140), across the width, at
    # mid-bay sections y = 200 (root bay) and y = 1000; shear lag shows as edge > centre
    xs = [7.5, 30.0, 100.0, 200.0, 300.0, 400.0, 500.0, 570.0, 592.5]
    for y in (200.0, 1000.0):
        pr = probe(g, [[x, y, 140.0] for x in xs], ["syy", "von_mises"])
        out[f"syy_upper_skin_y{int(y)}"] = dict(zip([str(x) for x in xs], pr["syy"]))
    # web shear at the neutral axis, mid-bay y = 1000, front web mid-thickness
    pr = probe(g, [[7.5, 1000.0, 75.0], [592.5, 1000.0, 75.0]], ["syz"])
    out["syz_webs_y1000_z75"] = pr["syz"]
    return out


def celik(h, phi, p_theory, fs=1.25):
    """Celik et al. (2008) 3-mesh procedure written out (fixed-point p for unequal ratios).

    feacore's gci3 classifies by R = e21 / e32 < 1 before computing p. With unequal ratios
    (here r21 = 1.46, r32 = 1.16) a monotone sequence converging at p ~ 2.3 has R ~ 1.44, so gci3
    reports 'divergent'. Monotonicity is checked here on the signs of e21, e32 instead."""
    idx = np.argsort(h)
    (h1, h2, h3), (f1, f2, f3) = np.asarray(h)[idx], np.asarray(phi)[idx]
    r21, r32 = h2 / h1, h3 / h2
    e21, e32 = f2 - f1, f3 - f2
    s = float(np.sign(e32 / e21))
    p = _observed_order(e21, e32, r21, r32, s)
    phi_ext = (r21**p * f1 - f2) / (r21**p - 1)
    ea21 = abs(e21 / f1)
    gci_fine = fs * ea21 / (r21**p - 1)
    gci_med = fs * abs(e32 / f2) / (r32**p - 1)
    p_lim = min(max(p, 0.5), p_theory)
    return {"h": [h1, h2, h3], "phi": [f1, f2, f3], "r21": r21, "r32": r32,
            "monotone": s > 0, "p": p, "phi_ext": phi_ext, "e_ext21": abs((phi_ext - f1) / phi_ext),
            "gci_fine": gci_fine, "gci_medium": gci_med,
            "gci_fine_conservative": 3.0 * ea21 / (r21**p_lim - 1),
            "asymptotic_ratio": gci_med / (r21**p * gci_fine)}


def main():
    bt = beam_theory.main()
    delta_bt = bt["delta_total_mm"]
    rows, stress = {}, {}
    for c in CASES:
        res, kv, lin, t_lin, vtu = load(c)
        q = res["qois"]
        rows[c] = {
            "dofs": res["info"]["n_dofs"], "nodes": res["info"]["n_nodes"], "cells": res["info"]["n_cells"],
            "h_rep_solver": res["info"]["h_rep"], "solve_wall_s": res["info"]["wall_time"],
            "linear_solve_s": t_lin, "linear_solver": lin, "scope_wall_s": float(kv["wall_s"]),
            "min_MemAvailable_GiB": float(kv["min_MemAvailable_GiB"]),
            "MemAvailable_drop_GiB": float(kv["drop_GiB"]), "cgroup_peak_GiB": float(kv["cgroup_peak_GiB"]),
            "watchdog_killed": kv["killed"] != "0", "qois": q,
            "equilibrium": {
                "rel_err_Rz": abs(q["reaction_root_z"] + P) / P,
                "rel_Rx": abs(q["reaction_root_x"]) / P, "rel_Ry": abs(q["reaction_root_y"]) / P,
            },
            "energy_check_rel": abs(q["strain_energy"] - 0.5 * P * q["mean_uz_tip"]) / q["strain_energy"],
            "tip_vs_beam_rel": q["mean_uz_tip"] / delta_bt - 1.0,
        }
        rows[c]["equilibrium"]["pass"] = max(rows[c]["equilibrium"][k] for k in
                                             ("rel_err_Rz", "rel_Rx", "rel_Ry")) < EQ_TOL
        stress[c] = stress_study(vtu)
        print(c, rows[c]["dofs"], q["mean_uz_tip"], stress[c]["vm_max"], stress[c]["vm_max_at"], flush=True)

    dofs = np.array([rows[c]["dofs"] for c in CASES], float)
    h = dofs ** (-1.0 / 3.0)
    conv = {}
    series = {n: [rows[c]["qois"][n] for c in CASES] for n in
              ("mean_uz_tip", "mean_uz_tip_ends", "strain_energy", "max_von_mises")}
    series["vm_max_y300_1700"] = [stress[c]["vm_max_y300_1700"] for c in CASES]
    series["syy_upper_skin_y1000_x300"] = [stress[c]["syy_upper_skin_y1000"]["300.0"] for c in CASES]
    for n, vals in series.items():
        a = analyze(h, vals, p_theory=P_THEORY.get(n, 2.0))
        conv[n] = {"values_coarse_to_fine": vals, **a}
    for n in ("mean_uz_tip", "mean_uz_tip_ends", "strain_energy"):
        conv[n]["celik_fine_triple"] = celik(h[1:], series[n][1:], P_THEORY[n])
    conv["_h"] = {"definition": "h = DOFs^(-1/3)", "h": h.tolist(),
                  "r_xf_fine": float(h[2] / h[3]), "r_fine_c1": float(h[1] / h[2]),
                  "r_c1_c2": float(h[0] / h[1]), "gci3_meshes": CASES[1:]}
    # GCI on the 3 coarsest too, to show sensitivity of p to the mesh triple
    conv["_gci3_coarse_triple"] = {n: gci3(h[:3], series[n][:3], p_theory=P_THEORY.get(n, 2.0)).to_dict()
                                   for n in ("mean_uz_tip", "strain_energy")}

    xf = "assembly_xf"
    out = {"beam_theory": bt, "cases": rows, "stress": stress, "convergence": conv}
    (HERE / "verification.json").write_text(json.dumps(out, indent=2, default=float))

    g_uz = conv["mean_uz_tip"]["celik_fine_triple"]
    g_se = conv["strain_energy"]["celik_fine_triple"]
    lib_uz, lib_se = conv["mean_uz_tip"]["gci"], conv["strain_energy"]["gci"]
    summary = {
        "dofs": {c: rows[c]["dofs"] for c in CASES},
        "solve_s": {c: round(rows[c]["solve_wall_s"], 2) for c in CASES},
        "linear_solve_s": {c: rows[c]["linear_solve_s"] for c in CASES},
        "solver": "cudss on NVIDIA GB10 (DGX Spark)",
        "tip_deflection_mm": {c: rows[c]["qois"]["mean_uz_tip"] for c in CASES},
        "tip_deflection_extrapolated_mm": g_uz["phi_ext"],
        "beam_theory_mm": delta_bt,
        "gci": {
            "h_definition": "DOFs^(-1/3); 3-mesh GCI on c1/fine/xf (r21 = 1.46, r32 = 1.16), "
                            "least-squares fit on all 4",
            "note": "feacore gci3 labels both 'divergent' (R = e21/e32 = 1.44 > 1, an artefact of "
                    "unequal r); p/phi_ext/GCI below are Celik's procedure with feacore's "
                    "_observed_order; fit4 = feacore analyze() least-squares fit",
            "mean_uz_tip": {"p": g_uz["p"], "phi_ext": g_uz["phi_ext"], "gci_fine": g_uz["gci_fine"],
                            "gci_fine_conservative": g_uz["gci_fine_conservative"],
                            "gci3_library": {"convergence": lib_uz["convergence"], "R": lib_uz["R"], "flags": lib_uz["flags"]},
                            "fit4": conv["mean_uz_tip"]["fit"]},
            "strain_energy": {"p": g_se["p"], "phi_ext": g_se["phi_ext"], "gci_fine": g_se["gci_fine"],
                              "gci_fine_conservative": g_se["gci_fine_conservative"],
                              "gci3_library": {"convergence": lib_se["convergence"], "R": lib_se["R"], "flags": lib_se["flags"]},
                              "fit4": conv["strain_energy"]["fit"]},
        },
        "von_mises_max_mpa": stress[xf]["vm_max"],
        "von_mises_max_at": stress[xf]["vm_max_at"],
        "von_mises_max_by_mesh_mpa": {c: rows[c]["qois"]["max_von_mises"] for c in CASES},
        "reaction_z_n": rows[xf]["qois"]["reaction_root_z"],
        "verdict": None,   # set by hand after review (verdict.md)
    }
    (RUN / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in ("tip_deflection_mm", "beam_theory_mm", "gci")}, indent=1))


if __name__ == "__main__":
    main()
