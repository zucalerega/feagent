#!/usr/bin/env python3
"""Elastoplastic CalculiX ramp of the printed C-bracket compression test (problem abc_00000065_compression),
on feacore's opt-in nonlinear path (Material.plasticity + analysis.options.nonlinear, ccx backend).

The loop's solver is linear elastic; this standalone analysis follows the Instron test past yield, to compare
the whole force-displacement curve with the test and to see where and when the part yields.

Setup: the problem's own geometry, groups and P2 mesh; the bottom face held (group base_bottom); the prong
tops (group prong_tops) pushed down by a prescribed displacement, i.e. a rigid, level platen in full contact:
  slide  frictionless platen: only uy prescribed, the prong tops may move sideways (default);
  stick  sticking platen: ux = uz = 0 as well.
J2 plasticity with isotropic hardening, NLGEOM, displacement ramp in increments (cut back when needed).

Materials (Formlabs Alpha Fast, 2 min post-cure; Poisson's ratio 0.35 assumed), both from dogbone1.pdf:
  team       experiments/material/alpha_fast_dogbone1.json (nonlinear-setup branch): strain = crosshead / 115 mm
             grip separation, so E = 1880 MPa is apparent (grip and frame slack included) and the plastic
             strain is spread over the whole grip length;
  corrected  datasheet E = 2230 MPa; the dogbone's slack is its initial slope minus the elastic compliance of
             the printed Type I geometry (Tensile_Test_Type_I.STL: 13 x 3.2 mm narrow section, 57 mm long,
             fillets to 19 mm); the travel beyond the elastic line is plastic stretch of the narrow section.

Usage:
    python scripts/ccx_compression_ramp.py                                   # both materials, slide, h 3 mm
    python scripts/ccx_compression_ramp.py --materials corrected --platen stick --threads 4
    python scripts/ccx_compression_ramp.py --post-only                       # redo the comparison
Outputs (small, committed): physical_tests/abc_00000065_compression/ccx_ramp/{summary.json, curve_<run>.csv,
material_corrected.csv, compare.png}. CalculiX work dirs and (--vtu) the ParaView series stay in --work.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
TEST_DIR = ROOT / "physical_tests" / "abc_00000065_compression"
OUT_DIR = TEST_DIR / "ccx_ramp"
TEAM_MATERIAL = ROOT / "experiments" / "material" / "alpha_fast_dogbone1.json"
PID = "abc_00000065_compression"

DOG = {"t": 3.2, "w": 13.0, "L_narrow": 57.0, "w_grip": 19.0,              # Tensile_Test_Type_I.STL (mm)
       "fillet_x": [25.0, 32.1823, 39.4226, 46.7016, 54.0], "fillet_inset": [0.0, 1.3099, 2.248, 2.8119, 3.0]}
E_TDS, NU = 2230.0, 0.35            # MPa; Alpha Fast default post-cure (TDS); Poisson's ratio assumed
TEST_FIT = (1050.0, 1800.0)         # N: the test's straight part, after the toe and before yield


def read_curve(path: Path) -> tuple[np.ndarray, np.ndarray]:
    rows = [r for r in csv.reader(open(path)) if r and r[0][:1].isdigit()]
    d = np.array([[float(r[0]), float(r[1])] for r in rows])
    return d[:, 0], d[:, 1]


# --------------------------------------------------------------------------- materials (MPa, true values)

def dogbone_compliance(E: float) -> float:
    """Elastic compliance (mm/N) of the printed dogbone between the grips: narrow section plus two fillets."""
    xs = np.array(DOG["fillet_x"])
    w = DOG["w_grip"] - 2 * np.array(DOG["fillet_inset"])
    fillet = sum((xs[i + 1] - xs[i]) * np.log(w[i] / w[i + 1]) / (w[i] - w[i + 1]) for i in range(len(xs) - 1))
    return (DOG["L_narrow"] / DOG["w"] + 2 * fillet) / (E * DOG["t"])


def corrected_material(E: float = E_TDS, init=(30.0, 250.0)) -> dict:
    """Datasheet modulus; plastic curve from the dogbone with its grip / frame slack removed."""
    x, F = read_curve(TEST_DIR / "dogbone1.csv")
    ipk = int(np.argmax(F))
    x, F = x[: ipk + 1], F[: ipk + 1]
    m = (F >= init[0]) & (F <= init[1])
    k0, b0 = np.polyfit(x[m], F[m], 1)              # initial slope (N/mm) and its toe offset
    x0, C_meas, C_spec = -b0 / k0, 1.0 / k0, dogbone_compliance(E)
    A0 = DOG["w"] * DOG["t"]
    s_eng = F / A0
    d_pl = np.clip((x - x0) - F * C_meas, 0.0, None)    # travel beyond the elastic line (mm)
    d_pl[F < init[1]] = 0.0                              # the fit range and the toe below it are elastic
    e_eng = s_eng / E + d_pl / DOG["L_narrow"]
    s_true, e_true = s_eng * (1 + e_eng), np.log1p(e_eng)
    ep = np.maximum.accumulate(np.clip(e_true - s_true / E, 0.0, None))
    s_true = np.maximum.accumulate(s_true)
    i0 = int(np.argmax(ep > 2e-4))                       # initial yield: plastic strain passes 2e-4
    grid = np.unique(np.concatenate([[0.0], np.geomspace(5e-4, ep[-1], 18)]))
    table = [(float(np.interp(g + ep[i0], ep[i0:], s_true[i0:])), float(g)) for g in grid
             if g <= ep[-1] - ep[i0]]
    for i in range(1, len(table)):                       # non-decreasing flow stress
        table[i] = (max(table[i][0], table[i - 1][0]), table[i][1])
    return {"name": "corrected", "E": E, "nu": NU, "table": table,
            "source": "dogbone1.csv with slack removed: initial slope %.0f N/mm vs %.0f N/mm for the printed "
                      "Type I at E = %.0f MPa (slack %.2e mm/N); plastic stretch over the 57 mm narrow section"
                      % (k0, 1 / C_spec, E, C_meas - C_spec),
            "uts_eng_MPa": float(F[-1] / A0)}


def team_material() -> dict:
    m = json.loads(TEAM_MATERIAL.read_text())
    return {"name": "team", "E": float(m["E"]), "nu": float(m["nu"]),
            "table": [(float(s), float(e)) for s, e in m["plastic_table_true_stress_vs_plastic_strain"]],
            "source": f"{TEAM_MATERIAL.relative_to(ROOT)} ({m['name']}): " + "; ".join(m.get("caveats", [])),
            "uts_eng_MPa": float(m["uts_eng"]), "failure_peeq": m.get("failure_peeq")}


def plasticity(mat: dict) -> dict:
    """feacore Material.plasticity (SI): J2, yield at the first table stress, hardening [[eps_p, stress_Pa]]."""
    tab = sorted(mat["table"], key=lambda t: t[1])
    hard = [[0.0, tab[0][0] * 1e6]] + [[e, s * 1e6] for s, e in tab if e > 0]
    for i in range(1, len(hard)):
        hard[i][1] = max(hard[i][1], hard[i - 1][1])
    return {"model": "j2_isotropic", "yield_stress": hard[0][1], "hardening": hard}


# --------------------------------------------------------------------------- run

def ramp_case(mesh: Path, mat: dict, platen: str, umax_mm: float, increments: int):
    from feacore.case import BC, Material
    from feacore.problems.registry import get
    case = get(PID).case(mesh, order=2)
    case.materials = {name: Material(E=mat["E"] * 1e6, nu=mat["nu"], plasticity=plasticity(mat))
                      for name in case.materials}
    u = -umax_mm / 1000.0
    top = (BC(type="displacement", group="prong_tops", components=[1], value=u) if platen == "slide" else
           BC(type="displacement", group="prong_tops", components=[0, 1, 2], value=[0.0, u, 0.0]))
    case.bcs = [BC(type="fixed", group="base_bottom"), top]
    case.analysis.options = {**(case.analysis.options or {}), "nonlinear": True, "increments": increments,
                             "load_factor_max": 1.0, "nlgeom": True, "adaptive": True}
    case.meta = {**(case.meta or {}), "ramp": {"material": mat["name"], "platen": platen, "umax_mm": umax_mm}}
    return case


def run_one(name: str, mesh: Path, mat: dict, platen: str, umax_mm: float, increments: int, threads: int,
            work: Path, vtu: bool) -> dict:
    from feacore.solver import solve
    case = ramp_case(mesh, mat, platen, umax_mm, increments)
    t0 = time.time()
    r = solve(case, backend="ccx", threads=threads, workdir=work / name)
    wall = time.time() - t0
    rows = [r.increment_qois(i) for i in range(len(r.increments))]
    X = r.model.X * 1000.0                                           # mm

    def where(i, field, scale=1.0):
        r.use_increment(i)
        v = r.current.PEEQ if field == "peeq" else r.nodal(field)[1]
        n = int(np.argmax(v))
        return {"value": float(v[n]) * scale, "at_mm": X[n].round(2).tolist()}

    hist = [{"u_mm": row["lambda"] * umax_mm, "F_N": abs(float(row["reactions"]["prong_tops"][1])),
             "max_peeq": row["max_peeq"],
             "plastic_volume_fraction": row.get("plastic_volume_fraction", row["plastic_node_fraction"]),
             "max_von_mises_MPa": row["max_von_mises"] / 1e6, "iterations": row["iterations"]} for row in rows]
    out = {"name": name, "material": {k: mat[k] for k in ("name", "E", "nu", "source")}, "platen": platen,
           "umax_mm": umax_mm, "converged": bool(r.converged), "errors": r.errors[:3], "wall_s": round(wall),
           "n_nodes": int(r.model.n_nodes), "history": hist}
    first = next((i for i, h in enumerate(hist) if h["plastic_volume_fraction"] > 0), None)
    if first is not None:
        out["first_yield"] = {"u_mm": hist[first]["u_mm"], "F_N": hist[first]["F_N"], "peeq_max": where(first, "peeq")}
    last = len(rows) - 1
    out["last"] = {"u_mm": hist[-1]["u_mm"], "F_N": hist[-1]["F_N"], "peeq_max": where(last, "peeq"),
                   "s1_max_MPa": where(last, "s1", 1e-6)}
    tr = test_reference()          # the increment nearest the test's peak load: plastic zone, largest tension
    k = int(np.argmin([abs(h["F_N"] - tr["peak_N"]) for h in hist]))
    if hist[k]["F_N"] > 0.9 * tr["peak_N"]:
        r.use_increment(k)
        zone = X[r.current.PEEQ > 2e-3]
        out["at_test_peak_load"] = {"u_mm": hist[k]["u_mm"], "F_N": hist[k]["F_N"], "peeq_max": where(k, "peeq"),
                                    "s1_max_MPa": where(k, "s1", 1e-6),
                                    "plastic_zone_bbox_mm": [zone.min(0).round(1).tolist(), zone.max(0).round(1).tolist()]
                                    if len(zone) else None}
    if vtu:
        r.write_series(work / name / "series", extra={"ramp": out["material"], "platen": platen})
    (work / f"{name}.json").write_text(json.dumps(out, indent=1, default=float))
    return out


# --------------------------------------------------------------------------- comparison with the test

def test_reference() -> dict:
    x, F = read_curve(TEST_DIR / "cload_compression.csv")
    ipk = int(np.argmax(F))
    m = (F >= TEST_FIT[0]) & (F <= TEST_FIT[1]) & (np.arange(len(F)) < ipk)
    k, b = np.polyfit(x[m], F[m], 1)
    return {"x": x, "F": F, "slope": float(k), "toe_mm": float(-b / k), "peak_N": float(F[ipk]),
            "peak_mm": float(x[ipk])}


def compare(runs: list[dict]) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tr = test_reference()
    summary = {"test": {k: tr[k] for k in ("slope", "toe_mm", "peak_N", "peak_mm")},
               "note": "test displacement is crosshead travel; the toe is removed by shifting it to the intercept "
                       "of its straight part (%.0f-%.0f N)" % TEST_FIT, "runs": {}}
    fig, ax = plt.subplots(figsize=(8.8, 5.6), dpi=130)
    ax.plot(tr["x"] - tr["toe_mm"], tr["F"], color="0.5", lw=1.3, label="Instron test (toe removed)")
    for i, r in enumerate(runs):
        u = np.array([0.0] + [h["u_mm"] for h in r["history"]])
        F = np.array([0.0] + [h["F_N"] for h in r["history"]])
        k_fe = F[1] / u[1]
        C_s = 1.0 / tr["slope"] - 1.0 / k_fe              # load-train compliance that matches the test's slope
        u_s = u + F * C_s
        travel = tr["peak_mm"] - tr["toe_mm"]
        s = {k: r[k] for k in ("material", "platen", "converged", "wall_s", "n_nodes") if k in r}
        s.update({"initial_stiffness_N_per_mm": float(k_fe), "F_max_N": float(F.max()),
                  "u_at_F_max_mm": float(u[int(np.argmax(F))]), "u_reached_mm": float(u[-1]),
                  "series_compliance_to_match_test_mm_per_N": float(C_s),
                  "F_at_test_peak_travel_N": float(np.interp(travel, u, F)) if u[-1] >= travel else None,
                  "F_at_test_peak_travel_with_series_N": float(np.interp(travel, u_s, F)) if u_s[-1] >= travel else None})
        for key in ("first_yield", "at_test_peak_load", "last"):
            if key in r:
                s[key] = r[key]
        summary["runs"][r["name"]] = s
        with open(OUT_DIR / f"curve_{r['name']}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([f"# CalculiX J2 + NLGEOM ramp ({r['name']}): platen travel, force, max PEEQ; last column adds "
                        f"the series compliance {C_s:.3e} mm/N that matches the test's straight-part slope"])
            w.writerow(["u_mm", "F_N", "max_peeq", "u_with_series_mm"])
            for a, bF, pe, c in zip(u, F, [0.0] + [h["max_peeq"] for h in r["history"]], u_s):
                w.writerow([f"{a:.5f}", f"{bF:.2f}", f"{pe:.5f}", f"{c:.5f}"])
        col = f"C{i}"
        ax.plot(u, F, color=col, lw=1.7, label=f"{r['name']}: E {r['material']['E']:.0f} MPa, {r['platen']} platen")
        if C_s > 0:
            ax.plot(u_s, F, color=col, lw=1.1, ls="--", label=f"{r['name']} + load train {1 / C_s / 1e3:.1f} kN/mm")
        if "first_yield" in r:
            ax.plot(r["first_yield"]["u_mm"], r["first_yield"]["F_N"], "v", color=col, ms=7)
    ax.set_xlabel("platen travel [mm]"); ax.set_ylabel("force [N]"); ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 3.2); ax.set_ylim(0, 1.2 * tr["peak_N"])
    ax.set_title("abc_00000065 compression: Instron vs CalculiX (J2 plasticity, NLGEOM); triangles = first yield",
                 fontsize=9)
    ax.legend(fontsize=7.5, loc="lower right")
    fig.tight_layout(); fig.savefig(OUT_DIR / "compare.png")
    mc = corrected_material()
    with open(OUT_DIR / "material_corrected.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([f"# {mc['source']}; E = {mc['E']:.0f} MPa, nu = {mc['nu']}"])
        w.writerow(["true_stress_MPa", "true_plastic_strain"])
        w.writerows([[f"{s:.3f}", f"{e:.6f}"] for s, e in mc["table"]])
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--h", type=float, default=3.0, help="element size (mm)")
    ap.add_argument("--umax", type=float, default=3.0, help="platen travel (mm)")
    ap.add_argument("--increments", type=int, default=30)
    ap.add_argument("--materials", default="team,corrected")
    ap.add_argument("--platen", choices=["slide", "stick"], default="slide")
    ap.add_argument("--threads", type=int, default=2, help="threads per CalculiX run (runs go in parallel)")
    ap.add_argument("--work", type=Path, default=ROOT / "runs" / "ccx_ramp")
    ap.add_argument("--vtu", action="store_true", help="also write the per-increment .vtu series (ParaView)")
    ap.add_argument("--post-only", action="store_true", help="re-read <work>/<run>.json and redo the comparison")
    a = ap.parse_args(argv)
    mats = {"team": team_material, "corrected": corrected_material}
    names = [f"{m}_{a.platen}" for m in a.materials.split(",") if m]
    a.work.mkdir(parents=True, exist_ok=True)
    if not a.post_only:
        from feacore.problems.registry import get
        mesh = a.work / f"mesh_h{a.h:g}.msh"
        if not mesh.exists():
            get(PID).mesh(mesh, a.h / 1000.0, geom_order=2)
        jobs = [(n, mats[n.rsplit("_", 1)[0]]()) for n in names]
        with ThreadPoolExecutor(len(jobs)) as pool:
            futs = [pool.submit(run_one, n, mesh, m, a.platen, a.umax, a.increments, a.threads, a.work, a.vtu)
                    for n, m in jobs]
            for f in futs:
                r = f.result()
                print(f"{r['name']}: converged {r['converged']}, {len(r['history'])} increments, {r['wall_s']} s")
    runs = [json.loads((a.work / f"{n}.json").read_text()) for n in names if (a.work / f"{n}.json").exists()]
    s = compare(runs)
    print(json.dumps(s, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
