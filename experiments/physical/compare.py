"""Cross-system comparison of the mesh study: tables (Markdown + JSON) and one chart (PNG).

    python experiments/physical/compare.py --meshsys ROOT --study OUT [--crush-root runs/nonlinear]
        [--crush-run SYSTEM:PART:RUN_DIR ...]

Reads OUT/linear/<system>/<part>/summary.json (linear_ramp.py), OUT/f_full.json, each system's report.json under ROOT
(layout: experiments/mesh_systems/layout.json) and, where they exist, the crush runs (summary.json when the run is
done, else its force_displacement.csv chain: marked "running"). Rerunnable at any time.
Writes OUT/comparison.md, OUT/comparison.json, OUT/comparison.png."""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
import meshsys  # noqa: E402

# categorical slots 1-6 (fixed order = layout order of the systems)
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
AX = {"x": 0, "y": 1, "z": 2}


def _rows(run: Path, cut=None):
    out = []
    for r in csv.DictReader(open(run / "force_displacement.csv")):
        st = int(float(r["step"]))
        if cut is None or st <= cut:
            out.append((st, float(r.get("platen_travel_mm") or r.get("travel_mm")), float(r["force_N"]),
                        int(float(r.get("n_failed", 0))), float(r.get("wall_s") or 0.0)))
    return out


def crush_chain(run: Path):
    """[(step, travel_mm, force_N, n_failed, wall_s)] over the run and the runs it continued, + the run dirs."""
    runs = [run]
    while (runs[0] / "run.json").exists():
        p = json.loads((runs[0] / "run.json").read_text()).get("continued_from")
        if not p or not Path(p).exists():
            break
        runs.insert(0, Path(p))
    rows, wall = [], 0.0
    for i, r in enumerate(runs):
        cut = None
        if i + 1 < len(runs) and (runs[i + 1] / "run.json").exists():
            cut = json.loads((runs[i + 1] / "run.json").read_text()).get("continued_at_step")
        rr = _rows(r, cut)
        if rr:
            w = [x[4] for x in rr]
            wall += max(w) if max(w) > 0 else _mtime_span(r)
        rows += rr
    return rows, runs, wall


def _mtime_span(run: Path) -> float:
    v = sorted(f.stat().st_mtime for f in run.glob("result_*.vtu"))
    return v[-1] - v[0] if len(v) > 1 else 0.0


def first_failure_location(runs, step, part):
    """Height fraction (0 bottom platen, 1 top) and centroid [mm] of the cells failed at the first failing step."""
    import meshio
    f = next((r / f"result_{step:04d}.vtu" for r in reversed(runs) if (r / f"result_{step:04d}.vtu").exists()), None)
    if f is None:
        return None
    m = meshio.read(f)
    st = np.asarray(m.cell_data["state"][0] if "state" in m.cell_data else m.cell_data["failed"][0])
    conn = m.cells[0].data
    corners = {"tetra10": 4, "tetra": 4, "hexahedron": 8, "hexahedron20": 8}.get(m.cells[0].type, conn.shape[1])
    cen = m.points[conn[:, :corners]].mean(axis=1)
    k = AX[meshsys.axis(part)]
    lo, hi = m.points[:, k].min(), m.points[:, k].max()
    sel = st == 1
    if not sel.any():
        return None
    c = cen[sel].mean(axis=0)
    return {"height_fraction": float((c[k] - lo) / (hi - lo)), "centroid_mm": [round(float(x) * 1e3, 2) for x in c],
            "n_cells": int(sel.sum())}


def crush_metrics(run: Path, part: str) -> dict:
    s = json.loads((run / "summary.json").read_text()) if (run / "summary.json").exists() else None
    rows, runs, wall = crush_chain(run)
    if not rows:
        return {"run": str(run), "status": "no increments yet"}
    F = np.array([r[2] for r in rows]); ip = int(np.argmax(F))
    ff = next((r for r in rows if r[3] > 0), None)
    d = {"run": str(run), "runs": [str(r) for r in runs], "status": "done" if s else "running",
         "n_increments": len(rows), "travel_now_mm": rows[-1][1], "force_now_N": rows[-1][2],
         "peak_force_N": float(F[ip]), "travel_at_peak_mm": rows[ip][1],
         "first_failure_travel_mm": ff[1] if ff else None, "n_failed_now": rows[-1][3], "wall_s": wall,
         "curve": [[r[1], r[2]] for r in rows]}
    if s:
        d.update({"stop_reason": s.get("stop_reason"), "separated": s.get("separated"),
                  "travel_at_separation_mm": s.get("travel_at_separation_mm"),
                  "peak_force_N": s.get("peak_force_N", d["peak_force_N"]),
                  "travel_at_peak_mm": s.get("travel_at_peak_mm", d["travel_at_peak_mm"]),
                  "wall_s": s.get("wall_s_total_approx") or s.get("wall_s") or wall})
        fl = s.get("first_failure") or {}
        if fl.get("height_fraction") is not None:
            d["first_failure"] = {"height_fraction": fl["height_fraction"],
                                  "centroid_mm": [round(x * 1e3, 2) for x in fl.get("cells_centroid_m") or []]}
    if "first_failure" not in d and ff:
        d["first_failure"] = first_failure_location(runs, ff[0], part)
    return d


def newest_run(base: Path, system: str, part: str):
    cands = [x for x in (base / system / part).glob("*") if (x / "force_displacement.csv").exists()] \
        if (base / system / part).exists() else []
    if not cands and system == "harness_auto":                        # baseline runs: runs/nonlinear/<part>/run-*
        cands = [x for x in (base / part).glob("run-*") if (x / "force_displacement.csv").exists()]
    return max(cands, key=lambda x: x.stat().st_mtime) if cands else None


def spread(vals: dict) -> dict:
    v = {k: x for k, x in vals.items() if x is not None}
    if not v:
        return {}
    a = np.array(list(v.values()), float); med = float(np.median(a))
    return {"median": med, "min": float(a.min()), "max": float(a.max()), "n": len(a),
            "range_pct": float(100 * (a.max() - a.min()) / abs(med)) if med else None,
            "dev_pct": {k: float(100 * (x - med) / abs(med)) if med else None for k, x in v.items()}}


def fmt(x, nd=3):
    if x is None:
        return "-"
    if isinstance(x, (int, np.integer)):
        return f"{x:,}"
    return f"{x:.{nd}g}" if abs(x) < 1e5 else f"{x:,.0f}"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--meshsys", default=os.environ.get("MESHSYS_OUT"), required=os.environ.get("MESHSYS_OUT") is None)
    ap.add_argument("--study", required=True)
    ap.add_argument("--crush-root", default=str(Path(os.environ.get("FEACORE_RUNS_DIR", REPO / "runs")) / "nonlinear"))
    ap.add_argument("--crush-run", action="append", default=[])
    a = ap.parse_args(argv)
    root, study = Path(a.meshsys), Path(a.study)
    ff = json.loads((study / "f_full.json").read_text()) if (study / "f_full.json").exists() else {"parts": {}}
    explicit = {tuple(x.split(":", 2)[:2]): Path(x.split(":", 2)[2]) for x in a.crush_run}
    systems, parts = meshsys.systems(), meshsys.parts()
    data = {"f_full": ff, "systems": {s: meshsys.label(s) for s in systems}, "rows": {}, "spread": {}}
    for p in parts:
        Ff = ff["parts"].get(p, {}).get("F_full_N")
        for s in systems:
            row = {"system": s, "part": p}
            for u in meshsys.PURPOSES:
                md = meshsys.mesh_dir(root, s, p, u)
                if meshsys.ready(md):
                    row[f"mesh_{u}"] = meshsys.report(md)
            ls = study / "linear" / s / p / "summary.json"
            if ls.exists():
                L = json.loads(ls.read_text()); sc = (Ff or L["force_N"]) / L["force_N"]
                row["linear"] = {"dofs": L["n_dofs"], "nodes": L["n_nodes"], "solve_s": L["solve_s"],
                                 "reaction_N": L["reaction_N"],
                                 "compliance_um_at_100N": L["compliance_2U_over_F_m"] * 1e6 * 100 / L["force_N"],
                                 "stiffness_kN_per_mm": L["stiffness_N_per_m"] * 1e-6,
                                 "u_axis_top_um_at_Ffull": abs(L["mean_u_axis_top_m"]) * 1e6 * sc,
                                 "u_mag_top_um_at_Ffull": L["mean_u_mag_top_m"] * 1e6 * sc,
                                 "peak_vm_MPa_at_Ffull": L["peak_von_mises_Pa"] * 1e-6 * sc,
                                 "robust_vm_MPa_at_Ffull": L["robust_von_mises_Pa"] * 1e-6 * sc}
            rd = explicit.get((s, p)) or newest_run(Path(a.crush_root), s, p)
            if rd is not None and rd.exists():
                row["crush"] = crush_metrics(rd, p)
            data["rows"][f"{s}/{p}"] = row
        R = [data["rows"][f"{s}/{p}"] for s in systems]
        sp = {}
        for key in ("dofs", "compliance_um_at_100N", "u_axis_top_um_at_Ffull", "u_mag_top_um_at_Ffull",
                    "peak_vm_MPa_at_Ffull", "robust_vm_MPa_at_Ffull", "solve_s"):
            sp[key] = spread({r["system"]: r.get("linear", {}).get(key) for r in R})
        sp["build_s_linear"] = spread({r["system"]: r.get("mesh_linear", {}).get("build_s") for r in R})
        for key in ("peak_force_N", "travel_at_peak_mm", "travel_at_separation_mm", "wall_s"):
            sp["crush_" + key] = spread({r["system"]: r.get("crush", {}).get(key) for r in R
                                         if r.get("crush", {}).get("status") == "done"})
        data["spread"][p] = sp
    (study / "comparison.json").write_text(json.dumps(data, indent=1, default=float))
    write_md(study, data, systems, parts)
    chart(study, data, systems, parts)
    print(f"wrote {study / 'comparison.md'}, comparison.json, comparison.png")


def write_md(study, data, systems, parts):
    L = ["# Mesh-system comparison", "",
         "Same two parts, same loads and material for every system (experiments/mesh_systems/CONTRACT.md). Linear: "
         "feacore/skfem tet10, bottom fixed, total force along -axis on top spread per area, Formlabs Alpha Fast "
         "Resin from the dogbone (E 1879.8 MPa, nu 0.35 assumed, UTS 50.48 MPa). Compliance = 2U/F at 100 N "
         "(work-equivalent platen travel; equals the area-mean axial travel of the loaded face). Stresses are "
         "scaled to F_full; 'robust' = 99.9th percentile of the volume-weighted cell von Mises (the peak is "
         "singular at the platen edges and mesh-dependent). Spread = (max - min) / median.", ""]
    for p in parts:
        f = data["f_full"]["parts"].get(p, {})
        L += [f"## {p} (axis {meshsys.axis(p)})", "",
              f"F_full = **{f.get('F_full_N', float('nan')):.0f} N** (robust von Mises = 87 % of the UTS, median over "
              f"systems; per-system values {f.get('spread_pct', '-')} % apart). Video deformation x{f.get('video_mag', '-')}.", "",
              "| system | lin. DOFs | lin. nodes | lin. gates fail / warn | build s | compliance um @100 N | dev % | "
              "u_axis top um @F_full | mean abs(u) top um | peak vM MPa | robust vM MPa | dev % | solve s |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        sp = data["spread"][p]
        for s in systems:
            r = data["rows"][f"{s}/{p}"]; li = r.get("linear"); m = r.get("mesh_linear", {})
            if not li:
                L.append(f"| {s} | (no linear run) |||||||||||| "); continue
            L.append(f"| {s} | {li['dofs']:,} | {li['nodes']:,} | {', '.join(m.get('fail', [])) or '-'} / "
                     f"{', '.join(m.get('warn', [])) or '-'} | {fmt(m.get('build_s'))} | "
                     f"{li['compliance_um_at_100N']:.3f} | {sp['compliance_um_at_100N']['dev_pct'][s]:+.2f} | "
                     f"{li['u_axis_top_um_at_Ffull']:.1f} | {li['u_mag_top_um_at_Ffull']:.1f} | "
                     f"{li['peak_vm_MPa_at_Ffull']:.1f} | {li['robust_vm_MPa_at_Ffull']:.1f} | "
                     f"{sp['robust_vm_MPa_at_Ffull']['dev_pct'][s]:+.1f} | {li['solve_s']:.1f} |")
        L += ["", "Spread across systems:", "", "| quantity | median | min | max | spread % | n |", "|---|---|---|---|---|---|"]
        names = {"compliance_um_at_100N": "compliance um @100 N", "u_axis_top_um_at_Ffull": "u_axis top um @F_full",
                 "u_mag_top_um_at_Ffull": "mean abs(u) top um @F_full", "peak_vm_MPa_at_Ffull": "peak von Mises MPa",
                 "robust_vm_MPa_at_Ffull": "robust von Mises MPa", "dofs": "linear DOFs", "solve_s": "linear solve s",
                 "build_s_linear": "linear mesh build s", "crush_peak_force_N": "crush peak force N (done runs)",
                 "crush_travel_at_peak_mm": "crush travel at peak mm (done runs)",
                 "crush_travel_at_separation_mm": "crush travel at separation mm (done runs)"}
        for k, n in names.items():
            v = sp.get(k)
            if v:
                L.append(f"| {n} | {fmt(v['median'], 4)} | {fmt(v['min'], 4)} | {fmt(v['max'], 4)} | "
                         f"{fmt(v['range_pct'], 3)} | {v['n']} |")
        L += ["", "Nonlinear meshes and crush runs (CalculiX J2 + erosion; running runs show the state so far):", "",
              "| system | nl. nodes | nl. cells | element | nl. gates fail / warn | status | increments | peak force N | "
              "travel at peak mm | travel at separation mm | first failure travel mm | first failure height (0 bottom, 1 top) "
              "| first failure centroid mm | wall h |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for s in systems:
            r = data["rows"][f"{s}/{p}"]; m = r.get("mesh_nonlinear", {}); c = r.get("crush")
            head = (f"| {s} | {fmt(m.get('n_nodes'))} | {fmt(m.get('n_elements'))} | {m.get('element') or ('tet' + str(m.get('order') or ''))} | "
                    f"{', '.join(m.get('fail', [])) or '-'} / {', '.join(m.get('warn', [])) or '-'} | ")
            if not c or "peak_force_N" not in c:
                L.append(head + "no run | | | | | | | | |"); continue
            fl = c.get("first_failure") or {}
            st = c["status"] + (f" ({c.get('stop_reason')})" if c.get("stop_reason") else
                                f", now {c['travel_now_mm']:.3f} mm / {c['force_now_N']:.0f} N")
            L.append(head + f"{st} | {c['n_increments']} | {c['peak_force_N']:.0f} | {c['travel_at_peak_mm']:.3f} | "
                     f"{fmt(c.get('travel_at_separation_mm'))} | {fmt(c.get('first_failure_travel_mm'))} | "
                     f"{fmt(fl.get('height_fraction'), 2)} | {fl.get('centroid_mm', '-')} | {c['wall_s'] / 3600:.2f} |")
        L.append("")
    L += ["Chart: comparison.png. Data: comparison.json. Rerun: `python experiments/physical/compare.py --meshsys ROOT "
          "--study OUT`.", ""]
    (study / "comparison.md").write_text("\n".join(L))


def chart(study, data, systems, parts):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b", "xtick.color": "#52514e",
                         "ytick.color": "#52514e", "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6})
    fig, axs = plt.subplots(2, 3, figsize=(15, 8.5), facecolor="#fcfcfb")
    col = {s: COLORS[i] for i, s in enumerate(systems)}
    w = 0.8 / len(systems)
    for j, (key, title) in enumerate((("compliance_um_at_100N", "Compliance 2U/F: deviation from median (%)"),
                                      ("robust_vm_MPa_at_Ffull", "Robust von Mises (99.9 % vol.): deviation (%)"),
                                      ("peak_vm_MPa_at_Ffull", "Peak von Mises (singular): deviation (%)"))):
        ax = axs[0, j]
        for i, s in enumerate(systems):
            ys = [data["spread"][p][key].get("dev_pct", {}).get(s) for p in parts]
            xs = [k + (i - (len(systems) - 1) / 2) * w for k in range(len(parts))]
            ok = [(x, y) for x, y in zip(xs, ys) if y is not None]
            if ok:
                ax.bar([x for x, _ in ok], [y for _, y in ok], width=w * 0.9, color=col[s], label=s)
        ax.axhline(0, color="#52514e", lw=0.8); ax.set_xticks(range(len(parts))); ax.set_xticklabels(parts)
        ax.set_title(title, loc="left", fontsize=10)
    axs[0, 0].legend(frameon=False, fontsize=8, ncol=2)
    for k, p in enumerate(parts):            # cost vs answer
        ax = axs[1, k]
        for i, s in enumerate(systems):
            li = data["rows"][f"{s}/{p}"].get("linear")
            if li:
                ax.scatter(li["dofs"], li["compliance_um_at_100N"], s=60, color=col[s], edgecolor="#fcfcfb", lw=2, zorder=3)
                ax.annotate(s, (li["dofs"], li["compliance_um_at_100N"]), xytext=(6, 5 if i % 2 == 0 else -11), textcoords="offset points",
                            fontsize=8, color="#52514e")
        ax.set_xlabel("linear DOFs"); ax.set_ylabel("compliance at 100 N (um)")
        ax.set_title(f"{p}: cost vs answer", loc="left", fontsize=10)
    ax = axs[1, 2]
    any_c = False
    for p, ls in zip(parts, ("-", "--")):
        for s in systems:
            c = data["rows"][f"{s}/{p}"].get("crush")
            if c and c.get("curve"):
                cv = np.array([[0.0, 0.0]] + c["curve"]); any_c = True
                ax.plot(cv[:, 0], cv[:, 1] / 1e3, ls, color=col[s], lw=2, label=f"{s} {p[-2:]}"
                        + (" (running)" if c["status"] == "running" else ""))
    ax.set_xlabel("platen travel (mm)"); ax.set_ylabel("force (kN)")
    ax.set_title("Crush: force vs travel (solid 003, dashed 065)", loc="left", fontsize=10)
    if any_c:
        ax.legend(frameon=False, fontsize=7)
    for a_ in axs.ravel():
        a_.set_facecolor("#fcfcfb")
    fig.suptitle("Mesh-system study: same parts, loads, material; feacore/skfem linear + CalculiX crush", x=0.01,
                 ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(study / "comparison.png", dpi=130, facecolor=fig.get_facecolor())
    plt.close(fig)


if __name__ == "__main__":
    main()
