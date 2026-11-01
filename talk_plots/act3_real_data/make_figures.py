#!/usr/bin/env python3
"""Act 3 of the talk: the harness learns from real lab data (abc_00000065 crushed on an Instron).

Regenerates every figure of talk_plots/act3_real_data/ (PNG 200 dpi + PDF) and act3_data.json, the exact
numbers plotted. Read-only on its inputs; no solver and no API calls.

    .venv/bin/python talk_plots/act3_real_data/make_figures.py

Inputs (all already computed, CalculiX J2 + NLGEOM on fe-ref-1):
  physical_tests/abc_00000065_compression/cload_compression.csv   measured curve (the frozen scoring's test file)
  physical_tests/abc_00000065_compression/nonlinear_spec.json     frozen setup: checkpoints, delta, compliance
  <scratchpad>/nl_bc/<variant>/curve.csv     BC variants on a uniform 4 mm P2 mesh (u_mm = part travel, F_N)
  <scratchpad>/nl_mesh/<mesh>/curve.csv      sticking setup on 5 meshes (u_mm = part travel, F_N)
  <scratchpad>/nl_mesh/nodes.txt             node counts of those meshes
  physical_tests/abc_00000065_compression/ccx_ramp/render_team_{slide,stick}.png   teammates' PEEQ renders

Error definition (= ace.nonlinear.run_curve): u_cross = u_part + delta + c * F with (0, 0) prepended; e = mean
over the checkpoints of |F_sim(u_k) - F_meas(u_k)| / F_meas(u_k), F_meas on the test's loading branch up to
its force maximum; a checkpoint the curve never reaches counts 100 %.
"""
from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SCRATCH = Path("/private/tmp/claude-501/-Users-aryan-Documents-Aryan-CS-scienceclawhack/"
               "29d391ed-69f3-4559-b365-a260e486ee85/scratchpad")
PT = ROOT / "physical_tests" / "abc_00000065_compression"
TEST_CSV = PT / "cload_compression.csv"
SPEC = PT / "nonlinear_spec.json"
BC_DIR, MESH_DIR = SCRATCH / "nl_bc", SCRATCH / "nl_mesh"

# fixed talk colours
C_AUTO, C_TEST, C_UNIF, C_SWARM = "#888888", "#000000", "#6b6b6b", "#c23d6e"   # swarm bc0 mesh = div_s8 (8 agents)
AGENT_COLOURS = {1: "#2a78d6", 2: "#e0882a", 4: "#1e9e6a", 8: "#c23d6e"}
C_SLIDE, C_STICK, C_OTHER = "#7a5195", "#1f8a8a", "#b8b8b8"

MESH_LABEL = {"auto": "gmsh default (automesher)", "swarm": "swarm's best bc0 mesh", "h4": "uniform 4 mm",
              "h25": "uniform 2.5 mm", "h20": "uniform 2 mm"}
BC_LABEL = {"baseline": "frictionless platens", "bonded": "sticking platens",
            "tilt_f17": "frictionless, prong f17 at half travel", "tilt_f34": "frictionless, prong f34 at half travel",
            "one_face": "frictionless, only prong f17 pushed", "elastic": "frictionless, elastic (no plasticity)"}


# --------------------------------------------------------------------------- style

def style():
    plt.rcParams.update({
        "figure.figsize": (12, 6.75), "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "font.size": 15, "axes.labelsize": 16, "xtick.labelsize": 14,
        "ytick.labelsize": 14, "legend.fontsize": 14, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": "#e6e6e6", "grid.linewidth": 0.9,
        "axes.axisbelow": True, "legend.frameon": False,
    })


def titled(fig, title: str, subtitle: str):
    import textwrap
    sub = textwrap.fill(subtitle, 104)
    fig.suptitle(title, x=0.04, y=0.975, ha="left", fontsize=19, fontweight="bold")
    fig.text(0.04, 0.915, sub, ha="left", va="top", fontsize=14, color="#555555")
    top = 0.86 if "\n" not in sub else 0.82
    fig.subplots_adjust(left=0.08, right=0.97, top=top, bottom=0.11)


def save(fig, stem: str):
    fig.savefig(HERE / f"{stem}.png", dpi=200)
    fig.savefig(HERE / f"{stem}.pdf")
    plt.close(fig)
    print("wrote", stem)


# --------------------------------------------------------------------------- data

def read_test() -> tuple[np.ndarray, np.ndarray]:
    """Whole digitized test curve (mm, N), comment and unit rows skipped."""
    u, F = [], []
    for row in csv.reader(TEST_CSV.read_text().splitlines()):
        try:
            u.append(float(row[0])), F.append(float(row[1]))
        except (ValueError, IndexError):
            pass
    return np.array(u), np.array(F)


def loading_branch(u, F):
    p = int(np.argmax(F))
    o = np.argsort(u[: p + 1], kind="stable")
    return u[: p + 1][o], F[: p + 1][o]


def read_curve(path: Path) -> tuple[np.ndarray, np.ndarray]:
    c = np.loadtxt(path, delimiter=",", skiprows=1)
    return c[:, 0], c[:, 1]


def mapped(u_part, F, delta, c):
    return np.concatenate([[0.0], u_part[1:] + delta + c * F[1:]])


def errors(u_part, F, delta, c, ck, Fm):
    """(mean rel. error, per-checkpoint rel. errors, simulated forces) as ace.nonlinear.run_curve."""
    uc = mapped(u_part, F, delta, c)
    Fs = np.interp(ck, uc, F)
    rel = np.where(ck <= uc[-1], np.abs(Fs - Fm) / Fm, 1.0)
    return float(rel.mean()), rel, Fs


def fit(u_part, F, ck, Fm, start=(0.233, 2.9e-5), compliance=True):
    from scipy.optimize import minimize
    if compliance:
        f = lambda x: errors(u_part, F, x[0], x[1] * 1e-5, ck, Fm)[0]
        r = minimize(f, [start[0], start[1] * 1e5], method="Nelder-Mead",
                     options={"xatol": 1e-5, "fatol": 1e-7, "maxiter": 4000})
        return float(r.x[0]), float(r.x[1]) * 1e-5, float(r.fun)
    ds = np.arange(0.0, 0.8, 0.0005)
    es = [errors(u_part, F, d, 0.0, ck, Fm)[0] for d in ds]
    i = int(np.argmin(es))
    return float(ds[i]), 0.0, float(es[i])


def compute() -> dict:
    spec = json.loads(SPEC.read_text())
    ck = np.array(spec["checkpoints_mm"], float)
    delta, c = spec["delta_mm"], spec["compliance_mm_per_N"]
    u, F = read_test()
    ub, Fb = loading_branch(u, F)
    Fm = np.interp(ck, ub, Fb)

    # measured-curve features
    lin = (Fb >= 1050) & (Fb <= 1800)
    toe = (ub >= 0.1) & (ub <= 0.45)
    p = int(np.argmax(F))
    test = {"checkpoints_mm": ck.tolist(), "force_meas_N": Fm.round(2).tolist(),
            "peak_N": float(F[p]), "peak_mm": float(u[p]), "break_mm": float(u[-1]),
            "slope_linear_N_per_mm": float(np.polyfit(ub[lin], Fb[lin], 1)[0]), "linear_band_N": [1050, 1800],
            "slope_toe_N_per_mm": float(np.polyfit(ub[toe], Fb[toe], 1)[0]), "toe_band_mm": [0.1, 0.45],
            "F_at_0p5mm_N": float(np.interp(0.5, ub, Fb)),
            "n_points": int(len(u)), "file": str(TEST_CSV.relative_to(ROOT))}
    # second digitization (scratchpad), for the caveat only
    w = np.loadtxt(SCRATCH / "wf/cload_compression_curve.csv", delimiter=",", skiprows=1)
    test["alt_digitization"] = {"file": str(SCRATCH / "wf/cload_compression_curve.csv"),
                                "peak_N": float(w[:, 1].max()), "peak_mm": float(w[w[:, 1].argmax(), 0])}

    # BC variants, raw crosshead = part travel (no seating, no compliance)
    bc = {}
    for v in BC_LABEL:
        up, Fv = read_curve(BC_DIR / v / "curve.csv")
        e, rel, Fs = errors(up, Fv, 0.0, 0.0, ck, Fm)
        bc[v] = {"label": BC_LABEL[v], "u_mm": up.tolist(), "F_N": Fv.tolist(), "mean_rel_err": e,
                 "rel_err": rel.round(5).tolist(), "F_at_end_N": float(Fv[-1]), "u_end_mm": float(up[-1]),
                 "file": str(BC_DIR / v / "curve.csv")}

    # meshes, sticking setup
    nodes = {l.split()[0]: int(l.split()[2]) for l in (MESH_DIR / "nodes.txt").read_text().splitlines() if l.strip()}
    curves = {m: read_curve(MESH_DIR / m / "curve.csv") for m in MESH_LABEL}
    d_fit, c_fit, e_fit = fit(*curves["h20"], ck, Fm)
    d_seat, _, _ = fit(*curves["h20"], ck, Fm, compliance=False)
    meshes = {}
    for m, (up, Fv) in curves.items():
        e, rel, Fs = errors(up, Fv, delta, c, ck, Fm)
        meshes[m] = {"label": MESH_LABEL[m], "nodes": nodes[m], "mean_rel_err": e, "rel_err": rel.round(5).tolist(),
                     "force_sim_N": Fs.round(1).tolist(), "u_part_mm": up.tolist(), "F_N": Fv.tolist(),
                     "u_cross_mm": mapped(up, Fv, delta, c).round(5).tolist(),
                     "mean_rel_err_own_fit": errors(up, Fv, d_fit, c_fit, ck, Fm)[0],
                     "file": str(MESH_DIR / m / "curve.csv")}
    # bc0 4 mm "bonded" and mesh h4 are the same solve (same mesh + setup): check
    same = np.allclose(read_curve(BC_DIR / "bonded/curve.csv")[1], curves["h4"][1])

    # setup-refinement ladder: everything on the 4 mm mesh, then the 2 mm mesh; seating/compliance fitted on 2 mm
    h4, h20 = curves["h4"], curves["h20"]
    fr = read_curve(BC_DIR / "baseline/curve.csv")
    ladder = [
        ("frictionless platens\n(raw solver)", errors(*fr, 0, 0, ck, Fm)[0], "4 mm"),
        ("+ seating gap\n(\u03b4 = %.2f mm)" % d_seat, errors(*fr, d_seat, 0, ck, Fm)[0], "4 mm"),
        ("+ sticking platens", errors(*h4, d_seat, 0, ck, Fm)[0], "4 mm"),
        ("+ load-train\ncompliance (frozen)", errors(*h4, delta, c, ck, Fm)[0], "4 mm"),
        ("+ fine mesh\n(2 mm, 41k nodes)", errors(*h20, delta, c, ck, Fm)[0], "2 mm"),
    ]
    other_orders = {"sticking_without_seating_4mm": errors(*h4, 0, 0, ck, Fm)[0],
                    "frictionless_seating_own_fit": fit(*fr, ck, Fm, compliance=False)[2],
                    "frictionless_frozen_mapping": errors(*fr, delta, c, ck, Fm)[0]}
    ladder_h20 = {"sticking": errors(*h20, 0, 0, ck, Fm)[0], "sticking+seating": errors(*h20, d_seat, 0, ck, Fm)[0],
                  "sticking+seating+compliance": errors(*h20, delta, c, ck, Fm)[0]}
    # can the mapping rescue frictionless platens? fit delta, c on the frictionless curve itself
    fr_best = fit(*fr, ck, Fm, start=(0.0, 1e-6))

    # cross-check against the committed baselines (other workflow's results, read only)
    committed = {}
    bj = ROOT / "results/real_data_finetune/baselines/baselines.json"
    if bj.exists():
        b = json.loads(bj.read_text())["baselines"]
        committed = {k: {"nodes": v["nodes"], "dofs": v["dofs"], "mean_rel_err": v["mean_rel_err"]} for k, v in b.items()}

    return {"spec": {"file": str(SPEC.relative_to(ROOT)), "delta_mm": delta, "compliance_mm_per_N": c,
                     "checkpoints_mm": ck.tolist(), "reference_errors": spec.get("reference_errors")},
            "refit_on_h20": {"delta_mm": d_fit, "compliance_mm_per_N": c_fit, "mean_rel_err": e_fit,
                             "seating_only_delta_mm": d_seat},
            "test": test, "bc_variants": bc, "meshes": meshes, "bonded_equals_h4": bool(same),
            "ladder": [{"step": s, "mean_rel_err": e, "mesh": m} for s, e, m in ladder],
            "ladder_on_2mm_mesh": ladder_h20, "ladder_other_orders": other_orders,
            "frictionless_best_mapping": {"delta_mm": fr_best[0], "compliance_mm_per_N": fr_best[1],
                                          "mean_rel_err": fr_best[2]},
            "committed_baselines": committed}


# --------------------------------------------------------------------------- figures

def fig1_test(D):
    u, F = read_test()
    t = D["test"]
    fig, ax = plt.subplots()
    ax.axvspan(0, t["peak_mm"], color="#f2f2f2", zorder=0)
    ax.plot(u, F / 1e3, color=C_TEST, lw=2.6)
    ax.set_xlim(0, 9.2), ax.set_ylim(0, 5.9)
    ax.set_xlabel("crosshead travel (mm)"), ax.set_ylabel("force (kN)")
    kw = dict(fontsize=14, color="#222222", arrowprops=dict(arrowstyle="-", color="#777777", lw=1))
    ax.annotate(f"toe: soft start, ~{t['slope_toe_N_per_mm']/1e3:.1f} kN/mm\n(seating, one prong first?)",
                (0.3, 0.52), (0.9, 0.35), **kw)
    ax.annotate(f"linear: {t['slope_linear_N_per_mm']/1e3:.2f} kN/mm", (1.2, 2.67), (1.6, 1.75), **kw)
    ax.annotate("softening: the resin yields", (2.05, 4.22), (0.25, 4.95), **kw)
    ax.plot(t["peak_mm"], t["peak_N"] / 1e3, "o", color=C_TEST, ms=9)
    ax.annotate(f"peak {t['peak_N']/1e3:.2f} kN at {t['peak_mm']:.2f} mm", (t["peak_mm"], t["peak_N"] / 1e3),
                (3.1, 5.45), **kw)
    ax.annotate("cracking: saw-tooth load drops", (5.4, 3.2), (5.2, 4.55), **kw)
    ax.annotate(f"break at {t['break_mm']:.2f} mm", (t["break_mm"], 1.7), (6.6, 0.9), **kw)
    ax.text(0.08, 5.6, "scored region (loading branch)", fontsize=13, color="#555555")
    titled(fig, "Real test: seating, linear climb, yielding to 4.8 kN, then cracking",
           "3D-printed abc_00000065 (Formlabs Alpha Fast resin) crushed between Instron platens; "
           "force vs crosshead travel, digitized from the machine's PDF report")
    save(fig, "fig1_measured_curve")


def fig2_bc(D):
    t, bc = D["test"], D["bc_variants"]
    ub, Fb = loading_branch(*read_test())
    fig, ax = plt.subplots()
    ax.plot(ub, Fb / 1e3, color=C_TEST, lw=3.4, zorder=5)
    ax.text(2.0, 3.8, f"measured: {np.interp(2.5, ub, Fb)/1e3:.1f} kN", fontsize=14, fontweight="bold")
    style_of = {"baseline": dict(color=C_SLIDE, lw=3.2), "bonded": dict(color=C_STICK, lw=3.2),
                "tilt_f17": dict(color=C_OTHER, lw=2, ls="--"), "tilt_f34": dict(color=C_OTHER, lw=2, ls=":"),
                "one_face": dict(color=C_OTHER, lw=2, ls="-."), "elastic": dict(color="#9db4d6", lw=2)}
    # direct labels at the right end; y positions (kN) set by hand to avoid overlaps
    label_y = {"bonded": 5.42, "elastic": 4.98, "baseline": 3.12, "tilt_f17": 2.62, "one_face": 1.47}
    names = {"bonded": "sticking platens", "elastic": "elastic, no yield",
             "baseline": "frictionless platens", "tilt_f17": "one prong at half travel*",
             "one_face": "one prong only"}
    for v in ["tilt_f17", "tilt_f34", "one_face", "elastic", "bonded", "baseline"]:
        r = bc[v]
        ax.plot(r["u_mm"], np.array(r["F_N"]) / 1e3, **style_of[v], zorder=4 if v in ("baseline", "bonded") else 3)
        if v in label_y:
            col = style_of[v]["color"] if v in ("baseline", "bonded") else "#666666"
            ax.text(2.62, label_y[v], f"{names[v]}: {np.interp(2.5, r['u_mm'], r['F_N'])/1e3:.1f} kN, "
                                      f"{r['mean_rel_err']*100:.0f} % off", fontsize=13.5, va="center", color=col,
                    fontweight="bold" if v in ("baseline", "bonded") else "normal")
    for x in t["checkpoints_mm"]:
        ax.axvline(x, color="#cccccc", lw=0.8, ls=":", zorder=0)
    ax.plot(t["checkpoints_mm"], np.array(t["force_meas_N"]) / 1e3, "o", color=C_TEST, ms=8, zorder=6)
    ax.set_xlim(0, 4.6), ax.set_ylim(0, 5.8)
    ax.set_xticks([0, 0.5, 1, 1.5, 2, 2.5])
    ax.set_xlabel("travel (mm): crosshead for the test, platen for the simulation")
    ax.set_ylabel("force (kN)")
    a, b2 = np.array(bc["tilt_f17"]["F_N"][1:]), np.array(bc["tilt_f34"]["F_N"][1:])
    dmax = float(np.max(np.abs(a / b2 - 1)))
    ax.text(2.62, 0.25, "light lines: frictionless variants\nkN = force at 2.5 mm; % = mean error at the 5 dots\n"
                        f"* f17 or f34 at half travel: curves differ by \u2264 {dmax*100:.1f} %", fontsize=12,
            color="#666666")
    b = bc["baseline"]
    titled(fig, f"The raw solver misses reality: frictionless platens give out at {b['F_at_end_N']/1e3:.1f} kN",
           "CalculiX J2 plasticity + large deformation (NLGEOM), uniform 4 mm P2 mesh, dogbone material; "
           "no seating gap or machine compliance yet")
    save(fig, "fig2_raw_solver_vs_test")


def fig3_ladder(D):
    L = D["ladder"]
    fig, ax = plt.subplots()
    x = np.arange(len(L))
    e = np.array([s["mean_rel_err"] * 100 for s in L])
    cols = [C_SLIDE, "#a58fbd", "#4ba3a3", C_STICK, "#0f5c5c"]
    ax.bar(x, e, width=0.62, color=cols, zorder=3)
    for xi, ei in zip(x, e):
        ax.text(xi, ei + 0.6, f"{ei:.1f} %", ha="center", va="bottom", fontsize=17, fontweight="bold")
    ax.set_xticks(x, [s["step"] for s in L], fontsize=14)
    ax.set_ylabel("mean error vs the test (%)")
    ax.set_ylim(0, max(e) * 1.18)
    ax.axvline(3.5, color="#999999", lw=1, ls="--")
    ax.text(1.5, max(e) * 1.1, "setup learned from the experiment (same 4 mm mesh)", ha="center",
            fontsize=14, color="#555555")
    ax.text(4, max(e) * 1.1, "mesh", ha="center", fontsize=14, color="#555555")
    ax.grid(axis="x", visible=False)
    titled(fig, f"Learning the setup: error {e[0]:.0f} % → {e[-2]:.1f} %; a fine mesh: {e[-1]:.1f} %",
           "mean |F_sim - F_test| / F_test at 0.5, 1.0, 1.5, 2.0, 2.5 mm crosshead travel; CalculiX J2 + NLGEOM; "
           "seating gap and compliance fitted once on the 2 mm mesh, then frozen")
    save(fig, "fig3_setup_ladder")


def draw_error_vs_nodes(ax, D, target=True, extra=None):
    """Figure 4's axes: reference meshes, the model-error floor, the target box; `extra` = list of
    {"nodes", "mean_rel_err", "label"} (fine-tuned points, drawn as filled swarm diamonds)."""
    M = D["meshes"]
    floor = M["h20"]["mean_rel_err"] * 100
    uni = [M[k] for k in ("h4", "h25", "h20")]
    ax.axhspan(0, floor, color="#ececec", zorder=0)
    ax.text(78000, floor / 2, f"finest mesh ({floor:.1f} %): roughly the model-error floor", va="center",
            ha="right", fontsize=13, color="#555555")
    if target:
        ax.add_patch(Rectangle((2000, floor), 8000 - 2000, 2.0 - floor, fill=True, facecolor="#c23d6e14",
                               edgecolor=C_SWARM, lw=2, ls="--", zorder=1))
        ax.text(2120, 2.3, "TARGET: fine-tuned swarm\n≤ 2 % at ≤ 8k nodes\n"
                           "(to be filled by the run)", fontsize=13, color=C_SWARM, va="bottom",
                fontweight="bold")
    ax.plot([m["nodes"] for m in uni], [m["mean_rel_err"] * 100 for m in uni], "-o", color=C_UNIF, lw=2, ms=9,
            zorder=3)
    for i, m in enumerate(uni):
        off, ha = [((12, 4), "left"), ((0, 46), "center"), ((0, 12), "center")][i]
        ax.annotate(f"{m['label']}\n{m['nodes']:,} nodes: {m['mean_rel_err']*100:.1f} %",
                    (m["nodes"], m["mean_rel_err"] * 100), textcoords="offset points", xytext=off, ha=ha,
                    va="bottom", fontsize=13, color=C_UNIF)
    a, s = M["auto"], M["swarm"]
    ax.plot(a["nodes"], a["mean_rel_err"] * 100, "s", color=C_AUTO, ms=12, zorder=4)
    ax.annotate(f"gmsh default (automesher)\n{a['nodes']:,} nodes: {a['mean_rel_err']*100:.1f} %",
                (a["nodes"], a["mean_rel_err"] * 100), textcoords="offset points", xytext=(14, -6), fontsize=13,
                va="center", color="#444444")
    ax.plot(s["nodes"], s["mean_rel_err"] * 100, "D", mfc="white", mec=C_SWARM, mew=2.5, ms=12, zorder=4)
    ax.annotate(f"swarm's best bc0 mesh, before fine-tuning\n(trained on a different load case)\n"
                f"{s['nodes']:,} nodes: {s['mean_rel_err']*100:.1f} %", (s["nodes"], s["mean_rel_err"] * 100),
                textcoords="offset points", xytext=(14, 8), fontsize=13, color=C_SWARM, va="bottom")
    for p in extra or []:
        ax.plot(p["nodes"], p["mean_rel_err"] * 100, "D", color=C_SWARM, ms=11, zorder=5)
        ax.annotate(p["label"], (p["nodes"], p["mean_rel_err"] * 100), textcoords="offset points", xytext=(8, -14),
                    fontsize=12, color=C_SWARM)
    ax.set_xscale("log")
    xs = [p["nodes"] for p in extra or []]
    ax.set_xlim(min([2000] + [x / 1.2 for x in xs]), max([80000] + [x * 1.5 for x in xs]))
    ymax = max(18.0, max([a["mean_rel_err"] * 100] + [p["mean_rel_err"] * 100 for p in extra or []]) * 1.1)
    ax.set_ylim(0, ymax)
    ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator([2000, 5000, 10000, 20000, 50000]))
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("mesh nodes (log scale)")
    ax.set_ylabel("mean error vs the real test (%)")


def fig4_error_vs_nodes(D):
    fig, ax = plt.subplots()
    draw_error_vs_nodes(ax, D)
    a = D["meshes"]["auto"]
    titled(fig, f"With the learned setup, the automesher is {a['mean_rel_err']*100:.0f} % off the real test",
           "frozen setup (sticking platens + seating gap + compliance); mean error at the 5 checkpoints vs the "
           "Instron curve; one CalculiX J2 + NLGEOM ramp per mesh, all P2")
    save(fig, "fig4_error_vs_nodes")


def draw_curves(ax, D, extra=None, xmax=2.75):
    """Figure 5's axes; `extra` = list of {"u_cross_mm", "F_N", "label"} drawn in the swarm colour."""
    t, M = D["test"], D["meshes"]
    ub, Fb = loading_branch(*read_test())
    ax.plot(ub, Fb / 1e3, color=C_TEST, lw=3.6, zorder=5)
    ax.text(1.62, 3.0, "measured", fontsize=15, fontweight="bold")
    for x in t["checkpoints_mm"]:
        ax.axvline(x, color="#cccccc", lw=0.8, ls=":", zorder=0)
    ax.plot(t["checkpoints_mm"], np.array(t["force_meas_N"]) / 1e3, "o", color=C_TEST, ms=8, zorder=6)
    curves = [(f"gmsh default (automesher), {M['auto']['nodes']:,} nodes: {M['auto']['mean_rel_err']*100:.1f} %",
               M["auto"], dict(color=C_AUTO, lw=3, ls="--"), "#555555"),
              (f"swarm's best bc0 mesh, {M['swarm']['nodes']:,} nodes: {M['swarm']['mean_rel_err']*100:.1f} %",
               M["swarm"], dict(color=C_SWARM, lw=3), C_SWARM),
              (f"uniform 2 mm, {M['h20']['nodes']:,} nodes: {M['h20']['mean_rel_err']*100:.1f} %",
               M["h20"], dict(color=C_UNIF, lw=1.8, alpha=0.8), "#555555")]
    curves += [(e["label"], e, dict(color=C_SWARM, lw=3, ls="-."), C_SWARM) for e in extra or []]
    ends = []
    for lab, r, kw, tc in curves:
        uc, F = np.array(r["u_cross_mm"]), np.array(r["F_N"]) / 1e3
        keep = uc <= xmax
        uc_end = min(xmax, uc[-1])
        F_end = float(np.interp(uc_end, uc, F))
        ax.plot(np.append(uc[keep], uc_end), np.append(F[keep], F_end), **kw, zorder=4)
        ends.append([F_end, lab, tc])
    ends.sort(key=lambda z: -z[0])                     # labels top to bottom, pushed apart by >= 0.3 kN
    for i in range(1, len(ends)):
        ends[i][0] = min(ends[i][0], ends[i - 1][0] - 0.3)
    for y, lab, tc in ends:
        ax.text(xmax + 0.04, y, lab, fontsize=13, va="center", color=tc)
    ax.set_xlim(0, xmax + 2.1), ax.set_ylim(0, 6.2)
    ax.set_xticks([0, 0.5, 1, 1.5, 2, 2.5])
    ax.set_xlabel("crosshead travel (mm)")
    ax.set_ylabel("force (kN)")


def fig5_curves(D):
    fig, ax = plt.subplots()
    draw_curves(ax, D)
    a = D["meshes"]["auto"]
    over = np.array(a["force_sim_N"]) / np.array(D["test"]["force_meas_N"]) - 1
    titled(fig, f"The automesher's mesh is too stiff: {over.min()*100:.0f}-{over.max()*100:.0f} % above the "
                f"real test", "frozen setup, same solver and physics, only the mesh differs; dots = the 5 scoring "
                "checkpoints; this gap is what fine-tuning on the test must close")
    save(fig, "fig5_curves_vs_test")


def fig6_renders(D):
    from PIL import Image
    src = PT / "ccx_ramp"
    summ = json.loads((src / "summary.json").read_text())["runs"]
    for n in ("team_slide", "team_stick"):
        shutil.copyfile(src / f"render_{n}.png", HERE / f"render_{n}.png")
    fig, axs = plt.subplots(2, 1)
    sl, st = summ["team_slide"], summ["team_stick"]
    rows = [("team_slide", f"frictionless platens: prongs splay, force tops out at {sl['F_max_N']/1e3:.2f} kN",
             C_SLIDE),
            ("team_stick", f"sticking platens: still rising, {st['F_at_test_peak_travel_N']/1e3:.2f} kN at the "
                           "test's peak travel", C_STICK)]
    for ax, (n, lab, col) in zip(axs, rows):
        im = Image.open(HERE / f"render_{n}.png").crop((0, 572, 1650, 955))
        ax.imshow(im)
        ax.set_axis_off()
        ax.set_title(lab, fontsize=15, color=col, loc="left", pad=4, fontweight="bold")
    titled(fig, "Platen friction carries the strength: without it the prongs splay",
           "left: equivalent plastic strain (0-0.10); right: largest principal stress (MPa); deformed at true "
           "scale; CalculiX J2 + NLGEOM, 41k-node P2 mesh, dogbone material (teammates' renders, ccx_ramp/)")
    fig.subplots_adjust(left=0.02, right=0.98, top=0.78, bottom=0.01, hspace=0.14)
    save(fig, "fig6_platen_renders")


# --------------------------------------------------------------------------- outputs

def write_tables(D):
    slim = json.loads(json.dumps(D))
    (HERE / "act3_data.json").write_text(json.dumps(slim, indent=1))
    with open(HERE / "act3_errors.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["figure", "case", "nodes", "mean_rel_err", "err_0.5", "err_1.0", "err_1.5", "err_2.0", "err_2.5"])
        for v, r in D["bc_variants"].items():
            w.writerow(["fig2", v, "", f"{r['mean_rel_err']:.5f}", *r["rel_err"]])
        for s in D["ladder"]:
            w.writerow(["fig3", s["step"].replace("\n", " "), "", f"{s['mean_rel_err']:.5f}"])
        for m, r in D["meshes"].items():
            w.writerow(["fig4/fig5", m, r["nodes"], f"{r['mean_rel_err']:.5f}", *r["rel_err"]])


def main():
    style()
    D = compute()
    write_tables(D)
    fig1_test(D)
    fig2_bc(D)
    fig3_ladder(D)
    fig4_error_vs_nodes(D)
    fig5_curves(D)
    fig6_renders(D)
    import sys
    sys.path.insert(0, str(HERE))
    import plot_finetune_run as pf       # the marked slot for the (not yet run) fine-tuning
    pf.fig_progress(D, [], HERE, stem="fig7_finetune_progress_STUB")
    print("wrote fig7_finetune_progress_STUB")
    print(json.dumps({k: D[k] for k in ("refit_on_h20", "frictionless_best_mapping", "ladder_on_2mm_mesh",
                                        "ladder_other_orders", "bonded_equals_h4")}, indent=1))
    print("ladder", [(s["step"].replace("\n", " "), round(s["mean_rel_err"] * 100, 2)) for s in D["ladder"]])
    print("bc", {k: round(v["mean_rel_err"] * 100, 2) for k, v in D["bc_variants"].items()})
    print("mesh", {k: (v["nodes"], round(v["mean_rel_err"] * 100, 2), round(v["mean_rel_err_own_fit"] * 100, 2))
                   for k, v in D["meshes"].items()})
    print("committed", D["committed_baselines"])
    print("test", {k: v for k, v in D["test"].items()})


if __name__ == "__main__":
    main()
