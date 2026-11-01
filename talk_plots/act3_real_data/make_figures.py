#!/usr/bin/env python3
"""Act 3 of the talk: post-training the swarm on real lab data (abc_00000065 crushed on an Instron).

Regenerates every figure of talk_plots/act3_real_data/ (PNG 200 dpi + PDF + .md caption), act3_data.json (every
plotted number) and act3_errors.csv. Read-only on its inputs; no solver and no API calls.

    .venv/bin/python talk_plots/act3_real_data/make_figures.py

Inputs (all local, CalculiX J2 + NLGEOM):
  physical_tests/abc_00000065_compression/cload_compression.csv   measured curve
  physical_tests/abc_00000065_compression/nonlinear_spec.json     frozen setup: checkpoints, delta, compliance
  talk_plots/act3_real_data/videos/before_run/{summary.json,curve.csv}
        "before post-training": frictionless platens, automesher mesh, no seating gap or compliance
  results/real_data_finetune/baselines/baselines.json and <auto|swarm_bc0|h4|h25|h20>/curve.csv
        learned test conditions (sticking platens + seating gap + compliance) on the reference meshes
  ace_runs/nl_ft/gen*/runs/*.json (+ gen*/work/<pid>-a<k>/nonlinear/curve.csv)
        the post-training run: 10 generations x 4 agents on the real test
  physical_tests/abc_00000065_compression/ccx_ramp/render_team_{slide,stick}.png   teammates' renders

Error ("X % off the real test", = ace.nonlinear.run_curve): mean |F_sim - F_test| / F_test over the spec's
checkpoints on the test's loading branch. The checkpoints are not drawn in any figure (talk choice).
"""
from __future__ import annotations

import csv
import json
import shutil
import textwrap
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap, BoundaryNorm, LinearSegmentedColormap  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PID = "abc_00000065_compression"
PT = ROOT / "physical_tests" / PID
TEST_CSV = PT / "cload_compression.csv"
SPEC = PT / "nonlinear_spec.json"
BEFORE = HERE / "videos" / "before_run"
BASE = ROOT / "results" / "real_data_finetune" / "baselines"
RUN = ROOT / "ace_runs" / "nl_ft"

# fixed talk colours
C_AUTO, C_TEST, C_UNIF, C_SWARM = "#888888", "#000000", "#6b6b6b", "#c23d6e"
C_BEFORE, C_STICK = "#7a5195", "#1f8a8a"
GEN_CMAP = LinearSegmentedColormap.from_list("gen", ["#f2b8cc", "#c23d6e", "#5a1230"])

BASE_LABEL = {"auto": "automesher (gmsh default)", "swarm_bc0": "pre-trained swarm mesh",
              "h4": "uniform 4 mm", "h25": "uniform 2.5 mm", "h20": "uniform 2 mm"}


# --------------------------------------------------------------------------- style and output

def style():
    plt.rcParams.update({
        "figure.figsize": (12, 6.75), "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "font.size": 15, "axes.labelsize": 16, "xtick.labelsize": 14,
        "ytick.labelsize": 14, "legend.fontsize": 14, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": "#e6e6e6", "grid.linewidth": 0.9,
        "axes.axisbelow": True, "legend.frameon": False,
    })


def titled(fig, title: str, subtitle: str):
    sub = textwrap.fill(subtitle, 104)
    fig.suptitle(title, x=0.04, y=0.975, ha="left", fontsize=19, fontweight="bold")
    fig.text(0.04, 0.915, sub, ha="left", va="top", fontsize=14, color="#555555")
    top = 0.86 if "\n" not in sub else 0.82
    fig.subplots_adjust(left=0.08, right=0.97, top=top, bottom=0.11)


def save(fig, stem: str, title: str, md: str):
    fig.savefig(HERE / f"{stem}.png", dpi=200)
    fig.savefig(HERE / f"{stem}.pdf")
    plt.close(fig)
    (HERE / f"{stem}.md").write_text(f'# {stem}: "{title}"\n\n{md.strip()}\n')
    print("wrote", stem, "|", title)


def pct(e: float, nd: int = 1) -> str:
    return f"{e * 100:.{nd}f} %"


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


def read_cross_curve(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """(u_cross mm, F N) from a curve.csv with columns u_part_mm, u_cross_mm, force_N."""
    rows = list(csv.DictReader(path.read_text().splitlines()))
    return np.array([float(r["u_cross_mm"]) for r in rows]), np.array([float(r["force_N"]) for r in rows])


def curve_error(u, F, ck, Fm):
    """(mean rel. error, per-checkpoint errors, simulated forces) of a crosshead curve, as ace.nonlinear."""
    Fs = np.interp(ck, u, F)
    rel = np.where(ck <= u[-1], np.abs(Fs - Fm) / Fm, 1.0)
    return float(rel.mean()), rel, Fs


def case(label, nodes, u, F, ck, Fm, recorded, src, **extra):
    e, rel, Fs = curve_error(u, F, ck, Fm)
    assert abs(e - recorded) < 2e-4, (label, e, recorded)      # recomputed from the curve = the recorded value
    return {"label": label, "nodes": int(nodes), "mean_rel_err": recorded, "rel_err": rel.round(5).tolist(),
            "force_sim_N": Fs.round(1).tolist(), "u_cross_mm": u.round(5).tolist(), "F_N": F.round(2).tolist(),
            "file": src, **extra}


def load_runs(ck, Fm) -> list[dict]:
    runs = []
    for f in sorted(RUN.glob("gen*/runs/*.json")):
        rec = json.loads(f.read_text())
        if rec.get("problem_id") != PID:
            continue
        c = rec["checks"]
        nl = c.get("nonlinear") or {}
        agent = rec["run_id"].rsplit("-", 1)[1]
        r = {"run_id": rec["run_id"], "generation": int(rec["generation"]), "agent": agent,
             "ok": bool(c.get("valid") and nl.get("ok")), "base_score": c.get("base_score"),
             "mean_rel_err": nl.get("mean_rel_err"), "nodes": nl.get("nodes"), "dofs": nl.get("dofs"),
             "file": str(f.relative_to(ROOT))}
        cv = f.parent.parent / "work" / f"{PID}-{agent}" / "nonlinear" / "curve.csv"
        r["curve_csv"] = str(cv.relative_to(ROOT)) if cv.is_file() else None
        runs.append(r)
    return runs


def compute() -> dict:
    spec = json.loads(SPEC.read_text())
    ck = np.array(spec["checkpoints_mm"], float)
    u, F = read_test()
    ub, Fb = loading_branch(u, F)
    Fm = np.interp(ck, ub, Fb)

    lin = (Fb >= 1050) & (Fb <= 1800)
    toe = (ub >= 0.1) & (ub <= 0.45)
    p = int(np.argmax(F))
    test = {"checkpoints_mm": ck.tolist(), "force_meas_N": Fm.round(2).tolist(),
            "peak_N": float(F[p]), "peak_mm": float(u[p]), "break_mm": float(u[-1]),
            "F_at_2p5mm_N": float(np.interp(2.5, ub, Fb)),
            "slope_linear_N_per_mm": float(np.polyfit(ub[lin], Fb[lin], 1)[0]), "linear_band_N": [1050, 1800],
            "slope_toe_N_per_mm": float(np.polyfit(ub[toe], Fb[toe], 1)[0]), "toe_band_mm": [0.1, 0.45],
            "n_points": int(len(u)), "file": str(TEST_CSV.relative_to(ROOT))}

    bs = json.loads((BEFORE / "summary.json").read_text())
    before = case("before post-training", bs["nodes"], *read_cross_curve(BEFORE / "curve.csv"), ck, Fm,
                  bs["mean_rel_err"], str((BEFORE / "curve.csv").relative_to(ROOT)), dofs=bs["dofs"])

    bj = json.loads((BASE / "baselines.json").read_text())["baselines"]
    base = {k: case(BASE_LABEL[k], v["nodes"], *read_cross_curve(BASE / k / "curve.csv"), ck, Fm,
                    v["mean_rel_err"], str((BASE / k / "curve.csv").relative_to(ROOT)), dofs=v["dofs"],
                    base_score=v["score"])
            for k, v in bj.items() if k in BASE_LABEL}

    runs = load_runs(ck, Fm)
    ok = [r for r in runs if r["ok"]]
    gens = []
    for g in sorted({r["generation"] for r in runs}):
        rs = [r for r in ok if r["generation"] == g]
        bsc = max(rs, key=lambda r: r["base_score"])
        acc = min(rs, key=lambda r: r["mean_rel_err"])
        gens.append({"generation": g, "n_runs": sum(r["generation"] == g for r in runs), "n_ok": len(rs),
                     "best_base_score": bsc["base_score"], "best_score_run": bsc["run_id"],
                     "mean_base_score": float(np.mean([r["base_score"] for r in rs])),
                     "best_err": acc["mean_rel_err"], "best_err_run": acc["run_id"], "best_err_nodes": acc["nodes"]})
    most_acc = min(ok, key=lambda r: (r["mean_rel_err"], r["nodes"]))
    best_score = max(ok, key=lambda r: r["base_score"])
    lean = min((r for r in ok if r["nodes"] < 6000), key=lambda r: r["mean_rel_err"])
    picks = {}
    for name, r in (("most_accurate", most_acc), ("best_scoring", best_score), ("best_lean", lean)):
        uc, Fc = read_cross_curve(ROOT / r["curve_csv"])
        picks[name] = case(f"post-trained ({name.replace('_', ' ')}, {r['run_id']})", r["nodes"], uc, Fc, ck, Fm,
                           r["mean_rel_err"], r["curve_csv"], run_id=r["run_id"], base_score=r["base_score"],
                           same_mesh_runs=[q["run_id"] for q in ok if q["nodes"] == r["nodes"]
                                           and abs(q["mean_rel_err"] - r["mean_rel_err"]) < 1e-6])

    return {"spec": {"file": str(SPEC.relative_to(ROOT)), "delta_mm": spec["delta_mm"],
                     "compliance_mm_per_N": spec["compliance_mm_per_N"], "checkpoints_mm": ck.tolist()},
            "auto_score": bj["auto"]["score"], "test": test, "before": before, "baselines": base,
            "post_training": {"run_dir": str(RUN.relative_to(ROOT)), "generations": gens, "picks": picks,
                              "runs": runs}}


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
    ax.annotate(f"peak {t['peak_N']/1e3:.2f} kN at {t['peak_mm']:.2f} mm: first crack",
                (t["peak_mm"], t["peak_N"] / 1e3), (3.1, 5.45), **kw)
    ax.annotate("cracking: saw-tooth load drops", (5.4, 3.2), (5.2, 4.55), **kw)
    ax.annotate(f"break at {t['break_mm']:.2f} mm", (t["break_mm"], 1.7), (6.6, 0.9), **kw)
    ax.text(0.08, 5.6, "compared with the simulation", fontsize=14, color="#555555")
    title = "Real test: seating, linear climb, yielding to 4.8 kN, then cracking"
    titled(fig, title, "3D-printed abc_00000065 (Formlabs Alpha Fast resin) crushed between Instron platens; "
                       "force vs crosshead travel, digitized from the machine's PDF report")
    md = f"""
**Caption.** The 3D-printed abc_00000065 C-bracket (Formlabs Alpha Fast) crushed between Instron platens: a soft
seating toe, a straight climb, softening as the resin yields, a peak (the first crack), then saw-tooth load drops
to the break. The shaded loading branch, up to the peak, is what every simulation is compared with.

**Key numbers** (computed in make_figures.py from the test CSV):
- peak {t['peak_N']:.0f} N at {t['peak_mm']:.3f} mm crosshead travel in this digitization (the machine report says
  4.80 kN at 2.56 mm)
- toe slope {t['slope_toe_N_per_mm']/1e3:.2f} kN/mm (fit over 0.10-0.45 mm); linear slope
  {t['slope_linear_N_per_mm']/1e3:.2f} kN/mm (fit over 1050-1800 N)
- force at 2.5 mm: {t['F_at_2p5mm_N']:.0f} N; last point (break) at {t['break_mm']:.2f} mm; {t['n_points']} digitized points

**Data.** {t['file']} (the frozen spec's test file); test notes cload_compression.json next to it.

**Caveats.**
- The curve is digitized from the plot in the machine's PDF report (about +-0.01 mm, +-15 N), not raw data. A
  second digitization put the peak about 1 % higher (4798 N at 2.556 mm); the error uses this file.
- Displacement is crosshead travel (no extensometer): it includes seating and the load train's deflection.
- "Seating, one prong first?" for the toe is an interpretation from the test notes, not a measurement.
"""
    save(fig, "fig1_measured_curve", title, md)


def fig2_before(D):
    t, b = D["test"], D["before"]
    ub, Fb = loading_branch(*read_test())
    fig, ax = plt.subplots()
    ax.plot(ub, Fb / 1e3, color=C_TEST, lw=3.6, zorder=5)
    ax.plot(b["u_cross_mm"], np.array(b["F_N"]) / 1e3, color=C_BEFORE, lw=3.4, zorder=4)
    Fb25 = float(np.interp(2.5, b["u_cross_mm"], b["F_N"]))
    ax.text(2.58, t["F_at_2p5mm_N"] / 1e3, f"real test: {t['F_at_2p5mm_N']/1e3:.1f} kN", fontsize=15,
            fontweight="bold", va="center")
    ax.text(2.66, Fb25 / 1e3, f"before post-training: {Fb25/1e3:.1f} kN\n{pct(b['mean_rel_err'], 0)} off the "
                              "real test", fontsize=15, color=C_BEFORE, fontweight="bold", va="center")
    ax.annotate("", (2.5, Fb25 / 1e3 + 0.08), (2.5, t["F_at_2p5mm_N"] / 1e3 - 0.08),
                arrowprops=dict(arrowstyle="<->", color="#777777", lw=1.4))
    ax.set_xlim(0, 4.3), ax.set_ylim(0, 5.6)
    ax.set_xticks([0, 0.5, 1, 1.5, 2, 2.5])
    ax.set_xlabel("crosshead travel (mm)")
    ax.set_ylabel("force (kN)")
    title = f"Before post-training, the simulation is {pct(b['mean_rel_err'], 0)} off the real test"
    titled(fig, title, f"frictionless platens, automesher mesh ({b['nodes']:,} nodes), no seating gap or "
                       "machine compliance; CalculiX J2 plasticity + large deformation, dogbone material")
    over = np.array(b["force_sim_N"]) / np.array(t["force_meas_N"]) - 1
    md = f"""
**Caption.** The out-of-the-box nonlinear simulation, before post-training: frictionless platens on gmsh's default
mesh, crosshead travel = platen travel. It starts too stiff (no seating gap) and then gives out: the prongs splay
and the force levels off at {b['F_N'][-1]/1e3:.1f} kN, where the real part carried {t['F_at_2p5mm_N']/1e3:.1f} kN
at 2.5 mm. Mean error: **{pct(b['mean_rel_err'])} off the real test**.

**Key numbers.** {b['nodes']:,} nodes ({b['dofs']:,} DOFs). Force vs test at 0.5 / 1.0 / 1.5 / 2.0 / 2.5 mm:
{' / '.join(f'{x:.0f}' for x in b['force_sim_N'])} N vs {' / '.join(f'{x:.0f}' for x in t['force_meas_N'])} N
({' / '.join(f'{x*100:+.0f}' for x in over)} %).

**Data.** {b['file']} and summary.json next to it (CalculiX J2 + NLGEOM, the frozen ramp with frictionless
platens; error recomputed from the curve = the recorded {b['mean_rel_err']:.4f}).

**Caveats.** "Before post-training" means before the swarm was post-trained on this test: the solver and the
swarm were already capable. This run also lacks the test conditions learned from the experiment (sticking
platens, seating gap, load-train compliance); fig3 separates the two steps.
"""
    save(fig, "fig2_raw_solver_vs_test", title, md)


def fig3_ladder(D):
    B, P = D["baselines"], D["post_training"]["picks"]["most_accurate"]
    steps = [("before\npost-training", D["before"], C_BEFORE),
             ("+ learned test\nconditions\n(default mesh)", B["auto"], C_AUTO),
             ("+ pre-trained\nswarm mesh", B["swarm_bc0"], "#e09ab4"),
             ("+ post-trained\nswarm mesh", P, C_SWARM)]
    fig, ax = plt.subplots()
    x = np.arange(len(steps))
    e = np.array([s[1]["mean_rel_err"] * 100 for s in steps])
    ax.bar(x, e, width=0.6, color=[s[2] for s in steps], zorder=3)
    for xi, ei, s in zip(x, e, steps):
        ax.text(xi, ei + 0.5, f"{ei:.1f} %" if ei < 10 else f"{ei:.0f} %", ha="center", va="bottom", fontsize=18,
                fontweight="bold")
        ax.text(xi, -1.0, f"{s[1]['nodes']:,} nodes", ha="center", va="top", fontsize=14, color="#555555")
    ax.set_xticks(x, [s[0] for s in steps], fontsize=15)
    ax.tick_params(axis="x", pad=26, length=0)
    ax.set_ylabel("off the real test (%)")
    ax.set_ylim(0, max(e) * 1.15)
    ax.grid(axis="x", visible=False)
    title = f"From {e[0]:.0f} % to {e[-1]:.1f} % off the real test: test conditions, then post-training"
    titled(fig, title, "same solver and material throughout; learned test conditions = sticking platens, seating "
                       "gap and load-train compliance, frozen; then only the mesh changes")
    fig.subplots_adjust(bottom=0.2)
    md = f"""
**Caption.** Each step on top of the previous one. Before post-training the simulation is {e[0]:.0f} % off the
real test. Learning the test conditions from the experiment (sticking platens, a seating gap, load-train
compliance; then frozen) brings gmsh's default mesh to {e[1]:.1f} %. The swarm's pre-trained mesh (trained on a
different load case) gets {e[2]:.1f} %, and the mesh from post-training on the real test {e[3]:.1f} %.

| step | mesh | nodes | off the real test |
|---|---|---|---|
| before post-training (frictionless platens, no seating/compliance) | automesher | {steps[0][1]['nodes']:,} | {e[0]:.2f} % |
| + learned test conditions | automesher (curved P2) | {steps[1][1]['nodes']:,} | {e[1]:.2f} % |
| + pre-trained swarm mesh | swarm bc0 mesh | {steps[2][1]['nodes']:,} | {e[2]:.2f} % |
| + post-trained swarm mesh (most accurate, {P['run_id']}) | post-trained | {P['nodes']:,} | {e[3]:.2f} % |

**Data.** {D['before']['file']}; results/real_data_finetune/baselines/baselines.json (auto, swarm_bc0);
{P['file']}. Every error is recomputed from its curve and matches the recorded value.

**Caveats.**
- The automesher's 7.8 % uses its curved P2 mesh with only the midside nodes CalculiX rejects straightened. An
  earlier, higher automesher figure straightened every midside node (facetting the curved faces) and was wrong.
- The seating gap and compliance were fitted once on the finest uniform mesh against this same test, so the last
  steps are calibrated on the specimen they are scored against; there is one specimen.
- The last bar is the most accurate post-trained mesh, which uses more nodes than the others; the best-scoring
  one is {pct(D['post_training']['picks']['best_scoring']['mean_rel_err'])} off at
  {D['post_training']['picks']['best_scoring']['nodes']:,} nodes (fig4).
- Uniform 4 mm (9,506 nodes) is {pct(D['baselines']['h4']['mean_rel_err'])} off: about as accurate as the most
  accurate post-trained mesh with 8 % more nodes.
"""
    save(fig, "fig3_setup_ladder", title, md)


def fig4_error_vs_nodes(D):
    B, PT_ = D["baselines"], D["post_training"]
    ok = [r for r in PT_["runs"] if r["ok"]]
    ng = len(PT_["generations"])
    fig, ax = plt.subplots()
    cmap = ListedColormap(GEN_CMAP(np.linspace(0, 1, ng)))
    norm = BoundaryNorm(np.arange(-0.5, ng), ng)
    sc = ax.scatter([r["nodes"] for r in ok], [r["mean_rel_err"] * 100 for r in ok],
                    c=[r["generation"] for r in ok], cmap=cmap, norm=norm, s=70, edgecolor="white", lw=0.6,
                    zorder=4)
    cb = fig.colorbar(sc, ax=ax, ticks=range(ng), pad=0.015, fraction=0.03)
    cb.set_label("post-training generation", fontsize=14)
    cb.ax.tick_params(labelsize=14)
    uni = [B[k] for k in ("h4", "h25", "h20")]
    ax.plot([m["nodes"] for m in uni], [m["mean_rel_err"] * 100 for m in uni], "-o", color=C_UNIF, lw=2, ms=9,
            zorder=3)
    for m, off, ha in zip(uni, [(10, 10), (0, 12), (0, 12)], ["left", "center", "center"]):
        ax.annotate(f"{m['label']}\n{m['nodes']:,}: {pct(m['mean_rel_err'])}", (m["nodes"], m["mean_rel_err"] * 100),
                    textcoords="offset points", xytext=off, ha=ha, va="bottom", fontsize=14, color=C_UNIF)
    a, s = B["auto"], B["swarm_bc0"]
    ax.plot(a["nodes"], a["mean_rel_err"] * 100, "s", color=C_AUTO, ms=13, zorder=4)
    ax.annotate(f"automesher\n{a['nodes']:,} nodes: {pct(a['mean_rel_err'])}", (a["nodes"], a["mean_rel_err"] * 100),
                textcoords="offset points", xytext=(14, 0), fontsize=14, va="center", color="#444444")
    ax.plot(s["nodes"], s["mean_rel_err"] * 100, "D", mfc="white", mec=C_SWARM, mew=2.5, ms=13, zorder=4)
    ax.annotate(f"pre-trained swarm mesh\n{s['nodes']:,} nodes: {pct(s['mean_rel_err'])}",
                (s["nodes"], s["mean_rel_err"] * 100), textcoords="offset points", xytext=(14, 4), fontsize=14,
                color=C_SWARM, va="bottom")
    ma, bs = PT_["picks"]["most_accurate"], PT_["picks"]["best_scoring"]
    for p, txt, off, ha in ((ma, "most accurate", (6, -40), "left"), (bs, "best-scoring", (-10, -40), "right")):
        ax.plot(p["nodes"], p["mean_rel_err"] * 100, "o", mfc="none", mec="black", mew=1.8, ms=15, zorder=5)
        ax.annotate(f"{txt}\n{p['nodes']:,} nodes: {pct(p['mean_rel_err'])}", (p["nodes"], p["mean_rel_err"] * 100),
                    textcoords="offset points", xytext=off, ha=ha, va="top", fontsize=14, fontweight="bold",
                    color=C_SWARM, arrowprops=dict(arrowstyle="-", color="#777777", lw=1))
    ax.set_xscale("log")
    ax.set_xlim(2000, 60000)
    ax.set_ylim(0, 9)
    ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator([2000, 5000, 10000, 20000, 50000]))
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("mesh nodes (log scale)")
    ax.set_ylabel("off the real test (%)")
    title = (f"Post-trained meshes: {ma['mean_rel_err']*100:.1f}-{bs['mean_rel_err']*100:.1f} % off the real "
             f"test, automesher {pct(a['mean_rel_err'])}")
    titled(fig, title, f"dots: all {len(ok)} post-training runs (10 generations x 4 agents), coloured by "
                       "generation; same learned test conditions for every mesh; all P2")
    fig.subplots_adjust(right=0.94)
    last = [r for r in ok if r["generation"] == ng - 1]
    lean = PT_["picks"]["best_lean"]
    rows = "\n".join(f"| {r['label']} | {r['nodes']:,} | {pct(r['mean_rel_err'], 2)} |"
                     for r in (a, s, B["h4"], B["h25"], B["h20"]))
    md = f"""
**Caption.** With the learned test conditions frozen, the mesh decides how far the simulation is off the real
test. gmsh's default mesh is {pct(a['mean_rel_err'])} off and the swarm's pre-trained mesh
{pct(s['mean_rel_err'])}. Post-training on the real test moves the swarm's meshes to about 2 %: the most accurate
is {pct(ma['mean_rel_err'], 2)} off at {ma['nodes']:,} nodes ({ma['run_id']}; the same mesh in
{', '.join(ma['same_mesh_runs'])}), and the best-scoring {pct(bs['mean_rel_err'], 2)} at {bs['nodes']:,} nodes
({bs['run_id']}). The last generation's four meshes are all
{min(r['mean_rel_err'] for r in last)*100:.1f}-{max(r['mean_rel_err'] for r in last)*100:.1f} % off at {min(r['nodes'] for r in last):,}-{max(r['nodes'] for r in last):,} nodes.

| reference mesh | nodes | off the real test |
|---|---|---|
{rows}
| post-trained, most accurate ({ma['run_id']}) | {ma['nodes']:,} | {pct(ma['mean_rel_err'], 2)} |
| post-trained, best lean (< 6k nodes, {lean['run_id']}) | {lean['nodes']:,} | {pct(lean['mean_rel_err'], 2)} |
| post-trained, best-scoring ({bs['run_id']}) | {bs['nodes']:,} | {pct(bs['mean_rel_err'], 2)} |

The most accurate post-trained mesh is {a['mean_rel_err']/ma['mean_rel_err']:.1f}x more accurate than the
automesher; the best-scoring one {a['mean_rel_err']/bs['mean_rel_err']:.1f}x at {bs['nodes']/a['nodes']:.1f}x its nodes.

**Data.** {PT_['run_dir']}/gen*/runs/*.json (checks.nonlinear: mean_rel_err, nodes);
results/real_data_finetune/baselines/baselines.json and */curve.csv.

**Caveats.**
- Uniform 4 mm ({B['h4']['nodes']:,} nodes, {pct(B['h4']['mean_rel_err'], 2)}) is as accurate as the most accurate
  post-trained mesh with 8 % more nodes. The lean post-trained meshes are slightly less accurate (2.2-2.3 %) with
  37-41 % fewer nodes (5.6-6.0k vs 9.5k): the gain over uniform refinement is in nodes, not accuracy. The finer
  uniform meshes are more accurate (1.1-1.2 %) at 2.8-4.7x the nodes of the most accurate post-trained mesh.
- Below about 1 % the remaining difference is model error (material, conditions), not mesh error; the seating gap
  and compliance were fitted on this specimen.
- Several agents resubmitted the same mesh, so some dots overlap.
"""
    save(fig, "fig4_error_vs_nodes", title, md)


def fig5_curves(D):
    t, a, p = D["test"], D["baselines"]["auto"], D["post_training"]["picks"]["most_accurate"]
    ub, Fb = loading_branch(*read_test())
    xmax = 2.75
    fig, ax = plt.subplots()
    ax.plot(ub, Fb / 1e3, color=C_TEST, lw=3.8, zorder=5)
    ends = []
    for r, kw, lab, tc in ((a, dict(color=C_AUTO, lw=3.2, ls="--"),
                            f"automesher, {a['nodes']:,} nodes:\n{pct(a['mean_rel_err'])} off", "#555555"),
                           (p, dict(color=C_SWARM, lw=3.4),
                            f"post-trained, {p['nodes']:,} nodes:\n{pct(p['mean_rel_err'])} off", C_SWARM)):
        uc, F = np.array(r["u_cross_mm"]), np.array(r["F_N"]) / 1e3
        keep = uc <= xmax
        Fe = float(np.interp(xmax, uc, F))
        ax.plot(np.append(uc[keep], xmax), np.append(F[keep], Fe), **kw, zorder=4)
        ends.append((Fe, lab, tc))
    ax.text(xmax + 0.06, ends[0][0] + 0.15, ends[0][1], fontsize=15, va="center", color=ends[0][2])
    ax.text(xmax + 0.06, ends[1][0] - 0.45, ends[1][1], fontsize=15, va="center", color=ends[1][2],
            fontweight="bold")
    ax.text(1.75, 3.1, "real test", fontsize=16, fontweight="bold")
    ax.set_xlim(0, xmax + 1.5), ax.set_ylim(0, 6.0)
    ax.set_xticks([0, 0.5, 1, 1.5, 2, 2.5])
    ax.set_xlabel("crosshead travel (mm)")
    ax.set_ylabel("force (kN)")
    title = (f"Post-trained mesh tracks the real test: {pct(p['mean_rel_err'])} off, automesher "
             f"{pct(a['mean_rel_err'])}")
    titled(fig, title, "same learned test conditions, solver and material; only the mesh differs. Post-trained: "
                       "the run's most accurate mesh")
    over_a = np.array(a["force_sim_N"]) / np.array(t["force_meas_N"]) - 1
    over_p = np.array(p["force_sim_N"]) / np.array(t["force_meas_N"]) - 1
    md = f"""
**Caption.** Same conditions, only the mesh differs. gmsh's default mesh is too stiff: its curve runs
{over_a.min()*100:.0f}-{over_a.max()*100:.0f} % above the real test ({pct(a['mean_rel_err'])} off on average). The
most accurate mesh from post-training ({p['run_id']}, {p['nodes']:,} nodes) follows the measured curve,
{pct(p['mean_rel_err'])} off.

**Key numbers** (force at 0.5 / 1.0 / 1.5 / 2.0 / 2.5 mm crosshead):
- real test: {' / '.join(f'{x:.0f}' for x in t['force_meas_N'])} N
- automesher: {' / '.join(f'{x:.0f}' for x in a['force_sim_N'])} N ({' / '.join(f'{x*100:+.1f}' for x in over_a)} %)
- post-trained: {' / '.join(f'{x:.0f}' for x in p['force_sim_N'])} N ({' / '.join(f'{x*100:+.1f}' for x in over_p)} %)

**Data.** {a['file']}; {p['file']}.

**Caveats.** The largest remaining gap is at 1.0 mm ({over_p[1]*100:+.1f} %), inside the seating toe, which the
simple seating-gap model only approximates; every mesh shares it (uniform 2 mm: +3.9 %). This mesh is the one used
for the Act 3 crack and video pages.
"""
    save(fig, "fig5_curves_vs_test", title, md)


def fig6_renders(D):
    from PIL import Image
    src = PT / "ccx_ramp"
    summ = json.loads((src / "summary.json").read_text())["runs"]
    for n in ("team_slide", "team_stick"):
        shutil.copyfile(src / f"render_{n}.png", HERE / f"render_{n}.png")
    fig, axs = plt.subplots(2, 1)
    sl, st = summ["team_slide"], summ["team_stick"]
    rows = [("team_slide", f"frictionless platens: prongs splay, force tops out at {sl['F_max_N']/1e3:.2f} kN",
             C_BEFORE),
            ("team_stick", f"sticking platens: still rising, {st['F_at_test_peak_travel_N']/1e3:.2f} kN at the "
                           "test's peak travel", C_STICK)]
    for ax, (n, lab, col) in zip(axs, rows):
        im = Image.open(HERE / f"render_{n}.png").crop((0, 572, 1650, 955))
        ax.imshow(im)
        ax.set_axis_off()
        ax.set_title(lab, fontsize=15, color=col, loc="left", pad=4, fontweight="bold")
    title = "Platen friction carries the strength: without it the prongs splay"
    titled(fig, title, "left: equivalent plastic strain (0-0.10); right: largest principal stress (MPa); deformed "
                       "at true scale; CalculiX J2 + NLGEOM, 41k-node P2 mesh, dogbone material (teammates' renders)")
    fig.subplots_adjust(left=0.02, right=0.98, top=0.78, bottom=0.01, hspace=0.14)
    md = f"""
**Caption.** The teammates' CalculiX renders on a 41k-node mesh: with frictionless platens the prongs splay and
yield at the nut pockets, and the force tops out at {sl['F_max_N']/1e3:.2f} kN; with sticking platens the part
keeps carrying load ({st['F_at_test_peak_travel_N']/1e3:.2f} kN at the test's peak travel). This is why the
learned test conditions use sticking platens.

**Key numbers** (physical_tests/abc_00000065_compression/ccx_ramp/summary.json, dogbone material): team_slide
F_max {sl['F_max_N']:.0f} N; team_stick F at the test's peak travel {st['F_at_test_peak_travel_N']:.0f} N, F_max
{st['F_max_N']:.0f} N at {st['u_at_F_max_mm']:.1f} mm.

**Data.** physical_tests/abc_00000065_compression/ccx_ramp/render_team_slide.png and render_team_stick.png
(copied here unchanged; the figure crops their middle rows), summary.json.

**Caveats.**
- These are the teammates' ramps (scripts/ccx_compression_ramp.py), not the frozen setup: the base is held, there
  is no seating/compliance mapping, and the travel is platen travel.
- The panel subtitles inside the images are the renderer's own labels (small at slide size).
"""
    save(fig, "fig6_platen_renders", title, md)


def fig7_progress(D):
    G, B = D["post_training"]["generations"], D["baselines"]
    s0 = D["auto_score"]
    x = [g["generation"] for g in G]
    best = np.array([g["best_base_score"] for g in G]) / s0
    mean = np.array([g["mean_base_score"] for g in G]) / s0
    err = np.array([g["best_err"] for g in G]) * 100
    fig, (a1, a2) = plt.subplots(1, 2, gridspec_kw={"wspace": 0.42})
    a1.plot(x, best, "-o", color=C_SWARM, lw=3, ms=9, zorder=4)
    a1.plot(x, mean, "--s", color="#e09ab4", lw=2.5, ms=7, zorder=3)
    a1.text(2.6, 3.2, "best agent", ha="right", va="bottom", fontsize=14, color=C_SWARM, fontweight="bold")
    a1.text(4.2, 2.95, "average agent", ha="left", va="top", fontsize=14, color="#c2638a")
    refs = [(B["h4"]["base_score"] / s0, "uniform 4 mm", C_UNIF, ":"),
            (B["swarm_bc0"]["base_score"] / s0, "pre-trained swarm mesh", C_SWARM, ":"),
            (1.0, "automesher", C_AUTO, "--")]
    for y, lab, col, ls in refs:
        a1.axhline(y, color=col, ls=ls, lw=1.8, zorder=2)
        a1.text(9.3, y + 0.05, f"{lab} {y:.2f}x", fontsize=14, color=col, va="bottom", ha="right")
    a1.set_ylim(0, 4.0)
    a1.set_ylabel("score, x the automesher")
    a1.set_title("score: accuracy and node cost", fontsize=15, loc="left")
    a2.plot(x, err, "-o", color=C_SWARM, lw=3, ms=9, zorder=4)
    for xi, ei, g in zip(x, err, G):
        a2.text(xi, ei + (0.2 if xi % 2 == 0 else 0.75), f"{g['best_err_nodes']/1e3:.1f}k", ha="center", va="bottom", fontsize=14,
                color="#555555")
    refs2 = [(B["auto"]["mean_rel_err"] * 100, "automesher", C_AUTO, "--"),
             (B["swarm_bc0"]["mean_rel_err"] * 100, "pre-trained swarm mesh", C_SWARM, ":"),
             (B["h4"]["mean_rel_err"] * 100, "uniform 4 mm", C_UNIF, ":")]
    for y, lab, col, ls in refs2:
        a2.axhline(y, color=col, ls=ls, lw=1.8, zorder=2)
        below = lab == "uniform 4 mm"
        a2.text(9.3, y - 0.12 if below else y + 0.1, f"{lab} {y:.1f} %", fontsize=14, color=col,
                va="top" if below else "bottom", ha="right")
    a2.set_ylim(0, 9)
    a2.set_ylabel("off the real test (%)")
    a2.set_title("most accurate mesh (label: nodes)", fontsize=15, loc="left")
    for ax in (a1, a2):
        ax.set_xticks(x)
        ax.set_xlim(-0.4, x[-1] + 0.4)
        ax.set_xlabel("post-training generation")
    title = (f"Post-training: best score {best[0]:.1f}x → {best.max():.1f}x the automesher, average "
             f"{mean[0]:.1f}x → {mean[-1]:.1f}x")
    titled(fig, title, "4 agents per generation post-trained on the real test, learned test conditions frozen; "
                       "score = accuracy gain over the automesher, discounted by node cost (automesher = 1x)")
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.12, top=0.76)
    tab = "\n".join(f"| {g['generation']} | {g['best_base_score']:.3f} ({g['best_base_score']/s0:.2f}x, "
                    f"{g['best_score_run'].replace(PID + '-', '')}) | {g['mean_base_score']:.3f} "
                    f"({g['mean_base_score']/s0:.2f}x) | {g['best_err']*100:.2f} % at {g['best_err_nodes']:,} |"
                    for g in G)
    md = f"""
**Caption.** Post-training the swarm on the real test, 10 generations of 4 agents. Left: the best agent's score
rises from {best[0]:.2f}x to {best.max():.2f}x the automesher (peak in generation {int(np.argmax(best))}) and the
average agent's from {mean[0]:.2f}x to {mean[-1]:.2f}x: every agent, not just the best, ends well above the
uniform 4 mm mesh ({B['h4']['base_score']/s0:.2f}x) and the pre-trained swarm mesh
({B['swarm_bc0']['base_score']/s0:.2f}x). Right: the most accurate mesh is already at {err[0]:.2f} % in
generation 0; later generations keep the error near 2.2-2.6 % while cutting the nodes from 8.8k to about 6k,
which is what the score rewards.

**Score** (checks.base_score, field-mode formula against the automesher's run of the same frozen ramp): g =
e_auto / e, accuracy = (g/2)^2 below g = 2 and g/2 above, score = accuracy x automesher DOFs / own DOFs. The
automesher scores {s0:.2f} (= 1x here).

| gen | best base score | mean base score | most accurate mesh |
|---|---|---|---|
{tab}

**Data.** {D['post_training']['run_dir']}/gen*/runs/*.json (checks.base_score, checks.nonlinear.mean_rel_err and
nodes); references from results/real_data_finetune/baselines/baselines.json (score, mean_rel_err).

**Caveats.**
- One run, no repeats. The gain is mostly in node cost: the error of the most accurate mesh does not fall after
  generation 0-1 (1.97 %), it stays near 2.2-2.6 % with about 30 % fewer nodes.
- Generations 0 and 1 contain the same 8,753-node mesh (g0 a2, g1 a1, g1 a2).
"""
    save(fig, "fig7_post_training_progress", title, md)


# --------------------------------------------------------------------------- outputs

def write_tables(D):
    (HERE / "act3_data.json").write_text(json.dumps(D, indent=1))
    with open(HERE / "act3_errors.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case", "run_id", "generation", "nodes", "base_score", "mean_rel_err"])
        w.writerow(["before post-training", "", "", D["before"]["nodes"], "", f"{D['before']['mean_rel_err']:.5f}"])
        for k, r in D["baselines"].items():
            w.writerow([k, "", "", r["nodes"], f"{r['base_score']:.4f}", f"{r['mean_rel_err']:.5f}"])
        for r in D["post_training"]["runs"]:
            w.writerow(["post-training", r["run_id"], r["generation"], r["nodes"], f"{r['base_score']:.4f}",
                        f"{r['mean_rel_err']:.5f}"])


def main():
    style()
    D = compute()
    write_tables(D)
    fig1_test(D)
    fig2_before(D)
    fig3_ladder(D)
    fig4_error_vs_nodes(D)
    fig5_curves(D)
    fig6_renders(D)
    fig7_progress(D)
    for g in D["post_training"]["generations"]:
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in g.items()})
    for k, p in D["post_training"]["picks"].items():
        print(k, p["run_id"], p["nodes"], round(p["mean_rel_err"] * 100, 2), round(p["base_score"], 3))


if __name__ == "__main__":
    main()
