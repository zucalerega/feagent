"""Equal-budget gallery: agent meshes that beat a comparison mesh with NO MORE degrees of freedom.

Run gallery_scan.py first (setup check + pair search), then (~1 min, no API calls):

    .venv/bin/python talk_plots/equal_budget/make_gallery.py

Figures (each PNG 200 dpi + PDF + .md caption, in this folder):
  fig_gallery_le11_axisym   NAFEMS LE11 axisymmetric, ace_runs/deep1: the trained agent vs the
                            automesher refined to at least the agent's DOFs (error maps)
  fig_gallery_le1           NAFEMS LE1, ace_runs/split1: same comparison (error maps)
  fig_gallery_cook_within_run  Cook's membrane, ace_runs/cook_auto: a generation-1 agent, a
                            generation-7 agent of the same training run with fewer DOFs, and the
                            automesher refined to at least the generation-7 agent's DOFs
  fig_gallery_real_test     the Instron compression test: measured force curve vs the automesher's
                            and the swarm's bc0 mesh's simulated curves (local files only), plus a
                            slot for the fine-tuned mesh (pending: not local yet)

Error maps: each element's share of the scored field error (energy density of the stress error in the
element relative to the mean energy density of the true stress; its square integrates to field_error**2).
Dark = wrong. Every agent mesh is the run's own scored .msh with its own setup, re-solved with the fixed
skfem solver (ace/field_error.py _solve) and checked against the error and DOFs stored in the run file.
The refined automesher is gmsh with all default sizes times Mesh.MeshSizeFactor (as
talk_plots/act1_agentic_works/automesher_sweep.py), the factor chosen as the largest one whose mesh has
at least the agent's DOFs, solved with the reference setup.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import meshio  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402
from matplotlib.tri import Triangulation  # noqa: E402

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
DATA = OUT / "gallery_data"
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from ace import checkpoint as ckpt, fea  # noqa: E402
from ace import field_error as fe  # noqa: E402
from ace import nonlinear  # noqa: E402
from ace.checks import reference_setup  # noqa: E402
from ace.config import TASKS  # noqa: E402

# --------------------------------------------------------------------------- choices (from gallery_data/pairs.json)

LE11 = {"problem": "nafems_le11_axisym", "run_dir": "deep1", "agent": "g10-nafems_le11_axisym-a3"}
LE1 = {"problem": "nafems_le1", "run_dir": "split1", "agent": "g5-nafems_le1-a2"}
COOK = {"problem": "cooks_membrane", "run_dir": "cook_auto", "early": "g1-cooks_membrane-a1",
        "agent": "g7-cooks_membrane-a2"}
BASE = ROOT / "results/real_data_finetune/baselines"
FINETUNED_SUMMARY: Path | None = None   # set to the fine-tuned mesh's nonlinear summary.json when it is local

# style (talk-wide, as talk_plots/self_learning)
EDGE = "#26313b"
ERR_CMAP = "magma_r"                     # light = right, dark = wrong
ERR_RANGE = (0.0, 0.10)                  # linear 0-10 %, clipped above: the same scale as make_cook_figures.py
GREY, AGENT, EARLY, MEAS = "#888888", "#c23d6e", "#2a78d6", "black"
plt.rcParams.update({"font.size": 15, "axes.titlesize": 17, "axes.labelsize": 16, "xtick.labelsize": 14,
                     "ytick.labelsize": 14, "legend.fontsize": 14, "figure.facecolor": "white",
                     "axes.facecolor": "white", "savefig.facecolor": "white",
                     "axes.spines.top": False, "axes.spines.right": False})


def save(fig, stem: str, title: str, body: str) -> None:
    fig.savefig(OUT / f"{stem}.png", dpi=200)
    fig.savefig(OUT / f"{stem}.pdf")
    plt.close(fig)
    (OUT / f"{stem}.md").write_text(f"# {title}\n\n{body}\n")


def pct(x: float) -> str:
    return f"{100 * x:.2f} %"


# --------------------------------------------------------------------------- solving


def run_file(run_dir: str, run_id: str) -> Path:
    return next(Path("ace_runs", run_dir).glob(f"gen*/runs/{run_id}.json"))


def automesher_mesh(pid: str, factor: float, path: Path) -> dict:
    """gmsh default sizes x factor, quadratic: ace/field_error.py automesher with Mesh.MeshSizeFactor."""
    import gmsh
    from feacore.geometry.gmsh_tools import ensure_domain_group, gmsh_session
    p = fea.get_problem(pid)
    with gmsh_session(p.id):
        p.build_geometry()
        ensure_domain_group(p.dim)
        gmsh.option.setNumber("Mesh.MeshSizeFactor", factor)
        gmsh.model.mesh.generate(p.dim)
        gmsh.model.mesh.setOrder(2)
        gmsh.write(str(path))
    return {"path": str(path), "order": 2}


def automesher_at_least(pid: str, dofs: int) -> tuple[float, Path]:
    """The largest size factor (to 0.001) whose automesher mesh has >= dofs DOFs (2 per node in 2D)."""
    def n_dofs(f):
        path = DATA / f"{pid}_automesher_tmp.msh"
        automesher_mesh(pid, f, path)
        return 2 * len(meshio.read(path).points)
    lo, hi = 0.2, 1.0                          # n_dofs(lo) >= dofs > n_dofs(hi)
    assert n_dofs(lo) >= dofs > n_dofs(hi)
    while hi - lo > 1e-3:
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if n_dofs(mid) >= dofs else (lo, mid)
    path = DATA / f"{pid}_automesher_f{lo:.3f}.msh"
    automesher_mesh(pid, lo, path)
    (DATA / f"{pid}_automesher_tmp.msh").unlink()
    return lo, path


def solve_panel(pid: str, mesh: dict, setup: dict) -> dict:
    """Errors exactly as the scorer computes them, plus the per-element error map."""
    p = fea.get_problem(pid)
    ref = fe.reference(pid)
    res = fe._solve(p, mesh, setup, TASKS[pid]["qoi"])
    e = fe.errors_of_result(pid, res)
    m = meshio.read(mesh["path"])
    tris = np.vstack([c.data[:, :3] for c in m.cells if c.type.startswith("triangle")])
    used, tris = np.unique(tris, return_inverse=True)
    pts = m.points[used, :2]
    tris = tris.reshape(-1, 3)
    # local share of the field error: energy density of the stress error, per element (weighted mean over the
    # reference integration points inside it), relative to the mean energy density of the true stress.
    # Squared and integrated over the part this is exactly field_error**2, so dark = where the error comes from.
    nu = reference_setup(p)["material"]["nu"]
    s = fe.stresses(res, ref["pts"], p.dim)
    dens = lambda t: (1 + nu) * (sum(t[k] ** 2 for k in fe.DIAG) + 2 * sum(t[k] ** 2 for k in fe.OFFDIAG)) \
        - nu * sum(t[k] for k in fe.DIAG) ** 2
    w = ref["w"]
    err_d = dens({k: s[k] - ref["sigma"][k] for k in s})
    ref_mean = np.sum(w * dens(ref["sigma"])) / np.sum(w)
    el = Triangulation(pts[:, 0], pts[:, 1], tris).get_trifinder()(*ref["pts"][:, :2].T)
    ok = el >= 0
    ww = np.bincount(el[ok], w[ok], minlength=len(tris))
    el_err = np.sqrt(np.bincount(el[ok], w[ok] * err_d[ok], minlength=len(tris)) / np.maximum(ww, 1e-300) / ref_mean)
    return {"dofs": int(res.info.n_dofs), "elements": len(tris), "error": e["error"], "field": e["field"],
            "probe": e["points"], "pts": pts, "tris": tris, "el_err": el_err,
            "field_from_map": float(np.sqrt(np.sum(ww * el_err ** 2) / np.sum(w)))}   # == field (check)


def agent_panel(run_dir: str, run_id: str) -> dict:
    f = run_file(run_dir, run_id)
    run = json.loads(f.read_text())
    cp = ckpt.checkpoint(run)
    d = solve_panel(run["problem_id"], cp["mesh"], cp["setup"])
    c = run["checks"]
    d.update(source=str(f), run_id=run_id, gen=run["generation"], request=cp["request"],
             stored_error=c["rel_error"], stored_dofs=c["dofs"],
             matches_stored=abs(d["error"] - c["rel_error"]) <= 1e-6 * c["rel_error"] and d["dofs"] == c["dofs"])
    assert d["matches_stored"], f"{run_id}: re-solved {d['error']} / {d['dofs']} vs stored {c['rel_error']} / {c['dofs']}"
    return d


def auto_panel(pid: str, dofs: int) -> dict:
    f, path = automesher_at_least(pid, dofs)
    d = solve_panel(pid, {"path": str(path), "order": 2}, reference_setup(fea.get_problem(pid)))
    d.update(factor=f, mesh_file=str(path.relative_to(ROOT)))
    assert d["dofs"] >= dofs
    return d


# --------------------------------------------------------------------------- error-map figures


def draw_error(ax, d: dict, norm) -> None:
    tri = Triangulation(d["pts"][:, 0], d["pts"][:, 1], d["tris"])
    ax.tripcolor(tri, facecolors=np.clip(d["el_err"], *ERR_RANGE), cmap=ERR_CMAP, norm=norm)
    ax.triplot(tri, color=EDGE, lw=0.4, alpha=0.5)
    ax.set_aspect("equal")
    ax.axis("off")


def error_map_figure(panels: list[tuple[str, dict]], title: str, stem: str, cap_title: str, cap_body: str,
                     wspace: float = 0.05) -> None:
    norm = Normalize(*ERR_RANGE)
    fig, axes = plt.subplots(1, len(panels), figsize=(12, 6.75))
    fig.subplots_adjust(left=0.01, right=0.86, top=0.74, bottom=0.03, wspace=wspace)
    for ax, (name, d) in zip(axes, panels):
        draw_error(ax, d, norm)
        ax.set_title(f"{name}\n{d['dofs']} DOFs, error {pct(d['error'])}", fontsize=15)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=ERR_CMAP)
    cb = fig.colorbar(sm, cax=fig.add_axes([0.89, 0.08, 0.025, 0.6]), extend="max")
    ticks = np.linspace(*ERR_RANGE, 5)
    cb.set_ticks(ticks, labels=[f"{100 * t:g} %" for t in ticks])
    cb.set_label("local stress error (dark = wrong)", fontsize=14)
    fig.suptitle(title, fontsize=18, y=0.97)
    save(fig, stem, cap_title, cap_body)


def describe(d: dict) -> str:
    return (f"{d['dofs']} DOFs, {d['elements']} elements, scored error {pct(d['error'])} "
            f"(field {pct(d['field'])}, probe points {pct(d['probe'])})")


def common_caption() -> str:
    return ("Error = the scored error, max(whole-field energy-norm error, RMS von Mises error at the probe "
            "points), vs the fine reference in `ace/field_refs/`. Colours: each element's share of the field error: "
            "sqrt of (mean stress-error energy density in the element, over the reference integration points "
            "inside it) / (mean energy density of the true stress over the part). Integrated over the part, its "
            "square gives field error^2 exactly (checked: `field_from_map` in `gallery_data/gallery.json`), so dark "
            "= where the scored error comes from. Linear scale "
            f"{100 * ERR_RANGE[0]:g}-{100 * ERR_RANGE[1]:g} %, clipped above, the same on every panel (and as the Cook headline figures); mesh edges overlaid. Agent meshes: the "
            "run's own scored .msh and setup, re-solved with the fixed skfem solver; re-solved error and DOFs equal "
            "the run file's `checks.rel_error` / `checks.dofs` (`gallery_data/gallery.json`). Refined automesher: "
            "gmsh defaults x Mesh.MeshSizeFactor (as `talk_plots/act1_agentic_works/automesher_sweep.py`), the "
            "largest factor giving at least the agent's DOFs, solved with the reference setup "
            "(`ace/checks.py reference_setup`). Setups checked by `gallery_scan.py` (`gallery_data/setup_check.json`).")


def fig_le11(R: dict) -> None:
    a, g = R["le11_auto"], R["le11_agent"]
    ratio = a["error"] / g["error"]
    error_map_figure(
        [("Automesher, refined\nto the same budget", a), ("Trained agent\n(generation 10)", g)],
        f"NAFEMS LE11: same budget, {ratio:.1f}x less error with the agent's mesh",
        "fig_gallery_le11_axisym", "LE11 axisymmetric at an equal DOF budget",
        f"NAFEMS LE11 (thermal stress, axisymmetric cross-section), run `ace_runs/deep1`. Left: the automesher "
        f"refined by factor {a['factor']:.3f} ({describe(a)}). Right: `{g['run_id']}` (`{g['source']}`; {describe(g)}). "
        f"The agent has {a['dofs'] - g['dofs']} fewer DOFs and {ratio:.2f}x lower error. Mesh request: "
        f"`{json.dumps(g['request'])}`.\n\n{common_caption()}\n\nCaveats: only correct-setup runs are eligible; all "
        f"four generation-0 LE11 runs of deep1 used the temperature sqrt(x**2+y**2)+y instead of x+y and are excluded. "
        f"This is a win over the automesher, not learning across generations: the best correct-setup generation-1 "
        f"run (`g1-nafems_le11_axisym-a3`) already had 0.71 % at 812 DOFs; generations 1-10 mostly cut DOFs "
        f"(812 -> 678) at the same error. The automesher has {a['dofs'] - g['dofs']} more DOFs than the agent, which "
        f"favours the automesher.", wspace=-0.45)


def fig_le1(R: dict) -> None:
    a, g = R["le1_auto"], R["le1_agent"]
    ratio = a["error"] / g["error"]
    error_map_figure(
        [("Automesher, refined\nto the same budget", a), (f"Agent\n(generation {g['gen']})", g)],
        f"NAFEMS LE1: same budget, {ratio:.1f}x less error with the agent's mesh",
        "fig_gallery_le1", "LE1 at an equal DOF budget",
        f"NAFEMS LE1 (elliptic membrane, plane stress), run `ace_runs/split1` (Cook + LE1, agents split at "
        f"random each generation). Left: the automesher refined by factor {a['factor']:.3f} ({describe(a)}). "
        f"Right: `{g['run_id']}` (`{g['source']}`; {describe(g)}). Mesh request: `{json.dumps(g['request'])}` "
        f"(fine at the stress peak, point D = (2, 0)).\n\n{common_caption()}\n\nCaveats: a win over the automesher, "
        f"not clear learning: generation-0 agents were already at 0.96 % with 826 DOFs (1.8x better than the "
        f"automesher at those DOFs). Rejected alternatives: `g7-nafems_le1-a1` / `g18-nafems_le1-a5` (666 DOFs) "
        f"have field error 0.96 % but scored error 1.48 % (probe points), only ~1.55x better than the automesher "
        f"(~2.29 % at 666 DOFs, interpolated in automesher_sweep.json); quoting 0.96 % against 2.29 % mixes field "
        f"and scored errors. The automesher's error does not fall smoothly with DOFs (gmsh remeshes): refined to "
        f"702 DOFs it has 2.55 %, so `g1-nafems_le1-a2` (696 DOFs, 0.98 %) would show 2.6x; this figure uses the "
        f"later, less luck-dependent pair. "
        f"Within-run 2x pairs in split1 LE1 (e.g. g5-a1 1.72 % -> g14-a5 0.62 %) improve only the probe-point "
        f"error; their field errors are equal or worse (0.57 % -> 0.62 %), so an error map shows no difference.")


def fig_cook(R: dict) -> None:
    a, e, g = R["cook_auto"], R["cook_early"], R["cook_agent"]
    error_map_figure(
        [("Automesher, refined\nto the same budget", a), (f"Agent\n(generation {e['gen']})", e),
         (f"Same training run\n(generation {g['gen']})", g)],
        f"Cook's membrane, one training run: {e['error'] / g['error']:.1f}x less error by generation {g['gen']}, "
        f"with fewer DOFs",
        "fig_gallery_cook_within_run", "Cook's membrane: learning within one run at equal or fewer DOFs",
        f"Cook's membrane, run `ace_runs/cook_auto` (6 agents, 20 generations, field score). Left: the automesher "
        f"refined by factor {a['factor']:.3f} ({describe(a)}). Middle: `{e['run_id']}` (`{e['source']}`; {describe(e)}; "
        f"request `{json.dumps(e['request'])}`). Right: `{g['run_id']}` (`{g['source']}`; {describe(g)}; request "
        f"`{json.dumps(g['request'])}`: fine at all four corners). Generation {g['gen']} has fewer DOFs than both "
        f"others and {e['error'] / g['error']:.2f}x / {a['error'] / g['error']:.2f}x lower error.\n\n{common_caption()}\n\n"
        f"Caveats: generation-1 errors were 2.19-2.90 % (this run 2.78 %; the best, g1-a0, 2.19 % at 1092 DOFs, is "
        f"1.7x worse than generation 7 with more DOFs); "
        f"generation 0 (all six at 772 DOFs, 2.48 %) used fewer DOFs than generation 7, so it cannot be the "
        f"comparison under the equal-budget rule. The other 2x-or-better within-run pairs found by the scan are "
        f"listed in `gallery_data/pairs.json`.")


# --------------------------------------------------------------------------- real test


def read_curve(d: Path) -> dict:
    z = np.genfromtxt(d / "curve.csv", delimiter=",", names=True)
    return {"u": z["u_cross_mm"], "F": z["force_N"]}


def fig_real_test() -> dict:
    B = json.loads((BASE / "baselines.json").read_text())
    meas = nonlinear.measured()
    a, s = B["baselines"]["auto"], B["baselines"]["swarm_bc0"]
    fig, ax = plt.subplots(figsize=(12, 6.75))
    fig.subplots_adjust(left=0.09, right=0.97, top=0.86, bottom=0.12)
    um, Fm = np.array(meas["curve_mm"]), np.array(meas["curve_N"])
    xmax = max(meas["checkpoints_mm"]) * 1.08
    ax.plot(um[um <= xmax], Fm[um <= xmax] / 1e3, color=MEAS, lw=3.5, label="measured (Instron test)")
    for d, b, col in ((BASE / "auto", a, GREY), (BASE / "swarm_bc0", s, AGENT)):
        c = read_curve(d)
        ax.plot(c["u"], c["F"] / 1e3, color=col, lw=2.5,
                label=f"{b['label']}: {b['nodes']:,} nodes, {pct(b['mean_rel_err'])} off")
    if FINETUNED_SUMMARY is not None and FINETUNED_SUMMARY.exists():
        f = json.loads(FINETUNED_SUMMARY.read_text())
        ax.plot(f["u_cross_mm"], np.array(f["force_N"]) / 1e3, color=EARLY, lw=2.5,
                label=f"fine-tuned: {f['nodes']:,} nodes")
    else:
        ax.plot([], [], color=EARLY, lw=2.5, ls=":", label="fine-tuned mesh (pending)")
    ax.plot(meas["checkpoints_mm"], np.array(meas["force_N"]) / 1e3, "o", color=MEAS, ms=9, mfc="white", mew=2,
            label="checkpoints (where the error is measured)")
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, None)
    ax.set_xlabel("crosshead travel (mm)")
    ax.set_ylabel("force (kN)")
    ax.legend(loc="upper left", frameon=False)
    ax.set_title(f"Real test: the swarm's mesh is closer to the measured curve, but with "
                 f"{s['nodes'] / a['nodes']:.1f}x the nodes\n(not an equal-budget win yet: the fine-tuned mesh is pending)",
                 fontsize=17, loc="left")
    body = (f"The Instron compression test of the printed C-bracket (`physical_tests/abc_00000065_compression/"
            f"cload_compression.csv`, loading branch to the peak, via `ace/nonlinear.py measured`) against the frozen "
            f"elastoplastic CalculiX ramp (`ace/nonlinear.py`) on two meshes, from `results/real_data_finetune/baselines/"
            f"<name>/curve.csv` and `baselines.json` (computed on the Brev box by `scripts/nl_baselines.py`; nothing "
            f"re-solved here, CalculiX is not installed locally). Error = mean over the {len(meas['checkpoints_mm'])} "
            f"checkpoints {meas['checkpoints_mm']} mm of |F_sim - F_meas| / F_meas.\n\n"
            f"- automesher: {a['nodes']:,} nodes ({a['dofs']:,} DOFs), {pct(a['mean_rel_err'])}; per checkpoint "
            f"{[pct(x) for x in a['rel_err']]}\n"
            f"- swarm bc0 mesh (div_s8 g7 a2, trained on simulation only; request "
            f"`results/real_data_finetune/baselines/swarm_bc0_request.json`): {s['nodes']:,} nodes ({s['dofs']:,} DOFs), "
            f"{pct(s['mean_rel_err'])}; per checkpoint {[pct(x) for x in s['rel_err']]}\n"
            f"- fine-tuned mesh (ace_runs/nl_ft, running on the Brev box): pending; set `FINETUNED_SUMMARY` in "
            f"`make_gallery.py` to its nonlinear `summary.json` to draw it.\n\n"
            f"Caveat: this figure does NOT meet the equal-budget rule. The swarm's mesh has "
            f"{s['nodes'] / a['nodes']:.2f}x the automesher's nodes, and the only local mesh with fewer nodes than "
            f"it is the automesher (uniform 4 mm: {B['baselines']['h4']['nodes']:,} nodes, "
            f"{pct(B['baselines']['h4']['mean_rel_err'])}, has more). An equal-budget version needs either the "
            f"fine-tuned mesh at <= the automesher's nodes or the automesher refined to {s['nodes']:,} nodes through "
            f"the same ramp (CalculiX, Brev box). The two curves differ by only ~{(s['force_sim_N'][-1] - a['force_sim_N'][-1]):.0f} N "
            f"of ~{s['force_sim_N'][-1]:.0f} N at 2.5 mm, so the gap is modest on screen.")
    save(fig, "fig_gallery_real_test", "Real test: force curves vs the measurement", body)
    return {"auto": a, "swarm_bc0": s, "checkpoints_mm": meas["checkpoints_mm"], "force_meas_N": meas["force_N"]}


# --------------------------------------------------------------------------- main


def main() -> None:
    DATA.mkdir(exist_ok=True)
    R = {}
    R["le11_agent"] = agent_panel(LE11["run_dir"], LE11["agent"])
    R["le11_auto"] = auto_panel(LE11["problem"], R["le11_agent"]["dofs"])
    R["le1_agent"] = agent_panel(LE1["run_dir"], LE1["agent"])
    R["le1_auto"] = auto_panel(LE1["problem"], R["le1_agent"]["dofs"])
    R["cook_early"] = agent_panel(COOK["run_dir"], COOK["early"])
    R["cook_agent"] = agent_panel(COOK["run_dir"], COOK["agent"])
    R["cook_auto"] = auto_panel(COOK["problem"], R["cook_agent"]["dofs"])
    for k, d in R.items():
        print(f"{k:12s} dofs {d['dofs']:5d}  error {pct(d['error'])}  field {pct(d['field'])}  "
              f"map {pct(d['field_from_map'])}  el_err 5/50/95 % {np.percentile(d['el_err'], [5, 50, 95]).round(4)}  "
              f"factor {d.get('factor', '-')}  stored {d.get('stored_error', '-')}")
    fig_le11(R)
    fig_le1(R)
    fig_cook(R)
    real = fig_real_test()
    keep = {k: {kk: v for kk, v in d.items() if kk not in ("pts", "tris", "el_err")} for k, d in R.items()}
    (DATA / "gallery.json").write_text(json.dumps({**keep, "real_test": real}, indent=1, default=float))


if __name__ == "__main__":
    main()
