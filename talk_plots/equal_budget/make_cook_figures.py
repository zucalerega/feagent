"""Cook's membrane at an EQUAL DOF BUDGET: the automesher refined to the trained agent's DOFs vs an
early agent (more DOFs) vs the trained agent vs the true solution.

    .venv/bin/python talk_plots/equal_budget/make_cook_figures.py      (~1 min, no API calls)

Meshes (all solved with the fixed skfem solver, scored exactly as the runs were scored:
ace/field_error.py errors_of_result on the run's scored mesh + setup, from ace/checkpoint.py checkpoint):
    1. auto  gmsh automesher with Mesh.MeshSizeFactor = AUTO_FACTOR (every element AUTO_FACTOR x gmsh's
             default size), quadratic, the problem's correct setup. Same mesh as
             talk_plots/act1_agentic_works/automesher_sweep.py; the result is checked against
             automesher_sweep.json.
    2. early EARLY_RUN, an early agent (generation 1 of the ace_runs/cook_auto training run)
    3. final FINAL_RUN, the trained agent (generation 10 of ace_runs/deep1)
    4. ref   the fine reference (ace/field_refs/cooks_membrane.npz) on the dashboard's target tessellation

Setup check (rule: only correct-setup runs): each agent's setup is written differently from
ace/checks.py reference_setup (traction 1/16 on the 16-long loaded edge instead of total force 1,
nu = 0.3333333 instead of 1/3), so the run's mesh is also solved with reference_setup and the two
field errors must agree (to SETUP_TOL relative).

Figures (PNG 200 dpi + PDF + .md caption): fig_cook_equal_stress, fig_cook_equal_error,
fig_cook_same_budget. Numbers: data/cook_panels.json.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import meshio  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402
from matplotlib.tri import Triangulation  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
DATA = OUT / "data"
os.chdir(ROOT)                            # run records hold mesh paths relative to the repo
sys.path.insert(0, str(ROOT))

from ace import checkpoint as ckpt, fea  # noqa: E402
from ace import field_error as fe  # noqa: E402
from ace.checks import reference_setup  # noqa: E402
from ace.config import TASKS  # noqa: E402

PROBLEM = "cooks_membrane"
AUTO_FACTOR = 0.65                        # automesher_sweep.json: 734 DOFs, the row nearest the trained agent's 710
SWEEP = Path("talk_plots/act1_agentic_works/automesher_sweep.json")
EARLY_RUN = Path("ace_runs/cook_auto/gen1/runs/g1-cooks_membrane-a3.json")
FINAL_RUN = Path("ace_runs/deep1/gen10/runs/g10-cooks_membrane-a7.json")
SETUP_TOL = 1e-6

# style: as talk_plots/self_learning/make_figures.py (the agent dashboard's look)
EDGE = "#26313b"
CMAP = "turbo"
ERR_CMAP = "magma_r"                      # dark = large error
ERR_MAX = 0.10                            # linear error colour scale 0..10 % (clipped above)
ZOOM = (0.0, 10.0, 37.0, 47.5)            # clamped top-left corner (0, 44): the stress peak
INSET = (25.0, 0.0, 23.0, 21.0)           # where the zoom is drawn (empty area under the sloped edge)
plt.rcParams.update({"font.size": 15, "axes.titlesize": 17, "figure.facecolor": "white",
                     "savefig.facecolor": "white"})


def save(fig, stem: str, title: str, body: str) -> None:
    fig.savefig(OUT / f"{stem}.png", dpi=200, bbox_inches="tight", pad_inches=0.15)
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    (OUT / f"{stem}.md").write_text(f"# {title}\n\n{body}\n")


def pct(x: float) -> str:
    return f"{100 * x:.2f} %"


def sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# --------------------------------------------------------------------------- meshes


def refined_automesher(p, path: Path) -> dict:
    """gmsh defaults x AUTO_FACTOR, quadratic (automesher_sweep.py one())."""
    import gmsh
    from feacore.geometry.gmsh_tools import ensure_domain_group, gmsh_session
    with gmsh_session(p.id):
        p.build_geometry()
        ensure_domain_group(p.dim)
        gmsh.option.setNumber("Mesh.MeshSizeFactor", AUTO_FACTOR)
        gmsh.model.mesh.generate(p.dim)
        gmsh.model.mesh.setOrder(2)
        gmsh.write(str(path))
    return {"path": str(path), "order": 2}


def agent_mesh(p, run: dict, name: str) -> tuple[dict, dict, bool]:
    """The run's scored mesh, rebuilt from its request (falls back to the run's own .msh if gmsh
    does not reproduce it byte for byte)."""
    cp = ckpt.checkpoint(run)
    req = cp["request"]
    path = DATA / f"{name}.msh"
    new = fea.make_mesh(p, path, h=req["h"], order=req.get("order", 1), refinements=req.get("refinements"))
    same = new.get("sha256") == cp["mesh"].get("sha256")
    if not same:
        shutil.copy(cp["mesh"]["path"], path)
        new = dict(cp["mesh"], path=str(path))
    return new, cp, same


# --------------------------------------------------------------------------- solve + per-element error


def energy_density(s: dict, mat: dict) -> np.ndarray:
    """s : C^-1 : s at each point (the integrand of ace/field_error.py energy_sq)."""
    ss = sum(s[k] ** 2 for k in fe.DIAG) + 2 * sum(s[k] ** 2 for k in fe.OFFDIAG)
    tr = sum(s[k] for k in fe.DIAG)
    return ((1 + mat["nu"]) * ss - mat["nu"] * tr ** 2) / mat["E"]


def solve_panels() -> tuple[dict, dict]:
    p = fea.get_problem(PROBLEM)
    qoi = TASKS[PROBLEM]["qoi"]
    mat = reference_setup(p)["material"]
    ref = fe.reference(PROBLEM)
    vm_ref = fe.von_mises(ref["sigma"])
    w = ref["w"]
    ref_mean = np.sum(w * energy_density(ref["sigma"], mat)) / np.sum(w)     # mean reference energy density
    P, F = {}, {}

    def finish(key: str, title: str, mesh: dict, setup: dict, info: dict) -> None:
        res = fe._solve(p, mesh, setup, qoi)
        e = fe.errors_of_result(PROBLEM, res)
        m = meshio.read(mesh["path"])
        tris = np.vstack([c.data[:, :3] for c in m.cells if c.type.startswith("triangle")])
        used, tris = np.unique(tris, return_inverse=True)
        f = {"pts": m.points[used, :2], "tris": tris.reshape(-1, 3)}
        f["vm"] = np.asarray(res.probe("von_mises", f["pts"]))
        # per element: the scored error, localised. e_el^2 = (error energy density averaged over the
        # element) / (reference energy density averaged over the part), so that
        # sum(area_el x e_el^2) / total area = field error^2: the dark elements ARE the error.
        s = fe.stresses(res, ref["pts"], p.dim)
        d = energy_density({k: s[k] - ref["sigma"][k] for k in s}, mat)
        el = Triangulation(*f["pts"].T, f["tris"]).get_trifinder()(*ref["pts"][:, :2].T)
        ok = el >= 0
        num = np.bincount(el[ok], w[ok] * d[ok], minlength=len(f["tris"]))
        f["el_err"] = np.sqrt(num / np.maximum(np.bincount(el[ok], w[ok], minlength=len(f["tris"])), 1e-30) / ref_mean)
        F[key] = f
        top = np.sort(num)[::-1][: max(1, len(num) // 20)].sum() / num.sum()
        P[key] = {"title": title, "dofs": int(res.info.n_dofs), "nodes": int(len(m.points)),
                  "elements": int(len(f["tris"])), "field_error": e["field"], "probe_error": e["points"],
                  "scored_error": e["error"], "field_error_from_element_map": float(np.sqrt(num.sum() / (ref_mean * w.sum()))),
                  "error_share_of_worst_5pct_elements": float(top), **info}

    # 1. automesher refined to the trained agent's budget, correct setup
    path = DATA / "auto_refined.msh"
    finish("auto", f"Automesher, refined\n(gmsh defaults x {AUTO_FACTOR})", refined_automesher(p, path),
           reference_setup(p), {"mesh_file": str(path.relative_to(ROOT)), "size_factor": AUTO_FACTOR})
    row = next(r for r in json.loads(SWEEP.read_text())[PROBLEM]["rows"] if r["factor"] == AUTO_FACTOR)
    P["auto"]["stored"] = {"source": str(SWEEP), "field_error": row["field"], "dofs": row["dofs"]}

    # 2./3. agent runs: scored mesh and the run's own setup; then the same mesh with reference_setup
    for key, f, title in (("early", EARLY_RUN, "Early agent\n(generation 1)"),
                          ("final", FINAL_RUN, "Trained agent\n(generation 10)")):
        run = json.loads(f.read_text())
        mesh, cp, same = agent_mesh(p, run, key)
        c = run["checks"]
        finish(key, title, mesh, cp["setup"],
               {"run_id": run["run_id"], "request": cp["request"], "setup": cp["setup"],
                "rebuilt_sha_matches_run": same,
                "stored": {"source": str(f), "field_error": c["field_error"], "dofs": c["dofs"], "gain": c["gain"]}})
        e_ref = fe.errors_of_result(PROBLEM, fe._solve(p, mesh, reference_setup(p), qoi))["field"]
        P[key]["field_error_with_reference_setup"] = e_ref
        P[key]["setup_equivalent"] = abs(e_ref - P[key]["field_error"]) <= SETUP_TOL * e_ref

    for k in ("auto", "early", "final"):
        s = P[k]["stored"]
        P[k]["matches_stored"] = (abs(P[k]["field_error"] - s["field_error"]) <= 1e-6 * s["field_error"]
                                  and P[k]["dofs"] == s["dofs"])

    # 4. the truth on the dashboard's target tessellation (inverse-distance mean of 4 nearest ref points)
    m = meshio.read(f"ace/field_refs/{PROBLEM}_target.msh")
    tris = np.vstack([c.data[:, :3] for c in m.cells if c.type.startswith("triangle")])
    dist, idx = cKDTree(ref["pts"][:, :2]).query(m.points[:, :2], k=4)
    iw = 1 / np.maximum(dist, 1e-12)
    F["ref"] = {"pts": m.points[:, :2], "tris": tris, "vm": (vm_ref[idx] * iw).sum(1) / iw.sum(1)}
    rnpz = np.load(f"ace/field_refs/{PROBLEM}.npz")
    P["ref"] = {"title": "True solution\n(fine reference)", "dofs": int(rnpz["n_dofs"]),
                "source": f"ace/field_refs/{PROBLEM}.npz",
                "reference_self_error": fe.reference_check(PROBLEM)["coarse_vs_fine"]}
    P["_n_ref_points"] = int(len(ref["pts"]))
    return P, F


# --------------------------------------------------------------------------- drawing


def draw_stress(ax, f: dict, norm, edges: bool, lw: float = 0.6):
    tri = Triangulation(f["pts"][:, 0], f["pts"][:, 1], f["tris"])
    pc = ax.tripcolor(tri, f["vm"], shading="gouraud", cmap=CMAP, norm=norm)
    if edges:
        ax.triplot(tri, color=EDGE, lw=lw)
    ax.set_aspect("equal")
    ax.axis("off")
    return pc


def add_zoom(ax, f: dict, norm, edges: bool) -> None:
    """Magnified clamped top-left corner, drawn in the panel's empty lower-right area."""
    x0, x1, y0, y1 = ZOOM
    ins = ax.inset_axes(INSET, transform=ax.transData)
    draw_stress(ins, f, norm, edges, lw=0.8)
    ins.set_xlim(x0, x1)
    ins.set_ylim(y0, y1)
    ins.axis("on")
    ins.set_xticks([])
    ins.set_yticks([])
    for sp in ins.spines.values():
        sp.set_color("black")
        sp.set_linewidth(1.2)
    ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="black", lw=1.2, clip_on=False))
    ax.annotate("", xy=(INSET[0], INSET[1] + INSET[3]), xytext=(x1, y0), xycoords="data",
                arrowprops=dict(arrowstyle="-", color="black", lw=0.8, ls=":"))


def draw_error(ax, f: dict, norm):
    tri = Triangulation(f["pts"][:, 0], f["pts"][:, 1], f["tris"])
    pc = ax.tripcolor(tri, facecolors=np.minimum(f["el_err"], ERR_MAX), cmap=ERR_CMAP, norm=norm)
    ax.triplot(tri, color=EDGE, lw=0.4, alpha=0.5)
    ax.set_aspect("equal")
    ax.axis("off")
    return pc


def label(P: dict, k: str) -> str:
    d = P[k]
    if k == "ref":
        return f"{d['dofs']:,} DOFs\n(error 0 by definition)"
    return f"{d['dofs']} DOFs\nerror {100 * d['field_error']:.2f} %"


def colorbar(fig, pc, text: str, rect=(0.2, 0.07, 0.6, 0.03), extend: str = "both"):
    cb = fig.colorbar(pc, cax=fig.add_axes(list(rect)), orientation="horizontal", extend=extend)
    cb.set_label(text, fontsize=14)
    cb.ax.tick_params(labelsize=14)
    return cb


# --------------------------------------------------------------------------- figures + captions


def figures(P: dict, F: dict) -> None:
    a, z, f, r = P["auto"], P["early"], P["final"], P["ref"]
    ratio = a["field_error"] / f["field_error"]
    common = (f"Cook's membrane (plane stress, E = 1, nu = 1/3, unit total shear load on the right edge; "
              f"dimensionless stress). Error = relative energy-norm error of the whole stress field vs the fine "
              f"reference (`{r['source']}`, {r['dofs']:,} DOFs; a coarser reference solve differs from it by "
              f"{pct(r['reference_self_error'])}), the field error the runs were scored with. Every mesh was "
              f"rebuilt and re-solved with the fixed skfem solver; re-solved errors and DOFs match the stored "
              f"ones (`data/cook_panels.json`).")
    meshes = (f"(1) gmsh automesher with every element {AUTO_FACTOR} x gmsh's default size, quadratic, correct "
              f"setup ({a['dofs']} DOFs, {pct(a['field_error'])}; row factor {AUTO_FACTOR} of `{a['stored']['source']}`); "
              f"(2) early agent `{z['run_id']}` (`{z['stored']['source']}`; {z['dofs']} DOFs, {pct(z['field_error'])}; "
              f"mesh request `{json.dumps(z['request'])}`); (3) trained agent `{f['run_id']}` "
              f"(`{f['stored']['source']}`; {f['dofs']} DOFs, {pct(f['field_error'])}; mesh request "
              f"`{json.dumps(f['request'])}`).")
    setup = (f"Setup check: both agents wrote the load as a traction 0.0625 on the 16-long loaded edge (= total "
             f"force 1) and nu = 0.3333333; solving their meshes with `ace/checks.py reference_setup` gives the "
             f"same field error ({pct(z['field_error_with_reference_setup'])}, "
             f"{pct(f['field_error_with_reference_setup'])}), so both setups are correct.")
    caveats = (f"Caveats: the early agent is from a different training run (`ace_runs/cook_auto`) than the trained "
               f"agent (`ace_runs/deep1`), whose own generation-0/1 agents used fewer DOFs (252-532), so they do "
               f"not fit an equal-budget comparison. The early run shown has the highest field error of the "
               f"6 generation-1 runs of cook_auto (all have 772-1092 DOFs, 2.19-2.90 %; its generation 0: "
               f"6 identical runs, 772 DOFs, 2.48 %). The automesher row has {a['dofs'] - f['dofs']} more DOFs "
               f"than the trained agent.")
    errmap = (f"Error map: the scored error, element by element. For each element, e_el = sqrt(error energy "
              f"density averaged over the element / reference energy density averaged over the part), from the "
              f"{P['_n_ref_points']} reference integration points (error = stress of the mesh minus the reference "
              f"stress, energy density s : C^-1 : s). By construction the area-weighted mean of e_el^2 is the "
              f"field error squared (recomputed from the maps: {pct(a['field_error_from_element_map'])}, "
              f"{pct(z['field_error_from_element_map'])}, {pct(f['field_error_from_element_map'])}), so the dark "
              f"elements are where the number under the panel comes from. Linear colour scale 0-{100 * ERR_MAX:g} %, "
              f"clipped above, identical on all panels; dark = wrong. The worst 5 % of elements hold "
              f"{100 * a['error_share_of_worst_5pct_elements']:.0f} % (automesher), "
              f"{100 * z['error_share_of_worst_5pct_elements']:.0f} % (early) and "
              f"{100 * f['error_share_of_worst_5pct_elements']:.0f} % (trained) of the squared error: the "
              f"automesher's error sits at the two corners (the clamped top-left corner and the loaded top-right "
              f"corner); the trained agent spends its DOFs there and spreads the remaining error evenly. Honest "
              f"note: in the interior the trained agent's elements are larger and its local error is a little "
              f"higher (median element {100 * np.median(F['final']['el_err']):.2f} % vs "
              f"{100 * np.median(F['auto']['el_err']):.2f} % for the automesher); both are pale on this scale.")

    # (a) solved von Mises stress, 4 panels, dashboard style
    norm = Normalize(*np.percentile(F["ref"]["vm"], [1, 99]))
    fig, axes = plt.subplots(1, 4, figsize=(12, 6.75))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.78, bottom=0.27, wspace=0.06)
    for ax, k in zip(axes, ("auto", "early", "final", "ref")):
        pc = draw_stress(ax, F[k], norm, edges=k != "ref")
        add_zoom(ax, F[k], norm, edges=k != "ref")
        ax.set_title(P[k]["title"], fontsize=15)
        ax.text(0.5, -0.04, label(P, k), transform=ax.transAxes, ha="center", va="top", fontsize=14)
    colorbar(fig, pc, "von Mises stress (same scale on all four; boxes: corner magnified)")
    fig.suptitle(f"Same number of unknowns: the trained agent's stress matches the truth,\n"
                 f"{ratio:.1f}x more accurate than the refined automesher", fontsize=18, y=0.98)
    save(fig, "fig_cook_equal_stress", "Cook's membrane at an equal DOF budget: solved stress",
         f"Solved von Mises stress on one colour scale (Turbo, 1st-99th percentile of the reference, as the agent "
         f"dashboard). Left to right: {meshes} (4) true solution. Boxes in the lower right magnify the clamped "
         f"top-left corner (x {ZOOM[0]:g}-{ZOOM[1]:g}, y {ZOOM[2]:g}-{ZOOM[3]:g}), where the stress peaks. "
         f"{common}\n\n{setup}\n\n{caveats}")

    # (b) error maps, 3 panels (the truth has zero error)
    enorm = Normalize(0, ERR_MAX)
    fig, axes = plt.subplots(1, 3, figsize=(12, 6.75))
    fig.subplots_adjust(left=0.03, right=0.97, top=0.80, bottom=0.27, wspace=0.08)
    for ax, k in zip(axes, ("auto", "early", "final")):
        pc = draw_error(ax, F[k], enorm)
        ax.set_title(P[k]["title"], fontsize=15)
        ax.text(0.5, -0.04, label(P, k), transform=ax.transAxes, ha="center", va="top", fontsize=14)
    cb = colorbar(fig, pc, "local stress error vs the true solution (dark = wrong)", extend="max")
    cb.set_ticks(np.linspace(0, ERR_MAX, 5), labels=[f"{100 * t:g} %" for t in np.linspace(0, ERR_MAX, 5)])
    fig.suptitle(f"Same budget, far less error: the trained agent removes the dark (wrong) corners\n"
                 f"({f['dofs']} DOFs, {pct(f['field_error'])} vs automesher {a['dofs']} DOFs, "
                 f"{pct(a['field_error'])})", fontsize=18, y=0.98)
    save(fig, "fig_cook_equal_error", "Cook's membrane at an equal DOF budget: error maps",
         f"Where each mesh is wrong. Left to right: {meshes} {errmap} {common}\n\n{setup}\n\n{caveats}")

    # (c) slide version: two panels, refined automesher vs trained agent
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.75))
    fig.subplots_adjust(left=0.04, right=0.96, top=0.83, bottom=0.25, wspace=0.05)
    for ax, k, t in zip(axes, ("auto", "final"), ("Automesher (refined)", "Trained agent")):
        pc = draw_error(ax, F[k], enorm)
        ax.set_title(t, fontsize=17)
        ax.text(0.5, -0.03, f"{P[k]['dofs']} DOFs  ·  error {100 * P[k]['field_error']:.2f} %",
                transform=ax.transAxes, ha="center", va="top", fontsize=16)
    cb = colorbar(fig, pc, "local stress error vs the true solution (dark = wrong)", rect=(0.2, 0.08, 0.6, 0.035),
                  extend="max")
    cb.set_ticks(np.linspace(0, ERR_MAX, 5), labels=[f"{100 * t:g} %" for t in np.linspace(0, ERR_MAX, 5)])
    fig.suptitle(f"Same budget, {ratio:.1f}x more accurate", fontsize=22, y=0.97)
    save(fig, "fig_cook_same_budget", f"Same budget, {ratio:.1f}x more accurate",
         f"Slide version of fig_cook_equal_error: (left) {meshes.split('; (2)')[0]}; (right) trained agent "
         f"`{f['run_id']}` ({f['dofs']} DOFs, {pct(f['field_error'])}). {ratio:.2f}x = "
         f"{pct(a['field_error'])} / {pct(f['field_error'])}, with {a['dofs'] - f['dofs']} fewer DOFs for the agent. "
         f"{errmap} {common}\n\n{setup}")


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    P, F = solve_panels()
    for k in ("auto", "early", "final"):
        d = P[k]
        print(f"{k:6s} dofs {d['dofs']:5d} field {d['field_error']:.5f} (stored {d['stored']['field_error']:.5f}) "
              f"match {d['matches_stored']} setup_ok {d.get('setup_equivalent', '-')} "
              f"sha {d.get('rebuilt_sha_matches_run', '-')} map {d['field_error_from_element_map']:.5f} "
              f"worst5% share {d['error_share_of_worst_5pct_elements']:.2f}")
        assert d["matches_stored"], f"{k}: re-solved error/DOFs differ from the stored numbers"
        assert d.get("setup_equivalent", True), f"{k}: setup is not equivalent to reference_setup"
        assert abs(d["field_error_from_element_map"] - d["field_error"]) <= 1e-3 * d["field_error"], k
    (DATA / "cook_panels.json").write_text(json.dumps(P, indent=1, default=float))
    figures(P, F)


if __name__ == "__main__":
    main()
