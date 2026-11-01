"""Self-learning figures: the automesher vs an early agent vs the trained agent vs the truth.

Regenerates everything in this folder (run from anywhere, a few minutes the first time, no API calls):

    .venv/bin/python talk_plots/self_learning/make_figures.py

Two 2D problems of run ace_runs/deep1 (11 generations, Opus teacher), each in CASES:
    NAFEMS LE11 axisymmetric (figures fig_L*) and Cook's membrane (figures fig_A*, fig_B*).

(0) Setup check. Only runs whose setup is the problem's own are used anywhere (panels and graphs),
    so every difference shown is the mesh. A run passes when its recorded setup (checkpoint(run)
    ["setup"], the setup of its scored solve) gives, on its own mesh, the same solution as the
    problem's reference setup (ace/checks.py reference_setup): equal field error to 1e-6. (Setups
    that differ only in spelling, e.g. "x+y" vs "x + y" or T_ref left out, pass; a different
    temperature field does not.)
(A) Solved von Mises stress drawn as the agent dashboard draws it (ace/agent_dashboard.py: von Mises
    at the mesh vertices, "Turbo" colour scale, element edges #26313b, colour range = 1st..99th
    percentile of the reference stress on the target tessellation) for four meshes on ONE scale:
        1. gmsh automesher (gmsh defaults, quadratic; ace/field_error.py automesher)
        2. the earliest agent: best run (by score) of the first generation with a correct setup
        3. the trained agent: best run (by score) of the last generation
        4. the true solution: the fine reference (ace/field_refs/<problem>.npz) on the dashboard's
           target tessellation (ace/field_refs/<problem>_target.msh)
    plus an error-map version (local von Mises error vs the reference, one log scale). Each stress
    panel carries a magnified view (the case's "zoom") of where the meshes differ.
    Every mesh is rebuilt and re-solved exactly as the scorer did (ace/checks.py check_run ->
    ace/field_error.py errors_of_setup: the run's scored mesh + setup, fixed skfem solver), and
    the field error and DOFs are checked against the numbers stored in the run file.
(B) Learning curves from the run files (correct-setup runs only): score relative to the automesher
    per generation (best agent, mean agent, min..max band), and the accuracy gain e_auto / e.
(C) index.html showing every figure with its caption, LE11 first.

Data written to data/: per-run table (with the setup check), per-generation tables, the re-solved
panel numbers and verification, the meshes (Cook in data/, LE11 in data/nafems_le11_axisym/), and
data/errors_cache/ (errors_of_setup results, so a re-run skips the setup-check solves).
"""

from __future__ import annotations

import csv
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
from matplotlib.colors import LogNorm, Normalize  # noqa: E402
from matplotlib.tri import Triangulation  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]                     # the feagent repo
DATA = OUT / "data"
os.chdir(ROOT)                            # run records hold mesh paths relative to the repo
sys.path.insert(0, str(ROOT))

from ace import checkpoint as ckpt, fea  # noqa: E402
from ace import field_error as fe  # noqa: E402
from ace.checks import reference_setup  # noqa: E402
from ace.config import TASKS  # noqa: E402

fe.ERRORS_CACHE = DATA / "errors_cache"   # keep errors_of_setup's cache inside this folder

# --------------------------------------------------------------------------- choices

RUN_DIR = Path("ace_runs/deep1")          # 11 generations, Opus teacher, Cook + LE11 axisymmetric
AUTO_SCORE = 0.25                         # the automesher's own score: g = 1 -> (1/2)^2 x 1 (ace/config.py)
LE11, COOK = "nafems_le11_axisym", "cooks_membrane"

# One entry per problem, in page order. zoom = (x0, x1, y0, y1) of the magnified box; inset =
# (x, y, width, height) in data units where the magnified view is drawn (an empty area of the plot).
CASES = {
    LE11: {
        "prefix": "L", "name": "NAFEMS LE11 (axisymmetric)", "data": DATA / LE11, "early_tag": "early",
        "vm_scale": 1e-6, "vm_unit": " (MPa)",
        "about": "NAFEMS LE11 axisymmetric (hollow cylinder / taper / sphere, steel, E = 210 GPa, nu = 0.3, "
                 "alpha = 2.3e-4 /degC, temperature T = r + z, i.e. x + y in the section; no mechanical load; "
                 "stresses in MPa)",
        "zoom": (0.9, 1.1, 1.27, 1.51), "inset": (0.03, 0.78, 0.6, 0.72),
        "zoom_where": "the kink G = (1.0, 1.39) where the taper meets the upper cylinder",
        "suptitle": lambda P: (
            "LE11: both agents resolve the stress peak at the kink that the automesher blurs;\n"
            f"the trained agent does it with {100 * (1 - P['final']['dofs'] / P['early']['dofs']):.0f} % fewer "
            f"DOFs (finer at the kinks, coarser elsewhere)"),
        "err_title": "Where the error is: the automesher's is largest at the two kinks;\nboth agents shrink it there "
                     "to a few small elements",
        "zoom_note": lambda P: (
            f"The automesher's few large elements there blur the peak. The early agent asked for size "
            f"{size_at(P['early'], (1.0, 1.39)):g} at G on a background of {P['early']['request']['h']:g}; the "
            f"trained agent for {size_at(P['final'], (1.0, 1.39)):g} at G (and "
            f"{size_at(P['final'], (1.2124, 0.7)):g} at the lower kink D, vs {size_at(P['early'], (1.2124, 0.7)):g}) "
            f"on a coarser background of {P['final']['request']['h']:g}, plus {len(P['final']['request']['refinements']) - 2} "
            f"moderate refinements elsewhere (sizes {min(r['size'] for r in P['final']['request']['refinements'][2:]):g}-"
            f"{max(r['size'] for r in P['final']['request']['refinements'][2:]):g}). Both resolve the peak like the true "
            f"solution; the solved fields of panels 2 and 3 look alike (field error {pct(P['early']['field_error'])} "
            f"vs {pct(P['final']['field_error'])}). What changed is the mesh: smaller elements at the kinks, a coarser "
            f"background, {P['final']['dofs']} instead of {P['early']['dofs']} DOFs (the error maps show "
            f"where the error moved)."),
        "graph_note": lambda P, G: (
            f"From generation {G[0]['gen']} to {G[-1]['gen']} the mean field error went from "
            f"{pct(G[0]['mean_field_error'])} to {pct(G[-1]['mean_field_error'])} and the mean DOFs from "
            f"{G[0]['mean_dofs']:.0f} to {G[-1]['mean_dofs']:.0f}: a little more accurate with fewer DOFs. Every "
            f"generation-{G[-1]['gen']} run asks for smaller elements at the two kinks and a coarser background "
            f"than every generation-{G[0]['gen']} run, plus a few moderate refinements elsewhere (requests: "
            f"`data/runs_deep1.csv` -> run files; circled runs: "
            f"`data/{LE11}/panels.json`)."),
        "err_note": lambda P: (
            f"The automesher's error is largest at the two kinks G and D; both agents shrink it there to a few small "
            f"elements. The trained agent's "
            f"larger background elements leave somewhat more error away from the kinks than the early agent's, "
            f"with {P['final']['dofs']} instead of {P['early']['dofs']} DOFs and the same field error "
            f"({pct(P['final']['field_error'])} vs {pct(P['early']['field_error'])})."),
    },
    COOK: {
        "prefix": "A", "name": "Cook's membrane", "data": DATA, "early_tag": "gen0",
        "vm_scale": 1.0, "vm_unit": "",
        "about": "Cook's membrane (plane stress, E = 1, nu = 1/3, unit total shear load; dimensionless stress)",
        "zoom": (0.0, 10.0, 37.0, 47.5), "inset": (25.0, 0.0, 23.0, 21.0),
        "zoom_where": "the clamped top-left corner (0, 44), where the stress peaks",
        "suptitle": lambda P: ("Self-learning beats the automesher: the trained agent's mesh resolves the corner\n"
                               "stress peak that the automesher smears out"),
        "err_title": "Where the error is: training removes it from the stress concentrations",
        "zoom_note": lambda P: (
            f"The automesher's few large elements there smear the peak out; the one-shot agent asked for size "
            f"{P['early']['request']['refinements'][0]['size']:g} at that corner and gets part of it; the trained "
            f"agent asked for size {P['final']['request']['refinements'][0]['size']:g} there (and at the loaded "
            f"top-right corner) and matches the true solution."),
        "err_note": lambda P: "",
    },
}

# style (talk-wide)
GREY, BEST, MEAN, TRUE = "#888888", "#c23d6e", "#2a78d6", "black"
EDGE = "#26313b"                          # dashboard element-edge colour
CMAP = "turbo"                            # dashboard STRESS_SCALE = "Turbo"
ERR_CMAP = "magma_r"                      # dashboard error_map_png colour map
ERR_RANGE = (3e-3, 0.3)                   # element RMS relative von Mises error shown on the error maps
plt.rcParams.update({"font.size": 15, "axes.titlesize": 17, "axes.labelsize": 16, "xtick.labelsize": 14,
                     "ytick.labelsize": 14, "legend.fontsize": 14, "figure.facecolor": "white",
                     "axes.facecolor": "white", "savefig.facecolor": "white",
                     "axes.spines.top": False, "axes.spines.right": False})


def save(fig, stem: str) -> None:
    fig.savefig(OUT / f"{stem}.png", dpi=200, bbox_inches="tight", pad_inches=0.15)
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def pct(x: float) -> str:
    return f"{100 * x:.2f} %"


# --------------------------------------------------------------------------- (0)+(B) data: the run files


def setup_check(problem_id: str, run: dict) -> dict:
    """Re-solve the run's scored mesh with its own setup and with the problem's reference setup
    (ace/field_error.py errors_of_setup, as the scorer). Same field error -> same problem."""
    cp = ckpt.checkpoint(run)
    own = fe.errors_of_setup(problem_id, cp["mesh"], cp["setup"])["field"]
    ref = fe.errors_of_setup(problem_id, cp["mesh"], reference_setup(fea.get_problem(problem_id)))["field"]
    return {"setup_ok": abs(own - ref) <= 1e-6 * ref, "field_error_resolved": own,
            "field_error_with_reference_setup": ref,
            "temperature_load": " ; ".join(str(ld.get("value")) for ld in cp["setup"].get("loads", [])
                                           if ld.get("type") == "temperature")}


def load_runs() -> list[dict]:
    """One row per scored run in RUN_DIR/gen*/runs/*.json (checks written by the scorer) + setup check."""
    rows = []
    for f in sorted(RUN_DIR.glob("gen*/runs/*.json")):
        r = json.loads(f.read_text())
        c = r["checks"]
        rows.append({"gen": int(f.parent.parent.name[3:]), "problem": r["problem_id"], "run_id": r["run_id"],
                     "file": str(f), "valid": c.get("valid"), "base_score": c.get("base_score"),
                     "score_with_bonus": c.get("score"), "gain": c.get("gain"), "rel_error": c.get("rel_error"),
                     "field_error": c.get("field_error"), "probe_error": c.get("probe_error"), "dofs": c.get("dofs"),
                     "auto_error": c.get("auto_error"), "auto_dofs": c.get("auto_dofs"),
                     **setup_check(r["problem_id"], r)})
        assert abs(rows[-1]["field_error_resolved"] - c["field_error"]) <= 1e-6 * c["field_error"], f
    rows.sort(key=lambda d: (d["problem"], d["gen"], d["run_id"]))
    return rows


def per_generation(rows: list[dict], problem: str) -> list[dict]:
    """Best / mean / min of the score relative to the automesher (base_score / 0.25) and of the
    accuracy gain, per generation, over the correct-setup runs only (generations with none are
    left out). base_score is the scorer's accuracy x efficiency without the be-the-best /
    diversity bonuses (ace/checks.py)."""
    out = []
    for g in sorted({d["gen"] for d in rows if d["problem"] == problem}):
        rs = [d for d in rows if d["problem"] == problem and d["gen"] == g and d["setup_ok"]]
        if not rs:
            continue
        x = np.array([d["base_score"] / AUTO_SCORE for d in rs])
        gain = np.array([d["gain"] for d in rs])
        best = rs[int(np.argmax(x))]
        out.append({"gen": g, "n_runs": len(rs), "best_x_auto": x.max(), "mean_x_auto": x.mean(),
                    "min_x_auto": x.min(), "best_run": best["run_id"], "best_gain": gain.max(),
                    "mean_gain": gain.mean(), "min_gain": gain.min(),
                    "mean_field_error": np.mean([d["field_error"] for d in rs]),
                    "mean_dofs": np.mean([d["dofs"] for d in rs])})
    return out


def excluded(rows: list[dict], problem: str) -> list[dict]:
    return [d for d in rows if d["problem"] == problem and not d["setup_ok"]]


def pick_runs(rows: list[dict], problem: str) -> tuple[dict, dict]:
    """Panels 2 and 3: the best run (by base score) of the earliest generation with a correct-setup
    run, and the best correct-setup run of the last generation."""
    ok = [d for d in rows if d["problem"] == problem and d["setup_ok"]]
    first, last = min(d["gen"] for d in ok), max(d["gen"] for d in rows if d["problem"] == problem)
    best = lambda g: max((d for d in ok if d["gen"] == g), key=lambda d: d["base_score"])  # noqa: E731
    return best(first), best(last)


def size_at(panel: dict, point: tuple) -> float:
    """The element size a panel's mesh request asked for at a point refinement."""
    return next(r["size"] for r in panel["request"]["refinements"] if r.get("point") == list(point))


def write_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


# --------------------------------------------------------------------------- (A) data: re-solve the meshes


def sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def automesher_mesh(p, path: Path) -> dict:
    """gmsh with all its default sizes, quadratic: the same steps as ace/field_error.py automesher."""
    import gmsh
    from feacore.geometry.gmsh_tools import ensure_domain_group, gmsh_session
    with gmsh_session(p.id):
        p.build_geometry()
        ensure_domain_group(p.dim)
        gmsh.model.mesh.generate(p.dim)
        gmsh.model.mesh.setOrder(2)
        gmsh.write(str(path))
    return {"path": str(path), "order": 2}


def agent_mesh(p, run: dict, path: Path) -> tuple[dict, dict, dict]:
    """Rebuild the run's scored mesh from its recorded request (ace/fea.py make_mesh) and compare
    it with the .msh the run saved (sha256). Returns (mesh used, checkpoint, rebuild info)."""
    cp = ckpt.checkpoint(run)
    req = cp["request"]
    new = fea.make_mesh(p, path, h=req["h"], order=req.get("order", 1), refinements=req.get("refinements"))
    same = new.get("sha256") == cp["mesh"].get("sha256")
    mesh = new if same else cp["mesh"]     # fall back to the run's own file if gmsh rebuilt it differently
    if not same:
        shutil.copy(cp["mesh"]["path"], path)
        mesh = dict(cp["mesh"], path=str(path))
    return mesh, cp, {"rebuilt_sha_matches_run": same, "run_mesh_file": cp["mesh"]["path"], "request": req}


def vertex_field(mesh: dict, res, scale: float) -> dict:
    """Corner-node triangles of the mesh and the solved von Mises there (the dashboard's run_stress
    probes every mesh point; surface_of keeps the triangles' corner nodes), times scale."""
    m = meshio.read(mesh["path"])
    tris = np.vstack([c.data[:, :3] for c in m.cells if c.type.startswith("triangle")])
    used, tris = np.unique(tris, return_inverse=True)
    pts = m.points[used, :2]
    return {"pts": pts, "tris": tris.reshape(-1, 3), "vm": scale * np.asarray(res.probe("von_mises", pts))}


def solve_panels(pid: str, early: dict, final: dict) -> tuple[dict, dict]:
    """The three solved meshes plus the reference; numbers verified against the run files."""
    case = CASES[pid]
    case["data"].mkdir(parents=True, exist_ok=True)
    p = fea.get_problem(pid)
    qoi = TASKS[pid]["qoi"]
    ref = fe.reference(pid)
    vm_ref = fe.von_mises(ref["sigma"])
    floor = 0.2 * np.median(vm_ref)                      # the report's floor for relative errors
    auto_json = json.loads(Path(f"ace/field_refs/{pid}_auto.json").read_text())
    panels, fields = {}, {}

    def finish(key, title, mesh, setup, stored, extra):
        res = fe._solve(p, mesh, setup, qoi)
        e = fe.errors_of_result(pid, res)                 # field / probe-point / scored error, as the scorer
        f = vertex_field(mesh, res, case["vm_scale"])
        vm_run = fe.von_mises(fe.stresses(res, ref["pts"], p.dim))
        rel = np.abs(vm_run - vm_ref) / np.maximum(vm_ref, floor)            # at the reference points
        # per element of THIS mesh: RMS of the relative error over the reference points inside it
        # (pointwise values ring through zero inside every element; the element RMS reads cleanly)
        el = Triangulation(f["pts"][:, 0], f["pts"][:, 1], f["tris"]).get_trifinder()(*ref["pts"][:, :2].T)
        ok = el >= 0
        n = np.bincount(el[ok], minlength=len(f["tris"]))
        f["el_err"] = np.sqrt(np.bincount(el[ok], rel[ok] ** 2, minlength=len(f["tris"])) / np.maximum(n, 1))
        fields[key] = f
        dofs = int(res.info.n_dofs)
        panels[key] = {"title": title, "dofs": dofs, "nodes": int(len(meshio.read(mesh["path"]).points)),
                       "elements": int(len(f["tris"])), "field_error": e["field"], "probe_error": e["points"],
                       "scored_error": e["error"], "gain_vs_auto": auto_json["error"] / max(e["error"], 0.005),
                       "stored": stored,
                       "matches_stored": abs(e["field"] - stored["field_error"]) <= 1e-6 * stored["field_error"]
                       and dofs == stored["dofs"], **extra}

    # 1. automesher, solved with the problem's correct setup (as ace/field_error.py automesher)
    path = case["data"] / "automesher.msh"
    finish("auto", "gmsh automesher", automesher_mesh(p, path), reference_setup(p),
           {"source": f"ace/field_refs/{pid}_auto.json", "field_error": auto_json["field"],
            "dofs": auto_json["dofs"]}, {"mesh_file": str(path.relative_to(ROOT))})

    # 2./3. agent runs: their scored mesh and setup (ace/checks.py check_run -> errors_of_setup)
    g0 = early["gen"]
    for key, row, title, tag in (
            ("early", early, f"Agent, one shot (generation 0)" if g0 == 0 else f"Early agent (generation {g0})",
             case["early_tag"]),
            ("final", final, f"Trained agent (generation {final['gen']})", "final")):
        run = json.loads(Path(row["file"]).read_text())
        mesh, cp, info = agent_mesh(p, run, case["data"] / f"{tag}.msh")
        c = run["checks"]
        finish(key, title, mesh, cp["setup"],
               {"source": row["file"], "field_error": c["field_error"], "dofs": c["dofs"], "gain": c["gain"],
                "base_score": c["base_score"]},
               {"run_id": row["run_id"], "gen": row["gen"], "setup": cp["setup"], **info})

    # 4. the truth: the fine reference stress on the dashboard's target tessellation
    #    (ace/agent_dashboard.py target_structure: inverse-distance mean of the 4 nearest reference points)
    m = meshio.read(f"ace/field_refs/{pid}_target.msh")
    tris = np.vstack([c.data[:, :3] for c in m.cells if c.type.startswith("triangle")])
    dist, idx = cKDTree(ref["pts"][:, :2]).query(m.points[:, :2], k=4)
    w = 1 / np.maximum(dist, 1e-12)
    fields["ref"] = {"pts": m.points[:, :2], "tris": tris,
                     "vm": case["vm_scale"] * (vm_ref[idx] * w).sum(1) / w.sum(1)}
    rnpz = np.load(f"ace/field_refs/{pid}.npz")
    panels["ref"] = {"title": "True solution (fine reference)", "dofs": int(rnpz["n_dofs"]),
                     "nodes": int(rnpz["nodes"]), "source": f"ace/field_refs/{pid}.npz",
                     "reference_self_error": fe.reference_check(pid)["coarse_vs_fine"],
                     "n_ref_points": int(len(ref["pts"]))}
    return panels, fields


# --------------------------------------------------------------------------- (A) figures

ORDER = ("auto", "early", "final", "ref")


def stems(case: dict) -> dict:
    pre = case["prefix"]
    return {"combined": f"fig_{pre}0_combined", "auto": f"fig_{pre}1_auto",
            "early": f"fig_{pre}2_{case['early_tag']}", "final": f"fig_{pre}3_final", "ref": f"fig_{pre}4_ref",
            "errors": f"fig_{pre}5_error_maps"}


def caption(k: str, P: dict) -> str:
    d = P[k]
    if k == "ref":
        return f"{d['dofs']:,} DOFs"
    g = d["gain_vs_auto"]
    return (f"{d['dofs']} DOFs · field error {pct(d['field_error'])}"
            + ("" if k == "auto" else f" · {g:.2f}x more accurate"))


def stress_norm(fields: dict) -> Normalize:
    """Dashboard colour range: 1st..99th percentile of the reference stress on the target tessellation."""
    v = fields["ref"]["vm"]
    return Normalize(float(np.percentile(v, 1)), float(np.percentile(v, 99)))


def draw_stress(ax, f: dict, norm, edges: bool, lw: float = 0.6):
    tri = Triangulation(f["pts"][:, 0], f["pts"][:, 1], f["tris"])
    pc = ax.tripcolor(tri, f["vm"], shading="gouraud", cmap=CMAP, norm=norm)
    if edges:
        ax.triplot(tri, color=EDGE, lw=lw)
    ax.set_aspect("equal")
    ax.axis("off")
    return pc


def add_zoom(ax, f: dict, norm, edges: bool, case: dict) -> None:
    """Magnified view of the case's zoom box drawn in an empty area of the panel (inset), same
    colours, with a box on the part marking where it comes from. At slide size the four full
    views look alike; the zoom is where the meshes differ visibly."""
    x0, x1, y0, y1 = case["zoom"]
    ix, iy, iw, ih = case["inset"]
    ax.update_datalim([(ix, iy), (ix + iw, iy + ih)])     # keep the inset inside the panel
    ax.autoscale_view()
    ins = ax.inset_axes(case["inset"], transform=ax.transData)
    draw_stress(ins, f, norm, edges, lw=0.8)
    ins.set_xlim(x0, x1)
    ins.set_ylim(y0, y1)
    ins.axis("on")
    ins.set_xticks([])
    ins.set_yticks([])
    for sp in ins.spines.values():
        sp.set_visible(True)
        sp.set_color("black")
        sp.set_linewidth(1.2)
    ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="black", lw=1.2, clip_on=False))
    # dotted leader from the box corner nearest the inset to the inset's nearest corner
    bx = x1 if ix > x0 else x0
    jx = ix if ix > x0 else ix + iw
    ax.annotate("", xy=(jx, iy + ih), xytext=(bx, min(max(iy + ih, y0), y1)), xycoords="data",
                arrowprops=dict(arrowstyle="-", color="black", lw=0.8, ls=":"))


def draw_error(ax, f: dict, norm):
    tri = Triangulation(f["pts"][:, 0], f["pts"][:, 1], f["tris"])
    pc = ax.tripcolor(tri, facecolors=np.clip(f["el_err"], *ERR_RANGE), cmap=ERR_CMAP, norm=norm)
    ax.triplot(tri, color=EDGE, lw=0.5, alpha=0.6)
    ax.set_aspect("equal")
    ax.axis("off")
    return pc


def fig_panels(case: dict, P: dict, F: dict) -> None:
    norm = stress_norm(F)
    S = stems(case)
    vm_label = f"von Mises stress{case['vm_unit']}"
    # combined 16:9 figure, four panels in a row, one colour bar
    fig, axes = plt.subplots(1, 4, figsize=(12, 6.75))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.80, bottom=0.27, wspace=0.06)
    for ax, k in zip(axes, ORDER):
        pc = draw_stress(ax, F[k], norm, edges=k != "ref")
        add_zoom(ax, F[k], norm, edges=k != "ref", case=case)
        ax.set_title(f"{P[k]['title'].replace(' (', chr(10) + '(')}", fontsize=15)
        ax.text(0.5, -0.04, caption(k, P).replace(" · field", "\nfield").replace(" · ", "\n"),
                transform=ax.transAxes, ha="center", va="top", fontsize=14)
    cb = fig.colorbar(pc, cax=fig.add_axes([0.2, 0.06, 0.6, 0.03]), orientation="horizontal", extend="both")
    cb.set_label(f"{vm_label}, same scale on all four; boxes: magnified", fontsize=14)
    fig.suptitle(case["suptitle"](P), fontsize=18, y=0.97)
    save(fig, S["combined"])

    # each panel on its own
    for k in ORDER:
        fig, ax = plt.subplots(figsize=(5.2, 6.4))
        pc = draw_stress(ax, F[k], norm, edges=k != "ref")
        add_zoom(ax, F[k], norm, edges=k != "ref", case=case)
        ax.set_title(P[k]["title"], fontsize=16)
        ax.text(0.5, -0.02, caption(k, P), transform=ax.transAxes, ha="center", va="top", fontsize=14)
        cb = fig.colorbar(pc, ax=ax, orientation="horizontal", fraction=0.05, pad=0.08, extend="both")
        cb.set_label(vm_label, fontsize=14)
        save(fig, S[k])

    # error maps: local relative von Mises error vs the reference, one log scale
    enorm = LogNorm(*ERR_RANGE)
    fig, axes = plt.subplots(1, 3, figsize=(12, 6.75))
    fig.subplots_adjust(left=0.04, right=0.96, top=0.76, bottom=0.27, wspace=0.1)
    for ax, k in zip(axes, ORDER[:3]):
        pc = draw_error(ax, F[k], enorm)
        ax.set_title(P[k]["title"].replace(" (", "\n("), fontsize=15)
        ax.text(0.5, -0.04, caption(k, P).replace(" · field", "\nfield").replace(" · ", "\n"),
                transform=ax.transAxes, ha="center", va="top", fontsize=14)
    cb = fig.colorbar(pc, cax=fig.add_axes([0.2, 0.06, 0.6, 0.03]), orientation="horizontal", extend="both")
    cb.set_label("von Mises error per element vs the true solution (relative RMS, log scale)", fontsize=14)
    fig.suptitle(case["err_title"], fontsize=18, y=0.99)
    save(fig, S["errors"])


# --------------------------------------------------------------------------- (B) figures


def curve(ax, G: list[dict], key: str, shown, ylab: str):
    """shown: None, or the (generation, value) of the early and the final run drawn in (A), circled."""
    g = np.array([d["gen"] for d in G])
    best, mean, lo = (np.array([d[f"{s}_{key}"] for d in G]) for s in ("best", "mean", "min"))
    ax.fill_between(g, lo, best, color=MEAN, alpha=0.12, lw=0, label="all agents (min to max)")
    ax.axhline(1, color=GREY, ls="--", lw=2)
    ax.annotate("gmsh automesher (1x)", (g[-1], 1), xytext=(0, 5), textcoords="offset points",
                color=GREY, ha="right", va="bottom", fontsize=14)
    ax.plot(g, mean, "-o", color=MEAN, lw=2, ms=8, label="mean agent")
    ax.plot(g, best, "-o", color=BEST, lw=2, ms=8, label="best agent")
    if shown:      # the two runs whose meshes are drawn in (A), at their own values
        (ge, y0), (gf, y1) = shown
        ax.plot([ge, gf], [y0, y1], "o", ms=16, mfc="none", mec="black", mew=2)
        ax.annotate(f"{'one shot' if ge == 0 else 'early agent'} (shown): {y0:.1f}x", (ge, y0), xytext=(16, 0),
                    textcoords="offset points", ha="left", va="center", fontsize=13)
        top = (f"best (shown): {y1:.1f}x" if abs(best[-1] - y1) < 1e-9
               else f"best {best[-1]:.1f}x\ntrained run\nshown: {y1:.1f}x")
        ax.annotate(top, (g[-1], best[-1]), xytext=(0, 18),
                    textcoords="offset points", ha="center", va="bottom", fontsize=13)
        ax.annotate(f"mean {mean[-1]:.1f}x", (g[-1], mean[-1]), xytext=(10, -4), textcoords="offset points",
                    va="top", fontsize=13, color="black")
    ax.set_xlabel("generations trained")
    ax.set_ylabel(ylab)
    ax.set_xticks(g)
    ax.set_ylim(0, None)
    ax.set_xlim(g[0] - 0.4, g[-1] + 0.6)


def shown_runs(P: dict, key: str) -> list[tuple[int, float]]:
    val = (lambda s: s["base_score"] / AUTO_SCORE) if key == "x_auto" else (lambda s: s["gain"])
    return [(P[k]["gen"], val(P[k]["stored"])) for k in ("early", "final")]


def fig_graphs(G: dict, PP: dict) -> None:
    ylab = "score relative to the automesher (x)"
    # LE11
    fig, ax = plt.subplots(figsize=(12, 6.75))
    curve(ax, G[LE11], "x_auto", shown_runs(PP[LE11], "x_auto"), ylab)
    ax.set_ylim(0, 4.6)
    ax.legend(loc="upper left", frameon=False)
    g0, g1 = G[LE11][0], G[LE11][-1]
    ax.set_title(f"LE11 axisymmetric: best agent {g0['best_x_auto']:.1f}x → {g1['best_x_auto']:.1f}x the automesher, "
                 f"mean {g0['mean_x_auto']:.1f}x → {g1['mean_x_auto']:.1f}x", loc="left")
    save(fig, "fig_L6_score_vs_generation")

    # Cook
    fig, ax = plt.subplots(figsize=(12, 6.75))
    curve(ax, G[COOK], "x_auto", shown_runs(PP[COOK], "x_auto"), ylab)
    ax.set_ylim(0, 5.6)
    ax.legend(loc="upper left", frameon=False)
    ax.set_title("Self-learning crushes the automesher and keeps improving (Cook's membrane)", loc="left")
    save(fig, "fig_B1_score_vs_generation")

    fig, ax = plt.subplots(figsize=(12, 6.75))
    curve(ax, G[COOK], "gain", shown_runs(PP[COOK], "gain"),
          "accuracy gain over the automesher\n(automesher error / agent error)")
    ax.set_ylim(0, 6.2)
    ax.legend(loc="upper left", frameon=False)
    ax.set_title("Self-learning: 4.7x more accurate than the automesher after 10 generations", loc="left")
    save(fig, "fig_B2_gain_vs_generation")

    # both, shared generation axis
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.75), sharey=True)
    for ax, pid in zip(axes, (COOK, LE11)):
        curve(ax, G[pid], "x_auto", None, ylab if ax is axes[0] else "")
        ax.set_title(CASES[pid]["name"], fontsize=16)
        ax.set_ylim(0, 5.6)
        ax.set_xlim(-0.4, 10.6)
        ax.set_xticks(range(0, 11, 2))
    axes[0].legend(loc="upper left", frameon=False)
    fig.suptitle("Both 2D problems: the mean agent passes 3x the automesher by generation 2", fontsize=17)
    save(fig, "fig_B3_score_vs_generation_both")


# --------------------------------------------------------------------------- captions and page


def exclusion_sentence(rows: list[dict], pid: str) -> str:
    """The plain sentence saying which runs were left out of every panel and graph, and why."""
    ex = excluded(rows, pid)
    if not ex:
        return (f"No run was excluded: every {CASES[pid]['name']} run's setup gives the same solution as "
                f"the problem's own setup.")
    ref_load = next(ld["value"] for ld in reference_setup(fea.get_problem(pid))["loads"])
    gens = sorted({d["gen"] for d in ex})
    loads = sorted({d["temperature_load"] for d in ex})
    return (f"Excluded: {', '.join(d['run_id'] for d in ex)} (generation{'s' if len(gens) > 1 else ''} "
            f"{', '.join(map(str, gens))}), because they applied the temperature {' / '.join(loads)} instead "
            f"of the problem's {ref_load}, a setup mistake rather than meshing (field error "
            f"{pct(min(d['field_error'] for d in ex))}-{pct(max(d['field_error'] for d in ex))}).")


def setup_rule(rows: list[dict], pid: str) -> str:
    n_ok = sum(d["setup_ok"] for d in rows if d["problem"] == pid)
    n = sum(d["problem"] == pid for d in rows)
    return (f"Only correctly set-up runs are used ({n_ok} of {n}): a run counts when its recorded setup, "
            f"re-solved on its own mesh, gives the same field error (to 1e-6) as the problem's own setup "
            f"(`ace/checks.py reference_setup`), so the differences shown are the mesh alone "
            f"(`data/runs_deep1.csv`, column setup_ok).")


def panel_captions(pid: str, P: dict, rows: list[dict], G: list[dict]) -> dict:
    """Captions of the four panels, the combined figure and the error maps of one case."""
    case, S = CASES[pid], stems(CASES[pid])
    a, z, f, r = P["auto"], P["early"], P["final"], P["ref"]
    common = (f"{case['about']}, run `ace_runs/deep1`. Field error = relative energy-norm error of the whole "
              f"stress field vs the fine reference (`{r['source']}`, {r['dofs']:,} DOFs; a coarser reference solve "
              f"differs from it by {pct(r['reference_self_error'])}, so the reference's own error is below that). "
              f"Each mesh was rebuilt from its request and re-solved with the fixed skfem solver as the scorer "
              f"does; the re-solved field errors and DOFs match the stored ones (`{P['_panels_json']}`). "
              f"{exclusion_sentence(rows, pid)}")
    colours = (f"Colours: von Mises at the element corner nodes "
               f"(quadratic elements drawn by their corners, midside values not drawn, as the dashboard does), "
               f"interpolated linearly across each element, Turbo scale clipped to the 1st-99th percentile of the "
               f"reference (as in the agent dashboard), identical on every panel.")
    gain = lambda d: f"{d['gain_vs_auto']:.2f}x the automesher's accuracy"  # noqa: E731
    early_runs = [d for d in rows if d["problem"] == pid and d["gen"] == z["gen"] and d["setup_ok"]]
    fe_range = f"{pct(min(d['field_error'] for d in early_runs))}-{pct(max(d['field_error'] for d in early_runs))}"
    x0, x1, y0, y1 = case["zoom"]
    early_word = "one-shot" if z["gen"] == 0 else "early"
    return {
        S["combined"]: (
            f"{case['name']}: four solutions on one colour scale",
            f"Left to right: gmsh automesher ({a['dofs']} DOFs, field error {pct(a['field_error'])}); best "
            f"generation-{z['gen']} agent `{z['run_id']}` ({z['dofs']} DOFs, {pct(z['field_error'])}, {gain(z)}); best "
            f"generation-{f['gen']} agent `{f['run_id']}` ({f['dofs']} DOFs, {pct(f['field_error'])}, {gain(f)}); true "
            f"solution. {common} {colours} Sources: `{a['stored']['source']}`, `{z['stored']['source']}`, "
            f"`{f['stored']['source']}`.\n\n"
            f"Caveats: the generation-{z['gen']} panel is the BEST of the {len(early_runs)} correctly set-up "
            f"generation-{z['gen']} runs (by score), not the median; their field errors were {fe_range}. The "
            f"trained agent uses {f['dofs'] / a['dofs']:.1f}x the automesher's DOFs (still tiny); the score charges "
            f"for that (efficiency term).\n\n"
            f"Zoom: each panel's box magnifies {case['zoom_where']} (x {x0:g}-{x1:g}, y {y0:g}-{y1:g}; black box "
            f"on the part). Same colour scale. {case['zoom_note'](P)} Requests: `{P['_panels_json']}`."),
        S["auto"]: (f"{case['name']}, panel 1: gmsh automesher",
                    f"gmsh with default sizes, quadratic triangles, the problem's correct setup "
                    f"(ace/field_error.py automesher). {a['dofs']} DOFs, {a['elements']} elements, field error "
                    f"{pct(a['field_error'])} (stored: `{a['stored']['source']}`, {pct(a['stored']['field_error'])}). "
                    f"{common} {colours}"),
        S["early"]: (f"{case['name']}, panel 2: {z['title'][0].lower() + z['title'][1:]}",
                     f"`{z['run_id']}` (`{z['stored']['source']}`): best of the {len(early_runs)} correctly set-up "
                     f"generation-{z['gen']} runs by score (base score {z['stored']['base_score']:.3f} = "
                     f"{z['stored']['base_score'] / AUTO_SCORE:.2f}x the automesher), the {early_word} agent. "
                     f"{z['dofs']} DOFs, {z['elements']} elements, field error {pct(z['field_error'])}, {gain(z)}. "
                     f"Mesh request: `{json.dumps(z['request'])}`. {common} {colours}"),
        S["final"]: (f"{case['name']}, panel 3: trained agent (generation {f['gen']})",
                     f"`{f['run_id']}` (`{f['stored']['source']}`): best run of the last generation by score "
                     f"(base score {f['stored']['base_score']:.3f} = {f['stored']['base_score'] / AUTO_SCORE:.2f}x "
                     f"the automesher; best of the whole run: `{max(G, key=lambda d: d['best_x_auto'])['best_run']}`). "
                     f"{f['dofs']} DOFs, {f['elements']} elements, field error {pct(f['field_error'])}, {gain(f)}. "
                     f"Mesh request: `{json.dumps(f['request'])}`. {common} {colours}"),
        S["ref"]: (f"{case['name']}, panel 4: the true solution",
                   f"The fine reference solve (`{r['source']}`, {r['dofs']:,} DOFs) drawn on the dashboard's target "
                   f"tessellation (`ace/field_refs/{pid}_target.msh`, inverse-distance mean of the 4 nearest "
                   f"reference points, as `ace/agent_dashboard.py target_structure`). No mesh edges: this is the "
                   f"truth, not a mesh anyone submitted. Same Turbo scale as panels 1-3. {common}"),
        S["errors"]: (f"{case['name']}: error maps",
                      f"Per element: RMS over the reference integration points inside it ({r['n_ref_points']} "
                      f"points in all) of the relative von Mises error |vm - vm_ref| / max(vm_ref, 0.2 median(vm_ref)) "
                      f"(the run report's pointwise error, ace/run_report.py _field). Log colour scale clipped to "
                      f"{ERR_RANGE[0]:g}-{ERR_RANGE[1]:g}, identical on the three panels; mesh edges overlaid. Same "
                      f"meshes and numbers as the combined figure. {case['err_note'](P)} {common}".replace("  ", " ")),
    }


def write_captions(PP: dict, G: dict, rows: list[dict]) -> tuple[dict, dict]:
    """Captions per section: {section title: {stem: (title, body)}}, LE11 first. Also writes the .md files."""
    gen = lambda pid, g: next(d for d in G[pid] if d["gen"] == g)  # noqa: E731
    first = lambda pid: G[pid][0]  # noqa: E731
    lc, ll = G[COOK][-1], G[LE11][-1]
    c0, l0 = first(COOK), first(LE11)
    zc, zl = PP[COOK]["early"], PP[LE11]["early"]
    le_dropped = sorted({d["gen"] for d in rows if d["problem"] == LE11} - {d["gen"] for d in G[LE11]})
    le11 = panel_captions(LE11, PP[LE11], rows, G[LE11])
    le11["fig_L6_score_vs_generation"] = (
        "NAFEMS LE11 axisymmetric: score vs generations trained",
        f"Metric: the scorer's base score (accuracy x efficiency, no bonuses; `checks.base_score` in "
        f"`ace_runs/deep1/gen*/runs/*{LE11}*.json`) divided by the automesher's own score 0.25, so 1x = the "
        f"automesher (dashed). Best agent per generation (pink), mean of the agents per generation (blue), "
        f"min-max band. {setup_rule(rows, LE11)} {exclusion_sentence(rows, LE11)} Generation"
        f"{'s' if len(le_dropped) > 1 else ''} {', '.join(map(str, le_dropped))} therefore "
        f"{'have' if len(le_dropped) > 1 else 'has'} no correctly set-up run and is left out. "
        f"Generation {l0['gen']}: best {l0['best_x_auto']:.2f}x, mean {l0['mean_x_auto']:.2f}x "
        f"({l0['n_runs']} runs); generation {ll['gen']}: best {ll['best_x_auto']:.2f}x, mean {ll['mean_x_auto']:.2f}x. "
        f"Circled: the two runs drawn in the panels (`{zl['run_id']}`, `{PP[LE11]['final']['run_id']}`). "
        f"{CASES[LE11]['graph_note'](PP[LE11], G[LE11])} Values: `data/per_generation_{LE11}.csv`.\n\n"
        f"Caveats: one training run (no repeats); 4 runs per generation; agents are reassigned between the two "
        f"problems each generation.")

    cook = panel_captions(COOK, PP[COOK], rows, G[COOK])
    cook["fig_B1_score_vs_generation"] = (
        "Cook's membrane: score vs generations trained",
        f"Metric: the scorer's base score (accuracy x efficiency, no bonuses; `checks.base_score` in "
        f"`ace_runs/deep1/gen*/runs/*{COOK}*.json`) divided by the automesher's own score 0.25, so 1x = "
        f"the automesher. It rewards accuracy and charges for DOFs. Best agent per generation (pink), mean of the "
        f"agents per generation (blue), min-max band. {setup_rule(rows, COOK)} Generation {c0['gen']}: best "
        f"{c0['best_x_auto']:.2f}x, mean {c0['mean_x_auto']:.2f}x; generation {lc['gen']}: best "
        f"{lc['best_x_auto']:.2f}x, mean {lc['mean_x_auto']:.2f}x. Circled points are the runs drawn in the "
        f"panels. Values: `data/per_generation_{COOK}.csv`.\n\n"
        f"Caveats: one training run (no repeats); 4 runs per generation; agents are reassigned between the two "
        f"problems each generation. Chosen over the accuracy gain (fig_B2) as the headline because it is the "
        f"quantity the agents were trained on and includes the DOF cost.")
    final_gain = PP[COOK]["final"]["stored"]["gain"]
    cook_dofs = np.median([d["dofs"] for d in rows if d["problem"] == COOK and d["gen"] == lc["gen"]])
    cook["fig_B2_gain_vs_generation"] = (
        "Cook's membrane: accuracy gain vs generations trained",
        f"Metric: gain = automesher error / agent error (`checks.gain`), ignoring DOFs. Generation {c0['gen']}: best "
        f"{c0['best_gain']:.2f}x, mean {c0['mean_gain']:.2f}x; generation {lc['gen']}: best {lc['best_gain']:.2f}x, "
        f"mean {lc['mean_gain']:.2f}x (title: the mean). Circled: the two runs drawn in the panels at their own gain "
        f"(the generation-{lc['gen']} run shown, `{PP[COOK]['final']['run_id']}`, is the best by score, gain "
        f"{final_gain:.2f}x; the best gain, {lc['best_gain']:.2f}x, is another run). Agents use "
        f"~{cook_dofs / PP[COOK]['auto']['dofs']:.0f}x the automesher's DOFs by the end (median {cook_dofs:.0f} vs "
        f"{PP[COOK]['auto']['dofs']}), so this overstates per-DOF efficiency; fig_B1 is the fair version. "
        f"{setup_rule(rows, COOK)} Values: `data/per_generation_{COOK}.csv`.")
    cook["fig_B3_score_vs_generation_both"] = (
        "Both 2D problems: score vs generations",
        f"As fig_B1 for Cook's membrane (left) and as fig_L6 for NAFEMS LE11 axisymmetric (right), same run "
        f"`ace_runs/deep1`, correctly set-up runs only. {exclusion_sentence(rows, LE11)} So the LE11 curve starts "
        f"at generation {l0['gen']}. Mean agent at generation 2: Cook {gen(COOK, 2)['mean_x_auto']:.2f}x, LE11 "
        f"{gen(LE11, 2)['mean_x_auto']:.2f}x; generation 10: Cook best {lc['best_x_auto']:.2f}x, mean "
        f"{lc['mean_x_auto']:.2f}x; LE11 best {ll['best_x_auto']:.2f}x, mean {ll['mean_x_auto']:.2f}x. "
        f"Values: `data/per_generation_{COOK}.csv`, `data/per_generation_{LE11}.csv`.")
    sections = {CASES[LE11]["name"]: le11, CASES[COOK]["name"]: cook}
    for caps in sections.values():
        for stem, (title, body) in caps.items():
            (OUT / f"{stem}.md").write_text(f"# {title}\n\n{body}\n")
    return sections


def write_index(sections: dict, lead: str) -> None:
    import html
    parts = []
    for name, caps in sections.items():
        parts.append(f'<h2 class="problem">{html.escape(name)}</h2>')
        for stem, (title, body) in caps.items():
            paras = "".join(f"<p>{html.escape(p).replace('`', '')}</p>" for p in body.split("\n\n"))
            parts.append(f'<section id="{stem}"><h3>{html.escape(title)}</h3>'
                         f'<a href="{stem}.png"><img src="{stem}.png" alt="{html.escape(title)}"></a>'
                         f'<div class="cap"><code>{stem}.png / .pdf</code>{paras}</div></section>')
    (OUT / "index.html").write_text(f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Self-learning figures</title>
<style>
:root {{ --bg: #ffffff; --fg: #1d232a; --muted: #5b6672; --card: #f4f5f7; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg: #16191d; --fg: #e8eaed; --muted: #a3acb6; --card: #20252b; }} }}
body {{ background: var(--bg); color: var(--fg); font: 16px/1.5 system-ui, sans-serif; margin: 0; padding: 24px 16px; }}
main {{ max-width: 1200px; margin: 0 auto; }}
h1 {{ font-size: 26px; margin: 0 0 4px; }}
h2.problem {{ font-size: 24px; margin: 36px 0 14px; }}
.lead {{ color: var(--muted); margin: 0 0 28px; }}
section {{ background: var(--card); border-radius: 10px; padding: 18px; margin: 0 0 28px; }}
h3 {{ font-size: 20px; margin: 0 0 12px; }}
img {{ display: block; width: 100%; height: auto; background: #ffffff; border-radius: 6px; }}
section[id$="_auto"] img, section[id$="_gen0"] img, section[id$="_early"] img, section[id$="_final"] img,
section[id$="_ref"] img {{ max-width: 520px; margin: 0 auto; }}
.cap {{ color: var(--muted); font-size: 15px; margin-top: 12px; }}
.cap code {{ color: var(--fg); }}
</style></head><body><main>
<h1>Self-learning figures (draft for critique)</h1>
<p class="lead">{html.escape(lead)} Regenerate with <code>make_figures.py</code>; every number comes from the run files (see captions and <code>data/</code>).</p>
{''.join(parts)}
</main></body></html>
""")


# --------------------------------------------------------------------------- main


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    rows = load_runs()
    write_csv(DATA / "runs_deep1.csv", rows)
    G = {pid: per_generation(rows, pid) for pid in CASES}
    for pid, g in G.items():
        write_csv(DATA / f"per_generation_{pid}.csv", g)
        print(pid, "excluded:", [d["run_id"] for d in excluded(rows, pid)])

    PP = {}
    for pid, case in CASES.items():
        early, final = pick_runs(rows, pid)
        P, F = solve_panels(pid, early, final)
        for k in ("auto", "early", "final"):
            print(f"{pid} {k:6s} dofs {P[k]['dofs']:5d}  field {P[k]['field_error']:.5f} "
                  f"(stored {P[k]['stored']['field_error']:.5f})  gain {P[k]['gain_vs_auto']:.2f}  "
                  f"match {P[k]['matches_stored']}  sha match {P[k].get('rebuilt_sha_matches_run', '-')}")
            assert P[k]["matches_stored"], f"{pid} {k}: re-solved field error / DOFs differ from the run file"
        path = case["data"] / "panels.json"
        P["_panels_json"] = str(path.relative_to(OUT))
        path.write_text(json.dumps(P, indent=1, default=float))
        fig_panels(case, P, F)
        PP[pid] = P

    fig_graphs(G, PP)
    sections = write_captions(PP, G, rows)
    write_index(sections, LEAD)
    for pid in CASES:
        print(pid, [(d["gen"], d["n_runs"], round(d["best_x_auto"], 3), round(d["mean_x_auto"], 3)) for d in G[pid]])


LEAD = ("Run ace_runs/deep1. NAFEMS LE11 axisymmetric first, then Cook's membrane. Only correctly set-up runs "
        "are shown, so every difference is the mesh. NAFEMS LE11 3D (run ace_runs/le11) is not shown: its correctly "
        "set-up agents kept resubmitting the same few meshes from generation 0 to 18, so there is no clear "
        "early-vs-trained difference to draw.")

if __name__ == "__main__":
    main()
