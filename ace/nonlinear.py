"""Nonlinear scoring against a real compression test (abc_00000065_compression).

A task with TASKS[pid]["score"] == "nonlinear" is scored by ONE frozen elastoplastic CalculiX ramp of
the run's mesh, compared with the measured force-displacement curve of the physical test. The agents
choose only the mesh; everything else is fixed by physical_tests/<pid>/nonlinear_spec.json:

  supports   base_bottom (f33) held in x, y, z; the prong tops f17 and f34 move with a sticking platen
             (displacement [0, -u_part, 0]), ramped to u_part_max_mm in `increments` equal increments
             (NLGEOM); a run that does not converge is retried once with adaptive increments
  material   J2 plasticity with isotropic hardening from the resin's dogbone test (material_file)
  force      F = -(reaction_y on f17 + reaction_y on f34) at every increment
  crosshead  u_cross = u_part + delta + C * F (seating gap delta in mm, load-train compliance C in
             mm/N, both calibrated once on a fine mesh and frozen); (0, 0) is prepended
  error      e = mean over the checkpoints u_k (crosshead travel, mm) of |F_sim(u_k) - F_meas(u_k)| /
             F_meas(u_k); F_meas from the test's loading branch up to its peak. A checkpoint the run
             never reaches (it stopped early) counts as 100 % error.

The score (ace.checks) compares e and the DOFs with the automesher (gmsh default sizes, straight-sided
P2) run through the same ramp, as in field mode. CalculiX is needed for run_curve and
automesher_baseline (on the Brev box, not on a laptop).
"""

from __future__ import annotations

import functools
import hashlib
import json
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np

from ace.config import TASKS

ROOT = Path(__file__).resolve().parents[1]
SPEC_FILE = ROOT / "physical_tests" / "abc_00000065_compression" / "nonlinear_spec.json"
CACHE = Path(__file__).with_name("field_refs")        # automesher baseline, as ace.field_error

# Fixed pyvista camera (position, focal point, view-up; metres, the part's own coordinates): the
# part (x -30..30, y -44..34, z -8..8 mm) seen from the front (+z), a little from the right and above,
# so the prong tops, both nut pockets and the bore are in view. Same for every run, so frames line up.
NL_CAMERA = [(0.075, 0.045, 0.21), (0.0, -0.005, 0.0), (0.0, 1.0, 0.0)]
PEEQ_CLIM = (0.0, 0.05)        # fixed PEEQ colour range (the dogbone breaks at about 0.043)
CCX_TIMEOUT_S = 1800           # one ramp longer than this (a huge mesh) ends as an invalid run, not a hung one


# --------------------------------------------------------------------------- spec, test data, material


@functools.cache
def spec() -> dict:
    return json.loads(SPEC_FILE.read_text())


def is_nonlinear(problem_id: str) -> bool:
    return TASKS.get(problem_id, {}).get("score") == "nonlinear"


@functools.cache
def _measured() -> dict:
    """The test's loading branch up to its peak force (mm, N), and the measured force at every
    checkpoint (interpolated on that branch)."""
    from ace.evidence import read_table
    s = spec()
    t = read_table(ROOT / s["test_csv"], {"load": "Force", "displacement": "Displacement",
                                         "load_unit": "N", "displacement_unit": "mm"})
    peak = int(np.argmax(t["load"]))
    u, F = t["meas"][: peak + 1] * 1e3, t["load"][: peak + 1]      # read_table gives metres
    order = np.argsort(u, kind="stable")                        # digitized: keep u increasing
    u, F = u[order], F[order]
    ck = list(s["checkpoints_mm"])
    return {"checkpoints_mm": ck, "force_N": [float(f) for f in np.interp(ck, u, F)],
            "curve_mm": u.tolist(), "curve_N": F.tolist()}


def measured() -> dict:
    """{"checkpoints_mm", "force_N" (measured at the checkpoints), "curve_mm", "curve_N"} (a copy)."""
    return json.loads(json.dumps(_measured()))


def _material() -> tuple[float, float, dict]:
    """E (Pa), nu and the J2 plasticity dict from the material file (MPa table)."""
    m = json.loads((ROOT / spec()["material_file"]).read_text())
    tab = m["plastic_table_true_stress_vs_plastic_strain"]        # [[stress MPa, plastic strain], ...]
    plast = {"model": "j2_isotropic", "yield_stress": tab[0][0] * 1e6,
             "hardening": [[eps_p, stress * 1e6] for stress, eps_p in tab[1:]]}
    return m["E"] * 1e6, m["nu"], plast


# --------------------------------------------------------------------------- the frozen case


def build_case(problem, mesh: dict, adaptive: bool = False):
    """The frozen scoring case on `mesh` (dict as in ace.fea: path, order): sticking platen on the
    prong tops ramped to u_part_max, base held, J2 material, NLGEOM; reactions of f17 and f34 as QoIs."""
    from ace import fea
    from feacore.case import QoI
    s = spec()
    E, nu, plast = _material()
    u = s["u_part_max_mm"] * 1e-3

    def stick(group, v):   # every component prescribed: the platen holds the face (friction)
        return {"type": "displacement", "group": group, "components": [0, 1, 2], "value": [0.0, v, 0.0]}

    setup = {"formulation": "3d", "material": {"E": E, "nu": nu},
             "bcs": [stick("f33", 0.0), stick("f17", -u), stick("f34", -u)]}
    case = fea.build_case(problem, mesh, setup, "strain_energy")
    case.qois += [QoI(name=f"Ry_{g}", type="reaction", group=g, component=1) for g in ("f17", "f34")]
    case.materials["mat"].plasticity = plast
    case.analysis.options = {**(case.analysis.options or {}), "nonlinear": True,
                             "increments": s["increments"], "nlgeom": s["nlgeom"], "adaptive": adaptive}
    return case


def _need_ccx() -> None:
    from feacore.solver._ccx_io import CcxError, find_ccx
    try:
        find_ccx()
    except CcxError as e:
        raise RuntimeError("nonlinear scoring needs CalculiX (ccx), which is on the Brev box fe-ref-1, not "
                           f"on this machine: {e}") from e


def _straighten(path: Path) -> None:
    """Rewrite a quadratic tet mesh in place with every midside node at its straight edge midpoint
    (CalculiX refuses curved or misplaced midside nodes)."""
    import gmsh
    from feacore.geometry.gmsh_tools import gmsh_session
    tet10_edges = [(0, 1, 4), (1, 2, 5), (0, 2, 6), (0, 3, 7), (2, 3, 8), (1, 3, 9)]   # gmsh ordering
    with gmsh_session("straighten"):
        gmsh.open(str(path))
        tags, coords, _ = gmsh.model.mesh.getNodes()
        X = dict(zip(tags.tolist(), np.asarray(coords).reshape(-1, 3)))
        _, conn = gmsh.model.mesh.getElementsByType(11)
        conn = np.asarray(conn, dtype=np.int64).reshape(-1, 10)
        done = set()
        for a, b, m in tet10_edges:
            for na, nb, nm in conn[:, [a, b, m]]:
                if nm not in done:
                    done.add(nm)
                    gmsh.model.mesh.setNode(int(nm), list(0.5 * (X[na] + X[nb])), [])
        gmsh.write(str(path))


def _peaks(X: np.ndarray, v: np.ndarray, n: int = 3, sep: float = 8e-3) -> list[int]:
    """Indices of the n largest values of v that are at least `sep` (m) apart."""
    out = []
    for i in np.argsort(-v):
        if len(out) == n or v[i] <= 0:
            break
        if all(np.linalg.norm(X[i] - X[j]) >= sep for j in out):
            out.append(int(i))
    return out


def _hotspots(res) -> list[dict]:
    """Where the part yields most (nodal PEEQ) and is pulled hardest (max principal stress s1) at the
    last increment: up to 3 spots each, with the faces (f<tag>) within 2 mm."""
    m = res.model
    X = m.X
    faces = {g: X[m.nodes_of(g)] for g in m.group_names()
             if g.startswith("f") and g[1:].isdigit() and m.group_dim(g) == m.dim - 1}
    res.use_increment(-1)
    out = []
    for kind, v in (("peeq", res.current.PEEQ), ("s1", res.nodal("s1")[1])):
        for i in _peaks(X, np.asarray(v)):
            near = sorted((g for g, P in faces.items() if np.min(np.linalg.norm(P - X[i], axis=1)) < 2e-3),
                          key=lambda g: int(g[1:]))
            out.append({"kind": kind, "xyz_mm": [round(float(c) * 1e3, 2) for c in X[i]], "faces": near,
                        "value": float(v[i])})
    return out


# --------------------------------------------------------------------------- one scoring run


def _mapped(u_part: np.ndarray, F: np.ndarray) -> np.ndarray:
    """Crosshead travel (mm) of the simulated points: u_part + delta + C * F."""
    s = spec()
    return u_part + s["delta_mm"] + s["compliance_mm_per_N"] * F


def _sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_curve(problem_id: str, mesh: dict, workdir: Path, threads: int = 4, series: bool = True) -> dict:
    """The frozen nonlinear ramp on `mesh`, compared with the test. Writes curve.csv, summary.json
    (the returned dict), series/ (inc_*.vtu, with series), curve.png and peeq.png to `workdir`.
    A summary.json there from the same mesh file (by content) is reused instead of solving again."""
    from ace import fea
    from feacore.solver import solve
    from feacore.solver._ccx_io import CcxError
    from feacore.solver._ccx_mesh import MeshError

    workdir = Path(workdir)
    sha = _sha(mesh["path"])
    prev = workdir / "summary.json"
    if prev.exists():
        old = json.loads(prev.read_text())
        if old.get("mesh_sha256") == sha and old.get("ok") and (old.get("series_dir") or not series):
            return old
    _need_ccx()
    workdir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    problem = fea.get_problem(problem_id)
    s, meas = spec(), measured()
    mesh = dict(mesh)
    out = {"ok": False, "error": None, "nodes": None, "dofs": None, "wall_s": None, "straightened": False,
           "adaptive": False, "mesh_path": mesh["path"], "mesh_sha256": sha}

    res, err = None, None
    for adaptive in (False, True):
        try:
            try:
                res = solve(build_case(problem, mesh, adaptive), backend="ccx", threads=threads,
                            timeout=CCX_TIMEOUT_S)
            except (MeshError, NotImplementedError) as e:
                # CalculiX rejects curved / misplaced midside nodes: straighten a copy once and retry
                if out["straightened"] or "midside" not in str(e):
                    raise
                straight = workdir / "mesh_straight.msh"
                shutil.copyfile(mesh["path"], straight)
                _straighten(straight)
                mesh["path"], out["straightened"] = str(straight), True
                res = solve(build_case(problem, mesh, adaptive), backend="ccx", threads=threads,
                            timeout=CCX_TIMEOUT_S)
        except (CcxError, MeshError, NotImplementedError, ValueError) as e:
            err = f"{type(e).__name__}: {e}"   # keeps the fixed-increment result, if there was one
            if not adaptive and res is None and "no converged increment" in str(e):
                continue                 # the first fixed increment diverged: adaptive increments may pass
            break                        # a mesh or setup problem: adaptive increments do not help
        out["adaptive"] = adaptive
        if res.converged:
            break
        err = "; ".join(res.errors[:3]) or "did not reach the end of the ramp"
    out["wall_s"] = round(time.perf_counter() - t0, 1)
    if res is None:
        out["error"] = err
        prev.write_text(json.dumps(out, indent=1))
        return out

    # force and travel at every increment, the origin first
    u_part, F = [0.0], [0.0]
    for i, inc in enumerate(res.increments):
        res.use_increment(i)
        u_part.append(inc.lam * s["u_part_max_mm"])
        F.append(-(res.reaction("f17", 1) + res.reaction("f34", 1)))
    u_part, F = np.array(u_part), np.array(F)
    u_cross = np.concatenate([[0.0], _mapped(u_part[1:], F[1:])])
    ck = np.array(meas["checkpoints_mm"])
    F_meas = np.array(meas["force_N"])
    F_sim = np.interp(ck, u_cross, F)
    reached = ck <= u_cross[-1]
    rel = np.where(reached, np.abs(F_sim - F_meas) / F_meas, 1.0)   # not reached: 100 %
    last = res.increments[-1]
    peeq_max = float(last.peeq_ip.max()) if last.peeq_ip is not None else float(last.PEEQ.max())
    out.update({
        "ok": True, "error": None if res.converged else f"stopped at u_part {u_part[-1]:.3f} mm: {err}",
        "converged": bool(res.converged),
        "nodes": int(res.model.n_nodes), "dofs": int(res.info.n_dofs),
        "u_part_mm": u_part.round(5).tolist(), "force_N": F.round(3).tolist(), "u_cross_mm": u_cross.round(5).tolist(),
        "checkpoints_mm": ck.tolist(), "force_meas_N": F_meas.round(2).tolist(),
        "force_sim_N": [round(float(f), 2) if r else None for f, r in zip(F_sim, reached)],
        "rel_err": rel.round(5).tolist(), "mean_rel_err": float(rel.mean()),
        "peeq_max": peeq_max, "hotspots": _hotspots(res),
    })
    rows = ["u_part_mm,u_cross_mm,force_N"] + [f"{a:.5f},{b:.5f},{c:.3f}" for a, b, c in zip(u_part, u_cross, F)]
    (workdir / "curve.csv").write_text("\n".join(rows) + "\n")
    out["curve_csv"] = str(workdir / "curve.csv")
    out["series_dir"] = None
    if series:
        res.write_series(workdir / "series")
        out["series_dir"] = str(workdir / "series")
    out["images"] = {"curve": curve_plot([{"label": f"this mesh ({out['nodes']} nodes): "
                                                    f"{out['mean_rel_err']:.1%}",
                                           "u_cross_mm": out["u_cross_mm"], "force_N": out["force_N"]}],
                                         workdir / "curve.png"),
                     "peeq": None}
    if series:
        try:
            out["images"]["peeq"] = render_peeq(out["series_dir"], workdir / "peeq.png")
        except Exception as e:  # noqa: BLE001  no off-screen rendering on this machine: score anyway
            out["render_error"] = f"{type(e).__name__}: {e}"
    out["wall_s"] = round(time.perf_counter() - t0, 1)
    prev.write_text(json.dumps(out, indent=1))
    return out


def automesher_baseline(problem_id: str) -> dict:
    """run_curve on the automesher's mesh (gmsh default sizes, straight-sided P2 so CalculiX accepts
    it). Cached in ace/field_refs/<pid>_nl_auto.json; computing it needs ccx."""
    f = CACHE / f"{problem_id}_nl_auto.json"
    if f.exists():
        return json.loads(f.read_text())
    _need_ccx()
    import gmsh
    from ace import fea
    from feacore.geometry.gmsh_tools import ensure_domain_group, gmsh_session
    p = fea.get_problem(problem_id)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "auto.msh"
        with gmsh_session(p.id):
            p.build_geometry()
            ensure_domain_group(p.dim)
            gmsh.model.mesh.generate(p.dim)
            gmsh.option.setNumber("Mesh.SecondOrderLinear", 1)      # straight midside nodes
            gmsh.model.mesh.setOrder(2)
            gmsh.write(str(path))
        res = run_curve(problem_id, {"path": str(path), "order": 2}, Path(tmp) / "nonlinear", series=False)
    if not res["ok"]:
        raise RuntimeError(f"the automesher's nonlinear run failed: {res['error']}")
    res = {k: v for k, v in res.items() if k not in ("curve_csv", "series_dir", "images", "mesh_path")}
    CACHE.mkdir(exist_ok=True)
    f.write_text(json.dumps(res, indent=1))
    return res


# --------------------------------------------------------------------------- pictures


def curve_plot(curves: list[dict], out_png, title: str | None = None) -> str:
    """Force vs crosshead travel: the measured curve in black with its checkpoints (circles), and each
    curve in `curves` ({label, u_cross_mm, force_N})."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    m = measured()
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    ax.plot(m["curve_mm"], m["curve_N"], color="black", lw=2.2, label="measured (test)")
    ax.plot(m["checkpoints_mm"], m["force_N"], "o", color="black", ms=6, mfc="white", label="checkpoints")
    for c in curves:
        ax.plot(c["u_cross_mm"], c["force_N"], lw=1.4, label=c.get("label"))
    ax.set_xlabel("crosshead travel (mm)")
    ax.set_ylabel("force (N)")
    ax.set_xlim(0, max(m["checkpoints_mm"]) * 1.12)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    if title:
        ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    return str(out_png)


def render_peeq(series_dir, out_png, increment: int = -1, camera=None) -> str:
    """The deformed part (true scale) coloured by nodal PEEQ at one increment of a series
    (inc_000.vtu = undeformed ... ; -1 = the last), off-screen, with the fixed camera NL_CAMERA and
    the fixed colour range PEEQ_CLIM."""
    import pyvista as pv
    files = sorted(Path(series_dir).glob("inc_*.vtu"))
    grid = pv.read(files[increment])
    surf = grid.extract_surface().warp_by_vector("displacement", factor=1.0)
    pl = pv.Plotter(off_screen=True, window_size=(900, 800))
    pl.add_mesh(surf, scalars="PEEQ", clim=PEEQ_CLIM, cmap="inferno", show_edges=False,
                scalar_bar_args={"title": "PEEQ", "fmt": "%.3f"})
    pl.camera_position = camera or NL_CAMERA
    pl.screenshot(str(out_png))
    pl.close()
    return str(out_png)
