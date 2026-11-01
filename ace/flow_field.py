"""Whole-field accuracy for flow tasks in field mode (ACE_SCORE=field): how far a run's velocity field
is from the true one over the whole domain. The flow counterpart of ace.field_error (stresses);
ace.field_error dispatches here for flow problems, so checks, the agent prompt and the run reports
use one interface.

    field error = ||u_run - u_ref||_L2 / ||u_ref||_L2    (the velocity, over the whole domain)
    point error = RMS relative error of the speed |u| at PROBE_POINTS check points
    error       = the larger of the two (the scored error), as for solids

* **Reference.** The problem's exact solution where it has one (Poiseuille channel, pipe,
  Taylor-Couette). Otherwise its own setup solved on its structured reference mesh (quads / hexes)
  refined REF_LEVEL times; the level twice as coarse estimates the reference's own error
  (reference_check).
* **Quadrature.** The cell centres and volumes of the structured reference mesh at QUAD_LEVEL (the
  midpoint rule). Every run is sampled at the same points, so the comparison does not depend on
  how a run's own mesh is laid out.
* **Sampling a run.** The velocity of the nearest cell centre, reconstructed linearly with that
  cell's gradient (cell value + grad u . offset): second order, as the solver's own interpolation.
* **Check points.** Quadrature points in the bulk of the flow: farther than 20 % of the domain's
  thickness (twice the largest distance to the boundary: a channel's height) from every boundary, where the speed is at least half its median and (numerical
  references) both reference levels agree to 0.1 % (relaxed up to 3 % where too few points do); spread by farthest-point sampling. A run's speed
  there is the solver's own continuous point interpolation (result.probe, as for point QoIs). Near
  walls the speed is small and its relative error is mostly interpolation, so they are left out.
* **The automesher.** The agents' own mesh tool with its defaults at the problem's base size h0:
  uniform triangles / tetrahedra on the CAD, the reference setup (linearUpwind). gmsh's own default
  size gives 36-98 nodes on these domains, too coarse for any flow solution.

Cached in ace.field_error.CACHE (ace/field_refs/): <id>.npz (points, weights, reference velocity),
<id>_coarse.npz, <id>_points.json, <id>_auto.json, as for solids.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from ace import fea

REF_LEVEL = {2: 4, 3: 2}                 # fine numerical reference: the structured mesh at h0 / this
QUAD_LEVEL = {2: 2, 3: 2}                # quadrature: the structured mesh at h0 / this
# the duct: the reference at h0 / 4 (330k hexes; at h0 / 2 the two levels still differ by 4 %), the
# quadrature at h0 (5k points)
LEVEL_OVERRIDE = {"square_duct_3d": {"ref": 4, "quad": 1}}
BULK = 0.2                               # check points: at least this x the domain's thickness from boundaries
SPEED_FLOOR = 0.5                        # check points: speed at least this x the median speed
AGREE = (1e-3, 3e-3, 1e-2, 3e-2)         # check points: the two reference levels agree to this (first that
                                         # leaves 4 x PROBE_POINTS candidates)


def _cache() -> Path:
    from ace.field_error import CACHE
    CACHE.mkdir(exist_ok=True)
    return CACHE


def _levels(problem) -> tuple[int, int]:
    o = LEVEL_OVERRIDE.get(problem.id, {})
    return o.get("ref", REF_LEVEL[problem.dim]), o.get("quad", QUAD_LEVEL[problem.dim])


def has_exact(problem) -> bool:
    """Does the problem give its exact velocity? (Problem.exact returns None unless overridden.)"""
    try:
        with np.errstate(all="ignore"):                    # the probe point may be singular (r = 0)
            v = problem.exact("ux", np.zeros((1, problem.dim)))
    except Exception:  # noqa: BLE001  an exact solution that cannot be evaluated is none
        return False
    return v is not None


# --------------------------------------------------------------------------- sampling


def velocity_at(result, pts: np.ndarray) -> np.ndarray:
    """(n, 3) velocity of a flow result at pts (n, >= dim): the nearest cell's linear reconstruction."""
    dim = result.dim
    C = np.asarray(result.mesh.C)
    tree = result.__dict__.get("_ace_ctree")
    if tree is None:
        tree = result._ace_ctree = cKDTree(C[:, :dim])
    X = np.zeros((len(pts), 3))
    X[:, :dim] = np.asarray(pts)[:, :dim]
    _, i = tree.query(X[:, :dim])
    off = X - C[i]
    if dim == 2:
        off[:, 2] = 0.0
    return np.asarray(result.U)[i] + np.einsum("ni,nij->nj", off, np.asarray(result.gradU)[i])


def check_speeds(result, pts: np.ndarray) -> np.ndarray:
    """The speed at check points by the solver's continuous point interpolation (result.probe, as point
    QoIs are evaluated); a point the probe cannot place falls back to the linear reconstruction."""
    from feacore.case import CaseError
    try:
        return np.asarray(result.probe("u_mag", pts), dtype=float)
    except (CaseError, ValueError):
        out = np.empty(len(pts))
        for k, x in enumerate(pts):
            try:
                out[k] = float(result.probe("u_mag", [x])[0])
            except (CaseError, ValueError):
                out[k] = float(np.linalg.norm(velocity_at(result, np.asarray([x]))[0]))
        return out


def _exact_velocity(problem, pts: np.ndarray) -> np.ndarray:
    P = np.asarray(pts)[:, : problem.dim]
    return np.column_stack([np.broadcast_to(np.asarray(problem.exact(f, P), dtype=float), (len(P),))
                            for f in ("ux", "uy", "uz")])


# --------------------------------------------------------------------------- reference pieces


def _structured(problem, level: int, path: Path) -> Path:
    return problem.mesh(path, h=problem.params["h0"] / level)


def _quadrature(problem, workdir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(points (n, 3), weights (n,), distance to the boundary (n,)) of the structured mesh at QUAD_LEVEL."""
    import meshio

    from feacore.geometry import fv_quality as fq
    _, ql = _levels(problem)
    m = meshio.read(str(_structured(problem, ql, workdir / "quad.msh")))
    want = ("triangle", "quad") if problem.dim == 2 else ("tetra", "hexahedron", "wedge", "pyramid")
    cells = {}
    for c in m.cells:
        if c.type in want:
            cells.setdefault(c.type, []).append(np.asarray(c.data))
    cells = {k: np.unique(np.vstack(v), axis=0) for k, v in cells.items()}
    C, V = fq.centres_volumes(m.points if problem.dim == 3 else m.points[:, :2], cells, problem.dim)
    facet = ("line",) if problem.dim == 2 else ("triangle", "quad")
    bpts = np.vstack([m.points[np.asarray(c.data)].mean(axis=1) for c in m.cells if c.type in facet])
    bdist = cKDTree(bpts[:, : problem.dim]).query(C[:, : problem.dim])[0]
    return C, V, bdist


def _solve_reference(problem, level: int, workdir: Path):
    from feacore.solver import solve
    mesh = _structured(problem, level, workdir / f"ref_{level}.msh")
    return solve(problem.case(str(mesh), order=2), backend="openfoam")


def reference(problem_id: str) -> dict:
    """{"pts", "w", "u" (n, 3), "bdist", "exact", "nodes", "n_dofs"}: the quadrature points, their
    weights and the reference velocity there (cached)."""
    f = _cache() / f"{problem_id}.npz"
    if not f.exists():
        p = fea.get_problem(problem_id)
        with tempfile.TemporaryDirectory() as tmp:
            pts, w, bdist = _quadrature(p, Path(tmp))
            if has_exact(p):
                u, nodes, ndofs, exact = _exact_velocity(p, pts), 0, 0, True
            else:
                res = _solve_reference(p, _levels(p)[0], Path(tmp))
                u, nodes, ndofs, exact = velocity_at(res, pts), res.info.n_nodes, res.info.n_dofs, False
        np.savez_compressed(f, pts=pts, w=w, u=u, bdist=bdist, exact=exact, nodes=nodes, n_dofs=ndofs)
    z = np.load(f)
    return {"pts": z["pts"], "w": z["w"], "u": z["u"], "bdist": z["bdist"], "exact": bool(z["exact"]),
            "nodes": int(z["nodes"]), "n_dofs": int(z["n_dofs"])}


def coarse_reference(problem_id: str) -> dict | None:
    """The reference velocity one level coarser at the same points (None with an exact solution)."""
    p = fea.get_problem(problem_id)
    if has_exact(p):
        return None
    f = _cache() / f"{problem_id}_coarse.npz"
    if not f.exists():
        pts = reference(problem_id)["pts"]
        with tempfile.TemporaryDirectory() as tmp:
            res = _solve_reference(p, max(1, _levels(p)[0] // 2), Path(tmp))
            np.savez_compressed(f, u=velocity_at(res, pts), nodes=res.info.n_nodes)
    z = np.load(f)
    return {"u": z["u"], "nodes": int(z["nodes"])}


def _l2_rel(d: np.ndarray, ref: np.ndarray, w: np.ndarray) -> float:
    return float(np.sqrt(np.sum(w * np.sum(d ** 2, axis=1)) / max(np.sum(w * np.sum(ref ** 2, axis=1)), 1e-300)))


def reference_check(problem_id: str) -> dict:
    """The reference's own error: exact, or the coarser level's field error against the finer one
    (a second-order solution's error is then about a third of it)."""
    ref, coarse = reference(problem_id), coarse_reference(problem_id)
    if coarse is None:
        return {"exact": True, "coarse_vs_fine": 0.0}
    return {"exact": False, "coarse_nodes": coarse["nodes"],
            "coarse_vs_fine": _l2_rel(coarse["u"] - ref["u"], ref["u"], ref["w"])}


def probe_points(problem_id: str) -> dict:
    """PROBE_POINTS check points in the bulk of the flow, with the reference speed there (cached;
    the first PROBE_DISCLOSED are told to the agent)."""
    from ace.config import PROBE_POINTS
    f = _cache() / (f"{problem_id}_points.json" if PROBE_POINTS == 12 else f"{problem_id}_points{PROBE_POINTS}.json")
    if not f.exists():
        p = fea.get_problem(problem_id)
        ref = reference(problem_id)
        pts, speed = ref["pts"], np.linalg.norm(ref["u"], axis=1)
        thickness = 2.0 * float(ref["bdist"].max())        # of the domain: a channel's height, an annulus' gap
        base = (ref["bdist"] >= BULK * thickness) & (speed >= SPEED_FLOOR * np.median(speed))
        coarse = coarse_reference(problem_id)
        ok, agree = base, 0.0
        if coarse is not None:
            # the two reference levels agree there: 0.1 %, relaxed (to 3 %) until enough points remain
            diff = np.abs(np.linalg.norm(coarse["u"], axis=1) - speed) / np.maximum(speed, 1e-300)
            for agree in AGREE:
                ok = base & (diff <= agree)
                if ok.sum() >= 4 * PROBE_POINTS:
                    break
        if ok.sum() < PROBE_POINTS:
            raise RuntimeError(f"{problem_id}: only {int(ok.sum())} candidate check points")
        cand, sp = pts[ok], speed[ok]
        pick = [int(np.argmin(np.linalg.norm(cand - cand.mean(0), axis=1)))]     # start near the centre
        d = np.linalg.norm(cand - cand[pick[0]], axis=1)
        while len(pick) < PROBE_POINTS:
            pick.append(int(np.argmax(d)))
            d = np.minimum(d, np.linalg.norm(cand - cand[pick[-1]], axis=1))
        pick = pick[1:] + pick[:1]          # the centre point last, so the disclosed ones are spread out
        f.write_text(json.dumps({"points": cand[pick][:, : p.dim].tolist(), "speed_ref": sp[pick].tolist(),
                                 "candidates_kept": int(ok.sum()), "candidates": len(pts),
                                 "levels_agree_within": agree}, indent=1))
    return json.loads(f.read_text())


# --------------------------------------------------------------------------- a run's errors


def errors_of_result(problem_id: str, result) -> dict:
    """field: relative L2 error of the velocity over the domain; points: RMS relative speed error at
    the check points; error: the larger (the scored error); point_errors: per check point."""
    ref = reference(problem_id)
    field = _l2_rel(velocity_at(result, ref["pts"]) - ref["u"], ref["u"], ref["w"])
    pp = probe_points(problem_id)
    sp = check_speeds(result, np.array(pp["points"]))
    rel = (sp - np.array(pp["speed_ref"])) / np.array(pp["speed_ref"])
    points = float(np.sqrt(np.mean(rel ** 2)))
    return {"field": field, "points": points, "error": max(field, points), "point_errors": rel.tolist()}


def error_density(problem_id: str, result) -> tuple[np.ndarray, np.ndarray]:
    """(points, w |u_run - u_ref|^2 per point): where a run's velocity error lives (reports)."""
    ref = reference(problem_id)
    d = velocity_at(result, ref["pts"]) - ref["u"]
    return ref["pts"], ref["w"] * np.sum(d ** 2, axis=1)


def automesher(problem_id: str, need_value: bool = False) -> dict:
    """The baseline: the agents' mesh tool with its defaults at the problem's base size h0 (uniform
    triangles / tetrahedra), the reference setup (linearUpwind), scored like a run (cached)."""
    from ace.checks import reference_setup
    from ace.config import TASKS
    f = _cache() / f"{problem_id}_auto.json"
    if not f.exists():
        p = fea.get_problem(problem_id)
        qoi = TASKS[problem_id]["qoi"]
        with tempfile.TemporaryDirectory() as tmp:
            mesh = fea.make_mesh(p, Path(tmp) / "auto.msh", h=p.params["h0"])
            if not mesh.get("ok"):
                raise RuntimeError(f"{problem_id}: automesher mesh failed: {mesh.get('error')}")
            from ace.field_error import _solve
            res = _solve(p, mesh, reference_setup(p), qoi)
            e = errors_of_result(problem_id, res)
            f.write_text(json.dumps({"nodes": mesh["nodes"], "dofs": int(res.info.n_dofs), "h": p.params["h0"],
                                     "error": e["error"], "field": e["field"], "points": e["points"],
                                     "value": float(res.evaluate_all()[qoi])}, indent=1))
    return json.loads(f.read_text())


def precompute(problem_ids: list[str]) -> None:
    import time
    for pid in problem_ids:
        for name, fn in (("reference", reference), ("coarse", coarse_reference), ("points", probe_points),
                         ("automesher", automesher)):
            t = time.perf_counter()
            fn(pid)
            print(f"{pid}: {name} ready ({time.perf_counter() - t:.0f}s)", flush=True)
        print(f"{pid}: reference check {reference_check(pid)}", flush=True)
