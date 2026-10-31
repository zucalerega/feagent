"""Whole-field accuracy: how far a run's stress field is from a fine reference solve, over the whole part.

The point score (ace.checks) looks at one number at one point, so an agent can refine a small ball
around that point and leave the rest of the mesh poor. This measures the error everywhere, in the
energy norm of the stress difference:

    ||d||_E^2 = integral over the part of  d : C^-1 : d,     d = sigma_run - sigma_ref
             = (1/E) integral [ (1 + nu) d:d - nu (tr d)^2 ]          (isotropic C)

    field error = ||sigma_run - sigma_ref||_E / ||sigma_ref||_E

For linear elasticity this is the strain energy of the error (the natural norm of the finite
element method), so it is the quantity a good mesh makes small. Axisymmetric integrals carry the
weight r (2 pi cancels in the ratio); plane-stress thickness cancels too.

The reference is the problem's own (correct) setup on a fine, uniform, quadratic mesh. Its own
error is estimated by comparing two reference levels (reference_check). Integrals use a degree-2
Gauss rule on every reference element; the run's stresses are probed at those points (feacore's
recovered, continuous stress field), so a wrong setup (e.g. a wrong temperature field) shows up as
a large field error just like a coarse mesh does.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path

import meshio
import numpy as np

from ace import checkpoint as ckpt
from ace import fea
from ace.checks import reference_setup
from feacore.solver._skfem_locate import Locator

# Cached references and baselines. ACE_FIELD_REFS points it elsewhere (e.g. files built on a GPU box).
# Per problem: <id>.npz (integration points + fine reference stresses), <id>_coarse.npz (coarser
# reference stresses at the same points), <id>_points.json (probe points), <id>_auto.json (automesher).
CACHE = Path(os.environ.get("ACE_FIELD_REFS") or Path(__file__).with_name("field_refs"))
ERRORS_CACHE: Path | None = None    # where errors_of_setup caches results (None: CACHE / "errors"); tests override

# Element sizes are the problem's default h0 divided by these divisors.
#   QUAD_DIVISOR: the mesh carrying the integration points (moderate, so probing a run stays fast)
#   REF_DIVISORS: the reference solves (fine, then finer); the finer one is the reference, the
#                 other only estimates the reference's own error (reference_check)
QUAD_DIVISOR = {2: 4.5, 3: 2.25}
REF_DIVISORS = {2: (6, 9), 3: (3, 4.5)}

# Zones left out of the integral: (group, distance). A support along a line in 3D makes the stress
# so singular that the energy integral diverges near it (no mesh converges there), so that zone
# measures the idealisation, not the mesh.
EXCLUDE = {"nafems_le10": ("line_outer_midplane", 0.1)}

# How far outside a run's mesh a point may be and still be evaluated (at the nearest point of the
# nearest cell), as a fraction of the bounding-box diagonal; feacore's default is 1%. Coarse tet10
# meshes of curved CAD, with straight-sided cells where curving folded them, cut off thin slivers of
# the part: on abc_00000003 the gmsh-default automesher leaves 139 reference points (0.03% of the
# volume) up to 1.9% outside, which would otherwise make the run unscorable.
LOCATE_TOL = 0.05

DIAG = ("sxx", "syy", "szz")          # in axisymmetric models szz is the hoop stress (feacore)
OFFDIAG = ("sxy", "syz", "sxz")

# Degree-2 Gauss rules in barycentric coordinates (weights sum to 1).
TRI_RULE = (np.array([[2 / 3, 1 / 6, 1 / 6], [1 / 6, 2 / 3, 1 / 6], [1 / 6, 1 / 6, 2 / 3]]), np.full(3, 1 / 3))
_a, _b = 0.5854101966249685, 0.1381966011250105
TET_RULE = (np.array([[_a, _b, _b, _b], [_b, _a, _b, _b], [_b, _b, _a, _b], [_b, _b, _b, _a]]), np.full(4, 0.25))


# --------------------------------------------------------------------------- quadrature on a mesh


def quadrature(msh_path: str | Path, dim: int, axisymmetric: bool) -> tuple[np.ndarray, np.ndarray]:
    """Gauss points (n, 3) and weights (n,) over all cells of a gmsh mesh (corner nodes only,
    so curved quadratic edges are treated as straight: fine on a fine reference mesh)."""
    m = meshio.read(msh_path)
    kind = "triangle" if dim == 2 else "tetra"
    cells = np.vstack([c.data[:, : dim + 1] for c in m.cells if c.type.startswith(kind)])
    X = m.points[cells]                                   # (ne, dim+1, 3)
    E = X[:, 1:, :dim] - X[:, :1, :dim]
    vol = np.abs(np.linalg.det(E)) / (2 if dim == 2 else 6)
    bary, w = TRI_RULE if dim == 2 else TET_RULE
    pts = np.einsum("qk,ekd->eqd", bary, X)              # (ne, nq, 3)
    wts = vol[:, None] * w[None, :]
    if axisymmetric:
        wts = wts * pts[:, :, 0]                          # weight r (x = r)
    return pts.reshape(-1, 3), wts.reshape(-1)


class FastLocator(Locator):
    """feacore's Locator with one change, in the Newton iteration that inverts the map of a curved
    (quadratic) cell: each candidate cell stops iterating once its own Newton step is below 1e-13,
    instead of every candidate iterating until the slowest one has converged. Each point is tried in
    its 16 nearest cells and most of them converge in 2-3 steps, so this does ~3.5x less work for the
    same answer (identical up to round-off, ~1e-13). Locating the 10^5 reference points was ~95% of
    the time to score a 3D run."""

    def _ref(self, e, pts):
        xi = np.einsum("nij,jn->in", self.Ainv[e], pts - self.x0[:, e])   # exact for straight cells
        if self.curved:
            act = np.arange(len(e))                                         # candidates still iterating
            for _ in range(12):
                x, J = self._map(e[act], xi[:, act])
                try:
                    d = np.linalg.solve(J, (x - pts[:, act]).T[..., None])[..., 0].T
                except np.linalg.LinAlgError:
                    break
                xi[:, act] -= d
                act = act[np.max(np.abs(d), axis=0) >= 1e-13]
                if not len(act):
                    break
        return xi


def stresses(result, pts: np.ndarray, dim: int) -> dict:
    """All six stress components at pts. Same as result.probe(k, pts) for each k, but the points are
    located in the mesh once instead of six times (locating is ~95% of probe's cost), with the faster
    FastLocator. Memoised on the result per point set: scoring a run and building its report ask the
    same result for the same points (see shared_solves)."""
    p = pts[:, :dim] if dim == 2 else pts
    key = hashlib.sha1(np.ascontiguousarray(p).tobytes()).hexdigest() + str(p.shape)
    memo = result.__dict__.setdefault("_ace_stresses", {})
    if key in memo:
        return memo[key]
    md = result.model
    if not isinstance(result._locator, FastLocator):
        result._locator = FastLocator(md.mesh, md.md.bbox_diag)
    cells, xi, _ = result._locator.locate(p, tol_out=LOCATE_TOL * md.md.bbox_diag)
    sb = md.sb
    phi = np.stack([sb.elem.lbasis(xi, i)[0] for i in range(sb.element_dofs.shape[0])])
    dofs = sb.element_dofs[:, cells]
    memo[key] = {k: np.sum(phi * result.fields[k][dofs], axis=0) for k in DIAG + OFFDIAG}
    return memo[key]


def energy_sq(s: dict, w: np.ndarray, E: float, nu: float) -> float:
    """integral of s : C^-1 : s, with s given by its components at the Gauss points."""
    ss = sum(s[k] ** 2 for k in DIAG) + 2 * sum(s[k] ** 2 for k in OFFDIAG)
    tr = sum(s[k] for k in DIAG)
    return float(np.sum(w * ((1 + nu) * ss - nu * tr ** 2)) / E)


# --------------------------------------------------------------------------- the reference


_SHARED: dict | None = None     # {(problem, mesh file, order, setup, qoi): Result} inside shared_solves()


@contextmanager
def shared_solves():
    """Inside this block, _solve returns the same Result for the same problem, mesh file (by content),
    order, setup and QoI. The scoring worker wraps one run's check_run and run report in it, so the
    run is solved once and its points located once (stresses memoises on the Result), not twice.
    The solver is deterministic, so the outputs are the same as with two solves."""
    global _SHARED
    _SHARED = {}
    try:
        yield
    finally:
        _SHARED = None


def _solve(problem, mesh: dict, setup: dict, qoi: str):
    from feacore.solver import solve
    key = None
    if _SHARED is not None and Path(mesh.get("path") or "").is_file():
        sha = hashlib.sha256(Path(mesh["path"]).read_bytes()).hexdigest()
        key = json.dumps([problem.id, sha, mesh.get("order"), setup, qoi], sort_keys=True, default=str)
        if key in _SHARED:
            return _SHARED[key]
    res = solve(fea.build_case(problem, mesh, setup, qoi), backend="skfem")
    if key is not None:
        _SHARED[key] = res
    return res


def _reference_solve(problem_id: str, divisor: float, workdir: Path):
    """The problem's own (correct) setup solved on a uniform quadratic mesh of size h0 / divisor."""
    from ace.config import TASKS
    p = fea.get_problem(problem_id)
    mesh = fea.make_mesh(p, workdir / f"ref_{divisor:g}.msh", h=p.params["h0"] / divisor, order=2)
    if not mesh.get("ok"):
        raise RuntimeError(f"reference mesh failed: {mesh.get('error')}")
    return _solve(p, mesh, reference_setup(p), TASKS[problem_id]["qoi"]), mesh


def _integration_points(problem_id: str, workdir: Path) -> tuple[np.ndarray, np.ndarray]:
    p = fea.get_problem(problem_id)
    mesh = fea.make_mesh(p, workdir / "quad.msh", h=p.params["h0"] / QUAD_DIVISOR[p.dim], order=1)
    pts, w = quadrature(mesh["path"], p.dim, reference_setup(p)["formulation"] == "axisymmetric")
    if problem_id in EXCLUDE:
        from scipy.spatial import cKDTree
        group, dist = EXCLUDE[problem_id]
        m = meshio.read(mesh["path"])
        nodes = np.unique(np.concatenate([m.cells_dict[t][i].ravel() for t, i in m.cell_sets_dict[group].items()]))
        keep = cKDTree(m.points[nodes]).query(pts)[0] > dist
        pts, w = pts[keep], w[keep]
    return pts, w


def _needs_uniform_reference(p) -> None:
    """The uniform reference needs the problem's default size h0. Problems without one (the gold CAD
    parts) get an adaptive reference built separately into the cache (ace.gold_reference)."""
    if "h0" not in p.params:
        raise RuntimeError(f"{p.id}: no cached reference in {CACHE} and no h0 for a uniform one; "
                           f"build it with: python -m ace.gold_reference {p.id}")


def reference(problem_id: str) -> dict:
    """Integration points, weights and reference stresses there (cached in ace/field_refs/)."""
    CACHE.mkdir(exist_ok=True)
    f = CACHE / f"{problem_id}.npz"
    if not f.exists():
        p = fea.get_problem(problem_id)
        _needs_uniform_reference(p)
        with tempfile.TemporaryDirectory() as tmp:
            pts, w = _integration_points(problem_id, Path(tmp))
            res, mesh = _reference_solve(problem_id, REF_DIVISORS[p.dim][1], Path(tmp))
            sigma = stresses(res, pts, p.dim)
        np.savez_compressed(f, pts=pts, w=w, nodes=mesh["nodes"], n_dofs=res.info.n_dofs,
                            **{f"s_{k}": v for k, v in sigma.items()})
    z = np.load(f)
    return {"pts": z["pts"], "w": z["w"], "nodes": int(z["nodes"]), "n_dofs": int(z["n_dofs"]),
            "sigma": {k: z[f"s_{k}"] for k in DIAG + OFFDIAG}}


def automesher(problem_id: str, need_value: bool = False) -> dict:
    """The baseline every run is scored against: gmsh with all its default sizes (none set) and
    quadratic elements, solved with the problem's correct setup. Cached in ace/field_refs/."""
    import json
    import gmsh
    from feacore.geometry.gmsh_tools import ensure_domain_group, gmsh_session
    from ace.config import TASKS
    CACHE.mkdir(exist_ok=True)
    f = CACHE / f"{problem_id}_auto.json"
    # need_value: the scored quantity of this solve too (ace.evidence compares it with a physical
    # test); entries cached before it was stored are recomputed.
    if not f.exists() or (need_value and "value" not in json.loads(f.read_text())):
        p = fea.get_problem(problem_id)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "auto.msh"
            with gmsh_session(p.id):
                p.build_geometry()
                ensure_domain_group(p.dim)
                gmsh.model.mesh.generate(p.dim)
                gmsh.model.mesh.setOrder(2)
                if "gold" in p.tags or "physical" in p.tags:    # physical tests reuse gold CAD
                    _unfold_tet10()
                gmsh.write(str(path))
            res = _solve(p, {"path": str(path), "order": 2}, reference_setup(p), TASKS[problem_id]["qoi"])
            e = errors_of_result(problem_id, res)
            f.write_text(json.dumps({"nodes": len(meshio.read(path).points), "dofs": int(res.info.n_dofs),
                                     "error": e["error"], "field": e["field"], "points": e["points"],
                                     "value": float(res.evaluate_all()[TASKS[problem_id]["qoi"]])}, indent=1))
    return json.loads(f.read_text())


def _unfold_tet10(sj_min: float = 0.05) -> None:
    """Gold CAD parts: gmsh's default coarse tet10 mesh, curved onto the CAD, has a few dozen folded
    cells (negative scaled Jacobian; abc_00000003 / abc_00000065: 35 / 38 of them). The solver accepts
    them but their stresses are garbage, so cells below sj_min (as the gold generator) are made
    straight-sided (midside nodes at edge midpoints; this fixed every one on both parts), and if any
    still fold the whole mesh becomes straight-sided order 2 (cannot fold where the linear tets are valid)."""
    import gmsh
    edges = [(0, 1, 4), (1, 2, 5), (0, 2, 6), (0, 3, 7), (2, 3, 8), (1, 3, 9)]   # gmsh tet10 ordering

    def folded():
        tags, _ = gmsh.model.mesh.getElementsByType(11)
        return np.asarray(tags)[np.asarray(gmsh.model.mesh.getElementQualities(tags, "minSJ")) < sj_min]

    def straighten(cells):
        for c in cells:
            nodes = gmsh.model.mesh.getElement(int(c))[1]
            X = {int(n): np.asarray(gmsh.model.mesh.getNode(int(n))[0]) for n in nodes[:4]}
            for a, b, m in edges:
                gmsh.model.mesh.setNode(int(nodes[m]), (0.5 * (X[int(nodes[a])] + X[int(nodes[b])])).tolist(), [])

    for _ in range(5):
        bad = folded()
        if not len(bad):
            return
        straighten(bad)
    if len(folded()):
        straighten(gmsh.model.mesh.getElementsByType(11)[0])


def coarse_reference(problem_id: str) -> dict:
    """Stresses of the coarser reference level at the same integration points (cached). Used to
    judge the reference itself (reference_check) and where it can be trusted (probe_points)."""
    CACHE.mkdir(exist_ok=True)
    f = CACHE / f"{problem_id}_coarse.npz"
    if not f.exists():
        p = fea.get_problem(problem_id)
        _needs_uniform_reference(p)
        pts = reference(problem_id)["pts"]
        with tempfile.TemporaryDirectory() as tmp:
            res, mesh = _reference_solve(problem_id, REF_DIVISORS[p.dim][0], Path(tmp))
            sigma = stresses(res, pts, p.dim)
        np.savez_compressed(f, nodes=mesh["nodes"], **{f"s_{k}": v for k, v in sigma.items()})
    z = np.load(f)
    return {"nodes": int(z["nodes"]), "sigma": {k: z[f"s_{k}"] for k in DIAG + OFFDIAG}}


def reference_check(problem_id: str) -> dict:
    """The reference's own error, estimated as the field error of the coarser reference solve
    measured against the finer one. On a smooth problem the quadratic-element energy error falls
    like h^2, so the reference's error is about this x (REF_DIVISORS ratio)^-2."""
    mat = reference_setup(fea.get_problem(problem_id))["material"]
    ref, coarse = reference(problem_id), coarse_reference(problem_id)
    diff = {k: coarse["sigma"][k] - ref["sigma"][k] for k in ref["sigma"]}
    err = np.sqrt(energy_sq(diff, ref["w"], mat["E"], mat["nu"]) / energy_sq(ref["sigma"], ref["w"], mat["E"], mat["nu"]))
    return {"coarse_nodes": coarse["nodes"], "coarse_vs_fine": float(err)}


# --------------------------------------------------------------------------- a run's errors


def von_mises(s: dict) -> np.ndarray:
    xx, yy, zz = (s[k] for k in DIAG)
    xy, yz, xz = (s[k] for k in OFFDIAG)
    return np.sqrt(0.5 * ((xx - yy) ** 2 + (yy - zz) ** 2 + (zz - xx) ** 2) + 3 * (xy ** 2 + yz ** 2 + xz ** 2))


def _geometry_vertices(problem) -> np.ndarray:
    """Coordinates of the CAD vertices (gmsh points) of the problem's geometry."""
    import gmsh
    from feacore.geometry.gmsh_tools import gmsh_session
    with gmsh_session(problem.id):
        problem.build_geometry()
        return np.array([gmsh.model.getValue(0, t, []) for _, t in gmsh.model.getEntities(0)])


def _geometry_edges(problem, n: int = 400) -> np.ndarray:
    """Points sampled along every CAD edge (gmsh curve) of the problem's geometry, n per curve."""
    import gmsh
    from feacore.geometry.gmsh_tools import gmsh_session
    with gmsh_session(problem.id):
        problem.build_geometry()
        out = []
        for _, c in gmsh.model.getEntities(1):
            lo, hi = gmsh.model.getParametrizationBounds(1, c)
            out.append(np.reshape(gmsh.model.getValue(1, c, np.linspace(lo[0], hi[0], n).tolist()), (-1, 3)))
        return np.vstack(out)


def probe_points(problem_id: str) -> dict:
    """PROBE_POINTS points spread over the part where the reference stress is trustworthy, with the
    reference von Mises stress there. Candidates are the integration points; a candidate is kept
    only if it is farther than 10% of the part's size from every geometric corner (where stresses
    are singular or concentrated; gold CAD parts: also 5% from every CAD edge), the two reference
    levels agree on its von Mises stress to 0.1%, and its stress is not near zero (where a relative error means little). From those,
    farthest-point sampling spreads the points out; the first PROBE_DISCLOSED are told to the agent."""
    import json
    from ace.config import PROBE_POINTS
    CACHE.mkdir(exist_ok=True)
    # The default 12 points keep their original file name; any other count gets its own file.
    f = CACHE / (f"{problem_id}_points.json" if PROBE_POINTS == 12 else f"{problem_id}_points{PROBE_POINTS}.json")
    if not f.exists():
        p = fea.get_problem(problem_id)
        ref = reference(problem_id)          # fine level: already solved and cached, not re-solved
        cand = ref["pts"]
        vf, vc = von_mises(ref["sigma"]), von_mises(coarse_reference(problem_id)["sigma"])
        corners = _geometry_vertices(p)
        diag = float(np.linalg.norm(np.ptp(cand, axis=0)))
        from scipy.spatial import cKDTree
        far = cKDTree(corners).query(cand[:, : corners.shape[1]])[0] > 0.1 * diag
        ok = (np.abs(vc - vf) <= 0.001 * vf) & (vf >= 0.2 * np.median(vf)) & far
        if "gold" in p.tags:     # CAD parts: also away from every CAD edge (the singular re-entrant / clamped
            ok &= cKDTree(_geometry_edges(p)).query(cand)[0] > 0.05 * diag    # edges; on a stand-in reference
            # (L1 vs L0) 4 of 12 points otherwise landed within 0.3-2% of the diagonal from an edge)
        pts, vm = cand[ok], vf[ok]
        pick = [int(np.argmin(np.linalg.norm(pts - pts.mean(0), axis=1)))]     # start near the centre
        d = np.linalg.norm(pts - pts[pick[0]], axis=1)
        while len(pick) < PROBE_POINTS:
            pick.append(int(np.argmax(d)))
            d = np.minimum(d, np.linalg.norm(pts - pts[pick[-1]], axis=1))
        pick = pick[1:] + pick[:1]          # the centre point last, so the disclosed ones are spread out
        f.write_text(json.dumps({"points": pts[pick].tolist(), "vm_ref": vm[pick].tolist(),
                                 "candidates_kept": int(ok.sum()), "candidates": len(cand)}, indent=1))
    return json.loads(f.read_text())


def errors_of_result(problem_id: str, result) -> dict:
    """field: relative energy-norm error of the whole stress field; points: RMS relative error of
    von Mises stress at the probe points; error: the larger of the two (the scored error)."""
    p = fea.get_problem(problem_id)
    mat = reference_setup(p)["material"]
    ref = reference(problem_id)
    s = stresses(result, ref["pts"], p.dim)
    diff = {k: s[k] - ref["sigma"][k] for k in s}
    field = float(np.sqrt(energy_sq(diff, ref["w"], mat["E"], mat["nu"])
                          / energy_sq(ref["sigma"], ref["w"], mat["E"], mat["nu"])))
    pp = probe_points(problem_id)
    vm = von_mises(stresses(result, np.array(pp["points"]), p.dim))
    rel = (vm - np.array(pp["vm_ref"])) / np.array(pp["vm_ref"])
    points = float(np.sqrt(np.mean(rel ** 2)))
    return {"field": field, "points": points, "error": max(field, points), "point_errors": rel.tolist()}


def field_error_of_result(problem_id: str, result) -> float:
    return errors_of_result(problem_id, result)["field"]


def errors_of_setup(problem_id: str, mesh: dict, setup: dict) -> dict:
    """Solve (mesh, setup) and return its errors (the solver is deterministic, so this is the same
    solution the agent got). Cached in CACHE/errors/ (opt out with ACE_NO_ERROR_CACHE=1)."""
    import json
    from ace.config import TASKS
    f = None if os.environ.get("ACE_NO_ERROR_CACHE") == "1" else _errors_cache_file(problem_id, mesh, setup)
    if f is not None and f.exists():
        return json.loads(f.read_text())
    res = _solve(fea.get_problem(problem_id), mesh, setup, TASKS[problem_id]["qoi"])
    out = errors_of_result(problem_id, res)
    if f is not None:                       # all plain floats: the JSON round trip is exact
        f.parent.mkdir(parents=True, exist_ok=True)
        tmp = f.with_suffix(f".{os.getpid()}.tmp")     # atomic: parallel workers may write the same entry
        tmp.write_text(json.dumps(out))
        os.replace(tmp, f)
    return out


def _errors_cache_file(problem_id: str, mesh: dict, setup: dict) -> Path | None:
    """Cache entry of errors_of_setup, keyed by everything the errors depend on: the mesh (its
    sha256), the setup, the probe-point counts and the reference (its file's mtime). None if the
    mesh cannot be fingerprinted."""
    import hashlib
    import json
    from ace.config import PROBE_DISCLOSED, PROBE_POINTS
    sha = mesh.get("sha256")
    if not sha:
        path = Path(mesh.get("path") or "")
        if not path.is_file():
            return None
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
    ref = CACHE / f"{problem_id}.npz"
    if not ref.exists():
        reference(problem_id)                 # build it first, so its mtime is the one used below
    points = probe_points(problem_id)       # the check points themselves: a rebuilt file means new entries
    key = json.dumps([problem_id, sha, mesh.get("order"), setup, PROBE_POINTS, PROBE_DISCLOSED, ref.stat().st_mtime_ns,
                      points["points"]], sort_keys=True, default=str)
    return (ERRORS_CACHE or CACHE / "errors") / f"{hashlib.sha256(key.encode()).hexdigest()}.json"


def field_error(run: dict) -> dict | None:
    """Errors of the run's scored solve (re-solved from its checkpoint: same mesh, same setup)."""
    cp = ckpt.checkpoint(run)
    return None if cp is None else errors_of_setup(run["problem_id"], cp["mesh"], cp["setup"])


# --------------------------------------------------------------------------- scoring a whole run directory


def _one(path: str) -> tuple[str, float | None, str | None]:
    import json
    run = json.load(open(path))
    try:
        e = field_error(run)
        return run["run_id"], None if e is None else e["error"], None
    except Exception as e:  # noqa: BLE001  report and continue
        return run["run_id"], None, f"{type(e).__name__}: {e}"


def precompute(problem_ids: list[str]) -> None:
    """Build every cached piece for these problems (run it in the background before a run)."""
    import time
    for pid in problem_ids:
        for name, fn in (("reference", reference), ("coarse", coarse_reference), ("points", probe_points),
                         ("automesher", automesher)):
            t = time.perf_counter()
            fn(pid)
            print(f"{pid}: {name} ready ({time.perf_counter() - t:.0f}s)", flush=True)
        print(f"{pid}: reference check {reference_check(pid)}", flush=True)


def main() -> None:
    """python -m ace.field_error ace_runs/train1 [--workers 6]  ->  <run dir>/field_errors.json
    python -m ace.field_error --precompute nafems_le1,cooks_membrane  ->  build the caches"""
    import argparse
    import json
    from concurrent.futures import ProcessPoolExecutor

    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", nargs="?")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--precompute", help="comma-separated problem ids: build their caches and exit")
    a = ap.parse_args()
    if a.precompute:
        precompute([x for x in a.precompute.split(",") if x])
        return
    paths = sorted(str(p) for p in Path(a.run_dir).glob("gen*/runs/*.json"))
    for pid in {json.load(open(p))["problem_id"] for p in paths}:
        reference(pid)                                    # build references once, before forking
    out = {}
    with ProcessPoolExecutor(a.workers) as ex:
        for i, (rid, err, msg) in enumerate(ex.map(_one, paths, chunksize=4)):
            out[rid] = {"field_error": err, "error": msg}
            if i % 50 == 0:
                print(f"{i}/{len(paths)}", flush=True)
    (Path(a.run_dir) / "field_errors.json").write_text(json.dumps(out, indent=1))
    print(f"wrote {Path(a.run_dir) / 'field_errors.json'}: {sum(v['field_error'] is not None for v in out.values())}"
          f"/{len(out)} runs")


if __name__ == "__main__":
    main()
