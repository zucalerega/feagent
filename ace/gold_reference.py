"""Adaptive reference fields for the gold benchmarks, for field-mode scoring (ace.field_error).

The gold sets certify QoIs only, no fields, and their finest study mesh L2 is not accurate enough
to serve as a field reference: re-entrant and clamped edges converge slowly, so on L2 the relative
energy-norm error is 4-15%. Here the reference mesh is refined where the error is, starting from L2:

  1. solve on the current mesh (P2) and estimate the error of every cell (ZZ estimator);
  2. remesh the STEP with gmsh, element sizes from that estimate (`new_sizes`);
  3. repeat until the energy-norm error is below `target` or the DOF budget is spent.

The error is measured against the certified strain energy U_ref. Under loads with fixed supports,
the Galerkin solution satisfies U - U_h = ||u - u_h||_E^2 / 2 and U = ||u||_E^2 / 2 (Pythagoras in
the energy inner product), so

    sqrt(|U_ref - U_h| / U_ref) = relative energy-norm error of the whole field

(and U_h <= U: the solves approach U_ref from below). U_ref is itself an extrapolation (its GCI is
about 0.5% of U), so an energy-norm error well below sqrt(GCI) ~ 7% cannot be confirmed by this
measure alone; the log therefore also records the ZZ estimate of every level.

Outputs, in the format ace.field_error reads (so reference, coarse_reference, reference_check,
probe_points and automesher work unchanged):
  <cache>/<id>.npz           integration points + weights (degree-2 tet rule on the L1 study mesh)
                             and the finest level's stresses there
  <cache>/<id>_coarse.npz    the previous level's stresses at the same points
  <cache>/<id>_adaptive.json one entry per level: DOFs, strain energy, errors, wall time
The meshes of every level stay in `out_dir`.

    python -m ace.gold_reference abc_00000003_bc7 --out work/gold_ref [--max-dofs 4500000]
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

P, D = 2, 3                 # element order, space dimension
SHRINK, GROW = 0.25, 2.0    # per level a cell's size changes by at most these factors (keeps gradation mild)
SJ_MIN = 0.05               # min scaled Jacobian a curved cell must reach (as in feacore.goldgen.mesh)
REPAIR_TARGET = 0.05        # linear-tet quality the gmsh optimizer is asked to reach before curving
MESH_TIMEOUT = 1800.0       # seconds per meshing attempt (a 1M-cell tet10 mesh takes a few minutes)
# Meshing attempts, in order: curved (boundary midside nodes on the CAD) with gmsh's HXT, then its
# Delaunay mesher (HXT fails on some CAD), then straight-sided order 2, which cannot fold.
ATTEMPTS = (("curved", 10), ("curved", 1), ("straight", 10), ("straight", 1))


# --------------------------------------------------------------------------- sizes from the error


def _cell_size(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Edge length of the regular tet with the same volume as each corner tet X (ne, 4, 3), and the volume."""
    vol = np.abs(np.linalg.det(X[:, 1:] - X[:, :1])) / 6
    return (6 * np.sqrt(2) * vol) ** (1 / 3), vol


def new_sizes(X: np.ndarray, eta2: np.ndarray, dofs_now: int, dofs_next: float) -> np.ndarray:
    """Target element size per cell for a mesh of about dofs_next DOFs.

    A cell's squared error contribution scales like eta_K^2 ~ h^(2p + d) (error density h^2p times
    volume h^d), so a cell of size h_new has eta_K (h_new / h_old)^(p + d/2). The best mesh for a
    given DOF count spreads the error evenly (equidistribution): every new cell gets the same eta*,

        h_new = h_old (eta* / eta_K)^(1 / (p + d/2)),

    with eta* chosen so that the predicted DOF count, dofs_now x sum(vol / h_new^3) / sum(vol / h_old^3),
    is dofs_next. Near the singular edges the true rate is lower than p, so one level under-refines
    there; the next level sees the remaining error and refines again."""
    h, vol = _cell_size(X)
    eta = np.sqrt(np.maximum(eta2, 1e-30 * eta2.max()))
    density = np.sum(vol / h ** 3)

    def sizes(log_eta_star: float) -> np.ndarray:
        return h * np.clip((np.exp(log_eta_star) / eta) ** (1 / (P + D / 2)), SHRINK, GROW)

    lo, hi = np.log(eta.min()) - 30, np.log(eta.max()) + 30
    for _ in range(100):                     # predicted DOFs fall as eta* grows: bisection
        mid = 0.5 * (lo + hi)
        if dofs_now * np.sum(vol / sizes(mid) ** 3) / density > dofs_next:
            lo = mid
        else:
            hi = mid
    return sizes(hi)


# --------------------------------------------------------------------------- meshing (gmsh, in a subprocess)


def _straighten(cells) -> None:
    """Move the midside nodes of these tet10 cells to their edge midpoints (from feacore.goldgen.mesh)."""
    import gmsh
    edges = [(0, 1, 4), (1, 2, 5), (0, 2, 6), (0, 3, 7), (2, 3, 8), (1, 3, 9)]   # gmsh tet10 ordering
    for c in cells:
        _, nodes, _, _ = gmsh.model.mesh.getElement(int(c))
        X = {int(n): np.asarray(gmsh.model.mesh.getNode(int(n))[0]) for n in nodes[:4]}
        for a, b, m in edges:
            gmsh.model.mesh.setNode(int(nodes[m]), (0.5 * (X[int(nodes[a])] + X[int(nodes[b])])).tolist(), [])


def _folded():
    """Tet10 cells that are folded or nearly so (min scaled Jacobian below SJ_MIN)."""
    import gmsh
    tags, _ = gmsh.model.mesh.getElementsByType(11)
    q = np.asarray(gmsh.model.mesh.getElementQualities(tags, "minSJ"))
    return np.asarray(tags)[q < SJ_MIN], float(q.min())


def mesh_worker(problem_id: str, sizes_file: str, out: str, scale: float, scheme: str, alg: int) -> dict:
    """Mesh the part with element sizes from sizes_file (x scale), as tet10. Runs in its own process:
    gmsh can abort the whole process on some CAD."""
    import gmsh
    from feacore.geometry.gmsh_tools import gmsh_session
    from feacore.problems import registry
    p = registry.get(problem_id)
    z = np.load(sizes_file)
    T = z["t"]
    X, h = z["X"][T], z["h"][T] * scale                     # (ne, 4, 3), (ne, 4)
    t0 = time.perf_counter()
    with gmsh_session(p.id):
        p.build_geometry()                                   # the STEP in metres, faces f<tag>, volume solid
        # The size field: the previous mesh's tets with a size at each corner, interpolated linearly.
        view = gmsh.view.add("size")
        gmsh.view.addListData(view, "SS", len(T), np.hstack([X[:, :, 0], X[:, :, 1], X[:, :, 2], h]).ravel())
        f = gmsh.model.mesh.field.add("PostView")
        gmsh.model.mesh.field.setNumber(f, "ViewTag", view)
        gmsh.model.mesh.field.setAsBackgroundMesh(f)
        opt = gmsh.option.setNumber
        for k in ("MeshSizeFromPoints", "MeshSizeFromCurvature", "MeshSizeExtendFromBoundary"):
            opt(f"Mesh.{k}", 0)                              # sizes from the field only
        opt("Mesh.MeshSizeMin", float(h.min()))
        opt("Mesh.MeshSizeMax", float(h.max()))              # also where the field is undefined (just outside it)
        opt("Mesh.Algorithm3D", alg)
        opt("General.NumThreads", 0)
        opt("Mesh.SecondOrderLinear", 1 if scheme == "straight" else 0)
        gmsh.model.mesh.generate(3)
        # Snapping boundary vertices onto curved CAD flattens a few tets; repair them before curving.
        for _ in range(3):
            tags, _ = gmsh.model.mesh.getElementsByType(4)
            if np.min(gmsh.model.mesh.getElementQualities(tags, "gamma")) >= REPAIR_TARGET:
                break
            gmsh.model.mesh.optimize("", niter=3)
            gmsh.model.mesh.optimize("Relocate3D", niter=3)
        gmsh.model.mesh.setOrder(2)
        bad, _ = _folded()
        folded = len(bad)
        for _ in range(5):          # curved cells that fold lose their curvature (cheap and local)
            if not len(bad):
                break
            _straighten(bad)
            bad, _ = _folded()
        if len(bad):
            raise RuntimeError(f"{len(bad)} cells below scaled Jacobian {SJ_MIN} remain")
        _, min_sj = _folded()
        nodes = len(gmsh.model.mesh.getNodes()[0])
        cells = len(gmsh.model.mesh.getElementsByType(11)[0])
        gmsh.write(out)
    _solver_check(out)
    return {"path": out, "scheme": scheme, "alg3d": alg, "scale": scale, "nodes": nodes, "dofs": 3 * nodes,
            "cells": cells, "folded_straightened": folded, "min_sj": min_sj, "seconds": time.perf_counter() - t0}


def _solver_check(path: str) -> None:
    """The fixed solver must be able to use the mesh: its curved mesh must locate every boundary
    quadrature point (FacetBasis inverts the isoparametric map there). From feacore.goldgen.mesh."""
    from skfem import ElementTetP2, FacetBasis
    from feacore.solver._skfem_mesh import read_mesh
    md = read_mesh(path, 3, 2)
    FacetBasis(md.mesh, ElementTetP2(), facets=md.mesh.boundary_facets(), intorder=4)


def _mesh(problem_id: str, sizes_file: Path, out: Path, scale: float) -> dict:
    """mesh_worker in a subprocess, trying ATTEMPTS in order."""
    errors = []
    for scheme, alg in ATTEMPTS:
        cmd = [sys.executable, "-m", "ace.gold_reference", "--mesh-worker", problem_id, str(sizes_file), str(out),
               repr(scale), scheme, str(alg)]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=MESH_TIMEOUT)
        except subprocess.TimeoutExpired:
            errors.append(f"{scheme}/alg{alg}: timed out")
            continue
        if r.returncode == 0:
            return json.loads(r.stdout.strip().splitlines()[-1]) | {"failed_attempts": errors}
        errors.append(f"{scheme}/alg{alg}: exit {r.returncode}: {' | '.join((r.stderr or r.stdout).strip().splitlines()[-2:])}")
        print(f"  mesh attempt failed: {errors[-1]}", flush=True)
    raise RuntimeError("every meshing attempt failed: " + "; ".join(errors))


def remesh(problem_id: str, sizes_file: Path, out: Path, dofs_next: float, max_dofs: int, scale: float):
    """Mesh with the sizes, rescaled until the DOF count is close to dofs_next and within max_dofs.
    gmsh's elements are not exactly the size asked for, so `scale` (gmsh size / asked size) is learned
    here and carried to the next level. Returns (mesh metadata, scale)."""
    for _ in range(4):
        meta = _mesh(problem_id, sizes_file, out, scale)
        print(f"  mesh: {meta['dofs']} DOFs (asked {dofs_next:.0f}), {meta['scheme']}, "
              f"{meta['seconds']:.0f}s", flush=True)
        if meta["dofs"] <= max_dofs and meta["dofs"] >= 0.7 * dofs_next:
            return meta, scale
        scale *= (meta["dofs"] / dofs_next) ** (1 / 3)      # DOFs ~ size^-3
    if meta["dofs"] > max_dofs:
        raise RuntimeError(f"could not mesh within {max_dofs} DOFs (last: {meta['dofs']})")
    return meta, scale


# --------------------------------------------------------------------------- the adaptive loop


def build_adaptive_reference(problem_id: str, out_dir, max_dofs: int = 4_500_000, target: float = 0.03,
                             start_level: int = 2, growth: float = 3.0, max_levels: int = 6,
                             quad_level: int = 1, cache_dir=None) -> list[dict]:
    """Adaptive reference for a gold problem (see the module docstring). growth: DOFs of each level
    over the previous one (capped by max_dofs). Returns the log of the levels."""
    from feacore.diagnostics.view import SolutionView
    from feacore.diagnostics.zz import zz_estimate
    from feacore.problems import registry
    from feacore.solver import solve
    from ace import field_error as fe

    p = registry.get(problem_id)
    U_ref = next(r.value for r in p.references() if r.qoi == "strain_energy")
    out_dir, cache = Path(out_dir), Path(cache_dir or fe.CACHE)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    pts, w = fe.quadrature(p.mesh_file(quad_level), 3, False)
    mesh, scale, log, sig = {"path": str(p.mesh_file(start_level)), "scheme": f"study L{start_level}"}, 1.0, [], []
    log_file = cache / f"{problem_id}_adaptive.json"
    for k in range(max_levels):
        t0 = time.perf_counter()
        res = solve(p.case(mesh["path"], order=2), backend="skfem")
        t_solve = time.perf_counter() - t0
        U = float(res.evaluate(next(q for q in res.case.qois if q.name == "strain_energy")))
        zz = zz_estimate(SolutionView.from_result(res))
        sig.append(fe.stresses(res, pts, 3))
        n_dofs, nodes = int(res.info.n_dofs), int(res.model.sb.N)
        err = float(np.sqrt(abs(U_ref - U) / U_ref))
        log.append({"level": k, "mesh": mesh["path"], "scheme": mesh["scheme"], "dofs": n_dofs, "nodes": nodes,
                    "cells": int(res.model.mesh.t.shape[1]), "strain_energy": U, "U_ref": U_ref,
                    "rel_energy_diff": (U - U_ref) / U_ref, "energy_norm_error": err,
                    "zz_eta_rel": zz["eta_rel"], "zz_top1pct": zz["concentration_top1pct"],
                    "worst_cell_at": zz["worst_cell_at"], "solve_s": round(t_solve, 1),
                    "mesh_s": round(mesh.get("seconds", 0.0), 1), "wall_s": round(time.perf_counter() - t0 + mesh.get("seconds", 0.0), 1)})
        log_file.write_text(json.dumps(log, indent=1))
        print(f"{problem_id} level {k}: {n_dofs} DOFs, U {U:.6e} (ref {U_ref:.6e}, {(U - U_ref) / U_ref:+.3%}), "
              f"energy-norm error {err:.2%}, ZZ {zz['eta_rel']:.2%}, solve {t_solve:.0f}s", flush=True)
        # stop when accurate enough or the budget is spent (at least two levels: the coarse cache needs one)
        if k >= 1 and (err < target or n_dofs >= 0.9 * max_dofs) or k == max_levels - 1:
            break
        X = res.model.mesh.p[:, res.model.mesh.t].transpose(2, 1, 0)       # corner tets (ne, 4, 3)
        h = new_sizes(X, zz["eta2_cell"], n_dofs, min(max_dofs, growth * n_dofs))
        t = res.model.mesh.t.T
        node_h = np.full(res.model.mesh.p.shape[1], np.inf)
        np.minimum.at(node_h, t.ravel(), np.repeat(h, 4))     # a vertex takes the smallest size around it
        sizes_file = out_dir / f"{problem_id}_sizes{k + 1}.npz"
        np.savez(sizes_file, X=res.model.mesh.p.T, t=t, h=node_h)
        del res, zz, X                                         # free the solve before meshing the next level
        mesh, scale = remesh(problem_id, sizes_file, out_dir / f"{problem_id}_A{k + 1}.msh",
                             min(max_dofs, growth * n_dofs), max_dofs, scale)
    fine, coarse = log[-1], log[-2]
    np.savez_compressed(cache / f"{problem_id}.npz", pts=pts, w=w, nodes=fine["nodes"], n_dofs=fine["dofs"],
                        **{f"s_{c}": v for c, v in sig[-1].items()})
    np.savez_compressed(cache / f"{problem_id}_coarse.npz", nodes=coarse["nodes"],
                        **{f"s_{c}": v for c, v in sig[-2].items()})
    for stale in cache.glob(f"{problem_id}_points*.json"):   # picked from the old reference
        stale.unlink()
    (cache / f"{problem_id}_auto.json").unlink(missing_ok=True)
    print(f"{problem_id}: wrote {cache / problem_id}.npz, _coarse.npz, _adaptive.json", flush=True)
    return log


# --------------------------------------------------------------------------- probe-point check


def probe_distances(problem_id: str) -> list[dict]:
    """For each probe point (ace.field_error.probe_points), its distance to the nearest CAD edge and
    to the nearest edge of a supported face (where the singular edges are), relative to the part's
    bounding-box diagonal. The probe filters should keep every point away from the singular edges."""
    import gmsh
    from feacore.geometry.gmsh_tools import gmsh_session
    from feacore.problems import registry
    from ace import field_error as fe
    from ace.checks import reference_setup
    p = registry.get(problem_id)
    pts = np.array(fe.probe_points(problem_id)["points"])
    supported = {bc["group"] for bc in reference_setup(p)["bcs"] if bc["type"] in ("fixed", "displacement", "symmetry")}
    with gmsh_session(p.id):
        p.build_geometry()
        b = np.array(gmsh.model.getBoundingBox(-1, -1))
        diag = float(np.linalg.norm(b[3:] - b[:3]))
        curves = [t for _, t in gmsh.model.getEntities(1)]
        faces = {gmsh.model.getPhysicalName(2, t): gmsh.model.getEntitiesForPhysicalGroup(2, t)
                 for _, t in gmsh.model.getPhysicalGroups(2)}
        clamp_edges = {abs(t) for g in supported for _, t in
                       gmsh.model.getBoundary([(2, int(f)) for f in faces[g]], oriented=False)}

        def dist(x, tags):
            return min(np.linalg.norm(np.array(gmsh.model.getClosestPoint(1, c, list(x))[0]) - x) for c in tags)

        return [{"point": x.tolist(), "edge": dist(x, curves) / diag, "support_edge": dist(x, clamp_edges) / diag}
                for x in pts]


def main() -> None:
    import argparse
    if sys.argv[1:2] == ["--mesh-worker"]:                     # subprocess entry of _mesh
        pid, sizes, out, scale, scheme, alg = sys.argv[2:8]
        try:
            print(json.dumps(mesh_worker(pid, sizes, out, float(scale), scheme, int(alg))))
        except Exception as e:  # noqa: BLE001  reported to the parent through stderr and the exit code
            print(f"{type(e).__name__}: {e}", file=sys.stderr)
            sys.exit(3)
        return
    ap = argparse.ArgumentParser()
    ap.add_argument("problem_ids", help="comma-separated gold problem ids")
    ap.add_argument("--out", default="work/gold_ref", help="where the level meshes go")
    ap.add_argument("--max-dofs", type=int, default=4_500_000)
    ap.add_argument("--target", type=float, default=0.03)
    ap.add_argument("--start-level", type=int, default=2)
    ap.add_argument("--growth", type=float, default=3.0)
    ap.add_argument("--max-levels", type=int, default=6)
    ap.add_argument("--quad-level", type=int, default=1)
    ap.add_argument("--check-probes", action="store_true", help="only print the probe points' distances to edges")
    a = ap.parse_args()
    for pid in a.problem_ids.split(","):
        if a.check_probes:
            for d in probe_distances(pid):
                print(f"{pid}: probe at {np.round(d['point'], 4).tolist()}: nearest edge {d['edge']:.3f} diag, "
                      f"nearest support edge {d['support_edge']:.3f} diag")
            continue
        build_adaptive_reference(pid, a.out, a.max_dofs, a.target, a.start_level, a.growth, a.max_levels,
                                 a.quad_level)


if __name__ == "__main__":
    main()
