"""Nested, systematically refined tet10 meshes of a part, every face a named group ``f<tag>``.

Level k = the base mesh (global size h0_frac x bbox diagonal, linear tets) split k times
(gmsh ``mesh.refine``: ratio 2 per level, ~8x cells; new boundary nodes are snapped to the CAD),
then made quadratic. Nested refinement gives clean observed orders for GCI; independently
remeshed sequences scatter them (see CLAUDE.md, "Mesh sequences for GCI"). The base mesh is
deterministic, so each level can be regenerated on its own. One sequence serves every BC set.

Pipeline per level (``mesh_level``):
  1. after EVERY split, repair the linear tets with gmsh's tet optimizer + Relocate3D: snapping new
     boundary vertices onto curved CAD flattens a few tets (mean-ratio quality ~0.003 on the pilot
     part), and their children inherit the defect at every later level;
  2. curve to order 2 (isoparametric: boundary midside nodes on the CAD);
  3. if any cell is folded or nearly so (min scaled Jacobian < SJ_MIN), run gmsh's high-order
     optimizer (``curved-opt``, elastic + optimisation);
  4. validate what the solver needs: min scaled Jacobian >= SJ_MIN everywhere, and the solver's own
     curved mesh (``feacore.solver._skfem_mesh``) can locate every boundary quadrature point.
For a curved cell the scaled Jacobian is the validity measure; the mean-ratio quality of its corner
tet is not (the high-order optimizer may move a vertex so the chord tet is flat while the curved
cell is fine), so corner quality is only gated on the linear mesh, before curving.

gmsh's high-order optimizer can abort the whole process (an uncaught C++ exception), so each
attempt runs in a subprocess (``mesh_level_safe``). Fallbacks, in order (``STRATEGIES``, then straight):
  curved-local  straighten only the nearly folded cells' midside nodes (those cells lose curvature)
  straight      every midside node at its edge midpoint: cannot fold; O(h^2) geometry error
Levels may use different curved strategies (the boundary is the CAD either way); if any level ends
up straight, every level of the part is remeshed straight so the sequence stays a systematic
refinement (``mesh_part``). Pilot part abc_00000003, L2 (223k cells): 30 s, 0.4 GB, curved-opt.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import gmsh
import numpy as np

from feacore.geometry.gmsh_tools import gmsh_session
from feacore.goldgen.parts import Part, import_scaled

H0_FRAC = 1 / 8        # base global size, fraction of the bbox diagonal
CURV = 8               # base elements per 2 pi of curvature (small holes get resolved)
RATIO = 2.0            # refinement ratio per level (nested splitting)
STRATEGIES = ("curved-opt", "curved-local")
ATTEMPT_TIMEOUT = 300.0   # seconds per attempt (L2 ~500k cells meshes in ~90 s; longer means a grinding optimizer)
SJ_MIN = 0.05             # min scaled Jacobian every curved cell must reach
QUALITY_MIN = 0.002       # linear (pre-curving) corner-tet quality: only true degenerates (curved
                          # validity is gated by SJ_MIN and solver_check, which bad slivers also fail)
ALGORITHMS = (10, 1)      # gmsh 3D meshers: HXT (fast, parallel), then Delaunay (HXT segfaults on some CAD)
REPAIR_TARGET = 0.05      # repair passes stop once the linear mesh reaches this quality


class MeshFailed(RuntimeError):
    pass


def _folded_tags(threshold: float = SJ_MIN) -> np.ndarray:
    """Tet10 cells that are folded or nearly so (min scaled Jacobian below ``threshold``)."""
    tags, _ = gmsh.model.mesh.getElementsByType(11)
    if len(tags) == 0:
        return np.zeros(0, dtype=np.int64)
    q = np.asarray(gmsh.model.mesh.getElementQualities(tags, "minSJ"))
    return np.asarray(tags)[q < threshold]


def _min_sj() -> float:
    tags, _ = gmsh.model.mesh.getElementsByType(11)
    return float(np.min(gmsh.model.mesh.getElementQualities(tags, "minSJ"))) if len(tags) else 1.0


def _min_quality() -> float:
    """Worst mean-ratio quality of the corner tets (1 = regular), the measure the diagnostics use
    (feacore.diagnostics.checks.mesh_quality); gmsh's "gamma" is on a much harsher scale."""
    tags, nodes = gmsh.model.mesh.getElementsByType(4)
    if len(tags) == 0:
        tags, nodes = gmsh.model.mesh.getElementsByType(11)
        nodes = np.asarray(nodes).reshape(len(tags), 10)[:, :4]
    else:
        nodes = np.asarray(nodes).reshape(len(tags), 4)
    if len(tags) == 0:
        return 1.0
    ntags, X, _ = gmsh.model.mesh.getNodes()
    pos = np.empty(int(np.max(ntags)) + 1, dtype=np.int64)
    pos[np.asarray(ntags, dtype=np.int64)] = np.arange(len(ntags))
    P = np.asarray(X).reshape(-1, 3)[pos[nodes.astype(np.int64)]]          # (n, 4, 3)
    vol = np.abs(np.linalg.det(P[:, 1:] - P[:, :1])) / 6
    l2 = sum(((P[:, i] - P[:, j]) ** 2).sum(1) for i in range(4) for j in range(i + 1, 4))
    return float(np.min(12 * (3 * vol) ** (2 / 3) / l2))


def _repair_linear() -> float:
    """gmsh's tet optimizer + vertex relocation until REPAIR_TARGET (at most 3 rounds)."""
    for _ in range(3):
        gmsh.model.mesh.optimize("", niter=3)
        gmsh.model.mesh.optimize("Relocate3D", niter=3)
        q = _min_quality()
        if q >= REPAIR_TARGET:
            break
    return q


def _straighten(cells: np.ndarray) -> None:
    """Move the midside nodes of the given tet10 cells to their edge midpoints."""
    edges = [(0, 1, 4), (1, 2, 5), (0, 2, 6), (0, 3, 7), (2, 3, 8), (1, 3, 9)]   # gmsh tet10 ordering
    for c in cells:
        _, nodes, _, _ = gmsh.model.mesh.getElement(int(c))
        X = {int(n): np.asarray(gmsh.model.mesh.getNode(int(n))[0]) for n in nodes[:4]}
        for a, b, m in edges:
            mid = 0.5 * (X[int(nodes[a])] + X[int(nodes[b])])
            gmsh.model.mesh.setNode(int(nodes[m]), mid.tolist(), [])


def solver_check(path: str | Path) -> None:
    """Raise MeshFailed unless the fixed solver can use the mesh: its curved mesh must locate every
    boundary quadrature point (FacetBasis inverts the isoparametric map there)."""
    from skfem import ElementTetP2, FacetBasis

    from feacore.solver._skfem_mesh import read_mesh

    try:
        md = read_mesh(path, 3, 2)
        FacetBasis(md.mesh, ElementTetP2(), facets=md.mesh.boundary_facets(), intorder=4)
    except Exception as e:  # noqa: BLE001 - any failure here means the solver would fail
        raise MeshFailed(f"solver check: {type(e).__name__}: {str(e)[:200]}") from e


def mesh_level(step: str | Path, part: Part, k: int, out: str | Path, threads: int = 0,
               h0_frac: float = H0_FRAC, scheme: str = "curved-opt", nested: bool = True, alg3d: int = 10) -> dict:
    """Mesh level k of the part in THIS process (gmsh may abort it: see mesh_level_safe).
    ``nested``: split the base mesh k times; else generate level k directly from the CAD with every
    size control scaled by RATIO**-k through Mesh.MeshSizeFactor (no snapping, but a non-nested
    sequence). ``alg3d``: gmsh Mesh.Algorithm3D."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    h0 = part.diag * h0_frac
    t0 = time.perf_counter()
    with gmsh_session(part.id):
        ents = import_scaled(step, part.scale)
        (_, vol), = [e for e in ents if e[0] == 3]
        for f in part.faces:
            gmsh.model.addPhysicalGroup(2, [f.tag], name=f"f{f.tag}")
        gmsh.model.addPhysicalGroup(3, [vol], name="solid")
        opt = gmsh.option.setNumber
        opt("Mesh.MeshSizeMax", h0)
        opt("Mesh.MeshSizeMin", h0 / 10)
        opt("Mesh.MeshSizeFromCurvature", CURV)
        # remeshed levels: one global factor scales every size gmsh computes (max, min, curvature)
        opt("Mesh.MeshSizeFactor", 1.0 if nested else RATIO**-k)
        opt("Mesh.Algorithm3D", alg3d)
        opt("General.NumThreads", threads)
        opt("Mesh.SecondOrderLinear", 1 if scheme == "straight" else 0)
        opt("Mesh.HighOrderOptimize", 0)     # optimise explicitly below, only if needed
        gmsh.model.mesh.generate(3)
        quality = _min_quality()
        if quality < REPAIR_TARGET:
            quality = _repair_linear()
        for _ in range(k if nested else 0):
            gmsh.model.mesh.refine()
            quality = _repair_linear()
        if quality < QUALITY_MIN:
            raise MeshFailed(f"L{k} {scheme}: linear corner-tet quality {quality:.2e} < {QUALITY_MIN} after repair")
        gmsh.model.mesh.setOrder(2)
        folded = len(_folded_tags())
        used = "straight" if scheme == "straight" else "curved"
        t_fix = time.perf_counter()
        if folded and scheme != "straight":
            used = scheme
            if scheme == "curved-opt":
                opt("Mesh.HighOrderOptimize", 2)
                gmsh.model.mesh.optimize("HighOrder")
            elif scheme == "curved-local":
                for _ in range(5):
                    bad = _folded_tags()
                    if not len(bad):
                        break
                    _straighten(bad)
        t_fix = time.perf_counter() - t_fix
        remaining = len(_folded_tags())
        if remaining:
            raise MeshFailed(f"L{k} {scheme}: {remaining} cells below scaled Jacobian {SJ_MIN} remain "
                             f"(of {folded} before fixing)")
        sj = _min_sj()
        cells = len(gmsh.model.mesh.getElementsByType(11)[0])   # tet10
        gmsh.write(str(out))
    solver_check(out)
    return {"path": str(out), "h": h0 / RATIO**k, "cells": cells, "seconds": time.perf_counter() - t0,
            "fix_seconds": round(t_fix, 1), "scheme": used, "nested": nested, "alg3d": alg3d, "h0_frac": h0_frac,
            "folded_before_fix": folded,
            "min_sj": sj, "min_linear_quality": quality}


def mesh_level_safe(step, part: Part, k: int, out, threads: int = 0, h0_frac: float = H0_FRAC,
                    scheme: str = "curved-opt", timeout: float | None = ATTEMPT_TIMEOUT,
                    nested: bool = True, alg3d: int = 10) -> dict:
    """mesh_level in a subprocess: a gmsh abort or a failed check raises MeshFailed here."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    spec = out.with_suffix(".part.json")
    spec.write_text(part.to_json())
    cmd = [sys.executable, "-m", "feacore.goldgen.mesh", str(step), str(spec), str(k), str(out),
           str(threads), repr(h0_frac), scheme, "nested" if nested else "remesh", str(alg3d)]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as e:
        raise MeshFailed(f"L{k} {scheme}{'' if nested else ' remesh'}: timed out after {timeout}s") from e
    finally:
        spec.unlink(missing_ok=True)
    if p.returncode != 0:
        tail = (p.stderr or p.stdout).strip().splitlines()[-3:]
        raise MeshFailed(f"L{k} {scheme}{'' if nested else ' remesh'}: exit {p.returncode}: {' | '.join(tail)}")
    return json.loads(p.stdout.strip().splitlines()[-1])


def fit_h0(step, part: Part, h0_frac: float, finest: int, max_cells: int, out_dir, threads: int = 0,
           timeout: float | None = ATTEMPT_TIMEOUT) -> float:
    """Base size fraction for which the finest nested level stays under ``max_cells`` (L_k ~ 8^k L0):
    a straight L0 (cheap, cannot fold) is counted and h0 enlarged by the cube root of the excess.
    Curvature sizing does not scale with h0, so this iterates (at most 4 times)."""
    for _ in range(4):
        m = mesh_level_safe(step, part, 0, Path(out_dir) / "L0_sizing.msh", threads, h0_frac, "straight", timeout)
        Path(m["path"]).unlink(missing_ok=True)
        predicted = m["cells"] * RATIO ** (3 * finest)
        if predicted <= max_cells:
            break
        h0_frac *= 1.05 * (predicted / max_cells) ** (1 / 3)
    return h0_frac


def mesh_part(step, part: Part, levels: list[int], out_dir, threads: int = 0, h0_frac: float = H0_FRAC,
              timeout: float | None = ATTEMPT_TIMEOUT) -> list[dict]:
    """All levels of a part under ONE plan, the first that works for every level. For each 3D mesher
    (ALGORITHMS): nested + curved, remeshed + curved; then nested + straight, remeshed + straight.
    Within a curved plan each level tries STRATEGIES in order. The metadata records the plan used
    and every failed attempt; MeshFailed if nothing works (the part is unusable: skip it)."""
    tried = []

    def attempt(k, scheme, nested, alg):
        return mesh_level_safe(step, part, k, Path(out_dir) / f"L{k}.msh", threads, h0_frac, scheme, timeout,
                               nested, alg)

    for alg in ALGORITHMS:
        for nested in (True, False):
            metas = []
            for k in levels:
                for scheme in STRATEGIES:
                    try:
                        metas.append(attempt(k, scheme, nested, alg))
                        break
                    except MeshFailed as e:
                        tried.append(f"alg{alg} " + str(e)[:300])
                else:
                    break
            else:
                return [m | {"failed_attempts": tried} for m in metas]
    for alg in ALGORITHMS:
        for nested in (True, False):
            try:
                straight = [attempt(k, "straight", nested, alg) for k in levels]
                return [m | {"failed_attempts": tried, "fallback_reason": "no curved plan succeeded"} for m in straight]
            except MeshFailed as e:
                tried.append(f"alg{alg} " + str(e)[:300])
    raise MeshFailed("; ".join(tried[-4:]))


if __name__ == "__main__":   # subprocess entry for mesh_level_safe
    step, spec, k, out, threads, h0_frac, scheme, mode, alg3d = sys.argv[1:10]
    part = Part.from_json(Path(spec).read_text())
    try:
        meta = mesh_level(step, part, int(k), out, int(threads), float(h0_frac), scheme, mode == "nested", int(alg3d))
    except MeshFailed as e:
        print(str(e), file=sys.stderr)
        sys.exit(3)
    print(json.dumps(meta))
