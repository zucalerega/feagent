"""Nested, systematically refined tet10 meshes of a part, every face a named group ``f<tag>``.

Level k = the base mesh (global size h0_frac x bbox diagonal, linear tets) split k times
(gmsh ``mesh.refine``: ratio 2 per level, ~8x cells; new boundary nodes are snapped to the CAD),
then made quadratic. Nested refinement gives clean observed orders for GCI; independently
remeshed sequences scatter them (see CLAUDE.md, "Mesh sequences for GCI"). The base mesh is
deterministic, so each level can be regenerated on its own. One sequence serves every BC set.

Quadratic geometry. Curved (isoparametric) cells keep their boundary midside nodes on the CAD;
some cells fold or nearly fold (min scaled Jacobian < SJ_MIN). Splitting a level also snaps new
boundary vertices onto the CAD, which can flatten a few tets; those are repaired with gmsh's linear
tet optimizer before curving, and every level must reach QUALITY_MIN corner quality. Strategies (``STRATEGIES`` lists the ones tried, in order):
  curved           no folded cells after curving: nothing to fix
  curved-local     straighten only the folded cells' midside nodes (a few cells lose curvature);
                   the default and only strategy tried (STRATEGIES)
  curved-opt       gmsh high-order optimizer (elastic + optimisation), default thresholds
  curved-relaxed   the same with a lower quality target (only untangle, don't polish)
  curved-fast      gmsh fast curving
  straight         every midside node at its edge midpoint: cannot fold; O(h^2) geometry error
gmsh's optimizer can abort the whole process (an uncaught C++ exception), so each attempt runs in a
subprocess (``mesh_level_safe``). Levels may use different curved strategies (the boundary is the
CAD either way); if any level ends up straight, every level of the part is remeshed straight so the
sequence stays a systematic refinement (``mesh_part``).
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
STRATEGIES = ("curved-local",)   # gmsh's optimizers stay available but unused: they abort / grind
ATTEMPT_TIMEOUT = 600.0   # seconds per curving attempt (the optimizer can grind)
SJ_MIN = 0.05             # curved cells below this scaled Jacobian are straightened (the solver must invert them)
QUALITY_MIN = 0.03        # corner-tet mean-ratio quality every level must reach (diagnostics fail < 0.01)


class MeshFailed(RuntimeError):
    pass


def _folded_tags(threshold: float = SJ_MIN) -> np.ndarray:
    """Tet10 cells that are folded or nearly so (min scaled Jacobian below ``threshold``)."""
    tags, _ = gmsh.model.mesh.getElementsByType(11)
    if len(tags) == 0:
        return np.zeros(0, dtype=np.int64)
    q = np.asarray(gmsh.model.mesh.getElementQualities(tags, "minSJ"))
    return np.asarray(tags)[q < threshold]


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


def _straighten(cells: np.ndarray) -> None:
    """Move the midside nodes of the given tet10 cells to their edge midpoints."""
    edges = [(0, 1, 4), (1, 2, 5), (0, 2, 6), (0, 3, 7), (2, 3, 8), (1, 3, 9)]   # gmsh tet10 ordering
    for c in cells:
        _, nodes, _, _ = gmsh.model.mesh.getElement(int(c))
        X = {int(n): np.asarray(gmsh.model.mesh.getNode(int(n))[0]) for n in nodes[:4]}
        for a, b, m in edges:
            mid = 0.5 * (X[int(nodes[a])] + X[int(nodes[b])])
            gmsh.model.mesh.setNode(int(nodes[m]), mid.tolist(), [])


def mesh_level(step: str | Path, part: Part, k: int, out: str | Path, threads: int = 0,
               h0_frac: float = H0_FRAC, scheme: str = "curved-local") -> dict:
    """Mesh level k of the part in THIS process (gmsh may abort it: see mesh_level_safe)."""
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
        opt("Mesh.Algorithm3D", 10)          # HXT: parallel
        opt("General.NumThreads", threads)
        opt("Mesh.SecondOrderLinear", 1 if scheme == "straight" else 0)
        opt("Mesh.HighOrderOptimize", 0)     # optimise explicitly below, only if needed
        gmsh.model.mesh.generate(3)
        for _ in range(k):
            gmsh.model.mesh.refine()
        # splitting snaps new boundary vertices onto the CAD, which can flatten a few tets: repair
        # them with gmsh's linear tet optimizer (robust, unlike the high-order one) before curving
        repaired = False
        if k and _min_quality() < 0.1:
            gmsh.model.mesh.optimize("")
            repaired = True
        gmsh.model.mesh.setOrder(2)
        folded = len(_folded_tags())
        used = "straight" if scheme == "straight" else "curved"
        if folded and scheme != "straight":
            used = scheme
            if scheme == "curved-opt":
                opt("Mesh.HighOrderOptimize", 2)
                gmsh.model.mesh.optimize("HighOrder")
            elif scheme == "curved-relaxed":
                opt("Mesh.HighOrderOptimize", 2)
                opt("Mesh.HighOrderThresholdMin", 0.01)
                opt("Mesh.HighOrderPassMax", 50)
                gmsh.model.mesh.optimize("HighOrder")
            elif scheme == "curved-fast":
                opt("Mesh.HighOrderOptimize", 4)
                gmsh.model.mesh.optimize("HighOrderFastCurving")
            elif scheme == "curved-local":
                for _ in range(5):
                    bad = _folded_tags()
                    if not len(bad):
                        break
                    _straighten(bad)
        remaining = len(_folded_tags())
        if remaining:
            raise MeshFailed(f"L{k} {scheme}: {remaining} cells below scaled Jacobian {SJ_MIN} remain "
                             f"(of {folded} before fixing)")
        quality = _min_quality()
        if quality < QUALITY_MIN:
            raise MeshFailed(f"L{k} {scheme}: worst corner-tet quality {quality:.2e} < {QUALITY_MIN}")
        cells = len(gmsh.model.mesh.getElementsByType(11)[0])   # tet10
        gmsh.write(str(out))
    return {"path": str(out), "h": h0 / RATIO**k, "cells": cells, "seconds": time.perf_counter() - t0,
            "scheme": used, "folded_before_fix": folded, "linear_repair": repaired, "min_quality": quality}


def mesh_level_safe(step, part: Part, k: int, out, threads: int = 0, h0_frac: float = H0_FRAC,
                    scheme: str = "curved-local", timeout: float | None = ATTEMPT_TIMEOUT) -> dict:
    """mesh_level in a subprocess: a gmsh abort or a remaining fold raises MeshFailed here."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    spec = out.with_suffix(".part.json")
    spec.write_text(part.to_json())
    cmd = [sys.executable, "-m", "feacore.goldgen.mesh", str(step), str(spec), str(k), str(out),
           str(threads), repr(h0_frac), scheme]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as e:
        raise MeshFailed(f"L{k} {scheme}: timed out after {timeout}s") from e
    finally:
        spec.unlink(missing_ok=True)
    if p.returncode != 0:
        tail = (p.stderr or p.stdout).strip().splitlines()[-3:]
        raise MeshFailed(f"L{k} {scheme}: exit {p.returncode}: {' | '.join(tail)}")
    return json.loads(p.stdout.strip().splitlines()[-1])


def mesh_part(step, part: Part, levels: list[int], out_dir, threads: int = 0, h0_frac: float = H0_FRAC,
              timeout: float | None = ATTEMPT_TIMEOUT) -> list[dict]:
    """All levels of a part, curved if every level can be curved by some strategy, else every
    level straight. Each level's metadata records the strategy and the failed attempts."""
    metas = []
    for k in levels:
        tried = []
        for scheme in STRATEGIES:
            try:
                m = mesh_level_safe(step, part, k, Path(out_dir) / f"L{k}.msh", threads, h0_frac, scheme, timeout)
                metas.append(m | {"failed_attempts": tried})
                break
            except MeshFailed as e:
                tried.append(str(e)[:300])
        else:
            straight = [mesh_level_safe(step, part, kk, Path(out_dir) / f"L{kk}.msh", threads, h0_frac, "straight",
                                        timeout) for kk in levels]
            return [m | {"failed_attempts": tried, "fallback_reason": f"L{k}: no curved strategy succeeded"}
                    for m in straight]
    return metas


if __name__ == "__main__":   # subprocess entry for mesh_level_safe
    step, spec, k, out, threads, h0_frac, scheme = sys.argv[1:8]
    part = Part.from_json(Path(spec).read_text())
    try:
        meta = mesh_level(step, part, int(k), out, int(threads), float(h0_frac), scheme)
    except MeshFailed as e:
        print(str(e), file=sys.stderr)
        sys.exit(3)
    print(json.dumps(meta))
