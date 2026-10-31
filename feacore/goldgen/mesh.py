"""Nested, systematically refined tet10 meshes of a part, every face a named group ``f<tag>``.

Level k = the base mesh (global size h0_frac x bbox diagonal, linear tets) split k times
(gmsh ``mesh.refine``: ratio 2 per level, ~8x cells; new boundary nodes are snapped to the CAD),
then made quadratic. Nested refinement gives clean observed orders for GCI; independently
remeshed sequences scatter them (see CLAUDE.md, "Mesh sequences for GCI"). The base mesh is
deterministic, so each level can be regenerated on its own. One sequence serves every BC set.

Quadratic geometry, by scheme:
  curved    midside nodes on the CAD (isoparametric). Cells that fold (min scaled Jacobian <= 0)
            are untangled with gmsh's high-order optimizer, only when there are any.
  straight  midside nodes at edge midpoints: cannot fold; geometry error O(h^2) on curved faces.
gmsh's optimizer can abort the whole process (an uncaught C++ exception), so ``mesh_level_safe``
runs each attempt in a subprocess. A part's levels must all use the same scheme (a mixed
sequence is not a systematic refinement): ``mesh_part`` falls back to straight for every level.
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
SCHEMES = ("curved", "straight")


class MeshFailed(RuntimeError):
    pass


def _folded() -> int:
    tags, _ = gmsh.model.mesh.getElementsByType(11)
    if len(tags) == 0:
        return 0
    q = np.asarray(gmsh.model.mesh.getElementQualities(tags, "minSJ"))
    return int((q <= 0).sum())


def mesh_level(step: str | Path, part: Part, k: int, out: str | Path, threads: int = 0,
               h0_frac: float = H0_FRAC, scheme: str = "curved") -> dict:
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
        gmsh.model.mesh.setOrder(2)
        folded = _folded()
        if folded and scheme == "curved":
            opt("Mesh.HighOrderOptimize", 2)
            gmsh.model.mesh.optimize("HighOrder")
        remaining = _folded()
        if remaining:
            raise MeshFailed(f"L{k} {scheme}: {remaining} folded cells remain (of {folded} before optimising)")
        cells = len(gmsh.model.mesh.getElementsByType(11)[0])   # tet10
        gmsh.write(str(out))
    return {"path": str(out), "h": h0 / RATIO**k, "cells": cells, "seconds": time.perf_counter() - t0,
            "scheme": scheme, "folded_before_opt": folded}


def mesh_level_safe(step, part: Part, k: int, out, threads: int = 0, h0_frac: float = H0_FRAC,
                    scheme: str = "curved", timeout: float | None = None) -> dict:
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
              timeout: float | None = None) -> list[dict]:
    """All levels of a part with ONE scheme: curved if every level succeeds, else straight."""
    errors = []
    for scheme in SCHEMES:
        try:
            return [mesh_level_safe(step, part, k, Path(out_dir) / f"L{k}.msh", threads, h0_frac, scheme, timeout)
                    | {"fallback_reason": errors[-1] if errors else None} for k in levels]
        except MeshFailed as e:
            errors.append(str(e))
    raise MeshFailed("; ".join(errors))


if __name__ == "__main__":   # subprocess entry for mesh_level_safe
    step, spec, k, out, threads, h0_frac, scheme = sys.argv[1:8]
    part = Part.from_json(Path(spec).read_text())
    try:
        meta = mesh_level(step, part, int(k), out, int(threads), float(h0_frac), scheme)
    except MeshFailed as e:
        print(str(e), file=sys.stderr)
        sys.exit(3)
    print(json.dumps(meta))
