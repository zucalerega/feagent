"""Nested, systematically refined tet10 meshes of a part, every face a named group ``f<tag>``.

Level k = the base mesh (global size H0_FRAC x bbox diagonal, linear tets) split k times
(gmsh ``mesh.refine``: ratio 2 per level, ~8x cells; new boundary nodes are snapped to the CAD),
then curved to order 2. Nested refinement gives clean observed orders for GCI; independently
remeshed sequences scatter them (see CLAUDE.md, "Mesh sequences for GCI"). The base mesh is
deterministic, so each level can be regenerated on its own. One sequence serves every BC set.
"""

from __future__ import annotations

import time
from pathlib import Path

import gmsh

from feacore.geometry.gmsh_tools import gmsh_session
from feacore.goldgen.parts import Part, import_scaled

H0_FRAC = 1 / 8        # base global size, fraction of the bbox diagonal
CURV = 8               # base elements per 2 pi of curvature (small holes get resolved)
RATIO = 2.0            # refinement ratio per level (nested splitting)


def mesh_level(step: str | Path, part: Part, k: int, out: str | Path, threads: int = 0,
               h0_frac: float = H0_FRAC) -> dict:
    """Mesh level k of the part; returns {"path", "h", "cells", "seconds"}."""
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
        gmsh.model.mesh.generate(3)
        for _ in range(k):
            gmsh.model.mesh.refine()
        gmsh.model.mesh.setOrder(2)
        opt("Mesh.HighOrderOptimize", 2)     # untangle curved tet10 cells (they fold on coarse levels)
        gmsh.model.mesh.optimize("HighOrder")
        cells = len(gmsh.model.mesh.getElementsByType(11)[0])   # tet10
        gmsh.write(str(out))
    return {"path": str(out), "h": h0 / RATIO**k, "cells": cells, "seconds": time.perf_counter() - t0}
