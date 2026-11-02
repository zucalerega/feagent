"""goldgen meshing: curved tet10 levels the fixed solver can use (fast: coarse levels, small parts)."""

from __future__ import annotations

import json

import gmsh
import numpy as np
import pytest

from feacore.geometry.gmsh_tools import gmsh_session


@pytest.fixture(scope="module")
def filleted_part(tmp_path_factory):
    """Block with a through hole and filleted edges: curved faces of several radii."""
    from feacore.goldgen.parts import screen

    d = tmp_path_factory.mktemp("part")
    step = d / "part.step"
    with gmsh_session("p"):
        occ = gmsh.model.occ
        box = occ.addBox(0, 0, 0, 60, 30, 20)
        occ.synchronize()
        vertical = [t for _, t in gmsh.model.getEntities(1)
                    if gmsh.model.getBoundingBox(1, t)[5] - gmsh.model.getBoundingBox(1, t)[2] > 19]
        body = occ.fillet([box], vertical, [4.0])
        hole = occ.addCylinder(30, 15, -1, 0, 0, 22, 6)
        occ.cut(body, [(3, hole)])
        occ.synchronize()
        gmsh.write(str(step))
    return step, screen(step, "test_part")


def test_mesh_part_is_curved_and_solver_safe(filleted_part, tmp_path):
    from feacore.goldgen.mesh import SJ_MIN, mesh_part, solver_check

    step, part = filleted_part
    metas = mesh_part(step, part, [0, 1], tmp_path, threads=1, h0_frac=0.25, timeout=300)
    assert len(metas) == 2
    for m in metas:
        assert m["scheme"] != "straight" and not m.get("fallback_reason"), m
        assert m["min_sj"] >= SJ_MIN
        solver_check(m["path"])                       # raises if the solver could not use it
    assert metas[1]["cells"] > 4 * metas[0]["cells"]  # a real refinement


def test_robust_inverse_map_on_curved_mesh(filleted_part, tmp_path):
    """Every boundary quadrature point of a curved mesh is located (FacetBasis needs the inverse map)."""
    from skfem import ElementTetP2, FacetBasis

    from feacore.goldgen.mesh import mesh_part
    from feacore.solver._skfem_mesh import CurvedTet, read_mesh

    step, part = filleted_part
    (m,) = mesh_part(step, part, [0], tmp_path, threads=1, h0_frac=0.4, timeout=300)
    md = read_mesh(m["path"], 3, 2)
    assert isinstance(md.mesh, CurvedTet)
    fb = FacetBasis(md.mesh, ElementTetP2(), facets=md.mesh.boundary_facets(), intorder=4)
    # the located points map back onto the facet points they came from
    x = fb.mapping.G(fb.X, find=fb.find)
    back = fb.mapping.F(md.mesh.mapping().invF(x, tind=fb.tind), tind=fb.tind)
    assert np.abs(back - x).max() < 1e-9 * md.bbox_diag


def test_precheck_passes_curved_goldgen_case(filleted_part, tmp_path):
    from feacore.diagnostics import precheck
    from feacore.goldgen.bcgen import generate
    from feacore.goldgen.mesh import mesh_part
    from feacore.goldgen.run import make_case

    step, part = filleted_part
    (m,) = mesh_part(step, part, [0], tmp_path, threads=1, h0_frac=0.25, timeout=300)
    sets = generate(part, 3)
    assert sets
    rep = precheck(make_case(m["path"], sets[0]))
    assert not [f for f in rep.flags if f.category == "mesh" and f.severity == "fail"], rep.ids()
    json.dumps(rep.to_dict())
