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


@pytest.fixture(scope="module")
def thin_tube(tmp_path_factory):
    """A long tube with a 6 % wall: its end faces are 1 mm rings at the generator's scale, like
    abc_00000003's ring under its bore. At the generator's base size the rings get 11 segments, and
    their coarse triangles' straight edges cut across the hole's edge."""
    from feacore.goldgen.parts import screen

    d = tmp_path_factory.mktemp("tube")
    step = d / "tube.step"
    with gmsh_session("t"):
        occ = gmsh.model.occ
        outer = occ.addCylinder(0, 0, 0, 0, 0, 5, 1.0)
        inner = occ.addCylinder(0, 0, -0.1, 0, 0, 5.2, 0.94)
        occ.cut([(3, outer)], [(3, inner)])
        occ.synchronize()
        gmsh.write(str(step))
    return step, screen(step, "thin_tube")


def _ring_vertex_radii(path, part) -> np.ndarray:
    """Radii of the triangle corners (split vertices) of the planar end rings in a mesh file. Midside
    nodes are left out: a straight-sided mesh has them at chord midpoints, inside the hole by the sagitta."""
    import meshio

    m = meshio.read(path)
    rings = [f"f{f.tag}" for f in part.faces if f.kind == "plane"]
    tri = np.concatenate([m.cells_dict["triangle6"][m.cell_sets_dict[g]["triangle6"]][:, :3] for g in rings])
    nodes = np.unique(tri)
    return np.hypot(m.points[nodes, 0], m.points[nodes, 1])


def test_nested_split_stays_on_thin_faces(thin_tube, tmp_path):
    """Splitting snaps new vertices onto a face's surface, not into the trimmed face: from a base whose
    ring triangles cut across the hole, they would land in the hole (the mesh then folds into it). The
    base is refined on the rings until two simulated splits stay on them."""
    from feacore.goldgen.mesh import mesh_level

    step, part = thin_tube
    m = mesh_level(step, part, 1, tmp_path / "L1.msh", threads=1, h0_frac=0.237, scheme="curved-local")
    assert set(m["face_sizes"]) == {f"f{f.tag}" for f in part.faces if f.kind == "plane"}
    r = _ring_vertex_radii(m["path"], part)
    r_in, r_out = 0.94 * part.scale, 1.0 * part.scale
    assert r.min() > r_in * (1 - 1e-9) and r.max() < r_out * (1 + 1e-9)


def test_face_check_finds_split_vertices_in_a_hole(thin_tube):
    """The coarse base is on its faces, one split of it is not (the rings' vertices land in the hole);
    the base _base_mesh makes keeps NESTED_SPLITS splits on the rings."""
    from feacore.goldgen import mesh as gm
    from feacore.goldgen.parts import import_scaled

    step, part = thin_tube
    rings = {f.tag for f in part.faces if f.kind == "plane"}
    tol = gm.LEAK_TOL * part.diag
    with gmsh_session("leak"):
        import_scaled(step, part.scale)
        h0 = 0.237 * part.diag
        for k, v in (("Mesh.MeshSizeMax", h0), ("Mesh.MeshSizeMin", h0 / 10), ("Mesh.MeshSizeFromCurvature", gm.CURV),
                     ("Mesh.Algorithm3D", 10), ("General.NumThreads", 1)):
            gmsh.option.setNumber(k, v)
        gmsh.model.mesh.generate(3)
        assert not gm._face_leaks(tol)
        gmsh.model.mesh.refine()
        leaks = gm._face_leaks(tol)
        assert set(leaks) == rings and all(d > 0.1e-3 for _, d in leaks.values())   # ~0.16 mm into the hole
        gmsh.model.mesh.clear()
        assert set(gm._base_mesh(part.diag, gm.NESTED_SPLITS)) == {f"f{t}" for t in rings}
        for _ in range(gm.NESTED_SPLITS):
            gmsh.model.mesh.refine()
        assert not gm._face_leaks(tol)


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
