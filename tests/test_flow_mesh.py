"""The agents' flow mesher (ace.flow_mesh through ace.fea.make_mesh): the CAD with the reference
geometry's group names, wall boundary layers, swept 3D meshes, finite-volume quality and its gate."""

import math
import tempfile
from pathlib import Path

import gmsh
import pytest

from ace import checks, config, fea, flow_mesh
from ace.agent import TOOLS_FLOW, Tools
from feacore.geometry.gmsh_tools import gmsh_session

CFD = [pid for pid in config.TASKS if fea.is_flow(fea.get_problem(pid))]


def _measures(builder) -> dict:
    """{group: total length / area} of the dim-1 / dim-2 groups the builder makes."""
    out = {}
    for d, t in gmsh.model.getPhysicalGroups():
        if d in (1, 2):
            out[gmsh.model.getPhysicalName(d, t)] = sum(gmsh.model.occ.getMass(d, int(e))
                                                        for e in gmsh.model.getEntitiesForPhysicalGroup(d, t))
    return out


@pytest.mark.parametrize("pid", CFD)
def test_cad_gets_every_reference_group(pid):
    """The agent meshes the plain CAD; every group of the reference geometry lands on it with the
    same total length / area, and no boundary entity is left without a group."""
    p = fea.get_problem(pid)
    with gmsh_session("ref"):
        p.build_geometry()
        gmsh.model.occ.synchronize()
        ref = _measures(None)
        n_ref = len(gmsh.model.getEntities(p.dim))
    with gmsh_session("cad"):
        info = fea.build_agent_geometry(p)
        got = _measures(None)
        assert len(gmsh.model.getEntities(p.dim)) == 1 <= n_ref           # one face / solid, no blocks
        named = {t for name, (d, ts) in info["groups"].items() if d == p.dim - 1 for t in ts}
        boundary = [t for _, t in gmsh.model.getEntities(p.dim - 1) if len(gmsh.model.getAdjacencies(p.dim - 1, t)[0]) == 1]
        assert set(boundary) <= named
    assert set(got) == set(ref)
    for name, v in ref.items():
        assert got[name] == pytest.approx(v, rel=1e-6), name


def test_geometry_summary_and_mesh_use_the_same_groups(tmp_path):
    p = fea.get_problem("cylinder_channel_dfg_2d1")
    groups = {g["name"]: g for g in fea.geometry_summary(p)["groups"]}
    assert groups["cylinder"]["measure"] == pytest.approx(math.pi * 0.1, rel=1e-6)
    m = fea.make_mesh(p, tmp_path / "m.msh", h=0.03, quads=True)
    assert m["ok"] and set(m["size_by_group"]) == set(groups)


def test_boundary_layer_2d_first_cell_and_ends_on_the_inlet(tmp_path):
    p = fea.get_problem("cylinder_channel_dfg_2d1")
    bl = {"groups": ["cylinder", "walls"], "first": 0.001, "ratio": 1.3, "layers": 4}
    m = fea.make_mesh(p, tmp_path / "m.msh", h=0.025, quads=True, boundary_layer=bl)
    assert m["ok"] and m["fv_ok"], m.get("fv_gate")
    near = m["fv_quality"]["near_wall"]
    for g in ("cylinder", "walls"):                      # the first cell centre sits at half its height
        assert near[g]["wall_distance"] == pytest.approx(0.0005, rel=0.02)
    assert near["inlet"]["wall_distance"] > 5 * near["walls"]["wall_distance"]
    assert m["boundary_layer"]["layers"] == 4
    assert m["boundary_layer"]["thickness"] == pytest.approx(0.001 * (1.3 ** 4 - 1) / 0.3, rel=1e-6)
    plain = fea.make_mesh(p, tmp_path / "p.msh", h=0.025, quads=True)
    assert plain["fv_quality"]["near_wall"]["walls"]["wall_distance"] > 5 * near["walls"]["wall_distance"]


def test_boundary_layers_in_a_narrow_gap_do_not_collide(tmp_path):
    """gmsh's default boundary-layer thickness (0.01 m) is the Taylor-Couette gap: without the
    explicit thickness the two layers grow into each other and leave slivers."""
    p = fea.get_problem("taylor_couette_2d")
    bl = {"groups": ["inner", "outer"], "first": 0.0001, "ratio": 1.3, "layers": 4}
    m = fea.make_mesh(p, tmp_path / "m.msh", h=0.0008, quads=True, boundary_layer=bl)
    assert m["ok"] and m["fv_ok"]
    assert m["fv_quality"]["volume_ratio"]["max"] < 10
    assert m["fv_quality"]["non_orthogonality"]["max"] < 45


def test_sweep_makes_hexahedra_with_graded_layers(tmp_path):
    p = fea.get_problem("pipe_poiseuille_3d")
    m = fea.make_mesh(p, tmp_path / "m.msh", h=0.001, quads=True, sweep={"source": "inlet", "layers": 12, "ratio": 3})
    assert m["ok"] and m["fv_ok"]
    assert set(m["cells"]) == {"hexahedron"} and m["cells"]["hexahedron"] % 12 == 0
    sw = m["sweep"]
    assert sw["layers"] == 12 and sw["length"] == pytest.approx(0.05)
    assert sw["last_layer"] / sw["first_layer"] == pytest.approx(3.0, rel=1e-3)
    assert set(m["size_by_group"]) >= {"inlet", "outlet", "wall"}
    tri = fea.make_mesh(p, tmp_path / "t.msh", h=0.001, sweep={"source": "inlet", "layers": 12})
    assert tri["ok"] and "wedge" in tri["cells"]               # triangles swept into prisms


def test_sweep_face_sets_the_size_floor(tmp_path):
    """A swept mesh is floored by its face, not by the 3D floor of the whole (long) domain: on the
    square duct (diagonal 0.1 m) a 3D floor would clamp h to 1 mm, 10 cells across."""
    p = fea.get_problem("square_duct_3d")
    m = fea.make_mesh(p, tmp_path / "m.msh", h=0.0007, quads=True, sweep={"source": "inlet", "layers": 4})
    assert m["ok"] and m["h_used"] == pytest.approx(0.0007)
    big = fea.make_mesh(p, tmp_path / "b.msh", h=0.0001, sweep={"source": "inlet", "layers": 400})
    assert not big["ok"] and "nodes" in big["error"]


def test_sweep_with_boundary_layer_on_the_face(tmp_path):
    p = fea.get_problem("square_duct_3d")
    bl = {"groups": ["walls"], "first": 0.0002, "ratio": 1.3, "layers": 3}
    m = fea.make_mesh(p, tmp_path / "m.msh", h=0.001, quads=True, sweep={"source": "inlet", "layers": 8},
                      boundary_layer=bl)
    assert m["ok"] and m["fv_ok"], m.get("fv_gate")
    assert m["fv_quality"]["near_wall"]["walls"]["wall_distance"] == pytest.approx(0.0001, rel=0.05)


@pytest.mark.parametrize("pid,kw,msg", [
    ("square_duct_3d", {"h": 0.002, "sweep": {"source": "walls", "layers": 4}}, "not planar"),
    ("square_duct_3d", {"h": 0.002, "sweep": {"source": "nope", "layers": 4}}, "not a surface group"),
    ("square_duct_3d", {"h": 0.002, "sweep": {"source": "inlet"}}, "sweep needs"),
    ("square_duct_3d", {"h": 0.002, "boundary_layer": {"groups": ["walls"], "first": 1e-4, "layers": 3}}, "needs a sweep"),
    ("square_duct_3d", {"h": 0.002, "quads": True}, "swept 3D"),
    ("poiseuille_channel", {"h": 0.002, "sweep": {"source": "inlet", "layers": 4}}, "3D domains"),
    ("poiseuille_channel", {"h": 0.002, "boundary_layer": {"groups": ["inlet_x"], "first": 1e-4, "layers": 3}},
     "not boundary groups"),
    ("poiseuille_channel", {"h": 0.002, "boundary_layer": {"groups": ["wall_top"], "first": 1e-4, "layers": 99}},
     "layers must be"),
    ("cooks_membrane", {"h": 4.0, "boundary_layer": {"groups": ["clamped"], "first": 0.1, "layers": 3}}, "flow meshes"),
])
def test_bad_requests_are_refused_with_a_reason(pid, kw, msg, tmp_path):
    m = fea.make_mesh(fea.get_problem(pid), tmp_path / "m.msh", **kw)
    assert not m["ok"] and msg in m["error"], m


def test_scale_request_keeps_the_boundary_layer_thickness():
    req = {"h": 0.01, "refinements": [{"group": "a", "size": 0.002, "radius": 0.1}],
           "boundary_layer": {"groups": ["w"], "first": 0.001, "ratio": 1.2, "layers": 5},
           "sweep": {"source": "inlet", "layers": 10, "ratio": 2}}
    fine = fea.scale_request(req, 0.5)
    assert fine["h"] == 0.005 and fine["refinements"][0] == {"group": "a", "size": 0.001, "radius": 0.1}
    bl = fine["boundary_layer"]
    assert (bl["first"], bl["layers"]) == (0.0005, 10) and bl["ratio"] == pytest.approx(1.1)

    def thickness(b):
        return b["first"] * (b["ratio"] ** b["layers"] - 1) / (b["ratio"] - 1)
    # about the same thickness (exact only in the limit of many thin layers): 7.97 mm vs 7.44 mm here
    assert thickness(bl) == pytest.approx(thickness(req["boundary_layer"]), rel=0.1)
    assert fine["sweep"] == {"source": "inlet", "layers": 20, "ratio": 2} and req["sweep"]["layers"] == 10


def test_flow_quality_gate_in_scoring():
    assert checks.quality_ok({"fv_ok": True, "min_quality": 0.01})          # thin boundary-layer cells are fine
    assert not checks.quality_ok({"fv_ok": False, "fv_gate": "max skewness 5 > 4", "min_quality": 0.9})
    assert "skewness" in checks.quality_gate_message({"fv_ok": False, "fv_gate": "max skewness 5 > 4"})
    assert checks.quality_ok({"min_quality": 0.5}) and not checks.quality_ok({"min_quality": 0.05})


def test_agent_mesh_tool_passes_the_flow_options(tmp_path):
    props = TOOLS_FLOW[1]["input_schema"]["properties"]
    assert {"boundary_layer", "sweep", "quads", "refinements"} <= set(props)
    t = Tools("pipe_poiseuille_3d", tmp_path)
    out, err = t.call("mesh", {"h": 0.0012, "quads": True, "sweep": {"source": "inlet", "layers": 5},
                               "boundary_layer": {"groups": ["wall"], "first": 0.0002, "layers": 2}, "using": []})
    assert not err and out["fv_ok"] and out["sweep"]["layers"] == 5 and out["boundary_layer"]["layers"] == 2
    assert "min_quality" not in out and "quality" not in out              # element-shape stats would mislead
    assert t.meshes[0]["request"]["sweep"] == {"source": "inlet", "layers": 5}
    with tempfile.TemporaryDirectory() as d:                               # the request rebuilds the same mesh
        again = fea.mesh_from_request(t.problem, Path(d) / "m.msh", t.meshes[0]["request"])
        # (with a boundary layer only up to the node numbering: gmsh's layer code is not order-deterministic)
        assert _sorted_points(again["path"]) == pytest.approx(_sorted_points(t.meshes[0]["path"]), abs=1e-12)
    assert again["fv_quality"] == t.meshes[0]["fv_quality"]
    bad, err = t.call("mesh", {"h": 0.0012, "boundary_layer": {"groups": ["wall"], "first": 0.0002, "layers": 2},
                               "using": []})
    assert err and "sweep" in bad["error"]


def _sorted_points(path) -> list:
    import meshio
    import numpy as np
    P = np.round(meshio.read(str(path)).points, 12)
    return P[np.lexsort(P.T[::-1])].ravel().tolist()


def test_meshes_without_boundary_layers_rebuild_bit_for_bit(tmp_path):
    p = fea.get_problem("pipe_poiseuille_3d")
    req = {"h": 0.0012, "quads": True, "sweep": {"source": "inlet", "layers": 5}}
    a = fea.mesh_from_request(p, tmp_path / "a.msh", req)
    b = fea.mesh_from_request(p, tmp_path / "b.msh", req)
    assert a["sha256"] == b["sha256"]


def test_closed_curves_are_split_into_arcs():
    with gmsh_session("c"):
        gmsh.model.occ.addDisk(0, 0, 0, 1, 1)
        gmsh.model.occ.synchronize()
        assert flow_mesh._split_closed_curves(2) == 3
        assert len(gmsh.model.getEntities(1)) == 4 and len(gmsh.model.getEntities(2)) == 1


def test_local_refinement_goes_below_the_global_floor(tmp_path):
    """Flow: h keeps the global floor (diagonal / 400 = 5.6 mm on the DFG channel) but a refinement
    may go 5x below it; gmsh's Mesh.MeshSizeMin used to clamp it back to the floor."""
    p = fea.get_problem("cylinder_channel_dfg_2d1")
    m = fea.make_mesh(p, tmp_path / "m.msh", h=0.03, refinements=[{"group": "cylinder", "size": 0.002, "radius": 0.05}])
    assert m["ok"] and m["sizes_used"] == [0.002]
    assert m["size_by_group"]["cylinder"] < 0.003            # the floor would leave it at >= 5.6 mm
    coarse = fea.make_mesh(p, tmp_path / "c.msh", h=0.001)    # h itself is still floored
    assert coarse["h_used"] == pytest.approx(math.dist([0, 0], [2.2, 0.41]) / 400, rel=1e-3)
