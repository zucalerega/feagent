import gmsh
import numpy as np
import pytest

from feagent import gmsh_util as gu
from feagent.gates import failed, passed, skipped, threshold_gate, warned
from feagent.gmsh_session import gmsh_session
from feagent.graph import connected_groups
from feagent.vec import (angle_deg, axis_angle_deg, bbox_diag, box_dist, dist, extreme_gap, unit,
                         unit_rows, vec3)

BB = ((0.0, 0.0, 0.0), (1.0, 2.0, 3.0))


def test_vec_basics():
    assert vec3(np.array([1, 2, 3])) == (1.0, 2.0, 3.0) and type(vec3([1, 2, 3])[0]) is float
    assert dist((0, 0, 0), (3, 4, 0)) == 5.0
    assert angle_deg((1, 0, 0), (0, 2, 0)) == pytest.approx(90.0)
    assert angle_deg((1, 0, 0), (-1, 0, 0)) == pytest.approx(180.0)
    assert axis_angle_deg((0, 0, 1), (0, 0, -1)) == pytest.approx(0.0)
    assert axis_angle_deg((1, 0, 0), (1, 1, 0)) == pytest.approx(45.0)


def test_unit_forms():
    assert unit([3, 0, 4]).tolist() == [0.6, 0.0, 0.8] and unit([0, 0, 0]).tolist() == [0, 0, 0]
    rows = unit_rows([[0, 2, 0], [0, 0, 0]])
    assert rows.tolist() == [[0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]


def test_connected_groups():
    assert connected_groups([5, 1, 3, 2, 4], [(3, 1), (4, 5), (5, 3)]) == [[1, 3, 4, 5], [2]]
    assert connected_groups([], []) == []


def test_bbox_helpers():
    assert box_dist((0.5, 1, 1), BB) == 0.0 and box_dist((-3, 1, -4), BB) == 5.0
    assert bbox_diag(((0, 0, 0), (2, 3, 6))) == 7.0 and type(bbox_diag(BB)) is float
    face = ((0, 0, 3), (1, 2, 3))
    assert extreme_gap(face, BB, "zmax") == 0.0
    assert extreme_gap(face, BB, "zmin") == 3.0
    assert extreme_gap(((0, 0, 0), (0, 2, 3)), BB, "xmin") == 0.0


def test_gate_constructors():
    assert passed("a", "A", "ok").status == "pass"
    assert warned("a", "A", "m", 1.0, 2.0, "h").hint == "h"
    assert failed("a", "A", "m").status == "fail"
    g = skipped("a", "A", "none", threshold=1.0)
    assert (g.status, g.value, g.threshold) == ("skip", None, 1.0)


def test_threshold_gate():
    kw = dict(message_fmt="v {value:.1f} lim {limit} x {x}", hint="fix it", x="X")
    g = threshold_gate("t", "T", 2.0, 1, **kw)
    assert (g.status, g.message, g.threshold, g.hint) == ("fail", "v 2.0 lim 1 x X", 1.0, "fix it")
    assert threshold_gate("t", "T", 0.5, 1, **kw).hint == ""
    g = threshold_gate("t", "T", 0.5, 1, worse="below", fail_status="warn", **kw)
    assert g.status == "warn"
    assert threshold_gate("t", "T", 1.0, 1, worse="below", **kw).status == "pass"


def test_gmsh_util_on_box_mesh():
    with gmsh_session():
        gmsh.model.occ.addBox(0, 0, 0, 1, 1, 1)
        gmsh.model.occ.synchronize()
        gmsh.model.addPhysicalGroup(2, [1, 2], name="ends")
        gmsh.option.setNumber("Mesh.MeshSizeMax", 0.5)
        gmsh.model.mesh.generate(3)
        assert gu.entity_tags(3) == [1] and len(gu.entity_tags(2)) == 6
        tags, coords, lookup = gu.node_table()
        assert coords.shape == (len(tags), 3) and (lookup[tags] == np.arange(len(tags))).all()
        tets = gu.corner_rows(3, 1, gu.TET_TYPES, 4, lookup)
        assert tets.shape[1] == 4 and tets.min() >= 0
        assert gu.corner_rows(3, 1, gu.TRI_TYPES, 3, lookup).shape == (0, 3)
        assert [(d, n, e) for d, n, e in gu.physical_groups()] == [(2, "ends", [1, 2])]
        _, etags, _ = next(gu.element_blocks(3, 1, gu.TET_TYPES))
        q = gu.element_qualities(etags[:3], ("minSICN", "volume"))
        assert set(q) == {"minSICN", "volume"} and len(q["volume"]) == 3
        assert gu.element_qualities([], ("gamma",))["gamma"].shape == (0,)
        assert all(f in range(1, 7) for f in gu.element_faces(etags[0]))
        face = gu.closure(2, 1)
        assert (2, 1) in face and len(face) == 1 + 4 + 4  # face, its 4 curves, 4 corners
        assert len(gu.closure(3, 1)) == 1 + 6 + 12 + 8
        assert gu.node_entity(tags[0])[0] == 0
        assert [t for t, _, _ in gu.element_blocks(2, 1)] == [2]  # types=None: every type


def test_pinned_options_restores_on_error():
    with gmsh_session():
        gmsh.option.setNumber("Mesh.MeshSizeMax", 7.0)
        with pytest.raises(RuntimeError):
            with gu.pinned_options({"Mesh.MeshSizeMax": 0.5, "Mesh.ElementOrder": 2}):
                assert gmsh.option.getNumber("Mesh.MeshSizeMax") == 0.5
                raise RuntimeError("boom")
        assert gmsh.option.getNumber("Mesh.MeshSizeMax") == 7.0
        assert gmsh.option.getNumber("Mesh.ElementOrder") == 1
