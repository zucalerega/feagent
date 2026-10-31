from types import SimpleNamespace

import gmsh
import numpy as np
import pytest

from feagent.gmsh_session import gmsh_session
from feagent.quality import assess_mesh, tet_non_ortho_skewness
from feagent.types import (FaceInfo, GeometryModel, GroupResult, Hazard,
                           MeshResult, VolumeInfo)

Q = SimpleNamespace(min_sicn=0.1, min_gamma=0.05, max_volume_error=0.01,
                    max_area_error=0.02, max_non_ortho=70.0, max_skewness=4.0,
                    thin_wall_min_elements=2)


def _face(tag, area, volumes):
    return FaceInfo(tag=tag, kind="plane", area=area, center=(0, 0, 0),
                    bbox=((0, 0, 0), (0, 0, 0)), normal=None, radius=None, axis=None,
                    axis_point=None, concave=None, angle_span=None, volumes=volumes,
                    edges=[], neighbors=[])


def _vol(tag, v):
    return VolumeInfo(tag=tag, volume=v, center=(0, 0, 0), bbox=((0, 0, 0), (0, 0, 0)), faces=[])


def _result(path, volumes, faces, groups, hazards=()):
    geo = GeometryModel(source="x", units="mm", bbox=((0, 0, 0), (1, 1, 1)), diag=1.0,
                        ref_size=1.0, volumes=volumes, faces=faces, edges=[], features=[],
                        hazards=list(hazards))
    return MeshResult(ok=True, spec_path="", out_dir="", mesh_path=str(path), exports={},
                      groups=groups, n_nodes=0, n_elements=0, order=1, size_scale=1.0,
                      elapsed_s=0.0, errors=[], warnings=[], gates=[], geometry=geo)


@pytest.fixture
def cantilever(tmp_path):
    p = tmp_path / "c.msh"
    with gmsh_session():
        gmsh.model.occ.addBox(0, 0, 0, 100, 10, 10)
        gmsh.model.occ.synchronize()
        gmsh.model.addPhysicalGroup(2, [1], name="fixed")
        gmsh.model.addPhysicalGroup(3, [1], name="domain")
        gmsh.option.setNumber("Mesh.MeshSizeMax", 4)
        gmsh.model.mesh.generate(3)
        gmsh.write(str(p))
    return p


def _cant_result(p, vol=10000.0):
    return _result(p, [_vol(1, vol)], [_face(1, 100.0, [1])],
                   {"fixed": GroupResult("fixed", 2, [1], 0, 100.0)})


def test_good_mesh_passes(cantilever):
    rep = assess_mesh(cantilever, _cant_result(cantilever), Q, "fea")
    assert not [g for g in rep.gates if g.status == "fail"], rep.gates
    assert rep.inverted == 0
    assert rep.volume_error < 1e-6
    assert rep.group_area["fixed"][2] < 1e-6
    assert rep.components == rep.expected_components == 1
    assert len(rep.worst) == 10 and len(rep.sicn.hist) == 10
    assert len(rep.sicn_per_tet) == rep.n_tets
    assert rep.non_ortho is None
    ids = {g.id for g in rep.gates}
    assert {"inverted", "sicn", "gamma", "volume_fidelity", "area_fidelity",
            "connectivity", "thin_walls"} <= ids
    d = rep.to_dict()
    assert "sicn_per_tet" not in d
    assert rep.worst[0].faces


def test_volume_mismatch_fails(cantilever):
    rep = assess_mesh(cantilever, _cant_result(cantilever, 12000.0), Q, "fea")
    g = next(g for g in rep.gates if g.id == "volume_fidelity")
    assert g.status == "fail" and "quality.max_volume_error" in g.hint


def test_two_disjoint_boxes_connectivity(tmp_path):
    p = tmp_path / "two.msh"
    with gmsh_session():
        gmsh.model.occ.addBox(0, 0, 0, 10, 10, 10)
        gmsh.model.occ.addBox(50, 0, 0, 10, 10, 10)
        gmsh.model.occ.synchronize()
        gmsh.model.addPhysicalGroup(3, [1, 2], name="domain")
        gmsh.option.setNumber("Mesh.MeshSizeMax", 5)
        gmsh.model.mesh.generate(3)
        gmsh.write(str(p))
    vols = [_vol(1, 1000.0), _vol(2, 1000.0)]
    one = _result(p, vols, [_face(1, 100, [1, 2])], {})
    two = _result(p, vols, [_face(1, 100, [1]), _face(2, 100, [2])], {})
    g1 = next(g for g in assess_mesh(p, one, Q, "fea").gates if g.id == "connectivity")
    g2 = next(g for g in assess_mesh(p, two, Q, "fea").gates if g.id == "connectivity")
    assert g1.status == "fail"
    assert g2.status == "pass"
    area = next(g for g in assess_mesh(p, two, Q, "fea").gates if g.id == "area_fidelity")
    assert area.status == "skip"


def test_non_ortho_regular_tet():
    pts = np.array([[0, 0, 0], [1, 0, 0], [0.5, 0.8660254, 0],
                    [0.5, 0.28867513, 0.81649658], [0.5, 0.28867513, -0.81649658]], float)
    tets = np.array([[0, 1, 2, 3], [0, 1, 2, 4]])
    no, sk = tet_non_ortho_skewness(pts, tets)
    assert len(no) == 1 and len(sk) == 1
    assert no[0] < 40.0 and no[0] < 1e-6
    assert sk[0] < 1e-6
    # skewed neighbour raises both
    pts[4] = [1.5, 0.28867513, -0.81649658]
    no2, sk2 = tet_non_ortho_skewness(pts, tets)
    assert no2[0] > 10 and sk2[0] > 0.1


def test_cfd_gates_present(cantilever):
    rep = assess_mesh(cantilever, _cant_result(cantilever), Q, "cfd")
    ids = {g.id for g in rep.gates}
    assert {"non_ortho", "skewness"} <= ids
    assert rep.non_ortho is not None and rep.skewness is not None


def test_thin_wall_gate_warns(cantilever):
    haz = Hazard(kind="thin_wall", entities=[(2, 1)], value=2.0, threshold=None,
                 location=None, message="", suggestion="")
    r = _cant_result(cantilever)
    r.geometry.hazards.append(haz)
    rep = assess_mesh(cantilever, r, Q, "fea")
    g = next(g for g in rep.gates if g.id == "thin_walls")
    assert g.status == "warn" and "size.thin_wall_layers" in g.hint
