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


def test_area_fidelity_fails_on_wrong_cad_area(cantilever):
    r = _cant_result(cantilever)
    r.geometry.faces[0].area = 150.0
    g = next(g for g in assess_mesh(cantilever, r, Q, "fea").gates if g.id == "area_fidelity")
    assert g.status == "fail" and "quality.max_area_error" in g.hint


def test_area_fidelity_fails_on_missing_group(cantilever):
    r = _cant_result(cantilever)
    r.groups["ghost"] = GroupResult("ghost", 2, [1], 0, 100.0)
    rep = assess_mesh(cantilever, r, Q, "fea")
    g = next(g for g in rep.gates if g.id == "area_fidelity")
    assert g.status == "fail" and "ghost" in g.message


def test_volume_fidelity_skips_without_geometry(cantilever):
    r = _cant_result(cantilever)
    r.geometry.volumes = []
    g = next(g for g in assess_mesh(cantilever, r, Q, "fea").gates if g.id == "volume_fidelity")
    assert g.status == "skip"


def test_sicn_gate_fails_when_threshold_above_min(cantilever):
    q = SimpleNamespace(**{**vars(Q), "min_sicn": 0.999})
    g = next(g for g in assess_mesh(cantilever, _cant_result(cantilever), q, "fea").gates
             if g.id == "sicn")
    assert g.status == "fail" and "quality.min_sicn" in g.hint


def test_sicn_per_tet_aligned_with_tets(cantilever):
    from feagent.meshdata import read_msh
    rep = assess_mesh(cantilever, _cant_result(cantilever), Q, "fea")
    md = read_msh(cantilever)
    i = int(np.argmin(rep.sicn_per_tet))
    with gmsh_session():
        gmsh.open(str(cantilever))
        ref = gmsh.model.mesh.getElementQualities([int(md.tet_tags[i])], "minSICN")[0]
    assert rep.sicn_per_tet[i] == pytest.approx(ref)
    assert rep.worst[0].tag == int(md.tet_tags[i])
    # centroid of worst element matches tets[i] corner mean
    assert np.allclose(rep.worst[0].centroid, md.points[md.tets[i]].mean(axis=0))


def test_hint_spec_keys_exist_in_schema(cantilever, tmp_path):
    """Every dotted spec key cited in a hint must be accepted by parse_spec."""
    import re
    from pathlib import Path

    from feagent.spec import parse_spec
    q = SimpleNamespace(min_sicn=2.0, min_gamma=2.0, max_volume_error=0.0, max_area_error=0.0,
                        max_non_ortho=0.0, max_skewness=0.0, thin_wall_min_elements=2)
    r = _cant_result(cantilever)
    r.geometry.faces[0].area = 150.0
    r.geometry.volumes[0].volume = 12000.0
    r.geometry.hazards.append(Hazard(kind="thin_wall", entities=[(2, 1)], value=2.0,
                                     threshold=None, location=None, message="", suggestion=""))
    hints = " ".join(g.hint for g in assess_mesh(cantilever, r, q, "cfd").gates)
    hints += " " + " ".join(g.hint for g in assess_mesh(
        cantilever, r, q, "fea").gates)
    keys = set(re.findall(r"\b(?:size|element|algorithm|quality)\.[a-z_0-9]+", hints))
    keys |= {"heal", "refine"}
    assert {"element.optimize", "algorithm.mesh3d", "size.curvature"} <= keys
    assert not re.search(r"algorithm_3d|\boptimize:|\border:", hints.replace("element.optimize:", "")
                         .replace("element.order:", ""))
    base = {"name": "t", "geometry": "p.step", "analysis": "fea", "heal": True,
            "size": {"global": 3.0}, "refine": [], "quality": {}, "element": {}, "algorithm": {}}
    (tmp_path / "p.step").write_text("x")
    for k in sorted(keys - {"heal", "refine"}):
        sect, field_ = k.split(".")
        data = {**base, sect: {**base[sect], field_: {"quality.min_sicn": 0.1}.get(k) or
                               {"element.optimize": "netgen", "element.order": 2,
                                "algorithm.mesh3d": 1, "algorithm.mesh2d": 6}.get(k, 1)}}
        data["quality"] = data["quality"] if sect == "quality" else {}
        parse_spec(data, tmp_path / "spec.yaml")  # raises SpecError on an unknown key


def test_thin_wall_fallback_matches_surface_triangles(tmp_path):
    """msh without face elements: edge lengths come from the tet skin. Triangles touching the
    face's boundary curves must count, so the result equals the true surface-mesh mean."""
    from feagent import gmsh_util as gu
    from feagent.meshdata import read_msh
    from feagent.quality import _face_edge_lengths, _mean_edge_length

    p, ref = tmp_path / "slab.msh", {}
    with gmsh_session():
        gmsh.model.occ.addBox(0, 0, 0, 10, 10, 1)
        gmsh.model.occ.synchronize()
        gmsh.model.addPhysicalGroup(3, [1], name="domain")  # no surface groups saved
        gmsh.option.setNumber("Mesh.MeshSizeMax", 1.0)
        gmsh.model.mesh.generate(3)
        _, pts, lookup = gu.node_table()
        for f in gu.entity_tags(2):
            ref[f] = _mean_edge_length(gu.corner_rows(2, f, gu.TRI_TYPES, 3, lookup), pts)
        gmsh.write(str(p))
    md = read_msh(p)
    assert not md.tris
    with gmsh_session():
        gmsh.open(str(p))
        got = _face_edge_lengths(sorted(ref), md.tets, md.points)
    assert got == pytest.approx(ref)
