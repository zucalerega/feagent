import math

import gmsh
import numpy as np
import pytest

from feagent.geometry import analyze, analyze_current_model, digest_text
from feagent.gmsh_session import gmsh_session, load_cad
from feagent.types import GeometryModel


def _close(a, b, tol=1e-6):
    return np.allclose(np.asarray(a, float), np.asarray(b, float), atol=tol)


@pytest.fixture(scope="module")
def bracket(parts_dir):
    return analyze(parts_dir / "bracket.step")


def test_plate_hole_detects_hole(parts_dir):
    m = analyze(parts_dir / "plate_hole.step")
    holes = [f for f in m.features if f.kind == "hole"]
    assert len(holes) == 1
    p = holes[0].params
    assert p["radius"] == pytest.approx(10, abs=1e-3)
    assert _close(p["axis"], (0, 0, 1), 1e-6)
    assert p["depth"] == pytest.approx(5, abs=1e-3)
    assert _close(p["axis_point"], (100, 50, 0), 1e-4)
    big = [f for f in m.faces if f.kind == "plane" and f.area > 1000]
    assert len(big) == 2
    assert sorted(round(f.normal[2]) for f in big) == [-1, 1]
    for f in big:
        assert f.area == pytest.approx(200 * 100 - math.pi * 100, rel=1e-6)
        assert _close(f.normal[:2], (0, 0))
    cyl = m.face(holes[0].faces[0])
    assert cyl.concave is True


def test_bracket_features(bracket):
    holes = [f for f in bracket.features if f.kind == "hole"]
    small = [h for h in holes if h.params["radius"] == pytest.approx(3.25, abs=1e-3)]
    assert len(small) == 4
    for h in small:
        assert _close(h.params["axis"], (0, 0, 1), 1e-6)
        assert h.params["depth"] == pytest.approx(6, abs=1e-3)
    big = [h for h in holes if h.params["radius"] == pytest.approx(5, abs=1e-3)]
    assert len(big) == 1
    assert _close(big[0].params["axis"], (1, 0, 0), 1e-6)
    assert len(holes) == 5
    fillets = [f for f in bracket.features if f.kind == "fillet"]
    assert len(fillets) >= 1
    f4 = [f for f in fillets if f.params["radius"] == pytest.approx(4, abs=1e-3)]
    assert f4 and f4[0].params["angle_span"] == pytest.approx(90, abs=2)
    face = bracket.face(f4[0].faces[0])
    assert face.concave is True
    assert _close(face.axis, (0, 1, 0), 1e-6)
    assert _close(face.axis_point, (10, 0, 10), 1e-4)
    assert not [f for f in bracket.features if f.kind in ("boss", "round")]


def test_outward_normals_cantilever(parts_dir):
    m = analyze(parts_dir / "cantilever.step")
    assert len(m.faces) == 6
    by_x = {round(f.center[0], 6): f for f in m.faces if abs(f.normal[0]) > 0.5}
    assert _close(by_x[0.0].normal, (-1, 0, 0))
    assert _close(by_x[100.0].normal, (1, 0, 0))
    for f in m.faces:  # every normal points away from the centroid of the box
        assert np.dot(np.subtract(f.center, (50, 5, 5)), f.normal) > 0
    t = m.tessellation
    assert t is not None and t.triangles.shape[1] == 3
    assert t.triangles.shape[0] == t.face_tag.shape[0] == t.normals.shape[0]
    # triangles oriented outward: geometric normal agrees with stored normal and face normal
    p = t.points[t.triangles]
    cr = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    assert np.all(np.einsum("ij,ij->i", cr, t.normals) > 0)
    fn = {f.tag: np.array(f.normal) for f in m.faces}
    assert np.all(np.einsum("ij,ij->i", t.normals, np.array([fn[k] for k in t.face_tag])) > 0.999)


def test_hole_and_boss_concavity(tmp_path):
    with gmsh_session():
        occ = gmsh.model.occ
        plate = occ.addBox(0, 0, 0, 40, 40, 4)
        boss = occ.addCylinder(30, 30, 4, 0, 0, 6, 4)
        out, _ = occ.fuse([(3, plate)], [(3, boss)])
        hole = occ.addCylinder(10, 10, -1, 0, 0, 6, 3)
        occ.cut(out, [(3, hole)])
        occ.synchronize()
        m = analyze_current_model()
    kinds = sorted(f.kind for f in m.features)
    assert kinds == ["boss", "hole"]
    boss_f = next(f for f in m.features if f.kind == "boss")
    assert boss_f.params["radius"] == pytest.approx(4, abs=1e-3)
    assert _close(boss_f.params["axis_point"], (30, 30, 0), 1e-4)
    assert m.face(boss_f.faces[0]).concave is False
    hole_f = next(f for f in m.features if f.kind == "hole")
    assert m.face(hole_f.faces[0]).concave is True


def test_thin_cup_thin_wall_hazard(parts_dir):
    m = analyze(parts_dir / "thin_cup.step", ref_size=4)
    thin = [h for h in m.hazards if h.kind == "thin_wall"]
    assert thin
    for h in thin:
        assert h.value == pytest.approx(1.0, abs=0.1)
        assert h.threshold == pytest.approx(8.0)
        assert h.suggestion
    faces = {t for h in thin for _, t in h.entities}
    # the five outer and five inner wall/floor faces are thin; the rim strips are not
    assert len(faces) == 10
    for t in faces:
        assert m.face(t).thickness == pytest.approx(1.0, abs=0.1)


def test_sliver_hazards(parts_dir):
    m = analyze(parts_dir / "sliver.step", ref_size=2)
    kinds = {h.kind for h in m.hazards}
    assert kinds & {"sliver_face", "small_face"}
    short = [h for h in m.hazards if h.kind == "short_edge"]
    assert short
    assert all(h.value == pytest.approx(0.05, abs=1e-6) for h in short)
    assert all(h.entities[0][0] == 1 for h in short)


def test_two_touching_boxes_fragment(tmp_path):
    path = tmp_path / "two_boxes.step"
    with gmsh_session():
        gmsh.model.occ.addBox(0, 0, 0, 10, 10, 10)
        gmsh.model.occ.addBox(10, 0, 0, 10, 10, 10)
        gmsh.model.occ.synchronize()
        gmsh.write(str(path))
    m = analyze(path)
    assert len(m.volumes) == 2
    shared = [f for f in m.faces if len(f.volumes) == 2]
    assert len(shared) == 1
    assert shared[0].center[0] == pytest.approx(10)
    iface = [h for h in m.hazards if h.kind == "multi_volume_interface"]
    assert iface and iface[0].entities == [(2, shared[0].tag)]
    assert len(m.faces) == 11


def test_no_volume_hazard():
    with gmsh_session():
        gmsh.model.occ.addRectangle(0, 0, 0, 10, 5)
        gmsh.model.occ.synchronize()
        m = analyze_current_model()
    assert len(m.volumes) == 0
    assert any(h.kind == "no_volume" for h in m.hazards)
    assert len(m.faces) == 1 and m.faces[0].area == pytest.approx(50)


def test_digest_text_sections(bracket):
    s = digest_text(bracket)
    for sec in ("FEATURES", "PLANAR FACES", "HAZARDS", "FACES"):
        assert sec in s
    assert s.index("FEATURES") < s.index("PLANAR FACES") < s.index("HAZARDS") < s.index("\nFACES")
    assert "4× hole" in s
    assert "r=3.25" in s and "+z" in s
    assert len(s) < 6000
    full = digest_text(bracket, detail="full")
    assert "EDGES" in full and len(full) > len(s) - 1
    trunc = digest_text(bracket, max_faces=3)
    assert "more (use detail=full)" in trunc
    with pytest.raises(ValueError):
        digest_text(bracket, detail="bogus")


def test_model_json_roundtrip(bracket):
    import json

    d = json.loads(json.dumps(bracket.to_dict()))
    assert GeometryModel.from_dict(d).to_dict() == bracket.to_dict()


def test_no_mesh_left_behind(parts_dir):
    with gmsh_session():
        load_cad(parts_dir / "bracket.step")
        before = gmsh.option.getNumber("Mesh.MeshSizeMax")
        m = analyze_current_model(source="bracket")
        assert gmsh.model.mesh.getNodes()[0].size == 0
        assert gmsh.option.getNumber("Mesh.MeshSizeMax") == before
    assert m.tessellation is not None and m.tessellation.triangles.size > 0


def test_analyze_without_tessellation(parts_dir):
    m = analyze(parts_dir / "plate_hole.step", tessellate=False)
    assert m.tessellation is None
    holes = [f for f in m.features if f.kind == "hole"]
    assert len(holes) == 1 and holes[0].params["radius"] == pytest.approx(10, abs=1e-3)
    assert all(f.thickness is None for f in m.faces)
    top = [f for f in m.faces if f.kind == "plane" and f.center[2] == pytest.approx(5)]
    assert len(top) == 1 and _close(top[0].normal, (0, 0, 1))


def test_split_hole_faces_merge_into_one_hole():
    with gmsh_session():
        occ = gmsh.model.occ
        plate = occ.addBox(0, 0, 0, 40, 30, 5)
        halves = [occ.addCylinder(20, 15, -1, 0, 0, 7, 4, angle=math.pi) for _ in range(2)]
        occ.rotate([(3, halves[1])], 20, 15, 0, 0, 0, 1, math.pi)
        occ.cut([(3, plate)], [(3, t) for t in halves])
        occ.synchronize()
        m = analyze_current_model()
    cyl = [f for f in m.faces if f.kind == "cylinder"]
    assert len(cyl) == 2 and all(f.angle_span == pytest.approx(180) for f in cyl)
    assert len(m.features) == 1
    hole = m.features[0]
    assert hole.kind == "hole" and hole.faces == sorted(f.tag for f in cyl)
    assert hole.params["angle_span"] == pytest.approx(360)
    assert hole.params["radius"] == pytest.approx(4) and hole.params["depth"] == pytest.approx(5)
