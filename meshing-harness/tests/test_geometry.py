import math
import time

import gmsh
import numpy as np
import pytest

from feagent import geometry
from feagent.errors import FeagentError
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
        assert "distance: 2}" in h.suggestion  # the mesher's thin-wall reach is 2t
    faces = {t for h in thin for _, t in h.entities}
    # the five outer and five inner wall/floor faces are thin; the rim strips are not
    assert len(faces) == 10
    for t in faces:
        assert m.face(t).thickness == pytest.approx(1.0, abs=0.1)


@pytest.mark.parametrize("factor", [0.99, 1.0, 1.01])
def test_hole_face_thickness_finds_the_thin_ligament(parts_dir, factor):
    """The r=5 load hole sits 5 mm below the bracket top; its other ligaments are 15-36 mm. The
    face thickness must see the 5 mm one and not jump with a tiny ref-size (tessellation) change."""
    from feagent.select import select_faces
    m = analyze(parts_dir / "bracket.step", ref_size=2.6 * factor)
    (hole,) = select_faces(m, {"kind": "cylinder", "radius": 5})
    t = m.face(hole).thickness
    assert 5.0 <= t < 7.0 and t == pytest.approx(5.25, rel=0.03)


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
        m = analyze_current_model(ref_size=5)
    cyl = [f for f in m.faces if f.kind == "cylinder"]
    assert len(cyl) == 2 and all(f.angle_span == pytest.approx(180) for f in cyl)
    # the other half of the same hole is not a "gap" (consistent with a single-face hole)
    assert not [h for h in m.hazards if h.kind == "narrow_gap"]
    assert len(m.features) == 1
    hole = m.features[0]
    assert hole.kind == "hole" and hole.faces == sorted(f.tag for f in cyl)
    assert hole.params["angle_span"] == pytest.approx(360)
    assert hole.params["radius"] == pytest.approx(4) and hole.params["depth"] == pytest.approx(5)


def _holed_plate(n: int, distinct: bool = False) -> None:
    occ = gmsh.model.occ
    k = int(math.ceil(math.sqrt(n)))
    plate = occ.addBox(0, 0, 0, 12.0 * k, 12.0 * k, 4)
    tools = [(3, occ.addCylinder(6 + 12 * (i % k), 6 + 12 * (i // k), -1, 0, 0, 6,
                                 2.0 + (0.01 * i if distinct else 0.0))) for i in range(n)]
    occ.cut([(3, plate)], tools)
    occ.synchronize()


def test_many_holes_analyze_fast():
    with gmsh_session():
        _holed_plate(200)
        t0 = time.perf_counter()
        m = analyze_current_model()
        dt = time.perf_counter() - t0
    assert dt < 10.0
    holes = [f for f in m.features if f.kind == "hole"]
    assert len(holes) == 200 and all(m.face(h.faces[0]).concave for h in holes)
    top = [f for f in m.faces if f.kind == "plane" and f.center[2] == pytest.approx(4)]
    assert len(top) == 1 and _close(top[0].normal, (0, 0, 1))
    assert "200× hole r=2 axis +z" in digest_text(m)


def test_summary_digest_is_token_bounded():
    with gmsh_session():
        _holed_plate(400, distinct=True)
        m = analyze_current_model()
    assert len(m.faces) > 400 and len(m.features) == 400
    s = digest_text(m)
    assert len(s) < 8000
    assert "more feature patterns (use detail=full)" in s and "more (use detail=full)" in s
    for hz in m.hazards:  # id lists inside hazard texts are capped
        assert len(hz.message) < 400 and len(hz.suggestion) < 500
    assert len(digest_text(m, detail="full")) > len(s)


@pytest.mark.parametrize("preset", [("Mesh.ElementOrder", 2), ("Mesh.RecombineAll", 1),
                                    ("Mesh.MeshSizeMin", 30), ("Mesh.Algorithm", 8)])
def test_preset_mesh_options_do_not_break_tessellation(parts_dir, preset):
    with gmsh_session():
        load_cad(parts_dir / "thin_cup.step")
        gmsh.option.setNumber(*preset)
        m = analyze_current_model(ref_size=4)
        assert gmsh.option.getNumber(preset[0]) == preset[1]
    assert m.tessellation is not None
    assert [h for h in m.hazards if h.kind == "thin_wall"][0].value == pytest.approx(1.0, abs=0.1)
    assert not [h for h in m.hazards if h.kind == "analysis_incomplete"]


def test_per_surface_recombine_still_tessellates(parts_dir):
    with gmsh_session():
        load_cad(parts_dir / "cantilever.step")
        for _, f in gmsh.model.getEntities(2):
            gmsh.model.mesh.setRecombine(2, f)
        m = analyze_current_model()
    p = m.tessellation.points[m.tessellation.triangles]
    cr = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    assert np.all(np.einsum("ij,ij->i", cr, m.tessellation.normals) > 0)
    assert all(f.thickness is not None for f in m.faces)


def test_existing_mesh_is_not_destroyed(parts_dir):
    with gmsh_session():
        load_cad(parts_dir / "cantilever.step")
        gmsh.model.mesh.generate(2)
        n0 = gmsh.model.mesh.getNodes()[0].size
        with pytest.raises(FeagentError) as ei:
            analyze_current_model()
        assert "before meshing" in ei.value.hint
        m = analyze_current_model(tessellate=False)
        assert gmsh.model.mesh.getNodes()[0].size == n0
    assert len(m.faces) == 6


def test_tessellation_failure_is_reported(parts_dir, monkeypatch):
    def boom(dim=3):
        raise Exception("synthetic mesher failure")

    with gmsh_session():
        load_cad(parts_dir / "thin_cup.step")
        monkeypatch.setattr(gmsh.model.mesh, "generate", boom)
        m = analyze_current_model(ref_size=4)
    inc = [h for h in m.hazards if h.kind == "analysis_incomplete"]
    assert inc and "synthetic mesher failure" in inc[0].message
    assert len(inc[0].entities) == len(m.faces)
    assert m.tessellation is None and all(f.thickness is None for f in m.faces)
    assert _close(next(f for f in m.faces if f.center[0] == 0).normal, (-1, 0, 0))
    assert "NOTE: temporary tessellation failed" in digest_text(m)


def test_unresolved_shape_is_reported(parts_dir, monkeypatch):
    def boom(fd):
        raise RuntimeError("curvature query failed")

    monkeypatch.setattr(geometry, "_curvatures", boom)
    m = analyze(parts_dir / "plate_hole.step")
    cyl = next(f for f in m.faces if f.kind == "cylinder")
    assert cyl.radius is None and cyl.normal is not None
    inc = [h for h in m.hazards if h.kind == "analysis_incomplete"]
    assert inc and (2, cyl.tag) in inc[0].entities


@pytest.mark.parametrize("bad", [0, -1.0])
def test_bad_ref_size_raises(parts_dir, bad):
    with pytest.raises(ValueError, match="ref_size"):
        analyze(parts_dir / "cantilever.step", ref_size=bad)


def _plate_with_hole():
    occ = gmsh.model.occ
    occ.cut([(3, occ.addBox(0, 0, 0, 40, 30, 5))], [(3, occ.addCylinder(20, 15, -1, 0, 0, 7, 4))])


def _three_box_fragment():
    occ = gmsh.model.occ
    boxes = [occ.addBox(x, 0, 0, 10, 10, 10) for x in (20, 10, 0)]
    occ.fragment([(3, boxes[0])], [(3, b) for b in boxes[1:]])


def _normals_of(build, **kw):
    """Orientation facts per face: outward normal of planes, concavity of curved faces (their
    reported normal is taken at a sample point that differs between runs)."""
    with gmsh_session():
        build()
        gmsh.model.occ.synchronize()
        m = analyze_current_model(**kw)
    return {f.tag: tuple(np.round(f.normal, 9)) if f.kind == "plane" else f.concave
            for f in m.faces}, m


@pytest.mark.parametrize("build", [_plate_with_hole, _three_box_fragment])
@pytest.mark.parametrize("invert", [False, True])
def test_orientation_survives_partial_tessellation(build, invert, monkeypatch):
    truth, m = _normals_of(build, tessellate=False)  # exact isInside orientation
    assert any(len(f.volumes) == 2 for f in m.faces) or build is _plate_with_hole
    if invert:  # make getNormal point the wrong way so the parity rays must flip it
        normals_at = geometry._normals_at
        monkeypatch.setattr(geometry, "_normals_at", lambda f, uv: -normals_at(f, uv))
    face_triangles = geometry._face_triangles
    for drop in [None, *truth]:
        monkeypatch.setattr(geometry, "_face_triangles", lambda f, d=drop: (
            np.zeros((0, 3), dtype=np.int64) if f == d else face_triangles(f)))
        got, m2 = _normals_of(build)
        bad = [t for t in truth if got[t] != truth[t]]
        assert not bad, f"dropped face {drop}: wrong normals on faces {bad}"
        assert (drop is None) != any(h.kind == "analysis_incomplete" for h in m2.hazards)


def test_full_tessellation_needs_no_isinside_probes(parts_dir, monkeypatch):
    calls = []
    is_inside = gmsh.model.isInside
    def counting(dim, *a, **k):
        calls.extend([1] if dim == 3 else [])
        return is_inside(dim, *a, **k)

    monkeypatch.setattr(gmsh.model, "isInside", counting)
    with gmsh_session():
        _holed_plate(100)
        m = analyze_current_model()
    assert len(m.faces) > 100 and len(calls) == 0
    with gmsh_session():
        _holed_plate(100)
        m = analyze_current_model(tessellate=False)
    assert 0 < len(calls) <= 2 * len(m.faces)  # at most one probe pair per face (simple faces)
    assert all(m.face(h.faces[0]).concave for h in m.features)
