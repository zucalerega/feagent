import numpy as np
import pytest

from feagent.meshdata import MeshData
from feagent.render import RenderResult, render_faces, render_field, render_groups, render_quality
from feagent.types import FaceInfo, GeometryModel, Tessellation


def _face(tag, center, area):
    return FaceInfo(tag=tag, kind="plane", area=area, center=center, bbox=(center, center),
                    normal=(0, 0, 1), radius=None, axis=None, axis_point=None, concave=None,
                    angle_span=None, volumes=[1], edges=[], neighbors=[])


@pytest.fixture
def model():
    pts = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], float)
    tris = np.array([[0, 1, 2], [0, 2, 3]])
    tess = Tessellation(points=pts, triangles=tris, face_tag=np.array([1, 2]),
                        normals=np.tile([0, 0, 1.0], (2, 1)))
    return GeometryModel(
        source="syn", units="mm", bbox=((0, 0, 0), (1, 1, 0)), diag=1.5, ref_size=0.5,
        volumes=[], faces=[_face(1, (2 / 3, 1 / 3, 0), 0.5), _face(2, (1 / 3, 2 / 3, 0), 0.5)],
        edges=[], features=[], hazards=[], tessellation=tess)


@pytest.fixture
def mesh():
    pts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], float)
    return MeshData(
        points=pts, tets=np.array([[0, 1, 2, 3]]), tet_tags=np.array([1]),
        tet_volume=np.array([1 / 6]),
        tris={"fixed": np.array([[0, 1, 2]]), "load": np.array([[0, 1, 3]])},
        tri_face={"fixed": np.array([1]), "load": np.array([2])},
        volume_groups={"body": [1]}, order=1)


def _width(path):
    with open(path, "rb") as f:
        h = f.read(24)
    assert h[:8] == b"\x89PNG\r\n\x1a\n"
    return int.from_bytes(h[16:20], "big"), int.from_bytes(h[20:24], "big")


def test_render_faces_png(model, tmp_path):
    r = render_faces(model, tmp_path / "f.png")
    assert _width(r.path) == (1600, 1200)
    assert "f1" in r.caption
    assert RenderResult.from_dict(r.to_dict()) == r


def test_render_faces_highlight_caption(model, tmp_path):
    r = render_faces(model, tmp_path / "h.png", highlight=[2])
    assert "highlighted f2" in r.caption


def test_render_groups_and_quality_png(mesh, tmp_path):
    g = render_groups(mesh, tmp_path / "g.png")
    assert _width(g.path) == (1600, 1200) and "fixed" in g.caption
    q = render_quality(mesh, np.array([0.3]), tmp_path / "q.png")
    assert _width(q.path) == (1600, 1200) and "0.3" in q.caption


def test_render_field_png(mesh, tmp_path):
    r = render_field(mesh, np.arange(4.0), "u", tmp_path / "u.png", warp=np.zeros((4, 3)))
    assert _width(r.path) == (1600, 1200) and "u" in r.caption
