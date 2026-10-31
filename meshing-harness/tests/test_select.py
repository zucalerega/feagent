import pytest

from feagent.errors import QueryError
from feagent.select import describe_faces, explain_miss, select_faces
from feagent.types import FaceInfo, Feature, GeometryModel


def _plane(tag, normal, bbox, center, nbrs):
    return FaceInfo(tag=tag, kind="plane", area=1.0, center=center, bbox=bbox, normal=normal,
                    radius=None, axis=None, axis_point=None, concave=None, angle_span=None,
                    volumes=[1], edges=[], neighbors=nbrs)


@pytest.fixture
def model():
    faces = [
        _plane(1, (-1, 0, 0), ((0, 0, 0), (0, 1, 1)), (0, .5, .5), [3, 4, 5, 6]),
        _plane(2, (1, 0, 0), ((1, 0, 0), (1, 1, 1)), (1, .5, .5), [3, 4, 5, 6]),
        _plane(3, (0, -1, 0), ((0, 0, 0), (1, 0, 1)), (.5, 0, .5), [1, 2, 5, 6]),
        _plane(4, (0, 1, 0), ((0, 1, 0), (1, 1, 1)), (.5, 1, .5), [1, 2, 5, 6]),
        _plane(5, (0, 0, -1), ((0, 0, 0), (1, 1, 0)), (.5, .5, 0), [1, 2, 3, 4, 7]),
        _plane(6, (0, 0, 1), ((0, 0, 1), (1, 1, 1)), (.5, .5, 1), [1, 2, 3, 4, 7]),
        FaceInfo(tag=7, kind="cylinder", area=0.0628, center=(.5, .5, .5),
                 bbox=((.4, .4, 0), (.6, .6, 1)), normal=None, radius=0.1, axis=(0, 0, 1),
                 axis_point=(.5, .5, 0), concave=True, angle_span=6.28, volumes=[1],
                 edges=[], neighbors=[5, 6]),
    ]
    return GeometryModel(source="t", units="mm", bbox=((0, 0, 0), (1, 1, 1)), diag=3 ** .5,
                         ref_size=0.1, volumes=[], faces=faces, edges=[],
                         features=[Feature("hole", [7], {"radius": 0.1})], hazards=[])


def test_normal_and_at(model):
    assert select_faces(model, {"normal": [1, 0, 0]}) == [2]
    assert select_faces(model, {"at": "xmin"}) == [1]


def test_kind_radius_feature(model):
    assert select_faces(model, {"kind": "cylinder", "radius": 0.1}) == [7]
    assert select_faces(model, {"feature": "hole"}) == [7]
    assert select_faces(model, {"kind": ["cylinder", "plane"], "concave": True}) == [7]
    assert select_faces(model, {"axis": [0, 0, -1]}) == [7]


def test_combinators(model):
    assert select_faces(model, {"not": {"kind": "plane"}}) == [7]
    assert len(select_faces(model, {"any": [{"at": "xmin"}, {"at": "xmax"}]})) == 2
    assert select_faces(model, {"adjacent_to": {"ids": [7]}}) == [5, 6]
    assert select_faces(model, {"all": [{"kind": "plane"}, {"at": "zmax"}]}) == [6]
    assert len(select_faces(model, {})) == 7


def test_geometric_keys(model):
    assert select_faces(model, {"box": [[0.9, 0, 0], [1.1, 1, 1]]}) == [2]
    assert select_faces(model, {"near": [1, .5, .5], "within": 0.01}) == [2]
    assert select_faces(model, {"area_max": 0.1}) == [7]


def test_largest(model):
    assert select_faces(model, {"kind": "plane", "largest": 2}) == [1, 2]  # ties: lowest tag
    assert select_faces(model, {"smallest": 1}) == [7]


def test_unknown_key_raises(model):
    with pytest.raises(QueryError) as e:
        select_faces(model, {"nromal": [1, 0, 0]})
    assert "normal" in e.value.hint


def test_bad_value_raises(model):
    with pytest.raises(QueryError):
        select_faces(model, {"normal": [1, 0]})
    with pytest.raises(QueryError) as e:
        select_faces(model, {"at": "top"})
    assert "xmin, xmax, ymin, ymax, zmin, zmax" in e.value.hint and "bounding box" in e.value.hint


def test_explain_miss_lists_candidates(model):
    q = {"normal": [0.9962, 0.0872, 0], "tol_deg": 2}
    assert select_faces(model, q) == []
    text = explain_miss(model, q, 1)
    print(text)
    assert "expected 1" in text and "0 matched" in text
    assert "tag=2" in text and "off by 5" in text


def test_describe_faces(model):
    lines = describe_faces(model, [2, 7]).splitlines()
    assert len(lines) == 2 and "tag=2" in lines[0] and "axis=" in lines[1]


def test_nested_unknown_key_raises(model):
    with pytest.raises(QueryError) as e:
        select_faces(model, {"any": [{"kind": "plane"}, {"nromal": [1, 0, 0]}]})
    assert "normal" in e.value.hint
    with pytest.raises(QueryError):
        select_faces(model, {"not": {"adjacent_to": {"bogus": 1}}})


def test_area_min_relaxed_slack_is_ten_percent(model):
    assert "area_min: 0 matched / 6 relaxed" in explain_miss(model, {"area_min": 1.05}, 1)
    assert "area_min: 0 matched / 0 relaxed" in explain_miss(model, {"area_min": 1.5}, 1)


def test_explain_adjacent_to_names_inner_key(model):
    text = explain_miss(model, {"adjacent_to": {"ids": [7], "kind": "sphere"}}, 1)
    assert "sub-query 'adjacent_to'" in text
    assert "kind: 0 matched" in text and "ids: 1 matched" in text


def test_unknown_kind_is_rejected_with_a_suggestion(model):
    with pytest.raises(QueryError) as e:
        select_faces(model, {"kind": "cylindr"})
    assert "unknown face kind 'cylindr'" in e.value.message and "'cylinder'" in e.value.hint
    with pytest.raises(QueryError):
        select_faces(model, {"any": [{"kind": ["plane", "blob"]}]})


def test_geometry_kinds_are_known_query_kinds():
    from feagent.geometry import SURFACE_KINDS
    from feagent.types import FACE_KINDS
    assert set(SURFACE_KINDS.values()) | {"other"} <= set(FACE_KINDS)


def test_face_lookup_is_cached_and_not_serialized(model):
    import dataclasses
    assert model.face(7).kind == "cylinder" and model._by_tag is not None
    with pytest.raises(KeyError):
        model.face(99)
    assert "_by_tag" not in model.to_dict()
    assert GeometryModel.from_dict(model.to_dict()) == model  # the cache does not affect eq
    model.faces.append(dataclasses.replace(model.faces[0], tag=8))
    assert model.face(8).tag == 8  # rebuilt when the faces list changes
