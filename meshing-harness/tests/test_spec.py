from pathlib import Path

import pytest

from feagent.errors import SpecError
from feagent.spec import load_spec


def write(tmp_path: Path, body: str, geometry: bool = True) -> Path:
    d = tmp_path / "proj"
    d.mkdir(exist_ok=True)
    if geometry:
        (d / "part.step").touch()
    p = d / "mesh.yaml"
    p.write_text(body)
    return p


MIN = "geometry: part.step\nanalysis: fea\nsize: {global: 3}\n"


def test_minimal_spec_defaults(tmp_path):
    s = load_spec(write(tmp_path, MIN))
    assert s.name == "proj"
    assert s.order == 1 and s.optimize == "netgen"
    assert (s.algorithm_2d, s.algorithm_3d) == (6, 1)
    assert s.export == ["msh"]
    assert s.size.global_ == 3 and s.size.curvature == 12
    assert s.quality.min_sicn == 0.1 and s.quality.max_skewness == 4.0
    assert s.default_group is None and s.bcs_fea is None and s.bcs_cfd is None
    assert s.heal is False
    assert s.out_dir() == (tmp_path / "proj" / "out").resolve()


def test_cfd_default_group_walls(tmp_path):
    s = load_spec(write(tmp_path, "geometry: part.step\nanalysis: cfd\nsize: {global: 3}\n"))
    assert s.default_group == "walls"
    assert s.domain.type == "internal"


def test_unknown_key_error_names_path(tmp_path):
    with pytest.raises(SpecError) as e:
        load_spec(write(tmp_path, "geometry: part.step\nanalysis: fea\nsize: {globl: 3}\n"))
    assert e.value.message.startswith("size.globl")


def test_missing_group_reference(tmp_path):
    body = MIN + "groups:\n  a: {select: {kind: plane}}\nbcs:\n  fea: {fixed: [nope], loads: []}\n"
    with pytest.raises(SpecError, match="nope"):
        load_spec(write(tmp_path, body))


def test_expect_range(tmp_path):
    body = MIN + ("groups:\n  a: {select: {kind: plane}, expect: [2, 4]}\n"
                  "  b: {select: {kind: plane}, expect: 1}\n")
    s = load_spec(write(tmp_path, body))
    assert s.groups["a"].expect == (2, 4)
    assert s.groups["b"].expect == 1


def test_geometry_must_exist(tmp_path):
    with pytest.raises(SpecError) as e:
        load_spec(write(tmp_path, MIN, geometry=False))
    assert e.value.message.startswith("geometry")


def test_bad_select_key_and_bcs_mismatch(tmp_path):
    with pytest.raises(SpecError, match=r"groups\.a\.select\.bogus"):
        load_spec(write(tmp_path, MIN + "groups:\n  a: {select: {bogus: 1}}\n"))
    with pytest.raises(SpecError, match=r"bcs\.cfd"):
        load_spec(write(tmp_path, MIN + "bcs:\n  cfd: {inlet: [], outlet: []}\n"))


def test_refine_and_default_group_validation(tmp_path):
    g = "groups:\n  a: {select: {kind: plane}}\n"
    with pytest.raises(SpecError, match=r"refine\[0\]"):
        load_spec(write(tmp_path, MIN + g + "refine: [{size: 1, distance: 2}]\n"))
    with pytest.raises(SpecError, match="default_group"):
        load_spec(write(tmp_path, MIN + g + "default_group: a\n"))
    s = load_spec(write(tmp_path, MIN + g + "refine: [{group: a, size: 1, distance: 2}]\n"))
    assert s.refine[0].group == "a"


def test_fea_bcs_parsed(tmp_path):
    body = MIN + ("groups:\n  f: {select: {kind: plane}}\n  l: {select: {kind: plane}}\n"
                  "bcs:\n  fea: {fixed: [f], loads: [{group: l, force: [0, 0, -1000]}]}\n")
    s = load_spec(write(tmp_path, body))
    assert s.bcs_fea.fixed == ["f"]
    assert s.bcs_fea.loads[0].force == (0.0, 0.0, -1000.0) and s.bcs_fea.E == 210000.0


def test_cfd_bcs_parsed(tmp_path):
    body = ("geometry: part.step\nanalysis: cfd\nsize: {global: 3}\n"
            "groups:\n  i: {select: {at: xmin}}\n  o: {select: {at: xmax}}\n"
            "bcs:\n  cfd: {inlet: [i], outlet: [o], velocity: 2}\n")
    s = load_spec(write(tmp_path, body))
    assert (s.bcs_cfd.inlet, s.bcs_cfd.outlet, s.bcs_cfd.velocity) == (["i"], ["o"], 2.0)
    assert s.bcs_fea is None


def test_old_solve_key_points_to_bcs(tmp_path):
    with pytest.raises(SpecError) as e:
        load_spec(write(tmp_path, MIN + "solve:\n  fea: {fixed: [], loads: []}\n"))
    assert e.value.message.startswith("solve")
    assert "bcs" in e.value.hint


def test_cfd_walls_collision_hint_explains_default(tmp_path):
    body = ("geometry: part.step\nanalysis: cfd\nsize: {global: 3}\n"
            "groups:\n  walls: {select: {kind: plane}}\n")
    with pytest.raises(SpecError) as e:
        load_spec(write(tmp_path, body))
    assert e.value.message.startswith("default_group")
    assert "cfd" in e.value.hint and "walls" in e.value.hint
    assert "default_group: null" in e.value.hint
    s = load_spec(write(tmp_path, body + "default_group: null\n"))
    assert s.default_group is None and "walls" in s.groups


def test_query_keys_come_from_select():
    from feagent import select, spec

    assert spec.QUERY_KEYS is select.QUERY_KEYS
    assert not hasattr(spec, "_FALLBACK_QUERY_KEYS")


def test_file_and_yaml_errors_have_hints(tmp_path):
    with pytest.raises(SpecError) as e:
        load_spec(tmp_path / "nope.yaml")
    assert "not found" in e.value.message and e.value.hint
    with pytest.raises(SpecError) as e:
        load_spec(write(tmp_path, "geometry: [unclosed\n"))
    assert "invalid YAML" in e.value.message and e.value.hint
