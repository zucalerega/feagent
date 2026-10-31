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
    assert s.default_group is None and s.fea is None and s.cfd is None
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
    body = MIN + "groups:\n  a: {select: {kind: plane}}\nsolve:\n  fea: {fixed: [nope], loads: []}\n"
    with pytest.raises(SpecError, match="nope"):
        load_spec(write(tmp_path, body))


def test_expect_range(tmp_path):
    body = MIN + "groups:\n  a: {select: {kind: plane}, expect: [2, 4]}\n  b: {select: {kind: plane}, expect: 1}\n"
    s = load_spec(write(tmp_path, body))
    assert s.groups["a"].expect == (2, 4)
    assert s.groups["b"].expect == 1


def test_geometry_must_exist(tmp_path):
    with pytest.raises(SpecError) as e:
        load_spec(write(tmp_path, MIN, geometry=False))
    assert e.value.message.startswith("geometry")


def test_bad_select_key_and_solve_mismatch(tmp_path):
    with pytest.raises(SpecError, match=r"groups\.a\.select\.bogus"):
        load_spec(write(tmp_path, MIN + "groups:\n  a: {select: {bogus: 1}}\n"))
    with pytest.raises(SpecError, match=r"solve\.cfd"):
        load_spec(write(tmp_path, MIN + "solve:\n  cfd: {inlet: [], outlet: []}\n"))


def test_refine_and_default_group_validation(tmp_path):
    g = "groups:\n  a: {select: {kind: plane}}\n"
    with pytest.raises(SpecError, match=r"refine\[0\]"):
        load_spec(write(tmp_path, MIN + g + "refine: [{size: 1, distance: 2}]\n"))
    with pytest.raises(SpecError, match="default_group"):
        load_spec(write(tmp_path, MIN + g + "default_group: a\n"))
    s = load_spec(write(tmp_path, MIN + g + "refine: [{group: a, size: 1, distance: 2}]\n"))
    assert s.refine[0].group == "a"


def test_fea_solve_parsed(tmp_path):
    body = MIN + ("groups:\n  f: {select: {kind: plane}}\n  l: {select: {kind: plane}}\n"
                  "solve:\n  fea: {fixed: [f], loads: [{group: l, force: [0, 0, -1000]}]}\n")
    s = load_spec(write(tmp_path, body))
    assert s.fea.loads[0].force == (0.0, 0.0, -1000.0) and s.fea.E == 210000.0
