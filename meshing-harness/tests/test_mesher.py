"""Mesher: spec -> gmsh volume mesh with verified named boundary groups."""
import json
from pathlib import Path

import gmsh
import pytest
import yaml

from feagent.gmsh_session import gmsh_session
from feagent.mesher import auto_mesh, auto_spec, run_mesh, run_mesh_subprocess
from feagent.parts import PARTS
from feagent.spec import load_spec
from feagent.types import MeshResult

BRACKET_GROUPS = {
    "mount": {"select": {"feature": "hole", "radius": 3.25}, "expect": [4, 8]},
    "load": {"select": {"feature": "hole", "radius": 5}, "expect": [1, 2]},
}


def write_spec(d, geometry, **body):
    d.mkdir(parents=True, exist_ok=True)
    p = d / "mesh.yaml"
    p.write_text(yaml.safe_dump({"geometry": str(geometry), **body}))
    return p


def bracket_spec(d, parts_dir, **extra):
    body = {"analysis": "fea", "groups": BRACKET_GROUPS, "size": {"global": 5.0}, **extra}
    return write_spec(d, parts_dir / "bracket.step", **body)


def gate(res, gid):
    return next(g for g in res.gates if g.id == gid)


@pytest.fixture(scope="module")
def bracket(tmp_path_factory, parts_dir):
    p = bracket_spec(tmp_path_factory.mktemp("bracket"), parts_dir, export=["msh", "inp", "msh22"])
    return p, run_mesh(load_spec(p))


def test_bracket_fea_mesh(bracket):
    path, res = bracket
    assert res.ok, res.errors
    assert all(gate(res, g).status == "pass" for g in ("geometry", "groups", "mesh_generated"))
    assert res.mesh_path and (path.parent / "out" / "mesh.msh").is_file()
    assert res.n_elements > 1000 and res.n_nodes > 0
    assert 4 <= len(res.groups["mount"].tags) <= 8 and 1 <= len(res.groups["load"].tags) <= 2
    assert all(g.elements > 0 for g in res.groups.values())
    assert res.groups["domain"].dim == 3 and res.groups["domain"].elements == res.n_elements
    manifest = json.loads((path.parent / "out" / "manifest.json").read_text())
    assert MeshResult.from_dict(manifest).to_dict() == res.to_dict()
    assert (path.parent / "out" / "mesh.log").stat().st_size > 0


def test_bracket_exports(bracket):
    path, res = bracket
    out = path.parent / "out"
    assert set(res.exports) == {"msh", "inp", "msh22"}
    assert (out / "mesh_v22.msh").read_text().splitlines()[1].startswith("2.2")
    assert (out / "mesh.msh").read_text().splitlines()[1].startswith("4.1")
    inp = (out / "mesh.inp").read_text().upper()
    assert "NSET=MOUNT" in inp.replace(" ", "") and "ELSET=LOAD" in inp.replace(" ", "")


def test_refine_increases_elements(tmp_path, parts_dir, bracket):
    refine = [{"group": "mount", "size": 0.5, "distance": 3}]
    res = run_mesh(load_spec(bracket_spec(tmp_path, parts_dir, refine=refine)))
    assert res.ok, res.errors
    assert res.n_elements > bracket[1].n_elements


def test_thin_wall_layers_refine(tmp_path, parts_dir):
    def run(layers):
        body = {"analysis": "fea", "size": {"global": 6.0, "thin_wall_layers": layers}}
        return run_mesh(load_spec(write_spec(tmp_path / str(layers), parts_dir / "thin_cup.step",
                                             **body)))
    off, on = run(0), run(1)
    assert off.ok and on.ok, (off.errors, on.errors)
    assert on.n_elements > 2 * off.n_elements
    assert any("thin wall" in w for w in on.warnings)


def test_group_count_mismatch_fails_cleanly(tmp_path, parts_dir):
    groups = {"load": {"select": {"feature": "hole", "radius": 5}, "expect": 3}}
    p = write_spec(tmp_path, parts_dir / "bracket.step", analysis="fea", groups=groups,
                   size={"global": 5.0})
    res = run_mesh(load_spec(p))
    assert not res.ok and res.mesh_path is None
    assert gate(res, "groups").status == "fail"
    assert gate(res, "mesh_generated").status == "skip"
    assert "expected 3" in res.errors[0] and "load" in res.errors[0]
    assert not (tmp_path / "out" / "mesh.msh").exists()
    assert json.loads((tmp_path / "out" / "manifest.json").read_text())["ok"] is False


def test_all_group_mismatches_reported(tmp_path, parts_dir):
    groups = {"a": {"select": {"feature": "hole", "radius": 5}, "expect": 3},
              "b": {"select": {"feature": "hole", "radius": 9}}}
    p = write_spec(tmp_path, parts_dir / "bracket.step", analysis="fea", groups=groups,
                   size={"global": 5.0})
    err = run_mesh(load_spec(p)).errors[0]
    assert "group 'a'" in err and "group 'b'" in err and "at least 1" in err


def test_overlapping_groups_fail(tmp_path, parts_dir):
    groups = {"a": {"select": {"feature": "hole", "radius": 5}},
              "b": {"select": {"radius": 5}}}
    p = write_spec(tmp_path, parts_dir / "bracket.step", analysis="fea", groups=groups,
                   size={"global": 5.0})
    res = run_mesh(load_spec(p))
    assert not res.ok and gate(res, "groups").status == "fail"
    assert "are in groups a and b" in res.errors[0]


def test_failed_run_removes_stale_mesh(tmp_path, parts_dir):
    out = tmp_path / "out"
    out.mkdir()
    (out / "mesh.msh").write_text("stale")
    groups = {"load": {"select": {"feature": "hole", "radius": 5}, "expect": 3}}
    p = write_spec(tmp_path, parts_dir / "bracket.step", analysis="fea", groups=groups,
                   size={"global": 5.0})
    assert not run_mesh(load_spec(p)).ok
    assert not (out / "mesh.msh").exists()


def test_surface_only_step_fails_cleanly(tmp_path):
    step = tmp_path / "rect.step"
    with gmsh_session():
        gmsh.model.occ.addRectangle(0, 0, 0, 10, 5)
        gmsh.model.occ.synchronize()
        gmsh.write(str(step))
    res = run_mesh(load_spec(write_spec(tmp_path, step, analysis="fea", size={"global": 1.0})))
    assert not res.ok and res.mesh_path is None
    assert gate(res, "geometry").status == "fail"
    assert "no 3D volume" in res.errors[0]


def test_cfd_external_bullet(tmp_path, parts_dir):
    groups = {"inlet": {"select": {"at": "xmin"}, "expect": 1},
              "outlet": {"select": {"at": "xmax"}, "expect": 1}}
    p = write_spec(tmp_path, parts_dir / "bullet.step", analysis="cfd", groups=groups,
                   domain={"type": "external", "upstream": 1.0, "downstream": 2.0, "lateral": 1.0},
                   size={"global": 8.0})
    res = run_mesh(load_spec(p))
    assert res.ok, res.errors
    assert {"inlet", "outlet", "walls", "domain"} <= set(res.groups)
    assert len(res.groups["domain"].tags) == 1 and len(res.geometry.volumes) == 1
    assert res.geometry.bbox[0][0] == pytest.approx(-35.0, abs=1e-3)
    assert res.geometry.bbox[1][0] == pytest.approx(105.0, abs=1e-3)
    assert len(res.groups["walls"].tags) == 4 + 3  # 4 box sides + bullet's 3 faces


def test_scale_writes_subdir(tmp_path, parts_dir):
    p = bracket_spec(tmp_path, parts_dir)
    res = run_mesh(load_spec(p), 1.3)
    assert res.ok, res.errors
    assert res.size_scale == 1.3
    assert (tmp_path / "out" / "scale_1.3" / "mesh.msh").is_file()
    assert res.out_dir == str(tmp_path / "out" / "scale_1.3")


def test_thin_wall_refinement_scales_with_size_scale(tmp_path, parts_dir, bracket):
    def walls(res):
        return [w.split(": auto-refined")[0] for w in res.warnings if w.startswith("thin wall")]
    res = run_mesh(load_spec(bracket_spec(tmp_path, parts_dir)), 1.3)
    assert walls(res) and walls(res) == walls(bracket[1])  # same faces at every scale
    assert any("t=6 on faces" in w and "size 3.12 " in w for w in res.warnings)  # 0.8 * 6/2 * 1.3


def test_subprocess_matches(bracket):
    path, res = bracket
    sub = run_mesh_subprocess(path)
    assert sub.ok, sub.errors
    assert set(sub.groups) == set(res.groups)
    assert sub.n_elements == res.n_elements
    assert sub.geometry is not None and len(sub.geometry.faces) == len(res.geometry.faces)


def test_subprocess_timeout(bracket):
    res = run_mesh_subprocess(bracket[0], timeout=0.01)
    assert not res.ok
    assert res.errors == ["mesh timed out after 0.01s"]
    assert gate(res, "mesh_generated").status == "fail"


def test_cli_spec_error_is_json(tmp_path, capsys):
    from feagent.mesher import main
    p = tmp_path / "bad.yaml"
    p.write_text("analysis: fea\n")
    assert main([str(p), "--json"]) == 1
    out = capsys.readouterr().out
    assert len(out.splitlines()) == 1
    doc = json.loads(out)
    assert doc["ok"] is False and "geometry" in doc["errors"][0]


def test_cli_crash_is_one_json_doc(bracket, capsys, monkeypatch):
    import feagent.mesher as mesher

    def boom(*a, **k):
        raise RuntimeError("kaboom")
    monkeypatch.setattr(mesher, "run_mesh", boom)
    assert mesher.main([str(bracket[0]), "--json"]) == 1
    out = capsys.readouterr().out
    assert len(out.splitlines()) == 1
    doc = json.loads(out)
    assert doc["ok"] is False and "kaboom" in doc["errors"][0]
    assert doc["gates"][0]["status"] == "fail"


def test_export_failure_leaves_no_mesh_files(tmp_path, parts_dir, monkeypatch):
    real = gmsh.write

    def write(path):
        if path.endswith(".vtk"):
            raise Exception("disk full")
        real(path)
    monkeypatch.setattr(gmsh, "write", write)
    res = run_mesh(load_spec(bracket_spec(tmp_path, parts_dir, export=["msh", "inp", "vtk"])))
    assert not res.ok and res.mesh_path is None and res.exports == {}
    assert gate(res, "mesh_generated").status == "fail" and "disk full" in res.errors[0]
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == ["manifest.json", "mesh.log"]


@pytest.mark.parametrize("where", ["group", "default_group"])
def test_domain_group_name_is_reserved(tmp_path, parts_dir, where):
    body = {"analysis": "fea", "size": {"global": 5.0}}
    if where == "group":
        body["groups"] = {"domain": {"select": {"feature": "hole", "radius": 5}}}
    else:
        body["default_group"] = "domain"
    res = run_mesh(load_spec(write_spec(tmp_path, parts_dir / "bracket.step", **body)))
    assert not res.ok and res.mesh_path is None
    g = gate(res, "groups")
    assert g.status == "fail" and "reserved" in g.message and "domain" in g.hint + g.message


def test_auto_spec_cfd_passes_volume_gate(tmp_path, parts_dir):
    from feagent.quality import assess_mesh
    from feagent.spec import QualitySpec
    res = auto_mesh(parts_dir / "tee.step", "cfd", out_dir=tmp_path)
    assert res.ok and res.order == 1, res.errors
    q = assess_mesh(res.mesh_path, res, QualitySpec(), "cfd")
    vf = next(g for g in q.gates if g.id == "volume_fidelity")
    assert vf.status == "pass", vf.message


def test_auto_spec_rejects_unknown_analysis(parts_dir):
    with pytest.raises(ValueError, match="fea' or 'cfd"):
        auto_spec(parts_dir / "bracket.step", "thermal")


AUTO_EXPECTED_FAILS = {"sliver": {"sicn"}}  # a 0.05 mm sliver face forces flat tets at zero config


def _failing_gates(res, analysis):
    from feagent.quality import assess_mesh
    from feagent.spec import QualitySpec
    return {g.id for g in assess_mesh(res.mesh_path, res, QualitySpec(), analysis).gates
            if g.status == "fail"}


@pytest.mark.parametrize("name", PARTS)
def test_auto_mesh_every_part(tmp_path, parts_dir, name):
    analysis = "cfd" if name == "tee" else "fea"
    res = auto_mesh(parts_dir / f"{name}.step", analysis, out_dir=tmp_path / name)
    assert res.ok, res.errors
    assert res.n_elements > 0 and res.order == (1 if analysis == "cfd" else 2)
    assert (tmp_path / name / "mesh.yaml").is_file()
    assert _failing_gates(res, analysis) == AUTO_EXPECTED_FAILS.get(name, set())


def test_tee_fea_automesh_passes(tmp_path, parts_dir):
    """Order-2 tets are curved: their volume is integrated (det J), not taken from the corners."""
    from feagent.quality import assess_mesh
    from feagent.spec import QualitySpec
    res = auto_mesh(parts_dir / "tee.step", "fea", out_dir=tmp_path)
    assert res.ok and res.order == 2, res.errors
    q = assess_mesh(res.mesh_path, res, QualitySpec(), "fea")
    assert all(g.status != "fail" for g in q.gates), [(g.id, g.message) for g in q.gates]
    assert q.volume_error < 1e-3


def test_auto_spec_roundtrips_through_load_spec(tmp_path, parts_dir):
    p = tmp_path / "specs" / "bracket.yaml"
    d = auto_spec(parts_dir / "bracket.step", "fea", spec_path=p)
    text = p.read_text()
    assert text.startswith("#") and "auto-generated" in text.lower()
    assert not Path(yaml.safe_load(text)["geometry"]).is_absolute()
    s = load_spec(p)
    assert s.geometry == (parts_dir / "bracket.step").resolve()
    assert s.size.global_ == d["size"]["global"] == 2.6  # diag 102.5 / 40 = 2.56 -> 2 s.f.
    assert s.size.min == d["size"]["min"] == 0.13  # 2.6 / 20, 2 s.f.
    assert (s.order, s.optimize, s.export, s.groups) == (2, "netgen", ["msh", "inp"], {})


def test_auto_spec_cfd_defaults(parts_dir):
    d = auto_spec(parts_dir / "tee.step", "cfd")
    assert Path(d["geometry"]).is_absolute()
    assert d["default_group"] == "walls" and d["domain"] == {"type": "internal"}
    assert d["element"]["order"] == 1 and d["export"] == ["msh", "msh22", "su2"]
    assert d["size"]["curvature"] == 24


def test_group_warnings_for_ids_and_unchecked_multi_matches():
    from feagent.mesher import _group_warnings
    from feagent.spec import GroupSpec
    spec = type("S", (), {"groups": {
        "tagged": GroupSpec("tagged", {"not": {"ids": [3]}}, expect=2),
        "holes": GroupSpec("holes", {"kind": "cylinder"}),
        "one": GroupSpec("one", {"at": "xmin"})}})()
    w = _group_warnings(spec, {"tagged": [1, 2], "holes": [4, 5, 6], "one": [7]})
    assert len(w) == 2
    assert "'tagged'" in w[0] and "ids" in w[0]
    assert "'holes'" in w[1] and "groups.holes.expect: 3" in w[1]


def test_element_budget_relaxes_thin_walls(tmp_path, parts_dir):
    def run(name, **size):
        body = {"analysis": "fea", "element": {"order": 2},
                "size": {"global": 6.0, "min": 0.3, "thin_wall_layers": 2, **size}}
        return run_mesh(load_spec(write_spec(tmp_path / name, parts_dir / "thin_cup.step", **body)))
    free, capped = run("free"), run("capped", max_dofs=150_000)
    assert free.ok and capped.ok, (free.errors, capped.errors)
    note = next(w for w in capped.warnings if "element budget" in w)
    assert "thin_wall_layers 2->1" in note and not any("budget" in w for w in free.warnings)
    assert capped.n_elements < free.n_elements
    assert capped.n_nodes * 3 < 1.3 * 150_000  # the estimate held within 30 %


def test_budget_estimate_tracks_actual_tets(tmp_path, parts_dir):
    from feagent.budget import estimate_tets
    from feagent.geometry import analyze_current_model
    from feagent.gmsh_session import gmsh_session, load_cad
    spec = load_spec(bracket_spec(tmp_path, parts_dir, size={"global": 3.0}))
    res = run_mesh(spec)
    with gmsh_session():
        load_cad(spec.geometry)
        est = estimate_tets(spec, analyze_current_model(ref_size=spec.size.global_), spec.size)
    assert 0.4 < est / res.n_elements < 2.5, (est, res.n_elements)

