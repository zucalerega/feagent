"""Check orchestration (mesh -> quality gates -> renders -> report) and the `feagent` CLI."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from feagent.check import CheckReport, record_state, report_markdown, run_check, state_path

FEAGENT = str(Path(sys.executable).parent / "feagent")

CANTILEVER = {
    "analysis": "fea",
    "groups": {"fixed": {"select": {"at": "xmin"}, "expect": 1},
               "load": {"select": {"at": "xmax"}, "expect": 1}},
    "size": {"global": 4.0},
    "element": {"order": 2},
    "bcs": {"fea": {"fixed": ["fixed"], "loads": [{"group": "load", "force": [0, 0, -100]}]}},
}


def write_spec(d: Path, geometry: Path, body: dict) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    p = d / "mesh.yaml"
    p.write_text(yaml.safe_dump({"geometry": str(geometry), **body}))
    return p


def cli(*args: str, env: dict | None = None) -> subprocess.CompletedProcess:
    return subprocess.run([FEAGENT, *args], capture_output=True, text=True, timeout=600, env=env)


@pytest.fixture(scope="module")
def cantilever(tmp_path_factory, parts_dir):
    d = tmp_path_factory.mktemp("cantilever")
    spec = write_spec(d / "proj", parts_dir / "cantilever.step", CANTILEVER)
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("FEAGENT_STATE", str(d / "state.json"))
        report = run_check(spec)
        state = json.loads(state_path().read_text())
    return spec, report, state


def test_check_cantilever_passes(cantilever):
    spec, r, _ = cantilever
    assert r.status == "pass", report_markdown(r)
    md = (spec.parent / "out" / "REPORT.md").read_text()
    first_table = md.split("\n\n| status")[1].split("\n\n")[0]
    assert "volume_fidelity" in first_table and "mesh_generated" in first_table
    assert "## Boundary conditions (for the solver)" in md
    assert {g.id for g in r.gates} >= {"geometry", "groups", "mesh_generated", "sicn", "inverted"}


def test_check_report_json(cantilever):
    spec, r, _ = cantilever
    d = json.loads((spec.parent / "out" / "report.json").read_text())
    assert d["schema_version"] == 1 and d["status"] == "pass"
    assert d["bcs"] == {"fea": {"fixed": ["fixed"], "loads": [{"group": "load",
                                                               "force": [0.0, 0.0, -100.0]}],
                                "E": 210000.0, "nu": 0.3}}
    assert [g["id"] for g in d["gates"]] == [g.id for g in r.gates]
    assert d["quality"]["n_tets"] > 0 and d["mesh"]["n_elements"] == d["quality"]["n_tets"]
    assert {Path(x["path"]).name for x in d["renders"]} == {"groups.png", "quality.png"}
    assert all(Path(x["path"]).is_file() for x in d["renders"])
    assert CheckReport.from_dict(d).to_dict() == d


def test_check_records_state(cantilever):
    spec, _, state = cantilever
    entry = state["specs"][str(spec.resolve())]
    assert entry["status"] == "pass" and entry["waiver"] is None
    assert entry["checked_at"] >= entry["spec_mtime"]


def test_record_state_keeps_other_keys(tmp_path, monkeypatch):
    monkeypatch.setenv("FEAGENT_STATE", str(tmp_path / "s" / "state.json"))
    spec = tmp_path / "mesh.yaml"
    spec.write_text("x: 1\n")
    record_state(str(spec), "fail")
    data = json.loads(state_path().read_text())
    data["hook_blocks"] = 2
    state_path().write_text(json.dumps(data))
    record_state(str(spec), "fail", waiver="known issue")
    data = json.loads(state_path().read_text())
    assert data["hook_blocks"] == 2
    assert data["specs"][str(spec.resolve())]["waiver"] == "known issue"


def test_check_group_miss_fails_with_explanation(tmp_path, parts_dir, monkeypatch):
    monkeypatch.setenv("FEAGENT_STATE", str(tmp_path / "state.json"))
    body = {**CANTILEVER, "groups": {**CANTILEVER["groups"],
                                     "fixed": {"select": {"at": "xmin"}, "expect": 3}}}
    r = run_check(write_spec(tmp_path / "p", parts_dir / "cantilever.step", body), render=False)
    assert r.status == "fail" and r.quality is None and r.renders == []
    md = report_markdown(r)
    assert "| fail | groups |" in md and "expected 3" in md


def test_cli_inspect_json(parts_dir):
    p = cli("inspect", str(parts_dir / "plate_hole.step"), "--json")
    assert p.returncode == 0, p.stderr
    d = json.loads(p.stdout)
    assert [f["kind"] for f in d["features"]].count("hole") == 1


def test_cli_inspect_text_digest(parts_dir):
    p = cli("inspect", str(parts_dir / "plate_hole.step"))
    assert p.returncode == 0, p.stderr
    assert "hole" in p.stdout


def test_cli_faces_expect_miss_exit2(parts_dir):
    p = cli("faces", str(parts_dir / "cantilever.step"), "--query",
            '{"at":"xmin","kind":"cylinder"}', "--expect", "1")
    assert p.returncode == 2
    assert "0 matched" in p.stderr


def test_cli_faces_json(parts_dir):
    p = cli("faces", str(parts_dir / "cantilever.step"), "--query", '{"at":"xmin"}',
            "--expect", "1", "--json")
    assert p.returncode == 0, p.stderr
    d = json.loads(p.stdout)
    assert len(d["tags"]) == 1 and d["faces"][0]["kind"] == "plane"


def test_cli_bad_query_json_exit2(parts_dir):
    p = cli("faces", str(parts_dir / "cantilever.step"), "--query", "{at: xmin", "--json")
    assert p.returncode == 2
    assert json.loads(p.stdout)["ok"] is False and "hint" in p.stderr


def test_cli_spec_error_exit2(tmp_path, parts_dir):
    spec = write_spec(tmp_path, parts_dir / "cantilever.step",
                      {**CANTILEVER, "solve": CANTILEVER["bcs"]})
    p = cli("check", str(spec))
    assert p.returncode == 2
    assert "solve" in p.stderr and "bcs" in p.stderr


def test_cli_render_faces(tmp_path, parts_dir):
    out = tmp_path / "f.png"
    p = cli("render", str(parts_dir / "bracket.step"), "--view", "faces",
            "--query", '{"feature":"hole","radius":5}', "-o", str(out), "--json")
    assert p.returncode == 0, p.stderr
    assert out.is_file() and json.loads(p.stdout)["path"] == str(out)


def test_cli_mesh_then_waive(tmp_path, parts_dir):
    env = {**os.environ, "FEAGENT_STATE": str(tmp_path / "state.json")}
    spec = write_spec(tmp_path / "p", parts_dir / "cantilever.step", CANTILEVER)
    p = cli("mesh", str(spec), "--json", env=env)
    assert p.returncode == 0, p.stderr
    assert json.loads(p.stdout)["ok"] is True
    key = str(spec.resolve())
    assert json.loads(Path(env["FEAGENT_STATE"]).read_text())["specs"][key]["status"] == "unchecked"
    p = cli("waive", str(spec), "--reason", "smoke test", env=env)
    assert p.returncode == 0, p.stderr
    entry = json.loads(Path(env["FEAGENT_STATE"]).read_text())["specs"][key]
    assert (entry["status"], entry["waiver"]) == ("unchecked", "smoke test")


def test_cli_automesh_bracket(tmp_path, parts_dir):
    env = {**os.environ, "FEAGENT_STATE": str(tmp_path / "state.json")}
    out = tmp_path / "auto"
    p = cli("automesh", str(parts_dir / "bracket.step"), "-o", str(out), env=env)
    assert p.returncode == 0, p.stdout + p.stderr
    assert str(out / "mesh.yaml") in p.stdout and "| pass | volume_fidelity |" in p.stdout
    d = json.loads((out / "out" / "report.json").read_text())
    assert d["status"] == "pass" and d["schema_version"] == 1
