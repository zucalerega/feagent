"""Agent integration: Stop hook, MCP server, agent_context (skill, tool docs, examples)."""
import asyncio
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from feagent import hook, mcp_server
from feagent.check import load_state, record_state, run_check, waive, write_state

ROOT = Path(__file__).resolve().parents[1]
CONTEXT = ROOT / "agent_context"
EXAMPLES = ("cantilever", "bracket", "plate_hole", "thin_cup", "tee", "bullet")
TOOLS = {"inspect_geometry", "find_faces", "render_view", "automesh", "mesh", "check_mesh"}


@pytest.fixture(autouse=True)
def state_file(tmp_path, monkeypatch):
    path = tmp_path / "state" / "state.json"
    monkeypatch.setenv("FEAGENT_STATE", str(path))
    monkeypatch.delenv(mcp_server.CONTEXT_ENV, raising=False)
    return path


@pytest.fixture
def spec(tmp_path):
    p = tmp_path / "case" / "mesh.yaml"
    p.parent.mkdir()
    p.write_text("# placeholder spec for state tests\n")
    return p


def _bump_mtime(p: Path) -> None:
    st = p.stat()
    os.utime(p, (st.st_atime, st.st_mtime + 10))


# --- Stop hook ----------------------------------------------------------------

def test_hook_allows_without_state(state_file):
    assert hook.main("{}") == 0
    assert not state_file.exists()


def test_hook_blocks_on_fail(spec, capsys):
    record_state(str(spec), "fail")
    assert hook.main("{}") == 2
    err = capsys.readouterr().err
    assert str(spec.resolve()) in err and "fail" in err and "waive" in err
    record_state(str(spec), "fail", waiver="x")
    assert hook.main("{}") == 0


def test_hook_allows_after_three_consecutive_blocks(spec, capsys):
    record_state(str(spec), "unchecked")
    assert [hook.main("{}") for _ in range(4)] == [2, 2, 2, 0]
    assert "3" in capsys.readouterr().err.splitlines()[-1]


def test_hook_escape_persists_for_unchanged_set(spec, capsys):
    record_state(str(spec), "fail")
    assert [hook.main("{}") for _ in range(5)] == [2, 2, 2, 0, 0]
    assert "already reported" in capsys.readouterr().err.splitlines()[-1]
    record_state(str(spec), "fail")  # re-checked, same mtime and status: still the same set
    assert hook.main("{}") == 0


def test_hook_rearms_on_a_new_failing_spec_or_an_edit(spec, tmp_path):
    record_state(str(spec), "fail")
    assert [hook.main("{}") for _ in range(4)] == [2, 2, 2, 0]
    other = tmp_path / "other.yaml"
    other.write_text("# other\n")
    record_state(str(other), "fail")
    assert [hook.main("{}") for _ in range(4)] == [2, 2, 2, 0]
    _bump_mtime(spec)  # the escaped set changed: blocks again
    assert hook.main("{}") == 2


def _transcript(tmp_path: Path, started: float) -> str:
    from datetime import datetime, timezone
    p = tmp_path / "session.jsonl"
    stamp = datetime.fromtimestamp(started, timezone.utc).isoformat().replace("+00:00", "Z")
    p.write_text(json.dumps({"type": "mode"}) + "\n" + json.dumps({"timestamp": stamp}) + "\n")
    return json.dumps({"transcript_path": str(p), "stop_hook_active": False})


def test_hook_ignores_entries_of_older_sessions(spec, tmp_path):
    import time
    record_state(str(spec), "fail")
    data = load_state()
    data["specs"][str(spec.resolve())]["checked_at"] = time.time() - 3600
    write_state(data)
    payload = _transcript(tmp_path, time.time() - 60)
    assert hook.main(payload) == 0  # recorded before this session started
    assert hook.main("{}") == 2  # no transcript: every entry counts
    record_state(str(spec), "fail")  # recorded in this session
    assert hook.main(payload) == 2


def test_hook_fresh_stop_attempt_restarts_the_count(spec):
    record_state(str(spec), "fail")
    active = json.dumps({"stop_hook_active": True})
    assert [hook.main(active) for _ in range(2)] == [2, 2]
    assert hook.main(json.dumps({"stop_hook_active": False})) == 2
    assert load_state()["hook_blocks"]["count"] == 1
    assert hook.main('{"stop_hook_active": "odd", "transcript_path": 5}') == 2  # never crashes


@pytest.mark.parametrize("state", [{"specs": []}, {"specs": {"/x/mesh.yaml": "fail"}}])
def test_hook_allows_on_malformed_state(state_file, state, capsys):
    state_file.parent.mkdir(parents=True, exist_ok=True)
    state_file.write_text(json.dumps(state))
    assert hook.main("{}") == 0
    assert "malformed" in capsys.readouterr().err


def test_hook_counter_restarts_when_blocking_set_changes(spec, tmp_path):
    other = tmp_path / "other.yaml"
    other.write_text("# other\n")
    record_state(str(spec), "fail")
    assert [hook.main("{}") for _ in range(3)] == [2, 2, 2]
    record_state(str(other), "fail")
    assert hook.main("{}") == 2
    assert load_state()["hook_blocks"]["count"] == 1


def test_hook_allows_passing_and_blocks_stale_pass(spec, capsys):
    record_state(str(spec), "pass")
    assert hook.main("{}") == 0
    _bump_mtime(spec)
    assert hook.main("{}") == 2
    assert "edited since" in capsys.readouterr().err


def test_hook_waiver_expires_when_spec_is_edited(spec):
    record_state(str(spec), "fail")
    waive(str(spec), "accepted: coarse preview mesh")
    assert hook.main("{}") == 0
    _bump_mtime(spec)
    assert hook.main("{}") == 2


def test_hook_ignores_deleted_specs(spec):
    record_state(str(spec), "fail")
    spec.unlink()
    assert hook.main("{}") == 0


def test_hook_lists_failing_gate_ids(parts_dir, tmp_path, capsys):
    p = tmp_path / "beam" / "mesh.yaml"
    p.parent.mkdir()
    p.write_text(yaml.safe_dump({"geometry": str(parts_dir / "cantilever.step"),
                                 "analysis": "fea", "size": {"global": 5.0}}))
    gates = [{"id": "groups", "status": "pass"}, {"id": "sicn", "status": "fail"},
             {"id": "volume_fidelity", "status": "fail"}, {"id": "gamma", "status": "warn"}]
    (p.parent / "out").mkdir()
    (p.parent / "out" / "report.json").write_text(json.dumps({"created_at": "T1", "gates": gates}))
    record_state(str(p), "fail", report_created_at="T1")
    assert hook.main("{}") == 2
    line = next(x for x in capsys.readouterr().err.splitlines() if str(p.resolve()) in x)
    assert "sicn" in line and "volume_fidelity" in line
    assert "groups" not in line and "gamma" not in line


def test_cli_hook_stop_delegates_to_hook(spec, state_file):
    record_state(str(spec), "fail")
    env = {**os.environ, "FEAGENT_STATE": str(state_file)}
    p = subprocess.run([sys.executable, "-m", "feagent.cli", "hook-stop"], input="{}",
                       capture_output=True, text=True, env=env, timeout=60)
    assert p.returncode == 2 and str(spec.resolve()) in p.stderr and p.stdout == ""


# --- examples -----------------------------------------------------------------

@pytest.fixture(scope="module")
def examples_copy(tmp_path_factory):
    """A copy of agent_context/examples, so checks do not write out/ into the repo."""
    d = tmp_path_factory.mktemp("examples") / "examples"
    shutil.copytree(CONTEXT / "examples", d, ignore=shutil.ignore_patterns("out"))
    return d


def test_examples_are_the_documented_cases():
    found = {p.parent.name for p in (CONTEXT / "examples").glob("*/mesh.yaml")}
    assert found == set(EXAMPLES)


@pytest.mark.parametrize("case", EXAMPLES)
def test_example_declares_bcs_and_expected_counts(case):
    data = yaml.safe_load((CONTEXT / "examples" / case / "mesh.yaml").read_text())
    assert "bcs" in data and "solve" not in data
    assert data["groups"] and all("expect" in g for g in data["groups"].values())
    assert (CONTEXT / "examples" / case / data["geometry"]).resolve().is_file()


@pytest.mark.parametrize("case", EXAMPLES)
def test_example_passes_check(examples_copy, case):
    r = run_check(examples_copy / case / "mesh.yaml", render=False)
    failing = [(g.id, g.message) for g in r.gates if g.status == "fail"]
    assert r.status == "pass", failing
    assert r.bcs


# --- skill and agent_context ------------------------------------------------------

def test_skill_copy_is_identical():
    assert (ROOT / ".claude/skills/fe-mesh/SKILL.md").read_text() == (CONTEXT / "SKILL.md").read_text()


def test_skill_frontmatter_and_required_sections():
    text = (CONTEXT / "SKILL.md").read_text()
    front = yaml.safe_load(text.split("---")[1])
    assert front["name"] == "fe-mesh" and len(front["description"]) > 40
    for needle in ("automesh", "--expect", "waive", "out of scope", "| gate", "threshold"):
        assert needle in text, needle


def test_claude_code_config_files():
    mcp_cfg = json.loads((ROOT / ".mcp.json").read_text())["mcpServers"]["feagent"]
    assert [mcp_cfg["command"], *mcp_cfg["args"]] == ["uv", "run", "feagent-mcp"]
    stop = json.loads((ROOT / ".claude/settings.json").read_text())["hooks"]["Stop"]
    assert stop[0]["hooks"][0]["command"] == "uv run feagent hook-stop"


# --- MCP server -----------------------------------------------------------------

def _tools(server) -> dict[str, str]:
    return {t.name: t.description for t in asyncio.run(server.list_tools())}


def test_mcp_lists_tools():
    assert set(_tools(mcp_server.mcp)) == TOOLS


def test_tool_descriptions_come_from_tools_yaml():
    docs = yaml.safe_load((CONTEXT / "tools.yaml").read_text())
    assert _tools(mcp_server.mcp) == {k: v.strip() for k, v in docs.items()}


def test_context_dir_override(tmp_path, monkeypatch):
    ctx = tmp_path / "ctx"
    shutil.copytree(CONTEXT, ctx, ignore=shutil.ignore_patterns("examples"))
    docs = yaml.safe_load((ctx / "tools.yaml").read_text())
    docs["mesh"] = "Improver-edited description."
    (ctx / "tools.yaml").write_text(yaml.safe_dump(docs))
    monkeypatch.setenv(mcp_server.CONTEXT_ENV, str(ctx))
    assert mcp_server.context_dir() == ctx.resolve()
    assert _tools(mcp_server.build_server())["mesh"] == "Improver-edited description."


def test_tools_yaml_must_match_registered_tools(tmp_path):
    ctx = tmp_path / "ctx"
    shutil.copytree(CONTEXT, ctx, ignore=shutil.ignore_patterns("examples"))
    docs = yaml.safe_load((ctx / "tools.yaml").read_text())
    del docs["mesh"]
    docs["convergence_study"] = "gone"
    (ctx / "tools.yaml").write_text(yaml.safe_dump(docs))
    with pytest.raises(ValueError, match="mesh.*convergence_study|convergence_study.*mesh"):
        mcp_server.build_server(ctx)


def test_meshing_workflow_prompt_is_the_skill():
    res = asyncio.run(mcp_server.mcp.get_prompt("meshing_workflow"))
    text = res.messages[0].content.text
    assert text == (CONTEXT / "SKILL.md").read_text()


def test_find_faces_tool(parts_dir):
    text = mcp_server.find_faces(str(parts_dir / "cantilever.step"), {"at": "xmin"}, expect=1)
    assert text.startswith("1 face(s) matched") and "kind=plane" in text


def test_tool_errors_carry_the_hint(parts_dir):
    with pytest.raises(mcp_server.ToolError) as e:
        mcp_server.find_faces(str(parts_dir / "cantilever.step"), {"at": "left"})
    assert "query key 'at'" in str(e.value) and "hint:" in str(e.value)


def test_cli_timeout_is_reported(parts_dir):
    with pytest.raises(mcp_server.ToolError, match="timed out"):
        mcp_server.run_cli(["inspect", str(parts_dir / "bracket.step")], timeout=0.01)


def test_inspect_geometry_tool_returns_digest(parts_dir):
    text = mcp_server.inspect_geometry(str(parts_dir / "bracket.step"))
    assert text.startswith("GEOMETRY") and "FEATURES" in text and "HAZARDS" in text


def test_render_view_tool_returns_caption_and_image(parts_dir, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    caption, image = mcp_server.render_view(str(parts_dir / "plate_hole.step"),
                                            query={"kind": "cylinder"})
    assert isinstance(image, mcp_server.Image) and Path(image.path).is_file()
    assert str(image.path) in caption
    assert Path(image.path) == tmp_path / "renders" / "plate_hole_faces.png"  # not next to the part
    _, own = mcp_server.render_view(str(parts_dir / "plate_hole.step"), output="mine/f.png")
    assert Path(own.path) == tmp_path / "mine" / "f.png" and Path(own.path).is_file()


def test_automesh_tool_writes_into_the_server_cwd(parts_dir, tmp_path, monkeypatch):
    """Not next to the CAD: agents sharing a part must not collide; the part dir may be read-only."""
    part_dir = tmp_path / "ro"
    part_dir.mkdir()
    shutil.copy(parts_dir / "cantilever.step", part_dir / "cantilever.step")
    part_dir.chmod(0o555)
    work = tmp_path / "agent"
    work.mkdir()
    monkeypatch.chdir(work)
    try:
        text = mcp_server.automesh(str(part_dir / "cantilever.step"))
    finally:
        part_dir.chmod(0o755)
    assert str(work / "cantilever_mesh" / "mesh.yaml") in text and "status: pass" in text
    assert sorted(p.name for p in part_dir.iterdir()) == ["cantilever.step"]


def test_automesh_then_check_mesh_tools(parts_dir, tmp_path, state_file):
    step = tmp_path / "cantilever.step"
    shutil.copy(parts_dir / "cantilever.step", step)
    text = mcp_server.automesh(str(step), output=str(tmp_path / "cantilever_mesh"))
    spec = tmp_path / "cantilever_mesh" / "mesh.yaml"
    assert str(spec) in text and "status: pass" in text and "| pass | sicn |" in text
    mesh_text = mcp_server.mesh(str(spec), scale=2.0)
    assert mesh_text.startswith("mesh ok") and "| pass | mesh_generated |" in mesh_text
    report = mcp_server.check_mesh(str(spec))
    assert report.startswith("# Mesh check: cantilever_mesh — PASS")
    assert load_state()["specs"][str(spec)]["status"] == "pass"
