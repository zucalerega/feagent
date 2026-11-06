"""The dashboard's Agents section: the ACE tools (feacore/ui/tools/agents.py) and the ACE folders API
(feacore/ui/ace_runs.py). No API key, no network: argv checks, a hand-made run folder in a temporary
run store, path refusals, the token, and one real mock run (train, export, run the export) through the
task lane."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from feacore.runstore import RunStore  # noqa: E402
from feacore.ui import ace_runs  # noqa: E402
from feacore.ui import tasks as T  # noqa: E402
from feacore.ui.server import create_app  # noqa: E402

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64   # enough for a FileResponse; nothing decodes it
AGENT_TOOLS = {"ace.swarm", "ace.single", "ace.compare", "ace.report", "ace.export_swarm", "ace.infer"}


@pytest.fixture
def roots(monkeypatch):
    """The ACE roots inside the (temporary) run store, so nothing in the repository is read."""
    monkeypatch.setattr(ace_runs, "RUN_ROOTS", ["@runs/ace_runs"])
    monkeypatch.setattr(ace_runs, "SWARM_ROOTS", ["@runs/swarms"])


def argv_env(tool: str, args: dict, store: RunStore) -> tuple[list[str], dict]:
    t = T.get_tool(tool)
    clean = T.validate_args(t, args, store)
    if t.out_param and not clean.get(t.out_param):
        clean[t.out_param] = "/abs/out"
    return T.build_argv(t, clean), (t.build_env(clean) if t.build_env else {})


# --------------------------------------------------------------------------- tools


def test_the_agents_tools_are_registered():
    T.discover_tools()
    assert {n for n, t in T.TOOLS.items() if t.group == "agents"} == AGENT_TOOLS
    assert T.TOOLS["ace.swarm"].stop_file == "STOP" and T.TOOLS["ace.single"].stop_file == "STOP"
    assert T.TOOLS["ace.infer"].requires_env == ["ANTHROPIC_API_KEY"] and T.TOOLS["ace.infer"].unless_param == "mock"
    assert {t.out_root for n, t in T.TOOLS.items() if n in ("ace.swarm", "ace.single", "ace.infer", "ace.compare")} == {"ace_runs"}


def test_the_readme_swarm_recipes_build_the_same_command_lines(tmp_path):
    store = RunStore(tmp_path)
    # Result 2: ACE_SCORE=field run_loop --deep --split --problems abc_00000065_bc0 --holdout "" ...
    argv, env = argv_env("ace.swarm", {"mock": False, "deep": True, "problems": "abc_00000065_bc0", "holdout": "none",
                                       "agents": 8, "concurrency": 8, "persona_count": 4, "generations": 8, "budget_usd": 45},
                         store)
    assert argv[:2] == ["-m", "ace.run_loop"] and env == {"ACE_SCORE": "field"}
    s = " ".join(argv)
    for part in ("--problems abc_00000065_bc0", "--agents 8", "--concurrency 8", "--persona-count 4",
                 "--generations 8", "--budget-usd 45", "--deep", "--split", "--out /abs/out"):
        assert part in s, part
    assert argv[argv.index("--holdout") + 1] == ""          # 'none' in the form = no held-out problems
    assert "--memory" not in argv and "--mock" not in argv  # the loop's defaults stay off the command line
    # Result 3: post-training from a committed trained swarm
    argv, env = argv_env("ace.swarm", {"deep": True, "problems": "abc_00000065_compression", "holdout": "none",
                                       "init_from": "results/trained_swarms/div_s8", "init_agents": "a6,a3,a1,a7",
                                       "pre_teach": True, "agents": 4}, store)
    s = " ".join(argv)
    assert "--init-from results/trained_swarms/div_s8 --init-agents a6,a3,a1,a7 --pre-teach" in s and env["ACE_SCORE"] == "field"
    # the basic loop: point scoring, the suite's problems, the harness switch
    argv, env = argv_env("ace.swarm", {"mock": True, "harness": True, "peers": "off"}, store)
    assert "--problems" not in argv and "--holdout" not in argv and "--no-peers" in argv and "--mock" in argv
    assert env == {"ACE_HARNESS": "1"}


def test_single_compare_export_report_and_infer_command_lines(tmp_path):
    store = RunStore(tmp_path)
    argv, env = argv_env("ace.single", {"mock": True, "problems": "cooks_membrane", "generations": 2,
                                        "swarm_dir": "ace_runs/deep2"}, store)
    assert argv[:2] == ["-m", "ace.single_loop"] and env == {"ACE_SCORE": "field"}   # always field-scored
    assert "--swarm-dir ace_runs/deep2" in " ".join(argv) and argv[-2:] == ["--out", "/abs/out"]
    argv, _ = argv_env("ace.compare", {"swarm_dir": "ace_runs/a", "single_dir": "ace_runs/b"}, store)
    assert argv == ["-m", "ace.compare", "--out", "/abs/out", "ace_runs/a", "ace_runs/b"]
    argv, _ = argv_env("ace.export_swarm", {"run_dir": "ace_runs/a", "out_dir": "swarms/a"}, store)
    assert argv == ["-m", "ace.export_swarm", "ace_runs/a", "swarms/a"]
    argv, _ = argv_env("ace.report", {"run_dir": "ace_runs/a"}, store)
    assert argv == ["-m", "ace.report", "ace_runs/a"]
    argv, _ = argv_env("ace.infer", {"swarm": "results/trained_swarms/div_s8", "problem": "nafems_le1", "mock": True}, store)
    assert argv == ["-m", "ace.infer", "--swarm", "results/trained_swarms/div_s8", "--problem", "nafems_le1",
                    "--agents", "all", "--mock", "--out", "/abs/out"]


def test_combinations_the_loops_cannot_run_are_refused(tmp_path):
    from ace import config
    from feacore.problems import registry

    registry.discover()
    outside = next(p for p in sorted(registry._REGISTRY) if p not in config.TASKS)   # noqa: SLF001
    store = RunStore(tmp_path)
    for tool, args, needle in [
        ("ace.swarm", {"problems": outside}, "task catalogue"),
        ("ace.swarm", {"holdout": "not_a_problem"}, "task catalogue"),
        ("ace.swarm", {"deep": True, "memory": "shared"}, "needs memory shared"),
        ("ace.swarm", {"init_agents": "a6;a3"}, "agent names"),
        ("ace.single", {"problems": outside}, "task catalogue"),
        ("ace.infer", {"swarm": "swarms/x", "problem": "cooks_membrane", "agents": "everyone"}, "agent names"),
    ]:
        with pytest.raises(T.ToolArgsError, match=needle):
            argv_env(tool, args, store)


def test_real_runs_need_the_api_key_unless_mock(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with TestClient(create_app(runs_dir=tmp_path, token="")) as c:
        for tool, args in [("ace.swarm", {"mock": False}), ("ace.single", {"mock": False}),
                           ("ace.infer", {"swarm": "swarms/x", "problem": "cooks_membrane", "mock": False})]:
            r = c.post("/api/tasks", json={"tool": tool, "args": args})
            assert r.status_code == 422 and "ANTHROPIC_API_KEY" in r.text, tool
        assert {t["name"] for t in c.get("/api/tools?group=agents").json()["tools"]} == AGENT_TOOLS


# --------------------------------------------------------------------------- a hand-made run folder


def _write(p: Path, data) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, bytes):
        p.write_bytes(data)
    else:
        p.write_text(data if isinstance(data, str) else json.dumps(data))
    return p


def _record(gen: int, agent: str, score: float, base: float) -> dict:
    return {"problem_id": "cooks_membrane", "run_id": f"g{gen}-cooks_membrane-{agent}", "generation": gen, "turns": 8,
            "stop": "submitted", "cited": ["mesh-0001"], "submitted": 1, "playbook_text": f"context of {agent} in gen {gen}",
            "trace": [{"type": "thinking", "text": "plan", "turn": 0},
                      {"type": "tool_call", "id": "t0", "name": "mesh", "input": {"h": 2.0, "order": 2, "using": ["mesh-0001"]}, "turn": 0},
                      {"type": "tool_result", "id": "t0", "name": "mesh", "result": {"ok": True, "nodes": 500}, "turn": 0}],
            "meshes": [{"ok": True, "nodes": 500, "order": 2, "index": 0}],
            "solves": [{"mesh": {"nodes": 300, "order": 2, "h_used": 3.0}, "result": {"ok": True, "qoi": "uy_C", "value": 23.9},
                        "ui_run_id": "run-x1"},
                       {"mesh": {"nodes": 500, "order": 2, "h_used": 2.0}, "result": {"ok": True, "qoi": "uy_C", "value": 23.95,
                                                                                  "info": {"n_dofs": 1000}}}],
            "usage": [{"usd": 0.01}, {"usd": 0.02}],
            "checks": {"score": score, "base_score": base, "best_bonus": score - base, "diversity_bonus": 0.0, "dofs": 1000,
                       "field_error": 0.02, "probe_error": 0.01, "rel_error": 0.02, "meets_tol": True, "valid": True,
                       "failed_gate": None, "node_cap": 20000, "nodes": 500, "order": 2, "accuracy": 1.0, "efficiency": 0.5,
                       "solved": True, "n_solves": 2, "setup": {}}}


def make_run(root: Path) -> Path:
    """A finished two-generation --deep style swarm run on one problem, agents a0 and a1."""
    d = root / "fake"
    _write(d / "run_config.json", {"suite": "fea", "train": ["cooks_membrane"], "holdout": []})
    for g, scores in ((0, {"a0": (0.30, 0.20), "a1": (0.50, 0.40)}), (1, {"a0": (0.70, 0.60), "a1": (0.60, 0.50)})):
        for a, (s, b) in scores.items():
            _write(d / f"gen{g}" / "runs" / f"g{g}-cooks_membrane-{a}.json", _record(g, a, s, b))
            _write(d / f"gen{g}" / "context" / f"{a}.md", f"## SHARED PLAYBOOK\nline one\n{'line two' if g else 'old line'}\n")
            _write(d / "private" / f"agent{a[1:]}_gen{g}.json", {"bullets": [{"id": f"{a}-mesh-000{g + 1}", "section": "meshing",
                                                                               "content": f"note {g}"}]})
        _write(d / f"playbook_gen{g}.json", {"bullets": [{"id": "mesh-0001", "section": "meshing", "content": "order 2"}]})
    rep = d / "gen0" / "reports"
    _write(rep / "g0-cooks_membrane-a0_mesh.png", PNG)
    # the report lists its pictures relative to wherever the loop ran: found by name in reports/
    _write(rep / "g0-cooks_membrane-a0.json", {"numbers": {"score": 0.3}, "images": {"mesh": "elsewhere/run/gen0/reports/g0-cooks_membrane-a0_mesh.png"}})
    _write(d / "gen0" / "teacher_cooks_membrane.json", {"task_summary": "summary", "per_agent": [
        {"agent": "a0", "run_id": "g0-cooks_membrane-a0", "diagnosis": "too coarse", "strengths": ["order 2"], "mistakes": [],
         "notes": [{"section": "meshing", "content": "refine the corner"}]}]})
    _write(d / "gen0" / "reflections" / "g0-cooks_membrane-a0.json", {"what_happened": "it went fine", "what_i_learned": [], "plan_next": "refine"})
    _write(d / "gen0" / "organizer" / "a0.json", {"before": [], "incoming": [{"content": "x"}], "after": [{"id": "a0-mesh-0001", "content": "note 0"}]})
    _write(d / "cost.jsonl", "".join(json.dumps({"t": 1000 + i, "role": "generator", "run_id": f"g{i // 2}-cooks_membrane-a0", "usd": 0.05}) + "\n"
                                     for i in range(4)))
    _write(d / "summary.md", "# ACE run summary\n\n| gen | mean |\n|---|---|\n| 0 | 0.40 |\n")
    _write(d / "scores.png", PNG)
    return d


@pytest.fixture
def client(tmp_path, roots):
    store = tmp_path / "store"
    make_run(store / "ace_runs")
    with TestClient(create_app(runs_dir=store, token="")) as c:
        yield c


def test_runs_are_listed_with_a_summary(client):
    runs = client.get("/api/ace/runs").json()["runs"]
    r = next(x for x in runs if x["name"] == "fake")
    assert r["key"] == "@runs/ace_runs/fake" and r["kind"] == "swarm" and r["deep"] and r["status"] == "finished"
    assert r["generations_scored"] == 2 and r["agents"] == ["a0", "a1"] and r["train"] == ["cooks_membrane"]
    assert r["usd"] == pytest.approx(0.2) and r["calls"] == 4
    assert r["last"]["generation"] == 1 and r["last"]["mean"] == pytest.approx(0.65) and r["last"]["best"] == pytest.approx(0.7)
    assert r["has"]["summary"] and r["has"]["scores_png"] and r["has"]["private"] and not r["has"]["report"]


def test_run_detail_scores_agents_and_files(client):
    d = client.get("/api/ace/run", params={"dir": "@runs/ace_runs/fake"}).json()
    g0, g1 = d["generations"]
    assert g0["problems"]["cooks_membrane"]["mean"] == pytest.approx(0.4) and g1["problems"]["cooks_membrane"]["best"] == pytest.approx(0.7)
    assert g0["usd"] == pytest.approx(0.1) and g1["cum_usd"] == pytest.approx(0.2)
    a0 = next(a for a in d["agents"] if a["agent"] == "a0")
    assert [x["rank"] for x in a0["generations"]["0"] + a0["generations"]["1"]] == [2, 1]
    assert d["score_label"] == "base score" and d["summary_md"].startswith("# ACE run summary") and d["bridge"] is False
    f = client.get("/api/ace/file", params={"dir": "@runs/ace_runs/fake", "path": "scores.png"})
    assert f.status_code == 200 and f.headers["content-type"] == "image/png"


def test_one_agent_run_has_what_the_streamlit_tracker_showed(client):
    a = client.get("/api/ace/agent-run", params={"dir": "@runs/ace_runs/fake", "run_id": "g0-cooks_membrane-a0"}).json()
    assert a["brief"]["score"] == pytest.approx(0.3) and a["checks"]["field_error"] == 0.02
    assert a["report"]["images"] == {"mesh": "gen0/reports/g0-cooks_membrane-a0_mesh.png"}
    assert a["teacher"]["diagnosis"] == "too coarse" and a["teacher_task"]["task_summary"] == "summary"
    assert a["reflection"]["what_happened"] == "it went fine" and len(a["organizer"]["added"]) == 1
    assert [s["ui_run_id"] for s in a["solves"]] == ["run-x1", None] and a["solves"][1]["scored"]
    assert a["trace"][1] == {"type": "call", "name": "mesh", "turn": 0, "using": ["mesh-0001"], "input": '{"h": 2.0, "order": 2}'}
    assert a["notes"]["shared"][0]["cited"] is True
    b = client.get("/api/ace/agent-run", params={"dir": "@runs/ace_runs/fake", "run_id": "g1-cooks_membrane-a0"}).json()
    assert b["context"]["prev_gen"] == 0 and "+line two" in b["context"]["diff"] and "-old line" in b["context"]["diff"]
    assert b["notes"]["private"][0]["new"] is True and [r["gen"] for r in b["context_history"]] == [0, 1]
    img = client.get("/api/ace/file", params={"dir": "@runs/ace_runs/fake", "path": a["report"]["images"]["mesh"]})
    assert img.status_code == 200 and img.content == PNG


def test_paths_outside_the_known_folders_are_refused(client, tmp_path):
    store = tmp_path / "store"
    outside = tmp_path / "outside"
    _write(outside / "run_config.json", {"train": []})
    _write(outside / "secret.png", PNG)
    (store / "ace_runs" / "escape").symlink_to(outside, target_is_directory=True)   # a symlink out of the root
    (store / "ace_runs" / "fake" / "leak.png").symlink_to(outside / "secret.png")
    assert "escape" not in [r["name"] for r in client.get("/api/ace/runs").json()["runs"]]
    for key in ["../outside", "/etc", "@runs/../outside", "@runs/ace_runs/../../outside", "@runs/ace_runs/escape",
                "@runs/ace_runs/fake/../escape", "feacore", "ace", ".", "@runs", "@runs/ace_runs", "@runs/tasks"]:
        for url in ("/api/ace/run", "/api/ace/tracker", "/api/ace/tracker/data", "/api/ace/report", "/api/ace/infer"):
            assert client.get(url, params={"dir": key}).status_code == 404, (url, key)
    for path in ["../run_config.json", "/etc/passwd", "gen0/../../fake/scores.png", "leak.png", "run_config.json.html",
                 "summary.md/../../../outside/secret.png", "gen0\\reports\\x.png"]:
        assert client.get("/api/ace/file", params={"dir": "@runs/ace_runs/fake", "path": path}).status_code == 404, path
    assert client.get("/api/ace/agent-run", params={"dir": "@runs/ace_runs/fake", "run_id": "../../x"}).status_code == 404


def test_task_outputs_and_a_symlinked_root(tmp_path, monkeypatch):
    """A task's output folder counts only inside a root, the run store or two levels below the
    repository; a root may be a symlink to another disk (folders are then named through it)."""
    store = tmp_path / "store"
    disk = tmp_path / "bigdisk" / "ace_runs"
    make_run(disk)
    store.mkdir()
    (store / "ace_runs").symlink_to(disk, target_is_directory=True)
    monkeypatch.setattr(ace_runs, "RUN_ROOTS", ["@runs/ace_runs"])
    ts = T.TaskStore(RunStore(store))
    for tid, out in [("task-a1", T.REPO_ROOT / "feacore"), ("task-a2", store / "ace_runs" / "queued"),
                     ("task-a3", tmp_path / "elsewhere")]:
        ts.save({"id": tid, "tool": "ace.swarm", "group": "agents", "state": "queued", "args": {"generations": 3},
                 "out": str(out), "out_abs": str(out), "created_at": "2026-11-05T10:00:00Z"})
    with TestClient(create_app(runs_dir=store, token="")) as c:
        runs = {r["key"]: r for r in c.get("/api/ace/runs").json()["runs"]}
        assert set(runs) == {"@runs/ace_runs/fake", "@runs/ace_runs/queued"}
        assert runs["@runs/ace_runs/queued"]["status"] == "interrupted"   # the server restarted: marked at startup
        assert c.get("/api/ace/run", params={"dir": "@runs/ace_runs/fake"}).json()["summary"]["generations_scored"] == 2
        assert c.get("/api/ace/run", params={"dir": "feacore"}).status_code == 404


def test_the_tracker_is_served_under_the_api_and_carries_the_token(tmp_path, roots):
    store = tmp_path / "store"
    make_run(store / "ace_runs")
    with TestClient(create_app(runs_dir=store, token="s3cret")) as c:
        q = {"dir": "@runs/ace_runs/fake"}
        assert c.get("/api/ace/runs").status_code == 401
        assert c.get("/api/ace/tracker", params=q).status_code == 401
        page = c.get("/api/ace/tracker", params={**q, "token": "s3cret"})
        assert page.status_code == 200 and "default-src 'none'" in page.headers["content-security-policy"]
        assert 'fetch("/api/ace/tracker/data?dir=%40runs%2Face_runs%2Ffake&token=s3cret"' in page.text
        assert c.get("/api/ace/tracker/data", params=q).status_code == 401
        data = c.get("/api/ace/tracker/data", params={**q, "token": "s3cret"}).json()
        assert data["run"] == "fake" and data["phase"] == "finished" and len(data["generations"]) == 2


def test_swarms_inference_results_and_comparisons(client, tmp_path):
    store = tmp_path / "store"
    sw = store / "swarms" / "sw1"
    _write(sw / "manifest.json", {"run": "fake", "kind": "swarm", "generation": 1, "tasks": ["cooks_membrane"],
                                  "agents": ["a0", "a1"], "models": {"generator": "m"}, "exported_at": "2026-11-05 10:00:00",
                                  "source": str(store / "ace_runs" / "fake")})
    for a in ("a0", "a1"):
        _write(sw / "private" / f"{a}.json", {"bullets": [{"id": "n1", "content": "x"}]})
        _write(sw / "context" / f"{a}.md", f"context {a}")
    s = next(x for x in client.get("/api/ace/swarms").json()["swarms"] if x["name"] == "sw1")
    assert s["key"] == "@runs/swarms/sw1" and s["agents"] == ["a0", "a1"] and s["source_key"] == "@runs/ace_runs/fake"
    det = client.get("/api/ace/swarm", params={"dir": s["key"]}).json()
    assert [a["notes"] for a in det["agent_list"]] == [1, 1] and det["agent_list"][0]["context"] == "context/a0.md"
    assert client.get("/api/ace/file", params={"dir": s["key"], "path": "context/a0.md", "swarm": True}).text == "context a0"
    assert client.get("/api/ace/swarm", params={"dir": "@runs/ace_runs/fake"}).status_code == 404   # not a swarm folder

    inf = store / "ace_runs" / "inf1"
    _write(inf / "results.json", {"problem": "nafems_le1", "task": {"qoi": "syy_D"}, "mode": "point", "usd": 0.1,
                                  "swarm": {"run": "fake", "generation": 1, "tasks": ["cooks_membrane"]},
                                  "agents": {"a0": {"run_id": "infer-nafems_le1-a0", "prediction": {
                                      "qois": {"syy_D": 92.7}, "images": {"mesh": "/somewhere/work/a0/final_mesh.png"}}}},
                                  "consensus": {"qois": {"syy_D": {"median": 92.7, "min": 92.7, "max": 92.7, "n": 1, "spread_rel": 0.0}},
                                                "most_confident": None}})
    _write(inf / "work" / "a0" / "final_mesh.png", PNG)
    i = next(x for x in client.get("/api/ace/infers").json()["infers"] if x["name"] == "inf1")
    assert i["problem"] == "nafems_le1" and i["n_ok"] == 1 and i["consensus"]["median"] == 92.7
    det = client.get("/api/ace/infer", params={"dir": i["key"]}).json()
    assert det["agents"][0]["images"] == {"mesh": "work/a0/final_mesh.png"}

    _write(store / "ace_runs" / "cmp1" / "compare.md", "# Swarm vs single agent\n\nswarm: `ace_runs/deep2`; single agent: `ace_runs/single1`\n")
    cm = next(x for x in client.get("/api/ace/compares").json()["compares"] if x["name"] == "cmp1")
    assert cm["swarm"] == "ace_runs/deep2" and cm["single"] == "ace_runs/single1"
    assert client.get("/api/ace/compare", params={"dir": cm["key"]}).json()["compare_md"].startswith("# Swarm vs single")
    assert not any(r["name"] in ("inf1", "cmp1") for r in client.get("/api/ace/runs").json()["runs"])   # not training runs


# --------------------------------------------------------------------------- end to end (mock LLM)


def _wait(c, tid, timeout=300):
    t0 = time.time()
    while time.time() - t0 < timeout:
        rec = c.get(f"/api/tasks/{tid}").json()
        if rec["state"] not in ("queued", "running"):
            return rec
        time.sleep(0.5)
    raise AssertionError(f"task {tid} still {rec['state']}")


def test_train_export_and_run_a_mock_swarm_through_the_dashboard(tmp_path, roots, monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    with TestClient(create_app(runs_dir=tmp_path / "store", token="")) as c:
        # 1. train: two agents, two generations, one problem (about ten seconds with the mock LLM)
        r = c.post("/api/tasks", json={"tool": "ace.swarm", "args": {
            "mock": True, "problems": "cooks_membrane", "holdout": "none", "agents": 2, "concurrency": 2,
            "generations": 2, "budget_usd": 1, "out": "@runs/ace_runs/e2e"}})
        assert r.status_code == 202, r.text
        rec = r.json()
        assert rec["out_abs"].endswith(os.path.join("ace_runs", "e2e")) and rec["argv"][rec["argv"].index("--out") + 1] == rec["out_abs"]
        listed = next(x for x in c.get("/api/ace/runs").json()["runs"] if x["name"] == "e2e")   # listed at once
        assert listed["task"]["id"] == rec["id"] and listed["status"] in ("queued", "running")
        done = _wait(c, rec["id"])
        assert done["state"] == "done", c.get(f"/api/tasks/{rec['id']}/log").text[-3000:]
        run = next(x for x in c.get("/api/ace/runs").json()["runs"] if x["name"] == "e2e")
        assert run["status"] == "finished" and run["generations_scored"] == 2 and run["agents"] == ["a0", "a1"]
        detail = c.get("/api/ace/run", params={"dir": run["key"]}).json()
        assert detail["bridge"] and len(detail["runs"]) == 4                   # the run-store bridge recorded it
        assert any(loop["loop_id"] == "e2e" for loop in c.get("/api/swarm").json()["loops"])
        assert c.get("/api/ace/tracker/data", params={"dir": run["key"]}).json()["phase"] == "finished"
        first = detail["runs"][0]["run_id"]
        solves = c.get("/api/ace/agent-run", params={"dir": run["key"], "run_id": first}).json()["solves"]
        assert all(s["ui_run_id"] for s in solves if s["ok"])                  # field-viewer links

        # 2. export what it learned
        ex = c.post("/api/tasks", json={"tool": "ace.export_swarm", "args": {"run_dir": run["key"], "out_dir": "@runs/swarms/e2e"}}).json()
        assert _wait(c, ex["id"])["state"] == "done", c.get(f"/api/tasks/{ex['id']}/log").text[-3000:]
        sw = next(x for x in c.get("/api/ace/swarms").json()["swarms"] if x["name"] == "e2e")
        assert sw["agents"] == ["a0", "a1"] and sw["status"] == "ready"

        # 3. run the exported swarm on a problem (one agent, mock LLM)
        inf = c.post("/api/tasks", json={"tool": "ace.infer", "args": {"swarm": sw["key"], "problem": "cooks_membrane",
                                                                       "agents": "a0", "mock": True, "out": "@runs/ace_runs/infer-e2e"}}).json()
        assert _wait(c, inf["id"])["state"] == "done", c.get(f"/api/tasks/{inf['id']}/log").text[-3000:]
        res = c.get("/api/ace/infer", params={"dir": "@runs/ace_runs/infer-e2e"}).json()
        assert res["problem"] == "cooks_membrane" and res["n_ok"] == 1 and res["results"]["consensus"]["qois"]
        a0 = res["agents"][0]
        assert a0["ui_submission_id"] and all(s["ui_run_id"] for s in a0["solves"] if s["ok"])   # bridged too


def test_a_mock_single_agent_run_through_the_dashboard(tmp_path, roots, monkeypatch):
    """The single-agent baseline: always field-scored (ACE_SCORE=field set for the task), and recorded
    in the run store like the swarm (its run-store bridge, on because the task runner sets
    FEACORE_RUNS_DIR)."""
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    with TestClient(create_app(runs_dir=tmp_path / "store", token="")) as c:
        rec = c.post("/api/tasks", json={"tool": "ace.single", "args": {
            "mock": True, "problems": "cooks_membrane", "generations": 1, "out": "@runs/ace_runs/single-e2e"}}).json()
        assert rec["env"] == {"ACE_SCORE": "field"}
        assert _wait(c, rec["id"])["state"] == "done", c.get(f"/api/tasks/{rec['id']}/log").text[-3000:]
        run = next(x for x in c.get("/api/ace/runs").json()["runs"] if x["name"] == "single-e2e")
        assert run["kind"] == "single" and run["status"] == "finished" and run["agents"] == ["big"]
        detail = c.get("/api/ace/run", params={"dir": run["key"]}).json()
        assert detail["bridge"] and detail["runs"][0]["run_id"] == "g0-cooks_membrane-big"
        a = c.get("/api/ace/agent-run", params={"dir": run["key"], "run_id": "g0-cooks_membrane-big"}).json()
        assert a["context"]["text"] and a["report"]["images"]   # its context and the run report's pictures
