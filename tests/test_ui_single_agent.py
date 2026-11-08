"""The single analyst agent on the dashboard (feacore/ui/tools/single_agent.py): the tools, their command lines
and refusals, and one real mock run of one problem through the task lane, then a comparison of it (about 15 s)."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from feacore.runstore import RunStore  # noqa: E402
from feacore.ui import tasks as T  # noqa: E402
from feacore.ui.server import create_app  # noqa: E402

PROBLEM = "abc_00000065_bc1"


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    app = create_app(runs_dir=tmp_path_factory.mktemp("runs"), token="")
    with TestClient(app) as c:
        yield c


def _wait(client, tid, timeout=300):
    t0 = time.time()
    while time.time() - t0 < timeout:
        rec = client.get(f"/api/tasks/{tid}").json()
        if rec["state"] in ("done", "failed", "cancelled"):
            return rec
        time.sleep(0.5)
    raise AssertionError(f"task {tid} still {rec['state']}")


def test_the_three_tools_are_registered(client):
    tools = {t["name"]: t for t in client.get("/api/tools?group=single_agent").json()["tools"]}
    assert set(tools) == {"single_agent.loop", "single_agent.oracle", "single_agent.compare"}
    assert tools["single_agent.loop"]["requires_env"] == ["ANTHROPIC_API_KEY"]


def _argv(name: str, args: dict, tmp_path: Path) -> list[str]:
    tool = T.get_tool(name)
    return T.build_argv(tool, T.validate_args(tool, args, RunStore(tmp_path)))


def test_loop_command_line(tmp_path):
    argv = _argv("single_agent.loop", {"mock": True, "problems": PROBLEM, "holdout": "none", "generations": 2,
                                       "no_learn": True, "out": "sa_runs/x"}, tmp_path)
    assert argv[:2] == ["-m", "single_agent.loop"] and "--mock" in argv and "--no-learn" in argv
    assert argv[argv.index("--problems") + 1] == PROBLEM
    assert argv[argv.index("--holdout") + 1] == ""                       # 'none' = no holdout problems
    assert argv[argv.index("--generations") + 1] == "2" and argv[argv.index("--model") + 1] == "sonnet"
    default = _argv("single_agent.loop", {"mock": True}, tmp_path)
    assert "--problems" not in default and "--holdout" not in default      # the loop's own defaults


def test_oracle_and_compare_command_lines(tmp_path):
    assert _argv("single_agent.oracle", {"problems": f"{PROBLEM},abc_00000003_bc0", "levels": 2}, tmp_path)[2:] == \
        [PROBLEM, "abc_00000003_bc0", "--h", "0.006", "--levels", "2", "--factor", "1.4"]
    argv = _argv("single_agent.compare", {"run1": "sa_runs/a", "run2": "sa_runs/b", "oracle": "sa_runs/o"}, tmp_path)
    assert argv[2:] == ["sa_runs/a", "sa_runs/b", "--oracle", "sa_runs/o"]


@pytest.mark.parametrize("tool, args, needle", [
    ("single_agent.loop", {"mock": True, "problems": "cooks_membrane"}, "no design package"),
    ("single_agent.loop", {"mock": True, "holdout": "nafems_le1"}, "no design package"),
    ("single_agent.oracle", {"problems": "nope"}, "no design package"),
    ("single_agent.compare", {"run1": "/etc"}, "inside the repository or the run store"),
])
def test_refusals(client, tool, args, needle):
    r = client.post("/api/tasks", json={"tool": tool, "args": args})
    assert r.status_code == 422 and needle in r.text


def test_a_real_run_needs_an_api_key(client, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    r = client.post("/api/tasks", json={"tool": "single_agent.loop", "args": {"mock": False}})
    assert r.status_code == 422 and "ANTHROPIC_API_KEY" in r.text


def test_a_mock_run_and_its_comparison_through_the_task_lane(client):
    r = client.post("/api/tasks", json={"tool": "single_agent.loop", "args": {
        "mock": True, "problems": PROBLEM, "holdout": "none", "generations": 1, "concurrency": 1}})
    assert r.status_code == 202, r.text
    run = _wait(client, r.json()["id"])
    assert run["state"] == "done", client.get(f"/api/tasks/{run['id']}/log").text[-2000:]
    out = Path(run["out_abs"])
    assert (out / "summary.md").exists() and (out / "rulebook_final.json").exists()
    assert PROBLEM in client.get(f"/api/tasks/{run['id']}/log").text
    r = client.post("/api/tasks", json={"tool": "single_agent.compare", "args": {"run1": str(out)}})
    assert r.status_code == 202, r.text
    assert _wait(client, r.json()["id"])["state"] == "done"
