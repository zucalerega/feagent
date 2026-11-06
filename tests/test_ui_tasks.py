"""The dashboard's task lane (feacore/ui/tasks.py): whitelisted tools, argument checks, child processes,
logs, graceful stop and kill. The tools here are tiny `python -c` programs registered by the test."""

from __future__ import annotations

import sys
import time

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from feacore.runstore import RunStore  # noqa: E402
from feacore.ui import tasks as T  # noqa: E402
from feacore.ui.server import create_app  # noqa: E402

ECHO = T.Tool(name="test.echo", title="Echo", group="test",
              command=["-c", "import sys, pathlib; print('hello', sys.argv[1:]); "
                             "o = sys.argv[sys.argv.index('--out') + 1]; pathlib.Path(o, 'done.txt').write_text('ok')"],
              params=[T.Param("word", required=True), T.Param("count", type="int", default=1, min=1, max=5),
                      T.Param("loud", type="bool", default=False), T.Param("out", type="path")],
              out_param="out", out_root="@runs/test_outputs")
SLEEP = T.Tool(name="test.sleep", title="Sleep", group="test",
               command=["-c", "import time, pathlib, sys\n"
                              "o = pathlib.Path(sys.argv[sys.argv.index('--out') + 1])\n"
                              "for _ in range(600):\n"
                              "    if (o / 'STOP').exists(): print('stopping'); sys.exit(0)\n"
                              "    time.sleep(0.1)\n"],
               params=[T.Param("out", type="path")], out_param="out", stop_file="STOP")
NEEDS_KEY = T.Tool(name="test.needs_key", title="Needs a key", group="test", command=["-c", "print(1)"],
                   params=[T.Param("mock", type="bool", default=False)],
                   requires_env=["FEACORE_TEST_SURELY_UNSET_KEY"], unless_param="mock")
PROBLEMS = T.Tool(name="test.problems", title="Problems", group="test", command=["-c", "print(1)"],
                  params=[T.Param("problems", type="problems")])


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    for t in (ECHO, SLEEP, NEEDS_KEY, PROBLEMS):
        T.TOOLS[t.name] = t
    app = create_app(runs_dir=tmp_path_factory.mktemp("runs"), token="")
    with TestClient(app) as c:
        yield c
    for t in (ECHO, SLEEP, NEEDS_KEY, PROBLEMS):
        T.TOOLS.pop(t.name, None)


def _wait(client, tid, states=("done", "failed", "cancelled"), timeout=60):
    t0 = time.time()
    while time.time() - t0 < timeout:
        rec = client.get(f"/api/tasks/{tid}").json()
        if rec["state"] in states:
            return rec
        time.sleep(0.2)
    raise AssertionError(f"task {tid} still {rec['state']}")


def test_tools_are_listed_with_their_schema(client):
    d = client.get("/api/tools").json()
    echo = next(t for t in d["tools"] if t["name"] == "test.echo")
    assert [p["name"] for p in echo["params"]] == ["word", "count", "loud", "out"]
    assert echo["out_param"] == "out" and echo["can_stop"] is False
    assert next(t for t in d["tools"] if t["name"] == "test.sleep")["can_stop"] is True


def test_a_task_runs_logs_and_writes_its_output(client):
    r = client.post("/api/tasks", json={"tool": "test.echo", "args": {"word": "abc", "count": "3", "loud": True}})
    assert r.status_code == 202, r.text
    rec = r.json()
    assert rec["argv"][2:] == ["--word", "abc", "--count", "3", "--loud", "--out", rec["out_abs"]]
    assert rec["out"].startswith("@runs/test_outputs/echo-")
    done = _wait(client, rec["id"])
    assert done["state"] == "done" and done["returncode"] == 0
    log = client.get(f"/api/tasks/{rec['id']}/log").text
    assert "hello" in log and "'abc'" in log
    from pathlib import Path

    assert (Path(done["out_abs"]) / "done.txt").read_text() == "ok"
    listed = client.get("/api/tasks?group=test").json()["tasks"]
    assert rec["id"] in [t["id"] for t in listed]


@pytest.mark.parametrize("args, needle", [
    ({}, "word is required"),
    ({"word": "x", "nope": 1}, "unknown parameter"),
    ({"word": "--evil"}, "must not start with '-'"),
    ({"word": "x", "count": 9}, "must be <= 5"),
    ({"word": "x", "count": "many"}, "not a number"),
    ({"word": "x", "out": "../../etc"}, "inside the repository or the run store"),
    ({"word": "x", "out": "/etc"}, "inside the repository or the run store"),
    ({"word": "x\x01"}, "control characters"),
])
def test_bad_arguments_are_refused_before_anything_runs(client, args, needle):
    r = client.post("/api/tasks", json={"tool": "test.echo", "args": args})
    assert r.status_code == 422 and needle in r.text


def test_unknown_tools_and_tasks_are_404(client):
    assert client.post("/api/tasks", json={"tool": "rm -rf", "args": {}}).status_code == 404
    assert client.get("/api/tasks/task-nope").status_code == 404
    assert client.get("/api/tasks/..%2F..%2Fetc").status_code == 404


def test_problem_ids_are_checked_against_the_registry(client):
    r = client.post("/api/tasks", json={"tool": "test.problems", "args": {"problems": "cooks_membrane,not_a_problem"}})
    assert r.status_code == 422 and "not_a_problem" in r.text
    assert client.post("/api/tasks", json={"tool": "test.problems", "args": {"problems": "cooks_membrane"}}).status_code == 202


def test_a_missing_api_key_is_reported_unless_mock(client):
    r = client.post("/api/tasks", json={"tool": "test.needs_key", "args": {}})
    assert r.status_code == 422 and "FEACORE_TEST_SURELY_UNSET_KEY" in r.text
    assert client.post("/api/tasks", json={"tool": "test.needs_key", "args": {"mock": True}}).status_code == 202


def test_graceful_stop_writes_the_stop_file_and_kill_terminates(client):
    a = client.post("/api/tasks", json={"tool": "test.sleep", "args": {}}).json()
    _wait(client, a["id"], states=("running",))
    client.post(f"/api/tasks/{a['id']}/cancel?mode=stop")
    assert _wait(client, a["id"])["state"] == "done"            # the tool saw STOP and exited cleanly
    assert "stopping" in client.get(f"/api/tasks/{a['id']}/log").text
    b = client.post("/api/tasks", json={"tool": "test.sleep", "args": {}}).json()
    _wait(client, b["id"], states=("running",))
    assert client.get("/api/health").json()["active_jobs"] >= 1  # serve.sh waits for running tasks
    client.post(f"/api/tasks/{b['id']}/cancel?mode=kill")
    assert _wait(client, b["id"])["state"] == "cancelled"


def test_interrupted_tasks_are_marked_at_startup(tmp_path):
    store = RunStore(tmp_path)
    ts = T.TaskStore(store)
    ts.save({"id": "task-x1", "tool": "test.echo", "state": "running", "pid": None, "created_at": "2026-01-01T00:00:00Z"})
    assert ts.mark_interrupted() == 1 and ts.get("task-x1")["state"] == "interrupted"


def test_build_argv_positional_and_bool_flags():
    tool = T.Tool(name="x", title="x", group="g", command=["-m", "m"],
                  params=[T.Param("target", flag=""), T.Param("mock", type="bool", flag_false="--real"),
                          T.Param("n", type="int")])
    assert T.build_argv(tool, {"target": "abc", "mock": False, "n": 2}) == ["-m", "m", "--real", "--n", "2", "abc"]
    assert T.build_argv(tool, {"target": "abc", "mock": True, "n": None}) == ["-m", "m", "--mock", "abc"]
    assert sys.executable  # the runner always uses the server's own interpreter


def test_display_tools_start_through_the_offscreen_launcher(tmp_path):
    """A tool that renders (Tool.display) runs as python -m feacore.ui.offscreen <its command>: VTK gets a
    virtual display on a headless server instead of segfaulting. Already-wrapped commands stay as they are."""
    runner = T.TaskRunner(RunStore(tmp_path))
    plain = T.Tool(name="test.render", title="Render", group="test", command=["-m", "x.render"], display=True)
    wrapped = T.Tool(name="test.render2", title="Render", group="test", command=[*T.OFFSCREEN, "scripts/r.py"],
                     display=True)
    for t in (plain, wrapped):
        T.TOOLS[t.name] = t
    try:
        assert runner.submit("test.render", {})["argv"] == [*T.OFFSCREEN, "-m", "x.render"]
        assert runner.submit("test.render2", {})["argv"] == [*T.OFFSCREEN, "scripts/r.py"]
    finally:
        runner.stop()
        for t in (plain, wrapped):
            T.TOOLS.pop(t.name, None)
