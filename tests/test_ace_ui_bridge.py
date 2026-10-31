"""ace.ui_bridge: ACE solves and agent runs recorded in the feacore run store, for the web UI.
Mock LLM only: no API calls, no key needed."""

import json

import pytest

import feacore.runstore
from ace import config, run_loop, ui_bridge
from ace.agent import Tools, run_agent
from ace.checks import load_calibration
from ace.mock import MockLLM
from ace.run_loop import main
from feacore.runstore import RunStore

PID = config.TRAIN[0]
RUN_META = {"source", "loop_id", "generation", "ace_run_id", "solve_index", "mesh"}
SUB_META = {"source", "loop_id", "generation", "ace_run_id", "submitted_run_id", "score", "gates",
            "setup_diagnostics", "cited_bullets", "tokens"}


def scored(ace_run: dict) -> dict:
    """The solve ACE scored: the submitted one, else the last successful one."""
    ok = [s for s in ace_run["solves"] if s["result"]["ok"]]
    return ace_run["solves"][ace_run["submitted"]] if ace_run["submitted"] is not None else ok[-1]


def test_loop_records_every_solve_and_agent_run(tmp_path, monkeypatch):
    monkeypatch.setenv("FEACORE_RUNS_DIR", str(tmp_path / "store"))
    store = RunStore(tmp_path / "store")
    loop_check_run = run_loop.check_run

    def check_run(pid, run, calibration):   # the loop's own checks, once the whole generation is done
        assert store.list_submissions(agent_id=f"ace:{run['run_id']}"), "not recorded when the run finished"
        return loop_check_run(pid, run, calibration)

    monkeypatch.setattr(run_loop, "check_run", check_run)
    out = tmp_path / "loop-a"
    main(["--mock", "--runs-per-problem", "1", "--generations", "2", "--concurrency", "2", "--out", str(out)])
    ace_runs = {p.stem: json.loads(p.read_text()) for p in out.glob("gen*/runs/*.json")}

    runs = store.list_runs()
    assert len(runs) == sum(len(a["solves"]) for a in ace_runs.values()) > 0
    for r in runs:                                   # one run record per solve
        m = r["meta"]
        ace = ace_runs[m["ace_run_id"]]
        assert RUN_META <= set(m) and r["source"] == m["source"] == "ace" and m["loop_id"] == "loop-a"
        assert r["agent_id"] == f"ace:{ace['run_id']}" and r["problem_id"] == ace["problem_id"]
        assert m["generation"] == ace["generation"]
        assert ace["solves"][m["solve_index"]]["ui_run_id"] == r["run_id"]
        assert {"h", "order"} <= set(m["mesh"]) and "using" not in m["mesh"]
        assert r["status"] == "succeeded" and r["qois"] and r["fields"]
        assert (store.root / r["mesh_file"]).exists()

    subs = store.list_submissions()
    assert sorted(s["meta"]["ace_run_id"] for s in subs) == sorted(ace_runs)   # one per agent run
    for sub in subs:
        m = sub["meta"]
        ace = ace_runs[m["ace_run_id"]]
        assert SUB_META <= set(m) and m["source"] == "ace" and m["loop_id"] == "loop-a"
        assert sub["agent_id"] == f"ace:{ace['run_id']}" and sub["problem_id"] == ace["problem_id"]
        assert m["generation"] == ace["generation"] and m["score"] == ace["checks"]["score"]
        assert sub["run_ids"] == [s["ui_run_id"] for s in ace["solves"] if s["result"]["ok"]]
        assert m["submitted_run_id"] == scored(ace)["ui_run_id"]
        assert m["gates"] == ([ace["checks"]["failed_gate"]] if ace["checks"]["failed_gate"] else [])
        assert isinstance(m["setup_diagnostics"], list) and m["cited_bullets"] == ace["cited"]
        assert m["tokens"]["total"] == sum(m["tokens"][k] for k in ("input", "output", "cache_write", "cache_read")) > 0
    assert any(s["meta"]["cited_bullets"] for s in subs)   # gen 1 agents cite the gen-0 bullet


def test_failed_solve_and_runs_without_a_successful_solve(tmp_path):
    ui = ui_bridge.settings(tmp_path / "store", loop_id="t")
    tools = Tools(PID, tmp_path / "work", ui_bridge.for_run(ui, "g0-t-0", 0))
    h = load_calibration([PID])[PID]["by_order"][0]["h"] * 1.2
    refine = [{"point": [0.0, 0.0], "size": h, "radius": h}]
    out, err = tools.call("mesh", {"h": h, "order": 1, "refinements": refine, "using": []})
    assert not err and out["within_cap"]
    out, err = tools.call("solve", {"formulation": "plane_strain", "material": {"E": 1.0, "nu": 0.3},
                                    "bcs": [{"type": "fixed", "group": "no_such_group"}], "using": []})
    assert err
    store = RunStore(ui["root"])
    (r,) = store.list_runs()                         # a failed solve is a failed run
    assert r["status"] == "failed" and "no_such_group" in r["error"] and r["agent_id"] == "ace:g0-t-0"
    assert r["meta"]["solve_index"] == 0 and r["meta"]["mesh"] == {"h": h, "order": 1, "refinements": refine}
    assert tools.solves[0]["ui_run_id"] == r["run_id"]

    # No successful solve: create_submission refuses that, so the bridge writes the same record
    # with no runs (else the UI would show the agent as running forever).
    run = {"problem_id": PID, "run_id": "g0-t-0", "generation": 0, "submitted": None, "cited": [],
           "meshes": tools.meshes, "solves": tools.solves, "usage": []}
    ui_bridge.record_agent_run(ui, run, load_calibration([PID]))
    (sub,) = store.list_submissions()
    assert sub["schema"] == feacore.runstore.SUBMISSION_SCHEMA and sub["run_ids"] == []
    m = sub["meta"]
    assert m["score"] == config.SCORE_FLOOR and m["gates"] == ["no successful solve"]
    assert m["submitted_run_id"] is None and m["tokens"] is None

    # A run that crashed (API or code error): ACE leaves it out of the scores, so score None.
    crashed = {**run, "run_id": "g0-t-1", "error": "Traceback ...\nAPIError: overloaded", "solves": []}
    ui_bridge.record_agent_run(ui, crashed, load_calibration([PID]))
    m = store.list_submissions(agent_id="ace:g0-t-1")[0]["meta"]
    assert m["score"] is None and m["gates"] == [] and m["error"] == "APIError: overloaded"


def test_off_without_env_or_flag_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.delenv("FEACORE_RUNS_DIR", raising=False)
    assert ui_bridge.settings(None, "x") is None
    default = RunStore()                             # where an unconfigured store would write

    def snapshot():
        return sorted(map(str, default.root.glob("*/*"))) if default.root.exists() else []

    before = snapshot()
    out = tmp_path / "loop"
    main(["--mock", "--runs-per-problem", "1", "--generations", "1", "--problems", PID, "--holdout", "",
          "--out", str(out)])
    assert snapshot() == before
    (ace,) = [json.loads(p.read_text()) for p in out.glob("gen0/runs/*.json")]
    assert ace["solves"] and all("ui_run_id" not in s for s in ace["solves"])

    made = []                                        # in-process: the store is never even opened
    monkeypatch.setattr(feacore.runstore.RunStore, "__init__", lambda self, *a, **k: made.append(a))
    run_agent(PID, "g0-off-0", 0, "", tmp_path / "work", MockLLM())
    assert made == []


def test_store_errors_only_warn_and_leave_the_run_unchanged(tmp_path, monkeypatch):
    plain = run_agent(PID, "g0-w-0", 0, "", tmp_path / "plain", MockLLM())   # bridge off

    ui = ui_bridge.settings(tmp_path / "store", loop_id="t")

    def disk_full(*a, **k):
        raise OSError("disk full")

    for name in ("record_result", "create_run", "create_submission"):
        monkeypatch.setattr(RunStore, name, disk_full)
    monkeypatch.setattr(feacore.runstore, "write_json", disk_full)
    with pytest.warns(RuntimeWarning, match="ui_bridge.record_solve failed.*disk full"):
        run = run_agent(PID, "g0-w-0", 0, "", tmp_path / "work", MockLLM(), ui=ui)
    assert run["stop"] == plain["stop"] == "submitted" and run["submitted"] == plain["submitted"]
    assert [s["result"] for s in run["solves"]] == [s["result"] for s in plain["solves"]]
    assert all("ui_run_id" not in s for s in run["solves"])
    with pytest.warns(RuntimeWarning, match="ui_bridge.record_agent_run failed.*disk full"):
        ui_bridge.record_agent_run(ui, run, load_calibration([PID]))


def test_unusable_store_dir_warns_once_and_the_loop_runs(tmp_path):
    not_a_dir = tmp_path / "file"
    not_a_dir.write_text("")
    with pytest.warns(RuntimeWarning, match="ui_bridge.settings failed"):
        main(["--mock", "--runs-per-problem", "1", "--generations", "1", "--problems", PID, "--holdout", "",
              "--ui-runs-dir", str(not_a_dir), "--out", str(tmp_path / "loop")])
    assert (tmp_path / "loop" / "summary.md").exists()
    assert len(list((tmp_path / "loop" / "gen0" / "runs").glob("*.json"))) == 1


def test_ui_runs_dir_flag_turns_it_on(tmp_path, monkeypatch):
    monkeypatch.delenv("FEACORE_RUNS_DIR", raising=False)
    main(["--mock", "--runs-per-problem", "1", "--generations", "1", "--problems", PID, "--holdout", "",
          "--ui-runs-dir", str(tmp_path / "store"), "--out", str(tmp_path / "loop")])
    store = RunStore(tmp_path / "store")
    (sub,) = store.list_submissions()
    assert sub["meta"]["ace_run_id"] == f"g0-{PID}-0" and sub["run_ids"]
    assert {r["run_id"] for r in store.list_runs()} == set(sub["run_ids"])
