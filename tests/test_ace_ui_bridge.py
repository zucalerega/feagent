"""ace.ui_bridge: ACE solves and agent runs recorded in the feacore run store, for the web UI.
Mock LLM only: no API calls, no key needed."""

import json

import pytest

import feacore.runstore
from ace import fea, run_loop, ui_bridge
from ace.agent import Tools, run_agent
from ace.checkpoint import scored_solve_index
from ace.checks import check_run, score_population
from ace.mock import MockLLM
from ace.run_loop import main
from feacore.runstore import RunStore

TRAIN, HOLDOUT = "cooks_membrane", "nafems_le1"     # small 2D problems keep the loop tests fast
SHARED = ["--mock", "--memory", "shared", "--runs-per-problem", "1"]
RUN_META = {"source", "loop_id", "generation", "ace_run_id", "solve_index", "mesh"}
SUB_META = {"source", "loop_id", "generation", "ace_run_id", "agent", "submitted_run_id", "score", "score_parts",
            "gates", "setup_diagnostics", "cited_bullets", "tokens", "diagnosis"}


def test_loop_records_every_solve_and_agent_run(tmp_path, monkeypatch):
    monkeypatch.setenv("FEACORE_RUNS_DIR", str(tmp_path / "store"))
    store = RunStore(tmp_path / "store")
    loop_check_run = run_loop.check_run

    def check_run_after_the_bridge(pid, run):   # the loop's own checks, once the whole generation is done
        assert store.list_submissions(agent_id=f"ace:{run['run_id']}"), "not recorded when the run finished"
        return loop_check_run(pid, run)

    monkeypatch.setattr(run_loop, "check_run", check_run_after_the_bridge)
    out = tmp_path / "loop-a"
    main(SHARED + ["--generations", "2", "--concurrency", "2", "--problems", TRAIN, "--holdout", HOLDOUT,
                   "--out", str(out)])
    ace_runs = {p.stem: json.loads(p.read_text()) for p in out.glob("gen*/runs/*.json")}

    runs = store.list_runs()
    assert len(runs) == sum(len(a["solves"]) for a in ace_runs.values()) > 0
    for r in runs:                                   # one run record per solve
        m = r["meta"]
        ace = ace_runs[m["ace_run_id"]]
        assert RUN_META <= set(m) and r["source"] == m["source"] == "ace" and m["loop_id"] == "loop-a"
        assert r["agent_id"] == f"ace:{ace['run_id']}" and r["problem_id"] == ace["problem_id"]
        assert m["generation"] == ace["generation"]
        solve = ace["solves"][m["solve_index"]]
        assert solve["ui_run_id"] == r["run_id"] and m["mesh"] == solve["mesh"]["request"]
        assert r["status"] == "succeeded" and r["qois"] and r["fields"]
        assert (store.root / r["mesh_file"]).exists()

    subs = store.list_submissions()
    assert sorted(s["meta"]["ace_run_id"] for s in subs) == sorted(ace_runs)   # one per agent run
    for sub in subs:
        m = sub["meta"]
        ace = ace_runs[m["ace_run_id"]]
        c = ace["checks"]
        assert SUB_META <= set(m) and m["source"] == "ace" and m["loop_id"] == "loop-a" and m["agent"] is None
        assert sub["agent_id"] == f"ace:{ace['run_id']}" and sub["problem_id"] == ace["problem_id"]
        assert ace["ui_submission_id"] == sub["submission_id"] and m["generation"] == ace["generation"]
        # the loop's final speed/accuracy score and its parts, among all runs on the problem
        assert m["score"] == c["score"] and 0 <= m["score"] <= 1
        assert {k: m["score_parts"][k] for k in ui_bridge.SCORE_PARTS} == {k: c.get(k) for k in ui_bridge.SCORE_PARTS}
        assert m["score_parts"]["population"] == sum(a["problem_id"] == ace["problem_id"] for a in ace_runs.values())
        if c["failed_gate"]:
            assert m["gates"] == [c["failed_gate"]]
        else:                                        # invalid = feacore fail flags, else no gate
            assert m["gates"] == [] if c["valid"] else m["gates"][0].startswith("invalid: ")
        assert sub["run_ids"] == [s["ui_run_id"] for s in ace["solves"] if s["result"]["ok"]]
        assert m["submitted_run_id"] == ace["solves"][scored_solve_index(ace)]["ui_run_id"]
        if stage := ace["stages"]["first_failing_stage"]:
            assert f"first failing stage: {stage}" in m["setup_diagnostics"]
        for name, o in (ace["oracles"] or {}).items():
            if isinstance(o, dict) and o.get("status") == "fail":
                assert f"{name}: {o['why']}" in m["setup_diagnostics"]
        assert m["cited_bullets"] == ace["cited"]
        assert m["tokens"]["total"] == sum(m["tokens"][k] for k in ("input", "output", "cache_write", "cache_read")) > 0
    assert any(s["meta"]["cited_bullets"] for s in subs)   # gen 1 agents cite the gen-0 bullet

    # The diagnoser's verdict (gen 0, training problem) survives gen 1's rescoring.
    diag = json.loads((out / "gen0" / f"diagnosis_{TRAIN}.json").read_text())["diagnosis"]
    assert diag["runs"]
    for rd in diag["runs"]:
        m = store.list_submissions(agent_id=f"ace:{rd['run_id']}")[0]["meta"]
        assert m["diagnosis"]["status"] == rd["diagnosis_status"] and m["diagnosis"]["cause"] == rd["cause"]
        assert any(x.startswith(f"diagnoser ({rd['diagnosis_status']})") for x in m["setup_diagnostics"])


def test_failed_solve_pending_score_rescoring_and_diagnosis(tmp_path):
    from ace.diagnoser import Diagnosis, RunDiagnosis

    ui = ui_bridge.settings(tmp_path / "store", loop_id="t")
    tools = Tools(TRAIN, tmp_path / "work", ui_bridge.for_run(ui, "g0-t-a1", 0))
    h = fea.get_problem(TRAIN).params["h0"]
    refine = [{"point": [0.0, 0.0], "size": h, "radius": h}]
    out, err = tools.call("mesh", {"h": h, "order": 1, "refinements": refine, "using": []})
    assert not err and out["within_cap"]
    out, err = tools.call("solve", {"formulation": "plane_strain", "material": {"E": 1.0, "nu": 0.3},
                                    "bcs": [{"type": "fixed", "group": "no_such_group"}], "using": []})
    assert err
    store = RunStore(ui["root"])
    (r,) = store.list_runs()                         # a failed solve is a failed run
    assert r["status"] == "failed" and "no_such_group" in r["error"] and r["agent_id"] == "ace:g0-t-a1"
    assert r["meta"]["solve_index"] == 0 and r["meta"]["mesh"] == {"h": h, "order": 1, "refinements": refine}
    assert tools.solves[0]["ui_run_id"] == r["run_id"]

    # When the run ends: a submission right away (create_submission refuses one without a successful
    # solve, so the bridge writes the same record), score null until the loop scores the population.
    run = {"problem_id": TRAIN, "run_id": "g0-t-a1", "generation": 0, "submitted": None, "cited": [],
           "meshes": tools.meshes, "solves": tools.solves, "usage": []}
    ui_bridge.record_agent_run(ui, run)
    (sub,) = store.list_submissions()
    assert sub["schema"] == feacore.runstore.SUBMISSION_SCHEMA and sub["run_ids"] == []
    assert run["ui_submission_id"] == sub["submission_id"]
    m = sub["meta"]
    assert m["score"] is None and m["gates"] == ["no successful solve"] and m["agent"] == "a1"
    assert m["submitted_run_id"] is None and m["tokens"] is None and m["score_parts"]["population"] is None

    # The loop scores the population, then the diagnoser looks at the run, then a later rescoring.
    run["checks"] = check_run(TRAIN, run)
    score_population([run])
    ui_bridge.record_scores(ui, {TRAIN: [run]})
    ui_bridge.record_diagnosis(ui, [run], Diagnosis(divergence="", lessons=[], runs=[RunDiagnosis(
        run_id="g0-t-a1", diagnosis_status="confirmed", stage="bcs", category="missing_constraint",
        cause="supports on a group that does not exist", evidence="rerun")]))
    ui_bridge.record_scores(ui, {TRAIN: [run]})
    (sub,) = store.list_submissions()                # updated in place, not duplicated
    m = sub["meta"]
    assert m["score"] == 0 and m["score_parts"]["valid"] is False and m["score_parts"]["population"] == 1
    assert m["gates"] == ["no successful solve"]
    assert m["diagnosis"]["status"] == "confirmed" and m["diagnosis"]["stage"] == "bcs"
    assert ("diagnoser (confirmed): bcs / missing_constraint: supports on a group that does not exist"
            in m["setup_diagnostics"])

    # A run that crashed (API or code error): ACE leaves it out of its scores, so score null.
    crashed = {**run, "run_id": "g0-t-a2", "error": "Traceback ...\nAPIError: overloaded", "solves": []}
    crashed.pop("ui_submission_id")
    ui_bridge.record_agent_run(ui, crashed)
    m = store.list_submissions(agent_id="ace:g0-t-a2")[0]["meta"]
    assert m["score"] is None and m["gates"] == [] and m["error"] == "APIError: overloaded"


def test_off_without_env_or_flag_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.delenv("FEACORE_RUNS_DIR", raising=False)
    assert ui_bridge.settings(None, "x") is None
    default = RunStore()                             # where an unconfigured store would write

    def snapshot():
        return sorted(map(str, default.root.glob("*/*"))) if default.root.exists() else []

    before = snapshot()
    out = tmp_path / "loop"
    main(SHARED + ["--generations", "1", "--problems", TRAIN, "--holdout", "", "--out", str(out)])
    assert snapshot() == before
    (ace,) = [json.loads(p.read_text()) for p in out.glob("gen0/runs/*.json")]
    assert ace["solves"] and "ui_submission_id" not in ace and all("ui_run_id" not in s for s in ace["solves"])

    made = []                                        # in-process: the store is never even opened
    monkeypatch.setattr(feacore.runstore.RunStore, "__init__", lambda self, *a, **k: made.append(a))
    run_agent(TRAIN, "g0-off-0", 0, "", tmp_path / "work", MockLLM())
    assert made == []


def test_store_errors_only_warn_and_leave_the_run_unchanged(tmp_path, monkeypatch):
    plain = run_agent(TRAIN, "g0-w-0", 0, "", tmp_path / "plain", MockLLM())   # bridge off

    ui = ui_bridge.settings(tmp_path / "store", loop_id="t")

    def disk_full(*a, **k):
        raise OSError("disk full")

    for name in ("record_result", "create_run", "create_submission", "get_submission"):
        monkeypatch.setattr(RunStore, name, disk_full)
    monkeypatch.setattr(feacore.runstore, "write_json", disk_full)
    with pytest.warns(RuntimeWarning, match="ui_bridge.record_solve failed.*disk full"):
        run = run_agent(TRAIN, "g0-w-0", 0, "", tmp_path / "work", MockLLM(), ui=ui)
    assert run["stop"] == plain["stop"] == "submitted" and run["submitted"] == plain["submitted"]
    assert [s["result"]["value"] for s in run["solves"]] == [s["result"]["value"] for s in plain["solves"]]
    assert all("ui_run_id" not in s for s in run["solves"])
    with pytest.warns(RuntimeWarning, match="ui_bridge.record_agent_run failed.*disk full"):
        ui_bridge.record_agent_run(ui, run)
    assert "ui_submission_id" not in run
    run["checks"] = check_run(TRAIN, run)
    score_population([run])
    with pytest.warns(RuntimeWarning, match="ui_bridge.record_scores failed.*1 submission"):
        ui_bridge.record_scores(ui, {TRAIN: [run]})


def test_unusable_store_dir_warns_once_and_the_loop_runs(tmp_path):
    not_a_dir = tmp_path / "file"
    not_a_dir.write_text("")
    with pytest.warns(RuntimeWarning, match="ui_bridge.settings failed"):
        main(SHARED + ["--generations", "1", "--problems", TRAIN, "--holdout", "",
                       "--ui-runs-dir", str(not_a_dir), "--out", str(tmp_path / "loop")])
    assert (tmp_path / "loop" / "summary.md").exists()
    assert len(list((tmp_path / "loop" / "gen0" / "runs").glob("*.json"))) == 1


def test_ui_runs_dir_flag_and_persistent_agents(tmp_path, monkeypatch):
    monkeypatch.delenv("FEACORE_RUNS_DIR", raising=False)
    main(["--mock", "--memory", "shared+private", "--agents", "2", "--generations", "1", "--problems", TRAIN,
          "--holdout", "", "--ui-runs-dir", str(tmp_path / "store"), "--out", str(tmp_path / "loop")])
    store = RunStore(tmp_path / "store")
    subs = store.list_submissions()
    assert sorted((s["meta"]["ace_run_id"], s["meta"]["agent"]) for s in subs) == \
        [(f"g0-{TRAIN}-a0", "a0"), (f"g0-{TRAIN}-a1", "a1")]
    assert all(s["meta"]["score"] is not None and s["meta"]["score_parts"]["population"] == 2 for s in subs)
    assert {r["run_id"] for r in store.list_runs()} == {i for s in subs for i in s["run_ids"]}
