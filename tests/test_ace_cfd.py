"""The ACE loop on flow problems (the CFD suite, run_loop --suite cfd).

The plumbing tests (tools, setups, meshes, cases, interventions) run anywhere. The solver tests
need OpenFOAM and are skipped without it; on a Mac, scripts/openfoam_docker.sh runs it in Docker.
"""

import json

import pytest

from ace import checks, config, fea
from ace.agent import TOOLS, TOOLS_FLOW, system_prompt, tools_for
from ace.counterfactual import apply_bc_change

CFD = [pid for pid in config.TASKS if fea.is_flow(fea.get_problem(pid))]


def _has_openfoam() -> bool:
    from feacore.solver._foam_env import find_openfoam

    try:
        return find_openfoam() is not None
    except Exception:  # noqa: BLE001 - a broken install counts as none
        return False


FOAM = pytest.mark.skipif(not _has_openfoam(), reason="OpenFOAM not found (on a Mac: scripts/openfoam_docker.sh)")


def test_cfd_suite_lists_registered_flow_tasks():
    assert set(config.SUITES) == {"fea", "cfd", "cfd3d"}
    for suite in ("cfd", "cfd3d"):
        train, holdout = config.SUITES[suite]
        assert set(train + holdout) <= set(CFD)
        assert not set(train) & set(holdout)
    assert all(fea.get_problem(p).dim == 3 for p in sum(config.SUITES["cfd3d"], []))


@pytest.mark.parametrize("pid", CFD)
def test_flow_task_plumbing(pid, tmp_path):
    p = fea.get_problem(pid)
    assert tools_for(pid) is TOOLS_FLOW
    s = checks.reference_setup(p)
    assert s["convection"] == "linearUpwind" and checks.reference_setup(p, 1)["convection"] == "upwind"
    m = fea.make_mesh(p, tmp_path / "m.msh", h=p.params["h0"] * 1.5, quads=p.dim == 2)
    assert m["ok"] and m["order"] == 1
    fea.build_case(p, m, s, config.TASKS[pid]["qoi"]).check()
    r = checks.bc_report(p, s)
    assert r["material_ok"] and r["formulation_ok"] and not r["missing"] and not r["unexpected"]
    assert checks.bc_report(p, dict(s, fluid=dict(s["fluid"], mu=s["fluid"]["mu"] * 1000)))["material_ok"] is False


def test_flow_prompt_and_tools():
    assert "flow (CFD)" in system_prompt("", flow=True) and "linear-elastic" in system_prompt("")
    names = [t["name"] for t in TOOLS_FLOW]          # flow's own view / compare (ace.flow_feedback)
    assert names == [t["name"] for t in TOOLS]
    assert {t["name"]: t for t in TOOLS_FLOW}["view"]["input_schema"]["properties"]["kind"]["enum"] == \
        ["mesh", "speed", "pressure", "error"]
    flow = system_prompt("", flow=True)
    assert "von Mises" not in flow and "stress" not in flow
    assert "- view:" in flow and "- compare:" in flow and "error indicator" in flow
    mesh = TOOLS_FLOW[1]["input_schema"]["properties"]
    assert "order" not in mesh and "quads" in mesh


def test_quads_only_for_2d_flow(tmp_path):
    assert not fea.make_mesh(fea.get_problem("square_duct_3d"), tmp_path / "m.msh", h=0.003, quads=True)["ok"]


def test_unknown_convection_scheme_is_rejected(tmp_path):
    p = fea.get_problem("poiseuille_channel")
    m = fea.make_mesh(p, tmp_path / "m.msh", h=0.002)
    with pytest.raises(ValueError, match="convection"):
        fea.build_case(p, m, dict(checks.reference_setup(p), convection="QUICK"), "dp_AB")


def test_flow_interventions():
    s = checks.reference_setup(fea.get_problem("poiseuille_channel"))
    t = apply_bc_change(s, "convert_units", {"target": "mu", "factor": 1000})
    assert t["fluid"]["mu"] == pytest.approx(1000 * s["fluid"]["mu"])
    inlet = next(b for b in apply_bc_change(s, "convert_units", {"target": "loads", "factor": 2})["bcs"]
                 if b["type"] == "velocity_inlet")
    assert inlet["value"][0].endswith(")*2.0") and inlet["value"][1] == 0.0    # expression profiles are wrapped
    assert apply_bc_change(s, "set_convection", {"scheme": "upwind"})["convection"] == "upwind"


def test_flow_has_no_baseline_diagnostic_flags():
    from ace.oracles import baseline_fail_flags

    assert baseline_fail_flags("taylor_couette_2d") == frozenset()


@FOAM
def test_flow_solve_on_an_agent_mesh(tmp_path):
    p = fea.get_problem("taylor_couette_2d")
    m = fea.make_mesh(p, tmp_path / "m.msh", h=0.0007, quads=True)
    res = fea.run_solve(p, m, checks.reference_setup(p), "u_theta_mid")
    assert res["ok"], res
    assert abs(res["value"] / fea.reference_value(p, "u_theta_mid") - 1) < 0.03


@FOAM
def test_cfd_loop_end_to_end_mock(tmp_path):
    from ace.run_loop import main

    main(["--mock", "--suite", "cfd", "--memory", "shared", "--runs-per-problem", "1", "--generations", "2",
          "--concurrency", "1", "--problems", "poiseuille_channel", "--holdout", "taylor_couette_2d",
          "--out", str(tmp_path)])
    assert json.loads((tmp_path / "run_config.json").read_text())["suite"] == "cfd"
    rec = json.loads((tmp_path / "gen1" / "runs" / "g1-poiseuille_channel-0.json").read_text())
    assert rec["checks"]["solved"] and rec["checks"]["score"] is not None
    assert rec["oracles"]["known_answer"]["status"] in ("pass", "fail")
    assert rec["oracles"]["equilibrium"]["status"] == "unavailable"     # elasticity-only oracle


def test_run_loop_refuses_flow_tasks_it_cannot_run(monkeypatch):
    """Before anything is spent: requirements statements and the inspecting teacher are
    elasticity-only, and flow needs OpenFOAM. Field scoring (ace.flow_field) works on flow."""
    from ace import run_loop
    from feacore.solver import _foam_env

    run_loop._check_flow_tasks(["cooks_membrane"], "inspect")     # no flow task: nothing to check
    monkeypatch.setattr(config, "SCORE", "field")
    if _has_openfoam():
        run_loop._check_flow_tasks(["poiseuille_channel"], "rerun")
    with pytest.raises(SystemExit, match="--teacher inspect"):
        run_loop._check_flow_tasks(["poiseuille_channel"], "inspect")
    monkeypatch.setattr(config, "SCORE", "point")
    monkeypatch.setattr(config, "STATEMENT", "requirements")
    with pytest.raises(SystemExit, match="--statement requirements"):
        run_loop._check_flow_tasks(["poiseuille_channel"])
    monkeypatch.setattr(config, "STATEMENT", "bcs")

    def missing():
        raise _foam_env.FoamNotFound("OpenFOAM not found")

    monkeypatch.setattr(_foam_env, "require_openfoam", missing)
    with pytest.raises(SystemExit, match="need OpenFOAM"):
        run_loop._check_flow_tasks(["pipe_poiseuille_3d"])


@FOAM
def test_preflight_solves_the_good_mesh(capsys):
    from ace import preflight

    assert preflight.main(["--problems", "poiseuille_channel"]) == 0
    assert "pre-flight PASSED" in capsys.readouterr().out
