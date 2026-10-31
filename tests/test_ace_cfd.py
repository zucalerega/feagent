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
    assert set(config.SUITES) == {"fea", "cfd"}
    assert set(config.TRAIN_CFD + config.HOLDOUT_CFD) <= set(CFD)
    assert not set(config.TRAIN_CFD) & set(config.HOLDOUT_CFD)


@pytest.mark.parametrize("pid", CFD)
def test_flow_task_plumbing(pid, tmp_path):
    p = fea.get_problem(pid)
    assert tools_for(p) is TOOLS_FLOW
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
    assert [t["name"] for t in TOOLS_FLOW] == [t["name"] for t in TOOLS]
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
