"""Field mode on flow tasks (ace.flow_field through ace.field_error, ace.checks, ace.flow_report). Needs OpenFOAM."""

import numpy as np
import pytest

from ace import checks, config, fea, field_error, flow_field, planted
from ace.agent import Tools, task_prompt
from ace.checks import reference_setup


def _has_openfoam() -> bool:
    from feacore.solver._foam_env import find_openfoam
    try:
        return find_openfoam() is not None
    except Exception:  # noqa: BLE001
        return False


pytestmark = pytest.mark.skipif(not _has_openfoam(), reason="OpenFOAM not found (on a Mac: scripts/openfoam_docker.sh)")
PID = "poiseuille_channel"          # exact solution: no fine reference solve needed


@pytest.fixture(scope="module", autouse=True)
def cache(tmp_path_factory):
    """Field caches in a temp dir; field mode on for the module."""
    mp = pytest.MonkeyPatch()
    mp.setattr(field_error, "CACHE", tmp_path_factory.mktemp("refs"))
    mp.setattr(field_error, "ERRORS_CACHE", tmp_path_factory.mktemp("errors"))
    for mod in (config, checks):
        mp.setattr(mod, "SCORE", "field")
    import ace.agent as agent
    mp.setattr(agent, "SCORE", "field")
    yield
    mp.undo()


def test_reference_quadrature_and_check_points():
    p = fea.get_problem(PID)
    ref = field_error.reference(PID)                       # dispatched to ace.flow_field
    assert ref["exact"] and ref["u"].shape == (len(ref["pts"]), 3)
    assert ref["w"].sum() == pytest.approx(p.params["L"] * p.params["H"], rel=1e-9)
    assert np.allclose(ref["u"][:, 0], p.exact("ux", ref["pts"]))
    pp = field_error.probe_points(PID)
    pts, H = np.array(pp["points"]), p.params["H"]
    assert len(pts) == config.PROBE_POINTS
    assert np.all(pts[:, 1] >= 0.2 * H - 1e-12) and np.all(pts[:, 1] <= 0.8 * H + 1e-12)   # the bulk
    assert np.allclose(pp["speed_ref"], np.abs(p.exact("ux", pts)))
    assert field_error.reference_check(PID) == {"exact": True, "coarse_vs_fine": 0.0}
    # Problem.exact returns None unless a problem overrides it: those get a fine numerical reference
    assert flow_field.has_exact(p) and not flow_field.has_exact(fea.get_problem("lid_driven_cavity_re100"))


def test_finer_mesh_smaller_field_error_and_good_mesh_beats_the_automesher(tmp_path):
    p = fea.get_problem(PID)
    errs = []
    for h in (0.002, 0.001):
        m = fea.make_mesh(p, tmp_path / f"m{h}.msh", h=h)
        errs.append(field_error.errors_of_setup(PID, m, reference_setup(p)))
    assert errs[1]["field"] < errs[0]["field"] and errs[1]["error"] == max(errs[1]["field"], errs[1]["points"])
    auto = field_error.automesher(PID)
    assert auto["h"] == p.params["h0"] and auto["dofs"] > 0
    good = fea.mesh_from_request(p, tmp_path / "good.msh", planted.GOOD_MESH[PID])
    assert field_error.errors_of_setup(PID, good, reference_setup(p))["error"] < auto["error"]


def test_field_mode_scores_and_reports_a_flow_run(tmp_path):
    t = Tools(PID, tmp_path / "run")
    out, err = t.call("mesh", {**planted.GOOD_MESH[PID], "using": []})
    assert not err, out
    out, err = t.call("solve", {**reference_setup(t.problem), "using": []})
    assert not err, out
    t.call("submit", {"solve": 1, "using": []})
    run = {"run_id": "g0-poiseuille_channel-a0", "problem_id": PID, "generation": 0, "meshes": t.meshes,
           "solves": t.solves, "submitted": t.submitted, "trace": []}
    c = checks.check_run(PID, run)
    assert c["solved"] and c["field_error"] > 0 and len(c["point_errors"]) == config.PROBE_POINTS
    assert c["rel_error"] == max(c["field_error"], c["probe_error"])
    run["checks"] = c
    checks.score_population([run])
    assert run["checks"]["valid"] and run["checks"]["gain"] > 1 and run["checks"]["score"] > 0

    from ace.flow_report import build_flow_report
    from ace.run_report import add_scores, disagreement_map, report_text
    rep = add_scores(build_flow_report(run, tmp_path / "reports"), run)
    assert "error" not in rep, rep.get("error")
    assert rep["flow"] and rep["grid_quantity"] == "speed" and set(rep["images"]) == {"mesh", "error", "speed"}
    assert rep["error_map"]["field_error"] == pytest.approx(c["field_error"], rel=1e-9)
    text = report_text(rep, hide_hidden=True)
    assert "velocity field error" in text and "H1" not in text and "hidden check points" in text
    dm = disagreement_map([rep, rep])
    assert dm["quantity"] == "speed"


def test_task_prompt_in_field_mode_scores_the_velocity():
    text = task_prompt(PID)
    assert "whole velocity field" in text and "exact solution" in text and "von Mises" not in text
    assert "uniform triangles / tetrahedra at h = 0.00125" in text
