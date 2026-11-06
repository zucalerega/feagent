"""Flow solve feedback, view and compare (ace.flow_feedback) through the agent's tools. Needs OpenFOAM."""

import pytest

from ace import config, fea, flow_feedback
from ace.agent import Tools
from ace.checks import reference_setup


def _has_openfoam() -> bool:
    from feacore.solver._foam_env import find_openfoam
    try:
        return find_openfoam() is not None
    except Exception:  # noqa: BLE001
        return False


pytestmark = pytest.mark.skipif(not _has_openfoam(), reason="OpenFOAM not found (on a Mac: scripts/openfoam_docker.sh)")


@pytest.fixture(scope="module")
def cylinder(tmp_path_factory):
    """Two solves of the DFG cylinder through the agent's tools: a coarse and a finer mesh."""
    t = Tools("cylinder_channel_dfg_2d1", tmp_path_factory.mktemp("cyl"))
    setup = {**reference_setup(t.problem), "using": []}
    outs = []
    for h, size in ((0.03, 0.006), (0.02, 0.004)):
        m, err = t.call("mesh", {"h": h, "quads": True, "refinements": [{"group": "cylinder", "size": size, "radius": 0.1}],
                                 "using": []})
        assert not err, m
        out, err = t.call("solve", setup)
        assert not err, out
        outs.append(out)
    return t, outs


def test_solve_feedback_reads_the_flow(cylinder):
    _, (out, _) = cylinder
    fb = out["feedback"]
    assert fb["convergence"]["converged"] and fb["cells"] > 0
    assert set(fb["qois"]) == {"drag", "lift", "dp_front_rear"}
    mb = fb["mass_balance"]
    assert mb["flow_rates"]["inlet"] < 0 < mb["flow_rates"]["outlet"] and mb["imbalance"] < 1e-6
    assert set(fb["wall_forces"]) == {"walls", "cylinder"}
    assert fb["wall_forces"]["cylinder"]["total"][0] == pytest.approx(fb["qois"]["drag"], rel=1e-3)
    zone = fb["reversed_flow"]["zones"][0]               # the recirculation bubble behind the cylinder
    assert zone["nearest_group"] == "cylinder" and zone["from"][0] > 0.25 and zone["to"][0] < 0.4
    hs = fb["error_indicator"]["hotspots"]
    assert 0 < fb["error_indicator"]["rel_indicator"] < 1 and 1 <= len(hs) <= 5
    assert sum(h["share"] for h in hs) <= 1.0 + 1e-9 and all(h["local_h"] > 0 for h in hs)


def test_finer_mesh_lowers_the_indicator(cylinder):
    _, (coarse, fine) = cylinder
    assert fine["feedback"]["error_indicator"]["rel_indicator"] < coarse["feedback"]["error_indicator"]["rel_indicator"]


def test_compare_and_view(cylinder, tmp_path):
    t, _ = cylinder
    cmp_, err = t.call("compare", {"solve_a": 1, "solve_b": 2, "using": []})
    assert not err and cmp_["coarse"]["is"] == "a" and 0 < cmp_["velocity_diff_l2_rel"] < 0.5
    assert cmp_["regions"] and set(cmp_["qoi_change"]) == {"drag", "lift", "dp_front_rear"}
    for kind in ("mesh", "speed", "pressure", "error"):
        v, err = t.call("view", {"solve": 2, "kind": kind, "using": []})
        assert not err and v["image"].endswith(".png")
    bad, err = t.call("view", {"solve": 2, "kind": "stress", "using": []})
    assert err and "kind" in bad["error"]


def test_3d_section_render_and_feedback(tmp_path):
    p = fea.get_problem("pipe_poiseuille_3d")
    m = fea.make_mesh(p, tmp_path / "m.msh", h=0.0015, quads=True, sweep={"source": "inlet", "layers": 6})
    setup = reference_setup(p)
    sol = fea.run_solve(p, m, setup, config.TASKS[p.id]["qoi"], keep_result=True)
    res = sol.pop("_result")
    fb = flow_feedback.feedback(p, m, setup, res)
    assert fb["reversed_flow"]["volume_share"] == 0.0 and fb["mass_balance"]["imbalance"] < 1e-6
    assert fb["max_speed"]["value"] == pytest.approx(0.02, rel=0.05)            # 2 U on the axis
    for kind in ("mesh", "speed", "error"):
        assert flow_feedback.render(p, m, res, kind, "z", tmp_path / f"{kind}.png").endswith(".png")
