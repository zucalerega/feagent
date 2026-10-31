"""ace.run_report: the full per-run report of the --deep loop (one re-solve, error map, region grid, images)."""

import json

import pytest

from ace import config, fea
from ace import run_report as rr
from ace.checks import reference_setup
from ace.field_error import CACHE

PID = "cooks_membrane"
GROUPS = {"clamped", "loaded", "top", "bottom"}


@pytest.fixture
def coarse_run(tmp_path, monkeypatch):
    """A coarse uniform Cook's membrane run with the correct setup, in field mode."""
    if not (CACHE / f"{PID}.npz").exists() or not (CACHE / f"{PID}_points.json").exists():
        pytest.skip("no cached field reference for cooks_membrane")
    monkeypatch.setattr(config, "SCORE", "field")
    p = fea.get_problem(PID)
    req = {"h": 2 * p.params["h0"], "order": 1, "refinements": []}
    m = fea.make_mesh(p, tmp_path / "run.msh", **req)
    return {"run_id": f"g0-{PID}-a3", "problem_id": PID, "submitted": 0, "meshes": [{**m, "request": req}],
            "solves": [{"mesh": m, "mesh_index": 0, "setup": reference_setup(p),
                        "result": {"ok": True, "info": {"n_dofs": 2 * m["nodes"]}}}]}


def test_report_of_a_coarse_cooks_membrane(coarse_run, tmp_path):
    out = tmp_path / "reports"
    rep = rr.build_run_report(coarse_run, out)
    assert "error" not in rep, rep.get("error")
    assert {"run_id", "agent", "task", "mesh_request", "mesh_stats", "setup", "error_map", "mesh_layout",
            "region_grid", "check_points", "images", "seconds"} <= set(rep)
    assert rep["agent"] == "a3" and rep["task"] == PID
    assert rep["mesh_stats"]["n_elements"] > 0 and rep["mesh_stats"]["dofs"] == 2 * coarse_run["meshes"][0]["nodes"]
    assert json.loads((out / f"{coarse_run['run_id']}.json").read_text())["run_id"] == coarse_run["run_id"]

    # The error map (computed from the report's one solve) is the inspection tool's, number for number.
    from ace import inspect_tools
    em = inspect_tools.error_map(PID, coarse_run, tmp_path)
    assert rep["error_map"] == em
    assert set(rep["mesh_layout"]["groups"]) == GROUPS

    for kind in ("mesh", "error", "stress"):
        f = rep["images"][kind]
        with open(f, "rb") as fh:
            assert fh.read(8) == b"\x89PNG\r\n\x1a\n"
        import matplotlib.image
        assert matplotlib.image.imread(f).shape[1] <= 1100

    grid = rep["region_grid"]
    assert grid["shape"] == [10, 10] and 0 < len(grid["cells"]) <= 100
    assert sum(c["error_share"] for c in grid["cells"]) == pytest.approx(1.0, abs=1e-6)
    assert sum(c["n_elements"] for c in grid["cells"]) == rep["mesh_stats"]["n_elements"]

    cps = rep["check_points"]
    assert len(cps) == config.PROBE_POINTS
    assert [c["disclosed"] for c in cps] == [i < config.PROBE_DISCLOSED for i in range(config.PROBE_POINTS)]

    coarse_run["checks"] = {"score": 0.3, "base_score": 0.25, "best_bonus": 0.05, "field_error": 0.04, "gain": 1.5,
                            "auto_error": 0.06, "auto_dofs": 300, "dofs": 200, "rank": 2, "group_size": 3}
    n = rr.add_scores(rep, coarse_run)["numbers"]
    assert n["meets_target"] is False and n["target_error"] == pytest.approx(0.03) and n["rank_in_group"] == 2

    text = rr.report_text(rep)
    assert len(text) <= 6000 and "score 0.3" in text and "H11" in text
    assert len(rr.report_text(rep, max_chars=1500)) <= 1500


def test_report_never_raises(tmp_path):
    rep = rr.build_run_report({"run_id": "g0-cooks_membrane-a1", "problem_id": PID, "solves": [], "meshes": []},
                              tmp_path)
    assert set(rep) >= {"run_id", "error"} and (tmp_path / "g0-cooks_membrane-a1.json").exists()
    assert "no report" in rr.report_text(rep)
