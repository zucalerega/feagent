"""single_agent: design packages, the blind harness, the evaluator, the rulebook and the loop plumbing."""

from __future__ import annotations

import json
import math

import pytest

pytest.importorskip("gmsh")

from single_agent import config, workspace  # noqa: E402
from single_agent.harness import Session, gci_unequal  # noqa: E402
from single_agent.oracle import reference_setup  # noqa: E402
from single_agent.rulebook import Rulebook, lint  # noqa: E402

PID = "abc_00000003_bc0"


# --------------------------------------------------------------------------- design packages


@pytest.mark.parametrize("pid", workspace.packaged())
def test_package_is_clean_and_keyed(pid):
    """No face tags or solver vocabulary in what the agent reads; every question keyed to a gold QoI."""
    assert workspace.check(pid) == []


def test_package_lint_catches_leaks():
    issues = workspace.lint("The face f21 is fixed; apply a traction of 3 MPa (boundary condition).")
    assert any("face tag" in i for i in issues)
    assert any("support-type" in i for i in issues)
    assert any("traction" in i or "solver" in i for i in issues)


def test_default_split_is_packaged_and_disjoint():
    assert set(config.TRAIN) <= set(workspace.packaged())
    assert set(config.HOLDOUT) <= set(workspace.packaged())
    assert not set(config.TRAIN) & set(config.HOLDOUT)
    suspect = {p for p in workspace.packaged() if workspace.load_key(p).get("suspect")}
    assert not suspect & (set(config.TRAIN) | set(config.HOLDOUT))


# --------------------------------------------------------------------------- the study's GCI


def test_gci_unequal_ratios():
    # converging at p = 1 on unequal ratios (fine step 1.34, coarse 1.21): feacore's gci3 calls R > 1 divergent
    h = [1.0, 1.34, 1.34 * 1.21]
    phi = [2.0 - 0.1 * x for x in h]
    r = gci_unequal(h, phi, 2.0)
    assert r["convergence"] == "monotone"
    assert math.isclose(r["p"], 1.0, rel_tol=1e-6)
    assert math.isclose(r["phi_ext"], 2.0, rel_tol=1e-9)
    # diverging (a singular peak) with the small step on the fine side
    h = [1.0, 1.1, 1.1 * 1.6]
    assert gci_unequal(h, [x ** -0.3 for x in h], 2.0)["convergence"] == "divergent"
    # equal ratios: gci3 is right, nothing to correct
    assert gci_unequal([1.0, 1.5, 2.25], [1.0, 1.1, 1.3], 2.0) is None


# --------------------------------------------------------------------------- rulebook


def test_rule_lint():
    assert lint("When a face rests on a greased plate, hold only its normal displacement, because it slides.") == []
    assert lint("Fix face f21 and apply 7.8 kN, because the package says so in its review notes.")
    assert lint("The answer is about 2.86e-05 for this shoulder, so report that value directly to the review.")


def test_rulebook_apply_dedup_and_prune():
    rb = Rulebook.initial("handbook")
    text = "When a peak stress sits at the rim of a bonded face, report it as singular, because it grows without bound."
    log = rb.apply([{"op": "add", "section": "singularities", "text": text, "reason": "r"}], 0)
    assert log[0]["result"] == "added" and rb.learned[0]["id"] == "L001"
    log = rb.apply([{"op": "add", "section": "singularities", "text": text.replace("report", "flag"), "reason": "r"},
                    {"op": "add", "section": "faces", "text": "Use face f12 for the pad.", "reason": "r"}], 1)
    assert log[0]["result"].startswith("duplicate of L001")
    assert log[1]["result"].startswith("rejected")
    rb.apply([{"op": "vote", "id": "L001", "vote": "harmful", "reason": "r"}] * 4, 2)
    assert rb.learned == []                                  # 4 harmful vs 1 helpful: pruned
    assert "A2" in rb.ids() and "Learned rules" not in rb.render()


# --------------------------------------------------------------------------- harness + evaluator


@pytest.fixture(scope="module")
def ws(tmp_path_factory):
    return workspace.make(PID, tmp_path_factory.mktemp("sa"))


def test_workspace_holds_nothing_private(ws):
    assert sorted(p.name for p in ws.iterdir()) == ["geometry.step", "package.md"]
    text = (ws / "package.md").read_text()
    assert PID not in text and ws.name.startswith("part-")


def test_harness_is_blind(ws):
    """The geometry description must not reveal which faces the reference holds or loads."""
    s = Session(ws)
    geo, err = s.call("inspect_geometry", {})
    assert not err
    blob = json.dumps(geo)
    assert "NAMED in statement" not in blob and '"role"' not in blob
    assert not geo.get("support_boundary") and not geo.get("load_boundary")
    faces, err = s.call("find_faces", {"query": {"kind": "plane", "normal": [-1, 0, 0]}})
    assert not err and "f18" in faces["faces"]


def _study(session, setup, h=0.012):
    out, err = session.call("mesh", {"h": h, "order": 2})
    assert not err, out
    meshes = [out["mesh"]]
    for _ in range(2):
        out, err = session.call("refine_mesh", {"mesh": meshes[-1], "factor": 1.4})
        assert not err, out
        meshes.append(out["mesh"])
    solves = []
    for m in meshes:
        out, err = session.call("solve", {"mesh": m, **setup})
        assert not err, out
        solves.append(out["solve"])
    out, err = session.call("study", {"solves": solves})
    assert not err, out
    return solves, out


def test_correct_setup_scores_equivalent(tmp_path):
    from single_agent.evaluate import evaluate
    ws = workspace.make(PID, tmp_path)
    s = Session(ws)
    setup = reference_setup(PID)
    solves, study = _study(s, setup)
    assert set(study["qois"]) >= {"u_mean_f18", "vm_mean"}
    fb = s.state["solves"][-1]["feedback"]["equilibrium"]
    assert fb["residual"] < 1e-6                             # reactions balance the 7.8 kN pull
    answers = [{"id": "Q1", "quantity": "u_mean_f18", "value": study["qois"]["u_mean_f18"]["values"][-1],
                "uncertainty": 0.05, "status": "converged"},
               {"id": "Q2", "quantity": "vm_mean", "value": study["qois"]["vm_mean"]["values"][-1],
                "uncertainty": 0.05, "status": "converged"}]
    out, err = s.call("submit_report", {"idealization": [], "final_solve": solves[-1], "study_solves": solves,
                                        "answers": answers, "summary": "test"})
    assert not err and s.finished
    card = evaluate(PID, {"problem_id": PID, "session": s.state, "usage": [], "turns": 0})
    assert card["setup"]["equivalent"] is True
    assert card["setup"]["match"]["matched"] == card["setup"]["match"]["of"] == 3
    assert card["parts"]["study"] == 1.0
    assert [a["id"] for a in card["answers"]] == ["Q1", "Q2"]
    assert all(a["supported"] for a in card["answers"])
    assert 0.0 < card["score"] <= 1.0


def test_wrong_setup_is_caught(tmp_path):
    """The pull applied on the wrong shoulder: not equivalent, and the item match says where."""
    from single_agent.evaluate import evaluate
    ws = workspace.make(PID, tmp_path)
    s = Session(ws)
    setup = reference_setup(PID)
    setup["bcs"] = [b if b["type"] != "force" else {**b, "group": "f17"} for b in setup["bcs"]]
    out, err = s.call("mesh", {"h": 0.012, "order": 2})
    out, err = s.call("solve", {"mesh": 1, **setup})
    assert not err
    s.call("submit_report", {"idealization": [], "final_solve": 1, "answers": [{"id": "Q1", "value": 1e-5,
                                                                              "status": "converged"}],
                             "summary": "test"})
    card = evaluate(PID, {"problem_id": PID, "session": s.state, "usage": [], "turns": 0})
    assert card["setup"]["equivalent"] is False
    statuses = {it["reference"]: it["status"] for it in card["setup"]["match"]["items"]}
    assert statuses["fixed on f21"] == "match"
    assert any(st == "missing" for ref, st in statuses.items() if ref.startswith("force"))
    assert card["setup"]["match"]["extra"]                   # the force on f17
    assert card["parts"]["study"] == 0.0
    assert card["answers"][1]["verdict"] == "missing"


# --------------------------------------------------------------------------- the loop, end to end


@pytest.mark.slow
def test_mock_loop_runs_and_learns(tmp_path):
    from single_agent import loop
    out = tmp_path / "run"
    assert loop.main(["--mock", "--generations", "2", "--problems", PID, "--holdout", "", "--concurrency", "1",
                      "--out", str(out)]) == 0
    rows = [json.loads(ln) for ln in (out / "generations.jsonl").read_text().splitlines()]
    assert [r["generation"] for r in rows] == [0, 1]
    assert rows[0]["train"]["submitted"] == 1
    assert Rulebook.load(out / "rulebook_gen1.json").learned       # the learning step added a rule
    assert (out / "gen0" / "cards" / f"{PID}.json").exists() and (out / "summary.md").exists()


def test_duplicate_solve_is_refused_and_ratio_reported(tmp_path):
    """From the Sonnet pilot: a solve that repeats a mesh and setup is refused without using a solve, and the mesh
    tools report the representative element size and the effective refinement ratio."""
    ws = workspace.make(PID, tmp_path)
    s = Session(ws)
    m1, err = s.call("mesh", {"h": 0.012, "order": 2})
    assert not err and m1["h_rep"] > 0
    m2, err = s.call("refine_mesh", {"mesh": 1, "factor": 1.5})
    assert not err and m2["effective_ratio"] > 1.0
    setup = reference_setup(PID)
    out, err = s.call("solve", {"mesh": 1, **setup})
    assert not err
    out, err = s.call("solve", {"mesh": 1, **setup})
    assert err and "already ran this setup" in out["error"]
    assert len(s.state["solves"]) == 1                       # the refused call did not use a solve
    out, err = s.call("solve", {"mesh": 2, **setup})
    assert not err and out["solve"] == 2
    assert math.isclose(out["h_rep"], m2["h_rep"], rel_tol=0.02)   # the mesh tool's h_rep is the study's h
