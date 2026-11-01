"""Tests for the ACE loop. Everything runs in --mock mode: no API calls, no key needed."""

import json

import pytest

from ace import config
from ace.agent import parse_citations
from ace.curator import curate
from ace.merge import apply_ops, save_generation
from ace.curator import Op
from ace.mock import MockLLM
from ace.playbook import Bullet, Playbook
from ace.reflector import BulletVote, Lesson, Reflection, condense_trace
from ace.run_loop import main


def lesson(section="meshing", scope="general", content="When X, do Y, because Z."):
    return Lesson(section=section, scope=scope, content=content, evidence="test")


# --------------------------------------------------------------------------- playbook


def test_render_groups_by_section_and_caps_top_n():
    pb = Playbook([
        Bullet("mesh-0001", "meshing", "low", helpful=0),
        Bullet("bc-0001", "bcs", "high", helpful=5),
        Bullet("mesh-0002", "meshing", "mid", helpful=2),
    ])
    text = pb.render(top_n=2)
    assert "[bc-0001] high" in text and "[mesh-0002] mid" in text
    assert "mesh-0001" not in text                       # cut by top-N
    assert text.index("## meshing") < text.index("## bcs")   # section order from config


def test_next_id_and_empty_render():
    pb = Playbook()
    assert pb.render() == ""
    assert pb.next_id("meshing") == "mesh-0001"
    pb.bullets.append(Bullet("mesh-0007", "meshing", "x"))
    assert pb.next_id("meshing") == "mesh-0008"


def test_playbook_roundtrip(tmp_path):
    pb = Playbook([Bullet("geo-0001", "geometry", "c", helpful=1, source_runs=["r1"])])
    pb.save(tmp_path / "p.json")
    assert Playbook.load(tmp_path / "p.json") == pb


# --------------------------------------------------------------------------- merge


def test_merge_add_vote_dedup_prune(tmp_path):
    pb = Playbook([Bullet("mesh-0001", "meshing", "When stress is scored, use quadratic elements.", helpful=1),
                   Bullet("bc-0001", "bcs", "Bad advice.", helpful=0, harmful=1)])
    ops = [
        Op("VOTE", bullet_id="bc-0001", vote="harmful"),              # -> harmful 2 >= 0 + 2: pruned
        Op("ADD", section="meshing", content="When stress is scored, use quadratic elements!"),  # dup
        Op("ADD", section="geometry", content="When groups overlap, inspect first, because names lie."),
    ]
    changes = apply_ops(pb, ops, generation=0)
    assert [i for i, _ in changes["added"]] == ["geo-0001"]
    assert changes["deduped"][0][1] == "mesh-0001"
    assert pb.get("mesh-0001").helpful == 2
    assert [p[0] for p in changes["pruned"]] == ["bc-0001"] and pb.get("bc-0001") is None
    save_generation(pb, changes, tmp_path, 1)
    assert "geo-0001" in (tmp_path / "diff_gen1.md").read_text()
    assert len(json.loads((tmp_path / "playbook_gen1.json").read_text())["bullets"]) == 2


# --------------------------------------------------------------------------- curator


def test_curator_drops_one_off_and_skips_llm_when_section_empty():
    refl = Reflection(divergence="", root_causes=[], bullet_votes=[],
                      lessons=[lesson(), lesson(scope="one_off", content="only here")])
    ops, usage = curate([("p", refl, ["r1"])], Playbook(), MockLLM())
    assert [o.kind for o in ops] == ["ADD"] and usage == []      # no LLM call needed


def test_curator_votes_and_duplicates():
    pb = Playbook([Bullet("mesh-0001", "meshing", "existing")])
    refl = Reflection(divergence="", root_causes=[], lessons=[lesson()],
                      bullet_votes=[BulletVote(bullet_id="mesh-0001", vote="harmful", reason=""),
                                    BulletVote(bullet_id="nope-0001", vote="helpful", reason="")])
    ops, usage = curate([("p", refl, ["r1"])], pb, MockLLM())
    assert [(o.kind, o.bullet_id, o.vote) for o in ops] == [
        ("VOTE", "mesh-0001", "harmful"),        # unknown id ignored
        ("VOTE", "mesh-0001", "helpful"),        # mock curator: lesson duplicates mesh-0001
    ]
    assert len(usage) == 1


# --------------------------------------------------------------------------- small helpers


def test_parse_citations_from_tool_args_and_text():
    trace = [
        {"type": "tool_call", "name": "mesh", "input": {"h": 1, "using": ["mesh-0003", "not an id"]}},
        {"type": "tool_call", "name": "solve", "input": {"using": ["bc-0010"]}},
        {"type": "tool_call", "name": "inspect_geometry", "input": {}},
        {"type": "text", "text": "done [using solv-0002]"},
    ]
    assert parse_citations(trace) == ["bc-0010", "mesh-0003", "solv-0002"]


def test_ids_are_never_reused_after_pruning(tmp_path):
    pb = Playbook([Bullet("mesh-0001", "meshing", "a", harmful=2)])
    apply_ops(pb, [], generation=0)                                  # prunes mesh-0001
    assert pb.get("mesh-0001") is None
    pb.save(tmp_path / "p.json")
    pb = Playbook.load(tmp_path / "p.json")
    apply_ops(pb, [Op("ADD", section="meshing", content="When b, do c, because d.")], generation=1)
    assert [b.id for b in pb.bullets] == ["mesh-0002"]


def test_curator_drops_same_idea_twice_in_one_batch():
    refl = Reflection(divergence="", root_causes=[], bullet_votes=[],
                      lessons=[lesson(content="first"), lesson(content="first, reworded")])
    ops, usage = curate([("p", refl, ["r1"])], Playbook(), MockLLM())
    assert [o.kind for o in ops] == ["ADD"] and len(usage) == 1     # mock marks #1 as dup of #0


def test_submitted_solve_is_scored_not_the_last():
    from ace import fea
    from ace.checks import check_run
    pid = config.TRAIN[0]
    ref = fea.reference_value(fea.get_problem(pid), config.TASKS[pid]["qoi"])
    mesh = {"ok": True, "nodes": 10, "order": 2, "min_quality": 0.9}
    run = {"meshes": [mesh], "submitted": 0, "solves": [
        {"mesh": mesh, "setup": {}, "result": {"ok": True, "value": ref * 1.001}},
        {"mesh": mesh, "setup": {}, "result": {"ok": True, "value": ref * 1.5}},
    ]}
    c = check_run(pid, run)
    assert c["rel_error"] == pytest.approx(0.001) and c["submitted"]
    run["submitted"] = None                                          # never submitted: last solve
    assert check_run(pid, run)["rel_error"] == pytest.approx(0.5)

def test_over_cap_mesh_gets_a_scaling_hint(tmp_path):
    from ace.agent import Tools
    tools = Tools(config.TRAIN[0], tmp_path)
    cap = config.TASKS[config.TRAIN[0]]["node_cap"]
    out, is_err = tools.call("mesh", {"h": 0.002, "order": 2, "using": []})
    assert not is_err and out["nodes"] > cap and "hint" in out
    out, is_err = tools.call("solve", {"formulation": "plane_stress", "material": {"E": 1, "nu": 0.3},
                                       "bcs": [], "using": []})
    assert is_err and "over the cap" in out["error"]                # refused, no solve used
    assert tools.solves == []


def test_speed_accuracy_score_rewards_the_cheapest_accurate_run():
    from ace.checks import score_population
    mk = lambda rid, err, dofs: {"run_id": rid, "problem_id": "nafems_le1", "checks": {   # noqa: E731
        "tolerance": 0.02, "rel_error": err, "dofs": dofs, "failed_gate": None, "solved": True}}
    runs = [mk("cheap_ok", 0.01, 1000), mk("dear_ok", 0.005, 4000), mk("cheap_bad", 0.08, 500)]
    score_population(runs)
    s = {r["run_id"]: r["checks"] for r in runs}
    assert s["cheap_ok"]["score"] == pytest.approx(1.0)                 # cheapest within tolerance
    assert s["dear_ok"]["score"] == pytest.approx(0.25)                 # accurate, 4x the DOFs
    assert s["cheap_bad"]["accuracy"] == pytest.approx(0.25)            # 0.02 / 0.08
    assert s["cheap_bad"]["score"] == pytest.approx(0.25)               # efficiency capped at 1, not 2
    runs.append({"run_id": "crashed", "problem_id": "nafems_le1", "error": "boom",
                 "checks": {"tolerance": 0.02, "rel_error": None, "dofs": None, "failed_gate": "no successful solve", "solved": False}})
    score_population(runs)
    assert runs[-1]["checks"]["score"] == 0 and not runs[-1]["checks"]["valid"]


def test_condense_trace_is_capped():
    trace = [{"type": "text", "text": "x" * 1000, "turn": i} for i in range(50)]
    assert len(condense_trace(trace, max_chars=2000)) <= 2010


# --------------------------------------------------------------------------- end to end


def test_loop_end_to_end_mock(tmp_path):
    main(["--mock", "--memory", "shared", "--runs-per-problem", "1", "--generations", "2", "--concurrency", "2",
          "--out", str(tmp_path)])
    for name in ("playbook_gen0.json", "playbook_gen1.json", "diff_gen1.md", "summary.md", "cost.jsonl"):
        assert (tmp_path / name).exists(), name
    assert len(Playbook.load(tmp_path / "playbook_gen1.json").bullets) >= 1
    for pid in config.TRAIN:
        assert (tmp_path / "gen0" / f"diagnosis_{pid}.json").exists()     # REFLECTOR_MODE = "diagnoser"
    for pid in config.HOLDOUT:                                 # held out: never reflected on
        assert not (tmp_path / "gen0" / f"diagnosis_{pid}.json").exists()
    record = json.loads((tmp_path / "gen1" / "runs" / f"g1-{config.TRAIN[0]}-0.json").read_text())
    for key in ("problem_id", "run_id", "generation", "trace", "cited", "checks", "oracles", "stages"):
        assert key in record
    assert record["cited"]                                     # gen 1 agent cited a gen-0 bullet
    assert "score" in record["checks"]


def test_budget_stops_cleanly(tmp_path):
    main(["--mock", "--memory", "shared", "--runs-per-problem", "1", "--generations", "3", "--concurrency", "1",
          "--budget-usd", "0.01", "--out", str(tmp_path)])
    assert (tmp_path / "summary.md").exists()
    assert not (tmp_path / "playbook_gen1.json").exists()      # stopped before any update


# --------------------------------------------------------------------------- attribution (not wired in yet)


def test_attribution_recovers_planted_effects():
    import numpy as np
    from ace.attribution import estimate
    rng = np.random.default_rng(0)
    true = {"good-0001": 0.4, "bad-0001": -0.4, "null-0001": 0.0}
    rows = []
    for g in (1, 2, 3):
        for p, base in (("A", 0.5), ("B", 1.5), ("H", 1.0)):
            for i in range(8):
                shown = [b for b in true if rng.random() >= 0.5]
                rows.append({"run_id": f"{g}{p}{i}", "gen": g, "problem": p, "offered": list(true), "shown": shown,
                             "cited": [], "score": base + sum(true[b] for b in shown) + rng.normal(0, 0.3)})
    est = estimate(rows, train=["A", "B"], n_boot=300)
    assert est["good-0001"]["lo"] > 0 and est["bad-0001"]["hi"] < 0
    assert est["null-0001"]["lo"] < 0 < est["null-0001"]["hi"]


# --------------------------------------------------------------------------- shared + private memory


def _diag(lessons):
    from ace.diagnoser import Diagnosis, DiagnosisLesson
    return Diagnosis(divergence="", runs=[], lessons=[
        DiagnosisLesson(section="meshing", scope="general", evidence="", content=c, evidence_type=et,
                        confidence="medium", run_ids=rids) for c, et, rids in lessons])


def test_only_counterfactual_lessons_reach_shared_memory_contrast_stays_private():
    from ace.memory import Memory
    mem = Memory(3)
    d = _diag([("When refining, use quadratic elements, because they resolve gradients.", "counterfactual", ["g0-p-a0"]),
               ("When over the cap, coarsen the far field first, because the hot spot matters.", "contrast", ["g0-p-a1"])])
    mem.update([("p", d, ["g0-p-a0", "g0-p-a1"])], MockLLM(), generation=0)
    assert [b.content[:20] for b in mem.shared.bullets] == ["When refining, use q"]
    assert mem.private[1].bullets and mem.private[1].bullets[0].id.startswith("a1-")
    assert not mem.private[2].bullets                                 # agent 2 had no evidence


def test_an_idea_two_agents_learned_privately_is_promoted_to_shared():
    from ace.memory import Memory
    mem = Memory(3)
    text = "When over the cap, coarsen the far field first, because the hot spot matters."
    d = _diag([(text, "contrast", ["g0-p-a0", "g0-p-a2"])])
    log = mem.update([("p", d, ["g0-p-a0", "g0-p-a2"])], MockLLM(), generation=0)
    assert len(log["promoted"]) == 1 and mem.shared.bullets[0].content == text
    assert not mem.private[0].bullets and not mem.private[2].bullets   # moved, not duplicated


def test_private_ids_are_cited_and_routed():
    from ace.memory import Memory
    trace = [{"type": "tool_call", "name": "mesh", "input": {"using": ["a3-mesh-0002", "mesh-0001"]}}]
    assert parse_citations(trace) == ["a3-mesh-0002", "mesh-0001"]
    mem = Memory(4)
    assert mem.owner("a3-mesh-0002") is mem.private[3] and mem.owner("mesh-0001") is mem.shared
