"""Single-agent baselines and the swarm-vs-baseline comparison (mock LLM, real FEA solves)."""

import json

from ace import baselines, compare
from ace.compare import check_setups
from ace.run_loop import main as run_loop

PROBLEMS = ["--problems", "cooks_membrane", "--holdout", "nafems_le1", "--generations", "2", "--concurrency", "2"]


def _exp(label, **cfg):
    base = {"problems": ["a"], "holdout": ["b"], "models": {"generator": "m"}, "max_tokens": {}, "effort": {},
            "tasks": {}, "max_solves": 4, "max_turns": 25, "min_quality": 0.1, "top_n": 30,
            "reflector_mode": "diagnoser", "diagnoser": {}, "memory_margins": {}, "code_sha256": "x", "mock": True,
            "memory": "shared+private", "agents": 6, "attempts": 1}
    return {"label": label, "loops": [{"config": {**base, **cfg}, "progress": None}], "runs": []}


def test_setup_check_allows_only_treatment_differences():
    same = [_exp("swarm"), _exp("single", agents=1, attempts=6, budget_usd=3, generations=9)]
    assert check_setups(same) == ([], [])
    diffs, _ = check_setups([_exp("swarm"), _exp("other", holdout=["c"], max_solves=8)])
    assert len(diffs) == 2 and any("holdout" in d for d in diffs) and any("max_solves" in d for d in diffs)
    _, warns = check_setups([_exp("swarm"), {"label": "old", "loops": [{"config": None}], "runs": []}])
    assert warns and "not verified" in warns[0]


def test_swarm_vs_baselines_end_to_end(tmp_path):
    run_loop(["--mock", "--memory", "shared+private", "--agents", "2", *PROBLEMS, "--out", str(tmp_path / "swarm")])
    baselines.main(["matched", "--mock", "--swarm-agents", "2", *PROBLEMS, "--out", str(tmp_path / "matched")])
    baselines.main(["isolated", "--mock", "--swarm-agents", "2", "--jobs", "2", *PROBLEMS, "--out", str(tmp_path / "isolated")])

    # the matched single agent makes as many runs per generation as the 2-agent swarm, all as agent a0
    ids = sorted(p.stem for p in (tmp_path / "matched" / "gen0" / "runs").glob("*.json"))
    assert ids == ["g0-cooks_membrane-r0-a0", "g0-cooks_membrane-r1-a0", "g0-nafems_le1-r0-a0", "g0-nafems_le1-r1-a0"]
    cfg = json.loads((tmp_path / "matched" / "run_config.json").read_text())
    assert (cfg["memory"], cfg["agents"], cfg["attempts"]) == ("shared+private", 1, 2)
    for k in ("agent0", "agent1"):
        assert (tmp_path / "isolated" / k / "summary.md").exists(), (tmp_path / "isolated" / f"{k}.log").read_text()[-2000:]

    compare.main(["--exp", f"swarm={tmp_path / 'swarm'}", "--exp", f"matched={tmp_path / 'matched'}",
                  "--exp", f"isolated={tmp_path / 'isolated'}", "--out", str(tmp_path / "cmp")])
    res = json.loads((tmp_path / "cmp" / "compare.json").read_text())
    assert res["setup_differences"] == [] and res["warnings"] == []
    by = {x["label"]: x for x in res["experiments"]}
    assert by["isolated"]["loops"] == 2
    # same compute: 2 problems x 2 runs per problem per generation x 2 generations, in every arm
    assert by["swarm"]["total_runs"] == by["matched"]["total_runs"] == by["isolated"]["total_runs"] == 8
    for x in res["experiments"]:
        assert x["final_holdout"] is not None and len(x["curve"]) == 2
    assert "diff_vs_ref" in by["matched"] and "diff_vs_ref" not in by["swarm"]
    md = (tmp_path / "cmp" / "compare.md").read_text()
    assert "matched spend" in md and "isolated" in md
    assert (tmp_path / "cmp" / "compare.png").exists()
