"""--split with a single train task: every agent works on it (the swarm-vs-single comparison on one part)."""

import json

from ace.run_loop import main


def test_split_runs_with_one_task(tmp_path):
    out = tmp_path / "one"
    main(["--mock", "--split", "--memory", "shared+private", "--agents", "2", "--concurrency", "2",
          "--problems", "cooks_membrane", "--holdout", "", "--generations", "1", "--out", str(out)])
    groups = json.loads((out / "gen0" / "groups.json").read_text())
    assert groups == {"a0": "cooks_membrane", "a1": "cooks_membrane"}
    assert {p.name for p in (out / "gen0" / "runs").glob("*.json")} == {
        "g0-cooks_membrane-a0.json", "g0-cooks_membrane-a1.json"}
