"""Build a self-contained HTML report of one ACE run: every stage, every prompt, every number.

    python -m ace.report ace_runs/<dir>          # writes ace_runs/<dir>/report.html

The page is static: all data is embedded as JSON and rendered by a small inline script
(ace/report_template.html).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from ace import agent, config, curator, reflector
from ace.playbook import Playbook


def _read_json(path: Path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def collect(run_dir: Path) -> dict:
    run_dir = Path(run_dir)
    gens = []
    g = 0
    while (gdir := run_dir / f"gen{g}").exists():
        playbook = Playbook.load(run_dir / f"playbook_gen{g}.json")
        runs = [json.loads(p.read_text()) for p in sorted((gdir / "runs").glob("*.json"))]
        pids = sorted({r["problem_id"] for r in runs})
        refl = {}
        for pid in pids:
            out = _read_json(gdir / f"reflection_{pid}.json")
            prompt_path = gdir / f"reflection_prompt_{pid}.md"
            if prompt_path.exists():
                prompt = prompt_path.read_text()
            elif out is not None:
                # Older runs did not save the prompt. build_prompt is deterministic over the saved
                # run records and this generation's playbook, so rebuild exactly what was sent.
                pruns = [r for r in runs if r["problem_id"] == pid]
                prompt = reflector.build_prompt(pid, pruns, {r["run_id"]: r["checks"] for r in pruns}, playbook)
            else:
                continue
            refl[pid] = {"output": out, "prompt": prompt}
        nxt = run_dir / f"diff_gen{g + 1}.md"
        gens.append({
            "gen": g,
            "playbook": [vars(b) | {"net": b.net} for b in playbook.bullets],
            "playbook_text": playbook.render(config.TOP_N_BULLETS),
            "system_prompt": agent.system_prompt(playbook.render(config.TOP_N_BULLETS)),
            "runs": runs,
            "reflections": refl,
            "curator_ops": _read_json(gdir / "curator_ops.json"),
            "merge_changes": _read_json(gdir / "merge_changes.json"),
            "next_diff": nxt.read_text() if nxt.exists() else None,
        })
        g += 1

    pids = sorted({r["problem_id"] for gen in gens for r in gen["runs"]})
    split = _read_json(run_dir / "run_config.json", {})          # written by run_loop (suite, train, holdout)
    cost = [json.loads(line) for line in (run_dir / "cost.jsonl").read_text().splitlines()] \
        if (run_dir / "cost.jsonl").exists() else []
    return {
        "run_dir": run_dir.name,
        "config": {
            "models": config.MODELS, "effort": config.EFFORT, "max_tokens": config.MAX_TOKENS,
            "prices": config.PRICES, "tasks": config.TASKS, "train": split.get("train", config.TRAIN),
            "holdout": split.get("holdout", config.HOLDOUT),
            "max_solves": config.MAX_SOLVES, "max_turns": config.MAX_TURNS, "min_quality": config.MIN_QUALITY,
            "score_floor": config.SCORE_FLOOR, "score_ceil": config.SCORE_CEIL, "top_n": config.TOP_N_BULLETS,
            "dedup_threshold": config.DEDUP_THRESHOLD, "prune_margin": config.PRUNE_MARGIN,
        },
        "calibration": _read_json(Path(__file__).with_name("calibration.json"), {}),
        "prompts": {
            "generator_base": agent.BASE_PROMPT,
            "tasks": {pid: agent.task_prompt(pid) for pid in pids},
            "tools": agent.TOOLS,
            "reflector_system": reflector.SYSTEM,
            "curator_system": curator.SYSTEM,
        },
        "generations": gens,
        "cost": cost,
        "summary_md": (run_dir / "summary.md").read_text() if (run_dir / "summary.md").exists() else "",
        "audit": _read_json(run_dir / "audit.json"),   # optional: independent review of the run
    }


def build(run_dir: Path) -> Path:
    data = json.dumps(collect(run_dir), default=str).replace("</", "<\\/")
    template = Path(__file__).with_name("report_template.html").read_text()
    out = Path(run_dir) / "report.html"
    out.write_text(template.replace("/*__DATA__*/null", data))
    return out


if __name__ == "__main__":
    print(build(Path(sys.argv[1])))
