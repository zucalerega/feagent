"""Teacher demonstrations: every recipe the teacher tried, verified by the solver, ranked per problem.

    python -m wsp.demos            # wsp/data/teacher/<pid>.jsonl -> wsp/data/demos.jsonl (+ a summary)

The teacher (Claude Opus) wrote recipes for each training part from the student's own prompt and scored
each one with `python -m wsp.task score ... --log` (mesh + solve + field error vs the fine reference).
Here each attempt is re-scored from its recipe (the score is deterministic and cached, so a logged number
cannot drift from what the recipe gives), identical recipes are merged, and the valid ones are ranked
by x the automesher: rank 0 = that problem's best recipe, the distillation target.
"""

from __future__ import annotations

import json
from pathlib import Path

DATA = Path(__file__).with_name("data")


def collect(min_x: float = 0.0, workers: int = 3) -> list[dict]:
    from wsp import problems, task
    rows = []
    for pid in problems.train_ids():
        f = DATA / "teacher" / f"{pid}.jsonl"
        if not f.exists():
            continue
        attempts = [json.loads(line) for line in f.read_text().splitlines() if line.strip()]
        seen, uniq = set(), []
        for i, a in enumerate(attempts):
            try:
                key = task.dumps(a["spec"])
            except Exception:  # noqa: BLE001  a malformed attempt (it scored invalid anyway)
                continue
            if key not in seen:
                seen.add(key)
                uniq.append((i, a))
        scored = task.score_many([(pid, a["spec"], None) for _, a in uniq], workers=workers)
        good = sorted(((r, i, a) for (i, a), r in zip(uniq, scored) if r["valid"] and r["x_auto"] >= min_x),
                      key=lambda t: -t[0]["x_auto"])
        for rank, (r, i, a) in enumerate(good):
            rows.append({"pid": pid, "rank": rank, "attempt": i, "spec": task.canonical(a["spec"]),
                         "x_auto": r["x_auto"], "gain": r["gain"], "cost": r["cost"], "e": r["e"],
                         "dofs": r["dofs"], "note": a.get("note", ""), "attempts": len(attempts),
                         "source": "teacher"})
    return rows


def main() -> int:
    rows = collect()
    out = DATA / "demos.jsonl"
    out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    best = [r for r in rows if r["rank"] == 0]
    print(f"{len(rows)} valid recipes, {len(best)} problems; best x_auto per problem:")
    for r in best:
        print(f"  {r['pid']:42s} {r['x_auto']:6.2f}x  ({r['attempts']} attempts)  {json.dumps(r['spec'])}")
    if best:
        import statistics
        print(f"mean of the best: {statistics.mean(r['x_auto'] for r in best):.2f}x, "
              f"median {statistics.median(r['x_auto'] for r in best):.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
