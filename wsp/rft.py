"""Rejection-sampling fine-tuning (RFT, one round of expert iteration) with the solver as the verifier.

    python -m wsp.rft --adapter wsp_runs/sft1/adapter --out wsp_runs/rft1        # sample + score + keep
    python -m wsp.sft --data wsp/data/demos.jsonl,wsp_runs/rft1/keep.jsonl --out wsp_runs/sft_rft1

The student (usually the distilled one) samples --k recipes per training part; every sample is meshed,
solved and scored; samples that beat the automesher and come within --margin of the best known recipe
for the part (teacher's or the student's own) are kept as extra training targets. This is the cheapest
form of RL on this task: the reward is exact and costs a 2D solve, and no reward model is needed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv=None) -> int:
    from wsp import problems, task
    from wsp.editor import load_teacher
    from wsp.student import DEFAULT_MODEL, Student
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--edits", default=None)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--margin", type=float, default=0.8, help="keep samples >= margin x the best known")
    ap.add_argument("--per-problem", type=int, default=2)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pids = problems.train_ids()
    s = Student(a.model, adapter=a.adapter, edits=a.edits, threads=a.threads)
    gen = s.generate([task.prompt(p) for p in pids], temperature=a.temperature, n=a.k)
    jobs = [(p, task.parse(t)[0], None) for p, ts in zip(pids, gen) for t in ts]
    scored = task.score_many(jobs, workers=a.workers)
    teacher = load_teacher(None)
    rows, keep, i = [], [], 0
    for p, ts in zip(pids, gen):
        mine = []
        for t in ts:
            r = scored[i]
            i += 1
            rows.append({"pid": p, "answer": t, "valid": r["valid"], "x_auto": r["x_auto"], "error": r.get("error")})
            if r["valid"]:
                mine.append((r["x_auto"], r["spec"]))
        best = max([teacher.get(p, {}).get("x_auto", 0.0)] + [x for x, _ in mine])
        seen, k = set(), 0
        for x, spec in sorted(mine, key=lambda t: -t[0]):
            key = task.dumps(spec)
            if x >= max(1.0, a.margin * best) and key not in seen and k < a.per_problem:
                seen.add(key)
                keep.append({"pid": p, "rank": k, "spec": spec, "x_auto": x, "source": "rft"})
                k += 1
    (out / "samples.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (out / "keep.jsonl").write_text("".join(json.dumps(r) + "\n" for r in keep))
    valid = sum(r["valid"] for r in rows) / max(1, len(rows))
    summary = {"samples": len(rows), "valid": valid, "kept": len(keep), "problems_kept": len({r["pid"] for r in keep}),
               "mean_x": sum(r["x_auto"] for r in rows) / max(1, len(rows))}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
