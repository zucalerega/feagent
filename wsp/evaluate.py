"""Evaluate one variant of the student (or a non-model baseline) on a split, scored by the fixed solver.

    python -m wsp.evaluate --variant base --out wsp_runs/eval/base
    python -m wsp.evaluate --variant playbook --playbook wsp/data/playbook.md --out wsp_runs/eval/playbook
    python -m wsp.evaluate --variant sft --adapter wsp_runs/sft1/adapter --out wsp_runs/eval/sft
    python -m wsp.evaluate --variant wsp --edits wsp_runs/wsp_s1/edits --out wsp_runs/eval/wsp
    python -m wsp.evaluate --variant recipes --recipes FILE.jsonl --out ...   # score given recipes (no model)

Greedy decoding (temperature 0), one answer per part, is the deployment setting and the headline number;
--n and --temperature give sampled answers. Writes results.jsonl (one row per answer, with the answer
text, the parsed recipe and every score part), summary.json and the meshes (meshes/<pid>_<k>.msh).
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path


def group_of(pid: str) -> str:
    from wsp import problems
    if pid in problems.TEST_OUT_OF_FAMILY:
        return "out_of_family"
    if pid in problems.TEST_IN_FAMILY:
        return "in_family"
    return "train"


def summarize(rows: list[dict]) -> dict:
    def stats(rs):
        xs = [r["x_auto"] for r in rs]
        if not xs:
            return {}
        return {"n": len(xs), "mean_x": statistics.mean(xs), "median_x": statistics.median(xs),
                "valid": sum(r["valid"] for r in rs) / len(rs), "beats_auto": sum(x > 1 for x in xs) / len(xs),
                "geo_mean_x_valid": (statistics.geometric_mean([x for x in xs if x > 0]) if any(x > 0 for x in xs)
                                     else 0.0)}
    out = {"all": stats(rows)}
    for g in ("in_family", "out_of_family", "train"):
        rs = [r for r in rows if r["group"] == g]
        if rs:
            out[g] = stats(rs)
    fails: dict[str, int] = {}
    for r in rows:
        if not r["valid"]:
            k = (r.get("error") or "invalid").split(":")[0][:50]
            fails[k] = fails.get(k, 0) + 1
    out["failures"] = fails
    return out


def run(variant: str, pids: list[str], out: str | Path, *, answers: dict[str, list[str]] | None = None,
        keep_meshes: bool = True, workers: int = 3, meta: dict | None = None) -> dict:
    """Score answers {pid: [answer text, ...]} and write the results."""
    from wsp import task
    out = Path(out)
    (out / "meshes").mkdir(parents=True, exist_ok=True)
    jobs, idx = [], []
    for pid in pids:
        for k, a in enumerate(answers[pid]):
            spec, _ = task.parse(a)
            jobs.append((pid, spec, str(out / "meshes" / f"{pid}_{k}.msh") if keep_meshes else None))
            idx.append((pid, k, a, spec))
    scored = task.score_many(jobs, workers=workers)
    rows = []
    for (pid, k, a, spec), r in zip(idx, scored):
        if spec is None:
            r = task.invalid(pid, "no JSON object in the answer")
        rows.append({"variant": variant, "pid": pid, "group": group_of(pid), "sample": k, "answer": a,
                     **{key: r.get(key) for key in ("valid", "error", "x_auto", "gain", "cost", "e", "field", "points",
                                                    "nodes", "dofs", "order", "spec", "mesh_path")}})
    with open(out / "results.jsonl", "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    summary = {"variant": variant, **(meta or {}), **summarize(rows)}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    return summary


def main(argv=None) -> int:
    from wsp import problems
    from wsp.student import DEFAULT_MODEL
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--edits", default=None)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--playbook", default=None)
    ap.add_argument("--recipes", default=None, help="JSONL {pid, spec}: score these recipes, no model")
    ap.add_argument("--split", default="test", help="test | train | comma-separated ids")
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-new-tokens", type=int, default=320)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args(argv)
    pids = (problems.test_ids() if a.split == "test" else problems.train_ids() if a.split == "train"
            else a.split.split(","))
    problems.ensure_tasks(pids)
    from wsp import task
    t0 = time.perf_counter()
    meta = {"split": a.split, "n": a.n, "temperature": a.temperature}
    if a.recipes:
        rec: dict[str, list[str]] = {}
        for line in Path(a.recipes).read_text().splitlines():
            r = json.loads(line)
            rec.setdefault(r["pid"], []).append(json.dumps(r["spec"]) if "spec" in r else r["answer"])
        answers = {p: rec.get(p, [""]) for p in pids}
        meta["source"] = a.recipes
    else:
        from wsp.student import Student
        s = Student(a.model, edits=a.edits, adapter=a.adapter, threads=a.threads)
        playbook = Path(a.playbook).read_text() if a.playbook else ""
        gen = s.generate([task.prompt(p, playbook) for p in pids], temperature=a.temperature, n=a.n,
                         max_new_tokens=a.max_new_tokens)
        answers = dict(zip(pids, gen))
        meta.update(model=s.path, edits=a.edits, adapter=a.adapter, playbook=a.playbook,
                    generate_seconds=round(time.perf_counter() - t0, 1))
    summary = run(a.variant, pids, a.out, answers=answers, workers=a.workers, meta=meta)
    summary["seconds"] = round(time.perf_counter() - t0, 1)
    (Path(a.out) / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
