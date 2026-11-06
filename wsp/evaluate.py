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


def zz_of(pid: str, spec: dict) -> float | None:
    """Reference-free error estimate of a recipe's solution: the ZZ (Zienkiewicz-Zhu) relative energy-norm
    error, the same estimate the swarm agents see after every solve (ace.solve_feedback)."""
    import tempfile

    from ace import config, fea, field_error
    from ace.checks import reference_setup
    from feacore.diagnostics.zz import zz_estimate
    from feacore.diagnostics.view import SolutionView
    from wsp import problems, task
    problems.ensure_tasks([pid])
    req = task.to_request(pid, spec)
    p = fea.get_problem(pid)
    with tempfile.TemporaryDirectory() as tmp:
        mesh = fea.make_mesh(p, Path(tmp) / "m.msh", h=req["h"], order=req["order"], refinements=req["refinements"])
        if not mesh.get("ok"):
            return None
        res = field_error._solve(p, mesh, reference_setup(p), config.TASKS[pid]["qoi"])
        return float(zz_estimate(SolutionView.from_result(res))["eta_rel"])


def _zz_job(job):
    import os
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    pid, spec = job
    try:
        return zz_of(pid, spec)
    except Exception:  # noqa: BLE001
        return None


def select_zz(eval_dir: str | Path, out: str | Path, workers: int = 3) -> dict:
    """Best-of-k without the reference: per part, keep the valid sample with the smallest ZZ error x DOFs
    (the score is proportional to 1 / (error x DOFs) once a mesh beats the automesher 2x)."""
    from concurrent.futures import ProcessPoolExecutor
    eval_dir, out = Path(eval_dir), Path(out)
    rows = [json.loads(x) for x in (eval_dir / "results.jsonl").read_text().splitlines()]
    cand = [r for r in rows if r["valid"]]
    with ProcessPoolExecutor(workers) as ex:
        zz = list(ex.map(_zz_job, [(r["pid"], r["spec"]) for r in cand]))
    for r, z in zip(cand, zz):
        r["zz"] = z
    picked = []
    for pid in dict.fromkeys(r["pid"] for r in rows):
        mine = [r for r in rows if r["pid"] == pid]
        ok = [r for r in mine if r["valid"] and r.get("zz") is not None]
        best = min(ok, key=lambda r: r["zz"] * r["dofs"]) if ok else mine[0]
        picked.append({**best, "variant": best["variant"] + "+zz", "k": len(mine)})
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in picked))
    summary = {"variant": picked[0]["variant"] if picked else "", "source": str(eval_dir), "select": "zz_x_dofs",
               **summarize(picked)}
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
    ap.add_argument("--select-zz", default=None, help="post-process an eval dir: best-of-k by the ZZ estimate")
    a = ap.parse_args(argv)
    if a.select_zz:
        print(json.dumps(select_zz(a.select_zz, a.out, workers=a.workers), indent=1))
        return 0
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
