"""The ACE loop.

    python -m ace.run_loop --mock                       # free, end to end
    python -m ace.run_loop --runs-per-problem 3 --generations 3 --budget-usd 30

Each generation:
  1. run every agent (train + held-out problems) in parallel with the current playbook
  2. check + score every run
  3. reflect once per TRAIN problem (held-out problems are never reflected on)
  4. curate the reflections into ADD/VOTE ops
  5. merge the ops into the playbook used by the next generation
  6. write the summary table
Generation 0 uses an empty playbook (the baseline). The last generation is evaluated, not
reflected on. Outputs go to ace_runs/<timestamp>/ (see ace/README.md).
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import statistics
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

from ace import config
from ace.agent import run_agent
from ace.checks import check_run, load_calibration
from ace.cost import BudgetExceeded, CostLog
from ace.curator import curate
from ace.llm import make_llm
from ace.merge import apply_ops, save_generation
from ace.playbook import Playbook
from ace.reflector import reflect


# --------------------------------------------------------------------------- one agent run


def _worker(job: dict) -> dict:
    """Runs in a child process (gmsh is process-global, so one run per process at a time)."""
    try:
        return run_agent(job["problem_id"], job["run_id"], job["generation"], job["playbook_text"],
                         Path(job["workdir"]), make_llm(job["mock"]))
    except Exception:  # noqa: BLE001  one broken run must not kill the generation
        return {**{k: job[k] for k in ("problem_id", "run_id", "generation")},
                "error": traceback.format_exc(), "stop": "error", "turns": 0, "cited": [],
                "trace": [], "meshes": [], "solves": [], "usage": []}


def run_generation(jobs: list[dict], concurrency: int, cost: CostLog) -> list[dict]:
    """Run jobs with at most `concurrency` in flight; stop starting new ones once over budget."""
    runs, pending, todo = [], set(), list(jobs)
    with ProcessPoolExecutor(concurrency, mp_context=mp.get_context("spawn")) as pool:
        while todo or pending:
            while todo and len(pending) < concurrency and not cost.over_budget():
                pending.add(pool.submit(_worker, todo.pop(0)))
            if not pending:
                break
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for f in done:
                run = f.result()
                cost.add(run["usage"])
                runs.append(run)
                status = "ERROR" if run.get("error") else f"{run['turns']} turns"
                print(f"  run {run['run_id']} done ({status}), spent ${cost.total:.2f}", flush=True)
    return runs


# --------------------------------------------------------------------------- summary


def summarize(rows: list[dict], gen_notes: list[dict], cost: CostLog) -> str:
    lines = ["# ACE run summary", "",
             "score = log10(e_uniform / e_agent): 0 = naive uniform mesh, +1 = 10x more accurate, -2 = failed.",
             "", "| gen | problem | split | mean score | min | max | failed | median rel. error |",
             "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['gen']} | {r['problem']} | {r['split']} | {r['mean']:+.2f} | {r['min']:+.2f} | "
                     f"{r['max']:+.2f} | {r['failed']}/{r['n']} | {r['median_err']} |")
    lines += ["", "| gen | playbook bullets used | added | voted | deduped | pruned | cumulative $ |",
              "|---|---|---|---|---|---|---|"]
    for g in gen_notes:
        lines.append(f"| {g['gen']} | {g['bullets']} | {g.get('added', '-')} | {g.get('voted', '-')} | "
                     f"{g.get('deduped', '-')} | {g.get('pruned', '-')} | {g['usd']:.2f} |")
    lines += ["", "Spend by role: " + ", ".join(f"{k} ${v:.2f}" for k, v in cost.by_role().items()), ""]
    return "\n".join(lines)


def plot_scores(rows: list[dict], path: Path) -> None:
    """Mean score per generation, one line per problem; held-out problems dashed."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
    fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
    ax.axhline(0, color="#999999", lw=1)
    ax.annotate("naive uniform mesh", (0, 0), xytext=(2, 3), textcoords="offset points",
                color="#666666", fontsize=8)
    for i, pid in enumerate(sorted({r["problem"] for r in rows})):
        pts = [r for r in rows if r["problem"] == pid]
        label = pid + (" (held out)" if pts[0]["split"] == "holdout" else "")
        ax.plot([r["gen"] for r in pts], [r["mean"] for r in pts], lw=2, marker="o", ms=8,
                color=colors[i % len(colors)], ls="--" if pts[0]["split"] == "holdout" else "-",
                label=label)
        ax.annotate(f"{pts[-1]['mean']:+.2f}", (pts[-1]["gen"], pts[-1]["mean"]), xytext=(8, -3),
                    textcoords="offset points", fontsize=8, color="#333333")
    ax.set_xlabel("generation (0 = empty playbook)")
    ax.set_ylabel("mean score (log10 error reduction)")
    ax.set_xticks(sorted({r["gen"] for r in rows}))
    ax.grid(axis="y", color="#e5e5e5", lw=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def score_rows(gen: int, runs: list[dict], checks: dict, holdout: list[str]) -> list[dict]:
    rows = []
    for pid in sorted({r["problem_id"] for r in runs}):
        scores = [checks[r["run_id"]]["score"] for r in runs if r["problem_id"] == pid]
        errs = [checks[r["run_id"]].get("rel_error") for r in runs if r["problem_id"] == pid]
        errs = [e for e in errs if e is not None]
        rows.append({
            "gen": gen, "problem": pid, "split": "holdout" if pid in holdout else "train",
            "n": len(scores), "mean": statistics.mean(scores), "min": min(scores), "max": max(scores),
            "failed": sum(checks[r["run_id"]]["failed_gate"] is not None for r in runs if r["problem_id"] == pid),
            "median_err": f"{statistics.median(errs):.2e}" if errs else "-",
        })
    return rows


# --------------------------------------------------------------------------- main loop


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--problems", default=",".join(config.TRAIN), help="train problems (reflected on)")
    ap.add_argument("--holdout", default=",".join(config.HOLDOUT), help="held-out problems (evaluated only)")
    ap.add_argument("--runs-per-problem", type=int, default=config.RUNS_PER_PROBLEM)
    ap.add_argument("--generations", type=int, default=config.GENERATIONS)
    ap.add_argument("--budget-usd", type=float, default=config.BUDGET_USD)
    ap.add_argument("--concurrency", type=int, default=config.CONCURRENCY)
    ap.add_argument("--top-n", type=int, default=config.TOP_N_BULLETS, help="bullets shown to the agent")
    ap.add_argument("--playbook", type=Path, help="start from this playbook JSON instead of empty")
    ap.add_argument("--out", type=Path, help="output dir (default ace_runs/<timestamp>)")
    ap.add_argument("--mock", action="store_true", help="canned LLM responses, no API calls")
    args = ap.parse_args(argv)

    train = [p for p in args.problems.split(",") if p]
    holdout = [p for p in args.holdout.split(",") if p]
    out = args.out or Path("ace_runs") / (time.strftime("%Y%m%d-%H%M%S") + ("-mock" if args.mock else ""))
    out.mkdir(parents=True, exist_ok=True)
    cost = CostLog(out / "cost.jsonl", args.budget_usd)
    llm = make_llm(args.mock)
    playbook = Playbook.load(args.playbook) if args.playbook else Playbook()
    calibration = load_calibration(train + holdout)
    rows, gen_notes = [], []
    print(f"output: {out}")

    try:
        for gen in range(args.generations):
            gdir = out / f"gen{gen}"
            (gdir / "runs").mkdir(parents=True, exist_ok=True)
            playbook.save(out / f"playbook_gen{gen}.json")
            text = playbook.render(args.top_n)
            print(f"\n=== generation {gen}: {len(playbook.bullets)} bullets, ${cost.total:.2f} spent")

            # 1. run all agents
            jobs = [{"problem_id": pid, "run_id": f"g{gen}-{pid}-{i}", "generation": gen,
                     "playbook_text": text, "workdir": str(gdir / "work" / f"{pid}-{i}"), "mock": args.mock}
                    for pid in train + holdout for i in range(args.runs_per_problem)]
            runs = run_generation(jobs, args.concurrency, cost)
            if runs and all(r.get("error") for r in runs):
                print(runs[0]["error"])
                raise SystemExit("every run failed; see the error above")

            # 2. checks
            checks = {r["run_id"]: check_run(r["problem_id"], r, calibration) for r in runs}
            for r in runs:
                (gdir / "runs" / f"{r['run_id']}.json").write_text(
                    json.dumps({**r, "checks": checks[r["run_id"]]}, indent=1, default=str))
            rows += score_rows(gen, runs, checks, holdout)
            note = {"gen": gen, "bullets": len(playbook.bullets)}
            gen_notes.append(note)

            if gen == args.generations - 1:
                note["usd"] = cost.total
                break                           # last generation: evaluate only

            # 3. reflect (train problems only)
            cost.check()
            reflections = []
            for pid in train:
                pruns = [r for r in runs if r["problem_id"] == pid]
                if not pruns:
                    continue
                refl, usage = reflect(pid, pruns, checks, playbook, llm)
                cost.add(usage)
                if refl is None:
                    print(f"  reflector output invalid twice for {pid}; skipped")
                    continue
                (gdir / f"reflection_{pid}.json").write_text(refl.model_dump_json(indent=2))
                reflections.append((pid, refl, [r["run_id"] for r in pruns]))
                cost.check()

            # 4. curate  5. merge
            ops, usage = curate(reflections, playbook, llm)
            cost.add(usage)
            changes = apply_ops(playbook, ops, generation=gen)
            save_generation(playbook, changes, out, gen + 1)
            note.update({k: len(v) for k, v in changes.items()}, usd=cost.total)
            print(f"  playbook: +{len(changes['added'])} added, {len(changes['voted'])} votes, "
                  f"{len(changes['deduped'])} deduped, {len(changes['pruned'])} pruned")
    except BudgetExceeded as e:
        print(f"\nBUDGET REACHED ({e}); stopping cleanly.")
    finally:
        for n in gen_notes:
            n.setdefault("usd", cost.total)
        (out / "summary.md").write_text(summarize(rows, gen_notes, cost))
        if rows:
            plot_scores(rows, out / "scores.png")
        print("\n" + (out / "summary.md").read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
