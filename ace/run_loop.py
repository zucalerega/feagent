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
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

from ace import config
from ace.agent import run_agent
from ace.checks import check_run, load_calibration
from ace.cost import BudgetExceeded, CostLog
from ace.curator import curate
from ace.llm import make_llm
from ace.merge import apply_ops, save_generation
from ace.playbook import Playbook
from ace.reflector import build_prompt, reflect


# --------------------------------------------------------------------------- one agent run


def _error_run(job: dict, error: str, usage: list) -> dict:
    return {**{k: job[k] for k in ("problem_id", "run_id", "generation")}, "error": error,
            "stop": "error", "turns": len(usage), "cited": [], "submitted": None, "trace": [],
            "meshes": [], "solves": [], "usage": usage}


def _worker(job: dict) -> dict:
    """Runs in a child process (gmsh is process-global, so one run per process at a time)."""
    usage: list = []   # filled call by call, so a crash halfway still reports its tokens
    try:
        return run_agent(job["problem_id"], job["run_id"], job["generation"], job["playbook_text"],
                         Path(job["workdir"]), make_llm(job["mock"]), usage=usage)
    except Exception:  # noqa: BLE001  one broken run must not kill the generation
        return _error_run(job, traceback.format_exc(), usage)


def run_generation(jobs: list[dict], concurrency: int, cost: CostLog) -> list[dict]:
    """Run jobs with at most `concurrency` in flight; stop starting new ones once over budget."""
    runs, pending, todo = [], {}, list(jobs)
    new_pool = lambda: ProcessPoolExecutor(concurrency, mp_context=mp.get_context("spawn"))  # noqa: E731
    pool = new_pool()
    try:
        while todo or pending:
            while todo and len(pending) < concurrency and not cost.over_budget():
                job = todo.pop(0)
                try:
                    pending[pool.submit(_worker, job)] = job
                except BrokenProcessPool:   # a worker died earlier: carry on with a fresh pool
                    pool.shutdown(wait=False, cancel_futures=True)
                    pool = new_pool()
                    pending[pool.submit(_worker, job)] = job
            if not pending:
                break
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for f in done:
                job = pending.pop(f)
                try:
                    run = f.result()
                except Exception as e:  # noqa: BLE001  the worker process itself died
                    # Its token usage died with it: it is missing from the cost log (flagged).
                    run = _error_run(job, f"worker process failed: {e!r} (token usage unknown)", [])
                cost.add(run["usage"])
                runs.append(run)
                status = "ERROR" if run.get("error") else f"{run['turns']} turns"
                print(f"  run {run['run_id']} done ({status}), spent ${cost.total:.2f}", flush=True)
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
    return runs


# --------------------------------------------------------------------------- summary


def summarize(rows: list[dict], gen_notes: list[dict], cost: CostLog) -> str:
    lines = ["# ACE run summary", "",
             "score = log10(e_uniform / e_agent): 0 = best uniform mesh at the node cap, +1 = 10x more accurate, -2 = failed.",
             "'failed' counts runs that failed a gate; 'errored' counts runs that crashed (API or code errors).",
             "", "| gen | problem | split | mean score | min | max | failed | errored | median rel. error |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['gen']} | {r['problem']} | {r['split']} | {r['mean']:+.2f} | {r['min']:+.2f} | "
                     f"{r['max']:+.2f} | {r['failed']}/{r['n']} | {r['errored']} | {r['median_err']} |")
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
    """Per-problem score statistics. Crashed runs are counted but left out of the scores: they
    say nothing about the agent."""
    rows = []
    for pid in sorted({r["problem_id"] for r in runs}):
        mine = [r for r in runs if r["problem_id"] == pid]
        ok = [r for r in mine if not r.get("error")]
        if not ok:
            continue
        scores = [checks[r["run_id"]]["score"] for r in ok]
        errs = [e for r in ok if (e := checks[r["run_id"]].get("rel_error")) is not None]
        rows.append({
            "gen": gen, "problem": pid, "split": "holdout" if pid in holdout else "train",
            "n": len(scores), "mean": statistics.mean(scores), "min": min(scores), "max": max(scores),
            "failed": sum(checks[r["run_id"]]["failed_gate"] is not None for r in ok),
            "errored": len(mine) - len(ok),
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
            # Interleaved (run 0 of every problem, then run 1, ...) so that if the budget stops a
            # generation early, every problem still has a similar number of runs.
            jobs = [{"problem_id": pid, "run_id": f"g{gen}-{pid}-{i}", "generation": gen,
                     "playbook_text": text, "workdir": str(gdir / "work" / f"{pid}-{i}"), "mock": args.mock}
                    for i in range(args.runs_per_problem) for pid in train + holdout]
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
                # Runs that crashed (API or code errors) say nothing about the agent: leave them out.
                pruns = [r for r in runs if r["problem_id"] == pid and not r.get("error")]
                if not pruns:
                    continue
                (gdir / f"reflection_prompt_{pid}.md").write_text(build_prompt(pid, pruns, checks, playbook))
                usage = []          # filled per API call, so billed attempts count even on an error
                try:
                    refl, _ = reflect(pid, pruns, checks, playbook, llm, records=usage)
                    if refl is None:
                        print(f"  reflector output invalid twice for {pid}; skipped")
                except Exception as e:  # noqa: BLE001  e.g. an API error: skip this problem
                    print(f"  reflector call failed for {pid}: {e!r}; skipped")
                    refl = None
                cost.add(usage)
                if refl is None:
                    continue
                (gdir / f"reflection_{pid}.json").write_text(refl.model_dump_json(indent=2))
                reflections.append((pid, refl, [r["run_id"] for r in pruns]))
                cost.check()

            # 4. curate  5. merge
            usage = []
            try:
                ops, _ = curate(reflections, playbook, llm, records=usage)
            except Exception as e:  # noqa: BLE001  keep the playbook unchanged this generation
                print(f"  curator call failed: {e!r}; no playbook update")
                ops = []
            cost.add(usage)
            (gdir / "curator_ops.json").write_text(json.dumps([vars(o) for o in ops], indent=1))
            changes = apply_ops(playbook, ops, generation=gen)
            (gdir / "merge_changes.json").write_text(json.dumps(changes, indent=1))
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
