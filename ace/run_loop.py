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
import os
import random
import statistics
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

from ace import config, ui_bridge
from ace.agent import run_agent
from ace.checkpoint import checkpoint
from ace.checks import check_run, score_population, tolerance
from ace.cost import BudgetExceeded, CostLog
from ace.curator import curate
from ace.llm import make_llm
from ace.memory import Memory, agent_of
from ace.merge import apply_ops, diff_markdown, save_generation
from ace.playbook import Playbook
from ace import stages
from ace.diagnoser import Diagnosis, diagnose_problem
from ace.mock import fake_rerun
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
                         Path(job["workdir"]), make_llm(job["mock"]), usage=usage, ui=job.get("ui"))
    except Exception:  # noqa: BLE001  one broken run must not kill the generation
        return _error_run(job, traceback.format_exc(), usage)


def run_generation(jobs: list[dict], concurrency: int, cost: CostLog, on_done=None) -> list[dict]:
    """Run jobs with at most `concurrency` in flight; stop starting new ones once over budget.
    `on_done(run)`, if given, is called here as each run finishes (e.g. by ace.ui_bridge)."""
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
                run.setdefault("playbook_text", job["playbook_text"])   # what the agent saw (dashboard)
                runs.append(run)
                if on_done is not None:
                    on_done(run)
                status = "ERROR" if run.get("error") else f"{run['turns']} turns"
                print(f"  run {run['run_id']} done ({status}), spent ${cost.total:.2f}", flush=True)
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
    return runs


# --------------------------------------------------------------------------- scoring


def _score_worker(run: dict) -> tuple:
    """check_run, oracles and stage report of one run (10-30 s each on 3D runs, hence a process)."""
    checks = check_run(run["problem_id"], run)
    if run.get("error"):
        return checks, None, None
    tol = tolerance(run["problem_id"])
    orc = stages.run_oracles(run, tol)
    return checks, orc, stages.stage_report(run, orc, tol)


def score_runs(runs: list[dict], concurrency: int) -> None:
    """Fill r["checks"] (and r["oracles"], r["stages"] for runs that did not crash), in parallel.
    A run whose worker fails (or every run, if the pool cannot start) is scored here instead."""
    results = [None] * len(runs)
    try:
        with ProcessPoolExecutor(concurrency, mp_context=mp.get_context("spawn")) as pool:
            futs = [pool.submit(_score_worker, r) for r in runs]
            for i, f in enumerate(futs):
                try:
                    results[i] = f.result()
                except Exception as e:  # noqa: BLE001
                    print(f"  scoring {runs[i]['run_id']} in a worker failed ({e!r}); scoring it here")
    except Exception as e:  # noqa: BLE001  the pool itself failed
        print(f"  scoring pool failed ({e!r}); scoring the rest here")
    for r, res in zip(runs, results):
        checks, orc, rep = res if res is not None else _score_worker(r)
        r["checks"] = checks
        if not r.get("error"):
            r["oracles"], r["stages"] = orc, rep


# --------------------------------------------------------------------------- diagnosis


def solver_log(out: Path, rec: dict) -> None:
    """Solver wall-clock time (oracle studies, counterfactual reruns), kept apart from token cost."""
    with (out / "solver.jsonl").open("a") as f:
        f.write(json.dumps(rec) + "\n")


def _teacher_worker(job: dict) -> dict:
    """One teacher (diagnoser) session in a child process (gmsh is process-global, so the reruns
    of different problems cannot share a process). The diagnosis comes back as a plain dict."""
    usage = []          # filled per API call, so billed calls count even if the session fails
    try:
        res = diagnose_problem(job["pid"], job["runs"], job["memory"], make_llm(job["mock"]), job["target_err"],
                               Path(job["workdir"]), rerun_fn=fake_rerun() if job["mock"] else None, usage=usage)
    except Exception as e:  # noqa: BLE001  e.g. an API error: skip this problem
        return {"error": repr(e), "usage": usage}
    diag = res["diagnosis"]
    return {**res, "diagnosis": diag.model_dump() if diag else None, "usage": usage, "error": None}


def diagnose_all(tasks: list[tuple], memory, gdir, out, mock, cost, gen, concurrency: int = 4) -> list:
    """One teacher session per training problem, all at once. tasks: [(pid, runs)]. Saves each
    diagnosis, its tool log and reruns; returns [(pid, Diagnosis)] for those that submitted one."""
    jobs = [{"pid": pid, "runs": pruns, "memory": memory, "target_err": tolerance(pid), "mock": mock,
             "workdir": str(gdir / "diagnoser" / pid)} for pid, pruns in tasks]
    with ProcessPoolExecutor(max(1, min(len(jobs), concurrency)), mp_context=mp.get_context("spawn")) as pool:
        futs = [pool.submit(_teacher_worker, j) for j in jobs]
        results = []
        for j, f in zip(jobs, futs):
            try:
                results.append(f.result())
            except Exception as e:  # noqa: BLE001  the worker process itself died
                results.append({"error": f"worker process failed: {e!r} (token usage unknown)", "usage": []})
    out_ = []
    for j, res in zip(jobs, results):     # in problem order, so the log reads as before
        pid = j["pid"]
        cost.add(res["usage"])
        if res["error"]:
            print(f"  diagnoser failed for {pid}: {res['error']}; skipped")
            continue
        diag = Diagnosis.model_validate(res["diagnosis"]) if res["diagnosis"] is not None else None
        (gdir / f"diagnosis_{pid}.json").write_text(json.dumps({
            "diagnosis": res["diagnosis"], "tool_calls": res["tool_calls"],
            "reruns": res["reruns"], "solver_seconds": res["solver_seconds"], "log": res["log"]}, indent=1, default=str))
        solver_log(out, {"gen": gen, "what": f"reruns {pid}", "seconds": res["solver_seconds"], "n": len(res["reruns"])})
        if diag is None:
            print(f"  diagnoser for {pid} did not submit a valid diagnosis; skipped")
            continue
        st = {}
        for rd in diag.runs:
            st[rd.diagnosis_status] = st.get(rd.diagnosis_status, 0) + 1
        print(f"  diagnosed {pid}: {res['tool_calls']} tool calls, {len(res['reruns'])} reruns, runs {st}, "
              f"{len(diag.lessons)} lessons" + (f", {len(diag.verification_notes)} claims corrected" if diag.verification_notes else ""))
        out_.append((pid, diag))
    return out_


# --------------------------------------------------------------------------- what an agent sees

# --personas: a fixed exploration style per agent. Without one, agents reading the same playbook
# converge on near-identical meshes, so the population stops exploring (and ties earn nothing).
PERSONAS = [
    "Grade strongly at singular or re-entrant corners and other stress concentrations: very small local "
    "sizes there, coarse elsewhere.",
    "Economy: reach the accuracy target with as few DOFs as possible; prefer the coarsest mesh that still "
    "beats the automesher.",
    "Refine every geometric feature (holes, fillets, corners, thin sections) moderately rather than any "
    "single one strongly.",
    "Refine along the loaded and supported boundaries, where the boundary conditions drive the stress gradients.",
    "Compare element orders: try linear and quadratic elements at similar DOFs and keep the better one.",
    "Run a mini convergence study within the solve budget: solve on two or three successively refined "
    "meshes and submit the one where the result has settled.",
]


def _fmt(x) -> str:
    return "-" if x is None else f"{x:.3g}" if isinstance(x, float) else str(x)


def peers_text(k: int, pid: str, prev: list[dict]) -> str:
    """--peers: what the other agents did on task `pid` last generation (prev = its runs)."""
    lines = []
    for r in sorted(prev, key=lambda r: r["run_id"]):
        j = agent_of(r["run_id"])
        if r["problem_id"] != pid or j is None or j == k or r.get("error"):
            continue
        c = r["checks"]
        req = (checkpoint(r) or {}).get("request")
        if req is None:
            mesh = "no successful solve"
        else:
            refs = "; ".join(f"{x.get('group') or x.get('point')} size {_fmt(x.get('size'))} radius {_fmt(x.get('radius'))}"
                             for x in req.get("refinements") or []) or "none"
            mesh = f"h {_fmt(req.get('h'))}, order {req.get('order')}, refinements: {refs}"
        lines.append(f"- a{j}: {mesh}; DOFs {_fmt(c.get('dofs'))}, field error {_fmt(c.get('rel_error'))}, "
                     f"base score {_fmt(c.get('base_score', c.get('score')))}")
    if not lines:
        return ""
    return ("\n\n## WHAT OTHER AGENTS DID LAST GENERATION ON THIS TASK\n" + "\n".join(lines)
            + "\nA tie with another agent on the hidden check points pays nothing, so try something "
              "different and better.")


# --------------------------------------------------------------------------- summary


def summarize(rows: list[dict], gen_notes: list[dict], cost: CostLog) -> str:
    if config.SCORE == "field":
        how = ["score = accuracy x efficiency against the automesher (gmsh default sizes, quadratic): error =",
               "max(whole-field energy-norm error, RMS von Mises error at the probe points); gain g = automesher",
               "error / error; accuracy = (g/2)^2 below the 2x target, g/2 above; efficiency = automesher DOFs /",
               "own DOFs. 1 = 2x the automesher's accuracy at its DOFs; the automesher scores 0.25. 'within tol'",
               "counts runs at or past the 2x target;"]
    else:
        how = ["score = accuracy x efficiency (the team's speed/accuracy metric), 0..1: 1 = the cheapest mesh within",
               "tolerance among all runs on that problem in this run. accuracy = min(1, tolerance / error);",
               "efficiency = cheapest in-tolerance DOFs / own DOFs. 'within tol' counts runs inside the tolerance;"]
    lines = ["# ACE run summary", "", *how,
             "'failed' counts runs that failed a gate; 'errored' counts runs that crashed (API or code errors).",
             "'base' is the mean score without the be-the-best bonus; 'bonus' the mean bonus (field mode).",
             "", "| gen | problem | split | mean score | base | bonus | min | max | mean accuracy | mean efficiency | within tol | median DOFs | median rel. error | failed | errored |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['gen']} | {r['problem']} | {r['split']} | {r['mean']:.2f} | {r['base']:.2f} | {r['bonus']:.2f} | "
                     f"{r['min']:.2f} | {r['max']:.2f} | "
                     f"{r['accuracy']:.2f} | {r['efficiency']:.2f} | {r['within_tol']}/{r['n']} | {r['median_dofs']} | "
                     f"{r['median_err']} | {r['failed']}/{r['n']} | {r['errored']} |")
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
    ax.set_ylim(0, max(1.05, max(r["max"] for r in rows) * 1.05))   # field-mode scores can exceed 1
    for i, pid in enumerate(sorted({r["problem"] for r in rows})):
        pts = [r for r in rows if r["problem"] == pid]
        label = pid + (" (held out)" if pts[0]["split"] == "holdout" else "")
        ax.plot([r["gen"] for r in pts], [r["mean"] for r in pts], lw=2, marker="o", ms=8,
                color=colors[i % len(colors)], ls="--" if pts[0]["split"] == "holdout" else "-",
                label=label)
        ax.annotate(f"{pts[-1]['mean']:.2f}", (pts[-1]["gen"], pts[-1]["mean"]), xytext=(8, -3),
                    textcoords="offset points", fontsize=8, color="#333333")
    ax.set_xlabel("generation (0 = empty playbook)")
    ax.set_ylabel("mean score (accuracy x efficiency)")
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
        bonus = [checks[r["run_id"]].get("best_bonus") or 0.0 for r in ok]
        errs = [e for r in ok if (e := checks[r["run_id"]].get("rel_error")) is not None]
        dofs = [d for r in ok if (d := checks[r["run_id"]].get("dofs"))]
        rows.append({
            "gen": gen, "problem": pid, "split": "holdout" if pid in holdout else "train",
            "n": len(scores), "mean": statistics.mean(scores), "min": min(scores), "max": max(scores),
            "base": statistics.mean(s - b for s, b in zip(scores, bonus)), "bonus": statistics.mean(bonus),
            "failed": sum(checks[r["run_id"]]["failed_gate"] is not None for r in ok),
            "errored": len(mine) - len(ok),
            "median_err": f"{statistics.median(errs):.2e}" if errs else "-",
            "accuracy": statistics.mean(checks[r["run_id"]].get("accuracy") or 0 for r in ok),
            "efficiency": statistics.mean(checks[r["run_id"]].get("efficiency") or 0 for r in ok),
            "within_tol": sum(bool(checks[r["run_id"]].get("meets_tol")) for r in ok),
            "median_dofs": int(statistics.median(dofs)) if dofs else "-",
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
    ap.add_argument("--memory", choices=["shared", "shared+private"], default=config.MEMORY,
                    help="one shared playbook, or shared + one private playbook per persistent agent")
    ap.add_argument("--agents", type=int, default=config.AGENTS, help="persistent agents (shared+private mode)")
    ap.add_argument("--split", action="store_true",
                    help="shared+private only: each generation, randomly split the agents into one group per "
                         "train problem; private notes follow the agent")
    ap.add_argument("--teacher", choices=["rerun", "inspect", "both"], default="rerun",
                    help="what the teacher may do: counterfactual reruns, a detailed inspection, or both")
    ap.add_argument("--shared-scope", choices=["all", "task"], default="all",
                    help="with --split: one shared playbook for all tasks, or one per task")
    ap.add_argument("--shared-bar", choices=["counterfactual", "any"], default="any",
                    help="shared+private: which teacher lessons may enter the shared playbook (see ace/memory.py)")
    ap.add_argument("--peers", action=argparse.BooleanOptionalAction, default=None,
                    help="show each agent what the other agents did on its task last generation (default: on with --split)")
    ap.add_argument("--personas", action="store_true", help="give each agent a fixed exploration style")
    ap.add_argument("--ui-runs-dir", type=Path, help="also record every solve and agent run in this feacore run "
                    "store, for the feacore web UI (default: $FEACORE_RUNS_DIR if set, else off)")
    args = ap.parse_args(argv)
    # Before any process pool exists: spawned children (agents, scoring, teacher) read it at import.
    os.environ["ACE_TEACHER"] = args.teacher
    config.TEACHER_MODE = args.teacher
    peers = args.split if args.peers is None else args.peers

    train = [p for p in args.problems.split(",") if p]
    holdout = [p for p in args.holdout.split(",") if p]
    out = args.out or Path("ace_runs") / (time.strftime("%Y%m%d-%H%M%S") + ("-mock" if args.mock else ""))
    out.mkdir(parents=True, exist_ok=True)
    cost = CostLog(out / "cost.jsonl", args.budget_usd)
    llm = make_llm(args.mock)
    playbook = Playbook.load(args.playbook) if args.playbook else Playbook()
    private = args.memory == "shared+private"
    if args.teacher != "rerun" and config.SCORE != "field":
        raise SystemExit("--teacher inspect/both needs field scoring (ACE_SCORE=field): its tools read the field error")
    if args.split and not (private and len(train) > 1):
        raise SystemExit("--split needs --memory shared+private and at least two --problems")
    if (peers or args.personas) and not private:
        raise SystemExit("--peers and --personas need --memory shared+private (persistent agents)")
    per_task = args.split and args.shared_scope == "task"     # one shared playbook per task
    mem = (Memory(args.agents, tasks=train) if per_task else Memory(args.agents, shared=playbook)) if private else None
    if config.SCORE == "field":     # build baselines once here, not concurrently in every worker
        from ace.field_error import automesher, probe_points
        for pid in train + holdout:
            automesher(pid), probe_points(pid)
    all_runs: dict[str, list[dict]] = {pid: [] for pid in train + holdout}   # every run so far, per problem
    rows, gen_notes = [], []
    print(f"output: {out}")
    ui = ui_bridge.settings(args.ui_runs_dir, loop_id=out.resolve().name)   # None = off

    def agent_text(k: int, pid: str | None, gen: int) -> str:
        """Agent k's prompt text on task pid (None: held out): memory, then peers and persona."""
        text = mem.prompt_text(k, args.top_n, pid)
        if peers and pid is not None:
            text += peers_text(k, pid, [r for r in all_runs[pid] if r["generation"] == gen - 1])
        if args.personas:
            text += "\n\n## YOUR EXPLORATION STYLE (yours alone)\n" + PERSONAS[k % len(PERSONAS)]
        return text

    try:
        for gen in range(args.generations):
            gdir = out / f"gen{gen}"
            (gdir / "runs").mkdir(parents=True, exist_ok=True)
            playbook.save(out / f"playbook_gen{gen}.json")
            if private:
                mem.save(out, gen)
            text = playbook.render(args.top_n)
            n_shared = sum(len(pb.bullets) for pb in mem.shared_by_task.values()) if mem else len(playbook.bullets)
            print(f"\n=== generation {gen}: {n_shared} shared bullets, ${cost.total:.2f} spent")

            # 1. run all agents
            # Interleaved (run 0 of every problem, then run 1, ...) so that if the budget stops a
            # generation early, every problem still has a similar number of runs.
            if private and args.split:   # agents shuffled into one group per task, every generation
                order = random.Random(gen).sample(range(args.agents), args.agents)
                group = {k: train[i % len(train)] for i, k in enumerate(order)}
                (gdir / "groups.json").write_text(json.dumps({f"a{k}": pid for k, pid in sorted(group.items())}))
                jobs = [{"problem_id": pid, "run_id": f"g{gen}-{pid}-a{k}", "generation": gen,
                         "playbook_text": agent_text(k, pid, gen), "workdir": str(gdir / "work" / f"{pid}-a{k}"),
                         "mock": args.mock} for k, pid in sorted(group.items())]
                jobs += [{"problem_id": pid, "run_id": f"g{gen}-{pid}-a{k}", "generation": gen,
                          "playbook_text": agent_text(k, None, gen), "workdir": str(gdir / "work" / f"{pid}-a{k}"),
                          "mock": args.mock} for k in range(args.agents) for pid in holdout]
            elif private:   # persistent agents a0..: each sees the shared playbook + its own notes
                jobs = [{"problem_id": pid, "run_id": f"g{gen}-{pid}-a{k}", "generation": gen,
                         "playbook_text": agent_text(k, pid if pid in train else None, gen),
                         "workdir": str(gdir / "work" / f"{pid}-a{k}"),
                         "mock": args.mock} for k in range(args.agents) for pid in train + holdout]
            else:
                jobs = [{"problem_id": pid, "run_id": f"g{gen}-{pid}-{i}", "generation": gen,
                         "playbook_text": text, "workdir": str(gdir / "work" / f"{pid}-{i}"), "mock": args.mock}
                        for i in range(args.runs_per_problem) for pid in train + holdout]
            jobs = [{**j, "ui": ui} for j in jobs]   # ace.ui_bridge settings (None = off)
            t0 = time.perf_counter()
            runs = run_generation(jobs, args.concurrency, cost, on_done=lambda r: ui_bridge.record_agent_run(ui, r))
            phase = {"gen": gen, "what": "phase", "agents": round(time.perf_counter() - t0, 2)}
            if runs and all(r.get("error") for r in runs):
                print(runs[0]["error"])
                raise SystemExit("every run failed; see the error above")

            # 2. checks: score, then oracles and stage invariants (layers 1 and 2 of diagnosis)
            t_oracles = time.perf_counter()
            score_runs(runs, args.concurrency)
            for r in runs:
                all_runs[r["problem_id"]].append(r)
            # Speed/accuracy score: each problem's runs scored against every run on it so far.
            for pid in all_runs:
                if all_runs[pid]:
                    score_population(all_runs[pid])
            ui_bridge.record_scores(ui, all_runs)   # every run so far was just (re)scored
            checks = {r["run_id"]: r["checks"] for r in runs}
            for r in runs:
                (gdir / "runs" / f"{r['run_id']}.json").write_text(json.dumps(r, indent=1, default=str))
            solver_log(out, {"gen": gen, "what": "oracles", "seconds": round(time.perf_counter() - t_oracles, 2)})
            phase["scoring"] = round(time.perf_counter() - t_oracles, 2)
            note = {"gen": gen, "bullets": sum(len(pb.bullets) for pb in mem.shared_by_task.values())
                    if mem else len(playbook.bullets)}
            gen_notes.append(note)

            if gen == args.generations - 1:
                note["usd"] = cost.total
                solver_log(out, phase)
                break                           # last generation: evaluate only

            # 3. reflect (train problems only)
            cost.check()
            t0 = time.perf_counter()
            reflections = []
            # Runs that crashed (API or code errors) say nothing about the agent: leave them out.
            by_task = {pid: [r for r in runs if r["problem_id"] == pid and not r.get("error")] for pid in train}
            by_task = {pid: pruns for pid, pruns in by_task.items() if pruns}
            if config.REFLECTOR_MODE == "diagnoser":   # one teacher session per problem, in parallel
                for pid, refl in diagnose_all(list(by_task.items()), mem or playbook, gdir, out, args.mock, cost, gen,
                                              args.concurrency):
                    ui_bridge.record_diagnosis(ui, by_task[pid], refl)
                    reflections.append((pid, refl, [r["run_id"] for r in by_task[pid]]))
                cost.check()
            else:
                for pid, pruns in by_task.items():
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
            phase["teacher"] = round(time.perf_counter() - t0, 2)

            # 4. curate  5. merge
            t0 = time.perf_counter()
            if private:
                usage = []
                try:
                    log = mem.update(reflections, llm, gen, records=usage, shared_bar=args.shared_bar)
                except Exception as e:  # noqa: BLE001  keep the memories unchanged this generation
                    print(f"  memory update failed: {e!r}; no change")
                    empty = {"added": [], "voted": [], "deduped": [], "pruned": []}
                    log = {"shared": {t: dict(empty) for t in train} if per_task else empty, "private": {}, "promoted": []}
                cost.add(usage)
                (gdir / "memory_changes.json").write_text(json.dumps(log, indent=1, default=str))
                mem.save(out, gen + 1)
                if per_task:   # one shared playbook per task: one diff section each
                    diff = "\n".join(f"# Task {t}\n" + diff_markdown(log["shared"][t], gen + 1, mem.shared_by_task[t])
                                     for t in train)
                    shared_log = {k: [x for t in train for x in log["shared"][t][k]] for k in log["shared"][train[0]]}
                else:
                    diff = diff_markdown(log["shared"], gen + 1, mem.shared)
                    shared_log = log["shared"]
                diff += "\n## Promoted from private notes\n" + ("\n".join(
                    f"- `{p['shared_id']}` (agents {p['from_agents']}): {p['content']}" for p in log["promoted"]) or "none")
                diff += "\n\n## Private notes added\n" + ("\n".join(
                    f"- agent a{k}: " + "; ".join(f"`{i}` {c}" for i, c in ch["added"])
                    for k, ch in sorted(log["private"].items()) if ch["added"]) or "none") + "\n"
                (out / f"diff_gen{gen + 1}.md").write_text(diff)
                n_priv = sum(len(c["added"]) for c in log["private"].values())
                note.update({k: len(v) for k, v in shared_log.items()}, usd=cost.total,
                            private_added=n_priv, promoted=len(log["promoted"]))
                print(f"  shared: +{len(shared_log['added'])} added, {len(shared_log['pruned'])} pruned; "
                      f"private: +{n_priv} added across agents; promoted to shared: {len(log['promoted'])}")
                solver_log(out, {**phase, "memory": round(time.perf_counter() - t0, 2)})
                continue
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
            solver_log(out, {**phase, "memory": round(time.perf_counter() - t0, 2)})
    except BudgetExceeded as e:
        print(f"\nBUDGET REACHED ({e}); stopping cleanly.")
    finally:
        for n in gen_notes:
            n.setdefault("usd", cost.total)
        # Final scores: every generation against the whole run's population, so they compare.
        rows = []
        for pid, prs in all_runs.items():
            if prs:
                score_population(prs)
                for r in prs:
                    (out / f"gen{r['generation']}" / "runs" / f"{r['run_id']}.json").write_text(
                        json.dumps(r, indent=1, default=str))
        for g in sorted({r["generation"] for prs in all_runs.values() for r in prs}):
            gruns = [r for prs in all_runs.values() for r in prs if r["generation"] == g]
            rows += score_rows(g, gruns, {r["run_id"]: r["checks"] for r in gruns}, holdout)
        (out / "summary.md").write_text(summarize(rows, gen_notes, cost))
        if rows:
            plot_scores(rows, out / "scores.png")
        print("\n" + (out / "summary.md").read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
