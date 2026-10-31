"""The single-agent baseline for the --deep swarm (ace/run_loop.py --deep): ONE big agent does every
training task, each generation, in ONE conversation, and learns through the same loop.

    ACE_SCORE=field python -m ace.single_loop --mock --generations 2            # free, end to end
    ACE_SCORE=field python -m ace.single_loop --problems cooks_membrane,nafems_le11_axisym \\
        --generations 20 --budget-usd 60 --out ace_runs/single1 --swarm-dir ace_runs/deep2

It is the swarm's moving baseline: run both in parallel on the same tasks, then
`python -m ace.compare <swarm dir> <single dir>` puts their scores side by side, against generations,
cumulative $ and wall time.

The same as the swarm:
  - the tools (ace.agent: inspect_geometry, mesh, solve, view, compare, submit), each now with a `task`
    argument; TaskTools passes each call, view images included, to that task's ace.agent.Tools;
  - the field-mode score (ace.checks.check_run + score_population against the automesher). The
    be-the-best bonus needs peers: a lone run "wins" every hidden point, so it is set to 0 here and
    `score` equals `base_score`, the number to compare with the swarm's base score;
  - learning: run reports (ace.run_report), the deep teacher, self-reflection and organizer with the
    same prompts, except that PROCESS_BRIEF is replaced by process_brief.SINGLE_BRIEF ("one agent on
    every task") everywhere; the same organizer inputs (run_loop.private_incoming / shared_incoming).
Different, by design:
  - one model (config role "big_generator": Opus, max effort) instead of SWARM_AGENTS Sonnet agents;
  - the budget of the whole swarm in one agent: per task, MAX_SOLVES x (swarm agents per task) solves,
    and MAX_TURNS x SWARM_AGENTS turns in all;
  - no peers: no highlights from others, no disagreement map (it needs two runs on a task).

Outputs (the --deep layout, so ace.run_report, ace.compare, the dashboard and analysis scripts read it):
  gen<g>/runs/g<g>-<task>-big.json   one run record per task, as the swarm's (checks, trace, meshes,
                                     solves, cited, playbook_text); the conversation's token usage is
                                     on the first task's record only (it is one conversation)
  gen<g>/context/big.md, reports/, teacher_<task>.json, reflections/, organizer/{big,shared}.json
  playbook_gen<g>.json, private/agent0_gen<g>.json   its two note lists (see `learn`)
  generations.jsonl, summary.md, scores.png, cost.jsonl, solver.jsonl
"""

from __future__ import annotations

import argparse
import copy
import json
import multiprocessing as mp
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from ace import config, process_brief
from ace.agent import TOOLS, Tools, parse_citations, system_prompt, task_prompt, tool_loop
from ace.checks import score_population
from ace.cost import BudgetExceeded, CostLog
from ace.llm import make_llm
from ace.memory import Memory, notes_as_dicts
from ace.run_loop import (_dump, _first_text, _llm_jobs, _run_summary, plot_scores, private_incoming,
                          score_runs, shared_incoming, solver_log)

SWARM_AGENTS = 8      # the swarm this baseline is matched to (ace_runs/run_deep.sh: --agents 8, two tasks)
AGENT = "big"         # its name in run ids (g<g>-<task>-big) and output files
MAX_NUDGES = 5        # "continue" messages per conversation when it stops with tasks unsubmitted (see _big_worker)


# --------------------------------------------------------------------------- one generation's run


def task_tool_specs(tasks: list[str]) -> list[dict]:
    """ace.agent.TOOLS, each with a required `task` argument (which task the call is for)."""
    specs = copy.deepcopy(TOOLS)
    for t in specs:
        t["input_schema"]["properties"]["task"] = {"type": "string", "enum": list(tasks),
                                                   "description": "the problem id this call is for"}
        t["input_schema"]["required"] = ["task"] + t["input_schema"].get("required", [])
    return specs


class TaskTools:
    """One ace.agent.Tools per task (its own meshes, solves and submission); each call is routed by
    its `task` argument. A submitted task is closed, as a swarm run ends at its submit."""

    def __init__(self, tasks: list[str], gdir: Path, max_solves: int):
        self.tools = {t: Tools(t, gdir / "work" / f"{t}-{AGENT}", max_solves=max_solves) for t in tasks}

    def call(self, name: str, args: dict) -> tuple[dict, bool]:
        args = dict(args)
        task = args.pop("task", None)
        if task not in self.tools:
            return {"error": f"unknown task {task!r}; the tasks are {list(self.tools)}"}, True
        if self.tools[task].submitted is not None:
            return {"error": f"task {task} is already submitted; work on the others"}, True
        return self.tools[task].call(name, args)

    def finished(self) -> bool:
        return all(t.submitted is not None for t in self.tools.values())


def first_message(tasks: list[str], max_solves: int) -> str:
    head = (f"You have {len(tasks)} tasks, below. Work on all of them in this conversation, in any order. Every "
            f"tool call takes a `task` argument (the problem id) saying which task it is for; each task has its "
            f"own geometry, meshes, solves and submission, and its own budget of {max_solves} solves. Submitting "
            f"a task ends that task (as 'ends the run' above); the run ends when every task is submitted.")
    return head + "".join(f"\n\n=== TASK {i + 1} of {len(tasks)} ===\n{task_prompt(t, max_solves=max_solves, bonus=False)}"
                          for i, t in enumerate(tasks))


def split_records(tasks: list[str], gen: int, trace: list[dict], tools: TaskTools, usage: list, stop: str,
                  error: str | None, nudges: int = 0) -> list[dict]:
    """One run record per task, in the swarm's format (ace.agent.run_agent). A task's trace holds its
    own tool calls and results, and the text/thinking of the turns that worked on it (or on no task).
    A crash marks only the tasks not yet submitted as errors: a submitted task is scored, as a swarm
    run that ended at its submit is never touched by a later crash elsewhere."""
    task_of = {t["id"]: t["input"].get("task") for t in trace if t["type"] == "tool_call"}
    turn_tasks: dict[int, set] = {}
    for t in trace:
        if t["type"] == "tool_call":
            turn_tasks.setdefault(t["turn"], set()).add(task_of[t["id"]])

    def belongs(t: dict, task: str) -> bool:
        if t["type"] in ("tool_call", "tool_result"):
            return task_of.get(t["id"]) == task
        return not turn_tasks.get(t["turn"]) or task in turn_tasks[t["turn"]]

    recs = []
    for i, task in enumerate(tasks):
        mine = [t for t in trace if belongs(t, task)]
        tl = tools.tools[task]
        rec = {"problem_id": task, "run_id": f"g{gen}-{task}-{AGENT}", "generation": gen,
               "stop": "submitted" if tl.submitted is not None else stop, "turns": len(usage),
               "cited": parse_citations(mine), "submitted": tl.submitted, "trace": mine,
               "meshes": tl.meshes, "solves": tl.solves,
               "usage": usage if i == 0 else [],     # one conversation: counted once
               "nudges": nudges}                     # "continue" messages sent (the same on every task)
        if error and tl.submitted is None:
            rec["error"] = error
        elif error:
            rec["conversation_error"] = error         # kept visible, but this task's run stands
        recs.append(rec)
    return recs


def _big_worker(job: dict) -> list[dict]:
    """The whole conversation, in a child process (gmsh is process-global, as in the swarm's workers)."""
    tasks, gen = job["tasks"], job["generation"]
    tools = TaskTools(tasks, Path(job["gdir"]), job["max_solves"])
    usage: list = []      # filled call by call, so a crash halfway still reports its tokens
    trace, stop, error = [], "error", None
    nudges = [0]
    # Time-matched with the swarm: the conversation stops once its wall-clock budget is spent (checked
    # after each turn; a turn already in flight finishes). Unsubmitted tasks are then scored on their
    # last successful solve, the same rule as for a swarm run that never submits.
    deadline = time.monotonic() + job["seconds"] if job.get("seconds") else None
    out_of_time = lambda: deadline is not None and time.monotonic() > deadline  # noqa: E731

    def on_stop(reason: str) -> str | None:
        """One turn without a tool call (a text-only status line, or a turn spent on thinking up to
        max_tokens) would end the conversation for every task, where in the swarm it costs one run.
        So remind it of the open tasks, at most MAX_NUDGES times."""
        left = [t for t, tl in tools.tools.items() if tl.submitted is None]
        if not left or nudges[0] >= MAX_NUDGES or out_of_time():
            return None
        nudges[0] += 1
        return (f"Not submitted yet: {', '.join(left)}. Continue working on them with tool calls "
                f"(each tool call names its task).")

    try:
        trace, stop = tool_loop(make_llm(job["mock"]), system_prompt(job["context"]),
                                first_message(tasks, job["max_solves"]), task_tool_specs(tasks), tools.call,
                                lambda: tools.finished() or out_of_time(), job["max_turns"], f"g{gen}-{AGENT}", usage, role="big_generator",
                                on_stop=on_stop)
    except Exception:  # noqa: BLE001  as the swarm: a crashed run is recorded, not scored
        error = traceback.format_exc()
    return split_records(tasks, gen, trace, tools, usage, stop, error, nudges[0])


def swarm_agent_seconds(swarm_dir: Path | None, gen: int) -> float | None:
    """How long the swarm's agents took in generation `gen` (its solver.jsonl phase line), or in its
    latest finished generation if `gen` is not done yet; None without a swarm run."""
    f = Path(swarm_dir) / "solver.jsonl" if swarm_dir else None
    if f is None or not f.exists():
        return None
    phases = [json.loads(l) for l in f.read_text().splitlines() if '"phase"' in l]
    phases = [p for p in phases if p.get("what") == "phase" and p.get("agents")]
    same = [p["agents"] for p in phases if p["gen"] == gen]
    return same[0] if same else (phases[-1]["agents"] if phases else None)


def run_big(job: dict) -> list[dict]:
    """The single agent works on its tasks IN PARALLEL: one conversation per task, all started at once,
    all with the same context (its one memory) and each with that task's share of the turn budget.
    Records come back in task order, as before."""
    tasks = job["tasks"]
    per_task = [dict(job, tasks=[t], max_turns=max(1, job["max_turns"] // len(tasks))) for t in tasks]
    out = []
    with ProcessPoolExecutor(len(tasks), mp_context=mp.get_context("spawn")) as pool:
        futures = [pool.submit(_big_worker, j) for j in per_task]
        for t, f in zip(tasks, futures):
            try:
                out += f.result()
            except Exception as e:  # noqa: BLE001  the worker process itself died (its usage is lost)
                out.append({"problem_id": t, "run_id": f"g{job['generation']}-{t}-{AGENT}", "generation": job["generation"],
                            "error": f"worker process failed: {e!r} (token usage unknown)", "stop": "error", "turns": 0,
                            "cited": [], "submitted": None, "trace": [], "meshes": [], "solves": [], "usage": []})
    return out


def rescore(prs: list[dict]) -> None:
    """score_population over one task's runs (all generations), without the be-the-best bonus (a lone
    run would win every hidden point): score = base_score, as comparable with the swarm's base score."""
    score_population(prs)
    for r in prs:
        c = r["checks"]
        if "base_score" in c:
            c["score"], c["best_bonus"] = c["base_score"], 0.0


# --------------------------------------------------------------------------- learning


class SingleBrief:
    """The LLM with process_brief.PROCESS_BRIEF replaced by SINGLE_BRIEF in every system prompt, so the
    teacher, self-reflection and organizer (whose prompts embed the swarm's brief) read the single-agent
    version; and the swarm's solve budget (config.MAX_SOLVES) replaced by this agent's, so the critique
    of its solve use is pitched for the budget it had. Nothing else in their prompts changes."""

    def __init__(self, llm, max_solves: int = config.MAX_SOLVES):
        self.llm = llm
        n = config.MAX_SOLVES
        self.swaps = [(process_brief.PROCESS_BRIEF, process_brief.SINGLE_BRIEF),
                      (f"called solve(setup) at most {n} times", f"called solve(setup) at most {max_solves} times per task"),
                      (f"(you had {n} solves)", f"(you had {max_solves} solves on this task)")]

    def _swap(self, system: str) -> str:
        for old, new in self.swaps:
            system = system.replace(old, new)
        return system

    def structured(self, role, system, user, schema, records=None):
        return self.llm.structured(role, self._swap(system), user, schema, records)

    def warm_cache(self, role, system, *args, **kwargs):
        return self.llm.warm_cache(role, self._swap(system), *args, **kwargs)


def learn(gen: int, gdir: Path, runs: list[dict], reports: dict, mem: Memory, llm, cost: CostLog,
          state: dict) -> dict:
    """run_loop.deep_learn for the one agent. The teacher reviews each task (its one run there, no
    peers) and the agent reflects on each run, all in parallel; then the organizer rewrites its notes.

    Memory: two lists, as a swarm agent has. mem.private[0] = its notes (the teacher's notes and next
    mesh calls for it, its own reflections); mem.shared = its playbook of task-level lessons (the
    teacher's shared lessons, highlights, setup warnings). With one agent "shared" has a single
    reader, but keeping both lists gives it the same note capacity and the same organizer calls as a
    swarm agent, so the comparison differs only in the population.
    `state` ({"teacher", "feedback", "reflection", "last_run"}, each {task: dict}) is updated in place
    for the next generation's context."""
    from ace.deep_teacher import teach
    from ace.organizer import organize_private, organize_shared
    from ace.self_reflect import reflect_on_run

    ok = [r for r in runs if not r.get("error") and "error" not in reports.get(r["run_id"], {"error": "no report"})]
    notes_text = process_brief.render_notes(mem.private[0])
    jobs = {}
    for r in ok:
        pid = r["problem_id"]
        last = {AGENT: {"previous_run": state["last_run"].get(pid), "previous_feedback": state["feedback"].get(pid),
                        "previous_reflection": state["reflection"].get(pid), "promised_this_run": _first_text(r)}}
        jobs[f"teacher {pid}"] = (teach, dict(
            task=pid, reports=[reports[r["run_id"]]], runs=[r], notes_by_agent={AGENT: notes_text},
            shared_text=process_brief.render_notes(mem.shared), prev_feedback=None, disagreement=None,
            last_by_agent=last, llm=llm))
        jobs[f"reflect {r['run_id']}"] = (reflect_on_run, dict(report=reports[r["run_id"]], run=r,
                                                               own_notes_text=notes_text, llm=llm))
    t0 = time.perf_counter()
    res, _ = _llm_jobs(jobs, cost)
    timings = {"teacher+reflect": round(time.perf_counter() - t0, 2)}

    (gdir / "reflections").mkdir(exist_ok=True)
    private_in, shared_in = [], []
    for r in ok:
        pid = r["problem_id"]
        fb, refl = _dump(res.get(f"teacher {pid}")), _dump(res.get(f"reflect {r['run_id']}"))
        af = None
        if fb is not None:
            (gdir / f"teacher_{pid}.json").write_text(json.dumps(fb, indent=1))
            state["teacher"][pid] = fb
            shared_in += shared_incoming(pid, fb)
            if fb.get("per_agent"):
                af = {**fb["per_agent"][0], "agent": AGENT, "task": pid}
                state["feedback"][pid] = af
        if refl is not None:
            (gdir / "reflections" / f"{r['run_id']}.json").write_text(json.dumps(refl, indent=1))
            refl = {**refl, "run_id": r["run_id"], "task": pid}
            state["reflection"][pid] = refl
        private_in += private_incoming(af, refl)
        state["last_run"][pid] = _run_summary(r, reports.get(r["run_id"]))
    cost.check()

    t0 = time.perf_counter()
    owners = {AGENT: (mem.private[0], private_in), "shared": (mem.shared, shared_in)}
    jobs = {f"organize {AGENT}": (organize_private, dict(agent=AGENT, current=notes_as_dicts(mem.private[0]),
                                                         incoming=private_in, llm=llm))}
    if shared_in:
        jobs["organize shared"] = (organize_shared, dict(current=notes_as_dicts(mem.shared), incoming=shared_in, llm=llm))
    res, _ = _llm_jobs(jobs, cost)
    (gdir / "organizer").mkdir(exist_ok=True)
    for owner, (pb, incoming) in owners.items():
        if f"organize {owner}" not in jobs:
            continue
        before, after = notes_as_dicts(pb), res.get(f"organize {owner}")
        rec = {"before": before, "incoming": incoming, "after": None, "changes": None}
        if after is None:
            rec["error"] = "organizer gave no valid output; notes kept unchanged"
        else:
            rec.update(changes=Memory.replace_notes(pb, after, gen), after=notes_as_dicts(pb))
        (gdir / "organizer" / f"{owner}.json").write_text(json.dumps(rec, indent=1))
    timings["organize"] = round(time.perf_counter() - t0, 2)
    print(f"  notes: {len(mem.private[0].bullets)} own, {len(mem.shared.bullets)} playbook")
    cost.check()
    return timings


# --------------------------------------------------------------------------- summary


def summarize(rows: list[dict]) -> str:
    lines = ["# Single-agent baseline summary", "",
             "One big agent (config role big_generator) on every task each generation, in one conversation.",
             "'base score' is the field-mode score without the be-the-best bonus (a lone agent has no peers to",
             "beat), the number to compare with the swarm's base score. $ and wall time are per generation,",
             "including learning (teacher, reflection, organizer). Compare with a swarm run:",
             "python -m ace.compare <swarm dir> <this dir>", "",
             "| gen | task | base score | field error | probe error | DOFs | solves | failed gate | $ | cumulative $ | wall s |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    f = lambda x: "-" if x is None else f"{x:.3g}" if isinstance(x, float) else str(x)   # noqa: E731
    for r in rows:
        lines.append(f"| {r['gen']} | {r['task']} | {f(r['base'])} | {f(r['field_error'])} | {f(r['probe_error'])} | "
                     f"{f(r['dofs'])} | {r['solves']} | {r['failed'] or '-'} | {r['usd']:.2f} | {r['cum_usd']:.2f} | "
                     f"{r['seconds']:.0f} |")
    return "\n".join(lines) + "\n"


def write_summary(out: Path, all_runs: dict, gens: list[dict]) -> None:
    by_gen = {g["gen"]: g for g in gens}
    rows = []
    for pid, prs in all_runs.items():
        for r in prs:
            c, g = r.get("checks") or {}, by_gen.get(r["generation"], {})
            rows.append({"gen": r["generation"], "task": pid, "base": None if r.get("error") else c.get("base_score", c.get("score")),
                         "field_error": c.get("field_error"), "probe_error": c.get("probe_error"), "dofs": c.get("dofs"),
                         "solves": len(r.get("solves") or []), "failed": r.get("error") and "crashed" or c.get("failed_gate"),
                         "usd": g.get("usd", 0.0), "cum_usd": g.get("cum_usd", 0.0), "seconds": g.get("seconds", 0.0)})
    rows.sort(key=lambda r: (r["gen"], r["task"]))
    (out / "summary.md").write_text(summarize(rows))
    pts = [{"gen": r["gen"], "problem": r["task"], "split": "train", "mean": r["base"], "max": r["base"]}
           for r in rows if r["base"] is not None]
    if pts:
        plot_scores(pts, out / "scores.png")     # the swarm's plot; "mean" here = the one run's base score


# --------------------------------------------------------------------------- main loop


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--problems", default="cooks_membrane,nafems_le11_axisym", help="tasks, all done every generation")
    ap.add_argument("--generations", type=int, default=config.GENERATIONS)
    ap.add_argument("--budget-usd", type=float, default=config.BUDGET_USD)
    ap.add_argument("--swarm-agents", type=int, default=SWARM_AGENTS,
                    help="size of the swarm to match: sets the default solve and turn budgets below")
    ap.add_argument("--solves-per-task", type=int,
                    help="solve budget per task (default MAX_SOLVES x swarm agents per task, e.g. 4 x 4 = 16)")
    ap.add_argument("--turns", type=int, help="model calls per generation (default MAX_TURNS x swarm agents, e.g. 200)")
    ap.add_argument("--concurrency", type=int, default=config.CONCURRENCY, help="scoring processes")
    ap.add_argument("--out", type=Path, help="output dir (default ace_runs/single-<timestamp>)")
    ap.add_argument("--mock", action="store_true", help="canned LLM responses, no API calls")
    ap.add_argument("--gen-seconds", type=float, default=60.0,
                    help="wall-clock budget of the big agent per generation when the swarm's own time for "
                         "that generation is not known yet (see --swarm-dir)")
    ap.add_argument("--swarm-dir", type=Path, help="a swarm run dir (run_loop --deep) to compare $ with each "
                                                   "generation, and with ace.compare at the end")
    args = ap.parse_args(argv)
    if config.SCORE != "field":
        raise SystemExit("ace.single_loop needs field scoring (ACE_SCORE=field), as run_loop --deep does")
    os.environ["ACE_SKIP_CONVERGENCE"] = "1"    # before any pool: as --deep (the run report replaces it)

    tasks = [p for p in args.problems.split(",") if p]
    solves = args.solves_per_task or config.MAX_SOLVES * max(1, args.swarm_agents // len(tasks))
    turns = args.turns or config.MAX_TURNS * args.swarm_agents
    out = args.out or Path("ace_runs") / ("single-" + time.strftime("%Y%m%d-%H%M%S") + ("-mock" if args.mock else ""))
    out.mkdir(parents=True, exist_ok=True)
    cost = CostLog(out / "cost.jsonl", args.budget_usd)
    llm = SingleBrief(make_llm(args.mock), solves)
    mem = Memory(1)
    from ace.field_error import automesher, probe_points
    for pid in tasks:       # build the baselines once here, not in every worker
        automesher(pid), probe_points(pid)
    from ace.run_report import add_scores
    from ace.compare import gen_costs

    all_runs: dict[str, list[dict]] = {pid: [] for pid in tasks}
    state = {"teacher": {}, "feedback": {}, "reflection": {}, "last_run": {}}
    gens: list[dict] = []
    print(f"output: {out}\nbig agent: {config.MODELS['big_generator']} (effort {config.EFFORT['big_generator']}), "
          f"{solves} solves per task, {turns} turns, tasks {tasks}")
    try:
        for gen in range(args.generations):
            if (out / "STOP").exists():
                print(f"\nSTOP file found in {out}; stopping before generation {gen}.")
                break
            cost.check()
            t_gen, usd0 = time.perf_counter(), cost.total
            gdir = out / f"gen{gen}"
            (gdir / "runs").mkdir(parents=True, exist_ok=True)
            (gdir / "context").mkdir(exist_ok=True)
            mem.save(out, gen)
            print(f"\n=== generation {gen}: ${cost.total:.2f} spent")

            # 1. the big agent, every task in one conversation
            context = process_brief.single_context(tasks, mem.shared, mem.private[0],
                                                   {t: {"last_run": state["last_run"].get(t), "feedback": state["feedback"].get(t),
                                                        "reflection": state["reflection"].get(t),
                                                        "task_feedback": state["teacher"].get(t)} for t in tasks})
            (gdir / "context" / f"{AGENT}.md").write_text(context)
            seconds = swarm_agent_seconds(args.swarm_dir, gen) or args.gen_seconds
            print(f"  time budget: {seconds:.0f} s (the swarm's agents phase)")
            t0 = time.perf_counter()
            runs = run_big({"tasks": tasks, "generation": gen, "gdir": str(gdir), "context": context,
                            "max_solves": solves, "max_turns": turns, "mock": args.mock, "seconds": seconds})
            phase = {"gen": gen, "what": "phase", "agents": round(time.perf_counter() - t0, 2)}
            for r in runs:                       # one conversation per task: each record carries its own usage
                cost.add(r["usage"])
            for r in runs:
                r["playbook_text"] = context
            err = next((r.get("error") or r.get("conversation_error") for r in runs
                        if r.get("error") or r.get("conversation_error")), None)
            print(f"  big agent: turns {[r['turns'] for r in runs]}, solves {[len(r['solves']) for r in runs]}, "
                  f"stops {[r['stop'] for r in runs]}, nudges {runs[0].get('nudges', 0)}, spent ${cost.total:.2f}"
                  + (" (ERROR)" if err else ""))
            if err:
                print(err)

            # 2. score (each task against all its runs so far, no bonus) and run reports
            t0 = time.perf_counter()
            reports = score_runs(runs, args.concurrency, gdir / "reports")
            for r in runs:
                all_runs[r["problem_id"]].append(r)
            for prs in all_runs.values():
                rescore(prs)
            for r in runs:
                rep = reports.get(r["run_id"])
                if rep is not None and "error" not in rep and not r.get("error"):
                    reports[r["run_id"]] = add_scores(rep, {**r, "checks": {**r["checks"], "rank": 1, "group_size": 1}})
                    (gdir / "reports" / f"{r['run_id']}.json").write_text(json.dumps(reports[r["run_id"]], indent=1, default=str))
                (gdir / "runs" / f"{r['run_id']}.json").write_text(json.dumps(r, indent=1, default=str))
                c = r["checks"]
                print(f"  {r['problem_id']}: base score {c.get('base_score', c.get('score'))}, DOFs {c.get('dofs')}"
                      + (f", failed gate: {c['failed_gate']}" if c.get("failed_gate") else ""))
            phase["scoring"] = round(time.perf_counter() - t0, 2)

            # 3. learn (not after the last generation, which is only evaluated)
            if gen < args.generations - 1:
                phase.update(learn(gen, gdir, runs, reports, mem, llm, cost, state))
            solver_log(out, phase)
            g = {"gen": gen, "seconds": round(time.perf_counter() - t_gen, 1), "usd": round(cost.total - usd0, 4),
                 "cum_usd": round(cost.total, 4)}
            gens.append(g)
            with (out / "generations.jsonl").open("a") as f:
                f.write(json.dumps(g) + "\n")
            line = f"  generation {gen}: ${g['usd']:.2f} ({g['seconds']:.0f} s), cumulative ${g['cum_usd']:.2f}"
            if args.swarm_dir is not None and (args.swarm_dir / "cost.jsonl").exists():
                sw = gen_costs(args.swarm_dir).get(gen)
                line += (f"; swarm cumulative after generation {gen}: ${sw['cum_usd']:.2f}" if sw
                         else f"; swarm has not reached generation {gen} yet")
            print(line)
    except BudgetExceeded as e:
        print(f"\nBUDGET REACHED ({e}); stopping cleanly.")
    finally:
        for prs in all_runs.values():           # final scores, every generation against the whole run
            if prs:
                rescore(prs)
                for r in prs:
                    (out / f"gen{r['generation']}" / "runs" / f"{r['run_id']}.json").write_text(
                        json.dumps(r, indent=1, default=str))
        write_summary(out, all_runs, gens)
        print("\n" + (out / "summary.md").read_text())
        if args.swarm_dir is not None and args.swarm_dir.exists():
            from ace.compare import write_comparison
            print(write_comparison(args.swarm_dir, out, out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
