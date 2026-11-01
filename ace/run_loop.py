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

--deep replaces steps 3-5 (see deep_learn): every run also gets a run report (ace.run_report: its
whole mesh, where its error is, images); a teacher per task (ace.deep_teacher) and a self-reflection
per run (ace.self_reflect) read them in parallel; an organizer (ace.organizer) then rewrites each
agent's notes and the shared playbook, and each agent's next context is built by
ace.process_brief.deep_context (feedback on its last run, peers' highlights, all its notes).

--init-from <exported swarm> --init-agents a6,a3,a1,a7 starts from a trained swarm (ace.export_swarm):
its shared playbook, and the chosen agents' private notes, personas and last feedback, renamed
a0, a1, ... in that order (the mapping is in <out>/init.json). --pre-teach (with --deep) has the
teacher revise all of it for the new task before generation 0 (ace.pre_teach; <out>/pre_teach/).

Tasks scored by their nonlinear force curve against a physical test (config.TASKS[pid]["score"] ==
"nonlinear", ace.nonlinear) need no field reference: --deep works on them, the scoring worker skips
the linear oracles and builds a light run report from the curve check (run_report.build_nl_report).
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
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

from ace import config, evidence, process_brief, ui_bridge
from ace.agent import run_agent
from ace.checkpoint import checkpoint
from ace.checks import check_run, score_population, tolerance
from ace.deep_teacher import is_nonlinear, run_error
from ace.cost import BudgetExceeded, CostLog
from ace.curator import curate
from ace.llm import make_llm
from ace.memory import Memory, agent_of, notes_as_dicts, renamed
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


def _score_worker(run: dict, report_dir: str | None = None) -> tuple:
    """check_run, oracles and stage report of one run (10-30 s each on 3D runs, hence a process).
    With report_dir (--deep) also its run report (ace.run_report, written there), else None.
    check_run and the report share one solve of the run and one location of its points (shared_solves)."""
    from ace.field_error import shared_solves
    with shared_solves():
        checks = check_run(run["problem_id"], run)
        if run.get("error"):
            return checks, None, None, None
        if is_nonlinear(run["problem_id"]):
            # Scored by the nonlinear curve (check_run ran the ramp): the linear oracles and the field
            # report do not apply; the report is the curve check with its pictures.
            report = None
            if report_dir is not None:
                from ace.run_report import build_nl_report
                report = build_nl_report({**run, "checks": checks}, Path(report_dir))
            return checks, None, None, report
        tol = tolerance(run["problem_id"])
        orc = stages.run_oracles(run, tol)
        report = None
        if report_dir is not None:
            from ace.run_report import build_run_report   # --deep only
            report = build_run_report(run, Path(report_dir))
    return checks, orc, stages.stage_report(run, orc, tol), report


class Scoring:
    """The scoring pool of one generation. start(run) as each agent finishes, so scoring overlaps the
    slower agents instead of waiting for the last one; finish(runs) then waits for all of them and
    fills the runs exactly as scoring them all at the end would. A run whose worker fails (or every
    run, if the pool cannot start) is scored in this process instead."""

    def __init__(self, concurrency: int, report_dir: Path | None = None):
        self.rdir = str(report_dir) if report_dir is not None else None
        self.futs = {}
        try:
            self.pool = ProcessPoolExecutor(concurrency, mp_context=mp.get_context("spawn"))
        except Exception as e:  # noqa: BLE001
            print(f"  scoring pool failed ({e!r}); scoring here")
            self.pool = None

    def start(self, run: dict) -> None:
        if self.pool is None or run["run_id"] in self.futs:
            return
        try:
            self.futs[run["run_id"]] = self.pool.submit(_score_worker, run, self.rdir)
        except Exception as e:  # noqa: BLE001  the pool itself broke: the rest are scored here
            print(f"  scoring pool failed ({e!r}); scoring the rest here")
            self.pool = None

    def finish(self, runs: list[dict]) -> dict:
        """Fill r["checks"] (and r["oracles"], r["stages"] for runs that did not crash).
        Returns {run_id: run report} for the runs that got one (only with report_dir, i.e. --deep)."""
        for r in runs:
            self.start(r)                          # any run not started yet
        results = {}
        for r in runs:
            f = self.futs.get(r["run_id"])
            try:
                results[r["run_id"]] = f.result() if f is not None else None
            except Exception as e:  # noqa: BLE001
                print(f"  scoring {r['run_id']} in a worker failed ({e!r}); scoring it here")
        if self.pool is not None:
            self.pool.shutdown()
        reports = {}
        for r in runs:
            res = results.get(r["run_id"])
            checks, orc, rep, report = res if res is not None else _score_worker(r, self.rdir)
            r["checks"] = checks
            if not r.get("error"):
                r["oracles"], r["stages"] = orc, rep
            if report is not None:
                reports[r["run_id"]] = report
        return reports


def score_runs(runs: list[dict], concurrency: int, report_dir: Path | None = None) -> dict:
    """Score runs that are all done already (see Scoring)."""
    return Scoring(concurrency, report_dir).finish(runs)


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


# --------------------------------------------------------------------------- --deep: teacher, reflection, organizer


def _llm_jobs(jobs: dict, cost: CostLog, then=None) -> tuple[dict, dict]:
    """Run {name: (fn, kwargs)} LLM calls all at once in threads (they only wait on the API; no
    gmsh). Each call fills its own usage records list, added to the cost log as it finishes. A call
    that raises or returns None is logged and gives None, so the rest of the generation carries on.
    then(results so far), if given, is called after each call finishes and returns more jobs to
    start at once (e.g. an organizer call whose inputs are all in); their results are returned too.
    Returns ({name: result or None}, {name: seconds})."""
    records, seconds, results = {}, {}, {}

    def call(name, fn, kwargs):
        t0 = time.perf_counter()
        try:
            res = fn(**kwargs, records=records[name])
            if res is None:
                print(f"  {name}: no valid output; skipped")
        except Exception as e:  # noqa: BLE001  e.g. an API error: skip this one
            print(f"  {name} failed: {e!r}; skipped")
            res = None
        seconds[name] = round(time.perf_counter() - t0, 2)
        return res

    if not jobs:
        return {}, {}
    # Threads are only started as needed: every call runs at once, as no call waits for a free thread.
    with ThreadPoolExecutor(256) as pool:
        pending = {}

        def start(new):
            for name, (fn, kwargs) in new.items():
                records[name] = []
                pending[pool.submit(call, name, fn, kwargs)] = name
        start(jobs)
        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for f in done:
                name = pending.pop(f)
                results[name] = f.result()
                cost.add(records[name])
            if then is not None:
                start(then(results))
    return {name: results[name] for name in records}, seconds


def _dump(x):
    """A pydantic result as a plain dict (None stays None)."""
    return x.model_dump() if hasattr(x, "model_dump") else x


def _highlight_note(h: dict) -> dict:
    return {"section": "meshing", "kind": "highlight", "source": f"peer:{h.get('agent')}@{h.get('task') or 'general'}",
            "content": f"{h.get('agent')} on {h.get('task')}: {h.get('what')} Why it works: {h.get('why_it_works')} "
                       f"How to copy: {h.get('how_to_copy')}"}


def _first_text(run: dict, max_chars: int = 4000) -> str:
    """The agent's first message (its checklist: what it promised to do about last feedback)."""
    for t in run.get("trace") or []:
        if t.get("type") == "text" and (t.get("text") or "").strip():
            return t["text"][:max_chars]
    return "(no first message)"


def _run_summary(run: dict, report: dict | None = None) -> dict:
    """A run in a few numbers: task, the submitted mesh request, score, errors, rank. Only
    aggregates (no hidden check-point data), so it is safe to show to the agent."""
    c = run.get("checks") or {}
    n = (report or {}).get("numbers") or {}
    meshes, solves, i = run.get("meshes") or [], run.get("solves") or [], run.get("submitted")
    mesh = None
    if isinstance(i, int) and 0 <= i < len(solves):
        mi = solves[i].get("mesh_index")
        if isinstance(mi, int) and 0 <= mi < len(meshes):
            mesh = meshes[mi].get("request")
    if mesh is None and meshes:                       # no submit: its last mesh
        mesh = meshes[-1].get("request")
    nl = c.get("nonlinear") is not None           # a nonlinear task: its force-curve error, no field error
    return {"run_id": run["run_id"], "task": run["problem_id"], "mesh_call": mesh,
            "score": c.get("score"), "field_error": None if nl else c.get("rel_error"),
            "nonlinear_error": run_error(run) if nl else None, "probe_error": c.get("probe_error"),
            "dofs": c.get("dofs"), "rank": n.get("rank_in_group"), "group_size": n.get("group_size"),
            "failed_gate": c.get("failed_gate"), "error": run.get("error")}


def _accepted(fn, kwargs: dict) -> dict:
    """kwargs minus those fn does not take (so new teacher inputs can land before teach accepts them)."""
    import inspect
    params = inspect.signature(fn).parameters
    if any(p.kind is p.VAR_KEYWORD for p in params.values()):
        return kwargs
    dropped = [k for k in kwargs if k not in params]
    if dropped:
        print(f"  note: {fn.__name__} does not take {dropped} yet; not passed")
    return {k: v for k, v in kwargs.items() if k in params}


def private_incoming(af: dict | None, refl: dict | None) -> list[dict]:
    """What the organizer files into one agent's notes after a run: the teacher's notes and next mesh
    call for it (af: its AgentFeedback dict with "task") and its own reflection's notes (refl: its
    SelfReflection dict with "task"), each tagged "@<task>". Also used by ace/single_loop.py."""
    items = []
    if af is not None:
        items += [{**n, "source": f"teacher@{n.get('task') or af['task']}"} for n in af.get("notes") or []]
        items.append({"section": "meshing", "kind": "fix", "source": f"teacher@{af['task']}",
                      "content": f"Teacher's next mesh call for {af['task']}: "
                                 f"{process_brief.mesh_call_text(af.get('next_mesh_call'))}"})
    if refl is not None:
        items += [{**n, "source": f"self@{n.get('task') or refl['task']}"} for n in refl.get("what_i_learned") or []]
    return items


def shared_incoming(pid: str, fb: dict) -> list[dict]:
    """What the organizer files into the shared playbook from the teacher's review of task pid (fb: a
    TeacherFeedback dict): shared lessons, highlights, setup warnings and contested regions. Also used
    by ace/single_loop.py."""
    items = [{**n, "source": f"shared-lesson@{n.get('task') or pid}"} for n in fb.get("shared_lessons") or []]
    items += [_highlight_note({**h, "task": pid}) for h in fb.get("highlights") or []]
    items += [{"section": "solver_setup", "kind": "avoid", "source": f"teacher@{pid}", "content": w}
              for w in fb.get("setup_warnings") or []]
    items += [{"section": "meshing", "kind": "fix", "source": f"teacher@{pid}",
               "content": f"Contested region on {pid}: {c.get('region')}. Why: {c.get('why_contested')} "
                          f"What to do: {c.get('what_to_do')}"}
              for c in fb.get("contested_regions") or [] if isinstance(c, dict)]
    return items


def deep_learn(gen: int, gdir: Path, runs: list[dict], reports: dict, mem: Memory, train: list[str], llm,
               cost: CostLog, state: dict, history: dict | None = None) -> dict:
    """--deep steps 3-4 for one generation.
      3. one thread pool: the teacher once per task and a self-reflection once per run, all at once;
      4. in the same pool, as soon as its inputs are in: the organizer once per agent (its private
         notes) and once for the shared playbook; their new note lists replace the old ones
         (Memory.replace_notes).
    `state` carries what the next generation's contexts need and is updated in place:
    "teacher" {task: last TeacherFeedback dict}, "feedback" {agent: its last AgentFeedback dict},
    "reflection" {agent: its last SelfReflection dict}, "highlights" [this generation's highlights],
    "last_run" {agent: _run_summary of its last taught run}.
    history: {task: every run so far on it}, for the teacher's error-vs-nodes summary (nonlinear tasks).
    Returns {"timings": {...}, "private": {agent: changes}, "shared": changes or None}."""
    from ace.deep_teacher import teach        # --deep only (and written alongside this mode)
    from ace.organizer import organize_private, organize_shared
    from ace.self_reflect import reflect_on_run

    # Only runs with a usable report are taught or reflected on (crashed runs say nothing).
    ok = [r for r in runs if r["problem_id"] in train and not r.get("error")
          and r["run_id"] in reports and "error" not in reports[r["run_id"]]]
    taught = {r["run_id"] for r in ok}
    for r in runs:
        if r["problem_id"] in train and not r.get("error") and r["run_id"] not in taught:
            print(f"  no run report for {r['run_id']} ({reports.get(r['run_id'], {}).get('error')}); not taught on")
    name = lambda r: f"a{agent_of(r['run_id'])}"                       # noqa: E731
    notes = lambda r: process_brief.render_notes(mem.private[agent_of(r["run_id"])])   # noqa: E731
    shared_text = process_brief.render_notes(mem.shared)

    state.setdefault("last_run", {})
    timings = {}

    # Disagreement map per task (no ground truth needed): where the runs' von Mises fields differ.
    t0 = time.perf_counter()
    disagreement = {}
    for pid in train:
        pruns = [r for r in ok if r["problem_id"] == pid]
        if len(pruns) >= 2 and not is_nonlinear(pid):       # a nonlinear task has no von Mises map
            try:
                from ace.run_report import disagreement_map
                disagreement[pid] = disagreement_map([reports[r["run_id"]] for r in pruns])
            except Exception as e:  # noqa: BLE001  the teacher still works without it
                print(f"  disagreement map for {pid} failed: {e!r}; teacher gets none")
    timings["disagreement"] = round(time.perf_counter() - t0, 2)

    # Continuity: each agent's OWN previous run, the feedback and reflection it got on it, and the
    # first message of this run (its promises), so the teacher can check each promise was kept.
    def last_by_agent(pruns):
        return {name(r): {"previous_run": state["last_run"].get(name(r)),
                          "previous_feedback": state["feedback"].get(name(r)),
                          "previous_reflection": state["reflection"].get(name(r)),
                          "promised_this_run": _first_text(r)} for r in pruns}

    # 3. teacher per task + self-reflection per run: one pool, so they overlap fully.
    jobs = {}
    for pid in train:
        pruns = [r for r in ok if r["problem_id"] == pid]
        if pruns:
            jobs[f"teacher {pid}"] = (teach, _accepted(teach, dict(
                task=pid, reports=[reports[r["run_id"]] for r in pruns], runs=pruns,
                notes_by_agent={name(r): notes(r) for r in pruns}, shared_text=shared_text,
                prev_feedback=None, disagreement=disagreement.get(pid),
                last_by_agent=last_by_agent(pruns), history=(history or {}).get(pid), llm=llm)))
    for r in ok:
        jobs[f"reflect {r['run_id']}"] = (reflect_on_run, dict(report=reports[r["run_id"]], run=r,
                                                               own_notes_text=notes(r), llm=llm))

    def collect(res: dict) -> tuple[dict, dict, dict]:
        """The teacher's feedback per task, its feedback per agent and each agent's own reflection,
        from the calls finished so far (a missing call counts as no output)."""
        feedback = {pid: _dump(res.get(f"teacher {pid}")) for pid in train}
        per_agent, own = {}, {}
        for pid, fb in feedback.items():
            for af in (fb or {}).get("per_agent") or []:
                k = agent_of(af.get("run_id") or "")
                a = f"a{k}" if k is not None else af.get("agent")      # trust the run id over the name
                per_agent[a] = {**af, "agent": a, "task": pid}
        for r in ok:
            refl = _dump(res.get(f"reflect {r['run_id']}"))
            if refl is not None:
                own[name(r)] = {**refl, "run_id": r["run_id"], "task": r["problem_id"]}
        return feedback, per_agent, own

    # 4. organizer: what each agent gets = the teacher's notes and mesh call for it and its own
    #    reflection's notes, each tagged "@<task>"; the shared playbook gets every task's shared
    #    lessons, highlights, setup warnings and contested regions. (Peer highlights stay out of the
    #    private lists: agents see them in the shared playbook and the context's highlights section.)
    #    An agent's organizer call starts as soon as its inputs are complete: the teacher of each task
    #    it ran (the teacher attributes feedback by run id, so no other teacher writes about it) and
    #    its own reflections. The shared one starts once every teacher is done.
    needs = {}
    for r in ok:
        needs.setdefault(name(r), set()).update({f"teacher {r['problem_id']}", f"reflect {r['run_id']}"})
    needs["shared"] = {j for j in jobs if j.startswith("teacher ")}
    incoming, owners, decided = {}, {}, set()
    t_phase3 = []                               # when the last teacher/reflection call finished

    def organize_ready(res: dict) -> dict:
        if set(jobs) <= set(res) and not t_phase3:
            t_phase3.append(time.perf_counter())
        if cost.over_budget():                  # as cost.check() before the organizer: no new spend
            return {}
        new, (feedback, per_agent, own) = {}, collect(res)
        for owner, need in needs.items():
            if owner in decided or not need <= set(res):
                continue
            decided.add(owner)
            if owner == "shared":
                items = [x for pid, fb in feedback.items() if fb is not None for x in shared_incoming(pid, fb)]
                if not items:                   # nothing to file: no shared organizer call
                    continue
                owners[owner], fn, kw = mem.shared, organize_shared, {}
            else:
                items = private_incoming(per_agent.get(owner), own.get(owner))
                owners[owner], fn, kw = mem.private[int(owner[1:])], organize_private, {"agent": owner}
            incoming[owner] = items
            new[f"organize {owner}"] = (fn, dict(**kw, current=notes_as_dicts(owners[owner]), incoming=items, llm=llm))
        return new

    res, secs = _llm_jobs(jobs, cost, then=organize_ready)
    timings.update(teacher=max([v for k, v in secs.items() if k.startswith("teacher")], default=0),
                   reflect=max([v for k, v in secs.items() if k.startswith("reflect")], default=0))
    # organize: how long the organizer calls ran on after the last teacher / reflection call
    timings["organize"] = round(time.perf_counter() - (t_phase3[0] if t_phase3 else time.perf_counter()), 2)

    feedback, per_agent, own = collect(res)
    (gdir / "reflections").mkdir(exist_ok=True)
    highlights = []
    for pid, fb in feedback.items():
        if fb is None:
            continue
        (gdir / f"teacher_{pid}.json").write_text(json.dumps(fb, indent=1))
        state["teacher"][pid] = fb
        highlights += [{**h, "task": pid} for h in fb.get("highlights") or []]
    for r in ok:
        refl = _dump(res.get(f"reflect {r['run_id']}"))
        if refl is not None:
            (gdir / "reflections" / f"{r['run_id']}.json").write_text(json.dumps(refl, indent=1))
    state["feedback"].update(per_agent)
    state["reflection"].update(own)
    state["last_run"].update({name(r): _run_summary(r, reports.get(r["run_id"])) for r in ok})
    state["highlights"] = highlights
    cost.check()

    (gdir / "organizer").mkdir(exist_ok=True)
    changes = {}
    for owner in [o for o in needs if o in owners]:     # agents in run order, then shared (as before)
        pb = owners[owner]
        before, after = notes_as_dicts(pb), res.get(f"organize {owner}")
        rec = {"before": before, "incoming": incoming[owner], "after": None, "changes": None}
        if after is None:
            rec["error"] = "organizer gave no valid output; notes kept unchanged"
        else:
            changes[owner] = Memory.replace_notes(pb, after, gen)
            rec.update(after=notes_as_dicts(pb), changes=changes[owner])
        (gdir / "organizer" / f"{owner}.json").write_text(json.dumps(rec, indent=1))

    def counts(chs: list[dict]) -> str:
        n = max(1, len(chs))
        return " ".join(f"{sign}{sum(len(c[k]) for c in chs) / n:.1f}"
                        for sign, k in (("+", "added"), ("~", "edited"), ("-", "removed")))
    priv = [c for o, c in changes.items() if o != "shared"]
    print(f"  context: private {counts(priv)} (mean over {len(priv)} agents), "
          f"shared {counts([changes['shared']]) if 'shared' in changes else 'unchanged'}")
    cost.check()
    return {"timings": timings, "private": {o: c for o, c in changes.items() if o != "shared"},
            "shared": changes.get("shared")}


# --------------------------------------------------------------------------- what an agent sees

# --personas: a fixed exploration style for some (or all) agents. Without one, agents reading the same
# playbook converge on near-identical meshes, so the population stops exploring (and ties earn
# nothing). The i-th chosen agent gets PERSONAS[i], so up to len(PERSONAS) chosen agents all differ.
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
    "Point refinements: refine at specific points (corners, load application points, the disclosed check "
    "points) with `point` refinements and small radii, rather than whole named groups.",
    "Gentle grading: give every local refinement a radius of 3-5 times its local size, so element sizes grow "
    "smoothly and no region has abrupt size jumps.",
    "Coarse quadratic background: order 2 with a global h 1.5-2 times the automesher's, and spend the DOFs "
    "saved on two or three local refinements where the stress varies fastest.",
    "Error-driven moves: after each solve, move elements from where your results changed least to where they "
    "changed most between your last two meshes, keeping the DOF count about the same.",
]


def persona_agents(spec: str | None, count: int | None, n_agents: int) -> list[int]:
    """Which agents get a persona: --personas alone ("all") = every agent, --personas a0,a3,a5 = those,
    --persona-count N = a0..a{N-1}; neither = none."""
    if count is not None:
        return list(range(min(count, n_agents)))
    if spec is None:
        return []
    if spec == "all":
        return list(range(n_agents))
    ks = sorted({int(a.strip().removeprefix("a")) for a in spec.split(",") if a.strip()})
    if any(k >= n_agents for k in ks):
        raise SystemExit(f"--personas {spec}: there are only {n_agents} agents (a0..a{n_agents - 1})")
    return ks


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
        err = "force error vs the test" if c.get("nonlinear") is not None else "field error"
        lines.append(f"- a{j}: {mesh}; DOFs {_fmt(c.get('dofs'))}, {err} {_fmt(run_error(r))}, "
                     f"base score {_fmt(c.get('base_score', c.get('score')))}")
    if not lines:
        return ""
    return ("\n\n## WHAT OTHER AGENTS DID LAST GENERATION ON THIS TASK\n" + "\n".join(lines)
            + "\nA tie with another agent on the hidden check points pays nothing, so try something "
              "different and better.")


# --------------------------------------------------------------------------- --init-from: a trained swarm


def load_init(export: Path, names: list[str]) -> dict:
    """What a run starts from with --init-from (an ace.export_swarm folder): its shared playbook and,
    for the agents `names` in that order (renamed a0, a1, ...), their private notes, personas and
    last feedback, own reflection and last-run summary (state.json). The training swarm's
    highlights are left out: they name its agents by their old names."""
    manifest = json.loads((export / "manifest.json").read_text())
    if missing := [a for a in names if a not in manifest["agents"]]:
        raise SystemExit(f"--init-agents {','.join(missing)}: not in {export} (agents {manifest['agents']})")
    if (export / "playbook.json").exists():
        shared = Playbook.load(export / "playbook.json")
    else:   # per-task scope: one playbook for all tasks (--deep uses one shared playbook)
        shared = Playbook(bullets=[b for p in sorted((export / "playbooks").glob("*.json"))
                                   for b in Playbook.load(p).bullets])
    personas = json.loads((export / "personas.json").read_text()) if (export / "personas.json").exists() else {}
    state = json.loads((export / "state.json").read_text()) if (export / "state.json").exists() else {}
    agents = state.get("agents") or {}
    new = {old: f"a{k}" for k, old in enumerate(names)}
    private = [renamed(Playbook.load(export / "private" / f"{old}.json"), old, new[old]) for old in names]
    deep = {"feedback": {}, "reflection": {}, "last_run": {}}
    for old, a in new.items():
        for key in deep:
            if (agents.get(old) or {}).get(key) is not None:
                deep[key][a] = {**agents[old][key], **({"agent": a} if key == "feedback" else {})}
    return {"shared": shared, "private": private, "deep": deep,
            "personas": {k: personas[old] for k, old in enumerate(names) if personas.get(old)},
            "record": {"from": str(export), "run": manifest.get("run"), "generation": manifest.get("generation"),
                       "trained_on": manifest.get("tasks"), "mapping": {a: old for old, a in new.items()},
                       "personas": {a: personas.get(old) for old, a in new.items()},
                       "loaded": ["shared playbook", "private notes", "personas", "last feedback, reflection and run"]}}


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
    if any(is_nonlinear(r["problem"]) for r in rows):
        how += ["nonlinear tasks: error = mean |F_sim - F_test| / F_test over the force curve's checkpoints (a",
                "frozen nonlinear CalculiX ramp of the run's mesh, ace.nonlinear), scored like field mode against",
                "the automesher run through the same ramp;"]
    told = ("Agents got each problem's engineering requirements (ace/requirements.py) and set the boundary "
            "conditions themselves." if config.STATEMENT == "requirements" else
            "Agents got each problem's statement, boundary conditions included.")
    lines = ["# ACE run summary", "", told, "", *how,
             "'failed' counts runs that failed a gate; 'errored' counts runs that crashed (API or code errors).",
             "'base' is the mean score without bonuses; 'bonus' the mean be-the-best + diversity bonus (field mode).",
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
        base = [checks[r["run_id"]].get("base_score", checks[r["run_id"]]["score"]) for r in ok]
        errs = [e for r in ok if (e := run_error({"checks": checks[r["run_id"]]})) is not None]
        dofs = [d for r in ok if (d := checks[r["run_id"]].get("dofs"))]
        rows.append({
            "gen": gen, "problem": pid, "split": "holdout" if pid in holdout else "train",
            "n": len(scores), "mean": statistics.mean(scores), "min": min(scores), "max": max(scores),
            "base": statistics.mean(base), "bonus": statistics.mean(scores) - statistics.mean(base),
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


def _check_flow_tasks(problem_ids: list[str]) -> None:
    """Fail before anything is spent when flow tasks cannot run: they need OpenFOAM + foamlib (or
    every agent's solve fails while its API calls are still billed), and field scoring and
    requirements statements are defined for elasticity only."""
    from ace import fea

    flow = [p for p in problem_ids if fea.is_flow(fea.get_problem(p))]
    if not flow:
        return
    if config.SCORE == "field":
        raise SystemExit(f"ACE_SCORE=field scores the elastic stress field; it is not defined for the flow tasks "
                         f"{flow}. Unset ACE_SCORE (point scoring) for the CFD suite.")
    if config.STATEMENT == "requirements":
        raise SystemExit(f"--statement requirements is defined for elasticity setups only (ace/requirements.py); "
                         f"the flow tasks {flow} need --statement bcs.")
    from feacore.solver._foam_env import FoamNotFound, require_openfoam
    from feacore.solver._foam_io import FOAMLIB_ERROR

    try:
        fe = require_openfoam()
    except FoamNotFound as e:
        raise SystemExit(f"the flow tasks {flow} need OpenFOAM: {e}. On a Mac: bash scripts/openfoam_docker.sh, then "
                         f"export FEACORE_OPENFOAM_BASHRC=~/.feacore_foam/bashrc TMPDIR=~/.feacore_foam/tmp "
                         f"(ace/README.md, 'The CFD suite').") from None
    if FOAMLIB_ERROR is not None:
        raise SystemExit(f"the flow tasks {flow} need foamlib (pip install -e '.[cfd]'): {FOAMLIB_ERROR}")
    print(f"flow tasks {flow}: {fe.describe()}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", choices=sorted(config.SUITES), default="fea",
                    help="default train / held-out problems: fea (elasticity), cfd (2D laminar flow, OpenFOAM) "
                         "or cfd3d (3D laminar flow on tets, OpenFOAM)")
    ap.add_argument("--problems", default=None, help="train problems (reflected on); default: the suite's")
    ap.add_argument("--holdout", default=None, help="held-out problems (evaluated only); default: the suite's")
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
                         "train problem; private notes follow the agent. With one train problem every agent "
                         "works on it")
    ap.add_argument("--teacher", choices=["rerun", "inspect", "both"], default="rerun",
                    help="what the teacher may do: counterfactual reruns, a detailed inspection, or both")
    ap.add_argument("--shared-scope", choices=["all", "task"], default="all",
                    help="with --split: one shared playbook for all tasks, or one per task")
    ap.add_argument("--shared-bar", choices=["counterfactual", "any"], default="any",
                    help="shared+private: which teacher lessons may enter the shared playbook (see ace/memory.py)")
    ap.add_argument("--peers", action=argparse.BooleanOptionalAction, default=None,
                    help="show each agent what the other agents did on its task last generation (default: on with --split)")
    ap.add_argument("--personas", nargs="?", const="all", default=None,
                    help="give agents a fixed exploration style: every agent (the flag alone) or some, e.g. a0,a3,a5")
    ap.add_argument("--persona-count", type=int, help="give the first N agents (a0..a{N-1}) a persona")
    ap.add_argument("--deep", action="store_true",
                    help="run reports + deep teacher + self-reflection + organizer (see deep_learn); needs "
                         "--memory shared+private, --split and ACE_SCORE=field; implies --peers, --shared-scope all")
    ap.add_argument("--statement", choices=["bcs", "requirements"], default=config.STATEMENT,
                    help="what agents are told: the problem statement with its boundary conditions, or the "
                         "problem's engineering requirements (ace/requirements.py), from which they set the "
                         "boundary conditions themselves")
    ap.add_argument("--evidence", type=Path, action="append", default=[],
                    help="physical test data (ace/evidence.py): a test spec JSON or a directory of them; repeat for "
                         "more. Files dropped into <out>/evidence/ during the run are picked up at the next generation")
    ap.add_argument("--evidence-score", action="store_true",
                    help="score runs on problems with test data against the test instead of the problem's reference")
    ap.add_argument("--init-from", type=Path,
                    help="start from a trained swarm (an ace.export_swarm folder): its shared playbook and the "
                         "--init-agents' private notes, personas and last feedback")
    ap.add_argument("--init-agents", help="with --init-from: which of its agents, best first, e.g. a6,a3,a1,a7; "
                                          "they become a0, a1, ... (default: all, in its order)")
    ap.add_argument("--pre-teach", action="store_true",
                    help="--deep, one train task: before generation 0 the teacher revises the shared playbook and "
                         "every agent's notes for the task from its teacher brief (ace/pre_teach.py)")
    ap.add_argument("--ui-runs-dir", type=Path, help="also record every solve and agent run in this feacore run "
                    "store, for the feacore web UI (default: $FEACORE_RUNS_DIR if set, else off)")
    args = ap.parse_args(argv)
    # Before any process pool exists: spawned children (agents, scoring, teacher) read it at import.
    os.environ["ACE_TEACHER"] = args.teacher
    config.TEACHER_MODE = args.teacher
    os.environ["ACE_STATEMENT"] = args.statement
    config.STATEMENT = args.statement
    peers = args.split if args.peers is None else args.peers

    suite_train, suite_holdout = config.SUITES[args.suite]
    train = [p for p in (args.problems if args.problems is not None else ",".join(suite_train)).split(",") if p]
    holdout = [p for p in (args.holdout if args.holdout is not None else ",".join(suite_holdout)).split(",") if p]
    _check_flow_tasks(train + holdout)
    out = args.out or Path("ace_runs") / (time.strftime("%Y%m%d-%H%M%S") + ("-mock" if args.mock else ""))
    out.mkdir(parents=True, exist_ok=True)
    (out / "run_config.json").write_text(json.dumps({"suite": args.suite, "train": train, "holdout": holdout}, indent=1))
    # Physical test data (ace.evidence): processed into <out>/evidence_active/, which the spawned agents,
    # scorers and teachers read; set before any pool exists.
    os.environ[evidence.ACTIVE_ENV] = str((out / "evidence_active").resolve())
    os.environ[evidence.SCORE_ENV] = "1" if args.evidence_score else "0"
    (out / "evidence").mkdir(exist_ok=True)
    cost = CostLog(out / "cost.jsonl", args.budget_usd)
    llm = make_llm(args.mock)
    playbook = Playbook.load(args.playbook) if args.playbook else Playbook()
    private = args.memory == "shared+private"
    if args.statement == "requirements":
        from ace.requirements import problems_ready
        if not_ready := problems_ready(train + holdout):
            raise SystemExit("--statement requirements: these problems have no clean requirements:\n  "
                             + "\n  ".join(not_ready))
    if args.teacher != "rerun" and config.SCORE != "field":
        raise SystemExit("--teacher inspect/both needs field scoring (ACE_SCORE=field): its tools read the field error")
    if args.split and not (private and train):
        raise SystemExit("--split needs --memory shared+private and at least one --problems")
    persona_of = {k: PERSONAS[i % len(PERSONAS)]
                  for i, k in enumerate(persona_agents(args.personas, args.persona_count, args.agents))}
    if (peers or persona_of or args.init_from) and not private:
        raise SystemExit("--peers, --personas and --init-from need --memory shared+private (persistent agents)")
    nl_tasks = [p for p in train + holdout if is_nonlinear(p)]     # scored by their force curve (ace.nonlinear)
    if args.deep:
        # field mode, or every train task nonlinear-scored (no field reference needed there)
        if not (private and args.split and (config.SCORE == "field" or all(is_nonlinear(p) for p in train))):
            raise SystemExit("--deep needs --memory shared+private, --split and ACE_SCORE=field "
                             "(or only nonlinear-scored train tasks)")
        peers, args.shared_scope = True, "all"
        # Before any pool: the scoring workers skip the slow mesh-convergence oracle (the run
        # report gives the teacher a better picture of the error for far less time).
        os.environ["ACE_SKIP_CONVERGENCE"] = "1"
    if args.pre_teach and not (args.deep and len(train) == 1):
        raise SystemExit("--pre-teach needs --deep and exactly one train task (--problems)")
    init = None
    if args.init_from:
        if not private or (args.split and args.shared_scope == "task"):
            raise SystemExit("--init-from needs --memory shared+private with one shared playbook")
        if args.playbook:
            raise SystemExit("--init-from brings its own shared playbook: drop --playbook")
        names = [a.strip() for a in (args.init_agents or "").split(",") if a.strip()] or \
            json.loads((args.init_from / "manifest.json").read_text())["agents"]
        init = load_init(args.init_from, names)
        args.agents = len(names)
        playbook = init["shared"]
        persona_of.update(init["personas"])     # the swarm's own personas win over --personas
    per_task = args.split and args.shared_scope == "task"     # one shared playbook per task
    mem = (Memory(args.agents, tasks=train) if per_task else Memory(args.agents, shared=playbook)) if private else None
    if init is not None:
        mem.private = init["private"]
        (out / "init.json").write_text(json.dumps(init["record"], indent=1))
    if config.SCORE == "field":     # build baselines once here, not concurrently in every worker
        from ace.field_error import automesher, probe_points
        for pid in train + holdout:
            if not is_nonlinear(pid):     # nonlinear tasks: no field reference
                automesher(pid), probe_points(pid)
    if nl_tasks:
        # Each of the --concurrency scoring workers runs one CalculiX ramp at a time with
        # ace.nonlinear.run_curve's 4 threads: 4 workers x 4 threads on a 16-vCPU box.
        for pid in nl_tasks:          # the automesher's ramp once here (cached), not in every worker
            try:
                from ace import nonlinear
                nonlinear.automesher_baseline(pid)
            except Exception as e:  # noqa: BLE001  no CalculiX here (a laptop): only a mock run goes on
                if not args.mock:
                    raise
                print(f"  (mock) no nonlinear automesher baseline for {pid}: {e!r}")
    all_runs: dict[str, list[dict]] = {pid: [] for pid in train + holdout}   # every run so far, per problem
    rows, gen_notes = [], []
    print(f"output: {out}")
    if persona_of:   # who has which persona, so analysis can split the population by it
        (out / "personas.json").write_text(json.dumps({f"a{k}": p for k, p in persona_of.items()}, indent=1))
    ui = ui_bridge.settings(args.ui_runs_dir, loop_id=out.resolve().name)   # None = off
    # --deep: what the next generation's contexts are built from (filled by deep_learn)
    deep = {"teacher": {}, "feedback": {}, "reflection": {}, "highlights": [], "last_run": {}}
    if init is not None:
        for key, per_agent in init["deep"].items():
            deep[key].update(per_agent)

    def agent_text(k: int, pid: str | None, gen: int) -> str:
        """Agent k's prompt text on task pid (None: held out): memory, then peers and persona."""
        prev = [r for r in all_runs[pid] if r["generation"] == gen - 1] if pid is not None else []
        if args.deep:
            a = f"a{k}"
            text = process_brief.deep_context(
                a, pid, mem.shared, mem.private[k],     # Playbooks: rendered grouped by task
                deep["feedback"].get(a), deep["reflection"].get(a), deep["highlights"],
                peers_text(k, pid, prev) if pid is not None else "",
                last_run=deep["last_run"].get(a), task_feedback=deep["teacher"].get(pid), persona=persona_of.get(k))
        else:
            text = mem.prompt_text(k, args.top_n, pid)
        if peers and pid is not None and not args.deep:
            text += peers_text(k, pid, prev)
        if k in persona_of and not args.deep:     # --deep: inside deep_context, before its closing block
            text += "\n\n" + process_brief.persona_block(persona_of[k])
        return text

    if args.pre_teach:   # the teacher revises everyone's context for the new task before generation 0
        from ace import pre_teach
        t0 = time.perf_counter()
        origin = (f"exported swarm {init['record']['run']} (trained {init['record']['generation'] + 1} generations on "
                  f"{', '.join(init['record']['trained_on'] or [])}); agents renamed: "
                  + ", ".join(f"{a} (was {old})" for a, old in init["record"]["mapping"].items())
                  if init is not None else "(no trained swarm: the notes are whatever the run started with)")
        res, _ = _llm_jobs({"pre-teach": (pre_teach.ask, dict(
            task=train[0], mem=mem, persona_of=persona_of, last_run=deep["last_run"], origin=origin, llm=llm))}, cost)
        if res.get("pre-teach") is not None:
            ch = pre_teach.apply(res["pre-teach"], train[0], mem, deep, out / "pre_teach")
            print(f"  pre-teach ({time.perf_counter() - t0:.0f} s): " + "; ".join(
                f"{o} +{len(c['added'])} ~{len(c['edited'])} -{len(c['removed'])}" for o, c in ch.items()))

    try:
        for gen in range(args.generations):
            # Early stop without killing the process: `touch <out>/STOP` ends the run cleanly before the
            # next generation starts (nothing more is spent; the summary and plot are still written).
            if (out / "STOP").exists():
                print(f"\nSTOP file found in {out}; stopping before generation {gen}.")
                break
            gdir = out / f"gen{gen}"
            (gdir / "runs").mkdir(parents=True, exist_ok=True)
            # Physical test data, also when it arrives mid-run (after the population has settled): every
            # agent, scorer and teacher of this generation sees it.
            for pid in evidence.refresh(out, args.evidence, all_runs, gen):
                if evidence.scored_by_test(pid) and all_runs[pid]:     # earlier runs: re-checked against the test
                    for r in all_runs[pid]:
                        if not r.get("error"):
                            r["checks"] = check_run(pid, r)
                    score_population(all_runs[pid])
            if gen == 0 and (missing := evidence.missing_tests([p for p in train + holdout if not is_nonlinear(p)])):
                raise SystemExit(f"{', '.join(missing)}: no reference value (no ground truth) and no physical test "
                                 "data to score against; give the test with --evidence (see ace/evidence.py)")
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
            if args.deep:   # the exact context each agent sees this generation
                (gdir / "context").mkdir(exist_ok=True)
                for j in jobs:
                    a = f"a{agent_of(j['run_id'])}"
                    fname = a if j["problem_id"] in train else f"{a}-{j['problem_id']}"
                    (gdir / "context" / f"{fname}.md").write_text(j["playbook_text"])
            t0 = time.perf_counter()
            scoring = Scoring(args.concurrency, gdir / "reports" if args.deep else None)

            def on_done(r):                     # each run's scoring starts as soon as its agent is done
                ui_bridge.record_agent_run(ui, r)
                scoring.start(r)
            runs = run_generation(jobs, args.concurrency, cost, on_done=on_done)
            phase = {"gen": gen, "what": "phase", "agents": round(time.perf_counter() - t0, 2)}
            if runs and all(r.get("error") for r in runs):
                print(runs[0]["error"])
                raise SystemExit("every run failed; see the error above")

            # 2. checks: score, then oracles and stage invariants (layers 1 and 2 of diagnosis). Most
            #    runs were scored while the last agents worked: "scoring" is the wait after them.
            t_oracles = time.perf_counter()
            reports = scoring.finish(runs)
            for r in runs:
                all_runs[r["problem_id"]].append(r)
            # Speed/accuracy score: each problem's runs scored against every run on it so far.
            for pid in all_runs:
                if all_runs[pid]:
                    score_population(all_runs[pid])
            if args.deep:   # the scores go into the run reports, with each run's rank on its task
                from ace.run_report import add_scores
                for pid in train + holdout:
                    group = sorted((r for r in runs if r["problem_id"] == pid and not r.get("error")),
                                   key=lambda r: -(r["checks"].get("score") or 0))
                    for i, r in enumerate(group, 1):
                        report = reports.get(r["run_id"])
                        if report is not None and "error" not in report:
                            reports[r["run_id"]] = add_scores(
                                report, {**r, "checks": {**r["checks"], "rank": i, "group_size": len(group)}})
                            (gdir / "reports" / f"{r['run_id']}.json").write_text(
                                json.dumps(reports[r["run_id"]], indent=1, default=str))
                # slowest single report (they run in parallel inside the scoring phase)
                phase["report"] = max((x.get("seconds") or 0 for x in reports.values()), default=0)
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
            if args.deep:   # 3-5 replaced: deep teacher + self-reflection, then the organizer
                learned = deep_learn(gen, gdir, runs, reports, mem, train, llm, cost, deep,
                                     history={pid: all_runs[pid] for pid in train})
                mem.save(out, gen + 1)
                phase.update(learned["timings"])
                sh = learned["shared"] or {"added": [], "removed": []}
                note.update(usd=cost.total, added=len(sh["added"]), pruned=len(sh["removed"]),
                            private_added=sum(len(c["added"]) for c in learned["private"].values()))
                solver_log(out, phase)
                print("  timings (s): " + ", ".join(f"{k} {v}" for k, v in phase.items() if k not in ("gen", "what")))
                continue
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
