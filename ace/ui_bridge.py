"""Optional bridge to the feacore run store, so ACE runs show up live in the feacore web UI.

Off unless `python -m ace.run_loop --ui-runs-dir DIR` is used or FEACORE_RUNS_DIR is set. When on:

  each solve an agent makes    -> one run record (RunStore.record_result, with the viewer fields;
                                  a failed solve -> a run with status "failed" and the error)
  each agent run, as it ends   -> one submission (RunStore.create_submission) of its successful
                                  solves; the agent shows as finished, its score is still null
  each time the loop scores    -> every submission of the loop gets its current score (the score is
                                  relative to all runs on the problem so far, so it moves as the
                                  population grows)
  each diagnoser session       -> its verdict on each run it diagnosed

Both records have agent_id "ace:<ace run id>" and source "ace". Their meta:
  run:        {source, loop_id, generation, ace_run_id, solve_index, mesh}
              solve_index counts from 0 (the agent's "solve n" is n - 1); mesh = the agent's mesh
              call arguments as given (h, order, refinements as a list of dicts, ...)
  submission: {source, loop_id, generation, ace_run_id, agent, submitted_run_id, score, score_parts,
               gates, setup_diagnostics, cited_bullets, tokens, diagnosis}
    agent             the persistent agent ("a3") in shared+private memory mode, else null
    submitted_run_id  the solve ACE scores: the submitted one, else the last successful one
    score             the loop's speed/accuracy score, valid x accuracy x efficiency in [0, 1]
                      (ace/checks.py); null until the generation is scored, and for a run that
                      crashed (API or code error, see `error`), which ACE leaves out of its scores
    score_parts       {accuracy, efficiency, valid, rel_error, tolerance, meets_tol, dofs,
                       pareto_rank, population}; population = runs on the problem it was scored among
    gates             why the run is invalid (scores 0): a failed gate or feacore fail flags
    setup_diagnostics the first failing stage, failing oracles, differences from the reference
                      setup and the diagnoser's verdict, as short strings
    diagnosis         {status, stage, category, cause, evidence, gaming} from the diagnoser, or null
                      (it sees training problems only, and not in the last generation)
loop_id is the name of the loop's output directory. With the bridge on, ACE's own run records
also carry "ui_submission_id", and each solve "ui_run_id". Every function here turns any
exception into one warning: a missing or broken store never stops, slows or changes the loop.
"""

from __future__ import annotations

import functools
import os
import warnings
from pathlib import Path

SCORE_PARTS = ("accuracy", "efficiency", "valid", "rel_error", "tolerance", "meets_tol", "dofs", "pareto_rank")


def _safe(fn):
    """Any exception in `fn` becomes one RuntimeWarning (and a None result)."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as e:  # noqa: BLE001  the bridge must never break the loop
            warnings.warn(f"ace.ui_bridge.{fn.__name__} failed, not recorded: {type(e).__name__}: {e}",
                          RuntimeWarning, stacklevel=2)
            return None

    return wrapper


@_safe
def settings(runs_dir: str | Path | None, loop_id: str) -> dict | None:
    """Settings for one loop invocation, or None (off). On when `runs_dir` (--ui-runs-dir) is
    given or FEACORE_RUNS_DIR is set. A plain dict, so it can travel to the workers in each job."""
    root = runs_dir or os.environ.get("FEACORE_RUNS_DIR")
    if not root:
        return None
    from feacore.runstore import RunStore

    store = RunStore(Path(root).expanduser().resolve()).ensure()   # an unusable dir fails here, once
    print(f"feacore UI: recording solves and agent runs in {store.root} (loop {loop_id})")
    return {"root": str(store.root), "loop_id": str(loop_id)}


@_safe
def for_run(ui: dict | None, run_id: str, generation: int) -> dict | None:
    """The settings plus the agent run that the solves belong to (None stays None)."""
    return {**ui, "ace_run_id": run_id, "generation": generation} if ui else None


@_safe
def record_solve(ctx: dict | None, tools, result) -> None:
    """Record the solve just added to tools.solves as one run. `result` is its feacore Result
    (None if the solve failed). The new run id is kept on the solve as "ui_run_id"."""
    if not ctx:
        return
    from feacore.runstore import RunStore, now_iso

    store = RunStore(ctx["root"])
    i = len(tools.solves) - 1
    solve = tools.solves[i]
    meta = {"source": "ace", "loop_id": ctx["loop_id"], "generation": ctx["generation"],
            "ace_run_id": ctx["ace_run_id"], "solve_index": i, "mesh": solve["mesh"].get("request") or {}}
    who = {"problem_id": tools.problem_id, "agent_id": f"ace:{ctx['ace_run_id']}", "source": "ace", "meta": meta}
    if result is not None:
        rid = store.record_result(result, mesh_path=solve["mesh"]["path"], **who)
    else:
        rid = store.create_run(case=_case_dict(tools, solve), status="failed", **who)["run_id"]
        store.update_run(rid, error=solve["result"].get("error"), finished_at=now_iso())
    solve["ui_run_id"] = rid


@_safe
def record_agent_run(ui: dict | None, run: dict) -> None:
    """When an agent run ends: a submission of its successful solves, so the UI shows the agent
    as finished. Its score stays null until the loop scores the generation (record_scores)."""
    if not ui:
        return
    from ace.checks import check_run

    run["ui_submission_id"] = _write(ui, run, check_run(run["problem_id"], run))


@_safe
def record_scores(ui: dict | None, all_runs: dict[str, list[dict]]) -> None:
    """After the loop scored the population ({problem: every run on it so far}): each run's
    submission gets its current score, gates and oracle / stage findings."""
    if not ui:
        return
    failed = []
    for runs in all_runs.values():
        for r in runs:
            try:
                r["ui_submission_id"] = _write(ui, r, r["checks"], population=len(runs))
            except Exception as e:  # noqa: BLE001  one bad record must not stop the others
                failed.append(f"{r['run_id']}: {type(e).__name__}: {e}")
    if failed:
        raise RuntimeError(f"{len(failed)} submission(s) not updated, first: {failed[0]}")


@_safe
def record_diagnosis(ui: dict | None, runs: list[dict], diagnosis) -> None:
    """The diagnoser's verdict (an ace.diagnoser.Diagnosis, or None) on each run it diagnosed."""
    if not ui or diagnosis is None:
        return
    by_id = {r["run_id"]: r for r in runs}
    for rd in diagnosis.runs:
        if (r := by_id.get(rd.run_id)) is not None:
            r["ui_submission_id"] = _write(ui, r, r["checks"], diagnosis={
                "status": rd.diagnosis_status, "stage": rd.stage, "category": rd.category, "cause": rd.cause,
                "evidence": rd.evidence, "gaming": rd.gaming})


# --------------------------------------------------------------------------- helpers


def _write(ui: dict, run: dict, checks: dict, population: int | None = None, diagnosis: dict | None = None) -> str:
    """Create the run's submission, or rewrite its meta (keeping the population and diagnosis it
    already has unless new ones are given). Returns the submission id."""
    from feacore.runstore import RunStore, write_json

    store = RunStore(ui["root"])
    sid = run.get("ui_submission_id")
    old = store.get_submission(sid) if sid else None
    before = (old or {}).get("meta") or {}
    if population is None:
        population = (before.get("score_parts") or {}).get("population")
    meta = _meta(ui, run, checks, population, diagnosis or before.get("diagnosis"))
    if old is not None:
        write_json(store.submissions_dir / f"{sid}.json", {**old, "meta": meta})
        return sid
    agent_id = f"ace:{run['run_id']}"
    run_ids = [s["ui_run_id"] for s in run.get("solves") or [] if s["result"].get("ok") and s.get("ui_run_id")]
    if run_ids:
        return store.create_submission(run["problem_id"], agent_id, run_ids=run_ids, meta=meta)["submission_id"]
    return _empty_submission(store, run["problem_id"], agent_id, meta)


def _meta(ui: dict, run: dict, checks: dict, population: int | None, diagnosis: dict | None) -> dict:
    from ace.checkpoint import scored_solve_index
    from ace.memory import agent_of

    crashed = bool(run.get("error"))
    i = scored_solve_index(run)
    agent = agent_of(run["run_id"])
    meta = {
        "source": "ace", "loop_id": ui["loop_id"], "generation": run["generation"], "ace_run_id": run["run_id"],
        "agent": None if agent is None else f"a{agent}",
        "submitted_run_id": None if i is None else run["solves"][i].get("ui_run_id"),
        "score": None if crashed else checks.get("score"),
        "score_parts": {**{k: checks.get(k) for k in SCORE_PARTS}, "population": population},
        "gates": [] if crashed else _gates(run, checks),
        "setup_diagnostics": _diagnostics(run, checks) + _diagnosis_lines(diagnosis),
        "cited_bullets": list(run.get("cited") or []),
        "tokens": _tokens(run.get("usage") or []),
        "diagnosis": diagnosis,
    }
    if crashed:   # the last line of the traceback
        meta["error"] = (str(run["error"]).strip().splitlines() or ["error"])[-1][:500]
    return meta


def _gates(run: dict, checks: dict) -> list[str]:
    """Why a run is invalid (score 0): its failed gate, else the feacore fail flags that the loop's
    score_population holds against it (flags that also fire on the reference setup don't count)."""
    if checks.get("failed_gate"):
        return [checks["failed_gate"]]
    if checks.get("valid") is not False:
        return []
    from ace.oracles import baseline_fail_flags

    base = baseline_fail_flags(run["problem_id"])
    fails = [f["id"] for f in (run.get("oracles") or {}).get("flags", []) if f["severity"] == "fail" and f["id"] not in base]
    return [f"invalid: {', '.join(fails) or 'feacore flags'}"]


def _diagnostics(run: dict, checks: dict) -> list[str]:
    """The first failing stage (ace/stages.py), failing oracles (ace/oracles.py) and differences
    from the reference setup (ace.checks.bc_report), as short strings."""
    out = []
    if stage := (run.get("stages") or {}).get("first_failing_stage"):
        out.append(f"first failing stage: {stage}")
    out += [f"{name}: {o.get('why')}" for name, o in (run.get("oracles") or {}).items()
            if isinstance(o, dict) and o.get("status") == "fail"]
    s = checks.get("setup") or {}
    out += [f"missing {x}" for x in s.get("missing", [])] + [f"unexpected {x}" for x in s.get("unexpected", [])]
    out += list(s.get("sign_flips", []))
    return out + [f"{k.removesuffix('_ok')} differs from the reference"
                  for k in ("formulation_ok", "thickness_ok", "material_ok") if s.get(k) is False]


def _diagnosis_lines(d: dict | None) -> list[str]:
    """The diagnoser's verdict as setup_diagnostics lines (none for a run it found nothing wrong with)."""
    if not d:
        return []
    out = []
    if d.get("category") not in (None, "none"):
        where = " / ".join(x for x in (d.get("stage"), d.get("category")) if x)
        out.append(f"diagnoser ({d.get('status')}): {where}: {d.get('cause')}")
    if d.get("gaming"):
        out.append(f"diagnoser: gaming: {d['gaming']}")
    return out


def _case_dict(tools, solve) -> dict | None:
    """The case a failed solve asked for, if it can be built at all (it may have failed there)."""
    from ace import fea

    try:
        return fea.build_case(tools.problem, solve["mesh"], solve["setup"], tools.task["qoi"]).to_dict()
    except Exception:  # noqa: BLE001
        return None


def _tokens(usage: list[dict]) -> dict | None:
    if not usage:
        return None
    t = {k: sum(int(u.get(k) or 0) for u in usage) for k in ("input", "output", "cache_write", "cache_read")}
    return {**t, "total": sum(t.values())}


def _empty_submission(store, problem_id: str, agent_id: str, meta: dict) -> str:
    """The record create_submission writes, for a run with no successful solve (which
    create_submission refuses). Without it the UI would show the agent as still running."""
    from feacore.runstore import SUBMISSION_SCHEMA, new_id, now_iso, write_json

    store.ensure()
    sid = new_id("sub")
    write_json(store.submissions_dir / f"{sid}.json", {
        "schema": SUBMISSION_SCHEMA, "submission_id": sid, "created_at": now_iso(), "problem_id": problem_id,
        "agent_id": agent_id, "run_ids": [], "levels": [], "order": None, "notes": "", "meta": meta,
        "analysis": {"order": None, "n_levels": 0, "qois": []}})
    return sid
