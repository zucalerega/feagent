"""Optional bridge to the feacore run store, so ACE runs show up live in the feacore web UI.

Off unless `python -m ace.run_loop --ui-runs-dir DIR` is used or FEACORE_RUNS_DIR is set. When on:

  each solve an agent makes    -> one run record (RunStore.record_result, with the viewer fields;
                                  a failed solve -> a run with status "failed" and the error)
  each finished agent run      -> one submission (RunStore.create_submission) of its successful
                                  solves, with the ACE checks, cited bullets and token counts

Both have agent_id "ace:<ace run id>" and source "ace". Their meta:
  run:        {source, loop_id, generation, ace_run_id, solve_index, mesh}
              solve_index counts from 0 (the agent's "solve n" is n - 1); mesh = the agent's mesh
              call arguments as given (h, order, refinements as a list of dicts, ...)
  submission: {source, loop_id, generation, ace_run_id, submitted_run_id, score, gates,
               setup_diagnostics, cited_bullets, tokens}
              submitted_run_id = the solve ACE scores (the submitted one, else the last successful
              one); gates = failed gates (empty = passed); a run that crashed (API or code error)
              has score None and an `error`, as ACE leaves it out of the scores.
loop_id is the name of the loop's output directory. Every function here turns any exception
into one warning: a missing or broken store never stops, slows or changes the loop.
"""

from __future__ import annotations

import functools
import os
import warnings
from pathlib import Path


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
            "ace_run_id": ctx["ace_run_id"], "solve_index": i, "mesh": tools.mesh_args[-1]}
    who = {"problem_id": tools.problem_id, "agent_id": f"ace:{ctx['ace_run_id']}", "source": "ace", "meta": meta}
    if result is not None:
        rid = store.record_result(result, mesh_path=solve["mesh"]["path"], **who)
    else:
        rid = store.create_run(case=_case_dict(tools, solve), status="failed", **who)["run_id"]
        store.update_run(rid, error=solve["result"].get("error"), finished_at=now_iso())
    solve["ui_run_id"] = rid


@_safe
def record_agent_run(ui: dict | None, run: dict, calibration: dict) -> None:
    """Record one agent run (a run_agent record) as a submission of its successful solves, as
    soon as it finishes. It is scored here with ace.checks.check_run, the same checks the loop
    runs after the whole generation (cheap today; if they ever get expensive, record the
    submissions after the loop's checks instead)."""
    if not ui:
        return
    from ace.checks import check_run
    from feacore.runstore import RunStore

    checks = check_run(run["problem_id"], run, calibration)
    crashed = bool(run.get("error"))
    meta = {
        "source": "ace", "loop_id": ui["loop_id"], "generation": run["generation"], "ace_run_id": run["run_id"],
        "submitted_run_id": (_scored_solve(run) or {}).get("ui_run_id"),
        "score": None if crashed else checks.get("score"),
        "gates": [checks["failed_gate"]] if checks.get("failed_gate") and not crashed else [],
        "setup_diagnostics": _setup_diagnostics(checks.get("setup") or {}),
        "cited_bullets": list(run.get("cited") or []),
        "tokens": _tokens(run.get("usage") or []),
    }
    if crashed:   # the last line of the traceback
        meta["error"] = (str(run["error"]).strip().splitlines() or ["error"])[-1][:500]
    store = RunStore(ui["root"])
    agent_id = f"ace:{run['run_id']}"
    run_ids = [s["ui_run_id"] for s in run.get("solves") or [] if s["result"].get("ok") and s.get("ui_run_id")]
    if run_ids:
        store.create_submission(run["problem_id"], agent_id, run_ids=run_ids, meta=meta)
    else:
        _empty_submission(store, run["problem_id"], agent_id, meta)


# --------------------------------------------------------------------------- helpers


def _case_dict(tools, solve) -> dict | None:
    """The case a failed solve asked for, if it can be built at all (it may have failed there)."""
    from ace import fea

    try:
        return fea.build_case(tools.problem, solve["mesh"], solve["setup"], tools.task["qoi"]).to_dict()
    except Exception:  # noqa: BLE001
        return None


def _scored_solve(run: dict) -> dict | None:
    """Same rule as ace.checks.check_run: the submitted solve, else the last successful one."""
    solves, sub = run.get("solves") or [], run.get("submitted")
    if sub is not None and 0 <= sub < len(solves) and solves[sub]["result"].get("ok"):
        return solves[sub]
    return next((s for s in reversed(solves) if s["result"].get("ok")), None)


def _setup_diagnostics(s: dict) -> list[str]:
    """ace.checks.bc_report (differences from the reference setup) as short strings."""
    out = [f"missing {x}" for x in s.get("missing", [])] + [f"unexpected {x}" for x in s.get("unexpected", [])]
    out += list(s.get("sign_flips", []))
    return out + [f"{k.removesuffix('_ok')} differs from the reference"
                  for k in ("formulation_ok", "thickness_ok", "material_ok") if s.get(k) is False]


def _tokens(usage: list[dict]) -> dict | None:
    if not usage:
        return None
    t = {k: sum(int(u.get(k) or 0) for u in usage) for k in ("input", "output", "cache_write", "cache_read")}
    return {**t, "total": sum(t.values())}


def _empty_submission(store, problem_id: str, agent_id: str, meta: dict) -> None:
    """The record create_submission writes, for a run with no successful solve (which
    create_submission refuses). Without it the UI would show the agent as still running."""
    from feacore.runstore import SUBMISSION_SCHEMA, new_id, now_iso, write_json

    store.ensure()
    sid = new_id("sub")
    write_json(store.submissions_dir / f"{sid}.json", {
        "schema": SUBMISSION_SCHEMA, "submission_id": sid, "created_at": now_iso(), "problem_id": problem_id,
        "agent_id": agent_id, "run_ids": [], "levels": [], "order": None, "notes": "", "meta": meta,
        "analysis": {"order": None, "n_levels": 0, "qois": []}})
