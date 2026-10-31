"""Run a trained swarm on a problem: predictions, no learning.

    python -m ace.infer --swarm <exported_dir> --problem <problem_id> [--agents a0,a3 | all] [--out <dir>] [--mock]

<exported_dir> comes from `python -m ace.export_swarm ace_runs/<run> <exported_dir>`. Each chosen
agent runs on the problem in its own process with its saved context (ace.export_swarm.agent_context:
the context training would build for it there), the same tools and the same agent loop as in
training (ace.agent.run_agent). There is no teacher, reflection, organizer or memory update: the
exported folder is only read.

The problem can be anything feacore registers (ace.fea.get_problem): a training task, a NAFEMS
benchmark, a gold ABC part. One the loop's config does not list gets its gold scored quantity (or
its first quantity with a reference, else its first quantity) and a node cap of DEFAULT_NODE_CAP.

Output in <out> (default ace_runs/infer-<problem>-<time>):
  results.json         per agent: its submitted solve's statement quantities, mesh, the
                       reference-free error estimate, equilibrium, image paths (and the field
                       score when there is one); the consensus across agents
  summary.md           the same as tables
  runs/<agent>.json    each agent's whole run record (its reasoning trace, every mesh and solve)
  context/<agent>.md   the context each agent was given
  work/<agent>/        its meshes and images (final_mesh.png, final_stress.png: the submitted solve)

Scoring. With no ground truth, agreement is the signal: the consensus gives, for each statement
quantity, the median of the agents' values and their spread, and names the agent whose own error
estimate (ZZ, relative energy norm) is smallest. When the problem has references and the field
caches (ace/field_refs/<problem>.npz, _auto.json, _points.json) exist, the agents get the field-mode
task statement, as in training, and each run is also scored in field mode (ACE_SCORE=field):
whole-field error against the fine reference and score against the automesher.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import statistics
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

# NOTE: no ace.config import at module level. config reads ACE_SCORE when first imported, so main()
# decides the scoring mode and sets ACE_SCORE before anything imports it (the agent processes are
# spawned afterwards and inherit it).

DEFAULT_NODE_CAP = 40000      # for a problem the loop's config does not list (about the gold L1 meshes)


# --------------------------------------------------------------------------- which problem, which mode


def field_ready(problem_id: str) -> bool:
    """True when the problem can be scored in field mode without building anything: it has
    references and its reference, automesher and check-point caches exist (ace.field_error)."""
    from feacore.problems import registry
    cache = Path(os.environ.get("ACE_FIELD_REFS") or Path(__file__).with_name("field_refs"))
    cached = all((cache / f"{problem_id}{suffix}").exists() for suffix in (".npz", "_auto.json", "_points.json"))
    return cached and bool(registry.get(problem_id).references())


def task_entry(problem_id: str) -> dict:
    """The problem's entry in config.TASKS (scored quantity and node cap); one is made up for a
    problem the config does not list, preferring a quantity that has a reference."""
    from ace import config, fea
    if problem_id in config.TASKS:
        return config.TASKS[problem_id]
    p = fea.get_problem(problem_id)
    if hasattr(p, "scored_qoi"):                  # gold parts say which quantity they certify best
        qoi = p.scored_qoi()
    else:
        with_ref = {r.qoi for r in p.references()}
        qoi = next((q.name for q in p.qois() if q.name in with_ref), p.qois()[0].name)
    return {"qoi": qoi, "node_cap": DEFAULT_NODE_CAP}


# --------------------------------------------------------------------------- one agent (in its own process)


def _worker(job: dict) -> dict:
    """One agent's run, then its prediction. In a child process: gmsh is process-global."""
    from ace import config
    from ace.agent import run_agent
    from ace.llm import make_llm

    # The spawned child re-imports config: register the problem here too (agent and checks look it up).
    config.TASKS.setdefault(job["problem"], job["task"])
    usage: list = []
    try:
        run = run_agent(job["problem"], job["run_id"], 0, job["context"], Path(job["workdir"]),
                        make_llm(job["mock"]), usage=usage)
    except Exception:  # noqa: BLE001  one broken agent must not take the others down
        return {"agent": job["agent"], "problem_id": job["problem"], "run_id": job["run_id"],
                "error": traceback.format_exc(), "usage": usage}
    run["agent"] = job["agent"]
    try:
        run["prediction"] = predict(run, Path(job["workdir"]))
        if job["field"]:
            from ace.checks import check_run
            run["checks"] = check_run(job["problem"], run)
    except Exception:  # noqa: BLE001  keep the run; say why there is no prediction
        run["prediction"] = {"error": traceback.format_exc()}
    return run


def predict(run: dict, workdir: Path) -> dict:
    """What the agent predicts: its submitted solve (else its last successful one, the rule the
    scoring uses) re-solved once more, so that the statement quantities, the error estimate and
    the pictures all come from the very solve it chose (the solve is direct, so it is the same
    answer)."""
    from ace import fea, solve_feedback

    solves = run["solves"]
    i = run.get("submitted")
    submitted = i is not None and solves[i]["result"].get("ok")
    if not submitted:
        i = next((j for j in reversed(range(len(solves))) if solves[j]["result"].get("ok")), None)
    if i is None:
        return {"error": "no successful solve"}
    s = solves[i]
    mesh, problem, qoi = s["mesh"], fea.get_problem(run["problem_id"]), s["result"]["qoi"]
    res = fea.run_solve(problem, mesh, s["setup"], qoi, keep_result=True)
    result = res.pop("_result")
    fb = solve_feedback.feedback(problem, mesh, s["setup"], result)
    images = {kind: str(solve_feedback.render(problem, mesh, result, kind, "iso", workdir / f"final_{kind}.png"))
              for kind in ("mesh", "stress")}
    return {
        "solve": i + 1, "submitted": bool(submitted),
        "scored_quantity": {"name": qoi, "value": res["value"]},
        "qois": fb["qois"],                                          # every statement quantity
        "mesh": {"nodes": mesh["nodes"], "elements": mesh.get("elements"), "dofs": fb["dofs"],
                 "order": mesh["order"], "h": mesh.get("h_used"), "min_quality": mesh["min_quality"],
                 "folded_cells": mesh.get("folded_cells"), "request": mesh.get("request")},
        "error_estimate": fb["error_estimate"],                      # ZZ: reference-free
        "equilibrium": fb["equilibrium"],
        "max_von_mises": fb["max_von_mises"], "max_displacement": fb["max_displacement"],
        "notes": fb.get("notes", []),
        "images": images,
    }


# --------------------------------------------------------------------------- across agents


def consensus(preds: dict[str, dict]) -> dict:
    """Agreement between agents, the signal when there is no ground truth. For each statement
    quantity: the median of the agents' values, their range, and the relative spread
    (max - min) / |median|. And the agent most confident by its own error estimate (smallest ZZ
    relative energy-norm error)."""
    ok = {a: p for a, p in preds.items() if p and "qois" in p}
    names = list(dict.fromkeys(q for p in ok.values() for q in p["qois"]))
    per_qoi = {}
    for q in names:
        vals = {a: p["qois"][q] for a, p in ok.items() if p["qois"].get(q) is not None}
        if not vals:
            continue
        med = statistics.median(vals.values())
        lo, hi = min(vals.values()), max(vals.values())
        per_qoi[q] = {"median": med, "min": lo, "max": hi, "n": len(vals),
                      "spread_rel": (hi - lo) / abs(med) if med else None, "values": vals}
    eta = {a: p["error_estimate"].get("eta_rel") for a, p in ok.items()}
    eta = {a: e for a, e in eta.items() if e is not None}
    best = min(eta, key=eta.get) if eta else None
    return {"agents_with_prediction": sorted(ok), "qois": per_qoi,
            "most_confident": {"agent": best, "eta_rel": eta.get(best)} if best else None}


def _f(x) -> str:
    return "-" if x is None else (f"{x:.4g}" if isinstance(x, float) else str(x))


def summary_md(problem: str, swarm: dict, mode: str, rows: list[dict], cons: dict, usd: float) -> str:
    field = mode == "field"
    head = ["agent", "solve", "submitted", "nodes", "DOFs", "order", "h", "ZZ error est.", "scored quantity"]
    head += ["field error", "score (A/c)"] if field else []
    lines = [f"# Inference: {problem}",
             f"Swarm {swarm['run']} (generation {swarm['generation']}, trained on {', '.join(swarm['tasks'])}). "
             f"Task statement and scoring mode: {mode}. Spent ${usd:.2f}.", "",
             "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in rows:
        p = r.get("prediction") or {}
        if r.get("error") or "error" in p:
            lines.append(f"| {r['agent']} | failed: {(r.get('error') or p['error']).strip().splitlines()[-1][:80]} |")
            continue
        m, sq = p["mesh"], p["scored_quantity"]
        cells = [r["agent"], p["solve"], "yes" if p["submitted"] else "no (last solve)", m["nodes"], m["dofs"],
                 m["order"], _f(m["h"]), _f(p["error_estimate"].get("eta_rel")), f"{sq['name']} = {_f(sq['value'])}"]
        if field:
            c = r.get("checks") or {}
            base = c["accuracy"] * c["efficiency"] if c.get("valid") else c.get("score")
            cells += [_f(c.get("rel_error")), _f(base)]
        lines.append("| " + " | ".join(map(str, cells)) + " |")
    lines += ["", "## Consensus (statement quantities across agents)", "",
              "| quantity | median | min | max | spread (max-min)/abs(median) | agents |", "|---|---|---|---|---|---|"]
    lines += [f"| {q} | {_f(c['median'])} | {_f(c['min'])} | {_f(c['max'])} | {_f(c['spread_rel'])} | {c['n']} |"
              for q, c in cons["qois"].items()]
    mc = cons["most_confident"]
    lines += ["", f"Most confident by its own error estimate: {mc['agent']} (ZZ eta_rel {_f(mc['eta_rel'])})."
              if mc else "No agent produced a solve."]
    lines += ["", "## Images (submitted solve)", ""]
    lines += [f"- {r['agent']}: mesh `{p['images']['mesh']}`, stress `{p['images']['stress']}`"
              for r in rows if "images" in (p := r.get("prediction") or {})]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- main


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--swarm", type=Path, required=True, help="an exported swarm (ace.export_swarm)")
    ap.add_argument("--problem", required=True, help="any feacore problem id, e.g. nafems_le1 or a gold part")
    ap.add_argument("--agents", default="all", help="comma-separated agents (a0,a3) or all")
    ap.add_argument("--out", type=Path, help="output dir (default ace_runs/infer-<problem>-<time>)")
    ap.add_argument("--mock", action="store_true", help="canned LLM responses, no API calls")
    args = ap.parse_args(argv)

    # Scoring mode first, before ace.config is imported anywhere (see the note at the top).
    mode = "field" if field_ready(args.problem) else "point"
    os.environ["ACE_SCORE"] = mode
    from ace import config
    from ace.cost import CostLog
    from ace.export_swarm import agent_context

    swarm = json.loads((args.swarm / "manifest.json").read_text())
    agents = swarm["agents"] if args.agents == "all" else [a.strip() for a in args.agents.split(",") if a.strip()]
    unknown = [a for a in agents if a not in swarm["agents"]]
    if unknown:
        raise SystemExit(f"unknown agents {unknown}; this swarm has {', '.join(swarm['agents'])}")
    task = task_entry(args.problem)
    config.TASKS.setdefault(args.problem, task)
    out = (args.out or Path("ace_runs") / f"infer-{args.problem}-{time.strftime('%Y%m%d-%H%M%S')}"
           + ("-mock" if args.mock else "")).resolve()
    for sub in ("runs", "context", "work"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    print(f"output: {out}\nproblem {args.problem} (scored quantity {task['qoi']}, node cap {task['node_cap']}), "
          f"mode {mode}; agents {', '.join(agents)}")

    jobs = []
    for a in agents:
        context = agent_context(args.swarm, a, args.problem)
        (out / "context" / f"{a}.md").write_text(context)
        jobs.append({"agent": a, "problem": args.problem, "task": task, "context": context, "mock": args.mock,
                     "run_id": f"infer-{args.problem}-{a}", "workdir": str(out / "work" / a), "field": mode == "field"})

    # One process per agent, all at once, as training runs a generation.
    cost = CostLog(out / "cost.jsonl", float("inf"))
    with ProcessPoolExecutor(len(jobs), mp_context=mp.get_context("spawn")) as pool:
        runs = list(pool.map(_worker, jobs))
    for r in runs:
        cost.add(r.get("usage") or [])
        (out / "runs" / f"{r['agent']}.json").write_text(json.dumps(r, indent=1, default=str))

    if mode == "field":         # score against the automesher, among these runs only
        from ace.checks import score_population
        scored = [dict(r, generation=0) for r in runs if r.get("checks")]
        if scored:
            score_population(scored)
            for r, s in zip([r for r in runs if r.get("checks")], scored):
                r["checks"] = s["checks"]

    preds = {r["agent"]: r.get("prediction") for r in runs}
    cons = consensus(preds)
    per_agent = {}
    for r in runs:
        entry = {"run_id": r["run_id"], "error": r.get("error"), "stop": r.get("stop"), "turns": r.get("turns"),
                 "prediction": r.get("prediction")}
        if mode == "field" and r.get("checks"):
            c = r["checks"]
            entry["field_score"] = {k: c.get(k) for k in ("rel_error", "field_error", "probe_error", "dofs", "gain",
                                                           "accuracy", "efficiency", "valid", "failed_gate",
                                                           "auto_error", "auto_dofs")}
            entry["field_score"]["base_score"] = c["accuracy"] * c["efficiency"] if c.get("valid") else 0.0
        per_agent[r["agent"]] = entry
    results = {"problem": args.problem, "task": task, "mode": mode, "swarm": swarm, "agents": per_agent,
               "consensus": cons, "usd": cost.total}
    (out / "results.json").write_text(json.dumps(results, indent=1, default=str))
    md = summary_md(args.problem, swarm, mode, runs, cons, cost.total)
    (out / "summary.md").write_text(md)
    print("\n" + md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
