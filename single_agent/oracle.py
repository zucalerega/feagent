"""A scripted analyst with the right setup: the evaluator's sanity check and the loop's non-LLM baseline.

It is given the REFERENCE setup (which no agent ever sees) and runs a fixed recipe through the same
harness tools an agent uses: one quadratic mesh refined at the re-entrant edges, `levels` systematic
refinements by `factor`, a solve on each, the study tool, and a report that takes each answer from the
study (the extrapolated value when the convergence is monotone near the theoretical order, else the finest
value) with the GCI as its uncertainty, and marks a peak stress singular when the study says it diverges.

    python -m single_agent.oracle abc_00000003_bc0 [--h 0.006] [--levels 3] [--factor 1.4] [--out DIR]

What it shows: a perfect idealization plus a by-the-book study under the agent's node cap; how far the
setup-independent parts (mesh, study, answers) can go. The mock agent (single_agent.mock) uses it too.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from single_agent import config, workspace


def reference_setup(problem_id: str) -> dict:
    """The gold setup and its QoIs, in the solve tool's argument format."""
    from feacore.problems import registry
    p = registry.get(problem_id)
    c = p.case("X.msh").to_dict()
    mat = next(iter(c["materials"].values()))
    bcs = [{k: v for k, v in b.items() if v is not None and k in ("type", "group", "point", "components", "value",
                                                                  "where")} for b in c["bcs"]]
    qois = [{k: v for k, v in q.items() if v is not None and k in ("name", "type", "field", "group", "point",
                                                                   "component")} for q in c["qois"]]
    return {"material": {"E": mat["E"], "nu": mat["nu"]}, "bcs": bcs, "loads": [], "qois": qois}


def run(session, problem_id: str, setup: dict | None = None, h: float = 0.006, levels: int = 3,
        factor: float = 1.4, edge_size: float | None = None) -> dict:
    """Drive `session` (a harness.Session in the problem's workspace) through the recipe; returns a summary.
    `setup` defaults to the reference setup."""
    setup = setup or reference_setup(problem_id)
    geo, _ = session.call("inspect_geometry", {})
    reent = list(geo.get("reentrant_edges") or [])[:12]
    es = edge_size or h / 3
    refinements = [{"edge": int(e), "size": es, "radius": 3 * h} for e in reent]
    meshes = []
    out, err = session.call("mesh", {"h": h, "order": 2, "refinements": refinements})
    if err:
        return {"error": f"mesh failed: {out}"}
    meshes.append(out["mesh"])
    # Fit the levels under the node cap: nodes grow about like factor^2.5 per level on these parts (fixed
    # features and the size floor keep it below factor^3), so shrink the factor when it would overshoot.
    cap = session.limits["node_cap"]
    n0 = out["nodes"]
    if levels > 1:
        factor = max(1.2, min(factor, (0.85 * cap / n0) ** (1 / (2.5 * (levels - 1)))))
    for _ in range(levels - 1):
        out, err = session.call("refine_mesh", {"mesh": meshes[-1], "factor": factor})
        if err:
            break
        if out.get("nodes", 0) > cap:                 # overshot: try once more with a gentler step
            out, err = session.call("refine_mesh", {"mesh": meshes[-1], "factor": max(1.15, factor ** 0.6)})
            if err or out.get("nodes", 0) > cap:
                break
        meshes.append(out["mesh"])
    solves = []
    for m in meshes:
        out, err = session.call("solve", {"mesh": m, **setup})
        if not err:
            solves.append(out["solve"])
    study = {}
    if len(solves) >= 2:
        study, _ = session.call("study", {"solves": solves})
    key = workspace.load_key(problem_id)
    finest = session.state["solves"][solves[-1] - 1]["qois"] if solves else {}
    answers = []
    for q in key["questions"]:
        row = (study.get("qois") or {}).get(q["qoi"], {})
        value, U, status = finest.get(q["qoi"]), row.get("gci_fine"), "converged"
        conv, p, pt = row.get("convergence"), row.get("p_observed"), row.get("p_theory")
        peak = q["qoi"] == "vm_max"          # a peak stress: the only kind of answer the recipe may call singular
        if peak and (conv in ("divergent", "oscillatory_divergent") or (p is not None and pt and p < 0.5 * pt)):
            status, U = "singular", None
        elif conv == "monotone" and p and pt and 0.5 * pt <= p <= 1.5 * pt + 0.5 and row.get("extrapolated"):
            value = row["extrapolated"]
        elif conv in ("oscillatory", "divergent", "oscillatory_divergent") and row.get("values"):
            vals = row["values"][-3:]       # no reliable order: the finest value, the spread as the uncertainty
            U = (max(vals) - min(vals)) / max(abs(value or 1e-300), 1e-300)
        if U is None and status == "converged" and row.get("change") is not None:
            U = 2 * row["change"]
        answers.append({"id": q["id"], "quantity": q["qoi"], "value": value, "uncertainty": U, "status": status})
    rep, err = session.call("submit_report", {
        "idealization": [{"design_item": "reference setup (oracle)", "faces": [b.get("group") for b in setup["bcs"]],
                          "model": "as the reference"}],
        "final_solve": solves[-1] if solves else 0, "study_solves": solves, "answers": answers,
        "summary": f"scripted oracle: {len(solves)} meshes, factor {factor}"})
    return {"meshes": meshes, "solves": solves, "submitted": not err}


def run_problem(problem_id: str, root: Path, **kw) -> tuple[dict, dict]:
    """Oracle run in a fresh workspace; returns (run record, scorecard)."""
    from single_agent.evaluate import evaluate
    from single_agent.harness import Session
    ws = workspace.make(problem_id, root)
    t0 = time.time()
    s = Session(ws)
    summary = run(s, problem_id, **kw)
    rec = {"problem_id": problem_id, "label": ws.name, "run_id": f"oracle-{problem_id}", "workspace": str(ws),
           "agent": "oracle", "summary": summary, "session": s.state, "usage": [], "turns": 0,
           "wall_s": round(time.time() - t0, 1)}
    return rec, evaluate(problem_id, rec)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("problems", nargs="*", default=None)
    ap.add_argument("--h", type=float, default=0.006)
    ap.add_argument("--levels", type=int, default=3)
    ap.add_argument("--factor", type=float, default=1.4)
    ap.add_argument("--out", default="sa_runs/oracle")
    a = ap.parse_args()
    from single_agent.evaluate import summary_line
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for pid in a.problems or workspace.packaged():
        rec, card = run_problem(pid, out / "work", h=a.h, levels=a.levels, factor=a.factor)
        (out / f"{pid}.json").write_text(json.dumps({"run": rec, "scorecard": card}, indent=1, default=float))
        print(summary_line(card), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
