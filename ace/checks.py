"""Deterministic checks and the score. No LLM calls: these numbers are ground truth for the Reflector.

score = log10(e_uniform / e_agent), clipped to [SCORE_FLOOR, SCORE_CEIL]
  e_agent   relative error of the scored QoI from the solve the agent submitted (its last
            successful solve if it never called submit)
  e_uniform relative error of the best uniform mesh (linear or quadratic, whichever is better)
            with about node_cap nodes and the correct setup (calibrated once per problem,
            cached in ace/calibration.json)
Gates (mesh within the node cap, element quality, solve succeeded) score SCORE_FLOOR on failure.

TODO: mesh-refinement convergence check (re-run the agent's mesh recipe at h/2 and h/4 and use
feacore.verification.gci). Skipped for now: refined meshes break the node cap by design, so it
would only be a diagnostic, and the reference values already tell us the true error.
"""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

from ace import fea
from ace.config import MIN_QUALITY, SCORE_CEIL, SCORE_FLOOR, TASKS

CALIBRATION_PATH = Path(__file__).with_name("calibration.json")


# --------------------------------------------------------------------------- calibration


def reference_setup(problem) -> dict:
    """The problem's own (correct) setup, in the same shape the agent submits."""
    rc = problem.case("REF.msh").to_dict()
    return {
        "formulation": rc["analysis"]["formulation"],
        "thickness": rc["analysis"]["thickness"],
        "material": next(iter(rc["materials"].values())),
        "bcs": [{k: v for k, v in bc.items() if v is not None} for bc in rc["bcs"]],
        "loads": rc["loads"],
    }


def _uniform_error(problem, qoi, h, order, workdir) -> tuple[int, float]:
    mesh = fea.make_mesh(problem, Path(workdir) / "uniform.msh", h=h, order=order)
    sol = fea.run_solve(problem, mesh, reference_setup(problem), qoi)
    ref = fea.reference_value(problem, qoi)
    return mesh["nodes"], abs(sol["value"] - ref) / abs(ref)


def _calibrate_order(problem, qoi, cap, order, tmp) -> dict:
    lo, hi = problem.params["h0"] / 20, problem.params["h0"] * 20   # bisection on h
    for _ in range(25):
        h = math.sqrt(lo * hi)
        n, _ = _uniform_error(problem, qoi, h, order, tmp)
        lo, hi = (h, hi) if n > cap else (lo, h)
    nodes, errs = [], []
    for f in (1.0, 1.05, 1.1, 1.15, 1.2):
        n, e = _uniform_error(problem, qoi, hi * f, order, tmp)
        nodes.append(n)
        errs.append(e)
    return {"order": order, "h": hi, "nodes": nodes, "errors": errs,
            "e_uniform": math.exp(sum(math.log(e) for e in errs) / len(errs))}


def calibrate(problem_id: str) -> dict:
    """The best uniform mesh that fits the node cap, and its error.

    For each element order, find the uniform size that just fits the cap. Point and peak errors
    on coarse meshes jump around with node placement, so average (geometric mean) over five
    slightly coarser sizes that all fit. The baseline is the better of the two orders, so an
    agent gets no credit for merely choosing quadratic elements.
    """
    task = TASKS[problem_id]
    problem = fea.get_problem(problem_id)
    with tempfile.TemporaryDirectory() as tmp:
        per_order = [_calibrate_order(problem, task["qoi"], task["node_cap"], o, tmp) for o in (1, 2)]
    best = min(per_order, key=lambda c: c["e_uniform"])
    return {**best, "by_order": per_order}


def load_calibration(problem_ids: list[str]) -> dict:
    """Cached calibration; computes (and saves) any missing or stale entries."""
    cal = json.loads(CALIBRATION_PATH.read_text()) if CALIBRATION_PATH.exists() else {}
    changed = False
    for pid in problem_ids:
        stale = cal.get(pid, {}).get("node_cap") != TASKS[pid]["node_cap"] or "by_order" not in cal.get(pid, {})
        if pid not in cal or stale:
            cal[pid] = {"node_cap": TASKS[pid]["node_cap"], **calibrate(pid)}
            changed = True
    if changed:
        CALIBRATION_PATH.write_text(json.dumps(cal, indent=2) + "\n")
    return cal


# --------------------------------------------------------------------------- per-run checks


def score_from_error(e_agent: float, e_uniform: float) -> float:
    if e_agent <= 0:
        return SCORE_CEIL
    return max(SCORE_FLOOR, min(SCORE_CEIL, math.log10(e_uniform / e_agent)))


def bc_report(problem, setup: dict) -> dict:
    """Compare the agent's setup with the reference setup (diagnostic, not a gate: other
    setups can be equivalent, and a wrong one already shows up as a large error)."""
    ref = reference_setup(problem)

    def key(bc):
        # Point BCs (e.g. pinning one node against rigid-body motion) are matched by type and
        # components only: any suitable node is equivalent, and [-0.1, 0] == [-0.1, 0.0].
        where = bc.get("group") or f"a point (components {sorted(bc.get('components') or [])})"
        return where, bc.get("type")

    ref_bcs = {key(b): b for b in ref["bcs"]}
    got_bcs = {key(b): b for b in setup.get("bcs", [])}
    sign_flips = []
    for k in ref_bcs.keys() & got_bcs.keys():
        a, b = ref_bcs[k].get("value"), got_bcs[k].get("value")
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            flipped = a * b < 0
        elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
            try:   # vectors (tractions, forces): pointing in opposite directions
                flipped = sum(float(x) * float(y) for x, y in zip(a, b)) < 0
            except (TypeError, ValueError):   # expressions: not compared
                flipped = False
        else:
            flipped = False
        if flipped:
            sign_flips.append(f"{k[1]} on {k[0]}: reversed relative to the reference")
    mat, ref_mat = setup.get("material") or {}, ref["material"]
    return {
        "missing": [f"{t} on {g}" for g, t in ref_bcs.keys() - got_bcs.keys()],
        "unexpected": [f"{t} on {g}" for g, t in got_bcs.keys() - ref_bcs.keys()],
        "sign_flips": sign_flips,
        "formulation_ok": setup.get("formulation") == ref["formulation"],
        "thickness_ok": math.isclose(float(setup.get("thickness", 1.0)), ref["thickness"], rel_tol=1e-6),
        "material_ok": all(math.isclose(float(mat.get(k, 0)), ref_mat[k], rel_tol=1e-3) for k in ("E", "nu")),
    }


def check_run(problem_id: str, run: dict, calibration: dict) -> dict:
    """Checks + score for one agent run. `run` is the record produced by ace.agent.run_agent."""
    task = TASKS[problem_id]
    problem = fea.get_problem(problem_id)
    cal = calibration[problem_id]
    out = {
        "node_cap": task["node_cap"],
        "e_uniform": cal["e_uniform"],
        "n_meshes": len(run["meshes"]),
        "n_solves": len(run["solves"]),
        "meshed": any(m.get("ok") for m in run["meshes"]),
    }
    # The scored solve: the one the agent submitted, else its last successful one.
    sub = run.get("submitted")
    if sub is not None and 0 <= sub < len(run["solves"]) and run["solves"][sub]["result"].get("ok"):
        final = run["solves"][sub]
    else:
        final = next((s for s in reversed(run["solves"]) if s["result"].get("ok")), None)
    out["submitted"] = sub is not None
    if final is None:
        return {**out, "solved": False, "failed_gate": "no successful solve", "score": SCORE_FLOOR}

    mesh = final["mesh"]
    ref = fea.reference_value(problem, task["qoi"])
    rel_err = abs(final["result"]["value"] - ref) / abs(ref)
    out.update({
        "solved": True,
        "nodes": mesh["nodes"],
        "order": mesh["order"],
        "min_quality": mesh["min_quality"],
        "within_cap": mesh["nodes"] <= task["node_cap"],
        "quality_ok": (mesh["min_quality"] or 0) >= MIN_QUALITY,
        "qoi_value": final["result"]["value"],
        "rel_error": rel_err,
        "setup": bc_report(problem, final["setup"]),
    })
    if not out["within_cap"]:
        return {**out, "failed_gate": "over node cap", "score": SCORE_FLOOR}
    if not out["quality_ok"]:
        return {**out, "failed_gate": f"min element quality < {MIN_QUALITY}", "score": SCORE_FLOOR}
    return {**out, "failed_gate": None, "score": score_from_error(rel_err, cal["e_uniform"])}
