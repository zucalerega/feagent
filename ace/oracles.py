"""Layer 1 of diagnosis: physics oracles. Pure code, no LLM; the Reflector/diagnoser must never
contradict them.

Each oracle looks at one agent run's scored solve (its mesh + setup) and returns
    {"status": "pass" | "fail" | "unavailable", "values": {...}, "why": "<one line>"}

Most oracles are thin wrappers over the team's feacore.diagnostics (see docs/failure-modes.md),
used with its authors' definitions and thresholds:

  equilibrium      feacore's `equilibrium` check (sum of reactions + sum of applied loads = 0).
                   Note: with reactions computed as K u - f this balance holds for any setup that
                   solves, so it catches broken solves, not a wrong load (spec_compliance does that).
  mesh_convergence feacore.diagnostics.verify.refinement_study on the agent's own mesh request,
                   every size x2 and x0.5 (radii unchanged): does the scored quantity settle?
  singularity      feacore's singularity flags + the refinement study's `singular` class for the
                   scored quantity. Point LOADS are not expressible in this harness (unavailable).
  sanity_bounds    feacore's linear-validity checks (displacement / strain far too large, the
                   classic symptom of a unit error).
  symmetry         a symmetry plane may carry only a normal reaction: the tangential reaction on
                   each plane the problem declares symmetric must be ~0.
  spec_compliance  setup vs the problem's reference setup (ace.checks.bc_report, extended with load
                   magnitude) + feacore's precheck (rigid-body modes, no load).
  gaming           did the run make the problem easier: a load missing, or extra supports added?
  known_answer     relative error vs the reference value, against its stated tolerance.
"""

from __future__ import annotations

import functools
import math
import os
import tempfile
from pathlib import Path

import numpy as np
from feacore.case import CaseError
from feacore.diagnostics import diagnose
from feacore.diagnostics.view import SolutionView
from feacore.diagnostics.verify import refinement_study

from ace import fea
from ace.checks import bc_report, reference_setup
from ace.config import TASKS

LOAD_TYPES = {"pressure", "traction", "force"}
SUPPORT_TYPES = {"fixed", "displacement", "symmetry"}
SYMMETRY_TOL = 1e-3
STUDY_MAX_NODES_3D = 8000    # skip the refinement study above this (its fine level is ~8x larger)          # tangential reaction / applied load on a symmetry plane


@functools.lru_cache(maxsize=None)
def baseline_fail_flags(problem_id: str) -> frozenset:
    """feacore fail-severity flags that fire on the problem's OWN reference setup (on a mesh at the
    problem's default size). They are properties of the benchmark or limits of a check (e.g. Cook's
    membrane is posed with E = 1 and huge displacements; LE11's axisymmetric thermal load trips the
    equilibrium check), so they say nothing about an agent and are not held against it."""
    problem = fea.get_problem(problem_id)
    with tempfile.TemporaryDirectory() as tmp:
        if "h0" in problem.params:
            m = fea.make_mesh(problem, Path(tmp) / "ref.msh", h=problem.params["h0"], order=2)
        else:    # gold CAD problems have no default size: use their own coarsest study mesh
            m = {"path": str(problem.mesh_file(0)), "order": 2}
        rep, _ = diagnose(fea.build_case(problem, m, reference_setup(problem), TASKS[problem_id]["qoi"]))
    return frozenset(f.id for f in rep.flags if f.severity == "fail")


def _agent_fails(rep, problem_id: str, ids=None):
    """Fail flags that count against the agent (baseline flags excluded)."""
    base = baseline_fail_flags(problem_id)
    return [f for f in rep.flags if f.severity == "fail" and f.id not in base and (ids is None or f.id in ids)]


def result(status: str, why: str, **values) -> dict:
    return {"status": status, "why": why, "values": values}


def unavailable(why: str) -> dict:
    return result("unavailable", why)


def _flags(rep, *ids):
    return [f for f in rep.flags if f.id in ids]


# --------------------------------------------------------------------------- the oracles


def equilibrium(rep, res, problem_id: str) -> dict:
    if res is None:
        return unavailable("no solution to check (the setup failed precheck or the solver failed)")
    if "equilibrium" in baseline_fail_flags(problem_id):
        return unavailable("feacore's equilibrium check also fails on this problem's own reference setup, "
                           "so it says nothing about the agent")
    bad = _flags(rep, "equilibrium", "linear_residual", "non_finite")
    err = rep.metrics.get("equilibrium_error")
    if bad:
        return result("fail", "; ".join(f.message for f in bad), equilibrium_error=err)
    return result("pass", "reactions balance the applied loads", equilibrium_error=err)


def mesh_convergence(problem, request: dict | None, setup: dict, qoi: str, target_err: float,
                     nodes: int | None = None) -> dict:
    """Refinement study of the agent's own setup. Fails if the scored quantity is singular or
    oscillating, or its estimated error is above the accuracy target (the reference tolerance)
    (under a node cap, demanding the reference tolerance would fail nearly every run).
    ACE_SKIP_CONVERGENCE=1 skips it (the --deep loop: the study's extra solves are the slowest part
    of scoring, and the field error already measures the mesh)."""
    if os.environ.get("ACE_SKIP_CONVERGENCE") == "1":
        return unavailable("skipped (ACE_SKIP_CONVERGENCE)")
    if not request:
        return unavailable("the mesh request was not recorded for this run")
    if problem.dim == 3 and (nodes or 0) > STUDY_MAX_NODES_3D:
        return unavailable(f"mesh too large for a 3D refinement study (> {STUDY_MAX_NODES_3D} nodes; the "
                           "finer level would have ~8x more)")
    with tempfile.TemporaryDirectory() as tmp:
        def case_at(s: float):
            refs = [dict(r, size=r["size"] * s) for r in request.get("refinements") or []]
            m = fea.make_mesh(problem, Path(tmp) / f"m{s}.msh", h=request["h"] * s,
                              order=request.get("order", 1), refinements=refs)
            if not m["ok"]:
                raise CaseError(m["error"])
            return fea.build_case(problem, m, setup, qoi)
        try:
            study = refinement_study(case_at)
        except Exception as e:  # noqa: BLE001  any failure of the extra solves: no verdict
            return unavailable(f"refinement study failed: {type(e).__name__}: {e}")
    q = study["qois"][qoi]
    coarse, agent, fine = q["values"]
    step = abs(fine - agent) / max(abs(agent), 1e-300)   # change from one refinement of the agent's mesh
    vals = dict(kind=q["kind"], values=q["values"], est_err=q["est_err"], step_change=step,
                dofs=study["dofs"], threshold=target_err)
    if q["kind"] in ("singular", "oscillatory"):
        # Three point values rarely show a clean convergence ratio, so these classes are common on
        # good meshes too (e.g. 90.8 -> 91.4 -> 92.2). Judge by how much one refinement moves the
        # answer instead; a real singularity is the singularity oracle's job (it needs a source).
        if step > target_err:
            return result("fail", f"one refinement changes {qoi} by {step:.1%}, more than the accuracy target "
                                  f"({target_err:.1%}): the mesh is not resolving it ({q['kind']})", **vals)
        return result("pass", f"{qoi} moves {step:.1%} under one refinement ({q['kind']} pattern, small change)", **vals)
    if q["est_err"] is not None and q["est_err"] > target_err:
        return result("fail", f"estimated discretisation error {q['est_err']:.1%} is above the accuracy target "
                              f"({target_err:.1%})", **vals)
    est = "unknown" if q["est_err"] is None else f"{q['est_err']:.2%}"
    return result("pass", f"{qoi} settles under refinement ({q['kind']}, est. error {est})", **vals)


# feacore flags that name a physical source of infinite stress (a point/edge support, or a peak
# sitting at a support rim or a load-patch edge)
SINGULAR_SOURCES = {"point_constraint", "peak_at_support", "peak_at_load_edge"}


NEAR = 0.1                    # a singular source counts if within this fraction of the model size


def _near_sources(problem, rep, setup: dict, qoi_point, view) -> list[str]:
    """Singular sources close to the scored point. A source far away (e.g. a line support across
    the part) makes stresses infinite THERE, not at the scored point. Distances are measured to the
    flagged location, or for a point/edge support to its actual mesh nodes."""
    if qoi_point is None:
        return [f.id for f in rep.flags if f.id in SINGULAR_SOURCES]
    bb = fea.geometry_summary(problem)["bbox"]
    size = math.dist(bb[:3], bb[3:])
    q = np.array((list(qoi_point) + [0.0, 0.0])[:view.dim])
    low_dim = {g["name"] for g in fea.geometry_summary(problem)["groups"] if g["dim"] <= problem.dim - 2}
    out = []
    for f in rep.flags:
        if f.id not in SINGULAR_SOURCES:
            continue
        if f.where is not None:
            d = float(np.linalg.norm(np.array(f.where[:view.dim]) - q))
        else:                             # point_constraint: nearest node of a point/edge support
            locs = [float(np.linalg.norm(np.array((list(b["point"]) + [0.0, 0.0])[:view.dim]) - q))
                    for b in setup.get("bcs", []) if b.get("type") in ("fixed", "displacement") and b.get("point")]
            for b in setup.get("bcs", []):
                if b.get("type") in ("fixed", "displacement") and b.get("group") in low_dim:
                    xyz = view.doflocs[:, view.group_dofs(b["group"])]
                    locs.append(float(np.min(np.linalg.norm(xyz - q[:, None], axis=0))))
            d = min(locs) if locs else math.inf
        if d <= NEAR * size:
            out.append(f.id)
    return out


def singularity(rep, conv: dict, qoi: str, problem=None, setup: dict | None = None, qoi_point=None, res=None) -> dict:
    """A quantity that keeps growing under refinement is a singularity only if there is a source
    for one near it; otherwise it is more likely a mesh too coarse to be converging yet."""
    notes = [f"{f.id}: {f.message}" for f in rep.flags if f.category == "singularity"]
    kind = conv.get("values", {}).get("kind")
    sources = _near_sources(problem, rep, setup or {}, qoi_point, SolutionView.from_result(res)) \
        if problem is not None and res is not None else [f.id for f in rep.flags if f.id in SINGULAR_SOURCES]
    if kind == "singular" and sources:
        return result("fail", f"{qoi} grows without bound under refinement and the setup has a singular "
                              f"source ({', '.join(sorted(set(sources)))}): its value is not a real peak", flags=notes)
    if conv["status"] == "unavailable":
        return result("unavailable", "no refinement study; feacore flags only: " + ("; ".join(notes) or "none"),
                      flags=notes, point_loads="not expressible in this harness")
    return result("pass", "the scored quantity converges" + (f" (other flags: {len(notes)})" if notes else ""),
                  flags=notes, point_loads="not expressible in this harness")


def sanity_bounds(rep, res, problem_id: str) -> dict:
    if res is None:
        return unavailable("no solution to check")
    base = baseline_fail_flags(problem_id)
    bad = [f for f in rep.flags if f.category == "validity" and f.severity == "fail" and f.id not in base]
    if base & {"large_displacement", "large_strain"} and not bad:
        return unavailable("this problem's own reference setup already trips feacore's validity limits "
                           f"({', '.join(sorted(base))}), so they cannot judge the agent")
    vals = {k: rep.metrics.get(k) for k in ("disp_ratio", "max_strain")}
    if bad:
        return result("fail", "; ".join(f.message for f in bad), **vals)
    return result("pass", "displacements and strains are in the small-strain range", **vals)


def symmetry(problem, res, setup: dict) -> dict:
    ref = reference_setup(problem)
    planes = [b["group"] for b in ref["bcs"] if b["type"] == "symmetry" and b.get("group")]
    if not planes:
        return unavailable("the problem declares no symmetry planes")
    if res is None:
        return unavailable("no solution to check")
    groups = {g["name"]: g for g in fea.geometry_summary(problem)["groups"]}
    v = SolutionView.from_result(res)                 # feacore's view: reactions R, dofs per group
    agent_supports = {b.group for b in v.case.bcs if b.type in SUPPORT_TYPES and b.group}
    # Scale: the applied load, or the support reactions when the load is thermal (a temperature
    # field applies no net force, so the applied resultant is ~0 and cannot be the yardstick).
    scale = max(float(np.abs([v.f[idx].sum() for idx in v.comp]).sum()),
                float(np.abs(v.R[np.concatenate([v.group_dofs(g) for g in agent_supports])]).sum()) if agent_supports else 0.0,
                1e-300)
    worst, out = 0.0, {}
    for name in planes:
        bb = groups[name]["bbox"]
        extent = [bb[i + 3] - bb[i] for i in range(problem.dim)]
        normal = min(range(problem.dim), key=lambda i: abs(extent[i]))      # the flat direction
        # Only this plane's own nodes: nodes it shares with another support (e.g. an edge also on
        # a clamped face) are held tangentially by that support, not by the plane.
        others = [v.group_dofs(g) for g in agent_supports if g != name]
        own = np.setdiff1d(v.group_dofs(name), np.concatenate(others)) if others else v.group_dofs(name)
        r = [abs(float(v.R[np.intersect1d(own, v.comp[k])].sum())) for k in range(problem.dim) if k != normal]
        out[name] = max(r) / scale
        worst = max(worst, out[name])
    if worst > SYMMETRY_TOL:
        return result("fail", "a symmetry plane carries a tangential reaction: it was clamped, not "
                              "given a symmetry condition", tangential_over_load=out)
    return result("pass", "symmetry planes carry only normal reactions", tangential_over_load=out)


def _load_magnitudes(setup: dict) -> dict:
    out = {}
    for b in setup.get("bcs", []):
        if b.get("type") in LOAD_TYPES and b.get("group"):
            v = b.get("value")
            try:
                mag = math.hypot(*v) if isinstance(v, list) else abs(float(v))
            except (TypeError, ValueError):
                continue
            out[(b["group"], b["type"])] = mag
    return out


def spec_compliance(problem, setup: dict, rep) -> dict:
    r = bc_report(problem, setup)
    ref_loads, got_loads = _load_magnitudes(reference_setup(problem)), _load_magnitudes(setup)
    problems = []
    for key, mag in ref_loads.items():
        if key not in got_loads:
            problems.append(f"load {key[1]} on {key[0]} is missing")
        elif mag > 0 and not 0.99 <= got_loads[key] / mag <= 1.01:
            problems.append(f"load {key[1]} on {key[0]} is {got_loads[key] / mag:.3g}x the specified magnitude")
    problems += [f"load {t} on {g} is not in the problem statement" for g, t in got_loads.keys() - ref_loads.keys()]
    problems += r["sign_flips"]
    for k, what in (("formulation_ok", "formulation"), ("thickness_ok", "thickness"), ("material_ok", "material (E, nu)")):
        if not r[k]:
            problems.append(f"wrong {what}")
    problems += [f"{f.id}: {f.message}" for f in rep.flags if f.category == "setup" and f.severity == "fail"
                 and f.id not in baseline_fail_flags(problem.id)]
    # Supports written differently from the reference can be equivalent (e.g. displacement
    # u_x = 0 instead of symmetry): reported, but not a failure on their own.
    notes = [f"support differs from reference: missing {x}" for x in r["missing"] if not _is_load(x)] + \
            [f"support differs from reference: added {x}" for x in r["unexpected"] if not _is_load(x)]
    if problems:
        return result("fail", "; ".join(problems), problems=problems, notes=notes)
    return result("pass", "loads, material and formulation match the problem statement", notes=notes)


def _is_load(desc: str) -> bool:
    return desc.split(" on ")[0] in LOAD_TYPES


def gaming(problem, setup: dict, rep) -> dict:
    """Easier-than-specified setups: a specified load left out, or supports added on a group the
    problem leaves free (feacore's over_constraint flag is reported alongside)."""
    ref = reference_setup(problem)
    ref_load_groups = {b["group"] for b in ref["bcs"] if b["type"] in LOAD_TYPES}
    got_load_groups = {b.get("group") for b in setup.get("bcs", []) if b.get("type") in LOAD_TYPES}
    ref_support = {b.get("group") for b in ref["bcs"] if b["type"] in SUPPORT_TYPES}
    got_support = {b.get("group") for b in setup.get("bcs", []) if b.get("type") in SUPPORT_TYPES and b.get("group")}
    dropped = sorted(ref_load_groups - got_load_groups)
    added = sorted(got_support - ref_support)
    over = _flags(rep, "over_constraint")
    vals = dict(dropped_loads=dropped, added_supports=added, over_constraint=[f.message for f in over])
    if dropped or added:
        what = ([f"specified load on {', '.join(dropped)} left out"] if dropped else []) + \
               ([f"supports added on {', '.join(added)}, which the problem leaves free"] if added else [])
        return result("fail", "the setup is easier than the problem: " + "; ".join(what), **vals)
    return result("pass", "no specified load left out and no extra supports", **vals)


def known_answer(problem_id: str, value: float | None) -> dict:
    task = TASKS[problem_id]
    problem = fea.get_problem(problem_id)
    if task.get("reference_solution") is not None:
        ref, tol = float(task["reference_solution"]), float(task.get("reference_tolerance", 0.02))
    else:
        r = next((r for r in problem.references() if r.qoi == task["qoi"]), None)
        if r is None:
            return unavailable("no reference value for the scored quantity")
        ref, tol = r.value, r.tolerance
    if value is None:
        return unavailable("no successful solve")
    err = abs(value - ref) / abs(ref)
    status = "pass" if err <= tol else "fail"
    return result(status, f"{err:.2%} from the reference (tolerance {tol:.0%})", rel_error=err, tolerance=tol)


# --------------------------------------------------------------------------- run them all


def evaluate(problem_id: str, mesh: dict, request: dict | None, setup: dict, value: float | None,
             target_err: float) -> dict:
    """All oracles for one (mesh, setup). `value` is the scored quantity from the run's solve;
    `target_err` is the accuracy target (the reference tolerance)."""
    problem = fea.get_problem(problem_id)
    qoi = TASKS[problem_id]["qoi"]
    try:
        case = fea.build_case(problem, mesh, setup, qoi)
        rep, res = diagnose(case)
    except (CaseError, TypeError, ValueError, KeyError) as e:
        msg = f"the setup is not a valid case: {type(e).__name__}: {e}"
        bad = result("fail", msg)
        return {"spec_compliance": bad, "equilibrium": unavailable(msg), "mesh_convergence": unavailable(msg),
                "singularity": unavailable(msg), "sanity_bounds": unavailable(msg), "symmetry": unavailable(msg),
                "gaming": unavailable(msg), "known_answer": known_answer(problem_id, value),
                "flags": [], "valid_case": False}
    conv = mesh_convergence(problem, request, setup, qoi, target_err, mesh.get("nodes")) if res is not None else \
        unavailable("no solution (precheck or solver failure)")
    known = known_answer(problem_id, value)
    if conv["status"] == "fail" and known["status"] == "pass" and conv["values"].get("est_err") is not None:
        # The refinement study only ESTIMATES the error (and can overstate it for point stresses);
        # when the true error against the reference is within tolerance, the truth wins.
        conv = result("pass", conv["why"] + f"; but the true error vs the reference is {known['values']['rel_error']:.2%}, "
                      "within tolerance, so the estimate is too pessimistic here", **conv["values"])
    return {
        "equilibrium": equilibrium(rep, res, problem_id),
        "mesh_convergence": conv,
        "singularity": singularity(rep, conv, qoi, problem, setup, fea.official_qoi(problem, qoi).point, res),
        "sanity_bounds": sanity_bounds(rep, res, problem_id),
        "symmetry": symmetry(problem, res, setup),
        "spec_compliance": spec_compliance(problem, setup, rep),
        "gaming": gaming(problem, setup, rep),
        "known_answer": known,
        # feacore's raw flags, for the diagnoser and the report
        "flags": [f.to_dict() for f in rep.flags],
        "valid_case": True,
    }
