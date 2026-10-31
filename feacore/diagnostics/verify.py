"""Reference-free accuracy of an agent's setup: re-solve the *same* setup on systematically
coarser / finer meshes and on a higher element order, then extrapolate.

refinement_study: the agent's mesh is the middle of three (sizes s*r, s, s/r). Richardson
  extrapolation + GCI (feacore.verification.gci) give, per QoI, the extrapolated value phi_ext and
  the agent's estimated error |phi_agent - phi_ext| / |phi_ext|. A QoI that diverges, or converges
  far below its theoretical order while growing, is classified "singular": its value is a mesh
  artefact and is excluded from accuracy scoring (and flagged).
p_sensitivity: P1 setups re-solved with P2 on the same mesh; a large change in strain energy
  exposes locking / under-resolution that a P1-only refinement study converges to slowly.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Callable

from feacore.case import Case, QoI
from feacore.diagnostics.flags import THRESHOLDS as T, Flag, Report
from feacore.verification.gci import gci3
from feacore.verification.study import default_expected_order

ENERGY = QoI(name="strain_energy", type="strain_energy")


def _qois(case: Case) -> list[QoI]:
    qs = list(case.qois)
    if not any(q.type == "strain_energy" for q in qs):
        qs.append(ENERGY)
    return qs


def classify(g: dict, p_theory: float | None) -> str:
    """converged | reduced_order | oscillatory | singular.

    Singular = successive changes grow under refinement (the value has no limit, e.g. a peak
    stress at a re-entrant corner or clamped edge). Reduced order = converges, but far below the
    theoretical rate: finite, yet polluted by a singularity elsewhere (e.g. strain energy of an
    L-shape converges at ~2 lambda instead of 2k); the extrapolation uses the observed rate."""
    conv = g["convergence"]
    if conv == "exact" or "two_solutions_identical" in g["flags"]:
        return "converged"
    if conv in ("divergent", "oscillatory_divergent") or g.get("p") is None:
        return "singular"
    if conv == "oscillatory":
        return "oscillatory"
    if p_theory and g["p"] < 0.5 * p_theory:
        return "reduced_order"
    return "converged"


def refinement_study(case_at: Callable[[float], Case], r: float = 2.0, solve_options: dict | None = None) -> dict:
    """``case_at(scale)`` must return the agent's setup meshed with every size multiplied by
    ``scale`` (1.0 = the agent's own mesh)."""
    from feacore.solver.skfem_backend import solve

    scales = [r, 1.0, 1.0 / r]                          # coarse, agent, fine
    runs = []
    for s in scales:
        res = solve(case_at(s), **(solve_options or {}))
        runs.append((res.info, {q.name: res.evaluate(q) for q in _qois(res.case)}, res.case))
    case = runs[1][2]
    out = {"scales": scales, "h": [i.h_rep for i, _, _ in runs], "dofs": [i.n_dofs for i, _, _ in runs],
           "qois": {}}
    for q in _qois(case):
        vals = [v[q.name] for _, v, _ in runs]
        p_th = default_expected_order(q, case.analysis.order)
        g = gci3(out["h"], vals, p_theory=p_th).to_dict()
        kind = classify(g, p_th)
        agent = vals[1]
        if kind in ("converged", "reduced_order") and g.get("phi_ext") not in (None, 0):
            err = abs(agent - g["phi_ext"]) / abs(g["phi_ext"])
        elif kind == "oscillatory":
            err = (max(vals) - min(vals)) / max(abs(agent), 1e-300)
        else:
            err = None
        out["qois"][q.name] = {"values": vals, "kind": kind, "p": g.get("p"), "p_theory": p_th,
                               "phi_ext": g.get("phi_ext"), "est_err": err, "gci": g}
    return out


def study_flags(study: dict, rep: Report) -> None:
    for name, q in study["qois"].items():
        if q["kind"] == "singular":
            rep.add(Flag("singular_qoi", "singularity", "warn",
                         f"QoI {name!r} does not converge under refinement (values {['%.4g' % v for v in q['values']]}, "
                         f"p = {q['p'] if q['p'] is None else round(q['p'], 2)}): it sits on a singularity "
                         "and has no mesh-independent value"))
        elif q["kind"] == "reduced_order":
            rep.add(Flag("reduced_order", "singularity", "info",
                         f"QoI {name!r} converges at p = {q['p']:.2f} (theory {q['p_theory']:g}): a singularity "
                         "elsewhere slows convergence; local refinement there is the cheapest fix"))
        elif q["kind"] == "oscillatory":
            rep.add(Flag("oscillatory_qoi", "discretization", "info",
                         f"QoI {name!r} oscillates under refinement; error bound is the spread"))


def p_sensitivity(case: Case, rep: Report | None = None, solve_options: dict | None = None) -> dict | None:
    """Order-1 setups only: strain energy and QoIs at P1 vs P2 on the same mesh."""
    from feacore.solver.skfem_backend import solve

    if case.analysis.order != 1:
        return None
    hi = dataclasses.replace(case, analysis=dataclasses.replace(case.analysis, order=2))
    r1, r2 = solve(case, **(solve_options or {})), solve(hi, **(solve_options or {}))
    out = {}
    for q in _qois(case):
        a, b = r1.evaluate(q), r2.evaluate(q)
        out[q.name] = {"p1": a, "p2": b, "rel_change": abs(a - b) / abs(b) if b else math.inf}
    dU = out["strain_energy"]["rel_change"]
    if rep is not None and dU > T["p_sensitivity_warn"]:
        rep.add(Flag("p_sensitive", "discretization", "warn",
                     f"strain energy changes {dU:.1%} from P1 to P2 on the same mesh: linear elements "
                     "are locking / under-resolved here", value=dU, threshold=T["p_sensitivity_warn"]))
    return out
