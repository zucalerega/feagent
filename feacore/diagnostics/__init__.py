"""Solution diagnostics: deterministic failure-mode flags for one FE solution, reference-free
accuracy estimates, and the speed / accuracy score of an agent's setup.

    from feacore.diagnostics import diagnose
    report, result = diagnose(case, design={"yield": 275e6})
    report.status, report.ids(), report.metrics

Failure-mode catalogue (ids, method, thresholds): docs/failure-modes.md.
"""

from __future__ import annotations

import time

from feacore.case import Case
from feacore.diagnostics import checks
from feacore.diagnostics.flags import THRESHOLDS, Flag, Report
from feacore.solver.linsolve import LinearSolverError


def precheck(case: Case) -> Report:
    """Setup checks on the assembled system, without solving (cheap; usable by agents before
    they commit to a configuration, e.g. to answer INVALID_CONFIG)."""
    from feacore.solver.skfem_backend import setup

    rep = Report()
    res = setup(case)
    for check in checks.PRECHECKS:
        check(res, rep)
    rep.metrics["n_dofs"] = int(res.basis.N)
    return rep


def diagnose_result(res, design: dict | None = None, rep: Report | None = None) -> Report:
    """Post-solve checks on a solved ``SkfemResult``."""
    rep = rep or Report()
    rep.metrics.update(n_dofs=res.info.n_dofs, n_cells=res.info.n_cells, wall_time=res.info.wall_time,
                       h_rep=res.info.h_rep, order=res.info.order, strain_energy=res.strain_energy())
    if not checks.check_finite(res, rep):
        return rep
    checks.check_residual(res, rep)
    checks.check_equilibrium(res, rep)
    checks.check_validity(res, rep)
    zz = checks.check_discretization(res, rep)
    checks.check_peaks(res, rep, zz)
    checks.trusted_peak(res, rep)
    checks.check_overconstraint(res, rep)
    checks.check_design(res, rep, design)
    return rep


def diagnose(case: Case, design: dict | None = None, **solve_options):
    """Precheck -> solve (only if the setup is valid) -> post-solve checks.
    Returns (Report, SkfemResult | None)."""
    from feacore.solver.skfem_backend import solve

    rep = precheck(case)
    if rep.status == "fail":
        return rep, None
    t0 = time.perf_counter()
    try:
        res = solve(case, **solve_options)
    except LinearSolverError as e:
        rep.add(Flag("solver_failure", "solve", "fail", str(e)))
        return rep, None
    rep.metrics["solve_time"] = time.perf_counter() - t0
    return diagnose_result(res, design, rep), res


__all__ = ["diagnose", "diagnose_result", "precheck", "Report", "Flag", "THRESHOLDS"]
