"""Solution diagnostics: deterministic failure-mode flags for one FE solution, reference-free
accuracy estimates, and the speed / accuracy score of an agent's setup.

    from feacore.diagnostics import diagnose
    report, result = diagnose(case, design={"yield": 275e6})
    report.status, report.ids(), report.metrics

Failure-mode catalogue (ids, method, thresholds): docs/failure-modes.md. The detectors read the
fixed solver through ``feacore.diagnostics.view.SolutionView``.
"""

from __future__ import annotations

from feacore.case import Case
from feacore.diagnostics import checks
from feacore.diagnostics.flags import THRESHOLDS, Flag, Report
from feacore.diagnostics.view import SolutionView
from feacore.solver.linsolve import LinearSolverError


def precheck(case: Case) -> Report:
    """Setup checks on the assembled system, without solving (cheap; usable by agents before
    they commit to a configuration, e.g. to answer INVALID_CONFIG)."""
    rep = Report()
    view = SolutionView.setup(case)
    for check in checks.PRECHECKS:
        check(view, rep)
    rep.metrics["n_dofs"] = int(view.N)
    return rep


def diagnose_result(res, design: dict | None = None, rep: Report | None = None) -> Report:
    """Post-solve checks on a solved skfem ``Result``."""
    rep = rep or Report()
    view = SolutionView.from_result(res)
    rep.metrics.update(n_dofs=res.info.n_dofs, n_cells=res.info.n_cells, wall_time=res.info.wall_time,
                       h_rep=res.info.h_rep, order=res.info.order, strain_energy=view.strain_energy())
    if not checks.check_finite(view, rep):
        return rep
    checks.check_residual(view, rep)
    checks.check_equilibrium(view, rep)
    checks.check_validity(view, rep)
    zz = checks.check_discretization(view, rep)
    checks.check_peaks(view, rep, zz)
    checks.trusted_peak(view, rep)
    checks.check_overconstraint(view, rep)
    checks.check_design(view, rep, design)
    return rep


def diagnose(case: Case, design: dict | None = None, **solve_options):
    """Precheck -> solve with the fixed solver (only if the setup is valid) -> post-solve checks.
    Returns (Report, Result | None)."""
    from feacore.solver.skfem_backend import InsufficientConstraints, SolverError, solve

    rep = precheck(case)
    if rep.status == "fail":
        return rep, None
    try:
        res = solve(case, **solve_options)
    except (LinearSolverError, SolverError, InsufficientConstraints) as e:
        rep.add(Flag("solver_failure", "solve", "fail", str(e)))
        return rep, None
    return diagnose_result(res, design, rep), res


__all__ = ["diagnose", "diagnose_result", "precheck", "Report", "Flag", "THRESHOLDS"]
