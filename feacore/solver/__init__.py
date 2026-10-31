"""Fixed solver entry point: ``solve(case, backend="skfem") -> Result``.

Backends are modules exposing ``solve(case: Case, **options) -> Result``:
  skfem  scikit-fem, the primary (fixed) solver
  ccx    CalculiX, an independent second solver for cross-checks
  nn     neural surrogate (optional, GPU); fast but unverified, never the reference
"""

from __future__ import annotations

import importlib
from pathlib import Path

from feacore.case import Case
from feacore.solver.base import Result, SolveInfo

BACKENDS = {
    "skfem": "feacore.solver.skfem_backend",
    "ccx": "feacore.solver.ccx_backend",
    "nn": "feacore.solver.nn_backend",
}


def solve(case: Case | str | Path | dict, backend: str = "skfem", **options) -> Result:
    if isinstance(case, (str, Path)):
        case = Case.load(case)
    elif isinstance(case, dict):
        case = Case.from_dict(case)
    case.check()
    if backend not in BACKENDS:
        raise ValueError(f"unknown backend {backend!r}; available: {sorted(BACKENDS)}")
    module = importlib.import_module(BACKENDS[backend])
    return module.solve(case, **options)


__all__ = ["solve", "Result", "SolveInfo", "BACKENDS"]
