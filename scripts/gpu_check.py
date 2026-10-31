"""Check the linear solvers on this machine (Brev / DGX Spark): every available method is run on
the same systems and compared with SciPy, then one benchmark is solved end to end per method.

    python scripts/gpu_check.py                 # 3D Laplacian n=40 (64k unknowns) + one 3D problem
    python scripts/gpu_check.py --n 60 --problem nafems_le10 --h 0.12

Pass criterion: every method agrees with SciPy to 1e-8 relative (direct solves should agree to
~1e-12; anything worse means the GPU path is not trustworthy as the fixed solver).
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time

import numpy as np
import scipy.sparse as sp

from feacore.solver import linsolve

TOL = 1e-8


def laplace3d(n: int) -> sp.csr_matrix:
    e = np.ones(n)
    T = sp.diags([-e[:-1], 2 * e, -e[:-1]], [-1, 0, 1])
    I = sp.identity(n)
    return (sp.kron(sp.kron(T, I), I) + sp.kron(sp.kron(I, T), I) + sp.kron(sp.kron(I, I), T)).tocsr()


def gpu_info() -> str:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                              "--format=csv,noheader"], capture_output=True, text=True, timeout=20)
        return out.stdout.strip() or "nvidia-smi: no output"
    except (OSError, subprocess.SubprocessError):
        return "no nvidia-smi"


def check_matrix(n: int) -> bool:
    A = laplace3d(n)
    b = np.random.default_rng(0).standard_normal(A.shape[0])
    avail = linsolve.available_methods()
    ref, _ = linsolve.solve_linear(A, b, method="scipy", symmetric=True, strict=True, return_report=True)
    ok = True
    print(f"\n3D Laplacian, n = {A.shape[0]}, nnz = {A.nnz}")
    for m in linsolve.DIRECT_METHODS:
        if not avail.get(m):
            print(f"  {m:8s} not available")
            continue
        x, rep = linsolve.solve_linear(A, b, method=m, symmetric=True, strict=True, return_report=True)
        diff = float(np.linalg.norm(x - ref) / np.linalg.norm(ref))
        good = diff < TOL
        ok &= good
        print(f"  {m:8s} {rep.time:8.3f} s  residual {rep.residual:.1e}  vs scipy {diff:.1e}  "
              f"{'ok' if good else 'MISMATCH'}")
    return ok


def check_problem(pid: str, h: float | None) -> bool:
    import tempfile
    from pathlib import Path

    from feacore.problems.registry import get
    from feacore.solver import solve

    try:
        prob = get(pid)
    except KeyError as e:
        print(f"\nskipping end-to-end check: {e}")
        return True
    h = h or prob.study_sizes(2)[1]
    avail = linsolve.available_methods()
    with tempfile.TemporaryDirectory() as tmp:
        mesh = prob.mesh(Path(tmp) / "m.msh", h)
        print(f"\n{pid} at h = {h:g} (order 2)")
        ref = None
        ok = True
        for m in linsolve.DIRECT_METHODS:
            if not avail.get(m):
                continue
            case = prob.case(mesh, order=2)
            case.analysis.options["linear_solver"] = m
            t0 = time.perf_counter()
            res = solve(case, backend="skfem")
            dt = time.perf_counter() - t0
            q = res.evaluate_all()
            if ref is None:
                ref = q
            worst = max((abs(q[k] - ref[k]) / max(abs(ref[k]), 1e-300) for k in q), default=0.0)
            good = worst < TOL
            ok &= good
            print(f"  {m:8s} dofs {res.info.n_dofs:8d}  {dt:7.2f} s total  max QoI diff {worst:.1e}  "
                  f"{'ok' if good else 'MISMATCH'}  {json.dumps({k: round(v, 6) for k, v in q.items()})}")
    return ok


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--n", type=int, default=40, help="Laplacian grid size per axis")
    ap.add_argument("--problem", default="nafems_le10", help="3D problem for the end-to-end check")
    ap.add_argument("--h", type=float, default=None, help="mesh size (default: the medium study size)")
    args = ap.parse_args(argv)
    print(f"host: {platform.machine()} {platform.platform()}")
    print(f"gpu:  {gpu_info()}")
    print(f"linear solvers available: {linsolve.available_methods()}")
    ok = check_matrix(args.n) and check_problem(args.problem, args.h)
    print("\nGPU CHECK " + ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
