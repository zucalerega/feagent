"""Tests of the pluggable linear solver (feacore.solver.linsolve).

The cuDSS test only runs on a machine with a CUDA GPU and nvmath-python installed (Brev);
it compares cuDSS against SciPy on a 3D P2 elasticity system to ~1e-10 relative.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from feacore.solver import linsolve
from feacore.solver.linsolve import LinearSolverError, available_methods, solve_linear


def _elasticity_3d(nref: int = 2):
    """Clamped unit cube, P2 tetrahedra, body load: SPD system after Dirichlet elimination."""
    from skfem import Basis, ElementTetP2, ElementVector, MeshTet1, condense
    from skfem.models.elasticity import lame_parameters, linear_elasticity

    m = MeshTet1().refined(nref)
    ub = Basis(m, ElementVector(ElementTetP2()))
    K = linear_elasticity(*lame_parameters(200e3, 0.3)).assemble(ub)
    rng = np.random.default_rng(0)
    f = rng.standard_normal(ub.N)
    D = ub.get_dofs(lambda x: x[0] < 1e-12).all()
    A, b, _, _ = condense(K, f, D=D)
    return sp.csr_matrix(A), b


@pytest.fixture(scope="module")
def system():
    return _elasticity_3d(2)


def _rel(x, y):
    return float(np.linalg.norm(x - y) / np.linalg.norm(y))


def test_scipy_direct(system):
    A, b = system
    x, rep = solve_linear(A, b, method="scipy", return_report=True)
    assert rep.method == "scipy" and not rep.iterative
    assert rep.residual < 1e-10
    x_ref = sp.linalg.spsolve(A.tocsc(), b)
    assert _rel(x, x_ref) < 1e-10


@pytest.mark.skipif(not available_methods()["pardiso"], reason="pypardiso not installed")
def test_pardiso_matches_scipy(system):
    A, b = system
    x_s = solve_linear(A, b, method="scipy")
    x_p, rep = solve_linear(A, b, method="pardiso", return_report=True, strict=True)
    assert rep.method == "pardiso" and "mtype=2" in rep.detail
    assert _rel(x_p, x_s) < 1e-10
    # several right-hand sides at once, and the general (non-symmetric) path
    B = np.column_stack([b, 2 * b])
    X = solve_linear(A, B, method="pardiso", symmetric=False, strict=True)
    assert X.shape == B.shape and _rel(X[:, 1], 2 * x_s) < 1e-10


@pytest.mark.skipif(not available_methods()["cudss"], reason="needs a CUDA GPU and nvmath-python (cuDSS)")
def test_cudss_matches_scipy_3d_elasticity():
    A, b = _elasticity_3d(3)  # ~25k unknowns
    x_s = solve_linear(A, b, method="scipy")
    x_c, rep = solve_linear(A, b, method="cudss", return_report=True, strict=True)
    assert rep.method == "cudss"
    assert _rel(x_c, x_s) < 1e-10
    B = np.column_stack([b, -b])
    X = solve_linear(A, B, method="cudss", strict=True)
    assert _rel(X[:, 1], -x_s) < 1e-10


def test_auto_and_env_var(system, monkeypatch):
    A, b = system
    monkeypatch.delenv(linsolve.ENV_VAR, raising=False)
    _, rep = solve_linear(A, b, return_report=True)
    gpu = available_methods()["cudss"] and A.shape[0] >= linsolve.cudss_min_n()
    expected = "cudss" if gpu else "pardiso" if available_methods()["pardiso"] else "scipy"
    assert rep.method == expected
    monkeypatch.setenv(linsolve.ENV_VAR, "scipy")
    _, rep = solve_linear(A, b, method="auto", return_report=True)
    assert rep.method == "scipy"
    _, rep = solve_linear(A, b, method=None, return_report=True)
    assert rep.method == "scipy"
    monkeypatch.setenv(linsolve.ENV_VAR, "bogus")
    with pytest.raises(ValueError):
        solve_linear(A, b)


def test_unavailable_method_falls_back_or_raises(system, monkeypatch):
    A, b = system
    monkeypatch.setattr(linsolve, "_have_cudss", lambda: False)
    x, rep = solve_linear(A, b, method="cudss", return_report=True)
    assert rep.method != "cudss" and any("unavailable" in n for n in rep.notes)
    with pytest.raises(LinearSolverError):
        solve_linear(A, b, method="cudss", strict=True)


def test_iterative_cg_is_opt_in_and_reported():
    A, b = _elasticity_3d(1)
    x, rep = solve_linear(A, b, method="cg", return_report=True)
    assert rep.iterative and rep.tolerance == 1e-10 and "ITERATIVE" in rep.describe()
    assert _rel(x, solve_linear(A, b, method="scipy")) < 1e-7


def test_singular_matrix_is_reported():
    A = sp.csr_matrix(np.array([[1.0, -1.0], [-1.0, 1.0]]))
    with pytest.raises(LinearSolverError):
        solve_linear(A, np.array([1.0, 0.0]), method="scipy")


def test_backend_selects_solver_from_options(tmp_path):
    from feacore.case import BC, Analysis, Case, Material
    from feacore.solver import solve
    import meshio

    # tiny structured 2D mesh written with meshio (no gmsh needed)
    n = 4
    xs = np.linspace(0, 1, n + 1)
    X, Y = np.meshgrid(xs, xs)
    pts = np.column_stack([X.ravel(), Y.ravel(), np.zeros(X.size)])
    idx = np.arange(X.size).reshape(n + 1, n + 1)
    tris = []
    for i in range(n):
        for j in range(n):
            a, b_, c, d = idx[i, j], idx[i, j + 1], idx[i + 1, j + 1], idx[i + 1, j]
            tris += [[a, b_, c], [a, c, d]]
    left = [[idx[i, 0], idx[i + 1, 0]] for i in range(n)]
    right = [[idx[i, n], idx[i + 1, n]] for i in range(n)]
    cells = [("line", np.array(left)), ("line", np.array(right)), ("triangle", np.array(tris))]
    m = meshio.Mesh(pts, cells, cell_data={"gmsh:physical": [np.full(n, 1), np.full(n, 2), np.full(2 * n * n, 3)],
                                           "gmsh:geometrical": [np.full(n, 1), np.full(n, 2), np.full(2 * n * n, 1)]},
                    field_data={"left": np.array([1, 1]), "right": np.array([2, 1]), "domain": np.array([3, 2])})
    path = tmp_path / "sq.msh"
    meshio.write(str(path), m, file_format="gmsh22", binary=False)
    case = Case(mesh=str(path), analysis=Analysis(formulation="plane_stress", order=1, options={"linear_solver": "scipy"}),
                materials={"m": Material(E=1.0, nu=0.3)},
                bcs=[BC(type="fixed", group="left"), BC(type="force", group="right", value=[1.0, 0.0])])
    r = solve(case)
    assert any("linear_solver=scipy" in w for w in r.info.warnings)
