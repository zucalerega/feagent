"""Tests of the analytical benchmark problems (``feacore/problems/analytical``).

Fast checks for every registered problem (no solver): the reference case validates, the reference
mesh builds at h0 with every group the case references, STEP export works, the agent-facing
statement contains no reference value, and the exact fields satisfy equilibrium + Hooke's law (or
Laplace's equation) and reproduce the point references. With the skfem backend available, one
coarse solve per family is compared with the references; slow tests (3D, full convergence
studies) are marked ``slow``.
"""

from __future__ import annotations

import json
import math
import re

import numpy as np
import pytest

gmsh = pytest.importorskip("gmsh")
meshio = pytest.importorskip("meshio")

from feacore.case import Case, eval_expr  # noqa: E402
from feacore.geometry.gmsh_tools import gmsh_session  # noqa: E402
from feacore.problems.registry import all_problems, discover  # noqa: E402

EXPECTED_IDS = {
    "lame_cylinder_plane_strain", "lame_cylinder_axisym", "lame_cylinder_axisym_open", "lame_sphere_axisym",
    "lame_sphere_3d", "kirsch_exact_traction", "plate_hole_finite_width_d01", "plate_hole_finite_width_d03",
    "plate_hole_finite_width_d05", "timoshenko_cantilever", "lshape_poisson", "lbracket_reentrant",
    "rotating_disk",
}


def _analytical():
    discover()
    return [p for p in all_problems() if type(p).__module__.startswith("feacore.problems.analytical")]


PROBLEMS = _analytical()
BY_ID = {p.id: p for p in PROBLEMS}
IDS = sorted(BY_ID)
EXACT_IDS = [p.id for p in PROBLEMS if p.exact("T" if p.physics == "heat" else "ux", np.array([[0.13, 0.17, 0.11][: p.dim]])) is not None]


def _referenced_groups(case: Case) -> set[str]:
    g = {b.group for b in case.bcs if b.group}
    g |= {ld.group for ld in case.loads if ld.group not in (None, "*")}
    g |= {r.group for r in case.regions if r.group not in (None, "*")}
    g |= {q.group for q in case.qois if q.group not in (None, "*")}
    return g


# --------------------------------------------------------------------------- registry / case


def test_all_expected_problems_registered():
    errors = {k: v for k, v in discover().items() if ".analytical" in k}
    assert not errors, errors
    assert EXPECTED_IDS <= set(IDS), sorted(EXPECTED_IDS - set(IDS))
    fams = {p.family for p in PROBLEMS}
    assert len([p for p in PROBLEMS if p.family == "plate_hole_finite_width"]) == 3
    assert {"lame_cylinder", "lame_sphere", "kirsch", "timoshenko_cantilever", "lshape_poisson"} <= fams


@pytest.mark.parametrize("pid", IDS)
def test_case_and_metadata(pid):
    P = BY_ID[pid]
    assert P.title and P.family and P.tags and P.description()
    assert P.params["h0"] > 0
    for order in (1, 2):
        c = P.case("MESH.msh", order=order).check()
        assert c.analysis.order == order
        assert c.analysis.physics == P.physics and c.analysis.formulation == P.formulation
        assert [q.name for q in c.qois] == [q.name for q in P.qois()]
        c2 = Case.from_dict(json.loads(json.dumps(c.to_dict()))).check()
        assert c2.to_dict() == c.to_dict()
        sizes = P.study_sizes(order)
        assert len(sizes) == 3 and sizes == sorted(sizes, reverse=True)
    names = {q.name for q in P.qois()}
    refs = P.references()
    assert {r.qoi for r in refs} <= names
    assert set(P.expectations()) <= names
    assert set(P.expectations().values()) <= {"converge", "diverge"}
    for r in refs:
        assert math.isfinite(r.value) and r.value != 0.0
        assert r.source and 0.0 < r.tolerance <= 0.05
        assert r.kind in ("analytical", "handbook")
    for q in P.qois():
        if P.expectations().get(q.name) == "diverge":
            assert q.name not in {r.qoi for r in refs}
    json.dumps(P.private())
    json.dumps(P.statement())


# --------------------------------------------------------------------------- geometry / mesh / STEP


@pytest.mark.parametrize("pid", IDS)
def test_mesh_builds_with_all_groups(pid, tmp_path):
    P = BY_ID[pid]
    h0 = P.study_sizes(2)[0]
    path = P.mesh(tmp_path / f"{pid}.msh", h0, geom_order=2)
    m = meshio.read(path)
    groups = set(m.field_data)
    case = P.case(path, 2).check()
    missing = _referenced_groups(case) - groups
    assert not missing, f"mesh lacks groups {missing}; has {sorted(groups)}"
    cell = "tetra10" if P.dim == 3 else "triangle6"
    assert cell in m.cells_dict and len(m.cells_dict[cell]) > 10
    # point QoIs lie in the meshed region (within one element of a node)
    pts = m.points[:, : P.dim]
    for q in P.qois():
        if q.type == "point":
            d = np.linalg.norm(pts - np.asarray(q.point[: P.dim]), axis=1).min()
            assert d < h0, f"{q.name} point {q.point} is {d:.3g} from the mesh"


@pytest.mark.parametrize("pid", IDS)
def test_step_export(pid, tmp_path):
    P = BY_ID[pid]
    path = P.export_step(tmp_path / "geometry.step")
    text = path.read_text(errors="ignore")
    assert text.startswith("ISO-10303-21") and len(text) > 500
    with gmsh_session("reimport"):
        gmsh.model.occ.importShapes(str(path))
        gmsh.model.occ.synchronize()
        assert len(gmsh.model.getEntities(P.dim)) >= 1


# --------------------------------------------------------------------------- statement hygiene

_NUM_EDGE = r"(?<![\d.eE+\-])"


def _number_strings(v: float) -> set[str]:
    out = set()
    for scale in (1.0, 1e-6, 1e-3, 1e3, 1e6):  # SI, MPa / kJ, mm, um
        x = v * scale
        for fmt in (".3g", ".4g", ".5g", ".6g", ".8g", ".10g", ".3e", ".4e", ".6e", ".3f", ".4f"):
            s = format(x, fmt)
            if len(s.lstrip("-").replace(".", "").replace("0", "")) >= 2:  # skip "0.000", "1", ...
                out.add(s)
        out.add(repr(x))
    return out


@pytest.mark.parametrize("pid", IDS)
def test_statement_has_no_reference_values(pid):
    P = BY_ID[pid]
    text = json.dumps(P.statement())
    assert "reference" not in json.dumps(P.statement()["qois"]).lower()
    for r in P.references():
        for s in _number_strings(r.value):
            assert not re.search(_NUM_EDGE + re.escape(s) + r"(?!\d)", text), \
                f"statement of {pid} contains {s!r} (reference {r.qoi} = {r.value})"


# --------------------------------------------------------------------------- exact fields


def _sample_points(P, n_max=60):
    """Mesh nodes of a coarse linear reference mesh (deterministic), away from r = 0 corners."""
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as d:
        m = meshio.read(P.mesh(Path(d) / "s.msh", P.study_sizes(2)[0], geom_order=1))
    pts = m.points[:, : P.dim]
    pts = pts[np.linalg.norm(pts, axis=1) > 1e-3 * np.abs(pts).max()]
    idx = np.linspace(0, len(pts) - 1, min(n_max, len(pts))).astype(int)
    return pts[idx]


def _fd(P, field, pts, axis, h):
    e = np.zeros(pts.shape[1])
    e[axis] = h
    return (P.exact(field, pts + e) - P.exact(field, pts - e)) / (2 * h)


@pytest.mark.parametrize("pid", EXACT_IDS)
def test_point_references_equal_exact_field(pid):
    P = BY_ID[pid]
    qs = {q.name: q for q in P.qois()}
    n = 0
    for r in P.references():
        q = qs[r.qoi]
        if q.type == "point":
            v = P.exact(q.field, np.asarray([q.point]))[0]
            assert v == pytest.approx(r.value, rel=1e-12)
            n += 1
    assert n >= 1


@pytest.mark.parametrize("pid", [i for i in EXACT_IDS if BY_ID[i].physics == "elasticity"])
def test_exact_field_satisfies_elasticity(pid):
    """Finite-difference check of Hooke's law (from the exact displacement) and equilibrium
    div(sigma) + f = 0 (body force from the reference case) at sample points."""
    P = BY_ID[pid]
    pts = _sample_points(P)
    L = np.abs(pts).max()
    if P.formulation == "axisymmetric":  # hoop strain u_r / r: stay off the axis
        pts = pts[pts[:, 0] > 1e-3 * L]
    E, nu = P.params["E"], P.params["nu"]
    h = 1e-6 * L
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    f = P.exact
    S = {k: f(k, pts) for k in ("sxx", "syy", "szz", "sxy", "syz", "sxz", "stt") if f(k, pts) is not None}
    scale = max(np.abs(v).max() for v in S.values())
    d = lambda fld, ax: _fd(P, fld, pts, ax, h)  # noqa: E731
    form = P.formulation
    if form == "3d":
        g = np.array([[d(f"u{c}", ax) for ax in range(3)] for c in "xyz"])  # g[i, j] = du_i/dx_j
        eps = 0.5 * (g + g.transpose(1, 0, 2))
        tr = eps[0, 0] + eps[1, 1] + eps[2, 2]
        hooke = {"sxx": lam * tr + 2 * mu * eps[0, 0], "syy": lam * tr + 2 * mu * eps[1, 1],
                 "szz": lam * tr + 2 * mu * eps[2, 2], "sxy": 2 * mu * eps[0, 1], "syz": 2 * mu * eps[1, 2],
                 "sxz": 2 * mu * eps[0, 2]}
    elif form == "axisymmetric":
        err, ezz, ett = d("ux", 0), d("uy", 1), f("ux", pts) / pts[:, 0]
        grz = d("ux", 1) + d("uy", 0)
        tr = err + ezz + ett
        hooke = {"sxx": lam * tr + 2 * mu * err, "syy": lam * tr + 2 * mu * ezz, "stt": lam * tr + 2 * mu * ett,
                 "sxy": mu * grz}
    else:
        exx, eyy, gxy = d("ux", 0), d("uy", 1), d("ux", 1) + d("uy", 0)
        if form == "plane_stress":
            c = E / (1 - nu**2)
            hooke = {"sxx": c * (exx + nu * eyy), "syy": c * (eyy + nu * exx), "sxy": mu * gxy}
        else:
            hooke = {"sxx": lam * (exx + eyy) + 2 * mu * exx, "syy": lam * (exx + eyy) + 2 * mu * eyy,
                     "szz": lam * (exx + eyy), "sxy": mu * gxy}
    for k, v in hooke.items():
        assert np.max(np.abs(v - S[k])) < 1e-5 * scale, k
    # body force of the reference case (expressions in x, y, z)
    case = P.case("MESH.msh")
    fb = np.zeros((P.dim, len(pts)))
    for ld in case.loads:
        if ld.type == "body_force":
            z = pts[:, 2] if P.dim == 3 else None
            fb += np.array([eval_expr(str(v), pts[:, 0], pts[:, 1], z) for v in ld.value])
    ds = lambda fld, ax: _fd(P, fld, pts, ax, h)  # noqa: E731
    if form == "3d":
        res = [ds("sxx", 0) + ds("sxy", 1) + ds("sxz", 2), ds("sxy", 0) + ds("syy", 1) + ds("syz", 2),
               ds("sxz", 0) + ds("syz", 1) + ds("szz", 2)]
    elif form == "axisymmetric":
        r = pts[:, 0]
        res = [ds("sxx", 0) + ds("sxy", 1) + (S["sxx"] - S["stt"]) / r,
               ds("sxy", 0) + ds("syy", 1) + S["sxy"] / r]
    else:
        res = [ds("sxx", 0) + ds("sxy", 1), ds("sxy", 0) + ds("syy", 1)]
    for i, rr in enumerate(res):
        assert np.max(np.abs(rr + fb[i])) < 1e-4 * scale / L, f"equilibrium component {i}"


def test_lshape_exact_is_harmonic_and_flux_consistent():
    P = BY_ID["lshape_poisson"]
    pts = _sample_points(P)
    pts = pts[np.linalg.norm(pts, axis=1) > 0.05]
    h = 1e-4
    lap = sum((P.exact("T", pts + e) - 2 * P.exact("T", pts) + P.exact("T", pts - e)) / h**2
              for e in (np.array([h, 0]), np.array([0, h])))
    assert np.max(np.abs(lap)) < 1e-5
    for ax, q in ((0, "qx"), (1, "qy")):
        assert np.allclose(-_fd(P, "T", pts, ax, 1e-6), P.exact(q, pts), atol=1e-7)
    # Dirichlet data of the reference case = exact field on each edge; zero on the re-entrant edges
    case = P.case("MESH.msh")
    s = np.linspace(0.0, 1.0, 11)
    edges = {"edge_x_pos": np.c_[np.ones_like(s), s], "edge_y_pos": np.c_[2 * s - 1, np.ones_like(s)],
             "edge_x_neg": np.c_[-np.ones_like(s), 2 * s - 1], "edge_y_neg": np.c_[-s, -np.ones_like(s)],
             "reentrant_edges": np.r_[np.c_[s, -0.0 * s], np.c_[0 * s, -s]]}
    for bc in case.bcs:
        x = edges[bc.group]
        val = eval_expr(bc.value, x[:, 0], x[:, 1]) if isinstance(bc.value, str) else np.full(len(x), bc.value)
        assert np.allclose(val, P.exact("T", x), atol=1e-14)


def test_expression_bcs_match_exact_fields():
    K = BY_ID["kirsch_exact_traction"]
    L = K.params["L"]
    s = np.linspace(0, L, 13)
    case = K.case("MESH.msh")
    bc = {b.group: b for b in case.bcs}
    right, top = np.c_[np.full_like(s, L), s], np.c_[s, np.full_like(s, L)]
    for pts, comps in ((right, ("sxx", "sxy")), (top, ("sxy", "syy"))):
        b = bc["right" if pts is right else "top"]
        for expr, comp in zip(b.value, comps):
            assert np.allclose(eval_expr(expr, pts[:, 0], pts[:, 1]), K.exact(comp, pts), rtol=1e-12, atol=1e-3)
    C = BY_ID["timoshenko_cantilever"]
    D = C.params["D"]
    y = np.linspace(-D / 2, D / 2, 11)
    case = C.case("MESH.msh")
    root, tip = case.bcs[0], case.bcs[1]
    for expr, comp in zip(root.value, ("ux", "uy")):
        pts = np.c_[np.zeros_like(y), y]
        assert np.allclose(eval_expr(expr, pts[:, 0], pts[:, 1]), C.exact(comp, pts), rtol=1e-12, atol=1e-18)
    pts = np.c_[np.full_like(y, C.params["L"]), y]
    assert np.allclose(eval_expr(tip.value[1], pts[:, 0], pts[:, 1]), C.exact("sxy", pts), rtol=1e-12)
    assert np.allclose(C.exact("sxx", pts), 0.0, atol=1e-6)


def test_handbook_and_constants():
    from feacore.problems.analytical.lshape import WILLIAMS_270, _williams_root
    from feacore.problems.analytical.plate_hole import peterson_ktn

    assert _williams_root() == pytest.approx(WILLIAMS_270, rel=1e-12)
    assert peterson_ktn(0.0) == pytest.approx(3.004)  # -> Kirsch
    assert peterson_ktn(1.0) == pytest.approx(2.0)
    assert BY_ID["plate_hole_finite_width_d05"].exact("sxx", np.zeros((1, 2))) is None
    assert BY_ID["lbracket_reentrant"].exact("ux", np.zeros((1, 2))) is None


# --------------------------------------------------------------------------- solver (one coarse solve per family)


def _solver_available():
    try:
        import feacore.solver.skfem_backend  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return True


needs_solver = pytest.mark.skipif(not _solver_available(), reason="skfem backend not available")

# coarse-mesh (h0, P2) bound on |FE - reference| / |reference|: studies show <~3 % at h0
FAMILY_SOLVES = ["lame_cylinder_plane_strain", "lame_cylinder_axisym", "lame_sphere_axisym", "kirsch_exact_traction",
                 "plate_hole_finite_width_d03", "timoshenko_cantilever", "lshape_poisson", "rotating_disk"]


def _solve_at(P, h, tmp_path, order=2):
    from feacore.solver import solve

    m = P.mesh(tmp_path / f"{P.id}_{h:.4g}.msh", h, geom_order=order)
    return solve(P.case(m, order=order), backend="skfem")


@needs_solver
@pytest.mark.parametrize("pid", FAMILY_SOLVES)
def test_coarse_solve_matches_references(pid, tmp_path):
    P = BY_ID[pid]
    res = _solve_at(P, P.study_sizes(2)[0], tmp_path)
    vals = res.evaluate_all()
    assert set(vals) == {q.name for q in P.qois()}
    assert all(math.isfinite(v) for v in vals.values())
    for r in P.references():
        err = abs(vals[r.qoi] - r.value) / abs(r.value)
        assert err < max(3 * r.tolerance, 0.03), f"{r.qoi}: {vals[r.qoi]:.6g} vs {r.value:.6g} ({100 * err:.2f} %)"


@needs_solver
def test_lbracket_singularity_grows(tmp_path):
    P = BY_ID["lbracket_reentrant"]
    h0 = P.study_sizes(2)[0]
    a, b = (_solve_at(P, h, tmp_path).evaluate_all() for h in (h0, h0 / 2))
    assert b["vm_max"] > 1.1 * a["vm_max"]                        # singular: grows ~ h^-0.46
    assert abs(b["tip_uy"] - a["tip_uy"]) < 0.02 * abs(b["tip_uy"])  # converges
    assert b["tip_uy"] < 0 and b["strain_energy"] > 0


@needs_solver
@pytest.mark.slow
def test_lame_sphere_3d_solve(tmp_path):
    P = BY_ID["lame_sphere_3d"]
    vals = _solve_at(P, P.study_sizes(2)[0], tmp_path).evaluate_all()
    for r in P.references():
        assert abs(vals[r.qoi] - r.value) < 0.05 * abs(r.value), r.qoi


@needs_solver
@pytest.mark.slow
@pytest.mark.parametrize("pid", ["kirsch_exact_traction", "lshape_poisson", "lbracket_reentrant",
                                 "plate_hole_finite_width_d05"])
def test_convergence_study_passes(pid, tmp_path):
    from feacore.verification.study import convergence_study

    st = convergence_study(BY_ID[pid], order=2, backend="skfem", workdir=tmp_path)
    assert st["passed"] is True, [(q["name"], q["passed"], q.get("gci", {}).get("convergence")) for q in st["qois"]]
    for q in st["qois"]:
        if q["expectation"] == "diverge":
            assert q["gci"]["convergence"] in ("divergent", "oscillatory_divergent")
