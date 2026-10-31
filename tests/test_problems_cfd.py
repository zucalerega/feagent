"""Tests for the incompressible-flow (CFD) benchmark suite (``feacore.problems.cfd``).

For every problem: the reference case validates (both convection orders), the reference mesh at h0
builds with every group the case references (structured quads / hexes, counter-clockwise 2D cells),
nested refinement splits it, STEP export gives one clean face / solid, and the agent-facing statement
leaks no reference value. Exact-solution and reference-conversion checks need no solver. With the
OpenFOAM backend available, one coarse solve per problem is compared with the references (skipped
when OpenFOAM is missing); the slower solves and the full reference studies run only with
``FEACORE_SLOW=1``.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os

import numpy as np
import pytest

gmsh = pytest.importorskip("gmsh")
meshio = pytest.importorskip("meshio")

from feacore.case import Case, eval_expr  # noqa: E402
from feacore.geometry.gmsh_tools import gmsh_session  # noqa: E402
from feacore.problems.registry import all_problems, discover, get  # noqa: E402

EXPECTED = {"poiseuille_channel", "lid_driven_cavity_re100", "cylinder_channel_dfg_2d1"}
REFERENCE_KINDS = {"analytical", "published", "handbook", "manufactured", "converged_numerical",
                   "published_numerical"}
slow = pytest.mark.skipif(not os.environ.get("FEACORE_SLOW"), reason="slow; set FEACORE_SLOW=1")


def _suite():
    errors = {k: v for k, v in discover().items() if ".cfd" in k}
    assert not errors, errors
    return [p for p in all_problems() if type(p).__module__.startswith("feacore.problems.cfd")]


PROBLEMS = _suite()
IDS = [p.id for p in PROBLEMS]


def _cells(m, kinds):
    return sum(len(c.data) for c in m.cells if c.type in kinds)


def test_suite_registered():
    assert EXPECTED <= set(IDS)
    for p in PROBLEMS:
        assert "cfd" in p.tags and "flow" in p.tags
        assert p.physics == "flow" and p.formulation in ("2d", "3d")
        assert p.params.get("h0", 0) > 0
        assert p.title and p.family and p.expected_behavior


# --------------------------------------------------------------------------- reference setup


@pytest.mark.parametrize("order", [1, 2])
@pytest.mark.parametrize("problem", PROBLEMS, ids=IDS)
def test_case_check(problem, order):
    case = problem.case("MESH.msh", order=order).check()
    assert case.analysis.physics == "flow"
    assert case.analysis.formulation == problem.formulation
    assert case.analysis.order == order
    assert case.analysis.options.get("residual_tol", 1) <= 1e-6
    m = case.material_for("fluid")
    assert m.rho > 0 and m.mu > 0
    names = [q.name for q in case.qois]
    assert names == [q.name for q in problem.qois()]
    again = Case.from_dict(json.loads(json.dumps(case.to_dict())))
    assert again.to_dict() == case.to_dict()
    refs = problem.references()
    assert refs, "every suite problem has at least one reference value"
    assert len({r.qoi for r in refs}) == len(refs), "one reference per QoI"
    for r in refs:
        assert r.qoi in names
        assert r.kind in REFERENCE_KINDS
        assert 0 < r.tolerance <= 0.05
        assert len(r.source) > 40 and ("http" in r.source or "(19" in r.source or "(20" in r.source)
        assert math.isfinite(r.value) and r.value != 0


@pytest.mark.parametrize("problem", PROBLEMS, ids=IDS)
def test_private_data(problem):
    priv = problem.private()
    assert priv["references"] and priv["reference_case"]["qois"]
    sizes = priv["study_sizes"]
    assert len(sizes) >= 3 and sizes == sorted(sizes, reverse=True)


@pytest.mark.parametrize("problem", PROBLEMS, ids=IDS)
def test_mesh_has_all_groups(problem, tmp_path):
    path = problem.mesh(tmp_path / f"{problem.id}.msh", problem.params["h0"], geom_order=2)
    m = meshio.read(path)
    gdim = {name: int(v[1]) for name, v in m.field_data.items()}
    # finite volumes: linear, fully structured cells (geom_order is ignored)
    kinds = {c.type for c in m.cells}
    if problem.dim == 2:
        assert kinds == {"quad", "line"}, kinds
    else:
        assert "hexahedron" in kinds and not kinds & {"tetra", "tetra10", "quad9", "triangle6"}, kinds
    case = problem.case(path)
    for b in case.bcs:
        assert b.group in gdim, f"BC group {b.group!r} missing from the mesh"
        assert gdim[b.group] == problem.dim - 1, f"{b.type} needs a facet group"
    for r in case.regions:
        assert gdim.get(r.group) == problem.dim
    for q in case.qois:
        if q.group is not None:
            assert gdim.get(q.group) == problem.dim - 1
    # every boundary facet is in exactly one BC group (no unassigned walls)
    bc_groups = {b.group for b in case.bcs}
    assert {g for g, d in gdim.items() if d == problem.dim - 1} == bc_groups
    # points of point QoIs lie in the meshed region's bounding box
    lo, hi = m.points.min(axis=0), m.points.max(axis=0)
    for q in case.qois:
        for pt in (q.point, getattr(q, "point2", None)):
            if pt is not None:
                for k, x in enumerate(pt):
                    assert lo[k] - 1e-9 <= x <= hi[k] + 1e-9
    if problem.dim == 2:   # counter-clockwise cells (+z normal) for the extrusion
        P = m.points[:, :2]
        q4 = np.vstack([c.data for c in m.cells if c.type == "quad"])
        area = sum(P[q4[:, i], 0] * P[q4[:, (i + 1) % 4], 1] - P[q4[:, (i + 1) % 4], 0] * P[q4[:, i], 1]
                   for i in range(4)) / 2
        assert (area > 0).all()


@pytest.mark.parametrize("problem", PROBLEMS, ids=IDS)
def test_refinement_scales_cell_count(problem, tmp_path):
    """h0/2 doubles every block count (4x cells in 2D, 8x in 3D); one nested split gives the same."""
    h0 = problem.params["h0"]
    kind = ("quad",) if problem.dim == 2 else ("hexahedron",)
    n0 = _cells(meshio.read(problem.mesh(tmp_path / "m0.msh", h0)), kind)
    n1 = _cells(meshio.read(problem.mesh(tmp_path / "m1.msh", h0 / 2)), kind)
    nn = _cells(meshio.read(problem.mesh(tmp_path / "mn.msh", h0, refine_steps=1)), kind)
    factor = 2 ** problem.dim
    assert n1 == factor * n0
    assert nn == factor * n0


@pytest.mark.parametrize("problem", PROBLEMS, ids=IDS)
def test_step_export(problem, tmp_path):
    path = problem.export_step(tmp_path / "geometry.step")
    text = path.read_text(errors="ignore")
    assert text.startswith("ISO-10303-21")
    with gmsh_session("reimport"):
        gmsh.model.occ.importShapes(str(path))
        gmsh.model.occ.synchronize()
        assert len(gmsh.model.getEntities(problem.dim)) == 1   # one clean face / solid, no mesh blocks
        bbox = gmsh.model.getBoundingBox(-1, -1)
    with gmsh_session("orig"):
        problem.build_geometry()
        ref = gmsh.model.getBoundingBox(-1, -1)
    assert max(abs(a - b) for a, b in zip(bbox, ref)) < 1e-3 * max(abs(v) for v in ref)


def _forbidden(value: float) -> set[str]:
    """Spellings of a reference value (SI and scaled by 1e-3 / 1e-6) with >= 3 significant digits."""
    out = set()
    for scale in (1.0, 1e-3, 1e-6):
        v = value * scale
        for x in (v, abs(v)):
            for fmt in ("{:g}", "{:.3g}", "{:.4g}", "{:.2f}"):
                s = fmt.format(x)
                if sum(ch.isdigit() for ch in s.lstrip("-0.")) >= 3:
                    out.add(s)
    return out


@pytest.mark.parametrize("problem", PROBLEMS, ids=IDS)
def test_statement_has_no_reference_values(problem):
    st = problem.statement()
    text = json.dumps(st)
    assert "reference" not in st and "references" not in st
    for q in problem.qois():
        assert q.name in st["description"], f"description must name QoI {q.name}"
    for r in problem.references():
        for s in _forbidden(r.value):
            assert s not in text, f"{problem.id}: statement contains reference value spelling {s!r}"
    for word in ("target", "Target", "TARGET", "expected value", "answer", "c_D =", "c_L ="):
        assert word not in st["description"]


# --------------------------------------------------------------------------- exact / reference checks


def test_poiseuille_exact_solution_and_references():
    p = get("poiseuille_channel")
    par = p.params
    H, L, U, mu, t = par["H"], par["L"], par["U"], par["mu"], p.thickness
    y = np.linspace(0, H, 101)
    # the inlet expression is the exact profile, with mean U
    u_in = eval_expr(p.inlet_profile(), np.zeros_like(y), y)
    assert np.allclose(u_in, p.exact("ux", np.c_[np.zeros_like(y), y]), rtol=1e-12, atol=1e-15)
    assert np.isclose(np.trapezoid(u_in, y) / H, U, rtol=1e-3)
    # mu u'' = dp/dx = -G
    assert np.isclose(mu * np.gradient(np.gradient(u_in, y), y)[50], -p.G, rtol=1e-6)
    refs = {r.qoi: r.value for r in p.references()}
    pts = p.points()
    assert np.isclose(refs["dp_AB"], (p.exact("p", [pts["A"]]) - p.exact("p", [pts["B"]]))[0], rtol=1e-12)
    assert np.isclose(refs["u_centreline"], p.exact("ux", [pts["C"]])[0], rtol=1e-12)
    # force balance: pressure drop over the channel = shear on both walls; flow rate = U H t
    assert np.isclose(refs["wall_force_bottom"] + refs["wall_force_top"], p.G * L * H * t, rtol=1e-12)
    assert np.isclose(refs["flow_rate_outlet"], U * H * t)
    assert np.isclose(p.reynolds, 10.0)


def test_pipe_exact_solution_and_references():
    p = get("pipe_poiseuille_3d")
    par = p.params
    R, L, U, mu = par["R"], par["L"], par["U"], par["mu"]
    r = np.linspace(0, R, 2001)
    zero = np.zeros_like(r)
    u = eval_expr(p.inlet_profile(), zero, r, zero)
    assert np.allclose(u, p.exact("ux", np.c_[zero, r, zero]), rtol=1e-12, atol=1e-15)
    # mean over the section = U; (mu / r) d/dr (r du/dr) = -G
    assert np.isclose(np.trapezoid(u * 2 * np.pi * r, r) / (np.pi * R**2), U, rtol=1e-5)
    assert np.isclose(mu * np.gradient(r * np.gradient(u, r), r)[1000] / r[1000], -p.G, rtol=1e-4)
    refs = {ref.qoi: ref.value for ref in p.references()}
    pts = p.points()
    assert np.isclose(refs["dp_AB"], (p.exact("p", [pts["A"]]) - p.exact("p", [pts["B"]]))[0], rtol=1e-12)
    assert np.isclose(refs["u_centreline"], 2 * U)
    assert np.isclose(refs["wall_force"], p.G * L * np.pi * R**2, rtol=1e-12)   # force balance
    assert np.isclose(refs["flow_rate_outlet"], np.pi * R**2 * U)
    assert np.isclose(p.reynolds, 10.0)


def test_dfg_2d1_reference_conversion():
    from feacore.problems.cfd.cylinder import CD_REF, CL_REF, DP_REF

    p = get("cylinder_channel_dfg_2d1")
    assert np.isclose(p.reynolds, 20.0)
    assert np.isclose(p.U_mean, 0.2)
    refs = {r.qoi: r.value for r in p.references()}
    assert np.isclose(refs["drag"], CD_REF * 0.5 * 1.0 * 0.2**2 * 0.1)
    assert np.isclose(refs["lift"], CL_REF * 0.002)
    assert refs["dp_front_rear"] == DP_REF
    # published acceptance intervals (Schaefer & Turek 1996)
    assert 5.57 <= CD_REF <= 5.59 and 0.0104 <= CL_REF <= 0.0110 and 0.1172 <= DP_REF <= 0.1176
    # the inflow profile has maximum Um at mid-height and mean 2 Um / 3
    y = np.linspace(0, p.params["H"], 2001)
    u = eval_expr(p.inlet_profile(), np.zeros_like(y), y)
    assert np.isclose(u.max(), p.params["Um"], rtol=1e-6)
    assert np.isclose(np.trapezoid(u, y) / p.params["H"], p.U_mean, rtol=1e-5)
    # the pressure probes are the front and rear points of the cylinder surface
    q = next(q for q in p.qois() if q.name == "dp_front_rear")
    assert np.allclose(q.point, [0.15, 0.2]) and np.allclose(q.point2, [0.25, 0.2])


def test_cavity_points_are_ghia_grid_points():
    p = get("lid_driven_cavity_re100")
    assert np.isclose(p.reynolds, 100.0)
    for q in p.qois():
        assert q.type == "point"
        for c in q.point:            # Ghia et al. tabulate on a uniform 129-point grid (multiples of 1/128)
            assert np.isclose(c * 128, round(c * 128))
    # the table prints the grid coordinates to 4 decimals
    printed = {"u_vcl_min": (0.5, 0.4531), "v_hcl_max": (0.2344, 0.5), "v_hcl_min": (0.8047, 0.5),
               "u_centre": (0.5, 0.5)}
    assert {q.name for q in p.qois()} == set(printed)
    for q in p.qois():
        assert np.allclose(q.point, printed[q.name], atol=5e-5)


# --------------------------------------------------------------------------- solves (OpenFOAM)


def _openfoam_or_skip():
    if importlib.util.find_spec("feacore.solver.openfoam_backend") is None:
        pytest.skip("OpenFOAM backend not available")
    from feacore.solver._foam_env import find_openfoam

    if find_openfoam() is None:
        pytest.skip("OpenFOAM not installed")


# Coarse solve per problem: (mesh size as a fraction of h0, relative tolerance). The reference
# studies (docs/benchmarks_cfd.md) check the reference tolerances on refined meshes; these bounds
# only catch a broken setup or backend. The pipe's h0 section is a 16-gon (7 % off), so it uses h0/2.
COARSE = {
    "poiseuille_channel": (1.0, 0.05),
    "lid_driven_cavity_re100": (1.0, 0.06),     # Ghia's table is itself 1.5-3.5 % off
    "cylinder_channel_dfg_2d1": (1.0, 0.02),
    "pipe_poiseuille_3d": (0.5, 0.05),
}


@pytest.mark.parametrize("pid", sorted(COARSE))
def test_coarse_solve_matches_references(pid, tmp_path):
    _openfoam_or_skip()
    from feacore.solver import solve

    problem = get(pid)
    factor, tol = COARSE[pid]
    path = problem.mesh(tmp_path / f"{pid}.msh", problem.params["h0"] * factor)
    try:
        res = solve(problem.case(path, order=2), backend="openfoam")
    except NotImplementedError as e:   # backend feature still missing
        pytest.skip(f"openfoam backend: {e}")
    assert res.info.backend == "openfoam" and res.info.physics == "flow"
    assert not [w for w in res.info.warnings if "NOT CONVERGED" in w], res.info.warnings
    values = res.evaluate_all()
    for ref in problem.references():
        v = values[ref.qoi]
        assert math.isfinite(v)
        assert abs(v - ref.value) <= tol * abs(ref.value), (pid, ref.qoi, v, ref.value)


def test_first_order_upwind_runs(tmp_path):
    """analysis.order = 1 (upwind convection) also solves; for fully developed flow convection
    vanishes, so it matches the second-order result."""
    _openfoam_or_skip()
    from feacore.solver import solve

    problem = get("poiseuille_channel")
    path = problem.mesh(tmp_path / "p.msh", problem.params["h0"])
    v1 = solve(problem.case(path, order=1), backend="openfoam").evaluate_all()
    v2 = solve(problem.case(path, order=2), backend="openfoam").evaluate_all()
    for k in v2:
        assert v1[k] == pytest.approx(v2[k], rel=1e-3)


@slow
@pytest.mark.parametrize("pid,nested", [("poiseuille_channel", True), ("lid_driven_cavity_re100", False),
                                        ("cylinder_channel_dfg_2d1", True), ("pipe_poiseuille_3d", False)])
def test_reference_study(pid, nested, tmp_path):
    """Full 3-mesh reference study (10 s to ~1.5 min): every reference QoI passes, and the QoIs that
    are clean second order in the validation (docs/benchmarks_cfd.md) show it."""
    _openfoam_or_skip()
    from feacore.verification.study import convergence_study

    problem = get(pid)
    study = convergence_study(problem, sizes=problem.study_sizes()[:3], order=2, backend="openfoam",
                              nested=nested, workdir=tmp_path)
    assert study["passed"], [(q["name"], q["values"], q["err_fine"]) for q in study["qois"]]
    second_order = {"poiseuille_channel": ["dp_AB", "u_centreline", "flow_rate_outlet"],
                    "lid_driven_cavity_re100": ["v_hcl_min", "u_centre"],
                    "cylinder_channel_dfg_2d1": ["dp_front_rear"],
                    "pipe_poiseuille_3d": ["dp_AB", "flow_rate_outlet"]}[pid]
    for q in study["qois"]:
        if q["name"] in second_order:
            assert 1.7 <= q["gci"]["p"] <= 2.6, (q["name"], q["gci"]["p"])
