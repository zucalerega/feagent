"""Tests for the NAFEMS / published-reference benchmark suite (``feacore.problems.nafems``).

For every problem of the suite: the reference case validates, the reference mesh at h0 has every
group the case uses, STEP export works, the agent-facing statement leaks no reference value, and
the fixed solver reproduces the published target on the h0 mesh (seconds per problem).

Slow tests (full 3-mesh reference studies of the 3D problems, ~20-60 s each) run only with
``FEACORE_SLOW=1``.
"""

from __future__ import annotations

import json
import math
import os

import pytest

gmsh = pytest.importorskip("gmsh")
meshio = pytest.importorskip("meshio")

from feacore.case import Case  # noqa: E402
from feacore.geometry.gmsh_tools import gmsh_session  # noqa: E402
from feacore.problems.registry import all_problems, discover, get  # noqa: E402

EXPECTED = {"nafems_le1", "nafems_le10", "nafems_le11_axisym", "nafems_le11_3d", "cooks_membrane",
            "nafems_t4"}
REFERENCE_KINDS = {"analytical", "published", "handbook", "manufactured", "converged_numerical",
                   "published_numerical"}
slow = pytest.mark.skipif(not os.environ.get("FEACORE_SLOW"), reason="slow; set FEACORE_SLOW=1")


def _suite():
    discover()
    return [p for p in all_problems() if type(p).__module__.startswith("feacore.problems.nafems")]


PROBLEMS = _suite()
IDS = [p.id for p in PROBLEMS]
TWO_D = [p.id for p in PROBLEMS if p.dim == 2]
THREE_D = [p.id for p in PROBLEMS if p.dim == 3]


def test_suite_registered():
    assert EXPECTED <= set(IDS)
    for p in PROBLEMS:
        assert "nafems" in p.tags
        assert p.params.get("h0", 0) > 0
        assert p.title and p.family and p.expected_behavior


# --------------------------------------------------------------------------- reference setup


@pytest.mark.parametrize("order", [1, 2])
@pytest.mark.parametrize("problem", PROBLEMS, ids=IDS)
def test_case_check(problem, order):
    case = problem.case("MESH.msh", order=order).check()
    assert case.analysis.physics == problem.physics
    assert case.analysis.formulation == problem.formulation
    assert case.analysis.order == order
    names = [q.name for q in case.qois]
    assert names == [q.name for q in problem.qois()]
    # JSON round trip
    again = Case.from_dict(json.loads(json.dumps(case.to_dict())))
    assert again.to_dict() == case.to_dict()
    refs = problem.references()
    assert refs, "every suite problem has at least one reference value"
    for r in refs:
        assert r.qoi in names
        assert r.kind in REFERENCE_KINDS
        assert 0 < r.tolerance <= 0.05
        assert len(r.source) > 40 and ("http" in r.source or "(19" in r.source)
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
    cell_types = {c.type for c in m.cells}
    assert ("triangle6" if problem.dim == 2 else "tetra10") in cell_types
    case = problem.case(path)
    for b in case.bcs:
        if b.group is not None:
            assert b.group in gdim, f"BC group {b.group!r} missing from the mesh"
            assert gdim[b.group] < problem.dim
            if b.type not in ("fixed", "displacement", "temperature"):
                assert gdim[b.group] == problem.dim - 1, f"{b.type} needs a facet group"
    for ld in case.loads:
        if ld.group != "*":
            assert gdim.get(ld.group) == problem.dim
    for q in case.qois:
        if q.group is not None:
            assert q.group in gdim
    # every domain cell belongs to a named group of the mesh dimension
    assert any(d == problem.dim for d in gdim.values())
    # point QoIs lie in the meshed region's bounding box
    lo, hi = m.points.min(axis=0), m.points.max(axis=0)
    for q in case.qois:
        if q.point is not None:
            for k, x in enumerate(q.point):
                assert lo[k] - 1e-9 <= x <= hi[k] + 1e-9


@pytest.mark.parametrize("problem", PROBLEMS, ids=IDS)
def test_step_export(problem, tmp_path):
    path = problem.export_step(tmp_path / "geometry.step")
    text = path.read_text(errors="ignore")
    assert text.startswith("ISO-10303-21")
    with gmsh_session("reimport"):
        gmsh.model.occ.importShapes(str(path))
        gmsh.model.occ.synchronize()
        assert len(gmsh.model.getEntities(problem.dim)) == 1
        bbox = gmsh.model.getBoundingBox(-1, -1)
    with gmsh_session("orig"):
        problem.build_geometry()
        ref = gmsh.model.getBoundingBox(-1, -1)
    assert max(abs(a - b) for a, b in zip(bbox, ref)) < 1e-3 * max(abs(v) for v in ref)


def _forbidden(value: float) -> set[str]:
    """Spellings of a reference value (SI and scaled by 1e-6 / 1e-3) with >= 3 significant digits."""
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
    for word in ("target", "Target", "TARGET", "expected value", "answer"):
        assert word not in st["description"]


# --------------------------------------------------------------------------- solves


def _solve(problem, tmp_path, order=2):
    pytest.importorskip("feacore.solver.skfem_backend")
    from feacore.solver import solve

    path = problem.mesh(tmp_path / f"{problem.id}_o{order}.msh", problem.params["h0"], geom_order=order)
    try:
        return solve(problem.case(path, order=order), backend="skfem")
    except NotImplementedError as e:  # backend feature still missing
        pytest.skip(f"skfem backend: {e}")


@pytest.mark.parametrize("pid", sorted(EXPECTED))
def test_h0_solve_matches_reference_p2(pid, tmp_path):
    """P2 on the h0 reference mesh is already within the reference tolerance."""
    problem = get(pid)
    values = _solve(problem, tmp_path, order=2).evaluate_all()
    for ref in problem.references():
        v = values[ref.qoi]
        assert math.isfinite(v)
        assert abs(v - ref.value) <= ref.tolerance * abs(ref.value), (pid, ref.qoi, v, ref.value)


@pytest.mark.parametrize("pid", sorted(EXPECTED & {"nafems_le1", "nafems_le11_axisym", "cooks_membrane",
                                                    "nafems_t4"}))
def test_h0_solve_p1_2d(pid, tmp_path):
    problem = get(pid)
    values = _solve(problem, tmp_path, order=1).evaluate_all()
    for ref in problem.references():
        assert abs(values[ref.qoi] - ref.value) <= 0.05 * abs(ref.value)


def _study(pid, order, tmp_path):
    pytest.importorskip("feacore.solver.skfem_backend")
    from feacore.verification.study import convergence_study

    return convergence_study(get(pid), order=order, backend="skfem", workdir=tmp_path)


@pytest.mark.parametrize("pid", ["nafems_le1", "nafems_le11_axisym", "cooks_membrane", "nafems_t4"])
def test_reference_study_2d_p2(pid, tmp_path):
    """Three-mesh P2 study: monotone convergence, extrapolation within tolerance (~1-4 s each)."""
    study = _study(pid, 2, tmp_path)
    assert study["passed"] is True
    for q in study["qois"]:
        assert q["gci"]["convergence"] == "monotone"
        assert "p_far_below_theory" not in q["gci"]["flags"]


@slow
@pytest.mark.parametrize("pid", ["nafems_le10", "nafems_le11_3d"])
def test_reference_study_3d_p2(pid, tmp_path):
    study = _study(pid, 2, tmp_path)
    for q in study["qois"]:
        ref = q["reference"]
        assert abs(q["values"][-1] - ref["value"]) <= ref["tolerance"] * abs(ref["value"])
