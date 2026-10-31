"""Tests for the test parts (feacore/problems/parts).

Fast by default: every registered part is checked for a valid reference case, a reference mesh
at h0 with every group the case uses, a STEP that re-imports in gmsh, and an agent statement
free of reference values. One coarse solve per 2D family runs by default; the 3D solves are
slow and run only with FEACORE_SLOW_TESTS=1.
"""

from __future__ import annotations

import json
import math
import os

import gmsh
import meshio
import numpy as np
import pytest

from feacore.problems.registry import all_problems, discover

os.environ.setdefault("OMP_NUM_THREADS", "1")

SLOW = pytest.mark.skipif(not os.environ.get("FEACORE_SLOW_TESTS"),
                          reason="slow 3D solve; set FEACORE_SLOW_TESTS=1")


def _parts():
    discover()
    return all_problems(tags=["part"])


PARTS = _parts()
IDS = [p.id for p in PARTS]
FAMILIES = sorted({p.family for p in PARTS})


def test_part_modules_import_cleanly():
    errs = {m: e for m, e in discover().items() if ".parts." in m}
    assert not errs, errs


def test_families_present():
    expected = {"stepped_shaft", "notched_bar", "l_bracket", "clevis_lug", "mounting_bracket",
                "pressure_vessel_axisym"}
    assert expected <= set(FAMILIES)
    for fam in FAMILIES:
        assert len([p for p in PARTS if p.family == fam]) >= 2, fam


@pytest.mark.parametrize("problem", PARTS, ids=IDS)
def test_metadata(problem):
    assert "part" in problem.tags
    assert problem.title and problem.family
    assert problem.dim == (3 if problem.formulation == "3d" else 2)
    assert "h0" in problem.params and problem.params["h0"] > 0
    json.dumps(problem.statement())  # serialisable
    for name in problem.expectations():
        assert name in {q.name for q in problem.qois()}
    for ref in problem.references():
        assert ref.qoi in {q.name for q in problem.qois()}
        assert ref.value != 0 and math.isfinite(ref.value)
        assert 0 < ref.tolerance <= 0.1
        assert ref.source


@pytest.mark.parametrize("problem", PARTS, ids=IDS)
def test_case_checks(problem):
    case = problem.case("MESH.msh", order=2).check()
    assert {q.name for q in case.qois} == {q.name for q in problem.qois()}
    case1 = problem.case("MESH.msh", order=1).check()
    assert case1.analysis.order == 1
    assert case.analysis.formulation == problem.formulation
    # round trip through JSON
    from feacore.case import Case
    assert Case.from_dict(json.loads(json.dumps(case.to_dict()))).to_dict() == case.to_dict()


def _referenced_groups(case) -> set[str]:
    names = {b.group for b in case.bcs if b.group}
    names |= {q.group for q in case.qois if q.group}
    names |= {r.group for r in case.regions if r.group != "*"}
    names |= {l.group for l in case.loads if l.group != "*"}
    return names


@pytest.mark.parametrize("problem", PARTS, ids=IDS)
def test_mesh_at_h0_has_groups(problem, tmp_path):
    path = problem.mesh(tmp_path / "m.msh", problem.params["h0"])
    m = meshio.read(path)
    names = set(m.field_data)
    missing = _referenced_groups(problem.case(path)) - names
    assert not missing, f"groups missing from the mesh: {missing}; have {sorted(names)}"
    top = "tetra10" if problem.dim == 3 else "triangle6"
    assert sum(len(c.data) for c in m.cells if c.type == top) > 100
    for name in _referenced_groups(problem.case(path)):
        assert sum(len(v) for v in m.cell_sets[name] if v is not None) > 0, name


def _occ_mass(dim: int) -> float:
    return sum(gmsh.model.occ.getMass(dim, t) for _, t in gmsh.model.getEntities(dim))


@pytest.mark.parametrize("problem", PARTS, ids=IDS)
def test_step_export_reimports(problem, tmp_path):
    step = problem.export_step(tmp_path / "geometry.step")
    assert step.stat().st_size > 0
    from feacore.geometry.gmsh_tools import gmsh_session
    with gmsh_session("reimport"):
        gmsh.model.occ.importShapes(str(step))
        gmsh.model.occ.synchronize()
        ents = gmsh.model.getEntities(problem.dim)
        assert len(ents) == 1, f"expected one {problem.dim}D entity, got {ents}"
        mass_step = _occ_mass(problem.dim)
    # the reference mesh geometry is the stated fraction of the part
    with gmsh_session("ref"):
        problem.build_geometry()
        mass_ref = _occ_mass(problem.dim)
    frac = {"full": 1.0, "half": 0.5, "quarter": 0.25}[getattr(problem, "reference_portion", "full")]
    assert mass_ref == pytest.approx(frac * mass_step, rel=1e-6)


def _number_forms(v: float) -> set[str]:
    forms = set()
    for x in (v, v / 1e6):
        for d in (3, 4, 5):
            forms.add(f"{x:.{d}g}")
        forms.add(f"{x:.1f}")
        forms.add(f"{x:.2f}")
    forms.add(f"{v:.3e}")
    return {f for f in forms if sum(ch.isdigit() for ch in f) >= 3}


@pytest.mark.parametrize("problem", PARTS, ids=IDS)
def test_statement_has_no_reference_values(problem):
    text = json.dumps(problem.statement())
    assert "Kt" not in problem.description()
    for ref in problem.references():
        for form in _number_forms(ref.value):
            assert form not in text, f"reference value {ref.value} appears in the statement as {form!r}"
    for key in ("reference", "expectation", "tolerance"):
        assert key not in problem.statement()


def test_kt_fits_are_continuous_and_match_limits():
    from feacore.problems.parts._common import (kt_opposite_u_notches_tension,
                                                kt_shoulder_fillet_round_bar_tension)
    # the two coefficient sets of each fit meet at depth/radius = 2
    d = 1.0
    for D in (1.2, 1.5, 2.0):
        h = (D - d) / 2
        lo = kt_shoulder_fillet_round_bar_tension(D, d, h / (2.0 - 1e-9))
        hi = kt_shoulder_fillet_round_bar_tension(D, d, h / (2.0 + 1e-9))
        assert lo == pytest.approx(hi, rel=0.01)
    for H in (10.0, 5.0, 2.5):
        lo = kt_opposite_u_notches_tension(H, 1.0, 1.0 / (2.0 - 1e-9))
        hi = kt_opposite_u_notches_tension(H, 1.0, 1.0 / (2.0 + 1e-9))
        assert lo == pytest.approx(hi, rel=0.015)
    # shallow semicircular edge notch -> 3.065 (semi-infinite plate)
    assert kt_opposite_u_notches_tension(1e6, 1.0, 1.0) == pytest.approx(3.065, rel=0.01)
    # monotone in r: sharper fillet -> higher Kt
    assert kt_shoulder_fillet_round_bar_tension(2, 1, 0.05) > kt_shoulder_fillet_round_bar_tension(2, 1, 0.1) \
        > kt_shoulder_fillet_round_bar_tension(2, 1, 0.2)


# --------------------------------------------------------------------------- solves


def _solve_coarse(problem, tmp_path, scale=1.0):
    from feacore.solver import solve
    path = problem.mesh(tmp_path / "m.msh", problem.params["h0"] * scale)
    res = solve(problem.case(path, order=2), backend="skfem")
    vals = res.evaluate_all()
    assert all(math.isfinite(v) for v in vals.values()), vals
    return vals


def _one_per_family(dim3: bool):
    out = {}
    for p in PARTS:
        if (p.dim == 3) == dim3 and p.family not in out and "singular_trap" not in p.tags:
            out[p.family] = p
    return list(out.values())


def _check_against_refs(problem, vals, rel):
    for ref in problem.references():
        assert vals[ref.qoi] == pytest.approx(ref.value, rel=rel), (ref.qoi, vals[ref.qoi], ref.value)


@pytest.mark.parametrize("problem", _one_per_family(False), ids=lambda p: p.id)
def test_coarse_solve_2d(problem, tmp_path):
    pytest.importorskip("feacore.solver.skfem_backend")
    vals = _solve_coarse(problem, tmp_path)
    _check_against_refs(problem, vals, rel=0.05)


@SLOW
@pytest.mark.parametrize("problem", _one_per_family(True), ids=lambda p: p.id)
def test_coarse_solve_3d(problem, tmp_path):
    pytest.importorskip("feacore.solver.skfem_backend")
    vals = _solve_coarse(problem, tmp_path)
    _check_against_refs(problem, vals, rel=0.05)


def test_sharp_shoulder_trap_grows_under_refinement(tmp_path):
    """The r = 0 axisymmetric shoulder is singular: the corner peak must grow with refinement."""
    pytest.importorskip("feacore.solver.skfem_backend")
    from feacore.problems.registry import get
    p = get("stepped_shaft_axi_Dd2_sharp")
    v1 = _solve_coarse(p, tmp_path, 1.0)["peak_stress_shoulder"]
    v2 = _solve_coarse(p, tmp_path, 0.5)["peak_stress_shoulder"]
    assert v2 > 1.15 * v1


def test_symmetric_reference_matches_full_part(tmp_path):
    """Half-model reference (L-bracket) agrees with a full-part solve (coarse mesh)."""
    pytest.importorskip("feacore.solver.skfem_backend")
    if not os.environ.get("FEACORE_SLOW_TESTS"):
        pytest.skip("slow 3D solve; set FEACORE_SLOW_TESTS=1")
    from feacore.problems.parts.l_bracket import LBracket
    half = LBracket(id="lb_half", fillet_radius=5e-3)
    full = LBracket(id="lb_full", fillet_radius=5e-3)
    full.reference_portion = "full"
    vh = _solve_coarse(half, tmp_path)
    vf = _solve_coarse(full, tmp_path)
    assert vh["load_displacement"] == pytest.approx(vf["load_displacement"], rel=0.01)
    assert vh["max_vm_corner"] == pytest.approx(vf["max_vm_corner"], rel=0.05)
    assert np.isfinite(vh["max_vm_corner"])
