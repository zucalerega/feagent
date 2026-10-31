"""Failure-mode detectors on benchmarks (should come out clean, estimators calibrated) and on
synthetic setups built to trigger one failure mode each."""

from __future__ import annotations

import gmsh
import numpy as np
import pytest

from feacore.case import BC, Analysis, Case, Material, QoI, Region
from feacore.diagnostics import diagnose, precheck
from feacore.diagnostics.flags import Report
from feacore.diagnostics.score import ZZ_ENERGY_EFFECTIVITY, score_population, single_mesh_errors
from feacore.diagnostics.verify import p_sensitivity, refinement_study
from feacore.geometry.gmsh_tools import add_group, gmsh_session, in_box, set_uniform_size
from feacore.problems.registry import get

# ----------------------------------------------------------------------------- fixtures


def _mesh(path, build, h, order=2):
    with gmsh_session("t"):
        build()
        set_uniform_size(h)
        gmsh.model.mesh.generate(2)
        if order == 2:
            gmsh.model.mesh.setOrder(2)
        gmsh.write(str(path))
    return str(path)


def _rect(x0, y0, w, hgt, groups):
    occ = gmsh.model.occ
    s = occ.addRectangle(x0, y0, 0, w, hgt)
    occ.synchronize()
    add_group(2, [s], "domain")
    for name, box in groups.items():
        add_group(1, in_box(1, **box), name)


def cantilever(path, h=0.25, order=2):
    """10 x 1 plate, clamped at x = 0, tip shear at x = 10."""
    return _mesh(path, lambda: _rect(0, 0, 10, 1, {"root": dict(xmax=0), "tip": dict(xmin=10)}), h, order)


def l_shape(path, h=0.2, order=2):
    """[0,2]x[0,1] U [0,1]x[0,2]; re-entrant corner at (1, 1)."""
    def build():
        occ = gmsh.model.occ
        a = occ.addRectangle(0, 0, 0, 2, 1)
        b = occ.addRectangle(0, 0, 0, 1, 2)
        out, _ = occ.fuse([(2, a)], [(2, b)])
        occ.synchronize()
        add_group(2, [t for _, t in gmsh.model.getEntities(2)], "domain")
        add_group(1, in_box(1, xmax=0), "left")
        add_group(1, in_box(1, ymax=0), "bottom")
        add_group(1, in_box(1, xmin=2), "right")
        add_group(1, in_box(1, ymin=2), "top")
    return _mesh(path, build, h, order)


def case(mesh, bcs, order=2, E=210e9, qois=()):
    return Case(mesh=mesh, analysis=Analysis(formulation="plane_stress", order=order, thickness=0.1),
                materials={"m": Material(E=E, nu=0.3)}, bcs=bcs,
                regions=[Region("domain", "m")], qois=list(qois))


CANTI_BCS = [BC("fixed", group="root"), BC("traction", group="tip", value=[0.0, -1e6])]
L_BCS = [BC("symmetry", group="left"), BC("symmetry", group="bottom"),
         BC("traction", group="right", value=[1e6, 0.0]), BC("traction", group="top", value=[0.0, 1e6])]


# ----------------------------------------------------------------------------- benchmarks


@pytest.fixture(scope="module")
def lame(tmp_path_factory):
    p = get("lame_cylinder_plane_strain")
    d = tmp_path_factory.mktemp("lame")
    return p, [p.case(p.mesh(d / f"m{i}.msh", h), order=2) for i, h in enumerate(p.study_sizes(2))]


def test_lame_is_clean_and_zz_effectivity_is_stable(lame):
    p, cases = lame
    U = next(r.value for r in p.references() if r.qoi == "strain_energy")
    eff = []
    for c in cases:
        rep, res = diagnose(c)
        assert rep.status == "ok", rep.ids()
        eff.append(rep.metrics["zz_energy_error"] / (abs(rep.metrics["strain_energy"] - U) / U))
    # the calibration constant in score.py must match what the benchmark measures
    assert all(abs(e - ZZ_ENERGY_EFFECTIVITY) < 0.15 for e in eff), eff


def test_le1_refinement_study_estimates_true_error():
    p = get("nafems_le1")
    import tempfile
    d = tempfile.mkdtemp()
    h = p.study_sizes(2)[0]
    study = refinement_study(lambda s: p.case(p.mesh(f"{d}/m{s}.msh", h * s), order=2))
    q = study["qois"]["syy_D"]
    assert q["kind"] == "converged"
    ref = next(r for r in p.references() if r.qoi == "syy_D")
    true = abs(q["values"][1] - ref.value) / ref.value
    # GCI-based estimate within a factor 3 of the true error, and phi_ext near the reference
    assert true / 3 < q["est_err"] < 3 * true + 1e-3, (q["est_err"], true)
    assert abs(q["phi_ext"] - ref.value) / ref.value < 0.01


# ----------------------------------------------------------------------------- failure modes


def test_rigid_body_mode_caught_before_solving(tmp_path):
    m = cantilever(tmp_path / "c.msh")
    rep = precheck(case(m, [BC("traction", group="tip", value=[0.0, -1e6])]))
    assert rep.status == "fail" and "rigid_body_mode" in rep.ids()
    roller = [BC("displacement", group="root", components=[0], value=0.0), CANTI_BCS[1]]
    rep, res = diagnose(case(m, roller))
    assert res is None and "rigid_body_mode" in rep.ids()
    assert "ty" in rep.flags[0].message


def test_disconnected_body_is_unsupported(tmp_path):
    def build():
        _rect(0, 0, 1, 1, {"fix": dict(xmax=0)})
        s2 = gmsh.model.occ.addRectangle(2, 0, 0, 1, 1)
        gmsh.model.occ.synchronize()
        add_group(2, [s2], "domain2")
        add_group(1, in_box(1, xmin=3), "pull")
    m = _mesh(tmp_path / "d.msh", build, 0.2)
    c = case(m, [BC("fixed", group="fix"), BC("traction", group="pull", value=[1e6, 0.0])])
    c.regions.append(Region("domain2", "m"))
    rep = precheck(c)
    assert "disconnected_bodies" in rep.ids() and "rigid_body_mode" in rep.ids()


def test_no_load(tmp_path):
    rep = precheck(case(cantilever(tmp_path / "c.msh"), [BC("fixed", group="root")]))
    assert "no_load" in rep.ids()


def test_point_constraint_warned(tmp_path):
    m = cantilever(tmp_path / "c.msh")
    bcs = [BC("displacement", group="root", components=[0], value=0.0),
           BC("fixed", point=[0.0, 0.0]), CANTI_BCS[1]]
    rep = precheck(case(m, bcs))
    assert rep.status == "warn" and "point_constraint" in rep.ids()


def test_unit_mismatch_is_a_blowup(tmp_path):
    # E in MPa-ish units while loads are in Pa: displacements explode
    rep, _ = diagnose(case(cantilever(tmp_path / "c.msh"), CANTI_BCS, E=210e3))
    assert rep.status == "fail"
    assert {"large_displacement", "large_strain"} & set(rep.ids())


def test_clamped_root_peak_is_flagged_as_support_singularity(tmp_path):
    rep, _ = diagnose(case(cantilever(tmp_path / "c.msh"), CANTI_BCS))
    assert "peak_at_support" in rep.ids()
    assert rep.metrics["vm_trusted"] < rep.metrics["vm_max"]


def test_reentrant_corner_peak_is_singular_energy_converges_slowly(tmp_path):
    rep, _ = diagnose(case(l_shape(tmp_path / "l.msh"), L_BCS))
    assert np.allclose(rep.metrics["vm_peak_at"], [1, 1], atol=0.05)
    qois = [QoI("vm_max", "max", field="von_mises")]
    study = refinement_study(lambda s: case(l_shape(tmp_path / f"l{s}.msh", 0.2 * s), L_BCS, qois=qois))
    assert study["qois"]["vm_max"]["kind"] == "singular"
    U = study["qois"]["strain_energy"]
    # finite, but the corner singularity (lambda = 0.544) caps the energy rate near 2 * lambda
    assert U["kind"] == "reduced_order" and 0.7 < U["p"] < 1.6, U["p"]
    assert U["est_err"] is not None


def test_linear_elements_lock_in_bending(tmp_path):
    m = cantilever(tmp_path / "c1.msh", h=0.5, order=1)
    rep = Report()
    out = p_sensitivity(case(m, CANTI_BCS, order=1), rep)
    assert "p_sensitive" in rep.ids(), out["strain_energy"]


# ----------------------------------------------------------------------------- scoring


def test_score_prefers_cheapest_accurate_setup(tmp_path):
    agents = []
    for i, h in enumerate([0.5, 0.25, 0.1]):
        rep, _ = diagnose(case(cantilever(tmp_path / f"c{i}.msh", h=h), CANTI_BCS))
        agents.append({"agent_id": f"a{i}", "report": rep, "errors": single_mesh_errors(rep)})
    bad = precheck(case(cantilever(tmp_path / "c0.msh", h=0.5), [CANTI_BCS[1]]))
    agents.append({"agent_id": "broken", "report": bad, "errors": {}})
    rows = score_population(agents, tol={"strain_energy": 0.01})
    by = {r["agent_id"]: r for r in rows}
    assert by["broken"]["score"] == 0 and by["broken"]["pareto_rank"] is None
    passing = [r for r in rows if r["meets_tol"]]
    assert passing, rows
    best = max(rows, key=lambda r: r["score"])
    assert best["score"] == pytest.approx(1.0) and best["cost"] == min(r["cost"] for r in passing)


def test_lame_sphere_3d_is_clean(tmp_path):
    p = get("lame_sphere_3d")
    rep, _ = diagnose(p.case(p.mesh(tmp_path / "m.msh", p.study_sizes(2)[0]), order=2))
    assert rep.status == "ok" and "over_constraint" not in rep.ids(), rep.ids()


def test_block_squeezed_between_walls_is_over_constrained(tmp_path):
    m = _mesh(tmp_path / "b.msh", lambda: _rect(0, 0, 1, 1, {"l": dict(xmax=0), "r": dict(xmin=1),
                                                              "top": dict(ymin=1)}), 0.2)
    bcs = [BC("fixed", group="l"), BC("fixed", group="r"), BC("pressure", group="top", value=1e6)]
    rep, _ = diagnose(case(m, bcs))
    assert "over_constraint" in rep.ids(), rep.metrics.get("reaction_ratio")
