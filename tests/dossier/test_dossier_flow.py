"""The flow analysis engine of dossier (dossier/analysis/flow.py) and the RANS validation it rests on.

Fast tests (no OpenFOAM): the friction formula, the reading of a STEP file as a stepped bore (the example's
flow_domain.step and the body, which is not a bore), the hexahedral mesh, the uncertainty rule, and every
not_supported answer of `run` (no flow domain, no OpenFOAM, a Reynolds number outside the validated range).

With OpenFOAM (skipped without it): the plumbing of the engine on laminar flow in a straight bore against
the Hagen-Poiseuille value (STEP -> mesh -> solve -> pressure drop -> record).

Slow (pytest -m slow, OpenFOAM): the k-omega SST validation on developed pipe flow against the Colebrook
formula at Re 1e5 and 1.384e6 (the example's Reynolds number), the registered benchmark problem
pipe_turbulent_3d, and the example valve with opts.quick.
"""

from __future__ import annotations

import math
from types import SimpleNamespace
import shutil

import numpy as np
import pytest

gmsh = pytest.importorskip("gmsh")

from dossier import context as context_mod  # noqa: E402
from dossier.analysis import flow  # noqa: E402
from dossier.model import FeatureDef, Value  # noqa: E402
from dossier.pipeline import Options  # noqa: E402

HAS_FOAM = flow.availability()[0]
needs_foam = pytest.mark.skipif(not HAS_FOAM, reason="OpenFOAM or foamlib is not installed")

INLET = FeatureDef(description="inlet", select={"kind": "plane", "normal": [-1, 0, 0], "near": [-0.075, 0, 0],
                                                  "within": 0.002}, expect=1)
OUTLET = FeatureDef(description="outlet", select={"kind": "plane", "normal": [1, 0, 0], "near": [0.075, 0, 0],
                                                    "within": 0.002}, expect=1)


def _cylinder_step(path, r=0.0125, x0=-0.075, length=0.15):
    gmsh.initialize(readConfigFiles=False, interruptible=False)
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("bore")
        gmsh.model.occ.addCylinder(x0, 0, 0, length, 0, 0, r)
        gmsh.model.occ.synchronize()
        gmsh.write(str(path))
    finally:
        gmsh.finalize()
    return path


def _flow_lc(project, ctx):
    from dossier.loadcases import plan
    lcs = [lc for lc in plan(project, ctx, None).load_cases if lc.analysis_type == "flow"]
    assert lcs, "the example has no flow load case"
    return lcs[0]


# --------------------------------------------------------------------------- formulas, geometry, mesh


def test_colebrook_smooth_and_rough():
    # Prandtl's law: 1/sqrt(f) = 2 log10(Re sqrt(f)) - 0.8 (the smooth Colebrook form to 0.1 %)
    for re in (1e5, 1.384e6):
        f = flow.colebrook(re)
        assert 1 / math.sqrt(f) == pytest.approx(-2 * math.log10(2.51 / (re * math.sqrt(f))), rel=1e-10)
        assert 1 / math.sqrt(f) == pytest.approx(2 * math.log10(re * math.sqrt(f)) - 0.8, rel=2e-3)
    assert flow.colebrook(1e5) == pytest.approx(0.01799, rel=1e-3)
    assert flow.colebrook(1.384e6, 3.2e-6 / 0.025) > flow.colebrook(1.384e6)


def test_example_flow_domain_is_a_stepped_bore(example):
    comp = example.components["CMP-MOV-BODY"]
    assert comp.flow is not None
    path = example.component_dir(comp.id) / comp.flow.cad.file
    bore = flow.read_bore(path, comp.flow.inlet, comp.flow.outlet, comp.flow.cad.length_unit)
    assert bore.diameter == pytest.approx(0.025)
    assert bore.length == pytest.approx(0.150)
    assert np.allclose(bore.radii, [0.0125, 0.016, 0.0125, 0.016, 0.0125])
    # groove from the seat face (17 mm from the ball centre) to the volume-equivalent plane, both sides
    assert bore.xs[1] == pytest.approx(0.075 - 0.017, abs=1e-6)
    assert bore.xs[4] == pytest.approx(0.075 + 0.017, abs=1e-6)
    assert bore.profile_volume() == pytest.approx(bore.volume, rel=1e-4)
    assert np.allclose(bore.axis, [1, 0, 0])


def test_body_is_not_a_stepped_bore(example):
    comp = example.components["CMP-MOV-BODY"]
    path = example.component_dir(comp.id) / comp.cad.file
    with pytest.raises(flow.FlowDomainError):
        flow.read_bore(path, INLET, OUTLET, "m")


def test_hex_mesh_keeps_the_section_area(tmp_path):
    bore = flow.straight_bore(0.025, 0.05)
    mesh = flow.build_mesh(bore, 5, 0.05, 0.05)
    assert mesh.hexes.shape[1] == 8
    path = flow.write_msh(tmp_path / "m.msh", mesh)
    import meshio
    m = meshio.read(path)
    vol = 0.0
    for c in m.cells:
        if c.type == "hexahedron":
            p = m.points[c.data]
            # volume of a hexahedron from 6 tetrahedra over the diagonal 0-6
            for a, b, cc in ((1, 2, 6), (2, 3, 6), (3, 7, 6), (7, 4, 6), (4, 5, 6), (5, 1, 6)):
                vol += np.abs(np.einsum("ij,ij->i", p[:, a] - p[:, 0],
                                        np.cross(p[:, b] - p[:, 0], p[:, cc] - p[:, 0]))).sum() / 6
    assert vol == pytest.approx(math.pi * 0.0125**2 * 0.15, rel=5e-3)


def test_mesh_uncertainty_rule():
    assert flow.mesh_uncertainty([1.0], [10.0])["uncertainty"] is None
    two = flow.mesh_uncertainty([1.0, 0.5], [10.0, 10.1])
    assert two["uncertainty"] == pytest.approx(3 * 0.1 / 1.0 / 10.1)
    three = flow.mesh_uncertainty([1.0, 1 / 1.5, 1 / 2.25], [10.0, 10.4, 10.578])
    assert three["uncertainty"] is not None and three["uncertainty"] > 0


def test_mesh_study_check_is_information_when_over_the_limit():
    """CONTRACTS.md: convergence is carried by the result's status and uncertainty, never by a failed check."""
    def level(m, dp):
        return flow.Level(m=m, cells=1000 * m, nodes=1100 * m, h_wall=1e-4, x=np.zeros(3), p=np.zeros(3),
                          p_total=np.zeros(3), gradient=100.0, gradient_parts=(100.0, 100.0), dp=dp, dp_taps={},
                          dp_total_faces=dp, mass_in=1.0, mass_balance=0.0, momentum_residual=0.0, y_plus={},
                          iterations=200, converged=True, residuals={"p": 1e-6}, qoi_change=None, warnings=[],
                          solve_s=1.0, quality={})
    setup = flow.Setup(mass_flow=1.0, rho=1000.0, mu=1e-3, regime="laminar", upstream=0.1, downstream=0.1)
    bore = flow.Bore(origin=np.zeros(3), axis=np.array([1.0, 0, 0]), xs=[0.0, 0.1], radii=[0.01],
                     volume=math.pi * 1e-4 * 0.1, source="test.step")

    def mesh_study(u):
        out = flow._checks("AN-FLOW-X", [level(1, 14.0e3), level(2, 14.4e3)], setup, bore, "laminar", 1000.0,
                           u, 0.05, {}, None, SimpleNamespace(features=[], interface="INT-X"), 0.02)
        return next(c for c in out if c.id.endswith("MESH-STUDY"))

    c = mesh_study(0.08)
    assert c.passed is None and c.value == pytest.approx(0.08) and c.limit == 0.05
    assert "8.0 %" in c.statement and "5 %" in c.statement and "information only" in c.statement
    c = mesh_study(0.02)
    assert c.passed is True and "information only" not in c.statement
    assert mesh_study(None).passed is None


def test_benchmark_problem_registered():
    from feacore.problems.registry import get
    p = get("pipe_turbulent_3d")
    assert p.reynolds == pytest.approx(1e5)
    ref = {r.qoi: r for r in p.references()}
    f = flow.colebrook(1e5)
    assert ref["dp_AB"].value == pytest.approx(f * 15 * 0.5 * 1000 * 16)
    assert p.options()["turbulence"] == "kOmegaSST"


# --------------------------------------------------------------------------- not_supported answers


@pytest.fixture(scope="module")
def flow_lc(example, example_ctx):
    return _flow_lc(example, example_ctx)


def test_example_load_case(flow_lc, example, example_ctx):
    kinds = {x.kind for x in flow_lc.loads}
    assert {"mass_flow", "pressure", "fluid_temperature"} <= kinds
    fluid = flow._fluid(example, example_ctx, example.components["CMP-MOV-BODY"])
    mdot = [x for x in flow_lc.loads if x.kind == "mass_flow"][0].si
    re = 4 * mdot / (math.pi * 0.025 * fluid["mu"])
    assert re == pytest.approx(1.384e6, rel=2e-3)
    assert flow.VALIDATED_RE[0] <= re <= flow.VALIDATED_RE[1]


def test_no_openfoam_gives_not_supported(workproject, flow_lc, tmp_path, monkeypatch):
    ctx = context_mod.build(workproject)
    reason = "OpenFOAM is not installed. The flow analysis needs the OpenFOAM ESI version v1912 or newer."
    monkeypatch.setattr(flow, "availability", lambda: (False, reason, ""))
    rec = flow.run(workproject, ctx, None, flow_lc, tmp_path, Options(quick=True)).record
    assert rec.status == "not_supported"
    assert rec.message == reason
    assert not rec.results


def test_no_flow_domain_gives_not_supported(workproject, flow_lc, tmp_path):
    ctx = context_mod.build(workproject)
    workproject.components["CMP-MOV-BODY"].flow = None
    rec = flow.run(workproject, ctx, None, flow_lc, tmp_path, Options(quick=True)).record
    assert rec.status == "not_supported"
    assert "no flow domain" in rec.message


def test_reynolds_outside_validated_range(workproject, flow_lc, tmp_path):
    ctx = context_mod.build(workproject)
    comp = workproject.components["CMP-MOV-BODY"]
    comp.flow.fluid["mu"] = Value(value=1e-2, unit="Pa*s")       # Re about 2.7e4: transitional to low turbulent
    rec = flow.run(workproject, ctx, None, flow_lc, tmp_path, Options(quick=True)).record
    assert rec.status == "not_supported"
    assert "Reynolds number" in rec.message and "validated" in rec.message
    assert not rec.results


def test_body_as_flow_domain_gives_not_supported(workproject, flow_lc, tmp_path):
    ctx = context_mod.build(workproject)
    comp = workproject.components["CMP-MOV-BODY"]
    comp.flow.cad.file = comp.cad.file
    comp.flow.inlet, comp.flow.outlet = INLET, OUTLET
    rec = flow.run(workproject, ctx, None, flow_lc, tmp_path, Options(quick=True)).record
    assert rec.status == "not_supported"
    assert "stepped bore" in rec.message


# --------------------------------------------------------------------------- with OpenFOAM


@needs_foam
def test_laminar_straight_bore_against_hagen_poiseuille(workproject, flow_lc, tmp_path):
    """The plumbing on a case the backend is validated for: laminar flow (Re about 270) in a straight 25 mm
    bore. The pressure drop of the part is the friction of its 150 mm: 128 mu Q L / (pi D^4)."""
    ctx = context_mod.build(workproject)
    comp = workproject.components["CMP-MOV-BODY"]
    step = _cylinder_step(workproject.component_dir(comp.id) / "straight.step")
    comp.flow.cad.file = step.name
    mu = 1.0
    comp.flow.fluid["mu"] = Value(value=mu, unit="Pa*s")
    comp.flow.fluid.pop("roughness", None)
    an = flow.run(workproject, ctx, None, flow_lc, tmp_path / "w", Options(quick=True, log=lambda s: None))
    rec = an.record
    assert rec.status == "done", rec.message
    r = rec.results[flow.RESULT_KEY]
    mdot = [x for x in flow_lc.loads if x.kind == "mass_flow"][0].si
    rho = flow._fluid(workproject, ctx, comp)["rho"]
    q = mdot / rho
    exact = 128 * mu * q * 0.150 / (math.pi * 0.025**4)
    assert len(r.per_level) == 1
    assert r.value == pytest.approx(exact, rel=0.03)
    kinds = {c.kind for c in rec.checks}
    assert all(c.passed is True for c in rec.checks if c.kind == "mass_balance")
    assert "y_plus" not in kinds
    assert all(c.passed in (True, None) or c.limit is not None for c in rec.checks)
    titles = {d.title for d in an.journal.decisions}
    assert "Laminar flow" in titles and "Definition of the pressure drop" in titles


# --------------------------------------------------------------------------- slow: the RANS validation


@needs_foam
@pytest.mark.slow
@pytest.mark.parametrize("re,m", [(1e5, 5), (1e5, 7), (1.384e6, 5), (1.384e6, 7), (1.384e6, 10)])
def test_kOmegaSST_developed_pipe_against_colebrook(re, m, tmp_path):
    """Measured (OpenFOAM v1912): Re 1e5: -4.1 % (m 5), -4.6 % (m 7); Re 1.384e6: -2.6 % (m 5), -2.9 % (m 7),
    -3.2 % (m 10). About 6 s to 2 min each.
    The engine's stated band is VALIDATION["pipe"]["error"]; acceptance 10 %."""
    r = flow.developed_pipe(re, m, tmp_path)
    lo, hi = flow.VALIDATION["pipe"]["error"]
    assert r["converged"] or max(r["residuals"].values()) < 1e-4
    assert lo - 0.005 <= r["error"] <= hi + 0.005, r
    assert r["y_plus"]["in_range"] == pytest.approx(1.0)
    assert abs(r["mass_balance"]) < 1e-6


@needs_foam
@pytest.mark.slow
def test_benchmark_pipe_turbulent_3d(tmp_path):
    from feacore.problems.registry import get
    from feacore.solver import solve
    p = get("pipe_turbulent_3d")
    ref = {r.qoi: r for r in p.references()}
    h = p.study_sizes()[1]
    res = solve(p.case(p.mesh(tmp_path / "m.msh", h)), backend="openfoam")
    try:
        q = res.evaluate_all()
        assert abs(q["dp_AB"] / ref["dp_AB"].value - 1) <= ref["dp_AB"].tolerance
        assert q["dp_AB"] < ref["dp_AB"].value                       # the model reads low on this case
        assert any(w.startswith("info: y+ on wall") for w in res.info.warnings)
        yp = res.y_plus("wall")["y_plus"]
        assert np.median(yp) > 60
    finally:
        res.cleanup()


@needs_foam
@pytest.mark.slow
def test_example_valve_quick(workproject, flow_lc, tmp_path):
    ctx = context_mod.build(workproject)
    an = flow.run(workproject, ctx, None, flow_lc, tmp_path / "w", Options(quick=True, log=lambda s: None))
    rec = an.record
    assert rec.status == "done", rec.message
    r = rec.results[flow.RESULT_KEY]
    assert r.unit == "Pa" and len(r.per_level) == 1
    # the bore is 6 diameters of smooth pipe plus two narrow grooves: a little more than the friction alone
    rho, mu, d = 1141.0, 1.95e-4, 0.025
    v = 5.3 / (rho * math.pi * d * d / 4)
    friction = flow.colebrook(rho * v * d / mu) * 6.0 * 0.5 * rho * v * v
    assert 0.95 * friction < r.value < 1.3 * friction      # measured: 3.55 kPa, friction 3.38 kPa
    assert "first look" in r.note
    kinds = {c.kind for c in rec.checks}
    assert {"mass_balance", "residuals", "y_plus"} <= kinds
    shutil.rmtree(tmp_path / "w", ignore_errors=True)
