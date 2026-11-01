"""OpenFOAM backend (physics "flow"): laminar 2D and 3D checks against closed forms.

Skipped when OpenFOAM (ESI line) or foamlib is missing. Small gmsh meshes generated into a
temporary directory; every solve takes about a second (the 3D tet pipe a few seconds).
"""

from __future__ import annotations

import numpy as np
import pytest

gmsh = pytest.importorskip("gmsh")
pytest.importorskip("foamlib")

from feacore.case import BC, Analysis, Case, CaseError, Material, QoI, Region  # noqa: E402
from feacore.solver import solve  # noqa: E402
from feacore.solver._foam_env import find_openfoam  # noqa: E402
from feacore.solver._skfem_mesh import MeshError  # noqa: E402

pytestmark = pytest.mark.skipif(find_openfoam() is None, reason="OpenFOAM not installed")

MU, RHO, UM, H, L, T = 0.01, 1.0, 1.0, 0.1, 0.6, 0.01   # Re = rho Um H / mu = 10


def _rect(path, Lx, Ly, nx, ny, names, quad=True, reverse=False):
    """Rectangle [0, Lx] x [0, Ly]; names(x, y) -> group of the edge with that midpoint.
    ``reverse``: write the cells in reverse order (same mesh, different cell numbering)."""
    gmsh.initialize(readConfigFiles=False, interruptible=False)
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("r")
        s = gmsh.model.occ.addRectangle(0, 0, 0, Lx, Ly)
        gmsh.model.occ.synchronize()
        groups: dict = {}
        for _, c in gmsh.model.getBoundary([(2, s)], oriented=False):
            x, y, _ = gmsh.model.occ.getCenterOfMass(1, c)
            groups.setdefault(names(x, y), []).append(c)
            horizontal = abs(y) < 1e-12 or abs(y - Ly) < 1e-12
            if quad:
                gmsh.model.mesh.setTransfiniteCurve(c, (nx if horizontal else ny) + 1)
        for name, cs in groups.items():
            gmsh.model.addPhysicalGroup(1, cs, name=name)
        gmsh.model.addPhysicalGroup(2, [s], name="fluid")
        if quad:
            gmsh.model.mesh.setTransfiniteSurface(s)
            gmsh.model.mesh.setRecombine(2, s)
        else:
            gmsh.option.setNumber("Mesh.MeshSizeMax", Ly / ny)
        gmsh.model.mesh.generate(2)
        if reverse:
            for et, tags in zip(*gmsh.model.mesh.getElements(2, s)[:2]):
                gmsh.model.mesh.reorderElements(et, s, list(range(len(tags)))[::-1])
        gmsh.write(str(path))
    finally:
        gmsh.finalize()
    return path


def _channel_names(x, y):
    return "inlet" if abs(x) < 1e-12 else "outlet" if abs(x - L) < 1e-12 else "walls"


def _channel_case(mesh, inlet_value=None, **opts) -> Case:
    prof = f"6*{UM}*y*({H}-y)/{H}**2"
    return Case(
        mesh=str(mesh),
        analysis=Analysis(physics="flow", formulation="2d", order=2, thickness=T, options=opts),
        materials={"water": Material(rho=RHO, mu=MU)},
        bcs=[BC("velocity_inlet", group="inlet", value=inlet_value if inlet_value is not None else [prof, 0.0]),
             BC("pressure_outlet", group="outlet", value=0.0),
             BC("wall", group="walls")],
        qois=[QoI("dp", "point_difference", field="p", point=[0.2, 0.5 * H], point2=[0.5, 0.5 * H]),
              QoI("q_in", "flow_rate", group="inlet"), QoI("q_out", "flow_rate", group="outlet"),
              QoI("fx_walls", "force", group="walls", component=0),
              QoI("fy_walls", "force", group="walls", component=1),
              QoI("u_mid", "point", field="ux", point=[0.4, 0.5 * H])],
    )


@pytest.fixture(scope="module")
def poiseuille(tmp_path_factory):
    d = tmp_path_factory.mktemp("foam_channel")
    out = {}
    for n in (8, 16):
        mesh = _rect(d / f"ch{n}.msh", L, H, int(round(n * L / H)), n, _channel_names)
        res = solve(_channel_case(mesh), backend="openfoam")
        out[n] = (res, res.evaluate_all())
    return out


def test_poiseuille_pressure_gradient_second_order(poiseuille):
    """dp/dx = -12 mu U_mean / H^2 with a parabolic inlet; the FV error is O(h^2)."""
    exact = 12.0 * MU * UM / H**2                    # -dp/dx
    err = {n: abs(q["dp"] / 0.3 - exact) / exact for n, (_, q) in poiseuille.items()}
    assert err[16] < 0.01
    assert 3.0 < err[8] / err[16] < 5.0              # second order
    for res, _ in poiseuille.values():
        assert res.info.converged and res.info.iterations > 0
        assert res.info.residuals["p"] <= 1e-8 and res.info.residuals["U"] <= 1e-8


def test_poiseuille_wall_shear_force(poiseuille):
    """Viscous force on both walls = 12 mu U_mean L T / H (per thickness T); no net lift."""
    exact = 12.0 * MU * UM * L * T / H
    res, q = poiseuille[16]
    assert q["fx_walls"] == pytest.approx(exact, rel=0.01)
    assert abs(q["fy_walls"]) < 1e-6 * exact
    assert q["u_mid"] == pytest.approx(1.5 * UM, rel=0.01)


def test_mass_conservation(poiseuille):
    for res, q in poiseuille.values():
        assert q["q_in"] < 0 < q["q_out"]            # positive out of the domain
        assert q["q_in"] + q["q_out"] == pytest.approx(0.0, abs=1e-7 * q["q_out"])
        # the inlet flow rate is U_mean H T up to the midpoint sampling of the parabola
        assert -q["q_in"] == pytest.approx(UM * H * T, rel=0.01)


def test_solve_info_2d(poiseuille):
    res, _ = poiseuille[8]
    i = res.info
    assert (i.backend, i.physics, i.dim, i.cell_type) == ("openfoam", "flow", 2, "quad")
    assert i.n_cells == 8 * 48                       # 2D cells, not the extruded count
    assert i.measure == pytest.approx(L * H)
    assert i.h_rep == pytest.approx((L * H / i.n_cells) ** 0.5)
    xy, p = res.nodal("p")
    assert xy.shape == (i.n_cells, 2) and p.shape == (i.n_cells,)
    assert p.max() > 0 and abs(p.min()) < 0.05 * p.max()      # p in Pa, 0 at the outlet


def test_triangles_and_normal_speed_inlet(tmp_path):
    """Unstructured triangles; uniform inlet given as a scalar inward-normal speed."""
    mesh = _rect(tmp_path / "tri.msh", L, H, 0, 10, _channel_names, quad=False)
    case = _channel_case(mesh, inlet_value=UM)
    case.regions = [Region("fluid", "water")]                   # a named fluid region resolves too
    res = solve(case, backend="openfoam")
    q = res.evaluate_all()
    assert q["q_in"] == pytest.approx(-UM * H * T, rel=1e-9)
    assert q["q_out"] == pytest.approx(UM * H * T, rel=1e-6)
    assert res.info.cell_type == "triangle" and res.info.converged
    assert q["dp"] > 0 and q["fx_walls"] > 0


def test_lid_driven_cavity_vortex(tmp_path):
    """Re = 100 cavity: one clockwise primary vortex; closed domain, p has zero mean."""
    mesh = _rect(tmp_path / "cav.msh", 1.0, 1.0, 20, 20, lambda x, y: "lid" if abs(y - 1) < 1e-12 else "walls")
    case = Case(
        mesh=str(mesh),
        analysis=Analysis(physics="flow", formulation="2d", order=2, thickness=0.1),
        materials={"fluid": Material(rho=1.0, mu=0.01)},
        bcs=[BC("wall", group="lid", value=[1.0, 0.0]), BC("wall", group="walls")],
        qois=[QoI("ux_low", "point", field="ux", point=[0.5, 0.25]),
              QoI("ux_high", "point", field="ux", point=[0.5, 0.9]),
              QoI("uy_left", "point", field="uy", point=[0.2, 0.6]),
              QoI("uy_right", "point", field="uy", point=[0.8, 0.6]),
              QoI("fx_lid", "force", group="lid", component=0),
              QoI("p_mean", "mean", field="p"), QoI("ux_min", "min", field="ux")],
    )
    res = solve(case, backend="openfoam")
    q = res.evaluate_all()
    assert res.info.converged
    assert q["ux_high"] > 0 > q["ux_low"]
    assert q["uy_left"] > 0 > q["uy_right"]
    assert q["fx_lid"] < 0                      # the fluid resists the lid
    assert q["p_mean"] == pytest.approx(0.0, abs=1e-12)
    assert -0.25 < q["ux_min"] < -0.15          # Ghia et al. (1982): min u on the centreline ~ -0.21
    assert any("closed domain" in w for w in res.info.warnings)


def test_not_converged_warns(tmp_path):
    mesh = _rect(tmp_path / "ch.msh", L, H, 24, 4, _channel_names)
    res = solve(_channel_case(mesh, max_iterations=5), backend="openfoam")
    assert not res.info.converged and res.info.iterations == 5
    assert any("NOT CONVERGED" in w for w in res.info.warnings)


def test_case_validation_errors(tmp_path):
    mesh = _rect(tmp_path / "ch.msh", L, H, 12, 2, _channel_names)
    bad = _channel_case(mesh)
    bad.bcs.append(BC("fixed", group="walls"))
    with pytest.raises(CaseError, match="type must be one of"):
        solve(bad, backend="openfoam")
    bad = _channel_case(mesh, inlet_value=[1.0, 0.0, 0.0])          # 3 components in 2D
    with pytest.raises(CaseError, match="velocity_inlet"):
        solve(bad, backend="openfoam")
    bad = _channel_case(mesh)
    bad.materials["water"].mu = 0.0
    with pytest.raises(CaseError, match="mu > 0"):
        solve(bad, backend="openfoam")
    bad = _channel_case(mesh)
    bad.bcs[2].group = "wals"
    with pytest.raises(MeshError, match="wals"):
        solve(bad, backend="openfoam")


def test_symmetry_half_channel_matches_full(tmp_path, poiseuille):
    """Symmetry plane at the centreline: the half channel reproduces the full one."""
    names = lambda x, y: ("inlet" if abs(x) < 1e-12 else "outlet" if abs(x - L) < 1e-12
                          else "walls" if abs(y) < 1e-12 else "centre")
    mesh = _rect(tmp_path / "half.msh", L, 0.5 * H, 48, 4, names)
    case = _channel_case(mesh)
    case.bcs.append(BC("symmetry", group="centre"))
    case.qois = [q for q in case.qois if q.name in ("dp", "q_in")]
    q = solve(case, backend="openfoam").evaluate_all()
    full = poiseuille[8][1]
    assert q["dp"] == pytest.approx(full["dp"], rel=1e-6)
    assert q["q_in"] == pytest.approx(0.5 * full["q_in"], rel=1e-9)


def test_slip_walls_give_plug_flow(tmp_path):
    mesh = _rect(tmp_path / "ch.msh", L, H, 24, 4, _channel_names)
    case = _channel_case(mesh, inlet_value=[UM, 0.0])
    case.bcs[2] = BC("slip", group="walls")
    res = solve(case, backend="openfoam")
    q = res.evaluate_all()
    _, ux = res.nodal("ux")
    assert np.allclose(ux, UM, rtol=1e-6)
    assert abs(q["dp"]) < 1e-6 * RHO * UM**2
    assert abs(q["fx_walls"]) < 1e-9


def test_mesh_harness_cfd_adapter_tee(tmp_path):
    """Harness spec (analysis: cfd, bcs.cfd) -> tet mesh -> flow case -> OpenFOAM, 3D."""
    yaml = pytest.importorskip("yaml")
    from feagent.mesher import run_mesh
    from feagent.parts import make_part
    from feagent.spec import load_spec

    from feacore.adapters.mesh_harness_cfd import case_from_spec

    step = make_part("tee", tmp_path / "tee.step")
    spec = {"geometry": str(step), "analysis": "cfd", "size": {"global": 5.0},
            "groups": {"inlet": {"select": {"at": "xmin"}, "expect": 1},
                       "outlet_main": {"select": {"at": "xmax"}, "expect": 1},
                       "outlet_branch": {"select": {"at": "ymax"}, "expect": 1}},
            "bcs": {"cfd": {"inlet": ["inlet"], "outlet": ["outlet_main", "outlet_branch"], "velocity": 10.0}}}
    p = tmp_path / "mesh.yaml"
    p.write_text(yaml.safe_dump(spec))
    assert run_mesh(load_spec(p)).ok
    case = case_from_spec(p).check()
    assert [(b.type, b.group) for b in case.bcs] == [
        ("velocity_inlet", "inlet"), ("pressure_outlet", "outlet_main"), ("pressure_outlet", "outlet_branch"),
        ("wall", "walls")]
    res = solve(case, backend="openfoam")
    q = res.evaluate_all()
    assert res.info.dim == 3 and res.info.cell_type == "tetra" and res.info.converged
    qin = q["flow_rate_inlet"]
    assert qin < 0 and q["flow_rate_outlet_main"] > 0 and q["flow_rate_outlet_branch"] > 0
    assert qin + q["flow_rate_outlet_main"] + q["flow_rate_outlet_branch"] == pytest.approx(0, abs=1e-7 * -qin)
    assert -qin == pytest.approx(np.pi * 10.0**2 * 10.0, rel=0.05)    # pi R^2 U (faceted inlet)
    assert q["pressure_drop"] > 0 and q["force_walls_x"] > 0


def _tet_pipe(path, R, Lp, h):
    """Unstructured tets in a circular pipe along x (0 <= x <= Lp), as an agent meshes the STEP."""
    gmsh.initialize(readConfigFiles=False, interruptible=False)
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("pipe")
        v = gmsh.model.occ.addCylinder(0, 0, 0, Lp, 0, 0, R)
        gmsh.model.occ.synchronize()
        groups: dict = {}
        for _, s in gmsh.model.getBoundary([(3, v)], oriented=False):
            x = gmsh.model.occ.getCenterOfMass(2, s)[0]
            groups.setdefault("inlet" if x < 1e-9 else "outlet" if x > Lp - 1e-9 else "wall", []).append(s)
        for name, ss in groups.items():
            gmsh.model.addPhysicalGroup(2, ss, name=name)
        gmsh.model.addPhysicalGroup(3, [v], name="fluid")
        gmsh.option.setNumber("Mesh.MeshSizeMax", h)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.model.mesh.generate(3)
        gmsh.write(str(path))
    finally:
        gmsh.finalize()
    return path


def test_pipe_3d_tets_pressure_drop(tmp_path):
    """Hagen-Poiseuille flow on unstructured tets (3.6k cells): the axial pressure drop and the wall
    force match the exact values. With OpenFOAM's plain schemes (Gauss linear gradients, linear
    momentum-interpolation flux) tets converge to a pressure gradient 15-18 % high, and the wall
    force taken with 1/|Cf - C| instead of the wall-normal distance was 10 % off on its own.

    The exact values are those of the meshed (faceted) section of area A: -dp/dx = 8 pi mu Q / A^2
    and F = 8 pi mu Q L / A, with Q the solved flow rate; the 21-gon is 1.5 % short in area."""
    R, Lp, mu, Ub = 0.005, 0.03, 0.01, 0.01                 # Re = rho Ub 2R / mu = 10
    xa, xb = 0.4 * Lp, 0.8 * Lp
    case = Case(
        mesh=str(_tet_pipe(tmp_path / "pipe.msh", R, Lp, 0.0015)),
        analysis=Analysis(physics="flow", formulation="3d", order=2, options={"max_iterations": 3000}),
        materials={"oil": Material(rho=1000.0, mu=mu)},
        bcs=[BC("velocity_inlet", group="inlet", value=[f"{2 * Ub}*(1 - (y**2 + z**2)/{R**2})", 0.0, 0.0]),
             BC("pressure_outlet", group="outlet", value=0.0), BC("wall", group="wall")],
        qois=[QoI("dp", "point_difference", field="p", point=[xa, 0, 0], point2=[xb, 0, 0]),
              QoI("fx_wall", "force", group="wall", component=0),
              QoI("p_int", "integral", field="p", group="inlet"), QoI("p_mean", "mean", field="p", group="inlet"),
              QoI("q_out", "flow_rate", group="outlet")],
    )
    res = solve(case, backend="openfoam")
    q = res.evaluate_all()
    assert res.info.cell_type == "tetra" and res.info.converged
    area = q["p_int"] / q["p_mean"]                              # faceted inlet section
    assert 0.97 < area / (np.pi * R**2) < 1.0
    assert q["dp"] == pytest.approx(8 * np.pi * mu * q["q_out"] / area**2 * (xb - xa), rel=0.03)
    assert q["fx_wall"] == pytest.approx(8 * np.pi * mu * q["q_out"] * Lp / area, rel=0.03)
    assert q["fx_wall"] == pytest.approx(q["p_int"], rel=0.03)     # inlet pressure force = wall force
    assert q["dp"] == pytest.approx(8 * mu * Ub * (xb - xa) / R**2, rel=0.06)   # exact, circular section
    assert any("skew-consistent schemes" in w for w in res.info.warnings)


def test_skew_correction_option(tmp_path):
    mesh = _rect(tmp_path / "ch.msh", L, H, 12, 2, _channel_names)
    with pytest.raises(CaseError, match="skew_correction"):
        solve(_channel_case(mesh, skew_correction="yes"), backend="openfoam")
    res = solve(_channel_case(mesh, max_iterations=3), backend="openfoam")        # 2D: plain schemes
    assert not any("skew-consistent" in w for w in res.info.warnings)
    res = solve(_channel_case(mesh, max_iterations=3, skew_correction=True), backend="openfoam")
    assert any("skew-consistent" in w for w in res.info.warnings)


@pytest.mark.slow
def test_komega_sst_channel_runs(tmp_path):
    """RANS wiring (k, omega, nut, wall functions): runs, conserves mass, drags the walls."""
    mesh = _rect(tmp_path / "ch.msh", 1.0, 0.1, 50, 20, lambda x, y: "inlet" if abs(x) < 1e-12 else
                 "outlet" if abs(x - 1.0) < 1e-12 else "walls")
    case = Case(mesh=str(mesh),
                analysis=Analysis(physics="flow", formulation="2d", order=2, thickness=1.0,
                                  options={"turbulence": "kOmegaSST", "max_iterations": 400}),
                materials={"air": Material(rho=1.2, mu=1.8e-5)},
                bcs=[BC("velocity_inlet", group="inlet", value=[10.0, 0.0]),
                     BC("pressure_outlet", group="outlet", value=0.0), BC("wall", group="walls")],
                qois=[QoI("q_in", "flow_rate", group="inlet"), QoI("q_out", "flow_rate", group="outlet"),
                      QoI("fx", "force", group="walls", component=0)])
    res = solve(case, backend="openfoam")
    q = res.evaluate_all()
    assert q["q_in"] + q["q_out"] == pytest.approx(0, abs=1e-6)
    assert q["fx"] > 0 and np.isfinite(q["fx"])
    assert {"k", "omega", "p", "U"} <= set(res.info.residuals)


def test_probe_continuous_and_cell_order_independent(tmp_path):
    """Probes on shared vertices / faces (the Ghia points of a uniform grid sit there): the same value
    for any cell numbering, and continuous across faces (cellPoint-style interpolation)."""
    def run(reverse):
        mesh = _rect(tmp_path / f"cav_{reverse}.msh", 1.0, 1.0, 16, 16,
                     lambda x, y: "lid" if abs(y - 1) < 1e-12 else "walls", reverse=reverse)
        case = Case(mesh=str(mesh), analysis=Analysis(physics="flow", formulation="2d", order=2, thickness=0.1),
                    materials={"fluid": Material(rho=1.0, mu=0.01)},
                    bcs=[BC("wall", group="lid", value=[1.0, 0.0]), BC("wall", group="walls")])
        return solve(case, backend="openfoam")

    a, b = run(False), run(True)
    assert not np.allclose(a.nodal("ux")[0], b.nodal("ux")[0])          # really a different numbering
    h = 1.0 / 16
    pts = [[0.5, 0.5], [0.5 + 0.5 * h, 0.5], [0.5, 0.25 + 0.5 * h], [0.3, 0.7], [0.5, 1.0], [0.0, 0.5]]
    for fld in ("ux", "uy", "p"):
        va, vb = a.probe(fld, pts), b.probe(fld, pts)
        assert np.allclose(va, vb, rtol=1e-6, atol=1e-8), (fld, va, vb)
    assert a.probe("ux", [[0.5, 1.0]])[0] == pytest.approx(1.0)              # lid value on the lid
    assert a.probe("uy", [[0.0, 0.5]])[0] == pytest.approx(0.0, abs=1e-12)   # no-slip wall
    for x0, y0 in ((0.5, 0.3), (0.5 + 0.5 * h, 0.5)):                       # across a vertical / horizontal face
        for d in ([1e-9, 0.0], [0.0, 1e-9]):
            pm = a.probe("ux", [[x0 - d[0], y0 - d[1]], [x0 + d[0], y0 + d[1]]])
            assert abs(pm[0] - pm[1]) < 1e-7
