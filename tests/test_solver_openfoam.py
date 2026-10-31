"""OpenFOAM backend (physics "flow"): laminar 2D checks against closed forms.

Skipped when OpenFOAM (ESI line) or foamlib is missing. Small gmsh meshes generated into a
temporary directory; every solve takes about a second.
"""

from __future__ import annotations

import numpy as np
import pytest

gmsh = pytest.importorskip("gmsh")
pytest.importorskip("foamlib")

from feacore.case import BC, Analysis, Case, CaseError, Material, QoI  # noqa: E402
from feacore.solver import solve  # noqa: E402
from feacore.solver._foam_env import find_openfoam  # noqa: E402
from feacore.solver._skfem_mesh import MeshError  # noqa: E402

pytestmark = pytest.mark.skipif(find_openfoam() is None, reason="OpenFOAM not installed")

MU, RHO, UM, H, L, T = 0.01, 1.0, 1.0, 0.1, 0.6, 0.01   # Re = rho Um H / mu = 10


def _rect(path, Lx, Ly, nx, ny, names, quad=True):
    """Rectangle [0, Lx] x [0, Ly]; names(x, y) -> group of the edge with that midpoint."""
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
