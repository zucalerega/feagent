"""Tests of the CalculiX backend (feacore.solver.ccx_backend) against closed-form solutions,
plus a cross-check against the scikit-fem backend. Small meshes; each ccx run takes < 1 s."""

from __future__ import annotations

import math
import shutil

import numpy as np
import pytest

from feacore.case import BC, Analysis, Case, CaseError, Load, Material, QoI
from feacore.solver import solve

pytestmark = pytest.mark.skipif(shutil.which("ccx") is None, reason="CalculiX (ccx) not installed")

E, NU = 1000.0, 0.3


# ----------------------------------------------------------------------------- meshes
def _session(name):
    from feacore.geometry.gmsh_tools import gmsh_session
    return gmsh_session(name)


def _rect(path, L, H, h, order=2, x0=0.0, msh=4.1):
    import gmsh
    from feacore.geometry.gmsh_tools import add_group, in_box, set_uniform_size
    with _session("rect"):
        s = gmsh.model.occ.addRectangle(x0, 0, 0, L, H)
        gmsh.model.occ.synchronize()
        add_group(2, [s], "body")
        add_group(1, in_box(1, xmax=x0), "left")
        add_group(1, in_box(1, xmin=x0 + L), "right")
        add_group(1, in_box(1, ymax=0), "bottom")
        add_group(1, in_box(1, ymin=H), "top")
        add_group(0, in_box(0, xmax=x0, ymax=0), "origin")
        set_uniform_size(h)
        gmsh.option.setNumber("Mesh.MshFileVersion", msh)
        gmsh.model.mesh.generate(2)
        if order == 2:
            gmsh.model.mesh.setOrder(2)
        gmsh.write(str(path))
    return str(path)


def _quarter_annulus(path, a, b, h, order=2):
    import gmsh
    from feacore.geometry.gmsh_tools import add_group, set_uniform_size
    with _session("ann"):
        d1 = gmsh.model.occ.addDisk(0, 0, 0, b, b)
        d2 = gmsh.model.occ.addDisk(0, 0, 0, a, a)
        r = gmsh.model.occ.addRectangle(0, 0, 0, 2 * b, 2 * b)
        ring, _ = gmsh.model.occ.cut([(2, d1)], [(2, d2)])
        q, _ = gmsh.model.occ.intersect(ring, [(2, r)])
        gmsh.model.occ.synchronize()
        add_group(2, [t for _, t in q], "body")
        g = {"inner": [], "outer": [], "sym_y": [], "sym_x": []}
        for _, c in gmsh.model.getEntities(1):
            x, y, _ = gmsh.model.occ.getCenterOfMass(1, c)
            key = ("sym_y" if abs(y) < 1e-9 else "sym_x" if abs(x) < 1e-9 else
                   "inner" if math.hypot(x, y) < 0.5 * (a + b) else "outer")
            g[key].append(c)
        for k, v in g.items():
            add_group(1, v, k)
        set_uniform_size(h)
        gmsh.model.mesh.generate(2)
        if order == 2:
            gmsh.model.mesh.setOrder(2)
        gmsh.write(str(path))
    return str(path)


def _two_blocks(path, h=0.25):
    import gmsh
    from feacore.geometry.gmsh_tools import add_group, in_box, set_uniform_size
    with _session("two"):
        r1 = gmsh.model.occ.addRectangle(0, 0, 0, 1.0, 0.5)
        r2 = gmsh.model.occ.addRectangle(1.0, 0, 0, 1.0, 0.5)
        gmsh.model.occ.fragment([(2, r1)], [(2, r2)])
        gmsh.model.occ.synchronize()
        from feacore.geometry.gmsh_tools import where
        add_group(2, where(2, lambda x, y, z: x < 1.0), "a")
        add_group(2, where(2, lambda x, y, z: x > 1.0), "b")
        add_group(1, in_box(1, xmax=0), "left")
        add_group(1, in_box(1, xmin=2.0), "right")
        set_uniform_size(h)
        gmsh.model.mesh.generate(2)
        gmsh.model.mesh.setOrder(2)
        gmsh.write(str(path))
    return str(path)


def _box(path, L, W, H, h, order=2):
    import gmsh
    from feacore.geometry.gmsh_tools import add_group, in_box, set_uniform_size
    with _session("box"):
        v = gmsh.model.occ.addBox(0, 0, 0, L, W, H)
        gmsh.model.occ.synchronize()
        add_group(3, [v], "body")
        add_group(2, in_box(2, xmax=0), "left")
        add_group(2, in_box(2, xmin=L), "right")
        add_group(2, in_box(2, ymax=0), "y0")
        add_group(2, in_box(2, zmax=0), "z0")
        add_group(1, in_box(1, xmax=0, ymax=0), "edge_x0y0")
        add_group(0, in_box(0, xmax=0, ymax=0, zmax=0), "origin")
        set_uniform_size(h)
        gmsh.model.mesh.generate(3)
        if order == 2:
            gmsh.model.mesh.setOrder(2)
        gmsh.write(str(path))
    return str(path)


def _quarter_cylinder(path, R, L, h):
    import gmsh
    from feacore.geometry.gmsh_tools import add_group, set_uniform_size
    with _session("qcyl"):
        cyl = gmsh.model.occ.addCylinder(0, 0, 0, 0, 0, L, R)
        box = gmsh.model.occ.addBox(0, 0, 0, 2 * R, 2 * R, 2 * L)
        q, _ = gmsh.model.occ.intersect([(3, cyl)], [(3, box)])
        gmsh.model.occ.synchronize()
        add_group(3, [t for _, t in q], "body")
        curved = []
        for _, s in gmsh.model.getEntities(2):
            x, y, z = gmsh.model.occ.getCenterOfMass(2, s)
            if abs(x) < 1e-9:
                add_group(2, [s], "x0")
            elif abs(y) < 1e-9:
                add_group(2, [s], "y0")
            elif abs(z) < 1e-9:
                add_group(2, [s], "z0")
            elif abs(z - L) > 1e-9:
                curved.append(s)
        add_group(2, curved, "curved")
        set_uniform_size(h)
        gmsh.model.mesh.generate(3)
        gmsh.model.mesh.setOrder(2)
        gmsh.write(str(path))
    return str(path)


@pytest.fixture(scope="module")
def meshes(tmp_path_factory):
    d = tmp_path_factory.mktemp("ccx_meshes")
    return {
        "rect2": _rect(d / "rect2.msh", 2.0, 0.5, 0.25, order=2),
        "rect1": _rect(d / "rect1.msh", 2.0, 0.5, 0.25, order=1),
        "rect2_v22": _rect(d / "rect2_v22.msh", 2.0, 0.5, 0.25, order=2, msh=2.2),
        "cyl_axi": _rect(d / "cyl_axi.msh", 0.5, 2.0, 0.2, order=2),          # r in [0, R], z in [0, L]
        "ring_axi": _rect(d / "ring_axi.msh", 1.0, 0.5, 0.1, order=2, x0=1.0),  # r in [a, b]
        "annulus": _quarter_annulus(d / "annulus.msh", 1.0, 2.0, 0.1),
        "box2": _box(d / "box2.msh", 2.0, 0.5, 0.5, 0.25, order=2),
        "box1": _box(d / "box1.msh", 2.0, 0.5, 0.5, 0.25, order=1),
        "qcyl": _quarter_cylinder(d / "qcyl.msh", 1.0, 1.0, 0.3),
        "two_blocks": _two_blocks(d / "two_blocks.msh"),
    }


def _case(mesh, physics="elasticity", form="3d", order=2, bcs=(), loads=(), qois=(), mats=None, **an):
    return Case(mesh=mesh, analysis=Analysis(physics, form, order=order, **an),
                materials=mats or {"m": Material(E=E, nu=NU)}, bcs=list(bcs), loads=list(loads),
                qois=list(qois))


def rel(a, b):
    return abs(a - b) / abs(b)


# ----------------------------------------------------------------------------- mesh handling
def test_node_ordering_and_conversion(meshes):
    from feacore.solver._ccx_mesh import build_model, read_gmsh

    for key, dim in (("box2", 3), ("rect2", 2), ("rect2_v22", 2)):
        gm = read_gmsh(meshes[key])
        m = build_model(gm, dim, 2)
        # straight-sided quadratic mesh: midside nodes exactly at edge midpoints (CalculiX order)
        assert m.check_midside() < 1e-12
        assert np.all(np.sign(np.linalg.det(np.stack(
            [m.X[m.conn[:, i], :dim] - m.X[m.conn[:, 0], :dim] for i in range(1, dim + 1)], axis=1))) > 0)
    # order 2 on a linear mesh: midside nodes inserted, Dirichlet sets include them
    m = build_model(read_gmsh(meshes["box1"]), 3, 2)
    assert m.cell_type == "tetra10" and m.check_midside() < 1e-12
    left = m.nodes_of("left")
    assert np.allclose(m.X[left, 0], 0.0) and len(left) > len(m.mesh.groups["left"][0][1])
    edge = m.nodes_of("edge_x0y0")                      # 3D curve group: corners + midside nodes
    assert np.allclose(m.X[edge, :2], 0.0) and len(edge) == 2 * (len(edge) // 2) + 1
    # order 1 on a quadratic mesh: corner nodes only
    m1 = build_model(read_gmsh(meshes["box2"]), 3, 1)
    assert m1.cell_type == "tetra" and m1.conn.shape[1] == 4


# ----------------------------------------------------------------------------- elasticity
@pytest.mark.parametrize("order", [1, 2])
def test_uniaxial_bar_tet(meshes, order):
    L, A, P = 2.0, 0.25, 2.5
    c = _case(meshes["box2"], order=order,
              bcs=[BC("symmetry", group="left"), BC("symmetry", group="y0"), BC("symmetry", group="z0"),
                   BC("force", group="right", value=[P, 0.0, 0.0])],
              qois=[QoI("ux", "point", field="ux", point=[L, 0.25, 0.25]),
                    QoI("sxx", "point", field="sxx", point=[1.0, 0.2, 0.3]),
                    QoI("R", "reaction", group="left", component=0), QoI("W", "strain_energy")])
    r = solve(c, backend="ccx")
    v = r.evaluate_all()
    assert r.info.cell_type == ("tetra10" if order == 2 else "tetra") and r.info.backend == "ccx"
    assert rel(v["ux"], P * L / (A * E)) < 1e-6
    assert rel(v["R"], -P) < 1e-6
    assert rel(v["sxx"], P / A) < 2e-5           # ccx nodal extrapolation carries ~1e-5
    assert rel(v["W"], 0.5 * P * P * L / (A * E)) < 1e-6


@pytest.mark.parametrize("form", ["plane_stress", "plane_strain"])
@pytest.mark.parametrize("load", ["pressure", "traction", "force"])
@pytest.mark.parametrize("mesh,order", [("rect2", 2), ("rect2_v22", 1), ("rect1", 2)])
def test_plane_bar(meshes, form, load, mesh, order):
    t, sig, L, H = 0.1, 10.0, 2.0, 0.5
    bc = {"pressure": BC("pressure", group="right", value=-sig),
          "traction": BC("traction", group="right", value=[sig, 0.0]),
          "force": BC("force", group="right", value=[sig * H * t, 0.0])}[load]
    c = _case(meshes[mesh], form=form, order=order, thickness=t,
              bcs=[BC("displacement", group="left", components=[0], value=0.0), BC("fixed", point=[0, 0]), bc],
              qois=[QoI("ux", "point", field="ux", point=[L, H / 2]), QoI("R", "reaction", group="left", component=0),
                    QoI("W", "strain_energy"), QoI("mean", "mean", field="sxx", group="body")])
    v = solve(c, backend="ccx").evaluate_all()
    f = (1 - NU ** 2) if form == "plane_strain" else 1.0
    assert rel(v["ux"], sig * L / E * f) < 1e-6
    assert rel(v["R"], -sig * H * t) < 1e-6
    assert rel(v["W"], 0.5 * sig ** 2 / E * f * L * H * t) < 1e-6
    assert rel(v["mean"], sig) < 2e-5


def _lame(a=1.0, b=2.0, p=10.0):
    A = p * a * a / (b * b - a * a)
    B = p * a * a * b * b / (b * b - a * a)
    return A, B, (lambda r: (1 + NU) / E * ((1 - 2 * NU) * A * r + B / r))


def test_lame_plane_strain_quarter(meshes):
    a, b, p = 1.0, 2.0, 10.0
    A, B, ur = _lame(a, b, p)
    c = _case(meshes["annulus"], form="plane_strain", thickness=1.0,
              bcs=[BC("symmetry", group="sym_y"), BC("symmetry", group="sym_x"),
                   BC("pressure", group="inner", value=p)],
              qois=[QoI("ua", "point", field="ux", point=[a, 0]), QoI("ub", "point", field="uy", point=[0, b]),
                    QoI("stt_a", "point", field="syy", point=[a, 0]), QoI("szz", "point", field="szz", point=[1.3, 0.4]),
                    QoI("Ry", "reaction", group="sym_y", component=1), QoI("W", "strain_energy")])
    v = solve(c, backend="ccx").evaluate_all()
    assert rel(v["ua"], ur(a)) < 1e-4 and rel(v["ub"], ur(b)) < 1e-4
    assert rel(v["stt_a"], A + B / a ** 2) < 2e-3
    assert rel(v["szz"], 2 * NU * A) < 1e-2        # P2 stress discretization error at h = 0.1
    assert rel(v["Ry"], -p * a) < 1e-6           # pressure resultant through the symmetry cut
    assert rel(v["W"], 0.5 * p * ur(a) * math.pi / 2 * a) < 1e-4


def test_refused_configurations(meshes):
    """CAX3 and C3D10 / CAX6 cells with curved interior edges raise NotImplementedError."""
    c = _case(meshes["ring_axi"], form="axisymmetric", order=1,
              bcs=[BC("symmetry", group="bottom"), BC("pressure", group="left", value=1.0)])
    with pytest.raises(NotImplementedError, match="CAX3"):
        solve(c, backend="ccx")
    import meshio
    m = meshio.read(meshes["box2"])
    for blk in m.cells:
        if blk.type == "tetra10":
            m.points[blk.data[:, 9]] += [0.004, -0.003, 0.002]   # curve interior and boundary edges
    bent = meshes["box2"].replace(".msh", "_bent.msh")
    meshio.write(bent, m, file_format="gmsh22", binary=False)
    c = _case(bent, bcs=[BC("fixed", group="left"), BC("force", group="right", value=[1.0, 0, 0])])
    with pytest.raises(NotImplementedError, match="curved interior"):
        solve(c, backend="ccx")


@pytest.mark.parametrize("order,tol_u", [(2, 3e-5)])
def test_lame_axisymmetric(meshes, order, tol_u):
    a, b, p, Hh = 1.0, 2.0, 10.0, 0.5
    A, B, ur = _lame(a, b, p)
    c = _case(meshes["ring_axi"], form="axisymmetric", order=order,
              bcs=[BC("symmetry", group="bottom"), BC("symmetry", group="top"),
                   BC("pressure", group="left", value=p)],
              qois=[QoI("ua", "point", field="ux", point=[a, 0.25]), QoI("ub", "point", field="ux", point=[b, 0.25]),
                    QoI("stt_a", "point", field="stt", point=[a, 0.25]), QoI("szz", "point", field="syy", point=[1.2, 0.3]),
                    QoI("Rz", "reaction", group="bottom", component=1), QoI("W", "strain_energy")])
    v = solve(c, backend="ccx").evaluate_all()
    assert rel(v["ua"], ur(a)) < tol_u and rel(v["ub"], ur(b)) < tol_u
    assert rel(v["stt_a"], A + B / a ** 2) < (2e-3 if order == 2 else 6e-2)
    assert rel(v["szz"], 2 * NU * A) < (2e-3 if order == 2 else 2e-2)
    assert rel(v["Rz"], -2 * NU * A * math.pi * (b * b - a * a)) < 1e-5     # full 360 degrees
    assert rel(v["W"], 0.5 * p * ur(a) * 2 * math.pi * a * Hh) < (1e-5 if order == 2 else 2e-3)


@pytest.mark.parametrize("load", ["pressure", "force", "traction"])
def test_axisymmetric_units(meshes, load):
    """Solid cylinder in uniaxial tension: CAX loads / reactions / energy are full-360 totals."""
    R, L, p = 0.5, 2.0, 10.0
    bc = {"pressure": BC("pressure", group="top", value=-p),
          "force": BC("force", group="top", value=[0.0, p * math.pi * R * R]),
          "traction": BC("traction", group="top", value=[0.0, p])}[load]
    c = _case(meshes["cyl_axi"], form="axisymmetric",
              bcs=[BC("displacement", group="bottom", components=[1], value=0.0), bc],
              qois=[QoI("uz", "point", field="uy", point=[0.3, L]), QoI("ur", "point", field="ux", point=[R, 1.0]),
                    QoI("R", "reaction", group="bottom", component=1), QoI("W", "strain_energy")])
    v = solve(c, backend="ccx").evaluate_all()
    assert rel(v["uz"], p * L / E) < 1e-6
    assert rel(v["ur"], -NU * p * R / E) < 1e-5
    assert rel(v["R"], -p * math.pi * R * R) < 1e-6
    assert rel(v["W"], 0.5 * p * p / E * math.pi * R * R * L) < 1e-6


@pytest.mark.parametrize("order,tol", [(1, 1e-6), (2, 5e-5)])
def test_pressure_total_load_curved_face(meshes, order, tol):
    """Quarter cylinder, pressure on the curved face: the symmetry reactions equal p R L, and the
    consistent nodal loads integrate to the exact resultant (checks *DLOAD face numbering)."""
    p, R, L = 10.0, 1.0, 1.0
    c = _case(meshes["qcyl"], order=order,
              bcs=[BC("symmetry", group="x0"), BC("symmetry", group="y0"), BC("symmetry", group="z0"),
                   BC("pressure", group="curved", value=p)],
              qois=[QoI("Rx", "reaction", group="x0", component=0), QoI("Ry", "reaction", group="y0", component=1),
                    QoI("Rz", "reaction", group="z0", component=2)])
    r = solve(c, backend="ccx")
    v = r.evaluate_all()
    assert rel(v["Rx"], p * R * L) < tol and rel(v["Ry"], p * R * L) < tol
    assert abs(v["Rz"]) < tol * p * R * L
    fy = r.deck.fext[r.model.nodes_of("curved"), 1].sum()
    assert rel(fy, -p * R * L) < 1e-12


def test_pressure_expression_matches_constant(meshes):
    base = [BC("symmetry", group="sym_y"), BC("symmetry", group="sym_x")]
    q = [QoI("ua", "point", field="ux", point=[1.0, 0.0])]
    v1 = solve(_case(meshes["annulus"], form="plane_strain", bcs=base + [BC("pressure", group="inner", value=10.0)],
                     qois=q), backend="ccx").evaluate_all()
    v2 = solve(_case(meshes["annulus"], form="plane_strain", qois=q,
                     bcs=base + [BC("pressure", group="inner", value="10 + 0*x")]), backend="ccx").evaluate_all()
    assert rel(v2["ua"], v1["ua"]) < 1e-5


def test_displacement_expression_and_lower_dim_groups(meshes):
    """Prescribed ux = 1e-3 * x on the right face of a box held by a point and a curve: rigid body
    modes removed with lower-dimensional groups; uniaxial strain field with free lateral faces."""
    eps = 1e-3
    c = _case(meshes["box1"], order=2,
              bcs=[BC("symmetry", group="left"), BC("fixed", group="origin"),
                   BC("displacement", group="edge_x0y0", components=[0, 1], value=[0.0, 0.0]),
                   BC("symmetry", group="z0"),
                   BC("displacement", group="right", components=[0], value=f"{eps} * x")],
              qois=[QoI("sxx", "max", field="sxx"), QoI("R", "reaction", group="right", component=0),
                    QoI("uy", "point", field="uy", point=[2.0, 0.5, 0.25])])
    v = solve(c, backend="ccx").evaluate_all()
    assert rel(v["sxx"], E * eps) < 2e-5
    assert rel(v["R"], E * eps * 0.25) < 1e-6
    assert rel(v["uy"], -NU * eps * 0.5) < 1e-5


def _where_case(meshes, order, where="abs(z) < 1e-9"):
    P = 2.5
    bcs = [BC("symmetry", group="left"), BC("symmetry", group="y0"),
           BC("displacement", group="y0", components=[2], value=0.0, where=where),
           BC("force", group="right", value=[P, 0.0, 0.0])]
    qois = [QoI("ux", "point", field="ux", point=[2.0, 0.3, 0.4]), QoI("uz", "point", field="uz", point=[1.0, 0.2, 0.5]),
            QoI("sxx", "point", field="sxx", point=[1.0, 0.25, 0.25]), QoI("Rz", "reaction", group="y0", component=2),
            QoI("W", "strain_energy")]
    return _case(meshes["box2"], order=order, bcs=bcs, qois=qois)


@pytest.mark.parametrize("order", [1, 2])
def test_dirichlet_where_filter(meshes, order):
    """uz = 0 on face y0 filtered by ``where`` to the line y = z = 0: with ux = 0 on x = 0 and
    uy = 0 on y = 0 the bar is held without restraining its Poisson contraction, so uniaxial
    tension is exact (on the whole face uz = 0 would restrain it)."""
    P, L, A = 2.5, 2.0, 0.25
    r = solve(_where_case(meshes, order), backend="ccx")
    v = r.evaluate_all()
    eps = P / (A * E)
    line = [n for (n, d) in r.deck.dirichlet if d == 2]
    assert np.allclose(r.model.X[line, 1:], 0.0) and len(line) == (17 if order == 2 else 9)
    assert rel(v["ux"], eps * L) < 1e-6 and rel(v["uz"], -NU * eps * 0.5) < 1e-5
    assert rel(v["sxx"], P / A) < 2e-5 and abs(v["Rz"]) < 1e-6 * P
    assert rel(v["W"], 0.5 * P * P * L / (A * E)) < 1e-6
    with pytest.raises(CaseError, match="selects no node"):            # selecting no node is an error
        solve(_where_case(meshes, order, where="z > 100"), backend="ccx")


@pytest.mark.parametrize("order", [1, 2])
def test_dirichlet_where_cross_check_skfem(meshes, order):
    """Same ``where`` case through both backends (skipped until skfem applies BC.where)."""
    c = _where_case(meshes, order)
    v = solve(c, backend="ccx").evaluate_all()
    try:
        vs = solve(c, backend="skfem").evaluate_all()
    except NotImplementedError as e:  # pragma: no cover
        pytest.skip(f"skfem: {e}")
    for k in ("ux", "uz", "sxx", "W"):
        print(f"[ccx vs skfem where] P{order} {k}: ccx={v[k]:+.7e} skfem={vs[k]:+.7e} rel={abs(v[k] / vs[k] - 1):.1e}")
    if abs(vs["uz"] / v["uz"] - 1) > 1e-2:
        pytest.skip("skfem does not apply BC.where yet (it constrains the whole group)")  # pragma: no cover
    for k in ("ux", "uz", "W"):
        assert rel(v[k], vs[k]) < 1e-5, (k, v[k], vs[k])
    assert rel(v["sxx"], vs["sxx"]) < 1e-4


@pytest.mark.parametrize("order", [1, 2])
def test_free_thermal_expansion(meshes, order):
    al, Tref, dT = 1e-5, 20.0, 100.0
    mats = {"m": Material(E=E, nu=NU, alpha=al)}
    c = _case(meshes["box2"], order=order, mats=mats, T_ref=Tref,
              bcs=[BC("symmetry", group="left"), BC("symmetry", group="y0"), BC("symmetry", group="z0")],
              loads=[Load("temperature", Tref + dT)],
              qois=[QoI("ux", "point", field="ux", point=[2.0, 0.25, 0.25]), QoI("vm", "max", field="von_mises"),
                    QoI("W", "strain_energy")])
    v = solve(c, backend="ccx").evaluate_all()
    assert rel(v["ux"], al * dT * 2.0) < 1e-6
    assert v["vm"] < 1e-9 * E * al * dT * 1e6        # zero stress (round-off only)
    for form in ("plane_stress", "plane_strain") + (("axisymmetric",) if order == 2 else ()):
        bcs = [BC("symmetry", group="bottom")] + ([] if form == "axisymmetric" else [BC("symmetry", group="left")])
        c = _case(meshes["rect2"], form=form, order=order, mats=mats, T_ref=Tref, thickness=0.1, bcs=bcs,
                  loads=[Load("temperature", Tref + dT)],
                  qois=[QoI("ux", "point", field="ux", point=[2.0, 0.25]), QoI("sxx", "max_abs", field="sxx"),
                        QoI("szz", "point", field="szz", point=[1.0, 0.25])])
        v = solve(c, backend="ccx").evaluate_all()
        f = (1 + NU) if form == "plane_strain" else 1.0
        tol = 1e-5 if form == "axisymmetric" else 1e-6     # frd output has 6 digits
        assert rel(v["ux"], al * dT * 2.0 * f) < tol
        assert v["sxx"] < 1e-7      # round-off (thin internal plane-stress slab: ~1e-8)
        assert (rel(v["szz"], -E * al * dT) < 2e-5) if form == "plane_strain" else abs(v["szz"]) < 1e-7


def test_body_force_and_multi_material(meshes):
    b = 3.0
    for val in ([b, 0.0, 0.0], ["3.0", "0", "0"]):
        c = _case(meshes["box2"], loads=[Load("body_force", val)],
                  bcs=[BC("symmetry", group="left"), BC("symmetry", group="y0"), BC("symmetry", group="z0")],
                  qois=[QoI("R", "reaction", group="left", component=0)])
        assert rel(solve(c, backend="ccx").evaluate(c.qois[0]), -b * 2.0 * 0.25) < 1e-6
    # two materials in series (regions: '*' fallback after a named group): u = sigma (L_a/E_a + L_b/E_b)
    from feacore.case import Region
    c = _case(meshes["two_blocks"], form="plane_stress", thickness=1.0,
              mats={"soft": Material(E=E, nu=0.0), "stiff": Material(E=4 * E, nu=0.0)},
              bcs=[BC("displacement", group="left", components=[0], value=0.0), BC("fixed", point=[0, 0]),
                   BC("traction", group="right", value=[10.0, 0.0])],
              qois=[QoI("ux", "point", field="ux", point=[2.0, 0.25]), QoI("smax", "max", field="sxx")])
    c.regions = [Region("a", "soft"), Region("*", "stiff")]
    v = solve(c, backend="ccx").evaluate_all()
    assert rel(v["ux"], 10.0 * (1.0 / E + 1.0 / (4 * E))) < 1e-6 and rel(v["smax"], 10.0) < 2e-5


# ----------------------------------------------------------------------------- heat
@pytest.mark.parametrize("order", [1, 2])
def test_heat_1d_slab(meshes, order):
    k, L = 2.0, 2.0
    mats = {"m": Material(k=k)}
    pt, mid = [L, 0.25, 0.25], [1.0, 0.2, 0.3]
    run = lambda bcs, loads=(), qois=(): solve(_case(meshes["box2"], "heat", "3d", order, bcs, loads, qois, mats),
                                               backend="ccx").evaluate_all()
    v = run([BC("temperature", group="left", value=0.0), BC("temperature", group="right", value=100.0)],
            qois=[QoI("T", "point", field="T", point=mid), QoI("qx", "point", field="qx", point=mid),
                  QoI("Q", "reaction", group="left", component=0)])
    assert rel(v["T"], 50.0) < 1e-5 and rel(v["qx"], -k * 100 / L) < 2e-5
    assert rel(v["Q"], -k * 100 / L * 0.25) < 1e-5            # heat leaves through the cold face
    v = run([BC("temperature", group="left", value=0.0), BC("flux", group="right", value=30.0)],
            qois=[QoI("T", "point", field="T", point=pt)])
    assert rel(v["T"], 30 * L / k) < 1e-6
    v = run([BC("temperature", group="left", value=0.0), BC("convection", group="right", h=5.0, T_inf=80.0)],
            qois=[QoI("T", "point", field="T", point=pt)])
    assert rel(v["T"], 5.0 * 80.0 / (5.0 + k / L)) < 1e-5
    v = run([BC("temperature", group="left", value=0.0), BC("temperature", group="right", value=0.0)],
            loads=[Load("source", 7.0)], qois=[QoI("T", "point", field="T", point=mid)])
    assert rel(v["T"], 7.0 * L * L / (8 * k)) < (1e-5 if order == 2 else 2e-2)


def test_heat_axisymmetric(meshes):
    k, a, b, Hh = 2.0, 1.0, 2.0, 0.5
    c = _case(meshes["ring_axi"], "heat", "axisymmetric", 2, mats={"m": Material(k=k)},
              bcs=[BC("temperature", group="left", value=100.0), BC("temperature", group="right", value=0.0)],
              qois=[QoI("T", "point", field="T", point=[1.5, 0.25]), QoI("Q", "reaction", group="left", component=0)])
    v = solve(c, backend="ccx").evaluate_all()
    assert rel(v["T"], 100 * (1 - math.log(1.5) / math.log(2.0))) < 1e-5
    assert rel(v["Q"], 2 * math.pi * k * Hh * 100 / math.log(b / a)) < 1e-5   # full 360 degrees


# ----------------------------------------------------------------------------- cross-check
def test_cross_check_skfem(meshes):
    """Same cases and P2 meshes through both backends; differences are printed (pytest -s)."""
    try:
        import feacore.solver.skfem_backend  # noqa: F401
    except Exception as e:  # pragma: no cover
        pytest.skip(f"skfem backend not importable: {e}")
    p = 10.0
    cases = {
        "lame_plane_strain": _case(meshes["annulus"], form="plane_strain",
                                   bcs=[BC("symmetry", group="sym_y"), BC("symmetry", group="sym_x"),
                                        BC("pressure", group="inner", value=p)],
                                   qois=[QoI("ua", "point", field="ux", point=[1.0, 0.0]),
                                         QoI("stt_a", "point", field="syy", point=[1.0, 0.0]),
                                         QoI("vm_max", "max", field="von_mises"),
                                         QoI("Ry", "reaction", group="sym_y", component=1),
                                         QoI("W", "strain_energy"),
                                         QoI("mean_vm", "mean", field="von_mises", group="body")]),
        "lame_axisymmetric": _case(meshes["ring_axi"], form="axisymmetric",
                                   bcs=[BC("symmetry", group="bottom"), BC("symmetry", group="top"),
                                        BC("pressure", group="left", value=p)],
                                   qois=[QoI("ua", "point", field="ux", point=[1.0, 0.25]),
                                         QoI("stt_a", "point", field="stt", point=[1.0, 0.25]),
                                         QoI("Rz", "reaction", group="bottom", component=1),
                                         QoI("W", "strain_energy")]),
        "pressure_3d": _case(meshes["qcyl"],
                             bcs=[BC("symmetry", group="x0"), BC("symmetry", group="y0"),
                                  BC("symmetry", group="z0"), BC("pressure", group="curved", value=p)],
                             qois=[QoI("ur", "point", field="ux", point=[1.0, 0.0, 0.5]),
                                   QoI("Ry", "reaction", group="y0", component=1), QoI("W", "strain_energy")]),
        "thermal_plane_strain": _case(meshes["annulus"], form="plane_strain", T_ref=20.0,
                                      mats={"m": Material(E=E, nu=NU, alpha=1e-5)},
                                      bcs=[BC("symmetry", group="sym_y"), BC("symmetry", group="sym_x")],
                                      loads=[Load("temperature", "20 + 100*(2 - sqrt(x*x + y*y))")],
                                      qois=[QoI("ua", "point", field="ux", point=[1.0, 0.0]),
                                            QoI("W", "strain_energy")]),
        "heat_axisymmetric": _case(meshes["ring_axi"], "heat", "axisymmetric", 2, mats={"m": Material(k=2.0)},
                                   bcs=[BC("temperature", group="left", value=100.0),
                                        BC("convection", group="right", h=5.0, T_inf=20.0)],
                                   loads=[Load("source", 3.0)],
                                   qois=[QoI("Tb", "point", field="T", point=[2.0, 0.25]),
                                         QoI("Q", "reaction", group="left", component=0),
                                         QoI("Tmean", "mean", field="T", group="body")]),
    }
    # displacement / temperature / reaction / energy / integral QoIs agree to the ccx output precision;
    # point and nodal-max stresses differ by the stress recovery (ccx: extrapolation + nodal averaging,
    # skfem: L2 projection), largest at boundary points
    tol = {"stt_a": 5e-3, "vm_max": 1e-3, "mean_vm": 1e-4, "ur": 2e-4}
    worst = {}
    for name, c in cases.items():
        vc = solve(c, backend="ccx").evaluate_all()
        vs = solve(c, backend="skfem").evaluate_all()
        for k in vc:
            d = abs(vc[k] - vs[k]) / max(abs(vs[k]), 1e-30)
            print(f"[ccx vs skfem] {name:22s} {k:8s} ccx={vc[k]:+.7e} skfem={vs[k]:+.7e} rel={d:.1e}")
            worst[(name, k)] = d
            assert d < tol.get(k, 2e-5), (name, k, vc[k], vs[k])
