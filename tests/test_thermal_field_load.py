"""A temperature field from a file as the thermal load of the fixed solver.

``Load(type="temperature", value={"field": "<.npz with x and T>"})`` (feacore/case.py, "Loads").

What is verified
    * a uniform field gives the displacements of the number;
    * a field that the FE space holds exactly gives the results of its expression to round-off
      (a quadratic field on straight-sided P2 cells, a linear field on P1 and on curved P2 cells);
    * a long hollow cylinder with the steady radial temperature profile: the hoop and axial stresses
      at both surfaces against the closed form of Timoshenko and Goodier (Theory of Elasticity, the
      long circular cylinder), axisymmetric and 3D, with the temperatures of a heat solve on the same
      mesh; the closed form is itself checked against the governing equations first;
    * a field from another mesh, or from a lower order, is refused; the case round-trips through JSON.

The closed form
    Temperature T_i at the inner radius a and 0 at the outer radius b, T = T_i ln(b/r) / ln(b/a); free
    surfaces, free ends (no axial force), far from the ends. With c = alpha E T_i / (2 (1 - nu) ln(b/a))
    and q = a^2 / (b^2 - a^2):
        s_r     = c (-ln(b/r) - q (1 - b^2/r^2) ln(b/a))
        s_theta = c (1 - ln(b/r) - q (1 + b^2/r^2) ln(b/a))
        s_z     = c (1 - 2 ln(b/r) - 2 q ln(b/a))
    so at the surfaces s_theta = s_z = c (1 - 2 b^2/(b^2 - a^2) ln(b/a)) at r = a and
    c (1 - 2 a^2/(b^2 - a^2) ln(b/a)) at r = b.
    The FE models hold the ends (u_z = 0, plane strain). A uniform axial stress is an exact solution of
    the free cylinder, so the free-end axial stress is the plane-strain one less its mean over the
    section; s_r and s_theta are the same in both.
"""

from __future__ import annotations

import json
import math
from contextlib import contextmanager

import numpy as np
import pytest

gmsh = pytest.importorskip("gmsh")

from feacore.case import (BC, FIELD_MATCH_TOL, Analysis, Case, CaseError, Load, Material, QoI,  # noqa: E402
                          as_field, read_temperature_field, write_temperature_field)
from feacore.solver import solve  # noqa: E402

E, NU, ALPHA, K = 1000.0, 0.3, 1e-3, 3.0
MAT = {"m": Material(E=E, nu=NU, alpha=ALPHA, k=K)}
A, B = 1.0, 2.0                       # inner and outer radius of the cylinder
T_A, T_B, T_REF = 400.0, 300.0, 250.0  # surface temperatures and the stress-free temperature


# --------------------------------------------------------------------------- meshes


@contextmanager
def _session(name):
    gmsh.initialize(readConfigFiles=False, interruptible=False)
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)
        gmsh.model.add(name)
        yield
    finally:
        gmsh.finalize()


def _groups(dim, rules):
    """Name the entities of dimension ``dim`` by their centre of mass: rules = [(name, predicate)],
    the first match wins."""
    found: dict[str, list[int]] = {}
    for _, tag in gmsh.model.getEntities(dim):
        c = gmsh.model.occ.getCenterOfMass(dim, tag)
        for name, pred in rules:
            if pred(*c):
                found.setdefault(name, []).append(tag)
                break
    for name, tags in found.items():
        gmsh.model.addPhysicalGroup(dim, tags, name=name)


def _write(path, dim, h, order):
    gmsh.option.setNumber("Mesh.MeshSizeMin", h)
    gmsh.option.setNumber("Mesh.MeshSizeMax", h)
    gmsh.model.mesh.generate(dim)
    if order == 2:
        gmsh.model.mesh.setOrder(2)
    gmsh.write(str(path))
    return str(path)


def box_mesh(path, h=0.35, order=2):
    """Unit cube (straight-sided cells); faces x0, x1; volume solid."""
    with _session("box"):
        gmsh.model.occ.addBox(0, 0, 0, 1, 1, 1)
        gmsh.model.occ.synchronize()
        _groups(2, [("x0", lambda x, y, z: abs(x) < 1e-9), ("x1", lambda x, y, z: abs(x - 1) < 1e-9)])
        _groups(3, [("solid", lambda x, y, z: True)])
        return _write(path, 3, h, order)


def two_block_mesh(path, h=0.3, order=2):
    """[0, 2] x [0, 1] in two cell groups: part_a (x < 1) and part_b (x > 1); edges left, right."""
    with _session("two"):
        occ = gmsh.model.occ
        a, b = occ.addRectangle(0, 0, 0, 1, 1), occ.addRectangle(1, 0, 0, 1, 1)
        occ.fragment([(2, a)], [(2, b)])
        occ.synchronize()
        _groups(1, [("left", lambda x, y, z: abs(x) < 1e-9), ("right", lambda x, y, z: abs(x - 2) < 1e-9)])
        _groups(2, [("part_a", lambda x, y, z: x < 1), ("part_b", lambda x, y, z: x > 1)])
        return _write(path, 2, h, order)


def axi_wall_mesh(path, h, height=0.5):
    """The wall of the cylinder in the r-z plane, [A, B] x [0, height]: inner, outer, bottom, top, domain."""
    with _session("axi"):
        gmsh.model.occ.addRectangle(A, 0, 0, B - A, height)
        gmsh.model.occ.synchronize()
        _groups(1, [("inner", lambda x, y, z: abs(x - A) < 1e-9), ("outer", lambda x, y, z: abs(x - B) < 1e-9),
                    ("bottom", lambda x, y, z: abs(y) < 1e-9), ("top", lambda x, y, z: abs(y - height) < 1e-9)])
        _groups(2, [("domain", lambda x, y, z: True)])
        return _write(path, 2, h, 2)


def quarter_cylinder_mesh(path, h, height=0.5, order=2):
    """A quarter of the cylinder (x, y >= 0), 0 <= z <= height, curved tetrahedra: inner, outer, xsym
    (x = 0), ysym (y = 0), z0, z1, solid."""
    with _session("quarter"):
        occ = gmsh.model.occ
        ring, _ = occ.cut([(2, occ.addDisk(0, 0, 0, B, B))], [(2, occ.addDisk(0, 0, 0, A, A))])
        quarter, _ = occ.intersect(ring, [(2, occ.addRectangle(0, 0, 0, 2 * B, 2 * B))])
        occ.extrude(quarter, 0, 0, height)
        occ.synchronize()
        _groups(2, [("z0", lambda x, y, z: abs(z) < 1e-7), ("z1", lambda x, y, z: abs(z - height) < 1e-7),
                    ("xsym", lambda x, y, z: abs(x) < 1e-7), ("ysym", lambda x, y, z: abs(y) < 1e-7),
                    ("inner", lambda x, y, z: math.hypot(x, y) < (A + B) / 2), ("outer", lambda x, y, z: True)])
        _groups(3, [("solid", lambda x, y, z: True)])
        return _write(path, 3, h, order)


@pytest.fixture(scope="module")
def meshdir(tmp_path_factory):
    return tmp_path_factory.mktemp("thermal_field_meshes")


@pytest.fixture(scope="module")
def box(meshdir):
    return box_mesh(meshdir / "box.msh")


def _elastic(mesh, loads, formulation="3d", order=2, bcs=None, qois=(), T_ref=20.0, base_dir=None):
    return Case(mesh=str(mesh), analysis=Analysis(physics="elasticity", formulation=formulation, order=order, T_ref=T_ref),
                materials=MAT, bcs=bcs or [BC(type="fixed", group="x0")], loads=loads, qois=list(qois),
                base_dir=str(base_dir) if base_dir else None)


def _field_load(path, group="*"):
    return Load(type="temperature", value={"field": str(path)}, group=group)


def _rel(a, b):
    return float(np.abs(np.asarray(a) - np.asarray(b)).max() / np.abs(np.asarray(b)).max())


# --------------------------------------------------------------------------- uniform, expression = file


def test_uniform_field_equals_the_number(box, tmp_path):
    ref = solve(_elastic(box, [Load(type="temperature", value=70.0)]))
    X, _ = ref.nodal("ux")
    f = write_temperature_field(tmp_path / "uniform.npz", X, np.full(len(X), 70.0))
    res = solve(_elastic(box, [_field_load(f)]))
    assert np.abs(ref.x).max() > 1e-3           # the load does something
    assert _rel(res.x, ref.x) < 1e-12           # observed 5e-15
    assert _rel(res.nodal("von_mises")[1], ref.nodal("von_mises")[1]) < 1e-11
    info = [w for w in res.info.warnings if "temperature field" in w]       # what was mapped is in the solve info
    assert len(info) == 1 and info[0].startswith("info: loads[0] temperature field 'uniform.npz'")
    assert f"{len(X)} points mapped to {len(X)} nodes" in info[0] and "T from 70 to 70" in info[0]


QUADRATIC = "20 + 100*x + 30*y**2 - 50*z*x"


def _quadratic(X):
    x, y, z = X.T
    return 20 + 100 * x + 30 * y**2 - 50 * z * x


def test_expression_and_file_give_identical_results(box, tmp_path):
    """Straight-sided P2 cells hold a quadratic field exactly: the expression (evaluated at the quadrature
    points) and the file (interpolated from the nodes) are the same load."""
    ref = solve(_elastic(box, [Load(type="temperature", value=QUADRATIC)]))
    X, _ = ref.nodal("ux")
    f = write_temperature_field(tmp_path / "quadratic.npz", X, _quadratic(X))
    res = solve(_elastic(box, [_field_load(f)]))
    assert _rel(res.x, ref.x) < 1e-12           # observed 5e-15
    for name in ("sxx", "syy", "szz", "sxy", "syz", "sxz", "von_mises"):
        assert _rel(res.nodal(name)[1], ref.nodal(name)[1]) < 1e-11, name
    # the order of the points in the file and single-precision coordinates do not matter
    p = np.random.default_rng(0).permutation(len(X))
    f2 = tmp_path / "shuffled.npz"
    np.savez(f2, x=X[p].astype(np.float32), T=_quadratic(X)[p], stress=np.zeros((len(X), 6)))   # extra arrays are ignored
    res2 = solve(_elastic(box, [_field_load(f2)]))
    assert _rel(res2.x, ref.x) < 1e-12


def test_linear_field_p1_and_p2_field_on_p1_analysis(box, tmp_path):
    """Order 1 holds a linear field exactly. Its nodes are the corners of the quadratic mesh, so a field at
    the nodes of an order-2 solve serves an order-1 analysis; the reverse is refused."""
    expr = "20 + 100*x - 40*y + 15*z"
    ref = solve(_elastic(box, [Load(type="temperature", value=expr)], order=1))
    X1, _ = ref.nodal("ux")
    X2, _ = solve(_elastic(box, [Load(type="temperature", value=1.0)], order=2)).nodal("ux")
    assert len(X2) > 4 * len(X1)

    def lin(X):
        return 20 + 100 * X[:, 0] - 40 * X[:, 1] + 15 * X[:, 2]

    f1 = write_temperature_field(tmp_path / "p1.npz", X1, lin(X1))
    f2 = write_temperature_field(tmp_path / "p2.npz", X2, lin(X2))
    assert _rel(solve(_elastic(box, [_field_load(f1)], order=1)).x, ref.x) < 1e-12
    assert _rel(solve(_elastic(box, [_field_load(f2)], order=1)).x, ref.x) < 1e-12
    with pytest.raises(CaseError, match="does not belong to this mesh"):
        solve(_elastic(box, [_field_load(f1)], order=2))


def test_linear_field_on_curved_cells(meshdir, tmp_path):
    """Curved (isoparametric) P2 cells hold a linear field exactly, and no more: the file and the expression
    agree to round-off for a linear field, and differ by the interpolation error for a quadratic one."""
    mesh = quarter_cylinder_mesh(meshdir / "quarter_coarse.msh", 0.3)
    bcs = [BC(type="symmetry", group=g) for g in ("xsym", "ysym", "z0")]

    def pair(expr, fn):
        ref = solve(_elastic(mesh, [Load(type="temperature", value=expr)], bcs=bcs))
        X, _ = ref.nodal("ux")
        f = write_temperature_field(tmp_path / "T.npz", X, fn(X))
        return _rel(solve(_elastic(mesh, [_field_load(f)], bcs=bcs)).x, ref.x)

    assert pair("20 + 100*x - 40*y + 15*z", lambda X: 20 + 100 * X[:, 0] - 40 * X[:, 1] + 15 * X[:, 2]) < 1e-12
    d = pair(QUADRATIC, _quadratic)
    assert 1e-8 < d < 1e-3                       # observed 1.4e-5 (4e-14 for the linear field)


def test_field_on_a_cell_group_and_2d_coordinates(meshdir, tmp_path):
    """A field that covers only the cells of its load's group is enough; the other cells keep their own
    load. A 2D case also takes a file with a constant third coordinate."""
    mesh = two_block_mesh(meshdir / "two.msh")
    bcs = [BC(type="fixed", group="left")]
    expr = "20 + 60*x + 25*x*y"
    ref = solve(_elastic(mesh, [Load(type="temperature", value=expr, group="part_a"),
                                Load(type="temperature", value=55.0, group="part_b")], "plane_strain", bcs=bcs))
    X, _ = ref.nodal("ux")
    in_a = X[:, 0] <= 1 + 1e-9
    Xa = np.column_stack([X[in_a], np.zeros(in_a.sum())])            # (n, 3) with z = 0
    f = write_temperature_field(tmp_path / "part_a.npz", Xa, 20 + 60 * Xa[:, 0] + 25 * Xa[:, 0] * Xa[:, 1])
    loads = [_field_load(f, "part_a"), Load(type="temperature", value=55.0, group="part_b")]
    res = solve(_elastic(mesh, loads, "plane_strain", bcs=bcs))
    assert _rel(res.x, ref.x) < 1e-12
    with pytest.raises(CaseError, match="on the cell group 'part_b'"):   # the file has no points there
        solve(_elastic(mesh, [_field_load(f, "part_b")], "plane_strain", bcs=bcs))
    np.savez(tmp_path / "tilted.npz", x=np.column_stack([X, X[:, 0]]), T=np.zeros(len(X)))
    with pytest.raises(CaseError, match="3 coordinates but the analysis is 2D"):
        solve(_elastic(mesh, [_field_load(tmp_path / "tilted.npz")], "plane_strain", bcs=bcs))


# --------------------------------------------------------------------------- the hollow cylinder


def _T_exact(r):
    return T_A + (T_B - T_A) * np.log(np.asarray(r, dtype=float) / A) / math.log(B / A)


def _cylinder_stress(r, a=A, b=B, Ti=T_A - T_B, E_=E, nu=NU, alpha=ALPHA):
    """(s_r, s_theta, s_z) of the closed form in the module docstring."""
    r = np.asarray(r, dtype=float)
    L = math.log(b / a)
    c = alpha * E_ * Ti / (2 * (1 - nu) * L)
    q = a**2 / (b**2 - a**2)
    lr = np.log(b / r)
    return (c * (-lr - q * (1 - b**2 / r**2) * L), c * (1 - lr - q * (1 + b**2 / r**2) * L),
            c * (1 - 2 * lr - 2 * q * L))


def _surface_stress(r):
    """The textbook's surface value s_theta = s_z at r = a or r = b."""
    L = math.log(B / A)
    c = ALPHA * E * (T_A - T_B) / (2 * (1 - NU) * L)
    other = B if r == A else A
    return c * (1 - 2 * other**2 / (B**2 - A**2) * L)


@pytest.mark.parametrize("a,b,nu", [(A, B, NU), (0.7, 0.9, 0.25), (1.0, 5.0, 0.4)])
def test_closed_form_satisfies_the_governing_equations(a, b, nu):
    """The derivation, checked numerically before the solver is compared with it. A stress field that is in
    equilibrium, has free surfaces and no axial force, and whose strains (with the thermal strain of the
    steady profile) are compatible with a uniform axial strain, is the solution: it is unique."""
    Ti, E_, alpha = 100.0, 1000.0, 1e-3
    r = np.linspace(a, b, 2001)
    d = 1e-6 * a

    def stress(x):
        return _cylinder_stress(x, a, b, Ti, E_, nu, alpha)

    def T(x):
        return Ti * np.log(b / x) / math.log(b / a)

    def strains(x):
        sr, st, sz = stress(x)
        return ((sr - nu * (st + sz)) / E_ + alpha * T(x), (st - nu * (sr + sz)) / E_ + alpha * T(x),
                (sz - nu * (sr + st)) / E_ + alpha * T(x))

    sr, st, sz = stress(r)
    scale = np.abs(st).max()
    # the temperature: steady radial conduction, d/dr (r dT/dr) = 0, with T(a) = T_i and T(b) = 0
    flow = r * (T(r + d) - T(r - d)) / (2 * d)
    assert np.ptp(flow) < 1e-7 * abs(flow).max() and abs(T(a) - Ti) < 1e-12 and abs(T(b)) < 1e-12
    # equilibrium d s_r / dr + (s_r - s_theta) / r = 0
    dsr = (stress(r + d)[0] - stress(r - d)[0]) / (2 * d)
    assert np.abs(dsr + (sr - st) / r).max() < 1e-6 * scale / a
    # free surfaces, no axial force
    assert abs(sr[0]) < 1e-12 * scale and abs(sr[-1]) < 1e-12 * scale
    xg, wg = np.polynomial.legendre.leggauss(60)             # Gauss-Legendre on [a, b]
    rg = 0.5 * (a + b) + 0.5 * (b - a) * xg
    force = 0.5 * (b - a) * float(np.sum(wg * stress(rg)[2] * rg))
    assert abs(force) < 1e-10 * scale * (b**2 - a**2)
    # compatibility: e_r = d(r e_theta)/dr (one radial displacement u = r e_theta), e_z uniform
    er, et, ez = strains(r)
    d_ret = ((r + d) * strains(r + d)[1] - (r - d) * strains(r - d)[1]) / (2 * d)
    assert np.abs(er - d_ret).max() < 1e-6 * np.abs(et).max()
    assert np.ptp(ez) < 1e-12 * np.abs(et).max()
    # the surface values quoted in the textbook: s_theta = s_z there
    L, c = math.log(b / a), alpha * E_ * Ti / (2 * (1 - nu) * math.log(b / a))
    assert st[0] == pytest.approx(c * (1 - 2 * b**2 / (b**2 - a**2) * L), rel=1e-12) and sz[0] == pytest.approx(st[0], rel=1e-12)
    assert st[-1] == pytest.approx(c * (1 - 2 * a**2 / (b**2 - a**2) * L), rel=1e-12) and sz[-1] == pytest.approx(st[-1], rel=1e-12)
    assert st[0] < 0 < st[-1]                   # the hot inner surface is in compression


def _axi_cylinder(mesh, tmp_path, from_heat_solve):
    heat = Case(mesh=mesh, analysis=Analysis(physics="heat", formulation="axisymmetric", order=2), materials=MAT,
                bcs=[BC(type="temperature", group="inner", value=T_A), BC(type="temperature", group="outer", value=T_B)])
    X, Th = solve(heat).nodal("T")
    f = write_temperature_field(tmp_path / "T.npz", X, Th if from_heat_solve else _T_exact(X[:, 0]))
    bcs = [BC(type="displacement", group=g, components=[1], value=0.0) for g in ("bottom", "top")]
    res = solve(_elastic(mesh, [_field_load(f)], "axisymmetric", bcs=bcs, T_ref=T_REF,
                         qois=[QoI(name="mean_sz", type="mean", field="syy", group="domain")]))
    mean_sz = res.evaluate_all()["mean_sz"]
    out = {}
    for name, r in (("a", A), ("b", B)):
        p = [[r, 0.25]]
        out[name] = (res.probe("stt", p)[0], res.probe("syy", p)[0] - mean_sz, res.probe("sxx", p)[0])
    return out, float(np.abs(Th - _T_exact(X[:, 0])).max())


@pytest.mark.parametrize("from_heat_solve", [False, True])
def test_hollow_cylinder_axisymmetric_against_closed_form(meshdir, tmp_path, from_heat_solve):
    """Hoop and axial stress at both surfaces, with the exact profile at the nodes and with the
    temperatures of a heat solve on the same mesh. Observed errors (h = 0.05): hoop +8.8e-4 and axial
    +8.7e-4 at r = a, -1.5e-4 and -1.4e-4 at r = b; (h = 0.1): +2.4e-3, +2.6e-3, -4.3e-4, -4.7e-4."""
    err = {}
    for h in (0.1, 0.05):
        got, dT = _axi_cylinder(axi_wall_mesh(meshdir / f"axi_{h}.msh", h), tmp_path, from_heat_solve)
        assert dT < 1e-3                         # the heat solve gives the logarithmic profile (K, of 100 K)
        err[h] = {}
        for name, r in (("a", A), ("b", B)):
            exact = _surface_stress(r)
            hoop, axial, radial = got[name]
            err[h][name] = max(abs(hoop / exact - 1), abs(axial / exact - 1))
            assert abs(radial) < 1e-2 * abs(exact)       # the free surface carries no radial stress
    assert err[0.05]["a"] < 2e-3 and err[0.05]["b"] < 5e-4
    assert err[0.1]["a"] < 5e-3 and err[0.1]["b"] < 1.5e-3
    assert err[0.05]["a"] < 0.6 * err[0.1]["a"] and err[0.05]["b"] < 0.6 * err[0.1]["b"]     # it converges


def test_hollow_cylinder_3d_with_the_field_of_a_heat_solve(meshdir, tmp_path):
    """The chain a coupled analysis uses, on curved quadratic tetrahedra: heat solve -> result.nodal("T")
    -> file -> elasticity. Stresses on the planes y = 0 and x = 0, where the hoop direction is a global
    axis, as the mean over five heights. Observed errors (h = 0.12, 6910 nodes): hoop -2.7e-3 and axial
    -7.1e-4 at r = a, -2.3e-4 and -3.2e-4 at r = b. A single point value at r = a is off by up to 7e-3,
    and the mean by 1e-3 to 5e-3 on meshes of h = 0.1 to 0.2 (it does not fall steadily: recovered point
    stresses on unstructured tetrahedra scatter)."""
    mesh = quarter_cylinder_mesh(meshdir / "quarter.msh", 0.12)
    heat = Case(mesh=mesh, analysis=Analysis(physics="heat", formulation="3d", order=2), materials=MAT,
                bcs=[BC(type="temperature", group="inner", value=T_A), BC(type="temperature", group="outer", value=T_B)])
    X, Th = solve(heat).nodal("T")
    assert np.abs(Th - _T_exact(np.hypot(X[:, 0], X[:, 1]))).max() < 0.05       # K, of 100 K
    f = write_temperature_field(tmp_path / "T.npz", X, Th)
    bcs = [BC(type="symmetry", group=g) for g in ("xsym", "ysym", "z0", "z1")]
    res = solve(_elastic(mesh, [_field_load(f)], bcs=bcs, T_ref=T_REF,
                         qois=[QoI(name="mean_sz", type="mean", field="szz", group="solid")]))
    mean_sz = res.evaluate_all()["mean_sz"]
    zs = np.linspace(0.1, 0.4, 5)
    for r, tol in ((A, 1e-2), (B, 3e-3)):
        exact = _surface_stress(r)
        on_x = np.column_stack([np.full(5, r), np.zeros(5), zs])        # hoop direction y
        on_y = np.column_stack([np.zeros(5), np.full(5, r), zs])        # hoop direction x
        hoop = 0.5 * (res.probe("syy", on_x).mean() + res.probe("sxx", on_y).mean())
        axial = 0.5 * (res.probe("szz", on_x).mean() + res.probe("szz", on_y).mean()) - mean_sz
        assert abs(hoop / exact - 1) < tol, (r, hoop, exact)
        assert abs(axial / exact - 1) < tol, (r, axial, exact)


# --------------------------------------------------------------------------- refusals, JSON


def test_field_from_another_mesh_is_rejected(box, meshdir, tmp_path):
    X, _ = solve(_elastic(box, [Load(type="temperature", value=1.0)])).nodal("ux")
    f = write_temperature_field(tmp_path / "T.npz", X, _quadratic(X))
    other = box_mesh(meshdir / "box_other.msh", h=0.3)
    with pytest.raises(CaseError) as e:
        solve(_elastic(other, [_field_load(f)]))
    msg = str(e.value)
    assert "loads[0] temperature field 'T.npz' does not belong to this mesh" in msg
    assert f"The file has {len(X)} points" in msg and "result.nodal('T')" in msg
    # the same nodes moved by more than the tolerance, and a field in other length units
    diag = math.sqrt(3.0)
    write_temperature_field(tmp_path / "near.npz", X + 0.4 * FIELD_MATCH_TOL * diag, _quadratic(X))
    write_temperature_field(tmp_path / "far.npz", X + 4 * FIELD_MATCH_TOL * diag, _quadratic(X))
    write_temperature_field(tmp_path / "mm.npz", 1000 * X, _quadratic(X))
    solve(_elastic(box, [_field_load(tmp_path / "near.npz")]))
    for name in ("far.npz", "mm.npz"):
        with pytest.raises(CaseError, match="does not belong to this mesh"):
            solve(_elastic(box, [_field_load(tmp_path / name)]))


def test_bad_field_files_are_case_errors(box, tmp_path):
    X, _ = solve(_elastic(box, [Load(type="temperature", value=1.0)])).nodal("ux")
    with pytest.raises(CaseError, match="temperature field file not found"):
        solve(_elastic(box, [_field_load(tmp_path / "missing.npz")]))
    np.savez(tmp_path / "no_T.npz", x=X, temperature=np.zeros(len(X)))
    with pytest.raises(CaseError, match=r"x \(n, dim\) and T \(n,\) is required"):
        solve(_elastic(box, [_field_load(tmp_path / "no_T.npz")]))
    np.savez(tmp_path / "short.npz", x=X, T=np.zeros(len(X) - 1))
    with pytest.raises(CaseError, match="need x"):
        solve(_elastic(box, [_field_load(tmp_path / "short.npz")]))
    T = np.zeros(len(X))
    T[3] = np.nan
    np.savez(tmp_path / "nan.npz", x=X, T=T)
    with pytest.raises(CaseError, match="not finite"):
        solve(_elastic(box, [_field_load(tmp_path / "nan.npz")]))
    (tmp_path / "text.npz").write_text("not an archive")
    with pytest.raises(CaseError, match="cannot read temperature field"):
        solve(_elastic(box, [_field_load(tmp_path / "text.npz")]))
    with pytest.raises(CaseError, match="x .* and T"):
        write_temperature_field(tmp_path / "bad.npz", X, np.zeros(3))
    x, T = read_temperature_field(write_temperature_field(tmp_path / "sub" / "field.dat", X, _quadratic(X)), 3)
    assert np.array_equal(x, X) and np.array_equal(T, _quadratic(X))      # any file name, folders made


def test_validation_of_the_field_value(box):
    def errors(load, physics="elasticity"):
        bcs = [BC(type="fixed", group="x0")] if physics == "elasticity" else [BC(type="temperature", group="x0", value=0.0)]
        return Case(mesh=str(box), analysis=Analysis(physics=physics, formulation="3d"), materials=MAT, bcs=bcs,
                    loads=[load]).validate()

    assert errors(Load(type="temperature", value={"field": "T.npz"})) == []
    for bad in ({"file": "T.npz"}, {"field": ""}, {"field": 3}, {"field": "T.npz", "scale": 2.0}, {}):
        errs = errors(Load(type="temperature", value=bad))
        assert len(errs) == 1 and "a field value must be" in errs[0], bad
    errs = errors(Load(type="source", value={"field": "T.npz"}), physics="heat")
    assert len(errs) == 1 and "temperature load only" in errs[0]
    assert len(errors(Load(type="body_force", value={"field": "T.npz"}))) == 1      # as before: a list is required
    with pytest.raises(CaseError, match="not a number or an expression"):
        as_field({"field": "T.npz"}, np.zeros(3), np.zeros(3))


def test_json_round_trip_and_relative_path(box, tmp_path, monkeypatch):
    """The case round-trips through JSON; a relative field path is relative to the case file's folder."""
    ref = solve(_elastic(box, [Load(type="temperature", value=QUADRATIC)]))
    X, _ = ref.nodal("ux")
    folder = tmp_path / "case"
    write_temperature_field(folder / "fields" / "T.npz", X, _quadratic(X))
    case = _elastic(box, [Load(type="temperature", value={"field": "fields/T.npz"})],
                    qois=[QoI(name="e", type="strain_energy")])
    d = case.to_dict()
    assert d["loads"] == [{"type": "temperature", "value": {"field": "fields/T.npz"}, "group": "*"}]
    assert Case.from_dict(json.loads(json.dumps(d))) == case
    path = case.save(folder / "case.json")
    loaded = Case.load(path)
    assert loaded == case and loaded.file_path("fields/T.npz") == folder / "fields" / "T.npz"
    monkeypatch.chdir(tmp_path)                 # the folder of the case file counts, not the working directory
    res = solve(str(path))
    assert _rel(res.x, ref.x) < 1e-12
    assert res.evaluate_all()["e"] == pytest.approx(ref.evaluate(QoI(name="e", type="strain_energy")), rel=1e-12)
    with pytest.raises(CaseError, match="temperature field file not found"):      # no base_dir: the working directory
        solve(case)


def test_ccx_backend_refuses_a_field_file(box, tmp_path):
    X, _ = solve(_elastic(box, [Load(type="temperature", value=1.0)])).nodal("ux")
    f = write_temperature_field(tmp_path / "T.npz", X, _quadratic(X))
    with pytest.raises(NotImplementedError, match="field file"):
        solve(_elastic(box, [_field_load(f)]), backend="ccx")
