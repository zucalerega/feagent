"""Verification tests of the fixed scikit-fem solver against closed-form solutions.

Fast (a few seconds): small gmsh meshes generated into a temporary directory.
"""

from __future__ import annotations

import json
import math
from contextlib import contextmanager

import numpy as np
import pytest

gmsh = pytest.importorskip("gmsh")

from feacore.case import BC, Analysis, Case, CaseError, Load, Material, QoI, Region  # noqa: E402
from feacore.solver import solve  # noqa: E402

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


def _name_by_com(dim, rules):
    groups = {n: [] for n, _ in rules}
    for _, t in gmsh.model.getEntities(dim):
        x, y, z = gmsh.model.occ.getCenterOfMass(dim, t)
        for n, pred in rules:
            if pred(x, y, z):
                groups[n].append(t)
    for n, tags in groups.items():
        if tags:
            gmsh.model.addPhysicalGroup(dim, tags, name=n)


def _point_group(name, x, y, z=0.0, e=1e-9):
    pts = gmsh.model.getEntitiesInBoundingBox(x - e, y - e, z - e, x + e, y + e, z + e, 0)
    gmsh.model.addPhysicalGroup(0, [t for _, t in pts], name=name)


def _distort(dim, h, amount, seed):
    rng = np.random.default_rng(seed)
    for _, tag in gmsh.model.getEntities(dim):
        nt, xyz, _ = gmsh.model.mesh.getNodes(dim, tag, includeBoundary=False, returnParametricCoord=False)
        xyz = xyz.reshape(-1, 3)
        d = rng.uniform(-1, 1, size=xyz.shape) * amount * h
        if dim == 2:
            d[:, 2] = 0
        for t_, p in zip(nt, xyz + d):
            gmsh.model.mesh.setNode(int(t_), p.tolist(), [])


def _straighten(dim):
    """Midside nodes at the midpoints of the (perturbed) straight edges."""
    etype, pairs = (9, [(0, 1), (1, 2), (2, 0)]) if dim == 2 else \
        (11, [(0, 1), (1, 2), (2, 0), (0, 3), (2, 3), (1, 3)])
    tags, coords, _ = gmsh.model.mesh.getNodes()
    pos = dict(zip(tags.tolist(), coords.reshape(-1, 3)))
    _, enodes = gmsh.model.mesh.getElementsByType(etype)
    enodes = enodes.reshape(-1, 6 if dim == 2 else 10)
    new = {}
    for row in enodes:
        for j, (a, b) in enumerate(pairs):
            new.setdefault(int(row[dim + 1 + j]), 0.5 * (pos[int(row[a])] + pos[int(row[b])]))
    for t, x in new.items():
        gmsh.model.mesh.setNode(t, x.tolist(), [])


def _finish(path, dim, h, order, distort=0.0, seed=0):
    gmsh.option.setNumber("Mesh.MeshSizeMin", h)
    gmsh.option.setNumber("Mesh.MeshSizeMax", h)
    gmsh.model.mesh.generate(dim)
    if distort > 0:
        _distort(dim, h, distort, seed)
    if order == 2:
        gmsh.model.mesh.setOrder(2)
        if distort > 0:
            _straighten(dim)
    gmsh.write(str(path))
    return str(path)


def rect_mesh(path, L=1.0, H=1.0, h=0.25, order=1, distort=0.0, seed=0, x0=0.0):
    """[x0, x0+L] x [0, H]; groups left/right/bottom/top, p00/p10/p01, domain."""
    with _session("rect"):
        gmsh.model.occ.addRectangle(x0, 0, 0, L, H)
        gmsh.model.occ.synchronize()
        e = 1e-9
        _name_by_com(1, [("left", lambda x, y, z: abs(x - x0) < e), ("right", lambda x, y, z: abs(x - x0 - L) < e),
                         ("bottom", lambda x, y, z: abs(y) < e), ("top", lambda x, y, z: abs(y - H) < e)])
        _point_group("p00", x0, 0)
        _point_group("p10", x0 + L, 0)
        _point_group("p01", x0, H)
        gmsh.model.addPhysicalGroup(2, [t for _, t in gmsh.model.getEntities(2)], name="domain")
        return _finish(path, 2, h, order, distort, seed)


def two_material_mesh(path, L=2.0, H=1.0, h=0.25, order=2):
    with _session("rect2"):
        occ = gmsh.model.occ
        a = occ.addRectangle(0, 0, 0, L / 2, H)
        b = occ.addRectangle(L / 2, 0, 0, L / 2, H)
        occ.fragment([(2, a)], [(2, b)])
        occ.synchronize()
        e = 1e-9
        _name_by_com(1, [("left", lambda x, y, z: abs(x) < e), ("right", lambda x, y, z: abs(x - L) < e)])
        _name_by_com(2, [("mat_a", lambda x, y, z: x < L / 2), ("mat_b", lambda x, y, z: x > L / 2)])
        _point_group("p00", 0, 0)
        return _finish(path, 2, h, order)


def box_mesh(path, L=1.0, h=0.4, order=1, distort=0.0, seed=0):
    """Unit cube; faces x0..z1, edge_z (x=y=0), points p000/p100/p010, volume solid."""
    with _session("box"):
        gmsh.model.occ.addBox(0, 0, 0, L, L, L)
        gmsh.model.occ.synchronize()
        e = 1e-9
        rules = []
        for i, ax in enumerate("xyz"):
            rules.append((f"{ax}0", lambda x, y, z, i=i: abs((x, y, z)[i]) < e))
            rules.append((f"{ax}1", lambda x, y, z, i=i: abs((x, y, z)[i] - L) < e))
        _name_by_com(2, rules)
        ed = gmsh.model.getEntitiesInBoundingBox(-1e-6, -1e-6, -1e-6, 1e-6, 1e-6, L + 1e-6, 1)
        gmsh.model.addPhysicalGroup(1, [t for _, t in ed], name="edge_z")
        _point_group("p000", 0, 0, 0)
        _point_group("p100", L, 0, 0)
        _point_group("p010", 0, L, 0)
        gmsh.model.addPhysicalGroup(3, [t for _, t in gmsh.model.getEntities(3)], name="solid")
        return _finish(path, 3, h, order, distort, seed)


def quarter_ring_mesh(path, a=1.0, b=2.0, h=0.2, order=2):
    """Quarter annulus a < r < b, x, y >= 0; groups inner/outer/xsym (x=0)/ysym (y=0)."""
    with _session("qring"):
        occ = gmsh.model.occ
        ring, _ = occ.cut([(2, occ.addDisk(0, 0, 0, b, b))], [(2, occ.addDisk(0, 0, 0, a, a))])
        occ.intersect(ring, [(2, occ.addRectangle(0, 0, 0, 2 * b, 2 * b))])
        occ.synchronize()
        groups = {"inner": [], "outer": [], "xsym": [], "ysym": []}
        for _, t in gmsh.model.getEntities(1):
            x, y, _ = occ.getCenterOfMass(1, t)
            key = "xsym" if abs(x) < 1e-7 else "ysym" if abs(y) < 1e-7 else \
                "inner" if math.hypot(x, y) < (a + b) / 2 else "outer"
            groups[key].append(t)
        for n, tg in groups.items():
            gmsh.model.addPhysicalGroup(1, tg, name=n)
        gmsh.model.addPhysicalGroup(2, [t for _, t in gmsh.model.getEntities(2)], name="domain")
        return _finish(path, 2, h, order)


@pytest.fixture(scope="module")
def meshdir(tmp_path_factory):
    return tmp_path_factory.mktemp("skfem_meshes")


E, NU = 1000.0, 0.3
S11, S22, S12 = 1.0, 2.0, 0.5


def _case(mesh, physics="elasticity", formulation="plane_stress", order=1, bcs=(), loads=(), qois=(),
          materials=None, regions=(), **an):
    return Case(mesh=str(mesh), analysis=Analysis(physics=physics, formulation=formulation, order=order, **an),
                materials=materials or {"m": Material(E=E, nu=NU, alpha=1e-3, k=3.0)},
                bcs=list(bcs), loads=list(loads), qois=list(qois), regions=list(regions))


def _maxerr(res, field, exact):
    return float(np.abs(res.nodal(field)[1] - exact).max())


# --------------------------------------------------------------------------- patch tests


@pytest.mark.parametrize("formulation", ["plane_stress", "plane_strain"])
@pytest.mark.parametrize("mesh_order,order", [(1, 1), (1, 2), (2, 1), (2, 2)])
def test_patch_plane(meshdir, formulation, mesh_order, order):
    mesh = rect_mesh(meshdir / f"patch2d_{mesh_order}.msh", h=0.2, order=mesh_order, distort=0.3, seed=1)
    szz = 0.0 if formulation == "plane_stress" else NU * (S11 + S22)
    exx = (S11 - NU * (S22 + szz)) / E
    eyy = (S22 - NU * (S11 + szz)) / E
    exy = (1 + NU) * S12 / E
    bcs = [BC(type="traction", group="left", value=[-S11, -S12]), BC(type="traction", group="right", value=[S11, S12]),
           BC(type="traction", group="bottom", value=[-S12, -S22]), BC(type="traction", group="top", value=[S12, S22]),
           BC(type="fixed", group="p00"), BC(type="displacement", group="p10", components=[1], value=exy)]
    r = solve(_case(mesh, formulation=formulation, order=order, bcs=bcs, thickness=0.7))
    for f, ex in [("sxx", S11), ("syy", S22), ("sxy", S12), ("szz", szz), ("sxz", 0.0)]:
        assert _maxerr(r, f, ex) < 1e-9, f
    X, ux = r.nodal("ux")
    _, uy = r.nodal("uy")
    assert np.abs(ux - (exx * X[:, 0] + exy * X[:, 1])).max() < 1e-9 * exx
    assert np.abs(uy - (exy * X[:, 0] + eyy * X[:, 1])).max() < 1e-9 * exx
    pts = np.random.default_rng(0).uniform(0.02, 0.98, size=(15, 2))
    vm = math.sqrt(0.5 * ((S11 - S22) ** 2 + (S22 - szz) ** 2 + (szz - S11) ** 2) + 3 * S12 ** 2)
    assert np.abs(r.probe("von_mises", pts) - vm).max() < 1e-9
    assert np.abs(r.probe("sxy", pts) - S12).max() < 1e-9
    s1 = (S11 + S22) / 2 + math.hypot((S11 - S22) / 2, S12)
    assert abs(r.probe("s1", [[0.5, 0.5]])[0] - max(s1, szz)) < 1e-9
    assert r.info.order == order and r.info.cell_type == ("triangle6" if mesh_order == 2 else "triangle")


@pytest.mark.parametrize("order", [1, 2])
def test_patch_dirichlet_expressions(meshdir, order):
    mesh = rect_mesh(meshdir / "patch2d_2.msh", h=0.2, order=2, distort=0.3, seed=1)
    exx, eyy, exy = 1e-3, -2e-3, 5e-4
    val = [f"{exx}*x + {exy}*y", f"{exy}*x + {eyy}*y"]
    bcs = [BC(type="displacement", group=g, components=[0, 1], value=val) for g in ("left", "right", "bottom", "top")]
    r = solve(_case(mesh, formulation="plane_stress", order=order, bcs=bcs, thickness=2.0,
                    qois=[QoI(name="Rx", type="reaction", group="right", component=0)]))
    lam = 2 * (E * NU / ((1 + NU) * (1 - 2 * NU))) * (E / (2 * (1 + NU))) / (
        E * NU / ((1 + NU) * (1 - 2 * NU)) + 2 * E / (2 * (1 + NU)))
    mu = E / (2 * (1 + NU))
    sxx = lam * (exx + eyy) + 2 * mu * exx
    assert _maxerr(r, "sxx", sxx) < 1e-9
    assert _maxerr(r, "sxy", 2 * mu * exy) < 1e-9
    # reaction on the right edge = sxx * height * thickness
    assert abs(r.evaluate(QoI(name="Rx", type="reaction", group="right", component=0)) - sxx * 1.0 * 2.0) < 1e-9


@pytest.mark.parametrize("x0", [0.0, 0.5])
@pytest.mark.parametrize("mesh_order,order", [(1, 1), (1, 2), (2, 2)])
def test_patch_axisymmetric(meshdir, x0, mesh_order, order):
    mesh = rect_mesh(meshdir / f"patchaxi_{mesh_order}_{x0}.msh", h=0.2, order=mesh_order, distort=0.3, seed=2, x0=x0)
    a = (S11 - NU * (S11 + S22)) / E   # e_rr = e_tt
    b = (S22 - 2 * NU * S11) / E       # e_zz
    bcs = [BC(type="traction", group="right", value=[S11, 0.0]), BC(type="traction", group="bottom", value=[0.0, -S22]),
           BC(type="traction", group="top", value=[0.0, S22]), BC(type="displacement", group="p00", components=[1], value=0.0)]
    if x0 > 0:
        bcs.append(BC(type="traction", group="left", value=[-S11, 0.0]))
    r = solve(_case(mesh, formulation="axisymmetric", order=order, bcs=bcs))
    for f, ex in [("sxx", S11), ("syy", S22), ("sxy", 0.0), ("stt", S11), ("szz", S11)]:
        assert _maxerr(r, f, ex) < 1e-9, f
    X, ur = r.nodal("ux")
    _, uz = r.nodal("uy")
    assert np.abs(ur - a * X[:, 0]).max() < 1e-9 * abs(a)
    assert np.abs(uz - b * X[:, 1]).max() < 1e-9 * abs(a)


@pytest.mark.parametrize("mesh_order,order", [(1, 1), (1, 2), (2, 2)])
def test_patch_3d(meshdir, mesh_order, order):
    mesh = box_mesh(meshdir / f"patch3d_{mesh_order}.msh", h=0.4, order=mesh_order, distort=0.25, seed=3)
    s = np.array([[1.0, 0.3, -0.2], [0.3, 2.0, 0.4], [-0.2, 0.4, -1.5]])
    eps = ((1 + NU) * s - NU * np.trace(s) * np.eye(3)) / E
    normals = {"x0": [-1, 0, 0], "x1": [1, 0, 0], "y0": [0, -1, 0], "y1": [0, 1, 0], "z0": [0, 0, -1], "z1": [0, 0, 1]}
    bcs = [BC(type="traction", group=g, value=(s @ np.array(n, float)).tolist()) for g, n in normals.items()]
    bcs += [BC(type="fixed", group="p000"),
            BC(type="displacement", group="p100", components=[1, 2], value=[eps[1, 0], eps[2, 0]]),
            BC(type="displacement", group="p010", components=[2], value=[eps[2, 1]])]
    r = solve(_case(mesh, formulation="3d", order=order, bcs=bcs))
    for f, (i, j) in [("sxx", (0, 0)), ("syy", (1, 1)), ("szz", (2, 2)), ("sxy", (0, 1)), ("syz", (1, 2)), ("sxz", (0, 2))]:
        assert _maxerr(r, f, s[i, j]) < 1e-9, f
    X = r.nodal("ux")[0]
    U = np.column_stack([r.nodal(c)[1] for c in ("ux", "uy", "uz")])
    assert np.abs(U - X @ eps.T).max() < 1e-9 * np.abs(eps).max()
    ev = np.linalg.eigvalsh(s)
    pts = np.random.default_rng(1).uniform(0.05, 0.95, size=(10, 3))
    assert np.abs(r.probe("s3", pts) - ev[0]).max() < 1e-9
    assert np.abs(r.probe("tresca", pts) - (ev[2] - ev[0])).max() < 1e-9


# --------------------------------------------------------------------------- Lamé cylinder

P_IN, A_IN, B_OUT = 10.0, 1.0, 2.0
LA = P_IN * A_IN ** 2 / (B_OUT ** 2 - A_IN ** 2)
LB = P_IN * A_IN ** 2 * B_OUT ** 2 / (B_OUT ** 2 - A_IN ** 2)


def _lame_ur(r):  # plane strain / axisymmetric with u_z = 0
    return (1 + NU) / E * ((1 - 2 * NU) * LA * r + LB / r)


def _lame_stt(r):
    return LA + LB / r ** 2


def _lame_srr(r):
    return LA - LB / r ** 2


def _lame_plane_strain(meshdir, h, order):
    mesh = quarter_ring_mesh(meshdir / f"qring_{h}_{order}.msh", a=A_IN, b=B_OUT, h=h, order=order)
    q = [QoI(name="ur_a", type="point", field="ux", point=[A_IN, 0.0]),
         QoI(name="stt_a", type="point", field="syy", point=[A_IN, 0.0]),
         QoI(name="stt_b", type="point", field="sxx", point=[0.0, B_OUT]),
         QoI(name="srr_mid", type="point", field="sxx", point=[1.5, 0.0]),
         QoI(name="Rx", type="reaction", group="xsym", component=0),
         QoI(name="U", type="strain_energy")]
    bcs = [BC(type="symmetry", group="xsym"), BC(type="symmetry", group="ysym"),
           BC(type="pressure", group="inner", value=P_IN)]
    return solve(_case(mesh, formulation="plane_strain", order=order, bcs=bcs, qois=q)).evaluate_all()


@pytest.mark.parametrize("order,tol_u,tol_s", [(1, 1e-2, 4e-2), (2, 1e-4, 1e-2)])
def test_lame_plane_strain(meshdir, order, tol_u, tol_s):
    v = _lame_plane_strain(meshdir, 0.1, order)
    assert abs(v["ur_a"] / _lame_ur(A_IN) - 1) < tol_u
    assert abs(v["stt_a"] / _lame_stt(A_IN) - 1) < tol_s
    assert abs(v["stt_b"] / _lame_stt(B_OUT) - 1) < tol_s
    # equilibrium of the quarter: x-reaction on x = 0 balances the hoop force: -int_a^b s_tt dy
    exact_rx = -(LA * (B_OUT - A_IN) + LB * (1 / A_IN - 1 / B_OUT))
    assert abs(v["Rx"] / exact_rx - 1) < 1e-9


def test_lame_p1_vs_p2_convergence(meshdir):
    """Strain energy (= half the work of the pressure) converges at ~h^2 (P1) and ~h^4 (P2)."""
    U_exact = 0.5 * P_IN * _lame_ur(A_IN) * (math.pi / 2) * A_IN
    errs = {}
    for order in (1, 2):
        errs[order] = [abs(_lame_plane_strain(meshdir, h, order)["U"] / U_exact - 1) for h in (0.2, 0.1)]
    rate1 = math.log2(errs[1][0] / errs[1][1])
    rate2 = math.log2(errs[2][0] / errs[2][1])
    assert 1.6 < rate1 < 2.6, errs
    assert rate2 > 3.3, errs
    assert errs[2][1] < 1e-2 * errs[1][1]


@pytest.mark.parametrize("order,tol_u,tol_s", [(1, 3e-3, 4e-2), (2, 5e-5, 5e-3)])
def test_lame_axisymmetric(meshdir, order, tol_u, tol_s):
    mesh = rect_mesh(meshdir / f"axring_{order}.msh", L=B_OUT - A_IN, H=0.5, h=0.1, order=order, x0=A_IN)
    q = [QoI(name="ur_a", type="point", field="ux", point=[A_IN, 0.25]),
         QoI(name="stt_a", type="point", field="stt", point=[A_IN, 0.25]),
         QoI(name="srr", type="point", field="sxx", point=[1.5, 0.25]),
         QoI(name="szz", type="point", field="syy", point=[1.5, 0.25]),
         QoI(name="Fz", type="reaction", group="bottom", component=1)]
    bcs = [BC(type="symmetry", group="bottom"), BC(type="symmetry", group="top"),
           BC(type="pressure", group="left", value=P_IN)]
    v = solve(_case(mesh, formulation="axisymmetric", order=order, bcs=bcs, qois=q)).evaluate_all()
    assert abs(v["ur_a"] / _lame_ur(A_IN) - 1) < tol_u
    assert abs(v["stt_a"] / _lame_stt(A_IN) - 1) < tol_s
    assert abs(v["srr"] / _lame_srr(1.5) - 1) < tol_s
    assert abs(v["szz"] / (2 * NU * LA) - 1) < tol_s
    # full-360 axial reaction on z = 0: -s_zz * pi (b^2 - a^2)
    assert abs(v["Fz"] / (-2 * NU * LA * math.pi * (B_OUT ** 2 - A_IN ** 2)) - 1) < 1e-6


# --------------------------------------------------------------------------- equilibrium / loads


@pytest.mark.parametrize("formulation,thickness", [("plane_stress", 0.1), ("plane_strain", 2.0)])
def test_reaction_equals_applied_force_2d(meshdir, formulation, thickness):
    mesh = rect_mesh(meshdir / "eq2d.msh", L=2.0, H=1.0, h=0.25, order=2)
    F = [3.0, -1.5]
    q = [QoI(name="Rx", type="reaction", group="left", component=0),
         QoI(name="Ry", type="reaction", group="left", component=1),
         QoI(name="int_sxx", type="integral", field="sxx", group="right")]
    r = solve(_case(mesh, formulation=formulation, order=2, thickness=thickness,
                    bcs=[BC(type="fixed", group="left"), BC(type="force", group="right", value=F)], qois=q))
    v = r.evaluate_all()
    assert abs(v["Rx"] + F[0]) < 1e-9 and abs(v["Ry"] + F[1]) < 1e-9
    assert abs(v["int_sxx"] / F[0] - 1) < 0.02  # recovered traction resultant (incl. thickness)


def test_reaction_equals_applied_force_3d_and_lowdim_dirichlet(meshdir):
    mesh = box_mesh(meshdir / "eq3d.msh", h=0.4, order=2)
    F = [0.0, 0.0, -5.0]
    bcs = [BC(type="fixed", group="x0"), BC(type="force", group="x1", value=F),
           BC(type="displacement", group="edge_z", components=[0], value=0.0)]  # curve group (redundant)
    q = [QoI(name="Rz", type="reaction", group="x0", component=2), QoI(name="U", type="strain_energy"),
         QoI(name="uz_tip", type="min", field="uz", group="x1")]
    r = solve(_case(mesh, formulation="3d", order=2, bcs=bcs, qois=q))
    v = r.evaluate_all()
    assert abs(v["Rz"] - 5.0) < 1e-9
    # strain energy = half the external work = 0.5 * sum(f . u)
    assert abs(v["U"] - 0.5 * float(r.f @ r.x)) < 1e-12 * abs(v["U"])
    assert v["uz_tip"] < 0


def test_pressure_resultant_and_point_bc(meshdir):
    mesh = rect_mesh(meshdir / "pres.msh", L=2.0, H=1.0, h=0.25, order=2)
    bcs = [BC(type="displacement", group="bottom", components=[1], value=0.0),
           BC(type="fixed", point=[0.0, 0.0]), BC(type="pressure", group="top", value="3*x")]
    q = [QoI(name="Ry", type="reaction", group="bottom", component=1)]
    v = solve(_case(mesh, formulation="plane_stress", order=2, bcs=bcs, qois=q, thickness=0.5)).evaluate_all()
    # pressure 3x on the top pushes down: total = 0.5 * int_0^2 3x dx = 3; bottom supports push up
    assert abs(v["Ry"] - 3.0) < 1e-9


def test_body_force_reaction(meshdir):
    mesh = rect_mesh(meshdir / "body.msh", h=0.25, order=2)
    r = solve(_case(mesh, formulation="plane_stress", order=2, thickness=0.3, bcs=[BC(type="fixed", group="top")],
                    loads=[Load(type="body_force", value=[0.0, "-7.85*(1 + 0*x)"])],
                    qois=[QoI(name="Ry", type="reaction", group="top", component=1)]))
    assert abs(r.evaluate_all()["Ry"] - 7.85 * 1.0 * 0.3) < 1e-9


def test_two_materials_in_series(meshdir):
    mesh = two_material_mesh(meshdir / "twomat.msh")
    E1, E2, F = 1000.0, 3000.0, 2.0
    case = _case(mesh, formulation="plane_stress", order=2,
                 materials={"a": Material(E=E1, nu=0.0), "b": Material(E=E2, nu=0.0)},
                 regions=[Region("mat_a", "a"), Region("mat_b", "b")],
                 bcs=[BC(type="displacement", group="left", components=[0], value=0.0),
                      BC(type="displacement", group="p00", components=[1], value=0.0),
                      BC(type="force", group="right", value=[F, 0.0])],
                 qois=[QoI(name="u_end", type="max", field="ux", group="right"),
                       QoI(name="sxx_mean", type="mean", field="sxx", group="mat_b")])
    v = solve(case).evaluate_all()
    assert abs(v["u_end"] / (F / E1 + F / E2) - 1) < 1e-9
    assert abs(v["sxx_mean"] / F - 1) < 1e-9


# --------------------------------------------------------------------------- thermal


@pytest.mark.parametrize("formulation,x0", [("plane_stress", 0.0), ("plane_strain", 0.0),
                                            ("axisymmetric", 0.0), ("axisymmetric", 0.5), ("3d", 0.0)])
@pytest.mark.parametrize("order", [1, 2])
def test_free_thermal_expansion(meshdir, formulation, x0, order):
    alpha, T_ref, T = 1e-3, 20.0, 70.0
    if formulation == "3d":
        mesh = box_mesh(meshdir / "fte3.msh", h=0.5, order=2, distort=0.2)
        bcs = [BC(type="fixed", group="p000"), BC(type="displacement", group="p100", components=[1, 2], value=0.0),
               BC(type="displacement", group="p010", components=[2], value=0.0)]
    else:
        mesh = rect_mesh(meshdir / f"fte2_{x0}.msh", h=0.25, order=2, distort=0.2, x0=x0)
        bcs = [BC(type="displacement", group="p00", components=[1], value=0.0)]
        if formulation != "axisymmetric":
            bcs += [BC(type="fixed", group="p00"), BC(type="displacement", group="p10", components=[1], value=0.0)]
    r = solve(_case(mesh, formulation=formulation, order=order, bcs=bcs, T_ref=T_ref,
                    materials={"m": Material(E=E, nu=NU, alpha=alpha)}, loads=[Load(type="temperature", value=T)]))
    scale = E * alpha * (T - T_ref)
    fields = ["sxx", "syy", "sxy", "syz", "sxz", "stt"] + (["szz"] if formulation != "plane_strain" else [])
    for f in fields:
        assert _maxerr(r, f, 0.0) < 1e-9 * scale, f
    if formulation == "plane_strain":
        assert _maxerr(r, "szz", -scale) < 1e-9 * scale  # s_zz = nu (sxx + syy) - E alpha dT
    X, ux = r.nodal("ux")
    fac = (1 + NU) if formulation == "plane_strain" else 1.0
    assert np.abs(ux - fac * alpha * (T - T_ref) * X[:, 0]).max() < 1e-9


def test_constrained_thermal_strip_and_expression(meshdir):
    mesh = rect_mesh(meshdir / "strip.msh", h=0.25, order=2)
    alpha, dT = 1e-3, 50.0
    bcs = [BC(type="displacement", group="left", components=[0], value=0.0),
           BC(type="displacement", group="right", components=[0], value=0.0),
           BC(type="displacement", group="p00", components=[1], value=0.0)]
    q = [QoI(name="s", type="mean", field="sxx", group="domain"), QoI(name="R", type="reaction", group="right", component=0)]
    v = solve(_case(mesh, formulation="plane_stress", order=2, bcs=bcs, loads=[Load(type="temperature", value=dT)],
                    materials={"m": Material(E=E, nu=NU, alpha=alpha)}, qois=q)).evaluate_all()
    assert abs(v["s"] / (-E * alpha * dT) - 1) < 1e-9
    assert abs(v["R"] / (-E * alpha * dT) - 1) < 1e-9
    # a linear temperature field causes no stress in a free plate
    r = solve(_case(mesh, formulation="plane_stress", order=2, materials={"m": Material(E=E, nu=NU, alpha=alpha)},
                    bcs=[BC(type="fixed", group="p00"), BC(type="displacement", group="p10", components=[1], value=0.0)],
                    loads=[Load(type="temperature", value="100*x + 30*y")]))
    assert max(_maxerr(r, f, 0.0) for f in ("sxx", "syy", "sxy")) < 1e-9


# --------------------------------------------------------------------------- heat


def test_heat_slab_with_source(meshdir):
    L, H, k, s, T0, T1 = 2.0, 1.0, 3.0, 12.0, 10.0, 40.0
    mesh = rect_mesh(meshdir / "slab.msh", L=L, H=H, h=0.25, order=2, distort=0.2)

    def T_ex(x):
        return T0 + (T1 - T0) * x / L + s * x * (L - x) / (2 * k)

    q = [QoI(name="T", type="point", field="T", point=[0.7, 0.3]), QoI(name="qx", type="point", field="qx", point=[0.7, 0.3]),
         QoI(name="Ql", type="reaction", group="left", component=0), QoI(name="Qr", type="reaction", group="right", component=0),
         QoI(name="Tmean", type="mean", field="T")]
    case = _case(mesh, physics="heat", formulation="2d", order=2, thickness=0.5, qois=q,
                 bcs=[BC(type="temperature", group="left", value=T0), BC(type="temperature", group="right", value=T1)],
                 loads=[Load(type="source", value=s)])
    r = solve(case)
    v = r.evaluate_all()
    X, T = r.nodal("T")
    assert np.abs(T - T_ex(X[:, 0])).max() < 1e-9          # P2 is exact for a quadratic profile
    assert abs(v["T"] - T_ex(0.7)) < 1e-9
    assert abs(v["qx"] - (-k * ((T1 - T0) / L + s * (L - 2 * 0.7) / (2 * k)))) < 1e-8
    assert abs(v["Ql"] + v["Qr"] + s * L * H * 0.5) < 1e-9   # heat in through Dirichlet = -generated
    assert abs(v["Tmean"] - (T0 + T1) / 2 - s * L ** 2 / (12 * k)) < 1e-9
    assert r.nodal("q_mag")[1].min() >= 0


@pytest.mark.parametrize("order", [1, 2])
def test_heat_flux_and_convection(meshdir, order):
    L, k, q0, h, T_inf = 2.0, 3.0, 5.0, 2.0, 15.0
    mesh = rect_mesh(meshdir / "slab.msh", L=L, H=1.0, h=0.25, order=2, distort=0.2)
    r = solve(_case(mesh, physics="heat", formulation="2d", order=order,
                    bcs=[BC(type="flux", group="left", value=q0), BC(type="convection", group="right", h=h, T_inf=T_inf)]))
    X, T = r.nodal("T")
    assert np.abs(T - (T_inf + q0 / h + q0 / k * (L - X[:, 0]))).max() < 1e-9


@pytest.mark.parametrize("formulation", ["axisymmetric", "3d"])
def test_heat_axisymmetric_and_3d(meshdir, formulation):
    k, Ta, Tb = 3.0, 100.0, 0.0
    if formulation == "3d":
        mesh = box_mesh(meshdir / "heatbox.msh", h=0.35, order=2)
        r = solve(_case(mesh, physics="heat", formulation="3d", order=2,
                        bcs=[BC(type="temperature", group="x0", value=Ta), BC(type="temperature", group="x1", value="0*x")],
                        qois=[QoI(name="Q", type="reaction", group="x0", component=0)]))
        X, T = r.nodal("T")
        assert np.abs(T - Ta * (1 - X[:, 0])).max() < 1e-9
        assert abs(r.evaluate_all()["Q"] - k * Ta) < 1e-9
        assert abs(r.probe("qx", [[0.5, 0.5, 0.5]])[0] - k * Ta) < 1e-9
        return
    a, b = 1.0, 2.0
    mesh = rect_mesh(meshdir / "axheat.msh", L=b - a, H=0.5, h=0.125, order=2, x0=a)
    r = solve(_case(mesh, physics="heat", formulation="axisymmetric", order=2,
                    bcs=[BC(type="temperature", group="left", value=Ta), BC(type="temperature", group="right", value=Tb)],
                    qois=[QoI(name="T", type="point", field="T", point=[1.5, 0.25]),
                          QoI(name="Q", type="reaction", group="left", component=0)]))
    v = r.evaluate_all()
    assert abs(v["T"] - (Ta + (Tb - Ta) * math.log(1.5 / a) / math.log(b / a))) < 1e-3
    assert abs(v["Q"] / (2 * math.pi * k * 0.5 * (Ta - Tb) / math.log(b / a)) - 1) < 1e-5


# --------------------------------------------------------------------------- API behaviour


def test_info_probe_errors_vtu_cli(meshdir, tmp_path):
    mesh = rect_mesh(meshdir / "api.msh", L=2.0, H=1.0, h=0.25, order=2)
    case = _case(mesh, formulation="plane_stress", order=2,
                 bcs=[BC(type="fixed", group="left"), BC(type="traction", group="right", value=[1.0, 0.0])],
                 qois=[QoI(name="ux_max", type="max", field="ux"), QoI(name="vm", type="max_abs", field="von_mises")])
    r = solve(case)
    i = r.info
    assert i.backend == "skfem" and i.dim == 2 and i.n_dofs == 2 * i.n_nodes
    assert abs(i.measure - 2.0) < 1e-12 and abs(i.h_rep - (2.0 / i.n_cells) ** 0.5) < 1e-12
    assert any("linear_solver=" in w for w in i.warnings)
    # probes on the boundary and marginally outside evaluate; far outside raises
    assert np.isfinite(r.probe("ux", [[2.0, 0.5], [2.0 + 1e-6, 0.5], [0.0, 0.0]])).all()
    with pytest.raises(ValueError):
        r.probe("ux", [[3.0, 0.5]])
    with pytest.raises(ValueError, match="available"):
        solve(_case(mesh, bcs=[BC(type="fixed", group="nope"), BC(type="fixed", group="left")]))
    with pytest.raises(CaseError):
        solve(_case(mesh, bcs=[BC(type="fixed", group="left"), BC(type="pressure", group="domain", value=1.0)]))
    vtu = r.write_vtu(tmp_path / "out.vtu")
    import meshio

    m = meshio.read(vtu)
    assert "von_mises" in m.point_data and m.cells[0].type == "triangle6"
    # CLI round trip
    from feacore.cli import main

    cpath = case.save(tmp_path / "case.json")
    out = tmp_path / "res.json"
    assert main([str(cpath), "--out", str(out), "--vtu", str(tmp_path / "cli.vtu"), "--quiet"]) == 0
    res = json.loads(out.read_text())
    assert abs(res["qois"]["ux_max"] - r.evaluate(case.qois[0])) < 1e-15
    assert (tmp_path / "cli.vtu").exists()
