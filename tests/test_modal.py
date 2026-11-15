"""Modal analysis of the fixed solver (``feacore/solver/_skfem_modal.py``, docs/solver_transient.md "Modal analysis").

Every reference is independent of the solver: Euler-Bernoulli beam theory, the exact axial modes of a bar, the
exact Rayleigh-Lamb frequencies of a simply supported strip (``feacore.problems.dynamics.exact_modes``), and the
published NAFEMS free-vibration values (FV32, FV41, FV52; sources in ``feacore.problems.dynamics.nafems_fv``).
Fast: small P2 meshes, a few seconds each. ``FEACORE_SLOW=1`` adds the 3D plate and finer studies.
"""

from __future__ import annotations

import math
import os
from contextlib import contextmanager

import numpy as np
import pytest

gmsh = pytest.importorskip("gmsh")

from feacore.case import BC, Analysis, Case, Load, Material, QoI  # noqa: E402
from feacore.problems.dynamics.exact_modes import strip_frequencies  # noqa: E402
from feacore.problems.registry import discover, get  # noqa: E402
from feacore.solver import solve  # noqa: E402

slow = pytest.mark.skipif(not os.environ.get("FEACORE_SLOW"), reason="slow; set FEACORE_SLOW=1")
E, NU, RHO = 2e11, 0.3, 8000.0


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


def rect(path, L, H, h, y0=0.0, order=2):
    """A rectangle [0, L] x [y0, y0 + H] with edge groups left / right / bottom / top and surface 'domain'."""
    with _session("rect"):
        gmsh.model.occ.addRectangle(0, y0, 0, L, H)
        gmsh.model.occ.synchronize()
        e = 1e-9
        names = {"left": [], "right": [], "bottom": [], "top": []}
        for _, t in gmsh.model.getEntities(1):
            x, y, _ = gmsh.model.occ.getCenterOfMass(1, t)
            key = "left" if abs(x) < e else "right" if abs(x - L) < e else "bottom" if abs(y - y0) < e else "top"
            names[key].append(t)
        for n, tg in names.items():
            gmsh.model.addPhysicalGroup(1, tg, name=n)
        gmsh.model.addPhysicalGroup(2, [t for _, t in gmsh.model.getEntities(2)], name="domain")
        gmsh.option.setNumber("Mesh.MeshSizeMin", h)
        gmsh.option.setNumber("Mesh.MeshSizeMax", h)
        gmsh.model.mesh.generate(2)
        if order == 2:
            gmsh.model.mesh.setOrder(2)
        gmsh.write(str(path))
    return str(path)


def box(path, L, B, H, h):
    with _session("box"):
        gmsh.model.occ.addBox(0, 0, 0, L, B, H)
        gmsh.model.occ.synchronize()
        e = 1e-9
        for _, t in gmsh.model.getEntities(2):
            x, _, _ = gmsh.model.occ.getCenterOfMass(2, t)
            if abs(x) < e:
                gmsh.model.addPhysicalGroup(2, [t], name="x0")
        gmsh.model.addPhysicalGroup(3, [t for _, t in gmsh.model.getEntities(3)], name="solid")
        gmsh.option.setNumber("Mesh.MeshSizeMin", h)
        gmsh.option.setNumber("Mesh.MeshSizeMax", h)
        gmsh.model.mesh.generate(3)
        gmsh.model.mesh.setOrder(2)
        gmsh.write(str(path))
    return str(path)


def modal(mesh, bcs, n=6, form="plane_stress", thickness=1.0, nu=NU, shift=None, loads=(), qois=(), **opts):
    mo = {"n_modes": n}
    if shift is not None:
        mo["shift"] = shift
    return Case(mesh=mesh, analysis=Analysis(formulation=form, order=2, thickness=thickness,
                                             options={"modal": mo, **opts}),
                materials={"s": Material(E=E, nu=nu, rho=RHO)}, bcs=list(bcs), loads=list(loads), qois=list(qois))


def eb_frequency(beta_L, L, E_, I, rho, A):
    """Euler-Bernoulli: f = (beta L)^2 / (2 pi L^2) sqrt(E I / (rho A))."""
    return beta_L**2 / (2 * math.pi * L**2) * math.sqrt(E_ * I / (rho * A))


CANTILEVER_BL = (1.875104069, 4.694091133, 7.854757438)     # roots of 1 + cos b cosh b = 0
FREE_FREE_BL = (4.730040745, 7.853204624, 10.99560784)       # roots of 1 - cos b cosh b = 0


@pytest.fixture(scope="module")
def tmp(tmp_path_factory):
    return tmp_path_factory.mktemp("modal")


# --------------------------------------------------------------------------- beams against Euler-Bernoulli


def test_cantilever_against_euler_bernoulli(tmp):
    """A plane-stress cantilever, L / H = 10, 20, 40, four P2 cells through the depth. The solid has shear
    deformation and rotary inertia (Timoshenko effects, ~ (H / L)^2), so it lies BELOW beam theory and the gap
    shrinks with the slenderness. Measured f1: -0.65 %, -0.13 %, -0.017 %; f2: -4.9 %, -1.27 %, -0.31 % (a ratio
    of 3.9 and 4.1 per doubling of L / H). At L / H = 80 f1 is +0.004 %: the mesh error takes over."""
    H, t = 0.05, 0.01
    gaps = []
    for slender in (10, 20, 40):
        L = slender * H
        r = solve(modal(rect(tmp / f"cant{slender}.msh", L, H, H / 4), [BC(type="fixed", group="left")], n=2,
                        thickness=t))
        assert r.rigid_modes == 0
        eb = np.array([eb_frequency(b, L, E, t * H**3 / 12, RHO, t * H) for b in CANTILEVER_BL[:2]])
        gaps.append(r.frequencies / eb - 1)
    gaps = np.array(gaps)
    assert np.all(gaps < 0), gaps                       # the solid is softer than the beam
    ratio = gaps[:-1, 1] / gaps[1:, 1]
    assert np.all((ratio > 3.3) & (ratio < 5.0)), ratio  # ~ (H / L)^2
    assert abs(gaps[2, 0]) < 3e-4 and abs(gaps[0, 0]) < 0.008


def test_simply_supported_strip_tends_to_euler_bernoulli(tmp):
    """The sliding simple support (u_y = 0 on the end edges) gives the exact Rayleigh-Lamb A0 standing wave;
    for growing slenderness it tends to the pinned-pinned Euler-Bernoulli beam (plane strain: E / (1 - nu^2))."""
    t = 0.1
    gaps = []
    for slender in (10, 20):
        L = slender * t
        r = solve(modal(rect(tmp / f"ss{slender}.msh", L, t, t / 4, y0=-t / 2),
                        [BC(type="displacement", group=g, components=[1], value=0.0) for g in ("left", "right")],
                        n=2, form="plane_strain"))
        exact = strip_frequencies(L, t, E, NU, RHO, 1)[0][0]
        assert r.frequencies[0] == pytest.approx(exact, rel=2e-4)
        eb = eb_frequency(math.pi, L, E / (1 - NU**2), t**3 / 12, RHO, t)
        gaps.append(r.frequencies[0] / eb - 1)
        assert r.rigid_modes == 1                       # the axial translation is free
    assert gaps[0] < gaps[1] < 0
    assert abs(gaps[1]) < abs(gaps[0]) / 3            # ~ (t / L)^2


def test_free_free_beam_and_rigid_modes_2d(tmp):
    """Free-free plane-stress bar: three rigid-body modes, the lowest bending modes near Euler-Bernoulli and the
    first axial mode at c / (2 L) (nu = 0 makes the axial mode exactly one-dimensional)."""
    L, H = 1.0, 0.05
    r = solve(modal(rect(tmp / "free2d.msh", L, H, H / 2), [], n=6, nu=0.0))
    assert r.rigid_modes == 3
    assert any("3 rigid-body modes" in w for w in r.info.warnings)
    eb = [eb_frequency(b, L, E, H**3 / 12, RHO, H) for b in FREE_FREE_BL[:2]]
    assert r.frequencies[0] == pytest.approx(eb[0], rel=0.01)
    assert r.frequencies[1] == pytest.approx(eb[1], rel=0.03)
    c = math.sqrt(E / RHO)
    assert np.min(np.abs(r.frequencies / (c / (2 * L)) - 1)) < 1e-5
    # the elastic modes are M-orthogonal to the translations: no effective mass
    assert np.abs(r.effective_mass).max() < 1e-10


def test_free_bar_3d_six_rigid_modes_and_axial(tmp):
    L, B = 0.4, 0.04
    r = solve(modal(box(tmp / "free3d.msh", L, B, B, B / 2), [], n=8, form="3d", nu=0.0))
    assert r.rigid_modes == 6
    c = math.sqrt(E / RHO)
    assert np.min(np.abs(r.frequencies / (c / (2 * L)) - 1)) < 1e-4
    assert r.total_mass == pytest.approx(RHO * L * B * B, rel=1e-12)


# --------------------------------------------------------------------------- the exact strip: convergence order


def test_strip_eigenvalue_order(tmp):
    """Eigenvalues of P2 elements converge with h^(2p) = h^4 (observed 3.8-4.6 on unstructured meshes)."""
    L, t = 10.0, 1.0
    exact = np.array([f for f, _, _ in strip_frequencies(L, t, E, NU, RHO, 3)])
    errs = []
    for h in (0.5, 0.25):
        r = solve(modal(rect(tmp / f"strip{h}.msh", L, t, h, y0=-t / 2),
                        [BC(type="displacement", group=g, components=[1], value=0.0) for g in ("left", "right")],
                        n=3, form="plane_strain"))
        errs.append(r.frequencies / exact - 1)
    assert np.all(errs[0] > 0) and np.all(errs[1] > 0)          # consistent mass: from above
    p = np.log2(errs[0] / errs[1])
    assert np.all((p > 3.3) & (p < 5.2)), p
    assert np.abs(errs[1]).max() < 1e-4


# --------------------------------------------------------------------------- mass, orthonormality, effective mass


def test_mass_orthonormality_effective_mass(tmp):
    L, H, t = 1.0, 0.05, 0.01
    mesh = rect(tmp / "cant.msh", L, H, H / 2)
    r = solve(modal(mesh, [BC(type="fixed", group="left")], n=30, thickness=t,
                    qois=[QoI("m1y", "effective_mass", mode=1, component=1), QoI("f1", "natural_frequency", mode=1)]))
    m = RHO * L * H * t
    assert r.total_mass == pytest.approx(m, rel=1e-12)
    md = r.model
    for k in range(2):                                 # a rigid translation carries the whole mass
        one = np.zeros(md.N)
        one[md.split[k]] = 1.0
        assert one @ (r.M @ one) == pytest.approx(m, rel=1e-12)
    G = r.modes.T @ (r.M @ r.modes)
    assert np.abs(G - np.eye(len(G))).max() < 1e-9
    K_ = r.modes.T @ (r.K @ r.modes)
    assert np.allclose(np.diag(K_), (2 * math.pi * r.frequencies) ** 2, rtol=1e-9)
    assert r.residuals.max() < 1e-8
    # cantilever: the first bending mode moves ~61 % of the mass (Euler-Bernoulli: 0.613); 30 modes collect
    # most of the mass in y (the rest sits near the clamp)
    assert r.effective_mass[0, 1] == pytest.approx(0.613, abs=0.005)
    s = r.effective_mass.sum(axis=0)
    assert np.all(s <= 1 + 1e-12) and s[1] > 0.9
    v = r.evaluate_all()
    assert v["m1y"] == r.effective_mass[0, 1] and v["f1"] == r.frequencies[0]


def test_axisymmetric_mass_and_fv41_effective_mass(tmp):
    """Axisymmetric mass carries 2 pi r: the total mass of a tube is rho pi (ro^2 - ri^2) L."""
    p = get_problem("nafems_fv41")
    mesh = p.mesh(tmp / "fv41.msh", 0.4)
    r = solve(p.case(mesh))
    assert r.total_mass == pytest.approx(RHO * math.pi * (2.2**2 - 1.8**2) * 10.0, rel=1e-4)   # curved? no: exact
    assert r.rigid_modes == 1
    assert np.all(np.isnan(r.effective_mass[:, 0]))


# --------------------------------------------------------------------------- NAFEMS


def get_problem(pid):
    discover()
    return get(pid)


@pytest.mark.parametrize("pid", ["nafems_fv32", "nafems_fv41"])
def test_nafems_published_frequencies(tmp, pid):
    """FV32 (plane stress) and FV41 (axisymmetric, free) at the reference mesh h0: every published frequency
    within its tolerance (0.5 %); measured errors -0.014 % .. +0.34 % at h0."""
    p = get_problem(pid)
    r = solve(p.case(p.mesh(tmp / f"{pid}.msh", p.params["h0"])))
    v = r.evaluate_all()
    for ref in p.references():
        assert v[ref.qoi] == pytest.approx(ref.value, rel=ref.tolerance), (ref.qoi, v[ref.qoi], ref.value)


def test_nafems_fv32_converges_to_published(tmp):
    """Refined (h = 0.25 m) every FV32 frequency is within 0.05 % of the published value."""
    p = get_problem("nafems_fv32")
    r = solve(p.case(p.mesh(tmp / "fv32f.msh", 0.25)))
    ref = np.array(p.FREQUENCIES)
    assert np.abs(r.frequencies / ref - 1).max() < 5e-4, r.frequencies / ref - 1


def test_nafems_fv52_coarse(tmp):
    """FV52 on the coarse h = 2 m mesh: three rigid-body modes (in-plane translations, rotation about z), the
    support-independent in-plane mode f5 at the published 193.59 Hz (0.5 %), the bending modes within the 5 %
    of a thick-plate reference (they have no mesh-converged answer, see the problem module)."""
    p = get_problem("nafems_fv52")
    r = solve(p.case(p.mesh(tmp / "fv52.msh", 2.0)))
    assert r.rigid_modes == 3
    v = r.evaluate_all()
    for ref in p.references():
        assert v[ref.qoi] == pytest.approx(ref.value, rel=ref.tolerance), (ref.qoi, v[ref.qoi], ref.value)


@slow
def test_ss_thick_plate_3d(tmp):
    p = get_problem("ss_thick_plate_modes")
    r = solve(p.case(p.mesh(tmp / "plate.msh", 1.0)))
    v = r.evaluate_all()
    for ref in p.references():
        assert v[ref.qoi] == pytest.approx(ref.value, rel=ref.tolerance), (ref.qoi, v[ref.qoi], ref.value)


# --------------------------------------------------------------------------- options, warnings, outputs


def test_shift_dense_and_warnings(tmp):
    L, H = 1.0, 0.05
    mesh = rect(tmp / "beam.msh", L, H, H / 2)
    base = solve(modal(mesh, [BC(type="fixed", group="left")], n=6))
    above = solve(modal(mesh, [BC(type="fixed", group="left")], n=3, shift=float(base.frequencies[2]) * 1.01))
    assert np.allclose(above.frequencies, base.frequencies[3:6], rtol=1e-9)
    # loads ignored, a non-zero prescribed displacement acts as zero, an unknown option warns
    r = solve(modal(mesh, [BC(type="fixed", group="left"), BC(type="displacement", group="right", components=[1],
                                                                value=1e-3)],
                    n=2, loads=[Load(type="body_force", value=[0.0, -1e4])], bogus_option=1))
    w = " | ".join(r.info.warnings)
    assert "loads" in w and "act as zero" in w and "unknown analysis options" in w
    # the dense LAPACK path (tiny model) and the Lanczos path agree
    tiny = rect(tmp / "tiny.msh", L, H, H)
    a = solve(modal(tiny, [BC(type="fixed", group="left")], n=3))
    assert "dense LAPACK" in a.info.warnings[-1] or "Lanczos" in a.info.warnings[-1]
    from feacore.solver import _skfem_modal as sm
    old = sm.DENSE_BELOW
    try:
        sm.DENSE_BELOW = 10**9
        b = solve(modal(tiny, [BC(type="fixed", group="left")], n=3))
    finally:
        sm.DENSE_BELOW = old
    assert "dense LAPACK" in b.info.warnings[-1]
    assert np.allclose(a.frequencies, b.frequencies, rtol=1e-9)


def test_shapes_probe_vtu(tmp):
    L, H = 1.0, 0.05
    r = solve(modal(rect(tmp / "beam2.msh", L, H, H / 2), [BC(type="fixed", group="left")], n=2))
    x, d = r.shape(1)
    assert x.shape == d.shape == (r.model.sb.N, 2)
    tip = r.probe("uy", [[L, H / 2]])[0]
    assert abs(tip) == pytest.approx(np.abs(d[:, 1]).max(), rel=0.02)   # the cantilever's first mode peaks at the tip
    _, umag = r.nodal("u_mag", 2)
    assert umag.max() > 0
    with pytest.raises(Exception):
        r.probe("sxx", [[0.5, 0.0]])
    with pytest.raises(ValueError):
        r.shape(3)
    p = r.write_vtu(tmp / "modes.vtu")
    import meshio
    m = meshio.read(p)
    assert "mode_1" in m.point_data and "mode_2" in m.point_data
    s = r.summary()
    assert s["modal"]["rigid_modes"] == 0 and len(s["modal"]["frequencies"]) == 2
