"""Opt-in elastoplastic (J2) load-stepped analysis with the CalculiX backend, against closed forms:

* uniaxial bar (C3D10), elastic-perfectly plastic and linear hardening, displacement and force control:
  stress caps at sy (or follows the hardening line), PEEQ = eps - sy / E (/ (1 + H / E));
* thick-walled sphere under internal pressure (CAX6), elastic-perfectly plastic: elastic limit pressure
  and elastic-plastic boundary radius c(p) from the closed-form solution (Hill 1950, ch. V);
* reactions balance the applied load at every increment.

Small meshes; the whole file runs in well under a minute. The schema / skfem tests need no ccx.
"""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import numpy as np
import pytest

from feacore.case import BC, Analysis, Case, Material, QoI
from feacore.solver import solve

needs_ccx = pytest.mark.skipif(shutil.which("ccx") is None, reason="CalculiX (ccx) not installed")

E, NU, SY = 210000.0, 0.3, 355.0
L, W = 2.0, 0.5                     # bar length, square cross-section side


def _box(path, h=0.25):
    import gmsh
    from feacore.geometry.gmsh_tools import add_group, gmsh_session, in_box, set_uniform_size
    with gmsh_session("box"):
        v = gmsh.model.occ.addBox(0, 0, 0, L, W, W)
        gmsh.model.occ.synchronize()
        add_group(3, [v], "body")
        for name, kw in (("left", {"xmax": 0}), ("right", {"xmin": L}), ("y0", {"ymax": 0}), ("z0", {"zmax": 0})):
            add_group(2, in_box(2, **kw), name)
        set_uniform_size(h)
        gmsh.model.mesh.generate(3)
        gmsh.model.mesh.setOrder(2)
        gmsh.write(str(path))
    return str(path)


def _quarter_annulus(path, a, b, h):
    """r-z section of a hemispherical shell a <= sqrt(x^2 + y^2) <= b, x, y >= 0 (axis x = 0)."""
    import gmsh
    from feacore.geometry.gmsh_tools import add_group, gmsh_session, set_uniform_size
    with gmsh_session("ann"):
        d1, d2 = gmsh.model.occ.addDisk(0, 0, 0, b, b), gmsh.model.occ.addDisk(0, 0, 0, a, a)
        ring, _ = gmsh.model.occ.cut([(2, d1)], [(2, d2)])
        q, _ = gmsh.model.occ.intersect(ring, [(2, gmsh.model.occ.addRectangle(0, 0, 0, 2 * b, 2 * b))])
        gmsh.model.occ.synchronize()
        add_group(2, [t for _, t in q], "body")
        g = {"inner": [], "outer": [], "equator": [], "axis": []}
        for _, c in gmsh.model.getEntities(1):
            x, y, _ = gmsh.model.occ.getCenterOfMass(1, c)
            g["equator" if abs(y) < 1e-9 else "axis" if abs(x) < 1e-9 else
              "inner" if math.hypot(x, y) < 0.5 * (a + b) else "outer"].append(c)
        for k, v in g.items():
            add_group(1, v, k)
        set_uniform_size(h)
        gmsh.model.mesh.generate(2)
        gmsh.model.mesh.setOrder(2)
        gmsh.write(str(path))
    return str(path)


@pytest.fixture(scope="module")
def meshes(tmp_path_factory):
    d = tmp_path_factory.mktemp("ccx_nl")
    return {"box": _box(d / "box.msh"), "sphere": _quarter_annulus(d / "sphere.msh", 1.0, 2.0, 0.05)}


def _plastic(H=0.0):
    p = {"model": "j2_isotropic", "yield_stress": SY}
    if H:
        p["hardening"] = [[0.0, SY], [1.0, SY + H]]
    return p


def _bar(mesh, load, H=0.0, n=8, **opts):
    """Quarter-symmetric bar: symmetry on x = 0, y = 0, z = 0; ``load`` on the face x = L."""
    return Case(mesh=mesh, analysis=Analysis("elasticity", "3d", order=2,
                                             options={"nonlinear": True, "increments": n, **opts}),
                materials={"steel": Material(E=E, nu=NU, plasticity=_plastic(H))},
                bcs=[BC("symmetry", group="left"), BC("symmetry", group="y0"), BC("symmetry", group="z0"), load],
                qois=[QoI("R", "reaction", group="left", component=0),
                      QoI("sxx", "mean", field="sxx", group="body")])


# ----------------------------------------------------------------------------- schema (no ccx)
def test_schema_backward_compatible_and_validated(tmp_path):
    lin = Case(mesh="m.msh", analysis=Analysis(), materials={"m": Material(E=1.0, nu=0.3)},
               bcs=[BC("fixed", group="a")])
    d = lin.to_dict()
    assert "plasticity" not in d["materials"]["m"] and not lin.analysis.nonlinear
    assert Case.from_dict(d).to_dict() == d
    nl = Case(mesh="m.msh", analysis=Analysis(options={"nonlinear": True, "increments": 5}),
              materials={"m": Material(E=1.0, nu=0.3, plasticity=_plastic(1000.0))}, bcs=[BC("fixed", group="a")])
    assert nl.validate() == [] and Case.from_dict(nl.to_dict()).materials["m"].plasticity == _plastic(1000.0)
    nl.analysis.options = {}
    assert any("nonlinear = true" in e for e in nl.validate())
    bad = {"increments": 0, "load_factor_max": -1.0, "nonlinear": True}
    nl.analysis.options = bad
    assert len(nl.validate()) == 2
    nl.analysis.options = {"nonlinear": True}
    for p, msg in ((dict(_plastic(), model="tresca"), "model"), (dict(_plastic(), yield_stress=0), "yield_stress"),
                   (dict(_plastic(), hardening=[[0, SY], [0.1, SY - 1]]), "soften"),
                   (dict(_plastic(), hardening=[[0, SY + 5]]), "start at yield_stress"),
                   (dict(_plastic(), hardening=[[0, SY], [0.0, SY + 1]]), "increase")):
        nl.materials["m"].plasticity = p
        assert any(msg in e for e in nl.validate()), (p, nl.validate())


def test_skfem_refuses_nonlinear():
    c = Case(mesh="missing.msh", analysis=Analysis(options={"nonlinear": True}),
             materials={"m": Material(E=E, nu=NU, plasticity=_plastic())}, bcs=[BC("fixed", group="a")])
    with pytest.raises(NotImplementedError, match="skfem backend: nonlinear"):
        solve(c, backend="skfem")


# ----------------------------------------------------------------------------- uniaxial bar
@needs_ccx
@pytest.mark.parametrize("H", [0.0, 2000.0])
def test_uniaxial_bar_displacement_control(meshes, H):
    """sxx = min(E eps, sy + H eps_p), PEEQ = (eps - sy / E) / (1 + H / E) at every increment."""
    eps_max = 4.0 * SY / E
    r = solve(_bar(meshes["box"], BC("displacement", group="right", components=[0], value=eps_max * L), H=H),
              backend="ccx")
    assert r.converged and len(r.increments) == 8
    for i, inc in enumerate(r.increments):
        q = r.increment_qois(i)
        eps = inc.lam * eps_max
        ep = max(eps - SY / E, 0.0) / (1.0 + H / E)
        sig = E * eps if ep == 0 else SY + H * ep
        assert abs(q["qois"]["sxx"] - sig) < 2e-5 * SY        # ccx nodal stress carries ~1e-5
        assert abs(q["qois"]["R"] + sig * W * W) < 1e-5 * SY * W * W
        assert abs(q["max_peeq_ip"] - ep) < 1e-6 * SY / E and abs(q["max_peeq"] - ep) < 2e-5 * ep + 1e-9
        assert q["plastic_volume_fraction"] == pytest.approx(1.0 if ep > 0 else 0.0)
        assert q["imbalance_rel"] < 1e-5
    assert r.increments[1].PEEQ.max() == 0.0 and r.increments[2].PEEQ.min() > 0   # yield at lambda = 1/4


@needs_ccx
def test_uniaxial_bar_force_control_hardening(meshes):
    """Load control past yield with hardening: sxx = F / A, eps_p = (sxx - sy) / H, reactions = -F."""
    H, F = 2000.0, 1.2 * SY * W * W
    r = solve(_bar(meshes["box"], BC("force", group="right", value=[F, 0.0, 0.0]), H=H, n=6), backend="ccx")
    assert r.converged
    for i, inc in enumerate(r.increments):
        q = r.increment_qois(i)
        sig = inc.lam * F / (W * W)
        assert abs(q["qois"]["R"] + inc.lam * F) < 1e-6 * F
        assert abs(q["applied"][0] - inc.lam * F) < 1e-9 * F and q["imbalance_rel"] < 1e-5
        assert abs(q["max_peeq_ip"] - max(sig - SY, 0.0) / H) < 1e-6
    ux = r.probe("ux", [[L, 0.1, 0.1]])[0]
    assert ux == pytest.approx((1.2 * SY / E + 0.2 * SY / H) * L, rel=1e-5)


@needs_ccx
def test_nonconvergence_keeps_converged_increments(meshes):
    """Perfectly plastic bar loaded by force beyond its limit load sy A: fixed increments diverge
    after lambda = 1 / 1.5; the converged increments are kept and the run is flagged."""
    r = solve(_bar(meshes["box"], BC("force", group="right", value=[1.5 * SY * W * W, 0.0, 0.0]), n=6),
              backend="ccx")
    assert not r.converged and r.errors
    assert 0.5 <= r.increments[-1].lam < 1.0 and any("stopped at load factor" in w for w in r.info.warnings)


@needs_ccx
def test_nlgeom_and_series_output(meshes, tmp_path):
    """NLGEOM at 0.7 % strain stays within 1 % of the small-strain answer; the series writer gives one
    .vtu per increment (+ lambda = 0), a .pvd and summary.json."""
    import meshio

    eps_max = 4.0 * SY / E
    c = _bar(meshes["box"], BC("displacement", group="right", components=[0], value=eps_max * L), H=2000.0, n=4,
             nlgeom=True)
    r = solve(c, backend="ccx")
    s = r.write_series(tmp_path / "series", extra={"set": "bar"})
    sig = SY + 2000.0 * (eps_max - SY / E) / (1 + 2000.0 / E)
    assert s["converged"] and s["set"] == "bar" and abs(s["increments"][-1]["qois"]["sxx"] / sig - 1) < 1e-2
    assert s["first_plastic_increment_lambda"] == pytest.approx(0.5)   # lambda = 0.25 is exactly at yield
    files = sorted(p.name for p in (tmp_path / "series").iterdir())
    assert files == ["inc.pvd"] + [f"inc_{k:03d}.vtu" for k in range(5)] + ["summary.json"]
    m = meshio.read(tmp_path / "series" / "inc_004.vtu")
    assert m.point_data["PEEQ"].max() > 0 and m.point_data["displacement"].shape[1] == 3
    assert json.loads((tmp_path / "series" / "summary.json").read_text())["n_increments"] == 4


# ----------------------------------------------------------------------------- thick sphere (Hill)
def _hill_p(c, a=1.0, b=2.0, sy=SY):
    """Internal pressure at which the plastic zone of a perfectly plastic thick sphere reaches radius c."""
    return 2.0 * sy * math.log(c / a) + 2.0 * sy / 3.0 * (1.0 - c ** 3 / b ** 3)


@needs_ccx
def test_thick_sphere_internal_pressure(meshes):
    """Elastic limit p_e = 2 sy / 3 (1 - a^3 / b^3); plastic radius c(p) from _hill_p (independent of nu).
    c is measured twice: from the elastic zone, where von Mises = sy (c / r)^3, at the outer surface;
    and from the plastic volume (integration points with PEEQ > 0) of the hemispherical shell."""
    from scipy.optimize import brentq

    a, b = 1.0, 2.0
    pe, pmax = _hill_p(a), _hill_p(1.6)
    c = Case(mesh=meshes["sphere"],
             analysis=Analysis("elasticity", "axisymmetric", order=2, options={"nonlinear": True, "increments": 20}),
             materials={"steel": Material(E=E, nu=NU, plasticity=_plastic())},
             bcs=[BC("symmetry", group="equator"), BC("pressure", group="inner", value=pmax)],
             qois=[QoI("vm_outer", "mean", field="von_mises", group="outer"),
                   QoI("Rz", "reaction", group="equator", component=1)])
    r = solve(c, backend="ccx")
    assert r.converged
    elastic = []
    for i, inc in enumerate(r.increments):
        q = r.increment_qois(i)
        p = inc.lam * pmax
        assert abs(q["qois"]["Rz"] + p * math.pi * a * a) < 1e-5 * p * math.pi * a * a
        assert q["imbalance_rel"] < 1e-4
        if p < pe:
            assert q["max_peeq"] == 0.0
            elastic.append(p * SY / q["max_von_mises"])          # vM(a) scales with p
            continue
        assert q["max_peeq"] > 0
        c_ex = brentq(lambda x: _hill_p(x) - p, a, b)
        c_stress = b * (q["qois"]["vm_outer"] / SY) ** (1 / 3)
        c_vol = (a ** 3 + 3.0 * q["plastic_volume"] / (2.0 * math.pi)) ** (1 / 3)
        assert abs(c_stress / c_ex - 1) < 2e-3, (p, c_stress, c_ex)
        assert abs(c_vol / c_ex - 1) < 1e-2, (p, c_vol, c_ex)
        assert q["max_von_mises"] < 1.02 * SY
    assert abs(np.mean(elastic) / pe - 1) < 3e-3          # nodal vM at the bore overshoots ~0.2 %


# ----------------------------------------------------------------------------- progressive failure (erosion)
def _notched(path, h=0.18):
    """4 x 1 x 0.25 bar with two semicircular edge notches (r = 0.2) at x = 2."""
    import gmsh
    from feacore.geometry.gmsh_tools import add_group, gmsh_session, in_box, set_uniform_size
    with gmsh_session("notch"):
        b = gmsh.model.occ.addBox(0, 0, 0, 4.0, 1.0, 0.25)
        cuts = [(3, gmsh.model.occ.addCylinder(2.0, y, -0.1, 0, 0, 0.45, 0.2)) for y in (0.0, 1.0)]
        v, _ = gmsh.model.occ.cut([(3, b)], cuts)
        gmsh.model.occ.synchronize()
        add_group(3, [t for _, t in v], "body")
        add_group(2, in_box(2, xmax=0), "left")
        add_group(2, in_box(2, xmin=4.0), "right")
        set_uniform_size(h)
        gmsh.model.mesh.generate(3)
        gmsh.model.mesh.setOrder(2)
        gmsh.write(str(path))
    return str(path)


def _failure_case(mesh, limit, value, bcs_extra=(), **fo):
    return Case(mesh=mesh, analysis=Analysis("elasticity", "3d", order=2, options={
        "nonlinear": True, "nlgeom": True, "increments": 50,
        "failure": {"criterion": "peeq", "limit": limit, "fine_increments": 400, **fo}}),
        materials={"m": Material(E=1000.0, nu=0.3, plasticity={"model": "j2_isotropic", "yield_stress": 1.0,
                                                              "hardening": [[0, 1.0], [1.0, 3.0]]})},
        bcs=[BC("fixed", group="left"), BC("displacement", group="right", components=[0, 1, 2], value=value),
             *bcs_extra])


def test_failure_option_validated():
    c = _failure_case("m.msh", 0.05, [0.1, 0, 0])
    assert c.validate() == []
    c.analysis.options["failure"] = {"criterion": "damage", "limit": 0.05}
    assert any("criterion" in e for e in c.validate())
    c.analysis.options["failure"] = {"criterion": "peeq", "limit": -1}
    assert any("limit" in e for e in c.validate())
    c.materials["m"].plasticity = None
    c.analysis.options["failure"] = {"criterion": "peeq", "limit": 0.05}
    assert any("plasticity" in e for e in c.validate())


@needs_ccx
def test_failure_restart_blocks_are_path_correct(meshes):
    """A failure run whose limit is never reached runs the ramp in restart blocks; the bar must give the
    closed-form hardening answer at every increment (small strain: NLGEOM off)."""
    eps_max, H = 4.0 * SY / E, 2000.0
    c = _bar(meshes["box"], BC("displacement", group="right", components=[0], value=eps_max * L), H=H, n=8,
             failure={"criterion": "peeq", "limit": 10.0})
    r = solve(c, backend="ccx")
    assert r.stop_reason == "end of ramp" and r.history[-1]["block"] >= 7 and r.state.sum() == 0
    for h in r.history:
        eps = h["lambda"] * eps_max
        ep = max(eps - SY / E, 0.0) / (1.0 + H / E)
        sig = E * eps if ep == 0 else SY + H * ep
        assert abs(h["force"] - sig * W * W) < 2e-5 * SY * W * W, h
        assert abs(h["max_peeq_alive"] - ep) < 1e-6
    with pytest.raises(NotImplementedError, match="FORCE|CLOAD"):
        solve(_bar(meshes["box"], BC("force", group="right", value=[1.0, 0, 0]), failure={"criterion": "peeq",
                                                                                          "limit": 1.0}),
              backend="ccx")


@needs_ccx
def test_failure_notched_bar_separates(tmp_path):
    """Notched bar pulled by bonded grips (NLGEOM): cells fail only at the notch, the bar separates and the
    force falls to ~0; one .vtu per increment with the failed flag and cell PEEQ."""
    import meshio

    c = _failure_case(_notched(tmp_path / "notch.msh"), 0.05, [0.4, 0.0, 0.0])
    r = solve(c, backend="ccx", series_dir=str(tmp_path / "series"))
    assert r.separated and r.converged, r.stop_reason
    F = np.array([h["force"] for h in r.history])
    assert F[-1] < 0.02 * F.max() and F.argmax() < len(F) - 3
    cen = r.model.X[r.model.conn[:, :4]].mean(axis=1)
    failed = r.state == 1
    assert failed.sum() > 10 and np.all(np.abs(cen[failed, 0] - 2.0) < 0.35)
    assert np.all(np.diff([h["n_failed"] for h in r.history]) >= 0)
    files = sorted((tmp_path / "series").glob("result_*.vtu"))
    assert len(files) == len(r.history)
    m = meshio.read(files[-1])
    assert set(m.cell_data) >= {"PEEQ", "failed"} and {"displacement", "von_mises"} <= set(m.point_data)
    assert int(m.cell_data["failed"][0].sum()) == int((r.state != 0).sum())


@needs_ccx
def test_failure_checkpoint_resume_continues_the_ramp(meshes, tmp_path):
    """A failure run stopped at half the ramp (end_fraction) and resumed from its checkpoint in a new
    workdir continues the restart chain: steps continue, the closed-form hardening answer holds at every
    increment of both parts, and the first run's restart file is left in place."""
    from feacore.solver._ccx_failure import load_checkpoint

    eps_max, H = 4.0 * SY / E, 2000.0

    def case(**fo):
        return _bar(meshes["box"], BC("displacement", group="right", components=[0], value=eps_max * L), H=H,
                    n=8, failure={"criterion": "peeq", "limit": 10.0, **fo})

    r1 = solve(case(end_fraction=0.5), backend="ccx", workdir=str(tmp_path / "w1"),
               checkpoint_dir=str(tmp_path / "c1"))
    ck = load_checkpoint(tmp_path / "c1")
    assert r1.stop_reason.startswith("end of ramp") and abs(ck["t"] - 0.5) < 1e-9
    assert ck["step"] == r1.history[-1]["step"] and Path(ck["rout"]).exists()
    r2 = solve(case(), backend="ccx", workdir=str(tmp_path / "w2"), resume=str(tmp_path / "c1"),
               checkpoint_dir=str(tmp_path / "c2"))
    assert r2.stop_reason == "end of ramp" and Path(ck["rout"]).exists()
    assert r2.history[0]["step"] == ck["step"] + 1 and r2.history[0]["block"] == ck["block"]
    hist = r1.history + r2.history
    assert [h["step"] for h in hist] == list(range(1, len(hist) + 1)) and abs(hist[-1]["lambda"] - 1.0) < 1e-9
    for h in hist:
        eps = h["lambda"] * eps_max
        ep = max(eps - SY / E, 0.0) / (1.0 + H / E)
        sig = E * eps if ep == 0 else SY + H * ep
        assert abs(h["force"] - sig * W * W) < 2e-5 * SY * W * W, h


@needs_ccx
def test_failure_step_policy_tail_and_sustained_stop(tmp_path):
    """Notched bar with the crush step policy: increments <= max_increment_near from failure onset until the
    force is below tail_force_ratio x peak, <= max_increment_tail after; tail flag latched; separates."""
    c = _failure_case(_notched(tmp_path / "notch.msh"), 0.05, [0.4, 0.0, 0.0], max_increment=0.002,
                      max_increment_near=0.0005, near_increments=500, tail_force_ratio=0.6,
                      max_increment_tail=0.002, tail_increments=100, stop_force_count=3)
    r = solve(c, backend="ccx")
    assert r.separated and r.converged, r.stop_reason
    lam = np.r_[0.0, [h["lambda"] for h in r.history]]
    d = np.diff(lam)
    tail = np.array([h["tail"] for h in r.history])
    failed = np.array([h["n_failed"] for h in r.history]) > 0
    assert np.all(d <= 0.002 + 1e-9)
    pre = failed & ~tail
    assert pre.any() and np.all(d[pre] <= 0.0005 + 1e-9)
    assert np.all(np.diff(tail.astype(int)) >= 0)
    F = np.array([h["force"] for h in r.history])
    if tail.any():
        assert F[np.argmax(tail) - 1] < 0.6 * F.max()


def _hex_box(path, n=(4, 2, 2)):
    """All-hex (C3D8) L x W x W box, transfinite + recombined."""
    import gmsh
    from feacore.geometry.gmsh_tools import add_group, gmsh_session, in_box
    with gmsh_session("hexbox"):
        v = gmsh.model.occ.addBox(0, 0, 0, L, W, W)
        gmsh.model.occ.synchronize()
        add_group(3, [v], "body")
        for name, kw in (("left", {"xmax": 0}), ("right", {"xmin": L}), ("y0", {"ymax": 0}), ("z0", {"zmax": 0})):
            add_group(2, in_box(2, **kw), name)
        for _, c in gmsh.model.getEntities(1):
            a, b = gmsh.model.getBoundingBox(1, c)[:3], gmsh.model.getBoundingBox(1, c)[3:]
            k = int(np.argmax(np.subtract(b, a)))
            gmsh.model.mesh.setTransfiniteCurve(c, n[k] + 1)
        for _, sf in gmsh.model.getEntities(2):
            gmsh.model.mesh.setTransfiniteSurface(sf)
            gmsh.model.mesh.setRecombine(2, sf)
        gmsh.model.mesh.setTransfiniteVolume(v)
        gmsh.model.mesh.generate(3)
        gmsh.write(str(path))
    return str(path)


@needs_ccx
def test_hex_c3d8_restart_blocks_closed_form(tmp_path):
    """C3D8 (gmsh hexahedron) through the failure driver: 8 integration points per cell, closed-form
    hardening force and PEEQ at every increment, hexahedron .vtu."""
    import meshio

    eps_max, H = 4.0 * SY / E, 2000.0
    c = _bar(_hex_box(tmp_path / "hex.msh"), BC("displacement", group="right", components=[0], value=eps_max * L),
             H=H, n=4, failure={"criterion": "peeq", "limit": 10.0})
    c.analysis.order = 1
    r = solve(c, backend="ccx", series_dir=str(tmp_path / "s"))
    assert r.model.cell_type == "hexahedron" and r.deck.etype == "C3D8"
    assert r.increments[-1].peeq_ip.shape == (r.model.n_elems, 8)
    for h in r.history:
        eps = h["lambda"] * eps_max
        ep = max(eps - SY / E, 0.0) / (1.0 + H / E)
        sig = E * eps if ep == 0 else SY + H * ep
        assert abs(h["force"] - sig * W * W) < 2e-5 * SY * W * W, h
        assert abs(h["max_peeq_alive"] - ep) < 1e-6
    assert abs(r.info.measure - L * W * W) < 1e-9
    assert meshio.read(sorted((tmp_path / "s").glob("*.vtu"))[-1]).cells[0].type == "hexahedron"


@needs_ccx
def test_hex_notched_bar_erodes_and_separates(tmp_path):
    """All-hex notched bar (gmsh subdivision) pulled to separation: erosion and the fragment topology work
    on quad faces; failed cells only at the notch."""
    import gmsh
    from feacore.geometry.gmsh_tools import add_group, gmsh_session, in_box, set_uniform_size
    p = tmp_path / "hexnotch.msh"
    with gmsh_session("notch"):
        b = gmsh.model.occ.addBox(0, 0, 0, 4.0, 1.0, 0.25)
        cuts = [(3, gmsh.model.occ.addCylinder(2.0, y, -0.1, 0, 0, 0.45, 0.2)) for y in (0.0, 1.0)]
        v, _ = gmsh.model.occ.cut([(3, b)], cuts)
        gmsh.model.occ.synchronize()
        add_group(3, [t for _, t in v], "body")
        add_group(2, in_box(2, xmax=0), "left")
        add_group(2, in_box(2, xmin=4.0), "right")
        set_uniform_size(0.3)
        gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 2)
        gmsh.model.mesh.generate(3)
        gmsh.write(str(p))
    c = _failure_case(str(p), 0.05, [0.4, 0.0, 0.0])
    c.analysis.order = 1
    r = solve(c, backend="ccx")
    assert r.model.cell_type == "hexahedron" and r.separated and r.converged, r.stop_reason
    F = np.array([h["force"] for h in r.history])
    assert F[-1] < 0.02 * F.max()
    cen = r.model.X[r.model.conn].mean(axis=1)
    failed = r.state == 1
    assert failed.sum() > 10 and np.all(np.abs(cen[failed, 0] - 2.0) < 0.35)


@needs_ccx
def test_failure_adaptive_release_matches_fixed(tmp_path):
    """Adaptive release (cap 64, x2 / /2, platen held while a backlog is released) only changes how failed cells
    are split into removals: the notched bar gives the same force-travel curve as a fixed cap of 8 with the platen
    held the same way, and separates at about the same travel."""
    mesh = _notched(tmp_path / "notch.msh")
    runs = {}
    for name, fo in (("fixed", {"max_remove": 8, "release_travel": 1e-3 / 400}),
                     ("adaptive", {"max_remove": 64, "adaptive_release": True})):
        r = solve(_failure_case(mesh, 0.05, [0.4, 0.0, 0.0], **fo), backend="ccx")
        assert r.separated and r.converged, (name, r.stop_reason)
        runs[name] = r
    lam = {k: np.array([h["lambda"] for h in r.history]) for k, r in runs.items()}
    F = {k: np.array([h["force"] for h in r.history]) for k, r in runs.items()}
    peak = F["fixed"].max()
    assert abs(F["adaptive"].max() - peak) < 1e-3 * peak
    nf = {k: np.array([h["n_failed"] for h in r.history]) for k, r in runs.items()}
    # before the first failure: the same curve
    m = nf["adaptive"] == 0
    assert np.allclose(F["adaptive"][m], np.interp(lam["adaptive"][m], lam["fixed"], F["fixed"]), rtol=1e-6)
    # crack run (platen held): the force at the same number of failed cells (last increment with that count)
    common = sorted(set(nf["fixed"][:-1]) & set(nf["adaptive"][:-1]) - {0})
    assert len(common) >= 1
    last = {k: {c: F[k][np.nonzero(nf[k] == c)[0][-1]] for c in common} for k in F}
    diff = max(abs(last["adaptive"][c] - last["fixed"][c]) for c in common)
    assert diff < 0.03 * peak, (diff / peak, common)
    assert abs(lam["adaptive"][-1] - lam["fixed"][-1]) < 0.05 * lam["fixed"][-1]     # 1.5 % here
