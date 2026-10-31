"""SimJEB .fem handling, tested on synthetic Nastran decks (the real dataset is not reachable here)."""

import math
import shutil

import numpy as np
import pytest

from feabench.suites import simjeb

needs_ccx = pytest.mark.skipif(shutil.which("ccx") is None, reason="CalculiX not installed")


def _small(name, fields):
    """Small-field card with continuation lines (8 fields per line)."""
    lines, cur = [], f"{name:<8}"
    for i, f in enumerate(fields):
        if i and i % 8 == 0:
            lines.append(cur + "+")
            cur = "+       "
        cur += f"{str(f):>8}"[:8]
    lines.append(cur)
    return "\n".join(lines)


def test_nastran_formats_and_shorthand_exponents():
    text = "\n".join([
        "SOL 101", "CEND", "SUBCASE 7", "  LABEL = vertical", "  LOAD = 2", "  SPC = 1", "BEGIN BULK",
        "$ comment",
        _small("GRID", [1, "", "0.0", "1.5-3", "2.+2"]),
        "GRID*                  2                            1.25            -3.5",
        "*                    7.0",
        "GRID,3,,1.0,2.0,3.0",
        _small("SPC1", [1, 123456, 1, "THRU", 3]),
        "FORCE,2,1,,10.,0.,0.,-1.",
        "ENDDATA",
    ])
    m = simjeb.parse_fem(text)
    assert np.allclose(m.nodes[1], [0.0, 1.5e-3, 200.0])
    assert np.allclose(m.nodes[2], [1.25, -3.5, 7.0])
    assert np.allclose(m.nodes[3], [1.0, 2.0, 3.0])
    assert [g for g, c, v in m.spcs[1]] == [1, 2, 3]
    assert np.allclose(m.forces[2][0][1], [0, 0, -10.0])
    assert m.subcases == [{"id": 7, "LABEL": "vertical", "LOAD": 2, "SPC": 1}]


def test_rbe3_distribution_reproduces_resultants():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(40, 3)) * [10, 10, 1]
    coords = {i + 1: x for i, x in enumerate(X)}
    w = rng.uniform(0.5, 2.0, size=40)
    groups = [(float(wi), "123", [i + 1]) for i, wi in enumerate(w)]
    ref = np.array([1.0, -2.0, 5.0])
    F, M = np.array([3.0, -1.0, 2.0]), np.array([0.5, 4.0, -7.0])
    f = simjeb.rbe3_nodal_forces(coords, ref, groups, F, M)
    tot_F = sum(f.values())
    tot_M = sum(np.cross(coords[g] - ref, v) for g, v in f.items())
    assert np.allclose(tot_F, F)
    assert np.allclose(tot_M, M)


def test_fit_cylinder_recovers_axis_and_radius():
    t = np.linspace(0, 2 * np.pi, 60, endpoint=False)
    s = np.repeat([0.0, 4.0, 8.0], 20)
    a = np.array([1.0, 1.0, 0.0]) / math.sqrt(2)
    u = np.array([0.0, 0.0, 1.0])
    v = np.cross(a, u)
    P = np.array([5.0, 6.0, 7.0]) + np.outer(s, a) + 3.0 * (np.outer(np.cos(t), u) + np.outer(np.sin(t), v))
    fit = simjeb.fit_cylinder(P)
    assert abs(fit["radius"] - 3.0) < 1e-3
    assert abs(abs(np.dot(fit["axis"], a)) - 1) < 1e-4


def _shaft_fem(R=10.0, L=100.0, T=1.0e5, h=3.0):
    """Nastran deck of a shaft: base face RBE2 to a fixed node, end face RBE3 with a MOMENT."""
    gmsh = pytest.importorskip("gmsh")
    from feabench.fem.build import mesh_gmsh_model

    gmsh.initialize(readConfigFiles=False)
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("shaft")
    gmsh.model.occ.addCylinder(0, 0, 0, 0, 0, L, R)
    gmsh.model.occ.synchronize()
    mesh = mesh_gmsh_model(h, order=2)
    gmsh.finalize()
    lines = ["SOL 101", "CEND", "SUBCASE 1", "  LABEL = torsion", "  LOAD = 2", "  SPC = 1", "BEGIN BULK"]
    for i, (x, y, z) in enumerate(mesh.coords, start=1):
        if i % 2:
            lines.append(f"GRID,{i},,{x:.12g},{y:.12g},{z:.12g}")
        else:  # large field
            lines.append(f"{'GRID*':<8}{i:>16}{'':>16}{x:>16.9e}{y:>16.9e}")
            lines.append(f"{'*':<8}{z:>16.9e}")
    for e, conn in enumerate(mesh.tets, start=1):
        lines.append(_small("CTETRA", [e, 1] + [int(n) + 1 for n in conn]))
    n = len(mesh.coords)
    base = [i + 1 for i in np.where(np.abs(mesh.coords[:, 2]) < 1e-9)[0]]
    end = [i + 1 for i in np.where(np.abs(mesh.coords[:, 2] - L) < 1e-9)[0]]
    lines.append(f"GRID,{n + 1},,0.,0.,-5.")
    lines.append(f"GRID,{n + 2},,0.,0.,{L}")
    lines.append(_small("RBE2", [9001, n + 1, 123456] + base))
    lines.append(_small("RBE3", [9002, "", n + 2, 123456, "1.0", 123] + end))
    lines.append(_small("SPC1", [1, 123456, n + 1]))
    lines.append(f"MOMENT,2,{n + 2},,{T},0.,0.,1.")
    lines.append("MAT1,1,210000.,,0.3,7.85-9")
    lines.append("PSOLID,1,1")
    lines.append("ENDDATA")
    return "\n".join(lines)


@needs_ccx
def test_fem_to_ccx_torsion_matches_saint_venant(tmp_path):
    from feabench.ccx import read_frd, run_ccx

    R, L, T = 10.0, 100.0, 1.0e5
    m = simjeb.parse_fem(_shaft_fem(R, L, T))
    assert len(m.rbe2) == 1 and len(m.rbe3) == 1 and len(m.tets) > 100
    info = simjeb.fem_to_ccx(m, m.subcases[0], tmp_path / "shaft.inp")
    assert np.allclose(info["resultant"], 0.0, atol=1e-6 * T)  # pure moment
    rr = run_ccx(tmp_path / "shaft.inp")
    assert rr.ok, rr.errors
    res = read_frd(tmp_path / "shaft.frd")
    U = res.field_on_nodes("DISP")
    X = res.coords
    G = 210000.0 / (2 * 1.3)
    twist = T / (G * math.pi * R**4 / 2)
    # surface nodes near mid-length: tangential displacement = twist * z * R
    sel = (np.abs(np.hypot(X[:, 0], X[:, 1]) - R) < 1e-3) & (np.abs(X[:, 2] - L / 2) < 10)
    ut = (U[sel, 1] * X[sel, 0] - U[sel, 0] * X[sel, 1]) / R
    assert np.allclose(ut, twist * X[sel, 2] * R, rtol=0.01)
