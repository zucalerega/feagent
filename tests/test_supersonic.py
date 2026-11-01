"""Compressible (supersonic) flow: the case schema's physics "compressible", the OpenFOAM backend's
rhoCentralFoam path (feacore/solver/_foam_compressible.py) and supersonic_blunt_wedge_2d
(feacore/problems/supersonic/blunt_wedge.py). The solves run on the coarsest reference mesh."""

import json
import math

import meshio
import pytest

from feacore.case import BC, Analysis, Case, CaseError, Material, Region
from feacore.problems.registry import get
from feacore.problems.supersonic.blunt_wedge import SupersonicBluntWedge2D, billig_standoff, pitot_ratio, standoff


def _has_openfoam() -> bool:
    from feacore.solver._foam_env import find_openfoam
    try:
        return find_openfoam() is not None
    except Exception:  # noqa: BLE001
        return False


needs_foam = pytest.mark.skipif(not _has_openfoam(), reason="OpenFOAM not found (on a Mac: scripts/openfoam_docker.sh)")


def _gas_case(**bc) -> Case:
    fs = {"value": [1000.0, 0.0], "p_inf": 1e5, "T_inf": 300.0, **bc}
    return Case(mesh="m.msh", analysis=Analysis(physics="compressible", formulation="2d"),
                materials={"gas": Material(gamma=1.4, gas_constant=287.0)},
                bcs=[BC("freestream", group="far", **fs), BC("outlet", group="out"), BC("slip", group="body")],
                regions=[Region("fluid", "gas")])


def test_schema():
    _gas_case().check()
    for bad in ({"p_inf": None}, {"T_inf": -1.0}, {"value": [1000.0]}, {"value": ["1000*y", 0.0]}):
        with pytest.raises(CaseError):
            _gas_case(**bad).check()
    c = _gas_case()
    c.materials["gas"].gamma = 1.0
    with pytest.raises(CaseError, match="gamma"):
        c.check()
    d = _gas_case().to_dict()
    assert Case.from_dict(json.loads(json.dumps(d))).to_dict() == d
    # incompressible and solid cases serialise exactly as before (no gas or freestream keys)
    flow = Case(mesh="m.msh", analysis=Analysis(physics="flow", formulation="2d"),
                materials={"water": Material(rho=1000.0, mu=1e-3)}, bcs=[BC("wall", group="w")]).to_dict()
    assert {"gamma", "gas_constant"}.isdisjoint(flow["materials"]["water"]) and "p_inf" not in flow["bcs"][0]


def test_exact_anchors():
    """Rayleigh pitot (NACA 1135 tables: p02/p1 = 12.06 at Mach 3) and Billig's cylinder standoff."""
    assert pitot_ratio(3.0) == pytest.approx(12.0610, rel=1e-4)
    assert pitot_ratio(2.0) == pytest.approx(5.6404, rel=1e-4)
    assert billig_standoff(2.0) == pytest.approx(1.241, abs=1e-3)      # the published table value at Mach 2


def test_problem_mesh_and_case(tmp_path):
    p = get("supersonic_blunt_wedge_2d")
    assert p.physics == "compressible" and p.params["alpha"] == 10.0
    case = p.case(tmp_path / "m.msh").check()
    assert [q.name for q in case.qois] == ["p_stag", "axial_force", "normal_force", "p_windward", "p_leeward"]
    u = case.bcs[0].value
    assert math.hypot(*u) / math.sqrt(1.4 * 287.0 * 300.0) == pytest.approx(3.0)
    assert math.degrees(math.atan2(u[1], u[0])) == pytest.approx(10.0)
    n = {}
    for k in (1, 2):
        m = meshio.read(p.mesh(tmp_path / f"m{k}.msh", p.params["h0"] / k))
        assert {"body", "freestream", "outlet", "fluid"} <= set(m.field_data)
        n[k] = sum(len(c.data) for c in m.cells if c.type == "quad")
    assert n[1] == 3072 and n[2] == 4 * n[1]
    with pytest.raises(ValueError, match="sized for"):
        SupersonicBluntWedge2D(id="x", alpha=20.0)


@needs_foam
def test_symmetric_case_meets_the_exact_values(tmp_path):
    """alpha = 0 on the coarsest mesh: a steady field, the stagnation pressure within 0.5 % of the pitot
    pressure, the standoff within 10 % of Billig's, and a symmetric flow (no normal force)."""
    from feacore.solver import solve

    p = SupersonicBluntWedge2D(id="sbw_a0", alpha=0.0)
    res = solve(p.case(p.mesh(tmp_path / "m.msh", 2 * p.params["h0"])), backend="openfoam")
    q = res.evaluate_all()
    assert res.info.converged and res.info.steady in ("field", "qois")
    assert q["p_stag"] == pytest.approx(p.pitot(), rel=5e-3)
    assert abs(q["normal_force"]) < 1e-6 * abs(q["axial_force"])
    assert q["p_windward"] == pytest.approx(q["p_leeward"], rel=1e-6)
    assert standoff(res, p)["standoff_over_R"] == pytest.approx(billig_standoff(3.0), rel=0.1)
    assert res.probe("mach", [[0.15, 0.125]])[0] > 1.0         # (3 R, 2.5 R): supersonic over the flank


@needs_foam
def test_angle_of_attack_breaks_the_symmetry(tmp_path):
    from feacore.solver import solve

    p = get("supersonic_blunt_wedge_2d")
    res = solve(p.case(p.mesh(tmp_path / "m.msh", 2 * p.params["h0"])), backend="openfoam")
    q = res.evaluate_all()
    assert res.info.converged
    assert q["normal_force"] > 0.5 * q["axial_force"] > 0          # lift towards +y (the flow comes from below)
    assert q["p_windward"] > 2.5 * q["p_leeward"]
    assert q["p_stag"] == pytest.approx(p.pitot(), rel=0.02)
    mach = res._cell_values("mach")
    assert mach.max() == pytest.approx(3.0, rel=0.02) and mach.min() < 0.1
