"""point_difference QoI: field(point) - field(point2), backend-independent via Result.probe."""

import numpy as np
import pytest

from feacore.case import QoI
from feacore.problems.registry import get
from feacore.solver import solve


def test_point_difference_matches_two_point_qois(tmp_path):
    prob = get("lame_cylinder_plane_strain")
    mesh = prob.mesh(tmp_path / "m.msh", prob.study_sizes(2)[0])
    case = prob.case(mesh, order=2)
    a, b = [0.12, 0.0], [0.18, 0.0]          # two radii on the x axis (inside the wall)
    r_in = prob.params.get("a", prob.params.get("r_inner"))
    if r_in is not None:
        r_out = prob.params.get("b", prob.params.get("r_outer"))
        a, b = [r_in + 0.25 * (r_out - r_in), 0.0], [r_in + 0.75 * (r_out - r_in), 0.0]
    case.qois += [QoI("ux_a", "point", "ux", point=a), QoI("ux_b", "point", "ux", point=b),
                  QoI("dux", "point_difference", "ux", point=a, point2=b)]
    case.check()
    q = solve(case).evaluate_all()
    assert q["dux"] == pytest.approx(q["ux_a"] - q["ux_b"], rel=1e-12, abs=1e-15)


def test_point_difference_validation():
    from feacore.case import Analysis, BC, Case, Material
    c = Case(mesh="m.msh", analysis=Analysis(formulation="plane_stress"),
             materials={"s": Material(E=1.0, nu=0.3)}, bcs=[BC("fixed", group="l")],
             qois=[QoI("d", "point_difference", "ux", point=[0, 0])])
    assert any("point2" in e for e in c.validate())
