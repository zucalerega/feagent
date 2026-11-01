"""Shape functions, point location, volumes, regions, scoring maths."""

import itertools
import math

import numpy as np
import pytest

from feabench.fem import shape as sf
from feabench.fem.locate import Locator, element_volumes
from feabench.fem.regions import select_nodes
from feabench import scoring


def _ref_nodes():
    t4 = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], float)
    w6 = np.array([[0, 0, -1], [1, 0, -1], [0, 1, -1], [0, 0, 1], [1, 0, 1], [0, 1, 1]], float)
    return {
        "tet4": t4,
        "tet10": np.vstack([t4] + [(t4[a] + t4[b]) / 2 for a, b in sf.TET10_EDGES]),
        "hex8": sf._HEX_CORNERS,
        "hex20": np.vstack([sf._HEX_CORNERS, sf._HEX_MID]),
        "wedge6": w6,
        "wedge15": np.vstack([w6] + [(w6[a] + w6[b]) / 2 for a, b in sf.WEDGE15_EDGES]),
    }


@pytest.mark.parametrize("etype", ["tet4", "tet10", "hex8", "hex20", "wedge6", "wedge15"])
def test_shape_functions(etype):
    X = _ref_nodes()[etype]
    assert np.allclose(sf.shape(etype, X), np.eye(len(X)))  # Kronecker delta
    q = np.random.default_rng(0).random((20, 3)) * 0.3 + 0.1
    assert np.allclose(sf.shape(etype, q).sum(1), 1.0)  # partition of unity
    assert np.allclose(sf.shape(etype, q) @ X, q)  # linear completeness


@pytest.mark.parametrize("etype,vol", [("tet10", 1 / 6), ("hex20", 8.0), ("wedge15", 1.0)])
def test_isoparametric_volume(etype, vol):
    X = _ref_nodes()[etype]
    v = element_volumes(X, {etype: np.arange(len(X))[None]})[etype]
    assert v[0] == pytest.approx(vol)


def test_locator_interpolates_quadratic_field_exactly_in_tet10():
    X = _ref_nodes()["tet10"] * 2.0 + [1.0, -1.0, 0.5]
    f = lambda p: p[:, 0] ** 2 + 3 * p[:, 1] * p[:, 2] - p[:, 2]
    L = Locator(X, {"tet10": np.arange(10)[None]})
    P = np.array([[1.4, -0.6, 0.9], [1.1, -0.9, 0.6]])
    loc = L.locate(P)
    assert loc.inside.all()
    assert np.allclose(L.interpolate(loc, f(X)), f(P))


def test_locator_on_hex_grid_and_outside_points():
    g = np.linspace(0, 1, 3)
    P = np.array(list(itertools.product(g, g, g)))
    idx = {tuple(p): i for i, p in enumerate(P)}
    hexes = []
    for i, j, k in itertools.product(range(2), repeat=3):
        c = [(i, j, k), (i + 1, j, k), (i + 1, j + 1, k), (i, j + 1, k),
             (i, j, k + 1), (i + 1, j, k + 1), (i + 1, j + 1, k + 1), (i, j + 1, k + 1)]
        hexes.append([idx[tuple(np.array(x) / 2)] for x in c])
    L = Locator(P, {"hex8": np.array(hexes)})
    q = np.random.default_rng(2).random((50, 3))
    loc = L.locate(q)
    assert loc.inside.all()
    assert np.allclose(L.interpolate(loc, P @ [1, 2, 3]), q @ [1, 2, 3])
    out = L.locate(np.array([[1.2, 0.5, 0.5]]))
    assert not out.inside[0] and out.distance[0] == pytest.approx(0.2)


def test_regions():
    X = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [3, 4, 0], [0.5, 0.5, 0.5]], float)
    assert select_nodes(X, {"type": "plane", "point": [0, 0, 0], "normal": [0, 0, 1]}).sum() == 4
    cyl = {"type": "cylinder", "point": [0, 0, 0], "axis": [0, 0, 1], "radius": 5.0}
    assert select_nodes(X, cyl).tolist() == [False] * 4 + [True, False]
    both = {"type": "and", "regions": [{"type": "plane", "point": [0, 0, 0], "normal": [0, 0, 1]},
                                       {"type": "box", "min": [0, 0, 0], "max": [1, 1, 0]}]}
    assert select_nodes(X, both).sum() == 3
    ell = {"type": "elliptic_cylinder", "center": [0, 0], "semi_axes": [3.0, 4.0 * math.sqrt(1 - 1 / 9)]}
    assert select_nodes(np.array([[1.0, 4.0 * math.sqrt(1 - 1 / 9) * math.sqrt(1 - 1 / 9), 0.0]]), ell).all()


def test_credit_and_efficiency():
    assert scoring.qoi_credit(0.005, 0.01) == 1.0
    assert scoring.qoi_credit(0.1, 0.01) == pytest.approx(0.0)
    assert scoring.qoi_credit(math.sqrt(10) * 0.01, 0.01) == pytest.approx(0.5)
    assert scoring.efficiency(1.0, 1.0, 10.0) == 1.0
    assert scoring.efficiency(10.0, 1.0, 10.0) == 0.0
    assert scoring.efficiency(math.sqrt(10), 1.0, 10.0) == pytest.approx(0.5)
    assert scoring.efficiency(None, 1.0, 10.0) == 1.0


def test_case_score_gates_and_bounds():
    cfg = scoring.ScoreConfig()
    q_ok = scoring.score_qoi({"id": "a", "reference": 1.0, "tolerance": {"rel": 0.01}}, {"value": 1.005}, cfg)
    q_gate = scoring.score_qoi({"id": "r", "reference": 10.0, "tolerance": {"rel": 0.005}, "gate": True},
                               {"value": 5.0}, cfg)
    s = scoring.case_score([q_ok], {"solver_ok": True}, cost=0.1, wall_s=10, cfg=cfg)
    assert s["score"] == pytest.approx(1.0) and s["all_qois_within_tolerance"]
    s = scoring.case_score([q_ok, q_gate], {"solver_ok": True}, cost=0.1, wall_s=10, cfg=cfg)
    assert s["score"] == 0.0 and not s["gates_passed"]  # a failed load-path gate zeroes the case
    s = scoring.case_score([q_ok], {"solver_ok": True}, cost=1e3, wall_s=1e5, cfg=cfg)
    assert s["score"] == pytest.approx(1 - cfg.efficiency_weight)  # efficiency costs at most w_eff
    cost = scoring.cost_usd({"input_tokens": 1e6, "output_tokens": 1e5}, 3600.0, scoring.ScoreConfig.for_model("claude-opus-5-5"))
    assert cost == pytest.approx(4.0 + 2.0 + 0.05)


def test_population_metrics():
    res = [
        {"agent": "a", "case": "c1", "score": 0.9, "accuracy": 0.9, "gates_passed": True, "cost_usd": 1.0,
         "all_qois_within_tolerance": True},
        {"agent": "b", "case": "c1", "score": 0.5, "accuracy": 0.5, "gates_passed": True, "cost_usd": 0.5,
         "all_qois_within_tolerance": False},
        {"agent": "a", "case": "c2", "score": 0.7, "accuracy": 0.8, "gates_passed": True, "cost_usd": 2.0,
         "all_qois_within_tolerance": True},
        {"agent": "b", "case": "c2", "score": 0.2, "accuracy": 0.3, "gates_passed": True, "cost_usd": 2.0,
         "all_qois_within_tolerance": False},
    ]
    summ = scoring.population_summary(res, cost_cap=10.0)
    assert summ["ratings"]["a"] > summ["ratings"]["b"]
    assert summ["coverage"] == 1.0 and summ["pass_rate"] == 0.5
    assert summ["pareto_front"]["c1"] == ["b", "a"]  # b cheaper, a more accurate on c1
    assert summ["pareto_front"]["c2"] == ["a"]  # same cost, a more accurate
    assert summ["hypervolume"] > 0
