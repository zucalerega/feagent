"""Fast tests of the neural-surrogate pipeline (feacore.surrogate + the "nn" backend).

No GPU and no pre-trained checkpoint needed: the end-to-end backend test trains a tiny model on
synthetic targets in a temporary directory.
"""

from __future__ import annotations

import copy
import math

import numpy as np
import pytest

from feacore.case import QoI
from feacore.surrogate.data import load_sample, save_sample
from feacore.surrogate.family import SurrogatePlate, sample_params
from feacore.surrogate.graph import (NODE_FEATURES, TARGETS, UnsupportedCase, boundary, build_graph, graph_edges,
                                     nondimensionalise, physical_fields)
from feacore.surrogate.interp import TriInterpolator
from feacore.surrogate.mesh import read_msh

H_COARSE = 0.15  # relative mesh size for test meshes (~150-250 nodes)


def _plate(tmp_path, name="p", **over):
    params = {"W": 0.1, "H": 0.07, "kind": "hole", "r": 0.015, "cx": 0.05, "cy": 0.035, "angle_deg": 20.0,
              "traction": 5e6, "E": 70e9, "nu": 0.33}
    params.update(over)
    prob = SurrogatePlate(id=name, **params)
    path = prob.mesh(tmp_path / f"{name}.msh", H_COARSE * max(params["W"], params["H"]))
    return prob, prob.case(path)


def _col(name):
    return NODE_FEATURES.index(name)


# ------------------------------------------------------------------ graph construction


def test_graph_edges_single_tri6():
    cells = np.array([[0, 1, 2, 3, 4, 5]])
    s, r = graph_edges(cells, "triangle6")
    assert len(s) == 18  # 9 sub-triangulation edges, both directions
    pairs = set(zip(s.tolist(), r.tolist()))
    assert all((b, a) in pairs for a, b in pairs)
    assert (0, 1) not in pairs and (0, 3) in pairs and (3, 4) in pairs
    pts = np.array([[0, 0], [1, 0], [0, 1], [0.5, 0], [0.5, 0.5], [0, 0.5]], float)
    b = boundary(pts, cells, "triangle6")
    assert b.node_mask.all() and len(b.a) == 3
    centroid = pts[:3].mean(axis=0)
    mids = 0.5 * (pts[b.a] + pts[b.b])
    assert np.all(np.einsum("ij,ij->i", b.normal, mids - centroid) > 0)  # outward


def test_build_graph_features(tmp_path):
    _, case = _plate(tmp_path)
    mesh = read_msh(case.mesh_path())
    assert mesh.cell_type == "triangle6"
    assert {"left", "right", "top", "bottom", "hole", "plate"} <= set(mesh.group_nodes)
    g = build_graph(case, mesh)
    F = g.node_features
    assert F.shape == (len(mesh.points), len(NODE_FEATURES))
    assert g.edge_features.shape == (len(g.senders), 3)
    cats = F[:, [_col("fixed"), _col("loaded"), _col("free"), _col("interior")]]
    np.testing.assert_allclose(cats.sum(axis=1), 1.0)  # one-hot BC category
    left = mesh.group_nodes["left"]
    assert np.all(F[left, _col("fix_x")] == 1) and np.all(F[left, _col("fix_y")] == 1)
    t = np.hypot(F[:, _col("tx")], F[:, _col("ty")])
    assert math.isclose(t.max(), 1.0, rel_tol=1e-6)  # tractions are normalised by t0
    a = math.radians(20.0)
    right = np.setdiff1d(mesh.group_nodes["right"], left)
    np.testing.assert_allclose(F[right, _col("tx")], math.cos(a), rtol=1e-6)
    bn = F[:, _col("interior")] == 0
    nrm = np.hypot(F[:, _col("nx")], F[:, _col("ny")])
    assert np.allclose(nrm[mesh.group_nodes["hole"]], 1.0, atol=1e-6) and np.all(nrm[~bn] == 0)
    assert g.descriptors["n_loops"] == 2 and g.descriptors["min_ligament"] > 0
    assert 0 < F[:, _col("x")].min() + 1e-12 and F[:, _col("x")].max() <= 1 + 1e-9
    assert math.isclose(g.descriptors["load_angle"], 20.0, abs_tol=1e-6)


def test_notch_topology(tmp_path):
    _, case = _plate(tmp_path, kind="notch", r=0.02, cx=0.05, cy=0.07)
    g = build_graph(case)
    assert g.descriptors["n_loops"] == 1 and math.isnan(g.descriptors["min_ligament"])


def test_nondimensional_invariance(tmp_path):
    """Scaling geometry, E and the load leaves the network inputs unchanged; outputs scale exactly."""
    _, case = _plate(tmp_path)
    mesh = read_msh(case.mesh_path())
    g1 = build_graph(case, mesh)
    case2 = copy.deepcopy(case)
    case2.materials["material"].E *= 3.0
    case2.bcs[1].value = [7.0 * v for v in case2.bcs[1].value]
    mesh2 = copy.deepcopy(mesh)
    mesh2.points = mesh2.points * 2.5 + np.array([1.0, -2.0])
    g2 = build_graph(case2, mesh2)
    np.testing.assert_allclose(g1.node_features, g2.node_features, atol=1e-6)
    np.testing.assert_allclose(g1.edge_features, g2.edge_features, atol=1e-6)
    y = np.random.default_rng(0).normal(size=(g1.n_nodes, len(TARGETS)))
    f1, f2 = physical_fields(y, g1.scales), physical_fields(y, g2.scales)
    np.testing.assert_allclose(f2["sxx"], 7.0 * f1["sxx"], rtol=1e-12)
    np.testing.assert_allclose(f2["ux"], f1["ux"] * 7.0 * 2.5 / 3.0, rtol=1e-12)
    back = nondimensionalise({k: f1[k] for k in TARGETS}, g1.scales)
    y2 = y.copy()
    y2[:, 5] = np.maximum(y2[:, 5], 0.0)  # von Mises head is clipped at 0
    np.testing.assert_allclose(back, y2, atol=1e-12)


def test_unsupported_cases(tmp_path):
    _, case = _plate(tmp_path)
    axi = copy.deepcopy(case)
    axi.analysis.formulation = "axisymmetric"
    with pytest.raises(UnsupportedCase, match="axisymmetric"):
        build_graph(axi)
    from feacore.case import Load

    body = copy.deepcopy(case)
    body.loads = [Load("body_force", [0.0, -1e4])]
    with pytest.raises(UnsupportedCase, match="body forces"):
        build_graph(body)


def test_interpolator_reproduces_quadratics(tmp_path):
    _, case = _plate(tmp_path)
    m = read_msh(case.mesh_path())
    f = lambda p: 1.0 + 2.0 * p[:, 0] - 3.0 * p[:, 1] + 4.0 * p[:, 0] * p[:, 1] + p[:, 0] ** 2  # noqa: E731
    it = TriInterpolator(m.points, m.cells, m.cell_type)
    cells = m.cells[:40]
    lam = np.random.default_rng(1).dirichlet([2, 2, 2], size=len(cells))
    q = np.einsum("ij,ijk->ik", lam, m.points[cells[:, :3]])
    # straight-sided interior cells: P2 interpolation of a quadratic is exact
    P = m.points
    straight = np.ones(len(cells), bool)
    for mid, (i, j) in ((3, (0, 1)), (4, (1, 2)), (5, (2, 0))):
        straight &= np.all(np.abs(P[cells[:, mid]] - 0.5 * (P[cells[:, i]] + P[cells[:, j]])) < 1e-12, axis=1)
    assert straight.sum() > 10
    np.testing.assert_allclose(it(f(m.points), q)[straight], f(q)[straight], atol=1e-10)
    np.testing.assert_allclose(it(f(m.points), m.points[:50]), f(m.points[:50]), atol=1e-10)


# ------------------------------------------------------------------ dataset round-trip


def test_dataset_roundtrip(tmp_path):
    _, case = _plate(tmp_path)
    g = build_graph(case)
    y = np.random.default_rng(2).normal(size=(g.n_nodes, len(TARGETS))).astype(np.float32)
    p = save_sample(tmp_path / "s.npz", g, y, {"category": "id", "params": {"a": 1.5}})
    d = load_sample(p)
    np.testing.assert_array_equal(d["node_features"], g.node_features)
    np.testing.assert_array_equal(d["targets"], y)
    np.testing.assert_array_equal(d["senders"], g.senders)
    assert d["meta"]["params"]["a"] == 1.5 and d["scales"]["t0"] == g.scales.t0
    assert d["descriptors"]["n_loops"] == 2.0


# ------------------------------------------------------------------ model + backend


def test_model_forward_shapes(tmp_path):
    torch = pytest.importorskip("torch")
    from feacore.surrogate.model import Surrogate
    from feacore.surrogate.train import collate, to_tensor_graph

    graphs = []
    for i, r in enumerate((0.012, 0.02)):
        _, case = _plate(tmp_path, name=f"m{i}", r=r)
        g = build_graph(case)
        d = {"node_features": g.node_features, "edge_features": g.edge_features, "senders": g.senders,
             "receivers": g.receivers, "targets": np.zeros((g.n_nodes, 6), np.float32), "path": f"m{i}.npz"}
        graphs.append(to_tensor_graph(d))
    model = Surrogate({"node_in": len(NODE_FEATURES), "edge_in": 3, "out": 6, "hidden": 16, "layers": 3})
    b = collate(graphs, "cpu")
    y, h = model(b["x"], b["e"], b["s"], b["r"], b["batch"], b["n_graphs"], return_latent=True)
    assert y.shape == (sum(len(g.x) for g in graphs), 6) and h.shape == (len(y), 16)
    y1 = model.predict(graphs[0].x.numpy(), graphs[0].e.numpy(), graphs[0].s.numpy(), graphs[0].r.numpy())
    assert y1.shape == (len(graphs[0].x), 6) and np.isfinite(y1).all()
    # batching must not mix graphs: graph 0 alone == graph 0 inside the batch
    np.testing.assert_allclose(model.denormalize_y(y[: len(graphs[0].x)]).detach().numpy(), y1, atol=1e-5)
    assert torch.isfinite(y).all()


@pytest.fixture(scope="module")
def tiny_checkpoint(tmp_path_factory):
    """4 samples with synthetic (smooth, FEM-free) targets -> 2 epochs of a tiny network."""
    pytest.importorskip("torch")
    from feacore.surrogate.train import train

    root = tmp_path_factory.mktemp("surr")
    rng = np.random.default_rng(3)
    for i in range(5):
        params, _ = sample_params(rng, "id")
        prob = SurrogatePlate(id=f"id_{i:05d}", **params)
        (root / "id").mkdir(exist_ok=True)
        path = prob.mesh(root / "id" / f"id_{i:05d}.msh", H_COARSE * params["W"])
        case = prob.case(path.name)
        case.base_dir = str(root / "id")
        case.save(root / "id" / f"id_{i:05d}.json")
        g = build_graph(case)
        x = g.node_features
        y = np.stack([x[:, 0], -0.3 * x[:, 1], 1 + x[:, 0], 0.1 * x[:, 1], 0.05 * x[:, 0], 1 + 0.5 * x[:, 0]], 1)
        save_sample(root / "id" / f"id_{i:05d}.npz", g, y.astype(np.float32), {"category": "id", "h_rel": H_COARSE})
    res = train(root, root / "run", device="cpu", epochs=2, batch_size=2, hidden=8, layers=2, val_frac=0.2,
                test_frac=0.2, threads=1, verbose=False)
    assert res["epochs_run"] == 2
    return root, root / "run" / "best.pt"


def test_backend_end_to_end(tiny_checkpoint, tmp_path, monkeypatch):
    from feacore.case import Case
    from feacore.solver import solve

    root, ckpt = tiny_checkpoint
    case = Case.load(root / "id" / "id_00000.json")
    monkeypatch.setenv("FEACORE_NN_CHECKPOINT", str(ckpt))
    res = solve(case, backend="nn", device="cpu")
    assert res.info.backend == "nn" and any(w.startswith("surrogate:") for w in res.info.warnings)
    coords, vm = res.nodal("von_mises")
    assert coords.shape == (res.info.n_nodes, 2) and vm.shape == (res.info.n_nodes,) and np.all(vm >= 0)
    vals = res.evaluate_all()
    assert set(vals) == {q.name for q in case.qois} and all(np.isfinite(v) for v in vals.values())
    _, ux = res.nodal("ux")
    np.testing.assert_allclose(res.probe("ux", coords[:20]), ux[:20], rtol=1e-9, atol=1e-30)
    assert res.evaluate(QoI("m", "max_abs", field="sxx", group="hole")) >= 0
    with pytest.raises(NotImplementedError):
        res.evaluate(QoI("R", "reaction", group="left", component=0))
    out = res.write_vtu(tmp_path / "nn.vtu")
    assert out.exists()
    # checkpoint from analysis.options; an unseen topology (notch) is flagged out of distribution
    prob, notch = _plate(tmp_path, name="notch", kind="notch", r=0.02, cx=0.05, cy=0.07)
    monkeypatch.delenv("FEACORE_NN_CHECKPOINT")
    notch.analysis.options["nn_checkpoint"] = str(ckpt)
    r2 = solve(notch, backend="nn", device="cpu")
    assert r2.ood and any("n_loops" in w for w in r2.info.warnings)


def test_backend_errors(tiny_checkpoint, tmp_path):
    from feacore.solver import solve

    _, ckpt = tiny_checkpoint
    _, case = _plate(tmp_path)
    with pytest.raises(FileNotFoundError, match="checkpoint"):
        solve(case, backend="nn")
    case.analysis.formulation = "axisymmetric"
    with pytest.raises(NotImplementedError, match="axisymmetric"):
        solve(case, backend="nn", checkpoint=str(ckpt))


def test_fem_targets_scale_exactly(tmp_path):
    """Nondimensional FEM targets are invariant to E and load magnitude (needs the skfem backend)."""
    try:
        from feacore.solver import solve

        _, case = _plate(tmp_path)
        r1 = solve(case, backend="skfem")
    except Exception as e:  # noqa: BLE001 - solver still being written by another workstream
        pytest.skip(f"skfem backend unavailable: {e}")
    from feacore.surrogate.data import fem_targets

    case2 = copy.deepcopy(case)
    case2.materials["material"].E *= 4.0
    case2.bcs[1].value = [0.1 * v for v in case2.bcs[1].value]
    r2 = solve(case2, backend="skfem")
    y1, y2 = fem_targets(r1, build_graph(case)), fem_targets(r2, build_graph(case2))
    np.testing.assert_allclose(y1, y2, rtol=1e-6, atol=1e-9 * np.abs(y1).max())
