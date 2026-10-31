"""Surrogate dataset samples: FEM targets on the graph nodes and ``.npz`` round-trip (numpy only).

One sample = ``<stem>.npz`` (graph, features, nondimensional targets, scales, descriptors, meta),
next to ``<stem>.msh`` and ``<stem>.json`` (the case, mesh path relative to the case file) so the
end-to-end backends can be re-run on it.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from feacore.surrogate.graph import (DESCRIPTORS, EDGE_FEATURES, NODE_FEATURES, TARGETS, Graph, Scales,
                                     nondimensionalise)


def fem_fields_at(res, points: np.ndarray, L: float) -> dict:
    """FEM nodal values of every target field at ``points`` (matched to the solver's nodes when
    they coincide, otherwise probed)."""
    out = {}
    mapping = None
    for f in TARGETS:
        coords, vals = res.nodal(f)
        coords = np.asarray(coords, dtype=float)[:, :2]
        idx = None
        if mapping is not None and mapping[0] == coords.shape:
            idx = mapping[1]
        elif len(coords) >= len(points):
            d, j = cKDTree(coords).query(points)
            if np.max(d) <= 1e-8 * L:
                idx = j
                mapping = (coords.shape, j)
        out[f] = np.asarray(vals, dtype=float)[idx] if idx is not None else np.asarray(res.probe(f, points), float)
    return out


def fem_targets(res, graph: Graph) -> np.ndarray:
    return nondimensionalise(fem_fields_at(res, graph.points, graph.scales.L), graph.scales)


def save_sample(path: str | Path, graph: Graph, y: np.ndarray | None, meta: dict | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        points=graph.points.astype(np.float64),
        cells=graph.cells.astype(np.int32),
        cell_type=np.array(graph.cell_type),
        senders=graph.senders.astype(np.int32),
        receivers=graph.receivers.astype(np.int32),
        node_features=graph.node_features.astype(np.float32),
        edge_features=graph.edge_features.astype(np.float32),
        targets=(np.zeros((graph.n_nodes, len(TARGETS))) if y is None else y).astype(np.float32),
        node_feature_names=np.array(NODE_FEATURES),
        edge_feature_names=np.array(EDGE_FEATURES),
        target_names=np.array(TARGETS),
        descriptor_names=np.array(DESCRIPTORS),
        descriptors=np.array([graph.descriptors[k] for k in DESCRIPTORS], dtype=np.float64),
        scales=np.array(json.dumps(graph.scales.to_dict())),
        meta=np.array(json.dumps(meta or {}, default=float)),
    )
    return path


def load_sample(path: str | Path) -> dict:
    with np.load(Path(path), allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    for k in ("node_feature_names", "edge_feature_names", "target_names", "descriptor_names"):
        d[k] = tuple(str(s) for s in d[k])
    if d["node_feature_names"] != NODE_FEATURES or d["target_names"] != TARGETS:
        raise ValueError(f"{path}: feature / target names differ from this feacore version")
    d["cell_type"] = str(d["cell_type"])
    d["scales"] = json.loads(str(d["scales"]))
    d["meta"] = json.loads(str(d["meta"]))
    d["descriptors"] = dict(zip(d["descriptor_names"], map(float, d["descriptors"])))
    d["path"] = str(path)
    return d


def sample_graph(d: dict) -> Graph:
    """Rebuild a ``Graph`` from a loaded sample."""
    s = dict(d["scales"])
    s["origin"] = tuple(s["origin"])
    return Graph(points=d["points"], cells=d["cells"].astype(np.int64), cell_type=d["cell_type"],
                 senders=d["senders"].astype(np.int64), receivers=d["receivers"].astype(np.int64),
                 node_features=d["node_features"], edge_features=d["edge_features"], scales=Scales(**s),
                 descriptors=d["descriptors"])


def list_samples(root: str | Path, split: str | None = None) -> list[Path]:
    root = Path(root)
    base = root / split if split else root
    return sorted(base.rglob("*.npz"))
