"""Out-of-distribution checks for the surrogate (numpy only).

Two independent signals, both calibrated on the training split and stored in the checkpoint:

1. ranges: continuous node features (``RANGE_CHECKED``), edge lengths and graph descriptors
   (extent, solid fraction, number of boundary loops, hole perimeter, minimum ligament, mean edge
   length, node count, load angle, Poisson ratio, fixed / loaded boundary fractions) outside the
   training range (+ a small tolerance). Interpretable: says *which* input is unusual.
2. latent: Mahalanobis distance of the graph-mean latent of the processor's last block from the
   training distribution, flagged above a threshold calibrated on train + validation graphs.

A flag means "the training data did not cover this input"; no flag does NOT mean the prediction
is right. The verification layer still has to judge the surrogate like any other submission.
"""

from __future__ import annotations

import math

import numpy as np

from feacore.surrogate.graph import DESCRIPTORS, EDGE_FEATURES, INTEGER_DESCRIPTORS, NODE_FEATURES, Graph

DESCRIPTOR_TOL = {"n_nodes": 0.25}   # relative tolerance (fraction of the value) for some descriptors
DEFAULT_TOL = 0.05                   # fraction of the training span
EDGE_OUTSIDE_FRACTION = 0.01         # flag if more than this fraction of edges is outside the range


def range_check(graph: Graph, ranges: dict, tol: float = DEFAULT_TOL) -> list[str]:
    msgs: list[str] = []
    F = graph.node_features
    for f, (lo, hi) in ranges.get("node", {}).items():
        v = F[:, NODE_FEATURES.index(f)]
        pad = tol * max(hi - lo, 1e-12)
        out = (v < lo - pad) | (v > hi + pad)
        if out.any():
            msgs.append(f"node feature '{f}': {out.mean():.1%} of nodes outside the training range "
                        f"[{lo:.3g}, {hi:.3g}] (this case: [{v.min():.3g}, {v.max():.3g}])")
    for i, f in enumerate(EDGE_FEATURES):
        if f != "length" or f not in ranges.get("edge", {}):
            continue
        lo, hi = ranges["edge"][f]
        v = graph.edge_features[:, i]
        out = (v < 0.75 * lo) | (v > 1.25 * hi)
        if out.mean() > EDGE_OUTSIDE_FRACTION:
            msgs.append(f"edge length: {out.mean():.1%} of edges outside the training range "
                        f"[{lo:.3g}, {hi:.3g}] L (mesh much finer / coarser than the training meshes)")
    for k in DESCRIPTORS:
        if k not in ranges.get("descriptors", {}):
            continue
        v = float(graph.descriptors[k])
        lo, hi = ranges["descriptors"][k]
        if k in INTEGER_DESCRIPTORS:
            allowed = ranges.get("descriptor_values", {}).get(k, [])
            if allowed and v not in allowed:
                msgs.append(f"descriptor '{k}' = {v:g} never seen in training (training: {allowed})")
            continue
        if not math.isfinite(v):
            if not ranges.get("descriptor_nan", {}).get(k, False):
                msgs.append(f"descriptor '{k}' undefined for this case (always defined in training)")
            continue
        if not (math.isfinite(lo) and math.isfinite(hi)):
            msgs.append(f"descriptor '{k}' = {v:.3g} but undefined for every training case")
            continue
        pad = DESCRIPTOR_TOL[k] * max(abs(lo), abs(hi)) if k in DESCRIPTOR_TOL else tol * max(hi - lo, 1e-12)
        if v < lo - pad or v > hi + pad:
            msgs.append(f"descriptor '{k}' = {v:.3g} outside the training range [{lo:.3g}, {hi:.3g}]")
    return msgs


def fit_latent(Z: np.ndarray, Z_cal: np.ndarray | None = None, quantile: float = 0.99) -> dict:
    """Gaussian model of graph latents (rows of Z) with shrinkage; threshold from Z and Z_cal."""
    Z = np.asarray(Z, dtype=np.float64)
    mu = Z.mean(axis=0)
    C = np.cov(Z, rowvar=False)
    C = C + (1e-3 * np.trace(C) / C.shape[0] + 1e-12) * np.eye(C.shape[0])
    P = np.linalg.inv(C)
    d_tr = mahalanobis(Z, mu, P)
    d_cal = mahalanobis(np.asarray(Z_cal), mu, P) if Z_cal is not None and len(Z_cal) else d_tr
    thr = float(max(np.quantile(d_tr, 0.995), np.quantile(d_cal, quantile)))
    return {"mean": mu.tolist(), "precision": P.tolist(), "threshold": thr,
            "train_median": float(np.median(d_tr)), "cal_median": float(np.median(d_cal))}


def mahalanobis(Z, mu, P) -> np.ndarray:
    D = np.atleast_2d(Z) - np.asarray(mu)
    return np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", D, np.asarray(P), D), 0.0))


def latent_check(z: np.ndarray | None, latent: dict | None) -> tuple[float | None, list[str]]:
    if z is None or not latent:
        return None, []
    d = float(mahalanobis(z, latent["mean"], latent["precision"])[0])
    if d > latent["threshold"]:
        return d, [f"latent distance {d:.3g} above the calibrated threshold {latent['threshold']:.3g} "
                   f"(training median {latent['train_median']:.3g})"]
    return d, []
