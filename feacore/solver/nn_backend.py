"""Neural-surrogate backend (``feacore.solver.solve(case, backend="nn")``): fast, UNVERIFIED.

A graph network (``feacore.surrogate``) trained on fixed-solver (skfem, P2) results of the
``surrogate_plate`` family predicts the nodal fields (ux, uy, sxx, syy, sxy, von Mises) directly on
the case mesh. It is never a reference: mesh refinement / Richardson extrapolation does not apply
to it, and it can be confidently wrong outside its training distribution. Every result carries a
"surrogate" warning in ``SolveInfo.warnings``; inputs outside the training distribution add
"surrogate OOD" warnings (``result.ood`` is True). The verification layer judges it like any other
submission.

Checkpoint: ``options["checkpoint"]``, ``case.analysis.options["nn_checkpoint"]`` or the
``FEACORE_NN_CHECKPOINT`` environment variable (relative paths: to the case directory, then the
working directory). Device: ``options["device"]``, ``analysis.options["nn_device"]``,
``FEACORE_NN_DEVICE``, else CUDA when available.

Supported: 2D linear elasticity, plane stress (plane strain via the exact E', nu' mapping), one
homogeneous material, tri3 / tri6 meshes, BCs fixed / zero displacement / symmetry, loads
traction / pressure / force on edge groups. Anything else raises ``UnsupportedCase``
(a ``NotImplementedError``): 3D, axisymmetric, heat, thermal or body loads, nonzero prescribed
displacements, point loads, several materials.
QoIs: point, max, min, max_abs. Others (reaction, integral, mean, strain_energy) raise
``NotImplementedError``.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np

from feacore.case import ELASTICITY_FIELDS, Case, QoI
from feacore.solver.base import Result, SolveInfo
from feacore.surrogate.graph import UnsupportedCase, build_graph, check_supported, physical_fields

SURROGATE_WARNING = ("surrogate: neural-network prediction (graph network trained on skfem results of the "
                     "surrogate_plate family); fast but UNVERIFIED and never a reference; mesh refinement / "
                     "Richardson extrapolation does not apply to it")
SUPPORTED_QOIS = ("point", "max", "min", "max_abs")
_MODELS: dict = {}

__all__ = ["solve", "NNResult", "UnsupportedCase", "SURROGATE_WARNING"]


def _checkpoint_path(case: Case, options: dict) -> Path:
    raw = (options.get("checkpoint") or case.analysis.options.get("nn_checkpoint")
           or os.environ.get("FEACORE_NN_CHECKPOINT"))
    if not raw:
        raise FileNotFoundError("nn backend: no checkpoint given; pass checkpoint=..., set "
                                "analysis.options['nn_checkpoint'] or the FEACORE_NN_CHECKPOINT environment variable "
                                "(train one with scripts/train_surrogate.py, see docs/surrogate.md)")
    p = Path(raw).expanduser()
    if not p.is_absolute():
        cands = ([Path(case.base_dir) / p] if case.base_dir else []) + [Path.cwd() / p]
        p = next((c for c in cands if c.exists()), cands[-1])
    if not p.exists():
        raise FileNotFoundError(f"nn backend: checkpoint {str(p)!r} not found")
    return p


def _device(case: Case, options: dict) -> str:
    dev = options.get("device") or case.analysis.options.get("nn_device") or os.environ.get("FEACORE_NN_DEVICE")
    if dev:
        return str(dev)
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def load_model(path: str | Path, device: str = "cpu"):
    """Load (and cache) a surrogate checkpoint -> (model, checkpoint dict)."""
    from feacore.surrogate.model import load_checkpoint

    path = Path(path).resolve()
    key = (str(path), path.stat().st_mtime_ns, device)
    if key not in _MODELS:
        _MODELS[key] = load_checkpoint(path, device=device)
    return _MODELS[key]


class NNResult(Result):
    def __init__(self, case: Case, info: SolveInfo, mesh, fields: dict, ood_messages: list[str],
                 latent_distance: float | None, checkpoint: str):
        self.case = case
        self.info = info
        self.mesh = mesh
        self.fields = fields
        self.ood_messages = list(ood_messages)
        self.ood = bool(ood_messages)
        self.latent_distance = latent_distance
        self.checkpoint = checkpoint
        self._interp = None

    def _field(self, name: str) -> np.ndarray:
        if name not in self.fields or name not in ELASTICITY_FIELDS:
            raise ValueError(f"nn backend: unknown field {name!r}; available: {list(ELASTICITY_FIELDS)}")
        return self.fields[name]

    def nodal(self, field: str):
        return self.mesh.points.copy(), self._field(field).copy()

    def probe(self, field: str, points) -> np.ndarray:
        from feacore.surrogate.interp import TriInterpolator

        vals = self._field(field)
        if self._interp is None:
            self._interp = TriInterpolator(self.mesh.points, self.mesh.cells, self.mesh.cell_type)
        pts = np.atleast_2d(np.asarray(points, dtype=float))
        return self._interp(vals, pts[:, :2])

    def evaluate(self, qoi: QoI) -> float:
        if qoi.type == "point_difference":   # backend-independent, see Result.evaluate_qoi
            return Result.evaluate_qoi(self, qoi)
        if qoi.type == "point":
            return float(self.probe(qoi.field, [qoi.point])[0])
        if qoi.type in ("max", "min", "max_abs"):
            v = self._field(qoi.field)[self.mesh.nodes_of(qoi.group)]
            if qoi.type == "max":
                return float(v.max())
            if qoi.type == "min":
                return float(v.min())
            return float(np.abs(v).max())
        raise NotImplementedError(f"nn backend: QoI type {qoi.type!r} ({qoi.name}) is not supported by the surrogate "
                                  f"(supported: {', '.join(SUPPORTED_QOIS)}); use backend='skfem'")

    def write_vtu(self, path: str | Path) -> Path:
        import meshio

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        pts = np.column_stack([self.mesh.points, np.zeros(len(self.mesh.points))])
        data = {k: v for k, v in self.fields.items() if k in ELASTICITY_FIELDS}
        data["displacement"] = np.column_stack([self.fields["ux"], self.fields["uy"], self.fields["uz"]])
        meshio.write(str(path), meshio.Mesh(pts, [(self.mesh.cell_type, self.mesh.cells)], point_data=data))
        return path


def solve(case: Case, **options) -> NNResult:
    """Predict the case with the surrogate. Options: checkpoint, device, vm_source ("head" =
    von Mises from the network's own output channel, "components" = from the predicted stresses)."""
    from feacore.surrogate.mesh import read_msh
    from feacore.surrogate.ood import latent_check, range_check

    t0 = time.perf_counter()
    check_supported(case)  # raises UnsupportedCase before any heavy work
    ckpt = _checkpoint_path(case, options)
    device = _device(case, options)
    model, blob = load_model(ckpt, device)
    mesh = read_msh(case.mesh_path())
    graph = build_graph(case, mesh)
    vm_source = options.get("vm_source") or case.analysis.options.get("nn_vm_source") or "head"
    if graph.no_load:
        y, z = np.zeros((graph.n_nodes, 6)), None
    else:
        y, z = model.predict(graph.node_features, graph.edge_features, graph.senders, graph.receivers,
                             device=device, return_latent=True)
    fields = physical_fields(y, graph.scales, vm_source=vm_source)
    ood = range_check(graph, blob.get("ranges", {}))
    dist, lat_msgs = latent_check(z, blob.get("latent"))
    ood += lat_msgs
    warnings = [SURROGATE_WARNING] + [f"surrogate OOD: {m}" for m in ood] + [f"surrogate: {m}" for m in graph.notes]
    if ood:
        warnings.insert(1, "surrogate OOD: input outside the training distribution; expect large errors and do "
                           "not use this result without verification")
    tri = mesh.points[mesh.cells[:, :3]]
    e1, e2 = tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
    area = float(0.5 * np.abs(e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]).sum())
    info = SolveInfo(backend="nn", physics=case.analysis.physics, formulation=case.analysis.formulation,
                     order=mesh.order, dim=2, cell_type=mesh.cell_type, n_nodes=len(mesh.points),
                     n_cells=len(mesh.cells), n_dofs=2 * len(mesh.points), measure=area,
                     h_rep=(area / max(len(mesh.cells), 1)) ** 0.5, wall_time=time.perf_counter() - t0,
                     warnings=warnings)
    return NNResult(case, info, mesh, fields, ood, dist, str(ckpt))
