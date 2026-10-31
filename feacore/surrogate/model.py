"""Graph network surrogate (PyTorch).

``MeshGraphNet``: compact MeshGraphNet-style network (Pfaff et al. 2021, "Learning Mesh-Based
Simulation with Graph Networks") in plain PyTorch: MLP encoders for nodes and edges, a processor
of residual message-passing blocks (edge MLP on [edge, sender, receiver] -> sum aggregation with
``index_add_`` -> node MLP on [node, aggregate, graph mean]), each MLP ending in LayerNorm, and an
MLP decoder. The per-graph mean ("global context") gives every block a global receptive field,
which matters for elliptic problems where the load on one edge changes the field everywhere.

``PhysicsNeMoMGN``: optional drop-in that wraps NVIDIA PhysicsNeMo's ``MeshGraphNet``. It needs
``nvidia-physicsnemo`` plus ``torch_geometric`` and the compiled ``torch_scatter`` extension
(PyG wheel index, matching the torch / CUDA build). Not used for the CPU proof-of-concept.

``Surrogate`` wraps either network with the normalisation statistics (registered buffers, so
they travel inside the checkpoint's ``state_dict``).
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


def mlp(n_in: int, n_hidden: int, n_out: int, layers: int = 2, norm: bool = True) -> nn.Sequential:
    mods: list[nn.Module] = [nn.Linear(n_in, n_hidden), nn.SiLU()]
    for _ in range(layers - 2):
        mods += [nn.Linear(n_hidden, n_hidden), nn.SiLU()]
    mods.append(nn.Linear(n_hidden, n_out))
    if norm:
        mods.append(nn.LayerNorm(n_out))
    return nn.Sequential(*mods)


class ProcessorBlock(nn.Module):
    """One residual message-passing step. The edge MLP's first layer on [e, h_s, h_r] is split
    into three linears (identical maths, avoids materialising the concatenation per edge)."""

    def __init__(self, hidden: int, global_context: bool = True):
        super().__init__()
        self.e_lin = nn.Linear(hidden, hidden)
        self.s_lin = nn.Linear(hidden, hidden, bias=False)
        self.r_lin = nn.Linear(hidden, hidden, bias=False)
        self.e_out = nn.Sequential(nn.SiLU(), nn.Linear(hidden, hidden), nn.LayerNorm(hidden))
        self.global_context = global_context
        self.n_mlp = mlp((3 if global_context else 2) * hidden, hidden, hidden)

    def forward(self, h, e, s, r, batch, inv_counts, n_graphs: int):
        m = self.e_lin(e) + self.s_lin(h)[s] + self.r_lin(h)[r]
        e = e + self.e_out(m)
        agg = torch.zeros_like(h).index_add_(0, r, e)
        parts = [h, agg]
        if self.global_context:
            g = torch.zeros(n_graphs, h.shape[1], device=h.device, dtype=h.dtype).index_add_(0, batch, h)
            parts.append((g * inv_counts[:, None])[batch])
        h = h + self.n_mlp(torch.cat(parts, dim=1))
        return h, e


class MeshGraphNet(nn.Module):
    def __init__(self, node_in: int, edge_in: int, out: int, hidden: int = 64, layers: int = 10,
                 global_context: bool = True):
        super().__init__()
        self.node_enc = mlp(node_in, hidden, hidden)
        self.edge_enc = mlp(edge_in, hidden, hidden)
        self.blocks = nn.ModuleList([ProcessorBlock(hidden, global_context) for _ in range(layers)])
        self.dec = mlp(hidden, hidden, out, norm=False)

    def forward(self, x, e, s, r, batch, n_graphs: int, return_latent: bool = False):
        counts = torch.bincount(batch, minlength=n_graphs).clamp(min=1).to(x.dtype)
        inv = 1.0 / counts
        h, ee = self.node_enc(x), self.edge_enc(e)
        for blk in self.blocks:
            h, ee = blk(h, ee, s, r, batch, inv, n_graphs)
        y = self.dec(h)
        return (y, h) if return_latent else y


class PhysicsNeMoMGN(nn.Module):
    """NVIDIA PhysicsNeMo MeshGraphNet behind the same interface (optional dependency)."""

    def __init__(self, node_in: int, edge_in: int, out: int, hidden: int = 128, layers: int = 15, **_):
        super().__init__()
        try:
            from physicsnemo.models.meshgraphnet import MeshGraphNet as _PNMGN
            import torch_geometric  # noqa: F401
            import torch_scatter  # noqa: F401
        except ImportError as e:  # pragma: no cover - optional
            raise ImportError("model_type='physicsnemo' needs nvidia-physicsnemo, torch_geometric and "
                              "torch_scatter (see docs/surrogate.md)") from e
        self.net = _PNMGN(input_dim_nodes=node_in, input_dim_edges=edge_in, output_dim=out,
                          processor_size=layers, hidden_dim_processor=hidden, hidden_dim_node_encoder=hidden,
                          hidden_dim_edge_encoder=hidden, hidden_dim_node_decoder=hidden)

    def forward(self, x, e, s, r, batch, n_graphs: int, return_latent: bool = False):  # pragma: no cover
        from torch_geometric.data import Data

        y = self.net(x, e, Data(edge_index=torch.stack([s, r]), num_nodes=x.shape[0]))
        return (y, None) if return_latent else y


MODEL_TYPES = {"mgn": MeshGraphNet, "physicsnemo": PhysicsNeMoMGN}


class Surrogate(nn.Module):
    """Network + normalisation. ``forward`` takes raw features and returns *normalised* targets;
    ``predict`` returns nondimensional targets for one graph."""

    def __init__(self, config: dict):
        super().__init__()
        self.config = dict(config)
        c = self.config
        self.net = MODEL_TYPES[c.get("model_type", "mgn")](
            c["node_in"], c["edge_in"], c["out"], hidden=c.get("hidden", 64), layers=c.get("layers", 10),
            global_context=c.get("global_context", True))
        for name, n in (("x", c["node_in"]), ("e", c["edge_in"]), ("y", c["out"])):
            self.register_buffer(f"{name}_mean", torch.zeros(n))
            self.register_buffer(f"{name}_std", torch.ones(n))

    def set_stats(self, stats: dict) -> None:
        for k, v in stats.items():
            getattr(self, k).copy_(torch.as_tensor(np.asarray(v), dtype=torch.float32))

    def forward(self, x, e, s, r, batch, n_graphs: int, return_latent: bool = False):
        xn = (x - self.x_mean) / self.x_std
        en = (e - self.e_mean) / self.e_std
        return self.net(xn, en, s, r, batch, n_graphs, return_latent=return_latent)

    def normalize_y(self, y):
        return (y - self.y_mean) / self.y_std

    def denormalize_y(self, yn):
        return yn * self.y_std + self.y_mean

    @torch.no_grad()
    def predict(self, node_features: np.ndarray, edge_features: np.ndarray, senders: np.ndarray,
                receivers: np.ndarray, device: str | torch.device = "cpu", return_latent: bool = False):
        dev = torch.device(device)
        x = torch.as_tensor(node_features, dtype=torch.float32, device=dev)
        e = torch.as_tensor(edge_features, dtype=torch.float32, device=dev)
        s = torch.as_tensor(senders, dtype=torch.int64, device=dev)
        r = torch.as_tensor(receivers, dtype=torch.int64, device=dev)
        batch = torch.zeros(len(x), dtype=torch.int64, device=dev)
        yn, h = self.forward(x, e, s, r, batch, 1, return_latent=True)
        y = self.denormalize_y(yn).cpu().numpy().astype(np.float64)
        if return_latent:
            return y, (None if h is None else h.mean(dim=0).cpu().numpy())
        return y


def save_checkpoint(path, model: Surrogate, extra: dict) -> None:
    """Checkpoint = plain dict (weights_only-loadable): config, state_dict, stats, ranges, training info."""
    blob = {"format": "feacore-surrogate-v1", "config": model.config,
            "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()}}
    blob.update(extra)
    torch.save(blob, path)


def load_checkpoint(path, device: str | torch.device = "cpu") -> tuple[Surrogate, dict]:
    from torch.torch_version import TorchVersion  # a str subclass; early checkpoints stored one

    with torch.serialization.safe_globals([TorchVersion]):
        blob = torch.load(path, map_location=device, weights_only=True)
    if blob.get("format") != "feacore-surrogate-v1":
        raise ValueError(f"{path}: not a feacore surrogate checkpoint")
    model = Surrogate(blob["config"])
    model.load_state_dict(blob["state_dict"])
    model.to(device).eval()
    return model, blob
