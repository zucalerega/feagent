"""Training loop for the graph surrogate (used by ``scripts/train_surrogate.py``).

* in-distribution samples (``<data>/id``) are split train / val / test (deterministic, seeded);
  the split is saved next to the checkpoint. ``<data>/ood`` is never trained on.
* normalisation statistics and the training ranges of features / descriptors (for the
  out-of-distribution checks at inference) are computed on the training split only.
* loss: MSE of standardised nodal targets; AdamW; cosine learning-rate decay driven by
  max(step fraction, wall-time fraction) so a time-limited run still anneals.
* checkpoints: ``best.pt`` (lowest validation loss) and ``last.pt``, plus ``history.json``.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from feacore.surrogate.data import list_samples, load_sample
from feacore.surrogate.graph import (DESCRIPTORS, EDGE_FEATURES, INTEGER_DESCRIPTORS, NODE_FEATURES,
                                     RANGE_CHECKED, TARGETS)
from feacore.surrogate.model import Surrogate, save_checkpoint


@dataclass
class TensorGraph:
    x: torch.Tensor
    e: torch.Tensor
    s: torch.Tensor
    r: torch.Tensor
    y: torch.Tensor
    name: str


def to_tensor_graph(d: dict) -> TensorGraph:
    return TensorGraph(x=torch.from_numpy(d["node_features"].astype(np.float32)),
                       e=torch.from_numpy(d["edge_features"].astype(np.float32)),
                       s=torch.from_numpy(d["senders"].astype(np.int64)),
                       r=torch.from_numpy(d["receivers"].astype(np.int64)),
                       y=torch.from_numpy(d["targets"].astype(np.float32)),
                       name=Path(d["path"]).stem)


def collate(graphs: list[TensorGraph], device) -> dict:
    offs = np.cumsum([0] + [len(g.x) for g in graphs])
    batch = torch.cat([torch.full((len(g.x),), i, dtype=torch.int64) for i, g in enumerate(graphs)])
    out = {
        "x": torch.cat([g.x for g in graphs]), "e": torch.cat([g.e for g in graphs]),
        "s": torch.cat([g.s + int(o) for g, o in zip(graphs, offs)]),
        "r": torch.cat([g.r + int(o) for g, o in zip(graphs, offs)]),
        "y": torch.cat([g.y for g in graphs]), "batch": batch,
    }
    out = {k: v.to(device, non_blocking=True) for k, v in out.items()}
    out["n_graphs"] = len(graphs)
    return out


def split_names(names: list[str], val_frac: float, test_frac: float, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    perm = [names[i] for i in rng.permutation(len(names))]
    n_val = max(1, int(round(val_frac * len(names))))
    n_test = max(1, int(round(test_frac * len(names))))
    return {"test": sorted(perm[:n_test]), "val": sorted(perm[n_test:n_test + n_val]),
            "train": sorted(perm[n_test + n_val:])}


def compute_stats(samples: list[dict]) -> dict:
    X = np.concatenate([d["node_features"] for d in samples]).astype(np.float64)
    Ee = np.concatenate([d["edge_features"] for d in samples]).astype(np.float64)
    Y = np.concatenate([d["targets"] for d in samples]).astype(np.float64)

    def ms(a):
        m, s = a.mean(axis=0), a.std(axis=0)
        return m.tolist(), np.where(s < 1e-6, 1.0, s).tolist()

    st = {}
    st["x_mean"], st["x_std"] = ms(X)
    st["e_mean"], st["e_std"] = ms(Ee)
    st["y_mean"], st["y_std"] = ms(Y)
    return st


def compute_ranges(samples: list[dict]) -> dict:
    X = np.concatenate([d["node_features"] for d in samples])
    E = np.concatenate([d["edge_features"] for d in samples])
    node = {f: [float(X[:, NODE_FEATURES.index(f)].min()), float(X[:, NODE_FEATURES.index(f)].max())]
            for f in RANGE_CHECKED}
    edge = {f: [float(np.percentile(E[:, i], 0.05)), float(np.percentile(E[:, i], 99.95))]
            for i, f in enumerate(EDGE_FEATURES)}
    desc, desc_values, desc_nan = {}, {}, {}
    for k in DESCRIPTORS:
        v = np.array([d["descriptors"][k] for d in samples], dtype=float)
        fin = v[np.isfinite(v)]
        desc[k] = [float(fin.min()), float(fin.max())] if len(fin) else [math.nan, math.nan]
        desc_nan[k] = bool(np.isnan(v).any())
        if k in INTEGER_DESCRIPTORS:
            desc_values[k] = sorted(set(float(x) for x in fin))
    return {"node": node, "edge": edge, "descriptors": desc, "descriptor_values": desc_values,
            "descriptor_nan": desc_nan}


def _evaluate(model, graphs, device, batch_size):
    model.eval()
    tot, n = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(graphs), batch_size):
            b = collate(graphs[i:i + batch_size], device)
            yn = model(b["x"], b["e"], b["s"], b["r"], b["batch"], b["n_graphs"])
            tot += float(((yn - model.normalize_y(b["y"])) ** 2).sum())
            n += b["y"].numel()
    model.train()
    return tot / max(n, 1)


def train(data: str | Path, out: str | Path, device: str = "cpu", epochs: int = 50, batch_size: int = 4,
          lr: float = 1e-3, lr_min: float = 1e-5, weight_decay: float = 1e-5, hidden: int = 64, layers: int = 10,
          global_context: bool = True, model_type: str = "mgn", val_frac: float = 0.1, test_frac: float = 0.1,
          max_train: int | None = None, max_minutes: float | None = None, seed: int = 0, threads: int | None = None,
          amp: bool = False, log_every: int = 1, verbose: bool = True, init: str | Path | None = None) -> dict:
    t_start = time.perf_counter()
    if threads:
        torch.set_num_threads(threads)
    torch.manual_seed(seed)
    dev = torch.device(device)
    data, out = Path(data), Path(out)
    out.mkdir(parents=True, exist_ok=True)

    paths = list_samples(data, "id")
    if not paths:
        raise FileNotFoundError(f"no samples under {data / 'id'}")
    by_name = {p.stem: p for p in paths}
    split = split_names(sorted(by_name), val_frac, test_frac, seed)
    if max_train is not None:
        split["train"] = split["train"][:max_train]
    (out / "split.json").write_text(json.dumps(split, indent=1))
    load = lambda names: [load_sample(by_name[n]) for n in names]  # noqa: E731
    tr_s, va_s = load(split["train"]), load(split["val"])
    stats = compute_stats(tr_s)
    ranges = compute_ranges(tr_s)
    train_g = [to_tensor_graph(d) for d in tr_s]
    val_g = [to_tensor_graph(d) for d in va_s]
    h_rel = sorted({d["meta"].get("h_rel") for d in tr_s if d["meta"].get("h_rel") is not None})
    del tr_s, va_s

    config = {"model_type": model_type, "node_in": len(NODE_FEATURES), "edge_in": len(EDGE_FEATURES),
              "out": len(TARGETS), "hidden": hidden, "layers": layers, "global_context": global_context}
    model = Surrogate(config)
    model.set_stats(stats)
    if init is not None:  # warm start (same architecture); keeps the checkpoint's normalisation
        from feacore.surrogate.model import load_checkpoint

        prev, prev_blob = load_checkpoint(init, device="cpu")
        if prev.config != config:
            raise ValueError(f"--init checkpoint config {prev.config} differs from {config}")
        model.load_state_dict(prev.state_dict())
        stats = prev_blob["stats"]
        if verbose:
            print(f"warm start from {init}", flush=True)
    model.to(dev).train()
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    steps_per_epoch = math.ceil(len(train_g) / batch_size)
    total_steps = epochs * steps_per_epoch
    budget = None if max_minutes is None else 60.0 * max_minutes
    use_amp = amp and dev.type == "cuda"

    from feacore.surrogate.family import DEFAULT_H_REL, ID_RANGES, OOD_CATEGORIES

    extra_base = {
        "stats": stats, "ranges": ranges,
        "features": {"node": list(NODE_FEATURES), "edge": list(EDGE_FEATURES), "targets": list(TARGETS),
                     "range_checked": list(RANGE_CHECKED), "descriptors": list(DESCRIPTORS)},
        "family": {"name": "feacore.surrogate.family.SurrogatePlate", "id_ranges": ID_RANGES,
                   "ood_categories": OOD_CATEGORIES, "h_rel": h_rel or [DEFAULT_H_REL]},
        "versions": {"torch": str(torch.__version__)},
    }
    if verbose:
        print(f"train {len(train_g)} / val {len(val_g)} / test {len(split['test'])} graphs; model {model_type} "
              f"hidden={hidden} layers={layers} params={n_params}; device={dev}", flush=True)
    history, best, step, stop = [], math.inf, 0, False
    rng = np.random.default_rng(seed)
    for epoch in range(1, epochs + 1):
        t_ep = time.perf_counter()
        order = rng.permutation(len(train_g))
        tot, cnt = 0.0, 0
        for i in range(0, len(order), batch_size):
            progress = step / total_steps
            if budget is not None:
                progress = max(progress, (time.perf_counter() - t_start) / budget)
            if progress >= 1.0:
                stop = True
                break
            for gr in opt.param_groups:
                gr["lr"] = lr_min + 0.5 * (lr - lr_min) * (1.0 + math.cos(math.pi * progress))
            b = collate([train_g[j] for j in order[i:i + batch_size]], dev)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                yn = model(b["x"], b["e"], b["s"], b["r"], b["batch"], b["n_graphs"])
            loss = torch.mean((yn.float() - model.normalize_y(b["y"])) ** 2)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * b["y"].shape[0]
            cnt += b["y"].shape[0]
            step += 1
        if cnt == 0:
            break
        val = _evaluate(model, val_g, dev, batch_size)
        rec = {"epoch": epoch, "step": step, "train_loss": tot / cnt, "val_loss": val,
               "lr": opt.param_groups[0]["lr"], "epoch_s": time.perf_counter() - t_ep,
               "elapsed_s": time.perf_counter() - t_start}
        history.append(rec)
        info = {"epochs_run": epoch, "steps": step, "best_val": min(best, val), "history": history,
                "n_train": len(train_g), "n_val": len(val_g), "n_test": len(split["test"]), "dataset": str(data),
                "device": str(dev), "params": n_params, "batch_size": batch_size, "lr": lr, "seed": seed}
        if val < best:
            best = val
            save_checkpoint(out / "best.pt", model, {**extra_base, "training": info})
        save_checkpoint(out / "last.pt", model, {**extra_base, "training": info})
        (out / "history.json").write_text(json.dumps(history, indent=1))
        if verbose and (epoch % log_every == 0 or stop):
            print(f"epoch {epoch:4d} step {step:6d} train {rec['train_loss']:.4e} val {val:.4e} "
                  f"lr {rec['lr']:.2e} {rec['epoch_s']:.1f}s/epoch total {rec['elapsed_s'] / 60:.1f} min", flush=True)
        if stop:
            break
    (out / "stats.json").write_text(json.dumps({"stats": stats, "ranges": ranges, "config": config}, indent=1))
    if (out / "best.pt").exists():
        calibrate_latent(out / "best.pt", data, device=str(dev))
    return {"best_val": best, "epochs_run": len(history), "steps": step, "out": str(out),
            "minutes": (time.perf_counter() - t_start) / 60, "params": n_params}


def graph_latents(model, samples: list[dict], device="cpu") -> np.ndarray:
    """Graph-mean latent of the last processor block for each sample (rows)."""
    Z = []
    for d in samples:
        _, z = model.predict(d["node_features"], d["edge_features"], d["senders"], d["receivers"],
                             device=device, return_latent=True)
        if z is None:
            return np.zeros((0, 0))
        Z.append(z)
    return np.asarray(Z)


def calibrate_latent(ckpt: str | Path, data: str | Path | None = None, device: str = "cpu") -> dict | None:
    """Fit the latent-space OOD model on the checkpoint's training split (threshold from the
    validation split) and store it in the checkpoint under "latent"."""
    from feacore.surrogate.model import load_checkpoint
    from feacore.surrogate.ood import fit_latent

    ckpt = Path(ckpt)
    model, blob = load_checkpoint(ckpt, device=device)
    data = Path(data or blob["training"]["dataset"])
    split = json.loads((ckpt.parent / "split.json").read_text())
    by_name = {p.stem: p for p in list_samples(data, "id")}
    Z_tr = graph_latents(model, [load_sample(by_name[n]) for n in split["train"]], device)
    Z_va = graph_latents(model, [load_sample(by_name[n]) for n in split["val"]], device)
    if Z_tr.size == 0:
        return None
    latent = fit_latent(Z_tr, Z_va)
    extra = {k: v for k, v in blob.items() if k not in ("format", "config", "state_dict")}
    extra["latent"] = latent
    save_checkpoint(ckpt, model, extra)
    return latent
