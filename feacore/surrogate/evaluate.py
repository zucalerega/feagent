"""Evaluate a surrogate checkpoint against the fixed solver.

* field errors on held-out samples: relative L2 error of the displacement vector, the stress
  components and von Mises, and the relative error of the maximum von Mises stress, for the
  in-distribution test split and every out-of-distribution category;
* out-of-distribution flag rates (feature / descriptor ranges and latent distance);
* end-to-end timings on the same cases: ``solve(case, "skfem")`` vs ``solve(case, "nn")`` (both
  include reading the mesh), the model-only inference time and batched throughput;
* QoI comparison (max von Mises, tip displacements) through the real backends;
* plots for docs/surrogate.md.

  python -m feacore.surrogate.evaluate --ckpt /tmp/surrogate/runs/cpu2/best.pt --data /tmp/surrogate/ds1 \\
      --out /tmp/surrogate/eval --img docs/img/surrogate
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from feacore.surrogate.data import list_samples, load_sample, sample_graph
from feacore.surrogate.graph import TARGETS
from feacore.surrogate.ood import latent_check, range_check

BLUE, ORANGE, INK, INK2, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df"
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
ORANGE_RAMP = ["#fde7dc", "#f9c3a8", "#f39a72", "#eb6834", "#c94f1f", "#9c3a14", "#6e280c"]


def _rel(a, b) -> float:
    nb = float(np.linalg.norm(b))
    return float(np.linalg.norm(a - b) / nb) if nb > 0 else float("nan")


def sample_metrics(y: np.ndarray, t: np.ndarray) -> dict:
    out = {f"rel_l2_{f}": _rel(y[:, i], t[:, i]) for i, f in enumerate(TARGETS)}
    out["rel_l2_u"] = _rel(y[:, :2], t[:, :2])
    out["rel_l2_stress"] = _rel(y[:, 2:5], t[:, 2:5])
    vt, vp = float(t[:, 5].max()), float(y[:, 5].max())
    out.update(vm_max_fem=vt, vm_max_nn=vp, vm_max_err=(vp - vt) / vt)
    return out


def _summ(rows: list[dict]) -> dict:
    def med(k):
        return float(np.nanmedian([r[k] for r in rows]))

    def p90(k):
        return float(np.nanpercentile([r[k] for r in rows], 90))

    vme = [abs(r["vm_max_err"]) for r in rows]
    return {
        "n": len(rows),
        "rel_l2_u_median": med("rel_l2_u"), "rel_l2_stress_median": med("rel_l2_stress"),
        "rel_l2_vm_median": med("rel_l2_von_mises"), "rel_l2_vm_p90": p90("rel_l2_von_mises"),
        "vm_max_abs_err_median": float(np.median(vme)), "vm_max_abs_err_p90": float(np.percentile(vme, 90)),
        "flag_range": float(np.mean([r["flag_range"] for r in rows])),
        "flag_latent": float(np.mean([r["flag_latent"] for r in rows])),
        "flag_any": float(np.mean([r["flag_range"] or r["flag_latent"] for r in rows])),
    }


def evaluate(ckpt: str | Path, data: str | Path, out: str | Path, img_dir: str | Path | None = None,
             n_timing: int = 12, n_timing_ood: int = 2, device: str = "cpu", threads: int | None = 2,
             scaling: bool = False, benchmarks: bool = False) -> dict:
    import torch

    from feacore.surrogate.model import load_checkpoint

    if threads:
        torch.set_num_threads(threads)
    ckpt, data, out = Path(ckpt), Path(data), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    model, blob = load_checkpoint(ckpt, device=device)
    split = json.loads((ckpt.parent / "split.json").read_text())
    by_name = {p.stem: p for p in list_samples(data, "id")}
    samples = [load_sample(by_name[n]) for n in split["test"]] + [load_sample(p) for p in list_samples(data, "ood")]

    rows, preds = [], {}
    for d in samples:
        t0 = time.perf_counter()
        y, z = model.predict(d["node_features"], d["edge_features"], d["senders"], d["receivers"],
                             device=device, return_latent=True)
        t_inf = time.perf_counter() - t0
        g = sample_graph(d)
        rmsg = range_check(g, blob.get("ranges", {}))
        dist, lmsg = latent_check(z, blob.get("latent"))
        name = Path(d["path"]).stem
        rows.append({"name": name, "category": d["meta"]["category"], **sample_metrics(y, d["targets"]),
                     "flag_range": bool(rmsg), "flag_latent": bool(lmsg), "latent_distance": dist,
                     "range_messages": rmsg, "t_model_s": t_inf, "n_nodes": int(len(y)),
                     "fem_t_solve_s": d["meta"].get("t_solve")})
        preds[name] = y
    cats = ["id"] + sorted({r["category"] for r in rows} - {"id"})
    summary = {c: _summ([r for r in rows if r["category"] == c]) for c in cats}
    ood_rows = [r for r in rows if r["category"] != "id"]
    if ood_rows:
        summary["ood_all"] = _summ(ood_rows)

    # ---- batched model throughput (no mesh I/O)
    from feacore.surrogate.train import collate, to_tensor_graph

    tg = [to_tensor_graph(d) for d in samples[: min(32, len(samples))]]
    with torch.no_grad():
        b = collate(tg[:8], device)
        model(b["x"], b["e"], b["s"], b["r"], b["batch"], b["n_graphs"])  # warm-up
        t0 = time.perf_counter()
        for i in range(0, len(tg), 8):
            b = collate(tg[i:i + 8], device)
            model(b["x"], b["e"], b["s"], b["r"], b["batch"], b["n_graphs"])
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        t_batched = (time.perf_counter() - t0) / len(tg)

    timing = _timing(ckpt, data, rows, n_timing, n_timing_ood, device)
    timing["model_batched_per_case_s"] = t_batched
    timing["model_single_median_s"] = float(np.median([r["t_model_s"] for r in rows]))
    report = {"checkpoint": str(ckpt), "data": str(data), "device": device, "threads": threads,
              "training": {k: blob["training"].get(k) for k in ("epochs_run", "steps", "n_train", "n_val", "n_test",
                                                                "best_val", "params", "device", "batch_size")},
              "config": blob["config"], "summary": summary, "timing": timing, "rows": rows}
    report["training"]["minutes"] = blob["training"]["history"][-1]["elapsed_s"] / 60
    if scaling:
        report["scaling"] = scaling_timing(ckpt, device=device)
    if benchmarks:
        report["benchmarks"] = benchmark_check(ckpt, device=device)
    (out / "report.json").write_text(json.dumps(report, indent=1, default=float))
    (out / "report.md").write_text(markdown_tables(report))
    if img_dir is not None:
        plots(report, samples, preds, blob, Path(img_dir))
    return report


def _timing(ckpt, data, rows, n_id, n_ood, device) -> dict:
    """End-to-end backend timings and QoI comparison on a subset of the evaluated cases."""
    from feacore.case import Case
    from feacore.solver import solve

    pick = [r for r in rows if r["category"] == "id"][:n_id]
    for c in sorted({r["category"] for r in rows} - {"id"}):
        pick += [r for r in rows if r["category"] == c][:n_ood]
    entries = []
    warm = False
    for r in pick:
        split = "id" if r["category"] == "id" else "ood"
        case = Case.load(Path(data) / split / f"{r['name']}.json")
        if not warm:  # first calls pay imports / model load
            solve(case, backend="skfem")
            solve(case, backend="nn", checkpoint=str(ckpt), device=device)
            warm = True
        t0 = time.perf_counter()
        fem = solve(case, backend="skfem")
        t_fem = time.perf_counter() - t0
        t0 = time.perf_counter()
        nn = solve(case, backend="nn", checkpoint=str(ckpt), device=device)
        t_nn = time.perf_counter() - t0
        qf, qn = fem.evaluate_all(), nn.evaluate_all()
        entries.append({"name": r["name"], "category": r["category"], "n_nodes": fem.info.n_nodes,
                        "t_fem_s": t_fem, "t_nn_s": t_nn, "fem_wall_time": fem.info.wall_time,
                        "nn_ood": nn.ood, "qoi_fem": qf, "qoi_nn": qn,
                        "qoi_rel_err": {k: (qn[k] - qf[k]) / abs(qf[k]) if qf[k] else float("nan") for k in qf}})
    out = {"cases": entries}
    for c in ("id", "ood"):
        es = [e for e in entries if (e["category"] == "id") == (c == "id")]
        if not es:
            continue
        out[c] = {"n": len(es), "t_fem_median_s": float(np.median([e["t_fem_s"] for e in es])),
                  "t_nn_median_s": float(np.median([e["t_nn_s"] for e in es])),
                  "speedup_median": float(np.median([e["t_fem_s"] / e["t_nn_s"] for e in es])),
                  "qoi_abs_rel_err_median": {k: float(np.nanmedian([abs(e["qoi_rel_err"][k]) for e in es
                                                                     if k in e["qoi_rel_err"]]))
                                             for k in ("vm_max", "ux_tip", "uy_tip")},
                  "flagged": float(np.mean([e["nn_ood"] for e in es]))}
    return out


def scaling_timing(ckpt, h_rels=(0.06, 0.03, 0.015, 0.01), device="cpu", workdir=None, repeat=2) -> list[dict]:
    """FEM vs surrogate wall time for one geometry meshed at several sizes (finer meshes are out of
    distribution for the surrogate; this measures cost only)."""
    import tempfile

    from feacore.solver import solve
    from feacore.surrogate.family import SurrogatePlate

    tmp = tempfile.TemporaryDirectory(prefix="surr_scaling_") if workdir is None else None
    wd = Path(workdir or tmp.name)
    out = []
    try:
        prob = SurrogatePlate(id="scaling", W=0.1, H=0.075, r=0.012, cx=0.05, cy=0.0375, angle_deg=15.0)
        for i, h in enumerate(h_rels):
            path = prob.mesh(wd / f"scaling_{i}.msh", h * 0.1)
            case = prob.case(path)
            solve(case, backend="nn", checkpoint=str(ckpt), device=device)  # warm-up / model load
            tf, tn = [], []
            for _ in range(repeat):
                t0 = time.perf_counter()
                fem = solve(case, backend="skfem")
                tf.append(time.perf_counter() - t0)
                t0 = time.perf_counter()
                solve(case, backend="nn", checkpoint=str(ckpt), device=device)
                tn.append(time.perf_counter() - t0)
            out.append({"h_rel": h, "n_nodes": fem.info.n_nodes, "n_dofs": fem.info.n_dofs,
                        "t_fem_s": min(tf), "t_nn_s": min(tn), "speedup": min(tf) / min(tn)})
    finally:
        if tmp is not None:
            tmp.cleanup()
    return out


def benchmark_check(ckpt, device="cpu", workdir=None, max_nodes=20000) -> list[dict]:
    """Run the surrogate on every registered 2D elasticity benchmark (all outside the training
    family): supported?, flagged out of distribution?, QoI error vs the fixed solver."""
    import tempfile
    import warnings

    from feacore.problems.registry import all_problems
    from feacore.solver import solve

    tmp = tempfile.TemporaryDirectory(prefix="surr_bench_") if workdir is None else None
    wd = Path(workdir or tmp.name)
    out = []
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            probs = [p for p in all_problems() if p.physics == "elasticity" and p.formulation in
                     ("plane_stress", "plane_strain")]
        for p in probs:
            e = {"problem": p.id, "formulation": p.formulation}
            try:
                h = p.study_sizes(2)[0]
                case = p.case(p.mesh(wd / f"{p.id}.msh", h), order=2)
                fem = solve(case, backend="skfem")
                e["n_nodes"] = fem.info.n_nodes
                if fem.info.n_nodes > max_nodes:
                    e["skipped"] = "mesh too large"
                    out.append(e)
                    continue
                try:
                    nn = solve(case, backend="nn", checkpoint=str(ckpt), device=device)
                except NotImplementedError as err:
                    e.update(supported=False, reason=str(err))
                    out.append(e)
                    continue
                qf = fem.evaluate_all()
                errs = {}
                for q in case.qois:
                    try:
                        v = nn.evaluate(q)
                    except NotImplementedError:
                        continue
                    errs[q.name] = (v - qf[q.name]) / abs(qf[q.name]) if qf[q.name] else float("nan")
                e.update(supported=True, ood=nn.ood, ood_messages=nn.ood_messages[:3], latent=nn.latent_distance,
                         qoi_rel_err=errs)
            except Exception as err:  # noqa: BLE001 - another workstream's problem may be in flux
                e["error"] = f"{type(err).__name__}: {err}"
            out.append(e)
    finally:
        if tmp is not None:
            tmp.cleanup()
    return out


def markdown_tables(rep: dict) -> str:
    s = rep["summary"]
    lines = ["| split / category | n | rel. L2 u | rel. L2 stress | rel. L2 von Mises (median / p90) | "
             "max von Mises error (median / p90) | OOD flagged |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for c, v in s.items():
        label = {"id": "**in distribution (test)**", "ood_all": "**all out-of-distribution**"}.get(c, c)
        lines.append(f"| {label} | {v['n']} | {v['rel_l2_u_median']:.1%} | {v['rel_l2_stress_median']:.1%} | "
                     f"{v['rel_l2_vm_median']:.1%} / {v['rel_l2_vm_p90']:.1%} | "
                     f"{v['vm_max_abs_err_median']:.1%} / {v['vm_max_abs_err_p90']:.1%} | {v['flag_any']:.0%} |")
    t = rep["timing"]
    lines += ["", "| timing (median per case) | in distribution | out of distribution |", "|---|---:|---:|"]
    if "id" in t and "ood" in t:
        lines.append(f"| FEM `solve(case, \"skfem\")` | {t['id']['t_fem_median_s'] * 1e3:.0f} ms | "
                     f"{t['ood']['t_fem_median_s'] * 1e3:.0f} ms |")
        lines.append(f"| surrogate `solve(case, \"nn\")` | {t['id']['t_nn_median_s'] * 1e3:.0f} ms | "
                     f"{t['ood']['t_nn_median_s'] * 1e3:.0f} ms |")
        lines.append(f"| speed-up (FEM / surrogate) | {t['id']['speedup_median']:.1f}x | {t['ood']['speedup_median']:.1f}x |")
        for k in ("vm_max", "ux_tip", "uy_tip"):
            lines.append(f"| QoI `{k}` abs. rel. error | {t['id']['qoi_abs_rel_err_median'][k]:.1%} | "
                         f"{t['ood']['qoi_abs_rel_err_median'][k]:.1%} |")
    lines.append(f"\nmodel only: {t['model_single_median_s'] * 1e3:.1f} ms per graph single, "
                 f"{t['model_batched_per_case_s'] * 1e3:.1f} ms per graph batched (8 graphs), device {rep['device']}, "
                 f"{rep['threads']} threads")
    if rep.get("scaling"):
        lines += ["", "| mesh size h / L | nodes | FEM dofs | FEM | surrogate | speed-up |", "|---:|---:|---:|---:|---:|---:|"]
        for e in rep["scaling"]:
            lines.append(f"| {e['h_rel']:g} | {e['n_nodes']} | {e['n_dofs']} | {e['t_fem_s'] * 1e3:.0f} ms | "
                         f"{e['t_nn_s'] * 1e3:.0f} ms | {e['speedup']:.1f}x |")
    if rep.get("benchmarks"):
        lines += ["", "| registered benchmark (never seen) | supported | OOD flagged | QoI error vs FEM (surrogate-evaluable QoIs) |",
                  "|---|---|---|---|"]
        for e in rep["benchmarks"]:
            if "error" in e or "skipped" in e:
                lines.append(f"| {e['problem']} | n/a ({e.get('error') or e.get('skipped')}) | | |")
            elif not e["supported"]:
                lines.append(f"| {e['problem']} | no: {e['reason'].split(':')[-1].strip()[:70]} | | |")
            else:
                q = ", ".join(f"{k} {v:+.0%}" for k, v in e["qoi_rel_err"].items()) or "none evaluable"
                lines.append(f"| {e['problem']} | yes | {'yes' if e['ood'] else '**no**'} | {q} |")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------- plots


def _style(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(INK2)
        ax.spines[sp].set_linewidth(0.8)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def _subtri(cells, cell_type):
    if cell_type == "triangle6":
        return np.concatenate([cells[:, [0, 3, 5]], cells[:, [3, 1, 4]], cells[:, [5, 4, 2]], cells[:, [3, 4, 5]]])
    return cells[:, :3]


def plots(rep: dict, samples: list[dict], preds: dict, blob: dict, img: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.tri import Triangulation

    img.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "text.color": INK, "axes.labelcolor": INK2, "axes.titlesize": 10})
    rows = rep["rows"]
    files = []

    # 1. parity of max von Mises
    fig, ax = plt.subplots(figsize=(5.0, 4.4), dpi=150)
    _style(ax)
    allv = np.array([[r["vm_max_fem"], r["vm_max_nn"]] for r in rows])
    lo, hi = 0.0, float(allv.max()) * 1.05
    xs = np.linspace(lo, hi, 2)
    ax.fill_between(xs, 0.9 * xs, 1.1 * xs, color=GRID, alpha=0.7, linewidth=0, label="±10 %")
    ax.plot(xs, xs, color=INK2, linewidth=1)
    for sel, color, label, mk in ((lambda r: r["category"] == "id", BLUE, "in distribution (test)", "o"),
                                  (lambda r: r["category"] != "id", ORANGE, "out of distribution", "s")):
        pts = np.array([[r["vm_max_fem"], r["vm_max_nn"]] for r in rows if sel(r)])
        if len(pts):
            ax.scatter(pts[:, 0], pts[:, 1], s=22, color=color, marker=mk, edgecolor="white", linewidth=0.6,
                       label=label, zorder=3)
    ax.set_xlabel("FEM (skfem P2) max von Mises / applied traction")
    ax.set_ylabel("surrogate max von Mises / applied traction")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.set_title("Max von Mises stress: surrogate vs fixed solver", loc="left", color=INK)
    fig.tight_layout()
    files.append(img / "parity_vm_max.png")
    fig.savefig(files[-1])
    plt.close(fig)

    # 2. error by category (median, p10-p90)
    cats = list(rep["summary"])
    cats = [c for c in cats if c != "ood_all"]
    fig, ax = plt.subplots(figsize=(6.4, 0.42 * len(cats) + 1.0), dpi=150)
    _style(ax)
    for i, c in enumerate(cats):
        e = np.array([abs(r["vm_max_err"]) for r in rows if r["category"] == c]) * 100
        color = BLUE if c == "id" else ORANGE
        ax.plot([np.percentile(e, 10), np.percentile(e, 90)], [i, i], color=color, linewidth=2, solid_capstyle="round")
        ax.scatter([np.median(e)], [i], s=40, color=color, edgecolor="white", linewidth=0.8, zorder=3)
        fl = rep["summary"][c]["flag_any"]
        ax.text(1.01, i, f"{fl:.0%} flagged", transform=ax.get_yaxis_transform(), va="center", fontsize=8, color=INK2)
    ax.set_yticks(range(len(cats)))
    ax.set_yticklabels(["in distribution (test)" if c == "id" else c.replace("_", " ") for c in cats])
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlabel("|max von Mises error| vs FEM, % (dot: median, bar: 10th-90th percentile)")
    ax.set_title("Where the surrogate holds up, and where it is flagged", loc="left", color=INK)
    fig.tight_layout()
    files.append(img / "error_by_category.png")
    fig.savefig(files[-1])
    plt.close(fig)

    # 3. field comparisons: median in-distribution case and the worst flagged OOD case
    cmap_v = LinearSegmentedColormap.from_list("blue", BLUE_RAMP)
    cmap_e = LinearSegmentedColormap.from_list("orange", ORANGE_RAMP)
    by_name = {Path(d["path"]).stem: d for d in samples}
    idr = sorted([r for r in rows if r["category"] == "id"], key=lambda r: r["rel_l2_von_mises"])
    oodr = sorted([r for r in rows if r["category"] != "id"], key=lambda r: abs(r["vm_max_err"]))
    picks = []
    if idr:
        picks.append(("fields_in_distribution.png", idr[len(idr) // 2], "in distribution (median case)"))
    if oodr:
        picks.append(("fields_out_of_distribution.png", oodr[-1], f"out of distribution: {oodr[-1]['category']} "
                                                                    f"(worst max-von-Mises error)"))
    for fname, r, title in picks:
        d = by_name[r["name"]]
        p = d["points"]
        tri = Triangulation(p[:, 0], p[:, 1], _subtri(d["cells"], d["cell_type"]))
        t, y = d["targets"][:, 5], preds[r["name"]][:, 5]
        vmax = float(max(t.max(), y.max()))
        fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.2), dpi=150)
        for ax, v, ttl, cm, vm in ((axs[0], t, "FEM von Mises / traction", cmap_v, vmax),
                                   (axs[1], y, "surrogate von Mises / traction", cmap_v, vmax),
                                   (axs[2], np.abs(y - t), "|surrogate - FEM|", cmap_e, None)):
            pc = ax.tripcolor(tri, v, shading="gouraud", cmap=cm, vmin=0.0, vmax=vm)
            ax.set_aspect("equal")
            ax.set_axis_off()
            ax.set_title(ttl, fontsize=9, color=INK)
            cb = fig.colorbar(pc, ax=ax, fraction=0.046, pad=0.02)
            cb.ax.tick_params(labelsize=7, colors=INK2)
            cb.outline.set_visible(False)
        fig.suptitle(f"{title}: rel. L2 error {r['rel_l2_von_mises']:.1%}, max error {r['vm_max_err']:+.1%}, "
                     f"OOD flagged: {'yes' if (r['flag_range'] or r['flag_latent']) else 'no'}",
                     x=0.01, ha="left", fontsize=10, color=INK)
        fig.tight_layout()
        files.append(img / fname)
        fig.savefig(files[-1])
        plt.close(fig)

    # 4. training curve
    hist = blob["training"]["history"]
    fig, ax = plt.subplots(figsize=(5.0, 3.2), dpi=150)
    _style(ax)
    ep = [h["epoch"] for h in hist]
    ax.plot(ep, [h["train_loss"] for h in hist], color=BLUE, linewidth=2, label="train")
    ax.plot(ep, [h["val_loss"] for h in hist], color=ORANGE, linewidth=2, label="validation")
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE of standardised targets")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title(f"Training ({blob['training']['device']}, {hist[-1]['elapsed_s'] / 60:.0f} min)", loc="left",
                 color=INK)
    fig.tight_layout()
    files.append(img / "training_curve.png")
    fig.savefig(files[-1])
    plt.close(fig)
    return files


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--img", default=None, help="directory for PNG plots (e.g. docs/img/surrogate)")
    ap.add_argument("--n-timing", type=int, default=12, help="in-distribution cases for backend timings")
    ap.add_argument("--n-timing-ood", type=int, default=2, help="cases per OOD category for timings")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--scaling", action="store_true", help="also time FEM vs surrogate on finer meshes")
    ap.add_argument("--benchmarks", action="store_true", help="also run on the registered 2D benchmark problems")
    args = ap.parse_args(argv)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    rep = evaluate(args.ckpt, args.data, args.out, args.img, args.n_timing, args.n_timing_ood, args.device,
                   args.threads, scaling=args.scaling, benchmarks=args.benchmarks)
    print(markdown_tables(rep))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
