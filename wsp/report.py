"""Figures and tables for the student study.

    python -m wsp.report --evals wsp_runs/eval/base,wsp_runs/eval/sft --names "Base,LoRA SFT" --out wsp/results
    python -m wsp.report --meshes cooks_membrane --evals ... --names ... --out wsp/results   # mesh renders

Colors follow the dataviz reference palette (validated: blue / orange / aqua pass all-pairs CVD checks
in light mode; aqua sits below 3:1 contrast, so every figure ships with a table: results.md).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def load(eval_dir: str | Path) -> tuple[dict, list[dict]]:
    d = Path(eval_dir)
    rows = [json.loads(x) for x in (d / "results.jsonl").read_text().splitlines()]
    return json.loads((d / "summary.json").read_text()), rows


def fig_methods(evals: list[tuple[str, dict]], out: Path) -> Path:
    """Mean x the automesher per method, in-family vs out-of-family held-out parts."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = [n for n, _ in evals]
    fig, ax = plt.subplots(figsize=(7.2, 0.55 * len(names) + 1.4), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    groups = [("in_family", "In-family parts (new sizes of trained families)", SERIES[0]),
              ("out_of_family", "Out-of-family parts (Cook's, LE1, LE11, L-bracket)", SERIES[1])]
    hgt = 0.36
    for gi, (g, label, color) in enumerate(groups):
        ys = [i + (gi - 0.5) * (hgt + 0.04) for i in range(len(names))]
        vals = [s.get(g, {}).get("mean_x", 0.0) for _, s in evals]
        ax.barh(ys, vals, height=hgt, color=color, label=label, edgecolor=SURFACE, linewidth=1)
        for y, v in zip(ys, vals):
            ax.text(v + 0.05, y, f"{v:.2f}x", va="center", ha="left", fontsize=8, color=INK2)
    ax.axvline(1.0, color=MUTED, linewidth=1, linestyle=(0, (3, 3)))
    ax.text(1.0, len(names) - 0.45, " automesher = 1x", color=MUTED, fontsize=8, va="bottom")
    ax.set_yticks(range(len(names)), names, color=INK, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("mean score on held-out benchmarks, x the automesher (higher is better)", color=INK2, fontsize=9)
    ax.legend(frameon=False, fontsize=8, loc="lower right", labelcolor=INK2)
    fig.tight_layout()
    p = out / "fig_methods.png"
    fig.savefig(p, facecolor=SURFACE)
    plt.close(fig)
    return p


def fig_per_problem(evals: list[tuple[str, list[dict]]], out: Path) -> Path:
    """x the automesher per held-out part, up to three methods (dots)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    evals = evals[:3]
    pids = []
    for _, rows in evals:
        for r in rows:
            if r["pid"] not in pids:
                pids.append(r["pid"])
    pids.sort(key=lambda p: (0 if rows_group(evals, p) == "out_of_family" else 1, p))
    fig, ax = plt.subplots(figsize=(7.2, 0.26 * len(pids) + 1.3), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    for k, (name, rows) in enumerate(evals):
        by = {}
        for r in rows:
            by.setdefault(r["pid"], []).append(r["x_auto"])
        xs = [sum(by.get(p, [0])) / max(1, len(by.get(p, [0]))) for p in pids]
        ax.scatter(xs, range(len(pids)), s=26, color=SERIES[k], label=name, zorder=3 + k, edgecolor=SURFACE,
                   linewidth=1.2)
    ax.axvline(1.0, color=MUTED, linewidth=1, linestyle=(0, (3, 3)))
    ax.set_yticks(range(len(pids)), pids, fontsize=7, color=INK)
    ax.invert_yaxis()
    ax.grid(axis="y", color=GRID, linewidth=0.5)
    ax.set_xlabel("x the automesher (0 = invalid answer; dashed line = automesher)", color=INK2, fontsize=9)
    ax.legend(frameon=False, fontsize=8, loc="lower right", labelcolor=INK2)
    fig.tight_layout()
    p = out / "fig_per_problem.png"
    fig.savefig(p, facecolor=SURFACE)
    plt.close(fig)
    return p


def rows_group(evals, pid):
    for _, rows in evals:
        for r in rows:
            if r["pid"] == pid:
                return r["group"]
    return ""


def automesher_mesh(pid: str, path: Path) -> Path:
    """The automesher's mesh (gmsh defaults, order 2), as ace.field_error.automesher builds it."""
    import gmsh

    from ace import fea
    from ace.field_error import _unfold_tet10
    from feacore.geometry.gmsh_tools import ensure_domain_group, gmsh_session
    p = fea.get_problem(pid)
    with gmsh_session(p.id):
        p.build_geometry()
        ensure_domain_group(p.dim)
        gmsh.model.mesh.generate(p.dim)
        gmsh.model.mesh.setOrder(2)
        if "gold" in p.tags or "physical" in p.tags:
            _unfold_tet10()
        gmsh.write(str(path))
    return path


def _mesh_xy(msh):
    import meshio
    import numpy as np
    m = meshio.read(msh)
    tri = np.vstack([c.data[:, :3] for c in m.cells if c.type.startswith("triangle")])
    return m.points[:, :2], tri


def draw_mesh(ax, msh: str | Path, title: str = "", window=None) -> None:
    """Element edges of a 2D mesh (corner nodes; quadratic cells drawn straight), optionally zoomed."""
    import numpy as np
    from matplotlib.collections import LineCollection
    P, tri = _mesh_xy(msh)
    e = np.unique(np.sort(np.vstack([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]]), axis=1), axis=0)
    ax.add_collection(LineCollection(P[e], colors=INK, linewidths=0.25 if window is None else 0.4))
    if window is None:
        ax.set_xlim(P[:, 0].min(), P[:, 0].max())
        ax.set_ylim(P[:, 1].min(), P[:, 1].max())
    else:
        ax.set_xlim(window[0], window[2])
        ax.set_ylim(window[1], window[3])
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(window is not None)
        sp.set_color(AXIS)
    if title:
        ax.set_title(title, fontsize=8, color=INK, loc="left")


def _zoom_window(msh, frac: float = 0.12):
    """A square window around the smallest elements of a mesh (where the recipe refined)."""
    import numpy as np
    P, tri = _mesh_xy(msh)
    X = P[tri]
    area = 0.5 * np.abs((X[:, 1, 0] - X[:, 0, 0]) * (X[:, 2, 1] - X[:, 0, 1])
                        - (X[:, 2, 0] - X[:, 0, 0]) * (X[:, 1, 1] - X[:, 0, 1]))
    c = X.mean(axis=1)
    small = c[area <= np.quantile(area, 0.03)]
    centre = np.median(small, axis=0)
    # the densest cluster of small cells: the one nearest to most others
    d = np.linalg.norm(small[:, None, :] - small[None, :, :], axis=2)
    diag = float(np.linalg.norm(np.ptp(P, axis=0)))
    centre = small[np.argmax((d < frac * diag).sum(axis=1))]
    half = frac * diag / 2
    lo, hi = P.min(axis=0), P.max(axis=0)
    win = []
    for i in range(2):          # keep the window inside the part's bounding box (same size)
        a, b = centre[i] - half, centre[i] + half
        if b - a >= hi[i] - lo[i]:
            a, b = (lo[i] + hi[i]) / 2 - half, (lo[i] + hi[i]) / 2 + half
        elif a < lo[i]:
            a, b = lo[i], lo[i] + 2 * half
        elif b > hi[i]:
            a, b = hi[i] - 2 * half, hi[i]
        win.append((a, b))
    return [win[0][0], win[1][0], win[0][1], win[1][1]]


def fig_meshes(pid: str, panels: list[tuple[str, dict]], out: Path) -> Path:
    """The automesher's mesh next to each method's mesh for one part (score, DOFs, error), with a zoom
    row on the region the last panel refined most."""
    import tempfile

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Rectangle

    from ace.field_error import automesher
    a = automesher(pid)
    with tempfile.TemporaryDirectory() as tmp:
        meshes = [(f"Automesher: 1.00x\n{a['dofs']} DOFs, error {a['error']:.2%}", automesher_mesh(pid, Path(tmp) / "a.msh"))]
        for name, r in panels:
            ok = r.get("mesh_path") and Path(r["mesh_path"]).exists()
            meshes.append((f"{name}: {r['x_auto']:.2f}x\n{r['dofs']} DOFs, error {r['e']:.2%}" if ok
                           else f"{name}: invalid\n{(r.get('error') or '')[:44]}", r["mesh_path"] if ok else None))
        P, _ = _mesh_xy(meshes[0][1])
        w, h = np.ptp(P, axis=0)
        n = len(meshes)
        col_w = 3.2
        full_h = float(np.clip(col_w * h / w, 0.7, 3.6))
        win = next((_zoom_window(m) for _, m in reversed(meshes) if m), None)
        fig, axes = plt.subplots(2, n, figsize=(col_w * n, full_h + 2.6 + 0.6), dpi=220,
                                 gridspec_kw={"height_ratios": [full_h, 2.6]}, squeeze=False)
        fig.patch.set_facecolor(SURFACE)
        for k, (title, m) in enumerate(meshes):
            top, bot = axes[0][k], axes[1][k]
            if m is None:
                top.axis("off")
                bot.axis("off")
                top.set_title(title, fontsize=8, color=INK, loc="left")
                continue
            draw_mesh(top, m, title)
            if win is not None:
                top.add_patch(Rectangle(win[:2], win[2] - win[0], win[3] - win[1], fill=False, edgecolor=SERIES[1],
                                        linewidth=0.8))
                draw_mesh(bot, m, window=win)
                for sp in bot.spines.values():
                    sp.set_color(SERIES[1])
        fig.suptitle(f"{pid}: the full part (top) and the boxed region enlarged (bottom)", fontsize=9, color=INK2,
                     x=0.01, ha="left")
        fig.tight_layout()
        p = out / f"fig_mesh_{pid}.png"
        fig.savefig(p, facecolor=SURFACE)
        plt.close(fig)
    return p


def table_md(evals: list[tuple[str, dict, list[dict]]]) -> str:
    lines = ["| method | all | in-family | out-of-family | valid | beats automesher |",
             "|---|---|---|---|---|---|"]
    for name, s, _ in evals:
        g = lambda k: s.get(k, {}).get("mean_x")  # noqa: E731
        lines.append(f"| {name} | {g('all') or 0:.2f}x | {g('in_family') or 0:.2f}x | {g('out_of_family') or 0:.2f}x | "
                     f"{s['all']['valid']:.0%} | {s['all']['beats_auto']:.0%} |")
    pids = sorted({r["pid"] for _, _, rows in evals for r in rows})
    lines += ["", "| part | " + " | ".join(n for n, _, _ in evals) + " |", "|---|" + "---|" * len(evals)]
    for p in pids:
        cells = []
        for _, _, rows in evals:
            xs = [r["x_auto"] for r in rows if r["pid"] == p]
            cells.append(f"{sum(xs) / len(xs):.2f}" if xs else "")
        lines.append(f"| {p} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"




FAMILY_GROUPS = [("Notched bars, lugs", ("notched_bar", "clevis")), ("Stepped shafts", ("stepped_shaft",)),
                 ("Plates with holes", ("plate_hole", "kirsch")), ("Vessels, Lame, beams", ("pressure_vessel", "lame",
                                                                                           "cantilever"))]


def teacher_stats(demo_dir: str | Path) -> list[dict]:
    """Per training part: the teacher's first scored recipe, its best, and how many it tried."""
    rows = []
    for f in sorted(Path(demo_dir).glob("*.jsonl")):
        att = [json.loads(x) for x in f.read_text().splitlines() if x.strip()]
        xs = [a.get("x_auto") or 0.0 for a in att]
        if xs:
            rows.append({"pid": f.stem, "first": xs[0], "best": max(xs), "attempts": len(xs),
                         "group": next((g for g, keys in FAMILY_GROUPS if any(k in f.stem for k in keys)), "other")})
    return rows


def fig_teacher(demo_dir: str | Path, out: Path) -> Path:
    """Mean x the automesher of the teacher's first recipe vs its best verified one, per family group."""
    import statistics

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = teacher_stats(demo_dir)
    groups = [g for g, _ in FAMILY_GROUPS if any(r["group"] == g for r in rows)]
    fig, ax = plt.subplots(figsize=(7.2, 0.6 * len(groups) + 1.4), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    hgt = 0.36
    for k, (key, label) in enumerate((("first", "Teacher's first recipe"), ("best", "Teacher's best, solver-verified"))):
        ys = [i + (k - 0.5) * (hgt + 0.04) for i in range(len(groups))]
        vals = [statistics.mean(r[key] for r in rows if r["group"] == g) for g in groups]
        ax.barh(ys, vals, height=hgt, color=SERIES[k], label=label, edgecolor=SURFACE, linewidth=1)
        for y, v in zip(ys, vals):
            ax.text(v + 0.06, y, f"{v:.2f}x", va="center", fontsize=8, color=INK2)
    ax.axvline(1.0, color=MUTED, linewidth=1, linestyle=(0, (3, 3)))
    ax.text(1.0, len(groups) - 0.45, " automesher = 1x", color=MUTED, fontsize=8, va="bottom")
    counts = {g: sum(r["group"] == g for r in rows) for g in groups}
    ax.set_yticks(range(len(groups)), [f"{g} ({counts[g]} parts)" for g in groups], fontsize=9, color=INK)
    ax.invert_yaxis()
    ax.set_xlabel("mean x the automesher on training parts", color=INK2, fontsize=9)
    ax.legend(frameon=False, fontsize=8, loc="lower right", labelcolor=INK2)
    fig.tight_layout()
    p = out / "fig_teacher.png"
    fig.savefig(p, facecolor=SURFACE)
    plt.close(fig)
    return p


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--evals", required=True)
    ap.add_argument("--names", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--meshes", default="", help="comma-separated part ids to render")
    ap.add_argument("--mesh-methods", default="", help="names of the methods to show in mesh renders")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    names = [n.strip() for n in a.names.split(",")]
    loaded = [(n, *load(d)) for n, d in zip(names, a.evals.split(","))]
    fig_methods([(n, s) for n, s, _ in loaded], out)
    pick = [n.strip() for n in a.mesh_methods.split(",") if n.strip()] or names[:3]
    fig_per_problem([(n, rows) for n, _, rows in loaded if n in pick], out)
    (out / "results.md").write_text(table_md(loaded))
    for pid in [p for p in a.meshes.split(",") if p]:
        panels = []
        for n, _, rows in loaded:
            if n in pick:
                r = next((r for r in rows if r["pid"] == pid), None)
                if r is not None:
                    panels.append((n, r))
        fig_meshes(pid, panels, out)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
