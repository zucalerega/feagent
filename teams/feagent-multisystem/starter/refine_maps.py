"""Where each mesh system refines abc_00000065: element-size maps on the mid-thickness cut (z = 0),
size relative to the consensus of the set, and the budget share per region.

usage: refine_maps.py OUT_DIR SET_NAME LABEL=MSH [LABEL=MSH ...]
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from matplotlib.collections import PolyCollection
from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize

SOLID = {10: "tet", 24: "tet", 12: "hex", 25: "hex", 29: "hex"}
BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SEQ = LinearSegmentedColormap.from_list("seq", BLUES[::-1])           # dark = fine
DIV = LinearSegmentedColormap.from_list("div", ["#184f95", "#3987e5", "#9ec5f4", "#f0efec",
                                                "#f2b0ae", "#e34948", "#a32a2a"])
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "text.color": INK, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": GRID})
H_LIM = (0.5, 5.0)


def load(path):
    g = pv.read(path)
    keep = np.isin(g.celltypes, list(SOLID))
    g = g.extract_cells(np.where(keep)[0]).linear_copy()
    g.points = g.points * 1e3
    vol = np.abs(g.compute_cell_sizes(length=False, area=False)["Volume"])
    hexes = np.isin(g.celltypes, [12, 25, 29])
    h = np.where(hexes, np.cbrt(vol), np.cbrt(6 * np.sqrt(2) * vol))   # edge of the regular cell with that volume
    g.cell_data["h"] = h
    g.cell_data["vol"] = vol
    return g, ("hex" if hexes.mean() > 0.5 else "tet")


def regions(c):
    r = np.hypot(c[:, 0], c[:, 1])
    neck = (r > 19.67) & (r < 27.77) & (c[:, 1] > -15) & (c[:, 1] < 12)
    tip = (c[:, 1] > 19.5) & (np.abs(c[:, 0]) > 12.9) & (np.abs(c[:, 0]) < 23.8)
    base = (c[:, 1] < -27.0) & ~neck
    return {"neck": neck, "tip pocket": tip & ~neck, "base": base, "rest": ~(neck | tip | base)}


TET_EDGES = [(0, 1), (1, 2), (0, 2), (0, 3), (1, 3), (2, 3)]
HEX_EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]


def cut_polys(g, z0=0.0):
    """Cross-section of every solid cell with the plane z = z0, in numpy (convex cells -> convex polygons)."""
    polys, hs = [], []
    conn = g.cells_dict
    for ct, cells in conn.items():
        edges = HEX_EDGES if ct in (12,) else TET_EDGES if ct in (10,) else None
        if edges is None:
            continue
        cid = np.where(g.celltypes == ct)[0]
        P = g.points[cells]                       # (n, k, 3)
        z = P[:, :, 2] - z0
        cross = (z.min(1) < 0) & (z.max(1) > 0)
        P, z, cid = P[cross], z[cross], cid[cross]
        for a, b in edges:
            pass
        for i in range(len(P)):
            pts = []
            for a, b in edges:
                za, zb = z[i, a], z[i, b]
                if (za < 0) != (zb < 0):
                    t = za / (za - zb)
                    pts.append(P[i, a, :2] + t * (P[i, b, :2] - P[i, a, :2]))
            if len(pts) >= 3:
                q = np.array(pts); c = q.mean(0)
                q = q[np.argsort(np.arctan2(q[:, 1] - c[1], q[:, 0] - c[0]))]
                polys.append(q); hs.append(g.cell_data["h"][cid[i]])
    return polys, np.array(hs)


def main():
    out, name, pairs = Path(sys.argv[1]), sys.argv[2], [a.split("=", 1) for a in sys.argv[3:]]
    out.mkdir(parents=True, exist_ok=True)
    meshes, cuts = {}, {}
    for lab, p in pairs:
        meshes[lab] = load(p)
        cuts[lab] = cut_polys(meshes[lab][0])
    labs = list(meshes)
    n = len(labs); cols = min(4, n); rows = int(np.ceil(n / cols))

    # 1) element size on the mid-thickness cut
    fig, axs = plt.subplots(rows, cols, figsize=(4.2 * cols, 4.9 * rows + 0.8), squeeze=False)
    norm = LogNorm(*H_LIM)
    for ax, lab in zip(axs.flat, labs):
        g, kind = meshes[lab]
        polys, h = cuts[lab]
        ax.add_collection(PolyCollection(polys, array=h, cmap=SEQ, norm=norm, edgecolors="#ffffff", linewidths=0.12))
        ax.set_xlim(-31, 31); ax.set_ylim(-45, 35); ax.set_aspect("equal"); ax.axis("off")
        c = g.cell_centers().points; rg = regions(c)
        med = {k: np.median(g.cell_data["h"][m]) for k, m in rg.items() if m.any()}
        ax.set_title(f"{lab}\n{g.n_points:,} nodes ({kind})", fontsize=11, loc="left")
        ax.text(-31, -47, f"median size: neck {med['neck']:.1f} mm, tip {med['tip pocket']:.1f} mm, "
                f"base {med['base']:.1f} mm", fontsize=8, color=INK2, va="top")
    for ax in list(axs.flat)[n:]:
        ax.axis("off")
    cax = fig.add_axes([0.25, 0.035, 0.5, 0.014])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=SEQ), cax=cax, orientation="horizontal")
    cb.set_label("element size (mm, edge of a regular element of the same volume; darker = finer)")
    cb.minorticks_off(); cb.set_ticks([0.5, 1, 2, 3, 5]); cb.set_ticklabels(["0.5", "1", "2", "3", "5"])
    fig.suptitle(f"abc_00000065, {name}: element size on the mid-thickness cut (z = 0)", x=0.02, ha="left",
                 fontsize=14, fontweight="bold")
    fig.savefig(out / f"{name}_size_maps.png", dpi=170, bbox_inches="tight", facecolor="#fcfcfb"); plt.close(fig)

    # 2) size relative to the consensus (median over the set) at the same point
    xs, ys = np.arange(-30.5, 30.5, 0.25), np.arange(-44.0, 34.0, 0.25)
    X, Y = np.meshgrid(xs, ys); P = np.c_[X.ravel(), Y.ravel(), np.zeros(X.size)]
    tlabs = [l for l in labs if meshes[l][1] == "tet"]
    H = []
    for lab in tlabs:
        g = meshes[lab][0]
        cid = g.find_containing_cell(P)
        H.append(np.where(cid >= 0, g.cell_data["h"][np.clip(cid, 0, None)], np.nan))
    H = np.array(H)
    inside = np.isfinite(H).sum(0) >= max(2, len(tlabs) - 1)
    cons = np.nanmedian(np.where(inside, H, np.nan), axis=0)
    n2 = len(tlabs); cols2 = min(4, n2); rows2 = int(np.ceil(n2 / cols2))
    fig, axs = plt.subplots(rows2, cols2, figsize=(4.2 * cols2, 4.9 * rows2 + 0.8), squeeze=False)
    dn = Normalize(-1, 1)
    for k, (ax, lab) in enumerate(zip(axs.flat, tlabs)):
        R = np.log2(H[k] / cons).reshape(X.shape)
        R[~inside.reshape(X.shape)] = np.nan
        ax.imshow(R, origin="lower", extent=(xs[0], xs[-1], ys[0], ys[-1]), cmap=DIV, norm=dn, interpolation="nearest")
        ax.set_xlim(-31, 31); ax.set_ylim(-45, 35); ax.set_aspect("equal"); ax.axis("off")
        fin = np.nanmean(R < -0.5) * 100 / max(np.nanmean(np.isfinite(R)), 1e-9)
        ax.set_title(f"{lab}", fontsize=11, loc="left")
    for ax in list(axs.flat)[n2:]:
        ax.axis("off")
    if n2 < n:
        fig.text(0.02, 0.005, "Hex meshes are left out of this comparison: a linear hex and a quadratic tet of the same size "
                 "do not resolve the same field.", fontsize=9, color=INK2)
    cax = fig.add_axes([0.25, 0.035, 0.5, 0.014])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=dn, cmap=DIV), cax=cax, orientation="horizontal")
    cb.set_ticks([-1, -0.5, 0, 0.5, 1]); cb.set_ticklabels(["2x finer", "1.4x finer", "same as the set's median",
                                                              "1.4x coarser", "2x coarser"])
    fig.suptitle(f"abc_00000065, {name}: where each tet system refines more (blue) or less (red) than the median of the tet systems",
                 x=0.02, ha="left", fontsize=14, fontweight="bold")
    fig.savefig(out / f"{name}_vs_consensus.png", dpi=170, bbox_inches="tight", facecolor="#fcfcfb"); plt.close(fig)

    # 3) element density per region, relative to that mesh's whole-part average
    table = {}
    for lab in labs:
        g = meshes[lab][0]; c = g.cell_centers().points; rg = regions(c)
        V, N = g.cell_data["vol"].sum(), g.n_cells
        table[lab] = {k: float((m.sum() / N) / (g.cell_data["vol"][m].sum() / V)) for k, m in rg.items()}
    (out / f"{name}_regions.json").write_text(json.dumps(table, indent=1))
    reg = ["neck", "tip pocket", "base", "rest"]
    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
    fig, ax = plt.subplots(figsize=(1.3 * n + 3, 4.6))
    w = 0.8 / len(reg); x = np.arange(n)
    for j, (r_, col) in enumerate(zip(reg, colors)):
        v = [table[l][r_] for l in labs]
        ax.bar(x + (j - 1.5) * w, v, w * 0.92, color=col, label=r_)
    ax.axhline(1, color=INK2, lw=1, ls="--"); ax.text(-0.45, 1.03, "part average", color=INK2, fontsize=8, ha="left")
    ax.set_xticks(x); ax.set_xticklabels(labs, rotation=0, fontsize=9)
    ax.set_ylabel("elements per volume, relative to the part average")
    ax.spines[["top", "right"]].set_visible(False); ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True)
    ax.legend(ncol=4, frameon=False, loc="upper left", bbox_to_anchor=(0, 1.12))
    ax.set_title(f"abc_00000065, {name}: where each system spends its elements", loc="left", fontsize=13,
                 fontweight="bold", pad=28)
    fig.savefig(out / f"{name}_region_density.png", dpi=170, bbox_inches="tight", facecolor="#fcfcfb"); plt.close(fig)
    print(json.dumps(table, indent=1))


if __name__ == "__main__":
    main()
