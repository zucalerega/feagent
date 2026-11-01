"""Render a tet mesh in the style of an Ansys Mechanical mesh view (headless, matplotlib only).

    python scripts/render_mesh.py MESH.msh [--out mesh.png] [--title NAME] [--cut y] [--at 0.5]

Left: the shaded part with element edges. Right: a section plane (normal to --cut, at fraction --at
of the bounding box) that keeps the cells on one side, exposing the tetrahedra inside. Gradient
background, XYZ triad and a statistics box as in Ansys. Works on tet4 / tet10 gmsh meshes.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import meshio  # noqa: E402
import numpy as np  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

BODY = np.array([0.60, 0.72, 0.74])        # muted grey-teal, like the Ansys mesh body colour
CUT = np.array([0.93, 0.80, 0.42])         # section-plane faces
EDGE = (0, 0, 0, 0.55)
TET_FACES = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])


def boundary(tets: np.ndarray, P: np.ndarray):
    """Faces used by exactly one tet of the set, oriented outward. Returns (faces (n,3), owner (n,))."""
    f = tets[:, TET_FACES].reshape(-1, 3)
    owner = np.repeat(np.arange(len(tets)), 4)
    key = np.sort(f, axis=1)
    _, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    keep = cnt[inv.ravel()] == 1
    f, owner = f[keep], owner[keep]
    c = P[tets[owner]].mean(axis=1)
    n = np.cross(P[f[:, 1]] - P[f[:, 0]], P[f[:, 2]] - P[f[:, 0]])
    flip = np.einsum("ij,ij->i", n, P[f].mean(axis=1) - c) < 0
    f[flip] = f[flip][:, [0, 2, 1]]
    return f, owner


def shade(P, faces, base, view_dir, light_dir):
    n = np.cross(P[faces[:, 1]] - P[faces[:, 0]], P[faces[:, 2]] - P[faces[:, 0]])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-300
    diffuse = np.clip(n @ light_dir, 0, 1)
    h = (light_dir + view_dir) / np.linalg.norm(light_dir + view_dir)
    spec = np.clip(n @ h, 0, 1) ** 40
    rgb = np.clip(base * (0.38 + 0.62 * diffuse[:, None]) + 0.25 * spec[:, None], 0, 1)
    front = n @ view_dir > -0.05                  # back-face culling keeps the painter's sort clean
    return rgb, front


def view_vectors(elev, azim):
    e, a = np.radians(elev), np.radians(azim)
    v = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
    light = v + np.array([-0.3, 0.2, 0.6])
    return v, light / np.linalg.norm(light)


def draw(ax, P, groups, elev, azim):
    v, light = view_vectors(elev, azim)
    for faces, base in groups:
        rgb, front = shade(P, faces, base, v, light)
        ax.add_collection3d(Poly3DCollection(P[faces[front]], facecolors=rgb[front], edgecolors=EDGE,
                                             linewidths=0.25))
    lo, hi = P.min(0), P.max(0)
    c, r = (lo + hi) / 2, (hi - lo).max() / 2 * 0.82
    ax.set_xlim(c[0] - r, c[0] + r); ax.set_ylim(c[1] - r, c[1] + r); ax.set_zlim(c[2] - r, c[2] + r)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev, azim)
    ax.set_axis_off()
    ax.patch.set_alpha(0)


def triad(fig, rect, elev, azim):
    ax = fig.add_axes(rect, projection="3d")
    for d, col, lab in zip(np.eye(3), ("#d62728", "#2ca02c", "#1f4fd6"), "XYZ"):
        ax.quiver(0, 0, 0, *d, color=col, linewidth=2, arrow_length_ratio=0.25)
        ax.text(*(d * 1.3), lab, color=col, fontsize=10, fontweight="bold", ha="center", va="center")
    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)
    ax.view_init(elev, azim)
    ax.set_axis_off()
    ax.patch.set_alpha(0)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("mesh")
    ap.add_argument("--out")
    ap.add_argument("--title")
    ap.add_argument("--cut", choices=list("xyz"), default="y")
    ap.add_argument("--at", type=float, default=0.5)
    ap.add_argument("--elev", type=float, default=22)
    ap.add_argument("--azim", type=float, default=-55)
    a = ap.parse_args(argv)

    m = meshio.read(a.mesh)
    P = m.points * 1000.0                          # m -> mm
    ctype = "tetra10" if "tetra10" in m.cells_dict else "tetra"
    tets = m.cells_dict[ctype][:, :4]
    used = np.unique(m.cells_dict[ctype])
    full, _ = boundary(tets, P)

    k = "xyz".index(a.cut)
    lo, hi = P[:, k].min(), P[:, k].max()
    plane = lo + a.at * (hi - lo)
    # keep the half away from the camera so the cut face looks at the viewer
    toward_camera = view_vectors(a.elev, a.azim)[0][k] > 0
    keep = (P[tets, k].mean(axis=1) <= plane) if toward_camera else (P[tets, k].mean(axis=1) >= plane)
    half, _ = boundary(tets[keep], P)
    outer = {tuple(sorted(f)) for f in full}
    is_cut = np.array([tuple(sorted(f)) not in outer for f in half])

    fig = plt.figure(figsize=(15, 7.5), dpi=120)
    bg = fig.add_axes([0, 0, 1, 1], zorder=-1)
    grad = np.linspace(0, 1, 256)[:, None]
    top, bottom = np.array([0.36, 0.52, 0.76]), np.array([0.97, 0.98, 1.0])
    bg.imshow(top + (bottom - top) * grad[..., None], aspect="auto", extent=[0, 1, 0, 1])
    bg.set_axis_off()

    ax1 = fig.add_axes([0.0, 0.04, 0.5, 0.86], projection="3d")
    draw(ax1, P, [(full, BODY)], a.elev, a.azim)
    ax2 = fig.add_axes([0.5, 0.04, 0.5, 0.86], projection="3d")
    draw(ax2, P, [(half[~is_cut], BODY), (half[is_cut], CUT)], a.elev, a.azim)

    name = a.title or Path(a.mesh).stem
    fig.text(0.012, 0.965, name, fontsize=14, fontweight="bold", color="#10233f", va="top")
    fig.text(0.012, 0.925, "Mesh", fontsize=11, color="#10233f", va="top")
    fig.text(0.5, 0.925, f"Section plane: {a.cut} = {plane:.2f} mm", fontsize=11, color="#10233f", va="top")
    stats = (f"Statistics{'':>14}\nNodes      {len(used):>10,}\nElements   {len(tets):>10,}\n"
             f"Type       {'Tet10' if ctype == 'tetra10' else 'Tet4':>10}")
    fig.text(0.985, 0.04, stats, family="monospace", fontsize=9.5, ha="right", va="bottom",
             bbox=dict(boxstyle="square,pad=0.6", fc="white", ec="#6b7a90", lw=0.8))
    triad(fig, [0.005, 0.0, 0.12, 0.2], a.elev, a.azim)
    out = a.out or str(Path(a.mesh).with_suffix(".png"))
    fig.savefig(out)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
