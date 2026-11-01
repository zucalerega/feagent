"""The photo of the broken part next to several simulated parts, each at the same post-peak state: the first
increment after the peak where the force is below --ratio x peak (default 0.5). The view and orientation are
the photo's; intact cells grey, broken cells red, pieces no longer face-connected to the base faint.

    python experiments/physical/break_stills.py PHOTO OUT.png LABEL=RUN_DIR [LABEL=RUN_DIR ...] [--ratio 0.5]
        [--cols 4] [--azim 28] [--elev 15] [--roll -12]

CPU rendering (llvmpipe); follows continued_from chains (render_crush.series).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _k, _v in (("LIBGL_ALWAYS_SOFTWARE", "1"), ("GALLIUM_DRIVER", "llvmpipe"), ("__GLX_VENDOR_LIBRARY_NAME", "mesa")):
    os.environ.setdefault(_k, _v)
import numpy as np  # noqa: E402
import pyvista as pv  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402
from scipy.sparse import coo_matrix  # noqa: E402
from scipy.sparse.csgraph import connected_components  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import render_crush as rc  # noqa: E402

pv.OFF_SCREEN = True
TW, TH = 900, 1200


def font(sz):
    for f in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if Path(f).exists():
            return ImageFont.truetype(f, sz)
    return ImageFont.load_default()


def on_base(conn, alive, pts):
    """Intact cells face-connected to the base (cells touching the lowest y)."""
    faces_l = rc.FACES[12] if conn.shape[1] == 8 else rc.FACES[10]
    ids = np.where(alive)[0]
    faces = np.sort(np.concatenate([conn[ids][:, list(f)] for f in faces_l]), axis=1)
    owner = np.tile(ids, len(faces_l))
    _, inv = np.unique(faces, axis=0, return_inverse=True)
    inv = inv.ravel()
    o = np.argsort(inv, kind="stable")
    same = inv[o][1:] == inv[o][:-1]
    a, b = owner[o][:-1][same], owner[o][1:][same]
    n = len(conn)
    _, lab = connected_components(coo_matrix((np.ones(len(a)), (a, b)), shape=(n, n)), directed=False)
    base = np.abs(pts[:, 1] - pts[:, 1].min()) < 1e-3
    seed = alive & base[conn].any(axis=1)
    return alive & np.isin(lab, np.unique(lab[seed]))


def tile(run: Path, ratio: float, azim: float, elev: float, roll: float):
    rows, _ = rc.series(run)
    F = np.array([r[2] for r in rows])
    ip = int(np.argmax(F))
    j = next((i for i in range(ip, len(rows)) if F[i] < ratio * F[ip]), len(rows) - 1)
    pts, u, vm, st, g = rc.load(rows[j][4])
    conn = np.asarray(g.cells_dict[next(iter(g.cells_dict))])
    nc = 8 if conn.shape[1] == 8 else 4
    conn = conn[:, :nc]
    ct = pv.CellType.HEXAHEDRON if nc == 8 else pv.CellType.TETRA
    grid = pv.UnstructuredGrid({ct: conn}, pts + u)
    keep = on_base(conn, st == 0, pts)
    off = (st == 0) & ~keep
    pl = pv.Plotter(off_screen=True, window_size=(TW, TH))
    pl.set_background("black")
    pl.add_mesh(grid.extract_cells(np.where(keep)[0]).extract_surface(), color="#b9c2bd", specular=0.25)
    if off.any():
        pl.add_mesh(grid.extract_cells(np.where(off)[0]).extract_surface(), color="#b9c2bd", opacity=0.12)
    if (st == 1).any():
        pl.add_mesh(grid.extract_cells(np.where(st == 1)[0]).extract_surface(), color="#e0312b", opacity=0.9)
    pl.camera_position = "xy"
    pl.camera.azimuth, pl.camera.elevation, pl.camera.roll = azim, elev, roll
    pl.camera.zoom(1.05)
    img = Image.fromarray(pl.screenshot(return_img=True))
    pl.close()
    return img, rows[j], F[ip], rows[ip][1]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("photo"), ap.add_argument("out"), ap.add_argument("runs", nargs="+")
    ap.add_argument("--ratio", type=float, default=0.5), ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--azim", type=float, default=28), ap.add_argument("--elev", type=float, default=15)
    ap.add_argument("--roll", type=float, default=-12)
    a = ap.parse_args()
    ph = Image.open(a.photo).convert("RGB")
    w, h = ph.size
    ch = min(h, int(w * TH / TW))                     # crop to the tile's aspect (no squash), keep the middle
    top = max(0, (h - ch) // 2 - h // 12)
    tiles = [ph.crop((0, top, w, top + ch)).resize((TW, TH))]
    caps = ["Real part (Formlabs Alpha Fast) after the crush"]
    for spec in a.runs:
        lab, d = spec.split("=", 1)
        img, row, peak, tpk = tile(Path(d), a.ratio, a.azim, a.elev, a.roll)
        tiles.append(img)
        caps.append(f"{lab}: peak {peak / 1000:.2f} kN at {tpk:.2f} mm; shown at {row[1]:.2f} mm, "
                    f"{row[2] / 1000:.2f} kN ({int(round(100 * row[2] / peak))} % of peak)")
        print(caps[-1], flush=True)
    cols = a.cols
    nrow = (len(tiles) + cols - 1) // cols
    cap_h = 90
    M = Image.new("RGB", (cols * TW, nrow * (TH + cap_h)), "black")
    d = ImageDraw.Draw(M)
    for i, (t, c) in enumerate(zip(tiles, caps)):
        x, y = (i % cols) * TW, (i // cols) * (TH + cap_h)
        M.paste(t, (x, y + cap_h))
        words, line, lines = c.split(" "), "", []
        for w in words:
            if len(line) + len(w) > 52:
                lines.append(line)
                line = ""
            line += w + " "
        lines.append(line)
        for k, ln in enumerate(lines[:2]):
            d.text((x + 14, y + 8 + 38 * k), ln, fill="white", font=font(30))
    M.save(a.out)
    print(a.out, M.size)
    return 0


if __name__ == "__main__":
    sys.exit(main())
