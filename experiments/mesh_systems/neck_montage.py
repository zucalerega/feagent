#!/usr/bin/env python
"""Montage of abc_00000065 meshes: per mesh a full front view (z face, surface edges) and a zoom on the right arm's
neck and tip pocket, with node count and the neck / tip edge sizes (neck_tip_sizes.py JSON).

    python experiments/mesh_systems/neck_montage.py SIZES.json OUT.png [--order a,b,c]

CPU rendering (llvmpipe); never uses the GPU.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _k, _v in (("LIBGL_ALWAYS_SOFTWARE", "1"), ("GALLIUM_DRIVER", "llvmpipe"), ("__GLX_VENDOR_LIBRARY_NAME", "mesa")):
    os.environ.setdefault(_k, _v)
import numpy as np  # noqa: E402
import pyvista as pv  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

pv.OFF_SCREEN = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
W, H = 700, 900


def surface(d: Path) -> pv.PolyData:
    from feacore.solver._ccx_mesh import build_model, read_gmsh
    gm = read_gmsh(d / "mesh.msh")
    order = 2 if any(ct == "tetra10" for ct, _ in gm.groups["domain"]) else 1
    m = build_model(gm, 3, order)
    ct = pv.CellType.HEXAHEDRON if m.n_corners == 8 else pv.CellType.TETRA
    g = pv.UnstructuredGrid({ct: m.conn[:, :m.n_corners]}, m.X * 1e3)
    return g.extract_surface()


def shot(surf, focus=None, scale=None) -> Image.Image:
    pl = pv.Plotter(off_screen=True, window_size=(W, H))
    pl.set_background("white")
    pl.add_mesh(surf, color="#cfd8dc", show_edges=True, edge_color="#37474f", line_width=0.6)
    pl.camera_position = "xy"
    pl.enable_parallel_projection()
    if focus is not None:
        pl.camera.focal_point = focus
        pl.camera.position = (focus[0], focus[1], focus[2] + 200)
        pl.camera.parallel_scale = scale
    img = Image.fromarray(pl.screenshot(return_img=True))
    pl.close()
    return img


def font(sz):
    f = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    return ImageFont.truetype(f, sz) if Path(f).exists() else ImageFont.load_default()


def main() -> int:
    sizes = json.loads(Path(sys.argv[1]).read_text())
    out = Path(sys.argv[2])
    order = sys.argv[sys.argv.index("--order") + 1].split(",") if "--order" in sys.argv else list(sizes)
    tiles = []
    for lab in order:
        r = sizes[lab]
        s = surface(Path(r["dir"]))
        full = shot(s)
        zoom = shot(s, focus=(21.0, 5.0, 8.1), scale=19.0)         # right arm: neck (y -15..12) up to the tip
        t = Image.new("RGB", (2 * W, H + 150), "white")
        t.paste(full, (0, 150))
        t.paste(zoom, (W, 150))
        d = ImageDraw.Draw(t)
        d.text((14, 8), f"{lab}: {r['nodes']:,} nodes, {r['cells']:,} {r['cell_type']}", fill="black", font=font(34))
        nk, tp = r["neck"], r["tip"]
        d.text((14, 56), f"neck: median edge {nk['median_edge_mm']} mm (min {nk['min_edge_mm']}), "
                         f"{nk['across_wall']} cells across the 8.1 mm wall", fill="#333", font=font(24))
        d.text((14, 92), f"tip pocket: median edge {tp['median_edge_mm']} mm (min {tp['min_edge_mm']})   |   right: zoom on "
                         f"the right arm (neck + tip)", fill="#333", font=font(24))
        tiles.append(t)
        print(lab, flush=True)
    cols = 2
    rows = (len(tiles) + cols - 1) // cols
    tw, th = tiles[0].size
    M = Image.new("RGB", (cols * tw, rows * th), "white")
    for i, t in enumerate(tiles):
        M.paste(t, ((i % cols) * tw, (i // cols) * th))
    M.save(out)
    print(out, M.size)
    return 0


if __name__ == "__main__":
    sys.exit(main())
