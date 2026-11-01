"""Video of a crush-to-failure run (snap.py output), rendered on the CPU (llvmpipe).

usage: render_snap.py SNAP.npz OUT.mp4 TITLE [MODEL_NOTE]
Panels: outside (von Mises up to the UTS, broken elements red), cut in half (same), force vs platen travel.
Deformation at true scale (1x)."""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyvista as pv

UTS_MPA, FPS, PER_STEP, HOLD = 46.0, 30, 5, 45
CELL_TET = 10


def grid(points, tets):
    cells = np.hstack([np.full((tets.shape[0], 1), 4), tets]).ravel()
    return pv.UnstructuredGrid(cells, np.full(tets.shape[0], CELL_TET, np.uint8), points * 1e3)


def state(d, k):
    g = grid(d["points"] + d["u"][k], d["tets"])
    g.cell_data["vm"] = d["vm"][k] / 1e6
    g.cell_data["failed"] = d["failed"][k].astype(np.uint8)
    return g


def half(g, axis):
    side = (axis + 1) % 3
    keep = np.where(g.cell_centers().points[:, side] < g.center[side])[0]
    return g.extract_cells(keep)


def draw_part(pl, g, title, bar=False):
    intact = g.extract_cells(np.where(g.cell_data["failed"] == 0)[0])
    broken = g.extract_cells(np.where(g.cell_data["failed"] == 1)[0])
    pl.add_mesh(intact.extract_surface(), scalars="vm", clim=[0, UTS_MPA], cmap="turbo", show_scalar_bar=bar,
                scalar_bar_args=dict(title="von Mises (MPa)   red = broken", color="white", position_x=0.2,
                                     position_y=0.05, width=0.6, title_font_size=14, label_font_size=12))
    if broken.n_cells:
        pl.add_mesh(broken.extract_surface(), color="#ff2a2a")
    pl.add_text(title, font_size=12, position="upper_left", color="white")


def chart(d, k):
    c = pv.Chart2D(size=(0.94, 0.55), loc=(0.03, 0.2), x_label="platen travel (mm)", y_label="force (N)")
    c.background_color = (0.12, 0.14, 0.19, 1.0)
    x, y = np.asarray(d["delta"]) * 1e3, np.asarray(d["force"])
    c.line(x, y, color="#5a6a85", width=2.0)
    c.line(x[: k + 1], y[: k + 1], color="#ffe14d", width=4.0)
    c.scatter([x[k]], [y[k]], color="#ff2a2a", size=12, style="o")
    for ax in (c.x_axis, c.y_axis):
        ax.label_size, ax.tick_label_size = 14, 12
    c.x_axis.range = [0, x.max() * 1.05]; c.y_axis.range = [0, y.max() * 1.1]
    return c


def frame(d, k, axis, cams, title, note):
    pl = pv.Plotter(off_screen=True, shape=(1, 3), window_size=(1920, 1080), border=False)
    g = state(d, k)
    for c, (mesh, label) in enumerate(((g, "outside"), (half(g, axis), "inside: cut in half"))):
        pl.subplot(0, c); pl.set_background("#1d2330", top="#3a4356")
        draw_part(pl, mesh, label, bar=(c == 0))
        if cams[c] is None:
            pl.camera_position = "iso"; pl.camera.Azimuth(-30 if c else 0); pl.reset_camera(); pl.camera.zoom(1.2)
            cams[c] = pl.camera_position
        else:
            pl.camera_position = cams[c]
    pl.subplot(0, 2); pl.set_background("#1d2330", top="#3a4356")
    pl.add_chart(chart(d, k))
    f, dl = d["force"][k], d["delta"][k] * 1e3
    nb, ny = int(d["failed"][k].sum()), int(d["yielded"][k].sum())
    pl.add_text(f"force vs crush\n{dl:6.2f} mm   {f:8.0f} N\nyielded {ny} / broken {nb} elements",
                font_size=12, position="upper_left", color="white")
    pl.subplot(0, 1)
    pl.add_text(f"{title}\n{note}", font_size=10, position="lower_left", color="white")
    return pl


def main(npz, out, title, note):
    d = dict(np.load(npz))
    axis = int(d["axis"]); n = len(d["delta"]); out = Path(out)
    frames = out.parent / (out.stem + "_frames"); frames.mkdir(parents=True, exist_ok=True)
    cams, i = [None, None], 0
    for k in range(n):
        reps = PER_STEP + (HOLD if k == n - 1 else 0)
        pl = frame(d, k, axis, cams, title, note)
        pl.screenshot(str(frames / f"f{i:04d}.png")); pl.close()
        for r in range(1, reps):
            os.link(frames / f"f{i:04d}.png", frames / f"f{i + r:04d}.png")
        i += reps
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", str(frames / "f%04d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(out)], check=True)
    print(f"wrote {out}: {n} steps, {i} frames")


if __name__ == "__main__":
    main(*sys.argv[1:4], sys.argv[4] if len(sys.argv) > 4 else "")
