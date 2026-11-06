"""Close-up of the neck crack forming in a crush.py run of abc_00000065, in the original look (von Mises up to the
UTS, broken cells red; CPU / llvmpipe).

    python experiments/physical/neck_closeup.py RUN_DIR --out OUT.mp4 --label "system" [--seconds 8] [--fps 30]
        [--end-ratio 0.5] [--before-mm 0.15] [--size 1280x720]

The camera looks at the outer face and the front (+z) of the arm whose neck cracks first (side of the first failed neck cells; the
neck is the ring wall between bore and outer arc, -15 mm < y < 12 mm). It frames that neck at true scale and
follows its deformed position. Time window: from --before-mm of platen travel before the first failed neck
cell, to the first increment after the peak where the force is below --end-ratio x peak (or the run's end).
The increments in that window are spread evenly over --seconds. Follows continued_from chains (render_crush.series).
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _k, _v in (("LIBGL_ALWAYS_SOFTWARE", "1"), ("GALLIUM_DRIVER", "llvmpipe"), ("__GLX_VENDOR_LIBRARY_NAME", "mesa")):
    os.environ.setdefault(_k, _v)
import numpy as np  # noqa: E402
import pyvista as pv  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import render_crush as rc  # noqa: E402

pv.OFF_SCREEN = True


def neck_mask(cen_mm: np.ndarray) -> np.ndarray:
    r = np.hypot(cen_mm[:, 0], cen_mm[:, 1])
    return (r > 19.5) & (r < 28.0) & (cen_mm[:, 1] > -15) & (cen_mm[:, 1] < 12)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run"), ap.add_argument("--out", required=True), ap.add_argument("--label", default="")
    ap.add_argument("--seconds", type=float, default=8.0), ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--end-ratio", type=float, default=0.5), ap.add_argument("--before-mm", type=float, default=0.15)
    ap.add_argument("--size", default="1280x720")
    a = ap.parse_args()
    W, H = (int(v) for v in a.size.split("x"))
    rows, _ = rc.series(Path(a.run))
    steps = [r[0] for r in rows]
    travel = np.array([r[1] for r in rows])
    force = np.array([r[2] for r in rows])
    nfail = np.array([r[3] for r in rows])
    ip = int(np.argmax(force))
    end = next((i for i in range(ip, len(rows)) if force[i] < a.end_ratio * force[ip]), len(rows) - 1)
    # first failed neck cell: scan the .vtus where n_failed grows
    pts0, _, _, _, g0 = rc.load(rows[0][4])
    cells = np.asarray(g0.cells_dict[next(iter(g0.cells_dict))])
    nc = 8 if cells.shape[1] == 8 else 4
    cen = pts0[cells[:, :nc]].mean(axis=1)
    neck = neck_mask(cen)
    first, side = None, 1.0
    for i in range(len(rows)):
        if nfail[i] == 0:
            continue
        _, _, _, st, _ = rc.load(rows[i][4])
        hit = neck & (st == 1)
        if hit.any():
            first, side = i, float(np.sign(cen[hit, 0].mean()) or 1.0)
            ycrack = float(cen[hit, 1].mean())
            break
    if first is None:
        raise SystemExit("no failed neck cell in this run")
    start = int(np.searchsorted(travel, travel[first] - a.before_mm))
    start = min(start, first)
    end = max(end, first + 1)
    idx = list(range(start, min(end, len(rows) - 1) + 1))
    nfr = int(a.seconds * a.fps)
    pick = [idx[int(round(k * (len(idx) - 1) / max(nfr - 1, 1)))] for k in range(nfr)]
    frames = Path(str(Path(a.out).with_suffix("")) + "_frames")
    frames.mkdir(parents=True, exist_ok=True)
    focus = np.array([side * 23.7, ycrack, 8.1])
    uts = 50.5
    cache = {}
    for k, i in enumerate(pick):
        f = frames / f"f{k:04d}.png"
        if f.exists():
            continue
        if i not in cache:
            cache.clear()
            cache[i] = rc.load(rows[i][4])
        pts, u, vm, st, g = cache[i]
        cur = g.copy()
        cur.points = pts + u
        cur.point_data["vm"] = vm
        pl = pv.Plotter(off_screen=True, window_size=(W, H))
        pl.set_background(rc.BG["color"], top=rc.BG["top"])
        body = rc._surf(cur, st == 0)
        if body is not None:
            pl.add_mesh(body, scalars="vm", clim=[0, uts], cmap="turbo",
                        scalar_bar_args=dict(title="von Mises (MPa), colour limit = UTS; red = broken", color="white",
                                             position_x=0.25, position_y=0.03, width=0.5, height=0.08, fmt="%.0f"))
        ov = None
        if (st == 1).any():
            bad = rc._surf(cur, st == 1)
            pl.add_mesh(bad, color="#ff2a2a")
            ov = rc.xray(pl, bad, "#ff2a2a", 0.85)        # broken cells inside the wall stay visible
        # follow the neck: focus on the deformed position of the points nearest the crack start
        near = np.argsort(np.linalg.norm(pts - focus, axis=1))[:50]
        fp = (pts[near] + u[near]).mean(axis=0)
        pl.camera.focal_point = fp
        pl.camera.position = fp + np.array([side * 70.0, 0.0, 100.0])      # outer fibre + front face
        pl.camera.up = (0.0, 1.0, 0.0)
        pl.enable_parallel_projection()
        pl.camera.parallel_scale = 16.0
        if ov is not None:
            ov[0].SetActiveCamera(ov[1].GetActiveCamera())
        pl.add_text(f"{a.label}   -   neck crack close-up, {'right' if side > 0 else 'left'} arm (outer + front face,"
                    f" broken cells drawn through the part)", font_size=12, position="upper_left", color="white")
        pl.add_text(f"travel {travel[i]:.3f} mm   force {force[i]:.0f} N (peak {force[ip]:.0f} N at "
                    f"{travel[ip]:.3f} mm)   broken {nfail[i]}   step {steps[i]}",
                    font_size=11, position=(10, int(H * 0.9)), color="white")
        pl.screenshot(str(f))
        pl.close()
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-framerate", str(a.fps), "-i", str(frames / "f%04d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", a.out], check=True)
    print(a.out, f"steps {steps[start]}..{steps[idx[-1]]}, travel {travel[start]:.3f}..{travel[idx[-1]]:.3f} mm,"
          f" first neck failure at step {steps[first]} ({travel[first]:.3f} mm), side {'R' if side > 0 else 'L'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
