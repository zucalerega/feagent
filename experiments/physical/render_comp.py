"""Load-ramp video of a platen compression test, rendered on the CPU (llvmpipe).

usage: render_comp.py PART_DIR AXIS F_FULL MAG OUT.mp4 TITLE
Fields in PART_DIR/result.vtu are for the 100 N solve; linear elasticity, so every field scales with F / 100 N.
Three views: outside (von Mises), mid cut plane (von Mises), mid cut plane (displacement magnitude + arrows)."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyvista as pv

pv.OFF_SCREEN = True
FPS, RAMP_S, HOLD_S, F_REF, VM_MAX_MPA = 30, 8.0, 3.0, 100.0, 46.0
ORBIT_DEG = 70.0   # slow camera orbit over the whole clip
AX = {"x": 0, "y": 1, "z": 2}


def load(part_dir: Path):
    g = pv.read(part_dir / "result.vtu").linear_copy()
    g.points = g.points * 1e3                      # m -> mm for display
    g.point_data["u_mm"] = np.asarray(g.point_data["displacement"]) * 1e3
    g.point_data["vm_mpa"] = np.asarray(g.point_data["von_mises"]) / 1e6
    return g


def cut_normal(axis: int) -> list[float]:
    n = [0.0, 0.0, 0.0]; n[(axis + 1) % 3] = 1.0  # a plane containing the load axis
    return n


def platens(g, axis: int):
    lo, hi = np.array(g.bounds[0::2]), np.array(g.bounds[1::2])
    pad, th = 0.15 * (hi - lo), 0.04 * (hi[axis] - lo[axis])
    def slab(at, outward):
        b0, b1 = lo - pad, hi + pad
        b0[axis], b1[axis] = (at - th, at) if outward < 0 else (at, at + th)
        return pv.Box(bounds=(b0[0], b1[0], b0[1], b1[1], b0[2], b1[2]))
    return slab(lo[axis], -1), slab(hi[axis], +1)


def frame_state(k: int, f_full: float):
    t = k / FPS
    lam = min(1.0, t / RAMP_S)
    return lam, lam * f_full / F_REF


def build(g, axis, mag):
    n = cut_normal(axis)
    surf = g.extract_surface(algorithm="dataset_surface")
    side = int(np.argmax(n))
    keep = np.where(g.cell_centers().points[:, side] < g.center[side])[0]   # cells on one side: a crinkle cut
    half = g.extract_cells(keep).extract_surface(algorithm="dataset_surface")
    on_cut = np.where(np.abs(half.points[:, side] - g.center[side]) < 0.04 * np.ptp(g.points[:, side]))[0]
    idx = np.random.default_rng(0).choice(on_cut, size=min(300, on_cut.size), replace=False)   # arrows on the cut face
    return dict(base=g.points.copy(), surf=surf, sl=half, sl2=half.copy(), s_base=surf.points.copy(),
                c_base=half.points.copy(), arrows_src=half.extract_points(idx, adjacent_cells=False), mag=mag)


def update(st, scale):
    m = st["mag"] * scale
    for mesh, b in ((st["surf"], st["s_base"]), (st["sl"], st["c_base"]), (st["sl2"], st["c_base"])):
        mesh.points = b + m * np.asarray(mesh.point_data["u_mm"]) * (1 if mesh is st["surf"] else 1)
        mesh.point_data["vm"] = np.asarray(mesh.point_data["vm_mpa"]) * scale
        mesh.point_data["umag_um"] = np.linalg.norm(mesh.point_data["u_mm"], axis=1) * 1e3 * scale


def plotter(st, g, axis, u_max_um):
    pl = pv.Plotter(off_screen=True, shape=(1, 3), window_size=(1920, 1080), border=False)
    bot, top = platens(g, axis)
    sb = dict(color="white", title_font_size=14, label_font_size=12, n_labels=5, position_y=0.06, height=0.07, width=0.6, position_x=0.2)
    pl.subplot(0, 0)
    pl.add_mesh(st["surf"], scalars="vm", clim=[0, VM_MAX_MPA], cmap="turbo", smooth_shading=True,
                scalar_bar_args={**sb, "title": "von Mises (MPa)"})
    pl.add_mesh(bot, color="#8a8f98", opacity=0.55); st["top"] = pl.add_mesh(top.copy(), color="#8a8f98", opacity=0.55)
    pl.add_text("outside", font_size=12, position="upper_left", color="white")
    pl.subplot(0, 1)
    pl.add_mesh(st["sl"], scalars="vm", clim=[0, VM_MAX_MPA], cmap="turbo", show_scalar_bar=False)
    pl.add_text("inside: cut in half, von Mises", font_size=12, position="upper_left", color="white")
    pl.subplot(0, 2)
    pl.add_mesh(st["sl2"], scalars="umag_um", clim=[0, u_max_um], cmap="viridis",
                scalar_bar_args={**sb, "title": "displacement (um)"})
    pl.add_text("inside: movement (arrows = displacement)", font_size=12, position="upper_left", color="white")
    for c in range(3):
        pl.subplot(0, c); pl.set_background("#1d2330", top="#3a4356")
        pl.camera_position = "iso"; pl.camera.Azimuth(-30 if c else 0); pl.reset_camera(); pl.camera.zoom(float(__import__("os").environ.get("ZOOM", "1.5")))
    return pl, st["top"]


def arrows(st, scale):
    src = st["arrows_src"]
    pts = src.points + st["mag"] * scale * np.asarray(src.point_data["u_mm"])
    a = pv.PolyData(pts); a["v"] = np.asarray(src.point_data["u_mm"]) * scale
    length = 0.08 * max(np.ptp(st["base"], axis=0))
    norm = np.linalg.norm(a["v"], axis=1).max() or 1.0
    return a.glyph(orient="v", scale="v", factor=length / norm, geom=pv.Arrow(tip_length=0.35, shaft_radius=0.04))


def render(part_dir: Path, axis: int, f_full: float, mag: float, out: Path, title: str):
    g = load(part_dir); st = build(g, axis, mag); update(st, 0.0)
    s100 = json.loads((part_dir / "summary.json").read_text())["qois"]
    u_max_um = float(np.linalg.norm(g.point_data["u_mm"], axis=1).max()) * 1e3 * f_full / F_REF
    pl, top_actor = plotter(st, g, axis, u_max_um)
    _, top0 = platens(g, axis); top_base = top0.points.copy()
    frames = out.parent / (out.stem + "_frames"); frames.mkdir(parents=True, exist_ok=True)
    arrow_actor, text_actor = None, None
    n = int((RAMP_S + HOLD_S) * FPS)
    for k in range(n):
        lam, scale = frame_state(k, f_full)
        update(st, scale)
        shift = np.zeros(3); shift[axis] = -mag * s100["mean_u_mag_top"] * 1e3 * scale
        top_actor.mapper.dataset.points = top_base + shift
        pl.subplot(0, 2)
        if arrow_actor is not None:
            pl.remove_actor(arrow_actor)
        arrow_actor = pl.add_mesh(arrows(st, max(scale, 1e-9)), color="#ffe14d")
        pl.subplot(0, 0)
        if text_actor is not None:
            pl.remove_actor(text_actor)
        text_actor = pl.add_text(
            f"{title}\nplaten compression along {'xyz'[axis]}: F = {lam * f_full:6.0f} N of {f_full:.0f} N   "
            f"peak von Mises {s100['max_von_mises'] / 1e6 * scale:5.1f} MPa   "
            f"top moves {s100['mean_u_mag_top'] * 1e6 * scale:6.1f} um\n(deformation x{mag:.0f}; linear elastic below the 46 MPa UTS; "
            f"Formlabs Alpha Fast Resin, default post-cure, E = 2.23 GPa; solved by feacore/skfem on CPU)", font_size=15, position="lower_left", color="white")
        for c in range(3):
            pl.subplot(0, c); pl.camera.Azimuth(ORBIT_DEG / n)
        pl.screenshot(str(frames / f"f{k:04d}.png"))
    pl.close()
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", str(frames / "f%04d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(out)], check=True)
    print(f"wrote {out} ({n} frames)")


if __name__ == "__main__":
    d, ax, f, mag, out, title = sys.argv[1:7]
    render(Path(d), AX[ax], float(f), float(mag), Path(out), title)
