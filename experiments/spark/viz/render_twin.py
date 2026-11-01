#!/usr/bin/env python3
"""Offscreen quality preview of twin.usdc with VTK (EGL on the GB10 via render.sh).

    python render_twin.py twin.usdc OUTDIR [--reps 20] [--size 1920x1080]

Reads the stage through pxr (the file is not modified: variant switches go to the session layer).
Views:  iso.png (von Mises, deformed at the last time code, undeformed outline), top.png (von Mises,
undeformed, plan view), root_closeup.png (von Mises at the clamped root corner), deformed.png (|u|,
side view, deformed + undeformed outline), plus field_<variant>.png for every "field" variant.
Colour ranges are the stage's (summary vm_color_range_MPa / u_color_range_mm). Writes render_log.json.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pyvista as pv
from pxr import Usd, UsdGeom

pv.OFF_SCREEN = True
TITLE = {"von_mises": "von Mises stress [MPa]", "u_mag": "displacement magnitude [mm]"}


def lin_to_srgb(c):
    c = np.clip(np.asarray(c, float), 0, 1)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * c ** (1 / 2.4) - 0.055)


def parts_at(stage, t):
    out = []
    for prim in stage.Traverse():
        path = prim.GetPath().pathString
        if not prim.IsA(UsdGeom.Mesh) or path.startswith("/World/Legend"):
            continue
        m = UsdGeom.Mesh(prim)
        xf = np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(t)).T
        f = np.array(m.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)
        p = np.array(m.GetPointsAttr().Get(t), float) @ xf[:3, :3].T + xf[:3, 3]
        p0 = np.array(m.GetPointsAttr().Get(Usd.TimeCode(0)), float) @ xf[:3, :3].T + xf[:3, 3]
        api = UsdGeom.PrimvarsAPI(m)
        dc = api.GetPrimvar("displayColor")
        c = np.array(dc.Get(), float)
        if dc.GetInterpolation() == UsdGeom.Tokens.constant or len(c) == 1:
            c = np.repeat(c[:1], len(p), 0)
        pd = pv.PolyData.from_regular_faces(p, f)
        pd.point_data["rgb"] = (lin_to_srgb(c) * 255).astype(np.uint8)
        for k in ("von_mises", "u_mag"):
            pd.point_data[k] = np.array(api.GetPrimvar(k).Get(), float)
        edges = pv.PolyData.from_regular_faces(p0, f).extract_feature_edges(
            feature_angle=30, boundary_edges=True, non_manifold_edges=False, manifold_edges=False)
        out.append(dict(name=prim.GetParent().GetName(), mesh=pd, edges=edges,
                        gate=api.GetPrimvar("gate_status").Get()))
    return out


def fmt(summ, k, f="{:.4g}"):
    v = summ.get(k)
    return "n/a" if v is None else f.format(v)


def render(stage, summ, field, t, cam, out_png, size, reps, outline=True, note=None):
    vset = stage.GetPrimAtPath("/World").GetVariantSet("field")
    vset.SetVariantSelection(field if field in vset.GetVariantNames() else "von_mises")
    parts = parts_at(stage, t)
    pl = pv.Plotter(off_screen=True, window_size=size, lighting="three lights")
    pl.set_background("#262a31", top="#58606e")
    rng = {"von_mises": summ.get("vm_color_range_MPa"), "u_mag": summ.get("u_color_range_mm")}
    for q in parts:
        if outline:
            pl.add_mesh(q["edges"], color="#e3e6ec", opacity=0.45, line_width=1.2)
        if field in TITLE:
            lo, hi = rng[field] if rng[field] is not None else (q["mesh"][field].min(), q["mesh"][field].max())
            pl.add_mesh(q["mesh"], scalars=field, cmap="turbo", clim=(float(lo), float(hi)), smooth_shading=True,
                        specular=0.25, show_scalar_bar=False)
        else:
            op = 0.22 if (field == "gate_status" and q["gate"] == "pass") else 1.0
            pl.add_mesh(q["mesh"], scalars="rgb", rgb=True, smooth_shading=True, specular=0.25, opacity=op)
    if field in TITLE:
        clipped = field == "von_mises" and summ.get("vm_color_clipped")
        pl.add_scalar_bar(title=TITLE[field] + ("  (clipped at p%g)" % summ.get("vm_clip_percentile", 99.5)
                                                if clipped else ""),
                          n_labels=7, vertical=True, position_x=0.86, position_y=0.16, height=0.68, width=0.045,
                          fmt="%.3g", title_font_size=16, label_font_size=15, color="white")
    else:
        names, counts = {}, {}
        for q in parts:
            key = q["gate"] if field == "gate_status" else q["name"]
            names.setdefault(key, tuple(q["mesh"]["rgb"][0] / 255.0))
            counts[key] = counts.get(key, 0) + 1
        W, H = size
        for k_, (key, col) in enumerate(names.items()):
            label = f"{key}  ({counts[key]} part{'s' if counts[key] > 1 else ''})" if field == "gate_status" else key
            pl.add_text("\u25a0 " + label, position=(W - 330, H - 70 - 30 * k_), font_size=13, color=col)
        if field == "gate_status" and summ.get("gate_note"):
            note = summ["gate_note"]
    if cam.get("mark_peak"):
        q = max(parts, key=lambda q: q["mesh"]["von_mises"].max())
        i = int(np.argmax(q["mesh"]["von_mises"]))
        pl.add_point_labels(q["mesh"].points[i:i + 1], [f"peak {q['mesh']['von_mises'][i]:.1f} MPa: root-corner singularity"],
                            point_color="white", point_size=14, render_points_as_spheres=True, font_size=16,
                            text_color="black", shape_color="#ffd28a", shape_opacity=0.9, always_visible=True)
    defl = "undeformed" if t.GetValue() == 0 else f"deformation x{summ.get('deformation_scale', 1):g}"
    pl.add_text(f"{summ.get('title', '')}  |  {field}  |  {defl}", position="upper_left", font_size=11, color="white")
    sub = (f"verdict {summ.get('verdict', 'n/a')}   DOFs {fmt(summ, 'n_dofs', '{:,}')}   {summ.get('solver') or ''} "
           f"{fmt(summ, 'solve_time_s', '{:.1f}')} s   tip {fmt(summ, 'tip_deflection_mm', '{:.3f}')} mm "
           f"(extrap. {fmt(summ, 'tip_deflection_extrapolated_mm', '{:.3f}')}) vs beam {fmt(summ, 'beam_theory_mm', '{:.3f}')} mm   "
           f"GCI {fmt(summ, 'gci_percent', '{:.3f}')} %")
    pl.add_text(sub, position="lower_left", font_size=9, color="#dddddd")
    if note:
        pl.add_text(note, position=(12, 34), font_size=9, color="#ffd28a")
    pl.camera.position, pl.camera.focal_point, pl.camera.up = cam["pos"], cam["focal"], cam["up"]
    if cam.get("angle"):
        pl.camera.view_angle = cam["angle"]
    if cam.get("fit", True):
        pl.reset_camera()
        pl.camera.zoom(cam.get("zoom", 1.0))
    pl.show(auto_close=False)
    t0 = time.time()
    for _ in range(reps):
        pl.render()
    pl.screenshot(str(out_png))
    dt = (time.time() - t0) / (reps + 1)
    caps = pl.render_window.ReportCapabilities().split("\n")
    gl = [c.strip() for c in caps if "OpenGL renderer" in c]
    rec = {"png": str(out_png), "field": field, "time": t.GetValue(), "window": type(pl.render_window).__name__,
           "gl": gl, "tris": int(sum(q["mesh"].n_faces_strict for q in parts)), "ms_per_frame": round(dt * 1000, 2),
           "size": list(size)}
    pl.close()
    print(json.dumps(rec), flush=True)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("usd")
    ap.add_argument("outdir")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--size", default="1920x1080")
    a = ap.parse_args()
    size = tuple(map(int, a.size.split("x")))
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.Open(a.usd)
    stage.SetEditTarget(stage.GetSessionLayer())
    summ = dict(stage.GetRootLayer().customLayerData.get("feagent", {}))
    for k in ("vm_color_range_MPa", "u_color_range_mm"):
        if k in summ:
            summ[k] = [float(x) for x in summ[k]]
    t0, tN = Usd.TimeCode(0), Usd.TimeCode(stage.GetEndTimeCode())
    bbox = UsdGeom.BBoxCache(t0, [UsdGeom.Tokens.default_]).ComputeWorldBound(
        stage.GetPrimAtPath("/World/" + [p.GetName() for p in stage.GetPrimAtPath("/World").GetChildren()
                                         if p.GetTypeName() == "Xform" and p.GetName() not in ("Legend",)][0]))
    rng = bbox.ComputeAlignedRange()
    lo, hi = np.array(rng.GetMin()), np.array(rng.GetMax())
    c, L = 0.5 * (lo + hi), float(np.linalg.norm(hi - lo))
    st_cam = UsdGeom.Camera(stage.GetPrimAtPath("/World/Camera")).GetCamera(t0).frustum
    iso_dir = -np.array(st_cam.ComputeViewDirection())
    peak = summ.get("vm_peak_at")  # optional
    corner = np.array(peak, float) if peak is not None else np.array([hi[0], lo[1], hi[2]])
    clipnote = None
    if summ.get("vm_color_clipped"):
        clipnote = (f"colour max {summ['vm_color_range_MPa'][1]:.1f} MPa = p{summ.get('vm_clip_percentile', 99.5):g}; "
                    f"peak {summ.get('max_von_mises_MPa', float('nan')):.1f} MPa at the clamped root corner is a "
                    f"singularity (grows with refinement), not a design value")
    views = [
        ("iso", "von_mises", tN, dict(pos=tuple(c + iso_dir * L), focal=tuple(c), up=(0, 0, 1), zoom=1.25), True, clipnote),
        ("top", "von_mises", t0, dict(pos=tuple(c + np.array([0, 0, 1.5 * L])), focal=tuple(c), up=(-1, 0, 0), zoom=1.35),
         False, clipnote),
        ("root_closeup", "von_mises", t0, dict(pos=tuple(corner + np.array([0.55, -0.75, 0.45]) * 0.32 * L),
                                               focal=tuple(corner + np.array([-0.08, 0.06, -0.03]) * L),
                                               up=(0, 0, 1), angle=30, fit=False, mark_peak=True), False, clipnote),
        ("deformed", "u_mag", tN, dict(pos=tuple(c + np.array([-1.0, -0.12, 0.10]) * 1.5 * L), focal=tuple(c),
                                       up=(0, 0, 1), zoom=1.45), True, None),
    ]
    vnames = stage.GetPrimAtPath("/World").GetVariantSet("field").GetVariantNames()
    views += [(f"field_{v}", v, tN, dict(pos=tuple(c + iso_dir * L), focal=tuple(c), up=(0, 0, 1), zoom=1.25), True, None)
              for v in vnames]
    logs = [render(stage, summ, f, t, cam, out / f"{n}.png", size, a.reps, outline=o, note=nt)
            for n, f, t, cam, o, nt in views]
    (out / "render_log.json").write_text(json.dumps(logs, indent=1))


if __name__ == "__main__":
    main()
