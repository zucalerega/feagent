#!/usr/bin/env python3
"""FEA virtual twin -> OpenUSD stage.

    python export_twin.py run_manifest.json [--out DIR] [--name twin] [--frames 24] [--scale auto]
                          [--max-tris 2500000] [--no-usdz]

Inputs (paths in the manifest are relative to the manifest file):

    {"title": "...", "placeholder": "placeholder: bracket" | null, "assembly": "Wingbox",
     "mesh": "assembly.msh",            # gmsh MSH 4.1 with dim-3 physical groups / volume entities
     "result_vtu": "r.vtu",             # python -m feacore.cli case.json --vtu r.vtu
     "result_json": "r.json",           # optional: n_dofs, wall_time, QoIs
     "parts": [{"name": "spar_front", "owner": "Spars mesher", "gate": "pass|warn|fail",
                "dofs": 123456, "report": "spars/report.json",
                "physical": "spar_front" | "volumes": [3, 4] | "mesh": "spar_front.msh"}],
     "summary": {"tip_qoi": "tip_uz", "tip_deflection_mm": .., "beam_theory_mm": .., "gci_percent": ..,
                 "solve_time_s": .., "solver": "cuDSS on GB10", "notes": ".."},
     "deform_scale": null, "vm_range": null}

A part's cells are found by matching vtu cell centroids to the msh tets of its physical group,
volume entities or own mesh (KD-tree). A single part with no selector takes every cell.

Output in DIR: twin.usdc (+ twin.usdz with the textures packed in), textures/legend_*.png,
summary.json. Stage layout:

    /World                 Xform, kind=assembly, variantSet "field" = von_mises|u_mag|gate_status|part_colors,
                           customData feagent:summary; layer customLayerData feagent = run summary
    /World/<Assembly>/<p>  Xform, kind=component, fea:* attributes + customData (gate, DOFs, owner, ...)
    /World/<Assembly>/<p>/Surface  Mesh: boundary surface of the part's tets (tet10 faces split in 4),
                           points time-sampled t=0 undeformed .. t=N full deformation scale,
                           primvars von_mises [MPa], u_mag [mm], displacement [mm], gate_status, gate_code,
                           displayColor (authored only inside the "field" variants)
    /World/Looks/<p>       Material: UsdPreviewSurface, diffuseColor <- UsdPrimvarReader_float3(displayColor)
    /World/Legend          textured quad (colour bar + min/max + units), texture switched by the variant
    /World/Lights          DistantLight + DomeLight
    /World/Camera          framed iso camera
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path

import numpy as np
from pxr import Gf, Kind, Sdf, Usd, UsdGeom, UsdLux, UsdShade, UsdUtils, Vt

FIELDS = ("von_mises", "u_mag", "gate_status", "part_colors")
GATE_RGB = {"pass": (0.16, 0.66, 0.32), "warn": (0.95, 0.66, 0.10), "fail": (0.86, 0.15, 0.15),
            "unknown": (0.55, 0.55, 0.58)}
GATE_CODE = {"pass": 0, "warn": 1, "fail": 2, "unknown": 3}
PART_RGB = [(0.12, 0.47, 0.71), (1.0, 0.50, 0.05), (0.17, 0.63, 0.17), (0.84, 0.15, 0.16),
            (0.58, 0.40, 0.74), (0.55, 0.34, 0.29), (0.89, 0.47, 0.76), (0.50, 0.50, 0.50),
            (0.74, 0.74, 0.13), (0.09, 0.75, 0.81)]
# local tet faces (VTK order) and tet10 edge midside nodes: 4:(0,1) 5:(1,2) 6:(0,2) 7:(0,3) 8:(1,3) 9:(2,3)
TET_FACES = np.array([[0, 2, 1], [0, 1, 3], [1, 2, 3], [0, 3, 2]])
MID = np.full((4, 4), -1)
for (a, b), m in {(0, 1): 4, (1, 2): 5, (0, 2): 6, (0, 3): 7, (1, 3): 8, (2, 3): 9}.items():
    MID[a, b] = MID[b, a] = m


def log(*a):
    print("[export_twin]", *a, flush=True)


def srgb_to_linear(c):
    c = np.asarray(c, np.float64)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4).astype(np.float32)


def cmap(name, x):
    from matplotlib import colormaps
    return colormaps[name](np.clip(x, 0, 1))[:, :3]


def safe_name(s):
    s = re.sub(r"[^A-Za-z0-9_]", "_", str(s))
    return s if s and not s[0].isdigit() else "p_" + s


def nice(x):
    """Round a positive number down to 1, 2 or 5 x 10^k."""
    if not x or x <= 0 or not math.isfinite(x):
        return 1.0
    e = 10 ** math.floor(math.log10(x))
    return max(m for m in (1, 2, 5, 10) if m * e <= x) * e


# ---------------------------------------------------------------- inputs
def read_result(vtu):
    import meshio
    m = meshio.read(str(vtu))
    blocks = [(i, c) for i, c in enumerate(m.cells) if c.type in ("tetra10", "tetra")]
    if not blocks:
        raise SystemExit(f"{vtu}: no tetra/tetra10 cells")
    conn = np.vstack([c.data for _, c in blocks]).astype(np.int64)
    pd = m.point_data
    if "displacement" in pd:
        U = np.asarray(pd["displacement"], float)
    else:
        U = np.column_stack([pd[k] for k in ("ux", "uy", "uz")]).astype(float)
    vm = np.asarray(pd["von_mises"], float)
    um = np.asarray(pd["u_mag"], float) if "u_mag" in pd else np.linalg.norm(U, axis=1)
    mat = None
    if "material" in m.cell_data:
        mat = np.concatenate([np.asarray(m.cell_data["material"][i]) for i, _ in blocks])
    return np.asarray(m.points, float), conn, U, vm, um, mat


def msh_tets(path):
    """-> corner centroids (n,3), physical tag (n,), entity tag (n,), {name: tag} for dim-3 groups."""
    import meshio
    m = meshio.read(str(path))
    cen, phys, geo = [], [], []
    for i, c in enumerate(m.cells):
        if c.type not in ("tetra", "tetra10"):
            continue
        cen.append(m.points[c.data[:, :4]].mean(1))
        n = len(c.data)
        cd = m.cell_data
        phys.append(np.asarray(cd["gmsh:physical"][i]) if "gmsh:physical" in cd else np.zeros(n, int))
        geo.append(np.asarray(cd["gmsh:geometrical"][i]) if "gmsh:geometrical" in cd else np.zeros(n, int))
    names = {k: int(v[0]) for k, v in (m.field_data or {}).items() if len(v) > 1 and int(v[1]) == 3}
    return np.vstack(cen), np.concatenate(phys), np.concatenate(geo), names


def assign_parts(parts, base, mesh_path, P, conn, mat):
    """Part index per result cell (-1 = unassigned)."""
    nc = len(conn)
    sel = [p for p in parts if any(k in p for k in ("physical", "volumes", "mesh", "material"))]
    if not sel:
        if len(parts) != 1:
            raise SystemExit("several parts but no part has a selector (physical / volumes / mesh / material)")
        return np.zeros(nc, np.int64)
    lab = np.full(nc, -1, np.int64)
    if all("material" in p for p in parts) and mat is not None:
        for i, p in enumerate(parts):
            lab[mat == int(p["material"])] = i
        return lab
    from scipy.spatial import cKDTree
    cache, cen_all, lab_all = {}, [], []
    for i, p in enumerate(parts):
        src = base / p["mesh"] if "mesh" in p else mesh_path
        if src not in cache:
            cache[src] = msh_tets(src)
        cen, phys, geo, names = cache[src]
        if "mesh" in p:
            mask = np.ones(len(cen), bool)
        elif "physical" in p:
            if p["physical"] not in names:
                raise SystemExit(f"part {p['name']}: physical group {p['physical']!r} not in {src} "
                                 f"(dim-3 groups: {sorted(names)})")
            mask = phys == names[p["physical"]]
        elif "volumes" in p:
            mask = np.isin(geo, np.asarray(p["volumes"], int))
        else:
            continue
        if not mask.any():
            raise SystemExit(f"part {p['name']}: selector matches no tets in {src}")
        cen_all.append(cen[mask])
        lab_all.append(np.full(int(mask.sum()), i))
    tree = cKDTree(np.vstack(cen_all))
    lab_all = np.concatenate(lab_all)
    c = P[conn[:, :4]].mean(1)
    d, j = tree.query(c)
    h = np.median(np.linalg.norm(P[conn[:, 0]] - P[conn[:, 1]], axis=1))
    ok = d < 0.25 * h
    lab[ok] = lab_all[j[ok]]
    if (~ok).any():
        log(f"WARNING: {int((~ok).sum())} of {nc} result cells match no part tet (tol {0.25 * h:.3g})")
    return lab


def boundary(P, conn, quadratic):
    """Outward boundary triangles of a tet(10) cell set -> (global node ids used, local tris)."""
    fl = np.repeat(TET_FACES[None], len(conn), 0).reshape(-1, 3)        # local corner ids per face
    owner = np.repeat(np.arange(len(conn)), 4)
    fg = conn[owner[:, None], fl]
    k = np.sort(fg, axis=1)
    o = np.lexsort((k[:, 2], k[:, 1], k[:, 0]))
    ks = k[o]
    same = np.all(ks[1:] == ks[:-1], axis=1)
    dup = np.zeros(len(ks), bool)
    dup[1:] |= same
    dup[:-1] |= same
    keep = np.sort(o[~dup])
    fl, owner, fg = fl[keep], owner[keep], fg[keep]
    n = np.cross(P[fg[:, 1]] - P[fg[:, 0]], P[fg[:, 2]] - P[fg[:, 0]])
    cen = P[conn[owner, :4]].mean(1)
    flip = np.einsum("ij,ij->i", n, P[fg[:, 0]] - cen) < 0
    fl[flip] = fl[flip][:, [0, 2, 1]]
    a, b, c = fl[:, 0], fl[:, 1], fl[:, 2]
    if quadratic and conn.shape[1] >= 10:
        ab, bc, ca = MID[a, b], MID[b, c], MID[c, a]
        loc = np.stack([np.stack([a, ab, ca], 1), np.stack([ab, b, bc], 1),
                        np.stack([ca, bc, c], 1), np.stack([ab, bc, ca], 1)], 1).reshape(-1, 3)
        tri = conn[np.repeat(owner, 4)[:, None], loc]
    else:
        tri = conn[owner[:, None], fl]
    used, inv = np.unique(tri, return_inverse=True)
    return used, inv.reshape(-1, 3).astype(np.int32)


def decimate(p, t, keep_frac):
    """vtkDecimatePro keeps a subset of the original points -> returns (point subset ids, tris)."""
    import pyvista as pv
    pd = pv.PolyData.from_regular_faces(p, t)
    pd.point_data["_id"] = np.arange(len(p))
    d = pd.decimate_pro(1.0 - keep_frac, preserve_topology=True, feature_angle=30, splitting=False)
    return np.asarray(d.point_data["_id"]), d.regular_faces.astype(np.int32)


def gate_from_report(path):
    r = json.loads(Path(path).read_text())
    st = str(r.get("status", "unknown")).lower()
    gates = r.get("gates", [])
    bad = [f"{g.get('id')}={g.get('status')}" for g in gates if g.get("status") not in ("pass", "skip")]
    return st, bad, len(gates)


# ---------------------------------------------------------------- legend textures
def legend_png(path, kind, lo=None, hi=None, unit="", parts=None, placeholder=None, note=None, clipped=False,
               extra=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import colormaps
    from matplotlib.colorbar import ColorbarBase
    from matplotlib.colors import Normalize
    fig = plt.figure(figsize=(3.2, 6.4), dpi=160, facecolor="#f4f4f2")
    title = {"von_mises": "von Mises stress", "u_mag": "|u| displacement",
             "gate_status": "mesh-gate status", "part_colors": "parts"}[kind]
    fig.text(0.5, 0.955, title, ha="center", va="top", fontsize=13, weight="bold", color="#1a1a1a")
    if placeholder:
        import textwrap
        fig.text(0.5, 0.91, "\n".join(textwrap.wrap(placeholder.upper(), 30)), ha="center", va="top",
                 fontsize=9, weight="bold", color="#c0392b", linespacing=1.2)
    if kind in ("von_mises", "u_mag"):
        ax = fig.add_axes([0.30, 0.13, 0.16, 0.6 if extra else 0.68])
        cb = ColorbarBase(ax, cmap=colormaps["turbo"], norm=Normalize(lo, hi))
        cb.set_ticks(np.linspace(lo, hi, 9))
        cb.ax.tick_params(labelsize=10)
        labels = [f"{v:.3g}" for v in np.linspace(lo, hi, 9)]
        if clipped:
            labels[-1] = "\u2265 " + labels[-1]
        cb.ax.set_yticklabels(labels)
        fig.text(0.5, 0.085, f"[{unit}]   min {lo:.3g}   {'colour max' if clipped else 'max'} {hi:.3g}",
                 ha="center", fontsize=9.5)
    else:
        items = list(GATE_RGB.items()) if kind == "gate_status" else parts
        y = 0.78
        for name, rgb in items:
            fig.patches.append(plt.Rectangle((0.12, y - 0.025), 0.12, 0.05, color=rgb,
                                             transform=fig.transFigure, figure=fig))
            fig.text(0.29, y, str(name), va="center", fontsize=11)
            y -= 0.075
            if y < 0.12:
                break
    import textwrap
    if extra:
        fig.text(0.5, 0.905 if kind in ("von_mises", "u_mag") else 0.40, "\n".join(textwrap.wrap(extra, 44)),
                 ha="center", va="top", fontsize=7.6,
                 color="#7a1f1f" if kind == "von_mises" else "#333333", linespacing=1.15)
    if note:
        fig.text(0.5, 0.03, note, ha="center", fontsize=8, color="#444444")
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)


# ---------------------------------------------------------------- USD authoring
def look_at_xform(eye, target, up=(0, 0, 1)):
    view = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(*up))
    return view.GetInverse()


def preview_material(stage, path, roughness=0.45, texture=None):
    mat = UsdShade.Material.Define(stage, path)
    sh = UsdShade.Shader.Define(stage, f"{path}/PreviewSurface")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    sh.CreateInput("useSpecularWorkflow", Sdf.ValueTypeNames.Int).Set(0)
    if texture is None:
        ro = UsdShade.Shader.Define(stage, f"{path}/DisplayOpacityReader")
        ro.CreateIdAttr("UsdPrimvarReader_float")
        ro.CreateInput("varname", Sdf.ValueTypeNames.String).Set("displayOpacity")
        ro.CreateInput("fallback", Sdf.ValueTypeNames.Float).Set(1.0)
        sh.CreateInput("opacity", Sdf.ValueTypeNames.Float).ConnectToSource(
            ro.ConnectableAPI(), "result", UsdShade.AttributeType.Output, Sdf.ValueTypeNames.Float)
        rd = UsdShade.Shader.Define(stage, f"{path}/DisplayColorReader")
        rd.CreateIdAttr("UsdPrimvarReader_float3")
        rd.CreateInput("varname", Sdf.ValueTypeNames.String).Set("displayColor")
        rd.CreateInput("fallback", Sdf.ValueTypeNames.Float3).Set(Gf.Vec3f(0.6, 0.6, 0.62))
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
            rd.ConnectableAPI(), "result", UsdShade.AttributeType.Output, Sdf.ValueTypeNames.Float3)
    else:  # unlit-looking legend: texture into emissive, black diffuse
        st = UsdShade.Shader.Define(stage, f"{path}/StReader")
        st.CreateIdAttr("UsdPrimvarReader_float2")
        st.CreateInput("varname", Sdf.ValueTypeNames.String).Set("st")
        tx = UsdShade.Shader.Define(stage, f"{path}/LegendTexture")
        tx.CreateIdAttr("UsdUVTexture")
        tx.CreateInput("file", Sdf.ValueTypeNames.Asset)  # value authored only in the "field" variants
        tx.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
        tx.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("clamp")
        tx.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("clamp")
        tx.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(st.ConnectableAPI(), "result")
        tx.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0, 0, 0))
        sh.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tx.ConnectableAPI(), "rgb")
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    return mat


def build(args):
    t_start = time.time()
    man_path = Path(args.manifest).resolve()
    base = man_path.parent
    man = json.loads(man_path.read_text())
    out = Path(args.out).expanduser().resolve()
    (out / "textures").mkdir(parents=True, exist_ok=True)
    name = args.name
    placeholder = man.get("placeholder")
    asm = safe_name(man.get("assembly", "Wingbox"))
    parts = man["parts"]
    mesh_path = base / man["mesh"] if man.get("mesh") else None

    P, conn, U, vm, um, mat = read_result(base / man["result_vtu"])
    res = json.loads((base / man["result_json"]).read_text()) if man.get("result_json") else {}
    info, qois = res.get("info", {}), res.get("qois", {})
    log(f"result: {len(P)} nodes, {len(conn)} cells ({conn.shape[1]}-node), n_dofs={info.get('n_dofs')}")
    lab = assign_parts(parts, base, mesh_path, P, conn, mat)

    # ---- per-part boundary surfaces (quadratic split if it fits the budget, else linear, else decimated)
    groups = [np.flatnonzero(lab == i) for i in range(len(parts))]
    for p, g in zip(parts, groups):
        if len(g) == 0:
            raise SystemExit(f"part {p['name']} has no cells: check its selector")
    surf = {}
    quad = True
    for attempt in ("quadratic", "linear"):
        quad = attempt == "quadratic"
        surf = {i: boundary(P, conn[g], quad) for i, g in enumerate(groups)}
        ntri = sum(len(t) for _, t in surf.values())
        if ntri <= args.max_tris or attempt == "linear":
            break
        log(f"{ntri} tris with quadratic faces > {args.max_tris}: using linear faces")
    ntri = sum(len(t) for _, t in surf.values())
    decimated = False
    if ntri > args.max_tris:
        frac = args.max_tris / ntri
        log(f"{ntri} tris > {args.max_tris}: decimating every part to {frac:.2f}")
        for i, (used, t) in list(surf.items()):
            ids, t2 = decimate(P[used], t, frac)
            surf[i] = (used[ids], t2)
        decimated = True
        ntri = sum(len(t) for _, t in surf.values())

    # ---- global ranges, deformation scale, frames
    allused = np.unique(np.concatenate([u for u, _ in surf.values()]))
    vclip = man.get("vm_clip_percentile")
    if man.get("vm_range"):
        vr = [float(v) for v in man["vm_range"]]
    elif vclip:
        vr = [float(vm.min()), float(np.percentile(vm, float(vclip)))]
    else:
        vr = [float(vm.min()), float(vm.max())]
    vm_clipped = vr[1] < float(vm.max())
    ur = [0.0, float(um.max())]
    bb_lo, bb_hi = P.min(0), P.max(0)
    diag = float(np.linalg.norm(bb_hi - bb_lo))
    umax = float(np.linalg.norm(U, axis=1).max())
    scale = float(args.scale if args.scale not in (None, "auto") else
                  man.get("deform_scale") or nice(0.08 * float((bb_hi - bb_lo).max()) / max(umax, 1e-30)))
    nverts = sum(len(u) for u, _ in surf.values())
    frames = max(2, min(args.frames, int(2.0e8 / max(12 * nverts, 1))))
    log(f"surface: {ntri} tris, {nverts} verts ({'quadratic faces split in 4' if quad else 'linear faces'}"
        f"{', decimated' if decimated else ''}); deformation scale {scale:g}x over {frames} frames")

    # ---- summary
    sm = dict(man.get("summary", {}))
    tip_q = sm.get("tip_qoi")
    summary = {
        "title": man.get("title", "FEA virtual twin"),
        "placeholder": placeholder,
        "n_dofs": int(info.get("n_dofs", 0)) or None,
        "n_nodes": int(info.get("n_nodes", len(P))), "n_cells": int(info.get("n_cells", len(conn))),
        "cell_type": info.get("cell_type"), "backend": info.get("backend"),
        "solver": sm.get("solver"),
        "solve_time_s": sm.get("solve_time_s", info.get("wall_time")),
        "max_von_mises_MPa": float(vm.max()), "max_displacement_mm": umax,
        "tip_deflection_mm": sm.get("tip_deflection_mm", qois.get(tip_q) if tip_q else None),
        "beam_theory_mm": sm.get("beam_theory_mm"),
        "gci_percent": sm.get("gci_percent"),
        "units": {"length": "mm", "stress": "MPa", "force": "N"},
        "deformation_scale": scale, "frames": frames,
        "surface_tris": int(ntri), "surface_verts": int(nverts),
        "qois": {k: v for k, v in qois.items()},
        "notes": sm.get("notes"),
        "vm_color_range_MPa": vr, "vm_color_clipped": vm_clipped,
        "vm_clip_percentile": float(vclip) if vclip and not man.get("vm_range") else None,
        "u_color_range_mm": ur,
        "parts": [],
    }
    used_keys = {"tip_qoi", "solver", "solve_time_s", "tip_deflection_mm", "beam_theory_mm", "gci_percent", "notes"}
    for k, v in sm.items():  # pass every other manifest summary key through (verdict, notes, caveats, ...)
        if k not in used_keys:
            summary[k] = v
    td, bt = summary["tip_deflection_mm"], summary["beam_theory_mm"]
    summary["tip_vs_beam_percent"] = (100.0 * (abs(td) - abs(bt)) / abs(bt)) if td and bt else None

    # ---- stage
    usd_path = out / f"{name}.usdc"
    if usd_path.exists():
        usd_path.unlink()
    stage = Usd.Stage.CreateNew(str(usd_path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 0.001)
    stage.SetStartTimeCode(0)
    stage.SetEndTimeCode(frames)
    stage.SetTimeCodesPerSecond(frames)        # the 0 -> full-scale ramp plays in 1 s
    stage.SetFramesPerSecond(frames)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    Usd.ModelAPI(world).SetKind(Kind.Tokens.assembly)
    asm_x = UsdGeom.Xform.Define(stage, f"/World/{asm}")
    Usd.ModelAPI(asm_x).SetKind(Kind.Tokens.assembly)
    UsdGeom.Scope.Define(stage, "/World/Looks")

    pnames = [safe_name(p["name"]) for p in parts]
    gate_of, bad_of = [], []
    surfaces = []
    for i, p in enumerate(parts):
        used, tri = surf[i]
        g = str(p.get("gate", "")).lower()
        bad, ngates, spec_st = [], 0, None
        if p.get("report"):
            spec_st, bad, ngates = gate_from_report(base / p["report"])
            bad = [f"part spec: {b}" for b in bad]
        sts = [x for x in (g, spec_st) if x]
        for eg in p.get("extra_gates", []):
            sts.append(str(eg["status"]).lower())
            ngates += 1
            if eg["status"] not in ("pass", "skip"):
                bad.append(f"{eg['id']}={eg['status']}" + (f" ({eg['detail']})" if eg.get("detail") else ""))
        order = ["unknown", "pass", "warn", "fail"]
        sts = [x for x in sts if x in order]
        g = max(sts, key=order.index) if sts else "unknown"
        g = g if g in GATE_RGB else "unknown"
        gate_of.append(g)
        cells = groups[i]
        part_nodes = np.unique(conn[cells])
        dofs = int(p.get("dofs") or 3 * len(part_nodes))
        pp = f"/World/{asm}/{pnames[i]}"
        xf = UsdGeom.Xform.Define(stage, pp)
        Usd.ModelAPI(xf).SetKind(Kind.Tokens.component)
        prim = xf.GetPrim()
        meta = {"part": str(p["name"]), "gate_status": g, "dofs": dofs,
                "dofs_source": "manifest" if p.get("dofs") else "3 x part nodes in the result mesh",
                "owner": str(p.get("owner", "")), "n_cells": int(len(cells)), "n_nodes": int(len(part_nodes)),
                "n_surface_tris": int(len(tri)), "gates_not_passing": Vt.StringArray(bad), "n_gates": ngates,
                "max_von_mises_MPa": float(vm[part_nodes].max()), "max_u_mag_mm": float(um[part_nodes].max()),
                "part_spec_status": spec_st or "n/a",
                "part_spec_report": str(p.get("report", "")),
                "volume_tags": Vt.IntArray([int(v) for v in p.get("volumes", [])])}
        if placeholder:
            meta["placeholder"] = True
        for k, v in p.get("extra", {}).items():
            meta[k] = v
        prim.SetCustomDataByKey("feagent", meta)
        prim.CreateAttribute("fea:gate_status", Sdf.ValueTypeNames.Token, custom=True).Set(g)
        prim.CreateAttribute("fea:dofs", Sdf.ValueTypeNames.Int, custom=True).Set(dofs)
        prim.CreateAttribute("fea:owner", Sdf.ValueTypeNames.String, custom=True).Set(meta["owner"])
        prim.CreateAttribute("fea:max_von_mises", Sdf.ValueTypeNames.Float, custom=True).Set(meta["max_von_mises_MPa"])
        prim.CreateAttribute("fea:max_u_mag", Sdf.ValueTypeNames.Float, custom=True).Set(meta["max_u_mag_mm"])
        prim.SetDocumentation(f"{p['name']}: gate {g.upper()}, {dofs} DOFs, owner {meta['owner'] or '-'}"
                              + (f"; not passing: {', '.join(bad)}" if bad else ""))
        summary["parts"].append({k: (list(v) if isinstance(v, (Vt.StringArray, Vt.IntArray)) else v)
                                 for k, v in meta.items()})

        m = UsdGeom.Mesh.Define(stage, f"{pp}/Surface")
        p0 = P[used].astype(np.float32)
        du = U[used].astype(np.float32)
        pts = m.CreatePointsAttr()
        ext = m.CreateExtentAttr()
        pts.Set(Vt.Vec3fArray.FromNumpy(p0))
        ext.Set(Vt.Vec3fArray.FromNumpy(np.stack([p0.min(0), p0.max(0)])))
        for k in range(frames + 1):
            pk = p0 + du * np.float32(scale * k / frames)
            pts.Set(Vt.Vec3fArray.FromNumpy(pk), Usd.TimeCode(k))
            ext.Set(Vt.Vec3fArray.FromNumpy(np.stack([pk.min(0), pk.max(0)])), Usd.TimeCode(k))
        m.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(tri), 3, np.int32)))
        m.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(tri.ravel()))
        m.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        m.CreateOrientationAttr(UsdGeom.Tokens.rightHanded)
        m.CreateDoubleSidedAttr(False)
        api = UsdGeom.PrimvarsAPI(m)
        pv_ = api.CreatePrimvar("von_mises", Sdf.ValueTypeNames.FloatArray, UsdGeom.Tokens.vertex)
        pv_.Set(Vt.FloatArray.FromNumpy(vm[used].astype(np.float32)))
        pv_.GetAttr().SetDocumentation("von Mises stress [MPa], nodal")
        pu = api.CreatePrimvar("u_mag", Sdf.ValueTypeNames.FloatArray, UsdGeom.Tokens.vertex)
        pu.Set(Vt.FloatArray.FromNumpy(um[used].astype(np.float32)))
        pu.GetAttr().SetDocumentation("displacement magnitude [mm], nodal")
        pd_ = api.CreatePrimvar("displacement", Sdf.ValueTypeNames.Vector3fArray, UsdGeom.Tokens.vertex)
        pd_.Set(Vt.Vec3fArray.FromNumpy(du))
        pd_.GetAttr().SetDocumentation(f"displacement vector [mm], unscaled; points at time t are "
                                       f"rest + displacement * {scale:g} * t / {frames}")
        api.CreatePrimvar("gate_status", Sdf.ValueTypeNames.Token, UsdGeom.Tokens.constant).Set(g)
        api.CreatePrimvar("gate_code", Sdf.ValueTypeNames.Int, UsdGeom.Tokens.constant).Set(GATE_CODE[g])
        m.GetPrim().SetCustomDataByKey("feagent", {"range_von_mises_MPa": Vt.DoubleArray(vr),
                                                   "range_u_mag_mm": Vt.DoubleArray(ur),
                                                   "deformation_scale": scale})
        mat_ = preview_material(stage, f"/World/Looks/{pnames[i]}")
        UsdShade.MaterialBindingAPI.Apply(m.GetPrim()).Bind(mat_)
        surfaces.append((m, used, g, i))

    # ---- camera, lights, legend placement (iso from -y, +x side, above)
    ctr = 0.5 * (bb_lo + bb_hi)
    view_dir = np.array([0.55, -1.0, 0.65])
    view_dir /= np.linalg.norm(view_dir)
    fov_v = 2 * math.atan(20.25 / (2 * 35.0))
    dist = 0.5 * diag * 1.25 / math.sin(fov_v / 2) * 0.9
    eye = ctr + view_dir * dist
    cam = UsdGeom.Camera.Define(stage, "/World/Camera")
    cam.CreateFocalLengthAttr(35.0)
    cam.CreateHorizontalApertureAttr(36.0)
    cam.CreateVerticalApertureAttr(20.25)
    cam.CreateClippingRangeAttr(Gf.Vec2f(max(dist * 0.01, 1e-3), dist * 20))
    cam.AddTransformOp().Set(look_at_xform(eye, ctr))

    UsdGeom.Scope.Define(stage, "/World/Lights")
    sun = UsdLux.DistantLight.Define(stage, "/World/Lights/Key")
    sun.CreateIntensityAttr(2.2)
    sun.CreateAngleAttr(1.0)
    sun.CreateColorAttr(Gf.Vec3f(1.0, 0.98, 0.95))
    key_dir = np.array([0.35, -0.6, 0.72])  # light comes from here
    sun.AddTransformOp().Set(look_at_xform(key_dir * 10, (0, 0, 0)))
    dome = UsdLux.DomeLight.Define(stage, "/World/Lights/Dome")
    dome.CreateIntensityAttr(0.7)
    dome.CreateColorAttr(Gf.Vec3f(0.92, 0.94, 1.0))

    # legend quad: right of the model, facing the camera's horizontal direction
    n = np.array([view_dir[0], view_dir[1], 0.0])
    n /= np.linalg.norm(n)
    right = np.cross([0, 0, 1], n)
    hgt = 0.45 * diag
    wid = hgt * 0.5
    cx = ctr + right * (0.5 * float(np.abs((bb_hi - bb_lo) @ np.abs(right))) + 0.12 * diag + 0.5 * wid)
    up = np.array([0.0, 0.0, 1.0])
    corners = [cx - right * wid / 2 - up * hgt / 2, cx + right * wid / 2 - up * hgt / 2,
               cx + right * wid / 2 + up * hgt / 2, cx - right * wid / 2 + up * hgt / 2]
    UsdGeom.Xform.Define(stage, "/World/Legend")
    lg = UsdGeom.Mesh.Define(stage, "/World/Legend/Bar")
    lg.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.array(corners, np.float32)))
    lg.CreateFaceVertexCountsAttr([4])
    lg.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    lg.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    lg.CreateDoubleSidedAttr(True)
    c4 = np.array(corners, np.float32)
    lg.CreateExtentAttr(Vt.Vec3fArray.FromNumpy(np.stack([c4.min(0), c4.max(0)])))
    UsdGeom.PrimvarsAPI(lg).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex).Set(
        Vt.Vec2fArray([(0, 0), (1, 0), (1, 1), (0, 1)]))
    lmat = preview_material(stage, "/World/Looks/Legend", texture=True)
    UsdShade.MaterialBindingAPI.Apply(lg.GetPrim()).Bind(lmat)

    # ---- legend textures
    note = f"deformation x{scale:g} on the timeline (t = 0..{frames})"
    pc = [(pn, PART_RGB[i % len(PART_RGB)]) for i, pn in enumerate(pnames)]
    ln = man.get("legend_notes", {})
    legend_png(out / "textures/legend_von_mises.png", "von_mises", vr[0], vr[1], "MPa", placeholder=placeholder,
               note=note, clipped=vm_clipped, extra=ln.get("von_mises"))
    legend_png(out / "textures/legend_u_mag.png", "u_mag", ur[0], ur[1], "mm", placeholder=placeholder, note=note,
               extra=ln.get("u_mag"))
    legend_png(out / "textures/legend_gate_status.png", "gate_status", placeholder=placeholder, note=note,
               extra=ln.get("gate_status"))
    legend_png(out / "textures/legend_part_colors.png", "part_colors", parts=pc, placeholder=placeholder, note=note)

    # ---- the "field" variant set on /World switches every part's displayColor and the legend texture
    vset = world.GetPrim().GetVariantSets().AddVariantSet("field")
    for f in FIELDS:
        vset.AddVariant(f)
        vset.SetVariantSelection(f)
        with vset.GetVariantEditContext():
            for m, used, g, i in surfaces:
                api = UsdGeom.PrimvarsAPI(stage.OverridePrim(m.GetPath()))
                if f in ("von_mises", "u_mag"):
                    s, (lo, hi) = (vm, vr) if f == "von_mises" else (um, ur)
                    col = srgb_to_linear(cmap("turbo", (s[used] - lo) / max(hi - lo, 1e-30)))
                    pr = api.CreatePrimvar("displayColor", Sdf.ValueTypeNames.Color3fArray, UsdGeom.Tokens.vertex)
                    pr.Set(Vt.Vec3fArray.FromNumpy(col))
                else:
                    rgb = GATE_RGB[g] if f == "gate_status" else PART_RGB[i % len(PART_RGB)]
                    pr = api.CreatePrimvar("displayColor", Sdf.ValueTypeNames.Color3fArray, UsdGeom.Tokens.constant)
                    pr.Set(Vt.Vec3fArray.FromNumpy(srgb_to_linear([rgb])))
                    if f == "gate_status":  # fade passing parts so warn/fail parts stand out
                        po = api.CreatePrimvar("displayOpacity", Sdf.ValueTypeNames.FloatArray,
                                               UsdGeom.Tokens.constant)
                        po.Set(Vt.FloatArray([0.25 if g == "pass" else 1.0]))
            tx = UsdShade.Shader(stage.OverridePrim("/World/Looks/Legend/LegendTexture"))
            tx.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(f"./textures/legend_{f}.png")
    vset.SetVariantSelection(args.field)

    # ---- stage-level summary
    def clean(d):
        o = {}
        for k, v in d.items():
            if v is None:
                continue
            if isinstance(v, dict):
                o[k] = clean(v)
            elif isinstance(v, (list, tuple)):
                if v and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in v):
                    o[k] = Vt.DoubleArray([float(x) for x in v])
                elif v and all(isinstance(x, str) for x in v):
                    o[k] = Vt.StringArray(v)
            else:
                o[k] = v
        return o
    layer_sum = clean(summary)
    layer_sum["part_names"] = Vt.StringArray([str(p["name"]) for p in parts])
    layer_sum["part_gates"] = Vt.StringArray(gate_of)
    stage.GetRootLayer().customLayerData = {"feagent": layer_sum}
    world.GetPrim().SetCustomDataByKey("feagent:summary", layer_sum)

    def fmt(v, f="{:.4g}"):
        return "n/a" if v is None else f.format(v)
    doc = (("PLACEHOLDER DATA (" + placeholder + "). " if placeholder else "")
           + f"{summary['title']}: {len(parts)} part(s), {fmt(summary['n_dofs'], '{}')} DOFs, solve "
           f"{fmt(summary['solve_time_s'], '{:.2f}')} s, max von Mises {vm.max():.4g} MPa, max |u| "
           f"{umax:.4g} mm, tip deflection {fmt(td)} mm vs beam theory {fmt(bt)} mm, GCI "
           f"{fmt(summary['gci_percent'], '{:.2f}')} %. Gates: "
           + ", ".join(f"{p['name']}={gt}" for p, gt in zip(parts, gate_of))
           + f". Timeline 0..{frames}: deformation x{scale:g}. Variant set 'field' switches the colours."
           + "".join(f" {k}: {summary[k]}" for k in ("verdict", "verdict_note", "stress_note", "mesh_caveat")
                     if summary.get(k)))
    world.GetPrim().SetDocumentation(doc)
    stage.GetRootLayer().documentation = doc
    stage.GetRootLayer().Save()
    t_usd = time.time() - t_start
    summary["export_time_s"] = round(t_usd, 2)
    if not placeholder:
        summary.pop("placeholder", None)
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=float))

    if not args.no_usdz:
        z = out / f"{name}.usdz"
        if z.exists():
            z.unlink()
        UsdUtils.CreateNewUsdzPackage(Sdf.AssetPath(str(usd_path)), str(z))
    log(f"wrote {usd_path} ({usd_path.stat().st_size / 1e6:.1f} MB) in {t_usd:.1f} s"
        + ("" if args.no_usdz else f", {name}.usdz"))
    log(doc)
    return usd_path


def check(usd_path):
    """usdchecker equivalent: every registered UsdValidation validator (usd-core 26 ships no usdchecker
    script; older USD falls back to UsdUtils.ComplianceChecker)."""
    try:
        from pxr import UsdValidation as V
    except ImportError:
        cc = UsdUtils.ComplianceChecker(rootPackageOnly=False, skipVariants=False)
        cc.CheckCompliance(str(usd_path))
        bad = cc.GetErrors() + cc.GetFailedChecks()
        for e in bad + cc.GetWarnings():
            log("check:", e)
        log(f"compliance {usd_path.name}: {len(bad)} errors/failed checks")
        return not bad
    vals = V.ValidationRegistry().GetOrLoadAllValidators()
    errs = V.ValidationContext(vals).Validate(Usd.Stage.Open(str(usd_path)))
    nerr = 0
    for e in errs:
        is_err = e.GetType() == V.ValidationErrorType.Error
        nerr += is_err
        log("CHECK " + ("ERROR" if is_err else "warn") + ":", e.GetName(), e.GetMessage()[:300])
    log(f"validation {usd_path.name}: {len(vals)} validators, {nerr} errors, {len(errs) - nerr} other issues")
    return nerr == 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("manifest")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent))
    ap.add_argument("--name", default="twin")
    ap.add_argument("--frames", type=int, default=24)
    ap.add_argument("--scale", default="auto", help="deformation scale at the last frame (default: auto)")
    ap.add_argument("--max-tris", type=int, default=2_500_000)
    ap.add_argument("--field", default="von_mises", choices=FIELDS, help="default variant selection")
    ap.add_argument("--no-usdz", action="store_true")
    a = ap.parse_args()
    if a.scale != "auto":
        a.scale = float(a.scale)
    p = build(a)
    ok = check(p)
    if not a.no_usdz:
        ok &= check(p.with_suffix(".usdz"))
    sys.exit(0 if ok else 1)
