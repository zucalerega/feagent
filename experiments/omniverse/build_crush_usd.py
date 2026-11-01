"""USD time series of a crush-to-snap run (crush.py), for the Omniverse RTX video (render_crush_rtx.py).

    python experiments/omniverse/build_crush_usd.py RUN_DIR OUT_DIR [--label "mesh system"] [--fps 30]
        [--min-s 11] [--hold-s 4] [--no-lift]

RUN_DIR is a crush.py run dir; continued runs are followed back to step 0 (render_crush.series / chain, which also
finds a parent copied next to the run). Writes into OUT_DIR:
  crush.usdc   Z-up, mm (the loading axis is turned to +Z, bottom platen face at z = 0). Time code k = increment k
               (0 = unloaded), then LIFT_TC more codes for the held ending.
    /World/Part/Intact      surface of the intact cells per increment (triangles; time-sampled topology on one fixed
                            node set, so points interpolate), primvars:displayColor = turbo(von Mises / UTS) (linear;
                            the map starts at CMAP_LO, its near-black end vanishes on the dark set)
    /World/Part/Broken      surface of the broken cells (state 1), red; compact node set per sample (primvars:node_ids)
    /World/Part/Fragments   detached fragments (state 2), grey; same layout as Broken
    /World/Part/Upper       separated runs: from the last increment on, the intact piece attached to the top platen;
                            lifted LIFT_FRAC of the height over the held ending (not simulated, to open the crack)
    /World/Platens/{Bottom,Top}  brushed-metal slabs (UsdPreviewSurface, metallic, streak texture), Top follows the travel
    /World/Studio/Floor     dark floor under the bottom platen; /World/Lights key / fill / rim distant lights + a
                            dim blue-grey dome, which is also the dark backdrop
  crush_meta.json  per time code: travel, force, broken, detached, phase; per video frame: time code, camera
                   (4x4, row-major, USD convention) and the caption numbers; run facts (system, part, peak, separation)
  brushed.png      the platen streak texture
The frame plan follows render_crush.py: 1 s title at zero load, the loading part over >= --min-s s (uniform in
increments), a --hold-s s hold on the last increment (with the lift); slow orbit, low camera for abc_00000003 (its crack
runs round the bottom ring)."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "physical"))
import render_crush as RC  # noqa: E402  (series, load, upper_piece, FACES, PART_NAMES, AX)
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade, Vt  # noqa: E402

LIFT_TC = 12                     # time codes of the lift (held ending)
VIEW = {"abc_00000003": {"elev": 9.0, "az0": -140.0}, "abc_00000065": {"elev": 16.0, "az0": -125.0}}
ORBIT = 70.0
FOCAL, HAP, VAP = 35.0, 36.0, 20.25
RED, GREY = (0.95, 0.04, 0.03), (0.62, 0.64, 0.68)
KEY_I, DOME_I = 1.2e6, 40.0          # sphere-light radiance (key) and dome, for RTX with auto-exposure off



def srgb_to_linear(c):
    c = np.asarray(c, np.float64)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4).astype(np.float32)


CMAP_LO = 0.08                   # turbo from here up: its near-black end would vanish against the dark set


def turbo(x):
    from matplotlib import colormaps
    return colormaps["turbo"](CMAP_LO + (1 - CMAP_LO) * np.clip(x, 0, 1))[:, :3]


def rot_to_z(axis: int) -> np.ndarray:
    """Proper rotation taking the loading axis to +Z."""
    return {2: np.eye(3), 1: np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float),
            0: np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], float)}[axis]


class Faces:
    """Oriented boundary triangles of any cell subset of a single-type linear mesh (tets or hexes)."""

    def __init__(self, g0, pts):
        cd = g0.cells_dict
        if len(cd) != 1:
            raise SystemExit("mixed cell types are not supported")
        ct, conn = next(iter(cd.items()))
        conn = np.asarray(conn)
        fl = RC.FACES[ct]
        nf = len(fl)
        F = np.stack([conn[:, list(f)] for f in fl], axis=1)           # (cells, nf, k)
        cen = pts[conn].mean(1)
        fc = pts[F].mean(2)
        nrm = np.cross(pts[F[..., 1]] - pts[F[..., 0]], pts[F[..., 2]] - pts[F[..., 0]])
        flip = np.einsum("cfi,cfi->cf", nrm, fc - cen[:, None, :]) < 0
        F = np.where(flip[..., None], F[..., ::-1], F)                 # outward
        _, fid = np.unique(np.sort(F.reshape(-1, F.shape[-1]), axis=1), axis=0, return_inverse=True)
        self.F, self.fid, self.nf, self.n = F, fid.reshape(len(conn), nf), nf, len(conn)
        self.conn = conn

    def tris(self, mask):
        if not mask.any():
            return np.zeros((0, 3), np.int64)
        cnt = np.bincount(self.fid[mask].ravel(), minlength=self.fid.max() + 1)
        sel = mask[:, None] & (cnt[self.fid] == 1)
        f = self.F[sel]
        if f.shape[1] == 3:
            return f
        return np.concatenate([f[:, [0, 1, 2]], f[:, [0, 2, 3]]])


def box_mesh(stage, path, b0, b1, mat, tiles=2.0):
    """Axis-aligned box as a Mesh with per-face st (for the brushed texture)."""
    x0, y0, z0 = b0
    x1, y1, z1 = b1
    P = [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0), (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]
    quads = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    m = UsdGeom.Mesh.Define(stage, path)
    m.CreatePointsAttr([Gf.Vec3f(*p) for p in P])
    m.CreateFaceVertexCountsAttr([4] * 6)
    m.CreateFaceVertexIndicesAttr([i for q in quads for i in q])
    m.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    m.CreateExtentAttr([Gf.Vec3f(*b0), Gf.Vec3f(*b1)])
    st = UsdGeom.PrimvarsAPI(m).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying)
    st.Set([Gf.Vec2f(u, v) for _ in quads for (u, v) in ((0, 0), (tiles, 0), (tiles, tiles), (0, tiles))])
    UsdShade.MaterialBindingAPI.Apply(m.GetPrim()).Bind(mat)
    return m


def surface_material(stage, path, color=None, roughness=0.42, emissive=None, metallic=0.0, clearcoat=0.0):
    mat = UsdShade.Material.Define(stage, path)
    sh = UsdShade.Shader.Define(stage, f"{path}/PBR")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
    sh.CreateInput("clearcoat", Sdf.ValueTypeNames.Float).Set(clearcoat)
    sh.CreateInput("clearcoatRoughness", Sdf.ValueTypeNames.Float).Set(0.15)
    if color is None:   # per-vertex displayColor
        rd = UsdShade.Shader.Define(stage, f"{path}/DisplayColor")
        rd.CreateIdAttr("UsdPrimvarReader_float3")
        rd.CreateInput("varname", Sdf.ValueTypeNames.String).Set("displayColor")
        rd.CreateInput("fallback", Sdf.ValueTypeNames.Float3).Set(Gf.Vec3f(0.6, 0.6, 0.62))
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
            rd.ConnectableAPI(), "result", UsdShade.AttributeType.Output, Sdf.ValueTypeNames.Float3)
    else:
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    if emissive is not None:
        sh.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*emissive))
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    return mat


def brushed_material(stage, path, tex: str):
    """Brushed metal: metallic, base colour and roughness modulated by a streak texture."""
    mat = UsdShade.Material.Define(stage, path)
    sh = UsdShade.Shader.Define(stage, f"{path}/PBR")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(1.0)
    rd = UsdShade.Shader.Define(stage, f"{path}/St")
    rd.CreateIdAttr("UsdPrimvarReader_float2")
    rd.CreateInput("varname", Sdf.ValueTypeNames.String).Set("st")
    tx = UsdShade.Shader.Define(stage, f"{path}/Streaks")
    tx.CreateIdAttr("UsdUVTexture")
    tx.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(tex)
    tx.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("raw")
    tx.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
    tx.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
    tx.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(rd.ConnectableAPI(), "result")
    tx.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(0.16, 0.16, 0.17, 1))
    tx.CreateInput("bias", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(0.56, 0.57, 0.59, 0))
    tx.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
    tx.CreateOutput("r", Sdf.ValueTypeNames.Float)
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tx.ConnectableAPI(), "rgb")
    tr = UsdShade.Shader.Define(stage, f"{path}/StreaksRough")
    tr.CreateIdAttr("UsdUVTexture")
    tr.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(tex)
    tr.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("raw")
    tr.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
    tr.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
    tr.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(rd.ConnectableAPI(), "result")
    tr.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(0.22, 0.22, 0.22, 1))
    tr.CreateInput("bias", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(0.18, 0.18, 0.18, 0))
    tr.CreateOutput("r", Sdf.ValueTypeNames.Float)
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).ConnectToSource(tr.ConnectableAPI(), "r")
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    return mat


def brushed_png(path: Path, n=1024, seed=0):
    """Fine streaks along u: per-row noise smeared along the rows."""
    from PIL import Image
    rng = np.random.default_rng(seed)
    a = rng.random((n, n))
    k = 96
    c = np.cumsum(np.pad(a, ((0, 0), (k, 0)), mode="wrap"), axis=1)
    a = (c[:, k:] - c[:, :-k]) / k                           # box blur along rows
    a = 0.6 * (a - a.mean()) / (a.std() + 1e-9) + 0.4 * (rng.random((n, 1)) - 0.5) * 2
    a = np.clip(0.5 + 0.25 * a, 0, 1)
    Image.fromarray((a * 255).astype(np.uint8), "L").convert("RGB").save(path)


def look_at(eye, tgt, up=(0, 0, 1)):
    v = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*tgt), Gf.Vec3d(*up))
    m = v.GetInverse()
    return [[float(m[i][j]) for j in range(4)] for i in range(4)]


def ease(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--label", default=None)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--min-s", type=float, default=11.0)
    ap.add_argument("--hold-s", type=float, default=4.0)
    ap.add_argument("--no-lift", action="store_true")
    a = ap.parse_args(argv)
    run, out = Path(a.run_dir), Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows, runs = RC.series(run)
    meta = {}
    for r in runs:
        if (r / "run.json").exists():
            meta = {**meta, **json.loads((r / "run.json").read_text())}
    summ = json.loads((run / "summary.json").read_text()) if (run / "summary.json").exists() else {}
    part = meta.get("part") or summ.get("part")
    system = meta.get("system") or summ.get("system") or "?"
    axis = RC.AX[meta.get("axis") or {"abc_00000003": "z", "abc_00000065": "y"}[part]]
    mat = json.loads((RC.REPO / "experiments" / "material" / "alpha_fast_dogbone1.json").read_text())
    uts = float(mat["uts_eng"])
    files = [r[4] for r in rows]
    pts, _, _, _, g0 = RC.load(files[0])
    travel = np.r_[0.0, [r[1] for r in rows]]
    force = np.r_[0.0, [r[2] for r in rows]]
    n_inc = len(files) + 1
    separated = bool(summ.get("separated"))
    R = rot_to_z(axis)
    P0 = pts @ R.T
    lo, hi = P0.min(0), P0.max(0)
    off = np.array([-(lo[0] + hi[0]) / 2, -(lo[1] + hi[1]) / 2, -lo[2]])
    P0 = P0 + off
    lo, hi = P0.min(0), P0.max(0)
    height = float(hi[2] - lo[2])
    faces = Faces(g0, pts)

    # all increments: displacement (rotated), von Mises, state
    U, VM, ST = [np.zeros_like(P0)], [np.zeros(len(P0))], [np.zeros(g0.n_cells, int)]
    for f in files:
        _, u, vm, st, _ = RC.load(f)
        U.append(u @ R.T); VM.append(vm); ST.append(st)
    last_state = ST[-1]
    lift = separated and not a.no_lift
    upper = RC.upper_piece(g0, pts, last_state, axis) if lift else np.zeros(g0.n_cells, bool)
    if lift and not upper.any():
        lift, upper = False, np.zeros(g0.n_cells, bool)
    lift_mm = RC.LIFT_FRAC * height

    intact_tris = []
    for k in range(n_inc):
        m = ST[k] == 0
        if k == n_inc - 1 and lift:
            m = m & ~upper
        intact_tris.append(faces.tris(m))
    node_set = np.unique(np.concatenate([t.ravel() for t in intact_tris]))
    remap = np.full(len(P0), -1, np.int64); remap[node_set] = np.arange(len(node_set))

    stage_path = out / "crush.usdc"
    if stage_path.exists():
        stage_path.unlink()
    st = Usd.Stage.CreateNew(str(stage_path))
    UsdGeom.SetStageUpAxis(st, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(st, 0.001)
    t_end = n_inc - 1 + (LIFT_TC if lift else 0)
    st.SetStartTimeCode(0); st.SetEndTimeCode(t_end)
    st.SetTimeCodesPerSecond(24); st.SetFramesPerSecond(24)
    world = UsdGeom.Xform.Define(st, "/World"); st.SetDefaultPrim(world.GetPrim())
    UsdGeom.Scope.Define(st, "/World/Looks")
    m_part = surface_material(st, "/World/Looks/Resin", None, roughness=0.38, clearcoat=0.35)
    m_red = surface_material(st, "/World/Looks/Broken", RED, roughness=0.5, emissive=(0.55, 0.02, 0.01))
    m_grey = surface_material(st, "/World/Looks/Fragment", GREY, roughness=0.6)
    brushed_png(out / "brushed.png")
    m_metal = brushed_material(st, "/World/Looks/BrushedSteel", "./brushed.png")
    m_floor = surface_material(st, "/World/Looks/Floor", (0.045, 0.048, 0.056), roughness=0.32)
    UsdGeom.Xform.Define(st, "/World/Part")

    def mesh(path, material):
        msh = UsdGeom.Mesh.Define(st, path)
        msh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        msh.CreateDoubleSidedAttr(True)
        UsdShade.MaterialBindingAPI.Apply(msh.GetPrim()).Bind(material)
        return msh

    intact = mesh("/World/Part/Intact", m_part)
    pa, ca = intact.CreatePointsAttr(), UsdGeom.PrimvarsAPI(intact).CreatePrimvar(
        "displayColor", Sdf.ValueTypeNames.Color3fArray, UsdGeom.Tokens.vertex)
    fc, fi, ex = intact.CreateFaceVertexCountsAttr(), intact.CreateFaceVertexIndicesAttr(), intact.CreateExtentAttr()
    prev = None
    for k in range(n_inc):
        p = (P0 + U[k])[node_set].astype(np.float32)
        pa.Set(Vt.Vec3fArray.FromNumpy(p), k)
        ex.Set(Vt.Vec3fArray.FromNumpy(np.stack([p.min(0), p.max(0)])), k)
        ca.Set(Vt.Vec3fArray.FromNumpy(srgb_to_linear(turbo(VM[k][node_set] / uts))), k)
        tri = remap[intact_tris[k]]
        key = (len(tri), int(tri.sum()))
        if key != prev or k in (0, n_inc - 1):
            fc.Set(Vt.IntArray.FromNumpy(np.full(len(tri), 3, np.int32)), k)
            fi.Set(Vt.IntArray.FromNumpy(tri.ravel().astype(np.int32)), k)
            prev = key

    def compact(path, material, masks, extra_tc=()):
        """Mesh per time code from cell masks, own node set per sample (primvars:node_ids)."""
        msh = mesh(path, material)
        pa_, fc_, fi_, ex_ = (msh.CreatePointsAttr(), msh.CreateFaceVertexCountsAttr(),
                              msh.CreateFaceVertexIndicesAttr(), msh.CreateExtentAttr())
        ids_ = UsdGeom.PrimvarsAPI(msh).CreatePrimvar("node_ids", Sdf.ValueTypeNames.IntArray, UsdGeom.Tokens.constant)
        vis = msh.CreateVisibilityAttr()
        for k, mk in masks:
            tri = faces.tris(mk)
            used = np.unique(tri.ravel())
            if not len(used):
                vis.Set(UsdGeom.Tokens.invisible, k)
                pa_.Set(Vt.Vec3fArray(), k); fc_.Set(Vt.IntArray(), k); fi_.Set(Vt.IntArray(), k)
                ids_.Set(Vt.IntArray(), k)
                continue
            vis.Set(UsdGeom.Tokens.inherited, k)
            rm = np.full(len(P0), -1, np.int64); rm[used] = np.arange(len(used))
            p = (P0 + U[min(k, n_inc - 1)])[used].astype(np.float32)
            pa_.Set(Vt.Vec3fArray.FromNumpy(p), k)
            ex_.Set(Vt.Vec3fArray.FromNumpy(np.stack([p.min(0), p.max(0)])), k)
            fc_.Set(Vt.IntArray.FromNumpy(np.full(len(tri), 3, np.int32)), k)
            fi_.Set(Vt.IntArray.FromNumpy(rm[tri].ravel().astype(np.int32)), k)
            ids_.Set(Vt.IntArray.FromNumpy(used.astype(np.int32)), k)
        return msh

    compact("/World/Part/Broken", m_red, [(k, ST[k] == 1) for k in range(n_inc)])
    compact("/World/Part/Fragments", m_grey, [(k, ST[k] == 2) for k in range(n_inc)])
    lift_tr = None
    if lift:
        up = compact("/World/Part/Upper", m_part, [(n_inc - 2, np.zeros(g0.n_cells, bool)), (n_inc - 1, upper)])
        um = UsdGeom.PrimvarsAPI(up).CreatePrimvar("displayColor", Sdf.ValueTypeNames.Color3fArray, UsdGeom.Tokens.vertex)
        used = np.unique(faces.tris(upper).ravel())
        um.Set(Vt.Vec3fArray.FromNumpy(srgb_to_linear(turbo(VM[-1][used] / uts))), n_inc - 1)
        um.Set(Vt.Vec3fArray(), n_inc - 2)
        lift_tr = up.AddTranslateOp()

    # platens
    pad = 0.15 * (hi - lo)
    th = 0.06 * height
    UsdGeom.Xform.Define(st, "/World/Platens")
    box_mesh(st, "/World/Platens/Bottom", (lo[0] - pad[0], lo[1] - pad[1], -th), (hi[0] + pad[0], hi[1] + pad[1], 0.0),
             m_metal)
    top = UsdGeom.Xform.Define(st, "/World/Platens/TopXf")
    box_mesh(st, "/World/Platens/TopXf/Top", (lo[0] - pad[0], lo[1] - pad[1], height),
             (hi[0] + pad[0], hi[1] + pad[1], height + th), m_metal)
    ttr = top.AddTranslateOp()
    for k in range(n_inc):
        ttr.Set(Gf.Vec3d(0, 0, -travel[k]), k)
    if lift:
        for j in range(LIFT_TC + 1):
            d = ease(j / LIFT_TC) * lift_mm
            ttr.Set(Gf.Vec3d(0, 0, -travel[-1] + d), n_inc - 1 + j)
            lift_tr.Set(Gf.Vec3d(0, 0, d), n_inc - 1 + j)
        lift_tr.Set(Gf.Vec3d(0, 0, 0), 0)

    # studio: dark floor under the bottom platen and a curved backdrop sweep behind the part
    size = float(np.linalg.norm(hi - lo))
    UsdGeom.Xform.Define(st, "/World/Studio")
    R0 = 80 * size
    box_mesh(st, "/World/Studio/Floor", (-R0, -R0, -th - 0.02 * size), (R0, R0, -th - 0.01 * size), m_floor, tiles=1)

    # studio lights: three shaped sphere lights (key / fill / rim spots) aimed at the part, so the floor falls off into
    # the dark; a dim dome for ambient fill and the backdrop colour. Intensities for RTX with auto-exposure off.
    UsdGeom.Scope.Define(st, "/World/Lights")
    v = VIEW.get(part, {"elev": 18.0, "az0": -40.0})
    tgt = np.array([0.0, 0.0, 0.45 * height])
    az_key = math.radians(v["az0"] + ORBIT / 2 + 40)
    lights = (("Key", az_key, 42, 3.2, 0.16, KEY_I, 24, (1.0, 0.96, 0.9)),
              ("Fill", az_key - 1.9, 18, 3.6, 0.22, 0.3 * KEY_I, 32, (0.82, 0.88, 1.0)),
              ("Rim", az_key + math.pi - 0.35, 50, 3.0, 0.12, 0.9 * KEY_I, 22, (0.9, 0.95, 1.0)))
    for name, az, el, dmul, rmul, inten, cone, col in lights:
        lt = UsdLux.SphereLight.Define(st, f"/World/Lights/{name}")
        lt.CreateRadiusAttr(rmul * size); lt.CreateIntensityAttr(float(inten)); lt.CreateColorAttr(Gf.Vec3f(*col))
        sh = UsdLux.ShapingAPI.Apply(lt.GetPrim())
        sh.CreateShapingConeAngleAttr(float(cone)); sh.CreateShapingConeSoftnessAttr(0.35)
        e = math.radians(el)
        d = np.array([math.cos(e) * math.cos(az), math.cos(e) * math.sin(az), math.sin(e)])
        lt.AddTransformOp().Set(Gf.Matrix4d(*[x for row in look_at(tgt + d * dmul * size, tgt) for x in row]))
    dome = UsdLux.DomeLight.Define(st, "/World/Lights/Dome")
    dome.CreateIntensityAttr(DOME_I); dome.CreateColorAttr(Gf.Vec3f(0.55, 0.6, 0.72))

    # frame plan: same pacing as render_crush.frame_plan; time code = increment (+ lift codes in the hold)
    plan = RC.frame_plan(n_inc, a.fps, a.min_s, a.hold_s, lift)
    ctr = np.array([0.0, 0.0, 0.5 * height - 0.10 * height])
    span = np.r_[hi[:2] - lo[:2] + 2 * pad[:2], height + 2 * th + (lift_mm if lift else 0.0)]
    fov_v = 2 * math.atan(VAP / (2 * FOCAL))
    fov_h = 2 * math.atan(HAP / (2 * FOCAL))
    rad = 0.5 * float(np.linalg.norm(span))
    dist = max(rad / math.sin(fov_v / 2) * 1.08, rad / math.sin(fov_h / 2) * 1.08)
    frames = []
    for f, (ia, ib, t, lf) in enumerate(plan):
        tc = ia + t * (ib - ia) if lf == 0 else (n_inc - 1) + lf * LIFT_TC
        az = math.radians(v["az0"] + ORBIT * ease(f / max(len(plan) - 1, 1)))
        el = math.radians(v["elev"])
        eye = ctr + dist * np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
        s = min(max(t, 0.0), 1.0)
        tr = (1 - s) * travel[ia] + s * travel[ib]
        fo = (1 - s) * force[ia] + s * force[ib]
        stt = ST[ib] if t > 0.5 else ST[ia]
        frames.append({"tc": round(float(tc), 5), "cam": look_at(eye, ctr), "travel_mm": round(float(tr), 5),
                       "force_N": round(float(fo), 2), "broken": int((stt == 1).sum()), "detached": int((stt == 2).sum()),
                       "lift_mm": round(lf * lift_mm, 3), "inc": int(ib if t > 0.5 else ia)})
    peak_i = int(np.argmax(force))
    label = a.label or system
    import pyvista as pv
    ctype = {24: "C3D10 (tet10)", 10: "C3D4 (tet4)", 12: "C3D8 (hex)", 25: "C3D20 (hex20)"}.get(
        int(pv.read(files[0]).celltypes[0]), "")
    info = {"system": system, "label": label, "part": part, "part_name": RC.PART_NAMES.get(part, ""),
            "run_dirs": [r.name for r in runs], "n_increments": n_inc, "separated": separated,
            "peak_force_N": float(force[peak_i]), "travel_at_peak_mm": float(travel[peak_i]),
            "travel_at_separation_mm": summ.get("travel_at_separation_mm"), "n_nodes": int(g0.n_points),
            "n_cells": int(g0.n_cells), "cell_type": ctype, "uts_MPa": uts, "failure_peeq": float(mat["failure_peeq"]),
            "lift": bool(lift), "lift_mm": lift_mm, "lift_tc": LIFT_TC, "fps": a.fps, "width": 1920, "height": 1080,
            "lens": {"focal": FOCAL, "hap": HAP, "vap": VAP}, "time_codes": [0, t_end],
            "per_tc": {"travel_mm": travel.round(6).tolist(), "force_N": force.round(3).tolist(),
                       "broken": [int((s_ == 1).sum()) for s_ in ST], "detached": [int((s_ == 2).sum()) for s_ in ST]},
            "frames": frames}
    st.GetRootLayer().customLayerData = {"feagent": {"system": system, "part": part, "kind": "crush time series",
                                                     "vm_color_range_MPa": Vt.DoubleArray([0.0, uts])}}
    st.GetRootLayer().Save()
    (out / "crush_meta.json").write_text(json.dumps(info))
    mb = stage_path.stat().st_size / 1e6
    print(f"[build_crush_usd] {stage_path}: {n_inc} increments, {len(node_set)} surface nodes, {len(frames)} frames "
          f"({len(frames) / a.fps:.1f} s), lift {lift} ({lift_mm:.1f} mm), {mb:.0f} MB")


if __name__ == "__main__":
    main()
