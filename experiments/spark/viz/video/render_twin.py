"""Omniverse RTX video frames of the FEA virtual twin (twin.usdc), rendered headless by Isaac Sim 6.2.

Run inside the Isaac Sim 6.2.0 container (make_video.sh does this):
    ./python.sh /twin/video/render_twin.py --usd /twin/twin.usdc --out /out [--renderer auto|pt|rt] [--stages a.usdc,b.usdc]
Plan only (no Isaac Sim, no GPU; needs usd-core, e.g. the feagent micromamba env):
    python render_twin.py --plan-only --usd ~/feagent-viz/twin.usdc --out ./plan

Writes <out>/<shot>/frame_00000.png ... (1920x1080 RGBA PNG) and <out>/manifest.json (shots, frame counts,
renderer actually used, timings, the singularity label's pixel anchor, gallery titles). make_video.sh turns
these into wingbox_twin.mp4 with assemble.py.

Shot list (30 fps):
  a_orbit    6 s  field=von_mises, t=0 (undeformed), slow 70 deg orbit about the stage's 3/4 view
  b_deform   5 s  field=von_mises, stage camera held, time code 0 -> 24 (deformation x20) then held
  c_root     3 s  field=von_mises, t=24, static close-up of the clamped root corner + 3D leader marker
  d_umag     3 s  field=u_mag, t=24, slow drift
  e_gates    3 s  field=gate_status, t=24, drift continues
  f_closing  4 s  field=von_mises, t=24, pull-back then hold (closing title is drawn by ffmpeg)
  g_<k>      3 s  per --stages entry: turntable of an extra stage (gold-parts gallery hook)

Renderer (--renderer):
  pt   RTX path tracing ("PathTracing"), 64 spp/frame, OptiX denoiser. Best quality (soft shadows, GI).
  rt   RTX Real-Time 2.0 ("RealTimePathTracing", the 6.2 default), DLSS Quality, 8 subframes per frame.
  auto (default) path tracing, unless the first timed frames cost more than --pt-budget-s s/frame; then it
       switches to rt for the whole run and re-renders what it had written. The choice is in manifest.json.

Isaac Sim 6.2 APIs used, checked against the v6.2.0 sources/docs on 2026-10-31:
  SimulationApp config keys renderer/samples_per_pixel_per_frame/denoiser/anti_aliasing/disable_viewport_updates
    source/extensions/isaacsim.simulation_app/isaacsim/simulation_app/simulation_app.py (DEFAULT_LAUNCHER_CONFIG)
  isaacsim.core.experimental.utils.stage.open_stage / is_stage_loading
    source/extensions/isaacsim.core.experimental.utils/python/impl/stage.py
  rep.create.render_product, rep.AnnotatorRegistry.get_annotator("rgb").attach, rep.orchestrator.set_capture_on_play,
  rep.orchestrator.step(rt_subframes=, delta_time=0.0, pause_timeline=), rep.orchestrator.wait_until_complete,
  omni.replicator.core.functional.write_image, "/rtx-transient/post/dlss/forceParamReset", "/rtx/rendermode",
  "/rtx/pathtracing/spp", "/rtx/pathtracing/totalSpp", "/rtx/pathtracing/optixDenoiser/enabled", "/rtx/post/dlss/execMode"
    docs.isaacsim.omniverse.nvidia.com/6.2.0/replicator_tutorials/tutorial_replicator_isaac_snippets.html and
    source/standalone_examples/api/isaacsim.replicator.examples/{simulation_get_data,motion_blur_pathtracing}.py
  omni.timeline: get_timeline_interface, set_current_time, get_time_codes_per_seconds, commit (motion_blur_pathtracing.py)
Not verified until the first GPU run: that a paused-timeline set_current_time + orchestrator.step re-poses the
time-sampled points (fallback: --deform-mode points), the RTX exposure of the stage's lights (auto exposure is on;
--light-exposure adds stops), and the PathTracing cost on GB10.
"""
import argparse
import json
import math
import os
import re
import sys
import time
from pathlib import Path

W, H = 1920, 1080
FPS = 30
FOCAL, HAP, VAP = 35.0, 36.0, 20.25            # mm; 16:9, same lens as the stage's /World/Camera
SHOTS = [  # name, seconds, field, what
    ("a_orbit", 6.0, "von_mises", "orbit"),
    ("b_deform", 5.0, "von_mises", "deform"),
    ("c_root", 3.0, "von_mises", "root"),
    ("d_umag", 3.0, "u_mag", "drift1"),
    ("e_gates", 3.0, "gate_status", "drift2"),
    ("f_closing", 4.0, "von_mises", "pullback"),
]
GALLERY_S = 3.0
SINGULAR_DEFAULT = (600.0, 0.0, 150.0)


# ------------------------------------------------------------------ small vector maths (no numpy needed)
def add(a, b): return tuple(x + y for x, y in zip(a, b))
def sub(a, b): return tuple(x - y for x, y in zip(a, b))
def mul(a, s): return tuple(x * s for x in a)
def dot(a, b): return sum(x * y for x, y in zip(a, b))
def norm(a): return math.sqrt(dot(a, a))
def unit(a): return mul(a, 1.0 / norm(a))
def cross(a, b): return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])
def lerp(a, b, s): return a + (b - a) * s
def ease(s): s = min(max(s, 0.0), 1.0); return s * s * (3 - 2 * s)


def rotate(v, axis, ang):
    """Rodrigues rotation of v about unit axis."""
    c, s = math.cos(ang), math.sin(ang)
    return add(add(mul(v, c), mul(cross(axis, v), s)), mul(axis, dot(axis, v) * (1 - c)))


def look_at(eye, target, up):
    """Camera-to-world matrix (USD row-vector convention: rows = right, up, back, eye); camera looks down -Z."""
    back = unit(sub(eye, target))
    right = unit(cross(up, back))
    upv = cross(back, right)
    return [list(right) + [0.0], list(upv) + [0.0], list(back) + [0.0], list(eye) + [1.0]]


def project(m, p):
    """Pixel (u, v) of world point p for camera matrix m (None if behind the camera)."""
    d = sub(p, m[3][:3])
    xc, yc, zc = dot(d, m[0][:3]), dot(d, m[1][:3]), dot(d, m[2][:3])
    if zc >= -1e-6:
        return None
    return (W / 2 + xc / -zc * FOCAL / HAP * W, H / 2 - yc / -zc * FOCAL / VAP * H)


def coverage(m, lo, hi):
    """Screen bbox of the 8 corners of [lo, hi], as fractions of the frame (x0, y0, x1, y1)."""
    pts = [project(m, (x, y, z)) for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]
    pts = [p for p in pts if p]
    if not pts:
        return None
    return [round(min(p[0] for p in pts) / W, 3), round(min(p[1] for p in pts) / H, 3),
            round(max(p[0] for p in pts) / W, 3), round(max(p[1] for p in pts) / H, 3)]


# ------------------------------------------------------------------ the plan (pure python + pxr reads)
def stage_info(stage, Usd, UsdGeom):
    up = (0.0, 0.0, 1.0) if UsdGeom.GetStageUpAxis(stage) == UsdGeom.Tokens.z else (0.0, 1.0, 0.0)
    t0, t1 = stage.GetStartTimeCode(), stage.GetEndTimeCode()
    root = stage.GetPrimAtPath("/World/Wingbox")
    if not root or not root.IsValid():  # export_twin layout: /World/<Assembly> next to Looks/Lights/Legend/Camera
        w = stage.GetPrimAtPath("/World")
        kids = [c for c in (w.GetChildren() if w and w.IsValid() else [])
                if c.GetName() not in ("Looks", "Lights", "Legend", "Camera") and c.IsA(UsdGeom.Xformable)]
        root = kids[0] if kids else (stage.GetDefaultPrim() or stage.GetPseudoRoot())
    bc = UsdGeom.BBoxCache(Usd.TimeCode(t0), ["default", "render"])
    r0 = bc.ComputeWorldBound(root).ComputeAlignedRange()
    bc.SetTime(Usd.TimeCode(t1))
    r1 = bc.ComputeWorldBound(root).ComputeAlignedRange()
    lo = tuple(min(a, b) for a, b in zip(r0.GetMin(), r1.GetMin()))
    hi = tuple(max(a, b) for a, b in zip(r0.GetMax(), r1.GetMax()))
    cam = None
    cp = stage.GetPrimAtPath("/World/Camera")
    if cp and cp.IsValid():
        m = UsdGeom.Xformable(cp).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        cam = [[float(m[i][j]) for j in range(4)] for i in range(4)]
    w = stage.GetPrimAtPath("/World")
    fields = []
    if w and w.IsValid() and w.GetVariantSets().HasVariantSet("field"):
        fields = list(w.GetVariantSet("field").GetVariantNames())
    meta = (stage.GetRootLayer().customLayerData or {}).get("feagent", {})
    vr = meta.get("vm_color_range_MPa")
    return {"up": up, "t0": t0, "t1": t1, "lo": lo, "hi": hi, "cam": cam, "fields": fields,
            "vm_range": [float(v) for v in vr] if vr is not None else None,
            "clip": meta.get("vm_clip_percentile"), "scale": meta.get("deformation_scale"),
            "tcps": stage.GetTimeCodesPerSecond(), "title": str(meta.get("title", "")),
            "stress_note": str(meta.get("stress_note", ""))}


def singular_point(info, summary):
    note = (summary or {}).get("stress_note") or info.get("stress_note") or ""
    m = re.search(r"\((-?[\d.]+),\s*(-?[\d.]+),\s*(-?[\d.]+)\)", note)
    return tuple(float(g) for g in m.groups()) if m else SINGULAR_DEFAULT


def orbit_basis(up):
    ref = (1.0, 0.0, 0.0) if abs(up[0]) < 0.9 else (0.0, 1.0, 0.0)
    e1 = unit(sub(ref, mul(up, dot(ref, up))))
    return e1, cross(up, e1)


def main_plan(info, summary):
    """Per-frame camera matrix, time code and field for the twin's shots."""
    up, lo, hi = info["up"], info["lo"], info["hi"]
    ctr = mul(add(lo, hi), 0.5)
    diag = norm(sub(hi, lo))
    fov_v = 2 * math.atan(VAP / (2 * FOCAL))
    if info["cam"]:
        eye0 = tuple(info["cam"][3][:3])
        fwd = mul(tuple(info["cam"][2][:3]), -1.0)
        target0 = add(eye0, mul(fwd, dot(sub(ctr, eye0), fwd)))
    else:
        target0 = ctr
        eye0 = add(ctr, mul(unit((0.55, -1.0, 0.65)), 0.5 * diag * 1.25 / math.sin(fov_v / 2) * 0.9))
    off0 = sub(eye0, target0)
    tend = info["t1"]
    sp = singular_point(info, summary)
    shots = []
    for name, secs, field, kind in SHOTS:
        n = int(round(secs * FPS))
        frames = []
        for i in range(n):
            s = i / max(n - 1, 1)
            tc, extra = tend, {}
            if kind == "orbit":
                tc = info["t0"]
                off = rotate(off0, up, math.radians(-35 + 70 * ease(s)))
                eye, tgt = add(target0, mul(off, 0.95)), target0
            elif kind == "deform":
                tc = info["t0"] + (tend - info["t0"]) * ease(min(1.0, s / 0.84))  # ramp 4.2 s, hold 0.8 s
                eye, tgt = eye0, target0
            elif kind == "root":
                d = unit((0.80, -0.95, 0.62))
                eye, tgt = add(sp, mul(d, 470.0)), add(sp, (-60.0, 70.0, -25.0))
            elif kind in ("drift1", "drift2"):
                a = (0.0 if kind == "drift1" else 6.0) + 6.0 * s
                eye, tgt = add(target0, rotate(off0, up, math.radians(a))), target0
            else:  # pullback
                k = lerp(1.0, 1.45, ease(min(1.0, s / 0.55)))
                off = rotate(off0, up, math.radians(-8 * ease(min(1.0, s / 0.55))))
                eye, tgt = add(target0, mul(off, k)), add(target0, mul(up, 0.06 * diag * ease(min(1.0, s / 0.55))))
            m = look_at(eye, tgt, up)
            frames.append({"cam": m, "tc": tc, **extra})
        shot = {"name": name, "seconds": secs, "field": field, "kind": kind, "frames": frames,
                "coverage_first": coverage(frames[0]["cam"], lo, hi), "coverage_last": coverage(frames[-1]["cam"], lo, hi)}
        if kind == "root":
            tip = add(sp, mul(up, 90.0))
            a = project(frames[0]["cam"], tip)
            shot["label"] = {"point_mm": list(sp), "anchor_mm": list(tip),
                             "anchor_px": [round(a[0]), round(a[1])] if a else None,
                             "point_px": [round(v) for v in (project(frames[0]["cam"], sp) or (0, 0))]}
        shots.append(shot)
    return shots, sp


def gallery_plan(info, k, label):
    up, lo, hi = info["up"], info["lo"], info["hi"]
    ctr = mul(add(lo, hi), 0.5)
    diag = max(norm(sub(hi, lo)), 1e-6)
    fov_v = 2 * math.atan(VAP / (2 * FOCAL))
    dist = 0.5 * diag * 1.15 / math.sin(fov_v / 2)
    e1, e2 = orbit_basis(up)
    n = int(round(GALLERY_S * FPS))
    frames = []
    for i in range(n):
        az = math.radians(-60 + 180 * i / max(n - 1, 1))
        el = math.radians(25)
        d = add(add(mul(e1, math.cos(el) * math.cos(az)), mul(e2, math.cos(el) * math.sin(az))), mul(up, math.sin(el)))
        frames.append({"cam": look_at(add(ctr, mul(d, dist)), ctr, up), "tc": info["t1"]})
    return {"name": f"g_{k:02d}", "seconds": GALLERY_S, "field": "von_mises" if "von_mises" in info["fields"] else None,
            "kind": "turntable", "frames": frames, "title": info["title"] or label,
            "vm_range": info["vm_range"], "clip": info["clip"], "scale": info["scale"],
            "coverage_first": coverage(frames[0]["cam"], lo, hi)}


def summarize(shots):
    out = []
    for s in shots:
        d = {k: v for k, v in s.items() if k != "frames"}
        d["n_frames"] = len(s["frames"])
        d["tc_first"], d["tc_last"] = s["frames"][0]["tc"], s["frames"][-1]["tc"]
        d["eye_first"] = [round(v, 1) for v in s["frames"][0]["cam"][3][:3]]
        d["eye_last"] = [round(v, 1) for v in s["frames"][-1]["cam"][3][:3]]
        out.append(d)
    return out


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--usd", default="/twin/twin.usdc")
    ap.add_argument("--summary", default=None, help="summary.json (default: next to --usd)")
    ap.add_argument("--out", default="/out")
    ap.add_argument("--stages", default="", help="comma-separated extra stages: a 3 s turntable each (gallery)")
    ap.add_argument("--ramps", default="", help="comma-separated load-ramp stages (build_ramp.py): render ONLY these, "
                    "one --ramp-seconds clip each, time codes start..end, stage camera with a slight drift")
    ap.add_argument("--ramp-seconds", type=float, default=5.0)
    ap.add_argument("--renderer", default="auto", choices=["auto", "pt", "rt"])
    ap.add_argument("--spp", type=int, default=64, help="path-tracing samples per pixel per frame")
    ap.add_argument("--pt-subframes", type=int, default=2)
    ap.add_argument("--rt-subframes", type=int, default=8)
    ap.add_argument("--pt-budget-s", type=float, default=4.0, help="auto: fall back to rt above this s/frame")
    ap.add_argument("--warmup", type=int, default=30, help="app updates after the stage loads")
    ap.add_argument("--cut-warmup", type=int, default=3, help="extra render steps at each cut / field switch")
    ap.add_argument("--deform-mode", default="timeline", choices=["timeline", "points", "bake"],
                    help="timeline: set the timeline time; points: author the interpolated points in the session layer; "
                         "bake: author every time-sampled attribute's value at t in the session layer (ramps use this)")
    ap.add_argument("--auto-exposure", type=int, default=1)
    ap.add_argument("--light-exposure", type=float, default=0.0, help="stops added to the Key and Dome lights")
    ap.add_argument("--shots", default="", help="comma-separated subset of shot names (default all)")
    ap.add_argument("--preview", action="store_true", help="one frame (the middle) per shot + a pt/rt timing probe")
    ap.add_argument("--resume", action="store_true", help="skip frames whose PNG already exists")
    ap.add_argument("--capture-with-update", action="store_true", help="render with app.update() instead of "
                    "rep.orchestrator.step (automatic fallback if the orchestrator returns no data)")
    ap.add_argument("--plan-only", action="store_true", help="print/write the plan with usd-core; no Isaac Sim")
    return ap.parse_args(argv)


def load_summary(args):
    p = Path(args.summary) if args.summary else Path(args.usd).with_name("summary.json")
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def ramp_plan(info, k, path, seconds):
    up, lo, hi = info["up"], info["lo"], info["hi"]
    ctr = mul(add(lo, hi), 0.5)
    eye0 = tuple(info["cam"][3][:3])
    fwd = mul(tuple(info["cam"][2][:3]), -1.0)
    tgt = add(eye0, mul(fwd, dot(sub(ctr, eye0), fwd)))
    off = sub(eye0, tgt)
    n = int(round(seconds * FPS))
    frames = []
    for i in range(n):
        s = i / max(n - 1, 1)
        eye = add(tgt, rotate(off, up, math.radians(-5 + 10 * s)))
        frames.append({"cam": look_at(eye, tgt, up), "tc": info["t0"] + (info["t1"] - info["t0"]) * s})
    return {"name": f"r_{k:02d}", "seconds": seconds, "field": None, "kind": "ramp", "frames": frames,
            "title": info["title"] or Path(path).stem, "usd": path, "vm_range": info["vm_range"],
            "clip": info["clip"], "scale": info["scale"], "coverage_first": coverage(frames[0]["cam"], lo, hi)}


def build_plan(args, Usd, UsdGeom):
    if args.ramps:
        rs = [s for s in args.ramps.split(",") if s.strip()]
        return None, [ramp_plan(stage_info(Usd.Stage.Open(p), Usd, UsdGeom), k, p, args.ramp_seconds)
                      for k, p in enumerate(rs)], None
    summary = load_summary(args)
    info = stage_info(Usd.Stage.Open(args.usd), Usd, UsdGeom)
    shots, sp = main_plan(info, summary)
    if args.shots:
        keep = set(args.shots.split(","))
        shots = [s for s in shots if s["name"] in keep]
    gallery = [s for s in args.stages.split(",") if s.strip()]
    for k, path in enumerate(gallery):
        gi = stage_info(Usd.Stage.Open(path), Usd, UsdGeom)
        g = gallery_plan(gi, k, Path(path).stem)
        g["usd"] = path
        shots.append(g)
    return info, shots, sp


# ------------------------------------------------------------------ plan-only (host, no GPU)
def plan_only(args):
    from pxr import Usd, UsdGeom
    info, shots, sp = build_plan(args, Usd, UsdGeom)
    info = info or {"lo": None, "hi": None, "t0": None, "t1": None, "fields": None}
    plan = {"usd": args.usd, "fps": FPS, "width": W, "height": H, "singular_point_mm": sp,
            "bbox_mm": [info["lo"], info["hi"]], "time_codes": [info["t0"], info["t1"]], "fields": info["fields"],
            "shots": summarize(shots), "n_frames": sum(len(s["frames"]) for s in shots)}
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "plan.json").write_text(json.dumps(plan, indent=1))
    for s in plan["shots"]:
        print(f"[plan] {s['name']:10s} {s['n_frames']:4d} frames  field={s['field']}  tc {s['tc_first']:.1f}->"
              f"{s['tc_last']:.1f}  model on screen {s.get('coverage_first')} -> {s.get('coverage_last')}"
              + (f"  label px {s['label']['anchor_px']} (peak px {s['label']['point_px']})" if s.get("label") else ""))
    print(f"[plan] {plan['n_frames']} frames = {plan['n_frames'] / FPS:.1f} s at {FPS} fps -> {Path(args.out) / 'plan.json'}")


# ------------------------------------------------------------------ the render (inside Isaac Sim)
def render(args):
    renderer = {"auto": "PathTracing", "pt": "PathTracing", "rt": "RealTimePathTracing"}[args.renderer]
    from isaacsim import SimulationApp
    app = SimulationApp({"headless": True, "width": W, "height": H, "renderer": renderer,
                         "samples_per_pixel_per_frame": args.spp, "denoiser": True, "anti_aliasing": 3,
                         "disable_viewport_updates": False})
    import carb.settings
    import isaacsim.core.experimental.utils.stage as stage_utils
    import omni.replicator.core as rep
    import omni.timeline
    from omni.replicator.core.functional import write_image
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade
    from concurrent.futures import ThreadPoolExecutor
    try:
        from PIL import Image
    except Exception:
        Image = None

    os.umask(0)                      # container runs as uid 1234: let the host user delete what it writes
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    settings = carb.settings.get_settings()
    t_start = time.time()
    info, shots, sp = build_plan(args, Usd, UsdGeom)
    log = lambda *a: print("[render_twin]", *a, flush=True)  # noqa: E731

    def set_mode(mode):
        settings.set("/rtx/rendermode", mode)
        if mode == "PathTracing":
            settings.set("/rtx/pathtracing/spp", args.spp)
            settings.set("/rtx/pathtracing/totalSpp", args.spp)
            settings.set("/rtx/pathtracing/optixDenoiser/enabled", True)
        else:
            settings.set("/rtx/post/dlss/execMode", 2)  # Quality
        return args.pt_subframes if mode == "PathTracing" else args.rt_subframes

    pool = ThreadPoolExecutor(max_workers=4)
    pending = []

    def save(path, data):
        if Image is not None:
            arr = data[:, :, :3].copy()     # rgb annotator: H x W x 4 uint8
            pending.append(pool.submit(lambda: Image.fromarray(arr).save(path, compress_level=3)))
        else:
            write_image(path=str(path), data=data)

    def open_stage(path):
        ok, stage = stage_utils.open_stage(path)
        if not ok:
            raise SystemExit(f"could not open {path}")
        while stage_utils.is_stage_loading():
            app.update()
        for _ in range(args.warmup):
            app.update()
        return stage

    def dress(stage, with_marker):
        """Session-layer edits only (the twin is mounted read-only): shot camera, light exposure, marker."""
        with Usd.EditContext(stage, stage.GetSessionLayer()):
            UsdGeom.Scope.Define(stage, "/Video")
            cam = UsdGeom.Camera.Define(stage, "/Video/ShotCam")
            cam.CreateFocalLengthAttr(FOCAL)
            cam.CreateHorizontalApertureAttr(HAP)
            cam.CreateVerticalApertureAttr(VAP)
            cam.CreateClippingRangeAttr(Gf.Vec2f(5.0, 200000.0))
            op = cam.AddTransformOp()
            if args.light_exposure:
                for lp in ("/World/Lights/Key", "/World/Lights/Dome"):
                    p = stage.GetPrimAtPath(lp)
                    if p and p.IsValid():
                        a = p.GetAttribute("inputs:exposure") or p.CreateAttribute("inputs:exposure", Sdf.ValueTypeNames.Float)
                        a.Set(float(args.light_exposure))
            if with_marker:  # thin red leader from the singular peak up to the label anchor, plus a bead
                mat = UsdShade.Material.Define(stage, "/Video/MarkerMat")
                sh = UsdShade.Shader.Define(stage, "/Video/MarkerMat/PBR")
                sh.CreateIdAttr("UsdPreviewSurface")
                sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.9, 0.05, 0.05))
                sh.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(1.0, 0.1, 0.1))
                mat.CreateSurfaceOutput().ConnectToSource(sh.CreateOutput("surface", Sdf.ValueTypeNames.Token))
                up = Gf.Vec3d(*info["up"])
                base = Gf.Vec3d(*sp) + up * 10.0
                cyl = UsdGeom.Cylinder.Define(stage, "/Video/Marker/Leader")
                cyl.CreateRadiusAttr(1.6)
                cyl.CreateHeightAttr(80.0)
                cyl.CreateAxisAttr("Z" if info["up"][2] else "Y")
                cyl.AddTranslateOp().Set(base + up * 40.0)
                bead = UsdGeom.Sphere.Define(stage, "/Video/Marker/Bead")
                bead.CreateRadiusAttr(5.0)
                bead.AddTranslateOp().Set(base + up * 80.0)
                for g in (cyl, bead):
                    UsdShade.MaterialBindingAPI.Apply(g.GetPrim()).Bind(mat)
        return op

    def make_rp():
        rp = rep.create.render_product("/Video/ShotCam", (W, H), name="feagent_video")
        rgb = rep.AnnotatorRegistry.get_annotator("rgb")
        rgb.attach(rp)
        return rp, rgb

    tl = omni.timeline.get_timeline_interface()

    baked = {}

    def set_time(stage, tc, mode=None):
        mode = mode or args.deform_mode
        if mode == "bake":
            key = id(stage)
            if key not in baked:
                baked[key] = [a for p in stage.Traverse() for a in p.GetAttributes() if a.GetNumTimeSamples() > 1]
            with Usd.EditContext(stage, stage.GetSessionLayer()):
                for a in baked[key]:
                    a.Set(a.Get(Usd.TimeCode(tc)))
            return
        if mode == "timeline":
            tl.set_current_time(tc / tl.get_time_codes_per_seconds())
            if hasattr(tl, "commit"):
                tl.commit()
        else:
            with Usd.EditContext(stage, stage.GetSessionLayer()):
                for prim in stage.Traverse():
                    if prim.IsA(UsdGeom.Mesh):
                        a = UsdGeom.Mesh(prim).GetPointsAttr()
                        if a.GetNumTimeSamples() > 1:
                            a.Set(a.Get(Usd.TimeCode(tc)))

    def set_field(stage, field):
        w = stage.GetPrimAtPath("/World")
        if field and w and w.IsValid() and w.GetVariantSets().HasVariantSet("field"):
            if field in w.GetVariantSet("field").GetVariantNames():
                with Usd.EditContext(stage, stage.GetSessionLayer()):
                    w.GetVariantSet("field").SetVariantSelection(field)

    capture = {"orch": not args.capture_with_update}

    def step(sub):
        if capture["orch"]:
            rep.orchestrator.step(rt_subframes=sub, delta_time=0.0, pause_timeline=True)
        else:            # render the render product with plain app updates (timeline stays paused)
            for _ in range(max(1, sub)):
                app.update()

    def frame_data():
        d = rgb.get_data()
        if isinstance(d, dict):
            d = d.get("data")
        ok = d is not None and getattr(d, "ndim", 0) == 3 and d.shape[0] == H and d.shape[1] == W
        return d if ok else None

    def wait_ready(max_updates=900):
        """First frames: shaders compile and materials load; render until the annotator returns a full frame.
        Falls back from rep.orchestrator.step to app.update() if the orchestrator never fills the annotator."""
        for k in range(max_updates):
            if k == 60 and capture["orch"]:
                log("orchestrator.step gave no rgb data in 60 tries: switching to app.update() capture")
                capture["orch"] = False
            step(1)
            if frame_data() is not None:
                log(f"renderer ready after {k + 1} steps ({'orchestrator' if capture['orch'] else 'app.update'})")
                return True
        raise SystemExit("no rgb data from the render product")

    def mat4(m):
        return Gf.Matrix4d(*[v for row in m for v in row])

    rep.orchestrator.set_capture_on_play(False)
    settings.set("/rtx/post/histogram/enabled", bool(args.auto_exposure))
    tl.pause()

    manifest = {"usd": args.usd, "fps": FPS, "width": W, "height": H, "singular_point_mm": sp,
                "isaac_sim": "6.2.0", "gpu": "NVIDIA GB10 (DGX Spark)", "shots": [], "renderer_requested": args.renderer}

    def render_shots(stage, op, shot_list, mode, sub, probe=False):
        """Returns mean s/frame of the timed frames."""
        times = []
        for shot in shot_list:
            sdir = out / shot["name"]
            sdir.mkdir(parents=True, exist_ok=True)
            frames = shot["frames"]
            idx = [len(frames) // 2] if args.preview else range(len(frames))
            if probe:
                idx = range(min(len(frames), 6))
            set_field(stage, shot["field"])
            lg = stage.GetPrimAtPath("/World/Legend")
            if lg and lg.IsValid() and shot["kind"] == "turntable":  # the legend quad would spin edge-on
                with Usd.EditContext(stage, stage.GetSessionLayer()):
                    UsdGeom.Imageable(lg).MakeInvisible()
            mk = stage.GetPrimAtPath("/Video/Marker")
            if mk and mk.IsValid():
                with Usd.EditContext(stage, stage.GetSessionLayer()):
                    im = UsdGeom.Imageable(mk)
                    im.MakeVisible() if shot["kind"] == "root" else im.MakeInvisible()
            t_shot = time.time()
            first = True
            for i in idx:
                f = frames[i]
                path = sdir / f"frame_{i:05d}.png"
                if args.resume and path.exists() and not probe:
                    continue
                op.Set(mat4(f["cam"]))
                set_time(stage, f["tc"], "bake" if shot["kind"] == "ramp" else None)
                if first:  # a cut: new camera / field; flush temporal history and let materials settle
                    settings.set("/rtx-transient/post/dlss/forceParamReset", True)
                    for _ in range(args.cut_warmup):
                        step(sub)
                    first = False
                t0 = time.time()
                step(sub)
                data = frame_data()
                times.append(time.time() - t0)
                if data is None:
                    log(f"WARNING empty frame {shot['name']} {i}")
                    continue
                save(path, data)
                if i % 30 == 0 or args.preview:
                    done = sum(len(s["frames"]) for s in manifest["shots"]) + i + 1
                    mean = sum(times) / len(times)
                    total = sum(len(s["frames"]) for s in shots)
                    log(f"{shot['name']} frame {i + 1}/{len(frames)}  {mean:.2f} s/frame ({mode}), "
                        f"ETA {(total - done) * mean / 60:.1f} min")
            if not probe:
                entry = {k: v for k, v in summarize([shot])[0].items()}
                entry.update({"dir": shot["name"], "render_s": round(time.time() - t_shot, 1),
                              "renderer": mode, "rt_subframes": sub})
                manifest["shots"].append(entry)
        return sum(times) / len(times) if times else 0.0

    if args.ramps:   # load-ramp clips only (no twin)
        mode = renderer
        if args.renderer == "auto":
            mode = "RealTimePathTracing"
        rp = rgb = None
        for g in shots:
            if rp is not None:
                rgb.detach()
                rp.destroy()
            stage = open_stage(g["usd"])
            op = dress(stage, with_marker=False)
            rp, rgb = make_rp()
            sub = set_mode(mode)
            wait_ready()
            render_shots(stage, op, [g], mode, sub)
            manifest["shots"][-1].update({"title": g["title"], "usd": g["usd"], "vm_range": g["vm_range"],
                                          "clip": g["clip"], "scale": g["scale"]})
        manifest["renderer"] = mode
        rep.orchestrator.wait_until_complete()
        for p in pending:
            p.result()
        manifest["n_frames"] = sum(m["n_frames"] for m in manifest["shots"])
        manifest["total_s"] = round(time.time() - t_start, 1)
        manifest["preview"] = bool(args.preview)
        (out / ("manifest_preview.json" if args.preview else "manifest.json")).write_text(json.dumps(manifest, indent=1))
        log(f"done: {manifest['n_frames']} ramp frames in {manifest['total_s']} s -> {out}")
        app.close()
        return

    # --- the twin
    stage = open_stage(args.usd)
    op = dress(stage, with_marker=True)
    rp, rgb = make_rp()
    wait_ready()
    log(f"stage ready in {time.time() - t_start:.1f} s; {sum(len(s['frames']) for s in shots)} frames planned")

    mode = renderer
    sub = set_mode(mode)
    probe = {}
    twin_shots = [s for s in shots if s["kind"] != "turntable"]
    if args.renderer == "auto" or args.preview:
        probe["PathTracing"] = render_shots(stage, op, twin_shots[:1], "PathTracing", set_mode("PathTracing"), probe=True)
        if args.preview:
            probe["RealTimePathTracing"] = render_shots(stage, op, twin_shots[:1], "RealTimePathTracing",
                                                        set_mode("RealTimePathTracing"), probe=True)
        log("timing probe (s/frame incl. subframes): " + ", ".join(f"{k} {v:.2f}" for k, v in probe.items()))
        if args.renderer == "auto":
            mode = "PathTracing" if probe["PathTracing"] <= args.pt_budget_s else "RealTimePathTracing"
            log(f"auto: using {mode} (path tracing {probe['PathTracing']:.2f} s/frame, budget {args.pt_budget_s} s)")
        sub = set_mode(mode)
    manifest["probe_s_per_frame"] = probe
    manifest["capture"] = "rep.orchestrator.step" if capture["orch"] else "app.update"
    manifest["renderer"] = mode
    manifest["spp"] = args.spp if mode == "PathTracing" else None

    render_shots(stage, op, twin_shots, mode, sub)
    for s in shots:
        if s.get("label"):
            next(m for m in manifest["shots"] if m["name"] == s["name"])["label"] = s["label"]

    # --- gallery hook: extra stages, 3 s turntable each
    for g in [s for s in shots if s["kind"] == "turntable"]:
        rgb.detach()
        rp.destroy()
        stage = open_stage(g["usd"])
        op = dress(stage, with_marker=False)
        rp, rgb = make_rp()
        sub = set_mode(mode)
        wait_ready()
        render_shots(stage, op, [g], mode, sub)
        manifest["shots"][-1].update({"title": g["title"], "usd": g["usd"], "vm_range": g["vm_range"],
                                      "clip": g["clip"], "scale": g["scale"]})

    rep.orchestrator.wait_until_complete()
    for p in pending:
        p.result()
    manifest["n_frames"] = sum(m["n_frames"] for m in manifest["shots"])
    manifest["total_s"] = round(time.time() - t_start, 1)
    manifest["preview"] = bool(args.preview)
    (out / ("manifest_preview.json" if args.preview else "manifest.json")).write_text(json.dumps(manifest, indent=1))
    log(f"done: {manifest['n_frames']} frames ({'preview' if args.preview else 'full'}) in {manifest['total_s']} s -> {out}")
    try:
        rgb.detach()
        rp.destroy()
    except Exception:
        pass
    app.close()


if __name__ == "__main__":
    a = parse_args()
    if a.plan_only:
        plan_only(a)
    else:
        render(a)
