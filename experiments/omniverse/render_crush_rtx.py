"""Omniverse RTX frames of a crush time series (build_crush_usd.py), rendered headless by Isaac Sim 6.2.

Run inside the Isaac Sim 6.2.0 container (make_crush_video.sh does this):
    ./python.sh /crush/render_crush_rtx.py --usd /work/crush.usdc --meta /work/crush_meta.json --out /out
        [--renderer rt|pt] [--rt-subframes 8] [--only 0,200,-1] [--resume]

Per frame of crush_meta.json "frames": the camera matrix goes on a session-layer camera, and every time-sampled
attribute of the stage is authored into the session layer at the frame's (fractional) time code:
  * float arrays of equal length (points, displayColor, extent) and translations are interpolated linearly;
  * topology, node ids and visibility take the nearer sample;
  * a prim with primvars:node_ids (Broken / Fragments / Upper) is interpolated only when the node ids of both samples
    agree, otherwise it takes the nearer sample (its points then match its topology).
Values are read from a second, read-only stage of the same file, so session-layer defaults never mask the samples.
--light-scales 0.5,1,2 (with --only): each frame once per scale of all light intensities (exposure calibration).
Writes <out>/frame_%05d.png (1920x1080 RGB) and <out>/manifest.json (renderer, timings).

Isaac Sim 6.2 APIs (as ~/feagent-viz/video/render_twin.py, checked against the v6.2.0 sources): SimulationApp config,
isaacsim.core.experimental.utils.stage.open_stage / is_stage_loading, rep.create.render_product,
rep.AnnotatorRegistry.get_annotator("rgb"), rep.orchestrator.step(rt_subframes=, delta_time=0.0, pause_timeline=True),
"/rtx/rendermode", "/rtx/post/dlss/execMode", "/rtx-transient/post/dlss/forceParamReset", "/rtx/post/histogram/enabled".
"""
import argparse
import json
import math
import os
import time
from pathlib import Path


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--usd", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--renderer", default="rt", choices=["rt", "pt"])
    ap.add_argument("--spp", type=int, default=32)
    ap.add_argument("--rt-subframes", type=int, default=8)
    ap.add_argument("--pt-subframes", type=int, default=2)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--auto-exposure", type=int, default=0)
    ap.add_argument("--light-scales", default="", help="--only look: render each frame once per light scale (comma list)")
    ap.add_argument("--only", default="", help="comma-separated frame indices (negative = from the end): a look first")
    ap.add_argument("--resume", action="store_true")
    return ap.parse_args(argv)


def main(args):
    meta = json.loads(Path(args.meta).read_text())
    W, H = meta.get("width", 1920), meta.get("height", 1080)
    lens = meta["lens"]
    frames = meta["frames"]
    idx = list(range(len(frames)))
    if args.only:
        idx = sorted({int(v) % len(frames) for v in args.only.split(",")})
    renderer = {"rt": "RealTimePathTracing", "pt": "PathTracing"}[args.renderer]
    from isaacsim import SimulationApp
    app = SimulationApp({"headless": True, "width": W, "height": H, "renderer": renderer,
                         "samples_per_pixel_per_frame": args.spp, "denoiser": True, "anti_aliasing": 3,
                         "disable_viewport_updates": False})
    import carb.settings
    import isaacsim.core.experimental.utils.stage as stage_utils
    import numpy as np
    import omni.replicator.core as rep
    import omni.timeline
    from concurrent.futures import ThreadPoolExecutor
    from PIL import Image
    from pxr import Gf, Sdf, Usd, UsdGeom

    os.umask(0)                       # container uid 1234: the host user can delete what it writes
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    settings = carb.settings.get_settings()
    t_start = time.time()
    log = lambda *a: print("[render_crush_rtx]", *a, flush=True)  # noqa: E731

    ok, stage = stage_utils.open_stage(args.usd)
    if not ok:
        raise SystemExit(f"could not open {args.usd}")
    while stage_utils.is_stage_loading():
        app.update()
    for _ in range(args.warmup):
        app.update()
    src = Usd.Stage.Open(args.usd)    # read-only twin: samples stay readable under the session-layer defaults

    # animated attributes, grouped per prim
    groups = []
    for prim in src.Traverse():
        attrs = [a for a in prim.GetAttributes() if a.GetNumTimeSamples() > 0]
        ops = []
        if prim.IsA(UsdGeom.Xformable):
            ops = [a for a in attrs if a.GetName().startswith("xformOp:")]
        if attrs:
            ids = prim.GetAttribute("primvars:node_ids")
            groups.append((str(prim.GetPath()), attrs, ids if ids and ids.GetNumTimeSamples() > 0 else None, ops))
    log(f"{len(groups)} animated prims, {sum(len(g[1]) for g in groups)} attributes")
    t_end = src.GetEndTimeCode()
    cache = {}

    def val(attr, k):
        key = (str(attr.GetPath()), k)
        if key not in cache:
            cache[key] = attr.Get(Usd.TimeCode(k))
        return cache[key]

    def is_float_array(v):
        return hasattr(v, "__len__") and type(v).__name__ in ("Vec3fArray", "FloatArray", "Vec3dArray", "Color3fArray")

    def set_time(tc):
        a = int(math.floor(tc))
        b = min(a + 1, int(t_end))
        w = tc - a
        if len(cache) > 4000:
            cache.clear()
        near = a if w < 0.5 else b
        with Usd.EditContext(stage, stage.GetSessionLayer()):
            for path, attrs, ids, _ in groups:
                same = True
                if ids is not None:
                    ia, ib = val(ids, a), val(ids, b)
                    same = len(ia) == len(ib) and (len(ia) == 0 or np.array_equal(np.asarray(ia), np.asarray(ib)))
                prim = stage.GetPrimAtPath(path)
                for at in attrs:
                    dst = prim.GetAttribute(at.GetName())
                    va, vb = val(at, a), val(at, b)
                    if w == 0 or a == b:
                        v = va
                    elif not same:
                        v = val(at, near)
                    elif is_float_array(va) and len(va) == len(vb) and len(va):
                        x = np.asarray(va) * (1 - w) + np.asarray(vb) * w
                        v = type(va).FromNumpy(x.astype(np.asarray(va).dtype))
                    elif isinstance(va, (Gf.Vec3d, Gf.Vec3f)):
                        v = va * (1 - w) + vb * w
                    else:
                        v = val(at, near)
                    dst.Set(v)

    with Usd.EditContext(stage, stage.GetSessionLayer()):
        UsdGeom.Scope.Define(stage, "/Video")
        cam = UsdGeom.Camera.Define(stage, "/Video/ShotCam")
        cam.CreateFocalLengthAttr(lens["focal"])
        cam.CreateHorizontalApertureAttr(lens["hap"])
        cam.CreateVerticalApertureAttr(lens["vap"])
        cam.CreateClippingRangeAttr(Gf.Vec2f(1.0, 1.0e6))
        cop = cam.AddTransformOp()
    rp = rep.create.render_product("/Video/ShotCam", (W, H), name="feagent_crush")
    rgb = rep.AnnotatorRegistry.get_annotator("rgb")
    rgb.attach(rp)
    rep.orchestrator.set_capture_on_play(False)
    settings.set("/rtx/post/histogram/enabled", bool(args.auto_exposure))
    settings.set("/rtx/rendermode", renderer)
    if renderer == "PathTracing":
        settings.set("/rtx/pathtracing/spp", args.spp)
        settings.set("/rtx/pathtracing/totalSpp", args.spp)
        settings.set("/rtx/pathtracing/optixDenoiser/enabled", True)
        sub = args.pt_subframes
    else:
        settings.set("/rtx/post/dlss/execMode", 2)   # Quality
        sub = args.rt_subframes
    omni.timeline.get_timeline_interface().pause()
    capture = {"orch": True}

    def step(n):
        if capture["orch"]:
            rep.orchestrator.step(rt_subframes=n, delta_time=0.0, pause_timeline=True)
        else:
            for _ in range(max(1, n)):
                app.update()

    def frame_data():
        d = rgb.get_data()
        if isinstance(d, dict):
            d = d.get("data")
        return d if d is not None and getattr(d, "ndim", 0) == 3 and d.shape[0] == H and d.shape[1] == W else None

    def mat4(m):
        return Gf.Matrix4d(*[v for row in m for v in row])

    cop.Set(mat4(frames[idx[0]]["cam"]))
    set_time(frames[idx[0]]["tc"])
    for k in range(900):              # shaders compile, materials load
        if k == 60 and capture["orch"]:
            log("orchestrator.step gave no rgb data in 60 tries: app.update() capture")
            capture["orch"] = False
        step(1)
        if frame_data() is not None:
            log(f"renderer ready after {k + 1} steps, {time.time() - t_start:.1f} s")
            break
    else:
        raise SystemExit("no rgb data from the render product")
    settings.set("/rtx-transient/post/dlss/forceParamReset", True)
    for _ in range(3):
        step(sub)

    scales = [float(v) for v in args.light_scales.split(",")] if args.light_scales else [None]
    lights0 = {}
    for p in stage.Traverse():
        a_ = p.GetAttribute("inputs:intensity")
        if a_ and a_.IsValid() and a_.Get() is not None:
            lights0[str(p.GetPath())] = a_.Get()

    def light_scale(s_):
        with Usd.EditContext(stage, stage.GetSessionLayer()):
            for path_, v_ in lights0.items():
                stage.GetPrimAtPath(path_).GetAttribute("inputs:intensity").Set(v_ * s_)

    pool = ThreadPoolExecutor(max_workers=4)
    pending, times, t_loop = [], [], time.time()
    jobs = [(i, sc) for i in idx for sc in scales]
    for n, (i, sc) in enumerate(jobs):
        path = out / (f"frame_{i:05d}.png" if sc is None else f"frame_{i:05d}_x{sc:g}.png")
        if sc is not None:
            light_scale(sc)
        if args.resume and path.exists():
            continue
        f = frames[i]
        t0 = time.time()
        cop.Set(mat4(f["cam"]))
        set_time(f["tc"])
        if args.only:                 # isolated frames: no temporal history to reuse
            settings.set("/rtx-transient/post/dlss/forceParamReset", True)
            step(sub)
        step(sub)
        d = frame_data()
        times.append(time.time() - t0)
        if d is None:
            log(f"WARNING empty frame {i}")
            continue
        arr = d[:, :, :3].copy()
        pending.append(pool.submit(lambda p=path, x=arr: Image.fromarray(x).save(p, compress_level=3)))
        if n % 30 == 0:
            m = sum(times) / len(times)
            log(f"frame {i + 1}/{len(frames)}  {m:.2f} s/frame, ETA {(len(jobs) - n - 1) * m / 60:.1f} min")
    rep.orchestrator.wait_until_complete()
    for p in pending:
        p.result()
    man = {"usd": args.usd, "renderer": renderer, "rt_subframes": sub if renderer != "PathTracing" else None,
           "spp": args.spp if renderer == "PathTracing" else None, "n_frames": len(idx), "fps": meta.get("fps", 30),
           "s_per_frame": round(sum(times) / max(len(times), 1), 3), "render_loop_s": round(time.time() - t_loop, 1),
           "total_s": round(time.time() - t_start, 1), "isaac_sim": "6.2.0", "gpu": "NVIDIA GB10 (DGX Spark)",
           "capture": "rep.orchestrator.step" if capture["orch"] else "app.update", "only": args.only or None}
    (out / ("manifest_only.json" if args.only else "manifest.json")).write_text(json.dumps(man, indent=1))
    log(f"done: {len(idx)} frames in {man['total_s']} s ({man['s_per_frame']} s/frame) -> {out}")
    try:
        rgb.detach()
        rp.destroy()
    except Exception:
        pass
    app.close()


if __name__ == "__main__":
    main(parse_args())
