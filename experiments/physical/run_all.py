"""One entry point for the multi-system mesh study (experiments/mesh_systems/CONTRACT.md).

    python experiments/physical/run_all.py --meshsys ROOT --study OUT [--systems a,b] [--parts p,q]
        [--build] [--rebuild] [--no-linear] [--no-videos] [--crush] [--crush-runs DIR] [--no-crush-videos]
        [--threads 4] [--render-workers 4] [--dry-run]

Stages, in order (each one skips work whose output already exists, so the script is rerunnable):
  build      (--build) each system's build.py STEP AXIS ROOT/<layout path> --purpose linear|nonlinear, only for
             mesh dirs without report.json (--rebuild: always); then ROOT/_canon/<system>/<part>/<purpose> symlinks.
  linear     linear_ramp.py per system x part -> OUT/linear/<system>/<part>/{result.vtu, summary.json}.
  f_full     per part, identical for every system: the force that brings the 99.9th-percentile volume-weighted von
             Mises to 87 % of the UTS, median over the systems solved -> OUT/f_full.json (also the per-part
             deformation magnification of the videos).
  videos     render_linear.py -> OUT/videos/<system>_<part>_linear.mp4 (10 s ramp + 3 s hold, 30 fps, 1920x1080),
             then OUT/videos/all_systems_<part>_linear.mp4 (ffmpeg xstack, 3 x 2).
  crush      (--crush) crush.py per system x part on the nonlinear mesh, into --crush-runs (default
             runs/nonlinear/<system>/<part>/<run-id>); runs one after the other, each with --crush-threads.
  crush videos  render_crush.py on the newest run dir per system x part (or the dirs listed with --crush-run
             SYSTEM:PART:DIR) -> OUT/videos/<system>_<part>_crush.mp4.
  compare    compare.py -> OUT/comparison.{md,json}, OUT/comparison.png.
CPU only (CUDA_VISIBLE_DEVICES=""; software GL for rendering)."""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
import meshsys  # noqa: E402

FRACTION_OF_UTS = 0.87
NICE_MAG = (1, 2, 3, 5, 10, 20, 30, 50, 100, 200, 500, 1000)
ENV = {"CUDA_VISIBLE_DEVICES": "", "LIBGL_ALWAYS_SOFTWARE": "1", "GALLIUM_DRIVER": "llvmpipe",
       "__GLX_VENDOR_LIBRARY_NAME": "mesa"}


def run(cmd, dry, env=None, log=None):
    print("+", " ".join(shlex.quote(str(c)) for c in cmd), flush=True)
    if dry:
        return 0
    e = {**os.environ, **ENV, **(env or {})}
    if log:
        Path(log).parent.mkdir(parents=True, exist_ok=True)
        with open(log, "w") as fh:
            return subprocess.run([str(c) for c in cmd], env=e, stdout=fh, stderr=subprocess.STDOUT).returncode
    return subprocess.run([str(c) for c in cmd], env=e).returncode


def stage_build(a, root):
    lay = meshsys.layout()
    for s in a.systems:
        sysdef = lay["systems"][s]
        pp = [str(root / p) for p in sysdef.get("pythonpath", []) if (root / p).exists()]
        env = {"PYTHONPATH": os.pathsep.join(pp + [os.environ.get("PYTHONPATH", "")]).strip(os.pathsep)} if pp else {}
        for p in a.parts:
            for u in meshsys.PURPOSES:
                d = meshsys.mesh_dir(root, s, p, u)
                if meshsys.ready(d) and not a.rebuild:
                    continue
                step = REPO / lay["parts"][p]["step"]
                run([sys.executable, REPO / "experiments" / "mesh_systems" / s / "build.py", step, meshsys.axis(p), d,
                     "--purpose", u], a.dry_run, env, a.study / "logs" / f"build_{s}_{p}_{u}.log")
    meshsys.canon(root)


def stage_linear(a, root):
    for s in a.systems:
        for p in a.parts:
            md = meshsys.mesh_dir(root, s, p, "linear")
            out = a.study / "linear" / s / p
            if not meshsys.ready(md):
                print(f"skip linear {s}/{p}: no mesh in {md}")
                continue
            if (out / "summary.json").exists() and not a.resolve and \
                    (out / "summary.json").stat().st_mtime > (md / "mesh.msh").stat().st_mtime:
                continue
            run([sys.executable, HERE / "linear_ramp.py", "--system", s, "--part", p, "--mesh-dir", md, "--out", out,
                 "--threads", a.threads], a.dry_run, log=a.study / "logs" / f"linear_{s}_{p}.log")


def nice_mag(x):
    return max([m for m in NICE_MAG if m <= x] or [1])


def stage_f_full(a):
    """F_full per part from every linear summary under OUT/linear (all systems solved, not only the selected)."""
    mat = json.loads((REPO / "experiments" / "material" / "alpha_fast_dogbone1.json").read_text())
    uts = float(mat["uts_eng"]) * 1e6
    res = {"fraction_of_uts": FRACTION_OF_UTS, "uts_Pa": uts, "rule": "F_full = median over systems of "
           "fraction * UTS * F_ref / (99.9th-percentile volume-weighted von Mises at F_ref)", "parts": {}}
    import numpy as np
    for p in meshsys.parts():
        per, umax, ext = {}, [], []
        for f in sorted((a.study / "linear").glob(f"*/{p}/summary.json")):
            s = json.loads(f.read_text())
            per[s["system"]] = FRACTION_OF_UTS * uts * s["force_N"] / s["robust_von_mises_Pa"]
            umax.append(s["max_u_mag_m"] / s["force_N"])
        if not per:
            continue
        F = float(np.median(list(per.values())))
        lo, hi = _bbox(a, p)
        mag_raw = 0.06 * max(hi - lo) / (float(np.median(umax)) * F)
        res["parts"][p] = {"F_full_N": round(F, 1), "per_system_N": {k: round(v, 1) for k, v in per.items()},
                           "spread_pct": round(100 * (max(per.values()) - min(per.values())) / F, 2),
                           "video_mag": nice_mag(mag_raw), "video_mag_rule": "largest of 1,2,3,5,10,... with the "
                           "magnified max |u| at F_full <= 6 % of the largest bbox side"}
    out = a.study / "f_full.json"
    if not a.dry_run:
        out.write_text(json.dumps(res, indent=1))
    print(json.dumps({p: v["F_full_N"] for p, v in res["parts"].items()}))
    return res


def _bbox(a, part):
    import numpy as np
    import meshio
    for f in sorted((a.study / "linear").glob(f"*/{part}/result.vtu")):
        pts = meshio.read(f).points
        return pts.min(0), pts.max(0)
    return np.zeros(3), np.ones(3)


def stage_videos(a, ff):
    vid = a.study / "videos"
    jobs = []
    for s in a.systems:
        for p in a.parts:
            rd = a.study / "linear" / s / p
            if not (rd / "summary.json").exists() or p not in ff["parts"]:
                continue
            out = vid / f"{s}_{p}_linear.mp4"
            if out.exists() and out.stat().st_mtime > (rd / "summary.json").stat().st_mtime and not a.rerender:
                continue
            cfg = ff["parts"][p]
            if (frames := vid / f"{s}_{p}_linear_frames").exists() and a.rerender:
                for fr in frames.glob("f*.png"):
                    fr.unlink()
            jobs.append([sys.executable, HERE / "render_linear.py", rd, "--f-full", cfg["F_full_N"], "--mag",
                         cfg["video_mag"], "--out", out, "--label", meshsys.label(s)])
    env = {"LP_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}
    with ThreadPoolExecutor(max_workers=a.render_workers) as ex:     # one single-threaded renderer per video
        list(ex.map(lambda c: run(c, a.dry_run, env, a.study / "logs" / (Path(c[-3]).stem + ".log")), jobs))
    for p in a.parts:
        compile_grid(a, p, "linear")


def compile_grid(a, part, kind):
    vid = a.study / "videos"
    ins = [vid / f"{s}_{part}_{kind}.mp4" for s in meshsys.systems()]
    ins = [f for f in ins if f.exists()]
    if len(ins) < 2:
        return
    w, h, cols = 1280, 720, 3
    rows = (len(ins) + cols - 1) // cols
    filt = "".join(f"[{i}:v]scale={w}:{h},setsar=1[v{i}];" for i in range(len(ins)))
    n = len(ins)
    if n < rows * cols:                                          # fill the grid with black
        filt += "".join(f"color=c=0x1d2330:s={w}x{h}:d=600[b{j}];" for j in range(rows * cols - n))
    labels = [f"[v{i}]" for i in range(n)] + [f"[b{j}]" for j in range(rows * cols - n)]
    layout = "|".join(f"{(i % cols) * w}_{(i // cols) * h}" for i in range(rows * cols))
    filt += "".join(labels) + f"xstack=inputs={rows * cols}:layout={layout}:shortest=1[out]"
    out = vid / f"all_systems_{part}_{kind}.mp4"
    cmd = ["ffmpeg", "-y", "-loglevel", "error"] + sum([["-i", str(f)] for f in ins], []) + \
        ["-filter_complex", filt, "-map", "[out]", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
         "-threads", "4", str(out)]
    run(cmd, a.dry_run)


def newest_run(base: Path, system: str, part: str) -> Path | None:
    cands = [d for d in (base / system / part).glob("*") if (d / "force_displacement.csv").exists()] \
        if (base / system / part).exists() else []
    if not cands and system == "harness_auto":                      # baseline runs predate the per-system layout
        cands = [d for d in (base / part).glob("run-*") if (d / "force_displacement.csv").exists()]
    return max(cands, key=lambda d: d.stat().st_mtime) if cands else None


def stage_crush(a, root):
    for s in a.systems:
        for p in a.parts:
            md = meshsys.mesh_dir(root, s, p, "nonlinear")
            if not meshsys.ready(md):
                print(f"skip crush {s}/{p}: no nonlinear mesh in {md}")
                continue
            cmd = [sys.executable, HERE / "crush.py", "--system", s, "--part", p, "--mesh-dir", md,
                   "--threads", a.crush_threads]
            if a.crush_runs:
                cmd += ["--out", Path(a.crush_runs) / s / p]
            run(cmd, a.dry_run, log=a.study / "logs" / f"crush_{s}_{p}_{time.strftime('%Y%m%d-%H%M')}.log")


def crush_dirs(a):
    base = Path(a.crush_runs) if a.crush_runs else Path(os.environ.get("FEACORE_RUNS_DIR", REPO / "runs")) / "nonlinear"
    explicit = {}
    for x in a.crush_run or []:
        s, p, d = x.split(":", 2)
        explicit[(s, p)] = Path(d)
    out = {}
    for s in a.systems:
        for p in a.parts:
            d = explicit.get((s, p)) or newest_run(base, s, p)
            if d is not None:
                out[(s, p)] = d
    return out


def stage_crush_videos(a):
    jobs = []
    for (s, p), d in crush_dirs(a).items():
        out = a.study / "videos" / f"{s}_{p}_crush.mp4"
        newest = max((f.stat().st_mtime for f in d.glob("result_*.vtu")), default=0)
        if out.exists() and out.stat().st_mtime > newest and not a.rerender:
            continue
        jobs.append([sys.executable, HERE / "render_crush.py", d, "--out", out, "--label", meshsys.label(s),
                     "--workers", a.render_workers])
    for c in jobs:          # one video at a time, its frames over --render-workers processes
        run(c, a.dry_run, {"LP_NUM_THREADS": "1"}, a.study / "logs" / (Path(c[4]).stem + ".log"))
    for p in a.parts:
        compile_grid(a, p, "crush")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__.split("\n\n", 2)[-1])
    ap.add_argument("--meshsys", default=os.environ.get("MESHSYS_OUT"), help="mesh-systems root (MESHSYS_OUT)")
    ap.add_argument("--study", required=True, help="results root")
    ap.add_argument("--systems", default=",".join(meshsys.systems()))
    ap.add_argument("--parts", default=",".join(meshsys.parts()))
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--no-linear", action="store_true")
    ap.add_argument("--resolve", action="store_true", help="re-solve linear runs that exist")
    ap.add_argument("--no-videos", action="store_true")
    ap.add_argument("--rerender", action="store_true", help="re-render videos that exist")
    ap.add_argument("--crush", action="store_true", help="launch crush.py runs (hours each)")
    ap.add_argument("--crush-runs", default=None, help="crush run root (default runs/nonlinear)")
    ap.add_argument("--crush-run", action="append", help="SYSTEM:PART:RUN_DIR to render/compare (repeatable)")
    ap.add_argument("--crush-threads", type=int, default=8)
    ap.add_argument("--no-crush-videos", action="store_true")
    ap.add_argument("--no-compare", action="store_true")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--render-workers", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    if not a.meshsys:
        ap.error("--meshsys (or MESHSYS_OUT) is required")
    root, a.study = Path(a.meshsys), Path(a.study)
    a.systems, a.parts = a.systems.split(","), a.parts.split(",")
    a.study.mkdir(parents=True, exist_ok=True)
    if a.build or a.rebuild:
        stage_build(a, root)
    else:
        meshsys.canon(root)
    if not a.no_linear:
        stage_linear(a, root)
    ff = stage_f_full(a)
    if not a.no_videos:
        stage_videos(a, ff)
    if a.crush:
        stage_crush(a, root)
    if not a.no_crush_videos:
        stage_crush_videos(a)
    if not a.no_compare:
        cmd = [sys.executable, HERE / "compare.py", "--meshsys", root, "--study", a.study]
        for (s, p), d in crush_dirs(a).items():
            cmd += ["--crush-run", f"{s}:{p}:{d}"]
        run(cmd, a.dry_run)


if __name__ == "__main__":
    main()
