"""Keep the crush videos of the multi-system study up to date while the runs finish (run on the laptop).

    python experiments/omniverse/watch_crush_videos.py --spark dell@<spark-host> --runs runs/nonlinear \
        --out DIR [--every 600] [--max-hours 30] [--after 2040] [--skip run-...-s0,...] [--once]

Every --every seconds it scans RUNS/<system>/<part>/run-* (the laptop's own runs plus the copies the crush-queue sync
pulls from the Spark) and picks, per system x part, the newest run whose id time is >= --after (HHMM; Spark ids are
US Central) and whose summary.json says it separated or has a stop_reason; the baseline (RUNS/<part>/<run>,
system harness_auto) is given with --baseline PART:RUN_ID. For each run not rendered yet, on the Spark:
  1. original look: experiments/physical/render_crush.py -> <system>_<part>_crush.mp4;
  2. Omniverse: experiments/omniverse/make_crush_video.sh -> <system>_<part>_crush_rtx.mp4 (retried next round if
     the preflight refuses, e.g. MemAvailable too low);
  3. snap comparison of the part over every finished system: physical/snap_compare.py -> <part>_snap_compare.mp4;
then copies the videos to --out and a contact sheet of each to --out/sheets/ (look at them). The Spark side lives in
SPARK_DIR (default ~/feagent-viz/crush: repo/ with experiments/{physical,omniverse,material}, runs/, work/); a run
that is not on the Spark under SPARK_RUNS (~/feagent-runs/nonlinear) is rsynced to SPARK_DIR/runs first.
State: --out/watch_state.json. Exits when every expected system x part is done, or after --max-hours."""
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import time
from pathlib import Path

SYSTEMS = ["harness_auto", "harness_hex", "gold_nested", "ace_sizing", "netgen", "harness_expert"]
PARTS = ["abc_00000003", "abc_00000065"]
LABELS = {"harness_auto": "harness auto (feagent automesh)", "harness_hex": "harness hex (all-hex C3D8)",
          "gold_nested": "gold nested (uniform split, L1)", "ace_sizing": "ACE sizing (global h + hotspots)",
          "netgen": "Netgen (OCC geometry)", "harness_expert": "harness expert (digest, local refinement)"}
ENV = "CUDA_VISIBLE_DEVICES= LP_NUM_THREADS=1 PATH=$HOME/micromamba/envs/feagent-ffmpeg/bin:$PATH"
PY = "$HOME/micromamba/envs/feagent/bin/python"
CAP = "systemd-run --user --scope -q -p MemoryMax=24G"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def ssh(host, cmd, check=True):
    log("spark:", cmd[:200])
    return subprocess.run(["ssh", "-o", "BatchMode=yes", host, cmd], text=True, capture_output=True, check=check)


def finished(d: Path) -> bool:
    try:
        s = json.loads((d / "summary.json").read_text())
    except Exception:
        return False
    return bool(s.get("separated") or s.get("stop_reason"))


def candidates(runs: Path, after: int, skip: set, baseline: dict):
    out = {}
    for part, rid in baseline.items():
        d = runs / part / rid
        if finished(d):
            out[("harness_auto", part)] = d
    for s in SYSTEMS:
        for p in PARTS:
            if (s, p) in out:
                continue
            best = None
            for d in sorted((runs / s / p).glob("run-*")):
                m = re.match(r"run-\d{8}-(\d{4})", d.name)
                if not m or int(m.group(1)) < after or d.name in skip or not finished(d):
                    continue
                best = d
            if best is not None:
                out[(s, p)] = best
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--spark", required=True)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--every", type=int, default=600)
    ap.add_argument("--max-hours", type=float, default=30)
    ap.add_argument("--after", type=int, default=2040)
    ap.add_argument("--skip", default="")
    ap.add_argument("--baseline", action="append", default=[], help="PART:RUN_ID (system harness_auto)")
    ap.add_argument("--spark-dir", default="~/feagent-viz/crush")
    ap.add_argument("--spark-runs", default="~/feagent-runs/nonlinear")
    ap.add_argument("--spark-videos", default="~/feagent-viz/videos/mesh_study/crush")
    ap.add_argument("--once", action="store_true")
    a = ap.parse_args(argv)
    runs, out = Path(a.runs), Path(a.out)
    (out / "sheets").mkdir(parents=True, exist_ok=True)
    stf = out / "watch_state.json"
    state = json.loads(stf.read_text()) if stf.exists() else {}
    skip = set(filter(None, a.skip.split(",")))
    baseline = dict(x.split(":", 1) for x in a.baseline)
    home = ssh(a.spark, "echo $HOME").stdout.strip()
    absp = lambda x: x.replace("~", home, 1) if x.startswith("~") else x  # noqa: E731
    SD, SV = absp(a.spark_dir), absp(a.spark_videos)
    t_end = time.time() + 3600 * a.max_hours
    while True:
        cand = candidates(runs, a.after, skip, baseline)
        new_parts = set()
        for (s, p), d in sorted(cand.items()):
            key = f"{s}/{p}"
            st = state.get(key, {})
            if st.get("run") != d.name:
                st = {"run": d.name}
            # where the run lives on the Spark
            if "spark_path" not in st:
                sp = f"{absp(a.spark_runs)}/{s}/{p}/{d.name}"
                if ssh(a.spark, f"test -f {sp}/summary.json", check=False).returncode != 0:
                    sp = f"{SD}/runs/{s}/{p}/{d.name}"
                    ssh(a.spark, f"mkdir -p {SD}/runs/{s}/{p}")
                    subprocess.run(["rsync", "-az", "--exclude", "ccx_work", "--exclude", "checkpoint.*", f"{d}/",
                                    f"{a.spark}:{sp}/"], check=True)
                st["spark_path"] = sp
            sp, name, lab = st["spark_path"], f"{s}_{p}", shlex.quote(LABELS.get(s, s))
            if not st.get("orig"):
                r = ssh(a.spark, f"cd {SD} && rm -rf work/orig/{name}_crush_frames && {ENV} {CAP} {PY} "
                                 f"repo/experiments/physical/render_crush.py {sp} --out work/orig/{name}_crush.mp4 "
                                 f"--label {lab} --workers 4 > logs/orig_{name}.log 2>&1 && "
                                 f"cp work/orig/{name}_crush.mp4 {SV}/", check=False)
                st["orig"] = r.returncode == 0
                new_parts.add(p)
            if not st.get("rtx"):
                r = ssh(a.spark, f"cd {SD} && ALLOW_CONTAINER=ridgeline-isaac ./make_crush_video.sh {sp} {name} {lab} "
                                 f"> logs/rtx_{name}.log 2>&1", check=False)
                st["rtx"] = r.returncode == 0
                if not st["rtx"]:
                    log(f"rtx {name}: exit {r.returncode}, retry next round")
            state[key] = st
            stf.write_text(json.dumps(state, indent=1))
            for kind in ("crush", "crush_rtx"):
                if st.get("orig" if kind == "crush" else "rtx") and not st.get(f"copied_{kind}"):
                    f = out / f"{name}_{kind}.mp4"
                    if subprocess.run(["scp", "-q", f"{a.spark}:{SV}/{f.name}", str(f)]).returncode == 0:
                        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(f), "-vf",
                                        "select='not(mod(n\\,48))',scale=640:-1,tile=3x4", "-frames:v", "1",
                                        str(out / "sheets" / f"{name}_{kind}.png")])
                        st[f"copied_{kind}"] = True
            stf.write_text(json.dumps(state, indent=1))
        for p in sorted(new_parts):
            done = [(s, state[f"{s}/{p}"]) for s in SYSTEMS if state.get(f"{s}/{p}", {}).get("orig")]
            if len(done) < 2:
                continue
            args = " ".join("--run " + shlex.quote(f"{s}:{st['spark_path']}:{LABELS.get(s, s)}")
                            for s, st in done)
            r = ssh(a.spark, f"cd {SD} && {ENV} {CAP} {PY} snap_compare.py --part {p} {args} --out work/snap "
                             f"--ffmpeg $HOME/micromamba/envs/feagent-ffmpeg/bin/ffmpeg > logs/snap_{p}.log 2>&1 && "
                             f"cp work/snap/{p}_snap_compare.mp4 {SV}/", check=False)
            if r.returncode == 0:
                f = out / f"{p}_snap_compare.mp4"
                subprocess.run(["scp", "-q", f"{a.spark}:{SV}/{f.name}", str(f)])
                subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(f), "-vf",
                                "select='not(mod(n\\,60))',scale=640:-1,tile=3x4", "-frames:v", "1",
                                str(out / "sheets" / f"{p}_snap_compare.png")])
                log(f"snap comparison {p}: {len(done)} systems")
        complete = sum(1 for s in SYSTEMS for p in PARTS if state.get(f"{s}/{p}", {}).get("rtx"))
        log(f"{complete}/{len(SYSTEMS) * len(PARTS)} system x part done")
        if a.once or complete == len(SYSTEMS) * len(PARTS) or time.time() > t_end:
            break
        time.sleep(a.every)


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    main()
