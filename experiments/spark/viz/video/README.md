# Omniverse RTX videos: wingbox twin + gold-parts gallery, and load ramps

`make_video.sh` renders `~/feagent-viz/twin.usdc` with Isaac Sim 6.2.0 headless (Docker, Omniverse RTX) on the
GB10. It then captions the frames with ffmpeg and writes **`~/feagent-viz/video/wingbox_twin.mp4`**: 1920x1080,
H.264 High, yuv420p, 30 fps, 24 s.

```
DRY_RUN=1 ~/feagent-viz/video/make_video.sh   # preflight verdict + every command; starts nothing
PREVIEW=1 ~/feagent-viz/video/make_video.sh   # first real run: 1 frame per shot + pt/rt timing -> preview_sheet.png
~/feagent-viz/video/make_video.sh             # full render + MP4
SKIP_RENDER=1 ~/feagent-viz/video/make_video.sh   # re-caption / re-encode from existing frames
STAGES=/abs/a.usdc,/abs/b.usdc ~/feagent-viz/video/make_video.sh   # + 3 s turntable per extra stage (gallery)
```

Other knobs:

- `RENDERER=auto|pt|rt`, `PT_BUDGET_S=4.0`
- `RESUME=1`
- `RENDER_TIMEOUT=7200`, `MEM_LIMIT=64g`
- `RENDER_EXTRA="--light-exposure 1 --deform-mode points"`, passed through to `render_twin.py`.

## Files

| File | What |
|---|---|
| `render_twin.py` | Isaac Sim standalone script: `SimulationApp` headless, opens `/twin/twin.usdc`, renders `frames/<shot>/frame_%05d.png` and `frames/manifest.json`. `--plan-only` runs with plain usd-core and needs no Isaac Sim or GPU. It is not the VTK `~/feagent-viz/render_twin.py`. |
| `make_video.sh` | Preflight, `docker run --rm --gpus all` headless, then assembly. |
| `assemble.py` | Host python3 (stdlib only): captions from `../summary.json`, one H.264 segment per shot, then concat. |
| `plan/plan.json` | Shot plan from `--plan-only`: frame counts, time codes, on-screen bbox of the model, and the singularity label pixel. |

## Shots (720 frames at 30 fps)

| Shot | Length | Field | Time code | Camera |
|---|---|---|---|---|
| a_orbit | 6 s | von_mises | 0 (undeformed) | 70° eased orbit about the stage's 3/4 view |
| b_deform | 5 s | von_mises | 0→24 (eased over 4.2 s, then held) | stage `/World/Camera`, held |
| c_root | 3 s | von_mises | 24 | static close-up of the root corner (600, 0, 150); red 3D leader + ffmpeg label "60.6 MPa peak: root-corner singularity, not a design value" |
| d_umag | 3 s | u_mag | 24 | slow drift |
| e_gates | 3 s | gate_status | 24 | drift continues |
| f_closing | 4 s | von_mises | 24 | pull-back ×1.45, then hold; closing title and numbers |
| g_NN | 3 s each | von_mises if present | end | 180° turntable per `--stages` entry |

How the shots are driven:

- **Colours:** set with the `field` variant on `/World`.
- **Deformation:** comes from the timeline. Time code / timeCodesPerSecond (24) is the timeline time.
- **Stage edits:** all go in the session layer, because the twin is mounted read-only. These are the shot
  camera `/Video/ShotCam`, the marker, the variant selection and the optional light exposure.

## Renderer: path-traced by default, with a time guard

`--renderer auto` (the default) renders the first 6 frames with **RTX path tracing** ("PathTracing"):

- 64 spp per frame (`totalSpp` 64), OptiX denoiser, 2 subframes.
- It keeps path tracing if that costs ≤ `PT_BUDGET_S` (4 s/frame, so ≤ ~50 min for 720 frames).

Otherwise it switches the whole run to **RTX Real-Time 2.0** ("RealTimePathTracing", the 6.2 default):

- DLSS Quality, 8 subframes per frame.

Every cut and field switch first resets DLSS history and renders 3 warm-up steps. RTX auto-exposure is on
(`/rtx/post/histogram/enabled`), as in `open_in_kit.py`. `manifest.json` records the renderer used and the probe
timings.

## Preflight (refuses with exit 3, prints why, starts nothing)

It refuses in any of these cases:

- An Isaac Sim container is running (name or image matches `isaac`, e.g. `ridgeline-isaac`).
- Any other container is running. Override with `ALLOW_OTHER_CONTAINERS=1`.
- `nvidia-smi` lists a compute process.
- `MemAvailable` < 40 GiB.
- A gold run is in progress:
  - a `*.lock`, `LOCK`, `RUNNING`, `*.inprogress` or `.running` file, or a live `*.pid`, in `~/gold-runs`
    (depth 3). `~/gold-runs` does not exist today and none of the gold scripts write a lock;
  - or a `gold-runs`, `gold-catalog`, `scripts/gold.py` or `goldgen` process (`pgrep -af`). This is what catches
    the current gold import.
- Less than 10 GB free disk.
- The image or ffmpeg is missing.

A lock file `.make_video.lock` (`flock`) stops two runs at once.

## Docker run

- Container `feagent-video`, `--rm --gpus all --memory 64g`, no X or DISPLAY, uid 1234.
- `-e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y -e OMNI_KIT_ACCEPT_EULA=YES`.
- Mounts:
  - `~/feagent-viz:/twin:ro`;
  - `~/feagent-viz/video/frames:/out:rw`;
  - caches `~/.cache/feagent-twin-isaac/*`, shared with `open.sh`;
  - extra stages at `/stages/N:ro`.
- Runs `./python.sh /twin/video/render_twin.py` with libgomp preloaded, wrapped in `timeout`, with a trap that runs
  `docker rm -f` if the script is interrupted.
- `render_twin.py` sets `umask 0`, so `dell` can delete the frames written by uid 1234.

## ffmpeg

- Not on the host (`command -v ffmpeg`: none).
- Not checkable inside the image:
  - its layers are root-only;
  - `docker image inspect` and `docker image history` show no ffmpeg;
  - the factory prior art states the 6.2 image has no ffmpeg.
- Installed instead: micromamba env **`feagent-ffmpeg`**, ffmpeg 9.0.2 with libx264, drawtext (freetype/fontconfig)
  and xfade. That is 102 MB downloaded and **510 MB on disk**. Remove it with
  `MAMBA_ROOT_PREFIX=~/micromamba ~/micromamba/bin/micromamba env remove -y -n feagent-ffmpeg`.
- It is a separate env, not `feagent`. The dry run into `feagent` would have downgraded icu (79.1 → 78.3) and
  changed cairo, libsqlite and libpsl, and the running gold benchmark uses that env.

## Time estimate

- **Measured on this GB10** with Isaac Sim 6.2, headless, 1920x1080, default RTX Real-Time:
  - 19–22 ms per rendered frame (`~/sculptured-examples/logs/factory4,5.log`, the other agent's factory scene, read
    only); 23–28 ms windowed;
  - app ready in about 20 s warm and about 175 s cold (shader compile).
- **RT mode, estimated:** (720 + 18) frames × 8 subframes × ~25 ms ≈ 0.2 s/frame. PNG writes overlap on 4 threads.
  So about **3 min of rendering**, plus 0.5–3 min of startup, plus the stage load (81.6 MB; time unknown).
- **Path tracing at 64 spp on GB10:** no measured number found, so **unknown**. The auto probe measures it.
  - The guard caps it at 4 s/frame (≤ ~50 min).
  - A guess of 0.5–2 s/frame would give 6–25 min.
- **Assembly:** 6 libx264 segments at preset slow plus a concat take about 1 min. The same assembly on 720
  synthetic frames took 8 s on the laptop CPU.

## Not verified until the first GPU run (run `PREVIEW=1` first)

1. That a paused timeline (`set_current_time` + `rep.orchestrator.step(pause_timeline=True)`) re-poses the
   time-sampled points. If shot b looks static, use `RENDER_EXTRA="--deform-mode points"`, which writes the
   interpolated points into the session layer.
2. Exposure of the stage's moderate UsdLux intensities (2.2 distant, 0.7 dome) under RTX with auto-exposure. If it
   is too dark, use `RENDER_EXTRA="--light-exposure 2"`.
3. Path-tracing cost on GB10, and whether auto-exposure adapts visibly across a cut.
4. Whether `rgb` `get_data()` is H×W×4 uint8 on the CPU here. It is the documented default.

## API sources (Isaac Sim 6.2, checked 2026-10-31)

- `SimulationApp` config keys (`renderer` = RealTimePathTracing | PathTracing | RaytracedLighting |
  MinimalRendering, `samples_per_pixel_per_frame`, `denoiser`, `anti_aliasing`, `disable_viewport_updates`):
  github.com/isaac-sim/IsaacSim `v6.2.0`,
  `source/extensions/isaacsim.simulation_app/isaacsim/simulation_app/simulation_app.py`.
- `isaacsim.core.experimental.utils.stage.open_stage` and `is_stage_loading`:
  `source/extensions/isaacsim.core.experimental.utils/python/impl/stage.py` (v6.2.0).
- Replicator calls:
  - `rep.create.render_product`, `rep.AnnotatorRegistry.get_annotator("rgb").attach`;
  - `rep.orchestrator.set_capture_on_play(False)`;
  - `rep.orchestrator.step(rt_subframes=, delta_time=0.0, pause_timeline=)`, `rep.orchestrator.wait_until_complete`;
  - `omni.replicator.core.functional.write_image`;
  - the settings `/rtx-transient/post/dlss/forceParamReset`, `/rtx/rendermode`, `/rtx/pathtracing/spp`,
    `/rtx/pathtracing/totalSpp` and `/rtx/pathtracing/optixDenoiser/enabled`.

  Sources: https://docs.isaacsim.omniverse.nvidia.com/6.2.0/replicator_tutorials/tutorial_replicator_isaac_snippets.html
  and `source/standalone_examples/api/isaacsim.replicator.examples/{simulation_get_data,motion_blur_pathtracing}.py`
  (v6.2.0).
- `rt_subframes` ≥ 2 and ghosting: https://docs.isaacsim.omniverse.nvidia.com/6.2.0/replicator_tutorials/troubleshooting.html
- The container launch follows `~/feagent-viz/open.sh`, which follows the 6.2 container guide
  (https://docs.isaacsim.omniverse.nvidia.com/6.2.0/installation/install_container.html).

## Cleanup

```
rm -rf ~/feagent-viz/video/{frames,frames_preview,work,logs,plan}
```

To also remove ffmpeg, delete the `feagent-ffmpeg` env (command above). Nothing else is created except the shared
Isaac Sim caches in `~/.cache/feagent-twin-isaac`.

## Added 2026-10-31 (user-approved run, GPU shared with ridgeline-isaac)

- `ALLOW_CONTAINER=ridgeline-isaac` tolerates exactly that container and its GPU processes (found with `docker top`).
  The memory (≥ 40 GiB), disk and gold-run checks stay. The gold check now waits for `gold_case.py`, `run_all.py`
  and `gold.py` to finish: it polls every 30 s for up to `GOLD_WAIT_S` = 900 s.
- Gallery: run `~/feagent-viz/gallery/prep_gallery.sh` first.
  - It re-solves each pilot set with feacore + cuDSS into `gallery/<set>/r.vtu`, under a 40G scope and a 12 GiB
    watchdog (`gallery/wd.sh`, which logs to `gallery/watchdog.log`).
  - `to_twin.py` converts m → mm and Pa → MPa and writes `gallery/<set>/<set>.usdc` with `export_twin.py`.
    Each stage stays in its own folder because its legend textures are relative paths.
  - Then run `STAGES=<4 usdc> ALLOW_CONTAINER=ridgeline-isaac make_video.sh`. A 3 s title card goes before the
    turntables, and the legend quad is hidden while the part spins.
- Load ramps: run `~/feagent-viz/ramps/prep_ramps.sh`, then
  `RAMPS=1 RAMP_SETS="set ..." ALLOW_CONTAINER=ridgeline-isaac make_video.sh`. That writes
  `~/feagent-viz/ramps/load_ramps.mp4` and `ramps/stills/<set>_full_load.png`.
  - `build_ramp.py`: λ runs 0→1 over time codes 0..20, then holds to 24.
    - points = X + λ·scale·u.
    - displayColor = turbo(λ·σ / p99.5 at full load), on a fixed range.
    - Top-3 hotspot clusters of surface nodes; spikes more than 3× the median of their neighbours are excluded and
      marked violet as "singular corner". Markers turn amber at 50 % and red at 80 % of the colour max.
    - 3/4 camera on the octant that faces the hotspots.
  - `render_twin.py --ramps` renders these with `--deform-mode bake` (every time-sampled attribute is written into the
    session layer per frame). `ramp_assemble.py` adds the colour bar, a live load-factor counter and the captions.
