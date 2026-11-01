# 06: Omniverse RTX video of the wingbox twin: prepared, NOT run

- **Where:** Spark `~/feagent-viz/video/`: `render_twin.py`, `make_video.sh`, `assemble.py`, `README.md`,
  `plan/plan.json`.
- **Local copies:** in this folder's `video/`, plus the synthetic-frame assemble test in `../asmtest/`.

## Constraints kept

- No container and no Isaac Sim process were started.
- Docker was used read-only (`ps`, `image inspect`, `image history`).
- No sudo. Nothing in `ridgeline-isaac`, `~/gold-runs`, `~/gold-catalog`, `~/runs` or `~/feagent-mh` was
  touched. Two other things were only read:
  - `pgrep` reads the process list;
  - the `~/sculptured-examples` Isaac logs were read for timings.
- No laptop GPU was used. The laptop CPU ran ffmpeg for the caption test.
- No subagents, no commit.

## What it does

### Shots

These six shots run 24 s in total, 720 frames at 30 fps:

| Shot | Length | Content |
|---|---|---|
| a | 6 s | orbit, von_mises, t=0 |
| b | 5 s | stage 3/4 camera, t 0→24 |
| c | 3 s | static root close-up; red 3D leader and an ffmpeg label at the projected pixel (963, 254) |
| d | 3 s | u_mag |
| e | 3 s | gate_status |
| f | 4 s | pull-back, then the closing title and numbers |

`--stages a.usdc,b.usdc` appends a 3 s, 180° turntable per stage (gallery hook).

### Renderer

**Path tracing by default** (64 spp, OptiX denoiser), with an automatic fallback to **RTX Real-Time 2.0**
(RealTimePathTracing, DLSS Quality, 8 subframes):

- The first 6 frames are timed.
- It falls back if path tracing costs more than 4 s/frame.

The renderer actually used is written to `manifest.json`.

### Capture

- APIs: Replicator `render_product` + `rgb` annotator + `rep.orchestrator.step(rt_subframes, delta_time=0,
  pause_timeline=True)`, with PNGs written by PIL on 4 threads.
- Colours: the `field` variant, set in the session layer.
- Deformation: timeline time = time code / 24. Fallback: `--deform-mode points`.

### Docker command

- Headless; no X.
- EULA and privacy env.
- `~/feagent-viz:/twin:ro`, `video/frames:/out:rw`, caches shared with open.sh, uid 1234, `--memory 64g`.
- Wrapped in `timeout` and a `docker rm -f` trap.

### Captions

All caption numbers come from `summary.json`. `drawtext` uses `textfile=` and `expansion=none`, so nothing needs
escaping. The test showed:

- the encode is correct (h264 High, yuv420p, 30/1, 720 frames, 24.000 s);
- the label is aligned with the projected anchor.

## ffmpeg

- It is on neither the host nor (as far as can be checked) the image:
  - layers root-only;
  - inspect and history show none;
  - the prior art says none.
- Installed **micromamba env `feagent-ffmpeg`**: ffmpeg 9.0.2 with libx264 and drawtext. That is 102 MB download,
  **510 MB on disk**.
- It is not in `feagent`. Installing there would have downgraded icu 79.1→78.3 and changed cairo, libsqlite and
  libpsl in the env the running gold benchmark uses.

## Time estimate

### Measured on this GB10 by the other agent

Isaac Sim 6.2, headless, 1920x1080, RTX Real-Time:

- 19–22 ms per rendered frame (`~/sculptured-examples/logs/factory4,5.log`);
- app ready in 20 s warm and 175 s cold.

### Estimates for this video

| Phase | Estimate |
|---|---|
| RT render | ~0.2 s/frame, about 3 min for 720 frames |
| Startup and stage load | +0.5–3 min |
| Path tracing at 64 spp | **unknown**, no GB10 number found; guarded at ≤ 4 s/frame (≤ ~50 min); a guess of 6–25 min |
| Assembly | ~1 min |

## First real run

1. Check that `docker ps` is empty and the gold import has finished.
2. Run `PREVIEW=1 ~/feagent-viz/video/make_video.sh`. It renders one frame per shot plus a pt/rt timing probe,
   then writes `preview_sheet.png`.
3. Check the exposure and that shot b deforms.
4. Run the full `~/feagent-viz/video/make_video.sh`.

## DRY_RUN=1 output (2026-10-31 13:38, Spark)

The preflight correctly REFUSES right now: `ridgeline-isaac` is running and holds the GPU, and a
gold-catalog import is in progress.

```
make_video: REFUSE: an Isaac Sim container is running (the GB10 runs one large GPU job at a time):
      ridgeline-isaac nvcr.io/nvidia/isaac-sim:6.2.0
make_video: REFUSE: processes are using the GPU:
      1207360, /isaac-sim/kit/python/bin/python3, 6696 MiB
make_video: ok: MemAvailable 104 GiB >= 40 GiB
make_video: REFUSE: a gold run / import is in progress:
      1198841 bash -c until grep -q DONE ~/gold-catalog/pilot/pilot.tsv; do sleep 5; done; cd ~/gold-catalog/pilot; cat pilot.tsv; for p in abc_*; do echo "== $p"; py
      1199192 ~/micromamba/envs/feagent/bin/python -m feagent.cli automesh ~/gold-catalog/pilot/abc_00000223/part.step -o ~/gold-catalog/pi
      1199228 ~/micromamba/envs/feagent/bin/python -m feagent.mesher ~/gold-catalog/pilot/abc_00000223/auto/mesh.yaml --scale 1.0 --json
make_video: preflight would REFUSE now; DRY_RUN continues printing the commands
make_video: plan (CPU, read-only stage read with usd-core):
+ ~/micromamba/envs/feagent/bin/python -W ignore ~/feagent-viz/video/render_twin.py --plan-only --usd ~/feagent-viz/twin.usdc --out ~/feagent-viz/video/plan
[plan] a_orbit     180 frames  field=von_mises  tc 0.0->0.0  model on screen [0.413, 0.245, 0.623, 0.814] -> [0.238, 0.29, 0.753, 0.711]
[plan] b_deform    150 frames  field=von_mises  tc 0.0->24.0  model on screen [0.292, 0.256, 0.653, 0.791] -> [0.292, 0.256, 0.653, 0.791]
[plan] c_root       90 frames  field=von_mises  tc 24.0->24.0  model on screen [-0.088, -0.243, 1.201, 0.995] -> [-0.088, -0.243, 1.201, 0.995]  label px [963, 254] (peak px [963, 602])
[plan] d_umag       90 frames  field=u_mag  tc 24.0->24.0  model on screen [0.292, 0.256, 0.653, 0.791] -> [0.277, 0.26, 0.671, 0.782]
[plan] e_gates      90 frames  field=gate_status  tc 24.0->24.0  model on screen [0.277, 0.26, 0.671, 0.782] -> [0.266, 0.265, 0.688, 0.77]
[plan] f_closing   120 frames  field=von_mises  tc 24.0->24.0  model on screen [0.292, 0.256, 0.653, 0.791] -> [0.382, 0.353, 0.591, 0.723]
[plan] 720 frames = 24.0 s at 30 fps -> ~/feagent-viz/video/plan/plan.json
+ mkdir -p ~/.cache/feagent-twin-isaac/main ~/.cache/feagent-twin-isaac/computecache ~/.cache/feagent-twin-isaac/kit ~/.cache/feagent-twin-isaac/logs ~/.cache/feagent-twin-isaac/config ~/.cache/feagent-twin-isaac/data ~/.cache/feagent-twin-isaac/pkg ~/.cache/feagent-twin-isaac/hub ~/feagent-viz/video/frames
+ chmod -R a+rwX ~/.cache/feagent-twin-isaac ~/feagent-viz/video/frames
+ find ~/feagent-viz/video/frames -mindepth 1 -delete
make_video: render: Isaac Sim 6.2 headless (renderer auto), frames -> ~/feagent-viz/video/frames; first start compiles shaders (~3 min)
+ timeout --signal=TERM --kill-after=60 7200 docker run --name feagent-video --rm --gpus all --entrypoint bash --memory 64g -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y -e OMNI_KIT_ACCEPT_EULA=YES -v ~/feagent-viz:/twin:ro -v ~/feagent-viz/video/frames:/out:rw -v ~/.cache/feagent-twin-isaac/main:/isaac-sim/.cache:rw -v ~/.cache/feagent-twin-isaac/computecache:/isaac-sim/.nv/ComputeCache:rw -v ~/.cache/feagent-twin-isaac/kit:/isaac-sim/kit/cache:rw -v ~/.cache/feagent-twin-isaac/logs:/isaac-sim/.nvidia-omniverse/logs:rw -v ~/.cache/feagent-twin-isaac/config:/isaac-sim/.nvidia-omniverse/config:rw -v ~/.cache/feagent-twin-isaac/data:/isaac-sim/.local/share/ov/data:rw -v ~/.cache/feagent-twin-isaac/pkg:/isaac-sim/.local/share/ov/pkg:rw -v ~/.cache/feagent-twin-isaac/hub:/var/cache/hub:rw -u 1234:1234 nvcr.io/nvidia/isaac-sim:6.2.0 -c export\ LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1\;\ exec\ ./python.sh\ /twin/video/render_twin.py\ \"\$@\" bash --usd /twin/twin.usdc --summary /twin/summary.json --out /out --renderer auto --pt-budget-s 4.0
    (stdout/stderr tee'd to ~/feagent-viz/video/logs/render-20261031-133838.log)
make_video: assemble (commands only; the captions below are written to ~/feagent-viz/video/work at run time):
# caption a_orbit.0.title.txt: 'FEAgent virtual twin — wingbox stage 1'
# caption a_orbit.1.sub.txt: 'LC1: 20 kN tip lift · 1,150,527 DOFs · verdict PASS'
# caption a_orbit.2.cap.txt: 'von Mises stress · colour max 26.9 MPa (p99.5)'
# caption a_orbit.3.credit.txt: 'rendered with NVIDIA Omniverse RTX (Isaac Sim 6.2) on DGX Spark GB10'
# caption b_deform.0.cap.txt: 'Deformation ×20 · tip deflection 7.683 mm (extrapolated 7.687)'
# caption b_deform.1.cap2.txt: 'beam theory 7.498 mm (+2.5 %) · GCI 0.054 % · equilibrium 1e-11'
# caption c_root.0.label.txt: '60.6 MPa peak: root-corner singularity\nnot a design value (grows with refinement)'
# caption c_root.1.cap.txt: 'Clamped root corner · the 60.6 MPa peak is a stress singularity'
# caption c_root.2.cap2.txt: 'converged field away from the root ≈ 24.2 MPa · colour max 26.9 MPa'
# caption d_umag.0.cap.txt: 'Displacement magnitude |u| · max 7.73 mm · shown ×20'
# caption e_gates.0.cap.txt: 'Mesh-quality gates · 10 warn, 0 fail, 0 pass of 10 parts'
# caption e_gates.1.cap2.txt: 'warn = thin_walls on the solved mesh (~1 element through the wall, gate needs 2); every part spec passes'
# caption f_closing.0.bigtitle.txt: 'FEAgent virtual twin — wingbox stage 1'
# caption f_closing.1.numbers.txt: 'Verdict PASS · 1,150,527 DOFs · cuDSS on NVIDIA GB10 (DGX Spark) 24.2 s\ntip 7.683 mm (extrap. 7.687) vs beam theory 7.498 mm (+2.5 %) · GCI 0.054 % · equilibrium 1e-11\nvon Mises colour max 26.9 MPa (p99.5); 60.6 MPa root-corner peak is a singularity'
# caption f_closing.2.credit_c.txt: 'rendered with NVIDIA Omniverse RTX (Isaac Sim 6.2) on DGX Spark GB10'
+ ~/micromamba/envs/feagent-ffmpeg/bin/ffmpeg -hide_banner -v error -y -framerate 30 -start_number 0 -i ~/feagent-viz/video/frames/a_orbit/frame_%05d.png -frames:v 180 -vf 'format=rgb24,drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf:textfile=~/feagent-viz/video/work/a_orbit.0.title.txt:expansion=none:x=40:y=40:fontsize=46:fontcolor=white:line_spacing=10:alpha='"'"'if(lt(t,0.4),0,if(lt(t,0.8),(t-0.4)/0.4,if(lt(t,5.6),1,if(lt(t,6.0),(6.0-t)/0.4,0))))'"'"':box=1:boxcolor=0x0b1416@0.72:boxborderw=14,drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf:textfile=~/feagent-viz/video/work/a_orbit.1.sub.txt:expansion=none:x=40:y=112:fontsize=28:fontcolor=white:line_spacing=10:alpha='"'"'if(lt(t,0.6),0,if(lt(t,1.0),(t-0.6)/0.4,if(lt(t,5.6),1,if(lt(t,6.0),(6.0-t)/0.4,0))))'"'"':box=1:boxcolor=0x0b1416@0.72:boxborderw=14,drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf:textfile=~/feagent-viz/video/work/a_orbit.2.cap.txt:expansion=none:x=40:y=h-150:fontsize=32:fontcolor=white:line_spacing=10:alpha='"'"'if(lt(t,0.25),0,if(lt(t,0.65),(t-0.25)/0.4,if(lt(t,5.75),1,if(lt(t,6.15),(6.15-t)/0.4,0))))'"'"':box=1:boxcolor=0x0b1416@0.72:boxborderw=14,drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf:textfile=~/feagent-viz/video/work/a_orbit.3.credit.txt:expansion=none:x=w-tw-36:y=h-th-28:fontsize=20:fontcolor=white:line_spacing=10:alpha='"'"'if(lt(t,0.25),0,if(lt(t,0.65),(t-0.25)/0.4,if(lt(t,5.75),1,if(lt(t,6.15),(6.15-t)/0.4,0))))'"'"':shadowcolor=0x000000@0.8:shadowx=2:shadowy=2,fade=t=in:st=0:d=0.25,fade=t=out:st=5.750:d=0.25,format=yuv420p' -c:v libx264 -preset slow -crf 16 -pix_fmt yuv420p -r 30 -an ~/feagent-viz/video/work/00_a_orbit.mp4
+ ~/micromamba/envs/feagent-ffmpeg/bin/ffmpeg -hide_banner -v error -y -framerate 30 -start_number 0 -i ~/feagent-viz/video/frames/b_deform/frame_%05d.png -frames:v 150 -vf 'format=rgb24,drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf:textfile=~/feagent-viz/video/work/b_deform.0.cap.txt:expansion=none:x=40:y=h-150:fontsize=32:fontcolor=white:line_spacing=10:alpha='"'"'if(lt(t,0.25),0,if(lt(t,0.65),(t-0.25)/0.4,if(lt(t,4.75),1,if(lt(t,5.15),(5.15-t)/0.4,0))))'"'"':box=1:boxcolor=0x0b1416@0.72:boxborderw=14,drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf:textfile=~/feagent-viz/video/work/b_deform.1.cap2.txt:expansion=none:x=40:y=h-96:fontsize=26:fontcolor=white:line_spacing=10:alpha='"'"'if(lt(t,0.25),0,if(lt(t,0.65),(t-0.25)/0.4,if(lt(t,4.75),1,if(lt(t,5.15),(5.15-t)/0.4,0))))'"'"':box=1:boxcolor=0x0b1416@0.72:boxborderw=14,fade=t=in:st=0:d=0.25,fade=t=out:st=4.750:d=0.25,format=yuv420p' -c:v libx264 -preset slow -crf 16 -pix_fmt yuv420p -r 30 -an ~/feagent-viz/video/work/01_b_deform.mp4
+ ~/micromamba/envs/feagent-ffmpeg/bin/ffmpeg -hide_banner -v error -y -framerate 30 -start_number 0 -i ~/feagent-viz/video/frames/c_root/frame_%05d.png -frames:v 90 -vf 'format=rgb24,drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf:textfile=~/feagent-viz/video/work/c_root.0.label.txt:expansion=none:x=981:y=214:fontsize=28:fontcolor=white:line_spacing=10:alpha='"'"'if(lt(t,0.25),0,if(lt(t,0.65),(t-0.25)/0.4,if(lt(t,2.75),1,if(lt(t,3.15),(3.15-t)/0.4,0))))'"'"':box=1:boxcolor=0x0b1416@0.72:boxborderw=14,drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf:textfile=~/feagent-viz/video/work/c_root.1.cap.txt:expansion=none:x=40:y=h-150:fontsize=32:fontcolor=white:line_spacing=10:alpha='"'"'if(lt(t,0.25),0,if(lt(t,0.65),(t-0.25)/0.4,if(lt(t,2.75),1,if(lt(t,3.15),(3.15-t)/0.4,0))))'"'"':box=1:boxcolor=0x0b1416@0.72:boxborderw=14,drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf:textfile=~/feagent-viz/video/work/c_root.2.cap2.txt:expansion=none:x=40:y=h-96:fontsize=26:fontcolor=white:line_spacing=10:alpha='"'"'if(lt(t,0.25),0,if(lt(t,0.65),(t-0.25)/0.4,if(lt(t,2.75),1,if(lt(t,3.15),(3.15-t)/0.4,0))))'"'"':box=1:boxcolor=0x0b1416@0.72:boxborderw=14,fade=t=in:st=0:d=0.25,fade=t=out:st=2.750:d=0.25,format=yuv420p' -c:v libx264 -preset slow -crf 16 -pix_fmt yuv420p -r 30 -an ~/feagent-viz/video/work/02_c_root.mp4
+ ~/micromamba/envs/feagent-ffmpeg/bin/ffmpeg -hide_banner -v error -y -framerate 30 -start_number 0 -i ~/feagent-viz/video/frames/d_umag/frame_%05d.png -frames:v 90 -vf 'format=rgb24,drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf:textfile=~/feagent-viz/video/work/d_umag.0.cap.txt:expansion=none:x=40:y=h-150:fontsize=32:fontcolor=white:line_spacing=10:alpha='"'"'if(lt(t,0.25),0,if(lt(t,0.65),(t-0.25)/0.4,if(lt(t,2.75),1,if(lt(t,3.15),(3.15-t)/0.4,0))))'"'"':box=1:boxcolor=0x0b1416@0.72:boxborderw=14,fade=t=in:st=0:d=0.25,fade=t=out:st=2.750:d=0.25,format=yuv420p' -c:v libx264 -preset slow -crf 16 -pix_fmt yuv420p -r 30 -an ~/feagent-viz/video/work/03_d_umag.mp4
+ ~/micromamba/envs/feagent-ffmpeg/bin/ffmpeg -hide_banner -v error -y -framerate 30 -start_number 0 -i ~/feagent-viz/video/frames/e_gates/frame_%05d.png -frames:v 90 -vf 'format=rgb24,drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf:textfile=~/feagent-viz/video/work/e_gates.0.cap.txt:expansion=none:x=40:y=h-150:fontsize=32:fontcolor=white:line_spacing=10:alpha='"'"'if(lt(t,0.25),0,if(lt(t,0.65),(t-0.25)/0.4,if(lt(t,2.75),1,if(lt(t,3.15),(3.15-t)/0.4,0))))'"'"':box=1:boxcolor=0x0b1416@0.72:boxborderw=14,drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf:textfile=~/feagent-viz/video/work/e_gates.1.cap2.txt:expansion=none:x=40:y=h-96:fontsize=26:fontcolor=white:line_spacing=10:alpha='"'"'if(lt(t,0.25),0,if(lt(t,0.65),(t-0.25)/0.4,if(lt(t,2.75),1,if(lt(t,3.15),(3.15-t)/0.4,0))))'"'"':box=1:boxcolor=0x0b1416@0.72:boxborderw=14,fade=t=in:st=0:d=0.25,fade=t=out:st=2.750:d=0.25,format=yuv420p' -c:v libx264 -preset slow -crf 16 -pix_fmt yuv420p -r 30 -an ~/feagent-viz/video/work/04_e_gates.mp4
+ ~/micromamba/envs/feagent-ffmpeg/bin/ffmpeg -hide_banner -v error -y -framerate 30 -start_number 0 -i ~/feagent-viz/video/frames/f_closing/frame_%05d.png -frames:v 120 -vf 'format=rgb24,drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf:textfile=~/feagent-viz/video/work/f_closing.0.bigtitle.txt:expansion=none:x=(w-tw)/2:y=h*0.11:fontsize=60:fontcolor=white:line_spacing=10:alpha='"'"'if(lt(t,0.5),0,if(lt(t,0.9),(t-0.5)/0.4,if(lt(t,3.75),1,if(lt(t,4.15),(4.15-t)/0.4,0))))'"'"':shadowcolor=0x000000@0.8:shadowx=2:shadowy=2,drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf:textfile=~/feagent-viz/video/work/f_closing.1.numbers.txt:expansion=none:x=(w-tw)/2:y=h*0.76:fontsize=26:fontcolor=white:line_spacing=10:alpha='"'"'if(lt(t,0.9),0,if(lt(t,1.3),(t-0.9)/0.4,if(lt(t,3.75),1,if(lt(t,4.15),(4.15-t)/0.4,0))))'"'"':box=1:boxcolor=0x0b1416@0.72:boxborderw=14,drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf:textfile=~/feagent-viz/video/work/f_closing.2.credit_c.txt:expansion=none:x=(w-tw)/2:y=h-th-40:fontsize=24:fontcolor=white:line_spacing=10:alpha='"'"'if(lt(t,1.2),0,if(lt(t,1.6),(t-1.2)/0.4,if(lt(t,3.75),1,if(lt(t,4.15),(4.15-t)/0.4,0))))'"'"':shadowcolor=0x000000@0.8:shadowx=2:shadowy=2,fade=t=in:st=0:d=0.25,fade=t=out:st=3.750:d=0.25,format=yuv420p' -c:v libx264 -preset slow -crf 16 -pix_fmt yuv420p -r 30 -an ~/feagent-viz/video/work/05_f_closing.mp4
+ write ~/feagent-viz/video/work/segments.txt: 00_a_orbit.mp4 01_b_deform.mp4 02_c_root.mp4 03_d_umag.mp4 04_e_gates.mp4 05_f_closing.mp4
+ ~/micromamba/envs/feagent-ffmpeg/bin/ffmpeg -hide_banner -v error -y -f concat -safe 0 -i ~/feagent-viz/video/work/segments.txt -c copy -movflags +faststart ~/feagent-viz/video/wingbox_twin.mp4
make_video: DRY_RUN: nothing was started.
```
