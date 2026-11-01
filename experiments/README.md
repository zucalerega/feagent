# Experiments: rerun guide

Everything here is portable: scripts resolve paths relative to themselves, or take args/env vars. Run all
commands from the repo root of a fresh clone after `uv sync` (see ../CLAUDE.md). On a laptop prefix any local
Python with `CUDA_VISIBLE_DEVICES=""` (the scripts already export it). The Spark host is written `<spark-host>`
(a DGX Spark, user `dell`, reached over SSH/Tailscale); the `spark/` scripts are copied as they ran there and use
`~/...` paths of the Spark account (micromamba env `feagent`, see `spark/SETUP.md`).

| dir | what | needs |
|---|---|---|
| `material/` | calibrate Alpha Fast Resin from the user's dogbone test (`dogbone1.pdf`, digitized into the script) | CPU, seconds |
| `physical/` | platen compression and crush-to-snap of gold parts abc_00000003 and abc_00000065 | CPU; compression about 2-10 min per part |
| `wingbox/` | stage-1 10-part wingbox (7075-T6, 20 kN tip lift): geometry, meshes, solve, beam theory + GCI verification | smallest case CPU; all four meshes (up to 1.15 M DOFs) on the Spark (cuDSS) |
| `spark/` | DGX Spark setup, cuDSS benchmark, gold-benchmark harness runner, USD digital twin export/render/video | **Spark / GPU**; twin and video need the Isaac Sim 6.2 container |
| `notes/` | GPU and whole-aircraft study notes (01-07), Spark notes (spark 01-06 files), laptop `bench_gpu.py` | reading |

Prerequisites: `uv sync --extra pardiso` (the CPU runs use PARDISO via pypardiso; without the extra the solve falls back to a far slower solver, >15 min vs ~30 s for abc_00000003) (feacore, feabench, meshing-harness/feagent; gmsh comes with it on x86 Linux). For the
CalculiX path: `micromamba create -n ccx -c conda-forge calculix` and `export FEACORE_CCX=$(micromamba run -n ccx which ccx)`.
Isaac Sim 6.2 (container `nvcr.io/nvidia/isaac-sim:6.2.0`, aarch64 on the Spark) only for the twin/video.

## 1. Material calibration (CPU, about 2 s)

    uv run python experiments/material/calibrate_dogbone.py

Rewrites `alpha_fast_dogbone1.json` and `alpha_fast_dogbone1_fit.png` next to the script. The curve is digitized
from `dogbone1.pdf` inside the script (+-15 N, +-0.05 mm); ASTM D638 Type I geometry is assumed, nu 0.35 assumed.
Expected: E 1879.8 MPa (apparent, crosshead strain), UTS 50.48 MPa, failure PEEQ 0.043, true failure strain 0.0715,
15 plastic-table rows.

## 2. Physical tests of gold parts (`physical/`)

Three models, deliberately of different fidelity:
- **Compression** (`run_comp.py`): linear elastic, feacore skfem backend (tet10, PARDISO). Bottom face fixed, load
  on the top face.
- **Crush-to-snap** (`snap.py`): a simplified small-strain secant plasticity model with element erosion
  (displacement-controlled, bonded platens), driven by the calibrated material JSON in `../material/`
  (override with `MATERIAL_JSON`). Treat it as a cross-check, not a validated model.
- **CalculiX nonlinear** (J2 elastoplastic, load-stepped; feacore nonlinear ccx backend, `feacore/solver/_ccx_nl.py`
  and `ccx_backend.py` on this branch, enabled with `analysis.options.nonlinear`). `docs/solver_ccx_nonlinear.md`
  is referenced by the code but is not on this branch's base commit; see `docs/solver_ccx.md`. `render_nl.py`
  renders its `force_displacement.csv` run dir.

Run compression (copies the two STEPs from `benchmarks/gold/`, runs `feagent automesh`, adds groups/BCs, solves):

    experiments/physical/prepare.sh                                  # WORKDIR=experiments/physical/work, both parts, E 2.23 GPa
    COMP_E_PA=10e9 experiments/physical/prepare.sh /some/dir abc_00000003    # the E 10 GPa check below

Env: `COMP_E_PA` (Pa, default 2.23e9 = Alpha Fast, default post-cure TDS), `COMP_FORCE_N` (default 100).
Output `WORK/<part>/summary.json` (gates, QoIs), `result.vtu`. Render (software GL):

    LIBGL_ALWAYS_SOFTWARE=1 GALLIUM_DRIVER=llvmpipe __GLX_VENDOR_LIBRARY_NAME=mesa uv run python \
        experiments/physical/render_comp.py WORK/abc_00000003 z F_FULL_N MAG out.mp4 "title"   # load-ramp video; MAG = deformation exaggeration

Crush-to-snap: `uv run python experiments/physical/snap.py MESH.msh AXIS OUT.npz` (MESH = `WORK/<part>/out/mesh.msh`),
then `render_snap.py OUT.npz OUT.mp4 TITLE [NOTE]` (same software-GL env). Axes: abc_00000003 z, abc_00000065 y.

Expected (abc_00000003, 100 N, mesh as produced by automesh, 13.9 MB .msh, all gates pass, thin_walls warn):
- peak von Mises **2.05 MPa** (reproduced: 2.047 MPa); top-face mean displacement **3.46 um with E = 10 GPa** (reproduced: 3.463 um; solve 29 s, whole script 87 s) (displacement scales as 1/E:
  about 15.5 um at the default Alpha Fast E = 2.23 GPa; stress is E-independent). Reaction 100.000 N.

### Multi-system mesh study (`mesh_systems/` + `physical/run_all.py`)

Six mesh systems (`mesh_systems/CONTRACT.md`; where their outputs live: `mesh_systems/layout.json`) mesh both parts.
One command runs the experiments on all of them (CPU, <= 4 solver threads, <= 4 render processes):

    python experiments/physical/run_all.py --meshsys MESHSYS_OUT --study STUDY_OUT [--build] [--crush] \
        [--crush-runs runs/nonlinear] [--systems a,b] [--parts abc_00000003]

- `--build` runs each system's `build.py` for mesh dirs without `report.json`; the canonical index
  `MESHSYS_OUT/_canon/<system>/<part>/<linear|nonlinear>` is always (re)linked (`physical/meshsys.py`).
- `linear_ramp.py`: one 100 N skfem/PARDISO solve per system x part (`STUDY_OUT/linear/<system>/<part>/`):
  compliance 2U/F, top travel, peak and 99.9th-percentile volume-weighted von Mises.
- `STUDY_OUT/f_full.json`: per part, the force that brings the 99.9th-percentile von Mises to 87 % of the UTS (median
  over systems); the same F_full drives every system's video.
- `render_linear.py` -> `STUDY_OUT/videos/<system>_<part>_linear.mp4` (10 s ramp + 3 s hold, 1920x1080) and
  `all_systems_<part>_linear.mp4` (3 x 2 grid, ffmpeg xstack).
- `--crush` launches `crush.py` per system x part (hours each, one after the other); `render_crush.py` renders the
  newest run per system x part (follows `continued_from`; tet10 and hex vtus) -> `<system>_<part>_crush.mp4`.
  - Broken cells are red in all three panels and are also drawn as an x-ray overlay. Otherwise abc_00000003's crack,
    which runs round the bottom ring inside the wall, stays hidden behind intact cells and the platen. Its camera
    also sits lower (`PART_VIEW`).
  - In the held ending of a separated run, the intact piece on the top platen lifts 6 % of the height. The caption
    says it is not simulated.
  - `--only i,j` renders a few frames to look at first.
- `snap_compare.py --part P --run SYSTEM:RUN_DIR[:LABEL] ...` -> `<part>_snap_compare.mp4`: one tile per system
  (`render_crush.py --layout single`), all on the same platen travel in every frame. The clock slows down from each
  run's peak force to its end, so each snap plays in slow motion. Rerun it as more systems finish.
- `compare.py` -> `STUDY_OUT/comparison.{md,json,png}`; rerun it as crush runs finish.
- Omniverse version (`experiments/omniverse/`, on the DGX Spark with the Isaac Sim 6.2 container):
  `make_crush_video.sh RUN_DIR NAME "label"` runs three steps.
  - `build_crush_usd.py` writes a USD time series: the intact surface with von Mises colours, broken cells red,
    fragments grey, brushed-metal platens, a studio floor and lights, and a frame plan with an orbiting camera.
  - `render_crush_rtx.py` renders it headless with RTX Real-Time, at about 0.7 s per frame.
  - `assemble_crush.py` adds the caption bar, the force-travel inset and the colour key, giving
    `<name>_crush_rtx.mp4`.
  - Check frames first with `ONLY=0,250,-1`.
  - `watch_crush_videos.py --spark <user@spark-host> --runs runs/nonlinear --out DIR` runs on the laptop and polls
    every 10 min. For each newly separated run it renders both videos on the Spark, refreshes the snap comparison of
    that part and copies the videos and contact sheets back to DIR.

Expected (2026-10-31, six systems): F_full 4558 N (abc_00000003) and 2394 N (abc_00000065); spread across systems:
compliance 1.3 % / 0.8 %, robust von Mises 15.5 % / 4.8 %, peak von Mises 88 % / 42 %.

## 3. Wingbox (`wingbox/`)

    experiments/wingbox/run_cpu.sh assembly_c2      # geometry -> mesh -> case -> PARDISO solve; case outputs in <case>/cpu/

Geometry only: `uv run python experiments/wingbox/geometry/build_wingbox.py`. Details, mesh tables, gates: `wingbox/README.md`,
`wingbox/verify/verdict.md`. The full verification (`SOLVE_DIR=cpu|gpu uv run python experiments/wingbox/verify/verify_all.py`)
needs all four meshes solved; the 1.15 M DOF xf case wants the Spark (`verify/run_case.sh <case_dir>`, cuDSS, 24 s).
Expected: tip deflection c2/c1/fine/xf = 7.671 / 7.676 / 7.679 / **7.683 mm**, extrapolated 7.687; beam theory
**7.498 mm**; GCI(fine) **0.054 %**; equilibrium 1e-11; DOFs 150,213 / 237,273 / 370,170 / 1,150,527; verdict pass.
CPU sanity (c2, SuperLU): 19.5 s, R_z = -20000.000 N, tip 7.671 mm.

## 4. Spark / GPU (`spark/`)

- `SETUP.md`: micromamba env, MemoryMax scope, versions.
- `bench/`: cuDSS vs scipy scaling (`run_one.sh h method`, needs a box mesh cache; the Spark environment). Expected
  total solve(): 14.8k 0.53 s; 37.8k 1.01 s (scipy 3.98); 128.9k 3.35 s (scipy 196); 199k 6.46 s (scipy 596);
  479.5k 18.6 s; 928.9k 51.7 s (peak RSS 5.4 GB + ~20 GB GPU/unified). Full table: `bench/RESULTS.md`.
- `gold-catalog/`: gold-benchmark harness runner (`scripts/gold_case.py mesh|case|solve`, `run_all.py`, `pilot.sh`,
  `wd.sh` memory watchdog). Expected pilot: **39/39 certified QoIs in band** on 5 of 6 parts solved (worst error 1.93 %);
  abc_00000223 fails (3.84 M-node mesh). See `PILOT.md`, `CATALOG.md`.
- `viz/`: `export.sh run_manifest.json` -> `twin.usdc/usdz`; `open.sh` (Isaac Sim 6.2 container on the Spark display),
  `render.sh`; `viz/video/make_video.sh` (headless Isaac Sim render + ffmpeg -> mp4; `DRY_RUN=1` prints the plan),
  `viz/gallery/prep_gallery.sh`. Expected twin: 1,150,527 DOFs, 7.683 mm vs 7.498 mm beam (+2.5 %). Binary outputs
  (twin.usdc, frames, mp4) are not in the repo; regenerate with these scripts.

## Not reproducible from a clone alone
Spark-only items above; `wingbox` meshes/results and gold meshes are regenerated, not stored; `viz/` needs the Spark's
run manifest (`inputs/wingbox-2026-10-31/run_manifest.json`, produced by the wingbox solve on the Spark).
