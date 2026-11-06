# 03: FEA virtual twin of the wingbox (OpenUSD) on the GB10, and the Isaac Sim viewer

**Current state (real results).** `~/feagent-viz/twin.usdc` is the wingbox stage-1 LC1 result. The
placeholder twins are moved to `~/feagent-viz/examples/`. The stage has no placeholder labels left:
grep finds none in the usdc, the usdz, `summary.json` or the textures.

- **Constraints met:** no sudo, no persistence, nothing left running, laptop GPU never used.
- **Docker:** only read-only commands (`docker ps`, `docker image inspect`). `ridgeline-isaac` was
  not touched. No container was started.

## Real wingbox twin

- **Inputs.** All are in `~/runs/wingbox-2026-10-31/assembly_xf/`: `result.vtu` (tetra10,
  383,509 nodes, 208,781 cells), `result.json`, `out/mesh.msh`.
  - Laptop-side checks: the verdict in `runs/wingbox-2026-10-31/summary.json` is **"pass"**, and
    `verify/verdict.md` agrees. Neither was re-run.
  - The manifest is `~/feagent-viz/inputs/wingbox-2026-10-31/run_manifest.json`, built from these.
- **Parts.** The msh has one dim-3 physical group (`domain`), so each part is selected by its volume
  entity tag. I checked the tags against the README's part boxes using the bounding box of each
  entity's tet centroids:

  | tag | part |
  |---|---|
  | 1 | upper_skin |
  | 2 | lower_skin |
  | 3 | front_spar |
  | 4 | rear_spar |
  | 5..10 | rib_0..rib_5 (y = 0, 400, 800, 1200, 1600, 1990) |

  - Each part's DOFs are 3 × its nodes in the solved mesh; shared interface nodes count in each part.
    For example, upper_skin has 409,593, front_spar 75,309 and rib_0 38,145.
  - The part-spec mesh DOFs are kept in customData: skin 741,585, spar 267,402, rib 161,148.
- **Gate status.** Every part is **warn**, which is the honest status of the mesh that was solved:
  - The part specs (`parts/{upper_skin,front_spar,rib_0}/out/report.json`) pass every gate. The other
    parts share these specs: lower_skin uses the skin spec, rear_spar the spar spec, and
    rib_1..rib_5 the rib spec.
  - The solved xf assembly mesh passes every gate except thin_walls, which warns. The flagged faces
    cover every family: ribs about 1.08 elements through the wall, spar webs 0.96–1.39, skins
    1.06–1.98. The gate needs 2.
  - Each part's customData lists both the part-spec status and the assembly-mesh gate. The gate
    legend and `gate_note` explain the warn.
- **Stage.** `twin.usdc` is 81.6 MB and `twin.usdz` 81.8 MB. The export takes 2.5 s on the Spark.
  - The surface is 456,672 triangles and 228,356 vertices (quadratic faces split into 4), over
    24 time samples.
  - The deformation scale is chosen automatically: **×20**. That gives a tip lift of about 154 mm on
    screen, against a 150 mm deep box and a 2000 mm span.
  - All 30 UsdValidation validators report **0 errors** on both the usdc and the usdz.
- **von Mises colours.** The colour range is 0–**26.9 MPa**, the 99.5th percentile of the nodal
  field. For comparison, p99 is 25.3 and p99.9 is 32.7; the maximum for 300 ≤ y ≤ 1700 is 24.19.
  - The legend top reads "≥ 26.9".
  - `stress_note`, the legend and the renders all state that the 60.61 MPa peak at (600, 0, 150) is
    the clamped-root-corner singularity: 43.8 → 48.2 → 50.7 → 60.6 MPa with refinement, so not a
    design value. The converged field away from the root is about 24 MPa, and M c / I gives 28.4 MPa
    at the root.
- **Stage summary.** This is `customLayerData.feagent`, `/World` customData `feagent:summary` and
  `summary.json`. It holds:
  - verdict pass and its note;
  - 1,150,527 DOFs; solver "cuDSS on NVIDIA GB10 (DGX Spark)"; solve 24.24 s (linear solve 9.79 s);
  - tip 7.683 mm (extrapolated 7.687) vs beam theory 7.498 mm (+2.5 %);
  - GCI 0.054 % (conservative 0.128 %), observed order 2.29;
  - equilibrium 1e-11 (R_z rel. error 6.8e-12);
  - load case, stress note and mesh caveat.
- **EGL renders.** VTK 9.7.1 through `vtkEGLRenderWindow`, renderer "NVIDIA GB10/PCIe", at
  1920x1080, 457k triangles. Frame times:

  | render | content | ms/frame |
  |---|---|---|
  | `iso.png` | von Mises, deformed ×20, undeformed outline | 24.7 |
  | `top.png` | von Mises, plan view | 17.2 |
  | `root_closeup.png` | von Mises, with the singular peak labelled | 14.4 |
  | `deformed.png` | \|u\|, side view, ×20 with outline | 26.3 |

  - There are also `field_{von_mises,u_mag,gate_status,part_colors}.png`.
  - The four main renders are copied to `<repo>/runs/wingbox-2026-10-31/viz/`
    together with `render_log.json`.
  - The CPU fallback (llvmpipe) took about 130 ms/frame on the 29k-triangle bracket.

## Isaac Sim launch: prepared, NOT run (`~/feagent-viz/open.sh`)

- **Image.** `docker image inspect nvcr.io/nvidia/isaac-sim:6.2.0`:
  - arm64/linux, 9.25 GB of layers (26.8 GB as `docker images` reports it);
  - User `isaac-sim`, WorkingDir `/isaac-sim`, Entrypoint `/isaac-sim/runheadless.sh`;
  - Env: `NVIDIA_DRIVER_CAPABILITIES=all`, `VK_DRIVER_FILES=/etc/vulkan/icd.d/nvidia_icd.json`,
    `MIN_DRIVER_VERSION=570.169`; the Spark has 580.178.04.
  - The image's layer files (`/var/lib/docker`, `/var/lib/containerd`) are root-only, so I could not
    read `isaac-sim.sh` or `runapp.sh` directly.
  - The launch follows the official container docs: GUI = `xhost +local:`, `--entrypoint bash`,
    `-e DISPLAY`, an `.Xauthority` mount, `-u 1234:1234`, the cache mounts, then `./runapp.sh`.
  - It also follows the other agent's working headless pattern (`~/sculptured-examples/isaac-gpu.sh`,
    read only): libgomp preload, chmod a+rwX cache dirs instead of chown.
- **Display.** Xorg `:1` (`/tmp/.X11-unix/X1`), session type x11, auth
  `/run/user/1000/gdm/Xauthority`. The current ACL is `SI:localuser:dell` only.
  - `open.sh` therefore copies the cookie as FamilyWild into a short-lived 0644 file (removed on
    exit) and mounts it as `/isaac-sim/.Xauthority`, so `xhost` should not be needed.
  - If the window still does not appear, it prints the `xhost +local:` / `xhost -local:` hint. It
    never runs xhost itself.
- **Command.** `docker run --name feagent-twin --rm --gpus all --entrypoint bash`, with:
  - EULA env: `ACCEPT_EULA=Y`, `PRIVACY_CONSENT=Y`, `OMNI_KIT_ACCEPT_EULA=YES`;
  - display: `DISPLAY=:1`, `/tmp/.X11-unix` mounted read-only;
  - the twin: `~/feagent-viz` mounted as `/twin:ro`, `TWIN_USD=/twin/twin.usdc`;
  - its own caches in `~/.cache/feagent-twin-isaac/*`, not the other agent's;
  - `-u 1234:1234`;
  - then `./isaac-sim.sh --exec /twin/open_in_kit.py`, falling back to `./runapp.sh --exec …`.
  - It uses the default bridge network, not `--network=host`, so no Kit listener is exposed on the
    LAN.
- **`open_in_kit.py`:**
  - waits 120 frames, then opens the stage and reopens it if startup replaced it with an empty stage
    (a forum-reported issue);
  - turns on RTX auto-exposure (`/rtx/post/histogram/enabled`);
  - sets the viewport camera to `/World/Camera`;
  - loops the timeline.
- **Guard.** If `docker ps` shows any container whose name or image matches `isaac` (e.g.
  `ridgeline-isaac`), `open.sh` refuses (exit 3) and explains why: one large GPU job at a time.
  - Tested: it refused while `ridgeline-isaac` was running.
  - `DRY_RUN=1` printed the full command from a temporary copy with the guard disabled; nothing ran.
- **On start, `open.sh` prints:**
  - how to switch fields: select `/World`, then Property → Variants → `field`;
  - that the timeline plays t = 0..24 = ×20;
  - the camera and the per-part `fea:*` data;
  - how to stop it.
- **Unverified until the first real start:**
  - that `--exec` is forwarded by the in-image script (it is the documented Kit flag; the forum shows
    `isaac-sim.sh --exec`);
  - the exact viewport/auto-exposure APIs (they are wrapped in try/except);
  - the first-start shader compile time (several minutes).
  - The `--gpus all` graphics path in this image already works headless for the other agent.

### Stage design (what export_twin.py writes)

- **Stage metadata.** `upAxis = Z`, `metersPerUnit = 0.001`. The time codes run 0..N (N = 24) at
  N per second.
- **Parts.** Each part is `/World/Wingbox/<part>` (Xform, kind=component), under `/World/Wingbox`
  (kind=assembly). A part carries the attributes `fea:gate_status`, `fea:dofs`, `fea:owner`,
  `fea:max_von_mises` and `fea:max_u_mag`. Its customData `feagent` holds the gate status, DOFs and
  their source, owner, cell and node counts, the gates that did not pass (part-spec report plus
  `extra_gates`; the worst status wins), and a doc string. A placeholder flag is written only for
  placeholder runs.
- **Surface.** `<part>/Surface` is a UsdGeomMesh: the boundary of the part's tets, with tet10 faces
  split into 4 triangles. If the triangle count exceeds `--max-tris` (2.5 M), it falls back to linear
  faces and then to vtkDecimatePro.
  - **Deformed shape.** The points are time-sampled: rest + displacement × scale × t/N. Extents are
    sampled too. The scale is picked automatically (0.08 × the largest extent ÷ max |u|,
    rounded to 1/2/5; 20× for the wingbox) or set with `--scale`.
  - **Primvars.** `von_mises` [MPa], `u_mag` [mm], `displacement` [mm, unscaled], `gate_status`
    (token) and `gate_code`.
- **Materials.** Each part has its own UsdPreviewSurface in `/World/Looks/<part>`. Its diffuseColor
  comes from `UsdPrimvarReader_float3("displayColor")` and its opacity from
  `UsdPrimvarReader_float("displayOpacity")`.
  - I did not add an MDL output on purpose. Kit/RTX would then render the MDL, which ignores the
    primvar colours. RTX renders UsdPreviewSurface natively.
- **Variant set `field` on `/World`.** The variants are `von_mises`, `u_mag`, `gate_status` and
  `part_colors`. Each variant authors every part's `displayColor` and switches the legend texture.
  - von_mises and u_mag are per-vertex turbo colours over one global range, converted from sRGB to
    linear.
  - gate_status gives each part a constant colour and fades parts that pass (`displayOpacity` 0.25),
    so warn/fail parts stand out.
  - `displayColor` is never authored locally. A local opinion would beat the variant.
- **Legend.** `/World/Legend/Bar` is a textured quad (UsdUVTexture into the emissive channel) beside
  the model, facing the camera. It shows the colour bar, min/max, units and deformation scale, plus
  a red PLACEHOLDER line.
- **Camera and lights.** `/World/Camera` is framed iso. The lights are `/World/Lights/Key`
  (DistantLight) and `Dome` (DomeLight).
- **Run summary.** It is in `customLayerData.feagent`, in `/World` customData `feagent:summary` and
  in the doc string: DOFs, solve time, max von Mises and |u|, tip deflection against beam theory, and
  GCI (n/a for the placeholders).
- **Validation.** usd-core 26.11 has no `usdchecker` script. I ran every one of its 30 registered
  UsdValidation validators instead, which is what `usdchecker` uses. Result: **0 errors on both the
  usdc and the usdz**. The first run caught `varname` authored as a token instead of a string; that
  is fixed.


## Which native OpenUSD viewer can run on the GB10

| Option | aarch64 / DGX Spark evidence | Download | Memory | How the user opens twin.usdc | Risks |
|---|---|---|---|---|---|
| **(b) Isaac Sim 6.2.0 standalone** (Kit app, Omniverse RTX real-time and path-traced) | Requirements page: "Isaac Sim aarch64 builds are currently only supported on DGX Spark system". Tested on DGX OS 7 with driver 580.159.03; this Spark has 580.178.04. An official linux-aarch64 zip exists. Limitations on Spark: no livestreaming, no cuRobo/cuMotion, no App Selector, and (5.1) no OBJ import. None of these matter for viewing USD. | **10.45 GB zip** (HEAD request: 10,451,064,004 B; md5 535e23a8…). Unpacked: an estimated ~20–25 GB, plus shader/extension caches in `~/.cache/ov`. | x86 guidance: 32 GB RAM minimum, 16 GB VRAM minimum, 48 GB ideal. That is fine with ~117 GB unified memory now free. | `~/feagent-viz/open.sh`, which runs `isaac-sim.sh --exec open_in_kit.py`: opens the stage, sets `/World/Camera`, plays the timeline. Fields: Property panel → Variants → `field`. | Size; first start compiles shaders (minutes); GPU-heavy, so it should not run alongside a large GPU job. The pip route needs a 4.8 GB `isaacsim-extscache-kit` wheel. `LD_PRELOAD=libgomp.so.1` workaround (already in open.sh). Light intensities are tuned for Storm, so RTX may need exposure raised (untested). |
| **(a) Omniverse Kit SDK / kit-app-template** (USD Composer, Explorer, Viewer) | Kit 109.0.1 highlights: "Kit SDK now runs on ARM-based hardware" and "Resolved DGX Spark rendering issues". NGC has `kit-kernel-linux-arm` 110.4.0. But the kit-app-template README names no aarch64 support. Forum users hit missing `kit-kernel …aarch64` packages (108.1, 110.0) and only got it working by pinning the NGC ARM kernel by hand. | Kernel 92.5 MB, but NGC sign-in (Developer Program) is required. The template build then pulls the RTX and extension packages: several GB, exact size not stated. | Similar to Isaac Sim. | Build `usd_viewer` or `usd_composer`, then `./repo.sh launch`. | Unofficial build path for templates; fragile; sign-in needed. Not recommended now; worth revisiting for a branded viewer. |
| **(c) usdview** (OpenUSD 26.08 from conda-forge: Storm/OpenGL, Qt6) | conda-forge `openusd` has linux-aarch64 builds. The recipe tests `usdview --help` and depends on pyside6 and pyopengl. Resolved on the Spark with a dry run (nothing downloaded). | **315.6 MB total**, 121 packages; the largest single package is qt6-main at 63 MB. About 1 GB on disk. | < 2 GB | `~/feagent-viz/open.sh twin.usdc usdview`, which runs `usdview twin.usdc --camera /World/Camera`. Variants are in the Metadata tab; the timeline plays the deformation. | OpenGL raster (Storm), not RTX. conda Qt on the Spark's display is untested (try `QT_QPA_PLATFORM=xcb` if it fails). Building from source instead takes about 30–60 min with build tools. NVIDIA's usd-exchange and `usd-core` wheels contain no imaging or usdview. |
| (d) others | ParaView 6.1 is on conda-forge for linux-aarch64, but I found no OpenUSD reader (it would need VTU, not USD). Blender has no official Linux arm64 build; Debian arm64 packages exist, USD support unverified. F3D's USD plugin is an optional build with no aarch64 binary found. | — | — | — | Not credible as the USD viewer here. |

## Recommendation (the user chose Isaac Sim; the 6.2.0 image is already on the Spark)

**Isaac Sim 6.2.0 standalone (aarch64).** It is the only officially supported Omniverse RTX
application on DGX Spark. With the vLLM model gone, memory is no longer a constraint. It opens
`twin.usdc` natively: the variants, time samples, UsdPreviewSurface primvar readers, UsdLux lights
and camera all work.

It needs **user approval for a 10.45 GB download**. Install it user-level in `~/isaacsim`, with no
sudo and no Docker. If a quick, light option is wanted first, or as a fallback, use **usdview from
conda-forge** (315.6 MB total, no single file over 63 MB).

**Found during the final check:** the Spark already has the Docker image
`nvcr.io/nvidia/isaac-sim:6.2.0` (26.8 GB, aarch64). Another agent was running it headless as the
container `ridgeline-isaac` (`isaac-gpu.sh smoke`). I left Docker alone, as my rules require.

If the user allows Docker for viewing, this image avoids the 10.45 GB download. It would need a GUI
run with something like:

    xhost +local: && docker run --rm --gpus all -e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -e ACCEPT_EULA=Y \
      -v ~/feagent-viz:/twin:ro nvcr.io/nvidia/isaac-sim:6.2.0 ./isaac-sim.sh

This command is untested. Coordinate it with the other agent's GPU use: run one large GPU job at a
time.

### Install steps (not executed)

Isaac Sim (after approval):

    cd ~/Downloads && wget https://downloads.isaacsim.nvidia.com/isaac-sim-standalone-6.2.0-linux-aarch64.zip
    echo "535e23a85a36f6e159e3d92cc7d957b3  isaac-sim-standalone-6.2.0-linux-aarch64.zip" | md5sum -c
    mkdir -p ~/isaacsim && unzip -q isaac-sim-standalone-6.2.0-linux-aarch64.zip -d ~/isaacsim
    ~/feagent-viz/open.sh        # foreground; EULA accepted through OMNI_KIT_ACCEPT_EULA=YES; first start compiles shaders
    # removal: rm -rf ~/isaacsim ~/.cache/ov ~/.local/share/ov ~/.nvidia-omniverse ~/Downloads/isaac-sim-standalone-6.2.0-linux-aarch64.zip

usdview:

    export MAMBA_ROOT_PREFIX=~/micromamba
    ~/micromamba/bin/micromamba create -y -n usdview -c conda-forge openusd=26.08 python=3.12
    ~/feagent-viz/open.sh ~/feagent-viz/twin.usdc usdview
    # removal: ~/micromamba/bin/micromamba env remove -y -n usdview

## Sources

- [Isaac Sim 5.1 requirements (DGX Spark aarch64)](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html)
- [Isaac Sim latest requirements](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/requirements.html)
- [Isaac Sim download page (6.2.0 linux-aarch64 zip)](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/download.html)
- [Isaac Sim 5.1 pip install notes (libgomp preload on Spark)](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/install_python.html)
- [pypi.nvidia.com isaacsim wheels](https://pypi.nvidia.com/isaacsim/)
- [Kit 109.0.1 release highlights (ARM, DGX Spark)](https://docs.omniverse.nvidia.com/dev-guide/latest/release-notes/109_0_1_highlights.html)
- [NGC kit-kernel-linux-arm](https://catalog.ngc.nvidia.com/orgs/nvidia/omniverse/resources/kit-kernel-linux-arm)
- [kit-app-template](https://github.com/NVIDIA-Omniverse/kit-app-template)
- [Forum: kit-kernel aarch64 on DGX Spark (Kit 110)](https://forums.developer.nvidia.com/t/seeking-kit-kernel-110-0-0-feature-linux-aarch64-release-for-e2cc-on-dgx-spark/365622)
- [Forum: building Kit on DGX Spark, package not found](https://forums.developer.nvidia.com/t/build-omniverse-kit-on-dgx-spark-package-not-found-on-specified-remote-servers/348212)
- [conda-forge openusd](https://anaconda.org/conda-forge/openusd)
- [openusd-feedstock recipe](https://github.com/conda-forge/openusd-feedstock/blob/main/recipe/recipe.yaml)
- [ParaView 6.0 release](https://discourse.paraview.org/t/paraview-6-0-0-has-been-released/17036)
- [Blender on ARM64 Linux](https://www.blenderkit.com/articles/blenderkit-on-arm64-linux/)
- [F3D releases](https://github.com/f3d-app/f3d/releases)

