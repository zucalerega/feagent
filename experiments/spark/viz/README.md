# FEA virtual twin of the wingbox (OpenUSD), on the GB10

`twin.usdc` (and `twin.usdz` with its textures packed in) is the stage for wingbox stage 1, LC1, solved
on the xf mesh:

- **Run:** 20 kN tip lift, 1,150,527 DOFs, cuDSS on the NVIDIA GB10, 24.2 s.
- **Tip deflection:** 7.683 mm (extrapolated 7.687), against 7.498 mm from beam theory (+2.5 %).
- **Checks:** GCI 0.054 %, equilibrium to 1e-11. Verdict: **pass**.
- **Source run:** `~/runs/wingbox-2026-10-31/assembly_xf`.

## How to open at the Spark

1. At the Spark's desktop, open a terminal and run `~/feagent-viz/open.sh`. It starts Isaac Sim 6.2.0 (RTX) in the foreground as container `feagent-twin`.
2. It refuses to start while another Isaac Sim container is running, such as `ridgeline-isaac`, because the GB10 runs one large GPU job at a time.
3. First start compiles shaders and takes several minutes. Close the window or press Ctrl-C to stop; the container removes itself (`--rm`).
4. To switch fields, select `/World`, then in the Property panel set Variants → `field` to `von_mises`, `u_mag`, `gate_status` or `part_colors`. The timeline plays the deformation (×20).
5. If no window appears, run `xhost +local:` once in the desktop session, then `xhost -local:` afterwards. `DRY_RUN=1 ./open.sh` prints the docker command without running it.

## What is in the stage

- **Parts:** `/World/Wingbox/<part>`: upper_skin, lower_skin, front_spar, rear_spar, rib_0..rib_5.
  These are volume entities 1..10 of `out/mesh.msh`.
  - Attributes: `fea:gate_status`, `fea:dofs`, `fea:owner`, `fea:max_von_mises`, `fea:max_u_mag`.
  - customData `feagent`: gate details, part spec, part-spec mesh DOFs, cell/node/triangle counts.
  - **Gates:** every part spec passes all gates. The solved xf assembly mesh passes everything
    except thin_walls, which warns: about 1 quadratic element through the walls, minimum 0.96, and
    the gate needs 2. So `gate_status` is **warn** on all 10 parts.
- **Surface:** `<part>/Surface` is the boundary of the part's tet10 cells, each quadratic face split
  in 4: 456,672 triangles in total.
  - Primvars: `von_mises` [MPa], `u_mag` [mm], `displacement` [mm, unscaled], `gate_status`, `gate_code`.
  - The points are time-sampled: t = 0 is undeformed, and t = 24 is the deformation × 20 (tip lift
    about 154 mm on screen).
- **Materials:** one UsdPreviewSurface per part, in `/World/Looks/<part>`. diffuseColor and opacity
  come from primvar readers of `displayColor` and `displayOpacity`.
- **Colour variants:** the variant set `field` on `/World` switches every part's colours and the
  legend quad (`/World/Legend`).
  - von Mises uses the converged range, 0–26.9 MPa, which is the 99.5th percentile. The 60.6 MPa
    root-corner peak is a singularity: it grows with refinement and is not a design value. Away from
    the root the converged field is about 24 MPa.
  - This note is in the stage summary (`stress_note`) and on the legend.
- **Run summary:** in the layer's `customLayerData.feagent`, in `/World` customData
  `feagent:summary` and in `summary.json`. It holds the verdict, DOFs, solver, solve time, tip vs
  beam theory, GCI, equilibrium, stress note and mesh caveat.
- **Camera and lights:** camera `/World/Camera`; lights `/World/Lights/{Key,Dome}`. `open_in_kit.py`
  also turns on RTX auto-exposure.
- **Scene:** Z-up, `metersPerUnit` 0.001 (mm). The stage passes all 30 UsdValidation validators with
  0 errors. usd-core 26.11 has no `usdchecker` script, so `export_twin.py` runs those validators
  instead.

## Re-export with new results (one command)

    ~/feagent-viz/export.sh ~/feagent-viz/inputs/wingbox-2026-10-31/run_manifest.json   # or your own manifest

This writes the stage, legends and `summary.json`, validates them, and makes GPU preview renders
through EGL on the GB10 (`renders/`): `iso`, `top`, `root_closeup`, `deformed`, plus `field_<variant>`.

- It runs under `systemd-run --user --scope -p MemoryMax=8G` and starts no service.
- Manifest format: see the docstring of `export_twin.py`.
- **Part selectors:** `physical`, `volumes`, `mesh` or `material`.
- **Gate status:** the gate comes from `report`, `gate` and `extra_gates`; the worst one wins.
- **Optional keys:** `vm_clip_percentile`, `legend_notes`, and `summary.*`, which is passed through
  to the stage.
- **Options:** `--scale`, `--frames`, `--max-tris`.

## Files

| Path | What |
|---|---|
| `export_twin.py`, `export.sh` | manifest + msh + vtu → stage, legends, summary, validation; plus preview renders |
| `render_twin.py`, `render.sh` | VTK offscreen preview: GPU via EGL (GB10), with Mesa llvmpipe (EGL device 1) as the CPU fallback |
| `open.sh`, `open_in_kit.py` | Isaac Sim GUI launcher (docker, not started by default) and the Kit script that opens the stage, sets the camera and plays the timeline |
| `inputs/wingbox-2026-10-31/` | run manifest, part-spec and assembly gate reports, run summary |
| `examples/` | earlier placeholder test twins (bracket, 5-part box beam); not the wingbox |
| `VIEWER-RESEARCH.md` | the viewer survey (Isaac Sim / Kit / usdview / others) |
