# Mesh systems: the contract every system follows

Six mesh systems mesh the same two physical-test parts so every experiment (linear compression over time,
nonlinear crush to snap, videos) can be rerun per system and the results compared: how much do the answers
depend on the mesh system, and which system gets there cheapest?

| id | System | Built from (FEAgent) |
|---|---|---|
| `harness_auto` | the mesh harness, zero-config `feagent automesh` + groups by query | `meshing-harness/` (baseline, already used) |
| `harness_expert` | the mesh harness driven like an expert: digest + hazards → local refinement, thin-wall layers, curvature | `meshing-harness/` (`inspect`, `refine`, `size.*`) |
| `ace_sizing` | the ACE loop's mesher: global `h` + refinements at groups/points (`ace/fea.make_mesh` style) on `feacore.geometry.gmsh_tools` | `ace/fea.py`, `feacore/geometry/` |
| `gold_nested` | the gold generator's approach: one base mesh, uniformly refined by splitting (`gmsh mesh.refine`) into nested levels | `feacore/goldgen` (branch gold-benchmarks), `feacore/verification` (`nested=True`) |
| `netgen` | an independent mesh generator: Netgen (`netgen-mesher`, OCC geometry) | new; Netgen also optimises gmsh meshes in the harness |
| `harness_hex` | the harness's gmsh with hex-dominant / subdivision meshing where it works, else tets with Netgen optimisation | `meshing-harness/` + gmsh options |

## Parts and loading (identical for every system)

| Part | STEP (metres) | Axis | Bottom (fixed platen) | Top (moving platen) |
|---|---|---|---|---|
| `abc_00000003` (closed tube with stubs) | `benchmarks/gold/abc_00000003_bc0/public/geometry.step` | z | plane face at z = 0 (1 face) | plane face at z = 0.06125 (1 face) |
| `abc_00000065` (shaft clamp) | `benchmarks/gold/abc_00000065_bc0/public/geometry.step` | y | plane face at y = −0.04396 (1 face) | plane faces at y = 0.03384 (2 faces) |

Material: `experiments/material/alpha_fast_dogbone1.json` (Formlabs Alpha Fast Resin, 2-min cure, calibrated
from the user's dogbone; SI conversion E [MPa] × 1e6). Units: metres, N, Pa.

## What a system delivers

`experiments/mesh_systems/<id>/build.py PART_STEP AXIS OUT_DIR --purpose linear|nonlinear` writes into OUT_DIR:

- `mesh.msh` — gmsh MSH 4.1, physical groups named exactly `bottom`, `top` (dim 2) and `domain` (dim 3).
  - `--purpose linear`: second order (tet10) preferred, for feacore/skfem; target 50k–400k DOFs.
  - `--purpose nonlinear`: for CalculiX elastoplastic + erosion: C3D10 with straight interior edges
    (CalculiX refuses curved-interior quadratic tets; see CLAUDE.md) or C3D4 if the system argues for it; target
    ≤ 40k nodes so a crush to separation finishes in hours, not days.
- `mesh.inp` — the same mesh for CalculiX with `*NSET`/`*ELSET` `bottom`, `top`, `domain` (no surface elements in
  `domain`).
- `report.json` — the harness quality gates on this mesh (`feagent.quality.assess_mesh`; same thresholds for every
  system), plus `{"system", "part", "purpose", "n_nodes", "n_elements", "order", "dofs", "build_s"}`.
- `mesh.png` — a picture of the mesh made with our tooling (`feagent render ... --view quality`, CPU renderer:
  `LIBGL_ALWAYS_SOFTWARE=1 GALLIUM_DRIVER=llvmpipe __GLX_VENDOR_LIBRARY_NAME=mesa`).

Plus `experiments/mesh_systems/<id>/README.md`: the approach in one paragraph, which FEAgent pieces it uses, its
knobs, and the numbers for both parts (DOFs, gates, build time).

## Rules

- Never use the laptop's GPU: every command with `CUDA_VISIBLE_DEVICES=""`; CPU rendering as above.
- Faces are chosen by geometry (normal + position), never by remembered tag; check the counts above.
- No absolute `/home/...` or `/tmp/...` paths in committed files; paths relative to the repo or arguments.
- Each system is deterministic for the same inputs (fixed seeds/threads where the mesher allows).

## The experiments run per system (by `experiments/physical/run_all.py`, after the systems exist)

1. Linear compression over time: feacore/skfem, Alpha Fast elastic (E, nu from the material JSON), top force ramped
   0 → F_full (F_full = force that brings the converged peak von Mises to 87 % of the UTS), ≥ 10 s video in the
   original 3-panel look (outside / cut von Mises / cut displacement + arrows).
2. Nonlinear crush to snap: feacore nonlinear CalculiX backend (`scripts/ccx_crush.py`, J2 plasticity from the
   dogbone table, NLGEOM, erosion at PEEQ = failure_peeq), displacement control, small increments through yield and
   failure onset, until full separation; ≥ 10 s video in the original look + an Omniverse (Isaac Sim RTX) video.
3. Comparison report: per system and part — DOFs, gates, linear stiffness and peak stress, nonlinear peak force,
   travel at peak, travel at separation, where it cracked, wall time; spread across systems.
