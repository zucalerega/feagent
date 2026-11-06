# Wingbox stage 1: geometry, meshes and solver cases (2026-10-31)

Everything here was built on the laptop CPU (`CUDA_VISIBLE_DEVICES=""` on every command, no GPU
solve). Commands run from the repo root with `uv run`. Units: mm, N, MPa. Material: 7075-T6,
E = 71700 MPa, nu = 0.33.

## 1. Geometry (`geometry/`)

`geometry/build_wingbox.py` (gmsh OCC boxes) writes `geometry/parts/<part>.step` (10 files) and
`geometry/wingbox.step` (all 10 solids; each part is its own solid, touching face to face; the
harness fragments them conformally on load). `geometry/volumes.json` holds the volume check;
`geometry/inspect.json` holds the `feagent inspect --json` output.

| part | box (x, y, z ranges) | volume, mm^3 |
|---|---|---|
| upper_skin | 0..600, 0..2000, 130..150 | 24,000,000 |
| lower_skin | 0..600, 0..2000, 0..20 | 24,000,000 |
| front_spar | 0..15, 0..2000, 20..130 | 3,300,000 |
| rear_spar | 585..600, 0..2000, 20..130 | 3,300,000 |
| rib_0..rib_5 | 15..585, y0..y0+10 (y0 = 0, 400, 800, 1200, 1600, 1990), 20..130 | 627,000 each |
| **total** | | **58,362,000** |

Check (`uv run feagent inspect geometry/wingbox.step --json`):
- 10 volumes, 76 faces (all planar), 124 edges.
- Volume: hand 58,362,000 = OCC 58,362,000 = inspect sum 58,362,000 (relative error 0, limit 0.1 %).
- 28 `multi_volume_interface` hazards: the shared (bonded, conformal) faces. Per pair: rib_i x
  upper_skin 6, rib_i x lower_skin 6, rib_i x front_spar 6, rib_i x rear_spar 6, spar x skin 4.
- Other hazards are expected for this geometry: 24 sliver faces and 24 short edges (the 10 mm rib
  footprints on the spar and skin faces) and 3 thin_wall hazards (t = 10, 15, 20).

## 2. Per-part meshes (`parts/`), one per part family

Each part was auto-meshed first (`feagent automesh geometry/parts/<part>.step -o parts/<part>`).
Then I edited `parts/<part>/mesh.yaml` (interface and end faces named by query, `expect: 1` each)
and ran `feagent check` until every gate passed with >= 2 quadratic elements through the
thickness (the thin_walls gate). All meshes are order 2 (tet10) with netgen optimization.

Sizing method: `thin_wall_layers: 0`, a global size just above t/2 (that keeps the thin_wall
hazard, so the gate actually measures), plus `refine` on the two large wall faces to bring the
mean edge just under t/2. That gives the fewest DOFs that still pass.

**Harness finding:** automesh's thin-wall auto-refine does not reach its target on the long parts.
On front_spar it asked for size 6 and got a mean edge of 11.6 (thin_walls warn 1.29). On
upper_skin it asked for 8 and got 10.6 (warn 1.88). The rib, at 570 mm long, worked. Likely
cause: the gmsh Distance field samples the long faces coarsely (default Sampling 20; mesher.py
`_threshold` does not set it). The automesh reports are kept as `parts/{front_spar,upper_skin}/automesh_report.json`.

| part (family) | spec sizes | tets | nodes | DOFs (3 x nodes) | min SICN | min gamma | thin_walls (elems through t) |
|---|---|---|---|---|---|---|---|
| rib_0 (ribs, t = 10) | global 6.0; web faces (+-y) 4.7, dist 10 | 30,134 | 53,716 | 161,148 | 0.389 | 0.318 | 2.03 pass |
| front_spar (spars, t = 15) | global 7.55; web faces 7.2, dist 30 | 49,794 | 89,134 | 267,402 | 0.413 | 0.384 | 2.07 pass |
| upper_skin (skins, t = 20) | global 10.05; skin faces 9.85, dist 40 | 140,109 | 247,195 | 741,585 | 0.397 | 0.299 | 2.04 pass |
| automesh rib_0 (reference) | auto (thin wall -> 4) | 32,872 | 58,762 | 176,286 | 0.358 | 0.215 | 2.13 pass |
| automesh front_spar | auto (thin wall -> 6) | 15,753 | 30,388 | 91,164 | 0.283 | 0.178 | 1.29 **warn** |
| automesh upper_skin | auto (thin wall -> 8) | 105,377 | 190,719 | 572,157 | 0.272 | 0.173 | 1.88 **warn** |

Gate table for the three tuned specs (all `status: pass`, every gate pass):

| gate | rib_0 | front_spar | upper_skin | threshold |
|---|---|---|---|---|
| geometry | pass (1 vol) | pass | pass | 1 |
| groups | pass (6 groups, 1 face each) | pass (6) | pass (6) | |
| mesh_generated | pass | pass | pass | >= 1 |
| inverted | pass (0) | pass (0) | pass (0) | 0 |
| sicn | 0.389 | 0.413 | 0.397 | >= 0.1 |
| gamma | 0.318 | 0.384 | 0.299 | >= 0.05 (warn) |
| volume_fidelity | 3.4e-12 | 3.4e-12 | 3.4e-12 | <= 0.01 |
| area_fidelity | 4.1e-15 | 4.1e-15 | 3.9e-15 | <= 0.02 |
| connectivity | 1 | 1 | 1 | = 1 |
| thin_walls | 2.03 | 2.07 | 2.04 | >= 2 |

The other parts use the same specs and were not meshed separately: rib_1..rib_5 are identical to
rib_0, rear_spar mirrors front_spar and lower_skin mirrors upper_skin. At this full thin-wall
resolution the assembly would have about 2 x 247 k + 2 x 89 k + 6 x 54 k = 995 k nodes (~3.0 M
DOFs), which is 7.5x the budget.

## 3. Assembly meshes (`assembly/`, `assembly_c1/`, `assembly_c2/`)

`assembly/mesh.yaml` is for `geometry/wingbox.step`: order 2, conformal (fragmented, 28 bonded
interfaces). Sizes are set per part family by refine fields on that family's faces; the smallest
size wins at shared faces. Skins use the global size, 34. They are thickness-limited, so the
mean edge on the skin outer faces is about 20. Spar webs use 19 (the x-normal faces between the
skins, distance 6). Ribs use 14 (the 12 rib web faces, distance 4).

Size sweep, in nodes: 20/15/10 -> 236 k; 28/18/12 -> 159 k; 34/18/13 -> 135 k; 34/19/14 -> 123 k
(chosen). Fitting the 400 k DOF budget therefore means about one quadratic element through each
wall (or less in the ribs). The **thin_walls gate warns** on all three assembly meshes. It is
warn-only, so the status is still pass. This is a budget-driven, known limitation:
- Box-beam bending is carried mainly by skin membrane action and web shear, which one tet10
  layer represents reasonably.
- Through-thickness bending of the 10 mm ribs and 15 mm webs is under-resolved.

The coarse GCI meshes are copies of the spec with every size (global, min, refine sizes) x 1.3
(`assembly_c1/`) and x 1.69 (`assembly_c2/`). Refine distances are unchanged, the same convention
as `feagent mesh --scale`.

| mesh | sizes skin / spar / rib | tets | nodes | **DOFs** | min SICN | min gamma | thin_walls | status |
|---|---|---|---|---|---|---|---|---|
| assembly (fine) | 34 / 19 / 14 | 64,491 | 123,390 | **370,170** | 0.288 | 0.176 | warn 0.59 | **pass** |
| assembly_c1 (x1.3) | 44.2 / 24.7 / 18.2 | 42,076 | 79,091 | **237,273** | 0.261 | 0.156 | warn 0.46 | **pass** |
| assembly_c2 (x1.69) | 57.5 / 32.1 / 23.7 | 26,915 | 50,071 | **150,213** | 0.231 | 0.131 | warn 0.35 | **pass** |

All other gates pass on all three meshes: geometry 10 volumes; groups; inverted 0;
volume_fidelity 3.4e-12; area_fidelity < 1e-14; **connectivity 1 component**.

**GCI caveat:** the effective refinement ratio from node counts is
r = (N_fine/N_c1)^(1/3) = 1.16 and (N_c1/N_c2)^(1/3) = 1.165, not the nominal 1.3. Element size
through the walls is limited by the wall thickness, so it does not coarsen with the spec sizes.
That is below Celik's r > 1.3 recommendation. Use the solver's `h_rep` (or node counts) for r,
and expect a noisy observed order. A 4th mesh (x0.8, about 240 k nodes / 720 k DOFs, over budget
for the laptop) would firm it up.

### Groups (all meshes; queries with exact `expect`)

| group | query | faces | area, mm^2 | use |
|---|---|---|---|---|
| root | `{at: ymin}` | 5 (2 skin ends, 2 spar ends, rib_0 outer face) | 90,000 | fixed |
| tip | `{at: ymax, box: [[100,1999,30],[500,2001,120]]}` | 1 (rib_5 outer face) | 62,700 | 20 kN +z |
| tip_ends | `{at: ymax, not: <tip box>}` | 4 (skin and spar ends at y = 2000) | 27,300 | unloaded; tip-section deflection |
| lower_outer | `{at: zmin}` | 1 (lower skin z = 0) | 1,200,000 | LC2 pressure face (not loaded in LC1) |

Load choice: as the ICD says, the 20 kN acts on the tip rib's outer face only, and is spread
uniformly per area by the solver (0.319 MPa). The rib carries it in-plane to the spar webs. The
skin and spar end faces at y = 2000 are their own group, `tip_ends`, so the solver can report the
mean deflection of the whole tip section. bcs.fea: `fixed: [root]`,
`loads: [{group: tip, force: [0, 0, 20000]}]`, E 71700, nu 0.33.

## 4. Solver cases

Each case was made with `uv run python -m feacore.adapters.mesh_harness <dir>/out --out <dir>/case.json`.

| case.json | mesh it references | DOFs |
|---|---|---|
| `assembly/case.json` | `assembly/out/mesh.msh` (9.9 MB) | 370,170 |
| `assembly_c1/case.json` | `assembly_c1/out/mesh.msh` (6.3 MB) | 237,273 |
| `assembly_c2/case.json` | `assembly_c2/out/mesh.msh` (4.0 MB) | 150,213 |

Two changes after the adapter:
1. **The mesh path is relative.** The adapter writes an absolute path
   (`<abs path of the run dir>/<dir>/out/mesh.msh`). I rewrote
   `"mesh"` to `"out/mesh.msh"`. `feacore.case.Case.load` resolves a relative mesh path against
   the case JSON's directory, so the case works anywhere as long as `case.json` and
   `out/mesh.msh` keep the same layout. `meta.out_dir` and `meta.spec_path` still hold the
   laptop's absolute paths; they are informational only and the solver does not read them. If
   you regenerate a case with the adapter, rewrite the path again:
   `python3 -c "import json,sys; p=sys.argv[1]; d=json.load(open(p)); d['mesh']='out/mesh.msh'; json.dump(d,open(p,'w'),indent=2)" <dir>/case.json`.
2. **Two QoIs were added** to the adapter defaults: `mean_uz_tip` and `mean_uz_tip_ends` (`mean`
   of `uz`), the tip deflection for the GCI study.

The adapter defaults are strain_energy, max_von_mises, max_displacement, mean_u_mag_tip and
reaction_root_{x,y,z}. On the GPU machine:
- Copy the run keeping its layout: `rsync -a --include='*/' --include='case.json' --include='mesh.msh' --exclude='*' runs/wingbox-2026-10-31/ <host>:<dest>/`
  (or copy the three `assembly*/` directories).
- Solve each case: `python -m feacore.cli <dest>/assembly/case.json --out <dest>/assembly/lc1/result.json --vtu <dest>/assembly/lc1/result.vtu --linear-solver auto`.

### CPU sanity solve (coarsest case only)

`CUDA_VISIBLE_DEVICES="" uv run python -m feacore.cli assembly_c2/case.json --out assembly_c2/cpu_sanity/result.json --linear-solver scipy`:
- 150,213 DOFs, wall time 19.5 s (SuperLU, 16.3 s solve, relative residual 1.6e-9).
- Reactions: R_z = -20000.000 N (equals the applied 20 kN); R_x = 3.6e-7 N, R_y = -7.5e-8 N.
- Tip deflection: mean u_z = 7.671 mm on `tip` and 7.661 mm on `tip_ends`.
- Beam theory: I_xx = 105.53e6 mm^4 (skins 102.2e6 + webs 3.33e6). Bending PL^3/(3EI) = 7.05 mm;
  web shear PL/(G A_web) with G = 26,955 MPa and A_web = 2 x 15 x 110 gives 0.45 mm. Total 7.50 mm.
  The FE value is 2.3 % above it, inside the ICD's 10 % acceptance; this is the coarsest mesh only.
- Strain energy 76,711 N mm; 0.5 P u_tip = 76,711 N mm (consistent).
- Max von Mises 43.8 MPa vs beam-theory root fibre stress M c / I = 28.4 MPa. The peak sits at the
  clamped-root corners (a singularity under a fully fixed root), so it is not a converging QoI.

No solve ran on the fine or c1 meshes, and none on the GPU.

## Files

- `geometry/build_wingbox.py`, `geometry/parts/*.step` (10), `geometry/wingbox.step`,
  `geometry/volumes.json`, `geometry/inspect.json`
- `parts/{rib_0,front_spar,upper_skin}/mesh.yaml` with `out/` (report.json, REPORT.md, mesh.msh,
  mesh.inp, renders); `parts/*.automesh.log`
- `assembly{,_c1,_c2}/mesh.yaml`, `out/` (report.json, REPORT.md, mesh.msh, mesh.inp,
  groups.png, quality.png), `case.json`
- `assembly_c2/cpu_sanity/result.json`, `solve.log`

`mesh.inp` also holds surface elements (CPS6) for each group. Those must not get a section; only
the `domain` ELSET is the solid.

Side effect outside this directory: `feagent check` records Stop-hook state in the repo-root
`.feagent/state.json` (git-ignored, shared with other sessions). I left it unchanged. It contains
pass entries for six intermediate `assembly_try_*` specs from the size sweep, which I have since
deleted.
