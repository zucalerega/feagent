# Nonlinear crush runs (to snap)

Runner: `experiments/physical/crush.py` (CalculiX elastoplastic + NLGEOM + erosion at PEEQ >= 0.043, Alpha Fast
dogbone material). Run dirs are `runs/nonlinear/...` in the main checkout (not committed). Each run writes
`progress.log`, `force_displacement.csv`, `result_NNNN.vtu`, `summary.json` (at the end), `run.json`, `case.json`,
and `checkpoint.json/.npz` plus `ccx_work/` (the restart file of its last block). Nothing is ever deleted or
overwritten: a continuation goes to a new run dir.

## Step policy (user requirement)

Platen travel per output increment (every converged ccx increment is written):

| phase | bound |
|---|---|
| through yield (until a cell reaches half the failure PEEQ) | <= 0.05 mm |
| from then on (before the first failed cell) until the force at a block end is < 60 % of its peak | <= 0.02 mm |
| tail (crack established, latched) | <= 0.05 mm |

Erosion blocks (one ccx restart step each; failed cells are removed between blocks, at most 8 per block,
highest PEEQ first): <= 0.25 mm, <= 0.06 mm near failure, >= 0.02 mm, <= 0.10 mm in the tail; blocks that only
release deferred failed cells advance 0.002 mm. Stop: full separation (no face-connected, supported chain of intact
cells joins bottom and top), or force < 1 % of peak for 3 consecutive increments after the first failure, or 30 % of
the part height of travel.

## Baseline (`harness_auto`, coarse harness meshes), started 2026-10-31 20:31 ET, 8 threads each

| part | run dir | continues | from | mesh |
|---|---|---|---|---|
| abc_00000003 | `runs/nonlinear/abc_00000003/run-20261031-2031c` | `run-20261031-1920` (stopped when its session ended) | step 107, 1.190 mm, block-31 restart file | 23 926 nodes / 12 489 C3D10 |
| abc_00000065 | `runs/nonlinear/abc_00000065/run-20261031-2031c` | `run-20261031-1925` (same) | step 93, 1.292 mm, block-24 restart file | 33 824 nodes / 19 076 C3D10 (straight edges) |

The parent runs ran with ccx_crush.py, which had no checkpoint support. Each continuation rebuilt that checkpoint
from the parent's last `.vtu` (cell state and PEEQ, point displacement), the CSV (step, block, peak force) and the
first `.vtu` that flags each cell (removal step). It took the block-end travel from the parent's last block deck
(14 digits) and used the restart file `.rout` of that block, so CalculiX continues from the saved state.
Two things restart fresh: the blocks a stuck cell waits and the PEEQ growth rate (it only sizes the next block).
Step numbers continue (108..., 94...). `force_displacement_full.csv` and `summary.json` (written at the end) cover
parent + continuation. The parent runs used a ramp of 50 % of the height with the same 0.05 / 0.02 mm bounds;
the continuation keeps that ramp (the restart needs the same deck) and stops at 30 % of the height.

Resume a run of this runner after an interruption (new run dir, steps continue):

    python experiments/physical/crush.py --system <id> --part <part> --resume runs/nonlinear/<...>/<run-id> --out <dir>

## Other mesh systems, started 2026-10-31 20:51 ET (same step policy and stops, from zero, 30 % of height ramp)

Queue: `experiments/physical/crush_queue.py`. Each slot runs its jobs one after another. The queue is a nohup
foreground process; nothing is installed, so it does not survive a reboot.

- DGX Spark (`<spark-host>`, aarch64, user `dell`):
  - Code: clone of `nonlinear-setup` in `~/feagent-setup-run`, branch `run-local` = e5aca15, which carries this
    branch's unpushed commits.
  - Software: ccx 2.23 from `~/micromamba/envs/ccx` and Python from `~/micromamba/envs/feagent`.
  - Meshes: copied to `~/feagent-meshes/nonlinear/<system>/<part>/`, with the sha256 of each in `SHA256SUMS`.
  - Queue: 3 slots x 4 threads. Each run is in a `systemd-run --user --scope -p MemoryMax=12G` and starts only
    while MemAvailable >= 32 GiB. The ridgeline-isaac container is left alone.
  - Runs go to `~/feagent-runs/nonlinear/<system>/<part>/<run-id>/`. The queue state is in `~/feagent-runs/queue/`.
  - The Spark clock is US Central, so its run ids read one hour earlier than ET.
- Laptop: 1 slot x 4 threads (plus the two baseline runs), MemoryMax 10G. Runs go to
  `runs/nonlinear/<system>/<part>/<run-id>/`.
- Copy back: `crush_queue.py sync` runs on the laptop. Every 10 min it rsyncs the Spark's `nonlinear/` (without
  `ccx_work/`) into `runs/nonlinear/` and the queue state into `runs/queue_<spark-host>/`, and stops after the
  queue is done.

| system | part | machine | run dir (under runs/nonlinear on its machine) | mesh (nodes / cells, element) | mesh sha256 (16) | start (ET) |
|---|---|---|---|---|---|---|
| harness_hex | abc_00000003 | Spark slot 0 | `harness_hex/abc_00000003/run-20261031-1950-s0` | 38 417 / 29 340, C3D8 | 74f454b6a295cc78 | 20:50 |
| gold_nested | abc_00000003 | Spark slot 1 | `gold_nested/abc_00000003/run-20261031-1951-s1` | 39 841 / 23 774, C3D10 | 89e43fa36b0a6fc5 | 20:51 |
| gold_nested | abc_00000065 | Spark slot 2 | `gold_nested/abc_00000065/run-20261031-1951-s2` | 39 234 / 23 560, C3D10 | 6e6260157f8df193 | 20:51 |
| harness_hex | abc_00000065 | laptop | `harness_hex/abc_00000065/run-20261031-2053-s0` | 56 656 / 44 776, C3D8 | 9fe760a86bdcff8f | 20:53 |
| ace_sizing | abc_00000003 | Spark, queued | | 29 128 / 16 007, C3D10 | 2932c87d2c30c174 | |
| ace_sizing | abc_00000065 | Spark, queued | | 31 518 / 17 848, C3D10 | 7c77d5736f8bd452 | |
| netgen | abc_00000003 | Spark, queued | | 28 537 / 15 298, C3D10 | c40bf931866e5996 | |
| netgen | abc_00000065 | Spark, queued | | 26 395 / 14 751, C3D10 | 10526c8c4a361836 | |
| harness_expert | abc_00000003 | Spark, queued | | 36 425 / 19 453, C3D10 | 365a751d102164a5 | |
| harness_expert | abc_00000065 | Spark, queued | | 39 608 / 22 650, C3D10 | 303e92a735e919c7 | |

Mesh sources are the scratch outputs of each system's build. The harness_expert meshes were copied at 20:49 ET,
from reports dated 20:27 and 20:45. If that agent rebuilds them, the queued runs still use the copied files (check
the sha256). The queue fills in the queued run dirs and start times in the Spark's `queue/<system>__<part>.json`.

## Release policy change, 2026-10-31 ~21:45 ET: adaptive release, platen held while a backlog is released

Up to now each erosion block removed at most 8 failed cells (highest PEEQ first). A block that left failed cells
waiting still advanced the platen 0.002 mm. In 003 most removals of 8 (or 4) cells did not converge, and each
diverging try cost ~70 s of ccx cut-backs. So a 234-cell backlog was released at ~42 s per cell on 8 threads, and
the platen crept 0.002 mm per release block (0.016 mm per 20 min).

New policy (`--release adaptive`, the default; `_ccx_failure` `adaptive_release`):

- The cap starts at 64 and halves on non-convergence down to 1. It doubles after a block that converged at its
  first try; after a block that needed a smaller cap, the next block starts at the cap that converged.
- Removal tries fail fast: one cut-back. Every converged removal in the slow run needed at most one.
- The platen is held while failed cells wait (2e-5 mm per release block).
- Unchanged: the increment bounds (0.05 / 0.02 / 0.05 mm), the failure criterion and the stop rules.

Check on the 003 baseline (scratchpad `crush3/cmp003/`): all three runs resumed from the same checkpoint (step 161,
1.2197 mm, 202 failed cells, 279 waiting).

| failed cells | slow (cap 8, creep) | fixed cap 8, held | adaptive, held |
|---|---|---|---|
| 206 | 3308 N at 1.2202 mm | 3307 N at 1.2197 mm | 3307 N at 1.2197 mm |
| 210 | 3213 N at 1.2222 mm | 3209 N | 3209 N |
| 212 / 214 | 3006 N (214) at 1.2242 mm | 3146 N (212) | 3146 N (212), 3000 N (214) |
| 218 | 2911 N at 1.2262 mm | | 2888 N (-0.8 %) at 1.2198 mm |
| 222 / 220 | 2797 N (222) | | 2799 N (220) |
| 226 / 224 | 2741 N (226) at 1.2302 mm | | 2742 N (224) at 1.2198 mm |
| 234 / 236 | 2601 N (234) at 1.2336 mm | | 2459 N (236) at 1.2199 mm |

- At the same number of failed cells the force agrees within ~1 % (identical to the held fixed-cap run while
  their release orders coincide).
- At the same travel it does not: while a backlog is released, the slow schedule's force falls over a creep of
  0.002 mm per release block (+0.016 mm for these 30 cells), and the held schedules' force falls at 1.2197 mm.
- That travel offset is the slow schedule's creep: the crack speed depended on how many cells each block could
  remove. With the platen held, the crack runs at the travel where it starts, whatever the cap.

Pace in the crack phase:

| schedule | threads | time per failed cell |
|---|---|---|
| slow (cap 8, creep) | 8 | 42 s |
| fixed cap 8, held | 4 | 145 s |
| adaptive, held | 4 | 17 s |

Applied to every run below; the runs above that used the slow schedule are kept as they are.

### Runs with the adaptive release (all systems use it from here on)

| system | part | machine | run dir (under runs/nonlinear on its machine) | start (ET) | note |
|---|---|---|---|---|---|
| harness_auto | abc_00000003 | laptop, 8 thr | `abc_00000003/run-20261031-2139a` | 21:39 | `--resume run-20261031-2031c` (step 207, 1.240 mm, 245 failed) |
| harness_auto | abc_00000065 | laptop, 8 thr | `abc_00000065/run-20261031-2139a` | 21:39 | `--resume run-20261031-2031c` (step 167, 1.470 mm, 218 failed) |
| harness_hex | abc_00000003 | Spark slot 0 | `harness_hex/abc_00000003/run-20261031-2040-s0` | 21:40 | from zero |
| gold_nested | abc_00000003 | Spark slot 1 | `gold_nested/abc_00000003/run-20261031-2040-s1` | 21:40 | from zero |
| gold_nested | abc_00000065 | Spark slot 2 | `gold_nested/abc_00000065/run-20261031-2040-s2` | 21:40 | from zero |
| harness_hex | abc_00000065 | laptop, 4 thr | `harness_hex/abc_00000065/run-20261031-2141-s0` | 21:41 | from zero |
| ace_sizing, netgen, harness_expert | both | Spark, queued | | | same queue, same meshes |

These four slow-release runs were stopped at 21:40 and are kept as they are:

- Spark `harness_hex/abc_00000003/run-20261031-1950-s0`, at 1.44 mm.
- Spark `gold_nested/abc_00000003/run-20261031-1951-s1`, at 0.99 mm.
- Spark `gold_nested/abc_00000065/run-20261031-1951-s2`, at 1.34 mm.
- Laptop `harness_hex/abc_00000065/run-20261031-2053-s0`, which reached 1 increment.

Their queue state is in `queue/superseded-slow-release/`. The Spark clone now runs `origin/nonlinear-setup` (24987c2).

## Neck study: making abc_00000065 break where the printed part broke (2026-11-01)

The printed clamp (Alpha Fast, flat steel plates, press) lost one arm at the **neck**: the 8.1 mm ring wall
between bore and outer surface, at about the bore's mid-height. The other arm's tip was crushed. Every run
above separates both arm **tips** under the platen (hf 0.89, the 1.5 mm nut-pocket walls, in compression).
The variants below run on abc_00000065 with the same step policy and adaptive release. Options and the
method are in `docs/solver_ccx_nonlinear.md` ("Tension-only erosion, platen options, buckling imperfection").
Run dirs are under `runs/nonlinear/neck_study/` (laptop) and `~/feagent-runs/nonlinear/neck_study/` (Spark).

Labels: **M** = harness_neck mesh, **P** = frictionless platens (`--platen sliding`), **T** = tension-only
erosion (`--failure peeq_tension --compression-factor 0`), **I** = 0.1 mm buckling-mode imperfection
(`imperfection.py --platen sliding`, mode 1 at 2.07 kN). "hf" = height fraction of the failed cells' centroid
(0 = base, 1 = platen). The neck is the ring wall between bore (r 19.7 mm) and outer arc (r 27.8 mm); +8 deg is
y = +4 mm, above the bore centre. The photo's break is at about hf 0.45-0.6.

| variant | mesh (nodes) | platen | erosion | first failure: where (hf), travel, force | crack path | peak force (= predicted snap load) | travel at peak | travel at separation / end |
|---|---|---|---|---|---|---|---|---|
| baseline (earlier) | harness_auto (33,824) | bonded | PEEQ, any state | R tip, nut-pocket wall, compression (0.89), 0.94 mm | both tips (149 + 138 cells) | 3234 N | 1.16 mm | separated at 1.47 mm |
| harness_expert (earlier) | harness_expert (39,608) | bonded | PEEQ | R tip (0.89), 0.94 mm | both tips | 3260 N | 1.20 mm | separated at 1.40 mm |
| M | harness_neck (58,020) | bonded | PEEQ | R tip (0.89), 0.94 mm | both tips | 3190 N | 1.34 mm | separated at the tips at 1.48 mm (380 cells: R tip 200, L tip 180); 2.2 h wall |
| PT | harness_auto | frictionless | T | R tip inner face at the bolt hole, arm folding (0.90), 3.30 mm, 2.93 kN; tips slid 8 mm inward | at 50 % of peak (6.0 mm): tips 188 cells, R neck 75, L neck 5 | 2966 N | 2.55 mm | stopped at 9.26 mm, 6 % of peak (no separation: compressed hinges) |
| T | harness_auto | bonded | T | 1 nut-pocket cell (mixed shear, triaxiality ~0.05) (0.92) at 4.53 mm, then **both necks' outer fibre** within 0.1 mm: R at +8 deg (hf 0.62), L at -41 deg, bore side (hf 0.37) | at 50 % of peak (6.1 mm): L neck 107, R neck 101, bore bottom 33, tips 17 | 5916 N | 4.74 mm | stopped at 11.6 mm, 10 % of peak (no separation) |
| **MTI** (best match) | harness_neck + I (58,020) | bonded | T | **L neck outer fibre, +8 deg, y = +4.1 mm (hf 0.62)**, 4.49 mm, 5.74 kN; no tip cell before it | both necks across the outer half (L 347, R 333 cells at about +5 deg) at a fixed travel of 4.59 mm, while the force falls from 5.75 to 3.1 kN (unstable crack); then the right tip's pocket (50) and the bore bottom (40) | **5745 N** | 4.53 mm | never separated. The bore-side ligament of each neck stays in compression (triaxiality -0.5) and hinges. `-MTI-c5` continued from 4.84 mm with compressive erosion at 5 x 0.043 and stopped at 11.94 mm on force < 1 % of peak over 3 increments. It ended with 1455 failed cells (L neck 469, R neck 433, tips 366, bore bottom and notches 187) and 14 detached; c5 alone ran 3.7 h |
| MPTI | harness_neck + I | frictionless | T | none by 1.67 mm (stopped: P superseded, see below) | | 2795 N (still rising) | | stopped at 1.67 mm |
| MFTI | harness_neck + I | friction mu 0.3 (contact) | T | contact chatter, ~4 increments in 4 min (stopped) | | | | |

Run dirs:

- Laptop: `runs/nonlinear/neck_study/harness_auto/abc_00000065/run-20261101-{PT2,PT3,T2,T3}`. Each X3 continues
  its X2 with the damage-based step bounds; the X2 runs are the same model.
- Laptop: `runs/nonlinear/neck_study/harness_neck/abc_00000065/run-20261031-M` is a copy from the Spark, with
  the mesh and restart paths rewritten. `run-20261101-M-r` continues it.
- Spark `~/feagent-runs/nonlinear/neck_study/`:
  - `harness_neck_imp/abc_00000065/run-20261101-0036-MTI`, continued (8 threads) by `-MTI-r8` at step 191
    (4.595 mm). `-MTI-r8` is continued by `-MTI-c5` at step 236 (4.836 mm, `--compression-factor 5`, so the
    compressed ligaments, crushed to PEEQ 0.2-0.3, can break).
  - Laptop copies of the three, without `ccx_work`, are in
    `runs/nonlinear/neck_study/harness_neck_imp/abc_00000065/`.
  - `run-20261031-2350-MPTI` and `-MPTI-r`.
  - `run-20261101-0008-MFTI` and the contact-tuning runs in `ctune/`.
  - `arm_ratio_auto_bonded.json`.
- `superseded/`: PT, T, Tc0, PTc0 and MPTI. These ran before tensile PEEQ was accumulated per increment, so
  crushed cells that turned slightly tensile counted their compressive PEEQ.

Meshes: `harness_neck` nonlinear (sha256 123511d697acf449), and the same mesh moved by the sliding mode 1
(`imperfection.json` next to it; max move 0.1 mm at the left tip, both tips inward by 0.07-0.09 mm).

Other checks:

- Bonded tips, J2 + NLGEOM, no erosion (`arm_ratio`): the lateral / normal force per arm tip is 0.061 at
  0.1 mm, 0.08 at 1 mm, 0.12 at 3 mm and 0.17 at 5 mm. With mu 0.2-0.4 the tips stick, so bonded is the
  friction-consistent platen.
- Linear *BUCKLE (harness_neck), 1 kN reference:
  - Frictionless tips: 2.07, 2.42, 4.54, 5.37 kN. Mode 1 is one arm swaying inward.
  - Stuck tips: 13.4, 17.0 kN. Mode 1 is local, at the tip face / nut-pocket walls.

Videos and stills (scratchpad of the session that ran the study, not committed):

- `study/videos/crush/harness_neck_imp_abc_00000065_neckbreak_crush.mp4`: MTI to 4.84 mm, 16 s, original
  3-panel look.
- `physical_photo/abc_00000065_photo_vs_sim_neckbreak.png`: the photo next to MTI at 4.84 mm.

Reading the table:

- The **erosion criterion moves the crack**. The mesh does not (M = baseline). Frictionless platens do not
  either: they move it to the tips' inner face.
- Tension-only erosion with stuck tips puts the first crack at the neck's outer fibre, y = +4 mm. Stuck tips
  are bonded, which is Coulomb friction with mu >= 0.18. With the imperfection, the neck is the very first
  failure.
- The press loaded the part in force control, so the arm snaps at the **peak force: predicted snap load
  5.75 kN**, against 3.2 kN for the old model's tip crush. The simulated crack runs through the outer half of
  both necks at a fixed travel, which is the unstable snap in displacement control. Under force control one arm
  goes first and unloads the part. In the run both necks start in the same increment (4.49 mm: 2 left-neck
  cells, 1 right-neck cell, 1 tip cell), and the right neck's crack grows a little faster.

## Neck recipe on matched node counts (40,000 +- 2 %), every mesh system (2026-11-01)

The user's follow-up: "same mesh count between them but different refinements by different agents".
`experiments/mesh_systems/match_nodes.py` rebuilds each system's p65 nonlinear mesh with its own build.py and
refinement logic, and tunes only its size knob to a target node count (the count the solver sees, so equal
DOFs). The tries are in `match.json`. New dirs: `MESHSYS_OUT/<system>/p65_nonlinear_n40k`; the old meshes are
untouched. No neck-recipe run had been started on the unmatched study meshes, so none is superseded.
`neck_tip_sizes.py` gives the median / min corner-edge length of the cells in the neck wall (ring between bore
and outer arc, -15 < y < 12 mm) and in the tip pocket block (y > 19.5 mm). `neck_montage.py` draws all of them
(full view and a zoom on the right neck).

| system | knob (value) | nodes | cells | neck median (min) edge, cells across 8.1 mm | tip median (min) edge | gates not passing |
|---|---|---|---|---|---|---|
| harness_auto | `--global` 3.05 mm | 40,078 | 22,832 C3D10 | 2.83 (0.81) mm, 2.9 | 1.73 (0.54) mm | thin_walls warn |
| harness_expert | `--global` 12 mm (its default) | 39,608 | 22,650 C3D10 | 3.19 (0.86) mm, 2.5 | 1.65 (0.53) mm | none |
| ace_sizing | `--h` 3.5 mm, `--node-cap` 40,800 (its loop settles at h 2.38 mm) | 40,095 | 23,081 C3D10 | 2.88 (1.58) mm, 2.8 | 1.87 (0.77) mm | SICN fail (0.025), thin_walls warn |
| gold_nested | `--h0-frac` 0.324 (its own fit to 40k) | 39,234 | 23,560 C3D10 | 3.50 (1.27) mm, 2.3 | 1.79 (0.52) mm | SICN fail (0.016), gamma warn |
| netgen | `--maxh` 2.10 mm | 39,706 | 22,817 C3D10 | 2.78 (1.51) mm, 2.9 | 1.44 (0.47) mm | thin_walls warn |
| harness_neck | `--neck-size` 3.83 mm | 40,538 | 23,197 C3D10 | 3.13 (0.97) mm, 2.6 | 1.67 (0.56) mm | none |
| harness_hex (**not matched**) | `--thin-factor` 12.9 (h 4 mm chosen by its own search) | 54,408 (+36 %) | 42,940 C3D8 | 1.08 (0.29) mm, 7.5 | 0.57 (0.10) mm | none (gamma, thin_wall n/a) |

- **harness_hex** has no valid hex mesh near 40k. Its own search keeps the largest base size whose hex split
  has min scaled Jacobian >= 0.1. That gives >= 47.7k nodes, and those fail SICN; no hex mesh passing every
  gate is below 54.4k. With `--h` fixed, or a thin factor >= 1.45, it falls back to tets. It runs at the
  closest gate-passing count and is labelled unmatched. Its edges are not comparable 1:1: a C3D8 has nodes
  only at the corners, while C3D10 node spacing is half its edge.
- **harness_neck** at equal cost: harness_expert alone is 39.6k. Any neck refinement finer than ~3.5 mm costs
  >= 41.1k nodes (the ring and bore faces), so its neck size ends at 3.83 mm, about the expert's own. The 58k
  MTI run above stays as the reference for a fine neck (1.6 mm).
- Mode-1 buckling loads of the sliding imperfection: 2.07-2.09 kN on the tet meshes, 2.17 kN on hex.

Recipe, identical for every mesh:

    imperfection.py --platen sliding --amp-mm 0.1          # its own mode 1
    crush.py --platen bonded --failure peeq_tension --compression-factor 0 --compression-switch 0.5:5

The **switch rule** is built into the driver (`compression_switch`, the same for every mesh): at the first block
end after the first failed cell where the force is below 50 % of the peak so far, the compression factor
becomes 5. From then on, cells crushed beyond 5 x 0.043 PEEQ can fail too, so the compressed hinge behind the
tensile crack can break. Same step policy and adaptive release as before.

Compute: Spark (4 runs x 5 threads): harness_auto, harness_expert, ace_sizing, gold_nested,
`~/feagent-runs/nonlinear/neck_study/<system>_n40k_imp/abc_00000065/run-20261101-0834-n40k`. Laptop (3 x 4):
netgen `run-20261101-0935-n40k`, harness_hex and harness_neck `run-20261101-0942-n40k`, under
`runs/nonlinear/neck_study/<system>_n40k_imp/abc_00000065/`.

### Results

hf is the height fraction (0 = base, 1 = platen); "neck" is the ring wall. The crack is shown at the first
increment after the peak with force < 50 % of the peak.

| system | nodes | neck / tip median edge | first failed cell: where (hf), travel, force | peak force (snap load) | travel at peak | crack at 50 % of peak (travel; failed cells) | end state | wall |
|---|---|---|---|---|---|---|---|---|
| harness_auto | 40,078 | 2.83 / 1.73 mm | R tip pocket wall (0.88), 4.31 mm, 5.79 kN; necks follow | 5825 N | 4.62 mm | 5.14 mm: L neck 167, R neck 140, base 9, tip 4 | force < 1 % of peak at 14.99 mm; not separated | 3.6 h (Spark, 5 thr) |
| harness_expert | 39,608 | 3.19 / 1.65 mm | **R neck outer fibre (0.62, y +4.0 mm)**, 4.62 mm, 5.81 kN | 5815 N | 4.72 mm | 5.95 mm: R neck 96, L neck 83, bore bottom 43, tip 3 | force < 1 % at 19.46 mm; not separated | 4.9 h (Spark) |
| ace_sizing | 40,095 | 2.88 / 1.87 mm | R tip pocket wall (0.91), 4.19 mm, 5.74 kN; necks follow | 5808 N | 4.74 mm | 5.06 mm: L neck 165, R neck 131, tip 5, base 4 | force < 1 % at 18.90 mm; not separated | 4.9 h (Spark) |
| gold_nested | 39,234 | 3.50 / 1.79 mm | **R neck (0.63, y +4.9 mm)**, 4.99 mm, 5.85 kN | 5855 N | 4.97 mm | 5.86 mm: R neck 103, L neck 86, base 15, tips 3 | force < 1 % at 16.93 mm; not separated | 4.4 h (Spark) |
| netgen | 39,706 | 2.78 / 1.44 mm | **L neck (0.61, y +3.8 mm)**, 4.80 mm, 5.86 kN | 5861 N | 4.78 mm | 4.99 mm: L neck 165, R neck 144, base 4 | force < 1 % at 16.22 mm; not separated | 5.6 h (laptop, 4 thr) |
| harness_neck | 40,538 | 3.13 / 1.67 mm | R tip pocket wall (0.92), 4.50 mm, 5.77 kN; necks follow | 5787 N | 4.69 mm | 5.77 mm: L neck 136, R neck 123, base 11, tips 9 | **stopped by hand** at 11.77 mm: force flat at 169 N (2.9 % of peak) since ~10 mm, held by 7 cells whose removal never converges | 5.8 h (laptop) |
| harness_hex (unmatched, C3D8) | 54,408 | 1.08 / 0.57 mm | R tip at the bolt hole (0.90), 2.14 mm, 4.95 kN | 5995 N | 4.38 mm | 5.22 mm: **R tip 382, L neck 189**, L tip 19 | force < 1 % at 11.65 mm; not separated | 5.8 h (laptop) |
| reference: harness_neck 58k MTI | 58,020 | 2.0 / 1.6 mm | L neck (0.62, y +4.1 mm), 4.49 mm, 5.74 kN | 5745 N | 4.53 mm | 4.60 mm: both necks | force < 1 % at 11.94 mm (after the manual cf 5 resume); not separated | 6.5 h |

**Spread across the six tet systems at ~40k nodes:**

- Peak (snap load): 5787-5861 N, mean 5825 N, range 75 N (1.3 %), sd 28 N.
- Travel at peak: 4.62-4.97 mm.

With harness_hex (54k, C3D8): peak 5787-5995 N (range 3.6 %), travel at peak 4.38-4.97 mm.

**Where it breaks:**

- **Every system cracks through both necks.** In the tet meshes the necks hold 80-99 % of the failed cells at
  50 % of peak. Where the neck fails first, it is at y +3.8 to +4.9 mm (hf 0.61-0.63), the photo's break.
- Three systems (harness_expert, gold_nested, netgen) fail first in the neck.
- The other four (harness_auto, ace_sizing, harness_neck, harness_hex) first lose a single cell in a tip's
  nut-pocket wall (mixed shear, triaxiality ~0.05). That cell is not the start of the break; the neck cells
  follow within 0.1-0.5 mm (hex: 2.2 mm).
- harness_hex is the odd one. Its C3D8 cells are finer and stiffer, so its tips start cracking early
  (2.14 mm), and its peak is 3 % higher. At 50 % of peak its R arm is broken at the tip and its L arm at the
  neck. That is the photo's pattern: one arm off at the neck, the other's tip crushed.
- No run separates topologically. After the switch to compressive erosion, the force falls to < 1 % of peak
  (harness_neck to 2.9 %), while a few crushed tip cells keep the fragments face-connected.

**At equal node count the refinement choice barely matters for this question.** The neck edge sizes differ
(2.8-3.5 mm) and so do the tip sizes (1.4-1.9 mm), yet the snap load varies by 1.3 % and the crack location not
at all. The 58k neck-refined mesh (1.6 mm at the neck) gives 5745 N, -1.4 % against the 40k mean.

Videos are in the scratchpad of the session that ran the study, under `study/videos/crush_n40k/`; not committed.

- Per system, `<system>_abc_00000065_crush.mp4`: the original 3-panel look, 25-35 s (one frame per increment,
  at least 16 s).
- Per system, `<system>_abc_00000065_crush_rtx.mp4`: Omniverse RTX, rendered on the Spark in `feagent-crush`
  (24 GB cap), one at a time, 25-35 s, 11-16 min each. Copies stay on the Spark in
  `~/feagent-viz/videos/n40k/`. Their caption bar is assemble_crush.py's fixed text, which still says "a cell
  breaks at PEEQ 0.043"; the erosion rule is the tension-only one above.
- Per system, `closeup/<system>_neck_closeup.mp4`: the neck crack forming, 8 s (`neck_closeup.py`).
- `all_systems_abc_00000065_crush{,_rtx}.mp4`: 3x2 grids of the six study systems, 36 s
  (`crush_grid.sh ... 36`; shorter tiles hold their last frame).
- `neck_closeups_7_systems.mp4`: the 7 close-ups in a row, each after a title card.
- harness_neck is its own (7th) video, not in the grids.

Still: `physical_photo/abc_00000065_photo_vs_7_matched_meshes.png` (`break_stills.py`) shows the photo next to
all 7, each at its first increment below 50 % of peak.
