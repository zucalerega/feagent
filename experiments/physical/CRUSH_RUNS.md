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
