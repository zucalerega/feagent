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
