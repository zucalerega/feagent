# Neural surrogate (`backend="nn"`): optional, fast, unverified

A graph neural network that predicts 2D linear-elastic fields directly on the case mesh. In the
ScienceClaw loop it is **one more opinion, never the reference**: the fixed solver (skfem) and the
verification layer judge it like any other submission, and mesh refinement / Richardson
extrapolation does not apply to it. Its value for the demo is showing **where it is accurate (in
distribution) and where it gets caught (out of distribution)**.

```python
from feacore.solver import solve
res = solve(case, backend="nn", checkpoint="models/surrogate/cpu/best.pt")   # or FEACORE_NN_CHECKPOINT
res.evaluate_all(); res.nodal("von_mises"); res.probe("ux", [[0.1, 0.0375]])
res.info.warnings   # always "surrogate: ... UNVERIFIED ..."; plus "surrogate OOD: ..." when out of distribution
res.ood             # True when any out-of-distribution check fired
```

## What was built

| piece | file |
|---|---|
| training family `SurrogatePlate` (Problem API, **not** registered): plate W x H with a circular hole or an edge notch; BC templates `clamped` (left fixed, angled traction on the right) and `biaxial` (rollers + biaxial traction); in-distribution ranges + 8 out-of-distribution categories | `feacore/surrogate/family.py` |
| case + mesh -> graph and nondimensional features (one code path for data, training and the backend) | `feacore/surrogate/graph.py`, `mesh.py` |
| dataset generation, parallel over processes, FEM = `solve(case, "skfem")`, order 2 | `scripts/make_surrogate_dataset.py`, `feacore/surrogate/data.py` |
| MeshGraphNet-style GNN in plain PyTorch (+ optional PhysicsNeMo wrapper) | `feacore/surrogate/model.py` |
| training (normalisation, train/val/test split, checkpoints, latent OOD calibration) | `scripts/train_surrogate.py`, `feacore/surrogate/train.py` |
| backend: checkpoint loading, prediction, probe / nodal / evaluate, OOD warnings, VTU | `feacore/solver/nn_backend.py`, `feacore/surrogate/interp.py`, `ood.py` |
| evaluation: errors in / out of distribution, timings, benchmark check, plots | `feacore/surrogate/evaluate.py` |
| tests (12, ~5 s, no GPU, no pre-trained checkpoint) | `tests/test_surrogate.py` |

### Graph and features

* **Nodes** = all mesh nodes (tri6: corners + midside nodes). **Edges** = the linear
  sub-triangulation (each tri6 is split into 4 triangles), both directions. Edge features:
  relative position and length, divided by L.
* **Node features (21)**: normalised coordinates; BC category one-hot (fixed / loaded / free
  boundary / interior) plus per-component constraint flags; normalised traction vector; outward
  boundary normal; nu; distances to the nearest fixed, loaded and free-boundary node; and five
  graph-level values broadcast to every node: resultant load force (x, y) and moment, and the
  bounding-box extents. The broadcast globals mattered a lot: without them the first CPU run was
  still at the "predict the mean" loss after 3 epochs (val 0.90); with them, val was 0.43 after one
  epoch.
* **Nondimensionalisation (exact for linear elasticity)**: L = largest bounding-box extent,
  t0 = largest applied traction; inputs use x/L and t/t0, targets are s/t0 and u E/(t0 L). Scaling
  the geometry, E or the load leaves the network inputs unchanged and the outputs scale exactly
  (tested against the real solver: `test_fem_targets_scale_exactly`). Plane strain maps exactly
  to plane stress with E' = E/(1-nu^2), nu' = nu/(1-nu).
* **Targets**: nodal ux, uy, sxx, syy, sxy, von Mises from the skfem P2 solution (L2-recovered
  stresses), on the same nodes. Other fields (u_mag, principal, Tresca, szz) are derived.

### Model: own MeshGraphNet-style GNN (PhysicsNeMo not used)

Encoder MLPs (nodes, edges) -> 8 residual message-passing blocks (edge MLP on [edge, sender,
receiver] -> sum aggregation with `index_add_` -> node MLP on [node, aggregate, graph mean]; every
MLP ends in LayerNorm) -> MLP decoder; hidden 64, 281k parameters for the CPU run. The per-graph
mean ("global context") gives each block a global receptive field, which an elliptic problem needs.

PhysicsNeMo: `nvidia-physicsnemo` 2.2.2 installs from PyPI but pulls in about 50 packages
(torchvision, timm, warp-lang, onnx, s3fs, hydra, zarr, ...). Its `MeshGraphNet` needs
`torch_geometric` + the compiled `torch_scatter`, which has no PyPI wheel for this torch / Python 3.13
(the PyG wheel host is not reachable from the dev container). So the default is the plain-PyTorch
model. `--model-type physicsnemo` wraps PhysicsNeMo's MeshGraphNet behind the same interface for a
GPU box where `pip install nvidia-physicsnemo torch_geometric torch_scatter -f
https://data.pyg.org/whl/torch-<ver>+<cuda>.html` works. **This path is untested.** No usable
pretrained linear-elasticity checkpoint was found: NVIDIA Apollo has no confirmed
structural-mechanics release, and the PhysicsNeMo structural example is the hyperelastic deforming
plate.

### Out-of-distribution flags

Two signals, both calibrated on the training split and stored in the checkpoint:

1. **Ranges**: continuous node features, edge lengths and 12 graph descriptors (extents, solid
   fraction, number of boundary loops, hole perimeter, minimum ligament, mean edge length, node
   count, load angle, nu, fixed / loaded boundary fractions) outside the training range. These
   say *which* input is unusual.
2. **Latent distance**: Mahalanobis distance of the graph-mean latent (last message-passing block)
   from the training graphs. The threshold is the larger of the training graphs' 99.5th percentile
   and the validation graphs' 99th percentile.

A flag means "not covered by the training data". **No flag does not mean the answer is right.**

RESULTS_PLACEHOLDER

## Reproduce

CPU proof of concept (dev container, 4 shared cores; dataset generation takes about 25 s):

```bash
export OMP_NUM_THREADS=1
python scripts/make_surrogate_dataset.py --out /tmp/surrogate/ds1 --split id  --n 600 --workers 3
python scripts/make_surrogate_dataset.py --out /tmp/surrogate/ds1 --split ood --n 12  --workers 3
python scripts/train_surrogate.py --data /tmp/surrogate/ds1 --out /tmp/surrogate/runs/cpu2 --device cpu --threads 2 \
    --epochs 40 --max-minutes 25 --hidden 64 --layers 8 --batch-size 2
# second cosine cycle, warm-started (the loss was still falling when the first schedule ended)
python scripts/train_surrogate.py --data /tmp/surrogate/ds1 --out models/surrogate/cpu --device cpu --threads 2 \
    --epochs 40 --max-minutes 20 --hidden 64 --layers 8 --batch-size 2 --lr 5e-4 --init /tmp/surrogate/runs/cpu2/best.pt
python -m feacore.surrogate.evaluate --ckpt models/surrogate/cpu/best.pt --data /tmp/surrogate/ds1 \
    --out models/surrogate/cpu/eval --img docs/img/surrogate --scaling --benchmarks
```

Feature changes do not need new FEM solves: `make_surrogate_dataset.py --out DIR --refeaturize`
rebuilds the features of every sample from its stored case + mesh.

### Larger GPU run (Brev)

On an x86_64 Brev GPU box, from the repo root (`scripts/setup_gpu_box.sh` installs system libs and
Python deps; plain `pip install torch` brings the CUDA wheel). Dataset generation is CPU / FEM-bound
and parallel; mixing mesh sizes makes the network less sensitive to how fine an agent meshes:

```bash
bash scripts/setup_gpu_box.sh && pip install torch && export OMP_NUM_THREADS=1
D=/data/surrogate/ds_gpu
python scripts/make_surrogate_dataset.py --out $D --split id --n 4000 --h-rel 0.06 --start 0     --workers $(nproc)
python scripts/make_surrogate_dataset.py --out $D --split id --n 4000 --h-rel 0.04 --start 4000  --workers $(nproc)
python scripts/make_surrogate_dataset.py --out $D --split id --n 4000 --h-rel 0.08 --start 8000  --workers $(nproc)
python scripts/make_surrogate_dataset.py --out $D --split ood --n 50 --workers $(nproc)
# the training step (one command):
python scripts/train_surrogate.py --data $D --out models/surrogate/gpu --device cuda \
    --epochs 200 --batch-size 16 --hidden 128 --layers 15 --lr 5e-4 --amp --max-minutes 120
python -m feacore.surrogate.evaluate --ckpt models/surrogate/gpu/best.pt --data $D \
    --out models/surrogate/gpu/eval --img docs/img/surrogate --device cuda --scaling --benchmarks
export FEACORE_NN_CHECKPOINT=$PWD/models/surrogate/gpu/best.pt
```

`--max-minutes` drives the cosine learning-rate schedule by wall time, so a time-limited run still
anneals. Training holds every graph in memory (about 0.25 MB per 1000-node sample, so 12k samples
is about 3 GB). Checkpoints are plain dicts loadable with `torch.load(weights_only=True)`.
`models/` has its own `.gitignore`; datasets stay outside the repo. DGX Spark (aarch64) can train
too (torch has aarch64 wheels), but generating data there needs gmsh from conda-forge (see
CLAUDE.md).

## Limitations

* **One training family.** It covers a rectangular plate with one interior hole, left edge clamped
  and a uniform traction on the right edge. Everything else (notches, other BC layouts, other
  shapes, all the registered benchmarks) is out of distribution, and the errors there are large.
  The checks flag these cases, but that is a heuristic, not a guarantee.
* **Accuracy is limited by the CPU budget.** The proof-of-concept network trained for about 25
  minutes on roughly one contended core. Expect the GPU run above to do much better in
  distribution, but measure it; no GPU numbers are claimed here.
* **No speed-up for these 2D cases on CPU.** The fixed solver needs about 60 ms for a
  1000-node P2 case. The surrogate is linear in the node count but has a larger constant on CPU.
  Speed-ups need a GPU and batched inference (many variants per forward pass), or much larger
  meshes; this has not been measured yet.
* **No error estimate.** The surrogate has no discretisation-error structure, so GCI / observed
  order are meaningless for it. Its own error vs the fixed solver is only known on the test sets
  above.
* **Supported inputs**: 2D plane stress / plane strain, one homogeneous material, tri3 / tri6 meshes,
  BCs fixed / zero displacement / symmetry, edge loads traction / pressure / force, and QoIs
  point / max / min / max_abs. Anything else raises `UnsupportedCase` (a `NotImplementedError`):
  3D, axisymmetric, heat, thermal or body loads, nonzero prescribed displacements, point loads,
  several materials, or the reaction / integral / mean / strain-energy QoIs.
* Stresses near the clamped corners are singular in the FEM targets (mesh-dependent). The surrogate
  learns the value at the training mesh size, so max von Mises there is not a physical quantity
  (the verification layer's divergence flags apply to the FEM itself).
