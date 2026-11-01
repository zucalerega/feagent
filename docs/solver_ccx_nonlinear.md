# Elastoplastic, load-stepped analysis with CalculiX (opt-in)

`backend="ccx"` can run a nonlinear static analysis: J2 (von Mises) plasticity with isotropic
hardening, a fixed-increment load ramp and, on request, geometric nonlinearity (NLGEOM). It is
**opt-in** and owned by the solver owner to adopt. The fixed solver (skfem) and linear cases are
unchanged: a linear case writes the same deck and serialises to the same JSON as before. skfem
refuses a nonlinear case with `NotImplementedError`.

Files: `feacore/case.py` (schema + validation), `feacore/solver/ccx_backend.py` (deck),
`feacore/solver/_ccx_nl.py` (run, per-increment parsing, `CcxNonlinearResult`, `.vtu` series),
`feacore/solver/_ccx_io.py` (streaming per-increment `.frd` / `.dat` / `.sta` parsers),
`scripts/ccx_plastic_ramp.py` (load ramp of a gold part, linear comparison). Tests:
`tests/test_solver_ccx_nonlinear.py` (8 tests, about 15 s; the ccx ones skip without ccx).

## Case format (backward compatible)

```json
"materials": {"steel": {"E": 210e9, "nu": 0.3,
  "plasticity": {"model": "j2_isotropic", "yield_stress": 355e6,
                 "hardening": [[0.0, 355e6], [1.0, 1.866e9]]}}},
"analysis": {"physics": "elasticity", "formulation": "3d", "order": 2,
  "options": {"nonlinear": true, "increments": 20, "load_factor_max": 1.0, "nlgeom": false}}
```

* `hardening`: flow stress vs equivalent plastic strain, piecewise linear, starting at
  `[0, yield_stress]` (prepended if missing), constant beyond the last point. Omit it for
  elastic-perfectly plastic. Validation rejects other models, `yield_stress <= 0`, decreasing
  strains, a first stress other than `yield_stress`, and softening.
* `increments` (default 20): fixed, equal increments over the ramp (`*STATIC, DIRECT`).
  `adaptive: true` lets ccx cut an increment back (max increment = 1 / increments).
* `load_factor_max` (default 1): every load and prescribed displacement is multiplied by it at
  the end of the ramp. Loads ramp linearly from 0 (thermal loads from `T_ref`).
* `nlgeom` (default false): `*STEP, NLGEOM`. Without it ccx runs geometrically linear (the
  stdout says only "Nonlinear material laws are taken into account").
* A material with `plasticity` requires `nonlinear: true`; `nonlinear` is elasticity-only.

## Deck

`*PLASTIC` (stress, plastic strain) after `*ELASTIC`; `*STEP, INC=max(100, 20 n)[, NLGEOM]`;
`*STATIC, DIRECT` with `1/n, 1.`, so step time t = load factor / `load_factor_max`;
`*NODE FILE U, RF` and `*EL FILE S, PEEQ` at every increment; `*EL PRINT PEEQ` (integration
points) and the reaction / energy totals in the `.dat` at every increment.

## Result

`solve(case, backend="ccx")` returns a `CcxNonlinearResult` (a `CcxResult`) positioned on the
last converged increment, so every QoI of the case works there. Also:

* `increments`: one `Increment` per output increment (`lam`, `U`, `S`, `RF`, nodal `PEEQ`,
  integration-point `peeq_ip`, `.dat` totals, Newton `iterations` from the `.sta`);
* `use_increment(i)`: evaluate fields / QoIs / reactions at increment i;
* `increment_qois(i)`: max von Mises (nodal), max PEEQ (nodal and integration point), plastic
  volume fraction (integration points with PEEQ > 1e-9, × cell measure incl. thickness / 2πr),
  reactions per Dirichlet group, total reaction vs applied load and `imbalance_rel`;
* `write_series(dir, extra=..., point_data=...)`: `inc_000.vtu` (λ = 0) .. `inc_NNN.vtu`
  (displacement, PEEQ, plastic flag, von Mises, stress components, s1, s3; cell PEEQ_ip_max and
  plastic_ip_fraction), `inc.pvd` (timestep = λ) and `summary.json` (per-increment QoIs);
* `converged` / `errors`: when a fixed-increment run diverges ("solution seems to diverge"), the
  converged increments are kept, `converged` is false and a warning names the λ reached. The
  workdir is kept in that case.
* `strain_energy` is CalculiX's total internal energy (elastic energy + plastic work).

Units and conventions as in `docs/solver_ccx.md`. Two ccx facts found here: the `.frd` RF of CAX
elements is already the full-360° value (the `.dat` totals are the 2° wedge, ×180), and the
`.sta` attempt column reads `1U` for a cut-back attempt.

## Verification (`tests/test_solver_ccx_nonlinear.py`, ccx 2.23)

| case | check | result |
|---|---|---|
| uniaxial bar C3D10, E 210 GPa, σy 355 MPa, displacement control to 4 εy, perfectly plastic and H = 2 GPa | σxx = min(Eε, σy + H εp), εp = (ε − σy/E)/(1 + H/E), reaction = −σA, every increment | stress 1e-5 (nodal output digits), integration-point PEEQ 1e-6 · εy, reaction 1e-5, plastic volume fraction 0 / 1 |
| same, force control to 1.2 σy A with H = 2 GPa | reaction = −λF, applied = λF, εp = (σ − σy)/H, tip displacement | 1e-6 / 1e-9 / 1e-6 / 1e-5 |
| same, perfectly plastic, force 1.5 σy A (beyond the limit load) | diverges after λ = 1/1.5 | stops at λ = 0.667, converged increments kept, flagged |
| NLGEOM, 0.7 % strain, H = 2 GPa | within 1 % of small strain; series files | passes |
| thick sphere a = 1, b = 2, internal pressure, perfectly plastic, CAX6 (2229 cells), 20 increments to c = 1.6 | elastic limit p_e = 2σy/3 (1 − a³/b³) | −0.20 % (nodal vM at the bore) |
|  | plastic radius c(p) from 2σy ln(c/a) + 2σy/3 (1 − c³/b³) = p (Hill), measured from the elastic zone (vM = σy c³/r³ at r = b) | max error 1.7e-4 over 11 plastic increments |
|  | c from the plastic volume (integration points, V = 2π/3 (c³ − a³)) | max error 1.3e-3 |
|  | axial reaction = −p π a², total imbalance | 3e-7, 6e-7 |

The linear ccx tests (`tests/test_solver_ccx.py`, 42) still pass with ccx 2.23.
`tests/test_ccx_io.py::test_eigen_and_buckling_tables` fails with 2.23 on a line count of the
2.21 reference `.dat` (65 vs 58 lines); that is a ccx version difference in feabench, not this
change.

Nodal von Mises (extrapolated and averaged) overshoots σy by up to ~1.5 % in a perfectly
plastic zone; the integration-point values are capped. Use `max_peeq_ip` /
`plastic_volume_fraction` (integration points) for plasticity measures.

## Installing ccx without sudo

conda-forge `calculix` (2.23, spooles solver, OpenMP) in a micromamba env:

```bash
curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj -C ~/micromamba-local bin/micromamba
MAMBA_ROOT_PREFIX=~/micromamba-local ~/micromamba-local/bin/micromamba create -y -p ~/micromamba-local/envs/ccx -c conda-forge calculix
export PATH=~/micromamba-local/envs/ccx/bin:$PATH
```

On x86-64: env 86 MB, package cache 217 MB. conda-forge also has calculix 2.23 for linux-aarch64 (DGX Spark; the package itself is 2.7 MB; not installed there in this work). The build has spooles only (no PARDISO / PaStiX), so a 390k-dof
C3D10 model takes about 30 s per Newton iteration on 8 threads and ~3 GB.

## Progressive failure by element erosion (opt-in)

```json
"options": {"nonlinear": true, "nlgeom": true, "increments": 100,
            "failure": {"criterion": "peeq", "limit": 0.043, "aggregate": "mean",
                        "near_increments": 250, "fine_increments": 500,
                        "path_groups": ["top", "bottom"], "stop_force_ratio": 0.01}}
```

`feacore/solver/_ccx_failure.py`. CalculiX has no criterion-driven element deletion, so the ramp
runs as a chain of **restart blocks**, one ccx static step each:

* block 0 = model + step + `*RESTART, WRITE`; block k = `*RESTART, READ` (the previous `.rout`)
  + a step that starts with `*MODEL CHANGE, TYPE=ELEMENT, REMOVE` (the cells that failed at the
  end of block k-1) + `*RESTART, WRITE`. Dirichlet values are written as totals at the block end;
  ccx ramps them from the previous step's values. The restart carries the whole state (stresses,
  plastic strains, NLGEOM configuration), so a chain of blocks gives the same answer as one step.
  Verified: a bar split into two restart runs reproduces the single run's forces to all printed
  digits, and `test_failure_restart_blocks_are_path_correct` (8 blocks) matches the closed-form
  hardening answer to 2e-5.
* Block length: automatic incrementation bounded by dt (`*STATIC` dt, dt, dt/1000, dt). dt is
  chosen from the last block's growth rate of the max cell PEEQ so that it grows by about
  `peeq_step` × limit (0.2), clipped to [1/`fine_increments`, 1/`increments`] of the ramp, and
  at most 1/`near_increments` once a cell is within `refine_at` (0.5) × limit or has failed.
* After a block: cells with PEEQ (mean over the cell's integration points; `aggregate: "max"`
  for the max) ≥ limit fail. They are removed at the start of the next block, at most
  `max_remove` (8) per block, highest PEEQ first. The rest are **deferred**, and the blocks that
  remove them advance the travel by only 1/10 of the fine increment, so the crack runs at nearly
  constant travel. If a block does not converge it is retried at dt/4, then with the cap halved
  down to one cell. Removing many stressed cells at once makes Newton diverge (residual ~1e6 %,
  in the 003 crush: 29 cells at once diverge at any increment size, 4 converge, and the same step
  without removal converges in 4 iterations). Then, repeatedly, cells at **hinge contacts** (a node shared by two
  face-connected fragments: the smaller fragments' cells at that node) are removed, and so are
  fragments with fewer than 3 constrained nodes (unsupported). Without this, cells hanging on a
  node or an edge are mechanisms: Newton diverges (2000 % residual at any increment size), and
  those contacts still carry load, so a face-separated part kept 16 % of its peak force.
* Stop: no supported fragment joins all `path_groups` (default: the Dirichlet groups); the block
  that applies the cut is still solved, so the last increment shows the residual force (0 in
  the notched-bar test). Also stops when the force on the driven groups drops below
  `stop_force_ratio` × peak after the first failure, or at the end of the ramp. A block that
  fails every retry stops the run (`converged = False`, `stop_reason` names it).
* Result: `history` (one record per increment: λ, reaction on the driven groups, n_failed,
  n_detached, max PEEQ, iterations, block), `state` per cell (0 intact, 1 failed, 2 removed as a
  hinge / unsupported fragment), `removed_at`, `stop_reason`, `separated`. Solver option
  `series_dir` writes `result_NNNN.vtu` per increment while running (point: displacement (frozen
  at its last value on nodes left without intact cells), von_mises, PEEQ; cell: PEEQ, failed,
  state; failed = state > 0; cells removed at the start of a block are flagged from that block's
  first increment on). `on_increment(record)` streams the records.
  Each block's `.frd` / `.dat` is deleted once parsed (`keep_files` keeps them).

* Step policy options (used by `experiments/physical/crush.py`): `tail_force_ratio` (once a cell has failed and
  the force at a block end is below this x peak, the crack is established, latched: ccx increments are bounded by
  `max_increment_tail` and blocks by 1 / `tail_increments` instead of the near-failure bounds), `stop_force_count`
  (consecutive increments below `stop_force_ratio` x peak that stop the run; default 1) and `end_fraction` (stop at
  this fraction of the ramp). Every record carries `tail`.
* Adaptive release (`adaptive_release: true`; `max_remove` = the cap's upper bound, 64 in the crush runner):
  * The per-block cap on failed cells removed starts at `max_remove`. It halves through the retry ladder when a
    block does not converge, down to 1; after that, other single cells are tried and kept "stuck", and the last
    resort is a block without removal.
  * After a block that converged at its first try the cap doubles; after a block that needed a smaller cap, the
    next block starts at the cap that converged.
  * Removal attempts fail fast: one ccx cut-back instead of about five. The released stress acts at once, so a
    smaller increment does not rescue a removal; in the 003 crush, every converged removal needed at most one
    cut-back.
  * While failed cells wait, the platen is held: a block that leaves failed cells waiting advances
    `release_travel` = 1e-3 / `fine_increments` (2e-5 mm in the crush). The cap therefore changes how a backlog
    is split into removals, not the travel at which cells fail.
  * Before (fixed cap of 8, `release_travel` 0.1 / `fine_increments` = 0.002 mm per block), a 234-cell backlog
    in the 003 crush crept 0.002 mm per block while it was released, ~0.016 mm per 20 min of wall time.
* Checkpoint / resume: solver option `checkpoint_dir` writes `checkpoint.json` / `checkpoint.npz` after every block
  (time, block, step, peak, PEEQ growth rate, tail flag, stuck cells, cell state, removal steps, cell PEEQ, frozen
  displacement, and the path of that block's `.rout`, which is kept until the next checkpoint is written).
  `resume=<dir or dict>` continues the restart chain in a new workdir: steps continue, and the old workdir is only
  read. Tested: a run stopped at half the ramp and resumed gives the closed-form answer at every increment of both
  parts (`test_failure_checkpoint_resume_continues_the_ramp`).
* Hexahedral meshes: gmsh `hexahedron` cells run as C3D8 (full integration, 8 points) with `analysis.order = 1`,
  3D elasticity only. Quad facet groups give the node sets, erosion fragments are face-connected through quad
  faces, and the `.vtu` cells are VTK hexahedra. Tested against the closed-form bar and on a notched all-hex bar
  (subdivision mesh) that separates at the notch (`test_hex_*`).

Limits:

* No contact: eroded faces and fragments can interpenetrate, and a crushed part does not
  re-densify against the platens.
* Removal releases the cell's stress at once (no softening law), so the post-peak response
  depends on the mesh size at the crack (no regularisation).
* Displacement-driven loading only: force / pressure / body / thermal loads are refused, because
  they would keep acting on eroded faces.
* The staggered release limits the crack speed to `max_remove` cells per block; deferred cells
  wait (over the limit) for a few nearly fixed-travel blocks.
* Each block is a separate ccx run (restart read + matrix setup): fine increments near failure
  make this overhead dominate on small meshes.

Verification (`tests/test_solver_ccx_nonlinear.py`): notched bar 4 × 1 × 0.25, two r = 0.2
notches, 1145 C3D10, bonded grips pulled 10 % (NLGEOM), E 1000, σy 1, H 2, limit PEEQ 0.05. The bar
separates at 0.021 grip travel after 46 increments in 8 blocks (52 s). Peak force 0.186 at 0.014;
force after the cut 3e-15. 20 cells failed and 21 were removed as hinges / fragments, all within
x = 1.78–2.05 (notch at x = 2).

## Tension-only erosion, platen options, buckling imperfection (crush of abc_00000065)

Why: a 3D-printed abc_00000065 shaft clamp (Formlabs Alpha Fast) was crushed between flat steel plates. The
arm tips buckled and were crushed, and one arm broke clean off at its neck, the 8.1 mm ring wall between bore
and outer surface at about the bore's mid-height. Every simulation with the default model instead separated
both arm tips just under the platen.

Options (defaults unchanged; `feacore/solver/_ccx_failure.py`, `experiments/physical/crush.py`):

* `failure.criterion`:
  * `"peeq"` (default): PEEQ >= limit in any stress state.
  * `"peeq_tension"`: erosion at the *tensile* PEEQ, the PEEQ a cell gains while its triaxiality
    sigma_m / sigma_vm > `triaxiality_min` (0), summed over the output increments. Cell stress is the mean of
    the nodal Cauchy stress at its corner nodes. Cells also fail at a total PEEQ >= `compression_factor` x
    limit (default 5; `null` = never).
  * `"principal"`: cell sigma_1 >= `stress_limit` (the dogbone UTS in crush.py).

  The damage index (PEEQ / limit, tensile PEEQ / limit, or sigma_1 / stress_limit) orders the release. It also
  drives the near-failure step bounds and the block length, so for `peeq_tension` the 0.02 mm bounds start at
  half the *tensile* failure PEEQ. The `.vtu` cells get `triaxiality`, `s1_cell`, `damage` and `PEEQ_tension`.
  * Assumption: Alpha Fast's compressive strength is not on the TDS. A brittle resin breaks in tension, and a
    crushed zone pressed by a platen keeps carrying load. In practice CalculiX does not converge when it removes
    cells crushed to PEEQ ~0.2 (every removal try diverged), so the runs below use `--compression-factor 0`
    (no compressive erosion).
* `crush.py --platen`:
  * `bonded` (default): every top / bottom node fully held.
  * `sliding_top`: top nodes constrained along the axis only.
  * `sliding`: both platens frictionless. Two bottom nodes take the rigid-body modes: the one nearest the
    face centroid holds x and z, and the one farthest along x near the mid-plane in z holds z. Their coordinates
    are in `run.json`.
  * `friction --mu 0.3`: a rigid C3D8 platen slab (`*RIGID BODY`, its REF node driven) in penalty
    surface-to-surface contact with the top faces and `*FRICTION`, with a bonded base. Penalty 20 E/h normal and
    10 E/h stick; `--contact` overrides them. The force is the base reaction (`failure.force_groups`). Works
    through the restart blocks (tested), but see the results.
* `experiments/physical/imperfection.py`: CalculiX `*BUCKLE` with the platen BCs and 1 kN on the top. The mesh
  nodes are moved by the first mode, scaled to `--amp-mm`. Mid-edge nodes take the mean of their corners' moves,
  so interior C3D10 edges stay straight.
* Mesh `harness_neck` (`experiments/mesh_systems/harness_neck/`): harness_expert plus refinement of the ring
  wall found from the digest. It has ~4 quadratic tets across the neck, against 2.3-2.4 for the other systems.

Tests (`tests/test_solver_ccx_nonlinear.py`):

* `test_failure_criteria_damage_index`: the damage index on synthetic stress states.
* `test_failure_tension_criterion_spares_compression`: a bar compressed past the failure PEEQ is eroded by
  `peeq` but not by `peeq_tension` (closed-form hardening force), and the pulled bar is eroded.
* `test_crush_platen_options`: `sliding` gives the free uniaxial force E eps A and lateral strain nu eps;
  `bonded` holds the top.
* `test_crush_friction_platen_brackets`: friction mu 10 gives the bonded force and mu 0 the `sliding_top`
  force, within 3 % (penalty compliance at strain 1e-3).

Results on abc_00000065 (Alpha Fast dogbone material; the run table is in `experiments/physical/CRUSH_RUNS.md`,
"Neck study"):

* **Why the tips failed first.** The arm-tip load sits ~5 mm inboard of the ring wall's centre line, so the wall
  bends: outer fibre in tension, bore side in compression. Linear elastic at 1.7-1.8 kN: the neck's outer fibre
  sees sigma_1 9-10 MPa, while the 1.5 mm nut-pocket walls at the tips see -88 MPa (bonded) to -106 MPa
  (frictionless) in compression. With yield at 12 MPa, the tips reach PEEQ 0.043 at 0.94 mm / 2.8 kN, in
  compression. The PEEQ criterion ignores the stress state and erodes them, and the part separates at the tips.
  This happens in every mesh system, including the neck-refined `harness_neck`, so the mesh is not the cause.
* **Tension-only erosion** (`peeq_tension`, no compressive erosion): the tips crush (the nut pockets collapse)
  but keep carrying load. The force rises to ~5.7-5.9 kN, and the crack starts in the **neck's outer fibre at
  +5-10 degrees, y = +3-4 mm (height fraction 0.61)**. That is the photo's break, at the bore's mid-height. The
  crack then runs through the neck at almost fixed platen travel, so the snap is unstable. The leading tensile
  sites are the same in every run: neck outer fibre (bending), tip inner face at the bolt hole (only if the
  tips can slide), and nut-pocket walls (mixed shear, triaxiality ~0.05).
* **Platens.** A frictionless top lets the arm tips slide inward, 6.5 mm at 2.7 mm of travel. The arms fold
  (sway buckling): with frictionless tips the first `*BUCKLE` mode, one arm swaying inward, is at 2.07 kN, the
  second at 2.42 kN. The force caps at ~3.0 kN, and the crack starts on the tip's *inner* face at the bolt hole
  (hf 0.90), not at the neck.
  * The bonded run without erosion needs a lateral / normal force ratio per arm of 0.06 (elastic), rising to
    0.17 at 5 mm. Dry steel on resin (mu 0.2-0.4) can supply that, so the tips stick and **bonded is the
    Coulomb-friction answer for mu >= ~0.18**. Frictionless is a lower bound that the test does not reach.
  * With stuck tips the first buckling mode is local, at the tip face, at 13.4 kN. Under load the thin
    nut-pocket walls buckle and crush, which is the "tips buckled" of the test.
  * The friction contact option works on small models. On the 58k-node part, CalculiX's contact chatters:
    ~16k contact elements change status every iteration, there are ~4 increments in 4 min with a 10x softer
    penalty, and node-to-surface diverges. It was not used for the study; mu stays an uncertainty bracketed by
    the two limits.
* **Imperfection.** A buckling-mode imperfection of 0.1 mm (sliding mode 1: about 1 % of the 10.8 mm arm and
  1.2 % of the 8.1 mm neck) on the neck mesh makes the neck the first failure: 3 of the 4 cells of the first
  failed increment, at 4.49 mm, are in the neck. It
  also breaks the symmetry, so one neck cracks ahead of the other. Without it (harness_auto, bonded,
  tension-only), one shear-loaded nut-pocket cell fails first at 4.53 mm and both necks follow within 0.1 mm.
* **Snap load.** The press was force-controlled, so the real break is dynamic at the peak force. The runs are
  displacement-controlled to follow the crack, and their **peak force is the predicted snap load: 5.7 kN** on
  the neck mesh with the imperfection (5.9 kN on harness_auto). The PEEQ-everywhere model gives 3.2 kN (tips),
  and frictionless platens with tension-only erosion give 3.0 kN (tips).
* After the snap: the tensile crack crosses the outer half of both necks at a fixed travel. The bore-side half
  is in compression (triaxiality -0.5, crushed to PEEQ 0.2-0.3) and acts as a hinge, which tension-only erosion
  never breaks. That run was continued with `--compression-factor 5` so the hinge can go.
* Limits: no self-contact (crushed pockets and crack faces can interpenetrate). Only one tensile data point is
  used (failure PEEQ 0.043 at triaxiality 1/3), with no triaxiality dependence above 0. With no compressive
  erosion, the bending hinge left after the tensile crack keeps carrying load, so the post-peak tail is long.
  The photo shows the break; the measured force was not recorded.

<!-- RESULTS -->
