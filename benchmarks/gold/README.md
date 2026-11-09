# Gold benchmarks

Real third-party CAD with generated supports and loads, and reference values certified by a
nested-mesh convergence study on the fixed solver. The parts come from the ABC dataset, rescaled
to a 100 mm bounding-box diagonal. The gold-benchmark generator (`feacore.goldgen`) produced them on an H100, and `scripts/gold.py import` brought them in from its
archives.

| | |
|---|---|
| Sets | 9 BC sets on 2 parts, from archive `goldgen-l2_0000` (a tenth, `abc_00000003_bc3`, is retired: see [Retired](#retired-abc_00000003_bc3)) |
| Physics | static 3D linear elasticity, steel (E = 210 GPa, nu = 0.3), no gravity |
| Supports, loads | clamped faces and rollers; total forces (1-50 kN) or pressures (1-50 MPa) on faces |
| Study | nested tet10 meshes L0, L1, L2 (h, h/2, h/4; up to 1.0M DOFs), solved with skfem P2 |
| References | QoIs that converged with a fine-grid GCI within 1 % (energy), 2 % (displacement, reaction) or 3 % (stress). Each reference's tolerance is max(that limit, 2 x GCI) |
| Singular QoIs | QoIs that diverge under refinement (peak stress at clamp edges): `expectations()` says "diverge", and they have no reference |

The QoIs are strain energy, max displacement, volume-mean von Mises stress, the mean displacement
of each loaded face and, with several supports, the support reactions. Each set's statement names
the ones it asks for.

![The gold solutions on their L1 meshes](solutions_L1.jpg)

The image shows every set re-solved on its L1 mesh (it predates the retirement of `abc_00000003_bc3`,
whose panel it still shows). Each panel has von Mises stress on the
deformed shape from two views, with supports outlined black, loads orange, and its own colour
scale. Below each panel, a table compares the L1 values with the gold references.
`scripts/gold.py render --level 1 --png DIR` regenerates it.

## Layout

```
benchmarks/gold/
  index.json                 every set (part, source archive, reference and singular QoIs, DOFs)
                             and every mesh (h, cells, sha256)
  <id>/public/problem.json   statement for agents: supports and loads in words, faces named f<tag>
  <id>/public/geometry.step  the part in metres; its OCC face tags are the f<tag> names
  <id>/private/reference.json   references, the BC set, the reference case, the part's faces
  <id>/private/study.json       QoIs on every mesh level, GCI per QoI
  _meshes/<part>/L<k>.msh    the study's meshes (not in git: 82 MB here; index.json has their sha256)
```

All of a part's BC sets share one STEP file and one mesh sequence. Agents get `public/` only.
Never mount `private/` or `_meshes/` into an agent sandbox.

## How the swarm uses them

- Each set is a registered feacore problem (`feacore.problems.gold.GoldProblem`, tag `gold`,
  family = part id), so `registry.get("abc_00000003_bc0")`, the UI's Benchmarks page and the ACE
  loop's `get_problem` all find it. Nothing needs exporting: the files are the problem.
- `build_geometry()` imports the STEP as is and names every face `f<tag>` and the volume `solid`.
  Those are the groups an agent's mesh must carry, and the groups `inspect_geometry` lists.
- Scoring: every reference in `private/reference.json` with its tolerance.
  `scripts/compare_nafems.py REPLY --problem <id> --root benchmarks/gold` already works for a
  single reply. For a loop that scores one QoI, `GoldProblem.scored_qoi()` picks one: strain
  energy, else a loaded face's mean displacement, else `u_max`, `vm_mean`, then a reaction.
  For ACE: `TASKS[p.id] = {"qoi": p.scored_qoi(), "node_cap": ...}`.
- Uniform refinement is a weak baseline here, because re-entrant edges slow convergence:

  | mesh | DOFs | references within tolerance | median error | worst error |
  |---|---|---|---|---|
  | L0 | 17-23k | 6 of 31 | 4.8 % | 16 % |
  | L1 | 113-146k | 22 of 31 | 1.7 % | 3.9 % |
  | L2 | 0.8-1.0M | 31 of 31 | 0.5 % | 1.3 % |

  An agent working under a node cap of L0-L1 size has to refine locally, where it matters.

| set | supports and loads | references | singular | `scored_qoi()` |
|---|---|---|---|---|
| `abc_00000003_bc0` | clamp f21, roller x f19, force 7.8 kN f18 | vm_mean, vm_max, u_mean_f18 | - | u_mean_f18 |
| `abc_00000003_bc1` | clamp f23, clamp f19, force 13.8 kN f25 | vm_mean, Rz_f23 | vm_max | vm_mean |
| `abc_00000003_bc2` | clamp f23, roller z f25, pressure 6 MPa f3, pressure 1 MPa f16 | strain_energy, u_max, vm_mean, u_mean_f3, u_mean_f16 | vm_max, Rx_f23, Ry_f23 | strain_energy |
| `abc_00000003_bc4` | clamp f20, pressure 10.8 MPa f3 | strain_energy, vm_mean, u_mean_f3 | u_max, vm_max | strain_energy |
| `abc_00000003_bc5` | clamp f3, force 6.6 kN f18, pressure 1.5 MPa f21 | vm_mean | vm_max | vm_mean |
| `abc_00000003_bc6` | clamp f20, pressure 14.2 MPa f16, force 1.2 kN f21 | u_max, u_mean_f16 | vm_mean, vm_max | u_mean_f16 |
| `abc_00000003_bc7` | clamp f2, pressure 4 MPa f23, pressure 2.1 MPa f16 | strain_energy, vm_mean, u_mean_f23, u_mean_f16 | - | strain_energy |
| `abc_00000065_bc0` | clamp f12, pressure 2.1 MPa f7 | strain_energy, u_max, vm_mean, u_mean_f7 | vm_max | strain_energy |
| `abc_00000065_bc1` | clamp f9, clamp f18, pressure 21.5 MPa f13 | u_max, vm_mean, u_mean_f13, Rx_f9, Rz_f9, Rx_f18, Rz_f18 | vm_max | u_mean_f13 |

## Commands

```bash
python scripts/gold.py import goldgen-l2_0001.tgz [more archives or extracted dirs]   # checks, then adds
python scripts/gold.py list [ID | PART ...]                       # references (evaluator-only)
python scripts/gold.py verify [ID ...] --level 0                  # re-solve: the study reproduces
python scripts/gold.py restate [ID ...] [--check]                 # plane faces described again (outward normals)
python scripts/gold.py render [ID ...] --level 1 --png out/gold   # runs for the UI viewer, PNGs
```

`import` checks each set before adding it. It needs all four files, consistent ids and nothing
private in `public/`. The reference case must load, every reference must be a stated QoI, and
every group must be a face of the part, and the study's strain energy must rise from level to level
(the energy bound: `feacore.goldgen.run.energy_drop`). The STEP must re-import with the part's face tags. Sets
without a reference are skipped, since there is nothing to score. Importing the same archive
again changes nothing, and new archives from the generator's batcher (`goldgen-l2_0001`, ...) add
to the index. The generator writes a separate STEP copy for each BC set, differing only in a
timestamp. These are stored byte-identical, so git keeps one blob per part.

`verify` and `render` need the meshes, which come with the archive. On a laptop CPU (SciPy
SuperLU), L0 solves in about 1 s and L1 in about 20 s at 3 GB. L2 (1.0M DOFs) needs a GPU (cuDSS)
or tens of GB.

`render` records each solve in the run store (source `gold`, agent `gold:L<level>`), where the
UI's field viewer shows it. `--png DIR` also writes one image per set: von Mises stress on the
deformed shape from two views, with the supports and loads marked. It also writes a contact sheet.

## Checked

- All sets re-solve on L0 with this checkout's solver to the generator's values, to 1e-12
  relative (`verify --level 0`, and `tests/test_gold.py` when the meshes are present).
  Spot-checked on L1 too.
- Every STEP re-imports with the part's 25 / 55 faces at the recorded areas and centroids, to
  1e-11.
- The fixed solver needs its curved-mesh fixes (robust inverse
  isoparametric map, Jacobian-based validity for curved cells, reduced SVD in the rigid-body
  checks). Without them, skfem rejects some of these tet10 meshes ("Newton iteration didn't
  converge").
- Every planar face's outward normal comes from points on the face (`feacore.goldgen.parts.face_points`):
  `restate --check` finds nothing to change. The generator used to probe at the face's centroid, which
  for a ring lies in its hole, where both sides can be inside the solid; it stated abc_00000003's
  shoulders f18 (x = -19.5 mm) and f19 (x = +19.5 mm) facing inward, and `restate` corrected the words
  of bc0, bc1 and bc5 and every set's part faces. Forces carry explicit vectors, so no load changed.

## Retired: `abc_00000003_bc3`

Its references broke the energy bound. With pressure loads and clamped supports the FE strain energy
rises under nested refinement, but the study's fell: 2.166, 1.983, 1.940 J to a reference of 1.927 J,
while independent meshes rise to about 2.20 J (`single_agent/README.md`, Findings). Its GCI looked
clean (p = 2.1), so the generator certified it.

Cause: abc_00000003's nested L1 and L2 meshes fold into the bore near its open end. The base mesh
(23.7 mm, 10 segments around the 14.7 mm bore) triangulates the 1 mm ring f24 under the bore with
edges that cut across the bore's edge. A split puts a vertex at each edge's midpoint and snaps it onto
the plane of f24, not into the ring, so from L1 on some vertices of f24 lie 0.25 mm inside the bore
mouth and the cells above them reach into the bore. Every such cell is well shaped, so neither the
scaled-Jacobian gate nor the solver noticed. On some bore facets the cell then lies on the bore side,
and the 45.6 MPa bore pressure pushes outward there. The pressure gains a net force (the stored study:
955 N at L1 and 2.4 kN at L2, where it should be zero), and the extra material stiffens the part.

| sequence (regenerated here, gmsh 4.15) | cells in the bore | f20 facets with reversed normal | mesh volume vs CAD | bc3 strain energy |
|---|---|---|---|---|
| original L0 / L1 / L2 | 0 / 111 / 672 | 0 / 18 / 81 | +0.006 / +0.18 / +0.24 % | 2.165 / 1.978 J / - |
| fixed L0 / L1 / L2 | 0 / 0 / 0 | 0 / 0 / 0 | -0.14 / +0.002 / +0.002 % | 2.171 / 2.192 J / - |

(The stored meshes are not in git; the regenerated L1 is close to the stored one, 1.978 vs 1.983 J.
L2, 1.1M DOFs, was not solved here. The fixed L0's -0.14 % comes from the curved-local fallback, which
straightens a few cells.)

The generator is fixed (`feacore/goldgen/mesh.py`): the base mesh is refined on faces where two
simulated splits would put nodes off the face (`_base_mesh`; for abc_00000003 only f24, to 4.85 mm),
a level with a node outside its trimmed face is rejected (`_face_leaks`), and `certify` gives no
references when the strain energy falls from one level to the next (`feacore.goldgen.run.energy_drop`,
also an import check). bc3 was removed rather than re-certified: its new references need the L2 solve
(a GPU box) on a sequence its sibling sets do not share. To bring it back, regenerate abc_00000003 with
the fixed generator and re-solve all its sets.

The other abc_00000003 sets were solved on the same folded meshes. Their stored references against a
two-level Richardson estimate from the fixed L0/L1 (same observed order):

| set | effect of the folded meshes |
|---|---|
| bc1, bc2, bc4, bc6 | none: L1 values do not change (< 0.01 %) when the cells in the bore get zero stiffness |
| bc5 | none on its reference (vm_mean within 0.4 % of the estimate) |
| bc0 | L1 values up to 1.5 % low; references within 0.5-0.8 % of the estimate (tolerance 2.8-3 %) |
| bc7 | L1 values 0.8-2.5 % low; references 0.4-1.4 % below the estimate (tolerance 1.2-3 %; strain energy 0.7 % of 1.2 %) |

No other reference leaves its tolerance, but bc0's and bc7's lean low by up to about 1.4 %;
re-certifying abc_00000003 removes that too. abc_00000065 shows the mechanism only marginally (a few
nodes 0.04 mm off f15, one 1 mm off f8 at L2 from the high-order optimiser), and its mesh volume
converges to the CAD (+5e-5, +1e-5, +3e-6 at L0-L2), so its references stand.
