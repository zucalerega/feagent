# harness_neck: harness_expert plus a fine mesh through the necks of the ring

**Approach.** It is the `harness_expert` spec, built by importing `../harness_expert/build.py` without changing
it. That spec has digest-driven groups, thin-wall layers, and refinement at the platens, fillets and holes.
`harness_neck` adds one refinement that it finds from the `feagent` geometry digest (`find_necks` in `build.py`):

- **Bore:** the largest concave cylinder whose axis is normal to the load axis.
- **Ring:** the convex cylinders coaxial with the bore and larger than it. The wall between bore and ring
  (thickness t = r_ring - r_bore) is the arms' neck. When the part is crushed, the arm-tip loads sit inboard of
  this wall, so its outer fibre is in bending tension.
- **Refinement:** both the ring faces and the bore get size `size` (1.6 mm for nonlinear, 1.0 mm for linear).
  At the ring faces it grows to the global size over 1.2 t; at the bore, over 0.5 t.

The faces are selected by query (kind, radius, concave, axis), never by tag. A part without a bore and a coaxial
ring keeps the harness_expert spec unchanged. The necks found are written to `report.json`, under `contract.necks`.

Why it exists: a real crush of abc_00000065 (Formlabs Alpha Fast) broke one arm at this wall, at the bore's
mid-height. Every earlier mesh had 2.3-2.4 quadratic tets across the 8.1 mm wall, too few for the bending
gradient that drives a tensile crack. This mesh has about 4.

**Knobs.** `NECK` in `build.py` (`size`, `dist_factor`), plus `--neck-size` and `--neck-dist-factor` on the CLI.
The rest are `harness_expert`'s `PARTS` knobs.

| part | purpose | order | nodes | tets | gates | min SICN | ring wall (mean edge, elements across) | build s |
|---|---|---|---|---|---|---|---|---|
| abc_00000065 | nonlinear | 2 (straight interior edges) | 58,020 | 34,317 | all 10 pass (thin_walls 1.28 >= 1, lowered as in harness_expert) | 0.32 | 2.0 mm, ~4.0 | 62 |

The other systems' meshes have 2.3-2.4 elements across the same wall: harness_auto (33,824 nodes) at a mean
edge of 3.4 mm, and harness_expert (39,608 nodes) at 3.5 mm. With a 1.4 mm neck size the mesh has 64,660
nodes, over the ~60k budget. On abc_00000065 the digest finds the bore (r 19.67 mm, axis z) and two ring faces
(r 27.77 mm), giving a wall of 8.10 mm. Each ring face spans y = -15.0 to +12.0 mm.

Build (contract CLI; CPU renderer for mesh.png):

    LIBGL_ALWAYS_SOFTWARE=1 GALLIUM_DRIVER=llvmpipe __GLX_VENDOR_LIBRARY_NAME=mesa CUDA_VISIBLE_DEVICES="" \
      python experiments/mesh_systems/harness_neck/build.py benchmarks/gold/abc_00000065_bc0/public/geometry.step y OUT --purpose nonlinear

Straight interior edges were checked with `../harness_expert/straightness.py`: 0 curved interior edges.

**Imperfect variant.** The crush study also uses a buckling-mode imperfection:
`experiments/physical/imperfection.py --mesh-dir OUT --out OUT_imp --platen sliding` moves the nodes by the
first CalculiX *BUCKLE mode. It writes a contract mesh dir; see `docs/solver_ccx_nonlinear.md`.
