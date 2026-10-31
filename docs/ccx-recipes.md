# CalculiX (ccx 2.21) recipes validated by the benchmark harness

Every recipe here was exercised by `feabench`'s reference pipeline or tests and scored by the
evaluator. Use it as seed context for agents. Improvers are expected to extend it.

## Units (all verification tasks)

| Quantity | Unit |
|---|---|
| length | mm |
| force | N |
| stress / E | MPa (N/mm²) |
| mass | tonne |
| time | s |
| density | **tonne/mm³** (steel 7.85e-9) |
| acceleration | **mm/s²** (g = 9810) |
| frequency | Hz |

A density in kg/m³ (7850) instead of 7.85e-9 is the classic error. The evaluator's material gate
reports it explicitly.

## Mesh

* **Use quadratic tetrahedra (C3D10).** Linear tets (C3D4) are far too stiff in bending.
* gmsh: `Mesh.ElementOrder = 2` and `Mesh.HighOrderOptimize = 1`. Run gmsh in a subprocess: the
  high-order optimiser can abort the whole Python process. See `feabench/fem/meshworker.py`.
* **gmsh to ccx node order for tet10:** take gmsh nodes in the order `[0,1,2,3,4,5,6,7,9,8]`.
  gmsh's last two mid-edge nodes are swapped relative to ccx.
* **Refine where the evaluated quantity lives.** The verification convergence studies show:
  * displacements are within 0.5 % at a few thousand DOF;
  * stresses at curved surfaces need h ≲ R/8 locally;
  * stresses in fillets need a local refinement box.
* **Corner-node stresses are unreliable** where three faces meet. Few elements share those nodes,
  and extrapolated values can be off by 15 %. Prefer values on an edge or face, or averages.

## Boundary conditions

```
*NSET, NSET=SYM_X
...node ids on the plane x=0...
*BOUNDARY
SYM_X, 1, 1, 0.0          ** symmetry plane x=0: ux = 0
FIXED, 1, 3, 0.0          ** fully fixed
```

* Remove rigid-body modes without over-constraining. Use a 3-2-1 set at nodes that exist (CAD
  corners), e.g. ux=uy=0 at one base corner and uy=0 at another. Restraining all lateral motion
  of a whole face kills Poisson contraction and changes the answer.

## Loads

**Pressure.** Use element-face distributed loads. Positive pressure pushes *onto* the face;
negative pulls (tension).

```
*DLOAD
<elem>, P<face>, <p>
```

C3D10 / C3D4 face numbers by corner nodes: 1 = (1,2,3), 2 = (1,4,2), 3 = (2,4,3), 4 = (3,4,1).

**Total force spread uniformly over a face.** Either:
* consistent nodal forces `*CLOAD` with f_i = ∫ N_i t dA (on a flat tri6 face, corner nodes get 0
  and mid-edge nodes get A/3 each); or
* a distributing coupling, as below with dofs `1, 3`. Area-weighted, intended for plane faces.

**Torque or moment on a face.** Validated on `torsion_shaft`: twist exact, von Mises within 0.4 %.

```
*NODE, NSET=NREF
100001, 0., 0., 100.
*SURFACE, NAME=SEND, TYPE=ELEMENT
<elem>, S<face>
...
*COUPLING, REF NODE=100001, SURFACE=SEND, CONSTRAINT NAME=TQ
*DISTRIBUTING
1, 6
*STEP
*STATIC
*CLOAD
100001, 6, 100000.      ** moment about z (dof 6) in N*mm
```

**Rigid interface (bolt, RBE2-like).** Use `*COUPLING` + `*KINEMATIC` (dofs `1, 3`) and fix the
reference node with `*BOUNDARY ref, 1, 6`. If the interface is simply rigid and fixed, fix its
surface nodes directly.

**Body force.** Needs `*DENSITY`.

```
*DLOAD
EALL, GRAV, 1.0E6, 0., 0., -1.
```

## Analyses

| Analysis | Keyword |
|---|---|
| linear static | `*STATIC` |
| modes | `*FREQUENCY` then `<n>`. Needs `*DENSITY`. Results in the `.dat` "E I G E N V A L U E   O U T P U T" table (cycles/time = Hz). |
| linear buckling | `*BUCKLE` then `<n>`, with the reference load in the same step; factors in the `.dat` table |

## Output and self-checks

* `*NODE FILE` with `U, RF` and `*EL FILE` with `S, E`. The evaluator appends these anyway.
* **ccx can exit with status 0 after printing `*ERROR`.** Always scan stdout.
* **ccx rejects numeric fields longer than 20 characters.** Python's `repr` writes
  `1.0780369855799999E-17` (22 characters) and the whole `*NODE` card fails. Write numbers with
  `f"{x:.12g}"`.
* **`RF` in the .frd is the nodal internal force at every node.** At a support node that also
  carries load it equals reaction plus applied load. Check equilibrium on support regions that
  carry no load.
* **The .frd stores 6 significant digits.** Use the deck's coordinates for geometry, not the frd's.
* **Cheap checks that catch most catastrophic errors:**
  * mesh volume equals CAD volume (units, missing bodies);
  * support reactions equal the applied resultant (missing, doubled or misdirected loads);
  * a second, finer mesh changes the answer by much less than the tolerance (convergence).
