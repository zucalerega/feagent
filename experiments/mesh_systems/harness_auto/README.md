# harness_auto: the mesh harness, zero-config

The baseline system. It is the mesh harness's zero-config start: the spec `feagent automesh` writes
(`feagent.mesher.auto_spec`) plus the two contract groups, which are selected by geometric query. Each group is a
`plane` face with the platen's outward normal (±axis) inside a thin box (±0.5 mm along the axis, across the whole
part). The `at:` key is too tight for these metre-unit parts. `feagent check` meshes and gates it. Nothing else is
tuned, so the linear mesh is exactly what an agent gets from `automesh` once it has named its faces. The nonlinear
mesh is the coarse spec the earlier crush runs used: fewer, larger cells, so a crush to separation takes hours.

FEAgent pieces: `meshing-harness/` (`auto_spec`, the query selectors, `feagent check`, `feagent.quality.assess_mesh`,
`feagent render --view quality`) and `feacore.solver._ccx_mesh` (the CalculiX node order for `mesh.inp`).

    python experiments/mesh_systems/harness_auto/build.py benchmarks/gold/abc_00000003_bc0/public/geometry.step z OUT --purpose linear
    python experiments/mesh_systems/harness_auto/build.py benchmarks/gold/abc_00000065_bc0/public/geometry.step y OUT --purpose nonlinear

Knobs:

| purpose | size (global / min / curvature / thin-wall layers) | order |
|---|---|---|
| linear (automesh defaults) | diag/40 = 2.5 mm / diag/800 = 0.13 mm / 12 / 2 | 2 (gmsh: curved boundary edges, straight interior) |
| nonlinear, abc_00000003 | 8 mm / 1 mm / 4 / 1 | 2 (C3D10) |
| nonlinear, abc_00000065 | 6 mm / 0.5 mm / 8 / 1 | 1, then midside nodes at the straight edge midpoints (gmsh `SecondOrderLinear`), giving C3D10 |

All meshes use netgen optimisation. Other notes:

- `mesh.inp` is written from `mesh.msh` by this script, not by gmsh's exporter. It holds the volume cells only
  (`*ELEMENT, TYPE=C3D10, ELSET=domain`) and `*NSET` `bottom` / `top` / `domain`, with no CPS surface elements.
- `report.json` comes from `assess_mesh` on the delivered `mesh.msh`, with the CAD analysed as the mesher does it,
  plus the harness's own geometry / groups / mesh gates (`mesh_gates`).
- The build is deterministic: rebuilding gives exactly the meshes of the earlier crush runs (003: 23 926 nodes /
  12 489 cells; 065: 33 824 / 19 076).

## Numbers (built 2026-10-31 on the laptop, CPU only)

| part | purpose | element | nodes | cells | DOFs | SICN min | gamma min | volume error | thin walls | build |
|---|---|---|---|---|---|---|---|---|---|---|
| abc_00000003 | linear | C3D10 | 128 644 | 75 172 | 385 932 | 0.331 | 0.215 | 0.00 % | warn | 52 s |
| abc_00000003 | nonlinear | C3D10 | 23 926 | 12 489 | 71 778 | 0.230 | 0.235 | 0.01 % | warn | 29 s |
| abc_00000065 | linear | C3D10 | 97 385 | 58 323 | 292 155 | 0.269 | 0.254 | 0.00 % | pass | 61 s |
| abc_00000065 | nonlinear | C3D10 (straight) | 33 824 | 19 076 | 101 472 | 0.302 | 0.140 | 0.54 % | warn | 50 s |

All other gates pass: no inverted cells, SICN, gamma, volume and area fidelity, one connected component. The face
counts match the contract (003: 1 bottom / 1 top; 065: 1 / 2). The thin-walls warning means fewer than 2 cells
across some walls; it shows on the coarse nonlinear meshes and, for 003, on one wall of the linear mesh.
