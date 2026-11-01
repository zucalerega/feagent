# fig4_mesh_before_after: "Same part, same solver: the trained agent grades the mesh into the corners"

**Caption.** Cook's membrane in deep1. Left: a generation-0 agent's mesh, which is automesher-level. Right: the best generation-10 mesh, refined into the re-entrant top-left corner and the loaded top-right corner.

**Key numbers**
- Left, g0-cooks_membrane-a1: 282 DOFs, error 5.09%, g = 1.00, base score 0.300.
- Right, g10-cooks_membrane-a7: 710 DOFs, error 1.07%, g = 4.76, base score 1.126.
- Automesher: 336 DOFs, error 5.11%.

**Sources.** `ace_runs/deep1/gen0/reports/g0-cooks_membrane-a1_mesh.png` and `ace_runs/deep1/gen10/reports/g10-cooks_membrane-a7_mesh.png`, cropped to the mesh. The numbers come from the matching `gen*/runs/*.json`.

**Caveats**
- Nothing was re-meshed. The left mesh is an agent's, not the automesher's own, but its accuracy matches the automesher's (1.00x).
- The report error maps (`*_error.png`) are not shown, because their colour scales differ between the two runs.
