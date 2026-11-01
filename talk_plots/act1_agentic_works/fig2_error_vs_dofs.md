# fig2_error_vs_dofs: "2D: the best agents beat the automesher's whole cost curve; 3D: a tie"

**Caption.** Scored error against DOFs (log-log) for every valid agent run in the 4 field-mode runs. Open circles are generation 0. The grey line is the automesher refined uniformly (gmsh default sizes x MeshSizeFactor), and the X is the automesher itself. The star is the best-scoring run.

**Key numbers** (best-scoring run against the uniformly refined automesher, log-log interpolated)

| problem | best run | error / DOFs | automesher error at the same DOFs | lower error | automesher DOFs for the same error | fewer DOFs |
|---|---|---|---|---|---|---|
| Cook | deep1 g10-cooks_membrane-a7 | 1.07% / 710 | 3.69% | **3.4x** | 7,690 | **10.8x** |
| LE1 | split1 g5-nafems_le1-a2 | 0.81% / 782 | 1.87% | 2.3x | 2,122 | 2.7x |
| LE11 axisym. | deep1 g10-nafems_le11_axisym-a4 | 0.71% / 648 | 1.68% | 2.4x | 3,267 | 5.0x |
| LE11 3D | le11 g6-nafems_le11_3d-a4 | 2.63% / 8,739 | 2.80% | 1.06x | 9,670 | 1.11x |

**Sources**
- `act1_runs.csv` (from `ace_runs/*/gen*/runs/*.json`).
- `automesher_sweep.json` (made by `automesher_sweep.py`). Its factor-1 point reproduces `ace/field_refs/<problem>_auto.json` exactly on all 4 problems (same DOFs, same error to 1e-9).

**Caveats**
- "The best agents beat the whole curve" is about the best runs: on LE1 many agent runs sit on or above the uniform curve; on Cook almost all runs after gen 0 sit below it.
- The uniform sweep is **new computation** made for this talk: gmsh plus the fixed skfem solver plus the same error function the runs were scored with. No agent or API was involved.
- "Error" = max(energy-norm field error, RMS von Mises error at 12 probe points), the scored error.
- The 37 LE11-axisymmetric runs near 30% error (4 in deep1 gen 0, 33 in le11), and the 2 LE11-3D runs at 124%, are wrong setups (temperature field; fixed instead of sliding supports), not meshes. They are drawn so nothing is hidden.
- In LE11 3D the agents mostly refine uniformly: no gain over the refined automesher.
