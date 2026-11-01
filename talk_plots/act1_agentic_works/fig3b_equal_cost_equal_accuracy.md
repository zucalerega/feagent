# fig3b_equal_cost_equal_accuracy: "Matching the best agent costs the automesher up to 11x the DOFs"

**Caption.** For each problem's best-scoring agent run, two factors against gmsh's default mesh refined uniformly:
- how much lower the agent's error is at the same DOFs (light bars);
- how many times more DOFs the automesher needs to reach the agent's error (dark bars).

**Key numbers.** Cook 3.4x / 10.8x; LE1 2.3x / 2.7x; LE11 axisymmetric 2.4x / 5.0x; LE11 3D 1.06x / 1.11x. All are interpolated inside the sweep; nothing is extrapolated.

**Sources.** `automesher_sweep.json`, `act1_numbers.json` (`error_reduction_at_equal_dofs`, `dof_saving_at_equal_error`).

**Caveats**
- The comparison mesher is gmsh's default refined *uniformly*. A human expert, or gmsh with curvature/size fields, would do better than this baseline.
- One best run per problem, interpolated linearly in log-log between sweep points.
- This is the fairest form of "Nx better than the automesher": the score's own reference, the unrefined automesher at 336 DOFs, flatters the agents' meshes, which use 2x the DOFs.
