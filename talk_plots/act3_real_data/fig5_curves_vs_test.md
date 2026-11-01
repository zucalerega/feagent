# fig5_curves_vs_test: "The automesher's mesh is too stiff: 14-19 % above the real test"

**Caption.** Same solver, same learned setup, only the mesh differs: the gmsh default mesh over-predicts the force
at every checkpoint, the swarm's bc0 mesh is close, a 41k-node uniform mesh sits on the test. Closing the gap at
few nodes is what fine-tuning on the real test must do.

**Key numbers** (force at 0.5 / 1.0 / 1.5 / 2.0 / 2.5 mm):
- measured: 860 / 2164 / 3308 / 4171 / 4743 N
- gmsh default (2937 nodes): 983 / 2584 / 3837 / 4828 / 5572 N, i.e. +14 to +19 %, mean 16.6 %
- swarm bc0 mesh (6430 nodes): 887 / 2325 / 3439 / 4315 / 4966 N, mean 4.5 %
- uniform 2 mm (41186 nodes): 859 / 2246 / 3309 / 4126 / 4723 N, mean 1.1 %

**Data.** /private/tmp/claude-501/-Users-aryan-Documents-Aryan-CS-scienceclawhack/29d391ed-69f3-4559-b365-a260e486ee85/scratchpad/nl_mesh/<auto|swarm|h20>/curve.csv mapped with the frozen delta 0.2335 mm and c 2.898e-5 mm/N
(physical_tests/abc_00000065_compression/nonlinear_spec.json); test CSV as fig1.

**Caveats.**
- The simulated curves start with a straight segment from (0, 0) to the first increment: that is the seating gap
  of the mapping (u_cross = u_part + delta + c F), not physics; the real toe is curved.
- Simulated curves are cut at 2.75 mm for display; the scored range ends at 2.5 mm.
