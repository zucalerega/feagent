# fig_supersonic_reference: "The target flow"

**Caption.** Inviscid (Euler) flow of a perfect gas (gamma = 1.4) at Mach 3 and 10 deg angle of attack over a
blunted wedge (nose radius 0.05 m, half-angle 10 deg): the Mach number of the reference solve the swarm is
scored against, on the problem's own structured mesh (12,288 cells, h = 0.00208 m). A detached
bow shock stands ahead of the nose. Behind its nearly normal part the flow is subsonic (inside the dashed sonic
line M = 1), then it accelerates back to supersonic speed along both flanks.

**Key numbers**: 100,585 iterations, converged = True; mass imbalance
1.0e-06.

**Sources.** `supersonic_blunt_wedge_2d_reference_flow.npz` / `.msh` from the run's `ace/field_refs/` (nodal values), rendered with
linear (Gouraud) interpolation on a Delaunay triangulation of the mesh points.
