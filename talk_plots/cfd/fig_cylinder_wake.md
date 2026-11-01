# fig_cylinder_wake: "Flow past a cylinder at Re = 20"

**Caption.** Steady laminar flow around a cylinder in a channel, the DFG benchmark 2D-1 (Schaefer & Turek 1996):
speed with streamlines, x <= 1.1 m of the 2.2 m channel. The inflow is parabolic with mean speed
0.2 m/s, and Re = U_mean D / nu = 20. The flow splits around the cylinder, speeds up in the
two gaps, and closes behind it in a small recirculation bubble (83.7 mm long).

**Key numbers** (`cfd_numbers.json`, cylinder)
- Mesh: the problem's own block-structured quad mesh at h = h0 / 4 (47,264 cells).
- Pressure drop front to back of the cylinder: 0.11737 Pa against the reference
  0.11752 Pa (-0.12%).
- Drag coefficient 5.5702 against 5.5795 (-0.17%); lift coefficient
  0.01066 against 0.01062 (+0.4%).
- Solver: 839 SIMPLEC iterations, converged = True, 160 s (v1912).

**Sources.** `feacore/problems/cfd/cylinder.py` (setup, mesh, FEATFLOW reference values), solved with
`feacore.solver.solve(case, backend="openfoam")`.

**Caveats**
- The drag is a wall-force integral. On this branch it converges about 0.15 % low (`docs/benchmarks_cfd.md`).
- Streamlines are drawn from the solution interpolated linearly between cell centres and wall faces. The two
  eddies behind the cylinder are contours of the stream function there (`eddies()`), the rest matplotlib
  streamlines.
