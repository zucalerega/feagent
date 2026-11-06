# fig_cylinder_mesh: "Close to the cylinder"

**Caption.** The DFG 2D-1 flow close to the cylinder, solved on the problem's block-structured quad mesh at
h = h0 / 2 (11,816 cells; grey lines): an O-grid of 128 cells around the
cylinder inside a square box, graded towards the walls. Behind the cylinder the streamlines close into two
counter-rotating eddies, 82.0 mm long on the axis (83.7 mm on
the finer h0 / 4 mesh of `fig_cylinder_wake`).

**Key numbers** (`cfd_numbers.json`, cylinder_mesh): 256 iterations, converged = True.

**Caveats**
- The recirculation length is where u turns positive on the axis y = 0.2 m behind the cylinder, read with the
  solver's own probe (`OpenFOAMResult.probe`).
- The eddies are contours of the stream function behind the cylinder (`eddies()`), with an arrowhead for the
  direction of turn; the other lines are matplotlib streamlines.
