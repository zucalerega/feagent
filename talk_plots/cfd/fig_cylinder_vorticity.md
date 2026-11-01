# fig_cylinder_vorticity: "Vorticity: the shear layers"

**Caption.** Vorticity (dv/dx - du/dy) of the DFG 2D-1 flow, 0.1 <= x <= 0.62 m: the same solve as
`fig_cylinder_wake` (47,264 cells). Clockwise vorticity (blue) leaves the top of the cylinder
and the bottom wall, anticlockwise (red) the bottom of the cylinder and the top wall. At Re = 20 the two layers
behind the cylinder stay attached to a closed, steady bubble; there is no vortex shedding.

**Caveats**
- Cell-centred Gauss gradients, interpolated linearly between cell centres. The colour scale is clipped at
  +-20 1/s; the vorticity on the cylinder's surface is larger.
