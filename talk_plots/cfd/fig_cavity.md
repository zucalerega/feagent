# fig_cavity: "Lid-driven cavity at Re = 100"

**Caption.** A unit square of fluid driven by its lid (1 m/s, Re = 100), the classic test of Ghia, Ghia & Shin
(1982). Streamlines are contours of the stream function psi = integral of u dy: solid for the main vortex
(psi = -0.1, -0.09, -0.07, -0.05, -0.03, -0.01, -0.001, -0.0001), dashed for the two counter-rotating corner eddies
(psi = 1e-07, 1e-06, 4e-06, 1e-05). The main vortex centre sits at (0.614, 0.738), where
psi = -0.1034 m^2/s.

**Key numbers** (`cfd_numbers.json`, cavity): uniform 128 x 128 quad
grid, 1932 iterations, converged = True.

**Caveats**
- psi is integrated on an 801 x 801 sampling grid of the linearly interpolated velocity; the vortex centre is
  that grid's psi minimum (spacing 1.25 mm).
