# CFD benchmarks (`feacore/problems/cfd/`)

Incompressible, steady, laminar flow problems with exact or published reference values, so the
verification layer (Richardson extrapolation / GCI, `feacore/verification/`) runs on CFD as well as
FEA. Every problem is a `feacore.problems.base.Problem` with physics `"flow"`, registered at import
and tagged `cfd`. The fixed flow solver is the OpenFOAM backend (`backend="openfoam"`, `simpleFoam`).

| id | analysis | QoIs (agent reports) | reference | tol | kind |
|---|---|---|---|---|---|
| `poiseuille_channel` | plane Poiseuille flow, 2D channel 5 cm x 1 cm, fully developed inflow, Re = 10 | `dp_AB` = p(A) - p(B) on the centreline [Pa]; `u_centreline` [m/s]; `wall_force_bottom`, `wall_force_top` (x-force of the fluid on each wall) [N/m]; `flow_rate_outlet` [m^2/s] | exact: 0.3 Pa; 1.5 U; tau_w L = 6 mu U L / H; U H | 1 % | analytical |
| `lid_driven_cavity_re100` | lid-driven unit square, Re = 100 | `u_vcl_min`, `u_centre` = u_x on x = 0.5; `v_hcl_max`, `v_hcl_min` = u_y on y = 0.5 [m/s] | Ghia, Ghia & Shin (1982) Tables I and II | see below | published_numerical |
| `cylinder_channel_dfg_2d1` | Schaefer & Turek (1996) DFG 2D-1: cylinder D = 0.1 in a 2.2 x 0.41 channel, Re = 20 | `drag`, `lift` (force of the fluid on the cylinder) [N/m]; `dp_front_rear` = p(0.15, 0.2) - p(0.25, 0.2) [Pa] | c_D = 5.57953523384, c_L = 0.010618948146, dp = 0.11752016697 | 1 % / 5 % / 1 % | published_numerical |
| `pipe_poiseuille_3d` | Hagen-Poiseuille flow, 3D pipe R = 5 mm, L = 5 cm, Re = 10 | `dp_AB` on the axis [Pa]; `u_centreline`; `wall_force` [N]; `flow_rate_outlet` [m^3/s] | exact: 8 mu U (x_B - x_A) / R^2; 2 U; 8 pi mu U L; pi R^2 U | 2 % | analytical |

Files: `_common.py` (the `FlowProblem` base class, block-structured mesh builder, citations),
`poiseuille.py`, `cavity.py`, `cylinder.py`, `pipe.py`. Tests: `tests/test_problems_cfd.py`.

## Conventions

- 2D problems lie in the xy-plane. The backend extrudes one cell layer of `analysis.thickness`, which
  is 1 m in every 2D problem, so forces and flow rates are **per metre of depth**.
- `analysis.order` is the convection scheme order (1 upwind, 2 second order). The theoretical order
  of every QoI is the scheme order (`convergence_study` uses it for physics "flow").
- Options: `turbulence: laminar`, `residual_tol: 1e-8`, `max_iterations: 20000`. The iteration error
  then stays far below the discretization error that the GCI measures.
- Pressure drops published only as differences use the QoI type `point_difference`
  (`field(point) - field(point2)`).
- Reference meshes are **linear and fully structured** (transfinite, recombined blocks; quads in 2D,
  hexahedra in 3D). Every block count scales with 1/h and graded edges keep their first/last
  cell-size ratio, so the sequence is geometrically similar. `geom_order` is ignored: finite volumes
  use straight-sided cells. 2D cells are counter-clockwise (+z normal) for the extrusion. One
  `refine_steps` split (nested studies) gives exactly the same cell count as halving h.
- The STEP file for agents is the clean CAD (one face / one solid). The mesh blocks exist only in
  the reference geometry (`build_geometry`); `build_cad` gives the agent geometry.
- Pressure levels: open channels have p = 0 at the outlet. The cavity is closed, so its QoIs are
  velocities only.

## Sources and verification

Some hosts are blocked from the build session: arxiv.org, the FEATFLOW site
(wwwold.mathematik.tu-dortmund.de), featool.com, mfix.netl.doe.gov and the course copy of the
Schaefer-Turek paper (www2.karlin.mff.cuni.cz). Values from those pages were checked through
search-engine extracts, as noted below. raw.githubusercontent.com is reachable.

**Poiseuille channel and pipe.** These are exact solutions of the steady Navier-Stokes equations,
derived in the module docstrings (fully developed inflow, so u . grad u = 0 everywhere). The
textbook form is in F.M. White, *Viscous Fluid Flow*, 3rd ed. (2006), Sec. 3-3; the section title
is from memory. `tests/test_problems_cfd.py` checks the inlet expression against the exact field,
mu u'' = dp/dx, the mean velocity and the force balance (pressure drop x area = wall shear).

**Lid-driven cavity, Re = 100.** Values from Ghia, Ghia & Shin, J. Comput. Phys. 48 (1982) 387-411,
Table I (u on x = 0.5) and Table II (v on y = 0.5), Re = 100 column. They were checked against the
public transcription in PySPH (`pysph/examples/ghia_cavity_data.py`, fetched from
raw.githubusercontent.com):

| QoI | point | Ghia |
|---|---|---|
| `u_vcl_min` | (0.5, 58/128 = 0.4531) | -0.21090 |
| `v_hcl_max` | (30/128 = 0.2344, 0.5) | 0.17527 |
| `v_hcl_min` | (103/128 = 0.8047, 0.5) | -0.24533 |
| `u_centre` | (0.5, 0.5) | -0.20581 |

The tables print the grid coordinates (multiples of 1/128) to 4 decimals; the QoIs use the exact
grid points.

- **Ghia's Re = 100 extremes are known to be low in magnitude.** A published comparison table (found
  through a search-engine extract; the column labels were cut off, so the reading is our inference)
  lists Sahin & Owens (2003, IJNMF 42:57-77, fine-grid finite volume) as u_min = -0.21392 at
  y = 0.4598, v_max = 0.18088 at x = 0.2354 and v_min = -0.2566 at x = 0.8127. Ghia gives -0.21090,
  0.17527 and -0.24533, which are 1.4 %, 3.1 % and 4.4 % smaller.
- The extrema of the continuous profiles lie within 0.01 of Ghia's grid points, where the profiles are
  flat. Moving the point therefore changes the value by well under 0.5 %, which does not explain the
  difference.
- TOLERANCE_NOTE

**DFG 2D-1 (Schaefer & Turek 1996).** The problem definition follows M. Schaefer & S. Turek,
"Benchmark computations of laminar flow around a cylinder", Notes on Numerical Fluid Mechanics 52
(1996) 547-566:
- channel 2.2 x 0.41, cylinder D = 0.1 at (0.2, 0.2);
- inflow 4 U_m y (H - y) / H^2 with U_m = 0.3;
- nu = 1e-3, rho = 1, so Re = U_bar D / nu = 20;
- c_D = 2 F_D / (rho U_bar^2 D) with U_bar = 2 U_m / 3;
- dp between (0.15, 0.2) and (0.25, 0.2).

The high-accuracy reference values are those of the FEATFLOW benchmark page for 2D-1, under
"reference results" (computed with high-order methods): c_D = 5.57953523384, c_L = 0.010618948146,
dp = 0.11752016697. The page itself is blocked. The values were confirmed in three independent
search-engine extracts, of the FEATFLOW page itself and of arXiv:1807.07380 (Table 1) and
arXiv:2606.13219, which use them as references. The QoIs are forces per metre of depth:
F = c rho U_bar^2 D / 2 = 0.002 c N/m, so drag = 0.0111590705 N/m and lift = 2.12379e-5 N/m.
The paper's acceptance intervals are c_D in [5.57, 5.59], c_L in [0.0104, 0.0110] and
dp in [0.1172, 0.1176]; these intervals are quoted from memory.
- **Outflow.** The benchmark uses a "do-nothing" outflow. Here the outlet is p = 0 with zero-gradient
  velocity. The flow is fully developed 2 m downstream of the cylinder, where the two conditions
  coincide. Forces and dp are pressure differences, so the pressure level does not matter.
- **Lift tolerance.** Lift is about 0.2 % of the drag and needs a well-resolved boundary layer and wake.
  Its tolerance is 5 %, against the paper's interval of +-3 %.

## Validation (fixed solver `openfoam`)

VALIDATION_TABLE
