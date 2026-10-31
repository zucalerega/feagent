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
- Options: `turbulence: laminar`, `residual_tol: 1e-8`, `max_iterations: 20000` (3000 for the two
  fully developed flows). The iteration error then stays far below the discretization error that the
  GCI measures.
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
- **Converged values here.** The fixed solver's mesh-converged values come from the 32/64/128 study
  (observed order 1.8-2.4, GCI <= 0.25 %, Richardson-extrapolated; a 256 x 256 level agrees to
  1e-4): u_vcl_min = -0.21392, v_hcl_max = 0.17951, v_hcl_min = -0.25379 and u_centre = -0.20914.
  - These are 1.4 %, 2.4 %, 3.4 % and 1.6 % larger in magnitude than Ghia's table, in the same
    direction as Sahin & Owens.
  - High-accuracy Re = 100 values often quoted in the literature agree with ours to 0.1 %. They are
    u_min = -0.2140424 at y = 0.4581, v_max = 0.1795728 at x = 0.2370 and v_min = -0.2538030 at
    x = 0.8104. **These values are from memory, not verified here.**
- **Tolerances.** A 1 % tolerance against Ghia would fail a converged solution. The tolerances are
  therefore 3 % (`u_vcl_min`, `u_centre`), 4 % (`v_hcl_max`) and 5 % (`v_hcl_min`): the table's own
  error plus about 1.5 % for the agent's discretization error.
  - Coarse solutions have smaller peaks, so they sit closer to Ghia; a 32 x 32 grid passes too.
  - A wrong setup (Re, lid speed, a wall condition) is off by far more than 5 %.

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
  velocity.
  - The two conditions coincide for fully developed flow. At x = 2.2 the wake has almost, but not
    quite, recovered: centreline u = 0.2975 against 0.3 for the developed profile (11816 cells).
  - Doubling the channel length to 4.2 changes c_D by -0.007 %, c_L by -0.2 % and dp by -0.02 %, so
    the outlet condition does not matter at the reference tolerances.
  - Forces and dp are pressure differences, so the pressure level does not matter either.
- **Lift tolerance.** Lift is about 0.2 % of the drag and needs a well-resolved boundary layer and wake.
  Its tolerance is 5 %, against the paper's interval of +-3 %.

## Validation (fixed solver `openfoam`)

Studies were run with `convergence_study(problem, order=2, backend="openfoam", verbose=True)`.
- Default `study_sizes`, ratio 2: 4 meshes for `poiseuille_channel`, 3 for the others.
- Each study was run twice: remeshed at every size, and `nested=True` (one split per level of the
  h0 mesh).
- Saved studies: `/tmp/cfd_bench/studies/<id>_openfoam_p2[_nested].json`.
- Runner: `/tmp/cfd_bench/run_study.py`.
- Backend: OpenFOAM v1912 `simpleFoam`, second-order convection (`bounded Gauss linearUpwind`),
  `Gauss linear corrected`, SIMPLEC, `residual_tol` 1e-8.
- The point values use the backend's continuous (cellPoint-style) probe.

How to read the table:
- Error % is relative to the reference. It uses the Richardson-extrapolated value when the study
  judges the extrapolation reliable, otherwise the finest value, marked "(finest)".
- p is the observed order; the theory is 2 (the scheme order) for every QoI.
- "both" means the remeshed and nested studies gave identical numbers. The structured grids split
  into exactly the remeshed grids.

| problem | seq | QoI | reference | values (coarse -> fine) | extrapolated | p | GCI % | error % | convergence | pass |
|---|---|---|---|---|---|---|---|---|---|---|
| `poiseuille_channel` | both | dp_AB | 0.3 | 0.293179, 0.298255, 0.299561, 0.29989 | 0.300001 | 1.99 | 0.046 | +0.000 | monotone | pass |
| `poiseuille_channel` | both | u_centreline | 0.015 | 0.0146591, 0.0149128, 0.0149781, 0.0149945 | 0.015 | 1.99 | 0.046 | +0.000 | monotone | pass |
| `poiseuille_channel` | both | wall_force_bottom (= top) | 0.003 | 0.00292445, 0.00297977, 0.00299468, 0.00299861 | 0.00300002 | 1.92 | 0.059 | +0.001 | monotone | pass |
| `poiseuille_channel` | both | flow_rate_outlet | 0.0001 | 0.000100781, 0.000100195, 0.000100049, 0.000100012 | 0.0001 | 2.00 | 0.015 | +0.000 | monotone | pass |
| `lid_driven_cavity_re100` | both | u_vcl_min | -0.2109 | -0.209337, -0.213024, -0.213744 | -0.213918 | 2.36 | 0.102 | +1.431 | monotone | pass (3 %) |
| `lid_driven_cavity_re100` | both | v_hcl_max | 0.17527 | 0.175387, 0.178686, 0.179348 | 0.179514 | 2.32 | 0.116 | +2.421 | monotone | pass (4 %) |
| `lid_driven_cavity_re100` | both | v_hcl_min | -0.24533 | -0.247644, -0.252042, -0.253291 | -0.253786 | 1.82 | 0.244 | +3.447 | monotone | pass (5 %) |
| `lid_driven_cavity_re100` | both | u_centre | -0.20581 | -0.205007, -0.208154, -0.208904 | -0.209139 | 2.07 | 0.140 | +1.617 | monotone | pass (3 %) |
| `cylinder_channel_dfg_2d1` | remeshed | drag | 0.0111591 | 0.0111245, 0.0111358, 0.0111405 | 0.0111438 | 1.26 | 0.038 | -0.136 | monotone (p_below_theory) | pass |
| `cylinder_channel_dfg_2d1` | nested | drag | 0.0111591 | 0.0111245, 0.0111399, 0.0111425 | 0.011143 | 2.58 | 0.006 | -0.144 | monotone | pass |
| `cylinder_channel_dfg_2d1` | remeshed | lift | 2.12379e-05 | 2.09929e-05, 2.1295e-05, 2.13133e-05 | 2.13145e-05 | 4.05 | 0.007 | +0.355 (finest) | monotone (p_above_theory) | pass |
| `cylinder_channel_dfg_2d1` | nested | lift | 2.12379e-05 | 2.09929e-05, 2.13182e-05, 2.13287e-05 | 2.13291e-05 | 4.95 | 0.002 | +0.428 (finest) | monotone (p_above_theory) | pass |
| `cylinder_channel_dfg_2d1` | remeshed | dp_front_rear | 0.11752 | 0.116144, 0.117112, 0.117373 | 0.11747 | 1.89 | 0.103 | -0.043 | monotone | pass |
| `cylinder_channel_dfg_2d1` | nested | dp_front_rear | 0.11752 | 0.116144, 0.117111, 0.117374 | 0.117471 | 1.88 | 0.104 | -0.041 | monotone | pass |
| `pipe_poiseuille_3d` | remeshed | dp_AB | 0.64 | 0.684597, 0.653932, 0.645101 | 0.64159 | 1.82 | 0.680 | +0.248 | monotone | pass |
| `pipe_poiseuille_3d` | nested | dp_AB | 0.64 | 0.684597, 0.656157, 0.647186 | 0.643124 | 1.69 | 0.785 | +0.488 | monotone | pass |
| `pipe_poiseuille_3d` | remeshed | u_centreline | 0.02 | 0.0201423, 0.0200519, 0.0200194 | 0.0200015 | 1.50 | 0.112 | +0.007 | monotone (p_below_theory) | pass |
| `pipe_poiseuille_3d` | nested | u_centreline | 0.02 | 0.0201423, 0.020068, 0.0200373 | 0.020016 | 1.29 | 0.133 | +0.080 | monotone (p_below_theory) | pass |
| `pipe_poiseuille_3d` | remeshed | wall_force | 0.000125664 | 0.000129074, 0.000127156, 0.000126295 | 0.000125605 | 1.17 | 0.683 | -0.047 | monotone (p_below_theory) | pass |
| `pipe_poiseuille_3d` | nested | wall_force | 0.000125664 | 0.000129074, 0.000126993, 0.000126372 | 0.000126112 | 1.77 | 0.257 | +0.357 | monotone | pass |
| `pipe_poiseuille_3d` | remeshed | flow_rate_outlet | 7.85398e-07 | 7.99922e-07, 7.89261e-07, 7.86379e-07 | 7.85329e-07 | 1.91 | 0.167 | -0.009 | monotone | pass |
| `pipe_poiseuille_3d` | nested | flow_rate_outlet | 7.85398e-07 | 7.99922e-07, 7.8927e-07, 7.86382e-07 | 7.85327e-07 | 1.91 | 0.168 | -0.009 | monotone | pass |

All 32 QoI judgements pass (16 QoIs x 2 sequences). Cavity tolerances are in brackets; the others are
1 % (Poiseuille, DFG drag and dp), 5 % (DFG lift) and 2 % (pipe).

Extra 4-level studies:
- `cylinder_channel_dfg_2d1` nested, up to 189 056 cells (18 min): drag 0.0111245, 0.0111399,
  0.0111425, 0.0111428. That is p = 3.4, extrapolated 0.0111428 (c_D = 5.5714), GCI 3e-6.
  Lift 2.12844e-5 (+0.22 %) on the finest mesh.
- `lid_driven_cavity_re100` remeshed, up to 256 x 256 (9 min): extrapolated u_vcl_min -0.21394,
  v_hcl_max 0.17954, u_centre -0.20914. These agree with the 3-level values to 1e-4.

These two studies predate the probe change. Drag and lift are force QoIs and are unaffected; the
cavity limits match the new 3-level study.

Runtimes are for OpenFOAM v1912 on one core of a shared 4-core VM. The fluid solve dominates;
meshing takes under 1 s.

| problem | cells | simpleFoam iterations | solve s per mesh | study total s |
|---|---|---|---|---|
| `poiseuille_channel` | 320 / 1280 / 5120 / 20480 | 154 / 145 / 252 / 477 | 0.2 / 0.4 / 1.5 / 11 | 14 |
| `lid_driven_cavity_re100` | 1024 / 4096 / 16384 | 186 / 566 / 1932 | 0.5 / 3.3 / 36 | 41 |
| `cylinder_channel_dfg_2d1` | 2954 / 11816 / 47264 | 151 / 256 / 839 | 0.8 / 4.5 / 65 | 72 |
| `pipe_poiseuille_3d` | 640 / 5120 / 40960 hexahedra | 150 / 145 / 274 | 0.4 / 1.4 / 26 | 29 |

### Findings

- **Poiseuille: textbook second order, and the errors are predicted exactly.**
  - With wall-cell height h/2, the discrete solution is the exact parabola plus a uniform shift. The
    face-centre inlet profile carries (1 + (h/H)^2 / 2) U H. The wall flux balance then gives
    dp error = (1 + (h/H)^2 / 2) / (1 + 2 (h/H)^2) - 1, which is -2.27 % at 8 cells across, as observed.
  - The flow rate is +(h/H)^2 / 2 (midpoint rule of the inlet profile).
  - The centreline probe interpolates linearly, which flattens the parabola's peak: -2.3 % at h0.
  - Extrapolated values match the exact ones to <= 7e-6.
- **Cavity:** order 1.8-2.4 and GCI <= 0.25 %. The converged values differ from Ghia's table by
  1.4-3.5 % (see Sources). That is why the tolerances are 3-5 % rather than 1 %. Coarse grids sit
  closer to Ghia: 32 x 32 is within 0.1-0.9 %.
- **DFG 2D-1:** the coarsest mesh (2954 cells) is already close: drag -0.31 %, lift -1.15 %,
  dp -1.17 %.
  - **Drag** is within 0.31 % of the reference on every mesh, but it converges to 0.0111428
    (c_D = 5.5714): 0.146 % low, outside the GCI band.
  - Ruled out as causes: the outlet (doubling the channel length changes c_D by -0.007 %) and the
    block layout (an O-grid box 0.24 wide instead of 0.2 moves the limit by about +0.02 %). The
    schemes are consistent: corrected Laplacian, linearUpwind, SIMPLEC.
  - The offset is inside the paper's interval [5.57, 5.59] and well inside the 1 % tolerance. Its
    origin (a consistency detail of the finite-volume wall-stress / force evaluation?) is open.
  - The pressure and viscous parts converge from opposite sides: c_D,p = 3.578 -> 3.602 and
    c_D,visc = 1.985 -> 1.966 on the first two meshes. That is why the remeshed sequence shows
    p = 1.26 and the nested one 2.58.
  - **Lift** converges fast (p 4-5, flagged `p_above_theory`) to +0.2-0.4 % of the reference. That
    is well inside its 5 % tolerance and the paper's +-3 % interval.
  - **dp** at the two cylinder surface points converges at order 1.9 to -0.04 % of the reference.
- **Pipe (3D):** the flow rate is order 2 and dp is order 1.7-1.8. u_centreline (1.3-1.5) and the
  wall force (1.2-1.8) converge more slowly on the curved wall.
  - Finest-mesh errors are <= 1.1 % (dp; the others <= 0.6 %). Extrapolated errors are <= 0.5 %.
  - The h0 section is a 16-gon, 2.5 % short in area, so the coarse pressure gradient is 7 % high.
  - After the fixed-velocity inlet with zero-gradient pressure, the solution has an odd-even
    pressure wiggle over the first ~3 (long, 5:1) cells. The probes therefore sit at x = L/2 and
    0.9 L. Probes at L/4 gave oscillatory dp sequences.
- **Iteration control:** `residual_tol` 1e-8 is reached in 150-1900 SIMPLE iterations.
  - In the fully developed flows the cross-stream velocity is zero to round-off. Its normalised
    residual stalls near 1e-7, so the first backend version ran to `max_iterations`.
  - The backend now accepts such a stalled component once the others converge. The two
    fully developed problems also cap `max_iterations` at 3000.
- **Backend issues found during this build (both fixed by the backend agent):**
  - Named fluid regions failed (`material_for(None)`).
  - The point probe was discontinuous at faces and vertices: nearest cell centre plus gradient,
    ties broken by cell order. On the cavity, identical 64 x 64 grids in two cell orders gave
    -0.213355 vs -0.213519. Benchmark probe points sit on grid vertices, so this showed up in the
    studies as 0.05-0.1 % scatter.
  - Reproducers: `/tmp/cfd_bench/repro/region_material.py`, `probe_ties.py`.
