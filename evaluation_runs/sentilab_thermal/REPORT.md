# FEAgent independent slab verification for SentiLab

Completed locally with the existing FEAgent `feacore.solver.solve`, scikit-fem 3D heat backend,
SciPy SuperLU direct solve. No remote AI model was used. No core solver code was changed.
Specification: GB10 Host's reply to message #57170, 2026-10-31 20:54:14 UTC.

## Model and exact answer

Full 20 x 20 x 1 mm slab; all inputs converted to SI. Conductivity 150 W/(m K),
uniform source 25,000,000 W/m3 (10 W total). Top and sides have natural zero flux.
Case A: bottom fixed at 25 C. Case B: bottom convection h=1000 W/(m2 K), ambient 25 C.
With L=0.001 m, T(z)=T_bottom+q*z*(2*L-z)/(2*k). Case A T_bottom=25 C;
case B T_bottom=25+10/(1000*0.02**2)=50 C.
Exact peaks: A 25.083333333 C, B 50.083333333 C.
Exact volume means: A 25.055555556 C, B 50.055555556 C. Outward bottom heat: 10 W.

## Results

|Case|Basis|Grid subdivisions|DOFs|Peak C|Mean C|RMS field error K|Conservative bottom W|Raw gradient bottom W|Relative residual|Assembly+solve+recovery s|
|---|---|---|---|---|---|---|---|---|---|---|
|A|P1|4 x 4 x 2|75|25.103790549|25.052610897|0.00521|10.000000000|7.875914|2.8e-16|0.011|
|B|P1|4 x 4 x 2|75|50.094567844|50.052620285|0.0038|10.000000000|7.886526|1.1e-13|0.005|
|A|P2|4 x 4 x 2|405|25.083333333|25.055555556|3.83e-14|10.000000000|10.000000|8.1e-16|0.010|
|B|P2|4 x 4 x 2|405|50.083333333|50.055555556|1.71e-11|10.000000000|10.000000|1.1e-12|0.010|
|A|P1|8 x 8 x 4|405|25.093141330|25.054751205|0.00168|10.000000000|8.837612|6.1e-16|0.015|
|B|P1|8 x 8 x 4|405|50.088718850|50.054755545|0.00115|10.000000000|8.848886|5e-13|0.014|
|A|P2|8 x 8 x 4|2601|25.083333333|25.055555556|1.5e-13|10.000000000|10.000000|1.7e-15|0.078|
|B|P2|8 x 8 x 4|2601|50.083333333|50.055555555|8.63e-11|10.000000000|10.000000|4.2e-12|0.068|
|A|P1|16 x 16 x 8|2601|25.087616696|25.055345232|0.000499|10.000000000|9.391418|1.2e-15|0.115|
|B|P1|16 x 16 x 8|2601|50.085865482|50.055346618|0.000344|10.000000000|9.399721|1.7e-12|0.106|
|A|P2|16 x 16 x 8|18513|25.083333333|25.055555556|3.67e-13|10.000000000|10.000000|3e-15|4.176|
|B|P2|16 x 16 x 8|18513|50.083333333|50.055555555|2.21e-10|10.000000000|10.000000|1.5e-11|4.993|

Grid cells are split into 6 affine tetrahedra. Each refinement halves all edge lengths;
in-plane grid spacing is 10 times through-thickness spacing. P1 and P2 share the same
straight geometry. P2 adds edge DOFs. Meshes, case JSON, field VTU, 101-point central
z-profiles, CSV results and software versions are included beside this report.

## Checks and interpretation

All 12 solves pass volume, free-equation residual and global heat-balance checks.
P1 RMS field error decreases by more than 2.5x on each refinement (approximately second
order). P2 represents the exact quadratic profile, so its remaining errors are numerical
roundoff; P2 is a patch check, not an estimate of convergence for complicated chip packages.
Peak is the maximum FE nodal temperature; mean and RMS error use volume quadrature.

Conservative bottom heat uses negative Dirichlet reaction for A and the integrated
h*(T-T_ambient) Robin law for B. Independently, raw heat flux integrates
(-k*grad(T)).n using element gradients on the bottom, without flux recovery. The coarse
P1 gradient need not give 10 W even when conservative heat balance closes; it converges
under refinement. P2 gives the exact flux to numerical precision.

Residual is ||(K*T-f)_free|| / ||(f-K*T_prescribed)_free||. Timing in the table includes
assembly, main solve and flux recovery; `linear_solve_s_reported` in CSV is the main
factorization/solve time parsed from the solver diagnostic (rounded there). Mesh generation,
postprocessing and output are excluded. Timings are single local runs, not performance claims.

This verifies uniform-source units, flux sign and convection mapping on this slab.
It does not validate chiplet contact resistances, multilayer packages, measured material
properties, nonlinear heat transfer, or physical measurements. Next useful collaboration:
reuse this benchmark in the receiving team's solver, then agree on one shared package case.
