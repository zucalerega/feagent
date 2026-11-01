# Mesh-system comparison

Same two parts, same loads and material for every system (experiments/mesh_systems/CONTRACT.md). Linear: feacore/skfem tet10, bottom fixed, total force along -axis on top spread per area, Formlabs Alpha Fast Resin from the dogbone (E 1879.8 MPa, nu 0.35 assumed, UTS 50.48 MPa). Compliance = 2U/F at 100 N (work-equivalent platen travel; equals the area-mean axial travel of the loaded face). Stresses are scaled to F_full; 'robust' = 99.9th percentile of the volume-weighted cell von Mises (the peak is singular at the platen edges and mesh-dependent). Spread = (max - min) / median.

## abc_00000003 (axis z)

F_full = **4558 N** (robust von Mises = 87 % of the UTS, median over systems; per-system values 17.56 % apart). Video deformation x3.

| system | lin. DOFs | lin. nodes | lin. gates fail / warn | build s | compliance um @100 N | dev % | u_axis top um @F_full | mean abs(u) top um | peak vM MPa | robust vM MPa | dev % | solve s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| harness_auto | 385,932 | 128,644 | - / thin_walls | 52.4 | 18.229 | +0.34 | 830.9 | 839.7 | 93.3 | 44.7 | +1.8 | 24.2 |
| harness_expert | 374,859 | 124,953 | - / - | 51.5 | 18.240 | +0.40 | 831.4 | 840.2 | 94.7 | 44.7 | +1.7 | 29.8 |
| ace_sizing | 87,591 | 29,197 | - / thin_walls | 8.45 | 18.005 | -0.89 | 820.7 | 829.3 | 124.7 | 37.9 | -13.6 | 4.5 |
| gold_nested | 394,515 | 131,505 | sicn / gamma | 62.1 | 18.106 | -0.34 | 825.3 | 834.0 | 154.1 | 43.3 | -1.4 | 25.2 |
| netgen | 154,587 | 51,529 | - / thin_walls | 7.81 | 18.024 | -0.79 | 821.6 | 830.2 | 69.3 | 39.0 | -11.1 | 6.7 |
| harness_hex | 361,161 | 120,387 | - / thin_walls | 33 | 18.229 | +0.34 | 830.9 | 839.7 | 97.9 | 44.5 | +1.4 | 24.6 |

Spread across systems:

| quantity | median | min | max | spread % | n |
|---|---|---|---|---|---|
| compliance um @100 N | 18.17 | 18.01 | 18.24 | 1.29 | 6 |
| u_axis top um @F_full | 828.1 | 820.7 | 831.4 | 1.29 | 6 |
| mean abs(u) top um @F_full | 836.8 | 829.3 | 840.2 | 1.3 | 6 |
| peak von Mises MPa | 96.3 | 69.33 | 154.1 | 88 | 6 |
| robust von Mises MPa | 43.93 | 37.95 | 44.74 | 15.5 | 6 |
| linear DOFs | 368,010 | 8.759e+04 | 394,515 | 83.4 | 6 |
| linear solve s | 24.39 | 4.54 | 29.83 | 104 | 6 |
| linear mesh build s | 42.25 | 7.81 | 62.14 | 129 | 6 |
| crush peak force N (done runs) | 6883 | 6267 | 8364 | 30.5 | 6 |
| crush travel at peak mm (done runs) | 1.071 | 0.962 | 1.369 | 38 | 6 |
| crush travel at separation mm (done runs) | 1.306 | 1.005 | 1.393 | 29.8 | 4 |

Nonlinear meshes and crush runs (CalculiX J2 + erosion; running runs show the state so far):

| system | nl. nodes | nl. cells | element | nl. gates fail / warn | status | increments | peak force N | travel at peak mm | travel at separation mm | first failure travel mm | first failure height (0 bottom, 1 top) | first failure centroid mm | wall h |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| harness_auto | 23,926 | 12,489 | C3D10 | - / thin_walls | done (separated: no remaining fragment connects top, bottom) | 266 | 7210 | 1.148 | 1.24 | 0.948 | 0.0088 | [6.73, -0.42, 0.54] | 0.43 |
| harness_expert | 36,425 | 19,453 | tet2 | - / - | done (force 32.23 < 0.01 x peak 6267 over 3 increments) | 141 | 6267 | 0.962 | - | 0.819 | 0.0061 | [7.51, 3.73, 0.38] | 1.00 |
| ace_sizing | 29,128 | 16,007 | tet2 | - / thin_walls | done (force 60.17 < 0.01 x peak 6556 over 3 increments) | 275 | 6556 | 0.995 | - | 0.95 | 0.0072 | [4.0, 5.27, 0.44] | 3.08 |
| gold_nested | 39,841 | 23,774 | tet2 | area_fidelity, sicn / gamma | done (separated: no remaining fragment connects top, bottom) | 106 | 6267 | 0.984 | 1 | 0.701 | 0.017 | [12.94, -1.81, 1.05] | 0.68 |
| netgen | 28,537 | 15,298 | tet2 | - / thin_walls | done (separated: no remaining fragment connects top, bottom) | 119 | 8044 | 1.332 | 1.39 | 1.35 | 0.015 | [1.35, -0.06, 0.9] | 0.56 |
| harness_hex | 38,417 | 29,340 | hex8 (C3D8) | - / - | done (separated: no remaining fragment connects top, bottom) | 146 | 8364 | 1.369 | 1.37 | 1.01 | 0.0082 | [8.8, 4.24, 0.5] | 0.57 |

## abc_00000065 (axis y)

F_full = **2394 N** (robust von Mises = 87 % of the UTS, median over systems; per-system values 4.9 % apart). Video deformation x1.

| system | lin. DOFs | lin. nodes | lin. gates fail / warn | build s | compliance um @100 N | dev % | u_axis top um @F_full | mean abs(u) top um | peak vM MPa | robust vM MPa | dev % | solve s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| harness_auto | 292,155 | 97,385 | - / - | 60.9 | 36.989 | +0.31 | 885.6 | 2560.1 | 170.1 | 44.1 | +0.5 | 20.7 |
| harness_expert | 396,189 | 132,063 | - / - | 86.8 | 36.981 | +0.29 | 885.4 | 2559.5 | 169.5 | 44.2 | +0.6 | 30.3 |
| ace_sizing | 119,820 | 39,940 | sicn / thin_walls | 36.2 | 36.719 | -0.42 | 879.1 | 2566.3 | 134.0 | 42.2 | -3.9 | 5.7 |
| gold_nested | 390,288 | 130,096 | sicn / gamma | 31.8 | 36.768 | -0.29 | 880.3 | 2558.0 | 107.9 | 42.5 | -3.2 | 27.8 |
| netgen | 179,229 | 59,743 | - / thin_walls | 9.25 | 36.707 | -0.45 | 878.8 | 2567.1 | 112.8 | 43.7 | -0.5 | 7.9 |
| harness_hex | 387,117 | 129,039 | - / - | 44.1 | 36.990 | +0.31 | 885.6 | 2559.4 | 171.0 | 44.3 | +0.8 | 28.1 |

Spread across systems:

| quantity | median | min | max | spread % | n |
|---|---|---|---|---|---|
| compliance um @100 N | 36.87 | 36.71 | 36.99 | 0.767 | 6 |
| u_axis top um @F_full | 882.8 | 878.8 | 885.6 | 0.767 | 6 |
| mean abs(u) top um @F_full | 2560 | 2558 | 2567 | 0.354 | 6 |
| peak von Mises MPa | 151.7 | 107.9 | 171 | 41.6 | 6 |
| robust von Mises MPa | 43.92 | 42.21 | 44.29 | 4.75 | 6 |
| linear DOFs | 339,636 | 119,820 | 396,189 | 81.4 | 6 |
| linear solve s | 24.25 | 5.72 | 30.27 | 101 | 6 |
| linear mesh build s | 40.17 | 9.25 | 86.8 | 193 | 6 |
| crush peak force N (done runs) | 3252 | 3215 | 3863 | 19.9 | 6 |
| crush travel at peak mm (done runs) | 1.208 | 1.135 | 1.424 | 23.9 | 6 |
| crush travel at separation mm (done runs) | 1.489 | 1.404 | 1.874 | 31.5 | 6 |

Nonlinear meshes and crush runs (CalculiX J2 + erosion; running runs show the state so far):

| system | nl. nodes | nl. cells | element | nl. gates fail / warn | status | increments | peak force N | travel at peak mm | travel at separation mm | first failure travel mm | first failure height (0 bottom, 1 top) | first failure centroid mm | wall h |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| harness_auto | 33,824 | 19,076 | C3D10 | - / thin_walls | done (separated: no remaining fragment connects top, bottom) | 204 | 3234 | 1.160 | 1.47 | 0.938 | 0.89 | [22.47, 25.29, -0.01] | 0.47 |
| harness_expert | 39,608 | 22,650 | tet2 | - / - | done (separated: no remaining fragment connects top, bottom) | 166 | 3260 | 1.198 | 1.4 | 0.938 | 0.89 | [22.58, 25.23, 0.69] | 1.26 |
| ace_sizing | 31,518 | 17,848 | tet2 | sicn / thin_walls | done (separated: no remaining fragment connects top, bottom) | 200 | 3215 | 1.364 | 1.51 | 0.829 | 0.89 | [22.16, 25.46, -6.93] | 1.04 |
| gold_nested | 39,234 | 23,560 | tet2 | sicn / gamma | done (separated: no remaining fragment connects top, bottom) | 147 | 3355 | 1.217 | 1.44 | 0.922 | 0.9 | [6.06, 25.85, 2.1] | 0.74 |
| netgen | 26,395 | 14,751 | tet2 | - / thin_walls | done (separated: no remaining fragment connects top, bottom) | 129 | 3863 | 1.424 | 1.87 | 1.38 | 0.89 | [-16.91, 25.26, 0.72] | 0.47 |
| harness_hex | 56,656 | 44,776 | hex8 (C3D8) | - / - | done (separated: no remaining fragment connects top, bottom) | 129 | 3245 | 1.135 | 1.58 | 0.933 | 0.89 | [22.42, 25.32, 0.73] | 1.00 |

Chart: RESULTS.png. Data: comparison.json. Rerun: `python experiments/physical/compare.py --meshsys ROOT --study OUT`.
