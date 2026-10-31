# NAFEMS LE10: thick elliptic plate under pressure

NAFEMS standard benchmark LE10 (in mm). Quarter of an elliptic annular plate, 600 mm thick (z from -300 to +300). Inner boundary: ellipse (x/2000)^2 + (y/1000)^2 = 1; outer boundary: ellipse (x/3250)^2 + (y/2750)^2 = 1. Uniform normal pressure of 1 MPa on the upper face z=+300. Boundary conditions: uy=0 on the face y=0 (DCD'C'); ux=0 on the face x=0 (ABA'B'); ux=uy=0 on the outer elliptic face (BCB'C'); uz=0 on the line EE' where the outer face meets the mid-plane z=0 (the CAD face is split along that line).

## Units

mm, N, MPa (N/mm^2), tonne, s  (density in tonne/mm^3, acceleration in mm/s^2)

## Geometry

`geometry.step` (one solid, millimetres).

## Material

- steel: E = 210000, nu = 0.3 (linear elastic, isotropic)

## Boundary conditions

- **sym_y**: uy = 0 on face y=0
- **sym_x**: ux = 0 on face x=0
- **outer**: ux = uy = 0 on outer elliptic face BCB'C'
- **line_EE**: uz = 0 on line EE' (outer face at z=0)

## Loads

- **pressure**: 1 MPa normal pressure (pushing down) on upper face z=+300

## Analysis

Linear static.

## Quantities that will be evaluated

- `syy_D`: sigma_yy at point D = (2000, 0, 300)
- `reaction_z`: sum of z-reactions on line EE' (load-path check)

## What to submit

A directory containing `model.inp`: a complete CalculiX (ccx 2.21) input deck in the units above. The evaluator runs the fixed solver on it, appends output requests (U, RF, S, E), and measures the quantities above on your mesh. Optionally add `submission.json` with your own estimates and notes.
