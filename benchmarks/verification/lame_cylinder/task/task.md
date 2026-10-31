# Thick-walled cylinder under internal pressure (plane strain)

A quarter (90 degree) segment of a thick-walled steel tube, inner radius 50 mm, outer radius 100 mm, axial length 20 mm (axis = z). The planes x=0 and y=0 are planes of symmetry of the full tube. Both end faces (z=0 and z=20) are held between frictionless rigid platens: no axial displacement, free in-plane motion (plane strain). An internal pressure of 100 MPa acts on the bore.

## Units

mm, N, MPa (N/mm^2), tonne, s  (density in tonne/mm^3, acceleration in mm/s^2)

## Geometry

`geometry.step` (one solid, millimetres).

## Material

- steel: E = 210000, nu = 0.3, density = 7.85e-09 (linear elastic, isotropic)

## Boundary conditions

- **sym_x**: symmetry plane x=0: ux = 0 on face x=0
- **sym_y**: symmetry plane y=0: uy = 0 on face y=0
- **end_z0**: frictionless platen: uz = 0 on end face z=0
- **end_zL**: frictionless platen: uz = 0 on end face z=20

## Loads

- **p_inner**: internal pressure 100 MPa on bore (inner cylindrical face r=50)

## Analysis

Linear static.

## Quantities that will be evaluated

- `ur_inner`: radial displacement of the bore at 45 deg, mid-length [35.35533905932738, 35.35533905932738, 10.0]
- `ur_outer`: radial displacement of the outer surface at [70.71067811865476, 70.71067811865476, 10.0]
- `hoop_inner`: hoop stress at the bore [35.35533905932738, 35.35533905932738, 10.0]
- `hoop_outer`: hoop stress at the outer surface [70.71067811865476, 70.71067811865476, 10.0]
- `axial_mid`: axial (plane-strain) stress mid-wall [53.033008588991066, 53.033008588991066, 10.0]
- `reaction_x`: sum of x-reactions on x=0 (load-path check)

## What to submit

A directory containing `model.inp`: a complete CalculiX (ccx 2.21) input deck in the units above. The evaluator runs the fixed solver on it, appends output requests (U, RF, S, E), and measures the quantities above on your mesh. Optionally add `submission.json` with your own estimates and notes.
