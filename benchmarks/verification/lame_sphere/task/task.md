# Thick-walled sphere under internal pressure

One octant (x, y, z >= 0) of a thick-walled steel sphere, inner radius 50 mm, outer radius 100 mm, centred at the origin. The three coordinate planes are planes of symmetry of the full sphere. An internal pressure of 100 MPa acts on the inner spherical surface.

## Units

mm, N, MPa (N/mm^2), tonne, s  (density in tonne/mm^3, acceleration in mm/s^2)

## Geometry

`geometry.step` (one solid, millimetres).

## Material

- steel: E = 210000, nu = 0.3, density = 7.85e-09 (linear elastic, isotropic)

## Boundary conditions

- **sym_x**: symmetry plane x=0: ux = 0 on face x=0
- **sym_y**: symmetry plane y=0: uy = 0 on face y=0
- **sym_z**: symmetry plane z=0: uz = 0 on face z=0

## Loads

- **p_inner**: internal pressure 100 MPa on inner sphere r=50

## Analysis

Linear static.

## Quantities that will be evaluated

- `ur_inner`: radial displacement of the inner surface at [28.868, 28.868, 28.868]
- `ur_outer`: radial displacement of the outer surface at [57.735, 57.735, 57.735]
- `hoop_inner`: tangential (hoop) stress at the inner surface
- `hoop_outer`: tangential (hoop) stress at the outer surface
- `reaction_x`: sum of x-reactions on x=0 (load-path check)

## What to submit

A directory containing `model.inp`: a complete CalculiX (ccx 2.21) input deck in the units above. The evaluator runs the fixed solver on it, appends output requests (U, RF, S, E), and measures the quantities above on your mesh. Optionally add `submission.json` with your own estimates and notes.
