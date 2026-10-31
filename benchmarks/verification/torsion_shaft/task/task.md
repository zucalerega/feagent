# Solid circular shaft in torsion

A solid steel shaft of radius 10 mm and length 100 mm along z. The face z=0 is fully fixed. A pure torque of 100000 N*mm about the shaft axis (+z, right-hand rule) is applied to the free end face z=100, distributed as the Saint-Venant shear traction (proportional to the distance from the axis); the face is free to deform otherwise.

## Units

mm, N, MPa (N/mm^2), tonne, s  (density in tonne/mm^3, acceleration in mm/s^2)

## Geometry

`geometry.step` (one solid, millimetres).

## Material

- steel: E = 210000, nu = 0.3, density = 7.85e-09 (linear elastic, isotropic)

## Boundary conditions

- **root**: fully fixed (all displacement components zero) on face z=0

## Loads

- **torque**: torque 100000 N*mm about +z (Saint-Venant shear traction) on end face z=100

## Analysis

Linear static.

## Quantities that will be evaluated

- `ut_mid`: tangential displacement of the surface point [10.0, 0, 50.0]
- `ut_3q`: tangential displacement of the surface point [0, 10.0, 75.0]
- `mises_mid`: von Mises stress at the surface point [10.0, 0, 50.0]
- `reaction_torque`: reaction moment about z at the fixed face (load-path check)

## What to submit

A directory containing `model.inp`: a complete CalculiX (ccx 2.21) input deck in the units above. The evaluator runs the fixed solver on it, appends output requests (U, RF, S, E), and measures the quantities above on your mesh. Optionally add `submission.json` with your own estimates and notes.
