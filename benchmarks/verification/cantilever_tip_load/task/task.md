# Cantilever with an end shear load

A steel bar 200 mm long (x), cross-section 10 mm (y) x 20 mm (z), centred on the x axis. The face x=0 is fully fixed. A total force of 1000 N in -z is applied to the free end face x=200 as a uniformly distributed traction.

## Units

mm, N, MPa (N/mm^2), tonne, s  (density in tonne/mm^3, acceleration in mm/s^2)

## Geometry

`geometry.step` (one solid, millimetres).

## Material

- steel: E = 210000, nu = 0.3, density = 7.85e-09 (linear elastic, isotropic)

## Boundary conditions

- **root**: fully fixed on face x=0

## Loads

- **tip**: 1000 N in -z as a uniform traction on end face x=200

## Analysis

Linear static.

## Quantities that will be evaluated

- `tip_deflection`: mean vertical displacement of the end face
- `bending_stress_mid`: bending stress at the top surface, mid-span [100.0, 0, 10.0]
- `reaction_z`: vertical reaction at the root (load-path check)

## What to submit

A directory containing `model.inp`: a complete CalculiX (ccx 2.21) input deck in the units above. The evaluator runs the fixed solver on it, appends output requests (U, RF, S, E), and measures the quantities above on your mesh. Optionally add `submission.json` with your own estimates and notes.
