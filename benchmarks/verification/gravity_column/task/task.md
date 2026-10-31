# Prismatic column under body acceleration (self-weight)

A steel column with a 50 x 50 mm square cross-section, height 500 mm along z, base at z=0 (centred on the z axis). It stands on a frictionless rigid floor (no vertical displacement of the base, free horizontal motion) and is loaded only by a uniform body acceleration of 1e+06 mm/s^2 in -z (an inertial load of ~102 g). Rigid-body motion in the horizontal plane must be suppressed without restraining the free lateral (Poisson) deformation, e.g. ux=uy=0 at base corner [-25.0, -25.0, 0.0] and uy=0 at base corner [25.0, -25.0, 0.0].

## Units

mm, N, MPa (N/mm^2), tonne, s  (density in tonne/mm^3, acceleration in mm/s^2)

## Geometry

`geometry.step` (one solid, millimetres).

## Material

- steel: E = 210000, nu = 0.3, density = 7.85e-09 (linear elastic, isotropic)

## Boundary conditions

- **floor**: frictionless floor: uz = 0 on base z=0
- **pin_a**: ux = uy = 0 on the point [-25.0, -25.0, 0.0] mm
- **pin_b**: uy = 0 on the point [25.0, -25.0, 0.0] mm

## Loads

- **accel**: uniform body acceleration 1e+06 mm/s^2 in -z

## Analysis

Linear static.

## Quantities that will be evaluated

- `uz_top`: vertical displacement of the top centre [0, 0, 500.0]
- `szz_mid`: axial stress at mid-height on a side face [25.0, 0, 250.0]
- `reaction_z`: total vertical reaction at the base (load-path check)

## What to submit

A directory containing `model.inp`: a complete CalculiX (ccx 2.21) input deck in the units above. The evaluator runs the fixed solver on it, appends output requests (U, RF, S, E), and measures the quantities above on your mesh. Optionally add `submission.json` with your own estimates and notes.
