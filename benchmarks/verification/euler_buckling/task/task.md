# Linear buckling of a slender column

A steel column with a 10 x 10 mm square cross-section and length 400 mm along z, base z=0 fully fixed, top free. A reference load of 1000 N in -z (compression) is applied to the top face as a uniform traction. Compute the linear (eigenvalue) buckling load factor of the lowest mode.

## Units

mm, N, MPa (N/mm^2), tonne, s  (density in tonne/mm^3, acceleration in mm/s^2)

## Geometry

`geometry.step` (one solid, millimetres).

## Material

- steel: E = 210000, nu = 0.3, density = 7.85e-09 (linear elastic, isotropic)

## Boundary conditions

- **base**: fully fixed on face z=0

## Loads

- **top**: reference load 1000 N compression on top face z=400

## Analysis

Linear buckling (at least 2 modes).

## Quantities that will be evaluated

- `lambda1`: lowest buckling load factor

## What to submit

A directory containing `model.inp`: a complete CalculiX (ccx 2.21) input deck in the units above. The evaluator runs the fixed solver on it, appends output requests (U, RF, S, E), and measures the quantities above on your mesh. Optionally add `submission.json` with your own estimates and notes.
