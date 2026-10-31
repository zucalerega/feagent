# Natural frequencies of a cantilever bar

The steel bar of the cantilever case (200 x 10 x 20 mm, x along the length, 10 mm in y, 20 mm in z), fully fixed at x=0. Compute the lowest natural frequencies (Hz).

## Units

mm, N, MPa (N/mm^2), tonne, s  (density in tonne/mm^3, acceleration in mm/s^2)

## Geometry

`geometry.step` (one solid, millimetres).

## Material

- steel: E = 210000, nu = 0.3, density = 7.85e-09 (linear elastic, isotropic)

## Boundary conditions

- **root**: fully fixed on face x=0

## Analysis

Natural frequencies (at least 6 modes).

## Quantities that will be evaluated

- `f1`: first natural frequency [Hz] (bending in y)
- `f2`: second natural frequency [Hz] (bending in z)

## What to submit

A directory containing `model.inp`: a complete CalculiX (ccx 2.21) input deck in the units above. The evaluator runs the fixed solver on it, appends output requests (U, RF, S, E), and measures the quantities above on your mesh. Optionally add `submission.json` with your own estimates and notes.
