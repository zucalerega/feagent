# Filleted angle bracket loaded through a pin hole

A steel angle bracket: base plate 100 x 80 x 10 mm (x, y, z) and an upright wall 100 x 10 x 90 mm along the edge y in [0, 10], joined by an 8 mm fillet on the inside corner. Two 11 mm bolt holes through the base at (25, 55) and (75, 55); a 16 mm pin hole through the wall at x=50, z=65 (axis along y). The bolt-hole surfaces are fully fixed (idealised rigid bolts). A force of 3000 N in -y (pulling the wall away from the base, opening the angle) is applied through the pin hole as a uniform traction on its cylindrical surface.

## Units

mm, N, MPa (N/mm^2), tonne, s  (density in tonne/mm^3, acceleration in mm/s^2)

## Geometry

`geometry.step` (one solid, millimetres).

## Material

- steel: E = 210000, nu = 0.3, density = 7.85e-09 (linear elastic, isotropic)

## Boundary conditions

- **bolts**: fully fixed on both bolt-hole surfaces

## Loads

- **pin_load**: 3000 N in -y as a uniform traction on pin hole (diameter 16 mm, axis along y through (50, *, 65))

## Analysis

Linear static.

## Quantities that will be evaluated

- `pin_disp_y`: mean y-displacement of the pin-hole surface
- `fillet_mises_max`: peak von Mises stress in the fillet, 10 <= x <= 90 (away from the side faces)
- `reaction_y`: total y-reaction at the bolts (load-path check)

## What to submit

A directory containing `model.inp`: a complete CalculiX (ccx 2.21) input deck in the units above. The evaluator runs the fixed solver on it, appends output requests (U, RF, S, E), and measures the quantities above on your mesh. Optionally add `submission.json` with your own estimates and notes.
