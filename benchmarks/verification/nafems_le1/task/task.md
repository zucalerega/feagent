# NAFEMS LE1: elliptic membrane (3D thin plate)

NAFEMS benchmark LE1 (plane stress) modelled as a 3D plate, in mm. Quarter of an elliptic annular plate of total thickness 100 mm; only the half-thickness z in [0, 50] is modelled and z=0 is the plate's mid-plane of symmetry. Inner boundary (x/2000)^2 + (y/1000)^2 = 1, outer boundary (x/3250)^2 + (y/2750)^2 = 1. A uniform outward normal traction (tension) of 10 MPa acts on the outer elliptic face. Symmetry: ux=0 on x=0, uy=0 on y=0, uz=0 on z=0.

## Units

mm, N, MPa (N/mm^2), tonne, s  (density in tonne/mm^3, acceleration in mm/s^2)

## Geometry

`geometry.step` (one solid, millimetres).

## Material

- steel: E = 210000, nu = 0.3 (linear elastic, isotropic)

## Boundary conditions

- **sym_x**: ux = 0 on face x=0
- **sym_y**: uy = 0 on face y=0
- **mid**: uz = 0 on mid-plane z=0

## Loads

- **tension**: 10 MPa outward normal traction (i.e. a pressure of -10 MPa) on outer elliptic face

## Analysis

Linear static.

## Quantities that will be evaluated

- `syy_D`: sigma_yy averaged through the thickness along the hole edge at D (x=2000, y=0, 0<=z<=50)
- `reaction_x`: sum of x-reactions on x=0 (load-path check)

## What to submit

A directory containing `model.inp`: a complete CalculiX (ccx 2.21) input deck in the units above. The evaluator runs the fixed solver on it, appends output requests (U, RF, S, E), and measures the quantities above on your mesh. Optionally add `submission.json` with your own estimates and notes.
