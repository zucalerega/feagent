# Teacher brief: the C-bracket compression test (abc_00000065_compression)

What the analysis of the Instron test and of CalculiX elastoplastic ramps found. It is given to the
teacher (and, through the teacher's revisions, to the agents) as the experiment-analysis step a
scaled-up teacher would run itself: many full nonlinear solves of setup and mesh variants against
the measured curve. Numbers marked (*) are recomputed by `scripts/nl_baselines.py` against
`cload_compression.csv` and kept in `nonlinear_spec.json`.

## The test

- Formlabs Alpha Fast resin, 2 min post-cure; the part stands on its base face f33 on the lower
  platen of an Instron 68TM-50; the upper platen pushes the two prong tops f17 and f34 down (-y).
- Displacement is crosshead travel (no extensometer). Loaded to a peak of about 4.8 kN at about
  2.5 mm, then cracking (saw-tooth load drops) to the break at 8.66 mm.
- The curve is soft up to about 0.5 mm / 880 N (about 1.7 kN/mm), then about 2.6-2.7 kN/mm, then
  softens continuously to the peak: plasticity before the peak, cracking after it.

## The physics that reproduces it (the frozen scoring setup)

- **Sticking platens.** With frictionless platens the prongs splay outward and the part collapses
  at 2.8-2.9 kN (33 % mean error over the checkpoints). With the prong tops and the base held in all
  components by the platens (friction stops the splay), the force keeps rising to about 4.9 kN at
  the test's peak travel, like the test. Platen friction carries the strength.
- **Material:** J2 plasticity with isotropic hardening from the same resin's dogbone test
  (`experiments/material/alpha_fast_dogbone1.json`: E 1.88 GPa apparent, yield from about 12 MPa,
  hardening to about 53 MPa), large deformation (NLGEOM).
- **Crosshead mapping:** the crosshead reads the part's shortening plus a seating gap plus the
  load train's deflection: u_crosshead = u_part + delta + c * F, with delta about 0.2335 mm and
  c about 0.290e-4 mm/N (about 34.5 kN/mm) (*). Both were fitted once on a 41k-node mesh and are
  frozen; the soft start (one prong touching first, seating) and the frame compliance are thereby
  part of the setup, not of the mesh.
- With this setup the converged (finest-mesh) simulation is about 1.1 % from the test (*) over the
  five checkpoints 0.5 / 1.0 / 1.5 / 2.0 / 2.5 mm: that is the floor no mesh can beat; the rest is
  model error (crude seating, full sticking, the dogbone's crosshead-based calibration).

## What the mesh controls, and where

Mesh calls (h, size, radius, point, box) are in METRES (uniform 4 mm = h 0.004; x 12-31 mm = box x 0.012..0.031); this brief and the nonlinear hotspots (xyz_mm) give mm.

- Every mesh sits ABOVE the test at every checkpoint, and refining moves the curve down toward it:
  error vs the test = (mostly) the mesh's discretization error. A coarse mesh is too stiff and too
  strong because it under-resolves the plastic zones.
- Mean error vs the test by mesh (same solver and setup) (*):

  | mesh | nodes | error |
  |---|---|---|
  | gmsh default sizes (the automesher baseline) | 2937 | 16.6 % |
  | the swarm's best bc0 mesh (refined for a one-sided load on f7) | 6430 | 4.6 % |
  | uniform 4 mm | 9506 | 2.0 % |
  | uniform 2.5 mm | 24194 | 1.2 % |
  | uniform 2 mm | 41186 | 1.1 % |

- **Where it yields:** plastic flow starts at about 0.4 kN at the sharp corners of the prongs' hex
  nut pockets (faces f8 with f38/f39 and f41/f42), on BOTH prongs (the load is symmetric about
  x = 0, unlike bc0's load on f7). At the peak the largest tension, 47-51 MPa, is there, with
  40-50 MPa bands on the lower inner surface of the bore. These are the regions whose resolution
  sets the curve after yield; the bulk of the base and the outer arms carry little plastic strain.
- The bc0 lessons (boxes around the f7 side, x 12-31 mm) are for a different, one-sided load; they
  under-resolve the left prong here. Symmetric refinement of both prongs' pocket corners and the bore
  bands is the obvious first step; how fine, how far the grading reaches and what h the bulk can take
  is what the runs must find, at the fewest DOFs.

## Scoring

Error = mean |F_sim - F_test| / F_test over the five checkpoints (crosshead travel). Score as in
field mode against the automesher: g = e_auto / e, A = (g/2)^2 below 2 and g/2 above, divided by
the DOF ratio dofs / dofs_auto. A 2 % mesh at 6.4k nodes scores about 1.9; uniform 4 mm about 1.3;
the swarm's bc0 mesh about 0.8.
