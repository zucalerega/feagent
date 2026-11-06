# Analyst's handbook (general rules)

These rules apply to any part and any load case. Cite a rule by its id (e.g. A2) in your report's
`rules_used` when it shaped a decision.

## A. From design data to the model

- **A1.** Model every interface the design data describes, and nothing else. Each support and load must trace to
  a sentence of the package; quote or paraphrase it in the report's idealization.
- **A2.** A face that is bonded, brazed, shrink-fitted, bolted flat or gripped by a rigid body "with no slip and no
  separation" is held over that whole face: all three displacement components zero.
- **A3.** A face that bears on a smooth or greased plate, rides on a guide or linear bearing, or is otherwise stopped
  in one direction only: zero displacement in that one direction, free in the others. Read the face's normal
  to know which component that is.
- **A4.** A total force "spread evenly" or "with equal force per unit area" over a face is a total-force load on
  exactly that face (or the same force divided by the area, as a traction). Keep the stated direction and sign
  in the CAD frame.
- **A5.** A contact, fit, fluid or bladder pressure is a pressure load (normal to the surface everywhere, positive
  pushing into the part) on exactly the faces the package names. Note what it says is NOT touched (end faces,
  the other pin, a vented bore) and leave those free.
- **A6.** The CAD is in metres. Convert every input to SI before solving: mm to m, MPa and GPa to Pa, kN to N.
- **A7.** Do not add supports the design data does not state. If the stated supports leave a rigid-body motion
  free, re-read the package before inventing one.
- **A8.** After the first solve, check equilibrium: the reactions must balance the applied loads, and the applied
  total must equal the stated load (a force's magnitude and direction; a pressure times the area on a flat face).

## B. Finding the faces

- **B1.** Map each interface to faces with inspect_geometry and find_faces (type, normal, axis, radius,
  position), and confirm the choice with view_geometry before you solve.
- **B2.** "The outer cylindrical surface", "the pin surface", "the bore wall" is usually one cylinder face. End
  faces, shoulders, chamfers, rounds and blends are separate faces, and are not part of it unless the package
  says so.
- **B3.** When a description could match several faces, use every detail it gives: which way the face faces, its
  position along each axis, its diameter, and what it is next to.

## C. Meshing

- **C1.** Use quadratic (order 2) tetrahedra on parts with curved faces.
- **C2.** Start from a global size of about 1/15 to 1/20 of the part's largest dimension, refined locally at
  re-entrant edges, small rounds, thin walls, and where loads and supports end.
- **C3.** A mesh an answer rests on must have folded_cells = 0 and minimum quality of at least 0.1. Let local sizes
  grow back over a radius of at least 2 to 3 times the local size.
- **C4.** Put nodes where the ZZ hotspots are. Halving the size everywhere costs about 8 times the nodes in 3D.

## D. Convergence studies

- **D1.** Every reported value rests on at least three meshes with the same setup, refined systematically
  (refine_mesh with a factor of 1.3 to 1.6), analysed with the study tool.
- **D2.** Monotone convergence on three well-spaced meshes (ratios of 1.3 or more) with a plausible order: report
  the extrapolated value, with the GCI as its uncertainty. Plausible means near the theoretical order, or the lower
  order a nearby singular edge allows (energies and mean displacements then converge at about order 1 to 2).
- **D3.** Oscillating values, or an implausible order (well above theory, or below about 0.7): report the finest
  value, with the largest of the spread of the last three values, the GCI and the conservative GCI as its
  uncertainty. Values that only scatter by a few tenths of a percent without a trend are converged within that
  scatter: use the spread.
- **D4.** Global quantities (strain energy, reactions, mean displacements, volume averages) converge even where
  peak stresses do not. Prefer them for decisions.
- **D5.** Make the finest mesh as fine as the node cap allows. The coarsest mesh of the study must still resolve
  the features (about two quadratic elements through a wall).

## E. Singularities

- **E1.** In linear elasticity the stress at a re-entrant edge, at the rim of a rigid support, or at the edge of a
  load patch grows without bound under refinement: it has no converged value.
- **E2.** If a peak stress keeps growing under refinement (divergent, or an order far below theory) and the
  feedback places it at such an edge, report it as singular, say where it is, and give no number as converged.
  Recommend how to assess it instead (a more realistic support model, the stress at a distance, a fillet).
- **E3.** Do not call a quantity singular only because it is a stress. A peak in a smooth region that converges
  is reported with its value and uncertainty.

## F. The report

- **F1.** Answer every question with a value from your own solves, in SI units, with the relative uncertainty
  from your study, or say why no converged value exists.
- **F2.** final_solve is the solve on your finest mesh with the final setup; study_solves are the solves of the
  study behind your answers.
- **F3.** State the assumptions you made where the design data is silent, and say what the design team should
  do with the results.
