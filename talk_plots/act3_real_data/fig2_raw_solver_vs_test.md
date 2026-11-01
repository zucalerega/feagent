# fig2_raw_solver_vs_test: "The raw solver misses reality: frictionless platens give out at 3.0 kN"

**Caption.** The out-of-the-box nonlinear simulation (CalculiX J2 plasticity + NLGEOM, frictionless platens)
plateaus at about 3 kN while the real part reaches 4.7 kN at 2.5 mm. Sticking platens (friction stops the prongs
splaying) reach the right strength; tilted or one-sided loading does not help.

**Key numbers** (force at 2.5 mm travel; mean |error| at the 5 checkpoints, no seating or compliance mapping):

| variant (4 mm P2 mesh) | F at 2.5 mm | mean error |
|---|---|---|
| measured | 4743 N | - |
| frictionless platens (baseline) | 3.0 kN (3009 N at 2.56 mm) | 33.2 % |
| sticking platens | 5.1 kN (5184 N at 2.56 mm) | 35.6 % |
| frictionless, elastic (no plasticity) | 5.1 kN | 30.4 % |
| frictionless, prong f17 at half travel | 2.8 kN | 29.1 % |
| frictionless, prong f34 at half travel | 2.8 kN | 29.5 % |
| frictionless, only prong f17 pushed | 1.5 kN | 50.3 % |

**Data.** /private/tmp/claude-501/-Users-aryan-Documents-Aryan-CS-scienceclawhack/29d391ed-69f3-4559-b365-a260e486ee85/scratchpad/nl_bc/<variant>/curve.csv (u_mm = platen travel, F_N), set up by /private/tmp/claude-501/-Users-aryan-Documents-Aryan-CS-scienceclawhack/29d391ed-69f3-4559-b365-a260e486ee85/scratchpad/nl_bc.py;
test: physical_tests/abc_00000065_compression/cload_compression.csv. Numbers in act3_data.json / act3_errors.csv.

**Caveats.**
- Raw: platen travel is read as crosshead travel. That is why the sticking curve scores WORSE (35.6 %) than
  frictionless (33.2 %) on the mean error: without the seating gap it is far too high early (99 % off at 0.5 mm)
  even though it matches the strength at 2.5 mm (8 % off). Do not quote "sticking fixes it" from this slide alone:
  the fix needs sticking AND seating (fig3).
- The elastic frictionless run also reaches 5.1 kN: the collapse of the frictionless model needs plasticity plus
  splay. Elastic is not a candidate (the dogbone shows the resin yields from about 12 MPa).
- The two tilt variants (f17 or f34 at half travel) differ by at most 1.6 % and share one label.
- All on one uniform 4 mm P2 mesh (the bonded run equals the h4 mesh run of the mesh sweep exactly).
