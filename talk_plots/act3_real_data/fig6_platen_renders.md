# fig6_platen_renders: "Platen friction carries the strength: without it the prongs splay"

**Caption.** The teammates' CalculiX renders on a 41k-node mesh: with frictionless platens the prongs splay and
yield at the nut pockets, and the force tops out at 2.92 kN; with sticking platens the part keeps carrying load
(4.79 kN at 2.2 mm platen travel, 4.90 kN at the test's peak travel).

**Key numbers** (physical_tests/abc_00000065_compression/ccx_ramp/summary.json, "team" material = the dogbone
fit, E 1879.8 MPa): team_slide F_max 2919 N; team_stick F at the test's peak travel 4903 N, F_max 5352 N at
3.0 mm. Plastic strain is already present at the first increment (0.1 mm travel, 0.33 kN slide / 0.36 kN stick), at the hex nut pocket corners (faces f8 with f38/f39 or f41/f42).

**Data.** physical_tests/abc_00000065_compression/ccx_ramp/render_team_slide.png and render_team_stick.png
(copied here unchanged; the figure crops their middle rows), summary.json.

**Caveats.**
- These are the teammates' ramps (scripts/ccx_compression_ramp.py), not the frozen scoring setup: the base is
  held, there is no seating/compliance mapping, and the travel is platen travel.
- The panel subtitles inside the images are the renderer's own labels (small at slide size).
