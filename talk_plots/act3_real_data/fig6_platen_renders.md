# fig6_platen_renders: "Platen friction carries the strength: without it the prongs splay"

**Caption.** The teammates' CalculiX renders on a 41k-node mesh: with frictionless platens the prongs splay and
yield at the nut pockets, and the force tops out at 2.92 kN; with sticking platens the part
keeps carrying load (4.90 kN at the test's peak travel). This is why the
learned test conditions use sticking platens.

**Key numbers** (physical_tests/abc_00000065_compression/ccx_ramp/summary.json, dogbone material): team_slide
F_max 2919 N; team_stick F at the test's peak travel 4903 N, F_max
5352 N at 3.0 mm.

**Data.** physical_tests/abc_00000065_compression/ccx_ramp/render_team_slide.png and render_team_stick.png
(copied here unchanged; the figure crops their middle rows), summary.json.

**Caveats.**
- These are the teammates' ramps (scripts/ccx_compression_ramp.py), not the frozen setup: the base is held, there
  is no seating/compliance mapping, and the travel is platen travel.
- The panel subtitles inside the images are the renderer's own labels (small at slide size).
