# fig1_measured_curve: "Real test: seating, linear climb, yielding to 4.8 kN, then cracking"

**Caption.** The 3D-printed abc_00000065 C-bracket (Formlabs Alpha Fast) crushed between Instron platens: a soft
seating toe, a straight climb, softening as the resin yields, a peak, then saw-tooth load drops (cracking) to the
break. The shaded loading branch is what the simulation is scored against.

**Key numbers** (all computed in make_figures.py from the test CSV):
- peak 4755 N at 2.513 mm crosshead travel in this digitization (the test notes say 4.80 kN at 2.56 mm; see caveats)
- toe slope 1.66 kN/mm (fit over 0.10-0.45 mm); linear slope 2.70 kN/mm (fit over 1050-1800 N)
- force at the 5 scoring checkpoints 0.5 / 1.0 / 1.5 / 2.0 / 2.5 mm: 860 / 2164 / 3308 / 4171 / 4743 N
- last point (break) at 8.68 mm; 2581 digitized points

**Data.** physical_tests/abc_00000065_compression/cload_compression.csv (the frozen scoring's test file);
test notes physical_tests/abc_00000065_compression/cload_compression.json.

**Caveats.**
- The curve is digitized from the plot in the machine's PDF report (about +-0.01 mm, +-15 N), not raw data.
- A second digitization (/private/tmp/claude-501/-Users-aryan-Documents-Aryan-CS-scienceclawhack/29d391ed-69f3-4559-b365-a260e486ee85/scratchpad/wf/cload_compression_curve.csv) puts the peak at 4798 N at 2.556 mm, about 1 % higher
  than this file near the peak. The scoring uses this file, so the figure does too.
- Displacement is crosshead travel (no extensometer): it includes seating and the load train's deflection.
- "Seating, one prong first?" for the toe is an interpretation from the test notes, not a measurement.
