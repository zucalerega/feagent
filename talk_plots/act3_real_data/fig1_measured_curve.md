# fig1_measured_curve: "Real test: seating, linear climb, yielding to 4.8 kN, then cracking"

**Caption.** The 3D-printed abc_00000065 C-bracket (Formlabs Alpha Fast) crushed between Instron platens: a soft
seating toe, a straight climb, softening as the resin yields, a peak (the first crack), then saw-tooth load drops
to the break. The shaded loading branch, up to the peak, is what every simulation is compared with.

**Key numbers** (computed in make_figures.py from the test CSV):
- peak 4755 N at 2.513 mm crosshead travel in this digitization (the machine report says
  4.80 kN at 2.56 mm)
- toe slope 1.66 kN/mm (fit over 0.10-0.45 mm); linear slope
  2.70 kN/mm (fit over 1050-1800 N)
- force at 2.5 mm: 4743 N; last point (break) at 8.68 mm; 2581 digitized points

**Data.** physical_tests/abc_00000065_compression/cload_compression.csv (the frozen spec's test file); test notes cload_compression.json next to it.

**Caveats.**
- The curve is digitized from the plot in the machine's PDF report (about +-0.01 mm, +-15 N), not raw data. A
  second digitization put the peak about 1 % higher (4798 N at 2.556 mm); the error uses this file.
- Displacement is crosshead travel (no extensometer): it includes seating and the load train's deflection.
- "Seating, one prong first?" for the toe is an interpretation from the test notes, not a measurement.
