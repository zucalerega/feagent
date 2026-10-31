# Physical validation: printed pull tests

Check the fixed solver, and the agents' setups, against Instron measurements on SLA-printed parts
(Formlabs Rigid 10K). Three parts, all loaded by a straight pull.

## Parts (`parts/`, rebuilt by `python scripts/physical_parts.py`)

| Part | Source | Grip ends added | Size (mm) | Resin (cm³, no supports) |
|---|---|---|---|---|
| `abc_00000003_pull` | gold part `abc_00000003`: closed tube (OD 34.5, wall 2.5 mm) with two opposed Ø15.3 stubs on Ø27.6 bosses | both stubs lengthened from 15.4 to 55.4 mm (same axis and diameter) | 149.7 × 34.5 × 61.2 | 44.2 |
| `abc_00000065_pull` | gold part `abc_00000065`: shaft clamp, Ø39.3 bore, Ø6.74 (M6) cross-holes, nut traps | 45 × 6 × 16.2 mm tab under the base, flush with the faces, 3 mm root fillets | 60.7 × 122.8 × 16.2 | 38.5 |
| `mounting_bracket_vertical` | `benchmarks/mounting_bracket_vertical` | none | 100 × 70 × 75 | 81.8 |

- `.stl` is for printing: millimetres (STL has no units, so check PreForm shows the sizes above).
- `.step` (mm) is for FEA.
- Only grip ends were added. Everything the tests measure is the original geometry.
- The repo's gold set has only these two parts (its 10 BC sets are on them). More gold parts are
  in the H100 goldgen archives (`scripts/pull_gold_batches.sh` on `gold-benchmarks`).
- **The gold reference values do not apply here.** They belong to the generated support/load
  sets, not to these pull setups. Each test needs an FEA of the setup as tested.

## Test setups

Breakerspace Instron 68TM-50: 5 kN load cell, wedge grips, video extensometer.

| Part | Lower end | Upper end | Extensometer marks | Fixtures to make |
|---|---|---|---|---|
| `abc_00000003_pull` | one stub in V-face wedge jaws | other stub in V-face wedge jaws | one dot on each boss (about 39 mm apart) | none, if the V faces take Ø15.3 rounds |
| `abc_00000065_pull` | base tab in flat wedge jaws | M6 bolt through the clamp-bolt hole, finger-tight (a pin, not a clamp), through a steel tongue in the slot | arm vs base block; a transverse pair across the bore | tongue: 6 mm steel flat bar, at most 16 mm wide, Ø6.5 hole |
| `mounting_bracket_vertical` | base bolted with 4 × M8 (at x ±38, y ±25 mm) to a steel plate of at least 10 mm with a tongue for the grip | Ø14 pin through the load hole, in a U-clevis over the 8 mm web | web near the load hole vs base near the web root | base plate, U-clevis |

The benchmark models the bracket's bolts as fixed bores. Real bolts clamp the base onto the plate,
so expect a measurable difference there: that is data on how agents should model bolted joints.

## Material: Formlabs Rigid 10K (TDS 2001479, rev 04, 12.12.2022)

| | Green | UV cure (60 min at 70 °C) | UV + thermal (125 min at 90 °C) | UV + media blast |
|---|---|---|---|---|
| Tensile modulus | 7.5 GPa | 10 GPa | 10 GPa | 11 GPa |
| Ultimate tensile strength | 55 MPa | 65 MPa | 53 MPa | 88 MPa |
| Elongation at break | 2 % | 1 % | 1 % | 1.7 % |
| Flexural modulus | 6 GPa | 9 GPa | 10 GPa | 9.9 GPa |

Thermal expansion is about 47 µm/m/°C. The TDS gives no Poisson's ratio and no density.

What this means for the tests:
- **Cure:** post-cure every part and coupon the same way and record it. The modulus spans
  7.5–11 GPa across cure states.
- **Brittle:** keep the peak FEA stress at or below 25 MPa (about 0.25 % strain) for the elastic
  runs. Until the per-part FEA load caps exist, stop the first loading at 100 N.
- **Temperature:** 1 °C over a 40 mm gauge length is about 2 µm, a few % of the expected signal.
  Use LED lighting, keep cycles short, and re-zero before each cycle.

## Procedure

1. **Print.** Print all parts and 3–5 ASTM D638 Type I dogbones in one batch, with the same
   orientation, layer height and post-cure.
   - Put supports on the grip ends only, off the bosses, fillets and bores.
   - Measure the actual dimensions with calipers.
   - SLA holes print slightly undersize: check the pin fits and ream if needed.
2. **Calibrate.** Test the dogbones with the 5 kN cell and the video extensometer, using axial and
   transverse marks, to get E and Poisson's ratio. Optionally, test D790 bars on the flexure
   fixture against the TDS flexural modulus.
3. **Test each part.**
   - Put contrast dots at the marks and preload about 10 N to seat the part.
   - Run 3 load/unload cycles to the load cap at a slow, constant crosshead rate.
   - Log load and mark displacement.
   - Take displacement from the extensometer, not the crosshead: the crosshead also includes grip
     slip and frame compliance.
4. **Optional, last.** Load one of each part to failure, photograph the fracture origin, and
   compare it with the FEA hot spot.

## Comparison

Run FEA of each setup as tested, using the measured E and Poisson's ratio:
- grips are clamped surfaces over the gripped length;
- pins are bearing on the loaded half of the hole.

The quantity compared is the slope of mark displacement against load (mm/N), measured vs predicted
at the same points.

Still to do:
- FEA predictions and load caps for each part;
- problem statements of these setups for the agents.
