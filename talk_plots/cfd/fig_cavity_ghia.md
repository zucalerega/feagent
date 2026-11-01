# fig_cavity_ghia: "Cavity: the classic check"

**Caption.** Velocity along the two centrelines of the Re = 100 lid-driven cavity: u on the vertical line x = 0.5
(left) and v on the horizontal line y = 0.5 (right). Lines: the OpenFOAM solve of `fig_cavity`
(16,384 cells), read with the solver's own probe. Circles: the Re = 100 columns of Ghia, Ghia & Shin
(1982), Tables I and II (17 points each, at their grid points j / 128).

**Key numbers** (`cfd_numbers.json`, cavity)
- Largest gap to the 15 interior table points: 0.0049 m/s in u, 0.0091 m/s in v
  (RMS 0.0024 and 0.0050), with the lid at 1 m/s.
- The four checked points of `feacore/problems/cfd/cavity.py`:

| QoI | this solve | Ghia | mesh-converged solver value |
|---|---|---|---|
| `u_vcl_min` | -0.21374 | -0.21090 (+1.3%) | -0.21392 (-0.08%) |
| `v_hcl_max` | 0.17935 | 0.17527 (+2.3%) | 0.17951 (-0.09%) |
| `v_hcl_min` | -0.25329 | -0.24533 (+3.2%) | -0.25379 (-0.20%) |
| `u_centre` | -0.20890 | -0.20581 (+1.5%) | -0.20914 (-0.11%) |

**Caveats**
- Ghia's Re = 100 values are 1.4-3.5 % smaller in magnitude than mesh-converged solutions (Richardson
  extrapolation of the solver's 32 / 64 / 128 study, GCI <= 0.25 %, in line with Sahin & Owens 2003;
  `feacore/problems/cfd/cavity.py`). Most of the gap to the table is the table's own error.
