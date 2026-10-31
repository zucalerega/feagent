# CalculiX upstream regression suite

* **Source:** <https://github.com/Dhondtguido/CalculiX> (Guido Dhondt, the CalculiX author),
  directory `test/`, commit `3e742abb894ce071206c2a3836dac80dfbaaf9d0` (2023-12-02).
* **Licence:** GPL-2.0-or-later (`LICENSE`).

## What these references are

Each `<deck>.dat.ref` is CalculiX's **own output** from the version that produced the suite. A
pass means the installed ccx reproduces upstream ccx to within 0.1 % of each output block's
largest value. That is solver/build verification, **not** an independent check of the physics.
We use it for two things:

1. **Accepting the fixed solver.** Every machine that runs the swarm's evaluator should reproduce
   this report before its scores are trusted.
2. **A corpus of known-good decks.** There are 214 decks, each a working example of keywords
   agents need: `*BOUNDARY`, `*CLOAD`/`*DLOAD`, `*COUPLING` with `*DISTRIBUTING`/`*KINEMATIC`,
   `*RIGID BODY`, `*EQUATION`, `*FREQUENCY`, `*BUCKLE`, C3D4/10/8/20(R), shells, beams, contact,
   `NLGEOM`, plasticity.

## Snapshot choice

No tagged 2.21 test suite is reachable from this environment. The `test/` directory first
appears in 3e742ab, four months after the 2.21 release (99ef753) and before 2.22. Differences
between that snapshot and the 2.21 binary show up below as failures (new keywords, changed
output).

## Verification with ccx 2.21 (Ubuntu 24.04 `calculix-ccx 2.21-1`, SPOOLES)

The full report is in `verification_ccx-2.21.json`. It holds, per deck: status, worst relative
error, CPU time, number of equations, tags and element types.

| Status | Decks | Meaning |
|---|---|---|
| pass | 474 | clean solver exit and `.dat` within 0.1 % of block maximum (upstream `datcheck.pl` rule) |
| pass_vacuous | 94 | ran cleanly but `.dat.ref` is empty, so nothing could be compared |
| solver_error | 20 | ccx `*ERROR` or non-zero exit: auxiliary files not in the snapshot (`.stiff`, `.vwf`, `twobeam.frd`), or keywords newer than 2.21 |
| fail | 9 | numbers differ; version drift between 2.21 and the snapshot |
| line_count | 6 | output layout differs; version drift |

Linear structural decks (static, frequency or buckle; no contact, NLGEOM or nonlinear material)
split into 113 pass, 29 vacuous, 1 fail and 5 solver errors.

Our rule is stricter than upstream's `compare` script, which reports a pass when an empty `.dat`
matches an empty `.dat.ref`, even if ccx crashed.

## Vendored subset (`decks/`)

Decks with status `pass`, only static / frequency / buckle procedures, and `.inp` < 0.5 MB:
214 decks (`.inp` + `.dat.ref`), 10 MB. Of these:

| Category | Decks |
|---|---|
| linear structural | 113 |
| `NLGEOM` | 87 |
| contact | 31 |
| beams / trusses | 23 |
| frequency | 21 |
| nonlinear material | 17 |
| shells | 14 |
| buckling | 3 |

Each deck's record in the report has `"vendored": true`.

```bash
feabench calculix-fetch /tmp/ccx-src     # full suite, pinned commit
feabench calculix-verify /tmp/ccx-src/test --out report.json --workers 4
```
