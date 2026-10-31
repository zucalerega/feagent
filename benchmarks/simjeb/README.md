# SimJEB: Simulated Jet Engine Bracket Dataset

> Whalen, Beyene & Mueller, *SimJEB: Simulated Jet Engine Bracket Dataset*, Computer Graphics
> Forum 40(5), 2021. Data: Harvard Dataverse, doi:10.7910/DVN/XFUWJG.

381 cleaned brackets from the GE / GrabCAD Jet Engine Bracket Challenge. Each was meshed and
simulated in Altair OptiStruct under the challenge's four separate load cases, all applied
through the 3/4" clevis pin at interface 1, with bolt interfaces 2-5 fixed, in Ti-6Al-4V:

| Load case | Load |
|---|---|
| vertical | 8,000 lbf, up |
| horizontal | 8,500 lbf, out |
| diagonal | 9,500 lbf at 42° from vertical |
| torsion | 5,000 lbf·in |

Per bracket the dataset has clean CAD (`.stp`), the FE model (`.fem`), a tet mesh (`.vtk`), a
surface mesh (`.obj`) and results (`.csv`). Each file type is a separate zip, plus a metadata
file with summary statistics and train/test splits.

## Status: integrated, data not yet downloaded

`dataverse.harvard.edu` is blocked by this environment's network policy, so **no SimJEB files
are in the repo and none of the code has seen real SimJEB files.**

**Verified** by unit tests on synthetic decks (`tests/test_simjeb_fem.py`):
* the Nastran / OptiStruct bulk-data parser: small, large and free field, continuations,
  `1.5-3` exponents, `THRU`;
* case control (`SUBCASE`, `LOAD`, `SPC`, `LABEL`);
* RBE2 and RBE3 handling. RBE3 loads become explicit nodal forces that reproduce the resultant
  force and moment exactly (weighted least-squares distribution);
* `.fem` to CalculiX conversion: a synthetic shaft with an RBE3 moment reproduces the exact
  Saint-Venant twist within 1 %;
* cylinder fitting of interface node sets.

**Not verified:**
* zip member naming;
* the results-CSV column layout. `pick_columns` recognises columns from `CSV_HINTS` and raises
  an error listing the columns it found, rather than guessing.

## First run (with network access to dataverse.harvard.edu)

```bash
feabench simjeb fetch --dest data/simjeb --dry-run                 # list files, sizes, MD5s
feabench simjeb fetch --dest data/simjeb --include "*metadata*" "*sample*"
python -m feabench.suites.simjeb inspect data/simjeb               # zip contents; check CSV headers
# adjust feabench/suites/simjeb.py CSV_HINTS if needed, then:
python -c "from feabench.suites import simjeb; simjeb.extract('data/simjeb')"
feabench simjeb build-tasks --dest data/simjeb --out data/simjeb/cases --ids 0 1 2
```

`data/` is git-ignored. The data is licensed ODC-By, and the CAD comes from the GrabCAD
challenge (non-commercial use), so do not commit raw files.

## How SimJEB cases are scored

The geometry is the bracket's STEP. The task gives the load case, and the interfaces both as
fitted cylinders and as the interface node coordinates from the OptiStruct model
(`interfaces.npz`). The reference is OptiStruct's solution:

| QoI | Tolerance |
|---|---|
| relative-L2 error of the displacement field, interpolated at SimJEB nodes | 5 % |
| max \|U\| | 3 % |
| 99.9th-percentile von Mises (the maximum sits on singular interface edges) | 10 % |

These tolerances are deliberately wide. A single OptiStruct solution with RBE2 bolts and an RBE3
pin load is a code-to-code reference, and an agent that idealises the interfaces differently
(contact, distributing coupling with area weights) can be right and still differ by a few
percent.

**Recommended next step once the data is reachable.** For a handful of brackets:
1. convert the `.fem` with `fem_to_ccx`;
2. solve in ccx;
3. compare with the SimJEB CSV.

That measures how reproducible the references are on our fixed solver before agents are scored
on them.
