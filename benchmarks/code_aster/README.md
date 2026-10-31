# code_aster validation tests

* **Source:** <https://gitlab.com/codeaster/src> (EDF), directory `astest/`, commit
  `7cb48262af47548277f4824ea47f1cc90848f057` (2026-10-21).
* **Licence:** GPL-3.0-or-later (`LICENSE`).
* **Families:** SSLV (linear statics of 3D solids) and SDLV (linear dynamics of 3D solids).

## Why this is valuable

code_aster's validation tests are carefully documented problems with **independent** reference
values: closed-form solutions (`REFERENCE="ANALYTIQUE"`) or literature and other codes
(`SOURCE_EXTERNE`). Each value comes with the accuracy code_aster itself achieves on the test's
mesh (`VALE_CALC`, within `PRECISION`). The code_aster CI re-runs all of them on every release,
so the agreement is continuously demonstrated upstream.

## `manifest.json`

There is one record per test variant (253 SSLV + SDLV variants), with:

* `title` / `header`: the problem statement from the command file header (mostly French).
* `modelisations` (`3D`, `D_PLAN`, `AXIS`, ...), `operators` (`MECA_STATIQUE`, `CALC_MODES`,
  ...), and `materials` (`ELAS` E, NU, RHO).
* `entries`: every `TEST_RESU` / `TEST_TABLE` / `TEST_FONCTION` value. Each entry carries:
  * the field, component and node/cell group;
  * `VALE_REFE`, `VALE_CALC`, `PRECISION`, `CRITERE`;
  * `ref_class` (`independent` / `other_code_aster_model` / `undefined` / `regression_only`);
  * `calc_vs_refe_rel`: how close code_aster gets on that mesh;
  * `location`: group centroid, bounding box and node count, read from the mesh.
* `vendored`: whether the test's files are in `astest/`.

| | Count |
|---|---|
| reference entries | 4,736 |
| independent (`ANALYTIQUE` 3,424 + `SOURCE_EXTERNE` 165) | 3,589 |
| independent with a literal value | 3,313 |
| independent placed in space | 2,536 |

### How the extraction works

Command files are Python, but they are **parsed with `ast`, never executed**. Constants and
arithmetic are resolved by a whitelist evaluator (numbers, `math`, module-level constants).
Anything else is kept verbatim as `{"expr": ...}`, and values generated in loops are flagged
`_in_loop`. Mesh groups are read from MED files with a small direct HDF5 reader that handles
MED 2.x-4.x, wedge15 and older layouts that meshio rejects, and from native `.mail` files.

## Vendored subset (`astest/`)

The vendored tests:
* have at least one independent reference;
* use 3D solid modelling;
* involve no fracture mechanics (`CALC_G`, X-FEM, `POST_K1_K2_K3`) and no nonlinear dynamics;
* total ≤ 1.5 MB of files.

That is 80 tests and 150 files (11 MB): `.export`, `.comm`, meshes (`.mmed`, `.mail`, `.msh`)
and Gibi geometry scripts (`.datg`).

## Using it

* **Mine new agent tasks.** Filter `ref_class == "independent"` and `modelisations` containing
  `3D`. Each entry gives the quantity, its reference and its location. Re-create the geometry as
  CAD (dimensions are in the mesh bounding boxes and headers), as was done for the Lamé cylinder.
* **Cross-check our fixed solver.** Rebuild the same problem in ccx from the mesh and compare.
* **Run the originals** with code_aster: `run_aster sslv04a.export`, using the official
  container or conda-forge `code-aster`. Not done here.

```bash
feabench aster-fetch /tmp/aster --families sslv sdlv ssls      # pinned sparse checkout
feabench aster-manifest /tmp/aster/astest --families sslv sdlv ssls --out manifest.json
```
