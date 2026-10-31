# Benchmarks

Four suites of decreasing reference trust and increasing geometric complexity. Each serves a
different purpose for the swarm.

| Suite | Purpose | References | In repo | Status |
|---|---|---|---|---|
| [`verification/`](verification/) | **Agent tasks** (CAD in, deck out) on canonical problems | analytical, NAFEMS-published, or converged-numerical (Richardson + GCI); every case demonstrated by a convergence study | 10 cases, complete | ready |
| [`simjeb/`](simjeb/) | **Agent tasks** on 381 real crowd-sourced brackets (complex geometry) | OptiStruct fields from the SimJEB dataset (code-to-code) | code + docs; data fetched on demand | blocked here by the network policy (dataverse.harvard.edu) |
| [`code_aster/`](code_aster/) | **Reference library**: 3,589 independent reference values (analytical / literature) from 253 code_aster validation tests, with locations | code_aster `TEST_RESU` (VALE_REFE, PRECISION) | manifest of all 253 + 80 core tests vendored | ready (source of future tasks) |
| [`calculix/`](calculix/) | **Solver acceptance** for the fixed solver, plus a **corpus of known-good decks** per keyword | ccx's own regression outputs | 214 verified decks + report on all 603 | ready |

## How a benchmark run works

```
task/ (STEP + task.json + task.md)  ->  agent  ->  submission/model.inp
                                                          |
         reference/reference.json  ->  feabench evaluate  ->  evaluation.json (gates, QoIs, score)
```

* Agents only ever get `task/`. Use `feabench export-task <case> <dir>` and never mount
  `benchmarks/` into an agent sandbox.
* The evaluator runs the fixed solver (ccx 2.21) on the submitted deck, after appending output
  requests.
* It reads the results and computes every quantity of interest on the **agent's own mesh**:
  shape-function interpolation, area-weighted surface means, reaction sums, eigenvalues.
* Scoring is described in [`docs/eval-design.md`](../docs/eval-design.md).

```bash
pip install -e .            # needs ccx (apt install calculix-ccx), gmsh, meshio, numpy, scipy
feabench list
feabench export-task verification/nafems_le10 /tmp/agent_ws
feabench evaluate verification/nafems_le10 /path/to/submission --out eval.json \
    --agent-cost '{"input_tokens": 120000, "output_tokens": 15000, "wall_s": 240}'
feabench reference verification/nafems_le10 --level 3     # scripted baseline submission + score
feabench convergence verification/nafems_le10             # re-run the convergence study
```

## Reference trust tiers

| Tier | Meaning | Where |
|---|---|---|
| analytical | exact solution of exactly the posed problem | verification (Lamé, Saint-Venant, self-weight), code_aster `ANALYTIQUE` |
| published | community benchmark target | verification (NAFEMS LE1, LE10), code_aster `SOURCE_EXTERNE` |
| numerical | converged FE with an estimated uncertainty (GCI) | verification (cantilever deflection, frequencies, buckling, bracket) |
| code-to-code | another solver's single solution, with its own idealisation | SimJEB (OptiStruct) |
| regression | the solver agrees with an earlier version of itself | calculix `.dat.ref`, code_aster `VALE_CALC` |

Only the first four are used to score agents. Regression references accept the solver build.

## Licences

Vendored third-party files keep their licences. CalculiX tests are GPL-2.0
(`calculix/LICENSE`); code_aster tests are GPL-3.0 (`code_aster/LICENSE`). SimJEB data is ODC-By
with GrabCAD-sourced CAD for non-commercial use, and is never committed. The verification suite
and `feabench` are original to this repo.
