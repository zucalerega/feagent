# Decision log

Decisions taken during the overnight benchmark build (2026-10-31), when no teammate was available
to ask. Each has a short rationale and what would make us revisit it. Newest last.

## D1. Work on a `benchmarks` branch with a draft PR, not on `main`
The session was configured to push to `main`, but CLAUDE.md says "branch off `main` per
workstream". The PR is reversible and reviewable. Merging is the team's call.

## D2. CalculiX 2.21 is the fixed solver (provisional)
* Free, a single binary, and `apt install calculix-ccx` on Ubuntu 24.04 gives exactly 2.21.
* Its Abaqus-style decks are well known to LLMs.
* It is fast enough (a sub-second solve for every verification case at the accuracy needed) and
  supports C3D10/C3D20, couplings, modal and buckling analyses.

code_aster stays a *reference source*, not the swarm's solver: heavier to install, and its Python
command files are a much larger surface for agents. The evaluation layer only needs results with
U/S/RF fields, so changing solver later means a new runner and reader, not new benchmarks.
**Revisit if** the team wants nonlinear contact/plasticity at scale or a different MCP backend.

## D3. Submission contract: a directory with `model.inp`
The agent MCP is undefined (open question in CLAUDE.md). The most solver-faithful artefact an
agent can hand over is the deck itself. The harness:
* appends output requests;
* runs ccx with resource accounting;
* reads `.frd`/`.dat` plus full-precision node coordinates from the deck;
* measures everything on the agent's mesh.

**Revisit when** the MCP exists. The MCP's "solve" tool should produce exactly this directory.

## D4. Units: mm, N, MPa, tonne, s for all verification tasks
This is the most common consistent set for mechanical CAD (STEP files are in mm), and it makes
density units (tonne/mm³) a deliberate test of unit discipline.

## D5. Regions are geometric predicates, not CAD face ids or node numbers
Tasks say *where* BCs go with planes, cylinders, elliptic cylinders, spheres and boxes, plus a
human label. The evaluator uses the same predicates on any agent's mesh. CAD face ids are not
stable across kernels and mesh node ids are agent-specific. Default tolerance is 1e-5 × the model
diagonal; that absorbs the 6-significant-digit coordinates in `.frd` files.

## D6. Reference trust tiers, and agents never see references
Every QoI declares `reference_type` (analytical > published > numerical > code-to-code >
regression). Files are split into `task/` (agent-visible) and `reference/` (harness-only).
`feabench export-task` copies only `task/`, and a test asserts no reference value appears in
agent-visible files.

## D7. Numerical references by convergence study with Roache GCI
Where no closed form exists, the reference is the Richardson extrapolation of the reference
pipeline's three finest meshes when the observed order is plausible (0.5-4). Otherwise it is the
finest value, with GCI safety factor 3 at the formal order. Each tolerance should be at least 3×
the reference uncertainty, and the build warns otherwise. Closed-form beam theory values sit
alongside as sanity checks:
* cantilever: 3D -1.911 mm vs EB 1.905 / Timoshenko 1.919;
* modal: f1 209.4 Hz vs EB 208.9;
* buckling: 2.7006 vs Euler 2.6987.

## D8. Every verification case must be *demonstrated*, not just specified
The scripted pipeline (gmsh quadratic tets fed to ccx) runs each case at five mesh sizes on the
task's own STEP and is scored by the agents' evaluator. This caught three problems before any
agent saw them:
* ccx RF is the nodal internal force, not just the reaction, so gates now use load-free support
  regions;
* corner-node stress at NAFEMS LE1 point D depends on how many tets share the node, so the QoI is
  now the thickness average (the plane-stress definition);
* the Lamé-sphere bore hoop stress scatters about 1.7 %, so its tolerance is 2.5 %;
* the bracket's fillet peak converges erratically (conservative GCI 1.1 %), so its tolerance is
  3.5 %, keeping it at least 3x the reference uncertainty;
* one LE1 pipeline mesh has sliver tets at the notch and is 14 % low. That is kept as evidence;
  QoIs now report element quality at the evaluation point.

## D9. Region means are area-weighted, field statistics are nodal
A plain average over nodes depends on mesh grading. It moved the bracket's pin-hole displacement
by 0.3 % between meshes. Means over surfaces are now integrated over the agent's boundary faces
(tri6 for C3D10). Maxima and percentiles stay nodal.

## D10. Scoring: S = G × A × (1 − 0.2·(1 − E)); ranking by within-case Bradley-Terry
Full argument and alternatives are in `docs/eval-design.md`. Short version:
* correctness dominates;
* errors are scored on a log scale with partial credit up to 10× the tolerance;
* efficiency (tokens in $, wall time) costs at most 20 %;
* agents are ranked only against agents that saw the same case.

**Revisit** w_eff after the first population run (if every agent passes, raise it).

## D11. Default prices: Claude Opus 5.5 list prices
$4 / $20 per MTok input / output, $0.20 cache reads, $5 cache writes. `ScoreConfig.for_model()`
switches models. Solver CPU is priced at $0.05 per CPU-hour; it is negligible in every case
measured.

## D12. What is vendored and what is fetched
| Suite | In the repo | Fetched on demand (pinned) |
|---|---|---|
| CalculiX tests | 214 decks that pass the strict check with ccx 2.21, structural procedures, < 0.5 MB each (10 MB); verification report for all 603 | `feabench calculix-fetch` (GitHub Dhondtguido/CalculiX @ 3e742ab) |
| code_aster | 80 SSLV/SDLV tests with independent references, 3D modelling, no fracture, ≤ 1.5 MB each (11 MB); manifest of all 253 SSLV+SDLV tests with every extracted TEST_RESU | `feabench aster-fetch` (gitlab codeaster/src @ 7cb4826) |
| SimJEB | code + docs only: the network policy blocked dataverse.harvard.edu | `feabench simjeb fetch` (Harvard Dataverse doi:10.7910/DVN/XFUWJG, MD5-verified) |
| verification | everything (STEP files are small) | regenerate with `feabench build-verification` |

Total added: about 26 MB. Large binary datasets (SimJEB) never go in git (`data/` is ignored).

## D13. CalculiX test-suite snapshot and the stricter pass rule
Upstream has no tagged 2.21 test suite on an allowed host. The `test/` directory first appears in
commit 3e742ab (2023-12-02), between 2.21 and 2.22, so that is the snapshot. Upstream's `compare`
counts a run as a pass when an empty `.dat` matches an empty `.dat.ref`, even if ccx errored. Ours
requires a clean solver exit and reports `pass_vacuous` when the reference is empty.

Result with ccx 2.21:
* 474 pass;
* 94 vacuous;
* 20 solver errors: missing auxiliary files, and newer keywords than 2.21;
* 9 fail and 6 line-count mismatches: version drift.

Among linear structural decks, 113 of 148 pass.

## D14. code_aster command files are parsed, never executed
`.comm` files are Python. References are extracted with `ast` plus a whitelist evaluator
(numbers, arithmetic, `math`, module-level constants). Anything else is kept as an unresolved
expression. Node groups are located from the MED meshes with a small direct HDF5 reader; meshio
fails on wedge15 and old MED files.

Result: 3,589 independent reference values (ANALYTIQUE / SOURCE_EXTERNE) across 253 tests,
2,536 of them with coordinates.

## D15. SimJEB is integrated but unverified against real files
The paper and challenge spec give the file types and load cases; the network policy blocked the
data itself. What is tested on synthetic decks:
* the Nastran parser (small, large and free field, continuations, shorthand exponents);
* the RBE3 load distribution, which reproduces resultant force and moment exactly;
* `.fem` to ccx conversion: a synthetic shaft with an RBE3 moment matches Saint-Venant twist
  within 1 %;
* cylinder fitting for interfaces.

What is not tested: zip layout and results-CSV columns. `pick_columns` fails loudly with the
columns it saw. SimJEB tolerances are wide (5 % field L2, 3 % max |U|, 10 % p99.9 von Mises)
because the reference is a single OptiStruct solution with its own RBE idealisation.

## D16. LE10 and LE1 targets taken from the published NAFEMS values
Targets: LE10 σyy(D) = -5.38 MPa, LE1 σyy(D) = 92.7 MPa.
* Sources: the NAFEMS benchmark definitions as reproduced by vendor verification manuals. Web
  search confirmed values, boundary conditions and geometry; the original document is not
  available here.
* Our finest meshes give LE10 -5.393 MPa (0.24 % off) and LE1 92.57 MPa (0.14 % off, thickness
  average).

## D17. Meshing runs in a subprocess
gmsh's high-order optimiser twice threw an uncaught C++ exception that aborted the Python
process. Hits were not deterministic: one mesh crashed on a re-run but not the first time. The
pipeline now meshes through `python -m feabench.fem.meshworker` and retries without the
optimiser on a crash. It uses a plain subprocess rather than multiprocessing, so callers' scripts
need no `__main__` guard. Agents' MCP meshing tools should do the same.

## D18. Things deliberately not done
* **Running code_aster.** Not needed to use its references, which upstream CI validates.
  Possible later via the official container or conda-forge.
* **Hunting for unofficial SimJEB mirrors** on allowed hosts. The right fix is to allow
  dataverse.harvard.edu in the environment's network settings.
* **Committing generated SimJEB tasks.** The CAD is GrabCAD-licensed for non-commercial use;
  generated tasks live in git-ignored `data/`.
