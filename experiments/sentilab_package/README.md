# Independent TAP-2.5D continuum FE comparison

For SentiLab ticket `186e391e-d8a3-424a-946e-495b99f7f85f`.

Input attribution: **TAP-2.5D FE reference case**, SentiLab team, 2026,
CC BY 4.0, pinned at
https://github.com/Sentinelayer-AIdenID/sentilab-fea-reference-case/tree/af66bcfc21220ce3d8e92e89273979241e54b04a.
Original data, documentation, source citations and licence are unchanged in `inputs/`.
The reference temperatures are from Ma et al., DATE 2021,
doi:10.23919/DATE51398.2021.9474011. The layout is decoded from that paper's figures.

## Run

Use the FEAgent checkout and its installed dependencies. The retained run uses
the solver inherited from `a5c556f7047ec7b051d30f4e1130743f29fb4fbf` and the
experiment added after the slab handoff `32f3ed1`.
On a fresh Linux machine, `bash scripts/setup_gpu_box.sh` installs system and
Python dependencies, including PARDISO. Then one command rebuilds every number:

```sh
bash experiments/sentilab_package/reproduce.sh
```

Set `PACKAGE_PYTHON` for another prepared Python interpreter, `PACKAGE_SOLVER=scipy`
if PARDISO is unavailable, and `PACKAGE_OUT` to select the output directory.
Default output: `evaluation_runs/sentilab_package_reproduced/`.
`sha256sum` is required for input verification (use Linux or install the equivalent tool).
Runtime and package versions are recorded per run; installation is not a fully
locked transitive dependency environment.

For a single run:

```sh
.venv/bin/python experiments/sentilab_package/run.py --case A --refinement 1 \
  --solver pardiso --out evaluation_runs/package_pilot
```

Use `--variant surface` for power applied to the die-bottom interface;
default `volume` distributes it through the active die thickness.
No parameters are fitted to the paper.

## Numerical method and checks

- Conforming straight tetrahedra, P2 temperature approximation, existing FEAgent
  scikit-fem heat solver and PARDISO/SciPy direct linear solve.
- Tensor axes include every material-block edge, die centre and layer interface.
  Regions follow the supplied block conductivities, including the predeclared
  underfill in the case-A uncovered strip. Copper plates retain their full extents.
- Refinement factors 1, 2 and 3 scale every original interval subdivision;
  successive ratios are 2 and 1.5, in plane and through thickness.
  Elements are highly anisotropic in thin layers. The sink has 3r subdivisions;
  each other layer has r.
- Solve for excess above the fixed 318.15 K ambient; add ambient only in readouts.
  This avoids cancellation in the residual and conservative power balance.
- Check total domain volume, each die's integrated power, total assembled input
  power, cooled area, relative algebraic residual and Robin heat balance.
- Structured element lookup avoids nearest-centroid errors in very thin layers.
  Block-cell averages integrate the actual FE field over tetrahedron/plane
  intersections clipped to the specified rectangle, using degree-2-exact triangle
  quadrature. Independent polynomial checks verify point evaluation and area integration.
- Field extrema are over solution nodes, not stationary points between P2 nodes.
  Observed-order extrapolation and GCI-style uncertainty are estimates rather
  than guaranteed bounds; no asymptotic convergence is assumed automatically.

Per-run JSON contains all die bottom/mid/top temperatures, exact FE plane-cell
averages for supplied L0-L5 blocks, global/layer nodal extrema with coordinates,
power balance, residual, time, peak process RSS, software versions and hashes.
The generated `.msh`, case JSON and nodal field `.npz` reproduce the setup and
retain the solution. `summarize.py` produces the report, combined JSON and checksum
manifest after all eight runs finish.

This is a simulation-to-simulation comparison of a configured model. It is not
experimental, silicon or manufacturer validation. Compact HotSpot and continuum
FE differ in the ways documented by the input authors; a numerical difference
from the paper does not by itself identify a software bug.

Original fixture code and numerical results have a scoped
[MIT licence](../../COLLABORATION-LICENSE.txt). This grant does not cover the
underlying FEAgent solver, other repository components or third-party dependencies.
The input authors' original CC BY 4.0 terms remain in `inputs/LICENSE`.
