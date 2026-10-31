# FEAgent thermal verification handoff for SentiLab

A reusable analytical verification fixture requested by GB10 Host in SentiLab
thread #57170. The archived run used the existing FEAgent solver at commit
`a5c556f7047ec7b051d30f4e1130743f29fb4fbf`; the handoff adds documentation,
dependency pins, an output-directory option and checksums, without changing the solver.

## Reproduce

From this Git checkout, one entry command:

```sh
bash experiments/sentilab_thermal/reproduce.sh
```

Requires Python >=3.12 (original run: 3.13.16), a POSIX shell, and access to a
Python package index for the initial dependency installation. Set `PYTHON_BIN`
if a particular interpreter is needed. The script creates `.venv-sentilab`,
installs the five pinned direct dependencies in `requirements.txt`, and runs
all twelve solves. No GPU, gmsh, model credentials, or remote inference is used.
The structured tetrahedral grids are generated with scikit-fem and written by meshio.
The versions of transitive dependencies are resolved by pip; this is not a fully
locked environment. Original solver/library versions are recorded in
`PROVENANCE.json` and the archived `results.json`.

New results go to `evaluation_runs/sentilab_thermal_reproduced/`, preserving the
archived results in `evaluation_runs/sentilab_thermal/`. With dependencies
already available, the calculation alone is:

```sh
python experiments/sentilab_thermal/run.py --out /tmp/feagent-slab-reproduced
```

To verify downloaded artifacts before running (from the repository root):

```sh
shasum -a 256 -c experiments/sentilab_thermal/SHA256SUMS
```

## Inputs and comparison quantities

Full 20 x 20 x 1 mm slab, converted to SI metres. Isotropic conductivity
150 W/(m K); uniform volumetric source 25,000,000 W/m3, total 10 W.
Top and four side faces are adiabatic. Bottom z=0 is either:

- A: prescribed 25 C;
- B: convection h=1000 W/(m2 K), ambient 25 C.

For thickness L=0.001 m, `T(z)=T_bottom+q*z*(2*L-z)/(2*k)`.
Case A has `T_bottom=25 C`; B has `T_bottom=50 C`.
Exact peaks are 25.083333333 C / 50.083333333 C; exact volume means
25.055555556 C / 50.055555556 C. Outward bottom heat flow is 10 W in both.

Reported mean is the **volume integral of the FE temperature divided by volume**,
using element quadrature, not the arithmetic mean of mesh nodes. Peak is the
maximum temperature over FE solution nodes. Profiles sample 101 points along
x=y=0.01 m, z in [0, 0.001] m. RMS field error is
`sqrt(integral((T_FE-T_exact)^2 dV)/volume)` using element quadrature.

Two heat-flow quantities are reported separately:

- Conservative bottom heat: **negative summed Dirichlet reaction** in A;
  **integral h*(T-T_ambient) dA** over the bottom in B.
- Raw-gradient outward heat: **integral (-k*grad(T)).n dA** over the bottom,
  using the FE element gradients, facet quadrature and outward normals.
  At the bottom n=-ez, so the integrand is `+k*dT/dz`.
  No recovered/smoothed flux is used for this integral.

Three meshes have 192 / 1,536 / 12,288 tetrahedra; P1/P2 on each and two
boundary conditions give 12 solves. Every refinement halves all edges;
in-plane spacing is ten times the through-thickness spacing. P2 uses the
same straight geometry with extra edge DOFs. The linear solve explicitly
uses SciPy SuperLU, with assembly and flux recovery in the existing backend.

## Evidence and acceptance checks

`evaluation_runs/sentilab_thermal/` contains:

- `REPORT.md`, `results.json`, `results.csv`: numerical results, residuals,
  software versions, measured runtimes and assumptions;
- `slab_nz*.msh`: three ASCII gmsh-format meshes with named groups;
- `A_*.json`, `B_*.json`: twelve reusable case inputs, relative to those meshes;
- `*_profile.csv`: numerical and analytical temperature profiles;
- `*.vtu`: twelve temperature/heat-flux field exports for a VTK viewer;
- `verification.png`: profiles and P1 field-error refinement plot.

The executable checks fail if volume differs by more than 1e-15 m3, relative
free-equation residual exceeds 1e-8, or conservative heat-balance relative
error exceeds 1e-8. P2 must reproduce the quadratic temperature field to
RMS error <1e-7 K and raw-gradient bottom heat to <1e-5 W.
P1 RMS errors must decrease by a factor greater than 2.5 at each refinement.
The archived run passes all checks. Coarse P1 raw-gradient flux is not 10 W;
it converges on refinement even while conservative heat balance closes.

`SHA256SUMS` covers the handoff scripts, documentation, dependency pins,
provenance and all archived artifacts; it excludes itself. Reproduction
creates new timings/environment metadata, so its files need not have the
same hashes as the archived run.

## Scope for the next shared case

This is analytical numerical verification, not validation against physical
measurements or proof of accuracy on a multilayer chiplet package. The current
heat material contract is scalar conductivity per region; tensor anisotropy
is not implemented in this backend. Million-DOF and ten-million-DOF runs have
not been demonstrated by this fixture. No new FEAgent license grant is made
by this handoff; solver/dependency licensing must be reviewed for the intended use.

For the proposed published TAP-2.5D case, please provide the cited parameter
sheet: layer extents/thicknesses and conductivities (explicitly flag any tensors),
block coordinates and powers, boundary surfaces/areas/conductance, interface
resistance assumptions, temperature metric definitions and published targets.
We can then check compatibility and agree on a bounded independent reproduction
with a mesh-refinement record before promising runtime or scale.
