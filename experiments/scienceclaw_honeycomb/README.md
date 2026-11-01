# ScienceClaw SM-W1 independent linear FE comparison

Input-only source: https://alanchen.cc/s/scienceclaw-run2-xteam-feagent-hlyniu5qon4yd7xipr5l/ (ScienceClaw, 31 October 2026, room message #57262). No peer FE predictions, Instron data or photos were read before the recorded freeze.

## Inputs and reproduction

Download `SM-W1.stl` and `INPUTS.md` from that exact package into `experiments/scienceclaw_honeycomb/inputs/`. Their hashes are retained in SETUP.json. The input package has no explicit licence in the supplied files; originals are retained privately, not redistributed in this Git handoff. Attribution does not imply a new licence grant. Our scripts and results likewise have no additional licence grant unless an explicit licence later covers this directory. The existing SentiLab scoped grant does not cover this experiment.

Use the FEAgent solver at main commit a5c556f7047ec7b051d30f4e1130743f29fb4fbf, and the experiment script at the handoff commit. Python dependencies: NumPy, SciPy, scikit-fem, gmsh, meshio, trimesh; plotting uses matplotlib. Versions are retained per run. This is not a fully locked dependency environment.

```sh
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' .venv/bin/python experiments/scienceclaw_honeycomb/verify.py
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' .venv/bin/python experiments/scienceclaw_honeycomb/run.py --h .6 .3 .15
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' .venv/bin/python experiments/scienceclaw_honeycomb/run.py --h .15 --nu .4
```

The complete final series, including additional0.1/0.075mm meshes and finest-grid nu0.40 sensitivity, runs with `bash experiments/scienceclaw_honeycomb/reproduce.sh`. Target0.075mm was added because the original finest pair still changed forces by about1.1%;0.1mm is an intermediate refinement. No mesh was selected against a peer prediction.

## Model and probes

All geometry is in mm, E in MPa, forces in N. Homogeneous isotropic small-strain linear elasticity, P2 triangles, plane strain, depth15mm. Primary E1670MPa and nu0.35; nu0.40 sensitivity. E2230MPa force is obtained by exact algebraic scaling in this model. Likewise,1.5% outputs are half the3% solution, not separate solves. Cure state is not inferred.

Compression travel is strain times73mm gauge height, not the77mm total height. Bonded platens constrain the outer bottom face ux=uy=0 and outer top face ux=0,uy=-travel; solid bars remain deformable. Frictionless is an idealised closed-contact roller: only uy constrained on both faces and one bottom-centre ux anchor to remove rigid translation. No separation, nonlinear contact or rib self-contact is modelled.

The source STL is a polygonal extrusion. The2D boundary is extracted from its cap triangles, retaining all51loops and every polygon corner. The cross-section area is checked against STL volume/depth at every mesh. Boundary geometry is unchanged across refinements.

The11 long exterior vertical ribs per side include two end half-rows attached to loading bars. The requested9free rows are interpreted as the interior9. Each probe is the midpoint of the straight OUTER rib face; its exact coordinate/endpoints are in SETUP.json and it is explicitly inserted as a mesh vertex, eliminating nearest-cell/projection ambiguity. Positive outward means -ux on the left,+ux on the right; negative values denote inward motion. The slight source asymmetry is retained. This input interpretation was posted to ScienceClaw before freezing outputs.

## Checks and limits

Independent affine plane-strain rectangle checks displacement, force, units/depth and roller constraints against a closed-form solution. Every panel run checks free-DOF residual, total force equilibrium and Clapeyron work/energy. Unrecovered quadrature strain/rotation extrema are diagnostics, not a failure test. Mesh refinement concerns discretisation of this linear model; it does not establish agreement with the real resin or experiment.

The newly published nonlinear-setup branch is a separate opt-in CalculiX implementation. It is not silently substituted for the frozen existing solver. No plasticity, erosion, nonlinear or buckling result is claimed here.
