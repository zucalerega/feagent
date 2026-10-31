"""Optional neural-network surrogate for 2D linear elasticity (``backend="nn"``).

Fast but UNVERIFIED: it is never the reference. In the system it is one more "opinion" that the
verification layer judges; mesh refinement / Richardson extrapolation does not apply to it.

Modules (numpy-only unless noted):
  family    training family ``SurrogatePlate`` (plate with a hole / notch; NOT registered) + sampling
  mesh      minimal gmsh .msh reader (tri3 / tri6, named groups)
  graph     case + mesh -> graph and nondimensional features (shared by data, training, backend)
  data      FEM targets, .npz sample round-trip
  interp    point location + P1 / isoparametric P2 interpolation (probe)
  ood       out-of-distribution checks (feature / descriptor ranges, latent distance)
  model     MeshGraphNet-style GNN in plain PyTorch (+ optional PhysicsNeMo wrapper)   [torch]
  train     training loop, checkpoints, latent calibration                            [torch]
  evaluate  errors in / out of distribution, timings, plots                           [torch]

The backend lives in ``feacore.solver.nn_backend``. See docs/surrogate.md.
"""
