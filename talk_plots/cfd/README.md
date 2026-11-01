# Flow (CFD): the fixed flow solver on two classic benchmarks

`feacore` solves steady laminar flow with an OpenFOAM backend (`backend="openfoam"`, physics "flow"; see
`docs/solver_openfoam.md` and `docs/benchmarks_cfd.md`). These figures show two of its benchmarks, each solved with
the problem's own reference setup on its block-structured quad mesh. Every number is in `cfd_numbers.json`, and
each figure's `.md` has its caption, numbers and caveats.

| figure | what it shows |
|---|---|
| [fig_cylinder_wake](fig_cylinder_wake.md) ([png](fig_cylinder_wake.png)) | DFG 2D-1 cylinder, Re = 20: speed and streamlines. Pressure drop across the cylinder 0.11737 Pa, 0.12 % below the reference (47,264 cells). |
| [fig_cylinder_vorticity](fig_cylinder_vorticity.md) ([png](fig_cylinder_vorticity.png)) | The same flow's vorticity: shear layers from the cylinder and the walls, no shedding at Re = 20. |
| [fig_cylinder_mesh](fig_cylinder_mesh.md) ([png](fig_cylinder_mesh.png)) | Close up on the 11,816-cell mesh: the O-grid around the cylinder and the 82.0 mm recirculation bubble. |
| [fig_cavity](fig_cavity.md) ([png](fig_cavity.png)) | Lid-driven cavity, Re = 100: the main vortex and the two corner eddies (16,384 cells). |
| [fig_cavity_ghia](fig_cavity_ghia.md) ([png](fig_cavity_ghia.png)) | The cavity's centreline velocities on Ghia et al.'s table: largest gap 0.0091 m/s at a lid speed of 1 m/s. |

Rebuild (needs OpenFOAM and foamlib; a few minutes for the three solves, cached in `_cache/`):

```bash
python talk_plots/cfd/make_figures.py            # --resolve to solve again
```

The ACE learning loop runs on these problems too, as the `cfd` and `cfd3d` suites (`python -m ace.run_loop
--suite cfd`, PR #21).

## Supersonic flow: the swarm's first generation

[supersonic_g0.md](supersonic_g0.md): generation 0 of a 6-agent swarm run on a supersonic suite, with compressible
flow from PR #24 (the run is in progress). The target flow around a Mach 3 blunted wedge
([png](fig_supersonic_reference.png)), agent a0's first mesh with the reference bow shock over it
([png](fig_supersonic_mesh_g0.png)), and a0's run report as the teacher saw it ([`supersonic_g0/`](supersonic_g0/)).
`make_supersonic_figures.py` rebuilds them from the run folder and its field references, which are not in git.
