# 07 - OpenUSD / GLB / GPU-render visualization spike

Worktree on `mesh-harness` (dd7292c). `uv sync --extra ui`, then `uv pip install usd-core trimesh` (usd-core 26.11, trimesh 5.2). Nothing committed. Code and outputs are in `usd_spike/`.

## Pipeline (works end to end)
1. Parts: `feagent parts`, `feagent automesh` for plate_hole and thin_cup. cantilever and bracket use the passing `meshing-harness/agent_context/examples/*/mesh.yaml` (they have `bcs:`), run through `feagent check`.
2. Solve: `python -m feacore.adapters.mesh_harness <out> --out case.json`, then `python -m feacore.cli case.json --out result.json --vtu x.vtu`.
   - The result JSON holds only `info` and `qois`. The nodal fields come from `--vtu`: ux/uy/uz, u_mag, sxx.., von_mises, tresca, s1-s3, displacement. Solving cantilever took 2.4 s (23k DOFs).
3. `usd_export.py cfg.json outdir`. For each part it:
   - extracts the outward-oriented boundary triangles from the linear corners of the tet or tet10 mesh,
   - writes `/World/Assembly/<part>` (Xform, kind=component, translate op, customData: part_id, gate_status, n_dofs, QoIs, tri/tet counts, scalar_is_mock),
   - writes `<part>/Surface` (UsdGeomMesh: points, counts, indices, vertex normals),
   - writes vertex-interpolated float primvars (`u_mag`, `von_mises`) and a `displayColor` primvar from the turbo colormap.
   - Unsolved parts get a `mock_height` scalar (flagged `scalar_is_mock`). plate_hole shows all red because the mock scalar is constant on a flat plate.
4. GLB: `stage_to_glb` traverses the USD stage (not the source meshes) with pxr and writes it with trimesh. It carries per-part nodes with world transforms and vertex colors. The `viewer.html` (three.js 0.160 from unpkg, GLTFLoader, OrbitControls, vertexColors) embeds the GLB as base64, so it works from file://. It was written but not run in a browser (the CDN needs internet).
5. Render: `render_usd.py` reads the .usdc via pxr, builds PyVista meshes with RGB scalars, and renders offscreen. `shot_nvidia.png` shows 4 parts side by side, with von Mises on the cantilever/bracket (hot spots at the bracket holes and the cantilever tip).

Assembly: 4 parts, 152k surface tris, from 4 volume meshes. Timings:

| Output | Size | Time |
|---|---|---|
| usda | 9.3 MB | 4.6 s |
| usdc | 4.0 MB | 4.1 s |
| glb | 3.0 MB | 1.1 s |

Most of the export time is the numpy boundary extraction (`np.unique` on faces) in Python, not USD itself.

How to run (uses the repo venv; with the spike dir as cwd):
`python usd_export.py cfg.json out/`, then `./run_render.sh nvidia out/assembly.usdc out/shot.png`. The paths in `cfg.json` point to a scratch dir (e.g. ./w), so regenerate or edit them.

## Scale (synthetic surface, 1 scalar + color, `scale.py`)
| Case | Author+save | usdc | GLB write | GLB |
|---|---|---|---|---|
| 1.0M tris | 0.7 s | 20 MB | 0.74 s | 20 MB |
| 5.0M tris | 5.5 s | 100 MB | 2.1 s | 100 MB |

That is about 20 MB per Mtri in both formats (about 1.1 s/Mtri for USD, 0.4 s/Mtri for GLB). The 20 MB/Mtri breaks down as follows:
- points: 12 B/vertex
- normals: 12 B/vertex
- color: 12 B/vertex
- scalar: 4 B/vertex
- indices: 12 B/tri
- crate compression gained almost nothing on this float-heavy synthetic data.

Cost with real boundary extraction is higher, since tet mesh counts are ~10-20x the surface tri count. At 10^7 volume tets the surface is only a few percent to ~10% of that. 10^7 surface tris would be about 200 MB per scalar set. To get there:
- drop normals (the viewer computes them) and store color as uint8 or per-scalar only (about 40% saving);
- quantize points to uint16 (Draco/meshopt in GLB);
- store scalars as float16 or normalized uint8 with a range in metadata;
- decimate (surface decimation with scalar error) and build LOD variants as USD payloads/variants, loaded on demand;
- make per-part instancing of repeated parts (USD native instancing, `instanceable=True`);
- chunk large parts into tiles for frustum culling;
- run the boundary extraction in C++/numba/vtk, not `np.unique`.

## GPU rendering check (RTX 4060 Laptop, VTK 9.7.1 default wheel)
- EGL is available out of the box. The stock `vtk` wheel already contains `vtkEGLRenderWindow`, so no `vtk-egl` install was needed.
- GPU path: unset DISPLAY, `__EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json`, `VTK_DEFAULT_OPENGL_WINDOW=vtkEGLRenderWindow`. This gives renderer "NVIDIA GeForce RTX 4060 Laptop GPU".
- X11 path: the default with DISPLAY set picks the Intel Arc iGPU (Mesa). CPU path: `LIBGL_ALWAYS_SOFTWARE=1` under X11 gives llvmpipe. A pure-EGL llvmpipe route failed because VTK picks NVIDIA device 0. OSMesa is not present.
- Frame render time, 1600x900, smooth shading, per-vertex RGB (pl.render loop + final screenshot):

| Tris | NVIDIA EGL | Intel Arc iGPU (X11) | llvmpipe CPU |
|---|---|---|---|
| 152k (assembly) | 5.6 ms (178 fps) | 19.4 ms (51 fps) | 157 ms (6.4 fps) |
| 1.0M | 10.0 ms (100 fps) | 25 ms (40 fps) | 200 ms (5 fps) |
| 5.0M | 11.9 ms (84 fps) | 27 ms (37 fps) | 673 ms (1.5 fps) |

The 152k case is dominated by fixed overhead. The marginal cost on GPU is about 1.5 ms/Mtri (about 100 us/Mtri in the 1M to 5M slope), versus about 120 ms/Mtri on llvmpipe, a roughly 50-70x gap at 5M. Windows were unmeasured for GPU memory. Not tried: trame server-side-rendered viewer (bounded time).

## Limitations
- `viewer.html` untested in a browser. GLB loads in trimesh (4 geometries, bounds 569x100x50 mm).
- Surface only: no interior/clip cut views, no per-cell scalars, no deformed shape (displacement vector is in the vtu but not exported; the USD could carry it as a point-time-sampled attribute or a vector primvar).
- Tet10 midside nodes dropped, so the boundary is linear. Surface group names (physical groups) are not carried: automesh's .msh only has `domain`. A named-group surface split would give GeomSubsets for BC faces.
- No materials/MaterialX, USD display uses `displayColor` only. usd-core has no Hydra renderer, so USD itself is viewed here via the pxr to PyVista path or by a USD-capable viewer (usdview, Omniverse, Blender).
- Colormap range is per part; an assembly-wide shared range should be a stage-level setting.
