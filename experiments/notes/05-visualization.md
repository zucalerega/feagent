# 05 Visualization for whole-airplane FEA + swarm (GPU-weighted)
Evidence grade: [W]=seen in web search this session, [K]=from model knowledge, NOT re-verified (check before committing). Search tool returned thin results; do not treat [K] as sourced.

## Repo context (read)
- feacore/ui: FastAPI server.py + static SPA (no build step), viewdata.py (field viewer data), swarm.py (ACE swarm page), jobs/worker. Mesh harness renders offscreen PyVista PNGs. python/pyvista not installed in default shell (no `python` on PATH; use venv).
- Constraint (coordinator): harness + UI stay CPU; the VIZ service is GPU-backed and embedded/linked from the FastAPI UI.

## Sources actually seen
- trame-vtk: choose server-side remote rendering or client vtk.js rendering. https://github.com/Kitware/trame-vtk [W]
- trame-mesh-streamer (Kitware, Apache-2.0, v0.1.0): progressive mesh delivery, vtk.js client + VTK/ParaView server. https://github.com/Kitware/trame-mesh-streamer , https://pypi.org/project/trame-mesh-streamer [W]
- Kitware: vtk.js is a VTK subset; VTK.wasm + trame-vtklocal is the newer local-render path. https://kitware.com/vtk-wasm-and-its-trame-integration [W]
- VTK discourse: huge data forced remote rendering; remote fixed many-actor slowness. https://discourse.vtk.org/t/trame-performance-with-many-actors-or-a-large-structured-grid/12773 [W]
- Omniverse licensing: governed by NVIDIA Software License Agreement + product terms; last doc update Jun 16 2026; NGC Kit App Streaming collection asks for NVIDIA AI Enterprise subscription; forum claims free tier of 2 collaborating users, and a May-2026 licensing question for SaaS use is unresolved. https://docs.omniverse.nvidia.com/ovas/latest/common/NVIDIA_Omniverse_License_Agreement.html , https://catalog.ngc.nvidia.com/orgs/nvidia/omniverse/collections/kit-appstreaming-collection/- , https://forums.developer.nvidia.com/t/can-we-use-nvidia-omniverse-sdk-kit-and-nucleus-server-as-saas-in-cloud-for-our-commercial-project/374101 [W] => hackathon use of Kit App Streaming: ASSUME enterprise-gated; verify with NVIDIA.
- Innoactive resells Kit streaming with enterprise license included. https://innoactive.io/integrations/omniverse [W]
- NGC "usd-viewer" container exists (Kit-based USD viewer sample). https://catalog.ngc.nvidia.com/orgs/nvidia/omniverse/containers/usd-viewer/- [W]

## 1. OpenUSD [K]
- `pip install usd-core` (Pixar build, no Hydra/usdview GUI in wheel; has pxr.Usd/UsdGeom/Sdf). Mesh: UsdGeomMesh points/faceVertexCounts/Indices; scalars as `primvars:` (interpolation vertex|uniform(per-face)|faceVarying), display color via displayColor primvar; real colormap done in material/Preview shader or by baking RGB. Time samples = per-load-step/iteration. References/payloads (lazy load) + variants (LOD, mesh-size) + PointInstancer/native instancing for fasteners (airplane has 10^5-10^6 fasteners: huge win). Contract: one .usd per part (payload), assembly stage references them.
- Viewers: usdview (needs full USD build from source/NVIDIA), Blender (imports USD, slow at 10^7), Omniverse Kit/USD Composer (RTX, scales best, licensing above), three.js USDZ loader (limited: no payload/timesample, small scenes), needle/other web USD = immature [K]. Verdict: USD = good archival/assembly interchange + feed for Omniverse; NOT the lowest-effort in-browser path.
- Gotcha: scalar visualization in USD is weak (no first-class colormap/range legend); engineering result browsing (clip, probe, legend) is better in VTK/ParaView.

## 2. Scientific viz [K]
- Rule of thumb: extract boundary surface only (vtkDataSetSurfaceFilter / pyvista extract_surface) -> a 10^7-cell tet mesh ≈ 2-4x10^6 surface tris (P2 midnodes ignored). Per-part LOD via decimation (quadric, preserve scalar) or element-averaged coarse; whole aircraft skin often 10^6-10^7 tris -> OK on modern GPU in native renderers.
- Browser WebGL (three.js/vtk.js): comfortable to ~2-5M tris on laptop/iGPU, one draw call per part, uint32 indices, Float32 scalar attribute + 1D colormap texture (change colormap/range in shader without re-upload). Beyond that: chunk/tile + LOD (Potree-style), frustum culling. WebGPU (Chrome stable; Safari/Firefox 2025-26) adds compute culling, ~2x draw headroom; still bounded by client GPU/download size.
- Server-side GPU (trame + VTK EGL or ParaView pvserver EGL): scene lives on server GPU (4060 8GB: ~10^7-3x10^7 tris with VBOs; L40S 48GB / H100 80GB: >10^8 tris). Client gets JPEG/H.264 stream (trame remote view, WebRTC option); interaction latency fine on LAN/Brev tunnel. Needs VTK built with EGL (vtk-osmesa is CPU; pip `vtk` wheels are X/GLX; for headless GPU use NVIDIA EGL wheel/`vtk-egl`/conda or ParaView `--egl` container). Set `__EGL_VENDOR_LIBRARY_FILENAMES`, NVIDIA_DRIVER_CAPABILITIES=graphics,utility.
- ParaView server (pvserver, EGL) + Catalyst: Catalyst = in-situ extract hook from solver (not viewer); useful later for 10^8-cell solves writing surfaces/slices each iteration. Use ParaView's multi-block + Python (`paraview.simple`) for pipelines; trame ParaView wrappers (`paraview.web`/trame-vtk) embed it.
- NVIDIA IndeX (ParaView plugin): GPU volume rendering of huge volumetric/ unstructured fields, multi-GPU; relevant only if showing interior volume fields (>10^8). Licensing for IndeX in 2026 uncertain [unverified]; skip for hackathon. Surface stress contours do not need it.
- deck.gl: geospatial/point clouds; poor fit for CAD meshes. Skip.
- Kitware newer: VTK.wasm + trame-vtklocal (client WebGL/WebGPU w/ real VTK) [W]; trame-mesh-streamer = progressive refinement [W, v0.1.0 young].

## 3. Fit to FastAPI static UI (least effort -> payoff)
Tier A (1-2 days): per-part boundary surface + scalar -> glTF/GLB (pyvista->trimesh, or custom binary: positions f32, indices u32, scalar f32, plus JSON manifest of part bbox/min/max) served by FastAPI; three.js in static page (import map from CDN or vendored ES module, no build) with custom ShaderMaterial colormap; per-part toggle, click -> run link. Handles ~2-5M tris total; good for per-agent small/medium runs.
Tier B (2-4 days): trame app (own process/port) with VTK EGL on GPU; FastAPI UI embeds via iframe `#/view/<run>` or opens new tab; trame loads precomputed surface .vtp/.vtkhdf per part, uses multi-block, LOD actors (vtkLODActor/ decimation), scalar bar, clip plane. Handles whole airplane on one GPU. Reverse-proxy via FastAPI (or CORS-free iframe on same host). Auth: bind localhost / SSH tunnel on Brev.
Tier C: Omniverse (below).
Swarm overview: ADD a lightweight 2D layer regardless: small multiples (thumbnail PNG from existing PyVista per agent/part, colored by status/score) + a single instanced 3D scene in the viewer where each agent's part tile is color-coded by state (queued/running/converged/failed); data from existing swarm.py/runstore. Live updates: SSE/WebSocket pushing {part_id, status, scalar_url}; viewer swaps scalar buffer only (cheap) rather than geometry.
Data pipeline: solver output (CPU) -> `viz_export.py` (extract_surface, map point/cell scalars, store as .glb + .npy scalar sidecars, per-part LOD0/1/2) -> runstore. Keep geometry static, scalars per timestep/iteration as separate small files (enables time scrubbing and agent comparison for same mesh).

## 4. Recommended stacks (GPU-weighted)
(a) RTX 4060 laptop (8GB VRAM, Optimus hybrid GPU: force NVIDIA via `__NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia` or EGL device select):
  - Primary: trame + VTK (EGL or GLX on the dGPU) server-side render, surface-only + per-part LOD, budget ~10-15M tris (8GB). Embedded in FastAPI UI by iframe. Plus Tier A three.js for lightweight run pages. Fallback: browser WebGPU/WebGL on the same dGPU (Chrome flag `--use-angle=vulkan`/ prime offload on Linux).
  - Omniverse Kit local on 4060 is possible but RTX Real-Time on 8GB is tight for airplane; only for a demo.
(b) L40S / H100 Brev box:
  - L40S: has RT cores + NVENC: best for RTX path (Kit) and trame/ParaView EGL streaming w/ H.264. H100: NO RT cores/ no display-oriented graphics support (compute-only; NVIDIA states H100 not for graphics/ rendering [K]) -> use H100 for solve/ML, render on L40S or another RTX GPU. Important: don't plan EGL rendering on H100.
  - Stack: ParaView 5.13+/6 pvserver EGL (or trame-VTK) in Docker with `--gpus all` + NVIDIA_DRIVER_CAPABILITIES=graphics; USD assembly + payloads; Catalyst hooks if solver goes distributed. Expose via Brev port-forward; FastAPI UI iframe.
(c) Stretch: Omniverse Kit App Streaming (RTX path-traced/ raster, WebRTC to browser, USD assembly with payloads/instancing/time samples). Needs L40S-class RTX GPU, NGC containers, and license confirmation (enterprise-gated per NGC page [W]). Embed the Kit web viewer (ov-web-sdk) in iframe in the FastAPI UI. Dassault-style story: USD as shared scene, results as primvars/time samples. Do only if license/time permit; trame covers 90% of value.

## Recommendation / order of work
1. viz_export (surface + scalar + LOD, USD optional) 2. three.js Tier A for quick payoff 3. trame-EGL GPU viewer service (separate process, iframe) 4. swarm status tiles 5. USD assembly writer (usd-core) w/ PointInstancer for fasteners 6. Omniverse stretch.
Risks: pip VTK wheels lack EGL; Optimus selection; H100 no graphics; Omniverse licensing; vtk.js/trame-mesh-streamer maturity; scalar-on-USD tooling weak. Verify all [K] items.
