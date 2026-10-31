# FEAgent Meshing Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A `feagent` Python library + CLI + MCP server + Claude Code skill/hook that lets an agent read an existing STEP part, select faces by geometric query, mesh it from an editable YAML spec, and prove the mesh with quality gates, a built-in solver smoke test, and a GCI convergence study.

**Architecture:** All gmsh work happens in short-lived processes (CLI) around a single in-process API. Modules communicate through the dataclasses in `types.py` and the `.msh` + `manifest.json` file contract. The MCP server is a thin subprocess wrapper over the CLI's `--json` mode.

**Tech Stack:** Python 3.12, uv, gmsh 4.15 (OCC kernel), numpy/scipy, pyvista/vtk (offscreen), scikit-fem + pyamg, PyYAML, `mcp` (FastMCP), pytest.

**Spec:** `docs/superpowers/specs/2026-10-30-feagent-meshing-harness-design.md` (research: `reports/CAD understanding for agentic meshing.md`)

## Global Constraints

- Python `>=3.12`; run everything with `uv run` from the repo root; tests with `uv run pytest -q`.
- Package lives in `src/feagent/`; tests in `tests/`; no module imports `gmsh` at import time except `gmsh_session.py`, `geometry.py`, `mesher.py`, `meshdata.py` (import inside functions elsewhere).
- Units: model units are millimetres. `load_cad` sets `Geometry.OCCTargetUnit = "MM"`. Forces N, stress MPa.
- Never print to stdout from library code. gmsh/OCC native output is redirected to stderr by `gmsh_session()`. CLI `--json` writes exactly one JSON document to stdout.
- All dataclasses in `types.py` are JSON-round-trippable via `to_dict()` / `from_dict()` (numpy arrays excluded from serialization; tuples serialize as lists).
- Face/volume tags are gmsh OCC tags **of the current session only**; anything persisted refers to faces by group name + query, never by bare tag across runs.
- Gate ids, status strings and query keys below are exact strings; do not rename.
- Subagents do not `git commit`; the coordinator commits after each wave.

## Review Focus

1. Multi-solid STEP (touching bodies) → fragmented conformally; one interface face reported with `volumes` of length 2; connectivity gate expects 1 component. Test: Task 2 `test_two_touching_boxes_fragment`.
2. A group query matching 0 or the wrong number of faces → mesh run fails with an error naming the group, the expected/actual count, and near-miss candidates; never meshes with a wrong BC. Test: Task 3 `test_explain_miss_lists_candidates`, Task 8 `test_group_count_mismatch_fails_cleanly`.
3. Surface-only / no-volume CAD → `MeshResult.ok == False` with a `no_volume` error, no exception, no hang. Test: Task 8 `test_surface_only_step_fails_cleanly`.
4. OCC/gmsh writing to stdout would corrupt MCP stdio and `--json` output. Test: Task 1 `test_load_cad_keeps_stdout_clean` (subprocess, stdout must be empty).
5. Large meshes in the smoke solve → P1 + iterative solver instead of P2 direct, finishing in bounded memory. Test: Task 7 `test_choose_order_thresholds`.

---

## File map

| File | Responsibility | Task |
|---|---|---|
| `src/feagent/types.py` | All shared dataclasses + `to_dict/from_dict` | 1 |
| `src/feagent/errors.py` | `FeagentError(message, hint)`, `CadLoadError`, `QueryError`, `SpecError` | 1 |
| `src/feagent/gmsh_session.py` | `gmsh_session()`, `load_cad()`, `gmsh_messages()` | 1 |
| `src/feagent/meshdata.py` | `read_msh()` → `MeshData` (numpy) | 1 |
| `src/feagent/parts.py` | Six synthetic test parts → STEP | 1 |
| `src/feagent/geometry.py` | Analysis, tessellation, features, hazards, `digest_text` | 2 |
| `src/feagent/select.py` | Face query language | 3 |
| `src/feagent/spec.py` | YAML mesh spec → `MeshSpec` | 4 |
| `src/feagent/quality.py` | Mesh quality metrics + mesh gates | 5 |
| `src/feagent/render.py` | PNG renders with captions | 6 |
| `src/feagent/solve.py` | scikit-fem FEA + potential-flow smoke tests | 7 |
| `src/feagent/mesher.py` | Spec → domain → groups → sizing → mesh → exports | 8 |
| `src/feagent/converge.py` | GCI (Celik 2008) 3-grid study | 9 |
| `src/feagent/check.py` | Gate orchestration, report.json/REPORT.md, state file | 9 |
| `src/feagent/cli.py` | `feagent` CLI | 9 |
| `src/feagent/mcp_server.py` | FastMCP server over the CLI | 10 |
| `src/feagent/hook.py` | Stop-hook logic | 10 |
| `.claude/skills/fe-mesh/SKILL.md`, `.mcp.json`, `.claude/settings.json`, `examples/*/mesh.yaml` | Agent integration + worked examples | 10 |

Waves: **W1** Task 1 → **W2** Tasks 2–7 in parallel → **W3** Task 8 → Task 9 → **W4** Task 10.

---

### Task 1: Foundation — types, errors, gmsh session, mesh reader, test parts

**Files:**
- Modify: `pyproject.toml` (add deps `pyamg`; add `[project.scripts] feagent = "feagent.cli:main"`; description), create `.gitignore` (`.venv/`, `__pycache__/`, `out/`, `.feagent/`, `*.msh` under `examples/**/out`, `.pytest_cache/`)
- Create: `src/feagent/{__init__.py,types.py,errors.py,gmsh_session.py,meshdata.py,parts.py}`, `src/feagent/cli.py` (stub `def main(argv=None) -> int: return 0` — Task 9 replaces)
- Test: `tests/conftest.py`, `tests/test_foundation.py`

**Interfaces — Produces (exact):**

```python
# errors.py
class FeagentError(Exception):
    def __init__(self, message: str, hint: str = ""): ...   # .message, .hint
class CadLoadError(FeagentError): ...
class QueryError(FeagentError): ...
class SpecError(FeagentError): ...   # message starts with the dotted field path, e.g. "groups.fixed.expect: ..."

# types.py   (Vec3 = tuple[float, float, float]; BBox = tuple[Vec3, Vec3])
@dataclass class FaceInfo:
    tag: int; kind: str            # "plane","cylinder","cone","sphere","torus","bspline","revolution","extrusion","other"
    area: float; center: Vec3; bbox: BBox
    normal: Vec3 | None            # outward unit normal at a point on the face (exact for planes)
    radius: float | None           # cylinder/sphere radius; cone: at sample point; torus: minor radius
    axis: Vec3 | None              # unit axis (cylinder/cone/torus), sign arbitrary but canonical: first nonzero comp > 0
    axis_point: Vec3 | None        # point on axis closest to origin
    concave: bool | None           # True = material is outside the cylinder (hole, fillet); False = boss/round
    angle_span: float | None       # degrees of arc covered about the axis (cylinders/cones/tori)
    volumes: list[int]; edges: list[int]; neighbors: list[int]
    thickness: float | None = None       # median inward ray distance (local wall thickness)
    thickness_min: float | None = None
    gap: float | None = None             # min outward ray distance to another surface
@dataclass class EdgeInfo: tag: int; kind: str; length: float; faces: list[int]; bbox: BBox
@dataclass class VolumeInfo: tag: int; volume: float; center: Vec3; bbox: BBox; faces: list[int]
@dataclass class Feature: kind: str; faces: list[int]; params: dict   # kind in "hole","boss","fillet","round"
@dataclass class Hazard:
    kind: str     # "small_face","short_edge","sliver_face","thin_wall","narrow_gap","small_fillet","no_volume","multi_volume_interface"
    entities: list[tuple[int, int]]; value: float | None; threshold: float | None
    location: Vec3 | None; message: str; suggestion: str
@dataclass class Tessellation:          # NOT serialized
    points: np.ndarray        # (N,3) float
    triangles: np.ndarray     # (M,3) int, 0-based into points, oriented outward
    face_tag: np.ndarray      # (M,) int
    normals: np.ndarray       # (M,3) outward unit
@dataclass class GeometryModel:
    source: str; units: str; bbox: BBox; diag: float; ref_size: float
    volumes: list[VolumeInfo]; faces: list[FaceInfo]; edges: list[EdgeInfo]
    features: list[Feature]; hazards: list[Hazard]
    tessellation: Tessellation | None = None
    def face(self, tag: int) -> FaceInfo: ...       # KeyError if missing
@dataclass class Gate:
    id: str; name: str; status: str     # "pass" | "warn" | "fail" | "skip"
    value: float | None; threshold: float | None; message: str; hint: str = ""
@dataclass class GroupResult: name: str; dim: int; tags: list[int]; elements: int; area: float | None
@dataclass class MeshResult:
    ok: bool; spec_path: str; out_dir: str; mesh_path: str | None
    exports: dict[str, str]; groups: dict[str, GroupResult]
    n_nodes: int; n_elements: int; order: int; size_scale: float; elapsed_s: float
    errors: list[str]; warnings: list[str]; gates: list[Gate]   # gates: "geometry","groups","mesh_generated"
    geometry: GeometryModel | None
# every dataclass above: def to_dict(self) -> dict ; @classmethod def from_dict(cls, d) -> Self
def overall_status(gates: list[Gate]) -> str   # "fail" if any fail, else "pass"

# gmsh_session.py
@contextmanager
def gmsh_session(model_name: str = "feagent") -> Iterator[None]:
    """initialize(interruptible=False), General.Terminal=1 with fd 1 dup2'd to fd 2 for the whole block,
    gmsh.logger.start(); finalize on exit (also on exceptions)."""
def load_cad(path: str | Path, heal: bool = False) -> list[tuple[int, int]]:
    """Import STEP/STP/IGES/IGS/BREP into the current model (OCCTargetUnit="MM", highestDimOnly=True).
    If >1 volume: occ.fragment all volumes (conformal interfaces). heal=True: occ.healShapes() BEFORE fragment,
    then occ.removeAllDuplicates(). synchronize(). Returns 3D dimTags (may be []). Raises CadLoadError for missing
    file / unsupported extension (hint lists supported) / import failure (message includes gmsh errors)."""
def gmsh_messages(level: str = "Error") -> list[str]   # logger lines containing level

# meshdata.py
@dataclass class MeshData:
    points: np.ndarray            # (N,3)
    tets: np.ndarray              # (M,4) int, 0-based corner nodes (order-2 meshes: corners only)
    tet_tags: np.ndarray          # (M,) gmsh element tags
    tet_volume: np.ndarray        # (M,) int, gmsh volume entity tag per tet
    tris: dict[str, np.ndarray]   # physical-group name -> (K,3) 0-based corner nodes (dim-2 groups)
    tri_face: dict[str, np.ndarray]  # name -> (K,) gmsh surface entity tag per triangle
    volume_groups: dict[str, list[int]]  # dim-3 physical name -> volume entity tags
    order: int
def read_msh(path: str | Path) -> MeshData   # via gmsh.open inside gmsh_session; KeyError-free: missing groups -> {}

# parts.py
PARTS: tuple[str, ...] = ("cantilever","plate_hole","bracket","thin_cup","sliver","tee","bullet")
def make_part(name: str, path: str | Path) -> Path      # writes STEP; ValueError for unknown name
def make_all(out_dir: str | Path) -> dict[str, Path]
```

Part geometry (mm, exact):
- `cantilever`: box x∈[0,100], y∈[0,10], z∈[0,10].
- `plate_hole`: box 200×100×5 at origin, through hole r=10 axis z at (100,50).
- `bracket`: base box x∈[0,80],y∈[0,40],z∈[0,6] ∪ upright x∈[0,6],y∈[0,40],z∈[0,50]; fillet r=4 on the inner corner edge (x=6,z=6); 4 through holes r=3.25 axis z at (x,y)∈{(25,10),(25,30),(65,10),(65,30)}; 1 through hole r=5 axis x at (y=20,z=40).
- `thin_cup`: box 60×40×20 minus box x∈[1,59],y∈[1,39],z∈[1,21] (open top, wall 1.0).
- `sliver`: box 50×50×10 ∪ box x∈[10,30],y∈[10,30],z∈[0,10.05] (0.05 step → sliver side faces, short edges).
- `tee` (fluid volume): cylinder r=10 axis x from x=0 to 100 at (y=0,z=0) ∪ cylinder r=6 axis y from y=0 to 50 at (x=50,z=0).
- `bullet` (external-flow body): cylinder r=5 axis x from x=5 to 35 ∪ sphere r=5 at (5,0,0).

- [ ] **Step 1: Write failing tests** in `tests/test_foundation.py`; `tests/conftest.py` provides session-scoped fixture `parts_dir(tmp_path_factory) -> Path` = `make_all(...)`.
  - `test_make_all_writes_step_files`: all 7 names exist, size > 0.
  - `test_load_cad_returns_volumes`: `bracket` → 1 volume; `cantilever` volume mass (`occ.getMass(3,tag)`) == 10000 ± 1e-6.
  - `test_load_cad_missing_file`: raises `CadLoadError`, `.hint` non-empty.
  - `test_load_cad_keeps_stdout_clean`: run `sys.executable -c "from feagent.gmsh_session import *; ..."` in a subprocess loading `bracket.step`; `stdout == ""`.
  - `test_types_roundtrip`: build a `MeshResult` containing a `GeometryModel` with one face/hazard/gate; `MeshResult.from_dict(json.loads(json.dumps(r.to_dict()))) == r` (tessellation None).
  - `test_read_msh_groups`: in-test gmsh: mesh `cantilever` (size 5), physical surface "fixed" = face at x=0, physical volume "domain"; write msh; `read_msh` → `tets.shape[1]==4`, `"fixed" in tris`, all `tris["fixed"][:,:]` points have x≈0, `volume_groups["domain"]` non-empty, `order==1`.
- [ ] **Step 2:** `uv run pytest tests/test_foundation.py -q` → FAIL (ImportError).
- [ ] **Step 3:** Implement the files above; `uv add pyamg`.
- [ ] **Step 4:** `uv run pytest tests/test_foundation.py -q` → all PASS.

---

### Task 2: Geometry analysis and digest (`geometry.py`)

**Interfaces:**
- Consumes: `types.*`, `gmsh_session.gmsh_session/load_cad`, `errors.CadLoadError`.
- Produces:
```python
def analyze_current_model(ref_size: float | None = None, source: str = "", tessellate: bool = True) -> GeometryModel
    # operates on the model in the CURRENT gmsh session; leaves no mesh behind (mesh.clear()).
    # ref_size default = diag / 40. Uses a temporary 2D tessellation (MeshSizeMax = min(ref_size, diag/25),
    # MeshSizeFromCurvature = 12) for thickness/gap rays and FaceInfo samples.
def analyze(path: str | Path, ref_size: float | None = None, heal: bool = False, tessellate: bool = True) -> GeometryModel
def digest_text(model: GeometryModel, detail: str = "summary", max_faces: int = 60) -> str   # detail: "summary" | "full"
```
Approach (decided):
- Outward normal: take an interior node of the face's tessellation (exact surface point + its uv), `getNormal(tag, uv)`; flip if `isInside(3, vol, p + δn)` with δ = 1e-4·diag. Orient that face's triangles by the same sign.
- Cylinder axis: from `getPrincipalCurvatures`, the axis is the direction paired with the curvature of **smallest** |k|; radius = 1/max|k|. Curvature **sign is not usable** for concavity (gmsh gives the same sign for holes and bosses). Concavity: candidate axis points `p + r·n_out` (concave) and `p − r·n_out` (convex); pick the one whose distance-to-axis from 3 other face samples is closest to r.
- `angle_span`: angular extent of the sampled points around the axis.
- Thickness: vtk `vtkOBBTree` over the tessellation; for ≤50 triangles per face, ray from centroid along −normal (length diag), first hit beyond 1e-6·diag excluding the source cell; `thickness` = median, `thickness_min` = min. `gap`: same along +normal, only hits on other faces.
- Features: concave cylinders grouped by (radius ±1e-3·r, collinear axis) → `hole` with `params={radius, axis, axis_point, depth, angle_span}` (faces of a split hole merge). `angle_span ≥ 300` → hole/boss; concave & span < 300 & radius < 0.1·diag → `fillet`; convex & span < 300 → `round`; convex & span ≥ 300 → `boss`.
- Hazards (h = ref_size): `small_face` area < (0.5h)²; `short_edge` length < 0.25h; `sliver_face` 2A/P < 0.25h and not already small; `thin_wall` thickness < 2h (value = thickness); `narrow_gap` gap < 2h; `small_fillet` fillet radius < 0.5h; `no_volume` if no 3D entity; `multi_volume_interface` for faces with 2 volumes. Each with message + concrete suggestion text (e.g. thin wall: "local size ≤ t/2 on faces [..] or element order 2 (spec: size.thin_wall_layers)").
- `digest_text` summary sections in this order: header (source, units, bbox, size, diag, volumes, total area, face/edge counts by kind, ref_size), `FEATURES` (patterns collapsed: "4× hole r=3.25 axis +z depth 6.0 faces [..]"), `PLANAR FACES` grouped by outward normal (rounded, ±axis labels like `+z`) with area and plane offset, `HAZARDS`, then `FACES` table (tag kind area center normal|axis r concave nbrs) truncated to `max_faces` largest with "… N more (use detail=full)". Full = all faces + all edges with hazards.

- [ ] **Step 1: Failing tests** `tests/test_geometry.py` (use `parts_dir`):
  - `test_plate_hole_detects_hole`: one `hole` feature, radius 10 ± 1e-6·… (abs 1e-3), axis ≈ (0,0,1), depth 5 ± 1e-3; the big plate faces have normals ±z with area 200·100 − π·100.
  - `test_bracket_features`: 4 holes r=3.25 axis z, 1 hole r=5 axis x, ≥1 `fillet` r=4 with angle_span 90 ± 2.
  - `test_outward_normals_cantilever`: face at x=0 has normal (−1,0,0); x=100 → (+1,0,0).
  - `test_thin_cup_thin_wall_hazard`: `analyze(thin_cup, ref_size=4)` has `thin_wall` hazards; reported thickness 1.0 ± 0.1.
  - `test_sliver_hazards`: `analyze(sliver, ref_size=2)` has `sliver_face` or `small_face`, and `short_edge` with value ≈ 0.05.
  - `test_two_touching_boxes_fragment`: in-test STEP of two boxes sharing face x=10 → 2 volumes, exactly one face with `len(volumes)==2`, a `multi_volume_interface` hazard.
  - `test_digest_text_sections`: bracket summary contains "FEATURES", "HAZARDS", "4× hole", and is < 6000 chars.
  - `test_no_mesh_left_behind`: after `analyze_current_model` in a session, `gmsh.model.mesh.getNodes()[0].size == 0`.
- [ ] **Step 2:** run → FAIL. **Step 3:** implement. **Step 4:** `uv run pytest tests/test_geometry.py -q` → PASS.

---

### Task 3: Face query language (`select.py`)

**Interfaces:**
- Consumes: `GeometryModel`, `FaceInfo`, `QueryError`.
- Produces:
```python
QUERY_KEYS = ("ids","kind","normal","axis","tol_deg","radius","radius_tol","radius_min","radius_max",
              "area_min","area_max","concave","at","box","near","within","feature","adjacent_to",
              "largest","smallest","any","all","not")
def select_faces(model: GeometryModel, query: dict) -> list[int]   # sorted tags; QueryError on unknown key / bad value
def explain_miss(model: GeometryModel, query: dict, expected: int | tuple[int, int] | None) -> str
def describe_faces(model: GeometryModel, tags: list[int]) -> str   # one line per face, same columns as digest FACES
```
Semantics (exact): keys AND together. `ids` list[int]. `kind` str|list. `normal` [x,y,z] vs outward normal, `axis` [x,y,z] sign-agnostic, both within `tol_deg` (default 5). `radius` with `radius_tol` (default max(1e-3·radius, 1e-6·diag)). `concave` bool. `at` ∈ {"xmin","xmax","ymin","ymax","zmin","zmax"}: face bbox touches the model bbox extreme within 1e-6·diag + 1e-9. `box` [[x0,y0,z0],[x1,y1,z1]]: face center inside. `near` [x,y,z] with `within` (default 1e-3·diag): distance from point to face center **or** face bbox (whichever smaller). `feature` str: face belongs to a `Feature` of that kind. `adjacent_to` sub-query: neighbor of any match. `largest`/`smallest` int n: applied last, by area. `any` list[query], `all` list[query], `not` query. Empty query `{}` = all faces.
`explain_miss`: states actual vs expected count; if 0 matches, re-evaluates each key alone and with `tol_deg` ×3 / tolerances ×10, listing up to 5 nearest candidates via `describe_faces` with which keys they fail and by how much (e.g. "normal off by 7.2°").

- [ ] **Step 1: Failing tests** `tests/test_select.py` with a **synthetic** `GeometryModel` fixture (unit cube: 6 plane faces tags 1–6 with correct normals/bboxes/areas/neighbors, plus a concave cylinder face tag 7 r=0.1 axis z feature `hole`; no gmsh needed):
  - `test_normal_and_at`: `{"normal":[1,0,0]}` → [x+ face]; `{"at":"xmin"}` → [x− face].
  - `test_kind_radius_feature`: `{"kind":"cylinder","radius":0.1}` == `{"feature":"hole"}` == [7].
  - `test_combinators`: `{"not":{"kind":"plane"}}` → [7]; `{"any":[{"at":"xmin"},{"at":"xmax"}]}` → 2 faces; `{"adjacent_to":{"ids":[7]}}` = neighbors of 7.
  - `test_largest`: `{"kind":"plane","largest":2}` returns 2 tags.
  - `test_unknown_key_raises`: `{"nromal":[1,0,0]}` → `QueryError` whose hint mentions "normal".
  - `test_explain_miss_lists_candidates`: `{"normal":[0.9962,0.0872,0], "tol_deg":2}` (5° off +x) → text contains "expected 1", "0 matched", the +x face tag, and "off by 5".
- [ ] **Step 2:** FAIL. **Step 3:** implement. **Step 4:** `uv run pytest tests/test_select.py -q` → PASS.

---

### Task 4: Mesh spec (`spec.py`)

**Interfaces:**
- Consumes: `SpecError`, `select.QUERY_KEYS` (import lazily for validation of top-level query keys only).
- Produces:
```python
@dataclass class GroupSpec: name: str; select: dict; expect: int | tuple[int, int] | None
@dataclass class RefineSpec: size: float; distance: float; group: str | None = None; select: dict | None = None
@dataclass class SizeSpec: global_: float; min: float = 0.0; curvature: int = 12; thin_wall_layers: int = 2
@dataclass class DomainSpec: type: str = "internal"; flow_axis: str = "+x"; upstream: float = 3.0; downstream: float = 6.0; lateral: float = 2.0
@dataclass class QualitySpec: min_sicn: float = 0.1; min_gamma: float = 0.05; max_volume_error: float = 0.01; max_area_error: float = 0.02; max_non_ortho: float = 70.0; max_skewness: float = 4.0; thin_wall_min_elements: int = 2
@dataclass class FeaLoad: group: str; force: tuple[float, float, float]
@dataclass class FeaSolveSpec: fixed: list[str]; loads: list[FeaLoad]; E: float = 210000.0; nu: float = 0.3
@dataclass class CfdSolveSpec: inlet: list[str]; outlet: list[str]; velocity: float = 1.0
@dataclass class MeshSpec:
    path: Path; name: str; geometry: Path; analysis: str; heal: bool
    domain: DomainSpec; groups: dict[str, GroupSpec]; default_group: str | None
    size: SizeSpec; refine: list[RefineSpec]
    order: int; optimize: str; algorithm_2d: int; algorithm_3d: int
    export: list[str]; quality: QualitySpec
    fea: FeaSolveSpec | None; cfd: CfdSolveSpec | None; output: Path
    def out_dir(self) -> Path    # (path.parent / output).resolve()
def load_spec(path: str | Path) -> MeshSpec
def parse_spec(data: dict, path: Path) -> MeshSpec
```
YAML schema (exact keys; unknown keys → `SpecError` naming the key and the allowed set):
```yaml
name: bracket                 # default: spec file's parent dir name
geometry: ../../parts/bracket.step   # relative to the spec file; must exist
analysis: fea                 # fea | cfd
heal: false
domain: {type: internal, flow_axis: "+x", upstream: 3, downstream: 6, lateral: 2}   # cfd only; external = box minus body
groups:
  fixed: {select: {at: xmin, kind: plane}, expect: 1}     # expect: int | [min, max] | omitted
default_group: null           # cfd default "walls"; name for unassigned boundary faces
size: {global: 4.0, min: 0.0, curvature: 12, thin_wall_layers: 2}   # global required > 0
refine: [{group: fixed, size: 1.0, distance: 5.0}]       # exactly one of group|select
element: {order: 1, optimize: netgen}                     # order 1|2; optimize none|default|netgen
algorithm: {mesh2d: 6, mesh3d: 1}
export: [msh]                 # subset of msh, msh22, inp, vtk, unv, su2
quality: {min_sicn: 0.1, ...}
solve:
  fea: {E: 210000, nu: 0.3, fixed: [fixed], loads: [{group: load, force: [0, 0, -1000]}]}
  cfd: {inlet: [inlet], outlet: [outlet], velocity: 1.0}
output: out
```
Validation: groups referenced by `refine.group`, `solve.fea.fixed/loads`, `solve.cfd.inlet/outlet` must exist; `solve.fea` only with `analysis: fea`, `solve.cfd` only with `cfd`; `default_group` must not collide with a group name; top-level keys of each `select` must be in `QUERY_KEYS`.

- [ ] **Step 1: Failing tests** `tests/test_spec.py` (tmp YAML files; geometry file = an empty touched file):
  - `test_minimal_spec_defaults`: only `geometry`, `analysis: fea`, `size: {global: 3}` → order 1, optimize "netgen", export ["msh"], quality defaults, `out_dir()` = spec dir / "out".
  - `test_cfd_default_group_walls`.
  - `test_unknown_key_error_names_path`: `size: {globl: 3}` → `SpecError` message starts with "size.globl".
  - `test_missing_group_reference`: `solve.fea.fixed: [nope]` → `SpecError` mentioning "nope".
  - `test_expect_range`: `expect: [2, 4]` → `(2, 4)`.
  - `test_geometry_must_exist`.
- [ ] **Step 2:** FAIL. **Step 3:** implement. **Step 4:** `uv run pytest tests/test_spec.py -q` → PASS.

---

### Task 5: Mesh quality and mesh gates (`quality.py`)

**Interfaces:**
- Consumes: `MeshData`/`read_msh`, `MeshResult`, `GeometryModel`, `Gate`, `spec.QualitySpec` (duck-typed: attributes only).
- Produces:
```python
@dataclass class Stats: min: float; p01: float; p05: float; median: float; max: float; bins: list[float]; hist: list[int]   # 10 bins
@dataclass class WorstElement: tag: int; sicn: float; gamma: float; centroid: Vec3; faces: list[int]   # CAD faces touched (via node classification)
@dataclass class QualityReport:
    n_nodes: int; n_tets: int; sicn: Stats; gamma: Stats; sj: Stats; inverted: int
    worst: list[WorstElement]                     # 10 lowest SICN
    volume_cad: float; volume_mesh: float; volume_error: float
    group_area: dict[str, tuple[float, float, float]]   # name -> (cad, mesh, rel_err)
    components: int; expected_components: int
    non_ortho: Stats | None; skewness: Stats | None     # cfd only, OpenFOAM definitions on tet dual
    sicn_per_tet: np.ndarray                      # not serialized, aligned with MeshData.tets
    gates: list[Gate]
    def to_dict(self) -> dict
def assess_mesh(mesh_path: str | Path, result: MeshResult, q: "QualitySpec", analysis: str) -> QualityReport
def tet_non_ortho_skewness(points: np.ndarray, tets: np.ndarray) -> tuple[np.ndarray, np.ndarray]   # per internal face, degrees / OpenFOAM skewness
```
Gates produced (exact ids): `inverted` (fail if >0), `sicn` (fail if min < q.min_sicn), `gamma` (warn if min < q.min_gamma), `volume_fidelity` (fail if > q.max_volume_error), `area_fidelity` (fail if any group > q.max_area_error; skip if no groups), `connectivity` (fail if components ≠ expected; expected = clusters of volumes sharing a face in `result.geometry`), `thin_walls` (for each `thin_wall` hazard face: mean surface-triangle edge length on that face must be ≤ thickness / q.thin_wall_min_elements, else warn listing faces), `non_ortho` (cfd: fail if max > q.max_non_ortho, warn if > 65), `skewness` (cfd: fail if max > q.max_skewness). Every gate message cites the measured value; hint names the spec key to change.
Element quality via `gmsh.model.mesh.getElementQualities(tags, "minSICN"|"gamma"|"minSJ")` and volumes via `"volume"` in a session on the opened msh. Area per group from triangle areas; CAD area = sum of `FaceInfo.area` over `GroupResult.tags`.

- [ ] **Step 1: Failing tests** `tests/test_quality.py` — build fixtures in-test with gmsh (cantilever, size 4, physical "fixed" at x=0 and "domain"), and a hand-built `MeshResult` + `GeometryModel` (one VolumeInfo volume=10000, face 1 area 100):
  - `test_good_mesh_passes`: no gate `fail`; `inverted==0`; `volume_error < 1e-6`; `group_area["fixed"][2] < 1e-6`.
  - `test_volume_mismatch_fails`: manifest volume 12000 → `volume_fidelity` fail.
  - `test_two_disjoint_boxes_connectivity`: two separate boxes, expected 1 → `connectivity` fail; expected 2 → pass.
  - `test_non_ortho_regular_tet`: a single regular-ish 2-tet pair gives non_ortho < 40°.
  - `test_cfd_gates_present`: analysis "cfd" → gate ids include `non_ortho`, `skewness`.
- [ ] **Step 2:** FAIL. **Step 3:** implement. **Step 4:** `uv run pytest tests/test_quality.py -q` → PASS.

---

### Task 6: Renders (`render.py`)

**Interfaces:**
- Consumes: `GeometryModel` (with `tessellation`), `MeshData`, optional `QualityReport.sicn_per_tet`.
- Produces:
```python
@dataclass class RenderResult: path: str; caption: str      # to_dict/from_dict
def render_faces(model: GeometryModel, out_png: str | Path, highlight: list[int] | None = None, max_labels: int = 60) -> RenderResult
def render_groups(mesh: MeshData, out_png: str | Path) -> RenderResult
def render_quality(mesh: MeshData, sicn_per_tet: np.ndarray, out_png: str | Path, clip_axis: str = "z") -> RenderResult
def render_field(mesh: MeshData, point_values: np.ndarray, name: str, out_png: str | Path, warp: np.ndarray | None = None) -> RenderResult
```
All: pyvista offscreen, 2×2 subplots (iso, +x, +y, +z views; `render_quality` uses iso + 3 clip views at bbox mid-planes), 1600×1200 px. `render_faces`: categorical colors per face; labels "f<tag>" at face centers for the `max_labels` largest faces plus all highlighted; with `highlight`, highlighted faces red, others light grey at opacity 0.35. `render_groups`: one color per group with legend. `render_quality`: tets colored by SICN (0..1, red low), worst 10 marked with spheres. Captions are plain text with numbers (e.g. "4 groups: fixed (1 face, 212 tris), …; highlighted f12,f13").

- [ ] **Step 1: Failing tests** `tests/test_render.py` with synthetic `Tessellation` (two triangles forming a unit square, face tags 1,2) and a 1-tet `MeshData`:
  - `test_render_faces_png`: file exists, PIL/pyvista-readable, width 1600; caption mentions "f1".
  - `test_render_faces_highlight_caption`: highlight [2] → caption contains "highlighted f2".
  - `test_render_groups_and_quality_png`: both PNGs written.
- [ ] **Step 2:** FAIL. **Step 3:** implement. **Step 4:** `uv run pytest tests/test_render.py -q` → PASS.

---

### Task 7: Solver smoke tests (`solve.py`)

**Interfaces:**
- Consumes: `MeshData`, `Gate`, `spec.FeaSolveSpec` / `spec.CfdSolveSpec` (duck-typed attributes).
- Produces:
```python
@dataclass class FeaResult:
    ok: bool; order: int; n_dofs: int; elapsed_s: float
    max_disp: float; max_disp_at: Vec3; compliance: float
    applied: Vec3; reaction_sum: Vec3; vm_max: float; vm_p999: float; vm_max_at: Vec3; vm_max_tet: int
    gates: list[Gate]
    disp: np.ndarray        # (N,3) at mesh points, not serialized
    vm_tet: np.ndarray      # (M,) not serialized
    def to_dict(self) -> dict
@dataclass class CfdResult:
    ok: bool; n_dofs: int; elapsed_s: float; inflow: float; outflow: float; imbalance: float
    u_inlet_mean: float; u_max: float; u_ratio: float; u_max_at: Vec3; gates: list[Gate]
    speed: np.ndarray       # (N,) not serialized
    def to_dict(self) -> dict
def choose_order(n_points: int, n_tets: int) -> int        # 2 if 3*(n_points + 1.2*n_tets) <= 600_000 (≈ P2 vector dofs) else 1
def solve_fea(mesh: MeshData, cfg, order: int | None = None, sicn_per_tet: np.ndarray | None = None) -> FeaResult
def solve_potential_flow(mesh: MeshData, cfg) -> CfdResult
```
Approach (decided): scikit-fem `MeshTet(points.T, tets.T)`; boundary facets per group by matching sorted triangle corner triples to `mesh.facets`; `ElementVector(ElementTetP2())` (or P1); `linear_elasticity(*lame_parameters(E, nu))`; uniform traction = force / group area on each load group; direct `solve` when n_dofs ≤ 150_000 else `pyamg.smoothed_aggregation_solver` with rigid-body near-nullspace + CG (tol 1e-8). Reactions r = K u − f over fixed dofs.
FEA gates (exact ids): `solve_ran` (fail on exception/NaN), `equilibrium` (fail if |Σreaction + Σapplied| / |Σapplied| > 1e-6), `load_path` (fail if max_disp ≤ 0 or max-displacement point lies on a fixed face), `stress_artifact` (warn if vm_max > 3·vm_p999 **and** the max tet's SICN is in the worst 1% — message: likely mesh artifact or re-entrant-corner singularity, names location).
CFD (Laplace φ, ∂φ/∂n = −U on inlet, φ=0 on outlet, natural elsewhere, P1): gates `solve_ran`, `mass_balance` (fail if |in − out| / in > 1e-6), `velocity_ratio` (warn if u_max / U_inlet_mean > 10: possible tiny gap or singular corner).

- [ ] **Step 1: Failing tests** `tests/test_solve.py` (fixture: gmsh-mesh `cantilever` size 2.5 with physical "fixed" x=0, "load" x=100, "domain"; and `tee` size 2 with "inlet" x=0, "outlet" = faces at x=100 and y=50, "domain"):
  - `test_cantilever_tip_deflection`: P2, load (0,0,−100), E 210000, ν 0.3 → `max_disp` in [0.188, 0.197] (Euler–Bernoulli 0.1905 + shear ≈ 0.192); `equilibrium` pass.
  - `test_reaction_balances_applied`: `reaction_sum` ≈ (0,0,+100) within 1e-6 relative.
  - `test_choose_order_thresholds`: `choose_order(1_000, 5_000) == 2`; `choose_order(400_000, 2_400_000) == 1`.
  - `test_tee_mass_balance`: `imbalance < 1e-6`; `inflow ≈ U·π·10²` within 2 %.
- [ ] **Step 2:** FAIL. **Step 3:** implement. **Step 4:** `uv run pytest tests/test_solve.py -q` → PASS.

---

### Task 8: Mesher (`mesher.py`)

**Interfaces:**
- Consumes: Task 1 all; `geometry.analyze_current_model`; `select.select_faces/explain_miss`; `spec.MeshSpec/load_spec`.
- Produces:
```python
def build_domain(spec: MeshSpec) -> list[tuple[int, int]]   # in current session: load_cad; for cfd external: box (padding from DomainSpec, in multiples of body size along flow axis / max lateral extent) cut by body volumes; returns fluid/solid volume dimTags
def run_mesh(spec: MeshSpec, size_scale: float = 1.0, out_dir: Path | None = None) -> MeshResult
def run_mesh_subprocess(spec_path: str | Path, size_scale: float = 1.0, timeout: float = 900) -> MeshResult   # `python -m feagent.mesher SPEC --scale S --json` ; timeout -> ok False, error "mesh timed out after Ns"
```
File contract written to `out_dir` (default `spec.out_dir()`; for `size_scale != 1`: `out_dir / f"scale_{size_scale:g}"`): `mesh.msh` (MSH 4.1 ASCII), each export (`mesh.inp` with `Mesh.SaveGroupsOfNodes=1`, `mesh_v22.msh` for msh22, `mesh.vtk`, `mesh.unv`, `mesh.su2`), `manifest.json` (= `MeshResult.to_dict()`), `mesh.log` (gmsh logger).
Physical groups: each spec group → dim-2 physical with its name; `default_group` → all remaining boundary faces (faces with 1 volume); all volumes → dim-3 physical `"domain"`. A face in two groups → error "faces [..] are in groups a and b". Group resolution uses `analyze_current_model(ref_size=spec.size.global_*scale)` **after** `build_domain`; count mismatch → error = `explain_miss(...)` text, `ok=False`, no mesh.
Sizing: `Mesh.MeshSizeMax = global·s`, `Mesh.MeshSizeMin = min·s`, `Mesh.MeshSizeFromCurvature = curvature`, `Mesh.MeshSizeFromPoints = 0`; each `refine` → Distance(SurfacesList) + Threshold(SizeMin=size·s, SizeMax=global·s, DistMin=0, DistMax=distance); `thin_wall_layers > 0` → for each `thin_wall` hazard face a Threshold with SizeMin = thickness / layers, DistMax = 2·thickness; combined with a Min field as background mesh. `Mesh.Algorithm = algorithm_2d`, `Mesh.Algorithm3D = algorithm_3d`, generate(3), `optimize("Netgen")` if `optimize == "netgen"` (or `""` for default), `setOrder(order)` (+ `optimize("HighOrder")` for order 2).
Gates on `MeshResult`: `geometry` (fail on CadLoadError or `no_volume`), `groups` (fail on resolution errors), `mesh_generated` (fail if gmsh raised or 0 tets). gmsh exceptions caught → `errors` includes last gmsh error lines.

- [ ] **Step 1: Failing tests** `tests/test_mesher.py` (write specs into tmp dirs pointing at `parts_dir`):
  - `test_bracket_fea_mesh`: groups `mount` = `{feature: hole, radius: 3.25}` expect 4..8 faces (split cylinders) — use `expect: [4, 8]`, `load` = `{feature: hole, radius: 5}` expect `[1, 2]`; ok, msh exists, `n_elements > 1000`, groups non-empty, manifest round-trips.
  - `test_refine_increases_elements`: same spec with `refine: [{group: mount, size: 0.5, distance: 3}]` → more elements than without.
  - `test_group_count_mismatch_fails_cleanly`: `expect: 3` for a 1-face group → `ok False`, `groups` gate fail, error mentions "expected 3".
  - `test_surface_only_step_fails_cleanly`: STEP containing only a rectangle surface → `ok False`, `geometry` gate fail, no exception.
  - `test_cfd_external_bullet`: `analysis: cfd`, domain external; groups `inlet` `{at: xmin}`, `outlet` `{at: xmax}`; default_group walls → ok, physical groups include inlet/outlet/walls, 1 volume.
  - `test_scale_writes_subdir`: `run_mesh(spec, 1.3)` writes to `out/scale_1.3/`.
  - `test_subprocess_matches`: `run_mesh_subprocess` returns ok and same group names.
- [ ] **Step 2:** FAIL. **Step 3:** implement (incl. `if __name__ == "__main__"` entry). **Step 4:** `uv run pytest tests/test_mesher.py -q` → PASS.

---

### Task 9: Convergence, check orchestration, CLI

**Interfaces:**
- Consumes: everything above.
- Produces:
```python
# converge.py
@dataclass class GciResult: p: float; extrapolated: float; gci_fine: float; r21: float; r32: float; behavior: str   # "monotonic"|"oscillatory"|"divergent"
@dataclass class ConvergenceResult: scales: list[float]; n_elements: list[int]; quantities: dict[str, list[float]]; gci: dict[str, GciResult]; gates: list[Gate]
def gci3(f1: float, f2: float, f3: float, n1: int, n2: int, n3: int, dim: int = 3) -> GciResult   # f1 finest; Celik et al. 2008, Fs = 1.25, fixed-point for p
def run_convergence(spec_path: str | Path, scales: tuple[float, float, float] = (1.0, 1.3, 1.69)) -> ConvergenceResult
    # quantities: fea -> "compliance","max_disp"; cfd -> "u_max","inflow". Gate `gci`: warn if GCI_fine(compliance|inflow) > 5%, fail if divergent.
# check.py
@dataclass class CheckReport: spec_path: str; status: str; gates: list[Gate]; mesh: MeshResult; quality: dict | None; solve: dict | None; convergence: dict | None; renders: list[RenderResult]; created_at: str
def run_check(spec_path: str | Path, solve: bool = True, converge: bool = False, render: bool = True) -> CheckReport
    # mesh via run_mesh_subprocess -> assess_mesh -> solve_fea|solve_potential_flow (skip gate `solve` with status "skip" if no solve section) -> optional convergence -> renders (groups, quality, result field) -> writes out/report.json, out/REPORT.md; updates state
def report_markdown(r: CheckReport) -> str     # gate table first (status, id, value, threshold, message, hint), then numbers, then render paths
def state_path() -> Path                       # $FEAGENT_STATE or ./.feagent/state.json
def record_state(spec_path: str, status: str, waiver: str | None = None) -> None   # {"specs": {abs_path: {"status", "checked_at", "spec_mtime", "waiver"}}}
```
CLI (`feagent`, argparse, every command supports `--json`; exit 0 = ok, 1 = gate fail/mesh fail, 2 = usage/spec/query error with message + hint on stderr):
```
feagent parts OUTDIR
feagent inspect PATH [--size H] [--heal] [--detail summary|full]         # PATH = .step/.stp/.igs/.iges/.brep or mesh.yaml (domain-built geometry)
feagent faces PATH --query JSON [--expect N]                             # describe_faces; explain_miss when count != expect
feagent render PATH --view faces|groups|quality|result [--highlight 1,2] [--query JSON] [-o PNG]
feagent mesh SPEC [--scale S]                                            # records state "unchecked"
feagent check SPEC [--no-solve] [--converge] [--no-render]
feagent converge SPEC
feagent waive SPEC --reason TEXT
feagent hook-stop                                                        # Task 10 implements hook.main(); CLI delegates
```

- [ ] **Step 1: Failing tests** `tests/test_converge.py`, `tests/test_check_cli.py`:
  - `test_gci3_quadratic`: f = 1 + h² sampled at h = 1, 1.3, 1.69 with n_i = h⁻³·1000 → `p` ≈ 2 ± 0.05, `extrapolated` ≈ 1 ± 1e-3, behavior monotonic.
  - `test_gci3_oscillatory`: f = (1.0, 1.1, 0.95) → behavior "oscillatory".
  - `test_check_cantilever_passes`: spec cantilever (fixed `{at: xmin}`, load `{at: xmax}` force (0,0,−100), size 4, order 2) → `status == "pass"`, REPORT.md exists and its first table contains `equilibrium`.
  - `test_check_records_state` (FEAGENT_STATE → tmp): state has the spec with status "pass".
  - `test_cli_inspect_json`: `feagent inspect plate_hole.step --json` (subprocess, `uv run`) → valid JSON with one hole feature; stderr may contain gmsh noise, stdout parses.
  - `test_cli_faces_expect_miss_exit2`: `--query '{"at":"xmin","kind":"cylinder"}' --expect 1` on cantilever → exit 2, stderr contains "0 matched".
- [ ] **Step 2:** FAIL. **Step 3:** implement. **Step 4:** `uv run pytest -q` (whole suite) → PASS.

---

### Task 10: Agent integration — MCP server, Stop hook, skill, examples

**Files:** `src/feagent/mcp_server.py`, `src/feagent/hook.py`, `.mcp.json`, `.claude/settings.json`, `.claude/skills/fe-mesh/SKILL.md`, `examples/{cantilever,bracket,plate_hole,thin_cup,tee,bullet}/mesh.yaml`, `examples/parts/*.step` (generated by `feagent parts examples/parts`), `README.md` (append "Meshing harness" section), `CLAUDE.md` (Status + answer "Which MCP tools does the agent get"), tests `tests/test_integration.py`.

**Interfaces:**
- Consumes: CLI `--json` contract (Task 9), `check.state_path/record_state`.
- Produces:
```python
# mcp_server.py — FastMCP("feagent"); every tool runs `sys.executable -m feagent.cli ... --json` with timeout and returns text (+ Image for renders)
TOOL_DOCS: dict[str, str]   # tool name -> description, kept in one place so an improver can edit tool descriptions
@mcp.tool() inspect_geometry(path: str, size: float | None = None, detail: str = "summary") -> str
@mcp.tool() find_faces(path: str, query: dict, expect: int | None = None) -> str
@mcp.tool() render_view(path: str, view: str = "faces", highlight: list[int] | None = None, query: dict | None = None) -> list   # [caption text, Image]
@mcp.tool() mesh(spec_path: str, scale: float = 1.0) -> str
@mcp.tool() check_mesh(spec_path: str, solve: bool = True, converge: bool = False) -> str      # returns REPORT.md text
@mcp.tool() convergence_study(spec_path: str) -> str
def main() -> None   # mcp.run() stdio ; [project.scripts] feagent-mcp = "feagent.mcp_server:main"
# hook.py
def main(stdin_text: str | None = None) -> int
    # Claude Code Stop hook. Blocks (prints reason to stderr, returns 2) if any tracked spec has status "fail" or "unchecked",
    # or spec_mtime > checked_at, and no waiver. Allows (returns 0) when no state file, all pass/waived, or when it has
    # already blocked 3 consecutive times for the same set (counter stored in state as "hook_blocks").
```
`.mcp.json`: server `feagent` → `uv run feagent-mcp`. `.claude/settings.json`: Stop hook `uv run feagent hook-stop`. Skill `fe-mesh` (frontmatter name/description) — the workflow: inspect digest + faces render → decide physics groups by query and confirm with `faces --expect` + highlight render → write `mesh.yaml` → `check` → read failing gates and fix spec → `check --converge` → report gates & numbers to the user; never pick faces by remembered tag, never claim done with failing gates (waive only with a stated reason). Include a gate→fix table and threshold table. Examples: one passing spec per part.

- [ ] **Step 1: Failing tests** `tests/test_integration.py`:
  - `test_hook_blocks_on_fail`: state with one failing spec → `hook.main("{}") == 2`; after `record_state(..., waiver="x")` → 0; three consecutive blocks then 0.
  - `test_examples_all_pass`: every `examples/*/mesh.yaml` → `run_check(..., converge=False, render=False).status == "pass"`.
  - `test_mcp_lists_tools`: import `mcp_server`, the registered tool names == {inspect_geometry, find_faces, render_view, mesh, check_mesh, convergence_study}.
- [ ] **Step 2:** FAIL. **Step 3:** implement. **Step 4:** `uv run pytest -q` → PASS; manual: `uv run feagent check examples/bracket/mesh.yaml --converge` prints a passing report.
