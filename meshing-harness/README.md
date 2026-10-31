# feagent — meshing harness

Lets an LLM agent mesh **existing CAD** (STEP/IGES/BREP) for FEA or CFD the way a coding agent edits
code: it reads a kernel-computed digest of the part, names boundary-condition faces by geometric
query (never by remembered tag), edits a YAML mesh spec, and iterates until a suite of mesh quality
gates passes. Solving is out of scope; a separate solver consumes the exports and `report.json`.

## Quick start

The package is shipped by the repo's single root `pyproject.toml` (together with `feacore` and
`feabench`); `uv run` works from the repo root or from this directory.

```bash
uv sync                                             # from the repo root
uv run feagent parts /tmp/parts                     # 7 synthetic test parts
uv run feagent automesh /tmp/parts/bracket.step     # zero-config mesh + gates -> editable mesh.yaml
uv run feagent inspect /tmp/parts/bracket.step      # geometry digest: features, planar faces, hazards
uv run feagent faces /tmp/parts/bracket.step --query '{"feature": "hole", "radius": 3.25}' --expect 4
uv run feagent check /tmp/parts/bracket_mesh/mesh.yaml   # mesh + gates -> out/report.json, REPORT.md
uv run pytest -q meshing-harness/tests             # harness tests (root `pytest` runs everything)
uv run python -m feacore.adapters.mesh_harness /tmp/parts/bracket_mesh/out --out case.json  # → solver
```

Every command takes `--json` (exactly one JSON document on stdout). Exit codes: 0 ok, 1 gate or
mesh failure, 2 usage/spec/query error (message + hint on stderr), 3 internal error.

## For agents

- **MCP server:** `uv run feagent-mcp` (configured in `.mcp.json`). Tools: `automesh`,
  `inspect_geometry`, `find_faces`, `render_view`, `mesh`, `check_mesh`; prompt `meshing_workflow`.
- **Claude Code:** launch it from this directory to get the `fe-mesh` skill and a Stop hook that
  blocks finishing while a spec of this session is failing or unchecked (`feagent waive SPEC --reason ...`
  to accept a known failure).
- **Improver-editable context:** `agent_context/` holds `SKILL.md` (workflow), `tools.yaml` (MCP
  tool descriptions) and `examples/` (passing specs). Point a server at another copy with
  `FEAGENT_CONTEXT_DIR=/path/to/agent_context`.
- **Evaluator input:** `out/report.json` per spec, schema in `docs/report-schema.md`.

## Layout

| Path | What |
|---|---|
| `src/feagent/geometry.py` | STEP → faces, normals, holes/fillets, wall thickness, meshing hazards, text digest |
| `src/feagent/select.py` | geometric face query language + near-miss explanations |
| `src/feagent/spec.py` | YAML mesh spec (`groups`, `size`, `refine`, `element`, `bcs`, `quality`, ...) |
| `src/feagent/mesher.py` | spec → gmsh mesh, physical groups, exports (`.msh`, `.inp`, `.su2`, ...), `auto_spec` |
| `src/feagent/quality.py` | mesh gates: inverted, SICN, gamma, volume/area fidelity, connectivity, thin walls, CFD |
| `src/feagent/check.py`, `cli.py` | gate pipeline, `report.json`, state for the Stop hook, CLI |
| `src/feagent/mcp_server.py`, `hook.py` | MCP server, Claude Code Stop hook |
| `src/feagent/{vec,gates,gmsh_util,graph}.py` | shared primitives |
| `docs/` | design, implementation plan, report schema |
| `reports/`, `research_notes/` | literature review behind the design |

Code style is enforced by `tests/test_style.py`: every function ≤ 25 lines and ≤ 25 statements,
lines ≤ 110 characters.
