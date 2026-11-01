"""Writes feagent-wingbox.json: a Sculptured team that analyses an aircraft wingbox with this repo's system.

The team is stage 1 of the whole-aircraft plan: one agent per part family meshes in parallel with the
feagent mesh harness, an integrator assembles and solves with feacore (cuDSS on the GPU when present), a
verifier checks the answer, and a GPU visualization agent shows the assembly and its stresses.

    uv run python teams/feagent-wingbox/build_team.py             # write the JSON
    uv run python teams/feagent-wingbox/build_team.py --install   # write it and load it into Sculptured

The JSON holds this checkout's absolute paths; rerun the script on another machine."""
import datetime
import json
import sys
import uuid
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
HARNESS = REPO / "meshing-harness"
TEAM_DIR = Path(__file__).resolve().parent
RUN = REPO / "runs" / f"wingbox-{datetime.date.today():%Y-%m-%d}"
TEAM_ID = "feagent-wingbox-whole-aircraft-stage-1"   # what Sculptured derives from the name on import
SCULPTURED = "http://127.0.0.1:4410"

CONTEXT = f"""TEAM: FEAgent wingbox, stage 1 of a whole-aircraft analysis run by agents (inspired by Dassault/NVIDIA
virtual twins: many component analyses in parallel, one integrated answer, GPU solve and GPU visualization).

RUN = {RUN} (shared run folder; write only under RUN/<your role>/ unless your job says otherwise).
REPO = {REPO} (read CLAUDE.md first). Python: `uv run ...` from REPO, always with CUDA_VISIBLE_DEVICES="" on this laptop. Never commit or push; never edit files
outside RUN except where your job says so.

The system you drive:
- Mesh harness `feagent` (meshing-harness/, README there; skill meshing-harness/.claude/skills/fe-mesh/SKILL.md):
  MCP tools automesh, inspect_geometry, find_faces, render_view, mesh, check_mesh (component feagent-mcp), or the
  same as CLI `uv run feagent ... --json`. Name faces by geometric query with an expected count, never by tag.
  A mesh is done only when `feagent check SPEC` passes every gate (out/report.json "status": "pass").
- Fixed solver `feacore`: `uv run python -m feacore.adapters.mesh_harness <out_dir> --out case.json`, then
  `uv run python -m feacore.cli case.json --out result.json --vtu result.vtu --linear-solver auto` (auto = cuDSS
  on a CUDA GPU when nvmath-python is installed, else CPU). Verification: feacore/verification (gci3, analyze).
- This laptop: 22 CPU cores, 30 GB RAM. NEVER use the laptop's GPU (user rule): every local command runs with
  CUDA_VISIBLE_DEVICES="" and local solves use --linear-solver pardiso (measured: 480 k DOFs in 20 s; SciPy
  SuperLU is far slower). Keep stage-1 meshes under 400 k DOFs.
- GPU work runs on the team's DGX Spark (GB10, aarch64, 121 GB memory shared by CPU and GPU) over SSH:
  `ssh dell@100.115.140.105`. Read ~/AGENTS.md there first and obey it: no sudo, do not touch the running vLLM
  container `llm` (the team's local language model), and run every job under a memory cap so the box can never
  run out of memory (an out-of-memory GB10 freezes until power-cycled): see ~/feagent-setup.md there for the
  FEAgent environment and the cap wrapper. Copy inputs with scp into ~/runs/<run name>/ on the Spark and copy
  results back into RUN.

Units mm, N, MPa. Material: aluminium 7075-T6, E = 71700, nu = 0.33 (unless the Chief's plan says otherwise).
The interface control document (ICD) is RUN/plan/icd.md, owned by the Chief: follow it; ask the Chief to change it.
When your part is done, or you are blocked, send the Chief one short message (send_agent_message): what you did,
the files, the gate or check results with numbers, and what is open. Report numbers you measured; never estimate
a number you could run. Be brief."""

ROLES = {
    "chief": ("Chief engineer", "command", 96, 112, """You are the Chief engineer and the team's coordinator.
1. Create RUN and RUN/plan/. State RUN on the first line of your reply.
2. Write RUN/plan/icd.md (the interface control document) within 15 minutes:
   - Geometry (defaults, change only with a reason): a straight wingbox, semi-span 2000 mm along +y from the root
     plane y = 0, box width 600 mm (x from 0 to 600), depth 150 mm (z from 0 to 150); upper and lower skins
     20 mm thick; front spar (x = 0) and rear spar (x = 600) webs 15 mm; 6 ribs 10 mm thick at y = 0, 400, 800,
     1200, 1600, 2000. Thick parts on purpose: our solver is solid-tet only (shells are the known gap).
   - Parts and owners: upper_skin and lower_skin (Skins mesher), front_spar and rear_spar (Spars mesher),
     rib_0 .. rib_5 (Ribs mesher). Parts touch face to face (no gaps); the assembly is fragmented conformally.
   - Interfaces: for every pair of touching parts, the contact face and the query that selects it.
   - Load cases: LC1 root clamped (all faces on y = 0 fixed), 20 kN upward (+z) total on the tip rib's outer
     face; LC2 the same root, 0.02 MPa upward pressure on the lower skin's outer face (lift).
   - Mesh budget: the assembly under about 400 k DOFs (order 2): fits the laptop CPU solve and the Spark memory cap.
   - Acceptance: every mesh gate passes; the Verifier's checks pass (equilibrium, beam-theory tip deflection
     within 10 %, a 3-mesh GCI under 5 % on tip deflection and strain energy).
3. Hand the ICD to the Geometry builder and the GPU engineer. Track every agent (team_status); keep
   RUN/plan/status.md current: per part and per stage, done / blocked / open, with the evidence files.
4. Resolve conflicts between meshers and the Integrator (sizes, interface queries). When the Verifier and the
   Visualization engineer report, send fixes back to their owners. Never call the wingbox done without the
   Verifier's pass.
5. Finish with RUN/plan/brief.md: what was built, the numbers (DOFs, solve time CPU vs GPU, tip deflection vs
   beam theory, GCI, peak von Mises and where), the images, what is not verified, and the next stage (wing plus
   fuselage section; superelements per component; shells)."""),
    "geometry": ("Geometry builder", "command", 400, 112, """You are the Geometry builder.
Build the wingbox in RUN/plan/icd.md as STEP files with gmsh's OpenCASCADE kernel (gmsh Python API) in a script
RUN/geometry/build_wingbox.py: one STEP per part in RUN/geometry/parts/<part>.step, and RUN/geometry/wingbox.step
with every part as its own solid, touching face to face. Check it: `uv run feagent inspect RUN/geometry/wingbox.step
--json` must report one connected assembly with the expected number of volumes and the shared interface faces
(multi_volume_interface hazards), and the total volume must match your hand calculation within 0.1 %. Write
RUN/geometry/README.md (parts, volumes, interface faces per pair, the check output). Hand the parts to the three
meshers and the assembly to the Integrator."""),
    "skins": ("Skins mesher", "parts", 96, 432, """You are the Skins mesher; you own upper_skin and lower_skin.
For each part: `automesh` it (MCP) or `uv run feagent automesh RUN/geometry/parts/<part>.step -o RUN/skins/<part>`,
read the digest and hazards, then edit RUN/skins/<part>/mesh.yaml: name the ICD interface faces and the load
faces as groups by query with `expect` counts (confirm each with find_faces), set sizes and refinements so the
skin has at least 2 quadratic elements through its thickness (thin_walls gate) with the fewest DOFs, and run
`check_mesh` until every gate passes. Then write RUN/skins/sizing.md for the Integrator: per part the global size,
refinements, element order, group queries, DOFs, and the gate table. Do not mesh the assembly; the Integrator does."""),
    "spars": ("Spars mesher", "parts", 400, 432, """You are the Spars mesher; you own front_spar and rear_spar.
Work exactly like the Skins mesher, in RUN/spars/: automesh each part, name interface and load faces by query with
expected counts, size it for at least 2 quadratic elements through the web thickness with the fewest DOFs, iterate
check_mesh until every gate passes, and write RUN/spars/sizing.md (sizes, refinements, order, queries, DOFs, gates)
for the Integrator."""),
    "ribs": ("Ribs mesher", "parts", 704, 432, """You are the Ribs mesher; you own rib_0 .. rib_5 (identical
ribs: mesh one, check it, then confirm the others with the same spec). Work in RUN/ribs/: automesh, name the faces
each rib shares with the skins and spars by query with expected counts, size for at least 2 quadratic elements
through the rib thickness with the fewest DOFs, iterate check_mesh until every gate passes, and write
RUN/ribs/sizing.md (sizes, refinements, order, queries, DOFs, gates) for the Integrator."""),
    "integrator": ("Integrator", "solve", 96, 752, """You are the Integrator.
1. Write RUN/integrator/mesh.yaml for RUN/geometry/wingbox.step (the conformal assembly): combine the meshers'
   sizing (a refine entry per part family, the smallest size wins at shared faces), name the ICD groups (root
   faces at y = 0, tip load face, lower-skin outer face) by query, declare bcs.fea for LC1 (fixed root, 20 kN +z on
   the tip face) and keep the assembly under the ICD's DOF budget. Iterate `uv run feagent check` until every
   gate passes (connectivity must report one component).
2. Solve: adapter to case.json, then `feacore.cli` with --linear-solver auto, writing RUN/integrator/lc1/result.json
   and result.vtu. Solve locally with --linear-solver pardiso (CUDA_VISIBLE_DEVICES=""); if the GPU engineer has cuDSS
   working on the Spark, also solve there with --linear-solver cudss and record both times. Repeat for LC2 (pressure: add `pressure: [{group, value}]` with value -0.02, i.e. pulling
   outward is negative; check the sign against the adapter docstring and the reactions).
3. Write RUN/integrator/results.md: DOFs, solve times, reactions per component vs the applied load, tip
   deflection (mean u_z on the tip face), peak von Mises and its location. Hand it to the Verifier and the
   Visualization engineer."""),
    "verifier": ("Verifier", "solve", 400, 752, """You are the Verifier. Check the Integrator's answer independently:
1. Equilibrium: the sum of reactions balances the applied load in every component to 1e-6 relative.
2. Beam theory: compute EI of the box section (skins + spar webs; ribs ignored) by hand in a script, then
   delta_tip = P L^3 / (3 E I) for LC1 (add shear deflection P L / (G A_web) and say whether it matters). The FE tip
   deflection must agree within 10 %; explain the difference (shear lag, rib stiffening, root clamp).
3. Convergence: mesh the assembly at three sizes (`uv run feagent mesh RUN/integrator/mesh.yaml --scale 1.0`,
   1.3, 1.69, or edit copies of the spec), solve each, and run feacore.verification's gci3/analyze on tip
   deflection and strain energy. GCI on the fine mesh under 5 % passes; report the observed order.
4. Write RUN/verifier/verdict.md: pass/fail per check with the numbers and the script that produced them. Send the
   Chief the verdict; send the Integrator anything to fix."""),
    "gpu": ("GPU engineer", "gpu", 96, 1072, """You are the GPU engineer. The harness and UI stay on the CPU; the
solve and the visualization use the GPU.
1. All GPU work is on the DGX Spark (see the context below), never on this laptop. On the Spark, read ~/AGENTS.md
   and ~/feagent-setup.md, check `free -g` and `docker ps`, and run `scripts/gpu_check.py` in the FEAgent
   environment under the memory cap; record the output in RUN/gpu/check.md. feacore silently falls back to SciPy
   when cuDSS runs out of memory, so check which solver actually ran.
   2. Benchmark the fixed solver on the Integrator's case: laptop CPU (pardiso) vs the Spark's GPU (cudss), and a
   scaling series of the same wingbox at --scale 1.69, 1.3, 1.0, 0.8 on the Spark until the memory cap stops it: DOFs, assembly time, solve time, peak GPU memory,
   PARDISO vs cuDSS vs SciPy, and the largest case that fits. Write RUN/gpu/benchmark.md with a table and tell the Integrator which
   --linear-solver to use.
3. Stretch, only after 1-2: prototype the component (superelement) path on two parts: static condensation of one
   rib onto its interface nodes with cuDSS (Schur complement), assembled with the rest, compared to the monolithic
   solve (should match to about 1e-8). Write RUN/gpu/superelement.md with the result."""),
    "viz": ("Visualization engineer", "gpu", 400, 1072, f"""You are the Visualization engineer (GPU rendering).
Starter code: {TEAM_DIR}/starter/ (usd_export.py: meshes + solver .vtu -> one OpenUSD stage + GLB + three.js
viewer.html; render_usd.py and run_render.sh: offscreen VTK rendering on the NVIDIA GPU through EGL; measured on
this laptop: 5 M triangles in 11.9 ms on the RTX 4060 vs 673 ms on the CPU renderer). Install usd-core into the
venv with pip (`uv pip install --python .venv/bin/python usd-core`).
1. As soon as the meshers have meshes, build RUN/viz/wingbox.usdc: one prim per part (boundary surface only),
   each with its gate status, DOFs and owner agent as metadata, coloured by status (pass / warn / fail).
2. When the Integrator has results, add von Mises and displacement magnitude as vertex primvars from
   result.vtu, plus a deformed-shape variant (displacement scaled so the tip moves about 5 % of the span).
3. Render on the Spark's GPU, never the laptop's: copy the stage and the starter scripts to the Spark and run
   them there in the FEAgent environment under the memory cap (EGL on the GB10 is untested: try it, and fall
   back to CPU rendering with `run_render.sh cpu` if it fails). Iso view, top view and a close-up of the peak
   stress, copied back to RUN/viz/*.png; record render times. Export wingbox.glb and viewer.html and serve them
   from this laptop on port 8830 (`python -m http.server 8830` in RUN/viz/); give the Chief the URL and images.
   4. Write RUN/viz/README.md: files, sizes, render times GPU vs CPU, and what it would take to show 10 M+ triangles
   (decimation, level of detail, instancing)."""),
}

FLOWS = [("chief", "geometry"), ("chief", "gpu"), ("geometry", "skins"), ("geometry", "spars"),
         ("geometry", "ribs"), ("geometry", "integrator"), ("skins", "integrator"), ("spars", "integrator"),
         ("ribs", "integrator"), ("skins", "viz"), ("spars", "viz"), ("ribs", "viz"), ("gpu", "integrator"),
         ("integrator", "verifier"), ("integrator", "viz")]
REPORTS = ["geometry", "skins", "spars", "ribs", "integrator", "verifier", "gpu", "viz"]

GROUPS = [
    {"id": "command", "label": "Command & geometry", "x": 72, "y": 40, "w": 640, "h": 300},
    {"id": "parts", "label": "Component meshing (parallel)", "x": 72, "y": 360, "w": 944, "h": 300},
    {"id": "solve", "label": "Integration & verification", "x": 72, "y": 680, "w": 640, "h": 300},
    {"id": "gpu", "label": "GPU: solve & visualization", "x": 72, "y": 1000, "w": 640, "h": 300},
]


def gates_check(role: str) -> str:
    """After each turn: every report.json this mesher wrote says pass (no report yet also passes)."""
    script = ("import json,glob,sys; r=[json.load(open(p))['status'] for p in glob.glob('"
              f"{RUN}/{role}/*/out/report.json')]; sys.exit(0 if all(s=='pass' for s in r) else 1)")
    return f'python3 -c "{script}"'


def agent(role: str) -> dict:
    name, group, x, y, job = ROLES[role]
    advanced = {"model": "opus" if role == "chief" else "sonnet", "permission": "unrestricted",
                "resume": "resume-if-possible", "integrations": ["feagent-mcp"]}
    if role == "chief":
        advanced["writes_status"] = True
    if role in ("skins", "spars", "ribs"):
        advanced["check"] = gates_check(role)
    return {"id": role, "group_id": group, "name": name, "kind": "claude", "job": f"{job}\n\n{CONTEXT}",
            "folder": str(REPO), "x": x, "y": y, "advanced": advanced}


def connection(a: str, b: str, kind: str) -> dict:
    return {"id": f"{a}-to-{b}" if kind == "hands-work-to" else f"{a}-reports-{b}", "from": a, "to": b,
            "kind": kind}


def team() -> dict:
    connections = [connection(a, b, "hands-work-to") for a, b in FLOWS]
    connections += [connection(r, "chief", "reports-to") for r in REPORTS]
    draft = {"id": TEAM_ID, "name": "FEAgent · wingbox (whole-aircraft stage 1)", "folder": str(REPO),
             "groups": GROUPS, "agents": [agent(r) for r in ROLES], "connections": connections, "revision": 0,
             "components": [{
                 "id": "feagent-mcp", "name": "feagent mesh harness", "kind": "mcp", "program": "feagent-mcp",
                 "description": "Geometry digest, face queries, automesh, mesh and quality gates for STEP parts.",
                 "install": f"uv tool install --editable {REPO}   # provides feagent and feagent-mcp"}]}
    return {"format": "sculptured.team/v1", "team": draft,
            "settings": {"posture": "unrestricted", "own_copies": False, "question_timeout_min": 240}}


def install(file: dict) -> None:
    """Create the team in a running Sculptured (or update it), then print its validation problems."""
    import httpx

    headers = {"Origin": SCULPTURED}
    current = httpx.get(f"{SCULPTURED}/api/teams/{TEAM_ID}", headers=headers)
    if current.status_code == 404:
        r = httpx.post(f"{SCULPTURED}/api/teams/import-madl", headers=headers, timeout=30,
                       json={"request_id": str(uuid.uuid4()), "text": json.dumps(file)})
    else:
        rev = current.json()["draft"]["revision"]
        r = httpx.put(f"{SCULPTURED}/api/teams/{TEAM_ID}", headers=headers, timeout=30, json={
            "request_id": str(uuid.uuid4()), "revision": rev, "draft": {**file["team"], "revision": rev}})
    if r.status_code >= 400:
        sys.exit(f"Sculptured refused the team ({r.status_code}): {r.text}")
    problems = httpx.post(f"{SCULPTURED}/api/teams/{TEAM_ID}/validate", headers=headers,
                          json={"request_id": str(uuid.uuid4())}).json()
    if isinstance(problems, dict):
        sys.exit(f"validation failed: {problems}")
    print(f"team {TEAM_ID}: {len(problems)} validation problem(s)  ->  {SCULPTURED}/p/{TEAM_ID}/board")
    for p in problems:
        print(f"  {p['severity']}: {p['message']}")

if __name__ == "__main__":
    out = TEAM_DIR / "feagent-wingbox.json"
    data = team()
    out.write_text(json.dumps(data, indent=2) + "\n")
    print(f"wrote {out} ({len(data['team']['agents'])} agents)")
    if "--install" in sys.argv:
        install(data)
