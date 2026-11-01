"""Writes feagent-multisystem.json: a Sculptured team that shows how FEAgent works on one study.

Many agents in parallel, each running a slightly different mesh system on the same contract, each writing its own
analysis; a verifier gates their evidence; a synthesis agent builds the consensus and the final brief. It mirrors the
mesh-system study of 2026-11-01 (experiments/mesh_systems/ and experiments/physical/ on branch nonlinear-setup).

    uv run --no-sync python teams/feagent-multisystem/build_team.py             # write the JSON
    uv run --no-sync python teams/feagent-multisystem/build_team.py --install   # write it and load it into Sculptured
        [--worktree PATH]   checkout of branch nonlinear-setup the agents work in (default: ../FEAgent-setup)
        [--photo PATH]      photo of the broken print, copied to RUN/inputs/abc_00000065_real_break.png

The Spark's ssh target comes from FEAGENT_SPARK (e.g. user@host); without it the prompts say <spark-host>.
The JSON holds this machine's absolute paths, so it is generated (git-ignored): rerun the script on another machine."""
import datetime
import json
import os
import shutil
import sys
import uuid
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TEAM_DIR = Path(__file__).resolve().parent
TEAM_ID = "feagent-parallel-mesh-systems-synthesis"   # what Sculptured derives from the name on import
TEAM_NAME = "FEAgent · parallel mesh systems → synthesis"
SCULPTURED = "http://127.0.0.1:4410"
SPARK = os.environ.get("FEAGENT_SPARK", "<spark-host>")


def _arg(flag: str, default):
    return Path(sys.argv[sys.argv.index(flag) + 1]).resolve() if flag in sys.argv else default


WORK = _arg("--worktree", REPO.parent / "FEAgent-setup")
RUN = WORK / "runs" / f"multisystem-{datetime.date.today():%Y-%m-%d}"
P65 = "benchmarks/gold/abc_00000065_bc0/public/geometry.step"
P03 = "benchmarks/gold/abc_00000003_bc0/public/geometry.step"
CPU_GL = 'LIBGL_ALWAYS_SOFTWARE=1 GALLIUM_DRIVER=llvmpipe __GLX_VENDOR_LIBRARY_NAME=mesa'

CONTEXT = f"""TEAM: FEAgent, parallel mesh systems -> synthesis. Seven agents each run a slightly different mesh system on
the same contract (same parts, material, loads, node budget, crush recipe); each writes its own analysis; a Verifier
gates the evidence; the Synthesis agent builds the consensus and compares it with the physical test.

WORK = {WORK} (checkout of branch nonlinear-setup; read CLAUDE.md, experiments/mesh_systems/CONTRACT.md and
experiments/physical/CRUSH_RUNS.md first). Starter scripts and reference figures: {TEAM_DIR}/starter/.
RUN = {RUN} (shared run folder, git-ignored; write only under RUN/<your folder> unless your job says otherwise).
The contract for this run is RUN/plan/contract.md, owned by the Study lead: follow it; ask the lead to change it.

Hard rules (user rules; no exceptions):
- Never use this laptop's GPU. Every command runs with CUDA_VISIBLE_DEVICES="" and every render on the CPU:
  {CPU_GL}.
- GPU work (this team needs none) goes to the DGX Spark only: `ssh {SPARK}`. Read ~/AGENTS.md there first and obey
  it: no sudo, one large GPU job at a time, every job under the memory cap in ~/feagent-setup.md (an out-of-memory
  GB10 freezes until power-cycled), never touch the `ridgeline-isaac` container, no persistence (no systemd units or
  timers, no cron). CPU crush runs may go to the Spark too, under the same cap, when the lead assigns them.
- Python: `uv run --no-sync python ...` from WORK. Never `uv sync` or `uv pip install` into the shared venv
  (Netgen lives in its own --target dir on PYTHONPATH; see experiments/mesh_systems/netgen/README.md).
- New run-id dirs only: never overwrite or delete a run output; a continuation is `crush.py --resume` into a new dir.
- Never commit or push. In anything meant for git: no absolute /home or /tmp paths (paths relative to WORK), and the
  Spark is written `<spark-host>`, never its address.

Units: STEP and meshes in metres; reports in mm, N, MPa. Material: experiments/material/alpha_fast_dogbone1.json
(Formlabs Alpha Fast Resin, from the user's dogbone): E 1879.8 MPa, nu 0.35 (assumed), UTS 50.48 MPa, failure PEEQ
0.043. When your part is done, or you are blocked, send the Study lead one short message (send_agent_message): what
you did, the files, the numbers. Report numbers you measured; never estimate a number you could run. Be brief."""

# id: (label, what makes it different, its size knob for 40k nodes (match_nodes.py), expected numbers from the study)
SYSTEMS = {
    "harness_auto": ("harness auto", "zero-config `feagent automesh` spec plus the contract groups by query; nothing tuned"
                     " (the baseline: what an agent gets from automesh once it has named its faces).",
                     "--global 3.05 mm -> 40,078 nodes; neck median edge 2.83 mm, 2.9 cells across the 8.1 mm wall;"
                     " thin_walls warn", "linear p65 97,385 nodes, compliance 36.989 um @100 N, robust vM 44.1 MPa @F_full;"
                     " crush: peak 5.825 kN, crack hf 0.619 (neck)"),
    "harness_expert": ("harness expert", "the SKILL.md workflow frozen: geometry digest + hazards -> local refinement at"
                       " platens, fillets and holes, thin-wall layers, curvature sizing, Netgen optimisation.",
                       "--global 12 mm (its default) -> 39,608 nodes; neck 3.19 mm, 2.5 across; every gate passes"
                       " (thin_walls lowered to 1 on purpose, see its README)", "linear p65 132,063 nodes, 36.981 um,"
                       " 44.2 MPa; crush: peak 5.815 kN, hf 0.616 (neck)"),
    "ace_sizing": ("ACE sizing", "the ACE loop's mesher: one global h plus Distance/Threshold refinements at groups and Ball"
                   " refinements at hotspots; coarse look -> refine hotspots -> read gates -> remesh.",
                   "--h 3.5 mm with --node-cap 40,800 (its loop settles at h 2.38 mm) -> 40,095 nodes; neck 2.88 mm,"
                   " 2.8 across; SICN fail (0.025), thin_walls warn", "linear p65 39,940 nodes, 36.719 um, 42.2 MPa;"
                   " crush: peak 5.808 kN, hf 0.619 (neck)"),
    "gold_nested": ("gold nested", "the gold generator's approach: one base mesh split uniformly (gmsh mesh.refine) into"
                    " nested levels; level L1 is the mesh. No local refinement at all.",
                    "--h0-frac 0.324 -> 39,234 nodes; neck 3.50 mm, 2.3 across (coarsest neck); SICN fail (0.016),"
                    " gamma warn", "linear p65 130,096 nodes, 36.768 um, 42.5 MPa; crush: peak 5.855 kN, hf 0.629 (neck)"),
    "netgen": ("Netgen", "an independent mesh generator: Netgen on the OCC geometry (maxh, grading 0.3, curvature"
               " safety 2), groups found geometrically; no gmsh in the meshing.",
               "--maxh 2.10 mm -> 39,706 nodes; neck 2.78 mm, 2.9 across; thin_walls warn",
               "linear p65 59,743 nodes, 36.707 um, 43.7 MPa (build 9 s, the fastest); crush: peak 5.861 kN at 4.78 mm,"
               " hf 0.614 (neck)"),
    "harness_hex": ("harness hex", "hexahedra where they work: tet10 for the linear solve (skfem has no hexes), an all-hex"
                    " C3D8 mesh (tet-to-hex subdivision, base size from its own validity search) for the crush.",
                    "--thin-factor 12.9 (h 4 mm by its own search) -> 54,408 nodes / 42,940 C3D8: NOT matched (+36 %)."
                    " No hex mesh passing every gate exists below 54.4k; label it unmatched, do not force 40k",
                    "linear p65 129,039 nodes, 36.990 um, 44.3 MPa; crush: peak 5.995 kN at 4.38 mm (the outlier),"
                    " first failure already at 2.14 mm, hf 0.602 (neck)"),
    "harness_neck": ("harness neck", "harness_expert plus one refinement found from the digest (find_necks): the bore and"
                     " the coaxial ring faces (the arms' 8.1 mm neck wall), refined to --neck-size.",
                     "--neck-size 3.83 mm -> 40,538 nodes; neck 3.13 mm, 2.6 across (at 40k the neck refinement costs"
                     " what the expert already spends); every gate passes", "crush: peak 5.787 kN at 4.69 mm, hf 0.615"
                     " (neck). Reference fine-neck run: 58,020 nodes, 1.6 mm neck, peak 5.745 kN at 4.53 mm"),
}


def system_job(sid: str) -> str:
    label, how, knob, expect = SYSTEMS[sid]
    out = f"RUN/systems/{sid}"
    pypath = "PYTHONPATH=<netgen --target dir> " if sid == "netgen" else ""
    return f"""You are the {label} mesh agent; you run mesh system `{sid}` (experiments/mesh_systems/{sid}/: read its
README.md and build.py first). What makes it different: {how}
Work only in {out}/. Wait for RUN/plan/contract.md (the lead's message), then:
1. Meshes. Linear, both parts: `{pypath}CUDA_VISIBLE_DEVICES="" uv run --no-sync python experiments/mesh_systems/{sid}/build.py
   {P65} y {out}/p65_linear --purpose linear` (and {P03} z {out}/p03_linear). Nonlinear p65 at the
   contract's node budget: `uv run --no-sync python experiments/mesh_systems/match_nodes.py {sid} {out}/mesh --part abc_00000065
   --target 40000` (tunes only this system's own knob; the mesh lands in {out}/mesh/{sid}/p65_nonlinear_n40k).
   Expected: {knob}.
2. Gates: read each report.json (feagent.quality.assess_mesh, same thresholds for every system); list fail / warn
   with value and threshold. A gate this system fails by design stays failed: report it, do not hide it.
3. Linear compression (feacore/skfem, CPU, PARDISO): `experiments/physical/linear_ramp.py --system {sid} --part <part>
   --mesh-dir <linear mesh> --out {out}/linear/<part>`. Check the reaction equals 100 N.
4. Crush, the neck recipe on p65: `experiments/physical/imperfection.py --part abc_00000065 --mesh-dir <n40k mesh>
   --out {out}/mesh_imp --platen sliding --amp-mm 0.1`, then `experiments/physical/crush.py --system {sid} --part
   abc_00000065 --mesh-dir {out}/mesh_imp --out {out}/crush --threads <lead's slot> --platen bonded --failure
   peeq_tension --compression-factor 0 --compression-switch 0.5:5`, on the machine the lead assigns. Hours of wall time:
   check progress.log now and then; if it stops, resume into a new run dir.
5. Videos ({CPU_GL}): render_linear.py per part (F_full and magnification from the contract) ->
   {out}/videos/{sid}_<part>_linear.mp4; render_crush.py -> {out}/videos/{sid}_abc_00000065_crush.mp4;
   neck_closeup.py -> {out}/videos/{sid}_neck_closeup.mp4.
6. Analysis: {out}/analysis.md and analysis.json with keys system, nodes, cells, element, gates, compliance_um,
   robust_vm_MPa, reaction_N, peak_force_N, travel_at_peak_mm, first_failure_travel_mm, crack_site, crack_hf,
   refinement_map, caveats. crack_site and crack_hf from the failed cells (starter/neck_hf.py on the run dir);
   refinement_map = element density in neck / tip pocket / base / rest relative to the part average
   (starter/refine_maps.py {out}/refine {sid} {sid}=<n40k mesh.msh>). Caveats: what this system cannot do.
   Expected (the study's numbers; yours should land close): {expect}.
Hand the analysis to the Verifier."""


ROLES = {
    "lead": ("Study lead", "lead", 1008, 112, f"""You are the Study lead and the team's coordinator.
1. Create RUN, RUN/plan/ and RUN/inputs/. State RUN on the first line of your reply.
2. Within 15 minutes write RUN/plan/contract.md, from experiments/mesh_systems/CONTRACT.md and CRUSH_RUNS.md (neck recipe):
   - Parts: abc_00000065 (shaft clamp, axis y; bottom = 1 plane face at y = -0.04396 m, top = 2 plane faces at
     y = 0.03384 m) and abc_00000003 (closed tube, axis z; bottom 1 face at z = 0, top 1 face at z = 0.06125 m).
     Groups bottom, top (dim 2) and domain (dim 3), chosen by normal + position, never by tag.
   - Material: Alpha Fast Resin, E 1879.8 MPa, nu 0.35, UTS 50.48 MPa, failure PEEQ 0.043.
   - Linear: bottom fixed, 100 N along -axis on top spread per area; 50k-400k DOFs, tet10. F_full (robust von Mises
     = 87 % of UTS): 2394 N for p65 (video magnification x1), 4558 N for p03 (x3).
   - Nonlinear, p65 only: 40,000 nodes +- 2 % for every system (harness_hex excepted, labelled unmatched), C3D10 with
     straight interior edges (hex: C3D8). Crush recipe: 0.1 mm sliding-platen buckling imperfection, then bonded
     platens, tension-only PEEQ erosion, compression switch 0.5:5, step policy and adaptive release of crush.py.
   - Machines (one crush per slot): laptop 3 slots x 4 threads (netgen, harness_hex, harness_neck); Spark ({SPARK})
     4 slots x 5 threads under the memory cap (harness_auto, harness_expert, ace_sizing, gold_nested), results copied
     back with rsync into RUN/systems/<id>/crush/ (without ccx_work/). Spark code: the nonlinear-setup clone
     ~/feagent-setup-run (fast-forward it to this branch first); runs under ~/feagent-runs/multisystem/<id>/.
   - Deliverables per system: the system agent's step list; the Verifier's gate; the Synthesis brief.
3. Ask the user for the photo of the broken print if RUN/inputs/abc_00000065_real_break.png is missing.
4. Hand the contract to all seven system agents at once. Track them (team_status); keep RUN/plan/status.md current:
   per system mesh / gates / linear / crush / videos / analysis, done / running / blocked, with the evidence files.
5. Send the Verifier's rejections back to their owners. Never call the study done without the Synthesis brief."""),
    **{sid: (SYSTEMS[sid][0], "systems", 96 + i * 304, 432, system_job(sid)) for i, sid in enumerate(SYSTEMS)},
    "verifier": ("Verifier (gate)", "gate", 1008, 752, """You are the Verifier: the gate between the seven systems and the
synthesis. Check every system independently from its files, not from its analysis text, and write RUN/verify/verdict.json
({system: {verdict: pass|flag|fail, checks: {...}, note}}) and RUN/verify/verdict.md. Checks per system:
1. Equilibrium: linear reaction = 100 N applied to 1e-6 relative (linear/<part>/summary.json); crush: the reaction
   force in force_displacement.csv matches the platen force sign and is non-negative before the peak.
2. Gates: report.json present for every mesh; fails / warns listed with value and threshold. A known design failure
   (ace_sizing and gold_nested SICN) is a flag, not a fail; an undisclosed one is a fail.
3. Budget: n40k mesh within 40,000 +- 2 % nodes (harness_hex: flag "unmatched, 54k C3D8").
4. Crush: run.json has the recipe (bonded, peeq_tension, compression-factor 0, switch 0.5:5, imperfection mesh); the
   run reached its peak (last force < 0.9 x peak); peak and travel at peak recomputed from the CSV match analysis.json.
5. Analysis complete: every analysis.json key filled; videos exist and play (ffprobe duration >= 10 s for crush).
Send fixes to the owner; when all seven are pass or flag, hand the verdict to the Synthesis agent."""),
    "synthesis": ("Synthesis", "synth", 1008, 1072, f"""You are the Synthesis agent. Read the seven analyses and the Verifier's
verdict (use pass and flag systems; name every fail and leave it out). Model your output on
experiments/mesh_systems/RESULTS.md and experiments/physical/compare.py (tables), and starter/make_figs.py with the
figures in starter/reference/ (look and wording). Write everything under RUN/synthesis/.
1. consensus.json + consensus.md: per quantity the median, min, max, spread = (max - min) / median, n. Quantities:
   compliance, robust and peak von Mises, crush peak force, travel at peak, first-failure travel, crack hf, nodes,
   build / solve / crush wall time.
2. Trust: agreement = trustworthy, spread = mesh-sensitive. Expected from the study: compliance spread 0.77 % (p65),
   1.29 % (p03); robust von Mises 4.75 % (p65); peak point von Mises 41.6 % (singular at platen edges: mesh-dependent,
   never quote it alone); crush peak 5.79-6.00 kN, median 5.83, spread 3.6 % (tets only 1.3 %); crack hf 0.60-0.63,
   all seven at the arm necks.
3. Outliers and why: harness_hex (54k nodes, linear C3D8, 7.5 cells across the neck, first failure at 2.14 mm,
   peak +2.9 %); any other system outside the band, with the evidence.
4. Refinement maps: starter/refine_maps.py RUN/synthesis/refine matched_40k <id>=<n40k mesh.msh> ... for all seven
   (size maps, size vs consensus, budget per region).
5. Physical test: the printed clamp broke one arm at the neck (ring wall between bore and outer arc), hf about
   0.45-0.6 (RUN/inputs/abc_00000065_real_break.png). The first recipe (PEEQ erosion, any stress state) crushed the arm
   tips at hf 0.89 on all six systems at 3.2-3.9 kN; the neck recipe moved the crack to the neck: the recipe moves the
   crack, the mesh does not. Images: experiments/physical/break_stills.py PHOTO RUN/synthesis/photo_vs_sim.png
   <id>=<run dir> ... (all seven), and a figure like reference/fig_ma3_neck_recipe.png.
6. Grid video: the seven crush videos in a 4 x 2 ffmpeg xstack (the eighth tile a caption: recipe, nodes, median
   peak), like experiments/physical/crush_grid.sh -> RUN/synthesis/all_systems_crush.mp4 ({CPU_GL}).
7. brief.md (one page): what was run, the consensus numbers with spreads, what to trust, the outliers, the
   comparison with the test, the recommendation (which system to use by default and at what cost, which numbers to
   report and which to report only as a band), and what is not verified. Send it to the Study lead."""),
}

FLOWS = [("lead", s) for s in SYSTEMS] + [(s, "verifier") for s in SYSTEMS] + [("verifier", "synthesis")]
REPORTS = [*SYSTEMS, "verifier", "synthesis"]

W = 96 + 7 * 304 - 16 + 24 - 72   # systems group spans the seven cards
GROUPS = [
    {"id": "lead", "label": "Study lead: contract (parts, material, loads, 40k-node budget, crush recipe)",
     "x": 960, "y": 40, "w": 384, "h": 300},
    {"id": "systems", "label": "Seven mesh systems in parallel, one contract (fan-out)", "x": 72, "y": 360, "w": W,
     "h": 300},
    {"id": "gate", "label": "Gate: equilibrium, gates, budget, peak reached", "x": 960, "y": 680, "w": 384, "h": 300},
    {"id": "synth", "label": "Synthesis: consensus, outliers, test comparison, brief (fan-in)", "x": 960, "y": 1000,
     "w": 384, "h": 300},
]


def py_check(script: str) -> str:
    return f'python3 -c "{script}"'


def analysis_check(sid: str) -> str:
    """After each turn: an analysis.json, once written, carries every required key (no file yet also passes)."""
    keys = ["peak_force_N", "travel_at_peak_mm", "crack_site", "crack_hf", "refinement_map", "caveats", "reaction_N"]
    return py_check(f"import json,os,sys; p='{RUN}/systems/{sid}/analysis.json'; "
                    f"sys.exit(0 if not os.path.exists(p) or all(k in json.load(open(p)) for k in {keys}) else 1)")


def verdict_check() -> str:
    """After each turn: a verdict.json, once written, covers all seven systems with pass / flag / fail."""
    return py_check(f"import json,os,sys; p='{RUN}/verify/verdict.json'; S={list(SYSTEMS)}; "
                    "sys.exit(0 if not os.path.exists(p) or all(json.load(open(p)).get(s,{}).get('verdict') "
                    "in ('pass','flag','fail') for s in S) else 1)")


def agent(role: str) -> dict:
    name, group, x, y, job = ROLES[role]
    opus = role in ("lead", "synthesis")
    advanced = {"model": "opus" if opus else "sonnet", "permission": "unrestricted", "resume": "resume-if-possible"}
    if not opus:
        advanced["integrations"] = ["feagent-mcp"]
    if role == "lead":
        advanced["writes_status"] = True
    if role in SYSTEMS:
        advanced["check"] = analysis_check(role)
    if role == "verifier":
        advanced["check"] = verdict_check()
    return {"id": role, "group_id": group, "name": name, "kind": "claude", "job": f"{job}\n\n{CONTEXT}",
            "folder": str(WORK), "x": x, "y": y, "advanced": advanced}


def connection(a: str, b: str, kind: str) -> dict:
    return {"id": f"{a}-to-{b}" if kind == "hands-work-to" else f"{a}-reports-{b}", "from": a, "to": b,
            "kind": kind}


def team() -> dict:
    connections = [connection(a, b, "hands-work-to") for a, b in FLOWS]
    connections += [connection(r, "lead", "reports-to") for r in REPORTS]
    draft = {"id": TEAM_ID, "name": TEAM_NAME, "folder": str(WORK), "groups": GROUPS,
             "agents": [agent(r) for r in ROLES], "connections": connections, "revision": 0,
             "components": [{
                 "id": "feagent-mcp", "name": "feagent mesh harness", "kind": "mcp", "program": "feagent-mcp",
                 "description": "Geometry digest, face queries, automesh, mesh, quality gates and renders for STEP parts.",
                 "install": f"uv tool install --editable {WORK}   # provides feagent and feagent-mcp"}]}
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
    out = TEAM_DIR / "feagent-multisystem.json"
    data = team()
    out.write_text(json.dumps(data, indent=2) + "\n")
    print(f"wrote {out} ({len(data['team']['agents'])} agents, worktree {WORK})")
    photo = _arg("--photo", None)
    if photo:
        dest = RUN / "inputs" / "abc_00000065_real_break.png"
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            shutil.copy2(photo, dest)
        print(f"photo: {dest}")
    if "--install" in sys.argv:
        install(data)
