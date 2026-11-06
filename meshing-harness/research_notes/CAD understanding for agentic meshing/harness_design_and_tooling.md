# Harness Design Principles and CAD/CAE Tooling Ecosystem for an Agentic Meshing Harness

Research date: 2026-10-30. GitHub star counts and last-push dates were pulled live with the `gh` API on 2026-10-30 unless noted. "Unverified" marks claims taken from secondary sources or vendor marketing that I could not check against a primary source.

## 1. Coding-agent harness design: which choices measurably improve success?

### Takeaway
The best evidence (the SWE-agent ablations and Anthropic's tool/context guidance) says what matters most is the interface the agent works through. Good interfaces have a few consolidated, poka-yoke tools, concise bounded observations, guardrails that reject bad edits right away with actionable errors, and active context management (collapsing or compacting old observations, sub-agents, file-based notes). There is one counter-trend: with frontier models, a bash-only ~100-line loop (mini-swe-agent) now scores >74% on SWE-bench Verified. Custom tools matter less for generic actions and more for domain checks the model cannot easily do through a shell.

### Cited Findings

**SWE-agent / ACI (Princeton, 2024)**
- SWE-agent introduced the "agent-computer interface" (ACI): LM agents are a new kind of end user and benefit from interfaces built for them. It reported pass@1 of 12.5% on SWE-bench and 87.7% on HumanEvalFix. arXiv 2405.15793, v1 6 May 2024, v3 11 Nov 2024 — [arXiv](https://arxiv.org/abs/2405.15793)
- Four ACI design principles: (1) actions simple and easy to understand; (2) actions compact and efficient; (3) environment feedback informative but concise; (4) guardrails mitigate error propagation and speed recovery — [SWE-agent paper HTML](https://arxiv.org/html/2405.15793v3)
- ACI components: search tools return at most 50 results, and if there are more the agent is told to refine the query. The file viewer shows a 100-line window with line numbers and omitted-line counts. `edit` replaces a line range and is checked by a flake8 linter; edits that introduce syntax errors are discarded and the agent is asked to retry. Observations older than the last 5 are collapsed to one line. Empty output returns "Your command ran successfully and did not produce any output." — [SWE-agent paper HTML](https://arxiv.org/html/2405.15793v3)
- Ablation (SWE-bench Lite, GPT-4 Turbo; full config 18.0% resolved):
  - No edit command: 10.3% (−7.7)
  - Edit without linting: 15.0% (−3.0)
  - Iterative search: 12.0% (−6.0). This is worse than no search tools at all (15.7%), because agents exhaustively stepped through every match.
  - 30-line viewer: 14.3%. Full-file viewer: 12.7%.
  - Full history with no collapsing: 15.0%
  - No demonstration: 16.3%
  
  — [SWE-agent paper HTML](https://arxiv.org/html/2405.15793v3)
- mini-swe-agent (same group, created 2025-06-28, ~9.0k stars) uses bash as its only tool and needs no tool-calling API. Each action is an independent `subprocess.run`, and the history is linear. It "Scores >74% on the SWE-bench verified benchmark" (the README doesn't say which model). The authors argue that "as LMs improved, much of that scaffolding became unnecessary." — [mini-swe-agent GitHub](https://github.com/SWE-agent/mini-swe-agent)
- SWE-agent repo: ~20.7k stars, pushed 2026-10-26 — [GitHub](https://github.com/SWE-agent/SWE-agent)

**Anthropic guidance**
- "Building effective agents" (19 Dec 2024):
  - Distinguishes workflows (predefined code paths) from agents (LLM directs its own tool use).
  - Patterns: prompt chaining, routing, parallelization (sectioning/voting), orchestrator-workers, and evaluator-optimizer. Evaluator-optimizer is for tasks with "clear evaluation criteria" where "iterative refinement provides measurable value."
  - Advice: start simple and use LLM APIs directly before frameworks.
  
  — [Anthropic](https://www.anthropic.com/engineering/building-effective-agents)
- Same post, ACI appendix:
  - Treat ACI like HCI.
  - Write tool docs like a good docstring for a junior dev, with examples, edge cases and boundaries.
  - Poka-yoke tools: "Change the arguments so that it is harder to make mistakes."
  - Prefer formats close to natural text, with no line counting and no code escaped in JSON.
  - On their SWE-bench agent they "actually spent more time optimizing our tools than the overall prompt." Requiring absolute file paths fixed relative-path errors, and the model then used them "flawlessly."
  
  — [Anthropic](https://www.anthropic.com/engineering/building-effective-agents)
- "Writing effective tools for agents" (11 Sep 2025):
  - Don't just wrap API endpoints. Build a few high-impact consolidated tools (e.g. `search_contacts` instead of `list_contacts`, `schedule_event` instead of list+create), and avoid overlapping tools.
  - Namespace tools (e.g. `asana_projects_search`).
  - Return high-signal fields such as names and types, not UUIDs. Resolving IDs to readable names reduced hallucinations.
  - Offer a `response_format` enum (concise/detailed). In their example a concise response was ~72 tokens vs ~206 for detailed.
  - Use pagination, filtering and truncation with sensible defaults. Claude Code caps tool responses at 25,000 tokens by default.
  - Error messages should say what went wrong and how to fix it, with correct examples, not stack traces.
  - Use unambiguous parameter names (`user_id`, not `user`).
  - Do evaluation-driven tool development: realistic multi-call tasks with lenient verifiers and held-out sets, and let Claude refactor the tools from transcripts.
  - Refining tool descriptions helped Claude Sonnet 3.5 reach SOTA on SWE-bench Verified.
  
  — [Anthropic](https://www.anthropic.com/engineering/writing-tools-for-agents)
- "Effective context engineering for AI agents" (29 Sep 2025):
  - "Context rot": recall degrades as context grows, so aim for the "smallest possible set of high-signal tokens."
  - Bloated tool sets are a common failure mode: "If a human engineer can't clearly say which tool to use, the agent likely can't either."
  - Just-in-time retrieval: keep lightweight references (paths, queries) and load data with tools. Claude Code uses `head`/`tail`/grep rather than loading full datasets, while CLAUDE.md is loaded up front.
  - Compaction: keep decisions, unresolved bugs and the five most recently accessed files; drop redundant tool output. Clearing old tool results is the lightest form.
  - Structured note-taking: to-do lists, NOTES.md, the memory tool.
  - Sub-agents with clean contexts return condensed 1,000–2,000-token summaries.
  
  — [Anthropic](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)
- "Effective harnesses for long-running agents" (26 Nov 2025):
  - An initializer agent sets up `init.sh`, a progress log, an initial git commit, and a JSON feature list with every feature initially failing. JSON was chosen because models are less likely to inappropriately overwrite it than Markdown.
  - Coding sessions work one feature at a time and start by reading progress notes, git log and the feature list, then running a basic end-to-end check.
  - Failure modes observed: "declares victory on the entire project too early," marking features done without end-to-end testing, and leaving the environment broken.
  
  — [Anthropic](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents)

**Claude Code / Agent SDK**
- Claude Code best practices (current docs):
  - Separates non-deterministic guidance (prompt, CLAUDE.md, skills) from deterministic harness enforcement (hooks, deny rules, managed settings).
  - "If something has to happen every time, make it a hook... CLAUDE.md is for guidelines."
  - CLAUDE.md should hold only what Claude can't discover itself, such as exact test/typecheck commands.
  - Skills are loaded lazily: only the name and description sit in every session, and the full file is read when needed.
  
  — [Claude Code docs](https://code.claude.com/docs/en/best-practices)
- The Claude Agent SDK gives "the same tools, agent loop, and context management that power Claude Code," in Python and TypeScript. It includes built-in tools, hooks, subagents, MCP, permissions, sessions (resume/fork), skills/commands/memory loaded from `.claude/`, and plugins — [Agent SDK overview](https://code.claude.com/docs/en/agent-sdk/overview)
- Repo activity: anthropics/claude-code ~154k stars; claude-agent-sdk-python ~8.4k stars (created 2025-06-11); both pushed 2026-10-31 — [GitHub](https://github.com/anthropics/claude-agent-sdk-python)

**Other harnesses**
- CodeAct (ICML 2024; basis of the OpenHands agent lineage, but the abstract doesn't mention OpenHands — relationship from my background knowledge, unverified here):
  - Executable Python code as a unified action space beats JSON/text actions by up to 20% success rate across 17 LLMs.
  - Code can compose multiple tools and be revised across turns.
  
  — [arXiv 2402.01030](https://arxiv.org/abs/2402.01030)
- OpenHands/OpenHands: ~92.3k stars, pushed 2026-10-30 — [GitHub](https://github.com/OpenHands/OpenHands)
- Aider repo map: a concise map of files and key symbols (classes, functions, signatures). It is ranked with a graph algorithm over file dependencies and fit into a token budget (`--map-tokens`, default 1k). It helps the model use APIs from signatures alone and reuse existing abstractions — [Aider docs](https://aider.chat/docs/repomap.html)
- Aider edit formats:
  - whole-file
  - search/replace `diff`
  - `diff-fenced` (for Gemini)
  - `udiff` (for GPT-4 Turbo; it reduced placeholder/"lazy" code)
  - architect/editor split: a planner model writes plain-text changes and an editor model formats the edits
  
  The best edit format depends on the model — [Aider docs](https://aider.chat/docs/more/edit-formats.html)
- Aider-AI/aider: ~49.8k stars, last push 2026-05-22 (slowing activity) — [GitHub](https://github.com/Aider-AI/aider)
- AGENTS.md is an open "README for agents" format: build/test commands, style, testing instructions, and nested per-subproject files where the closest file wins. It emerged from OpenAI Codex, Amp, Jules, Cursor and Factory, is now stewarded by the Agentic AI Foundation (Linux Foundation), and is used by 60k+ projects and many agents including Codex and Aider — [agents.md](https://agents.md/)
- openai/codex: ~132k stars, created 2025-04-13, pushed 2026-10-31 — [GitHub](https://github.com/openai/codex)

### Inferences
- Transfers to meshing:
  - Make "edit geometry/mesh parameters" a guarded action: validate (e.g. check the OCC shape and that the mesh generates without inverted elements) and reject or roll back on failure. This is the analogue of SWE-agent's lint-gated edit, which was worth 3 points out of 18.
  - Keep mesh statistics bounded and summarized, like the 100-line window and 50-result cap: histograms, worst-N elements, per-surface summaries. Don't dump node lists.
  - Collapse old observations, especially old images, which are token-expensive.
- Iterative search was *worse than no tool*, so beware tools that let the agent page through entities one at a time. Provide summarized queries instead, e.g. "list surfaces with area < X" or "faces adjacent to volume 3".
- Tools should take human-meaningful handles (physical-group names, "inlet face") rather than raw gmsh integer tags, which shift after boolean ops and re-imports. This follows Anthropic's "resolve UUIDs to names" result. OFFTECH's gmsh-mcp revision guards (see §4) address the same tag-instability problem.
- Hybrid design. mini-swe-agent shows that a strong model with a Python/bash sandbox can do most generic work through code (consistent with CodeAct). The harness's added value should then be domain verifiers and renderers the model can't easily build itself: quality metrics, watertightness and topology checks, multi-view renders, and solver smoke tests. That is better than dozens of thin wrappers over the gmsh API.
- Use deterministic hooks (Stop-hook equivalent) to enforce "no export until mesh-quality gate passes," and skills to hold meshing procedures (boundary layers, CFD vs. structural sizing), which load only on demand.
- A JSON requirements/feature list (all initially failing) plus progress notes maps well onto meshing specs: element type, target size, BL layers, named groups, quality thresholds.

### Gaps
- No published ablation measures ACI choices on geometry/meshing tasks. All quantitative evidence is from code (SWE-bench) or graphics editing (BlenderGym).
- I didn't fetch the OpenHands paper (arXiv 2407.16741) or DeepSeek/other open-harness write-ups. OpenHands claims beyond the repo stats are uncited here.
- mini-swe-agent's >74% claim gives no model name on the README, and its comparison with SWE-agent ACI on the same model wasn't found.
- Codex harness internals (sandboxing, apply_patch format) were not researched beyond AGENTS.md.

## 2. Verification loops and "tests": how harnesses do it, and how to translate to geometry

### Takeaway
Every effective coding harness gives the agent an executable check and requires evidence: test output, typecheck output, screenshots. Without one, agents "declare victory" early. For meshing, the equivalent is a layered, machine-checkable gate:
1. CAD validity and healing
2. Mesh generation success
3. Element-quality thresholds (gmsh `minSICN`, `minSJ`, `gamma`, `minDetJac`)
4. Topology and boundary checks (named groups present, no duplicate nodes, watertight)
5. Downstream solver acceptance (`checkMesh`, a short CalculiX run)
6. Multi-view renders for VLM/human review

These are enforced by hooks, not just suggested in prompts.

### Cited Findings
- Claude Code docs:
  - "Claude stops when the work looks done. If it has nothing to run that tells it whether the work actually is done, 'looks done' is all it has to go on."
  - Verification sources include test files and "a browser screenshot next to the mockup."
  - Ask for *evidence*: "the command it ran and its output, or the screenshot."
  
  — [Claude Code best practices](https://code.claude.com/docs/en/best-practices)
- Enforcement mechanisms in Claude Code:
  - `/goal` sets a condition that a separate model checks after every turn.
  - A `Stop` hook runs a script whenever Claude tries to end; exit code 2 sends Claude back to work with the script's stderr. After eight consecutive blocks with no tool call, the harness overrides.
  - `/code-review` runs a reviewer subagent "that never saw your conversation."
  - Reviewer prompts should say "skip style," because "A reviewer that's asked to find problems will always find something."
  
  — [Claude Code best practices](https://code.claude.com/docs/en/best-practices)
- Long-running harness: browser automation with screenshots, testing "as a human user would," substantially improved bug detection compared with unit tests or curl alone. Agents may only flip a `passes` flag after end-to-end testing, and "It is unacceptable to remove or edit tests" — [Anthropic](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents)
- Tool evals should pair each task with a verifier but "Avoid overly strict verifiers that reject correct responses" — [Anthropic](https://www.anthropic.com/engineering/writing-tools-for-agents)
- Gmsh API quality measures (`gmsh.model.mesh.getElementQualities`, Gmsh 4.15.2 manual):
  - `minDetJac`/`maxDetJac`: adaptively computed Jacobian determinant
  - `minSJ`: sampled min scaled Jacobian
  - `minSICN`: signed inverted condition number
  - `minSIGE`: signed inverted gradient error
  - `gamma`: inscribed/circumscribed radius ratio
  - `innerRadius`, `outerRadius`, `minIsotropy`, `angleShape`, `minEdge`, `maxEdge`, `volume`
  
  — [Gmsh reference manual](https://gmsh.info/doc/texinfo/gmsh.html)
- Gmsh `Plugin(AnalyseMeshQuality)` computes J (validity only, fast), IGE (related to gradient error of the FE solution; equal to the scaled Jacobian for quads/hexes) and ICN (related to stiffness-matrix condition number). It can hide elements below a threshold — [Gmsh manual](https://gmsh.info/doc/texinfo/gmsh.html)
- Other gmsh API calls usable as checks or repairs — [Gmsh manual](https://gmsh.info/doc/texinfo/gmsh.html):
  - `gmsh.logger.getLastError()`
  - `gmsh.model.occ.healShapes(tolerance, fixDegenerated, fixSmallEdges, fixSmallFaces, sewFaces, makeSolids)`
  - `occ.importShapes` (BREP/STEP/IGES)
  - `mesh.getDuplicateNodes` / `removeDuplicateNodes`
  - size control via `setSizeCallback`, fields (Box/Distance/MathEval) and `Mesh.MeshSizeFromCurvature`
- Forum explanation: "Gamma" is undefined for hexes and pyramids, and the "S" in SICN/SIGE means "signed," so negative values indicate inverted elements — [CalculiX forum](https://calculix.discourse.group/t/mesh-quality-check/743) (secondary)
- Existing CAE MCP servers already build verification into tools:
  - OFFTECH gmsh-mcp-server: `minSICN`/`minDetJac` quality histograms with "threshold failures," and OpenFOAM `checkMesh` validation on export — [GitHub](https://github.com/OFFTECH/gmsh-mcp-server)
  - build123d-mcp: `validate()` before export, and `edit_feature` registers a change "only when the hole, other recognised holes, and the outer envelope pass its checks" — [GitHub](https://github.com/pzfreo/build123d-mcp)
  - Foam-Agent: a Reviewer agent with automatic error correction for up to 25 iterations — [GitHub](https://github.com/csml-rpi/Foam-Agent)

### Inferences
- Proposed "test suite" for a meshing harness, stored as a JSON spec that the agent can't edit:
  - (a) geometry imports, and healing changes nothing beyond tolerance
  - (b) volume, area and bounding box match the source CAD within ε (catches lost or merged bodies)
  - (c) all required physical groups exist and are non-empty
  - (d) mesh generates without error (`getLastError` empty)
  - (e) min SICN > 0, i.e. no inverted elements, and P1 of SICN/gamma above the threshold
  - (f) no duplicate nodes, conformal interfaces between volumes
  - (g) element count within budget
  - (h) the target solver reads the mesh (`checkMesh` / CalculiX dry run) and, optionally, a coarse solve converges or a mesh-convergence check passes
- Enforce the hard gates (d)–(f) with a Stop-hook-like mechanism. Keep soft judgments such as "BL resolution looks adequate" for a reviewer sub-agent with fresh context, consistent with the `/code-review` pattern and the "skip style" caveat (don't over-refine).
- Render-based checks belong after the metric gates, as a complement. Quantitative checks are cheap and deterministic; renders catch semantic errors such as a wrong surface refined or a missing fillet.

### Gaps
- No standard published "unit test" suite for meshes exists for agent use. Thresholds (e.g. SICN > 0.2, gamma > 0.1) are community practice, and I found no authoritative citation for specific cutoffs.
- I didn't verify whether `getElementQualities` works for all element orders and types in 4.15.x. An older issue suggested parts of SICN were not exposed in the API (now superseded per the 4.15.2 manual) — [gmsh issue 1676](https://gitlab.onelab.info/gmsh/gmsh/-/issues/1676)

## 3. Visual feedback: does rendering images back to multimodal models help in 3D/CAD tasks?

### Takeaway
Yes, with caveats. Render feedback and visual verification measurably help: BlenderGym verifier scaling, CADCodeVerify (+5% success, −7.3% point-cloud distance), CAD-Assistant, LL3M, and the build123d-mcp score jump on CADGenBench. But VLMs remain weak 3D verifiers. BlenderGym's best VLM verifier agrees with humans at 0.66 vs 0.79 human-human, and models hallucinate or miss differences. Renders should complement quantitative metrics, not replace them.

### Cited Findings
- BlenderGym (CVPR 2025, Stanford):
  - Setup: 245 start/goal Blender scene pairs across 5 tasks (object placement, lighting, procedural material, blend shapes, procedural geometry). The VLM edits Blender Python to match a goal render, and scoring uses photometric loss, Chamfer distance and negative-CLIP, not AI judges.
  - Pipeline: a generator (brainstormer + code editor) makes 4 candidates per step, a VLM verifier picks via pairwise render comparison, and the loop runs to depth 3.
  
  — [arXiv 2504.01786](https://arxiv.org/html/2504.01786v1)
- BlenderGym findings:
  - Scaling verification helps. Scaled InternVL2-8B beat unscaled GPT-4o and Claude 3.5 Sonnet on blend-shape edits.
  - Budget allocation matters: with a limited budget, favour generation; with a large budget, a higher share of verification is better.
  - Humans (average Blender users, 8-minute limit) beat every VLM system on all tasks.
  - Human-human verifier agreement was 0.79; the best VLM verifier (Claude 3.5 Sonnet) reached 0.66.
  - Failure modes: hallucinated or missed differences, code changes that don't match the identified visual differences, position bias, and >75% non-executable code from open models on procedural tasks.
  
  — [arXiv 2504.01786](https://arxiv.org/html/2504.01786v1)
- CADCodeVerify (ICLR 2025):
  - The VLM generates and answers validation questions about the rendered object, then corrects deviations.
  - The CADPrompt benchmark has 200 prompts with expert code.
  - Reported 7.30% reduction in point-cloud distance and a ~5% success-rate gain for GPT-4. Sources disagree between 5.0% and 5.5%, with the 5.5% sometimes described as compile rate.
  
  — [arXiv 2410.05340](https://arxiv.org/abs/2410.05340v1); [ICLR page](https://www.iclr.cc/virtual/2025/poster/30576)
- CAD-Assistant (arXiv 2412.13810, Dec 2024, v3 Aug 2025):
  - A training-free GPT-4o planner writes FreeCAD Python.
  - "a dedicated rendering and parameter serialization module, providing a multimodal representation of a CAD model."
  - It assesses each command's effect on the geometry and adapts subsequent steps.
  - It outperforms VLLM baselines and some supervised methods on CAD QA, auto-constraining and sketch parametrization.
  - Code at github.com/dimitrismallis/CAD-Assistant.
  
  — [arXiv 2412.13810](https://arxiv.org/abs/2412.13810)
- LL3M (arXiv 2508.08228, Aug 2025, UChicago):
  - Six agents (planner, retrieval with BlenderRAG, coding, critic, verification, user feedback) write Blender code.
  - The critic analyzes multi-view renders, and the verifier checks that the suggested fixes were applied.
  - The model stack (GPT-4o / Claude 3.7 Sonnet / Gemini 2.0 Flash on AutoGen) is from secondary summaries, unverified.
  
  — [arXiv 2508.08228](https://arxiv.org/html/2508.08228v1)
- build123d-mcp ("Give your AI CAD eyes") README claim: on the public CADGenBench leaderboard in June 2026, using build123d-mcp raised the same model's score from 0.360 to 0.457 and CAD validity from 88% to 100%. Its recommended loop is to build one feature at a time, render or measure after important steps, validate before export, then export — [GitHub](https://github.com/pzfreo/build123d-mcp). Self-reported, not independently verified.
- CADGenBench (Hugging Face) has two tasks: engineering drawing to STEP (generation) and STEP + change request (editing). It scores geometry, topology, interface compatibility and validity. A BenchLM snapshot (Aug 2026) shows `build123d-mcp-...-claude-opus-5-xhigh` leading generation at 59.6% — [BenchLM](https://benchlm.ai/benchmarks/cadgenbenchgeneration); [HF Space](https://huggingface.co/spaces/HuggingAI4Engineering/CADGenBench). Leaderboard figures conflict across mirrors, so treat them as unverified.
- ParaView-MCP (LLNL, IEEE VIS 2025 short paper) has "visual feedback capabilities that allow it to observe the viewport and iteratively refine visualizations" — [GitHub](https://github.com/llnl/paraview_mcp)
- Zoo's Zookeeper agent has "engine-level tools that let the agent inspect, snapshot, and debug geometry" and can "interpret selected geometry and renders" — [Zoo press](https://zoo.dev/press/zoo-introduces-zookeeper)
- FreeCAD MCP (neka-nat) returns screenshots by default from mutating tools (`include_screenshot=true`, `view_name` Isometric/Front/Top/...). It offers a `--only-text-feedback` flag to suppress them — a deliberate token/benefit trade-off knob — [GitHub docs/tools.md](https://github.com/neka-nat/freecad-mcp)
- In Anthropic's long-running harness, screenshot-based end-to-end testing improved bug detection, with limits such as the agent not seeing browser-native modals — [Anthropic](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents)

### Inferences
- For meshing, render the right thing:
  - fixed canonical views (iso + axis-aligned) of a surface mesh coloured by quality metric
  - clipped or section views for volume meshes
  - close-ups of worst-element locations
  - boundary-group colouring to confirm named surfaces
  
  Pair every image with a text summary (counts, min/percentiles) so the model isn't asked to read numbers from pixels, given the documented VLM verification weaknesses.
- Use pairwise "before vs. after" renders when choosing between candidate meshing parameters. This is the BlenderGym verifier pattern, and it's more reliable than absolute judgments.
- Make screenshots opt-in or off for intermediate steps (FreeCAD MCP pattern) and collapse old images from context to manage cost.

### Gaps
- No study found that ablates render feedback specifically for *mesh* generation or quality assessment. All evidence is for CAD/3D-asset generation or visualization.
- Exact CADCodeVerify numbers conflict across sources (5.0% vs 5.5%). The CADGenBench gain is self-reported by the tool author.
- I didn't fetch the LL3M full text to get quantitative ablations of the critic.

## 4. Existing MCP servers and agent tool integrations for CAD and CAE

### Takeaway
As of late 2026 the CAD side is crowded and partly official: Autodesk Fusion (local + Data + Compute beta), PTC Onshape FeatureScript MCP (Aug 2026), Blender Lab's official MCP, Zoo/KittyCAD MCP, and Ansys PyAnsys MCP packages. Community servers are very popular: MCP for Blender ~32.8k stars, FreeCAD MCP ~3.1k, KiCad ~3.1k. The meshing/CFD side is thin and immature. There's essentially one dedicated Gmsh MCP server (0 stars, created Sept 2026), OpenFOAM MCPs are mostly toy or educational, and Foam-Agent is the most substantive CFD agent. That leaves a clear gap for a high-quality meshing harness.

### Cited Findings (stars and last push as of 2026-10-30 via GitHub API)

**General-purpose CAD**
- **FreeCAD MCP (neka-nat/freecad-mcp)**: 3,145 stars, pushed 2026-10-25. It has an addon RPC server plus a `uvx freecad-mcp` client. Tools include:
  - `create/edit/delete_object`, `get_objects`, `get_view` (screenshot)
  - `execute_code` (GUI thread), `execute_code_async` + `get_async_status`, `execute_code_headless` (separate `freecadcmd`)
  - parts library
  - `run_fem_analysis`, which runs CalculiX and returns max von Mises, displacement and node count
  
  — [GitHub](https://github.com/neka-nat/freecad-mcp)
- Other FreeCAD MCPs:
  - bonninr/freecad_mcp (228 stars, stale since 2025-03)
  - contextform/freecad-mcp (139)
  - blwfish/freecad-mcp (62, "32 tools")
  - sandraschi/freecad-mcp (33, headless plus OpenFOAM CFD)
  
  — [GitHub search](https://github.com/search?q=freecad+mcp&type=repositories)
- **Blender**:
  - Community "MCP for Blender" (ahujasid, formerly blender-mcp): 32,836 stars, pushed 2026-10-29. "Not affiliated with the official" Blender; one-command setup for Claude Code, Codex, Cursor and others — [GitHub](https://github.com/ahujasid/mcp-for-blender)
  - Official Blender Lab MCP server (projects.blender.org/lab/blender_mcp; blender.org/lab/mcp-server). Reportedly requires Blender 5.1+, with add-on v1.0.3 released 2026-09-11. It reportedly offers Python API access, bundled docs, scene summaries, screenshots and code execution, and carries a warning that it executes LLM code "without any guards" — [StraySpark (secondary)](https://www.strayspark.studio/blog/official-blender-mcp-server-comparison-2026). Unverified from the primary page; tool count reported as 15 vs 20+ by different sources.
- **Autodesk Fusion**:
  - Official "Fusion MCP Server" (local): turns a running Fusion desktop session into an MCP endpoint for modelling, commands and document inspection. Enabled in Preferences > General > API. Tools are discovered dynamically and "may evolve between releases."
  - "Fusion Data MCP Server" (cloud, OAuth): hubs, projects and items.
  
  — [Autodesk Help](https://help.autodesk.com/view/fusion360/ENU/?guid=FMCP-OVERVIEW)
  - A cloud "Fusion Compute MCP" public beta (no local Fusion needed) was announced — [Autodesk blog](https://www.autodesk.com/products/fusion-360/blog/fusion-compute-mcp/) (not fetched; unverified details).
  - Community servers: faust-machines/fusion360-mcp-server (138), AuraFriday (127) — [GitHub search](https://github.com/search?q=fusion+360+mcp&type=repositories)
- **Onshape**:
  - PTC launched the official FeatureScript MCP Server on 13 Aug 2026 (Onshape Labs; endpoint fs-mcp.labs.onshape.app/mcp). It is narrowly scoped to generating FeatureScript custom features — [Onshape blog](https://www.onshape.com/en/blog/featurescript-mcp-server-enables-text-code-cad); [PR via Nasdaq](https://www.nasdaq.com/press-release/ptc-advances-leadership-ai-driven-design-launch-onshape-featurescript-mcp-server-2026)
  - Community servers: ReshefElisha/jarvis-onshape-mcp (175), hedless/onshape-mcp (159) — [GitHub search](https://github.com/search?q=onshape+mcp&type=repositories)
- **SolidWorks** (community only):
  - vespo92/SolidworksMCP-TS (247, pushed 2026-09)
  - alisamsam/Solidworks-MCP (185, "22 tools")
  - andrewbartels1/SolidworksMCP-python (88)
  - arthurle3210 swapi-pilot (32; API-doc search)
  
  — [GitHub search](https://github.com/search?q=solidworks+mcp&type=repositories)
- **Zoo.dev (KittyCAD)**:
  - Zookeeper is a conversational CAD agent that writes KCL (Zoo's CAD language) on Zoo's own geometry kernel.
  - The same tools are exposed through the Zoo-MCP server and the ML-ephant/KittyCAD API.
  - Zoo claims that building "a readable CAD language and a custom 3D kernel" is why it works well with LLMs (vendor claim).
  
  — [Zoo press](https://zoo.dev/press/zoo-introduces-zookeeper)
  - KittyCAD/mcp ("zoo-mcp"): 11 stars, created 2025-09-02, pushed 2026-10-30. Needs `ZOO_API_TOKEN`; supports stdio and streamable HTTP and ships prebuilt binaries — [GitHub](https://github.com/KittyCAD/mcp)
  - KittyCAD/modeling-app (Zoo Design Studio): 1,327 stars — [GitHub](https://github.com/KittyCAD/modeling-app)
  - A third-party review says Zoo is closer to single-part generation than to assemblies — [Leo AI (secondary)](https://www.getleo.ai/blog/zoo-text-to-cad-review)
- **Code-CAD (OCCT-based)**:
  - **build123d-mcp (pzfreo)**: 136 stars, created 2026-04-30, pushed 2026-10-15 — [GitHub](https://github.com/pzfreo/build123d-mcp)
    - persistent CAD session
    - PNG/SVG/DXF renders
    - measure volume, area, bounding box, topology, centre of mass
    - feature recognition (holes, bosses, countersinks, patterns)
    - printability and fit checks, STEP/STL import for comparison
    - session snapshots, 2D drawing previews
    - `last_error`, `validate()`; full reference in llms.md
  - Svetlana-DAO-LLC/cad-agent (56, build123d + MCP) — [GitHub search](https://github.com/search?q=build123d+mcp&type=repositories)
  - rishigundakaram/cadquery-mcp-server (20 stars, stale 2025-06). `verify_cad_query` is "a basic validator" and `generate_cad_query` is a stub — [GitHub](https://github.com/rishigundakaram/cadquery-mcp-server)
- Other CAD: AutoCAD (puran-water/autocad-mcp 606; U-C4N 207), Rhino (always-tinkering 57), KiCad EDA (mixelpixx 3,059) — [GitHub search](https://github.com/search?q=autocad+mcp&type=repositories)

**CAE / meshing / simulation**
- **Gmsh**: OFFTECH/gmsh-mcp-server is the only dedicated one found. 0 stars, created and last pushed 2026-09-23, pinned to gmsh 4.15.2 — [GitHub](https://github.com/OFFTECH/gmsh-mcp-server)
  - 36 tools; native gmsh state runs in isolated worker processes
  - optimistic revision guards (`expected_revision`, `request_key`, `worker_epoch` because gmsh tags change after worker recovery), async mesh jobs with cancellation
  - OCC operation batches, named surface patches
  - structured templates (transfinite box, O-H cylinder, pipe, extruded quad blocks), grading/BL layer planning
  - `minSICN`/`minDetJac` histograms
  - viewer checkpoints with 6 axis presets + iso, clipping, PNG capture
  - MSH 2.2 export with checksummed manifests, optional OpenFOAM conversion + `checkMesh`
  - versioned gmsh docs cache
  - It "does not automatically decompose arbitrary CAD into structured blocks."
- **OpenFOAM**:
  - **Foam-Agent (csml-rpi)**: 354 stars, pushed 2026-10-31. Architect / Input Writer / Runner / Reviewer agents on LangGraph, RAG over OpenFOAM tutorials (FAISS), Gmsh `.msh` import, Slurm, PyVista visualization, and core functions exposed as MCP tools. The README reports 100% on FoamBench (110 tasks) with Claude Opus 4.6, described as "a benchmark result, not a guarantee" — [GitHub](https://github.com/csml-rpi/Foam-Agent); [FoamBench arXiv 2509.20374](https://arxiv.org/abs/2509.20374)
  - MetaOpenFOAM: 210 stars, last push 2026-04 — [GitHub](https://github.com/Terry-cyx/MetaOpenFOAM)
  - webworn/openfoam-mcp-server: 125 stars, stale 2026-01. Self-reported "75% Functional," education-focused (Socratic questioning) — [GitHub](https://github.com/webworn/openfoam-mcp-server)
  - Other OpenFOAM MCPs have fewer than 5 stars — [GitHub search](https://github.com/search?q=openfoam+mcp&type=repositories)
- **ParaView**: llnl/paraview_mcp has 70 stars, last push 2026-04 (IEEE VIS 2025). Its README disclaimer says it relies on pvserver-client sync that is "deprecated in most recent ParaView versions" and that stability issues may occur — [GitHub](https://github.com/llnl/paraview_mcp)
- **Ansys**: official PyAnsys MCP packages on PyPI — `ansys-common-mcp` (shared infra, persistent Python sessions), `ansys-fluent-mcp` 0.5.0 (20 tools), `ansys-mapdl-mcp` 0.3.2, `ansys-mechanical-mcp` 0.2.1 (23 tools; local, remote or containerized). Agent loops are deliberately kept outside the packages — [PyPI fluent](https://pypi.org/project/ansys-fluent-mcp/0.5.0/); [PyPI mechanical](https://pypi.org/project/ansys-mechanical-mcp/0.2.1/); [PyPI common](https://pypi.org/project/ansys-common-mcp/0.3.5/)
  - Underlying libs: pymapdl (526 stars), pyfluent (538), pyansys-geometry (99), pyprimemesh (34; "Pythonic Meshing Client for Ansys Prime Server"), all pushed 2026-10-30/31 — [GitHub](https://github.com/ansys/pyprimemesh)
  - Community knewnothing-git/ansys-mcp-server: 70 stars, stale 2025-09 — [GitHub](https://github.com/knewnothing-git/ansys-mcp-server)
- **COMSOL** (community, unofficial): 777gegewu/comsol-mcp (125, controls the GUI via Java Shell), yushi-zhou/COMSOL-MCP-verifier (2, "verification-first") — [GitHub search](https://github.com/search?q=comsol+mcp&type=repositories)
- **Abaqus** (community): Cai-aa/abaqus-mcp (316, created 2026-01, pushed 2026-10-26), jianzhichun/abaqus-mcp-server (108, stale) — [GitHub](https://github.com/Cai-aa/abaqus-mcp)
- **Salome**: gnshb/salome-mcp (12 stars), grcarmenaty/SalomeMCP (0) — [GitHub search](https://github.com/search?q=salome+mcp&type=repositories)
- No dedicated MCP servers found for CalculiX or FEniCS (GitHub search returned none) — [GitHub search](https://github.com/search?q=calculix+mcp&type=repositories)
- **FE agent research**:
  - FeaGPT (arXiv 2510.21993, Oct 2025): an end-to-end Geometry-Mesh-Simulation-Analysis pipeline with "physics-aware adaptive meshes" and CalculiX. Validated on turbocharger cases and 432 NACA configurations. The abstract gives no quantitative metrics — [arXiv](https://arxiv.org/abs/2510.21993)
  - Gmsh+GetDP chatbot (Gemini 2.0 Flash, 2D eddy-current FE) — [arXiv 2511.17680](https://arxiv.org/html/2511.17680v2)
  - Birmingham evaluation of 9 LLMs generating Gmsh geometry for FE — [Birmingham](https://research.birmingham.ac.uk/en/publications/evaluating-the-performance-of-large-language-models-for-geometry-/)
  - GmshNet-8B fine-tune (model card only, no benchmarks) — [HF](https://huggingface.co/hexera-org/GmshNet-8B-v0.1-GGUF)
- **Directory**: punkpeye/awesome-mcp-servers, 97k stars — [GitHub](https://github.com/punkpeye/awesome-mcp-servers)

### Inferences
- Most CAD MCPs follow the same architecture: a GUI add-on RPC server (localhost socket) plus a thin MCP client, with an `execute_code` escape hatch plus a handful of structured tools and screenshots. The arbitrary-code tool does most of the work, which is consistent with CodeAct and mini-swe-agent. The structured tools add value mainly as inspection and verification.
- For a meshing harness, the strongest building blocks to reuse or learn from are:
  - OFFTECH gmsh-mcp: revision guards, isolated workers, quality histograms, viewer checkpoints. It's the best design reference despite 0 stars.
  - build123d-mcp: measure/validate/render loop with evidence of benchmark gains.
  - FreeCAD MCP: headless vs GUI execution modes; CalculiX FEM hook.
  - Foam-Agent: reviewer/error-correction loop, RAG over tutorials.
  - PyAnsys MCP: an official pattern for persistent solver sessions.
- Gmsh's integer-tag instability is a key harness hazard, already recognised by OFFTECH's `worker_epoch` guard. The harness should expose stable names (physical groups, or geometric signatures such as centroid/area/normal) instead.
- The real gap for meshing is not lack of servers but lack of mature *meshing* intelligence: automatic CAD defeaturing, block decomposition, BL sizing, quality-driven remeshing loops, and benchmarks. A harness filling that gap would be differentiated.

### Gaps
- Blender Lab official MCP details (tool list, maturity) come from secondary sources; I couldn't fetch projects.blender.org.
- Fusion Compute MCP and Zoo-MCP tool lists weren't enumerated.
- Activity beyond stars and last push (issues, release cadence, contributor counts) wasn't collected.
- Star counts are a weak maturity signal: several high-star CAD MCPs are mostly thin wrappers, and quality wasn't independently tested.
- No Siemens NX/Simcenter, Altair HyperMesh, Cubit/Coreform, or Pointwise MCP was searched for. These are relevant commercial meshing tools and a follow-up gap.

## 5. Open-source Python stacks suited for agent tools

### Takeaway
A fully open, headless, pip-installable stack exists and is actively maintained in 2026:
- geometry: OCCT via CadQuery/build123d/OCP or pythonocc
- meshing: gmsh Python API (primary), with netgen/NGSolve as an alternative
- conversion: meshio
- inspection and rendering: PyVista (offscreen)
- solvers: OpenFOAM via foamlib, FEniCSx (dolfinx), CalculiX

pygmsh and meshio are stale, so call the gmsh API directly and treat meshio as an I/O utility only.

### Cited Findings (stars and last push via GitHub API, 2026-10-30)
- **gmsh Python API** (Gmsh 4.15.2 manual):
  - OCC CAD import (`occ.importShapes` BREP/STEP/IGES) and healing (`occ.healShapes`)
  - size fields (Box, Distance, MathEval), curvature-based sizing (`Mesh.MeshSizeFromCurvature`), `setSizeCallback`
  - quality metrics via `getElementQualities` (minSJ/minSICN/minSIGE/gamma/minDetJac...) and `Plugin(AnalyseMeshQuality)`
  - duplicate node detection and removal
  - `gmsh.logger.getLastError()`
  - homology, cross fields
  
  — [Gmsh manual](https://gmsh.info/doc/texinfo/gmsh.html)
  - OFFTECH's server pins the `gmsh==4.15.2` pip wheel, which needs system OpenGL/GLU on Linux for the GUI — [GitHub](https://github.com/OFFTECH/gmsh-mcp-server)
- **pygmsh** (nschloe): 974 stars, last push 2023-10-04, so effectively unmaintained — [GitHub](https://github.com/nschloe/pygmsh)
- **meshio** (nschloe): 2,337 stars, last push 2024-07-23, so maintenance has slowed — [GitHub](https://github.com/nschloe/meshio)
- **PyVista**: 3,868 stars, pushed 2026-10-30. "3D visualization and mesh analysis for science and engineering" — [GitHub](https://github.com/pyvista/pyvista). Foam-Agent uses PyVista for visualization — [GitHub](https://github.com/csml-rpi/Foam-Agent)
- **Kitware trame** (Python web apps for VTK/ParaView, useful for a human-in-the-loop viewer): 709 stars, pushed 2026-09 — [GitHub](https://github.com/Kitware/trame)
- **CadQuery**: 6,026 stars, pushed 2026-10-24, "parametric CAD scripting framework based on OCCT" — [GitHub](https://github.com/CadQuery/cadquery)
  - CQ-editor: 1,263 stars — [GitHub](https://github.com/CadQuery/CQ-editor)
  - OCP bindings: 211 stars, pushed 2026-10-30 — [GitHub](https://github.com/CadQuery/OCP)
- **build123d**: 3,562 stars, pushed 2026-10-30 — [GitHub](https://github.com/gumyr/build123d). ocp-vscode viewer: 420 stars — [GitHub](https://github.com/bernhard-42/vscode-ocp-cad-viewer)
- **pythonocc-core**: 1,991 stars, pushed 2026-10-18, "Python package for 3D geometry CAD/BIM/CAM" — [GitHub](https://github.com/tpaviot/pythonocc-core)
- **FreeCAD**: 34,970 stars, pushed 2026-10-31. It has headless `freecadcmd` (used by FreeCAD MCP's `execute_code_headless`) and a FEM workbench with CalculiX — [GitHub](https://github.com/FreeCAD/FreeCAD); [freecad-mcp tools](https://github.com/neka-nat/freecad-mcp)
- **Netgen/NGSolve**: netgen 396 stars, ngsolve 590 stars, both pushed 2026-10-30. NGSolve is "high performance multiphysics finite element software" — [GitHub](https://github.com/NGSolve/ngsolve)
- **FEniCSx (dolfinx)**: 1,242 stars, pushed 2026-10-30 — [GitHub](https://github.com/FEniCS/dolfinx)
- **foamlib**: 231 stars, created 2024-03, pushed 2026-10-29, "A modern Python package for interacting with OpenFOAM" — [GitHub](https://github.com/gerlero/foamlib)
- **preCICE** (multiphysics coupling): 987 stars — [GitHub](https://github.com/precice/precice)
- **OpenSCAD**: 10,555 stars. CascadeStudio (browser OCCT): 1,498 — [GitHub](https://github.com/openscad/openscad)
- Benchmark context for code-CAD choice: the CADGenBench leader in an Aug 2026 snapshot used build123d-mcp with Claude Opus 5 — [BenchLM](https://benchlm.ai/benchmarks/cadgenbenchgeneration) (unverified mirror)

### Inferences
- Recommended core stack for the harness sandbox:
  - **gmsh API**: meshing + OCC import/heal + quality metrics
  - **build123d or CadQuery on OCP**: geometry creation, defeaturing, measurement, plus a code-CAD language LLMs already write well; build123d has demonstrated MCP gains
  - **PyVista offscreen**: deterministic multi-view renders coloured by quality, clipping and section views, worst-element close-ups
  - **meshio**: format conversion only
  - **foamlib** for OpenFOAM and **FreeCAD/CalculiX** or **dolfinx** for structural smoke tests
  
  Netgen (via `netgen.occ`) is a useful second mesher for cross-checks or fallback; that API name is from background knowledge, unverified here.
- Avoid pygmsh as a dependency (stale since 2023). The raw gmsh API is better documented and is what LLMs have most training data on.
- All of these can run headless in a container. The harness can therefore run each tool call in an isolated worker process, as OFFTECH does, to survive gmsh/OCCT crashes and segfaults. That robustness is a prerequisite for long autonomous loops.

### Gaps
- Salome/SMESH headless Python (and its MCPs, ~12 stars) wasn't researched in depth, and no primary Salome doc was fetched.
- No primary documentation was fetched for PyVista offscreen rendering specifics, netgen.occ, PyFoam, CalculiX Python wrappers (e.g. pygccx), or dolfinx's gmsh I/O. Capability claims beyond repo descriptions for these are unverified.
- Licensing (gmsh GPL, OCCT LGPL, FreeCAD LGPL, NGSolve LGPL) matters for a commercial harness but wasn't checked in this pass.
