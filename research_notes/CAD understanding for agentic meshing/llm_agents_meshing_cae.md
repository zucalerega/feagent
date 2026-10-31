# LLM Agents and ML Methods for Automating Meshing, FEA, and CFD Workflows

Research date: 2026-10-30. Window: 2022-2026, weighted to the most recent work. Most success-rate numbers below are self-reported by authors on benchmarks they built or chose. Few of them have been reproduced independently. Unless stated otherwise, "success" means the case *executes* (it reaches endTime with no crash), not that the results are physically correct.

## 1. LLM agent systems for CFD/FEA: architectures, meshing handling, benchmarks, success rates, failure modes

### Takeaway
The field started (2024-25) with multi-agent, RAG-heavy OpenFOAM systems: MetaOpenFOAM, OpenFOAMGPT, Foam-Agent and ChatCFD. With strong 2026 models, a **single generic coding-agent harness with an execution-feedback repair loop and access to solver tutorials matches or beats those systems**: 96.4% vs 88.2% on FoamBench. Across every system, meshing of *novel geometry* is the weakest stage. Almost no system checks mesh quality quantitatively.

### Cited Findings

**OpenFOAM / CFD agents**
- **MetaOpenFOAM (2024)**: an LLM-based multi-agent framework for CFD in OpenFOAM, and the first-generation baseline that later papers compare against — [arXiv 2407.21320](https://arxiv.org/pdf/2407.21320)
- **MetaOpenFOAM 2.0 (Feb 2025)**: chain-of-thought decomposition covering simulation and post-processing. On its own new benchmark it reports an 86.9% pass rate and an executability score of 6.3/7, up from 0% for MetaOpenFOAM 1.0 on the same benchmark, at about $0.15 per case — [arXiv 2502.00498](https://arxiv.org/abs/2502.00498)
- **OpenFOAMGPT 2.0 (Apr 2025; Feng, Xu, Chu)**: four agents (Pre-processing, Prompt Generation, OpenFOAMGPT simulator, Post-processing). Claims "100% success and reproducibility" over 450+ simulations (Poiseuille, porous-media single- and multi-phase, aerodynamics). The abstract does not name the LLMs or describe the meshing method. **The claim is unverified and the cases are mostly simple** — [arXiv 2504.19338](https://arxiv.org/abs/2504.19338). A secondary source says it uses blockMesh for simple geometries — [arXiv PDF](https://arxiv.org/pdf/2504.19338)
- OpenFOAMGPT 1 was RAG-based and was never released. Foam-Agent had to re-implement it as "OpenFOAMGPT-Alt" (no reviewer) to compare against it — [Foam-Agent 2.0 arXiv 2509.18178](https://arxiv.org/html/2509.18178v1); [OpenFOAMGPT 1 on ResearchGate](https://www.researchgate.net/publication/389557078_OpenFOAMGPT_A_retrieval-augmented_large_language_model_LLM_agent_for_OpenFOAM-based_computational_fluid_dynamics)
- **Foam-Agent (May 2025; v2 Sep 2025; published in CMAME vol. 461, 2026)**. Architecture:
  - Six agents (Architect, Meshing, Input Writer, Runner, Reviewer, Visualization), orchestrated as a LangGraph stateful graph with Pydantic-validated I/O.
  - Four FAISS RAG indices (Tutorial Structure, Tutorial Details, Execution Scripts, Command Docs), using text-embedding-3-small.
  - A 14-function MCP tool table (create_case, generate_mesh, run_simulation, get_simulation_logs, review_and_suggest_fix, apply_fix, generate_hpc_script…) — [arXiv 2509.18178](https://arxiv.org/html/2509.18178v1)
- Foam-Agent meshing pathways:
  - (a) writes blockMeshDict and/or snappyHexMeshDict and runs them;
  - (b) imports an external .msh file, or an existing blockMeshDict/snappyHexMeshDict;
  - (c) writes a **Gmsh Python script** from a natural-language geometry description, runs it, and converts the result with gmshToFoam.
  - The authors say native OpenFOAM meshing "could not capture" the obstacles in their cylinder and two-square cases, which motivated the Gmsh path.
  - **The paper describes no checkMesh or other mesh-quality verification** — [arXiv 2509.18178](https://arxiv.org/html/2509.18178v1)
- Foam-Agent results on FoamBench (110 cases, execution success):
  - Claude 3.5 Sonnet: 88.2%, vs MetaOpenFOAM 55.5% and OpenFOAMGPT-Alt 37.3%.
  - GPT-4o: 59.1% / 17.3% / 45.5%.
  - Ablation: the **Reviewer (error-log repair loop) is the dominant factor**, lifting success from about 48-57% to 86-88%, at an average of only 0.79-1.87 reviewer loops.
  - Hierarchical RAG vs a single index: 57.3% vs 44.6% without the reviewer, 88.2% vs 84.6% with it — [arXiv 2509.18178](https://arxiv.org/html/2509.18178v1)
- Foam-Agent caveats:
  - The paper concedes that FoamBench-Basic and the RAG knowledge base both come from the OpenFOAM tutorial corpus.
  - Under a stricter criterion, the complete-success rate is 46/110 = 41.8% — [arXiv 2505.04997](https://arxiv.org/html/2505.04997)
- **Foam-Agent v2.0.0 GitHub README (2026; repo-reported, not peer reviewed)**:

  | Model | Reviewer loops | Basic | Advanced |
  |---|---|---|---|
  | Opus 4.6 | 25 | 100% | 100% |
  | Opus 4.6 | 10 | 85.45% | 100% |
  | Sonnet 4.6 | 25 | 87.88% | 75% |
  | Haiku 4.6 | 25 | 54.55% | 37.5% |
  | gpt-5.4 | 25 | 45.45% | 75% |
  | gpt-5.3-codex | 25 | 54.55% | 62.5% |

  - The README says the 100% result is "a benchmark result, not a guarantee for arbitrary cases."
  - Accepts external Gmsh .msh files (ASCII 2.2). Meshing failures are routed to the Reviewer.
  - Ships an MCP server (`foamagent-mcp`) and a Claude Code `/foam` skill (up to 5 repair iterations).
  - The README does not mention checkMesh or snappyHexMesh — [GitHub csml-rpi/Foam-Agent](https://github.com/csml-rpi/Foam-Agent)
- **ChatCFD (May 2025; v3 Feb 2026; Adv. Intell. Discov.)**:
  - Components: structured domain knowledge bases, an Error Locator, iterative reflection, a Physics Interpreter, and a Solver Template DB. MCP-compatible. Runs on DeepSeek-R1/V3.
  - On a 315-case benchmark: 82.1% execution success, vs 42.3% for Foam-Agent and 6.2% for MetaOpenFOAM; 68.12% physical fidelity; about $0.208 per case.
  - Removing the Solver Template DB drops accuracy to 48%.
  - The abstract does not describe how meshes are handled — [arXiv 2506.02019](https://arxiv.org/abs/2506.02019); [Wiley](https://advanced.onlinelibrary.wiley.com/doi/full/10.1002/aidi.202500174?af=R)
  - **Conflict:** ChatCFD reports MetaOpenFOAM at 6.2%, while MetaOpenFOAM 2.0 reports 86.9% on its own benchmark. Cross-paper numbers are not comparable.
- **"A Preliminary Assessment of Coding Agents for CFD Workflows" (Peking Univ., Feb 2026)**. Setup: the generic OpenCode harness (bash/read/edit), plus or minus a "FoamHelper" system prompt (tutorial-first search, copy one baseline, minimal dictionary edits, log-driven repair to endTime, checkMesh must pass), on OpenFOAM v10. Results on FoamBench-Advanced:
  - Tutorial-derivative tasks, MiniMax-M2.1: 4/9 reach endTime with the default prompt, 9/9 with the custom prompt.
  - Obstacle-flow tasks (geometry plus meshing): MiniMax-M2.1 failed meshing. It produced meshes that did not represent the obstacle, or reused a tutorial mesh unchanged. Its one usable case used snappyHexMesh instead of the requested multi-block blockMesh.
  - GPT-5.2 completed all 4 obstacle cases, using the Gmsh Python API for quad-dominant and hybrid meshes. The hybrid mesh needed human-guided iterations.
  - Mesh verification was **visual only**. No checkMesh numbers were reported.
  - Conclusion: "meshing capability depends largely on the underlying model." Physics errors that leave no trace in the logs go undetected — [arXiv 2602.11689](https://arxiv.org/html/2602.11689)
- **"What Do CAE Simulation Agents Really Need Beyond a Generic Harness?" (Shi & Zhang, 3 Sep 2026)**:
  - Holding information access and repair budget equal, a single-agent harness scores **96.4% vs 88.2%** for multi-agent specialized systems on FoamBench.
  - No repair round vs execution-feedback repair: 71.8% → 96.4%.
  - Providing solver tutorials as domain knowledge: 80.9% → 96.4%, the largest measured gain.
  - **Scripted reflection: no measurable benefit** — [arXiv 2609.03718](https://arxiv.org/abs/2609.03718) (only the abstract was read; the per-model details were not)
- IteraSim RAG (Jul 2026), a multi-stage retrieval-augmented agentic back-end for OpenFOAM, exists. I did not read its details — [arXiv 2607.20346](https://arxiv.org/pdf/2607.20346)
- AWS HPC blog: an LLM writes Gmsh Python API code with JSON-controlled procedural mesh generation, claiming an 8-12x time saving. **This is a vendor case study; treat it as marketing** — [AWS HPC blog](https://aws.amazon.com/blogs/hpc/accelerating-cfd-development-from-years-to-weeks-with-agentic-ai-and-aws/)

**FEA agents**
- **MechAgents (Ni & Buehler, 2024, Extreme Mechanics Letters 67:102131)**: GPT-4 multi-agent role-play that solves elasticity problems with FEniCS. Iterative agent-to-agent correction outperformed a single agent — [ResearchGate](https://www.researchgate.net/publication/378645504_MechAgents_Large_language_model_multi-agent_collaborations_can_solve_mechanics_problems_generate_new_data_and_integrate_knowledge); citation details via [VFEAgent](https://arxiv.org/html/2605.28978v1)
- **FeaGPT (Oct 2025; Qi, Xu, Chu)**:
  - Pipeline: Gemini 2.5 Pro writes FreeCAD Python scripts that produce STEP geometry. Gmsh's Python API builds the mesh from LLM-extracted parameters (density, refinement zones, element type). Physical groups are auto-created from descriptions such as "fixed" surfaces. Mesh fields are configured for smooth size transitions. Output is a CalculiX INP file.
  - Open-source CLI — [arXiv 2510.21993](https://arxiv.org/pdf/2510.21993); [GitHub naividh/FeaGPT](https://github.com/naividh/FeaGPT/blob/main/README.md)
- **AbaqusAgent (UT Arlington/RPI, 28 May 2026)**:
  - Six agents (Interpreter, Architect, Input Writer, Runner, Reviewer, Visualization) that mirror Foam-Agent. FAISS case retrieval. The LLM writes the Abaqus .inp mesh directly (using *NGEN/*ELGEN). The only mesh rule is a minimum of 20 elements; there are no quality metrics and no convergence study.
  - Claude Opus 4.6: 86% success and 86% accuracy on 50 cases (92.5% in-RAG, **60% on the 10 external cases**).
  - Ablation: full system 90%, no RAG 80%, **no Reviewer 45%**. GPT-5.2: 65% success, 45% accuracy.
  - Meshing failure: an open-hole plate got an unstructured mesh that misrepresented the hole until the prompt specified a radial mesh. GPT-5.2 also hallucinated a square section for a circular rod.
  - Stated scope limit: no "custom-mesh" solid mechanics — [arXiv 2606.00138](https://arxiv.org/html/2606.00138v1)
- **VFEAgent (Peking Univ., 27 May 2026)**: Stage A is a VLM that turns engineering drawings into a solver-agnostic intermediate representation (nodes, connectivity, materials, BCs), with 4-level validation. Stage B (Text2Abaqus) adds an AST preflight check, sandboxed execution, a reflexive debugger with an experience-replay buffer, and a template fallback.
  - Claims 1.00 on execution with a ~15-case benchmark, vs the best baseline (GPT-5) at 0.60. **The sample is small and no run-to-run variance is reported.**
  - Baseline Abaqus scripting failures: "lifecycle blindness" 45% (never submitting or waiting on the job), API hallucination 30%, and unsafe state manipulation 25% (deleting the root model).
  - No mesh-quality metrics are reported — [arXiv 2605.28978](https://arxiv.org/html/2605.28978v1)
- **ALL-FEM (CMAME 2026; arXiv 2603.21011)**: fine-tuned agentic LLMs for FEM code generation. Its authors note the models are fine-tuned for code generation, "rather than for robust automated meshing or tagging" — [arXiv 2603.21011](https://arxiv.org/html/2603.21011v2); [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0045782526002586)
- Other 2026 FEM agents: PDE-Agents (knowledge-graph-augmented; claims no manual intervention for heat-transfer meshing) — [arXiv 2606.07850](https://arxiv.org/html/2606.07850). A constrained natural-language interface for FEniCS restricts the LLM to front-end work, including custom Gmsh geometry, and counts mesh generation plus boundary-tag validation as first-pass success criteria — [arXiv 2606.10928](https://arxiv.org/html/2606.10928). VFEAgent also cites MooseAgent (RAG for MOOSE), AutoFEA (RAG plus templates) and FRAME, and criticizes the RAG/template systems as "pseudo-modeling" — [arXiv 2605.28978](https://arxiv.org/html/2605.28978v1)
- FreeCAD tooling: AnkusDrive is an MCP server exposing the FreeCAD Python API (hundreds of tools) plus CalculiX solves for LLM agents. It is a practitioner project, not research — [glama.ai](https://glama.ai/mcp/servers/gchen19/AnkusDrive/tree)

### Inferences
- Across Foam-Agent, AbaqusAgent, the harness paper and the coding-agent paper, the same three levers matter, in this order: (1) the base model, (2) an **execution-feedback repair loop**, worth +30-45 points everywhere it was ablated, and (3) **retrieval of working tutorial cases** to copy from. Multi-agent decomposition and scripted reflection add little on top of a strong model.
- Every system's headline number measures *executability*. When results are checked for correctness, success falls by roughly half (Foam-Agent 88% → 42% strict; CFDLLMBench 34%). A meshing harness should treat "the solver ran" as necessary but nowhere near sufficient.
- Meshing of non-tutorial geometry is the gap all of these systems share. They handle it by importing external meshes, by having the LLM write Gmsh scripts, or not at all.

### Gaps
- No paper reports per-stage (meshing-specific) success rates, except the qualitative notes in 2602.11689 and 2606.00138.
- I did not read the full text of 2609.03718 (which models, which harness, what tutorials). I did not read the IteraSim, MetaOpenFOAM 3, or CFD-copilot papers. I found nothing on "CFD-copilot" specifically.
- I found no peer-reviewed work on LLM-driven ANSYS Fluent/Mechanical (PyAnsys) or COMSOL scripting beyond FEABench.

## 2. Work specifically on LLM-driven meshing and how mesh quality was verified

### Takeaway
**I found no dedicated paper or benchmark on LLM-driven meshing** (Gmsh script generation, snappyHexMeshDict authoring, size-field selection, boundary-layer/y+ setup, or quality-driven iteration). Meshing appears only as a sub-step inside CFD/FEA agents. Where it is "verified", that means visual inspection, an obstacle being present, or the solver running. No system closes the loop on checkMesh/Jacobian metrics, and none runs a convergence study.

### Cited Findings
- Foam-Agent's Gmsh path (NL geometry → Python Gmsh script → .msh → OpenFOAM) and its blockMesh/snappyHexMesh dict generation have no mesh-quality check described — [arXiv 2509.18178](https://arxiv.org/html/2509.18178v1)
- In the coding-agent study, the custom prompt *required* checkMesh to pass, but the paper reports only mesh images and "physically plausible" velocity fields. Weaker models produced meshes missing the obstacle, which logs never flagged — [arXiv 2602.11689](https://arxiv.org/html/2602.11689)
- CFDLLMBench's doubleSquare example: Foam-Agent's geometry and mesh did not match the expert reference. The authors attribute this to weak spatial reasoning — [arXiv 2509.20374](https://arxiv.org/html/2509.20374)
- FeaGPT: the Gmsh mesh agent receives a parameter dictionary (mesh density, refinement zones, element type) and configures fields for smooth size transitions. It is parameter-driven, not iterative — [arXiv 2510.21993](https://arxiv.org/pdf/2510.21993)
- AbaqusAgent: mesh quality depended on the prompt. A radial mesh had to be specified explicitly for the hole. The only rule is ≥20 elements — [arXiv 2606.00138](https://arxiv.org/html/2606.00138v1)
- FEABench: meshing is "usually the default mesh" and only a short step. Problems avoid imported CAD — [arXiv 2504.06260](https://arxiv.org/html/2504.06260)
- ALL-FEM explicitly calls robust automated meshing and tagging out of scope — [arXiv 2603.21011](https://arxiv.org/html/2603.21011v2)
- A third-party "GMSH Engineering Mesh Agent" skill exists on MCP Market, with no validation — [mcpmarket](https://mcpmarket.com/tools/skills/gmsh-engineering-mesh-agent)
- Useful native hooks for an agent:
  - snappyHexMesh has an internal quality loop: vertex displacement is reduced when meshQualityControls are violated, and layers are removed and retried if post-layer checks fail.
  - A `-dryRun` style option checks case setup only — [OpenFOAM v14 user guide](https://doc.cfd.direct/openfoam/user-guide-v14/snappyhexmesh); [OpenFOAM.com 4.4](https://www.openfoam.com/documentation/user-guide/4-mesh-generation-and-conversion/4.4-mesh-generation-with-the-snappyhexmesh-utility)
  - checkMesh warns about nonAlignedEdges, a known intrinsic snappyHexMesh artifact — [snappyWiki](https://sites.google.com/site/snappywiki/snappyhexmesh)

### Inferences
- This is open territory. A harness that (a) emits Gmsh/sHM scripts, (b) parses checkMesh or Gmsh quality output into structured feedback, (c) checks geometric fidelity (for example, that a boundary patch exists for each named surface and the obstacle is resolved), and (d) runs a refinement study would go beyond anything published.
- Failures that logs do not catch (a missing obstacle, a wrong patch, a tutorial mesh silently reused) are the main meshing risk. That argues for explicit geometric assertions and rendering plus VLM inspection, not just checking the solver's exit code.

### Gaps
- I found no published benchmark of LLMs authoring snappyHexMeshDict, setting boundary layers/y+, or choosing size fields, and no measured comparison of Gmsh scripting vs sHM authoring by LLMs.

## 3. Benchmarks: tasks, metrics, frontier scores

### Takeaway
FoamBench is the de facto CFD-agent benchmark, but it is being saturated on its tutorial-derived Basic tier (100% claimed with Opus 4.6 plus 25 repair loops). Correctness-weighted scores are much lower. FEABench (COMSOL) is still essentially unsolved under strict correctness by 2025 models. No meshing benchmark exists.

### Cited Findings
- **CFDLLMBench (NREL/RPI; v3 dated 26 Apr 2026)** has three parts:
  - CFDQuery: 90 multiple-choice questions. o3-mini scored 92% (best), Gemma-2-9B 60%.
  - CFDCodeBench: 24 Python PDE/CFD problems; the best models reached about 14% success.
  - FoamBench: 126 OpenFOAM cases (110 Basic, 16 Advanced). Metrics are executability, folder-structure ROUGE, file-similarity ROUGE, NMSE vs reference, and an overall Success Rate.
  - On FoamBench Basic, Sonnet 3.5 + Foam-Agent scored success 0.336 (executability 0.836, NMSE score 0.427). Zero-shot Sonnet 3.5 scored 0.045.
  - On FoamBench Advanced: 0.25 with Foam-Agent, 0.007 zero-shot. Gemini 2.5 Flash scored 0.
  - The Advanced tier requires new geometry plus mesh generation without a tutorial. Removing RAG or the Reviewer costs about 10 points — [arXiv 2509.20374](https://arxiv.org/html/2509.20374); [GitHub NLR-Theseus/cfdllmbench](https://github.com/NLR-Theseus/cfdllmbench)
- The FoamBench numbers conflict: Foam-Agent reports 88.2% (execution only), while CFDLLMBench reports 34% for the same system (accuracy-gated) — [arXiv 2509.18178](https://arxiv.org/html/2509.18178v1) vs [arXiv 2509.20374](https://arxiv.org/html/2509.20374)
- 2026 FoamBench scores:
  - Generic single-agent harness: 96.4% — [arXiv 2609.03718](https://arxiv.org/abs/2609.03718)
  - Foam-Agent 2.0.0 with Opus 4.6 (25 loops): 100% Basic / 100% Advanced (repo-reported) — [GitHub](https://github.com/csml-rpi/Foam-Agent)
- **FEABench (Google Research; Apr 2025; ICLR 2025)**:
  - Tasks: FEABench Gold has 15 verified COMSOL-tutorial problems (ModelSpecs and Plan variants). FEABench Large has 200 algorithmically parsed problems. There is also a 768-snippet annotated library for retrieval.
  - Setup: the LLM writes COMSOL Java API calls, executed via MPh/JPype. The agent samples 20 initial solutions and corrects over 20 steps, with tool-lookup tools (QueryPhysicsInterfaces, etc.) and a VerifierLLM.
  - Metrics: executability, Model Tree Score, physics-interface factuality, Valid Target, and Relative Error under 10% (strict).
  - Best results: the multi-turn agent reached executability 0.88 with 2/15 valid targets. Claude 3.5 Sonnet baseline: 0.79 and 1/15. Only one problem passed strict, and it was a false match (a COMSOL default temperature). A SWE-agent Python run got 11 valid and 4 strict.
  - Failures concentrate in the physics block (hallucinated interface names, e.g. "Heat Transfer in Solids" vs the API name "HeatTransfer") — [arXiv 2504.06260](https://arxiv.org/html/2504.06260); [OpenReview](https://openreview.net/forum?id=hDkLpu1E64)
- FEM-Bench: a structured scientific-reasoning benchmark for code-generating LLMs on FEM math implementation — [arXiv 2512.20732](https://arxiv.org/html/2512.20732)
- PDEAgent-Bench: a multi-metric, multi-library benchmark for PDE solver generation — [arXiv 2605.09636](https://arxiv.org/pdf/2605.09636)
- EngDesign is cited as a benchmark for engineering design thinking — [VFEAgent](https://arxiv.org/html/2605.28978v1)
- Small in-house benchmarks: ChatCFD (315 cases), MetaOpenFOAM 2.0 (its own set), AbaqusAgent (50), and VFEAgent (~15) — sources as in section 1.

### Inferences
- A meshing harness needs its own benchmark. Candidate metrics:
  - geometric fidelity (Hausdorff distance to CAD, patch/physical-group completeness);
  - checkMesh/Jacobian pass rates;
  - y+ attainment;
  - GCI-based convergence of a target QoI;
  - cost (tokens, wall time, element count).
- FoamBench-Advanced (16 cases) and FEABench Gold (15) are too small to rank systems at the 95-100% level. Repo results claiming 100% should be read as saturation of tutorial-like tasks.

### Gaps
- I did not verify per-model figures that appear only in figures, such as CFDCodeBench per-model scores, nor FEABench scores for 2026 frontier models. None appeared to be published.

## 4. ML for meshing (non-LLM) — reusable as agent tools

### Takeaway
There is a substantial non-LLM literature. It includes learned sizing fields (GNN-predicted, iterative imitation of experts), GNN mesh-quality classifiers, RL for quad meshing and block decomposition, RL for adaptive refinement, and learned boundary-layer meshing. Most are research code trained on 2D/narrow domains. The most reusable pattern for an agent is "**learned sizing field → off-the-shelf mesher (Gmsh background field)**".

### Cited Findings
- **Learned sizing fields**
  - MeshingNet (2020) is a deep-learning sizing-field method for unstructured mesh generation and the early reference point — [ResearchGate](https://www.researchgate.net/publication/340704579_MeshingNet_A_New_Mesh_Generation_Method_based_on_Deep_Learning)
  - Ojha, Chen & Fidkowski (2022): neural networks predict element size and anisotropy for initial meshes in solution-adaptive flow problems — [UMich PDF](https://public.websites.umich.edu/~kfid/MYPUBS/Ojha_Chen_Fidkowski_2022.pdf)
  - **AMBER** (2024-25): imitation learning in which a GNN predicts a per-element sizing field from the current mesh. The field drives an off-the-shelf mesher over several iterations and is trained toward sizing fields projected from expert meshes — [arXiv 2406.14161](https://arxiv.org/html/2406.14161v1); [arXiv 2505.23663](https://arxiv.org/pdf/2505.23663)
  - **LAMG** (May 2025): a small GNN predicts the sizing field from sparse Monte-Carlo or coarse FEM data, then standard tools mesh and a single FEM solve runs. It acts as a surrogate for iterative multi-solve adaptation — [arXiv 2505.20457](https://arxiv.org/abs/2505.20457)
  - GCN-based background-grid simplification (Oct 2025): edge classification replaces per-edge procedural evaluation with one forward pass to generate sizing fields — [arXiv 2510.08645](https://arxiv.org/html/2510.08645)
- **Mesh-quality prediction**
  - MQENet (2023): dynamic graph attention classifies structured mesh quality, evaluated on the NACA-Market dataset — [arXiv 2309.01067](https://arxiv.org/pdf/2309.01067)
  - GMeshNet: a GNN mesh-quality evaluator (graph conv + pooling + jumping connections), evaluated on NACA-Market and NACA6510 — [Bohrium listing](https://www.bohrium.com/paper-details/evaluating-mesh-quality-with-graph-neural-networks/817350823129382918-3886)
- **RL for mesh generation**
  - Pan et al. (2022): quad mesh generation as an MDP solved with soft actor-critic — [arXiv 2203.11203](https://arxiv.org/abs/2203.11203v2)
  - DiPrete et al.: RL learns axis-aligned cuts that decompose CAD into meshable quad blocks — [arXiv 2302.11066](https://ar5iv.labs.arxiv.org/html/2302.11066)
  - Self-play RL for improving tri/quad mesh quality via local edits — [eScholarship](https://escholarship.org/content/qt34b0h2rp/qt34b0h2rp.pdf?t=sptjq3)
  - Dmsh (Jun 2026): multi-agent RL for all-quad meshing (topology simplification, geometric regularization, and meshing agents, with curriculum learning) — [arXiv 2606.10601](https://arxiv.org/pdf/2606.10601)
  - "Playing to Par" (2026): RL for quad block decompositions that reach the irregularity lower bound. Claims all-quad meshes on 96/96 held-out domains vs 51 for Gmsh's best configuration. **I saw only an aggregator listing, and no maintained code was found** — [listing](https://www.opentrain.ai/papers/playing-to-par-reinforcement-learning-for-provably-optimal-quadrilateral-block-d--arxiv-2609.32146/)
  - Deep RL for automatic quad meshing of turbomachinery blade passages, using dynamic action masking — [JCP 2025](https://sciencedirect.com/science/article/abs/pii/S0021999125005893?via=ihub%3D)
  - Agent-based mesh generator for flow analysis using deep RL — [ResearchGate](https://www.researchgate.net/publication/383036418_Development_of_agent-based_mesh_generator_for_flow_analysis_using_deep_reinforcement_learning)
- **RL for AMR**
  - Yang et al. (LLNL): AMR as an MDP; the policy size is independent of mesh size — [arXiv 2103.01342](https://arxiv.org/pdf/2103.01342)
  - ASMR / Adaptive Swarm Mesh Refinement (Jun 2024; v2 Jan 2026): elements act as collaborating agents with local rewards. Reportedly matches error-oracle AMR and beats heuristic and learned baselines — [arXiv 2406.08440](https://arxiv.org/abs/2406.08440)
  - RL for anisotropic p-adaptation in high-order solvers — [arXiv 2407.19000](https://arxiv.org/abs/2407.19000v1)
- **Boundary layers**
  - BLMeshNet (2026): a U-Net that treats boundary-layer growth as spatiotemporal sequence prediction — [Chinese J. Aeronautics](https://www.sciencedirect.com/science/article/pii/S1000936126001639)
  - Conventional tools control y+ by prescribing the first-layer thickness — [cfMesh+](https://cfmesh.com/cfmesh/); [Simmetrix](https://www.simmetrix.com/index.php/technologies/boundary-layers)
- A Dec 2025 survey frames AI as *complementing* established CAD-to-mesh algorithms rather than replacing them — [arXiv 2512.23719](https://arxiv.org/pdf/2512.23719v2)

### Inferences
- The best tool candidates for an LLM harness are:
  - learned or heuristic **sizing-field predictors**, emitted as a Gmsh background mesh / PostView field or as sHM refinement regions;
  - a **fast quality classifier** to triage candidate meshes before running full checkMesh/solver;
  - **RL-based AMR** for solution-driven refinement in FEM codes (e.g. MFEM, which is LLNL's ecosystem; this is an inference).
- Most of these models are trained on 2D or narrow families (NACA, blade passages). Expect poor transfer to arbitrary 3D CAD. Deterministic tools such as Gmsh fields, sHM and cfMesh should stay the backbone, with the LLM choosing parameters.

### Gaps
- I did not verify MeshingNet's method details from its primary paper. I found no maintained, general-purpose open-source learned sizing-field package for 3D CAD, and no learned y+/boundary-layer tool integrated with Gmsh or sHM.

## 5. Mesh-quality metrics, acceptance criteria, and verification practice

### Takeaway
There are concrete, citable thresholds an agent can enforce. OpenFOAM checkMesh errors at non-orthogonality > 70° and skewness > 4, warns at aspect ratio > 1000, and the snappyHexMesh meshQualityControls default to maxNonOrtho 65. Fluent practice (forum and Ansys staff guidance) is minimum orthogonal quality > 0.1 and maximum skewness < 0.9 (ideally < 0.8). Grid independence should be shown with the Celik et al. (2008) GCI procedure.

### Cited Findings
- **OpenFOAM checkMesh defaults** (source documentation v11-v14): `nonOrthThreshold` 70° and `skewThreshold` 4 are command-line options. `aspectThreshold` 1000 is a library constant in polyMeshCheck. The same 70°/4 defaults apply to the checkMesh functionObject — [checkMesh.C v13](https://cpp.openfoam.org/v13/applications_2utilities_2mesh_2manipulation_2checkMesh_2checkMesh_8C_source.html); [polyMeshCheck namespace](https://cpp.openfoam.org/v11/namespaceFoam_1_1polyMeshCheck.html); [OpenFOAM-dev commit](https://github.com/OpenFOAM/OpenFOAM-dev/commit/4ab99793058a1facf3ea49f99097e868e14c1517)
- **snappyHexMesh meshQualityDict defaults** (etc/caseDicts/meshQualityDict):

  | Setting | Default |
  |---|---|
  | maxNonOrtho | 65 (180 disables) |
  | maxBoundarySkewness | 20 |
  | maxInternalSkewness | 4 |
  | maxConcave | 80 |
  | minVol / minPyrVol | 1e-13 |
  | minTetQuality | 1e-15 (negative disables) |
  | minTwist | 0.02 |
  | minDeterminant | 0.001 |
  | minFaceWeight | 0.05 |
  | minVolRatio | 0.01 |
  | minArea | -1 (disabled) |

  A `relaxed {}` sub-dictionary (e.g. maxNonOrtho 75) is used during layer addition. `checkMesh -meshQuality` reports violations against a case's meshQualityDict — [OpenFOAM meshQuality docs v2306](https://doc.openfoam.com/2306/tools/pre-processing/mesh/generation/snappyhexmesh/meshquality); [OpenFOAM.com v2006 guide](https://www.openfoam.com/documentation/guides/v2006/doc/guide-meshing-snappyhexmesh-meshquality.html); [NFDI4ING how-to](https://nfdi4ing.pages.rwth-aachen.de/knowledge-base/how-tos/all_articles/how_to_report_additional_mesh_quality_measures_in_openfoam/)
- **Ansys Fluent practice** (forum and Ansys-staff posts, not official documentation):
  - Orthogonal quality: 0.01 is the bare floor; Ansys staff recommend > 0.1 before solving.
  - Maximum skewness: < 0.9 acceptable, ≤ 0.7-0.8 recommended, 0.95 the upper limit for tets.
  - Skewness matters most for tets, orthogonality for hex/quad. Skewness is not recommended for CutCell meshes.
  - Small first-cell heights chosen for y+ can drive orthogonal quality toward 0 — [Ansys Innovation Space](https://innovationspace.ansys.com/forum/?p=327234); [CFD Online](https://www.cfd-online.com/Forums/ansys-meshing/82122-mesh-quality-criteria-icem-print.html)
- **Verification (GCI)**: Celik, Ghia, Roache & Freitas, "Procedure for Estimation and Reporting of Uncertainty Due to Discretization in CFD Applications", J. Fluids Eng. 130 (2008) 078001.
  - The procedure uses three grids with refinement ratio r > 1.3 and requires iterative convergence first.
  - It reports apparent order p, approximate and extrapolated relative errors, and the fine-grid GCI. Negative p signals oscillatory convergence.
  - It does not cover modeling error. JFE refuses papers that lack systematic discretization-error testing — [ASME JFE policy PDF](https://qa12.asme.org/wwwasmeorg/media/ResourceFiles/Shop/Journals/JFENumAccuracy.pdf); [Scribd excerpt](https://es.scribd.com/document/288804583/Procedure-for-Estimation-and-Reporting-of-Uncertainty-Due-to-Discretization-in-CFD-Applications); [Volupe GCI calculator](https://volupe.com/support/grid-convergence-index-calculator/)
- None of the agent papers surveyed (Foam-Agent, coding-agent assessment, AbaqusAgent, VFEAgent, FeaGPT) report a mesh-convergence study or quantitative mesh-quality metrics — sources in sections 1-2.

### Inferences
- A harness should encode acceptance criteria as machine-checkable gates:
  - OpenFOAM: checkMesh "Mesh OK" with zero failed checks; max non-orthogonality ideally < 65-70°; skewness < 4; the number of cells over the thresholds reported.
  - FEA: scaled Jacobian > 0 (ideally > 0.2-0.3 for tets; this is a common rule of thumb, not sourced here) and aspect-ratio limits.
  - y+ computed from the run (OpenFOAM's yPlus function object) and compared against the turbulence-model wall-treatment target. This is standard practice, not sourced here.
- After these gates, run an automated three-level GCI on the target quantities. Together they would put verification rigor well ahead of every published agent.

### Gaps
- I did not retrieve official Fluent documentation for the thresholds, nor authoritative sources for scaled-Jacobian limits (Verdict, Gmsh `Mesh.QualityType`/SICN) or for y+ targets per turbulence model. Treat those as needing primary sourcing.

## 6. Commercial products adding AI meshing / agents — claims

### Takeaway
By 2026, major vendors have shipped *first* agentic features. They are mostly assistants for meshing failures or for simulation setup, not autonomous meshers. Ansys Mechanical has an exploratory "Mesh Agent". SimScale has a prompt-driven agent inside Onshape. Siemens (which now owns Altair) offers ShapeAI and PhysicsAI. Surrogate-first companies (PhysicsX, Neural Concept) sidestep meshing at inference time. **No vendor publishes success-rate or accuracy metrics for its agents.**

### Cited Findings
- **Ansys 2026 R1 (Synopsys, 11 Mar 2026)**:
  - Introduces the portfolio's "first agentic engineering capabilities".
  - **Mesh Agent** in Ansys Mechanical, for exploratory use only, "helps engineers debug and resolve meshing failures during model pre-processing" and "guides engineers with proven, validated remediation steps".
  - Discovery Validation Agent (in early customer evaluation); GeomAI (generative concept geometry); Engineering Copilot in medini analyze, ModelCenter and Rocky; SimAI Pro (desktop) and SimAI connectors for optiSLang.
  - **No performance metrics are given** — [PR Newswire](https://www.prnewswire.com/news-releases/synopsys-launches-ansys-2026-r1-to-re-engineer-engineering-with-joint-solutions-and-ai-powered-products-302711215.html)
- **Siemens Simcenter 2026.1 (summer 2026)**:
  - Altair products are now folded into Simcenter.
  - Simcenter HyperMesh 2026.1 adds "Retrieve", which uses ShapeAI similarity matching to find geometrically similar parts and load them, speeding model build.
  - Simcenter PhysicsAI (geometric deep learning) claims predictions "up to 1,000x faster" than solvers. **This is a marketing claim.** PhysicsAI Generate is experimental.
  - BatchMesher is still an automatic, but not AI, mesher for large assemblies — [Siemens blog HyperMesh 2026.1](https://blogs.sw.siemens.com/simcenter/whats-new-in-simcenter-hypermesh-2026-1/); [Siemens PhysicsAI 2026.1](https://blogs.sw.siemens.com/simcenter/whats-new-in-simcenter-physicsai-2026-1/); [Siemens news](https://news.siemens.com/en-us/siemens-simcenter-summer-2026/)
- **SimScale Engineering AI Agent for PTC Onshape (15 Sep 2026, IMTS)**:
  - From a single prompt it prepares non-simulation-ready CAD, meshes it, proposes physics and boundary conditions, runs CFD/FEA/thermal/EM, and reports results in plain language.
  - The engineer can intervene or override at any point. Built on the SimScale Agent API ("auditable").
  - **No agent accuracy or success metrics, and no claim of self-verification** — [SimScale press](https://www.simscale.com/press/simscale-launches-engineering-ai-agent-for-onshape/); [ENGtechnica](https://engtechnica.com/simscale-launches-ai-agent-for-simulation-in-ptc-onshape/)
- **Cadence Fidelity Pointwise**: rebranded after Cadence's 2021 acquisition; scripting and multi-solver export. **I found no AI or automesh claim** — [Cadence Fidelity pre-processing](https://www.cadence.com/en_US/home/tools/system-analysis/computational-fluid-dynamics/pre-processing-meshing.html); [engineeringsoftwaretrials](https://engineeringsoftwaretrials.com/what-happened-to-pointwise/)
- **PhysicsX**: the LGM-Aero model is pretrained on tens of thousands of generic-shape CFD/FEA simulations. The CEO frames solving PDEs "on some mesh" as the bottleneck, and the company pitches CAE-level accuracy at the concept stage. That this skips meshing is an inference — [Aviation Week](https://aviationweek.com/aerospace/emerging-technologies/physics-based-ai-promises-accelerate-aerospace-design-optimization); [IEEE Spectrum mirror](https://backiee.wasmer.app/https_spectrum_ieee_org/large-physics-models-design-engineering)
- **Neural Concept**: geometric deep learning predicts 3D performance directly from geometry "in seconds". It launched an AI Design Copilot (Jul 2026) that generates "thousands of manufacturing-ready" geometry options. These are vendor-authored claims — [Neural Concept blog](https://www.neuralconcept.com/post/ai-tools-for-mechanical-engineers); [KI-Syndikat](https://www.ki-syndikat.de/tools/neural-concept/)
- **Startups / funding**:
  - Vinci4D (founded 2023; CEO Hardik Kabaria): a physics foundation model and agentic system pitched as simulation "without meshing". Other coverage describes GPU-accelerated automated meshing, so **the sources conflict**. It raised $46M in Dec 2025 (Xora, Eclipse, Khosla). A reported $250M Series B at a $1.5B valuation (Oct 2026) is **single-source and unverified** — [SiliconANGLE Dec 2025](https://siliconangle.com/2025/12/03/vinci-nabs-46m-speed-chip-simulations-ai/); [VentureBeat](https://venturebeat.com/business/vinci-emerges-from-stealth-to-transform-semiconductor-design-and-simulation)
  - SiliconANGLE (30 Oct 2026) says Vinci4D "automates the so-called meshing phase", and reports that Sebastian Thrun's Dulo raised $130M with an unclear focus, possibly simulation optimization — [SiliconANGLE](https://siliconangle.com/2026/10/30/google-x-co-founder-sebastian-thrun-raises-130m-for-ai-engineering-startup-dulo)
  - Mercor acquired AI-simulation startup Deeptune in Jul 2026 — [Wikipedia: Mercor](https://en.wikipedia.org/wiki/Mercor)
  - A vendor-comparison site claims 50+ AI-CAE startups with $960M+ in VC. **This is an aggregator figure and unverified** — [ThreadMoat](https://threadmoat.com/insights/ai-trends-engineering-simulation)

### Inferences
- Commercial agentic meshing in 2026 is at the "repair assistant" stage (Ansys) or the "black-box end-to-end for simple CAD" stage (SimScale). Neither publishes benchmarks. An open, benchmarked harness built on Gmsh/sHM with explicit quality gates would be differentiated mainly by **measurable verification**.
- Surrogate-first companies (PhysicsX, Neural Concept, Siemens PhysicsAI) still need large meshed simulation datasets for training. Robust automated meshing feeds their data pipelines too, which is a market inference.

### Gaps
- I found no 2026 AI-meshing announcements from Hexagon (MSC Apex/Marc), Altair (pre-merger standalone), Cadence Fidelity, Beta CAE (ANSA), Luminary Cloud, or Flexcompute. I also did not check the Fluent Meshing "Watertight Geometry" AI features or STAR-CCM+ AI features. These need targeted vendor-page searches.

### Cross-cutting inferences: harness-design implications (synthesized from all sections)
- **Architecture:** use a single strong coding agent (Opus-class) in a generic harness with bash, file edit and run tools. Multi-agent decomposition is unnecessary at 2026 model strength [2609.03718].
- **Investment priorities:** (1) an execution-feedback loop that can run many times (25 reviewer loops vs 10 moved Opus 4.6 from 85% to 100% on FoamBench Basic) [Foam-Agent README]; (2) tutorial and case retrieval to copy from [2609.03718, 2602.11689, ChatCFD template-DB ablation].
- **Mesh-quality gates:** convert checkMesh and Gmsh quality output into structured feedback, with thresholds of non-ortho 65/70°, skewness 4, aspect ratio 1000 and minDeterminant 0.001. Published agents do not do this.
- **Silent failures:** add geometric-fidelity assertions and image/VLM checks. These catch the failure modes no log reports: a missing obstacle, a reused tutorial mesh, a wrong cross-section [2602.11689, 2606.00138].
- **Verification and evaluation:** automate GCI convergence (Celik 2008). Evaluate on correctness-gated metrics (NMSE vs reference, the FoamBench "Success Rate"), not executability.
