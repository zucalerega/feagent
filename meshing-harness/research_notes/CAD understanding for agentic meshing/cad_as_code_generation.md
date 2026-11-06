# CAD as Code: How LLMs/VLMs Generate, Edit, and Reason About CAD Through Text/Code (state of the art through Oct 2026)

Notes compiled 2026-10-30. Verification legend: items marked **[verified]** were checked against the primary source (arXiv abs/HTML or GitHub) during this session. Items marked **[prior knowledge, not re-verified]** come from background knowledge of the literature; the arXiv IDs and descriptions are believed correct, but their numbers were not re-checked this session and should be confirmed before they are quoted.

---

## 1. Which text/code CAD representations are used, and which do models handle best?

### Takeaway
Since 2025 the field has largely converged on **CadQuery (Python on OpenCASCADE)** as the target representation, for generation (text→CAD), reconstruction (point cloud/mesh/image→CAD) and benchmarks alike. Older work used **DeepCAD-style sketch-and-extrude command sequences** (JSON/token sequences). Head-to-head evidence shows command sequences produce *more invalid outputs* than CadQuery for general LLMs, because Python lets the models reuse what they learned from code pretraining. OpenSCAD, FreeCAD Python, build123d and Onshape FeatureScript are much less studied. No benchmark found compares CadQuery against build123d or FeatureScript, and **no work found feeds raw STEP text to an LLM as a primary representation**.

### Cited Findings
- **Command sequences vs CadQuery (direct comparison):** Text2CAD-Bench (arXiv 2605.18430, 18 May 2026) uses CadQuery as its target and argues that Python CadQuery lets models "exploit pretrained code generation capabilities". On L1–L2, DeepCAD-style command sequences gave higher invalidity rate (IR) and worse CD/IoU for nearly all models. Some CD/IoU numbers improved under command sequences only because the higher IR left a smaller, easier set of executed samples (survivorship bias). OpenSCAD, build123d and JSON were not tested. **[verified]** — [Text2CAD-Bench](https://arxiv.org/html/2605.18430v1)
- **Domain-specific command-sequence models fall far behind general LLMs writing CadQuery on human-curated prompts.** Text2CAD-Bench, L1 geometric prompts (CD↓ / IR↓): GPT-5.2 44.31 / 11.1%; Claude-4.5-Sonnet 52.62 / 20.3%; DeepSeek-V3.2 53.15 / 13.3%. The command-sequence models Text2CAD and CADFusion score 219.57 / 11.0% and 229.62 / 60.0%, and the fine-tuned Text2CADQuery scores 227.46 / 44.0%. Specialists "execute but are geometrically wrong" (low IR, near-zero IoU). **[verified]** — [Text2CAD-Bench](https://arxiv.org/html/2605.18430v1)
- **CadQuery as Python code:** CAD-Recode (Rukhovich et al., arXiv 2412.14042, Dec 2024) represents sketch-extrude sequences as CadQuery Python and decodes with Qwen2-1.5B. The authors stress that the code "is interpretable by off-the-shelf LLMs", which enables editing and CAD question answering. **[verified]** — [CAD-Recode](https://arxiv.org/html/2412.14042v2)
- **CADCodeVerify chose CadQuery over OpenSCAD** for its CADPrompt benchmark. **[verified]** — [CADCodeVerify](https://arxiv.org/html/2410.05340v2)
- **Constrained CadQuery DSL for stepwise agents:** CADENA (arXiv 2608.00799, 1 Aug 2026) uses a CadQuery-based DSL where "each line consumes and rebinds one variable `r`", so any prefix of the program is valid and executable. This design suits incremental build-and-check loops. **[verified]** — [CADENA](https://arxiv.org/html/2608.00799)
- **CadQuery parametric generators as training data:** CADEvolve (arXiv 2602.16317, 18 Feb 2026) grows programs by VLM-guided evolution from simple primitives into about 8k complex parts, then about 1.3M scripts after augmentation, "exercising the full CadQuery operation set". **[verified, abstract only]** — [CADEvolve](https://arxiv.org/abs/2602.16317)
- **Industrial CadQuery benchmark:** BenchCAD (arXiv 2605.10865, May 2026) has 17,900 execution-verified CadQuery parts across 106 industrial families. About 49% of the families are tied to ISO/DIN/EN/ASME/IEC standards. **[verified]** — [BenchCAD](https://arxiv.org/html/2605.10865v1)
- **FreeCAD Python API as an agent action space:** CAD-Assistant (Mallis et al., arXiv 2412.13810, Dec 2024, rev. Aug 2025) has a VLLM planner emit FreeCAD Python actions that run iteratively in an interpreter. **[verified]** — [CAD-Assistant](https://arxiv.org/abs/2412.13810)
- **Query2CAD** (arXiv May 2024) generates executable CAD macros (FreeCAD) with an LLM acting as both generator and self-refiner, without supervised data. **[verified via search snippet]** — [search result, Query2CAD](https://arxiv.org/abs/2406.00144)
- **OpenSCAD:** CadQueryEval is a CadQuery port of the CadEval benchmark, which was originally OpenSCAD. **[verified]** — [cadqueryeval](https://github.com/danwahl/cadqueryeval)
- **Older command-sequence and JSON systems [prior knowledge, not re-verified]:**
  - Text2CAD (Khan et al., NeurIPS 2024; arXiv 2409.17106): text to DeepCAD sequences, with about 170k models and multi-level prompts.
  - CAD-Llama (CVPR 2025; arXiv 2505.04481): a hierarchical "SPCC" code-like representation of command sequences, plus adaptive pretraining and instruction tuning of Llama.
  - FlexCAD (ICLR 2025; arXiv 2411.05823): hierarchical textual CAD representation with masked-field LLM fine-tuning for controllable generation.
  - CAD-MLLM (arXiv 2411.04954): command sequences conditioned on text, image or point cloud, with the Omni-CAD dataset (~453K) **[Omni-CAD size verified via search snippet]**.
  - CAD-GPT (AAAI 2025; arXiv 2412.19663): spatial-reasoning-enhanced MLLM over sequences.
  - OpenECAD (arXiv 2406.09913): VLMs that output Python-like CAD code.
  - Text-to-CadQuery (arXiv 2505.06507): re-annotates Text2CAD into about 170k text→CadQuery pairs **[verified via search snippet]**.
  - BlenderLLM (arXiv 2412.14203): Blender Python with the CADBench benchmark.
  - Img2CAD (arXiv 2408.01437): GPT-4V for structure plus a transformer for parameters.
- **B-rep/topology is not handled well by LLM-written code.** In FutureCAD (arXiv 2603.11831, v2 26 Apr 2026), an "LLM-Query" baseline that writes CadQuery selector code to pick faces and edges reaches only **21.51% F1**. The authors instead delegate face/edge selection (e.g., for fillets) to a separate B-rep GNN (BRepGround, UV-Net features plus a face-adjacency GNN). The LLM never sees STEP/B-rep as text. **[verified]** — [FutureCAD](https://arxiv.org/html/2603.11831v2)

### Inferences
- **Why CadQuery wins:**
  - It is Python, so it is well represented in pretraining.
  - It is executable against a real B-rep kernel (OCCT), so errors are machine-checkable.
  - It supports fillets, revolves, lofts and sweeps beyond sketch-extrude.
  - It exports STEP directly, which matters for FEA/CFD meshing (gmsh/OCC can consume the B-rep).
- **For an FEA/CFD meshing agent:** CadQuery (or a constrained CadQuery DSL like CADENA's) is the best-supported "lingua franca" for representing a part as code. Face and edge references (for boundary conditions, fillets, mesh refinement zones) are the weak point and may need non-LLM geometric grounding. Examples are B-rep graph models, or deterministic selectors computed from kernel queries.
- **build123d** is API-similar to CadQuery (both use OCCT) and likely transfers well, but no benchmark evidence was found for it.

### Gaps
- No study found compares CadQuery, build123d, OpenSCAD, FreeCAD, FeatureScript and STEP text on the same benchmark with the same models.
- No paper found evaluates LLMs reading raw ISO-10303-21 STEP text. Whether frontier models can parse STEP entity graphs reliably remains unverified.
- Onshape FeatureScript: no 2025–2026 LLM benchmark found.

---

## 2. Key papers/systems, benchmarks and results

### Takeaway
For **reconstruction** (point cloud, mesh or images → CadQuery), specialized small VLMs (1.5–2B Qwen) trained on large procedural CadQuery corpora, then refined with online RL that uses geometric rewards, dominate:
- CAD-Recode (Dec 2024)
- cadrille (ICLR 2026 Oral)
- CADEvolve, CADReasoner, CADFit (2026)
- CADENA (Aug 2026)

On DeepCAD/Fusion360 they reach about 90%+ IoU, median CD about 0.15–0.17×10⁻³, and near-0% invalidity. They are much weaker on real or complex parts (CC3D, MCB, industrial families). For **text→CAD**, general frontier LLMs writing CadQuery now beat older specialist models. Accuracy still collapses on sweeps, lofts and shells, and on real-world parts.

### Cited Findings

**Benchmarks and metrics**
- **Standard metrics:**
  - Chamfer distance (CD, usually median ×10³ on normalized shapes)
  - Volumetric IoU (%)
  - Invalidity rate (IR, % of programs that fail to execute or yield no valid solid)
  - Newer: Hausdorff distance, Feature-F1, essential-operation recall, and VLM-judge scores

  Sources: [cadrille](https://arxiv.org/html/2505.22914v3); [BenchCAD](https://arxiv.org/html/2605.10865v1)
- **Reconstruction datasets:** DeepCAD, Fusion360 Gallery, CC3D (real scans), MCB, and Omni-CAD. — [cadrille](https://arxiv.org/html/2505.22914v3); [CADReasoner](https://arxiv.org/html/2603.29847v1)
- **CADPrompt** (CADCodeVerify, arXiv 2410.05340, Oct 2024): 200 natural-language prompts with expert-annotated CadQuery code. Metrics are IoGT, point-cloud distance, Hausdorff distance and compile rate. **[verified]** — [CADCodeVerify](https://arxiv.org/html/2410.05340v2)
- **Text2CAD-Bench** (May 2026): 600 human-curated examples (1,200 prompts) at levels L1–L4. L3 covers sweeps, lofts and shells. L4 covers real-world parts (~40% industrial) and is VLM-judged by GLM-4.6V. **[verified]** — [Text2CAD-Bench](https://arxiv.org/html/2605.18430v1)
- **BenchCAD** (May 2026): tasks are img2cq (4 views → code), code edit (748 pairs), and image/code QA (2,400 numeric questions). **[verified]** — [BenchCAD](https://arxiv.org/html/2605.10865v1)
- **Other 2026 benchmarks** found but not read in detail: CADBench (arXiv 2605.10873), RealCADBench (arXiv 2609.03773, industrial design intents), CADWorld (arXiv 2609.16251, computer-use benchmark for long-horizon CAD in GUI tools), and textcad-bench (GitHub, 200 prompts, 7 models, CD in the 17–27 range). — [CADBench](https://arxiv.org/pdf/2605.10873); [RealCADBench](https://arxiv.org/pdf/2609.03773); [CADWorld](https://arxiv.org/pdf/2609.16251); [textcad-bench](https://github.com/prashantkul366/textcad-bench)

**Reconstruction systems (verified numbers)**
- **CAD-Recode** (Dec 2024; v2 Mar 2025). Point cloud → CadQuery with a Qwen2-1.5B decoder and a 256-point projector, trained on a procedural set (1M per the abstract, 11M per the tables, an internal inconsistency). Results (mean CD / median CD / IoU / IR):

  | Dataset | Mean CD | Median CD | IoU | IR |
  |---|---|---|---|---|
  | DeepCAD | 0.30 | 0.16 | 92.0 | 0.4% |
  | Fusion360 | 0.35 | 0.15 | 87.8 | 0.5% |
  | CC3D (real scans) | 0.76 | 0.31 | 74.2 | 0.3% |

  The prior best, CAD-SIGNet, scored 77.6 IoU on DeepCAD and 42.6 on CC3D. Supports only sketch-extrude (no revolve, fillet or B-spline). **[verified]** — [CAD-Recode](https://arxiv.org/html/2412.14042v2)
- **cadrille** (Kolodiazhnyi et al., arXiv 2505.22914, May 2025; v3 Feb 2026; **ICLR 2026 Oral**). Multimodal (point cloud, images, text) Qwen2-VL with SFT on procedural data, then online RL ("Dr. CPPO", a GRPO variant). Multi-view image results after RL (CD / IoU / IR):

  | Dataset | CD | IoU | IR |
  |---|---|---|---|
  | DeepCAD | 0.17 | 92.2 | 0.0% |
  | Fusion360 | 0.17 | 84.6 | 0.0% |
  | CC3D | 0.57 | 65.0 | 0.1% |

  Text input on DeepCAD: cadrille 0.21 / 81.1 / 1.4%, versus Text2CAD 0.37 / 71.5 / 3.7%, CAD-Coder 0.33 CD / 5.3% IR, and Text-to-CadQuery 0.22 CD / 1.3% IR. **[verified]** — [cadrille](https://arxiv.org/html/2505.22914v3)
- **CADEvolve** (Feb 2026): a VLM fine-tuned on evolved CadQuery data reports state of the art on image-to-CAD for DeepCAD, Fusion 360 and MCB. Abstract gives no numbers. CADENA cites CADEvolve's DeepCAD GMS as 95.3. — [CADEvolve](https://arxiv.org/abs/2602.16317); [CADENA](https://arxiv.org/html/2608.00799)
- **CADReasoner** (arXiv 2603.29847, Feb 2026). Iterative self-editing with Qwen2-VL-2B. Image-only, greedy, at t=5 (CD / IoU / IR):

  | Dataset | CD | IoU | IR |
  |---|---|---|---|
  | DeepCAD | 0.16 | 90.0 | 0.5% |
  | Fusion360 | 0.15 | 83.1 | 0.7% |
  | MCB | 1.67 | 42.7 | 4.0% |

  **[verified]** — [CADReasoner](https://arxiv.org/html/2603.29847v1)
- **CADFit** (arXiv 2605.01171, ICML 2026): an optimization-based mesh→CAD program fitter. It fits parametric operations incrementally and validates candidates in a CAD kernel. **[verified via search snippet]** — [CADFit](https://arxiv.org/html/2605.01171); [ICML 2026 poster](https://icml.cc/virtual/2026/poster/61824)
- **CADENA** (Aug 2026). Mesh → stepwise CadQuery DSL, with SFT on 18M stepwise samples and then GRPO-style RL with a volumetric IoU reward. Results (GMS):

  | Dataset | CADENA-RL | Best baseline |
  |---|---|---|
  | DeepCAD | 97.0 | CADEvolve 95.3 (cadrille 94.8, CAD-Recode 92.9) |
  | Fusion360 | 93.3 | cadrille 86.8 |
  | MCB | 73.7 (IR 0.7%) | CADFit 60.1 |
  | CADENA-Bench | 67.0 | CADFit 54.8 (cadrille 49.8) |

  On MCB, CAD-Recode's IR is 21.5% and CADReasoner's is 31.7%. **[verified]** — [CADENA](https://arxiv.org/html/2608.00799)
- **FutureCAD** (Mar/Apr 2026). Text → CadQuery with Qwen2.5-7B plus a B-rep grounding model, SFT then GSPO RL with a CD reward. Results: advanced subset median CD 31.12, IR 1.01%; standard subset median CD 14.68, IR 0.91%. SFT alone had 42.67% IR on the advanced subset. **[verified]** — [FutureCAD](https://arxiv.org/html/2603.11831v2)

**Text→CAD systems**
- **CADFusion** (Wang et al., Microsoft, arXiv 2501.19054, **ICML 2025**): alternates a sequential-learning stage with a visual-feedback stage that rewards sequences rendering to visually preferred objects. **[verified, abstract only; numbers not retrieved]** — [CADFusion](https://arxiv.org/abs/2501.19054)
- **CAD-Coder** (Guan et al., **NeurIPS 2025**): text → CadQuery with chain-of-thought and a geometric (CD) reward via GRPO. Its DeepCAD numbers as reported in cadrille's table: CD 0.33, IR 5.3%. — [NeurIPS 2025](https://proceedings.nips.cc/paper_files/paper/2025/hash/564a224e88f2490f3c1deaae877d37e3-Abstract-Conference.html); [cadrille](https://arxiv.org/html/2505.22914v3)
- A separate, same-named **CAD-Coder** (Doris et al., arXiv 2505.14646, image → CadQuery, GenCAD-Code dataset). **[prior knowledge, not re-verified]**
- **CAD-Llama** reports competitive command-type F1 but weaker geometric fidelity, according to FutureCAD's comparison (one group's evaluation). — [FutureCAD](https://arxiv.org/pdf/2603.11831)

### Inferences
- **Reconstruction on DeepCAD/Fusion360 is close to saturated.** IoU is about 90–97 GMS, and RL brings IR to ~0.
- **The real frontier is complex or real-world parts:**
  - MCB IoU for CADReasoner and cadrille-SFT is around 40–50%.
  - CC3D IoU is 60–74%.
  - CADENA-Bench GMS is about 67.
- **For a meshing agent:** a part from DeepCAD-like distributions (prismatic, sketch-extrude) can likely be recovered as CadQuery with high fidelity. Industrial parts with threads, gears, springs, sweeps and fillets cannot yet.

### Gaps
- Numbers not retrieved this session: CADFusion, Text2CAD, FlexCAD, CAD-MLLM, CAD-Llama, BlenderLLM, Img2CAD, Query2CAD and CAD-Assistant.
- Venue acceptance for CAD-Recode (believed ICCV 2025), CADCodeVerify (believed ICLR 2025) and CAD-Assistant (believed ICCV 2025) was not confirmed on the arXiv pages.
- "GMS" in CADENA is that paper's geometric metric score. Its exact definition was not extracted.

---

## 3. Which verification and feedback loops help, and by how much?

### Takeaway
The biggest, best-quantified gains come from **executing code and using geometric comparison against a target**:
- online RL with an IoU/CD reward
- best-of-N selection by Chamfer distance
- iterative render-overlay self-editing

These push invalidity from 5–20% to under 1% and add several IoU points. **VLM-only visual self-verification** (rendering and asking a VLM yes/no questions) gives smaller gains, around 5–7%. Its own answers are only about 65–68% reliable, and an oracle geometric-solver feedback beats it. Execution-error retries are standard, but they are rarely ablated.

### Cited Findings
- **Online RL with a geometric reward (cadrille).** The reward is 10 × IoU, with a −10 penalty for invalid code, plus hard-example mining.
  - Point clouds, SFT → Dr. CPPO RL: IoU on CC3D rises from 61.8 to 67.9, and IR falls from 5.9% to 0.2%. Fusion360 IoU rises from 79.8 to 85.0, with IR from 2.8% to 0.2%.
  - Images: DeepCAD IoU rises from 86.1 to 92.2, with IR from 1.5% to 0.0%.
  - Offline DPO gives only small gains (e.g., CC3D image IR 7.7% → 3.9%) and is "limited by the best sample it generates".

  **[verified]** — [cadrille](https://arxiv.org/html/2505.22914v3)
- **Test-time sampling with geometric selection (CAD-Recode).** Ten candidates are generated and the one with the lowest CD to the input is kept. Without sampling, IR rises to 4.9% (DeepCAD), 8.7% (Fusion360) and 16.8% (CC3D), against 0.4/0.5/0.3% with it. Mean CD is 0.75 vs 0.30 on DeepCAD and 3.05 vs 0.76 on CC3D. **[verified]** — [CAD-Recode](https://arxiv.org/html/2412.14042v2)
- **Iterative render-and-compare self-editing (CADReasoner).** Inputs are eight views of the target overlaid with the current prediction, point-cloud offsets, and the previous code. From t=1 to t=5 on MCB (image), IR falls from 21.9% to 4.0% and CD from 3.5 to 1.67. On point clouds, MCB CD falls from 1.03 to 0.52, IoU rises from 41.3 to 49.1, and IR falls from 16.2% to 3.1%. Gains on DeepCAD are small (IoU 88.7 → 90.0). The best-so-far program is kept by CD. **[verified]** — [CADReasoner](https://arxiv.org/html/2603.29847v1)
- **Stepwise residual-guided building (CADENA):**
  - a red/green target-vs-build overlay
  - text "hint points" localizing discrepancies
  - prefix selection by IoU, so later steps cannot make the output worse
  - sampling 12 candidates per step, which lifts MCB SFT GMS from 63.3 to 70.0

  **[verified]** — [CADENA](https://arxiv.org/html/2608.00799)
- **VLM visual self-verification (CADCodeVerify, CADPrompt, GPT-4 few-shot).** CADCodeVerify reaches point-cloud distance 0.127 vs 0.155 for generated code, 0.137 for 3D-Premise, and 0.103 for an oracle geometric solver that uses ground truth. Compile rate is 96.5% vs 96.0%. The abstract reports a 7.30% point-cloud-distance reduction and 5.0% success-rate improvement over prior work. Further details:
  - Gains saturate after 2 refinement rounds.
  - For CodeLlama, Refine-2 sometimes *degraded* results.
  - The VLM's yes/no verification answers were correct only 64.6% (Refine 1) and 68.2% (Refine 2) of the time.

  **[verified]** — [CADCodeVerify](https://arxiv.org/html/2410.05340v2)
- **Execution-error retries** (up to 3 with error messages) are used in Text2CAD-Bench without ablation. Even so, L3 IR stays at 68–92% for most frontier LLMs. **[verified]** — [Text2CAD-Bench](https://arxiv.org/html/2605.18430v1)
- **Visual-feedback preference training (CADFusion, ICML 2025)** improves results qualitatively and quantitatively per its abstract. Numbers were not retrieved. — [CADFusion](https://arxiv.org/abs/2501.19054)
- **Tool-augmented agent (CAD-Assistant):** a FreeCAD interpreter, renderer, 2D cross-section generator and sketch parameterizer. The agent "checks how each generated CAD command changes the geometry". It outperforms VLLM baselines and supervised task-specific methods per its abstract (numbers not retrieved). — [CAD-Assistant](https://arxiv.org/abs/2412.13810)
- **Kernel-validated optimization (CADFit)** fits operations incrementally and validates them in the CAD kernel, which gives fewer invalid programs on complex designs. — [CADFit](https://arxiv.org/html/2605.01171)

### Inferences
- **A ranking of feedback signals for an agent:**
  1. Execute and check validity (a hard filter).
  2. Quantitative geometric comparison against ground truth (IoU, CD, volume, bounding box), used for best-of-N selection and accept/reject decisions.
  3. Rendered overlays or residuals shown to the VLM to localize errors.
  4. Pure VLM yes/no visual judgment, the least reliable.
- **For a meshing agent that already has a STEP file:** it holds a ground-truth target, so loops of types 1–2 are directly available. Examples are volume, surface area, bounding box, face count and Hausdorff distance computed via OCC/gmsh.

### Gaps
- No controlled study was found isolating execution-error feedback from visual feedback for frontier LLMs on the same benchmark.
- Little evidence was found on how many self-repair rounds frontier models need on complex parts.

---

## 4. How well do frontier models (Claude, GPT, Gemini, open models) do?

### Takeaway
Frontier general models are strong at **code-level tasks**:
- editing existing CadQuery (~0.80–0.87 accuracy)
- answering numeric questions from code (~0.80–0.84)
- simple text→CadQuery tasks, where the newest models (Sep 2026) saturate a small 25-task benchmark

They are weak at **vision→code** and at advanced features (sweep, loft, shell). There, small fine-tuned and RL-trained VLMs beat them by about 2×.

### Cited Findings
- **BenchCAD (May 2026), img2cq total score:**

  | Model | Score |
  |---|---|
  | Qwen3-VL-2B RL (IID) | 0.768 |
  | Qwen3-VL-2B RL (OOD) | 0.723 |
  | CADEvolve v3 | 0.601 |
  | Gemini 3.1 Pro (thinking) | 0.397 |
  | Claude Sonnet 4.6 (thinking-high) | 0.385 |
  | Claude Opus 4.7 | 0.383 |
  | cadrille-RL | 0.338 |
  | GPT-4o | 0.227 |
  | GPT-5.3 | 0.216 |
  | o3 | 0.108 |

  The paper's text gives Gemini 3.1 Pro (thinking) as 0.318, which conflicts with the table's 0.397. **[verified]** — [BenchCAD](https://arxiv.org/html/2605.10865v1)
- **BenchCAD QA:** Code QA reaches up to about 0.84 (Gemini 3.1 Pro 0.838, Opus 4.7 thinking 0.829, GPT-5.3 0.823; blank-code baseline 0.223). Vision QA reaches only about 0.59 (Gemini 3.1 Pro 0.587, Opus 4.7 0.53; blank-image baseline 0.375). **Code access dramatically improves understanding.** **[verified]** — [BenchCAD](https://arxiv.org/html/2605.10865v1)
- **BenchCAD Edit:** GPT-5.3 (thinking) 0.865, Opus 4.7 (thinking) 0.853, Gemini 3.1 Pro (thinking) 0.837, GPT-4o 0.615, gpt-oss-120b 0.561. Additional findings:
  - Edits specified by a *target render* instead of text drop every model to near zero.
  - About **64% of nominally successful edits silently corrupt unrelated features**.

  **[verified]** — [BenchCAD](https://arxiv.org/html/2605.10865v1)
- **Text2CAD-Bench L3** (sweeps/lofts/shells), IR: GPT-5.2 68%, Claude-4.5-Sonnet 70%, Qwen3-max 92%, Gemini-3-Flash 12.8% (an outlier). On L4 real-world parts, IR ranges from 17% (Gemini-3-Flash) to 81% (MiniMax-M2.1), with VLM-judge Overall scores of about 6.9–8.3/10. Executability, geometric similarity and feature quality "do not track each other". **[verified]** — [Text2CAD-Bench](https://arxiv.org/html/2605.18430v1)
- **CadQueryEval** (Inspect AI; 25 text→CadQuery tasks; pass requires watertightness, a single component, bounding box within 1 mm and volume within 2%).
  - Current leaderboard (repo archived Sep 2026 as saturated): claude-opus-5.5, gpt-5.6-sol-pro, gpt-6-astra and gpt-6-sol at **100%**; gemini-3.8-flash and gpt-6-luna at 96%; gemini-3.1-pro-preview 88%; claude-opus-5 80%; gpt-5.5 76%; claude-opus-4.6 68%; claude-opus-4.7 60%; claude-sonnet-4.5 48%; gpt-4o 32%.
  - **Conflict:** a search-engine summary of an earlier (May 2026) leaderboard snapshot gave Gemini 3.1 Pro 0.80, GPT-5.5 0.64 and Opus 4.6 0.60, likely an earlier run.
  - Single epoch and only 25 tasks give wide confidence intervals (standard error ~0.1).

  **[verified]** — [cadqueryeval](https://github.com/danwahl/cadqueryeval)
- **CADENA** reports that the strongest vendor-reported frontier entry on BenchCAD (voxel IoU), "GPT-5.6 Sol", scores 0.706, and that re-graded frontier entries top out at 0.355. CADENA-RL scores 0.910 with mesh input, which the authors flag as not like-for-like. **[verified, secondhand within CADENA]** — [CADENA](https://arxiv.org/html/2608.00799)
- **Earlier generation (CADPrompt, 2024):** GPT-4 compile rate was 92–96%, Gemini 1.5 85%, CodeLlama-70B 64.5–73.5%. — [CADCodeVerify](https://arxiv.org/html/2410.05340v2)
- **GPT-4o as a CAD-code reader:** when CAD-Recode's reconstructed code is given to GPT-4o, it scores 76.5% on SGP-Bench CAD QA, vs 63.2% via CAD-SIGNet output and 42.3% for PointLLM directly on points. **[verified]** — [CAD-Recode](https://arxiv.org/html/2412.14042v2)

### Inferences
- An agent that has (or can obtain) a CadQuery representation of a part will understand it far better through the code than through renders (≈0.84 vs ≈0.59 QA on BenchCAD).
- Frontier models are reliable for simple, well-specified parametric parts. Late-2026 models pass all 25 CadQueryEval tasks.
- Frontier models are unreliable for vision-only reverse engineering and advanced features.
- Edits must be geometrically verified, given the 64% silent-corruption figure.

### Gaps
- No independent benchmark from late 2026 (Sep–Oct) was found evaluating the newest models (Claude Opus 5.5, GPT-6) on BenchCAD or Text2CAD-Bench. Only CadQueryEval covers them, and it is saturated.
- BenchCAD's official leaderboard and its "vendor-reported" entries were not inspected directly.

---

## 5. Can these approaches recover code from an existing STEP or mesh (reverse engineering)?

### Takeaway
Yes, for **meshes and point clouds**, with high fidelity on prismatic sketch-extrude parts and moderate fidelity on complex ones. CadQuery is the target in CAD-Recode, cadrille, CADReasoner, CADFit and CADENA. All of these work from tessellated or sampled geometry, not from STEP B-rep topology, so a STEP file would first be tessellated or sampled. **No work found converts STEP B-rep directly into editable feature code with an LLM.**

### Cited Findings
- **CAD-Recode:** point cloud (256 points) → CadQuery. On real CC3D scans: IoU 74.2%, median CD 0.31, IR 0.3%. Only sketch-extrude is supported; revolutions, fillets and B-splines yield invalid outputs. Features thinner than about 1 quantization unit can be predicted as zero-height. **[verified]** — [CAD-Recode](https://arxiv.org/html/2412.14042v2)
- **cadrille:** point clouds or multi-view images → CadQuery. It "always predicts a geometrically relevant shape, but still might miss details", especially for complex, granular surfaces. **[verified]** — [cadrille](https://arxiv.org/html/2505.22914v3)
- **CADENA:** triangle mesh → stepwise CadQuery DSL. GMS is 97.0 on DeepCAD, 93.3 on Fusion360, 73.7 on MCB and 67.0 on CADENA-Bench. The authors caution that "exported CadQuery programs may not transfer cleanly to industrial CAD". **[verified]** — [CADENA](https://arxiv.org/html/2608.00799)
- **CADReasoner:** scan → CadQuery with iterative editing, using a simulated-scan track. A model trained on clean data and tested on scans drops on MCB from CD 1.43 to 4.42 and IoU from 41.0 to 17.8. **[verified]** — [CADReasoner](https://arxiv.org/html/2603.29847v1)
- **CADFit:** mesh → CAD program via hybrid optimization validated in the kernel. However, 25.6–29.7% of external parts are left unconverged, per CADENA's evaluation. — [CADFit](https://arxiv.org/html/2605.01171); [CADENA](https://arxiv.org/html/2608.00799)
- **Downstream use:** recovered code is LLM-readable. GPT-4o answers CAD questions at 76.5% from CAD-Recode code, and refactors code to expose parameters (shown only qualitatively). **[verified]** — [CAD-Recode](https://arxiv.org/html/2412.14042v2)

### Inferences
- **For a meshing agent with a STEP input:** the lossless source of truth is the B-rep itself. Reverse-engineering to code is approximate (IoU 65–97%, depending on complexity) and should not replace the STEP for meshing.
- **Its value is semantic.** It yields a compact, LLM-readable feature description (e.g., "a plate with 4 through-holes and a fillet"), which supports:
  - choosing defeaturing
  - identifying symmetry
  - picking boundary-condition faces
  - setting refinement zones
- A safer hybrid is to extract a feature/topology summary deterministically from the B-rep (face types, areas, adjacency, holes, fillets via OCC). The LLM can reason over that summary, with optional code reconstruction cross-checked by IoU against the original.

### Gaps
- No published LLM method was found for direct STEP B-rep → feature-tree code that preserves exact topology.
- Real-world industrial parts (assemblies, threads, gears) remain poorly handled. The weakest families are gears/bearings and springs/fasteners per CADENA.

---

## 6. Failure modes

### Takeaway
The dominant failures are:
- execution failures on advanced operations (sweep, loft, shell)
- outputs that execute but are geometrically wrong
- spatial and coordinate-frame errors (wrong workplane, misalignment)
- missed fine features (threads, chamfers, fillets, counted features such as gear teeth)
- silent collateral damage during edits
- the inability of LLMs to reference B-rep faces and edges reliably (topology selection)

### Cited Findings
- **Wrong workplane / coordinate frame:** "some outputs are right in shape but built in the wrong workplane". Models replace sweeps, lofts and twist-extrudes with simpler sketch-and-extrude, miss threads and chamfers, and misread industrial parameters such as spring end-coil pitch. — [BenchCAD](https://arxiv.org/html/2605.10865v1)
- **Spatial misalignment** is attributed to LLMs' limited 3D spatial reasoning. Other failures: omitted fillets, and geometry contradicting the textual spatial arrangement. — [FutureCAD](https://arxiv.org/html/2603.11831v2)
- **Early wrong-plane errors** are "worked around rather than undone". Countable features such as gear teeth are approximated rather than counted, geometry hidden from all 8 views is invisible to the policy, and out-of-DSL features are approximated by stacked extrusions. — [CADENA](https://arxiv.org/html/2608.00799)
- **Topology naming and selection:** LLM-written CadQuery selector code for choosing faces and edges reaches only 21.51% F1. A learned B-rep grounding model reaches 50.23% F1 / 63.07% mAP. Replacing grounding with nearest-point selection raised IR to 8.29%. Persistent naming across edits is not addressed. — [FutureCAD](https://arxiv.org/html/2603.11831v2)
- **Edit side effects:** about 64% of nominally successful edits silently corrupt unrelated features. — [BenchCAD](https://arxiv.org/html/2605.10865v1)
- **Compile errors** such as CadQuery's "Cannot find a solid on the stack". In a 50-example analysis: Structural Configuration Error 48% and Logical Error 18%, along with spatial precision errors. Point-cloud and Hausdorff metrics can miss structural problems such as small gaps in desk legs. — [CADCodeVerify](https://arxiv.org/html/2410.05340v2)
- **Survivorship bias in metrics:** CD and IoU are computed only on executed samples, so high-IR models can look better than they are. — [Text2CAD-Bench](https://arxiv.org/html/2605.18430v1)
- **Kernel robustness and thin features:** OpenCASCADE is less robust than commercial kernels. Thin features can collapse to zero-height. — [FutureCAD](https://arxiv.org/html/2603.11831v2); [CAD-Recode](https://arxiv.org/html/2412.14042v2)
- **Domain gap:**
  - Synthetic → real scans: CC3D IoU is about 60% after SFT, because of noise, smoothed edges and missing parts.
  - Mixing datasets with mismatched operations (symmetric extrude, extruded cut) degrades results.

  — [cadrille](https://arxiv.org/html/2505.22914v3)

### Inferences
- **Meshing-agent implications:**
  - Never trust LLM-generated face selectors for boundary conditions. Resolve faces deterministically, by geometric predicates on the kernel (normal, centroid, area, type), and verify them.
  - Always re-check geometry after edits, such as defeaturing or splitting for structured meshing. Check volume, bounding box, face count and a Hausdorff distance against the original.
  - Normalize and state the coordinate frame and units explicitly in prompts.
  - Small features matter for FEA (fillets at stress concentrations) and CFD (thin gaps). These are exactly what models miss, so they need explicit detection from the B-rep.

### Gaps
- No quantitative study was found of coordinate-frame/unit confusion rates specifically.
- No study was found of LLM handling of the topological naming problem across parametric edits (e.g., FreeCAD TNP) beyond FutureCAD's selection results.
