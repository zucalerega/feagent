"""Every knob in one place: models, token caps, prices, tasks, loop limits."""

import os

# --------------------------------------------------------------------------- models
# Switch any role to another model with one line.
MODELS = {
    "generator": "claude-sonnet-5-5",   # the FEA agent (many calls per run)
    "reflector": "claude-sonnet-5-5",   # single-call reflector (REFLECTOR_MODE = "single_call")
    "diagnoser": "claude-sonnet-5-5",   # agentic diagnoser; switch to "claude-opus-5-5" if Sonnet falls short
    "curator": "claude-sonnet-5-5",     # bookkeeping: lesson -> ADD or duplicate-of
    # --deep (ace/deep_teacher.py, ace/self_reflect.py, ace/organizer.py)
    "teacher": "claude-opus-5-5",       # reviews every run on a task: whole meshes, error maps, setups
    "reflect": "claude-sonnet-5-5",     # each agent looks at its own final results and writes notes
    "organizer": "claude-sonnet-5-5",   # merges and rewrites an agent's (or the shared) notes
    # the single-agent baseline (ace/single_loop.py): ONE big agent doing every task in one conversation
    "big_generator": "claude-sonnet-5-5",   # the swarm agents' model
}

# max_tokens caps one response (thinking counts toward it). You pay for what is used, not the cap.
MAX_TOKENS = {"generator": 8000, "reflector": 16000, "diagnoser": 12000, "curator": 4000,
              # teacher: one agent's feedback per call (~3k + thinking); its cross-run call once hit 16k
              # and was retried, so 32k (the call streams, so a long answer does not time out)
              "teacher": 32000, "reflect": 8000, "organizer": 8000,
              "big_generator": 32000}   # max effort can think long in one turn (ace.llm passes a long timeout)

# output_config.effort: how hard the model thinks. Lower = cheaper.
EFFORT = {"generator": "high", "reflector": "high", "diagnoser": "medium", "curator": "low",
          "teacher": "high", "reflect": "low", "organizer": "low", "big_generator": "high"}   # big_generator: same model and effort as the swarm agents

# Fast mode (research preview, Claude API only): the same model and effort, output generated faster,
# at twice the price ($8 / $40 per MTok on Opus 5.5 instead of $4 / $20). Off to save money. If the
# API rejects it, ace.llm logs that once and continues at normal speed.
# NOTE ace.cost prices every call at the normal rate, so with this on the $ tracker under-counts the
# teacher by 2x.
FAST = {"teacher": False}

# ace/requirements.py (offline, not part of a loop run): the translator rewrites a problem's statement
# as engineering requirements (the teacher's model); the back-translator reads them the way an agent
# would and proposes a setup, to check they still pin down the reference setup (the agent's model).
MODELS.update(translator=MODELS["teacher"], backtranslator=MODELS["generator"])
MAX_TOKENS.update(translator=16000, backtranslator=8000)
EFFORT.update(translator="high", backtranslator="high")

# USD per million tokens. VERIFY these against https://www.anthropic.com/pricing before a real run.
# cache_write = 5-minute cache write (1.25x input); cache_read = cache hit.
PRICES = {
    "claude-sonnet-5-5": {"input": 2.00, "output": 10.00, "cache_write": 2.50, "cache_read": 0.20},
    "claude-opus-5-5": {"input": 4.00, "output": 20.00, "cache_write": 5.00, "cache_read": 0.20},
    "claude-haiku-5-5": {"input": 0.10, "output": 0.50, "cache_write": 0.125, "cache_read": 0.01},
}

# --------------------------------------------------------------------------- tasks
# Catalogue of problems the loop can use: one scored quantity (QoI) each, and a node cap that is
# only a safety limit (the speed/accuracy score already makes wasted nodes cost). Tolerances come
# from each problem's reference (feacore references(): 2% for NAFEMS, 1% for Cook).
TASKS = {
    # NAFEMS linear elastic benchmarks
    "nafems_le1": {"qoi": "syy_D", "node_cap": 20000},            # 2D elliptic membrane, plane stress
    "nafems_le10": {"qoi": "syy_D", "node_cap": 30000},           # 3D thick plate under pressure
    "nafems_le11_axisym": {"qoi": "szz_A", "node_cap": 20000},    # solid cylinder/cone, thermal load, axisymmetric
    "nafems_le11_3d": {"qoi": "szz_A", "node_cap": 30000},        # the same, as a 3D solid
    "cooks_membrane": {"qoi": "uy_C", "node_cap": 20000},         # tapered panel in shear (classic, not NAFEMS)
    # easy sanity problem (planted-failure tests use it; not in the loop)
    "lame_cylinder_plane_strain": {"qoi": "hoop_bore", "node_cap": 20000},
    # printed gold part abc_00000003 pulled on the Instron (feacore/problems/physical.py); scored on the
    # video-extensometer elongation; physical test data can be fed in with run_loop --evidence
    "abc_00000003_pull": {"qoi": "ext_marks", "node_cap": 60000},
    # gold ABC CAD parts (feacore/problems/gold.py): qoi = GoldProblem.scored_qoi(); node cap about the
    # L1 study mesh (48827 / 37613 nodes). Field scoring needs their adaptive reference (ace.gold_reference).
    "abc_00000003_bc7": {"qoi": "strain_energy", "node_cap": 49000},
    "abc_00000065_bc0": {"qoi": "strain_energy", "node_cap": 38000},
}
TRAIN = ["nafems_le10", "nafems_le11_axisym", "cooks_membrane"]
HOLDOUT = ["nafems_le1", "nafems_le11_3d"]   # evaluated every generation, never reflected on

# What the agent is told about a problem (ACE_STATEMENT, set by run_loop --statement; an environment
# variable so the spawned workers see it too):
#   bcs           the problem statement: supports, loads and analysis type spelled out
#   requirements  the problem's engineering requirements instead (ace/requirements.py, stored in
#                 ace/requirements_data/): how the part is held and loaded in service, what is
#                 symmetric, what the CAD represents. The agent decides the boundary conditions and
#                 the analysis type itself. The problem id is replaced by a neutral task label, and
#                 groups named after a support or load get neutral names. Scoring is unchanged.
STATEMENT = os.environ.get("ACE_STATEMENT", "bcs")

# --------------------------------------------------------------------------- agent limits
MAX_SOLVES = 4                    # solve calls per run (stand-in for a wall-clock budget)
MAX_TURNS = 35                    # model calls per run
MIN_QUALITY = 0.1                 # gate: minimum element quality (gmsh minSICN) of the mesh

# --------------------------------------------------------------------------- score
# The team's speed/accuracy score (feacore.diagnostics.score.score_population, docs/failure-modes.md):
#   accuracy   = min(1, tolerance / error)        1 when the scored quantity is within tolerance
#   efficiency = DOFs of the cheapest agent that met tolerance / own DOFs
#   score      = valid * accuracy * efficiency, in [0, 1]; 1 = the cheapest setup accurate enough
# The population is every agent on that problem in the whole run (all generations), so scores
# compare across generations. A run that fails a gate (no solve, over the cap, bad mesh) is invalid.
# What "error" means in the score. "point": the scored quantity at its point vs the published value.
# "field": the energy-norm error of the whole stress field vs a fine reference solve (ace.field_error),
# so refining only around the scored point no longer suffices. Set with ACE_SCORE=field (an
# environment variable, so the agent worker processes see it too).
SCORE = os.environ.get("ACE_SCORE", "point")
# Field mode is scored against an AUTOMESHER: gmsh with all its default sizes (nothing set; gmsh sizes
# from the bounding box) and quadratic elements, on the problem's correct setup (ace.field_error.automesher).
#   g = e_auto / e        accuracy gain over the automesher (e = the run's field error)
#   c = DOFs / DOFs_auto  cost relative to the automesher
#   A = (g / G)^2 below the target gain G (missing it is penalised quadratically), g / G at or above
#       it (beating it keeps paying)
#   score = A / c         1 = exactly G x the automesher's accuracy at its DOFs; the automesher
#                         itself scores 1/G^2 = 0.25. Above the target an extra DOF pays only if
#                         it buys at least proportional accuracy.
AUTO_GAIN_TARGET = 2.0
# Besides the whole field, von Mises stress is checked at PROBE_POINTS points spread over the part,
# only where the reference has converged (not at singularities); the agent is told PROBE_DISCLOSED
# of them, the rest are hidden. The scored error is max(field error, RMS point error).
# ACE_PROBE_POINTS / ACE_PROBE_DISCLOSED change them for a run (cheap: the expensive reference solve
# is cached; points are only picked from it).
PROBE_POINTS = int(os.environ.get("ACE_PROBE_POINTS", 12))
PROBE_DISCLOSED = int(os.environ.get("ACE_PROBE_DISCLOSED", 4))
# "Be the best" bonus: on each hidden probe point, the run with the smallest error among that
# generation's runs on the problem wins 1/n_hidden of BEST_BONUS (a tie pays nobody), added to the score.
BEST_BONUS = 0.25
# Diversity bonus: a valid run whose mesh recipe is structurally different from every other run of
# its generation on the problem, and whose base score is at least DIVERSITY_MARGIN x the best base
# score there, gets +DIVERSITY_BONUS. Different = another element order, or global h off by more than
# DIVERSITY_H_RATIO, or Jaccard distance >= DIVERSITY_JACCARD between the sets of refinement targets
# (groups, edges, and boxes/points rounded to a grid of DIVERSITY_GRID x the part's diagonal).
# Why: agents on a task converged on one recipe, so the swarm explored little (results/scaling_gold_2026-10-31).
DIVERSITY_BONUS = 0.1
DIVERSITY_MARGIN = 0.9
DIVERSITY_H_RATIO = 1.5
DIVERSITY_JACCARD = 0.5
DIVERSITY_GRID = 0.05
FIELD_ERR_FLOOR = 0.005   # errors below this count as this: the reference itself is good to ~0.3%
SCORE_FLOOR = 0.0
SCORE_CEIL = 1.0

# --------------------------------------------------------------------------- playbook
SECTIONS = {  # section -> bullet id prefix
    "meshing": "mesh",
    "bcs": "bc",
    "geometry": "geo",
    "solver_setup": "solv",
    "postprocessing": "post",
    "tool_use": "tool",
}
TOP_N_BULLETS = 30                # only the best N bullets (helpful - harmful) go in the prompt
DEDUP_THRESHOLD = 85              # rapidfuzz token_set_ratio at/above which two bullets are "the same"
PRUNE_MARGIN = 2                  # drop a bullet when harmful >= helpful + PRUNE_MARGIN

# --------------------------------------------------------------------------- attribution (experimental)
# ace/attribution.py is not wired into the loop yet (being evaluated). Plan: hide each bullet
# from each run with probability HIDE_P (seeded by run id) and regress score on what was shown.
HIDE_P = 0.5
BOOTSTRAP = 2000                  # bootstrap resamples for confidence intervals
CI = 0.90                         # confidence level of the reported intervals
# Retire a bullet when its effect's upper bound is below 0 (confidently harmful) once it has been
# both shown and hidden at least MIN_ARM runs each.
MIN_ARM = 6

# --------------------------------------------------------------------------- diagnosis
# "diagnoser": the agentic diagnoser (ace/diagnoser.py) with oracles, stage checks and
# counterfactual reruns; "single_call": the original one-call Reflector (ace/reflector.py).
REFLECTOR_MODE = "diagnoser"
DIAGNOSER_MAX_TOOL_CALLS = 15       # per session (one problem, one generation); then it must wrap up
DIAGNOSER_MAX_RERUNS = 3            # counterfactual reruns per session
DIAGNOSER_MIN_RERUNS = 1            # a submit before this many reruns is refused once (tests, not guesses)
# What the teacher may do (ACE_TEACHER, set by run_loop --teacher), for the ablation:
#   rerun    experiments only: counterfactual reruns, at least DIAGNOSER_MIN_RERUNS of them
#   inspect  no reruns: a detailed look instead (where each run's error is, how its mesh is laid out)
#   both     the inspection tools and reruns (reruns optional)
TEACHER_MODE = os.environ.get("ACE_TEACHER", "rerun")
TRACE_PAGE = 6                      # read_trace returns this many steps per page
TRACE_FIELD_CHARS = 500             # each step's text is cut to this many characters

# --------------------------------------------------------------------------- memory
# Persistent agents a0..a{AGENTS-1}; each attempts every problem every generation. Each has a
# PRIVATE playbook (low bar) and all share one SHARED playbook (high bar). See ace/memory.py.
MEMORY = "shared+private"          # or "shared": one playbook, anonymous runs (RUNS_PER_PROBLEM each)
AGENTS = 6
PRIVATE_PRUNE_MARGIN = 3          # private bullet dropped when harmful >= helpful + 3 (private notes keep ideas)
PRIVATE_TOP_N = 40                # private notes shown to their agent (more than the shared playbook)
PRIVATE_DEDUP_THRESHOLD = 93      # private notes merge only near-identical text, so variants of an idea stay
SHARED_PRUNE_MARGIN = 3           # shared bullet dropped only when harmful >= helpful + 3
PROMOTE_MIN_AGENTS = 2            # an idea in this many agents' private notes becomes shared

# --------------------------------------------------------------------------- loop defaults
RUNS_PER_PROBLEM = 6              # more runs = scores you can tell apart from noise
GENERATIONS = 4                   # gen 0 = empty playbook (baseline)
CONCURRENCY = 4
BUDGET_USD = 10.0
