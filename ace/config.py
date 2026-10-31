"""Every knob in one place: models, token caps, prices, tasks, loop limits."""

# --------------------------------------------------------------------------- models
# Switch any role to another model with one line.
MODELS = {
    "generator": "claude-sonnet-5-5",   # the FEA agent (many calls per run)
    "reflector": "claude-sonnet-5-5",   # single-call reflector (REFLECTOR_MODE = "single_call")
    "diagnoser": "claude-sonnet-5-5",   # agentic diagnoser; switch to "claude-opus-5-5" if Sonnet falls short
    "curator": "claude-sonnet-5-5",     # bookkeeping: lesson -> ADD or duplicate-of
}

# max_tokens caps one response (thinking counts toward it). You pay for what is used, not the cap.
MAX_TOKENS = {"generator": 8000, "reflector": 16000, "diagnoser": 12000, "curator": 4000}

# output_config.effort: how hard the model thinks. Lower = cheaper.
EFFORT = {"generator": "medium", "reflector": "high", "diagnoser": "medium", "curator": "low"}

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
    # CFD (physics "flow", steady laminar, openfoam backend), 2D only for now. The caps are safety
    # limits like the others: a uniform mesh at the cap meets the tolerance (calibration.json), so
    # the score is about the cheapest accurate mesh. 3D flow tasks (pipe_poiseuille_3d,
    # square_duct_3d) are left out: on the unstructured tets the agent's mesh tool makes, the
    # backend's axial pressure gradient is 12-19 % high at 4k-33k nodes (2D triangles and structured
    # hexes are fine), so no agent could meet a 2 % tolerance (ace/README.md, "The CFD suite").
    "poiseuille_channel": {"qoi": "dp_AB", "node_cap": 1200},             # 2D channel, exact (easy sanity case)
    "lid_driven_cavity_re100": {"qoi": "u_vcl_min", "node_cap": 2500},    # 2D closed cavity, Ghia et al. 1982
    "taylor_couette_2d": {"qoi": "u_theta_mid", "node_cap": 2500},        # 2D annulus, rotating wall, exact
    "cylinder_channel_dfg_2d1": {"qoi": "drag", "node_cap": 12000},       # 2D external flow, Schaefer & Turek 1996
}
TRAIN = ["nafems_le10", "nafems_le11_axisym", "cooks_membrane"]
HOLDOUT = ["nafems_le1", "nafems_le11_3d"]   # evaluated every generation, never reflected on
# The CFD suite (run_loop --suite cfd): held out is a new topology, external flow past a cylinder,
# where the cheapest accurate mesh refines at the body instead of everywhere.
TRAIN_CFD = ["poiseuille_channel", "lid_driven_cavity_re100", "taylor_couette_2d"]
HOLDOUT_CFD = ["cylinder_channel_dfg_2d1"]
SUITES = {"fea": (TRAIN, HOLDOUT), "cfd": (TRAIN_CFD, HOLDOUT_CFD)}

# --------------------------------------------------------------------------- agent limits
MAX_SOLVES = 4                    # solve calls per run (stand-in for a wall-clock budget)
MAX_TURNS = 25                    # model calls per run
MIN_QUALITY = 0.1                 # gate: minimum element quality (gmsh minSICN) of the mesh

# --------------------------------------------------------------------------- score
# The team's speed/accuracy score (feacore.diagnostics.score.score_population, docs/failure-modes.md):
#   accuracy   = min(1, tolerance / error)        1 when the scored quantity is within tolerance
#   efficiency = DOFs of the cheapest agent that met tolerance / own DOFs
#   score      = valid * accuracy * efficiency, in [0, 1]; 1 = the cheapest setup accurate enough
# The population is every agent on that problem in the whole run (all generations), so scores
# compare across generations. A run that fails a gate (no solve, over the cap, bad mesh) is invalid.
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
TRACE_PAGE = 6                      # read_trace returns this many steps per page
TRACE_FIELD_CHARS = 500             # each step's text is cut to this many characters

# --------------------------------------------------------------------------- memory
# Persistent agents a0..a{AGENTS-1}; each attempts every problem every generation. Each has a
# PRIVATE playbook (low bar) and all share one SHARED playbook (high bar). See ace/memory.py.
MEMORY = "shared+private"          # or "shared": one playbook, anonymous runs (RUNS_PER_PROBLEM each)
AGENTS = 6
PRIVATE_PRUNE_MARGIN = 1          # private bullet dropped when harmful >= helpful + 1
SHARED_PRUNE_MARGIN = 3           # shared bullet dropped only when harmful >= helpful + 3
PROMOTE_MIN_AGENTS = 2            # an idea in this many agents' private notes becomes shared

# --------------------------------------------------------------------------- loop defaults
RUNS_PER_PROBLEM = 6              # more runs = scores you can tell apart from noise
GENERATIONS = 4                   # gen 0 = empty playbook (baseline)
CONCURRENCY = 4
BUDGET_USD = 10.0
