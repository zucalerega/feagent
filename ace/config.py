"""Every knob in one place: models, token caps, prices, tasks, loop limits."""

# --------------------------------------------------------------------------- models
# Switch any role to another model with one line.
MODELS = {
    "generator": "claude-sonnet-5-5",   # the FEA agent (many calls per run)
    "reflector": "claude-opus-5-5",     # the hard step: one call per problem per generation
    "curator": "claude-sonnet-5-5",     # bookkeeping: lesson -> ADD or duplicate-of
}

# max_tokens caps one response (thinking counts toward it). You pay for what is used, not the cap.
MAX_TOKENS = {"generator": 8000, "reflector": 16000, "curator": 4000}

# output_config.effort: how hard the model thinks. Lower = cheaper.
EFFORT = {"generator": "medium", "reflector": "high", "curator": "low"}

# USD per million tokens. VERIFY these against https://www.anthropic.com/pricing before a real run.
# cache_write = 5-minute cache write (1.25x input); cache_read = cache hit.
PRICES = {
    "claude-sonnet-5-5": {"input": 2.00, "output": 10.00, "cache_write": 2.50, "cache_read": 0.20},
    "claude-opus-5-5": {"input": 4.00, "output": 20.00, "cache_write": 5.00, "cache_read": 0.20},
    "claude-haiku-5-5": {"input": 0.10, "output": 0.50, "cache_write": 0.125, "cache_read": 0.01},
}

# --------------------------------------------------------------------------- tasks
# One scored QoI per problem. node_cap is a hard limit on mesh nodes (mid-side nodes of
# quadratic elements count). Caps were picked so the best uniform mesh (linear or quadratic) at
# the cap is roughly 4-9% off; the agent has to place nodes well to beat that.
# The references of the notched bar and stepped shaft come from the team's own converged FE
# studies, so no model can recall them (it could recall the NAFEMS and Lame answers).
TASKS = {
    "notched_bar_t5_r2": {"qoi": "peak_stress_notch", "node_cap": 1500},
    "stepped_shaft_axi_Dd2_rd0p1": {"qoi": "peak_stress_fillet", "node_cap": 800},
    "notched_bar_t10_r4": {"qoi": "peak_stress_notch", "node_cap": 1500},
    "plate_hole_finite_width_d03": {"qoi": "s_peak", "node_cap": 1500},
}
TRAIN = ["notched_bar_t5_r2", "stepped_shaft_axi_Dd2_rd0p1"]
# Evaluated every generation, never reflected on: the same family with a different notch (near
# transfer) and a different shape whose reference is a handbook value (far transfer).
HOLDOUT = ["notched_bar_t10_r4", "plate_hole_finite_width_d03"]

# --------------------------------------------------------------------------- agent limits
MAX_SOLVES = 4                    # solve calls per run (stand-in for a wall-clock budget)
MAX_TURNS = 25                    # model calls per run
MIN_QUALITY = 0.1                 # gate: minimum element quality (gmsh minSICN) of the mesh

# --------------------------------------------------------------------------- score
# score = log10(e_uniform / e_agent): 0 = as good as the best uniform mesh (linear or quadratic)
# at the cap, +1 = 10x less error. Clipped; any failed gate scores SCORE_FLOOR.
SCORE_FLOOR = -2.0
SCORE_CEIL = 3.0

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

# --------------------------------------------------------------------------- loop defaults
RUNS_PER_PROBLEM = 6              # more runs = scores you can tell apart from noise
GENERATIONS = 4                   # gen 0 = empty playbook (baseline)
CONCURRENCY = 4
BUDGET_USD = 10.0
