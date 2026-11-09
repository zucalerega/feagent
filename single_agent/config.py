"""Every knob of the single-agent loop in one place."""

from __future__ import annotations

from pathlib import Path

HERE = Path(__file__).resolve().parent
DESIGN_DIR = HERE / "design"          # design packages: <problem_id>/package.md (agent) + key.json (evaluator)
HANDBOOK = HERE / "handbook.md"       # the general rules every arm starts from

# --------------------------------------------------------------------------- models
# One arm = one model doing everything: the analyst and (by default) its own learning step.
ARMS = {"opus": "claude-opus-5-5", "sonnet": "claude-sonnet-5-5"}
# Thinking depth. Opus 5.5 defaults to "medium" when unset, Sonnet 5.5 to "high": set both, so the
# two arms are compared at the same nominal effort.
EFFORT = {"agent": "high", "reflect": "high"}
MAX_TOKENS = {"agent": 16000, "reflect": 16000}
# Seconds of a non-streamed agent turn before the SDK gives up (a long think on a hard step).
TIMEOUT = {"agent": 900, "reflect": 600}

# --------------------------------------------------------------------------- agent limits
NODE_CAP = 50_000        # nodes of any mesh the agent solves on (P2 tets: ~150k DOFs, ~20 s and ~3 GB per solve)
MAX_SOLVES = 12          # solve calls per problem (a convergence study needs 3-4)
MAX_TURNS = 60           # model calls per problem
MAX_VIEWS = 12           # pictures per problem (view, view_geometry)
MAX_NUDGES = 2           # "you have not submitted" reminders when the model stops early
MIN_QUALITY = 0.1        # minimum element quality (gmsh minSICN) of a mesh a reported answer may rest on

# --------------------------------------------------------------------------- problems
# The complex geometries: gold ABC parts (feacore.problems.gold) with a design package each.
# Train: the agent learns from these. Holdout: never reflected on; same part with a new installation
# (abc_00000003_bc7) and a different part (abc_00000065_*), to see whether the rules transfer.
TRAIN = ["abc_00000003_bc0", "abc_00000003_bc1", "abc_00000003_bc2", "abc_00000003_bc5"]
HOLDOUT = ["abc_00000003_bc7", "abc_00000065_bc0", "abc_00000065_bc1"]
# Spares: abc_00000003_bc4 and _bc6 (use them with --problems / --holdout). abc_00000003_bc3 (the shrink-fit, LC-11)
# is retired with its package: its gold meshes fold into the bore (benchmarks/gold/README.md, "Retired").

# --------------------------------------------------------------------------- evaluation
EQUIV_TOL = 2e-3         # setups giving the same QoIs on the same mesh to this are equivalent
EVAL_MESH_H = 0.006      # global size (m) of the common mesh setups are compared on (P2; parts are 100 mm across)
# Composite score = weighted mean of the parts (each in [0, 1]); see evaluate.py.
WEIGHTS = {"setup": 0.30, "answers": 0.40, "calibration": 0.15, "study": 0.15}

# --------------------------------------------------------------------------- learning
LEARNED_MAX = 30         # learned rules kept in the rulebook
RULE_MAX_WORDS = 70      # a rule longer than this is rejected (rules must be short and general)
PRUNE_MARGIN = 2         # a learned rule is dropped when harmful >= helpful + PRUNE_MARGIN

# --------------------------------------------------------------------------- loop defaults
GENERATIONS = 4
CONCURRENCY = 2          # problems run in parallel (each in its own process: gmsh is process-global)
BUDGET_USD = 40.0
HOLDOUT_EVERY = 0        # 0: holdout only in the first and the last generation; k: every k-th as well
