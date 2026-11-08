"""single_agent: one analyst agent that turns component design data into the analysis layer.

The agent gets a CAD part (STEP) and a design package (what the part does, how it sits in its
system, the design review's open questions). It must decide the idealization (supports, loads,
material), mesh the part, run a convergence study and answer the review's questions with honest
uncertainties. A deterministic evaluator scores it against the gold references, and a learning
step turns the scorecard into general rules for the next problems. See single_agent/README.md.
"""
