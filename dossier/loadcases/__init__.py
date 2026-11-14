"""The plan stage: which load cases verify the component, across the kinds of analysis.

    from dossier.loadcases import plan
    plan(project, ctx, geo) -> dossier.pipeline.Plan(load_cases, hand_calcs, journal)

A load case is one phase of the system at limit loads. The requirements say which phases need which kind of
analysis (structural, thermal, flow, fatigue); the system context gives the loads. The plan also holds the
hand calculations of each load case, and a journal with one decision for every load case that is selected,
scaled or left out.

    select.py      the rules (read its docstring first), and the Planner that applies them
    explain.py     the statements of the load cases and the decisions, assumptions and findings of the plan
    handcalcs.py   the hand calculations of each load case (dossier.handcalc evaluates them)
    wording.py     helpers for text in ASD-STE100 Simplified Technical English
"""

from dossier.loadcases.select import FAMILY, plan

__all__ = ["FAMILY", "plan"]
