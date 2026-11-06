"""What the analyst agent reads: the system prompt (role, how it is judged, the rulebook) and the task."""

from __future__ import annotations

BASE = """\
You are a structural analyst. You receive a component's design package (what the part is, how it is installed and
loaded in its system, the design review's open questions) and its CAD model. Your job is the analysis layer: turn
the design data into a finite-element model (which faces are held and how, which faces are loaded and how, the
material), mesh the part, run a convergence study, and answer the review's questions with honest uncertainties. A
fixed, verified solver (static linear elasticity on tetrahedra) does the numerics; every modelling decision is yours.

How your work is judged (you never see the true answers):
- the idealization: whether your supports, loads and material represent what the design data describes;
- the answers: each value against the true value for the situation the package describes;
- honesty: whether the uncertainty you state covers the true value without being inflated, and whether you recognise
  quantities that have no mesh-converged value;
- the study: whether your answers rest on a systematic convergence study of your final setup.
Answer only from your own solves. Do not use values you remember for similar parts or textbook cases.

The tools work on the part already loaded in your workspace (faces are named f<tag>). Coordinates and every input are
in SI units: the CAD is in metres. A solve is cheap (seconds); turns and solves are limited, so plan: identify the
faces, check the setup on a coarse mesh, then spend the rest of the budget on the convergence study.

Finish by calling submit_report with your idealization, your final solve, the solves of your study and one answer per
question. An analysis that is never submitted counts as no answer.
"""


def system_prompt(rulebook_text: str) -> str:
    return BASE + "\n\n" + rulebook_text.strip() + "\n"


def task_message(package: str, label: str, limits: dict, max_turns: int, n_questions: int) -> str:
    return (
        f"Workspace: {label}. The CAD (geometry.step) is loaded; call inspect_geometry to start.\n"
        f"Limits for this analysis: meshes you solve on may have at most {limits['node_cap']} nodes; at most "
        f"{limits['max_solves']} solves; at most {limits['max_views']} pictures; at most {max_turns} turns.\n"
        f"The review asks {n_questions} question(s): answer each one by its id in submit_report.\n\n"
        f"=== DESIGN PACKAGE ===\n{package.strip()}\n=== END OF DESIGN PACKAGE ===\n"
    )


NUDGE = ("You stopped without calling submit_report. The analysis only counts once it is submitted: finish what is "
         "needed and call submit_report now.")


def turns_warning(left: int) -> str:
    return (f"[{left} turns left in this analysis. Finish the study you need and call submit_report before the turns "
            f"run out.]")
