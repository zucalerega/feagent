"""The learning step: from a generation's scorecards to general rules (single_agent.rulebook).

One structured call per generation (role "sa_reflect": by default the arm's own model, so the agent learns
from its own mistakes; --reflector opus gives a Sonnet arm an Opus teacher). It sees, for every TRAINING
analysis of the generation (never a holdout one):
  - the design package the agent read;
  - what the agent did (its tool calls in short) and its report (idealization, answers, rules it cited);
  - the scorecard: the setup check with the intended idealization item by item, each answer's error and
    whether its uncertainty covered the truth, the study, the meshing error.
and the current rulebook. It returns a diagnosis per analysis and rule operations (add / edit / remove /
vote). single_agent.rulebook lints them (general, short, no part, face or load magnitude) and merges them.
"""

from __future__ import annotations

import json
from typing import Literal

import pydantic

from single_agent.rulebook import SECTIONS, Rulebook

SYSTEM = """\
You improve the rulebook of a structural-analysis agent. The agent receives a component's design package (what the part
is, how it is installed and loaded, the design review's questions) and a CAD model. With tools it decides the
supports, loads and material, meshes the part, runs a convergence study and answers the questions with uncertainties.
Its rulebook (a fixed handbook A1..F3, plus learned rules L001...) is part of its prompt for every analysis.

You get the agent's latest analyses with their scorecards. The scorecard knows the intended idealization and the true
answers; the agent never sees them. Work in two steps.

1. Diagnose each analysis: what went well, what went wrong, and the root cause of each failure: misreading the design
   data (which face, which kind of support or load, which value, units), the mesh, the study (too few meshes,
   ratios too small, wrong extrapolation), the singularity judgement, the uncertainty (too narrow, too wide), or the
   process (ran out of turns or solves, never submitted).

2. Propose rulebook operations that would have prevented the failures in OTHER parts and load cases too:
   - add: a new rule. One instruction, at most 60 words, in the form "When <situation>, <do this>, because <reason>".
     It must be general: never name a face (f12), a problem, a part, a load case or a load magnitude; describe the
     situation instead ("a face that rests on a greased plate", "a peak stress at the rim of a bonded face").
   - edit: rewrite a learned rule (L...) that is wrong, vague or too narrow. Handbook rules (A1..F3) cannot be edited.
   - remove: a learned rule that misled the agent.
   - vote: helpful or harmful, for a learned rule the agent followed in these analyses (see rules_used and its
     actions), judged by the outcome.
   Prefer fixing an existing rule to adding a near-duplicate. Propose few operations (0 to 4 adds): each must address a
   failure you saw, not restate the handbook. If the agent did well and followed the rules, propose nothing new.
"""


class Diagnosis(pydantic.BaseModel):
    task: str = pydantic.Field(description="the workspace label of the analysis")
    went_well: list[str]
    went_wrong: list[str]
    root_causes: list[str]


class RuleOp(pydantic.BaseModel):
    op: Literal["add", "edit", "remove", "vote"]
    id: str | None = pydantic.Field(None, description="the rule id for edit / remove / vote")
    section: Literal[SECTIONS] | None = None   # type: ignore[valid-type]
    text: str | None = pydantic.Field(None, description="the rule text for add / edit")
    vote: Literal["helpful", "harmful"] | None = None
    reason: str


class Reflection(pydantic.BaseModel):
    diagnoses: list[Diagnosis]
    ops: list[RuleOp]


# --------------------------------------------------------------------------- what the reflector reads


def _short(x, n: int = 300) -> str:
    s = json.dumps(x, default=float) if not isinstance(x, str) else x
    return s if len(s) <= n else s[: n - 3] + "..."


def process_summary(run: dict) -> list[str]:
    """The agent's tool calls in one line each, with the gist of each result."""
    lines, results = [], {t["id"]: t for t in run.get("trace", []) if t["type"] == "tool_result"}
    for t in run.get("trace", []):
        if t["type"] == "nudge":
            lines.append(f"turn {t['turn']}: stopped without submitting (nudged)")
        if t["type"] != "tool_call":
            continue
        name, inp = t["name"], t["input"] or {}
        r = results.get(t["id"], {})
        out, err = r.get("result") or {}, r.get("is_error")
        if name == "mesh":
            gist = f"h={inp.get('h')} order={inp.get('order')} {len(inp.get('refinements') or [])} refinements"
            res = f"-> mesh {out.get('mesh')}: {out.get('nodes')} nodes, min quality {out.get('min_quality')}"
        elif name == "refine_mesh":
            gist = f"mesh {inp.get('mesh')} / {inp.get('factor')}"
            res = f"-> mesh {out.get('mesh')}: {out.get('nodes')} nodes"
        elif name == "solve":
            bcs = "; ".join(f"{b.get('type')} {b.get('group') or b.get('point')}"
                            + (f" {b.get('components')}" if b.get("components") is not None else "")
                            + (f" = {b.get('value')}" if b.get("value") is not None else "") for b in inp.get("bcs") or [])
            gist = f"mesh {inp.get('mesh')}, E={((inp.get('material') or {}).get('E'))}, bcs: {bcs}; qois: " \
                   f"{[q.get('name') for q in inp.get('qois') or []]}"
            eq = ((out.get("feedback") or {}).get("equilibrium") or {}).get("residual")
            res = f"-> solve {out.get('solve')}: {out.get('dofs')} DOFs, qois {_short(out.get('qois'), 200)}, " \
                  f"equilibrium residual {eq}"
        elif name == "study":
            gist = f"solves {inp.get('solves')}"
            res = "-> " + "; ".join(f"{k}: {v.get('convergence', '')} p={v.get('p_observed')} GCI={v.get('gci_fine')}"
                                    for k, v in (out.get("qois") or {}).items())
        elif name == "submit_report":
            gist, res = "", "-> submitted"
        else:
            gist, res = _short(inp, 120), "-> " + _short(out, 160)
        lines.append(f"turn {t['turn']}: {name} {gist} {('ERROR ' + _short(out.get('error'), 200)) if err else res}")
    return lines


def debrief(run: dict, card: dict, package: str) -> str:
    rep = (run.get("session") or {}).get("report") or {}
    parts = [f"### Analysis {run['label']}", "", "Design package the agent read:", "", package.strip(), "",
             f"Outcome: stop={run.get('stop')}, turns={run.get('turns')}, error={run.get('error')}", "",
             "What the agent did:", *[f"- {ln}" for ln in process_summary(run)], "",
             "The agent's report:",
             f"- idealization: {_short(rep.get('idealization'), 2000)}",
             f"- answers: {_short(rep.get('answers'), 2000)}",
             f"- rules it cited: {rep.get('rules_used')}",
             f"- summary: {_short(rep.get('summary'), 600)}", "",
             "Scorecard (the agent never saw this):",
             f"- score {card.get('score')} = parts {card.get('parts')}"]
    s = card.get("setup") or {}
    if s:
        parts.append(f"- setup equivalent to the intended one: {s.get('equivalent')} (largest QoI difference "
                     f"{s.get('max_difference')}; material ok: {s.get('material_ok')})")
        for it in (s.get("match") or {}).get("items", []):
            parts.append(f"  - intended {it['reference']}: {it['status']} {it.get('detail', '')}")
        for e in (s.get("match") or {}).get("extra", []):
            parts.append(f"  - agent added {e}: not in the intended model")
    for a in card.get("answers") or []:
        parts.append(f"- {a['id']} ({a['expect']}): {a.get('verdict')}; reported {a.get('value')} "
                     f"(status {a.get('status')}, uncertainty {a.get('uncertainty')}, covers truth: {a.get('covered')}, "
                     f"from own solves: {a.get('supported')}); true {a.get('reference')}")
    st = card.get("study") or {}
    parts.append(f"- study: {st.get('meshes_with_final_setup')} meshes with the final setup, studies {st.get('studies_on_final_setup')}, "
                 f"h ratios {st.get('h_rep_ratios')}")
    m = card.get("meshing") or {}
    if m:
        parts.append(f"- the agent's final mesh ({m.get('dofs')} DOFs) with the intended setup: "
                     + "; ".join(f"{k} error {v['error']} (uniform mesh with the same DOFs: {v['uniform_error_same_dofs']})"
                                 for k, v in (m.get("by_qoi") or {}).items()))
    if card.get("failed"):
        parts.append(f"- FAILED: {card['failed']}")
    return "\n".join(parts)


def build_prompt(items: list[tuple[dict, dict, str]], rulebook: Rulebook) -> str:
    return ("## The current rulebook\n\n" + rulebook.render() + "\n\n## The analyses of this generation\n\n"
            + "\n\n".join(debrief(r, c, p) for r, c, p in items)
            + "\n\nDiagnose each analysis, then propose rulebook operations (JSON).")


def reflect(llm, items: list[tuple[dict, dict, str]], rulebook: Rulebook) -> tuple[Reflection | None, list, str]:
    """items: (run record, scorecard, package text) of the training analyses. Returns (reflection or None,
    usage records, the prompt sent)."""
    from single_agent.agent import ROLE_REFLECT
    prompt = build_prompt(items, rulebook)
    records: list = []
    out, records = llm.structured(ROLE_REFLECT, SYSTEM, prompt, Reflection, records)
    return out, records, prompt
