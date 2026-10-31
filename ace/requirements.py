"""Engineering requirements in place of boundary conditions.

In requirements mode (run_loop --statement requirements, config.STATEMENT) the agent is not told a
problem's supports, loads and analysis type. It gets the problem's engineering requirements: what the
part is and what the CAD holds, its material, how it is loaded and held in service, what is
symmetric, and the result wanted. From these it sets the boundary conditions itself, with the same
tools. Scoring does not change (the scored quantity of the agent's own solve against the problem's
reference value), so a setup that misreads the requirements scores badly; checks.bc_report and the
teachers still compare the agent's setup with the reference setup.

Each problem's requirements are one JSON file, ace/requirements_data/<problem_id>.json:
  requirements   the agent-facing text; replaces problem.description() in the task prompt
  group_aliases  {real group name: neutral name} for groups whose names give a support or load away
                 (Cook's "clamped" -> "left_edge"); ace.fea.get_problem applies them everywhere
  mapping        for reviewers, never shown to agents: which requirement implies each reference item
  source         how the file was written, and the fingerprint of the statement and reference setup
                 it was written from (a changed problem makes the file stale)
  validation     lint issues and the round-trip result when it was written

The task prompt also shows a neutral task label instead of the problem id, so a model cannot recall
a named benchmark's setup.

Making them (needs ANTHROPIC_API_KEY): python -m ace.requirements generate [--problems a,b]
  1. translate   the translator model rewrites statement + reference setup as requirements, under
                 rules that forbid boundary-condition language (TRANSLATOR_SYSTEM)
  2. lint        deterministic: no BC or solver vocabulary, no benchmark names, no group identifiers,
                 every number of the reference setup present, aliases neutral and complete, not stale
  3. round trip  the back-translator (the agents' model) reads only what an agent sees (task label,
                 requirements, scored quantity, aliased groups, the solve tool's BC types) and
                 proposes a setup, `trials` times. A trial passes when that setup gives the same scored
                 quantity as the reference setup on the same mesh (to EQUIVALENCE_TOL), so equivalent
                 setups pass even when written differently (force vs traction, fixed vs displacement)
  4. revise      lint issues and failed trials go back to the translator, up to --revisions times
Checking them: python -m ace.requirements check [--round-trip] (lint + staleness; --round-trip
re-runs step 3). Reading one as the agent sees it: python -m ace.requirements show <problem_id>.
"""

from __future__ import annotations

import argparse
import datetime
import functools
import hashlib
import json
import math
import re
import sys
import tempfile
from pathlib import Path
from typing import Literal

import pydantic

from ace import config, fea
from ace.config import TASKS

STORE = Path(__file__).with_name("requirements_data")
EQUIVALENCE_TOL = 1e-3      # relative difference of the scored quantity: same setup, up to round-off
NUMBER_RTOL = 2e-3          # a number in the text matches a required one to this (0.333 ~ 1/3)

TASK_PREFACE = (
    "ENGINEERING REQUIREMENTS. This task gives requirements, not a solver setup: nothing below is stated "
    "as a boundary condition. From the requirements (what the CAD holds, how the part is held and loaded "
    "in service, what is symmetric) decide the analysis type, the supports and the loads yourself, and "
    "set them in your solve call on the groups that inspect_geometry lists."
)

# Phrases the agent-facing text must not contain: they state the solver input instead of the situation.
FORBIDDEN = [
    (r"\bboundary[- ]conditions?\b|\bBCs?\b", "names boundary conditions"),
    (r"\b(?:fixed|clamped|clamping|encastr[eé]|pinned|rollers?)\b", "uses a support-type word"),
    (r"\b(?:zero|null|prescribed)\s+(?:[\w-]+\s+){0,2}(?:[\w-]*displacements?|translations?|rotations?)\b",
     "prescribes a displacement"),
    (r"\bdisplacements?\s+(?:is|are|=|equals?|set\s+to)\s*(?:to\s+)?(?:zero|0)\b", "prescribes a displacement"),
    (r"\bu_?\{?[xyzr]\}?\s*=\s*0\b", "prescribes a displacement"),
    (r"\bsymmetry\s+(?:boundary|condition|constraint|bc)s?\b", "names a symmetry boundary condition"),
    (r"\btractions?\b", "uses the solver's load-type word 'traction'"),
    (r"\bplane[- ]stress\b|\bplane[- ]strain\b|\baxi-?symmetric\b", "names the analysis formulation"),
    (r"\brigid[- ]body\s+(?:motions?|modes?)\b", "talks about rigid-body modes (solver input)"),
    (r"\bNAFEMS\b|\bLE\s?-?\d+\b|\bCook'?s?\b|\bLam[eé]\b|\bTimoshenko\b|\bbenchmark\b",
     "names the benchmark (an agent could recall its setup)"),
]
# A group whose name contains one of these gives its support or load away and needs an alias.
ROLE_WORDS = ("clamp", "fix", "load", "sym", "support", "constrain", "restrain", "pressure", "force",
              "traction", "anchor", "roller", "encastre", "pinned")   # not "bc": edge_BC is points B-C


# --------------------------------------------------------------------------- store


def path_for(problem_id: str) -> Path:
    return STORE / f"{problem_id}.json"


@functools.lru_cache(maxsize=None)
def _read(path: str, mtime_ns: int) -> dict:
    return json.loads(Path(path).read_text())


def load(problem_id: str) -> dict:
    p = path_for(problem_id)
    if not p.exists():
        raise FileNotFoundError(f"no requirements for {problem_id} ({p}); write them with "
                                f"python -m ace.requirements generate --problems {problem_id}")
    return _read(str(p), p.stat().st_mtime_ns)


def save(problem_id: str, data: dict) -> Path:
    STORE.mkdir(parents=True, exist_ok=True)
    p = path_for(problem_id)
    p.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    return p


def aliases_for(problem_id: str) -> dict[str, str]:
    """{real group name: alias} for this problem; {} when it has no requirements file."""
    return dict(load(problem_id).get("group_aliases") or {}) if path_for(problem_id).exists() else {}


def agent_text(problem_id: str) -> str:
    return load(problem_id)["requirements"]


def label_for(problem_id: str) -> str:
    """The neutral task label shown instead of the problem id (stable, not invertible by eye)."""
    return "task-" + hashlib.sha256(problem_id.encode()).hexdigest()[:8]


def problem_for_label(label: str) -> str:
    for pid in {*TASKS, *(p.stem for p in STORE.glob("*.json"))}:
        if label_for(pid) == label:
            return pid
    raise KeyError(f"no problem has task label {label!r}")


def raw_problem(problem_id: str):
    """The problem with its real group names, whatever config.STATEMENT says."""
    from feacore.problems import registry
    return registry.get(problem_id)


def view_problem(problem_id: str, aliases: dict[str, str]):
    """The problem as an agent sees it with these aliases."""
    raw = raw_problem(problem_id)
    return fea.AliasedProblem(raw, aliases) if aliases else raw


def scored_qoi_name(problem_id: str, problem) -> str:
    return TASKS[problem_id]["qoi"] if problem_id in TASKS else problem.qois()[0].name


def fingerprint(problem_id: str) -> str:
    """Hash of what the requirements were written from: the statement and the reference setup."""
    from ace.checks import reference_setup
    raw = raw_problem(problem_id)
    blob = json.dumps({"statement": raw.description(), "setup": reference_setup(raw)}, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()


# --------------------------------------------------------------------------- lint

_UNITS = {"gpa": 1e9, "mpa": 1e6, "kpa": 1e3, "pa": 1.0, "mn": 1e6, "kn": 1e3, "n": 1.0,
          "mm": 1e-3, "cm": 1e-2, "m": 1.0}
_NUMBER = re.compile(
    r"(?<![\w.])(\d+(?:\.\d+)?(?:[eE][-+−]?\d+)?)"           # 210, 0.3, 2.3e-4
    r"(?:\s*/\s*(\d+(?:\.\d+)?))?"                                 # 1/3, 1/16
    r"(?:\s*[×x*]\s*10\s*\^?\s*\(?([-+−]?\d+)\)?)?"      # 2.3 x 10^-4
    r"(?:\s*(GPa|MPa|kPa|Pa|MN|kN|N|mm|cm|m)\b)?")


def numbers_in(text: str) -> list[float]:
    """Every number in the text, as written and (when a unit follows) in SI units."""
    out = []
    for m in _NUMBER.finditer(text):
        v = float(m.group(1).replace("−", "-"))
        if m.group(2):
            v /= float(m.group(2))
        if m.group(3):
            v *= 10.0 ** int(m.group(3).replace("−", "-"))
        out.append(v)
        if m.group(4):
            out.append(v * _UNITS[m.group(4).lower()])
    return out


def _numeric(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def required_numbers(setup: dict) -> list[tuple[str, list[list[float]]]]:
    """(label, options): the text must contain every value of at least one option. A vector needs its
    magnitude or each of its nonzero components."""
    req = []

    def value(label, v):
        if _numeric(v) and v != 0:
            req.append((label, [[abs(float(v))]]))
        elif isinstance(v, list) and any(_numeric(x) and x != 0 for x in v):
            comps = [abs(float(x)) for x in v if _numeric(x) and x != 0]
            norm = math.sqrt(sum(float(x) ** 2 for x in v if _numeric(x)))
            req.append((label, [[norm], comps]))

    mat = setup.get("material") or {}
    for k in ("E", "nu", "alpha"):
        value(f"material {k} = {mat.get(k)}", mat.get(k))
    if setup.get("formulation") in ("plane_stress", "plane_strain"):
        value(f"thickness = {setup.get('thickness')}", setup.get("thickness"))
    value(f"stress-free temperature = {setup.get('T_ref')}", setup.get("T_ref"))
    for bc in setup.get("bcs", []):
        where = bc.get("group") or f"the node nearest {bc.get('point')}"
        value(f"{bc['type']} value on {where} = {bc.get('value')}", bc.get("value"))
    for ld in setup.get("loads", []):
        value(f"{ld['type']} load = {ld.get('value')}", ld.get("value"))
    return req


def _is_role(name: str) -> bool:
    tokens = [t for t in re.split(r"[^a-z0-9]+", name.lower()) if t]
    return any(t.startswith(w) for t in tokens for w in ROLE_WORDS)


def lint(problem_id: str, data: dict) -> list[str]:
    """Deterministic checks of one requirements record; [] = clean."""
    from ace.checks import reference_setup
    raw = raw_problem(problem_id)
    text = data.get("requirements") or ""
    issues: list[str] = []
    if not text.strip():
        return ["the requirements text is empty"]
    for pattern, why in FORBIDDEN:
        found = sorted({m.group(0) for m in re.finditer(pattern, text, re.IGNORECASE)})
        if found:
            issues.append(f"{why}: {', '.join(repr(f) for f in found)}")
    groups = [g["name"] for g in fea.geometry_summary(raw)["groups"]]
    for g in groups:
        if re.search(r"[_']", g) and g in text:
            issues.append(f"names the CAD group {g!r}: describe the place instead")
    aliases = data.get("group_aliases") or {}
    for real, alias in aliases.items():
        if real not in groups:
            issues.append(f"alias for {real!r}, which is not a group of this problem ({groups})")
        if not re.fullmatch(r"[a-z][a-z0-9_]*", alias or ""):
            issues.append(f"alias {alias!r} for {real!r}: use lowercase letters, digits and _")
        elif _is_role(alias):
            issues.append(f"alias {alias!r} for {real!r} still gives a support or load away")
        if alias in groups and alias != real:
            issues.append(f"alias {alias!r} for {real!r} is the name of another group")
    if len(set(aliases.values())) != len(aliases):
        issues.append("two groups have the same alias")
    for g in groups:
        if _is_role(g) and g not in aliases:
            issues.append(f"group {g!r} gives its support or load away: give it a neutral alias")
    nums = numbers_in(text)
    present = lambda v: any(abs(n - v) <= NUMBER_RTOL * abs(v) for n in nums)  # noqa: E731
    for label, options in required_numbers(reference_setup(raw)):
        if not any(all(present(v) for v in opt) for opt in options):
            issues.append(f"a number of the reference setup is missing: {label}")
    fp = (data.get("source") or {}).get("fingerprint")
    if fp and fp != fingerprint(problem_id):
        issues.append("stale: the problem's statement or reference setup changed since these were written")
    return issues


# --------------------------------------------------------------------------- AI: translate and read back

TRANSLATOR_SYSTEM = """\
You rewrite a finite-element benchmark statement as the ENGINEERING REQUIREMENTS an engineer would be
handed for the same analysis. An agent will get your requirements INSTEAD of the statement and must work
out the boundary conditions, loads and analysis type from them on its own. A separate reader then checks
that your requirements lead back to the reference setup.

You get: the statement (it spells the boundary conditions out), the reference setup (the solver input the
statement means, with the CAD's group names), the CAD's named groups with bounding boxes, and the quantity
that will be scored.

Rules:
1. Same physics. The only correct reading of your requirements must be the reference setup: the same
   supports, the same loads (magnitude, direction, where they act, how they are distributed), the same
   material, the same analysis type, the same thickness for planar models. A reader who knows mechanics
   must not be able to defend a different setup.
2. Describe the physical situation, never the solver input. How the part is held: what it is attached,
   bolted, welded or bonded to, what it rests or slides on, guides, knife-edges, rigid frames or platens,
   and what each lets move. How it is loaded in service: pressures, pulls, distributed shear, temperature
   fields, with magnitudes, directions and distribution. When the reference setup is an idealization with
   no everyday counterpart (support along one line only, a face that may slide one way only), describe a
   physical arrangement that produces exactly it (a knife-edge, a frictionless guide, a rigid frame that
   also stops sliding along the edge, ...).
3. Symmetry and sections. When the CAD is a cut of a larger part (a half, a quarter, a slice of a long
   body, the meridian section of a body of revolution), say what the full part is, that its geometry,
   supports and loads are mirror-symmetric (or the same at every angle around the axis), which faces of
   the CAD are cuts through the full part, and what a planar CAD represents: a thin plate loaded only in
   its own plane, a slice of a long body whose length cannot change, or the meridian section of a body of
   revolution with x = distance from the axis and y = position along it.
4. Forbidden (a lint rejects them): boundary condition(s), BC, fixed, clamped, pinned, roller, encastre,
   traction, symmetry boundary/condition/constraint, plane stress, plane strain, axisymmetric, rigid-body
   motion, and any statement that a displacement is zero or prescribed ("zero x-displacement", "u_x = 0",
   "displacement = 0"). Do not name the benchmark or where it comes from (NAFEMS, LE10, Cook, Lame, ...):
   an agent could recall its setup.
5. Refer to places by geometry (coordinates, planes, edges, faces, features such as "the outer rim" or
   "the face on the plane y = 0"), never by the group names in the list.
6. Keep everything else the statement gives, unchanged: the geometry description and what the CAD file
   holds, the units, every material constant, every load value, the planar thickness, the stress-free
   temperature, and the quantity to report (name, location, component). Write every number of the
   reference setup explicitly, in plain decimal or e-notation with its unit (210 GPa, 2.3e-4 per degC,
   1/16).
7. Say the analysis is static, linear and with small deformations, and that there is no gravity unless
   the reference setup has a body force.
8. Layout: short sections titled Component, Geometry (CAD), Material, Service loads, Installation and
   supports, Analysis scope, Required result. Plain text, no tables, no markdown headings.
9. group_aliases: for every group whose name gives a support or load away (it contains a word such as
   clamp, fix, load, sym, support, constrain, restrain, pressure, force, traction, anchor), a neutral
   geometric name (lowercase letters, digits and _, e.g. left_edge, face_y0) that is not the name of
   another group. Agents see the alias instead. Leave purely geometric names alone.
10. mapping: one entry per item of the reference setup (each boundary condition, each volume load, the
   formulation): the requirement sentence that implies it and why no other reading fits. Reviewers read
   this; agents never see it.
"""


class GroupAlias(pydantic.BaseModel):
    group: str = pydantic.Field(description="a group name from the list, exactly as written")
    alias: str = pydantic.Field(description="neutral geometric name: lowercase letters, digits and _")


class MappingItem(pydantic.BaseModel):
    reference: str = pydantic.Field(description="one item of the reference setup, e.g. 'symmetry on edge_AB'")
    implied_by: str = pydantic.Field(description="the requirement sentence that implies it, and why only that reading fits")


class RequirementsDraft(pydantic.BaseModel):
    requirements: str = pydantic.Field(description="the agent-facing requirements text")
    group_aliases: list[GroupAlias] = pydantic.Field(default_factory=list)
    mapping: list[MappingItem]


class BCItem(pydantic.BaseModel):
    type: Literal["fixed", "displacement", "symmetry", "pressure", "traction", "force"]
    group: str | None = None
    point: list[float] | None = None
    components: list[int] | None = None
    value: float | list[float] | str | None = None


class LoadItem(pydantic.BaseModel):
    type: Literal["temperature", "body_force"]
    value: float | list[float] | str
    group: str = "*"


class MaterialItem(pydantic.BaseModel):
    E: float
    nu: float
    alpha: float = 0.0


class SetupProposal(pydantic.BaseModel):
    reasoning: str = pydantic.Field(description="which requirement gives each item of the setup")
    formulation: Literal["plane_stress", "plane_strain", "axisymmetric", "3d"]
    thickness: float = 1.0
    material: MaterialItem
    bcs: list[BCItem]
    loads: list[LoadItem] = pydantic.Field(default_factory=list)
    T_ref: float = 0.0


def _backtranslator_system() -> str:
    from ace.agent import BC_HELP, TOOLS
    solve = next(t for t in TOOLS if t["name"] == "solve")
    loads = solve["input_schema"]["properties"]["loads"]["description"]
    return ("You set up linear-elastic finite-element analyses. You get a task the way an agent gets it: "
            "its engineering requirements, the scored quantity and the named groups of its CAD (what "
            "inspect_geometry returns). Propose the setup you would pass to the solve tool: formulation "
            "(plane_stress, plane_strain, axisymmetric or 3d), thickness for plane models, material, boundary "
            "conditions on the named groups (or a point), volume loads and the stress-free temperature T_ref. "
            "Give your reasoning first: which requirement gives each item.\n" + BC_HELP + "\nLoads: " + loads + "\n")


def translator_user(problem_id: str, feedback: str | None = None) -> str:
    from ace.checks import reference_setup
    raw = raw_problem(problem_id)
    q = fea.official_qoi(raw, scored_qoi_name(problem_id, raw))
    parts = [f"PROBLEM {problem_id}",
             "--- statement (spells the boundary conditions out) ---", raw.description(),
             "--- reference setup (the solver input the statement means; CAD group names) ---",
             json.dumps(reference_setup(raw), indent=1, default=str),
             "--- named groups of the CAD (name, kind, bounding box, size) ---",
             json.dumps(fea.geometry_summary(raw)["groups"]),
             f"--- scored quantity ---\n{q.name} = {q.type} of {q.field}"
             + (f" at point {q.point}" if q.point else f" over group {q.group!r}" if q.group else "")]
    if feedback:
        parts += ["--- problems with your previous draft: fix every one ---", feedback]
    return "\n".join(parts)


def draft_to_record(problem_id: str, draft: RequirementsDraft) -> dict:
    return {
        "problem_id": problem_id,
        "requirements": draft.requirements.strip(),
        "group_aliases": {a.group: a.alias for a in draft.group_aliases},
        "mapping": [m.model_dump() for m in draft.mapping],
    }


def translate(problem_id: str, llm, feedback: str | None = None, records: list | None = None) -> dict:
    draft, _ = llm.structured("translator", TRANSLATOR_SYSTEM, translator_user(problem_id, feedback),
                              RequirementsDraft, records)
    if draft is None:
        raise RuntimeError(f"{problem_id}: the translator gave no valid answer")
    return draft_to_record(problem_id, draft)


def agent_view(problem_id: str, data: dict) -> str:
    """What an agent sees of the task in requirements mode: label, preface, requirements, the scored
    quantity and the inspect_geometry result (aliased)."""
    problem = view_problem(problem_id, data.get("group_aliases") or {})
    q = fea.official_qoi(problem, scored_qoi_name(problem_id, problem))
    where = f" at point {q.point}" if q.point else (f" over group {q.group!r}" if q.group else "")
    return (f"Problem id: {label_for(problem_id)}\n\n{TASK_PREFACE}\n\n{data['requirements']}\n\n"
            f"SCORED QUANTITY: {q.name} = {q.type} of {q.field}{where}.\n\n"
            f"inspect_geometry returns:\n{json.dumps(fea.geometry_summary(problem))}")


def proposal_to_setup(p: SetupProposal) -> dict:
    return {
        "formulation": p.formulation, "thickness": p.thickness,
        "material": p.material.model_dump(),
        "bcs": [{k: v for k, v in b.model_dump().items() if v is not None} for b in p.bcs],
        "loads": [ld.model_dump() for ld in p.loads],
        "T_ref": p.T_ref,
    }


def round_trip_mesh(problem, workdir: Path) -> dict:
    """One quadratic mesh at the problem's own reference size: both setups are solved on it, so
    any difference in the scored quantity comes from the setup alone."""
    mesh = fea.make_mesh(problem, Path(workdir) / "round_trip.msh", h=problem.params["h0"], order=2)
    if not mesh["ok"]:
        raise RuntimeError(f"{problem.id}: round-trip mesh failed: {mesh.get('error')}")
    return mesh


def equivalence(problem_id: str, problem, setup: dict, mesh: dict) -> dict:
    """Does `setup` mean the same as the reference setup? Both are solved on `mesh`; equivalent when
    the scored quantity agrees to EQUIVALENCE_TOL. bc_report says how the setups differ."""
    from ace.checks import bc_report, reference_setup
    qoi = scored_qoi_name(problem_id, problem)
    ref = fea.run_solve(problem, mesh, reference_setup(problem), qoi)
    if not ref["ok"]:
        raise RuntimeError(f"{problem_id}: the reference setup failed on the round-trip mesh: {ref['error']}")
    out = {"setup_report": bc_report(problem, setup), "reference_value": ref["value"]}
    got = fea.run_solve(problem, mesh, setup, qoi)
    if not got["ok"]:
        return {**out, "equivalent": False, "error": got["error"]}
    rel = abs(got["value"] - ref["value"]) / abs(ref["value"]) if ref["value"] else abs(got["value"])
    return {**out, "equivalent": rel <= EQUIVALENCE_TOL, "rel_diff": rel, "value": got["value"]}


def round_trip(problem_id: str, llm, data: dict, trials: int = 3, records: list | None = None) -> dict:
    """`trials` independent read-backs of the requirements by the back-translator; each is checked
    for equivalence with the reference setup."""
    problem = view_problem(problem_id, data.get("group_aliases") or {})
    system, user = _backtranslator_system(), agent_view(problem_id, data)
    results = []
    with tempfile.TemporaryDirectory() as d:
        mesh = round_trip_mesh(problem, Path(d))
        for _ in range(trials):
            prop, _ = llm.structured("backtranslator", system, user, SetupProposal, records)
            if prop is None:
                results.append({"equivalent": False, "error": "the back-translator gave no valid answer"})
                continue
            setup = proposal_to_setup(prop)
            results.append({**equivalence(problem_id, problem, setup, mesh), "setup": setup,
                            "reasoning": prop.reasoning})
    return {"trials": trials, "recovered": sum(r["equivalent"] for r in results), "results": results}


def feedback_text(issues: list[str], rt: dict | None) -> str:
    lines = [f"- lint: {i}" for i in issues]
    for k, r in enumerate((rt or {}).get("results", [])):
        if r["equivalent"]:
            continue
        rep = r.get("setup_report") or {}
        how = [r.get("error") or "",
               f"missing {rep['missing']}" if rep.get("missing") else "",
               f"unexpected {rep['unexpected']}" if rep.get("unexpected") else "",
               f"reversed {rep['sign_flips']}" if rep.get("sign_flips") else "",
               "different formulation" if rep and not rep.get("formulation_ok", True) else "",
               "different thickness" if rep and not rep.get("thickness_ok", True) else "",
               "different material" if rep and not rep.get("material_ok", True) else "",
               f"scored quantity off by {r['rel_diff']:.2%}" if "rel_diff" in r else ""]
        lines.append(f"- reader {k + 1} set the problem up differently from the reference: "
                     + "; ".join(h for h in how if h) + f". Its reasoning: {(r.get('reasoning') or '')[:800]}")
    return "\n".join(lines)


def generate(problem_id: str, llm, trials: int = 3, revisions: int = 2, records: list | None = None) -> dict:
    """translate -> lint -> round trip, revising up to `revisions` times. Returns the record to save
    (with its validation), whether or not it ended clean: `check` and the loop refuse unclean ones."""
    feedback, data, issues, rt = None, {}, [], None
    for attempt in range(revisions + 1):
        data = translate(problem_id, llm, feedback, records)
        issues = lint(problem_id, data)
        rt = round_trip(problem_id, llm, data, trials, records) if not issues else None
        print(f"  {problem_id} draft {attempt + 1}: {len(issues)} lint issue(s)"
              + (f", round trip {rt['recovered']}/{rt['trials']}" if rt else ""))
        if not issues and rt["recovered"] == rt["trials"]:
            break
        feedback = feedback_text(issues, rt)
    data["source"] = {"method": "ace.requirements generate", "model": config.MODELS["translator"],
                      "reader_model": config.MODELS["backtranslator"],
                      "created": datetime.date.today().isoformat(), "fingerprint": fingerprint(problem_id)}
    data["validation"] = {"lint": issues, "round_trip": _summary(rt), "drafts": attempt + 1}
    return data


def _summary(rt: dict | None) -> dict | None:
    """The round trip without the readers' full setups (kept short in the stored file)."""
    if rt is None:
        return None
    keep = ("equivalent", "rel_diff", "error", "setup_report")
    return {"trials": rt["trials"], "recovered": rt["recovered"],
            "results": [{k: r[k] for k in keep if k in r} for r in rt["results"]]}


def problems_ready(problem_ids: list[str]) -> list[str]:
    """Why the loop cannot run these problems in requirements mode ([] = all ready)."""
    out = []
    for pid in problem_ids:
        if not path_for(pid).exists():
            out.append(f"{pid}: no requirements file ({path_for(pid)}); "
                       f"python -m ace.requirements generate --problems {pid}")
            continue
        out += [f"{pid}: {i}" for i in lint(pid, load(pid))]
    return out


# --------------------------------------------------------------------------- CLI


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate", help="write requirements with the translator and validate them")
    c = sub.add_parser("check", help="lint the stored requirements (and --round-trip them)")
    s = sub.add_parser("show", help="print a problem's task prompt as the agent sees it in requirements mode")
    s.add_argument("problem")
    for p in (g, c):
        p.add_argument("--problems", default=",".join(TASKS), help="comma-separated problem ids")
        p.add_argument("--mock", action="store_true", help="canned LLM answers, no API calls")
        p.add_argument("--trials", type=int, default=3, help="back-translations per round trip")
    g.add_argument("--revisions", type=int, default=2, help="re-drafts after a failed lint or round trip")
    g.add_argument("--force", action="store_true", help="rewrite problems whose file is present and current")
    c.add_argument("--round-trip", action="store_true", help="also re-run the round trip (LLM calls)")
    args = ap.parse_args(argv)

    if args.cmd == "show":
        config.STATEMENT = "requirements"
        from ace.agent import task_prompt
        print(task_prompt(args.problem))
        return 0

    from ace.llm import make_llm
    llm = make_llm(args.mock)
    records: list[dict] = []
    failed = False
    for pid in [p for p in args.problems.split(",") if p]:
        if args.cmd == "generate":
            if path_for(pid).exists() and not args.force and not lint(pid, load(pid)):
                print(f"{pid}: present and clean (use --force to rewrite)")
                continue
            data = generate(pid, llm, args.trials, args.revisions, records)
            print(f"{pid}: wrote {save(pid, data)}")
            v = data["validation"]
            failed |= bool(v["lint"]) or v["round_trip"] is None or v["round_trip"]["recovered"] < args.trials
        else:
            if not path_for(pid).exists():
                print(f"{pid}: MISSING")
                failed = True
                continue
            data = load(pid)
            issues = lint(pid, data)
            line = f"{pid}: " + ("lint clean" if not issues else "LINT: " + " | ".join(issues))
            failed |= bool(issues)
            if args.round_trip:
                rt = round_trip(pid, llm, data, args.trials, records)
                line += f"; round trip {rt['recovered']}/{rt['trials']}"
                failed |= rt["recovered"] < rt["trials"]
                if rt["recovered"] < rt["trials"]:
                    line += "\n" + feedback_text([], rt)
            else:
                stored = (data.get("validation") or {}).get("round_trip")
                line += ("; round trip not run yet" if not stored
                         else f"; stored round trip {stored['recovered']}/{stored['trials']}")
            print(line)
    if records and not args.mock:
        print(f"LLM cost: ${sum(r['usd'] for r in records):.2f} over {len(records)} calls")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
