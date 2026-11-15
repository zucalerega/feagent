"""What a role reads: its instructions and the part of the dossier it needs, as one Markdown document.

    build(project, role, focus=(), max_chars=60000)      the brief: instructions + the dossier
    dossier(project, role, focus=(), max_chars=60000)    the dossier part only (the user message of an API call)
    pictures(project, role)                              the picture files the brief names: [(path, caption)]

The dossier part is complete and compact: tables and one-line records with the record ids visible (in
`code`), numbers with their units as dossier.units shows them, never raw JSON. It says plainly when a
stage has not run. Picture files are named by their absolute path, so an agent with file access can open
them. The data come from dossier.context (the evaluated numbers and effects), Project.collection (the
records of the stages) and dossier.trace (what rests on what: the "used by" column of the named numbers).

Sections (SECTIONS; each role lists the ones it reads in dossier.agents.roles):
    status, project, system, phases, effects, numbers, component, hand_calc_specs, requirements, policy,
    material, geometry, findings, plan, analyses, compliance, previous

Size: the brief aims at `max_chars`. When the full form is longer, the builder writes the compact form
(the same records, fewer columns, results that no criterion or hand calculation uses as a count) and says
so at the top. `focus` (ids of load cases or analyses) keeps those in full and the other ones on one line:
a reading aid for one analysis at a time. Nothing is cut without a sentence that says what was left out.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Iterable

from dossier import context as context_mod
from dossier import model, trace, units
from dossier.agents import roles as roles_mod
from dossier.project import STAGE_FILES, Project

MAX_CHARS = 60_000


# --------------------------------------------------------------------------- small helpers


def _c(text: Any) -> str:
    """One line for a table cell or a list item."""
    return " ".join(str(text if text is not None else "").split()).replace("|", "\\|")


def _id(i: Any) -> str:
    return f"`{i}`" if i else ""


def _ids(items: Iterable[Any], limit: int | None = None) -> str:
    items = list(items)
    shown = items if limit is None else items[:limit]
    more = f" (+{len(items) - len(shown)})" if len(items) > len(shown) else ""
    return ", ".join(_id(i) for i in shown) + more


def _table(head: list[str], rows: list[list[Any]]) -> list[str]:
    if not rows:
        return []
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    out += ["| " + " | ".join(_c(x) for x in r) + " |" for r in rows]
    return out


def _show(si: Any, unit: str | None) -> str:
    """An SI number of a record for people: 143200000, "Pa" -> "143.2 MPa"."""
    if si is None:
        return ""
    try:
        return units.display(units.from_si(si, unit or ""))
    except Exception:  # noqa: BLE001  a unit text that pint cannot read
        return f"{si:g} {unit or ''}".strip() if isinstance(si, (int, float)) else f"{si} {unit or ''}".strip()


def _series(values: list[float], unit: str | None) -> str:
    """Values on the meshes, coarse to fine: "139.1 → 141.9 → 143.2 MPa" (the unit once when it is the same)."""
    parts = [_show(v, unit) for v in values]
    tails = {p.split(" ", 1)[1] if " " in p else "" for p in parts}
    if len(parts) > 1 and len(tails) == 1:
        tail = tails.pop()
        return " → ".join(p.split(" ", 1)[0] for p in parts) + (f" {tail}" if tail else "")
    return " → ".join(parts)


def _value(v: model.Value | None, names: dict | None = None) -> str:
    """A quantity of an input file in the form the dossier shows (dossier.units.display); an expression
    with its value."""
    if v is None:
        return ""
    if v.is_literal:
        try:
            return units.display(v.quantity())
        except Exception:  # noqa: BLE001
            return f"{v.value} {v.unit}".strip()
    expr = _expr(v)
    try:
        return f"{units.display(v.quantity(names or {}))} (= {expr})"
    except Exception:  # noqa: BLE001  an expression of a load case: load(), feat()
        return f"`{expr}` (evaluated in the load case)"


def _expr(v: model.Value | None) -> str:
    if v is None or v.expr is None:
        return ""
    if isinstance(v.expr, list):
        return "[" + ", ".join(f"{x:g}" if isinstance(x, (int, float)) else str(x) for x in v.expr) + "]"
    return str(v.expr)


def _s(text: Any) -> str:
    """A text of an input file inside a sentence of the brief: one line, no full stop at its end."""
    return _c(text).rstrip(". ")


def _pct(x: float | None) -> str:
    return "" if x is None else f"{100 * x:.3g} %"


def _num(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, (int, float)):
        return f"{x:.4g}"
    if isinstance(x, (list, tuple)):
        return "[" + ", ".join(_num(v) for v in x) + "]"
    if isinstance(x, dict):
        return "{" + ", ".join(f"{k}: {_num(v)}" for k, v in x.items()) + "}"
    return str(x)


def _same(a: Any, b: Any, rtol: float = 1e-3) -> bool:
    """Two SI values (numbers or vectors) are the same within rtol of the larger magnitude."""
    va = list(a) if isinstance(a, (list, tuple)) else [a]
    vb = list(b) if isinstance(b, (list, tuple)) else [b]
    if len(va) != len(vb) or not all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in (*va, *vb)):
        return False
    big = max(abs(x) for x in (*va, *vb)) or 1.0
    return all(abs(x - y) <= rtol * big for x, y in zip(va, vb))


def _yes(x: bool | None, unknown: str = "not evaluated") -> str:
    return unknown if x is None else ("yes" if x else "NO")


# --------------------------------------------------------------------------- the builder


class _Brief:
    def __init__(self, project: Project, role: roles_mod.Role, focus: Iterable[str] = (), compact: bool = False):
        self.p = project
        self.role = role
        self.focus = set(focus or ())
        self.compact = compact
        # evaluated now from the input files (not read from out/context.json): the numbers are the same, and
        # the display texts are then the ones of this version of dossier.units
        self.ctx = context_mod.build(project)
        self.names = self.ctx.names()
        self.comp = project.subject
        self.inputs = project.inputs()
        doc = project.read_stage("geometry")
        self.geo: dict | None = doc.get("geometry") if doc else None
        self.graph = trace.graph(project, self.ctx)
        self.used_by: dict[str, list[str]] = {}
        for e in self.graph["edges"]:
            if e["rel"] in ("derives_from", "uses") and e["from"] not in self.used_by.setdefault(e["to"], []):
                self.used_by[e["to"]].append(e["from"])
        for r in project.reqs():                         # an explicit check reads context names too
            for prm in r.parameters.values():
                for n in units.names_in(prm.check.against) if prm.check else ():
                    if n in self.ctx.params and r.id not in self.used_by.setdefault(n, []):
                        self.used_by[n].append(r.id)
        self.rec = {k: project.collection(k) for k in ("findings", "decisions", "assumptions", "load_cases",
                                                         "hand_calcs", "analyses", "compliance", "verdicts")}
        self.reviews = project.reviews()
        self.ran = {s: project.stage_path(s).exists() for s in STAGE_FILES}
        self.pictures: list[tuple[Path, str]] = []
        self.left_out: list[str] = []          # what the compact form leaves out, in words

    # ---- records of other origins and of this role
    def own(self, rec: Any) -> bool:
        return roles_mod.role_of(rec) == self.role.name

    def others(self, key: str) -> list:
        return [r for r in self.rec[key] if not self.own(r)]

    def review(self, rec: Any) -> str:
        rv, stale = self.p.review_of(rec, self.reviews)
        if rv.state == "open" and not stale:
            return "open"
        text = rv.state + (" (the record changed after the review)" if stale else "")
        return text + (f" by {rv.by}" if rv.by else "") + (f": \"{_c(rv.comment)}\"" if rv.comment else "")

    def picture(self, rel: str, caption: str = "") -> str:
        path = (self.p.out / rel).resolve()
        if (path, caption) not in self.pictures:
            self.pictures.append((path, caption))
        return f"- `{path}`" + (f": {_c(caption)}" if caption else "") + ("" if path.exists() else " (the file is missing)")

    # ---- one record on one line
    def finding(self, f: model.Finding, full: bool = True) -> str:
        out = (f"- {_id(f.id)} [{f.severity}, {f.category}; {f.origin.by}; review: {self.review(f)}] about "
               f"{_id(f.subject)}: **{_c(f.title)}** {_c(f.statement)}")
        if full:
            out += f" Evidence: {_c(f.rationale)}" + (f" Suggestion: {_c(f.suggestion)}" if f.suggestion else "")
        return out

    def decision(self, d: model.Decision) -> str:
        alts = "; ".join(f"{_c(a.option)} (not chosen: {_c(a.why_not)})" for a in d.alternatives)
        return (f"- {_id(d.id)} [decision, {d.stage}, basis {d.basis}{', needs review' if d.needs_review else ''}; "
                f"{d.origin.by}; review: {self.review(d)}] about {_id(d.subject)}: **{_c(d.title)}** "
                f"Choice: {_c(d.choice)} Why: {_c(d.rationale)}" + (f" Alternatives: {alts}" if alts else ""))

    def assumption(self, a: model.Assumption) -> str:
        cons = {True: "conservative", False: "NOT conservative", None: "conservative: not known"}[a.conservative]
        return (f"- {_id(a.id)} [assumption, {a.stage}, {cons}; {a.origin.by}; review: {self.review(a)}] about "
                f"{_id(a.subject)}: **{_c(a.title)}** Assumes: {_c(a.statement)} Why: {_c(a.rationale)} "
                f"If wrong: {_c(a.impact)}" + (f" Check: {_c(a.verify_by)}" if a.verify_by else ""))

    def journal(self, keep: Callable[[Any], bool]) -> list[str]:
        out = [self.decision(d) for d in self.others("decisions") if keep(d)]
        out += [self.assumption(a) for a in self.others("assumptions") if keep(a)]
        return out

    # =================================================================== sections

    def status(self) -> list[str]:
        p, r = self.p, self.rec
        n_an = len(r["analyses"])
        by_status: dict[str, int] = {}
        for a in r["analyses"]:
            by_status[a.status] = by_status.get(a.status, 0) + 1
        rule_findings = [f for f in r["findings"] if roles_mod.role_of(f) is None]
        rows = [
            ["context", f"{len(self.ctx.params)} named numbers and {len(self.ctx.effects)} effects, evaluated from the "
                        f"input files"],
            ["geometry", (f"has run: {len(self.geo.get('features', {}))} features measured, "
                          f"{len(self.geo.get('problems', []))} problem(s)") if self.geo else "has NOT run"],
            ["validate", f"has run: {len(rule_findings)} finding(s) of the rules and engines" if self.ran["validate"]
             else "has NOT run: the rules made no findings yet"],
            ["plan", (f"has run: {len(r['load_cases'])} load case(s), {len(r['hand_calcs'])} hand calculation(s)")
             if self.ran["plan"] else "has NOT run: there is no load-case plan"],
            ["analyze", (f"{n_an} analysis record(s) for {len(r['load_cases'])} load case(s): "
                         + ", ".join(f"{k} {v}" for k, v in sorted(by_status.items()))) if n_an
             else "has NOT run: there is no analysis"],
            ["assess", f"has run: {len(r['compliance'])} compliance item(s), {len(r['verdicts'])} verdict(s)"
             if self.ran["assess"] else "has NOT run: there are no margins"],
        ]
        out = ["### Status of the stages", "", *_table(["stage", "state"], rows), ""]
        stale = p.stale()
        if stale:
            out += [f"STALE: the input files changed after these stage files were written: {', '.join(stale)}. "
                    f"Their records can be out of date.", ""]
        agents = sorted({roles_mod.role_of(x) for k in ("findings", "decisions", "assumptions") for x in r[k]} - {None})
        if agents:
            out += [f"Roles that have a submission in the dossier: {', '.join(agents)}.", ""]
        name = self.role.name
        if name == "requirements-reviewer" and not self.ran["validate"]:
            out += ["The validate stage has not run, so this brief has no rule-made findings. A finding of yours can "
                    "repeat what the rules find later. If you can, run the pipeline to the validate stage first "
                    f"(`python -m dossier run {p.root} --to validate`).", ""]
        if name == "plan-reviewer" and not self.ran["plan"]:
            out += ["The plan stage has not run. There is no load-case plan to review. Do not write about a plan that "
                    "you do not see. You can submit empty lists, or findings about what the plan will use: the "
                    "phases, the effects and the hand-calculation models of the component.", ""]
        if name == "analysis-checker" and not n_an:
            out += ["No analysis has run. There is nothing to check. Do not write about results that you do not see. "
                    "Submit empty lists.", ""]
        return out

    def project(self) -> list[str]:
        f, c = self.p.file, self.comp
        return ["### Project", "",
                f"{_id(f.id)}: {_s(f.name)}. {_c(f.description)}", "",
                f"The component under verification is {_id(c.id)} ({_c(c.name)}). It realizes the element "
                f"{_id(c.element)} of the system {_id(self.p.system.id)} ({_c(self.p.system.name)}).", ""]

    def system(self) -> list[str]:
        s, c = self.p.system, self.comp
        by_conn = {it.connection: it for it in c.interfaces}
        out = ["### System", "", _c(s.description), ""]
        if self.role.name == "requirements-reviewer":
            out += _table(["element", "name", "kind", "group", "description"],
                          [[_id(e.id) + (" (the component)" if e.id == c.element else ""), e.name, e.kind, e.group,
                            e.description] for e in s.elements])
        else:
            out.append("Elements: " + "; ".join(f"{_id(e.id)} {_s(e.name)}" + (" (the component)" if e.id == c.element else "")
                                               for e in s.elements) + ".")
        out += ["", "Connections. A connection carries effects: what it does to an element in a phase.", ""]
        rows = []
        for cn in s.connections:
            on_subject = [e for e in cn.effects if (e.acts_on or cn.to) == c.element and e.kind != "signal"]
            it = by_conn.get(cn.id)
            rows.append([_id(cn.id), cn.name, f"{_id(cn.from_)} → {_id(cn.to)}",
                         cn.kind + (f", {cn.medium}" if cn.medium else ""),
                         _id(it.id) if it else "",
                         (f"{len(cn.effects)}, {len(on_subject)} of them on the component" if cn.effects else "none"),
                         cn.description])
        out += _table(["connection", "name", "from → to", "kind", "interface of the component", "effects",
                       "description"], rows)
        return out + [""]

    def phases(self) -> list[str]:
        p, s = self.p, self.p.system
        direct: dict[str, list[str]] = {ph.id: [] for ph in s.phases}
        cyclic: dict[str, list[str]] = {ph.id: [] for ph in s.phases}
        dcs = {d.id: d for d in s.duty_cycles}
        for r in p.reqs():
            for ph in p.phases_of(r):
                direct.setdefault(ph, []).append(r.id)
            for d in r.applies.duty_cycles:
                for ph in dcs[d].sequence if d in dcs else ():
                    if r.id not in cyclic.setdefault(ph, []):
                        cyclic[ph].append(r.id)
        rows = []
        for ph in s.phases:
            reqs = "; ".join(x for x in (_ids(direct[ph.id]), "through a duty cycle: " + _ids(cyclic[ph.id])
                                         if cyclic[ph.id] else "") if x)
            rows.append([_id(ph.id), ph.name, ph.category, _value(ph.duration) if ph.duration else "",
                         ", ".join(f"{k}: {v}" for k, v in ph.state.items()), reqs or "NONE", ph.description])
        out = ["### Phases", "",
               "A phase is an operating state of the system. The effects of one phase act together.", "",
               *_table(["phase", "name", "category", "duration", "state", "requirements that apply", "description"],
                       rows), ""]
        nowhere = [r.id for r in p.reqs() if not p.phases_of(r) and not r.applies.duty_cycles]
        if nowhere:
            out += [f"Requirements that name no phase and no duty cycle: {_ids(nowhere)}.", ""]
        if s.duty_cycles:
            out += ["Duty cycles (a sequence of phases that repeats in the life of the part):", ""]
            out += _table(["duty cycle", "name", "sequence", "repetitions", "source", "description"],
                          [[_id(d.id), d.name, " → ".join(_id(x) for x in d.sequence),
                            f"{self.ctx.duty_cycles.get(d.id, 0):g}" + (f" (= {_expr(d.count)})" if d.count.expr else ""),
                            d.source, " ".join(x for x in (d.description, d.note) if x)] for d in s.duty_cycles])
            out.append("")
        return out

    def _effect_rows(self, phase: str) -> list[list[str]]:
        p, c = self.p, self.comp
        loads = {x.effect: x for x in self.ctx.loads(p, c.id, phase)}
        rows = []
        for e in self.ctx.effects_on(c.element, phase):
            if e.kind == "signal":
                continue
            item = loads.get(e.id)
            src = self.inputs.get(e.id, (None, None))[1]
            where = (f"{_id(item.interface)}: {item.role}" if item is not None and item.interface
                     else (item.role if item is not None else ""))
            value = "holds the component" if e.kind == "support" else e.display
            frm = _expr(getattr(src, "value", None))
            if e.h_si is not None:
                value += f", h = {_show(e.h_si, 'W/(m^2*K)')}"
                frm += (", " if frm else "") + "h = " + (_expr(getattr(src, "h", None)) or "a number")
            if e.at_si is not None:
                value += f", at {_show(e.at_si, 'm')}"
            extra = "".join(f" [{k}: {v}]" for k, v in (("basis", e.basis if e.basis != "specified" else ""),
                                                        ("source", e.source), ("note", e.note)) if v)
            rows.append([_id(e.id), where, e.kind, value, frm, e.description + extra])
        return rows

    def effects(self) -> list[str]:
        c = self.comp
        if self.role.name == "analysis-checker" and self.rec["load_cases"]:
            return ["### Effects on the component", "",
                    "Each analysis below gives its loads with their effect ids and values. The plan reviewer checks "
                    "the load cases against the effects of the phases; the full tables are in its brief "
                    f"(`python -m dossier.agents brief {self.p.root} plan-reviewer`).", ""]
        out = ["### Effects on the component, phase by phase", "",
               f"Each row is one effect on {_id(c.element)}: what a connection does to the component in the phase. "
               "\"from\" is the expression that gives the value: its names are in the table of the named numbers, "
               "with their sources. Vectors are [x, y, z] in the CAD frame of the component. A requirement gives no "
               "load: the loads come from these effects.", ""]
        for ph in self.p.system.phases:
            rows = self._effect_rows(ph.id)
            out.append(f"**{_id(ph.id)} {_c(ph.name)}** ({ph.category})" + ("" if rows else ": no effect on the component."))
            out += ["", *_table(["effect", "on (interface: role)", "kind", "value", "from", "description"], rows), ""] \
                if rows else [""]
        return out

    def numbers(self) -> list[str]:
        params = self.ctx.params
        system = [x for x in params.values() if x.origin == "system"]
        stated = [x for x in params.values() if x.origin in {r.id for r in self.p.reqs()}]
        material = [x for x in params.values() if x not in system and x not in stated]
        out = ["### Named numbers", ""]
        if self.role.name != "requirements-reviewer":
            out += ["Every named number of the project, with its expression when it is derived:", "",
                    "; ".join(f"{_id(x.name)} = {x.display}" + (f" (= {x.expr})" if x.expr else "")
                              for x in [*system, *stated, *material]) + ".", ""]
            return out
        out += ["Numbers of the system context. Basis: specified (a document gives it), measured, derived (an "
                "expression of other numbers), assumed (no proof), reference (a handbook). \"used by\" lists what "
                "rests directly on the number.", ""]
        out += _table(["name", "value", "expression", "basis", "source", "title and note", "used by"],
                      [[_id(x.name), x.display, x.expr or "", x.basis, x.source or ("NO SOURCE" if not x.expr else ""),
                        x.title + (f". Note: {x.note}" if x.note else ""),
                        _ids(self.used_by.get(x.name, []), 4) or "nothing"] for x in system])
        out += ["", "Numbers that the requirements state. A number of a requirement is a claim; the rules compare it "
                "with the context when it has a role or a check.", ""]
        rows = []
        for x in stated:
            prm = self.p.requirement(x.origin).parameters[x.name]
            chk = (f"{x.name} {prm.check.relation} {prm.check.against}"
                   + (f" ({_c(prm.check.why)})" if prm.check.why else "")) if prm.check else ""
            rows.append([_id(x.name), x.display, _id(x.origin), prm.role or "", chk,
                         _ids(self.used_by.get(x.name, []), 4) or "nothing"])
        out += _table(["name", "value", "stated by", "role", "explicit check", "used by"], rows)
        out += ["", "Material properties as names of expressions: "
                + "; ".join(f"{_id(x.name)} = {x.display}" for x in material) + ".", ""]
        return out

    def component(self) -> list[str]:
        c, geo = self.comp, self.geo or {}
        out = ["### Component", "",
               f"{_id(c.id)}: {_c(c.name)}, drawing {_c(c.drawing) or 'not given'}, material {_id(c.material)}, "
               f"stress-free at {_value(c.reference_temperature)}, CAD file `{self.p.cad_path(c.id)}`.", "",
               _c(c.description), ""]
        if c.notes:
            out += [f"Notes: {_c(c.notes)}", ""]
        feats = geo.get("features", {})
        rows = []
        for fid, f in c.features.items():
            g = feats.get(fid, {})
            measured = ", ".join(x for x in (
                f"{len(g.get('faces', []))} face(s)" if g.get("faces") else "",
                f"area {_show(g.get('area'), 'm^2')}" if g.get("area") else "",
                f"radius {_show(g.get('radius'), 'm')}" if g.get("radius") else "") if x)
            rows.append([_id(fid), f.description, measured] if geo else [_id(fid), f.description])
        out += ["Features (named sets of CAD faces" + ("" if geo else "; the geometry stage has not measured them") + "):",
                "", *_table(["feature", "description", "measured"] if geo else ["feature", "description"], rows), ""]
        out += ["Interfaces (how the component realizes a connection; the role says what the faces do: `wetted` "
                "takes pressure and fluid temperature, `seat` takes forces, moments and supports, `bolts`, "
                "`external`):", ""]
        for it in c.interfaces:
            roles = "; ".join(f"{role}: {_ids(fs)}" for role, fs in it.features.items())
            out.append(f"- {_id(it.id)} realizes {_id(it.connection)}, joint: {it.joint or 'not given'}. **{_c(it.name)}.** "
                       f"{_c(it.description)} Faces: {roles}.")
        out.append("")
        if c.scls:
            ok = geo.get("scls", {})
            out += ["Stress classification lines (straight lines through a wall; the criteria at `scls` use them):", ""]
            out += _table(["line", "description", "start → end"] + (["geometry stage"] if ok else []),
                          [[_id(s.id), s.description, f"{_show(list(s.start), 'm')} → {_show(list(s.end), 'm')}"]
                           + ([("ok" if ok[s.id].get("ok") else f"PROBLEM: {ok[s.id].get('message', '')}")
                               if s.id in ok else ""] if ok else []) for s in c.scls])
            out.append("")
        if c.hand_calcs and "hand_calc_specs" not in self.role.sections:
            out += ["Hand-calculation models of the component: "
                    + "; ".join(f"{_id(h.id)} ({h.model}: {_c(h.title)})" for h in c.hand_calcs) + ".", ""]
        if c.flow is not None:
            out += [f"Flow domain: {_c(c.flow.description)} Fluid: "
                    + ", ".join(f"{k} = {_value(v, self.names)}" for k, v in c.flow.fluid.items()) + ".", ""]
        return out

    def hand_calc_specs(self) -> list[str]:
        c = self.comp
        if not c.hand_calcs:
            return ["### Hand-calculation models", "", "The component has no hand-calculation model.", ""]
        out = ["### Hand-calculation models of the component", "",
               "A closed-form model of a region of the part. The pipeline compares its output with a result of the "
               "analysis; the band is the accepted ratio analysis / hand calculation "
               f"(default of the policy: {self.p.policy.handcalc_band[0]:g} to {self.p.policy.handcalc_band[1]:g}).", ""]
        for h in c.hand_calcs:
            when = "; ".join(x for x in (
                f"analysis types: {', '.join(h.when.analysis)}" if h.when.analysis else "",
                f"needs the loads: {', '.join(h.when.needs)}" if h.when.needs else "",
                f"phases: {_ids(h.when.phases)}" if h.when.phases else "") if x) or "any load case"
            inputs = "; ".join(f"{k} = {_value(v, self.names)}" for k, v in h.inputs.items())
            out.append(f"- {_id(h.id)} [model `{h.model}`] **{_c(h.title)}.** {_c(h.description)} Applies to: {when}. "
                       f"Inputs: {inputs}.")
            for cmp in h.compare:
                band = (f"band {cmp.band[0]:g} to {cmp.band[1]:g}" if cmp.band else
                        f"tolerance {_value(cmp.tolerance, self.names)}" if cmp.tolerance else "the default band")
                out.append(f"  - compares `{cmp.output}` with `{cmp.with_}`, {band}. Why: {_c(cmp.why) or 'not given'}")
        return out + [""]

    def _criterion(self, r: model.Requirement, c: model.Criterion) -> str:
        if isinstance(c.factor, str):
            fp = self.ctx.params.get(c.factor)
            pf = self.p.policy.factors.get(c.factor)
            factor = f"{c.factor} ({fp.display if fp else (f'{pf.value:g}, policy' if pf else 'NOT A NAME OF THE PROJECT')}) × "
        else:
            factor = "" if c.factor == 1.0 else f"{c.factor:g} × "
        if c.allowable:
            v = getattr(self.p.material_of(), c.allowable, None)
            limit = f"{c.allowable} of the material" + (f" ({_value(v)})" if isinstance(v, model.Value) else "")
        else:
            limit = _value(c.limit, self.names)
        return (f"`{c.id or 'criterion'}`: {factor}`{c.key()}` {c.comparator} {limit}"
                + (f" (note: {_s(c.note)})" if c.note else ""))

    def requirements(self) -> list[str]:
        p = self.p
        full = self.role.name == "requirements-reviewer"
        doc = p.requirement_docs.get(self.comp.id, "")
        out = ["### Requirements" + (f" ({_c(doc)})" if doc else ""), ""]
        if full:
            out += ["Each requirement in full: the source text as written, then what the pipeline reads from it. A "
                    "criterion is `factor × response <= limit`; the response is a result key "
                    "`quantity.measure@location` of the analysis of each phase the requirement applies to. A "
                    "requirement with no criterion gets no margin.", ""]
        verdicts = {v.requirement: v for v in self.rec["verdicts"]}
        for r in p.reqs():
            phases = p.phases_of(r)
            applies = "; ".join(x for x in (
                f"category {r.applies.category}" if r.applies.category else "",
                f"phases {_ids(phases)}" if phases else "",
                f"duty cycles {_ids(r.applies.duty_cycles)}" if r.applies.duty_cycles else "") if x) or "no phase"
            crit = [self._criterion(r, c) for c in r.criteria]
            v = verdicts.get(r.id)
            verdict = (f" Verdict of the assess stage: {v.verdict}"
                       + (f", margin {v.margin:+.3g}" if v.margin is not None else "") + ".") if v else ""
            if not full:
                out.append(f"- {_id(r.id)} [{r.kind}; verification by {r.verification}] **{_c(r.title)}.** "
                           f"\"{_c(r.text)}\" Applies to: {applies}. Criteria: {'; '.join(crit) or 'NONE'}.{verdict}")
                continue
            out += [f"#### {_id(r.id)} {_c(r.title)}", "",
                    f"- Text: \"{_c(r.text)}\"",
                    f"- Kind: {r.kind}. Verification by: {r.verification}. Source: {_c(r.source) or 'NOT GIVEN'}."
                    + (f" Parent: {_id(r.parent)}." if r.parent else ""),
                    f"- Rationale of the specification: {_c(r.rationale) or 'not given'}"]
            if r.parameters:
                out.append("- Numbers it states: " + "; ".join(
                    f"{_id(k)} = {_value(v_, self.names)}" + (f" (role `{v_.role}`)" if v_.role else "")
                    + (f" [check: {k} {v_.check.relation} {v_.check.against}]" if v_.check else "")
                    + (f" [note: {_c(v_.note)}]" if v_.note else "") for k, v_ in r.parameters.items()) + ".")
            out.append(f"- Applies to: {applies}.")
            out.append("- Criteria:" + ("" if crit else " NONE (the pipeline gets no margin from this requirement)"))
            out += [f"  - {x}" for x in crit]
            if verdict:
                out.append(f"-{verdict}")
            out.append("")
        return out + ([] if full else [""])

    def policy(self) -> list[str]:
        pol = self.p.policy
        out = ["### Factor policy", "",
               f"{_id(pol.id)}: {_s(pol.name)}. Source: {_s(pol.source) or 'NOT GIVEN'}." + (f" {_c(pol.note)}" if pol.note else ""),
               ""]
        out += _table(["factor", "value", "source", "note"],
                      [[f"`{k}`", f"{f.value:g}", f.source, f.note] for k, f in pol.factors.items()])
        out += ["", f"Accepted ratio analysis / hand calculation: {pol.handcalc_band[0]:g} to {pol.handcalc_band[1]:g}. "
                f"A result with a discretization uncertainty above {_pct(pol.max_uncertainty)} is not converged. A phase "
                f"whose temperatures differ from the stress-free temperature by more than {_value(pol.thermal_delta)} "
                f"needs a thermal analysis.", ""]
        return out

    def material(self) -> list[str]:
        m = self.p.material_of()
        props = [("E", m.E), ("density", m.rho), ("thermal expansion", m.alpha), ("thermal conductivity", m.k),
                 ("specific heat", m.cp), ("yield strength", m.yield_strength), ("ultimate strength", m.ultimate_strength)]
        out = ["### Material", "",
               f"{_id(m.id)}: {_s(m.name)}, specification {_s(m.spec) or 'not given'}. Source of the numbers: "
               f"{_s(m.source) or 'NOT GIVEN'}." + (f" Note: {_c(m.note)}" if m.note else ""), "",
               "Properties: " + "; ".join(f"{k} = {_value(v)}" for k, v in props if v is not None)
               + f"; Poisson's ratio = {m.nu:g}."
               + (f" The properties apply at {_value(m.properties_temperature)}." if m.properties_temperature else "")
               + (f" Allowed service temperature: {_value(m.service_temperature[0])} to "
                  f"{_value(m.service_temperature[1])}." if m.service_temperature else "")
               + (" Compatibility: " + ", ".join(f"{k}: {v}" for k, v in m.compatibility.items()) + "."
                  if m.compatibility else " Compatibility: not given."), ""]
        if m.fatigue:
            f = m.fatigue
            out += [f"Fatigue: {f.model}, sigma_f = {_value(f.sigma_f)}, b = {f.b:g}, mean stress: {f.mean_stress}"
                    + (f", endurance limit {_value(f.endurance_limit)}" if f.endurance_limit else "")
                    + f". Source: {_s(f.source) or 'NOT GIVEN'}." + (f" Note: {_c(f.note)}" if f.note else ""), ""]
        return out

    def geometry(self) -> list[str]:
        geo = self.geo
        if not geo:
            return ["### Geometry stage", "", "The geometry stage has not run: the features are not measured on the CAD.", ""]
        cad = geo.get("cad", {})
        out = ["### Geometry stage", "",
               f"CAD: {cad.get('n_faces', '?')} faces, volume {_show(cad.get('volume'), 'm^3')}, mass "
               f"{_show(cad.get('mass'), 'kg')}."]
        if geo.get("problems"):
            out += ["", "Problems that the geometry stage reports:"] + [f"- {_c(x)}" for x in geo["problems"]]
        edges = (geo.get("edges") or {}).get("reentrant") or []
        if edges:
            out += ["", f"The CAD has {len(edges)} sharp re-entrant edge(s). A stress peak at such an edge is singular."]
        pics = [self.picture(x.get("path", ""), x.get("caption", "")) for x in geo.get("pictures") or [] if x.get("path")]
        if (self.p.out / "system.svg").exists():
            pics.append(self.picture("system.svg", "The system diagram."))
        if pics:
            out += ["", "Picture files:"] + pics
        return out + [""]

    def findings(self) -> list[str]:
        name = self.role.name
        found = self.others("findings")
        out = ["### Existing findings", ""]
        if not found:
            out += ["There is no finding in the dossier."
                    + ("" if self.ran["validate"] else " The validate stage has not run."), ""]
        else:
            out += [f"{len(found)} finding(s) of the rules, the engines and the other roles. Do not repeat them.", ""]
            out += [self.finding(f, full=(name == "requirements-reviewer" or not self.compact)) for f in found]
            out.append("")
        if name == "requirements-reviewer":
            early = self.journal(lambda x: (x.origin.stage or x.stage) in ("context", "geometry", "validate"))
            if early:
                out += ["Decisions and assumptions of the context, geometry and validate stages:", "", *early, ""]
        return out

    # ------------------------------------------------------------------ plan
    def _load(self, x: model.LoadItem, value: bool = True) -> str:
        """One load of a load case: the effect id; the value only when asked or when it is not the effect's."""
        out = _id(x.effect)
        if x.factor != 1.0:
            out += f" × {x.factor:g}"
        if value and x.kind != "support":
            out += f" {x.kind} {x.display or _show(x.si, x.unit)}"
            if x.h_si is not None:
                out += f", h = {_show(x.h_si, 'W/(m^2*K)')}"
            if x.interface:
                out += f" on {_id(x.interface)}"
        return out

    def _changed(self, lc: model.LoadCase) -> list[str]:
        """Loads of a load case whose value is not the value of their effect in the phase."""
        if not lc.phase:
            return []
        effects = {e.id: e for e in self.ctx.effects_on(self.comp.element, lc.phase)}
        out = []
        for x in lc.loads:
            e = effects.get(x.effect)
            if e is None:
                out.append(f"{_id(x.effect)} is not an effect of {_id(lc.phase)} on the component")
            elif x.si is not None and e.si is not None and not _same(x.si, e.si):
                out.append(f"{_id(x.effect)}: the load case has {x.display or _show(x.si, x.unit)}, the effect has {e.display}")
        return out

    def _load_case(self, lc: model.LoadCase, full: bool) -> list[str]:
        head = (f"{_id(lc.id)} [{lc.analysis_type}, {lc.status}"
                + (f", scaled from {_id(lc.scaled_from)} by {lc.scale:.6g}" if lc.scaled_from else "") + "]"
                + (f" phase {_id(lc.phase)}" if lc.phase else "") + f": **{_c(lc.title)}**")
        if not full:
            return [f"- {head}. {len(lc.loads)} load(s); supports: {_ids(lc.supports) or 'none'}"
                    + (f"; uses {_ids(lc.uses)}" if lc.uses else "") + "."]
        out = [f"#### {head}", ""]
        if lc.statement:
            out.append(f"- Statement: {_c(lc.statement)}")
        out.append(f"- Supports (interfaces that hold the part): {_ids(lc.supports) or 'none: the part is free'}."
                   + (f" Uses the results of: {_ids(lc.uses)}." if lc.uses else ""))
        loads = [x for x in lc.loads if x.kind != "support"]
        out.append(f"- Loads ({len(loads)}; the values are in the table of the effects of the phase): "
                   + (", ".join(self._load(x, value=False) for x in loads) or "none") + ".")
        changed = self._changed(lc)
        if changed:
            out.append("- NOT THE VALUE OF THE EFFECT: " + "; ".join(changed) + ".")
        for cy in lc.cycles:
            out.append(f"- Cycle {_id(cy.duty_cycle)}: {cy.count:g} repetitions of the states {_ids(cy.states)}.")
        if lc.criteria:
            out.append("- Serves the criteria: " + "; ".join(
                f"{_id(c.requirement)} criterion {c.criterion}" for c in lc.criteria) + ".")
        if lc.hand_calcs:
            out.append(f"- Hand calculations: {_ids(lc.hand_calcs)}.")
        if lc.result_keys and not self.compact:
            out.append("- Results that the analysis must give: " + ", ".join(f"`{k}`" for k in lc.result_keys) + ".")
        if lc.note:
            out.append(f"- Note: {_c(lc.note)}")
        about = self.journal(lambda x: x.subject == lc.id and (x.origin.stage or x.stage) == "plan")
        if about:
            out += ["- Decisions and assumptions of the planner about this load case:"] + [f"  {x}" for x in about]
        return out + [""]

    def _not_in_a_load_case(self) -> list[str]:
        """Per phase that has a load case: the effects on the component that no load case of the phase contains."""
        out = []
        for ph in self.p.system.phases:
            cases = [lc for lc in self.rec["load_cases"] if lc.phase == ph.id]
            if not cases:
                continue
            used = {x.effect for lc in cases for x in lc.loads}
            held = {s for lc in cases for s in lc.supports}
            items = {x.effect: x for x in self.ctx.loads(self.p, self.comp.id, ph.id)}
            left = [e for e in self.ctx.effects_on(self.comp.element, ph.id)
                    if e.kind != "signal" and e.id not in used
                    and not (e.kind == "support" and items.get(e.id) is not None and items[e.id].interface in held)]
            if left:
                out.append(f"- {_id(ph.id)} (load cases {_ids(lc.id for lc in cases)}): "
                           + "; ".join(f"{_id(e.id)} {e.kind} {e.display}".rstrip() for e in left) + ".")
        return out

    def _comparison(self, h: model.HandCalcResult, cmp: model.Comparison, why: bool = True) -> str:
        unit = next((o.unit for o in h.outputs if o.name == cmp.output), None)
        band = (f"band {cmp.band[0]:g} to {cmp.band[1]:g}" if cmp.band else
                f"tolerance {_show(cmp.tolerance_si, unit)}" if cmp.tolerance_si is not None else "no band")
        fe = (f"analysis {_show(cmp.fe_value, unit)}, ratio {cmp.ratio:.3g}" if cmp.fe_value is not None and cmp.ratio is not None
              else f"analysis {_show(cmp.fe_value, unit)}" if cmp.fe_value is not None else "the analysis has no such result yet")
        agree = {True: "agree", False: "DO NOT AGREE", None: "not compared"}[cmp.agree]
        return (f"`{cmp.output}` = {_show(cmp.hand_value, unit)} against `{cmp.result_key}`: {fe}; {band}: {agree}."
                + (f" Why they must agree: {_c(cmp.why)}" if why and cmp.why else ""))

    def _hand_calcs(self) -> list[str]:
        """The hand-calculation results, by model of the component: what is the same for every load case once,
        then one line per load case."""
        groups: dict[str, list[model.HandCalcResult]] = {}
        for h in self.rec["hand_calcs"]:
            groups.setdefault(h.spec, []).append(h)
        out = []
        for spec, hs in groups.items():
            h0 = hs[0]
            out.append(f"- From {_id(spec)} [model `{h0.model}`]"
                       + (f". Reference: {_s(h0.reference)}" if h0.reference else "")
                       + ("" if not h0.inputs else ". Inputs: " + "; ".join(
                           f"{i.name}" + (f" (from {_s(i.source)})" if i.source else "") for i in h0.inputs))
                       + ("" if not h0.outputs else ". Outputs: " + "; ".join(
                           f"`{o.name}`" + (f" ({_s(o.description)})" if o.description else "") for o in h0.outputs))
                       + ("" if not h0.assumptions else ". Assumptions of the formula: " + " ".join(_c(a) for a in h0.assumptions)))
            for h in hs:
                line = (f"  - {_id(h.id)}" + (f" for {_id(h.load_case)}" if h.load_case else "") + ": "
                        + ", ".join(f"{i.name} = {i.display}" for i in h.inputs) + " → "
                        + ", ".join(f"`{o.name}` = {o.display}" for o in h.outputs) + ".")
                if h.steps and not self.compact and h is h0:
                    line += " Steps: " + " ".join(_c(s.text) + (f" [{_c(s.result)}]" if s.result else "") for s in h.steps)
                if h.assumptions != h0.assumptions and h.assumptions:
                    line += " Assumptions: " + " ".join(_c(a) for a in h.assumptions)
                if h.warnings:
                    line += " WARNINGS (outside the validity range of the formula): " + " ".join(_c(w) for w in h.warnings)
                cmps = [self._comparison(h, c, why=False) for c in h.comparisons]
                if cmps:
                    line += " Compared: " + " ".join(cmps)
                out.append(line)
        return out

    def plan(self) -> list[str]:
        r, name = self.rec, self.role.name
        if not self.ran["plan"] and not r["load_cases"]:
            return ["### Load-case plan", "", "The plan stage has not run. There is no load case and no hand "
                    "calculation result.", ""]
        full_role = name == "plan-reviewer"
        out = ["### Load-case plan", "",
               "A load case is one phase at limit loads (factor 1.0); the factors of safety are in the margins. A "
               "proportional load case is not solved: its results are the scale times the results of another load "
               "case.", ""]
        with_case = {lc.phase for lc in r["load_cases"] if lc.phase}
        none = [ph.id for ph in self.p.system.phases if ph.id not in with_case]
        out += [f"Phases with no load case: {_ids(none) or 'none'}.", ""]
        left = self._not_in_a_load_case()
        out += (["Effects on the component that no load case of their phase contains:", "", *left, ""] if left else
                ["Every effect of a phase that has a load case is in a load case of that phase.", ""])
        for lc in r["load_cases"]:
            out += self._load_case(lc, full=full_role and (not self.focus or lc.id in self.focus))
        if not full_role:
            out.append("")
            return out
        ids = {lc.id for lc in r["load_cases"]}
        rest = self.journal(lambda x: (x.origin.stage or x.stage) == "plan" and x.subject not in ids)
        if rest:
            out += ["Other decisions and assumptions of the planner:", "", *rest, ""]
        if r["hand_calcs"]:
            out += ["#### Hand-calculation results", "",
                    "By model of the component (section \"Hand-calculation models\" gives each model, its band and "
                    "why the two values must agree).", "", *self._hand_calcs(), ""]
        else:
            out += ["The plan has no hand-calculation result.", ""]
        return out

    # ------------------------------------------------------------------ analyses
    def _result_rows(self, a: model.AnalysisRecord, keys_used: set[str]) -> tuple[list[list[str]], int]:
        rows, hidden = [], 0
        for key, v in a.results.items():
            if self.compact and v.status == "converged" and key not in keys_used and ".peak@" not in key:
                hidden += 1
                continue
            where = v.near + (f" at {_show(v.location, 'm')}" if v.location else "")
            unc = getattr(v, "uncertainty_abs", None)
            rows.append([f"`{key}`", _show(v.value, v.unit), v.status.upper() if v.status != "converged" else v.status,
                         f"± {_show(unc, v.unit)}" if unc is not None else _pct(v.uncertainty),
                         _series(v.per_level, v.unit) if v.per_level else "",
                         _show(v.extrapolated, v.unit), _num(v.order), where,
                         (v.method if v.method != "fe" else "") + (f" {v.note}" if v.note else "")])
        return rows, hidden

    def _bc(self, b: model.BCItem, item: model.LoadItem | None, decisions: dict, told: set[str]) -> str:
        on = _ids(b.features) or ", ".join(b.faces) or "the body"
        if b.value is None:
            value = ""
        elif item is not None and item.si is not None and _same(b.value, item.si):
            value = f" {item.display or _show(item.si, item.unit)}"
        else:
            value = f" value (SI) {_num(b.value)}" + (
                f" (the load case gives {item.display or _show(item.si, item.unit)} for the effect)"
                if item is not None and item.si is not None else "")
        out = (f"  - `{b.type}` on {on}{value}" + (f" [effect {_id(b.effect)}]" if b.effect else "")
               + (f" [interface {_id(b.interface)}]" if b.interface and not b.effect else "")
               + (f": {_c(b.description)}" if b.description else ""))
        d = decisions.get(b.decision) if b.decision else None
        if d is None:
            return out + (f" Decision {_id(b.decision)}: THE RECORD IS MISSING." if b.decision else " No decision record.")
        if d.id in told:
            return out + f" Decision {_id(d.id)} (above)."
        told.add(d.id)
        return (out + f" Decision {_id(d.id)} [{d.origin.by}; review: {self.review(d)}]: {_c(d.choice)} Why: {_c(d.rationale)}"
                + "".join(f" Alternative: {_c(x.option)} (not chosen: {_c(x.why_not)})" for x in d.alternatives))

    def _analysis(self, a: model.AnalysisRecord, full: bool) -> list[str]:
        r = self.rec
        lc = next((x for x in r["load_cases"] if x.id == a.load_case), None)
        head = (f"{_id(a.id)} [{a.analysis_type}, {a.status.upper() if a.status != 'done' else a.status}] of load case "
                f"{_id(a.load_case)}" + (f", phase {_id(lc.phase)}" if lc and lc.phase else "") + f": **{_c(a.title)}**")
        if a.status != "done":
            return [f"#### {head}", "", f"- No result: {_c(a.message) or 'no reason given'}", ""]
        scaled = lc is not None and lc.scaled_from
        heads = ["result", "value", "status", "uncertainty", "on each mesh, coarse → fine", "extrapolated", "order",
                 "where", "note"]
        if not full or scaled:
            worst = [k for k, v in a.results.items() if v.status in ("not_converged", "singular")]
            failed = [c.id for c in a.checks if c.passed is False]
            line = (f"- {head}. " + (f"Scaled from {_id(lc.scaled_from)} by {lc.scale:.6g}: no solve, every result is "
                                     f"that multiple. " if scaled else "")
                    + f"{len(a.results)} result(s)"
                    + (f"; not converged or singular: {', '.join(f'`{k}`' for k in worst)}" if worst else "")
                    + (f"; checks that did not pass: {_ids(failed)}" if failed else "") + ".")
            if scaled and full and not self.compact:
                rows, _ = self._result_rows(a, set(lc.result_keys))
                return [line, "", *_table(heads[:2] + heads[2:3] + heads[7:], [[x[0], x[1], x[2], x[7], x[8]] for x in rows]), ""]
            return [line, ""] if scaled else [line]
        out = [f"#### {head}", ""]
        if a.statement:
            out.append(f"- Statement: {_c(a.statement)}")
        out.append(f"- Solver: {_c(a.solver) or 'not given'}." + (f" Run time: {a.runtime_s:g} s." if a.runtime_s else "")
                   + (" Material properties used (SI): " + ", ".join(f"{k} = {_num(v)}" for k, v in a.material.items()) + "."
                      if a.material else ""))
        items = {x.effect: x for x in lc.loads} if lc is not None else {}
        decisions = {d.id: d for d in r["decisions"]}
        told: set[str] = set()
        out.append("- Supports and loads of the analysis model"
                   + (f" (the load case asks for the supports {_ids(lc.supports) or 'none: the part is free'})" if lc else "") + ":")
        out += [self._bc(b, items.get(b.effect), decisions, told) for b in a.bcs] or ["  - none recorded"]
        applied = {b.effect for b in a.bcs}
        missing = [x for x in items.values() if x.effect not in applied and x.kind != "support"]
        if missing:
            out.append("- LOADS OF THE LOAD CASE THAT NO SUPPORT OR LOAD OF THE MODEL NAMES: "
                       + "; ".join(self._load(x) for x in missing) + ".")
        if a.mesh:
            out.append("- Meshes: " + "; ".join(
                f"L{m.level}: h = {_show(m.h, 'm')}, {m.nodes} nodes, {m.cells} cells, order {m.order}"
                + (f", min quality {m.min_quality:.3g}" if m.min_quality is not None else "") for m in a.mesh) + ".")
        rows, hidden = self._result_rows(a, set(lc.result_keys) if lc else set())
        out += ["", *_table(heads, rows), ""]
        if hidden:
            self.left_out.append(f"{hidden} converged result(s) of {a.id} that no criterion and no hand calculation uses")
            out += [f"(Left out: {hidden} converged result(s) that no criterion and no hand calculation uses. "
                    f"`python -m dossier show {self.p.root} {a.id}` gives them.)", ""]
        hcs = [h for h in r["hand_calcs"] if h.load_case == a.load_case and h.comparisons]
        compared = {h.id for h in hcs}
        if a.checks:
            out.append("- Checks of the analysis on itself:")
            passed = []
            for c in a.checks:
                short = c.kind == "hand_calc" and c.link in compared        # its numbers are in the comparison below
                numbers = ((f", value {_num(c.value)}" if c.value is not None else "")
                           + (f", limit {_num(c.limit)}" if c.limit is not None else ""))
                if self.compact and c.passed:
                    passed.append(f"`{c.id}` ({c.kind}{'' if short else numbers})")
                    continue
                out.append(f"  - `{c.id}` ({c.kind}): passed: {_yes(c.passed)}"
                           + ("" if short else numbers + f". {_c(c.statement)}") + (f" [{_id(c.link)}]" if c.link else ""))
            if passed:
                out.append("  - passed: " + "; ".join(passed) + ".")
        if hcs:
            out.append("- Hand-calculation comparisons:")
            for h in hcs:
                out += [f"  - {_id(h.id)} (from {_id(h.spec)}, model `{h.model}`): "
                        + self._comparison(h, cmp, why=not (self.compact and cmp.agree)) for cmp in h.comparisons]
                if h.warnings:
                    out.append(f"    - WARNINGS of {_id(h.id)}: " + " ".join(_c(w) for w in h.warnings))
        about = self.journal(lambda x: x.subject in (a.id, a.load_case) and x.id not in told
                             and (x.origin.stage or x.stage) != "plan")
        if about:
            out += ["- Other decisions and assumptions about this analysis:"] + [f"  {x}" for x in about]
        found = [f for f in self.others("findings") if f.subject in (a.id, a.load_case)]
        if found:
            out.append(f"- Existing findings about this analysis: {_ids(f.id for f in found)} (section \"Existing findings\").")
        folders: dict[Path, list[str]] = {}
        for x in a.artifacts:
            if x.kind == "picture":
                self.picture(x.path, x.caption)
                path = (self.p.out / x.path).resolve()
                folders.setdefault(path.parent, []).append(
                    f"`{path.name}`" + (f" ({_s(x.caption)})" if x.caption else "") + ("" if path.exists() else " (THE FILE IS MISSING)"))
        for folder, names in folders.items():
            out.append(f"- Picture files in `{folder}` (open them if you have file access): " + "; ".join(names) + ".")
        rest: dict[str, int] = {}
        for x in a.artifacts:
            if x.kind != "picture":
                rest[x.kind] = rest.get(x.kind, 0) + 1
        if rest:
            out.append("- Other files of the record: " + ", ".join(f"{v} {k} file(s)" for k, v in rest.items()) + ".")
        return out + [""]

    def analyses(self) -> list[str]:
        r = self.rec
        out = ["### Analyses", ""]
        if not r["analyses"]:
            return out + ["No analysis has run.", ""]
        out += [f"Status of a result: `converged` = the discretization uncertainty of the mesh study is under "
                f"{_pct(self.p.policy.max_uncertainty)}; `NOT_CONVERGED`; `SINGULAR` = the value grows without bound "
                "under refinement; `ESTIMATE` = a hand calculation or a bound. \"on each mesh\" is the value on each "
                "mesh of the study, coarse to fine. The peak evaluation leaves out the zone next to a fixed support. "
                "A support or load of the model shows the value of its effect in the load case; when the model has "
                "another value, the line gives both.", ""]
        have = {a.load_case for a in r["analyses"]}
        missing = [lc.id for lc in r["load_cases"] if lc.id not in have]
        if missing:
            out += [f"Load cases with no analysis record: {_ids(missing)}.", ""]
        if self.focus:
            out += [f"In full: {_ids(sorted(self.focus))}. The other analyses are on one line each.", ""]
        for a in r["analyses"]:
            out += self._analysis(a, full=not self.focus or a.id in self.focus or a.load_case in self.focus)
        return out + [""]

    def compliance(self) -> list[str]:
        items = self.rec["compliance"]
        if not items:
            return ["### Margins", "", "The assess stage has not run: no margin uses the results yet.", ""]
        out = ["### Margins (what the results are used for)", "",
               "Margin of safety: limit / (factor × response) - 1 for `<=`. The response enters with its uncertainty "
               "on the unsafe side.", ""]
        out += _table(["item", "requirement", "load case", "result", "response", "with uncertainty", "factor", "limit",
                       "margin", "verdict", "reasons"],
                      [[_id(c.id), _id(c.requirement), _id(c.load_case), f"`{c.result_key}`", _show(c.value, c.unit),
                        _show(c.conservative_value, c.unit), f"{c.factor:g}", _show(c.limit, c.unit),
                        "" if c.margin is None else f"{c.margin:+.3g}", c.verdict, " ".join(c.reasons)] for c in items])
        return out + [""]

    def previous(self) -> list[str]:
        mine = [x for k in ("findings", "decisions", "assumptions") for x in self.rec[k] if self.own(x)]
        if not mine:
            return []
        out = ["### Your earlier submission", "",
               f"The role `{self.role.name}` has {len(mine)} record(s) in the dossier from an earlier submission. A new "
               "submission replaces all of them. Give again, with the same subject and key, each one that is still "
               "right; leave out what is no longer right. Read the review of the person where there is one.", ""]
        for x in mine:
            out.append(self.finding(x) if isinstance(x, model.Finding)
                       else self.decision(x) if isinstance(x, model.Decision) else self.assumption(x))
        return out + [""]


SECTIONS = ("status", "project", "system", "phases", "effects", "numbers", "component", "hand_calc_specs",
            "requirements", "policy", "material", "geometry", "findings", "plan", "analyses", "compliance", "previous")


def _dossier(project: Project, role: roles_mod.Role, focus: Iterable[str], compact: bool) -> tuple[str, _Brief]:
    b = _Brief(project, role, focus, compact)
    body: list[str] = []
    for name in role.sections:
        body += getattr(b, name)()
    head = [f"## The dossier: {project.file.name} (`{project.file.id}`)", "",
            f"Project folder: `{project.root}`. Ids of records are in `code`. Numbers have the unit that the dossier "
            f"shows. The part is verified by analysis; you read what the pipeline has made so far.", ""]
    if compact:
        head += ["This is the COMPACT form of the brief, because the full form is longer than the size the tool aims "
                 "for. It holds every record. It leaves out: "
                 "the steps of the hand calculations; the result keys that each load case asks for; the statement of a "
                 "check that passed; why a hand calculation and a result must agree, when they agree; the evidence text "
                 "of an existing finding, for a role that does not review the requirements"
                 + ("; " + "; ".join(b.left_out) if b.left_out else "") + ". `python -m dossier show "
                 f"{project.root} ID` gives a record in full.", ""]
    text = "\n".join([*head, *body]).rstrip() + "\n"
    while "\n\n\n" in text:
        text = text.replace("\n\n\n", "\n\n")
    return text, b


def dossier(project: Project, role: "str | roles_mod.Role", focus: Iterable[str] = (), max_chars: int = MAX_CHARS,
            reserve: int = 0) -> str:
    """The dossier part of a brief. The full form when it fits in `max_chars` - `reserve`, else the compact form."""
    r = roles_mod.get(role)
    text, _ = _dossier(project, r, focus, compact=False)
    if len(text) + reserve > max_chars:
        text, _ = _dossier(project, r, focus, compact=True)
    return text


def build(project: Project, role: "str | roles_mod.Role", focus: Iterable[str] = (), max_chars: int = MAX_CHARS,
          shell: bool = True) -> str:
    """The brief of a role: its instructions, then the dossier. `shell=False` leaves the shell commands out
    (a call through the API)."""
    r = roles_mod.get(role)
    head = (f"# Brief for the role `{r.name}`\n\n"
            f"Read the instructions, then the dossier. Then write your submission.\n\n")
    ins = roles_mod.instructions(r, project=str(project.root), shell=shell)
    data = dossier(project, r, focus, max_chars, reserve=len(head) + len(ins) + 1)
    text = head + ins + "\n" + data
    if len(text) > max_chars and not list(focus or ()) and ("plan" in r.sections or "analyses" in r.sections):
        text += (f"\nNote of the tool: this brief has {len(text)} characters, more than the {max_chars} that the tool "
                 f"aims for. `--focus ID` gives one load case or one analysis in full and the other ones on one line "
                 f"each. Give one submission for the whole role at the end.\n")
    return text


def pictures(project: Project, role: "str | roles_mod.Role") -> list[tuple[Path, str]]:
    """The picture files that the brief of a role names, with their captions (files that exist only)."""
    _, b = _dossier(project, roles_mod.get(role), (), compact=False)
    return [(p, cap) for p, cap in b.pictures if p.exists()]
