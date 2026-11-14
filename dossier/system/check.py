"""The system model check: is the system model sound and complete enough to analyse the component?

    check(project, ctx, geo) -> Journal        findings (category model, context, coverage) and assumptions

`Project.load` rejects references that point nowhere and ids used twice; `context.build` rejects unit
errors. This module looks for what is legal but wrong or weak: a load that the analysis cannot apply, a
number without a source, an assumption without a reason, a phase in which nothing holds the part.
`geo` is the dict of the geometry stage; it can be None or lack keys, and the checks that need it are then
left out (they are marked "geo" below).

Every record has a stable id: `make_id("FND", subject, key)` for a finding, `make_id("ASM", subject,
"assumed")` for an assumption. `<role>` is a feature role (wetted, seat, ...), `<kind>` an effect kind.

    subject         key                      severity   what is wrong
    --------------  -----------------------  ---------  -----------------------------------------------------
    connection      no-interface             major      effects on the component, but no interface for it
    interface       no-<role>-features       major      effects act on a role for which it has no features
    interface       no-<role>-faces          major      geo: the features of that role give no CAD face
    connection      two-interfaces           major      two interfaces realize it; the context uses the last
    connection      acts-on                  minor      load effects without `acts_on` on a connection that
                                                        starts at the component: they act on the other end
    connection      no-effect                info       a connection of the component that states no effect
    connection      no-thermal-effect        info       no thermal effect in phases that have thermal effects
    connection      duplicate-<kind>         minor      two effects of one state kind on the same faces
    connection      no-medium                minor/info a fluid connection without a medium
    connection      self-loop                minor      `from` and `to` are the same element
    parameter       no-justification         major      basis assumed, empty note
    parameter       no-source                minor      basis specified / measured / reference, empty source
                                                        (or derived, with no expression and no source)
    REQ / MAT id    parameters-no-source     minor      the same, for the numbers of a requirement or material
    parameter       unused                   info       nothing reads it
    parameter       shadows-unit             minor      its name is also a unit of expressions
    effect          no-h                     major      fluid_temperature without a heat transfer coefficient
    effect          h-not-positive           major      a heat transfer coefficient of zero or less
    effect          h-unused                 minor      `h` on an effect that is not a fluid temperature
    effect          temperature              error      an absolute temperature of 0 K or less
    effect          pressure                 major      a negative pressure
    effect          zero                     minor      a pressure of zero, or a vector that is all zeros
    effect          no-justification         major      basis assumed, empty note
    effect          no-source                minor      a literal value (no parameter) and no source
    effect          on-support               minor      a force, moment or displacement on faces that a
                                                        support holds in the same phase
    phase           no-effect                info       no load on the component
    phase           free                     info       mechanical loads and no support: the loads must balance
    phase           duration                 minor      a duration that is not a positive time
    phase           not-in-duty-cycle        info       an operation phase with loads that no duty cycle has
    duty cycle      phase-no-effect          minor      a phase of its sequence has no load on the component
    duty cycle      count                    major/minor  a count below zero (major) or of zero (minor)
    duty cycle      empty                    major      no phase in its sequence
    element         no-connection            minor      no connection joins it to the system
    element         no-support               major      the component: no support in any operation phase
    element         component                minor      `component` names another component than the one
                                                        that realizes the element, or one that does not exist
    component       reference-temperature    major      not a positive absolute temperature
    material        no-conductivity          major      a thermal analysis is necessary, `k` is not given
    material        no-expansion             major      the same, for `alpha`

Assumptions: every parameter and every effect with `basis: assumed` (subject = its name or id). The
statement gives the value, the rationale is the `note` as the engineer wrote it, the impact names the
parameters, effects and other records that change with the value.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, Iterable

from dossier import units
from dossier.context import MATERIAL_NAMES, MECHANICAL, THERMAL, Context, ResolvedEffect
from dossier.model import DEFAULT_ROLE, ROLES, Value
from dossier.project import Project
from dossier.records import Journal

KIND_WORDS = {"pressure": "pressure", "force": "force", "moment": "moment", "acceleration": "acceleration",
              "fluid_temperature": "fluid temperature", "temperature": "temperature", "heat_flux": "heat flux",
              "mass_flow": "mass flow", "displacement": "displacement", "support": "support", "signal": "signal"}
# effect kinds that give the state of a medium: two of one kind on the same faces are a contradiction or a sum
STATE_KINDS = ("pressure", "fluid_temperature", "temperature", "mass_flow")
HOLDS = ("support", "displacement")        # effect kinds that hold the part
_WHAT_A_BASIS_NEEDS = {"specified": "A specified value needs the document that states it.",
                       "measured": "A measured value needs the record of the measurement.",
                       "reference": "A reference value needs the handbook or the standard that gives it.",
                       "derived": "A derived value needs the expression or the document that gives the derivation."}
_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


# --------------------------------------------------------------------------- words


def _list(items: Iterable[Any], cap: int = 10) -> str:
    """a | a and b | a, b and c. More than `cap` items: the first `cap`, then the number of the others."""
    items = [str(x) for x in items]
    if len(items) > cap:
        return ", ".join(items[:cap]) + f" and {len(items) - cap} more"
    if len(items) < 2:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def _n(n: int, word: str, plural: str | None = None) -> str:
    return f"{n} {word if n == 1 else (plural or word + 's')}"


def _kinds(counts: dict[str, int]) -> str:
    """pressure 3, force 4 (in the order of the model's kinds)."""
    return ", ".join(f"{KIND_WORDS.get(k, k)} {counts[k]}" for k in KIND_WORDS if counts.get(k))


def _phases(ids: list[str]) -> str:
    return ("phase " if len(ids) == 1 else "phases ") + _list(ids)


def _named(record_id: str, name: str) -> str:
    return f'{record_id} ("{name}")' if name else record_id


def _names(value: Value | None, known: set[str]) -> set[str]:
    return set() if value is None else (value.names() & known)


def _expr_names(expr: str, known: set[str]) -> set[str]:
    """Parameter names in an expression that the context has not evaluated (a requirement check). When the
    expression cannot be read, every word of it counts as a read: this check must not call a parameter
    unused because another check has a problem."""
    try:
        return units.names_in(expr) & known
    except units.UnitsError:
        return set(_NAME.findall(expr)) & known


# --------------------------------------------------------------------------- the facts the checks read


class _Facts:
    """The project as the checks need it: the effects on the component, who reads which parameter."""

    def __init__(self, project: Project, ctx: Context, geo: dict | None):
        self.project, self.ctx = project, ctx
        self.geo: dict = geo if isinstance(geo, dict) else {}
        self.system = project.system
        self.comp = project.subject
        self.el = self.comp.element
        self.connections = {c.id: c for c in self.system.connections}
        self.raw = {ef.id: (c, ef) for c in self.system.connections for ef in c.effects}
        self.interfaces: dict[str, list] = {}
        for it in self.comp.interfaces:
            self.interfaces.setdefault(it.connection, []).append(it)
        self.phase_ids = [ph.id for ph in self.system.phases]
        self.phase = {ph.id: ph for ph in self.system.phases}
        self.on = [e for e in ctx.effects if e.on == self.el]               # one per effect and phase
        self.by_id: dict[str, list[ResolvedEffect]] = {}
        for e in self.on:
            self.by_id.setdefault(e.id, []).append(e)
        self.by_phase: dict[str, list[ResolvedEffect]] = {p: [] for p in self.phase_ids}
        for e in self.on:
            self.by_phase.setdefault(e.phase, []).append(e)
        self.readers = self._readers()

    def interface(self, connection: str):
        """The interface the context maps the effects of a connection to (the last one that names it)."""
        its = self.interfaces.get(connection)
        return its[-1] if its else None

    def role(self, e: ResolvedEffect) -> str:
        return e.role or DEFAULT_ROLE.get(e.kind, "")

    def loads(self, phase: str) -> list[ResolvedEffect]:
        return [e for e in self.by_phase.get(phase, []) if e.kind != "signal"]

    def _readers(self) -> dict[str, list[tuple[str, str]]]:
        """name -> [(kind of reader, its id)]: everything that reads a named number."""
        known = set(self.ctx.params)
        out: dict[str, list[tuple[str, str]]] = {n: [] for n in known}

        def read(names: set[str], kind: str, rid: str):
            for n in sorted(names):
                if (kind, rid) not in out[n]:
                    out[n].append((kind, rid))

        for name, p in self.ctx.params.items():
            read(set(p.depends_on) & known, "parameter", name)
        for c in self.system.connections:
            for ef in c.effects:
                read(_names(ef.value, known) | _names(ef.h, known) | _names(ef.at, known), "effect", ef.id)
        for dc in self.system.duty_cycles:
            read(_names(dc.count, known), "duty cycle", dc.id)
        for ph in self.system.phases:
            read(_names(ph.duration, known), "phase", ph.id)
        read(_names(self.project.policy.thermal_delta, known), "policy", self.project.policy.id)
        for m in self.project.materials.values():
            extra = [m.properties_temperature, *(m.service_temperature or ()),
                     *((m.fatigue.sigma_f, m.fatigue.endurance_limit) if m.fatigue else ())]
            for v in extra:
                read(_names(v, known), "material", m.id)
        for comp in self.project.components.values():
            read(_names(comp.reference_temperature, known), "component", comp.id)
            for hc in comp.hand_calcs:
                for v in hc.inputs.values():
                    read(_names(v, known), "hand calculation", hc.id)
                for cmp in hc.compare:
                    read(_names(cmp.tolerance, known), "hand calculation", hc.id)
            if comp.flow is not None:
                for v in comp.flow.fluid.values():
                    read(_names(v, known), "flow domain", comp.id)
            for req in self.project.requirements.get(comp.id, []):
                for pname, rp in req.parameters.items():
                    if rp.check is not None:
                        read(_expr_names(rp.check.against, known), "requirement", req.id)
                    if (rp.role or rp.check is not None) and pname in known:
                        read({pname}, "requirement", req.id)       # the validation compares it with the context
                for crit in req.criteria:
                    read(_names(crit.limit, known), "requirement", req.id)
                    if isinstance(crit.factor, str) and crit.factor in known:
                        read({crit.factor}, "requirement", req.id)
        return out

    def downstream(self, name: str) -> tuple[list[str], list[tuple[str, str]]]:
        """What changes with a parameter: (the parameters, the other readers), direct or through parameters."""
        params: list[str] = []
        others: list[tuple[str, str]] = []
        todo = [name]
        while todo:
            n = todo.pop(0)
            for kind, rid in self.readers.get(n, []):
                if kind == "parameter":
                    if rid not in params and rid != name:
                        params.append(rid)
                        todo.append(rid)
                elif (kind, rid) not in others and rid != name:
                    others.append((kind, rid))
        order = {n: i for i, n in enumerate(self.ctx.params)}
        return sorted(params, key=order.get), others

    def impact(self, name: str) -> tuple[str, list[str], list[str]]:
        """The impact text of a parameter, the parameters and the effects that change with it."""
        params, others = self.downstream(name)
        effects = [rid for kind, rid in others if kind == "effect"]
        out = [f"A different value changes {_n(len(params), 'parameter')}: {_list(params)}." if params
               else f"No other parameter reads {name}."]
        if effects:
            out.append(f"A different value changes {_n(len(effects), 'effect')}: {_list(effects)}.")
            phases = [p for p in self.phase_ids if any(e.phase == p for e in self.ctx.effects if e.id in effects)]
            out.append(f"{'The effect acts' if len(effects) == 1 else 'The effects act'} in the {_phases(phases)}.")
        for kind, word in (("hand calculation", "hand calculation"), ("requirement", "requirement"),
                           ("duty cycle", "duty cycle"), ("phase", "phase"), ("component", "component"),
                           ("flow domain", "flow domain of the component"), ("material", "material"),
                           ("policy", "policy")):
            ids = [rid for k, rid in others if k == kind]
            if ids:
                out.append(f"The {word}{'' if len(ids) == 1 else 's'} {_list(ids)} "
                           f"{'depends' if len(ids) == 1 else 'depend'} on the value.")
        if not params and not others:
            out = [f"Nothing reads {name}. A different value does not change the analysis."]
        return " ".join(out), params, effects


# --------------------------------------------------------------------------- the check


def check(project: Project, ctx: Context, geo: dict | None = None) -> Journal:
    """Findings and assumptions about the system model, for the component under verification."""
    j = Journal("validate", "engine:system.check")
    f = _Facts(project, ctx, geo)
    _interfaces(j, f)
    _connections(j, f)
    _parameters(j, f)
    _effects(j, f)
    _phases_and_support(j, f)
    _duty_cycles(j, f)
    _elements(j, f)
    _component(j, f)
    return j


# --------------------------------------------------------------------------- interfaces: can each effect be applied?


def _interfaces(j: Journal, f: _Facts) -> None:
    comp = f.comp
    no_interface: dict[str, list[str]] = {}
    no_features: dict[tuple[str, str], list[str]] = {}
    no_faces: dict[tuple[str, str], list[str]] = {}
    for eid, rs in f.by_id.items():
        e = rs[0]
        role = f.role(e)
        if e.kind == "signal" or role in ("", "body"):       # a signal is not a load; a body load needs no faces
            continue
        it = f.interface(e.connection)
        if it is None:
            no_interface.setdefault(e.connection, []).append(eid)
        elif not it.features.get(role):
            no_features.setdefault((it.id, role), []).append(eid)
        else:
            faces = f.geo.get("interfaces", {}).get(it.id, {}).get(role) if f.geo else None
            if isinstance(faces, list) and not faces:
                no_faces.setdefault((it.id, role), []).append(eid)

    for cid, eids in no_interface.items():
        kinds = Counter(f.by_id[e][0].kind for e in eids)
        roles = sorted({f.role(f.by_id[e][0]) for e in eids})
        j.find("system.effect_interface", "major", "model", cid, "no-interface",
               f"No interface for the connection {cid}",
               f"The component {comp.id} has no interface for the connection {cid}. "
               f"The analysis cannot apply the {_n(len(eids), 'effect')} of this connection.",
               f"The connection {cid} states {_n(len(eids), 'effect')} on the element {f.el}: {_kinds(kinds)}. "
               f"An effect acts on the features of an interface. "
               f"No interface of the component {comp.id} has `connection: {cid}`.",
               f"Add an interface with `connection: {cid}` to the component. "
               f"Give the features for the {'role' if len(roles) == 1 else 'roles'} {_list(roles)}.",
               values={"effects": eids, "kinds": dict(kinds), "roles": roles},
               links=[("about", comp.id), *(("about", e) for e in eids)])

    for (iid, role), eids in no_features.items():
        it = next(i for i in comp.interfaces if i.id == iid)
        have = [r for r in ROLES if it.features.get(r)]
        unknown = "" if role in ROLES else (f" The model has no role `{role}`. Its roles are {_list(ROLES)}.")
        j.find("system.effect_interface", "major", "model", iid, f"no-{role}-features",
               f"The interface {iid} has no features for the role {role}",
               f"The interface {iid} has no features for the role `{role}`. "
               f"The analysis cannot apply {_n(len(eids), 'effect')} that {'acts' if len(eids) == 1 else 'act'} "
               f"on this role.",
               f"The {'effect' if len(eids) == 1 else 'effects'} {_list(eids)} of the connection {it.connection} "
               f"{'acts' if len(eids) == 1 else 'act'} on the role `{role}`. "
               f"The role comes from the kind of the effect or from its `role` field. "
               + (f"The interface gives features for the {'role' if len(have) == 1 else 'roles'} {_list(have)} only."
                  if have else "The interface gives no features.") + unknown,
               f"Add the features for the role `{role}` to the interface {iid}. "
               f"As an alternative, set the `role` field of the effects to a role that the interface has.",
               values={"effects": eids, "role": role, "roles_of_interface": have},
               links=[("about", it.connection), *(("about", e) for e in eids)])

    for (iid, role), eids in no_faces.items():
        it = next(i for i in comp.interfaces if i.id == iid)
        j.find("system.effect_interface", "major", "model", iid, f"no-{role}-faces",
               f"The role {role} of the interface {iid} has no CAD face",
               f"The features of the interface {iid} for the role `{role}` give no CAD face. "
               f"The analysis cannot apply {_n(len(eids), 'effect')} that {'acts' if len(eids) == 1 else 'act'} "
               f"on this role.",
               f"The geometry stage found 0 faces for the role `{role}` of the interface {iid}. "
               f"The {'effect' if len(eids) == 1 else 'effects'} {_list(eids)} "
               f"{'acts' if len(eids) == 1 else 'act'} on this role.",
               f"Correct the features of the role `{role}`. Then run the geometry stage again.",
               values={"effects": eids, "role": role, "features": list(it.features.get(role, []))},
               links=[("about", it.connection), *(("about", e) for e in eids)])

    for cid, its in f.interfaces.items():
        if len(its) > 1:
            used, ignored = its[-1], its[:-1]
            n_eff = len({e.id for e in f.on if e.connection == cid})
            j.find("system.effect_interface", "major", "model", cid, "two-interfaces",
                   f"Two interfaces realize the connection {cid}",
                   f"{_n(len(its), 'interface')} of the component {comp.id} realize the connection {cid}. "
                   f"The context maps the effects of the connection to the interface {used.id} only.",
                   f"The interfaces {_list(i.id for i in its)} have `connection: {cid}`. "
                   f"The connection states {_n(n_eff, 'effect')} on the element {f.el}. "
                   f"No effect acts on the features of {_list(i.id for i in ignored)}.",
                   f"Make one interface for the connection {cid} and give it all the features. "
                   f"As an alternative, make one connection for each interface in the system model.",
                   values={"interfaces": [i.id for i in its], "used": used.id},
                   links=[("about", i.id) for i in its])


# --------------------------------------------------------------------------- connections


def _connections(j: Journal, f: _Facts) -> None:
    comp = f.comp
    thermal_phases = [p for p in f.phase_ids if any(e.kind in THERMAL for e in f.by_phase[p])]
    for c in f.system.connections:
        touches = f.el in (c.from_, c.to)
        other = c.to if c.from_ == f.el else c.from_
        on_el = [e for e in f.on if e.connection == c.id]

        if c.from_ == c.to:
            j.find("system.connection_ends", "minor", "model", c.id, "self-loop",
                   f"The connection {c.id} joins an element to itself",
                   f"The connection {c.id} has the element {c.from_} as `from` and as `to`.",
                   "A connection is an interface between two elements. "
                   "The system diagram cannot show which element acts on which.",
                   "Give the two elements that the connection joins. "
                   "If the two parts are in one element, make an element for each part.",
                   links=[("about", c.from_)])

        if c.kind == "fluid" and not c.medium.strip():
            j.find("system.connection_medium", "minor" if touches else "info", "model", c.id, "no-medium",
                   f"The fluid connection {c.id} gives no medium",
                   f"The connection {c.id} has the kind `fluid`, but it does not give its medium.",
                   (f"The connection joins the component to the element {other}. "
                    "A person cannot check the material against a medium that the model does not name."
                    if touches else
                    f"The connection joins the elements {c.from_} and {c.to}. "
                    "The system diagram cannot show what flows through it."),
                   f"Give the medium of the connection {c.id}, for example `medium: LOX`.",
                   links=[("about", c.from_), ("about", c.to)])

        if not touches or c.from_ == c.to:
            continue

        # load effects without `acts_on` on a connection that starts at the component act on the other end
        if c.from_ == f.el:
            lost = [ef for ef in c.effects if ef.acts_on is None and ef.kind != "signal"]
            if lost:
                kinds = Counter(ef.kind for ef in lost)
                j.find("system.acts_on", "minor", "model", c.id, "acts-on",
                       f"Effects of the connection {c.id} act on {c.to}, not on the component",
                       f"{_n(len(lost), 'effect')} of the connection {c.id} {'acts' if len(lost) == 1 else 'act'} "
                       f"on the element {c.to}, not on the component {comp.id}.",
                       f"The connection goes from {c.from_} to {c.to}. "
                       f"An effect without `acts_on` acts on the `to` element. "
                       f"The {'effect' if len(lost) == 1 else 'effects'} {_list(ef.id for ef in lost)} "
                       f"({_kinds(kinds)}) {'does' if len(lost) == 1 else 'do'} not give `acts_on`. "
                       f"The analysis of the component does not use {'it' if len(lost) == 1 else 'them'}.",
                       f"If these effects act on the component, add `acts_on: {c.from_}` to each of them. "
                       f"If they act on {c.to}, write `acts_on: {c.to}` to show the intent.",
                       values={"effects": [ef.id for ef in lost], "kinds": dict(kinds)},
                       links=[("about", ef.id) for ef in lost])

        it = f.interface(c.id)
        if not on_el:
            j.find("system.connection_effects", "info", "coverage", c.id, "no-effect",
                   f"The connection {c.id} states no effect on the component",
                   f"The connection {c.id} joins the component to the element {other}, "
                   f"but it states no effect on the component.",
                   (f"The component realizes the connection with the interface {it.id}. " if it is not None else
                    "The component has no interface for the connection. ")
                   + f"The connection has {_n(len(c.effects), 'effect')}, and none acts on the element {f.el}. "
                   "The analysis applies no load at this connection.",
                   f"Add the effects of the connection {c.id} on the component. "
                   "If the connection applies no load, no action is necessary.",
                   links=[("about", other), *([("about", it.id)] if it is not None else [])])
            continue

        # phases with thermal effects in which this connection acts but gives no thermal effect
        quiet = [p for p in thermal_phases
                 if any(e.phase == p and e.kind != "signal" for e in on_el)
                 and not any(e.phase == p and e.kind in THERMAL for e in on_el)]
        if quiet and it is not None:
            j.find("system.thermal_coverage", "info", "coverage", c.id, "no-thermal-effect",
                   f"The connection {c.id} states no thermal effect",
                   f"The connection {c.id} states no thermal effect on the component "
                   f"in {_n(len(quiet), 'phase')} that {'has' if len(quiet) == 1 else 'have'} thermal effects.",
                   f"Other connections state temperatures or heat flows in the {_phases(quiet)}. "
                   f"The connection {c.id} acts on the component in {'this phase' if len(quiet) == 1 else 'these phases'}, "
                   f"but it gives no temperature and no heat flow. "
                   f"A thermal analysis then has no heat flow through the faces of the interface {it.id}.",
                   f"Add a thermal effect to the connection {c.id}. "
                   "As an alternative, record that the heat flow through this connection is negligible.",
                   values={"phases": quiet},
                   links=[("about", it.id), *(("about", p) for p in quiet)])

        # two effects of one state kind on the same faces in the same phase
        groups: dict[tuple[str, str], dict[str, list[ResolvedEffect]]] = {}
        for e in on_el:
            if e.kind in STATE_KINDS:
                groups.setdefault((e.kind, f.role(e)), {}).setdefault(e.phase, []).append(e)
        for (kind, role), per_phase in groups.items():
            clashes = {p: es for p, es in per_phase.items() if len(es) > 1}
            if not clashes:
                continue
            first_phase = next(p for p in f.phase_ids if p in clashes)
            es = clashes[first_phase]
            ids = list(dict.fromkeys(e.id for es_ in clashes.values() for e in es_))
            word = KIND_WORDS[kind]
            j.find("system.duplicate_effect", "minor", "context", c.id, f"duplicate-{kind}" + ("" if role == DEFAULT_ROLE[kind] else f"-{role}"),
                   f"Two {word} effects of the connection {c.id} act on the same faces",
                   f"{_n(len(es), f'{word} effect')} of the connection {c.id} act on the role `{role}` "
                   f"in the phase {first_phase}: {_list(e.id for e in es)}.",
                   f"The values are {_list(f'{e.display} ({e.id})' for e in es)}. "
                   f"A {word} is a state of the medium, so one value for each face is usual. "
                   + (f"The same occurs in the {_phases([p for p in f.phase_ids if p in clashes and p != first_phase])}. "
                      if len(clashes) > 1 else "")
                   + "The analysis applies the effects together.",
                   "If one effect replaces the other, remove one of them from the phase. "
                   "If the effects act together, say so in their descriptions.",
                   values={"phases": [p for p in f.phase_ids if p in clashes], "effects": ids, "role": role},
                   links=[*(("about", i) for i in ids), *(("about", p) for p in f.phase_ids if p in clashes)])


# --------------------------------------------------------------------------- parameters


def _parameters(j: Journal, f: _Facts) -> None:
    ctx = f.ctx
    material_names = {*MATERIAL_NAMES, "nu"}
    by_origin: dict[str, list[str]] = {}
    for name, prm in ctx.params.items():
        system = prm.origin == "system"
        if prm.basis == "assumed":
            _assume_parameter(j, f, name)
        elif not prm.source.strip() and not (prm.basis == "derived" and prm.expr):
            if system:
                _no_source(j, f, name)
            else:
                by_origin.setdefault(prm.origin, []).append(name)

        if prm.origin not in f.project.materials and not f.readers.get(name):
            where = "The system model" if system else f"The requirement {prm.origin}"
            j.find("system.parameter_unused", "info", "context", name, "unused",
                   f"Nothing reads the parameter {name}",
                   f"No expression, effect, duty cycle, requirement or hand calculation reads the parameter {name}.",
                   f"{where} gives {name} = {prm.display}" + (f' ("{prm.title}")' if prm.title and system else "")
                   + ". The value does not change the analysis.",
                   "Use the parameter in the effect or the expression that it describes. "
                   "If the model does not need the parameter, remove it.",
                   values={"si": prm.si, "unit": prm.unit},
                   links=[] if system else [("about", prm.origin)])

        if name not in material_names and units.is_unit(name):
            unit = f"{units.ureg.parse_units(name):P}"
            j.find("system.parameter_name", "minor", "context", name, "shadows-unit",
                   f"The parameter name {name} is also a unit",
                   f"The parameter {name} has the name of the unit `{name}` ({unit}). "
                   f"An expression that contains `{name}` reads the parameter, not the unit.",
                   f"The parameter {name} has the value {prm.display}. "
                   f"An expression reads a name as a unit only when no parameter has that name. "
                   f"A person who writes `2 * {name}` for a quantity in {unit} gets a wrong number.",
                   f"Give the parameter a name that is not a unit, for example `{name}_{'sys' if system else 'req'}`.",
                   values={"si": prm.si, "unit": prm.unit},
                   links=[] if system else [("about", prm.origin)])

    for origin, names in by_origin.items():
        is_material = origin in f.project.materials
        what = "material" if is_material else "requirement"
        j.find("system.parameter_source", "minor", "context", origin, "parameters-no-source",
               f"No source for the numbers of {origin}",
               f"The {what} {origin} gives no source for its {'numbers' if len(names) > 1 else 'number'} "
               f"{_list(names)}.",
               f"The context reads {'these numbers' if len(names) > 1 else 'this number'} from the {what} {origin}: "
               f"{_list(f'{n} = {ctx.params[n].display}' for n in names)}. "
               "A person cannot check a number without a source.",
               ("Give the `source` or the `spec` of the material." if is_material else
                "Give the `source` of the requirement, or the `document` of the requirements file."),
               values={"parameters": names},
               links=[("about", n) for n in names])


def _assume_parameter(j: Journal, f: _Facts, name: str) -> None:
    prm = f.ctx.params[name]
    impact, params, effects = f.impact(name)
    shown = f"{prm.display} (from the expression `{prm.expr}`)" if prm.expr else prm.display
    statement = (f"The parameter {_named(name, prm.title)} has the value {shown}. "
                 + (f'This value is an assumption. The system model gives this source: "{prm.source}".'
                    if prm.source.strip() else "This value is an assumption: no document gives it."))
    note = prm.note.strip()
    a = j.assume(name, "assumed", f"Assumed value of {name}", statement,
                 f'The system model gives this justification: "{note}"' if note
                 else "The system model gives no justification for this value.",
                 impact,
                 verify_by="Compare the value with a measurement, a calculation or a source document. "
                           "Then change the basis of the parameter and give the source.",
                 links=[("source", f"doc:{prm.source.strip()}")] if prm.source.strip() else [],
                 by="rule:system.assumed_parameter")
    if not note:
        j.find("system.assumption_justification", "major", "context", name, "no-justification",
               f"No justification for the assumed parameter {name}",
               f"The parameter {name} has the basis `assumed`, but its note is empty. "
               "The assumption has no justification.",
               f"The system model assumes {name} = {prm.display}. "
               + (f"{_n(len(effects), 'effect')} {'depends' if len(effects) == 1 else 'depend'} on this value: "
                  f"{_list(effects)}. " if effects else
                  f"{_n(len(params), 'parameter')} {'depends' if len(params) == 1 else 'depend'} on this value: "
                  f"{_list(params)}. " if params else "")
               + "A person who reviews the assumption needs the reason for the value.",
               f"Write the reason for the value in the `note` of the parameter {name}. "
               "If a document gives the value, change the basis and give the source.",
               values={"si": prm.si, "unit": prm.unit, "effects": effects, "parameters": params},
               links=[("about", a.id), *(("about", e) for e in effects)])


def _no_source(j: Journal, f: _Facts, name: str) -> None:
    prm = f.ctx.params[name]
    _, params, effects = f.impact(name)
    depends = []
    if params:
        depends.append(_n(len(params), "parameter"))
    if effects:
        depends.append(_n(len(effects), "effect"))
    derived = prm.basis == "derived"
    j.find("system.parameter_source", "minor", "context", name, "no-source",
           f"No source for the parameter {name}",
           (f"The parameter {name} has the basis `derived`, but it has no expression and no source." if derived else
            f"The parameter {name} has the basis `{prm.basis}`, but it gives no source."),
           f"The system model gives {name} = {prm.display}. "
           + (f"{_list(depends)} depend on this value. " if depends else "")
           + _WHAT_A_BASIS_NEEDS.get(prm.basis, "A person cannot check a number without a source."),
           "Give the document, the table or the record in the `source` of the parameter."
           + (" If the expression is the source, set the basis to `derived`." if prm.expr else "")
           + (" As an alternative, give the derivation as an expression." if derived else ""),
           values={"si": prm.si, "unit": prm.unit, "basis": prm.basis, "effects": effects, "parameters": params},
           links=[("about", e) for e in effects])


# --------------------------------------------------------------------------- effects: values, sources, supports


def _effects(j: Journal, f: _Facts) -> None:
    known = set(f.ctx.params)
    for eid, rs in f.by_id.items():
        e = rs[0]
        conn, raw = f.raw[eid]
        phases = [r.phase for r in rs]
        word = KIND_WORDS.get(e.kind, e.kind)
        where = f"on the connection {e.connection} in the {_phases(phases)}"
        from_expr = f" The value comes from the expression `{e.expr}`." if e.expr else ""
        links = [("about", e.connection), *(("about", p) for p in phases)]

        # ---- heat transfer coefficient
        if e.kind == "fluid_temperature" and e.h_si is None:
            j.find("system.heat_transfer", "major", "context", eid, "no-h",
                   f"No heat transfer coefficient for the effect {eid}",
                   f"The effect {eid} is a fluid temperature, but it gives no heat transfer coefficient `h`. "
                   "The thermal analysis cannot apply it.",
                   f"The effect gives a fluid temperature of {e.display} {where}. "
                   "The heat flow between the fluid and the wall is `h` times the temperature difference. "
                   "Without `h`, the temperature of the wall has no value.",
                   f"Add `h` to the effect {eid}, with its source. "
                   "If the wall has the temperature of the fluid, use the kind `temperature`.",
                   values={"temperature": e.si}, links=links)
        elif e.h_si is not None and e.kind == "fluid_temperature" and e.h_si <= 0.0:
            j.find("system.heat_transfer", "major", "context", eid, "h-not-positive",
                   f"The heat transfer coefficient of the effect {eid} is not positive",
                   f"The effect {eid} gives a heat transfer coefficient of {e.h_si:g} W/(m²·K). "
                   "A heat transfer coefficient must be more than zero.",
                   f"The effect gives a fluid temperature of {e.display} {where}. "
                   f"With `h` = {e.h_si:g} W/(m²·K), the fluid does not heat or cool the wall as the model says.",
                   f"Correct `h` of the effect {eid}, or the parameters that it reads.",
                   values={"h": e.h_si}, links=links)
        elif e.h_si is not None and e.kind != "fluid_temperature":
            j.find("system.heat_transfer", "minor", "context", eid, "h-unused",
                   f"The effect {eid} gives a heat transfer coefficient that nothing uses",
                   f"The effect {eid} has the kind `{e.kind}`, but it gives a heat transfer coefficient `h`. "
                   "Only a fluid temperature uses `h`.",
                   f"The effect gives a {word} of {e.display} and `h` = {e.h_si:g} W/(m²·K) {where}. "
                   "The analysis does not read `h` for this kind of effect.",
                   "If the effect is the temperature of a fluid, change its kind to `fluid_temperature`. "
                   "If it is not, remove `h`.",
                   values={"h": e.h_si}, links=links)

        # ---- values that are not possible, or that do nothing
        if e.kind in ("fluid_temperature", "temperature") and isinstance(e.si, float) and e.si <= 0.0:
            j.find("system.effect_value", "error", "context", eid, "temperature",
                   f"The temperature of the effect {eid} is not possible",
                   f"The effect {eid} gives an absolute temperature of {e.display}. "
                   "An absolute temperature must be more than 0 K.",
                   f"The effect gives this {word} {where}.{from_expr} "
                   "The context shows each temperature on the absolute scale.",
                   "Correct the value or its unit. For a temperature on the Celsius scale, write the unit `degC`.",
                   values={"si": e.si}, links=links)
        if e.kind == "pressure" and isinstance(e.si, float) and e.si < 0.0:
            j.find("system.effect_value", "major", "context", eid, "pressure",
                   f"The pressure of the effect {eid} is negative",
                   f"The effect {eid} gives a pressure of {e.display}. An absolute pressure cannot be negative.",
                   f"The effect gives this pressure {where}.{from_expr} "
                   "The analysis applies a pressure to the faces as the model gives it. "
                   "A negative value pulls on the faces.",
                   "Correct the value. If the pressure is relative to the ambient pressure, "
                   "say so in the description of the effect.",
                   values={"si": e.si}, links=links)
        zero_vector = e.kind in ("force", "moment", "acceleration") and isinstance(e.si, list) \
            and all(c == 0.0 for c in e.si)
        if zero_vector or (e.kind == "pressure" and e.si == 0.0):
            j.find("system.effect_value", "minor", "context", eid, "zero",
                   f"The effect {eid} has the value zero",
                   f"The effect {eid} gives {'an' if word[0] == 'a' else 'a'} {word} of {e.display}. "
                   "The effect applies no load.",
                   f"The effect has this value {where}.{from_expr} "
                   + ("Each component of the vector is zero." if zero_vector else
                      "An absolute pressure of zero is a perfect vacuum."),
                   "Correct the value or the parameters that it reads. If the effect applies no load, remove it.",
                   values={"si": e.si}, links=links)

        # ---- where the number comes from
        if raw.basis == "assumed":
            _assume_effect(j, f, eid)
        elif raw.value is not None:
            literal = [fld for fld, v in (("value", raw.value), ("h", raw.h)) if v is not None and not _names(v, known)]
            if literal and not raw.source.strip():
                shown = e.display if "value" in literal else f"{e.h_si:g} W/(m²·K)"
                j.find("system.effect_source", "minor", "context", eid, "no-source",
                       f"No source for the value of the effect {eid}",
                       f"The effect {eid} gives the literal value {shown} "
                       f"{'for `h` ' if 'value' not in literal else ''}and no source.",
                       f"The effect is a {word} {where}. "
                       f"Its {_list(f'`{x}`' for x in literal)} {'reads' if len(literal) == 1 else 'read'} no parameter, "
                       "so the model has no source for the number. "
                       + _WHAT_A_BASIS_NEEDS.get(raw.basis, "A person cannot check a number without a source."),
                       f"Give the `source` of the effect {eid}. "
                       "As an alternative, make the number a parameter with a source.",
                       values={"si": e.si, "fields": literal, "basis": raw.basis}, links=links)

    # ---- a load on faces that a support holds in the same phase goes into the support
    reported: set[str] = set()
    for p in f.phase_ids:
        held = {(e.connection, f.role(e)): e for e in f.by_phase[p] if e.kind == "support"}
        for e in f.by_phase[p]:
            sup = held.get((e.connection, f.role(e)))
            if sup is None or e.kind not in ("force", "moment", "displacement") or e.id in reported:
                continue
            reported.add(e.id)                             # one finding for each effect, with all its phases
            it = f.interface(e.connection)
            word = KIND_WORDS[e.kind]
            both = [q for q in f.phase_ids if any(x.id == e.id for x in f.by_phase[q])
                    and any(x.id == sup.id for x in f.by_phase[q])]
            j.find("system.load_on_support", "minor", "model", e.id, "on-support",
                   f"The {word} {e.id} acts on faces that a support holds",
                   f"The {word} effect {e.id} acts on the faces that the support {sup.id} holds "
                   f"in the {_phases(both)}. The load goes into the support.",
                   f"The two effects act on the role `{f.role(e)}` of the "
                   + (f"interface {it.id}" if it is not None else f"connection {e.connection}")
                   + f". The effect {e.id} gives {e.display}. A load on held faces does not load the component.",
                   f"Apply the {word} at the interface that carries it. "
                   f"As an alternative, remove the support of the connection {e.connection} from this phase.",
                   values={"si": e.si, "phases": both, "support": sup.id},
                   links=[("about", sup.id), ("about", e.connection), *(("about", q) for q in both)])


def _assume_effect(j: Journal, f: _Facts, eid: str) -> None:
    rs = f.by_id[eid]
    e, (conn, raw) = rs[0], f.raw[eid]
    phases = [r.phase for r in rs]
    word = KIND_WORDS.get(e.kind, e.kind)
    it = f.interface(e.connection)
    value = f"the value {e.display}" if e.display else "no value"
    shown = f"{value} (from the expression `{e.expr}`)" if e.expr else value
    note = raw.note.strip()
    a = j.assume(eid, "assumed", f"Assumed effect {eid}",
                 f"The effect {eid} ({word}) has {shown} in the {_phases(phases)}. "
                 + (f'This effect is an assumption. The system model gives this source: "{raw.source}".'
                    if raw.source.strip() else "This effect is an assumption: no document gives it."),
                 f'The system model gives this justification: "{note}"' if note
                 else "The system model gives no justification for this effect.",
                 f"The effect acts on the component through the connection {e.connection}"
                 + (f" (interface {it.id})" if it is not None else "")
                 + f". A different value changes the loads of the {_phases(phases)}.",
                 verify_by="Compare the effect with a measurement, a calculation or a source document. "
                           "Then change the basis of the effect and give the source.",
                 links=[("about", e.connection), *(("about", p) for p in phases),
                        *([("source", f"doc:{raw.source.strip()}")] if raw.source.strip() else [])],
                 by="rule:system.assumed_effect")
    if not note:
        j.find("system.assumption_justification", "major", "context", eid, "no-justification",
               f"No justification for the assumed effect {eid}",
               f"The effect {eid} has the basis `assumed`, but its note is empty. "
               "The assumption has no justification.",
               f"The system model assumes a {word} with {value} on the connection {e.connection} "
               f"in the {_phases(phases)}. A person who reviews the assumption needs the reason for the value.",
               f"Write the reason for the value in the `note` of the effect {eid}. "
               "If a document gives the value, change the basis and give the source.",
               values={"si": e.si},
               links=[("about", a.id), ("about", e.connection), *(("about", p) for p in phases)])


# --------------------------------------------------------------------------- phases: loads and supports


def _phases_and_support(j: Journal, f: _Facts) -> None:
    mass = (f.geo.get("cad") or {}).get("mass") if f.geo else None
    in_cycle = {p for dc in f.system.duty_cycles for p in dc.sequence}
    for ph in f.system.phases:
        effects = f.by_phase[ph.id]
        loads = f.loads(ph.id)
        named = _named(ph.id, ph.name)

        if ph.duration is not None:
            problem = _duration_problem(ph.duration, f.ctx)
            if problem:
                j.find("system.phase_duration", "minor", "context", ph.id, "duration",
                       f"The duration of the phase {ph.id} is not a positive time",
                       f"The duration of the phase {named} is not a positive time.",
                       problem + " The duration of a phase is a time, for example `120 s`.",
                       f"Correct the duration of the phase {ph.id}.")

        if not loads:
            signals = len(effects)
            j.find("system.phase_effects", "info", "coverage", ph.id, "no-effect",
                   f"The phase {ph.id} has no load on the component",
                   f"No load acts on the element {f.el} in the phase {named}.",
                   (f"The phase has {_n(signals, 'signal effect')} on the element {f.el}, and a signal is not a load. "
                    if signals else f"The system model states no effect on the element {f.el} for this phase. ")
                   + "The analysis has no load for this phase.",
                   "If the component has loads in this phase, add the effects to its connections. "
                   "If it has no load, no action is necessary.",
                   values={"signals": signals})
            continue

        mech = [e for e in loads if e.kind in MECHANICAL and e.kind not in HOLDS]
        held = [e for e in loads if e.kind in HOLDS]
        if mech and not held:
            kinds = Counter(e.kind for e in mech)
            forces = [e.si for e in mech if e.kind == "force" and isinstance(e.si, list)]
            total = [sum(v[i] for v in forces) for i in range(3)] if forces else None
            accel = [e.si for e in mech if e.kind == "acceleration" and isinstance(e.si, list)]
            values: dict[str, Any] = {"kinds": dict(kinds)}
            rationale = (f"The phase has {_n(len(mech), 'mechanical effect')} on the element {f.el}: {_kinds(kinds)}. "
                         "It has no support effect. ")
            if total is not None:
                values["force_sum"] = total
                rationale += f"The sum of the force effects is {units.display(units.from_si(total, 'N'))}. "
            if accel:
                a_sum = [sum(v[i] for v in accel) for i in range(3)]
                values["acceleration"] = a_sum
                if isinstance(mass, (int, float)) and mass > 0:
                    body = [mass * c for c in a_sum]
                    values["mass"], values["body_force"] = float(mass), body
                    rationale += (f"The acceleration acts on the mass of {units.display(units.from_si(float(mass), 'kg'))} "
                                  f"with a body force of {units.display(units.from_si(body, 'N'))}. ")
                else:
                    rationale += "The acceleration acts on the mass of the component as a body force. "
            if kinds.get("pressure"):
                rationale += "The pressure on the wetted faces must balance the other loads. "
            rationale += "If the loads do not balance, a static analysis has no solution."
            j.find("system.phase_support", "info", "coverage", ph.id, "free",
                   f"Nothing holds the component in the phase {ph.id}",
                   f"No connection holds the component in the phase {named}. "
                   "The component is free, so the loads of this phase must balance.",
                   rationale,
                   "Make sure that the loads of this phase balance. "
                   "If a connection holds the component, add a `support` effect to that connection.",
                   values=values,
                   links=[("about", e) for e in dict.fromkeys(x.id for x in mech)])

        if f.system.duty_cycles and ph.category == "operation" and ph.id not in in_cycle \
                and any(e.kind in MECHANICAL or e.kind in THERMAL for e in loads):
            j.find("system.duty_cycle_coverage", "info", "coverage", ph.id, "not-in-duty-cycle",
                   f"No duty cycle contains the operation phase {ph.id}",
                   f"The operation phase {named} has loads on the component, but no duty cycle contains it.",
                   f"The system has {_n(len(f.system.duty_cycles), 'duty cycle')}: "
                   f"{_list(dc.id for dc in f.system.duty_cycles)}. "
                   f"None has {ph.id} in its sequence. A cyclic analysis does not count the loads of this phase.",
                   f"Add the phase {ph.id} to the sequence of a duty cycle. "
                   "If the phase occurs one time only, no action is necessary.",
                   values={"duty_cycles": [dc.id for dc in f.system.duty_cycles]})

    # the component in operation: something must hold it
    operation = [ph.id for ph in f.system.phases if ph.category == "operation"]
    loaded = [p for p in operation if any(e.kind in MECHANICAL and e.kind not in HOLDS for e in f.loads(p))]
    if loaded and not any(e.kind in HOLDS for p in operation for e in f.loads(p)):
        j.find("system.subject_support", "major", "coverage", f.el, "no-support",
               "Nothing holds the component in operation",
               f"No connection holds the component {f.comp.id} in a phase of the category operation. "
               "The system model does not say how the system holds the component.",
               f"The system has {_n(len(operation), 'operation phase')}: {_list(operation)}. "
               f"The element {f.el} has mechanical loads in {_n(len(loaded), 'phase')} of these. "
               "It has no support effect and no displacement effect in a phase of this category. "
               "The loads of each phase must then balance, which is not usual for a part in operation.",
               "Add an effect of the kind `support` to the connection that holds the component. "
               "Give each phase in which the connection holds it.",
               values={"operation_phases": operation, "loaded_phases": loaded},
               links=[("about", f.comp.id), *(("about", p) for p in loaded)])


def _duration_problem(duration: Value, ctx: Context) -> str:
    """Why a duration is wrong, or "" when it is a positive time."""
    try:
        q = duration.quantity(ctx.names())
    except (units.UnitsError, ValueError) as e:
        return f"The context cannot evaluate the duration: {e}."
    if units.is_vector(q) or not units.compatible(q, "s"):
        return f"The model gives a duration of {units.fmt(q)}, which is not a time."
    if units.si(q) <= 0.0:
        return f"The model gives a duration of {units.display(q)}."
    return ""


# --------------------------------------------------------------------------- duty cycles


def _duty_cycles(j: Journal, f: _Facts) -> None:
    for dc in f.system.duty_cycles:
        count = f.ctx.duty_cycles.get(dc.id)
        expr = None if dc.count.is_literal else dc.count.expr
        if not dc.sequence:
            j.find("system.duty_cycle", "major", "model", dc.id, "empty",
                   f"The duty cycle {dc.id} has no phase",
                   f"The duty cycle {_named(dc.id, dc.name)} has no phase in its sequence. "
                   "A cyclic analysis cannot use it.",
                   "A duty cycle is a sequence of phases that repeats. "
                   "The load of the component changes from one phase of the sequence to the next. "
                   "An empty sequence gives no change of load.",
                   f"Give the phases of the duty cycle {dc.id} in `sequence`, in the order in which they occur.")
        if count is not None and count <= 0.0:
            j.find("system.duty_cycle", "major" if count < 0.0 else "minor", "context", dc.id, "count",
                   f"The count of the duty cycle {dc.id} is not positive",
                   f"The duty cycle {_named(dc.id, dc.name)} has a count of {count:g}. "
                   "The count of a duty cycle must be more than zero.",
                   (f"The context evaluates the count to {count:g} from the expression `{expr}`. " if expr else
                    f"The system model gives the count {count:g}. ")
                   + "The count is the number of times that the sequence occurs in the life of the part. "
                   + ("A negative count decreases the calculated damage." if count < 0.0 else
                      "A count of zero adds no damage."),
                   f"Correct the count of the duty cycle {dc.id}"
                   + (" or the parameters that it reads." if expr else ".")
                   + (" If the sequence does not occur, remove the duty cycle." if count == 0.0 else ""),
                   values={"count": count},
                   links=[("about", n) for n in sorted(_names(dc.count, set(f.ctx.params)))])
        empty = [p for p in dict.fromkeys(dc.sequence) if not f.loads(p)]
        if empty:
            j.find("system.duty_cycle", "minor", "coverage", dc.id, "phase-no-effect",
                   f"A phase of the duty cycle {dc.id} has no load on the component",
                   f"The duty cycle {_named(dc.id, dc.name)} contains the {_phases(empty)}, "
                   f"in which no load acts on the component.",
                   f"The sequence of the duty cycle is {_list(dc.sequence)}. "
                   f"The element {f.el} has no load in the {_phases(empty)}. "
                   "A phase without loads adds an unloaded state to the cycle.",
                   f"Add the effects of the {_phases(empty)} to the system model. "
                   "As an alternative, remove the phase from the sequence.",
                   values={"sequence": list(dc.sequence), "phases_without_load": empty},
                   links=[("about", p) for p in empty])


# --------------------------------------------------------------------------- elements


def _elements(j: Journal, f: _Facts) -> None:
    connected = {x for c in f.system.connections for x in (c.from_, c.to) if c.from_ != c.to}
    realized_by = {comp.element: comp.id for comp in f.project.components.values()}
    for el in f.system.elements:
        if el.id not in connected:
            subject = el.id == f.el
            j.find("system.element_connection", "minor", "model", el.id, "no-connection",
                   f"No connection joins the element {el.id} to the system",
                   f"No connection joins the element {_named(el.id, el.name)} to a different element of the system.",
                   f"The system model has {_n(len(f.system.connections), 'connection')}. "
                   f"None has {el.id} as `from` or `to` together with a second element. "
                   + ("The element is the component under verification, so the system cannot load it."
                      if subject else "The element has no function in the system model."),
                   f"Add the connections of the element {el.id}. If the system does not need the element, remove it.")
        want = realized_by.get(el.id)
        if el.component is not None and el.component != want:
            exists = el.component in f.project.components
            j.find("system.element_component", "minor", "model", el.id, "component",
                   f"The element {el.id} names the wrong component",
                   f"The element {el.id} names the component {el.component}, "
                   + ("but that component realizes a different element." if exists else
                      "but the project has no component with this id."),
                   (f"The component {el.component} has `element: {f.project.components[el.component].element}`. "
                    if exists else f"The components of the project are {_list(f.project.components)}. ")
                   + (f"The component {want} has `element: {el.id}`. " if want else
                      f"No component of the project has `element: {el.id}`. ")
                   + "The pipeline reads the `element` field of the component.",
                   f"Set `component` of the element {el.id} to "
                   + (f"{want}." if want else "the component that realizes it, or remove the field."),
                   links=[("about", x) for x in (el.component if exists else None, want) if x])
        elif el.component is None and want is not None:
            j.find("system.element_component", "minor", "model", el.id, "component",
                   f"The element {el.id} does not name its component",
                   f"The component {want} realizes the element {el.id}, but the element does not name it.",
                   f"The component {want} has `element: {el.id}`. The element {el.id} has no `component` field. "
                   "A person who reads the system model cannot see which element is a component of the project.",
                   f"Add `component: {want}` to the element {el.id}.",
                   links=[("about", want)])


# --------------------------------------------------------------------------- the component: temperatures, material


def _component(j: Journal, f: _Facts) -> None:
    comp, ctx, policy = f.comp, f.ctx, f.project.policy
    t_ref = None
    try:
        q = comp.reference_temperature.quantity(ctx.names())
        if units.is_vector(q) or not units.compatible(q, "K"):
            problem = f"The component gives a reference temperature of {units.fmt(q) or 'no unit'}, which is not a temperature."
        else:
            t_ref = float(units.si(q.to("K")))
            problem = "" if t_ref > 0.0 else f"The component gives a reference temperature of {t_ref:g} K."
    except (units.UnitsError, ValueError) as e:
        problem = f"The context cannot evaluate the reference temperature: {e}."
    if problem:
        t_ref = None
        j.find("system.reference_temperature", "major", "model", comp.id, "reference-temperature",
               f"The reference temperature of the component {comp.id} is not an absolute temperature",
               f"The reference temperature of the component {comp.id} is not a positive absolute temperature.",
               problem + " The reference temperature is the temperature at which the part has no thermal stress. "
               "The analysis calculates the thermal strain from it.",
               "Give the reference temperature with its unit, for example `293.15 K` or `20 degC`.")
    if t_ref is None:
        return

    # phases that need a thermal analysis: a temperature away from the reference temperature, or a heat flux
    try:
        delta = float(units.si(units.to(policy.thermal_delta.quantity(ctx.names()), "K")))
    except (units.UnitsError, ValueError):
        return                                   # the policy is not this module's to judge
    hot: dict[str, float] = {}
    for p in f.phase_ids:
        temps = [e.si for e in f.by_phase[p] if e.kind in ("fluid_temperature", "temperature") and isinstance(e.si, float)]
        far = max(temps, key=lambda t: abs(t - t_ref)) if temps else None
        if far is not None and abs(far - t_ref) > delta:
            hot[p] = far
        elif any(e.kind == "heat_flux" and e.si for e in f.by_phase[p]):
            hot[p] = t_ref
    if not hot:
        return
    mat = f.project.material_of()
    shown = _list((f"{p} ({hot[p]:g} K)" if hot[p] != t_ref else f"{p} (a heat flux)") for p in hot)
    why = (f"{_n(len(hot), 'phase')} {'has' if len(hot) == 1 else 'have'} thermal loads away from the reference "
           f"temperature of {t_ref:g} K: {shown}. "
           f"The policy asks for a thermal analysis when the difference is more than {delta:g} K. ")
    for attr, key, name, use in (("k", "no-conductivity", "thermal conductivity", "A thermal analysis cannot calculate the temperatures of the component."),
                                 ("alpha", "no-expansion", "coefficient of thermal expansion", "The analysis cannot calculate the thermal stress.")):
        if getattr(mat, attr) is None:
            j.find("system.thermal_material", "major", "model", mat.id, key,
                   f"The material {mat.id} gives no {name}",
                   f"The material {mat.id} gives no {name} `{attr}`. {use}",
                   why + f"The component {comp.id} has the material {mat.id}.",
                   f"Add `{attr}` to the material {mat.id}, with its source.",
                   values={"phases": list(hot), "reference_temperature": t_ref, "thermal_delta": delta},
                   links=[("about", comp.id), *(("about", p) for p in hot)])
