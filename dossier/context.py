"""The resolved system context: every named number evaluated, every effect in SI, per phase.

The namespace of a project (what an expression may read)
    system parameters            system.yaml `parameters`
    requirement parameters       each requirement's `parameters`, under their own names (a number a
                                 requirement states is a claim; publishing it lets the proof pressure be
                                 `proof_factor * meop` and lets the validation compare `meop` with the context)
    the subject's material       E, nu, rho, alpha, k_th, cp, Sy, Su
A name may be defined once. Expressions are evaluated in dependency order; a cycle is an error.

`build(project)` returns the Context. `Context.loads(project, component, phase)` maps the effects of a
phase to the component's interfaces and features (the default feature role of each effect kind is in
dossier.model.DEFAULT_ROLE). `Context.functions(...)` gives the callables hand-calculation inputs use.
"""

from __future__ import annotations

from typing import Any, Callable

from pydantic import Field

from dossier import units
from dossier.model import DEFAULT_ROLE, EFFECT_UNITS, LoadItem, Model, Value
from dossier.project import Project, ProjectError

MATERIAL_NAMES = {"E": "E", "rho": "rho", "alpha": "alpha", "k_th": "k", "cp": "cp", "Sy": "yield_strength",
                  "Su": "ultimate_strength"}


class ResolvedParam(Model):
    name: str
    si: float | list[float]
    unit: str                       # SI unit of `si`
    display: str                    # for people: "10.72 MPa"
    title: str = ""
    expr: str | None = None         # the expression, when derived
    depends_on: list[str] = Field(default_factory=list)
    basis: str = "specified"
    source: str = ""
    note: str = ""
    origin: str = "system"          # system | a requirement id | a material id


class ResolvedEffect(Model):
    id: str
    connection: str
    phase: str
    kind: str
    on: str                         # the element that feels it
    si: float | list[float] | None = None
    unit: str = ""
    display: str = ""
    expr: str | None = None
    depends_on: list[str] = Field(default_factory=list)
    h_si: float | None = None       # W/(m^2 K)
    at_si: list[float] | None = None   # m
    role: str = ""
    description: str = ""
    basis: str = "specified"
    source: str = ""
    note: str = ""


class Context(Model):
    params: dict[str, ResolvedParam] = Field(default_factory=dict)
    effects: list[ResolvedEffect] = Field(default_factory=list)
    duty_cycles: dict[str, float] = Field(default_factory=dict)     # id -> count

    # ---- the namespace for expressions
    def names(self) -> dict[str, Any]:
        return {k: units.from_si(p.si, p.unit) for k, p in self.params.items()}

    def q(self, name: str):
        p = self.params[name]
        return units.from_si(p.si, p.unit)

    def evaluate(self, expr: str, functions: dict[str, Callable] | None = None):
        return units.evaluate(expr, self.names(), functions)

    # ---- effects
    def effects_on(self, element: str, phase: str | None = None, kind: str | None = None) -> list[ResolvedEffect]:
        return [e for e in self.effects if e.on == element and (phase is None or e.phase == phase)
                and (kind is None or e.kind == kind)]

    def loads(self, project: Project, component_id: str | None, phase: str) -> list[LoadItem]:
        """The effects of a phase on a component, mapped to its interfaces and features. `support` effects
        come back as LoadItems of kind "support" (the interface that holds the part)."""
        comp = project.components[component_id or project.file.subject]
        by_conn = {it.connection: it for it in comp.interfaces}
        out = []
        for e in self.effects_on(comp.element, phase):
            if e.kind == "signal":
                continue
            it = by_conn.get(e.connection)
            role = e.role or DEFAULT_ROLE.get(e.kind, "")
            feats = list(it.features.get(role, [])) if it is not None else []
            out.append(LoadItem(effect=e.id, interface=it.id if it is not None else None, kind=e.kind, role=role,
                                features=feats, si=e.si, unit=e.unit, display=e.display, h_si=e.h_si,
                                at_si=e.at_si, note=e.note))
        return out

    def envelope(self, project: Project, component_id: str | None, kind: str, phases: list[str] | None = None,
                 how: str = "max") -> tuple[float, ResolvedEffect] | None:
        """The largest (or smallest) scalar effect of one kind on a component over phases; vectors by
        magnitude. None when there is no such effect."""
        comp = project.components[component_id or project.file.subject]
        best = None
        for e in self.effects_on(comp.element, kind=kind):
            if e.si is None or (phases is not None and e.phase not in phases):
                continue
            v = e.si if not isinstance(e.si, list) else sum(c * c for c in e.si) ** 0.5
            if best is None or (v > best[0] if how == "max" else v < best[0]):
                best = (float(v), e)
        return best

    # ---- callables for hand-calculation inputs
    def functions(self, loads: list[LoadItem] | None = None, geometry: dict | None = None) -> dict[str, Callable]:
        """load("INT-ID", "pressure") -> the load case's value of that kind on that interface (the sum when
        several effects of the kind act there); feat("FT-ID", "radius") -> a measured property of a feature
        (from the geometry stage: area, radius, length, centroid, normal, axis)."""
        loads = loads or []
        feats = (geometry or {}).get("features", {})

        def load(interface: str, kind: str):
            hits = [x for x in loads if x.interface == interface and x.kind == kind and x.si is not None]
            if not hits:
                have = sorted({f"{x.kind}@{x.interface}" for x in loads})
                raise units.UnitsError(f"load({interface!r}, {kind!r}): this load case has no such load "
                                       f"(it has: {', '.join(have) or 'none'})")
            total = None
            for x in hits:
                q = units.from_si(x.si, x.unit) * x.factor
                total = q if total is None else total + q
            return total

        def feat(feature: str, prop: str):
            f = feats.get(feature)
            if f is None:
                raise units.UnitsError(f"feat({feature!r}, ...): the geometry stage has no such feature")
            if prop not in f or f[prop] is None:
                raise units.UnitsError(f"feat({feature!r}, {prop!r}): the feature has no {prop} "
                                       f"(it has: {', '.join(k for k, v in f.items() if v is not None)})")
            unit = {"area": "m^2", "radius": "m", "length": "m", "centroid": "m", "diameter": "m"}.get(prop, "")
            return units.from_si(f[prop], unit)

        return {"load": load, "feat": feat}


# --------------------------------------------------------------------------- building


def _resolve(defs: dict[str, tuple[Value, dict]]) -> tuple[dict[str, Any], list[str]]:
    """Evaluate named values in dependency order. Returns ({name: quantity}, problems)."""
    done: dict[str, Any] = {}
    problems: list[str] = []
    pending = dict(defs)
    while pending:
        progressed = False
        for name in list(pending):
            v, _ = pending[name]
            deps = {n for n in v.names() if n in defs}
            if deps - set(done):
                continue
            try:
                done[name] = v.quantity(done)
            except (units.UnitsError, ValueError) as e:
                problems.append(f"parameter {name}: {e}")
            pending.pop(name)
            progressed = True
        if not progressed:
            cyc = ", ".join(sorted(pending))
            failed = [n for n in pending if any(d not in done and d not in pending for d in pending[n][0].names()
                                                if d in defs)]
            problems.append(f"parameters depend on each other in a circle, or on a parameter that failed: {cyc}"
                            + (f" (waiting on a failed one: {', '.join(failed)})" if failed else ""))
            break
    return done, problems


def definitions(project: Project) -> tuple[dict[str, tuple[Value, dict]], list[str]]:
    """Every named value of the project with its metadata, and the name clashes."""
    defs: dict[str, tuple[Value, dict]] = {}
    problems: list[str] = []

    def add(name: str, v: Value, meta: dict):
        if name in defs:
            problems.append(f"the name {name!r} is defined twice: by {defs[name][1]['origin']} and by {meta['origin']}")
            return
        if name in units.FUNCTIONS or name in units.CONSTANTS:
            problems.append(f"the name {name!r} ({meta['origin']}) is a function or a constant of expressions")
            return
        defs[name] = (v, meta)

    for name, p in project.system.parameters.items():
        add(name, p, {"origin": "system", "title": p.title, "basis": p.basis, "source": p.source, "note": p.note})
    for comp_id, reqs in project.requirements.items():
        for r in reqs:
            for name, p in r.parameters.items():
                add(name, p, {"origin": r.id, "title": f"{r.title} ({name})", "basis": "specified",
                              "source": r.source or project.requirement_docs.get(comp_id, ""), "note": p.note})
    mat = project.material_of()
    for name, attr in MATERIAL_NAMES.items():
        v = getattr(mat, attr)
        if v is not None:
            add(name, v, {"origin": mat.id, "title": f"{mat.name}: {attr.replace('_', ' ')}", "basis": "reference",
                          "source": mat.source or mat.spec, "note": ""})
    add("nu", Value(value=mat.nu), {"origin": mat.id, "title": f"{mat.name}: Poisson's ratio", "basis": "reference",
                                     "source": mat.source or mat.spec, "note": ""})
    return defs, problems


def build(project: Project) -> Context:
    """Evaluate the project's parameters and effects. Raises ProjectError listing every problem."""
    defs, problems = definitions(project)
    values, p2 = _resolve(defs)
    problems += p2
    ctx = Context()
    for name, q in values.items():
        v, meta = defs[name]
        expr = None if v.is_literal else (v.expr if isinstance(v.expr, str) else str(v.expr))
        shown = units.fmt(q, v.unit) if (v.unit and not v.is_literal) else units.display(q)
        ctx.params[name] = ResolvedParam(name=name, si=units.si(q), unit=units.si_unit(q), display=shown, expr=expr,
                                         depends_on=sorted(n for n in v.names() if n in defs), **meta)
    names = dict(values)
    for c in project.system.connections:
        for ef in c.effects:
            on = ef.acts_on or c.to
            for ph in ef.phases():
                r = ResolvedEffect(id=ef.id, connection=c.id, phase=ph, kind=ef.kind, on=on, role=ef.role or "",
                                   description=ef.description, basis=ef.basis, source=ef.source, note=ef.note)
                if ef.value is not None:
                    want, vector = EFFECT_UNITS.get(ef.kind, ("", False))
                    try:
                        q = ef.value.quantity(names)
                        if not units.compatible(q, want):
                            raise units.UnitsError(f"a {ef.kind} must be in {want or 'no unit'}, got {units.fmt(q)}")
                        if units.is_vector(q) != vector:
                            raise units.UnitsError(f"a {ef.kind} must be {'a vector [x, y, z]' if vector else 'one number'}")
                        if vector and len(units.si(q)) != 3:
                            raise units.UnitsError(f"a {ef.kind} needs three components [x, y, z]")
                        q = q.to("K") if want == "K" else q
                        r.si, r.unit = units.si(q), (want if want != "N*m" else "N*m")
                        r.display = units.display(q)
                        r.expr = None if ef.value.is_literal else str(ef.value.expr)
                        r.depends_on = sorted(n for n in ef.value.names() if n in defs)
                        if ef.h is not None:
                            r.h_si = float(units.si(units.to(ef.h.quantity(names), "W/(m^2*K)")))
                            r.depends_on = sorted(set(r.depends_on) | {n for n in ef.h.names() if n in defs})
                        if ef.at is not None:
                            r.at_si = [float(x) for x in units.si(units.to(ef.at.quantity(names), "m"))]
                    except (units.UnitsError, ValueError) as e:
                        problems.append(f"effect {ef.id} ({c.id}, {ph}): {e}")
                        continue
                ctx.effects.append(r)
    for dc in project.system.duty_cycles:
        try:
            ctx.duty_cycles[dc.id] = float(units.si(units.to(dc.count.quantity(names), "")))
        except (units.UnitsError, ValueError) as e:
            problems.append(f"duty cycle {dc.id}: count: {e}")
    if problems:
        raise ProjectError(problems)
    return ctx


def run_stage(project: Project) -> Context:
    ctx = build(project)
    project.write_stage("context", context=ctx)
    return ctx


def load(project: Project) -> Context:
    """The context of the last `context` stage, or a fresh one when the stage has not run."""
    doc = project.read_stage("context")
    if doc is None or doc.get("inputs_digest") != project.inputs_digest():
        return build(project)
    return Context.model_validate(doc["context"])


# --------------------------------------------------------------------------- comparing phases

MECHANICAL = ("pressure", "force", "moment", "acceleration", "displacement")
THERMAL = ("fluid_temperature", "temperature", "heat_flux")


def proportional(a: list[LoadItem], b: list[LoadItem], rtol: float = 1e-6) -> float | None:
    """The scale s with b = s x a, when the mechanical loads of two phases are the same pattern (the same
    kinds on the same interfaces and roles, every value the same multiple, the same supports) and neither
    phase has a thermal load. Then a linear static analysis of `a` gives `b` by scaling. None otherwise."""
    def key(x: LoadItem):
        return (x.interface, x.kind, x.role)

    if any(x.kind in THERMAL for x in (*a, *b)):
        return None
    if sorted(key(x) for x in a if x.kind == "support") != sorted(key(x) for x in b if x.kind == "support"):
        return None

    def mech(items):
        out: dict[tuple, list[float]] = {}
        for x in items:
            if x.kind in MECHANICAL and x.si is not None:
                v = x.si if isinstance(x.si, list) else [x.si]
                acc = out.setdefault(key(x), [0.0] * len(v))
                for i, c in enumerate(v):
                    acc[i] += c * x.factor
        return out

    ma, mb = mech(a), mech(b)
    if set(ma) != set(mb) or not ma:
        return None
    scale = None
    big = max(abs(c) for v in (*ma.values(), *mb.values()) for c in v) or 1.0
    for k, va in ma.items():
        vb = mb[k]
        if len(va) != len(vb):
            return None
        for ca, cb in zip(va, vb):
            if abs(ca) <= rtol * big and abs(cb) <= rtol * big:
                continue
            if abs(ca) <= rtol * big:
                return None
            s = cb / ca
            if scale is None:
                scale = s
            elif abs(s - scale) > rtol * max(abs(scale), 1.0):
                return None
    return scale if scale is not None and scale > 0 else None
