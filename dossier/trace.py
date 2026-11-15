"""The trace graph: every record of a project and how the records depend on each other.

This is the data behind the three flows a reviewer follows:
    result flow          requirement -> load case -> analysis -> margin -> verdict   (and back)
    decision flow        the decisions and assumptions of each stage, in order
    justification flow   for any record: what it rests on, down to a number with its source document

    g = graph(project)            {"nodes": {id: node}, "edges": [edge, ...]}
    node: {"id", "kind", "title", "summary", "status", "stage", "origin", "review"}
    edge: {"from", "to", "rel", "note"}     `from` rests on / refers to `to`

    upstream(g, "CMPL-...")       what a record rests on (a tree, depth first, each id once per branch)
    downstream(g, "meop")         what depends on a record (which results change if this number changes)

Edge relations: the record links (derives_from, verifies, uses, about, evidence, covers, source) plus the
structure of the inputs: part_of (an effect of a connection, a parameter of a requirement), connects (a
connection and its elements), realizes (a component and its element, an interface and its connection),
has (an interface and its features), states (a requirement and its parameters).
"""

from __future__ import annotations

from typing import Any

from dossier import context as context_mod
from dossier.project import Project

KIND_ORDER = ("system", "element", "connection", "phase", "duty_cycle", "effect", "parameter", "component",
              "feature", "interface", "scl", "material", "policy", "requirement", "finding", "load_case",
              "hand_calc_spec", "hand_calc", "analysis", "compliance", "decision", "assumption")


def _short(text: str, n: int = 160) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"


def graph(project: Project, ctx=None) -> dict:
    ctx = ctx or context_mod.load(project)
    nodes: dict[str, dict] = {}
    edges: list[dict] = []

    def node(i: str, kind: str, title: str, summary: str = "", status: str = "", **extra: Any):
        nodes[i] = {"id": i, "kind": kind, "title": title or i, "summary": _short(summary), "status": status, **extra}

    def edge(a: str, b: str, rel: str, note: str = ""):
        if a != b:
            edges.append({"from": a, "to": b, "rel": rel, "note": note})

    s = project.system
    node(s.id, "system", s.name, s.description)
    for e in s.elements:
        node(e.id, "element", e.name, e.description, group=e.group)
        edge(e.id, s.id, "part_of")
    for ph in s.phases:
        node(ph.id, "phase", ph.name, ph.description, status=ph.category)
    for dc in s.duty_cycles:
        node(dc.id, "duty_cycle", dc.name, f"{ctx.duty_cycles.get(dc.id, 0):g} repetitions of " + ", ".join(dc.sequence))
        for ph in dc.sequence:
            edge(dc.id, ph, "uses")
        for n in dc.count.names():
            if n in ctx.params:
                edge(dc.id, n, "derives_from")
    for c in s.connections:
        node(c.id, "connection", c.name, c.description or f"{c.kind} connection" + (f", {c.medium}" if c.medium else ""),
             status=c.kind)
        edge(c.id, c.from_, "connects")
        edge(c.id, c.to, "connects")
    by_effect: dict[str, list] = {}
    for r in ctx.effects:
        by_effect.setdefault(r.id, []).append(r)
    for eid, rs in by_effect.items():
        r = rs[0]
        phases = ", ".join(x.phase for x in rs)
        value = f"{r.display}" + (f" = {r.expr}" if r.expr else "") if r.display else ""
        node(eid, "effect", r.description or f"{r.kind} on {r.on}", f"{r.kind} {value} on {r.on} in {phases}".strip(),
             status=r.kind)
        edge(eid, r.connection, "part_of")
        for x in rs:
            edge(eid, x.phase, "uses")
        for n in r.depends_on:
            edge(eid, n, "derives_from")
    for name, p in ctx.params.items():
        node(name, "parameter", p.title or name, f"{p.display}" + (f" = {p.expr}" if p.expr else "")
             + (f" ({p.source})" if p.source else ""), status=p.basis, origin=p.origin)
        for n in p.depends_on:
            edge(name, n, "derives_from")
        if p.origin not in ("system",):
            edge(name, p.origin, "part_of")
    for m in project.materials.values():
        node(m.id, "material", m.name, m.spec or m.source)
    node(project.policy.id, "policy", project.policy.name, project.policy.source)
    for comp in project.components.values():
        node(comp.id, "component", comp.name, comp.description)
        edge(comp.id, comp.element, "realizes")
        edge(comp.id, comp.material, "uses")
        for fid, f in comp.features.items():
            node(fid, "feature", f.description, f.description)
            edge(fid, comp.id, "part_of")
        for it in comp.interfaces:
            node(it.id, "interface", it.name, it.description, status=it.joint)
            edge(it.id, it.connection, "realizes")
            edge(it.id, comp.id, "part_of")
            for role, feats in it.features.items():
                for f in feats:
                    if f in comp.features:
                        edge(it.id, f, "has", role)
        for scl in comp.scls:
            node(scl.id, "scl", scl.description, scl.description)
            edge(scl.id, comp.id, "part_of")
        for hc in comp.hand_calcs:
            node(hc.id, "hand_calc_spec", hc.title, hc.description, status=hc.model)
            edge(hc.id, comp.id, "part_of")
        for r in project.requirements.get(comp.id, []):
            node(r.id, "requirement", r.title, r.text, status=r.kind)
            edge(r.id, comp.id, "about")
            for name in r.parameters:
                edge(r.id, name, "states")
            for ph in project.phases_of(r):
                edge(r.id, ph, "uses")
            for dc in r.applies.duty_cycles:
                edge(r.id, dc, "uses")
            for c in r.criteria:
                for n in (c.limit.names() if c.limit is not None else set()):
                    if n in ctx.params:
                        edge(r.id, n, "uses")
                if isinstance(c.factor, str) and c.factor in ctx.params:
                    edge(r.id, c.factor, "uses")

    reviews = project.reviews()
    verdicts = {v.requirement: v for v in project.collection("verdicts")}
    for rid, v in verdicts.items():
        if rid in nodes:
            nodes[rid]["verdict"] = v.verdict
            nodes[rid]["margin"] = v.margin
    for rec in project.records().values():
        kind = getattr(rec, "kind", "record")
        summary = (getattr(rec, "statement", "") or getattr(rec, "choice", "") or getattr(rec, "message", "")
                   or getattr(rec, "title", ""))
        status = (getattr(rec, "verdict", None) or getattr(rec, "severity", None) or getattr(rec, "status", None)
                  or getattr(rec, "analysis_type", None) or "")
        rv, stale = project.review_of(rec, reviews)
        node(rec.id, kind, rec.title or rec.id, summary, status=str(status), stage=rec.origin.stage or getattr(rec, "stage", ""),
             origin=rec.origin.by, review=("stale" if stale else rv.state))
        for l in rec.refs():
            edge(rec.id, l.to, l.rel, l.note)
        if kind == "hand_calc":
            edge(rec.id, rec.spec, "derives_from")
    known = set(nodes)
    for e in edges:                                   # a reference to an outside document, or to a missing record
        if e["to"] not in known:
            nodes[e["to"]] = {"id": e["to"], "kind": "document" if e["rel"] == "source" or e["to"].startswith("doc:")
                              else "missing", "title": e["to"], "summary": "", "status": ""}
            known.add(e["to"])
    return {"nodes": nodes, "edges": edges}


def _index(g: dict, forward: bool) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for e in g["edges"]:
        out.setdefault(e["from"] if forward else e["to"], []).append(e)
    return out


def upstream(g: dict, start: str, depth: int = 12) -> dict:
    """What `start` rests on, as a tree: {"id", "rel", "note", "children": [...]}. An id that is already on
    the branch is not expanded again (the graph has no direction guarantee)."""
    idx = _index(g, True)

    def walk(i: str, rel: str, note: str, seen: frozenset, d: int) -> dict:
        kids = [] if d <= 0 else [walk(e["to"], e["rel"], e["note"], seen | {i}, d - 1)
                                  for e in idx.get(i, []) if e["to"] not in seen]
        return {"id": i, "rel": rel, "note": note, "children": kids}

    return walk(start, "", "", frozenset(), depth)


def downstream(g: dict, start: str) -> list[str]:
    """Every record that rests on `start`, directly or through other records, nearest first."""
    idx = _index(g, False)
    out: list[str] = []
    todo = [start]
    while todo:
        i = todo.pop(0)
        for e in idx.get(i, []):
            if e["from"] not in out and e["from"] != start:
                out.append(e["from"])
                todo.append(e["from"])
    return out
