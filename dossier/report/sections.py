"""The eleven sections of the report, in the order a reviewer reads them.

    1  Summary                      the bottom line, the requirements with their verdicts, findings, reviews, stages
    2  Needs attention              one ordered list of everything that blocks a signature
    3  System context               the diagram, the phases, the effects on the component, the named numbers
    4  Component                    the part, its pictures, interfaces, features, material
    5  Requirements                 each requirement with its source text, criteria, findings and verdict
    6  Load cases                   which load case serves which criterion; each load case
    7  Hand calculations            each as a calculation sheet, with its comparison with the analysis
    8  Analyses                     the model, the meshes, the convergence, the results, the checks
    9  Compliance                   the margins, and the numbers behind each margin
    10 Decisions and assumptions    in pipeline order, with filters
    11 About this report            the STE check, what the tool did not do, the input files, how to make it again

Each function returns the HTML of its section. A section whose stage has not run says so in one sentence.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any

from dossier import units
from dossier.report import cards, charts, diagram, fmt
from dossier.report.data import PIPELINE, STAGE_ORDER, Data
from dossier.report.html import H, esc, join, tag
from dossier.report.page import BADGES, STAGE_WORD, Page, icon

VERDICTS = ("pass", "fail", "inconclusive", "not_assessed")
SEVERITIES = ("error", "major", "minor", "info")
REVIEW_STATES = ("accepted", "rejected", "changes_requested", "stale", "open")
KIND_PLURAL = {"decision": "Decisions", "assumption": "Assumptions", "finding": "Findings"}

# The rules of the review part of "Needs attention": (tone, heading, sentence). The page script uses the same table.
ATT_REVIEW = {
    "rejected": ("fail", "Rejected in a review", "A person rejected this record."),
    "changes_requested": ("warn", "Changes requested", "A person requested changes to this record."),
    "stale": ("stale", "Review out of date", "The record changed after its review."),
    "needs_review": ("serious", "Decision to review", "The decision asks for a review by a person."),
    "not_conservative": ("serious", "Assumption not on the safe side", "The assumption is not on the safe side."),
    "agent": ("info", "Proposal of an agent", "An agent proposed this record. A person must review it."),
}
ATT_ICON = {"fail": "cross", "warn": "question", "stale": "refresh", "serious": "flag", "info": "agent", "none": "dash"}


def review_rule(state: str, needs: bool, conservative: bool | None, agent: bool) -> str | None:
    """Why a reviewable record needs attention, or None. `state` is the review state, "stale" included."""
    if state in ("rejected", "changes_requested", "stale"):
        return state
    if state == "accepted":
        return None
    if needs:
        return "needs_review"
    if conservative is False:
        return "not_conservative"
    if agent:
        return "agent"
    return None


def section(p: Page, number: int, key: str, title: str, body: Any, tools: Any = None, lead: Any = None) -> H:
    p.own.section = title
    sid = p.sid(key)
    head = tag("div", tag("h2", tag("span", str(number), cls="sec-n"), " ", p.own.heading(title)), tools, cls="sec-h")
    return tag("section", head, lead, body, id=sid, cls="sec", data_sec=key)


def open_all(p: Page) -> H:
    return tag("span", tag("button", icon("expand"), " ", p.label("Open all"), type="button", cls="btn xall", data_xall="open"),
               tag("button", p.label("Close all"), type="button", cls="btn xall", data_xall="close"), cls="sec-tools", data_md="skip")


def sub(p: Page, title: str, key: str | None = None, count: Any = None) -> H:
    extra = tag("span", count, cls="h-count") if count not in (None, "") else None
    return p.h(3, title, sid=p.sid(key) if key else None, extra=(H(" ") + extra) if extra else None)


# =========================================================================== 1 Summary


def verdict_of(d: Data, req_id: str) -> str:
    v = d.verdicts.get(req_id)
    return v.verdict if v is not None else "not_assessed"


def title_block(p: Page) -> H:
    d = p.d
    proj, comp, mat = d.project, d.project.subject, d.project.material_of()
    cells = [
        ("Component", join([p.ref(comp.id), tag("span", comp.name, cls="tb-sub")], " ")),
        ("Drawing", comp.drawing or None),
        ("Material", join([p.ref(mat.id), tag("span", mat.name, cls="tb-sub")], " ")),
        ("System", join([p.ref(proj.system.id), tag("span", proj.system.name, cls="tb-sub")], " ")),
        ("Report made", d.built),
        ("Tool", f"dossier {d.version}"),
        ("Digest of the input files", p.code(d.digest)),
        ("Author of the input files", proj.file.author or None),
    ]
    grid = tag("dl", [tag("div", tag("dt", p.label(k)), tag("dd", v), cls="tb-c") for k, v in cells if v], cls="tb-grid")
    return tag("header", tag("div", tag("div", p.label("Verification by analysis"), cls="tb-kicker"),
                             tag("h1", proj.file.name), tag("div", proj.file.id, cls="tb-id"), cls="tb-top"), grid, cls="tb")


def bottom_line(p: Page, n_attention: int) -> H:
    d = p.d
    reqs = d.project.reqs()
    n = Counter(verdict_of(d, r.id) for r in reqs)
    total = len(reqs)
    if not d.ran("assess"):
        lines = [p.t("The assess stage has not run. The requirements have no verdict.")]
    else:
        lines = [p.t("The analysis shows that the part meets {} of {} requirements.", tag("b", n["pass"]), total)]
        if n["fail"]:
            lines.append(p.t("The part does not meet {} requirement.", tag("b", 1)) if n["fail"] == 1 else
                         p.t("The part does not meet {} requirements.", tag("b", n["fail"])))
        if n["inconclusive"]:
            lines.append(p.t("The analysis is inconclusive for {} requirement.", tag("b", 1)) if n["inconclusive"] == 1 else
                         p.t("The analysis is inconclusive for {} requirements.", tag("b", n["inconclusive"])))
        if n["not_assessed"]:
            lines.append(p.t("{} requirement has no verdict.", tag("b", 1)) if n["not_assessed"] == 1 else
                         p.t("{} requirements have no verdict.", tag("b", n["not_assessed"])))
    tiles = []
    for v in VERDICTS:
        tone, ico, word = BADGES["verdict"][v]
        p.own.add("label", word)
        tiles.append(tag("div", tag("span", n[v], cls="tile-n"), tag("span", icon(ico), " ", word, cls="tile-w"),
                         cls=f"tile t-{tone}", data_v=v))
    bar = charts.segments([(BADGES["verdict"][v][0], n[v], BADGES["verdict"][v][2]) for v in VERDICTS], "Verdicts of the requirements")
    att = tag("p", icon("flag"), " ",
              p.t("{} items need attention before a signature. They are in the section {}.",
                  tag("b", n_attention, data_att_count=""), tag("a", p.label("Needs attention"), href="#attention", cls="sub")),
              cls="att-pointer") if n_attention else \
        tag("p", icon("check"), " ", p.t("No item needs attention."), cls="att-pointer ok")
    return tag("div", tag("p", join(lines, " "), cls="bl-text"), tag("div", tiles, cls="tiles"), bar, att, cls="bl")


def requirement_rows(p: Page) -> H:
    d = p.d
    rows = []
    for r in d.project.reqs():
        v = d.verdicts.get(r.id)
        fnd = [f for f in d.findings if f.subject == r.id]
        worst = min((f.severity for f in fnd), key=SEVERITIES.index, default=None)
        gov = H("")
        if v is not None and v.governing:
            item = d.records.get(v.governing)
            gov = p.ref(v.governing, getattr(item, "load_case", None) or v.governing)
        rows.append(([join([p.ref(r.id), tag("span", r.title, cls="rl-t")], " "),
                      p.badge("verdict", v.verdict) if v is not None else
                      (p.badge("verdict", "not_assessed") if d.ran("assess") else tag("span", p.label("no verdict yet"), cls="muted")),
                      (p.num(fmt.margin(v.margin)), {"cls": "r"}) if v is not None and v.margin is not None else ("", {"cls": "r"}),
                      gov,
                      p.badge("severity", worst, fmt.count(len(fnd), "finding")) if fnd else H("")],
                     {"cls": f"v-{v.verdict}" if v is not None else None}))
    return p.table(["Requirement", "Verdict", ("Governing margin", {"cls": "r"}), "Governing load case", "Findings"], rows, cls="t-req")


def review_progress(p: Page) -> H:
    d = p.d
    out = []
    for kind, word in KIND_PLURAL.items():
        ids = [i for i, n in d.graph["nodes"].items() if n["kind"] == kind and i in d.reviews]
        n = Counter(d.review_state(i) for i in ids)
        parts = [(BADGES["review"][s][0], n[s], BADGES["review"][s][2]) for s in REVIEW_STATES]
        legend = join([tag("span", tag("b", cnt), " ", w, cls=f"lg t-{tone}") for tone, cnt, w in parts if cnt], H(" · "))
        out.append(tag("div", tag("div", tag("span", p.label(word), cls="rvp-k"),
                                  tag("span", p.lt("{} of {} accepted", n["accepted"], len(ids)), cls="rvp-n"), cls="rvp-h"),
                       charts.segments(parts, word), tag("div", legend, cls="rvp-l"), cls="rvp", data_rv_kind=kind))
    return join(out)


def stage_rows(p: Page) -> H:
    d = p.d
    rows = []
    for st in PIPELINE:
        files = [f for f in d.files if f.stage == st and f.path.startswith("out")]
        if not files:
            state = p.badge("flag", "not_run")
            when, what = "", H("")
        else:
            stale = [f for f in files if f.stale]
            state = p.badge("flag", "stale_file") if stale else p.badge("flag", "current")
            when = max(f.written for f in files).replace("T", " ").replace("+00:00", " UTC")
            counts: Counter = Counter()
            for f in files:
                counts.update(f.counts)
            if st == "context":
                what = p.lt("{} named numbers, {} effects", len(d.ctx.params), len({e.id for e in d.ctx.effects}))
            elif st == "analyze":
                what = p.lt("{} of {} load cases", len(files), len(d.load_cases))
            else:
                what = esc(", ".join(f"{n} {k.replace('_', ' ')}" for k, n in counts.items()))
        rows.append([p.label(STAGE_WORD.get(st, st)), state, tag("span", when, cls="num muted"), what])
    extra = [f for f in d.files if not f.path.startswith("out")]
    for f in extra:
        rows.append([p.code(f.path), p.badge("flag", "stale_file") if f.stale else p.badge("flag", "current"),
                     tag("span", f.written.replace("T", " ").replace("+00:00", " UTC"), cls="num muted"),
                     esc(", ".join(f"{n} {k.replace('_', ' ')}" for k, n in f.counts.items()))])
    return p.table(["Stage", "State", "Written", "What it holds"], rows, cls="t-stage")


def summary(p: Page, n_attention: int) -> H:
    d = p.d
    proj = d.project
    sev = Counter(f.severity for f in d.findings)
    top = [f for f in d.findings if f.severity in ("error", "major")]
    top.sort(key=lambda f: SEVERITIES.index(f.severity))
    if d.ran("validate") or d.findings:
        fnd = join([
            tag("div", [p.badge("severity", s, f"{sev[s]} {BADGES['severity'][s][2]}") for s in SEVERITIES], cls="sevrow"),
            p.items([p.recline(f.id) for f in top], cls="reclist") if top else None,
            tag("p", p.t("The section {} shows each finding with its evidence.",
                         tag("a", p.label("Requirements and their validation"), href="#requirements", cls="sub")), cls="muted"),
        ])
    else:
        fnd = p.empty("The validate stage has not run. There are no findings.")
    stale = d.stale_files
    stale_note = tag("p", icon("refresh"), " ",
                     p.t("{} stage files come from other input files. Run the stages again.", len(stale)), cls="warnline") if stale else \
        tag("p", p.t("Each stage file comes from the input files of this report."), cls="muted")
    panels = tag("div",
                 tag("div", p.h(3, "Findings by severity"), fnd, cls="panel"),
                 tag("div", p.h(3, "Review progress"), review_progress(p) if d.reviews else
                     p.empty("No stage has written a decision, an assumption or a finding."), cls="panel"),
                 tag("div", p.h(3, "Stages"), stage_rows(p), stale_note, cls="panel"),
                 cls="panels")
    body = join([
        title_block(p),
        p.source(proj.file.description, cite="project.yaml") if proj.file.description else None,
        bottom_line(p, n_attention),
        p.h(3, "Requirements and their verdicts"),
        requirement_rows(p),
        panels,
    ])
    return section(p, 1, "summary", "Summary", body)


# =========================================================================== 2 Needs attention


def _att(tone: str, heading: H, content: Any, **attrs: Any) -> H:
    return tag("li", tag("span", icon(ATT_ICON.get(tone, "dash")), " ", heading, cls="att-c"), tag("span", content, cls="att-b"),
               cls=f"att-i t-{tone}", **attrs)


def attention_items(p: Page) -> tuple[list[H], list[H]]:
    """The lines of "Needs attention": (the lines that do not depend on reviews, the lines that do)."""
    d = p.d
    p.own.section = "Needs attention"
    out: list[H] = []
    for f in d.stale_files:
        out.append(_att("stale", p.label("Stage file out of date"),
                        p.t("The stage file {} comes from other input files. Run its stage again.", p.code(f))))
    for i in sorted((i for i in d.compliance if i.verdict == "fail"), key=lambda i: i.margin if i.margin is not None else -math.inf):
        out.append(_att("fail", p.label("Criterion not met"), join([p.ref(i.id), p.rec_text(i.statement)], " ")))
    for f in d.findings:
        if f.severity == "error":
            out.append(_att("fail", p.label("Error in the inputs"), p.recline(f.id)))
    for an in d.analyses:
        if an.status != "done":
            out.append(_att("serious", p.label("Load case without analysis"),
                            join([p.ref(an.load_case), p.t("The analysis {} has the state {}.", p.ref(an.id), p.badge("analysis", an.status)),
                                  p.rec_text(an.message)], " ")))
    if d.ran("analyze"):
        for lc in d.load_cases:
            if d.analysis_of(lc.id) is None:
                out.append(_att("serious", p.label("Load case without analysis"),
                                join([p.ref(lc.id), p.t("The analysis of this load case has not run.")], " ")))
    for i in d.compliance:
        if i.verdict == "inconclusive":
            out.append(_att("warn", p.label("Criterion inconclusive"),
                            join([p.ref(i.id), p.rec_text(i.reasons[0] if i.reasons else i.statement)], " ")))
    for f in d.findings:
        if f.severity == "major":
            out.append(_att("serious", p.label("Major finding"), p.recline(f.id)))
    for an in d.analyses:
        for c in an.checks:
            if c.passed is False:
                out.append(_att("warn", p.label("Check of an analysis fails"),
                                join([p.ref(an.id), p.rec_text(c.statement), p.ref(c.link) if c.link else None], " ")))
    if d.ran("assess"):
        for r in d.project.reqs():
            v = d.verdicts.get(r.id)
            if v is None or v.verdict == "not_assessed":
                out.append(_att("none", p.label("Requirement without verdict"),
                                join([p.ref(r.id), tag("span", r.title, cls="rl-t"),
                                      p.rec_text(v.statement) if v is not None else p.t("The assess stage gave no verdict.")], " ")))
    dyn: list[tuple[int, H]] = []
    order = list(ATT_REVIEW)
    for rec in list(d.decisions) + list(d.assumptions) + list(d.findings):
        rule = review_rule(d.review_state(rec.id), bool(getattr(rec, "needs_review", False)),
                           getattr(rec, "conservative", None) if rec.kind == "assumption" else None,
                           p.origin_kind(rec.origin.by) == "agent")
        if rule is None:
            continue
        tone, heading, sentence = ATT_REVIEW[rule]
        dyn.append((order.index(rule), _att(tone, p.label(heading), join([p.ref(rec.id), tag("span", rec.title, cls="rl-t"),
                                                                         p.t(sentence)], " "), data_att_dyn=rule)))
    dyn.sort(key=lambda x: x[0])
    return out, [x[1] for x in dyn]


def attention(p: Page, static: list[H], dynamic: list[H]) -> H:
    d = p.d
    n_open = sum(1 for i in d.reviews if d.review_state(i) == "open")
    tail = None
    if d.reviews:
        tail = tag("p", icon("circle"), " ",
                   p.t("{} of {} decisions, assumptions and findings have no review. The section {} shows them.",
                       tag("b", n_open, data_open_count=""), len(d.reviews),
                       tag("a", p.label("Decisions and assumptions"), href="#decisions", cls="sub")), cls="att-tail")
    if not static and not dynamic:
        body = join([p.empty("No item needs attention."), tail])
    else:
        lead = p.say("The list shows what a person must look at before a signature. The most important items come first. "
                     "Each line has a link to its record.", cls="lead-p")
        body = join([lead, tag("ol", static + dynamic, cls="att", id=p.sid("att-list")), tail])
    return section(p, 2, "attention", "Needs attention", body)


# =========================================================================== 3 System context


def _vec_mag(si: Any) -> float | None:
    return fmt.magnitude(si)


def _effect_cell(p: Page, e: Any) -> H:
    """One effect in the table by phase: its value as a link to the effect."""
    if e.kind == "support":
        text = "holds the part"
    elif e.kind == "signal":
        text = "signal"
    elif isinstance(e.si, list):
        text = "|v| = " + fmt.quantity(_vec_mag(e.si), e.unit)
    else:
        text = fmt.quantity(e.si, e.unit)
    return p.ref(e.id, text)


def effects_by_phase(p: Page) -> H:
    """What acts on the component in each phase: rows are the interface and the kind of effect, columns the phases."""
    d = p.d
    comp = d.project.subject
    effects = [e for e in d.ctx.effects if e.on == comp.element]
    if not effects:
        return p.empty("The system context has no effect on the component.")
    by_conn = {it.connection: it for it in comp.interfaces}
    phases = d.project.system.phases
    rows_key: dict[tuple[str, str], dict[str, list]] = {}
    for e in effects:
        rows_key.setdefault((e.connection, e.kind), {}).setdefault(e.phase, []).append(e)
    conn_order = {c.id: i for i, c in enumerate(d.project.system.connections)}
    kinds = ["support", "pressure", "force", "moment", "acceleration", "displacement", "fluid_temperature", "temperature",
             "heat_flux", "mass_flow", "signal"]
    keys = sorted(rows_key, key=lambda k: (conn_order.get(k[0], 99), kinds.index(k[1]) if k[1] in kinds else 99))
    head: list[Any] = ["Interface", "Kind of effect"] + [(join([p.ref(ph.id), tag("span", ph.name, cls="ph-n")], H("<br>")), {"cls": "phc"})
                                                          for ph in phases]
    rows = []
    last = None
    for conn, kind in keys:
        it = by_conn.get(conn)
        first = conn != last
        last = conn
        who = join([p.ref(it.id) if it is not None else p.ref(conn), tag("div", (it.name if it is not None else d.title(conn)), cls="muted")], "") \
            if first else H("")
        cells: list[Any] = [(who, {"cls": "grp"}), kind.replace("_", " ")]
        for ph in phases:
            es = rows_key[(conn, kind)].get(ph.id, [])
            cells.append((join([_effect_cell(p, e) for e in es], H("<br>")), {"cls": "num"}) if es else ("", {"cls": "nil"}))
        rows.append((cells, {"cls": "grp-first" if first else None}))
    return p.table(head, rows, cls="t-pivot")


def effects_detail(p: Page) -> H:
    """Each effect once, by connection: its value, where the value comes from, its phases."""
    d = p.d
    comp = d.project.subject
    by_conn = {it.connection: it for it in comp.interfaces}
    resolved: dict[str, list] = {}
    for e in d.ctx.effects:
        resolved.setdefault(e.id, []).append(e)
    blocks = []
    for c in d.project.system.connections:
        if not c.effects:
            continue
        rows = []
        for ef in c.effects:
            rs = resolved.get(ef.id, [])
            r = rs[0] if rs else None
            value = H("")
            if r is not None and r.si is not None:
                value = p.qty(r.si, r.unit)
                if r.h_si is not None:
                    value = join([value, tag("div", H("h = ") + p.qty(r.h_si, "W/(m^2*K)"), cls="muted")], "")
            elif ef.kind == "support":
                value = p.label("holds the part")
            where = []
            if r is not None and r.expr:
                where.append(p.expr(r.expr))
            elif ef.value is not None:
                where.append(tag("span", p.label("a number of the input file"), cls="muted"))
            if ef.h is not None and not ef.h.is_literal:
                where.append(join([H("h = "), p.expr(str(ef.h.expr))]))
            where.append(p.chip(ef.basis, "basis"))
            if ef.source:
                where.append(p.doc(ef.source))
            rows.append(([p.row_id(ef.id), ef.kind.replace("_", " "), p.refs(ef.phases()), (value, {"cls": "num"}), join(where, " "),
                          p.ref(ef.acts_on or c.to), join([p.source(ef.description, block=False), p.source(ef.note, block=False) if ef.note else None], " "),
                          p.about_refs(ef.id)], {"id": p.anchor(ef.id)}))
        it = by_conn.get(c.id)
        title = join([p.ref(c.id), tag("span", c.name, cls="rl-t"),
                      (H(" · ") + p.label("interface") + H(" ") + p.ref(it.id)) if it is not None else None,
                      tag("span", fmt.count(len(c.effects), "effect"), cls="h-count")], " ")
        blocks.append(p.fold(title, p.table(["Effect", "Kind", "Phases", ("Value", {"cls": "r"}), "Where the value comes from", "Acts on",
                                             "Description", "Records about it"], rows, cls="t-eff"), cls="sub-fold"))
    return join(blocks)


def parameters_table(p: Page) -> H:
    d = p.d
    groups: dict[str, list] = {}
    for name, prm in d.ctx.params.items():
        groups.setdefault(prm.origin, []).append((name, prm))
    blocks = []
    for origin, items in groups.items():
        rows = []
        for name, prm in items:
            where = join([p.expr(prm.expr) if prm.expr else None, p.doc(prm.source) if prm.source else
                          (tag("span", p.label("no source document"), cls="muted") if not prm.expr else None)], " ")
            rows.append(([p.row_id(name), prm.title, (p.qty(prm.si, prm.unit), {"cls": "num"}), p.chip(prm.basis, "basis"), where,
                          p.source(prm.note, block=False) if prm.note else H(""), p.about_refs(name)], {"id": p.anchor(name)}))
        if origin == "system":
            head = p.t("Numbers of the system context")
        elif d.kind(origin) == "requirement":
            head = p.t("Numbers that the requirement {} states", p.ref(origin))
        else:
            head = p.t("Properties of the material {}", p.ref(origin))
        blocks.append(join([tag("h4", head, cls="h-sub"),
                            p.table(["Name", "What it is", ("Value", {"cls": "r"}), "Basis", "Expression or source document", "Note",
                                     "Records about it"], rows, cls="t-par")]))
    basis = Counter(prm.basis for prm in d.ctx.params.values())
    summ = join([p.t("{} named numbers.", tag("b", len(d.ctx.params))), " ",
                 join([tag("span", f"{n} {b}", cls="lg") for b, n in basis.items()], H(" · "))], "")
    return p.fold(summ, join(blocks), cls="sub-fold", sid=p.sid("named-numbers"))


def system_context(p: Page) -> H:
    d = p.d
    s = d.project.system
    comp = d.project.subject
    p.anchor(s.id)
    head = tag("div", tag("div", tag("span", p.label("System"), cls="kind"), " ", tag("a", s.id, cls="self", href=f"#{s.id}"), " ",
                          tag("b", s.name), p.why_button(s.id, small=True), cls="blk-h"),
               p.source(s.description, cite="system.yaml") if s.description else None, id=s.id, cls="blk")
    # ---- diagram
    svg_path = d.project.out / "system.svg"
    if svg_path.is_file():
        try:
            svg = diagram.inline(svg_path.read_text(), lambda i: i in p.known, d.title)
            dia = tag("figure", tag("div", H(svg), cls="dia-scroll", tabindex="0", role="group", aria_label="System diagram"),
                      tag("figcaption", p.t("Select an element or a connection in the diagram to open its record.")), cls="diagram",
                      data_md="[The system diagram is in the HTML report.]")
        except Exception as e:  # noqa: BLE001
            dia = p.empty("The report cannot show the file system.svg. The cause is: {}", p.code(f"{type(e).__name__}: {e}"))
    elif d.ran("validate"):
        dia = p.empty("The file system.svg is missing. The validate stage did not write the system diagram.")
    else:
        dia = p.empty("The validate stage has not run. The system diagram is not available.")
    # ---- elements, connections
    by_el = {c.element: c for c in d.project.components.values()}
    el_rows = [([p.row_id(e.id), e.name, e.kind, e.group, p.ref(by_el[e.id].id) if e.id in by_el else H(""),
                 p.source(e.description, block=False), p.about_refs(e.id)], {"id": p.anchor(e.id), "cls": "subject" if e.id == comp.element else None})
               for e in s.elements]
    by_conn = {it.connection: it for it in comp.interfaces}
    cn_rows = [([p.row_id(c.id), c.name, join([p.ref(c.from_), H(" → "), p.ref(c.to)]), c.kind, c.medium,
                 p.num(str(len(c.effects))) if c.effects else H(""), p.ref(by_conn[c.id].id) if c.id in by_conn else H(""),
                 p.source(c.description, block=False), p.about_refs(c.id)], {"id": p.anchor(c.id)}) for c in s.connections]
    elements = p.fold(p.t("{} elements and {} connections of the system.", tag("b", len(s.elements)), tag("b", len(s.connections))),
                      join([tag("h4", p.own.heading("Elements"), cls="h-sub"),
                            p.table(["Element", "Name", "Kind", "Group", "Component", "Description", "Records about it"], el_rows, cls="t-el"),
                            tag("h4", p.own.heading("Connections"), cls="h-sub"),
                            p.table(["Connection", "Name", "From, to", "Kind", "Medium", "Effects", "Interface of the component", "Description",
                                     "Records about it"], cn_rows, cls="t-el")]), cls="sub-fold")
    # ---- phases
    lcs_of: dict[str, list[str]] = {}
    for lc in d.load_cases:
        if lc.phase:
            lcs_of.setdefault(lc.phase, []).append(lc.id)
    n_eff = Counter(e.phase for e in d.ctx.effects if e.on == comp.element)
    ph_rows = []
    for ph in s.phases:
        dur = ""
        if ph.duration is not None:
            try:
                dur = fmt.quantity(units.si(ph.duration.quantity(d.ctx.names())), "s")
            except Exception:  # noqa: BLE001
                dur = ""
        state = ", ".join(f"{k}: {v}" for k, v in ph.state.items())
        ph_rows.append(([p.row_id(ph.id), ph.name, p.chip(ph.category, own=False), tag("span", dur, cls="num"), state,
                         p.num(str(n_eff.get(ph.id, 0))), p.refs(lcs_of.get(ph.id, [])), p.source(ph.description, block=False),
                         p.about_refs(ph.id)], {"id": p.anchor(ph.id)}))
    phases = p.table(["Phase", "Name", "Category", "Duration", "State", "Effects on the component", "Load cases", "Description",
                      "Records about it"], ph_rows, cls="t-ph")
    dc_rows = []
    for dc in s.duty_cycles:
        cnt = d.ctx.duty_cycles.get(dc.id)
        count = join([p.num(f"{cnt:g}") if cnt is not None else None,
                      (H(" = ") + p.expr(str(dc.count.expr))) if not dc.count.is_literal else None], "")
        dc_rows.append(([p.row_id(dc.id), dc.name, p.refs(dc.sequence, " → "), count, p.doc(dc.source) if dc.source else H(""),
                         join([p.source(dc.description, block=False), p.source(dc.note, block=False) if dc.note else None], " "),
                         p.about_refs(dc.id)], {"id": p.anchor(dc.id)}))
    duty = p.table(["Duty cycle", "Name", "Phases, in sequence", "Repetitions", "Source document", "Description", "Records about it"],
                   dc_rows, cls="t-ph") if dc_rows else p.empty("The system has no duty cycle.")
    body = join([
        head, dia, elements,
        sub(p, "Phases", "phases", len(s.phases)),
        p.say("A phase is an operating state of the system. The effects of one phase act together."),
        phases,
        sub(p, "Duty cycles", "duty-cycles", len(s.duty_cycles)), duty,
        sub(p, "Effects on the component in each phase", "effects"),
        p.say("Each cell is a link to its effect. Text in this style is source text of the input files: {}",
              p.source("quoted as written", block=False)),
        effects_by_phase(p),
        sub(p, "Effects and where their values come from", "effect-list"),
        p.say("A value that comes from an expression shows the expression. Each name in an expression is a link to its number. "
              "The control {} shows the chain down to the source documents.", tag("span", icon("why"), " ", p.label("Why"), cls="inl-why")),
        effects_detail(p),
        sub(p, "Named numbers", "numbers"),
        parameters_table(p),
    ])
    return section(p, 3, "system", "System context", body)


# =========================================================================== 4 Component


def _vec_mm(v: Any) -> str:
    return "(" + ", ".join(f"{1000 * float(c):.1f}" for c in v) + ") mm" if v else ""


def component(p: Page) -> H:
    d = p.d
    proj = d.project
    comp = proj.subject
    mat = proj.material_of()
    geo = d.geo or {}
    cad = geo.get("cad") or {}
    p.anchor(comp.id)
    ref_t = ""
    try:
        ref_t = fmt.quantity(units.si(comp.reference_temperature.quantity()), "K")
    except Exception:  # noqa: BLE001
        pass
    facts = [("Realizes the element", p.ref(comp.element)), ("Drawing", comp.drawing or None), ("Material", p.ref(mat.id)),
             ("Stress-free temperature", p.num(ref_t) if ref_t else None), ("CAD file", p.code(comp.cad.file))]
    if cad:
        box = cad.get("bbox")
        size = ""
        if box and len(box) == 2:
            size = " × ".join(f"{1000 * (b - a):.1f}" for a, b in zip(box[0], box[1])) + " mm"
        facts += [("Digest of the CAD file", p.code(str(cad.get("sha256", ""))[:16]) if cad.get("sha256") else None),
                  ("Volume", p.qty(cad.get("volume"), "m^3") if cad.get("volume") else None),
                  ("Mass", p.qty(cad.get("mass"), "kg") if cad.get("mass") else None),
                  ("Size of the bounding box", p.num(size) if size else None),
                  ("Faces of the CAD", p.num(str(cad.get("n_faces"))) if cad.get("n_faces") else None)]
    head = tag("div", tag("div", tag("span", p.label("Component"), cls="kind"), " ", tag("a", comp.id, cls="self", href=f"#{comp.id}"), " ",
                          tag("b", comp.name), p.why_button(comp.id, small=True), cls="blk-h"),
               p.source(comp.description, cite="component.yaml") if comp.description else None,
               p.source(comp.notes, cite="component.yaml") if comp.notes else None,
               p.kv(facts, cls="facts"), p.about_block(comp.id), id=comp.id, cls="blk")
    if not d.ran("geometry"):
        geo_note = p.empty("The geometry stage has not run. The measured features and the CAD pictures are not available.")
    else:
        geo_note = None
    pics = [(x.get("path", ""), x.get("caption", "")) for x in geo.get("pictures", []) if x.get("path")]
    pictures = p.gallery(pics) if pics else (p.empty("The geometry stage made no picture.") if d.ran("geometry") else None)
    problems = None
    if geo.get("problems"):
        problems = tag("div", p.h(4, "Problems that the geometry stage reports", cls="h-sub"),
                       p.items([join([icon("warn"), p.rec_text(str(x))], " ") for x in geo["problems"]], cls="warns"))
    # ---- interfaces
    gfeat = geo.get("features", {})
    it_rows = []
    for it in comp.interfaces:
        roles = []
        for role, feats in it.features.items():
            n_faces = len((geo.get("interfaces", {}).get(it.id, {}) or {}).get(role, []))
            items = join([p.ref(f) if f != "*" else p.label("every face without another role") for f in feats], ", ")
            roles.append(tag("div", tag("span", cards.ROLE_WORD.get(role, role), cls="role"), ": ", items,
                             tag("span", f" ({fmt.count(n_faces, 'face')})", cls="muted") if n_faces else None))
        it_rows.append(([p.row_id(it.id), it.name, p.ref(it.connection), it.joint.replace("_", " "), join(roles, ""),
                         p.source(it.description, block=False), p.about_refs(it.id)], {"id": p.anchor(it.id)}))
    interfaces = p.table(["Interface", "Name", "Connection of the system", "Joint", "Features by role", "Description", "Records about it"],
                         it_rows, cls="t-if") if it_rows else p.empty("The component has no interface.")
    # ---- features
    used: dict[str, list[str]] = {}
    for it in comp.interfaces:
        for role, feats in it.features.items():
            for f in feats:
                used.setdefault(f, []).append(f"{it.id} ({role})")
    ft_rows = []
    for fid, fd in comp.features.items():
        g = gfeat.get(fid) or {}
        faces = g.get("faces") or []
        n_ok = None if fd.expect is None or not g else (len(faces) == fd.expect)
        found = join([p.num(str(len(faces))) if g else None,
                      (H(" ") + p.badge("check", n_ok, "as expected" if n_ok else f"expected {fd.expect}")) if n_ok is not None else None,
                      tag("div", ", ".join(f"f{t}" for t in faces), cls="muted num") if faces else None], "")
        dims = []
        if g.get("radius"):
            dims.append(H("r = ") + p.qty(g["radius"], "m"))
        if g.get("length"):
            dims.append(H("l = ") + p.qty(g["length"], "m"))
        ft_rows.append(([p.row_id(fid), p.source(fd.description, block=False), g.get("kind", ""), found,
                         (p.qty(g.get("area"), "m^2") if g.get("area") else "", {"cls": "num"}), join(dims, H(", ")),
                         join([p.ref(u.split(" ")[0]) + H(" " + u.split(" ", 1)[1]) for u in used.get(fid, [])], H(", ")),
                         p.about_refs(fid)], {"id": p.anchor(fid)}))
    features = p.fold(p.t("{} features. A feature is a named set of CAD faces.", tag("b", len(comp.features))),
                      p.table(["Feature", "Description", "Kind", "Faces found", ("Area", {"cls": "r"}), "Dimensions", "Used by", "Records about it"],
                              ft_rows, cls="t-ft"), cls="sub-fold") if ft_rows else p.empty("The component has no feature.")
    # ---- stress classification lines
    gscl = geo.get("scls", {})
    scl_rows = []
    for sc in comp.scls:
        g = gscl.get(sc.id) or {}
        state = H("")
        if g:
            state = join([p.badge("check", bool(g.get("ok")), "on the surface" if g.get("ok") else "problem"),
                          p.rec_text(g.get("message", "")) if g.get("message") else None], " ")
        scl_rows.append(([p.row_id(sc.id), p.source(sc.description, block=False), tag("span", _vec_mm(sc.start), cls="num"),
                          tag("span", _vec_mm(sc.end), cls="num"), (p.qty(g.get("length") or math.dist(sc.start, sc.end), "m"), {"cls": "num"}),
                          state, p.about_refs(sc.id)], {"id": p.anchor(sc.id)}))
    scls = p.table(["Line", "Description", "Start", "End", ("Length", {"cls": "r"}), "State", "Records about it"], scl_rows, cls="t-scl") \
        if scl_rows else p.empty("The component has no stress classification line.")
    edges = (geo.get("edges") or {}).get("reentrant") or []
    edge_tab = None
    if edges:
        edge_tab = p.fold(p.t("{} sharp re-entrant edges. The stress at such an edge has no limit in a linear analysis.", tag("b", len(edges))),
                          p.table(["Edge", "Between the faces", "Angle", "Length", "Middle point"],
                                  [[p.num(str(e.get("tag", ""))), ", ".join(f"f{t}" for t in e.get("faces", [])),
                                    p.num(f"{e.get('angle_deg', 0):.0f}°"), p.qty(e.get("length"), "m"),
                                    tag("span", _vec_mm(e.get("midpoint")), cls="num")] for e in edges], cls="t-num"), cls="sub-fold")
    # ---- material
    p.anchor(mat.id)
    names = {"E": "E", "rho": "rho", "alpha": "alpha", "k": "k_th", "cp": "cp", "yield_strength": "Sy", "ultimate_strength": "Su"}
    words = {"E": "Elastic modulus", "rho": "Density", "alpha": "Coefficient of thermal expansion", "k": "Thermal conductivity",
             "cp": "Specific heat", "yield_strength": "Yield strength", "ultimate_strength": "Ultimate strength"}
    mrows = []
    for attr, word in words.items():
        v = getattr(mat, attr)
        if v is None:
            continue
        q = v.quantity()
        mrows.append([p.label(word), (p.qty(units.si(q), units.si_unit(q)), {"cls": "num"}), p.ref(names[attr])])
    mrows.append([p.label("Poisson's ratio"), (p.num(f"{mat.nu:g}"), {"cls": "num"}), p.ref("nu")])
    mfacts: list[tuple[str, Any]] = [("Specification", mat.spec or None), ("Source of the numbers", p.doc(mat.source) if mat.source else None)]
    if mat.properties_temperature is not None:
        mfacts.append(("Temperature of the properties", p.qty(units.si(mat.properties_temperature.quantity()), "K")))
    if mat.service_temperature is not None:
        lo, hi = (units.si(x.quantity()) for x in mat.service_temperature)
        mfacts.append(("Service temperature", p.lt("from {} to {}", p.qty(lo, "K"), p.qty(hi, "K"))))
    if mat.compatibility:
        mfacts.append(("Compatibility", join([p.chip(f"{k}: {v}", own=False) for k, v in mat.compatibility.items()], " ")))
    fat = None
    if mat.fatigue is not None:
        f = mat.fatigue
        sf = units.si(f.sigma_f.quantity())
        bits = [p.lt("model {}", p.code(f.model)), H("σ_f = ") + p.qty(sf, "Pa"), H("b = ") + p.num(f"{f.b:g}"),
                p.lt("mean stress correction {}", p.code(f.mean_stress))]
        if f.endurance_limit is not None:
            bits.append(p.lt("endurance limit {}", p.qty(units.si(f.endurance_limit.quantity()), "Pa")))
        fat = join([join(bits, H(", ")), p.doc(f.source) if f.source else None, p.source(f.note, block=False) if f.note else None], " ")
        mfacts.append(("Fatigue data", fat))
    material = tag("div", tag("div", tag("span", p.label("Material"), cls="kind"), " ", tag("a", mat.id, cls="self", href=f"#{mat.id}"), " ",
                              tag("b", mat.name), p.why_button(mat.id, small=True), cls="blk-h"),
                   p.source(mat.note, cite=proj.file.materials) if mat.note else None,
                   p.kv(mfacts, cls="facts"),
                   p.table(["Property", ("Value", {"cls": "r"}), "Name in expressions"], mrows, cls="t-mat"),
                   p.about_block(mat.id), id=mat.id, cls="blk")
    body = join([
        head, geo_note,
        sub(p, "Pictures of the CAD", "cad-pictures") if pictures else None, pictures, problems,
        sub(p, "Interfaces", "interfaces", len(comp.interfaces)),
        p.say("An interface says which features of the component realize one connection of the system."),
        interfaces, features,
        sub(p, "Stress classification lines", "scls", len(comp.scls)),
        p.say("A stress classification line goes through a wall. The section stresses along it do not depend on the mesh at a sharp edge."),
        scls, edge_tab,
        sub(p, "Material", "material"), material,
    ])
    return section(p, 4, "component", "Component", body)


# =========================================================================== 5 Requirements and their validation


def validation_rules() -> list[dict] | None:
    """The rules of the requirement validation, when the engine exposes them: [{"rule", "description"}]."""
    try:
        from dossier.requirements import validate as v
        rules = getattr(v, "RULES", None)
    except Exception:  # noqa: BLE001
        return None
    if not rules:
        return None
    out = []
    items = rules.items() if isinstance(rules, dict) else [(None, r) for r in rules]
    for key, r in items:
        if isinstance(r, str):
            out.append({"rule": str(key), "description": r})
            continue
        if isinstance(r, dict):
            rid = r.get("id") or r.get("rule") or r.get("name") or key
            desc = r.get("description") or r.get("title") or r.get("doc") or ""
        elif isinstance(r, (tuple, list)):
            rid, desc = (r[0], r[1] if len(r) > 1 else "")
        else:
            rid = getattr(r, "id", None) or getattr(r, "rule", None) or getattr(r, "name", None) or key
            desc = getattr(r, "description", None) or getattr(r, "title", None) or (getattr(r, "__doc__", "") or "")
        if rid:
            out.append({"rule": str(rid), "description": " ".join(str(desc).split()), "makes": str(getattr(r, "makes", "finding"))})
    return out or None


def requirements(p: Page) -> H:
    d = p.d
    proj = d.project
    comp = proj.subject
    reqs = proj.reqs()
    doc = proj.requirement_docs.get(comp.id, "")
    pol = proj.policy
    p.anchor(pol.id)
    frows = [[p.code(k), (p.num(f"{f.value:g}"), {"cls": "num"}), p.doc(f.source) if f.source else H(""),
              p.source(f.note, block=False) if f.note else H("")] for k, f in pol.factors.items()]
    try:
        tdelta = fmt.quantity(units.si(pol.thermal_delta.quantity()), "K")
    except Exception:  # noqa: BLE001
        tdelta = ""
    policy = p.fold(
        join([tag("span", p.label("Policy"), cls="kind"), " ", tag("a", pol.id, cls="self", href=f"#{pol.id}"), " ", tag("b", pol.name),
              p.why_button(pol.id, small=True)]),
        join([p.source(pol.source, cite=proj.file.policy) if pol.source else None, p.source(pol.note, cite=proj.file.policy) if pol.note else None,
              p.table(["Factor", ("Value", {"cls": "r"}), "Source document", "Note"], frows, cls="t-mat") if frows else None,
              p.kv([("Accepted ratio of analysis to hand calculation", p.lt("from {} to {}", p.num(f"{pol.handcalc_band[0]:g}"), p.num(f"{pol.handcalc_band[1]:g}"))),
                    ("Largest discretization uncertainty of a converged result", p.num(fmt.percent(pol.max_uncertainty, 0))),
                    ("Temperature difference that needs a thermal analysis", p.num(tdelta) if tdelta else None)], cls="facts"),
              p.about_block(pol.id)]),
        cls="sub-fold blk-fold", sid=pol.id)
    if d.ran("validate"):
        sev = Counter(f.severity for f in d.findings)
        intro = p.say("The specification {} has {} requirements. The validation compared them with the system context and found {} problems.",
                      tag("b", doc or "of the component"), tag("b", len(reqs)), tag("b", len(d.findings)))
        sevrow = tag("div", [p.badge("severity", s, f"{sev[s]} {BADGES['severity'][s][2]}") for s in SEVERITIES], cls="sevrow")
    else:
        intro = join([p.say("The specification {} has {} requirements.", tag("b", doc or "of the component"), tag("b", len(reqs))),
                      p.empty("The validate stage has not run. The findings are not available.")])
        sevrow = None
    req_cards = [cards.requirement_card(p, r, doc) for r in reqs]
    shown = {f.id for r in reqs for f in d.findings if f.subject == r.id}
    others = [f for f in d.findings if f.id not in shown]
    other_block = None
    if others:
        by_kind: dict[str, list] = {}
        for f in others:
            by_kind.setdefault(d.kind(f.subject) or "other", []).append(f)
        other_block = join([sub(p, "Findings about the system context, the component and the models", "other-findings", len(others)),
                            p.say("These findings are not about one requirement. Each finding names the record it is about."),
                            tag("div", [cards.finding_card(p, f, level=4) for f in sorted(others, key=lambda f: SEVERITIES.index(f.severity))],
                                cls="cards")])
    rules = validation_rules()
    rule_block = None
    if rules and d.ran("validate"):
        by_rule: dict[str, list] = {}
        for f in d.findings:
            by_rule.setdefault(f.rule, []).append(f.id)
        for a in d.assumptions:
            if a.origin.by.startswith("rule:"):
                by_rule.setdefault(a.origin.by[5:], []).append(a.id)
        rows = []
        quiet = 0
        for r in rules:
            made = by_rule.get(r["rule"], [])
            quiet += 0 if made else 1
            rows.append(([p.code(r["rule"]), p.rec_text(r["description"]),
                          p.refs(made) if made else tag("span", icon("check"), " ", p.label("found nothing"), cls="quiet")],
                         {"cls": "made" if made else "quiet-row"}))
        rule_block = join([sub(p, "Rules of the requirement validation", "rules", len(rules)),
                           p.fold(p.t("{} rules ran. {} rules found nothing.", tag("b", len(rules)), tag("b", quiet)),
                                  p.table(["Rule", "What the rule makes sure of", "What it found"], rows, cls="t-rules"), cls="sub-fold")])
    body = join([
        intro, sevrow, policy,
        tag("div", req_cards, cls="cards") if req_cards else p.empty("The component has no requirement."),
        other_block, rule_block,
    ])
    return section(p, 5, "requirements", "Requirements and their validation", body, tools=open_all(p))


# =========================================================================== 6 Load cases


def _crit_rows(d: Data) -> list[tuple[Any, int | None]]:
    rows: list[tuple[Any, int | None]] = []
    for r in d.project.reqs():
        if r.criteria:
            rows += [(r, k) for k in range(len(r.criteria))]
        else:
            rows.append((r, None))
    return rows


def load_case_matrix(p: Page) -> H:
    d = p.d
    lcs = d.load_cases
    head: list[Any] = ["Requirement", "Criterion"] + [(tag("span", p.ref(lc.id), cls="vh"), {"cls": "vcol"}) for lc in lcs]
    rows = []
    last = None
    for r, k in _crit_rows(d):
        first = r.id != last
        last = r.id
        who = join([p.ref(r.id), tag("div", r.title, cls="muted")], "") if first else H("")
        cells: list[Any] = [(who, {"cls": "grp"})]
        if k is None:
            cells.append((tag("span", p.label("no criterion"), cls="muted"), {}))
            cells += [("", {"cls": "nil"}) for _ in lcs]
        else:
            cells.append(tag("a", cards.criterion_label(r, k), href=f"#{r.id}:c{k}", cls="sub"))
            for lc in lcs:
                if any(c.requirement == r.id and c.criterion == k for c in lc.criteria):
                    item = next((i for i in d.compliance if i.requirement == r.id and i.criterion == k and i.load_case == lc.id), None)
                    mark = tag("a", icon("check"), tag("span", p.label("serves"), cls="sr"), href=f"#{item.id if item else lc.id}",
                               cls="mx-on", data_ref=item.id if item else lc.id, title=f"{lc.id} serves this criterion")
                    cells.append((mark, {"cls": "mx"}))
                else:
                    cells.append(("", {"cls": "nil"}))
        rows.append((cells, {"cls": "grp-first" if first else None}))
    foot_cells = [tag("td", p.label("How the tool gets the results"), colspan="2", cls="muted")]
    for lc in lcs:
        word = {"selected": "solved", "scaled": "scaled", "not_supported": "not supported"}[lc.status]
        foot_cells.append(tag("td", tag("span", p.label(word), cls="vh"), cls=f"vcol st-{lc.status}"))
    return p.table(head, rows, cls="t-matrix", foot=tag("tfoot", tag("tr", foot_cells)))


def load_cases(p: Page) -> H:
    d = p.d
    if not d.load_cases:
        why = "The plan stage has not run. There are no load cases." if not d.ran("plan") else "The plan stage selected no load case."
        return section(p, 6, "load-cases", "Load cases", p.empty(why))
    n = Counter(lc.status for lc in d.load_cases)
    intro = p.say("A load case is one phase at its limit loads. One load case serves each criterion of its phase. "
                  "The plan has {} load cases: the tool solves {}, scales {} from a solved load case, and cannot analyse {}.",
                  tag("b", len(d.load_cases)), n["selected"], n["scaled"], n["not_supported"])
    used = {lc.phase for lc in d.load_cases if lc.phase}
    rows = []
    for ph in d.project.system.phases:
        if ph.id in used:
            continue
        recs = d.about.get(ph.id, [])
        why = p.items([join([p.recline(r.id), tag("div", p.rec_text(getattr(r, "choice", None) or getattr(r, "statement", "")), cls="muted")], "")
                       for r in recs], cls="reclist") if recs else p.t("No record says why this phase has no load case.")
        rows.append([join([p.ref(ph.id), tag("span", ph.name, cls="rl-t")], " "), p.chip(ph.category, own=False), why])
    no_lc = join([sub(p, "Phases without a load case", "phases-without", len(rows)),
                  p.table(["Phase", "Category", "The records that say why"], rows, cls="t-nolc")]) if rows else \
        p.say("Each phase of the system has a load case.")
    body = join([
        intro,
        sub(p, "Which load case serves which criterion", "lc-matrix"),
        p.say("A mark is a link to the margin of that criterion in that load case."),
        load_case_matrix(p),
        sub(p, "The load cases", "lc-list", len(d.load_cases)),
        tag("div", [cards.load_case_card(p, lc) for lc in d.load_cases], cls="cards"),
        no_lc,
    ])
    return section(p, 6, "load-cases", "Load cases", body, tools=open_all(p))


# =========================================================================== 7 Hand calculations


def hand_calcs(p: Page) -> H:
    d = p.d
    comp = d.project.subject
    specs = list(comp.hand_calcs)
    by_spec: dict[str, list] = {}
    for h in d.hand_calcs:
        by_spec.setdefault(h.spec, []).append(h)
    if not d.ran("plan") and not d.hand_calcs:
        note = p.empty("The plan stage has not run. The hand calculations have no numbers yet.")
    else:
        note = None
    n_cmp = [c for h in d.hand_calcs for c in h.comparisons]
    agree = Counter(c.agree for c in n_cmp)
    intro = p.say("A hand calculation is a closed-form model of one region of the part. It is a check of the analysis: "
                  "the two results must agree in an accepted band.")
    tally = None
    if n_cmp:
        tally = tag("p", p.t("{} comparisons with the analysis:", tag("b", len(n_cmp))), " ",
                    join([p.badge("agree", k, f"{agree[k]} {BADGES['agree'][k][2]}") for k in (True, False, None) if agree[k]], " "),
                    cls="tally")
    blocks = []
    for spec in specs:
        p.anchor(spec.id)
        sheets = by_spec.get(spec.id, [])
        marks = Counter(c.agree for h in sheets for c in h.comparisons)
        when = []
        if spec.when.analysis:
            when.append(p.lt("analysis types {}", join([p.code(a) for a in spec.when.analysis], ", ")))
        if spec.when.phases:
            when.append(p.lt("phases {}", p.refs(spec.when.phases)))
        if spec.when.needs:
            when.append(p.lt("loads {}", join([p.code(x) for x in spec.when.needs], ", ")))
        inputs = p.table(["Input", "Value or expression"],
                         [[p.code(k), p.expr(str(v.expr)) if not v.is_literal else p.qty(units.si(v.quantity()), units.si_unit(v.quantity()))]
                          for k, v in spec.inputs.items()], cls="t-calc") if spec.inputs else None
        compares = p.table(["Result of the model", "Result of the analysis", "Accepted", "Why the two must agree"],
                           [[p.code(c.output), p.key(c.with_),
                             p.lt("ratio from {} to {}", p.num(f"{c.band[0]:g}"), p.num(f"{c.band[1]:g}")) if c.band else
                             (p.lt("difference up to {}", p.qty(units.si(c.tolerance.quantity()), units.si_unit(c.tolerance.quantity())))
                              if c.tolerance is not None else p.label("the band of the policy")),
                             p.source(c.why, block=False)] for c in spec.compare], cls="t-cmp") if spec.compare else None
        head = join([tag("span", p.label("Hand calculation model"), cls="kind"), " ", tag("a", spec.id, cls="self", href=f"#{spec.id}"), " ",
                     tag("b", spec.title), " ", p.chip(spec.model, own=False), " ",
                     tag("span", fmt.count(len(sheets), "sheet"), cls="h-count"), " ",
                     join([p.badge("agree", k, f"{marks[k]} {BADGES['agree'][k][2]}") for k in (True, False, None) if marks[k]], " "),
                     p.why_button(spec.id, small=True)])
        body = join([
            p.source(spec.description, cite="component.yaml") if spec.description else None,
            p.field("Applies to", join(when, H("; ")) if when else p.t("The model applies to each load case.")),
            p.field("Inputs of the model", inputs) if inputs else None,
            p.field("Comparisons", compares) if compares else None,
            p.about_block(spec.id),
            tag("div", [cards.hand_calc_sheet(p, h) for h in sheets], cls="cards") if sheets else
            p.empty("No load case uses this model. There is no calculation sheet."),
        ])
        blocks.append(p.fold(head, body, cls="sub-fold spec", sid=spec.id, open_=bool(sheets) and len(specs) <= 12))
    orphan = [h for h in d.hand_calcs if h.spec not in {s.id for s in specs}]
    if orphan:
        blocks.append(join([sub(p, "Hand calculations without a model in the component file", "hc-other", len(orphan)),
                            tag("div", [cards.hand_calc_sheet(p, h) for h in orphan], cls="cards")]))
    body = join([intro, note, tally, join(blocks) if blocks else p.empty("The component has no hand calculation model.")])
    return section(p, 7, "hand-calcs", "Hand calculations", body, tools=open_all(p))


# =========================================================================== 8 Analyses


def analyses(p: Page) -> H:
    d = p.d
    if not d.analyses:
        why = "The analyze stage has not run. There are no analyses." if not d.ran("analyze") else "The analyze stage wrote no analysis."
        return section(p, 8, "analyses", "Analyses", p.empty(why))
    n = Counter(a.status for a in d.analyses)
    rs = Counter(r.status for a in d.analyses for r in a.results.values())
    intro = join([
        p.say("Each analysis has one load case. A mesh study with two or more meshes gives the discretization uncertainty of each result. "
              "A result is converged when its uncertainty is not more than {}.", p.num(fmt.percent(d.project.policy.max_uncertainty, 0))),
        tag("p", p.t("{} analyses:", tag("b", len(d.analyses))), " ",
            join([p.badge("analysis", s, f"{n[s]} {BADGES['analysis'][s][2]}") for s in ("done", "failed", "skipped", "not_supported") if n[s]], " "),
            H(" · "), p.t("{} results:", tag("b", sum(rs.values()))), " ",
            join([p.badge("result", s, f"{rs[s]} {BADGES['result'][s][2]}") for s in ("converged", "not_converged", "singular", "estimate", "not_run")
                  if rs[s]], " "), cls="tally"),
    ])
    missing = [lc for lc in d.load_cases if d.analysis_of(lc.id) is None]
    miss = tag("p", icon("info"), " ", p.t("These load cases have no analysis: {}.", p.refs([lc.id for lc in missing])), cls="empty") \
        if missing else None
    body = join([intro, miss, tag("div", [cards.analysis_block(p, a) for a in d.analyses], cls="cards")])
    return section(p, 8, "analyses", "Analyses", body, tools=open_all(p))


# =========================================================================== 9 Compliance


def compliance_matrix(p: Page) -> H:
    d = p.d
    lc_ids = [lc.id for lc in d.load_cases if any(i.load_case == lc.id for i in d.compliance)]
    lc_ids += [i.load_case for i in d.compliance if i.load_case not in lc_ids and i.load_case not in [x.id for x in d.load_cases]]
    lc_ids = list(dict.fromkeys(lc_ids))
    head: list[Any] = ["Requirement", "Criterion"] + [(tag("span", p.ref(i), cls="vh"), {"cls": "vcol"}) for i in lc_ids] + ["Verdict"]
    rows = []
    last = None
    for r, k in _crit_rows(d):
        first = r.id != last
        last = r.id
        v = d.verdicts.get(r.id)
        who = join([p.ref(r.id), tag("div", r.title, cls="muted")], "") if first else H("")
        cells: list[Any] = [(who, {"cls": "grp"})]
        cells.append(tag("a", cards.criterion_label(r, k), href=f"#{r.id}:c{k}", cls="sub") if k is not None else
                     tag("span", p.label("no criterion"), cls="muted"))
        for lc in lc_ids:
            item = next((i for i in d.compliance if i.requirement == r.id and i.criterion == k and i.load_case == lc), None) if k is not None else None
            if item is None:
                cells.append(("", {"cls": "nil"}))
                continue
            tone, ico, word = BADGES["verdict"][item.verdict]
            p.own.add("label", word)
            m = fmt.margin(item.margin)
            cells.append((tag("a", icon(ico), tag("span", m or "—", cls="mx-m"), tag("span", word, cls="sr"), href=f"#{item.id}",
                              data_ref=item.id, cls=f"mx-cell t-{tone}", title=f"{item.id}: {word}" + (f", margin {m}" if m else "")),
                          {"cls": "mx"}))
        cells.append((p.badge("verdict", v.verdict) if v is not None else H(""), {"cls": "vd"}) if first else ("", {}))
        rows.append((cells, {"cls": "grp-first" if first else None}))
    return p.table(head, rows, cls="t-matrix t-cmatrix")


def compliance(p: Page) -> H:
    d = p.d
    if not d.compliance and not d.verdicts:
        why = "The assess stage has not run. There are no margins and no verdicts." if not d.ran("assess") else \
            "The assess stage wrote no compliance item."
        return section(p, 9, "compliance", "Compliance", p.empty(why))
    n = Counter(i.verdict for i in d.compliance)
    intro = join([
        p.say("A compliance item is one criterion of one requirement in one load case. The margin of safety is the limit, divided by the "
              "factor of safety times the response, minus one. The response includes its uncertainty on the unsafe side. "
              "A margin of zero or more is a pass."),
        tag("p", p.t("{} compliance items:", tag("b", len(d.compliance))), " ",
            join([p.badge("verdict", v, f"{n[v]} {BADGES['verdict'][v][2]}") for v in VERDICTS if n[v]], " "), cls="tally"),
    ])
    groups = []
    for r in d.project.reqs():
        items = d.items_of_req(r.id)
        v = d.verdicts.get(r.id)
        if not items and v is None:
            continue
        head = tag("div", p.ref(r.id), " ", tag("b", r.title), " ", p.badge("verdict", v.verdict) if v is not None else None,
                   (H(" ") + tag("span", p.label("margin"), " ", p.num(fmt.margin(v.margin)), cls="chip mg")) if v is not None and v.margin is not None else None,
                   cls="grp-h")
        state = tag("p", p.rec_text(v.statement), cls="grp-s") if v is not None and v.statement else None
        groups.append(tag("div", head, state,
                          tag("div", [cards.compliance_card(p, i) for i in items], cls="cards") if items else None, cls="cgroup"))
    body = join([
        intro,
        sub(p, "Margins of safety by requirement and load case", "margins"),
        p.say("Each cell shows the verdict as an icon and the margin of safety. A cell is a link to its compliance item."),
        tag("div", [tag("span", p.badge("verdict", v), cls="lg") for v in VERDICTS], cls="legend"),
        compliance_matrix(p),
        sub(p, "Compliance items", "items", len(d.compliance)),
        join(groups),
    ])
    return section(p, 9, "compliance", "Compliance", body, tools=open_all(p))


# =========================================================================== 10 Decisions and assumptions


def decisions(p: Page) -> H:
    d = p.d
    recs = list(d.decisions) + list(d.assumptions)
    if not recs:
        return section(p, 10, "decisions", "Decisions and assumptions", p.empty("No stage has written a decision or an assumption."))
    stages = [s for s in STAGE_ORDER if any(r.stage == s for r in recs)]
    stages += sorted({r.stage for r in recs} - set(stages))
    n_stage = Counter(r.stage for r in recs)
    n_state = Counter(d.review_state(r.id) for r in recs)
    n_origin = Counter(p.origin_kind(r.origin.by) for r in recs)

    def chips(group: str, options: list[tuple[str, str, int]]) -> H:
        return tag("span", [tag("button", p.label(word), " ", tag("span", n, cls="count"), type="button", cls="fchip", data_f=group,
                                data_v=value, aria_pressed="false") for value, word, n in options if n], cls="fgroup")

    filters = tag("div",
                  tag("div", tag("label", p.label("Find text"), for_="flt-q", cls="sr"),
                      tag("input", type="search", id="flt-q", placeholder="Find text in the decisions and assumptions", autocomplete="off"),
                      tag("button", icon("flag"), " ", p.label("Needs review"), " ",
                          tag("span", sum(1 for r in recs if getattr(r, "needs_review", False) or getattr(r, "conservative", None) is False), cls="count"),
                          type="button", cls="fchip needs", data_f="needs", data_v="1", aria_pressed="false",
                          title="Decisions that ask for a review, and assumptions that are not on the safe side"),
                      tag("button", p.label("Clear the filters"), type="button", cls="btn", id="flt-clear"), cls="frow"),
                  tag("div", tag("span", p.label("Kind"), cls="flabel"),
                      chips("type", [("decision", "decisions", len(d.decisions)), ("assumption", "assumptions", len(d.assumptions))]),
                      tag("span", p.label("Stage"), cls="flabel"),
                      chips("stage", [(s, STAGE_WORD.get(s, s), n_stage[s]) for s in stages]), cls="frow"),
                  tag("div", tag("span", p.label("Review"), cls="flabel"),
                      chips("review", [(s, BADGES["review"][s][2], n_state[s]) for s in ("open", "accepted", "rejected", "changes_requested", "stale")]),
                      tag("span", p.label("Made by"), cls="flabel"),
                      chips("origin", [(o, o, n_origin[o]) for o in ("rule", "engine", "agent", "person", "input")]), cls="frow"),
                  tag("div", p.lt("{} of {} records are shown.", tag("b", len(recs), id="flt-n"), len(recs)), cls="fcount", aria_live="polite"),
                  cls="filters", id="filters", data_md="skip")
    groups = []
    for s in stages:
        ds = [r for r in d.decisions if r.stage == s]
        as_ = [r for r in d.assumptions if r.stage == s]
        head = tag("h3", p.own.heading(STAGE_WORD.get(s, s)), " ",
                   tag("span", p.lt("{} decisions, {} assumptions", len(ds), len(as_)), cls="h-count"), id=p.sid(f"stage-{s}"))
        groups.append(tag("div", head, tag("div", [cards.decision_card(p, r) for r in ds] + [cards.assumption_card(p, r) for r in as_],
                                           cls="cards"), cls="stage-group", data_stage=s))
    intro = p.say("The stages wrote {} decisions and {} assumptions. A decision is a choice with its reason and its alternatives. "
                  "An assumption is something that the analysis takes as true without proof. A person accepts or rejects each record.",
                  tag("b", len(d.decisions)), tag("b", len(d.assumptions)))
    body = join([intro, filters, join(groups), tag("p", p.t("No record agrees with the filters."), id="flt-none", cls="empty", hidden=True,
                                                   data_md="skip")])
    return section(p, 10, "decisions", "Decisions and assumptions", body, tools=open_all(p))


# =========================================================================== 11 About this report


def about(p: Page, ste: dict | None) -> H:
    d = p.d
    proj = d.project
    # ---- STE
    if ste is None or not ste.get("ran"):
        why = (ste or {}).get("why") or "The STE checker is not installed."
        ste_block = p.empty("The STE check of the text did not run. {}", why)
    else:
        rows = []
        for group in ste.get("groups", []):
            rows.append([p.label(group["label"]), (p.num(str(group["texts"])), {"cls": "num"}), (p.num(str(group["clean"])), {"cls": "num"}),
                         (p.num(str(group["issues"])), {"cls": "num"})])
        by_rule = ste.get("by_rule", {})
        rule_rows = [[p.code(k), (p.num(str(v)), {"cls": "num"})] for k, v in sorted(by_rule.items(), key=lambda kv: -kv[1])]
        rec_rows = []
        for rec in ste.get("records", [])[:200]:
            issues = p.items([join([p.code(i["rule"]), esc(i["message"]),
                                    tag("q", i["text"], cls="muted") if i.get("text") else None], " ") for i in rec["issues"][:6]], cls="plain")
            rec_rows.append([p.ref(rec["id"]), tag("span", d.title(rec["id"]), cls="rl-t"), (p.num(str(len(rec["issues"]))), {"cls": "num"}), issues])
        own_rows = [[tag("q", o["text"]), p.items([join([p.code(i["rule"]), esc(i["message"])], " ") for i in o["issues"][:6]], cls="plain")]
                    for o in ste.get("own", [])[:100]]
        limits = ste.get("limits") or []
        ste_block = join([
            p.say("The STE checker read {} texts. {} texts have no issue. {} texts have one issue or more.",
                  tag("b", ste["texts"]), tag("b", ste["clean"]), tag("b", ste["texts"] - ste["clean"])),
            p.table(["Texts", ("Checked", {"cls": "r"}), ("Without issue", {"cls": "r"}), ("Issues", {"cls": "r"})], rows, cls="t-num") if rows else None,
            p.fold(p.t("Issues by rule: {} rules.", len(rule_rows)), p.table(["Rule", ("Issues", {"cls": "r"})], rule_rows, cls="t-num"),
                   cls="sub-fold") if rule_rows else None,
            p.fold(p.t("Records with issues: {}.", len(ste.get("records", []))),
                   p.table(["Record", "Title", ("Issues", {"cls": "r"}), "The issues"], rec_rows, cls="t-ste"), cls="sub-fold") if rec_rows else None,
            p.fold(p.t("Sentences of this report with issues: {}.", len(ste.get("own", []))),
                   p.table(["Sentence", "The issues"], own_rows, cls="t-ste"), cls="sub-fold") if own_rows else None,
            join([tag("h4", p.own.heading("Limits of the STE checker"), cls="h-sub"),
                  p.say("The checker states these limits about itself:"),
                  p.items([esc(x) for x in limits])]) if limits else None,
            p.say("The check does not include source text of the input files. A person wrote that text, and the report quotes it as written."),
        ])
    # ---- what the tool did not do
    not_done: list[H] = []
    for st in PIPELINE:
        if not d.ran(st):
            not_done.append(p.t("The stage {} has not run.", tag("b", STAGE_WORD.get(st, st))))
    for an in d.analyses:
        if an.status != "done":
            not_done.append(join([p.t("The analysis {} of the load case {} has the state {}.", p.ref(an.id), p.ref(an.load_case),
                                      p.badge("analysis", an.status)), p.rec_text(an.message)], " "))
    for lc in d.load_cases:
        an = d.analysis_of(lc.id)
        if an is None:
            if d.ran("analyze"):
                not_done.append(p.t("The load case {} has no analysis.", p.ref(lc.id)))
            continue
        miss = [k for k in lc.result_keys if k not in an.results or an.results[k].status == "not_run"]
        if miss and an.status == "done":
            not_done.append(p.t("The analysis {} gives no value for: {}.", p.ref(an.id), join([p.code(k) for k in miss], ", ")))
    for an in d.analyses:
        for c in an.checks:
            if c.passed is None:
                not_done.append(join([p.t("A check of the analysis {} did not run.", p.ref(an.id)), p.rec_text(c.statement)], " "))
    for h in d.hand_calcs:
        for c in h.comparisons:
            if c.agree is None:
                not_done.append(p.t("The hand calculation {} has no analysis value for {}.", p.ref(h.id), p.code(c.result_key)))
    for r in proj.reqs():
        if not r.criteria and r.verification == "analysis":
            not_done.append(p.t("The requirement {} has no criterion. The analysis cannot verify it.", p.ref(r.id)))
    used = {lc.phase for lc in d.load_cases if lc.phase}
    if d.ran("plan"):
        for ph in proj.system.phases:
            if ph.id not in used:
                not_done.append(p.t("The phase {} has no load case.", p.ref(ph.id)))
    if p.formulas.failed:
        not_done.append(p.t("The report shows {} formulas as LaTeX source, because the formula engine cannot draw them.", len(p.formulas.failed)))
    fixed = [
        p.t("The tool does a linear analysis. It does not analyse plasticity, contact, buckling or dynamic response."),
        p.t("The tool does not change a requirement. It reports a problem in a requirement as a finding."),
        p.t("The tool does not replace the review of a person. A person accepts or rejects each decision and each assumption."),
    ]
    # ---- files
    in_rows = [[p.code(f["path"]), tag("code", f["sha256"][:16], title=f["sha256"]), (p.num(f"{f['size']:,}".replace(",", " ")), {"cls": "num"})]
               for f in d.input_files]
    st_rows = [[p.code(f.path), p.label(STAGE_WORD.get(f.stage, f.stage)) if f.stage in STAGE_WORD else esc(f.stage), esc(f.tool),
                tag("span", f.written.replace("T", " ").replace("+00:00", " UTC"), cls="num"),
                p.badge("flag", "stale_file") if f.stale else p.badge("flag", "current")] for f in d.files]
    root = p.cli_project()
    cmds = [
        (f"python -m dossier run {root}", "Run each stage again and make this report."),
        (f"python -m dossier run {root} --from report", "Make only this report again from the stage files."),
        (f"python -m dossier status {root}", "Show the stages, the verdicts and the open reviews."),
        (f"python -m dossier show {root} ID", "Show one record with its links."),
        (f"python -m dossier review {root} ID --accept --by NAME", "Give a review of one record."),
    ]
    body = join([
        sub(p, "STE check of the text", "ste"),
        p.say("The text of each record and each sentence of this report must follow ASD-STE100 Simplified Technical English."),
        ste_block,
        sub(p, "What the tool did not do", "not-done"),
        p.items(not_done + fixed, cls="notdone"),
        sub(p, "Input files", "input-files", len(d.input_files)),
        p.say("The digest of the input files of this report is {}. A stage file with a different digest is out of date.", p.code(d.digest)),
        p.table(["File", "SHA-256, first 16 characters", ("Size in bytes", {"cls": "r"})], in_rows, cls="t-files"),
        sub(p, "Stage files", "stage-files", len(d.files)),
        p.table(["File", "Stage", "Tool", "Written", "State"], st_rows, cls="t-files") if st_rows else
        p.empty("No stage has written a file."),
        sub(p, "How to make this report again", "reproduce"),
        p.table(["Command", "What it does"], [[tag("code", c, cls="cmd"), p.t(w)] for c, w in cmds], cls="t-cmd"),
        p.say("The report is the file {}. The same content as plain text is the file {}.", p.code("out/report/index.html"),
              p.code("out/report/report.md")),
    ])
    return section(p, 11, "about", "About this report", body)
