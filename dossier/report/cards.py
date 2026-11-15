"""The card of each kind of derived record, and the card of a requirement.

Every card has the same build: a first line that is always visible (kind, id, title, status, the Why control),
one sentence of summary, and a body of labelled fields. The text of a record is shown as its engine wrote it.
The labels and the joining sentences are the report's own text (ASD-STE100).
"""

from __future__ import annotations

import math
from typing import Any

from dossier import units
from dossier.report import charts, fmt
from dossier.report.html import H, esc, join, tag
from dossier.report.page import BADGES, STAGE_WORD, Page, icon

BASIS_WORD = {"rule": "a rule", "calculation": "a calculation", "engineering_judgement": "engineering judgement",
              "input": "the input files"}
ANALYSIS_WORD = {"static_structural": "static structural", "thermal_steady": "steady thermal",
                 "thermo_structural": "thermo-structural", "fatigue": "fatigue", "flow": "flow"}
ROLE_WORD = {"wetted": "wetted faces", "seat": "seat faces", "bolts": "bolt holes", "external": "external faces",
             "body": "the body"}


def analysis_chip(p: Page, kind: str) -> H:
    return p.chip(ANALYSIS_WORD.get(kind, kind.replace("_", " ")))


def stage_chip(p: Page, stage: str) -> H:
    return p.chip(STAGE_WORD.get(stage, stage), "stage")


def origin_chip(p: Page, by: str) -> H:
    kind = p.origin_kind(by)
    word = {"rule": "rule", "engine": "engine", "agent": "agent", "person": "person", "input": "input"}[kind]
    ico = {"agent": "agent", "person": "person"}.get(kind)
    return tag("span", icon(ico) if ico else None, p.label(word), cls=f"chip o-{kind}", title=by)


def _values_field(p: Page, values: dict) -> H:
    """The numbers of a finding's evidence. A name that is a named number of the project gets its unit."""
    if not values:
        return H("")
    rows = []
    for k, v in values.items():
        prm = p.d.ctx.params.get(k)
        name = p.ref(k) if k in p.known else p.code(k)
        if prm is not None and isinstance(v, (int, float)) and not isinstance(v, bool):
            shown = p.qty(v, prm.unit)
        elif isinstance(v, (int, float, list, tuple)) and not isinstance(v, bool):
            shown = p.num(fmt.plain(v, 5))
        else:
            shown = esc(v)
        rows.append(tag("span", name, " = ", shown, cls="val"))
    return p.field("Numbers of the evidence", join(rows, H(", ")))


# =========================================================================== finding, decision, assumption


def finding_card(p: Page, f: Any, level: int = 5) -> H:
    rv = p.d.review_state(f.id)
    badges = [p.badge("severity", f.severity), p.chip(f.category), origin_chip(p, f.origin.by), p.review_badge(f.id)]
    if p.origin_kind(f.origin.by) == "agent":
        badges.insert(1, p.badge("flag", "agent"))
    subject = p.recline(f.subject) if f.subject in p.known else p.ref(f.subject)
    body = join([
        p.field("What is wrong", p.rec_text(f.statement)),
        p.field("Evidence", p.rec_text(f.rationale)),
        p.field("Proposed correction", p.rec_text(f.suggestion)),
        _values_field(p, f.values),
        p.field("About", subject),
        p.field("Rule", p.code(f.rule)),
        p.links_field(f, skip=[f.subject]),
        p.origin_field(f),
        p.about_block(f.id),
        p.review_block(f.id),
    ])
    return p.card(f.id, f.title, badges=badges, summary=p.rec_text(f.statement), body=body, level=level,
                  data_review=rv, data_sev=f.severity, data_origin=p.origin_kind(f.origin.by))


def decision_card(p: Page, d: Any, level: int = 4) -> H:
    rv = p.d.review_state(d.id)
    okind = p.origin_kind(d.origin.by)
    badges = [p.chip(BASIS_WORD.get(d.basis, d.basis), "basis"), origin_chip(p, d.origin.by)]
    if okind == "agent":
        badges.append(p.badge("flag", "agent"))
    if d.needs_review:
        badges.append(p.badge("flag", "needs_review"))
    badges.append(p.review_badge(d.id))
    alts = H("")
    if d.alternatives:
        alts = p.table(["Alternative", "Why not"], [[p.rec_text(a.option), p.rec_text(a.why_not)] for a in d.alternatives],
                       cls="t-alt")
    body = join([
        p.field("Decision", p.rec_text(d.choice), cls="lead"),
        p.field("Reason", p.rec_text(d.rationale)),
        p.field("Alternatives", alts) if d.alternatives else p.field("Alternatives", p.t("The record gives no alternative.")),
        p.field("About", p.recline(d.subject) if d.subject in p.known else p.ref(d.subject)),
        p.field("Basis", p.t("The decision rests on {}.", BASIS_WORD.get(d.basis, d.basis))),
        p.links_field(d, skip=[d.subject]),
        p.origin_field(d),
        p.about_block(d.id),
        p.review_block(d.id),
    ])
    return p.card(d.id, d.title, badges=badges, summary=p.rec_text(d.choice), body=body, level=level, cls="rv-card",
                  data_review=rv, data_stage=d.stage, data_origin=okind, data_needs="1" if d.needs_review else "0",
                  data_type="decision")


def assumption_card(p: Page, a: Any, level: int = 4) -> H:
    rv = p.d.review_state(a.id)
    okind = p.origin_kind(a.origin.by)
    badges = [p.badge("conservative", a.conservative), origin_chip(p, a.origin.by)]
    if okind == "agent":
        badges.append(p.badge("flag", "agent"))
    badges.append(p.review_badge(a.id))
    safety = {True: "The assumption is on the safe side.", False: "The assumption is not on the safe side.",
              None: "The effect of the assumption on safety is not known."}[a.conservative]
    body = join([
        p.field("Assumption", p.rec_text(a.statement), cls="lead"),
        p.field("Why it is acceptable", p.rec_text(a.rationale)),
        p.field("If it is wrong", p.rec_text(a.impact)),
        p.field("How to verify it", p.rec_text(a.verify_by) if a.verify_by else p.t("The record gives no method.")),
        p.field("Effect on safety", join([p.badge("conservative", a.conservative), p.t(safety)], " ")),
        p.field("About", p.recline(a.subject) if a.subject in p.known else p.ref(a.subject)),
        p.links_field(a, skip=[a.subject]),
        p.origin_field(a),
        p.about_block(a.id),
        p.review_block(a.id),
    ])
    return p.card(a.id, a.title, badges=badges, summary=p.rec_text(a.statement), body=body, level=level, cls="rv-card",
                  data_review=rv, data_stage=a.stage, data_origin=okind,
                  data_needs="1" if a.conservative is False else "0", data_type="assumption")


# =========================================================================== requirement


def _measure_words(key: str) -> str:
    q, m, _ = fmt.split_key(key)
    name = fmt.QUANTITY.get(q, q.replace("_", " "))
    measure = fmt.MEASURE.get(m, m.replace("_", " "))
    if m == "range":
        return f"range of the {name}"
    return f"{measure} {name}".strip()


def factor_value(p: Page, c: Any) -> tuple[float | None, H]:
    """The factor of a criterion as a number, and how to show it (a named factor is a link to its number)."""
    proj = p.d.project
    if isinstance(c.factor, (int, float)):
        return float(c.factor), p.num(fmt.plain(c.factor))
    name = str(c.factor)
    prm = p.d.ctx.params.get(name)
    if prm is not None and not isinstance(prm.si, list):
        return float(prm.si), join([p.ref(name), H(" = "), p.num(fmt.plain(prm.si))])
    if name in proj.policy.factors:
        v = proj.policy.factors[name].value
        return v, join([p.ref(proj.policy.id), " ", p.code(name), H(" = "), p.num(fmt.plain(v))])
    return None, p.code(name)


def limit_value(p: Page, c: Any) -> tuple[float | None, str, H]:
    """The limit of a criterion: SI value, SI unit, and how to show it."""
    d = p.d
    try:
        if c.allowable:
            mat = d.project.material_of()
            v = getattr(mat, c.allowable, None)
            q = v.quantity() if v is not None else None
            name = {"yield_strength": "Sy", "ultimate_strength": "Su"}.get(c.allowable)
            words = c.allowable.replace("_", " ")
            shown = join([p.lt("the {} of the material {}", words, p.ref(mat.id)),
                          H(" (") + (p.ref(name) + H(" = ") if name in p.known else H(""))
                          + (p.qty(units.si(q), units.si_unit(q)) if q is not None else H("")) + H(")")], "")
            return (float(units.si(q)) if q is not None else None), (units.si_unit(q) if q is not None else ""), shown
        q = c.limit.quantity(d.ctx.names())
        si, unit = units.si(q), units.si_unit(q)
        if c.limit.is_literal:
            return float(si), unit, p.qty(si, unit)
        return float(si), unit, join([p.expr(str(c.limit.expr)), H(" = "), p.qty(si, unit)])
    except Exception:  # noqa: BLE001  an expression that cannot be evaluated here is shown as written
        return None, "", p.code(c.allowable or str(getattr(c.limit, "expr", "")))


def criterion_sentence(p: Page, c: Any) -> H:
    """One criterion in words."""
    q, m, loc = fmt.split_key(c.key())
    where = {"part": p.label("in the part"), "scls": p.label("at each stress classification line")}.get(loc) or p.lt("at {}", p.ref(loc))
    fval, fshow = factor_value(p, c)
    _, _, lshow = limit_value(p, c)
    words = _measure_words(c.key())
    if fval is not None and abs(fval - 1.0) < 1e-12:
        if c.comparator == "<=":
            return p.t("The {} {} must not be more than {}.", words, where, lshow)
        return p.t("The {} {} must not be less than {}.", words, where, lshow)
    if c.comparator == "<=":
        return p.t("The {} {}, multiplied by the factor {}, must not be more than {}.", words, where, fshow, lshow)
    return p.t("The {} {}, divided by the factor {}, must not be less than {}.", words, where, fshow, lshow)


def criterion_label(req: Any, idx: int) -> str:
    c = req.criteria[idx] if 0 <= idx < len(req.criteria) else None
    return (c.id if c is not None and c.id else f"criterion {idx + 1}")


def item_chip(p: Page, item: Any) -> H:
    """A compliance item as a small link with its verdict and margin."""
    m = fmt.margin(item.margin)
    return tag("span", p.ref(item.id, item.load_case), " ", p.badge("verdict", item.verdict),
               (H(" ") + p.num(m)) if m else None, cls="itemchip")


def requirement_card(p: Page, req: Any, doc: str, level: int = 4) -> H:
    d = p.d
    v = d.verdicts.get(req.id)
    findings = d.findings_about(req.id)
    findings = [f for f in findings if f.subject == req.id]
    worst = min((f.severity for f in findings), key=lambda s: ["error", "major", "minor", "info"].index(s), default=None)
    badges: list[Any] = [p.chip(req.kind), p.chip(f"by {req.verification}")]
    if v is not None:
        badges.append(p.badge("verdict", v.verdict))
        if v.margin is not None:
            badges.append(tag("span", p.label("margin"), " ", p.num(fmt.margin(v.margin)), cls="chip mg"))
    elif d.ran("assess"):
        badges.append(p.badge("verdict", "not_assessed"))
    if findings:
        badges.append(p.badge("severity", worst, fmt.count(len(findings), "finding")))
    # ---- parameters
    params = H("")
    if req.parameters:
        rows = []
        for name, prm in req.parameters.items():
            rp = d.ctx.params.get(name)
            chk = H("")
            if prm.check is not None:
                chk = join([p.code(f"{name} {prm.check.relation} "), p.expr(prm.check.against),
                            p.source(prm.check.why, block=False) if prm.check.why else None], " ")
            rows.append([p.ref(name), p.qty(rp.si, rp.unit) if rp else H(""), p.chip(prm.role.replace("_", " ")) if prm.role else H(""),
                         chk, p.source(prm.note, block=False) if prm.note else H("")])
        params = p.table(["Number", "Value", "What it must cover", "Check against the system context", "Note"], rows)
    # ---- criteria
    crits = []
    for k, c in enumerate(req.criteria):
        lcs = [lc for lc in d.load_cases if any(r.requirement == req.id and r.criterion == k for r in lc.criteria)]
        items = [i for i in d.items_of_req(req.id) if i.criterion == k]
        sub = [tag("span", p.code(c.key()), cls="muted")]
        if c.note:
            sub.append(p.source(c.note, block=False))
        lines = [tag("div", tag("b", criterion_label(req, k), cls="crit-id"), " ", criterion_sentence(p, c), cls="crit-s"),
                 tag("div", join(sub, " "), cls="crit-k")]
        if items:
            lines.append(tag("div", p.label("Margins"), ": ", join([item_chip(p, i) for i in items], " "), cls="crit-l"))
        elif lcs:
            lines.append(tag("div", p.label("Load cases"), ": ", p.refs([lc.id for lc in lcs]), cls="crit-l"))
        elif d.ran("plan"):
            lines.append(tag("div", p.t("No load case serves this criterion."), cls="crit-l muted"))
        crits.append(tag("li", lines, id=f"{req.id}:c{k}"))
    criteria = tag("ol", crits, cls="crits") if crits else p.t("The requirement has no criterion that analysis can check.")
    # ---- applies to
    phases = d.project.phases_of(req)
    applies = []
    if phases:
        applies.append(tag("span", p.label("phases"), " ", p.refs(phases)))
    if req.applies.category:
        applies.append(tag("span", p.label("category"), " ", p.chip(req.applies.category, own=False)))
    if req.applies.duty_cycles:
        applies.append(tag("span", p.label("duty cycles"), " ", p.refs(req.applies.duty_cycles)))
    # ---- verdict
    verdict = H("")
    if v is not None:
        bits = [p.badge("verdict", v.verdict), p.rec_text(v.statement)]
        if v.governing:
            bits.append(tag("span", p.t("The governing item is {}.", p.ref(v.governing)), cls="muted"))
        verdict = join(bits, " ")
    elif d.ran("assess"):
        verdict = p.t("The assess stage gave no verdict for this requirement.")
    else:
        verdict = p.t("The assess stage has not run.")
    summary = p.source(req.text, block=False)
    body = join([
        p.field("Source text", p.source(req.text, cite=req.source or doc), cls="lead"),
        p.field("Rationale", p.source(req.rationale, block=False)) if req.rationale else H(""),
        p.field("Parent", p.ref(req.parent)) if req.parent else H(""),
        p.field("Numbers it states", params),
        p.field("Criteria", criteria),
        p.field("Applies to", join(applies, H("; ")) if applies else p.t("The requirement names no phase.")),
        p.field("Verification method", p.chip(req.verification)),
        p.field("Verdict", verdict),
        p.field("Findings", join([finding_card(p, f, level + 1) for f in findings])) if findings else
        (p.field("Findings", p.t("The validation found no problem in this requirement.")) if d.ran("validate") else H("")),
        p.about_block(req.id, findings=False),
    ])
    return p.card(req.id, req.title, badges=badges, summary=summary, body=body, level=level, kind="requirement",
                  data_verdict=v.verdict if v else "")


# =========================================================================== load case


def load_value(p: Page, x: Any) -> H:
    """The value of a load of a load case, for people."""
    if x.si is None:
        return H("")
    shown = p.qty(x.si, x.unit)
    if x.h_si is not None:
        shown = join([shown, tag("span", H(", h = ") + p.qty(x.h_si, "W/(m^2*K)"), cls="muted")], "")
    if x.factor != 1.0:
        shown = join([shown, tag("span", f" × {x.factor:g}", cls="muted")], "")
    return shown


def load_case_card(p: Page, lc: Any, level: int = 4) -> H:
    d = p.d
    an = d.analysis_of(lc.id)
    badges: list[Any] = [analysis_chip(p, lc.analysis_type), p.badge("load_case", lc.status)]
    if an is not None and an.status != "done":
        badges.append(p.badge("analysis", an.status, f"analysis {BADGES['analysis'][an.status][2]}"))
    loads = H("")
    if lc.loads:
        rows = []
        for x in lc.loads:
            feats = join([p.ref(f) if f != "*" else esc("every external face") for f in x.features], ", ")
            rows.append([p.ref(x.effect), p.ref(x.interface) if x.interface else p.label("the whole body"), x.kind.replace("_", " "),
                         join([tag("span", ROLE_WORD.get(x.role, x.role), cls="muted"), feats], ": ") if x.role or x.features else H(""),
                         load_value(p, x)])
        loads = p.table(["Effect", "Interface", "Kind", "Acts on", "Value"], rows)
    if lc.supports:
        holds = p.t("These interfaces hold the part: {}.", p.refs(lc.supports))
    elif lc.analysis_type in ("static_structural", "thermo_structural"):
        holds = p.t("No interface holds the part in this load case.")
    else:
        holds = H("")
    scaled = H("")
    if lc.scaled_from:
        scaled = p.t("The tool does not solve this load case. Its results are the results of {} multiplied by {}.",
                     p.ref(lc.scaled_from), p.num(f"{lc.scale:.4g}"))
    elif lc.status == "selected":
        scaled = p.t("The tool solves this load case.")
    elif lc.status == "not_supported":
        scaled = p.t("The tool cannot analyse this load case.")
    crits = []
    for r in lc.criteria:
        try:
            req = d.project.requirement(r.requirement)
        except KeyError:
            req = None
        item = next((i for i in d.items_of_lc(lc.id) if i.requirement == r.requirement and i.criterion == r.criterion), None)
        label = criterion_label(req, r.criterion) if req is not None else f"criterion {r.criterion + 1}"
        crits.append(join([p.ref(r.requirement), tag("span", label, cls="muted"),
                           (H("→ ") + p.ref(item.id) + H(" ") + p.badge("verdict", item.verdict)
                            + (H(" ") + p.num(fmt.margin(item.margin)) if item.margin is not None else H(""))) if item else None], " "))
    cycles = H("")
    if lc.cycles:
        cycles = p.table(["Duty cycle", "Repetitions", "States, in sequence"],
                         [[p.ref(c.duty_cycle), p.num(f"{c.count:g}"), p.refs(c.states, " → ")] for c in lc.cycles])
    used_by = [x.id for x in d.load_cases if lc.id in x.uses or x.scaled_from == lc.id]
    body = join([
        p.field("What it is", p.rec_text(lc.statement), cls="lead"),
        p.field("Phase", p.recline(lc.phase)) if lc.phase else H(""),
        p.field("Solved or scaled", scaled),
        p.field("Loads", loads) if lc.loads else H(""),
        p.field("What holds the part", holds),
        p.field("Results it uses", p.refs(lc.uses)) if lc.uses else H(""),
        p.field("Used by", p.refs(used_by)) if used_by else H(""),
        p.field("Duty cycles", cycles),
        p.field("Criteria it serves", p.items(crits, cls="plain")) if crits else
        p.field("Criteria it serves", p.t("The load case serves no criterion. Other load cases use its results.")
                if used_by else p.t("The load case serves no criterion.")),
        p.field("Hand calculations", p.refs(lc.hand_calcs)) if lc.hand_calcs else H(""),
        p.field("Results the analysis must give", join([p.code(k) for k in lc.result_keys], " ")) if lc.result_keys else H(""),
        p.field("Analysis", p.recline(an.id)) if an is not None else
        p.field("Analysis", p.t("The analysis of this load case has not run.")),
        p.field("Note", p.rec_text(lc.note)) if lc.note else H(""),
        p.about_block(lc.id),
        p.origin_field(lc),
    ])
    return p.card(lc.id, lc.title, badges=badges, summary=p.rec_text(lc.statement), body=body, level=level,
                  data_type=lc.analysis_type)


# =========================================================================== hand calculation


def _hc_unit(hc: Any, comp: Any) -> str:
    out = next((o for o in hc.outputs if o.name == comp.output), None)
    return out.unit if out is not None else ""


def comparison_rows(p: Page, hc: Any) -> list[list]:
    rows = []
    for c in hc.comparisons:
        unit = _hc_unit(hc, c)
        an = p.d.analysis_of(hc.load_case) if hc.load_case else None
        target = f"{an.id}:{c.result_key}" if an is not None and c.result_key in an.results else None
        fe = p.qty(c.fe_value, unit) if c.fe_value is not None else p.label("no value")
        if target:
            fe = tag("a", fe, href=f"#{target}", cls="sub", title="Open the result of the analysis")
        if c.band is not None or c.tolerance_si is None:
            measure = p.num(fmt.ratio(c.ratio)) if c.ratio is not None else H("")
            accepted = p.lt("ratio from {} to {}", p.num(f"{c.band[0]:g}"), p.num(f"{c.band[1]:g}")) if c.band else H("")
        else:
            diff = abs(c.fe_value - c.hand_value) if c.fe_value is not None else None
            measure = p.lt("difference {}", p.qty(diff, unit)) if diff is not None else H("")
            accepted = p.lt("difference up to {}", p.qty(c.tolerance_si, unit))
        rows.append([p.code(c.output), p.qty(c.hand_value, unit), join([fe, tag("div", p.code(c.result_key), cls="muted")], ""),
                     measure, accepted, p.badge("agree", c.agree), p.source(c.why, block=False) if c.why else H("")])
    return rows


def hand_calc_sheet(p: Page, hc: Any, level: int = 5) -> H:
    marks = [p.badge("agree", c.agree) for c in hc.comparisons]
    badges: list[Any] = [p.ref(hc.load_case)] if hc.load_case else []
    badges += marks
    if hc.warnings:
        badges.append(p.badge("severity", "minor", fmt.count(len(hc.warnings), "validity warning")))
    compared = {c.output for c in hc.comparisons}
    main = [o for o in hc.outputs if o.name in compared] or hc.outputs[-1:]
    summary = join([tag("span", p.code(o.name), " = ", p.qty(o.si, o.unit), cls="val") for o in main], H(", "))
    inputs = p.table(["Input", "Value", "Where it comes from"],
                     [[p.code(i.name), p.qty(i.si, i.unit), p.rec_text(i.source)] for i in hc.inputs], cls="t-calc") if hc.inputs else H("")
    steps = []
    for s in hc.steps:
        steps.append(tag("li", tag("div", p.rec_text(s.text), cls="step-t"),
                         tag("div", p.formula(s.latex), tag("span", "= ", s.result, cls="step-r") if s.result else None,
                             cls="step-f") if (s.latex or s.result) else None))
    outputs = p.table(["Result", "Value", "What it is"],
                      [[p.code(o.name), p.qty(o.si, o.unit), p.rec_text(o.description)] for o in hc.outputs], cls="t-calc") if hc.outputs else H("")
    comp = H("")
    if hc.comparisons:
        comp = p.table(["Result", "Hand calculation", "Analysis", "Comparison", "Accepted", "Mark", "Why the two must agree"],
                       comparison_rows(p, hc), cls="t-cmp")
    body = join([
        p.kv([("Model", p.code(hc.model)), ("Hand calculation model", p.ref(hc.spec)),
              ("Load case", p.ref(hc.load_case) if hc.load_case else None), ("Reference", p.rec_text(hc.reference))], cls="sheet-head"),
        p.field("Inputs", inputs),
        p.field("Steps", tag("ol", steps, cls="steps")) if steps else H(""),
        p.field("Results", outputs),
        p.field("Assumptions of the formula", p.items([p.rec_text(a) for a in hc.assumptions])) if hc.assumptions else H(""),
        p.field("Validity warnings", p.items([join([icon("warn"), p.rec_text(w)], " ") for w in hc.warnings], cls="warns"))
        if hc.warnings else H(""),
        p.field("Comparison with the analysis", comp) if hc.comparisons else
        p.field("Comparison with the analysis", p.t("The model gives no comparison for this hand calculation.")),
        p.links_field(hc, skip=[hc.load_case or ""]),
        p.about_block(hc.id),
        p.origin_field(hc),
    ])
    return p.card(hc.id, hc.title, badges=badges, summary=summary, body=body, level=level, cls="sheet")


# =========================================================================== analysis


def _status_sentence(p: Page, r: Any, max_unc: float) -> H:
    """The convergence status of one result in words."""
    if r.status == "converged":
        return p.t("Converged. The discretization uncertainty is {}.", p.num(fmt.percent(r.uncertainty))) if r.uncertainty is not None \
            else p.t("Converged.")
    if r.status == "not_converged":
        return p.t("Not converged. The discretization uncertainty is {}. The limit is {}.", p.num(fmt.percent(r.uncertainty)),
                   p.num(fmt.percent(max_unc))) if r.uncertainty is not None else p.t("Not converged.")
    if r.status == "singular":
        return p.t("Singular. The value increases without a limit when the mesh becomes finer.")
    if r.status == "estimate":
        return p.t("An estimate. The value does not come from a converged analysis.")
    return p.t("The result did not run.")


def _where(p: Page, r: Any) -> H:
    bits = []
    if r.near:
        bits.append(p.rec_text(r.near))
    if r.location:
        bits.append(tag("span", "(" + ", ".join(f"{1000 * c:.1f}" for c in r.location) + ") mm", cls="num muted",
                        title="x, y, z in the CAD frame"))
    return join(bits, " ")


def result_value(p: Page, r: Any) -> H:
    if r.value is None:
        return tag("span", p.label("no value"), cls="muted")
    shown = p.qty(r.value, r.unit, r.uncertainty)
    if r.uncertainty is not None:
        shown = join([shown, tag("span", H("± ") + esc(fmt.percent(r.uncertainty)), cls="unc")], " ")
    return shown


def results_table(p: Page, an: Any, keys: list[str]) -> H:
    d = p.d
    rows = []
    for key in keys:
        r = an.results[key]
        used = [p.ref(i.id, f"{i.requirement} {criterion_label(_req(d, i.requirement), i.criterion)}") for i in d.compliance
                if i.analysis == an.id and i.result_key == key]
        used += [p.ref(h.id) for h in d.hand_calcs_of(an.load_case) if any(c.result_key == key for c in h.comparisons)]
        levels = H("")
        if len(r.per_level) > 1:
            k, ulabel = charts.unit_scale(r.per_level[-1], r.unit)
            levels = tag("span", " → ".join(fmt.plain(v * k, 4) for v in r.per_level), f" {ulabel}".rstrip(), cls="num muted")
        note = tag("div", p.rec_text(r.note), cls="muted") if r.note else None
        extra = []
        if r.extrapolated is not None:
            extra.append(p.lt("extrapolated {}", p.qty(r.extrapolated, r.unit)))
        if r.order is not None:
            extra.append(p.lt("observed order {}", p.num(f"{r.order:.1f}")))
        if r.method != "fe":
            extra.append(p.chip({"hand_calc": "hand calculation", "derived": "derived", "scaled": "scaled"}.get(r.method, r.method)))
        rows.append(([join([p.key(key), note], ""), result_value(p, r), p.badge("result", r.status), _where(p, r),
                      join([levels, tag("div", join(extra, H(", ")), cls="muted") if extra else None], ""), join(used, ", ")],
                     {"id": f"{an.id}:{key}", "cls": f"res s-{r.status}"}))
    return p.table(["Result", "Value", "Status", "Where the extreme is", "Mesh levels, coarse to fine", "Used by"], rows, cls="t-res")


def _req(d: Any, rid: str) -> Any:
    try:
        return d.project.requirement(rid)
    except KeyError:
        return None


def convergence_grid(p: Page, an: Any, keys: list[str]) -> H:
    max_unc = p.d.project.policy.max_uncertainty
    nodes = [m.nodes for m in an.mesh]
    cells = []
    for key in keys:
        r = an.results[key]
        if len(r.per_level) < 2:
            continue
        words, loc = fmt.key_words(key)
        svg = charts.convergence(r.per_level, r.unit, r.uncertainty if r.status in ("converged", "not_converged") else None,
                                 r.extrapolated, nodes if len(nodes) == len(r.per_level) else None, label=key)
        cells.append(tag("figure", tag("figcaption", tag("a", words, tag("span", f" @ {loc}", cls="muted"), href=f"#{an.id}:{key}",
                                                         cls="sub")),
                         svg, tag("div", p.badge("result", r.status), " ", _status_sentence(p, r, max_unc), cls="cv-s"),
                         cls=f"cvfig s-{r.status}"))
    return tag("div", cells, cls="cvgrid") if cells else H("")


def analysis_block(p: Page, an: Any, level: int = 4) -> H:
    d = p.d
    lc = d.lc(an.load_case)
    by_status: dict[str, int] = {}
    for r in an.results.values():
        by_status[r.status] = by_status.get(r.status, 0) + 1
    badges: list[Any] = [analysis_chip(p, an.analysis_type), p.badge("analysis", an.status)]
    for st in ("not_converged", "singular", "estimate", "not_run"):
        if by_status.get(st):
            badges.append(p.badge("result", st, f"{by_status[st]} {BADGES['result'][st][2]}"))
    failed = [c for c in an.checks if c.passed is False]
    if an.checks:
        if failed:
            badges.append(p.badge("check", False, fmt.count(len(failed), "check fails", "checks fail")))
        else:
            badges.append(p.badge("check", True, "checks pass") if all(c.passed for c in an.checks) else
                          p.badge("check", None, "checks not complete"))
    summary = p.rec_text(an.message if an.status != "done" and an.message else an.statement)
    # ---- model
    model = H("")
    if an.bcs:
        rows = []
        for b in an.bcs:
            where = join([p.refs(b.features) if b.features else None,
                          tag("span", ", ".join(b.faces), cls="muted num") if b.faces else None], " ")
            shown, si = fmt.bc_value(b.type, b.value)
            dec = H("")
            if b.decision:
                rec = d.records.get(b.decision)
                dec = join([p.ref(b.decision), tag("div", p.rec_text(rec.choice), cls="muted") if rec is not None and hasattr(rec, "choice") else None], "")
            rows.append([p.chip(b.type.replace("_", " "), own=False), where, p.num(shown, si) if shown else H(""),
                         join([p.ref(b.effect) if b.effect else None, tag("span", p.ref(b.interface), cls="muted") if b.interface else None], " "),
                         dec, p.rec_text(b.description)])
        model = p.table(["Support or load", "Features", "Value", "Effect and interface", "Decision behind it", "What it does"], rows,
                        cls="t-bc")
    material = H("")
    if an.material:
        munit = {"E": "Pa", "rho": "kg/m^3", "alpha": "1/K", "k": "W/(m*K)", "cp": "J/(kg*K)", "sigma_f": "Pa", "yield_strength": "Pa",
                 "ultimate_strength": "Pa", "Sy": "Pa", "Su": "Pa"}
        material = join([tag("span", p.code(k), " = ", p.qty(v, munit.get(k, "")) if isinstance(v, (int, float)) and not isinstance(v, bool)
                             else esc(v), cls="val") for k, v in an.material.items()], H(", "))
    pics = [(a.path, a.caption) for a in an.artifacts if a.kind == "picture"]
    files = [a for a in an.artifacts if a.kind != "picture"]
    mesh = H("")
    if an.mesh:
        mesh = p.table(["Level", "Requested size", "Nodes", "Cells", "Degrees of freedom", "Order", "Smallest quality", "Typical size"],
                       [[p.num(f"L{m.level}"), p.qty(m.h, "m"), p.num(f"{m.nodes:,}".replace(",", " ")), p.num(f"{m.cells:,}".replace(",", " ")),
                         p.num(f"{m.dofs:,}".replace(",", " ")) if m.dofs else H(""), p.num(str(m.order)),
                         p.num(f"{m.min_quality:.2f}") if m.min_quality is not None else H(""),
                         p.qty(m.h_rep, "m") if m.h_rep else H("")] for m in an.mesh], cls="t-num")
    keys = list(an.results)
    key_set = [k for k in (lc.result_keys if lc is not None else []) if k in an.results] or keys
    others = [k for k in keys if k not in key_set]
    conv = convergence_grid(p, an, key_set)
    missing = [k for k in (lc.result_keys if lc is not None else []) if k not in an.results]
    checks = H("")
    if an.checks:
        checks = p.items([join([p.badge("check", c.passed), tag("span", p.rec_text(c.statement)),
                                tag("span", H("(") + p.code(c.kind)
                                    + (H(", ") + p.lt("value {}, limit {}", p.num(fmt.plain(c.value)), p.num(fmt.plain(c.limit)))
                                       if c.value is not None and c.limit is not None else H(""))
                                    + H(")"), cls="muted"),
                                p.ref(c.link) if c.link else None], " ") for c in an.checks], cls="checks")
    own_recs = [r.id for r in list(d.decisions) + list(d.assumptions) + list(d.findings)
                if d.source_of.get(r.id) == d.source_of.get(an.id)]
    body = join([
        p.kv([("Load case", p.ref(an.load_case)), ("Solver", an.solver or None),
              ("Run time", p.num(f"{an.runtime_s:.0f} s") if an.runtime_s is not None else None)], cls="sheet-head"),
        p.field("Why there is no result", p.rec_text(an.message)) if an.status != "done" and an.message else H(""),
        p.field("What was analysed", p.rec_text(an.statement), cls="lead") if an.statement else H(""),
        p.field("The model", join([model, tag("div", p.label("Material properties"), ": ", material, cls="matline") if material else None])) if (an.bcs or material) else H(""),
        p.field("Pictures", p.gallery(pics)) if pics else H(""),
        p.field("Mesh levels", mesh) if an.mesh else H(""),
        p.field("Convergence of the key results", conv) if conv else H(""),
        p.field("Results", join([results_table(p, an, key_set + others) if keys else None,
                                 tag("p", p.t("The analysis gives no value for these results of the load case: {}.",
                                              join([p.code(k) for k in missing], ", ")), cls="empty") if missing else None])) if (keys or missing) else H(""),
        p.field("Checks of the analysis", checks) if an.checks else H(""),
        p.field("Decisions and assumptions of this analysis", p.items([p.recline(i) for i in own_recs], cls="reclist")) if own_recs else H(""),
        p.field("Files", p.items([join([p.code(a.path), tag("span", p.rec_text(a.caption), cls="muted") if a.caption else None], " ")
                                  for a in files], cls="plain")) if files else H(""),
        p.links_field(an, skip=[an.load_case]),
        p.about_block(an.id),
        p.origin_field(an),
    ])
    return p.card(an.id, an.title or an.id, badges=badges, summary=summary, body=body, level=level, cls="analysis",
                  data_status=an.status)


# =========================================================================== compliance


def margin_formula(p: Page, item: Any) -> H:
    """The margin of safety of one item, written out with its numbers."""
    if item.margin is None or item.limit is None or item.conservative_value in (None, 0):
        return H("")
    resp = item.conservative_value
    k, ulabel = charts.unit_scale(item.limit, item.unit)
    unit_tex = (r"\,\mathrm{" + ulabel.replace("%", r"\%").replace("µ", r"\mu ").replace("²", "^2").replace("³", "^3").replace("·", r"\cdot ") + "}") if ulabel else ""
    lim, rsp = fmt.plain(item.limit * k, 4), fmt.plain(resp * k, 4)
    f = fmt.plain(item.factor, 4)
    ms = fmt.margin(item.margin)
    if item.comparator == "<=":
        latex = (r"\mathrm{MS} = \frac{\mathrm{limit}}{\mathrm{factor} \times \mathrm{response}} - 1 = "
                 rf"\frac{{{lim}{unit_tex}}}{{{f} \times {rsp}{unit_tex}}} - 1 = {ms}")
    else:
        latex = (r"\mathrm{MS} = \frac{\mathrm{response} \, / \, \mathrm{factor}}{\mathrm{limit}} - 1 = "
                 rf"\frac{{{rsp}{unit_tex} \, / \, {f}}}{{{lim}{unit_tex}}} - 1 = {ms}")
    return p.formula(latex)


def source_numbers(p: Page, rid: str, limit: int = 40) -> tuple[int, H]:
    """The named numbers a record rests on that have no expression: the numbers of the source documents.
    They are the leaves of dossier.trace.upstream."""
    from dossier import trace

    seen: dict[str, None] = {}

    def walk(node: dict) -> None:
        n = p.d.node(node["id"])
        if n is not None and n["kind"] == "parameter":
            prm = p.d.ctx.params.get(node["id"])
            if prm is not None and not prm.depends_on:
                seen.setdefault(node["id"], None)
        for c in node["children"]:
            walk(c)

    walk(trace.upstream(p.d.graph, rid, depth=10))
    if not seen:
        return 0, H("")
    rows = []
    for name in list(seen)[:limit]:
        prm = p.d.ctx.params[name]
        rows.append([p.ref(name), prm.title, p.qty(prm.si, prm.unit), p.chip(prm.basis, "basis"),
                     p.doc(prm.source) if prm.source else tag("span", p.label("no source document"), cls="muted")])
    more = tag("p", p.t("The table shows {} of {} numbers.", limit, len(seen)), cls="muted") if len(seen) > limit else None
    return len(seen), join([p.table(["Number", "What it is", "Value", "Basis", "Source document"], rows, cls="t-src"), more])


def compliance_card(p: Page, item: Any, level: int = 5) -> H:
    d = p.d
    req = _req(d, item.requirement)
    an = d.records.get(item.analysis) if item.analysis else None
    target = f"{an.id}:{item.result_key}" if an is not None and item.result_key in getattr(an, "results", {}) else None
    badges: list[Any] = [p.badge("verdict", item.verdict)]
    if item.margin is not None:
        badges.append(tag("span", p.label("margin"), " ", p.num(fmt.margin(item.margin)), cls=f"chip mg v-{item.verdict}"))
    badges.append(p.ref(item.load_case))
    comp_word = "must not be more than the limit" if item.comparator == "<=" else "must not be less than the limit"
    rows = []
    resp_src = join([p.key(item.result_key), (H(" ") + tag("a", p.lt("result of {}", an.id), href=f"#{target}", cls="sub")) if target
                     else ((H(" ") + p.ref(item.analysis)) if item.analysis else None)], "")
    if item.value is not None:
        shown = p.qty(item.value, item.unit, item.uncertainty)
        if item.uncertainty is not None:
            shown = join([shown, tag("span", H("± ") + esc(fmt.percent(item.uncertainty)), cls="unc")], " ")
        rows.append([p.label("Response"), shown, resp_src])
        if item.conservative_value is not None and item.uncertainty:
            rows.append([p.label("Response with its uncertainty"), p.qty(item.conservative_value, item.unit, item.uncertainty),
                         p.t("The uncertainty is on the unsafe side of the value.")])
    else:
        rows.append([p.label("Response"), tag("span", p.label("no value"), cls="muted"), resp_src])
    rows.append([p.label("Factor of safety"), p.num(fmt.plain(item.factor)), p.rec_text(item.factor_source)])
    rows.append([p.label("Limit"), p.qty(item.limit, item.unit) if item.limit is not None else tag("span", p.label("no value"), cls="muted"),
                 p.rec_text(item.limit_source)])
    rows.append(([tag("b", p.label("Margin of safety")), tag("b", p.num(fmt.margin(item.margin))) if item.margin is not None
                  else tag("span", p.label("no value"), cls="muted"),
                  p.t("The response with the factor {}. A margin of zero or more is a pass.", comp_word)], {"cls": "total"}))
    numbers = p.table(["Quantity", "Value", "Where it comes from"], rows, cls="t-margin")
    crit = H("")
    if req is not None and 0 <= item.criterion < len(req.criteria):
        crit = join([tag("a", criterion_label(req, item.criterion), href=f"#{req.id}:c{item.criterion}", cls="sub"), H(": "),
                     criterion_sentence(p, req.criteria[item.criterion])], "")
    flow = p.kv([("Requirement", p.recline(item.requirement)), ("Criterion", crit), ("Load case", p.recline(item.load_case)),
                 ("Analysis", p.recline(item.analysis) if item.analysis else p.t("No analysis gives this value."))], cls="flow")
    n_src, srcs = source_numbers(p, item.id)
    body = join([
        p.field("Verdict", join([p.badge("verdict", item.verdict), p.rec_text(item.statement)], " "), cls="lead"),
        p.field("The numbers of the margin", join([numbers, tag("div", margin_formula(p, item), cls="msf")])),
        p.field("Reasons", p.items([p.rec_text(r) for r in item.reasons])) if item.reasons else H(""),
        p.field("Caveats", join([p.t("These findings question the requirement. The verdict is correct only if the requirement is correct."),
                                 p.items([p.recline(c) for c in item.caveats], cls="reclist")])) if item.caveats else H(""),
        p.field("Where it comes from", flow),
        p.field("Source numbers", p.fold(p.t("This item rests on {} numbers that have no expression. Each number has its source document.",
                                             n_src), srcs)) if n_src else H(""),
        p.links_field(item, skip=[item.requirement, item.load_case, item.analysis or "", *item.caveats]),
        p.about_block(item.id),
        p.origin_field(item),
    ])
    return p.card(item.id, item.title or item.id, badges=badges, summary=p.rec_text(item.statement), body=body, level=level,
                  cls=f"cmpl v-{item.verdict}", data_verdict=item.verdict)
