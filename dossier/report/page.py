"""The parts every section of the report is made of: links to records, status badges, cards, tables, fields.

One `Page` builds one report. It knows which record ids have a card (so that every id on the page becomes a link),
keeps the register of the report's own text for the STE check, renders formulas and copies pictures.

Words of the page
    card      one record: a line that is always visible (kind, id, title, status) and a body that opens
    ref       a record id as a link to its card
    badge     a status with an icon, a word and a colour (never colour alone)
    chip      a neutral descriptor (a stage, a basis, an analysis type)
    field     a labelled part of a card ("Reason", "Alternatives")
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any, Iterable

from dossier.report import fmt
from dossier.report.data import Data
from dossier.report.formulas import Formulas
from dossier.report.html import H, Own, esc, join, tag, text_of

# --------------------------------------------------------------------------- icons (16 x 16, stroke drawings)

ICONS = {
    "check": "M3 8.6l3.1 3.1L13 4.6",
    "cross": "M4.2 4.2l7.6 7.6M11.8 4.2l-7.6 7.6",
    "dash": "M4 8h8",
    "question": "M5.9 6.1a2.1 2.1 0 1 1 3.3 1.7c-.8.6-1.2 1-1.2 1.9M8 12.2v.2",
    "warn": "M8 2.4l6 10.4H2zM8 6.4v3.1M8 11.2v.2",
    "error": "M8 1.9a6.1 6.1 0 1 0 0 12.2a6.1 6.1 0 0 0 0-12.2zM8 4.9v3.9M8 10.9v.2",
    "info": "M8 1.9a6.1 6.1 0 1 0 0 12.2a6.1 6.1 0 0 0 0-12.2zM8 7.2v3.9M8 4.9v.2",
    "circle": "M8 2.6a5.4 5.4 0 1 0 0 10.8a5.4 5.4 0 0 0 0-10.8z",
    "pencil": "M3 13l.6-2.8L10.7 3.1l2.2 2.2-7.1 7.1zM9.5 4.3l2.2 2.2",
    "refresh": "M13 8a5 5 0 1 1-1.5-3.6M13 2.6v2.6h-2.6",
    "trend": "M2.5 12.5l4-4.2 2.4 2.4L13 5M13 5h-3.2M13 5v3.2",
    "approx": "M3 6.4c1.6-1.5 3.2-1.5 5 0s3.4 1.5 5 0M3 10.4c1.6-1.5 3.2-1.5 5 0s3.4 1.5 5 0",
    "zigzag": "M2 9.5l3-4.5 3 6.5 3-5.5 3 3",
    "flag": "M4 14V2.5M4 3h7.5l-1.8 2.7L11.5 8.5H4",
    "why": "M3.5 2.5v6.5a3 3 0 0 0 3 3h5.5M3.5 6h5M9.6 9.6l2.6 2.4-2.6 2.4",
    "chevron": "M6 3.5l4.5 4.5L6 12.5",
    "sun": "M8 5.2a2.8 2.8 0 1 0 0 5.6a2.8 2.8 0 0 0 0-5.6zM8 1.5v1.6M8 12.9v1.6M1.5 8h1.6M12.9 8h1.6M3.4 3.4l1.1 1.1M11.5 11.5l1.1 1.1M3.4 12.6l1.1-1.1M11.5 4.5l1.1-1.1",
    "moon": "M13 9.4A5.6 5.6 0 0 1 6.6 3a5.6 5.6 0 1 0 6.4 6.4z",
    "menu": "M2.5 4.5h11M2.5 8h11M2.5 11.5h11",
    "search": "M7 2.5a4.5 4.5 0 1 0 0 9a4.5 4.5 0 0 0 0-9zM10.3 10.3l3.2 3.2",
    "expand": "M4 6l4-3.5L12 6M4 10l4 3.5L12 10",
    "copy": "M5.5 5.5v-3h8v8h-3M2.5 5.5h8v8h-8z",
    "doc": "M4 1.8h5.5L12.5 5v9.2H4zM9.3 1.8V5h3.2M6 8h4.5M6 10.5h4.5",
    "agent": "M8 2.2l1.5 3.3 3.6.4-2.7 2.4.8 3.5L8 10l-3.2 1.8.8-3.5L2.9 5.9l3.6-.4z",
    "person": "M8 2.3a2.6 2.6 0 1 0 0 5.2a2.6 2.6 0 0 0 0-5.2zM2.8 13.7c.5-2.7 2.5-4.2 5.2-4.2s4.7 1.5 5.2 4.2",
}


def icon_sprite() -> str:
    syms = "".join(f'<symbol id="i:{n}" viewBox="0 0 16 16"><path d="{d}" fill="none" stroke="currentColor" '
                   f'stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"/></symbol>' for n, d in ICONS.items())
    return ('<svg xmlns="http://www.w3.org/2000/svg" aria-hidden="true" focusable="false" '
            'style="position:absolute;width:0;height:0;overflow:hidden">' + syms + "</svg>")


def icon(name: str, cls: str = "") -> H:
    return H(f'<svg class="i{" " + cls if cls else ""}" aria-hidden="true" focusable="false"><use href="#i:{name}"/></svg>')


# --------------------------------------------------------------------------- status words
# family -> value -> (tone, icon, word). Tones: pass, fail, warn, serious, info, none, stale.

BADGES: dict[str, dict[Any, tuple[str, str, str]]] = {
    "verdict": {"pass": ("pass", "check", "pass"), "fail": ("fail", "cross", "fail"),
                "inconclusive": ("warn", "question", "inconclusive"), "not_assessed": ("none", "dash", "not assessed")},
    "severity": {"error": ("fail", "error", "error"), "major": ("serious", "warn", "major"),
                 "minor": ("warn", "info", "minor"), "info": ("info", "info", "information")},
    "review": {"open": ("none", "circle", "no review"), "accepted": ("pass", "check", "accepted"),
               "rejected": ("fail", "cross", "rejected"), "changes_requested": ("warn", "pencil", "changes requested"),
               "stale": ("stale", "refresh", "changed after its review")},
    "result": {"converged": ("pass", "check", "converged"), "not_converged": ("warn", "zigzag", "not converged"),
               "singular": ("fail", "trend", "singular"), "estimate": ("info", "approx", "estimate"),
               "not_run": ("none", "dash", "not run")},
    "check": {True: ("pass", "check", "pass"), False: ("fail", "cross", "fail"), None: ("none", "dash", "not evaluated")},
    "agree": {True: ("pass", "check", "agree"), False: ("fail", "cross", "disagree"), None: ("none", "dash", "no analysis value")},
    "analysis": {"done": ("pass", "check", "done"), "failed": ("fail", "cross", "failed"),
                 "skipped": ("none", "dash", "skipped"), "not_supported": ("warn", "dash", "not supported")},
    "load_case": {"selected": ("info", "check", "solved"), "scaled": ("info", "approx", "scaled"),
                  "not_supported": ("warn", "dash", "not supported")},
    "conservative": {True: ("pass", "check", "conservative"), False: ("serious", "warn", "not conservative"),
                     None: ("none", "question", "effect not known")},
    "flag": {"needs_review": ("serious", "flag", "needs review"), "agent": ("info", "agent", "proposal of an agent"),
             "stale_file": ("stale", "refresh", "stale"), "current": ("pass", "check", "current"),
             "not_run": ("none", "dash", "not run"), "source": ("none", "doc", "source text")},
}

KIND_WORD = {
    "system": "System", "element": "Element", "connection": "Connection", "phase": "Phase", "duty_cycle": "Duty cycle",
    "effect": "Effect", "parameter": "Named number", "component": "Component", "feature": "Feature",
    "interface": "Interface", "scl": "Stress classification line", "material": "Material", "policy": "Policy",
    "requirement": "Requirement", "finding": "Finding", "load_case": "Load case", "hand_calc_spec": "Hand calculation model",
    "hand_calc": "Hand calculation", "analysis": "Analysis", "compliance": "Compliance item", "decision": "Decision",
    "assumption": "Assumption", "document": "Document", "missing": "Missing record",
}
REL_WORD = {"derives_from": "comes from", "verifies": "verifies", "uses": "uses", "about": "is about",
            "evidence": "has the evidence", "covers": "covers", "source": "has the source", "part_of": "is part of",
            "connects": "connects", "realizes": "realizes", "has": "has", "states": "states"}
STAGE_WORD = {"context": "Context", "geometry": "Geometry", "validate": "Validation", "plan": "Plan",
              "idealize": "Idealization", "mesh": "Mesh", "solve": "Solve", "assess": "Assessment", "report": "Report",
              "analyze": "Analysis"}

# The sentences of a review. The page script uses the same sentences when a review changes in the browser.
REVIEW_TEXT = {
    "accepted": "{} accepted this record on {}.",
    "rejected": "{} rejected this record on {}.",
    "changes_requested": "{} requested changes to this record on {}.",
    "open": "{} opened the review again on {}.",
    "none": "No person reviewed this record.",
    "stale": "The record changed after this review. Review it again.",
    "cli": "To give a review, use this command: {}",
}

_ID_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9_.-]*[A-Za-z0-9]|[A-Za-z]")
_NAME_TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


class Page:
    def __init__(self, data: Data, out_dir: Path):
        self.d = data
        self.out_dir = out_dir
        self.assets = out_dir / "assets"
        self.own = Own()
        self.formulas = Formulas()
        self.known = data.anchors()
        self.emitted: dict[str, int] = {}
        self.problems: list[str] = []
        self.copied: dict[str, str] = {}
        self._sids: set[str] = set()
        self.t = self.own.t
        self.label = self.own.label
        self.lt = self.own.lt

    # ------------------------------------------------------------------ ids
    def anchor(self, rid: str) -> str:
        """Claim the card of a record: the element with this id is the target of every link to the record."""
        self.emitted[rid] = self.emitted.get(rid, 0) + 1
        if self.emitted[rid] > 1:
            self.problems.append(f"the record {rid} has two cards")
        return rid

    def sid(self, base: str) -> str:
        """An id for a part of the page that is not a record (a section, a table). It never equals a record id."""
        s = base
        while s in self.known or s in self._sids:
            s += "_"
        self._sids.add(s)
        return s

    # ------------------------------------------------------------------ inline parts
    def ref(self, rid: str | None, text: str | None = None) -> H:
        """A record id as a link to its card. An id that has no card is shown as plain text."""
        if not rid:
            return H("")
        if rid in self.known:
            return tag("a", text or rid, cls="ref", href=f"#{rid}", data_ref=rid)
        if rid.startswith("doc:"):
            return self.doc(rid[4:])
        return tag("code", text or rid, cls="noref", title="This id has no record in the project.")

    def refs(self, ids: Iterable[str], sep: str = ", ") -> H:
        return join([self.ref(i) for i in ids], sep)

    def doc(self, text: str) -> H:
        """The name of a source document."""
        return tag("span", icon("doc"), " ", text, cls="docref") if text else H("")

    def badge(self, family: str, value: Any, text: str | None = None, **attrs: Any) -> H:
        tone, ico, word = BADGES[family].get(value, ("none", "dash", str(value).replace("_", " ")))
        self.own.add("label", text or word)
        return tag("span", icon(ico), tag("span", text or word, cls="bw"), cls=f"badge t-{tone}", data_v=str(value), **attrs)

    def review_badge(self, rid: str) -> H:
        """The review state of a record. The page script updates it when a review changes."""
        state = self.d.review_state(rid)
        return self.badge("review", state, data_review_badge=rid) if state else H("")

    def chip(self, text: Any, cls: str = "", own: bool = True) -> H:
        if own:
            self.own.add("label", str(text))
        return tag("span", text, cls=f"chip {cls}".strip())

    def num(self, shown: str, si: str = "") -> H:
        return tag("span", shown, cls="num", title=f"SI: {si}" if si and si != shown else None)

    def qty(self, value: Any, unit: str = "", uncertainty: float | None = None, sig: int | None = None) -> H:
        """An SI number of a record in the unit people read, with the SI value in the tooltip. With an uncertainty,
        the number has only the digits that the uncertainty supports."""
        if value is None:
            return H("")
        n = sig or fmt.sig_for(fmt.magnitude(value), uncertainty)
        return self.num(fmt.quantity(value, unit, n), fmt.si_text(value, unit))

    def code(self, text: Any) -> H:
        return tag("code", text)

    def key(self, key: str) -> H:
        """A result key in words, with the key itself: von Mises stress, peak, at the part."""
        words, loc = fmt.key_words(key)
        where = {"part": "the part", "scls": "the stress classification lines (the worst line)"}.get(loc)
        place = esc(where) if where else self.ref(loc)
        return tag("span", words, " at ", place, " ", tag("code", key, cls="rkey"), cls="keyw") if loc else \
            tag("span", words, " ", tag("code", key, cls="rkey"), cls="keyw")

    def rec_text(self, text: str | None) -> H:
        """Text of a record, as its engine wrote it. Record ids in the text become links."""
        if not text:
            return H("")
        out, pos = [], 0
        for m in _ID_TOKEN.finditer(text):
            tok = m.group(0)
            if tok in self.known and ("-" in tok or "_" in tok):
                out.append(esc(text[pos:m.start()]))
                out.append(self.ref(tok))
                pos = m.end()
        out.append(esc(text[pos:]))
        return tag("span", H("".join(out)), cls="rt")

    def expr(self, expr: str | None) -> H:
        """An expression of named numbers. Each name is a link to its number."""
        if not expr:
            return H("")
        out, pos = [], 0
        for m in _NAME_TOKEN.finditer(expr):
            tok = m.group(0)
            if tok in self.known and self.d.kind(tok) == "parameter":
                out.append(esc(expr[pos:m.start()]))
                out.append(self.ref(tok))
                pos = m.end()
        out.append(esc(expr[pos:]))
        return tag("code", H("".join(out)), cls="expr")

    def source(self, text: str | None, cite: str = "", block: bool = True) -> H:
        """Text that a person wrote in an input file, quoted as written and marked as source text."""
        if not text or not text.strip():
            return H("")
        mark = tag("span", self.label("Source text"), cls="src-mark")
        body = tag("span", " ".join(text.split()), cls="src-text")
        ref = tag("span", cite, cls="src-cite") if cite else None
        if block:
            return tag("blockquote", mark, body, ref, cls="src")
        return tag("q", body, cls="src-inline", title="Source text, as written in the input file")

    def origin(self, by: str, stage: str = "") -> H:
        """Who or what made a record."""
        kind, _, name = by.partition(":")
        if kind == "rule":
            out = tag("span", icon("doc"), " ", self.label("rule"), " ", tag("code", name), cls="origin o-rule")
        elif kind == "engine":
            out = tag("span", icon("doc"), " ", self.label("engine"), " ", tag("code", name), cls="origin o-engine")
        elif kind == "agent":
            out = tag("span", icon("agent"), " ", self.label("agent"), " ", tag("code", name), cls="origin o-agent")
        elif kind == "human":
            out = tag("span", icon("person"), " ", self.label("person"), " ", name, cls="origin o-person")
        else:
            out = tag("span", icon("doc"), " ", self.label("input files"), cls="origin o-input")
        return out

    @staticmethod
    def origin_kind(by: str) -> str:
        k = by.partition(":")[0]
        return {"rule": "rule", "engine": "engine", "agent": "agent", "human": "person"}.get(k, "input")

    # ------------------------------------------------------------------ blocks
    def h(self, level: int, text: str, sid: str | None = None, extra: Any = None, cls: str | None = None) -> H:
        return tag(f"h{level}", self.own.heading(text), extra, id=sid, cls=cls)

    def p(self, content: Any, cls: str | None = None) -> H:
        return tag("p", content, cls=cls)

    def say(self, template: str, *parts: Any, cls: str | None = None) -> H:
        """A paragraph of the report's own text."""
        return tag("p", self.t(template, *parts), cls=cls)

    def empty(self, template: str, *parts: Any) -> H:
        """The one sentence that says why a part of the report has no content."""
        return tag("p", icon("info"), " ", self.t(template, *parts), cls="empty")

    def table(self, head: list[Any], rows: list[Any], cls: str = "", caption: Any = None, sid: str | None = None,
              foot: Any = None) -> H:
        """A table in its own scroll box. `head`: labels of the report (text) or ready cells (H). A row is a list
        of cells, or (cells, {attributes of the row}). A cell is content, or (content, {attributes of the cell})."""
        ths = []
        for hcell in head:
            attrs = {}
            if isinstance(hcell, tuple):
                hcell, attrs = hcell
            ths.append(tag("th", hcell if isinstance(hcell, H) else self.label(hcell), scope="col", **attrs))
        trs = []
        for row in rows:
            rattrs = {}
            if isinstance(row, tuple):
                row, rattrs = row
            tds = []
            for c in row:
                cattrs = {}
                if isinstance(c, tuple):
                    c, cattrs = c
                tds.append(tag("td", c, **cattrs))
            trs.append(tag("tr", tds, **rattrs))
        cap = tag("caption", caption) if caption else None
        return tag("div", tag("table", cap, tag("thead", tag("tr", ths)), tag("tbody", trs), foot, cls=cls or None),
                   cls="tw", id=sid, tabindex="0", role="group", aria_label=text_of(caption) if caption else None)

    def kv(self, pairs: list[tuple[str, Any]], cls: str = "") -> H:
        """Labelled values in a grid. Empty values are left out."""
        items = [tag("div", tag("dt", self.label(k)), tag("dd", v), cls="kv-i") for k, v in pairs if v not in (None, "", H(""))]
        return tag("dl", items, cls=f"kv {cls}".strip()) if items else H("")

    def field(self, label: str, content: Any, cls: str = "") -> H:
        """A labelled part of a card."""
        if content in (None, "", H("")):
            return H("")
        return tag("div", tag("div", self.label(label), cls="fld-l"), tag("div", content, cls="fld-v"),
                   cls=f"fld {cls}".strip())

    def items(self, items: Iterable[Any], cls: str = "", ordered: bool = False) -> H:
        lis = [tag("li", x) for x in items if x not in (None, "", H(""))]
        return tag("ol" if ordered else "ul", lis, cls=cls or None) if lis else H("")

    def fold(self, summary: Any, body: Any, open_: bool = False, cls: str = "", sid: str | None = None, **attrs: Any) -> H:
        return tag("details", tag("summary", summary), tag("div", body, cls="fold-b"), cls=f"fold {cls}".strip(),
                   open=open_, id=sid, **attrs)

    def why_button(self, rid: str, small: bool = False) -> H:
        text = self.label("Why")
        if small:
            return tag("button", icon("why"), tag("span", text, cls="sr"), type="button", cls="why sm", data_why=rid,
                       title=f"Why: what {rid} rests on, and what it affects", data_md="skip")
        return tag("button", icon("why"), " ", text, type="button", cls="why", data_why=rid,
                   title=f"What {rid} rests on, and what it affects", data_md="skip")

    def card(self, rid: str, title: Any, *, badges: Iterable[Any] = (), summary: Any = None, body: Any = None,
             open_: bool = False, level: int = 4, cls: str = "", why: bool = True, kind: str | None = None,
             **data: Any) -> H:
        """The card of one record. The first line is always visible; the body opens on demand (and in print)."""
        kind = kind or self.d.kind(rid)
        self.anchor(rid)
        word = KIND_WORD.get(kind, kind.replace("_", " ").capitalize())
        self.own.add("label", word)
        head = tag(f"h{level}", tag("span", word, cls="kind"), " ", tag("a", rid, cls="self", href=f"#{rid}"), " ",
                   tag("span", title, cls="ttl"), cls="card-h")
        bd = tag("span", join(badges, " "), cls="card-badges", data_md="block")
        btn = self.why_button(rid) if why else None
        summ = tag("span", summary, cls="card-sum", data_md="block") if summary not in (None, "", H("")) else None
        return tag("details", tag("summary", head, bd, btn, summ), tag("div", body, cls="card-b"),
                   cls=f"card k-{kind} {cls}".strip(), id=rid, open=open_, data_kind=kind, **data)

    def row_id(self, rid: str, why: bool = True) -> H:
        """The first cell of a table row that is the card of a record: its id and the Why control."""
        return tag("span", tag("a", rid, cls="self", href=f"#{rid}"), self.why_button(rid, small=True) if why else None,
                   cls="rowid")

    def recline(self, rid: str, with_title: bool = True, text: Any = None) -> H:
        """One line for a record in a list: its link, its title and its status."""
        n = self.d.node(rid) or {}
        bits: list[Any] = [self.ref(rid)]
        if with_title and n.get("title") and n["title"] != rid:
            bits.append(tag("span", n["title"], cls="rl-t"))
        if text:
            bits.append(tag("span", text, cls="rl-x"))
        kind = n.get("kind", "")
        st = n.get("status", "")
        if kind == "finding" and st in BADGES["severity"]:
            bits.append(self.badge("severity", st))
        if kind == "compliance" and st in BADGES["verdict"]:
            bits.append(self.badge("verdict", st))
        if rid in self.d.reviews:
            bits.append(self.review_badge(rid))
        return tag("span", join(bits, " "), cls="recline")

    def about_block(self, rid: str, findings: bool = True) -> H:
        """The decisions and assumptions (and findings) that are about a record."""
        recs = [r for r in self.d.about.get(rid, []) if findings or r.kind != "finding"]
        if not recs:
            return H("")
        return self.field("Records about it", self.items([self.recline(r.id) for r in recs], cls="reclist"), cls="about")

    def about_refs(self, rid: str) -> H:
        """The same, as links only (for a table cell)."""
        return self.refs([r.id for r in self.d.about.get(rid, [])])

    def links_field(self, rec: Any, skip: Iterable[str] = ()) -> H:
        """What a derived record refers to, by relation."""
        skip = set(skip)
        by_rel: dict[str, list] = {}
        for l in rec.refs():
            if l.to in skip or l.to == rec.id:
                continue
            item = join([self.ref(l.to), tag("span", f"({l.note})", cls="muted") if l.note else None], " ")
            if item not in by_rel.setdefault(l.rel, []):
                by_rel[l.rel].append(item)
        if not by_rel:
            return H("")
        parts = [tag("span", tag("span", self.label(fmt_rel(rel)), cls="rel"), " ", join(items, ", "), cls="lk")
                 for rel, items in by_rel.items()]
        return self.field("Links", join(parts, H("; ")), cls="links")

    def origin_field(self, rec: Any) -> H:
        src = self.d.source_of.get(rec.id, "")
        return self.field("Made by", join([self.origin(rec.origin.by), tag("span", self.lt("in the file {}", self.code(src)),
                                                                              cls="muted") if src else None], " "))

    def review_block(self, rid: str) -> H:
        """The review of a record: its state, who gave it and when, the comment, and how to give a review.
        The page script puts the review controls here when the dashboard serves the page, and writes this
        block again when a review changes (with the same sentences: REVIEW_TEXT)."""
        rv = self.d.reviews.get(rid)
        if rv is None:
            return H("")
        state = "stale" if rv["stale"] else rv["state"]
        bits: list[Any] = [self.badge("review", state, data_review_badge=rid)]
        if rv.get("by") or rv.get("date"):
            bits.append(tag("span", self.t(REVIEW_TEXT.get(rv["state"], REVIEW_TEXT["open"]), rv.get("by") or "A person",
                                           rv.get("date") or "a date that is not known"), cls="rv-meta"))
        else:
            bits.append(tag("span", self.t(REVIEW_TEXT["none"]), cls="rv-meta"))
        if rv["stale"]:
            bits.append(tag("span", self.t(REVIEW_TEXT["stale"]), cls="rv-stale"))
        if rv.get("comment"):
            bits.append(tag("q", rv["comment"], cls="rv-comment"))
        cli = tag("span", self.t(REVIEW_TEXT["cli"], tag("code", self.cli_review(rid))), cls="rv-cli")
        return tag("div", tag("div", self.label("Review"), cls="fld-l"),
                   tag("div", tag("span", join(bits, " "), cls="rv-now"), cli, cls="fld-v rv-body"),
                   cls="fld review", data_review_of=rid, data_state=state)

    def cli_review(self, rid: str) -> str:
        return f"python -m dossier review {self.cli_project()} {rid} --accept --by NAME"

    def cli_project(self) -> str:
        root = self.d.project.root
        try:
            return str(root.relative_to(Path.cwd()))
        except ValueError:
            return str(root)

    # ------------------------------------------------------------------ pictures and formulas
    def picture(self, rel_path: str, caption: str = "") -> H:
        """A picture of a stage, copied next to the page (assets/) and shown with its caption."""
        src = self.d.project.out / rel_path
        if not src.is_file():
            return tag("p", icon("info"), " ", self.t("The picture file {} is missing.", self.code(rel_path)), cls="empty")
        name = self.copied.get(rel_path)
        if name is None:
            name = re.sub(r"[^A-Za-z0-9_.-]+", "__", rel_path.strip("/"))
            self.assets.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, self.assets / name)
            self.copied[rel_path] = name
        img = tag("img", src=f"assets/{name}", alt=caption or rel_path, loading="lazy")
        return tag("figure", tag("a", img, href=f"assets/{name}", cls="pic-a", title="Open the picture"),
                   tag("figcaption", self.rec_text(caption)) if caption else None, cls="pic")

    def gallery(self, pictures: list[tuple[str, str]]) -> H:
        return tag("div", [self.picture(p, c) for p, c in pictures], cls="gallery") if pictures else H("")

    def formula(self, latex: str) -> H:
        """A formula as a picture made at build time, or its LaTeX source when it cannot be drawn."""
        if not latex or not latex.strip():
            return H("")
        svg = self.formulas.render(latex)
        if svg is None:
            return tag("pre", tag("code", latex), cls="latex", title="LaTeX source of the formula", data_latex=latex)
        return tag("span", H(svg), cls="formula", data_latex=latex, role="img", aria_label=latex)


def fmt_rel(rel: str) -> str:
    return REL_WORD.get(rel, rel.replace("_", " "))
