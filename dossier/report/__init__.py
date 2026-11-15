"""The report: one self-contained page `out/report/index.html` and the same content as `out/report/report.md`.

    build(project, opts) -> {"html": path, "md": path, "ste": {...}, "open_items": n}

The page has eleven sections (sections.py) and is written for the engineer who signs. The files of this package:
    data.py       what the report reads (the project, the stage files, the trace graph, the reviews)
    page.py       the parts of the page: links to records, badges, cards, tables; one `Page` builds one report
    sections.py   the eleven sections; cards.py the card of each kind of record; charts.py the small charts
    formulas.py   LaTeX to inline SVG at build time; diagram.py the system diagram in the page
    stecheck.py   the STE check of the text; markdown.py the plain text version
    assets/       report.css, report.js and page.html (the template), read with Path(__file__)

The page has no external script, font or picture. The dashboard sets `window.DOSSIER_API` and the review controls
work (see dossier/CONTRACTS.md); the page works without it and shows the reviews of the build time.
"""

from __future__ import annotations

import html as _html
import json
from pathlib import Path
from typing import Any

from dossier.report import markdown, sections, stecheck
from dossier.report.data import Data
from dossier.report.page import Page, icon_sprite
from dossier.report.sections import review_rule

ASSETS = Path(__file__).parent / "assets"

NAV = [("summary", "Summary"), ("attention", "Needs attention"), ("system", "System context"), ("component", "Component"),
       ("requirements", "Requirements"), ("load-cases", "Load cases"), ("hand-calcs", "Hand calculations"),
       ("analyses", "Analyses"), ("compliance", "Compliance"), ("decisions", "Decisions and assumptions"),
       ("about", "About this report")]


def _json(x: Any) -> str:
    """JSON that is safe inside a script element."""
    return json.dumps(x, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")


def _render(p: Page, ste: dict | None) -> tuple[list[str], int]:
    static, dynamic = sections.attention_items(p)
    n_att = len(static) + len(dynamic)
    parts = [str(sections.summary(p, n_att)), str(sections.attention(p, static, dynamic)), str(sections.system_context(p)),
             str(sections.component(p)), str(sections.requirements(p)), str(sections.load_cases(p)), str(sections.hand_calcs(p)),
             str(sections.analyses(p)), str(sections.compliance(p)), str(sections.decisions(p)), str(sections.about(p, ste))]
    return parts, n_att


def build(project, opts=None) -> dict:
    out_dir = project.out / "report"
    out_dir.mkdir(parents=True, exist_ok=True)
    d = Data.load(project)
    p = Page(d, out_dir)
    # first pass registers the sentences of the report; the STE check reads them; the second pass writes section 11 with the result
    sids, problems = set(p._sids), list(p.problems)
    emitted, copied = dict(p.emitted), dict(p.copied)
    _render(p, None)
    ste = stecheck.run(d, p.own.items)
    p._sids, p.problems, p.emitted, p.copied = sids, problems, emitted, copied
    p.own.section = ""
    parts, n_att = _render(p, ste)

    nav = "".join(f'<li><a href="#{k}" data-nav="{k}">{_html.escape(t)}</a></li>' for k, t in NAV)
    preview = {i: [n.get("kind", ""), n.get("title", ""), n.get("summary", ""), n.get("status", "")] for i, n in d.graph["nodes"].items()}
    data = {
        "graph": {"nodes": d.graph["nodes"], "edges": d.graph["edges"]},
        "reviews": d.reviews,
        "project": str(project.root),
        "cli": p.cli_project(),
        "rules": {k: [v[0], v[1], v[2]] for k, v in sections.ATT_REVIEW.items()},
        "cards": sorted(p.emitted),
        "params": {i: {"unit": x.unit, "si": x.si, "basis": x.basis, "source": x.source, "expr": x.expr, "title": x.title}
                   for i, x in d.ctx.params.items()} if hasattr(d.ctx, "params") else {},
    }
    from dossier.report import diagram
    template = (ASSETS / "page.html").read_text()
    page = (template.replace("{{TITLE}}", _html.escape(f"{project.file.name} - verification report"))
            .replace("{{CSS}}", (ASSETS / "report.css").read_text() + diagram.theme_css())
            .replace("{{SPRITE}}", icon_sprite() + p.formulas.sprite())
            .replace("{{NAV}}", nav)
            .replace("{{HEADLINE}}", _html.escape(project.file.name))
            .replace("{{BODY}}", "\n".join(parts))
            .replace("{{DATA}}", _json(data))
            .replace("{{JS}}", (ASSETS / "report.js").read_text().replace("</script", "<\\/script")))
    html_path = out_dir / "index.html"
    html_path.write_text(page, encoding="utf-8")
    md_path = out_dir / "report.md"
    md_path.write_text(markdown.convert("\n".join(parts), project.file.name), encoding="utf-8")
    summary = {k: v for k, v in (ste or {}).items() if k != "summary"} if ste else {"ran": False}
    return {"html": str(html_path), "md": str(md_path), "ste": summary, "open_items": n_att,
            "problems": list(p.problems)}
