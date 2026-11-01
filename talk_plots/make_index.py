"""Build talk_plots/index.html: every figure of the three acts with its takeaway title and caption, to
scroll through. Rerun after any act's make_figures.py:  python talk_plots/make_index.py"""
import html
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
ACTS = [
    ("act1_agentic_works", "Act 1: agentic meshing works, and training works",
     "On the benchmark problems our agents beat the gmsh automesher by a large factor, and they get better "
     "the longer they train."),
    ("act2_swarms_scale", "Act 2: swarms train better, and this scales",
     "On the hard 3D CAD parts, more agents trained together reach better meshes: 8 > 4 > 2 in every "
     "comparison, and 8 agents beat one budget-matched agent."),
    ("act3_real_data", "Act 3: the harness learns from real lab data",
     "A 3D-printed part crushed on an Instron: before post-training the simulation is 33 % off the real test; "
     "with the learned test conditions and the swarm post-trained on the test, its most accurate mesh is 2.0 % off."),
]
# sub-pages: (href, name, one-line description, act folder it belongs to)
SUBPAGES = [
    ("equal_budget/index.html", "Equal-budget meshes (Act 1)",
     "the key Act 1 example, Cook's membrane at an equal DOF budget, plus mesh galleries", "act1_agentic_works"),
    ("self_learning/index.html", "Self-learning figures",
     "automesher vs early vs trained agent, solved fields on NAFEMS LE11 and Cook's membrane", "act1_agentic_works"),
    ("act3_real_data/videos.html", "Act 3 videos",
     "before vs after post-training, simulated crush of the printed part", "act3_real_data"),
    ("act3_real_data/break/index.html", "Crack prediction (Act 3)",
     "where and when each simulation predicts the first crack, vs the real part", "act3_real_data"),
]
KEY_ACT1 = ("equal_budget/fig_cook_equal_stress.md",
            "KEY ACT 1 EXAMPLE: same DOF budget, the trained agent is 3.4x more accurate than the automesher")


def inline(s: str) -> str:
    """Markdown inline: escape, then `code`, **bold**, *italic*, [text](link)."""
    s = html.escape(s)
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    s = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", s)
    s = re.sub(r"(?<![*\w])\*([^*]+)\*(?!\w)", r"<em>\1</em>", s)
    return re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', s)


def md_to_html(text: str) -> str:
    """Enough Markdown for the captions: paragraphs, '-' lists (with continuation lines), tables."""
    out, para, items, table = [], [], [], []

    def flush():
        if para:
            out.append(f"<p>{inline(' '.join(para))}</p>"); para.clear()
        if items:
            out.append("<ul>" + "".join(f"<li>{inline(i)}</li>" for i in items) + "</ul>"); items.clear()
        if table:
            rows = [r for r in table if not re.fullmatch(r"\|[\s:|-]+\|", r)]
            cells = [[c.strip() for c in r.strip("|").split("|")] for r in rows]
            head = "".join(f"<th>{inline(c)}</th>" for c in cells[0])
            body = "".join("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in r) + "</tr>" for r in cells[1:])
            out.append(f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"); table.clear()

    for line in text.splitlines():
        s = line.rstrip()
        if not s.strip():
            flush()
        elif s.lstrip().startswith("|"):
            if para or items:
                flush()
            table.append(s.strip())
        elif re.match(r"^\s*[-*] ", s):
            if para or table:
                flush()
            items.append(re.sub(r"^\s*[-*] ", "", s))
        elif items and s.startswith("  "):          # continuation of the last list item
            items[-1] += " " + s.strip()
        else:
            if table:
                flush()
            para.append(s.strip())
    flush()
    return "\n".join(out)


def figure(md_path: Path, title: str | None = None) -> str:
    text = md_path.read_text()
    first, rest = (text.split("\n", 1) + [""])[:2]
    title = title or re.sub(r"^#\s*[^:]*:\s*", "", first).strip().strip('"')
    stem = md_path.stem
    png = md_path.with_suffix(".png")
    img = (f'<a href="{png.parent.name}/{png.name}"><img src="{png.parent.name}/{png.name}" alt="{html.escape(title)}" '
           f'loading="lazy"></a>') if png.exists() else '<div class="missing">(no image yet)</div>'
    return (f'<section class="fig" id="{md_path.parent.name}-{stem}"><div class="figname">{stem}</div>'
            f"<h3>{inline(title)}</h3>{img}<div class=\"caption\">{md_to_html(rest)}</div></section>")


def main():
    nav, body = [], []
    for folder, name, claim in ACTS:
        figs = sorted((HERE / folder).glob("fig*.md"))
        nav.append(f'<a href="#{folder}">{html.escape(name.split(":")[0])}</a>')
        body.append(f'<h2 id="{folder}">{html.escape(name)}</h2><p class="claim">{html.escape(claim)}</p>')
        subs = [s for s in SUBPAGES if s[3] == folder]
        if subs:
            body.append('<ul class="subpages">' + "".join(
                f'<li><a href="{h}">{html.escape(n)}</a>: {html.escape(d)}</li>' for h, n, d, _ in subs) + "</ul>")
        if folder == "act1_agentic_works" and (HERE / KEY_ACT1[0]).exists():
            body.append(figure(HERE / KEY_ACT1[0], KEY_ACT1[1]).replace('class="fig"', 'class="fig key"', 1))
        body += [figure(f) for f in figs]
    pages = "".join(f'<li><a href="{h}">{html.escape(n)}</a>: {html.escape(d)}</li>' for h, n, d, _ in SUBPAGES)
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Talk figures</title>
<style>
:root {{ --bg: #ffffff; --fg: #1d1d1f; --muted: #5f6368; --rule: #e3e3e6; --accent: #c23d6e; --code: #f4f4f6; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg: #161618; --fg: #ececf0; --muted: #a0a0a8; --rule: #2e2e33;
  --accent: #e8789f; --code: #26262b; }} }}
body {{ margin: 0; background: var(--bg); color: var(--fg); font: 16px/1.55 -apple-system, "Segoe UI", Helvetica, Arial, sans-serif; }}
nav {{ position: sticky; top: 0; background: var(--bg); border-bottom: 1px solid var(--rule); padding: 10px 16px; z-index: 1; }}
nav a {{ margin-right: 18px; color: var(--accent); text-decoration: none; font-weight: 600; }}
main {{ max-width: 1100px; margin: 0 auto; padding: 8px 16px 80px; }}
h1 {{ font-size: 28px; margin: 24px 0 4px; }}
h2 {{ font-size: 24px; margin: 56px 0 6px; padding-top: 12px; border-top: 3px solid var(--accent); scroll-margin-top: 56px; }}
.claim {{ color: var(--muted); font-size: 17px; margin: 0 0 12px; }}
.fig {{ margin: 36px 0; padding-bottom: 28px; border-bottom: 1px solid var(--rule); }}
.figname {{ color: var(--muted); font: 12px ui-monospace, Menlo, monospace; }}
h3 {{ font-size: 20px; margin: 2px 0 12px; }}
img {{ width: 100%; height: auto; border: 1px solid var(--rule); border-radius: 6px; background: #fff; }}
.caption {{ font-size: 15px; }}
.caption p {{ margin: 10px 0; }}
.caption ul {{ margin: 6px 0 10px; padding-left: 22px; }}
code {{ background: var(--code); padding: 1px 4px; border-radius: 4px; font-size: 13px; }}
table {{ border-collapse: collapse; margin: 10px 0; font-size: 14px; display: block; overflow-x: auto; }}
th, td {{ border: 1px solid var(--rule); padding: 4px 8px; text-align: left; }}
.subpages {{ margin: 4px 0 8px; padding-left: 22px; }}
.subpages a, .pages a {{ color: var(--accent); font-weight: 600; }}
.key {{ border: 2px solid var(--accent); border-radius: 8px; padding: 12px 16px 20px; }}
.missing {{ padding: 40px; text-align: center; color: var(--muted); border: 1px dashed var(--rule); border-radius: 6px; }}
</style></head><body>
<nav>{''.join(nav)}<a href="#pages">More pages</a></nav><main>
<h1>Self-improving meshing agents: talk figures</h1>
<p class="claim">Every number recomputed from the raw runs. Click a figure for the full-size image. Rebuild: <code>python talk_plots/make_index.py</code></p>
<h2 id="pages">More pages</h2><ul class="pages">{pages}</ul>
{''.join(body)}
</main></body></html>"""
    (HERE / "index.html").write_text(page)
    print(HERE / "index.html")


if __name__ == "__main__":
    main()
