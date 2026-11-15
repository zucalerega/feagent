"""Formulas: LaTeX to inline SVG at build time, with matplotlib's mathtext.

The page must work from a file on a machine without a network, so there is no formula script. mathtext knows a
large part of LaTeX mathematics (fractions, roots, Greek letters, sub- and superscripts, \\left / \\right). When a
formula does not parse, or matplotlib is not installed, `render` returns None and the report shows the LaTeX
source in a code block.

Every formula uses the same glyph outlines, so the outlines go once into one hidden sprite (`Formulas.sprite()`)
and each formula refers to them. The glyph ids start with "mt:"; a record id cannot hold a colon.
"""

from __future__ import annotations

import io
import re

FONT_PT = 12.0          # the size mathtext draws at; the page scales the picture in em
_REPLACE = [(r"\le ", r"\leq "), (r"\ge ", r"\geq "), (r"\ne ", r"\neq "), (r"\lvert", "|"), (r"\rvert", "|"),
            (r"\tfrac", r"\frac"), (r"\displaystyle", ""), (r"\;", r"\,"), (r"\!", ""), (r"\qquad", r"\quad"),
            (r"\mathbf", r"\mathrm"), (r"\boldsymbol", ""), (r"\,\,", r"\,")]


def _clean(latex: str) -> str:
    s = latex.strip()
    for a, b in (("$$", ""), ("\\[", ""), ("\\]", ""), ("\\(", ""), ("\\)", "")):
        s = s.replace(a, b)
    s = s.strip("$").strip()
    for a, b in _REPLACE:
        s = s.replace(a, b)
    s = re.sub(r"\\(le|ge|ne)(?![a-zA-Z])", lambda m: "\\" + m.group(1) + "q", s)
    return " ".join(s.split())


class Formulas:
    """Renders formulas and keeps the shared glyph outlines."""

    def __init__(self) -> None:
        self._glyphs: dict[str, str] = {}
        self._cache: dict[str, str | None] = {}
        self.failed: list[str] = []
        self._ok: bool | None = None

    def available(self) -> bool:
        if self._ok is None:
            try:
                import matplotlib.backends.backend_svg  # noqa: F401
                import matplotlib.figure  # noqa: F401
                self._ok = True
            except Exception:  # noqa: BLE001
                self._ok = False
        return self._ok

    def render(self, latex: str) -> str | None:
        """The formula as an <svg> element (em sized, drawn in the text colour), or None."""
        if latex in self._cache:
            return self._cache[latex]
        out = None
        if self.available() and latex.strip():
            try:
                out = self._svg(_clean(latex))
            except Exception:  # noqa: BLE001  mathtext raises ValueError with a parse report
                out = None
        if out is None and latex.strip() and latex not in self.failed:
            self.failed.append(latex)
        self._cache[latex] = out
        return out

    def _svg(self, latex: str) -> str:
        from matplotlib.backends.backend_svg import FigureCanvasSVG
        from matplotlib.figure import Figure

        fig = Figure(figsize=(0.01, 0.01))
        FigureCanvasSVG(fig)
        fig.text(0, 0, f"${latex}$", fontsize=FONT_PT, color="#010203")
        buf = io.StringIO()
        fig.savefig(buf, format="svg", bbox_inches="tight", pad_inches=0.02, transparent=True)
        raw = buf.getvalue()
        m = re.search(r'<svg[^>]*\bwidth="([\d.]+)pt"[^>]*\bheight="([\d.]+)pt"[^>]*viewBox="([^"]+)"', raw)
        if not m:
            raise ValueError("no size in the SVG of the formula")
        w, h, box = float(m.group(1)), float(m.group(2)), m.group(3)
        g = re.search(r'<g id="text_1">(.*)</g>\s*</g>\s*</svg>', raw, flags=re.S)
        if not g:
            raise ValueError("no text in the SVG of the formula")
        body = re.sub(r"<!--.*?-->", "", g.group(1), flags=re.S)

        def keep(match: re.Match) -> str:
            for gid, rest in re.findall(r'<path id="([^"]+)"([^>]*?)/>', match.group(1), flags=re.S):
                self._glyphs.setdefault(gid, " ".join(rest.split()))
            return ""

        body = re.sub(r"<defs>(.*?)</defs>", keep, body, flags=re.S)
        body = re.sub(r'xlink:href="#', 'href="#mt:', body)
        body = body.replace("fill: #010203", "fill: currentColor")
        body = re.sub(r'\sd="([^"]*)"', lambda mm: ' d="' + " ".join(mm.group(1).split()) + '"', body)
        body = " ".join(body.split())
        return (f'<svg class="mt" xmlns="http://www.w3.org/2000/svg" viewBox="{box}" '
                f'style="width:{w / FONT_PT:.3f}em;height:{h / FONT_PT:.3f}em" aria-hidden="true" focusable="false">'
                f"{body}</svg>")

    def sprite(self) -> str:
        """One hidden <svg> with the glyph outlines every formula of the page refers to."""
        if not self._glyphs:
            return ""
        paths = "".join(f'<path id="mt:{gid}" {rest}/>' for gid, rest in sorted(self._glyphs.items()))
        return ('<svg xmlns="http://www.w3.org/2000/svg" aria-hidden="true" focusable="false" '
                'style="position:absolute;width:0;height:0;overflow:hidden"><defs>' + paths + "</defs></svg>")
