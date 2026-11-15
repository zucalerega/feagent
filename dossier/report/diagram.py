"""The system diagram in the page: `<out>/system.svg` (written by the validate stage) put inline.

The diagram engine gives each element and each connection one SVG group whose `id` is the record id. The page
already uses each record id for the card of the record, so every id of the SVG gets the prefix "dg:" here (a
record id cannot hold a colon) and each record group gets `data-ref`, `tabindex` and `role="link"`: the page
script opens the record on a click or on Enter, and shows the same preview as for a link.
"""

from __future__ import annotations

import html
import re
from typing import Callable

PREFIX = "dg:"


def inline(svg: str, is_record: Callable[[str], bool], title_of: Callable[[str], str]) -> str:
    """The SVG document as an inline <svg> element, safe to put in the page."""
    s = re.sub(r"<\?xml.*?\?>", "", svg, flags=re.S)
    s = re.sub(r"<!DOCTYPE.*?>", "", s, flags=re.S)
    s = re.sub(r"<script\b.*?</script>", "", s, flags=re.S | re.I)
    s = re.sub(r"\son[a-z]+\s*=\s*(\"[^\"]*\"|'[^']*')", "", s, flags=re.I)
    s = s.strip()
    ids = set(re.findall(r'\bid="([^"]+)"', s)) | set(re.findall(r"\bid='([^']+)'", s))

    def new_id(m: re.Match) -> str:
        i = m.group(2)
        if is_record(i):
            t = title_of(i)
            label = html.escape(f"{i}: {t}" if t and t != i else i, quote=True)
            return (f'{m.group(1)}id="{PREFIX}{i}" data-ref="{i}" tabindex="0" role="link" aria-label="{label}"')
        return f'{m.group(1)}id="{PREFIX}{i}"'

    s = re.sub(r"(\s)id=[\"']([^\"']+)[\"']", new_id, s)
    for i in sorted(ids, key=len, reverse=True):
        s = s.replace(f"url(#{i})", f"url(#{PREFIX}{i})")
        s = re.sub(rf'href=(["\'])#{re.escape(i)}\1', rf'href=\1#{PREFIX}{i}\1', s)

    def labelled(m: re.Match) -> str:
        return m.group(1) + " ".join(PREFIX + x if x in ids else x for x in m.group(2).split()) + m.group(3)

    s = re.sub(r'((?:aria-labelledby|aria-describedby)=")([^"]*)(")', labelled, s)
    return s


def theme_css() -> str:
    """The dark colours of the diagram, when the diagram engine tells them (dossier.system.diagram.css_variables)."""
    try:
        from dossier.system import diagram
        dark, light = diagram.css_variables("dark"), diagram.css_variables("light")
    except Exception:  # noqa: BLE001  the engine is missing or older: the diagram keeps its own colours
        return ""
    important = " ".join(d.strip() + " !important;" for d in light.split(";") if d.strip())
    return ("@media (prefers-color-scheme: dark){:root:not([data-theme=\"light\"]) svg.dz-diagram{" + dark + "}}"
            ":root[data-theme=\"dark\"] svg.dz-diagram{" + dark + "}"
            "@media print{svg.dz-diagram{" + important + "}}")
