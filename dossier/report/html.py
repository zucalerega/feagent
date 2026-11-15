"""Small HTML builder of the report: safe strings, tags, and the register of the report's own text.

`H` marks a string as HTML that is already safe. Every helper escapes what is not `H`, so a record text or a
name from an input file can never break the page. `Own` collects each sentence, heading and label that the
report itself writes (not text of records, not source text): the STE check of the report runs over them.
"""

from __future__ import annotations

import html as _html
import re
from typing import Any, Iterable


class H(str):
    """HTML that is already safe to put in the page. `+` keeps it safe: the other side is escaped unless it is H."""

    __slots__ = ()

    def __add__(self, other: Any) -> "H":
        return H(str.__add__(self, esc(other)))

    def __radd__(self, other: Any) -> "H":
        return H(str.__add__(esc(other), self))


def esc(x: Any) -> H:
    """`x` as safe HTML: an `H` stays as it is, anything else is escaped."""
    if isinstance(x, H):
        return x
    return H(_html.escape("" if x is None else str(x), quote=True))


def join(parts: Iterable[Any], sep: Any = "") -> H:
    return H(esc(sep).join(esc(p) for p in parts if p is not None and p != ""))


_VOID = {"br", "hr", "img", "input", "meta", "link", "col", "wbr"}


def tag(name: str, *children: Any, **attrs: Any) -> H:
    """<name attrs>children</name>. `cls` -> class; `data_x` -> data-x; `aria_x` -> aria-x; None / False attributes are
    left out, True gives a bare attribute. Children that are not `H` are escaped; None children are skipped."""
    out = [f"<{name}"]
    for k, v in attrs.items():
        if v is None or v is False:
            continue
        k = "class" if k == "cls" else "for" if k == "for_" else k.replace("_", "-")
        out.append(f" {k}" if v is True else f' {k}="{_html.escape(str(v), quote=True)}"')
    out.append(">")
    if name in _VOID:
        return H("".join(out))
    for c in children:
        if c is None or c is False:
            continue
        if isinstance(c, (list, tuple)):
            out.extend(esc(x) for x in c if x is not None and x is not False)
        else:
            out.append(esc(c))
    out.append(f"</{name}>")
    return H("".join(out))


_TAG_RE = re.compile(r"<[^>]*>")
_WS_RE = re.compile(r"\s+")


def text_of(x: Any) -> str:
    """The plain text of an HTML fragment (tags removed, entities decoded, white space folded)."""
    s = str(x)
    if isinstance(x, H):
        s = _html.unescape(_TAG_RE.sub(" ", re.sub(r"<svg\b.*?</svg>", " ", s, flags=re.S)))
    s = _WS_RE.sub(" ", s).strip()
    return re.sub(r"\s+([.,;:)])", r"\1", s)


class Own:
    """The register of the text the report itself writes: (kind, text, section).
    kind: "sentence" (full sentences), "heading", "label" (column heads, field names, button captions)."""

    def __init__(self) -> None:
        self.items: list[tuple[str, str, str]] = []
        self.section = ""
        self._seen: set[tuple[str, str]] = set()

    def add(self, kind: str, text: str) -> None:
        text = text.strip()
        if text and (kind, text) not in self._seen:
            self._seen.add((kind, text))
            self.items.append((kind, text, self.section))

    def t(self, template: str, *parts: Any) -> H:
        """One or more sentences of the report, with `{}` places for values, links and badges.
        The template is the report's own text and must follow the STE rules."""
        segs = template.split("{}")
        if len(segs) - 1 != len(parts):
            raise ValueError(f"template {template!r} has {len(segs) - 1} places, got {len(parts)} parts")
        out, plain = [esc(segs[0])], [segs[0]]
        for part, seg in zip(parts, segs[1:]):
            out += [esc(part), esc(seg)]
            plain += [text_of(part), seg]
        self.add("sentence", "".join(plain))
        return H("".join(out))

    def label(self, text: str) -> H:
        """A short label of the report: a column head, a field name, a button caption."""
        self.add("label", text)
        return esc(text)

    def lt(self, template: str, *parts: Any) -> H:
        """A label with `{}` places for values: a part of a sentence, not a sentence ("ratio from {} to {}")."""
        segs = template.split("{}")
        if len(segs) - 1 != len(parts):
            raise ValueError(f"template {template!r} has {len(segs) - 1} places, got {len(parts)} parts")
        out = [esc(segs[0])]
        for part, seg in zip(parts, segs[1:]):
            out += [esc(part), esc(seg)]
        self.add("label", " ".join(" ".join(segs).split()))
        return H("".join(out))

    def heading(self, text: str) -> H:
        self.add("heading", text)
        return esc(text)
