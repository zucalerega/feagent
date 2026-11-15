"""The plain text version of the page: the same content in the same order, as Markdown.

The converter reads the HTML of the sections. Rules for the markup of the page:
    data-md="skip"       the element is not in the Markdown (buttons, filters)
    data-md="block"      the element is its own line
    data-md="TEXT"       the element is replaced by TEXT (for example a diagram)
A table becomes a Markdown table, a card (details) becomes a heading line with its body below it, a link to a record
becomes the id of the record.
"""

from __future__ import annotations

import re
from html.parser import HTMLParser

_SKIP_TAGS = {"svg", "script", "style", "button", "input", "select", "textarea", "img"}
_BLOCK = {"p", "div", "section", "figure", "figcaption", "header", "li", "ul", "ol", "dl", "dt", "dd", "details", "summary",
          "table", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "pre", "blockquote", "caption"}
_VOID = {"br", "hr", "img", "input", "meta", "link", "col", "wbr"}


class _Conv(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.out: list[str] = []
        self.stack: list[dict] = []
        self.skip = 0
        self.lists: list[list] = []
        self.table: list[list[str]] | None = None
        self.row: list[str] | None = None
        self.cell: list[str] | None = None
        self.head_row = False

    # ---- output helpers
    def w(self, s: str) -> None:
        if self.skip:
            return
        if self.cell is not None:
            self.cell.append(s)
        else:
            self.out.append(s)

    def nl(self, n: int = 1) -> None:
        if self.skip or self.cell is not None:
            return
        txt = "".join(self.out[-3:])
        have = len(txt) - len(txt.rstrip("\n"))
        if self.out and have < n:
            self.out.append("\n" * (n - have))

    def handle_starttag(self, tag: str, attrs: list) -> None:
        a = dict(attrs)
        md = a.get("data-md")
        entry = {"tag": tag, "skip": False, "end": ""}
        if tag in _VOID and tag != "br" and tag != "hr":
            return
        if md is not None and not self.skip:
            if md == "skip":
                self.skip += 1
                entry["skip"] = True
            elif md != "block":
                self.w(md)
                self.nl(2)
                self.skip += 1
                entry["skip"] = True
            else:
                self.nl()
        elif tag in _SKIP_TAGS and not self.skip:
            self.skip += 1
            entry["skip"] = True
        if tag not in _VOID:
            self.stack.append(entry)
        if self.skip:
            return
        cls = a.get("class", "")
        if tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            self.nl(2)
            self.w("#" * int(tag[1]) + " ")
            entry["end"] = "\n\n"
        elif tag == "summary":
            self.nl(2)
            self.w("**")
            entry["end"] = "**\n\n"
        elif tag == "details":
            self.nl(2)
        elif tag in ("p", "div", "section", "figure", "header", "dl", "dt", "dd", "pre", "figcaption", "caption", "blockquote"):
            self.nl(1 if tag in ("div", "dt", "dd") else 2)
            if tag == "pre":
                self.w("```\n")
                entry["end"] = "\n```\n\n"
        elif tag in ("ul", "ol"):
            self.nl(1)
            self.lists.append([tag, 0])
        elif tag == "li":
            self.nl(1)
            lst = self.lists[-1] if self.lists else ["ul", 0]
            lst[1] += 1
            self.w("  " * max(0, len(self.lists) - 1) + (f"{lst[1]}. " if lst[0] == "ol" else "- "))
        elif tag == "br":
            self.w(" " if self.cell is not None else "\n")
        elif tag == "hr":
            self.nl(2)
            self.w("---\n\n")
        elif tag in ("b", "strong"):
            self.w("**")
            entry["end"] = "**"
        elif tag in ("i", "em"):
            self.w("*")
            entry["end"] = "*"
        elif tag == "code" and "pre" not in [s["tag"] for s in self.stack[:-1]]:
            self.w("`")
            entry["end"] = "`"
        elif tag == "q":
            self.w("\"")
            entry["end"] = "\""
        elif tag == "table":
            self.nl(2)
            self.table = []
        elif tag == "tr":
            self.row = []
        elif tag in ("td", "th"):
            self.cell = []
            if tag == "th":
                self.head_row = True
        if "t-" in cls and tag == "span" and False:
            pass
        self.stack[-1]["cls"] = cls if self.stack else ""

    def handle_endtag(self, tag: str) -> None:
        if tag in _VOID:
            return
        while self.stack and self.stack[-1]["tag"] != tag:
            self.stack.pop()
        if not self.stack:
            return
        entry = self.stack.pop()
        if entry["skip"]:
            self.skip -= 1
            return
        if self.skip:
            return
        if tag in ("td", "th") and self.cell is not None:
            txt = " ".join("".join(self.cell).split()).replace("|", "\\|")
            self.cell = None
            if self.row is not None:
                self.row.append(txt)
        elif tag == "tr" and self.row is not None:
            if self.table is not None:
                self.table.append(self.row)
                if self.head_row:
                    self.table[-1].append("\0head")
            self.row = None
            self.head_row = False
        elif tag == "table" and self.table is not None:
            self.flush_table()
            self.table = None
        elif tag in ("ul", "ol") and self.lists:
            self.lists.pop()
            self.nl(2 if not self.lists else 1)
        elif tag == "li":
            self.nl(1)
        if entry["end"]:
            self.w(entry["end"])
        elif tag in ("p", "figure", "details", "section", "header", "div", "dl"):
            self.nl(2 if tag != "div" else 1)
        elif tag in ("dt", "dd"):
            self.nl(1)

    def flush_table(self) -> None:
        rows = [r for r in (self.table or []) if r]
        if not rows:
            return
        head = None
        body = []
        for r in rows:
            if r and r[-1] == "\0head":
                if head is None:
                    head = r[:-1]
                continue
            body.append(r)
        n = max([len(head or [])] + [len(r) for r in body])
        if head is None:
            head = [""] * n
        pad = lambda r: r + [""] * (n - len(r))  # noqa: E731
        lines = ["| " + " | ".join(pad(head)) + " |", "|" + "|".join(" --- " for _ in range(n)) + "|"]
        lines += ["| " + " | ".join(pad(r)) + " |" for r in body]
        self.out.append("\n".join(lines) + "\n\n")

    def handle_data(self, data: str) -> None:
        if self.skip:
            return
        in_pre = any(s["tag"] == "pre" for s in self.stack)
        if in_pre:
            self.w(data)
            return
        data = re.sub(r"\s+", " ", data)
        if self.cell is None:
            tail = "".join(self.out[-2:])
            if (not tail or tail.endswith("\n") or tail.endswith("- ")) and data.startswith(" "):
                data = data.lstrip()
        self.w(data)


def convert(fragment: str, title: str = "") -> str:
    c = _Conv()
    c.feed(fragment)
    c.close()
    text = "".join(c.out)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() + "\n"
