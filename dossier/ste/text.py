"""Text segmentation for the STE checker: paragraphs, sentences, tokens and the STE word count.

The word count follows Section 8 of ASD-STE100 Issue 9, "Punctuation and word count", as far as public
sources confirm it (the numbers of the rules are in `dossier.ste.rules`):

* A number is one word. A number with its unit ("10.72 MPa", "2 %", "50 W/(m^2 K)") is one word.
* An identifier ("REQ-MOV-001", "LC-FIRE", "von_mises.peak@part"), an abbreviation ("MEOP") and a
  unit are one word each.
* Text in quotation marks is one word. So is text in backquotes (a name of the software).
* A word with hyphens ("re-entrant", "thermo-structural") is one word.
* Text in parentheses is one word of the sentence that contains it. The text in the parentheses is
  counted again as a unit of its own.
* The colon before a vertical list ends the sentence, and each item of the list is counted by itself.
  The number or the mark of a list item is not a word.

Not in the standard, and a convention of this module: a formula ("MS = allowable / (factor x response) - 1")
is one word, because its operands are symbols and not words of a sentence.

Text in quotation marks is not checked at all: text that an engineer wrote is quoted as it is.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

FILL = ""      # a protected character (quoted text, text in parentheses, code)
QEND = ""      # the end of quoted text that ends with a full stop

WORD, NUMBER, IDENT, ACRONYM, UNIT, VAR = "word", "number", "ident", "acronym", "unit", "var"
QUOTE, CODE, PAREN, FORMULA, LATIN, ANDOR = "quote", "code", "paren", "formula", "latin", "andor"
OP, PUNCT, SYMBOL = "op", "punct", "symbol"

COUNTED = frozenset({WORD, NUMBER, IDENT, ACRONYM, UNIT, VAR, QUOTE, CODE, PAREN, FORMULA, LATIN, ANDOR})
OPERANDS = frozenset({WORD, NUMBER, IDENT, ACRONYM, UNIT, VAR, PAREN, CODE})


@dataclass
class Token:
    kind: str
    text: str
    start: int
    end: int
    lower: str = ""
    inner: tuple[int, int] | None = None     # PAREN: where the text in the parentheses is
    tech: bool = False                       # a part of a project technical name, or a technical verb
    name: str = ""                           # LATIN: the group of the abbreviation
    first: bool = False                      # the first word of the sentence
    proper: bool = False                     # a word with a capital letter inside a sentence (a name)
    known: bool = False                      # a part of a group of words that the dictionary approves

    def __post_init__(self):
        if not self.lower:
            self.lower = self.text.lower().replace("’", "'")


@dataclass
class Sentence:
    text: str                 # as written, with single spaces
    tokens: list[Token]
    start: int
    paragraph: int = 0
    terminated: bool = True   # ends with a full stop, a question mark, an exclamation mark or a list colon
    item: bool = False        # an item of a vertical list
    paren: bool = False       # text in parentheses
    names: list[tuple[int, int]] = field(default_factory=list)   # token ranges of technical names

    @property
    def words(self) -> int:
        return sum(1 for t in self.tokens if t.kind in COUNTED)


# --------------------------------------------------------------------------- units and numbers

_UNIT_SYMBOLS = """
Pa kPa MPa GPa hPa mPa bar mbar psi ksi N kN MN mN Nm Nmm kNm m mm cm dm km µm μm um nm ft g kg mg t lb lbf lbm
s ms µs us ns min h hr K mK °C °F ° deg rad mrad sr Hz kHz MHz GHz rpm W kW MW mW J kJ MJ mJ Wh kWh V mV kV A mA
kA Ω ohm mol L l mL ml % ppm dB
""".split()
UNIT_SYMBOLS = frozenset(_UNIT_SYMBOLS)
# Unit symbols in lower case that are not also English words.
_SAFE_LOWER_UNITS = frozenset({"mm", "cm", "dm", "km", "kg", "mg", "nm", "ms", "rpm", "psi", "ksi", "ppm", "mol",
                               "mbar"})

_U = "(?:" + "|".join(re.escape(u) for u in sorted(_UNIT_SYMBOLS, key=len, reverse=True)) + ")"
_E = r"(?:\^\(?[-+−]?\d+(?:\.\d+)?\)?|[²³])"
_U1 = rf"{_U}{_E}?"
_UG = rf"\({_U1}(?:[ */·⋅.]{_U1})*\)"
_UX = rf"(?:{_U1}|{_UG})(?:[*/·⋅.](?:{_U1}|{_UG}))*"
_NUM = (r"(?:\d{1,3}(?:[   ]\d{3})+(?!\d)|\d+)(?:[.,]\d+)*(?:[eE][-+−]?\d+)?"
        r"(?:\^[-+−]?\d+)?")
_NUM_UNIT = re.compile(rf"(?P<num>[-+−±]?{_NUM})(?P<unit>(?:[   ]?{_UX}|/{_UX})(?![\w^(]))?")
_UNIT_ALONE = re.compile(rf"{_UX}(?![\w^(])")
_RUN = re.compile(r"\w+(?:[-.@:/'’·]\w+)*")
_ANDOR = re.compile(r"and\s?/\s?or\b", re.I)
_OPS = re.compile(r"<=|>=|==|!=|->|=>|[→←≤≥≈≠×÷±=+*/^<>~|&\\]")
_DASHES = "-−–—"
_BLANK_LINE = re.compile(r"\n[ \t]*\n")

STRONG_OPS = frozenset({"=", "<=", ">=", "==", "!=", "≤", "≥", "≈", "≠", "+", "*", "^", "×",
                        "÷", "<", ">", "±", "->", "=>", "→"})


def latin_regex(latin) -> re.Pattern:
    """One pattern for the Latin abbreviations; the name of the group that matches is l<index>."""
    return re.compile("|".join(f"(?P<l{i}>{p})" for i, (p, _n, _a) in enumerate(latin)), re.I)


# --------------------------------------------------------------------------- protected spans


def _paragraph_end(text: str, i: int) -> int:
    m = _BLANK_LINE.search(text, i)
    return m.start() if m else len(text)


def protect(text: str) -> tuple[str, dict[int, tuple[int, str, int, int]]]:
    """Mask quoted text, code and text in parentheses.

    Returns the masked text (the same length as `text`) and the spans: start -> (end, kind, inner start,
    inner end). A span never crosses an empty line."""
    n = len(text)
    masked = list(text)
    spans: dict[int, tuple[int, str, int, int]] = {}
    i = 0
    while i < n:
        ch = text[i]
        end = -1
        kind = ""
        if ch == "`":
            j = text.find("`", i + 1, _paragraph_end(text, i))
            if j > i:
                end, kind = j + 1, CODE
        elif ch == '"':
            j = text.find('"', i + 1, _paragraph_end(text, i))
            if j > i + 1:
                end, kind = j + 1, QUOTE
        elif ch == "“":
            j = text.find("”", i + 1, _paragraph_end(text, i))
            if j > i:
                end, kind = j + 1, QUOTE
        elif ch in "'‘":
            before = text[i - 1] if i else " "
            if (before.isspace() or before in "([{:—–") and i + 1 < n and not text[i + 1].isspace():
                close = "'" if ch == "'" else "’"
                stop = _paragraph_end(text, i)
                j = i + 1
                while True:
                    j = text.find(close, j + 1, stop)
                    if j < 0:
                        break
                    if not text[j - 1].isspace() and (j + 1 >= n or not text[j + 1].isalnum()):
                        end, kind = j + 1, QUOTE
                        break
        elif ch in "([":
            before = text[i - 1] if i else " "
            if before.isspace() or before in "([{:,;\"'“‘" + _DASHES:
                close = ")" if ch == "(" else "]"
                stop = _paragraph_end(text, i)
                depth, j = 0, i
                while j < stop:
                    if text[j] == ch:
                        depth += 1
                    elif text[j] == close:
                        depth -= 1
                        if depth == 0:
                            end, kind = j + 1, PAREN
                            break
                    j += 1
        if end > 0:
            spans[i] = (end, kind, i + 1, end - 1)
            for k in range(i, end):
                masked[k] = FILL
            if kind == QUOTE and text[i + 1:end - 1].rstrip()[-1:] in (".", "!", "?"):
                masked[end - 1] = QEND
            i = end
        else:
            i += 1
    out = "".join(masked)
    # Markdown emphasis marks are not text: **word**, *word*, __word__, _word_
    out = re.sub(r"(?<![\w*])(\*\*|__|\*|_)(?=[^\s*_])([^\n]*?)(?<=[^\s*_])\1(?![\w*])",
                 lambda m: " " * len(m.group(1)) + m.group(2) + " " * len(m.group(1)), out)
    return out, spans


# --------------------------------------------------------------------------- blocks and sentences

_LIST_MARK = re.compile(r"[ \t]*(?:[-*•–—]|\d{1,3}[.)]|[a-zA-Z][.)])[ \t]+")
_HEADING = re.compile(r"[ \t]*#{1,6}[ \t]+")
# An abbreviation with a full stop that never ends a sentence.
_ABBR_NEVER = frozenset({"e.g", "i.e", "cf", "c.f", "vs", "viz", "q.v", "n.b", "dr", "mr", "mrs", "ms", "prof", "st"})
# An abbreviation before a number or a letter that identifies something ("Fig. 3", "rev. B").
_ABBR_DESIGNATOR = frozenset({"fig", "figs", "eq", "eqs", "eqn", "no", "nos", "ref", "refs", "rev", "sec", "vol",
                              "p", "pp", "tab", "ch", "para"})
# An abbreviation that ends a sentence only when a capital letter follows.
_ABBR_UPPER = frozenset({"etc", "approx", "max", "min", "incl", "resp", "ca", "al", "ibid"})
_END = re.compile(rf"([.!?]+|{QEND})([\"'”’)\]]*)(\s+|$)")


def _blocks(masked: str) -> list[tuple[int, int, int, bool]]:
    """(paragraph index, start, end, is a list item) of every block of text."""
    out: list[tuple[int, int, int, bool]] = []
    pos, par = 0, 0
    bounds = [(m.start(), m.end()) for m in re.finditer(r"\n(?:[ \t]*\n)+", masked)] + [(len(masked), len(masked))]
    for stop, nxt in bounds:
        if masked[pos:stop].strip():
            start, item, heading = None, False, False
            line_start = pos
            for line in masked[pos:stop].split("\n"):
                mark, head = _LIST_MARK.match(line), _HEADING.match(line)
                if mark or head or heading:
                    if start is not None:
                        out.append((par, start, line_start, item))
                    m = mark or head
                    start, item, heading = line_start + (m.end() if m else 0), bool(mark), bool(head)
                elif start is None:
                    start, item = line_start, False
                line_start += len(line) + 1
            if start is not None:
                out.append((par, start, stop, item))
            par += 1
        pos = nxt
    return out


def _sentence_spans(masked: str, a: int, b: int) -> list[tuple[int, int, bool]]:
    """(start, end, terminated) of every sentence of the block [a, b)."""
    out = []
    start = a
    for m in _END.finditer(masked, a, b):
        punct = m.group(1)
        after = masked[m.end():m.end() + 1]
        if punct == QEND:
            if after and not (after.isupper() or after in (FILL, QEND, "(", "[")):
                continue
        elif punct == ".":
            w = re.search(r"([A-Za-z]+(?:\.[A-Za-z]+)*)$", masked[a:m.start()])
            word = w.group(1).lower() if w else ""
            if word in _ABBR_NEVER:
                continue
            if word in _ABBR_DESIGNATOR and re.match(r"\d|[A-Z](?![a-z])", masked[m.end():m.end() + 2]):
                continue
            if word in _ABBR_UPPER and after and not after.isupper():
                continue
        elif set(punct) == {"."} and after and not after.isupper():     # an ellipsis inside a sentence
            continue
        end = m.start() + len(punct) + len(m.group(2))
        if masked[start:end].strip():
            out.append((start, end, True))
        start = m.end()
    if masked[start:b].strip():
        out.append((start, b, False))
    return out


# --------------------------------------------------------------------------- tokens


def _classify(run: str) -> str | None:
    """The kind of a run of word characters; None when the run must be split at its slashes."""
    if "'" in run or "’" in run:
        return WORD
    segs = re.split(r"[-.@:/·]", run)
    joiners = set(re.findall(r"[-.@:/·]", run))
    if len(segs) == 1:
        s = run
        if any(c.isdigit() for c in s) or "_" in s:
            return IDENT
        if len(s) == 1:
            return WORD if s.lower() in ("a", "i") else VAR
        if s in UNIT_SYMBOLS and (any(c.isupper() for c in s) or s in _SAFE_LOWER_UNITS):
            return UNIT
        if s.isupper():
            return ACRONYM
        if any(c.isupper() for c in s[1:]):
            return IDENT                     # a name of the software, or a symbol such as pH
        return WORD
    plain = all(s.isalpha() for s in segs)
    if joiners == {"-"}:
        if plain and not all(s.isupper() for s in segs):
            return WORD                      # a word with hyphens
        return IDENT
    if joiners == {"/"} and plain:
        return None
    return IDENT


def _standalone_unit(u: str) -> bool:
    if u in ("N/A", "n/a"):
        return False
    if re.search(r"[/^°%·*⋅²³]", u):
        return True                                               # kg/s, m^2, °C, %
    return len(u) > 1 and _classify(u) == UNIT                    # MPa, mm


def tokens_of(text: str, masked: str, spans: dict, a: int, b: int, latin: re.Pattern) -> list[Token]:
    """The tokens of masked[a:b]. Token texts come from `text`."""
    out: list[Token] = []
    i = a
    while i < b:
        ch = masked[i]
        if ch.isspace():
            i += 1
            continue
        if i in spans:
            end, kind, s0, s1 = spans[i]
            out.append(Token(kind, text[i:end], i, end, inner=(s0, s1)))
            i = end
            continue
        if ch in (FILL, QEND):
            i += 1
            continue
        prev = masked[i - 1] if i > a else " "
        boundary = not (prev.isalnum() or prev == "_")
        if boundary:
            m = _ANDOR.match(masked, i, b)
            if m:
                out.append(Token(ANDOR, text[i:m.end()], i, m.end()))
                i = m.end()
                continue
            m = latin.match(masked, i, b)
            if m and not masked[m.end():m.end() + 1].isalnum():
                name = next(k for k, v in m.groupdict().items() if v is not None)
                out.append(Token(LATIN, text[i:m.end()], i, m.end(), name=name))
                i = m.end()
                continue
        signed = ch in "-+−±" and i + 1 < b and masked[i + 1].isdigit() and \
            (prev.isspace() or prev in "([{=<>*/^:,;" or i == a)
        if (ch.isdigit() and boundary) or signed:
            m = _NUM_UNIT.match(masked, i, b)
            if m:
                tail = masked[m.end():m.end() + 2]
                glued = tail[:1].isalpha() or tail[:1] == "_" or (tail[:1] == "-" and tail[1:].isalpha())
                if not glued:
                    out.append(Token(NUMBER, text[i:m.end()], i, m.end()))
                    i = m.end()
                    continue
        if boundary and (ch.isalpha() or ch in "°%"):
            m = _UNIT_ALONE.match(masked, i, b)
            if m and _standalone_unit(m.group()):
                out.append(Token(UNIT, text[i:m.end()], i, m.end()))
                i = m.end()
                continue
        m = _RUN.match(masked, i, b)
        if m:
            end = m.end()
            if masked[end:end + 1] == "(":                      # name(...): a call, one name
                depth, j = 0, end
                while j < b and masked[j] != "\n":
                    if masked[j] == "(":
                        depth += 1
                    elif masked[j] == ")":
                        depth -= 1
                        if depth == 0:
                            break
                    j += 1
                if j < b and masked[j] == ")":
                    out.append(Token(CODE, text[i:j + 1], i, j + 1))
                    i = j + 1
                    continue
            run = m.group()
            kind = _classify(run)
            if kind is None:
                p = i
                for part in re.split(r"(/)", run):
                    if part == "/":
                        out.append(Token(OP, "/", p, p + 1))
                    elif part:
                        k = _classify(part) or WORD
                        if k == WORD and part == "A":
                            k = VAR
                        out.append(Token(k, text[p:p + len(part)], p, p + len(part)))
                    p += len(part)
            else:
                if kind == WORD and run == "A" and any(t.kind in COUNTED for t in out):
                    kind = VAR                                  # "option A", "N/A": a letter, not the article
                out.append(Token(kind, text[i:end], i, end))
            i = end
            continue
        m = _OPS.match(masked, i, b)
        if m:
            out.append(Token(OP, m.group(), i, m.end()))
            i = m.end()
            continue
        if ch in _DASHES:
            spaced = prev.isspace() and masked[i + 1:i + 2].isspace()
            out.append(Token(OP if spaced or ch == "−" else PUNCT, "—" if ch == "—" else "-", i, i + 1))
            i += 1
            continue
        out.append(Token(PUNCT if ch in ",;:.!?…\"'“”‘’()[]{}" else SYMBOL, ch, i, i + 1))
        i += 1
    return _formulas(out, text)


def _is_op(t: Token) -> bool:
    return t.kind == OP or (t.kind == VAR and t.lower == "x")


def _formulas(tokens: list[Token], text: str) -> list[Token]:
    """Join a chain operand, operator, operand, ... into one FORMULA token.

    A chain is a formula when it has an operator that is not also a punctuation mark (= + * ^ < >), or
    when none of its operands is a word (3 / 4, a - b, 2 x 3)."""
    out: list[Token] = []
    i, n = 0, len(tokens)
    while i < n:
        t = tokens[i]
        if t.kind in OPERANDS and i + 2 < n and _is_op(tokens[i + 1]) and tokens[i + 2].kind in OPERANDS:
            j = i
            while j + 2 < n and _is_op(tokens[j + 1]) and tokens[j + 2].kind in OPERANDS:
                j += 2
            ops = [tokens[k].text for k in range(i + 1, j, 2)]
            operands = [tokens[k] for k in range(i, j + 1, 2)]
            if any(o in STRONG_OPS for o in ops) or all(o.kind != WORD for o in operands):
                out.append(Token(FORMULA, text[t.start:tokens[j].end], t.start, tokens[j].end))
                i = j + 1
                continue
        out.append(t)
        i += 1
    return out


# --------------------------------------------------------------------------- the whole text


def segment(text: str, latin: re.Pattern, offset: int = 0, paren: bool = False) -> list[Sentence]:
    """The sentences of a text, in order. The text of each pair of parentheses follows the sentence that
    contains it, as a unit of its own (`paren`)."""
    masked, spans = protect(text)
    out: list[Sentence] = []
    blocks = _blocks(masked)
    for bi, (par, a, b, item) in enumerate(blocks):
        found = _sentence_spans(masked, a, b)
        for si, (s0, s1, terminated) in enumerate(found):
            toks = tokens_of(text, masked, spans, s0, s1, latin)
            if not any(t.kind in COUNTED for t in toks):
                continue
            lead_in = si == len(found) - 1 and not terminated and masked[s0:s1].rstrip().endswith(":") \
                and bi + 1 < len(blocks) and blocks[bi + 1][3]
            for t in toks:
                if t.kind in COUNTED:
                    t.first = True
                    break
            inner_spans = [t.inner for t in toks if t.kind == PAREN and t.inner]
            for t in toks:
                t.start += offset
                t.end += offset
            out.append(Sentence(" ".join(text[s0:s1].split()), toks, offset + s0, paragraph=par,
                                terminated=terminated or lead_in, item=item, paren=paren))
            for i0, i1 in inner_spans:
                inner = segment(text[i0:i1], latin, offset + i0, paren=True)
                for s in inner:
                    s.paragraph = par
                out.extend(inner)
    return out
