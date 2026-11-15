"""Helpers for the engines that write text. They are few and they do exactly what they say.

    listing(["INT-A", "INT-B", "INT-C"])           -> "INT-A, INT-B and INT-C"
    plural(3, "load case")                         -> "3 load cases"
    plural(1, "criterion", "criteria")             -> "1 criterion"
    plural(2, "margin of safety", count=False)     -> "margins of safety"
    a("MEOP")                                      -> "an MEOP"
    sentence("the requirement gives", "10.5 MPa")  -> "The requirement gives 10.5 MPa."
    paragraph(s1, s2, ...)                         -> the sentences, a new paragraph after each six
    quote('the body must be "stiff"')              -> "the body must be 'stiff'"   (in double quotation marks)
    too_long(text)                                 -> [(sentence, words)] for each sentence over the limit
    fit(long_wording, shorter_wording)             -> the first wording with no sentence over the limit

The word count is the STE word count of `dossier.ste.text`: a number with its unit, an identifier, quoted
text and text in parentheses are one word each. A helper does not make text correct STE by itself:
`dossier.ste.check` examines the result.
"""

from __future__ import annotations

import re
from typing import Iterable

from dossier.ste import text as T
from dossier.ste.checker import _LATIN, KINDS, LIMIT, PARAGRAPH_LIMIT

_IRREGULAR_PLURALS = {
    "criterion": "criteria", "analysis": "analyses", "axis": "axes", "basis": "bases", "hypothesis": "hypotheses",
    "index": "indices", "vertex": "vertices", "matrix": "matrices", "radius": "radii", "datum": "data",
    "phenomenon": "phenomena", "life": "lives", "half": "halves", "foot": "feet", "series": "series",
    "data": "data",
}
# Abbreviations that people say as a word: the article follows the sound of the word, not of the letter.
_SPOKEN = frozenset({"LOX", "CAD", "NASA", "RAM", "ROM", "FEM", "LASER", "RADAR", "NATO"})
_AN_LETTERS = frozenset("AEFHILMNORSX")
_A_WORDS = re.compile(r"(?:uni(?!n)|use|usu|uti|ura|euro|eu[lr]|one\b|once\b|u-|ubi)", re.I)
_AN_WORDS = re.compile(r"(?:hour|honest|honou?r|heir)", re.I)


def listing(items: Iterable, conjunction: str = "and") -> str:
    """"A", "A and B", "A, B and C". An empty list gives an empty string."""
    items = [str(x) for x in items]
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + f" {conjunction} " + items[-1]


def _plural_word(word: str) -> str:
    low = word.lower()
    if low in _IRREGULAR_PLURALS:
        p = _IRREGULAR_PLURALS[low]
        return p.upper() if word.isupper() else p.capitalize() if word[:1].isupper() else p
    if low.endswith(("s", "x", "z", "ch", "sh")):
        return word + "es"
    if low.endswith("y") and low[-2:-1] not in "aeiou":
        return word[:-1] + "ies"
    return word + "s"


def plural(n: float, singular: str, plural: str | None = None, count: bool = True) -> str:
    """The number and the noun in the correct form: "1 load case", "3 load cases", "0 load cases".

    The plural is made by rule ("stress" -> "stresses", "body" -> "bodies", "criterion" -> "criteria",
    "margin of safety" -> "margins of safety"); give `plural` when the rule is wrong for the noun.
    `count=False` gives the noun without the number."""
    one = float(n) == 1.0
    if one:
        noun = singular
    elif plural is not None:
        noun = plural
    else:
        head, sep, tail = singular.partition(" of ")
        words = head.split(" ")
        words[-1] = _plural_word(words[-1])
        noun = " ".join(words) + sep + tail
    if not count:
        return noun
    number = str(int(n)) if float(n).is_integer() else f"{n:g}"
    return f"{number} {noun}"


def a(noun: str) -> str:
    """The noun with "a" or "an": "a load case", "an interface", "an MEOP", "a LOX tank", "an 8 mm hole".

    An abbreviation takes the sound of its first letter ("an MEOP", "an SCL"), unless people say it as a
    word ("a LOX tank", "a CAD file")."""
    first = noun.split()[0] if noun.split() else ""
    core = first.split("-")[0]
    if not core:
        return noun
    if core[0].isdigit():
        an = core.startswith("8") or re.match(r"1[18](?:\D|$)", core) is not None
    elif core.isupper() and len(core) > 1 and core not in _SPOKEN:
        an = core[0] in _AN_LETTERS
    elif len(core) == 1 and core.isupper():
        an = core in _AN_LETTERS
    elif _AN_WORDS.match(core):
        an = True
    elif _A_WORDS.match(first):
        an = False
    else:
        an = core[0].lower() in "aeiou"
    return f"{'an' if an else 'a'} {noun}"


def sentence(*parts) -> str:
    """One sentence from its parts: one space between the parts, a capital letter at the start, a full stop
    at the end. An identifier at the start ("f44", "dp_surge", "von_mises.peak@part") stays as it is."""
    text = " ".join(str(p).strip() for p in parts if p is not None and str(p).strip())
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text).strip()
    while text and text[-1] in ",;:":
        text = text[:-1].rstrip()
    if not text:
        return ""
    m = re.match(r"[a-z][a-z'’-]*(?=[\s,.;:!?]|$)", text)
    if m and (len(m.group()) > 1 or m.group() == "a"):
        text = text[0].upper() + text[1:]
    if not re.search(r"[.!?][\"'”’)\]]*$", text):
        text += "."
    return text


def paragraph(*sentences) -> str:
    """The sentences as text. After each six sentences a new paragraph starts, because an STE paragraph has
    six sentences or fewer. Each part becomes a sentence (`sentence`); one list of sentences is also accepted."""
    if len(sentences) == 1 and isinstance(sentences[0], (list, tuple)):
        sentences = tuple(sentences[0])
    done = [sentence(s) for s in sentences if s is not None and str(s).strip()]
    groups = [" ".join(done[i:i + PARAGRAPH_LIMIT]) for i in range(0, len(done), PARAGRAPH_LIMIT)]
    return "\n\n".join(groups)


def quote(text: str) -> str:
    """Text that an engineer wrote, in double quotation marks. The checker does not examine quoted text and
    counts it as one word. Double quotation marks in the text become single ones."""
    return '"' + " ".join(str(text).split()).replace('"', "'").replace("“", "'").replace("”", "'") + '"'


def too_long(text: str, kind: str = "descriptive") -> list[tuple[str, int]]:
    """The parts of a text that break the word limit: (sentence, number of words) for each sentence, each
    list item and each text in parentheses with more words than the limit (25, or 20 for kind "procedure")."""
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {', '.join(KINDS)}, not {kind!r}")
    return [(s.text, s.words) for s in T.segment(text, _LATIN) if s.words > LIMIT[kind]]


def fit(*candidates: str, kind: str = "descriptive") -> str:
    """The first wording that has no sentence over the word limit; the last wording when every one has.
    Give the wordings from the most complete to the shortest."""
    for c in candidates:
        if not too_long(c, kind):
            return c
    return candidates[-1]
