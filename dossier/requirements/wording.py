"""What the text of a requirement says, read by rules (no language model).

    obligations(text)        the number of "shall"
    modals(text)             the other words that a text uses for an obligation ("must", "will", "should")
    vague(text)              the words that give no level a verification can show ("sufficiently", "TBD")
    numbers(text)            the numbers the text gives, each with the word after it (a unit, or not)
    gives(text, ...)         does the text give the value of a parameter? True | False | None (not sure)

The word lists are short on purpose: a word is on a list only when it is vague in almost every requirement.
"allowable", "normal", "effective", "minimum" and "detrimental" are technical words and are not on a list.
`gives` is conservative: it answers False only when the text has numbers that it can read and none of them
is the value in any unit form ("20 micrometres" for 20 um, "1.5 times", "-40 °C" for 233.15 K).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from dossier import units

# --------------------------------------------------------------------------- obligations


def obligations(text: str) -> int:
    """The number of "shall" in the text."""
    return len(re.findall(r"\bshall\b", text or "", flags=re.I))


def modals(text: str) -> list[str]:
    """Words that a text without "shall" uses for an obligation, as written, each once."""
    out: list[str] = []
    for m in re.finditer(r"\b(must|will|should|has to|have to|is to|are to|needs to|need to)\b", text or "", flags=re.I):
        if m.group(1).lower() not in [w.lower() for w in out]:
            out.append(m.group(1))
    return out


# --------------------------------------------------------------------------- words without a measure

# kind -> why a verification cannot use the word (the rules write the sentence)
#   measure      no level that a test or an analysis can show: sufficient, adequate, robust
#   escape       the reader decides the level: as far as possible, if necessary, as required
#   open         a value or a decision that is not final: TBD, TBC
#   list         an open list: etc., and so on
#   ambiguous    two readings: and/or
#   approximate  a number without a tolerance: about 5 mm
VAGUE_KINDS = ("measure", "escape", "open", "list", "ambiguous", "approximate")

_VAGUE: tuple[tuple[str, str], ...] = (
    ("escape", r"\bas\s+(?:far|much|little|low|high|small|large|soon|close|light|long|short|few|many)\s+as\s+"
               r"(?:is\s+)?(?:reasonably\s+)?(?:possible|practicable|practical|achievable)\b"),
    ("escape", r"\bto\s+the\s+(?:maximum\s+|greatest\s+)?extent\s+(?:possible|practicable|practical)\b"),
    ("escape", r"\b(?:where|if|when|wherever|whenever)\s+(?:possible|practicable|practical|necessary|needed|"
               r"appropriate|applicable)\b"),
    ("escape", r"\bas\s+(?:necessary|needed|applicable)\b"),
    ("escape", r"\b(?:as|if|when|where)\s+required\b(?!\s+(?:by|in|per)\b)"),      # "as required by <document>" is exact
    ("measure", r"\b(?:in)?sufficient(?:ly)?\b"),
    ("measure", r"\b(?:in)?adequate(?:ly)?\b"),
    ("measure", r"\b(?:in)?appropriate(?:ly)?\b"),
    ("measure", r"\breasonabl[ey]\b"),
    ("measure", r"\bsignificant(?:ly)?\b"),
    ("measure", r"\brobust(?:ly|ness)?\b"),
    ("measure", r"\beas(?:y|ily)\b"),
    ("measure", r"\buser[- ]friendly\b"),
    ("measure", r"\b(?:minimi[sz]|maximi[sz]|optimi[sz])(?:e|es|ed|ing)\b"),
    ("open", r"\bT\.?B\.?[CDRS]\b"),
    ("list", r"\betc\b\.?"),
    ("list", r"\band\s+so\s+(?:on|forth)\b"),
    ("ambiguous", r"\band\s*/\s*or\b"),
)
_APPROXIMATE = re.compile(r"\b(about|approximately|approx\b\.?|roughly|circa|around|nearly|almost)\s+(?=[-+−]?\d)", re.I)
_TOLERANCE = re.compile(r"±|\+\s*/\s*-|\+-|\bplus\s+or\s+minus\b|\btolerances?\b", re.I)


@dataclass(frozen=True)
class Vague:
    word: str       # as the text writes it
    kind: str       # one of VAGUE_KINDS
    start: int      # where it is in the text


def vague(text: str) -> list[Vague]:
    """The words of the text that give no level a verification can show, in text order, each word once.
    "about" and "approximately" count only before a number, and only when the text gives no tolerance."""
    text = text or ""
    hits: list[tuple[int, int, str, str]] = []
    for kind, pattern in _VAGUE:
        for m in re.finditer(pattern, text, flags=re.I):
            hits.append((m.start(), m.end(), " ".join(m.group(0).split()), kind))
    if not _TOLERANCE.search(text):
        for m in _APPROXIMATE.finditer(text):
            hits.append((m.start(1), m.end(1), m.group(1), "approximate"))
    hits.sort(key=lambda h: (h[0], -(h[1] - h[0])))
    out: list[Vague] = []
    end = -1
    for start, stop, word, kind in hits:
        if start < end:                                  # inside a longer phrase ("where appropriate")
            continue
        end = stop
        if word.lower() not in [v.word.lower() for v in out]:
            out.append(Vague(word, kind, start))
    return out


# --------------------------------------------------------------------------- numbers

_UNIT_WORD = r"[A-Za-zµμ°%Ω][A-Za-z0-9µμ°%Ω/^*·²³]*"
_DIGITS = re.compile(
    r"(?<![A-Za-z0-9_.,])(?<![A-Za-z]-)"                 # not inside a word, a number or an id such as MOV-100
    r"(?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:[.,]\d+)?(?:[eE][-+]?\d+)?)"
    r"(?![.,]?\d)"                                       # not the start of 3.2.1
    rf"(?:[  ]?(?P<word>{_UNIT_WORD}))?")
_NUMBER_WORDS = {"zero": 0.0, "one": 1.0, "once": 1.0, "two": 2.0, "twice": 2.0, "double": 2.0, "three": 3.0,
                 "triple": 3.0, "four": 4.0, "five": 5.0, "six": 6.0, "seven": 7.0, "eight": 8.0, "nine": 9.0,
                 "ten": 10.0, "eleven": 11.0, "twelve": 12.0, "dozen": 12.0, "half": 0.5}
_WORDS = re.compile(r"\b(?P<num>" + "|".join(_NUMBER_WORDS) + rf")\b(?:[  ](?P<word>{_UNIT_WORD}))?", re.I)
# number words that this module does not read: with one of them in the text, `gives` is not sure
_COMPOUND = re.compile(r"\b(?:thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|"
                       r"sixty|seventy|eighty|ninety|hundred|thousand|million|billion|quarter|thirds?|fifths?|tenths?)s?\b",
                       re.I)
_NOT_UNITS = {"a", "an", "as", "at", "in", "are", "to", "is", "it", "h", "d", "c", "e", "k", "b", "u", "t"}


@dataclass(frozen=True)
class TextNumber:
    raw: str                    # as written: "10.5", "five"
    values: tuple[float, ...]   # its readings ("1,500" reads as 1500 and as 1.5)
    word: str                   # the word after it: a unit ("MPa"), or not ("times"), or ""
    decimals: int               # digits after the decimal point, as written

    @property
    def shown(self) -> str:
        """The number as written, with the word after it when that word is a unit."""
        return f"{self.raw} {self.word}" if self.word and _quantity(1.0, self.word) is not None \
            and self.word.lower() not in _NOT_UNITS else self.raw


def _readings(raw: str) -> tuple[tuple[float, ...], int]:
    if re.fullmatch(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?", raw):               # 1,500 or 1,500,000.5
        whole = float(raw.replace(",", ""))
        decimals = len(raw.split(".")[1]) if "." in raw else 0
        if raw.count(",") == 1 and "." not in raw:                         # 1,500 can also be a decimal comma
            return (whole, float(raw.replace(",", "."))), 0
        return (whole,), decimals
    if "e" in raw.lower():
        return (float(raw.replace(",", ".")),), 12                         # 1e3: no rounding to shown digits
    sep = "." if "." in raw else "," if "," in raw else ""
    return (float(raw.replace(",", ".")),), (len(raw.split(sep)[1]) if sep else 0)


def numbers(text: str) -> list[TextNumber]:
    """The numbers of the text: digits ("10.5 MPa", "1,500", "-40 °C") and the words zero to twelve, half,
    twice. Section numbers and ids (3.2.1, MOV-100) are not numbers."""
    text = text or ""
    found: list[tuple[int, int, TextNumber]] = []
    for m in _DIGITS.finditer(text):
        values, decimals = _readings(m.group("num"))
        i = m.start("num")
        negative = i > 0 and text[i - 1] in "-−" and (i == 1 or text[i - 2] in " \t\n( ")
        raw = ("-" if negative else "") + m.group("num")
        word = (m.group("word") or "").rstrip("/^*·")
        found.append((m.start("num"), m.end("num"), TextNumber(raw, tuple(-v if negative else v for v in values),
                                                                word, decimals)))
    for (_, e0, a), (s1, _, b) in list(zip(found, found[1:])):             # 1 500: also read as one number
        if text[e0:s1] in (" ", " ") and not a.word and re.fullmatch(r"\d{1,3}", a.raw) and re.fullmatch(r"\d{3}", b.raw):
            found.append((s1, s1, TextNumber(f"{a.raw} {b.raw}", (float(a.raw + b.raw),), b.word, 0)))
    for m in _WORDS.finditer(text):
        found.append((m.start(), m.end("num"), TextNumber(m.group("num"), (_NUMBER_WORDS[m.group("num").lower()],),
                                                           m.group("word") or "", 0)))
    return [n for _, _, n in sorted(found, key=lambda f: f[0])]


def _quantity(value: float, word: str):
    try:
        return units.parse(f"{value!r} {word}")
    except Exception:  # noqa: BLE001  not a unit: the caller compares the bare number
        return None


def _close(a: float, b: float, half_digit: float) -> bool:
    """a is b, or a is b rounded to the digits the text shows (and within 0.1 %)."""
    return abs(a - b) <= max(1e-9 * max(abs(a), abs(b)), min(half_digit, 1e-3 * abs(b)))


def _same(n: TextNumber, v: float, written: float, si: float, unit: str) -> bool:
    half = 0.5 * 10.0 ** -n.decimals
    if unit and n.word:
        q = _quantity(v, n.word)
        if q is not None and units.compatible(q, unit):                    # the text gives a unit: compare in SI
            q1 = _quantity(v + 1.0, n.word)
            one = abs(units.si(q1) - units.si(q)) if q1 is not None else 0.0
            return _close(float(units.si(q)), si, half * one)
    if _close(v, written, half):                                           # the bare number, as the engineer wrote it
        return True
    if not unit and n.word.lower() in ("%", "percent"):
        return _close(v / 100.0, written, half / 100.0)
    return False


def gives(text: str, written: float, si: float, unit: str) -> bool | None:
    """Does the text give this value? `written` is the number as the engineer wrote it in the parameter
    (10.5 for "10.5 MPa"), `si` its SI value, `unit` its SI unit ("" for a pure number).
    True: a number of the text is the value. False: the text has no such number. None: not sure (the text
    has number words that this module does not read)."""
    for n in numbers(text):
        if any(_same(n, v, written, si, unit) for v in n.values):
            return True
    return None if _COMPOUND.search(text or "") else False
