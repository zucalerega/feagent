"""The user's STE dictionary file, and the forms of a word.

ASD owns the dictionary of ASD-STE100 (Part 2 of the standard). It is free on request from
https://www.asd-ste100.org/ but nobody else may publish it, so this package does not hold it. A project
that has the standard can write its own file and give it to the checker.

The file is a CSV file in UTF-8 with a header line and these columns:

    word,part_of_speech,status,alternative
    make sure,v,approved,
    test,n,approved,
    test,v,not_approved,do a test
    ensure,v,not_approved,make sure

* `word`: one word or a group of words. Capital letters make no difference.
* `part_of_speech`: n, v, adj, adv, prep, conj, pron, art (or empty). The checker uses it only to decide
  which forms of the word it accepts. A program cannot be sure about the part of speech of a word in a
  sentence.
* `status`: approved | not_approved.
* `alternative`: for a word that is not approved, the approved word to use. The checker shows it.
* A line that starts with "#" is a comment.

The checker accepts the regular forms of an approved word: the plural of a noun, the forms "-s" and "-ed"
of a verb, the forms "-er" and "-est" of an adjective. Give an irregular form ("gave", "given") its own line.
The "-ing" form of a verb is never accepted from the dictionary: STE permits it only as a technical name.

Where the file is: `terms["dictionary"]` (a path in terms.yaml; `dossier.ste.project_terms` makes it
absolute), or the environment variable DOSSIER_STE_DICTIONARY.
"""

from __future__ import annotations

import csv
import functools
import os
from dataclasses import dataclass, field
from pathlib import Path

ENV = "DOSSIER_STE_DICTIONARY"

_POS = {"noun": "n", "n": "n", "tn": "n", "verb": "v", "v": "v", "tv": "v", "adjective": "adj", "adj": "adj",
        "adverb": "adv", "adv": "adv", "preposition": "prep", "prep": "prep", "conjunction": "conj", "conj": "conj",
        "pronoun": "pron", "pron": "pron", "article": "art", "art": "art", "": ""}
_STATUS = {"approved": "approved", "yes": "approved", "not_approved": "not_approved", "unapproved": "not_approved",
           "notapproved": "not_approved", "no": "not_approved"}


class DictionaryError(ValueError):
    """The dictionary file is wrong. The message gives the file, the line and the problem."""


@dataclass(frozen=True)
class Entry:
    word: str
    pos: str
    status: str
    alternative: str = ""


def singulars(word: str) -> list[str]:
    """The possible base forms of a word that ends in "-s": the singular of a noun, the base of a verb."""
    out = []
    if word.endswith("ies") and len(word) > 4:
        out.append(word[:-3] + "y")
    if word.endswith("es") and len(word) > 3:
        out.append(word[:-2])
    if word.endswith("s") and not word.endswith("ss") and len(word) > 2:
        out.append(word[:-1])
    return out


def pasts(word: str) -> list[str]:
    """The possible base forms of a word that ends in "-ed"."""
    out = []
    if word.endswith("ied") and len(word) > 4:
        out.append(word[:-3] + "y")
    if word.endswith("ed") and len(word) > 3:
        out += [word[:-2], word[:-1]]
        if len(word) > 5 and word[-3] == word[-4]:
            out.append(word[:-3])                      # stopped -> stop
    return out


def comparatives(word: str) -> list[str]:
    """The possible base forms of a word that ends in "-er" or "-est"."""
    out = []
    for suffix in ("er", "est"):
        if word.endswith(suffix) and len(word) > len(suffix) + 2:
            stem = word[:-len(suffix)]
            out += [stem, stem + "e"]
            if stem.endswith("i"):
                out.append(stem[:-1] + "y")            # easier -> easy
            if len(stem) > 2 and stem[-1] == stem[-2]:
                out.append(stem[:-1])                  # bigger -> big
    return out


def verb_forms(base: str, irregular: tuple[str, ...] = ()) -> set[str]:
    """The forms of a verb that STE permits: the base, "-s", "-ed" (and the irregular forms given).
    The "-ing" form is not one of them."""
    out = {base, *irregular}
    if base.endswith(("s", "x", "z", "ch", "sh", "o")):
        out.add(base + "es")
    elif base.endswith("y") and base[-2:-1] not in "aeiou":
        out.add(base[:-1] + "ies")
    else:
        out.add(base + "s")
    if base.endswith("e"):
        out.add(base + "d")
    elif base.endswith("y") and base[-2:-1] not in "aeiou":
        out.add(base[:-1] + "ied")
    else:
        out.add(base + "ed")
        if len(base) > 2 and base[-1] not in "aeiouwxy" and base[-2] in "aeiou" and base[-3] not in "aeiou":
            out.add(base + base[-1] + "ed")            # stop -> stopped
    return out


def plurals(noun: str) -> set[str]:
    out = {noun}
    if noun.endswith(("s", "x", "z", "ch", "sh")):
        out.add(noun + "es")
    elif noun.endswith("y") and noun[-2:-1] not in "aeiou":
        out.add(noun[:-1] + "ies")
    else:
        out.add(noun + "s")
    return out


@dataclass
class Dictionary:
    path: Path
    approved: dict[str, set[str]] = field(default_factory=dict)            # word -> its parts of speech
    not_approved: dict[str, list[Entry]] = field(default_factory=dict)
    phrases: dict[str, list[tuple[tuple[str, ...], Entry]]] = field(default_factory=dict)   # first word -> groups

    @property
    def n_approved(self) -> int:
        return len(self.approved) + sum(1 for v in self.phrases.values() for _w, e in v if e.status == "approved")

    @property
    def n_not_approved(self) -> int:
        return len(self.not_approved) + sum(1 for v in self.phrases.values() for _w, e in v
                                            if e.status == "not_approved")

    def lookup(self, word: str) -> tuple[str, Entry | None]:
        """("approved" | "not_approved" | "unknown", the entry that decides) for one word in lower case."""
        if word in self.approved:
            return "approved", None
        if word in self.not_approved:
            return "not_approved", self.not_approved[word][0]
        blocked: Entry | None = None
        for bases, want in ((singulars(word), ("n", "v")), (pasts(word), ("v",)), (comparatives(word), ("adj", "adv"))):
            for base in bases:
                pos = self.approved.get(base)
                if pos is not None and ("" in pos or pos & set(want)):
                    return "approved", None
                for e in self.not_approved.get(base, ()):
                    if blocked is None and (not e.pos or e.pos in want):
                        blocked = e
        if blocked is not None:
            return "not_approved", blocked
        return "unknown", None

    def verb_only_noun(self, word: str) -> Entry | None:
        """The entry that says `word` is not approved as a verb, when the word is approved as another
        part of speech (for example: test, n, approved and test, v, not_approved)."""
        if word not in self.approved or "v" in self.approved[word] or "" in self.approved[word]:
            return None
        for e in self.not_approved.get(word, ()):
            if e.pos == "v":
                return e
        return Entry(word, "v", "not_approved")


def read(path: str | Path) -> Dictionary:
    """Read a dictionary file. Raises DictionaryError with the line of each problem."""
    path = Path(path)
    if not path.is_file():
        raise DictionaryError(f"{path}: the STE dictionary file does not exist")
    d = Dictionary(path)
    problems: list[str] = []
    with path.open(newline="", encoding="utf-8-sig") as fh:
        rows = csv.reader(line for line in fh if line.strip() and not line.lstrip().startswith("#"))
        header = [h.strip().lower().replace(" ", "_") for h in next(rows, [])]
        header = ["part_of_speech" if h in ("pos", "part") else h for h in header]
        missing = [c for c in ("word", "status") if c not in header]
        if missing:
            raise DictionaryError(f"{path}: the header line must have the columns word, part_of_speech, status, "
                                  f"alternative (missing: {', '.join(missing)})")
        col = {name: header.index(name) for name in ("word", "part_of_speech", "status", "alternative")
               if name in header}
        for n, row in enumerate(rows, start=2):
            def cell(name: str) -> str:
                i = col.get(name)
                return row[i].strip() if i is not None and i < len(row) else ""
            word = " ".join(cell("word").lower().replace("’", "'").split())
            if not word:
                continue
            pos = _POS.get(cell("part_of_speech").lower().strip("(). "))
            status = _STATUS.get(cell("status").lower().replace(" ", "_").replace("-", "_"))
            if pos is None:
                problems.append(f"row {n} ({word}): part_of_speech {cell('part_of_speech')!r} is not one of "
                                f"n, v, adj, adv, prep, conj, pron, art")
                continue
            if status is None:
                problems.append(f"row {n} ({word}): status {cell('status')!r} is not approved or not_approved")
                continue
            e = Entry(word, pos, status, cell("alternative"))
            if " " in word:
                parts = tuple(word.split())
                d.phrases.setdefault(parts[0], []).append((parts, e))
            elif status == "approved":
                d.approved.setdefault(word, set()).add(pos)
            else:
                d.not_approved.setdefault(word, []).append(e)
    if problems:
        raise DictionaryError(f"{path}: " + "; ".join(problems))
    for groups in d.phrases.values():
        groups.sort(key=lambda g: -len(g[0]))
    return d


@functools.lru_cache(maxsize=8)
def _read_cached(path: str, mtime: float) -> Dictionary:
    return read(path)


def load(terms: dict | None) -> Dictionary | None:
    """The dictionary of a project: `terms["dictionary"]`, else the environment variable; None when
    the project has none."""
    path = (terms or {}).get("dictionary") or os.environ.get(ENV)
    if not path:
        return None
    p = Path(path).expanduser()
    if not p.is_file():
        raise DictionaryError(f"{p}: the STE dictionary file does not exist (from "
                              f"{'terms' if (terms or {}).get('dictionary') else ENV})")
    return _read_cached(str(p.resolve()), p.stat().st_mtime)
