"""The STE checker: `check` for one text, `summarize` for the STE appendix of a report.

An `Issue` has the severity "error" when code can decide the rule (a word count, a contraction, a word of
a list) and "warning" when the check is a heuristic (the passive voice, a noun cluster, a sentence
without a verb, two instructions, a word that only public guidance gives as not approved).
`dossier.ste.__init__` says which rule is examined how.
"""

from __future__ import annotations

import dataclasses
import json
import os
import re
from dataclasses import dataclass
from typing import Callable, Iterable

from dossier.ste import dictionary as D
from dossier.ste import text as T
from dossier.ste import words as W
from dossier.ste.rules import NOT_CHECKED, RULES, STANDARD

KINDS = ("descriptive", "procedure")
LIMIT = {"descriptive": 25, "procedure": 20}     # words in a sentence (STE 6.3, STE 5.1)
PARAGRAPH_LIMIT = 6                              # sentences in a paragraph (STE 6.6)

_LATIN = T.latin_regex(W.LATIN)
_FUNCTION = W.DETERMINERS | W.PREPOSITIONS | W.CONJUNCTIONS | W.PRONOUNS | W.MID_ADVERBS
_MODALS = W.MODALS_APPROVED | W.MODALS_NOT_APPROVED
_PREFIXES = frozenset({"non", "re", "pre", "sub", "anti", "self", "semi", "multi", "co", "un", "over", "under",
                       "post", "inter", "intra", "mid", "quasi", "thermo", "hydro", "electro"})
_MATCHABLE = frozenset({T.WORD, T.ACRONYM, T.IDENT, T.NUMBER, T.VAR, T.UNIT})
_CONTRACTION = re.compile(r"^(?:\w+n't|\w+'(?:re|ve|ll|d|m)|(?:it|that|there|here|what|who|where|when|how|let|he|she)'s)$")
_EXPANSIONS = {"can't": "cannot", "won't": "will not", "shan't": "shall not", "it's": "it is", "that's": "that is",
               "there's": "there is", "here's": "here is", "what's": "what is", "who's": "who is",
               "where's": "where is", "let's": "let us", "he's": "he is", "she's": "she is", "i'm": "I am"}


@dataclass(frozen=True)
class Issue:
    """One problem that the checker found in a text."""

    rule: str               # the STE rule ("STE 4.2"), or the section name when the number is not sure
    name: str               # a few words: "contraction"
    severity: str           # "error": code can decide the rule; "warning": the check is a heuristic
    message: str            # one STE sentence that says what is wrong
    sentence: str           # the sentence that has the problem
    text: str               # the words that are the problem
    suggestion: str = ""    # the words to use, when there is a sure replacement

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)


# --------------------------------------------------------------------------- the words of a project


def _strings(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(v) for v in value if isinstance(v, (str, int, float)) and str(v).strip()]


def _keys(name: str) -> tuple[str, ...]:
    """A technical name as the tokens that the checker compares."""
    out: list[str] = []
    for s in T.segment(name, _LATIN):
        if not s.paren:
            out += [_key(t) for t in s.tokens if t.kind in _MATCHABLE]
    return tuple(out)


def _key(t: T.Token) -> str:
    w = t.lower
    return w[:-2] if w.endswith("'s") and len(w) > 2 else w


class Vocabulary:
    """The words that a project gives: technical names, technical verbs, words that are not approved,
    and the dictionary file."""

    def __init__(self, terms: dict | None):
        terms = terms or {}
        self.dictionary = D.load(terms)
        self.names: dict[str, list[tuple[str, ...]]] = {}
        self.name_nouns: set[str] = set()
        for name in _strings(terms.get("technical_names")):
            keys = _keys(name)
            if keys:
                self.names.setdefault(keys[0], []).append(keys)
                self.name_nouns |= {k for k in keys if k.isalpha() and k not in W.ADJECTIVES and k not in _FUNCTION}
        for group in self.names.values():
            group.sort(key=len, reverse=True)
        self.verb_bases = {v.lower().strip() for v in _strings(terms.get("technical_verbs"))}
        self.verbs: set[str] = set()
        for v in self.verb_bases:
            self.verbs |= D.verb_forms(v)
        # words that the project does not approve: {"loading scenario": "load case"}
        self.project: dict[str, list[tuple[tuple[str, ...], str]]] = {}
        na = terms.get("not_approved") or {}
        pairs = na.items() if isinstance(na, dict) else ((w, "") for w in _strings(na))
        for word, alt in pairs:
            keys = _keys(str(word))
            if keys:
                self.project.setdefault(keys[0], []).append((keys, str(alt or "")))
        # the small list of public guidance
        self.builtin: dict[str, tuple[str, str, str, str]] = {}        # form -> (base, pos, alternative, check)
        self.builtin_phrases: dict[str, list[tuple[tuple[str, ...], str]]] = {}
        for word, pos, alt, _source, check_name in W.NOT_APPROVED:
            if " " in word:
                parts = tuple(word.split())
                self.builtin_phrases.setdefault(parts[0], []).append((parts, alt))
            elif pos.startswith("v") and check_name != "modal verb":
                for form in D.verb_forms(word, W.IRREGULAR_FORMS.get(word, ())):
                    self.builtin[form] = (word, pos, alt, check_name)
            else:
                self.builtin[word] = (word, pos, alt, check_name)

    def limits(self) -> list[str]:
        """What the checker examines and what it cannot examine, as STE sentences."""
        out = [
            "The checker counts the words of each sentence and the sentences of each paragraph by the rules of "
            "the standard.",
            "The checker finds each contraction, each semicolon, each Latin abbreviation and each modal verb that "
            "is not an approved word.",
            "The checker finds each perfect tense, each progressive tense and each \"-ing\" form that is not a "
            "technical name.",
            "The checker accepts each technical name and each technical verb of the project.",
            "The checks of the passive voice, of noun clusters, of sentences without a verb and of two "
            "instructions in one sentence are estimates.",
            "These estimates give warnings, and a person must make the decision.",
            "The checker does not know the part of speech of a word or the meaning of a word.",
            "The checker cannot find a sentence with two topics, an article that is not there, or two names for "
            "one thing.",
        ]
        if self.dictionary is None:
            out += [
                "The vocabulary check is partial, because the project has no STE dictionary file.",
                "Without this file, the checker knows only a small list of words that public guidance gives as not "
                "approved.",
            ]
        else:
            out += [
                f"The vocabulary check uses the dictionary file \"{self.dictionary.path.name}\" of the project "
                f"({self.dictionary.n_approved} approved words).",
                "The checker accepts the regular forms of each approved word.",
            ]
        return out


def _fingerprint(terms: dict | None) -> str:
    terms = terms or {}
    path = terms.get("dictionary") or os.environ.get(D.ENV) or ""
    stamp = ""
    if path and os.path.isfile(os.path.expanduser(str(path))):
        stamp = str(os.stat(os.path.expanduser(str(path))).st_mtime)
    return json.dumps(terms, sort_keys=True, default=str) + "|" + str(path) + "|" + stamp


_VOCABULARIES: dict[str, Vocabulary] = {}


def vocabulary(terms: dict | None) -> Vocabulary:
    key = _fingerprint(terms)
    if key not in _VOCABULARIES:
        if len(_VOCABULARIES) > 16:
            _VOCABULARIES.clear()
        _VOCABULARIES[key] = Vocabulary(terms)
    return _VOCABULARIES[key]


# --------------------------------------------------------------------------- one sentence


class _Found:
    """The issues of one text, with their position for the order."""

    def __init__(self, full: str):
        self.full = full
        self.items: list[tuple[int, int, Issue]] = []

    def span(self, a: T.Token, b: T.Token | None = None) -> str:
        return " ".join(self.full[a.start:(b or a).end].split())

    def add(self, key: str, severity: str, message: str, sentence: T.Sentence, text: str, pos: int,
            suggestion: str = "") -> None:
        r = RULES[key]
        self.items.append((pos, len(self.items), Issue(r.rule, r.name, severity, message, sentence.text, text,
                                                       suggestion)))


def _q(text: str) -> str:
    """Quoted text for a message (quoted text is one word and the checker does not examine it)."""
    return '"' + text.replace('"', "'") + '"'


def _mark(s: T.Sentence, voc: Vocabulary) -> None:
    """Mark the technical names, the technical verbs and the names with a capital letter."""
    toks = s.tokens
    n = len(toks)
    after_colon = False
    for i, t in enumerate(toks):
        if t.kind == T.WORD and t.text[:1].isupper() and not t.first and not after_colon and not t.text.isupper():
            t.proper = True
        after_colon = t.kind == T.PUNCT and t.text == ":"
    i = 0
    while i < n:
        t = toks[i]
        if t.kind not in _MATCHABLE:
            i += 1
            continue
        k = _key(t)
        hit = 0
        for name in voc.names.get(k, ()):
            m = len(name)
            if i + m > n or any(toks[i + j].kind not in _MATCHABLE for j in range(1, m)):
                continue
            if all(_key(toks[i + j]) == name[j] for j in range(1, m - 1)) and \
                    (m == 1 or _key(toks[i + m - 1]) == name[-1] or name[-1] in D.singulars(_key(toks[i + m - 1]))):
                hit = m
                break
        if not hit and t.kind in (T.WORD, T.ACRONYM):
            for base in D.singulars(k):
                if any(len(name) == 1 for name in voc.names.get(base, ())):
                    hit = 1
                    break
        if hit:
            for j in range(hit):
                toks[i + j].tech = True
                toks[i + j].proper = False
            s.names.append((i, i + hit))
            i += hit
            continue
        if t.kind == T.WORD and k in voc.verbs:
            t.tech = True
            t.proper = False
        i += 1


def _adverb(t: T.Token) -> bool:
    return t.kind == T.WORD and (t.lower in W.MID_ADVERBS or (t.lower.endswith("ly") and len(t.lower) > 4))


def _participle(w: str) -> bool:
    return w in W.IRREGULAR_PARTICIPLES or (w.endswith("ed") and len(w) > 3 and w not in W.NOT_PARTICIPLES)


def _after(toks: list[T.Token], i: int, most: int = 3) -> int:
    """The index of the first token after i that is not an adverb (at most `most` adverbs)."""
    j = i + 1
    while j < len(toks) and j - i <= most and _adverb(toks[j]):
        j += 1
    return j


def _before(toks: list[T.Token], i: int, most: int = 2) -> int:
    j = i - 1
    while j >= 0 and i - j <= most and _adverb(toks[j]):
        j -= 1
    return j


def _follower(toks: list[T.Token], i: int, also: frozenset = frozenset()) -> bool:
    """True when the token after i starts an object: a determiner, a pronoun, a number or an identifier."""
    if i + 1 >= len(toks):
        return False
    t = toks[i + 1]
    if t.kind in (T.NUMBER, T.IDENT):
        return True
    return t.kind == T.WORD and (t.lower in also or (t.lower in W.DETERMINERS and t.lower != "that")
                                 or t.lower in ("it", "them"))


def _ing(t: T.Token, voc: Vocabulary) -> bool:
    """True when the word is an "-ing" form that is not a technical name."""
    if t.kind != T.WORD or t.tech or t.proper:
        return False
    w = _key(t).rsplit("-", 1)[-1]
    base = w[:-1] if w.endswith("ings") else w
    if not base.endswith("ing") or len(base) < 5 or not re.search("[aeiouy]", base[:-3]):
        return False
    if base in W.ING_NOT_VERB or base in W.ING_TECHNICAL or base in voc.name_nouns:
        return False
    if voc.dictionary is not None and (_key(t) in voc.dictionary.approved or base in voc.dictionary.approved):
        return False
    return True


def _verb_use(toks: list[T.Token], i: int, base: str) -> bool:
    """True when the word at i (a noun that is not approved as a verb) is clearly used as a verb."""
    t = toks[i]
    w = _key(t)
    if w != base and w.endswith("ed"):
        return True
    j = _before(toks, i)
    prev = toks[j].lower if j >= 0 and toks[j].kind == T.WORD else ""
    if prev in _MODALS or prev in W.DO or prev == "to":
        return True
    if w == base and t.first:
        return _follower(toks, i, frozenset({"that", "if", "whether", "for"}))
    if w != base:                                        # "checks": a verb when an object follows
        return _follower(toks, i)
    return prev in ("they", "we", "you", "i") and _follower(toks, i)


def _noun(t: T.Token, voc: Vocabulary) -> tuple[bool, bool]:
    """(the token is a noun for the noun-cluster check, it is a plural)."""
    if t.kind == T.ACRONYM:
        return True, False
    if t.kind != T.WORD or t.proper:
        return False, False
    w = _key(t)
    if "-" in w or w in W.ADJECTIVES or w in _FUNCTION:
        return False, False
    if w in W.NOUNS or w in voc.name_nouns:
        return True, False
    for base in D.singulars(w):
        if base in W.NOUNS or base in voc.name_nouns:
            return True, True
    return False, False


def _has_verb(toks: list[T.Token], voc: Vocabulary) -> bool:
    for i, t in enumerate(toks):
        if t.kind != T.WORD:
            continue
        w = _key(t)
        if re.match(r"^(?:do|does|did|is|are|was|were|has|have|had|can|won|must|should|would|could)n['’]t$", w):
            return True
        if w in W.BE or w in W.HAVE or w in W.DO or w in _MODALS or w in voc.verbs:
            return True
        if w in _VERB_FORMS:
            base = _VERB_FORMS[w]
            if base not in W.VERB_OR_NOUN or (w != base and not w.endswith("s")):
                return True
            if _follower(toks, i, frozenset({"that", "if", "sure"})):
                return True
            j = _before(toks, i)
            if (w != base and w.endswith("s") and j >= 0 and toks[j].kind == T.WORD and toks[j].lower not in
                    (_FUNCTION | W.ADJECTIVES | W.NUMBER_WORDS) and i + 1 < len(toks)
                    and toks[i + 1].kind == T.WORD and toks[i + 1].lower in (W.PREPOSITIONS | W.DETERMINERS | W.MID_ADVERBS)):
                return True      # "the pressure acts on", "the load moves to": a noun, the verb, a function word
            if j >= 0 and toks[j].kind == T.WORD and (toks[j].lower == "to" or toks[j].lower in ("they", "we", "you")):
                return True
            continue
        if _participle(w) or re.search(r"(?:ize|ise|ify|izes|ises|ifies)$", w) and len(w) > 5:
            return True
        if voc.dictionary is not None and "v" in voc.dictionary.approved.get(w, ()):
            return True
        if w in _FUNCTION or w in W.ADJECTIVES or w.endswith("ly") or t.proper:
            continue
        if i + 1 < len(toks):
            nxt = toks[i + 1]
            known_noun = _noun(t, voc)[0]
            if nxt.kind == T.WORD and (nxt.lower in ("it", "them") or (nxt.lower in W.DETERMINERS and nxt.lower != "that")):
                return True
            if nxt.kind == T.NUMBER and not known_noun and not t.tech:
                return True
    return False


def _forms_of_verbs() -> dict[str, str]:
    out: dict[str, str] = {}
    for base in W.VERBS:
        for form in D.verb_forms(base, W.IRREGULAR_VERBS.get(base, ())):
            out.setdefault(form, base)
    for base, forms in W.IRREGULAR_VERBS.items():
        for form in (base, *forms):
            out.setdefault(form, base)
    return out


_VERB_FORMS = _forms_of_verbs()


def _check_sentence(s: T.Sentence, kind: str, voc: Vocabulary, found: _Found) -> None:
    toks = s.tokens
    n = len(toks)
    counted = [t for t in toks if t.kind in T.COUNTED]
    limit = LIMIT[kind]

    # ---- sentence length (STE 5.1, STE 6.3; the word count is STE 8.4 to 8.7)
    if len(counted) > limit:
        key = "length_procedure" if kind == "procedure" else "length_descriptive"
        what = "an instruction" if kind == "procedure" else "a descriptive sentence"
        message = (f"The text in parentheses has {len(counted)} words, and the limit is {limit} words." if s.paren
                   else f"The sentence has {len(counted)} words, and the limit for {what} is {limit} words.")
        found.add(key, "error", message, s, found.span(counted[limit], counted[-1]), s.start)

    used: set[int] = set()          # tokens that already have an issue about their word
    tense: set[int] = set()         # "-ing" forms that are a part of a progressive tense

    for i, t in enumerate(toks):
        # ---- punctuation and abbreviations
        if t.kind == T.PUNCT and t.text == ";":
            if not any(x[2].name == "semicolon" and x[2].sentence == s.text for x in found.items):
                found.add("semicolon", "error", "The sentence has a semicolon.", s, ";", t.start)
            continue
        if t.kind == T.LATIN:
            index = int(t.name[1:])
            found.add("latin", "error", f"{_q(t.text)} is a Latin abbreviation.", s, t.text, t.start, W.LATIN[index][2])
            continue
        if t.kind == T.ANDOR:
            found.add("andor", "error", "The term \"and/or\" is not a word of the STE dictionary.", s, t.text, t.start)
            continue
        if t.kind == T.ACRONYM and voc.dictionary is not None and not t.tech:
            if voc.dictionary.lookup(t.lower)[0] != "approved":
                found.add("unknown", "error", f"The abbreviation {_q(t.text)} is not in the STE dictionary of the "
                          "project and is not a technical name.", s, t.text, t.start)
            continue
        if t.kind != T.WORD:
            continue
        w = _key(t)

        # ---- contractions (STE 4.2)
        if "'" in t.lower and _CONTRACTION.match(t.lower):
            fix = _EXPANSIONS.get(t.lower, "")
            if not fix and t.lower.endswith("n't"):
                fix = t.lower[:-3] + " not"
            elif not fix:
                head, _, tail = t.lower.partition("'")
                fix = {"re": f"{head} are", "ve": f"{head} have", "ll": f"{head} will", "m": f"{head} am"}.get(tail, "")
            found.add("contraction", "error", f"{_q(t.text)} is a contraction.", s, t.text, t.start, fix)
            used.add(i)
            continue

        # ---- verb forms: perfect tense, passive voice (STE 3.2, STE 3.6)
        if w in ("has", "have", "had"):
            j = _after(toks, i, 2)
            if j < n and toks[j].kind == T.WORD and (toks[j].lower == "had" or _participle(toks[j].lower)):
                k = toks[j + 1] if j + 1 < n else None
                sure = toks[j].lower == "been" or k is None or k.kind != T.WORD or k.lower in _FUNCTION \
                    or k.lower.endswith("ly")
                if sure:
                    words = found.span(t, toks[j])
                    found.add("perfect", "error", f"The verb {_q(words)} is in a perfect tense, and STE uses only "
                              "the simple tenses.", s, words, t.start)
        if w in W.BE:
            j = _after(toks, i)
            if j < n and toks[j].kind == T.WORD and not toks[j].proper:
                p = toks[j].lower
                if _participle(p) and p not in W.STATE_PARTICIPLES:
                    agent = any(x.kind == T.WORD and x.lower == "by" for x in toks[j + 1:j + 5])
                    words = found.span(t, toks[j])
                    if kind == "procedure":
                        found.add("passive", "error" if agent else "warning", f"The verb {_q(words)} is possibly in "
                                  "the passive voice, and an instruction uses the active voice.", s, words, t.start)
                    else:
                        found.add("passive", "warning", f"The verb {_q(words)} is possibly in the passive voice.",
                                  s, words, t.start)
                elif _ing(toks[j], voc):
                    words = found.span(t, toks[j])
                    found.add("progressive", "error", f"The verb {_q(words)} is in a progressive tense, and STE "
                              "uses only the simple tenses.", s, words, t.start)
                    tense.add(j)

        if t.tech or t.known:
            continue

        # ---- "-ing" forms (STE 3.5)
        if i not in tense and _ing(t, voc):
            found.add("ing", "error", f"The word {_q(t.text)} is an \"-ing\" form, and STE accepts this form only "
                      "in a technical name.", s, t.text, t.start)
            used.add(i)
            continue
        if i in tense:
            used.add(i)

    _check_words(s, kind, voc, found, used)
    _check_clusters(s, voc, found)

    # ---- a sentence without a verb (STE 4.2)
    if s.terminated and not s.paren and counted and not _has_verb(toks, voc) \
            and not any(t.kind in (T.FORMULA, T.QUOTE) for t in toks):
        found.add("no_verb", "warning", "The checker found no verb in this sentence.", s, s.text, s.start)

    # ---- more than one instruction (STE 5.2)
    if kind == "procedure" and not s.paren:
        for i, t in enumerate(toks):
            if t.kind != T.WORD or t.lower not in ("and", "then") or i == 0:
                continue
            j = i + 1
            if j < n and toks[j].kind == T.WORD and toks[j].lower == "then":
                j += 1
            if j >= n or toks[j].kind != T.WORD:
                continue
            v = _key(toks[j])
            if (v in W.VERBS or v in voc.verb_bases) and v not in W.BE and \
                    _follower(toks, j, frozenset({"sure", "that", "all"})):
                before = toks[_before(toks, i)]
                if before.kind == T.WORD and before.lower == "to":
                    continue
                words = found.span(t, toks[min(n - 1, j + 1)])
                found.add("instructions", "warning", f"The sentence possibly gives a second instruction at {_q(words)}.",
                          s, words, t.start)
                break


def _check_words(s: T.Sentence, kind: str, voc: Vocabulary, found: _Found, used: set[int]) -> None:
    """Words that are not approved (STE 1.1, STE 1.2, STE 1.11)."""
    toks = s.tokens
    n = len(toks)
    d = voc.dictionary

    def group(table: dict, i: int):
        """The longest group of words of `table` that starts at token i: (length, value)."""
        for keys, value in table.get(_key(toks[i]), ()):
            m = len(keys)
            if i + m <= n and all(toks[i + j].kind in _MATCHABLE and _key(toks[i + j]) == keys[j] for j in range(m)):
                return m, value
        return 0, None

    i = 0
    while i < n:
        t = toks[i]
        if t.kind not in (T.WORD, T.ACRONYM) or t.tech or i in used:
            i += 1
            continue
        w = _key(t)
        # the project's own list
        m, alt = group(voc.project, i)
        if m:
            words = found.span(t, toks[i + m - 1])
            found.add("project_word", "error", f"The terms of the project say that {_q(words)} is not an approved "
                      "word.", s, words, t.start, alt)
            i += m
            continue
        if t.kind != T.WORD:
            i += 1
            continue
        month = w == "may" and t.text[:1].isupper() and (t.proper or (i + 1 < n and toks[i + 1].kind == T.NUMBER))
        if d is not None:
            m, entry = group(d.phrases, i)
            if m and entry.status == "approved":
                for j in range(m):
                    toks[i + j].known = True
                i += m
                continue
            if m:
                words = found.span(t, toks[i + m - 1])
                found.add("word", "error", f"The STE dictionary of the project says that {_q(words)} is not an "
                          "approved word.", s, words, t.start, entry.alternative)
                i += m
                continue
            if t.proper or month or w in W.NUMBER_WORDS or len(w) == 1:
                i += 1
                continue
            status, entry = d.lookup(w)
            if status == "unknown" and "-" in w:
                parts = [p for p in w.split("-") if p]
                if all(p in _PREFIXES or p in W.NUMBER_WORDS or p in voc.verbs or p in voc.name_nouns
                       or d.lookup(p)[0] == "approved" for p in parts):
                    status = "approved"
            if status == "approved":
                e = d.verb_only_noun(w)
                if e is not None and _verb_use(toks, i, w):
                    found.add("pos", "warning", f"{_q(t.text)} is an approved noun, and it is not an approved verb.",
                              s, t.text, t.start, e.alternative)
            elif status == "not_approved":
                key = "modal" if w in W.MODALS_NOT_APPROVED else "word"
                alt = entry.alternative or voc.builtin.get(w, ("", "", "", ""))[2]
                message = (f"The modal verb {_q(t.text)} is not an approved STE word." if key == "modal" else
                           f"The STE dictionary of the project says that {_q(t.text)} is not an approved word.")
                found.add(key, "error", message, s, t.text, t.start, alt)
            else:
                alt = voc.builtin.get(w, ("", "", "", ""))[2]
                found.add("unknown", "error", f"The word {_q(t.text)} is not in the STE dictionary of the project "
                          "and is not a technical name.", s, t.text, t.start, alt)
            i += 1
            continue
        # no dictionary file: the small list of public guidance
        m, alt = group(voc.builtin_phrases, i)
        if m:
            if w == "close" and not _follower(toks, i + 1):
                i += 1
                continue                                  # "the valves close to stop the flow": a verb
            words = found.span(t, toks[i + m - 1])
            found.add("word", "warning", f"Public STE guidance says that {_q(words)} is not an approved word.", s,
                      words, t.start, alt)
            i += m
            continue
        hit = voc.builtin.get(w)
        if hit and not t.proper and not month:
            base, pos, alt, check_name = hit
            if check_name == "modal verb":
                found.add("modal", "error", f"The modal verb {_q(t.text)} is not an approved STE word.", s, t.text,
                          t.start, alt)
            elif pos == "v*":
                if _verb_use(toks, i, base):
                    found.add("pos", "warning", f"{_q(t.text)} is an approved noun, and it is not an approved verb.",
                              s, t.text, t.start, alt)
            else:
                found.add("word", "warning", f"Public STE guidance says that {_q(t.text)} is not an approved word.",
                          s, t.text, t.start, alt)
        i += 1


def _check_clusters(s: T.Sentence, voc: Vocabulary, found: _Found) -> None:
    """More than three nouns in a row (STE 2.1). The check is conservative: it counts only the words that it
    knows as nouns, and a plural ends a group."""
    toks = s.tokens
    run: list[T.Token] = []

    def close():
        if len(run) > 3:
            words = found.span(run[0], run[-1])
            found.add("cluster", "warning", f"The words {_q(words)} are possibly a group of {len(run)} nouns, and "
                      "the limit is 3 nouns.", s, words, run[0].start)
        run.clear()

    for t in toks:
        noun, plural = _noun(t, voc)
        if not noun:
            close()
            continue
        run.append(t)
        if plural:
            close()
    close()


# --------------------------------------------------------------------------- one text


def _analyse(text: str, kind: str, voc: Vocabulary) -> tuple[list[Issue], int, int]:
    """(the issues in the order of the text, the number of sentences, the number of words)."""
    sentences = T.segment(text, _LATIN)
    found = _Found(text)
    for s in sentences:
        _mark(s, voc)
        _check_sentence(s, kind, voc, found)
    top = [s for s in sentences if not s.paren]
    if kind == "descriptive":                       # STE 6.6 is a rule of descriptive text
        by_paragraph: dict[int, list[T.Sentence]] = {}
        for s in top:
            if not s.item:                          # the items of a vertical list belong to the sentence before them
                by_paragraph.setdefault(s.paragraph, []).append(s)
        for group in by_paragraph.values():
            if len(group) > PARAGRAPH_LIMIT:
                extra = " ".join(x.text for x in group[PARAGRAPH_LIMIT:])
                found.add("paragraph", "error", f"The paragraph has {len(group)} sentences, and the limit is "
                          f"{PARAGRAPH_LIMIT} sentences.", group[0], extra, group[0].start)
    found.items.sort(key=lambda x: (x[0], x[1]))
    return [x[2] for x in found.items], len(top), sum(s.words for s in top)


def check(text: str, kind: str = "descriptive", terms: dict | None = None) -> list[Issue]:
    """The STE problems of a text, in the order of the text.

    kind    "descriptive" (25 words in a sentence, six sentences in a paragraph) or "procedure" (an
            instruction: 20 words, the active voice only, one instruction in each sentence).
    terms   the content of the project's terms.yaml: `technical_names`, `technical_verbs`, optional
            `not_approved` ({word: the word to use}) and optional `dictionary` (the path of the user's
            dictionary file; see `dossier.ste.dictionary`). `dossier.ste.project_terms(project)` gives it.
    """
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {', '.join(KINDS)}, not {kind!r}")
    if not text or not str(text).strip():
        return []
    return _analyse(str(text), kind, vocabulary(terms))[0]


def summarize(texts: dict[str, str], kind: str | dict | Callable[[str], str] = "descriptive",
              terms: dict | None = None) -> dict:
    """The STE check of many texts, for the STE appendix of a report.

    texts   {id: text}; the id says where the text is, for example "FND-REQ-MOV-001-MEOP.statement".
    kind    one kind for every text, or {id: kind}, or a function of the id (`dossier.ste.field_kind`).

    Returns a dict that JSON can hold:
        standard   the issue of the standard
        checked    the number of texts examined (empty texts are not counted)
        clean      the number of texts without an issue
        errors, warnings            the number of issues of each severity
        texts_with_errors           the number of texts with one error or more
        sentences, words            the size of the text examined
        issues     for each text that has an issue: {"id", "kind", "errors", "warnings", "issues": [Issue as dict]}
        by_rule    {rule: {"count", "errors", "warnings", "names": {name of the check: count}}}
        fixes      {name of the check: what the writer does about it}
        limits     STE sentences: what the checker examines exactly, what by estimate, what not at all
        not_checked  [{"rule", "what"}]: the STE rules that the checker does not examine
        dictionary the dictionary file used, or None
    """
    voc = vocabulary(terms)

    def kind_of(key: str) -> str:
        k = kind(key) if callable(kind) else kind.get(key, "descriptive") if isinstance(kind, dict) else kind
        if k not in KINDS:
            raise ValueError(f"kind must be one of {', '.join(KINDS)}, not {k!r} (text {key!r})")
        return k

    out: dict = {"standard": STANDARD, "checked": 0, "clean": 0, "errors": 0, "warnings": 0, "texts_with_errors": 0,
                 "sentences": 0, "words": 0, "issues": [], "by_rule": {}, "fixes": {}}
    for key, text in texts.items():
        if not text or not str(text).strip():
            continue
        k = kind_of(key)
        issues, n_sentences, n_words = _analyse(str(text), k, voc)
        out["checked"] += 1
        out["sentences"] += n_sentences
        out["words"] += n_words
        if not issues:
            out["clean"] += 1
            continue
        errors = sum(1 for x in issues if x.severity == "error")
        out["errors"] += errors
        out["warnings"] += len(issues) - errors
        out["texts_with_errors"] += 1 if errors else 0
        out["issues"].append({"id": key, "kind": k, "errors": errors, "warnings": len(issues) - errors,
                              "issues": [x.as_dict() for x in issues]})
        for x in issues:
            r = out["by_rule"].setdefault(x.rule, {"count": 0, "errors": 0, "warnings": 0, "names": {}})
            r["count"] += 1
            r["errors" if x.severity == "error" else "warnings"] += 1
            r["names"][x.name] = r["names"].get(x.name, 0) + 1
    out["by_rule"] = dict(sorted(out["by_rule"].items(), key=lambda kv: -kv[1]["count"]))
    seen = {name for r in out["by_rule"].values() for name in r["names"]}
    out["fixes"] = {r.name: r.fix for r in RULES.values() if r.name in seen}
    out["limits"] = voc.limits()
    out["not_checked"] = [{"rule": rule, "what": what} for rule, what in NOT_CHECKED]
    out["dictionary"] = str(voc.dictionary.path) if voc.dictionary is not None else None
    return out


def count_words(sentence: str) -> int:
    """The STE word count of a text (STE 8.4 to 8.7): text in parentheses is one word."""
    return sum(s.words for s in T.segment(sentence, _LATIN) if not s.paren)


def split_sentences(text: str) -> list[str]:
    """The sentences of a text as the checker sees them (text in parentheses stays in its sentence)."""
    return [s.text for s in T.segment(text, _LATIN) if not s.paren]


def clean(texts: Iterable[str], kind: str = "descriptive", terms: dict | None = None) -> bool:
    """True when no text has an STE error (warnings do not count)."""
    return not any(x.severity == "error" for t in texts for x in check(t, kind, terms))

