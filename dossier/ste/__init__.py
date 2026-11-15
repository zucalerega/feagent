"""ASD-STE100 Simplified Technical English (STE) for dossier: a checker and helpers for writers.

    from dossier import ste
    ste.check("MEOP may be insufficient considering surge effects.")            # -> [Issue, Issue]
    ste.check(text, kind="procedure", terms=ste.project_terms(project))
    ste.summarize({"FND-1.statement": text, ...}, terms=...)                     # the STE appendix of a report
    python -m dossier.ste "Text to examine."        python -m dossier.ste out/validate.json
    python -m dossier.ste --rules                   python -m dossier.ste dossier/loadcases

The standard
    ASD-STE100, Issue 9, January 2025 (https://www.asd-ste100.org/). Part 1: 53 writing rules in nine
    sections (words; multi-word nouns; verbs; sentences; procedures; descriptive writing; safety
    instructions; punctuation and word count; writing practices) and general recommendations (GR).
    Part 2: the dictionary of approved words. ASD owns the dictionary: it is free on request, and this
    package does not hold it. `dossier.ste.rules` gives each rule that the checker uses with its number
    and the public source that confirms the number. Issue 9 names: "technical noun" (before: technical
    name) and "multi-word noun" (before: noun cluster).

What the checker examines exactly (an Issue with severity "error")
    STE 5.1, 6.3    the length of a sentence: 20 words in an instruction, 25 in a descriptive sentence
    STE 8.4-8.7     how words are counted: a number with its unit, an identifier, an abbreviation, quoted
                    text, a word with hyphens and text in parentheses are one word each; the colon before a
                    vertical list ends the sentence. A formula as one word is a convention of this package.
    STE 6.6         the length of a paragraph: six sentences (descriptive text)
    STE 4.2         contractions
    STE 8.1         semicolons
    STE GR-6        Latin abbreviations ("e.g.", "i.e.", "etc."): a general recommendation of the standard
    STE 3.2         perfect tenses ("has been", "have changed") and progressive tenses ("is running")
    STE 3.5         "-ing" forms that are not a technical name ("considering", "using")
    STE 1.1         modal verbs that are not approved: "should", "shall", "may", "might", "could", "would"
                    ("can", "must" and "will" are approved)
    STE 1.1, 1.11   with the user's dictionary file: every word that is not approved, and every word that is
                    neither in the dictionary, nor a technical name or verb, nor a number or an identifier;
                    the words that the project lists as not approved (terms["not_approved"])
    project rule    "and/or" (no STE rule number is confirmed for it)

What the checker examines by heuristic (severity "warning": a person decides)
    STE 3.6         the passive voice: a form of "be" and a past participle. In a description the standard
                    accepts the passive voice when the agent is not known, so it is always a warning there.
                    In a procedure it is an error when "by" and an agent follow, and a warning without.
    STE 2.1         noun clusters: more than three known nouns in a row
    STE 4.2         a sentence without a verb (only for text that ends with a full stop)
    STE 5.2         more than one instruction in a sentence (procedures)
    STE 1.1         without the dictionary file: a small list of words that public guidance gives as not
                    approved ("ensure", "utilize", "prior to", ...; `dossier.ste.words.NOT_APPROVED`)
    STE 1.2         "test" and "check" used as verbs (they are approved as nouns)

What the checker does not examine
    one topic in each sentence and in each paragraph (STE 4.1, 6.5); the meaning of a word (STE 1.3) and its
    part of speech in general (STE 1.2); one name for one thing (STE 1.11) other than the project's list; the
    article before a noun (STE 4.5); connecting words (STE 4.4); the imperative form of an instruction and
    the condition before it (STE 5.3, 5.4); a verb, not a noun, for an action (STE 3.7); American spelling
    (STE 1.14); safety instructions (Section 7); writing practices (Section 9). Without the user's dictionary
    file the vocabulary check is partial, and `summarize(...)["limits"]` says so.

What is always accepted
    The technical names and technical verbs of the project (STE 1.5, STE 1.12): terms["technical_names"],
    terms["technical_verbs"], and the names that the project's input files give (`project_terms`).
    Identifiers ("REQ-MOV-001", "LC-FIRE"), numbers with units ("10.72 MPa"), formulas, names in backquotes.
    Text in double quotation marks: text that an engineer wrote is quoted as it is (`quote`).

The user's dictionary file
    A CSV file with the columns word, part_of_speech, status (approved | not_approved), alternative
    (`dossier.ste.dictionary`). Its path is terms["dictionary"] or the environment variable
    DOSSIER_STE_DICTIONARY.

Helpers for writers (`dossier.ste.writer`)
    listing, plural, a, sentence, paragraph, quote, too_long, fit; also count_words and split_sentences.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Iterable

from dossier.ste.checker import (KINDS, LIMIT, PARAGRAPH_LIMIT, Issue, check, clean, count_words,
                                 split_sentences, summarize)
from dossier.ste.dictionary import DictionaryError
from dossier.ste.rules import NOT_CHECKED, RULES, SOURCES, STANDARD
from dossier.ste.writer import a, fit, listing, paragraph, plural, quote, sentence, too_long

__all__ = ["check", "summarize", "Issue", "clean", "listing", "plural", "a", "sentence", "paragraph", "quote",
           "too_long", "fit", "count_words", "split_sentences", "project_terms", "texts_of", "field_kind",
           "RULES", "SOURCES", "STANDARD", "NOT_CHECKED", "KINDS", "LIMIT", "PARAGRAPH_LIMIT", "DictionaryError",
           "TEXT_FIELDS", "PROCEDURE_FIELDS"]

# The fields of the records that hold text an engine wrote (dossier.model).
TEXT_FIELDS = ("title", "statement", "rationale", "choice", "impact", "suggestion", "verify_by", "message",
               "option", "why_not", "reasons", "text", "description", "caption", "assumptions", "warnings")
# Fields that hold an instruction: the limit is 20 words and the active voice is necessary.
PROCEDURE_FIELDS = ("suggestion", "verify_by")


def project_terms(project) -> dict:
    """The `terms` of a project for `check` and `summarize`.

    It is `project.terms` (the content of terms.yaml) with two additions. The names that the input files
    give (the system, its elements, phases, duty cycles and connections, the components, their interfaces,
    the materials) are technical names: an engineer gave them, and STE accepts the official name of a thing.
    A relative path of the dictionary file is made absolute from the project folder."""
    terms = dict(getattr(project, "terms", None) or {})
    names = [str(n) for n in (terms.get("technical_names") or []) if str(n).strip()]

    def add(obj: Any) -> None:
        name = getattr(obj, "name", None)
        if isinstance(name, str) and name.strip() and not re.search(r"[,;:()\"]", name):
            names.append(name.strip())

    system = getattr(project, "system", None)
    if system is not None:
        add(system)
        for attr in ("elements", "phases", "duty_cycles", "connections"):
            for x in getattr(system, attr, None) or ():
                add(x)
    for comp in (getattr(project, "components", None) or {}).values():
        add(comp)
        for it in getattr(comp, "interfaces", None) or ():
            add(it)
    for mat in (getattr(project, "materials", None) or {}).values():
        add(mat)
    terms["technical_names"] = list(dict.fromkeys(names))
    path = terms.get("dictionary")
    root = getattr(project, "root", None)
    if path and root is not None and not os.path.isabs(str(path)):
        terms["dictionary"] = str(Path(root) / str(path))
    return terms


def texts_of(records: Iterable, fields: Iterable[str] = TEXT_FIELDS) -> dict[str, str]:
    """The texts of records, for `summarize`: {"<record id>.<field>": text}.

    `records` are models of dossier.model (or their dicts). The function reads the fields of `fields` at
    every depth, so the options of a decision or the steps of a hand calculation are in the result:
    "DEC-LC-FIRE-SUPPORT.alternatives[0].why_not". A field of a list of texts gives one entry for each text."""
    wanted = set(fields)
    out: dict[str, str] = {}

    def walk(obj: Any, path: str) -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                p = f"{path}.{k}" if path else str(k)
                if k in wanted and isinstance(v, str):
                    if v.strip():
                        out[p] = v
                elif k in wanted and isinstance(v, list) and all(isinstance(x, str) for x in v):
                    for i, x in enumerate(v):
                        if x.strip():
                            out[f"{p}[{i}]"] = x
                else:
                    walk(v, p)
        elif isinstance(obj, (list, tuple)):
            for i, x in enumerate(obj):
                walk(x, f"{path}[{i}]")

    for n, rec in enumerate(records):
        data = rec.model_dump(mode="json") if hasattr(rec, "model_dump") else rec
        if isinstance(data, dict):
            walk(data, str(data.get("id") or f"record[{n}]"))
    return out


def field_kind(text_id: str) -> str:
    """The kind of a text from its id, for `summarize(texts, kind=field_kind)`: "procedure" for a field that
    holds an instruction (`PROCEDURE_FIELDS`: suggestion, verify_by), "descriptive" for the others."""
    last = re.sub(r"\[\d+\]$", "", text_id.rsplit(".", 1)[-1])
    return "procedure" if last in PROCEDURE_FIELDS else "descriptive"
