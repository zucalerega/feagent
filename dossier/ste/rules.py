"""The STE rules that the checker knows: the number, how the checker examines the rule, and the sources.

The standard is ASD-STE100, Issue 9 (January 2025). Part 1 holds 53 writing rules in nine sections and a
set of general recommendations (GR); Part 2 is the dictionary. The rule summaries below are this module's
own words, not the text of the standard. The standard is free on request from ASD, and its text is the
authority: https://www.asd-ste100.org/

How each rule number was confirmed (`SOURCES`): this work could not read the standard itself, so every number
comes from public sources that were read through a web search. "index" is the subject-to-rule index and the
change list of the Issue 9 document as the search showed them. "techscribe" is the rule table that a
vendor of an STE term checker publishes for Issue 9. Where only summaries by other people give a number, the
entry says so (`confirmed`).

Issue 9 changed two names that older texts and this project still use: a "technical name" is now a
"technical noun", and a "noun cluster" is now a "multi-word noun". The rule numbers did not change from
Issue 7 (2017) to Issue 9; older issues had other numbers (for example the article rule was 2.3, now 4.5).
"""

from __future__ import annotations

from dataclasses import dataclass

STANDARD = "ASD-STE100 Issue 9 (January 2025)"

SOURCES: dict[str, str] = {
    "asd": "https://www.asd-ste100.org/ (the standard, ASD-STE100 Issue 9, January 2025; free on request)",
    "index": "the subject-to-rule index and the highlights of changes of the Issue 9 document, read through a "
             "web search of https://www.asd-ste100.org/",
    "faq": "https://www.asd-ste100.org/STE_faq.html (procedures must not use the passive voice; a description "
           "can use it only when the agent is not known)",
    "techscribe": "https://www.simplified-english.co.uk/rules-ste9.html (TechScribe, term checker rules for "
                  "ASD-STE100 issue 9: one line for each rule, with its number)",
    "wikipedia": "https://en.wikipedia.org/wiki/Simplified_Technical_English",
    "asd-news": "https://www.asd-europe.org/news-media/news-events/news/simplified-technical-english-asd-ste100-"
                "issue-9/ (53 rules, the wording of 31 rules revised, no new rule)",
    "ceur": "https://ceur-ws.org/Vol-3990/short24.pdf (Issue 9: \"technical name\" became \"technical noun\")",
    "axiom": "https://axiom-verity.com/ste/guides/what-is-asd-ste100 and https://axiom-verity.com/ste/rules/8.6 "
             "(a guide to Issue 9: what counts as one word)",
    "nardit": "https://max.nardit.com/articles/karpathy-understanding-llm-outputs (a comparison of a rule sheet "
              "with Issue 9: rules 2.1, 3.2, 3.5, 5.1, 5.2, 6.3, 6.6, 8.1)",
    "markup": "https://support.markup.ai/hc/en-us/articles/10210995127698-STE-Simplified-Technical-English "
              "(a checker vendor's list of the rules it supports, by number)",
    "hermes": "https://hermes-agent.nousresearch.com/docs/user-guide/skills/optional/creative/creative-simple-"
              "english (a summary of the 53 rules of Issue 9 in other words)",
}


@dataclass(frozen=True)
class Rule:
    key: str           # the name of the check in this package
    rule: str          # the STE rule, as an Issue gives it
    name: str          # a few words
    summary: str       # what the rule says, in this module's words
    how: str           # "exact" | "word list" | "heuristic"
    fix: str           # what the writer does about it (an STE instruction)
    sources: tuple[str, ...]
    confirmed: str     # how sure the rule number is


RULES: dict[str, Rule] = {r.key: r for r in (
    Rule("length_procedure", "STE 5.1", "sentence length",
         "An instruction has 20 words or fewer.", "exact",
         "Write two sentences.", ("techscribe", "wikipedia", "nardit"),
         "The rule table of TechScribe and several summaries give 5.1."),
    Rule("length_descriptive", "STE 6.3", "sentence length",
         "A descriptive sentence has 25 words or fewer.", "exact",
         "Write two sentences.", ("nardit", "markup", "wikipedia"),
         "Several summaries of Issue 9 give 6.3; the limit of 25 words is in every source."),
    Rule("paragraph", "STE 6.6", "paragraph length",
         "A paragraph of a description has six sentences or fewer.", "exact",
         "Start a new paragraph for the next topic.", ("nardit", "markup", "wikipedia"),
         "Several summaries of Issue 9 give 6.6; the limit of six sentences is in every source."),
    Rule("contraction", "STE 4.2", "contraction",
         "Do not leave out words and do not use contractions to make a sentence shorter.", "exact",
         "Write the two words in full.", ("techscribe",),
         "The rule table of TechScribe gives 4.2."),
    Rule("no_verb", "STE 4.2", "sentence without a verb",
         "Do not leave out words to make a sentence shorter: a sentence has a verb.", "heuristic",
         "Write a full sentence with a subject and a verb.", ("techscribe",),
         "The rule is 4.2; that a sentence without a verb breaks it is the reading of this module."),
    Rule("semicolon", "STE 8.1", "semicolon",
         "Every punctuation mark of English is permitted but the semicolon.", "exact",
         "Write two sentences.", ("index", "nardit", "hermes"),
         "The index puts punctuation in 8.1 to 8.3; summaries give 8.1 for the semicolon."),
    Rule("latin", "STE GR-6", "Latin abbreviation",
         "General recommendation 6: do not use Latin abbreviations.", "exact",
         "Write the English words.", ("index", "techscribe"),
         "The index gives \"Latin abbreviations: Section 9, GR-6\". It is a recommendation, not one of the 53 rules; "
         "an abbreviation is also a word that the dictionary does not approve (rule 1.1)."),
    Rule("andor", "STE section 1 (Words)", "\"and/or\"",
         "\"and/or\" is not a word of the dictionary.", "exact",
         "Write \"A or B\", or \"A, B or the two\".", (),
         "No public source gives an STE rule about the slash. The text rules of this project (CONTRACTS.md) do not "
         "permit \"and/or\"; the checker reports it under the section name."),
    Rule("modal", "STE 1.1", "modal verb",
         "Use only approved words. Public guidance agrees that \"can\", \"must\" and \"will\" are approved and "
         "that \"should\", \"shall\", \"may\", \"might\", \"could\" and \"would\" are not.", "word list",
         "Use \"must\" for an obligation and \"can\" for an ability or a possibility.", ("techscribe", "wikipedia"),
         "Rule 1.1 is confirmed. The status of each word comes from summaries (dossier.ste.words), not from "
         "the dictionary."),
    Rule("word", "STE 1.1", "word not approved",
         "Use only approved words of the dictionary, technical names and technical verbs.", "word list",
         "Use the approved alternative.", ("techscribe", "wikipedia"),
         "Rule 1.1 is confirmed. Without the user's dictionary file the checker knows only a small list of "
         "words from public guidance."),
    Rule("unknown", "STE 1.1", "word not in the dictionary",
         "Use only approved words of the dictionary, technical names and technical verbs.", "word list",
         "Use an approved word, or add the word to the technical names of the project if it is one.",
         ("techscribe", "wikipedia"), "Rule 1.1 is confirmed. This check needs the user's dictionary file."),
    Rule("pos", "STE 1.2", "part of speech",
         "Use an approved word only as the part of speech that the dictionary gives.", "heuristic",
         "Use the word as the dictionary gives it (\"do a test\", not \"test\").", ("index", "wikipedia"),
         "The Issue 9 document gives this rule as 1.2."),
    Rule("project_word", "STE 1.11", "project word",
         "Do not use different technical names for the same thing.", "word list",
         "Use the word that the project gives.", ("techscribe",),
         "The rule table of TechScribe gives 1.11 for one name for each thing."),
    Rule("cluster", "STE 2.1", "noun cluster",
         "A multi-word noun (a noun cluster) has three nouns or fewer.", "heuristic",
         "Use \"of\" or \"for\" between the nouns, or use hyphens in a technical name.", ("techscribe", "nardit"),
         "The rule table of TechScribe gives 2.1."),
    Rule("perfect", "STE 3.2", "perfect tense",
         "The permitted verb forms are the infinitive, the imperative, the simple present, the simple past, the "
         "future and the past participle as an adjective.", "exact",
         "Use the simple past or the simple present.", ("techscribe", "nardit"),
         "The rule table of TechScribe gives 3.2 for verb forms and 3.4 for complex verb structures; a perfect "
         "tense breaks the two rules."),
    Rule("progressive", "STE 3.2", "progressive tense",
         "The permitted verb forms are the infinitive, the imperative, the simple present, the simple past, the "
         "future and the past participle as an adjective.", "exact",
         "Use the simple present or the simple past.", ("techscribe", "nardit"),
         "The rule table of TechScribe gives 3.2 for verb forms."),
    Rule("ing", "STE 3.5", "\"-ing\" form",
         "Use the \"-ing\" form of a verb only as a technical name or as a part of a technical name.", "word list",
         "Use a simple tense or \"to\" and the verb. Add the word to the technical names of the project if it is "
         "a technical name.", ("techscribe", "wikipedia", "nardit"),
         "The rule table of TechScribe gives 3.5."),
    Rule("passive", "STE 3.6", "passive voice",
         "Use only the active voice in an instruction. Use the active voice as much as possible in a description "
         "(the passive voice is acceptable when the agent is not known).", "heuristic",
         "Make the agent the subject of the sentence.", ("techscribe", "index", "faq"),
         "The index and the rule table of TechScribe give 3.6."),
    Rule("instructions", "STE 5.2", "more than one instruction",
         "Write one instruction in each sentence, unless two actions occur at the same time.", "heuristic",
         "Write one sentence for each instruction.", ("nardit", "hermes"),
         "Summaries of Issue 9 give 5.2."),
)}

# Rules about the word count (they have no issue of their own: they decide how the checker counts).
WORD_COUNT = (
    ("STE 8.4", "The colon before a vertical list ends the sentence for the word count.", ("index", "hermes")),
    ("STE 8.5", "Text in parentheses is one word of the sentence. The checker also counts the text in the "
                "parentheses as a unit of its own.", ("techscribe", "axiom")),
    ("STE 8.6", "A number, a number with its unit, an abbreviation, an identifier, quoted text and a title are one "
                "word each.", ("index", "axiom")),
    ("STE 8.7", "A word with hyphens is one word.", ("axiom", "hermes")),
)

# Rules that the checker uses but that cannot fail: the project's words are accepted.
ACCEPTED = (
    ("STE 1.5", "Technical names (Issue 9: technical nouns) of the project are accepted: terms[\"technical_names\"]. "
                "Numbers, units of measurement and identifiers are technical names too.", ("wikipedia", "techscribe")),
    ("STE 1.12", "Technical verbs of the project are accepted: terms[\"technical_verbs\"].", ("wikipedia", "techscribe")),
)

# What a program cannot examine, or what this one does not.
NOT_CHECKED = (
    ("STE 1.2", "the part of speech of a word, except a clear use of \"test\" or \"check\" as a verb"),
    ("STE 1.3", "the meaning of a word: an approved word has only its approved meaning"),
    ("STE 1.11", "one name for one thing, except the words that the project lists as not approved"),
    ("STE 1.14", "American English spelling"),
    ("STE 3.3", "the past participle as an adjective only (the passive voice check finds a part of this)"),
    ("STE 3.7", "a verb, not a noun, for an action"),
    ("STE 4.1", "one topic in each sentence, and a clear sentence"),
    ("STE 4.4", "connecting words between sentences"),
    ("STE 4.5", "the article before a noun"),
    ("STE 5.3", "the imperative form of an instruction"),
    ("STE 5.4", "the condition before the instruction"),
    ("STE 6.5", "one topic in each paragraph"),
    ("STE section 7", "safety instructions (warnings and cautions)"),
    ("STE 8.2 and 8.3", "the correct use of hyphens and of parentheses"),
    ("STE section 9", "writing practices, and the general recommendations other than GR-6"),
)


def rule(key: str) -> Rule:
    return RULES[key]
