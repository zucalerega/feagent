"""Word lists of the STE checker.

None of these lists is the ASD-STE100 dictionary (Part 2 of the standard). The dictionary is the
property of ASD and this package does not hold it. The lists here are of three kinds:

* English grammar words that the heuristics need (forms of "be", prepositions, irregular participles).
* A small list of words that public STE guidance gives as not approved, each with its source
  (`NOT_APPROVED`). The user's dictionary file overrides this list.
* General nouns and verbs of engineering text. The noun-cluster check and the verb check use them
  only to decide what a word probably is. They say nothing about what STE approves.
"""

from __future__ import annotations

# --------------------------------------------------------------------------- sources of the word pairs
#
# Grade A: the standard's own public pages, Wikipedia, and people who work with the standard.
# Grade B: summaries of the standard by other people. Several of them agree, but none is the standard.
WORD_SOURCES: dict[str, tuple[str, str]] = {
    "asd-highlights": ("A", "ASD-STE100 Issue 9, highlights of changes (the change list names \"therefore (adv)\" with the "
                            "approved alternative AS A RESULT): https://www.asd-ste100.org/"),
    "wikipedia": ("A", "https://en.wikipedia.org/wiki/Simplified_Technical_English (\"test\" is an approved noun and "
                       "not an approved verb; the adjective \"close\" is not approved, the alternative is NEAR)"),
    "techscribe-2013": ("A", "https://www.techscribe.co.uk/ta/technical-communication-conference-2013.htm "
                             "(\"check\" is approved as a noun and not as a verb)"),
    "ieee-procomm": ("A", "https://procomm.ieee.org/simplified-technical-english-and-clearer-documentation-through-the-"
                          "thumbs-up-technique-23/ (\"main\" is not approved, the approved word is \"primary\")"),
    "tcloc": ("A", "http://mastertcloc.unistra.fr/2019/08/26/tips-simplified-technical-english/ (\"may\" and \"would\": "
                   "the approved word is CAN; MAKE SURE is the approved verb)"),
    "lassure": ("A", "http://www.english-for-techies.net/grammar/STE_rules.htm (\"due to\": the replacement is "
                     "\"because of\")"),
    "mcgaw": ("B", "https://philipmcgaw.com/asd-ste100-clear-technical-english/ (Issue 9: \"shall\" and \"should\" "
                   "become \"must\", \"may\" becomes \"can\"; \"ought to\" is not permitted)"),
    "skills": ("B", "summaries of Issue 9 for writing tools, which agree with each other: "
                    "https://hermes-agent.nousresearch.com/docs/user-guide/skills/optional/creative/creative-simple-english "
                    "and https://github.com/nuelcyoung/asd-ste100 (may, might, could, should, would, shall are not "
                    "approved; ensure -> make sure; utilize -> use; prior to -> before; begin, commence, initiate -> "
                    "start; rotate -> turn)"),
    "asd-gr6": ("A", "ASD-STE100 Issue 9, subject index: Latin abbreviations, Section 9, general recommendation GR-6; "
                     "https://www.simplified-english.co.uk/rules-ste9.html gives the same number"),
}

# (word or phrase, part of speech, approved alternative, source key, name of the check)
#   part of speech "v*": the word is also a noun that STE approves, so only a use as a verb is a problem.
#   An empty alternative means: no single word replaces it, the writer must write the sentence again.
NOT_APPROVED: tuple[tuple[str, str, str, str, str], ...] = (
    # modal verbs (STE approves "can", "must" and "will")
    ("should", "v", "must", "mcgaw", "modal verb"),
    ("shall", "v", "must", "mcgaw", "modal verb"),
    ("may", "v", "can", "tcloc", "modal verb"),
    ("might", "v", "can", "skills", "modal verb"),
    ("could", "v", "can", "skills", "modal verb"),
    ("would", "v", "", "tcloc", "modal verb"),
    ("ought", "v", "must", "mcgaw", "modal verb"),
    # other words
    ("ensure", "v", "make sure", "skills", "word not approved"),
    ("utilize", "v", "use", "skills", "word not approved"),
    ("commence", "v", "start", "skills", "word not approved"),
    ("initiate", "v", "start", "skills", "word not approved"),
    ("begin", "v", "start", "skills", "word not approved"),
    ("rotate", "v", "turn", "skills", "word not approved"),
    ("prior to", "prep", "before", "skills", "word not approved"),
    ("due to", "prep", "because of", "lassure", "word not approved"),
    ("close to", "adj", "near", "wikipedia", "word not approved"),
    ("therefore", "adv", "as a result", "asd-highlights", "word not approved"),
    ("main", "adj", "primary", "ieee-procomm", "word not approved"),
    ("test", "v*", "do a test", "wikipedia", "part of speech"),
    ("check", "v*", "do a check", "techscribe-2013", "part of speech"),
)

# Modal verbs: several sources agree, so the checker gives an error. The other words of the list come
# from fewer sources, so the checker gives a warning unless the user's dictionary confirms them.
MODALS_NOT_APPROVED = frozenset(w for w, _p, _a, _s, name in NOT_APPROVED if name == "modal verb")
MODALS_APPROVED = frozenset({"can", "cannot", "must", "will"})

# Irregular forms of the verbs in NOT_APPROVED (the regular forms are made by rule).
IRREGULAR_FORMS: dict[str, tuple[str, ...]] = {
    "begin": ("begins", "began", "begun"),
}

# Latin abbreviations (general recommendation GR-6): pattern -> (name shown, alternative or "").
LATIN: tuple[tuple[str, str, str], ...] = (
    (r"e\.\s?g\.", "e.g.", "for example"),
    (r"i\.\s?e\.", "i.e.", "that is"),
    (r"etc\b\.?", "etc.", ""),
    (r"et\s+al\b\.?", "et al.", ""),
    (r"c\.?f\.", "cf.", ""),
    (r"viz\b\.?", "viz.", ""),
    (r"vs\b\.?", "vs.", ""),
    (r"n\.\s?b\.", "N.B.", ""),
    (r"ibid\b\.?", "ibid.", ""),
    (r"q\.\s?v\.", "q.v.", ""),
)

# --------------------------------------------------------------------------- grammar words

BE = frozenset({"am", "is", "are", "was", "were", "be", "been", "being"})
HAVE = frozenset({"has", "have", "had", "having"})
DO = frozenset({"do", "does", "did"})

DETERMINERS = frozenset("""
the a an this that these those each every all both any some no its their his her our your my another such
either neither one two three four five six seven eight nine ten many few several more most less other
""".split())

PREPOSITIONS = frozenset("""
of in on at by for with from to into onto through during before after between among above below under over
near against along across around behind without within about as per than until since toward towards upon via
off out up down inside outside except
""".split())

CONJUNCTIONS = frozenset("""
and or but if when while because although though unless that which who whose where whereas so then thus nor
yet whether
""".split())

PRONOUNS = frozenset("""
it they them he she we you i itself themselves there here what none nothing something anything everything
""".split())

# Adverbs that can stand between a helping verb and the main verb ("is not yet solved").
MID_ADVERBS = frozenset("""
not never always also only then still already now thus first often usually again just even ever yet both all
each almost too very more most less least well so here there
""".split())

NUMBER_WORDS = frozenset("""
zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen
seventeen eighteen nineteen twenty thirty forty fifty sixty seventy eighty ninety hundred thousand million
billion first second third fourth fifth sixth seventh eighth ninth tenth half twice once
""".split())

# Common adjectives. A word of this list is not a noun in the noun-cluster check and is not a verb in
# the verb check.
ADJECTIVES = frozenset("""
main primary secondary maximum minimum static dynamic thermal structural steady total local global nominal
allowable high low large small long short thick thin hot cold new old same different equal full empty free
open closed safe unsafe correct incorrect possible necessary sufficient applicable available important
internal external upper lower inner outer front rear left right top bottom first last next initial final
linear nonlinear elastic plastic rigid flexible coarse fine sharp smooth flat round axial radial tangential
normal parallel vertical horizontal positive negative conservative critical typical general specific
mechanical hydraulic electrical cryogenic liquid solid gaseous dry wet clean clear good bad true false
mean average peak rated effective relative absolute partial complete simple complex direct indirect
""".split())

# --------------------------------------------------------------------------- verb forms

# Words that end in "-ed" and are not past participles.
NOT_PARTICIPLES = frozenset("""
need speed feed seed bleed breed deed weed reed greed creed heed indeed exceed proceed succeed hundred embed
bed red sacred naked wicked kindred infrared biped moped shred sled ahead
""".split())

IRREGULAR_PARTICIPLES = frozenset("""
been done made given taken shown known seen found held kept left put set cut read written built bought brought
caught chosen come become begun broken drawn driven eaten fallen felt got gotten gone grown heard hidden hit
hurt laid led lost meant met paid run said sent shut sold spent split spread stood stuck taught thought told
understood worn won wound bent bound dealt fed fit frozen hung lit proven risen shaken shed shrunk sunk swept
swollen torn thrown woken withdrawn withheld withstood overcome undergone undertaken rewritten redone input
output reset offset preset upset
""".split())

# Past participles of verbs that have no passive (verbs without an object) and that are used as
# adjectives: "the result is converged" gives a state, it is not the passive voice.
STATE_PARTICIPLES = frozenset({"converged", "unconverged", "gone", "finished"})

# Words that end in "-ing" and are not forms of a verb.
ING_NOT_VERB = frozenset("""
during thing things nothing something anything everything morning evening ceiling ceilings missing
lightning sibling siblings sterling herring inning innings awning awnings
""".split())

# General technical names that end in "-ing": things, physical effects and processes. STE permits an
# "-ing" form as a technical name. A project adds its own names in terms.yaml.
ING_TECHNICAL = frozenset("""
housing bearing fitting opening coating casing bushing wiring piping tubing plating lining mounting coupling
gearing spacing packing seating rating reading setting timing drawing marking warning lighting building
landing cladding shielding bonding molding moulding ducting fairing cowling casting forging winding tooling
bending buckling damping cooling heating welding brazing soldering machining engineering manufacturing
handling
""".split())

# Base forms of common verbs of engineering text (this is not the list of approved STE verbs).
VERBS = frozenset("""
accept act add adjust agree align allow apply approve ask assemble assess assume attach avoid balance bear
become belong bend bolt break bring build calculate call carry cause change check choose clamp clean close
collect combine come compare complete compute connect consider contain continue control converge convert cool
correct count cover create cut damage decide decrease define delete depend describe design detect differ
disagree disconnect divide do drain draw drive drop enter estimate evaluate examine exceed expand expect
explain extend fail fall fasten fill find finish fit flow follow get give go govern grow handle happen have
heat help hold identify ignore include increase indicate inspect install join keep know lead leak leave let
lie lift limit link list load lock look lose lower lubricate make mark match mean measure meet melt mesh
miss mount move multiply need obey occur omit open operate pass permit place point prepare press pressurize
prevent produce protect prove provide pull push put raise reach react read record reduce refer refine reject
release remain remove repair repeat replace report represent require resist rest result return review rise
run satisfy say scale seal see select send separate serve set show shut solve start state stay stop store
study subtract supply support take tell tighten touch transfer transmit treat try turn use validate vary
verify wait want wear weld withstand work write yield
""".split())

# base form -> the other forms (third person, past, past participle); regular forms are made by rule.
IRREGULAR_VERBS: dict[str, tuple[str, ...]] = {
    "be": ("am", "is", "are", "was", "were", "been"), "have": ("has", "had"), "do": ("does", "did", "done"),
    "bear": ("bore", "borne"), "become": ("became",), "bend": ("bent",), "break": ("broke", "broken"),
    "bring": ("brought",), "build": ("built",), "choose": ("chose", "chosen"), "come": ("came",), "cut": (),
    "draw": ("drew", "drawn"), "drive": ("drove", "driven"), "fall": ("fell", "fallen"), "find": ("found",),
    "fit": (), "get": ("got",), "give": ("gave", "given"), "go": ("goes", "went", "gone"),
    "grow": ("grew", "grown"), "hold": ("held",), "keep": ("kept",), "know": ("knew", "known"),
    "lead": ("led",), "leave": ("left",), "let": (), "lie": ("lay", "lain"), "lose": ("lost",),
    "make": ("made",), "mean": ("meant",), "meet": ("met",), "put": (), "read": (), "rise": ("rose", "risen"),
    "run": ("ran",), "say": ("said",), "see": ("saw", "seen"), "send": ("sent",), "set": (), "show": ("shown",),
    "shut": (), "take": ("took", "taken"), "tell": ("told",), "wear": ("wore", "worn"),
    "withstand": ("withstood",), "write": ("wrote", "written"),
}

# Verbs that are also common nouns: alone they are not proof that a sentence has a verb.
VERB_OR_NOUN = frozenset("""
act balance bolt break change check clamp control cool cover cut damage design drain drive drop estimate fall
fill fit flow handle heat help hold increase decrease leak lift limit link list load lock look mark match
mean measure mesh mount move need open pass place point press pull push raise reach record release repair
report rest result return review rise run scale seal set show start state stay stop store study supply
support touch transfer turn use wait wear weld work yield
""".split())

# --------------------------------------------------------------------------- nouns (noun-cluster check)

NOUNS = frozenset("""
valve body shell nozzle neck flange seat pocket stem bore cavity actuator pump injector oxygen pressure load
case stress strain margin safety factor yield strength limit value result analysis model mesh node element
face edge surface wall thickness diameter radius length area volume mass force moment temperature material
requirement system context phase cycle life fatigue design test check bolt seal inlet outlet flow rate drop
line pipe tank engine feature interface support displacement heat flux coefficient transfer fluid gas
section membrane level size study convergence error estimate uncertainty policy criterion parameter effect
connection component part assembly unit number point axis plane cylinder fillet hole thread weld joint plate
beam bracket housing bearing fitting opening coating spring ring gasket washer nut screw pin shaft rotor
blade disk hub cover cap plug port channel duct tube hose filter sensor switch signal control power energy
speed velocity acceleration frequency mode shape time step increment iteration solver solution equation
formula input output file record report document table figure picture drawing standard specification source
reference review decision assumption finding rule data field group set list range type kind class state
status condition environment operation storage transport shutdown startup surge closure propellant oxidizer
fuel steel alloy metal property modulus ratio density conductivity expansion capacity damage crack failure
fracture leak leakage corrosion vibration shock impact contact friction gap clearance tolerance position
location direction angle distance height width depth weight centre center centroid origin frame coordinate
vector tensor magnitude sum difference product amount quantity count index key name label title
""".split())
