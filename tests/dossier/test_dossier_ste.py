"""Tests of dossier.ste: the rules, the word count, the dictionary file, the helpers and the CLI."""

from __future__ import annotations

import json

import pytest

from dossier import ste
from dossier.ste import __main__ as cli

TERMS = {"technical_names": ["valve body", "MEOP", "housing", "seat pocket"], "technical_verbs": ["mesh", "pressurize"]}


def rules(text, kind="descriptive", terms=None):
    return {i.rule for i in ste.check(text, kind, terms)}


def names(text, kind="descriptive", terms=None):
    return {i.name for i in ste.check(text, kind, terms)}


# ---- each rule: one sentence that breaks it, one that does not -----------------------------------------------

def test_descriptive_length():
    long = " ".join(["The valve body has a seat pocket"] + ["and a flange"] * 7) + "."
    assert ste.count_words(long) > 25
    assert "STE 6.3" in rules(long)
    assert "STE 6.3" not in rules("The valve body has a flange and a neck.")


def test_procedure_length():
    long = "Close the valve, open the drain, remove the cover, clean the seat and measure the gap between the stem and the seat."
    assert ste.count_words(long) > 20
    assert "STE 5.1" in rules(long, "procedure")
    assert "STE 5.1" not in rules("Close the valve.", "procedure")
    # 21 to 25 words: allowed in a description, not in an instruction
    s = "The valve body has a flange and a neck and a seat pocket and a stem bore and a cavity and a pump."
    assert 20 < ste.count_words(s) <= 25
    assert "STE 6.3" not in rules(s) and "STE 5.1" in rules(s, "procedure")


def test_paragraph_length():
    seven = " ".join(["The valve is open."] * 7)
    assert "STE 6.6" in rules(seven)
    assert "STE 6.6" not in rules(" ".join(["The valve is open."] * 6))
    assert "STE 6.6" not in rules("The valve is open.\n\n" + " ".join(["The valve is open."] * 6))


def test_contraction():
    assert "STE 4.2" in rules("The valve doesn't open.")
    assert "contraction" in names("It's open.")
    assert "contraction" not in names("The valve does not open.")


def test_passive():
    desc = ste.check("The seal is damaged by the fluid.")
    assert [i.severity for i in desc if i.rule == "STE 3.6"] == ["warning"]
    proc = ste.check("The valve is closed by the operator.", "procedure")
    assert [i.severity for i in proc if i.rule == "STE 3.6"] == ["error"]
    assert "STE 3.6" not in rules("The fluid damages the seal.")
    assert "STE 3.6" not in rules("The seal is open.")


def test_perfect_and_progressive():
    assert "STE 3.2" in rules("The load has been applied.")
    assert "STE 3.2" in rules("The pump is running.")
    assert "STE 3.2" not in rules("The load was applied.") and "STE 3.2" not in rules("The pump runs.")


def test_ing_forms_and_exceptions():
    assert "STE 3.5" in rules("MEOP may be insufficient considering surge effects.")
    assert "STE 3.5" not in rules("The housing has a bearing and a fitting.")
    assert "STE 3.5" not in rules("The opening has a coating.")
    assert "STE 3.5" in rules("The solver is using a mesh.") or "STE 3.2" in rules("The solver is using a mesh.")
    assert "STE 3.5" not in rules("The loading plate is thick.", terms={"technical_names": ["loading plate"]})


def test_modal_verbs():
    for w in ("should", "may", "might", "could", "would", "shall"):
        assert "STE 1.1" in rules(f"The valve {w} open."), w
    for w in ("must", "can", "will"):
        assert "STE 1.1" not in rules(f"The valve {w} open."), w
    issue = [i for i in ste.check("The valve may open.") if i.text == "may"][0]
    assert issue.severity == "error" and issue.suggestion == "can"


def test_noun_cluster_is_a_warning():
    found = [i for i in ste.check("The stress strain load data table is large.") if i.rule == "STE 2.1"]
    assert found and all(i.severity == "warning" for i in found)
    assert "STE 2.1" not in rules("The table of the load data is large.")


def test_latin_abbreviations():
    for s in ("The tool, e.g. a gauge, measures it.", "The tool, i.e. a gauge, measures it.", "It has a gauge, a pump, etc."):
        assert "STE GR-6" in rules(s), s
    assert "STE GR-6" not in rules("The tool, for example a gauge, measures it.")


def test_and_or_and_semicolon():
    assert '"and/or"' in names("Use A and/or B.")
    assert '"and/or"' not in names("Use A or B.")
    assert "STE 8.1" in rules("Close the valve; open the drain.")
    assert "STE 8.1" not in rules("Close the valve. Open the drain.")


def test_sentence_without_verb_is_a_warning():
    found = [i for i in ste.check("The valve body at the inlet.") if i.name == "sentence without a verb"]
    assert found and found[0].severity == "warning"
    assert not [i for i in ste.check("The valve body is at the inlet.") if i.name == "sentence without a verb"]


def test_several_instructions_in_one_sentence():
    found = [i for i in ste.check("Close the valve and open the drain and check the gauge.", "procedure") if i.rule == "STE 5.2"]
    assert found and found[0].severity == "warning"
    assert "STE 5.2" not in rules("Close the valve.", "procedure")


def test_not_approved_word_list():
    found = [i for i in ste.check("Ensure that the gauge is correct.") if i.rule == "STE 1.1"]
    assert found and found[0].severity == "warning"      # a word list from public guidance, not the dictionary
    assert "STE 1.1" not in rules("Make sure that the gauge is correct.")


# ---- the contract text, identifiers, numbers, technical names ---------------------------------------------

def test_contract_good_and_bad_examples():
    good = ("The requirement gives an MEOP of 10.5 MPa. The system context gives 10.72 MPa at the inlet during "
            "shutdown. The requirement value is too low.")
    assert ste.check(good, terms=TERMS) == []
    bad = ste.check("MEOP may be insufficient considering surge effects.", terms=TERMS)
    assert {"STE 1.1", "STE 3.5"} <= {i.rule for i in bad}


def test_identifiers_numbers_units_not_flagged():
    text = ("REQ-MOV-001 and LC-FIRE give 10.72 MPa at 90 K. The mesh has 25 000 nodes. "
            "The key is von_mises.peak@part and f44 is a face.")
    assert [i for i in ste.check(text, terms=TERMS) if i.severity == "error"] == []
    assert ste.count_words("The load is 10.72 MPa.") == 4        # a number with its unit is one word (STE 8.6)


def test_technical_names_and_verbs_accepted():
    assert "STE 3.5" not in rules("The housing is thick.", terms=TERMS)
    assert not [i for i in ste.check("The solver must mesh the valve body.", terms=TERMS) if i.severity == "error"]
    assert not [i for i in ste.check("Pressurize the valve body.", "procedure", TERMS) if i.severity == "error"]


def test_not_approved_words_of_the_project():
    terms = {"not_approved": {"loading scenario": "load case"}}
    assert [i for i in ste.check("Each loading scenario has a name.", terms=terms)]


# ---- word count rules ------------------------------------------------------------------------------------

def test_word_count_rules():
    assert ste.count_words("The pressure is 10.72 MPa.") == 4                 # STE 8.6 number + unit
    assert ste.count_words("The part is re-entrant-safe.") == 4               # STE 8.7 hyphens
    assert ste.count_words("The valve (see the figure on the next page) is open.") == 5   # STE 8.5 parentheses
    assert ste.count_words('The text says "The body must be stiff".') == 4     # quoted text
    two = ste.split_sentences("The valve is open. The pump is on.")
    assert two == ["The valve is open.", "The pump is on."]
    # the colon before a vertical list ends the sentence (STE 8.4)
    items = ste.split_sentences("The load case has these loads:\n- a pressure of 10 MPa\n- a force of 5 N")
    assert len(items) >= 2


# ---- the user's dictionary file --------------------------------------------------------------------------

@pytest.fixture
def dict_csv(tmp_path):
    p = tmp_path / "dict.csv"
    p.write_text("word,part_of_speech,status,alternative\n"
                 "valve,noun,approved,\nopen,verb,approved,\nthe,article,approved,\nis,verb,approved,\n"
                 "close,verb,approved,\nand,conjunction,approved,\nat,preposition,approved,\nleak,verb,approved,\n"
                 "leakage,noun,not_approved,leak\n", encoding="utf-8")
    return p


def test_dictionary_file(dict_csv):
    terms = {"dictionary": str(dict_csv)}
    flagged = ste.check("The leakage is large.", terms=terms)
    assert any(i.text.lower() == "leakage" and i.suggestion == "leak" for i in flagged)
    unknown = ste.check("The valve is gigantic.", terms=terms)
    assert any("gigantic" in i.text for i in unknown)
    assert not [i for i in ste.check("The valve is open.", terms=terms) if i.rule == "STE 1.1"]
    # technical names stay accepted, numbers and identifiers are not unknown words
    t2 = dict(terms, technical_names=["seat pocket"])
    assert not [i for i in ste.check("The seat pocket is open at REQ-MOV-001 and 5 mm.", terms=t2) if i.rule == "STE 1.1"]


def test_dictionary_by_environment(dict_csv, monkeypatch):
    monkeypatch.setenv("DOSSIER_STE_DICTIONARY", str(dict_csv))
    assert any("leakage" in i.text.lower() for i in ste.check("The leakage is large."))


def test_dictionary_errors(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("x,y\n1,2\n", encoding="utf-8")
    with pytest.raises(ste.DictionaryError):
        ste.check("The valve is open.", terms={"dictionary": str(bad)})


# ---- summarize -------------------------------------------------------------------------------------------

def test_summarize():
    texts = {"A.statement": "The valve is open.", "B.statement": "The valve may open.",
             "C.suggestion": "Close the valve; open the drain.", "D.rationale": ""}
    s = ste.summarize(texts, kind=ste.field_kind)
    assert s["checked"] == 3 and s["clean"] == 1 and s["errors"] >= 2
    assert {x["id"] for x in s["issues"]} == {"B.statement", "C.suggestion"}
    assert "STE 1.1" in s["by_rule"] and s["limits"] and json.dumps(s)
    assert any("partial" in x for x in s["limits"])


def test_summarize_limits_with_dictionary(dict_csv):
    s = ste.summarize({"A": "The valve is open."}, terms={"dictionary": str(dict_csv)})
    assert not any("partial" in x for x in s["limits"]) and s["dictionary"]


def test_bad_kind():
    with pytest.raises(ValueError):
        ste.summarize({"A": "x"}, kind="poem")


def test_texts_of_and_field_kind():
    rec = {"id": "FND-1", "statement": "The valve is open.",
           "alternatives": [{"choice": "A", "why_not": "It is slow."}], "suggestion": "Close the valve."}
    t = ste.texts_of([rec])
    assert t["FND-1.statement"] == "The valve is open." and "FND-1.alternatives[0].why_not" in t
    assert ste.field_kind("FND-1.suggestion") == "procedure" and ste.field_kind("FND-1.statement") == "descriptive"


# ---- helpers ---------------------------------------------------------------------------------------------

def test_helpers():
    assert ste.listing(["A", "B", "C"]) == "A, B and C"
    assert ste.listing(["A", "B"]) == "A and B" and ste.listing(["A"]) == "A" and ste.listing([]) == ""
    assert ste.listing(["A", "B", "C"], "or") == "A, B or C"
    assert ste.plural(1, "load case") == "1 load case" and ste.plural(3, "load case") == "3 load cases"
    assert ste.plural(2, "body") == "2 bodies" and ste.plural(0, "load case") == "0 load cases"
    assert ste.a("valve") == "a valve" and ste.a("MEOP") == "an MEOP" and ste.a("hour") == "an hour"
    assert ste.sentence("the valve is open") == "The valve is open."
    assert ste.sentence("The valve is open.") == "The valve is open."
    assert ste.sentence("f44 is a face") == "f44 is a face."
    assert ste.quote('say "x"') == "\"say 'x'\""


def test_paragraph_splits_at_six():
    p = ste.paragraph(*["The valve is open."] * 8)
    assert "\n\n" in p and not rules(p)


def test_too_long_and_fit():
    long = "The valve body has a shell and a neck and a flange and a seat pocket and a stem bore and a cavity and an actuator and a pump and more."
    assert ste.too_long("The valve is open. " + long)[0][0] == long
    assert ste.too_long("The valve is open.") == []
    assert ste.fit(long, "The valve body has many parts.") == "The valve body has many parts."
    assert ste.fit("The valve is open.", "x") == "The valve is open."


# ---- the rule table and the CLI ---------------------------------------------------------------------------

def test_rule_table():
    for r in ste.RULES.values():
        assert r.rule.startswith("STE ") and r.how in ("exact", "word list", "heuristic")
        assert all(k in ste.SOURCES for k in r.sources)
    assert ste.RULES["length_procedure"].rule == "STE 5.1" and ste.RULES["length_descriptive"].rule == "STE 6.3"
    assert ste.NOT_CHECKED


def test_cli(tmp_path, capsys):
    assert cli.main(["MEOP may be insufficient."]) == 1
    assert cli.main(["The valve is open."]) == 0
    assert cli.main(["--rules"]) == 0 and "STE 5.1" in capsys.readouterr().out
    f = tmp_path / "r.json"
    f.write_text(json.dumps([{"id": "X", "statement": "The valve may open."}]), encoding="utf-8")
    assert cli.main([str(f), "--json"]) == 1
    py = tmp_path / "m.py"
    py.write_text('x = "The valve should open at the inlet during the test."\n', encoding="utf-8")
    assert cli.main([str(tmp_path)]) == 1


# ---- the checker over the repository's own text ------------------------------------------------------------

def test_contracts_rules_text_is_examined():
    """The STE rule list of CONTRACTS.md is itself text; the checker runs over it without an error."""
    from pathlib import Path
    src = (Path(__file__).resolve().parents[2] / "dossier" / "CONTRACTS.md").read_text(encoding="utf-8")
    block = src.split("## Text: ASD-STE100", 1)[1].split("## Tests", 1)[0]
    s = ste.summarize({"contracts": block})
    assert s["checked"] == 1 and "standard" in s
