"""The STE check of the report: every text field of every derived record, and every sentence the report writes.

`dossier.ste` is the checker (another engine). When it is installed, `run` gives the numbers of section 11:
how many texts were checked, how many are clean, the issues by rule, the records with issues, and the limits
the checker states about itself. When it is not installed, `run` says so and the report says so.

Source text (requirement text, descriptions in the input files) is not checked: a person wrote it.
"""

from __future__ import annotations

from typing import Any

from dossier.report.data import Data

PROCEDURE_FIELDS = ("suggestion", "verify_by")


def record_texts(d: Data) -> dict[str, tuple[str, str, str]]:
    """{text id: (record id, field, text)} for every text an engine wrote."""
    out: dict[str, tuple[str, str, str]] = {}

    def add(rid: str, name: str, text: Any) -> None:
        if isinstance(text, str) and text.strip():
            out[f"{rid}.{name}"] = (rid, name, text)

    for f in d.findings:
        for name in ("title", "statement", "rationale", "suggestion"):
            add(f.id, name, getattr(f, name))
    for x in d.decisions:
        for name in ("title", "choice", "rationale"):
            add(x.id, name, getattr(x, name))
        for k, a in enumerate(x.alternatives):
            add(x.id, f"alternatives[{k}].option", a.option)
            add(x.id, f"alternatives[{k}].why_not", a.why_not)
    for x in d.assumptions:
        for name in ("title", "statement", "rationale", "impact", "verify_by"):
            add(x.id, name, getattr(x, name))
    for lc in d.load_cases:
        for name in ("title", "statement", "note"):
            add(lc.id, name, getattr(lc, name))
    for h in d.hand_calcs:
        add(h.id, "title", h.title)
        add(h.id, "reference", h.reference)
        for k, s in enumerate(h.steps):
            add(h.id, f"steps[{k}].text", s.text)
        for k, s in enumerate(h.assumptions):
            add(h.id, f"assumptions[{k}]", s)
        for k, s in enumerate(h.warnings):
            add(h.id, f"warnings[{k}]", s)
        for o in h.outputs:
            add(h.id, f"outputs[{o.name}].description", o.description)
    for a in d.analyses:
        for name in ("title", "statement", "message"):
            add(a.id, name, getattr(a, name))
        for k, b in enumerate(a.bcs):
            add(a.id, f"bcs[{k}].description", b.description)
        for c in a.checks:
            add(a.id, f"checks[{c.id}].statement", c.statement)
        for key, r in a.results.items():
            add(a.id, f"results[{key}].note", r.note)
        for k, art in enumerate(a.artifacts):
            add(a.id, f"artifacts[{k}].caption", art.caption)
    for i in d.compliance:
        for name in ("title", "statement", "factor_source", "limit_source"):
            add(i.id, name, getattr(i, name))
        for k, s in enumerate(i.reasons):
            add(i.id, f"reasons[{k}]", s)
    for rid, v in d.verdicts.items():
        add(rid, "verdict.statement", v.statement)
    return out


def _kind_of(field: str) -> str:
    leaf = field.rsplit(".", 1)[-1].split("[")[0]
    return "procedure" if leaf in PROCEDURE_FIELDS else "descriptive"


def run(d: Data, own: list[tuple[str, str, str]]) -> dict:
    """The STE check of the record texts and of the report's own sentences.

    own: the register of the report's text, (kind, text, section); only full sentences are checked.
    Returns {"ran": False, "why": ...} when the checker is not there, else the numbers for section 11 and the
    two summaries of dossier.ste.summarize under "records" and "report"."""
    try:
        import dossier.ste as ste
        summarize = ste.summarize
    except Exception as e:  # noqa: BLE001  the checker is another engine: it can be missing or unfinished
        return {"ran": False, "why": f"The module dossier.ste is not available ({type(e).__name__})."}
    try:
        terms = ste.project_terms(d.project) if hasattr(ste, "project_terms") else (d.project.terms or None)
    except Exception:  # noqa: BLE001
        terms = d.project.terms or None
    texts = record_texts(d)
    kinds = {k: _kind_of(v[1]) for k, v in texts.items()}
    sentences = {f"report[{n}]": text for n, (kind, text, _) in enumerate(own) if kind == "sentence"}
    try:
        rec = summarize({k: v[2] for k, v in texts.items()}, kinds, terms)
        rep = summarize(sentences, "descriptive", terms)
    except Exception as e:  # noqa: BLE001
        return {"ran": False, "why": f"The STE checker stopped with an error ({type(e).__name__}: {e})."}

    def n(s: dict, key: str, alt: str = "") -> int:
        return int(s.get(key, s.get(alt, 0)) or 0)

    def n_issues(s: dict) -> int:
        return sum(len(x.get("issues", [])) for x in s.get("issues", []))

    by_rule: dict[str, int] = {}
    for s in (rec, rep):
        for rule, v in (s.get("by_rule") or {}).items():
            by_rule[rule] = by_rule.get(rule, 0) + int(v.get("count", 0) if isinstance(v, dict) else v)
    by_record: dict[str, list[dict]] = {}
    for x in rec.get("issues", []):
        rid, field, _ = texts.get(x["id"], (x["id"], "", ""))
        for i in x.get("issues", []):
            by_record.setdefault(rid, []).append({"rule": i.get("rule", ""), "message": i.get("message", ""),
                                                  "text": i.get("text", ""), "field": field, "severity": i.get("severity", "")})
    own_issues = []
    for x in rep.get("issues", []):
        own_issues.append({"text": sentences.get(x["id"], ""), "issues": [{"rule": i.get("rule", ""), "message": i.get("message", ""),
                                                                           "text": i.get("text", "")} for i in x.get("issues", [])]})
    limits = list(rec.get("limits") or [])
    for x in rec.get("not_checked") or []:
        if isinstance(x, dict) and x.get("what"):
            limits.append(f"The checker does not examine {x.get('rule', 'a rule')}: {x['what']}.")
    return {
        "ran": True,
        "standard": rec.get("standard", ""),
        "texts": n(rec, "checked") + n(rep, "checked"),
        "clean": n(rec, "clean") + n(rep, "clean"),
        "errors": n(rec, "errors") + n(rep, "errors"),
        "warnings": n(rec, "warnings") + n(rep, "warnings"),
        "groups": [{"label": "Text of the records", "texts": n(rec, "checked"), "clean": n(rec, "clean"), "issues": n_issues(rec)},
                   {"label": "Sentences of this report", "texts": n(rep, "checked"), "clean": n(rep, "clean"), "issues": n_issues(rep)}],
        "by_rule": by_rule,
        "records": [{"id": rid, "issues": issues} for rid, issues in by_record.items()],
        "own": own_issues,
        "limits": limits,
        "summary": {"records": rec, "report": rep},
    }
