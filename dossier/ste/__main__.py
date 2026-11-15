"""Command line of the STE checker.

    python -m dossier.ste "Text to examine."          check one text
    python -m dossier.ste notes.txt                   check a text file (descriptive text)
    python -m dossier.ste out/validate.json           check the text fields of a JSON file (records)
    python -m dossier.ste dossier/loadcases           check the long strings in the Python files of a folder
    python -m dossier.ste --rules                     list the rules, how each is checked and what is not checked
    options: --kind descriptive|procedure, --terms terms.yaml, --dictionary words.csv, --json

The exit code is 1 when a text has an error, and 0 when it has only warnings or nothing.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

from dossier import ste
from dossier.ste.rules import NOT_CHECKED, RULES


def _python_strings(path: Path) -> dict[str, str]:
    """The string literals of a Python file that look like sentences (docstrings and comments are left out)."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError):
        return {}
    skip: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant):
                skip.add(id(body[0].value))
    out: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in skip:
            s = " ".join(node.value.split())
            if len(s.split()) >= 5 and s[0].isupper() and s.endswith((".", "?", "!")) and "{" not in s:
                out[f"{path}:{node.lineno}"] = s
    return out


def _collect(arg: str) -> dict[str, str]:
    p = Path(arg)
    if p.is_dir():
        out: dict[str, str] = {}
        for f in sorted(p.rglob("*.py")):
            out.update(_python_strings(f))
        return out
    if p.is_file():
        if p.suffix == ".json":
            data = json.loads(p.read_text(encoding="utf-8"))
            recs = data if isinstance(data, list) else [data]
            return ste.texts_of(recs)
        if p.suffix == ".py":
            return _python_strings(p)
        return {str(p): p.read_text(encoding="utf-8")}
    return {"text": arg}


def _print_rules() -> None:
    print(f"{ste.STANDARD}: the rules of this checker\n")
    for r in RULES.values():
        print(f"{r.rule:<10} {r.name:<28} {r.how:<10} {r.summary}")
    print("\nNot examined:")
    for rule, what in NOT_CHECKED:
        print(f"  {rule:<16} {what}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m dossier.ste", description="ASD-STE100 checker")
    ap.add_argument("target", nargs="?", help="text, a file or a folder")
    ap.add_argument("--rules", action="store_true", help="list the rules")
    ap.add_argument("--kind", choices=ste.KINDS, default="descriptive")
    ap.add_argument("--terms", help="terms.yaml of the project")
    ap.add_argument("--dictionary", help="CSV dictionary file (word, part_of_speech, status, alternative)")
    ap.add_argument("--json", action="store_true", help="write the summary as JSON")
    a = ap.parse_args(argv)
    if a.rules:
        _print_rules()
        return 0
    if not a.target:
        ap.error("give a text, a file or a folder (or --rules)")
    terms: dict = {}
    if a.terms:
        import yaml
        terms = yaml.safe_load(Path(a.terms).read_text(encoding="utf-8")) or {}
    if a.dictionary:
        terms["dictionary"] = a.dictionary
    texts = _collect(a.target)
    kind = ste.field_kind if a.target.endswith(".json") else a.kind
    summary = ste.summarize(texts, kind=kind, terms=terms or None)
    if a.json:
        print(json.dumps(summary, indent=2))
    else:
        for item in summary["issues"]:
            print(f"{item['id']}")
            for i in item["issues"]:
                tag = "ERROR" if i["severity"] == "error" else "warn "
                sug = f" Use: {i['suggestion']}" if i["suggestion"] else ""
                print(f"  {tag} {i['rule']:<9} {i['name']}: {i['message']} [{i['text']}]{sug}")
        print(f"\nchecked {summary['checked']}, clean {summary['clean']}, errors {summary['errors']}, "
              f"warnings {summary['warnings']}")
        for line in summary["limits"]:
            print(f"  limit: {line}")
    return 1 if summary["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
