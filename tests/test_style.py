"""Style gate: every function in src/feagent is short (user requirement)."""
import ast
from pathlib import Path

import pytest

MAX_LINES = 25
TEMPORARY_EXEMPT: set[str] = set()
SRC = Path(__file__).resolve().parents[1] / "src" / "feagent"
FILES = sorted(p for p in SRC.glob("*.py") if p.name not in TEMPORARY_EXEMPT)


def _docstring_lines(node: ast.FunctionDef | ast.AsyncFunctionDef) -> int:
    first = node.body[0] if node.body else None
    if (isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)):
        return first.end_lineno - first.lineno + 1
    return 0


def function_lengths(path: Path) -> list[tuple[str, int]]:
    tree = ast.parse(path.read_text(), filename=str(path))
    return [(f"{path.name}:{n.lineno} {n.name}",
             n.end_lineno - n.lineno + 1 - _docstring_lines(n))
            for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.name)
def test_functions_are_short(path):
    long = [(name, n) for name, n in function_lengths(path) if n > MAX_LINES]
    assert not long, f"functions over {MAX_LINES} lines: {long}"


def test_docstring_lines_are_excluded(tmp_path):
    body = "\n".join(f"    x{i} = {i}" for i in range(MAX_LINES - 1))
    src = tmp_path / "m.py"
    src.write_text(f'def f():\n    """Doc.\n\n    More.\n    """\n{body}\n')
    assert function_lengths(src) == [("m.py:1 f", MAX_LINES)]


def test_nested_functions_are_measured(tmp_path):
    src = tmp_path / "m.py"
    src.write_text("class C:\n    def m(self):\n        def inner():\n            pass\n")
    assert sorted(n for _, n in function_lengths(src)) == [2, 3]
