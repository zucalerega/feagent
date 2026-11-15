"""Formulas as text: one expression gives the number and the LaTeX that the report prints.

A model writes each step of its calculation as an arithmetic expression over the names of its quantities,
for example ``p * r_i / (r_o - r_i)``. `evaluate` computes the number, and `latex` prints the same expression
tree. The formula that the report shows is then the formula that made the number: the two cannot differ.

The language is Python arithmetic on numbers (SI): ``+ - * / **``, parentheses, the functions in FUNCTIONS and
the constant ``pi``. ``x ** 0.5`` and ``sqrt(x)`` print as a root, ``a / b`` prints as a fraction, ``log`` is
the natural logarithm. The LaTeX has no delimiters and uses only commands that MathJax, KaTeX and the
matplotlib mathtext know (frac, sqrt, left, right, the Greek letters, mathrm, ln, log, exp, sinh, cosh, tanh).
"""

from __future__ import annotations

import ast
import math
from functools import lru_cache
from typing import Callable, Mapping

FUNCTIONS: dict[str, Callable] = {
    "sqrt": math.sqrt, "exp": math.exp, "log": math.log, "log10": math.log10,
    "sinh": math.sinh, "cosh": math.cosh, "tanh": math.tanh, "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "abs": abs, "min": min, "max": max,
}
CONSTANTS = {"pi": math.pi}

_NODES = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Call, ast.Name, ast.Load, ast.Constant,
          ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.USub)

GREEK = {"alpha", "beta", "gamma", "delta", "epsilon", "varepsilon", "zeta", "eta", "theta", "kappa", "lambda",
         "mu", "nu", "xi", "rho", "sigma", "tau", "phi", "chi", "psi", "omega",
         "Gamma", "Delta", "Theta", "Lambda", "Sigma", "Phi", "Psi", "Omega"}


class FormulaError(ValueError):
    """A formula that cannot be read or evaluated (a programming error of a model, or a number outside the
    domain of a function)."""


@lru_cache(maxsize=None)
def _parse(expr: str) -> ast.Expression:
    try:
        tree = ast.parse(expr.strip(), mode="eval")
    except SyntaxError as e:
        raise FormulaError(f"cannot read the formula {expr!r}: {e.msg}") from e
    for node in ast.walk(tree):
        if not isinstance(node, _NODES):
            raise FormulaError(f"the formula {expr!r} uses {type(node).__name__}, which is not allowed")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in FUNCTIONS or node.keywords:
                raise FormulaError(f"the formula {expr!r} calls a function that is not one of: "
                                   f"{', '.join(sorted(FUNCTIONS))}")
        if isinstance(node, ast.Constant) and (isinstance(node.value, bool) or not isinstance(node.value, (int, float))):
            raise FormulaError(f"the formula {expr!r} has a constant that is not a number")
    return tree


@lru_cache(maxsize=None)
def _code(expr: str):
    return compile(_parse(expr), "<formula>", "eval")


@lru_cache(maxsize=None)
def names(expr: str) -> tuple[str, ...]:
    """The quantities a formula reads, in the order of their first appearance (not functions, not pi)."""
    tree = _parse(expr)
    called = {id(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)}
    out: list[str] = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Name) and id(n) not in called and n.id not in CONSTANTS and n.id not in out:
            out.append(n.id)
    return tuple(out)


def evaluate(expr: str, values: Mapping[str, float]) -> float:
    """The number a formula gives for the values of its quantities."""
    env: dict = {**FUNCTIONS, **CONSTANTS}
    for n in names(expr):
        if n not in values or values[n] is None:
            raise FormulaError(f"the formula {expr!r} reads {n!r}, which has no value")
        env[n] = values[n]
    try:
        out = float(eval(_code(expr), {"__builtins__": {}}, env))  # noqa: S307  AST-whitelisted, written by a model
    except ZeroDivisionError as e:
        raise FormulaError(f"the formula {expr!r} divides by zero") from e
    except (ValueError, OverflowError) as e:
        raise FormulaError(f"the formula {expr!r} has no value for these numbers ({e})") from e
    if not math.isfinite(out):
        raise FormulaError(f"the formula {expr!r} gives no finite number")
    return out


# --------------------------------------------------------------------------- LaTeX


def auto_symbol(name: str) -> str:
    """A LaTeX symbol from a plain name: r_i -> r_{i}, sigma_a -> \\sigma_{a}, T_base -> T_{\\mathrm{base}}."""
    def part(s: str) -> str:
        if s in GREEK:
            return "\\" + s
        return s if len(s) == 1 or s.isdigit() else rf"\mathrm{{{s}}}"

    head, *rest = name.split("_")
    return part(head) + ("_{" + ",".join(part(s) for s in rest) + "}" if rest else "")


def _number(v) -> str:
    if isinstance(v, int) or float(v).is_integer():
        return str(int(v))
    s = repr(float(v))
    if "e" in s:
        m, e = s.split("e")
        return rf"{m} \cdot 10^{{{int(e)}}}"
    return s


_ADD, _MUL, _NEG, _POW, _ATOM = 1, 2, 3, 4, 5
_CALLS = {"log": r"\ln", "log10": r"\log_{10}", "exp": r"\exp", "sinh": r"\sinh", "cosh": r"\cosh", "tanh": r"\tanh",
          "sin": r"\sin", "cos": r"\cos", "tan": r"\tan", "min": r"\min", "max": r"\max"}


def _paren(s: str) -> str:
    return rf"\left({s}\right)"


def _tex(n: ast.AST, sym: Callable[[str], str]) -> tuple[str, int]:
    """(LaTeX, precedence) of a node. A fraction counts as a product: it needs no parentheses as a factor
    or after a minus sign, and it gets them as the base of a power."""
    if isinstance(n, ast.Constant):
        return _number(n.value), _ATOM
    if isinstance(n, ast.Name):
        if n.id == "pi":
            return r"\pi", _ATOM
        s = sym(n.id)
        return s, (_ATOM if " " not in s and "^" not in s else _MUL)
    if isinstance(n, ast.UnaryOp):
        s, p = _tex(n.operand, sym)
        return "-" + (_paren(s) if p in (_ADD, _NEG) else s), _NEG
    if isinstance(n, ast.Call):
        args = [_tex(a, sym)[0] for a in n.args]
        f = n.func.id
        if f == "sqrt":
            return rf"\sqrt{{{args[0]}}}", _POW
        if f == "abs":
            return rf"\left|{args[0]}\right|", _ATOM
        return rf"{_CALLS[f]}\left({', '.join(args)}\right)", _POW
    left, pl = _tex(n.left, sym)
    right, pr = _tex(n.right, sym)
    if isinstance(n.op, ast.Div):
        return rf"\frac{{{left}}}{{{right}}}", _MUL
    if isinstance(n.op, ast.Pow):
        if isinstance(n.right, ast.Constant) and n.right.value == 0.5:
            return rf"\sqrt{{{left}}}", _POW
        return (left if pl == _ATOM else _paren(left)) + "^{" + right + "}", _POW
    if isinstance(n.op, ast.Mult):
        if pl == _ADD:
            left = _paren(left)
        if pr in (_ADD, _NEG):
            right = _paren(right)
        return left + (r" \cdot " if right[:1].isdigit() else r"\,") + right, _MUL
    if pr == _NEG or (isinstance(n.op, ast.Sub) and pr == _ADD):
        right = _paren(right)
    return f"{left} {'+' if isinstance(n.op, ast.Add) else '-'} {right}", _ADD


def latex(expr: str, symbols: Mapping[str, str] | Callable[[str], str] | None = None) -> str:
    """The formula as LaTeX. `symbols` gives the LaTeX symbol of a name (a mapping or a function); a name
    it does not know gets `auto_symbol`."""
    if callable(symbols):
        sym = symbols
    else:
        table = dict(symbols or {})

        def sym(name: str) -> str:
            return table.get(name) or auto_symbol(name)

    return _tex(_parse(expr).body, sym)[0]
