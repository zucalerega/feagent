"""Quantities with units, and expressions of named quantities.

Engineers write "10.4 MPa", "25 mm", "90 K"; the solver wants SI numbers. Every number in a project
is a quantity with a unit (pint), every derived number is an expression of named quantities, and the
dimension of a result is checked against the unit it is declared in. A unit error is then an error
message, not a wrong margin.

Conventions
-----------
* SI for computation: ``si(q)`` is the magnitude in base SI units (m, kg, s, K, and what follows:
  Pa, N, J, W). Temperatures are absolute (K); "20 degC" converts to 293.15 K.
* A literal is a number with a unit: ``"10.4 MPa"``, ``"1141 kg/m^3"``, ``"100 ms"``, ``"1.5"``
  (dimensionless), ``"8 g0"`` (standard gravity, 9.80665 m/s^2).
* An expression is Python arithmetic over names: ``"p_pump_max + dp_surge"``,
  ``"mdot / (rho * pi * d**2 / 4)"``. Names are project parameters; a name that is not a parameter is
  read as a unit (``"0.5 * m"``). Functions: sqrt, abs, min, max, exp, log, log10, sin, cos, tan,
  plus the callables the caller adds (``dossier.context`` adds ``load()``, ``feat()``). ``pi`` is the number.
  Subscripts pick a vector component: ``f[1]``.
* Expressions are parsed with an AST whitelist (no attribute access, no imports, no comprehensions).
"""

from __future__ import annotations

import ast
import math
import re
from typing import Any, Callable, Iterable

import pint

ureg = pint.UnitRegistry(autoconvert_offset_to_baseunit=True)
ureg.formatter.default_format = "~P"
Q = ureg.Quantity


class UnitsError(ValueError):
    """A quantity or expression that cannot be read, or whose dimension is wrong."""


# --------------------------------------------------------------------------- literals


def parse(text: Any) -> pint.Quantity:
    """A literal -> quantity: "10.4 MPa", 3.2 (dimensionless), {"value": 5, "unit": "mm"},
    {"value": [0, 0, -450], "unit": "N"} (a vector)."""
    if isinstance(text, pint.Quantity):
        return text
    if isinstance(text, bool):
        raise UnitsError(f"not a quantity: {text!r}")
    if isinstance(text, (int, float)):
        return Q(float(text))
    if isinstance(text, (list, tuple)):
        return Q([float(v) for v in text])
    if isinstance(text, dict):
        v = text.get("value")
        if isinstance(v, str):
            q = parse(v)
            return q if not text.get("unit") else q.to(text["unit"])
        unit = text.get("unit") or ""
        try:
            return Q([float(x) for x in v] if isinstance(v, (list, tuple)) else float(v), unit)
        except Exception as e:  # noqa: BLE001
            raise UnitsError(f"cannot read quantity {text!r}: {e}") from e
    if not isinstance(text, str) or not text.strip():
        raise UnitsError(f"not a quantity: {text!r}")
    s = text.strip()
    try:
        q = Q(_per_degree(s))
    except Exception as e:  # noqa: BLE001  pint raises several kinds
        raise UnitsError(f"cannot read quantity {text!r}: {e}") from e
    if not isinstance(q.magnitude, (int, float)):
        raise UnitsError(f"cannot read quantity {text!r}: it has no number")
    return q


_DEGREE = re.compile(r"(?<![A-Za-z_])(degC|degF|celsius|fahrenheit|°C|°F)(?![A-Za-z_])")
_DELTA = {"degC": "delta_degC", "celsius": "delta_degC", "°C": "delta_degC",
          "degF": "delta_degF", "fahrenheit": "delta_degF", "°F": "delta_degF"}


def _per_degree(text: str) -> str:
    """In a derived unit a degree is a temperature difference: "13e-6 1/degC" is 13e-6 per kelvin, and
    "15 W/(m*degC)" is 15 W/(m K). pint reads the degree there as an absolute temperature and gets a value
    that is wrong by a factor of hundreds, so the degree is written as a difference before pint reads it.
    A plain temperature ("20 degC") stays an absolute temperature."""
    if not _DEGREE.search(text):
        return text
    parts = text.strip().split(None, 1)
    unit = parts[1].strip() if len(parts) == 2 else ""
    if _DEGREE.fullmatch(unit) or not unit:
        return text
    return parts[0] + " " + _DEGREE.sub(lambda m: _DELTA[m.group(1)], unit)


def is_vector(q: Any) -> bool:
    m = q.magnitude if isinstance(q, pint.Quantity) else q
    return getattr(m, "ndim", 0) > 0 or isinstance(m, (list, tuple))


def _components(q: pint.Quantity) -> list[float]:
    m = q.magnitude
    return [float(x) for x in m.tolist()] if is_vector(q) else [float(m)]


def si(q: pint.Quantity) -> float | list[float]:
    """Magnitude in base SI units; a vector quantity gives a list."""
    b = q.to_base_units()
    return _components(b) if is_vector(b) else float(b.magnitude)


def _is_energy(q: pint.Quantity) -> bool:
    """Energy and torque share a dimension; a quantity written in J / Wh / cal is an energy."""
    return any(k in str(u) for u in q.units._units for k in ("joule", "watt_hour", "calorie", "electron_volt",
                                                              "british_thermal_unit"))


def si_unit(q: pint.Quantity) -> str:
    """The SI unit a quantity is computed in, written the usual way (Pa, N, J, W, not kg/m/s^2)."""
    d = q.dimensionality
    if d == ureg("J").dimensionality:
        return "J" if _is_energy(q) else "N*m"
    for name in _PREFERRED:
        if ureg(name).dimensionality == d:
            return name
    return f"{q.to_base_units().units:~P}" or ""


_PREFERRED = ("Pa", "N", "W", "m", "kg", "s", "K", "m/s", "m/s^2", "kg/s", "kg/m^3", "W/m^2",
              "W/(m*K)", "W/(m^2*K)", "J/(kg*K)", "Pa*s", "m^2", "m^3", "1/K", "Hz", "m^3/s", "")


def compatible(q: pint.Quantity, unit: str) -> bool:
    try:
        return q.check(ureg(unit).dimensionality) if unit else q.dimensionless
    except Exception:  # noqa: BLE001
        return False


def to(q: pint.Quantity, unit: str) -> pint.Quantity:
    """Convert, with a readable error when the dimension is wrong."""
    try:
        return q.to(unit) if unit else q.to("")
    except Exception as e:  # noqa: BLE001
        raise UnitsError(f"{fmt(q)} is not a quantity in {unit or 'no unit (a pure number)'}: {e}") from e


def _sig(x: float, n: int) -> str:
    """x with n significant digits, in plain notation between 1e-4 and 1e7 (15570, 0.00012), else 1.2e-9."""
    if x == 0 or not math.isfinite(x):
        return "0" if x == 0 else str(x)
    e = math.floor(math.log10(abs(x)))
    if -4 <= e < 7:
        r = round(x, n - 1 - e)
        s = f"{r:.{max(n - 1 - e, 0)}f}"
        return s.rstrip("0").rstrip(".") if "." in s else s
    m = f"{x / 10 ** e:.{n - 1}f}".rstrip("0").rstrip(".")
    return f"{m}e{e}"


# How the units people read are written (pint sorts factors alphabetically: "m*N", "W/K/m^2").
_LABEL = {"N*m": "N·m", "kN*m": "kN·m", "W/(m^2*K)": "W/(m²·K)", "W/(m*K)": "W/(m·K)", "J/(kg*K)": "J/(kg·K)",
          "kg/m^3": "kg/m³", "m/s^2": "m/s²", "Pa*s": "Pa·s", "mm^2": "mm²", "m^2": "m²", "mm^3": "mm³", "m^3": "m³",
          "W/m^2": "W/m²", "m^3/s": "m³/s", "um": "µm"}


def fmt(q: pint.Quantity, unit: str | None = None, sig: int = 4) -> str:
    """A quantity for people: "10.72 MPa", "[0, 686.5, 0] N", "1.5" (dimensionless)."""
    if unit is not None:
        q = to(q, unit)
    u = _LABEL.get(unit) if unit in _LABEL else f"{q.units:~P}"
    c = _components(q)
    body = ("[" + ", ".join(_sig(x, sig) for x in c) + "]") if is_vector(q) else _sig(c[0], sig)
    return f"{body} {u}".strip()


# Units an engineer expects to read for each kind of quantity (first that keeps the number readable).
_DISPLAY = [
    ("Pa", ("MPa", "kPa", "Pa", "GPa")), ("N", ("N", "kN")), ("m", ("mm", "m", "um")),
    ("K", ("K",)), ("kg", ("kg", "g")), ("s", ("s", "ms")), ("kg/s", ("kg/s",)), ("m/s", ("m/s",)),
    ("m/s^2", ("m/s^2",)), ("kg/m^3", ("kg/m^3",)), ("W/(m^2*K)", ("W/(m^2*K)",)), ("W/(m*K)", ("W/(m*K)",)),
    ("W", ("W", "kW")), ("m^2", ("mm^2", "m^2")), ("m^3", ("mm^3", "m^3")), ("Pa*s", ("Pa*s",)),
    ("J/(kg*K)", ("J/(kg*K)",)), ("1/K", ("1/K",)), ("W/m^2", ("W/m^2",)), ("m^3/s", ("m^3/s",)), ("", ("",)),
]


def display(q: pint.Quantity, sig: int = 4) -> str:
    """fmt() in the unit that reads best (10.72 MPa, not 10720000 Pa; 25 mm, not 0.025 m)."""
    d = q.dimensionality
    if d == ureg("J").dimensionality:                     # energy or torque: keep what it was written as
        choices: tuple[str, ...] = ("J", "kJ") if _is_energy(q) else ("N*m", "kN*m")
    else:
        choices = next((c for base, c in _DISPLAY if ureg(base).dimensionality == d), ())
    if not choices:
        return fmt(q, None, sig)
    if choices == ("K",):
        sig = max(sig, 5)                   # 293.15 K, not 293.1 K
    big = max((abs(x) for x in _components(q.to(choices[0]))), default=0.0)
    for u in choices:
        v = big * float(Q(1.0, choices[0]).to(u).magnitude)
        if big == 0.0 or 0.1 <= v < 1000:
            return fmt(q, u, sig)
    for u in choices:                       # nothing reads well: the first unit that is not tiny
        if big * float(Q(1.0, choices[0]).to(u).magnitude) >= 0.1:
            return fmt(q, u, sig)
    return fmt(q, choices[0], sig)


def from_si(value: float | Iterable[float], unit_si: str) -> pint.Quantity:
    return Q(list(value) if isinstance(value, (list, tuple)) else float(value), unit_si)


# --------------------------------------------------------------------------- expressions

_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Call, ast.Name, ast.Load, ast.Constant,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.USub, ast.UAdd, ast.Subscript, ast.List, ast.Tuple,
)


def _dimensionless(x, what: str) -> float:
    if isinstance(x, pint.Quantity):
        if not x.dimensionless:
            raise UnitsError(f"{what} needs a pure number, got {fmt(x)}")
        return float(x.to("").magnitude)
    return float(x)


def _sqrt(x):
    return x ** 0.5


def _minmax(fn):
    def f(*args):
        if len(args) == 1 and is_vector(args[0]):
            return Q(float(fn(args[0].magnitude)), args[0].units)
        try:
            return fn(args)
        except Exception as e:  # noqa: BLE001
            raise UnitsError(f"{fn.__name__}() of quantities with different dimensions: {e}") from e
    return f


def _norm(x):
    """Magnitude of a vector quantity."""
    if isinstance(x, pint.Quantity):
        return Q(math.sqrt(sum(c * c for c in _components(x))), x.units)
    return math.sqrt(sum(float(c) ** 2 for c in x))


FUNCTIONS: dict[str, Callable] = {
    "sqrt": _sqrt,
    "abs": abs,
    "min": _minmax(min),
    "max": _minmax(max),
    "norm": _norm,
    "exp": lambda x: math.exp(_dimensionless(x, "exp()")),
    "log": lambda x: math.log(_dimensionless(x, "log()")),
    "log10": lambda x: math.log10(_dimensionless(x, "log10()")),
    "sin": lambda x: math.sin(_dimensionless(x, "sin()")),
    "cos": lambda x: math.cos(_dimensionless(x, "cos()")),
    "tan": lambda x: math.tan(_dimensionless(x, "tan()")),
}
CONSTANTS = {"pi": math.pi}


def names_in(expr: str) -> set[str]:
    """The names an expression reads (parameters and units; not functions, not constants)."""
    tree = _parse(expr)
    called = {n.func.id for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    return {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} - called - set(CONSTANTS)


def _parse(expr: str) -> ast.Expression:
    if not isinstance(expr, str) or not expr.strip():
        raise UnitsError(f"empty expression: {expr!r}")
    try:
        tree = ast.parse(expr.strip(), mode="eval")
    except SyntaxError as e:
        raise UnitsError(f"cannot read expression {expr!r}: {e.msg}") from e
    for node in ast.walk(tree):
        if not isinstance(node, _NODES):
            raise UnitsError(f"expression {expr!r} uses {type(node).__name__}, which is not allowed "
                             f"(arithmetic, names, numbers, function calls and [index] only)")
        if isinstance(node, ast.Call) and not isinstance(node.func, ast.Name):
            raise UnitsError(f"expression {expr!r}: only plain function calls are allowed")
        if isinstance(node, ast.Constant) and not isinstance(node.value, (int, float, str)):
            raise UnitsError(f"expression {expr!r}: only numbers and text arguments are allowed")
    return tree


def is_unit(name: str) -> bool:
    try:
        ureg.parse_units(name)
        return True
    except Exception:  # noqa: BLE001
        return False


def evaluate(expr: str, names: dict[str, Any], functions: dict[str, Callable] | None = None) -> pint.Quantity:
    """Evaluate an expression. ``names`` maps parameter names to quantities; a name that is not in
    ``names`` is read as a unit. Returns a quantity (a pure number comes back dimensionless)."""
    tree = _parse(expr)
    fns = {**FUNCTIONS, **(functions or {})}
    env: dict[str, Any] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if node.func.id not in fns:
                raise UnitsError(f"expression {expr!r}: unknown function {node.func.id}(); "
                                 f"functions: {', '.join(sorted(fns))}")
    called = {n.func.id for n in ast.walk(tree) if isinstance(n, ast.Call)}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Name) or node.id in env:
            continue
        n = node.id
        if n in called and n in fns:
            env[n] = fns[n]
        elif n in names:
            env[n] = names[n]
        elif n in CONSTANTS:
            env[n] = CONSTANTS[n]
        elif is_unit(n):
            env[n] = Q(1.0, n)
        else:
            near = [k for k in names if k.lower() == n.lower() or n.lower() in k.lower() or k.lower() in n.lower()]
            hint = f" Did you mean {', '.join(sorted(near)[:4])}?" if near else ""
            raise UnitsError(f"expression {expr!r}: {n!r} is not a parameter and not a unit.{hint}")
    try:
        out = eval(compile(tree, "<expr>", "eval"), {"__builtins__": {}}, env)  # noqa: S307  AST-whitelisted
    except UnitsError:
        raise
    except pint.DimensionalityError as e:
        raise UnitsError(f"expression {expr!r} mixes dimensions: {e}") from e
    except ZeroDivisionError as e:
        raise UnitsError(f"expression {expr!r} divides by zero") from e
    except Exception as e:  # noqa: BLE001
        raise UnitsError(f"expression {expr!r} failed: {type(e).__name__}: {e}") from e
    if isinstance(out, (list, tuple)):
        qs = [x if isinstance(x, pint.Quantity) else Q(float(x)) for x in out]
        try:
            u = next((q.units for q in qs if float(abs(q.magnitude)) != 0.0), qs[0].units)
            out = Q([float(q.to(u).magnitude) if float(abs(q.magnitude)) != 0.0 else 0.0 for q in qs], u)
        except pint.DimensionalityError as e:
            raise UnitsError(f"expression {expr!r}: the vector's components have different dimensions: {e}") from e
    return out if isinstance(out, pint.Quantity) else Q(float(out))
