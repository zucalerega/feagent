"""Numbers and names for people.

Records hold SI numbers. The report shows each number in the unit people read (dossier.units.display), keeps the
SI value for a tooltip, and never shows more digits than the uncertainty of the number supports.
"""

from __future__ import annotations

import math
from typing import Any

from dossier import units


def sig_for(value: float | None, uncertainty: float | None, default: int = 4) -> int:
    """Significant digits a value with a relative uncertainty supports: the last digit shown is at the place of
    the first digit of the uncertainty. 148.3 MPa with 3 % -> 3 digits (148 MPa)."""
    if not value or not uncertainty or not math.isfinite(value) or uncertainty <= 0:
        return default
    u = abs(value) * uncertainty
    n = math.floor(math.log10(abs(value))) - math.floor(math.log10(u)) + 1
    return max(1, min(default, n))


def _number(x: float, sig: int = 4) -> str:
    return units._sig(float(x), sig)  # noqa: SLF001  the one number format of the project


def quantity(value: Any, unit: str = "", sig: int = 4) -> str:
    """An SI number (or vector) in the unit people read: 148300000, "Pa" -> "148.3 MPa"."""
    if value is None:
        return ""
    try:
        return units.display(units.from_si(value, unit or ""), sig)
    except Exception:  # noqa: BLE001  a unit pint does not know: show the number as it is
        if isinstance(value, (list, tuple)):
            return "[" + ", ".join(_number(v, sig) for v in value) + "]" + (f" {unit}" if unit else "")
        return _number(value, sig) + (f" {unit}" if unit else "")


def si_text(value: Any, unit: str = "") -> str:
    """The SI value for a tooltip: "1.483e8 Pa"."""
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        body = "[" + ", ".join(_number(v, 6) for v in value) + "]"
    else:
        body = _number(value, 6)
    return f"{body} {unit}".strip()


def magnitude(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return math.sqrt(sum(float(c) ** 2 for c in value))
    return float(value)


def margin(m: float | None) -> str:
    """A margin of safety with its sign and two decimals: +0.42, -0.13."""
    if m is None or not math.isfinite(m):
        return ""
    s = f"{m:+.2f}"
    return "-0.00" if s == "-0.00" else s


def percent(x: float | None, digits: int = 1) -> str:
    if x is None:
        return ""
    return f"{100 * x:.{digits}f} %"


def ratio(x: float | None) -> str:
    return "" if x is None else f"{x:.2f}"


def plain(x: Any, sig: int = 4) -> str:
    """A bare number of a record (a check value, a count)."""
    if x is None:
        return ""
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, (int, float)):
        return _number(x, sig)
    if isinstance(x, (list, tuple)):
        return "[" + ", ".join(plain(v, sig) for v in x) + "]"
    return str(x)


def count(n: int, one: str, many: str | None = None) -> str:
    """"1 finding", "3 findings"."""
    return f"{n} {one if n == 1 else (many or one + 's')}"


# --------------------------------------------------------------------------- names of results

QUANTITY = {
    "von_mises": "von Mises stress", "max_principal": "maximum principal stress", "sxx": "stress component xx",
    "syy": "stress component yy", "szz": "stress component zz", "displacement": "displacement",
    "radial_displacement": "radial displacement", "temperature": "temperature", "heat_flow": "heat flow",
    "pressure_drop": "pressure drop", "damage": "fatigue damage", "life": "life", "pressure": "pressure",
}
MEASURE = {
    "peak": "peak", "max": "maximum", "min": "minimum", "mean": "mean", "range": "range", "membrane": "membrane",
    "membrane_bending": "membrane plus bending", "total": "total",
}


def split_key(key: str) -> tuple[str, str, str]:
    """`von_mises.peak@part` -> ("von_mises", "peak", "part")."""
    head, _, loc = key.partition("@")
    q, _, m = head.partition(".")
    return q, m, loc


def key_words(key: str) -> tuple[str, str]:
    """A result key in words, without its location, and the location: ("von Mises stress, peak", "part")."""
    q, m, loc = split_key(key)
    name = QUANTITY.get(q, q.replace("_", " "))
    measure = MEASURE.get(m, m.replace("_", " "))
    return (f"{name}, {measure}" if measure else name), loc


BC_UNITS = {"pressure": "Pa", "traction": "Pa", "force": "N", "moment": "N*m", "displacement": "m", "temperature": "K",
            "acceleration": "m/s^2", "heat_flux": "W/m^2", "mass_flow": "kg/s", "velocity": "m/s", "gravity": "m/s^2"}
BC_KEY_UNITS = {"h": "W/(m^2*K)", "htc": "W/(m^2*K)", "T": "K", "T_inf": "K", "t_inf": "K", "temperature": "K",
                "T_fluid": "K", "p": "Pa", "pressure": "Pa", "force": "N", "moment": "N*m", "mdot": "kg/s",
                "mass_flow": "kg/s", "velocity": "m/s", "distance": "m"}


def bc_value(bc_type: str, value: Any) -> tuple[str, str]:
    """The value of a support or load of an analysis model for people, and its SI text. The record holds a bare
    SI number, a vector or a small table of numbers; the type of the item gives the unit."""
    if value is None:
        return "", ""
    unit = BC_UNITS.get(bc_type, "")
    if isinstance(value, dict):
        shown, raw = [], []
        for k, v in value.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                shown.append(f"{k} = {quantity(v, BC_KEY_UNITS.get(k, ''))}")
                raw.append(f"{k} = {si_text(v, BC_KEY_UNITS.get(k, ''))}")
            elif isinstance(v, (list, tuple)) and all(isinstance(c, (int, float)) for c in v):
                shown.append(f"{k} = {quantity(list(v), BC_KEY_UNITS.get(k, ''))}")
                raw.append(f"{k} = {si_text(list(v), BC_KEY_UNITS.get(k, ''))}")
            else:
                shown.append(f"{k} = {v}")
        return ", ".join(shown), ", ".join(raw)
    if isinstance(value, (list, tuple)) and all(isinstance(c, (int, float)) and not isinstance(c, bool) for c in value):
        if bc_type in ("displacement",) and not any(value):
            return "zero", si_text(list(value), unit)
        return quantity(list(value), unit), si_text(list(value), unit)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return quantity(value, unit), si_text(value, unit)
    return str(value), ""
