"""The parts of a hand-calculation model: its inputs and outputs, the calculation as steps, the registry.

A model is a closed-form formula set with everything an engineer needs to check it:

    name, title      how the specification names it (HandCalcSpec.model), and what it is
    reference        the textbook or standard the formulas come from (author, title, edition, section)
    inputs           name, description, SI unit of the dimension, and for a material property the name of the
                     project context that gives it when the specification does not (E, nu, alpha, k_th, Su)
    outputs          name, description, SI unit
    assumptions      what the formulas take as true (ASD-STE100 sentences)
    validity         a function of the numbers that returns one warning per limit of the theory that the
                     inputs are outside of (a thick plate in thin-plate theory, a Reynolds number outside a
                     correlation)
    run              the calculation: a function that writes the steps with `Calc.let` (one sentence, one
                     formula, one result each), in the order an engineer writes them by hand

`Model.calculate({"p": 15.75e6, "r_i": 0.022, "r_o": 0.028})` runs a model on SI numbers (tests, scripts);
`dossier.handcalc.evaluate` runs it for a specification of a project and makes the record.

To add a model: write its function in the module of its field, decorate it with `@model(...)`, state the
source, and add a test that compares every output with an independent number (a second derivation, a
published value, an identity or a numerical solve). A model without such a test does not go in.
"""

from __future__ import annotations

import difflib
import math
from dataclasses import dataclass, field
from typing import Callable, Mapping

from dossier import units
from dossier.handcalc import formula
from dossier.model import CalcStep

__all__ = ["HandCalcError", "Input", "Output", "Model", "Calc", "MODELS", "model", "register", "get", "shown"]


class HandCalcError(ValueError):
    """A hand calculation that cannot be made: an unknown model, an input that is missing, has the wrong
    dimension or makes the formula impossible. The message is a sentence that names the cause."""


@dataclass(frozen=True)
class Input:
    name: str
    description: str                # a noun phrase: "Inner radius of the cylinder"
    unit: str                       # the SI unit of its dimension; "" is a pure number
    default: str | None = None      # a name of the project context that gives the value when the
    #                                 specification does not (a property of the subject's material)
    sign: str = "positive"          # positive (> 0) | nonnegative (>= 0) | any
    difference: bool = False        # a temperature difference: an offset unit (degC) is an error
    optional: bool = False          # the calculation runs without it (it serves a validity check)
    symbol: str | None = None       # LaTeX; default: formula.auto_symbol(name)


@dataclass(frozen=True)
class Output:
    name: str
    description: str
    unit: str                       # SI
    symbol: str | None = None
    show: str | None = None         # the unit people read, when dossier.units.display has no rule for it


def shown(value: float, unit: str, show: str | None = None) -> str:
    """An SI number for people: "57.75 MPa", "13070 mm⁴"."""
    q = units.from_si(value, unit)
    return units.fmt(q, show) if show else units.display(q)


class Calc:
    """One run of a model: the SI numbers by name, and the steps in the order of the calculation."""

    def __init__(self, model: "Model", values: Mapping[str, float]):
        self.model = model
        self.values: dict[str, float] = dict(values)
        self.steps: list[CalcStep] = []
        self.warnings: list[str] = []
        self._unit = {x.name: x.unit for x in (*model.inputs, *model.outputs)}
        self._show = {o.name: o.show for o in model.outputs if o.show}
        self._symbol = {**model.symbols, **{x.name: x.symbol for x in (*model.inputs, *model.outputs) if x.symbol}}

    def __getitem__(self, name: str) -> float:
        return self.values[name]

    def __contains__(self, name: str) -> bool:
        return self.values.get(name) is not None

    def symbol(self, name: str) -> str:
        return self._symbol.get(name) or formula.auto_symbol(name)

    def show(self, name: str) -> str:
        """The value of a quantity for people, in the unit of its declaration."""
        return shown(self.values[name], self._unit.get(name, ""), self._show.get(name))

    def require(self, ok: bool, message: str) -> None:
        """Stop the calculation when the numbers make the formulas impossible."""
        if not ok:
            raise HandCalcError(message)

    def _declare(self, name: str, unit: str | None, symbol: str | None, show: str | None) -> None:
        if name in self.values:
            raise RuntimeError(f"model {self.model.name}: the step {name!r} has the name of an input or of an "
                               f"earlier step")
        if unit is not None:
            self._unit[name] = unit
        elif name not in self._unit:
            raise RuntimeError(f"model {self.model.name}: the step {name!r} has no unit")
        if symbol:
            self._symbol[name] = symbol
        if show:
            self._show[name] = show

    def _record(self, name: str, value: float, text: str, latex: str) -> float:
        if not math.isfinite(value):
            raise HandCalcError(f"The calculation of {name} in the model {self.model.name} gives no finite number.")
        self.values[name] = float(value)
        self.steps.append(CalcStep(text=text, latex=latex, result=self.show(name)))
        return float(value)

    def let(self, name: str, expr: str, text: str, unit: str | None = None, symbol: str | None = None,
            show: str | None = None) -> float:
        """One step: `name` = `expr` of the quantities so far. The expression gives the number and the
        LaTeX. `unit` is the SI unit of the result (an output has it from its declaration)."""
        self._declare(name, unit, symbol, show)
        try:
            value = formula.evaluate(expr, self.values)
        except formula.FormulaError as e:
            raise HandCalcError(f"The calculation of {name} in the model {self.model.name} is not possible: "
                                f"{e}.") from e
        return self._record(name, value, text, f"{self.symbol(name)} = {formula.latex(expr, self.symbol)}")

    def given(self, name: str, value: float, text: str, latex: str, unit: str | None = None,
              symbol: str | None = None, show: str | None = None) -> float:
        """One step whose number the model computes in code (the root of an implicit equation). `latex` is
        the equation as written by hand; a test of the model must show that the number satisfies it."""
        self._declare(name, unit, symbol, show)
        return self._record(name, value, text, latex)


@dataclass(frozen=True)
class Model:
    name: str
    title: str
    reference: str
    inputs: tuple[Input, ...]
    outputs: tuple[Output, ...]
    assumptions: tuple[str, ...]
    run: Callable[[Calc], None]
    validity: Callable[[Mapping[str, float]], list[str]] | None = None
    symbols: Mapping[str, str] = field(default_factory=dict)    # LaTeX symbols of intermediate quantities

    def input(self, name: str) -> Input:
        for x in self.inputs:
            if x.name == name:
                return x
        raise HandCalcError(f"The model {self.name} has no input {name!r}. Its inputs are: "
                            f"{', '.join(x.name for x in self.inputs)}.")

    def output(self, name: str) -> Output:
        for x in self.outputs:
            if x.name == name:
                return x
        raise HandCalcError(f"The model {self.name} has no output {name!r}. Its outputs are: "
                            f"{', '.join(x.name for x in self.outputs)}.")

    def calculate(self, values: Mapping[str, float]) -> Calc:
        """Run the model on SI numbers. Returns the Calc: `.values` (inputs, intermediate quantities and
        outputs by name), `.steps`, `.warnings`. Raises HandCalcError when an input is missing or impossible."""
        known = {x.name for x in self.inputs}
        extra = sorted(set(values) - known)
        if extra:
            raise HandCalcError(f"The model {self.name} has no input {', '.join(extra)}. Its inputs are: "
                                f"{', '.join(x.name for x in self.inputs)}.")
        clean: dict[str, float] = {}
        for x in self.inputs:
            v = values.get(x.name)
            if v is None:
                if x.optional:
                    continue
                raise HandCalcError(f"The model {self.name} has no value for its input {x.name} ({x.description}).")
            v = float(v)
            if not math.isfinite(v):
                raise HandCalcError(f"The input {x.name} of the model {self.name} is not a finite number.")
            if (x.sign == "positive" and v <= 0.0) or (x.sign == "nonnegative" and v < 0.0):
                need = "more than zero" if x.sign == "positive" else "zero or more"
                raise HandCalcError(f"The input {x.name} ({x.description}) of the model {self.name} must be {need}. "
                                    f"It is {shown(v, x.unit)}.")
            clean[x.name] = v
        calc = Calc(self, clean)
        self.run(calc)
        missing = [o.name for o in self.outputs if o.name not in calc.values]
        if missing:
            raise RuntimeError(f"model {self.name}: the calculation made no {', '.join(missing)}")
        if self.validity is not None:
            calc.warnings = list(self.validity(calc.values))
        return calc


MODELS: dict[str, Model] = {}


def register(m: Model) -> Model:
    if m.name in MODELS:
        raise RuntimeError(f"hand-calculation model {m.name!r} is defined twice")
    names = [x.name for x in (*m.inputs, *m.outputs)]
    if len(set(names)) != len(names):
        raise RuntimeError(f"hand-calculation model {m.name!r}: an input or output name is used twice")
    MODELS[m.name] = m
    return m


def model(name: str, title: str, reference: str, inputs: list[Input], outputs: list[Output],
          assumptions: list[str], validity: Callable[[Mapping[str, float]], list[str]] | None = None,
          symbols: Mapping[str, str] | None = None) -> Callable[[Callable[[Calc], None]], Model]:
    """Decorator: the function is the calculation, the arguments are the rest of the model."""
    def deco(run: Callable[[Calc], None]) -> Model:
        return register(Model(name=name, title=title, reference=reference, inputs=tuple(inputs),
                              outputs=tuple(outputs), assumptions=tuple(assumptions), run=run, validity=validity,
                              symbols=dict(symbols or {})))
    return deco


def get(name: str) -> Model:
    """The model of that name. An unknown name is an error that lists the models."""
    if name in MODELS:
        return MODELS[name]
    near = difflib.get_close_matches(str(name), list(MODELS), n=3)
    hint = f" The nearest names are: {', '.join(near)}." if near else ""
    raise HandCalcError(f"There is no hand-calculation model {name!r}.{hint} The models are: "
                        f"{', '.join(sorted(MODELS))}.")
