"""The data model: what an engineer writes (inputs) and what the pipeline derives (records).

Inputs (YAML, written and reviewed by people)
    System        elements, connections between them, phases, duty cycles, named parameters. A connection
                  carries effects: what the system does to an element at that interface in a phase
                  (pressure, force, moment, temperature, flow, acceleration, support).
    Component     the part under verification: CAD, material, named features (face sets selected by
                  geometry), interfaces (which features realize which system connection), stress
                  classification lines, hand-calculation models.
    Requirement   the source text, its parameters (claims about the environment, checked against the
                  system context) and its criteria (a response quantity, a limit, a factor of safety).
    Material, Policy   properties with sources; the factor policy of the governing standard.

Records (JSON, written by the pipeline stages; every one has an id, an origin and links)
    Finding        a problem in the requirements, found against the system context
    LoadCase       one analysis of one phase at limit loads, with the requirements it serves
    HandCalcResult a closed-form calculation with its formula, inputs, sources and limits
    AnalysisRecord the model (supports, loads, mesh), the convergence study, the results, the checks
    ComplianceItem one criterion of one requirement against one result: margin and verdict
    Decision       a choice the pipeline (a rule, an agent or a person) made, with its justification
    Assumption     something taken as true without proof, with what happens if it is wrong
    Review         a person's acceptance or rejection of a record (reviews.yaml, owned by people)

Rules of the model
    * Loads come from the system context (effects in phases). Requirements give criteria and factors.
      A requirement parameter is a claim; the validation stage compares it with the context.
    * A load case is a phase at limit loads (factor 1.0). Factors of safety are applied to the response
      in the margin, so one solve serves every criterion of that phase.
    * Every record says where it comes from (``links``, ``origin``). Text fields (``statement``,
      ``rationale``, ``choice``) are written in ASD-STE100 Simplified Technical English.
    * SI inside records (``si`` fields); the unit people read is kept next to it (``display``).
"""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from dossier import units

SCHEMA_VERSION = 1

ID_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]*$")


class Model(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


def _check_id(v: str) -> str:
    if not isinstance(v, str) or not ID_RE.match(v):
        raise ValueError(f"id {v!r} must start with a letter and use only letters, digits, '-', '_' and '.'")
    return v


# =========================================================================== quantities


class Value(Model):
    """A scalar or vector quantity: a literal or an expression of named project parameters.

    YAML forms: ``10.4 MPa`` | ``1.5`` | ``{value: [0, 0, -450], unit: N}`` |
    ``{expr: "p_pump_max + dp_surge", unit: MPa}`` | ``{expr: ["p * a_bore", 0, 0], unit: N}``.
    ``unit`` of an expression is the unit it is shown in; the result must have that dimension."""

    value: float | list[float] | None = None
    unit: str = ""
    expr: str | list[str | float] | None = None

    @model_validator(mode="before")
    @classmethod
    def _read(cls, v: Any) -> Any:
        if isinstance(v, bool):
            raise ValueError("a quantity cannot be true/false")
        if isinstance(v, (int, float)):
            return {"value": float(v)}
        if isinstance(v, str):
            q = units.parse(v)
            return {"value": float(q.magnitude), "unit": f"{q.units:~}"}
        if isinstance(v, (list, tuple)):
            return {"value": [float(x) for x in v]}
        if isinstance(v, dict) and isinstance(v.get("value"), str):
            q = units.parse(v["value"])
            if v.get("unit"):
                q = units.to(q, v["unit"])
            return {**v, "value": float(q.magnitude), "unit": f"{q.units:~}"}
        return v

    @model_validator(mode="after")
    def _one_of(self) -> "Value":
        if (self.value is None) == (self.expr is None):
            raise ValueError("give exactly one of `value` (a literal, e.g. 10.4 MPa) and `expr` (an expression)")
        if self.unit:
            try:
                units.ureg.parse_units(self.unit)
            except Exception as e:  # noqa: BLE001
                raise ValueError(f"unknown unit {self.unit!r}: {e}") from e
        return self

    @property
    def is_literal(self) -> bool:
        return self.expr is None

    def names(self) -> set[str]:
        """Names the expression reads (empty for a literal)."""
        if self.expr is None:
            return set()
        parts = self.expr if isinstance(self.expr, list) else [self.expr]
        out: set[str] = set()
        for p in parts:
            if isinstance(p, str):
                out |= units.names_in(p)
        return out

    def quantity(self, names: dict | None = None, functions: dict | None = None):
        """The pint quantity. ``names`` are the parameters an expression may read."""
        if self.expr is None:
            return units.Q(self.value, self.unit)
        names = names or {}
        if isinstance(self.expr, list):
            comps = []
            for p in self.expr:
                q = units.evaluate(p, names, functions) if isinstance(p, str) else None
                comps.append(q if q is not None else float(p))
            ref = next((c for c in comps if c is not None and not isinstance(c, float)), None)
            unit = self.unit or (f"{ref.units:~}" if ref is not None else "")
            mags = [float(units.to(c, unit).magnitude) if not isinstance(c, float) else c for c in comps]
            return units.Q(mags, unit)
        q = units.evaluate(self.expr, names, functions)
        return units.to(q, self.unit) if self.unit else q


Basis = Literal["specified", "measured", "derived", "assumed", "reference"]


class Parameter(Value):
    """A named quantity of the project, with where it comes from."""

    title: str = ""
    basis: Basis = "specified"
    source: str = ""          # the document, table or record the number comes from
    note: str = ""


# =========================================================================== system (input)

EffectKind = Literal["pressure", "force", "moment", "acceleration", "fluid_temperature", "temperature",
                     "heat_flux", "mass_flow", "displacement", "support", "signal"]

# The dimension of an effect's value (a unit of that dimension), and whether it is a vector.
EFFECT_UNITS: dict[str, tuple[str, bool]] = {
    "pressure": ("Pa", False), "force": ("N", True), "moment": ("N*m", True), "acceleration": ("m/s^2", True),
    "fluid_temperature": ("K", False), "temperature": ("K", False), "heat_flux": ("W/m^2", False),
    "mass_flow": ("kg/s", False), "displacement": ("m", True),
}
PhaseCategory = Literal["operation", "test", "handling", "storage"]
ConnectionKind = Literal["fluid", "mechanical", "thermal", "signal", "electrical", "environment"]


class Phase(Model):
    """An operating state of the system: the effects of a phase act together."""

    id: str
    name: str
    description: str = ""
    category: PhaseCategory = "operation"
    duration: Value | None = None
    state: dict[str, str] = Field(default_factory=dict)     # e.g. {valve: closed}

    _id = field_validator("id")(_check_id)


class DutyCycle(Model):
    """A sequence of phases that repeats `count` times in the life of the part (for cyclic analysis)."""

    id: str
    name: str
    description: str = ""
    sequence: list[str]            # phase ids, in order; the part returns to its unloaded state after the last
    count: Value                   # number of repetitions (a pure number or an expression)
    source: str = ""
    note: str = ""

    _id = field_validator("id")(_check_id)


class Element(Model):
    """A thing in the system: equipment, structure, a medium source, the environment."""

    id: str
    name: str
    kind: str = "equipment"        # free vocabulary: tank, pump, valve, line, actuator, controller, structure, environment
    description: str = ""
    component: str | None = None   # the component of this project that realizes the element
    group: str = ""                # subsystem (a lane of the diagram)
    position: tuple[float, float] | None = None   # diagram hint (column, row); optional

    _id = field_validator("id")(_check_id)


class Effect(Model):
    """What a connection does to an element in a phase. Vectors are in the CAD frame of the element's
    component. A `support` effect says the connection holds the element in that phase."""

    id: str
    phase: str | list[str]
    kind: EffectKind
    value: Value | None = None
    acts_on: str | None = None     # the element that feels the effect; default: the connection's `to`
    #                                (not named `on`: YAML reads a bare `on` as true)
    h: Value | None = None         # fluid_temperature: heat transfer coefficient between fluid and wall
    at: Value | None = None        # force / moment: point of application; default: centroid of the seat features
    role: str | None = None        # the feature role the effect acts on, when not the default of its kind
    description: str = ""
    basis: Basis = "specified"
    source: str = ""
    note: str = ""

    _id = field_validator("id")(_check_id)

    @model_validator(mode="after")
    def _shape(self) -> "Effect":
        if self.kind in ("support", "signal"):
            return self
        if self.value is None:
            raise ValueError(f"effect {self.id}: kind {self.kind} needs a value")
        return self

    def phases(self) -> list[str]:
        return [self.phase] if isinstance(self.phase, str) else list(self.phase)


class Connection(Model):
    """An interface between two elements. Its effects are stated for the element in `acts_on` (default `to`)."""

    id: str
    name: str
    from_: str = Field(alias="from")
    to: str
    kind: ConnectionKind
    medium: str = ""
    description: str = ""
    effects: list[Effect] = Field(default_factory=list)

    _id = field_validator("id")(_check_id)


class System(Model):
    id: str
    name: str
    description: str = ""
    parameters: dict[str, Parameter] = Field(default_factory=dict)
    phases: list[Phase] = Field(default_factory=list)
    duty_cycles: list[DutyCycle] = Field(default_factory=list)
    elements: list[Element] = Field(default_factory=list)
    connections: list[Connection] = Field(default_factory=list)

    _id = field_validator("id")(_check_id)


# =========================================================================== component (input)

# Feature roles of an interface, and the default role each effect kind acts on.
ROLES = ("wetted", "seat", "bolts", "external", "body")
DEFAULT_ROLE = {"pressure": "wetted", "fluid_temperature": "wetted", "temperature": "wetted", "heat_flux": "wetted",
                "mass_flow": "wetted", "force": "seat", "moment": "seat", "displacement": "seat", "support": "seat",
                "acceleration": "body"}


class Cad(Model):
    file: str                                   # relative to the component folder
    length_unit: Literal["m", "mm", "in"] = "m"


class FeatureDef(Model):
    """A named set of CAD faces, selected by geometry (the feagent query language) so that it survives
    a re-export of the CAD, or listed by face tag."""

    description: str
    select: dict | None = None      # e.g. {kind: cylinder, axis: [1, 0, 0], radius: 0.0125}
    faces: list[int] | None = None  # explicit OCC face tags (f<tag>)
    expect: int | None = None       # the number of faces the selection must give

    @model_validator(mode="after")
    def _one(self) -> "FeatureDef":
        if (self.select is None) == (self.faces is None):
            raise ValueError("a feature needs exactly one of `select` (a geometric query) and `faces` (tags)")
        return self


class Interface(Model):
    """How the component realizes one system connection: which features play which role."""

    id: str
    name: str
    connection: str
    description: str = ""
    joint: str = ""                 # bolted_flange, welded, threaded, clamped, sliding, bonded, free, ...
    features: dict[str, list[str]] = Field(default_factory=dict)   # role -> feature ids

    _id = field_validator("id")(_check_id)

    @field_validator("features")
    @classmethod
    def _roles(cls, v: dict) -> dict:
        bad = [r for r in v if r not in ROLES]
        if bad:
            raise ValueError(f"unknown feature role(s) {bad}; roles: {', '.join(ROLES)}")
        return v


class Scl(Model):
    """A stress classification line: a straight line through a wall, from one surface to the other.
    The section stresses (membrane, membrane plus bending) are the stress components averaged and
    linearized along it."""

    id: str
    description: str
    start: tuple[float, float, float]   # m, CAD frame, on one surface
    end: tuple[float, float, float]     # m, on the opposite surface

    _id = field_validator("id")(_check_id)


class Compare(Model):
    """One output of a hand calculation against one result of the analysis."""

    output: str                     # an output of the hand-calculation model
    with_: str = Field(alias="with")   # a result key, e.g. "max_principal.membrane@SCL-SHELL"
    band: tuple[float, float] | None = None    # accepted ratio analysis / hand calculation
    tolerance: Value | None = None  # or an absolute tolerance (temperatures)
    why: str = ""                   # why the two must agree this closely

    @model_validator(mode="after")
    def _one(self) -> "Compare":
        if self.band is not None and self.tolerance is not None:
            raise ValueError("give `band` (a ratio range) or `tolerance` (absolute), not both")
        return self


class HandCalcWhen(Model):
    analysis: list[str] = Field(default_factory=list)   # analysis types it applies to; empty = any
    needs: list[str] = Field(default_factory=list)      # "kind@INTERFACE" loads the load case must have
    phases: list[str] = Field(default_factory=list)     # restrict to these phases; empty = any


class HandCalcSpec(Model):
    """A closed-form model of a region of the part, used as a sanity check of the analysis.
    Input expressions may read project parameters, the material (E, nu, alpha, k_th, rho, Sy, Su), and
    the load case through ``load("INT-ID", "pressure")`` and ``feat("FT-ID", "radius")``."""

    id: str
    model: str                      # a model of dossier.handcalc
    title: str
    description: str = ""
    when: HandCalcWhen = Field(default_factory=HandCalcWhen)
    inputs: dict[str, Value] = Field(default_factory=dict)
    compare: list[Compare] = Field(default_factory=list)

    _id = field_validator("id")(_check_id)


class FlowDomain(Model):
    """The fluid volume for flow analysis: a second STEP file of the wetted volume, with its own
    features (inlet, outlet; every other face is a wall)."""

    cad: Cad
    inlet: FeatureDef
    outlet: FeatureDef
    fluid: dict[str, Value] = Field(default_factory=dict)   # rho, mu
    description: str = ""


class Component(Model):
    id: str
    element: str                    # the system element it realizes
    name: str
    drawing: str = ""
    description: str = ""
    cad: Cad
    material: str
    reference_temperature: Value = Field(default_factory=lambda: Value(value=293.15, unit="K"))  # stress-free
    features: dict[str, FeatureDef] = Field(default_factory=dict)
    interfaces: list[Interface] = Field(default_factory=list)
    scls: list[Scl] = Field(default_factory=list)
    hand_calcs: list[HandCalcSpec] = Field(default_factory=list)
    flow: FlowDomain | None = None
    notes: str = ""

    _id = field_validator("id")(_check_id)


# =========================================================================== material, policy (input)


class Fatigue(Model):
    """Stress-life data. basquin: stress amplitude S_a = sigma_f * (2 N)^b at zero mean stress."""

    model: Literal["basquin"] = "basquin"
    sigma_f: Value
    b: float
    endurance_limit: Value | None = None
    mean_stress: Literal["goodman", "swt", "none"] = "goodman"
    source: str = ""
    note: str = ""


class Material(Model):
    id: str
    name: str
    spec: str = ""
    source: str = ""                # where the numbers come from
    note: str = ""
    E: Value
    nu: float
    rho: Value
    alpha: Value | None = None      # mean coefficient of thermal expansion over the service range
    k: Value | None = None          # thermal conductivity
    cp: Value | None = None
    yield_strength: Value
    ultimate_strength: Value
    properties_temperature: Value | None = None   # the temperature the properties above apply at
    service_temperature: tuple[Value, Value] | None = None   # lowest and highest allowed temperature
    compatibility: dict[str, Literal["ok", "conditional", "incompatible"]] = Field(default_factory=dict)
    # medium -> rating (not "no": YAML reads a bare `no` as false)
    fatigue: Fatigue | None = None

    _id = field_validator("id")(_check_id)


class Factor(Model):
    value: float
    source: str = ""
    note: str = ""


class Policy(Model):
    """The factor policy of the governing standard, and the acceptance rules of the pipeline."""

    id: str = "POLICY"
    name: str
    source: str = ""
    note: str = ""
    factors: dict[str, Factor] = Field(default_factory=dict)   # yield, ultimate, proof, burst, fatigue_life, ...
    handcalc_band: tuple[float, float] = (0.8, 1.25)           # default accepted ratio analysis / hand calculation
    max_uncertainty: float = 0.05     # a result with a larger discretization uncertainty is not converged
    unconverged_margin_factor: float = 2.0
    # A result that is not converged but has a stated uncertainty can still decide a criterion when the margin
    # holds with this many times that uncertainty on the unsafe side. 0 turns the rule off: then a result that
    # is not converged never gives a pass or a fail.
    questioned_requirement_blocks_pass: bool = True
    # A finding of severity error or major that questions a value a criterion uses (the requirement, its
    # parameters, its factor or limit, a parameter its loads depend on) turns a pass into inconclusive until a
    # person resolves the finding. False restores the old rule: the finding is only a caveat in the statement.
    thermal_delta: Value = Field(default_factory=lambda: Value(value=5.0, unit="K"))
    # a phase whose temperatures differ from the reference temperature by more needs a thermal analysis


# =========================================================================== requirements (input)

ReqKind = Literal["strength", "stiffness", "life", "thermal", "flow", "leak", "functional", "mass", "environment",
                  "interface", "definition"]
Verification = Literal["analysis", "test", "inspection", "review", "similarity"]

# Built-in roles of a requirement parameter: the validation stage knows what each must cover.
PARAM_ROLES = ("meop", "min_temperature", "max_temperature", "rated_flow", "design_cycles", "proof_factor",
               "burst_factor", "yield_factor", "ultimate_factor", "life_factor")


class ParamCheck(Model):
    """An explicit check of a requirement parameter against the system context."""

    relation: Literal[">=", "<=", "=="]
    against: str                    # an expression of context parameters
    tolerance: float = 0.0          # relative
    why: str = ""


class ReqParameter(Value):
    """A number the requirement states. It is published under its name to every expression of the project."""

    role: str = ""                  # one of PARAM_ROLES, or empty
    check: ParamCheck | None = None
    note: str = ""


class Criterion(Model):
    """A machine-checkable acceptance criterion: `factor` x response `comparator` limit."""

    id: str = ""                    # a short label: yield, ultimate
    quantity: str                   # von_mises | max_principal | displacement | temperature | pressure_drop | life | ...
    measure: str = "peak"           # peak | max | min | mean | membrane | membrane_bending | range | total
    location: str = "part"          # part | a feature id | an interface id | an SCL id | scls (every SCL)
    comparator: Literal["<=", ">="]
    limit: Value | None = None      # a literal or an expression (parameters, Sy, Su, ...)
    allowable: str | None = None    # or a material property: yield_strength | ultimate_strength
    factor: float | str = 1.0       # factor of safety on the response; text names a requirement parameter or a policy factor
    note: str = ""

    @model_validator(mode="after")
    def _limit(self) -> "Criterion":
        if (self.limit is None) == (self.allowable is None):
            raise ValueError("a criterion needs exactly one of `limit` and `allowable`")
        return self

    def key(self) -> str:
        return f"{self.quantity}.{self.measure}@{self.location}"


class Applies(Model):
    phases: list[str] = Field(default_factory=list)          # phase ids
    category: PhaseCategory | Literal["all"] | None = None   # or every phase of a category
    duty_cycles: list[str] = Field(default_factory=list)     # life requirements


class Requirement(Model):
    id: str
    title: str
    text: str                       # the source text, as written in the specification
    kind: ReqKind
    source: str = ""
    parent: str = ""
    rationale: str = ""
    parameters: dict[str, ReqParameter] = Field(default_factory=dict)
    criteria: list[Criterion] = Field(default_factory=list)
    applies: Applies = Field(default_factory=Applies)
    verification: Verification = "analysis"

    _id = field_validator("id")(_check_id)


class ProjectFile(Model):
    """project.yaml"""

    id: str
    name: str
    description: str = ""
    system: str = "system.yaml"
    components: list[str]           # component folders (each holds component.yaml, requirements.yaml, CAD)
    subject: str                    # the component under verification
    materials: str = "materials.yaml"
    policy: str = "policy.yaml"
    terms: str | None = None        # project technical names and verbs for the STE check
    author: str = ""

    _id = field_validator("id")(_check_id)


# =========================================================================== records (derived)

Rel = Literal["derives_from", "verifies", "uses", "about", "evidence", "covers", "source"]


class Link(Model):
    """A typed reference to another record (by id) or to an outside document ("doc:ENG-ICD-002 table 4")."""

    rel: Rel
    to: str
    note: str = ""


class Origin(Model):
    by: str            # rule:<name> | engine:<name> | agent:<role> | human:<name> | input
    stage: str = ""    # context | geometry | validate | plan | analyze | assess | report


class Record(Model):
    """Base of every derived record."""

    id: str
    title: str = ""
    links: list[Link] = Field(default_factory=list)
    origin: Origin = Field(default_factory=lambda: Origin(by="input"))

    _id = field_validator("id")(_check_id)

    def refs(self) -> list[Link]:
        """Every reference of the record: the explicit links plus the typed fields (subclasses add theirs)."""
        return list(self.links)


Severity = Literal["error", "major", "minor", "info"]
FindingCategory = Literal["consistency", "coverage", "quality", "policy", "conflict", "material", "context", "model"]


class Finding(Record):
    """A problem in the requirements (or in the model), with the evidence and a proposed correction."""

    kind: Literal["finding"] = "finding"
    rule: str                       # the rule that made it, e.g. req.meop_envelope
    severity: Severity
    category: FindingCategory
    subject: str                    # the record it is about (a requirement, a parameter, an effect, a phase)
    statement: str                  # what is wrong
    rationale: str                  # the evidence, with the numbers
    suggestion: str = ""            # the proposed correction
    values: dict[str, Any] = Field(default_factory=dict)   # the numbers of the evidence (SI)

    def refs(self) -> list[Link]:
        return [*self.links, Link(rel="about", to=self.subject)]


class Alternative(Model):
    option: str
    why_not: str


Stage = Literal["context", "geometry", "validate", "plan", "idealize", "mesh", "solve", "assess", "report"]


class Decision(Record):
    """A choice, with why. `basis` says how strong the ground is; `needs_review` asks for a person."""

    kind: Literal["decision"] = "decision"
    stage: Stage
    subject: str                    # the record the decision is about
    choice: str                     # what was decided
    rationale: str                  # why
    alternatives: list[Alternative] = Field(default_factory=list)
    basis: Literal["rule", "calculation", "engineering_judgement", "input"] = "rule"
    needs_review: bool = False

    @field_validator("rationale")
    @classmethod
    def _justified(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("a decision needs a rationale")
        return v

    def refs(self) -> list[Link]:
        return [*self.links, Link(rel="about", to=self.subject)]


class Assumption(Record):
    """Something taken as true without proof."""

    kind: Literal["assumption"] = "assumption"
    stage: Stage
    subject: str
    statement: str                  # what is assumed
    rationale: str                  # why it is acceptable
    impact: str                     # what changes if it is wrong
    verify_by: str = ""             # how a person can check it
    conservative: bool | None = None   # True: errs to the safe side; None: not known

    @field_validator("rationale", "impact")
    @classmethod
    def _justified(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("an assumption needs a rationale and an impact")
        return v

    def refs(self) -> list[Link]:
        return [*self.links, Link(rel="about", to=self.subject)]


AnalysisType = Literal["static_structural", "thermal_steady", "thermo_structural", "fatigue", "flow"]


class LoadItem(Model):
    """One effect as it enters a load case."""

    effect: str
    interface: str | None = None    # the component interface it acts on (None: the whole body)
    kind: str
    role: str = ""                  # the feature role it acts on
    features: list[str] = Field(default_factory=list)
    si: float | list[float] | None = None
    unit: str = ""                  # SI unit of `si`
    display: str = ""
    h_si: float | None = None       # fluid_temperature: heat transfer coefficient, W/(m^2 K)
    at_si: list[float] | None = None   # force / moment: point of application, m
    factor: float = 1.0
    note: str = ""


class CriterionRef(Model):
    requirement: str
    criterion: int                  # index into the requirement's criteria


class CycleUse(Model):
    """A duty cycle in a fatigue load case: the load cases whose states make the cycle."""

    duty_cycle: str
    count: float
    states: list[str]               # load case ids, in sequence


class LoadCase(Record):
    kind: Literal["load_case"] = "load_case"
    analysis_type: AnalysisType
    component: str
    phase: str | None = None
    statement: str = ""             # what the load case is, in one or two sentences
    loads: list[LoadItem] = Field(default_factory=list)
    supports: list[str] = Field(default_factory=list)       # interface ids that hold the part; empty = free
    criteria: list[CriterionRef] = Field(default_factory=list)
    uses: list[str] = Field(default_factory=list)           # load cases whose results it needs
    cycles: list[CycleUse] = Field(default_factory=list)    # fatigue
    scaled_from: str | None = None  # a proportional load case: results = scale x results of that case, no solve
    scale: float = 1.0
    hand_calcs: list[str] = Field(default_factory=list)
    result_keys: list[str] = Field(default_factory=list)    # the results the analysis must give (criteria, hand calculations)
    status: Literal["selected", "scaled", "not_supported"] = "selected"
    note: str = ""

    def refs(self) -> list[Link]:
        out = list(self.links)
        out += [Link(rel="verifies", to=c.requirement) for c in self.criteria]
        out += [Link(rel="uses", to=x.effect) for x in self.loads]
        out += [Link(rel="uses", to=u) for u in self.uses]
        if self.phase:
            out.append(Link(rel="derives_from", to=self.phase))
        if self.scaled_from:
            out.append(Link(rel="derives_from", to=self.scaled_from))
        return out


class CalcInput(Model):
    name: str
    si: float
    unit: str
    display: str
    source: str = ""                # where the number comes from: an expression, a parameter, a load, a material


class CalcOutput(Model):
    name: str
    si: float
    unit: str
    display: str
    description: str = ""


class CalcStep(Model):
    text: str                       # one sentence
    latex: str = ""                 # the formula
    result: str = ""                # the number, with its unit


class Comparison(Model):
    output: str
    result_key: str
    hand_value: float
    fe_value: float | None = None
    ratio: float | None = None
    band: tuple[float, float] | None = None
    tolerance_si: float | None = None
    agree: bool | None = None       # None: the analysis has no such result (yet)
    why: str = ""


class HandCalcResult(Record):
    kind: Literal["hand_calc"] = "hand_calc"
    spec: str                       # the HandCalcSpec id
    model: str
    load_case: str | None = None
    reference: str = ""             # the textbook or standard the formula comes from
    inputs: list[CalcInput] = Field(default_factory=list)
    outputs: list[CalcOutput] = Field(default_factory=list)
    steps: list[CalcStep] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)    # outside the validity range of the formula
    comparisons: list[Comparison] = Field(default_factory=list)

    def refs(self) -> list[Link]:
        return [*self.links, *([Link(rel="about", to=self.load_case)] if self.load_case else [])]


class BCItem(Model):
    """One support or load of the analysis model, with the decision that justifies it."""

    type: str                       # the solver's type: fixed, displacement, pressure, force, traction, convection, ...
    features: list[str] = Field(default_factory=list)
    faces: list[str] = Field(default_factory=list)       # f<tag>
    value: Any = None               # SI
    effect: str | None = None
    interface: str | None = None
    decision: str | None = None
    description: str = ""           # one sentence


class MeshLevel(Model):
    level: int
    h: float                        # requested global size, m
    nodes: int
    cells: int
    dofs: int = 0
    order: int = 2
    min_quality: float | None = None
    h_rep: float | None = None      # representative cell size, m
    path: str = ""                  # relative to the project's out folder


ResultStatus = Literal["converged", "not_converged", "singular", "not_run", "estimate"]


class ResultValue(Model):
    """One response quantity. key = <quantity>.<measure>@<location>, e.g. von_mises.peak@part."""

    key: str
    value: float | None             # SI
    unit: str = ""
    uncertainty: float | None = None    # relative discretization uncertainty (0.02 = 2 %)
    uncertainty_abs: float | None = None   # the same in the unit of the value; when given it is the one to use
    #                                        (temperatures: a relative error of an absolute temperature hides the error)
    error_estimated: bool = False
    # True when the mesh study gave an error estimate from a sequence that converges (monotone, oscillating
    # with a decreasing change, or two levels). False when the uncertainty is only the spread of values that do
    # not converge: such a number is not a bound, and the compliance stage does not conclude from it.
    status: ResultStatus = "not_run"
    location: list[float] | None = None    # where the extreme is, m
    near: str = ""                  # the feature or edge it is at
    per_level: list[float] = Field(default_factory=list)   # the value on each mesh, coarse to fine
    extrapolated: float | None = None
    order: float | None = None      # observed order of convergence
    method: Literal["fe", "hand_calc", "derived", "scaled"] = "fe"
    note: str = ""


class Check(Model):
    """A check of the analysis itself (equilibrium, mesh quality, hand calculation, energy)."""

    id: str
    kind: str                       # equilibrium | reaction | mesh_quality | hand_calc | error_estimate | energy | ...
    statement: str
    passed: bool | None = None      # None: could not be evaluated
    value: float | None = None
    limit: float | None = None
    link: str | None = None         # a record id (a hand calculation)


class Artifact(Model):
    path: str                       # relative to the project's out folder
    kind: Literal["picture", "mesh", "field", "log", "data"] = "picture"
    caption: str = ""


class AnalysisRecord(Record):
    kind: Literal["analysis"] = "analysis"
    load_case: str
    analysis_type: AnalysisType
    status: Literal["done", "failed", "skipped", "not_supported"] = "done"
    message: str = ""               # why it failed or was skipped
    statement: str = ""             # what was analysed and how, in a few sentences
    solver: str = ""                # backend and version
    material: dict[str, Any] = Field(default_factory=dict)   # the properties used (SI)
    bcs: list[BCItem] = Field(default_factory=list)
    mesh: list[MeshLevel] = Field(default_factory=list)
    results: dict[str, ResultValue] = Field(default_factory=dict)
    checks: list[Check] = Field(default_factory=list)
    artifacts: list[Artifact] = Field(default_factory=list)
    run_ids: list[str] = Field(default_factory=list)       # run-store records (the dashboard's 3D viewer)
    runtime_s: float | None = None

    def refs(self) -> list[Link]:
        return [*self.links, Link(rel="about", to=self.load_case)]


Verdict = Literal["pass", "fail", "inconclusive", "not_assessed"]


class ComplianceItem(Record):
    """One criterion of one requirement against one result of one load case.
    Margin of safety MS = allowable / (factor x response) - 1 for "<=", and
    (response / factor) / limit - 1 for ">=": MS >= 0 passes. The response enters with its uncertainty on
    the unsafe side. `margin` is the margin that decides the verdict: with one times the uncertainty for a
    converged result, and with `policy.unconverged_margin_factor` times it for a result that is over the
    convergence limit but has an error estimate (dossier/compliance.py)."""

    kind: Literal["compliance"] = "compliance"
    requirement: str
    criterion: int
    load_case: str
    analysis: str | None = None
    result_key: str
    value: float | None = None           # the response, SI
    unit: str = ""
    uncertainty: float | None = None
    conservative_value: float | None = None   # the response with its uncertainty on the unsafe side
    factor: float = 1.0
    factor_source: str = ""
    limit: float | None = None           # the limit or allowable, SI
    limit_source: str = ""
    comparator: Literal["<=", ">="] = "<="
    margin: float | None = None
    verdict: Verdict = "not_assessed"
    statement: str = ""                  # the verdict in words
    reasons: list[str] = Field(default_factory=list)     # why it is not a pass, or what the pass rests on
    caveats: list[str] = Field(default_factory=list)     # finding ids that question the requirement

    def refs(self) -> list[Link]:
        out = [*self.links, Link(rel="verifies", to=self.requirement), Link(rel="uses", to=self.load_case)]
        if self.analysis:
            out.append(Link(rel="evidence", to=self.analysis))
        out += [Link(rel="about", to=c, note="open finding") for c in self.caveats]
        return out


class RequirementVerdict(Model):
    requirement: str
    verdict: Verdict
    margin: float | None = None          # the smallest margin of its items
    governing: str | None = None         # the compliance item that governs
    statement: str = ""
    items: list[str] = Field(default_factory=list)


ReviewState = Literal["open", "accepted", "rejected", "changes_requested"]


class Review(Model):
    """A person's judgement of one record (reviews.yaml). `digest` is the content the person saw: when
    the record changes, the review is stale."""

    record: str
    state: ReviewState = "open"
    by: str = ""
    date: str = ""
    comment: str = ""
    digest: str = ""
