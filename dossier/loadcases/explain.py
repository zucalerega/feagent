"""The records that say why the plan is what it is: the text of each load case, and the decisions,
assumptions and findings of the planner.

`dossier.loadcases.select` holds the rules and calls these functions with its `Planner` (`p`). Each function
writes one record into `p.journal`:

    Decision    key `selected`       one per load case: the requirements and criteria it serves, the phase, the
                                     kind of analysis and why (with the temperatures when they decide it)
                key `scaled`         one per proportional load case: the scale, why it is exact, and the
                                     alternative of an own solve
                key `not-supported`  one per load case that stays in the plan without an analysis
                key `no-load-case`   one per phase and per requirement that gets no load case
    Assumption  key `steady-state`   one per thermal load case: the steady state is analysed, transient
                                     temperature gradients are not (not conservative for thermal stress)
                key `thermal-strain` a static load case that leaves out a temperature difference under the
                                     policy limit, with the stress this can hide
    Finding     a criterion the plan cannot serve, and a load case that cannot be analysed

The text is ASD-STE100 Simplified Technical English (dossier/CONTRACTS.md): `rationale`, `statement` and
`impact` are descriptive (25 words or fewer in a sentence, six sentences or fewer); `choice`, `verify_by` and
the options of the alternatives are instructions (20 words or fewer). Ids and names an engineer wrote are
quoted as they are.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from dossier import units
from dossier.context import MECHANICAL, proportional
from dossier.loadcases import wording as w
from dossier.model import CycleUse, LoadCase, LoadItem, Phase, Requirement
from dossier.project import make_id

if TYPE_CHECKING:
    from dossier.loadcases.select import Planner

SELECT = "rule:loadcases.select"
PROPORTIONAL = "rule:loadcases.proportional"
COVERAGE = "rule:loadcases.coverage"
THERMAL_RULE = "rule:loadcases.thermal"

STRUCTURAL_TYPES = ("static_structural", "thermo_structural")

ENVELOPE_OPTION = "Use the results of a different load case as an envelope for this phase."
NO_CRITERION = ("A result without a criterion gives no margin. A person must first add a requirement with a "
                "criterion for this phase.")
NO_LOADS = ("Make a load case without loads.", "The analysis of a part without loads gives no information.")


# =========================================================================== the text of a load case


def structural_statement(phase: Phase, mechanical: list[LoadItem], held: list[str], thermal_case: str | None) -> str:
    first = f"This load case is the phase {w.phase_ref(phase)} at limit loads."
    field = f", and the temperature field of {thermal_case}" if thermal_case else ""
    if mechanical:
        loads = w.fit(f"It has {w.count(len(mechanical), 'mechanical load')} on {w.interfaces_of(mechanical)}{field}.",
                      f"It has {w.count(len(mechanical), 'mechanical load')} on {w.interfaces_of(mechanical, most=0)}{field}.")
    else:
        loads = f"It has no mechanical load, and it has the temperature field of {thermal_case}."
    return " ".join([first, loads, w.holds(held)])


def _temperatures(items: list[LoadItem]) -> list[float]:
    return sorted({float(x.si) for x in items if x.kind != "heat_flux" and isinstance(x.si, (int, float))})


def thermal_statement(phase: Phase, items: list[LoadItem]) -> str:
    first = f"This load case is the steady-state temperature field of the part in the phase {w.phase_ref(phase)}."
    if not items:
        return f"{first} The system context gives no thermal load on the part in this phase."
    loads = w.fit(f"It has {w.count(len(items), 'thermal load')} on {w.interfaces_of(items)}.",
                  f"It has {w.count(len(items), 'thermal load')} on {w.interfaces_of(items, most=0)}.")
    temps = _temperatures(items)
    if len(temps) > 1:
        last = f"The temperatures of the thermal loads are from {w.shown(temps[0], 'K')} to {w.shown(temps[-1], 'K')}."
    elif temps:
        last = f"The temperature of the thermal loads is {w.shown(temps[0], 'K')}."
    else:
        last = "The thermal loads are heat fluxes."
    return " ".join([first, loads, last])


def flow_statement(phase: Phase, items: list[LoadItem]) -> str:
    first = f"This load case is the flow of the fluid through the part in the phase {w.phase_ref(phase)}."
    if not items:
        return f"{first} The system context gives no mass flow, pressure or fluid temperature in this phase."
    order = ("mass_flow", "pressure", "fluid_temperature")
    kinds = w.kinds_of(sorted(items, key=lambda x: order.index(x.kind) if x.kind in order else len(order)))
    loads = w.fit(f"It has {kinds} on {w.interfaces_of(items)}.", f"It has {kinds} on {w.interfaces_of(items, most=0)}.")
    flows = [x for x in items if x.kind == "mass_flow"]
    last = (f"The effect {flows[0].effect} gives a mass flow of {flows[0].display}." if flows else
            "The system context gives no mass flow through the part in this phase.")
    return " ".join([first, loads, last])


def fatigue_title(duty_cycles: list[str]) -> str:
    if not duty_cycles:
        return "Fatigue analysis without a duty cycle"
    return w.fit(f"Fatigue analysis of {w.the('duty cycle', duty_cycles)}",
                 f"Fatigue analysis of {len(duty_cycles)} duty cycles", limit=12)


def _counts(cycles: list[CycleUse]) -> str:
    total = sum(c.count for c in cycles)
    return w.fit(f"The system context gives {w.join(w.count(c.count, 'cycle') + ' of ' + c.duty_cycle for c in cycles)}.",
                 f"The system context gives a total of {w.count(total, 'cycle')} in {len(cycles)} duty cycles.")


def fatigue_statement(cycles: list[CycleUse], uses: list[str]) -> str:
    if not cycles:
        return ("This load case is the fatigue damage of the part. The plan has no duty cycle with a stress cycle "
                "and a count for it.")
    ids = [c.duty_cycle for c in cycles]
    first = w.fit(f"This load case is the fatigue damage of the part from {w.the('duty cycle', ids)}.",
                  f"This load case is the fatigue damage of the part from {len(ids)} duty cycles.")
    last = w.fit(f"The states of the duty cycles are {w.the('load case', uses)}.",
                 f"The states of the duty cycles are {len(uses)} structural load cases.")
    return " ".join([first, _counts(cycles), last])


# =========================================================================== helpers


def _criteria(p: "Planner", lc: LoadCase) -> dict[str, list[str]]:
    """requirement id -> the labels of the criteria of the load case."""
    out: dict[str, list[str]] = {}
    for ref in lc.criteria:
        crit = p.project.requirement(ref.requirement).criteria[ref.criterion]
        out.setdefault(ref.requirement, []).append(crit.id or crit.key())
    return out


def _criteria_links(p: "Planner", lc: LoadCase) -> list[tuple[str, str, str]]:
    out = []
    for ref in lc.criteria:
        crit = p.project.requirement(ref.requirement).criteria[ref.criterion]
        out.append(("derives_from", ref.requirement, f"criterion {crit.id or ref.criterion + 1}: {crit.key()}"))
    return out


def _serves(p: "Planner", lc: LoadCase) -> str:
    where = f" in the phase {lc.phase}" if lc.phase else ""
    return w.fit(*[f"The load case gives the results for {c}{where}." for c in w.criteria_of(_criteria(p, lc))])


def _do(what: str, phase: Phase, tail: str = "") -> str:
    return w.fit(f"Do a {what} of the phase {w.phase_ref(phase)}{tail}.", f"Do a {what} of the phase {phase.id}{tail}.",
                 limit=w.INSTRUCTION)


def _no_case(phase: Phase, structural: bool = False) -> str:
    kind = "structural load case" if structural else "load case"
    return w.fit(f"Do not make a {kind} for the phase {w.phase_ref(phase)}.",
                 f"Do not make a {kind} for the phase {phase.id}.", limit=w.INSTRUCTION)


def _kelvin(value: float) -> str:
    return w.shown(value, "K")


def _plural(n: int, one: str, many: str) -> str:
    return one if n == 1 else many


# =========================================================================== decisions: a load case is selected


def selected(p: "Planner", lc: LoadCase) -> None:
    kind = lc.analysis_type
    what = w.ANALYSIS_WORDS[kind]
    phase = p.phases.get(lc.phase) if lc.phase else None
    links: list = _criteria_links(p, lc) + ([("derives_from", lc.phase)] if lc.phase else [])
    if kind in STRUCTURAL_TYPES:
        title = f"Load case {lc.id}: {what} of the phase {lc.phase}"
        choice = _do(what, phase, " at limit loads")
        rationale, alternatives, more = _structural(p, lc)
    elif kind == "thermal_steady":
        title = f"Load case {lc.id}: {what} of the phase {lc.phase}"
        choice = _do(what, phase)
        rationale, alternatives, more = _thermal(p, lc)
    elif kind == "flow":
        title = f"Load case {lc.id}: {what} of the phase {lc.phase}"
        choice = _do(what, phase)
        rationale = [_serves(p, lc),
                     "The pressure drop is a result of the flow through the part, thus the load case is a flow analysis.",
                     "The load case has the mass flow, the pressures and the fluid temperatures of the phase."]
        alternatives = [("Use only a hand calculation of the pressure drop.",
                         "The loss coefficients of a hand calculation are estimates. The flow analysis uses the "
                         "geometry of the part.")]
        more = []
    else:
        ids = [c.duty_cycle for c in lc.cycles]
        title = f"Load case {lc.id}: {what} of {len(ids)} duty {_plural(len(ids), 'cycle', 'cycles')}"
        choice = w.fit(f"Do a {what} of {w.the('duty cycle', ids)}.", f"Do a {what} of {len(ids)} duty cycles.",
                       limit=w.INSTRUCTION)
        rationale, alternatives, more = _fatigue(p, lc)
    p.journal.decide(subject=lc.id, key="selected", title=title, choice=choice, rationale=" ".join(rationale),
                     alternatives=alternatives, links=links + more, by=SELECT)


def _structural(p: "Planner", lc: LoadCase) -> tuple[list[str], list[tuple[str, str]], list]:
    pid = lc.phase
    th = p.thermal[pid]
    users = [f for f in p.cases.values() if f.analysis_type == "fatigue" and lc.id in f.uses]
    cycles = w.unique(c.duty_cycle for f in users for c in f.cycles if lc.id in c.states)
    names = w.join(f.id for f in users)
    out: list[str] = []
    alts: list[tuple[str, str]] = []
    links: list = [("derives_from", d, "this phase is a state of the duty cycle") for d in cycles]
    if lc.criteria:
        out.append(_serves(p, lc))
        if users:
            out.append(w.fit(f"{names} also {_plural(len(users), 'uses', 'use')} the load case as a state of "
                             f"{w.the('duty cycle', cycles)}.",
                             f"{names} also {_plural(len(users), 'uses', 'use')} the load case as a state of "
                             f"{len(cycles)} duty cycles."))
    else:
        out.append(f"No requirement gives a criterion for a stress or a displacement in the phase {pid}.")
        out.append(w.fit(f"{names} {_plural(len(users), 'needs', 'need')} the stress of this phase as a state of "
                         f"{w.the('duty cycle', cycles)}.",
                         f"{names} {_plural(len(users), 'needs', 'need')} the stress of this phase as a state of "
                         f"{len(cycles)} duty cycles."))
        alts.append(("Do not make a load case, because no requirement gives a criterion for this phase.",
                     f"{names} {_plural(len(users), 'needs', 'need')} the stress of each state of a duty cycle."))

    t_ref, limit, diff = _kelvin(p.t_ref), _kelvin(p.t_delta), _kelvin(th.difference)
    policy = ("uses", p.project.policy.id, "thermal_delta")
    reference = ("uses", p.comp.id, "reference_temperature")
    e = th.deciding
    if th.verdict == "over":
        out += [f"The effect {e.effect} gives a {w.kind_word(e.kind)} of {_kelvin(e.si)}, and the component gives a "
                f"reference temperature of {t_ref}.",
                f"The difference of {diff} is more than the policy limit of {limit} (thermal_delta).",
                f"Thus the analysis includes the thermal strain, with the temperature field of {lc.uses[0]}."]
        alts.append(("Do a static structural analysis without thermal strain.",
                     f"The temperature difference of {diff} is more than the policy limit of {limit}. "
                     f"The thermal stress adds to the stress of the mechanical loads."))
        links += [("uses", e.effect, "the largest temperature difference from the reference temperature"),
                  policy, reference]
    elif th.verdict == "flux":
        out += [f"The effect {e.effect} gives a heat flux of {e.display} on the part.",
                "Only a thermal analysis gives the temperature change from a heat flux.",
                f"Thus the analysis includes the thermal strain, with the temperature field of {lc.uses[0]}."]
        alts.append(("Do a static structural analysis without thermal strain.",
                     "The heat flux changes the temperature of the part. The plan cannot show that the change is "
                     "less than the policy limit."))
        links += [("uses", e.effect, "the heat flux"), policy]
    elif th.verdict == "under" and e is not None:
        out += [f"The largest temperature difference from the reference temperature of {t_ref} is {diff} "
                f"(effect {e.effect}).",
                f"This is not more than the policy limit of {limit} (thermal_delta), thus the analysis has no "
                f"thermal strain."]
        alts.append(("Do a thermo-structural analysis.",
                     f"The largest temperature difference of {diff} is not more than the policy limit of {limit}."))
        links += [("uses", e.effect, "the largest temperature difference from the reference temperature"),
                  policy, reference]
    else:
        out.append("The system context gives no thermal effect on the part in this phase, thus the analysis has "
                   "no thermal strain.")
        alts.append(("Do a thermo-structural analysis.",
                     "The system context gives no thermal effect on the part in this phase."))

    scaled_here = [c.id for c in p.cases.values() if c.scaled_from == lc.id]
    if scaled_here:
        out.append(w.fit(f"{w.join(scaled_here)} {_plural(len(scaled_here), 'gets its', 'get their')} results from this "
                         f"load case, thus it also gives {_plural(len(scaled_here), 'its', 'their')} result keys.",
                         f"{len(scaled_here)} load cases get their results from this load case, thus it also gives "
                         f"their result keys."))
        links += [("about", c, "a load case that gets its results from this one") for c in scaled_here]

    others = [c for c in p.cases.values() if c is not lc and c.analysis_type in STRUCTURAL_TYPES]
    if lc.status == "selected" and others:
        if th.significant:
            why = ("The thermal stress of this phase does not change in proportion to the mechanical loads. "
                   "Larger loads do not always give a larger stress, because loads can cancel.")
        else:
            why = ("No load case before this one has proportional loads. Without proportion, larger loads do not "
                   "always give a larger stress, because loads can cancel.")
        alts.append((ENVELOPE_OPTION, why))
    return out, alts, links


def _thermal(p: "Planner", lc: LoadCase) -> tuple[list[str], list[tuple[str, str]], list]:
    users = [c.id for c in p.cases.values() if lc.id in c.uses]
    asm = make_id("ASM", lc.id, "steady-state")
    out: list[str] = []
    if lc.criteria:
        out.append(_serves(p, lc))
    if users:
        out.append(f"{w.join(users)} {_plural(len(users), 'needs', 'need')} the temperature field of this phase for "
                   f"the thermal strain.")
    if not lc.criteria:
        out.append(f"No requirement gives a temperature criterion for the phase {lc.phase}.")
    out.append(f"The assumption {asm} records that the analysis is for the steady state only.")
    alts = [("Do a transient thermal analysis of the phase.",
             f"The plan has only steady-state thermal analyses. The assumption {asm} gives the effect of this limit.")]
    temps = _temperatures(lc.loads)
    if users and len(temps) > 1:
        alts.append(("Apply one uniform temperature to the part.",
                     f"The temperatures of the thermal loads are from {_kelvin(temps[0])} to {_kelvin(temps[-1])}. "
                     f"A uniform temperature gives no stress from temperature differences in the part."))
    return out, alts, [("uses", asm, "the steady state")]


def _fatigue(p: "Planner", lc: LoadCase) -> tuple[list[str], list[tuple[str, str]], list]:
    ids = [c.duty_cycle for c in lc.cycles]
    reqs = w.unique(c.requirement for c in lc.criteria)
    out = [_serves(p, lc),
           w.fit(f"{w.cap(w.the('requirement', reqs))} {_plural(len(reqs), 'names', 'name')} "
                 f"{w.the('duty cycle', ids)}.",
                 f"{w.cap(w.the('requirement', reqs))} {_plural(len(reqs), 'names', 'name')} {len(ids)} duty cycles."),
           _counts(lc.cycles),
           "The states of each duty cycle are the structural load cases of its phases, in sequence.",
           "The counts have no factor: the margin applies the factor of the criterion."]
    gaps = {d: phases for d, phases in p.cycle_gaps.get(lc.id, {}).items()}
    lost = w.unique(ph for phases in gaps.values() for ph in phases)
    dropped = [d for d in gaps if d not in ids]
    if dropped:
        out.append(w.fit(f"{w.cap(w.the('duty cycle', dropped))} {_plural(len(dropped), 'gives', 'give')} no "
                         f"stress cycle, because the part has no load in {_plural(len(dropped), 'its', 'their')} phases.",
                         f"{len(dropped)} duty cycles give no stress cycle, because the part has no load in their phases."))
    elif lost:
        out.append(w.fit(f"{w.cap(w.the('phase', lost))} {_plural(len(lost), 'is', 'are')} not a state, because "
                         f"the part has no load there.",
                         f"{len(lost)} phases are not states, because the part has no load there."))
    alts = [("Do a fatigue analysis of only the duty cycle with the largest stress range.",
             "Each duty cycle adds damage. The requirement names all of these duty cycles.")]
    return out, alts, [("derives_from", d, "a duty cycle of the requirement") for d in ids]


# =========================================================================== decisions: a load case is scaled


def _example_pair(base: LoadCase, lc: LoadCase) -> tuple[LoadItem, LoadItem] | None:
    """One load of the scaled load case and the same load of its base (a pressure reads best)."""
    def key(x: LoadItem):
        return (x.interface, x.kind, x.role)

    theirs = {key(x): x for x in reversed(base.loads)}
    ours = sorted(lc.loads, key=lambda x: x.kind != "pressure")
    for x in ours:
        if key(x) in theirs and x.display and theirs[key(x)].display:
            return theirs[key(x)], x
    return None


def scaled(p: "Planner", lc: LoadCase) -> None:
    base = p.cases[lc.scaled_from]
    k = w.number(lc.scale)
    rationale = [f"The phases {base.phase} and {lc.phase} have the same kinds of mechanical loads on the same interfaces.",
                 f"Each load of {lc.phase} is {k} times the same load of {base.phase}, and the supports are the same."]
    links: list = [("derives_from", base.id, f"the results of this load case, multiplied by {k}")]
    pair = _example_pair(base, lc)
    if pair:
        rationale.append(f"For example, the effect {pair[1].effect} gives {pair[1].display} and the effect "
                         f"{pair[0].effect} gives {pair[0].display}.")
        links += [("uses", pair[1].effect), ("uses", pair[0].effect)]
    rationale += ["The two phases have no thermal effect on the part.",
                  "The structural analysis is linear and static, thus each result changes in proportion to the loads.",
                  f"The phase {base.phase} comes first in the list of phases of the system, thus {base.id} is the "
                  f"base load case."]
    p.journal.decide(
        subject=lc.id, key="scaled", title=f"{lc.id}: the results of {base.id}, multiplied by {k}",
        choice=f"Do not solve {lc.id}. Multiply the results of {base.id} by {k}.",
        rationale=" ".join(rationale),
        alternatives=[(f"Solve {lc.id} with its own mesh study.",
                       "A linear static analysis gives the same results as the multiplication. A second analysis adds "
                       "calculation time and no new information."),
                      (f"Solve {lc.id}, and multiply its results to get {base.id}.",
                       "The results are the same. The rule solves the load case of the phase that comes first in "
                       "the system.")],
        basis="calculation", links=links, by=PROPORTIONAL)


# =========================================================================== assumptions


def _diffusivity(p: "Planner") -> float | None:
    """k / (rho cp) of the material, m^2/s; None when the material does not give the properties."""
    mat = p.project.material_of(p.comp.id)
    if mat.k is None or mat.cp is None:
        return None
    names = p.ctx.names()
    k = units.si(units.to(mat.k.quantity(names), "W/(m*K)"))
    cp = units.si(units.to(mat.cp.quantity(names), "J/(kg*K)"))
    rho = units.si(units.to(mat.rho.quantity(names), "kg/m^3"))
    return float(k) / (float(rho) * float(cp)) if rho and cp else None


def steady_state(p: "Planner", lc: LoadCase) -> None:
    """Every thermal load case is a steady-state analysis. The assumption says so, with the duration of the
    phase when the system context gives one, and with an estimate of how far a temperature change goes into
    the material in that time (the thermal penetration depth, sqrt(a t)). It is not conservative: during a
    fast temperature change the thermal stress can be higher than in the steady state."""
    phase = p.phases[lc.phase]
    users = [c.id for c in p.cases.values() if lc.id in c.uses]
    reqs = w.unique(c.requirement for c in lc.criteria)
    mat = p.project.material_of(p.comp.id)
    statement = [f"The temperature field of the phase {w.phase_ref(phase)} is the steady-state field of its thermal "
                 f"loads.", "The analysis does not calculate the change of the temperature with time."]
    impact = ["During a fast temperature change, the temperature differences in a wall can be larger than in the "
              "steady state."]
    if users:
        impact.append(f"Then the thermal stress is higher than the result of {w.join(users)}, and the margins of "
                      f"{_plural(len(users), 'that load case', 'those load cases')} are not conservative.")
    if reqs:
        impact.append(w.fit(f"The temperature during the phase can be different from the steady-state result for "
                            f"{w.join(reqs)}.",
                            "The temperature during the phase can be different from the steady-state result for the "
                            "temperature criteria."))
    links: list = [("uses", phase.id, "the duration of the phase")]
    if phase.duration is not None:
        seconds = float(units.si(units.to(phase.duration.quantity(p.ctx.names()), "s")))
        duration = w.shown(seconds, "s")
        statement.append(f"The system context gives a duration of {duration} for the phase.")
        a = _diffusivity(p)
        if a:
            depth = units.display(units.from_si((a * seconds) ** 0.5, "m"), sig=2)
            shown_a = units.fmt(units.from_si(a, "m^2/s"), "mm^2/s")
            impact += [f"In {duration}, a temperature change goes approximately {depth} into the material.",
                       f"This estimate is the square root of a t, with the thermal diffusivity a = k / (rho cp) = "
                       f"{shown_a} of {mat.id}.",
                       "A region at a larger distance from the surfaces with thermal loads can be far from the "
                       "steady state."]
            links.append(("uses", mat.id, "thermal conductivity, density and specific heat"))
        else:
            impact.append(f"The material {mat.id} gives no thermal conductivity or no specific heat, thus the plan has "
                          f"no estimate of the time to the steady state.")
    else:
        statement.append("The system context gives no duration for the phase.")
        impact.append("Without a duration, the plan cannot compare the phase with the time that the part needs to get "
                      "to the steady state.")
    p.journal.assume(
        subject=lc.id, key="steady-state", title=f"Steady-state temperature field in the phase {phase.id}",
        statement=" ".join(statement),
        rationale="The plan has no transient thermal analysis. The system context gives the thermal loads of each "
                  "phase, but it gives no initial temperature field and no temperature history. The steady state is "
                  "the condition that the part goes to when the thermal loads stay constant.",
        impact=" ".join(impact),
        verify_by=f"Do a transient thermal analysis of the phase {phase.id}. Start from the temperature field at the "
                  f"end of the phase before it. Compare the largest thermal stress with the steady-state result.",
        conservative=False, links=links, by=THERMAL_RULE)


def no_thermal_strain(p: "Planner", lc: LoadCase) -> None:
    """A static load case in a phase whose temperatures differ from the reference temperature by less than
    the policy limit: the thermal strain is left out. The bound is the stress of a fully restrained part."""
    th = p.thermal[lc.phase]
    diff, limit = _kelvin(th.difference), _kelvin(p.t_delta)
    mat = p.project.material_of(p.comp.id)
    impact = (f"The analysis does not include the thermal stress of this temperature difference. The real stress can "
              f"be higher than the result of {lc.id}.")
    links: list = [("uses", p.project.policy.id, "thermal_delta")]
    if th.deciding is not None:
        links.append(("uses", th.deciding.effect, "the largest temperature difference"))
    if mat.alpha is not None and mat.nu < 1.0:
        names = p.ctx.names()
        e_mod = float(units.si(units.to(mat.E.quantity(names), "Pa")))
        alpha = float(units.si(units.to(mat.alpha.quantity(names), "1/K")))
        stress = w.shown(e_mod * alpha * th.difference / (1.0 - mat.nu), "Pa")
        impact = (f"A part that cannot expand gets a thermal stress of E alpha dT / (1 - nu) = {stress}. "
                  f"Here dT is the temperature difference of {diff}, and E, alpha and nu are properties of {mat.id}. "
                  f"The analysis does not include this stress, thus the real stress can be higher than the result of "
                  f"{lc.id}.")
        links.append(("uses", mat.id, "E, alpha and nu"))
    p.journal.assume(
        subject=lc.id, key="thermal-strain", title=f"No thermal strain in {lc.id}",
        statement=f"The temperature difference of {diff} in the phase {lc.phase} gives a small thermal strain. "
                  f"The analysis of {lc.id} does not include it.",
        rationale=f"The policy gives a limit of {limit} (thermal_delta) for the temperature difference from the "
                  f"reference temperature. Below this limit, a phase gets a static structural analysis.",
        impact=impact,
        verify_by=f"Compare this stress with the smallest margin of {lc.id}. If the stress is important, decrease "
                  f"thermal_delta in the policy and make the plan again.",
        conservative=False, links=links, by=THERMAL_RULE)


# =========================================================================== a load case without an analysis


def not_supported(p: "Planner", lc: LoadCase) -> None:
    finding = p.reasons.get(lc.id)
    rationale = [lc.note]
    if lc.criteria:
        rationale.append(w.fit(*[f"The load case is in the plan for {c}." for c in w.criteria_of(_criteria(p, lc))]))
    rationale.append("It stays in the plan, thus the assessment shows these criteria as not assessed, with the reason.")
    if finding:
        rationale.append(f"The finding {finding} records the problem.")
    p.journal.decide(
        subject=lc.id, key="not-supported", title=f"{lc.id} stays in the plan without an analysis",
        choice=f"Keep {lc.id} in the plan with the status not_supported.",
        rationale=" ".join(rationale),
        alternatives=[("Remove the load case from the plan.", "Then no record says why its criteria have no result.")],
        needs_review=True, links=_criteria_links(p, lc) + ([("about", finding)] if finding else []), by=COVERAGE)


def no_thermal_load(p: "Planner", phase: Phase, lc: LoadCase) -> str:
    reqs = w.unique(c.requirement for c in lc.criteria)
    return p.journal.find(
        rule="plan.no_thermal_load", severity="major", category="coverage", subject=phase.id, key="no-thermal-load",
        title=f"No thermal effect on the part in the phase {phase.id}",
        statement=w.fit(f"{w.cap(w.the('requirement', reqs))} {_plural(len(reqs), 'gives', 'give')} a temperature "
                        f"criterion for the phase {phase.id}. The system context gives no thermal effect on the part "
                        f"in this phase.",
                        f"{len(reqs)} requirements give a temperature criterion for the phase {phase.id}. The system "
                        f"context gives no thermal effect on the part in this phase."),
        rationale="A thermal analysis needs a temperature or a heat flux on a surface of the part. Without one, the "
                  f"plan cannot analyse {lc.id}.",
        suggestion="Add the thermal effects of the phase to the system context, or remove the phase from the "
                   "requirement.",
        links=[("about", r) for r in reqs] + [("about", lc.id)]).id


def no_mass_flow(p: "Planner", phase: Phase, lc: LoadCase) -> str:
    reqs = w.unique(c.requirement for c in lc.criteria)
    return p.journal.find(
        rule="plan.no_mass_flow", severity="major", category="coverage", subject=phase.id, key="no-mass-flow",
        title=f"No mass flow through the part in the phase {phase.id}",
        statement=w.fit(f"{w.cap(w.the('requirement', reqs))} {_plural(len(reqs), 'gives', 'give')} a flow "
                        f"criterion for the phase {phase.id}. The system context gives no mass flow through the part "
                        f"in this phase.",
                        f"{len(reqs)} requirements give a flow criterion for the phase {phase.id}. The system context "
                        f"gives no mass flow through the part in this phase."),
        rationale=f"A flow analysis needs a mass flow. Without one, the plan cannot analyse {lc.id}.",
        suggestion="Add the mass flow of the phase to the system context, or remove the phase from the requirement.",
        links=[("about", r) for r in reqs] + [("about", lc.id)]).id


def no_cycles(p: "Planner", lc: LoadCase, duty_cycles: list[str]) -> str:
    reqs = w.unique(c.requirement for c in lc.criteria)
    subject = reqs[0]
    if not duty_cycles:
        statement = (f"{w.cap(w.the('requirement', reqs))} {_plural(len(reqs), 'gives', 'give')} a fatigue "
                     f"criterion, but {_plural(len(reqs), 'it names', 'they name')} no duty cycle.")
        suggestion = "Give the duty cycles in `applies.duty_cycles` of the requirement."
    else:
        statement = w.fit(f"The fatigue criterion of {w.join(reqs)} names {w.the('duty cycle', duty_cycles)}. "
                          f"No phase of {_plural(len(duty_cycles), 'this duty cycle', 'these duty cycles')} gives a "
                          f"stress cycle with a count.",
                          f"The fatigue criterion of {w.join(reqs)} names {len(duty_cycles)} duty cycles. No phase of "
                          f"these duty cycles gives a stress cycle with a count.")
        suggestion = ("Add the loads of these phases to the system context, or name different duty cycles in the "
                      "requirement.")
    return p.journal.find(
        rule="plan.no_duty_cycle", severity="major", category="coverage", subject=subject, key="no-duty-cycle",
        title=f"No duty cycle for the fatigue criterion of {subject}", statement=statement,
        rationale=f"A fatigue analysis needs duty cycles, their counts and the stress of their states. Without them, "
                  f"the plan cannot analyse {lc.id}.",
        suggestion=suggestion, links=[("about", lc.id)] + [("about", d) for d in duty_cycles]).id


def support_without_interface(p: "Planner", phase: Phase, item: LoadItem) -> str:
    return p.journal.find(
        rule="plan.support", severity="major", category="model", subject=item.effect, key=f"{phase.id}-no-interface",
        title=f"The support {item.effect} has no interface on the part",
        statement=f"The effect {item.effect} is a support of the part in the phase {phase.id}, but the component has "
                  f"no interface for its connection.",
        rationale="A load case names its supports by interface. Without an interface, the analysis does not know "
                  "the faces that hold the part, and the load case does not have this support.",
        suggestion="Add an interface for the connection of this effect to the component file.",
        links=[("about", phase.id)]).id


# =========================================================================== findings: a criterion without a load case


def unknown_quantity(p: "Planner", req: Requirement, index: int, known: list[str]) -> str:
    crit = req.criteria[index]
    label = crit.id or f"{index + 1}"
    return p.journal.find(
        rule="plan.quantity", severity="major", category="model", subject=req.id, key=f"criterion-{index + 1}-quantity",
        title=f"No kind of analysis for the quantity {crit.quantity}",
        statement=f"The criterion {label} of {req.id} has the quantity {crit.quantity}. The plan has no kind of "
                  f"analysis for this quantity, thus it makes no load case for the criterion.",
        rationale=f"The plan knows these quantities: {w.join(known)}.",
        suggestion="Use one of these quantities in the criterion, or give a different verification method.",
        values={"quantity": crit.quantity, "known": list(known)}).id


def no_phase(p: "Planner", req: Requirement, indices: list[int]) -> str:
    return p.journal.find(
        rule="plan.no_phase", severity="major", category="coverage", subject=req.id, key="no-phase",
        title=f"{req.id} applies to no phase",
        statement=f"The requirement {req.id} gives {w.count(len(indices), 'criterion', 'criteria')} for analysis, but "
                  f"it applies to no phase.",
        rationale="A load case is a phase at limit loads. Without a phase, the plan has no loads for "
                  f"{_plural(len(indices), 'this criterion', 'these criteria')}.",
        suggestion="Give the phases or the category of phases in `applies` of the requirement.",
        values={"criteria": [req.criteria[i].key() for i in indices]}).id


# =========================================================================== decisions: no load case


def requirement_left_out(p: "Planner", req: Requirement, reason: str) -> None:
    """`reason`: no-criterion | method | problems (every criterion has a finding of the plan)."""
    choice = w.fit(f"Do not make a load case for the requirement {req.id} ({req.title}).",
                   f"Do not make a load case for the requirement {req.id}.", limit=w.INSTRUCTION)
    title = f"No load case for the requirement {req.id}"
    if reason == "no-criterion":
        finding = make_id("FND", req.id, "not-verifiable")
        p.journal.decide(
            subject=req.id, key="no-load-case", title=title, choice=choice,
            rationale=f"The requirement {req.id} gives analysis as its verification method, but it gives no criterion. "
                      f"A load case needs a criterion: a response quantity, a location and a limit. "
                      f"The finding {finding} records this problem.",
            alternatives=[("Make a criterion from the text of the requirement.",
                           "The requirement gives no quantity and no limit as data. A criterion that the plan makes "
                           "is not a requirement.")],
            needs_review=True, links=[("about", finding)], by=COVERAGE)
    elif reason == "method":
        p.journal.decide(
            subject=req.id, key="no-load-case", title=title, choice=choice,
            rationale=f"The requirement {req.id} gives {w.count(len(req.criteria), 'criterion', 'criteria')}, but its "
                      f"verification method is {req.verification}. The plan makes load cases only for requirements "
                      f"with verification by analysis.",
            alternatives=[("Analyse the criteria of the requirement.",
                           "The specification gives the verification method. A person can change it to analysis.")],
            basis="input", by=COVERAGE)
    else:
        findings = p.problems.get(req.id, [])
        p.journal.decide(
            subject=req.id, key="no-load-case", title=title, choice=choice,
            rationale=f"The plan cannot make a load case for the {w.count(len(req.criteria), 'criterion', 'criteria')} "
                      f"of {req.id}. {w.cap(w.the('finding', findings))} "
                      f"{_plural(len(findings), 'gives', 'give')} the reason.",
            alternatives=[("Make a load case from an assumption about the data that the requirement does not give.",
                           "A load case needs a phase and a known response quantity from the requirement.")],
            needs_review=True, links=[("about", f) for f in findings], by=COVERAGE)


def _loads_sentence(p: "Planner", phase: Phase, mechanical: list[LoadItem]) -> list[str]:
    th = p.thermal[phase.id]
    out = []
    if mechanical:
        out.append(w.fit(f"The system context gives {w.kinds_of(mechanical)} on the part in this phase.",
                         f"The system context gives {w.count(len(mechanical), 'mechanical load')} on the part in "
                         f"this phase."))
    if th.verdict == "over":
        out.append(f"The effect {th.deciding.effect} gives a temperature difference of {_kelvin(th.difference)} from "
                   f"the reference temperature.")
    elif th.verdict == "flux":
        out.append(f"The effect {th.deciding.effect} gives a heat flux of {th.deciding.display} on the part.")
    return out


def _envelope(p: "Planner", pid: str) -> tuple[LoadCase, float] | None:
    """A static load case of the plan whose loads are proportional to the loads of the phase: (the load case,
    the scale from its loads to the loads of the phase). A load case with larger loads (scale <= 1) is first."""
    found = []
    for c in p.cases.values():
        if c.analysis_type == "static_structural" and c.phase:
            s = proportional(p.loads[c.phase], p.loads[pid])
            if s is not None:
                found.append((c, float(s)))
    found.sort(key=lambda x: (x[1] > 1.0 + 1e-9, abs(x[1] - 1.0)))
    return found[0] if found else None


def phase_left_out(p: "Planner", phase: Phase) -> None:
    """One decision for a phase whose loads no structural load case analyses, or that has no load case."""
    pid = phase.id
    if pid in p.structural_id:
        return
    made = [c for c in p.cases.values() if c.phase == pid]
    mechanical = [x for x in p.loads[pid] if x.kind in MECHANICAL]
    loaded = bool(mechanical) or p.thermal[pid].significant
    wanted = pid in p.need["structural"] or pid in p.state_of
    reqs = p.applies[pid]
    title = f"No {'structural ' if made else ''}load case for the phase {pid}"
    choice = _no_case(phase, structural=bool(made))
    other = ([f"The plan has {w.join(c.id for c in made)} for this phase, but {_plural(len(made), 'it gives', 'they give')} "
              f"no stress."] if made else [])
    links: list = [("uses", x.effect) for x in mechanical]

    if loaded and not reqs:                                  # loads that no requirement covers
        finding = make_id("FND", pid, "uncovered")
        rationale = _loads_sentence(p, phase, mechanical)
        rationale.append("No requirement applies to the phase, thus the plan has no criterion for a result of this phase.")
        alternatives = [("Make a load case for the phase without a criterion.", NO_CRITERION)]
        env = _envelope(p, pid)
        covered = env is not None and env[1] <= 1.0 + 1e-9
        if covered:
            rationale.append(f"The loads of the phase are {w.number(env[1])} times the loads of {env[0].phase}, thus "
                             f"the results of {env[0].id} are an envelope for this phase.")
            links.append(("uses", env[0].id, "the envelope of this phase"))
        else:
            if env is not None:
                rationale.append(f"The loads of the phase are {w.number(env[1])} times the loads of {env[0].phase}, "
                                 f"thus {env[0].id} is not an envelope for this phase.")
            else:
                rationale.append("No load case of the plan has loads in proportion to the loads of this phase.")
                alternatives.append((ENVELOPE_OPTION,
                                     "The loads of this phase are not proportional to the loads of a different phase. "
                                     "Larger loads do not always give a larger stress, because loads can cancel."))
            rationale.append(f"The finding {finding} records this problem.")
            links.append(("about", finding))
        p.journal.decide(subject=pid, key="no-load-case", title=title, choice=choice, rationale=" ".join(rationale + other),
                         alternatives=alternatives, needs_review=not covered, links=links, by=COVERAGE)
    elif loaded:                                             # requirements apply, but none asks for a stress
        rationale = _loads_sentence(p, phase, mechanical)
        rationale.append(w.fit(f"{w.cap(w.the('requirement', reqs))} {_plural(len(reqs), 'applies', 'apply')} "
                               f"to the phase.", f"{len(reqs)} requirements apply to the phase."))
        rationale.append(f"No criterion of {_plural(len(reqs), 'this requirement', 'these requirements')} is a limit on "
                         f"stress or displacement with verification by analysis.")
        p.journal.decide(subject=pid, key="no-load-case", title=title, choice=choice, rationale=" ".join(rationale + other),
                         alternatives=[("Make a structural load case for the phase without a criterion.", NO_CRITERION)],
                         needs_review=True, links=links + [("uses", r) for r in reqs], by=COVERAGE)
    elif wanted:                                             # a criterion asks for a stress, but there is no load
        rationale = [w.fit(f"{w.cap(w.the('requirement', reqs))} {_plural(len(reqs), 'needs', 'need')} the stress "
                           f"or the displacement of the part in this phase.",
                           f"{len(reqs)} requirements need the stress or the displacement of the part in this phase."),
                     "The system context gives no mechanical load on the part in this phase, and no temperature "
                     "difference above the policy limit.",
                     "A linear analysis without loads gives zero stress and zero displacement.",
                     "A person must make sure that the system context is complete for this phase."]
        p.journal.decide(subject=pid, key="no-load-case", title=f"No structural load case for the phase {pid}",
                         choice=_no_case(phase, structural=True), rationale=" ".join(rationale), alternatives=[NO_LOADS],
                         needs_review=True, links=[("uses", r) for r in reqs], by=COVERAGE)
    elif not made:                                           # no load, and no criterion that needs an analysis
        rationale = ["The system context gives no load on the part in this phase."]
        rationale.append(w.fit(f"{w.cap(w.the('requirement', reqs))} {_plural(len(reqs), 'applies', 'apply')} to "
                               f"the phase, but no criterion needs an analysis of it.",
                               f"{len(reqs)} requirements apply to the phase, but no criterion needs an analysis of it.")
                         if reqs else "No requirement applies to the phase.")
        p.journal.decide(subject=pid, key="no-load-case", title=title, choice=choice, rationale=" ".join(rationale),
                         alternatives=[NO_LOADS], links=[("uses", r) for r in reqs], by=COVERAGE)
