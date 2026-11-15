"""Select the load cases of the component from its requirements and the system context.

    plan(project, ctx, geo) -> dossier.pipeline.Plan(load_cases, hand_calcs, journal)

Load-case selection is where an analysis gets skipped without a trace. So the planner follows fixed rules,
and every load case it selects, scales or leaves out is a Decision with its reason and the alternatives
(`dossier.loadcases.explain` writes the records; this module holds the rules).

The rules
---------
1. Requirements give the needs. A requirement counts when its verification method is analysis and it
   has criteria. The quantity of a criterion gives the kind of analysis (`FAMILY`): stress and displacement
   -> structural; temperature, heat_flow -> thermal; pressure_drop -> flow; damage -> fatigue. The phases are
   `project.phases_of(requirement)`; a fatigue criterion takes the duty cycles of `applies.duty_cycles`.
2. A load case is one phase at limit loads (factor 1.0 on every load). The loads come from the system
   context, `ctx.loads(project, component, phase)`; the planner never reads a load from a requirement.
3. One structural load case per phase that needs one, `LC-<phase>`. It carries the mechanical loads of the
   phase (pressure, force, moment, acceleration, displacement); `supports` lists the interfaces that hold the
   part (a support that does not act on the default role `seat` is also in `loads`, with its role and
   features). It is thermo-structural when a thermal effect of the phase differs from the component's
   reference temperature by more than `policy.thermal_delta` (a heat flux always counts: only a thermal
   analysis gives its temperature); then it uses the thermal load case of the phase. Otherwise it is static.
4. One thermal load case per phase that needs one, `LC-TH-<phase>` (steady state), for a temperature
   criterion or for a thermo-structural load case. It carries the thermal loads. An Assumption records that
   the steady state is analysed and that transient temperature gradients are not.
5. One flow load case per phase with a flow criterion, `LC-FLOW-<phase>`: the mass flow, the pressures and
   the fluid temperatures of the phase.
6. One fatigue load case, `LC-FATIGUE`, per set of duty cycles that a fatigue criterion names: one CycleUse
   per duty cycle, with the count of the system context and the structural load cases of its phases as
   states. The structural load case of a state is made even when no other requirement covers its phase.
7. Proportional load cases. When `dossier.context.proportional` gives a scale between two static structural
   load cases (the same kinds of loads on the same interfaces, every value the same multiple, the same
   supports, no thermal effect in either phase), the load case of the phase that comes first in the system
   is solved and the other is scaled from it (`scaled_from`, `scale`, `status="scaled"`). A linear static
   analysis makes this exact. The base load case also gets the result keys of the load cases scaled from it.
8. Every criterion is listed in the load case that serves it (`criteria`), and `result_keys` lists the
   results the analysis must give: the keys of the criteria and of the hand-calculation comparisons.

What the planner does NOT do: it does not drop a load case because another load case looks more severe.
Larger loads do not always give a larger stress: loads can cancel, and a different load path can govern
at a different location. A load case is left to another one only when its loads are proportional, and then it
stays in the plan as a scaled load case with its own criteria.

What gets no load case, and where that is recorded
--------------------------------------------------
* a requirement with verification by analysis and no criterion: a Decision, linked to the finding of the
  validation stage (`FND-<requirement>-NOT-VERIFIABLE`);
* a requirement with criteria and a verification method other than analysis: a Decision;
* a phase in which the part has loads and that no requirement covers: a Decision that needs review, linked
  to the finding `FND-<phase>-UNCOVERED`; when the loads are a smaller multiple of the loads of a load case
  of the plan, the Decision names that load case as the envelope;
* a phase with loads for which the requirements give no criterion on stress or displacement: a Decision;
* a criterion the planner cannot serve (an unknown quantity, no phase, no duty cycle): a Finding;
* a load case that cannot be analysed (a temperature criterion in a phase without a thermal effect, a
  flow criterion in a phase without a mass flow, a fatigue criterion without a usable duty cycle) stays in
  the plan with `status="not_supported"` and the reason in `note`, with a Finding and a Decision, so that
  its criteria stay visible as not assessed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from dossier import units
from dossier.context import MECHANICAL, THERMAL, Context, proportional
from dossier.loadcases import explain, handcalcs
from dossier.loadcases import wording as w
from dossier.model import (DEFAULT_ROLE, CriterionRef, CycleUse, DutyCycle, Link, LoadCase, LoadItem, Origin, Phase,
                           Requirement)
from dossier.pipeline import Plan, StageError, analysis_order
from dossier.project import Project, make_id
from dossier.records import Journal

BY = "engine:loadcases"

# the quantity of a criterion -> the kind of analysis that gives it
STRUCTURAL_QUANTITIES = ("von_mises", "max_principal", "sxx", "syy", "szz", "displacement", "radial_displacement")
FAMILY: dict[str, str] = {**{q: "structural" for q in STRUCTURAL_QUANTITIES}, "temperature": "thermal",
                          "heat_flow": "thermal", "pressure_drop": "flow", "damage": "fatigue"}
STRUCTURAL_TYPES = ("static_structural", "thermo_structural")
FLOW_KINDS = ("mass_flow", "pressure", "fluid_temperature")     # the effects a flow load case carries


@dataclass
class ThermalState:
    """The thermal effects of one phase on the part, against the policy limit `thermal_delta`."""

    items: list[LoadItem] = field(default_factory=list)
    verdict: str = "none"                # none (no thermal effect) | under | over | flux
    difference: float = 0.0              # the largest |T - T_ref| of the phase, K
    deciding: LoadItem | None = None     # the effect that decides the verdict

    @property
    def significant(self) -> bool:
        """The structural load case of the phase needs the temperature field."""
        return self.verdict in ("over", "flux")


def short(phase_id: str) -> str:
    """The phase id without its "PH-" (PH-FIRE -> FIRE)."""
    return phase_id[3:] if phase_id.upper().startswith("PH-") and len(phase_id) > 3 else phase_id


class Planner:
    """The state of one planning run. `plan()` drives it; `explain` reads it to write the journal."""

    def __init__(self, project: Project, ctx: Context, geo: dict | None):
        self.project, self.ctx, self.geo = project, ctx, geo
        self.comp = project.subject
        self.journal = Journal("plan", BY)
        self.phases: dict[str, Phase] = {p.id: p for p in project.system.phases}
        self.duty_cycles: dict[str, DutyCycle] = {d.id: d for d in project.system.duty_cycles}
        self.loads: dict[str, list[LoadItem]] = {pid: ctx.loads(project, self.comp.id, pid) for pid in self.phases}
        names = ctx.names()
        self.t_ref = float(units.si(units.to(self.comp.reference_temperature.quantity(names), "K")))
        self.t_delta = float(units.si(project.policy.thermal_delta.quantity(names)))
        self.thermal: dict[str, ThermalState] = {pid: self._thermal_state(x) for pid, x in self.loads.items()}

        # needs: kind of analysis -> phase id -> the criteria that need a load case of that kind there
        self.need: dict[str, dict[str, list[CriterionRef]]] = {"structural": {}, "thermal": {}, "flow": {}}
        self.fatigue_need: dict[tuple[str, ...], list[CriterionRef]] = {}    # duty cycles -> criteria
        self.state_of: dict[str, list[str]] = {}         # phase id -> duty cycles whose fatigue needs its stress
        self.applies: dict[str, list[str]] = {pid: [] for pid in self.phases}   # phase id -> requirements that name it
        self.strength_phases: list[str] = []             # phases that a requirement of kind strength applies to
        self.skipped: list[tuple[Requirement, str]] = []     # requirements without a load case, with the reason
        self.problems: dict[str, list[str]] = {}         # requirement id -> findings of the plan about it

        self.cases: dict[str, LoadCase] = {}             # in the order of the plan
        self.made_by: dict[str, str] = {}                # load case id -> what it is (for the id clash message)
        self.structural_id: dict[str, str] = {}          # phase id -> its structural load case
        self.thermal_id: dict[str, str] = {}             # phase id -> its thermal load case
        self.cycle_gaps: dict[str, dict[str, list[str]]] = {}   # fatigue case -> duty cycle -> phases without a state
        self.cycle_named: dict[str, list[str]] = {}      # fatigue case -> the duty cycles its requirements name
        self.reasons: dict[str, str] = {}                # a load case that is not supported -> the finding that says why

    # ------------------------------------------------------------------ the thermal effects of a phase
    def _thermal_state(self, items: list[LoadItem]) -> ThermalState:
        thermal = [x for x in items if x.kind in THERMAL]
        if not thermal:
            return ThermalState()
        temps = [x for x in thermal if x.kind != "heat_flux" and isinstance(x.si, (int, float))]
        fluxes = [x for x in thermal if x.kind == "heat_flux" and isinstance(x.si, (int, float)) and x.si != 0.0]
        worst = max(temps, key=lambda x: abs(x.si - self.t_ref), default=None)
        diff = abs(worst.si - self.t_ref) if worst is not None else 0.0
        if diff > self.t_delta * (1.0 + 1e-9):          # "more than" the limit; the limit itself is not
            return ThermalState(thermal, "over", diff, worst)
        if fluxes:
            return ThermalState(thermal, "flux", diff, max(fluxes, key=lambda x: abs(x.si)))
        return ThermalState(thermal, "under", diff, worst)

    # ------------------------------------------------------------------ 1. what the requirements need
    def collect(self) -> None:
        for req in self.project.reqs(self.comp.id):
            phases = [p for p in self.project.phases_of(req) if p in self.phases]
            named = list(phases)
            for dc_id in req.applies.duty_cycles:
                named += self.duty_cycles[dc_id].sequence if dc_id in self.duty_cycles else []
            for pid in w.unique(named):
                if pid in self.applies:
                    self.applies[pid].append(req.id)
            if req.kind == "strength":
                self.strength_phases += [pid for pid in phases if pid not in self.strength_phases]
            if req.verification != "analysis":
                if req.criteria:
                    self.skipped.append((req, "method"))
                continue
            if not req.criteria:
                self.skipped.append((req, "no-criterion"))
                continue
            no_phase: list[int] = []
            for i, crit in enumerate(req.criteria):
                ref = CriterionRef(requirement=req.id, criterion=i)
                family = FAMILY.get(crit.quantity)
                if family is None:
                    self.problems.setdefault(req.id, []).append(explain.unknown_quantity(self, req, i, list(FAMILY)))
                elif family == "fatigue":
                    self.fatigue_need.setdefault(tuple(w.unique(req.applies.duty_cycles)), []).append(ref)
                elif not phases:
                    no_phase.append(i)
                else:
                    for pid in phases:
                        self.need[family].setdefault(pid, []).append(ref)
            if no_phase:
                self.problems.setdefault(req.id, []).append(explain.no_phase(self, req, no_phase))
        for dcs in self.fatigue_need:
            for dc_id in dcs:
                for pid in self.duty_cycles[dc_id].sequence if dc_id in self.duty_cycles else []:
                    users = self.state_of.setdefault(pid, [])
                    if dc_id not in users:
                        users.append(dc_id)

    # ------------------------------------------------------------------ 2. the load cases of each phase
    def _new_id(self, what: str, *parts: str) -> str:
        lc_id = make_id("LC", *parts)
        if lc_id in self.made_by:
            raise StageError("plan", f"the id {lc_id} names two load cases: {self.made_by[lc_id]} and {what}. "
                                     f"Rename a phase so that its id does not look like another load case "
                                     f"(LC-TH-<phase>, LC-FLOW-<phase>, LC-FATIGUE).")
        self.made_by[lc_id] = what
        return lc_id

    def _keys(self, refs: list[CriterionRef]) -> list[str]:
        return w.unique(self.project.requirement(r.requirement).criteria[r.criterion].key() for r in refs)

    def _case(self, lc_id: str, kind: str, title: str, statement: str, phase: str | None, loads: list[LoadItem],
              criteria: list[CriterionRef], **more: Any) -> LoadCase:
        lc = LoadCase(id=lc_id, title=title, analysis_type=kind, component=self.comp.id, phase=phase,
                      statement=statement, loads=[x.model_copy() for x in loads], criteria=list(criteria),
                      result_keys=self._keys(criteria), origin=Origin(by=BY, stage="plan"), **more)
        self.cases[lc.id] = lc
        return lc

    def phase_cases(self) -> None:
        for pid, phase in self.phases.items():
            th = self.thermal[pid]
            mechanical = [x for x in self.loads[pid] if x.kind in MECHANICAL]
            wanted = pid in self.need["structural"] or pid in self.state_of
            structural = wanted and (bool(mechanical) or th.significant)
            if pid in self.need["thermal"] or (structural and th.significant):
                self._thermal_case(phase)
            if structural:
                self._structural_case(phase, mechanical)
            if pid in self.need["flow"]:
                self._flow_case(phase)

    def _structural_case(self, phase: Phase, mechanical: list[LoadItem]) -> None:
        pid = phase.id
        supports = [x for x in self.loads[pid] if x.kind == "support"]
        held = w.unique(x.interface for x in supports if x.interface)
        thermal_case = self.thermal_id[pid] if self.thermal[pid].significant else None
        kind = "thermo_structural" if thermal_case else "static_structural"
        # `supports` names an interface, and the analysis then holds the features of its default role. A support
        # on another role (the bolt holes, not the seat face) goes into `loads` too, so that its role and its
        # features reach the analysis.
        default = DEFAULT_ROLE.get("support", "seat")
        other_role = [x for x in supports if x.interface and x.role and x.role != default]
        links = [Link(rel="uses", to=x.effect, note=f"the support at {x.interface}") for x in supports
                 if x.interface and x not in other_role]
        links += [Link(rel="derives_from", to=dc, note="a state of this duty cycle") for dc in self.state_of.get(pid, [])]
        lc = self._case(self._new_id(f"the structural load case of {pid}", short(pid)), kind,
                        f"{phase.name}: {w.ANALYSIS_WORDS[kind]}",
                        explain.structural_statement(phase, mechanical, held, thermal_case), pid,
                        mechanical + other_role, self.need["structural"].get(pid, []), supports=held,
                        uses=[thermal_case] if thermal_case else [], links=links)
        if other_role:
            ids = [x.effect for x in other_role]
            lc.note = (f"The loads include {w.the('support effect', ids)}, because "
                       f"{'it does' if len(ids) == 1 else 'they do'} not act on the role {default} of "
                       f"{'its' if len(ids) == 1 else 'their'} interface.")
        self.structural_id[pid] = lc.id
        for x in supports:
            if not x.interface:
                explain.support_without_interface(self, phase, x)

    def _thermal_case(self, phase: Phase) -> None:
        pid = phase.id
        items = self.thermal[pid].items
        lc = self._case(self._new_id(f"the thermal load case of {pid}", "TH", short(pid)), "thermal_steady",
                        f"{phase.name}: {w.ANALYSIS_WORDS['thermal_steady']}", explain.thermal_statement(phase, items),
                        pid, items, self.need["thermal"].get(pid, []))
        self.thermal_id[pid] = lc.id
        if not items:
            lc.status = "not_supported"
            lc.note = (f"The system context gives no thermal effect on the part in the phase {pid}. "
                       f"A thermal analysis needs a temperature or a heat flux on a surface of the part.")
            self.reasons[lc.id] = explain.no_thermal_load(self, phase, lc)

    def _flow_case(self, phase: Phase) -> None:
        pid = phase.id
        items = [x for x in self.loads[pid] if x.kind in FLOW_KINDS]
        lc = self._case(self._new_id(f"the flow load case of {pid}", "FLOW", short(pid)), "flow",
                        f"{phase.name}: {w.ANALYSIS_WORDS['flow']}", explain.flow_statement(phase, items), pid, items,
                        self.need["flow"].get(pid, []))
        if not any(x.kind == "mass_flow" for x in items):
            lc.status = "not_supported"
            lc.note = (f"The system context gives no mass flow through the part in the phase {pid}. "
                       f"A flow analysis needs a mass flow.")
            self.reasons[lc.id] = explain.no_mass_flow(self, phase, lc)

    # ------------------------------------------------------------------ 3. proportional load cases
    def scale_proportional(self) -> None:
        """Among the static structural load cases, in the order of the phases of the system: a load case
        whose loads are a multiple of the loads of an earlier solved one is scaled from it."""
        bases: list[LoadCase] = []
        for lc in self.cases.values():
            if lc.analysis_type != "static_structural" or lc.status != "selected":
                continue
            for base in bases:
                scale = proportional(self.loads[base.phase], self.loads[lc.phase])
                if scale is not None:
                    lc.scaled_from, lc.scale, lc.status = base.id, float(scale), "scaled"
                    lc.note = " ".join(x for x in (lc.note, f"The analysis does not solve this load case. Its results "
                                                            f"are {w.number(scale)} times the results of {base.id}.") if x)
                    break
            else:
                bases.append(lc)

    # ------------------------------------------------------------------ 4. fatigue
    def fatigue_cases(self) -> None:
        for n, (dcs, refs) in enumerate(self.fatigue_need.items(), start=1):
            cycles: list[CycleUse] = []
            gaps: dict[str, list[str]] = {}
            for dc_id in dcs:
                dc = self.duty_cycles.get(dc_id)
                count = self.ctx.duty_cycles.get(dc_id)
                if dc is None or count is None:
                    gaps[dc_id] = []
                    continue
                states = [self.structural_id[p] for p in dc.sequence if p in self.structural_id]
                missing = [p for p in dc.sequence if p not in self.structural_id]
                if missing or not states:
                    gaps[dc_id] = missing
                if states:
                    cycles.append(CycleUse(duty_cycle=dc_id, count=float(count), states=states))
            uses = w.unique(s for c in cycles for s in c.states)
            lc = self._case(self._new_id("the fatigue load case", "FATIGUE", *([str(n)] if n > 1 else [])), "fatigue",
                            explain.fatigue_title([c.duty_cycle for c in cycles] or list(dcs)),
                            explain.fatigue_statement(cycles, uses), None, [], refs, uses=uses, cycles=cycles,
                            links=[Link(rel="derives_from", to=d, note="a duty cycle of the requirement") for d in dcs])
            self.cycle_gaps[lc.id] = gaps
            self.cycle_named[lc.id] = list(dcs)
            if not cycles:
                lc.status = "not_supported"
                lc.note = ("The requirement names no duty cycle. A fatigue analysis needs the duty cycles and their "
                           "counts from the system context." if not dcs else
                           "No duty cycle of the requirement has a phase with loads on the part, or the system "
                           "context gives no count for it. A fatigue analysis needs a stress cycle with a count.")
                self.reasons[lc.id] = explain.no_cycles(self, lc, list(dcs))

    # ------------------------------------------------------------------ 5. hand calculations, result keys
    def hand_calculations(self, engine: Any = None) -> list:
        results = handcalcs.apply(self.project, self.ctx, self.geo, list(self.cases.values()), self.journal, engine)
        for lc in self.cases.values():          # a scaled load case gets its results from its base
            if lc.scaled_from:
                base = self.cases[lc.scaled_from]
                base.result_keys += [k for k in lc.result_keys if k not in base.result_keys]
        return results

    # ------------------------------------------------------------------ 6. the journal
    def write_journal(self) -> None:
        for lc in self.cases.values():
            if lc.status == "not_supported":
                explain.not_supported(self, lc)
                continue
            explain.selected(self, lc)
            if lc.scaled_from:
                explain.scaled(self, lc)
            if lc.analysis_type == "thermal_steady":
                explain.steady_state(self, lc)
            th = self.thermal.get(lc.phase or "")
            if lc.analysis_type == "static_structural" and th and th.verdict == "under" and th.difference > 0.0:
                explain.no_thermal_strain(self, lc)
        for phase in self.phases.values():
            explain.phase_left_out(self, phase)
        served = {c.requirement for lc in self.cases.values() for c in lc.criteria}
        for req, reason in self.skipped:
            explain.requirement_left_out(self, req, reason)
        for req in self.project.reqs(self.comp.id):
            if req.verification == "analysis" and req.criteria and req.id not in served:
                explain.requirement_left_out(self, req, "problems")

    def check(self) -> None:
        """Every load case a load case uses is in the plan, and no load case uses itself through others."""
        for lc in self.cases.values():
            for dep in [*lc.uses, *([lc.scaled_from] if lc.scaled_from else [])]:
                if dep not in self.cases:
                    raise StageError("plan", f"{lc.id} uses the load case {dep}, which is not in the plan")
        try:
            analysis_order(list(self.cases.values()))
        except StageError as e:
            raise StageError("plan", str(e)) from e


def plan(project: Project, ctx: Context, geo: dict | None = None) -> Plan:
    """The load cases of the component under verification, their hand calculations, and the journal of
    the choices. `geo` is the dict of the geometry stage; the hand calculations read it (None is accepted:
    a hand calculation that needs a measured feature then fails and becomes a finding)."""
    return _plan(project, ctx, geo)


def _plan(project: Project, ctx: Context, geo: dict | None, handcalc: Any = None) -> Plan:
    """`handcalc` replaces `dossier.handcalc` (tests: a stand-in with `applies` and `evaluate`)."""
    p = Planner(project, ctx, geo)
    p.collect()
    p.phase_cases()
    p.scale_proportional()
    p.fatigue_cases()
    p.check()
    hand_calcs = p.hand_calculations(handcalc)
    p.write_journal()
    return Plan(load_cases=list(p.cases.values()), hand_calcs=hand_calcs, journal=p.journal)
