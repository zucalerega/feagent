"""Two kinds of memory: one SHARED playbook every agent sees, and one PRIVATE playbook per agent.

Agents are persistent identities (a0, a1, ...). Every generation each agent attempts every
problem; its prompt holds the shared playbook plus its own private notes.

  private (low bar)  every general lesson from the diagnosis of that agent's own runs, the teacher's
                     advice to it, and the teacher's untested "ideas to try" for it (to make agents
                     explore and differ); only near-identical notes merge (PRIVATE_DEDUP_THRESHOLD),
                     pruned slowly (harmful >= helpful + PRIVATE_PRUNE_MARGIN), more of them shown
                     (PRIVATE_TOP_N). No LLM involved.
  shared             by default only lessons confirmed by a counterfactual rerun (through the
                     Curator); with shared_bar="any" every general lesson of the teacher. Plus any
                     idea PROMOTE_MIN_AGENTS different agents learned on their own (promotion);
                     pruned only on stronger evidence (harmful >= helpful + SHARED_PRUNE_MARGIN).

Private bullet IDs carry the agent's prefix ("a3-mesh-0002") so a citation or a vote reaches the
right memory.

With several tasks (`tasks=[...]`), there is one shared playbook PER TASK (IDs "p0-mesh-0001",
"p1-...": the task's index), and each agent's private notes follow it across tasks: an agent sees
the shared playbook of the task it is on plus its own notes. The teacher (diagnoser) also writes
advice addressed to single agents (Diagnosis.agent_advice); that goes straight to their notes.
"""

from __future__ import annotations

import re
from pathlib import Path

from ace.config import (AGENTS, PRIVATE_DEDUP_THRESHOLD, PRIVATE_PRUNE_MARGIN, PRIVATE_TOP_N, PROMOTE_MIN_AGENTS,
                        SHARED_PRUNE_MARGIN)
from ace.curator import Op, curate
from ace.merge import apply_ops, find_duplicate
from ace.playbook import Playbook

AGENT_OF_RUN = re.compile(r"-a(\d+)$")       # run ids end in "-a<k>"
TASK_OF_RUN = re.compile(r"^g\d+-(.+)-a\d+$")  # run ids are "g<gen>-<problem>-a<k>"


def agent_of(run_id: str) -> int | None:
    m = AGENT_OF_RUN.search(run_id)
    return int(m.group(1)) if m else None


class Memory:
    def __init__(self, n_agents: int = AGENTS, shared: Playbook | None = None, tasks: list[str] | None = None):
        self.tasks = list(tasks) if tasks else [None]
        if tasks:
            self.shared_by_task = {t: Playbook(id_prefix=f"p{i}-") for i, t in enumerate(self.tasks)}
        else:
            self.shared_by_task = {None: shared or Playbook()}
        self.shared = self.shared_by_task[self.tasks[0]]      # the only one when there is one task
        self.private = [Playbook(id_prefix=f"a{k}-") for k in range(n_agents)]

    def _shared_for(self, task):
        return self.shared_by_task.get(task, self.shared) if task is not None else self.shared

    # ---- what an agent sees
    def prompt_text(self, k: int, top_n: int, task: str | None = None) -> str:
        shared = self._shared_for(task).render(top_n) or "(empty)"
        own = self.private[k].render(max(top_n, PRIVATE_TOP_N)) or "(empty)"
        return f"## SHARED PLAYBOOK (all agents)\n{shared}\n\n## YOUR OWN NOTES (only you see these)\n{own}"

    def get(self, bullet_id: str):
        """A bullet from whichever memory holds it (lets the diagnoser read private bullets too)."""
        pb = self.owner(bullet_id)
        return pb.get(bullet_id) if pb is not None else None

    def owner(self, bullet_id: str):
        m = re.match(r"p(\d+)-", bullet_id)
        if m:
            i = int(m.group(1))
            return self.shared_by_task[self.tasks[i]] if i < len(self.tasks) else None
        k = agent_of_bullet(bullet_id)
        return self.shared if k is None else (self.private[k] if k < len(self.private) else None)

    # ---- learning from one generation's diagnoses
    def update(self, reflections: list[tuple], llm, generation: int, records: list | None = None,
               shared_bar: str = "counterfactual") -> dict:
        """reflections: [(problem_id, Diagnosis, run_ids)]. Returns a change log; with several tasks
        log["shared"] is {task: changes}. shared_bar: which of the teacher's overall lessons go to the
        Curator for the shared playbook: "counterfactual" (confirmed by a rerun) or "any" (all of its
        general lessons, so more of the teacher's feedback reaches every agent)."""
        log = {"shared": {}, "private": {}, "promoted": []}

        # 1. Shared: only counterfactual-confirmed lessons go to the Curator (the higher bar), into
        #    the shared playbook of the task they came from.
        for task in self.tasks:
            mine = [(pid, d, rids) for pid, d, rids in reflections if task is None or pid == task]
            keep = ((lambda les: getattr(les, "evidence_type", None) == "counterfactual") if shared_bar == "counterfactual"
                    else (lambda les: les.scope != "one_off"))
            strict = [(pid, d.model_copy(update={"lessons": [les for les in d.lessons if keep(les)]}), rids)
                      for pid, d, rids in mine]
            pb = self.shared_by_task[task]
            ops, _ = curate(strict, pb, llm, records=records) if mine else ([], [])
            changes = apply_ops(pb, ops, generation, prune_margin=SHARED_PRUNE_MARGIN)
            if task is None:
                log["shared"] = changes
            else:
                log["shared"][task] = changes

        # 2. Private: each evidence-backed lesson goes to the agents whose runs support it.
        per_agent: dict[int, list[Op]] = {}
        for _pid, d, rids in reflections:
            for les in d.lessons:
                if les.scope == "one_off":
                    continue
                support = list(getattr(les, "run_ids", None) or rids)   # no run_ids given: all the problem's runs
                agents = {agent_of(r) for r in support} - {None}
                for k in agents:
                    per_agent.setdefault(k, []).append(Op("ADD", section=les.section, content=les.content,
                                                          source_runs=support,
                                                          evidence_type=les.evidence_type))
            for adv in getattr(d, "agent_advice", None) or []:   # the teacher's advice to one agent
                k = int(adv.agent.lstrip("a")) if adv.agent.lstrip("a").isdigit() else None
                if k is not None and k < len(self.private):
                    src = list(adv.run_ids or []) or [r for r in rids if agent_of(r) == k]
                    per_agent.setdefault(k, []).append(Op("ADD", section=adv.section, content=adv.content,
                                                          source_runs=src, evidence_type="advice"))
                    for idea in getattr(adv, "ideas_to_try", None) or []:   # untested ideas: explore
                        per_agent[k].append(Op("ADD", section=adv.section, content=f"Idea to try: {idea}",
                                               source_runs=src, evidence_type="idea"))
            for v in d.bullet_votes:                       # votes on private bullets
                k = agent_of_bullet(v.bullet_id)
                if k is not None and v.vote != "neutral" and k < len(self.private):
                    per_agent.setdefault(k, []).append(Op("VOTE", bullet_id=v.bullet_id, vote=v.vote))
        for k, kops in per_agent.items():
            log["private"][k] = apply_ops(self.private[k], kops, generation, prune_margin=PRIVATE_PRUNE_MARGIN,
                                          dedup_threshold=PRIVATE_DEDUP_THRESHOLD)

        # 3. Promotion: an idea several agents learned on their own becomes shared.
        log["promoted"] = self.promote(generation)
        return log

    def promote(self, generation: int) -> list:
        promoted = []
        for k, pb in enumerate(self.private):
            for b in list(pb.bullets):
                holders = [j for j, other in enumerate(self.private)
                           if j == k or find_duplicate(other, b.section, b.content)[0] is not None]
                target = self._promotion_target(b)
                if target is None or len(holders) < PROMOTE_MIN_AGENTS or find_duplicate(target, b.section, b.content)[0]:
                    continue
                ch = apply_ops(target, [Op("ADD", section=b.section, content=b.content, source_runs=b.source_runs,
                                                evidence_type=b.evidence_type)], generation,
                               prune_margin=SHARED_PRUNE_MARGIN)
                for j in holders:                          # now shared: drop the private copies
                    dup = b if j == k else find_duplicate(self.private[j], b.section, b.content)[0]
                    if dup is not None and dup in self.private[j].bullets:
                        self.private[j].bullets.remove(dup)
                promoted.append({"from_agents": holders, "content": b.content,
                                 "shared_id": ch["added"][0][0] if ch["added"] else None})
        return promoted

    def _promotion_target(self, bullet):
        """The shared playbook a private idea is promoted into: the task of the runs it came from."""
        if self.tasks == [None]:
            return self.shared
        for rid in bullet.source_runs or []:
            m = TASK_OF_RUN.match(rid)
            if m and m.group(1) in self.shared_by_task:
                return self.shared_by_task[m.group(1)]
        return None

    # ---- persistence
    def save(self, out: Path, generation: int) -> None:
        if self.tasks == [None]:
            self.shared.save(Path(out) / f"playbook_gen{generation}.json")
        else:
            for t, pb in self.shared_by_task.items():
                pb.save(Path(out) / f"playbook_{t}_gen{generation}.json")
        pdir = Path(out) / "private"
        pdir.mkdir(exist_ok=True)
        for k, pb in enumerate(self.private):
            pb.save(pdir / f"agent{k}_gen{generation}.json")


def agent_of_bullet(bullet_id: str) -> int | None:
    m = re.match(r"a(\d+)-", bullet_id)
    return int(m.group(1)) if m else None
