"""Two kinds of memory: one SHARED playbook every agent sees, and one PRIVATE playbook per agent.

Agents are persistent identities (a0, a1, ...). Every generation each agent attempts every
problem; its prompt holds the shared playbook plus its own private notes.

  private (low bar)  any evidence-backed lesson (a confirmed counterfactual or a best-vs-worst
                     contrast) from the diagnosis of that agent's own runs; pruned quickly
                     (harmful >= helpful + PRIVATE_PRUNE_MARGIN). No LLM involved: deduplicated
                     by text similarity only.
  shared (high bar)  only lessons confirmed by a counterfactual rerun (through the Curator), or an
                     idea that PROMOTE_MIN_AGENTS different agents learned on their own (promotion);
                     pruned only on stronger evidence (harmful >= helpful + SHARED_PRUNE_MARGIN).

Private bullet IDs carry the agent's prefix ("a3-mesh-0002") so a citation or a vote reaches the
right memory.
"""

from __future__ import annotations

import re
from pathlib import Path

from ace.config import AGENTS, PRIVATE_PRUNE_MARGIN, PROMOTE_MIN_AGENTS, SHARED_PRUNE_MARGIN
from ace.curator import Op, curate
from ace.merge import apply_ops, find_duplicate
from ace.playbook import Playbook

AGENT_OF_RUN = re.compile(r"-a(\d+)$")       # run ids end in "-a<k>"


def agent_of(run_id: str) -> int | None:
    m = AGENT_OF_RUN.search(run_id)
    return int(m.group(1)) if m else None


class Memory:
    def __init__(self, n_agents: int = AGENTS, shared: Playbook | None = None):
        self.shared = shared or Playbook()
        self.private = [Playbook(id_prefix=f"a{k}-") for k in range(n_agents)]

    # ---- what an agent sees
    def prompt_text(self, k: int, top_n: int) -> str:
        shared = self.shared.render(top_n) or "(empty)"
        own = self.private[k].render(top_n) or "(empty)"
        return f"## SHARED PLAYBOOK (all agents)\n{shared}\n\n## YOUR OWN NOTES (only you see these)\n{own}"

    def get(self, bullet_id: str):
        """A bullet from whichever memory holds it (lets the diagnoser read private bullets too)."""
        pb = self.owner(bullet_id)
        return pb.get(bullet_id) if pb is not None else None

    def owner(self, bullet_id: str):
        k = agent_of_bullet(bullet_id)
        return self.shared if k is None else (self.private[k] if k < len(self.private) else None)

    # ---- learning from one generation's diagnoses
    def update(self, reflections: list[tuple], llm, generation: int, records: list | None = None) -> dict:
        """reflections: [(problem_id, Diagnosis, run_ids)]. Returns a change log."""
        log = {"shared": {}, "private": {}, "promoted": []}

        # 1. Shared: only counterfactual-confirmed lessons go to the Curator (the higher bar).
        strict = [(pid, d.model_copy(update={"lessons": [les for les in d.lessons
                                                         if getattr(les, "evidence_type", None) == "counterfactual"]}),
                   rids) for pid, d, rids in reflections]
        ops, _ = curate(strict, self.shared, llm, records=records)
        log["shared"] = apply_ops(self.shared, ops, generation, prune_margin=SHARED_PRUNE_MARGIN)

        # 2. Private: each evidence-backed lesson goes to the agents whose runs support it.
        per_agent: dict[int, list[Op]] = {}
        for _pid, d, rids in reflections:
            for les in d.lessons:
                if les.scope == "one_off" or getattr(les, "evidence_type", None) not in ("counterfactual", "contrast"):
                    continue
                agents = {agent_of(r) for r in (getattr(les, "run_ids", None) or rids)} - {None}
                for k in agents:
                    per_agent.setdefault(k, []).append(Op("ADD", section=les.section, content=les.content,
                                                          source_runs=list(getattr(les, "run_ids", []) or []),
                                                          evidence_type=les.evidence_type))
            for v in d.bullet_votes:                       # votes on private bullets
                k = agent_of_bullet(v.bullet_id)
                if k is not None and v.vote != "neutral" and k < len(self.private):
                    per_agent.setdefault(k, []).append(Op("VOTE", bullet_id=v.bullet_id, vote=v.vote))
        for k, kops in per_agent.items():
            log["private"][k] = apply_ops(self.private[k], kops, generation, prune_margin=PRIVATE_PRUNE_MARGIN)

        # 3. Promotion: an idea several agents learned on their own becomes shared.
        log["promoted"] = self.promote(generation)
        return log

    def promote(self, generation: int) -> list:
        promoted = []
        for k, pb in enumerate(self.private):
            for b in list(pb.bullets):
                holders = [j for j, other in enumerate(self.private)
                           if j == k or find_duplicate(other, b.section, b.content)[0] is not None]
                if len(holders) < PROMOTE_MIN_AGENTS or find_duplicate(self.shared, b.section, b.content)[0]:
                    continue
                ch = apply_ops(self.shared, [Op("ADD", section=b.section, content=b.content, source_runs=b.source_runs,
                                                evidence_type=b.evidence_type)], generation,
                               prune_margin=SHARED_PRUNE_MARGIN)
                for j in holders:                          # now shared: drop the private copies
                    dup = b if j == k else find_duplicate(self.private[j], b.section, b.content)[0]
                    if dup is not None and dup in self.private[j].bullets:
                        self.private[j].bullets.remove(dup)
                promoted.append({"from_agents": holders, "content": b.content,
                                 "shared_id": ch["added"][0][0] if ch["added"] else None})
        return promoted

    # ---- persistence
    def save(self, out: Path, generation: int) -> None:
        self.shared.save(Path(out) / f"playbook_gen{generation}.json")
        pdir = Path(out) / "private"
        pdir.mkdir(exist_ok=True)
        for k, pb in enumerate(self.private):
            pb.save(pdir / f"agent{k}_gen{generation}.json")


def agent_of_bullet(bullet_id: str) -> int | None:
    m = re.match(r"a(\d+)-", bullet_id)
    return int(m.group(1)) if m else None
