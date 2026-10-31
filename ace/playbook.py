"""The playbook: a JSON list of short, voted-on bullets that gets pasted into the agent's prompt.

Each bullet has a stable ID (e.g. "mesh-0003") so the agent can cite it and the Reflector can
vote on it. The playbook is only ever changed by ace.merge (deterministic code), never by an LLM.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ace.config import SECTIONS, TOP_N_BULLETS


@dataclass
class Bullet:
    id: str                     # "<prefix>-<4 digits>", prefix from config.SECTIONS
    section: str
    content: str                # "When X, do Y, because Z."
    helpful: int = 0
    harmful: int = 0
    source_runs: list[str] = field(default_factory=list)
    created_gen: int = 0

    @property
    def net(self) -> int:
        return self.helpful - self.harmful


@dataclass
class Playbook:
    bullets: list[Bullet] = field(default_factory=list)

    # ---- persistence
    @classmethod
    def load(cls, path: Path) -> "Playbook":
        path = Path(path)
        if not path.exists():
            return cls()
        return cls([Bullet(**b) for b in json.loads(path.read_text())["bullets"]])

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps({"bullets": [asdict(b) for b in self.bullets]}, indent=2) + "\n")

    # ---- lookup
    def get(self, bullet_id: str) -> Bullet | None:
        return next((b for b in self.bullets if b.id == bullet_id), None)

    def in_section(self, section: str) -> list[Bullet]:
        return [b for b in self.bullets if b.section == section]

    def next_id(self, section: str) -> str:
        prefix = SECTIONS[section]
        used = [int(b.id.split("-")[1]) for b in self.bullets if b.id.startswith(prefix + "-")]
        return f"{prefix}-{max(used, default=0) + 1:04d}"

    # ---- prompt text
    def render(self, top_n: int = TOP_N_BULLETS) -> str:
        """Top-N bullets by (helpful - harmful), grouped by section, each prefixed by its ID.
        Ties keep the older bullet first. Empty playbook -> empty string."""
        top = sorted(self.bullets, key=lambda b: -b.net)[:top_n]
        lines = []
        for section in SECTIONS:
            chosen = [b for b in top if b.section == section]
            if chosen:
                lines.append(f"## {section}")
                lines += [f"[{b.id}] {b.content}" for b in chosen]
        return "\n".join(lines)
