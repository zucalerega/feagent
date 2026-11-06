"""Union-find grouping shared by geometry (merging split cylinder faces) and quality (expected
mesh components)."""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable


def connected_groups(nodes: Iterable[int], pairs: Iterable[tuple[int, int]]) -> list[list[int]]:
    """Connected components of the graph ``pairs`` over ``nodes``: each sorted, ordered by smallest
    member. Nodes without a pair form singleton components."""
    parent = {n: n for n in nodes}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in pairs:
        parent[find(b)] = find(a)
    groups: dict[int, list[int]] = defaultdict(list)
    for n in parent:
        groups[find(n)].append(n)
    return sorted((sorted(g) for g in groups.values()), key=lambda g: g[0])
