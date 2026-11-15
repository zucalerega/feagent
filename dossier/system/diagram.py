"""The system diagram: the component in its system, as one self-contained SVG document.

    render(project, ctx) -> str                              the SVG document (text)
    effects_table(project, ctx, component_id=None) -> rows   one row for each effect on the component and phase
    layout(project, ctx) -> Layout                           the geometry behind the SVG (boxes, lines, labels)

What it shows
    Every element is a box, in the lane of its `group`. The element that the component under verification
    realizes has a heavy accent border and the mark "under verification". Every connection is a line with
    an arrowhead from `from` to `to`, drawn in the style of its kind and labelled with its name and medium.
    A connection of the component also shows its effects on the component, counted by kind over all phases
    ("pressure 7 · force 3"). The legend explains the line styles.

Layout (deterministic, no external program)
    Columns   left to right by the depth of an element along the fluid connections. Elements without a
              fluid connection go to the column of the element they connect to.
    Lanes     one horizontal band for each `group`, ordered so that connections between lanes are short.
    Rows      a lane gets a second row when two of its elements want the same column.
    Hints     `Element.position: [column, row]` pins an element: the column of the diagram and the row
              inside its lane, both counted from 0.
    Lines     run at right angles in the channels between the boxes, on separate tracks, and never
              through a box. Labels do not cover boxes, lines or other labels; the channels grow until they fit.

Hooks for the report and the dashboard (stable)
    <svg class="dz-diagram" data-system="SYS-ID" data-subject="EL-ID" data-component="CMP-ID" viewBox="...">
        no width and no height: the page gives the size, the drawing scales.
    <g id="EL-ID" class="dz-element dz-kind-<kind>" data-kind="pump" data-group="..." data-effects="...">
        one for each element. The element of the component also has class `dz-subject`,
        data-subject="true" and data-component="CMP-ID".
    <g id="IF-ID" class="dz-connection dz-kind-<kind>" data-kind="fluid" data-from="EL-A" data-to="EL-B"
       data-medium="LOX" data-effects="...">
        one for each connection. A connection of the component also has class `dz-subject` and
        data-subject="true". Inside: path.dz-hit (a wide, invisible line to point at), path.dz-line,
        g.dz-label.
    data-effects     the effects on the component, counted by kind over the phases, as `kind:count` pairs
                     with spaces between them: "pressure:7 force:3 fluid_temperature:4". For an element:
                     the effects of its connections to the component; for the element of the component:
                     all of them. Empty when there is none.
    <title>          the first child of each group: name, kind, description, and for a connection the
                     number of effects by kind. Browsers show it as a tooltip.
    <g class="dz-lane" data-group="...">, <g class="dz-legend">, <g class="dz-header">
    Classes a page can set on a group: `dz-active` (emphasis) and `dz-dim` (faded). The SVG has the styles.
    Each id is the record id, so a page that puts the SVG inline must not use these ids a second time.

Colours
    Every colour is a CSS variable with a fallback for a light page, `var(--dz-fg, #0b0b0b)`. A page that
    puts the SVG inline sets the variables to change the theme; `VARIABLES` lists each one with its light
    and dark value, and `css_variables("dark")` gives the declarations. Three kinds have a hue (fluid,
    thermal, environment: a set that people with a colour-vision deficiency can tell apart), the other
    kinds are neutral. Each kind also has its own line pattern, and each line has its name, so colour is
    never the only sign. No font and no image comes from outside the document.
"""

from __future__ import annotations

import heapq
import itertools
import re
from dataclasses import dataclass, field
from statistics import median_low
from xml.sax.saxutils import escape, quoteattr

from dossier.context import Context
from dossier.project import Project

# --------------------------------------------------------------------------- style

# CSS variable -> (light value = the fallback in the SVG, dark value for the page to set)
VARIABLES: dict[str, tuple[str, str]] = {
    "--dz-bg": ("#fcfcfb", "#1a1a19"),              # the canvas
    "--dz-fg": ("#0b0b0b", "#f0efec"),              # names, the title
    "--dz-muted": ("#52514e", "#c3c2b7"),           # secondary text
    "--dz-faint": ("#898781", "#898781"),           # effect counts, notes
    "--dz-lane": ("#f4f3ef", "#222221"),            # the band of a lane
    "--dz-lane-head": ("#eae9e3", "#2a2a28"),       # the cell with the name of a lane
    "--dz-rule": ("#d9d8d0", "#3a3a37"),            # hairlines
    "--dz-box": ("#ffffff", "#2c2c2a"),             # an element
    "--dz-box-stroke": ("#b4b3a8", "#5d5d57"),
    "--dz-accent": ("#6250d6", "#9085e9"),          # the component under verification
    "--dz-accent-bg": ("#efedfb", "#2f2c4d"),
    "--dz-accent-fg": ("#ffffff", "#15141f"),       # text on the accent colour
    "--dz-fluid": ("#2a78d6", "#3987e5"),
    "--dz-thermal": ("#eb6834", "#d95926"),
    "--dz-environment": ("#1baf7a", "#199e70"),
    "--dz-mechanical": ("#52514e", "#c3c2b7"),
    "--dz-signal": ("#52514e", "#c3c2b7"),
    "--dz-electrical": ("#52514e", "#c3c2b7"),
}
FONT = 'system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif'

# connection kind -> (stroke width, dash pattern, round caps, double line)
KINDS: dict[str, tuple[float, str, bool, bool]] = {
    "fluid": (2.4, "", False, False),
    "mechanical": (4.4, "", False, True),
    "thermal": (2.0, "7 4", False, False),
    "signal": (1.8, "0.1 4.6", True, False),
    "electrical": (1.6, "9 3 2 3", False, False),
    "environment": (2.0, "3 3.5", False, False),
}
KIND_ORDER = tuple(KINDS)
EFFECT_KINDS = ("pressure", "force", "moment", "acceleration", "fluid_temperature", "temperature", "heat_flux",
                "mass_flow", "displacement", "support", "signal")
EFFECT_WORDS = {"pressure": "pressure", "force": "force", "moment": "moment", "acceleration": "acceleration",
                "fluid_temperature": "temperature", "temperature": "wall temperature", "heat_flux": "heat flux",
                "mass_flow": "flow", "displacement": "displacement", "support": "support", "signal": "signal"}


def css_variables(mode: str = "light") -> str:
    """The CSS declarations of a theme ("light" or "dark"), for the page that puts the SVG inline:
    `.report.dark svg.dz-diagram { <css_variables("dark")> }`."""
    i = {"light": 0, "dark": 1}[mode]
    return " ".join(f"{k}: {v[i]};" for k, v in VARIABLES.items())


def _var(name: str) -> str:
    return f"var({name}, {VARIABLES[name][0]})"


# --------------------------------------------------------------------------- sizes

BOX_W, BOX_H = 168.0, 64.0
TRACK = 11.0                    # distance between two parallel lines
CHAN_X, CHAN_Y = 52.0, 38.0     # smallest gap between two columns, two rows
CHAN_EDGE = 22.0                # smallest gap at the outer sides
CLEAR = 13.0                    # between a line in a channel and a box
LANE_TOP, LANE_BOTTOM = 34.0, 14.0     # padding of a lane above its first row, below its last row
MARGIN = 14.0
ARROW = 9.0                     # length of an arrowhead
RADIUS = 6.0                    # corners of a line
FS_NAME, FS_KIND, FS_LABEL, FS_SMALL, FS_TITLE = 12.5, 10.5, 11.0, 10.0, 14.0
LH_LABEL, LH_SMALL = 13.0, 12.0
LABEL_W = 136.0                 # widest line of a connection label
LABEL_GAP = 5.0                 # between a label and its line
TAG = "UNDER VERIFICATION"      # the mark in the box of the component

# routing costs, in pixels of line length
STEP_X, STEP_Y = (BOX_W + CHAN_X) / 2, (BOX_H + CHAN_Y) / 2
BEND, CROSS, SHARE = 95.0, 70.0, 22.0
PORT_SIDE, PORT_TOP, FLUID_TOP = 34.0, 16.0, 60.0

# --------------------------------------------------------------------------- text

# widths of characters in 1/100 em: the larger of two common screen fonts, so that an estimate is never
# too small (a label that is drawn narrower than its estimate still fits)
_WIDTHS: dict[str, int] = {}
for _w, _chars in ((28, "ijl"), (30, "'I"), (32, " ,."), (34, ":;|·"), (36, "/\\f"), (40, "!()[]t"), (42, "r"),
                   (46, '"-²³'), (50, "*_`–"), (52, "s"), (54, "?"), (56, "Lz"), (58, "Jck"), (60, "Fvxyµ"),
                   (62, "aeo"), (64, "0123456789$EPSTbdghnpqu{}"), (68, "ABKVXYZ"), (70, "R"), (72, "C"), (74, "U"),
                   (76, "DHN"), (78, "&GOQ"), (82, "w"), (84, "#+<=>^~"), (90, "M…"), (98, "%Wm→"), (100, "@")):
    for _c in _chars:
        _WIDTHS[_c] = _w


def text_width(text: str, size: float, bold: bool = False) -> float:
    """An upper estimate of the width of a text in pixels."""
    return sum(_WIDTHS.get(c, 66) for c in text) / 100.0 * size * (1.14 if bold else 1.02)


def _wrap(text: str, width: float, size: float, bold: bool = False, max_lines: int = 3) -> list[str]:
    """Break a text into lines of at most `width` pixels; the last line gets "…" when the text is longer."""
    words = text.split()
    lines: list[str] = []
    cur = ""
    for word in words:
        trial = f"{cur} {word}".strip()
        if cur and text_width(trial, size, bold) > width:
            lines.append(cur)
            cur = word
        else:
            cur = trial
    if cur:
        lines.append(cur)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1] + "…"
    out = []
    for line in lines:
        while len(line) > 2 and text_width(line, size, bold) > width:      # one word that is too long
            line = line[:-2].rstrip("…") + "…"
        out.append(line)
    return out or [""]


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "none"


# --------------------------------------------------------------------------- effects on the component


def _counts(effects) -> dict[str, int]:
    out: dict[str, int] = {}
    for k in EFFECT_KINDS:
        n = sum(1 for e in effects if e.kind == k)
        if n:
            out[k] = n
    for e in effects:                                   # a kind this module does not know keeps its name
        if e.kind not in EFFECT_KINDS:
            out[e.kind] = out.get(e.kind, 0) + 1
    return out


def effect_counts(project: Project, ctx: Context, component_id: str | None = None) -> dict[str, dict[str, int]]:
    """connection id -> {effect kind: number of effects on the component, counted over the phases}."""
    el = project.components[component_id or project.file.subject].element
    return {c.id: _counts([e for e in ctx.effects if e.on == el and e.connection == c.id])
            for c in project.system.connections}


def _summary(counts: dict[str, int]) -> list[str]:
    return [f"{EFFECT_WORDS.get(k, k.replace('_', ' '))} {n}" for k, n in counts.items()]


def _data_effects(counts: dict[str, int]) -> str:
    return " ".join(f"{k}:{n}" for k, n in counts.items())


def effects_table(project: Project, ctx: Context, component_id: str | None = None) -> list[dict]:
    """One row for each effect on a component (default: the component under verification) and phase, in
    the order of the phases, then of the connections and effects in the system model.

    Keys: phase, phase_name, phase_category, connection, connection_name, connection_kind, medium,
    from_element, to_element, interface, interface_name, role, features, kind, display (the value as
    people read it), si, unit, h (the heat transfer coefficient as people read it, or None), h_si, at_si,
    expr (the expression the value comes from, or None), depends_on, basis, source, note, description,
    effect (the effect id), and what the value rests on: sources (the parameters without an expression
    that the value comes from: name, display, basis, source) and assumed (the assumed ones of these)."""
    from dossier import units
    from dossier.model import DEFAULT_ROLE

    comp = project.components[component_id or project.file.subject]
    connections = {c.id: c for c in project.system.connections}
    interfaces = {it.connection: it for it in comp.interfaces}
    phases = {ph.id: (i, ph) for i, ph in enumerate(project.system.phases)}

    def leaves(names: list[str]) -> list[str]:
        out: list[str] = []
        todo = list(names)
        seen = set()
        while todo:
            n = todo.pop(0)
            if n in seen or n not in ctx.params:
                continue
            seen.add(n)
            deps = ctx.params[n].depends_on
            if deps:
                todo += deps
            else:
                out.append(n)
        return out

    rows = []
    for e in ctx.effects:
        if e.on != comp.element:
            continue
        c = connections[e.connection]
        it = interfaces.get(e.connection)
        role = e.role or DEFAULT_ROLE.get(e.kind, "")
        rest = [ctx.params[n] for n in leaves(e.depends_on)]
        rows.append({
            "phase": e.phase, "phase_name": phases[e.phase][1].name, "phase_category": phases[e.phase][1].category,
            "connection": c.id, "connection_name": c.name, "connection_kind": c.kind, "medium": c.medium,
            "from_element": c.from_, "to_element": c.to,
            "interface": it.id if it is not None else None, "interface_name": it.name if it is not None else "",
            "role": role, "features": list(it.features.get(role, [])) if it is not None else [],
            "kind": e.kind, "display": e.display, "si": e.si, "unit": e.unit,
            "h": units.display(units.from_si(e.h_si, "W/(m^2*K)")) if e.h_si is not None else None, "h_si": e.h_si,
            "at_si": e.at_si, "expr": e.expr, "depends_on": list(e.depends_on),
            "basis": e.basis, "source": e.source, "note": e.note, "description": e.description, "effect": e.id,
            "sources": [{"name": p.name, "display": p.display, "basis": p.basis, "source": p.source} for p in rest],
            "assumed": [p.name for p in rest if p.basis == "assumed"],
        })
    rows.sort(key=lambda r: phases[r["phase"]][0])           # stable: the order of the model inside a phase
    return rows


# --------------------------------------------------------------------------- the layout, as data


@dataclass
class Box:
    id: str
    name: str
    kind: str
    group: str
    description: str = ""
    subject: bool = False
    component: str = ""
    col: int = 0
    row: int = 0                # row of the whole diagram
    lane: int = 0
    x: float = 0.0
    y: float = 0.0
    w: float = BOX_W
    h: float = BOX_H
    lines: list[str] = field(default_factory=list)
    effects: dict[str, int] = field(default_factory=dict)

    @property
    def rect(self) -> tuple[float, float, float, float]:
        return (self.x, self.y, self.x + self.w, self.y + self.h)


@dataclass
class Block:
    """A piece of a label: lines of text of one size, and where the layout put it."""
    lines: list[tuple[str, str]]            # (text, css class)
    size: float
    line_height: float
    w: float = 0.0
    h: float = 0.0
    x: float = 0.0                          # left
    y: float = 0.0                          # top
    anchor: str = "middle"                  # how the text sits in the block: start | middle | end

    @property
    def rect(self) -> tuple[float, float, float, float]:
        return (self.x, self.y, self.x + self.w, self.y + self.h)


@dataclass
class Seg:
    """A straight piece of a route, on one line of the routing grid."""
    edge: int
    horizontal: bool
    line: int                   # grid index of the line it is on (v when horizontal, u when vertical)
    s: int                      # grid coordinate where it starts, in the direction of the connection
    t: int                      # and where it ends
    first: bool = False         # it starts at the box of `from`
    last: bool = False          # it ends at the box of `to`
    level: int = 0
    offset: float = 0.0         # pixels away from the line of the grid

    @property
    def span(self) -> tuple[float, float]:
        """The part of the line it uses. Next to a box it stops at the box; at a corner it reaches a
        little past the corner, so that two pieces that meet at one corner count as side by side."""
        lo, hi = (self.s, self.t) if self.s < self.t else (self.t, self.s)
        forward = self.s < self.t
        lo_box = self.first if forward else self.last
        hi_box = self.last if forward else self.first
        return (lo + 0.5 if lo_box else lo - 0.25, hi - 0.5 if hi_box else hi + 0.25)


@dataclass
class Edge:
    id: str
    name: str
    kind: str
    medium: str
    src: str
    dst: str
    description: str = ""
    subject: bool = False                   # it is a connection of the component
    effects: dict[str, int] = field(default_factory=dict)
    nodes: list[tuple[int, int]] = field(default_factory=list)       # the route on the grid
    segs: list[Seg] = field(default_factory=list)
    points: list[tuple[float, float]] = field(default_factory=list)  # the line, in pixels
    blocks: list[Block] = field(default_factory=list)                # the label
    label_ok: bool = True
    label_cut: int | None = None            # the first row of the lane whose border goes through the label


@dataclass
class Lane:
    group: str
    rows: tuple[int, int]       # first and last row of the diagram
    y0: float = 0.0
    y1: float = 0.0
    lines: list[str] = field(default_factory=list)


@dataclass
class Layout:
    width: float
    height: float
    boxes: dict[str, Box]
    edges: list[Edge]
    lanes: list[Lane]
    panel: tuple[float, float, float, float]         # the area of the lanes
    head_w: float                                    # width of the cells with the lane names
    title: list[tuple[str, str]]                     # (text, css class)
    legend: list[dict]
    legend_y: float
    crossings: int = 0


# --------------------------------------------------------------------------- placement


def _lane_order(groups: list[str], lane_of: dict[str, int], connections, subject: str) -> list[int]:
    """The order of the lanes, top to bottom, that makes the connections between lanes short. A connection
    of the component counts three times. With equal length, the order of the system model wins."""
    n = len(groups)
    pairs = [(lane_of[c.from_], lane_of[c.to], 3 if subject in (c.from_, c.to) else 1) for c in connections
             if lane_of[c.from_] != lane_of[c.to]]
    if n < 3 or n > 7 or not pairs:
        return list(range(n))
    best, best_cost = None, None
    for perm in itertools.permutations(range(n)):          # in dictionary order: the first best is the nearest
        where = {lane: i for i, lane in enumerate(perm)}
        cost = sum(w * abs(where[a] - where[b]) for a, b, w in pairs)
        if best_cost is None or cost < best_cost:
            best, best_cost = perm, cost
    return list(best)


def _fluid_depths(members: list[str], edges: list[tuple[str, str]],
                  other: list[tuple[str, str]] = ()) -> dict[str, int]:
    """Depth of each element of one fluid network: the longest way from a source. A loop is cut where
    the search, started in the order of the model, comes back to an element it is still in. `other` are
    the connections of the network that are not fluid: an element at the end of a fluid line goes as far
    right as the elements they join it to (an actuator goes to the column of its valve)."""
    succ: dict[str, list[str]] = {m: [] for m in members}
    for a, b in edges:
        if b not in succ[a]:
            succ[a].append(b)
    state: dict[str, int] = {}
    kept: list[tuple[str, str]] = []

    def visit(v: str):
        state[v] = 1
        for w in succ[v]:
            if state.get(w) == 1:
                continue                                  # the edge closes a loop: leave it out
            kept.append((v, w))
            if w not in state:
                visit(w)
        state[v] = 2

    has_pred = {b for _, b in edges}
    for m in [m for m in members if m not in has_pred] + members:
        if m not in state:
            visit(m)
    depth = {m: 0 for m in members}
    for _ in members:                                     # longest path, by relaxation
        changed = False
        for a, b in kept:
            if depth[b] < depth[a] + 1:
                depth[b] = depth[a] + 1
                changed = True
        if not changed:
            break
    preds = {b for _, b in kept}
    for m in members:                                     # a source goes next to what it feeds
        nxt = [depth[b] for a, b in kept if a == m]
        if m not in preds and nxt:
            depth[m] = min(nxt) - 1
    for m in members:
        if not any(a == m for a, _ in kept):
            near = [depth[b if a == m else a] for a, b in other if m in (a, b)]
            if near and median_low(near) > depth[m]:
                depth[m] = median_low(near)
    return depth


def _place(system, subject: str) -> tuple[dict[str, tuple[int, int]], list[str], dict[str, int]]:
    """Where each element goes: {id: (column, row of the diagram)}, the groups in the order of the lanes,
    and {id: lane}."""
    elements = system.elements
    index = {e.id: i for i, e in enumerate(elements)}
    connections = [c for c in system.connections if c.from_ != c.to and c.from_ in index and c.to in index]
    groups = list(dict.fromkeys(e.group for e in elements))
    first_lane = {e.id: groups.index(e.group) for e in elements}
    order = _lane_order(groups, first_lane, connections, subject)
    where = {lane: i for i, lane in enumerate(order)}
    lane_of = {e.id: where[first_lane[e.id]] for e in elements}
    groups = [groups[i] for i in order]
    anchor_lane = lane_of.get(subject, 0)

    col: dict[str, int] = {}
    row: dict[str, int] = {}                              # row inside the lane; can be negative for now
    cells: dict[tuple[int, int, int], str] = {}           # (lane, row, column) -> element

    def put(eid: str, c: int, r: int):
        col[eid], row[eid] = c, r
        cells[(lane_of[eid], r, c)] = eid

    def rows_to_try(lane: int, start: int = 0) -> list[int]:
        """Rows of a lane in the order of preference: away from the lane of the component."""
        if lane < anchor_lane:
            return [start - i for i in range(0, 40)] + [start + i for i in range(1, 40)]
        if lane > anchor_lane:
            return [start + i for i in range(0, 40)] + [start - i for i in range(1, 40)]
        return [start] + [start + s * i for i in range(1, 40) for s in (1, -1)]

    for e in elements:                                    # the hints of the engineer come first
        if e.position is not None:
            c, r = int(round(e.position[0])), int(round(e.position[1]))
            while (lane_of[e.id], r, c) in cells:
                r += 1
            put(e.id, c, r)

    free = [e.id for e in elements if e.id not in col]
    fluid = [(c.from_, c.to) for c in connections if c.kind == "fluid" and c.from_ in free and c.to in free]
    parent = {m: m for m in free}

    def find(m: str) -> str:
        while parent[m] != m:
            parent[m] = parent[parent[m]]
            m = parent[m]
        return m

    for a, b in fluid:
        parent[find(a)] = find(b)
    units: dict[str, list[str]] = {}
    for m in free:
        units.setdefault(find(m), []).append(m)
    todo = list(units.values())
    depths = {id(u): _fluid_depths(u, [(a, b) for a, b in fluid if a in u],
                                   [(c.from_, c.to) for c in connections if c.kind != "fluid" and c.from_ in u and c.to in u])
              if len(u) > 1 else {u[0]: 0} for u in todo}

    def links(unit: list[str]) -> list[tuple[str, str, object]]:
        """(member, placed neighbour, connection) for each connection of a unit to a placed element."""
        out = []
        for c in connections:
            if c.from_ in unit and c.to in col:
                out.append((c.from_, c.to, c))
            elif c.to in unit and c.from_ in col:
                out.append((c.to, c.from_, c))
        return out

    def target(v: str, u: str, c) -> int:
        if c.kind == "fluid":
            return col[u] - 1 if c.from_ == v else col[u] + 1
        return col[u]

    first = True
    while todo:
        if first and any(subject in u for u in todo):
            unit = next(u for u in todo if subject in u)
        else:
            unit = max(todo, key=lambda u: (len(links(u)), len(u), -min(index[m] for m in u)))
        first = False
        todo.remove(unit)
        depth = depths[id(unit)]
        ties = links(unit)
        if len(unit) > 1:                                 # a fluid network: one shift for all, one row in each lane
            shift = median_low([target(m, u, c) - depth[m] for m, u, c in ties]) if ties else 0
            by_lane: dict[int, list[list[str]]] = {}      # lane -> branches: elements of one depth go to separate rows
            for m in sorted(unit, key=lambda m: (depth[m], index[m])):
                branches = by_lane.setdefault(lane_of[m], [])
                branch = next((b for b in branches if all(depth[x] != depth[m] for x in b)), None)
                if branch is None:
                    branches.append([m])
                else:
                    branch.append(m)
            for lane, branches in by_lane.items():
                for members in branches:
                    r = next(r for r in rows_to_try(lane)
                             if all((lane, r, depth[m] + shift) not in cells for m in members))
                    for m in members:
                        put(m, depth[m] + shift, r)
            continue
        v = unit[0]
        lane = lane_of[v]
        c0 = median_low([target(m, u, c) for m, u, c in ties]) if ties else 0
        mates = [(u, c) for _, u, c in ties if lane_of[u] == lane and col[u] == c0]
        placed = False
        if mates:                                         # beside the element it belongs to, in its row
            u, c = mates[0]
            sides = (c0 - 1, c0 + 1) if c.from_ == v else (c0 + 1, c0 - 1)
            for cc in sides:
                if (lane, row[u], cc) not in cells:
                    put(v, cc, row[u])
                    placed = True
                    break
        if not placed:
            start = row[mates[0][0]] if mates else 0
            options = [(abs(dc) + 1.2 * k, k, abs(dc), dc, r) for k, r in enumerate(rows_to_try(lane, start)[:12])
                       for dc in (0, -1, 1, -2, 2) if (lane, r, c0 + dc) not in cells]
            _, _, _, dc, r = min(options)
            put(v, c0 + dc, r)

    c_min = min(col.values(), default=0)
    lane_rows: dict[int, list[int]] = {}
    for eid in col:
        lane_rows.setdefault(lane_of[eid], []).append(row[eid])
    base: dict[int, int] = {}
    r_min: dict[int, int] = {}
    total = 0
    for lane in range(len(groups)):
        rs = lane_rows.get(lane, [0])
        base[lane], r_min[lane] = total, min(rs)
        total += max(rs) - min(rs) + 1
    return ({eid: (col[eid] - c_min, base[lane_of[eid]] + row[eid] - r_min[lane_of[eid]]) for eid in col},
            groups, lane_of)


# --------------------------------------------------------------------------- routes on the grid
#
# The routing grid has a line through the middle of each column and of each row (odd index) and a line in
# each channel between them (even index). A node (u, v) with two odd indices is the middle of a cell: a
# route cannot pass it when the cell holds a box.

_DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))                # east, west, south, north


def _route(src: tuple[int, int], dst: tuple[int, int], size: tuple[int, int], occupied: set[tuple[int, int]],
           use: dict, fluid: bool, pair: frozenset) -> list[tuple[int, int]]:
    """The cheapest route between two cells: length, corners, crossings with the routes that are there,
    and the number of lines that leave a box at one side."""
    n_u, n_v = size
    start = (2 * src[0] + 1, 2 * src[1] + 1)
    goal = (2 * dst[0] + 1, 2 * dst[1] + 1)
    heap: list = [(0.0, 0, start, -1)]
    best: dict = {(start, -1): 0.0}
    back: dict = {}
    counter = 0
    while heap:
        cost, _, node, d = heapq.heappop(heap)
        if best.get((node, d), 1e18) < cost:
            continue
        if node == goal:
            out = [node]
            key = (node, d)
            while key in back:
                key = back[key]
                out.append(key[0])
            return out[::-1]
        u, v = node
        for nd, (du, dv) in enumerate(_DIRS):
            if d >= 0 and (nd ^ 1) == d:                   # no way back
                continue
            nu, nv = u + du, v + dv
            if not (0 <= nu < n_u and 0 <= nv < n_v):
                continue
            nxt = (nu, nv)
            if nu % 2 and nv % 2 and nxt != goal and ((nu - 1) // 2, (nv - 1) // 2) in occupied:
                continue
            if nxt == start:
                continue
            step = STEP_X if dv == 0 else STEP_Y
            c = cost + step
            if d >= 0 and (nd < 2) != (d < 2):
                c += BEND
            elif d >= 0:                                   # straight on through this node: lines it crosses
                c += CROSS * use["through"].get((node, nd >= 2), 0)
            if node == start or nxt == goal:               # a port: the side of a box
                box = src if node == start else dst
                side = nd if node == start else nd ^ 1
                c += (PORT_SIDE if side < 2 else PORT_TOP) * sum(
                    1 for p in use["ports"].get((box, side), ()) if p != pair)
                if fluid and side >= 2:
                    c += FLUID_TOP
            else:
                c += SHARE * use["segments"].get((min(node, nxt), max(node, nxt)), 0)
            key = (nxt, nd)
            if c < best.get(key, 1e18):
                best[key] = c
                back[key] = (node, d)
                counter += 1
                heapq.heappush(heap, (c, counter, nxt, nd))
    return [start, goal]                                   # no way (cannot occur on a grid with channels)


def _record(nodes: list[tuple[int, int]], src, dst, use: dict, pair: frozenset) -> None:
    for i in range(len(nodes) - 1):
        a, b = nodes[i], nodes[i + 1]
        if 0 < i < len(nodes) - 2:
            use["segments"][(min(a, b), max(a, b))] = use["segments"].get((min(a, b), max(a, b)), 0) + 1
        if i > 0:
            p = nodes[i - 1]
            if (p[0] == a[0]) == (a[0] == b[0]):           # straight through `a`
                vertical = a[0] == b[0]
                use["through"][(a, not vertical)] = use["through"].get((a, not vertical), 0) + 1
    def side(a, b):
        return _DIRS.index((b[0] - a[0], b[1] - a[1]))
    use["ports"].setdefault((src, side(nodes[0], nodes[1])), []).append(pair)
    use["ports"].setdefault((dst, side(nodes[-1], nodes[-2])), []).append(pair)


def _segments(edge_index: int, nodes: list[tuple[int, int]]) -> list[Seg]:
    segs: list[Seg] = []
    i = 0
    while i < len(nodes) - 1:
        horizontal = nodes[i][1] == nodes[i + 1][1]
        j = i + 1
        while j < len(nodes) - 1 and (nodes[j][1] == nodes[j + 1][1]) == horizontal:
            j += 1
        a, b = nodes[i], nodes[j]
        segs.append(Seg(edge_index, horizontal, a[1] if horizontal else a[0],
                        a[0] if horizontal else a[1], b[0] if horizontal else b[1]))
        i = j
    segs[0].first = True
    segs[-1].last = True
    return segs


# --------------------------------------------------------------------------- tracks: lines side by side


def _clusters(edges: list[Edge]) -> list[list[Seg]]:
    """Groups of route pieces on one grid line that touch or overlap (directly or through others).
    Each group is in a first order: a piece whose neighbours lead up or left comes first."""
    lines: dict[tuple[bool, int], list[Seg]] = {}
    for e in edges:
        for s in e.segs:
            lines.setdefault((s.horizontal, s.line), []).append(s)

    def lead(e: Edge, s: Seg) -> float:
        """Where the neighbours of a piece lead, across its line (negative: up or left)."""
        i = e.segs.index(s)
        out = 0.0
        for k, end in ((i - 1, e.segs[i - 1].s if i > 0 else 0), (i + 1, e.segs[i + 1].t if i + 1 < len(e.segs) else 0)):
            if 0 <= k < len(e.segs):
                out += end - s.line
        return out

    out: list[list[Seg]] = []
    for key in sorted(lines):
        segs = sorted(lines[key], key=lambda s: s.span)
        group: list[Seg] = []
        reach = None
        for s in segs:
            lo, hi = s.span
            if reach is not None and lo < reach:
                group.append(s)
                reach = max(reach, hi)
            else:
                if group:
                    out.append(group)
                group, reach = [s], hi
        if group:
            out.append(group)
    for group in out:
        group.sort(key=lambda s: (lead(edges[s.edge], s), s.span, s.edge))
    return out


def _overlap(a: Seg, b: Seg) -> bool:
    (a0, a1), (b0, b1) = a.span, b.span
    return a0 < b1 and b0 < a1


def _levels(group: list[Seg]) -> int:
    """Give each piece of a group its level: one more than the highest level of the pieces before it that
    it overlaps. Returns the number of levels."""
    for i, s in enumerate(group):
        s.level = 1 + max((t.level for t in group[:i] if _overlap(s, t)), default=-1)
    return 1 + max(s.level for s in group)


def _offsets(group: list[Seg], n: int) -> None:
    """Levels to pixels. In a channel the levels are centred. On the line through the middle of a row or
    column, a straight line from box to box stays in the middle, and the others go beside it."""
    on_axis = group[0].line % 2 == 1
    straight = [s.level for s in group if s.first and s.last]
    centre = (sum(straight) / len(straight)) if (on_axis and straight) else (n - 1) / 2
    pitch = TRACK
    if on_axis:
        limit = (BOX_H / 2 - 9) if group[0].horizontal else (BOX_W / 2 - 14)
        reach = max(abs(s.level - centre) for s in group) * pitch
        if reach > limit:
            pitch *= limit / reach
    for s in group:
        s.offset = (s.level - centre) * pitch


# --------------------------------------------------------------------------- geometry


@dataclass
class _Grid:
    cols: int
    rows: int
    lane_rows: list[tuple[int, int]]                # first and last row of each lane
    gap_x: list[float]                              # width of each vertical channel (cols + 1)
    gap_y: list[float]                              # height of each horizontal channel (rows + 1)
    lines_x: list[int] = field(default_factory=list)      # number of lines side by side in each vertical channel
    lines_y: list[int] = field(default_factory=list)      # and in each horizontal channel
    left: set[int] = field(default_factory=set)     # vertical channels whose lines go to the left side
    top: set[int] = field(default_factory=set)      # horizontal channels whose lines go to the upper side
    x0: float = 0.0
    y0: float = 0.0
    xs: list[float] = field(default_factory=list)   # pixel x of each vertical grid line (2 * cols + 1)
    ys: list[float] = field(default_factory=list)
    lane_y: list[tuple[float, float]] = field(default_factory=list)

    def build(self) -> None:
        x = self.x0
        self.xs = []
        for c in range(self.cols + 1):
            bundle = max(0, (self.lines_x[c] if self.lines_x else 0) - 1) * TRACK
            self.xs.append(x + (CLEAR + bundle / 2 if c in self.left else self.gap_x[c] / 2))
            x += self.gap_x[c]
            if c < self.cols:
                self.xs.append(x + BOX_W / 2)
                x += BOX_W
        x -= self.gap_x[self.cols]
        self.right = x + self.gap_x[self.cols]
        y = self.y0
        self.ys = []
        self.lane_y = []
        for a, b in self.lane_rows:
            top = y
            for r in range(a, b + 1):
                self.ys.append(y + self._channel_y(r))
                y += self.gap_y[r]
                self.ys.append(y + BOX_H / 2)
                y += BOX_H
            if b == self.rows - 1:                         # the channel under the last row of the diagram
                self.ys.append(y + self._channel_y(self.rows))
                y += self.gap_y[self.rows]
            else:
                y += LANE_BOTTOM
            self.lane_y.append((top, y))
        self.bottom = y

    def _channel_y(self, r: int) -> float:
        """Where the lines of a horizontal channel run, measured from its upper side."""
        bundle = max(0, (self.lines_y[r] if self.lines_y else 0) - 1) * TRACK
        return CLEAR + bundle / 2 if r in self.top else self.gap_y[r] / 2

    def col_x(self, c: int) -> float:
        return self.xs[2 * c + 1] - BOX_W / 2

    def row_y(self, r: int) -> float:
        return self.ys[2 * r + 1] - BOX_H / 2


def _points(e: Edge, boxes: dict[str, Box], g: _Grid) -> list[tuple[float, float]]:
    src, dst = boxes[e.src], boxes[e.dst]

    def at_box(box: Box, s: Seg, leaving: bool) -> tuple[float, float]:
        forward = (s.t > s.s) if leaving else (s.s > s.t)   # away from the box, along +x or +y
        if s.horizontal:
            return (box.x + box.w if forward else box.x, g.ys[s.line] + s.offset)
        return (g.xs[s.line] + s.offset, box.y + box.h if forward else box.y)

    pts = [at_box(src, e.segs[0], True)]
    for a, b in zip(e.segs, e.segs[1:]):
        h, v = (a, b) if a.horizontal else (b, a)
        pts.append((g.xs[v.line] + v.offset, g.ys[h.line] + h.offset))
    pts.append(at_box(dst, e.segs[-1], False))
    return pts


def _pieces(points: list[tuple[float, float]]) -> list[tuple[float, float, float, float]]:
    """A line as straight pieces (x0, y0, x1, y1), each with x0 <= x1 and y0 <= y1."""
    return [(min(a[0], b[0]), min(a[1], b[1]), max(a[0], b[0]), max(a[1], b[1])) for a, b in zip(points, points[1:])]


def _crossings(a: list[tuple[float, float]], b: list[tuple[float, float]]) -> int:
    n = 0
    for p in _pieces(a):
        for q in _pieces(b):
            p_h, q_h = p[1] == p[3], q[1] == q[3]
            if p_h == q_h:
                continue
            h, v = (p, q) if p_h else (q, p)
            if h[0] + 0.01 < v[0] < h[2] - 0.01 and v[1] + 0.01 < h[1] < v[3] - 0.01:
                n += 1
    return n


def _total_crossings(edges: list[Edge], only: set[int] | None = None) -> int:
    n = 0
    for i, a in enumerate(edges):
        for k in range(i + 1, len(edges)):
            if only is None or i in only or k in only:
                n += _crossings(a.points, edges[k].points)
    return n


def _untangle(edges: list[Edge], groups: list[list[Seg]], boxes: dict[str, Box], g: _Grid) -> None:
    """Change the order of lines that run side by side while that takes away crossings. Two moves: two
    neighbours in one group change places; two connections change places in every group they share."""
    def redo(group: list[Seg]) -> None:
        _offsets(group, _levels(group))

    def refresh(which: set[int]) -> None:
        for i in which:
            if edges[i].segs:
                edges[i].points = _points(edges[i], boxes, g)

    def count(which: set[int]) -> int:
        return _total_crossings(edges, which)

    multi = [grp for grp in groups if len(grp) > 1]
    shared: dict[tuple[int, int], list[list[Seg]]] = {}
    for grp in multi:
        ids = sorted({s.edge for s in grp})
        for a, b in itertools.combinations(ids, 2):
            shared.setdefault((a, b), []).append(grp)
    for _ in range(6):
        improved = False
        for grp in multi:                                  # two neighbours change places
            for i in range(len(grp) - 1):
                touched = {s.edge for s in grp}
                before = count(touched)
                if before == 0:
                    break
                grp[i], grp[i + 1] = grp[i + 1], grp[i]
                redo(grp)
                refresh(touched)
                if count(touched) < before:
                    improved = True
                else:
                    grp[i], grp[i + 1] = grp[i + 1], grp[i]
                    redo(grp)
                    refresh(touched)
        for (a, b), grps in shared.items():                # two connections change places everywhere
            if len(grps) < 2:
                continue
            touched = {s.edge for grp in grps for s in grp}
            before = count(touched)
            if before == 0:
                continue

            def swap():
                for grp in grps:
                    ia = [i for i, s in enumerate(grp) if s.edge == a]
                    ib = [i for i, s in enumerate(grp) if s.edge == b]
                    if len(ia) == 1 and len(ib) == 1:
                        grp[ia[0]], grp[ib[0]] = grp[ib[0]], grp[ia[0]]
                    redo(grp)
                refresh(touched)

            swap()
            if count(touched) < before:
                improved = True
            else:
                swap()
        if not improved:
            break


# --------------------------------------------------------------------------- labels


def _hits(r: tuple[float, float, float, float], s: tuple[float, float, float, float], pad: float = 0.0) -> bool:
    return r[0] - pad < s[2] and s[0] - pad < r[2] and r[1] - pad < s[3] and s[1] - pad < r[3]


def _label_blocks(e: Edge) -> list[Block]:
    name = [(t, "dz-cname") for t in _wrap(e.name or e.id, LABEL_W, FS_LABEL)]
    if e.medium:
        name.append((e.medium, "dz-medium"))
    blocks = [Block(name, FS_LABEL, LH_LABEL)]
    if e.subject and e.effects:
        lines: list[str] = []
        cur = ""
        for item in _summary(e.effects):
            trial = f"{cur} · {item}" if cur else item
            if cur and text_width(trial, FS_SMALL) > LABEL_W:
                lines.append(cur)
                cur = item
            else:
                cur = trial
        lines.append(cur)
        blocks.append(Block([(t, "dz-effects") for t in lines], FS_SMALL, LH_SMALL))
    for b in blocks:
        b.w = max(text_width(t, b.size) for t, _ in b.lines) + 6
        b.h = len(b.lines) * b.line_height + 2
    return blocks


def _along(lo: float, hi: float, size: float, step: float) -> list[float]:
    """Places for the middle of a label of a given size along a piece of a line: the middle of the piece
    first, then to both sides in steps, as far as the label stays on the piece."""
    mid = (lo + hi) / 2
    room = (hi - lo - size - 6) / 2
    out = [mid]
    k = 1
    while k * step <= room:
        out += [mid - k * step, mid + k * step]
        k += 1
    if room > 0 and (k - 1) * step < room:
        out += [mid - room, mid + room]
    return out


def _label_options(e: Edge) -> list[tuple[float, list[tuple[float, float, str]]]]:
    """Places a label can go, each with its cost: (cost, [(left, top, anchor) for each block]). The best
    place is over the middle of the longest horizontal piece: name above the line, effects below it."""
    blocks = e.blocks
    out = []
    pieces = [(a, b) for a, b in zip(e.points, e.points[1:])]
    longest = max((abs(b[0] - a[0]) + abs(b[1] - a[1]) for a, b in pieces), default=1.0) or 1.0
    wide = max(b.w for b in blocks)
    tall = sum(b.h for b in blocks)
    for k, (a, b) in enumerate(pieces):
        length = abs(b[0] - a[0]) + abs(b[1] - a[1])
        short = 30.0 * (1 - length / longest) + (0 if k not in (0, len(pieces) - 1) or len(pieces) == 1 else 4.0)
        if a[1] == b[1]:                                            # horizontal piece
            x0, x1, y = min(a[0], b[0]), max(a[0], b[0]), a[1]
            tight = 0.0 if length >= wide + 6 else 18.0 + (wide - length) * 0.4
            for cx in _along(x0, x1, wide, 12.0):
                base = short + tight + abs(cx - (x0 + x1) / 2) * 0.06
                if len(blocks) == 2:
                    out.append((base, [(cx - blocks[0].w / 2, y - LABEL_GAP - blocks[0].h, "middle"),
                                       (cx - blocks[1].w / 2, y + LABEL_GAP, "middle")]))
                top = y - LABEL_GAP - tall
                above, below = [], []
                for blk in blocks:
                    above.append((cx - blk.w / 2, top, "middle"))
                    top += blk.h
                top = y + LABEL_GAP
                for blk in blocks:
                    below.append((cx - blk.w / 2, top, "middle"))
                    top += blk.h
                out.append((base + (3.0 if len(blocks) == 2 else 0.0), above))
                out.append((base + (6.0 if len(blocks) == 2 else 3.0), below))
        else:                                                       # vertical piece
            y0, y1, x = min(a[1], b[1]), max(a[1], b[1]), a[0]
            tight = 0.0 if length >= tall + 6 else 18.0 + (tall - length) * 0.6
            for cy in _along(y0, y1, tall, 10.0):
                base = short + tight + abs(cy - (y0 + y1) / 2) * 0.08 + 8.0
                for side, cost_s in (("start", 0.0), ("end", 2.0)):
                    top = cy - tall / 2
                    option = []
                    for blk in blocks:
                        option.append((x + 7 if side == "start" else x - 7 - blk.w, top, side))
                        top += blk.h
                    out.append((base + cost_s, option))
    out.sort(key=lambda o: o[0])
    return out


def _place_labels(edges: list[Edge], boxes: dict[str, Box], area: tuple[float, float, float, float],
                  borders: list[tuple[int, float]] = ()) -> None:
    """Give each label the best place that covers no box, no line, no other label and no border between
    two lanes (`borders`: the first row of the lower lane and the y of the border). A label without such
    a place gets the place with the least cover, and `label_ok` False."""
    taken: list[tuple[float, float, float, float]] = []
    lines = [(i, p) for i, e in enumerate(edges) for p in _pieces(e.points)]
    box_rects = [b.rect for b in boxes.values()]
    order = sorted(range(len(edges)), key=lambda i: (not edges[i].subject, i))
    for i in order:
        e = edges[i]
        if not e.points:
            continue
        best, best_cover = None, None
        e.label_ok, e.label_cut = False, None
        for cost, option in _label_options(e):
            rects = [(x, y, x + blk.w, y + blk.h) for (x, y, _), blk in zip(option, e.blocks)]
            cover = 0.0
            cut = None
            for r in rects:
                if r[0] < area[0] or r[1] < area[1] or r[2] > area[2] or r[3] > area[3]:
                    cover += 50.0
                cover += 20.0 * sum(1 for b in box_rects if _hits(r, b, 3.0))
                cover += 10.0 * sum(1 for t in taken if _hits(r, t, 3.0))
                cover += 4.0 * sum(1 for _, p in lines if _hits(r, p, 2.0))
                for first_row, y in borders:
                    if r[1] - 2.0 < y < r[3] + 2.0:
                        cover += 1.0
                        cut = first_row
            if cover == 0.0:
                best, e.label_ok, e.label_cut = option, True, None
                break
            if best_cover is None or cover < best_cover:
                best, best_cover = option, cover
                e.label_cut = cut if cover < 4.0 else None
        for (x, y, anchor), blk in zip(best, e.blocks):
            blk.x, blk.y, blk.anchor = x, y, anchor
            taken.append(blk.rect)


# --------------------------------------------------------------------------- the layout


def layout(project: Project, ctx: Context) -> Layout:
    """Places for the boxes, routes for the lines and places for the labels. The same inputs give the
    same layout."""
    system = project.system
    comp = project.subject
    subject = comp.element
    counts = effect_counts(project, ctx)
    cells, groups, lane_of = _place(system, subject)

    boxes: dict[str, Box] = {}
    for el in system.elements:
        c, r = cells[el.id]
        boxes[el.id] = Box(el.id, el.name, el.kind, el.group, el.description, el.id == subject,
                           comp.id if el.id == subject else (el.component or ""), c, r, lane_of[el.id])
    n_cols = 1 + max((b.col for b in boxes.values()), default=0)
    n_rows = 1 + max((b.row for b in boxes.values()), default=0)
    lane_rows = []
    for lane in range(len(groups)):
        rs = [b.row for b in boxes.values() if b.lane == lane]
        lane_rows.append((min(rs), max(rs)))

    edges = [Edge(c.id, c.name, c.kind, c.medium, c.from_, c.to, c.description, subject in (c.from_, c.to),
                  counts.get(c.id, {})) for c in system.connections]
    for b in boxes.values():
        b.lines = _wrap(b.name, BOX_W - 18, FS_NAME, bold=True, max_lines=2)
        mine = [e for e in edges if e.subject and (b.subject or b.id in (e.src, e.dst))]
        total: dict[str, int] = {}
        for e in mine:
            for k, n in e.effects.items():
                total[k] = total.get(k, 0) + n
        b.effects = {k: total[k] for k in [*EFFECT_KINDS, *total] if k in total}

    # ---- routes
    occupied = {(b.col, b.row) for b in boxes.values()}
    use: dict = {"segments": {}, "through": {}, "ports": {}}
    size = (2 * n_cols + 1, 2 * n_rows + 1)

    def reach(e: Edge) -> int:
        a, b = boxes[e.src], boxes[e.dst]
        return abs(a.col - b.col) + abs(a.row - b.row)

    for i in sorted(range(len(edges)), key=lambda i: (not edges[i].subject, edges[i].kind != "fluid", reach(edges[i]), i)):
        e = edges[i]
        if e.src == e.dst:
            continue                                       # drawn as a small loop at the box
        a, b = boxes[e.src], boxes[e.dst]
        pair = frozenset((e.src, e.dst))
        e.nodes = _route((a.col, a.row), (b.col, b.row), size, occupied, use, e.kind == "fluid", pair)
        _record(e.nodes, (a.col, a.row), (b.col, b.row), use, pair)
        e.segs = _segments(i, e.nodes)

    # ---- tracks and the size of each channel
    groups_of_segs = _clusters(edges)
    need_x = [0] * (n_cols + 1)
    need_y = [0] * (n_rows + 1)
    for grp in groups_of_segs:
        n = _levels(grp)
        _offsets(grp, n)
        s = grp[0]
        if s.line % 2 == 0:
            need = need_y if s.horizontal else need_x
            need[s.line // 2] = max(need[s.line // 2], n)

    lane_starts = {a for a, _ in lane_rows}

    def base_gap_y(k: int) -> float:
        if k in lane_starts:
            return LANE_TOP
        return LANE_BOTTOM if k == n_rows else CHAN_Y

    grid = _Grid(n_cols, n_rows, lane_rows,
                 [max(CHAN_EDGE if k in (0, n_cols) else CHAN_X, (need_x[k] - 1) * TRACK + 2 * CLEAR if need_x[k] else 0.0)
                  for k in range(n_cols + 1)],
                 [max(base_gap_y(k), (need_y[k] - 1) * TRACK + 2 * CLEAR if need_y[k] else 0.0) for k in range(n_rows + 1)],
                 need_x, need_y)

    heads = [_wrap(gname, 92.0, FS_LABEL, bold=True, max_lines=3) if gname else [] for gname in groups]
    head_w = (max((text_width(t, FS_LABEL, bold=True) for hl in heads for t in hl), default=0.0) + 22) if any(heads) else 0.0
    title = [(f"{system.name}", "dz-title"),
             (f"{system.id} · component under verification: {comp.name} ({comp.id})", "dz-subtitle")]
    grid.x0 = MARGIN + head_w
    grid.y0 = MARGIN + 44.0
    for e in edges:
        e.blocks = _label_blocks(e)

    def build() -> None:
        grid.build()
        for b in boxes.values():
            b.x, b.y = grid.col_x(b.col), grid.row_y(b.row)
        for e in edges:
            if e.segs:
                e.points = _points(e, boxes, grid)
            elif e.src == e.dst:                           # a loop above the right part of the box
                b = boxes[e.src]
                top = b.y - min(18.0, grid.gap_y[b.row] / 2)
                e.points = [(b.x + b.w - 54, b.y), (b.x + b.w - 54, top), (b.x + b.w - 20, top), (b.x + b.w - 20, b.y)]

    build()
    _untangle(edges, groups_of_segs, boxes, grid)

    tries: dict[str, int] = {}
    for _ in range(24):                                    # let the channels grow until every label has a place
        build()
        area = (grid.x0, grid.y0, grid.right, grid.bottom)
        borders = [(lane_rows[i][0], grid.lane_y[i][0]) for i in range(1, len(lane_rows))]
        _place_labels(edges, boxes, area, borders)
        bad = [e for e in edges if e.points and not e.label_ok]
        if not bad:
            break
        more_x: dict[int, float] = {}                      # channel -> the most that a label asks for
        more_y: dict[int, float] = {}

        def ask(more: dict[int, float], k: int, amount: float) -> None:
            more[k] = max(more.get(k, 0.0), amount)

        for e in bad:
            wide, tall = max(b.w for b in e.blocks), sum(b.h for b in e.blocks)
            n = tries[e.id] = tries.get(e.id, 0) + 1
            if not e.segs:
                ask(more_y, boxes[e.src].row, 14.0)
                continue
            if e.label_cut is not None:                    # the label is across the border of a lane
                ask(more_y, e.label_cut, 8.0)
                continue
            lengths = [abs(b[0] - a[0]) + abs(b[1] - a[1]) for a, b in zip(e.points, e.points[1:])]
            i = max(range(len(e.segs)), key=lambda i: (lengths[i], e.segs[i].horizontal))
            s, length = e.segs[i], lengths[i]
            lo, hi = sorted((s.s, s.t))
            evens = [k // 2 for k in range(lo, hi + 1) if k % 2 == 0]      # the channels it goes through
            k = evens[len(evens) // 2]
            if s.horizontal:
                busy = [c for c in evens if need_x[c] and c not in grid.left]
                if length < wide + 10:                     # the line is shorter than its label
                    ask(more_x, k, wide + 10 - length)
                elif busy:                                 # lines cross it: they go to one side of their channel
                    c = busy[len(busy) // 2]
                    grid.left.add(c)
                    ask(more_x, c, (need_x[c] - 1) * TRACK + 2 * CLEAR + wide + 14 - grid.gap_x[c])
                elif n % 2:                                # no room above or below the line
                    ask(more_y, s.line // 2, 10.0)
                else:
                    ask(more_x, k, 16.0)
            else:
                busy = [c for c in evens if need_y[c] and c not in grid.top]
                if length < tall + 10:
                    ask(more_y, k, tall + 10 - length)
                elif busy:
                    c = busy[len(busy) // 2]
                    grid.top.add(c)
                    ask(more_y, c, (need_y[c] - 1) * TRACK + 2 * CLEAR + tall + 14 - grid.gap_y[c])
                elif n % 2:                                # no room beside the line
                    ask(more_x, (s.line + 1) // 2, 12.0)
                else:
                    ask(more_y, k, 12.0)
        for k, amount in more_x.items():
            grid.gap_x[k] += max(amount, 0.0)
        for k, amount in more_y.items():
            grid.gap_y[k] += max(amount, 0.0)

    lanes = [Lane(gname, lane_rows[i], grid.lane_y[i][0], grid.lane_y[i][1], heads[i]) for i, gname in enumerate(groups)]
    panel = (MARGIN, grid.y0, grid.right, grid.bottom)

    # ---- the legend
    kinds = [k for k in KIND_ORDER if any(e.kind == k for e in edges)]
    kinds += sorted({e.kind for e in edges} - set(KIND_ORDER))
    items = [{"type": "line", "kind": k, "text": k} for k in kinds]
    items.append({"type": "subject", "text": "component under verification"})
    if any(e.subject and e.effects for e in edges):
        items.append({"type": "note", "text": "pressure 7: the number of effects of that kind on the component, in all phases"})
    x, y = MARGIN + 2, grid.bottom + 24.0
    width = max(grid.right + MARGIN, MARGIN + max(text_width(t, FS_TITLE if c == "dz-title" else FS_LABEL, c == "dz-title")
                                                    for t, c in title) + MARGIN)
    for it in items:
        w = (40 if it["type"] != "note" else 0) + text_width(it["text"], FS_KIND) + 20
        if x + w > width - MARGIN and x > MARGIN + 2:
            x, y = MARGIN + 2, y + 20.0
        it["x"], it["y"], it["w"] = x, y, w
        x += w
    return Layout(width, y + 16.0, boxes, edges, lanes, panel, head_w, title, items, grid.bottom + 24.0,
                  _total_crossings(edges))


# --------------------------------------------------------------------------- SVG


def _path(points: list[tuple[float, float]], trim: float = 0.0) -> str:
    """A line with round corners. `trim` shortens its end (the arrowhead goes there)."""
    pts = list(points)
    if trim and len(pts) >= 2:
        (x0, y0), (x1, y1) = pts[-2], pts[-1]
        length = abs(x1 - x0) + abs(y1 - y0)
        if length > trim:
            f = (length - trim) / length
            pts[-1] = (x0 + (x1 - x0) * f, y0 + (y1 - y0) * f)
    d = [f"M{pts[0][0]:.1f} {pts[0][1]:.1f}"]
    for i in range(1, len(pts) - 1):
        (ax, ay), (bx, by), (cx, cy) = pts[i - 1], pts[i], pts[i + 1]
        r = min(RADIUS, (abs(bx - ax) + abs(by - ay)) / 2, (abs(cx - bx) + abs(cy - by)) / 2)
        ux, uy = (bx - ax, by - ay)
        n = abs(ux) + abs(uy) or 1.0
        vx, vy = (cx - bx, cy - by)
        m = abs(vx) + abs(vy) or 1.0
        d.append(f"L{bx - ux / n * r:.1f} {by - uy / n * r:.1f}")
        d.append(f"Q{bx:.1f} {by:.1f} {bx + vx / m * r:.1f} {by + vy / m * r:.1f}")
    d.append(f"L{pts[-1][0]:.1f} {pts[-1][1]:.1f}")
    return " ".join(d)


def _style() -> str:
    v = _var
    rules = [
        f".dz-diagram{{font-family:var(--dz-font, {FONT});}}",
        f".dz-bg{{fill:{v('--dz-bg')};}}",
        f".dz-title{{font-size:{FS_TITLE}px;font-weight:600;fill:{v('--dz-fg')};}}",
        f".dz-subtitle{{font-size:{FS_LABEL}px;fill:{v('--dz-muted')};}}",
        f".dz-panel{{fill:{v('--dz-lane')};stroke:{v('--dz-rule')};stroke-width:1;}}",
        f".dz-lane-head{{fill:{v('--dz-lane-head')};}}",
        f".dz-lane-rule{{stroke:{v('--dz-rule')};stroke-width:1;}}",
        f".dz-lane-name{{font-size:{FS_LABEL}px;font-weight:600;fill:{v('--dz-muted')};}}",
        f".dz-box{{fill:{v('--dz-box')};stroke:{v('--dz-box-stroke')};stroke-width:1.2;}}",
        f".dz-name{{font-size:{FS_NAME}px;font-weight:600;fill:{v('--dz-fg')};text-anchor:middle;}}",
        f".dz-kind{{font-size:{FS_KIND}px;fill:{v('--dz-muted')};text-anchor:middle;}}",
        f".dz-subject .dz-box{{fill:{v('--dz-accent-bg')};stroke:{v('--dz-accent')};stroke-width:2.6;}}",
        f".dz-tag rect{{fill:{v('--dz-accent')};}}",
        f".dz-tag text{{font-size:9px;font-weight:600;fill:{v('--dz-accent-fg')};text-anchor:middle;letter-spacing:.03em;}}",
        ".dz-line{fill:none;stroke-linejoin:round;}",
        ".dz-hit{fill:none;stroke:transparent;stroke-width:14;pointer-events:stroke;}",
        f".dz-inner{{fill:none;stroke:{v('--dz-lane')};stroke-width:1.6;}}",
        f".dz-label text{{stroke:{v('--dz-lane')};stroke-width:3px;stroke-linejoin:round;paint-order:stroke;}}",
        f".dz-cname{{font-size:{FS_LABEL}px;fill:{v('--dz-fg')};}}",
        f".dz-medium{{font-size:{FS_SMALL}px;font-weight:600;fill:{v('--dz-muted')};}}",
        f".dz-effects{{font-size:{FS_SMALL}px;fill:{v('--dz-muted')};}}",
        f".dz-legend text{{font-size:{FS_KIND}px;fill:{v('--dz-muted')};}}",
        ".dz-connection .dz-line,.dz-connection .dz-inner{opacity:.8;}",
        ".dz-connection.dz-subject .dz-line,.dz-connection.dz-subject .dz-inner{opacity:1;}",
        ".dz-connection:hover .dz-line,.dz-connection.dz-active .dz-line{opacity:1;filter:brightness(.85);}",
        f".dz-element:hover .dz-box,.dz-element.dz-active .dz-box{{stroke:{v('--dz-accent')};stroke-width:2.2;}}",
        ".dz-dim{opacity:.22;}",
    ]
    for k in sorted({*KINDS, *KIND_ORDER}):
        width, dash, round_caps, _ = KINDS[k]
        rules.append(f".dz-kind-{k} .dz-line,.dz-legend .dz-line.dz-kind-{k}{{stroke:{v('--dz-' + k)};stroke-width:{width};"
                     + (f"stroke-dasharray:{dash};" if dash else "")
                     + ("stroke-linecap:round;" if round_caps else "") + "}")
        rules.append(f".dz-arrow-{k}{{fill:{v('--dz-' + k)};}}")
    rules.append(f".dz-kind-other .dz-line{{stroke:{v('--dz-muted')};stroke-width:1.8;}}")
    rules.append(f".dz-arrow-other{{fill:{v('--dz-muted')};}}")
    return "".join(rules)


def _kind_class(kind: str) -> str:
    return kind if kind in KINDS else "other"


def render(project: Project, ctx: Context) -> str:
    """The system diagram as one SVG document. See the module docstring for the hooks and the colours."""
    lay = layout(project, ctx)
    system, comp = project.system, project.subject
    names = {b.id: b.name for b in lay.boxes.values()}
    o: list[str] = []
    w, h = lay.width, lay.height
    o.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w:.0f} {h:.0f}" class="dz-diagram" role="img" '
             f'aria-labelledby="dz-doc-title dz-doc-desc" data-system={quoteattr(system.id)} '
             f'data-subject={quoteattr(comp.element)} data-component={quoteattr(comp.id)}>')
    o.append(f'<title id="dz-doc-title">{escape("System diagram: " + system.name)}</title>')
    o.append(f'<desc id="dz-doc-desc">{escape(f"{len(lay.boxes)} elements and {len(lay.edges)} connections. The component under verification is {comp.name} ({comp.id}), the element {comp.element}.")}</desc>')
    o.append(f"<style>{_style()}</style>")
    o.append("<defs>")
    for k in [*KIND_ORDER, "other"]:
        o.append(f'<marker id="dz-arrow-{k}" viewBox="0 0 10 8" markerWidth="{ARROW + 1:.0f}" markerHeight="8" refX="6.5" refY="4" '
                 f'orient="auto" markerUnits="userSpaceOnUse"><path class="dz-arrow-{k}" d="M0 0L10 4L0 8L2.2 4Z"/></marker>')
    o.append("</defs>")
    o.append(f'<rect class="dz-bg" x="0" y="0" width="{w:.0f}" height="{h:.0f}"/>')

    o.append('<g class="dz-header">')
    for i, (text, cls) in enumerate(lay.title):
        o.append(f'<text class="{cls}" x="{MARGIN + 2:.0f}" y="{MARGIN + 14 + i * 18:.0f}">{escape(text)}</text>')
    o.append("</g>")

    # ---- lanes
    x0, y0, x1, y1 = lay.panel
    o.append('<g class="dz-lanes">')
    o.append(f'<clipPath id="dz-panel-clip"><rect x="{x0:.1f}" y="{y0:.1f}" width="{x1 - x0:.1f}" height="{y1 - y0:.1f}" rx="9"/></clipPath>')
    o.append(f'<rect class="dz-panel" x="{x0:.1f}" y="{y0:.1f}" width="{x1 - x0:.1f}" height="{y1 - y0:.1f}" rx="9"/>')
    if lay.head_w:
        o.append(f'<rect class="dz-lane-head" x="{x0:.1f}" y="{y0:.1f}" width="{lay.head_w:.1f}" height="{y1 - y0:.1f}" clip-path="url(#dz-panel-clip)"/>')
    for i, lane in enumerate(lay.lanes):
        o.append(f'<g class="dz-lane" data-group={quoteattr(lane.group)}>')
        if i:
            o.append(f'<line class="dz-lane-rule" x1="{x0:.1f}" y1="{lane.y0:.1f}" x2="{x1:.1f}" y2="{lane.y0:.1f}"/>')
        mid = (lane.y0 + lane.y1) / 2 - (len(lane.lines) - 1) * 7.0 + 4
        for k, text in enumerate(lane.lines):
            o.append(f'<text class="dz-lane-name" x="{x0 + 11:.1f}" y="{mid + k * 14:.1f}">{escape(text)}</text>')
        o.append("</g>")
    o.append(f'<rect x="{x0:.1f}" y="{y0:.1f}" width="{x1 - x0:.1f}" height="{y1 - y0:.1f}" rx="9" fill="none" class="dz-lane-rule"/>')
    o.append("</g>")

    # ---- connections
    o.append('<g class="dz-connections">')
    for e in lay.edges:
        kc = _kind_class(e.kind)
        cls = f"dz-connection dz-kind-{kc}" + (" dz-subject" if e.subject else "")
        tip = f"{e.name}: {names.get(e.src, e.src)} → {names.get(e.dst, e.dst)} ({e.kind}{', ' + e.medium if e.medium else ''})."
        if e.description:
            tip += " " + e.description.strip()
        if e.subject:
            tip += (f" Effects on {comp.name}: " + ", ".join(_summary(e.effects)) + "." if e.effects
                    else f" No effect on {comp.name}.")
        attrs = (f'id={quoteattr(e.id)} class="{cls}" data-kind={quoteattr(e.kind)} data-from={quoteattr(e.src)} '
                 f'data-to={quoteattr(e.dst)} data-medium={quoteattr(e.medium)} data-effects={quoteattr(_data_effects(e.effects))}'
                 + (' data-subject="true"' if e.subject else ""))
        o.append(f"<g {attrs}><title>{escape(tip)}</title>")
        if e.points:
            d_full, d = _path(e.points), _path(e.points, trim=3.5)
            o.append(f'<path class="dz-hit" d="{d_full}"/>')
            if KINDS.get(e.kind, (0, "", False, False))[3]:
                o.append(f'<path class="dz-line" d="{d}"/>')
                o.append(f'<path class="dz-inner" d="{d}" marker-end="url(#dz-arrow-{kc})"/>')
            else:
                o.append(f'<path class="dz-line" d="{d}" marker-end="url(#dz-arrow-{kc})"/>')
            o.append('<g class="dz-label">')
            for blk in e.blocks:
                tx = blk.x + (blk.w / 2 if blk.anchor == "middle" else 3 if blk.anchor == "start" else blk.w - 3)
                for k, (text, tcls) in enumerate(blk.lines):
                    ty = blk.y + 1 + (k + 0.78) * blk.line_height
                    o.append(f'<text class="{tcls}" x="{tx:.1f}" y="{ty:.1f}" text-anchor="{blk.anchor}">{escape(text)}</text>')
            o.append("</g>")
        o.append("</g>")
    o.append("</g>")

    # ---- elements
    o.append('<g class="dz-elements">')
    for b in lay.boxes.values():
        cls = f"dz-element dz-kind-{_slug(b.kind)}" + (" dz-subject" if b.subject else "")
        tip = f"{b.name} ({b.kind})."
        if b.description:
            tip += " " + b.description.strip()
        if b.subject:
            tip += f" Component under verification: {comp.name} ({comp.id})."
        if b.effects:
            tip += (" Effects on the component: " if b.subject else f" Effects on {comp.name}: ") + ", ".join(_summary(b.effects)) + "."
        attrs = (f'id={quoteattr(b.id)} class="{cls}" data-kind={quoteattr(b.kind)} data-group={quoteattr(b.group)} '
                 f'data-effects={quoteattr(_data_effects(b.effects))}'
                 + (f' data-subject="true" data-component={quoteattr(b.component)}' if b.subject else ""))
        o.append(f"<g {attrs}><title>{escape(tip)}</title>")
        o.append(f'<rect class="dz-box" x="{b.x:.1f}" y="{b.y:.1f}" width="{b.w:.1f}" height="{b.h:.1f}" rx="7"/>')
        n = len(b.lines)
        tag = 17.0 if b.subject else 0.0
        top = b.y + b.h / 2 - (tag + n * 15 + 13) / 2
        if b.subject:
            tw = text_width(TAG, 9.0, bold=True) + 14
            o.append(f'<g class="dz-tag"><rect x="{b.x + b.w / 2 - tw / 2:.1f}" y="{top:.1f}" width="{tw:.1f}" height="14" rx="7"/>'
                     f'<text x="{b.x + b.w / 2:.1f}" y="{top + 10.2:.1f}">{TAG}</text></g>')
        top += tag
        for k, text in enumerate(b.lines):
            o.append(f'<text class="dz-name" x="{b.x + b.w / 2:.1f}" y="{top + 12 + k * 15:.1f}">{escape(text)}</text>')
        o.append(f'<text class="dz-kind" x="{b.x + b.w / 2:.1f}" y="{top + n * 15 + 11:.1f}">{escape(b.kind)}</text>')
        o.append("</g>")
    o.append("</g>")

    # ---- legend
    o.append('<g class="dz-legend">')
    for it in lay.legend:
        x, y = it["x"], it["y"]
        if it["type"] == "line":
            kc = _kind_class(it["kind"])
            d = f"M{x:.1f} {y:.1f}L{x + 26.5:.1f} {y:.1f}"
            if KINDS.get(it["kind"], (0, "", False, False))[3]:
                o.append(f'<path class="dz-line dz-kind-{kc}" d="{d}"/><path class="dz-inner" style="stroke:{_var("--dz-bg")}" d="{d}" marker-end="url(#dz-arrow-{kc})"/>')
            else:
                o.append(f'<path class="dz-line dz-kind-{kc}" d="{d}" marker-end="url(#dz-arrow-{kc})"/>')
            o.append(f'<text x="{x + 38:.1f}" y="{y + 3.6:.1f}">{escape(it["text"])}</text>')
        elif it["type"] == "subject":
            o.append(f'<g class="dz-subject"><rect class="dz-box" x="{x + 2:.1f}" y="{y - 6.5:.1f}" width="26" height="13" rx="3.5" style="stroke-width:2"/></g>')
            o.append(f'<text x="{x + 38:.1f}" y="{y + 3.6:.1f}">{escape(it["text"])}</text>')
        else:
            o.append(f'<text x="{x:.1f}" y="{y + 3.6:.1f}">{escape(it["text"])}</text>')
    o.append("</g>")
    o.append("</svg>")
    return "\n".join(o) + "\n"
