"""What a careful engineer reads off the CAD before meshing: the agents' inspect_geometry.

describe(problem) returns a compact, JSON-ready dict:
  bbox, diagonal, volume (area in 2D);
  every boundary entity (3D: faces, 2D: curves) with its named group, type, size, centroid,
  outward normal when flat, neighbours, and whether / how the statement names it;
  every feature where two boundary entities meet (3D: CAD edges, 2D: corner points) with the
  interior angle of the material there; re-entrant ones (angle > 180 deg) are where linear
  elasticity has stress singularities, so the discretisation error concentrates there;
  the smallest features (shortest edge, smallest radius, thinnest wall);
  a short plain-language "hints" paragraph.

Pure geometry (gmsh OCC queries, no mesh, no reference data). gmsh is process-global, so call
this outside any other gmsh session. Results are cached per problem id.
"""

from __future__ import annotations

import json
import math
import re

import gmsh
import numpy as np

from feacore.geometry.gmsh_tools import gmsh_session
from feacore.problems.base import Problem

REENTRANT_DEG = 190.0    # interior angle above this: re-entrant (10 deg margin for curved faces)
SHARP_DEG = 75.0         # interior angle below this: an acute wedge, hard to mesh well
SMOOTH_DEG = 10.0        # within this of 180 deg the two entities meet tangentially (fillets)

_CACHE: dict[str, dict] = {}

# flow BC type -> the role inspect_geometry reports for the boundary
FLOW_ROLES = {"velocity_inlet": "inlet", "pressure_outlet": "outlet", "wall": "wall", "slip": "slip",
              "symmetry": "symmetry"}

# Statement words that say what a sentence does to the entities it names.
_ROLE_WORDS = [
    ("support", r"clamp|fixed|support|roller|symmetr|displacements? (are |is )?zero"),
    ("load", r"pressure|traction|force|load|moment|flux"),
    ("free", r"\bfree\b"),
]


def describe(problem: Problem) -> dict:
    """Geometry summary of `problem` (cached). See the module docstring."""
    from ace import config
    key = (problem.id, config.STATEMENT)      # the roles come from the statement the agent reads
    if key not in _CACHE:
        from ace import fea
        with gmsh_session(problem.id):
            if fea.is_flow(problem):
                fea.build_agent_geometry(problem)    # the CAD the agent meshes (its tags), not the blocks
            else:
                problem.build_geometry()
            gmsh.model.occ.synchronize()
            _CACHE[key] = _Describer(problem).run()
    return _CACHE[key]


def agent_statement(problem) -> str:
    """The statement as the agent reads it. In requirements mode (config.STATEMENT) that is the
    problem's requirements: its own description spells out the supports and loads, and the roles
    this module reports (supported / loaded faces) would hand them to the agent."""
    from ace import config
    if config.STATEMENT == "requirements":
        from ace import requirements
        return requirements.agent_text(problem.id)
    return problem.description()


# --------------------------------------------------------------------------- small helpers


def _unit(v) -> np.ndarray:
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def _angle_deg(a, b) -> float:
    return math.degrees(math.acos(float(np.clip(np.dot(_unit(a), _unit(b)), -1.0, 1.0))))


def _sig(x: float, n: int = 4) -> float:
    return float(f"{x:.{n}g}")


class _Describer:
    """One pass over the model of the current gmsh session. D = dimension of the part;
    'bnd' = its boundary entities (dim D-1), 'feat' = where they meet (dim D-2)."""

    def __init__(self, problem: Problem):
        self.p = problem
        self.D = problem.dim
        self.domain = [t for _, t in gmsh.model.getEntities(self.D)]
        lo = np.min([gmsh.model.getBoundingBox(self.D, t)[:3] for t in self.domain], axis=0)
        hi = np.max([gmsh.model.getBoundingBox(self.D, t)[3:] for t in self.domain], axis=0)
        self.lo, self.hi = lo, hi
        self.diag = float(np.linalg.norm(hi - lo))
        # coordinates to ~1e-4 of the part's size: enough to place refinements, short to read
        self.dec = max(0, 4 - math.floor(math.log10(self.diag)))
        self.eps = 1e-3 * self.diag   # probe distance for inside/outside tests
        self.desc = agent_statement(problem)
        # Requirements mode: deciding which faces are supported or loaded is the agent's task, and a
        # keyword match on requirements misreads them ("unloaded" counts as a load), so no roles.
        from ace import config
        self.infer_roles = config.STATEMENT != "requirements"
        self.groups: dict[int, list[str]] = {}   # bnd tag -> physical group names
        for d, pt in gmsh.model.getPhysicalGroups(self.D - 1):
            for e in gmsh.model.getEntitiesForPhysicalGroup(d, pt):
                self.groups.setdefault(int(e), []).append(gmsh.model.getPhysicalName(d, pt))
        # Flow: a boundary's role is its BC in the problem's setup (inlet, outlet, wall, ...), not a
        # support / load read off the statement.
        self.flow = getattr(problem, "physics", "") == "flow"
        self.flow_roles: dict[str, str] = {}
        if self.flow:
            for b in problem.case("REF.msh", order=2).bcs:
                if b.group:
                    self.flow_roles[b.group] = FLOW_ROLES.get(b.type, b.type)
                    if b.type == "wall" and b.value is not None:
                        self.flow_roles[b.group] = "moving wall"

    # ----------------------------------------------------------------- formatting

    def xyz(self, v) -> list[float]:
        v = [round(float(c), self.dec) + 0.0 for c in v]   # + 0.0: no "-0.0"
        return v[: self.D]

    def label(self, tag: int) -> str:
        # Gold parts name each face f<tag>; we use the same ids everywhere (c<tag> for 2D curves).
        return f"{'f' if self.D == 3 else 'c'}{tag}"

    # ----------------------------------------------------------------- geometry probes

    def inside_domain(self, x) -> bool:
        return any(gmsh.model.isInside(self.D, t, list(map(float, x))) for t in self.domain)

    def bnd_point(self, b: int):
        """A point inside boundary entity b (not on its rim) and the unit tangent / normal data
        there: 3D returns (x, normal), 2D returns (x, tangent)."""
        lo, hi = gmsh.model.getParametrizationBounds(self.D - 1, b)
        if self.D == 2:
            t = 0.5 * (lo[0] + hi[0])
            return gmsh.model.getValue(1, b, [t]), _unit(gmsh.model.getDerivative(1, b, [t]))
        # a 5x5 grid of the parameter box; trimmed faces can miss the middle
        for fu in (0.5, 0.3, 0.7, 0.1, 0.9):
            for fv in (0.5, 0.3, 0.7, 0.1, 0.9):
                uv = [lo[0] + fu * (hi[0] - lo[0]), lo[1] + fv * (hi[1] - lo[1])]
                if gmsh.model.isInside(2, b, uv, parametric=True):
                    return gmsh.model.getValue(2, b, uv), _unit(gmsh.model.getNormal(b, uv))
        return None, None

    def outward_normal(self, b: int):
        """Unit outward normal of a flat boundary entity (None if no interior point was found)."""
        x, d = self.bnd_point(b)
        if x is None:
            return None
        n = d if self.D == 3 else np.array([d[1], -d[0], 0.0])
        # the normal points outward iff a small step along it leaves the material
        return -n if self.inside_domain(np.asarray(x) + self.eps * n) else n

    def into(self, feat: int, b: int, at: np.ndarray, tangent):
        """Unit direction from feature `feat` (at point `at`) into boundary entity b,
        perpendicular to the feature. None if it cannot be decided."""
        if self.D == 2:
            # b is a curve with `feat` at one end: its end tangent, pointing away from the corner
            lo, hi = gmsh.model.getParametrizationBounds(1, b)
            ends = [(lo[0], 1.0), (hi[0], -1.0)]
            dist = [np.linalg.norm(np.asarray(gmsh.model.getValue(1, b, [t])) - at) for t, _ in ends]
            if min(dist) > 1e-6 * self.diag or abs(dist[0] - dist[1]) < 1e-9 * self.diag:
                return None   # closed curve through the corner: no unique direction
            t, s = ends[int(np.argmin(dist))]
            return s * _unit(gmsh.model.getDerivative(1, b, [t]))
        # 3D: the in-face direction perpendicular to the edge is +-(normal x tangent); keep the
        # sign whose small step stays on the (trimmed) face. The step is mapped to the face's
        # parameters with the surface Jacobian (first order; projecting points that lie off the
        # face is slow in OCC), then tested against the trim curves. Shrink it for narrow faces.
        uv = np.asarray(gmsh.model.getParametrization(2, b, list(at)))
        d = _unit(np.cross(gmsh.model.getNormal(b, list(uv)), tangent))
        J = np.asarray(gmsh.model.getDerivative(2, b, list(uv))).reshape(2, 3).T   # columns dx/du, dx/dv
        lo, hi = gmsh.model.getParametrizationBounds(2, b)
        duv = np.linalg.lstsq(J, d, rcond=None)[0]
        for eps in (self.eps, 0.1 * self.eps, 0.01 * self.eps):
            on = []
            for s in (1.0, -1.0):
                q = uv + s * eps * duv
                # off the parameter range = off an untrimmed surface's natural boundary
                on.append(bool(np.all(q >= lo) and np.all(q <= hi)
                               and gmsh.model.isInside(2, b, list(q), parametric=True)))
            if on[0] != on[1]:
                return d if on[0] else -d
        return None

    def interior_angle(self, feat: int, faces: list[int], at: np.ndarray, tangent) -> float | None:
        """Angle (deg) of the material wedge at the feature, measured between the two boundary
        entities: < 180 convex, 180 smooth, > 180 re-entrant."""
        dA, dB = (self.into(feat, b, at, tangent) for b in faces)
        if dA is None or dB is None:
            return None
        theta = _angle_deg(dA, dB)
        bis = dA + dB
        if np.linalg.norm(bis) < 1e-6:
            return 180.0
        # The material fills either the wedge of angle theta between dA and dB or its complement.
        return theta if self.inside_domain(at + self.eps * _unit(bis)) else 360.0 - theta

    # ----------------------------------------------------------------- statement roles

    def roles(self, names: list[str]) -> tuple[bool, str | None]:
        """(named in the statement?, role there: support / load / free / mentioned; flow: the BC)."""
        if self.flow:
            role = next((self.flow_roles.get(n, "wall") for n in names if n in self.flow_roles), None)
            if role is None and names and not names[0].startswith("c") and not names[0].startswith("f"):
                role = "wall"                 # a named boundary without a BC is a no-slip wall
            return bool(role), role
        # lines, then sentences (". " before a capital; decimals like 60.08 have no space)
        sentences = re.split(r"(?<=[.;])\s+(?=[A-Z])|\n", self.desc)
        found, role = False, None
        for name in names:
            pat = re.compile(rf"(?<![\w-]){re.escape(name)}(?![\w-])")
            for s in sentences:
                if pat.search(s):
                    found = True
                    for r, words in _ROLE_WORDS:
                        if re.search(words, s, re.I):
                            role = role or r
                            break
        if not found:
            return False, None
        if not self.infer_roles:
            return True, "mentioned"
        if role is None:   # e.g. a group called "clamped" named only in passing
            for r, words in _ROLE_WORDS:
                if any(re.search(words, n, re.I) for n in names):
                    role = r
                    break
        return True, role or "mentioned"

    # ----------------------------------------------------------------- main pass

    def run(self) -> dict:
        D = self.D
        bnd_dt = gmsh.model.getBoundary([(D, t) for t in self.domain], combined=True, oriented=False)
        bnd = sorted({abs(t) for _, t in bnd_dt})
        feats = sorted({abs(t) for _, t in gmsh.model.getBoundary([(D - 1, b) for b in bnd],
                                                                    combined=False, oriented=False)})
        bset = set(bnd)

        # features and their two boundary entities
        feat_out, feat_faces, seams = [], {}, 0
        for e in feats:
            up = [int(b) for b in gmsh.model.getAdjacencies(D - 2, e)[0] if int(b) in bset]
            if D == 3:
                length = gmsh.model.occ.getMass(1, e)
                if length < 1e-9 * self.diag:
                    continue   # degenerate edge (e.g. a cone apex)
                lo, hi = gmsh.model.getParametrizationBounds(1, e)
                tm = 0.5 * (lo[0] + hi[0])
                at = np.asarray(gmsh.model.getValue(1, e, [tm]))
                tangent = _unit(gmsh.model.getDerivative(1, e, [tm]))
            else:
                at, tangent, length = np.asarray(gmsh.model.getValue(0, e, [])), None, None
            if len(up) != 2:
                seams += 1     # a seam of a closed face (cylinder) or a corner on a closed curve
                continue
            feat_faces[e] = up
            ang = self.interior_angle(e, up, at, tangent)
            rec = {"tag": e}
            if D == 3:
                rec.update(type=gmsh.model.getType(1, e), length=_sig(length))
            rec["faces" if D == 3 else "curves"] = [self.label(b) for b in up]
            rec["dihedral_angle_deg" if D == 3 else "interior_angle_deg"] = \
                None if ang is None else round(ang, 1)
            rec["concave"] = None if ang is None else ang > REENTRANT_DEG
            rec["midpoint" if D == 3 else "point"] = self.xyz(at)
            feat_out.append(rec)

        # boundary entities
        neighbours: dict[int, set[int]] = {b: set() for b in bnd}
        for a, b in feat_faces.values():
            neighbours[a].add(b)
            neighbours[b].add(a)
        bnd_out, radii, flats = [], [], {}
        for b in bnd:
            typ = gmsh.model.getType(D - 1, b)
            names = self.groups.get(b, [])
            in_stmt, role = self.roles(names + [self.label(b)])
            rec = {"group": names[0] if names else None, "tag": b, "type": typ,
                   ("area" if D == 3 else "length"): _sig(gmsh.model.occ.getMass(D - 1, b)),
                   "centroid": self.xyz(gmsh.model.occ.getCenterOfMass(D - 1, b))}
            if typ in ("Plane", "Line"):
                n = self.outward_normal(b)
                if n is not None:
                    rec["normal"] = [round(float(c), 3) + 0.0 for c in n[:D]]
                    flats[b] = n
            elif typ in ("Cylinder", "Sphere", "Torus", "Circle"):
                r = self.radius(b)
                if r:
                    rec["radius"] = _sig(r)
                    radii.append((r, b))
            rec["adjacent_" + ("faces" if D == 3 else "curves")] = [self.label(o) for o in sorted(neighbours[b])]
            rec["in_statement"] = in_stmt
            if role:
                rec["role"] = role
            bnd_out.append(rec)

        return self.summarise(bnd_out, feat_out, seams, radii, flats, neighbours)

    def radius(self, b: int) -> float | None:
        """Smallest radius of curvature of a cylinder / sphere / torus face or a circular arc."""
        if self.D == 3:
            x, _ = self.bnd_point(b)
            if x is None:
                return None
            uv = gmsh.model.getParametrization(2, b, list(x))
            k1, k2, *_ = gmsh.model.getPrincipalCurvatures(b, uv)
            k = max(abs(k1[0]), abs(k2[0]))
        else:
            lo, hi = gmsh.model.getParametrizationBounds(1, b)
            k = abs(gmsh.model.getCurvature(1, b, [0.5 * (lo[0] + hi[0])])[0])
        return 1.0 / k if k > 0 else None

    def thinnest_wall(self, flats: dict, neighbours: dict) -> tuple[float, int, int] | None:
        """Thinnest wall between two parallel flat faces (curves in 2D) facing each other:
        outward normals opposite and each lying behind the other. Exact OCC distance."""
        cand = []
        items = list(flats.items())
        for i, (a, na) in enumerate(items):
            ca = np.asarray(gmsh.model.occ.getCenterOfMass(self.D - 1, a))
            for b, nb in items[i + 1:]:
                if b in neighbours[a] or np.dot(na, nb) > -0.99:
                    continue
                gap = float(np.dot(ca - np.asarray(gmsh.model.occ.getCenterOfMass(self.D - 1, b)), na))
                if gap > 1e-9 * self.diag:   # b lies on the material side of a
                    cand.append((gap, a, b))
        best = None
        for gap, a, b in sorted(cand)[:40]:   # the 40 closest plane pairs are plenty
            d = gmsh.model.occ.getDistance(self.D - 1, a, self.D - 1, b)[0]
            if d > 1e-9 * self.diag and (best is None or d < best[0]):
                best = (d, a, b)
        return best

    # ----------------------------------------------------------------- summary + hints

    def summarise(self, bnd_out, feat_out, seams, radii, flats, neighbours) -> dict:
        D, L = self.D, self.label
        fk, ek = ("faces", "edges") if D == 3 else ("curves", "corners")
        angle_key = "dihedral_angle_deg" if D == 3 else "interior_angle_deg"
        reentrant = sorted((f for f in feat_out if f["concave"]), key=lambda f: -f[angle_key])
        sharp = sorted((f for f in feat_out if f[angle_key] is not None and f[angle_key] < SHARP_DEG),
                       key=lambda f: f[angle_key])
        by_label = {L(r["tag"]): r for r in bnd_out}
        sup = [r for r in bnd_out if r.get("role") == "support"]
        load = [r for r in bnd_out if r.get("role") == "load"]

        def rim(group_recs):
            # features where a supported (loaded) entity meets one that is not
            labels = {L(r["tag"]) for r in group_recs}
            key = "faces" if D == 3 else "curves"
            return [f["tag"] for f in feat_out if len(labels & set(f[key])) == 1]

        out = {
            "dim": D,
            "units": getattr(self.p, "units", ""),
            "bbox": self.xyz(self.lo)[:D] + self.xyz(self.hi)[:D],
            "diagonal": _sig(self.diag),
            ("volume" if D == 3 else "area"): _sig(sum(gmsh.model.occ.getMass(D, t) for t in self.domain)),
            fk: bnd_out,
            ek: feat_out,
            "reentrant_" + ek: [f["tag"] for f in reentrant],
            "sharp_" + ek: [f["tag"] for f in sharp],
            "support_boundary": rim(sup),   # stress singular where a clamp ends, whatever the angle
            "load_boundary": rim(load),
        }
        smallest = {}
        if D == 3 and feat_out:
            e = min(feat_out, key=lambda f: f["length"])
            smallest["shortest_edge"] = {"tag": e["tag"], "length": e["length"]}
        if D == 2 and bnd_out:
            c = min(bnd_out, key=lambda r: r["length"])
            smallest["shortest_curve"] = {"tag": c["tag"], "length": c["length"]}
        if radii:
            r, b = min(radii)
            smallest["smallest_radius"] = {fk[:-1]: L(b), "radius": _sig(r)}
        wall = self.thinnest_wall(flats, neighbours)
        if wall:
            smallest["thinnest_wall"] = {fk: [L(wall[1]), L(wall[2])], "thickness": _sig(wall[0])}
        out["smallest_features"] = smallest
        out["hints"] = self.hints(out, reentrant, sharp, sup, load, by_label, seams, angle_key)
        return out

    def flow_hints(self, out, reentrant, sharp, angle_key) -> str:
        """The plain-words summary for a flow domain: where the fluid enters and leaves, the walls
        (boundary layers), the corners the flow separates at, curved walls."""
        D = self.D
        fk, ek = ("faces", "edges") if D == 3 else ("curves", "corners")
        g = lambda x: f"{x:.3g}"
        unit = " m" if out["units"].startswith("SI") else ""
        h = [f"The fluid domain spans {' x '.join(g(b - a) for a, b in zip(out['bbox'][:D], out['bbox'][D:]))}{unit}"
             f" (diagonal {g(self.diag)}), with {len(out[fk])} boundary {fk} and {len(out[ek])} {ek}."]
        by_role: dict[str, list[str]] = {}
        for r in out[fk]:
            if r.get("role"):
                by_role.setdefault(r["role"], []).append(r["group"] or self.label(r["tag"]))
        for role in ("inlet", "outlet", "wall", "moving wall", "slip", "symmetry"):
            if role in by_role:
                names = ", ".join(dict.fromkeys(by_role[role]))
                h.append(f"{role.capitalize()}: {names}.")
        if any(r in by_role for r in ("wall", "moving wall")):
            h.append("No slip: the fluid takes each wall's velocity, so the velocity changes fast across a thin "
                     "layer along the walls; a boundary layer of thin cells resolves it cheaply.")
        feat_by_tag = {f["tag"]: f for f in out[ek]}
        role_of = {self.label(r["tag"]): r.get("role") for r in out[fk]}
        jumps = [f for f in out[ek] if {role_of.get(c) for c in f["curves" if D == 2 else "faces"]}
                 == {"moving wall", "wall"}]

        def ref(tag):
            return str(tag) if D == 3 else f"{tag} at ({', '.join(f'{c:g}' for c in feat_by_tag[tag]['point'])})"
        pair_key = "faces" if D == 3 else "curves"
        if reentrant:
            lst = ", ".join(f"{ref(f['tag'])} ({f[angle_key]:.0f} deg, {'/'.join(f[pair_key])})" for f in reentrant[:6])
            h.append(f"{len(reentrant)} re-entrant {ek if len(reentrant) > 1 else ek[:-1]} (fluid angle > 180 deg): "
                     f"{lst}. The flow separates at such a corner; the corner and the shear layer leaving it "
                     "downstream need small cells, and a recirculation zone forms behind it. "
                     + ('Refine along it with {"edge": tag}.' if D == 3 else 'Refine around it with {"point": [x, y]}.'))
        if jumps:
            h.append(f"A moving wall meets a stationary one at {ek} "
                     + ", ".join(ref(f["tag"]) for f in jumps)
                     + ": the wall velocity jumps there, the flow is singular and the error concentrates; small "
                       "cells around them.")
        if sharp:
            h.append(f"Acute {ek} (< {SHARP_DEG:.0f} deg; slow corner flow in a thin wedge): "
                     + ", ".join(f"{ref(f['tag'])} ({f[angle_key]:.0f} deg)" for f in sharp[:5]) + ".")
        curved = [r for r in out[fk] if r["type"] not in ("Plane", "Line")]
        if curved:
            h.append(f"{len(curved)} curved boundary {fk if len(curved) > 1 else fk[:-1]}: straight-sided cells "
                     "facet them, so cells along them must be small enough to keep the wetted shape.")
        sm = out["smallest_features"]
        if "smallest_radius" in sm:
            h.append(f"Smallest radius {g(sm['smallest_radius']['radius'])}.")
        return " ".join(h)

    def hints(self, out, reentrant, sharp, sup, load, by_label, seams, angle_key) -> str:
        if self.flow:
            return self.flow_hints(out, reentrant, sharp, angle_key)
        D = self.D
        fk, ek = ("faces", "edges") if D == 3 else ("curves", "corners")
        one = fk[:-1]
        g = lambda x: f"{x:.3g}"
        types: dict[str, int] = {}
        for r in out[fk]:
            types[r["type"]] = types.get(r["type"], 0) + 1
        unit = " m" if out["units"].startswith("SI") else ""
        h = [f"The part spans {' x '.join(g(b - a) for a, b in zip(out['bbox'][:D], out['bbox'][D:]))}{unit}"
             + f" (diagonal {g(self.diag)}), with {len(out[fk])} {fk} ("
             + ", ".join(f"{n} {t}" for t, n in sorted(types.items(), key=lambda kv: -kv[1]))
             + f") and {len(out[ek])} {ek}."]

        def names(recs):
            return ", ".join(r["group"] or self.label(r["tag"]) for r in recs)
        if sup:
            h.append(f"Supported: {names(sup)}.")
        if load:
            h.append(f"Loaded: {names(load)}.")
        if not sup and not load:
            h.append("The statement names no boundary entity by its group: read it to place the "
                     "supports and loads.")

        feat_by_tag = {f["tag"]: f for f in out[ek]}
        pair_key = "faces" if D == 3 else "curves"

        def ref(tag):   # 2D corners are refined with {"point": [x, y]}: say where they are
            return str(tag) if D == 3 else f"{tag} at ({', '.join(f'{c:g}' for c in feat_by_tag[tag]['point'])})"
        if reentrant:
            # list first the ones on a supported or loaded entity: they drive the scored quantities
            named = {self.label(r["tag"]) for r in sup + load}
            reentrant = sorted(reentrant, key=lambda f: (not named & set(f[pair_key]), -f[angle_key]))
            lst = ", ".join(f"{ref(f['tag'])} ({f[angle_key]:.0f} deg, {'/'.join(f[pair_key])})" for f in reentrant[:6])
            more = f" and {len(reentrant) - 6} more" if len(reentrant) > 6 else ""
            h.append(f"{len(reentrant)} re-entrant {ek if len(reentrant) > 1 else ek[:-1]} "
                     f"(material angle > 180 deg; stress singularities, the error concentrates there): "
                     f"{lst}{more}. "
                     + ('Refine along them with {"edge": tag}.' if D == 3
                        else 'Refine around them with {"point": [x, y]}.'))
        else:
            h.append(f"No re-entrant {ek}.")
        # where a support or load meets a re-entrant feature: the worst spots
        for kind, recs in (("supported", sup), ("loaded", load)):
            for r in recs:
                lab = self.label(r["tag"])
                for f in reentrant:
                    if lab in f[pair_key]:
                        other = [x for x in f[pair_key] if x != lab][0]
                        h.append(f"The {kind} {one} {r['group'] or lab} meets {other} at re-entrant "
                                 f"{ek[:-1]} {f['tag']}.")
        if out["support_boundary"]:
            tags = out["support_boundary"]
            h.append(f"The support ends at {ek} {', '.join(map(ref, tags[:8]))}"
                     f"{' ...' if len(tags) > 8 else ''}: a clamped-to-free transition is a stress "
                     f"singularity even where the surface is smooth, so refine there too.")
        if out["load_boundary"]:
            tags = out["load_boundary"]
            h.append(f"The load ends at {ek} {', '.join(map(ref, tags[:8]))}"
                     f"{' ...' if len(tags) > 8 else ''} (a traction jump: milder, local).")
        if sharp:
            h.append(f"Acute {ek} (< {SHARP_DEG:.0f} deg, thin wedges that need small elements): "
                     + ", ".join(f"{ref(f['tag'])} ({f[angle_key]:.0f} deg)" for f in sharp[:5]) + ".")
        sm = out["smallest_features"]
        parts = []
        if "shortest_edge" in sm:
            parts.append(f"shortest edge {sm['shortest_edge']['tag']} ({g(sm['shortest_edge']['length'])})")
        if "shortest_curve" in sm:
            parts.append(f"shortest curve {sm['shortest_curve']['tag']} ({g(sm['shortest_curve']['length'])})")
        if "smallest_radius" in sm:
            parts.append(f"smallest radius {g(sm['smallest_radius']['radius'])} at {sm['smallest_radius'][one]}")
        if "thinnest_wall" in sm:
            w = sm["thinnest_wall"]
            parts.append(f"thinnest wall {g(w['thickness'])} between {' and '.join(w[fk])}")
        if parts:
            h.append("Smallest features: " + "; ".join(parts) + ". Elements much larger than these "
                     "blur them (a wall needs about 2 quadratic elements through its thickness).")
        curved = sum(n for t, n in types.items() if t not in ("Plane", "Line"))
        if curved:
            h.append(f"{curved} curved boundary {fk if curved > 1 else one}: quadratic (order 2) elements follow "
                     f"them far better than linear ones.")
        return " ".join(h)


# --------------------------------------------------------------------------- the agent's view

AGENT_MAX_CHARS = 12_000   # above this, inspect_geometry shows the agent the compact form below


def for_agent(full: dict) -> dict:
    """What inspect_geometry returns to the agent, given `full` = geometry_summary | describe.

    A part that fits in AGENT_MAX_CHARS of JSON is returned unchanged. A bigger one (a gold CAD part
    has ~140 edges, ~44k characters) keeps every face, one text line each, but lists in full only
    the edges that decide where to refine: re-entrant, sharp, where a support or load ends, and
    every edge of a face the statement names. The other edges are summarised as counts; they are
    convex or smooth, so they need no local refinement. The groups that are just the faces
    (f<tag>, one entity each) are not repeated."""
    if len(json.dumps(full)) <= AGENT_MAX_CHARS:
        return full
    D = full["dim"]
    fk, ek = ("faces", "edges") if D == 3 else ("curves", "corners")
    ang_key = "dihedral_angle_deg" if D == 3 else "interior_angle_deg"
    pt_key = "midpoint" if D == 3 else "point"
    xyz = lambda v: "(" + ",".join(f"{c:g}" for c in v) + ")"
    label = (lambda t: f"f{t}") if D == 3 else (lambda t: f"c{t}")

    face_lines = []
    for r in full[fk]:
        size = r.get("area", r.get("length"))
        s = f"{label(r['tag'])} {r['type']} {'A' if D == 3 else 'L'}={size:g} c={xyz(r['centroid'])}"
        if "normal" in r:
            s += f" n={xyz(r['normal'])}"
        if "radius" in r:
            s += f" r={r['radius']:g}"
        s += " adj " + ",".join(r["adjacent_" + fk])
        if r["group"] and r["group"] != label(r["tag"]):
            s += f" group {r['group']}"
        if r["in_statement"]:
            s += f" NAMED in statement ({r.get('role', 'mentioned')})"
        face_lines.append(s)

    named = {label(r["tag"]) for r in full[fk] if r["in_statement"]}
    important = (set(full["reentrant_" + ek]) | set(full["sharp_" + ek])
                 | set(full["support_boundary"]) | set(full["load_boundary"]))
    edge_lines, rest = [], []
    for e in full[ek]:
        if e["tag"] in important or named & set(e[fk]):
            a = e[ang_key]
            s = f"{e['tag']}"
            if D == 3:
                s += f" {e['type']} L={e['length']:g}"
            s += f" {'/'.join(e[fk])} {'?' if a is None else f'{a:g}'}deg"
            if e["concave"]:
                s += " RE-ENTRANT"
            edge_lines.append(s + f" at {xyz(e[pt_key])}")
        else:
            rest.append(e)
    by_type: dict[str, int] = {}
    for e in rest:
        by_type[e.get("type", "corner")] = by_type.get(e.get("type", "corner"), 0) + 1
    angles = [e[ang_key] for e in rest if e[ang_key] is not None]
    other = (f"{len(rest)} more {ek}, not listed (none re-entrant, sharp, on a support or load rim, "
             f"or on a named {fk[:-1]}): " + ", ".join(f"{n} {t}" for t, n in sorted(by_type.items(), key=lambda kv: -kv[1]))
             + (f"; angles {min(angles):g}-{max(angles):g} deg" if angles else "") + ".")

    face_names = {label(r["tag"]) for r in full[fk]}
    groups = [g for g in full["groups"] if not (g["name"] in face_names and g["n_entities"] == 1)]
    keep = ("dim", "units", "bbox", "diagonal", "volume", "area")
    return {**{k: full[k] for k in keep if k in full},
            "groups": groups,
            "legend": (f"Each {fk[:-1]} f<tag> is also a named group (BCs and refinements can use it). "
                       f"{fk}: label, type, A/L = area/length, c = centroid, n = outward normal, r = radius, "
                       f"adj = adjacent {fk}. {ek}: tag, type, L = length, its two {fk}, material angle "
                       f"(> 180 = re-entrant), a point on it."),
            fk: face_lines,
            ek: edge_lines,
            "other_" + ek: other,
            **{k: full[k] for k in ("reentrant_" + ek, "sharp_" + ek, "support_boundary", "load_boundary",
                                    "smallest_features", "hints")}}
