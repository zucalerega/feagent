"""Independent continuum FE calculation of the supplied TAP-2.5D model.

The input data are CC BY 4.0, from SentiLab's pinned reference package.
Solver code is the existing FEAgent backend; no parameters are fitted to targets.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import meshio
import numpy as np
import scipy
import skfem
from skfem import MeshTet

from feacore.case import Analysis, BC, Case, Load, Material, Region
from feacore.solver import solve

TOL = 1e-12


def unique(values):
    out = []
    for value in sorted(values):
        if not out or value - out[-1] > TOL:
            out.append(value)
    return np.array(out)


def split_axis(edges, h, refinement):
    base = unique(edges)
    # Scale subdivision counts uniformly in every original interval.
    return np.concatenate([
        np.linspace(a, b, max(1, math.ceil((b-a)/h)) * refinement + 1)[:-1]
        for a, b in zip(base[:-1], base[1:])
    ] + [base[-1:]])


def inside(x, y, rect):
    return ((x >= rect['x0_m'] - TOL) &
            (x <= rect['x0_m'] + rect['width_m'] + TOL) &
            (y >= rect['y0_m'] - TOL) &
            (y <= rect['y0_m'] + rect['height_m'] + TOL))


def make_mesh(data, cid, refinement, out):
    layout = data['cases'][cid]
    edges = [[], []]
    for layer in data['stack']:
        e = layer['extent_m']
        edges[0] += [e['x0'], e['x1']]
        edges[1] += [e['y0'], e['y1']]
    for blocks in layout['layers'].values():
        for b in blocks:
            edges[0] += [b['x0_m'], b['x0_m'] + b['width_m']]
            edges[1] += [b['y0_m'], b['y0_m'] + b['height_m']]
    for b in layout['chiplets']:
        edges[0].append(b['x_centre_m'])
        edges[1].append(b['y_centre_m'])
    axes = [split_axis(v, .012, refinement) for v in edges]
    z = []
    for layer in data['stack']:
        n = (3 if layer['index'] == 7 else 1) * refinement
        z.extend(np.linspace(layer['z_bottom_m'], layer['z_top_m'], n+1)[:-1])
    axes.append(np.array(z + [data['model_top_z_m']]))
    m = MeshTet.init_tensor(*axes)
    ctr = m.p[:, m.t].mean(axis=1)
    keep = np.zeros(m.nelements, dtype=bool)
    for layer in data['stack']:
        e = layer['extent_m']
        keep |= ((ctr[2] > layer['z_bottom_m']) & (ctr[2] < layer['z_top_m']) &
                 (ctr[0] > e['x0']) & (ctr[0] < e['x1']) &
                 (ctr[1] > e['y0']) & (ctr[1] < e['y1']))
    m = m.remove_elements(np.flatnonzero(~keep)).remove_unused_nodes()
    ctr = m.p[:, m.t].mean(axis=1)
    tags = np.zeros(m.nelements, dtype=np.int32)
    regions, materials, loads, groups = [], {}, [], {}
    layer_index = np.zeros(m.nelements, dtype=np.int8)
    tag = 1
    expected_volume = 0.
    powers = {}
    for layer in data['stack']:
        idx = layer['index']
        sel = (ctr[2] > layer['z_bottom_m']) & (ctr[2] < layer['z_top_m'])
        layer_index[sel] = idx
        blocks = layout['layers'].get(f'L{idx}')
        if blocks is None:
            e = layer['extent_m']
            blocks = [dict(name=layer['name'], x0_m=e['x0'], y0_m=e['y0'],
                           width_m=e['x1']-e['x0'], height_m=e['y1']-e['y0'],
                           k_W_per_mK=layer['default_k_W_per_mK'], power_W=0)]
        for b in blocks:
            pick = sel & inside(ctr[0], ctr[1], b)
            if np.any(tags[pick]):
                raise ValueError(f'Overlapping material assignment: L{idx}/{b["name"]}')
            if not np.any(pick):
                raise ValueError(f'Missing block in mesh: L{idx}/{b["name"]}')
            name = f'L{idx}_{b["name"]}'.replace(' ', '_').replace(';', '_')
            tags[pick] = tag
            groups[name] = [tag, 3]
            materials[name] = Material(k=b['k_W_per_mK'])
            regions.append(Region(name, name))
            if b['power_W']:
                volume = b['width_m'] * b['height_m'] * layer['thickness_m']
                loads.append(Load('source', b['power_W']/volume, group=name))
                powers[name] = b['power_W']
            tag += 1
        missing = sel & (tags == 0)
        if np.any(missing):
            if cid != 'A' or idx not in (3, 4):
                raise ValueError(f'Uncovered geometry in L{idx}')
            rects = layout['tiling_check'][f'L{idx}']['uncovered_rectangles_m']
            declared = np.zeros(m.nelements, dtype=bool)
            for x0, x1, y0, y1 in rects:
                declared |= inside(ctr[0], ctr[1], dict(x0_m=x0, y0_m=y0,
                                   width_m=x1-x0, height_m=y1-y0))
            if np.any(missing & ~declared):
                raise ValueError('Gap extends beyond declared A4 strip')
            name = f'L{idx}_A4_underfill'
            tags[missing] = tag
            groups[name] = [tag, 3]
            materials[name] = Material(k=1.6)
            regions.append(Region(name, name))
            tag += 1
        e = layer['extent_m']
        expected_volume += (e['x1']-e['x0'])*(e['y1']-e['y0'])*layer['thickness_m']
    verts = m.p[:, m.t]
    jac = np.transpose(verts[:, 1:] - verts[:, :1], (2, 0, 1))
    volumes = np.abs(np.linalg.det(jac)) / 6
    if not np.isclose(volumes.sum(), expected_volume, rtol=1e-11, atol=1e-16):
        raise ValueError('Meshed volume does not equal exact package volume')
    for name, power in powers.items():
        density = next(ld.value for ld in loads if ld.group == name)
        actual = float(volumes[tags == groups[name][0]].sum()*density)
        if not np.isclose(actual, power, rtol=1e-10, atol=1e-9):
            raise ValueError(f'Wrong integrated power in {name}: {actual} vs {power}')
    # Include the internal die-bottom interface for the surface-source variant.
    fctr = m.p[:, m.facets].mean(axis=1)
    surf = np.zeros(m.nfacets, dtype=np.int32)
    surf[np.isclose(fctr[2], data['model_top_z_m'], atol=TOL, rtol=0)] = tag
    groups['sink_top'] = [tag, 2]
    tag += 1
    for chip in layout['chiplets']:
        pick = (np.isclose(fctr[2], data['metric']['z_die_bottom_m'], atol=TOL, rtol=0)
                & inside(fctr[0], fctr[1], chip))
        surf[pick] = tag
        groups[f'source_{chip["flp_name"]}'] = [tag, 2]
        tag += 1
    selected = surf > 0
    path = out / 'mesh.msh'
    meshio.write(path, meshio.Mesh(m.p.T,
        cells=[('tetra', m.t.T), ('triangle', m.facets[:, selected].T)],
        cell_data={'gmsh:physical': [tags, surf[selected]],
                   'gmsh:geometrical': [tags, surf[selected]]}, field_data=groups),
        file_format='gmsh22', binary=True)
    edge_lengths = np.linalg.norm(m.p[:, m.edges[0]]-m.p[:, m.edges[1]], axis=0)
    meta = dict(nodes=int(m.nvertices), elements=int(m.nelements),
                minimum_edge_m=float(edge_lengths.min()), maximum_edge_m=float(edge_lengths.max()),
                subdivisions_per_layer={str(i): (3 if i == 7 else 1)*refinement for i in range(8)},
                exact_volume_m3=expected_volume, meshed_volume_m3=float(volumes.sum()))
    del verts, jac, fctr
    return m, axes, layer_index, materials, regions, loads, meta


class TensorProbe:
    """Exact element lookup by tensor cell, avoiding nearest-centroid ambiguity.

    Very thin layers can defeat a global nearest-centroid search. Here every
    point is tested only against the six tetrahedra of its containing box.
    """
    def __init__(self, result, axes):
        self.result = result
        self.axes = axes
        mesh = result.model.mesh
        self.origin = mesh.p[:, mesh.t[0]].T
        corners = mesh.p[:, mesh.t]
        self.inverse = np.linalg.inv(np.transpose(corners[:, 1:]-corners[:, :1], (2, 0, 1)))
        ctr = corners.mean(axis=1)
        ijk = [np.searchsorted(a, ctr[i], side='right')-1 for i, a in enumerate(axes)]
        self.nxy = [len(a)-1 for a in axes]
        key = ijk[0] + self.nxy[0]*(ijk[1] + self.nxy[1]*ijk[2])
        self.lookup = {}
        for e, k in enumerate(key):
            self.lookup.setdefault(int(k), []).append(e)

    def on_elements(self, points, elements, coefficients=None):
        p = np.atleast_2d(points)
        elements = np.asarray(elements)
        X = np.einsum('pij,pj->pi', self.inverse[elements], p-self.origin[elements]).T
        sb = self.result.model.sb
        coefficients = self.result.x if coefficients is None else coefficients
        out = np.zeros(len(p))
        for j in range(sb.element_dofs.shape[0]):
            phi, _ = sb.elem.lbasis(X, j)
            out += phi*coefficients[sb.element_dofs[j, elements]]
        return out

    def __call__(self, points, coefficients=None):
        p = np.atleast_2d(points)
        ijk = [np.clip(np.searchsorted(a, p[:, i], side='right')-1, 0, len(a)-2)
               for i, a in enumerate(self.axes)]
        key = ijk[0] + self.nxy[0]*(ijk[1] + self.nxy[1]*ijk[2])
        out = np.empty(len(p))
        for k in np.unique(key):
            inds = np.flatnonzero(key == k)
            es = np.array(self.lookup.get(int(k), []))
            if len(es) != 6:
                raise ValueError(f'Probe outside domain or incomplete tensor cell: {p[inds[0]]}')
            delta = p[inds, None, :] - self.origin[es][None, :, :]
            xi = np.einsum('eij,pej->pei', self.inverse[es], delta)
            bary = np.concatenate([1-xi.sum(axis=2, keepdims=True), xi], axis=2)
            inside_score = bary.min(axis=2)
            choice = inside_score.argmax(axis=1)
            if np.any(inside_score[np.arange(len(inds)), choice] < -1e-7):
                raise ValueError('Probe was not contained in any tetrahedron')
            element = es[choice]
            out[inds] = self.on_elements(p[inds], element, coefficients)
        return out

    def validate(self):
        rng = np.random.default_rng(173)
        mesh = self.result.model.mesh
        es = rng.choice(mesh.nelements, min(100, mesh.nelements), replace=False)
        points = mesh.p[:, mesh.t[:, es]].mean(axis=1).T
        def field(p):
            x, y, z = np.asarray(p).T
            v = 2 + 3*x - 5*y + 7*z
            return v if self.result.model.order == 1 else v + 11*x*x + 13*y*z
        coeff = field(self.result.model.sb.doflocs.T)
        error = float(np.max(np.abs(self(points, coeff)-field(points))))
        if error > 1e-10:
            raise ValueError(f'Independent interpolation check failed: {error}')
        z, rect = .000465, (.01, .02, .015, .025)
        mx, my = (rect[0]+rect[1])/2, (rect[2]+rect[3])/2
        expected = 2+3*mx-5*my+7*z
        if self.result.model.order == 2:
            expected += 11*(rect[0]**2+rect[0]*rect[1]+rect[1]**2)/3+13*my*z
        plane_error = abs(self.rectangle_mean(z, rect, coeff)-expected)
        if plane_error > 1e-10:
            raise ValueError(f'Independent plane integration check failed: {plane_error}')
        self.plane_check_error = plane_error
        return error

    @staticmethod
    def clip(poly, axis, bound, sign):
        if not len(poly):
            return poly
        out = []
        for a, b in zip(poly, np.roll(poly, -1, axis=0)):
            ia, ib = sign*(a[axis]-bound) >= -1e-15, sign*(b[axis]-bound) >= -1e-15
            if ia:
                out.append(a)
            if ia != ib:
                t = (bound-a[axis])/(b[axis]-a[axis])
                out.append(a+t*(b-a))
        return np.array(out)

    def rectangle_mean(self, z, rect, coefficients=None):
        """Integrate the actual piecewise P1/P2 FE field on a plane rectangle.

        Intersect each crossed tetrahedron with the plane, clip to the rectangle,
        triangulate the convex polygon and use degree-2-exact triangle quadrature.
        This avoids treating a few point samples as an area average.
        """
        x0, x1, y0, y1 = rect
        ix0 = max(0, np.searchsorted(self.axes[0], x0, side='right')-1)
        ix1 = min(self.nxy[0], np.searchsorted(self.axes[0], x1, side='left'))
        iy0 = max(0, np.searchsorted(self.axes[1], y0, side='right')-1)
        iy1 = min(self.nxy[1], np.searchsorted(self.axes[1], y1, side='left'))
        iz = min(self.nxy[2]-1, np.searchsorted(self.axes[2], z, side='right')-1)
        mesh = self.result.model.mesh
        points, elements, weights = [], [], []
        for ix in range(ix0, ix1):
            for iy in range(iy0, iy1):
                key = ix + self.nxy[0]*(iy+self.nxy[1]*iz)
                for e in self.lookup.get(int(key), []):
                    v = mesh.p[:, mesh.t[:, e]].T
                    poly = [a[:2] for a in v if abs(a[2]-z) < 1e-14]
                    for i in range(4):
                        for j in range(i+1, 4):
                            if (v[i, 2]-z)*(v[j, 2]-z) < 0:
                                t = (z-v[i, 2])/(v[j, 2]-v[i, 2])
                                poly.append((v[i]+t*(v[j]-v[i]))[:2])
                    distinct = []
                    for p in poly:
                        if not any(np.linalg.norm(p-q) < 1e-14 for q in distinct):
                            distinct.append(p)
                    if len(distinct) < 3:
                        continue
                    poly = np.array(distinct)
                    center = poly.mean(axis=0)
                    angle = np.arctan2(poly[:, 1]-center[1], poly[:, 0]-center[0])
                    poly = poly[np.argsort(angle)]
                    for axis, bound, sign in [(0, x0, 1), (0, x1, -1), (1, y0, 1), (1, y1, -1)]:
                        poly = self.clip(poly, axis, bound, sign)
                        if len(poly) < 3:
                            break
                    for j in range(1, len(poly)-1):
                        tri = np.array([poly[0], poly[j], poly[j+1]])
                        a, b = tri[1]-tri[0], tri[2]-tri[0]
                        area = abs(a[0]*b[1]-a[1]*b[0])/2
                        if area < 1e-24:
                            continue
                        for bary in [(2/3, 1/6, 1/6), (1/6, 2/3, 1/6), (1/6, 1/6, 2/3)]:
                            xy = np.asarray(bary)@tri
                            points.append([xy[0], xy[1], z])
                            elements.append(e)
                            weights.append(area/3)
        weights = np.array(weights)
        expected_area = (x1-x0)*(y1-y0)
        if abs(weights.sum()/expected_area-1) > 1e-8:
            raise ValueError(f'Plane integration area mismatch: {weights.sum()} vs {expected_area}')
        values = self.on_elements(points, elements, coefficients)
        return float(weights@values/expected_area)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', type=Path, default=Path(__file__).parent/'inputs/case.json')
    ap.add_argument('--case', choices=['A', 'B'], required=True)
    ap.add_argument('--refinement', type=int, default=1)
    ap.add_argument('--order', type=int, choices=[1, 2], default=2)
    ap.add_argument('--variant', choices=['volume', 'surface'], default='volume')
    ap.add_argument('--solver', default='pardiso')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--mesh-only', action='store_true')
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    data = json.loads(args.input.read_text())
    layout = data['cases'][args.case]
    start = time.monotonic()
    mesh, axes, layer_ids, materials, regions, loads, meta = make_mesh(
        data, args.case, args.refinement, args.out)
    print('MESH', json.dumps(meta), flush=True)
    (args.out/'mesh_info.json').write_text(json.dumps(meta, indent=2))
    if args.mesh_only:
        return
    robin = data['boundary_conditions']['convective_face']
    # Solve for temperature EXCESS to keep the residual and balance well scaled.
    # Add the prescribed ambient to every temperature readout afterwards.
    bcs = [BC('convection', group='sink_top', h=robin['h_W_per_m2K'], T_inf=0.)]
    if args.variant == 'surface':
        loads = []
        bcs += [BC('flux', group=f'source_{c["flp_name"]}', value=c['power_density_W_m2'])
                for c in layout['chiplets']]
    case = Case('mesh.msh', Analysis(physics='heat', formulation='3d', order=args.order,
                options={'linear_solver': args.solver}), materials, bcs, loads=loads, regions=regions,
                meta={'unknown': 'temperature excess above ambient', 'ambient_K': robin['T_ref_K']})
    case.base_dir = str(args.out)
    case.save(args.out/'feacore_case.json')
    result = solve(case)
    probe = TensorProbe(result, axes)
    interpolation_check = probe.validate()
    ambient = robin['T_ref_K']
    coords, values = result.nodal('T')
    chip_values = {}
    for chip in layout['chiplets']:
        points = [[chip['x_centre_m'], chip['y_centre_m'], data['metric'][key]]
                  for key in ['z_die_bottom_m', 'z_die_mid_m', 'z_die_top_m']]
        vals = probe(points) + ambient
        chip_values[chip['flp_name']] = dict(zip(['bottom_K', 'mid_K', 'top_K'], map(float, vals)))
    winner = max(chip_values, key=lambda k: chip_values[k]['mid_K'])
    M = chip_values[winner]['mid_K'] - 273.15
    cell_averages = {}
    cache = {}
    for layer in data['stack'][:6]:
        idx = layer['index']
        z = (layer['z_bottom_m']+layer['z_top_m'])/2
        for block in layout['layers'][f'L{idx}']:
            value = 0.
            for cell in block['hotspot_centre_cells_64']:
                rect = tuple(cell[k] for k in ['x0_m', 'x1_m', 'y0_m', 'y1_m'])
                key = (z, *rect)
                if key not in cache:
                    cache[key] = probe.rectangle_mean(z, rect)+ambient
                value += cell['weight']*cache[key]
            cell_averages[f'L{idx}/{block["name"]}'] = value
    selector = max(cell_averages, key=cell_averages.get)
    md = result.model
    fb = md.facet_basis(md.facets_of('sink_top'))
    area = float(fb.dx.sum())
    qout = float(np.sum(robin['h_W_per_m2K']*fb.interpolate(result.x)*fb.dx))
    power = float(result.f.sum())
    residual = float(np.linalg.norm(result.K@result.x-result.f)/np.linalg.norm(result.f))
    if not np.isclose(area, robin['area_m2'], rtol=1e-10):
        raise ValueError('Wrong cooled area')
    if not np.isclose(power, layout['total_power_W'], rtol=1e-9):
        raise ValueError('Wrong total source power')
    if abs(qout-power)/power > 1e-7 or residual > 1e-7:
        raise ValueError(f'Balance/residual failure: {qout}, {power}, {residual}')
    layer_extremes = {}
    for idx in range(8):
        dofs = np.unique(md.sb.element_dofs[:, layer_ids == idx])
        maxi = dofs[np.argmax(result.x[dofs])]
        mini = dofs[np.argmin(result.x[dofs])]
        layer_extremes[f'L{idx}'] = dict(max_K=float(result.x[maxi]+ambient),
            max_at_m=md.sb.doflocs[:, maxi].tolist(), min_K=float(result.x[mini]+ambient),
            min_at_m=md.sb.doflocs[:, mini].tolist())
    out = dict(case=args.case, variant=args.variant, refinement=args.refinement, order=args.order,
        mesh=meta, dofs=result.info.n_dofs, M_C=M, hottest_die=winner,
        paper_target_C=layout['paper_target_C'], difference_from_paper_K=M-layout['paper_target_C'],
        chip_temperatures=chip_values, P_in_W=power, Q_out_W=qout,
        hotspot_cell_averages_K=cell_averages,
        selector_analogue_C=cell_averages[selector]-273.15, selector_block=selector,
        interpolation_polynomial_check_max_error=interpolation_check,
        plane_integration_polynomial_check_error=probe.plane_check_error,
        layer_nodal_extremes=layer_extremes,
        energy_relative_error=(qout-power)/power, residual_rel=residual,
        mean_sink_top_excess_K=qout/(robin['h_W_per_m2K']*area),
        field_max_K=float(values.max()+ambient), at_max_m=coords[values.argmax()].tolist(),
        field_min_K=float(values.min()+ambient), at_min_m=coords[values.argmin()].tolist(),
        time_s=time.monotonic()-start, solver_time_s=result.info.wall_time,
        peak_memory_MB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024 if sys.platform!='darwin' else 1024**2),
        versions={'python': sys.version, 'platform': platform.platform(), 'numpy': np.__version__,
                  'scipy': scipy.__version__, 'scikit-fem': skfem.__version__, 'meshio': meshio.__version__},
        input_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        warnings=result.info.warnings,
        limitations=['Field extrema are over FE solution nodes; P2 extrema between nodes are not searched.',
                     'Continuum package differs from the HotSpot compact network as documented by the input author.'])
    (args.out/'result.json').write_text(json.dumps(out, indent=2))
    np.savez_compressed(args.out/'temperature.npz', coords=coords, excess_K=values)
    print('RESULT', json.dumps(out), flush=True)


if __name__ == '__main__':
    main()
