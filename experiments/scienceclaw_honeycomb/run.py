"""Input-only ScienceClaw SM-W1 comparison; mm, N, MPa throughout."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import sys
import time

import gmsh
import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from feacore.case import Analysis, BC, Case, Material, QoI
from feacore.solver import solve

SOURCE = 'https://alanchen.cc/s/scienceclaw-run2-xteam-feagent-hlyniu5qon4yd7xipr5l/'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def outline(path):
    m = trimesh.load(path, force='mesh', process=True)
    assert m.is_watertight and len(m.split()) == 1
    v = np.asarray(m.vertices)
    assert np.all(np.isclose(v[:, 2], 0) | np.isclose(v[:, 2], 15))
    f = m.faces[np.all(np.isclose(v[m.faces, 2], 0), axis=1)]
    edges = np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1)
    e, count = np.unique(edges, axis=0, return_counts=True)
    adj = {}
    for a, b in e[count == 1]:
        adj.setdefault(int(a), []).append(int(b))
        adj.setdefault(int(b), []).append(int(a))
    assert all(len(n) == 2 for n in adj.values())
    remaining, loops = set(adj), []
    while remaining:
        start = min(remaining)
        loop, prev, cur = [start], None, start
        while True:
            nxt = next(n for n in adj[cur] if n != prev)
            if nxt == start:
                break
            loop.append(nxt)
            prev, cur = cur, nxt
        remaining.difference_update(loop)
        loops.append(v[loop, :2].copy())
    def area(p):
        return .5 * np.sum(p[:, 0] * np.roll(p[:, 1], -1) - p[:, 1] * np.roll(p[:, 0], -1))
    loops.sort(key=lambda p: abs(area(p)), reverse=True)
    info = {'bounds_mm': m.bounds.tolist(), 'volume_mm3': float(m.volume),
            'cross_section_mm2': float(m.volume / 15), 'loops': len(loops),
            'vertices': len(v), 'stl_triangles': len(m.faces)}
    return loops, info


def prepare(stl, out):
    loops, info = outline(stl)
    exterior = loops[0]
    segments = list(zip(exterior, np.roll(exterior, -1, axis=0)))
    probes, splits = [], {}
    for side in ['left', 'right']:
        ribs = []
        for i, (a, b) in enumerate(segments):
            # Long straight vertical OUTER rib faces. Two bar-attached half-rows
            # touch y~0.3 and y~72.7; exclude these, retaining exactly nine free rows.
            if (abs(b[1] - a[1]) > 2 and abs(b[0] - a[0]) < 1e-5
                and min(a[1], b[1]) > 4 and max(a[1], b[1]) < 70
                and ((max(a[0], b[0]) < 7) if side == 'left' else (min(a[0], b[0]) > 36))):
                ribs.append((i, (a + b) / 2, a, b))
        ribs.sort(key=lambda x: x[1][1])
        assert len(ribs) == 9, (side, len(ribs))
        for row, (i, p, a, b) in enumerate(ribs, 1):
            name = f'{side}_{row}'
            splits[i] = (name, p)
            probes.append({'name': name, 'side': side, 'row': row, 'point_mm': p.tolist(),
                           'rib_endpoints_mm': [a.tolist(), b.tolist()],
                           'outward_x_sign': -1 if side == 'left' else 1})
    y0, y1 = exterior[:, 1].min(), exterior[:, 1].max()
    for i, (a, b) in enumerate(segments):
        if abs(a[1]-y0) < 1e-7 and abs(b[1]-y0) < 1e-7:
            splits[i] = ('anchor', (a+b)/2)
    augmented, names = [], {}
    for i, (a, b) in enumerate(segments):
        augmented.append(a)
        if i in splits:
            name, p = splits[i]
            augmented.append(p)
            names[len(augmented)-1] = name
    loops[0] = np.asarray(augmented)
    left = [p['point_mm'][1] for p in probes if p['side'] == 'left']
    right = [p['point_mm'][1] for p in probes if p['side'] == 'right']
    assert np.allclose(left, right, atol=1e-5)
    setup = {'source': SOURCE, 'input_hashes': {'SM-W1.stl': digest(stl),
             'INPUTS.md': digest(Path(stl).with_name('INPUTS.md'))},
             'geometry': info, 'gauge_height_mm': 73., 'depth_mm': 15.,
             'strains': [.015, .03], 'primary_E_MPa': 1670., 'primary_nu': .35,
             'sensitivity_nu': .40, 'other_E_MPa': 2230.,
             'initial_mesh_sizes_mm': [.6, .3, .15], 'added_refinement_mm': .075, 'order': 2,
             'probes': probes,
             'row_definition': 'Nine free long exterior vertical ribs, excluding the two half-rows attached to loading bars; midpoint of each straight outer face.',
             'bonded': 'Outer bottom face ux=uy=0; outer top face ux=0,uy=-travel. Bars remain deformable.',
             'frictionless': 'Both outer faces constrained in y only; one bottom-centre point ux=0 removes rigid translation. Closed-contact roller idealisation, no unilateral contact.',
             'model': 'Homogeneous isotropic small-strain linear elasticity, plane strain. No damage, plasticity, geometric nonlinearity, contact or buckling.',
             'independence': 'No peer FE predictions, Instron data or photos read before freeze.'}
    out.mkdir(parents=True, exist_ok=True)
    (out/'SETUP.json').write_text(json.dumps(setup, indent=2)+'\n')
    np.savez(out/'outline.npz', **{f'loop_{i}': p for i,p in enumerate(loops)},
             point_names=np.array([json.dumps(names)]))
    return setup, loops, names


def make_mesh(path, loops, names, h, expected_area):
    gmsh.initialize()
    try:
        gmsh.option.setNumber('General.Terminal', 0)
        gmsh.option.setNumber('General.NumThreads', 1)
        gmsh.option.setNumber('Mesh.RandomSeed', 0)
        gmsh.model.add('SM-W1')
        cl, groups = [], {'top': [], 'bottom': []}
        y0, y1 = loops[0][:,1].min(), loops[0][:,1].max()
        for k,p in enumerate(loops):
            pts = [gmsh.model.geo.addPoint(float(x),float(y),0,h) for x,y in p]
            lines = []
            for i, (a,b) in enumerate(zip(pts,pts[1:]+pts[:1])):
                line = gmsh.model.geo.addLine(a,b)
                lines.append(line)
                if k == 0:
                    q = p[[i,(i+1)%len(p)]]
                    if np.all(abs(q[:,1]-y0)<1e-7): groups['bottom'].append(line)
                    if np.all(abs(q[:,1]-y1)<1e-7): groups['top'].append(line)
            if k == 0: probe_tags = {n:pts[i] for i,n in names.items()}
            cl.append(gmsh.model.geo.addCurveLoop(lines))
        surf = gmsh.model.geo.addPlaneSurface(cl)
        gmsh.model.geo.synchronize()
        gmsh.model.addPhysicalGroup(2,[surf],name='solid')
        for n,lines in groups.items():
            assert lines
            gmsh.model.addPhysicalGroup(1,lines,name=n)
        for n,tag in probe_tags.items(): gmsh.model.addPhysicalGroup(0,[tag],name=n)
        gmsh.option.setNumber('Mesh.Algorithm', 6)
        gmsh.option.setNumber('Mesh.MshFileVersion', 4.1)
        gmsh.model.mesh.generate(2)
        tags,coords,_ = gmsh.model.mesh.getNodes()
        coords = np.asarray(coords).reshape(-1,3)
        lookup = {int(n):i for i,n in enumerate(tags)}
        et,_,en = gmsh.model.mesh.getElements(2)
        assert list(et) == [2]
        tri = np.array([lookup[int(n)] for n in en[0]]).reshape(-1,3)
        p = coords[tri,:2]
        a = abs((p[:,1,0]-p[:,0,0])*(p[:,2,1]-p[:,0,1])-(p[:,1,1]-p[:,0,1])*(p[:,2,0]-p[:,0,0]))*.5
        assert a.min()>0 and abs(a.sum()/expected_area-1)<1e-9
        gmsh.write(str(path))
        return {'nodes':len(tags),'triangles':len(tri),'area_mm2':float(a.sum()),
                'min_triangle_area_mm2':float(a.min()),'target_h_mm':h}
    finally:
        gmsh.finalize()


def make_case(mesh, bc, nu, E=1670.):
    travel = .03*73
    if bc == 'bonded':
        bcs = [BC('fixed',group='bottom'),BC('displacement',group='top',components=[0,1],value=[0.,-travel])]
    else:
        bcs = [BC('displacement',group='bottom',components=[1],value=0.),
               BC('displacement',group='top',components=[1],value=-travel),
               BC('displacement',group='anchor',components=[0],value=0.)]
    return Case(mesh=str(mesh.resolve()), analysis=Analysis('elasticity','plane_strain',order=2,thickness=15.,
                options={'linear_solver':'scipy'}),materials={'AlphaFast_assumed':Material(E=E,nu=nu)},bcs=bcs,
                qois=[QoI(f'R_{g}_{k}','reaction',group=g,component=k) for g in ['top','bottom'] for k in [0,1]])


def run_case(mesh, setup, bc, nu, out, E=1670.):
    out.mkdir(parents=True,exist_ok=True)
    case = make_case(mesh,bc,nu,E)
    # Portable stored case; runtime case above points to the generated mesh.
    case_dict = case.to_dict()
    case_dict['mesh'] = os.path.relpath(mesh,out)
    (out/'case.json').write_text(json.dumps(case_dict,indent=2)+'\n')
    started = time.perf_counter()
    r = solve(case,backend='skfem')
    md = r.model
    q = r.evaluate_all()
    free = np.setdiff1d(np.arange(len(r.x)),r.dirichlet_dofs)
    residual = r.reaction_vector[free]
    rhs = -r.K[free][:,r.dirichlet_dofs] @ r.x[r.dirichlet_dofs]
    rel = float(np.linalg.norm(residual)/max(np.linalg.norm(rhs),1e-300))
    force = -q['R_top_1']
    balance = float(np.linalg.norm([q['R_top_0']+q['R_bottom_0'],q['R_top_1']+q['R_bottom_1']])/force)
    energy = float(.5*r.x@(r.K@r.x))
    work = .5*(.03*73)*force
    energy_error = abs(energy/work-1)
    assert force>0 and rel<1e-8 and balance<1e-7 and energy_error<1e-7
    rows=[]
    for p in setup['probes']:
        idx = md.sdofs_of(p['name'])
        assert len(idx)==1
        i = idx[0]
        xy = md.sb.doflocs[:,i]
        assert np.linalg.norm(xy-p['point_mm'])<1e-9
        ux,uy = float(r.fields['ux'][i]),float(r.fields['uy'][i])
        rows.append({**p,'ux_mm_3pct':ux,'uy_mm_3pct':uy,
                     'outward_mm_3pct':p['outward_x_sign']*ux,
                     'outward_mm_1p5pct':p['outward_x_sign']*ux*.5})
    # Quadrature-level diagnostics expose local rotations/strains hidden by
    # nominal platen strain. They are not plasticity or failure criteria.
    gx = md.sb.interpolate(r.fields['ux']).grad
    gy = md.sb.interpolate(r.fields['uy']).grad
    ex,ey,exy = gx[0],gy[1],.5*(gx[1]+gy[0])
    radius = np.sqrt((.5*(ex-ey))**2+exy**2)
    eig_lo,eig_hi = .5*(ex+ey)-radius,.5*(ex+ey)+radius
    strain_diag = {'min_inplane_principal_strain_qp':float(eig_lo.min()),
                   'max_inplane_principal_strain_qp':float(eig_hi.max()),
                   'max_abs_linear_rotation_rad_qp':float(abs(.5*(gy[0]-gx[1])).max()),
                   'min_deformed_map_det_qp':float(((1+gx[0])*(1+gy[1])-gx[1]*gy[0]).min()),
                   'note':'Unrecovered quadrature extrema can be mesh-sensitive near joints/supports; no material failure or nonlinear validity verdict.'}
    coords,ux = r.nodal('ux')
    np.savez_compressed(out/'fields.npz',coordinates_mm=coords,ux_mm=ux,uy_mm=r.fields['uy'],
                        triangles_p2=md.sb.element_dofs.T)
    result = {'info':asdict(r.info),'bc':bc,'E_MPa':E,'nu':nu,'travel_mm_3pct':2.19,
              'force_N_3pct':force,'force_N_1p5pct':force*.5,
              'force_N_3pct_scaled_to_E2230':force*2230/E,
              'rows':rows,'reactions_N_3pct':q,'strain_energy_Nmm':energy,
              'relative_free_residual':rel,'relative_force_imbalance':balance,
              'strain_diagnostics':strain_diag,
              'relative_energy_work_error':energy_error,'elapsed_s':time.perf_counter()-started,
              'peak_process_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024),
              'input_hashes':setup['input_hashes'],'run_script_sha256':digest(__file__),
              'software':{n:importlib.metadata.version(n) for n in ['numpy','scipy','scikit-fem','gmsh','trimesh']},
              'platform':platform.platform(),
              'scaling_note':'1.5% outputs and E2230 force are exact algebraic scalings of the same homogeneous linear model, not separate FE solves.'}
    (out/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ['bc','nu','force_N_3pct','relative_free_residual','elapsed_s']}),flush=True)
    return result


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--out',type=Path,default=ROOT/'evaluation_runs/scienceclaw_honeycomb')
    ap.add_argument('--stl',type=Path,default=Path(__file__).parent/'inputs/SM-W1.stl')
    ap.add_argument('--h',type=float,nargs='+',default=[.6,.3,.15])
    ap.add_argument('--nu',type=float,nargs='+',default=[.35])
    ap.add_argument('--E',type=float,default=1670.)
    ap.add_argument('--bc',nargs='+',default=['bonded','frictionless'],choices=['bonded','frictionless'])
    a=ap.parse_args()
    setup,loops,names=prepare(a.stl,a.out)
    for h in a.h:
        mesh=a.out/f'mesh_h{h:g}.msh'
        info=make_mesh(mesh,loops,names,h,setup['geometry']['cross_section_mm2'])
        (a.out/f'mesh_h{h:g}.json').write_text(json.dumps(info,indent=2)+'\n')
        for nu in a.nu:
            for bc in a.bc:
                run_case(mesh,setup,bc,nu,a.out/f'{bc}_nu{nu:g}_h{h:g}',E=a.E)


if __name__=='__main__': main()
