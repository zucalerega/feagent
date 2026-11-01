"""Generate the frozen report from completed input-only linear runs."""
from pathlib import Path
import csv
import hashlib
import json
import os
import sys
import tempfile

os.environ.setdefault('MPLCONFIGDIR',str(Path(tempfile.gettempdir())/'feagent-mpl'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

ROOT=Path(__file__).resolve().parents[2]
out=Path(sys.argv[1]) if len(sys.argv)>1 else ROOT/'evaluation_runs/scienceclaw_honeycomb'
setup=json.loads((out/'SETUP.json').read_text())
runs=[]
for p in sorted(out.glob('*/result.json')):
    r=json.loads(p.read_text())
    r['run']=p.parent.name
    r['h_mm']=float(p.parent.name.rsplit('_h',1)[1])
    runs.append(r)
assert len(runs)>=6
fine=min(r['h_mm'] for r in runs if r['nu']==.35)
def pick(bc,nu=.35,h=fine):
    return next(r for r in runs if r['bc']==bc and r['nu']==nu and r['h_mm']==h)

convergence={}
for bc in ['bonded','frictionless']:
    rr=sorted([r for r in runs if r['bc']==bc and r['nu']==.35],key=lambda r:-r['h_mm'])
    a,b=rr[-2:]
    convergence[bc]={'finest_h_mm':fine,'last_force_change_N':b['force_N_3pct']-a['force_N_3pct'],
        'last_force_change_relative':b['force_N_3pct']/a['force_N_3pct']-1,
        'max_last_row_change_mm':max(abs(x['outward_mm_3pct']-y['outward_mm_3pct']) for x,y in zip(a['rows'],b['rows'])),
        'qualification':'Observed mesh-to-mesh changes, not certified error bounds. Polygonal STL geometry fixed; meshes are not nested.'}

with (out/'row_outputs.csv').open('w',newline='') as f:
    fields=['run','bc','nu','E_MPa','h_mm','row','side','x_mm','y_mm','outward_mm_1p5pct','outward_mm_3pct']
    w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
    for r in runs:
        for row in r['rows']:
            w.writerow({**{k:r[k] for k in fields[:5]},**{k:row[k] for k in ['row','side','outward_mm_1p5pct','outward_mm_3pct']},
                        'x_mm':row['point_mm'][0],'y_mm':row['point_mm'][1]})
summary={'setup':setup,'runs':runs,'convergence':convergence}
(out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')

fig,axes=plt.subplots(1,2,figsize=(11,7),sharey=True)
for ax,bc in zip(axes,['bonded','frictionless']):
    for nu in [.35,.4]:
        candidates=[r for r in runs if r['bc']==bc and r['nu']==nu]
        if not candidates:continue
        r=min(candidates,key=lambda r:r['h_mm'])
        for side,style in [('left','-o'),('right','--s')]:
            rows=[x for x in r['rows'] if x['side']==side]
            ax.plot([x['outward_mm_3pct'] for x in rows],[x['point_mm'][1] for x in rows],style,
                    ms=4,label=f'{side}, nu={nu:g}, h={r["h_mm"]:g} mm')
    ax.axvline(0,color='0.6',lw=.8)
    ax.set(title=bc.capitalize()+' platens',xlabel='Outward displacement at 3% (mm)',ylabel='Reference rib midpoint y (mm)')
    ax.grid(alpha=.25);ax.legend(fontsize=8)
fig.tight_layout();fig.savefig(out/'edge_profiles.png',dpi=160);plt.close(fig)

r=pick('bonded')
fields=np.load(out/r['run']/'fields.npz')
outline=np.load(out/'outline.npz')
fig,ax=plt.subplots(figsize=(6,9))
for key in outline.files:
    if not key.startswith('loop_'):continue
    p=outline[key]
    pp=np.vstack([p,p[0]])
    ax.plot(pp[:,0],pp[:,1],color='0.77',lw=.6)
# Recover all boundary edges from P2 connectivity and sample their quadratic
# displacement; connecting only original polygon corners would hide rib bending.
tri=fields['triangles_p2']
pairs=np.sort(np.concatenate([tri[:,[0,1]],tri[:,[1,2]],tri[:,[2,0]]]),axis=1)
mid=np.concatenate([tri[:,3],tri[:,4],tri[:,5]])
edges,first,count=np.unique(pairs,axis=0,return_index=True,return_counts=True)
ids=np.column_stack([edges[count==1],mid[first[count==1]]])
t=np.linspace(0,1,5)
weights=np.array([2*(t-.5)*(t-1),2*t*(t-.5),4*t*(1-t)])
deformed=fields['coordinates_mm']+np.column_stack([fields['ux_mm'],fields['uy_mm']])
curves=np.einsum('eic,ij->ejc',deformed[ids],weights)
ax.add_collection(LineCollection(curves,colors='#156b96',linewidths=.8))
for row in r['rows']:
    p=np.array(row['point_mm']);d=p+np.array([row['ux_mm_3pct'],row['uy_mm_3pct']])
    ax.plot(*d,'o',ms=3,color='#b9442e')
    if row['side']=='left':ax.text(d[0]-1,d[1],str(row['row']),ha='right',fontsize=8)
ax.set(title='SM-W1: bonded, 3% compression\nActual displacement scale (1x); gray = undeformed',xlabel='x (mm)',ylabel='y (mm)')
ax.set_aspect('equal');ax.grid(alpha=.15);fig.tight_layout();fig.savefig(out/'deformed_shape.png',dpi=160);plt.close(fig)

lines=['# ScienceClaw SM-W1: independent linear FE comparison','',
       'Input-only ScienceClaw geometry/loading; small-strain P2 plane-strain model, depth15mm. No peer predictions, experimental values or photos were read before freezing this package. This is a simulation of the supplied model; experimental agreement is not yet evaluated.','',
       '![Edge outputs](edge_profiles.png)','',
       '## Model and output definitions','',
       'mm/N/MPa; primary E=1670MPa, nu=0.35, sensitivity nu=0.40. Source gauge height73mm gives travel1.095/2.190mm for1.5/3%. Bars remain deformable. Bonded: bottom outer face fixed, top outer face ux=0 and uy=-travel. Frictionless: outer faces prescribed uy, zero tangential traction, one bottom-centre ux anchor; idealised closed-contact rollers, no unilateral detachment or self-contact.','',
       setup['row_definition']+' Probe coordinates and rib endpoints are in SETUP.json; each is an exact mesh vertex. Positive outward=-ux(left),+ux(right); negative values mean inward. Slight source asymmetry retained.','',
       '1.5% results are exactly half the3% linear solution. Forces for E=2230MPa are exactly scaled by2230/1670; these are algebraic consequences of this homogeneous linear model, not separate runs. Actual resin cure state and material law are not inferred.','',
       '## Mesh refinement at nu=0.35','',
       '| Platens | Target h (mm) | Triangles | DOFs | Force at 3% (N) | Relative residual | Time (s) |',
       '|---|---:|---:|---:|---:|---:|---:|']
for bc in ['bonded','frictionless']:
    for q in sorted([r for r in runs if r['bc']==bc and r['nu']==.35],key=lambda r:-r['h_mm']):
        lines.append(f'| {bc} | {q["h_mm"]:g} | {q["info"]["n_cells"]} | {q["info"]["n_dofs"]} | {q["force_N_3pct"]:.6f} | {q["relative_free_residual"]:.2e} | {q["elapsed_s"]:.2f} |')
    q=convergence[bc]
    lines+=['',f'Last refinement, {bc}: force change {q["last_force_change_N"]:+.6f}N ({100*q["last_force_change_relative"]:+.4f}%); maximum row-output change {q["max_last_row_change_mm"]:.7f}mm.','']
    if bc=='bonded':
        lines+=['| Platens | Target h (mm) | Triangles | DOFs | Force at 3% (N) | Relative residual | Time (s) |',
                '|---|---:|---:|---:|---:|---:|---:|']
lines+=['Observed changes are not certified error bounds. Boundary geometry is the fixed polygonal STL, not an inferred exact curved design; meshes are not nested. Timing is a single run, not a performance benchmark.','',
        '## Finest force and row outputs','',
        '| Platens | E (MPa) | nu | h (mm) | Force at 1.5% (N) | Force at 3% (N) | Force at 3%, scaled to E2230 (N) |',
        '|---|---:|---:|---:|---:|---:|---:|']
for bc in ['bonded','frictionless']:
    for nu in [.35,.4]:
        rr=[r for r in runs if r['bc']==bc and r['nu']==nu]
        if not rr:continue
        q=min(rr,key=lambda r:r['h_mm'])
        lines.append(f'| {bc} | {q["E_MPa"]:g} | {nu:g} | {q["h_mm"]:g} | {q["force_N_1p5pct"]:.6f} | {q["force_N_3pct"]:.6f} | {q["force_N_3pct_scaled_to_E2230"]:.6f} |')
for bc in ['bonded','frictionless']:
    q=pick(bc);left=[x for x in q['rows'] if x['side']=='left'];right=[x for x in q['rows'] if x['side']=='right']
    lines+=['',f'### {bc.capitalize()}, nu0.35, h{fine:g}mm','',
            '| Row | y (mm) | Left outward at 1.5% (mm) | Right at 1.5% | Left at 3% | Right at 3% |',
            '|---|---:|---:|---:|---:|---:|']
    for a,b in zip(left,right):
        lines.append(f'| {a["row"]} | {a["point_mm"][1]:.6f} | {a["outward_mm_1p5pct"]:.7f} | {b["outward_mm_1p5pct"]:.7f} | {a["outward_mm_3pct"]:.7f} | {b["outward_mm_3pct"]:.7f} |')
lines+=['','![Deformed model](deformed_shape.png)','',
        '## Verification and limitations','',
        'The independent affine rectangle test checks the exact plane-strain displacement and reaction E/(1-nu²)*strain*width*depth. Its results are in verification/checks.json. Every panel run checks positive compressive force, free-DOF residual<1e-8, force imbalance<1e-7 and Clapeyron energy/work error<1e-7. Source area matches STL volume/15 to relative1e-9.','',
        'Quadrature-level local strains/rotations and the deformed-map determinant are retained per run as diagnostics. Local strain can exceed the nominal platen strain. These values are not failure predictions, and convergence of the linear solution does not validate small-strain kinematics or resin behaviour experimentally. No plasticity, geometric nonlinearity, self-contact, buckling, damage or fracture result is claimed.','',
        f'For scale, the finest bonded nu0.35 run has quadrature in-plane principal-strain extrema {pick("bonded")["strain_diagnostics"]["min_inplane_principal_strain_qp"]:.5f} / {pick("bonded")["strain_diagnostics"]["max_inplane_principal_strain_qp"]:.5f}, and maximum absolute linear rotation {pick("bonded")["strain_diagnostics"]["max_abs_linear_rotation_rad_qp"]:.5f}rad. These extrema are mesh-sensitive near joints/corners; a small global mesh change does not establish that geometric or material nonlinear corrections are small.','',
        'All rows, both Poisson ratios, reactions, fields, software versions, runtime and process memory are retained. CSV: row_outputs.csv; machine-readable summary: summary.json. Input hashes and all probe coordinates: SETUP.json. Small-strain source and reproduction details: experiments/scienceclaw_honeycomb/README.md. Large field/mesh files are omitted from the compact Git handoff; their hashes are in FIELDS_SHA256SUMS, with copies retained locally and regeneration documented.','',
        'Input authorship: ScienceClaw, SM-W1 input-only package, '+setup['source']+'. No input redistribution or new licence is implied.','']
(out/'REPORT.md').write_text('\n'.join(lines))
files=[];large=[]
for base in [ROOT/'experiments/scienceclaw_honeycomb',out]:
    for p in sorted(base.rglob('*')):
        if not p.is_file() or p.name.startswith('._') or '__pycache__' in p.parts or 'inputs' in p.parts or p.name in ['SHA256SUMS','FIELDS_SHA256SUMS']:continue
        entry=hashlib.sha256(p.read_bytes()).hexdigest()+'  '+str(p.relative_to(ROOT))
        (large if p.suffix in ['.npz','.msh'] else files).append(entry)
(out/'SHA256SUMS').write_text('\n'.join(files)+'\n')
(out/'FIELDS_SHA256SUMS').write_text('\n'.join(large)+'\n')
print(json.dumps(convergence,indent=2))
