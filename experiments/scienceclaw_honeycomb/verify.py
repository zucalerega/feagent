"""Independent affine plane-strain check of constraints, depth and reactions."""
import json
from pathlib import Path
import sys

import gmsh
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from feacore.case import Case,Analysis,Material,BC,QoI
from feacore.solver import solve

out=ROOT/'evaluation_runs/scienceclaw_honeycomb/verification'
out.mkdir(parents=True,exist_ok=True)
gmsh.initialize()
try:
    gmsh.option.setNumber('General.Terminal',0)
    gmsh.model.add('affine_check')
    xy=[(-4,0),(0,0),(4,0),(4,10),(-4,10)]
    p=[gmsh.model.geo.addPoint(x,y,0,1.5) for x,y in xy]
    e=[gmsh.model.geo.addLine(a,b) for a,b in zip(p,p[1:]+p[:1])]
    s=gmsh.model.geo.addPlaneSurface([gmsh.model.geo.addCurveLoop(e)])
    gmsh.model.geo.synchronize()
    gmsh.model.addPhysicalGroup(2,[s],name='body')
    gmsh.model.addPhysicalGroup(1,e[:2],name='bottom')
    gmsh.model.addPhysicalGroup(1,[e[3]],name='top')
    gmsh.model.addPhysicalGroup(0,[p[1]],name='anchor')
    gmsh.model.mesh.generate(2)
    gmsh.write(str(out/'mesh.msh'))
finally:
    gmsh.finalize()
E,nu,eps,depth=1670.,.35,.03,15.
c=Case(mesh=str(out/'mesh.msh'),analysis=Analysis('elasticity','plane_strain',2,depth,
       options={'linear_solver':'scipy'}),materials={'m':Material(E=E,nu=nu)},
       bcs=[BC('displacement',group='bottom',components=[1],value=0),
            BC('displacement',group='top',components=[1],value=-eps*10),
            BC('displacement',group='anchor',components=[0],value=0)],
       qois=[QoI('Ry','reaction',group='top',component=1)])
r=solve(c)
xy,ux=r.nodal('ux')
exact_ux=nu/(1-nu)*eps*xy[:,0]
exact_uy=-eps*xy[:,1]
force=-r.evaluate_all()['Ry']
exact_force=E/(1-nu**2)*eps*8*depth
checks={'max_ux_error_mm':float(abs(ux-exact_ux).max()),
        'max_uy_error_mm':float(abs(r.fields['uy']-exact_uy).max()),
        'reaction_N':force,'exact_reaction_N':exact_force,
        'relative_reaction_error':abs(force/exact_force-1)}
assert checks['max_ux_error_mm']<1e-10
assert checks['max_uy_error_mm']<1e-10
assert checks['relative_reaction_error']<1e-10
(out/'checks.json').write_text(json.dumps(checks,indent=2)+'\n')
print(json.dumps(checks,indent=2))
