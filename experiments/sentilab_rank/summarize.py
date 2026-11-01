"""Summarize every attempted run, without silently discarding failures."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'evaluation_runs/sentilab_rank'
attempts = {}
for line in (OUT/'status.txt').read_text().splitlines():
    name, code = line.split()
    attempts[name] = int(code)
rows = []
for name, exit_code in attempts.items():
    path = OUT/name/'result.json'
    if exit_code or not path.exists():
        rows.append(dict(run=name, status='failed', exit_code=exit_code))
        continue
    result = json.loads(path.read_text())
    layout_id = name.rsplit('_r',1)[0]
    src = (ROOT/'experiments/sentilab_package/inputs/case.json' if layout_id=='raw' else
           ROOT/f'experiments/sentilab_rank/inputs/B_{layout_id}.json')
    assert hashlib.sha256(src.read_bytes()).hexdigest() == result['input_sha256']
    data = json.loads(src.read_text())
    z = np.load(OUT/name/'temperature.npz')
    p, T = z['coords'], z['excess_K']
    if p.shape[1] != 3: p = p.T
    mask = np.zeros(len(T),dtype=bool)
    for c in data['cases']['B']['chiplets']:
        mask |= ((p[:,0]>=c['x0_m']-1e-12)&(p[:,0]<=c['x0_m']+c['width_m']+1e-12)&
                 (p[:,1]>=c['y0_m']-1e-12)&(p[:,1]<=c['y0_m']+c['height_m']+1e-12)&
                 (p[:,2]>=data['metric']['z_die_bottom_m']-1e-12)&
                 (p[:,2]<=data['metric']['z_die_top_m']+1e-12))
    assert mask.any()
    peak = float(T[mask].max()+data['boundary_conditions']['convective_face']['T_ref_K']-273.15)
    rows.append(dict(run=name, fixture='micro150-tap25d-fig5a', layout=layout_id,
                     version='fe', status='ok', refinement=result['refinement'],
                     M_C=result['M_C'], peak_C=peak, selector_C=None,
                     path=f"FEAgent conforming P2 tetrahedra r{result['refinement']}",
                     dofs=result['dofs'], input_sha256=result['input_sha256'],
                     result_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                     energy_relative_error=result['energy_relative_error'],
                     residual_rel=result['residual_rel'], time_s=result['time_s'],
                     peak_memory_MB=result['peak_memory_MB']))
(OUT/'rows.json').write_text(json.dumps(rows,indent=2)+'\n')
print(json.dumps(rows,indent=2))
