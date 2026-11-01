"""Derive preselected CPU-DRAM layouts; retain the fixed FE solver and input physics."""
import copy
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def cells(b):
    # HotSpot analogues are supplementary; the primary metric is FE die-centre T.
    n, extent = 64, .045
    x, y = b['x_centre_m'], b['y_centre_m']
    j = min(n-1, max(0, int(x/extent*n)))
    i = min(n-1, max(0, int(y/extent*n)))
    return [dict(weight=1., x0_m=j*extent/n, x1_m=(j+1)*extent/n,
                 y0_m=i*extent/n, y1_m=(i+1)*extent/n)]


def derive(original, layout):
    data = copy.deepcopy(original)
    target = data['cases']['B']
    old = copy.deepcopy(target['chiplets'])
    for i, chip in enumerate(target['chiplets']):
        x, y = [v*.001 for v in layout['centers_mm'][i]]
        w, h = old[i]['width_m'], old[i]['height_m']
        if layout['rotated'][i] != old[i]['rotated_vs_tap_cfg']:
            w, h = h, w
        chip.update(x_centre_m=x, y_centre_m=y, x0_m=x-w/2, y0_m=y-h/2,
                    width_m=w, height_m=h, rotated_vs_tap_cfg=layout['rotated'][i])
    for layer in ['L3', 'L4']:
        blocks = []
        for src in target['layers'][layer]:
            if src['name'].startswith('WS_'):
                continue
            b = copy.deepcopy(src)
            idx = (int(b['name'].split('_')[1]) if b['name'].startswith('Chiplet_') else
                   int(b['name'].split('_')[1])//4 if b['name'].startswith('Ubump_') else None)
            if idx is not None:
                a, c = old[idx], target['chiplets'][idx]
                dx, dy = b['x_centre_m']-a['x_centre_m'], b['y_centre_m']-a['y_centre_m']
                if layout['rotated'][idx] != a['rotated_vs_tap_cfg']:
                    dx, dy = -dy, dx
                    b['width_m'], b['height_m'] = b['height_m'], b['width_m']
                b.update(x_centre_m=c['x_centre_m']+dx, y_centre_m=c['y_centre_m']+dy)
                b.update(x0_m=b['x_centre_m']-b['width_m']/2,
                         y0_m=b['y_centre_m']-b['height_m']/2)
            b['hotspot_centre_cells_64'] = cells(b)
            blocks.append(b)
        xs = sorted(set(round(v, 12) for b in blocks for v in [b['x0_m'], b['x0_m']+b['width_m']]))
        ys = sorted(set(round(v, 12) for b in blocks for v in [b['y0_m'], b['y0_m']+b['height_m']]))
        solids = list(blocks)
        area = sum(b['width_m']*b['height_m'] for b in solids)
        for y0, y1 in zip(ys[:-1], ys[1:]):
            pending = None
            for x0, x1 in zip(xs[:-1], xs[1:]):
                x, y = (x0+x1)/2, (y0+y1)/2
                hit = sum(b['x0_m']-1e-13 < x < b['x0_m']+b['width_m']+1e-13 and
                          b['y0_m']-1e-13 < y < b['y0_m']+b['height_m']+1e-13 for b in solids)
                assert hit <= 1, 'overlapping blocks'
                if hit == 0:
                    if pending is None: pending = x0
                if pending is not None and (hit or x1 == xs[-1]):
                    end = x0 if hit else x1
                    b = dict(name=f'WS_{len(blocks)}', x0_m=pending, y0_m=y0,
                             width_m=end-pending, height_m=y1-y0, k_W_per_mK=1.6,
                             power_W=0., x_centre_m=(pending+end)/2, y_centre_m=y)
                    b['hotspot_centre_cells_64'] = cells(b)
                    blocks.append(b)
                    area += b['width_m']*b['height_m']
                    pending = None
        assert abs(area-.045**2) < 1e-12, area
        target['layers'][layer] = blocks
    target['rank_layout'] = layout
    data['rank_note'] = 'Only B derived. Primary metric FE die centre at mid-thickness; supplemental cell readouts use containing cell, not TAP selector.'
    return data


if __name__ == '__main__':
    source = ROOT/'experiments/sentilab_package/inputs/case.json'
    layouts = HERE/'inputs/rank-layouts.json'
    assert hashlib.sha256(layouts.read_bytes()).hexdigest() == '7106ce2a0fe87fa370643c7a4f2db197b93afeabbecbfeee2e42b6d506209ff6'
    original = json.loads(source.read_text())
    rows = json.loads(layouts.read_text())['fixtures']['micro150-tap25d-fig5a']['layouts']
    for row in rows:
        if row['id'] in ['reference', 'near-01', 'any-01']:
            path = HERE/'inputs'/f"B_{row['id']}.json"
            path.write_text(json.dumps(derive(original, row), indent=1)+'\n')
            print(path.name, hashlib.sha256(path.read_bytes()).hexdigest())
