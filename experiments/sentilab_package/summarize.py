"""Aggregate completed runs without treating a paper match as a pass criterion."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from scipy.optimize import brentq
from feacore.solver._skfem_mesh import read_mesh


def geometry_audit(path, data):
    md = read_mesh(path, 3, 1)
    mesh = md.mesh
    def area(facets):
        p = mesh.p[:, mesh.facets[:, facets]]
        return float(np.linalg.norm(np.cross((p[:, 1]-p[:, 0]).T,
                                             (p[:, 2]-p[:, 0]).T), axis=1).sum()/2)
    cooled = md.group('sink_top').index
    exterior = mesh.boundary_facets()
    if np.any(mesh.f2t[1, cooled] != -1):
        raise ValueError('The cooled surface contains an internal facet')
    sources = {}
    for name, group in md.groups.items():
        if name.startswith('source_'):
            if np.any(mesh.f2t[1, group.index] == -1):
                raise ValueError('A die source interface has an unintended external facet')
            sources[name] = area(group.index)
    got = dict(convective_m2=area(cooled), adiabatic_total_m2=area(exterior)-area(cooled),
               surface_sources_are_internal=True, source_areas_m2=sources)
    for key in ['convective_m2', 'adiabatic_total_m2']:
        if not np.isclose(got[key], data['boundary_conditions']['area_bookkeeping'][key], rtol=1e-10, atol=1e-14):
            raise ValueError(f'External area mismatch: {key}: {got[key]}')
    return got


def convergence(rows):
    rows = sorted(rows, key=lambda r: r['refinement'])
    h = np.array([1/r['refinement'] for r in rows])
    values = np.array([r['M_C'] for r in rows])
    d12, d23 = values[0]-values[1], values[1]-values[2]
    out = dict(refinements=[r['refinement'] for r in rows], values_C=values.tolist(),
               successive_changes_K=np.diff(values).tolist(),
               finest_pair_difference_K=float(abs(d23)),
               three_mesh_range_K=float(np.ptp(values)),
               note='Three-point observed order; an asymptotic regime is not independently established.')
    if abs(d23) < 1e-10 and abs(d12) < 1e-9:
        out.update(observed_order=None, extrapolated_C=float(values[-1]),
                   error_estimate_K=float(max(abs(d12), abs(d23))),
                   gci_style_uncertainty_K=float(1.25*max(abs(d12), abs(d23))))
        return out
    if d12*d23 <= 0:
        out['status'] = 'non-monotone; no Richardson/GCI estimate justified'
        return out
    def eq(p):
        return (h[0]**p-h[1]**p)/(h[1]**p-h[2]**p)-d12/d23
    try:
        p = brentq(eq, .01, 10)
    except ValueError:
        out['status'] = 'no positive observed order in (0.01, 10); no error certification'
        return out
    correction = (values[2]-values[1])/((h[1]/h[2])**p-1)
    out.update(status='monotone three-grid fit', observed_order=float(p),
               extrapolated_C=float(values[2]+correction),
               error_estimate_K=float(abs(correction)),
               gci_style_uncertainty_K=float(1.25*abs(correction)),
               accuracy_goal_met_by_estimate=bool(1.25*abs(correction) <= .1))
    if p > 4:
        out['caution'] = ('Observed order exceeds 4 for P2; cancellation or a pre-asymptotic '
                          'regime may dominate. The formal extrapolation is not an accuracy certification.')
        out['accuracy_goal_met_by_estimate'] = None
    return out


def plot_convergence(cases, destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for ax, (cid, entry) in zip(axes, cases.items()):
        rows, surface = entry['primary'], entry['surface']
        h = [1/r['refinement'] for r in rows]
        ax.plot(h, [r['M_C'] for r in rows], 'o-', color='#1764a0', label='FE volumetric source')
        ax.scatter([1/3], [surface['M_C']], marker='D', color='#c15d0e', label='FE surface source')
        ax.axhline(rows[-1]['paper_target_C'], color='#555555', linestyle='--', label='Supplied paper target')
        ax.set(xlabel='Relative mesh spacing (1 / refinement)', ylabel='Primary temperature metric (C)',
               title=f'Case {cid}: '+('Ascend 910-class' if cid == 'A' else 'CPU-DRAM'))
        ax.set_xlim(1.08, .25)
        ax.set_xticks([1, .5, 1/3], ['1', '1/2', '1/3'])
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('directory', type=Path)
    args = ap.parse_args()
    root = args.directory
    package = Path(__file__).parent
    repository = package.resolve().parents[1]
    data = json.loads((package/'inputs/case.json').read_text())
    audits = {cid: geometry_audit(root/f'{cid}_r1'/'mesh.msh', data) for cid in 'AB'}
    (root/'geometry_checks.json').write_text(json.dumps(audits, indent=2))
    allrows = {}
    for cid in 'AB':
        primary = [json.loads((root/f'{cid}_r{r}'/'result.json').read_text()) for r in [1, 2, 3]]
        surface = json.loads((root/f'{cid}_surface_r3'/'result.json').read_text())
        hashes = {row['script_sha256'] for row in primary+[surface]}
        if len(hashes) != 1 or len({row['input_sha256'] for row in primary+[surface]}) != 1:
            raise ValueError('Mixed scripts/inputs: rerun or explicitly resolve provenance')
        allrows[cid] = dict(primary=primary, surface=surface, convergence=convergence(primary))
        readouts = {f'{variant}/{plane}': max(v[f'{plane}_K'] for v in row['chip_temperatures'].values())-273.15
                    for variant, row in [('volume', primary[-1]), ('surface', surface)]
                    for plane in ['bottom', 'mid', 'top']}
        allrows[cid]['predeclared_readouts_C'] = readouts
        allrows[cid]['A1_A2_readout_bracket_C'] = [min(readouts.values()), max(readouts.values())]
    summary = dict(cases=allrows, geometry_checks=audits,
                   source_data_commit='af66bcfc21220ce3d8e92e89273979241e54b04a',
                   solver_base_commit='a5c556f7047ec7b051d30f4e1130743f29fb4fbf')
    (root/'summary.json').write_text(json.dumps(summary, indent=2))
    plot_convergence(allrows, root/'convergence.png')
    lines = ['# Independent FE comparison of the TAP-2.5D configured model', '',
        'FEAgent / SentiLab ticket 186e391e. This is a continuum simulation of the supplied model, '
        'not validation against measurements or manufacturer data. No parameter was fitted to the paper.', '',
        '## Method', '',
        'The existing FEAgent/scikit-fem heat backend solves the temperature excess above 318.15 K '
        'with P2 tetrahedra and SciPy/PARDISO as recorded per run. The complete spreader and sink are '
        'meshed. All block boundaries and layer interfaces conform; the declared case-A gap is underfill. '
        'Each original tensor interval is subdivided uniformly by factors 1, 2 and 3, giving successive '
        'refinement ratios 2 and 1.5 in every direction. Thin-layer elements are anisotropic. '
        'The coordinates and coefficients are never tuned toward a reference temperature.', '',
        'Point location uses the six tetrahedra of the containing tensor box. HotSpot-cell analogues '
        'are actual plane-area integrals: tetrahedron/plane intersection, rectangle clipping and '
        'degree-2-exact triangle quadrature. Independent polynomial checks verify both procedures. '
        'The only cooled face is the sink top. Input power is integrated from the assembled source '
        'vector; output heat is the integrated Robin flux.', '',
        '![Convergence and source variants](convergence.png)', '',
        '## Mesh convergence', '',
        '| Case | Refinement | Tetrahedra | DOFs | Primary M (C) | Relative balance error | Relative residual | Time (s) | Peak RSS (MB) |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for cid, entry in allrows.items():
        for r in entry['primary']:
            lines.append(f"| {cid} | {r['refinement']} | {r['mesh']['elements']} | {r['dofs']} | "
                         f"{r['M_C']:.8f} | {r['energy_relative_error']:.3g} | {r['residual_rel']:.3g} | "
                         f"{r['time_s']:.1f} | {r['peak_memory_MB']:.0f} |")
    for cid, entry in allrows.items():
        r, s, c = entry['primary'][-1], entry['surface'], entry['convergence']
        lines += ['', f'## Case {cid}', '',
            f"Finest primary M = **{r['M_C']:.8f} C** ({r['hottest_die']}); paper = "
            f"{r['paper_target_C']:.2f} C; FE minus paper = **{r['difference_from_paper_K']:+.8f} K**.", '',
            f"Three-grid convergence: `{json.dumps(c)}`", '',
            f"Finest surface-source variant M = {s['M_C']:.8f} C; change from volumetric source "
            f"= {s['M_C']-r['M_C']:+.8f} K. The selector analogue over all supplied L0-L5 blocks "
            f"is {r['selector_analogue_C']:.8f} C at {r['selector_block']} (primary).", '',
            f"Predeclared A1/A2 readout bracket (maximum over dies at each bottom/mid/top plane, "
            f"both source variants): {entry['A1_A2_readout_bracket_C'][0]:.8f} to "
            f"{entry['A1_A2_readout_bracket_C'][1]:.8f} C. All six values are retained in summary.json.", '',
            f"Input power {r['P_in_W']:.10f} W, output heat {r['Q_out_W']:.10f} W. "
            f"Mean sink-top temperature excess {r['mean_sink_top_excess_K']:.10f} K.", '',
            '| Die | Primary bottom (C) | Primary mid (C) | Primary top (C) | Surface-source mid (C) |',
            '|---|---:|---:|---:|---:|']
        for name, v in r['chip_temperatures'].items():
            lines.append(f"| {name} | {v['bottom_K']-273.15:.8f} | {v['mid_K']-273.15:.8f} | "
                         f"{v['top_K']-273.15:.8f} | {s['chip_temperatures'][name]['mid_K']-273.15:.8f} |")
        lines += ['', f"Nodal field extrema: {r['field_min_K']-273.15:.8f} to "
            f"{r['field_max_K']-273.15:.8f} C. Locations, layer extrema, all block-cell averages, "
            'Kelvin readouts, software versions, memory and solver warnings are in the per-run JSON and summary.json.']
    lines += ['', '## Interpretation and limitations', '',
        'Differences from the paper are reported without a pass/fail judgement. The supplied document '
        'identifies differences between HotSpot 6 and continuum FE: vertical node locations/resistances, '
        'material mixing in 64x64 cells, and lumped peripheral copper nodes. The primary, bottom/top '
        'readouts, and surface-source variant were specified before solving and are all retained. '
        'The source layout was decoded from a paper figure, rather than supplied by its authors.', '',
        'Richardson/GCI-style values are estimates from three meshes, not guaranteed bounds. '
        'If no positive monotone fit exists, no error certification is claimed. Nodal field extrema '
        'do not search stationary points of P2 polynomials between nodes. Solve timings are single runs, '
        'not performance benchmarks. Peak memory is per-process RSS, not total machine allocation.', '',
        '## Reproduce', '',
        'From the checkout with the documented FEAgent dependencies installed:', '',
        '```sh', 'bash experiments/sentilab_package/reproduce.sh', '```', '',
        'The input authorship and original CC BY 4.0 licence are retained in experiments/sentilab_package/inputs. '
        'See README.md there for the complete parameter sheet and source citations.', '',
        'Original collaboration fixture code and numerical outputs are available under the scoped '
        '[MIT grant](../../COLLABORATION-LICENSE.txt). This does not license the underlying FEAgent solver '
        'or dependencies. The compact evidence bundle omits generated .msh/.npz fields; their hashes '
        'are retained separately in FIELDS_SHA256SUMS and the fields can be regenerated.', '',
        '## Independence', '',
        'Published target values were supplied in case.json and known in advance. The FEAgent operator '
        'did not read the SentiLab team\'s newly posted solver values before freezing these runs. '
        'The notification that their independent solve remained wanted was read, but no peer-solver '
        'numbers were used in model construction, mesh selection or parameter settings.']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')
    manifest, fields = [], []
    for path in sorted(list(package.rglob('*'))+list(root.rglob('*'))+[repository/'COLLABORATION-LICENSE.txt']):
        if path.is_file() and not path.name.startswith('._') and path.resolve() not in [(root/n).resolve() for n in ['SHA256SUMS', 'FIELDS_SHA256SUMS']] and '__pycache__' not in str(path):
            relative = path.resolve().relative_to(repository)
            entry = f'{hashlib.sha256(path.read_bytes()).hexdigest()}  {relative}'
            (fields if path.suffix in ['.msh', '.npz'] else manifest).append(entry)
    (root/'SHA256SUMS').write_text('\n'.join(manifest)+'\n')
    (root/'FIELDS_SHA256SUMS').write_text('\n'.join(fields)+'\n')
    print(json.dumps({cid: e['convergence'] for cid, e in allrows.items()}, indent=2))


if __name__ == '__main__':
    main()
