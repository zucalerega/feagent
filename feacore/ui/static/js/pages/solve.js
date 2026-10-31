// Solve: upload a case JSON + gmsh mesh, run the fixed solver (or a cross-check backend), see results.
import { api, authFetch, withToken, h, clear, fmt, fmtInt, fmtTime, loading, errorBox, runStatusChip, timeAgo } from '../api.js';

export async function render(el) {
  el.append(loading());
  let probs, health, recent;
  try {
    [probs, health, recent] = await Promise.all([api('/api/problems').catch(() => ({ problems: [] })), api('/api/health').catch(() => null), api('/api/runs?limit=200')]);
  } catch (e) { clear(el).append(errorBox(e)); return; }
  clear(el);
  el.append(h('div', { class: 'page-head' }, h('div', {}, h('h1', {}, 'Solve a case'),
    h('p', { class: 'lede' }, 'The solver service: upload a case JSON (analysis, materials, boundary conditions, loads, QoIs) and a gmsh .msh with named physical groups. This is the same endpoint the agents’ MCP / mesh harness calls.'))));
  const backends = (health && health.backends) || {};
  const caseFile = h('input', { type: 'file', accept: '.json,application/json', name: 'case' });
  const caseText = h('textarea', { rows: 12, placeholder: '{ "mesh": "part.msh", "analysis": {...}, "materials": {...}, "bcs": [...], "qois": [...] }  — or choose a file above', style: { width: '100%' } });
  const meshFile = h('input', { type: 'file', accept: '.msh', name: 'mesh' });
  const backendSel = h('select', {}, ['skfem', 'ccx', 'nn'].map(b => h('option', { value: b, disabled: backends[b] && !backends[b].ok ? true : null }, `${b}${b === 'skfem' ? ' (fixed solver)' : b === 'ccx' ? ' (CalculiX cross-check)' : ' (neural surrogate)'}${backends[b] && !backends[b].ok ? ' — unavailable' : ''}`)));
  const agent = h('input', { type: 'text', placeholder: 'optional, e.g. agent-07' });
  const problemSel = h('select', {}, h('option', { value: '' }, '(none)'), probs.problems.map(p => h('option', { value: p.id }, p.id)));
  const fields = h('input', { type: 'checkbox', checked: true });
  const out = h('div');
  const btn = h('button', { class: 'btn primary', type: 'button', onclick: () => submit() }, 'Run solve');
  caseFile.addEventListener('change', async () => { if (caseFile.files[0]) caseText.value = await caseFile.files[0].text(); });

  async function submit() {
    clear(out);
    if (!caseText.value.trim()) { out.append(errorBox('Provide a case JSON (file or text).')); return; }
    if (!meshFile.files[0]) { out.append(errorBox('Choose a .msh mesh file.')); return; }
    const fd = new FormData();
    fd.append('case', caseText.value);
    fd.append('mesh', meshFile.files[0], meshFile.files[0].name);
    fd.append('backend', backendSel.value);
    if (agent.value.trim()) fd.append('agent_id', agent.value.trim());
    if (problemSel.value) fd.append('problem_id', problemSel.value);
    fd.append('fields', fields.checked ? 'true' : 'false');
    btn.disabled = true;
    out.append(h('div', { class: 'card' }, h('span', { class: 'spinner' }), ' Solving in a worker process…'));
    try {
      const res = await authFetch('/api/solve', { method: 'POST', body: fd });
      const body = await res.json();
      clear(out);
      if (res.status === 422 && body.errors) {
        out.append(h('div', { class: 'card' }, h('h2', {}, 'Invalid case'), errorBox(body.errors.join('\n'))));
      } else if (body.run_id) {
        out.append(result(body));
        if (body.status === 'queued' || body.status === 'running') poll(body.run_id);
      } else {
        out.append(errorBox(body.detail ? (typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail)) : `HTTP ${res.status}`));
      }
    } catch (e) { clear(out).append(errorBox(e)); }
    btn.disabled = false;
  }
  async function poll(rid) {
    const r = await api(`/api/runs/${rid}`);
    clear(out).append(result(r));
    if (r.status === 'queued' || r.status === 'running') setTimeout(() => poll(rid), 1500);
  }

  el.append(h('div', { class: 'grid-2' },
    h('div', { class: 'card stack' },
      h('label', { class: 'field' }, 'Case JSON', caseFile, caseText),
      h('label', { class: 'field' }, h('span', {}, 'Mesh ', h('span', { class: 'hint' }, '(gmsh .msh, v2.2 or v4.1, named physical groups)')), meshFile),
      h('div', { class: 'form-grid' },
        h('label', { class: 'field' }, 'Backend', backendSel),
        h('label', { class: 'field' }, 'Agent id', agent),
        h('label', { class: 'field' }, 'Problem', problemSel)),
      h('label', { class: 'check' }, fields, 'Store fields for the viewer'),
      h('div', {}, btn)),
    h('div', { class: 'stack' }, out, curlCard())));

  const apiRuns = recent.runs.filter(r => r.source === 'api' || r.source === 'import' || r.source === 'python');
  el.append(h('div', { class: 'card', style: { marginTop: '20px' } }, h('h2', {}, 'Recent solves (API, harness, imports)'),
    apiRuns.length ? h('table', { class: 'data' }, h('thead', {}, h('tr', {}, h('th', {}, 'run'), h('th', {}, 'agent'), h('th', {}, 'problem'), h('th', {}, 'backend'), h('th', { class: 'num' }, 'dofs'), h('th', {}, 'status'), h('th', {}, 'QoIs'))),
      h('tbody', {}, apiRuns.slice(0, 40).map(r => h('tr', { class: 'click', onclick: () => { location.hash = r.fields ? `#/viewer/${r.run_id}` : location.hash; } },
        h('td', {}, h('div', { class: 'mono small' }, r.run_id), h('div', { class: 'small muted' }, timeAgo(r.created_at))), h('td', {}, r.agent_id || '–'), h('td', { class: 'mono small' }, r.problem_id || '–'),
        h('td', {}, r.backend), h('td', { class: 'num' }, fmtInt(r.info && r.info.n_dofs)), h('td', {}, runStatusChip(r.status)),
        h('td', { class: 'small' }, Object.entries(r.qois || {}).slice(0, 3).map(([k, v]) => `${k} = ${fmt(v, 5)}`).join('; '))))))
      : h('p', { class: 'muted' }, 'No API solves yet.')));
}

function result(r) {
  const i = r.info || {};
  const q = Object.entries(r.qois || {});
  return h('div', { class: 'card' },
    h('div', { class: 'card-head' }, h('h2', {}, 'Result'), runStatusChip(r.status)),
    h('div', { class: 'mono small muted' }, r.run_id),
    r.error ? h('div', { style: { marginTop: '10px' } }, errorBox(r.error), r.traceback ? h('details', { class: 'more' }, h('summary', {}, 'Traceback'), h('pre', { class: 'code' }, r.traceback)) : null) : null,
    q.length ? h('table', { class: 'data', style: { marginTop: '12px' } }, h('thead', {}, h('tr', {}, h('th', {}, 'QoI'), h('th', { class: 'num' }, 'value'))),
      h('tbody', {}, q.map(([k, v]) => h('tr', {}, h('td', { class: 'mono' }, k), h('td', { class: 'num' }, fmt(v, 8)))))) : null,
    h('div', { class: 'row', style: { marginTop: '18px', gap: '10px' } },
      r.fields ? h('a', { class: 'btn primary', href: `#/viewer/${r.run_id}` }, 'Open in field viewer') : null),
    r.info || (i.warnings || []).length || (r.warnings || []).length ? h('details', { class: 'more' }, h('summary', {}, 'Solver details'),
      r.info ? h('table', { class: 'kv', style: { marginTop: '10px' } }, h('tbody', {},
        [['backend', i.backend], ['analysis', `${i.physics} · ${i.formulation} · P${i.order}`], ['mesh', `${i.cell_type} · ${fmtInt(i.n_cells)} cells · ${fmtInt(i.n_nodes)} nodes`],
         ['unknowns', fmtInt(i.n_dofs)], ['h (measured)', fmt(i.h_rep, 4)], ['solve time', fmtTime(i.wall_time)]].map(([k, v]) => h('tr', {}, h('th', {}, k), h('td', {}, v))))) : null,
      [...(i.warnings || []), ...(r.warnings || [])].map(w => h('div', { class: 'small secondary', style: { marginTop: '6px' } }, w)),
      h('div', { class: 'row', style: { marginTop: '10px', gap: '6px' } },
        h('a', { class: 'btn ghost small', href: withToken(`/api/runs/${r.run_id}`), target: '_blank' }, 'Run record (JSON)'),
        h('a', { class: 'btn ghost small', href: withToken(`/api/runs/${r.run_id}/log`), target: '_blank' }, 'Worker log'))) : null);
}

function curlCard() {
  return h('div', { class: 'card' }, h('h3', {}, 'Calling this from a script or an agent?'),
    h('p', { class: 'secondary', style: { margin: '0 0 14px' } }, 'The same solve is one HTTP call: POST the case JSON and the mesh to /api/solve. Copy-paste examples for curl and Python, and how to submit an agent\u2019s runs, are on the Connect page.'),
    h('a', { class: 'btn', href: '#/connect' }, 'Connect a script'));
}
