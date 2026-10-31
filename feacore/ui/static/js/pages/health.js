// System health: solver backends, hardware, versions, run store and jobs.
import { api, h, clear, loading, errorBox, statusChip } from '../api.js';

export async function render(el) {
  el.append(loading());
  let d;
  try { d = await api('/api/health?wait=1'); } catch (e) { clear(el).append(errorBox(e)); return; }
  clear(el);
  el.append(h('div', { class: 'page-head' }, h('div', {}, h('h1', {}, 'System health'), h('p', { class: 'lede' }, `${d.hostname} · Python ${d.python} · feacore ${d.feacore}`))));
  const b = d.backends || {};
  const gpus = d.gpu.gpus || [];
  el.append(h('div', { class: 'grid-2' },
    h('div', { class: 'stack' },
      h('div', { class: 'card' }, h('h2', {}, 'Solver backends'),
        h('table', { class: 'data' }, h('tbody', {}, Object.entries(b).map(([k, v]) => h('tr', {},
          h('td', { class: 'mono', style: { fontWeight: 600 } }, k), h('td', {}, statusChip(v.ok ? 'good' : 'critical', v.ok ? 'available' : 'unavailable')),
          h('td', { class: 'small secondary' }, v.ok ? (v.doc || '') : (v.error || ''), v.executable ? h('div', { class: 'mono small' }, v.executable) : null)))))),
      h('div', { class: 'card' }, h('h2', {}, 'Hardware'),
        h('table', { class: 'kv' }, h('tbody', {},
          kv('CPU', `${d.cpu.model || '?'} · ${d.cpu.count} cores${d.cpu.affinity && d.cpu.affinity !== d.cpu.count ? ` (${d.cpu.affinity} usable)` : ''}`),
          kv('load', (d.cpu.load || []).join(' / ')),
          kv('memory', d.cpu.memory_gb ? `${d.cpu.memory_gb.available} GB free of ${d.cpu.memory_gb.total} GB` : '?'),
          kv('threads', Object.entries(d.cpu.threads_env || {}).map(([k, v]) => `${k}=${v}`).join(', ') || 'library defaults'),
          kv('GPU', gpus.length ? gpus.map(g => `${g.name} (${g.memory}, driver ${g.driver})`).join('; ') : (d.gpu.nvidia_smi ? 'nvidia-smi found, no GPU listed' : 'none (no nvidia-smi)')),
          kv('GPU libraries', Object.entries(d.gpu.libs || {}).map(([k, v]) => `${k} ${v ? '✓' : '✕'}`).join(' · ')),
          kv('CalculiX', d.ccx || 'not found'))))),
    h('div', { class: 'stack' },
      h('div', { class: 'card' }, h('h2', {}, 'Jobs'),
        h('table', { class: 'kv' }, h('tbody', {},
          kv('workers', Object.entries(d.jobs.workers).map(([k, v]) => `${k}: ${v}`).join(', ')),
          kv('queued', Object.entries(d.jobs.queued).map(([k, v]) => `${k}: ${v}`).join(', ')),
          kv('running', d.jobs.running.map(j => `${j.kind} ${j.target}`).join(', ') || 'none'))),
        h('div', { class: 'small muted', style: { marginTop: '8px' } }, `run store: ${d.store.root}`)),
      h('div', { class: 'card' }, h('h2', {}, 'Problems & versions'),
        h('table', { class: 'kv' }, h('tbody', {}, kv('registered problems', d.problems.count),
          ...Object.entries(d.versions).map(([k, v]) => kv(k, v)))),
        Object.keys(d.problems.import_errors || {}).length ? h('pre', { class: 'code' }, Object.entries(d.problems.import_errors).map(([m, e]) => `${m}: ${e}`).join('\n')) : null))));
}

const kv = (k, v) => h('tr', {}, h('th', {}, k), h('td', {}, v ?? '–'));
