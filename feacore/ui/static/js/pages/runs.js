// Runs: one place for everything that has been computed: convergence studies, benchmark batches,
// individual solves (field viewer) and multi-agent consensus views.
import { api, h, clear, fmtInt, fmtTime, loading, errorBox, runStatusChip, timeAgo, qs } from '../api.js';

const TABS = [['studies', 'Studies'], ['batches', 'Benchmark batches'], ['solves', 'Solves'], ['consensus', 'Consensus']];

export async function render(el) {
  const tab = TABS.some(([k]) => k === qs().get('tab')) ? qs().get('tab') : 'studies';
  const body = h('div');
  el.append(h('div', { class: 'page-head' },
    h('div', {}, h('h1', {}, 'Runs'), h('p', { class: 'lede' }, 'Convergence studies, benchmark batches, single solves and the consensus between agents.')),
    h('div', { class: 'row', style: { gap: '10px' } },
      h('a', { class: 'btn', href: '#/study' }, 'New study'),
      h('a', { class: 'btn primary', href: '#/batch' }, 'Run benchmarks'))),
    h('div', { class: 'toolbar' }, h('div', { class: 'seg' }, TABS.map(([k, label]) =>
      h('button', { class: k === tab ? 'on' : '', onclick: () => { location.hash = `#/runs?tab=${k}`; } }, label)))),
    body);
  body.append(loading());
  try {
    await ({ studies, batches, solves, consensus })[tab](body);
  } catch (e) { clear(body).append(errorBox(e)); }
}

const empty = (title, text, action) => h('div', { class: 'card empty-state' }, h('h3', {}, title), h('p', { class: 'muted' }, text), action || null);
const verdictDots = v => h('span', { class: 'dots', title: Object.entries(v || {}).map(([q, x]) => `${q}: ${x.label}`).join('\n') },
  Object.values(v || {}).map(x => h('span', { class: `dot ${x.status}` })));

async function studies(el) {
  const d = await api('/api/studies?limit=100');
  clear(el);
  if (!d.studies.length) { el.append(empty('No studies yet', 'A convergence study runs a problem’s reference setup on a sequence of meshes.', h('a', { class: 'btn primary', href: '#/study' }, 'New study'))); return; }
  el.append(h('div', { class: 'card flush' }, h('table', { class: 'data' },
    h('thead', {}, h('tr', {}, h('th', {}, 'problem'), h('th', {}, 'settings'), h('th', {}, 'status'), h('th', {}, 'QoI verdicts'), h('th', { class: 'num' }, 'runtime'), h('th', {}, 'started'))),
    h('tbody', {}, d.studies.map(s => h('tr', { class: 'click', onclick: () => { location.hash = `#/study/${s.study_id}`; } },
      h('td', {}, h('div', { class: 'strong' }, s.problem_id), h('div', { class: 'mono small muted' }, s.study_id)),
      h('td', { class: 'small' }, `${s.backend} · P${s.order} · ${(s.sizes || []).length} meshes${s.nested ? ' · nested' : ''}${s.batch_id ? ' · batch' : ''}`),
      h('td', {}, runStatusChip(s.status)),
      h('td', {}, verdictDots(s.verdicts)),
      h('td', { class: 'num' }, fmtTime(s.elapsed)),
      h('td', { class: 'small muted' }, timeAgo(s.created_at))))))));
}

async function batches(el) {
  const d = await api('/api/batches?limit=50');
  clear(el);
  if (!d.batches.length) { el.append(empty('No benchmark batches yet', 'Pick several benchmarks and run a convergence study on each, with a results table you can download.', h('a', { class: 'btn primary', href: '#/batch' }, 'Run benchmarks'))); return; }
  const OUT = { pass: 'good', fail: 'critical', divergence_detected: 'good', divergence_missed: 'critical', no_reference: 'info' };
  el.append(h('div', { class: 'card flush' }, h('table', { class: 'data' },
    h('thead', {}, h('tr', {}, h('th', {}, 'batch'), h('th', {}, 'settings'), h('th', { class: 'num' }, 'problems'), h('th', {}, 'status'), h('th', {}, 'QoI outcomes'), h('th', {}, 'started'))),
    h('tbody', {}, d.batches.map(b => h('tr', { class: 'click', onclick: () => { location.hash = `#/batch/${b.batch_id}`; } },
      h('td', {}, h('div', { class: 'strong' }, b.name || `${b.n_problems} benchmarks`), h('div', { class: 'mono small muted' }, b.batch_id)),
      h('td', { class: 'small' }, `${b.settings.backend} · P${b.settings.order}${b.settings.nested ? ' · nested' : ''}`),
      h('td', { class: 'num' }, `${b.n_done} / ${b.n_problems}`),
      h('td', {}, runStatusChip(b.status === 'finished' ? 'succeeded' : b.status)),
      h('td', { class: 'small' }, Object.entries(b.outcomes || {}).map(([k, n]) => h('span', { class: 'nowrap', style: { marginRight: '12px' } }, h('span', { class: `dot ${OUT[k] || 'info'}` }), ` ${n} ${k.replace(/_/g, ' ')}`))),
      h('td', { class: 'small muted' }, timeAgo(b.created_at))))))));
}

async function solves(el) {
  const d = await api('/api/runs?limit=300');
  clear(el);
  if (!d.runs.length) { el.append(empty('No solves yet', 'Studies, uploads on the Solve page, the agents’ API calls and the ACE swarm all store their solves here.', h('a', { class: 'btn primary', href: '#/solve' }, 'Solve a case'))); return; }
  const filter = h('input', { type: 'search', placeholder: 'Filter by problem, agent or run id…', style: { width: '360px' } });
  const tbody = h('tbody');
  const draw = () => {
    const q = filter.value.trim().toLowerCase();
    clear(tbody);
    d.runs.filter(r => !q || `${r.run_id} ${r.problem_id || ''} ${r.agent_id || ''} ${r.backend}`.toLowerCase().includes(q)).forEach(r => {
      const i = r.info || {};
      tbody.append(h('tr', { class: r.fields ? 'click' : 'dim', onclick: () => { if (r.fields) location.hash = `#/viewer/${r.run_id}`; } },
        h('td', {}, h('div', { class: 'strong' }, r.problem_id || '–'), h('div', { class: 'mono small muted' }, r.run_id)),
        h('td', { class: 'small' }, r.agent_id || (r.study_id ? `study mesh ${(r.level ?? 0) + 1}` : r.source)),
        h('td', { class: 'small' }, `${r.backend}${i.order ? ` · P${i.order}` : ''}`),
        h('td', { class: 'num' }, fmtInt(i.n_dofs)), h('td', { class: 'num' }, fmtTime(i.wall_time)),
        h('td', {}, runStatusChip(r.status)),
        h('td', { class: 'small' }, r.fields ? 'open →' : h('span', { class: 'muted' }, 'no fields'))));
    });
  };
  filter.addEventListener('input', draw);
  el.append(h('div', { class: 'toolbar' }, filter, h('span', { class: 'muted small' }, `${d.runs.length} most recent solves; click one to open the field viewer`)),
    h('div', { class: 'card flush' }, h('table', { class: 'data' },
      h('thead', {}, h('tr', {}, h('th', {}, 'problem / run'), h('th', {}, 'agent / origin'), h('th', {}, 'backend'), h('th', { class: 'num' }, 'DOFs'), h('th', { class: 'num' }, 'solve time'), h('th', {}, 'status'), h('th', {}, ''))), tbody)));
  draw();
}

async function consensus(el) {
  const d = await api('/api/problems');
  clear(el);
  const list = d.problems.filter(p => p.n_submissions > 0).sort((a, b) => b.n_submissions - a.n_submissions);
  if (!list.length) { el.append(empty('No agent submissions yet', 'A submission is one agent’s solves on three or more of its own meshes. The consensus view compares all agents on a problem. See Connect for how agents submit.', h('a', { class: 'btn', href: '#/connect' }, 'How agents submit'))); return; }
  el.append(h('div', { class: 'card flush' }, h('table', { class: 'data' },
    h('thead', {}, h('tr', {}, h('th', {}, 'problem'), h('th', { class: 'num' }, 'submissions'), h('th', {}, ''))),
    h('tbody', {}, list.map(p => h('tr', { class: 'click', onclick: () => { location.hash = `#/consensus/${encodeURIComponent(p.id)}`; } },
      h('td', {}, h('div', { class: 'strong' }, p.title), h('div', { class: 'mono small muted' }, p.id)),
      h('td', { class: 'num' }, p.n_submissions), h('td', { class: 'small' }, 'consensus →')))))));
}
