// Swarm: live view of the ACE agent loop. Each generation's agents per problem (score, QoI vs
// reference, mesh, solves, gates) and the score distribution across generations.
import { api, h, clear, fmt, fmtInt, loading, errorBox, timeAgo, qs } from '../api.js';
import { stripChart } from '../plot.js';

let sel = { loop: null, gen: null };

export async function render(el) {
  const want = qs();
  if (want.get('loop')) sel.loop = want.get('loop');
  if (want.get('gen') !== null) sel.gen = want.get('gen');
  el.append(loading('Loading the swarm…'));
  let alive = true, timer = null, first = true;
  const body = h('div');
  const tick = async () => {
    let d;
    const p = new URLSearchParams();
    if (sel.loop) p.set('loop_id', sel.loop);
    if (sel.gen !== null && sel.gen !== undefined) p.set('generation', sel.gen);
    try { d = await api(`/api/swarm?${p}`); } catch (e) { clear(el).append(errorBox(e)); return; }
    if (!alive) return;
    if (first) { clear(el).append(body); first = false; }
    draw(body, d, () => { clearTimeout(timer); tick(); });
    timer = setTimeout(tick, d.active ? 4000 : 15000);
  };
  await tick();
  return () => { alive = false; clearTimeout(timer); };
}

// Two score schemes: 'speed_accuracy' (0-1: valid x accuracy x efficiency, records with score_parts)
// and the older 'log' score (log10(e_uniform / e_agent), gates at -2).
const SCHEME = {
  speed_accuracy: {
    lede: 'ACE agents set up and mesh the benchmark problems. Each finished agent run gets a speed / accuracy score from 0 to 1: valid × accuracy × efficiency, where 1 is the cheapest run within tolerance on that problem.',
    caption: 'One dot per scored agent run; the bar is the median. 1 = the cheapest valid run within tolerance; invalid runs score 0 (hollow dots). Scores are relative to every run on the problem so far, so they can change as the loop runs.',
    yTitle: 'score (0–1)', domain: [0, 1], signed: false, fmt: x => x.toFixed(2),
  },
  log: {
    lede: 'ACE agents set up and mesh the benchmark problems; each finished agent run is scored against the best uniform mesh with the same node budget.',
    caption: 'One dot per finished agent run; the bar is the median. +1 = ten times more accurate than a uniform mesh with the same number of nodes; a failed gate scores −2 (hollow dots).',
    yTitle: 'score (0 = uniform mesh, same nodes)', domain: null, signed: true, fmt: x => `${x > 0 ? '+' : ''}${x.toFixed(2)}`,
  },
};
const scheme = d => SCHEME[d.score_scheme] || SCHEME.log;

function draw(el, d, refresh) {
  clear(el);
  const sc = scheme(d);
  const head = h('div', { class: 'page-head' },
    h('div', {}, h('h1', {}, 'Agent swarm'), h('p', { class: 'lede' }, sc.lede)),
    d.loops.length ? h('div', { class: 'row', style: { alignItems: 'center', gap: '14px' } },
      live(d),
      h('select', { onchange: e => { sel.loop = e.target.value; sel.gen = null; refresh(); } },
        d.loops.map(l => h('option', { value: l.loop_id, selected: l.loop_id === d.loop_id }, `loop ${l.loop_id}`)))) : null);
  el.append(head);
  if (!d.loops.length) { el.append(emptyState(d)); return; }

  const loop = d.loops.find(l => l.loop_id === d.loop_id);
  const g = d.generations.find(x => x.generation === d.generation) || {};
  const F = x => (x === null || x === undefined || !isFinite(x)) ? '–' : sc.fmt(x);
  el.append(h('div', { class: 'toolbar' },
    h('span', { class: 'section-title' }, 'Generation'),
    h('div', { class: 'seg' }, loop.generations.map(x => h('button', { class: x === d.generation ? 'on' : '', onclick: () => { sel.loop = d.loop_id; sel.gen = x; refresh(); } }, String(x)))),
    h('span', { class: 'muted small' }, `${loop.problems.length} problem${loop.problems.length === 1 ? '' : 's'} · ${loop.n_submissions} finished agent runs · ${loop.n_runs} solves`)));

  const chart = stripChart({
    groups: d.generations.map(x => ({
      label: `gen ${x.generation}`, sub: `${x.n_submissions} run${x.n_submissions === 1 ? '' : 's'}${x.n_pending ? `, ${x.n_pending} pending` : ''}`, median: x.median,
      points: x.points.filter(p => !p.pending).map(p => ({ y: p.score, muted: p.gated || p.valid === false,
        tip: `<b>${p.agent || p.ace_run_id}</b> · ${p.problem_id}<br>score ${F(p.score)}${p.gated ? '<br><span class="t-muted">gated / invalid</span>' : ''}` })),
    })),
    yTitle: sc.yTitle, width: 600, height: 280, yDomain: sc.domain, signed: sc.signed,
  });
  const newStyle = d.score_scheme === 'speed_accuracy';
  el.append(h('div', { class: 'grid-2', style: { marginBottom: '28px' } },
    h('div', { class: 'card' }, h('h2', {}, 'Score by generation'),
      h('p', { class: 'muted small', style: { marginTop: '-6px' } }, sc.caption), chart),
    h('div', { class: 'card' }, h('h2', {}, `Generation ${d.generation}`),
      h('div', { class: 'stat-list' },
        stat('finished agent runs', g.n_submissions ?? 0),
        newStyle && g.n_pending ? stat('waiting for scores', g.n_pending) : null,
        stat('still running', g.n_running ?? 0),
        stat('median score', F(g.median)),
        stat('best score', F(g.best)),
        newStyle ? stat('within tolerance', g.n_within_tolerance ?? 0) : null,
        stat(newStyle ? 'invalid (score 0)' : 'gated (score −2)', newStyle ? (g.n_invalid ?? 0) : (g.n_gated ?? 0)),
        g.n_crashed ? stat('crashed (not scored)', g.n_crashed) : null),
      d.active ? h('p', { class: 'small muted' }, 'The loop is running: this page refreshes every few seconds.') : null)));

  if ((d.agents || []).length) el.append(agentsCard(d, F));
  if (!d.problems.length) el.append(h('div', { class: 'card empty-state' }, h('h3', {}, 'No agent runs in this generation yet')));
  for (const p of d.problems) el.append(board(p, d));
}

function agentsCard(d, F) {
  const gens = d.loops.find(l => l.loop_id === d.loop_id).generations;
  return h('div', { class: 'card', style: { marginBottom: '24px' } },
    h('h2', {}, 'Agents across generations'),
    h('p', { class: 'muted small', style: { marginTop: '-6px' } }, 'Each persistent agent’s mean score per generation (over the problems it ran), so you can follow one agent as its playbook changes.'),
    h('div', { style: { overflowX: 'auto' } }, h('table', { class: 'data' },
      h('thead', {}, h('tr', {}, h('th', {}, 'agent'), gens.map(g => h('th', { class: 'num' }, `gen ${g}`)))),
      h('tbody', {}, d.agents.map(a => h('tr', {}, h('td', { class: 'mono strong' }, a.agent),
        gens.map(g => {
          const c = a.generations[String(g)];
          if (!c) return h('td', { class: 'num muted' }, '');
          return h('td', { class: `num${g === d.generation ? ' strong' : ''}`, title: `${c.n} run${c.n === 1 ? '' : 's'}${c.n_pending ? `, ${c.n_pending} pending` : ''}; best ${F(c.best)}` },
            c.mean === null ? h('span', { class: 'muted' }, 'pending') : F(c.mean));
        })))))));
}

function live(d) {
  return h('span', { class: `live${d.active ? ' on' : ''}`, title: d.active ? 'new records in the last few minutes' : 'no new records recently' },
    h('span', { class: 'live-dot' }), d.active ? 'live' : 'idle');
}

const stat = (label, value) => h('div', { class: 'stat-line' }, h('span', { class: 'label' }, label), h('span', { class: 'value' }, value));

function partsTip(sp) {
  if (!sp) return '';
  const pct = x => (x === null || x === undefined) ? '–' : `${fmt(100 * x, 2)}%`;
  const lines = [];
  if (sp.accuracy !== undefined) lines.push(`accuracy ${fmt(sp.accuracy, 3)}`);
  if (sp.efficiency !== undefined) lines.push(`efficiency ${fmt(sp.efficiency, 3)}`);
  if (sp.valid !== undefined) lines.push(`valid: ${sp.valid ? 'yes' : 'no'}`);
  if (sp.rel_error !== undefined) lines.push(`rel. error ${pct(sp.rel_error)} vs tolerance ${pct(sp.tolerance)}${sp.meets_tol === true ? ' (within)' : sp.meets_tol === false ? ' (outside)' : ''}`);
  if (sp.dofs !== undefined) lines.push(`${fmtInt(sp.dofs)} DOFs`);
  if (sp.pareto_rank !== undefined) lines.push(`Pareto rank ${sp.pareto_rank}${sp.population ? ` of ${sp.population} runs` : ''}`);
  return lines.join('\n');
}

function scoreCell(r, d) {
  if (r.status === 'running') return h('span', { class: 'muted small' }, 'running…');
  if (r.status === 'crashed') return h('span', { class: 'gate', title: r.error || 'crashed' }, h('span', { class: 'dot critical' }), 'crashed');
  if (r.status === 'pending') return h('span', { class: 'muted', title: 'finished; the generation is not scored yet' }, 'pending');
  const score = r.score, gated = r.gates.length > 0;
  if (score === null || score === undefined) return h('span', { class: 'muted' }, 'not scored');
  const w = 90;
  let bar, label;
  if (d.score_scheme === 'speed_accuracy') {    // 0-1 bar
    const len = Math.max(0, Math.min(1, score)) * (w - 2);
    bar = `<svg width="${w}" height="12" style="vertical-align:middle"><rect x="0" y="3" width="${w}" height="6" rx="3" fill="#efeee9"/>` +
      `<rect x="0" y="3" width="${len}" height="6" rx="3" fill="${gated ? '#b9b8b2' : '#2a78d6'}"/></svg>`;
    label = score.toFixed(2);
  } else {                                       // older log score, -2..+2 diverging bar
    const c = w / 2, sc = Math.max(-2, Math.min(2, score)) / 2 * (c - 2);
    bar = `<svg width="${w}" height="12" style="vertical-align:middle"><line x1="${c}" x2="${c}" y1="0" y2="12" stroke="#c3c2b7"/>` +
      `<rect x="${sc >= 0 ? c : c + sc}" y="3" width="${Math.abs(sc)}" height="6" rx="2" fill="${score >= 0 && !gated ? '#2a78d6' : '#b9b8b2'}"/></svg>`;
    label = `${score > 0 ? '+' : ''}${score.toFixed(2)}`;
  }
  return h('span', { class: 'nowrap', title: partsTip(r.score_parts) }, h('span', { html: bar }), ' ', h('b', { class: 'num' }, label),
    r.score_parts ? h('div', { class: 'small muted' }, `acc ${fmt(r.score_parts.accuracy, 2)} · eff ${fmt(r.score_parts.efficiency, 2)}`) : null);
}

function board(p, d) {
  const ref = p.reference;
  const rows = p.rows;
  const anyDiag = rows.some(r => r.diagnosis);
  return h('div', { class: 'card', style: { marginBottom: '24px' } },
    h('div', { class: 'card-head' },
      h('div', {}, h('h2', { style: { marginBottom: '2px' } }, p.title),
        h('div', { class: 'meta-line' }, p.problem_id, p.qoi ? ` · QoI ${p.qoi}` : '', ref ? ` · reference ${fmt(ref.value, 5)}` : ' · no reference value')),
      h('div', { class: 'row', style: { gap: '10px' } },
        h('a', { class: 'btn ghost', href: `#/benchmarks/${encodeURIComponent(p.problem_id)}` }, 'Problem'),
        h('a', { class: 'btn', href: `#/consensus/${encodeURIComponent(p.problem_id)}` }, 'Consensus view'))),
    h('div', { style: { overflowX: 'auto' } }, h('table', { class: 'data' },
      h('thead', {}, h('tr', {}, h('th', {}, '#'), h('th', {}, 'agent'), h('th', {}, 'score'), h('th', { class: 'num' }, p.qoi || 'QoI'),
        h('th', { class: 'num' }, 'error vs ref'), h('th', {}, 'mesh'), h('th', { class: 'num' }, 'DOFs'), h('th', { class: 'num' }, 'solves'),
        h('th', {}, 'gates'), anyDiag ? h('th', {}, 'diagnosis') : null, h('th', {}, ''))),
      h('tbody', {}, rows.map((r, i) => h('tr', { class: r.status === 'running' ? 'dim' : '' },
        h('td', { class: 'num muted' }, r.status === 'finished' ? i + 1 : ''),
        h('td', {}, r.agent ? h('div', { class: 'mono strong' }, r.agent) : null,
          h('div', { class: r.agent ? 'mono small muted' : 'mono' }, r.ace_run_id),
          h('div', { class: 'small muted' }, r.status === 'running' ? `running · ${timeAgo(r.created_at)}` : timeAgo(r.created_at),
            r.tokens ? ` · ${fmtInt(r.tokens)} tokens` : '', r.n_cited_bullets ? ` · ${r.n_cited_bullets} playbook bullet${r.n_cited_bullets === 1 ? '' : 's'}` : '')),
        h('td', {}, scoreCell(r, d)),
        h('td', { class: 'num' }, fmt(r.value, 6), r.qoi && r.qoi !== p.qoi ? h('div', { class: 'small muted' }, r.qoi) : null),
        h('td', { class: 'num' }, r.rel_error === null || r.rel_error === undefined ? '' : `${r.rel_error > 0 ? '+' : ''}${fmt(100 * r.rel_error, 2)}%`),
        h('td', { class: 'small', title: refinementTip(r.mesh) }, meshText(r)),
        h('td', { class: 'num' }, fmtInt(r.n_dofs)),
        h('td', { class: 'num', title: r.scored_solve ? `ACE scored solve #${r.scored_solve}` : '' }, r.n_attempts > r.n_solves ? `${r.n_solves} / ${r.n_attempts}` : r.n_solves,
          r.scored_solve ? h('div', { class: 'small muted' }, `scored #${r.scored_solve}`) : null),
        h('td', { class: 'small' }, r.gates.length ? h('span', { class: 'gate', title: r.gates.join(', ') }, h('span', { class: 'dot critical' }), r.gates.join(', ')) : h('span', { class: 'muted' }, r.status === 'crashed' ? '' : 'none'),
          r.setup_diagnostics.length ? h('div', { class: 'muted', title: r.setup_diagnostics.join('\n') }, `${r.setup_diagnostics.length} setup diagnostic${r.setup_diagnostics.length === 1 ? '' : 's'}`) : null),
        anyDiag ? h('td', { class: 'small', style: { maxWidth: '320px' } }, r.diagnosis ? h('details', {}, h('summary', {}, 'show'), h('div', { class: 'secondary', style: { marginTop: '6px', whiteSpace: 'pre-wrap' } }, r.diagnosis)) : null) : null,
        h('td', { class: 'nowrap small' }, r.submitted_run_id || r.last_run_id ? h('a', { href: `#/viewer/${r.submitted_run_id || r.last_run_id}` }, 'field →') : null)))))));
}

function meshText(r) {
  const m = r.mesh || {};
  const parts = [];
  if (m.h !== undefined && m.h !== null) parts.push(`h ${fmt(m.h, 3)}`);
  if (m.order) parts.push(`P${m.order}`);
  if (m.n_refinements) parts.push(`${m.n_refinements} refinement${m.n_refinements === 1 ? '' : 's'}`);
  return parts.join(' \u00b7 ') || '\u2013';
}

function refinementTip(m) {
  if (!m || !m.refinements || !m.refinements.length) return '';
  return m.refinements.map((x, k) => `${k + 1}: ${Object.entries(x).map(([a, b]) => `${a}=${Array.isArray(b) ? `(${b.map(v => fmt(v, 3)).join(', ')})` : typeof b === 'number' ? fmt(b, 3) : b}`).join(', ')}`).join('\n');
}

function emptyState(d) {
  return h('div', { class: 'card empty-state' },
    h('h2', {}, 'No ACE agent runs yet'),
    h('p', {}, 'This page fills in as Aryan’s ACE loop runs with the run-store bridge enabled. Every solve an agent makes becomes a run record, and every finished agent run a scored submission, in this server’s run store:'),
    h('pre', { class: 'code' }, d.store_root || 'runs/'),
    h('p', {}, 'Start the loop on this machine with the bridge pointed at that directory (FEACORE_RUNS_DIR), and the leaderboards and the score chart update live. The record format is in docs/ui.md (“ACE swarm records”).'),
    h('p', { class: 'muted small' }, 'Meanwhile, benchmark studies and agent submissions from other sources are under Runs and in each problem’s Consensus view.'));
}
