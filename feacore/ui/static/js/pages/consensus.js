// Consensus: the multi-agent view. Per problem and QoI, every submission's finest-mesh value with its
// GCI band, its Richardson-extrapolated value, the population median and the reference.
import { api, h, clear, fmt, fmtPct, fmtSigned, loading, errorBox, statusChip, timeAgo } from '../api.js';
import { forestChart } from '../plot.js';

export async function render(el, args) {
  if (!args.length) return renderIndex(el);
  return renderProblem(el, args[0]);
}

async function renderIndex(el) {
  el.append(loading());
  let data;
  try { data = await api('/api/problems'); } catch (e) { clear(el).append(errorBox(e)); return; }
  clear(el);
  el.append(h('div', { class: 'page-head' }, h('div', {}, h('h1', {}, 'Consensus across agents'),
    h('p', { class: 'lede' }, 'Many agents solve the same problem with their own geometry, mesh and boundary conditions. Each submits a sequence of refined meshes; the verification layer checks every submission on its own (convergence, GCI band, observed order) and against the others.'))));
  const withSubs = data.problems.filter(p => p.n_submissions > 0).sort((a, b) => b.n_submissions - a.n_submissions);
  const others = data.problems.filter(p => !p.n_submissions);
  if (withSubs.length) {
    el.append(h('div', { class: 'card' }, h('h2', {}, 'Problems with submissions'),
      h('table', { class: 'data' }, h('thead', {}, h('tr', {}, h('th', {}, 'problem'), h('th', { class: 'num' }, 'submissions'), h('th', {}, 'tags'))),
        h('tbody', {}, withSubs.map(p => h('tr', { class: 'click', onclick: () => { location.hash = `#/consensus/${encodeURIComponent(p.id)}`; } },
          h('td', {}, h('div', {}, p.title), h('div', { class: 'mono small muted' }, p.id)), h('td', { class: 'num' }, p.n_submissions),
          h('td', {}, h('div', { class: 'chips' }, p.tags.slice(0, 5).map(t => h('span', { class: 'chip' }, t.replace(/_/g, ' ')))))))))));
  } else {
    el.append(h('div', { class: 'card empty' }, h('h2', {}, 'No submissions yet'),
      h('p', {}, 'Agents (or their harness) register a submission after solving on several meshes:'),
      h('pre', { class: 'code' }, submissionExample())));
  }
  if (others.length) {
    el.append(h('details', { class: 'card tight', style: { marginTop: '18px' } }, h('summary', {}, `${others.length} problems without submissions`),
      h('div', { class: 'chips', style: { marginTop: '10px' } }, others.map(p => h('a', { class: 'chip', href: `#/consensus/${encodeURIComponent(p.id)}` }, p.id)))));
  }
}

function submissionExample() {
  return `# solve each mesh (returns run records), then group them:
curl -F case=@case_h1.json -F mesh=@part_h1.msh -F agent_id=agent-07 \\
     -F problem_id=PROBLEM ${location.origin}/api/solve
curl -X POST ${location.origin}/api/submissions -H 'Content-Type: application/json' \\
     -d '{"problem_id": "PROBLEM", "agent_id": "agent-07", "run_ids": ["run-...", "run-...", "run-..."]}'`;
}

async function renderProblem(el, pid) {
  el.append(loading('Computing consensus…'));
  let d;
  try { d = await api(`/api/consensus/${encodeURIComponent(pid)}`); } catch (e) { clear(el).append(errorBox(e)); return; }
  clear(el);
  el.append(h('div', { class: 'page-head' }, h('div', {},
    h('div', { class: 'secondary small' }, h('a', { href: '#/consensus' }, '← Consensus'), ' · ', h('a', { href: `#/benchmarks/${encodeURIComponent(pid)}` }, 'problem')),
    h('h1', {}, d.title || pid), h('div', { class: 'chips', style: { marginTop: '6px' } }, h('span', { class: 'badge mono' }, pid),
      h('span', { class: 'badge' }, `${d.n_submissions} submissions`), h('span', { class: 'badge' }, `${d.agents.length} agents`)))));
  const rows = d.qois.flatMap(q => q.submissions);
  const conv = rows.filter(r => ['converged', 'exact', 'expected_divergence'].includes(r.verdict)).length;
  const flagged = rows.filter(r => r.flags.some(f => ['divergent', 'singular_suspected'].includes(f))).length;
  const outl = rows.filter(r => r.flags.includes('outlier_vs_consensus')).length;
  el.append(h('div', { class: 'tiles', style: { marginBottom: '20px' } },
    tile('Submissions', d.n_submissions, `${d.agents.length} distinct agents`),
    tile('Converged answers', `${conv} / ${rows.length}`, 'submission × QoI pairs'),
    tile('Diverging / singular', flagged, 'stress grows under refinement'),
    tile('Outliers vs consensus', outl, 'beyond band and tolerance')));
  if (!d.n_submissions) {
    el.append(h('div', { class: 'card empty' }, h('h2', {}, 'No submissions for this problem yet'),
      h('p', {}, 'Reference studies of the fixed solver are shown below once they exist.'), h('pre', { class: 'code' }, submissionExample().replaceAll('PROBLEM', pid))));
  }
  const width = Math.max(720, el.clientWidth - 48);
  for (const q of d.qois) el.append(qoiBlock(q, width));
}

const tile = (label, value, sub) => h('div', { class: 'tile' }, h('div', { class: 'label' }, label), h('div', { class: 'value' }, value), h('div', { class: 'sub' }, sub));

function rowStatus(r) {
  if (r.flags.includes('outlier_vs_consensus')) return 'critical';
  return r.status;
}

function qoiBlock(q, width) {
  const ref = q.reference;
  const cons = q.consensus;
  const rows = [];
  for (const s of q.reference_studies) {
    rows.push({ kind: 'study', label: s.source === 'validation' ? `validation · ${s.backend} P${s.order}` : `fixed solver · ${s.backend} P${s.order}`,
      value: s.fine, lo: s.band_abs !== null ? s.fine - s.band_abs : NaN, hi: s.band_abs !== null ? s.fine + s.band_abs : NaN,
      ext: s.extrapolation_reliable ? (s.phi_ext ?? NaN) : NaN,
      status: s.status, tip: `<b>reference study</b> ${s.study_id || ''}<br>finest mesh ${fmt(s.fine, 6)}${s.band_abs !== null ? ` ± ${fmt(s.band_abs, 3)}` : ''}<br>extrapolated ${fmt(s.phi_ext, 6)}<br>${s.label}` });
  }
  const subs = [...q.submissions].sort((a, b) => (a.agent_id || '').localeCompare(b.agent_id || ''));
  for (const r of subs) {
    rows.push({ kind: 'submission', label: r.agent_id || r.submission_id, value: r.fine,
      lo: r.band_abs !== null ? r.fine - r.band_abs : NaN, hi: r.band_abs !== null ? r.fine + r.band_abs : NaN,
      ext: r.extrapolation_reliable ? (r.phi_ext ?? NaN) : NaN,
      status: rowStatus(r),
      tip: `<b>${r.agent_id || ''}</b> <span class="t-muted">${r.submission_id}</span><br>finest mesh (${r.n_levels} meshes): ${fmt(r.fine, 6)}${r.band_abs !== null ? ` ± ${fmt(r.band_abs, 3)} (GCI)` : ''}<br>extrapolated: ${fmt(r.phi_ext, 6)} · p = ${fmt(r.p, 3)}<br>${r.label}${r.flags.length ? `<br><span class="t-muted">${r.flags.join(', ')}</span>` : ''}` });
  }
  const chart = forestChart({ rows, reference: ref ? { value: ref.value, tol: ref.tolerance } : null,
    consensus: cons ? { value: cons.median, lo: cons.median - cons.sigma_robust, hi: cons.median + cons.sigma_robust } : null, xTitle: q.name, width });
  const legend = h('div', { class: 'legend' },
    h('span', { class: 'key' }, h('span', { class: 'mk', style: { background: '#2a78d6' } }), 'finest-mesh value ± GCI band'),
    h('span', { class: 'key' }, h('span', { class: 'dia', style: { borderColor: '#eb6834' } }), 'Richardson extrapolated'),
    cons ? h('span', { class: 'key' }, h('span', { class: 'ln dash', style: { borderColor: '#6250d6' } }), 'consensus (median)') : null,
    ref ? h('span', { class: 'key' }, h('span', { class: 'ln', style: { borderColor: '#0b0b0b' } }), `reference ± ${fmtPct(ref.tolerance, 2)}`) : null,
    q.reference_studies.length ? h('span', { class: 'key' }, h('span', { class: 'mk', style: { background: '#0b0b0b' } }), 'fixed-solver reference study') : null);
  const summary = cons ? h('div', { class: 'stat-row' },
    h('div', { class: 'stat' }, h('div', { class: 'label' }, 'consensus (median of best estimates)'), h('div', { class: 'value' }, fmt(cons.median, 6))),
    h('div', { class: 'stat' }, h('div', { class: 'label' }, 'robust spread (1.48 MAD)'), h('div', { class: 'value sm' }, cons.median ? fmtPct(cons.sigma_robust / Math.abs(cons.median), 2) : fmt(cons.sigma_robust, 3))),
    h('div', { class: 'stat' }, h('div', { class: 'label' }, 'min – max'), h('div', { class: 'value sm' }, `${fmt(cons.min, 5)} – ${fmt(cons.max, 5)}`)),
    cons.err_ref !== undefined ? h('div', { class: 'stat' }, h('div', { class: 'label' }, 'consensus vs reference'), h('div', { class: 'value sm' }, fmtSigned(cons.err_ref, 2))) : null) : null;
  const table = subs.length ? h('table', { class: 'data', style: { marginTop: '14px' } },
    h('thead', {}, h('tr', {}, h('th', {}, 'agent'), h('th', { class: 'num' }, 'meshes'), h('th', { class: 'num' }, 'finest mesh'), h('th', { class: 'num' }, '± GCI'),
      h('th', { class: 'num' }, 'extrapolated'), h('th', { class: 'num' }, 'p (theory)'), h('th', {}, 'verdict'), ref ? h('th', { class: 'num' }, 'error vs ref') : null, h('th', {}, 'flags'))),
    h('tbody', {}, subs.map(r => h('tr', {},
      h('td', {}, h('div', { style: { fontWeight: 600 } }, r.agent_id || '–'), h('div', { class: 'mono small muted' }, `${r.submission_id} · ${timeAgo(r.created_at)}`)),
      h('td', { class: 'num' }, r.n_levels), h('td', { class: 'num' }, fmt(r.fine, 6)), h('td', { class: 'num' }, r.gci_fine !== null ? fmtPct(r.gci_fine, 2) : '–'),
      h('td', { class: 'num' }, fmt(r.phi_ext, 6), r.phi_ext !== null && r.extrapolation_reliable === false ? h('div', { class: 'small muted' }, 'unreliable') : null),
      h('td', { class: 'num' }, `${fmt(r.p, 3)} (${fmt(r.p_theory, 2)})`),
      h('td', {}, statusChip(rowStatus(r), r.flags.includes('outlier_vs_consensus') && r.status === 'good' ? `${r.label}, but an outlier` : r.label)),
      ref ? h('td', { class: 'num' }, fmtSigned(r.err_ref_best, 2)) : null,
      h('td', {}, h('div', { class: 'chips' }, r.explanations.map(e => h('span', { class: 'chip', title: `${e.title}: ${e.text}` }, h('span', { class: `dot ${e.severity === 'info' ? '' : e.severity}` }), e.title))))))))
    : null;
  return h('div', { class: 'card', style: { marginBottom: '20px' } },
    h('div', { class: 'card-head' }, h('h2', { class: 'mono' }, q.name),
      ref ? h('span', { class: 'secondary' }, `reference ${fmt(ref.value, 6)} (${ref.kind.replace(/_/g, ' ')})`) : h('span', { class: 'secondary' }, 'no reference value: judged on convergence and agreement')),
    summary, rows.length ? [legend, chart] : h('p', { class: 'muted' }, 'No submissions or reference studies for this QoI yet.'), table);
}
