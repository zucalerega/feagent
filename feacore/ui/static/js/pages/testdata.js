// Test data: the physical tests the agents learn from (machine exports of Instron tests), the printable test
// parts, and uploads of new exports. An upload is parsed with the ACE loop's own reader (ace/evidence.py), shown,
// and saved as a ready --evidence directory under <runs>/testdata/<id>/.
// #/testdata lists everything; #/testdata/<id> shows one test. Server: feacore/ui/testdata_api.py.
import { api, authFetch, h, clear, loading, errorBox, fmt, toast } from '../api.js';
import { toolForm, taskTable } from '../tasks.js';
import { xyChart, xyLegend, SERIES_COLORS, REF_COLOR } from '../xychart.js';
import { ensureStyles, fileList, fileUrl, lightbox } from '../media.js';

const kN = x => (x === null || x === undefined || !isFinite(x)) ? '–' : `${(x / 1000).toFixed(2)} kN`;
const tile = (label, value, sub) => h('div', { class: 'tile' }, h('div', { class: 'label' }, label), h('div', { class: 'value' }, value), sub ? h('div', { class: 'sub' }, sub) : null);
const testLink = id => `#/testdata/${encodeURIComponent(id)}`;
const compareLink = id => `#/nonlinear?tab=runs&test=${encodeURIComponent(id)}`;
// empty an element and append the parts (the DOM's append would write null as the text "null")
const put = (el, ...kids) => { clear(el).append(...kids.flat(Infinity).filter(k => k !== null && k !== undefined && k !== false)); return el; };

function stiffness(stats) {
  const c = ((stats || {}).cycles || []).find(x => x.stiffness_N_per_mm);
  return c ? `${fmt(c.stiffness_N_per_mm, 4)} N/mm` : '–';
}

function curveChart(curves, { height = 380, width = 900, legend = true } = {}) {
  // a single curve is the measured test: ink; extra curves (the dogbone in the same folder) take the palette
  const series = curves.map((c, i) => ({ label: c.label, color: i === 0 ? REF_COLOR : SERIES_COLORS[(i - 1) % SERIES_COLORS.length],
                                         width: i === 0 ? 2.2 : 2, points: c.points }));
  return h('div', {}, legend && series.length > 1 ? xyLegend(series) : null,
    xyChart({ series, xTitle: 'displacement (mm)', yTitle: 'force (N)', xUnit: 'mm', yUnit: 'N', height, width }));
}

export async function render(el, args) {
  ensureStyles();
  if (args && args[0]) return detail(el, args[0]);
  el.append(h('div', { class: 'page-head' },
    h('div', {}, h('h1', {}, 'Test data'),
      h('p', { class: 'lede' }, 'Physical tests of printed parts: the machine’s force-displacement curve, read the way the agents’ loop reads it. Upload a new export and it becomes evidence the agents learn from, and a curve to compare simulations with.')),
    h('div', { class: 'row', style: { gap: '10px' } }, h('a', { class: 'btn', href: '#/nonlinear' }, 'Nonlinear'), h('a', { class: 'btn primary', href: '#upload', onclick: e => { e.preventDefault(); document.getElementById('td-upload').scrollIntoView({ behavior: 'smooth' }); } }, 'Upload an export'))));
  const body = h('div', { class: 'stack' }, loading());
  el.append(body);
  let d;
  try { d = await api('/api/testdata'); } catch (e) { put(body, errorBox(e)); return; }
  const tasks = taskTable({ group: 'testdata' }, { limit: 10, empty: 'No checks run yet.' });
  put(body, 
    h('div', { class: 'card' }, h('div', { class: 'card-head' }, h('h2', {}, 'Physical tests'), h('span', { class: 'muted small' }, `uploads are saved in ${d.store}`)),
      testCards(d.tests)),
    uploadCard(d),
    partsCard(d.parts),
    h('div', { class: 'card' }, h('h2', {}, 'Check a test the way the loop reads it'),
      toolForm('testdata.summarize', { defaults: d.tests.length ? { spec: d.tests[0].spec_abs } : {}, submitLabel: 'Check', onStarted: t => { location.hash = `#/tasks/${t.id}`; } }),
      h('h3', { style: { marginTop: '16px' } }, 'Recent checks'), tasks.el));
  return () => tasks.stop();
}

// ------------------------------------------------------------------ the list
function testCards(tests) {
  if (!tests.length) return h('p', { class: 'muted' }, 'No tests yet.');
  return h('div', { class: 'grid-2' }, tests.map(t => h('div', { class: 'card tight stack' },
    h('div', { class: 'row', style: { justifyContent: 'space-between', alignItems: 'baseline' } },
      h('div', {}, h('a', { href: testLink(t.id), class: 'strong' }, t.test_id || t.id),
        h('div', { class: 'mono small muted' }, t.spec_ref || t.id)),
      h('span', { class: 'badge' }, t.uploaded ? 'uploaded' : 'committed')),
    t.error ? errorBox(t.error) : null,
    h('div', { class: 'small' }, 'problem ', t.problem_id ? h('a', { href: `#/benchmarks/${encodeURIComponent(t.problem_id)}` }, t.problem_id) : '–',
      t.specimen ? ` · specimen ${t.specimen}` : '', t.part ? ` · part ${t.part}` : ''),
    t.description ? h('p', { class: 'small secondary', style: { margin: 0 } }, t.description.length > 260 ? `${t.description.slice(0, 260)}…` : t.description) : null,
    t.curve && t.curve.length ? curveChart([{ label: t.test_id || t.id, points: t.curve }], { height: 280, width: 600, legend: false }) : null,
    h('div', { class: 'row small', style: { gap: '18px' } },
      h('span', {}, 'peak ', h('b', {}, kN((t.stats || {}).peak_N)), (t.stats || {}).disp_at_peak_mm != null ? ` at ${fmt(t.stats.disp_at_peak_mm, 3)} mm` : ''),
      h('span', {}, 'fitted slope ', h('b', {}, stiffness(t.stats))),
      h('span', {}, `${(t.stats || {}).rows ?? '–'} rows`)),
    h('div', { class: 'row', style: { gap: '8px' } },
      h('a', { class: 'btn small', href: testLink(t.id) }, 'Open'),
      h('a', { class: 'btn small', href: compareLink(t.id) }, 'Compare with simulations')))));
}

// ------------------------------------------------------------------ upload
function uploadCard(d) {
  const fileIn = h('input', { type: 'file', accept: '.csv,.txt,.tsv,.dat' });
  const problem = h('select', {}, h('option', { value: '' }, '(choose the problem it tests)'),
    d.problems.map(p => h('option', { value: p.id }, `${p.id}${p.physical ? '' : ' (benchmark)'}`)));
  const preview = h('div', { class: 'stack' });
  const err = h('div');
  let file = null, seq = 0;
  const f = {};      // form fields of the preview (columns, units, fit) and the metadata
  const field = (label, input, help) => h('label', { class: 'field' }, h('span', { class: 'section-title' }, label), input, help ? h('span', { class: 'hint' }, help) : null);
  const text = (key, placeholder, rows) => {
    const el = rows ? h('textarea', { rows, style: { width: '100%' }, placeholder }) : h('input', { type: 'text', placeholder });
    el.addEventListener('input', () => { f[key] = el.value; });
    return el;
  };

  function formData(extra = {}) {
    const fd = new FormData();
    fd.append('file', file, file.name);
    if (problem.value) fd.append('problem_id', problem.value);
    for (const [k, v] of Object.entries({ ...f, ...extra })) if (v !== undefined && v !== null && v !== '') fd.append(k, typeof v === 'boolean' ? String(v) : v);
    return fd;
  }

  async function post(url, fd) {
    const res = await authFetch(url, { method: 'POST', body: fd });
    const body = await res.json().catch(() => ({ detail: `${res.status} ${res.statusText}` }));
    return { ok: res.ok, status: res.status, body };
  }

  async function readIt() {
    clear(err);
    if (!file) { err.append(errorBox('Choose the export file first.')); return; }
    const p = d.problems.find(x => x.id === problem.value);
    if (p && f.sign === undefined) f.sign = p.compression ? '-1' : '+1';
    const my = ++seq;
    put(preview, loading('Reading it with the loop’s reader…'));
    let r;
    try { r = await post('/api/testdata/parse', formData()); } catch (e) { r = { ok: false, body: { detail: String(e) } }; }
    if (my === seq) drawPreview(r);        // a newer read replaced this one
  }

  function columnSelect(cols, key, value, none = false) {
    const sel = h('select', { onchange: () => { f[key] = sel.value; if (key === 'strain' && sel.value) f.displacement = ''; readIt(); } },
      none ? h('option', { value: '' }, '(none)') : null,
      cols.filter(c => c.name).map(c => h('option', { value: c.name, selected: c.name === value ? true : null }, `${c.index + 1}: ${c.name}${c.unit_text ? ` ${c.unit_text}` : ''}`)));
    return sel;
  }
  const unitSelect = (key, options, value) => {
    const sel = h('select', { onchange: () => { f[key] = sel.value; readIt(); } }, options.map(u => h('option', { value: u, selected: u === value ? true : null }, u)));
    return sel;
  };

  // the columns and units the reader used, editable: a change reads the file again
  function columnFields(sn, data) {
    const strainMode = !!(f.strain || data.strain);
    const toggle = h('input', { type: 'checkbox', checked: strainMode ? true : null });
    toggle.onchange = () => {
      const named = sn.columns.filter(x => x.name && x.name !== data.load);
      f.strain = toggle.checked ? ((named.find(x => /strain/i.test(x.name)) || named[0] || {}).name || '') : '';
      f.displacement = '';
      readIt();
    };
    const gauge = h('input', { type: 'number', step: 'any', value: f.gauge_length_mm || data.gauge_length_mm || '' });
    gauge.onchange = () => { f.gauge_length_mm = gauge.value; readIt(); };
    return h('div', { class: 'form-grid nl-form' },
      field('Load column', columnSelect(sn.columns, 'load', data.load || f.load)),
      field('Load unit', unitSelect('load_unit', d.units.load, data.load_unit || f.load_unit || 'n')),
      strainMode ? field('Strain column', columnSelect(sn.columns, 'strain', data.strain || f.strain, true))
        : field('Displacement column', columnSelect(sn.columns, 'displacement', data.displacement || f.displacement)),
      strainMode ? field('Gauge length (mm)', gauge)
        : field('Displacement unit', unitSelect('displacement_unit', d.units.displacement, data.displacement_unit || f.displacement_unit || 'mm')),
      field('Strain instead', h('span', { class: 'check' }, toggle, 'a strain column + gauge length')));
  }

  function drawPreview(r) {
    const b = r.body || {};
    const sn = b.sniff;
    const head = sn ? h('p', { class: 'small muted' }, `${sn.delimiter}-separated${sn.decimal_comma ? ' with decimal commas' : ''}; header on line ${sn.header_row}${sn.units_row ? `, units on line ${sn.units_row}` : ''}; numbers from line ${sn.first_data_row}.`) : null;
    if (!r.ok) { put(preview, errorBox(b.detail || 'could not read the file'), head, sn ? columnFields(sn, b.data || {}) : null); return; }
    const st = b.stats || {};
    put(preview, head, columnFields(sn, b.data || {}),
      b.warnings && b.warnings.length ? h('div', { class: 'flag-list' }, b.warnings.map(w => h('div', { class: 'flag warning' }, h('span', { class: 'ico' }, '!'), h('div', {}, h('span', {}, w))))) : null,
      h('div', { class: 'nl-facts' },
        tile('peak', kN(st.peak_N), `at ${fmt(st.disp_at_peak_mm, 3)} mm`), tile('rows', st.rows ?? '–', `units: ${b.units.load}, ${b.units.displacement}`),
        tile('loadings', st.loadings ?? '–', `fit window ${b.window.join('-')} of each peak${b.until_peak ? ', up to the peak' : ''}`),
        tile('fitted slope', stiffness(st), (st.cycles || []).length ? `${st.cycles.length} fitted loading${st.cycles.length > 1 ? 's' : ''}` : 'no loading fits')),
      curveChart([{ label: file.name, points: b.curve }], { height: 320, legend: false }),
      loopBox(b.loop));
  }

  const until = h('input', { type: 'checkbox' });
  until.onchange = () => { f.until_peak = until.checked; if (file) readIt(); };
  const lo = h('input', { type: 'number', step: '0.01', min: 0, max: 1, value: 0.1 });
  const hi = h('input', { type: 'number', step: '0.01', min: 0, max: 1, value: 0.9 });
  lo.onchange = () => { f.window_lo = lo.value; if (file) readIt(); };
  hi.onchange = () => { f.window_hi = hi.value; if (file) readIt(); };
  const sign = h('select', { onchange: () => { f.sign = sign.value; } }, h('option', { value: '+1' }, '+1: tension (the FEA quantity grows with the load)'), h('option', { value: '-1' }, '-1: compression (the machine reads it positive, the FEA negative)'));
  problem.addEventListener('change', () => {
    const p = d.problems.find(x => x.id === problem.value);
    if (p) { f.sign = p.compression ? '-1' : '+1'; sign.value = f.sign; if (p.compression) { until.checked = true; f.until_peak = true; } }
    if (file) readIt();
  });
  fileIn.addEventListener('change', () => { file = fileIn.files[0] || null; for (const k of ['load', 'displacement', 'strain', 'load_unit', 'displacement_unit']) delete f[k]; if (file) readIt(); });

  const save = h('button', { class: 'btn primary', type: 'button' }, 'Save as a test record');
  save.onclick = async () => {
    clear(err);
    if (!file) { err.append(errorBox('Choose the export file first.')); return; }
    if (!problem.value) { err.append(errorBox('Choose the problem it tests: the loop uses a test for one problem.')); return; }
    save.disabled = true;
    const r = await post('/api/testdata', formData());
    save.disabled = false;
    if (!r.ok) { err.append(errorBox(r.body.detail || `${r.status}`)); return; }
    toast('Saved');
    location.hash = testLink(r.body.id);
  };

  return h('div', { class: 'card', id: 'td-upload' },
    h('h2', {}, 'Upload a machine export'),
    h('p', { class: 'secondary' }, 'A CSV or TXT export as Bluehill (Instron) writes it: a header row of column names, units in the header ("Load (N)") or in a row under it ("(N)"), comma, semicolon or tab separated, decimal commas allowed. Load in N, kN, mN, lbf or kgf; displacement in m, mm, um or in, or a strain column with a gauge length. It is read with the loop’s own reader (ace/evidence.py), so what you see here is what the agents get.'),
    h('div', { class: 'form-grid nl-form' },
      field('Export file', fileIn), field('Problem', problem, 'the setup this test measures'),
      field('Fit up to the peak', h('span', { class: 'check' }, until, 'a test run to failure'), 'fit only the loading branch before the peak'),
      field('Fit window, low', lo, 'share of each loading’s peak'), field('Fit window, high', hi),
      field('Sign', sign)),
    preview,
    h('h3', { style: { marginTop: '14px' } }, 'About the test'),
    h('div', { class: 'form-grid nl-form' },
      field('Test id', text('test_id', 'e.g. cbracket-2')), field('Specimen', text('specimen', 'e.g. print 2, 2 min cure')),
      field('Part', text('part', 'e.g. abc_00000065')), field('Measured modulus E (GPa)', text('E_GPa', 'optional: rescales the predictions'))),
    field('Setup', text('description', 'resin, cure, printer; machine, load cell, fixtures; how the displacement is measured', 3)),
    field('Notes', text('notes', 'what you saw: seating, slips, where it cracked', 3)),
    h('div', { class: 'row', style: { alignItems: 'center', gap: '12px', marginTop: '10px' } }, save,
      h('span', { class: 'muted small' }, 'Saved under the run store as test.json + the export: pass that folder to the loop with --evidence.')),
    err,
    h('p', { class: 'muted small' }, 'Only a PDF report? The committed curve was traced from the report’s plot image with ', h('code', {}, 'scripts/digitize_bluehill_plot.py'), '.'));
}

function loopBox(loop) {
  if (!loop) return h('p', { class: 'muted small' }, 'Choose the problem to see how the loop reads it.');
  if (!loop.ok) return h('div', { class: 'flag-list' }, h('div', { class: 'flag critical' }, h('span', { class: 'ico' }, '✕'), h('div', {}, h('b', {}, 'The loop cannot use it as it is'), h('span', {}, loop.error))));
  const m = loop.measurement || {};
  const stiff = loop.compliance ? Math.abs(1e-3 / loop.compliance) : null;
  return h('div', { class: 'stack' },
    h('div', { class: 'nl-facts' },
      tile('measured slope', stiff ? `${fmt(stiff, 4)} N/mm` : '–', `compliance ${fmt(loop.compliance, 4)} m/N`),
      tile('uncertainty (1 sigma)', loop.u_rel_total != null ? `${(100 * loop.u_rel_total).toFixed(1)} %` : '–', Object.entries(loop.budget || {}).map(([k, v]) => `${k} ${(100 * v).toFixed(1)} %`).join(', ')),
      tile(`${m.name || 'measurement'} at ${fmt(loop.nominal_load, 4)} N`, `${fmt(loop.value_at_nominal_load, 4)} m`, `${m.type || ''} of ${m.field || ''}${m.group ? ` on ${m.group}` : ''}`),
      tile('loadings used', (loop.cycles_used || []).join(', ') || '–', (loop.cycles_dropped || []).length ? `dropped ${loop.cycles_dropped.join(', ')} (seating)` : null)),
    loop.nonlinear_note ? h('p', { class: 'small muted' }, loop.nonlinear_note) : null);
}

// ------------------------------------------------------------------ printable parts
function partsCard(p) {
  if (!p || !p.parts.length) return null;
  const cols = ['Source', 'Grip ends added', 'Size (mm)', 'Resin (cm³, no supports)'];
  const setup = ['Lower end', 'Upper end', 'Extensometer marks', 'Fixtures to make'];
  return h('div', { class: 'card' },
    h('div', { class: 'card-head' }, h('h2', {}, 'Printable test parts'), h('code', { class: 'muted small' }, p.readme)),
    h('p', { class: 'secondary' }, 'Parts with grip ends for pull tests on the Instron (STL for printing, in mm; STEP for FEA). The test plan and the resin data are in the README.'),
    p.image ? h('button', { class: 'nl-tile', type: 'button', style: { maxWidth: '520px', marginBottom: '14px' }, onclick: () => lightbox({ ref: p.image, kind: 'image', title: 'printable test parts' }) },
      h('div', { class: 'nl-thumb' }, h('img', { src: fileUrl(p.image), alt: 'the printable test parts', loading: 'lazy' }))) : null,
    h('table', { class: 'data' },
      h('thead', {}, h('tr', {}, h('th', {}, 'part'), h('th', {}, 'what it is'), h('th', {}, 'test setup'), h('th', {}, 'files'))),
      h('tbody', {}, p.parts.map(x => h('tr', {},
        h('td', {}, h('div', { class: 'strong mono' }, x.id), x.problem ? h('a', { class: 'small', href: `#/benchmarks/${encodeURIComponent(x.problem)}` }, 'problem') : h('span', { class: 'muted small' }, 'no problem yet')),
        h('td', { class: 'small' }, cols.filter(c => x.info[c]).map(c => h('div', {}, h('span', { class: 'muted' }, `${c}: `), x.info[c].replace(/`/g, '')))),
        h('td', { class: 'small' }, setup.filter(c => x.setup[c]).map(c => h('div', {}, h('span', { class: 'muted' }, `${c}: `), x.setup[c]))),
        h('td', { class: 'small nowrap' },
          h('a', { href: fileUrl(x.stl, true) }, 'STL'), x.step ? [' · ', h('a', { href: fileUrl(x.step, true) }, 'STEP')] : null))))));
}

// ------------------------------------------------------------------ one test
async function detail(el, id) {
  el.append(loading());
  let t;
  try { t = await api(`/api/testdata/item?id=${encodeURIComponent(id)}`); } catch (e) { put(el, errorBox(e)); return; }
  const spec = t.spec || {};
  const curves = [{ label: `${t.test_id || id} (${spec.data ? spec.data.file : ''})`, points: t.curve },
                  ...(t.extra_curves || []).map(c => ({ label: `${c.name} (same folder)`, points: c.curve }))];
  const del = t.uploaded ? h('button', { class: 'btn', type: 'button', onclick: async () => {
    if (!confirm(`Delete the uploaded test ${t.test_id || id}? Its folder in the run store is removed.`)) return;
    try { await api(`/api/testdata/item?id=${encodeURIComponent(id)}`, { method: 'DELETE' }); toast('Deleted'); location.hash = '#/testdata'; } catch (e) { toast(String(e)); }
  } }, 'Delete') : null;
  const kv = (k, v) => v === undefined || v === null || v === '' ? null : h('div', { class: 'small' }, h('span', { class: 'muted' }, `${k}: `), typeof v === 'object' ? JSON.stringify(v) : String(v));
  const pre = s => h('pre', { class: 'nl-pre' }, s);
  put(el, 
    h('div', { class: 'page-head' },
      h('div', {}, h('h1', {}, t.test_id || id, ' ', h('span', { class: 'badge' }, t.uploaded ? 'uploaded' : 'committed')),
        h('p', { class: 'lede' }, 'problem ', h('a', { href: `#/benchmarks/${encodeURIComponent(t.problem_id)}` }, t.problem_id), t.specimen ? ` · specimen ${t.specimen}` : '', t.part ? ` · part ${t.part}` : ''),
        h('p', { class: 'muted small' }, h('code', {}, t.spec_ref))),
      h('div', { class: 'row', style: { gap: '8px' } }, h('a', { class: 'btn primary', href: compareLink(id) }, 'Compare with simulations'), del, h('a', { class: 'btn', href: '#/testdata' }, 'All tests'))),
    h('div', { class: 'stack' },
      h('div', { class: 'card' }, h('h2', {}, 'The measured curve'), curveChart(curves),
        h('p', { class: 'muted small' }, `${(t.stats || {}).rows ?? '–'} rows; displacement as the machine recorded it (${(t.units || {}).displacement || 'mm'}, shown in mm), load in N.`)),
      h('div', { class: 'card' }, h('h2', {}, 'What the loop reads'), loopBox(t.loop)),
      h('div', { class: 'card' }, h('h2', {}, 'Use it with the agents'),
        h('p', { class: 'secondary' }, 'Check it, then give it to a loop from the start (or drop the folder into a running loop’s <out>/evidence/). With --evidence-score the runs of this problem are scored against the test instead of the reference.'),
        pre(t.commands.summarize), pre(t.commands.loop), pre(t.commands.score)),
      h('div', { class: 'card' }, h('h2', {}, 'The test'),
        spec.description ? h('p', {}, spec.description) : null,
        spec.notes ? h('p', { class: 'secondary' }, spec.notes) : null,
        kv('columns', spec.data), kv('measurement', spec.measurement), kv('fit', spec.fit), kv('uncertainty', spec.uncertainty),
        kv('material', spec.material), kv('lab', spec.lab),
        h('details', {}, h('summary', {}, 'The spec as JSON'), pre(JSON.stringify(spec, null, 1)))),
      h('div', { class: 'card' }, h('h2', {}, 'Files'), fileList(t.files))));
}
