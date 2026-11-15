// Click-through checks of the report page. Usage: node report_flows.js <url> <mode: api|file>
// Prints one JSON object {name: true|false|detail}. Needs `playwright` (require works from any directory).
const { chromium } = require('playwright');
(async () => {
  const [url, mode] = process.argv.slice(2);
  const browser = await chromium.launch();
  const page = await (await browser.newContext({ viewport: { width: 1400, height: 900 } })).newPage();
  const errors = [], external = [];
  page.on('pageerror', e => errors.push(e.message));
  page.on('console', m => { if (m.type() === 'error') errors.push(m.text()); });
  page.on('request', r => { const u = r.url(); if (/^https?:/.test(u) && !u.startsWith(new URL(url).origin)) external.push(u); });
  const R = {};
  const openAll = i => page.evaluate(x => { for (let e = document.getElementById(x); e; e = e.parentElement) if (e.tagName === 'DETAILS') e.open = true; document.getElementById(x).scrollIntoView({behavior: 'instant'}); }, i);
  await page.goto(url);
  await page.waitForTimeout(400);
  // 1. a link opens its card, marks it, and Back returns
  const first = await page.evaluate(() => location.hash);
  await page.click('#summary a.ref[href^="#REQ-MOV-002"]');
  await page.waitForTimeout(500);
  R.link_hash = await page.evaluate(() => location.hash) === '#REQ-MOV-002';
  R.link_card_open = await page.evaluate(() => { const e = document.getElementById('REQ-MOV-002'); return e.open === true && e.classList.contains('is-target'); });
  await page.goBack(); await page.waitForTimeout(300);
  R.back = (await page.evaluate(() => location.hash)) === first;
  // 2. deep link
  await page.goto(url + '#CMPL-REQ-MOV-010-ROUNDNESS-LC-PRESTART'); await page.reload(); await page.waitForTimeout(600);
  R.deep_link = await page.evaluate(() => document.getElementById('CMPL-REQ-MOV-010-ROUNDNESS-LC-PRESTART').open === true);
  // 3. Why panel
  await page.click('.why[data-why="CMPL-REQ-MOV-010-ROUNDNESS-LC-PRESTART"]');
  R.why_open = await page.isVisible('#why-panel');
  R.why_tree = await page.locator('#why-panel .tree li').count() > 3;
  await page.click('#why-panel .wp-tab[data-tab=down]');
  await page.click('#why-panel .wp-tab[data-tab=up]');
  await page.click('#why-panel button:has-text("Open all levels")');
  await page.waitForTimeout(200);
  R.why_source = await page.locator('#why-panel .tn-src').count() > 0;
  await page.keyboard.press('Escape');
  R.why_close = !(await page.isVisible('#why-panel'));
  await page.click('.why[data-why="roundness_max"], .why[data-why="REQ-MOV-010"]').catch(() => {});
  await page.keyboard.press('Escape');
  // 4. preview
  await page.evaluate(() => document.querySelector('#summary a.ref[href^="#REQ-MOV-002"]').scrollIntoView({behavior: 'instant', block: 'center'})); await page.waitForTimeout(400); await page.hover('#summary a.ref[href^="#REQ-MOV-002"]'); await page.waitForTimeout(900);
  R.preview = await page.isVisible('#peek') && (await page.textContent('#peek')).length > 10;
  // 5. filters
  await page.goto(url + '#decisions'); await page.waitForTimeout(500);
  const all = +(await page.textContent('#flt-n'));
  await page.click('.fchip.needs'); const needs = +(await page.textContent('#flt-n'));
  await page.click('#flt-clear');
  await page.fill('#flt-q', 'mesh'); const q = +(await page.textContent('#flt-n'));
  R.filters = needs > 0 && needs < all && q > 0 && q < all;
  await page.click('#flt-clear');
  await page.click('.fchip[data-f=stage][data-v=mesh]'); R.filter_stage = (+(await page.textContent('#flt-n'))) < all;
  await page.click('#flt-clear');
  // 6. diagram
  const dia = await page.locator('svg.dz-diagram [data-ref]').count();
  if (dia) {
    const ref = await page.evaluate(() => { const g = [...document.querySelectorAll('svg.dz-diagram [data-ref]')].find(x => document.getElementById(x.getAttribute('data-ref'))); return g && g.getAttribute('data-ref'); });
    await page.evaluate(r => { document.querySelector(`svg.dz-diagram [data-ref="${r}"]`).dispatchEvent(new MouseEvent('click', { bubbles: true })); }, ref);
    await page.waitForTimeout(300);
    R.diagram_click = (await page.evaluate(() => location.hash)) === '#' + ref;
  } else R.diagram_click = 'no diagram';
  // 7. reviews
  const forms = await page.locator('.rv-form').count();
  if (mode === 'api') {
    await page.goto(url + '#FND-REQ-MOV-001-MEOP-ENVELOPE'); await page.waitForTimeout(600);
    const id = await page.evaluate(() => { const b = document.querySelector('.fld.review[data-state=open]'); return b && b.getAttribute('data-review-of'); });
    R.forms = forms > 5;
    await openAll(id);
    const blk = page.locator(`.fld.review[data-review-of="${id}"]`);
    await blk.locator('input[type=text]').fill('Test Person');
    await blk.locator('.rv-btn.accept').click(); await page.waitForTimeout(600);
    R.review_badge = (await blk.locator('.rv-now .badge').getAttribute('data-v')) === 'accepted';
    R.review_text = (await blk.locator('.rv-meta').textContent()).includes('Test Person');
    const opens = await page.evaluate(() => document.querySelector('[data-open-count]').textContent);
    R.open_count = +opens >= 0;
    // reject needs a comment
    const id2 = await page.evaluate(() => { const b = document.querySelector('.fld.review[data-state=open]'); return b && b.getAttribute('data-review-of'); });
    await openAll(id2);
    const b2 = page.locator(`.fld.review[data-review-of="${id2}"]`);
    await b2.locator('input[type=text]').fill('Test Person');
    await b2.locator('.rv-btn.reject').click();
    R.reject_needs_comment = (await b2.locator('.rv-err').textContent()).length > 0;
    await b2.locator('textarea').fill('Not right.'); await b2.locator('.rv-btn.reject').click(); await page.waitForTimeout(500);
    R.attention_updates = await page.evaluate(i => !!document.querySelector(`[data-att-dyn="rejected"] a.ref[data-ref="${i}"]`), id2);
  } else {
    R.no_forms = forms === 0;
    R.cli_shown = await page.locator('.rv-cli').count() > 0;
  }
  // 8. theme toggle and print
  await page.click('#theme'); R.theme = !!(await page.evaluate(() => document.documentElement.getAttribute('data-theme')));
  await page.emulateMedia({ media: 'print' });
  await page.evaluate(() => window.dispatchEvent(new Event('beforeprint')));
  R.print_open = await page.evaluate(() => [...document.querySelectorAll('details')].every(d => d.open));
  R.print_nav_hidden = !(await page.isVisible('#side'));
  R.errors = errors; R.external = external;
  console.log(JSON.stringify(R));
  await browser.close();
})();
