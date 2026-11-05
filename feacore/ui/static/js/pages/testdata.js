// Test data: placeholder page (filled in by the dashboard consolidation).
import { h } from '../api.js';

export async function render(el) {
  el.append(h('div', { class: 'page-head' }, h('div', {},
    h('h1', {}, 'Test data'),
    h('p', { class: 'lede' }, 'Physical test data (machine exports) for the evidence-driven loop and the nonlinear comparisons.'))));
}
