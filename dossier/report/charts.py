"""Small charts of the report, as inline SVG made at build time (no script, so they print and work offline).

convergence()   one result over the mesh levels: a line with a dot per level, the discretization uncertainty of
                the finest level as a whisker, the extrapolated value as a reference line, the last value labelled
segments()      one whole split into parts (verdicts, review states): a thin bar; the words are next to it

The marks take their colours from the page's tokens through CSS classes (cv-*), so both themes work.
"""

from __future__ import annotations

import math
from typing import Any

from dossier.report import fmt
from dossier.report.html import H, esc, tag

W, HGT = 232, 104
ML, MR, MT, MB = 40, 46, 10, 20


def unit_scale(value: float | None, unit: str) -> tuple[float, str]:
    """The factor from SI to the unit people read for a value, and the label of that unit."""
    if value is None or value == 0 or not math.isfinite(value):
        return 1.0, unit
    shown = fmt.quantity(value, unit, 9)
    num, _, label = shown.partition(" ")
    try:
        return float(num) / value, label
    except (ValueError, ZeroDivisionError):
        return 1.0, unit


def _ticks(lo: float, hi: float, n: int = 3) -> list[float]:
    span = hi - lo
    if span <= 0 or not math.isfinite(span):
        return [lo]
    raw = span / n
    mag = 10 ** math.floor(math.log10(raw))
    step = min((m * mag for m in (1, 2, 2.5, 5, 10)), key=lambda s: abs(s - raw))
    first = math.ceil(lo / step - 1e-9) * step
    out, t = [], first
    while t <= hi + 1e-9 * abs(hi) + 1e-12:
        out.append(round(t, 12))
        t += step
    return out or [lo]


def _tick_text(v: float, step: float) -> str:
    if step <= 0:
        return fmt.plain(v, 3)
    dec = max(0, -math.floor(math.log10(step) + 1e-9))
    if step * 10 ** dec % 1 > 1e-6:      # a step like 2.5
        dec += 1
    s = f"{v:.{min(dec, 6)}f}"
    return s


def convergence(values: list[float], unit: str, uncertainty: float | None = None, extrapolated: float | None = None,
                nodes: list[int] | None = None, label: str = "") -> H:
    """The value of one result on each mesh level, coarse to fine."""
    vals = [v for v in values if v is not None and math.isfinite(v)]
    if len(vals) < 2:
        return H("")
    k, ulabel = unit_scale(vals[-1], unit)
    ys = [v * k for v in vals]
    last = ys[-1]
    lo, hi = min(ys), max(ys)
    band = abs(last) * uncertainty if uncertainty else 0.0
    lo, hi = min(lo, last - band), max(hi, last + band)
    if extrapolated is not None and math.isfinite(extrapolated):
        lo, hi = min(lo, extrapolated * k), max(hi, extrapolated * k)
    if hi - lo < 1e-12 * max(abs(hi), 1.0):
        lo, hi = lo - abs(lo) * 0.01 - 1e-12, hi + abs(hi) * 0.01 + 1e-12
    pad = (hi - lo) * 0.14
    lo, hi = lo - pad, hi + pad
    pw, ph = W - ML - MR, HGT - MT - MB
    n = len(ys)

    def x(i: int) -> float:
        return ML + (pw * i / (n - 1))

    def y(v: float) -> float:
        return MT + ph * (1 - (v - lo) / (hi - lo))

    ticks = _ticks(lo, hi)
    step = ticks[1] - ticks[0] if len(ticks) > 1 else 0.0
    out = []
    for t in ticks:
        out.append(f'<line class="cv-grid" x1="{ML}" x2="{W - MR}" y1="{y(t):.1f}" y2="{y(t):.1f}"/>')
        out.append(f'<text class="cv-txt" x="{ML - 5}" y="{y(t) + 3:.1f}" text-anchor="end">{esc(_tick_text(t, step))}</text>')
    for i in range(n):
        out.append(f'<text class="cv-txt" x="{x(i):.1f}" y="{HGT - 6}" text-anchor="middle">L{i + 1 + (0 if nodes is None else 0)}</text>')
    if extrapolated is not None and math.isfinite(extrapolated):
        ye = y(extrapolated * k)
        out.append(f'<line class="cv-ext" x1="{ML}" x2="{W - MR}" y1="{ye:.1f}" y2="{ye:.1f}"><title>Extrapolated value: '
                   f'{esc(fmt.quantity(extrapolated, unit))}</title></line>')
    if band:
        xl = x(n - 1)
        out.append(f'<path class="cv-unc" d="M{xl:.1f} {y(last - band):.1f}V{y(last + band):.1f}M{xl - 4:.1f} {y(last - band):.1f}h8'
                   f'M{xl - 4:.1f} {y(last + band):.1f}h8"><title>Discretization uncertainty: {fmt.percent(uncertainty)}</title></path>')
    pts = " ".join(f"{x(i):.1f},{y(v):.1f}" for i, v in enumerate(ys))
    out.append(f'<polyline class="cv-line" points="{pts}"/>')
    for i, v in enumerate(ys):
        tip = f"Mesh level {i + 1}: {fmt.quantity(vals[i], unit)}" + (f", {nodes[i]:,} nodes".replace(",", " ") if nodes and i < len(nodes) else "")
        out.append(f'<circle class="cv-dot{" cv-last" if i == n - 1 else ""}" cx="{x(i):.1f}" cy="{y(v):.1f}" r="{4.2 if i == n - 1 else 3.4}">'
                   f"<title>{esc(tip)}</title></circle>")
    sig = fmt.sig_for(vals[-1], uncertainty)
    end = fmt.quantity(vals[-1], unit, sig).partition(" ")[0]
    out.append(f'<text class="cv-end" x="{x(n - 1) + 8:.1f}" y="{y(last) + 3.5:.1f}">{esc(end)}</text>')
    series = ", ".join(fmt.quantity(v, unit) for v in vals)
    aria = f"{label + ': ' if label else ''}values on the mesh levels 1 to {n}: {series}"
    return H(f'<svg class="cv" viewBox="0 0 {W} {HGT}" role="img" aria-label="{esc(aria)}" focusable="false">'
             f'<text class="cv-unit" x="{ML - 5}" y="{MT - 1}" text-anchor="end">{esc(ulabel)}</text>' + "".join(out) + "</svg>")


def segments(parts: list[tuple[str, int, str]], label: str = "") -> H:
    """A thin bar of one whole split into parts: (tone, count, word). The words and counts are beside the bar."""
    total = sum(n for _, n, _ in parts)
    aria = ", ".join(f"{n} {w}" for _, n, w in parts if n) or "no items"
    segs = [tag("span", cls=f"seg t-{tone}", style=f"flex:{n} 1 0", title=f"{n} {word}") for tone, n, word in parts if n]
    return tag("span", segs if total else tag("span", cls="seg t-none", style="flex:1 1 0"), cls="segbar", role="img",
               aria_label=f"{label + ': ' if label else ''}{aria}")
