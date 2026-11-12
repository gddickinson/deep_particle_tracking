// Small SVG line chart (no library): one y axis, optional log x / log y, 2px lines, 8px markers with a surface ring,
// a legend for >= 2 series plus direct labels at the line ends, and hover tooltips (<title> on enlarged hit targets).
// Series colours are a categorical set validated for the dark surface (dataviz validate_palette.js: all checks pass).
import { esc } from './core.js';

export const SERIES = ['#1fa784', '#e0661c', '#6f7fe6'];

const ticksLin = (lo, hi, n = 5) => Array.from({ length: n + 1 }, (_, i) => lo + (hi - lo) * i / n);
const ticksLog = (lo, hi) => { const t = []; for (let e = Math.floor(Math.log10(lo)); e <= Math.ceil(Math.log10(hi)); e++) for (const m of [1, 2, 5]) { const v = m * 10 ** e; if (v >= lo * 0.999 && v <= hi * 1.001) t.push(v); } return t; };
const fmt = v => (Math.abs(v) >= 100 || Number.isInteger(v) ? String(Math.round(v * 100) / 100) : v.toFixed(Math.abs(v) < 1 ? 2 : 1));

export function lineChart({ series, xlabel, ylabel, xlog = false, ylog = false, ymin, ymax, xticks, w = 560, h = 300, label }) {
  const pts = series.flatMap(s => s.points).filter(([x, y]) => Number.isFinite(x) && Number.isFinite(y) && (!ylog || y > 0));
  const xs = pts.map(p => p[0]), ys = pts.map(p => p[1]);
  const x0 = Math.min(...xs), x1 = Math.max(...xs);
  const y0 = ymin ?? Math.min(...ys), y1 = ymax ?? Math.max(...ys);
  const L = 52, R = 150, T = 14, B = 44, pw = w - L - R, ph = h - T - B;
  const fx = xlog ? v => L + pw * (Math.log(v) - Math.log(x0)) / (Math.log(x1) - Math.log(x0)) : v => L + pw * (v - x0) / ((x1 - x0) || 1);
  const fy = ylog ? v => T + ph * (1 - (Math.log(v) - Math.log(y0)) / (Math.log(y1) - Math.log(y0))) : v => T + ph * (1 - (v - y0) / ((y1 - y0) || 1));
  const xt = xticks || (xlog ? ticksLog(x0, x1) : ticksLin(x0, x1)), yt = ylog ? ticksLog(y0, y1) : ticksLin(y0, y1, 4);
  let s = `<svg viewBox="0 0 ${w} ${h}" role="img" aria-label="${esc(label || ylabel)}">`;
  s += yt.map(v => `<line class="ax" x1="${L}" x2="${L + pw}" y1="${fy(v)}" y2="${fy(v)}" stroke-opacity=".5"/><text x="${L - 6}" y="${fy(v) + 4}" text-anchor="end">${fmt(v)}</text>`).join('');
  s += xt.map(v => `<text x="${fx(v)}" y="${T + ph + 16}" text-anchor="middle">${fmt(v)}</text>`).join('');
  s += `<line class="ax" x1="${L}" x2="${L + pw}" y1="${T + ph}" y2="${T + ph}"/>`;
  s += `<text class="lbl" x="${L + pw / 2}" y="${h - 6}" text-anchor="middle">${esc(xlabel)}</text>`;
  s += `<text class="lbl" transform="translate(13 ${T + ph / 2}) rotate(-90)" text-anchor="middle">${esc(ylabel)}</text>`;
  const ends = [];
  series.forEach(se => {
    const P = se.points.filter(([x, y]) => Number.isFinite(y) && (!ylog || y > 0));
    if (!P.length) return;
    s += `<polyline fill="none" stroke="${se.color}" stroke-width="2" stroke-linejoin="round" points="${P.map(([x, y]) => `${fx(x)},${fy(y)}`).join(' ')}"/>`;
    s += P.map(([x, y]) => `<circle cx="${fx(x)}" cy="${fy(y)}" r="4" fill="${se.color}" stroke="#0c1020" stroke-width="2"/>`).join('');
    s += P.map(([x, y]) => `<circle cx="${fx(x)}" cy="${fy(y)}" r="11" fill="transparent"><title>${esc(se.name)}: ${esc(xlabel)} ${fmt(x)}, ${esc(ylabel)} ${y.toFixed(3)}</title></circle>`).join('');
    const [lx, ly] = P[P.length - 1];
    ends.push({ y: fy(ly), x: fx(lx), name: se.name, color: se.color });
  });
  ends.sort((a, b) => a.y - b.y);
  for (let i = 1; i < ends.length; i++) if (ends[i].y - ends[i - 1].y < 13) ends[i].y = ends[i - 1].y + 13;
  s += ends.map(e => `<line x1="${e.x + 6}" x2="${L + pw + 10}" y1="${e.y}" y2="${e.y}" stroke="${e.color}" stroke-width="1" stroke-opacity=".6"/><text class="lbl" x="${L + pw + 14}" y="${e.y + 4}">${esc(e.name)}</text>`).join('');
  s += '</svg>';
  const legend = series.length > 1 ? `<div class="legend">${series.map(se => `<span><i style="background:${se.color}"></i>${esc(se.name)}</span>`).join('')}</div>` : '';
  return `<div class="chart">${legend}${s}</div>`;
}
