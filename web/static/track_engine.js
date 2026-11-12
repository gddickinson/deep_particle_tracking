// Detection, linking and metrics in JS: ports of prediction/classical.py (matched-filter score map), postprocess.py
// (detect_peaks, link_particles), metrics.py (MetricsCalculator) and the SciPy pieces they use (gaussian_filter,
// maximum_filter in 'reflect' mode, median, cdist, linear_sum_assignment). Checked against Python by
// track_engine.test.mjs. Images are flat Float64Arrays in row-major (y, x) order; positions are [y, x].

export const BASELINE_DEFAULTS = { sigma: 1.0, threshold: 6.0, nms_radius: 3, max_distance: 5.0, min_length: 3 };

// numpy add.reduce for float64: first element, plus the pairwise sum (8 accumulators) of the rest
function pairwise(a, lo, n) {
  if (n < 8) { let s = 0; for (let i = 0; i < n; i++) s += a[lo + i]; return s; }
  if (n <= 128) {
    const r = [];
    for (let j = 0; j < 8; j++) r.push(a[lo + j]);
    let i = 8;
    for (; i < n - (n % 8); i += 8) for (let j = 0; j < 8; j++) r[j] += a[lo + i + j];
    let s = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
    for (; i < n; i++) s += a[lo + i];
    return s;
  }
  let n2 = Math.floor(n / 2); n2 -= n2 % 8;
  return pairwise(a, lo, n2) + pairwise(a, lo + n2, n - n2);
}
export const npSum = a => (a.length ? a[0] + pairwise(a, 1, a.length - 1) : 0);
export const npMean = a => npSum(a) / a.length;

function reflect(j, n) { if (n === 1) return 0; const p = 2 * n; j = ((j % p) + p) % p; return j >= n ? p - 1 - j : j; }

export function gaussianKernel(sigma, truncate = 4.0) {
  const r = Math.trunc(truncate * sigma + 0.5), c = -0.5 / (sigma * sigma), w = new Float64Array(2 * r + 1);
  for (let i = -r; i <= r; i++) w[i + r] = Math.exp(c * (i * i));
  const s = npSum(w);
  for (let i = 0; i < w.length; i++) w[i] /= s;
  return w;
}

// scipy correlate1d with a symmetric kernel: centre term first, then (left + right) pairs from the outside in
function filterAxis(src, H, W, axis, w) {
  const out = new Float64Array(src.length), r = (w.length - 1) / 2, n = axis === 0 ? H : W, m = axis === 0 ? W : H;
  const line = new Float64Array(n + 2 * r);
  for (let a = 0; a < m; a++) {
    const at = j => (axis === 0 ? j * W + a : a * W + j);
    for (let j = -r; j < n + r; j++) line[j + r] = src[at(reflect(j, n))];
    for (let i = 0; i < n; i++) {
      const c = i + r; let s = line[c] * w[r];
      for (let j = r; j >= 1; j--) s += (line[c - j] + line[c + j]) * w[r - j];
      out[at(i)] = s;
    }
  }
  return out;
}
export function gaussianFilter(img, H, W, sigma) {
  const w = gaussianKernel(sigma);
  return filterAxis(filterAxis(img, H, W, 0, w), H, W, 1, w);
}

function maxAxis(src, H, W, axis, r) {
  const out = new Float64Array(src.length), n = axis === 0 ? H : W, m = axis === 0 ? W : H;
  for (let a = 0; a < m; a++) for (let i = 0; i < n; i++) {
    let mx = -Infinity;
    for (let j = i - r; j <= i + r; j++) { const k = reflect(j, n), v = src[axis === 0 ? k * W + a : a * W + k]; if (v > mx) mx = v; }
    out[axis === 0 ? i * W + a : a * W + i] = mx;
  }
  return out;
}
export const maximumFilter = (img, H, W, r) => maxAxis(maxAxis(img, H, W, 0, r), H, W, 1, r);

export function median(a) {
  const s = Float64Array.from(a).sort(), n = s.length, h = n >> 1;
  return n % 2 ? s[h] : (s[h - 1] + s[h]) / 2;
}

// classical.score_map: (G - median G) / (1.4826 MAD), G = gaussian_filter(frame, sigma)
export function scoreMap(frame, H, W, sigma = 1.0) {
  const g = gaussianFilter(frame, H, W, sigma), med = median(g);
  const mad = 1.4826 * median(g.map(v => Math.abs(v - med)));
  if (!(mad > 0)) return new Float64Array(g.length);
  return g.map(v => (v - med) / mad);
}

// postprocess.detect_peaks: above threshold and equal to the maximum within nms_radius (window 2r+1)
export function detectPeaks(map, H, W, threshold, nmsRadius) {
  const mx = maximumFilter(map, H, W, nmsRadius | 0), out = [];
  for (let i = 0; i < map.length; i++) if (map[i] > threshold && map[i] === mx[i]) out.push([Math.floor(i / W), i % W]);
  return out;
}

export const cdist = (A, B) => A.map(a => B.map(b => Math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)));

// scipy.optimize.linear_sum_assignment (rectangular_lsap.cpp: shortest augmenting path, Crouse 2016)
export function linearSumAssignment(cost) {
  let nr = cost.length, nc = nr ? cost[0].length : 0;
  if (!nr || !nc) return [[], []];
  const transpose = nc < nr;
  let C = cost;
  if (transpose) { C = Array.from({ length: nc }, (_, j) => cost.map(row => row[j])); [nr, nc] = [nc, nr]; }
  const u = new Float64Array(nr), v = new Float64Array(nc), spc = new Float64Array(nc), path = new Int32Array(nc).fill(-1);
  const col4row = new Int32Array(nr).fill(-1), row4col = new Int32Array(nc).fill(-1);
  const SR = new Uint8Array(nr), SC = new Uint8Array(nc), remaining = new Int32Array(nc);
  for (let cur = 0; cur < nr; cur++) {
    let minVal = 0, numRem = nc, sink = -1, i = cur;
    for (let it = 0; it < nc; it++) remaining[it] = nc - it - 1;
    SR.fill(0); SC.fill(0); spc.fill(Infinity);
    while (sink === -1) {
      let index = -1, lowest = Infinity;
      SR[i] = 1;
      for (let it = 0; it < numRem; it++) {
        const j = remaining[it], r = minVal + C[i][j] - u[i] - v[j];
        if (r < spc[j]) { path[j] = i; spc[j] = r; }
        if (spc[j] < lowest || (spc[j] === lowest && row4col[j] === -1)) { lowest = spc[j]; index = it; }
      }
      minVal = lowest;
      if (minVal === Infinity) throw new Error('cost matrix is infeasible');
      const j = remaining[index];
      if (row4col[j] === -1) sink = j; else i = row4col[j];
      SC[j] = 1; numRem -= 1; remaining[index] = remaining[numRem];
    }
    u[cur] += minVal;
    for (let k = 0; k < nr; k++) if (SR[k] && k !== cur) u[k] += minVal - spc[col4row[k]];
    for (let k = 0; k < nc; k++) if (SC[k]) v[k] -= minVal - spc[k];
    let j = sink;
    for (;;) { const k = path[j]; row4col[j] = k; const t = col4row[k]; col4row[k] = j; j = t; if (k === cur) break; }
  }
  if (!transpose) return [Array.from({ length: nr }, (_, k) => k), Array.from(col4row)];
  const order = Array.from({ length: nr }, (_, k) => k).sort((a, b) => col4row[a] - col4row[b]);
  return [order.map(k => col4row[k]), order];
}

const BIG = 1e6;
const capped = (D, d) => D.map(row => row.map(x => (x > d ? BIG : x)));

// postprocess.link_particles: Hungarian assignment between consecutive frames, no gap closing
export function linkParticles(positions, maxDistance, minLength = 3) {
  const T = positions.length, tracks = [];
  const newTrack = (t, p) => { const tr = Array.from({ length: T }, () => [NaN, NaN]); tr[t] = [p[0], p[1]]; tracks.push(tr); };
  for (const p of positions[0] || []) newTrack(0, p);
  for (let t = 1; t < T; t++) {
    const cur = positions[t];
    if (!cur.length) continue;
    const act = [], idx = [];
    tracks.forEach((tr, k) => { if (!Number.isNaN(tr[t - 1][0])) { act.push(tr[t - 1]); idx.push(k); } });
    if (!act.length) { for (const p of cur) newTrack(t, p); continue; }
    const D = capped(cdist(act, cur), maxDistance), [ri, ci] = linearSumAssignment(D), used = new Set();
    ri.forEach((r, k) => { const c = ci[k]; if (D[r][c] <= maxDistance) { tracks[idx[r]][t] = [cur[c][0], cur[c][1]]; used.add(c); } });
    cur.forEach((p, c) => { if (!used.has(c)) newTrack(t, p); });
  }
  return tracks.filter(tr => tr.filter(p => !Number.isNaN(p[0])).length >= minLength);
}

// MetricsCalculator.calculate_detection_metrics
export function detectionMetrics(pred, gt, d = 5.0) {
  if (!gt.length) return pred.length ? { precision: 0, recall: 1, f1: 0, tp: 0, fp: pred.length, fn: 0 } : { precision: 1, recall: 1, f1: 1, tp: 0, fp: 0, fn: 0 };
  if (!pred.length) return { precision: 1, recall: 0, f1: 0, tp: 0, fp: 0, fn: gt.length };
  const D = cdist(gt, pred), [gi, pi] = linearSumAssignment(capped(D, d));
  let tp = 0; gi.forEach((g, k) => { if (D[g][pi[k]] <= d) tp++; });
  const fp = pred.length - tp, fn = gt.length - tp;
  const precision = tp + fp > 0 ? tp / (tp + fp) : 1, recall = tp + fn > 0 ? tp / (tp + fn) : 1;
  const f1 = precision + recall > 0 ? 2 * precision * recall / (precision + recall) : 0;
  return { precision, recall, f1, tp, fp, fn };
}

// MetricsCalculator.calculate_tracking_metrics (MOTA, MOTP, ID switches with the last correspondence kept)
export function trackingMetrics(predTracks, gtTracks, d = 5.0) {
  if (!predTracks.length || !gtTracks.length) return { mota: 0, motp: 0, id_switches: 0, track_precision: 0, track_recall: 0, track_f1: 0 };
  const T = Math.max(...gtTracks.map(t => t.length), ...predTracks.map(t => t.length));
  let totGt = 0, tp = 0, fp = 0, fn = 0, sw = 0, dist = 0;
  const prev = new Map();
  const at = (tracks, t) => { const P = [], I = []; tracks.forEach((tr, k) => { if (t < tr.length && !Number.isNaN(tr[t][0])) { P.push(tr[t]); I.push(k); } }); return [P, I]; };
  for (let t = 0; t < T; t++) {
    const [G, gid] = at(gtTracks, t), [P, pid] = at(predTracks, t);
    totGt += G.length;
    if (!G.length || !P.length) { fn += G.length; fp += P.length; continue; }
    const D = cdist(G, P), [gi, pi] = linearSumAssignment(capped(D, d)), curr = new Map();
    let matched = 0;
    gi.forEach((g, k) => {
      const p = pi[k];
      if (D[g][p] <= d) {
        const a = gid[g], b = pid[p];
        curr.set(a, b);
        if (prev.has(a) && prev.get(a) !== b) sw++;
        dist += D[g][p]; tp++; matched++;
      }
    });
    fp += P.length - matched; fn += G.length - matched;
    for (const [a, b] of curr) prev.set(a, b);
  }
  const mota = totGt > 0 ? 1 - (fp + fn + sw) / totGt : 0, motp = tp > 0 ? dist / tp : 0;
  const pr = tp + fp > 0 ? tp / (tp + fp) : 0, rc = tp + fn > 0 ? tp / (tp + fn) : 0, f1 = pr + rc > 0 ? 2 * pr * rc / (pr + rc) : 0;
  return { mota, motp, id_switches: sw, track_precision: pr, track_recall: rc, track_f1: f1, tp, fp, fn };
}

// MetricsCalculator.calculate_all_metrics
export function allMetrics(predPos, gtPos, predTracks, gtTracks, d = 5.0) {
  const per = predPos.map((p, t) => detectionMetrics(p, gtPos[t], d));
  const sum = k => per.reduce((s, m) => s + m[k], 0);
  const out = { detection: { precision: npMean(per.map(m => m.precision)), recall: npMean(per.map(m => m.recall)),
    f1: npMean(per.map(m => m.f1)), tp: sum('tp'), fp: sum('fp'), fn: sum('fn') }, per_frame: per };
  if (predTracks && gtTracks) out.tracking = trackingMetrics(predTracks, gtTracks, d);
  return out;
}

// classical.run_baseline: score map -> detect_peaks -> link_particles
export function runBaseline(frames, H, W, params = {}) {
  const p = { ...BASELINE_DEFAULTS, ...params };
  const positions = frames.map(f => detectPeaks(scoreMap(f, H, W, p.sigma), H, W, p.threshold, p.nms_radius));
  const tracks = frames.length > 1 ? linkParticles(positions, p.max_distance, p.min_length) : [];
  return { positions, tracks, params: p };
}
