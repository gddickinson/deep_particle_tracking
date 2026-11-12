// Node test: the JS engines reproduce the Python code on tests/fixtures/engine_fixture.json
// (written by scripts/make_engine_fixture.py). Run: node web/static/engine.test.mjs
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { simulateMovie, besselJ1, gtTracks, measuredSnr } from './sim_engine.js';
import { runBaseline, scoreMap, linearSumAssignment, allMetrics, detectionMetrics, trackingMetrics } from './track_engine.js';

const fx = JSON.parse(readFileSync(fileURLToPath(new URL('../../tests/fixtures/engine_fixture.json', import.meta.url)), 'utf8'));
let checks = 0;
const fail = msg => { console.error('FAIL', msg); process.exit(1); };
const close = (a, b, what, rtol = 1e-9, atol = 1e-12) => {
  checks++;
  if (a === null || b === null || Number.isNaN(a) || Number.isNaN(b)) { if (!((a === null || Number.isNaN(a)) && (b === null || Number.isNaN(b)))) fail(`${what}: ${a} vs ${b}`); return; }
  if (Math.abs(a - b) > atol + rtol * Math.abs(b)) fail(`${what}: ${a} vs ${b}`);
};
const eq = (a, b, what) => { checks++; if (JSON.stringify(a) !== JSON.stringify(b)) fail(`${what}: ${JSON.stringify(a).slice(0, 200)} vs ${JSON.stringify(b).slice(0, 200)}`); };

// Bessel J1 against scipy.special.j1
for (const [x, y] of fx.j1) close(besselJ1(x), y, `j1(${x})`, 1e-9, 1e-10);

// Movies: same seed -> same frames (float32 steps kept) and ground truth
for (const m of fx.movies) {
  const mv = simulateMovie(m.seed, m.params);
  eq(mv.frames.length, m.frames.length, `movie ${m.seed} frame count`);
  m.frames.forEach((f, t) => f.forEach((v, i) => close(mv.frames[t][i], v, `movie ${m.seed} frame ${t} px ${i}`, 1e-6, 1e-7)));
  m.gt.forEach((g, t) => { eq(mv.gt[t].length, g.length, `movie ${m.seed} gt ${t} count`);
    g.forEach((row, k) => row.forEach((v, j) => close(mv.gt[t][k][j], v, `movie ${m.seed} gt ${t}`, 1e-12, 1e-12))); });
}

// Windowed Gaussian spots give the same movie as whole-frame sums (bit for bit after noise)
for (const [seed, prm] of [[11, { width: 80, height: 70, num_frames: 4, num_particles: 15, psf_sigma: 0.6, snr: 50 }],
  [12, { width: 64, height: 64, num_frames: 3, num_particles: 5, noise: 'Poisson + Gaussian', psf_sigma: 2.5, snr: 1 }],
  [13, { width: 64, height: 64, num_frames: 3, num_particles: 5, noise: 'Gaussian', snr: 50 }],
  [14, { width: 80, height: 80, num_frames: 3, num_particles: 9, noise: 'Poisson + Gaussian', snr: 50, psf_sigma: 0.6 }]]) {
  const a = simulateMovie(seed, prm), b = simulateMovie(seed, prm, { full: true });
  a.frames.forEach((f, t) => { checks++; if (!f.every((v, i) => v === b.frames[t][i])) fail(`windowed PSF differs (seed ${seed} frame ${t})`); });
}

// One SNR definition: measured SNR (peak 1 over unclipped background SD) matches the setting
for (const noise of ['Poisson', 'Gaussian', 'Poisson + Gaussian']) for (const snr of [1, 3, 10]) {
  const m = measuredSnr(simulateMovie(3, { width: 128, height: 128, num_frames: 3, num_particles: 0, noise, snr }));
  close(m.snr, snr, `measured SNR ${noise} ${snr}`, 0.03, 0);
}

// Baseline detector + linker + metrics on Python frames, and end-to-end on the JS movie
for (const b of fx.baseline) {
  const mv = simulateMovie(b.seed, b.params), H = mv.H, W = mv.W;
  const s0 = scoreMap(mv.frames[0], H, W, 1.0);
  b.score0.forEach((v, i) => close(s0[i], v, `score map ${b.seed} px ${i}`, 1e-6, 1e-6));
  const r = runBaseline(mv.frames, H, W, b.baseline_params);
  eq(r.positions, b.positions, `baseline ${b.seed} detections`);
  eq(r.tracks.length, b.tracks.length, `baseline ${b.seed} track count`);
  r.tracks.forEach((tr, k) => tr.forEach((p, t) => { close(p[0], b.tracks[k][t][0], 'track y'); close(p[1], b.tracks[k][t][1], 'track x'); }));
  const gtPos = mv.gt.map(g => g.map(([, y, x]) => [y, x]));
  const m = allMetrics(r.positions, gtPos, r.tracks, gtTracks(mv), 5.0);
  for (const k of Object.keys(b.metrics.detection)) close(m.detection[k], b.metrics.detection[k], `detection ${k}`);
  for (const k of Object.keys(b.metrics.tracking)) close(m.tracking[k], b.metrics.tracking[k], `tracking ${k}`);
  b.metrics.per_frame.forEach((pf, t) => { for (const k of Object.keys(pf)) close(m.per_frame[t][k], pf[k], `per-frame ${t} ${k}`); });
}

// linear_sum_assignment
for (const c of fx.lsap) { const [r, k] = linearSumAssignment(c.cost); eq(r, c.rows, 'lsap rows'); eq(k, c.cols, 'lsap cols'); }

// Metric regressions (same cases as tests/test_metrics.py)
const d = detectionMetrics([[7, 4], [5, 11], [3, 7]], [[6, 11], [11, 4], [7, 4]], 5.0);
eq([d.tp, d.fp, d.fn], [3, 0, 0], 'detection matching maximises TP');
const N = [NaN, NaN], far = [[20, 20], [20, 21], [20, 22], [20, 23]];
const tm = trackingMetrics([[[5, 5], [5, 6], N, N], [N, N, N, [5, 7]], far], [[[5, 5], [5, 6], N, [5, 7]], far], 2.0);
eq([tm.id_switches, tm.tp, tm.fp, tm.fn], [1, 7, 0, 0], 'ID switch across a gap');

console.log(`ok (${checks} checks)`);
