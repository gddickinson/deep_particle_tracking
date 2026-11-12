// Benchmark page: classical-baseline scores over SNR and density, precomputed by the Python code (/api/benchmark),
// plus a cross-check that the browser re-run of the demo movie gives the Python numbers.
import { route, api, app, esc } from './core.js';
import { lineChart, SERIES } from './chart.js';
import { simulateMovie } from './sim_engine.js';
import { scoreRun } from './scoring.js';

const f3 = v => (Number.isFinite(v) ? v.toFixed(3) : '—');

route('benchmark', async () => {
  const [B, demo] = await Promise.all([api('/benchmark'), api('/demo')]);
  const noises = [...new Set(B.snr_sweep.rows.map(r => r.noise))];
  const bySnr = key => noises.map((n, i) => ({ name: n, color: SERIES[i], points: B.snr_sweep.rows.filter(r => r.noise === n).map(r => [r.snr, r[key]]) }));
  const D = B.density_sweep.rows, bp = B.baseline;
  app().innerHTML = `
  <h1 class="glow">Benchmark</h1>
  <p class="lead">How well a <b>classical</b> detector and tracker does on the repository's simulated movies, as the SNR
  setting and the particle density change. Computed by the Python code (<code>web/benchmark.py</code>) and exported
  with the site.</p>
  <div class="callout"><b>Not the deep model</b>: there are no trained weights in the repository. The baseline is a Gaussian matched filter
  (σ ${bp.sigma} px), a threshold of ${bp.threshold} robust noise units, the repo's local-maximum detection
  (radius ${bp.nms_radius} px) and its Hungarian frame-to-frame linker (≤ ${bp.max_distance} px, tracks ≥ ${bp.min_length}
  frames). Scores: the repo's <code>MetricsCalculator</code>, match distance ${B.match_px} px. With the local app and a
  checkpoint you trained, the <a href="#/model">Trained model</a> page scores it on the same movies.</div>
  <p class="small mute">Each point: mean of seeds ${B.seeds.join(', ')}; ${B.size} × ${B.size} px (${(B.size * 0.1).toFixed(1)} µm square),
  ${B.frames} frames, Brownian motion (D 0.1 µm²/s), Gaussian PSF σ 1 px, no blinking.</p>

  <h2>SNR</h2>
  <p>${B.snr_sweep.particles} particles per movie. The SNR setting is the Simulation tab's: the background is set to
  (brightest PSF pixel) / SNR, and for Gaussian noise the noise SD is 10 / SNR. What that means for the image depends on
  the noise model, so the measured peak SNR (spot amplitude over background SD on the frames) is plotted too.</p>
  <div class="two">
    <div><h3>Detection F1</h3>${lineChart({ series: bySnr('f1'), xlabel: 'SNR setting', ylabel: 'detection F1', xlog: true, ymin: 0, ymax: 1 })}</div>
    <div><h3>Tracking MOTA</h3>${lineChart({ series: bySnr('mota'), xlabel: 'SNR setting', ylabel: 'MOTA', xlog: true, ymin: Math.min(0, ...B.snr_sweep.rows.map(r => r.mota)), ymax: 1 })}</div>
  </div>
  <div class="two">
    <div><h3>Measured peak SNR</h3>${lineChart({ series: bySnr('measured_snr'), xlabel: 'SNR setting', ylabel: 'measured peak SNR', xlog: true, ylog: true })}</div>
    <div class="callout a" style="align-self:center">With Poisson noise (photon scaling 100) the SNR setting only changes
    the background, so even a setting of 1 gives a measured SNR near 10 and the baseline barely notices. With Gaussian
    noise (SD 10 / SNR on spots of peak 0.5–1) the image SNR is roughly the setting divided by 10, so settings below
    about 20 bury the particles. Compare the same setting across noise models with care.</div>
  </div>
  <details><summary class="mono small">Table: SNR sweep</summary><div class="scroll"><table>
    <tr><th>noise</th><th class="num">SNR setting</th><th class="num">measured SNR</th><th class="num">precision</th><th class="num">recall</th><th class="num">F1</th><th class="num">MOTA</th><th class="num">MOTP px</th><th class="num">ID sw.</th></tr>
    ${B.snr_sweep.rows.map(r => `<tr><td>${esc(r.noise)}</td><td class="num">${r.snr}</td><td class="num">${r.measured_snr.toFixed(1)}</td><td class="num">${f3(r.precision)}</td><td class="num">${f3(r.recall)}</td><td class="num">${f3(r.f1)}</td><td class="num">${f3(r.mota)}</td><td class="num">${f3(r.motp)}</td><td class="num">${r.id_switches}</td></tr>`).join('')}
  </table></div></details>

  <h2>Density</h2>
  <p>${esc(B.density_sweep.noise)} noise, SNR setting ${B.density_sweep.snr}. Crowding makes spots merge (missed
  detections) and makes nearest-neighbour linking swap identities.</p>
  <div class="two">
    <div>${lineChart({ series: [{ name: 'detection F1', color: SERIES[0], points: D.map(r => [r.per_100_um2, r.f1]) },
      { name: 'MOTA', color: SERIES[1], points: D.map(r => [r.per_100_um2, r.mota]) }], xlabel: 'particles per 100 µm²', ylabel: 'score', xlog: true, ymin: 0, ymax: 1 })}</div>
    <div class="scroll"><table><tr><th class="num">particles</th><th class="num">per 100 µm²</th><th class="num">F1</th><th class="num">MOTA</th><th class="num">ID switches</th></tr>
      ${D.map(r => `<tr><td class="num">${r.particles}</td><td class="num">${r.per_100_um2.toFixed(1)}</td><td class="num">${f3(r.f1)}</td><td class="num">${f3(r.mota)}</td><td class="num">${r.id_switches}</td></tr>`).join('')}</table></div>
  </div>

  <h2>Browser vs Python</h2>
  <p>The demo movie (seed ${demo.seed}) simulated and scored again in this browser by the JavaScript ports:</p>
  <div class="scroll" id="xcheck"><p class="mute">Computing…</p></div>`;

  const mv = simulateMovie(demo.seed, demo.params), run = scoreRun(mv), s = run.summary;
  let sum = 0; for (const f of mv.frames) for (const v of f) sum += v;
  const rows = [['sum of all pixel values', sum, demo.checksum], ['measured peak SNR', s.measured_snr, demo.measured_snr],
    ...['precision', 'recall', 'f1', 'mota', 'motp', 'id_switches', 'tracks'].map(k => [k, s[k], demo.metrics[k]])];
  const same = rows.every(([, a, b]) => Math.abs(a - b) <= 1e-6 * Math.max(1, Math.abs(b)));
  document.getElementById('xcheck').innerHTML = `<table><tr><th>quantity</th><th class="num">browser (JS)</th><th class="num">Python</th></tr>
    ${rows.map(([k, a, b]) => `<tr><td>${k}</td><td class="num">${Number.isInteger(b) ? a : a.toFixed(6)}</td><td class="num">${Number.isInteger(b) ? b : b.toFixed(6)}</td></tr>`).join('')}</table>
    <p class="${same ? 'ok' : 'err'}">${same ? 'Identical to the Python results.' : 'Differs from the Python results.'}</p>`;
});
