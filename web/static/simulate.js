// Simulator page: the repo's synthetic movie generator (sim_engine.js, a port of simulator/movie.py) with the
// classical baseline (track_engine.js) run on every movie, live scores against the ground truth, and playback.
import { route, api, app, esc, isStatic, slider, animate, download } from './core.js';
import { simulateMovie, PIXEL_UM } from './sim_engine.js';
import { scoreRun } from './scoring.js';
import { drawFrame } from './sim_view.js';
import { buildControls, NUMERIC } from './sim_controls.js';

const WEB_START = { size: 128, num_frames: 30, num_particles: 20 };
const PRESETS = {
  'Simulation tab defaults': {},
  'Dense': { num_particles: 150 },
  'Low SNR': { snr: 2 },
  'Blinking': { blinking: true },
  'Directed transport': { motion: 'Directed', velocity: 5 },
  'Airy disk PSF': { psf: 'Airy Disk', size: 96 },
};
const f3 = v => (Number.isFinite(v) ? v.toFixed(3) : '—');

function fromQuery(q, P) {
  const p = { ...P.defaults, ...WEB_START, seed: 1 };
  for (const [k, v] of Object.entries(q)) {
    if (k === 'seed' || k in NUMERIC) p[k] = +v;
    else if (['motion', 'psf', 'noise'].includes(k)) p[k] = v;
    else if (k === 'blinking') p[k] = v === '1' || v === 'true';
  }
  return p;
}
const simParams = p => { const { size, seed, ...rest } = p; return { ...rest, width: size, height: size }; };

route('simulate', async (args, query) => {
  const P = await api('/params');
  let p = fromQuery(query, P), base = { ...P.baseline }, movie = null, run = null, t = 0, timer = 0;
  let playing = !matchMedia('(prefers-reduced-motion: reduce)').matches;
  const ov = { gt: true, det: true, tracks: true };
  app().innerHTML = `
  <h1 class="glow">Simulator</h1>
  <p class="lead">The Simulation tab's pipeline (<code>simulator/movie.py</code>) ported to JavaScript: with the same seed it
  makes the same movie as <code>simulate_movie(seed=…)</code> in Python. A classical detector and tracker
  (<span class="badge">baseline, not the deep model</span>) runs on every movie and is scored against the ground truth.</p>
  <div class="row" style="max-width:640px"><div><label for="preset">Preset</label><select id="preset">${Object.keys(PRESETS).map(k => `<option>${esc(k)}</option>`).join('')}</select></div>
    <div><label for="seed">Seed</label><input id="seed" type="number" min="0" max="4294967295" value="${p.seed}"></div></div>
  <div class="sim">
    <div class="controls" id="ctl"></div>
    <div class="view">
      <canvas id="cv" role="img" aria-label="Simulated microscopy frame"></canvas>
      <div class="toolbar">
        <button type="button" id="play" class="alt">${playing ? 'Pause' : 'Play'}</button>
        <label for="fr" class="inl">Frame <b id="frn">0</b></label><input id="fr" type="range" min="0" value="0">
      </div>
      <div class="toolbar">
        <label class="inl"><input type="checkbox" id="o-gt" checked><span class="key" style="background:#5cf2c8"></span>ground truth</label>
        <label class="inl"><input type="checkbox" id="o-det" checked><span class="key" style="background:#ff8a3d"></span>baseline detections</label>
        <label class="inl"><input type="checkbox" id="o-tracks" checked><span class="key" style="background:#8fa2ff"></span>baseline tracks</label>
      </div>
      <div class="stats"><div class="readout" id="ro"></div><div class="readout" id="bl"></div></div>
      <div id="tune"></div>
      <div class="toolbar"><button type="button" class="alt" id="dl-gt">Ground truth CSV</button><button type="button" class="alt" id="dl-det">Detections CSV</button>
        ${isStatic() ? '' : '<button type="button" id="py">Check with Python</button>'}</div>
      <div id="pyout"></div>
      <h3>Reproduce in Python</h3><pre id="code"></pre>
    </div>
  </div>`;
  const cv = document.getElementById('cv'), fr = document.getElementById('fr');
  const draw = () => { if (movie) { drawFrame(cv, movie, t, { ...ov, result: run.result }); fr.value = t; document.getElementById('frn').textContent = t; } };

  const tune = document.getElementById('tune');
  tune.innerHTML = '<h3>Baseline settings</h3>';
  const tg = document.createElement('div'); tg.className = 'two'; tune.appendChild(tg);
  slider(tg, { id: 'thr', label: 'Detection threshold (robust noise units)', min: 2, max: 20, step: 0.5, value: base.threshold }, v => { base.threshold = v; rescore(); });
  slider(tg, { id: 'link', label: 'Link distance', min: 1, max: 20, step: 0.5, value: base.max_distance, unit: ' px' }, v => { base.max_distance = v; rescore(); });

  function readouts() {
    const s = run.summary, n = p.num_particles, area = (p.size * PIXEL_UM) ** 2;
    document.getElementById('ro').innerHTML = `<h3>Movie</h3>
      <div>density<b>${(n / area * 100).toFixed(1)} per 100 µm²</b></div>
      <div>field of view<b>${(p.size * PIXEL_UM).toFixed(1)} × ${(p.size * PIXEL_UM).toFixed(1)} µm</b></div>
      <div>SNR setting<b>${p.snr}</b></div>
      <div title="peak signal 1.0 over the background noise SD measured on the unclipped frames">measured SNR<b>${f3(s.measured_snr)}</b></div>
      <div>visible in this frame<b>${movie.gt[t]?.length ?? 0} of ${n}</b></div>`;
    document.getElementById('bl').innerHTML = `<h3>Baseline vs ground truth (≤ 5 px)</h3>
      <div>detection precision<b>${f3(s.precision)}</b></div><div>detection recall<b>${f3(s.recall)}</b></div>
      <div>detection F1<b>${f3(s.f1)}</b></div><div>MOTA<b>${f3(s.mota)}</b></div>
      <div>MOTP<b>${f3(s.motp)} px</b></div><div>ID switches<b>${s.id_switches}</b></div><div>tracks (≥ 3 frames)<b>${s.tracks}</b></div>`;
  }
  function code() {
    const sp = simParams(p), keys = Object.keys(sp).filter(k => sp[k] !== P.defaults[k]);
    const args = keys.map(k => `${k}=${typeof sp[k] === 'string' ? `'${sp[k]}'` : sp[k] === true ? 'True' : sp[k] === false ? 'False' : sp[k]}`);
    document.getElementById('code').textContent = `from simulator.movie import simulate_movie\nfrom prediction.classical import run_baseline\n` +
      `movie = simulate_movie(seed=${p.seed}${args.length ? ', ' + args.join(', ') : ''})\n` +
      `baseline = run_baseline(movie['frames'], threshold=${base.threshold}, max_distance=${base.max_distance})`;
  }
  function rescore() { if (!movie) return; run = scoreRun(movie, base); readouts(); code(); draw(); }
  function regenerate() {
    clearTimeout(timer);
    timer = setTimeout(() => {
      movie = simulateMovie(p.seed >>> 0, simParams(p));
      fr.max = movie.frames.length - 1; t = Math.min(t, movie.frames.length - 1);
      document.getElementById('pyout').innerHTML = '';
      rescore();
    }, 120);
  }
  const ctl = document.getElementById('ctl');
  const rebuild = () => buildControls(ctl, p, P, (k, v) => { p[k] = v; regenerate(); });
  rebuild();
  document.getElementById('preset').addEventListener('change', e => {
    p = { ...P.defaults, ...WEB_START, seed: p.seed, ...PRESETS[e.target.value] }; rebuild(); regenerate();
  });
  document.getElementById('seed').addEventListener('change', e => { p.seed = Math.max(0, Math.floor(+e.target.value || 0)); regenerate(); });
  fr.addEventListener('input', () => { t = +fr.value; playing = false; document.getElementById('play').textContent = 'Play'; draw(); readouts(); });
  document.getElementById('play').addEventListener('click', e => { playing = !playing; e.target.textContent = playing ? 'Pause' : 'Play'; });
  for (const k of Object.keys(ov)) document.getElementById('o-' + k).addEventListener('change', e => { ov[k] = e.target.checked; draw(); });
  const csv = (name, head, rows) => download(name, head + '\n' + rows.map(r => r.join(',')).join('\n') + '\n', 'text/csv');
  document.getElementById('dl-gt').addEventListener('click', () => csv(`ground_truth_seed${p.seed}.csv`, 'frame,particle_id,y_px,x_px',
    movie.gt.flatMap((g, k) => g.map(([id, y, x]) => [k, id, y, x]))));
  document.getElementById('dl-det').addEventListener('click', () => csv(`baseline_detections_seed${p.seed}.csv`, 'frame,y_px,x_px',
    run.result.positions.flatMap((g, k) => g.map(([y, x]) => [k, y, x]))));
  document.getElementById('py')?.addEventListener('click', async e => {
    e.target.disabled = true;
    try {
      const r = await api('/score', { body: { seed: p.seed, sim: simParams(p), baseline: base } }), s = run.summary;
      const rows = ['precision', 'recall', 'f1', 'mota', 'motp', 'id_switches', 'tracks', 'measured_snr'];
      document.getElementById('pyout').innerHTML = `<div class="scroll"><table><tr><th>score</th><th class="num">browser (JS)</th><th class="num">Python</th></tr>${rows.map(k =>
        `<tr><td>${k}</td><td class="num">${f3(s[k])}</td><td class="num">${f3(r[k])}</td></tr>`).join('')}</table></div>`;
    } catch (err) { document.getElementById('pyout').innerHTML = `<p class="err">${esc(err.message)}</p>`; }
    e.target.disabled = false;
  });

  regenerate();
  let last = 0;
  animate(now => {
    if (!playing || !movie || now - last < 120) return;
    last = now; t = (t + 1) % movie.frames.length; draw();
    document.getElementById('ro') && readouts();
  });
});
