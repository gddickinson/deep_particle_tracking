// Method page: the pipeline from simulation to metrics. Network and loss facts come from /api/method, which reads
// models/network.py and models/losses.py with Python's ast module (names, docstrings, constructor defaults).
import { route, api, app, esc } from './core.js';

const defs = d => Object.entries(d).map(([k, v]) => `<span class="tag">${esc(k)} = ${esc(typeof v === 'object' ? JSON.stringify(v) : v)}</span>`).join('');

route('method', async () => {
  const [M, P] = await Promise.all([api('/method'), api('/params')]);
  const factory = Object.entries(M.create_model);
  const nets = M.networks.filter(n => Object.values(M.create_model).includes(n.name));
  const parts = M.networks.filter(n => !Object.values(M.create_model).includes(n.name));
  app().innerHTML = `
  <h1 class="glow">Method</h1>
  <p class="lead">Deep Particle Tracker learns to find particles from simulated movies whose ground truth is known,
  then links its detections across frames. Everything below is read from the code.</p>
  <div class="flow">
    <div class="card"><b>Simulate</b><span>Random particles, a motion model, optional blinking, a PSF image per frame,
      camera noise, clipping to 0–1, and per-pixel position / track-id targets.</span></div>
    <div class="card"><b>Train</b><span>A U-Net whose bottleneck is a convolutional LSTM sees a short sequence of frames
      and outputs a position probability map; ${esc(M.trainer_loss.split(',')[0])}.</span></div>
    <div class="card"><b>Detect</b><span>Threshold the probability map and keep local maxima within the NMS radius
      (<code>detect_peaks</code>).</span></div>
    <div class="card"><b>Link</b><span>Hungarian assignment between consecutive frames within a maximum distance; tracks
      shorter than 3 frames are dropped (<code>link_particles</code>).</span></div>
    <div class="card"><b>Score</b><span>Precision, recall, F1 per frame; MOTA, MOTP and identity switches over the movie
      (<code>MetricsCalculator</code>).</span></div>
  </div>

  <h2>Simulation</h2>
  <p>Coordinates are (y, x) in pixels with pixel centres at integers; the generator uses 0.1 µm pixels and 0.1 s frames.</p>
  <div class="three">
    <div class="panel"><h3>Motion</h3><p><b>Brownian</b>: Gaussian steps with SD √(2 D Δt) per axis, clamped to the image.
      <b>Directed</b>: constant speed (setting ± 20 %), heading turned by N(0, π/4) with the given probability, reflected at edges.
      <b>Confined</b>: Brownian steps plus a pull towards the start point growing as (distance / radius)².</p></div>
    <div class="panel"><h3>PSF</h3><p><b>Gaussian</b>: amplitude × exp(−r² / 2σ²), σ scaled by each particle's size (0.8–1.2).
      <b>Airy disk</b>: [2 J₁(v) / v]² with the first dark ring at the Airy radius. Amplitudes 0.5–1.</p></div>
    <div class="panel"><h3>Noise</h3><p>Background = brightest PSF pixel / SNR (the Background Level setting for empty frames).
      <b>Poisson</b>: shot noise at 100 photons per unit. <b>Gaussian</b>: additive, SD 10 / SNR.
      <b>Poisson + Gaussian</b>: shot noise plus read noise (counts). Blinking: a two-state on/off Markov chain.</p></div>
  </div>
  <p><a class="btn alt" href="#/simulate">Try it in the simulator</a></p>

  <h2>Networks</h2>
  <p><code>create_model(model_type)</code>: ${factory.map(([k, v]) => `<code>'${esc(k)}'</code> → ${esc(v)}`).join(', ')}.</p>
  <div class="grid wide">${nets.map(n => `<div class="card"><b>${esc(n.name)}</b><span>${esc(n.doc)}</span><p>${defs(n.defaults)}</p></div>`).join('')}</div>
  <p class="small mute">Building blocks: ${parts.map(n => `<code>${esc(n.name)}</code> (${esc(n.doc)})`).join('; ')}.</p>

  <h2>Losses</h2>
  <p>Training default: ${esc(M.trainer_loss)}.</p>
  <div class="scroll"><table><tr><th>class</th><th>what it does</th><th>defaults</th></tr>
    ${M.losses.map(l => `<tr><td><code>${esc(l.name)}</code></td><td>${esc(l.doc)}</td><td>${defs(l.defaults)}</td></tr>`).join('')}</table></div>

  <h2>What the web app adds</h2>
  <p>A classical baseline (<code>prediction/classical.py</code>) so the simulator and benchmark have something to score
  without trained weights: Gaussian matched filter (σ ${P.baseline.sigma} px), score in robust noise units
  (median and 1.4826 × MAD), then the same <code>detect_peaks</code> and <code>link_particles</code> the deep-model
  predictor uses. It is a reference point, not part of the repository's method.</p>`;
});
