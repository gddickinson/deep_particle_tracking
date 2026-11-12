// Home page: hero with the demo movie playing (simulated in the browser from the same seed as the Python demo),
// counts from the API and entry cards.
import { route, api, app, esc, animate } from './core.js';
import { simulateMovie } from './sim_engine.js';
import { runBaseline } from './track_engine.js';
import { drawFrame } from './sim_view.js';

const DOORS = [
  ['#/simulate', 'Simulator', 'The repository\'s synthetic movie generator, ported to JavaScript: change SNR, density, motion, PSF, noise and blinking and watch a classical tracker cope.'],
  ['#/benchmark', 'Benchmark', 'Detection F1 and tracking MOTA of the classical baseline over SNR and particle density, computed by the Python code.'],
  ['#/method', 'Method', 'Simulation, the U-Net + ConvLSTM networks, the training losses, peak detection, Hungarian linking and the metrics.'],
  ['#/model', 'Trained model', 'Local app only: run a checkpoint you trained with the desktop app against the baseline on the same movie.'],
  ['#/about', 'About', 'What the web version does and leaves out, units and conventions, how to run the desktop and web apps, credits.'],
];

route('home', async () => {
  const [info, demo] = await Promise.all([api('/info'), api('/demo')]);
  app().innerHTML = `
  <section class="hero two">
    <div>
      <p class="kicker">Single-particle tracking · fluorescence microscopy</p>
      <h1 class="glow">Deep Particle Tracker</h1>
      <p class="lead">${esc(info.description)} It simulates fluorescent particles (motion, point spread function,
      camera noise, blinking), trains U-Net / ConvLSTM networks on them and tracks particles with the trained model.
      Here the simulator runs in your browser.</p>
      <p class="small"><span class="badge">no trained weights in the repo</span> &nbsp;Results on this site come from a
      classical baseline detector and tracker, labelled as such; trained models run only in the local app.</p>
      <a class="btn" href="#/simulate">Open the simulator</a><a class="btn alt" href="#/benchmark">See the benchmark</a>
    </div>
    <figure class="view" style="margin:0"><canvas id="hero" role="img" aria-label="Simulated fluorescence movie with ground-truth particles and baseline tracks"></canvas>
      <figcaption class="small mute" id="hcap">Simulating…</figcaption></figure>
  </section>
  <div class="tiles">
    <div class="tile"><b>${info.motions}</b><span>motion models</span></div>
    <div class="tile"><b>${info.psfs}</b><span>PSF models</span></div>
    <div class="tile"><b>${info.noises}</b><span>noise models</span></div>
    <div class="tile"><b>${info.networks}</b><span>network architectures</span></div>
    <div class="tile"><b>${info.losses}</b><span>loss functions</span></div>
  </div>
  <div class="grid wide">${DOORS.map(([href, t, d]) => `<a class="card" href="${href}"><b>${t}</b><span>${d}</span></a>`).join('')}</div>`;

  const mv = simulateMovie(demo.seed, demo.params), res = runBaseline(mv.frames, mv.H, mv.W);
  const cv = document.getElementById('hero'), m = demo.metrics;
  document.getElementById('hcap').innerHTML = `Seed ${demo.seed}: ${demo.params.num_particles} blinking particles, Poisson noise, SNR setting
    ${demo.params.snr}. <span style="color:#5cf2c8">rings</span> ground truth, <span style="color:#8fa2ff">lines</span> baseline tracks.
    Baseline (Python): detection F1 ${m.f1.toFixed(3)}, MOTA ${m.mota.toFixed(3)}.`;
  const still = matchMedia('(prefers-reduced-motion: reduce)').matches;
  let t = 0, last = 0;
  drawFrame(cv, mv, 0, { gt: true, tracks: true, result: res, size: 480 });
  if (!still) animate(now => {
    if (now - last < 150) return;
    last = now; t = (t + 1) % mv.frames.length;
    drawFrame(cv, mv, t, { gt: true, tracks: true, result: res, size: 480 });
  });
});
