// Trained-model page (local app only): list checkpoints on this machine and score one against the classical baseline
// on the same seeded movie. Nothing is trained here, and no weights are part of the repository or the static site.
import { route, api, app, esc, isStatic, needsServer } from './core.js';

const f3 = v => (Number.isFinite(v) ? v.toFixed(3) : '—');
const KEYS = ['precision', 'recall', 'f1', 'mota', 'motp', 'id_switches', 'tracks'];

route('model', async () => {
  const M = await api('/models', { fresh: true });
  const intro = `<h1 class="glow">Trained model</h1>
  <p class="lead">Run a checkpoint trained with the desktop app (Training tab) on a seeded simulated movie and compare it
  with the classical baseline on the same movie, using the repository's <code>ParticlePredictor</code> and
  <code>MetricsCalculator</code>.</p>
  <div class="callout">No trained weights are included in the repository, and models are never trained by the web app or
  in CI. This page lists checkpoints on your machine (<code>checkpoints/</code>, where training saves
  <code>&lt;training id&gt;/best.pth</code>, and <code>models/</code>).</div>`;
  if (isStatic() || M.static) { app().innerHTML = intro + needsServer('Running a trained model'); return; }
  const torchLine = M.torch.available ? `<p class="ok small">PyTorch ${esc(M.torch.version)} is available.</p>`
    : `<p class="err small">PyTorch is not importable here (${esc(M.torch.error || '')}). Install it (requirements.txt) to use this page.</p>`;
  if (!M.checkpoints.length) {
    app().innerHTML = intro + torchLine + `<p>No checkpoints found. Train a model in the desktop app (<code>python main.py</code>,
      Training tab), then reload this page.</p>`;
    return;
  }
  app().innerHTML = intro + torchLine + `
  <div class="panel"><div class="row">
    <div><label for="ck">Checkpoint</label><select id="ck">${M.checkpoints.map(c => `<option value="${esc(c.path)}">${esc(c.path)} (${c.size_mb} MB)</option>`).join('')}</select></div>
    <div><label for="seed">Seed</label><input id="seed" type="number" min="0" value="1"></div>
    <div><label for="size">Image size (px)</label><input id="size" type="number" min="32" max="512" step="16" value="128"></div>
    <div><label for="np">Particles</label><input id="np" type="number" min="0" max="500" value="20"></div>
    <div><label for="snr">SNR setting</label><input id="snr" type="number" min="1" max="50" step="0.5" value="10"></div>
    <div><label for="thr">Probability threshold</label><input id="thr" type="number" min="0" max="1" step="0.05" value="0.5"></div>
  </div><button type="button" id="go" ${M.torch.available ? '' : 'disabled'}>Run model and baseline</button></div>
  <div id="out"></div>`;
  const val = id => +document.getElementById(id).value;
  document.getElementById('go').addEventListener('click', async e => {
    e.target.disabled = true;
    const out = document.getElementById('out');
    out.innerHTML = '<p class="mute">Running…</p>';
    try {
      const r = await api('/predict', { body: { checkpoint: document.getElementById('ck').value, seed: val('seed'), threshold: val('thr'),
        sim: { width: val('size'), height: val('size'), num_frames: 10, num_particles: val('np'), snr: val('snr') } } });
      out.innerHTML = `<div class="scroll"><table><tr><th>score (≤ 5 px)</th><th class="num">trained model</th><th class="num">classical baseline</th></tr>
        ${KEYS.map(k => `<tr><td>${k}</td><td class="num">${f3(r.model[k])}</td><td class="num">${f3(r.baseline[k])}</td></tr>`).join('')}</table></div>
        <p class="small mute">${esc(r.checkpoint)} · seed ${r.seed} · ${r.params.width} × ${r.params.height} px, ${r.params.num_frames} frames.</p>`;
    } catch (err) { out.innerHTML = `<p class="err">${esc(err.message)}</p>`; }
    e.target.disabled = false;
  });
});
