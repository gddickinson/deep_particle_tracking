// About page: what the web version does and leaves out, conventions, how to run, credits (from the README).
import { route, api, app, esc } from './core.js';

const REPO = 'https://github.com/gddickinson/deep_particle_tracking';

route('about', async () => {
  const info = await api('/info');
  app().innerHTML = `
  <h1 class="glow">About</h1>
  <p class="lead">${esc(info.description)} The desktop application (PyQt5 + PyTorch) has three tabs: Simulation,
  Training and Prediction.</p>
  <div class="two">
  <div>
  <h2>What the web version does</h2>
  <p><b>Simulator</b>: the Simulation tab's pipeline (<code>simulator/movie.py</code>, shared with the desktop app)
  ported to JavaScript. NumPy's random generator is reproduced too, so a seed gives the same movie in the browser as in
  Python (checked pixel by pixel in <code>web/static/engine.test.mjs</code>).</p>
  <p><b>Classical baseline</b>: a matched-filter detector with the repo's peak detection and Hungarian linker, so movies
  can be scored without trained weights. It is labelled wherever it appears and is not the deep model.</p>
  <p><b>Benchmark</b>: baseline scores over SNR and density, computed by the Python code when the site is built.</p>
  <p><b>Trained model</b> (local app only): runs a checkpoint you trained on this machine against the baseline.</p>
  <h2>Left out</h2>
  <p>Training (it needs PyTorch and a GPU-sized budget; it is never run in CI), the Prediction tab's file loading and
  viewers, and the simulator models that the Simulation tab does not offer (anomalous diffusion, active transport,
  astigmatic and empirical PSFs, EMCCD / sCMOS / background noise in <code>simulator/</code>).</p>
  <h2>Conventions</h2>
  <p>Positions are (y, x) in pixels, pixel centres at integer coordinates, row 0 at the top of the image (as in the
  arrays and saved TIFFs; the desktop viewer draws with <code>origin='lower'</code>, so its display is flipped
  vertically). Pixel size 0.1 µm, frame interval 0.1 s (<code>ParticleGenerator</code> defaults). Frames are clipped
  to 0–1; the desktop app saves them as 8-bit TIFF (× 255). Grey levels on this site are scaled per frame (min to max).</p>
  </div>
  <div>
  <h2>Run the desktop app</h2>
  <pre>git clone ${REPO}.git
cd deep_particle_tracking
pip install -r requirements.txt
python main.py            # --cpu to force CPU, --debug for debug logging</pre>
  <h2>Run the web app</h2>
  <pre>pip install -r requirements-web.txt
python run_web.py         # http://127.0.0.1:8201</pre>
  <p>PyTorch is only needed for the Trained model page.</p>
  <h2>Tests</h2>
  <pre>python -m pytest -q tests
node web/static/engine.test.mjs</pre>
  <h2>Credits</h2>
  <p>From the README: the Deep-STORM paper by Nehme et al. for inspiration, the PyTorch team for the deep learning
  framework, and the scientific Python community. MIT License.
  Source: <a href="${REPO}">${REPO.replace('https://', '')}</a>.</p>
  </div>
  </div>`;
});
