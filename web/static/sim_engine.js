// Synthetic movie generator: JS port of simulator/movie.py simulate_movie (ParticleGenerator motion and blinking,
// GaussianPSF / AiryDiskPSF, Poisson / Gaussian / Poisson + Gaussian noise). With the same seed it draws the same
// random numbers in the same order as NumPy's legacy RandomState, and keeps the float32 steps of the Python code, so
// a seed gives the same movie as `simulate_movie(seed=..)` (checked by sim_engine.test.mjs against Python fixtures).
// Coordinates are (y, x) pixels, pixel centres at integers; pixel size 0.1 µm, frame interval 0.1 s.
import { RandomState } from './np_random.js';

export const DEFAULTS = {
  width: 512, height: 512, num_frames: 10, num_particles: 20,
  motion: 'Brownian', diffusion: 0.1, velocity: 1.0, direction_change: 0.1,
  confinement_strength: 1.0, confinement_radius: 10.0,
  psf: 'Gaussian', psf_sigma: 1.0, airy_radius: 1.22,
  noise: 'Poisson', snr: 10.0, background: 0.1, read_noise: 3.0,
  blinking: false, on_prob: 0.7, off_prob: 0.2,
};
export const MOTIONS = ['Brownian', 'Directed', 'Confined'];
export const PSFS = ['Gaussian', 'Airy Disk'];
export const NOISES = ['Poisson', 'Gaussian', 'Poisson + Gaussian'];
export const PIXEL_UM = 0.1, FRAME_S = 0.1;
const f32 = Math.fround;
const AIRY_FIRST_ZERO = 3.8317059702075125;

// Bessel J1: power series below 12, Hankel asymptotic expansion above (|error| < 1e-10 against scipy.special.j1).
export function besselJ1(x) {
  const ax = Math.abs(x);
  if (ax < 12) {
    const h = x / 2, h2 = h * h;
    let term = h, sum = h;
    for (let k = 1; k < 60; k++) { term *= -h2 / (k * (k + 1)); sum += term; if (Math.abs(term) < 1e-17 * Math.abs(sum)) break; }
    return sum;
  }
  const mu = 4, z8 = 8 * ax, chi = ax - 0.75 * Math.PI;
  let P = 1, Q = 0, t = 1;
  for (let k = 1; k < 12; k++) {
    t *= (mu - (2 * k - 1) ** 2) / (k * z8);
    if (k % 2) Q += (k % 4 === 1 ? 1 : -1) * t; else P += (k % 4 === 2 ? -1 : 1) * t;
  }
  const v = Math.sqrt(2 / (Math.PI * ax)) * (P * Math.cos(chi) - Q * Math.sin(chi));
  return x < 0 ? -v : v;
}

// ---- ParticleGenerator --------------------------------------------------------------------------------------------
function createParticles(rs, n, H, W) {
  const ps = [];
  for (let i = 0; i < n; i++) {
    const y = rs.uniform(0, H), x = rs.uniform(0, W), intensity = rs.uniform(0.5, 1.0), size = rs.uniform(0.8, 1.2);
    ps.push({ id: i, pos: [[y, x]], intensity, size, active: [true] });
  }
  return ps;
}

function brownian(rs, ps, T, D, H, W) {
  const sd = Math.sqrt(2 * D * FRAME_S) / PIXEL_UM;
  for (let t = 0; t < T; t++) for (const p of ps) {
    const [y, x] = p.pos[p.pos.length - 1];
    const dy = rs.normal(0, sd), dx = rs.normal(0, sd);
    p.pos.push([Math.max(0, Math.min(H, y + dy)), Math.max(0, Math.min(W, x + dx))]);
  }
}

function directed(rs, ps, T, vmin, vmax, pChange, H, W) {
  const dir = new Map(), vel = new Map();
  for (const p of ps) { dir.set(p.id, rs.uniform(0, 2 * Math.PI)); vel.set(p.id, rs.uniform(vmin, vmax)); }
  const toPx = FRAME_S / PIXEL_UM;
  for (let t = 0; t < T; t++) for (const p of ps) {
    if (rs.random() < pChange) dir.set(p.id, dir.get(p.id) + rs.normal(0, Math.PI / 4));
    const d = dir.get(p.id), vp = vel.get(p.id) * toPx;
    const [y, x] = p.pos[p.pos.length - 1];
    let ny = y + vp * Math.sin(d), nx = x + vp * Math.cos(d);
    if (nx < 0 || nx >= W) { dir.set(p.id, Math.PI - d); nx = Math.max(0, Math.min(W - 1, nx)); }
    if (ny < 0 || ny >= H) { dir.set(p.id, -d); ny = Math.max(0, Math.min(H - 1, ny)); }
    p.pos.push([ny, nx]);
  }
}

function confined(rs, ps, T, D, strength, R, H, W) {
  const sd = Math.sqrt(2 * D * FRAME_S) / PIXEL_UM;
  for (let t = 0; t < T; t++) for (const p of ps) {
    const [y0, x0] = p.pos[0], [yc, xc] = p.pos[p.pos.length - 1];
    const dyc = yc - y0, dxc = xc - x0, dist = Math.sqrt(dyc * dyc + dxc * dxc);
    let fy = 0, fx = 0;
    if (dist > 0) { const q = dist / R, f = strength * (q * q); fy = f * (-dyc / dist); fx = f * (-dxc / dist); }
    const dy = rs.normal(0, sd) + fy * FRAME_S, dx = rs.normal(0, sd) + fx * FRAME_S;
    p.pos.push([Math.max(0, Math.min(H - 1, yc + dy)), Math.max(0, Math.min(W - 1, xc + dx))]);
  }
}

function blinking(rs, ps, T, onP, offP) {
  for (let t = 1; t < T; t++) for (const p of ps) {
    const prev = p.active[t - 1];
    p.active.push(prev ? rs.random() >= offP : rs.random() < onP);
  }
}

// ---- PSF image (float32, as the Python models) ----------------------------------------------------------------------
// Gaussian spots are summed only within 12 sigma of the centre (full = true: whole frame, as Python). Beyond that each
// term is below 1e-31, which cannot change a float32 pixel once the background (>= 1e-3 of the peak) is added, so the
// noisy frames are identical (checked in engine.test.mjs).
export function psfImage(spots, H, W, p, full = false) {
  const img = new Float32Array(H * W);
  for (const s of spots) {
    const [y, x] = s.pos;
    if (p.psf === 'Gaussian') {
      const sig = p.psf_sigma * s.size, den = 2 * (sig * sig), R = full ? Infinity : Math.ceil(12 * sig) + 1;
      const r0 = Math.max(0, Math.floor(y - R)), r1 = Math.min(H - 1, Math.ceil(y + R));
      const c0 = Math.max(0, Math.floor(x - R)), c1 = Math.min(W - 1, Math.ceil(x + R));
      for (let r = r0; r <= r1; r++) { const dy = (r - y) * (r - y);
        for (let c = c0; c <= c1; c++) { const dx = c - x, i = r * W + c; img[i] = f32(img[i] + s.intensity * Math.exp(-(dy + dx * dx) / den)); } }
    } else {
      const rad = p.airy_radius * s.size;
      for (let r = 0; r < H; r++) for (let c = 0; c < W; c++) {
        const d = Math.sqrt((r - y) * (r - y) + (c - x) * (c - x)), i = r * W + c;
        let a = 1.0;
        if (d > 0) { const v = AIRY_FIRST_ZERO * d / rad, q = 2 * besselJ1(v) / v; a = q * q; }
        img[i] = f32(img[i] + s.intensity * a);
      }
    }
  }
  return img;
}

// noise_model.apply(psf_image + bg_level) - bg_level, then clip to [0, 1]
function noisyFrame(rs, img, bg, p) {
  const n = img.length, out = new Float64Array(n), bg32 = f32(bg);
  if (p.noise === 'Gaussian') {
    const sd = 10.0 / p.snr;
    for (let i = 0; i < n; i++) out[i] = (f32(img[i] + bg32) + (0 + sd * rs.gauss())) - bg;
  } else {
    const counts = new Float64Array(n);
    for (let i = 0; i < n; i++) counts[i] = rs.poisson(Math.max(0, f32(f32(img[i] + bg32) * 100.0)));
    if (p.noise === 'Poisson') {
      for (let i = 0; i < n; i++) out[i] = f32(f32(counts[i] / 100.0) - bg32);
    } else {
      for (let i = 0; i < n; i++) out[i] = ((counts[i] + (0 + p.read_noise * rs.gauss())) / 100.0 - 0.0) - bg;
    }
  }
  for (let i = 0; i < n; i++) out[i] = Math.min(1, Math.max(0, out[i]));
  return out;
}

// simulateMovie(seed, params) -> { frames: [Float64Array(H*W)], gt: [[[id, y, x], ...] per frame], particles, params }
export function simulateMovie(seed, params = {}, { full = false } = {}) {
  for (const k of Object.keys(params)) if (!(k in DEFAULTS)) throw new Error(`Unknown simulation parameter: ${k}`);
  const p = { ...DEFAULTS, ...params };
  if (!MOTIONS.includes(p.motion) || !PSFS.includes(p.psf) || !NOISES.includes(p.noise)) throw new Error('Unknown model');
  const rs = new RandomState(seed), H = p.height | 0, W = p.width | 0, T = p.num_frames | 0;
  const ps = createParticles(rs, p.num_particles | 0, H, W);
  if (p.motion === 'Brownian') brownian(rs, ps, T, p.diffusion, H, W);
  else if (p.motion === 'Directed') directed(rs, ps, T, p.velocity * 0.8, p.velocity * 1.2, p.direction_change, H, W);
  else confined(rs, ps, T, p.diffusion, p.confinement_strength, p.confinement_radius, H, W);
  for (const q of ps) q.active = Array(T).fill(true);
  if (p.blinking) { for (const q of ps) q.active = [true]; blinking(rs, ps, T, p.on_prob, p.off_prob); }

  const frames = [], gt = [];
  for (let t = 0; t < T; t++) {
    const vis = ps.filter(q => q.active[t]);
    const img = psfImage(vis.map(q => ({ pos: q.pos[t], intensity: q.intensity, size: q.size })), H, W, p, full);
    let mx = 0; for (let i = 0; i < img.length; i++) if (img[i] > mx) mx = img[i];
    const bg = mx > 0 ? mx / p.snr : p.background;
    frames.push(noisyFrame(rs, img, bg, p));
    gt.push(vis.map(q => [q.id, q.pos[t][0], q.pos[t][1]]));
  }
  return { frames, gt, particles: ps, params: p, H, W };
}

// Ground-truth tracks for the metrics: one array of T [y, x] (NaN when not visible) per particle.
export function gtTracks(movie) {
  const T = movie.frames.length;
  return movie.particles.map(q => Array.from({ length: T }, (_, t) => (q.active[t] ? [q.pos[t][0], q.pos[t][1]] : [NaN, NaN])));
}

// Peak SNR measured on the (clipped) frames: mean pixel value at the visible particles' rounded positions minus the
// background mean, over the background standard deviation (every 2nd pixel further than R from all visible particles).
export function measuredSnr(movie) {
  const { H, W, frames, gt } = movie, R = 4 * Math.max(movie.params.psf_sigma, movie.params.airy_radius) + 2;
  let s = 0, s2 = 0, n = 0, peak = 0, np_ = 0;
  frames.forEach((f, t) => {
    for (const [, y, x] of gt[t]) { const r = Math.round(y), c = Math.round(x); if (r >= 0 && r < H && c >= 0 && c < W) { peak += f[r * W + c]; np_++; } }
    for (let r = 0; r < H; r += 2) for (let c = 0; c < W; c += 2) {
      if (gt[t].some(([, y, x]) => (r - y) ** 2 + (c - x) ** 2 < R * R)) continue;
      const v = f[r * W + c]; s += v; s2 += v * v; n++;
    }
  });
  if (!n || !np_) return { snr: NaN, signal: NaN, noise: NaN };
  const mean = s / n, sd = Math.sqrt(Math.max(0, s2 / n - mean * mean)), signal = peak / np_ - mean;
  return { snr: sd > 0 ? signal / sd : Infinity, signal, noise: sd, background: mean };
}
