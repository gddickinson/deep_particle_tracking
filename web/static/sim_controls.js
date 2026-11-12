// Control panel for the simulator page: builds the inputs from the parameter list and calls onChange(key, value).
import { esc, slider } from './core.js';

// key, label, step, unit, format; ranges and defaults come from /api/params
export const NUMERIC = {
  size: ['Image size (px)', 16, ' px'],
  num_frames: ['Frames', 1, ''],
  num_particles: ['Particles', 1, ''],
  diffusion: ['Diffusion coefficient D', 0.01, ' µm²/s'],
  velocity: ['Velocity', 0.1, ' µm/s'],
  direction_change: ['Direction change probability', 0.01, ''],
  confinement_strength: ['Confinement strength', 0.1, ''],
  confinement_radius: ['Confinement radius', 0.5, ' px'],
  psf_sigma: ['Gaussian σ', 0.05, ' px'],
  airy_radius: ['Airy radius (first dark ring)', 0.05, ' px'],
  snr: ['SNR (peak / background noise SD)', 0.5, ''],
  read_noise: ['Read noise', 0.5, ' counts'],
  on_prob: ['On probability (off → on)', 0.05, ''],
  off_prob: ['Off probability (on → off)', 0.05, ''],
};
export const SHOWN = {
  motion: { Brownian: ['diffusion'], Directed: ['velocity', 'direction_change'], Confined: ['diffusion', 'confinement_strength', 'confinement_radius'] },
  psf: { Gaussian: ['psf_sigma'], 'Airy Disk': ['airy_radius'] },
  noise: { Poisson: ['snr'], Gaussian: ['snr'], 'Poisson + Gaussian': ['snr', 'read_noise'] },
};

function seg(host, id, label, options, value, onPick) {
  const wrap = document.createElement('div');
  wrap.className = 'ctl';
  wrap.innerHTML = `<label id="lb-${id}">${esc(label)}</label><div class="seg" role="group" aria-labelledby="lb-${id}">${options.map(o =>
    `<button type="button" data-v="${esc(o)}" class="${o === value ? 'on' : ''}" aria-pressed="${o === value}">${esc(o)}</button>`).join('')}</div>`;
  wrap.querySelectorAll('button').forEach(b => b.addEventListener('click', () => {
    wrap.querySelectorAll('button').forEach(x => { x.classList.toggle('on', x === b); x.setAttribute('aria-pressed', x === b); });
    onPick(b.dataset.v);
  }));
  host.appendChild(wrap);
}

// Build the panel into host for state p (simulation params incl. size) and P (/api/params).
export function buildControls(host, p, P, onChange) {
  host.innerHTML = '';
  const fmt = v => (Number.isInteger(v) ? String(v) : (+v).toFixed(2));
  const num = (parent, key) => {
    const [label, step, unit] = NUMERIC[key], rng = key === 'size' ? P.ranges.width : P.ranges[key];
    slider(parent, { id: key, label, min: rng[0], max: rng[1], step, value: p[key], unit, fmt }, v => onChange(key, v));
  };
  const h = t => { const e = document.createElement('h3'); e.textContent = t; host.appendChild(e); };
  const box = () => { const d = document.createElement('div'); host.appendChild(d); return d; };

  h('Movie'); ['size', 'num_frames', 'num_particles'].forEach(k => num(host, k));
  h('Motion');
  seg(host, 'motion', 'Motion model', P.motions, p.motion, v => { onChange('motion', v); fill(mBox, SHOWN.motion[v]); });
  const mBox = box();
  h('Point spread function');
  seg(host, 'psf', 'PSF model', P.psfs, p.psf, v => { onChange('psf', v); fill(pBox, SHOWN.psf[v]); });
  const pBox = box();
  h('Noise');
  seg(host, 'noise', 'Noise model', P.noises, p.noise, v => { onChange('noise', v); fill(nBox, SHOWN.noise[v]); });
  const nBox = box();
  h('Blinking');
  const bl = document.createElement('label'); bl.className = 'inl';
  bl.innerHTML = `<input type="checkbox" id="blink" ${p.blinking ? 'checked' : ''}> Enable blinking`;
  host.appendChild(bl);
  const bBox = box();
  bl.querySelector('input').addEventListener('change', e => { onChange('blinking', e.target.checked); fill(bBox, e.target.checked ? ['on_prob', 'off_prob'] : []); });
  function fill(b, keys) { b.innerHTML = ''; keys.forEach(k => num(b, k)); }
  fill(mBox, SHOWN.motion[p.motion]); fill(pBox, SHOWN.psf[p.psf]); fill(nBox, SHOWN.noise[p.noise]);
  fill(bBox, p.blinking ? ['on_prob', 'off_prob'] : []);
}
