"""
Synthetic movie generator: the Simulation tab's pipeline as a pure (Qt-free, torch-free) function.

`simulate_movie` runs, in this order, exactly what SimulationTab.run_simulation did: random particles,
one motion model, optional blinking, a PSF image per frame, noise at the requested SNR (noise_levels: the
one SNR definition, shared with SimulatedParticleDataset), clipping to [0, 1], and the
per-pixel position / track-id targets used for training. The GUI and the web app both call it.

Coordinates are (y, x) in pixels with the pixel centre at integer coordinates (row y, column x), as in
the PSF models. Physical units: ParticleGenerator's pixel size (0.1 um) and frame interval (0.1 s).
"""

from typing import Dict, Optional, Tuple

import numpy as np

from simulator.particle_generator import ParticleGenerator
from simulator.psf_models import GaussianPSF, AiryDiskPSF
from simulator.noise_models import PoissonNoise, GaussianNoise, MixedNoise

# Defaults of the Simulation tab widgets (gui/main_window.py SimulationTab.setup_ui).
DEFAULTS = {
    'width': 512, 'height': 512, 'num_frames': 10, 'num_particles': 20,
    'motion': 'Brownian', 'diffusion': 0.1, 'velocity': 1.0, 'direction_change': 0.1,
    'confinement_strength': 1.0, 'confinement_radius': 10.0,
    'psf': 'Gaussian', 'psf_sigma': 1.0, 'airy_radius': 1.22,
    'noise': 'Poisson', 'snr': 10.0, 'read_noise': 3.0,
    'blinking': False, 'on_prob': 0.7, 'off_prob': 0.2,
}
MOTIONS = ('Brownian', 'Directed', 'Confined')
# Photons per image unit (PoissonNoise / MixedNoise scaling) and the amplitude of the brightest particle
# (intensity_range upper bound): the reference "peak signal" of the SNR definition.
PHOTONS_PER_UNIT = 100.0
PEAK_SIGNAL = 1.0
PSFS = ('Gaussian', 'Airy Disk')
NOISES = ('Poisson', 'Gaussian', 'Poisson + Gaussian')


def noise_levels(noise: str, snr: float, read_noise: float = 0.0) -> Tuple[float, float]:
    """The one SNR definition: SNR = peak signal above background / noise SD at the background.

    Peak signal = PEAK_SIGNAL (a particle of intensity 1). Returns (background, noise_sd) in image units.
    Gaussian noise: SD = peak / SNR, no background needed. Poisson: the background b gives a photon SD
    sqrt(b * PHOTONS_PER_UNIT), so b * PHOTONS_PER_UNIT = (PHOTONS_PER_UNIT * peak / SNR)^2. Poisson + Gaussian:
    sqrt(b * PHOTONS_PER_UNIT + read_noise^2) photons, so the background makes up what read noise does not; if read
    noise alone exceeds the target (SNR above PHOTONS_PER_UNIT * peak / read_noise) the background is 0 and the
    image SNR is that ceiling.
    """
    sd = PEAK_SIGNAL / snr
    if noise == 'Gaussian':
        return 0.0, sd
    var_photons = (sd * PHOTONS_PER_UNIT) ** 2
    if noise == 'Poisson + Gaussian':
        var_photons -= read_noise ** 2
    return max(var_photons, 0.0) / PHOTONS_PER_UNIT, sd


def add_noise(psf_image: np.ndarray, noise: str, snr: float, read_noise: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """Camera noise at the given SNR on a noise-free PSF image; the background is added for the shot noise and
    subtracted again. Returns (frame clipped to [0, 1], raw frame before clipping)."""
    bg, sd = noise_levels(noise, snr, read_noise)
    if noise == 'Poisson':
        model = PoissonNoise(scaling_factor=PHOTONS_PER_UNIT)
    elif noise == 'Gaussian':
        model = GaussianNoise(sigma=sd)
    else:
        model = MixedNoise(photon_scaling=PHOTONS_PER_UNIT, read_noise=read_noise, offset=0.0)
    raw = model.apply(psf_image + bg) - bg
    return np.clip(raw, 0, 1), raw


def simulate_movie(seed: Optional[int] = None, **params) -> Dict:
    """Simulate a movie with ground truth.

    Args:
        seed: if given, np.random.seed(seed) first (the GUI passes None and uses the global stream).
        **params: any key of DEFAULTS.

    Returns:
        dict with 'frames' (T, H, W) float in [0, 1], 'raw_frames' (the same before clipping), 'positions' (T, H, W) float32 0/1 targets,
        'track_ids' (T, H, W) int32 (particle id + 1), 'gt' (list per frame of (n, 3) arrays
        [id, y, x] for the particles visible in that frame), 'generator' (the ParticleGenerator,
        whose particles hold the full (y, x) trajectories) and 'params' (the values used).
    """
    unknown = set(params) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown simulation parameters: {sorted(unknown)}")
    p = {**DEFAULTS, **params}
    if p['motion'] not in MOTIONS or p['psf'] not in PSFS or p['noise'] not in NOISES:
        raise ValueError(f"Unknown model in {p['motion']!r}, {p['psf']!r}, {p['noise']!r}")
    if seed is not None:
        np.random.seed(seed)

    height, width = int(p['height']), int(p['width'])
    num_frames, num_particles = int(p['num_frames']), int(p['num_particles'])
    gen = ParticleGenerator(image_size=(height, width))
    particles = gen.create_random_particles(num_particles=num_particles,
                                            intensity_range=(0.5, 1.0), size_range=(0.8, 1.2))

    if p['motion'] == 'Brownian':
        gen.apply_brownian_motion(particles=particles, num_frames=num_frames,
                                  diffusion_coefficient=p['diffusion'])
    elif p['motion'] == 'Directed':
        gen.apply_directed_motion(particles=particles, num_frames=num_frames,
                                  velocity_range=(p['velocity'] * 0.8, p['velocity'] * 1.2),
                                  direction_change_prob=p['direction_change'])
    else:
        gen.apply_confined_diffusion(particles=particles, num_frames=num_frames,
                                     diffusion_coefficient=p['diffusion'],
                                     confinement_strength=p['confinement_strength'],
                                     confinement_radius=p['confinement_radius'])

    if p['blinking']:
        gen.apply_blinking(particles=particles, num_frames=num_frames,
                           on_probability=p['on_prob'], off_probability=p['off_prob'])

    if p['psf'] == 'Gaussian':
        psf_model = GaussianPSF(image_size=(height, width), sigma=p['psf_sigma'])
    else:
        psf_model = AiryDiskPSF(image_size=(height, width), airy_radius=p['airy_radius'])

    frames, raw_frames, positions, track_ids, gt = [], [], [], [], []
    for frame_idx in range(num_frames):
        active = gen.get_active_particles(frame_idx)
        pos = [pt.get_position(frame_idx) for pt in active]
        ids = [pt.id for pt in active]
        if pos:
            psf_image = psf_model.generate(positions=pos, intensities=[pt.intensity for pt in active],
                                           sizes=[pt.size for pt in active])
        else:
            psf_image = np.zeros((height, width), dtype=np.float32)

        frame, raw = add_noise(psf_image, p['noise'], p['snr'], p['read_noise'])
        frames.append(frame)
        raw_frames.append(raw)

        pos_target = np.zeros((height, width), dtype=np.float32)
        id_target = np.zeros((height, width), dtype=np.int32)
        for (y, x), pid in zip(pos, ids):
            iy, ix = int(round(y)), int(round(x))
            if 0 <= iy < height and 0 <= ix < width:
                pos_target[iy, ix] = 1.0
                id_target[iy, ix] = pid + 1  # 0 is background
        positions.append(pos_target)
        track_ids.append(id_target)
        gt.append(np.array([[pid, y, x] for (y, x), pid in zip(pos, ids)], dtype=float).reshape(-1, 3))

    return {'frames': np.stack(frames), 'raw_frames': np.stack(raw_frames), 'positions': np.stack(positions), 'track_ids': np.stack(track_ids),
            'gt': gt, 'generator': gen, 'params': p}


def gt_tracks(result: Dict) -> list:
    """Ground-truth tracks as MetricsCalculator expects: one (T, 2) (y, x) array per particle, NaN where
    the particle is not visible (blinked off)."""
    num_frames = len(result['frames'])
    tracks = {pt.id: np.full((num_frames, 2), np.nan) for pt in result['generator'].particles}
    for t, rows in enumerate(result['gt']):
        for pid, y, x in rows:
            tracks[int(pid)][t] = (y, x)
    return [tracks[k] for k in sorted(tracks)]
