"""Precomputed results for the web app: the classical baseline on simulated movies.

No trained weights are committed in this repository, so these numbers are for the classical baseline
(prediction/classical.py), not the deep model. Every movie comes from simulator/movie.py simulate_movie with
a fixed seed and every score from prediction/metrics.py MetricsCalculator (distance threshold 5 px, its default).
"""
from functools import lru_cache
from typing import Dict

import numpy as np

from prediction.classical import BASELINE_DEFAULTS, run_baseline
from prediction.metrics import MetricsCalculator
from simulator.movie import PEAK_SIGNAL, gt_tracks, simulate_movie

MATCH_PX = 5.0
SIZE, FRAMES, SEEDS = 128, 15, (1, 2)
SNRS = (1, 2, 3, 5, 10, 20, 50)
DENSITIES = (5, 10, 20, 40, 80, 160)
DEMO = dict(width=96, height=96, num_frames=30, num_particles=25, blinking=True, snr=10.0)
DEMO_SEED = 42


def measured_snr(result: Dict) -> float:
    """Measured SNR by the setting's own definition (simulator/movie.py noise_levels): PEAK_SIGNAL over the SD of
    the unclipped frames at the background (every 2nd pixel further than R from all visible particles,
    R = 4 * max(psf_sigma, airy_radius) + 2). As sim_engine.js measuredSnr."""
    p, raw = result['params'], result['raw_frames']
    H, W = raw.shape[1:]
    R = 4 * max(p['psf_sigma'], p['airy_radius']) + 2
    yy, xx = np.mgrid[0:H:2, 0:W:2]
    bg = []
    for f, g in zip(raw, result['gt']):
        far = np.ones(yy.shape, bool)
        for _, y, x in g:
            far &= (yy - y) ** 2 + (xx - x) ** 2 >= R * R
        bg.extend(f[0:H:2, 0:W:2][far].astype(float).tolist())
    if not bg:
        return float('nan')
    sd = float(np.std(bg))
    return PEAK_SIGNAL / sd if sd > 0 else float('inf')


def score(result: Dict, **baseline) -> Dict:
    b = run_baseline(result['frames'], **baseline)
    m = MetricsCalculator.calculate_all_metrics(b['positions'], [g[:, 1:] for g in result['gt']], b['tracks'],
                                                gt_tracks(result), MATCH_PX)
    d, t = m['detection'], m['tracking']
    return {'precision': float(d['precision']), 'recall': float(d['recall']), 'f1': float(d['f1']),
            'mota': float(t['mota']), 'motp': float(t['motp']), 'id_switches': int(t['id_switches']),
            'tracks': len(b['tracks'])}


def _avg(rows):
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}


@lru_cache(maxsize=1)
def benchmark() -> Dict:
    """SNR sweep for each noise model and a density sweep, averaged over SEEDS."""
    base = dict(width=SIZE, height=SIZE, num_frames=FRAMES)
    snr_rows = []
    for noise in ('Poisson', 'Gaussian', 'Poisson + Gaussian'):
        for snr in SNRS:
            runs = []
            for s in SEEDS:
                r = simulate_movie(seed=s, **base, num_particles=20, noise=noise, snr=float(snr))
                runs.append({**score(r), 'measured_snr': measured_snr(r)})
            snr_rows.append({'noise': noise, 'snr': snr, **_avg(runs)})
    dens_rows = []
    for n in DENSITIES:
        runs = [score(simulate_movie(seed=s, **base, num_particles=n)) for s in SEEDS]
        dens_rows.append({'particles': n, 'per_100_um2': n / (SIZE * 0.1) ** 2 * 100, **_avg(runs)})
    return {'method': 'classical baseline (matched filter + local maxima + Hungarian linking)',
            'deep_model': 'not included: no trained weights in the repository',
            'baseline': BASELINE_DEFAULTS, 'match_px': MATCH_PX, 'size': SIZE, 'frames': FRAMES, 'seeds': list(SEEDS),
            'snr_sweep': {'particles': 20, 'rows': snr_rows},
            'density_sweep': {'noise': 'Poisson', 'snr': 10.0, 'rows': dens_rows}}


@lru_cache(maxsize=1)
def demo() -> Dict:
    """One seeded movie with ground truth and the baseline's detections and tracks (Python results)."""
    r = simulate_movie(seed=DEMO_SEED, **DEMO)
    b = run_baseline(r['frames'])
    nan = lambda a: [[None if np.isnan(v) else float(v) for v in row] for row in a]  # noqa: E731
    return {'seed': DEMO_SEED, 'params': DEMO, 'baseline': b['params'],
            'gt': [g.tolist() for g in r['gt']],
            'detections': [q.astype(int).tolist() for q in b['positions']],
            'tracks': [nan(t) for t in b['tracks']],
            'metrics': score(r), 'measured_snr': measured_snr(r),
            'checksum': float(np.sum(r['frames'], dtype=np.float64))}
