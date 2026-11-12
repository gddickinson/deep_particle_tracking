"""Classical baseline on simulated movies."""
import numpy as np

from prediction.classical import run_baseline, score_map
from prediction.metrics import MetricsCalculator
from simulator.movie import simulate_movie, gt_tracks


def test_flat_frame_scores_zero():
    assert np.all(score_map(np.full((8, 8), 0.3)) == 0)


def test_easy_movie_is_found():
    r = simulate_movie(seed=4, width=64, height=64, num_frames=10, num_particles=8, snr=10.0)
    out = run_baseline(r['frames'])
    gt = [g[:, 1:] for g in r['gt']]
    m = MetricsCalculator.calculate_all_metrics(out['positions'], gt, out['tracks'], gt_tracks(r), 5.0)
    assert m['detection']['recall'] > 0.8 and m['detection']['precision'] > 0.8
    assert m['tracking']['mota'] > 0.6
