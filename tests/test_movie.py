"""simulator/movie.py: the Simulation tab pipeline as a function."""
import numpy as np
import pytest

from simulator.movie import simulate_movie, gt_tracks


def test_seed_reproduces_and_shapes():
    a = simulate_movie(seed=3, width=48, height=40, num_frames=6, num_particles=7)
    b = simulate_movie(seed=3, width=48, height=40, num_frames=6, num_particles=7)
    assert a['frames'].shape == (6, 40, 48)
    assert np.array_equal(a['frames'], b['frames'])
    assert a['frames'].min() >= 0 and a['frames'].max() <= 1


@pytest.mark.parametrize('motion', ['Brownian', 'Directed', 'Confined'])
@pytest.mark.parametrize('noise', ['Poisson', 'Gaussian', 'Poisson + Gaussian'])
def test_targets_match_ground_truth(motion, noise):
    r = simulate_movie(seed=1, width=32, height=32, num_frames=4, num_particles=5, motion=motion,
                       noise=noise, blinking=True)
    for t, rows in enumerate(r['gt']):
        for pid, y, x in rows:
            iy, ix = int(round(y)), int(round(x))
            if 0 <= iy < 32 and 0 <= ix < 32:
                assert r['positions'][t, iy, ix] == 1.0
    tracks = gt_tracks(r)
    assert len(tracks) == 5 and all(tr.shape == (4, 2) for tr in tracks)


def test_blinking_hides_particles():
    r = simulate_movie(seed=2, width=32, height=32, num_frames=30, num_particles=10, blinking=True,
                       off_prob=0.5, on_prob=0.5)
    counts = [len(g) for g in r['gt']]
    assert counts[0] == 10 and min(counts) < 10


def test_unknown_parameter_rejected():
    with pytest.raises(ValueError):
        simulate_movie(seed=0, colour='red')
