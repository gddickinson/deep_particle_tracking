"""Frames with no visible particle use the Background Level setting (the code's own fallback)."""
import numpy as np

from simulator.movie import simulate_movie


def test_empty_frame_uses_background_level():
    r = simulate_movie(seed=0, width=16, height=16, num_frames=3, num_particles=0, noise='Poisson',
                       snr=10.0, background=0.0)
    assert np.all(r['frames'] == 0)
