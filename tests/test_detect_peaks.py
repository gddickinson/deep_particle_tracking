"""detect_peaks: nms_radius is a radius in pixels (window 2r+1), as ParticlePredictor documents."""
import numpy as np

from prediction.postprocess import detect_peaks


def test_peaks_closer_than_radius_are_suppressed():
    m = np.zeros((15, 15))
    m[7, 5], m[7, 7] = 1.0, 0.9      # 2 px apart
    assert detect_peaks(m, 0.5, 3).tolist() == [[7, 5]]


def test_peaks_further_than_radius_are_kept():
    m = np.zeros((15, 15))
    m[7, 3], m[7, 7] = 1.0, 0.9      # 4 px apart
    assert detect_peaks(m, 0.5, 3).tolist() == [[7, 3], [7, 7]]


def test_threshold_and_empty():
    m = np.zeros((5, 5))
    assert detect_peaks(m, 0.5, 3).shape == (0, 2)
