"""
Classical (non-deep-learning) baseline detector and tracker, for comparison with the trained models.

Detection: a Gaussian matched filter (sigma = PSF sigma) turns each frame into a score map in units of
the filtered image's robust noise: score = (G - median(G)) / (1.4826 * MAD(G)). The score map then goes
through the same detect_peaks (threshold + local maximum within nms_radius) that the deep-model
predictor applies to its probability map, and the same frame-to-frame Hungarian linker.
This baseline is a reference point added for the web app; it is not part of the deep model.
"""

from typing import Dict, List

import numpy as np
from scipy.ndimage import gaussian_filter

from prediction.postprocess import detect_peaks, link_particles

BASELINE_DEFAULTS = {'sigma': 1.0, 'threshold': 6.0, 'nms_radius': 3, 'max_distance': 5.0, 'min_length': 3}


def score_map(frame: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """Matched-filter score in robust noise units (0 where the image is flat)."""
    g = gaussian_filter(np.asarray(frame, dtype=np.float64), sigma)
    med = np.median(g)
    mad = 1.4826 * np.median(np.abs(g - med))
    if mad <= 0:
        return np.zeros_like(g)
    return (g - med) / mad


def detect_frames(frames: np.ndarray, sigma: float = 1.0, threshold: float = 6.0,
                  nms_radius: int = 3) -> List[np.ndarray]:
    """Per-frame (y, x) integer detections."""
    return [detect_peaks(score_map(f, sigma), threshold, nms_radius).astype(float) for f in frames]


def run_baseline(frames: np.ndarray, **params) -> Dict:
    """Detect and link. Returns {'positions': [...], 'tracks': [...], 'params': {...}}."""
    p = {**BASELINE_DEFAULTS, **params}
    positions = detect_frames(frames, p['sigma'], p['threshold'], p['nms_radius'])
    tracks = link_particles(positions, p['max_distance'], p['min_length']) if len(frames) > 1 else []
    return {'positions': positions, 'tracks': tracks, 'params': p}
