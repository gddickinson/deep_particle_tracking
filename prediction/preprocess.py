"""
Intensity normalisation of microscopy input for the networks (torch-free).

The networks are trained on simulated frames in [0, 1] with the background subtracted (about 0) and particle
peaks up to about 1 (simulator/movie.py). Real recordings are mapped to the same range:
- float data already in [0, 1]: unchanged;
- 8-bit integers: divided by the dtype range (255), as for the 8-bit TIFFs the Simulation tab saves;
- anything else (16-bit or 32-bit integers, floats outside [0, 1]): percentiles of the whole array, so the
  background (median, 50th percentile) goes to 0 and the 99.99th percentile to 1, clipped to [0, 1]. Dividing a
  16-bit camera frame by 255 (the old behaviour) gave values in the hundreds.
"""

import numpy as np

LOW_PERCENTILE = 50.0
HIGH_PERCENTILE = 99.99


def normalize_intensity(data: np.ndarray) -> np.ndarray:
    """Return float32 data in [0, 1] (a frame or a whole stack; percentiles are taken over all of it)."""
    a = np.asarray(data)
    if a.dtype == np.uint8:
        return a.astype(np.float32) / 255.0
    if np.issubdtype(a.dtype, np.floating) and a.size and a.min() >= 0.0 and a.max() <= 1.0:
        return a.astype(np.float32)
    f = a.astype(np.float64)
    lo, hi = np.percentile(f, [LOW_PERCENTILE, HIGH_PERCENTILE])
    if not hi > lo:
        hi = f.max()
    if not hi > lo:
        return np.zeros(a.shape, dtype=np.float32)
    return np.clip((f - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)
