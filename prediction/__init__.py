"""Prediction package.

Contents:
- inference.py   re-exports ParticlePredictor (predictor.py) and PredictionManager (manager.py); need PyTorch
- postprocess.py detect_peaks, link_particles (numpy/scipy)
- metrics.py     MetricsCalculator (numpy/scipy)
- classical.py   classical (non-deep) baseline detector/tracker for comparison

The PyTorch classes are imported lazily so the torch-free modules work without PyTorch.
"""

from prediction.metrics import MetricsCalculator

__all__ = ['ParticlePredictor', 'PredictionManager', 'MetricsCalculator']


def __getattr__(name):
    if name in ('ParticlePredictor', 'PredictionManager'):
        from prediction import inference
        return getattr(inference, name)
    raise AttributeError(name)
