"""
Prediction and inference for particle tracking models.

Re-exports (old imports keep working):
- prediction/predictor.py ParticlePredictor
- prediction/manager.py   PredictionManager
- prediction/metrics.py   MetricsCalculator (torch-free)
"""

from prediction.predictor import ParticlePredictor
from prediction.manager import PredictionManager
from prediction.postprocess import detect_peaks, link_particles, link_particles_with_embeddings  # noqa: F401
from prediction.metrics import MetricsCalculator  # noqa: F401

__all__ = ['ParticlePredictor', 'PredictionManager', 'MetricsCalculator']
