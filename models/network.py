"""
Deep neural network architecture for particle tracking in microscopy images.

The classes live in focused modules and are re-exported here (old imports keep working):
- models/blocks.py            ConvBlock, EncoderBlock, DecoderBlock, ConvLSTMCell, super_resolution_head
- models/tracker_simple.py    ParticleTrackerModel
- models/tracker_dual.py      DualBranchParticleTracker
- models/tracker_attentive.py AttentiveParticleTracker
"""

from models.blocks import ConvBlock, EncoderBlock, DecoderBlock, ConvLSTMCell, super_resolution_head
from models.tracker_simple import ParticleTrackerModel
from models.tracker_dual import DualBranchParticleTracker
from models.tracker_attentive import AttentiveParticleTracker

__all__ = ['ConvBlock', 'EncoderBlock', 'DecoderBlock', 'ConvLSTMCell', 'ParticleTrackerModel',
           'DualBranchParticleTracker', 'AttentiveParticleTracker', 'super_resolution_head', 'create_model']


# Define model factory to create models with different configurations
def create_model(model_type, **kwargs):
    """
    Factory function to create a model with the specified configuration.

    Args:
        model_type: Type of model to create ('simple', 'dual', 'attentive')
        **kwargs: Model configuration parameters

    Returns:
        Instantiated model
    """
    if model_type == 'simple':
        return ParticleTrackerModel(**kwargs)
    elif model_type == 'dual':
        return DualBranchParticleTracker(**kwargs)
    elif model_type == 'attentive':
        return AttentiveParticleTracker(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
