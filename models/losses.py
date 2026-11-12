"""
Loss functions for training particle tracking models.

CombinedLoss (the trainer's loss) lives here; the component losses live in focused modules and are re-exported:
- models/loss_localization.py FocalLoss, DistanceTransformLoss
- models/loss_embedding.py    EmbeddingLoss, TemporalEmbeddingLoss
- models/loss_matching.py     HungarianMatchingLoss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import logging

logger = logging.getLogger(__name__)

from models.loss_localization import FocalLoss, DistanceTransformLoss
from models.loss_embedding import EmbeddingLoss, TemporalEmbeddingLoss
from models.loss_matching import HungarianMatchingLoss

__all__ = ['FocalLoss', 'DistanceTransformLoss', 'EmbeddingLoss', 'HungarianMatchingLoss', 'CombinedLoss',
           'TemporalEmbeddingLoss', 'test_temporal_processing']


class CombinedLoss(nn.Module):
    """
    Combined loss for particle tracking with reduced warnings.
    """

    # Class attribute to track which warnings have been shown
    _shown_warnings = set()

    def __init__(self,
                 loc_loss: nn.Module = None,
                 track_loss: nn.Module = None,
                 lambda_loc: float = 1.0,
                 lambda_track: float = 1.0):
        """
        Initialize Combined Loss.

        Args:
            loc_loss: Loss function for localization
            track_loss: Loss function for tracking
            lambda_loc: Weight for localization loss
            lambda_track: Weight for tracking loss
        """
        super().__init__()

        # Use default losses if not provided
        if loc_loss is None:
            self.loc_loss = FocalLoss(alpha=0.25, gamma=2.0)
        else:
            self.loc_loss = loc_loss

        if track_loss is None:
            self.track_loss = EmbeddingLoss(margin=1.0)
        else:
            self.track_loss = track_loss

        self.lambda_loc = lambda_loc
        self.lambda_track = lambda_track

    def forward(self, outputs: Union[torch.Tensor, Dict[str, torch.Tensor]],
                targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass with improved dimension handling.

        Args:
            outputs: Model output (tensor or dictionary)
            targets: Dictionary of target values

        Returns:
            Dictionary of loss components and total loss
        """
        losses = {}
        device = next(iter(targets.values())).device

        # Extract position prediction based on output type
        if isinstance(outputs, dict) and 'positions' in outputs:
            pred_pos = outputs['positions']
        else:
            # Output is directly the position prediction
            pred_pos = outputs

        # Get target positions
        target_pos = targets.get('positions')

        if target_pos is None:
            return {'loc_loss': torch.tensor(0.0, device=device),
                    'total': torch.tensor(0.0, device=device)}

        # Handle dimension mismatch - the core issue in our case
        # If pred_pos is (batch, channels, height, width) and target_pos is (batch, frames, channels, height, width)
        if pred_pos.dim() == 4 and target_pos.dim() == 5:
            # Add frame dimension to prediction
            pred_pos = pred_pos.unsqueeze(1)

        # If pred_pos is (batch, frames, channels, height, width) but frame count differs
        if pred_pos.dim() == 5 and target_pos.dim() == 5 and pred_pos.size(1) != target_pos.size(1):
            # Adjust frame count
            if pred_pos.size(1) < target_pos.size(1):
                # The models predict the last frame(s) of the input sequence (decoder skips and ConvLSTM state
                # come from the last frame), so compare with the last target frames (was the first)
                target_pos = target_pos[:, -pred_pos.size(1):]
            else:
                # Use only available frames from prediction
                pred_pos = pred_pos[:, :target_pos.size(1)]

        # Super-resolution models (Dual Branch, Attentive) predict at k x the target resolution; the targets mark
        # particle pixels at frame resolution, so take the most probable sub-pixel in each frame pixel (max-pool k)
        if pred_pos.shape[-2:] != target_pos.shape[-2:]:
            (ph, pw), (th, tw) = pred_pos.shape[-2:], target_pos.shape[-2:]
            k = ph // th
            if k > 1 and (ph, pw) == (th * k, tw * k):
                lead = pred_pos.shape[:-2]
                pred_pos = F.max_pool2d(pred_pos.reshape(-1, 1, ph, pw), k).reshape(*lead, th, tw)

        # Calculate localization loss
        pos_loss = self.loc_loss(pred_pos, target_pos)
        losses['loc_loss'] = pos_loss

        # Calculate tracking loss if available
        track_loss = torch.tensor(0.0, device=device)
        if self.lambda_track > 0.0 and isinstance(outputs, dict) and 'embeddings' in outputs and 'track_ids' in targets:
            embeddings = outputs['embeddings']
            track_ids = targets['track_ids']
            masks = targets.get('masks', None)

            # Ensure embeddings have correct dimensions if needed
            if embeddings.dim() == 4 and track_ids.dim() == 5:
                # Embeddings lack frame dimension - expand first
                embeddings = embeddings.unsqueeze(1)
                # Only one frame of embeddings: the last input frame
                track_ids = track_ids[:, -1:]

            # If embeddings and track_ids have different frame counts
            if embeddings.dim() == 5 and track_ids.dim() == 5 and embeddings.size(1) != track_ids.size(1):
                # Adjust frame count
                if embeddings.size(1) < track_ids.size(1):
                    track_ids = track_ids[:, -embeddings.size(1):]
                else:
                    embeddings = embeddings[:, :track_ids.size(1)]

            track_loss = self.track_loss(embeddings, track_ids, masks)
            losses['track_loss'] = track_loss
        else:
            losses['track_loss'] = track_loss

        # Compute total loss
        losses['total'] = self.lambda_loc * losses['loc_loss'] + self.lambda_track * losses['track_loss']

        return losses


def test_temporal_processing():
    """Test the improved temporal processing implementation."""
    import torch
    from models.network import ParticleTrackerModel, DualBranchParticleTracker, AttentiveParticleTracker

    # Create test input
    batch_size = 2
    num_frames = 5
    channels = 1
    height = 64
    width = 64

    x = torch.randn(batch_size, num_frames, channels, height, width)

    # Test ParticleTrackerModel
    print("Testing ParticleTrackerModel...")
    model1 = ParticleTrackerModel(
        input_channels=channels,
        num_frames=num_frames,
        base_filters=32,
        depth=3,
        embedding_dim=16
    )

    output1 = model1(x)
    print("Output keys:", output1.keys())
    print("Positions shape:", output1['positions'].shape)
    print("Embeddings shape:", output1['embeddings'].shape)

    # Test DualBranchParticleTracker
    print("\nTesting DualBranchParticleTracker...")
    model2 = DualBranchParticleTracker(
        input_channels=channels,
        num_frames=num_frames,
        base_filters=32,
        depth=3
    )

    output2 = model2(x)
    print("Output keys:", output2.keys())
    print("Positions shape:", output2['positions'].shape)
    print("Associations shape:", output2['associations'].shape)

    # Test AttentiveParticleTracker
    print("\nTesting AttentiveParticleTracker...")
    model3 = AttentiveParticleTracker(
        input_channels=channels,
        num_frames=num_frames,
        base_filters=32,
        depth=3,
        heads=4
    )

    output3 = model3(x)
    print("Output keys:", output3.keys())
    print("Positions shape:", output3['positions'].shape)
    print("Embeddings shape:", output3['embeddings'].shape)

    print("\nAll models successfully tested!")
