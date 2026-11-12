"""Embedding (association) losses: EmbeddingLoss, TemporalEmbeddingLoss."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import logging

logger = logging.getLogger(__name__)


class EmbeddingLoss(nn.Module):
    """
    Embedding loss for particle tracking.
    Encourages embeddings of the same particle across frames to be similar
    and embeddings of different particles to be dissimilar.

    The loss is a mean over particle-pixel pairs (then over frame pairs and windows), and by default the squared
    distance is also a mean over embedding channels (normalize_by_dim). Without that, the same-ID term grows with the
    embedding dimension and starts 15-60x larger than the localisation loss on an untrained model.
    """

    def __init__(self,
                 margin: float = 1.0,
                 pos_weight: float = 1.0,
                 neg_weight: float = 1.0,
                 normalize_by_dim: bool = True):
        """
        Initialize Embedding Loss.

        Args:
            margin: Margin for contrastive loss (on the per-channel mean squared distance when normalize_by_dim)
            pos_weight: Weight for positive pairs
            neg_weight: Weight for negative pairs
            normalize_by_dim: Divide squared embedding distances by the embedding dimension (mean over channels, not
                sum), so the loss scale does not depend on embedding_dim
        """
        super().__init__()
        # Ensure float32 dtype for internal parameters
        self.margin = float(margin)  # Ensure Python float (which will convert to fp32 in PyTorch)
        self.pos_weight = float(pos_weight)
        self.neg_weight = float(neg_weight)
        self.normalize_by_dim = bool(normalize_by_dim)

    def forward(self,
                embeddings: torch.Tensor,
                track_ids: torch.Tensor,
                masks: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.

        Args:
            embeddings: Predicted embeddings (B, F, C, H, W) where F is frames
            track_ids: Ground truth track IDs (B, F, 1, H, W), 0 indicates background
            masks: Optional masks to exclude background (B, F, 1, H, W)

        Returns:
            Loss value
        """
        # Ensure float32 for embeddings
        embeddings = embeddings.float()

        batch_size, num_frames, channels, height, width = embeddings.size()

        # Extract embeddings at particle positions
        total_loss = 0.0  # Use Python float which will convert to fp32
        valid_pairs = 0

        for b in range(batch_size):
            for f1 in range(num_frames):
                for f2 in range(f1+1, num_frames):
                    # Extract embeddings and IDs for the pair of frames
                    emb1 = embeddings[b, f1]  # (C, H, W)
                    emb2 = embeddings[b, f2]  # (C, H, W)

                    ids1 = track_ids[b, f1, 0]  # (H, W)
                    ids2 = track_ids[b, f2, 0]  # (H, W)

                    if masks is not None:
                        mask1 = masks[b, f1, 0]  # (H, W)
                        mask2 = masks[b, f2, 0]  # (H, W)
                    else:
                        mask1 = ids1 > 0
                        mask2 = ids2 > 0

                    # Skip if no particles in either frame
                    if not (torch.any(mask1) and torch.any(mask2)):
                        continue

                    # Extract particle positions
                    pos1_y, pos1_x = torch.where(mask1)
                    pos2_y, pos2_x = torch.where(mask2)

                    # Skip if no particles found
                    if len(pos1_y) == 0 or len(pos2_y) == 0:
                        continue

                    # Get embeddings at particle positions
                    emb1_particles = emb1[:, pos1_y, pos1_x].t()  # (N1, C)
                    emb2_particles = emb2[:, pos2_y, pos2_x].t()  # (N2, C)

                    # Get track IDs at particle positions
                    ids1_particles = ids1[pos1_y, pos1_x]  # (N1,)
                    ids2_particles = ids2[pos2_y, pos2_x]  # (N2,)

                    # Compute pairwise distances
                    n1, n2 = len(pos1_y), len(pos2_y)

                    # Expand dimensions for broadcasting
                    emb1_expanded = emb1_particles.unsqueeze(1)  # (N1, 1, C)
                    emb2_expanded = emb2_particles.unsqueeze(0)  # (1, N2, C)

                    # Compute squared Euclidean distance
                    dist_matrix = torch.sum((emb1_expanded - emb2_expanded) ** 2, dim=2)  # (N1, N2)
                    if self.normalize_by_dim:
                        dist_matrix = dist_matrix / channels

                    # Create match matrix (1 if same ID, 0 otherwise)
                    ids1_expanded = ids1_particles.unsqueeze(1)  # (N1, 1)
                    ids2_expanded = ids2_particles.unsqueeze(0)  # (1, N2)
                    match_matrix = (ids1_expanded == ids2_expanded).float()  # (N1, N2)

                    # Ignore background matches (ID = 0)
                    valid_mask = (ids1_expanded > 0) & (ids2_expanded > 0)

                    # Calculate positive and negative pairs loss
                    pos_loss = dist_matrix * match_matrix
                    neg_loss = torch.clamp(self.margin - dist_matrix, min=0) * (1 - match_matrix)

                    # Apply weights
                    pos_loss = self.pos_weight * pos_loss
                    neg_loss = self.neg_weight * neg_loss

                    # Apply mask and compute mean
                    combined_loss = pos_loss + neg_loss
                    masked_loss = combined_loss * valid_mask.float()

                    # Normalize by number of valid pairs
                    num_valid = valid_mask.sum()
                    if num_valid > 0:
                        frame_loss = masked_loss.sum() / num_valid
                        total_loss += frame_loss
                        valid_pairs += 1

        if valid_pairs > 0:
            return total_loss / valid_pairs
        else:
            # Return zero tensor with gradient - ensure float32
            return torch.tensor(0.0, requires_grad=True, device=embeddings.device)


class TemporalEmbeddingLoss(nn.Module):
    """
    Loss for enforcing temporal consistency in embeddings.
    This helps tracking particles by ensuring same particles have similar embeddings across frames.
    """

    def __init__(self, margin: float = 1.0):
        """
        Initialize the temporal embedding loss.

        Args:
            margin: Margin for contrastive loss
        """
        super().__init__()
        self.margin = margin

    def forward(self, embeddings: torch.Tensor, track_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            embeddings: Predicted embeddings (B, F, C, H, W) or (B, C, H, W)
            track_ids: Ground truth track IDs (B, F, 1, H, W), 0 indicates background

        Returns:
            Loss value
        """
        # Ensure embeddings have temporal dimension
        if embeddings.dim() == 4:  # (B, C, H, W)
            return torch.tensor(0.0, device=embeddings.device)

        batch_size, num_frames, embed_channels, height, width = embeddings.shape

        # Extract embeddings at particle positions
        total_loss = 0.0
        valid_pairs = 0

        for b in range(batch_size):
            for f1 in range(num_frames-1):
                f2 = f1 + 1  # Compare with next frame

                # Get track IDs for each frame
                ids1 = track_ids[b, f1, 0]  # (H, W)
                ids2 = track_ids[b, f2, 0]  # (H, W)

                # Get embeddings for each frame
                emb1 = embeddings[b, f1]  # (C, H, W)
                emb2 = embeddings[b, f2]  # (C, H, W)

                # Get unique IDs in both frames (excluding background)
                unique_ids1 = torch.unique(ids1)
                unique_ids1 = unique_ids1[unique_ids1 > 0]

                unique_ids2 = torch.unique(ids2)
                unique_ids2 = unique_ids2[unique_ids2 > 0]

                # Find common IDs
                common_ids = []
                for id1 in unique_ids1:
                    if id1 in unique_ids2:
                        common_ids.append(id1.item())

                # Compute loss for matching pairs
                for track_id in common_ids:
                    # Find positions for this ID in each frame
                    mask1 = (ids1 == track_id)
                    mask2 = (ids2 == track_id)

                    if not torch.any(mask1) or not torch.any(mask2):
                        continue

                    # Get embedding for this ID in each frame
                    # For simplicity, use average of all pixels with this ID
                    emb1_avg = torch.mean(emb1[:, mask1], dim=1)  # (C,)
                    emb2_avg = torch.mean(emb2[:, mask2], dim=1)  # (C,)

                    # Compute L2 distance
                    dist = torch.sum((emb1_avg - emb2_avg) ** 2)

                    # Add to loss
                    total_loss += dist
                    valid_pairs += 1

        if valid_pairs > 0:
            return total_loss / valid_pairs
        else:
            return torch.tensor(0.0, device=embeddings.device)
