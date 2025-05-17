"""
Loss functions for training particle tracking models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import logging

logger = logging.getLogger(__name__)


class FocalLoss(nn.Module):
    """
    Focal Loss for dense object detection with reduced warning messages.
    Reduces the relative loss for well-classified examples, focusing training on hard examples.
    """

    # Class attribute to track which warnings have been shown
    _shown_warnings = set()

    def __init__(self,
                 alpha: float = 0.25,
                 gamma: float = 2.0,
                 reduction: str = 'mean'):
        """
        Initialize Focal Loss.

        Args:
            alpha: Weighting factor for the rare class (positive examples)
            gamma: Focusing parameter that adjusts the rate at which easy examples are down-weighted
            reduction: Reduction method ('none', 'mean', 'sum')
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.shape_warning_shown = False

    def forward(self,
                inputs: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            inputs: Predicted probabilities
            targets: Target probabilities

        Returns:
            Focal loss
        """
        # Ensure float32 dtype
        inputs = inputs.float()  # Convert to float32
        targets = targets.float()  # Convert to float32

        # Handle shape mismatches - if dimensions are different, reshape to match
        if inputs.shape != targets.shape:
            # Only show shape mismatch warning once for this layer
            warning_key = f"shape_{inputs.shape}_{targets.shape}"
            if warning_key not in FocalLoss._shown_warnings:
                FocalLoss._shown_warnings.add(warning_key)
                if len(FocalLoss._shown_warnings) <= 5:  # Limit to first 5 unique shape warnings
                    logger.warning(f"Input shape {inputs.shape} doesn't match target shape {targets.shape}. Attempting to reshape.")

            # Try to reshape for compatibility
            try:
                # If inputs and targets have same number of dimensions but different sizes
                if inputs.dim() == targets.dim():
                    # Calculate total elements
                    input_elements = inputs.numel()
                    target_elements = targets.numel()

                    # If target has more elements, need to take a subset
                    if target_elements > input_elements:
                        # Reshape both to flat tensors
                        flat_inputs = inputs.reshape(-1)
                        flat_targets = targets.reshape(-1)

                        # Take only the first input_elements from targets
                        flat_targets = flat_targets[:input_elements]

                        # Reshape back to input shape
                        targets = flat_targets.reshape(inputs.shape)
                    else:
                        # If inputs have more elements, reshape inputs to match targets
                        inputs = inputs.reshape(targets.shape)
                else:
                    # If different number of dimensions, this is trickier
                    # Flatten both and truncate the longer one
                    flat_inputs = inputs.reshape(-1)
                    flat_targets = targets.reshape(-1)

                    # Use the smaller size
                    min_size = min(flat_inputs.size(0), flat_targets.size(0))
                    flat_inputs = flat_inputs[:min_size]
                    flat_targets = flat_targets[:min_size]

                    # Use flattened tensors for loss calculation
                    inputs = flat_inputs
                    targets = flat_targets
            except Exception as e:
                logger.error(f"Failed to reshape tensors: {str(e)}")
                raise ValueError(f"Cannot reconcile input shape {inputs.shape} with target shape {targets.shape}")

        # Binary case (C=1)
        if inputs.dim() == 1 or (inputs.dim() > 1 and inputs.size(1) == 1 and inputs.dim() == targets.dim()):
            inputs_flat = inputs.reshape(-1)
            targets_flat = targets.reshape(-1)

            # Binary cross entropy
            bce_loss = F.binary_cross_entropy(inputs_flat, targets_flat, reduction='none')

            # Apply focal weighting
            pt = torch.where(targets_flat == 1, inputs_flat, 1 - inputs_flat)
            alpha_factor = torch.ones_like(pt) * self.alpha
            alpha_factor = torch.where(targets_flat == 1, alpha_factor, 1 - alpha_factor)

            focal_weight = (1 - pt) ** self.gamma
            focal_weight = alpha_factor * focal_weight

            loss = focal_weight * bce_loss
        else:
            # Multi-class case
            inputs_reshape = inputs.reshape(-1, inputs.size(-1)) if inputs.dim() > 2 else inputs
            targets_reshape = targets.reshape(-1, targets.size(-1)) if targets.dim() > 2 else targets

            # Ensure shapes match
            if inputs_reshape.shape[0] != targets_reshape.shape[0]:
                min_size = min(inputs_reshape.shape[0], targets_reshape.shape[0])
                inputs_reshape = inputs_reshape[:min_size]
                targets_reshape = targets_reshape[:min_size]

            # Cross entropy
            ce_loss = F.binary_cross_entropy(inputs_reshape, targets_reshape, reduction='none')

            # Apply focal weighting
            pt = torch.exp(-ce_loss)
            focal_weight = (1 - pt) ** self.gamma

            # Apply alpha weighting
            if self.alpha is not None:
                alpha_factor = torch.ones_like(targets_reshape) * self.alpha
                alpha_factor = torch.where(targets_reshape > 0.5, alpha_factor, 1 - alpha_factor)
                focal_weight = alpha_factor * focal_weight

            loss = focal_weight * ce_loss

        # Apply reduction
        if self.reduction == 'none':
            return loss
        elif self.reduction == 'mean':
            return loss.mean()
        else:  # self.reduction == 'sum'
            return loss.sum()


class DistanceTransformLoss(nn.Module):
    """
    Distance transform based loss for particle localization.
    Penalizes predictions based on distance from ground truth particles.
    """

    def __init__(self,
                 sigma: float = 1.0,
                 alpha: float = 0.5,
                 distance_scale: float = 5.0):
        """
        Initialize Distance Transform Loss.

        Args:
            sigma: Standard deviation of Gaussian peaks at ground truth positions
            alpha: Weighting factor for balancing MSE and BCE
            distance_scale: Scaling factor for distance transform
        """
        super().__init__()
        self.sigma = sigma
        self.alpha = alpha
        self.distance_scale = distance_scale
        self.bce = nn.BCELoss(reduction='none')
        self.mse = nn.MSELoss(reduction='none')

    def forward(self,
                inputs: torch.Tensor,
                targets: torch.Tensor,
                target_dt: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.

        Args:
            inputs: Predicted probability maps (B, 1, H, W)
            targets: Target binary masks with particle positions (B, 1, H, W)
            target_dt: Pre-computed distance transform (B, 1, H, W), can be None

        Returns:
            Loss value
        """
        # If distance transform not provided, compute it on-the-fly
        if target_dt is None:
            # We'll use a simple approximation of distance transform
            # by applying a Gaussian filter to the binary target
            target_dt = self._approximate_distance_transform(targets)

        # Basic BCE loss
        bce_loss = self.bce(inputs, targets)

        # Distance weighted MSE loss
        mse_loss = self.mse(inputs, targets)
        weighted_mse = mse_loss * (1.0 + self.distance_scale * target_dt)

        # Combined loss
        loss = self.alpha * bce_loss + (1 - self.alpha) * weighted_mse

        return loss.mean()

    def _approximate_distance_transform(self,
                                        targets: torch.Tensor) -> torch.Tensor:
        """
        Apply Gaussian blur to approximate distance transform.

        Args:
            targets: Binary target tensor (B, 1, H, W)

        Returns:
            Approximate distance transform
        """
        # Invert the targets (distance should be 0 at particle centers)
        inverted = 1.0 - targets

        # Apply Gaussian blur as approximation
        kernel_size = int(6 * self.sigma) | 1  # Ensure odd kernel size
        sigma = (self.sigma, self.sigma)
        padding = kernel_size // 2

        blurred = F.gaussian_blur(
            inverted,
            kernel_size=(kernel_size, kernel_size),
            sigma=sigma,
            padding=padding
        )

        return blurred


class EmbeddingLoss(nn.Module):
    """
    Embedding loss for particle tracking.
    Encourages embeddings of the same particle across frames to be similar
    and embeddings of different particles to be dissimilar.
    """

    def __init__(self,
                 margin: float = 1.0,
                 pos_weight: float = 1.0,
                 neg_weight: float = 1.0):
        """
        Initialize Embedding Loss.

        Args:
            margin: Margin for contrastive loss
            pos_weight: Weight for positive pairs
            neg_weight: Weight for negative pairs
        """
        super().__init__()
        # Ensure float32 dtype for internal parameters
        self.margin = float(margin)  # Ensure Python float (which will convert to fp32 in PyTorch)
        self.pos_weight = float(pos_weight)
        self.neg_weight = float(neg_weight)

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


class HungarianMatchingLoss(nn.Module):
    """
    Hungarian matching loss for particle tracking.
    Assigns predictions to ground truths using optimal bipartite matching.
    """

    def __init__(self,
                 lambda_coord: float = 1.0,
                 lambda_conf: float = 1.0,
                 lambda_id: float = 1.0):
        """
        Initialize Hungarian Matching Loss.

        Args:
            lambda_coord: Weight for coordinate regression
            lambda_conf: Weight for confidence score
            lambda_id: Weight for ID assignment
        """
        super().__init__()
        self.lambda_coord = lambda_coord
        self.lambda_conf = lambda_conf
        self.lambda_id = lambda_id

    def forward(self,
                pred_particles: List[Dict[str, torch.Tensor]],
                gt_particles: List[Dict[str, torch.Tensor]]) -> torch.Tensor:
        """
        Forward pass.

        Args:
            pred_particles: List of dictionaries with predicted particles
                Each dict should have 'positions', 'scores', and optionally 'embeddings'
            gt_particles: List of dictionaries with ground truth particles
                Each dict should have 'positions', 'ids'

        Returns:
            Loss value
        """
        total_loss = 0.0
        batch_size = len(pred_particles)

        for b in range(batch_size):
            pred = pred_particles[b]
            gt = gt_particles[b]

            # Get predicted and ground truth positions
            pred_pos = pred['positions']  # (N, 2)
            gt_pos = gt['positions']  # (M, 2)

            # Get predicted scores and ground truth IDs
            pred_scores = pred['scores']  # (N,)
            gt_ids = gt['ids']  # (M,)

            # If either set is empty, handle specially
            if len(pred_pos) == 0:
                if len(gt_pos) > 0:
                    # All ground truths are missed
                    total_loss += self.lambda_conf * torch.tensor(1.0, device=pred_scores.device)
                continue

            if len(gt_pos) == 0:
                # All predictions are false positives
                total_loss += self.lambda_conf * F.binary_cross_entropy(
                    pred_scores, torch.zeros_like(pred_scores)
                )
                continue

            # Compute cost matrix for matching
            cost_matrix = torch.zeros(len(pred_pos), len(gt_pos), device=pred_pos.device)

            # Coordinate cost: squared Euclidean distance
            for i, p_pos in enumerate(pred_pos):
                for j, g_pos in enumerate(gt_pos):
                    cost_matrix[i, j] += self.lambda_coord * torch.sum((p_pos - g_pos) ** 2)

            # Add confidence cost
            for i in range(len(pred_pos)):
                for j in range(len(gt_pos)):
                    conf_cost = self.lambda_conf * F.binary_cross_entropy(
                        pred_scores[i].unsqueeze(0),
                        torch.ones(1, device=pred_scores.device)
                    )
                    cost_matrix[i, j] += conf_cost

            # If embeddings are available, add ID cost
            if 'embeddings' in pred and 'embeddings' in gt:
                pred_emb = pred['embeddings']  # (N, C)
                gt_emb = gt['embeddings']  # (M, C)

                for i in range(len(pred_pos)):
                    for j in range(len(gt_pos)):
                        emb_dist = torch.sum((pred_emb[i] - gt_emb[j]) ** 2)
                        cost_matrix[i, j] += self.lambda_id * emb_dist

            # Solve assignment problem using Hungarian algorithm
            # For simplicity, we use a greedy approach here
            # In a full implementation, you'd use scipy.optimize.linear_sum_assignment
            assignment = self._greedy_assignment(cost_matrix.detach().cpu().numpy())

            # Compute loss for matched pairs
            match_loss = 0.0
            matched_pred_indices = set()
            matched_gt_indices = set()

            for pred_idx, gt_idx in assignment:
                match_loss += cost_matrix[pred_idx, gt_idx]
                matched_pred_indices.add(pred_idx)
                matched_gt_indices.add(gt_idx)

            # Unmatched predictions cost (false positives)
            for i in range(len(pred_pos)):
                if i not in matched_pred_indices:
                    match_loss += self.lambda_conf * F.binary_cross_entropy(
                        pred_scores[i].unsqueeze(0),
                        torch.zeros(1, device=pred_scores.device)
                    )

            # Unmatched ground truths (false negatives)
            match_loss += self.lambda_conf * float(len(gt_pos) - len(matched_gt_indices))

            total_loss += match_loss

        return total_loss / max(1, batch_size)

    def _greedy_assignment(self, cost_matrix: np.ndarray) -> List[Tuple[int, int]]:
        """
        Greedy assignment algorithm.

        Args:
            cost_matrix: Cost matrix for assignment (N, M)

        Returns:
            List of (pred_idx, gt_idx) assignments
        """
        n, m = cost_matrix.shape
        assignment = []

        # Create a copy of the cost matrix
        costs = cost_matrix.copy()

        # Iteratively select the minimum cost assignment
        while costs.size > 0 and np.min(costs) < float('inf'):
            # Find the minimum cost
            i, j = np.unravel_index(np.argmin(costs), costs.shape)

            # Add the assignment
            assignment.append((i, j))

            # Remove the assigned row and column
            costs[i, :] = float('inf')
            costs[:, j] = float('inf')

        return assignment


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

    def forward(self,
                outputs: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass with reduced warnings.

        Args:
            outputs: Dictionary of model outputs
            targets: Dictionary of target values

        Returns:
            Dictionary of loss components and total loss
        """
        losses = {}
        device = None

        # Get the device from the first available tensor
        if isinstance(outputs, dict) and len(outputs) > 0:
            device = next(iter(outputs.values())).device
        elif not isinstance(outputs, dict) and isinstance(outputs, torch.Tensor):
            device = outputs.device
        elif isinstance(targets, dict) and len(targets) > 0:
            device = next(iter(targets.values())).device
        else:
            device = torch.device('cpu')

        # Handle localization loss
        if isinstance(outputs, dict) and 'positions' in outputs and 'positions' in targets:
            pred_pos = outputs['positions']
            target_pos = targets['positions']

            # Handle dimension mismatch - especially frame count mismatch
            if pred_pos.dim() != target_pos.dim():
                warning_key = f"dim_{pred_pos.dim()}_{target_pos.dim()}"
                if warning_key not in CombinedLoss._shown_warnings:
                    CombinedLoss._shown_warnings.add(warning_key)
                    logger.warning(f"Dimension mismatch between prediction ({pred_pos.dim()}) and target ({target_pos.dim()})")

                # If pred is (B, C, H, W) and target is (B, F, C, H, W)
                if pred_pos.dim() == 4 and target_pos.dim() == 5:
                    # Add frame dimension to prediction
                    pred_pos = pred_pos.unsqueeze(1)
                # If pred is (B, F, C, H, W) and target is (B, C, H, W)
                elif pred_pos.dim() == 5 and target_pos.dim() == 4:
                    # Take only the first frame of prediction
                    pred_pos = pred_pos[:, 0]

            # Handle frame count mismatch
            if pred_pos.dim() == 5 and target_pos.dim() == 5 and pred_pos.size(1) != target_pos.size(1):
                warning_key = f"frame_{pred_pos.size(1)}_{target_pos.size(1)}"
                if warning_key not in CombinedLoss._shown_warnings:
                    CombinedLoss._shown_warnings.add(warning_key)
                    logger.warning(f"Frame count mismatch: prediction has {pred_pos.size(1)} frames, target has {target_pos.size(1)} frames")

                # If prediction has fewer frames than target
                if pred_pos.size(1) < target_pos.size(1):
                    # Use only the first pred_pos.size(1) frames from target
                    target_pos = target_pos[:, :pred_pos.size(1)]
                else:
                    # Use only the first target_pos.size(1) frames from prediction
                    pred_pos = pred_pos[:, :target_pos.size(1)]

            # Handle different shapes (single frame vs multiple frames)
            if len(pred_pos.shape) == 4:  # (B, 1, H, W)
                pos_loss = self.loc_loss(pred_pos, target_pos)
            else:  # (B, F, 1, H, W)
                b, f, c, h, w = pred_pos.shape

                # Reshape for batch processing
                pos_loss = self.loc_loss(
                    pred_pos.reshape(b*f, c, h, w),
                    target_pos.reshape(b*f, c, h, w)
                )

            losses['loc_loss'] = pos_loss
        elif not isinstance(outputs, dict) and 'positions' in targets:
            # Direct outputs as position predictions
            pred_pos = outputs
            target_pos = targets['positions']

            # Handle dimension and shape differences
            if pred_pos.dim() != target_pos.dim():
                warning_key = f"direct_dim_{pred_pos.dim()}_{target_pos.dim()}"
                if warning_key not in CombinedLoss._shown_warnings:
                    CombinedLoss._shown_warnings.add(warning_key)
                    logger.warning(f"Dimension mismatch between prediction ({pred_pos.dim()}) and target ({target_pos.dim()})")

                # Try to make dimensions match
                if pred_pos.dim() < target_pos.dim():
                    # Add missing dimensions to prediction
                    while pred_pos.dim() < target_pos.dim():
                        pred_pos = pred_pos.unsqueeze(1)
                else:
                    # Remove extra dimensions from prediction
                    while pred_pos.dim() > target_pos.dim() and pred_pos.size(0) == 1:
                        pred_pos = pred_pos.squeeze(0)

            pos_loss = self.loc_loss(pred_pos, target_pos)
            losses['loc_loss'] = pos_loss
        else:
            losses['loc_loss'] = torch.tensor(0.0, requires_grad=True, device=device)

        # Handle tracking loss if available
        if isinstance(outputs, dict) and 'embeddings' in outputs and 'track_ids' in targets:
            embeddings = outputs['embeddings']
            track_ids = targets['track_ids']
            masks = targets.get('masks', None)

            track_loss = self.track_loss(embeddings, track_ids, masks)
            losses['track_loss'] = track_loss
        else:
            losses['track_loss'] = torch.tensor(0.0, requires_grad=True, device=device)

        # Compute total loss
        losses['total'] = self.lambda_loc * losses['loc_loss'] + self.lambda_track * losses['track_loss']

        return losses
