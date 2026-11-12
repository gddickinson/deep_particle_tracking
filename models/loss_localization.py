"""Localisation losses: FocalLoss, DistanceTransformLoss."""

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
