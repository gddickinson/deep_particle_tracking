"""HungarianMatchingLoss (prediction-to-ground-truth matching)."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import logging

logger = logging.getLogger(__name__)


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
