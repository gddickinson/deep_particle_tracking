"""
Evaluation metrics for particle detection and tracking (precision/recall/F1, MOTA/MOTP, ID switches).

Torch-free so the web app and tests can use it without PyTorch; prediction.inference re-exports it.
Positions are (y, x) pixels; distances are Euclidean in pixels.
"""

from typing import Dict, List, Optional

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


class MetricsCalculator:
    """Calculate evaluation metrics for particle tracking."""

    @staticmethod
    def calculate_detection_metrics(pred_positions: np.ndarray,
                                   gt_positions: np.ndarray,
                                   distance_threshold: float = 5.0) -> Dict:
        """
        Calculate detection metrics (precision, recall, F1).

        Args:
            pred_positions: Predicted particle positions (N, 2)
            gt_positions: Ground truth particle positions (M, 2)
            distance_threshold: Maximum distance to consider a detection correct

        Returns:
            Dictionary with metrics
        """
        # Handle empty arrays
        if len(gt_positions) == 0:
            if len(pred_positions) == 0:
                return {'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'tp': 0, 'fp': 0, 'fn': 0}
            else:
                return {'precision': 0.0, 'recall': 1.0, 'f1': 0.0, 'tp': 0, 'fp': len(pred_positions), 'fn': 0}

        if len(pred_positions) == 0:
            return {'precision': 1.0, 'recall': 0.0, 'f1': 0.0, 'tp': 0, 'fp': 0, 'fn': len(gt_positions)}

        # Compute distance matrix
        dist_matrix = cdist(gt_positions, pred_positions)

        # Find matches; pairs beyond the threshold get a large cost so the assignment maximises the
        # number of matches within the threshold (as calculate_tracking_metrics does)
        dist_matrix_thresholded = dist_matrix.copy()
        dist_matrix_thresholded[dist_matrix > distance_threshold] = 1e6
        gt_indices, pred_indices = linear_sum_assignment(dist_matrix_thresholded)

        # Count true positives, false positives, and false negatives
        tp = 0
        for gt_idx, pred_idx in zip(gt_indices, pred_indices):
            if dist_matrix[gt_idx, pred_idx] <= distance_threshold:
                tp += 1

        fp = len(pred_positions) - tp
        fn = len(gt_positions) - tp

        # Calculate metrics
        precision = tp / (tp + fp) if tp + fp > 0 else 1.0
        recall = tp / (tp + fn) if tp + fn > 0 else 1.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': tp,
            'fp': fp,
            'fn': fn
        }

    @staticmethod
    def calculate_tracking_metrics(pred_tracks: List[np.ndarray],
                                  gt_tracks: List[np.ndarray],
                                  distance_threshold: float = 5.0) -> Dict:
        """
        Calculate tracking metrics (MOTA, MOTP, ID switches).

        Args:
            pred_tracks: Predicted particle tracks
            gt_tracks: Ground truth particle tracks
            distance_threshold: Maximum distance to consider a detection correct

        Returns:
            Dictionary with metrics
        """
        if not pred_tracks or not gt_tracks:
            return {
                'mota': 0.0,
                'motp': 0.0,
                'id_switches': 0,
                'track_precision': 0.0,
                'track_recall': 0.0,
                'track_f1': 0.0
            }

        # Get number of frames
        gt_frames = max(len(track) for track in gt_tracks)
        pred_frames = max(len(track) for track in pred_tracks)
        num_frames = max(gt_frames, pred_frames)

        # Initialize counters
        total_gt = 0
        total_tp = 0
        total_fp = 0
        total_fn = 0
        total_id_switches = 0
        total_distance = 0.0

        # Previous frame assignments
        prev_assignments = {}

        # Process each frame
        for frame_idx in range(num_frames):
            # Get ground truth positions for this frame
            gt_positions = []
            gt_ids = []

            for track_idx, track in enumerate(gt_tracks):
                if frame_idx < len(track) and not np.isnan(track[frame_idx, 0]):
                    gt_positions.append(track[frame_idx])
                    gt_ids.append(track_idx)

            # Get predicted positions for this frame
            pred_positions = []
            pred_ids = []

            for track_idx, track in enumerate(pred_tracks):
                if frame_idx < len(track) and not np.isnan(track[frame_idx, 0]):
                    pred_positions.append(track[frame_idx])
                    pred_ids.append(track_idx)

            # Count ground truth
            total_gt += len(gt_positions)

            # If no ground truth or predictions, skip this frame
            if len(gt_positions) == 0 or len(pred_positions) == 0:
                if len(gt_positions) > 0:
                    total_fn += len(gt_positions)
                if len(pred_positions) > 0:
                    total_fp += len(pred_positions)
                continue

            # Convert to arrays
            gt_positions = np.array(gt_positions)
            pred_positions = np.array(pred_positions)

            # Compute distance matrix
            dist_matrix = cdist(gt_positions, pred_positions)

            # Set distances beyond threshold to a large value
            dist_matrix_thresholded = dist_matrix.copy()
            dist_matrix_thresholded[dist_matrix > distance_threshold] = 1e6

            # Find matches
            gt_indices, pred_indices = linear_sum_assignment(dist_matrix_thresholded)

            # Process assignments
            current_assignments = {}

            for gt_idx, pred_idx in zip(gt_indices, pred_indices):
                if dist_matrix[gt_idx, pred_idx] <= distance_threshold:
                    # Valid match
                    gt_id = gt_ids[gt_idx]
                    pred_id = pred_ids[pred_idx]

                    current_assignments[gt_id] = pred_id

                    # Check for ID switch
                    if gt_id in prev_assignments and prev_assignments[gt_id] != pred_id:
                        total_id_switches += 1

                    # Update total distance
                    total_distance += dist_matrix[gt_idx, pred_idx]

                    # Count true positive
                    total_tp += 1

            # Count false positives and false negatives
            matched_pred_indices = set(pred_indices[dist_matrix[gt_indices, pred_indices] <= distance_threshold])
            total_fp += len(pred_positions) - len(matched_pred_indices)

            matched_gt_indices = set(gt_indices[dist_matrix[gt_indices, pred_indices] <= distance_threshold])
            total_fn += len(gt_positions) - len(matched_gt_indices)

            # Remember each ground-truth particle's last correspondence, also across frames where it is
            # not matched (CLEAR MOT), so a new identity after a blink counts as a switch
            prev_assignments.update(current_assignments)

        # Calculate metrics
        mota = 1 - (total_fp + total_fn + total_id_switches) / total_gt if total_gt > 0 else 0.0
        motp = total_distance / total_tp if total_tp > 0 else 0.0

        precision = total_tp / (total_tp + total_fp) if total_tp + total_fp > 0 else 0.0
        recall = total_tp / (total_tp + total_fn) if total_tp + total_fn > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

        return {
            'mota': mota,
            'motp': motp,
            'id_switches': total_id_switches,
            'track_precision': precision,
            'track_recall': recall,
            'track_f1': f1,
            'tp': total_tp,
            'fp': total_fp,
            'fn': total_fn
        }

    @staticmethod
    def calculate_all_metrics(pred_positions: List[np.ndarray],
                            gt_positions: List[np.ndarray],
                            pred_tracks: Optional[List[np.ndarray]] = None,
                            gt_tracks: Optional[List[np.ndarray]] = None,
                            distance_threshold: float = 5.0) -> Dict:
        """
        Calculate all detection and tracking metrics.

        Args:
            pred_positions: List of predicted positions for each frame
            gt_positions: List of ground truth positions for each frame
            pred_tracks: Predicted particle tracks
            gt_tracks: Ground truth particle tracks
            distance_threshold: Maximum distance to consider a detection correct

        Returns:
            Dictionary with all metrics
        """
        # Calculate detection metrics for each frame
        detection_metrics_per_frame = []

        for frame_idx, (pred_pos, gt_pos) in enumerate(zip(pred_positions, gt_positions)):
            metrics = MetricsCalculator.calculate_detection_metrics(
                pred_positions=pred_pos,
                gt_positions=gt_pos,
                distance_threshold=distance_threshold
            )
            detection_metrics_per_frame.append(metrics)

        # Calculate mean detection metrics
        mean_precision = np.mean([m['precision'] for m in detection_metrics_per_frame])
        mean_recall = np.mean([m['recall'] for m in detection_metrics_per_frame])
        mean_f1 = np.mean([m['f1'] for m in detection_metrics_per_frame])

        total_tp = sum(m['tp'] for m in detection_metrics_per_frame)
        total_fp = sum(m['fp'] for m in detection_metrics_per_frame)
        total_fn = sum(m['fn'] for m in detection_metrics_per_frame)

        # Calculate tracking metrics if tracks are provided
        tracking_metrics = {}
        if pred_tracks is not None and gt_tracks is not None:
            tracking_metrics = MetricsCalculator.calculate_tracking_metrics(
                pred_tracks=pred_tracks,
                gt_tracks=gt_tracks,
                distance_threshold=distance_threshold
            )

        # Combine metrics
        all_metrics = {
            'detection': {
                'precision': mean_precision,
                'recall': mean_recall,
                'f1': mean_f1,
                'tp': total_tp,
                'fp': total_fp,
                'fn': total_fn
            },
            'per_frame': detection_metrics_per_frame
        }

        if tracking_metrics:
            all_metrics['tracking'] = tracking_metrics

        return all_metrics
