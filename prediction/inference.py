"""
Prediction and inference for particle tracking models.
"""

import torch
import numpy as np
import os
import logging
from typing import Dict, List, Tuple, Optional, Union, Callable
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from models.network import create_model
from utils.device_manager import device_manager
from utils.thread_manager import thread_manager

logger = logging.getLogger(__name__)


class ParticlePredictor:
    """Class for making predictions with trained models."""
    
    def __init__(self, 
                 model_path: str,
                 model_type: Optional[str] = None,
                 model_config: Optional[Dict] = None,
                 threshold: float = 0.5,
                 nms_radius: int = 3):
        """
        Initialize the particle predictor.
        
        Args:
            model_path: Path to the trained model checkpoint
            model_type: Type of model ('simple', 'dual', 'attentive')
            model_config: Model configuration dictionary
            threshold: Threshold for detection
            nms_radius: Non-maximum suppression radius in pixels
        """
        self.model_path = model_path
        self.model_type = model_type
        self.model_config = model_config
        self.threshold = threshold
        self.nms_radius = nms_radius
        
        # Set up device
        self.device = device_manager.get_device()
        
        # Load model
        self.model = self._load_model()
        
    def _load_model(self) -> torch.nn.Module:
        """
        Load the model from checkpoint.
        
        Returns:
            Loaded PyTorch model
        """
        # Load checkpoint
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # Extract model type and config if not provided
        if self.model_type is None:
            self.model_type = checkpoint.get('model_type', 'attentive')
            
        if self.model_config is None:
            self.model_config = checkpoint.get('model_config', {})
            
        # Create model
        model = create_model(self.model_type, **self.model_config)
        
        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Move model to device
        model = model.to(self.device)
        
        # Set to evaluation mode
        model.eval()
        
        logger.info(f"Model loaded from {self.model_path}")
        
        return model
    
    def predict_frame(self, 
                     frame: np.ndarray,
                     return_probability_map: bool = False) -> Dict:
        """
        Predict particle positions in a single frame.
        
        Args:
            frame: Input frame (2D array)
            return_probability_map: Whether to return probability map
            
        Returns:
            Dictionary with detection results
        """
        # Ensure frame is properly shaped and typed
        frame = self._preprocess_frame(frame)
        
        # Convert to tensor and add batch & channel dimensions
        x = torch.from_numpy(frame).float().to(self.device)
        x = x.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        
        # Forward pass
        with torch.no_grad():
            output = self.model(x)
            
        # Process output based on model type
        if isinstance(output, dict):
            # Dual branch or attentive model
            prob_map = output['positions'].cpu().numpy()[0, 0]
        else:
            # Simple model
            prob_map = output.cpu().numpy()[0, 0]
            
        # Apply threshold and non-maximum suppression
        positions = self._postprocess_detection(prob_map)
        
        # Prepare result
        result = {
            'positions': positions
        }
        
        if return_probability_map:
            result['probability_map'] = prob_map
            
        return result
    
    def predict_sequence(self, 
                        frames: np.ndarray,
                        link_particles: bool = True,
                        max_distance: float = 20.0,
                        return_probability_maps: bool = False) -> Dict:
        """
        Predict particle positions and tracks in a sequence of frames.
        
        Args:
            frames: Input frame sequence (3D array: time, height, width)
            link_particles: Whether to link particles into tracks
            max_distance: Maximum distance for linking particles
            return_probability_maps: Whether to return probability maps
            
        Returns:
            Dictionary with detection and tracking results
        """
        # Process each frame
        all_positions = []
        all_prob_maps = []
        
        for frame_idx, frame in enumerate(frames):
            # Predict positions
            result = self.predict_frame(frame, return_probability_map=return_probability_maps)
            
            # Store positions
            all_positions.append(result['positions'])
            
            # Store probability maps if requested
            if return_probability_maps:
                all_prob_maps.append(result['probability_map'])
                
        # Link particles into tracks
        tracks = None
        if link_particles and len(all_positions) > 1:
            tracks = self._link_particles(all_positions, max_distance)
            
        # Prepare result
        result = {
            'positions': all_positions,
            'tracks': tracks
        }
        
        if return_probability_maps:
            result['probability_maps'] = all_prob_maps
            
        return result
    
    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess a frame for input to the model.
        
        Args:
            frame: Input frame
            
        Returns:
            Preprocessed frame
        """
        # Ensure frame is 2D
        if frame.ndim > 2:
            frame = frame[..., 0]  # Take first channel if RGB
            
        # Convert to float32
        frame = frame.astype(np.float32)
        
        # Normalize to [0, 1]
        if frame.max() > 1.0:
            frame = frame / 255.0
            
        return frame
    
    def _postprocess_detection(self, prob_map: np.ndarray) -> np.ndarray:
        """
        Postprocess detection probability map to get particle positions.
        
        Args:
            prob_map: Probability map from model
            
        Returns:
            Array of particle positions (N, 2)
        """
        # Apply threshold
        binary_map = prob_map > self.threshold
        
        # Find local maxima
        from scipy.ndimage import maximum_filter
        local_max = maximum_filter(prob_map, size=self.nms_radius)
        maxima = (prob_map == local_max) & binary_map
        
        # Get positions
        y_coords, x_coords = np.where(maxima)
        positions = np.column_stack([x_coords, y_coords])
        
        return positions
    
    def _link_particles(self, 
                       positions: List[np.ndarray],
                       max_distance: float) -> List[np.ndarray]:
        """
        Link particles across frames into tracks.
        
        Args:
            positions: List of position arrays, one per frame
            max_distance: Maximum distance for linking particles
            
        Returns:
            List of track arrays, each with shape (num_frames, 2)
        """
        num_frames = len(positions)
        
        # Initialize tracks with particles from first frame
        tracks = []
        for pos in positions[0]:
            # Create track with first position and NaN for remaining frames
            track = np.full((num_frames, 2), np.nan)
            track[0] = pos
            tracks.append(track)
            
        # Process each subsequent frame
        for frame_idx in range(1, num_frames):
            # Current positions
            curr_positions = positions[frame_idx]
            
            # Skip if no particles in current frame
            if len(curr_positions) == 0:
                continue
                
            # Find active tracks (not terminated)
            active_tracks = []
            active_track_indices = []
            
            for track_idx, track in enumerate(tracks):
                # Check if track is active in previous frame
                if not np.isnan(track[frame_idx-1, 0]):
                    active_tracks.append(track[frame_idx-1])
                    active_track_indices.append(track_idx)
                    
            # Skip if no active tracks
            if len(active_tracks) == 0:
                # Add new tracks for all particles
                for pos in curr_positions:
                    # Create track with NaN for previous frames and current position
                    track = np.full((num_frames, 2), np.nan)
                    track[frame_idx] = pos
                    tracks.append(track)
                    
                continue
                
            # Convert to array
            active_tracks = np.array(active_tracks)
            
            # Compute distance matrix
            dist_matrix = cdist(active_tracks, curr_positions)
            
            # Set distances beyond max_distance to a large value
            dist_matrix[dist_matrix > max_distance] = 1e6
            
            # Solve assignment problem
            track_indices, particle_indices = linear_sum_assignment(dist_matrix)
            
            # Mark assigned particles
            assigned_particles = set()
            
            # Update assigned tracks
            for track_idx, particle_idx in zip(track_indices, particle_indices):
                # Check if distance is within threshold
                if dist_matrix[track_idx, particle_idx] <= max_distance:
                    # Update track
                    tracks[active_track_indices[track_idx]][frame_idx] = curr_positions[particle_idx]
                    assigned_particles.add(particle_idx)
                    
            # Create new tracks for unassigned particles
            for particle_idx in range(len(curr_positions)):
                if particle_idx not in assigned_particles:
                    # Create track with NaN for previous frames and current position
                    track = np.full((num_frames, 2), np.nan)
                    track[frame_idx] = curr_positions[particle_idx]
                    tracks.append(track)
                    
        # Remove tracks that are too short
        min_length = 3  # Minimum track length
        valid_tracks = []
        
        for track in tracks:
            # Count non-NaN positions
            valid_positions = np.sum(~np.isnan(track[:, 0]))
            
            if valid_positions >= min_length:
                valid_tracks.append(track)
                
        return valid_tracks


class PredictionManager:
    """Manager for handling prediction jobs with thread management."""
    
    def __init__(self, models_dir: str = 'models', results_dir: str = 'results'):
        """
        Initialize the prediction manager.
        
        Args:
            models_dir: Directory with model checkpoints
            results_dir: Directory to save prediction results
        """
        self.models_dir = models_dir
        self.results_dir = results_dir
        
        # Ensure directories exist
        os.makedirs(results_dir, exist_ok=True)
        
        # Current predictors
        self.predictors = {}
        self.prediction_results = {}
        
    def create_predictor(self,
                        model_id: str,
                        model_path: Optional[str] = None,
                        model_type: Optional[str] = None,
                        model_config: Optional[Dict] = None,
                        threshold: float = 0.5,
                        nms_radius: int = 3) -> ParticlePredictor:
        """
        Create a particle predictor.
        
        Args:
            model_id: Unique ID for the model
            model_path: Path to model checkpoint
            model_type: Model type
            model_config: Model configuration
            threshold: Detection threshold
            nms_radius: Non-maximum suppression radius
            
        Returns:
            ParticlePredictor object
        """
        # Resolve model path
        if model_path is None:
            model_path = os.path.join(self.models_dir, model_id, 'best.pth')
            
        # Create predictor
        predictor = ParticlePredictor(
            model_path=model_path,
            model_type=model_type,
            model_config=model_config,
            threshold=threshold,
            nms_radius=nms_radius
        )
        
        # Store predictor
        self.predictors[model_id] = predictor
        
        return predictor
    
    def get_predictor(self, model_id: str) -> Optional[ParticlePredictor]:
        """
        Get an existing predictor.
        
        Args:
            model_id: ID of the predictor
            
        Returns:
            ParticlePredictor object or None if not found
        """
        return self.predictors.get(model_id)
    
    def predict_frame(self,
                     model_id: str,
                     frame: np.ndarray,
                     prediction_id: Optional[str] = None,
                     return_probability_map: bool = False,
                     save_results: bool = False) -> Dict:
        """
        Predict particles in a single frame.
        
        Args:
            model_id: ID of the model to use
            frame: Input frame
            prediction_id: Unique ID for the prediction
            return_probability_map: Whether to return probability map
            save_results: Whether to save results to disk
            
        Returns:
            Dictionary with prediction results
        """
        # Get predictor
        predictor = self.get_predictor(model_id)
        if predictor is None:
            raise ValueError(f"Predictor with ID {model_id} not found")
            
        # Generate prediction ID if not provided
        if prediction_id is None:
            prediction_id = f"{model_id}_frame_{len(self.prediction_results)}"
            
        # Make prediction
        result = predictor.predict_frame(
            frame=frame,
            return_probability_map=return_probability_map
        )
        
        # Store result
        self.prediction_results[prediction_id] = result
        
        # Save results if requested
        if save_results:
            self._save_prediction(prediction_id, result)
            
        return result
    
    def predict_sequence(self,
                        model_id: str,
                        frames: np.ndarray,
                        prediction_id: Optional[str] = None,
                        link_particles: bool = True,
                        max_distance: float = 20.0,
                        return_probability_maps: bool = False,
                        save_results: bool = False,
                        callback: Optional[Callable] = None) -> str:
        """
        Predict particles in a sequence of frames (in a background thread).
        
        Args:
            model_id: ID of the model to use
            frames: Input frame sequence
            prediction_id: Unique ID for the prediction
            link_particles: Whether to link particles into tracks
            max_distance: Maximum distance for linking particles
            return_probability_maps: Whether to return probability maps
            save_results: Whether to save results to disk
            callback: Callback function to call after prediction
            
        Returns:
            Task ID of the prediction job
        """
        # Get predictor
        predictor = self.get_predictor(model_id)
        if predictor is None:
            raise ValueError(f"Predictor with ID {model_id} not found")
            
        # Generate prediction ID if not provided
        if prediction_id is None:
            prediction_id = f"{model_id}_sequence_{len(self.prediction_results)}"
            
        # Define prediction function
        def predict_job():
            try:
                logger.info(f"Starting prediction job {prediction_id}")
                
                # Make prediction
                result = predictor.predict_sequence(
                    frames=frames,
                    link_particles=link_particles,
                    max_distance=max_distance,
                    return_probability_maps=return_probability_maps
                )
                
                # Store result
                self.prediction_results[prediction_id] = result
                
                # Save results if requested
                if save_results:
                    self._save_prediction(prediction_id, result)
                    
                logger.info(f"Prediction job {prediction_id} completed")
                
                # Call callback if provided
                if callback:
                    callback(prediction_id, result)
                    
                return result
                
            except Exception as e:
                logger.error(f"Error in prediction job {prediction_id}: {str(e)}")
                raise e
                
        # Start prediction in a background thread
        task_id = thread_manager.submit_task(
            task_id=f"predict_{prediction_id}",
            func=predict_job
        )
        
        logger.info(f"Prediction job {prediction_id} started with task ID {task_id}")
        
        return task_id
    
    def get_prediction_status(self, prediction_id: str) -> Dict:
        """
        Get the status of a prediction job.
        
        Args:
            prediction_id: ID of the prediction
            
        Returns:
            Status dictionary
        """
        task_id = f"predict_{prediction_id}"
        task_status = thread_manager.get_task_status(task_id)
        
        # Check if prediction result is available
        result_available = prediction_id in self.prediction_results
        
        status = {
            'task_status': task_status['status'],
            'prediction_id': prediction_id,
            'result_available': result_available
        }
        
        return status
    
    def get_prediction_result(self, prediction_id: str) -> Optional[Dict]:
        """
        Get the result of a prediction job.
        
        Args:
            prediction_id: ID of the prediction
            
        Returns:
            Prediction result or None if not available
        """
        return self.prediction_results.get(prediction_id)
    
    def _save_prediction(self, prediction_id: str, result: Dict):
        """
        Save prediction results to disk.
        
        Args:
            prediction_id: ID of the prediction
            result: Prediction result dictionary
        """
        # Create directory for this prediction
        save_dir = os.path.join(self.results_dir, prediction_id)
        os.makedirs(save_dir, exist_ok=True)
        
        # Save positions
        if 'positions' in result:
            positions_path = os.path.join(save_dir, 'positions.npy')
            np.save(positions_path, result['positions'])
            
        # Save tracks
        if 'tracks' in result and result['tracks'] is not None:
            tracks_path = os.path.join(save_dir, 'tracks.npy')
            np.save(tracks_path, result['tracks'])
            
        # Save probability maps
        if 'probability_map' in result:
            prob_map_path = os.path.join(save_dir, 'probability_map.npy')
            np.save(prob_map_path, result['probability_map'])
            
        if 'probability_maps' in result:
            prob_maps_path = os.path.join(save_dir, 'probability_maps.npy')
            np.save(prob_maps_path, result['probability_maps'])
            
        logger.info(f"Prediction results saved to {save_dir}")
        
    def cleanup(self):
        """Clean up resources used by the prediction manager."""
        # Clear predictors and results
        self.predictors.clear()
        self.prediction_results.clear()


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
        
        # Find matches
        gt_indices, pred_indices = linear_sum_assignment(dist_matrix)
        
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
            
            # Update previous assignments
            prev_assignments = current_assignments
            
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
