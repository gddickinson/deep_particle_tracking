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
from prediction.postprocess import detect_peaks, link_particles
from prediction.preprocess import normalize_intensity
from utils.device_manager import device_manager
from utils.thread_manager import thread_manager

logger = logging.getLogger(__name__)


class ParticlePredictor:
    # Add a new parameter for sequence length
    def __init__(self,
                 model_path: str,
                 model_type: Optional[str] = None,
                 model_config: Optional[Dict] = None,
                 threshold: float = 0.5,
                 nms_radius: int = 3,
                 frame_sequence_length: int = 5):
        """
        Initialize the particle predictor.

        Args:
            model_path: Path to the trained model checkpoint
            model_type: Type of model ('simple', 'dual', 'attentive')
            model_config: Model configuration dictionary
            threshold: Threshold for detection
            nms_radius: Non-maximum suppression radius in pixels
            frame_sequence_length: Number of frames to process at once
        """
        self.model_path = model_path
        self.model_type = model_type
        self.model_config = model_config
        self.threshold = threshold
        self.nms_radius = nms_radius
        self.frame_sequence_length = frame_sequence_length

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
            # New model format: dictionary with 'positions' and possibly 'embeddings'
            positions = output['positions'].cpu().numpy()
            # Handle case where positions have sequence dimension
            if positions.ndim > 4:  # [batch, sequence, channel, height, width]
                positions = positions[:, 0]  # Take first sequence element
            prob_map = positions[0, 0]  # Take first batch, first channel
            embeddings = output.get('embeddings', None)
        else:
            # Legacy model format: direct tensor output
            positions = output.cpu().numpy()
            # Handle case where positions have sequence dimension
            if positions.ndim > 4:  # [batch, sequence, channel, height, width]
                positions = positions[:, 0]  # Take first sequence element
            prob_map = positions[0, 0]  # Take first batch, first channel
            embeddings = None

        # Apply threshold and non-maximum suppression
        positions = self._postprocess_detection(prob_map)

        # Prepare result
        result = {
            'positions': positions
        }

        if embeddings is not None:
            result['embeddings'] = embeddings.cpu().numpy()[0]

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
        all_embeddings = []

        for frame_idx, frame in enumerate(frames):
            # Predict positions
            result = self.predict_frame(frame, return_probability_map=return_probability_maps)

            # Store positions
            all_positions.append(result['positions'])

            # Store probability maps if requested
            if return_probability_maps:
                all_prob_maps.append(result.get('probability_map', None))

            # Store embeddings if available
            if 'embeddings' in result:
                all_embeddings.append(result['embeddings'])

        # Link particles into tracks
        tracks = None
        if link_particles and len(all_positions) > 1:
            if len(all_embeddings) > 0:
                # Use embeddings for better tracking
                tracks = self._link_particles_with_embeddings(all_positions, all_embeddings, max_distance)
            else:
                # Fall back to standard distance-based tracking
                tracks = self._link_particles(all_positions, max_distance)

        # Prepare result
        result = {
            'positions': all_positions,
            'tracks': tracks
        }

        if return_probability_maps:
            result['probability_maps'] = all_prob_maps

        if len(all_embeddings) > 0:
            result['embeddings'] = all_embeddings

        return result

    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """2D float32 frame in [0, 1] (first channel if RGB), scaled by prediction/preprocess.py
        normalize_intensity (8-bit: /255; 16-bit and other ranges: percentiles; [0, 1] floats unchanged)."""
        if frame.ndim > 2:
            frame = frame[..., 0]  # Take first channel if RGB
        return normalize_intensity(frame)

    def _postprocess_detection(self, prob_map: np.ndarray) -> np.ndarray:
        """Threshold + local-maximum detection (prediction/postprocess.py detect_peaks)."""
        return detect_peaks(prob_map, self.threshold, self.nms_radius)

    def _link_particles(self,
                       positions: List[np.ndarray],
                       max_distance: float) -> List[np.ndarray]:
        """Frame-to-frame Hungarian linking (prediction/postprocess.py link_particles)."""
        return link_particles(positions, max_distance)

    def _link_particles_with_embeddings(self,
                                      positions: List[np.ndarray],
                                      embeddings: List[np.ndarray],
                                      max_distance: float = 20.0,
                                      embedding_weight: float = 0.5) -> List[np.ndarray]:
        """
        Link particles across frames using both position and embedding similarity.

        Args:
            positions: List of position arrays, one per frame
            embeddings: List of embedding arrays, one per frame
            max_distance: Maximum spatial distance for linking
            embedding_weight: Weight of embedding similarity vs. spatial distance

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

        # For frames without corresponding embeddings, fall back to position-only linking
        if len(embeddings) != num_frames:
            return self._link_particles(positions, max_distance)

        # Process each subsequent frame
        for frame_idx in range(1, num_frames):
            # Current positions and embeddings
            curr_positions = positions[frame_idx]
            curr_embeddings = embeddings[frame_idx]

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

                # Make prediction with better error handling
                try:
                    result = predictor.predict_sequence(
                        frames=frames,
                        link_particles=link_particles,
                        max_distance=max_distance,
                        return_probability_maps=return_probability_maps
                    )
                except Exception as e:
                    logger.error(f"Error in prediction: {str(e)}")
                    import traceback
                    logger.error(f"Traceback: {traceback.format_exc()}")
                    raise e

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


# MetricsCalculator moved to prediction/metrics.py (torch-free); re-exported here for old imports.
from prediction.metrics import MetricsCalculator  # noqa: E402,F401
