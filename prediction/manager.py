"""
PredictionManager: predictor registry, background prediction jobs and saving results.
"""

import os
import logging
from typing import Callable, Dict, Optional

import numpy as np

from prediction.predictor import ParticlePredictor
from utils.thread_manager import thread_manager

logger = logging.getLogger(__name__)


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
