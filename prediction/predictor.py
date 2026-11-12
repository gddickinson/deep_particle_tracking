"""
ParticlePredictor: runs a trained network on a movie (sliding windows), detects and links particles.
"""

import torch
import numpy as np
import os
import logging
from typing import Dict, List, Tuple, Optional, Union, Callable
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from models.network import create_model
from prediction.postprocess import detect_peaks, link_particles, link_particles_with_embeddings
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

    def _window_length(self) -> int:
        """Frames per network input: the length the model was trained on (checkpoint model_config num_frames),
        else frame_sequence_length."""
        return max(1, int((self.model_config or {}).get('num_frames', self.frame_sequence_length)))

    def _run_model(self, window: np.ndarray):
        """Run the network on a window (T, H, W) of [0, 1] frames.

        Returns (probability map (H, W) for the window's last frame, embedding map (E, H, W) or None).
        """
        x = torch.from_numpy(np.ascontiguousarray(window, dtype=np.float32)).to(self.device)
        x = x[None, :, None]  # [batch 1, time T, channel 1, H, W]
        with torch.no_grad():
            output = self.model(x)
        if isinstance(output, dict):
            positions, embeddings = output['positions'], output.get('embeddings')
        else:
            positions, embeddings = output, None
        positions = positions.cpu().numpy()
        while positions.ndim > 4:  # [batch, time, channel, H, W]: keep the window's last frame
            positions = positions[:, -1]
        prob_map = positions[0, 0]
        if embeddings is not None:
            embeddings = embeddings.cpu().numpy()
            while embeddings.ndim > 4:
                embeddings = embeddings[:, -1]
            embeddings = embeddings[0]
        return prob_map, embeddings

    def predict_frame(self,
                     frame: np.ndarray,
                     return_probability_map: bool = False,
                     context: Optional[np.ndarray] = None) -> Dict:
        """
        Predict particle positions in a single frame.

        Args:
            frame: Input frame (2D array)
            return_probability_map: Whether to return probability map
            context: Optional preceding frames (T-1, H, W), already in [0, 1], so the ConvLSTM sees a sequence

        Returns:
            Dictionary with detection results
        """
        frame = self._preprocess_frame(frame)
        window = frame[None] if context is None else np.concatenate([np.asarray(context, np.float32), frame[None]])
        prob_map, embeddings = self._run_model(window)
        result = {'positions': self._postprocess_detection(prob_map, frame.shape)}
        if embeddings is not None:
            result['embeddings'] = embeddings
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

        Each frame t is predicted from a sliding window of the trained sequence length ending at t
        (frames t-L+1 .. t; shorter at the start of the movie), so the ConvLSTM sees the temporal context it
        was trained with. The stack is normalised once (prediction/preprocess.py).

        Args:
            frames: Input frame sequence (3D array: time, height, width)
            link_particles: Whether to link particles into tracks
            max_distance: Maximum distance for linking particles
            return_probability_maps: Whether to return probability maps

        Returns:
            Dictionary with detection and tracking results
        """
        frames = np.asarray(frames)
        if frames.ndim == 4:
            frames = frames[..., 0]  # first channel if RGB
        frames = normalize_intensity(frames)
        L = self._window_length()

        all_positions, all_prob_maps, all_embeddings = [], [], []
        for t in range(len(frames)):
            prob_map, embeddings = self._run_model(frames[max(0, t - L + 1):t + 1])
            all_positions.append(self._postprocess_detection(prob_map, frames.shape[1:]))
            if return_probability_maps:
                all_prob_maps.append(prob_map)
            if embeddings is not None:
                all_embeddings.append(embeddings)

        tracks = None
        if link_particles and len(all_positions) > 1:
            if len(all_embeddings) > 0:
                tracks = self._link_particles_with_embeddings(all_positions, all_embeddings, max_distance)
            else:
                tracks = self._link_particles(all_positions, max_distance)

        result = {'positions': all_positions, 'tracks': tracks}
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

    def _postprocess_detection(self, prob_map: np.ndarray, frame_shape: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """Threshold + local-maximum detection (prediction/postprocess.py detect_peaks), in frame pixels.

        A super-resolution map (k x frame_shape, Dual Branch / Attentive models) is searched with the NMS radius
        scaled by k and the peaks are mapped back to frame pixel coordinates ((p + 0.5) / k - 0.5, pixel centres
        at integers), giving sub-pixel positions.
        """
        k = prob_map.shape[-1] // frame_shape[-1] if frame_shape is not None else 1
        if k > 1:
            peaks = detect_peaks(prob_map, self.threshold, self.nms_radius * k)
            return (peaks + 0.5) / k - 0.5
        return detect_peaks(prob_map, self.threshold, self.nms_radius)

    def _link_particles(self,
                       positions: List[np.ndarray],
                       max_distance: float) -> List[np.ndarray]:
        """Frame-to-frame Hungarian linking (prediction/postprocess.py link_particles)."""
        return link_particles(positions, max_distance)

    def _link_particles_with_embeddings(self,
                                      positions: List[np.ndarray],
                                      embeddings: List[np.ndarray],
                                      max_distance: float = 20.0) -> List[np.ndarray]:
        """
        Link particles by the network's embeddings (prediction/postprocess.py link_particles_with_embeddings):
        each detection takes the embedding vector at its pixel from that frame's (E, H, W) embedding map; the cost
        is the embedding distance and positions only gate pairs to within max_distance.
        """
        if len(embeddings) != len(positions):
            return self._link_particles(positions, max_distance)
        descriptors = []
        for pos, emb in zip(positions, embeddings):
            yx = np.rint(np.asarray(pos, dtype=float)).astype(int).reshape(-1, 2)
            yx = np.clip(yx, 0, np.array(emb.shape[1:]) - 1)
            descriptors.append(emb[:, yx[:, 0], yx[:, 1]].T if len(yx) else np.zeros((0, emb.shape[0])))
        return link_particles_with_embeddings(positions, descriptors, max_distance)
