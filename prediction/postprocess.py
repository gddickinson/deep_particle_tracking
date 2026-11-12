"""
Detection post-processing and frame-to-frame linking shared by the deep-model predictor and the
classical baseline. Torch-free (numpy/scipy only).

detect_peaks: threshold a probability map and keep local maxima -> integer (y, x) pixel positions.
link_particles: Hungarian assignment between consecutive frames within max_distance; tracks that
end are not resumed (no gap closing); tracks shorter than min_length positions are dropped.
"""

import logging
from typing import List

import numpy as np
from scipy.ndimage import maximum_filter
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

logger = logging.getLogger(__name__)


def detect_peaks(prob_map: np.ndarray, threshold: float, nms_radius: int) -> np.ndarray:
    """Positions (N, 2) as (y, x) of pixels above threshold that are maxima of their neighbourhood."""
    if prob_map.ndim > 2:
        logger.warning(f"Probability map has {prob_map.ndim} dimensions, flattening to 2D")
        while prob_map.ndim > 2:
            prob_map = prob_map[0]
    binary_map = prob_map > threshold
    local_max = maximum_filter(prob_map, size=nms_radius)
    maxima = (prob_map == local_max) & binary_map
    y_coords, x_coords = np.where(maxima)
    if len(y_coords) > 0:
        return np.column_stack([y_coords, x_coords])
    return np.zeros((0, 2), dtype=np.int32)


def link_particles(positions: List[np.ndarray], max_distance: float, min_length: int = 3) -> List[np.ndarray]:
    """Link per-frame (y, x) positions into tracks: list of (num_frames, 2) arrays, NaN where absent."""
    num_frames = len(positions)
    tracks = []
    for pos in positions[0]:
        track = np.full((num_frames, 2), np.nan)
        track[0] = pos
        tracks.append(track)

    for frame_idx in range(1, num_frames):
        curr_positions = positions[frame_idx]
        if len(curr_positions) == 0:
            continue
        active_tracks, active_track_indices = [], []
        for track_idx, track in enumerate(tracks):
            if not np.isnan(track[frame_idx - 1, 0]):
                active_tracks.append(track[frame_idx - 1])
                active_track_indices.append(track_idx)

        if len(active_tracks) == 0:
            for pos in curr_positions:
                track = np.full((num_frames, 2), np.nan)
                track[frame_idx] = pos
                tracks.append(track)
            continue

        dist_matrix = cdist(np.array(active_tracks), curr_positions)
        dist_matrix[dist_matrix > max_distance] = 1e6
        track_indices, particle_indices = linear_sum_assignment(dist_matrix)
        assigned = set()
        for track_idx, particle_idx in zip(track_indices, particle_indices):
            if dist_matrix[track_idx, particle_idx] <= max_distance:
                tracks[active_track_indices[track_idx]][frame_idx] = curr_positions[particle_idx]
                assigned.add(particle_idx)
        for particle_idx in range(len(curr_positions)):
            if particle_idx not in assigned:
                track = np.full((num_frames, 2), np.nan)
                track[frame_idx] = curr_positions[particle_idx]
                tracks.append(track)

    return [t for t in tracks if np.sum(~np.isnan(t[:, 0])) >= min_length]
