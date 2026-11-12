"""
Detection post-processing and frame-to-frame linking shared by the deep-model predictor and the
classical baseline. Torch-free (numpy/scipy only).

detect_peaks: threshold a probability map and keep local maxima -> integer (y, x) pixel positions.
link_particles: Hungarian assignment between consecutive frames within max_distance; tracks that
end are not resumed (no gap closing); tracks shorter than min_length positions are dropped.
link_particles_with_embeddings: the same, but the assignment cost is the distance between per-detection
embedding vectors, with positions only gating which pairs are allowed (within max_distance).
"""

import logging
from typing import List, Optional

import numpy as np
from scipy.ndimage import maximum_filter
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

logger = logging.getLogger(__name__)


def detect_peaks(prob_map: np.ndarray, threshold: float, nms_radius: int) -> np.ndarray:
    """Positions (N, 2) as (y, x) of pixels above threshold that are the maximum within nms_radius pixels
    (square window of 2*nms_radius + 1). Equal values on a plateau are all kept."""
    if prob_map.ndim > 2:
        logger.warning(f"Probability map has {prob_map.ndim} dimensions, flattening to 2D")
        while prob_map.ndim > 2:
            prob_map = prob_map[0]
    binary_map = prob_map > threshold
    # Window of 2r+1 pixels so maxima closer than nms_radius pixels are suppressed (size=nms_radius was a
    # window of r pixels, i.e. a suppression radius of about r/2).
    local_max = maximum_filter(prob_map, size=2 * int(nms_radius) + 1)
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


def link_particles_with_embeddings(positions: List[np.ndarray], descriptors: List[np.ndarray], max_distance: float,
                                   min_length: int = 3,
                                   prev_descriptors: Optional[List[Optional[np.ndarray]]] = None) -> List[np.ndarray]:
    """Link detections by embedding similarity: cost = Euclidean distance between the track's embedding at frame t-1
    and the detection's embedding at frame t; pairs further apart than max_distance pixels are not allowed
    (position gating).

    Args:
        positions: per frame, (N_t, 2) (y, x) detections
        descriptors: per frame, (N_t, E) embedding vectors of those detections
        prev_descriptors: optional, per frame t, (N_{t-1}, E) embeddings of frame t-1's detections computed in the
            same network window as descriptors[t] (the pair the embedding loss trains); entry 0 or None entries
            fall back to the track's last descriptor
    Returns tracks as link_particles does.
    """
    num_frames = len(positions)
    tracks, last_desc, last_idx = [], [], []

    def new_track(t, pos, desc, idx):
        track = np.full((num_frames, 2), np.nan)
        track[t] = pos
        tracks.append(track)
        last_desc.append(desc)
        last_idx.append(idx)

    for i, (pos, desc) in enumerate(zip(positions[0], descriptors[0])):
        new_track(0, pos, desc, i)
    for t in range(1, num_frames):
        cur, cur_desc = np.asarray(positions[t], float), np.asarray(descriptors[t], float)
        if len(cur) == 0:
            continue
        active = [k for k, tr in enumerate(tracks) if not np.isnan(tr[t - 1, 0])]
        if not active:
            for c, (p, d) in enumerate(zip(cur, cur_desc)):
                new_track(t, p, d, c)
            continue
        prev = prev_descriptors[t] if prev_descriptors is not None and t < len(prev_descriptors) else None
        if prev is not None and len(prev):
            # active tracks were detected at t-1 (no gap closing): their detection's same-window embedding
            track_desc = np.asarray(prev, float)[[last_idx[k] for k in active]]
        else:
            track_desc = np.array([last_desc[k] for k in active])
        gate = cdist(np.array([tracks[k][t - 1] for k in active]), cur) <= max_distance
        cost = cdist(track_desc, cur_desc)
        cost[~gate] = 1e6
        rows, cols = linear_sum_assignment(cost)
        assigned = set()
        for r, c in zip(rows, cols):
            if gate[r, c]:
                tracks[active[r]][t] = cur[c]
                last_desc[active[r]] = cur_desc[c]
                last_idx[active[r]] = c
                assigned.add(c)
        for c in range(len(cur)):
            if c not in assigned:
                new_track(t, cur[c], cur_desc[c], c)
    return [t for t in tracks if np.sum(~np.isnan(t[:, 0])) >= min_length]
