"""
Motion models for simulating particle trajectories with different dynamics.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Union, Callable
import logging

logger = logging.getLogger(__name__)


class MotionModel:
    """Base class for motion models."""

    def apply(self,
             initial_positions: List[Tuple[float, float]],
             num_frames: int) -> List[List[Tuple[float, float]]]:
        """
        Apply motion model to generate trajectories.

        Args:
            initial_positions: List of initial (y, x) positions
            num_frames: Number of frames to simulate

        Returns:
            List of trajectories, where each trajectory is a list of (y, x) positions
        """
        raise NotImplementedError("Subclasses must implement the apply method")


class BrownianMotion(MotionModel):
    """Brownian motion (random walk) model."""

    def __init__(self,
                 diffusion_coefficient: float = 0.1,  # µm²/s
                 frame_interval: float = 0.1,  # seconds
                 pixel_size: float = 0.1,  # µm/pixel
                 confinement_radius: Optional[float] = None):  # pixels
        """
        Initialize the Brownian motion model.

        Args:
            diffusion_coefficient: Diffusion coefficient (D) in µm²/s
            frame_interval: Time between frames in seconds
            pixel_size: Physical size of a pixel in µm
            confinement_radius: Radius of confinement in pixels (None for no confinement)
        """
        self.diffusion_coefficient = diffusion_coefficient
        self.frame_interval = frame_interval
        self.pixel_size = pixel_size
        self.confinement_radius = confinement_radius

        # Calculate step size standard deviation
        # MSD = 4*D*t for 2D Brownian motion
        self.step_std = np.sqrt(2 * diffusion_coefficient * frame_interval) / pixel_size

    def apply(self,
             initial_positions: List[Tuple[float, float]],
             num_frames: int,
             image_size: Optional[Tuple[int, int]] = None) -> List[List[Tuple[float, float]]]:
        """
        Apply Brownian motion to generate trajectories.

        Args:
            initial_positions: List of initial (y, x) positions
            num_frames: Number of frames to simulate
            image_size: Optional image size for boundary conditions (height, width)

        Returns:
            List of trajectories, where each trajectory is a list of (y, x) positions
        """
        trajectories = []

        for y0, x0 in initial_positions:
            # Start trajectory with initial position
            trajectory = [(y0, x0)]

            # Apply Brownian steps for each frame
            for _ in range(1, num_frames):
                prev_y, prev_x = trajectory[-1]

                # Random step with Gaussian distribution
                dy = np.random.normal(0, self.step_std)
                dx = np.random.normal(0, self.step_std)

                # Apply confinement if specified
                if self.confinement_radius is not None:
                    # Distance from initial position
                    current_radius = np.sqrt((prev_y - y0)**2 + (prev_x - x0)**2)

                    if current_radius + np.sqrt(dy**2 + dx**2) > self.confinement_radius:
                        # Scale down the step to keep within confinement
                        scale_factor = max(0, (self.confinement_radius - current_radius) / np.sqrt(dy**2 + dx**2))
                        dy *= scale_factor
                        dx *= scale_factor

                # New position
                new_y = prev_y + dy
                new_x = prev_x + dx

                # Apply boundary conditions if image size is provided
                if image_size is not None:
                    height, width = image_size

                    # Reflective boundary conditions
                    if new_y < 0:
                        new_y = -new_y
                    elif new_y >= height:
                        new_y = 2 * height - new_y - 1

                    if new_x < 0:
                        new_x = -new_x
                    elif new_x >= width:
                        new_x = 2 * width - new_x - 1

                trajectory.append((new_y, new_x))

            trajectories.append(trajectory)

        return trajectories


class DirectedMotion(MotionModel):
    """Directed motion model with optional diffusion."""

    def __init__(self,
                 velocity_range: Tuple[float, float] = (0.5, 2.0),  # µm/s
                 direction_change_prob: float = 0.1,
                 diffusion_coefficient: float = 0.01,  # µm²/s
                 frame_interval: float = 0.1,  # seconds
                 pixel_size: float = 0.1):  # µm/pixel
        """
        Initialize the directed motion model.

        Args:
            velocity_range: Range of velocities in µm/s (min, max)
            direction_change_prob: Probability of changing direction per frame
            diffusion_coefficient: Diffusion coefficient (D) in µm²/s for random component
            frame_interval: Time between frames in seconds
            pixel_size: Physical size of a pixel in µm
        """
        self.velocity_range = velocity_range
        self.direction_change_prob = direction_change_prob
        self.diffusion_coefficient = diffusion_coefficient
        self.frame_interval = frame_interval
        self.pixel_size = pixel_size

        # Convert velocity from µm/s to pixels/frame
        self.velocity_to_pixels = frame_interval / pixel_size

        # Calculate diffusion step size standard deviation
        self.diff_step_std = np.sqrt(2 * diffusion_coefficient * frame_interval) / pixel_size

    def apply(self,
             initial_positions: List[Tuple[float, float]],
             num_frames: int,
             image_size: Optional[Tuple[int, int]] = None) -> List[List[Tuple[float, float]]]:
        """
        Apply directed motion to generate trajectories.

        Args:
            initial_positions: List of initial (y, x) positions
            num_frames: Number of frames to simulate
            image_size: Optional image size for boundary conditions (height, width)

        Returns:
            List of trajectories, where each trajectory is a list of (y, x) positions
        """
        trajectories = []

        # Initialize velocities and directions for each particle
        velocities = []
        directions = []

        for _ in initial_positions:
            velocity = np.random.uniform(*self.velocity_range)
            direction = np.random.uniform(0, 2*np.pi)
            velocities.append(velocity)
            directions.append(direction)

        for i, (y0, x0) in enumerate(initial_positions):
            # Start trajectory with initial position
            trajectory = [(y0, x0)]

            velocity = velocities[i]
            direction = directions[i]

            # Apply directed motion for each frame
            for _ in range(1, num_frames):
                prev_y, prev_x = trajectory[-1]

                # Check if direction should change
                if np.random.random() < self.direction_change_prob:
                    # Change direction slightly
                    direction += np.random.normal(0, np.pi/4)

                # Calculate directed movement
                velocity_pixels = velocity * self.velocity_to_pixels
                dy_directed = velocity_pixels * np.sin(direction)  # Note: y-component uses sin
                dx_directed = velocity_pixels * np.cos(direction)  # x-component uses cos

                # Add random diffusion component
                if self.diffusion_coefficient > 0:
                    dy_random = np.random.normal(0, self.diff_step_std)
                    dx_random = np.random.normal(0, self.diff_step_std)
                else:
                    dy_random = dx_random = 0

                # Combined movement
                dy = dy_directed + dy_random
                dx = dx_directed + dx_random

                # New position
                new_y = prev_y + dy
                new_x = prev_x + dx

                # Apply boundary conditions if image size is provided
                if image_size is not None:
                    height, width = image_size

                    # Reflective boundary conditions with direction change
                    if new_y < 0:
                        new_y = -new_y
                        direction = -direction  # Reflect vertically
                    elif new_y >= height:
                        new_y = 2 * height - new_y - 1
                        direction = -direction  # Reflect vertically

                    if new_x < 0:
                        new_x = -new_x
                        direction = np.pi - direction  # Reflect horizontally
                    elif new_x >= width:
                        new_x = 2 * width - new_x - 1
                        direction = np.pi - direction  # Reflect horizontally

                trajectory.append((new_y, new_x))

            trajectories.append(trajectory)

        return trajectories


class AnomalousDiffusion(MotionModel):
    """Anomalous diffusion model with time-dependent MSD."""

    def __init__(self,
                 diffusion_coefficient: float = 0.1,  # µm²/s^alpha
                 alpha: float = 0.8,  # Anomalous exponent (alpha < 1: subdiffusion, alpha > 1: superdiffusion)
                 frame_interval: float = 0.1,  # seconds
                 pixel_size: float = 0.1):  # µm/pixel
        """
        Initialize the anomalous diffusion model.

        Args:
            diffusion_coefficient: Generalized diffusion coefficient in µm²/s^alpha
            alpha: Anomalous exponent (alpha < 1: subdiffusion, alpha > 1: superdiffusion)
            frame_interval: Time between frames in seconds
            pixel_size: Physical size of a pixel in µm
        """
        self.diffusion_coefficient = diffusion_coefficient
        self.alpha = alpha
        self.frame_interval = frame_interval
        self.pixel_size = pixel_size

    def apply(self,
             initial_positions: List[Tuple[float, float]],
             num_frames: int,
             image_size: Optional[Tuple[int, int]] = None) -> List[List[Tuple[float, float]]]:
        """
        Apply anomalous diffusion to generate trajectories.

        Args:
            initial_positions: List of initial (y, x) positions
            num_frames: Number of frames to simulate
            image_size: Optional image size for boundary conditions (height, width)

        Returns:
            List of trajectories, where each trajectory is a list of (y, x) positions
        """
        trajectories = []

        for y0, x0 in initial_positions:
            # Start trajectory with initial position
            trajectory = [(y0, x0)]

            # For anomalous diffusion, we generate the whole trajectory at once
            # because steps are correlated in time

            # For subdiffusion (alpha < 1), we use fractional Brownian motion
            # For superdiffusion (alpha > 1), we approximate with a Levy flight-like process

            if self.alpha < 1:
                # Approximate fractional Brownian motion for subdiffusion
                # using a simplified power-law step size distribution

                for frame in range(1, num_frames):
                    prev_y, prev_x = trajectory[-1]

                    # Time-dependent step size
                    time = frame * self.frame_interval
                    step_std = np.sqrt(4 * self.diffusion_coefficient * time**self.alpha) / self.pixel_size

                    # Calculate step based on current frame's MSD
                    if frame > 1:
                        last_step_std = np.sqrt(4 * self.diffusion_coefficient *
                                             ((frame-1) * self.frame_interval)**self.alpha) / self.pixel_size
                        # Incremental step std for correlation
                        step_std = np.sqrt(step_std**2 - last_step_std**2)

                    # Random step with Gaussian distribution
                    dy = np.random.normal(0, step_std / np.sqrt(2))
                    dx = np.random.normal(0, step_std / np.sqrt(2))

                    # New position
                    new_y = prev_y + dy
                    new_x = prev_x + dx

                    # Apply boundary conditions if image size is provided
                    if image_size is not None:
                        height, width = image_size

                        # Reflective boundary conditions
                        if new_y < 0:
                            new_y = -new_y
                        elif new_y >= height:
                            new_y = 2 * height - new_y - 1

                        if new_x < 0:
                            new_x = -new_x
                        elif new_x >= width:
                            new_x = 2 * width - new_x - 1

                    trajectory.append((new_y, new_x))

            else:  # alpha >= 1, superdiffusion
                # Approximate superdiffusion with a combination of
                # random walk and occasional long jumps

                for _ in range(1, num_frames):
                    prev_y, prev_x = trajectory[-1]

                    # Normal diffusion component
                    normal_std = np.sqrt(2 * self.frame_interval) / self.pixel_size
                    dy_normal = np.random.normal(0, normal_std)
                    dx_normal = np.random.normal(0, normal_std)

                    # Superdiffusive component (long jumps with power-law tail)
                    # Probability of long jump decreases with alpha
                    if np.random.random() < 0.1 * (self.alpha - 1):
                        # Power-law distributed step length
                        # Using Pareto distribution for long tail
                        # Shape parameter controls the tail: smaller shape = heavier tail
                        shape = 1.0 / (self.alpha - 0.5)  # Heuristic mapping from alpha to shape
                        scale = self.diffusion_coefficient * self.frame_interval / self.pixel_size

                        jump_length = np.random.pareto(shape) * scale
                        jump_angle = np.random.uniform(0, 2*np.pi)

                        dy_super = jump_length * np.sin(jump_angle)  # y-component uses sin
                        dx_super = jump_length * np.cos(jump_angle)  # x-component uses cos
                    else:
                        dy_super = dx_super = 0

                    # Combined movement
                    dy = dy_normal + dy_super
                    dx = dx_normal + dx_super

                    # New position
                    new_y = prev_y + dy
                    new_x = prev_x + dx

                    # Apply boundary conditions if image size is provided
                    if image_size is not None:
                        height, width = image_size

                        # Reflective boundary conditions
                        if new_y < 0:
                            new_y = -new_y
                        elif new_y >= height:
                            new_y = 2 * height - new_y - 1

                        if new_x < 0:
                            new_x = -new_x
                        elif new_x >= width:
                            new_x = 2 * width - new_x - 1

                    trajectory.append((new_y, new_x))

            trajectories.append(trajectory)

        return trajectories


class ConfinedDiffusion(MotionModel):
    """Confined diffusion model within potential wells."""

    def __init__(self,
                 diffusion_coefficient: float = 0.1,  # µm²/s
                 confinement_strength: float = 1.0,  # strength of harmonic potential
                 confinement_radius: float = 10.0,  # pixels
                 frame_interval: float = 0.1,  # seconds
                 pixel_size: float = 0.1):  # µm/pixel
        """
        Initialize the confined diffusion model.

        Args:
            diffusion_coefficient: Diffusion coefficient (D) in µm²/s
            confinement_strength: Strength of the harmonic potential
            confinement_radius: Characteristic radius of confinement in pixels
            frame_interval: Time between frames in seconds
            pixel_size: Physical size of a pixel in µm
        """
        self.diffusion_coefficient = diffusion_coefficient
        self.confinement_strength = confinement_strength
        self.confinement_radius = confinement_radius
        self.frame_interval = frame_interval
        self.pixel_size = pixel_size

        # Calculate step size standard deviation
        self.step_std = np.sqrt(2 * diffusion_coefficient * frame_interval) / pixel_size

    def apply(self,
             initial_positions: List[Tuple[float, float]],
             num_frames: int,
             image_size: Optional[Tuple[int, int]] = None) -> List[List[Tuple[float, float]]]:
        """
        Apply confined diffusion to generate trajectories.

        Args:
            initial_positions: List of initial (y, x) positions
            num_frames: Number of frames to simulate
            image_size: Optional image size for boundary conditions (height, width)

        Returns:
            List of trajectories, where each trajectory is a list of (y, x) positions
        """
        trajectories = []

        for y0, x0 in initial_positions:
            # Start trajectory with initial position
            trajectory = [(y0, x0)]

            # Apply confined diffusion for each frame
            for _ in range(1, num_frames):
                prev_y, prev_x = trajectory[-1]

                # Calculate displacement from the center of confinement
                dy_from_center = prev_y - y0
                dx_from_center = prev_x - x0

                # Distance from the center
                distance_from_center = np.sqrt(dy_from_center**2 + dx_from_center**2)

                # Calculate restoring force (harmonic potential)
                if distance_from_center > 0:
                    # Unit vector pointing toward the center
                    uy = -dy_from_center / distance_from_center
                    ux = -dx_from_center / distance_from_center

                    # Force magnitude increases with distance
                    force_magnitude = self.confinement_strength * (distance_from_center / self.confinement_radius)**2

                    # Force components
                    fy = force_magnitude * uy
                    fx = force_magnitude * ux
                else:
                    fy = fx = 0

                # Random diffusion component
                dy_random = np.random.normal(0, self.step_std)
                dx_random = np.random.normal(0, self.step_std)

                # Combined movement with restoring force
                dy = dy_random + fy * self.frame_interval
                dx = dx_random + fx * self.frame_interval

                # New position
                new_y = prev_y + dy
                new_x = prev_x + dx

                # Apply boundary conditions if image size is provided
                if image_size is not None:
                    height, width = image_size

                    # Reflective boundary conditions
                    if new_y < 0:
                        new_y = -new_y
                    elif new_y >= height:
                        new_y = 2 * height - new_y - 1

                    if new_x < 0:
                        new_x = -new_x
                    elif new_x >= width:
                        new_x = 2 * width - new_x - 1

                trajectory.append((new_y, new_x))

            trajectories.append(trajectory)

        return trajectories


class ActiveTransport(MotionModel):
    """Active transport model for simulating molecular motor-driven motion."""

    def __init__(self,
                 velocity: float = 1.0,  # µm/s
                 processivity: float = 0.9,  # probability of staying bound
                 diffusion_coefficient: float = 0.05,  # µm²/s for unbound state
                 rebinding_prob: float = 0.3,  # probability of rebinding after detachment
                 frame_interval: float = 0.1,  # seconds
                 pixel_size: float = 0.1,  # µm/pixel
                 track_angle: Optional[float] = None):  # angle of tracks (None for random)
        """
        Initialize the active transport model.

        Args:
            velocity: Transport velocity in µm/s
            processivity: Probability of staying bound per frame
            diffusion_coefficient: Diffusion coefficient (D) in µm²/s for unbound state
            rebinding_prob: Probability of rebinding after detachment
            frame_interval: Time between frames in seconds
            pixel_size: Physical size of a pixel in µm
            track_angle: Angle of the transport tracks in radians (None for random)
        """
        self.velocity = velocity
        self.processivity = processivity
        self.diffusion_coefficient = diffusion_coefficient
        self.rebinding_prob = rebinding_prob
        self.frame_interval = frame_interval
        self.pixel_size = pixel_size
        self.track_angle = track_angle

        # Convert velocity from µm/s to pixels/frame
        self.velocity_to_pixels = velocity * frame_interval / pixel_size

        # Calculate diffusion step size standard deviation
        self.diff_step_std = np.sqrt(2 * diffusion_coefficient * frame_interval) / pixel_size

    def apply(self,
             initial_positions: List[Tuple[float, float]],
             num_frames: int,
             image_size: Optional[Tuple[int, int]] = None) -> List[List[Tuple[float, float]]]:
        """
        Apply active transport to generate trajectories.

        Args:
            initial_positions: List of initial (y, x) positions
            num_frames: Number of frames to simulate
            image_size: Optional image size for boundary conditions (height, width)

        Returns:
            List of trajectories, where each trajectory is a list of (y, x) positions
        """
        trajectories = []

        # Initialize track angles for each particle
        track_angles = []
        bound_states = []  # True if particle is bound to track

        for _ in initial_positions:
            if self.track_angle is None:
                # Random track angle
                angle = np.random.uniform(0, 2*np.pi)
            else:
                # Fixed track angle with small variation
                angle = self.track_angle + np.random.normal(0, 0.1)

            track_angles.append(angle)
            bound_states.append(True)  # Start in bound state

        for i, (y0, x0) in enumerate(initial_positions):
            # Start trajectory with initial position
            trajectory = [(y0, x0)]

            angle = track_angles[i]
            bound = bound_states[i]

            # Apply active transport for each frame
            for _ in range(1, num_frames):
                prev_y, prev_x = trajectory[-1]

                # Check if state changes
                if bound:
                    # Currently bound, may detach
                    bound = np.random.random() < self.processivity
                else:
                    # Currently unbound, may rebind
                    bound = np.random.random() < self.rebinding_prob

                if bound:
                    # Directed motion along track with small variance
                    speed = self.velocity_to_pixels * (1 + np.random.normal(0, 0.1))
                    dy = speed * np.sin(angle)  # y-component uses sin
                    dx = speed * np.cos(angle)  # x-component uses cos

                    # Small random component perpendicular to track
                    perp_angle = angle + np.pi/2
                    perp_std = self.diff_step_std * 0.1  # Much smaller than diffusion
                    dy_perp = np.random.normal(0, perp_std) * np.sin(perp_angle)
                    dx_perp = np.random.normal(0, perp_std) * np.cos(perp_angle)

                    dy += dy_perp
                    dx += dx_perp
                else:
                    # Pure diffusion when unbound
                    dy = np.random.normal(0, self.diff_step_std)
                    dx = np.random.normal(0, self.diff_step_std)

                # New position
                new_y = prev_y + dy
                new_x = prev_x + dx

                # Apply boundary conditions if image size is provided
                if image_size is not None:
                    height, width = image_size

                    # Reflective boundary conditions
                    if new_y < 0:
                        new_y = -new_y
                        if bound:
                            angle = -angle  # Reflect track angle vertically
                    elif new_y >= height:
                        new_y = 2 * height - new_y - 1
                        if bound:
                            angle = -angle  # Reflect track angle vertically

                    if new_x < 0:
                        new_x = -new_x
                        if bound:
                            angle = np.pi - angle  # Reflect track angle horizontally
                    elif new_x >= width:
                        new_x = 2 * width - new_x - 1
                        if bound:
                            angle = np.pi - angle  # Reflect track angle horizontally

                trajectory.append((new_y, new_x))
                track_angles[i] = angle  # Update track angle if reflected

            trajectories.append(trajectory)

        return trajectories
