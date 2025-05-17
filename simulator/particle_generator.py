"""
Particle generator for simulating fluorescent particles in microscopy images.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Union
import logging

logger = logging.getLogger(__name__)


class Particle:
    """Represents a single fluorescent particle with position and properties."""

    def __init__(self,
                 particle_id: int,
                 initial_position: Tuple[float, float],
                 intensity: float = 1.0,
                 size: float = 1.0):
        """
        Initialize a particle.

        Args:
            particle_id: Unique identifier for the particle
            initial_position: (y, x) coordinates
            intensity: Brightness of the particle
            size: Size modifier for the particle
        """
        self.id = particle_id
        self.positions = [initial_position]  # List to track positions over time (y, x)
        self.intensity = intensity
        self.size = size
        self.active = True  # Whether the particle is currently fluorescing
        self.properties = {}  # Additional properties

    def add_position(self, position: Tuple[float, float]):
        """Add a new position (y, x) for the particle."""
        self.positions.append(position)

    def get_position(self, frame_idx: int = -1) -> Tuple[float, float]:
        """Get the position at a specific frame (defaults to latest)."""
        if frame_idx < 0:
            frame_idx = len(self.positions) + frame_idx

        if 0 <= frame_idx < len(self.positions):
            return self.positions[frame_idx]
        else:
            raise IndexError(f"Frame index {frame_idx} out of range")

    def get_trajectory(self) -> List[Tuple[float, float]]:
        """Get the full trajectory of the particle."""
        return self.positions.copy()

    def set_property(self, key: str, value):
        """Set an additional property for the particle."""
        self.properties[key] = value

    def get_property(self, key: str, default=None):
        """Get a property with a default value if not found."""
        return self.properties.get(key, default)


class ParticleGenerator:
    """Generates simulated particles with realistic behavior."""

    def __init__(self,
                 image_size: Tuple[int, int] = (512, 512),
                 pixel_size: float = 0.1):  # microns per pixel
        """
        Initialize the particle generator.

        Args:
            image_size: Image dimensions (height, width) in pixels
            pixel_size: Physical size of a pixel in microns
        """
        self.image_size = image_size
        self.pixel_size = pixel_size
        self.particles = []
        self.next_id = 0

    def create_random_particles(self,
                              num_particles: int,
                              intensity_range: Tuple[float, float] = (0.5, 1.0),
                              size_range: Tuple[float, float] = (0.8, 1.2)) -> List[Particle]:
        """
        Create random particles within the image bounds.

        Args:
            num_particles: Number of particles to create
            intensity_range: Range of intensity values (min, max)
            size_range: Range of size values (min, max)

        Returns:
            List of created Particle objects
        """
        height, width = self.image_size
        new_particles = []

        for _ in range(num_particles):
            # Create coordinates in (y, x) order to match numpy convention
            y = np.random.uniform(0, height)
            x = np.random.uniform(0, width)
            intensity = np.random.uniform(*intensity_range)
            size = np.random.uniform(*size_range)

            particle = Particle(
                particle_id=self.next_id,
                initial_position=(y, x),  # (y, x) order
                intensity=intensity,
                size=size
            )

            self.next_id += 1
            new_particles.append(particle)
            self.particles.append(particle)

        logger.info(f"Created {num_particles} random particles")
        return new_particles

    def apply_brownian_motion(self,
                             particles: Optional[List[Particle]] = None,
                             num_frames: int = 10,
                             diffusion_coefficient: float = 0.1,  # microns^2/s
                             frame_interval: float = 0.1) -> None:  # seconds
        """
        Apply Brownian motion to particles over multiple frames.

        Args:
            particles: List of particles to move (defaults to all particles)
            num_frames: Number of frames to simulate
            diffusion_coefficient: Diffusion coefficient (D) in microns^2/s
            frame_interval: Time between frames in seconds
        """
        if particles is None:
            particles = self.particles

        # Calculate the standard deviation of step size
        # MSD = 4*D*t for 2D Brownian motion
        std_dev = np.sqrt(2 * diffusion_coefficient * frame_interval) / self.pixel_size

        for frame in range(num_frames):
            for particle in particles:
                last_pos = particle.get_position()
                dy = np.random.normal(0, std_dev)
                dx = np.random.normal(0, std_dev)

                # New position (y, x) order
                new_y = last_pos[0] + dy
                new_x = last_pos[1] + dx

                # Optional: Apply boundary conditions (e.g., reflection)
                height, width = self.image_size
                new_y = max(0, min(height, new_y))
                new_x = max(0, min(width, new_x))

                particle.add_position((new_y, new_x))

        logger.info(f"Applied Brownian motion to {len(particles)} particles over {num_frames} frames")

    def apply_directed_motion(self,
                             particles: Optional[List[Particle]] = None,
                             num_frames: int = 10,
                             velocity_range: Tuple[float, float] = (0.5, 2.0),  # microns/s
                             direction_change_prob: float = 0.1,
                             frame_interval: float = 0.1) -> None:  # seconds
        """
        Apply directed motion to particles over multiple frames.

        Args:
            particles: List of particles to move (defaults to all particles)
            num_frames: Number of frames to simulate
            velocity_range: Range of velocities in microns/s (min, max)
            direction_change_prob: Probability of changing direction per frame
            frame_interval: Time between frames in seconds
        """
        if particles is None:
            particles = self.particles

        # Initialize direction and velocity for each particle
        particle_directions = {}
        particle_velocities = {}

        for particle in particles:
            particle_directions[particle.id] = np.random.uniform(0, 2*np.pi)
            particle_velocities[particle.id] = np.random.uniform(*velocity_range)

        # Convert velocity from microns/s to pixels/frame
        velocity_to_pixels = frame_interval / self.pixel_size

        for frame in range(num_frames):
            for particle in particles:
                # Check if direction should change
                if np.random.random() < direction_change_prob:
                    # Change direction slightly
                    particle_directions[particle.id] += np.random.normal(0, np.pi/4)

                # Get current direction and velocity
                direction = particle_directions[particle.id]
                velocity_microns = particle_velocities[particle.id]
                velocity_pixels = velocity_microns * velocity_to_pixels

                # Calculate movement in (y, x) order
                dy = velocity_pixels * np.sin(direction)  # y component
                dx = velocity_pixels * np.cos(direction)  # x component

                # Update position
                last_pos = particle.get_position()
                new_y = last_pos[0] + dy
                new_x = last_pos[1] + dx

                # Apply boundary conditions
                height, width = self.image_size
                if new_x < 0 or new_x >= width:
                    # Reflect direction horizontally
                    particle_directions[particle.id] = np.pi - direction
                    new_x = max(0, min(width-1, new_x))

                if new_y < 0 or new_y >= height:
                    # Reflect direction vertically
                    particle_directions[particle.id] = -direction
                    new_y = max(0, min(height-1, new_y))

                particle.add_position((new_y, new_x))

        logger.info(f"Applied directed motion to {len(particles)} particles over {num_frames} frames")

    def apply_blinking(self,
                      particles: Optional[List[Particle]] = None,
                      num_frames: int = 10,
                      on_probability: float = 0.7,
                      off_probability: float = 0.2) -> Dict[int, List[bool]]:
        """
        Apply stochastic blinking behavior to particles.

        Args:
            particles: List of particles to apply blinking to
            num_frames: Number of frames to simulate
            on_probability: Probability of turning on if currently off
            off_probability: Probability of turning off if currently on

        Returns:
            Dictionary mapping particle IDs to list of active states per frame
        """
        if particles is None:
            particles = self.particles

        # Initialize state dictionary
        blinking_states = {particle.id: [] for particle in particles}

        # Make sure initial states are defined (all active by default)
        for particle in particles:
            blinking_states[particle.id].append(True)

        # Simulate the stochastic blinking process
        for frame in range(1, num_frames):  # Start from 1 since we already have frame 0
            for particle in particles:
                previous_state = blinking_states[particle.id][-1]

                if previous_state:  # Currently on
                    new_state = np.random.random() >= off_probability  # Stay on with 1-off_probability
                else:  # Currently off
                    new_state = np.random.random() < on_probability  # Turn on with on_probability

                blinking_states[particle.id].append(new_state)

                # Add property to the particle for this frame
                particle.set_property(f"active_frame_{frame}", new_state)

        logger.info(f"Applied blinking behavior to {len(particles)} particles over {num_frames} frames")
        return blinking_states

    def apply_confined_diffusion(self,
                                particles: Optional[List[Particle]] = None,
                                num_frames: int = 10,
                                diffusion_coefficient: float = 0.1,  # microns^2/s
                                confinement_strength: float = 1.0,
                                confinement_radius: float = 10.0,  # pixels
                                frame_interval: float = 0.1) -> None:  # seconds
        """
        Apply confined diffusion to particles over multiple frames.

        Args:
            particles: List of particles to move (defaults to all particles)
            num_frames: Number of frames to simulate
            diffusion_coefficient: Diffusion coefficient (D) in microns^2/s
            confinement_strength: Strength of confinement
            confinement_radius: Radius of confinement in pixels
            frame_interval: Time between frames in seconds
        """
        if particles is None:
            particles = self.particles

        # Calculate the standard deviation of step size
        std_dev = np.sqrt(2 * diffusion_coefficient * frame_interval) / self.pixel_size

        for frame in range(num_frames):
            for particle in particles:
                # Get initial position (y0, x0) from first frame
                initial_pos = particle.positions[0]
                y0, x0 = initial_pos

                # Get current position
                current_pos = particle.get_position()
                y_current, x_current = current_pos

                # Calculate displacement from center of confinement
                dy_from_center = y_current - y0
                dx_from_center = x_current - x0

                # Calculate distance from center
                distance_from_center = np.sqrt(dy_from_center**2 + dx_from_center**2)

                # Calculate restoring force (pointing back to center)
                if distance_from_center > 0:
                    # Unit vector pointing toward center
                    uy = -dy_from_center / distance_from_center
                    ux = -dx_from_center / distance_from_center

                    # Force magnitude increases with distance
                    force_magnitude = confinement_strength * (distance_from_center / confinement_radius)**2

                    # Force components
                    fy = force_magnitude * uy
                    fx = force_magnitude * ux
                else:
                    fy = fx = 0

                # Random diffusion component
                dy_random = np.random.normal(0, std_dev)
                dx_random = np.random.normal(0, std_dev)

                # Combined movement with restoring force
                dy = dy_random + fy * frame_interval
                dx = dx_random + fx * frame_interval

                # New position (y, x) order
                new_y = y_current + dy
                new_x = x_current + dx

                # Apply boundary conditions
                height, width = self.image_size
                new_y = max(0, min(height-1, new_y))
                new_x = max(0, min(width-1, new_x))

                particle.add_position((new_y, new_x))

        logger.info(f"Applied confined diffusion to {len(particles)} particles over {num_frames} frames")

    def get_particle_positions(self, frame_idx: int) -> Dict[int, Tuple[float, float]]:
        """
        Get positions of all particles at a specific frame.

        Args:
            frame_idx: Frame index

        Returns:
            Dictionary mapping particle IDs to (y, x) positions
        """
        positions = {}
        for particle in self.particles:
            try:
                positions[particle.id] = particle.get_position(frame_idx)
            except IndexError:
                # Particle doesn't have position for this frame
                pass

        return positions

    def get_active_particles(self, frame_idx: int) -> List[Particle]:
        """
        Get list of particles that are active (visible) in a specific frame.

        Args:
            frame_idx: Frame index

        Returns:
            List of active Particle objects
        """
        active_particles = []
        for particle in self.particles:
            try:
                # Check if the particle has a position for this frame
                _ = particle.get_position(frame_idx)

                # Check if the particle is active (not blinking off)
                is_active = particle.get_property(f"active_frame_{frame_idx}", True)

                if is_active:
                    active_particles.append(particle)
            except IndexError:
                # Particle doesn't have position for this frame
                pass

        return active_particles

    def clear(self):
        """Clear all particles."""
        self.particles = []
        logger.info("Cleared all particles")
