"""
Data loading and preprocessing for particle tracking.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import h5py
import tifffile
import os
import glob
from typing import Dict, List, Optional, Tuple, Union, Callable
import logging
from PIL import Image
import random

logger = logging.getLogger(__name__)


class ParticleDataset(Dataset):
    """Dataset for particle tracking with precomputed data."""

    def __init__(self,
                 data_path: str,
                 frame_sequence_length: int = 5,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 mode: str = 'train'):
        """
        Initialize the particle dataset.

        Args:
            data_path: Path to the data file or directory
            frame_sequence_length: Number of consecutive frames to include
            transform: Transforms to apply to the input data
            target_transform: Transforms to apply to the target data
            mode: Dataset mode ('train', 'val', 'test')
        """
        self.data_path = data_path
        self.frame_sequence_length = frame_sequence_length
        self.transform = transform
        self.target_transform = target_transform
        self.mode = mode

        # Load the data
        self._load_data()

    def _load_data(self):
        """Load the data based on the file type."""
        if self.data_path.endswith('.h5') or self.data_path.endswith('.hdf5'):
            self._load_hdf5()
        elif os.path.isdir(self.data_path):
            self._load_directory()
        else:
            raise ValueError(f"Unsupported data source: {self.data_path}")

    def _load_hdf5(self):
        """Load data from an HDF5 file."""
        with h5py.File(self.data_path, 'r') as f:
            # Check dataset structure
            if 'frames' not in f:
                raise ValueError("HDF5 file must contain a 'frames' dataset")

            # Load frames
            self.frames = np.array(f['frames'])

            # Get number of samples
            self.num_total_frames = self.frames.shape[0]
            self.num_samples = max(0, self.num_total_frames - self.frame_sequence_length + 1)

            # Load positions if available
            if 'positions' in f:
                self.positions = np.array(f['positions'])
            else:
                self.positions = None

            # Load track IDs if available
            if 'track_ids' in f:
                self.track_ids = np.array(f['track_ids'])
            else:
                self.track_ids = None

        logger.info(f"Loaded {self.num_total_frames} frames from {self.data_path}")
        logger.info(f"Dataset contains {self.num_samples} samples with sequence length {self.frame_sequence_length}")

    def _load_directory(self):
        """Load data from a directory of image files."""
        # Find all image files in the directory
        image_files = sorted(glob.glob(os.path.join(self.data_path, '*.tif')))
        image_files.extend(sorted(glob.glob(os.path.join(self.data_path, '*.tiff'))))
        image_files.extend(sorted(glob.glob(os.path.join(self.data_path, '*.png'))))
        image_files.extend(sorted(glob.glob(os.path.join(self.data_path, '*.jpg'))))
        image_files.extend(sorted(glob.glob(os.path.join(self.data_path, '*.jpeg'))))

        if not image_files:
            raise ValueError(f"No image files found in {self.data_path}")

        # Load all images
        self.frames = []
        for img_file in image_files:
            if img_file.endswith(('.tif', '.tiff')):
                img = tifffile.imread(img_file)
            else:
                img = np.array(Image.open(img_file).convert('L'))  # Convert to grayscale

            self.frames.append(img)

        self.frames = np.stack(self.frames)

        # Get number of samples
        self.num_total_frames = len(self.frames)
        self.num_samples = max(0, self.num_total_frames - self.frame_sequence_length + 1)

        # Check for position and track files
        pos_file = os.path.join(self.data_path, 'positions.npy')
        if os.path.exists(pos_file):
            self.positions = np.load(pos_file)
        else:
            self.positions = None

        track_file = os.path.join(self.data_path, 'track_ids.npy')
        if os.path.exists(track_file):
            self.track_ids = np.load(track_file)
        else:
            self.track_ids = None

        logger.info(f"Loaded {self.num_total_frames} frames from {self.data_path}")
        logger.info(f"Dataset contains {self.num_samples} samples with sequence length {self.frame_sequence_length}")

    def __len__(self) -> int:
        """Get the number of samples in the dataset."""
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample from the dataset.

        Args:
            idx: Sample index

        Returns:
            Dictionary with input frames and target data
        """
        # Get frame sequence
        start_idx = idx
        end_idx = start_idx + self.frame_sequence_length
        frame_sequence = self.frames[start_idx:end_idx]

        # Convert to float32 and normalize to [0, 1]
        frame_sequence = frame_sequence.astype(np.float32)
        if frame_sequence.max() > 1.0:
            frame_sequence = frame_sequence / 255.0

        # Create sample dictionary
        sample = {
            'frames': frame_sequence,
        }

        # Add positions if available
        if self.positions is not None:
            sample['positions'] = self.positions[start_idx:end_idx]

        # Add track IDs if available
        if self.track_ids is not None:
            sample['track_ids'] = self.track_ids[start_idx:end_idx]

        # Apply transforms
        if self.transform:
            sample = self.transform(sample)

        # Apply target transform
        if self.target_transform:
            sample = self.target_transform(sample)

        # Convert all arrays to tensors
        for key, value in sample.items():
            if isinstance(value, np.ndarray):
                sample[key] = torch.from_numpy(value)

        # Add frame indices for reference
        sample['frame_indices'] = torch.arange(start_idx, end_idx)

        return sample


class SimulatedParticleDataset(Dataset):
    """Dataset that generates simulated particle data on-the-fly."""

    def __init__(self,
                 num_samples: int = 1000,
                 frame_size: Tuple[int, int] = (512, 512),
                 frame_sequence_length: int = 5,
                 particles_per_frame: Tuple[int, int] = (10, 50),
                 motion_model: str = 'brownian',
                 psf_model: str = 'gaussian',
                 noise_model: str = 'poisson_gaussian',
                 snr_range: Tuple[float, float] = (2.0, 20.0),
                 transform: Optional[Callable] = None):
        """
        Initialize the simulated particle dataset.

        Args:
            num_samples: Number of samples to generate
            frame_size: Size of each frame (height, width)
            frame_sequence_length: Number of frames in each sequence
            particles_per_frame: Range of particles per frame (min, max)
            motion_model: Motion model to use ('brownian', 'directed', 'confined')
            psf_model: PSF model to use ('gaussian', 'airy')
            noise_model: Noise model to use ('poisson', 'gaussian', 'poisson_gaussian')
            snr_range: Signal-to-noise ratio range (min, max)
            transform: Transforms to apply to the data
        """
        self.num_samples = num_samples
        self.frame_size = frame_size
        self.frame_sequence_length = frame_sequence_length
        self.particles_per_frame = particles_per_frame
        self.motion_model = motion_model
        self.psf_model = psf_model
        self.noise_model = noise_model
        self.snr_range = snr_range
        self.transform = transform

        # Particle generator setup
        try:
            from simulator.particle_generator import ParticleGenerator
            from simulator.psf_models import GaussianPSF, AiryDiskPSF
            from simulator.noise_models import PoissonNoise, GaussianNoise, MixedNoise
            from simulator.motion_models import BrownianMotion, DirectedMotion, ConfinedDiffusion

            self.particle_generator = ParticleGenerator(image_size=frame_size)

            # Set up PSF model
            if psf_model == 'gaussian':
                self.psf_model_obj = GaussianPSF(image_size=frame_size, sigma=1.0)
            elif psf_model == 'airy':
                self.psf_model_obj = AiryDiskPSF(image_size=frame_size, airy_radius=1.22)
            else:
                raise ValueError(f"Unsupported PSF model: {psf_model}")

            # Set up noise model
            if noise_model == 'poisson':
                self.noise_model_obj = PoissonNoise(scaling_factor=100.0)
            elif noise_model == 'gaussian':
                self.noise_model_obj = GaussianNoise(sigma=10.0)
            elif noise_model == 'poisson_gaussian':
                self.noise_model_obj = MixedNoise(photon_scaling=100.0, read_noise=3.0)
            else:
                raise ValueError(f"Unsupported noise model: {noise_model}")

            # Set up motion model
            if motion_model == 'brownian':
                self.motion_model_obj = BrownianMotion(
                    diffusion_coefficient=0.1,
                    frame_interval=0.1
                )
            elif motion_model == 'directed':
                self.motion_model_obj = DirectedMotion(
                    velocity_range=(0.5, 2.0),
                    direction_change_prob=0.1,
                    frame_interval=0.1
                )
            elif motion_model == 'confined':
                self.motion_model_obj = ConfinedDiffusion(
                    diffusion_coefficient=0.1,
                    confinement_strength=1.0,
                    confinement_radius=10.0,
                    frame_interval=0.1
                )
            else:
                raise ValueError(f"Unsupported motion model: {motion_model}")

            self.simulator_available = True
            logger.info("Simulator modules imported successfully")

        except ImportError:
            self.simulator_available = False
            logger.warning("Simulator modules not available, will generate random data")

    def __len__(self) -> int:
        """Get the number of samples in the dataset."""
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Generate a simulated particle tracking sample.

        Args:
            idx: Sample index (used as random seed)

        Returns:
            Dictionary with input frames and target data
        """
        # Set random seed based on index for reproducibility
        np.random.seed(idx)

        if self.simulator_available:
            return self._generate_sample_with_simulator()
        else:
            return self._generate_random_sample()

    def _generate_sample_with_simulator(self) -> Dict[str, torch.Tensor]:
        """Generate a sample using the simulator."""
        # Clear previous particles
        self.particle_generator.clear()

        # Generate random number of particles
        num_particles = np.random.randint(*self.particles_per_frame)

        # Create random particles
        particles = self.particle_generator.create_random_particles(
            num_particles=num_particles,
            intensity_range=(0.5, 1.0),
            size_range=(0.8, 1.2)
        )

        # Apply motion model
        if self.motion_model == 'brownian':
            self.particle_generator.apply_brownian_motion(
                particles=particles,
                num_frames=self.frame_sequence_length,
                diffusion_coefficient=np.random.uniform(0.05, 0.2)
            )
        elif self.motion_model == 'directed':
            self.particle_generator.apply_directed_motion(
                particles=particles,
                num_frames=self.frame_sequence_length,
                velocity_range=(0.5, 2.0),
                direction_change_prob=0.1
            )
        elif self.motion_model == 'confined':
            self.particle_generator.apply_confined_diffusion(
                particles=particles,
                num_frames=self.frame_sequence_length,
                diffusion_coefficient=0.1,
                confinement_strength=1.0,
                confinement_radius=10.0
            )

        # Apply blinking behavior
        blinking_states = self.particle_generator.apply_blinking(
            particles=particles,
            num_frames=self.frame_sequence_length,
            on_probability=0.7,
            off_probability=0.2
        )

        # Generate frames
        frames = []
        positions = []
        track_ids = []

        for frame_idx in range(self.frame_sequence_length):
            # Get active particles for this frame
            active_particles = self.particle_generator.get_active_particles(frame_idx)

            # Extract positions and intensities
            particle_positions = []
            particle_intensities = []
            particle_sizes = []
            particle_ids = []

            for particle in active_particles:
                pos = particle.get_position(frame_idx)
                particle_positions.append(pos)
                particle_intensities.append(particle.intensity)
                particle_sizes.append(particle.size)
                particle_ids.append(particle.id)

            # Generate PSF image
            if particle_positions:
                psf_image = self.psf_model_obj.generate(
                    positions=particle_positions,
                    intensities=particle_intensities,
                    sizes=particle_sizes
                )
            else:
                psf_image = np.zeros(self.frame_size, dtype=np.float32)

            # Apply noise
            snr = np.random.uniform(*self.snr_range)
            background = np.max(psf_image) / snr if np.max(psf_image) > 0 else 0.1
            noisy_image = self.noise_model_obj.apply(psf_image + background) - background

            # Clip to [0, 1] and add to frames
            noisy_image = np.clip(noisy_image, 0, 1)
            frames.append(noisy_image)

            # Create position target (binary mask with particle positions)
            pos_target = np.zeros(self.frame_size, dtype=np.float32)
            for pos in particle_positions:
                y, x = int(round(pos[0])), int(round(pos[1]))
                if 0 <= y < self.frame_size[0] and 0 <= x < self.frame_size[1]:
                    pos_target[y, x] = 1.0

            positions.append(pos_target)

            # Create track ID target
            track_id_target = np.zeros(self.frame_size, dtype=np.int32)
            for pos, pid in zip(particle_positions, particle_ids):
                y, x = int(round(pos[0])), int(round(pos[1]))
                if 0 <= y < self.frame_size[0] and 0 <= x < self.frame_size[1]:
                    track_id_target[y, x] = pid + 1  # Add 1 because 0 is background

            track_ids.append(track_id_target)

        # Stack arrays
        frames = np.stack(frames).astype(np.float32)  # Explicitly use float32
        positions = np.stack(positions).astype(np.float32)  # Explicitly use float32
        track_ids = np.stack(track_ids)

        # Create sample dictionary - ensure correct shape for all arrays
        sample = {
            'frames': frames,  # Shape: (num_frames, height, width)
            'positions': positions[:, np.newaxis, :, :],  # Shape: (num_frames, 1, height, width)
            'track_ids': track_ids[:, np.newaxis, :, :],  # Shape: (num_frames, 1, height, width)
        }

        # Apply transforms
        if self.transform:
            sample = self.transform(sample)

        # Convert all arrays to tensors with explicit float32 dtype for floating point data
        for key, value in sample.items():
            if isinstance(value, np.ndarray):
                if np.issubdtype(value.dtype, np.floating):
                    # Convert floating point arrays to float32 before converting to tensor
                    value = value.astype(np.float32)
                sample[key] = torch.from_numpy(value)

        # Ensure frames have correct shape for the model (batch, frames, channels, height, width)
        # The batch dimension is implicit since we're returning a single sample
        if 'frames' in sample and sample['frames'].ndim == 3:  # (frames, height, width)
            # Add channel dimension
            sample['frames'] = sample['frames'].unsqueeze(1)  # (frames, 1, height, width)

        return sample

    def _generate_random_sample(self) -> Dict[str, torch.Tensor]:
        """Generate a random sample when simulator is not available."""
        # Generate random frames
        frames = np.random.random((self.frame_sequence_length, *self.frame_size)).astype(np.float32)

        # Generate random positions (binary masks)
        positions = np.zeros((self.frame_sequence_length, 1, *self.frame_size), dtype=np.float32)

        # Generate random track IDs
        track_ids = np.zeros((self.frame_sequence_length, 1, *self.frame_size), dtype=np.int32)

        # Generate random number of particles per frame
        for f in range(self.frame_sequence_length):
            num_particles = np.random.randint(*self.particles_per_frame)

            for p in range(num_particles):
                # Random position - now in (y, x) order
                y = np.random.randint(0, self.frame_size[0])
                x = np.random.randint(0, self.frame_size[1])

                # Random particle ID
                pid = p + 1  # Add 1 because 0 is background

                # Set position and track ID
                positions[f, 0, y, x] = 1.0
                track_ids[f, 0, y, x] = pid

                # Add a bright spot to the frame
                y_min = max(0, y - 2)
                y_max = min(self.frame_size[0], y + 3)
                x_min = max(0, x - 2)
                x_max = min(self.frame_size[1], x + 3)

                frames[f, y_min:y_max, x_min:x_max] += 0.5 * np.exp(
                    -0.5 * ((np.arange(y_min, y_max)[:, np.newaxis] - y) ** 2 +
                            (np.arange(x_min, x_max)[np.newaxis, :] - x) ** 2)
                )

        # Clip frames to [0, 1]
        frames = np.clip(frames, 0, 1)

        # Create sample dictionary
        sample = {
            'frames': frames,
            'positions': positions,
            'track_ids': track_ids,
        }

        # Apply transforms
        if self.transform:
            sample = self.transform(sample)

        # Convert all arrays to tensors
        for key, value in sample.items():
            if isinstance(value, np.ndarray):
                sample[key] = torch.from_numpy(value)

        return sample


class RandomCrop:
    """Random crop data augmentation."""

    def __init__(self, output_size: Tuple[int, int]):
        """
        Initialize random crop.

        Args:
            output_size: Desired output size (height, width)
        """
        self.output_size = output_size

    def __call__(self, sample: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Apply random crop to a sample.

        Args:
            sample: Dictionary with data arrays

        Returns:
            Transformed sample
        """
        # Get input frames
        frames = sample['frames']

        # Handle different input formats
        if frames.ndim == 3:  # (F, H, W)
            frames_shape = frames.shape
            num_frames, height, width = frames_shape

            # Get crop coordinates
            h_diff = height - self.output_size[0]
            w_diff = width - self.output_size[1]

            if h_diff < 0 or w_diff < 0:
                raise ValueError(f"Output size {self.output_size} larger than input size ({height}, {width})")

            top = np.random.randint(0, h_diff + 1)
            left = np.random.randint(0, w_diff + 1)
            bottom = top + self.output_size[0]
            right = left + self.output_size[1]

            # Apply crop to all arrays in the sample
            result = {}
            for key, value in sample.items():
                if key == 'frames':
                    result[key] = value[:, top:bottom, left:right]
                elif value.ndim == 4 and value.shape[0] == num_frames:  # (F, C, H, W)
                    result[key] = value[:, :, top:bottom, left:right]
                else:
                    result[key] = value

        elif frames.ndim == 4:  # (F, C, H, W)
            frames_shape = frames.shape
            num_frames, channels, height, width = frames_shape

            # Get crop coordinates
            h_diff = height - self.output_size[0]
            w_diff = width - self.output_size[1]

            if h_diff < 0 or w_diff < 0:
                raise ValueError(f"Output size {self.output_size} larger than input size ({height}, {width})")

            top = np.random.randint(0, h_diff + 1)
            left = np.random.randint(0, w_diff + 1)
            bottom = top + self.output_size[0]
            right = left + self.output_size[1]

            # Apply crop to all arrays in the sample
            result = {}
            for key, value in sample.items():
                if key == 'frames':
                    result[key] = value[:, :, top:bottom, left:right]
                elif value.ndim == 4 and value.shape[0] == num_frames:  # (F, C, H, W)
                    result[key] = value[:, :, top:bottom, left:right]
                else:
                    result[key] = value
        else:
            raise ValueError(f"Unsupported input shape: {frames.shape}")

        return result


class RandomRotation:
    """Random rotation data augmentation."""

    def __init__(self, max_angle: float = 30.0):
        """
        Initialize random rotation.

        Args:
            max_angle: Maximum rotation angle in degrees
        """
        self.max_angle = max_angle

    def __call__(self, sample: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Apply random rotation to a sample.

        Args:
            sample: Dictionary with data arrays

        Returns:
            Transformed sample
        """
        # Get angle
        angle = np.random.uniform(-self.max_angle, self.max_angle)

        # Apply rotation to all arrays in the sample
        result = {}

        for key, value in sample.items():
            if key == 'frames':
                result[key] = self._rotate_array(value, angle)
            elif key in ['positions', 'track_ids']:
                result[key] = self._rotate_array(value, angle)
            else:
                result[key] = value

        return result

    def _rotate_array(self, arr: np.ndarray, angle: float) -> np.ndarray:
        """Rotate array along last two dimensions."""
        from scipy.ndimage import rotate

        # Handle different input formats
        if arr.ndim == 3:  # (F, H, W)
            result = np.zeros_like(arr)
            for i in range(arr.shape[0]):
                result[i] = rotate(arr[i], angle, reshape=False, order=1, mode='constant')
        elif arr.ndim == 4:  # (F, C, H, W)
            result = np.zeros_like(arr)
            for i in range(arr.shape[0]):
                for j in range(arr.shape[1]):
                    result[i, j] = rotate(arr[i, j], angle, reshape=False, order=1, mode='constant')
        else:
            raise ValueError(f"Unsupported array shape: {arr.shape}")

        return result


class RandomFlip:
    """Random flip data augmentation."""

    def __init__(self, p: float = 0.5):
        """
        Initialize random flip.

        Args:
            p: Probability of applying a flip
        """
        self.p = p

    def __call__(self, sample: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Apply random flip to a sample.

        Args:
            sample: Dictionary with data arrays

        Returns:
            Transformed sample
        """
        # Determine if flips should be applied
        flip_h = np.random.random() < self.p
        flip_v = np.random.random() < self.p

        if not flip_h and not flip_v:
            return sample

        # Apply flips to all arrays in the sample
        result = {}

        for key, value in sample.items():
            if key in ['frames', 'positions', 'track_ids']:
                # Apply flips based on array dimensions
                if value.ndim == 3:  # (F, H, W)
                    result[key] = self._flip_array(value, flip_h, flip_v)
                elif value.ndim == 4:  # (F, C, H, W)
                    result[key] = self._flip_array(value, flip_h, flip_v)
                else:
                    result[key] = value
            else:
                result[key] = value

        return result

    def _flip_array(self, arr: np.ndarray, flip_h: bool, flip_v: bool) -> np.ndarray:
        """Flip array along last two dimensions."""
        result = arr.copy()

        if arr.ndim == 3:  # (F, H, W)
            if flip_v:
                result = result[:, ::-1, :]
            if flip_h:
                result = result[:, :, ::-1]
        elif arr.ndim == 4:  # (F, C, H, W)
            if flip_v:
                result = result[:, :, ::-1, :]
            if flip_h:
                result = result[:, :, :, ::-1]

        return result


class RandomNoise:
    """Random noise data augmentation."""

    def __init__(self,
                 noise_level: float = 0.05,
                 p: float = 0.5):
        """
        Initialize random noise.

        Args:
            noise_level: Maximum noise level to add
            p: Probability of applying noise
        """
        self.noise_level = noise_level
        self.p = p

    def __call__(self, sample: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Apply random noise to a sample.

        Args:
            sample: Dictionary with data arrays

        Returns:
            Transformed sample
        """
        if np.random.random() > self.p:
            return sample

        # Apply noise only to frames
        result = sample.copy()

        frames = sample['frames']

        # Generate noise
        noise_std = np.random.uniform(0, self.noise_level)

        # Handle different input formats
        if frames.ndim == 3:  # (F, H, W)
            noise = np.random.normal(0, noise_std, frames.shape)
            result['frames'] = np.clip(frames + noise, 0, 1)
        elif frames.ndim == 4:  # (F, C, H, W)
            noise = np.random.normal(0, noise_std, frames.shape)
            result['frames'] = np.clip(frames + noise, 0, 1)

        return result


class RandomIntensity:
    """Random intensity adjustment data augmentation."""

    def __init__(self,
                 brightness_range: Tuple[float, float] = (0.8, 1.2),
                 contrast_range: Tuple[float, float] = (0.8, 1.2),
                 p: float = 0.5):
        """
        Initialize random intensity adjustment.

        Args:
            brightness_range: Range of brightness adjustment factor
            contrast_range: Range of contrast adjustment factor
            p: Probability of applying adjustment
        """
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.p = p

    def __call__(self, sample: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Apply random intensity adjustment to a sample.

        Args:
            sample: Dictionary with data arrays

        Returns:
            Transformed sample
        """
        if np.random.random() > self.p:
            return sample

        # Apply adjustment only to frames
        result = sample.copy()

        frames = sample['frames']

        # Generate adjustment factors
        brightness = np.random.uniform(*self.brightness_range)
        contrast = np.random.uniform(*self.contrast_range)

        # Apply adjustment
        adjusted = frames * brightness
        mean = np.mean(adjusted)
        adjusted = (adjusted - mean) * contrast + mean

        # Clip to valid range
        result['frames'] = np.clip(adjusted, 0, 1)

        return result


def create_dataloaders(
    train_data_source: str,
    val_data_source: Optional[str] = None,
    batch_size: int = 16,
    frame_sequence_length: int = 5,
    num_workers: int = 4,
    use_simulated: bool = False,
    simulated_samples: int = 1000,
    augmentation: bool = True
) -> Dict[str, DataLoader]:
    """
    Create data loaders for training and validation.

    Args:
        train_data_source: Path to training data
        val_data_source: Path to validation data
        batch_size: Batch size
        frame_sequence_length: Number of frames in each sequence
        num_workers: Number of worker processes
        use_simulated: Whether to use simulated data
        simulated_samples: Number of simulated samples to generate
        augmentation: Whether to apply data augmentation

    Returns:
        Dictionary of data loaders for 'train' and 'val'
    """
    # Define transforms
    if augmentation:
        train_transform = lambda sample: RandomFlip(p=0.5)(
            RandomNoise(noise_level=0.05, p=0.5)(
                RandomIntensity(
                    brightness_range=(0.8, 1.2),
                    contrast_range=(0.8, 1.2),
                    p=0.5
                )(sample)
            )
        )
    else:
        train_transform = None

    val_transform = None

    # Create datasets
    if use_simulated:
        # Add debug output for simulated dataset
        print("Creating simulated datasets with frame_sequence_length =", frame_sequence_length)

        train_dataset = SimulatedParticleDataset(
            num_samples=simulated_samples,
            frame_sequence_length=frame_sequence_length,
            transform=train_transform
        )

        # Create a smaller validation set
        val_dataset = SimulatedParticleDataset(
            num_samples=max(100, simulated_samples // 10),
            frame_sequence_length=frame_sequence_length,
            transform=val_transform
        )
    else:
        train_dataset = ParticleDataset(
            data_path=train_data_source,
            frame_sequence_length=frame_sequence_length,
            transform=train_transform,
            mode='train'
        )

        if val_data_source:
            val_dataset = ParticleDataset(
                data_path=val_data_source,
                frame_sequence_length=frame_sequence_length,
                transform=val_transform,
                mode='val'
            )
        else:
            # Split training dataset
            train_size = int(0.8 * len(train_dataset))
            val_size = len(train_dataset) - train_size

            train_dataset, val_dataset = torch.utils.data.random_split(
                train_dataset, [train_size, val_size]
            )

    # Add a debug collate function to inspect batch shapes
    def debug_collate_fn(batch):
        # Standard collate function with shape debugging
        elem = batch[0]
        if isinstance(elem, torch.Tensor):
            out = torch.stack(batch, 0)
            print(f"Tensor batch shape: {out.shape}")
            return out
        elif isinstance(elem, dict):
            result = {key: debug_collate_fn([d[key] for d in batch]) for key in elem}
            return result
        else:
            return batch

    # Create data loaders with the custom collate function
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return {'train': train_loader, 'val': val_loader}
