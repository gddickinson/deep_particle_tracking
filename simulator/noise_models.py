"""
Noise models for simulating realistic microscopy image noise.
"""

import numpy as np
from typing import Tuple, Optional, Union
import logging

logger = logging.getLogger(__name__)


class NoiseModel:
    """Base class for noise models."""

    def apply(self, image: np.ndarray) -> np.ndarray:
        """
        Apply noise to an image.

        Args:
            image: Input image

        Returns:
            Noisy image
        """
        raise NotImplementedError("Subclasses must implement the apply method")


class GaussianNoise(NoiseModel):
    """Gaussian (additive) noise model."""

    def __init__(self, sigma: float = 1.0):
        """
        Initialize the Gaussian noise model.

        Args:
            sigma: Standard deviation of the Gaussian noise
        """
        self.sigma = sigma

    def apply(self, image: np.ndarray) -> np.ndarray:
        """
        Apply Gaussian noise to an image.

        Args:
            image: Input image

        Returns:
            Noisy image
        """
        noise = np.random.normal(0, self.sigma, image.shape)
        return image + noise


class PoissonNoise(NoiseModel):
    """Poisson (shot) noise model for photon counting."""

    def __init__(self, scaling_factor: float = 1.0):
        """
        Initialize the Poisson noise model.

        Args:
            scaling_factor: Scaling factor to convert image values to photon counts
        """
        self.scaling_factor = scaling_factor

    def apply(self, image: np.ndarray) -> np.ndarray:
        """
        Apply Poisson noise to an image.

        Args:
            image: Input image

        Returns:
            Noisy image
        """
        # Scale image to photon counts (must be non-negative)
        scaled_image = np.maximum(0, image * self.scaling_factor)

        # Apply Poisson noise
        noisy_image = np.random.poisson(scaled_image).astype(np.float32)

        # Scale back to original range
        return noisy_image / self.scaling_factor


class MixedNoise(NoiseModel):
    """Combined noise model with Poisson (shot) and Gaussian (read) noise."""

    def __init__(self,
                 photon_scaling: float = 1.0,
                 read_noise: float = 1.0,
                 offset: float = 0.0):
        """
        Initialize the mixed noise model.

        Args:
            photon_scaling: Scaling factor to convert image values to photon counts
            read_noise: Standard deviation of the Gaussian read noise
            offset: Camera offset (dark current or bias)
        """
        self.photon_scaling = photon_scaling
        self.read_noise = read_noise
        self.offset = offset

    def apply(self, image: np.ndarray) -> np.ndarray:
        """
        Apply realistic camera noise to an image (Poisson + Gaussian + offset).

        Args:
            image: Input image

        Returns:
            Noisy image
        """
        # Apply offset
        offset_image = image + self.offset

        # Scale to photon counts
        scaled_image = np.maximum(0, offset_image * self.photon_scaling)

        # Apply Poisson noise (photon shot noise)
        poisson_noisy = np.random.poisson(scaled_image).astype(np.float32)

        # Apply Gaussian noise (read noise)
        read_noisy = poisson_noisy + np.random.normal(0, self.read_noise, image.shape)

        # Scale back and subtract offset
        return (read_noisy / self.photon_scaling) - self.offset


class EMCCDNoise(NoiseModel):
    """
    Electron Multiplying CCD noise model with gain.
    Includes Poisson noise, EM gain noise, and readout noise.
    """

    def __init__(self,
                 photon_scaling: float = 1.0,
                 em_gain: float = 100.0,
                 read_noise: float = 1.0,
                 spurious_charge: float = 0.002,
                 offset: float = 100.0):
        """
        Initialize the EMCCD noise model.

        Args:
            photon_scaling: Scaling factor to convert image values to photon counts
            em_gain: Electron multiplying gain
            read_noise: Standard deviation of the Gaussian read noise
            spurious_charge: Clock-induced charge rate (spurious electrons)
            offset: Camera offset (bias level)
        """
        self.photon_scaling = photon_scaling
        self.em_gain = em_gain
        self.read_noise = read_noise
        self.spurious_charge = spurious_charge
        self.offset = offset

    def apply(self, image: np.ndarray) -> np.ndarray:
        """
        Apply realistic EMCCD noise to an image.

        Args:
            image: Input image

        Returns:
            Noisy image with EMCCD characteristics
        """
        # Scale to photoelectrons
        scaled_image = np.maximum(0, image * self.photon_scaling)

        # Apply Poisson noise for photon shot noise
        electrons = np.random.poisson(scaled_image).astype(np.float32)

        # Add spurious charge (clock-induced charge)
        if self.spurious_charge > 0:
            spurious = np.random.poisson(self.spurious_charge, image.shape).astype(np.float32)
            electrons += spurious

        # Apply EM gain process
        # For each input electron, the output follows a gamma distribution
        # with mean = gain and shape parameter = input electrons
        em_electrons = np.zeros_like(electrons)

        # Apply EM gain to each pixel with at least one electron
        mask = electrons > 0
        if np.any(mask):
            # Gamma distribution approximation of EM cascade
            # For pixel with n electrons, output mean = n*gain
            shape = electrons[mask]
            scale = self.em_gain
            em_electrons[mask] = np.random.gamma(shape, scale)

        # Add read noise after the EM stage
        final_electrons = em_electrons + np.random.normal(0, self.read_noise, image.shape)

        # Add camera offset
        final_signal = final_electrons + self.offset

        # Scale back to original units
        return final_signal / self.photon_scaling


class sCMOSNoise(NoiseModel):
    """
    Scientific CMOS noise model with pixel-dependent noise characteristics.
    """

    def __init__(self,
                 photon_scaling: float = 1.0,
                 read_noise_mean: float = 1.5,
                 read_noise_variance: float = 0.5,
                 dark_current: float = 0.05,
                 fixed_pattern_stddev: float = 0.01,
                 offset: float = 100.0):
        """
        Initialize the sCMOS noise model.

        Args:
            photon_scaling: Scaling factor to convert image values to photon counts
            read_noise_mean: Mean of the read noise standard deviation
            read_noise_variance: Variance of the read noise standard deviation
            dark_current: Average dark current (electrons/pixel/frame)
            fixed_pattern_stddev: Standard deviation of fixed pattern noise
            offset: Camera offset (bias level)
        """
        self.photon_scaling = photon_scaling
        self.read_noise_mean = read_noise_mean
        self.read_noise_variance = read_noise_variance
        self.dark_current = dark_current
        self.fixed_pattern_stddev = fixed_pattern_stddev
        self.offset = offset

        # Pixel-dependent noise maps
        self.read_noise_map = None
        self.fixed_pattern_map = None

    def initialize_noise_maps(self, shape: Tuple[int, int]):
        """
        Initialize pixel-dependent noise maps.

        Args:
            shape: Image shape (height, width)
        """
        # Generate read noise map (each pixel has its own read noise)
        if self.read_noise_variance > 0:
            self.read_noise_map = np.abs(
                np.random.normal(
                    self.read_noise_mean,
                    np.sqrt(self.read_noise_variance),
                    shape
                )
            )
        else:
            self.read_noise_map = np.ones(shape) * self.read_noise_mean

        # Generate fixed pattern noise map
        if self.fixed_pattern_stddev > 0:
            self.fixed_pattern_map = np.random.normal(1.0, self.fixed_pattern_stddev, shape)
        else:
            self.fixed_pattern_map = np.ones(shape)

    def apply(self, image: np.ndarray) -> np.ndarray:
        """
        Apply realistic sCMOS noise to an image.

        Args:
            image: Input image

        Returns:
            Noisy image with sCMOS characteristics
        """
        # Initialize noise maps if not already done
        if self.read_noise_map is None or self.read_noise_map.shape != image.shape:
            self.initialize_noise_maps(image.shape)

        # Apply fixed pattern noise (gain variations)
        pattern_image = image * self.fixed_pattern_map

        # Scale to photoelectrons
        scaled_image = np.maximum(0, pattern_image * self.photon_scaling)

        # Add dark current
        if self.dark_current > 0:
            dark = np.random.poisson(self.dark_current, image.shape).astype(np.float32)
            scaled_image += dark

        # Apply Poisson noise for photon shot noise
        electrons = np.random.poisson(scaled_image).astype(np.float32)

        # Apply pixel-dependent read noise
        read_noise = np.random.normal(0, self.read_noise_map)
        final_electrons = electrons + read_noise

        # Add camera offset
        final_signal = final_electrons + self.offset

        # Scale back to original units
        return final_signal / self.photon_scaling


class BackgroundNoise(NoiseModel):
    """
    Background fluorescence noise model for simulating out-of-focus fluorescence
    or autofluorescence in biological samples.
    """

    def __init__(self,
                 background_level: float = 1.0,
                 spatial_correlation: float = 10.0):
        """
        Initialize the background fluorescence noise model.

        Args:
            background_level: Mean background intensity level
            spatial_correlation: Correlation length for spatially correlated background
        """
        self.background_level = background_level
        self.spatial_correlation = spatial_correlation

    def apply(self, image: np.ndarray) -> np.ndarray:
        """
        Apply background fluorescence to an image.

        Args:
            image: Input image

        Returns:
            Image with added background
        """
        height, width = image.shape

        if self.spatial_correlation <= 1:
            # Uniform background
            background = np.ones(image.shape) * self.background_level
        else:
            # Generate spatially correlated background using low-pass filtered noise
            # Create random noise
            noise = np.random.normal(0, 1, (height, width))

            # Create frequency coordinates
            fy, fx = np.meshgrid(
                np.fft.fftfreq(width) * width,
                np.fft.fftfreq(height) * height,
                indexing='ij'  # Use 'ij' indexing to match (y, x) coordinate convention
            )
            f_radius = np.sqrt(fx**2 + fy**2)

            # Create low-pass filter in frequency domain
            sigma_f = width / (2 * np.pi * self.spatial_correlation)
            lpf = np.exp(-(f_radius**2) / (2 * sigma_f**2))

            # Apply filter in frequency domain
            noise_fft = np.fft.fft2(noise)
            filtered_fft = noise_fft * lpf
            filtered_noise = np.real(np.fft.ifft2(filtered_fft))

            # Scale and shift to desired background level
            filtered_noise = filtered_noise - np.min(filtered_noise)
            filtered_noise = filtered_noise / np.max(filtered_noise)
            background = filtered_noise * self.background_level

        # Add background to image
        return image + background
