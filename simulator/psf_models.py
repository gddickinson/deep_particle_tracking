"""
PSF models for simulating point spread functions in fluorescence microscopy.
"""

import numpy as np
from typing import Tuple, Optional, Dict, List, Union, Callable
import logging
from scipy import special

logger = logging.getLogger(__name__)


class PSFModel:
    """Base class for point spread function models."""

    def __init__(self, image_size: Tuple[int, int] = (512, 512)):
        """
        Initialize the PSF model.

        Args:
            image_size: Image dimensions (height, width) in pixels
        """
        self.image_size = image_size

    def generate(self,
                positions: List[Tuple[float, float]],
                intensities: Optional[List[float]] = None,
                sizes: Optional[List[float]] = None) -> np.ndarray:
        """
        Generate an image with PSFs at the specified positions.

        Args:
            positions: List of (y, x) positions in pixels
            intensities: List of intensity values for each position
            sizes: List of size modifiers for each position

        Returns:
            2D numpy array containing the generated image
        """
        raise NotImplementedError("Subclasses must implement the generate method")


class GaussianPSF(PSFModel):
    """Gaussian point spread function model."""

    def __init__(self,
                 image_size: Tuple[int, int] = (512, 512),
                 sigma: float = 1.0):
        """
        Initialize the Gaussian PSF model.

        Args:
            image_size: Image dimensions (height, width) in pixels
            sigma: Standard deviation of the Gaussian PSF in pixels
        """
        super().__init__(image_size)
        self.sigma = sigma

    def generate(self,
                positions: List[Tuple[float, float]],
                intensities: Optional[List[float]] = None,
                sizes: Optional[List[float]] = None) -> np.ndarray:
        """
        Generate an image with Gaussian PSFs at the specified positions.

        Args:
            positions: List of (y, x) positions in pixels
            intensities: List of intensity values for each position
            sizes: List of size modifiers for each position

        Returns:
            2D numpy array containing the generated image
        """
        height, width = self.image_size
        image = np.zeros((height, width), dtype=np.float32)

        # Default values if not provided
        if intensities is None:
            intensities = [1.0] * len(positions)

        if sizes is None:
            sizes = [1.0] * len(positions)

        # Create meshgrid for vectorized computation
        # Using indexing='ij' ensures y_coords corresponds to rows (y) and x_coords to columns (x)
        y_coords, x_coords = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')

        # Add each PSF to the image
        for (y, x), intensity, size in zip(positions, intensities, sizes):
            # Adjust sigma based on size parameter
            sigma = self.sigma * size

            # Calculate Gaussian PSF - note coordinates are now (y, x)
            gaussian = intensity * np.exp(
                -((y_coords - y) ** 2 + (x_coords - x) ** 2) / (2 * sigma ** 2)
            )

            # Add to the image
            image += gaussian

        return image


class AiryDiskPSF(PSFModel):
    """Airy disk point spread function model for diffraction-limited optics."""

    def __init__(self,
                 image_size: Tuple[int, int] = (512, 512),
                 airy_radius: float = 1.22):
        """
        Initialize the Airy disk PSF model.

        Args:
            image_size: Image dimensions (height, width) in pixels
            airy_radius: Radius of the first zero of the Airy disk in pixels
                         (typically 1.22 * wavelength / (2 * NA))
        """
        super().__init__(image_size)
        self.airy_radius = airy_radius

    def generate(self,
                positions: List[Tuple[float, float]],
                intensities: Optional[List[float]] = None,
                sizes: Optional[List[float]] = None) -> np.ndarray:
        """
        Generate an image with Airy disk PSFs at the specified positions.

        Args:
            positions: List of (y, x) positions in pixels
            intensities: List of intensity values for each position
            sizes: List of size modifiers for each position

        Returns:
            2D numpy array containing the generated image
        """
        height, width = self.image_size
        image = np.zeros((height, width), dtype=np.float32)

        # Default values if not provided
        if intensities is None:
            intensities = [1.0] * len(positions)

        if sizes is None:
            sizes = [1.0] * len(positions)

        # Create meshgrid for vectorized computation
        # Using indexing='ij' ensures y_coords corresponds to rows (y) and x_coords to columns (x)
        y_coords, x_coords = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')

        # Add each PSF to the image
        for (y, x), intensity, size in zip(positions, intensities, sizes):
            # Adjust airy radius based on size parameter
            radius = self.airy_radius * size

            # Calculate distances from the center using (y, x) coordinates
            r = np.sqrt((y_coords - y) ** 2 + (x_coords - x) ** 2)

            # Avoid division by zero at the center
            r_safe = np.where(r > 0, r, 1e-10)

            # Calculate normalized radial coordinate
            v = 2 * np.pi * r_safe / radius

            # Calculate Airy disk pattern: [2*J1(v)/v]^2
            # Where J1 is the Bessel function of the first kind of order 1
            airy = np.where(
                r > 0,
                (2 * special.j1(v) / v) ** 2,
                1.0  # At the center, the limit is 1
            )

            # Add to the image with intensity scaling
            image += intensity * airy

        return image


class AstigmaticPSF(PSFModel):
    """Astigmatic PSF model for 3D localization microscopy."""

    def __init__(self,
                 image_size: Tuple[int, int] = (512, 512),
                 sigma_x: float = 1.0,
                 sigma_y: float = 1.0,
                 z_dependence: Optional[Callable] = None):
        """
        Initialize the astigmatic PSF model.

        Args:
            image_size: Image dimensions (height, width) in pixels
            sigma_x: Standard deviation along x-axis in pixels
            sigma_y: Standard deviation along y-axis in pixels
            z_dependence: Function that modifies sigma_x and sigma_y based on z
                          Should take z as input and return (sigma_x_factor, sigma_y_factor)
        """
        super().__init__(image_size)
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y

        # Default z-dependence function if none provided
        if z_dependence is None:
            # Simple astigmatic dependence where x and y focus at different z positions
            def default_z_dependence(z):
                # z is in micrometers, centered at 0 (focal plane)
                # Returns factors to multiply with sigma_x and sigma_y
                sigma_x_factor = 1.0 + 0.5 * (z ** 2)
                sigma_y_factor = 1.0 + 0.5 * ((z - 0.5) ** 2)
                return sigma_x_factor, sigma_y_factor

            self.z_dependence = default_z_dependence
        else:
            self.z_dependence = z_dependence

    def generate(self,
                positions: List[Tuple[Union[float, float, Optional[float]]]],
                intensities: Optional[List[float]] = None,
                sizes: Optional[List[float]] = None) -> np.ndarray:
        """
        Generate an image with astigmatic PSFs at the specified positions.

        Args:
            positions: List of (y, x, z) positions in pixels/microns (z is optional)
            intensities: List of intensity values for each position
            sizes: List of size modifiers for each position

        Returns:
            2D numpy array containing the generated image
        """
        height, width = self.image_size
        image = np.zeros((height, width), dtype=np.float32)

        # Default values if not provided
        if intensities is None:
            intensities = [1.0] * len(positions)

        if sizes is None:
            sizes = [1.0] * len(positions)

        # Create meshgrid for vectorized computation
        # Using indexing='ij' ensures y_coords corresponds to rows (y) and x_coords to columns (x)
        y_coords, x_coords = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')

        # Add each PSF to the image
        for pos_idx, pos in enumerate(positions):
            # Handle both (y, x) and (y, x, z) formats
            if len(pos) >= 3 and pos[2] is not None:
                y, x, z = pos
                # Adjust sigmas based on z-position
                sigma_x_factor, sigma_y_factor = self.z_dependence(z)
                sigma_x = self.sigma_x * sigma_x_factor
                sigma_y = self.sigma_y * sigma_y_factor
            else:
                y, x = pos[0], pos[1]
                sigma_x = self.sigma_x
                sigma_y = self.sigma_y

            intensity = intensities[pos_idx]
            size = sizes[pos_idx]

            # Adjust sigmas based on size parameter
            sigma_x *= size
            sigma_y *= size

            # Calculate astigmatic Gaussian PSF - note (y, x) coordinate order
            gaussian = intensity * np.exp(
                -(
                    ((y_coords - y) ** 2) / (2 * sigma_y ** 2) +
                    ((x_coords - x) ** 2) / (2 * sigma_x ** 2)
                )
            )

            # Add to the image
            image += gaussian

        return image


class EmpiricPSF(PSFModel):
    """Empirically measured PSF model from experimental data."""

    def __init__(self,
                 psf_template: np.ndarray,
                 image_size: Tuple[int, int] = (512, 512),
                 psf_center: Optional[Tuple[int, int]] = None):
        """
        Initialize with an empirically measured PSF template.

        Args:
            psf_template: 2D array containing the measured PSF
            image_size: Image dimensions (height, width) in pixels
            psf_center: Center position (y, x) of the PSF in the template (if None, uses center of template)
        """
        super().__init__(image_size)
        self.psf_template = psf_template

        # Determine PSF center if not provided
        if psf_center is None:
            self.psf_center = (psf_template.shape[0] // 2, psf_template.shape[1] // 2)
        else:
            self.psf_center = psf_center

    def generate(self,
                positions: List[Tuple[float, float]],
                intensities: Optional[List[float]] = None,
                sizes: Optional[List[float]] = None) -> np.ndarray:
        """
        Generate an image with empirical PSFs at the specified positions.

        Args:
            positions: List of (y, x) positions in pixels
            intensities: List of intensity values for each position
            sizes: List of size modifiers for each position

        Returns:
            2D numpy array containing the generated image
        """
        height, width = self.image_size
        image = np.zeros((height, width), dtype=np.float32)

        # Default values if not provided
        if intensities is None:
            intensities = [1.0] * len(positions)

        if sizes is None:
            sizes = [1.0] * len(positions)

        # Get template dimensions
        psf_height, psf_width = self.psf_template.shape
        center_y, center_x = self.psf_center

        # Add each PSF to the image
        for (y, x), intensity, size in zip(positions, intensities, sizes):
            # Calculate the region to place the PSF
            y_int, x_int = int(round(y)), int(round(x))

            # Calculate the bounds for placing the PSF
            y_min = max(0, y_int - center_y)
            x_min = max(0, x_int - center_x)
            y_max = min(height, y_int + (psf_height - center_y))
            x_max = min(width, x_int + (psf_width - center_x))

            # Calculate the bounds for the PSF template
            template_y_min = max(0, center_y - y_int)
            template_x_min = max(0, center_x - x_int)
            template_y_max = min(psf_height, center_y + (height - y_int))
            template_x_max = min(psf_width, center_x + (width - x_int))

            # Skip if the PSF is completely outside the image
            if y_min >= y_max or x_min >= x_max:
                continue

            # Handle size change if needed
            if size != 1.0:
                # This is a simplification - for proper scaling, we should use interpolation
                scaled_template = self.psf_template * size
            else:
                scaled_template = self.psf_template

            # Add the PSF to the image with the specified intensity
            image[y_min:y_max, x_min:x_max] += (
                intensity *
                scaled_template[template_y_min:template_y_max, template_x_min:template_x_max]
            )

        return image
