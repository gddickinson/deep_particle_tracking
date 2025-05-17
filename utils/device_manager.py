"""
Device manager for GPU detection and allocation.
Supports Apple Metal Performance Shaders (MPS) for M1/M2/M3 Macs.
"""

import platform
import torch
import logging

logger = logging.getLogger(__name__)

class DeviceManager:
    """Manages device detection and allocation for deep learning models."""

    def __init__(self):
        self.device = None
        self.device_type = None
        self.detect_device()

    def detect_device(self):
        """Detects the best available device for computation."""
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.device_type = "cuda"
            gpu_name = torch.cuda.get_device_name(0)
            gpu_count = torch.cuda.device_count()
            logger.info(f"Using CUDA GPU: {gpu_name} (Device count: {gpu_count})")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            # Check for Apple Silicon MPS support
            self.device = torch.device("mps")
            self.device_type = "mps"
            apple_model = platform.processor()
            logger.info(f"Using Apple Metal Performance Shaders on {apple_model}")

            # Set default tensor type to float32 for MPS
            torch.set_default_dtype(torch.float32)
            logger.info("Set default tensor type to float32 for MPS compatibility")
        else:
            self.device = torch.device("cpu")
            self.device_type = "cpu"
            logger.info("Using CPU for computation")

    def move_to_device(self, data):
        """Moves tensors or models to the current device."""
        if isinstance(data, torch.Tensor):
            # For MPS, ensure float32 for floating point tensors
            if self.device_type == "mps" and torch.is_floating_point(data) and data.dtype != torch.float32:
                data = data.float()  # Convert to float32
            return data.to(self.device)
        elif hasattr(data, 'to'):
            return data.to(self.device)
        return data

    def get_device(self):
        """Returns the current device."""
        return self.device

    def get_device_info(self):
        """Returns information about the current device."""
        info = {
            "device_type": self.device_type,
            "device": str(self.device)
        }

        if self.device_type == "cuda":
            info["name"] = torch.cuda.get_device_name(0)
            info["memory_allocated"] = torch.cuda.memory_allocated(0)
            info["memory_reserved"] = torch.cuda.memory_reserved(0)
        elif self.device_type == "mps":
            info["name"] = platform.processor()
            info["tensor_dtype"] = str(torch.get_default_dtype())

        return info

    def is_mps_available(self):
        """Check if MPS is available for Apple Silicon devices."""
        return hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()

    def is_cuda_available(self):
        """Check if CUDA is available."""
        return torch.cuda.is_available()

    def __str__(self):
        """String representation of the device manager."""
        if self.device_type == "cuda":
            return f"DeviceManager(CUDA: {torch.cuda.get_device_name(0)})"
        elif self.device_type == "mps":
            return f"DeviceManager(MPS: {platform.processor()})"
        else:
            return "DeviceManager(CPU)"


# Singleton instance to be used across the application
device_manager = DeviceManager()
