"""
Building blocks shared by the particle tracking networks: ConvBlock, EncoderBlock, DecoderBlock,
ConvLSTMCell and super_resolution_head.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Dict, Optional, Union
import logging

logger = logging.getLogger(__name__)


class ConvBlock(nn.Module):
    """Basic convolutional block with batch normalization and activation."""

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 padding: int = 1,
                 stride: int = 1,
                 use_batch_norm: bool = True,
                 activation: nn.Module = nn.ReLU(inplace=True)):
        """
        Initialize convolutional block.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Size of the convolution kernel
            padding: Padding size
            stride: Convolution stride
            use_batch_norm: Whether to use batch normalization
            activation: Activation function to use
        """
        super().__init__()

        layers = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                stride=stride,
                bias=not use_batch_norm
            )
        ]

        if use_batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))

        if activation is not None:
            layers.append(activation)

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        """Forward pass through the block."""
        return self.block(x)


class EncoderBlock(nn.Module):
    """Encoder block for U-Net architecture."""

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 use_batch_norm: bool = True):
        """
        Initialize encoder block.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            use_batch_norm: Whether to use batch normalization
        """
        super().__init__()

        self.conv1 = ConvBlock(
            in_channels,
            out_channels,
            use_batch_norm=use_batch_norm
        )

        self.conv2 = ConvBlock(
            out_channels,
            out_channels,
            use_batch_norm=use_batch_norm
        )

        self.pool = nn.MaxPool2d(2, 2)

        # Store output channel info for debugging
        self.out_channels = out_channels

    def forward(self, x):
        """
        Forward pass through the encoder block.

        Returns:
            Tuple of (pooled output, skip connection tensor)
        """
        skip_features = self.conv2(self.conv1(x))
        return self.pool(skip_features), skip_features



class DecoderBlock(nn.Module):
    """Decoder block for U-Net architecture."""

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 skip_channels: int,  # NEW: explicitly specify skip connection channels
                 use_batch_norm: bool = True):
        """
        Initialize decoder block.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            skip_channels: Number of channels in skip connection
            use_batch_norm: Whether to use batch normalization
        """
        super().__init__()

        # Upsample input features
        self.up = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2
        )

        # After concatenation with skip features, channels = out_channels + skip_channels
        concat_channels = out_channels + skip_channels

        # First convolution after concatenation
        self.conv1 = ConvBlock(
            concat_channels,  # Explicitly use the calculated concatenated channels
            out_channels,
            use_batch_norm=use_batch_norm
        )

        # Second convolution
        self.conv2 = ConvBlock(
            out_channels,
            out_channels,
            use_batch_norm=use_batch_norm
        )

    def forward(self, x, skip_features):
        """
        Forward pass through the decoder block.

        Args:
            x: Input tensor
            skip_features: Skip connection tensor from the encoder

        Returns:
            Output tensor
        """
        # Debug prints for shapes
        # print(f"DecoderBlock input x shape: {x.shape}, skip shape: {skip_features.shape}")

        # Upsample input
        x = self.up(x)
        # print(f"After upsampling: {x.shape}")

        # Handle case where input dimensions don't match
        diff_h = skip_features.size()[2] - x.size()[2]
        diff_w = skip_features.size()[3] - x.size()[3]

        x = F.pad(x, [diff_w // 2, diff_w - diff_w // 2,
                      diff_h // 2, diff_h - diff_h // 2])
        # print(f"After padding: {x.shape}")

        # Concatenate skip features
        x = torch.cat([skip_features, x], dim=1)
        # print(f"After concatenation: {x.shape}")

        # Apply convolutions
        x = self.conv1(x)
        # print(f"After conv1: {x.shape}")
        x = self.conv2(x)
        # print(f"After conv2: {x.shape}")

        return x


class ConvLSTMCell(nn.Module):
    """Convolutional LSTM cell for spatiotemporal learning."""

    def __init__(self,
                 input_channels: int,
                 hidden_channels: int,
                 kernel_size: int = 3,
                 bias: bool = True):
        """
        Initialize convolutional LSTM cell.

        Args:
            input_channels: Number of input channels
            hidden_channels: Number of hidden channels
            kernel_size: Size of the convolution kernel
            bias: Whether to use bias
        """
        super().__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2  # Same padding
        self.bias = bias

        # Combined gate convolutions (input, forget, output, cell)
        self.conv = nn.Conv2d(
            in_channels=input_channels + hidden_channels,
            out_channels=4 * hidden_channels,
            kernel_size=kernel_size,
            padding=self.padding,
            bias=bias
        )

    def forward(self, x, hidden_state=None):
        """
        Forward pass through the ConvLSTM cell.

        Args:
            x: Input tensor of shape (batch, channels, height, width)
            hidden_state: Tuple of (hidden, cell) or None for initial state

        Returns:
            Tuple of (new_hidden, new_cell)
        """
        batch_size, _, height, width = x.size()

        # Initialize hidden state if needed
        if hidden_state is None:
            hidden = torch.zeros(batch_size, self.hidden_channels, height, width, device=x.device)
            cell = torch.zeros(batch_size, self.hidden_channels, height, width, device=x.device)
        else:
            hidden, cell = hidden_state

        # Concatenate input and hidden state
        combined = torch.cat([x, hidden], dim=1)

        # Calculate all gates at once
        gates = self.conv(combined)

        # Split into separate gates
        input_gate, forget_gate, cell_gate, output_gate = gates.chunk(4, dim=1)

        # Apply activations
        input_gate = torch.sigmoid(input_gate)
        forget_gate = torch.sigmoid(forget_gate)
        cell_gate = torch.tanh(cell_gate)
        output_gate = torch.sigmoid(output_gate)

        # Update cell state
        new_cell = forget_gate * cell + input_gate * cell_gate

        # Update hidden state
        new_hidden = output_gate * torch.tanh(new_cell)

        return new_hidden, new_cell


def super_resolution_head(channels: int, upsample_factor: int) -> nn.Sequential:
    """Probability map at upsample_factor x the input resolution: one (conv, PixelShuffle(2), ReLU) stage per
    factor of 2, then a 1x1 conv and a sigmoid. upsample_factor must be a power of 2 (1 = no upsampling).
    (The models used to hard-code 3 stages (Dual Branch, 8x) and 2 stages (Attentive, 4x) whatever was asked.)"""
    stages = int(round(math.log2(upsample_factor))) if upsample_factor >= 1 else -1
    if stages < 0 or 2 ** stages != upsample_factor:
        raise ValueError(f"upsample_factor must be a power of 2, got {upsample_factor}")
    layers = []
    for _ in range(stages):
        layers += [nn.Conv2d(channels, channels * 4, kernel_size=3, padding=1), nn.PixelShuffle(2), nn.ReLU(inplace=True)]
    layers += [nn.Conv2d(channels, 1, kernel_size=1), nn.Sigmoid()]
    return nn.Sequential(*layers)
