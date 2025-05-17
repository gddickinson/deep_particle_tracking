"""
Deep neural network architecture for particle tracking in microscopy images.
"""

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


class ParticleTrackerModel(nn.Module):
    """
    Neural network model for particle tracking that processes multiple frames at once.
    Uses a U-Net architecture with ConvLSTM for temporal information.
    """

    def __init__(self,
                 input_channels: int = 1,
                 num_frames: int = 5,
                 base_filters: int = 64,
                 depth: int = 4,
                 output_channels: int = 1,
                 use_batch_norm: bool = True):
        """
        Initialize the model.

        Args:
            input_channels: Number of input channels per frame
            num_frames: Number of frames to process at once
            base_filters: Number of filters in the first layer
            depth: Depth of the U-Net
            output_channels: Number of output channels (1 for positions, 2 for positions+tracking)
            use_batch_norm: Whether to use batch normalization
        """
        super().__init__()

        self.input_channels = input_channels
        self.num_frames = num_frames
        self.base_filters = base_filters
        self.depth = depth
        self.output_channels = output_channels

        # Calculate channel dimensions for each level
        self.encoder_channels = []
        for i in range(depth+1):  # +1 to include bottleneck
            channels = base_filters * (2 ** min(i, depth-1))
            self.encoder_channels.append(channels)

        # Store encoder channel info for debugging
        logger.info(f"Encoder channels: {self.encoder_channels}")

        # Encoder blocks
        self.encoders = nn.ModuleList()
        curr_channels = input_channels

        for i in range(depth):
            out_channels = self.encoder_channels[i]
            self.encoders.append(EncoderBlock(curr_channels, out_channels, use_batch_norm))
            curr_channels = out_channels

        # Temporal ConvLSTM at bottleneck
        bottleneck_channels = self.encoder_channels[depth-1]
        self.convlstm = ConvLSTMCell(bottleneck_channels, bottleneck_channels)

        # Decoder blocks
        self.decoders = nn.ModuleList()

        for i in range(depth):
            # For decoder at level i:
            # - Input comes from level (depth-i)
            # - Output goes to level (depth-i-1)
            # - Skip connection comes from level (depth-i-1)
            decoder_level = depth - i - 1

            # Calculate channel dimensions
            decoder_in_channels = self.encoder_channels[decoder_level+1]
            decoder_out_channels = self.encoder_channels[decoder_level]
            skip_channels = self.encoder_channels[decoder_level]

            logger.info(f"Decoder {i}: in={decoder_in_channels}, out={decoder_out_channels}, skip={skip_channels}")

            self.decoders.append(
                DecoderBlock(
                    in_channels=decoder_in_channels,
                    out_channels=decoder_out_channels,
                    skip_channels=skip_channels,
                    use_batch_norm=use_batch_norm
                )
            )

        # Final output convolution
        self.final_conv = nn.Conv2d(self.encoder_channels[0], output_channels, kernel_size=1)

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x: Input tensor of shape (batch, frames, channels, height, width)

        Returns:
            Output prediction tensor
        """
        # Fix for tensor dimension issue - handle both sequence and single frame inputs
        if x.dim() == 4:  # (batch, channels, height, width) - single frame
            # Add frames dimension
            x = x.unsqueeze(1)

        # Now we're sure x has shape (batch, frames, channels, height, width)
        batch_size, num_frames, channels, height, width = x.size()

        # We'll process each frame sequentially through the encoder
        # and use ConvLSTM to maintain temporal information
        hidden_state = None
        skip_connections = [[] for _ in range(self.depth)]

        for frame_idx in range(num_frames):
            # Get current frame
            frame = x[:, frame_idx]

            # Pass through encoder blocks
            features = frame
            for i, encoder in enumerate(self.encoders):
                features, skip = encoder(features)
                skip_connections[i].append(skip)

            # Apply ConvLSTM at the bottleneck
            hidden_state = self.convlstm(features, hidden_state)

        # Use the final hidden state for decoding
        features = hidden_state[0]  # Use hidden, not cell state

        # Pass through decoder blocks with skip connections
        for i, decoder in enumerate(self.decoders):
            # Use the skip connection from the last frame
            skip_idx = self.depth - i - 1
            skip_features = skip_connections[skip_idx][-1]
            features = decoder(features, skip_features)

        # Final convolution to get output
        output = self.final_conv(features)

        return output


class DualBranchParticleTracker(nn.Module):
    """
    Advanced particle tracking model with dual branches:
    1. Localization branch for precise particle positions
    2. Association branch for tracking particles across frames
    """

    def __init__(self,
                 input_channels: int = 1,
                 num_frames: int = 5,
                 base_filters: int = 64,
                 depth: int = 4,
                 use_batch_norm: bool = True,
                 upsample_factor: int = 8):
        """
        Initialize the dual branch particle tracker model.

        Args:
            input_channels: Number of input channels per frame
            num_frames: Number of frames to process at once
            base_filters: Number of filters in the first layer
            depth: Depth of the U-Net
            use_batch_norm: Whether to use batch normalization
            upsample_factor: Factor by which to upsample for super-resolution
        """
        super().__init__()

        self.input_channels = input_channels
        self.num_frames = num_frames
        self.base_filters = base_filters
        self.depth = depth
        self.upsample_factor = upsample_factor

        # Shared encoder
        self.shared_encoders = nn.ModuleList()
        curr_channels = input_channels

        for i in range(depth):
            out_channels = base_filters * (2 ** i)
            self.shared_encoders.append(EncoderBlock(curr_channels, out_channels, use_batch_norm))
            curr_channels = out_channels

        # Temporal ConvLSTM for shared features
        self.shared_convlstm = ConvLSTMCell(curr_channels, curr_channels)

        # Localization branch (super-resolution)
        self.loc_decoders = nn.ModuleList()

        for i in range(depth - 1, -1, -1):
            in_channels = base_filters * (2 ** i) * 2  # Double because of skip connection
            out_channels = base_filters * (2 ** max(0, i - 1))

            if i == 0:
                out_channels = base_filters

            self.loc_decoders.append(DecoderBlock(in_channels, out_channels, use_batch_norm))

        # Final super-resolution upsampling for localization
        self.loc_upsampler = nn.Sequential(
            nn.Conv2d(base_filters, base_filters * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),  # x2 upsampling
            nn.ReLU(inplace=True),
            nn.Conv2d(base_filters, base_filters * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),  # x2 upsampling
            nn.ReLU(inplace=True),
            nn.Conv2d(base_filters, base_filters * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),  # x2 upsampling
            nn.ReLU(inplace=True),
            nn.Conv2d(base_filters, 1, kernel_size=1),
            nn.Sigmoid()  # Probability map for particle positions
        )

        # Association branch (tracking)
        self.assoc_decoders = nn.ModuleList()

        for i in range(depth - 1, -1, -1):
            in_channels = base_filters * (2 ** i) * 2  # Double because of skip connection
            out_channels = base_filters * (2 ** max(0, i - 1))

            if i == 0:
                out_channels = base_filters

            self.assoc_decoders.append(DecoderBlock(in_channels, out_channels, use_batch_norm))

        # Final convolution for association map
        self.assoc_final = nn.Sequential(
            nn.Conv2d(base_filters, base_filters, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_filters, num_frames, kernel_size=1),
            nn.Softmax(dim=1)  # Association probabilities across frames
        )

    def forward(self, x):
        """
        Forward pass of the dual branch model.

        Args:
            x: Input tensor of shape (batch, frames, channels, height, width)

        Returns:
            Dictionary with two outputs:
            - 'positions': High-resolution probability map for particle positions
            - 'associations': Association map for tracking particles across frames
        """
        batch_size, num_frames, channels, height, width = x.size()

        # Process each frame through shared encoder
        hidden_state = None
        skip_connections = [[] for _ in range(self.depth)]

        for frame_idx in range(num_frames):
            # Get current frame
            frame = x[:, frame_idx]

            # Pass through encoder blocks
            features = frame
            for i, encoder in enumerate(self.shared_encoders):
                features, skip = encoder(features)
                skip_connections[i].append(skip)

            # Apply ConvLSTM at the bottleneck
            hidden_state = self.shared_convlstm(features, hidden_state)

        # Get final bottleneck features
        bottleneck_features = hidden_state[0]  # Use hidden, not cell state

        # Localization branch
        loc_features = bottleneck_features
        for i, decoder in enumerate(self.loc_decoders):
            # Use the skip connection from the last frame
            skip_idx = self.depth - i - 1
            skip_features = skip_connections[skip_idx][-1]
            loc_features = decoder(loc_features, skip_features)

        # Super-resolution upsampling for localization
        positions = self.loc_upsampler(loc_features)

        # Association branch
        assoc_features = bottleneck_features
        for i, decoder in enumerate(self.assoc_decoders):
            # Use the skip connection from the middle frame
            skip_idx = self.depth - i - 1
            mid_frame_idx = num_frames // 2
            skip_features = skip_connections[skip_idx][mid_frame_idx]
            assoc_features = decoder(assoc_features, skip_features)

        # Final convolution for association map
        associations = self.assoc_final(assoc_features)

        return {'positions': positions, 'associations': associations}


class AttentiveParticleTracker(nn.Module):
    """
    Attentive particle tracker with temporal and self-attention mechanisms
    for robust tracking in dense environments.
    """

    def __init__(self,
                 input_channels: int = 1,
                 num_frames: int = 5,
                 base_filters: int = 64,
                 depth: int = 4,
                 heads: int = 8,
                 use_batch_norm: bool = True,
                 upsample_factor: int = 8):
        """
        Initialize the attentive particle tracker.

        Args:
            input_channels: Number of input channels per frame
            num_frames: Number of frames to process at once
            base_filters: Number of filters in the first layer
            depth: Depth of the network
            heads: Number of attention heads
            use_batch_norm: Whether to use batch normalization
            upsample_factor: Factor by which to upsample for super-resolution
        """
        super().__init__()

        self.input_channels = input_channels
        self.num_frames = num_frames
        self.base_filters = base_filters
        self.depth = depth
        self.heads = heads
        self.upsample_factor = upsample_factor

        # Initial convolution
        self.initial_conv = ConvBlock(
            input_channels,
            base_filters,
            use_batch_norm=use_batch_norm
        )

        # Encoder blocks
        self.encoders = nn.ModuleList()
        curr_channels = base_filters

        for i in range(depth):
            out_channels = base_filters * (2 ** i)
            self.encoders.append(EncoderBlock(curr_channels, out_channels, use_batch_norm))
            curr_channels = out_channels

        # Bottleneck attention
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=curr_channels,
            num_heads=heads,
            batch_first=True
        )

        self.spatial_attention_norm = nn.LayerNorm([curr_channels])

        # Decoder blocks
        self.decoders = nn.ModuleList()

        for i in range(depth - 1, -1, -1):
            in_channels = base_filters * (2 ** i) * 2  # Double because of skip connection
            out_channels = base_filters * (2 ** max(0, i - 1))

            if i == 0:
                out_channels = base_filters

            self.decoders.append(DecoderBlock(in_channels, out_channels, use_batch_norm))

        # Position prediction branch
        self.position_upsampler = nn.Sequential(
            nn.Conv2d(base_filters, base_filters * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),  # x2 upsampling
            nn.ReLU(inplace=True),
            nn.Conv2d(base_filters, base_filters * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),  # x2 upsampling
            nn.ReLU(inplace=True),
            nn.Conv2d(base_filters, 1, kernel_size=1),
            nn.Sigmoid()  # Probability map for particle positions
        )

        # Tracking/embedding branch
        self.embedding_conv = nn.Sequential(
            nn.Conv2d(base_filters, base_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_filters, 32, kernel_size=1)  # Embedding dimension
        )

    def reshape_for_attention(self, x):
        """Reshape spatial dimensions to sequence for attention."""
        b, c, h, w = x.shape
        return x.flatten(2).permute(0, 2, 1)  # B, HW, C

    def reshape_from_attention(self, x, h, w):
        """Reshape from sequence back to spatial dimensions."""
        b, hw, c = x.shape
        return x.permute(0, 2, 0, 1).reshape(b, c, h, w)

    def forward(self, x):
        """
        Forward pass of the attentive particle tracker.

        Args:
            x: Input tensor of shape (batch, frames, channels, height, width)

        Returns:
            Dictionary with:
            - 'positions': Super-resolution probability map for particle positions
            - 'embeddings': Embeddings for tracking particles (association by similarity)
        """
        batch_size, num_frames, channels, height, width = x.size()

        # Process each frame and store features
        all_features = []
        all_skips = [[] for _ in range(self.depth)]

        for frame_idx in range(num_frames):
            frame = x[:, frame_idx]
            frame_features = self.initial_conv(frame)

            # Encoder
            features = frame_features
            for i, encoder in enumerate(self.encoders):
                features, skip = encoder(features)
                all_skips[i].append(skip)

            all_features.append(features)

        # Combine features from all frames
        bottleneck_seq = torch.stack(all_features, dim=1)  # (B, F, C, H, W)
        _, f, c, h, w = bottleneck_seq.shape

        # Reshape for temporal attention
        bottleneck_attn = bottleneck_seq.flatten(3).permute(0, 3, 1, 2)  # (B, HW, F, C)
        b, hw, f, c = bottleneck_attn.shape
        bottleneck_attn = bottleneck_attn.reshape(b * hw, f, c)  # (B*HW, F, C)

        # Apply temporal attention
        bottleneck_attn, _ = self.temporal_attention(
            bottleneck_attn, bottleneck_attn, bottleneck_attn
        )

        # Reshape back and take the central frame features with temporal context
        bottleneck_attn = bottleneck_attn.reshape(b, hw, f, c)
        central_idx = num_frames // 2
        bottleneck_attn = bottleneck_attn[:, :, central_idx].reshape(b, h, w, c)
        bottleneck_attn = bottleneck_attn.permute(0, 3, 1, 2)  # (B, C, H, W)

        # Decoder with skip connections
        features = bottleneck_attn
        for i, decoder in enumerate(self.decoders):
            # Use the skip connection from the central frame
            skip_idx = self.depth - i - 1
            central_skip = all_skips[skip_idx][central_idx]
            features = decoder(features, central_skip)

        # Position prediction
        positions = self.position_upsampler(features)

        # Embedding prediction for tracking
        embeddings = self.embedding_conv(features)

        return {'positions': positions, 'embeddings': embeddings}


# Define model factory to create models with different configurations
def create_model(model_type, **kwargs):
    """
    Factory function to create a model with the specified configuration.

    Args:
        model_type: Type of model to create ('simple', 'dual', 'attentive')
        **kwargs: Model configuration parameters

    Returns:
        Instantiated model
    """
    if model_type == 'simple':
        return ParticleTrackerModel(**kwargs)
    elif model_type == 'dual':
        return DualBranchParticleTracker(**kwargs)
    elif model_type == 'attentive':
        return AttentiveParticleTracker(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
