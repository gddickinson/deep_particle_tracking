"""ParticleTrackerModel: U-Net with a ConvLSTM at the bottleneck (create_model('simple'))."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Dict, Optional, Union
import logging

logger = logging.getLogger(__name__)

from models.blocks import EncoderBlock, DecoderBlock, ConvLSTMCell, decode_frames


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
                 use_batch_norm: bool = True,
                 embedding_dim: int = 32):
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

        # Add embedding head
        self.embedding_conv = nn.Sequential(
            nn.Conv2d(self.encoder_channels[0], self.encoder_channels[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(self.encoder_channels[0]) if use_batch_norm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.encoder_channels[0], embedding_dim, kernel_size=1)
        )

        self.embedding_dim = embedding_dim

    def forward(self, x):
        """
        Forward pass with embedding output.

        Args:
            x: Input tensor of shape (batch, frames, channels, height, width)

        Returns:
            Dictionary with 'positions' (batch, 1, out, H, W) for the last frame and 'embeddings'
            (batch, frames, embedding_dim, H, W), one map per input frame
        """
        # Fix for tensor dimension issue - handle both sequence and single frame inputs
        if x.dim() == 4:  # (batch, channels, height, width) - single frame
            # Add frames dimension
            x = x.unsqueeze(1)

        # Now we're sure x has shape (batch, frames, channels, height, width)
        batch_size, num_frames, channels, height, width = x.size()

        # Process frames with shared hidden state, keeping the ConvLSTM state after every frame
        hidden_state = None
        hiddens = []
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
            hiddens.append(hidden_state[0])  # Use hidden, not cell state

        # Decode every frame (state after frame t + frame t's skip connections): the embedding loss compares
        # embeddings of the same particle across frames, so it needs one embedding map per frame (with a single
        # last-frame map it had no frame pairs, was always 0 and the embeddings were never trained)
        features = decode_frames(self.decoders, hiddens, skip_connections)  # (B, T, C, H, W)

        # Position map for the last frame of the input sequence (as before); per-pixel embeddings for every frame
        # Sigmoid: a probability map, as FocalLoss (binary cross-entropy) and the predictor's threshold expect
        positions = torch.sigmoid(self.final_conv(features[:, -1])).unsqueeze(1)  # (B, 1, out, H, W)
        embeddings = self.embedding_conv(features.flatten(0, 1))
        embeddings = embeddings.reshape(batch_size, num_frames, *embeddings.shape[1:])  # (B, T, E, H, W)

        return {'positions': positions, 'embeddings': embeddings}
