"""DualBranchParticleTracker: shared encoder + ConvLSTM, localisation and association decoders
(create_model('dual'))."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Dict, Optional, Union
import logging

logger = logging.getLogger(__name__)

from models.blocks import EncoderBlock, DecoderBlock, ConvLSTMCell, super_resolution_head


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

        # Decoder for level i upsamples the current features and concatenates the level-i skip connection
        # (base_filters * 2**i channels); DecoderBlock needs skip_channels explicitly (passing use_batch_norm in
        # its place made every forward pass fail with a channel mismatch)
        dec_in = curr_channels
        for i in range(depth - 1, -1, -1):
            out_channels = base_filters * (2 ** max(0, i - 1))

            if i == 0:
                out_channels = base_filters

            self.loc_decoders.append(DecoderBlock(dec_in, out_channels, skip_channels=base_filters * (2 ** i),
                                             use_batch_norm=use_batch_norm))
            dec_in = out_channels

        # Final super-resolution upsampling for localization
        self.loc_upsampler = super_resolution_head(base_filters, upsample_factor)

        # Association branch (tracking)
        self.assoc_decoders = nn.ModuleList()

        # Decoder for level i upsamples the current features and concatenates the level-i skip connection
        # (base_filters * 2**i channels); DecoderBlock needs skip_channels explicitly (passing use_batch_norm in
        # its place made every forward pass fail with a channel mismatch)
        dec_in = curr_channels
        for i in range(depth - 1, -1, -1):
            out_channels = base_filters * (2 ** max(0, i - 1))

            if i == 0:
                out_channels = base_filters

            self.assoc_decoders.append(DecoderBlock(dec_in, out_channels, skip_channels=base_filters * (2 ** i),
                                             use_batch_norm=use_batch_norm))
            dec_in = out_channels

        # Final convolution for association map
        self.assoc_final = nn.Sequential(
            nn.Conv2d(base_filters, base_filters, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_filters, num_frames, kernel_size=1),
            nn.Softmax(dim=1)  # Association probabilities across frames
        )

    def forward(self, x):
        """
        Forward pass with improved temporal handling.

        Args:
            x: Input tensor of shape (batch, frames, channels, height, width)

        Returns:
            Dictionary with 'positions', 'embeddings', and 'associations'
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
            # Use the skip connection from the last frame for most recent information
            skip_idx = self.depth - i - 1
            skip_features = skip_connections[skip_idx][-1]
            loc_features = decoder(loc_features, skip_features)

        # Super-resolution upsampling for localization
        positions = self.loc_upsampler(loc_features)

        # Association branch - generates association scores between particles in consecutive frames
        assoc_features = bottleneck_features
        for i, decoder in enumerate(self.assoc_decoders):
            # Use the skip connection from the middle frame for context
            skip_idx = self.depth - i - 1
            mid_frame_idx = num_frames // 2
            skip_features = skip_connections[skip_idx][mid_frame_idx]
            assoc_features = decoder(assoc_features, skip_features)

        # Association map shows relationships between frames
        associations = self.assoc_final(assoc_features)

        # Add embedding output for particle identity tracking
        embeddings = assoc_features

        return {
            'positions': positions,
            'associations': associations,
            'embeddings': embeddings
        }
