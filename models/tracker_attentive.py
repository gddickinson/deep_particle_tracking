"""AttentiveParticleTracker: per-frame encoder, temporal multi-head attention at the bottleneck
(create_model('attentive'))."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Dict, Optional, Union
import logging

logger = logging.getLogger(__name__)

from models.blocks import ConvBlock, EncoderBlock, DecoderBlock, super_resolution_head


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

        # Decoder for level i upsamples the current features and concatenates the level-i skip connection
        # (base_filters * 2**i channels); DecoderBlock needs skip_channels explicitly (passing use_batch_norm in
        # its place made every forward pass fail with a channel mismatch)
        dec_in = curr_channels
        for i in range(depth - 1, -1, -1):
            out_channels = base_filters * (2 ** max(0, i - 1))

            if i == 0:
                out_channels = base_filters

            self.decoders.append(DecoderBlock(dec_in, out_channels, skip_channels=base_filters * (2 ** i),
                                             use_batch_norm=use_batch_norm))
            dec_in = out_channels

        # Position prediction branch
        self.position_upsampler = super_resolution_head(base_filters, upsample_factor)

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
        return x.permute(0, 2, 1).reshape(b, c, h, w)

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

        # Reshape back and take the last frame's features with temporal context (the frame the predictor and
        # CombinedLoss assign the output to; this used the central frame)
        bottleneck_attn = bottleneck_attn.reshape(b, hw, f, c)
        central_idx = num_frames - 1
        bottleneck_attn = bottleneck_attn[:, :, central_idx].reshape(b, h, w, c)
        bottleneck_attn = bottleneck_attn.permute(0, 3, 1, 2)  # (B, C, H, W)

        # Decoder with skip connections
        features = bottleneck_attn
        for i, decoder in enumerate(self.decoders):
            # Use the skip connection from the last frame
            skip_idx = self.depth - i - 1
            central_skip = all_skips[skip_idx][central_idx]
            features = decoder(features, central_skip)

        # Position prediction
        positions = self.position_upsampler(features)

        # Embedding prediction for tracking
        embeddings = self.embedding_conv(features)

        return {'positions': positions, 'embeddings': embeddings}
