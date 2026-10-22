"""Brain Attend and Decode (BAnD) for 4-D fMRI task-state decoding.

This is a faithful implementation of the architecture described by Nguyen et
al. (2020): a shared 3-D ResNet-18 embeds each volume, a two-layer temporal
Transformer operates on the frame embeddings plus a pooling token, and a
two-layer classifier predicts the task state.
"""

import math

import torch
import torch.nn as nn


class BasicBlock3D(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv3d(
            in_channels, out_channels, kernel_size=3, stride=stride,
            padding=1, bias=False,
        )
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(
            out_channels, out_channels, kernel_size=3, stride=1,
            padding=1, bias=False,
        )
        self.bn2 = nn.BatchNorm3d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv3d(
                    in_channels, out_channels, kernel_size=1,
                    stride=stride, bias=False,
                ),
                nn.BatchNorm3d(out_channels),
            )
        else:
            self.downsample = nn.Identity()

    def forward(self, x):
        identity = self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class ResNet3D18Encoder(nn.Module):
    """Per-frame spatial encoder from Table 1 of the BAnD paper."""

    def __init__(self, in_channels=1, base_channels=64):
        super().__init__()
        self.in_channels = base_channels
        self.conv1 = nn.Conv3d(
            in_channels, base_channels, kernel_size=7, stride=2,
            padding=3, bias=False,
        )
        self.bn1 = nn.BatchNorm3d(base_channels)
        self.relu = nn.ReLU(inplace=True)

        # The published table has no extra max-pooling operation: conv1 and
        # the first blocks of stages 3--5 provide the four spatial reductions.
        self.layer1 = self._make_layer(base_channels, blocks=2, stride=1)
        self.layer2 = self._make_layer(base_channels * 2, blocks=2, stride=2)
        self.layer3 = self._make_layer(base_channels * 4, blocks=2, stride=2)
        self.layer4 = self._make_layer(base_channels * 8, blocks=2, stride=2)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.output_dim = base_channels * 8

        self._initialize_weights()

    def _make_layer(self, out_channels, blocks, stride):
        layers = [BasicBlock3D(self.in_channels, out_channels, stride=stride)]
        self.in_channels = out_channels
        layers.extend(
            BasicBlock3D(self.in_channels, out_channels)
            for _ in range(1, blocks)
        )
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv3d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm3d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.pool(x).flatten(1)


def sinusoidal_position_encoding(max_length, embedding_dim):
    """Vaswani et al. sinusoidal encoding, including the pooling token."""
    position = torch.arange(max_length, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, embedding_dim, 2, dtype=torch.float32)
        * (-math.log(10000.0) / embedding_dim)
    )
    encoding = torch.zeros(max_length, embedding_dim, dtype=torch.float32)
    encoding[:, 0::2] = torch.sin(position * div_term)
    encoding[:, 1::2] = torch.cos(position * div_term[:encoding[:, 1::2].shape[1]])
    return encoding.unsqueeze(0)


class BAnD(nn.Module):
    """Brain Attend and Decode.

    Input shape is ``[batch, channels, height, width, depth, time]`` and the
    output contains unnormalized class logits.
    """

    def __init__(
        self,
        in_channels=1,
        num_classes=7,
        base_channels=64,
        transformer_layers=2,
        num_heads=8,
        feedforward_dim=2048,
        dropout=0.2,
        frame_chunk_size=4,
        max_sequence_length=40,
    ):
        super().__init__()
        self.frame_encoder = ResNet3D18Encoder(in_channels, base_channels)
        self.embedding_dim = self.frame_encoder.output_dim
        if self.embedding_dim % num_heads != 0:
            raise ValueError(
                f"embedding_dim={self.embedding_dim} must be divisible by num_heads={num_heads}"
            )
        if frame_chunk_size < 1:
            raise ValueError("frame_chunk_size must be positive")
        self.frame_chunk_size = frame_chunk_size
        self.max_sequence_length = max_sequence_length

        self.pool_token = nn.Parameter(torch.zeros(1, 1, self.embedding_dim))
        self.register_buffer(
            "position_encoding",
            sinusoidal_position_encoding(max_sequence_length + 1, self.embedding_dim),
            persistent=True,
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embedding_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation="relu",
            batch_first=True,
            norm_first=False,
        )
        self.temporal_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=transformer_layers,
            norm=nn.LayerNorm(self.embedding_dim),
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.embedding_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )
        nn.init.normal_(self.pool_token, std=0.02)

    def _encode_frames(self, fmri):
        batch_size, channels, height, width, depth, time = fmri.shape
        frames = fmri.permute(0, 5, 1, 2, 3, 4).reshape(
            batch_size * time, channels, height, width, depth
        )
        embeddings = []
        for start in range(0, frames.shape[0], self.frame_chunk_size):
            embeddings.append(
                self.frame_encoder(frames[start:start + self.frame_chunk_size])
            )
        return torch.cat(embeddings, dim=0).reshape(batch_size, time, -1)

    def forward_features(self, fmri):
        if fmri.ndim != 6:
            raise ValueError(
                "BAnD expects [B,C,H,W,D,T], "
                f"received shape {tuple(fmri.shape)}"
            )
        time = fmri.shape[-1]
        if time > self.max_sequence_length:
            raise ValueError(
                f"sequence length {time} exceeds max_sequence_length "
                f"{self.max_sequence_length}"
            )
        frames = self._encode_frames(fmri)
        pool_token = self.pool_token.expand(fmri.shape[0], -1, -1)
        tokens = torch.cat((pool_token, frames), dim=1)
        tokens = tokens + self.position_encoding[:, :time + 1].to(tokens.dtype)
        return self.temporal_encoder(tokens)[:, 0]

    def forward(self, fmri):
        return self.classifier(self.forward_features(fmri))

    def forward_with_features(self, fmri):
        features = self.forward_features(fmri)
        return self.classifier(features), features
