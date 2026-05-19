"""BrainNetCNN (Kawahara et al., 2017, NeuroImage).

Specialized CNN for brain connectivity matrices with cross-shaped convolutions:
  - E2E: Conv2d(in, out, (1,N)) + Conv2d(in, out, (N,1)) capturing row+column patterns
  - E2N: Conv2d(in, out, (1,N)) reducing edges to node-level
  - N2G: Conv2d(in, out, (N,1)) reducing nodes to graph-level

Reference: models/models/brainnetcnn/net/brainnetcnn.py
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class E2EBlock(nn.Module):
    """Edge-to-Edge: cross-shaped convolution on connectivity matrix."""

    def __init__(self, in_channels: int, out_channels: int, n_roi: int, bias: bool = True):
        super().__init__()
        self.row_conv = nn.Conv2d(in_channels, out_channels, (1, n_roi), bias=False)
        self.col_conv = nn.Conv2d(in_channels, out_channels, (n_roi, 1), bias=False)
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        row_out = self.row_conv(x)   # [B, C_out, N, 1]
        col_out = self.col_conv(x)   # [B, C_out, 1, N]
        out = row_out + col_out      # broadcast -> [B, C_out, N, N]
        if self.bias is not None:
            out = out + self.bias.view(1, -1, 1, 1)
        return out


class E2NBlock(nn.Module):
    """Edge-to-Node: Conv2d(1,N) reduces each row to a scalar per channel."""

    def __init__(self, in_channels: int, out_channels: int, n_roi: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, (1, n_roi))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)  # [B, C_out, N, 1]


class N2GBlock(nn.Module):
    """Node-to-Graph: Conv2d(N,1) reduces N nodes to graph-level."""

    def __init__(self, in_channels: int, out_channels: int, n_roi: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, (n_roi, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)  # [B, C_out, 1, 1]


class BrainNetCNN(nn.Module):
    """BrainNetCNN with cross-shaped E2E convolutions.

    Args:
        num_rois: number of ROIs (nodes), determines conv kernel sizes
        num_classes: output classes (2 for binary, 1 for regression)
        e2e_channels: channels in E2E layers (default [32])
        e2n_channels: channels in E2N layer (default 64)
        n2g_channels: channels in N2G layer (default 256)
        dropout: dropout rate (default 0.5)
    """

    def __init__(self, num_rois: int = 200, num_classes: int = 2,
                 e2e_channels=None, e2n_channels: int = 64,
                 n2g_channels: int = 256, dropout: float = 0.5,
                 fc_channels=None, **kwargs):
        super().__init__()
        if e2e_channels is None:
            e2e_channels = [32]
        self.num_rois = num_rois

        self.e2e_layers = nn.ModuleList()
        in_ch = 1
        for out_ch in e2e_channels:
            self.e2e_layers.append(E2EBlock(in_ch, out_ch, num_rois))
            in_ch = out_ch
        # Second E2E with same channel count (reference uses 2 E2E layers)
        self.e2e_layers.append(E2EBlock(in_ch, in_ch, num_rois))

        self.e2n = E2NBlock(in_ch, e2n_channels, num_rois)
        self.n2g = N2GBlock(e2n_channels, n2g_channels, num_rois)

        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(n2g_channels, 128),
            nn.LeakyReLU(0.33),
            nn.Dropout(dropout),
            nn.Linear(128, 30),
            nn.LeakyReLU(0.33),
            nn.Linear(30, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(1)
        for e2e in self.e2e_layers:
            x = F.leaky_relu(e2e(x), 0.33)
        x = F.leaky_relu(self.e2n(x), 0.33)
        x = F.leaky_relu(self.n2g(x), 0.33)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class BrainNetCNNRegression(BrainNetCNN):
    def __init__(self, **kwargs):
        kwargs['num_classes'] = 1
        super().__init__(**kwargs)


class BrainNetCNNDeep(BrainNetCNN):
    def __init__(self, num_rois=200, num_classes=2, dropout=0.5, **kwargs):
        super().__init__(
            num_rois=num_rois, num_classes=num_classes,
            e2e_channels=[32, 64], e2n_channels=128,
            n2g_channels=256, dropout=dropout
        )


def create_brainnetcnn(task_type='classification', variant='standard', **kwargs):
    if variant == 'deep':
        return BrainNetCNNDeep(**kwargs)
    if task_type == 'regression':
        return BrainNetCNNRegression(**kwargs)
    return BrainNetCNN(**kwargs)
