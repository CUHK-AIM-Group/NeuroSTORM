"""
BrainNetCNN: Convolutional Neural Networks for Brain Networks

A specialized CNN architecture designed for brain connectivity matrices.
Uses edge-to-edge, edge-to-node, and node-to-graph convolutions.

Based on the paper: "BrainNetCNN: Convolutional neural networks for brain networks;
towards predicting neurodevelopment"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class E2EConv(nn.Module):
    """
    Edge-to-Edge Convolution Layer.
    Applies convolution on the connectivity matrix to learn edge features.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        """
        Args:
            x: (batch_size, in_channels, num_rois, num_rois)

        Returns:
            x: (batch_size, out_channels, num_rois, num_rois)
        """
        x = self.conv(x)
        x = self.bn(x)
        x = F.leaky_relu(x, negative_slope=0.33)
        return x


class E2NConv(nn.Module):
    """
    Edge-to-Node Convolution Layer.
    Aggregates edge features to node features.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        """
        Args:
            x: (batch_size, in_channels, num_rois, num_rois)

        Returns:
            x: (batch_size, out_channels, num_rois, 1)
        """
        # Aggregate along one dimension (e.g., sum over columns)
        x = torch.sum(x, dim=3, keepdim=True)
        x = self.conv(x)
        x = self.bn(x)
        x = F.leaky_relu(x, negative_slope=0.33)
        return x


class N2GConv(nn.Module):
    """
    Node-to-Graph Convolution Layer.
    Aggregates node features to graph-level features.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        """
        Args:
            x: (batch_size, in_channels, num_rois, 1)

        Returns:
            x: (batch_size, out_channels, 1, 1)
        """
        # Aggregate along node dimension
        x = torch.sum(x, dim=2, keepdim=True)
        x = self.conv(x)
        x = self.bn(x)
        x = F.leaky_relu(x, negative_slope=0.33)
        return x


class BrainNetCNN(nn.Module):
    """
    BrainNetCNN: CNN for Brain Connectivity Matrices

    Architecture:
    1. E2E layers: Learn edge-level features from connectivity matrix
    2. E2N layer: Aggregate edges to nodes
    3. N2G layer: Aggregate nodes to graph
    4. FC layers: Classification/regression
    """

    def __init__(
        self,
        num_rois=200,
        num_classes=2,
        e2e_channels=[32, 64, 64],
        e2n_channels=128,
        n2g_channels=256,
        fc_channels=[128, 64],
        dropout=0.5,
        **kwargs
    ):
        """
        Args:
            num_rois: Number of brain regions
            num_classes: Number of output classes
            e2e_channels: List of channels for E2E layers
            e2n_channels: Channels for E2N layer
            n2g_channels: Channels for N2G layer
            fc_channels: List of channels for FC layers
            dropout: Dropout rate
        """
        super().__init__()

        self.num_rois = num_rois

        # Edge-to-Edge layers
        self.e2e_layers = nn.ModuleList()
        in_ch = 1  # Input is single-channel connectivity matrix
        for out_ch in e2e_channels:
            self.e2e_layers.append(E2EConv(in_ch, out_ch))
            in_ch = out_ch

        # Edge-to-Node layer
        self.e2n = E2NConv(e2e_channels[-1], e2n_channels)

        # Node-to-Graph layer
        self.n2g = N2GConv(e2n_channels, n2g_channels)

        # Fully connected layers
        fc_layers = []
        in_features = n2g_channels
        for out_features in fc_channels:
            fc_layers.extend([
                nn.Linear(in_features, out_features),
                nn.BatchNorm1d(out_features),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_features = out_features

        fc_layers.append(nn.Linear(in_features, num_classes))
        self.fc = nn.Sequential(*fc_layers)

    def forward(self, x):
        """
        Args:
            x: Connectivity matrix (batch_size, num_rois, num_rois) or (batch_size, 1, num_rois, num_rois)

        Returns:
            output: Predictions (batch_size, num_classes)
        """
        # Ensure 4D input
        if x.dim() == 3:
            x = x.unsqueeze(1)  # Add channel dimension

        # Edge-to-Edge convolutions
        for e2e in self.e2e_layers:
            x = e2e(x)

        # Edge-to-Node
        x = self.e2n(x)

        # Node-to-Graph
        x = self.n2g(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully connected layers
        output = self.fc(x)

        return output


class BrainNetCNNRegression(BrainNetCNN):
    """
    BrainNetCNN variant for regression tasks.
    """

    def __init__(self, **kwargs):
        kwargs['num_classes'] = 1
        super().__init__(**kwargs)


class BrainNetCNNDeep(nn.Module):
    """
    Deeper variant of BrainNetCNN with more layers.
    """

    def __init__(
        self,
        num_rois=200,
        num_classes=2,
        dropout=0.5,
        **kwargs
    ):
        super().__init__()

        self.num_rois = num_rois

        # More E2E layers
        self.e2e1 = E2EConv(1, 32, kernel_size=5)
        self.e2e2 = E2EConv(32, 64, kernel_size=5)
        self.e2e3 = E2EConv(64, 64, kernel_size=3)
        self.e2e4 = E2EConv(64, 128, kernel_size=3)

        # E2N layer
        self.e2n = E2NConv(128, 256)

        # N2G layer
        self.n2g = N2GConv(256, 512)

        # FC layers
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)

        x = self.e2e1(x)
        x = self.e2e2(x)
        x = self.e2e3(x)
        x = self.e2e4(x)
        x = self.e2n(x)
        x = self.n2g(x)
        x = x.view(x.size(0), -1)
        output = self.fc(x)

        return output


def create_brainnetcnn(task_type='classification', variant='standard', **kwargs):
    """
    Factory function to create BrainNetCNN model.

    Args:
        task_type: 'classification' or 'regression'
        variant: 'standard' or 'deep'
        **kwargs: Model hyperparameters

    Returns:
        BrainNetCNN model instance
    """
    if variant == 'deep':
        model_class = BrainNetCNNDeep
    else:
        model_class = BrainNetCNN if task_type == 'classification' else BrainNetCNNRegression

    return model_class(**kwargs)
