"""
BrainGNN: Graph Neural Network for Brain Connectivity Analysis

Based on: "BrainGNN: Interpretable Brain Graph Neural Network for fMRI Analysis"
https://github.com/LifangHe/BrainGNN_Pytorch

This implementation adapts BrainGNN for the NeuroSTORM framework.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter, Linear, Sequential, ReLU, BatchNorm1d
from torch_geometric.nn import MessagePassing, TopKPooling, global_mean_pool, global_max_pool
from torch_geometric.utils import add_self_loops, remove_self_loops, softmax


class BrainGraphConv(MessagePassing):
    """
    Graph convolution layer for brain connectivity networks.
    Uses edge attributes (connectivity strengths) to modulate message passing.
    """

    def __init__(self, in_channels, out_channels, nn_module, normalize=False, bias=True, **kwargs):
        super(BrainGraphConv, self).__init__(aggr='mean', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        self.nn = nn_module

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        if self.bias is not None:
            torch.nn.init.uniform_(self.bias, -1.0 / self.in_channels, 1.0 / self.in_channels)

    def forward(self, x, edge_index, edge_attr, pos):
        """
        Args:
            x: Node features (num_nodes, in_channels)
            edge_index: Graph connectivity (2, num_edges)
            edge_attr: Edge weights/attributes (num_edges,)
            pos: Position encoding for nodes (num_nodes, num_rois)
        """
        edge_attr = edge_attr.squeeze()

        # Add self-loops
        if torch.is_tensor(x):
            edge_index, edge_attr = add_self_loops(
                edge_index, edge_attr, fill_value=1.0, num_nodes=x.size(0)
            )

        # Compute edge-specific transformation weights
        weight = self.nn(pos).view(-1, self.in_channels, self.out_channels)

        # Apply transformation
        if torch.is_tensor(x):
            x = torch.matmul(x.unsqueeze(1), weight).squeeze(1)

        # Message passing
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j, edge_attr, edge_index_i, size_i, ptr):
        # Normalize edge weights using softmax
        edge_attr = softmax(edge_attr, edge_index_i, ptr, size_i)
        return x_j if edge_attr is None else edge_attr.view(-1, 1) * x_j

    def update(self, aggr_out):
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        if self.normalize:
            aggr_out = F.normalize(aggr_out, p=2, dim=-1)
        return aggr_out


class BrainGNN(nn.Module):
    """
    BrainGNN: Graph Neural Network for Brain Connectivity Analysis

    Architecture:
    1. Two graph convolution layers with TopK pooling
    2. Global pooling (mean + max)
    3. Fully connected layers for classification/regression
    """

    def __init__(
        self,
        in_channels=200,
        num_classes=2,
        num_rois=200,
        pooling_ratio=0.5,
        num_communities=8,
        hidden_dim1=32,
        hidden_dim2=32,
        fc_dim1=256,
        fc_dim2=512,
        dropout=0.5,
        **kwargs
    ):
        """
        Args:
            in_channels: Input feature dimension (number of ROIs for correlation matrix)
            num_classes: Number of output classes
            num_rois: Number of brain regions/ROIs
            pooling_ratio: Ratio of nodes to keep after pooling
            num_communities: Number of brain communities/modules
            hidden_dim1: Hidden dimension for first conv layer
            hidden_dim2: Hidden dimension for second conv layer
            fc_dim1: First FC layer dimension
            fc_dim2: Second FC layer dimension
            dropout: Dropout rate
        """
        super(BrainGNN, self).__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.num_rois = num_rois
        self.pooling_ratio = pooling_ratio
        self.num_communities = num_communities
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.dropout = dropout

        # First graph convolution layer
        self.nn1 = Sequential(
            Linear(num_rois, num_communities, bias=False),
            ReLU(),
            Linear(num_communities, hidden_dim1 * in_channels)
        )
        self.conv1 = BrainGraphConv(in_channels, hidden_dim1, self.nn1, normalize=False)
        self.pool1 = TopKPooling(hidden_dim1, ratio=pooling_ratio, multiplier=1, nonlinearity=torch.sigmoid)

        # Second graph convolution layer
        self.nn2 = Sequential(
            Linear(num_rois, num_communities, bias=False),
            ReLU(),
            Linear(num_communities, hidden_dim2 * hidden_dim1)
        )
        self.conv2 = BrainGraphConv(hidden_dim1, hidden_dim2, self.nn2, normalize=False)
        self.pool2 = TopKPooling(hidden_dim2, ratio=pooling_ratio, multiplier=1, nonlinearity=torch.sigmoid)

        # Fully connected layers
        self.fc1 = Linear((hidden_dim1 + hidden_dim2) * 2, fc_dim1)
        self.bn1 = BatchNorm1d(fc_dim1)
        self.fc2 = Linear(fc_dim1, fc_dim2)
        self.bn2 = BatchNorm1d(fc_dim2)
        self.fc3 = Linear(fc_dim2, num_classes)

    def forward(self, x, edge_index, edge_attr, batch, pos):
        """
        Args:
            x: Node features (num_nodes, in_channels)
            edge_index: Graph connectivity (2, num_edges)
            edge_attr: Edge weights (num_edges, 1)
            batch: Batch assignment (num_nodes,)
            pos: Position encoding (num_nodes, num_rois)

        Returns:
            out: Predictions (batch_size, num_classes)
            pool1_weight: First pooling layer weights
            pool2_weight: Second pooling layer weights
            pool1_score: First pooling scores
            pool2_score: Second pooling scores
        """
        # First conv + pool
        x = self.conv1(x, edge_index, edge_attr, pos)
        x, edge_index, edge_attr, batch, perm, score1 = self.pool1(
            x, edge_index, edge_attr, batch
        )
        pos = pos[perm]
        x1 = torch.cat([global_max_pool(x, batch), global_mean_pool(x, batch)], dim=1)

        # Augment adjacency (A^2 for higher-order connections)
        edge_attr = edge_attr.squeeze()
        edge_index, edge_attr = self.augment_adj(edge_index, edge_attr, x.size(0))

        # Second conv + pool
        x = self.conv2(x, edge_index, edge_attr, pos)
        x, edge_index, edge_attr, batch, perm, score2 = self.pool2(
            x, edge_index, edge_attr, batch
        )
        x2 = torch.cat([global_max_pool(x, batch), global_mean_pool(x, batch)], dim=1)

        # Concatenate features from both layers
        x = torch.cat([x1, x2], dim=1)

        # Fully connected layers
        x = self.bn1(F.relu(self.fc1(x)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.bn2(F.relu(self.fc2(x)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc3(x)

        # Return logits (not log_softmax) for compatibility with CrossEntropyLoss
        return x, self.pool1.select.weight, self.pool2.select.weight, torch.sigmoid(score1), torch.sigmoid(score2)

    def augment_adj(self, edge_index, edge_attr, num_nodes):
        """
        Augment adjacency matrix. For full graphs, simply add and remove
        self-loops to normalize the structure (avoids torch_sparse dependency).
        """
        edge_index, edge_attr = add_self_loops(
            edge_index, edge_attr, fill_value=1.0, num_nodes=num_nodes
        )
        edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
        return edge_index, edge_attr


class BrainGNNRegression(BrainGNN):
    """
    BrainGNN variant for regression tasks (e.g., age prediction).
    """

    def __init__(self, **kwargs):
        # Force num_classes=1 for regression
        kwargs['num_classes'] = 1
        super(BrainGNNRegression, self).__init__(**kwargs)

    def forward(self, x, edge_index, edge_attr, batch, pos):
        # Get base model output
        x, pool1_weight, pool2_weight, score1, score2 = super().forward(
            x, edge_index, edge_attr, batch, pos
        )
        # No activation for regression
        return x, pool1_weight, pool2_weight, score1, score2


def create_braingnn(task_type='classification', **kwargs):
    """
    Factory function to create BrainGNN model.

    Args:
        task_type: 'classification' or 'regression'
        **kwargs: Model hyperparameters

    Returns:
        BrainGNN model instance
    """
    if task_type == 'regression':
        return BrainGNNRegression(**kwargs)
    else:
        return BrainGNN(**kwargs)
