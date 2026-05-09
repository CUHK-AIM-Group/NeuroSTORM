"""
IBGNN: Interpretable Brain Graph Neural Network

A GNN-based model with built-in interpretability for brain network analysis.
Provides attention-based explanations for predictions.

Based on the paper: "Interpretable Graph Neural Networks for Connectome-Based Brain Disorder Analysis"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool
from torch_geometric.utils import add_self_loops, softmax


class InterpretableGraphConv(nn.Module):
    """
    Graph convolution layer with edge attention for interpretability.
    """

    def __init__(self, in_channels, out_channels, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Transformation weights
        self.weight = nn.Parameter(torch.Tensor(in_channels, out_channels))

        # Edge attention mechanism
        self.edge_att = nn.Sequential(
            nn.Linear(in_channels * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x, edge_index, edge_weight=None):
        """
        Args:
            x: Node features (num_nodes, in_channels)
            edge_index: Graph connectivity (2, num_edges)
            edge_weight: Edge weights (num_edges,) [optional]

        Returns:
            x: Updated node features (num_nodes, out_channels)
            edge_attention: Edge attention weights (num_edges,)
        """
        # Transform node features
        x_transformed = torch.matmul(x, self.weight)

        # Compute edge attention
        row, col = edge_index
        edge_features = torch.cat([x[row], x[col]], dim=-1)
        edge_attention = self.edge_att(edge_features).squeeze(-1)

        # Apply softmax per node
        edge_attention = softmax(edge_attention, row, num_nodes=x.size(0))

        # Combine with edge weights if provided
        if edge_weight is not None:
            edge_attention = edge_attention * edge_weight

        # Message passing
        out = torch.zeros(x.size(0), self.out_channels, device=x.device)
        for i in range(edge_index.size(1)):
            src, dst = edge_index[0, i], edge_index[1, i]
            out[dst] += edge_attention[i] * x_transformed[src]

        if self.bias is not None:
            out += self.bias

        return out, edge_attention


class NodeAttentionPooling(nn.Module):
    """
    Attention-based pooling to identify important nodes.
    """

    def __init__(self, in_channels):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(in_channels, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, x, batch=None):
        """
        Args:
            x: Node features (num_nodes, in_channels)
            batch: Batch assignment (num_nodes,)

        Returns:
            pooled: Pooled features (batch_size, in_channels)
            node_attention: Node attention weights (num_nodes,)
        """
        # Compute node attention
        node_attention = self.attention(x).squeeze(-1)

        if batch is not None:
            # Apply softmax per graph
            node_attention = softmax(node_attention, batch)

            # Weighted sum pooling
            batch_size = batch.max().item() + 1
            pooled = torch.zeros(batch_size, x.size(1), device=x.device)

            for i in range(batch_size):
                mask = (batch == i)
                pooled[i] = (x[mask] * node_attention[mask].unsqueeze(-1)).sum(dim=0)
        else:
            # Single graph
            node_attention = F.softmax(node_attention, dim=0)
            pooled = (x * node_attention.unsqueeze(-1)).sum(dim=0, keepdim=True)

        return pooled, node_attention


class IBGNN(nn.Module):
    """
    Interpretable Brain Graph Neural Network (IBGNN)

    Features:
    - Edge attention for interpretable connections
    - Node attention for important brain regions
    - Multi-layer graph convolutions
    - Built-in explanation generation
    """

    def __init__(
        self,
        num_rois=200,
        node_feature_dim=200,
        hidden_dims=[128, 64],
        num_classes=2,
        dropout=0.5,
        use_edge_attr=True,
        **kwargs
    ):
        """
        Args:
            num_rois: Number of brain regions
            node_feature_dim: Input feature dimension per node
            hidden_dims: List of hidden dimensions for graph conv layers
            num_classes: Number of output classes
            dropout: Dropout rate
            use_edge_attr: Whether to use edge attributes
        """
        super().__init__()

        self.num_rois = num_rois
        self.use_edge_attr = use_edge_attr

        # Graph convolution layers with interpretability
        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()

        dims = [node_feature_dim] + hidden_dims
        for i in range(len(hidden_dims)):
            self.conv_layers.append(
                InterpretableGraphConv(dims[i], dims[i + 1])
            )
            self.bn_layers.append(nn.BatchNorm1d(dims[i + 1]))

        self.dropout = nn.Dropout(dropout)

        # Attention-based pooling
        self.node_pooling = NodeAttentionPooling(hidden_dims[-1])

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dims[-1], 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

        # Store attention weights for interpretation
        self.edge_attentions = []
        self.node_attention = None

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        """
        Args:
            x: Node features (batch_size, num_rois, node_feature_dim) or (num_nodes, node_feature_dim)
            edge_index: Graph connectivity (2, num_edges)
            edge_attr: Edge attributes (num_edges,) [optional]
            batch: Batch assignment (num_nodes,) [optional]

        Returns:
            output: Predictions (batch_size, num_classes)
        """
        # Handle batched input
        if x.dim() == 3:
            batch_size = x.size(0)
            x = x.view(-1, x.size(-1))
            if batch is None:
                batch = torch.arange(batch_size, device=x.device).repeat_interleave(self.num_rois)

        # Clear previous attention weights
        self.edge_attentions = []

        # Apply graph convolution layers
        for i, (conv, bn) in enumerate(zip(self.conv_layers, self.bn_layers)):
            x, edge_att = conv(x, edge_index, edge_attr)
            x = bn(x)
            x = F.relu(x)
            x = self.dropout(x)

            # Store edge attention for interpretation
            self.edge_attentions.append(edge_att)

        # Attention-based pooling
        x_pooled, node_att = self.node_pooling(x, batch)
        self.node_attention = node_att

        # Classification
        output = self.classifier(x_pooled)

        return output

    def get_edge_importance(self, layer_idx=-1):
        """
        Get edge importance scores from a specific layer.

        Args:
            layer_idx: Layer index (-1 for last layer)

        Returns:
            edge_importance: Edge attention weights
        """
        if len(self.edge_attentions) == 0:
            return None
        return self.edge_attentions[layer_idx]

    def get_node_importance(self):
        """
        Get node importance scores.

        Returns:
            node_importance: Node attention weights
        """
        return self.node_attention

    def get_top_edges(self, edge_index, k=10, layer_idx=-1):
        """
        Get top-k most important edges.

        Args:
            edge_index: Graph connectivity (2, num_edges)
            k: Number of top edges to return
            layer_idx: Layer index

        Returns:
            top_edges: Top-k edge indices
            top_scores: Corresponding attention scores
        """
        edge_importance = self.get_edge_importance(layer_idx)
        if edge_importance is None:
            return None, None

        top_scores, top_indices = torch.topk(edge_importance, min(k, edge_importance.size(0)))
        top_edges = edge_index[:, top_indices]

        return top_edges, top_scores

    def get_top_nodes(self, k=10):
        """
        Get top-k most important nodes.

        Args:
            k: Number of top nodes to return

        Returns:
            top_nodes: Top-k node indices
            top_scores: Corresponding attention scores
        """
        node_importance = self.get_node_importance()
        if node_importance is None:
            return None, None

        top_scores, top_nodes = torch.topk(node_importance, min(k, node_importance.size(0)))

        return top_nodes, top_scores


class IBGNNRegression(IBGNN):
    """
    IBGNN variant for regression tasks.
    """

    def __init__(self, **kwargs):
        kwargs['num_classes'] = 1
        super().__init__(**kwargs)


def create_ibgnn(task_type='classification', **kwargs):
    """
    Factory function to create IBGNN model.

    Args:
        task_type: 'classification' or 'regression'
        **kwargs: Model hyperparameters

    Returns:
        IBGNN model instance
    """
    if task_type == 'regression':
        return IBGNNRegression(**kwargs)
    else:
        return IBGNN(**kwargs)
