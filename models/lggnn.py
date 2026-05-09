"""
LG-GNN: Learnable Graph Convolutional Network with Graph Pooling

A GNN-based model for brain network analysis with learnable graph structure
and hierarchical pooling.

Based on the paper: "Learnable Graph Convolutional Network and Feature Fusion for Multi-view Learning"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool
from torch_geometric.utils import dense_to_sparse, add_self_loops


class GraphLearningLayer(nn.Module):
    """
    Learn graph structure from node features.
    Computes similarity-based adjacency matrix.
    """

    def __init__(self, input_dim, hidden_dim=64, k=10, metric='cosine'):
        """
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden dimension for feature transformation
            k: Number of nearest neighbors to keep
            metric: Similarity metric ('cosine', 'euclidean', 'attention')
        """
        super().__init__()
        self.k = k
        self.metric = metric

        if metric == 'attention':
            self.query = nn.Linear(input_dim, hidden_dim)
            self.key = nn.Linear(input_dim, hidden_dim)
            self.scale = hidden_dim ** -0.5

    def forward(self, x):
        """
        Args:
            x: Node features (batch_size, num_nodes, input_dim)

        Returns:
            edge_index: Graph connectivity (2, num_edges)
            edge_weight: Edge weights (num_edges,)
        """
        batch_size, num_nodes, _ = x.shape

        if self.metric == 'cosine':
            # Cosine similarity
            x_norm = F.normalize(x, p=2, dim=-1)
            similarity = torch.bmm(x_norm, x_norm.transpose(1, 2))

        elif self.metric == 'euclidean':
            # Negative Euclidean distance
            x_expanded = x.unsqueeze(2)  # (B, N, 1, D)
            x_tiled = x.unsqueeze(1)     # (B, 1, N, D)
            distances = torch.sum((x_expanded - x_tiled) ** 2, dim=-1)
            similarity = -distances

        elif self.metric == 'attention':
            # Attention-based similarity
            q = self.query(x)  # (B, N, H)
            k = self.key(x)    # (B, N, H)
            similarity = torch.bmm(q, k.transpose(1, 2)) * self.scale

        else:
            raise ValueError(f"Unknown metric: {self.metric}")

        # Keep top-k neighbors for each node
        if self.k < num_nodes:
            # Get top-k values and indices
            topk_values, topk_indices = torch.topk(similarity, self.k, dim=-1)

            # Create sparse adjacency matrix
            edge_list = []
            edge_weights = []

            for b in range(batch_size):
                for i in range(num_nodes):
                    for j_idx, j in enumerate(topk_indices[b, i]):
                        edge_list.append([b * num_nodes + i, b * num_nodes + j.item()])
                        edge_weights.append(topk_values[b, i, j_idx].item())

            edge_index = torch.tensor(edge_list, dtype=torch.long, device=x.device).t()
            edge_weight = torch.tensor(edge_weights, dtype=torch.float, device=x.device)

        else:
            # Use full graph
            edge_index, edge_weight = dense_to_sparse(similarity.view(batch_size * num_nodes, num_nodes))

        # Apply softmax to normalize edge weights
        edge_weight = F.softmax(edge_weight, dim=0)

        return edge_index, edge_weight


class LGGNNLayer(nn.Module):
    """
    Single LG-GNN layer with graph learning and GCN.
    """

    def __init__(self, input_dim, output_dim, hidden_dim=64, k=10,
                 learn_graph=True, dropout=0.5):
        super().__init__()

        self.learn_graph = learn_graph

        if learn_graph:
            self.graph_learner = GraphLearningLayer(input_dim, hidden_dim, k)

        self.gcn = GCNConv(input_dim, output_dim)
        self.bn = nn.BatchNorm1d(output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index=None, edge_weight=None, batch=None):
        """
        Args:
            x: Node features (num_nodes, input_dim) or (batch_size, num_nodes, input_dim)
            edge_index: Graph connectivity (2, num_edges) [optional if learn_graph=True]
            edge_weight: Edge weights (num_edges,) [optional]
            batch: Batch assignment (num_nodes,)

        Returns:
            x: Updated node features
            edge_index: Graph connectivity
            edge_weight: Edge weights
        """
        # Learn graph structure if enabled
        if self.learn_graph:
            if x.dim() == 3:
                # Batched input
                batch_size, num_nodes, feat_dim = x.shape
                x_flat = x.view(-1, feat_dim)
                edge_index, edge_weight = self.graph_learner(x)
            else:
                # Already flattened
                raise NotImplementedError("Graph learning requires batched input")
        else:
            x_flat = x if x.dim() == 2 else x.view(-1, x.size(-1))

        # Add self-loops
        edge_index, edge_weight = add_self_loops(edge_index, edge_weight, num_nodes=x_flat.size(0))

        # GCN convolution
        x_out = self.gcn(x_flat, edge_index, edge_weight)
        x_out = self.bn(x_out)
        x_out = F.relu(x_out)
        x_out = self.dropout(x_out)

        return x_out, edge_index, edge_weight


class LGGNN(nn.Module):
    """
    LG-GNN: Learnable Graph GNN for brain network analysis.

    Features:
    - Learns graph structure from node features
    - Multiple GCN layers
    - Global pooling
    - Classification/regression head
    """

    def __init__(
        self,
        num_rois=200,
        node_feature_dim=200,
        hidden_dims=[128, 64],
        num_classes=2,
        k_neighbors=10,
        learn_graph=True,
        graph_metric='cosine',
        dropout=0.5,
        **kwargs
    ):
        """
        Args:
            num_rois: Number of brain regions
            node_feature_dim: Input feature dimension per node
            hidden_dims: List of hidden dimensions for GCN layers
            num_classes: Number of output classes
            k_neighbors: Number of neighbors in learned graph
            learn_graph: Whether to learn graph structure
            graph_metric: Similarity metric for graph learning
            dropout: Dropout rate
        """
        super().__init__()

        self.num_rois = num_rois
        self.learn_graph = learn_graph

        # Build GCN layers
        self.layers = nn.ModuleList()
        dims = [node_feature_dim] + hidden_dims

        for i in range(len(hidden_dims)):
            self.layers.append(
                LGGNNLayer(
                    input_dim=dims[i],
                    output_dim=dims[i + 1],
                    hidden_dim=64,
                    k=k_neighbors,
                    learn_graph=(learn_graph and i == 0),  # Only learn graph in first layer
                    dropout=dropout
                )
            )

        # Classification head
        self.fc = nn.Sequential(
            nn.Linear(hidden_dims[-1] * 2, 128),  # *2 for mean+max pooling
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x, edge_index=None, edge_weight=None, batch=None):
        """
        Args:
            x: Node features (batch_size, num_rois, node_feature_dim) or (num_nodes, node_feature_dim)
            edge_index: Initial graph connectivity (optional)
            edge_weight: Initial edge weights (optional)
            batch: Batch assignment for PyG format (optional)

        Returns:
            output: Predictions (batch_size, num_classes)
        """
        # Handle different input formats
        if x.dim() == 3:
            # Batched input: (batch_size, num_rois, feat_dim)
            batch_size = x.size(0)
            if batch is None:
                batch = torch.arange(batch_size, device=x.device).repeat_interleave(self.num_rois)
        else:
            # PyG format: (num_nodes, feat_dim)
            if batch is None:
                raise ValueError("batch must be provided for flattened input")
            batch_size = batch.max().item() + 1

        # Apply GCN layers
        for layer in self.layers:
            x, edge_index, edge_weight = layer(x, edge_index, edge_weight, batch)

        # Global pooling
        if x.dim() == 2:
            # PyG format
            x_mean = global_mean_pool(x, batch)
            x_max = global_max_pool(x, batch)
        else:
            # Batched format
            x_mean = x.mean(dim=1)
            x_max = x.max(dim=1)[0]

        # Concatenate pooled features
        x_pooled = torch.cat([x_mean, x_max], dim=-1)

        # Classification
        output = self.fc(x_pooled)

        return output


class LGGNNRegression(LGGNN):
    """
    LG-GNN variant for regression tasks.
    """

    def __init__(self, **kwargs):
        kwargs['num_classes'] = 1
        super().__init__(**kwargs)


def create_lggnn(task_type='classification', **kwargs):
    """
    Factory function to create LG-GNN model.

    Args:
        task_type: 'classification' or 'regression'
        **kwargs: Model hyperparameters

    Returns:
        LG-GNN model instance
    """
    if task_type == 'regression':
        return LGGNNRegression(**kwargs)
    else:
        return LGGNN(**kwargs)
