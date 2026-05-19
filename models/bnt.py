"""
Brain Network Transformer (BNT)

Based on: "Brain Network Transformer"
https://github.com/Wayfear/BrainNetworkTransformer

This implementation adapts BNT for the NeuroSTORM framework.
BNT uses Transformer encoders with learnable graph pooling for brain network analysis.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoderLayer
from typing import Optional, List


class InterpretableTransformerEncoder(TransformerEncoderLayer):
    """
    Transformer encoder layer that exposes attention weights for interpretability.
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation=F.relu, layer_norm_eps=1e-5, batch_first=False,
                 norm_first=False, device=None, dtype=None):
        super().__init__(d_model, nhead, dim_feedforward, dropout, activation,
                         layer_norm_eps, batch_first, norm_first, device, dtype)
        self.attention_weights: Optional[torch.Tensor] = None

    def _sa_block(self, x, attn_mask, key_padding_mask, is_causal=False):
        x, weights = self.self_attn(x, x, x,
                                    attn_mask=attn_mask,
                                    key_padding_mask=key_padding_mask,
                                    need_weights=True,
                                    is_causal=is_causal)
        self.attention_weights = weights
        return self.dropout1(x)

    def get_attention_weights(self):
        return self.attention_weights


class ClusterAssignment(nn.Module):
    """Cluster assignment with projection mode and orthogonal init."""

    def __init__(self, cluster_number, hidden_dimension, alpha=1.0,
                 orthogonal=True, freeze_center=False, project_assignment=True):
        super().__init__()
        self.cluster_number = cluster_number
        self.hidden_dimension = hidden_dimension
        self.alpha = alpha
        self.project_assignment = project_assignment

        centers = torch.zeros(cluster_number, hidden_dimension, dtype=torch.float)
        nn.init.xavier_uniform_(centers)
        if orthogonal:
            ortho = torch.zeros_like(centers)
            ortho[0] = centers[0]
            for i in range(1, cluster_number):
                proj = torch.zeros_like(centers[0])
                for j in range(i):
                    dot_uv = torch.dot(ortho[j], centers[i])
                    dot_uu = torch.dot(ortho[j], ortho[j]).clamp(min=1e-8)
                    proj = proj + (dot_uv / dot_uu) * ortho[j]
                ortho[i] = centers[i] - proj
                ortho[i] = ortho[i] / ortho[i].norm(p=2).clamp(min=1e-8)
            centers = ortho
        self.cluster_centers = nn.Parameter(centers, requires_grad=not freeze_center)

    def forward(self, batch):
        """
        Args:
            batch: [batch_size * num_nodes, hidden_dimension]
        Returns:
            assignment: [batch_size * num_nodes, cluster_number]
        """
        if self.project_assignment:
            assign = batch @ self.cluster_centers.T
            assign = assign.pow(2)
            norm = self.cluster_centers.norm(p=2, dim=-1)
            assign = assign / norm.clamp(min=1e-8)
            return F.softmax(assign, dim=-1)
        # Student's t-distribution fallback
        norm_squared = torch.sum((batch.unsqueeze(1) - self.cluster_centers) ** 2, 2)
        numerator = 1.0 / (1.0 + (norm_squared / self.alpha))
        power = (self.alpha + 1) / 2
        numerator = numerator ** power
        return numerator / torch.sum(numerator, dim=1, keepdim=True).clamp(min=1e-8)

    def get_cluster_centers(self):
        return self.cluster_centers


class DEC(nn.Module):
    """
    Deep Embedded Clustering (DEC) module for graph pooling.
    """

    def __init__(self, cluster_number, hidden_dimension, encoder,
                 alpha=1.0, orthogonal=True, freeze_center=True,
                 project_assignment=True):
        super().__init__()
        self.encoder = encoder
        self.hidden_dimension = hidden_dimension
        self.cluster_number = cluster_number
        self.alpha = alpha
        self.assignment = ClusterAssignment(
            cluster_number, hidden_dimension, alpha,
            orthogonal=orthogonal, freeze_center=freeze_center,
            project_assignment=project_assignment
        )
        self.loss_fn = nn.KLDivLoss(reduction='batchmean')

    def forward(self, batch):
        """
        Args:
            batch: [batch_size, node_num, hidden_dimension]

        Returns:
            node_repr: [batch_size, cluster_number, hidden_dimension]
            assignment: [batch_size, node_num, cluster_number]
        """
        batch_size = batch.size(0)
        node_num = batch.size(1)

        # Flatten and encode
        flattened_batch = batch.view(batch_size, -1)
        encoded = self.encoder(flattened_batch)
        encoded = encoded.view(batch_size * node_num, -1)

        # Compute cluster assignment
        assignment = self.assignment(encoded)
        assignment = assignment.view(batch_size, node_num, -1)
        encoded = encoded.view(batch_size, node_num, -1)

        # Aggregate nodes by cluster assignment
        node_repr = torch.bmm(assignment.transpose(1, 2), encoded)

        return node_repr, assignment

    def target_distribution(self, batch):
        """Compute target distribution for KL divergence loss."""
        weight = (batch ** 2) / torch.sum(batch, 0).clamp(min=1e-8)
        return (weight.t() / torch.sum(weight, 1).clamp(min=1e-8)).t()

    def loss(self, assignment):
        """Compute KL divergence loss for cluster assignment."""
        flattened_assignment = assignment.view(-1, assignment.size(-1))
        target = self.target_distribution(flattened_assignment).detach()
        return self.loss_fn(flattened_assignment.clamp(min=1e-10).log(), target)


class TransPoolingEncoder(nn.Module):
    """
    Transformer encoder with optional graph pooling.
    """

    def __init__(self, input_feature_size, input_node_num, hidden_size,
                 output_node_num, pooling=True, orthogonal=True,
                 freeze_center=False, project_assignment=True):
        super().__init__()

        nhead = 4 if input_feature_size % 4 == 0 else (2 if input_feature_size % 2 == 0 else 1)
        self.transformer = InterpretableTransformerEncoder(
            d_model=input_feature_size,
            nhead=nhead,
            dim_feedforward=hidden_size,
            batch_first=True
        )

        self.pooling = pooling
        if pooling:
            encoder_hidden_size = 32
            self.encoder = nn.Sequential(
                nn.Linear(input_feature_size * input_node_num, encoder_hidden_size),
                nn.LeakyReLU(),
                nn.Linear(encoder_hidden_size, encoder_hidden_size),
                nn.LeakyReLU(),
                nn.Linear(encoder_hidden_size, input_feature_size * input_node_num),
            )
            self.dec = DEC(
                cluster_number=output_node_num,
                hidden_dimension=input_feature_size,
                encoder=self.encoder,
                orthogonal=orthogonal,
                freeze_center=freeze_center,
                project_assignment=project_assignment
            )

    def is_pooling_enabled(self):
        return self.pooling

    def forward(self, x):
        x = self.transformer(x)
        if self.pooling:
            x, assignment = self.dec(x)
            return x, assignment
        return x, None

    def get_attention_weights(self):
        return self.transformer.get_attention_weights()

    def loss(self, assignment):
        return self.dec.loss(assignment)


class BrainNetworkTransformer(nn.Module):
    """
    Brain Network Transformer (BNT)

    Uses stacked Transformer encoders with learnable graph pooling
    to analyze brain connectivity networks.

    Input: Correlation matrices (num_rois, num_rois)
    Output: Class predictions or regression values
    """

    def __init__(
        self,
        num_rois=200,
        node_feature_size=200,
        num_classes=2,
        pos_encoding='identity',
        pos_embed_dim=8,
        pooling_sizes=[100, 50, 25],
        do_pooling=[True, True, False],
        hidden_size=1024,
        orthogonal=True,
        freeze_center=False,
        project_assignment=True,
        dropout=0.1,
        **kwargs
    ):
        """
        Args:
            num_rois: Number of brain regions
            node_feature_size: Feature dimension per node (usually same as num_rois for correlation matrix)
            num_classes: Number of output classes
            pos_encoding: Position encoding type ('identity' or 'none')
            pos_embed_dim: Dimension of position embeddings
            pooling_sizes: List of pooled node numbers for each layer
            do_pooling: List of booleans indicating whether to pool at each layer
            hidden_size: Hidden dimension for transformer feedforward
            orthogonal: Use orthogonal regularization for cluster centers
            freeze_center: Freeze cluster centers during training
            project_assignment: Project assignment matrix
            dropout: Dropout rate
        """
        super().__init__()

        self.num_rois = num_rois
        self.pos_encoding = pos_encoding
        self.do_pooling = do_pooling

        # Position encoding
        forward_dim = node_feature_size
        if pos_encoding == 'identity':
            self.node_identity = nn.Parameter(
                torch.zeros(num_rois, pos_embed_dim),
                requires_grad=True
            )
            nn.init.kaiming_normal_(self.node_identity)
            forward_dim = node_feature_size + pos_embed_dim

        # Build transformer layers with pooling
        self.attention_list = nn.ModuleList()
        sizes = [num_rois] + pooling_sizes

        for i in range(len(pooling_sizes)):
            self.attention_list.append(
                TransPoolingEncoder(
                    input_feature_size=forward_dim,
                    input_node_num=sizes[i],
                    hidden_size=hidden_size,
                    output_node_num=sizes[i + 1],
                    pooling=do_pooling[i],
                    orthogonal=orthogonal,
                    freeze_center=freeze_center,
                    project_assignment=project_assignment
                )
            )

        # Dimension reduction
        self.dim_reduction = nn.Sequential(
            nn.Linear(forward_dim, 8),
            nn.LeakyReLU()
        )

        # Final node count depends on which pooling layers are active: a layer
        # with do_pooling=False leaves the node dimension unchanged, so
        # sizes[-1] is only reached when every do_pooling entry is True.
        final_node_num = num_rois
        for i, do_p in enumerate(do_pooling):
            if do_p:
                final_node_num = pooling_sizes[i]

        # Classification/regression head
        self.fc = nn.Sequential(
            nn.Linear(8 * final_node_num, 256),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 32),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes)
        )

    def forward(self, node_feature):
        """
        Args:
            node_feature: [batch_size, num_rois, node_feature_size]

        Returns:
            output: [batch_size, num_classes]
        """
        batch_size = node_feature.size(0)

        # Add position encoding
        if self.pos_encoding == 'identity':
            pos_emb = self.node_identity.expand(batch_size, -1, -1)
            node_feature = torch.cat([node_feature, pos_emb], dim=-1)

        # Apply transformer layers with pooling
        assignments = []
        for layer in self.attention_list:
            node_feature, assignment = layer(node_feature)
            assignments.append(assignment)

        # Dimension reduction and classification
        node_feature = self.dim_reduction(node_feature)
        node_feature = node_feature.reshape(batch_size, -1)
        output = self.fc(node_feature)

        return output, assignments

    def get_attention_weights(self):
        """Get attention weights from all transformer layers."""
        return [layer.get_attention_weights() for layer in self.attention_list]

    def compute_pooling_loss(self, assignments):
        """
        Compute KL divergence loss for cluster assignments.
        """
        pooling_layers = [layer for layer in self.attention_list if layer.is_pooling_enabled()]
        valid_assignments = [a for a in assignments if a is not None]

        if len(valid_assignments) == 0:
            return torch.tensor(0.0, device=assignments[0].device if assignments else 'cpu')

        loss = sum(layer.loss(assignment)
                   for layer, assignment in zip(pooling_layers, valid_assignments))
        return loss


class BNTRegression(BrainNetworkTransformer):
    """
    BNT variant for regression tasks.
    """

    def __init__(self, **kwargs):
        kwargs['num_classes'] = 1
        super().__init__(**kwargs)


def create_bnt(task_type='classification', **kwargs):
    """
    Factory function to create BNT model.

    Args:
        task_type: 'classification' or 'regression'
        **kwargs: Model hyperparameters

    Returns:
        BNT model instance
    """
    if task_type == 'regression':
        return BNTRegression(**kwargs)
    else:
        return BrainNetworkTransformer(**kwargs)
