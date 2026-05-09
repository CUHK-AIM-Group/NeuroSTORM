"""
Com-BrainTF: Community-aware Brain Transformer

A Transformer-based model that incorporates brain network community structure
for improved fMRI analysis.

Based on the paper: "Community-aware Transformer for Autism Prediction in fMRI Connectome"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, List


class CommunityDetection(nn.Module):
    """
    Learnable community detection module.
    Assigns nodes to communities using soft clustering.
    """

    def __init__(self, num_nodes, num_communities, temperature=1.0):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_communities = num_communities
        self.temperature = temperature

        # Learnable community assignment matrix
        self.assignment = nn.Parameter(torch.randn(num_nodes, num_communities))
        nn.init.xavier_uniform_(self.assignment)

    def forward(self, x=None):
        """
        Returns soft community assignment matrix.

        Returns:
            assignment: (num_nodes, num_communities) soft assignment probabilities
        """
        # Apply softmax to get soft assignments
        assignment = F.softmax(self.assignment / self.temperature, dim=-1)
        return assignment


class CommunityAwareAttention(nn.Module):
    """
    Multi-head attention with community-aware masking.
    """

    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead

        assert self.head_dim * nhead == d_model, "d_model must be divisible by nhead"

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(self, x, community_mask=None):
        """
        Args:
            x: (batch_size, num_nodes, d_model)
            community_mask: (num_nodes, num_nodes) community-based attention mask

        Returns:
            output: (batch_size, num_nodes, d_model)
            attention_weights: (batch_size, nhead, num_nodes, num_nodes)
        """
        batch_size, num_nodes, _ = x.shape

        # Linear projections
        q = self.q_linear(x).view(batch_size, num_nodes, self.nhead, self.head_dim).transpose(1, 2)
        k = self.k_linear(x).view(batch_size, num_nodes, self.nhead, self.head_dim).transpose(1, 2)
        v = self.v_linear(x).view(batch_size, num_nodes, self.nhead, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Apply community mask if provided
        if community_mask is not None:
            # Expand mask for batch and heads
            mask = community_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, N, N)
            scores = scores + mask

        # Softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, num_nodes, self.d_model)

        # Output projection
        output = self.out_linear(context)

        return output, attn_weights


class CommunityTransformerLayer(nn.Module):
    """
    Single Transformer layer with community-aware attention.
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()

        self.self_attn = CommunityAwareAttention(d_model, nhead, dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, community_mask=None):
        """
        Args:
            x: (batch_size, num_nodes, d_model)
            community_mask: (num_nodes, num_nodes) attention mask

        Returns:
            x: (batch_size, num_nodes, d_model)
        """
        # Self-attention with residual
        attn_output, _ = self.self_attn(x, community_mask)
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)

        # Feedforward with residual
        ff_output = self.linear2(self.dropout(F.relu(self.linear1(x))))
        x = x + self.dropout2(ff_output)
        x = self.norm2(x)

        return x


class ComBrainTF(nn.Module):
    """
    Community-aware Brain Transformer (Com-BrainTF)

    Features:
    - Learnable community detection
    - Community-aware attention mechanism
    - Multi-layer Transformer encoder
    - Hierarchical pooling (node → community → global)
    """

    def __init__(
        self,
        num_rois=200,
        node_feature_dim=200,
        num_communities=10,
        d_model=128,
        nhead=4,
        num_layers=3,
        dim_feedforward=512,
        num_classes=2,
        dropout=0.1,
        use_community_mask=True,
        temperature=1.0,
        **kwargs
    ):
        """
        Args:
            num_rois: Number of brain regions
            node_feature_dim: Input feature dimension per node
            num_communities: Number of brain communities to detect
            d_model: Transformer hidden dimension
            nhead: Number of attention heads
            num_layers: Number of Transformer layers
            dim_feedforward: Feedforward network dimension
            num_classes: Number of output classes
            dropout: Dropout rate
            use_community_mask: Whether to use community-aware masking
            temperature: Temperature for community assignment softmax
        """
        super().__init__()

        self.num_rois = num_rois
        self.num_communities = num_communities
        self.use_community_mask = use_community_mask

        # Input projection
        self.input_proj = nn.Linear(node_feature_dim, d_model)

        # Community detection
        self.community_detector = CommunityDetection(
            num_rois, num_communities, temperature
        )

        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            CommunityTransformerLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])

        # Community-level aggregation
        self.community_proj = nn.Linear(d_model, d_model)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model * num_communities, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def compute_community_mask(self, assignment):
        """
        Compute attention mask based on community assignments.
        Nodes in the same community have higher attention weights.

        Args:
            assignment: (num_nodes, num_communities)

        Returns:
            mask: (num_nodes, num_nodes) attention mask
        """
        # Compute community similarity matrix
        # High similarity if nodes belong to same community
        similarity = torch.matmul(assignment, assignment.t())

        # Convert to attention mask (0 for high similarity, -inf for low)
        # We use log to convert probabilities to logits
        mask = torch.log(similarity + 1e-9)

        return mask

    def forward(self, x):
        """
        Args:
            x: Node features (batch_size, num_rois, node_feature_dim)

        Returns:
            output: Predictions (batch_size, num_classes)
            assignment: Community assignments (num_rois, num_communities)
        """
        batch_size = x.size(0)

        # Project input features
        x = self.input_proj(x)  # (B, N, d_model)

        # Get community assignments
        assignment = self.community_detector()  # (N, C)

        # Compute community-aware attention mask
        if self.use_community_mask:
            community_mask = self.compute_community_mask(assignment)
        else:
            community_mask = None

        # Apply Transformer layers
        for layer in self.transformer_layers:
            x = layer(x, community_mask)

        # Aggregate nodes to communities
        # x: (B, N, d_model), assignment: (N, C)
        # community_features: (B, C, d_model)
        community_features = torch.matmul(
            assignment.t().unsqueeze(0).expand(batch_size, -1, -1),
            x
        )

        # Normalize by community size
        community_sizes = assignment.sum(dim=0, keepdim=True).t().unsqueeze(0)
        community_features = community_features / (community_sizes + 1e-9)

        # Project community features
        community_features = self.community_proj(community_features)

        # Flatten for classification
        community_features = community_features.view(batch_size, -1)

        # Classification
        output = self.classifier(community_features)

        return output, assignment


class ComBrainTFRegression(ComBrainTF):
    """
    Com-BrainTF variant for regression tasks.
    """

    def __init__(self, **kwargs):
        kwargs['num_classes'] = 1
        super().__init__(**kwargs)


def create_combraintf(task_type='classification', **kwargs):
    """
    Factory function to create Com-BrainTF model.

    Args:
        task_type: 'classification' or 'regression'
        **kwargs: Model hyperparameters

    Returns:
        Com-BrainTF model instance
    """
    if task_type == 'regression':
        return ComBrainTFRegression(**kwargs)
    else:
        return ComBrainTF(**kwargs)
