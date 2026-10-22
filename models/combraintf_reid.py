"""Com-brainTF variant used by the scan retrieval checkpoints."""
import torch
from torch import nn
import torch.nn.functional as F
from .third_party.combraintf.blocks import TransPoolingEncoder


class CommunityEncoder(nn.Module):
    def __init__(self, arch, community_map):
        super().__init__()
        self.feature_size = arch["feature_size"]
        self.position_dimensions = arch["position_dimensions"]
        groups = community_map["roi_networks"]
        if len(groups) != 190:
            raise ValueError("The community map must contain 190 ROI labels")
        permutation = [i for g in range(8) for i, label in enumerate(groups) if label == g]
        if sorted(permutation) != list(range(190)):
            raise ValueError("Invalid community map")
        self.register_buffer("roi_permutation", torch.tensor(permutation, dtype=torch.long))
        self.community_sizes = [groups.count(g) for g in range(8)]
        if min(self.community_sizes) <= 0:
            raise ValueError("Every community must contain an ROI")
        self.node_identity = nn.Parameter(torch.zeros(190, self.position_dimensions))

        def block(pooling, local=False):
            result = TransPoolingEncoder(input_feature_size=self.feature_size,
                input_node_num=190, hidden_size=1024, output_node_num=8,
                pooling=pooling, orthogonal=True, freeze_center=True,
                project_assignment=True, nHead=8, local_transformer=local)
            if pooling and arch["encoder_hidden"] != 32:
                width = arch["encoder_hidden"]
                mlp = nn.Sequential(nn.Linear(190 * self.feature_size, width), nn.LeakyReLU(),
                    nn.Linear(width, width), nn.LeakyReLU(), nn.Linear(width, 190 * self.feature_size))
                result.encoder = mlp
                result.dec.encoder = mlp
            for child in result.modules():
                if isinstance(child, nn.Dropout):
                    child.p = 0.
            result.transformer.self_attn.dropout = 0.
            return result
        self.local_transformer = block(False, True)
        self.attention_list = nn.ModuleList([block(True)])
        self.mlp = nn.Sequential(nn.Linear(8 * self.feature_size, 512), nn.LeakyReLU(),
                                nn.Linear(512, self.feature_size), nn.LeakyReLU())
        projection = arch.get("projection_dimensions", 8)
        self.dim_reduction = nn.Sequential(nn.Linear(self.feature_size, projection), nn.LeakyReLU())
        dimension = arch["semantic_dimension"]
        self.fc = nn.Sequential(nn.Linear(8 * projection, 256), nn.LeakyReLU(),
                                nn.Linear(256, dimension), nn.LeakyReLU(), nn.Linear(dimension, 867))

    def forward_features(self, x):
        p = self.roi_permutation
        x = x[:, p][:, :, p]
        x = F.pad(x, (0, self.feature_size - self.position_dimensions - 190))
        positions = self.node_identity[p]
        x = torch.cat([x, positions[None].expand(len(x), -1, -1)], -1)
        pieces, tokens, start = [], [], 0
        for group, size in enumerate(self.community_sizes):
            nodes, _, token = self.local_transformer(x[:, start:start + size], group)
            pieces.append(nodes)
            tokens.append(token)
            start += size
        x = torch.cat(pieces, 1)
        cls = self.mlp(torch.cat(tokens, 1).flatten(1))[:, None]
        x = torch.cat([cls, x], 1)
        for layer in self.attention_list:
            x = layer(x)[0]
        return self.fc[:4](self.dim_reduction(x).flatten(1))


class ProjectionMetricROI(nn.Module):
    def __init__(self, arch, community_map, mean, matrix):
        super().__init__()
        self.encoder = CommunityEncoder(arch, community_map)
        self.register_buffer("metric_mean", torch.as_tensor(mean, dtype=torch.float64))
        self.metric_matrix = nn.Parameter(torch.as_tensor(matrix, dtype=torch.float64).clone())

    def forward_features(self, x):
        raw = self.encoder.forward_features(x)
        return ((raw.double() - self.metric_mean) @ self.metric_matrix).float()
