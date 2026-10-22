"""Feature interfaces built from the model library for scan retrieval."""
import torch
from torch import nn
import torch.nn.functional as F
from .braingnn import BrainGNN
from .bnt import BrainNetworkTransformer
from .lggnn import LGGNN


class ROIKernelFeatures(nn.Module):
    def __init__(self, num_rois=190, num_communities=8, hidden_dim=32,
                 auxiliary_classifier_config=None):
        super().__init__()
        self.num_rois, self.hidden_dim = num_rois, hidden_dim
        backbone = BrainGNN(in_channels=num_rois, num_rois=num_rois,
                            num_communities=num_communities, hidden_dim1=hidden_dim)
        self.nn1 = backbone.nn1
        if auxiliary_classifier_config is not None:
            self.auxiliary_classifier = nn.Linear(**auxiliary_classifier_config)

    def forward_features(self, correlation):
        x = correlation.clone()
        x.diagonal(dim1=1, dim2=2).fill_(1.)
        pos = torch.eye(self.num_rois, device=x.device, dtype=x.dtype)
        weights = self.nn1(pos).reshape(self.num_rois, self.num_rois, self.hidden_dim)
        return torch.einsum("bni,nio->bno", x, weights).flatten(1)


class GraphFeatures(LGGNN):
    def __init__(self, readout="final_nodes", **kwargs):
        super().__init__(**kwargs)
        if readout not in ("final_nodes", "final_pool", "final_head"):
            raise ValueError(readout)
        self.readout = readout

    def forward_features(self, x):
        from torch_geometric.nn import global_mean_pool, global_max_pool
        count = len(x)
        batch = torch.arange(count, device=x.device).repeat_interleave(self.num_rois)
        edges = weights = None
        for layer in self.layers:
            x, edges, weights = layer(x, edges, weights, batch)
        if self.readout == "final_nodes":
            return x.reshape(count, self.num_rois, -1).flatten(1)
        pooled = torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch)], 1)
        return pooled if self.readout == "final_pool" else self.fc[:-1](pooled)


class MetricBNT(nn.Module):
    def __init__(self, arch, mean, matrix):
        super().__init__()
        feature_size, pos = arch["feature_size"], arch["position_dimensions"]
        self.num_rois = 190
        self.input_features = feature_size - pos
        self.encoder = BrainNetworkTransformer(
            num_rois=190, node_feature_size=self.input_features, num_classes=867,
            pos_encoding="identity" if pos else "none", pos_embed_dim=pos,
            pooling_sizes=[190, 100], do_pooling=[False, True], hidden_size=1024,
            freeze_center=True, dropout=0., encoder_hidden_size=arch["encoder_hidden"])
        for child in self.encoder.modules():
            if isinstance(child, nn.Dropout):
                child.p = 0.
            if isinstance(child, nn.MultiheadAttention):
                child.dropout = 0.
        self.register_buffer("metric_mean", mean.clone())
        self.register_buffer("metric_matrix", matrix.clone())

    def forward_features(self, x):
        x = F.pad(x, (0, self.input_features - self.num_rois))
        if self.encoder.pos_encoding == "identity":
            x = torch.cat([x, self.encoder.node_identity.expand(len(x), -1, -1)], -1)
        for layer in self.encoder.attention_list:
            x, _ = layer(x)
        projected = self.encoder.dim_reduction(x).flatten(1)
        z = self.encoder.fc[:-1](projected)
        return ((z.double() - self.metric_mean) @ self.metric_matrix).float()


def build_encoder(name, hp, state):
    if name == "BrainGNN":
        model = ROIKernelFeatures(
            **hp["feature_config"],
            auxiliary_classifier_config=hp.get("auxiliary_classifier_config"))
    elif name == "LG-GNN":
        model = GraphFeatures(**hp["architecture"])
    elif name == "BNT":
        model = MetricBNT(hp["architecture"], state["metric_mean"], state["metric_matrix"])
        # Public BNT includes parameterless Dropout modules at fc indices 2/5.
        mapping = {"encoder.fc.2.": "encoder.fc.3.", "encoder.fc.4.": "encoder.fc.6."}
        state = {next((new + key[len(old):] for old, new in mapping.items() if key.startswith(old)), key): value
                 for key, value in state.items()}
    elif name == "Com-brainTF":
        from .combraintf_reid import ProjectionMetricROI
        model = ProjectionMetricROI(hp["architecture"], hp["community_map"],
                                    state["metric_mean"], state["metric_matrix"])
    elif name == "BrainLM":
        from transformers import ViTMAEConfig
        from .brainlm.features import RecoveryMetricBrain
        model = RecoveryMetricBrain(ViTMAEConfig.from_dict(hp["encoder_config"]),
                                    state["metric_mean"], state["metric_matrix"], hp["readout"])
    elif name == "TFF":
        from .tff.features import TFFFeatures
        model = TFFFeatures()
    else:
        raise ValueError(name)
    model.load_state_dict(state, strict=True)
    return model
