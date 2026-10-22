"""Strict checkpoint loading through the existing model library."""
from types import SimpleNamespace
import torch
from torch import nn

MODEL_NAMES = ("NeuroSTORM", "SwiFT", "TFF", "BrainLM", "Com-brainTF", "LG-GNN", "BNT", "BrainGNN")

CHECKPOINT_FILENAMES = {
    "NeuroSTORM": "neurostorm_hcpya_fmri_reid.ckpt",
    "SwiFT": "swift_hcpya_fmri_reid.ckpt",
    "TFF": "tff_hcpya_fmri_reid.ckpt",
    "BrainLM": "brainlm_hcpya_fmri_reid.ckpt",
    "Com-brainTF": "combraintf_hcpya_fmri_reid.ckpt",
    "LG-GNN": "lggnn_hcpya_fmri_reid.ckpt",
    "BNT": "bnt_hcpya_fmri_reid.ckpt",
    "BrainGNN": "braingnn_hcpya_fmri_reid.ckpt",
}


class NativeFeatures(nn.Module):
    def __init__(self, hp):
        super().__init__()
        from models.load_model import load_model
        params = SimpleNamespace(**hp)
        self.model = load_model(params.model, params)
        self.output_head = load_model("clf_mlp", params)

    def forward(self, x):
        feature = self.model(x)
        if isinstance(feature, tuple):
            feature = feature[0]
        return self.output_head.forward_with_features(feature)[1]


def load_checkpoint(path, device):
    packet = torch.load(path, map_location="cpu", weights_only=False)
    hp, state = packet["hyper_parameters"], packet["state_dict"]
    name = hp["model_name"]
    if name in ("NeuroSTORM", "SwiFT"):
        # Keep the learned temporal capacity when the test window is shorter.
        capacities = {value.shape[-1] for key, value in state.items()
                      if key.startswith("model.pos_embeds.") and key.endswith(".time_embed")}
        if len(capacities) != 1:
            raise ValueError("Checkpoint must have a consistent temporal capacity")
        capacity = capacities.pop()
        if not 0 < hp["sequence_length"] <= capacity:
            raise ValueError("Test window exceeds the checkpoint temporal capacity")
        model_hp = dict(hp, sequence_length=capacity,
                        img_size=[*hp["img_size"][:3], capacity])
        network = NativeFeatures(model_hp)
        network.load_state_dict(state, strict=True)
    else:
        from models.reid_encoders import build_encoder
        network = build_encoder(name, hp, state)
    return network.to(device).requires_grad_(False).eval(), hp
