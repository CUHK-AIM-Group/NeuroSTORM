"""Complete TFF encoder with the submitted checkpoint's feature interface."""
import types
import h5py
import numpy as np
import torch
from torch import nn
from .model import Encoder_Transformer_Decoder


class TFFFeatures(nn.Module):
    def __init__(self):
        super().__init__()
        self.pretrained = Encoder_Transformer_Decoder((75, 80, 81), task="transformer_reconstruction",
                                                      transformer_hidden_layers=2, batch_size=1, cuda=True)
        self.classifier = nn.Linear(2640, 867)
        def concatenate(module, x):
            token = torch.full((len(x), 1, 2640), .5, dtype=x.dtype, device=x.device)
            return torch.cat([module.cls_embedding(token), x], dim=1)
        self.pretrained.transformer.concatenate_cls = types.MethodType(concatenate, self.pretrained.transformer)

    def forward_tokens(self, tokens):
        feature = self.pretrained.transformer(tokens)["cls"]
        return self.classifier(feature), feature

    @torch.no_grad()
    def encode_scan(self, path):
        with h5py.File(path, "r") as handle:
            raw = np.asarray(handle["frames"][:20], dtype=np.float32) * np.float32(handle.attrs.get("scale", 1.))
        if raw.shape != (20, 91, 96, 91) or not np.isfinite(raw).all():
            raise ValueError("Invalid TFF input: " + str(path))
        z = (raw - raw.mean(0, keepdims=True)) / np.maximum(raw.std(0, keepdims=True), np.float32(1e-4))
        image = np.stack([raw[:, 8:-8, 8:-8, :-10], z[:, 8:-8, 8:-8, :-10]], axis=0).transpose(0, 2, 3, 4, 1).copy()
        images = torch.from_numpy(image).unsqueeze(0).to(next(self.parameters()).device)
        frames = images.permute(0, 5, 1, 2, 3, 4).reshape(20, 2, 75, 80, 81)
        return self.pretrained.into_bert(self.pretrained.encoder(frames)).reshape(20, 2640).cpu()
