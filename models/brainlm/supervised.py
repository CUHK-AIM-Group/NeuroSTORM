"""Supervised tasks on the bundled BrainLM ViT-MAE encoder."""

import torch
from torch import nn
import torch.nn.functional as F
from transformers import ViTMAEConfig

from ..encoder_initialization import initialize_encoder
from .brainlm_modeling import ViTMAEModel


class BrainLM(nn.Module):
    def __init__(self, num_outputs=1, num_rois=424, sequence_length=200,
                 image_size=432, patch_size=16, hidden_size=768,
                 num_hidden_layers=12, num_attention_heads=12,
                 intermediate_size=3072, dropout=0.1, readout="cls",
                 checkpoint_path=None):
        super().__init__()
        if min(num_outputs, num_rois, sequence_length, image_size, patch_size,
               hidden_size, num_hidden_layers, num_attention_heads, intermediate_size) < 1:
            raise ValueError("BrainLM dimensions must be positive")
        if image_size < max(num_rois, sequence_length) or image_size % patch_size:
            raise ValueError("BrainLM image_size must contain the ROI window and be divisible by patch_size")
        if hidden_size % num_attention_heads or hidden_size % 4:
            raise ValueError("BrainLM hidden_size must be divisible by num_attention_heads and by four")
        if not 0 <= dropout < 1 or readout not in ("cls", "foreground_mean"):
            raise ValueError("Invalid BrainLM dropout or readout")
        self.num_rois = num_rois
        self.sequence_length = sequence_length
        self.readout = readout
        dh, dw = image_size - num_rois, image_size - sequence_length
        self.padding = (dw // 2, dw - dw // 2, dh // 2, dh - dh // 2)
        config = ViTMAEConfig(
            image_size=image_size, patch_size=patch_size, num_channels=3,
            hidden_size=hidden_size, num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads, intermediate_size=intermediate_size,
            hidden_dropout_prob=dropout, attention_probs_dropout_prob=dropout,
            mask_ratio=0., train_mode="supervised",
        )
        self.backbone = ViTMAEModel(config)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_size, num_outputs))
        foreground = F.pad(torch.ones(1, 1, num_rois, sequence_length), self.padding)
        self.register_buffer("foreground_weights", F.avg_pool2d(foreground, patch_size).flatten(),
                             persistent=False)
        if checkpoint_path is not None:
            initialize_encoder(self.backbone, checkpoint_path,
                               ("backbone.", "vit.", "encoder.backbone.", ""))

    def forward_features(self, x):
        if x.ndim == 3:
            x = x.unsqueeze(1)
        if x.ndim != 4 or x.shape[1] != 1 or tuple(x.shape[-2:]) != (self.num_rois, self.sequence_length):
            raise ValueError(f"BrainLM expects [B,1,{self.num_rois},{self.sequence_length}] ROI windows")
        image = F.pad(x.expand(-1, 3, -1, -1), self.padding, value=-1.)
        # In supervised mode the original masking function consumes explicit
        # zero noise: all patches are retained and evaluation consumes no RNG.
        count = self.backbone.embeddings.num_patches
        noise = image.new_zeros((len(image), count))
        output = self.backbone(pixel_values=image, noise=noise)
        if self.readout == "cls":
            return output.last_hidden_state[:, 0]
        patches = torch.gather(output.last_hidden_state[:, 1:], 1,
                               output.ids_restore[:, :, None].expand_as(output.last_hidden_state[:, 1:]))
        weights = self.foreground_weights.to(patches)
        return (patches * weights[None, :, None]).sum(1) / weights.sum()

    def forward_with_features(self, x):
        features = self.forward_features(x)
        return self.classifier(features), features

    def forward(self, x):
        return self.forward_with_features(x)[0]

