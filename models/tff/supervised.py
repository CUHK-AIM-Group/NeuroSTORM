"""CNN/Transformer TFF adapter for shared supervised training."""

import torch
from torch import nn
from transformers import BertConfig

from ..encoder_initialization import initialize_encoder
from .model import BaseModel, Encoder, BottleNeck_in, Transformer_Block


class TFFBackbone(BaseModel):
    def __init__(self, spatial_shape, hidden_size, num_hidden_layers,
                 num_attention_heads, intermediate_size, max_sequence_length):
        super().__init__()
        kwargs = dict(task="fine_tune", transformer_hidden_layers=num_hidden_layers,
                      intermediate_vec=hidden_size, num_attention_heads=num_attention_heads,
                      intermediate_size=intermediate_size,
                      max_position_embeddings=max_sequence_length + 1, batch_size=1, cuda=False)
        self.encoder = Encoder(**kwargs)
        self.determine_shapes(self.encoder, spatial_shape)
        kwargs["shapes"] = self.shapes
        self.into_bert = BottleNeck_in(**kwargs)
        config = BertConfig(hidden_size=hidden_size, vocab_size=1,
                            num_hidden_layers=num_hidden_layers,
                            num_attention_heads=num_attention_heads,
                            intermediate_size=intermediate_size,
                            max_position_embeddings=max_sequence_length + 1)
        self.transformer = Transformer_Block(config, **kwargs)
        # Tokens are continuous CNN features, so the vocabulary embedding is
        # unused. Freeze it to keep every trainable parameter connected in DDP.
        self.transformer.bert.embeddings.word_embeddings.weight.requires_grad_(False)

    def forward(self, x):
        batch, channels, h, w, d, frames = x.shape
        images = x.permute(0, 5, 1, 2, 3, 4).reshape(batch * frames, channels, h, w, d)
        tokens = self.into_bert(self.encoder(images)).reshape(batch, frames, -1)
        return self.transformer(tokens)["cls"]


class TFF(nn.Module):
    def __init__(self, num_outputs=1, spatial_shape=(75, 80, 81), hidden_size=2640,
                 num_hidden_layers=2, num_attention_heads=16, intermediate_size=3072,
                 max_sequence_length=29, checkpoint_path=None):
        super().__init__()
        if len(spatial_shape) != 3 or min(spatial_shape) < 16:
            raise ValueError("TFF requires three spatial dimensions of at least 16 voxels")
        if min(num_outputs, hidden_size, num_hidden_layers, num_attention_heads,
               intermediate_size, max_sequence_length) < 1 or hidden_size % num_attention_heads:
            raise ValueError("Invalid TFF dimensions or attention head count")
        self.spatial_shape = tuple(spatial_shape)
        self.max_sequence_length = max_sequence_length
        self.backbone = TFFBackbone(self.spatial_shape, hidden_size, num_hidden_layers,
                                    num_attention_heads, intermediate_size, max_sequence_length)
        self.classifier = nn.Linear(hidden_size, num_outputs)
        if checkpoint_path is not None:
            initialize_encoder(self.backbone, checkpoint_path, ("backbone.", "pretrained.", ""))

    def forward_features(self, x):
        if x.ndim != 6 or x.shape[1] not in (1, 2) or tuple(x.shape[2:5]) != self.spatial_shape:
            raise ValueError(f"TFF expects [B,1|2,{','.join(map(str, self.spatial_shape))},T] voxel clips")
        if not 1 <= x.shape[-1] <= self.max_sequence_length:
            raise ValueError(f"TFF accepts between 1 and {self.max_sequence_length} frames per clip")
        if x.shape[1] == 1:
            # The shared volume loader supplies the preprocessed global channel;
            # form the complementary per-voxel channel from this same clip.
            mean = x.mean(dim=-1, keepdim=True)
            std = x.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-4)
            x = torch.cat((x, (x - mean) / std), dim=1)
        return self.backbone(x)

    def forward_with_features(self, x):
        features = self.forward_features(x)
        return self.classifier(features), features

    def forward(self, x):
        return self.forward_with_features(x)[0]

