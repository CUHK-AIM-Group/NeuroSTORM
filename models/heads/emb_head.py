from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F

from .base import BaseHead, register_head


@register_head("emb")
class EmbHead(BaseHead):
    def __init__(
        self,
        final_embedding_size: int = 128,
        num_tokens: int = 196,
        use_normalization: bool = True,
        n_local_frames: int = 4,
        **_,
    ):
        super().__init__()
        self.use_normalization = use_normalization
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(num_tokens, final_embedding_size, bias=False)
        self.bn1 = nn.BatchNorm1d(final_embedding_size)

    def forward(self, x, type: str = "g"):
        pooled = self._pool(x)
        out = self.bn1(self.fc1(pooled))
        if self.use_normalization:
            out = F.normalize(out, p=2, dim=1)
        return out


# backward-compatible alias
emb_head = EmbHead
