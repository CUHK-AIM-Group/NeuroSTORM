from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .base import BaseHead, register_head


# ---------------------------------------------------------------------------
# V1: Linear head (avgpool → linear)
# ---------------------------------------------------------------------------
@register_head("cls_v1")
class ClsHeadV1(BaseHead):
    def __init__(self, num_classes: int = 2, num_tokens: int = 96, **_):
        super().__init__()
        num_outputs = 1 if num_classes == 2 else num_classes
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(num_tokens, num_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self._pool(x))

    def forward_with_features(self, x) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        feat = self._pool(x)
        return self.head(feat), feat


# ---------------------------------------------------------------------------
# V2: Two-layer MLP head (avgpool → hidden → output)
# ---------------------------------------------------------------------------
@register_head("cls_v2")
class ClsHeadV2(BaseHead):
    def __init__(self, num_classes: int = 2, num_tokens: int = 96, **_):
        super().__init__()
        num_outputs = 1 if num_classes == 2 else num_classes
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.hidden = nn.Linear(num_tokens, 4 * num_tokens)
        self.head = nn.Linear(4 * num_tokens, num_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.hidden(self._pool(x)))

    def forward_with_features(self, x) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        feat = self.hidden(self._pool(x))
        return self.head(feat), feat


# ---------------------------------------------------------------------------
# V3: Transformer encoder + CLS token head
# ---------------------------------------------------------------------------
@register_head("cls_v3")
class ClsHeadV3(nn.Module):
    def __init__(
        self,
        num_classes: int = 2,
        num_tokens: int = 160,
        emb_size: int = 288,
        num_heads: int = 8,
        num_layers: int = 6,
        forward_expansion: int = 4,
        dropout: float = 0.1,
        **_,
    ):
        super().__init__()
        num_outputs = 1 if num_classes == 2 else num_classes
        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_size))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens + 1, emb_size))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_size,
            nhead=num_heads,
            dim_feedforward=emb_size * forward_expansion,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.norm = nn.LayerNorm(emb_size)
        self.head = nn.Linear(emb_size, num_outputs)

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        x = x.flatten(start_dim=2).transpose(1, 2)  # B L C
        B, L, _ = x.shape
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed[:, : L + 1, :]
        x = self.transformer_encoder(x)
        return x[:, 0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.norm(self._encode(x)))

    def forward_with_features(self, x) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        feat = self.norm(self._encode(x))
        return self.head(feat), feat


# ---------------------------------------------------------------------------
# Version-string → class mapping (backward-compatible factory)
# ---------------------------------------------------------------------------
_VERSION_MAP: dict[str, type[nn.Module]] = {
    "v1": ClsHeadV1,
    "v2": ClsHeadV2,
    "v3": ClsHeadV3,
}


class ClsHead(nn.Module):
    def __init__(self, version: str = "v1", num_classes: int = 2, num_tokens: int = 96, **kwargs):
        super().__init__()
        if version not in _VERSION_MAP:
            raise ValueError(f"Unknown cls_head version '{version}'. Choose from {list(_VERSION_MAP.keys())}")
        self.head = _VERSION_MAP[version](num_classes=num_classes, num_tokens=num_tokens, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)

    def forward_with_features(self, x) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        return self.head.forward_with_features(x)


# backward-compatible alias
cls_head = ClsHead
