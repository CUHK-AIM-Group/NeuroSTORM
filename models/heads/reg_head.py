from __future__ import annotations

import torch
import torch.nn as nn

from .base import BaseHead, register_head


@register_head("reg_v1")
class RegHeadV1(BaseHead):
    def __init__(self, num_tokens: int = 96, **_):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(num_tokens, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self._pool(x))


_VERSION_MAP: dict[str, type[nn.Module]] = {
    1: RegHeadV1,
}


class RegHead(nn.Module):
    def __init__(self, version: int = 1, num_tokens: int = 96, **kwargs):
        super().__init__()
        if version not in _VERSION_MAP:
            raise ValueError(f"Unknown reg_head version '{version}'. Choose from {list(_VERSION_MAP.keys())}")
        self.head = _VERSION_MAP[version](num_tokens=num_tokens, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


# backward-compatible alias
reg_head = RegHead
