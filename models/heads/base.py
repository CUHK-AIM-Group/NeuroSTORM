from __future__ import annotations

import torch
import torch.nn as nn
from typing import Optional


class BaseHead(nn.Module):
    """Base class for all head networks. Handles the common
    flatten → avgpool spatial reduction from encoder output."""

    def _pool(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, *spatial, T) → (B, C)"""
        x = x.flatten(start_dim=2).transpose(1, 2)  # B L C
        x = self.avgpool(x.transpose(1, 2))          # B C 1
        return torch.flatten(x, 1)                    # B C

    def forward_with_features(self, x: torch.Tensor) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        return self.forward(x), None


# ---------------------------------------------------------------------------
# Head registry
# ---------------------------------------------------------------------------
_HEAD_REGISTRY: dict[str, type[nn.Module]] = {}


def register_head(name: str):
    def decorator(cls):
        _HEAD_REGISTRY[name] = cls
        return cls
    return decorator


def build_head(name: str, **kwargs) -> nn.Module:
    if name not in _HEAD_REGISTRY:
        raise KeyError(f"Unknown head '{name}'. Available: {list(_HEAD_REGISTRY.keys())}")
    return _HEAD_REGISTRY[name](**kwargs)
