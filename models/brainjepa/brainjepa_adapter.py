"""Author ViT-Base target encoder with the author's mean-pool regression head.

Optional flash-attn import is omitted. PyTorch SDPA computes the same attention
with FP32 support for the required final tests; its numerical check is separate.
"""
import ast
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import torch.utils.checkpoint

ROOT = Path(__file__).resolve().parent


def sdpa_forward(self, x, return_attn=None):
    if return_attn:
        return self.original_forward_for_audit(x, return_attn)
    batch, tokens, channels = x.shape
    qkv = self.qkv(x).reshape(batch, tokens, 3, self.num_heads, channels // self.num_heads).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    out = F.scaled_dot_product_attention(q, k, v, scale=self.scale,
                                         dropout_p=self.attn_drop.p if self.training else 0.)
    out = out.transpose(1, 2).reshape(batch, tokens, channels)
    return self.proj_drop(self.proj(out)), None


def official_namespace(use_sdpa=True):
    root = ROOT / "source/Brain-JEPA"
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    path = root / "src/models/vision_transformer.py"
    tree = ast.parse(path.read_text())
    removed = [n for n in tree.body if isinstance(n, ast.ImportFrom) and n.module == "flash_attn"]
    assert len(removed) == 1
    tree.body = [n for n in tree.body if n not in removed]
    namespace = {"__name__": "brainjepa_official_classes", "__file__": str(path)}
    exec(compile(tree, str(path), "exec"), namespace)
    if use_sdpa:
        attention = namespace["Attention"]
        attention.original_forward_for_audit = attention.forward
        attention.forward = sdpa_forward
    return namespace


class BrainJEPA(nn.Module):
    """Brain-JEPA with a configurable supervised task head."""

    def __init__(self, checkpoint_path=None, gradient_checkpointing=True,
                 num_outputs=1, gradient_path=None, expected_pretrain_epoch=None):
        super().__init__()
        if num_outputs < 1:
            raise ValueError("num_outputs must be positive")
        namespace = official_namespace()
        gradient_path = gradient_path or ROOT / "source/Brain-JEPA/data/gradient_mapping_450.csv"
        gradient = np.loadtxt(gradient_path, delimiter=",").astype(np.float32)
        if gradient.shape != (450, 30) or not np.isfinite(gradient).all():
            raise ValueError("Brain-JEPA requires a finite [450, 30] gradient table")
        gradient = torch.as_tensor(gradient).unsqueeze(0)
        self.encoder = namespace["vit_base"](img_size=(450, 160), patch_size=16, in_chans=1,
                     gradient_pos_embed=gradient, attn_mode="normal", add_w="mapping",
                     gradient_checkpointing=gradient_checkpointing)
        # The author builds this plain attribute on the selected device. Make
        # it a nonpersistent buffer so later .to(device) preserves that behavior.
        del self.encoder.gradient_pos_embed
        self.encoder.register_buffer("gradient_pos_embed", gradient, persistent=False)
        self.fc_norm = nn.LayerNorm(self.encoder.embed_dim, eps=1e-6)
        self.head = nn.Linear(self.encoder.embed_dim, num_outputs)
        if checkpoint_path is not None:
            self.load_pretrained_encoder(checkpoint_path, expected_pretrain_epoch)

    def load_pretrained_encoder(self, checkpoint_path, expected_epoch=None):
        state = torch.load(checkpoint_path, map_location="cpu", mmap=True, weights_only=True)
        if "target_encoder" not in state:
            raise ValueError("Expected a Brain-JEPA pretrained checkpoint with target_encoder")
        if expected_epoch is not None and state.get("epoch") != expected_epoch:
            raise ValueError(f"Expected pretrained epoch {expected_epoch}")
        target = {name.removeprefix("module."): value for name, value in state["target_encoder"].items()}
        self.encoder.load_state_dict(target, strict=True)

    def forward_features(self, x):
        if x.ndim == 3:
            x = x.unsqueeze(1)
        if x.ndim != 4 or tuple(x.shape[1:]) != (1, 450, 160):
            raise ValueError("Brain-JEPA expects [B,450,160] or [B,1,450,160]")
        encoded = self.encoder(x)
        return self.fc_norm(encoded.mean(dim=1))

    def forward(self, x):
        return self.head(self.forward_features(x))

    def forward_with_features(self, x):
        features = self.forward_features(x)
        return self.head(features), features


class BrainJEPAP01(BrainJEPA):
    """Compatibility interface for the original epoch-299 regression adapter."""

    def __init__(self, checkpoint_path, gradient_checkpointing=True):
        super().__init__(checkpoint_path=checkpoint_path,
                         gradient_checkpointing=gradient_checkpointing,
                         expected_pretrain_epoch=299)

    def forward(self, x):
        return super().forward(x).squeeze(-1)
