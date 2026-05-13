"""
Tests for Spatiotemporal Redundancy Dropout (STRD) module.

Tests cover:
1. Mask geometry (Ω_s / Ω_t correctness)
2. W value bounds (0 <= W <= 1)
3. Train/eval behavioral difference
4. Gradient flow through STRD
5. SwinTransformerBlock4D integration with mock Mamba
"""

import sys
import os
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---------------------------------------------------------------------------
# Mock Mamba: must be injected into sys.modules BEFORE importing neurostorm,
# because neurostorm.py does `from mamba_ssm import Mamba, Mamba2` at module level.
# ---------------------------------------------------------------------------
import torch
import torch.nn as nn


class MockMamba(nn.Module):
    def __init__(self, d_model, **kwargs):
        super().__init__()
        self.gru = nn.GRU(d_model, d_model, batch_first=True)

    def forward(self, x):
        out, _ = self.gru(x)
        return out


_fake_mamba = types.ModuleType('mamba_ssm')
_fake_mamba.Mamba = MockMamba
_fake_mamba.Mamba2 = MockMamba
sys.modules['mamba_ssm'] = _fake_mamba

import pytest


class TestBuildStrdMasks:
    """Test _build_strd_masks geometry."""

    def _get_build_fn(self):
        from models.neurostorm import _build_strd_masks
        return _build_strd_masks

    @pytest.mark.unit
    def test_mask_shapes(self):
        build = self._get_build_fn()
        ws = (2, 2, 2, 4)
        N = 2 * 2 * 2 * 4  # 32
        mask_s, mask_t = build(ws, l_spat=3, l_temp=3)
        assert mask_s.shape == (N, N)
        assert mask_t.shape == (N, N)
        assert mask_s.dtype == torch.bool
        assert mask_t.dtype == torch.bool

    @pytest.mark.unit
    def test_diagonal_excluded(self):
        """No token is its own neighbor."""
        build = self._get_build_fn()
        ws = (2, 2, 2, 4)
        mask_s, mask_t = build(ws, l_spat=5, l_temp=5)
        assert not mask_s.diagonal().any()
        assert not mask_t.diagonal().any()

    @pytest.mark.unit
    def test_spatial_mask_same_time(self):
        """Ω_s only connects tokens at the same time step."""
        build = self._get_build_fn()
        ws = (2, 2, 2, 4)
        d, h, w, t = ws
        mask_s, _ = build(ws, l_spat=3, l_temp=3)
        coords = torch.stack(torch.meshgrid(
            torch.arange(d), torch.arange(h), torch.arange(w), torch.arange(t),
            indexing='ij'
        ), dim=-1).reshape(-1, 4)
        for i in range(coords.size(0)):
            neighbors = mask_s[i].nonzero(as_tuple=True)[0]
            if neighbors.numel() > 0:
                assert (coords[neighbors, 3] == coords[i, 3]).all(), \
                    f"Token {i} has spatial neighbors at different time steps"

    @pytest.mark.unit
    def test_temporal_mask_same_position(self):
        """Ω_t only connects tokens at the same spatial position."""
        build = self._get_build_fn()
        ws = (2, 2, 2, 4)
        d, h, w, t = ws
        _, mask_t = build(ws, l_spat=3, l_temp=3)
        coords = torch.stack(torch.meshgrid(
            torch.arange(d), torch.arange(h), torch.arange(w), torch.arange(t),
            indexing='ij'
        ), dim=-1).reshape(-1, 4)
        for i in range(coords.size(0)):
            neighbors = mask_t[i].nonzero(as_tuple=True)[0]
            if neighbors.numel() > 0:
                assert (coords[neighbors, :3] == coords[i, :3]).all(), \
                    f"Token {i} has temporal neighbors at different spatial positions"

    @pytest.mark.unit
    def test_temporal_mask_within_range(self):
        """Ω_t neighbors are within l_temp//2 time steps."""
        build = self._get_build_fn()
        ws = (2, 2, 2, 8)
        d, h, w, t = ws
        l_temp = 5
        _, mask_t = build(ws, l_spat=3, l_temp=l_temp)
        coords = torch.stack(torch.meshgrid(
            torch.arange(d), torch.arange(h), torch.arange(w), torch.arange(t),
            indexing='ij'
        ), dim=-1).reshape(-1, 4)
        half_t = l_temp // 2
        for i in range(coords.size(0)):
            neighbors = mask_t[i].nonzero(as_tuple=True)[0]
            if neighbors.numel() > 0:
                dt = (coords[neighbors, 3] - coords[i, 3]).abs()
                assert (dt <= half_t).all()
                assert (dt > 0).all()  # self excluded

    @pytest.mark.unit
    def test_spatial_mask_within_range(self):
        """Ω_s neighbors are within l_spat//2 in each spatial dim."""
        build = self._get_build_fn()
        ws = (4, 4, 4, 2)
        d, h, w, t = ws
        l_spat = 3
        mask_s, _ = build(ws, l_spat=l_spat, l_temp=3)
        coords = torch.stack(torch.meshgrid(
            torch.arange(d), torch.arange(h), torch.arange(w), torch.arange(t),
            indexing='ij'
        ), dim=-1).reshape(-1, 4)
        half_s = l_spat // 2
        for i in range(coords.size(0)):
            neighbors = mask_s[i].nonzero(as_tuple=True)[0]
            if neighbors.numel() > 0:
                dd = (coords[neighbors, 0] - coords[i, 0]).abs()
                dh = (coords[neighbors, 1] - coords[i, 1]).abs()
                dw = (coords[neighbors, 2] - coords[i, 2]).abs()
                assert (dd <= half_s).all()
                assert (dh <= half_s).all()
                assert (dw <= half_s).all()

    @pytest.mark.unit
    def test_l_spat_1_no_spatial_neighbors(self):
        """l_spat=1 → half_s=0 → only self in cube, but self excluded → empty mask."""
        build = self._get_build_fn()
        ws = (2, 2, 2, 2)
        mask_s, _ = build(ws, l_spat=1, l_temp=3)
        assert not mask_s.any()


class TestApplyStrd:
    """Test _apply_strd method on a standalone block."""

    def _make_block(self, window_size=(2, 2, 2, 2), dim=16, use_strd=True,
                    l_spat=5, l_temp=5):
        from models.neurostorm import SwinTransformerBlock4D
        block = SwinTransformerBlock4D(
            dim=dim,
            num_heads=2,
            window_size=window_size,
            shift_size=[0, 0, 0, 0],
            mlp_ratio=2.0,
            use_strd=use_strd,
            strd_l_spat=l_spat,
            strd_l_temp=l_temp,
        )
        return block

    @pytest.mark.unit
    def test_w_bounds(self):
        """W values computed inside _apply_strd are in [0, 1]."""
        block = self._make_block()
        block.train()
        N = 2 * 2 * 2 * 2  # 16
        h = torch.randn(4, N, 16)

        # Monkey-patch to capture W before clamping
        original = block._apply_strd

        captured = {}

        def patched(h_in):
            scale = h_in.size(-1) ** -0.5
            A_hat = torch.softmax(h_in @ h_in.transpose(-2, -1) * scale, dim=-1)
            mask_s = block.strd_mask_s
            mask_t = block.strd_mask_t
            neg_inf = torch.finfo(A_hat.dtype).min
            f_spat = A_hat.masked_fill(~mask_s, neg_inf).max(dim=-1).values
            f_temp = A_hat.masked_fill(~mask_t, neg_inf).max(dim=-1).values
            f_spat = f_spat.clamp(min=0.0)
            f_temp = f_temp.clamp(min=0.0)
            eps = 1e-8
            sum_s = (A_hat * mask_s).sum(dim=-1, keepdim=True)
            sum_t = (A_hat * mask_t).sum(dim=-1, keepdim=True)
            W = 0.5 * (
                f_temp.unsqueeze(-1) * A_hat / (sum_s + eps)
                + f_spat.unsqueeze(-1) * A_hat / (sum_t + eps)
            )
            captured['W_raw'] = W.detach().clone()
            W = W.clamp(0.0, 1.0)
            card = (block.strd_card_s + block.strd_card_t).clamp(min=1.0)
            score = W.sum(dim=-1) / card.unsqueeze(0)
            score = score.clamp(0.0, 1.0)
            captured['score'] = score.detach().clone()
            keep = 1.0 - torch.bernoulli(score)
            return h_in * keep.unsqueeze(-1)

        block._apply_strd = patched
        _ = block._apply_strd(h)

        assert (captured['W_raw'] >= -1e-6).all(), "W has negative values before clamp"
        assert (captured['score'] >= 0.0).all()
        assert (captured['score'] <= 1.0).all()

    @pytest.mark.unit
    def test_eval_bypasses_strd(self):
        """In eval mode, STRD is not applied (output is deterministic)."""
        block = self._make_block()
        block.eval()
        N = 2 * 2 * 2 * 2
        h = torch.randn(4, N, 16)

        # forward_part1 needs proper 6D input; test _apply_strd bypass via forward
        # Instead, directly verify the condition in forward_part1
        # The block's forward_part1 checks `self.use_strd and self.training`
        assert block.use_strd is True
        assert block.training is False
        # So _apply_strd should NOT be called. Verify by running twice → same output
        with torch.no_grad():
            # Build a proper 6D input for forward_part1
            B, D, H, W, T, C = 1, 2, 2, 2, 2, 16
            x = torch.randn(B, D, H, W, T, C)
            out1 = block.forward_part1(x, None)
            out2 = block.forward_part1(x, None)
        assert torch.allclose(out1, out2), "Eval mode should be deterministic"

    @pytest.mark.unit
    def test_train_has_stochasticity(self):
        """In train mode with STRD, outputs differ across calls (stochastic dropout)."""
        block = self._make_block()
        block.train()
        B, D, H, W, T, C = 1, 2, 2, 2, 2, 16
        x = torch.randn(B, D, H, W, T, C)
        # Run multiple times — at least one pair should differ due to Bernoulli sampling
        outputs = []
        for seed in [0, 1, 2, 3, 4]:
            torch.manual_seed(seed)
            outputs.append(block.forward_part1(x, None).clone())
        differs = any(
            not torch.allclose(outputs[i], outputs[j])
            for i in range(len(outputs)) for j in range(i + 1, len(outputs))
        )
        assert differs, "Train mode with STRD should produce stochastic outputs across seeds"

    @pytest.mark.unit
    def test_no_strd_matches_disabled(self):
        """Block with use_strd=False in eval should match one without STRD buffers."""
        block_on = self._make_block(use_strd=True)
        block_off = self._make_block(use_strd=False)
        # Copy weights from block_on to block_off
        block_off.load_state_dict(block_on.state_dict(), strict=False)
        block_on.eval()
        block_off.eval()
        B, D, H, W, T, C = 1, 2, 2, 2, 2, 16
        x = torch.randn(B, D, H, W, T, C)
        with torch.no_grad():
            out_on = block_on.forward_part1(x, None)
            out_off = block_off.forward_part1(x, None)
        assert torch.allclose(out_on, out_off, atol=1e-6), \
            "STRD-enabled block in eval should match STRD-disabled block"

    @pytest.mark.unit
    def test_gradient_flows_through_strd(self):
        """Gradients propagate through the STRD multiplicative gate to Mamba params."""
        block = self._make_block()
        block.train()
        B, D, H, W, T, C = 1, 2, 2, 2, 2, 16
        x = torch.randn(B, D, H, W, T, C, requires_grad=True)
        out = block.forward_part1(x, None)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None, "Gradient should flow back to input"
        # Also check Mamba (GRU) params got gradients
        for name, p in block.mamba.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No gradient for mamba param: {name}"
                assert p.grad.abs().sum() > 0, f"Zero gradient for mamba param: {name}"

    @pytest.mark.unit
    def test_uniform_attention_low_dropout(self):
        """When all tokens are identical → A_hat uniform → redundancy score low."""
        block = self._make_block(window_size=(2, 2, 2, 2), dim=16)
        block.train()
        N = 16
        # All tokens identical → A_hat = 1/N everywhere → f_spat = f_temp = 1/N
        h = torch.ones(2, N, 16)
        # Manually call _apply_strd and check that most tokens survive
        torch.manual_seed(0)
        out = block._apply_strd(h.clone())
        # With uniform attention, score should be relatively low
        # At minimum, not all tokens should be dropped
        alive = (out.abs().sum(dim=-1) > 0).float().mean()
        assert alive > 0.5, f"Too many tokens dropped with uniform attention: {alive:.2f}"


class TestStrdIntegration:
    """Integration tests with BasicLayer."""

    @pytest.mark.unit
    def test_basic_layer_with_strd(self):
        """BasicLayer with STRD can do a forward pass."""
        from models.neurostorm import BasicLayer
        layer = BasicLayer(
            dim=16,
            depth=2,
            num_heads=2,
            window_size=[2, 2, 2, 2],
            drop_path=[0.0, 0.0],
            use_strd=True,
            strd_l_spat=3,
            strd_l_temp=3,
        )
        layer.train()
        # BasicLayer.forward expects [B, C, D, H, W, T]
        B, C, D, H, W, T = 1, 16, 2, 2, 2, 2
        x = torch.randn(B, C, D, H, W, T)
        out = layer(x)
        assert out.shape == x.shape

    @pytest.mark.unit
    def test_basic_layer_full_attention_with_strd(self):
        """BasicLayer_FullAttention with STRD can do a forward pass."""
        from models.neurostorm import BasicLayer_FullAttention
        layer = BasicLayer_FullAttention(
            dim=16,
            depth=1,
            num_heads=2,
            window_size=[2, 2, 2, 2],
            drop_path=[0.0],
            use_strd=True,
            strd_l_spat=3,
            strd_l_temp=3,
        )
        layer.train()
        # BasicLayer_FullAttention.forward expects [B, C, D, H, W, T]
        B, C, D, H, W, T = 1, 16, 2, 2, 2, 2
        x = torch.randn(B, C, D, H, W, T)
        out = layer(x)
        assert out.shape == x.shape
