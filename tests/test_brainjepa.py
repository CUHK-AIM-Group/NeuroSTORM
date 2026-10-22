from pathlib import Path

import numpy as np
import torch

from models.brainjepa.brainjepa_adapter import ROOT, official_namespace


def test_brainjepa_attention_matches_author_calculation():
    original = official_namespace(use_sdpa=False)["Attention"](
        dim=32, num_heads=4, qkv_bias=True, qk_scale=0.25,
    ).eval()
    adapted = official_namespace()["Attention"](
        dim=32, num_heads=4, qkv_bias=True, qk_scale=0.25,
    ).eval()
    adapted.load_state_dict(original.state_dict(), strict=True)
    inputs = torch.randn(2, 11, 32)
    with torch.no_grad():
        expected, _ = original(inputs)
        actual, _ = adapted(inputs)
        with_attention, weights = adapted(inputs, return_attn=True)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(with_attention, expected, rtol=1e-5, atol=1e-6)
    assert weights.shape == (2, 4, 11, 11)


def test_brainjepa_packaged_gradient_and_encoder_forward():
    gradient_path = Path(ROOT) / "source/Brain-JEPA/data/gradient_mapping_450.csv"
    gradient = np.loadtxt(gradient_path, delimiter=",").astype(np.float32)
    assert gradient.shape == (450, 30)
    assert np.isfinite(gradient).all()
    model = official_namespace()["VisionTransformer"](
        img_size=(4, 160), patch_size=16, in_chans=1, embed_dim=32,
        depth=2, num_heads=4, mlp_ratio=2,
        gradient_pos_embed=torch.from_numpy(gradient[:4]).unsqueeze(0),
        attn_mode="normal", add_w="mapping", gradient_checkpointing=False,
    ).eval()
    inputs = torch.randn(2, 1, 4, 160, requires_grad=True)
    output = model(inputs)
    assert output.shape == (2, 40, 32)
    assert torch.isfinite(output).all()
    output[..., 0].sum().backward()
    assert inputs.grad is not None
    assert torch.isfinite(inputs.grad).all()
    assert torch.count_nonzero(inputs.grad) > 0
    masks = torch.arange(0, 40, 2).expand(2, -1)
    with torch.no_grad():
        masked = model(inputs.detach(), masks=masks)
    assert masked.shape == (2, 20, 32)
    assert torch.isfinite(masked).all()
