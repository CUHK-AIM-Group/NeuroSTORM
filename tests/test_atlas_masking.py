"""Test atlas masking logic using only torch (no numpy)."""
import torch


def test_atlas_masking():
    D, H, W, T = 16, 16, 16, 20
    C = 36
    B = 2
    mask_ratio = 0.5

    atlas_map = torch.zeros(D, H, W, dtype=torch.int32)
    atlas_map[0:4, 0:4, 0:4] = 1
    atlas_map[0:4, 4:8, 0:4] = 2
    atlas_map[4:8, 0:4, 0:4] = 3
    atlas_map[4:8, 4:8, 0:8] = 4
    atlas_map[8:12, 0:8, 0:8] = 5

    unique_regions = torch.unique(atlas_map)
    unique_regions = unique_regions[unique_regions > 0]
    num_regions = len(unique_regions)
    num_mask = max(1, int(num_regions * mask_ratio))

    print(f"Atlas: {num_regions} regions, masking {num_mask} (ratio={mask_ratio})")
    for r in unique_regions:
        print(f"  Region {r.item()}: {(atlas_map == r).sum().item()} patches")
    print(f"  Background: {(atlas_map == 0).sum().item()} patches")

    sequence = torch.randn(B, C, D, H, W, T)
    atlas_mask_token = torch.zeros(C)

    # --- Atlas masking logic (mirrors neurostorm.py) ---
    sequence = sequence.permute(0, 2, 3, 4, 5, 1)  # B D H W T C
    sequence = sequence.reshape(B, D * H * W * T, C)
    N = D * H * W * T
    overall_mask = torch.zeros(B, N, dtype=torch.bool)

    for i in range(B):
        perm = torch.randperm(num_regions)[:num_mask]
        selected = unique_regions[perm]
        spatial = torch.zeros(D, H, W, dtype=torch.bool)
        for r in selected:
            spatial |= (atlas_map == r)
        full_mask = spatial.unsqueeze(-1).expand(D, H, W, T).reshape(-1)
        overall_mask[i] = full_mask
        print(f"\n  Batch {i}: masked regions = {selected.tolist()}")
        print(f"  Masked spatial patches: {spatial.sum().item()} / {D * H * W}")
        print(f"  Masked tokens: {full_mask.sum().item()} / {N}")

    sequence_flat = sequence.reshape(B * N, C)
    mask_flat = overall_mask.reshape(B * N)
    sequence_flat[mask_flat] = atlas_mask_token
    new_sequence = sequence_flat.reshape(B, N, C)
    new_sequence = new_sequence.reshape(B, D, H, W, T, C).permute(0, 5, 1, 2, 3, 4)

    # --- Verify shapes ---
    assert new_sequence.shape == (B, C, D, H, W, T), f"Shape: {new_sequence.shape}"
    assert overall_mask.shape == (B, N), f"Mask shape: {overall_mask.shape}"

    # --- Verify masked positions are zero ---
    for b in range(B):
        mask_dhwt = overall_mask[b].reshape(D, H, W, T)
        mask_dhw = mask_dhwt[:, :, :, 0]
        for d in range(D):
            for h in range(H):
                for w in range(W):
                    if mask_dhw[d, h, w]:
                        vals = new_sequence[b, :, d, h, w, :]
                        assert torch.all(vals == 0), f"Position ({d},{h},{w}) not zero"
    print("\nMasked position verification: PASSED")

    # --- Verify time consistency (same spatial mask for all T) ---
    for b in range(B):
        mask_dhwt = overall_mask[b].reshape(D, H, W, T)
        first_t = mask_dhwt[:, :, :, 0]
        for t in range(1, T):
            assert torch.all(mask_dhwt[:, :, :, t] == first_t), f"Time step {t} differs from t=0"
    print("Time consistency: PASSED")

    # --- Loss shape test ---
    print("\n--- Loss shape test ---")
    x_patch = torch.randn(B, D, H, W, T, 216)
    pred_patch = torch.randn(B, D, H, W, T, 216)
    loss = (x_patch - pred_patch) ** 2
    loss = loss.mean(dim=-1)  # (B, D, H, W, T)
    loss = loss.reshape(B, D * H * W * T)
    assert loss.shape == overall_mask.shape, f"Loss {loss.shape} vs mask {overall_mask.shape}"
    loss_val = (loss * overall_mask.float()).sum() / overall_mask.float().sum()
    print(f"  Loss shape: {loss.shape}, Mask shape: {overall_mask.shape}")
    print(f"  Loss value: {loss_val.item():.4f}")

    # --- Edge case: mask_ratio=0 should mask at least 1 region ---
    num_mask_zero = max(1, int(num_regions * 0.0))
    assert num_mask_zero == 1, f"Expected 1, got {num_mask_zero}"
    print("\nEdge case (ratio=0): PASSED")

    # --- Edge case: mask_ratio=1 should mask all regions ---
    num_mask_all = max(1, int(num_regions * 1.0))
    assert num_mask_all == num_regions, f"Expected {num_regions}, got {num_mask_all}"
    print("Edge case (ratio=1): PASSED")

    print("\n=== All tests passed! ===")


if __name__ == '__main__':
    test_atlas_masking()
