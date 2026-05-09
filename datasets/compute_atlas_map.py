import argparse
import math
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import zoom
from preprocessing_volume import select_middle_96


def spatial_resampling_nearest(data, header, target_voxel_size=(2, 2, 2)):
    current_voxel_size = header.get_zooms()[:3]
    scale_factors = [current / target for current, target in zip(current_voxel_size, target_voxel_size)]
    new_dims = [int(np.round(dim * scale)) for dim, scale in zip(data.shape[:3], scale_factors)]
    resampled = zoom(data, [n / o for n, o in zip(new_dims, data.shape[:3])], order=0)
    return resampled.astype(np.int32)


def pad_to_target(data, target_size=96):
    padded = np.zeros((target_size, target_size, target_size), dtype=data.dtype)
    pad = [target_size - s for s in data.shape[:3]]
    starts = [math.ceil(p / 2) for p in pad]
    slices = tuple(slice(s, s + data.shape[i]) for i, s in enumerate(starts))
    padded[slices] = data
    return padded


def compute_patch_atlas(atlas_volume, patch_size=6):
    D, H, W = atlas_volume.shape
    pD, pH, pW = D // patch_size, H // patch_size, W // patch_size
    patch_map = np.zeros((pD, pH, pW), dtype=np.int32)

    for i in range(pD):
        for j in range(pH):
            for k in range(pW):
                block = atlas_volume[
                    i * patch_size:(i + 1) * patch_size,
                    j * patch_size:(j + 1) * patch_size,
                    k * patch_size:(k + 1) * patch_size,
                ]
                nonzero = block[block > 0]
                if len(nonzero) > 0:
                    values, counts = np.unique(nonzero, return_counts=True)
                    patch_map[i, j, k] = values[np.argmax(counts)]

    return patch_map


def main():
    parser = argparse.ArgumentParser(description='Compute patch-level atlas map for NeuroSTORM atlas masking.')
    parser.add_argument('--atlas_path', type=str, required=True, help='path to atlas NIfTI file')
    parser.add_argument('--output_path', type=str, required=True, help='output .pt file path')
    parser.add_argument('--patch_size', type=int, default=6, help='spatial patch size (default: 6)')
    parser.add_argument('--img_size', type=int, default=96, help='target spatial size (default: 96)')
    args = parser.parse_args()

    img = nib.load(args.atlas_path)
    atlas_data = img.get_fdata().astype(np.int32)
    header = img.header

    print(f"Original atlas shape: {atlas_data.shape}, voxel size: {header.get_zooms()[:3]}")

    resampled = spatial_resampling_nearest(atlas_data, header)
    print(f"After resampling to 2mm: {resampled.shape}")

    cropped = select_middle_96(resampled)
    print(f"After select_middle_96: {cropped.shape}")

    if any(s < args.img_size for s in cropped.shape[:3]):
        padded = pad_to_target(cropped, args.img_size)
        print(f"After padding to {args.img_size}: {padded.shape}")
    else:
        padded = cropped

    if any(s != args.img_size for s in padded.shape[:3]):
        tensor = torch.from_numpy(padded).float().unsqueeze(0).unsqueeze(0)
        resized = F.interpolate(tensor, size=(args.img_size,) * 3, mode='nearest').squeeze().numpy().astype(np.int32)
        print(f"After resize to {args.img_size}: {resized.shape}")
    else:
        resized = padded

    patch_map = compute_patch_atlas(resized, args.patch_size)

    unique_labels = np.unique(patch_map)
    unique_labels = unique_labels[unique_labels > 0]
    grid_size = args.img_size // args.patch_size

    print(f"Patch map shape: {patch_map.shape} (expected {grid_size}x{grid_size}x{grid_size})")
    print(f"Number of brain regions in patch map: {len(unique_labels)}")
    print(f"Background patches (label=0): {np.sum(patch_map == 0)} / {patch_map.size}")

    result = {
        'atlas_map': torch.from_numpy(patch_map).to(torch.int32),
        'atlas_name': args.atlas_path.split('/')[-1].replace('.nii.gz', '').replace('.nii', ''),
        'num_regions': len(unique_labels),
        'patch_size': [args.patch_size] * 3,
        'img_size': [args.img_size] * 3,
    }

    torch.save(result, args.output_path)
    print(f"Saved atlas patch map to {args.output_path}")


if __name__ == '__main__':
    main()
