"""
Test script to verify dual-format (pt/h5) support in NeuroSTORM.

This script demonstrates:
1. How to preprocess data with --output_format pt or h5
2. How to load data with --data_format auto/pt/h5
"""

import os
import sys

print("=" * 60)
print("NeuroSTORM Dual-Format Support Test")
print("=" * 60)

# Test 1: Check preprocessing_volume.py has --output_format
print("\n[1] Checking preprocessing_volume.py...")
with open('datasets/preprocessing_volume.py', 'r') as f:
    content = f.read()
    if '--output_format' in content and "choices=['pt', 'h5']" in content:
        print("✓ preprocessing_volume.py supports --output_format pt/h5")
    else:
        print("✗ preprocessing_volume.py missing --output_format")

# Test 2: Check compute_stats_and_mask.py has --output_format
print("\n[2] Checking compute_stats_and_mask.py...")
with open('datasets/compute_stats_and_mask.py', 'r') as f:
    content = f.read()
    if '--output_format' in content and "choices=['h5', 'pt']" in content:
        print("✓ compute_stats_and_mask.py supports --output_format h5/pt")
    else:
        print("✗ compute_stats_and_mask.py missing --output_format")

# Test 3: Check fmri_datasets.py has format detection
print("\n[3] Checking fmri_datasets.py...")
with open('datasets/fmri_datasets.py', 'r') as f:
    content = f.read()
    checks = [
        ('import h5py', 'h5py import'),
        ('_detect_format', 'format detection method'),
        ('_count_frames', 'frame counting method'),
        ('_load_frame', 'frame loading method'),
        ('_h5_handles', 'h5 file handle caching'),
    ]
    for check_str, desc in checks:
        if check_str in content:
            print(f"✓ {desc} present")
        else:
            print(f"✗ {desc} missing")

# Test 4: Check data_module.py passes data_format
print("\n[4] Checking data_module.py...")
with open('utils/data_module.py', 'r') as f:
    content = f.read()
    if '--data_format' in content and '"data_format": self.hparams.data_format' in content:
        print("✓ data_module.py supports --data_format and passes it to Dataset")
    else:
        print("✗ data_module.py missing --data_format support")

print("\n" + "=" * 60)
print("Usage Examples:")
print("=" * 60)
print("\n# Preprocessing with pt format (default):")
print("python datasets/preprocessing_volume.py \\")
print("  --dataset_name hcpya \\")
print("  --load_root /path/to/nii \\")
print("  --save_root /path/to/output \\")
print("  --output_format pt")

print("\n# Preprocessing with h5 format:")
print("python datasets/preprocessing_volume.py \\")
print("  --dataset_name hcpya \\")
print("  --load_root /path/to/nii \\")
print("  --save_root /path/to/output \\")
print("  --output_format h5")

print("\n# Training with auto-detection (default):")
print("python main.py \\")
print("  --dataset_name HCP1200 \\")
print("  --image_path /path/to/data \\")
print("  --data_format auto")

print("\n# Training with explicit pt format:")
print("python main.py \\")
print("  --dataset_name HCP1200 \\")
print("  --image_path /path/to/data \\")
print("  --data_format pt")

print("\n# Training with explicit h5 format:")
print("python main.py \\")
print("  --dataset_name HCP1200 \\")
print("  --image_path /path/to/data \\")
print("  --data_format h5")

print("\n" + "=" * 60)
print("Format Comparison:")
print("=" * 60)
print("\nPT format:")
print("  - One .pt file per frame (frame_0.pt, frame_1.pt, ...)")
print("  - Faster random access")
print("  - More disk space (no compression)")
print("  - Better for small datasets")

print("\nH5 format:")
print("  - Single frames.h5 file per subject")
print("  - LZF compression (smaller disk usage)")
print("  - Slightly slower first access (file open overhead)")
print("  - Better for large datasets")

print("\n" + "=" * 60)
