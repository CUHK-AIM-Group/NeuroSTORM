"""Compute ROI time series and Functional Connectivity from fMRI data.

Supports two input formats:
  1. Raw .nii.gz files (original pipeline)
  2. Preprocessed data.pt blobs (int8 quantized, 96^3)

Pipeline: fMRI volume -> ROI extraction (atlas parcellation) -> FC matrix

Usage:
    # From data.pt blobs
    python datasets/compute_roi_fc.py \
        --input_dir /data/cwang/remote/fmri/hcpya_preprocessed/img \
        --input_format blob \
        --atlas_names cc200 aal3 \
        --fc_types correlation partial_correlation \
        --output_dir ./processed_data \
        --num_processes 8

    # From raw .nii.gz files
    python datasets/compute_roi_fc.py \
        --input_dir /data/cwang/remote/fmri/raw_nii/hcp1200 \
        --input_format nii \
        --atlas_names cc200 \
        --fc_types correlation \
        --output_dir ./processed_data \
        --num_processes 8
"""

import os
import re
import argparse
import numpy as np
import nibabel as nib
import torch
from scipy.ndimage import zoom
from multiprocessing import Pool
from functools import partial
from sklearn.covariance import GraphicalLassoCV


ATLAS_DIR = os.path.join(os.path.dirname(__file__), 'atlas')


def resize_atlas_labels(atlas_labels, target_shape):
    zoom_factors = [t / a for t, a in zip(target_shape, atlas_labels.shape)]
    return zoom(atlas_labels, zoom_factors, order=0).astype(np.int32)


def load_atlas(atlas_name, target_shape=None, voxel_size=None):
    """Load atlas from datasets/atlas/<atlas_name>/atlas.nii.gz.

    Falls back to legacy flat file layout (atlas_name.nii.gz) if the
    directory-based layout is not found.
    """
    # New layout: atlas_name/atlas.nii.gz
    dir_path = os.path.join(ATLAS_DIR, atlas_name, 'atlas.nii.gz')
    if os.path.exists(dir_path):
        atlas_path = dir_path
    else:
        # Legacy flat layout
        candidates = []
        for suffix in ['_1mm', '_2mm', '']:
            p = os.path.join(ATLAS_DIR, f'{atlas_name}{suffix}.nii.gz')
            if os.path.exists(p):
                candidates.append(p)
        if not candidates:
            raise FileNotFoundError(f'Atlas not found: {atlas_name} in {ATLAS_DIR}')
        atlas_path = candidates[0]

    atlas_data = nib.load(atlas_path).get_fdata().astype(np.int32)
    if target_shape is not None and atlas_data.shape != target_shape:
        atlas_data = resize_atlas_labels(atlas_data, target_shape)
    return atlas_data


def extract_roi_from_volume(volume_4d, atlas_labels):
    """Extract ROI time series from 4D volume.

    Args:
        volume_4d: numpy array [X, Y, Z, T] or [T, X, Y, Z]
        atlas_labels: numpy array [X, Y, Z] with integer ROI labels

    Returns:
        roi_data: numpy array [num_rois, T]
    """
    if volume_4d.shape[0] == atlas_labels.shape[0]:
        pass  # [X, Y, Z, T]
    elif volume_4d.shape[1:4] == atlas_labels.shape:
        volume_4d = volume_4d.transpose(1, 2, 3, 0)  # [T,X,Y,Z] -> [X,Y,Z,T]
    else:
        raise ValueError(f'Shape mismatch: volume {volume_4d.shape} vs atlas {atlas_labels.shape}')

    unique_labels = np.unique(atlas_labels)
    unique_labels = unique_labels[unique_labels > 0]
    num_frames = volume_4d.shape[-1]

    roi_data = np.zeros((len(unique_labels), num_frames), dtype=np.float32)
    for i, label in enumerate(unique_labels):
        mask = atlas_labels == label
        if np.any(mask):
            roi_data[i] = volume_4d[mask].mean(axis=0)

    return roi_data


def load_blob(subject_path):
    """Load data.pt blob and return float32 [T, H, W, D]."""
    blob = torch.load(os.path.join(subject_path, 'data.pt'),
                      mmap=True, weights_only=False)
    frames = blob['frames'].to(torch.float32) * float(blob['scale'])
    return frames.numpy()  # [T, H, W, D]


def load_nii(nii_path):
    """Load .nii.gz and return float32 [X, Y, Z, T]."""
    return nib.load(nii_path).get_fdata().astype(np.float32)


def compute_correlation(roi_data):
    corr = np.corrcoef(roi_data)
    return np.nan_to_num(corr, nan=0.0, posinf=1.0, neginf=-1.0)


def compute_partial_correlation(roi_data):
    X = roi_data.T  # [T, num_rois]
    try:
        model = GraphicalLassoCV(cv=3, max_iter=100, tol=1e-3, verbose=0)
        model.fit(X)
        precision = model.precision_
    except Exception:
        cov = np.cov(roi_data)
        cov += np.eye(cov.shape[0]) * 1e-6
        try:
            precision = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            precision = np.linalg.pinv(cov)

    diag = np.sqrt(np.diag(precision))
    partial_corr = -precision / np.outer(diag, diag)
    np.fill_diagonal(partial_corr, 1.0)
    return np.nan_to_num(partial_corr, nan=0.0, posinf=1.0, neginf=-1.0)


def compute_fc(roi_data, fc_type):
    if fc_type == 'correlation':
        return compute_correlation(roi_data)
    elif fc_type == 'partial_correlation':
        return compute_partial_correlation(roi_data)
    else:
        raise ValueError(f'Unknown FC type: {fc_type}')


def process_subject_blob(subject_name, input_dir, output_dir, atlases, fc_types):
    subject_path = os.path.join(input_dir, subject_name)
    blob_path = os.path.join(subject_path, 'data.pt')
    if not os.path.isfile(blob_path):
        return subject_name, 'skip_no_blob'

    try:
        volume = load_blob(subject_path)  # [T, H, W, D]
        spatial_shape = volume.shape[1:4]  # (H, W, D)
    except Exception as e:
        return subject_name, f'error_load: {e}'

    for atlas_name, atlas_data in atlases.items():
        if atlas_data.shape != spatial_shape:
            atlas_resized = resize_atlas_labels(atlas_data, spatial_shape)
        else:
            atlas_resized = atlas_data

        roi_dir = os.path.join(output_dir, 'roi', atlas_name)
        roi_file = os.path.join(roi_dir, f'{subject_name}_{atlas_name}.npy')

        if not os.path.exists(roi_file):
            os.makedirs(roi_dir, exist_ok=True)
            roi_data = extract_roi_from_volume(volume, atlas_resized)
            np.save(roi_file, roi_data)
        else:
            roi_data = np.load(roi_file)

        for fc_type in fc_types:
            fc_dir = os.path.join(output_dir, 'fc', atlas_name, fc_type)
            fc_file = os.path.join(fc_dir, f'{subject_name}_{atlas_name}_{fc_type}.npy')
            if os.path.exists(fc_file):
                continue
            os.makedirs(fc_dir, exist_ok=True)
            fc_matrix = compute_fc(roi_data, fc_type)
            np.save(fc_file, fc_matrix)

    return subject_name, 'ok'


def process_subject_nii(nii_path, output_dir, atlases, fc_types, dataset_name):
    base_name = os.path.splitext(os.path.basename(nii_path))[0]
    if base_name.endswith('.nii'):
        base_name = base_name[:-4]

    try:
        volume = load_nii(nii_path)  # [X, Y, Z, T]
        spatial_shape = volume.shape[:3]
    except Exception as e:
        return base_name, f'error_load: {e}'

    for atlas_name, atlas_data in atlases.items():
        if atlas_data.shape != spatial_shape:
            atlas_resized = resize_atlas_labels(atlas_data, spatial_shape)
        else:
            atlas_resized = atlas_data

        roi_dir = os.path.join(output_dir, 'roi', atlas_name)
        roi_file = os.path.join(roi_dir, f'{base_name}_{atlas_name}.npy')

        if not os.path.exists(roi_file):
            os.makedirs(roi_dir, exist_ok=True)
            roi_data = extract_roi_from_volume(volume, atlas_resized)
            np.save(roi_file, roi_data)
        else:
            roi_data = np.load(roi_file)

        for fc_type in fc_types:
            fc_dir = os.path.join(output_dir, 'fc', atlas_name, fc_type)
            fc_file = os.path.join(fc_dir, f'{base_name}_{atlas_name}_{fc_type}.npy')
            if os.path.exists(fc_file):
                continue
            os.makedirs(fc_dir, exist_ok=True)
            fc_matrix = compute_fc(roi_data, fc_type)
            np.save(fc_file, fc_matrix)

    return base_name, 'ok'


def _worker_blob(subject_name, input_dir, output_dir, atlases, fc_types):
    try:
        return process_subject_blob(subject_name, input_dir, output_dir, atlases, fc_types)
    except Exception as e:
        return subject_name, f'error: {e}'


def _worker_nii(nii_path, output_dir, atlases, fc_types, dataset_name):
    try:
        return process_subject_nii(nii_path, output_dir, atlases, fc_types, dataset_name)
    except Exception as e:
        return os.path.basename(nii_path), f'error: {e}'


def find_nii_files(input_dir):
    nii_files = []
    for root, dirs, files in os.walk(input_dir):
        for f in files:
            if f.endswith('.nii.gz') and 'mask' not in f:
                nii_files.append(os.path.join(root, f))
    return sorted(nii_files)


def main():
    parser = argparse.ArgumentParser(
        description='Compute ROI time series and FC matrices from fMRI data'
    )
    parser.add_argument('--input_dir', required=True,
                        help='Input directory (img/ dir for blob, or dir with .nii.gz)')
    parser.add_argument('--input_format', choices=['blob', 'nii'], required=True,
                        help='Input data format')
    parser.add_argument('--output_dir', required=True,
                        help='Output directory for ROI and FC data')
    parser.add_argument('--atlas_names', nargs='+', default=['cc200'],
                        help='Atlas names (files in datasets/atlas/)')
    parser.add_argument('--fc_types', nargs='+',
                        default=['correlation', 'partial_correlation'],
                        help='FC types to compute')
    parser.add_argument('--dataset_name', default='',
                        help='Dataset name (for nii mode file discovery)')
    parser.add_argument('--num_processes', type=int, default=4)
    args = parser.parse_args()

    # Load atlases with resolution matching
    # blob format is always 2mm (96^3 preprocessed), nii varies
    voxel_size = (2.0, 2.0, 2.0) if args.input_format == 'blob' else None
    atlases = {}
    for name in args.atlas_names:
        atlases[name] = load_atlas(name, voxel_size=voxel_size)
        print(f'Loaded atlas {name}: shape {atlases[name].shape}, '
              f'{len(np.unique(atlases[name])) - 1} ROIs')

    if args.input_format == 'blob':
        subjects = sorted([d for d in os.listdir(args.input_dir)
                           if os.path.isdir(os.path.join(args.input_dir, d))])
        print(f'Found {len(subjects)} subjects (blob format)')

        worker_fn = partial(_worker_blob,
                            input_dir=args.input_dir,
                            output_dir=args.output_dir,
                            atlases=atlases,
                            fc_types=args.fc_types)

        done, errors, skipped = 0, 0, 0
        with Pool(args.num_processes) as pool:
            for name, status in pool.imap_unordered(worker_fn, subjects, chunksize=4):
                if status == 'ok':
                    done += 1
                elif status.startswith('skip'):
                    skipped += 1
                else:
                    errors += 1
                    print(f'  FAILED: {name} — {status}')
                total = done + skipped + errors
                if total % 100 == 0:
                    print(f'  progress: {total}/{len(subjects)} '
                          f'(done={done}, skipped={skipped}, errors={errors})')

    else:  # nii
        nii_files = find_nii_files(args.input_dir)
        print(f'Found {len(nii_files)} .nii.gz files')

        worker_fn = partial(_worker_nii,
                            output_dir=args.output_dir,
                            atlases=atlases,
                            fc_types=args.fc_types,
                            dataset_name=args.dataset_name)

        done, errors, skipped = 0, 0, 0
        with Pool(args.num_processes) as pool:
            for name, status in pool.imap_unordered(worker_fn, nii_files, chunksize=4):
                if status == 'ok':
                    done += 1
                elif status.startswith('skip'):
                    skipped += 1
                else:
                    errors += 1
                    print(f'  FAILED: {name} — {status}')
                total = done + skipped + errors
                if total % 100 == 0:
                    print(f'  progress: {total}/{len(nii_files)} '
                          f'(done={done}, skipped={skipped}, errors={errors})')

    print(f'\nDone. processed={done}, skipped={skipped}, errors={errors}')
    print(f'Output: {args.output_dir}/roi/<atlas>/ and {args.output_dir}/fc/<atlas>/<fc_type>/')


if __name__ == '__main__':
    main()
