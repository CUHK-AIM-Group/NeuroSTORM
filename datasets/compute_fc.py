"""
Compute Functional Connectivity (FC) matrices from ROI time series data.

Supports:
1. Pearson correlation
2. Partial correlation
3. Other connectivity measures
"""

import os
import argparse
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial
from sklearn.covariance import GraphicalLassoCV


def compute_correlation(roi_data):
    """
    Compute Pearson correlation matrix.

    Args:
        roi_data: ROI time series (num_rois, num_timepoints)

    Returns:
        Correlation matrix (num_rois, num_rois)
    """
    # Transpose to (num_timepoints, num_rois) for np.corrcoef
    corr_matrix = np.corrcoef(roi_data)

    # Handle NaN values
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0, posinf=1.0, neginf=-1.0)

    return corr_matrix


def compute_partial_correlation(roi_data, method='graphical_lasso'):
    """
    Compute partial correlation matrix.

    Args:
        roi_data: ROI time series (num_rois, num_timepoints)
        method: Method to use ('graphical_lasso' or 'inverse_covariance')

    Returns:
        Partial correlation matrix (num_rois, num_rois)
    """
    # Transpose to (num_timepoints, num_rois)
    X = roi_data.T

    if method == 'graphical_lasso':
        try:
            # Use GraphicalLassoCV for sparse inverse covariance estimation
            model = GraphicalLassoCV(cv=3, max_iter=100, tol=1e-3, verbose=0)
            model.fit(X)
            precision_matrix = model.precision_
        except Exception as e:
            print(f"GraphicalLasso failed: {e}, falling back to inverse covariance")
            # Fallback to simple inverse covariance
            cov_matrix = np.cov(roi_data)
            cov_matrix += np.eye(cov_matrix.shape[0]) * 1e-6  # Regularization
            try:
                precision_matrix = np.linalg.inv(cov_matrix)
            except np.linalg.LinAlgError:
                print("Matrix inversion failed, using pseudo-inverse")
                precision_matrix = np.linalg.pinv(cov_matrix)
    else:
        # Simple inverse covariance
        cov_matrix = np.cov(roi_data)
        cov_matrix += np.eye(cov_matrix.shape[0]) * 1e-6  # Regularization
        try:
            precision_matrix = np.linalg.inv(cov_matrix)
        except np.linalg.LinAlgError:
            print("Matrix inversion failed, using pseudo-inverse")
            precision_matrix = np.linalg.pinv(cov_matrix)

    # Convert precision matrix to partial correlation
    # partial_corr[i,j] = -precision[i,j] / sqrt(precision[i,i] * precision[j,j])
    diag = np.sqrt(np.diag(precision_matrix))
    partial_corr = -precision_matrix / np.outer(diag, diag)
    np.fill_diagonal(partial_corr, 1.0)

    # Handle NaN values
    partial_corr = np.nan_to_num(partial_corr, nan=0.0, posinf=1.0, neginf=-1.0)

    return partial_corr


def process_single_subject(args):
    """
    Process a single subject's ROI data to compute FC matrices.

    Args:
        args: Tuple of (roi_file, output_dir, atlas_name, fc_types)

    Returns:
        None (saves FC matrices to disk)
    """
    roi_file, output_dir, atlas_name, fc_types = args

    # Extract subject ID from filename
    base_name = os.path.basename(roi_file)
    subject_id = base_name.replace(f'_{atlas_name}.npy', '')

    # Check if all FC types already exist
    all_exist = True
    for fc_type in fc_types:
        fc_dir = os.path.join(output_dir, fc_type)
        output_file = os.path.join(fc_dir, f"{subject_id}_{atlas_name}_{fc_type}.npy")
        if not os.path.exists(output_file):
            all_exist = False
            break

    if all_exist:
        return

    try:
        # Load ROI time series: (num_rois, num_timepoints)
        roi_data = np.load(roi_file)

        # Check for invalid data
        if np.any(np.isnan(roi_data)) or np.any(np.isinf(roi_data)):
            print(f"Warning: {roi_file} contains NaN or Inf values, skipping")
            return

        # Compute each FC type
        for fc_type in fc_types:
            fc_dir = os.path.join(output_dir, fc_type)
            os.makedirs(fc_dir, exist_ok=True)
            output_file = os.path.join(fc_dir, f"{subject_id}_{atlas_name}_{fc_type}.npy")

            if os.path.exists(output_file):
                continue

            if fc_type == 'correlation':
                fc_matrix = compute_correlation(roi_data)
            elif fc_type == 'partial_correlation':
                fc_matrix = compute_partial_correlation(roi_data)
            else:
                raise ValueError(f"Unknown FC type: {fc_type}")

            # Save FC matrix
            np.save(output_file, fc_matrix)

    except Exception as e:
        print(f"Error processing {roi_file}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description='Compute functional connectivity matrices from ROI time series data'
    )
    parser.add_argument('--roi_dir', type=str, required=True,
                        help='Directory containing ROI .npy files')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for FC matrices')
    parser.add_argument('--atlas_name', type=str, required=True,
                        help='Name of brain atlas (e.g., cc200, aal, schaefer)')
    parser.add_argument('--fc_types', type=str, nargs='+',
                        default=['correlation', 'partial_correlation'],
                        help='Types of FC to compute')
    parser.add_argument('--num_processes', type=int, default=1,
                        help='Number of parallel processes')

    args = parser.parse_args()

    # Find all ROI files
    roi_files = []
    for file_name in os.listdir(args.roi_dir):
        if file_name.endswith(f'_{args.atlas_name}.npy'):
            roi_files.append(os.path.join(args.roi_dir, file_name))

    roi_files.sort()

    print(f"Found {len(roi_files)} ROI files")
    print(f"Computing FC types: {args.fc_types}")
    print(f"Using {args.num_processes} processes")

    # Prepare arguments for parallel processing
    process_args = [
        (roi_file, args.output_dir, args.atlas_name, args.fc_types)
        for roi_file in roi_files
    ]

    # Process files
    if args.num_processes == 1:
        for arg in tqdm(process_args, desc='Computing FC matrices'):
            process_single_subject(arg)
    else:
        with Pool(processes=args.num_processes) as pool:
            list(tqdm(
                pool.imap(process_single_subject, process_args),
                total=len(process_args),
                desc='Computing FC matrices'
            ))

    print(f"\nFC matrices saved to {args.output_dir}")


if __name__ == '__main__':
    main()
