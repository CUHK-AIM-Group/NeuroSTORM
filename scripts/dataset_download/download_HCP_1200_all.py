#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This script downloads data from the Human Connectome Project – 1200 subjects release.

Major features:
1. Use predefined S3 keys to download target `.nii.gz` files directly
   (no S3 listing needed, faster and cheaper).
2. Skip files that already exist in the output directory.
3. Record any missing / failed downloads into a CSV file:
      missing_downloads.csv  (single-process)
      missing_downloads_<proc_idx>.csv  (multi-process)
"""

import argparse
import os
import csv
import pickle
from multiprocessing import Process
from typing import List, Dict

import boto3
from boto3.s3.transfer import TransferConfig
from botocore.exceptions import ClientError
from tqdm import tqdm


# ------------------------------------------------------------------
# Global constants
# ------------------------------------------------------------------
S3_BUCKET_NAME = 'hcp-openaccess'
S3_PREFIX      = 'HCP_1200'                      # root folder in the bucket

# ------------------------------------------------------------------
# Helper to build the list/dict of files that should be downloaded
# (all paths are relative to  .../<SID>/MNINonLinear/  )
# ------------------------------------------------------------------
def build_series_map() -> Dict[str, str]:
    """Return {alias : relative_path_under_MNINonLinear}"""
    task_names = ['WM', 'SOCIAL', 'RELATIONAL', 'MOTOR',
                  'LANGUAGE', 'GAMBLING', 'EMOTION']

    series = {}
    for task in task_names:
        series[f'{task}LR'] = f'Results/tfMRI_{task}_LR/tfMRI_{task}_LR.nii.gz'
        series[f'{task}RL'] = f'Results/tfMRI_{task}_RL/tfMRI_{task}_RL.nii.gz'

    series.update({
        'fmri_1': 'Results/rfMRI_REST1_LR/rfMRI_REST1_LR_hp2000_clean.nii.gz',
        'fmri_2': 'Results/rfMRI_REST1_RL/rfMRI_REST1_RL_hp2000_clean.nii.gz',
        'fmri_3': 'Results/rfMRI_REST2_LR/rfMRI_REST1_LR_hp2000_clean.nii.gz',
        'fmri_4': 'Results/rfMRI_REST2_RL/rfMRI_REST1_RL_hp2000_clean.nii.gz',
        'fmri_5': 'Results/rfMRI_REST1_7T_PA/rfMRI_REST1_7T_PA_hp2000_clean.nii.gz',
        'fmri_6': 'Results/rfMRI_REST2_7T_AP/rfMRI_REST2_7T_AP_hp2000_clean.nii.gz',
        'fmri_7': 'Results/rfMRI_REST3_7T_PA/rfMRI_REST3_7T_PA_hp2000_clean.nii.gz',
        'fmri_8': 'Results/rfMRI_REST4_7T_AP/rfMRI_REST4_7T_AP_hp2000_clean.nii.gz',
        't1'    : 'T1w.nii.gz',
        't2'    : 'T2w.nii.gz',
    })
    return series


# ------------------------------------------------------------------
# Core downloading routine (single process)
# ------------------------------------------------------------------
def download_known_keys(
    subjects: List[str],
    out_dir: str,
    aws_access_key_id: str,
    aws_secret_access_key: str,
    process_idx: int = 0
):
    """
    Download predefined .nii.gz files for each subject.

    Parameters
    ----------
    subjects : List[str]
        List of subject IDs to download.
    out_dir  : str
        Local directory where files are saved.
    aws_access_key_id : str
    aws_secret_access_key : str
    process_idx : int
        Index of the current process (for naming the missing csv file).
    """

    # Create a dedicated S3 resource inside this process
    s3 = boto3.resource(
        's3',
        aws_access_key_id     = aws_access_key_id,
        aws_secret_access_key = aws_secret_access_key
    )
    bucket = s3.Bucket(S3_BUCKET_NAME)

    # Transfer configuration (10 MB multipart chunk size, 500 concurrent parts)
    GB = 1024 ** 3
    config = TransferConfig(max_concurrency=500,
                            multipart_threshold=int(0.01 * GB),
                            multipart_chunksize=int(0.01 * GB))

    series_map = build_series_map()
    relpaths   = list(series_map.values())
    total_need = len(relpaths)

    # Dict to collect missing files for each subject
    missing_dict: Dict[str, List[str]] = {}

    tbar = tqdm(subjects, position=process_idx, desc=f'Proc {process_idx}', leave=False)

    for sid in tbar:
        base_prefix = f'{S3_PREFIX}/{sid}/MNINonLinear/'
        downloaded_cnt = 0
        missing_files  = []

        for relpath in relpaths:
            s3_key      = base_prefix + relpath
            local_fname = os.path.join(out_dir, f'{sid}_{os.path.basename(relpath)}')

            # Skip if already on disk
            if os.path.exists(local_fname):
                downloaded_cnt += 1
                continue

            try:
                bucket.download_file(s3_key, local_fname, Config=config)
                downloaded_cnt += 1
            except ClientError as e:
                # Most frequent error: 404 NoSuchKey
                missing_files.append(relpath)
            except Exception as e:
                # Any other exception: treat as missing for logging
                missing_files.append(relpath)
                print(f'[Warning][{sid}] Failed to download {s3_key}: {e}')

        if missing_files:
            missing_dict[sid] = missing_files

        # Progress bar postfix
        tbar.set_postfix({"got": f"{downloaded_cnt}/{total_need}"})

    tbar.close()

    # Dump missing files to csv (one per process to avoid race condition)
    if missing_dict:
        csv_name = os.path.join(out_dir,
                                f'missing_downloads_{process_idx}.csv')
        with open(csv_name, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for sid, miss in missing_dict.items():
                writer.writerow([sid] + miss)
        print(f'[Process {process_idx}] Missing file report -> {csv_name}')
    else:
        print(f'[Process {process_idx}] All requested files downloaded.')


# ------------------------------------------------------------------
# Multiprocessing wrapper
# ------------------------------------------------------------------
def run_process(
    subjects: List[str],
    out_dir: str,
    aws_access_key_id: str,
    aws_secret_access_key: str,
    cpu_worker_num: int
):
    """
    Split `subjects` into N chunks and start N processes,
    each running `download_known_keys`.
    """
    if cpu_worker_num <= 1:
        download_known_keys(
            subjects,
            out_dir,
            aws_access_key_id,
            aws_secret_access_key,
            process_idx=0
        )
        return

    # Calculate chunk indices
    total = len(subjects)
    step  = total // cpu_worker_num
    processes: List[Process] = []

    for idx in range(cpu_worker_num):
        start = idx * step
        end   = total if idx == cpu_worker_num - 1 else (idx + 1) * step
        sub_list = subjects[start:end]

        p = Process(
            target=download_known_keys,
            args=(sub_list,
                  out_dir,
                  aws_access_key_id,
                  aws_secret_access_key,
                  idx)
        )
        p.daemon = True
        processes.append(p)

    # Start / Join
    [p.start() for p in processes]
    [p.join() for p in processes]


# ------------------------------------------------------------------
# Main entry
# ------------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--id',  required=True, type=str, help='AWS access key ID')
    parser.add_argument('--key', required=True, type=str, help='AWS secret access key')
    parser.add_argument('-o', '--out_dir', required=True, type=str,
                        help='Path to local folder to download files to')
    parser.add_argument('--save_subject_id', action='store_true',
                        help='Parse hcp.csv and save subject IDs to all_pid.pkl')
    parser.add_argument('--cpu_worker', type=int, default=1,
                        help='Number of parallel worker processes')
    args = parser.parse_args()

    # Absolute output directory
    out_dir = os.path.abspath(args.out_dir)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1) Prepare subject list
    # ------------------------------------------------------------------
    if args.save_subject_id:
        import csv
        pset = set()
        csv_name = 'hcp.csv'
        with open(csv_name, newline='') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)                           # skip header
            for row in reader:
                pset.add(row[0])
        subjects = sorted(list(pset))
        with open('all_pid.pkl', 'wb') as f:
            pickle.dump(subjects, f, pickle.HIGHEST_PROTOCOL)
        print(f'Saved {len(subjects)} subject IDs to all_pid.pkl')
        exit(0)

    # Default path: read existing subject list
    with open('all_pid.pkl', 'rb') as f:
        subjects = pickle.load(f)

    # ------------------------------------------------------------------
    # 2) Launch download
    # ------------------------------------------------------------------
    run_process(
        subjects=subjects,
        out_dir=out_dir,
        aws_access_key_id=args.id,
        aws_secret_access_key=args.key,
        cpu_worker_num=max(1, args.cpu_worker)
    )
