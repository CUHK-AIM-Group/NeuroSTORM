#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Download resting-state fMRI (either 3 T set or 7 T set) from
the HCP 1200 release.
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


S3_BUCKET_NAME = 'hcp-openaccess'
S3_PREFIX      = 'HCP_1200'               # root folder in the bucket


FMRI_3T_PATHS = [
    'Results/rfMRI_REST1_LR/rfMRI_REST1_LR_hp2000_clean.nii.gz',
    'Results/rfMRI_REST1_RL/rfMRI_REST1_RL_hp2000_clean.nii.gz',
    'Results/rfMRI_REST2_LR/rfMRI_REST1_LR_hp2000_clean.nii.gz',
    'Results/rfMRI_REST2_RL/rfMRI_REST1_RL_hp2000_clean.nii.gz',
]

FMRI_7T_PATHS = [
    'Results/rfMRI_REST1_7T_PA/rfMRI_REST1_7T_PA_hp2000_clean.nii.gz',
    'Results/rfMRI_REST2_7T_AP/rfMRI_REST2_7T_AP_hp2000_clean.nii.gz',
    'Results/rfMRI_REST3_7T_PA/rfMRI_REST3_7T_PA_hp2000_clean.nii.gz',
    'Results/rfMRI_REST4_7T_AP/rfMRI_REST4_7T_AP_hp2000_clean.nii.gz',
]


# ----------------------------------------------------------------------
# Single-process download routine
# ----------------------------------------------------------------------
def download_rest_fmri(
    subjects: List[str],
    out_dir: str,
    aws_access_key_id: str,
    aws_secret_access_key: str,
    proc_idx: int = 0
):
    """Download 3 T or 7 T resting-state fMRI for each subject."""

    # S3 connection in this process
    s3 = boto3.resource(
        's3',
        aws_access_key_id     = aws_access_key_id,
        aws_secret_access_key = aws_secret_access_key
    )
    bucket = s3.Bucket(S3_BUCKET_NAME)

    # Transfer configuration
    GB = 1024 ** 3
    config = TransferConfig(
        max_concurrency   = 500,
        multipart_threshold = int(0.01 * GB),
        multipart_chunksize = int(0.01 * GB)
    )

    # Missing download recorder
    missing_dict: Dict[str, List[str]] = {}

    bar = tqdm(subjects, position=proc_idx, desc=f'Proc {proc_idx}', leave=False)

    for sid in bar:
        base_prefix = f'{S3_PREFIX}/{sid}/MNINonLinear/'

        # First try 3 T paths
        chosen_set  = FMRI_3T_PATHS
        success     = 0
        tmp_missing = []

        for rel in chosen_set:
            key   = base_prefix + rel
            fname = os.path.join(out_dir, f'{sid}_{os.path.basename(rel)}')

            # Already on disk?
            if os.path.exists(fname):
                success += 1
                continue

            try:
                bucket.download_file(key, fname, Config=config)
                success += 1
            except ClientError:
                tmp_missing.append(rel)
            except Exception as e:
                tmp_missing.append(rel)
                print(f'[Warning][{sid}] download error: {e}')

        if success == 0:
            chosen_set  = FMRI_7T_PATHS
            tmp_missing = []
            for rel in chosen_set:
                key   = base_prefix + rel
                fname = os.path.join(out_dir, f'{sid}_{os.path.basename(rel)}')

                if os.path.exists(fname):
                    success += 1
                    continue
                try:
                    bucket.download_file(key, fname, Config=config)
                    success += 1
                except ClientError:
                    tmp_missing.append(rel)
                except Exception as e:
                    tmp_missing.append(rel)
                    print(f'[Warning][{sid}] download error: {e}')

        # Record missing (if any)
        if tmp_missing:
            missing_dict[sid] = tmp_missing

        bar.set_postfix({'ok': success, 'set': '7T' if success and chosen_set is FMRI_7T_PATHS else '3T'})

    bar.close()

    # Write missing CSV for this process
    if missing_dict:
        csv_name = os.path.join(out_dir, f'missing_downloads_{proc_idx}.csv')
        with open(csv_name, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for sid, miss in missing_dict.items():
                writer.writerow([sid] + miss)
        print(f'[Process {proc_idx}] missing list → {csv_name}')


# ----------------------------------------------------------------------
# Multiprocess wrapper
# ----------------------------------------------------------------------
def run_processes(
    subjects: List[str],
    out_dir: str,
    aws_access_key_id: str,
    aws_secret_access_key: str,
    workers: int
):
    """Split subject list and launch multiple processes."""
    if workers <= 1:
        download_rest_fmri(subjects, out_dir,
                           aws_access_key_id, aws_secret_access_key, 0)
        return

    total = len(subjects)
    stride = total // workers
    processes: List[Process] = []

    for idx in range(workers):
        s = idx * stride
        e = total if idx == workers - 1 else (idx + 1) * stride
        sub_slice = subjects[s:e]

        p = Process(target=download_rest_fmri,
                    args=(sub_slice, out_dir,
                          aws_access_key_id, aws_secret_access_key, idx))
        p.daemon = True
        p.start()
        processes.append(p)

    [p.join() for p in processes]


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--id',  required=True, type=str, help='AWS access key id')
    parser.add_argument('--key', required=True, type=str, help='AWS secret key')
    parser.add_argument('-o', '--out_dir', required=True, type=str,
                        help='local output directory')
    parser.add_argument('--save_subject_id', action='store_true',
                        help='Parse hcp.csv and store subject id list')
    parser.add_argument('--cpu_worker', type=int, default=1,
                        help='number of parallel processes')
    args = parser.parse_args()

    # Prepare subject list ------------------------------------------------
    if args.save_subject_id:
        import csv
        subj_set = set()
        with open('hcp.csv', newline='') as f:
            reader = csv.reader(f)
            next(reader)                     # skip header
            for row in reader:
                subj_set.add(row[0])
        subjects = sorted(subj_set)
        with open('all_pid.pkl', 'wb') as f:
            pickle.dump(subjects, f, pickle.HIGHEST_PROTOCOL)
        print(f'Saved {len(subjects)} subject IDs to all_pid.pkl')
        exit(0)
    else:
        with open('all_pid.pkl', 'rb') as f:
            subjects = pickle.load(f)

    # Ensure output directory exists -------------------------------------
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # Launch download -----------------------------------------------------
    run_processes(subjects, out_dir,
                  args.id, args.key,
                  max(1, args.cpu_worker))
    print('All processes finished.')
