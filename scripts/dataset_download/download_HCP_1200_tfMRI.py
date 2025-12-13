#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Download task-fMRI (tfMRI) runs from the HCP 1200 release.

For every subject we try to download 14 files:
    7 tasks  ×  LR / RL
        WM, SOCIAL, RELATIONAL, MOTOR,
        LANGUAGE, GAMBLING, EMOTION
Example key (relative to .../<SID>/MNINonLinear/):
    Results/tfMRI_WM_LR/tfMRI_WM_LR.nii.gz
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


# ----------------------------------------------------------------------
# S3 constants
# ----------------------------------------------------------------------
S3_BUCKET_NAME = 'hcp-openaccess'
S3_PREFIX      = 'HCP_1200'               # root folder in the bucket


# ----------------------------------------------------------------------
# Build relative path list for tfMRI
# ----------------------------------------------------------------------
def build_tfmri_relpaths() -> List[str]:
    """Return 14 relative paths under MNINonLinear/Results/"""
    tasks = ['WM', 'SOCIAL', 'RELATIONAL', 'MOTOR',
             'LANGUAGE', 'GAMBLING', 'EMOTION']
    relpaths = []
    for task in tasks:
        for run in ('LR', 'RL'):
            rel = f'Results/tfMRI_{task}_{run}/tfMRI_{task}_{run}.nii.gz'
            relpaths.append(rel)
    return relpaths


TFMRI_RELPATHS = build_tfmri_relpaths()
EXPECTED_N = len(TFMRI_RELPATHS)           # 14


# ----------------------------------------------------------------------
# Single process downloading
# ----------------------------------------------------------------------
def download_tfmri(
    subjects: List[str],
    out_dir: str,
    aws_access_key_id: str,
    aws_secret_access_key: str,
    proc_idx: int = 0
):
    """
    Download all tfMRI runs for `subjects`.

    Missing files are recorded into CSV: missing_tfmri_<proc_idx>.csv
    """
    # S3 connection (per process)
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

    # Missing recorder
    missing_dict: Dict[str, List[str]] = {}

    pbar = tqdm(
        subjects,
        position=proc_idx,
        desc=f'Proc {proc_idx}',
        leave=False
    )

    for sid in pbar:
        base_prefix = f'{S3_PREFIX}/{sid}/MNINonLinear/'
        got_cnt     = 0
        missing     = []

        for rel in TFMRI_RELPATHS:
            s3_key = base_prefix + rel
            # Use final filename (tfMRI_WM_LR.nii.gz) with subject id prefix
            local_fname = os.path.join(
                out_dir,
                f'{sid}_{os.path.basename(rel)}'
            )

            # Skip if file already exists
            if os.path.exists(local_fname):
                got_cnt += 1
                continue

            try:
                bucket.download_file(s3_key, local_fname, Config=config)
                got_cnt += 1
            except ClientError:
                missing.append(rel)
            except Exception as e:
                missing.append(rel)
                print(f'[Warning][{sid}] Failed to download {s3_key}: {e}')

        if missing:
            missing_dict[sid] = missing

        pbar.set_postfix({'ok': got_cnt})

    pbar.close()

    # Dump missing list
    if missing_dict:
        csv_path = os.path.join(out_dir, f'missing_tfmri_{proc_idx}.csv')
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            for sid, miss in missing_dict.items():
                writer.writerow([sid] + miss)
        print(f'[Process {proc_idx}] Missing files saved → {csv_path}')


# ----------------------------------------------------------------------
# Multiprocessing wrapper
# ----------------------------------------------------------------------
def run_processes(
    subjects: List[str],
    out_dir: str,
    aws_access_key_id: str,
    aws_secret_access_key: str,
    workers: int
):
    if workers <= 1:
        download_tfmri(subjects, out_dir,
                       aws_access_key_id, aws_secret_access_key, 0)
        return

    total   = len(subjects)
    step    = total // workers
    procs: List[Process] = []

    for idx in range(workers):
        s = idx * step
        e = total if idx == workers - 1 else (idx + 1) * step
        sub_list = subjects[s:e]

        p = Process(target=download_tfmri,
                    args=(sub_list, out_dir,
                          aws_access_key_id, aws_secret_access_key, idx))
        p.daemon = True
        p.start()
        procs.append(p)

    [p.join() for p in procs]


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--id',  required=True, type=str, help='AWS access key id')
    parser.add_argument('--key', required=True, type=str, help='AWS secret access key')
    parser.add_argument('--out_dir', required=True, type=str, help='Output directory')
    parser.add_argument('--save_subject_id', action='store_true',
                        help='Parse hcp.csv and save subject IDs to all_pid.pkl')
    parser.add_argument('--cpu_worker', type=int, default=1,
                        help='Number of parallel worker processes')
    args = parser.parse_args()

    # Prepare output dir --------------------------------------------------
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # Build / load subject list ------------------------------------------
    if args.save_subject_id:
        subj_set = set()
        with open('hcp.csv', newline='') as f:
            rdr = csv.reader(f)
            next(rdr)
            for row in rdr:
                subj_set.add(row[0])
        subjects = sorted(subj_set)
        with open('all_pid.pkl', 'wb') as f:
            pickle.dump(subjects, f, pickle.HIGHEST_PROTOCOL)
        print(f'Saved {len(subjects)} subject IDs to all_pid.pkl')
        exit(0)
    else:
        with open('all_pid.pkl', 'rb') as f:
            subjects = pickle.load(f)

    # Launch download -----------------------------------------------------
    run_processes(subjects, out_dir,
                  args.id, args.key,
                  max(1, args.cpu_worker))

    print('tfMRI download finished.')
