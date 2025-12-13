#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Download T1w / T2w images from the Human Connectome Project – 1200 subjects release.
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
# Single-process download function
# ----------------------------------------------------------------------
def download_t1_t2(
    subjects: List[str],
    out_dir: str,
    aws_access_key_id: str,
    aws_secret_access_key: str,
    proc_idx: int = 0
):
    """
    For every subject in `subjects`, download

        MNINonLinear/T1w.nii.gz
        MNINonLinear/T2w.nii.gz

    Skip files that already exist.  Record failures to CSV.
    """
    # Build S3 connection in this process
    s3 = boto3.resource(
        's3',
        aws_access_key_id     = aws_access_key_id,
        aws_secret_access_key = aws_secret_access_key
    )
    bucket = s3.Bucket(S3_BUCKET_NAME)

    # Transfer configuration: 10 MB multipart chunks, high concurrency
    GB = 1024 ** 3
    config = TransferConfig(
        max_concurrency   = 500,
        multipart_threshold = int(0.01 * GB),
        multipart_chunksize = int(0.01 * GB)
    )

    # Target relative paths
    REL_PATHS = {
        'T1w' : 'MNINonLinear/T1w.nii.gz',
        'T2w' : 'MNINonLinear/T2w.nii.gz'
    }

    # Collect missing downloads
    missing_dict: Dict[str, List[str]] = {}

    pbar = tqdm(
        subjects,
        position=proc_idx,
        leave=False,
        desc=f'Proc {proc_idx}'
    )

    for sid in pbar:
        base_prefix = f'{S3_PREFIX}/{sid}/'
        ok_cnt      = 0
        missing     = []

        for tag, rel in REL_PATHS.items():
            s3_key = base_prefix + rel
            local_fname = os.path.join(out_dir, f'{sid}_{tag}.nii.gz')

            # Skip if already downloaded
            if os.path.exists(local_fname):
                ok_cnt += 1
                continue

            try:
                bucket.download_file(s3_key, local_fname, Config=config)
                ok_cnt += 1
            except ClientError:
                # 404 / no such key
                missing.append(tag)
            except Exception as e:
                missing.append(tag)
                print(f'[Warning][{sid}] Cannot download {s3_key}: {e}')

        if missing:
            missing_dict[sid] = missing

        pbar.set_postfix({'ok': ok_cnt})

    pbar.close()

    # Save missing info
    if missing_dict:
        csv_name = os.path.join(out_dir, f'missing_downloads_{proc_idx}.csv')
        with open(csv_name, 'w', newline='') as f:
            writer = csv.writer(f)
            for sid, miss in missing_dict.items():
                writer.writerow([sid] + miss)
        print(f'[Process {proc_idx}] Missing list saved -> {csv_name}')


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
    """Split `subjects` into `workers` chunks and start processes."""
    if workers <= 1:
        download_t1_t2(subjects, out_dir,
                       aws_access_key_id, aws_secret_access_key, 0)
        return

    total   = len(subjects)
    stride  = total // workers
    procs: List[Process] = []

    for idx in range(workers):
        s = idx * stride
        e = total if idx == workers - 1 else (idx + 1) * stride
        subset = subjects[s:e]
        p = Process(target=download_t1_t2,
                    args=(subset, out_dir,
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

    # Make sure output directory exists
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # Build / load subject list ------------------------------------------
    if args.save_subject_id:
        import csv
        subj_set = set()
        with open('hcp.csv', newline='') as f:
            reader = csv.reader(f)
            next(reader)                       # skip header
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

    # Launch download ----------------------------------------------------
    run_processes(subjects, out_dir,
                  args.id, args.key,
                  max(1, args.cpu_worker))

    print('All downloads finished.')
