"""Convert legacy per-frame .pt files to single-file int8 blob format.

Reads frame_0.pt .. frame_N.pt (float16 [H, W, D, 1]) from each subject
directory and produces a single data.pt with:
    {'frames': int8[T, H, W, D], 'scale': float, 'num_frames': int}

Usage:
    python datasets/convert_frames_to_blob.py \
        --input_dir /data/cwang/remote/fmri/ABCD_MNI_to_TRs_minmax/img \
        --output_dir /data/cwang/remote/fmri/ABCD_MNI_to_TRs_minmax/img \
        --num_processes 16

If --output_dir equals --input_dir, data.pt is written alongside the
existing frame_*.pt files (the loader auto-detects blob format and
ignores the per-frame files).
"""

import os
import re
import torch
import argparse
from multiprocessing import Pool
from functools import partial


def convert_subject(subject_path, output_path):
    frame_files = [f for f in os.listdir(subject_path)
                   if re.match(r'frame_\d+\.pt$', f)]
    if not frame_files:
        return subject_path, 'skip_no_frames'

    if os.path.isfile(os.path.join(output_path, 'data.pt')):
        return subject_path, 'skip_exists'

    indices = sorted(int(re.search(r'(\d+)', f).group()) for f in frame_files)
    num_frames = len(indices)

    first = torch.load(os.path.join(subject_path, f'frame_{indices[0]}.pt'),
                       weights_only=True)
    H, W, D, _ = first.shape

    buf = torch.empty((num_frames, H, W, D), dtype=torch.float32)
    for i, idx in enumerate(indices):
        frame = torch.load(os.path.join(subject_path, f'frame_{idx}.pt'),
                           weights_only=True)
        buf[i] = frame.to(torch.float32).squeeze(-1)

    abs_max = float(buf.abs().max().item())
    scale = abs_max / 127.0 if abs_max > 0 else 1.0
    data_int = (buf / scale).round().clamp_(-127, 127).to(torch.int8)

    blob = {
        'frames': data_int,
        'scale': float(scale),
        'num_frames': num_frames,
    }

    os.makedirs(output_path, exist_ok=True)
    torch.save(blob, os.path.join(output_path, 'data.pt'))
    return subject_path, 'ok'


def _worker(subject_name, input_dir, output_dir):
    subject_path = os.path.join(input_dir, subject_name)
    output_path = os.path.join(output_dir, subject_name)
    try:
        return convert_subject(subject_path, output_path)
    except Exception as e:
        return subject_path, f'error: {e}'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True,
                        help='Path to img/ directory with per-subject folders')
    parser.add_argument('--output_dir', required=True,
                        help='Output img/ directory (can be same as input)')
    parser.add_argument('--num_processes', type=int, default=8)
    args = parser.parse_args()

    subjects = sorted([d for d in os.listdir(args.input_dir)
                       if os.path.isdir(os.path.join(args.input_dir, d))])
    print(f'Found {len(subjects)} subjects in {args.input_dir}')

    worker_fn = partial(_worker, input_dir=args.input_dir, output_dir=args.output_dir)

    done = 0
    errors = 0
    skipped = 0
    with Pool(args.num_processes) as pool:
        for path, status in pool.imap_unordered(worker_fn, subjects, chunksize=4):
            if status == 'ok':
                done += 1
            elif status.startswith('skip'):
                skipped += 1
            else:
                errors += 1
                print(f'  FAILED: {path} — {status}')
            total = done + skipped + errors
            if total % 100 == 0:
                print(f'  progress: {total}/{len(subjects)} '
                      f'(converted={done}, skipped={skipped}, errors={errors})')

    print(f'\nDone. converted={done}, skipped={skipped}, errors={errors}')


if __name__ == '__main__':
    main()
