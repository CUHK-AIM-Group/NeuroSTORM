import nibabel as nib
import torch
from scipy.ndimage import zoom
import os
import time
import math
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor
import argparse
import torch.nn.functional as F
import numpy as np


# Datasets where the foreground mask is derived from the fMRI itself (data == 0)
# instead of from a separate mask nii file.
NO_MASK_DATASETS = {'hcpya', 'hcpa', 'hcpd', 'ukb', 'hcptask'}


def select_middle_96(vector):
    start_index, end_index = [], []
    for i in range(3):
        if vector.shape[i] > 96:
            start_index.append((vector.shape[i] - 96) // 2)
            end_index.append(start_index[-1] + 96)
        else:
            start_index.append(0)
            end_index.append(vector.shape[i])

    if len(vector.shape) == 3:
        result = vector[start_index[0]:end_index[0], start_index[1]:end_index[1], start_index[2]:end_index[2]]
    elif len(vector.shape) == 4:
        result = vector[start_index[0]:end_index[0], start_index[1]:end_index[1], start_index[2]:end_index[2], :]
    
    return result


def spatial_resampling(data, header, target_voxel_size=(2, 2, 2)):
    current_voxel_size = header.get_zooms()[:3]
    scale_factors = [current / target for current, target in zip(current_voxel_size, target_voxel_size)]
    new_dims = [int(np.round(dim * scale)) for dim, scale in zip(data.shape[:3], scale_factors)]
    
    data = data.astype(np.float32)
    
    if data.ndim == 4:
        data_tensor = torch.from_numpy(data).permute(3, 0, 1, 2).unsqueeze(1)
    elif data.ndim == 3:
        data_tensor = torch.from_numpy(data).unsqueeze(0).unsqueeze(0)
    else:
        raise ValueError(f'unexpected data ndim={data.ndim}, shape={data.shape}')
    
    resampled_tensor = F.interpolate(data_tensor, size=new_dims, mode='trilinear', align_corners=False)
    
    if data.ndim == 4:
        resampled_data = resampled_tensor.squeeze(1).permute(1, 2, 3, 0).numpy()
    else:
        resampled_data = resampled_tensor.squeeze(0).squeeze(0).numpy()
    
    return resampled_data


def temporal_resampling(data, header, target_time_resolution=0.8):
    current_time_resolution = header.get_zooms()[3]
    scale_factor = current_time_resolution / target_time_resolution
    
    original_t = data.shape[3]
    new_t = max(int(np.round(original_t * scale_factor)), 1)

    x, y, z, t = data.shape
    data_reshaped = data.reshape(-1, t)
    data_tensor = torch.from_numpy(data_reshaped).unsqueeze(0)
    
    resampled_tensor = F.interpolate(data_tensor, size=new_t, mode='linear', align_corners=False)
    resampled_data = resampled_tensor.squeeze(0).numpy()
    resampled_data = resampled_data.reshape(x, y, z, new_t)
    
    return resampled_data


def _mask_path_for(dataset_name, path):
    """Return the brain-mask nii path for a non-HCP dataset, or None for HCP."""
    if dataset_name in NO_MASK_DATASETS:
        return None
    if dataset_name in {'abcd', 'cobre', 'hcpep'}:
        return path[:-19] + 'brain_mask.nii.gz'
    if dataset_name == 'movie':
        return path[:-57] + 'space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz'
    if dataset_name == 'transdiag':
        return path[:-19] + 'brainmask.nii.gz'
    if dataset_name == 'ucla':
        return path[:-14] + 'brainmask.nii.gz'
    return None


def _read_task_io(task):
    """IO stage: load fMRI nii (and mask nii if needed). GIL is released during
    nibabel's C-level decompression so this can run in a background thread."""
    dataset_name, _, filename, load_root, _, _, _, _ = task
    path = os.path.join(load_root, filename)
    img = nib.load(path)
    data = img.get_fdata()
    header = img.header

    mask_path = _mask_path_for(dataset_name, path)
    if mask_path is not None:
        mask_img = nib.load(mask_path)
        mask_data = mask_img.get_fdata()
        mask_header = mask_img.header
    else:
        mask_data, mask_header = None, None

    return {
        'path': path,
        'data': data,
        'header': header,
        'mask_data': mask_data,
        'mask_header': mask_header,
    }


def _process_task_cpu(task, io_pkg, fill_zeroback=False):
    """CPU stage: resample, normalize, int8-quantize, save data.pt."""
    dataset_name, delete_after_preprocess, filename, load_root, save_root, subj_name, count, scaling_method = task
    path = io_pkg['path']
    data = io_pkg['data']
    header = io_pkg['header']
    mask_data = io_pkg['mask_data']
    mask_header = io_pkg['mask_header']

    save_dir = os.path.join(save_root, subj_name)
    os.makedirs(save_dir, exist_ok=True)

    # resampling to fixed spatial and temporal resolution
    data = spatial_resampling(data, header)
    data = temporal_resampling(data, header)
    data = select_middle_96(data)

    # foreground/background mask
    if mask_data is None:
        background = data == 0
    else:
        mask = spatial_resampling(mask_data, mask_header)
        mask = select_middle_96(mask)
        background = mask == 0

    data[background] = 0
    data[data < 0] = 0
    data = torch.Tensor(data)

    # normalization
    if scaling_method == 'z-norm':
        global_mean = data[~background].mean()
        global_std = data[~background].std()
        data_temp = (data - global_mean) / global_std
    elif scaling_method == 'minmax':
        data_temp = (data - data[~background].min()) / (data[~background].max() - data[~background].min())

    data_global = torch.empty(data.shape)
    data_global[background] = data_temp[~background].min() if not fill_zeroback else 0
    data_global[~background] = data_temp[~background]

    # symmetric int8 quantization + permute to [T, H, W, D] for mmap-friendly clip reads
    abs_max = float(data_global.abs().max().item())
    scale = abs_max / 127.0 if abs_max > 0 else 1.0
    data_int = (data_global / scale).round().clamp_(-127, 127).to(torch.int8)
    assert data_int.ndim == 4, f'expected 4D data, got shape {tuple(data_int.shape)}'
    data_int = data_int.permute(3, 0, 1, 2).contiguous()

    blob = {
        'frames': data_int,
        'scale': float(scale),
        'num_frames': int(data_int.shape[0]),
    }
    torch.save(blob, os.path.join(save_dir, 'data.pt'))

    if delete_after_preprocess:
        os.remove(path)
        print('delete {}'.format(path), flush=True)


def _process_chunk(tasks_chunk):
    """Process a chunk of tasks with single-step IO lookahead.

    A background thread reads the next file's nii bytes while the main thread
    runs the current file's CPU pipeline (resample -> normalize -> save).
    Disk and CPU stay overlapping for the entire chunk.
    """
    if not tasks_chunk:
        return

    def safe_read(task):
        try:
            return _read_task_io(task), None
        except Exception as e:
            return None, e

    with ThreadPoolExecutor(max_workers=1, thread_name_prefix='nii-prefetch') as io_pool:
        # kick off the first read; everything else is one-step lookahead
        next_future = io_pool.submit(safe_read, tasks_chunk[0])
        for i, task in enumerate(tasks_chunk):
            filename = task[2]
            io_pkg, err = next_future.result()

            # schedule the next file's read BEFORE we start processing the current file
            if i + 1 < len(tasks_chunk):
                next_future = io_pool.submit(safe_read, tasks_chunk[i + 1])

            if err is not None:
                print(f'encountered problem with {filename} (read): {err}', flush=True)
                continue

            print('processing: ' + filename, flush=True)
            try:
                _process_task_cpu(task, io_pkg)
            except Exception as e:
                print(f'encountered problem with {filename}: {e}', flush=True)
            finally:
                # drop heavy arrays so GC can reclaim before the next iteration
                io_pkg = None


def main():
    parser = argparse.ArgumentParser(description='Process image data.')
    parser.add_argument('--dataset_name', type=str, required=True, help='dataset name')
    parser.add_argument('--load_root', type=str, required=True, help='directory to load data from')
    parser.add_argument('--save_root', type=str, required=True, help='directory to save data to')
    parser.add_argument('--delete_after_preprocess', action='store_true', help='delete nii file after preprocess')
    parser.add_argument('--delete_nii', action='store_true', help='if you did not delete after preprocess, you can use it to delete nii file')
    parser.add_argument('--num_processes', type=int, default=1, help='number of processes to use')
    parser.add_argument('--prefetch_chunk_size', type=int, default=0,
                        help='files per worker chunk for IO lookahead. 0 = auto '
                             '(~ceil(N / (num_processes*4))). Larger means more prefetch '
                             'benefit but worse load balancing. Set to 1 to disable prefetch.')
    args = parser.parse_args()

    dataset_name = args.dataset_name
    load_root = args.load_root
    save_root = args.save_root
    scaling_method = 'z-norm'

    filenames = os.listdir(load_root)
    os.makedirs(os.path.join(save_root, 'img'), exist_ok=True)
    os.makedirs(os.path.join(save_root, 'metadata'), exist_ok=True)
    save_root = os.path.join(save_root, 'img')

    count = 0

    tasks = []
    for filename in sorted(filenames):
        if not (filename.endswith('.nii.gz') or filename.endswith('.nii')) or 'mask' in filename or 'imagery' in filename or 'task-REST_acq' in filename:
            continue

        # Determine subject name based on dataset
        subj_name = determine_subject_name(dataset_name, filename)
        if args.delete_nii:
            handle_delete_nii(load_root, save_root, filename, subj_name)
            continue

        count += 1
        tasks.append((dataset_name, args.delete_after_preprocess, filename, load_root, save_root, subj_name, count, scaling_method))

    if not tasks:
        return

    num_procs = max(1, args.num_processes)

    # Pick chunk size:
    #   default = ceil(N / (num_processes * 4))  -> ~4 chunks per worker
    #   user-supplied value wins; 1 disables prefetch (chunks of size 1)
    if args.prefetch_chunk_size > 0:
        chunk_size = args.prefetch_chunk_size
    else:
        chunk_size = max(1, math.ceil(len(tasks) / max(num_procs * 4, 1)))

    chunks = [tasks[i:i + chunk_size] for i in range(0, len(tasks), chunk_size)]
    print(f'dispatching {len(tasks)} files -> {len(chunks)} chunk(s) of up to {chunk_size} across '
          f'{num_procs} process(es); IO prefetch {"enabled" if chunk_size > 1 else "disabled"}',
          flush=True)

    if num_procs == 1:
        for ch in chunks:
            _process_chunk(ch)
    else:
        with Pool(processes=num_procs) as pool:
            for _ in pool.imap_unordered(_process_chunk, chunks):
                pass


def determine_subject_name(dataset_name, filename):
    if dataset_name in ['abcd', 'cobre']:
        return filename.split('-')[1][:-4]
    elif dataset_name == 'adhd200':
        return filename.split('_')[2]
    elif dataset_name == 'god':
        return filename[:6] + '_' + filename.split('perception_')[1][:6]
    elif dataset_name == 'hcpya':
        return filename[:-7]
    elif dataset_name in ['hcpd', 'hcpa']:
        return filename[:10]
    elif dataset_name == 'hcpep':
        return filename[:8]
    elif dataset_name == 'ucla':
        return filename[:9]
    elif dataset_name == 'ukb':
        return filename.split('.')[0]
    elif dataset_name == 'hcptask':
        return filename.split('.')[0]
    elif dataset_name == 'movie':
        return filename.split('_acq')[0]
    elif dataset_name == 'transdiag':
        return filename.split('_task-testPA')[0].split('-')[1]

def handle_delete_nii(load_root, save_root, filename, subj_name):
    path = os.path.join(load_root, filename)
    save_dir = os.path.join(save_root, subj_name)
    data_pt = os.path.join(save_dir, 'data.pt')

    if os.path.isfile(data_pt):
        print(f'{subj_name} has data.pt at {data_pt}, removing nii: {path}')
        os.remove(path)
    else:
        print(f'{data_pt} not found, skipping nii removal for {subj_name}')

if __name__=='__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    print('\nTotal', round((end_time - start_time) / 60), 'minutes elapsed.')
