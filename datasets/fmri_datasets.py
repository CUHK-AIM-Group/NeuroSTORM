import os
import torch
from torch.utils.data import Dataset, IterableDataset
import torch.nn.functional as F

import numpy as np
import random
import math
import pandas as pd
from collections import OrderedDict


def pad_to_96(y):
    background_value = y.flatten()[0]
    y = y.permute(0,4,1,2,3)
    pad_1 = 96 - y.shape[-1]
    pad_2 = 96 - y.shape[-2]
    pad_3 = 96 - y.shape[-3]
    y = torch.nn.functional.pad(y, (math.ceil(pad_1/2), math.floor(pad_1/2), math.ceil(pad_2/2), math.floor(pad_2/2), math.ceil(pad_3/2), math.floor(pad_3/2)), value=background_value)[:,:,:,:,:]
    y = y.permute(0,2,3,4,1)

    return y

import torch
import torch.nn.functional as F

def resize_volume(y, target_size):
    """
    y: 5D tensor [B, H, W, D, T]
    target_size: 4d tuple/list (H', W', D', T)
    """
    current_size = y.shape

    if len(current_size) != 5:
        raise ValueError("Input y must be a 5-dimensional tensor.")
    if len(target_size) != 4:
        raise ValueError("Target size must be a tuple or list of length 4.")

    if current_size[-1] != target_size[-1]:
        raise ValueError(f"y's last dimension {current_size[-1]} and target_size's last dimension {target_size[-1]} must match.")
        
    if current_size[1:4] != tuple(target_size[:3]):
        resized_y = torch.empty(
            (current_size[0],) + tuple(target_size),
            dtype=y.dtype, device=y.device
        )

        for i in range(current_size[0]):
            for t in range(current_size[-1]):
                data_tensor = y[i, :, :, :, t]

                original_dtype = data_tensor.dtype
                resized_tensor = F.interpolate(data_tensor.float().unsqueeze(0).unsqueeze(0), size=tuple(target_size[:3]), mode='trilinear', align_corners=False).squeeze(0).squeeze(0).to(original_dtype)
                resized_y[i, :, :, :, t] = resized_tensor
        return resized_y
    else:
        return y


class BaseDataset(Dataset):
    # Upper bound on per-worker int8 blobs held in RAM at once.
    # Each blob is ~hundreds of MB; keep this small so 16 workers don't
    # blow up host memory (cache_max * num_workers * blob_size).
    _blob_cache_max = 8

    def __init__(self, **kwargs):
        super().__init__()
        self.register_args(**kwargs)
        self.sample_duration = self.sequence_length * self.stride_within_seq
        self.stride = max(round(self.stride_between_seq * self.sample_duration), 1)
        self._data_blobs = OrderedDict()
        self._format_cache = {}
        self._num_frames_cache = {}
        self.data = self._set_data(self.root, self.subject_dict)

    def __getstate__(self):
        """Drop mmap'd blobs before pickling (e.g. for DataLoader workers)."""
        state = self.__dict__.copy()
        state['_data_blobs'] = OrderedDict()
        return state

    def register_args(self,**kwargs):
        for name, value in kwargs.items():
            setattr(self, name, value)
        self.kwargs = kwargs

    def _detect_format(self, subject_path):
        """Detect storage layout for a subject directory.

        Returns:
            'blob'   : new int8 single-file format (data.pt with mmap)
            'frames' : legacy per-frame float16 .pt files (frame_*.pt)
        """
        fmt = self._format_cache.get(subject_path)
        if fmt is None:
            if os.path.isfile(os.path.join(subject_path, 'data.pt')):
                fmt = 'blob'
            else:
                fmt = 'frames'
            self._format_cache[subject_path] = fmt
        return fmt

    def _get_blob(self, subject_path):
        """Lazily load a subject's data.pt; LRU-evict older entries.

        mmap=True is the default and cheap, but some sshfs/NFS backends deny
        PROT_READ mappings (torch.load surfaces this as `Operation not permitted`).
        On that error we fall back to a plain read for that subject. Set
        NEUROSTORM_BLOB_MMAP=0 to skip the mmap attempt entirely.
        """
        blob = self._data_blobs.get(subject_path)
        if blob is not None:
            self._data_blobs.move_to_end(subject_path)
            return blob
        blob_path = os.path.join(subject_path, 'data.pt')
        prefer_mmap = os.environ.get("NEUROSTORM_BLOB_MMAP", "1") != "0"
        if prefer_mmap:
            try:
                blob = torch.load(blob_path, mmap=True, weights_only=True)
            except RuntimeError as e:
                if "Operation not permitted" not in str(e):
                    raise
                blob = torch.load(blob_path, mmap=False, weights_only=True)
        else:
            blob = torch.load(blob_path, mmap=False, weights_only=True)
        self._data_blobs[subject_path] = blob
        while len(self._data_blobs) > self._blob_cache_max:
            self._data_blobs.popitem(last=False)
        return blob

    def _count_frames(self, subject_path):
        cached = self._num_frames_cache.get(subject_path)
        if cached is not None:
            return cached
        if self._detect_format(subject_path) == 'blob':
            # peek metadata cheaply via mmap; fall back if the FS denies it
            peek_path = os.path.join(subject_path, 'data.pt')
            try:
                peek = torch.load(peek_path, mmap=True, weights_only=True)
            except RuntimeError as e:
                if "Operation not permitted" not in str(e):
                    raise
                peek = torch.load(peek_path, mmap=False, weights_only=True)
            n = int(peek['num_frames'])
            del peek
        else:
            n = len([f for f in os.listdir(subject_path)
                     if f.startswith('frame_') and f.endswith('.pt')])
        self._num_frames_cache[subject_path] = n
        return n

    def _load_clip(self, subject_path, indices):
        """Load the requested time indices and return float32 [1, H, W, D, T']."""
        if self._detect_format(subject_path) == 'blob':
            blob = self._get_blob(subject_path)
            frames = blob['frames']  # int8, [T, H, W, D]
            scale = float(blob['scale'])
            if isinstance(indices, range):
                clip = frames[indices.start:indices.stop:indices.step]
            else:
                clip = frames[torch.as_tensor(list(indices), dtype=torch.long)]
            clip = clip.to(torch.float32).mul_(scale)
            # [T', H, W, D] -> [1, H, W, D, T']
            return clip.permute(1, 2, 3, 0).unsqueeze(0)

        # legacy per-frame format: frame_i.pt is float16 [H, W, D, 1]
        idx_list = list(indices)
        parts = []
        for i in idx_list:
            frame = torch.load(os.path.join(subject_path, f'frame_{i}.pt'),
                               weights_only=True)
            parts.append(frame.to(torch.float32))
        clip = torch.cat(parts, dim=3)        # [H, W, D, T']
        return clip.unsqueeze(0)              # [1, H, W, D, T']

    def _maybe_append_voxel_norm(self, clip, subject_path):
        if not self.with_voxel_norm:
            return clip
        vm = torch.load(os.path.join(subject_path, 'voxel_mean.pt'), weights_only=True)
        vs = torch.load(os.path.join(subject_path, 'voxel_std.pt'), weights_only=True)
        # voxel_mean/std saved as [H, W, D, 1]; broadcast batch dim and concat along T
        if vm.ndim == 4:
            vm = vm.unsqueeze(0)
            vs = vs.unsqueeze(0)
        return torch.cat([clip, vm.to(clip.dtype), vs.to(clip.dtype)], dim=4)

    def load_sequence(self, subject_path, start_frame, sample_duration, num_frames=None):
        if self.contrastive or self.mae:
            num_frames = self._count_frames(subject_path)

            if self.shuffle_time_sequence:
                indices = random.sample(range(num_frames), sample_duration // self.stride_within_seq)
            else:
                indices = range(start_frame, start_frame + sample_duration, self.stride_within_seq)
            y = self._load_clip(subject_path, indices)
            y = self._maybe_append_voxel_norm(y, subject_path)

            if self.mae:
                random_y = torch.zeros(1)
            else:
                # contrastive: pick a non-overlapping second clip
                full_range = np.arange(0, num_frames - sample_duration + 1)
                exclude_range = np.arange(start_frame - sample_duration, start_frame + sample_duration)
                available_choices = np.setdiff1d(full_range, exclude_range)
                random_start_frame = int(np.random.choice(available_choices, size=1, replace=False)[0])
                rand_indices = range(random_start_frame, random_start_frame + sample_duration, self.stride_within_seq)
                random_y = self._load_clip(subject_path, rand_indices)
                random_y = self._maybe_append_voxel_norm(random_y, subject_path)

            return (y, random_y)

        # supervised path
        if self.shuffle_time_sequence:
            indices = random.sample(range(num_frames), sample_duration // self.stride_within_seq)
        else:
            indices = range(start_frame, start_frame + sample_duration, self.stride_within_seq)
        y = self._load_clip(subject_path, indices)
        y = self._maybe_append_voxel_norm(y, subject_path)
        return y

    def __len__(self):
        return  len(self.data)

    def __getitem__(self, index):
        _, subject_name, subject_path, start_frame, sequence_length, num_frames, target, sex = self.data[index]

        if self.contrastive or self.mae:
            y, rand_y = self.load_sequence(subject_path, start_frame, sequence_length)
            y = pad_to_96(y)
            y = resize_volume(y, self.img_size)

            if self.contrastive:
                rand_y = pad_to_96(rand_y)
                rand_y = resize_volume(rand_y, self.img_size)

            return {
                "fmri_sequence": (y, rand_y),
                "subject_name": subject_name,
                "target": target,
                "TR": start_frame,
                "sex": sex
            }
        else:   
            y = self.load_sequence(subject_path, start_frame, sequence_length, num_frames)
            y = pad_to_96(y)
            y = resize_volume(y, self.img_size)

            return {
                "fmri_sequence": y,
                "subject_name": subject_name,
                "target": target,
                "TR": start_frame,
                "sex": sex,
            }

    def _set_data(self, root, subject_dict):
        raise NotImplementedError("Required function")
 

class HCP1200(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _set_data(self, root, subject_dict):
        data = []
        img_root = os.path.join(root, 'img')
        for i, subject in enumerate(subject_dict):
            sex,target = subject_dict[subject]
            subject_path = os.path.join(img_root, subject)
            num_frames = self._count_frames(subject_path)
            session_duration = num_frames - self.sample_duration + 1
            for start_frame in range(0, session_duration, self.stride):
                data_tuple = (i, subject, subject_path, start_frame, self.stride, num_frames, target, sex)
                data.append(data_tuple)

        if self.train: 
            self.target_values = np.array([tup[6] for tup in data]).reshape(-1, 1)
        return data


class ABCD(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _set_data(self, root, subject_dict):
        data = []
        img_root = os.path.join(root, 'img')

        for i, subject_name in enumerate(subject_dict):
            sex, target = subject_dict[subject_name]
            # subject_name = subject[4:]
            
            subject_path = os.path.join(img_root, subject_name)

            num_frames = self._count_frames(subject_path)
            session_duration = num_frames - self.sample_duration + 1

            for start_frame in range(0, session_duration, self.stride):
                data_tuple = (i, subject_name, subject_path, start_frame, self.stride, num_frames, target, sex)
                data.append(data_tuple)

        if self.train: 
            self.target_values = np.array([tup[6] for tup in data]).reshape(-1, 1)

        return data


class Cobre(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _set_data(self, root, subject_dict):
        data = []
        img_root = os.path.join(root, 'img')

        for i, subject_name in enumerate(subject_dict):
            sex, target = subject_dict[subject_name]
            subject_path = os.path.join(img_root, subject_name)
            num_frames = self._count_frames(subject_path)
            session_duration = num_frames - self.sample_duration + 1

            for start_frame in range(0, session_duration, self.stride):
                data_tuple = (i, subject_name, subject_path, start_frame, self.stride, num_frames, target, sex)
                data.append(data_tuple)
                        
        if self.train: 
            self.target_values = np.array([tup[6] for tup in data]).reshape(-1, 1)

        return data


class ADHD200(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _set_data(self, root, subject_dict):
        data = []
        img_root = os.path.join(root, 'img')

        for i, subject_name in enumerate(subject_dict):
            sex, target = subject_dict[subject_name]
            subject_path = os.path.join(img_root, '{}'.format(subject_name))
            num_frames = self._count_frames(subject_path)
            session_duration = num_frames - self.sample_duration + 1

            for start_frame in range(0, session_duration, self.stride):
                data_tuple = (i, subject_name, subject_path, start_frame, self.stride, num_frames, target, sex)
                data.append(data_tuple)
                        
        if self.train: 
            self.target_values = np.array([tup[6] for tup in data]).reshape(-1, 1)

        return data
        

class UCLA(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _set_data(self, root, subject_dict):
        data = []
        img_root = os.path.join(root, 'img')

        for i, subject_name in enumerate(subject_dict):
            sex, target = subject_dict[subject_name]
            subject_path = os.path.join(img_root, '{}'.format(subject_name))
            num_frames = self._count_frames(subject_path)
            session_duration = num_frames - self.sample_duration + 1

            for start_frame in range(0, session_duration, self.stride):
                data_tuple = (i, subject_name, subject_path, start_frame, self.stride, num_frames, target, sex)
                data.append(data_tuple)
                        
        if self.train: 
            self.target_values = np.array([tup[6] for tup in data]).reshape(-1, 1)

        return data


class HCPEP(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _set_data(self, root, subject_dict):
        data = []
        img_root = os.path.join(root, 'img')

        for i, subject_name in enumerate(subject_dict):
            sex, target = subject_dict[subject_name]
            subject_path = os.path.join(img_root, '{}'.format(subject_name))
            num_frames = self._count_frames(subject_path)
            session_duration = num_frames - self.sample_duration + 1

            for start_frame in range(0, session_duration, self.stride):
                data_tuple = (i, subject_name, subject_path, start_frame, self.stride, num_frames, target, sex)
                data.append(data_tuple)
                        
        if self.train: 
            self.target_values = np.array([tup[6] for tup in data]).reshape(-1, 1)

        return data


class GOD(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _set_data(self, root, subject_dict):
        data = []
        img_root = os.path.join(root, 'img')

        for i, subject_name in enumerate(subject_dict):
            sex, target = subject_dict[subject_name]
            subject_path = os.path.join(img_root, '{}'.format(subject_name))
            num_frames = self._count_frames(subject_path)
            session_duration = num_frames - self.sample_duration + 1

            for start_frame in range(0, session_duration, self.stride):
                data_tuple = (i, subject_name, subject_path, start_frame, self.stride, num_frames, target, sex)
                data.append(data_tuple)
                        
        if self.train: 
            self.target_values = np.array([tup[6] for tup in data]).reshape(-1, 1)

        return data


class UKB(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _set_data(self, root, subject_dict):
        data = []
        img_root = os.path.join(root, 'img')

        for i, subject_name in enumerate(subject_dict):
            sex, target = subject_dict[subject_name]
            # subject_name = subject[4:]
            
            subject_path = os.path.join(img_root, subject_name)

            num_frames = self._count_frames(subject_path)
            if num_frames < self.stride:
                raise ValueError(f'subject {subject_name}: num_frames={num_frames} < stride={self.stride}')
            session_duration = num_frames - self.sample_duration + 1

            for start_frame in range(0, session_duration, self.stride):
                data_tuple = (i, subject_name, subject_path, start_frame, self.stride, num_frames, target, sex)
                data.append(data_tuple)

        if self.train: 
            self.target_values = np.array([tup[6] for tup in data]).reshape(-1, 1)

        return data
    

class HCPTASK(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _set_data(self, root, subject_dict):
        data = []
        img_root = os.path.join(root, 'img')

        for i, subject_name in enumerate(subject_dict):
            sex, target = subject_dict[subject_name]
            subject_path = os.path.join(img_root, subject_name)

            num_frames = self._count_frames(subject_path)
            if num_frames < self.stride:
                raise ValueError(f'subject {subject_name}: num_frames={num_frames} < stride={self.stride}')
            session_duration = num_frames - self.sample_duration + 1

            # we only use first n frames for task fMRI
            data_tuple = (i, subject_name, subject_path, 0, self.stride, num_frames, target, sex)
            data.append(data_tuple)

        if self.train:
            self.target_values = np.array([tup[6] for tup in data]).reshape(-1, 1)

        return data


class MOVIE(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def process_tsv(self, path):
        df = pd.read_csv(path, sep='\t')
        result = []
        for _, row in df.iterrows():
            if row['trial_type'] == 'High_cal_food':
                label = 0
            elif row['trial_type'] == 'Low_cal_food':
                label = 1
            elif row['trial_type'] == 'non-food':
                label = 2
            else:
                raise ValueError(f"unexpected trial_type: {row['trial_type']}")
            entry = {
                'start': int(row['onset'] / 0.8),
                'end': int((row['onset'] + row['duration']) / 0.8) + 10,
                'label': label
            }
            result.append(entry)

        result[-1]['end'] -= 10

        return result

    def _set_data(self, root, subject_dict):
        data = []
        img_root = os.path.join(root, 'img')

        for i, subject in enumerate(subject_dict):
            sex, target = subject_dict[subject]
            subject_path = os.path.join(img_root, subject)
            num_frames = self._count_frames(subject_path)
            session_duration = num_frames - self.sample_duration + 1
            for start_frame in range(0, session_duration, self.stride):
                data_tuple = (i, subject, subject_path, start_frame, self.stride, num_frames, target, sex)
                data.append(data_tuple)

        # label_root = os.path.join(root, 'metadata')

        # for i, subject_name in enumerate(subject_dict):
        #     sex, target = subject_dict[subject_name]
        #     subject_path = os.path.join(img_root, subject_name)
        #     label_path = os.path.join(label_root, '{}-food_events.tsv'.format(subject_name[:-5]))
        #     label = self.process_tsv(label_path)

        #     num_frames = len(os.listdir(subject_path))
        #     if num_frames < self.stride:
        #         import ipdb; ipdb.set_trace()
        #     session_duration = num_frames - self.sample_duration + 1

        #     for j in range(len(label)):
        #         for start_frame in range(label[j]['start'], label[j]['end'], self.stride):
        #             if start_frame + self.stride >= num_frames:
        #                 continue

        #             data_tuple = (i, subject_name, subject_path, start_frame, self.stride, num_frames, label[j]['label'], sex)
        #             data.append(data_tuple)
            
        if self.train: 
            self.target_values = np.array([tup[6] for tup in data]).reshape(-1, 1)

        return data


class TransDiag(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _set_data(self, root, subject_dict):
        data = []
        img_root = os.path.join(root, 'img')

        for i, subject in enumerate(subject_dict):
            sex, target = subject_dict[subject]
            subject_path = os.path.join(img_root, subject)
            num_frames = self._count_frames(subject_path)
            session_duration = num_frames - self.sample_duration + 1
            for start_frame in range(0, session_duration, self.stride):
                data_tuple = (i, subject, subject_path, start_frame, self.stride, num_frames, target, sex)
                data.append(data_tuple)

        if self.train: 
            self.target_values = np.array([tup[6] for tup in data]).reshape(-1, 1)
        
        return data
