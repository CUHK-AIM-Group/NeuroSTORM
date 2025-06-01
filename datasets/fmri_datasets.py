import os
import torch
from torch.utils.data import Dataset, IterableDataset
import torch.nn.functional as F

import numpy as np
import random
import math
import pandas as pd


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
    def __init__(self, **kwargs):
        super().__init__()      
        self.register_args(**kwargs)
        self.sample_duration = self.sequence_length * self.stride_within_seq
        self.stride = max(round(self.stride_between_seq * self.sample_duration), 1)
        self.data = self._set_data(self.root, self.subject_dict)

        # import ipdb; ipdb.set_trace()
        # index = 0
        # _, subject_name, subject_path, start_frame, sequence_length, num_frames, target, sex = self.data[index]
        # y = self.load_sequence(subject_path, start_frame, sequence_length, num_frames)
        # y = pad_to_96(y)
        # y = resize_volume(y, self.img_size)
    
    def register_args(self,**kwargs):
        for name, value in kwargs.items():
            setattr(self, name, value)
        self.kwargs = kwargs
    
    def load_sequence(self, subject_path, start_frame, sample_duration, num_frames=None): 
        if self.contrastive or self.mae:
            num_frames = len(os.listdir(subject_path))
            y = []
            load_fnames = [f'frame_{frame}.pt' for frame in range(start_frame, start_frame+sample_duration, self.stride_within_seq)]
            if self.with_voxel_norm:
                load_fnames += ['voxel_mean.pt', 'voxel_std.pt']

            last_y = None
            for fname in load_fnames:
                img_path = os.path.join(subject_path, fname)

                try:
                    y_loaded = torch.load(img_path).unsqueeze(0)
                    y.append(y_loaded)
                    last_y = y_loaded
                except:
                    print('load {} failed'.format(img_path))
                    if last_y is None:
                        y.append(self.previous_last_y)
                        last_y = self.previous_last_y
                    else:
                        y.append(last_y)
            
            self.previous_last_y = y[-1]
            y = torch.cat(y, dim=4)
            
            if self.mae:
                random_y = torch.zeros(1)
            else:
                random_y = []
                
                full_range = np.arange(0, num_frames-sample_duration+1)
                # exclude overlapping sub-sequences within a subject
                exclude_range = np.arange(start_frame-sample_duration, start_frame+sample_duration)
                available_choices = np.setdiff1d(full_range, exclude_range)
                random_start_frame = np.random.choice(available_choices, size=1, replace=False)[0]
                load_fnames = [f'frame_{frame}.pt' for frame in range(random_start_frame, random_start_frame+sample_duration, self.stride_within_seq)]
                if self.with_voxel_norm:
                    load_fnames += ['voxel_mean.pt', 'voxel_std.pt']

                last_y = None
                for fname in load_fnames:
                    img_path = os.path.join(subject_path, fname)

                    try:
                        y_loaded = torch.load(img_path).unsqueeze(0)
                        random_y.append(y_loaded)
                        last_y = y_loaded
                    except:
                        print('load {} failed'.format(img_path))
                        if last_y is None:
                            random_y.append(self.previous_last_y)
                            last_y = self.previous_last_y
                        else:
                            random_y.append(last_y)
                
                self.previous_last_y = y[-1]
                random_y = torch.cat(random_y, dim=4)
            
            return (y, random_y)

        else: # without contrastive learning
            y = []
            if self.shuffle_time_sequence: # shuffle whole sequences
                load_fnames = [f'frame_{frame}.pt' for frame in random.sample(list(range(0, num_frames)), sample_duration // self.stride_within_seq)]
            else:
                load_fnames = [f'frame_{frame}.pt' for frame in range(start_frame, start_frame+sample_duration, self.stride_within_seq)]
            
            if self.with_voxel_norm:
                load_fnames += ['voxel_mean.pt', 'voxel_std.pt']
                
            for fname in load_fnames:
                img_path = os.path.join(subject_path, fname)
                y_i = torch.load(img_path).unsqueeze(0)
                y.append(y_i)
            y = torch.cat(y, dim=4)
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
            num_frames = len(os.listdir(subject_path))
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

            num_frames = len(os.listdir(subject_path))
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
            num_frames = len(os.listdir(subject_path))
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
            num_frames = len(os.listdir(subject_path))
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
            num_frames = len(os.listdir(subject_path))
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
            num_frames = len(os.listdir(subject_path))
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
            num_frames = len(os.listdir(subject_path))
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

            num_frames = len(os.listdir(subject_path))
            if num_frames < self.stride:
                import ipdb; ipdb.set_trace()
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

            num_frames = len(os.listdir(subject_path))
            if num_frames < self.stride:
                import ipdb; ipdb.set_trace()
            session_duration = num_frames - self.sample_duration + 1

            # we only use first n frames for task fMRI
            data_tuple = (i, subject_name, subject_path, 0, self.stride, num_frames, target, sex)
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

            num_frames = len(os.listdir(subject_path))
            if num_frames < self.stride:
                import ipdb; ipdb.set_trace()
            session_duration = num_frames - self.sample_duration + 1

            for start_frame in range(0, session_duration, self.stride):
                data_tuple = (i, subject_name, subject_path, start_frame, self.stride, num_frames, target, sex)
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
                import ipdb; ipdb.set_trace()
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
            num_frames = len(os.listdir(subject_path))
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
            num_frames = len(os.listdir(subject_path))
            session_duration = num_frames - self.sample_duration + 1
            for start_frame in range(0, session_duration, self.stride):
                data_tuple = (i, subject, subject_path, start_frame, self.stride, num_frames, target, sex)
                data.append(data_tuple)

        if self.train: 
            self.target_values = np.array([tup[6] for tup in data]).reshape(-1, 1)
        
        return data
