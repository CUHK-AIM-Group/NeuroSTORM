import os
import pytorch_lightning as pl
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset, ConcatDataset, Sampler
from torch_geometric.data import DataLoader as GeometricDataLoader
from .fmri_datasets import HCP1200, ABCD, UKB, Cobre, ADHD200, UCLA, HCPEP, HCPTASK, GOD, MOVIE, TransDiag
from .roi_datasets import ROIDataset, FCDataset, FCGraphDataset
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from utils.parser import str2bool


# =============================================================================
# Multi-dataset sampling strategies for pretraining
# =============================================================================

class MultiDatasetSampler(Sampler):
    """Base class for multi-dataset sampling strategies.

    Subclasses implement `_compute_indices()` which returns the list of global
    indices (into the ConcatDataset) to use for the current epoch.

    DDP-aware: when running in distributed mode, each rank gets a disjoint
    slice of the indices. Set replace_sampler_ddp=False in the Trainer.
    """

    def __init__(self, concat_dataset, seed=0):
        self.concat_dataset = concat_dataset
        self.cumulative_sizes = concat_dataset.cumulative_sizes
        self.num_datasets = len(self.cumulative_sizes)
        self.seed = seed
        self.epoch = 0

    @property
    def rank(self):
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_rank()
        return 0

    @property
    def world_size(self):
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_world_size()
        return 1

    def set_epoch(self, epoch):
        self.epoch = epoch

    def _dataset_ranges(self):
        """Return list of (start, end) index ranges for each sub-dataset."""
        ranges = []
        prev = 0
        for end in self.cumulative_sizes:
            ranges.append((prev, end))
            prev = end
        return ranges

    def _compute_indices(self):
        raise NotImplementedError

    def __iter__(self):
        indices = self._compute_indices()
        # DDP sharding: each rank takes its slice
        if self.world_size > 1:
            indices = indices[self.rank::self.world_size]
        return iter(indices)

    def __len__(self):
        total = len(self._compute_indices())
        if self.world_size > 1:
            return (total + self.world_size - 1) // self.world_size
        return total


class UniformSubsampleStrategy(MultiDatasetSampler):
    """Strategy 1: Uniform subsampling by subject.

    Each epoch, randomly sample at most `max_subjects` subjects from each
    sub-dataset, then include ALL clips belonging to those subjects.
    Different random subset of subjects each epoch.
    """

    def __init__(self, concat_dataset, max_samples_per_dataset=500, seed=0):
        super().__init__(concat_dataset, seed=seed)
        self.max_subjects = max_samples_per_dataset
        self._subject_groups = self._build_subject_groups()

    def _build_subject_groups(self):
        """Build per-dataset mapping: subject_index -> list of global indices."""
        groups = []
        offset = 0
        for ds in self.concat_dataset.datasets:
            subj_to_indices = {}
            for local_idx, item in enumerate(ds.data):
                subj_idx = item[0]
                global_idx = offset + local_idx
                if subj_idx not in subj_to_indices:
                    subj_to_indices[subj_idx] = []
                subj_to_indices[subj_idx].append(global_idx)
            groups.append(subj_to_indices)
            offset += len(ds.data)
        return groups

    def _compute_indices(self):
        rng = np.random.RandomState(self.seed + self.epoch)
        indices = []
        for subj_to_indices in self._subject_groups:
            all_subjs = list(subj_to_indices.keys())
            n = min(len(all_subjs), self.max_subjects)
            chosen_subjs = rng.choice(all_subjs, n, replace=False)
            for s in chosen_subjs:
                indices.extend(subj_to_indices[s])
        rng.shuffle(indices)
        return indices

    def __len__(self):
        return len(self._compute_indices())


class LossWeightedStrategy(MultiDatasetSampler):
    """Strategy 2: Loss-weighted resampling.

    Each epoch, the number of samples drawn from each dataset is proportional
    to its average loss from the previous epoch. Datasets with higher loss get
    more samples, preventing training from being dominated by easy/large
    datasets. Falls back to uniform sampling on the first epoch.

    Call `update_losses(per_dataset_losses)` at the end of each epoch with a
    list of average losses (one per sub-dataset).
    """

    def __init__(self, concat_dataset, total_samples_per_epoch=2500, seed=0):
        super().__init__(concat_dataset, seed=seed)
        self.total_samples = total_samples_per_epoch
        self._per_dataset_weights = np.ones(self.num_datasets) / self.num_datasets

    def update_losses(self, per_dataset_losses):
        """Update sampling weights based on per-dataset average losses.

        Args:
            per_dataset_losses: list/array of length num_datasets, each entry
                is the mean training loss for that dataset in the last epoch.
        """
        losses = np.array(per_dataset_losses, dtype=np.float64)
        losses = np.clip(losses, 1e-8, None)
        self._per_dataset_weights = losses / losses.sum()

    def _compute_indices(self):
        rng = np.random.RandomState(self.seed + self.epoch)
        indices = []
        ranges = self._dataset_ranges()
        per_dataset_n = (self._per_dataset_weights * self.total_samples).astype(int)
        remainder = self.total_samples - per_dataset_n.sum()
        top_k = np.argsort(-self._per_dataset_weights)[:remainder]
        per_dataset_n[top_k] += 1

        for i, (start, end) in enumerate(ranges):
            size = end - start
            n = min(int(per_dataset_n[i]), size)
            chosen = rng.choice(size, n, replace=False) + start
            indices.extend(chosen.tolist())
        rng.shuffle(indices)
        return indices

    def __len__(self):
        return self.total_samples


# --- Add new strategies here ---
# class YourNewStrategy(MultiDatasetSampler):
#     """Strategy N: Description."""
#     def __init__(self, concat_dataset, ...):
#         super().__init__(concat_dataset, seed=seed)
#         ...
#     def _compute_indices(self):
#         ...


SAMPLING_STRATEGIES = {
    "uniform_subsample": UniformSubsampleStrategy,
    "loss_weighted": LossWeightedStrategy,
    # Register new strategies here
}


def build_multi_dataset_sampler(strategy_name, concat_dataset, hparams):
    """Factory function to build a multi-dataset sampler from hparams."""
    seed = getattr(hparams, 'seed', 0)
    if strategy_name == "uniform_subsample":
        max_samples = getattr(hparams, 'max_samples_per_dataset', 500)
        return UniformSubsampleStrategy(concat_dataset, max_samples_per_dataset=max_samples, seed=seed)
    elif strategy_name == "loss_weighted":
        num_datasets = len(concat_dataset.cumulative_sizes)
        max_samples = getattr(hparams, 'max_samples_per_dataset', 500)
        total = max_samples * num_datasets
        return LossWeightedStrategy(concat_dataset, total_samples_per_epoch=total, seed=seed)
    else:
        raise ValueError(f"Unknown sampling strategy: {strategy_name}. "
                         f"Available: {list(SAMPLING_STRATEGIES.keys())}")


def _rank0_print(*args, **kwargs):
    # DDP spawn re-executes main.py; those worker processes will have
    # NEUROSTORM_IS_WORKER=1 set by the launcher. Skip their setup prints so
    # we don't see every line twice.
    if os.environ.get("NEUROSTORM_IS_WORKER") == "1":
        return
    if int(os.environ.get("LOCAL_RANK", "0")) != 0:
        return
    print(*args, **kwargs)

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
import random


def select_elements(S, n):
    level_count = defaultdict(int)
    for value in S.values():
        level_count[value[1]] += 1

    total_elements = sum(level_count.values())
    level_quota = {level: int(n * count / total_elements) for level, count in level_count.items()}

    remaining = n - sum(level_quota.values())
    levels = sorted(level_count.keys(), key=lambda x: -level_count[x])
    for i in range(remaining):
        level_quota[levels[i % len(levels)]] += 1

    selected_elements = []
    for level in level_quota:
        elements_of_level = [k for k, v in S.items() if v[1] == level]
        selected_elements.extend(random.sample(elements_of_level, level_quota[level]))

    S_prime = {k: S[k] for k in selected_elements}

    return S_prime


class fMRIDataModule(pl.LightningDataModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.dataset_names, self.image_paths = self._parse_dataset_specs()
        if len(self.dataset_names) > 1 and not self.hparams.pretraining:
            raise ValueError(
                "Multiple datasets via --dataset_name are only supported in --pretraining mode. "
                f"Got dataset_name={self.hparams.dataset_name} without --pretraining."
            )

        # generate splits folder
        if self.hparams.pretraining:
            # per-dataset split paths so each dataset keeps its own subject list
            self.split_file_paths = []
            for name in self.dataset_names:
                split_dir_path = f'./data/splits/{name}/pretraining'
                os.makedirs(split_dir_path, exist_ok=True)
                self.split_file_paths.append(
                    os.path.join(split_dir_path, f"split_fixed_{self.hparams.dataset_split_num}.txt")
                )
            # keep legacy attribute pointing at the first dataset's split for backward compat
            self.split_file_path = self.split_file_paths[0]
        else:
            split_dir_path = f'./data/splits/{self.dataset_names[0]}'
            os.makedirs(split_dir_path, exist_ok=True)
            self.split_file_path = os.path.join(split_dir_path, f"split_fixed_{self.hparams.dataset_split_num}.txt")
            self.split_file_paths = [self.split_file_path]

        self._is_setup = False
        self.setup()

    def _parse_dataset_specs(self):
        """Parse --dataset_name / --image_path into aligned lists.

        Both arguments accept comma-separated strings in pretraining mode.
        If a single image_path is provided for multiple datasets we raise,
        since each dataset lives in its own root directory.
        """
        raw_names = str(self.hparams.dataset_name)
        names = [n.strip() for n in raw_names.split(',') if n.strip()]

        raw_paths = self.hparams.image_path
        if raw_paths is None:
            paths = [None] * len(names)
        else:
            paths = [p.strip() for p in str(raw_paths).split(',') if p.strip()]

        if len(names) > 1:
            if len(paths) != len(names):
                raise ValueError(
                    f"--dataset_name has {len(names)} entries but --image_path has {len(paths)}. "
                    "Pass a comma-separated --image_path with one root per dataset."
                )
        else:
            # single dataset: keep the path list aligned (may contain a single entry)
            if len(paths) == 0:
                paths = [None]
            elif len(paths) != 1:
                # user listed multiple paths but a single dataset name - unusual, keep first
                paths = paths[:1]

        return names, paths

    def get_dataset(self, dataset_name=None):
        name = dataset_name if dataset_name is not None else self.dataset_names[0]
        if name == "HCP1200":
            return HCP1200
        elif name == "ABCD":
            return ABCD
        elif name == 'UKB':
            return UKB
        elif name in ('HCPA', 'HCPD'):
            # HCPA / HCPD share the HCP-style directory layout and are only
            # used for pretraining today, so reuse the HCP1200 wrapper.
            return HCP1200
        elif name == 'Cobre':
            return Cobre
        elif name == 'ADHD200':
            return ADHD200
        elif name == 'UCLA':
            return UCLA
        elif name == 'HCPEP':
            return HCPEP
        elif name == 'GOD':
            return GOD
        elif name == 'HCPTASK':
            return HCPTASK
        elif name == 'MOVIE':
            return MOVIE
        elif name == 'TransDiag':
            return TransDiag
        else:
            raise NotImplementedError(f"Unsupported dataset: {name}")

    def convert_subject_list_to_idx_list(self, train_names, val_names, test_names, subj_list):
        subj_idx = np.array([str(x[1]) for x in subj_list])
        S = np.unique([x[1] for x in subj_list])
        print('unique subjects:',len(S))  
        train_idx = np.where(np.in1d(subj_idx, train_names))[0].tolist()
        val_idx = np.where(np.in1d(subj_idx, val_names))[0].tolist()
        test_idx = np.where(np.in1d(subj_idx, test_names))[0].tolist()
        
        return train_idx, val_idx, test_idx
    
    def save_split(self, sets_dict, split_file_path=None):
        path = split_file_path if split_file_path is not None else self.split_file_path
        with open(path, "w+") as f:
            for name, subj_list in sets_dict.items():
                f.write(name + "\n")
                for subj_name in subj_list:
                    f.write(str(subj_name) + "\n")

    def determine_split_randomly(self, S, split_file_path=None):
        np.random.seed(0)
        S_keys = list(S.keys())
        S_train = int(len(S_keys) * self.hparams.train_split)
        S_val = int(len(S_keys) * self.hparams.val_split)

        if self.hparams.downstream_task_type == 'classification':
            S_train = select_elements(S, S_train)
            S_remaining = {k: v for k, v in S.items() if k not in S_train}
            S_train_keys = list(S_train.keys())
        else:
            S_train_keys = np.random.choice(S_keys, S_train, replace=False)

        remaining_keys = np.setdiff1d(S_keys, S_train_keys)

        if self.hparams.downstream_task_type == 'classification':
            S_val = select_elements(S_remaining, S_val)
            S_val_keys = list(S_val.keys())
            if self.hparams.val_split + self.hparams.train_split < 1:
                S_test = {k: v for k, v in S_remaining.items() if k not in S_val}
                S_test_keys = list(S_test.keys())
        else:
            S_val_keys = np.random.choice(remaining_keys, S_val, replace=False)
            if self.hparams.val_split + self.hparams.train_split < 1:
                S_test_keys = np.setdiff1d(S_keys, np.concatenate([S_train_keys, S_val_keys]))

        if self.hparams.val_split + self.hparams.train_split < 1:
            self.save_split({"train_subjects": S_train_keys, "val_subjects": S_val_keys, "test_subjects": S_test_keys}, split_file_path)
            return S_train_keys, S_val_keys, S_test_keys
        else:
            self.save_split({"train_subjects": S_train_keys, "val_subjects": S_val_keys, "test_subjects": S_val_keys}, split_file_path)
            return S_train_keys, S_val_keys, S_val_keys

    def load_split(self, split_file_path=None):
        path = split_file_path if split_file_path is not None else self.split_file_path
        subject_order = open(path, "r").readlines()
        subject_order = [x[:-1] for x in subject_order]
        train_index = np.argmax(["train" in line for line in subject_order])
        val_index = np.argmax(["val" in line for line in subject_order])
        test_index = np.argmax(["test" in line for line in subject_order])
        train_names = subject_order[train_index + 1 : val_index]
        val_names = subject_order[val_index + 1 : test_index]
        test_names = subject_order[test_index + 1 :]

        return train_names, val_names, test_names

    def prepare_data(self):
        # This function is only called at global rank==0
        return
    
    # filter subjects with metadata and pair subject names with their target values (+ sex)
    def make_subject_dict(self, dataset_name=None, image_path=None):
        """Build the {subject_id: [sex, target]} mapping for one dataset.

        When ``dataset_name`` / ``image_path`` are provided we temporarily swap
        the corresponding hparams so the per-dataset branches below keep
        reading from ``self.hparams`` unchanged.
        """
        swapped = False
        if dataset_name is not None or image_path is not None:
            prev_name = self.hparams.dataset_name
            prev_path = self.hparams.image_path
            if dataset_name is not None:
                self.hparams.dataset_name = dataset_name
            if image_path is not None:
                self.hparams.image_path = image_path
            swapped = True

        try:
            # Fast path for pretraining: labels are not used by MAE / contrastive,
            # so we skip metadata parsing and accept every subject directory.
            if self.hparams.pretraining:
                img_root = os.path.join(self.hparams.image_path, 'img')
                final_dict = {subj: [0, 0] for subj in sorted(os.listdir(img_root))}
                _rank0_print('Load dataset {} for pretraining, {} subjects'.format(
                    self.hparams.dataset_name, len(final_dict)))
                return final_dict

            return self._make_subject_dict_with_labels()
        finally:
            if swapped:
                self.hparams.dataset_name = prev_name
                self.hparams.image_path = prev_path

    def _make_subject_dict_with_labels(self):
        img_root = os.path.join(self.hparams.image_path, 'img')
        final_dict = dict()

        if self.hparams.dataset_name == "HCP1200":
            subject_list = os.listdir(img_root)
            meta_data = pd.read_csv(os.path.join(self.hparams.image_path, "metadata", "HCP_1200_gender.csv"))
            meta_data_residual = pd.read_csv(os.path.join(self.hparams.image_path, "metadata", "HCP_1200_precise_age.csv"))
            if self.hparams.task_name == 'sex':
                task_name = 'Gender'
            elif self.hparams.task_name == 'age':
                task_name = 'age'
            elif self.hparams.task_name == 'fmri_reid':
                # re-identification uses subject id as class; we'll handle label mapping after collecting subjects
                task_name = 'fmri_reid'
            # MMSE_Score Social_Task_Random_Perc_TOM CogTotalComp_Unadj Emotion_Task_Acc Language_Task_Acc Strength_Unadj 
            elif self.hparams.downstream_task_id == 2:
                task_name = self.hparams.task_name
            else:
                raise NotImplementedError()

            print('downstream_task_id = {}, task_name = {}'.format(self.hparams.downstream_task_id, task_name))

            if task_name == 'Gender':
                meta_task = meta_data[['Subject',task_name]].dropna()
            elif task_name == 'age':
                meta_task = meta_data_residual[['subject',task_name,'sex']].dropna()
                meta_task = meta_task.rename(columns={'subject': 'Subject'})
            elif task_name == 'fmri_reid':
                # metadata not needed for labels; keep placeholder with Subject column for filtering
                meta_task = pd.DataFrame({'Subject': meta_data['Subject'].dropna()})
            elif self.hparams.downstream_task_id == 2:
                meta_task = meta_data[['Subject', task_name, 'Gender']].dropna()  
            
            for subject in subject_list:
                if int(subject) in meta_task['Subject'].values:
                    if task_name == 'Gender':
                        target = meta_task[meta_task["Subject"]==int(subject)][task_name].values[0]
                        target = 1 if target == "M" else 0
                        sex = target
                    elif task_name == 'age':
                        target = meta_task[meta_task["Subject"]==int(subject)][task_name].values[0]
                        sex = meta_task[meta_task["Subject"]==int(subject)]["sex"].values[0]
                        sex = 1 if sex == "M" else 0
                    elif task_name == 'fmri_reid':
                        # placeholder target; will be replaced by contiguous ids below
                        target = 0
                        # try to keep sex if available; default to 0 when missing
                        if int(subject) in meta_data['Subject'].values:
                            g = meta_data[meta_data["Subject"]==int(subject)]["Gender"].values
                            if len(g) > 0:
                                sex = 1 if g[0] == "M" else 0
                            else:
                                sex = 0
                        else:
                            sex = 0
                    elif self.hparams.downstream_task_id == 2:
                        target = meta_task[meta_task["Subject"]==int(subject)][task_name].values[0]
                        sex = meta_task[meta_task["Subject"]==int(subject)]["Gender"].values[0]
                        sex = 1 if sex == "M" else 0
                    final_dict[subject] = [sex, target]
            
            # If reid task, remap subjects to contiguous class ids and keep mapping
            if task_name == 'fmri_reid':
                subj_sorted = sorted(final_dict.keys(), key=lambda x: int(x))
                subj_to_label = {subj: idx for idx, subj in enumerate(subj_sorted)}
                label_to_subj = {idx: subj for subj, idx in subj_to_label.items()}
                for subj in subj_sorted:
                    sex, _ = final_dict[subj]
                    final_dict[subj] = [sex, subj_to_label[subj]]
                self.reid_label_map = label_to_subj
                self.reid_label_inverse = subj_to_label
                self.hparams.num_classes = len(subj_sorted)

            print('Load dataset HCP1200, {} subjects'.format(len(final_dict)))
            
        elif self.hparams.dataset_name == "ABCD":
            subject_list = [subj for subj in os.listdir(img_root)]
            
            meta_data = pd.read_csv(os.path.join(self.hparams.image_path, "metadata", "abcd-rest.csv"))
            if self.hparams.task_name == 'sex': task_name = 'sex'
            elif self.hparams.downstream_task == 'age': task_name = 'age'
            else: raise ValueError('downstream task not supported')
           
            if task_name == 'sex':
                meta_task = meta_data[['subjectkey', task_name]].dropna()
            else:
                meta_task = meta_data[['subjectkey', task_name, 'sex']].dropna()
            
            for subject in subject_list:
                if subject in meta_task['subjectkey'].values:
                    target = meta_task[meta_task["subjectkey"]==subject][task_name].values[0]
                    if task_name == 'sex':
                        target = 1 if target == "M" else 0
                    sex = meta_task[meta_task["subjectkey"]==subject]["sex"].values[0]
                    sex = 1 if sex == "M" else 0
                    final_dict[subject] = [sex, target]
            
            print('Load dataset ABCD, {} subjects'.format(len(final_dict)))
        
        elif self.hparams.dataset_name == "Cobre":
            subject_list = [subj for subj in os.listdir(img_root)]
            
            meta_data = pd.read_csv(os.path.join(self.hparams.image_path, "metadata", "cobre-rest.csv"))
            if self.hparams.task_name == 'sex': task_name = 'sex'
            elif self.hparams.task_name == 'age': task_name = 'age'
            elif self.hparams.task_name == 'diagnosis': task_name = 'dx'
            else: raise ValueError('downstream task not supported')
           
            if task_name == 'sex':
                meta_task = meta_data[['subject_id', task_name]].dropna()
            else:
                meta_task = meta_data[['subject_id', task_name, 'sex']].dropna()
            
            for subject in subject_list:
                if subject in meta_task['subject_id'].values:
                    target = meta_task[meta_task["subject_id"]==subject][task_name].values[0]
                    if task_name == 'sex':
                        target = 1 if target == "M" else 0
                    elif task_name == 'dx':
                        if target == 'Schizophrenia_Strict': target = 0
                        elif target == 'Schizoaffective': target = 1
                        elif target == 'No_Known_Disorder': target = 2
                        elif target == 'Bipolar_Disorder': target = 3
                        else: 
                            import ipdb; ipdb.set_trace()
                        
                    sex = meta_task[meta_task["subject_id"]==subject]["sex"].values[0]
                    sex = 1 if sex == "male" else 0
                    final_dict[subject] = [sex, target]
            
            print('Load dataset Cobre, {} subjects'.format(len(final_dict)))
        
        elif self.hparams.dataset_name == "ADHD200":
            subject_list = [subj for subj in os.listdir(img_root)]
            
            meta_data = pd.read_csv(os.path.join(self.hparams.image_path, "metadata", "adhd200-rest.csv"))
            if self.hparams.task_name == 'sex': task_name = 'Gender'
            elif self.hparams.task_name == 'age': task_name = 'Age'
            elif self.hparams.task_name == 'diagnosis': task_name = 'DX'
            else: raise ValueError('downstream task not supported')
           
            if task_name == 'sex':
                meta_task = meta_data[['subject_id', task_name]].dropna()
            else:
                meta_task = meta_data[['subject_id', task_name, 'Gender']].dropna()
            
            for subject in subject_list:
                if int(subject) in meta_task['subject_id'].values:
                    target = meta_task[meta_task["subject_id"]==int(subject)][task_name].values[0]
                    if task_name == 'DX':
                        if target == 'pending': continue
                        
                        target = int(target)
                        target = 1 if target > 0 else 0
                        
                    sex = meta_task[meta_task["subject_id"]==int(int(subject))]["Gender"].values[0]
                    sex = int(sex)
                    final_dict[subject] = [sex, target]
            
            print('Load dataset ADHD200, {} subjects'.format(len(final_dict)))
        
        elif self.hparams.dataset_name == "UCLA":
            subject_list = [subj for subj in os.listdir(img_root)]
            
            meta_data = pd.read_csv(os.path.join(self.hparams.image_path, "metadata", "ucla-rest.csv"))
            if self.hparams.task_name == 'sex': task_name = 'gender'
            elif self.hparams.task_name == 'age': task_name = 'age'
            elif self.hparams.task_name == 'diagnosis': task_name = 'diagnosis'
            else: raise ValueError('downstream task not supported')
           
            if task_name == 'sex':
                meta_task = meta_data[['subject_id', task_name]].dropna()
            else:
                meta_task = meta_data[['subject_id', task_name, 'gender']].dropna()
            
            for subject in subject_list:
                if subject in meta_task['subject_id'].values:
                    target = meta_task[meta_task["subject_id"]==subject][task_name].values[0]
                    if task_name == 'gender':
                        target = 1 if target == "M" else 0
                    elif task_name == 'diagnosis':
                        if target == 'CONTROL': target = 0
                        elif target == 'SCHZ': target = 1
                        elif target == 'BIPOLAR': target = 2
                        elif target == 'ADHD': target = 3
                        else: 
                            import ipdb; ipdb.set_trace()
                        
                    sex = meta_task[meta_task["subject_id"]==subject]["gender"].values[0]
                    sex = 1 if sex == "M" else 0
                    final_dict[subject] = [sex, target]
            
            print('Load dataset UCLA, {} subjects'.format(len(final_dict)))
        
        elif self.hparams.dataset_name == "HCPEP":
            subject_list = [subj for subj in os.listdir(img_root)]
            
            meta_data = pd.read_csv(os.path.join(self.hparams.image_path, "metadata", "hcpep-rest.csv"))
            if self.hparams.task_name == 'sex': task_name = 'sex'
            elif self.hparams.task_name == 'age': task_name = 'interview_age'
            elif self.hparams.task_name == 'diagnosis': task_name = 'phenotype'
            else: raise ValueError('downstream task not supported')
           
            if task_name == 'sex':
                meta_task = meta_data[['subject_id', task_name]].dropna()
            else:
                meta_task = meta_data[['subject_id', task_name, 'sex']].dropna()
            
            for subject in subject_list:
                if int(subject[-4:]) in meta_task['subject_id'].values:
                    target = meta_task[meta_task["subject_id"]==int(subject[-4:])][task_name].values[0]
                    if task_name == 'sex':
                        target = 1 if target == "M" else 0
                    elif task_name == 'phenotype':
                        if target == 'Control': target = 0
                        elif target == 'Patient': target = 1
                        else:
                            import ipdb; ipdb.set_trace()
                        
                    sex = meta_task[meta_task["subject_id"]==int(subject[-4:])]["sex"].values[0]
                    sex = 1 if sex == "M" else 0
                    final_dict[subject] = [sex, target]
            
            print('Load dataset HCPEP, {} subjects'.format(len(final_dict)))
        
        elif self.hparams.dataset_name == "GOD":
            subject_list = [subj for subj in os.listdir(img_root)]
            
            meta_data = pd.read_csv(os.path.join(self.hparams.image_path, "metadata", "god_label.csv"))
            if self.hparams.downstream_task_id == 4: task_name = 'class'
            else: raise ValueError('downstream task not supported')
           
            meta_task = meta_data[['subject_id', task_name]].dropna()
            
            for subject in subject_list:
                if subject in meta_task['subject_id'].values:
                    target = meta_task[meta_task["subject_id"]==subject][task_name].values[0]
                    if task_name == 'sex':
                        target = 1 if target == "M" else 0
                    elif task_name == 'class':
                        target = target - 1
                        if target >= 150:
                            import ipdb; ipdb.set_trace()
                        
                    sex = 0
                    final_dict[subject] = [sex, target]

            category_count = defaultdict(int)
            for subject_id, (gender, category) in final_dict.items():
                category_count[category] += 1

            categories_to_delete = {category for category, count in category_count.items() if count < 40}
            final_dict = {subject_id: [gender, category] for subject_id, (gender, category) in final_dict.items() if category not in categories_to_delete}

            unique_categories = sorted(set(category for gender, category in final_dict.values()))
            category_mapping = {old_category: new_category for new_category, old_category in enumerate(unique_categories)}

            for subject_id in final_dict:
                final_dict[subject_id][1] = category_mapping[final_dict[subject_id][1]]

            print('Load dataset GOD, {} subjects, {} classes'.format(len(final_dict), len(unique_categories)))

        elif self.hparams.dataset_name == "UKB":
            subject_list = [subj for subj in os.listdir(img_root)]

            meta_data = pd.read_csv(os.path.join(self.hparams.image_path, "metadata", "ukb-rest.csv"))
            if self.hparams.task_name == 'sex': task_name = 'sex'
            elif self.hparams.task_name == 'age': task_name = 'age'
            else: raise ValueError('downstream task not supported')
           
            if task_name == 'sex':
                meta_task = meta_data[['subject_id', task_name]].dropna()
            else:
                meta_task = meta_data[['subject_id', task_name, 'sex']].dropna()
            
            for subject in subject_list:
                if int(subject) in meta_task['subject_id'].values:
                    target = meta_task[meta_task["subject_id"]==int(subject)][task_name].values[0]
                    if task_name == 'sex':
                        target = int(target)
                        
                    sex = meta_task[meta_task["subject_id"]==int(subject)]["sex"].values[0]
                    sex = int(sex)
                    final_dict[subject] = [sex, target]
            
            print('Load dataset UKB, {} subjects'.format(len(final_dict)))
        
        elif self.hparams.dataset_name == "HCPTASK":
            subject_list = [subj for subj in os.listdir(img_root)]

            if self.hparams.downstream_task_id == 5: task_name = 'classification'
            else: raise ValueError('downstream task not supported')

            state_to_label = {'EMOTION': 0, 'GAMBLING': 1, 'LANGUAGE': 2, 'MOTOR': 3, 'RELATIONAL': 4, 'SOCIAL': 5, 'WM': 6}

            for subject in subject_list:
                state = subject.split('_')[-2]
                state = state_to_label[state]

                sex = 0
                final_dict[subject] = [sex, state]
            
            print('Load dataset HCPTASK, {} subjects'.format(len(final_dict)))

        elif self.hparams.dataset_name == "MOVIE":
            subject_list = [subj for subj in os.listdir(img_root)]
            parent_dir = os.path.dirname(os.path.abspath(img_root))
            metadata_dir = os.path.join(parent_dir, 'metadata')
            participants_file = os.path.join(metadata_dir, 'participants.tsv')

            try:
                df = pd.read_csv(participants_file, sep='\t')
            except Exception as e:
                import ipdb; ipdb.set_trace()

            participant_id = df.iloc[:, 0]
            group = df.iloc[:, -1]
            participant_dict = pd.Series(group.values, index=participant_id).to_dict()

            if self.hparams.downstream_task_id == 5: task_name = 'classification'
            else: raise ValueError('downstream task not supported')

            for subject in subject_list:
                subject_id = subject.split('_')[0]
                if participant_dict[subject_id] == 'Control':
                    final_dict[subject] = [0, 0]
                else:
                    final_dict[subject] = [0, 1]
            
            print('Load dataset MOVIE, {} subjects'.format(len(final_dict)))
        
        if self.hparams.dataset_name == "TransDiag":
            subject_list = os.listdir(img_root)

            # diagnosis, clinical_variables
            if self.hparams.task_name == 'diagnosis': 
                csv_file = self.hparams.task_name + '.csv'
            else:
                csv_file = 'clinical_variables.csv'

            # import chardet
            # with open(os.path.join(self.hparams.image_path, "metadata", csv_file), 'rb') as file:
            #     result = chardet.detect(file.read())
            #     print(result)

            meta_data = pd.read_csv(os.path.join(self.hparams.image_path, "metadata", csv_file), encoding='ISO-8859-1')
            if self.hparams.task_name == 'diagnosis': 
                task_name = 'diagnosis'
                # label_name = 'Group'
                label_name = 'Diagnostic_Category_Code'
            else: task_name = 'clinical_variables'

            print('downstream_task_id = {}, task_name = {}'.format(self.hparams.downstream_task_id, task_name))
            target_counts = defaultdict(int)

            if task_name == 'diagnosis':
                meta_task = meta_data[['subjectkey', label_name, 'sex']].dropna()
            elif task_name == 'clinical_variables':
                label_name = self.hparams.task_name
                meta_task = meta_data[['subjectkey', label_name]].dropna()
            
            for subject in subject_list:
                subject_id = subject[:4] + '_' + subject[4:-5]
                if subject_id in meta_task['subjectkey'].values:
                    if task_name == 'diagnosis':
                        sex = meta_task[meta_task["subjectkey"]==subject_id]['sex'].values[0]
                        if sex == 'F': sex = 0
                        else: sex = 1

                        target = meta_task[meta_task["subjectkey"]==subject_id][label_name].values[0]
                        if label_name == 'Diagnostic_Category_Code':
                            if target in [0, 5, 7]: continue
                            elif target == 8: target = 0
                            elif target == 6: target = 5
                        elif label_name == 'Group':
                            if target == 'Patient': target = 1
                            else: target = 0
                    else:
                        sex = 0
                        target = meta_task[meta_task["subjectkey"]==subject_id][label_name].values[0]
                        if target == 'n/a':
                            import ipdb; ipdb.set_trace()
                            continue
                    
                    target_counts[target] += 1
                    # print('sex = {}, target = {}'.format(sex, target))
                    final_dict[subject] = [sex, target]
            
            for category, count in target_counts.items():
                print(f"Target {category}: {count}")
            print('Load dataset TransDiag, {} subjects'.format(len(final_dict)))

        return final_dict

    def _build_params(self, data_type, image_path):
        if data_type == 'voxel':
            Dataset = self.get_dataset()  # resolved per-dataset in the caller
            params = {
                    "root": image_path,
                    "img_size": self.hparams.img_size,
                    "sequence_length": self.hparams.sequence_length,
                    "contrastive": self.hparams.use_contrastive,
                    "contrastive_type": self.hparams.contrastive_type,
                    "mae": self.hparams.use_mae,
                    "stride_between_seq": self.hparams.stride_between_seq,
                    "stride_within_seq": self.hparams.stride_within_seq,
                    "with_voxel_norm": self.hparams.with_voxel_norm,
                    "downstream_task_id": self.hparams.downstream_task_id,
                    "task_name": self.hparams.task_name,
                    "shuffle_time_sequence": self.hparams.shuffle_time_sequence,
                    "label_scaling_method": self.hparams.label_scaling_method}
        elif data_type == 'roi':
            Dataset = ROIDataset
            params = {
                    "root": image_path,
                    "atlas_name": getattr(self.hparams, 'atlas_name', 'cc200'),
                    "sequence_length": getattr(self.hparams, 'sequence_length', None),
                    "stride": getattr(self.hparams, 'stride_within_seq', 1)}
        elif data_type == 'fc':
            Dataset = FCDataset
            params = {
                    "root": image_path,
                    "atlas_name": getattr(self.hparams, 'atlas_name', 'cc200'),
                    "fc_type": getattr(self.hparams, 'fc_type', 'correlation'),
                    "return_format": getattr(self.hparams, 'fc_return_format', 'matrix')}
        elif data_type == 'fc_bnt':
            Dataset = FCDataset
            params = {
                    "root": image_path,
                    "atlas_name": getattr(self.hparams, 'atlas_name', 'cc200'),
                    "fc_type": getattr(self.hparams, 'fc_type', 'correlation'),
                    "return_format": 'bnt'}
        elif data_type == 'fc_graph':
            Dataset = FCGraphDataset
            params = {
                    "root": image_path,
                    "atlas_name": getattr(self.hparams, 'atlas_name', 'cc200'),
                    "fc_type": getattr(self.hparams, 'fc_type', 'partial_correlation'),
                    "threshold": getattr(self.hparams, 'fc_threshold', None)}
        else:
            raise ValueError(f"Unknown data_type: {data_type}")
        return Dataset, params

    def _build_single_dataset(self, dataset_name, image_path, split_file_path, data_type):
        """Build train/val/test datasets for a single fMRI dataset."""
        # Resolve the right Dataset class for this dataset name (voxel) while
        # keeping the generic path for roi / fc variants.
        if data_type == 'voxel':
            Dataset = self.get_dataset(dataset_name)
            _, params = self._build_params(data_type, image_path)
        else:
            Dataset, params = self._build_params(data_type, image_path)

        subject_dict = self.make_subject_dict(dataset_name=dataset_name, image_path=image_path)

        if os.path.exists(split_file_path):
            train_names, val_names, test_names = self.load_split(split_file_path)
        else:
            train_names, val_names, test_names = self.determine_split_randomly(subject_dict, split_file_path)

        if self.hparams.bad_subj_path:
            bad_subjects = open(self.hparams.bad_subj_path, "r").readlines()
            for bad_subj in bad_subjects:
                bad_subj = bad_subj.strip()
                if bad_subj in list(subject_dict.keys()):
                    print(f'removing bad subject: {bad_subj}')
                    del subject_dict[bad_subj]

        if self.hparams.limit_training_samples:
            selected_num = int(self.hparams.limit_training_samples * len(train_names))
            train_names = np.random.choice(train_names, size=selected_num, replace=False, p=None)

        train_dict = {key: subject_dict[key] for key in train_names if key in subject_dict}
        val_dict = {key: subject_dict[key] for key in val_names if key in subject_dict}
        test_dict = {key: subject_dict[key] for key in test_names if key in subject_dict}

        train_ds = Dataset(**params, subject_dict=train_dict, use_augmentations=False, train=True)
        val_ds = Dataset(**params, subject_dict=val_dict, use_augmentations=False, train=False)
        test_ds = Dataset(**params, subject_dict=test_dict, use_augmentations=False, train=False)

        _rank0_print(f"[{dataset_name}] train subjects: {len(train_dict)}, val: {len(val_dict)}, test: {len(test_dict)}")
        return train_ds, val_ds, test_ds

    def setup(self, stage=None):
        if getattr(self, "_is_setup", False):
            return
        self._is_setup = True
        data_type = getattr(self.hparams, 'data_type', 'voxel')

        train_parts, val_parts, test_parts = [], [], []
        for name, path, split_path in zip(self.dataset_names, self.image_paths, self.split_file_paths):
            t, v, te = self._build_single_dataset(name, path, split_path, data_type)
            train_parts.append(t)
            val_parts.append(v)
            test_parts.append(te)

        if len(train_parts) == 1:
            self.train_dataset = train_parts[0]
            self.val_dataset = val_parts[0]
            self.test_dataset = test_parts[0]
        else:
            self.train_dataset = ConcatDataset(train_parts)
            self.val_dataset = ConcatDataset(val_parts)
            self.test_dataset = ConcatDataset(test_parts)
            # Propagate target_values so downstream scaler logic keeps working.
            # In pretraining the values are dummies (zeros); they are not used
            # for loss computation.
            target_values = np.concatenate(
                [p.target_values for p in train_parts if hasattr(p, 'target_values')],
                axis=0,
            ) if any(hasattr(p, 'target_values') for p in train_parts) else np.zeros((len(self.train_dataset), 1))
            self.train_dataset.target_values = target_values

        if hasattr(self.train_dataset, 'data'):
            _rank0_print("number of train samples:", len(self.train_dataset.data))
            _rank0_print("number of val samples:", len(self.val_dataset.data))
            _rank0_print("number of test samples:", len(self.test_dataset.data))
        else:
            _rank0_print("number of train samples:", len(self.train_dataset))
            _rank0_print("number of val samples:", len(self.val_dataset))
            _rank0_print("number of test samples:", len(self.test_dataset))

        # DistributedSampler is internally called in pl.Trainer
        def get_params(train):
            params = {
                "batch_size": self.hparams.batch_size if train else self.hparams.eval_batch_size,
                "num_workers": self.hparams.num_workers,
                "drop_last": True,
                "pin_memory": bool(self.hparams.pin_memory),
                "persistent_workers": (self.hparams.num_workers > 0) and (train and (self.hparams.strategy == 'ddp')),
                "shuffle": train,
            }
            if self.hparams.num_workers > 0 and self.hparams.prefetch_factor is not None:
                params["prefetch_factor"] = int(self.hparams.prefetch_factor)
            return params

        # Per-dataset sampling strategy for pretraining with multiple datasets
        self._train_sampler = None
        use_strategy = (
            self.hparams.pretraining
            and isinstance(self.train_dataset, ConcatDataset)
            and getattr(self.hparams, 'sampling_strategy', None) is not None
        )

        # Use GeometricDataLoader for graph datasets
        if data_type == 'fc_graph':
            self.train_loader = GeometricDataLoader(self.train_dataset, **get_params(train=True))
            self.val_loader = GeometricDataLoader(self.val_dataset, **get_params(train=False))
            self.test_loader = GeometricDataLoader(self.test_dataset, **get_params(train=False))
        elif use_strategy:
            self._train_sampler = build_multi_dataset_sampler(
                self.hparams.sampling_strategy,
                self.train_dataset,
                self.hparams,
            )
            train_params = get_params(train=True)
            train_params["shuffle"] = False
            train_params["sampler"] = self._train_sampler
            self.train_loader = DataLoader(self.train_dataset, **train_params)
            self.val_loader = DataLoader(self.val_dataset, **get_params(train=False))
            self.test_loader = DataLoader(self.test_dataset, **get_params(train=False))
        else:
            self.train_loader = DataLoader(self.train_dataset, **get_params(train=True))
            self.val_loader = DataLoader(self.val_dataset, **get_params(train=False))
            self.test_loader = DataLoader(self.test_dataset, **get_params(train=False))

    def train_dataloader(self):
        if self._train_sampler is not None:
            self._train_sampler.set_epoch(self.trainer.current_epoch)
        return self.train_loader

    def val_dataloader(self):
        return [self.val_loader, self.test_loader]

    def test_dataloader(self):
        return self.test_loader

    def predict_dataloader(self):
        return self.test_dataloader()

    @classmethod
    def add_data_specific_args(cls, parent_parser: ArgumentParser, **kwargs) -> ArgumentParser:
        parser = ArgumentParser(parents=[parent_parser], add_help=True, formatter_class=ArgumentDefaultsHelpFormatter)
        group = parser.add_argument_group("DataModule arguments")
        group.add_argument("--dataset_split_num", type=int, default=1)
        group.add_argument("--label_scaling_method", default="standardization", choices=["minmax","standardization"], help="label normalization strategy for a regression task (mean and std are automatically calculated using train set)")
        group.add_argument("--image_path", default=None,
                          help="path to image datasets preprocessed for SwiFT. In --pretraining mode with multiple --dataset_name entries, pass a comma-separated list with one root per dataset.")
        group.add_argument("--bad_subj_path", default=None, help="path to txt file that contains subjects with bad fMRI quality")
        group.add_argument("--train_split", default=0.9, type=float)
        group.add_argument("--val_split", default=0.1, type=float)
        group.add_argument("--batch_size", type=int, default=4)
        group.add_argument("--eval_batch_size", type=int, default=8)
        group.add_argument("--img_size", nargs="+", default=[96, 96, 96, 20], type=int, help="image size (adjust the fourth dimension according to your --sequence_length argument)")
        group.add_argument("--sequence_length", type=int, default=20)
        group.add_argument("--stride_between_seq", type=int, default=1, help="skip some fMRI volumes between fMRI sub-sequences")
        group.add_argument("--stride_within_seq", type=int, default=1, help="skip some fMRI volumes within fMRI sub-sequences")
        group.add_argument("--num_workers", type=int, default=8)
        group.add_argument("--prefetch_factor", type=int, default=None,
                          help="DataLoader prefetch_factor (per-worker prefetched batches). Requires num_workers > 0.")
        group.add_argument("--pin_memory", type=str2bool, default=False,
                          help="Pin host memory in DataLoaders (faster H2D copy when training on GPU).")
        group.add_argument("--with_voxel_norm", type=str2bool, default=False)
        group.add_argument("--shuffle_time_sequence", action='store_true')
        group.add_argument("--limit_training_samples", type=float, default=None, help="use if you want to limit training samples")

        # Multi-dataset sampling strategy (pretraining only)
        group.add_argument("--sampling_strategy", type=str, default=None,
                          choices=list(SAMPLING_STRATEGIES.keys()),
                          help="Multi-dataset sampling strategy for pretraining. "
                               "'uniform_subsample': cap each dataset to --max_samples_per_dataset per epoch. "
                               "'loss_weighted': resample proportional to per-dataset loss.")
        group.add_argument("--max_samples_per_dataset", type=int, default=500,
                          help="Max samples per dataset per epoch (used by uniform_subsample and as base for loss_weighted)")

        # New arguments for ROI/FC data
        group.add_argument("--data_type", type=str, default="voxel", choices=["voxel", "roi", "fc", "fc_graph", "fc_bnt"],
                          help="type of data: voxel (4D fMRI), roi (2D time series), fc (2D connectivity matrix), fc_graph (graph for BrainGNN), fc_bnt (FC for BrainNetworkTransformer)")
        group.add_argument("--atlas_name", type=str, default="cc200",
                          help="brain atlas name (e.g., cc200, aal, schaefer)")
        group.add_argument("--fc_type", type=str, default="correlation", choices=["correlation", "partial_correlation"],
                          help="type of functional connectivity")
        group.add_argument("--fc_threshold", type=float, default=None,
                          help="threshold for FC edge pruning (None = keep all edges)")

        return parser
