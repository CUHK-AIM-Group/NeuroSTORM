import os
import pytorch_lightning as pl
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset, ConcatDataset, Sampler
from torch.utils.data.distributed import DistributedSampler
from torch_geometric.data import DataLoader as GeometricDataLoader
from .fmri_datasets import HCP1200, ABCD, UKB, Cobre, ADHD200, UCLA, HCPEP, HCPTASK, GOD, MOVIE, TransDiag, ABIDE
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
        prev = self.epoch
        self.epoch = epoch
        try:
            rank = self.rank
        except Exception:
            rank = 0
        print(f'[diag][rank{rank}] SAMPLER_SET_EPOCH cls={type(self).__name__} '
              f'prev={prev} new={epoch}', flush=True)

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
        n_raw = len(indices)
        if self.world_size > 1:
            # Pad indices to a multiple of world_size so every rank iterates the
            # same number of batches. This avoids DDP hangs from uneven loaders.
            rem = len(indices) % self.world_size
            if rem != 0:
                indices = list(indices) + list(indices[: self.world_size - rem])
            indices = indices[self.rank :: self.world_size]
        print(f'[diag][rank{self.rank}] SAMPLER_ITER cls={type(self).__name__} '
              f'epoch={self.epoch} world_size={self.world_size} '
              f'n_raw={n_raw} n_after_shard={len(indices)}', flush=True)
        return iter(indices)

    def __len__(self):
        total = len(self._compute_indices())
        if self.world_size > 1:
            rem = total % self.world_size
            if rem != 0:
                total += self.world_size - rem
            return total // self.world_size
        return total


class _SubjectGroupSampler(MultiDatasetSampler):
    """Base for subject-grouped reweighting strategies.

    Operates on subjects within each sub-dataset: scores all subjects, then either
    picks the top-k (hard) or weight-samples k of them (soft). All clips of the
    chosen subjects are included in that epoch's indices.

    Subclasses implement `_score_subjects(rng, ds_idx, all_subjs)` which returns
    a 1D weight array (one weight per subject in `all_subjs`).
    """

    def __init__(self, concat_dataset, topk=10000, mode='hard', seed=0, fixed=False):
        """
        Args:
            fixed: when True (val/test), compute indices ONCE at construction
                time and reuse forever — Lightning needs eval loaders to have
                stable length across epochs. When False (train), recompute each
                epoch; per-epoch length naturally varies because subjects have
                different clip counts.
        """
        super().__init__(concat_dataset, seed=seed)
        if mode not in ('hard', 'soft'):
            raise ValueError(f"mode must be 'hard' or 'soft', got {mode}")
        self.topk = topk
        self.mode = mode
        self._subject_groups = self._build_subject_groups()
        # Optional per-dataset, per-subject losses for the loss-based variants.
        # List[dict[subj_idx -> mean_loss]] of length num_datasets.
        self._subject_losses = []
        self.fixed = fixed
        # Indices cache for `fixed=True` (val/test). Set on first compute.
        self._frozen_indices = None
        if fixed:
            # Eagerly lock indices at init so DDP ranks agree on length.
            self._frozen_indices = self._sample_indices()

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

    def update_subject_losses(self, per_dataset_subject_losses):
        """Optional hook for loss-based variants.

        Args:
            per_dataset_subject_losses: list of dict {subj_idx -> mean_loss}
                aligned with the order of self.concat_dataset.datasets.
        """
        self._subject_losses = per_dataset_subject_losses

    def _score_subjects(self, rng, ds_idx, all_subjs):
        raise NotImplementedError

    def _sample_indices(self):
        """Run the raw subject-level sampling for the current epoch."""
        rng = np.random.RandomState(self.seed + self.epoch)
        indices = []
        for ds_idx, subj_to_indices in enumerate(self._subject_groups):
            all_subjs = list(subj_to_indices.keys())
            weights = self._score_subjects(rng, ds_idx, all_subjs)
            n = min(len(all_subjs), self.topk)
            if self.mode == 'hard':
                # Take top-k by weight (descending); ties broken stably.
                if n >= len(all_subjs):
                    chosen = np.arange(len(all_subjs))
                else:
                    chosen = np.argpartition(-weights, n - 1)[:n]
            else:
                # Soft: weighted sampling without replacement.
                w = np.clip(weights, 1e-8, None).astype(np.float64)
                probs = w / w.sum()
                chosen = rng.choice(len(all_subjs), size=n, replace=False, p=probs)
            for i in chosen:
                indices.extend(subj_to_indices[all_subjs[i]])
        rng.shuffle(indices)
        return indices

    def _compute_indices(self):
        # Fixed mode (val/test): always return the indices computed at __init__.
        if self.fixed:
            return self._frozen_indices
        # Train mode: re-sample each epoch. Length varies naturally because
        # different topk subjects have different clip counts.
        return self._sample_indices()


class _RandomScoreMixin:
    """Score subjects with i.i.d. Uniform[0, 2.0] weights each epoch."""
    def _score_subjects(self, rng, ds_idx, all_subjs):
        return rng.uniform(0.0, 2.0, size=len(all_subjs))


class _LossScoreMixin:
    """Score subjects by their last-epoch mean loss; falls back to random
    on the first epoch (or for unseen subjects)."""
    def _score_subjects(self, rng, ds_idx, all_subjs):
        if not self._subject_losses or ds_idx >= len(self._subject_losses) or not self._subject_losses[ds_idx]:
            return rng.uniform(0.0, 2.0, size=len(all_subjs))
        losses = self._subject_losses[ds_idx]
        # Random fill for subjects with no loss yet, so they get sampled in.
        rand_fill = rng.uniform(0.0, 2.0, size=len(all_subjs))
        weights = np.array([losses.get(s, rand_fill[i]) for i, s in enumerate(all_subjs)],
                           dtype=np.float64)
        return weights


class HardRandomReweighting(_RandomScoreMixin, _SubjectGroupSampler):
    """Hard random: every epoch, draw uniform random weights ∈ [0, 2.0] for ALL
    subjects in each dataset, then keep the top-k. Selected subjects differ each
    epoch but the count stays at `topk`."""
    def __init__(self, concat_dataset, topk=10000, seed=0, fixed=False):
        super().__init__(concat_dataset, topk=topk, mode='hard', seed=seed, fixed=fixed)


class HardLossReweighting(_LossScoreMixin, _SubjectGroupSampler):
    """Hard loss: keep the top-k highest-loss subjects per dataset. First epoch
    falls back to random (no loss info yet). Call `update_subject_losses(...)`
    at end of each epoch."""
    def __init__(self, concat_dataset, topk=10000, seed=0, fixed=False):
        super().__init__(concat_dataset, topk=topk, mode='hard', seed=seed, fixed=fixed)


class SoftRandomReweighting(_RandomScoreMixin, _SubjectGroupSampler):
    """Soft random: weighted sampling (without replacement) of `topk` subjects per
    dataset using Uniform[0, 2.0] weights."""
    def __init__(self, concat_dataset, topk=10000, seed=0, fixed=False):
        super().__init__(concat_dataset, topk=topk, mode='soft', seed=seed, fixed=fixed)


class SoftLossReweighting(_LossScoreMixin, _SubjectGroupSampler):
    """Soft loss: weighted sampling (without replacement) of `topk` subjects per
    dataset where each subject's weight is its last-epoch mean loss."""
    def __init__(self, concat_dataset, topk=10000, seed=0, fixed=False):
        super().__init__(concat_dataset, topk=topk, mode='soft', seed=seed, fixed=fixed)


# --- Add new strategies here ---
# class YourNewStrategy(MultiDatasetSampler):
#     """Strategy N: Description."""
#     def __init__(self, concat_dataset, ...):
#         super().__init__(concat_dataset, seed=seed)
#         ...
#     def _compute_indices(self):
#         ...


REWEIGHTING_STRATEGIES = {
    "hard_random": HardRandomReweighting,
    "hard_loss": HardLossReweighting,
    "soft_random": SoftRandomReweighting,
    "soft_loss": SoftLossReweighting,
    # Register new strategies here
}


def build_multi_dataset_sampler(strategy_name, concat_dataset, hparams, mode='train'):
    """Factory function to build a multi-dataset sampler from hparams.

    For mode in {'val', 'test'}, topk is scaled from the train value by
    (split_ratio / train_split) so eval sees the same subject ratio.
    """
    seed = getattr(hparams, 'seed', 0)
    train_topk = getattr(hparams, 'topk', 10000)
    train_split = getattr(hparams, 'train_split', 0.9)
    val_split = getattr(hparams, 'val_split', 0.1)
    test_split = max(0.0, 1.0 - train_split - val_split) or val_split

    if mode == 'train':
        topk = train_topk
        epoch_seed = seed
    elif mode == 'val':
        topk = max(1, int(round(train_topk * (val_split / max(train_split, 1e-8)))))
        epoch_seed = seed + 10_000
    elif mode == 'test':
        topk = max(1, int(round(train_topk * (test_split / max(train_split, 1e-8)))))
        epoch_seed = seed + 20_000
    else:
        raise ValueError(f"Unknown mode: {mode}")

    if strategy_name not in REWEIGHTING_STRATEGIES:
        raise ValueError(f"Unknown reweighting strategy: {strategy_name}. "
                         f"Available: {list(REWEIGHTING_STRATEGIES.keys())}")

    cls = REWEIGHTING_STRATEGIES[strategy_name]
    # Eval modes always use random scoring (no loss feedback during eval).
    if mode != 'train' and strategy_name.endswith('_loss'):
        cls = REWEIGHTING_STRATEGIES[strategy_name.replace('_loss', '_random')]
    # Val / test: lock the sampled subjects across epochs. Train: re-sample each epoch.
    return cls(concat_dataset, topk=topk, seed=epoch_seed, fixed=(mode != 'train'))


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

        model = getattr(self.hparams, 'model', None)
        if model in ('brainjepa', 'band', 'brainlm', 'tff'):
            expected = 'roi' if model in ('brainjepa', 'brainlm') else 'voxel'
            if getattr(self.hparams, 'data_type', 'voxel') != expected:
                raise ValueError(f"{model} requires --data_type {expected}")
            if self.hparams.pretraining:
                raise ValueError(f"{model} does not use the NeuroSTORM pretraining objective")

        self.dataset_names, self.image_paths = self._parse_dataset_specs()
        if len(self.dataset_names) > 1 and not self.hparams.pretraining:
            raise ValueError(
                "Multiple datasets via --dataset_name are only supported in --pretraining mode. "
                f"Got dataset_name={self.hparams.dataset_name} without --pretraining."
            )

        split_root = getattr(self.hparams, 'split_root', './nas/splits')
        if self.hparams.pretraining:
            self.split_file_paths = []
            for name in self.dataset_names:
                split_dir_path = os.path.join(split_root, name)
                self.split_file_paths.append(
                    os.path.join(split_dir_path, f"split_fixed_{self.hparams.dataset_split_num}.txt")
                )
            self.split_file_path = self.split_file_paths[0]
        else:
            split_dir_path = os.path.join(split_root, self.dataset_names[0])
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
        elif name == 'ABIDE':
            return ABIDE
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
                final_dict = {
                    subj: [0, 0]
                    for subj in sorted(os.listdir(img_root))
                    if os.path.isfile(os.path.join(img_root, subj, 'data.pt'))
                }
                _rank0_print('Load dataset {} for pretraining, {} subjects'.format(
                    self.hparams.dataset_name, len(final_dict)))
                return final_dict

            return self._make_subject_dict_with_labels()
        finally:
            if swapped:
                self.hparams.dataset_name = prev_name
                self.hparams.image_path = prev_path

    def _input_subject_names(self):
        if getattr(self.hparams, 'model', None) in ('brainjepa', 'brainlm'):
            atlas = self.hparams.atlas_name
            suffix = f"_{atlas}.npy"
            roi_root = os.path.join(self.hparams.image_path, 'roi', atlas)
            return sorted(name[:-len(suffix)] for name in os.listdir(roi_root)
                          if name.endswith(suffix))
        return os.listdir(os.path.join(self.hparams.image_path, 'img'))

    def _make_subject_dict_with_labels(self):
        img_root = os.path.join(self.hparams.image_path, 'img')
        final_dict = dict()

        if self.hparams.dataset_name == "HCP1200":
            subject_list = self._input_subject_names()
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
            
        elif self.hparams.dataset_name in ("HCPA", "HCPD"):
            csv_name = "hcpa-rest.csv" if self.hparams.dataset_name == "HCPA" else "hcpd-rest.csv"
            subject_list = self._input_subject_names()

            meta_data = pd.read_csv(os.path.join(self.hparams.image_path, "metadata", csv_name))
            if self.hparams.task_name == 'sex':
                task_name = 'sex'
            elif self.hparams.task_name == 'age':
                task_name = 'interview_age'
            else:
                raise ValueError('downstream task not supported')

            cols = ['subject_id', task_name] if task_name == 'sex' else ['subject_id', task_name, 'sex']
            meta_task = meta_data[cols].dropna()

            for subject in subject_list:
                if subject in meta_task['subject_id'].values:
                    target = meta_task[meta_task["subject_id"] == subject][task_name].values[0]
                    if task_name == 'sex':
                        target = 1 if str(target).upper().startswith("M") else 0
                        sex = target
                    else:
                        # interview_age is in months -> convert to years
                        target = float(target) / 12.0
                        sex_raw = meta_task[meta_task["subject_id"] == subject]["sex"].values[0]
                        sex = 1 if str(sex_raw).upper().startswith("M") else 0
                    final_dict[subject] = [sex, target]

            print(f'Load dataset {self.hparams.dataset_name}, {len(final_dict)} subjects')

        elif self.hparams.dataset_name == "ABCD":
            subject_list = self._input_subject_names()
            
            meta_data = pd.read_csv(os.path.join(self.hparams.image_path, "metadata", "abcd-rest.csv"))
            if self.hparams.task_name == 'sex': task_name = 'sex'
            elif self.hparams.task_name == 'age': task_name = 'interview_age'
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
                    elif task_name == 'interview_age':
                        # ABCD interview_age is in months -> convert to years
                        target = float(target) / 12.0
                    sex = meta_task[meta_task["subjectkey"]==subject]["sex"].values[0]
                    sex = 1 if sex == "M" else 0
                    final_dict[subject] = [sex, target]
            
            print('Load dataset ABCD, {} subjects'.format(len(final_dict)))
        
        elif self.hparams.dataset_name == "Cobre":
            subject_list = self._input_subject_names()
            
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
            subject_list = self._input_subject_names()
            
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
            subject_list = self._input_subject_names()
            
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
            subject_list = self._input_subject_names()
            
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
            subject_list = self._input_subject_names()
            
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
            subject_list = self._input_subject_names()

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
            subject_list = self._input_subject_names()

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
            subject_list = self._input_subject_names()
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

        elif self.hparams.dataset_name == "ABIDE":
            subject_list = self._input_subject_names()

            meta_data = pd.read_csv(os.path.join(self.hparams.image_path, "metadata", "phenotypic.csv"))

            if self.hparams.task_name == 'sex':
                task_name = 'sex'
            elif self.hparams.task_name == 'age':
                task_name = 'age'
            elif self.hparams.task_name == 'diagnosis':
                task_name = 'dx_group'
            else:
                raise ValueError('downstream task not supported')

            cols = ['subj', task_name] if task_name == 'sex' else ['subj', task_name, 'sex']
            meta_task = meta_data[cols].dropna()
            meta_task = meta_task[meta_task['subj'].astype(str).str.len() > 0]
            meta_task = meta_task.set_index('subj')

            for subject in subject_list:
                if subject not in meta_task.index:
                    continue
                row = meta_task.loc[subject]
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[0]

                sex_raw = row['sex']
                # phenotypic encodes sex as 1=M, 2=F -> binary 1/0
                sex = 1 if int(float(sex_raw)) == 1 else 0

                if task_name == 'sex':
                    target = sex
                elif task_name == 'age':
                    target = float(row['age'])
                elif task_name == 'dx_group':
                    # 1=ASD, 2=TC -> 1=patient, 0=control
                    target = 1 if int(float(row['dx_group'])) == 1 else 0

                final_dict[subject] = [sex, target]

            print('Load dataset ABIDE, {} subjects'.format(len(final_dict)))

        if self.hparams.dataset_name == "TransDiag":
            subject_list = self._input_subject_names()

            # diagnosis, clinical_variables
            if self.hparams.task_name == 'diagnosis': 
                csv_candidates = ['transdiag_diagnosis.csv', 'diagnosis.csv']
                csv_file = next(
                    (name for name in csv_candidates
                     if os.path.exists(os.path.join(self.hparams.image_path, "metadata", name))),
                    'diagnosis.csv',
                )
            else:
                csv_file = 'clinical_variables.csv'

            # import chardet
            # with open(os.path.join(self.hparams.image_path, "metadata", csv_file), 'rb') as file:
            #     result = chardet.detect(file.read())
            #     print(result)

            meta_data = pd.read_csv(os.path.join(self.hparams.image_path, "metadata", csv_file), encoding='ISO-8859-1')
            if self.hparams.task_name == 'diagnosis': 
                task_name = 'diagnosis'
                if 'Diagnostic_Category_Code' in meta_data.columns:
                    label_name = 'Diagnostic_Category_Code'
                elif 'Group' in meta_data.columns:
                    label_name = 'Group'
                else:
                    raise ValueError(
                        "TransDiag diagnosis metadata must contain either "
                        "'Diagnostic_Category_Code' or 'Group'."
                    )
            else: task_name = 'clinical_variables'

            print('downstream_task_id = {}, task_name = {}'.format(self.hparams.downstream_task_id, task_name))
            target_counts = defaultdict(int)

            if task_name == 'diagnosis':
                meta_task = meta_data[['subjectkey', label_name, 'sex']].dropna()
            elif task_name == 'clinical_variables':
                label_name = self.hparams.task_name
                meta_task = meta_data[['subjectkey', label_name]].dropna()
            
            for subject in subject_list:
                raw_subject = subject[4:] if subject.startswith('sub-') else subject
                if raw_subject.startswith('NDARINV'):
                    subject_id = raw_subject[:4] + '_' + raw_subject[4:]
                else:
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
            if getattr(self.hparams, 'model', None) == 'brainjepa':
                from .brainjepa_dataset import BrainJEPADataset
                Dataset = BrainJEPADataset
                params['window_length'] = getattr(self.hparams, 'brainjepa_window_length', 480)
            elif getattr(self.hparams, 'model', None) == 'brainlm':
                from .brainlm_dataset import BrainLMDataset
                Dataset = BrainLMDataset
                params['num_rois'] = getattr(self.hparams, 'brainlm_num_rois', 424)
                params['stats_path'] = getattr(self.hparams, 'brainlm_stats_path', None)
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
                "drop_last": train if getattr(self.hparams, 'model', None) in ('brainjepa', 'band', 'brainlm', 'tff') else True,
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
            and getattr(self.hparams, 'reweighting_strategy', None) is not None
        )

        # Use GeometricDataLoader for graph datasets
        if data_type == 'fc_graph':
            self.train_loader = GeometricDataLoader(self.train_dataset, **get_params(train=True))
            self.val_loader = GeometricDataLoader(self.val_dataset, **get_params(train=False))
            self.test_loader = GeometricDataLoader(self.test_dataset, **get_params(train=False))
        elif use_strategy:
            self._train_sampler = build_multi_dataset_sampler(
                self.hparams.reweighting_strategy,
                self.train_dataset,
                self.hparams,
                mode='train',
            )
            train_params = get_params(train=True)
            train_params["shuffle"] = False
            train_params["sampler"] = self._train_sampler
            self.train_loader = DataLoader(self.train_dataset, **train_params)
            self._eval_params = get_params(train=False)
            self._use_distributed_eval = True
            self.val_loader = None
            self.test_loader = None
        else:
            self.train_loader = DataLoader(self.train_dataset, **get_params(train=True))
            self.val_loader = DataLoader(self.val_dataset, **get_params(train=False))
            self.test_loader = DataLoader(self.test_dataset, **get_params(train=False))

    def _build_eval_loader(self, dataset, mode):
        params = dict(self._eval_params)
        # If user enabled a multi-dataset sampling strategy and this is a
        # ConcatDataset, mirror the train-time subsampling on val/test by
        # building an eval-mode strategy sampler. The custom sampler shards
        # itself across DDP ranks, so we don't also wrap it in DistributedSampler.
        strategy_name = getattr(self.hparams, 'reweighting_strategy', None)
        if strategy_name is not None and isinstance(dataset, ConcatDataset):
            sampler = build_multi_dataset_sampler(
                strategy_name, dataset, self.hparams, mode=mode,
            )
            params["sampler"] = sampler
            params["shuffle"] = False
        elif self._use_distributed_eval and torch.distributed.is_available() and torch.distributed.is_initialized():
            params["sampler"] = DistributedSampler(dataset, shuffle=False)
            params["shuffle"] = False
        return DataLoader(dataset, **params)

    def train_dataloader(self):
        if self._train_sampler is not None:
            self._train_sampler.set_epoch(self.trainer.current_epoch)
        return self.train_loader

    def val_dataloader(self):
        if self.val_loader is None:
            self.val_loader = self._build_eval_loader(self.val_dataset, mode='val')
            self.test_loader = self._build_eval_loader(self.test_dataset, mode='test')
        return [self.val_loader, self.test_loader]

    def test_dataloader(self):
        if self.test_loader is None:
            self.test_loader = self._build_eval_loader(self.test_dataset, mode='test')
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

        # Multi-dataset reweighting strategy (pretraining only)
        group.add_argument("--reweighting_strategy", type=str, default=None,
                          choices=list(REWEIGHTING_STRATEGIES.keys()),
                          help="Multi-dataset reweighting strategy for pretraining. "
                               "Two axes: hard vs soft × random vs loss. "
                               "'hard_random' / 'hard_loss': score every subject (uniform random or last-epoch loss), "
                               "keep top-k. 'soft_random' / 'soft_loss': weighted sampling of k subjects per dataset.")
        group.add_argument("--topk", type=int, default=10000,
                          help="Per-dataset, per-epoch budget of subjects (top-k for 'hard', "
                               "weighted-sample size for 'soft'). Not a critical hyperparameter; "
                               "may differ across runs without affecting checkpoint compatibility.")

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
