"""Ordered ROI input for Brain-JEPA's 450-ROI, 160-sample encoder."""

import random

import numpy as np
import torch

from .roi_datasets import ROIDataset


class BrainJEPADataset(ROIDataset):
    def __init__(self, *args, window_length=480, **kwargs):
        super().__init__(*args, **kwargs)
        if window_length < 160:
            raise ValueError("Brain-JEPA window_length must be at least 160")
        self.window_length = window_length

    def __getitem__(self, idx):
        subject_id, roi_file, target, sex = self.data[idx]
        array = np.load(roi_file, allow_pickle=False)
        if array.ndim != 2 or array.shape[0] != 450 or not np.isfinite(array).all():
            raise ValueError(f"Expected finite [450, T] ordered ROI signals: {roi_file}")
        if array.shape[1] < self.window_length:
            raise ValueError(f"Brain-JEPA requires at least {self.window_length} frames: {roi_file}")
        signals = torch.as_tensor(array.astype(np.float32, copy=False))
        mean, std = signals.mean(), signals.std()
        if not torch.isfinite(std) or std <= 0:
            raise ValueError(f"Brain-JEPA requires nonconstant ROI signals: {roi_file}")
        signals = (signals - mean) / std
        available = signals.shape[1] - self.window_length
        start = random.uniform(0, available) if self.train else available / 2
        indices = torch.linspace(start, start + self.window_length - 1, 160).long()
        return {
            "roi_sequence": signals.index_select(1, indices).unsqueeze(0),
            "subject_name": subject_id,
            "target": target,
            "sex": sex,
        }
