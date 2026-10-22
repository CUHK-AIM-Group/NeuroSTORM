"""Fixed ROI windows and scan-wise robust scaling for supervised BrainLM."""

import random

import numpy as np
import torch

from .roi_datasets import ROIDataset


class BrainLMDataset(ROIDataset):
    def __init__(self, *args, num_rois=424, stats_path=None, **kwargs):
        super().__init__(*args, **kwargs)
        if self.sequence_length is None or self.sequence_length < 1 or num_rois < 1 or self.stride < 1:
            raise ValueError("BrainLM requires positive ROI count, sequence length and stride")
        self.num_rois = num_rois
        self.stats = None
        if stats_path is not None:
            with np.load(stats_path, allow_pickle=False) as values:
                median = np.asarray(values["median"], dtype=np.float32)
                iqr = np.asarray(values["iqr"], dtype=np.float32)
                order = np.asarray(values["roi_order"]) if "roi_order" in values else np.arange(num_rois)
            if median.shape != (num_rois,) or iqr.shape != (num_rois,) or not np.isfinite(median).all() or not np.isfinite(iqr).all() or (iqr <= 0).any():
                raise ValueError("BrainLM statistics require finite median and positive IQR arrays with one value per ROI")
            if order.shape != (num_rois,) or order.dtype.kind not in "iu" or not np.array_equal(np.sort(order), np.arange(num_rois)):
                raise ValueError("BrainLM roi_order must be a permutation of all input ROIs")
            self.stats = (median, iqr, order)

    def __getitem__(self, idx):
        subject_id, path, target, sex = self.data[idx]
        array = np.load(path, allow_pickle=False).astype(np.float32)
        duration = 1 + (self.sequence_length - 1) * self.stride
        if array.ndim != 2 or array.shape[0] != self.num_rois or array.shape[1] < duration or not np.isfinite(array).all():
            raise ValueError(f"Expected finite [{self.num_rois},T] ROI signals with at least {duration} frames: {path}")
        array = array - array.mean(axis=1, keepdims=True)
        if self.stats is None:
            median = np.median(array, axis=1)
            q25, q75 = np.percentile(array, [25, 75], axis=1)
            iqr = np.maximum(q75 - q25, 1e-6).astype(np.float32)
            order = np.arange(self.num_rois)
        else:
            median, iqr, order = self.stats
        array = ((array - median[:, None]) / iqr[:, None])[order] / np.float32(5.6430855)
        available = array.shape[1] - duration
        start = random.randint(0, available) if self.train else available // 2
        window = array[:, start:start + duration:self.stride].copy()
        return {"roi_sequence": torch.from_numpy(window).unsqueeze(0),
                "subject_name": subject_id, "target": target, "sex": sex}

