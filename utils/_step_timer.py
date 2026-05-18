"""Tiny step-time profiling callback. Enable via env NEUROSTORM_STEP_PROFILE=1.

Times each training step using CUDA events:
  - fetch:    time between the previous step's optim end and current step's start (≈ dataloader wait)
  - fwd:      training_step forward call
  - bwd:      backward (between forward end and optimizer step)
  - opt+sync: optimizer.step + DDP allreduce/sync (between bwd end and after-batch end)
"""
from __future__ import annotations

import os
import time
import torch
import pytorch_lightning as pl


class StepTimer(pl.Callback):
    def __init__(self, max_steps: int = 30, warmup: int = 5):
        self.max_steps = max_steps
        self.warmup = warmup
        self._records = []
        self._batch_start = None
        self._fwd_end = None
        self._bwd_end = None
        self._prev_batch_end = None

    def _now(self) -> float:
        torch.cuda.synchronize()
        return time.perf_counter()

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if batch_idx >= self.max_steps:
            trainer.should_stop = True
            return
        t = self._now()
        self._batch_start = t
        if self._prev_batch_end is not None:
            fetch = t - self._prev_batch_end
        else:
            fetch = 0.0
        self._fetch = fetch

    def on_before_backward(self, trainer, pl_module, loss):
        self._fwd_end = self._now()

    def on_before_optimizer_step(self, trainer, pl_module, optimizer, opt_idx=0):
        self._bwd_end = self._now()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        end = self._now()
        fwd = (self._fwd_end - self._batch_start) if self._fwd_end else 0.0
        bwd = (self._bwd_end - self._fwd_end) if (self._bwd_end and self._fwd_end) else 0.0
        opt = (end - self._bwd_end) if self._bwd_end else 0.0
        total = end - self._batch_start
        self._records.append((batch_idx, self._fetch, fwd, bwd, opt, total))
        rank = trainer.global_rank
        if batch_idx >= self.warmup:
            print(
                f"[step-timer rank{rank}] step={batch_idx:3d} "
                f"fetch={self._fetch*1000:7.1f}ms "
                f"fwd={fwd*1000:7.1f}ms "
                f"bwd={bwd*1000:7.1f}ms "
                f"opt+sync={opt*1000:7.1f}ms "
                f"total={total*1000:7.1f}ms",
                flush=True,
            )
        self._prev_batch_end = end
        if batch_idx + 1 >= self.max_steps:
            self._summary(trainer)
            trainer.should_stop = True

    def _summary(self, trainer):
        rank = trainer.global_rank
        recs = self._records[self.warmup:]
        if not recs:
            return
        n = len(recs)
        s_fetch = sum(r[1] for r in recs) / n
        s_fwd = sum(r[2] for r in recs) / n
        s_bwd = sum(r[3] for r in recs) / n
        s_opt = sum(r[4] for r in recs) / n
        s_tot = sum(r[5] for r in recs) / n
        print(
            f"[step-timer rank{rank}] SUMMARY over {n} steps (after {self.warmup} warmup): "
            f"fetch={s_fetch*1000:.1f}ms fwd={s_fwd*1000:.1f}ms bwd={s_bwd*1000:.1f}ms "
            f"opt+sync={s_opt*1000:.1f}ms total={s_tot*1000:.1f}ms",
            flush=True,
        )


def maybe_attach(callbacks):
    if os.environ.get("NEUROSTORM_STEP_PROFILE", "") in ("1", "true", "True"):
        callbacks.append(StepTimer())
    return callbacks
