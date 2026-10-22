# Brain-JEPA

This package includes the Brain-JEPA encoder, predictor, gradient positioning
table, required tensor and masking utilities, and the regression adapter used
in our experiments. The files under `source/Brain-JEPA/` retain their original
contents and directory layout.

```python
from argparse import Namespace
from models.load_model import load_model

model = load_model("brainjepa", Namespace(
    downstream_task_type="regression", num_classes=1,
    brainjepa_checkpoint_path="/path/to/brainjepa_pretrained.pth.tar",
))
```

The adapter requires NumPy and PyTorch 2.1 or newer. It loads the original
pretrained checkpoint's `target_encoder` strictly. Omit
`brainjepa_checkpoint_path` to initialize from scratch. Checkpoints are supplied
separately. The task head is initialized for downstream fine-tuning; loading
the pretrained encoder alone does not produce a trained task model. The older
`BrainJEPAP01` interface remains available and checks its original epoch 299.

Inputs have shape `[batch, 450, 160]` or `[batch, 1, 450, 160]`: 450 ordered ROI
signals and 160 sampled time points. Outputs have shape `[batch, num_outputs]`,
with one output for binary classification or regression. The bundled gradient table has
shape `[450, 30]` and must use the same ROI ordering as the input.

The adapter uses PyTorch scaled dot-product attention without requiring
FlashAttention. The original attention calculation remains available when
requesting attention weights. The model can be imported directly through
`models.brainjepa` or the shared `load_model("brainjepa", hparams)` registry.

The shared runner uses `--model brainjepa --data_type roi`. Store `.npy` files
of shape `[450, T]` at `roi/brainjepa450/<subject>_brainjepa450.npy` under the
dataset root, alongside the existing metadata and subject split files. ROI
ordering must match the bundled gradient table; a 400-ROI atlas is insufficient.
The dataset normalizes the whole scan using a single mean and sample standard
deviation, then selects 160 uniformly spaced points from a 480-frame crop.
Training crops are random; validation and test crops are centered and repeatable.
`--brainjepa_window_length` changes the crop length explicitly.

Use the common `--test_only --test_ckpt_path <framework.ckpt>` or
`--resume_ckpt_path <framework.ckpt>` options for an existing framework
checkpoint. These restore the model without needing the original initializer.

Reference: Dong et al., *Brain-JEPA: Brain Dynamics Foundation Model with
Gradient Positioning and Spatiotemporal Masking*, NeurIPS 2024.
