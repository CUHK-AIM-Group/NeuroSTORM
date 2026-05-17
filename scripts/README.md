# NeuroSTORM Scripts

Unified experiment runner with YAML-based configuration.

## Quick Start

```bash
# Pretrain NeuroSTORM with MAE on HCP1200
bash scripts/run_experiment.sh --model neurostorm --dataset hcp1200 --mode pretrain

# Joint pretraining on multiple datasets (pretrain only)
bash scripts/run_experiment.sh --model neurostorm --dataset hcp1200,hcpa,hcpd,abcd,ukb --mode pretrain

# Finetune on downstream task
bash scripts/run_experiment.sh --model neurostorm --dataset hcp1200 --task task1 --mode finetune

# Train baseline from scratch
bash scripts/run_experiment.sh --model braingnn --dataset adhd200 --task task3 --mode train_scratch

# Dry run (show command without executing)
bash scripts/run_experiment.sh --model neurostorm --dataset hcp1200 --task task1 --mode finetune --dry_run
```

## Usage

```
bash scripts/run_experiment.sh --model <model> --dataset <dataset[,dataset...]> --mode <mode> [options]
```

### Required Arguments

| Argument     | Description                                                                 |
| ------------ | --------------------------------------------------------------------------- |
| `--model`    | Model name (see below)                                                      |
| `--dataset`  | Dataset name, or a comma-separated list for `--mode pretrain` (see below)   |
| `--mode`     | `pretrain`, `finetune`, `train_scratch`, or `test`                          |
| `--task`     | Task ID (required for all modes except `pretrain`)                          |

### Optional Arguments

| Argument        | Default  | Description                         |
| --------------- | -------- | ----------------------------------- |
| `--gpus`        | `0`      | GPU IDs (e.g. `0,1,2,3`)           |
| `--batch_size`  | config   | Override batch size                 |
| `--max_epochs`  | config   | Override max epochs                 |
| `--num_workers` | batch_sz | Override data loader workers        |
| `--seed`        | `1`      | Random seed                         |
| `--strategy`    | `ddp`    | Training strategy                   |
| `--task_name`   | config   | Override task name from dataset cfg |
| `--dry_run`     | -        | Print command without executing     |
| `-- <args>`     | -        | Pass extra args to `main.py`        |

## Available Models

| Model        | Type         | Config                           |
| ------------ | ------------ | -------------------------------- |
| `neurostorm` | Voxel (SSM)  | `configs/models/neurostorm.yaml` |
| `swift`      | Voxel (Attn) | `configs/models/swift.yaml`      |
| `braingnn`   | Graph        | `configs/models/braingnn.yaml`   |
| `bnt`        | FC           | `configs/models/bnt.yaml`        |
| `lggnn`      | Graph        | `configs/models/lggnn.yaml`      |
| `combraintf` | FC           | `configs/models/combraintf.yaml` |
| `ibgnn`      | Graph        | `configs/models/ibgnn.yaml`      |
| `brainnetcnn`| FC (CNN)     | `configs/models/brainnetcnn.yaml`|

## Available Datasets

| Dataset      | Tasks                     | Config                             |
| ------------ | ------------------------- | ---------------------------------- |
| `hcp1200`    | task1 (sex/age), task2    | `configs/datasets/hcp1200.yaml`    |
| `adhd200`    | task3 (diagnosis)         | `configs/datasets/adhd200.yaml`    |
| `cobre`      | task3 (diagnosis, 4-cls)  | `configs/datasets/cobre.yaml`      |
| `ucla`       | task3 (diagnosis, 4-cls)  | `configs/datasets/ucla.yaml`       |
| `hcptask`    | task5 (state, 7-cls)      | `configs/datasets/hcptask.yaml`    |
| `movie`      | task5 (movie, 5-cls)      | `configs/datasets/movie.yaml`      |
| `transdiag`  | task2, task3              | `configs/datasets/transdiag.yaml`  |
| `abcd`       | pretrain only             | `configs/datasets/abcd.yaml`       |
| `ukb`        | pretrain only             | `configs/datasets/ukb.yaml`        |
| `hcpa`       | pretrain only             | `configs/datasets/hcpa.yaml`       |
| `hcpd`       | pretrain only             | `configs/datasets/hcpd.yaml`       |

## Modes

- **`pretrain`**: Self-supervised pretraining (MAE or contrastive). Config from model YAML `pretrain:` section.
- **`finetune`**: Load pretrained checkpoint and finetune on downstream task. Checkpoint path from model YAML `pretrained_ckpt:`.
- **`train_scratch`**: Train downstream task from random initialization (no pretrained weights).
- **`test`**: Evaluate only. Pass checkpoint via `-- --test_ckpt_path <path>`.

## Multi-Dataset Pretraining

`--mode pretrain` accepts a comma-separated list of dataset names so you can
pretrain on the union of several cohorts. The five pretrain-capable datasets
today are HCP1200, HCPA, HCPD, ABCD, and UKB:

```bash
bash scripts/run_experiment.sh \
    --model neurostorm \
    --dataset hcp1200,hcpa,hcpd,abcd,ukb \
    --mode pretrain \
    --gpus 1,2,3 \
    --sampling_strategy uniform_subsample
```

### Sampling Strategies

When pretraining on multiple datasets of very different sizes (e.g. UKB ~12k vs
HCPD ~600), a sampling strategy prevents large datasets from dominating.

| Strategy             | Description                                                                 |
| -------------------- | --------------------------------------------------------------------------- |
| `uniform_subsample`  | Each epoch, randomly sample at most N subjects from each dataset. Different random subset each epoch. |
| `loss_weighted`      | Resample proportional to per-dataset average loss from the previous epoch. Datasets with higher loss get more samples. |

Use `--sampling_strategy <name>` and `--max_samples_per_dataset <N>` to enable.
Without `--sampling_strategy`, all samples from all datasets are used every epoch.

Under the hood:

- Each dataset in the list must have its own YAML under `configs/datasets/`.
  `dataset_name` and `image_path` (or `image_path_fc`) are joined with commas
  and forwarded to `main.py`.
- The datamodule scans every dataset's `img/` directory and skips metadata /
  label parsing (MAE and contrastive objectives don't use labels), then
  concatenates samples into a single training set via `torch.utils.data.ConcatDataset`.
- Per-dataset splits are written to `./data/splits/<DATASET>/pretraining/split_fixed_<N>.txt`
  so each cohort keeps its own train/val/test assignment.
- The resulting `project_name` uses a dash-joined suffix, e.g.
  `hcp1200-abcd-ukb_pt_neurostorm_mae0.5`.

Comma-separated `--dataset` is rejected for `finetune`, `train_scratch`, and
`test` modes because those need a single set of labels.

## FC / Graph Preprocessing

FC-based (`bnt`, `combraintf`, `brainnetcnn`) and graph-based (`braingnn`,
`lggnn`, `ibgnn`) models don't train on raw fMRI volumes — they need per-subject
ROI time series and functional-connectivity matrices. Generate them once per
dataset / atlas with:

```bash
# bash scripts/preprocess_fc.sh <dataset> [atlas] [fc_types...]
bash scripts/preprocess_fc.sh hcp1200                                  # cc200, both FC types
bash scripts/preprocess_fc.sh hcp1200 cc200 correlation                # BNT only
bash scripts/preprocess_fc.sh adhd200 aal correlation partial_correlation
```

Outputs land under the dataset's `image_path`:

```
<image_path>/roi/<atlas>/                # per-subject ROI time series (.npy)
<image_path>/fc/<atlas>/<fc_type>/       # FC matrices
```

After preprocessing, train with the universal runner:

```bash
bash scripts/run_experiment.sh --model bnt     --dataset hcp1200 --task task1 --mode train_scratch
bash scripts/run_experiment.sh --model braingnn --dataset adhd200 --task task3 --mode train_scratch
```

## Inference

See [`scripts/run_demo.sh`](run_demo.sh) for example `demo.py` invocations
covering single-subject inference and dataset-level evaluation.

## Configuration Structure

```
scripts/configs/
├── models/          # Model architecture + hyperparameters
│   ├── neurostorm.yaml
│   ├── swift.yaml
│   ├── braingnn.yaml
│   └── ...
└── datasets/        # Dataset paths + task definitions
    ├── hcp1200.yaml
    ├── adhd200.yaml
    └── ...
```

### Model Config Format

```yaml
model: neurostorm
data_type: voxel          # voxel, fc_bnt, fc_graph
optimizer: AdamW
learning_rate: 0.00005
max_epochs: 30
batch_size: 12

model_args:               # Architecture-specific args
  embed_dim: 36
  depth: [2, 2, 6, 2]
  ...

pretrain:                 # Pretraining config (optional)
  use_mae: true
  mask_ratio: 0.5

pretrained_ckpt: ./output/neurostorm/pt_neurostorm_mae_ratio0.5.ckpt
```

### Dataset Config Format

```yaml
dataset_name: HCP1200
image_path: ./data/HCP1200_MNI_to_TRs_minmax
image_path_fc: ./data/HCP1200_MNI_to_TRs_minmax     # For FC/graph models

tasks:
  task1:
    downstream_task_id: 1
    task_name: sex
    downstream_task_type: classification
    num_classes: 2

defaults:
  dataset_split_num: 1
  limit_training_samples: 1.0
  batch_size_override: 2          # Optional: override model batch size
```

## Priority Order

Config values are resolved with the following priority (highest first):
1. CLI arguments (`--batch_size`, `--max_epochs`, etc.)
2. Task-level overrides in dataset config
3. Dataset-level defaults
4. Model config values

## Examples

See `scripts/examples/` for complete example scripts:
- `pretrain.sh` — Pretraining workflows (including multi-dataset joint pretraining)
- `finetune.sh` — Finetuning with pretrained weights
- `baselines.sh` — Training baseline models
