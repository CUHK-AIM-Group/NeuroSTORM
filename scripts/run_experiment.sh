#!/bin/bash
# Universal experiment runner for NeuroSTORM
# Usage:
#   bash scripts/run_experiment.sh --model <model> --dataset <dataset[,dataset...]> --task <task> --mode <mode> [options]
#
# Examples:
#   bash scripts/run_experiment.sh --model neurostorm --dataset hcp1200 --task task1 --mode finetune
#   bash scripts/run_experiment.sh --model neurostorm --dataset hcp1200 --mode pretrain
#   bash scripts/run_experiment.sh --model neurostorm --dataset hcp1200,hcpa,hcpd,abcd,ukb --mode pretrain
#   bash scripts/run_experiment.sh --model braingnn --dataset adhd200 --task task3 --mode train_scratch
#   bash scripts/run_experiment.sh --model neurostorm --dataset hcp1200 --task task1 --mode finetune --gpus 0,1 --batch_size 8

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_DIR="${SCRIPT_DIR}/configs"

# ============================================================
# Lightweight YAML parser (no external deps)
# ============================================================
parse_yaml() {
    local file="$1"
    local prefix="${2:-}"
    local s='[[:space:]]*'
    local w='[a-zA-Z0-9_]*'
    sed -ne "s|^\($s\)\($w\)$s:$s\"\(.*\)\"$s\$|\1\2=\3|p" \
        -e "s|^\($s\)\($w\)$s:$s\(.*\)$s\$|\1\2=\3|p" "$file" | \
    sed -ne "s|^\([a-zA-Z0-9_]*\)=\(.*\)$|${prefix}\1=\"\2\"|p"
}

yaml_get() {
    local file="$1"
    local key="$2"
    local result
    result=$(grep -E "^[[:space:]]*${key}:" "$file" 2>/dev/null | head -1 | sed 's/^[^:]*:[[:space:]]*//' | sed 's/^"\(.*\)"$/\1/' | sed 's/[[:space:]]*$//') || true
    echo "$result"
}

yaml_get_list() {
    local file="$1"
    local key="$2"
    local val
    val=$(yaml_get "$file" "$key")
    echo "$val" | sed 's/\[//g; s/\]//g; s/,/ /g'
}

yaml_get_nested() {
    local file="$1"
    local section="$2"
    local key="$3"
    local result
    result=$(awk -v sec="$section" -v k="$key" '
        /^[a-zA-Z]/ { in_sec=0 }
        $0 ~ "^"sec":" { in_sec=1; next }
        in_sec && $0 ~ "^  "k":" {
            sub(/^[^:]*:[[:space:]]*/, ""); gsub(/["\047]/, ""); print; exit
        }
    ' "$file") || true
    echo "$result"
}

yaml_get_task_field() {
    local file="$1"
    local task="$2"
    local field="$3"
    local result
    result=$(awk -v task="$task" -v field="$field" '
        /^tasks:/ { in_tasks=1; next }
        /^[a-zA-Z]/ && !/^tasks:/ { in_tasks=0 }
        in_tasks && $0 ~ "^  "task":" { in_task=1; next }
        in_tasks && /^  [a-zA-Z]/ && !($0 ~ "^  "task":") { in_task=0 }
        in_task && $0 ~ "^    "field":" {
            sub(/^[^:]*:[[:space:]]*/, ""); gsub(/["\047]/, ""); gsub(/[[:space:]]*$/, ""); print; exit
        }
    ' "$file") || true
    echo "$result"
}

yaml_get_model_arg() {
    local file="$1"
    local key="$2"
    local result
    result=$(awk -v k="$key" '
        /^model_args:/ { in_sec=1; next }
        /^[a-zA-Z]/ && !/^model_args:/ { in_sec=0 }
        in_sec && $0 ~ "^  "k":" {
            sub(/^[^:]*:[[:space:]]*/, ""); gsub(/["\047]/, ""); print; exit
        }
    ' "$file") || true
    echo "$result"
}

yaml_get_defaults_field() {
    local file="$1"
    local field="$2"
    local result
    result=$(awk -v field="$field" '
        /^defaults:/ { in_sec=1; next }
        /^[a-zA-Z]/ && !/^defaults:/ { in_sec=0 }
        in_sec && $0 ~ "^  "field":" {
            sub(/^[^:]*:[[:space:]]*/, ""); gsub(/["\047]/, ""); gsub(/[[:space:]]*$/, ""); print; exit
        }
    ' "$file") || true
    echo "$result"
}

list_to_args() {
    local val="$1"
    echo "$val" | sed 's/\[//g; s/\]//g; s/,/ /g'
}

# ============================================================
# CLI argument parsing
# ============================================================
MODEL=""
DATASET=""
TASK=""
MODE=""
GPUS=""
BATCH_SIZE=""
SEED="1"
EXTRA_ARGS=""
TASK_NAME_OVERRIDE=""
NUM_WORKERS=""
MAX_EPOCHS=""
STRATEGY="ddp"
LOAD_MODEL_PATH=""
DRY_RUN=false

print_usage() {
    cat << 'EOF'
NeuroSTORM Universal Experiment Runner

Usage:
  bash scripts/run_experiment.sh [options]

Required:
  --model <name>          Model name (neurostorm, swift, braingnn, bnt, lggnn, combraintf, ibgnn, brainnetcnn)
  --dataset <name>        Dataset name (hcp1200, hcpa, hcpd, adhd200, cobre, ucla, hcptask, movie, transdiag, abcd, ukb).
                          In --mode pretrain a comma-separated list is accepted (e.g. hcp1200,hcpa,hcpd,abcd,ukb)
                          to jointly pretrain on multiple cohorts. Rejected for non-pretrain modes.
  --mode <mode>           Run mode: pretrain, finetune, train_scratch, test

Options:
  --task <task>           Task ID (task1, task2, task3, task5). Required for finetune/train_scratch/test modes.
  --task_name <name>      Override task_name from dataset config (e.g., sex, age, MMSE_Score)
  --gpus <ids>            GPU IDs (default: 0). Example: 0,1,2,3
  --batch_size <n>        Override batch size
  --max_epochs <n>        Override max epochs
  --num_workers <n>       Override num workers
  --seed <n>              Random seed (default: 1)
  --strategy <s>          Training strategy (default: ddp)
  --load_model_path <p>   Path to checkpoint. For pretrain: resume training. For finetune: load pretrained weights.
  --dry_run               Print command without executing
  -- <extra_args>         Pass additional arguments to main.py

Examples:
  # Pretrain NeuroSTORM with MAE on HCP1200 (from scratch)
  bash scripts/run_experiment.sh --model neurostorm --dataset hcp1200 --mode pretrain

  # Joint pretraining on multiple datasets (pretrain only)
  bash scripts/run_experiment.sh --model neurostorm --dataset hcp1200,hcpa,hcpd,abcd,ukb --mode pretrain --gpus 0,1,2,3

  # Resume pretraining from checkpoint
  bash scripts/run_experiment.sh --model neurostorm --dataset hcp1200 --mode pretrain --load_model_path output/neurostorm/xxx/last.ckpt

  # Finetune NeuroSTORM on HCP sex classification
  bash scripts/run_experiment.sh --model neurostorm --dataset hcp1200 --task task1 --mode finetune --load_model_path output/neurostorm/xxx/last.ckpt

  # Train BrainGNN from scratch on ADHD200
  bash scripts/run_experiment.sh --model braingnn --dataset adhd200 --task task3 --mode train_scratch

  # Multi-GPU finetuning
  bash scripts/run_experiment.sh --model neurostorm --dataset hcp1200 --task task1 --mode finetune --gpus 0,1,2,3 --batch_size 8
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)       MODEL="$2"; shift 2 ;;
        --dataset)     DATASET="$2"; shift 2 ;;
        --task)        TASK="$2"; shift 2 ;;
        --mode)        MODE="$2"; shift 2 ;;
        --gpus)        GPUS="$2"; shift 2 ;;
        --batch_size)  BATCH_SIZE="$2"; shift 2 ;;
        --max_epochs)  MAX_EPOCHS="$2"; shift 2 ;;
        --num_workers) NUM_WORKERS="$2"; shift 2 ;;
        --seed)        SEED="$2"; shift 2 ;;
        --strategy)    STRATEGY="$2"; shift 2 ;;
        --task_name)   TASK_NAME_OVERRIDE="$2"; shift 2 ;;
        --load_model_path) LOAD_MODEL_PATH="$2"; shift 2 ;;
        --dry_run)     DRY_RUN=true; shift ;;
        --help|-h)     print_usage; exit 0 ;;
        --)            shift; EXTRA_ARGS="$*"; break ;;
        *)             echo "Error: Unknown option '$1'"; print_usage; exit 1 ;;
    esac
done

# ============================================================
# Validate required arguments
# ============================================================
if [[ -z "$MODEL" || -z "$DATASET" || -z "$MODE" ]]; then
    echo "Error: --model, --dataset, and --mode are required."
    echo ""
    print_usage
    exit 1
fi

if [[ "$MODE" != "pretrain" && -z "$TASK" ]]; then
    echo "Error: --task is required for mode '$MODE'"
    exit 1
fi

if [[ "$MODE" != "pretrain" && "$MODE" != "finetune" && "$MODE" != "train_scratch" && "$MODE" != "test" ]]; then
    echo "Error: --mode must be one of: pretrain, finetune, train_scratch, test"
    exit 1
fi

# ============================================================
# Load config files
# ============================================================
MODEL_CONFIG="${CONFIG_DIR}/models/${MODEL}.yaml"

# --dataset may be a comma-separated list (pretrain only), e.g. hcp1200,abcd,ukb
IFS=',' read -r -a DATASET_LIST <<< "$DATASET"
if [[ "${#DATASET_LIST[@]}" -gt 1 && "$MODE" != "pretrain" ]]; then
    echo "Error: multiple --dataset entries are only supported in --mode pretrain (got: $DATASET for mode $MODE)"
    exit 1
fi

# Validate every dataset config and pick the first as the source of task / default fields
DATASET_CONFIG=""
for ds_entry in "${DATASET_LIST[@]}"; do
    ds_entry_trimmed="$(echo "$ds_entry" | tr -d '[:space:]')"
    ds_config="${CONFIG_DIR}/datasets/${ds_entry_trimmed}.yaml"
    if [[ ! -f "$ds_config" ]]; then
        echo "Error: Dataset config not found: $ds_config"
        echo "Available datasets:"
        ls "${CONFIG_DIR}/datasets/" 2>/dev/null | sed 's/.yaml$//'
        exit 1
    fi
    if [[ -z "$DATASET_CONFIG" ]]; then
        DATASET_CONFIG="$ds_config"
    fi
done

if [[ ! -f "$MODEL_CONFIG" ]]; then
    echo "Error: Model config not found: $MODEL_CONFIG"
    echo "Available models:"
    ls "${CONFIG_DIR}/models/" 2>/dev/null | sed 's/.yaml$//'
    exit 1
fi

# ============================================================
# Read model config
# ============================================================
cfg_model=$(yaml_get "$MODEL_CONFIG" "model")
cfg_data_type=$(yaml_get "$MODEL_CONFIG" "data_type")
cfg_optimizer=$(yaml_get "$MODEL_CONFIG" "optimizer")
cfg_lr=$(yaml_get "$MODEL_CONFIG" "learning_rate")
cfg_max_epochs=$(yaml_get "$MODEL_CONFIG" "max_epochs")
cfg_batch_size=$(yaml_get "$MODEL_CONFIG" "batch_size")
cfg_fc_type=$(yaml_get "$MODEL_CONFIG" "fc_type")
cfg_atlas_name=$(yaml_get "$MODEL_CONFIG" "atlas_name")

# ============================================================
# Read dataset config(s) and join dataset_name / image_path with commas
# when the user passed multiple datasets for pretraining.
# ============================================================
cfg_dataset_name=""
cfg_image_path=""
for ds_entry in "${DATASET_LIST[@]}"; do
    ds_entry_trimmed="$(echo "$ds_entry" | tr -d '[:space:]')"
    ds_config="${CONFIG_DIR}/datasets/${ds_entry_trimmed}.yaml"
    ds_name=$(yaml_get "$ds_config" "dataset_name")
    if [[ "$cfg_data_type" == "voxel" ]]; then
        ds_path=$(yaml_get "$ds_config" "image_path")
    else
        ds_path=$(yaml_get "$ds_config" "image_path_fc")
        if [[ -z "$ds_path" ]]; then
            ds_path=$(yaml_get "$ds_config" "image_path")
        fi
    fi

    if [[ -z "$cfg_dataset_name" ]]; then
        cfg_dataset_name="$ds_name"
        cfg_image_path="$ds_path"
    else
        cfg_dataset_name+=",${ds_name}"
        cfg_image_path+=",${ds_path}"
    fi
done

cfg_dataset_split_num=$(yaml_get_defaults_field "$DATASET_CONFIG" "dataset_split_num")
cfg_limit_training=$(yaml_get_defaults_field "$DATASET_CONFIG" "limit_training_samples")
cfg_batch_override=$(yaml_get_defaults_field "$DATASET_CONFIG" "batch_size_override")
cfg_seq_len_override=$(yaml_get_defaults_field "$DATASET_CONFIG" "sequence_length_override")
cfg_img_size_override=$(yaml_get_defaults_field "$DATASET_CONFIG" "img_size_override")

# ============================================================
# Read task-specific config (for non-pretrain modes)
# ============================================================
if [[ "$MODE" != "pretrain" ]]; then
    task_downstream_id=$(yaml_get_task_field "$DATASET_CONFIG" "$TASK" "downstream_task_id")
    task_name=$(yaml_get_task_field "$DATASET_CONFIG" "$TASK" "task_name")
    task_type=$(yaml_get_task_field "$DATASET_CONFIG" "$TASK" "downstream_task_type")
    task_num_classes=$(yaml_get_task_field "$DATASET_CONFIG" "$TASK" "num_classes")

    task_seq_override=$(yaml_get_task_field "$DATASET_CONFIG" "$TASK" "sequence_length_override")
    task_img_override=$(yaml_get_task_field "$DATASET_CONFIG" "$TASK" "img_size_override")
    task_lr_override=$(yaml_get_task_field "$DATASET_CONFIG" "$TASK" "learning_rate_override")
    task_split_num=$(yaml_get_task_field "$DATASET_CONFIG" "$TASK" "dataset_split_num")
    task_train_split=$(yaml_get_task_field "$DATASET_CONFIG" "$TASK" "train_split")
    task_val_split=$(yaml_get_task_field "$DATASET_CONFIG" "$TASK" "val_split")

    if [[ -n "$TASK_NAME_OVERRIDE" ]]; then
        task_name="$TASK_NAME_OVERRIDE"
    fi

    if [[ -z "$task_downstream_id" ]]; then
        echo "Error: Task '$TASK' not found in dataset config: $DATASET_CONFIG"
        exit 1
    fi
fi

# ============================================================
# Resolve final values (CLI > task override > dataset default > model config)
# ============================================================
final_batch_size="${BATCH_SIZE:-${cfg_batch_override:-$cfg_batch_size}}"
final_max_epochs="${MAX_EPOCHS:-$cfg_max_epochs}"
final_lr="$cfg_lr"
final_num_workers="${NUM_WORKERS:-$final_batch_size}"
final_dataset_split_num="$cfg_dataset_split_num"

# Model architecture args
model_embed_dim=$(yaml_get_model_arg "$MODEL_CONFIG" "embed_dim")
model_depth=$(yaml_get_model_arg "$MODEL_CONFIG" "depth")
model_depths=$(yaml_get_model_arg "$MODEL_CONFIG" "depths")
model_seq_len=$(yaml_get_model_arg "$MODEL_CONFIG" "sequence_length")
model_img_size=$(yaml_get_model_arg "$MODEL_CONFIG" "img_size")
model_first_window=$(yaml_get_model_arg "$MODEL_CONFIG" "first_window_size")
model_window=$(yaml_get_model_arg "$MODEL_CONFIG" "window_size")
model_c_mult=$(yaml_get_model_arg "$MODEL_CONFIG" "c_multiplier")
model_last_msa=$(yaml_get_model_arg "$MODEL_CONFIG" "last_layer_full_MSA")
model_clf_head=$(yaml_get_model_arg "$MODEL_CONFIG" "clf_head_version")

# Resolve sequence length and img_size (task override > dataset override > model default)
if [[ "$MODE" != "pretrain" ]]; then
    if [[ -n "$task_seq_override" ]]; then
        model_seq_len="$task_seq_override"
    elif [[ -n "$cfg_seq_len_override" ]]; then
        model_seq_len="$cfg_seq_len_override"
    fi

    if [[ -n "$task_img_override" ]]; then
        model_img_size="$task_img_override"
    elif [[ -n "$cfg_img_size_override" ]]; then
        model_img_size="$cfg_img_size_override"
    fi

    if [[ -n "$task_lr_override" ]]; then
        final_lr="$task_lr_override"
    fi

    if [[ -n "$task_split_num" ]]; then
        final_dataset_split_num="$task_split_num"
    fi
fi

# ============================================================
# Set GPU environment
# ============================================================
if [[ -n "$GPUS" ]]; then
    export CUDA_VISIBLE_DEVICES="$GPUS"
    GPU_IDS="$GPUS"
else
    # Use all available GPUs by default
    GPU_IDS=""
fi
export NCCL_P2P_DISABLE=1

# ============================================================
# Build project name
# ============================================================
dataset_short=$(echo "$DATASET" | tr '[:upper:]' '[:lower:]' | tr ',' '-')
if [[ "$MODE" == "pretrain" ]]; then
    pretrain_method=""
    use_mae=$(yaml_get_nested "$MODEL_CONFIG" "pretrain" "use_mae")
    use_contrastive=$(yaml_get_nested "$MODEL_CONFIG" "pretrain" "use_contrastive")
    if [[ "$use_mae" == "true" ]]; then
        mask_ratio=$(yaml_get_nested "$MODEL_CONFIG" "pretrain" "mask_ratio")
        pretrain_method="mae${mask_ratio}"
    elif [[ "$use_contrastive" == "true" ]]; then
        contrastive_type=$(yaml_get_nested "$MODEL_CONFIG" "pretrain" "contrastive_type")
        pretrain_method="contrastive${contrastive_type}"
    fi
    project_name="${dataset_short}_pt_${cfg_model}_${pretrain_method}"
else
    project_name="${dataset_short}_${MODE:0:2}_${cfg_model}_${TASK}_${task_name}_train${cfg_limit_training:-1.0}"
fi

# ============================================================
# Build command
# ============================================================
CMD="python main.py"
if [[ -n "$GPU_IDS" ]]; then
    CMD+=" --gpu_ids $GPU_IDS"
fi
CMD+=" --max_epochs $final_max_epochs"
CMD+=" --num_nodes 1"
CMD+=" --strategy $STRATEGY"
CMD+=" --loggername tensorboard"
CMD+=" --dataset_name $cfg_dataset_name"
CMD+=" --image_path $cfg_image_path"
CMD+=" --batch_size $final_batch_size"
CMD+=" --num_workers $final_num_workers"
CMD+=" --project_name $project_name"
CMD+=" --seed $SEED"
CMD+=" --learning_rate $final_lr"
CMD+=" --model $cfg_model"

# Voxel-based model architecture args
if [[ "$cfg_data_type" == "voxel" ]]; then
    if [[ -n "$model_embed_dim" ]]; then
        CMD+=" --embed_dim $model_embed_dim"
    fi
    depth_val="${model_depth:-$model_depths}"
    if [[ -n "$depth_val" ]]; then
        CMD+=" --depth $(list_to_args "$depth_val")"
    fi
    if [[ -n "$model_seq_len" ]]; then
        CMD+=" --sequence_length $model_seq_len"
    fi
    if [[ -n "$model_img_size" ]]; then
        CMD+=" --img_size $(list_to_args "$model_img_size")"
    fi
    if [[ -n "$model_first_window" ]]; then
        CMD+=" --first_window_size $(list_to_args "$model_first_window")"
    fi
    if [[ -n "$model_window" ]]; then
        CMD+=" --window_size $(list_to_args "$model_window")"
    fi
    if [[ -n "$model_c_mult" ]]; then
        CMD+=" --c_multiplier $model_c_mult"
    fi
    if [[ -n "$model_last_msa" ]]; then
        CMD+=" --last_layer_full_MSA $model_last_msa"
    fi
    if [[ -n "$model_clf_head" ]]; then
        CMD+=" --clf_head_version $model_clf_head"
    fi
fi

# FC-based / graph-based model args
if [[ "$cfg_data_type" == "fc_graph" || "$cfg_data_type" == "fc_bnt" || "$cfg_data_type" == "fc" ]]; then
    CMD+=" --data_type $cfg_data_type"
    CMD+=" --atlas_name ${cfg_atlas_name:-cc200}"
    CMD+=" --fc_type ${cfg_fc_type:-correlation}"

    # Dynamically add all model_args
    num_rois=$(yaml_get_model_arg "$MODEL_CONFIG" "num_rois")
    [[ -n "$num_rois" ]] && CMD+=" --num_rois $num_rois"

    pooling_ratio=$(yaml_get_model_arg "$MODEL_CONFIG" "pooling_ratio")
    [[ -n "$pooling_ratio" ]] && CMD+=" --pooling_ratio $pooling_ratio"

    num_communities=$(yaml_get_model_arg "$MODEL_CONFIG" "num_communities")
    [[ -n "$num_communities" ]] && CMD+=" --num_communities $num_communities"

    dropout=$(yaml_get_model_arg "$MODEL_CONFIG" "dropout")
    [[ -n "$dropout" ]] && CMD+=" --dropout $dropout"

    hidden_dims=$(yaml_get_model_arg "$MODEL_CONFIG" "hidden_dims")
    [[ -n "$hidden_dims" ]] && CMD+=" --hidden_dims $(list_to_args "$hidden_dims")"

    pooling_sizes=$(yaml_get_model_arg "$MODEL_CONFIG" "pooling_sizes")
    [[ -n "$pooling_sizes" ]] && CMD+=" --pooling_sizes $(list_to_args "$pooling_sizes")"

    do_pooling=$(yaml_get_model_arg "$MODEL_CONFIG" "do_pooling")
    [[ -n "$do_pooling" ]] && CMD+=" --do_pooling $(list_to_args "$do_pooling" | sed 's/true/True/g; s/false/False/g')"

    hidden_size=$(yaml_get_model_arg "$MODEL_CONFIG" "hidden_size")
    [[ -n "$hidden_size" ]] && CMD+=" --hidden_size $hidden_size"

    pos_encoding=$(yaml_get_model_arg "$MODEL_CONFIG" "pos_encoding")
    [[ -n "$pos_encoding" ]] && CMD+=" --pos_encoding $pos_encoding"

    pos_embed_dim=$(yaml_get_model_arg "$MODEL_CONFIG" "pos_embed_dim")
    [[ -n "$pos_embed_dim" ]] && CMD+=" --pos_embed_dim $pos_embed_dim"

    k_neighbors=$(yaml_get_model_arg "$MODEL_CONFIG" "k_neighbors")
    [[ -n "$k_neighbors" ]] && CMD+=" --k_neighbors $k_neighbors"

    learn_graph=$(yaml_get_model_arg "$MODEL_CONFIG" "learn_graph")
    [[ -n "$learn_graph" ]] && CMD+=" --learn_graph $(echo "$learn_graph" | sed 's/true/True/; s/false/False/')"

    graph_metric=$(yaml_get_model_arg "$MODEL_CONFIG" "graph_metric")
    [[ -n "$graph_metric" ]] && CMD+=" --graph_metric $graph_metric"

    use_edge_attr=$(yaml_get_model_arg "$MODEL_CONFIG" "use_edge_attr")
    [[ -n "$use_edge_attr" ]] && CMD+=" --use_edge_attr $(echo "$use_edge_attr" | sed 's/true/True/; s/false/False/')"

    d_model=$(yaml_get_model_arg "$MODEL_CONFIG" "d_model")
    [[ -n "$d_model" ]] && CMD+=" --d_model $d_model"

    nhead=$(yaml_get_model_arg "$MODEL_CONFIG" "nhead")
    [[ -n "$nhead" ]] && CMD+=" --nhead $nhead"

    num_layers=$(yaml_get_model_arg "$MODEL_CONFIG" "num_layers")
    [[ -n "$num_layers" ]] && CMD+=" --num_layers $num_layers"

    dim_feedforward=$(yaml_get_model_arg "$MODEL_CONFIG" "dim_feedforward")
    [[ -n "$dim_feedforward" ]] && CMD+=" --dim_feedforward $dim_feedforward"

    use_community_mask=$(yaml_get_model_arg "$MODEL_CONFIG" "use_community_mask")
    [[ -n "$use_community_mask" ]] && CMD+=" --use_community_mask $(echo "$use_community_mask" | sed 's/true/True/; s/false/False/')"

    brainnetcnn_variant=$(yaml_get_model_arg "$MODEL_CONFIG" "brainnetcnn_variant")
    [[ -n "$brainnetcnn_variant" ]] && CMD+=" --brainnetcnn_variant $brainnetcnn_variant"

    e2e_channels=$(yaml_get_model_arg "$MODEL_CONFIG" "e2e_channels")
    [[ -n "$e2e_channels" ]] && CMD+=" --e2e_channels $(list_to_args "$e2e_channels")"

    e2n_channels=$(yaml_get_model_arg "$MODEL_CONFIG" "e2n_channels")
    [[ -n "$e2n_channels" ]] && CMD+=" --e2n_channels $e2n_channels"

    n2g_channels=$(yaml_get_model_arg "$MODEL_CONFIG" "n2g_channels")
    [[ -n "$n2g_channels" ]] && CMD+=" --n2g_channels $n2g_channels"

    fc_channels=$(yaml_get_model_arg "$MODEL_CONFIG" "fc_channels")
    [[ -n "$fc_channels" ]] && CMD+=" --fc_channels $(list_to_args "$fc_channels")"
fi

# Optimizer override
if [[ -n "$cfg_optimizer" && "$cfg_optimizer" != "AdamW" ]]; then
    CMD+=" --optimizer $cfg_optimizer"
fi

# Mode-specific args
case "$MODE" in
    pretrain)
        CMD+=" --pretraining"
        CMD+=" --downstream_task_type classification"
        CMD+=" --dataset_split_num ${final_dataset_split_num:-1}"

        use_mae=$(yaml_get_nested "$MODEL_CONFIG" "pretrain" "use_mae")
        use_contrastive=$(yaml_get_nested "$MODEL_CONFIG" "pretrain" "use_contrastive")

        if [[ "$use_mae" == "true" ]]; then
            CMD+=" --use_mae"
            spatial_mask=$(yaml_get_nested "$MODEL_CONFIG" "pretrain" "spatial_mask")
            time_mask=$(yaml_get_nested "$MODEL_CONFIG" "pretrain" "time_mask")
            mask_ratio=$(yaml_get_nested "$MODEL_CONFIG" "pretrain" "mask_ratio")
            [[ -n "$spatial_mask" ]] && CMD+=" --spatial_mask $spatial_mask"
            [[ -n "$time_mask" ]] && CMD+=" --time_mask $time_mask"
            [[ -n "$mask_ratio" ]] && CMD+=" --mask_ratio $mask_ratio"
        elif [[ "$use_contrastive" == "true" ]]; then
            CMD+=" --use_contrastive"
            contrastive_type=$(yaml_get_nested "$MODEL_CONFIG" "pretrain" "contrastive_type")
            [[ -n "$contrastive_type" ]] && CMD+=" --contrastive_type $contrastive_type"
        fi
        if [[ -n "$LOAD_MODEL_PATH" ]]; then
            CMD+=" --load_model_path $LOAD_MODEL_PATH"
        fi
        CMD+=" --auto_resume"
        ;;

    finetune)
        CMD+=" --downstream_task_id $task_downstream_id"
        CMD+=" --downstream_task_type $task_type"
        CMD+=" --task_name $task_name"
        CMD+=" --num_classes $task_num_classes"
        CMD+=" --dataset_split_num ${final_dataset_split_num:-1}"
        if [[ -n "$cfg_limit_training" ]]; then
            CMD+=" --limit_training_samples $cfg_limit_training"
        fi

        if [[ -n "$LOAD_MODEL_PATH" ]]; then
            CMD+=" --load_model_path $LOAD_MODEL_PATH"
        else
            pretrained_ckpt=$(yaml_get "$MODEL_CONFIG" "pretrained_ckpt")
            if [[ -n "$pretrained_ckpt" ]]; then
                CMD+=" --load_model_path $pretrained_ckpt"
            fi
        fi

        if [[ -n "$task_train_split" ]]; then
            CMD+=" --train_split $task_train_split"
        fi
        if [[ -n "$task_val_split" ]]; then
            CMD+=" --val_split $task_val_split"
        fi
        ;;

    train_scratch)
        CMD+=" --downstream_task_id $task_downstream_id"
        CMD+=" --downstream_task_type $task_type"
        CMD+=" --task_name $task_name"
        CMD+=" --num_classes $task_num_classes"
        CMD+=" --dataset_split_num ${final_dataset_split_num:-1}"
        if [[ -n "$cfg_limit_training" ]]; then
            CMD+=" --limit_training_samples $cfg_limit_training"
        fi

        if [[ -n "$task_train_split" ]]; then
            CMD+=" --train_split $task_train_split"
        fi
        if [[ -n "$task_val_split" ]]; then
            CMD+=" --val_split $task_val_split"
        fi
        ;;

    test)
        CMD+=" --test_only"
        CMD+=" --downstream_task_id $task_downstream_id"
        CMD+=" --downstream_task_type $task_type"
        CMD+=" --task_name $task_name"
        CMD+=" --num_classes $task_num_classes"
        CMD+=" --dataset_split_num ${final_dataset_split_num:-1}"
        ;;
esac

# Extra args
if [[ -n "$EXTRA_ARGS" ]]; then
    CMD+=" $EXTRA_ARGS"
fi

# ============================================================
# Execute
# ============================================================
echo "=============================================="
echo " NeuroSTORM Experiment Runner"
echo "=============================================="
echo " Model:     $cfg_model"
echo " Dataset:   $cfg_dataset_name"
echo " Mode:      $MODE"
if [[ "$MODE" != "pretrain" ]]; then
    echo " Task:      $TASK ($task_name, $task_type)"
fi
echo " GPUs:      ${CUDA_VISIBLE_DEVICES:-all}"
echo " Batch:     $final_batch_size"
echo " LR:        $final_lr"
echo " Epochs:    $final_max_epochs"
echo " Project:   $project_name"
echo "=============================================="
echo ""
echo "Command:"
echo "$CMD"
echo ""

if [[ "$DRY_RUN" == true ]]; then
    echo "[DRY RUN] Command not executed."
    exit 0
fi

eval "$CMD"
