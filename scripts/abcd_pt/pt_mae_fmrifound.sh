TRAINER_ARGS='--accelerator gpu --max_epochs 40 --num_nodes 1 --devices 2 --strategy DDP'
MAIN_ARGS='--loggername tensorboard --clf_head_version v2 --dataset_name ABCD --image_path ./data/ABCD_MNI_to_TRs_minmax'
DATA_ARGS='--batch_size 16 --num_workers 16 --input_type rest --project_name abcd_mae_fmrifound_ratio0.9'
OPTIONAL_ARGS='--c_multiplier 2 --last_layer_full_MSA True --downstream_task sex'
RESUME_ARGS='--pretraining --use_mae --spatial_mask window --time_mask random --mask_ratio 0.9'

export CUDA_VISIBLE_DEVICES=2,3
export NCCL_P2P_DISABLE=1

python project/main.py $TRAINER_ARGS $MAIN_ARGS $DEFAULT_ARGS $DATA_ARGS $OPTIONAL_ARGS $RESUME_ARGS --dataset_split_num 1 --seed 1 --learning_rate 5e-5 --model swin4d_mae --depth 2 2 6 2 --embed_dim 36 --sequence_length 20 --first_window_size 4 4 4 4 --window_size 4 4 4 4 --img_size 96 96 96 20 --use_mamba