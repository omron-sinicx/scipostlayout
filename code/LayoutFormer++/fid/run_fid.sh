#!/bin/bash

#$-l rt_G.small=1
#$-l h_rt=2:00:00
#$-j y
#$-cwd

source /etc/profile.d/modules.sh
module load gcc/12.2.0 python/3.10/3.10.10 cuda/12.1

source /scratch/acd13848fc/paper2poster/LayoutGeneration/LayoutFormer++/layoutformer-venv/bin/activate

# ./scripts/scipostlayout_gen_t.sh train ../datasets ../tmp basic 1 none

./scripts/scipostlayout_gen_t.sh test ../datasets ../tmp basic 1 epoch_49

EPOCH=200
LR=3e-4
BATCH_SIZE=64
DATA_DIR=../datasets
OUT_DIR=../net

$COMMAND train.py --${MODE} \
    --dataset scipostlayout \
    --refinement_sort_by_pos_before_sort_by_label \
    --gen_ts_shuffle_before_sort_by_label \
    --gen_t_sort_by_pos_before_sort_by_label \
    --completion_sort_by_pos \
    --completion_sort_by_pos_before_sort_by_label \
    --ugen_sort_by_pos \
    --ugen_sort_by_pos_before_sort_by_label \
    --gen_r_compact \
    --gen_r_add_unk_token \
    --gen_r_discrete_before_induce_relations \
    --gen_r_shuffle_before_sort_by_label \
    --max_num_elements 50 \
    --gaussian_noise_mean 0.0 \
    --gaussian_noise_std 0.01 \
    --train_bernoulli_beta 1.0 \
    --test_bernoulli_beta 1.0 \
    --data_dir ${DATA_DIR} \
    --out_dir ${OUT_DIR} \
    --num_layers 8 \
    --nhead 8 \
    --d_model 512 \
    --dropout 0.1 \
    --share_embedding \
    --epoch ${EPOCH} \
    --batch_size ${BATCH_SIZE} \
    --eval_batch_size 1 \
    --decode_max_length 150 \
    --num_pos_embed 400 \
    --lr ${LR} \
    --warmup_num_steps 1000 \
    --gradient_accumulation 1 \
    --bbox_format ltwh \
    --discrete_x_grid 128 \
    --discrete_y_grid 128 \
    --trainer basic \
    --deepscale_config ../src/scripts/ds_config.json \
    --eval_seed 500 \
    --enable_task_measure \
    --eval_interval 50 \
    --eval_ckpt_tag none \
    --add_sep_token \
    --sort_by_dict \
    --load_vocab