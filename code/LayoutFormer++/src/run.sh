#!/usr/bin/bash

. /scipostlayout/code/LayoutFormer++/layoutformer-venv/bin/activate

# run the following commands seperately for faster training
CUDA_VISIBLE_DEVICES=0 ./scripts/scipostlayout_gen_t.sh train ../datasets ../results/gen_t basic 1 none
CUDA_VISIBLE_DEVICES=0 ./scripts/scipostlayout_gen_ts.sh train ../datasets ../results/gen_ts basic 1 none
CUDA_VISIBLE_DEVICES=0 ./scripts/scipostlayout_gen_r.sh train ../datasets ../results/gen_r basic 1 none
CUDA_VISIBLE_DEVICES=0 ./scripts/scipostlayout_completion.sh train ../datasets ../results/completion basic 1 none
CUDA_VISIBLE_DEVICES=0 ./scripts/scipostlayout_refinement.sh train ../datasets ../results/refinement basic 1 none

CUDA_VISIBLE_DEVICES=0 ./scripts/scipostlayout_gen_t.sh test ../datasets ../results/gen_t basic 1 epoch_199
CUDA_VISIBLE_DEVICES=0 ./scripts/scipostlayout_gen_ts.sh test ../datasets ../results/gen_ts basic 1 epoch_199
CUDA_VISIBLE_DEVICES=0 ./scripts/scipostlayout_gen_r.sh test ../datasets ../results/gen_r basic 1 epoch_199
CUDA_VISIBLE_DEVICES=0 ./scripts/scipostlayout_completion.sh test ../datasets ../results/completion basic 1 epoch_199
CUDA_VISIBLE_DEVICES=0 ./scripts/scipostlayout_refinement.sh test ../datasets ../results/refinement basic 1 epoch_199