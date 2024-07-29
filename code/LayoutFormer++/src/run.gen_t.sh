#!/usr/bin/bash

. /scipostlayout/code/LayoutFormer++/layoutformer-venv/bin/activate

# run the following commands seperately for faster training
CUDA_VISIBLE_DEVICES=0 ./scripts/scipostlayout_gen_t.sh train ../datasets ../results/gen_t basic 1 none

CUDA_VISIBLE_DEVICES=0 ./scripts/scipostlayout_gen_t.sh test ../datasets ../results/gen_t basic 1 epoch_199
