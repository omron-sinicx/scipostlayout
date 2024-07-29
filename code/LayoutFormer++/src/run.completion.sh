#!/usr/bin/bash

. /scipostlayout/code/LayoutFormer++/layoutformer-venv/bin/activate

# run the following commands seperately for faster training
CUDA_VISIBLE_DEVICES=0 ./scripts/scipostlayout_completion.sh train ../datasets ../results/completion basic 1 none

CUDA_VISIBLE_DEVICES=0 ./scripts/scipostlayout_completion.sh test ../datasets ../results/completion basic 1 epoch_199
