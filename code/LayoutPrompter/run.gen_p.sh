#!/usr/bin/bash

. /scipostlayout/code/LayoutPrompter/layoutprompter-venv/bin/activate

export OPENAI_API_KEY='xxx'
export OPENAI_ORGANIZATION='xxx'

CUDA_VISIBLE_DEVICES=0 python3 src/constraint_explicit.py \
    --task genp \
    --base_dir /scipostlayout/code/LayoutPrompter \
    --fid_model_path /scipostlayout/code/layout-dm/download/fid_weights/FIDNetV3/scipostlayout-max50/model_best.pth.tar \
    --mmd_dir /scipostlayout/code/Paper-to-Layout/mmd \
    # --use_saved_response