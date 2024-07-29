#!/usr/bin/bash

. /scipostlayout/code/layout-dm/layoutdm-venv/bin/activate

pip install --upgrade pip
pip install poetry
poetry install

CUDA_VISIBLE_DEVICES=0 poetry run python src/trainer/trainer/fid/train.py \
    src/trainer/trainer/config/dataset/scipostlayout.yaml \
    --out_dir download/fid_weights/FIDNetV3/scipostlayout-max50