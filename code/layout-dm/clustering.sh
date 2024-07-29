#!/usr/bin/bash

. /scipostlayout/code/layout-dm/layoutdm-venv/bin/activate

pip install poetry
poetry install

CUDA_VISIBLE_DEVICES=0 poetry run python bin/clustering_coordinates.py src/trainer/trainer/config/dataset/scipostlayout.yaml kmeans --result_dir download/clustering_weights