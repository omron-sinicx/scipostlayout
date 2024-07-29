#!/usr/bin/bash

. /scipostlayout/code/layout-dm/layoutdm-venv/bin/activate

pip install poetry
poetry install

CUDA_VISIBLE_DEVICES=0 bash bin/train.sh scipostlayout layoutdm