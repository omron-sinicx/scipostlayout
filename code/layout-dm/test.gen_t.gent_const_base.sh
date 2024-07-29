#!/usr/bin/bash

. /scipostlayout/code/layout-dm/layoutdm-venv/bin/activate

pip install poetry
poetry install

cond=c
JOB_DIR=/scipostlayout/code/layout-dm/tmp/jobs/scipostlayout/layoutdm_xxxxxxxx
RESULT_DIR=/scipostlayout/code/layout-dm/result_dir

CUDA_VISIBLE_DEVICES=0 poetry run python3 -m src.trainer.trainer.test \
    cond=$cond \
    job_dir=$JOB_DIR \
    result_dir=${RESULT_DIR}/${cond} \
    gen_const_path="/scipostlayout/code/Paper-to-Layout/results/test/prompt_base.json"
    # is_validation=true
