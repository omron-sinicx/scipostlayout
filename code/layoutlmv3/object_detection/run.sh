#!/usr/bin/bash

. /scipostlayout/code/layoutlmv3/object_detection/layoutlm-venv/bin/activate

MODEL_PATH=/scipostlayout/code/layoutlmv3/object_detection/layoutlmv3-base/pytorch_model.bin
OUT_PATH=/scipostlayout/code/layoutlmv3/object_detection

LR=0.0002
MAX_ITER=22500

python3 train_net.py --config-file cascade_layoutlmv3.yaml --num-gpus 4 \
        MODEL.WEIGHTS $MODEL_PATH \
        PUBLAYNET_DATA_DIR_TRAIN /scipostlayout/scipostlayout/poster/png/train \
        PUBLAYNET_DATA_DIR_TEST /scipostlayout/scipostlayout/poster/png/dev \
        SOLVER.GRADIENT_ACCUMULATION_STEPS 1 \
        SOLVER.IMS_PER_BATCH 4 \
        SOLVER.BASE_LR $LR \
        SOLVER.WARMUP_ITERS 1000 \
        SOLVER.MAX_ITER $MAX_ITER \
        SOLVER.CHECKPOINT_PERIOD 2250 \
        TEST.EVAL_PERIOD 2250 \
        OUTPUT_DIR $OUT_PATH/lr_${LR}_max_iter_${MAX_ITER}

python3 train_net.py --config-file cascade_layoutlmv3.yaml --eval-only --num-gpus 4 \
        MODEL.WEIGHTS $OUT_PATH/lr_0.0002_max_iter_22500/model_final.pth \
        PUBLAYNET_DATA_DIR_TEST /scipostlayout/scipostlayout/poster/png/test \
        OUTPUT_DIR $OUT_PATH/lr_0.0002_max_iter_22500
