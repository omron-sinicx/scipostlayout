#!/usr/bin/bash

. /scipostlayout/code/layoutlmv3/object_detection/layoutlm-venv/bin/activate

MODEL_PATH=/scipostlayout/code/dit/object_detection/dit-base-224-p16-500k.pth
OUT_PATH=/scipostlayout/code/dit/object_detection
LR=0.00002
MAX_ITER=22500

python3 train_net.py \
    --config-file scipostlayout_configs/cascade/cascade_dit_base.yaml \
    --num-gpus 4 \
    MODEL.WEIGHTS $MODEL_PATH \
    SOLVER.IMS_PER_BATCH 4 \
    SOLVER.BASE_LR $LR \
    SOLVER.WARMUP_ITERS 1000 \
    SOLVER.MAX_ITER $MAX_ITER \
    SOLVER.CHECKPOINT_PERIOD 2250 \
    TEST.EVAL_PERIOD 2250 \
    OUTPUT_DIR $OUT_PATH/lr_${LR}_max_iter_${MAX_ITER}

MODEL_PATH=$OUT_PATH/lr_${LR}_max_iter_${MAX_ITER}/model_final.pth

python3 train_net.py --config-file scipostlayout_configs/cascade/cascade_dit_base.yaml --eval-only --num-gpus 1 \
        MODEL.WEIGHTS $MODEL_PATH \
        OUTPUT_DIR $OUT_PATH/results/lr_${LR}_max_iter_${MAX_ITER}
