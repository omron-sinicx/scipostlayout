poetry run python -m src.trainer.trainer.test \
    cond=c \
    dataset_dir=./download/datasets \
    job_dir=./download/pretrained_weights/layoutdm_publaynet \
    result_dir=tmp/dummy_results
