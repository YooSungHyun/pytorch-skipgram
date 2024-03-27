#!/bin/bash
export WANDB_DISABLED=false
export WANDB_PROJECT="history-nlp"
export WANDB_ENTITY="bart_tadev"
export WANDB_NAME="cbow"

python ./cpu_train.py \
    --output_dir=model_outputs_cpu/ \
    --train_datasets_path=./raw_data/500_bart_test.csv \
    --seed=42 \
    --num_workers=12 \
    --per_device_train_batch_size=32 \
    --per_device_eval_batch_size=32 \
    --accumulate_grad_batches=1 \
    --max_epochs=100 \
    --learning_rate=0.0001 \
    --weight_decay=0.0001 \
    --warmup_ratio=0.01 \
    --div_factor=10 \
    --final_div_factor=10 \
    --log_every_n=100