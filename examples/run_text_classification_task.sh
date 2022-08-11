#!/bin/bash
# Train script.
set -ux

export CUDA_VISIBLE_DEVICES=4
#export TOKENIZERS_PARALLELISM=false

WORK_DIR=./
save_path=./output

python -m train \
    --task "text_clf" \
    --num_classes 2 \
    --train_file ${WORK_DIR}/train.csv \
    --valid_file ${WORK_DIR}/val.csv \
    --use_amp false \
    --batch_size 48 \
    --num_workers 8 \
    --optimizer "AdamW" \
    --scheduler "linear" \
    --learning_rate 1e-5 \
    --warmup_steps_ratio 0.1 \
    --weight_decay 0.01 \
    --num_epochs 10 \
    --accu_grad_steps 1 \
    --log_steps 10 \
    --valid_steps 50 \
    --save_path ${save_path} \
    --model_name_or_path "hfl/chinese-macbert-base" \
    --use_fast_tokenizer false \
    --max_seq_len 512
