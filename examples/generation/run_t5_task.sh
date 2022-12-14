#!/bin/bash
# Train script.
set -eux

mode=$1
gpus=$2

export CUDA_VISIBLE_DEVICES=${gpus}
#export TOKENIZERS_PARALLELISM=false
WORK_DIR=/home/huangliankai/code/Leeroy/examples/generation
CODE_DIR=./
save_path=./output

task=t5-test

model_name_or_path="Langboat/mengzi-t5-base-mt"
gpu_num=$(echo ${gpus} | grep -o "," | wc -l)

num_epochs=1
source_max_seq_len=512
target_max_seq_len=128
accu_grad_steps=1

# 1卡
if [[ $gpu_num == 0 ]]; then
    batch_size=56
    valid_steps=5000
    learning_rate=1e-4
# 4卡
    batch_size=56
    valid_steps=1000
    learning_rate=1e-5
fi


if [[ $mode == 'train' ]]; then
    python ${CODE_DIR}/train.py \
        --task ${task} \
        --train_file ${WORK_DIR}/train.csv \
        --valid_file ${WORK_DIR}/valid.csv \
        --use_amp false \
        --batch_size ${batch_size} \
        --num_workers 48 \
        --optimizer "AdamW" \
        --scheduler "linear" \
        --learning_rate ${learning_rate} \
        --warmup_steps_ratio 0.1 \
        --weight_decay 0.01 \
        --num_epochs ${num_epochs} \
        --accu_grad_steps ${accu_grad_steps} \
        --log_steps 10 \
        --valid_steps ${valid_steps} \
        --save_path ${save_path} \
        --model_name_or_path ${model_name_or_path} \
        --use_fast_tokenizer true \
        --source_max_seq_len ${source_max_seq_len} \
        --target_max_seq_len ${target_max_seq_len}
    exit $?
elif [[ $mode == 'infer' ]]; then
    echo 'todo.'
    exit $?
fi
