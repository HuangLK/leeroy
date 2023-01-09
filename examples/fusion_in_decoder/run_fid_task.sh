#!/bin/bash
# Train script.
set -eux

mode=$1
gpus=$2

export CUDA_VISIBLE_DEVICES=${gpus}
#export TOKENIZERS_PARALLELISM=false
WORK_DIR=./
CODE_DIR=./
save_path=./output

task=fid-test

model_name_or_path="Langboat/mengzi-t5-base-mt"
gpu_num=$(echo ${gpus} | grep -o "," | wc -l)

num_epochs=5
source_max_seq_len=512
target_max_seq_len=256
n_context=10
accu_grad_steps=1
batch_size=6

# 1卡
if [[ $gpu_num == 0 ]]; then
    valid_steps=10000
    learning_rate=5e-5
# 2卡
elif [[ $gpu_num == 1 ]]; then
    valid_steps=10000
    learning_rate=1e-4
fi


if [[ $mode == 'train' ]]; then
    python ${CODE_DIR}/train.py \
        --task ${task} \
        --train_file ${WORK_DIR}train.json \
        --valid_file ${WORK_DIR}valid.json \
        --n_context ${n_context} \
        --use_amp false \
        --batch_size ${batch_size} \
        --num_workers 24 \
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
fi
