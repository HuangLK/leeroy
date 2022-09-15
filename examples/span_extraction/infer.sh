#!/bin/bash
# Train script.
set -eux

python infer.py \
    --task span-test \
    --predict_file /home/huangliankai/code/Leeroy/examples/span_extraction/test.json \
    --batch_size 1 \
    --use_fast_tokenizer true \
    --num_workers 1 \
    --save_path ./output \
    --max_seq_len 512 \
    --threshold 0.3 \
    --model_name_or_path /platform_tech/huangliankai/pretrained-models/uie-medical-base-torch \
    --ckpt_path /platform_tech/huangliankai/pretrained-models/uie-medical-base-torch
