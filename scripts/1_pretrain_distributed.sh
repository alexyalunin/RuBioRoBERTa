#!/bin/bash
# To run distributed:
# source venv3.6/bin/activate && cd med_dataset && pip install torch==1.8.1 -f https://download.pytorch.org/whl/cu101/torch_stable.html && pip install transformers datasets
# sh scripts/1_pretr_cryst.sh 2>&1 | tee models/log.txt

IN_MODEL=sberbank-ai/ruRoberta-large
OUT_MODEL=models/sber_roberta_128_3_2e-5

python -m torch.distributed.launch \
    --nproc_per_node 3 1_pretrain.py \
    --train_file data/texts_wo_ref.csv \
    --cache_dir .cache \
    --model_name_or_path $IN_MODEL \
    --output_dir $OUT_MODEL \
    --do_train \
    --do_eval \
    --preprocessing_num_workers 9 \
    --save_strategy steps \
    --evaluation_strategy steps \
    --validation_split_percentage 1 \
    --num_train_epochs 3 \
    --max_seq_length 128 \
    --per_device_train_batch_size 70 \
    --per_device_eval_batch_size 70 \
    --save_total_limit 3 \
    --logging_steps 500 \
    --save_steps 50000 \
    --eval_steps 50000 \
    --warmup_steps 20000 \
    --learning_rate 2e-5 \
    --weight_decay 0.01 \
    --do_lower_case_file data/do_lower_case.json \
    --fp16 \
    --fp16_full_eval \
    --seed 44 \
    --max_grad_norm 0.5
# --overwrite_output_dir \

# python 1_pretrain.py \
#   --train_file data/texts_wo_ref.csv \
#   --cache_dir .cache \
#   --model_name_or_path $IN_MODEL \
#   --output_dir $OUT_MODEL \
#   --do_train \
#   --do_eval \
#   --overwrite_output_dir \
#   --preprocessing_num_workers 9 \
#   --save_strategy steps \
#   --evaluation_strategy steps \
#   --validation_split_percentage 1 \
#   --num_train_epochs 10 \
#   --max_seq_length 128 \
#   --per_device_train_batch_size 55 \
#   --per_device_eval_batch_size 55 \
#   --save_total_limit 1 \
#   --logging_steps 500 \
#   --save_steps 50000 \
#   --eval_steps 50000 \
#   --warmup_steps 20000 \
#   --learning_rate 5e-5 \
#   --weight_decay 0.01 \
#   --do_lower_case_file data/do_lower_case.json \
#   --fp16 \
#   --fp16_full_eval \
#   --local_rank 0 \
