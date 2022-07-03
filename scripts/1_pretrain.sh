#!/bin/bash

: '
Script to run MLM pretraining. It takes 14 hours to preprocess dataset (tokenize group)
The main dataset has 338,000 articles, 91,304,806 sentences, 1,2 bln words (PWC has 13,5 bln)
100 -> 4400
338,000 -> 265027 * 80 = 21,200,000 ~ 30 hours
3,000 -> 3203 * 80 = 256,240 ~ 17 min
'
#  --train_file data/sample.csv \
#  --train_file /gim/lv01/datasets/cyberleninka_med_2021/cyberleninka_medical_df.csv \
#  --train_file data/texts_wo_ref.csv \


#IN_MODEL=DeepPavlov/rubert-base-cased
#OUT_MODEL=models/rubiobert2

#IN_MODEL=sberbank-ai/ruBert-base
#OUT_MODEL=models/sber_rubert


IN_MODEL=sberbank-ai/ruRoberta-large
OUT_MODEL=models/sber_roberta_128_lower

#CUDA_LAUNCH_BLOCKING=1 use for debug but dont use for parallel

#python -m torch.distributed.launch \
#  --nproc_per_node 2 1_pretrain.py \

python 1_pretrain.py \
  --train_file data/texts_wo_ref.csv \
  --cache_dir .cache \
  --model_name_or_path $IN_MODEL \
  --output_dir $OUT_MODEL \
  --do_train \
  --do_eval \
  --overwrite_output_dir \
  --fp16 \
  --fp16_full_eval \
  --preprocessing_num_workers 9 \
  --save_strategy steps \
  --evaluation_strategy steps \
  --validation_split_percentage 1 \
  --num_train_epochs 1 \
  --max_seq_length 512 \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --save_total_limit 1 \
  --logging_steps 50 \
  --save_steps 50000 \
  --eval_steps 50000 \
  --warmup_steps 20000 \
  --learning_rate 5e-5 \
  --weight_decay 0.01 \
  --do_lower_case_file do_lower_case.json \
