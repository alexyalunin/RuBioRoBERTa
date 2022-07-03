#!/bin/bash

# export CUDA_VISIBLE_DEVICES=1 && zsh ./scripts/RuMedBench/2_RuMedBench.sh

declare -A MODEL_TO_OUTDIR
MODEL_TO_OUTDIR[sberbank-ai/ruBert-base]=sber-ruBert-base
MODEL_TO_OUTDIR[sberbank-ai/ruRoberta-large]=sber-ruRoberta-large
MODEL_TO_OUTDIR[DeepPavlov/rubert-base-cased]=DeepPavlov
MODEL_TO_OUTDIR[models/rubiobert2]=RuBioDeepPavlov
MODEL_TO_OUTDIR[models/sber_rubert]=sber_rubert
MODEL_TO_OUTDIR[models/sber_roberta_128]=sber_roberta_128
MODEL_TO_OUTDIR[models/sber_roberta_128_5_2e-5]=sber_roberta_128_5
MODEL_TO_OUTDIR[models/sber_roberta_128_3_2e-5]=sber_roberta_128_3
MODEL_TO_OUTDIR[models/sber_roberta_512]=sber_roberta_512
MODEL_TO_OUTDIR[models/sber_roberta_128_lower]=sber_roberta_128_lower

MAIN_DIR=models/RuMedBench_out

for key value in ${(kv)MODEL_TO_OUTDIR}; do
    OUT_DIR=$MAIN_DIR/$value
    echo $key $OUT_DIR

    IN_MODEL=$key

    TASKS=(RuMedTop3 RuMedSymptomRec RuMedDaNet RuMedNLI)

    for TASK in ${TASKS[@]}; do
        python 2_classify.py \
            --task_name $TASK \
            --cache_dir .cache \
            --model_name_or_path $IN_MODEL \
            --output_dir $OUT_DIR \
            --do_train \
            --do_eval \
            --do_predict \
            --overwrite_output_dir \
            --fp16 \
            --fp16_full_eval \
            --num_train_epochs 10 \
            --max_seq_length 128 \
            --per_device_train_batch_size 32 \
            --gradient_accumulation_steps 1 \
            --per_device_eval_batch_size 32 \
            --save_strategy no \
            --logging_steps 100 \
            --save_strategy no \
            --evaluation_strategy epoch \
            --learning_rate 3e-5 \
            --weight_decay 0.01 \
            --lr_scheduler_type cosine \
            --warmup_ratio 0.3 \
            --seed 2 \
            || { echo 'fine-tuning failed' ; exit 1; }
    done

    # RuMedNER
    python 2_ner.py \
        --task_name ner \
        --cache_dir .cache \
        --model_name_or_path $IN_MODEL \
        --output_dir $OUT_DIR \
        --do_train \
        --do_eval \
        --do_predict \
        --overwrite_output_dir \
        --fp16 \
        --fp16_full_eval \
        --num_train_epochs 10 \
        --max_seq_length 128 \
        --per_device_train_batch_size 32 \
        --gradient_accumulation_steps 1 \
        --per_device_eval_batch_size 32 \
        --save_strategy no \
        --logging_steps 100 \
        --save_strategy no \
        --evaluation_strategy epoch \
        --learning_rate 3e-5 \
        --weight_decay 0.01 \
        --lr_scheduler_type cosine \
        --warmup_ratio 0.3 \
        || { echo 'fine-tuning failed' ; exit 1; }

    python RuMedBench-draft/code/eval.py --out_dir $OUT_DIR
done


for key value in ${(kv)MODEL_TO_OUTDIR}; do
    OUT_DIR=$MAIN_DIR/$value
    python RuMedBench-draft/code/eval.py --out_dir $OUT_DIR
done
