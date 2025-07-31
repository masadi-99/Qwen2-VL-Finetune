#!/bin/bash
# FIXED TRAINING SCRIPT - Preserves original ECG image dimensions

MODEL_NAME="Qwen/Qwen2.5-VL-3B-Instruct"
GLOBAL_BATCH_SIZE=128
BATCH_PER_DEVICE=4
NUM_DEVICES=8
GRAD_ACCUM_STEPS=$((GLOBAL_BATCH_SIZE / (BATCH_PER_DEVICE * NUM_DEVICES)))

export PYTHONPATH=src:$PYTHONPATH

deepspeed src/train/train_sft.py \
    --use_liger True \
    --deepspeed scripts/zero2.json \
    --model_id $MODEL_NAME \
    --data_path /scratch/users/masadi/temp_env/Qwen2-VL-Finetune/ludb_conversations_wo_meta.json \
    --image_folder /scratch/users/masadi/temp_env/Qwen2-VL-Finetune/ecg_images \
    --remove_unused_columns False \
    --freeze_vision_tower False \
    --freeze_llm False \
    --freeze_merger False \
    --bf16 True \
    --fp16 False \
    --disable_flash_attn2 False \
    --output_dir output/ludb_wo_meta_fixed \
    --num_train_epochs 1 \
    --per_device_train_batch_size $BATCH_PER_DEVICE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --image_min_pixels $((200 * 28 * 28)) \
    --image_max_pixels $((2000 * 28 * 28)) \
    --learning_rate 1e-5 \
    --merger_lr 1e-5 \
    --vision_lr 2e-6 \
    --weight_decay 0.1 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --gradient_checkpointing True \
    --report_to tensorboard \
    --lazy_preprocess True \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 2 \
    --dataloader_num_workers 4

# KEY CHANGES:
# 1. Changed image_min_pixels from 401,408 to 156,800 (200*28*28)
#    This is BELOW your ECG image size (224,000), so no upscaling
# 2. Changed image_max_pixels from 401,408 to 1,568,000 (2000*28*28)  
#    This is ABOVE your ECG image size (224,000), so no downscaling
# 3. Your ECG images (224,000 pixels) now fall within the range and won't be resized! 