#!/bin/bash
# Precision Training Script for ECG Coordinate Accuracy
# Optimized parameters for sub-pixel coordinate learning

MODEL_NAME="Qwen/Qwen2.5-VL-3B-Instruct"

# Enhanced training parameters for coordinate precision
GLOBAL_BATCH_SIZE=32          # Smaller for stable coordinate learning
BATCH_PER_DEVICE=2            # Very small batches for precision
NUM_DEVICES=8
GRAD_ACCUM_STEPS=$((GLOBAL_BATCH_SIZE / (BATCH_PER_DEVICE * NUM_DEVICES)))

export PYTHONPATH=src:$PYTHONPATH

echo "🎯 Starting precision-optimized ECG training..."
echo "📊 Training parameters:"
echo "   - Model: $MODEL_NAME"
echo "   - Global batch size: $GLOBAL_BATCH_SIZE"
echo "   - Per-device batch size: $BATCH_PER_DEVICE" 
echo "   - Gradient accumulation steps: $GRAD_ACCUM_STEPS"

# Create enhanced dataset if it doesn't exist
if [ ! -f "enhanced_ecg_data.json" ]; then
    echo "📝 Creating enhanced dataset for coordinate precision..."
    python enhance_ecg_dataset.py \
        --input ludb_conversations_wo_meta.json \
        --output enhanced_ecg_data.json \
        --factor 2.5
fi

deepspeed src/train/train_sft.py \
    --use_liger True \
    --deepspeed scripts/zero2.json \
    --model_id $MODEL_NAME \
    --data_path enhanced_ecg_data.json \
    --image_folder /scratch/users/masadi/temp_env/Qwen2-VL-Finetune/ecg_images \
    --remove_unused_columns False \
    --freeze_vision_tower False \
    --freeze_llm False \
    --freeze_merger False \
    --bf16 True \
    --fp16 False \
    --disable_flash_attn2 False \
    --output_dir output/ludb_precision_model \
    --num_train_epochs 5 \
    --per_device_train_batch_size $BATCH_PER_DEVICE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --image_min_pixels $((200 * 28 * 28)) \
    --image_max_pixels $((2000 * 28 * 28)) \
    --learning_rate 3e-6 \
    --merger_lr 3e-6 \
    --vision_lr 1e-6 \
    --weight_decay 0.01 \
    --warmup_ratio 0.15 \
    --lr_scheduler_type "cosine" \
    --logging_steps 25 \
    --tf32 True \
    --gradient_checkpointing True \
    --report_to tensorboard \
    --lazy_preprocess True \
    --save_strategy "steps" \
    --save_steps 500 \
    --eval_steps 250 \
    --save_total_limit 3 \
    --dataloader_num_workers 4 \
    --max_grad_norm 0.3 \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --adam_epsilon 1e-8

echo "🎉 Precision training complete!"
echo "📁 Model saved to: output/ludb_precision_model"
echo "🔍 Use precision_inference.py for testing coordinate accuracy" 