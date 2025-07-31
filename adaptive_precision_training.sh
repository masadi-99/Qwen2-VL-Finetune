#!/bin/bash
# Adaptive Precision Training Script - Works with all granularity levels
# Automatically adjusts training parameters based on coordinate granularity

MODEL_NAME="Qwen/Qwen2.5-VL-3B-Instruct"

# 🎯 DETECT GRANULARITY from dataset filename
echo "🔍 Detecting granularity level from dataset files..."

if [ -f "ludb_conversations_wo_meta_ultra_fine_granularity.json" ]; then
    GRANULARITY="ultra_fine"
    DATA_FILE="ludb_conversations_wo_meta_ultra_fine_granularity.json"
    IMG_WIDTH=1000
    MS_PER_PIXEL=2.0
elif [ -f "ludb_conversations_wo_meta_fine_granularity.json" ]; then
    GRANULARITY="fine"
    DATA_FILE="ludb_conversations_wo_meta_fine_granularity.json"
    IMG_WIDTH=500
    MS_PER_PIXEL=4.0
elif [ -f "ludb_conversations_wo_meta_medium_granularity.json" ]; then
    GRANULARITY="medium"
    DATA_FILE="ludb_conversations_wo_meta_medium_granularity.json"
    IMG_WIDTH=250
    MS_PER_PIXEL=8.0
elif [ -f "ludb_conversations_wo_meta_coarse_granularity.json" ]; then
    GRANULARITY="coarse"
    DATA_FILE="ludb_conversations_wo_meta_coarse_granularity.json"
    IMG_WIDTH=200
    MS_PER_PIXEL=10.0
else
    echo "❌ No granularity-specific dataset found!"
    echo "Please run fixed_ecg_image_generation.py first to generate dataset."
    exit 1
fi

echo "✅ Detected granularity: $GRANULARITY"
echo "📊 Configuration:"
echo "   - Data file: $DATA_FILE"
echo "   - Image width: $IMG_WIDTH pixels"
echo "   - Temporal resolution: ${MS_PER_PIXEL}ms per pixel"

# 🎯 ADAPTIVE TRAINING PARAMETERS based on granularity
case $GRANULARITY in
    "ultra_fine")
        # Ultra-fine needs very careful training
        GLOBAL_BATCH_SIZE=16
        BATCH_PER_DEVICE=1
        LEARNING_RATE="2e-6"
        EPOCHS=7
        SAVE_STEPS=200
        echo "🎯 Ultra-fine granularity: Maximum precision training"
        ;;
    "fine")
        # Fine granularity - good balance
        GLOBAL_BATCH_SIZE=32
        BATCH_PER_DEVICE=2
        LEARNING_RATE="3e-6"
        EPOCHS=5
        SAVE_STEPS=300
        echo "🎯 Fine granularity: Balanced precision training"
        ;;
    "medium")
        # Medium granularity - faster training possible
        GLOBAL_BATCH_SIZE=64
        BATCH_PER_DEVICE=4
        LEARNING_RATE="5e-6"
        EPOCHS=4
        SAVE_STEPS=400
        echo "🎯 Medium granularity: Efficient precision training"
        ;;
    "coarse")
        # Coarse granularity - standard training
        GLOBAL_BATCH_SIZE=128
        BATCH_PER_DEVICE=8
        LEARNING_RATE="8e-6"
        EPOCHS=3
        SAVE_STEPS=500
        echo "🎯 Coarse granularity: Fast training"
        ;;
esac

NUM_DEVICES=8
GRAD_ACCUM_STEPS=$((GLOBAL_BATCH_SIZE / (BATCH_PER_DEVICE * NUM_DEVICES)))

export PYTHONPATH=src:$PYTHONPATH

echo "🚀 Starting adaptive precision training..."
echo "⚙️  Training parameters:"
echo "   - Granularity: $GRANULARITY"
echo "   - Global batch size: $GLOBAL_BATCH_SIZE"
echo "   - Per-device batch size: $BATCH_PER_DEVICE"
echo "   - Learning rate: $LEARNING_RATE"
echo "   - Epochs: $EPOCHS"
echo "   - Gradient accumulation: $GRAD_ACCUM_STEPS"

# Create enhanced dataset if it doesn't exist
ENHANCED_FILE="enhanced_${DATA_FILE}"
if [ ! -f "$ENHANCED_FILE" ]; then
    echo "📝 Creating enhanced dataset for $GRANULARITY granularity..."
    python enhance_ecg_dataset.py \
        --input "$DATA_FILE" \
        --output "$ENHANCED_FILE" \
        --factor 2.5
fi

# Calculate appropriate image constraints based on width
MIN_PIXELS=$(( (IMG_WIDTH * 224) / 4 ))  # Quarter of image pixels as minimum
MAX_PIXELS=$(( (IMG_WIDTH * 224) * 4 ))  # 4x image pixels as maximum

deepspeed src/train/train_sft.py \
    --use_liger True \
    --deepspeed scripts/zero2.json \
    --model_id $MODEL_NAME \
    --data_path "$ENHANCED_FILE" \
    --image_folder /scratch/users/masadi/temp_env/Qwen2-VL-Finetune/ecg_images \
    --remove_unused_columns False \
    --freeze_vision_tower False \
    --freeze_llm False \
    --freeze_merger False \
    --bf16 True \
    --fp16 False \
    --disable_flash_attn2 False \
    --output_dir "output/ludb_precision_${GRANULARITY}" \
    --num_train_epochs $EPOCHS \
    --per_device_train_batch_size $BATCH_PER_DEVICE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --image_min_pixels $MIN_PIXELS \
    --image_max_pixels $MAX_PIXELS \
    --learning_rate $LEARNING_RATE \
    --merger_lr $LEARNING_RATE \
    --vision_lr $(python3 -c "print(float('$LEARNING_RATE') / 3)") \
    --weight_decay 0.01 \
    --warmup_ratio 0.15 \
    --lr_scheduler_type "cosine" \
    --logging_steps 25 \
    --tf32 True \
    --gradient_checkpointing True \
    --report_to tensorboard \
    --lazy_preprocess True \
    --save_strategy "steps" \
    --save_steps $SAVE_STEPS \
    --eval_steps $(($SAVE_STEPS / 2)) \
    --save_total_limit 3 \
    --dataloader_num_workers 4 \
    --max_grad_norm 0.3 \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --adam_epsilon 1e-8

echo "🎉 Adaptive precision training complete!"
echo "📁 Model saved to: output/ludb_precision_${GRANULARITY}"
echo "🔍 Test with: python precision_inference.py --model_path output/ludb_precision_${GRANULARITY}"
echo ""
echo "📊 Training Summary:"
echo "   - Granularity: $GRANULARITY ($MS_PER_PIXEL ms/pixel)"
echo "   - Image width: $IMG_WIDTH pixels"  
echo "   - Coordinate range: 0-$((IMG_WIDTH-1))"
echo "   - Expected accuracy: ±1-2 pixels (±${MS_PER_PIXEL}-$((MS_PER_PIXEL*2))ms)" 