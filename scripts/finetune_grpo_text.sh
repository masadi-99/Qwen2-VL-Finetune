#!/bin/bash

MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"

echo "🚀 Starting Text-Only GRPO Training"
echo "Model: $MODEL_NAME"

deepspeed src/train/train_grpo_text.py \
    --deepspeed scripts/zero3.json \
    --model_id $MODEL_NAME \
    --data_path sample_grpo_train_data.json \
    --eval_data_path sample_grpo_eval_data.json \
    --freeze_llm False \
    --lora_enable False \
    --bf16 True \
    --fp16 False \
    --disable_flash_attn2 False \
    --output_dir output/qwen2.5_7b_grpo_text \
    --num_train_epochs 3 \
    --num_generations 4 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --max_completion_length 512 \
    --max_prompt_length 1024 \
    --learning_rate 1e-5 \
    --remove_unused_columns False \
    --weight_decay 0.01 \
    --warmup_ratio 0.1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --tf32 True \
    --gradient_checkpointing True \
    --report_to tensorboard \
    --lazy_preprocess True \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 3 \
    --evaluation_strategy "steps" \
    --eval_steps 500 \
    --dataloader_num_workers 4 \
    --beta 0.04 \
    --temperature 0.9 \
    --top_p 1.0 \
    --top_k 50 \
    --repetition_penalty 1.0 \
    --epsilon 0.0001 \
    --image_folder ./dummy_images \
    --image_min_pixels 3136 \
    --image_max_pixels 12845056 \
    --video_min_pixels 100352 \
    --video_max_pixels 602112 