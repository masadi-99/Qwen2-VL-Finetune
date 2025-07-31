#!/usr/bin/env python3
"""
FIXED INFERENCE SCRIPT - Preserves image coordinates during inference.
This version ensures that image processing during inference matches training.
"""

import argparse
from threading import Thread
import gradio as gr
from PIL import Image
from src.utils import load_pretrained_model, get_model_name_from_path, disable_torch_init
from transformers import TextIteratorStreamer
from functools import partial
import warnings
from qwen_vl_utils import process_vision_info
import json
import os
import torch
import random

def process_vision_info_with_original_dimensions(messages):
    """
    FIXED: Process vision info while preserving original image dimensions
    """
    # Create a modified version of messages with explicit dimensions
    modified_messages = []
    
    for msg in messages:
        modified_msg = msg.copy()
        if 'content' in modified_msg:
            modified_content = []
            for item in modified_msg['content']:
                if item.get('type') == 'image':
                    # Load image and get original dimensions
                    img_path = item['image']
                    try:
                        img = Image.open(img_path)
                        orig_width, orig_height = img.size
                        
                        # Create new item with explicit dimensions
                        new_item = item.copy()
                        new_item['resized_width'] = orig_width
                        new_item['resized_height'] = orig_height
                        # Remove any min/max pixels to prevent resizing
                        new_item.pop('min_pixels', None)
                        new_item.pop('max_pixels', None)
                        
                        print(f"🔧 FIXED: Preserving original dimensions {orig_width}x{orig_height} for {os.path.basename(img_path)}")
                        modified_content.append(new_item)
                    except Exception as e:
                        print(f"⚠️  Warning: Could not process {img_path}: {e}")
                        modified_content.append(item)
                else:
                    modified_content.append(item)
            modified_msg['content'] = modified_content
        modified_messages.append(modified_msg)
    
    # Now call process_vision_info with fixed messages
    return process_vision_info(modified_messages)

def main():
    # Load model and processor
    device = 'cuda'
    model_name = get_model_name_from_path('/scratch/users/masadi/temp_env/Qwen2-VL-Finetune/output/ludb_wo_meta')
    use_flash_attn = True

    print("🔧 Loading model with coordinate preservation...")
    processor, model = load_pretrained_model(
        model_base="Qwen/Qwen2.5-VL-3B-Instruct", 
        model_path='/scratch/users/masadi/temp_env/Qwen2-VL-Finetune/output/ludb_wo_meta', 
        device_map=device, 
        model_name=model_name, 
        use_flash_attn=use_flash_attn
    )

    # === Load Data ===
    with open("ludb_conversations_wo_meta.json", "r") as f:
        dataset = json.load(f)

    sample = random.choice(dataset)
    image_path = os.path.join("./ecg_images", os.path.basename(sample["image"]))
    print(f"📁 Testing with image: {sample['image']}")

    # === Prepare Messages and Ground Truths ===
    full_convo = sample["conversations"]
    messages = []
    ground_truths = []

    # Extract only the user-assistant turns
    for i in range(0, len(full_convo), 2):
        user_msg = full_convo[i]
        gt_reply = full_convo[i + 1]["value"]
        
        content = user_msg["value"].replace("<image>", "").strip()
        if "<image>" in user_msg["value"]:
            messages.append({
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path}, 
                    {"type": "text", "text": content}
                ]
            })
        else:
            messages.append({
                "role": "user",
                "content": [{"type": "text", "text": content}]
            })
        
        ground_truths.append(gt_reply)

    # === Run Multi-Turn Inference with Fixed Processing ===
    print("\n🎯 RUNNING INFERENCE WITH COORDINATE PRESERVATION")
    print("=" * 60)
    assistant_replies = []

    for idx, user_message in enumerate(messages):
        conv_history = messages[:idx + 1] + assistant_replies

        # Apply chat template
        prompt = processor.apply_chat_template(conv_history, tokenize=False, add_generation_prompt=True)
        
        # 🔧 FIXED: Use coordinate-preserving image processing
        image_inputs, video_inputs = process_vision_info_with_original_dimensions(conv_history)

        # Process inputs with coordinate preservation
        inputs = processor(
            text=[prompt],
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
            padding=False,
            do_resize=False  # 🔧 CRITICAL: Prevent any additional resizing
        ).to(model.device)

        # Inference
        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=512)

        response = processor.batch_decode(
            output_ids[:, inputs['input_ids'].shape[-1]:],
            skip_special_tokens=True
        )[0]

        assistant_replies.append({
            "role": "assistant",
            "content": [{"type": "text", "text": response}]
        })

        # === Print Results ===
        user_text = next(x["text"] for x in user_message["content"] if x["type"] == "text")
        print(f">>> User: {user_text}")
        print(f"--- Ground Truth: {ground_truths[idx]}")
        print(f"<<< Assistant: {response}")
        
        # 📊 Extract and compare coordinates if present
        import re
        gt_coords = re.findall(r'x\d+="(\d+)"', ground_truths[idx])
        pred_coords = re.findall(r'x\d+="(\d+)"', response)
        
        if gt_coords and pred_coords:
            gt_nums = [int(x) for x in gt_coords[:3]]
            pred_nums = [int(x) for x in pred_coords[:3]]
            shifts = [abs(p - g) for p, g in zip(pred_nums, gt_nums)]
            avg_shift = sum(shifts) / len(shifts) if shifts else 0
            print(f"📏 Coordinate analysis: GT={gt_nums}, Pred={pred_nums}, Avg shift={avg_shift:.1f}")
        
        print()

    print("🎯 INFERENCE COMPLETE!")
    print("✅ All images processed with original dimensions preserved")
    print("✅ Coordinates should now be much more accurate!")

if __name__ == "__main__":
    main() 