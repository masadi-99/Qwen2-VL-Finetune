#!/usr/bin/env python3
"""
Precision Inference Script for ECG Coordinate Accuracy
Uses temporal context and multiple sampling for improved precision
"""

import argparse
import json
import os
import torch
import random
import re
from PIL import Image
from src.utils import load_pretrained_model, get_model_name_from_path
from qwen_vl_utils import process_vision_info

def process_vision_info_with_precision(messages):
    """Process vision info while preserving exact dimensions for coordinate precision"""
    modified_messages = []
    
    for msg in messages:
        modified_msg = msg.copy()
        if 'content' in modified_msg:
            modified_content = []
            for item in modified_msg['content']:
                if item.get('type') == 'image':
                    img_path = item['image']
                    try:
                        img = Image.open(img_path)
                        orig_width, orig_height = img.size
                        
                        new_item = item.copy()
                        new_item['resized_width'] = orig_width
                        new_item['resized_height'] = orig_height
                        new_item.pop('min_pixels', None)
                        new_item.pop('max_pixels', None)
                        
                        print(f"🔧 Preserving dimensions: {orig_width}×{orig_height} for coordinate precision")
                        modified_content.append(new_item)
                    except Exception as e:
                        print(f"⚠️  Warning: Could not process {img_path}: {e}")
                        modified_content.append(item)
                else:
                    modified_content.append(item)
            modified_msg['content'] = modified_content
        modified_messages.append(modified_msg)
    
    return process_vision_info(modified_messages)

def extract_coordinates(text: str):
    """Extract coordinates from model response"""
    pattern = r'x\d+="(\d+(?:\.\d+)?)"'
    coords = re.findall(pattern, text)
    return [float(c) for c in coords] if coords else []

def create_precision_prompt(user_prompt: str) -> str:
    """Enhanced prompt with temporal context for coordinate precision"""
    
    temporal_context = """
ECG Analysis with Temporal Precision:
- Image dimensions: 1000×224 pixels
- Time duration: 2000ms (2 seconds)  
- Temporal resolution: 1 pixel = 2ms
- Coordinate range: 0-999 pixels
- Sampling rate: 500Hz (1 sample per pixel)

Temporal landmarks:
- Pixel 0 = 0ms (start)
- Pixel 250 = 500ms (quarter)
- Pixel 500 = 1000ms (middle)
- Pixel 750 = 1500ms (three-quarter)
- Pixel 999 = 1998ms (end)

"""
    
    precision_instructions = """
Provide coordinates with maximum precision:
- Use exact pixel values (integers or decimals)
- Ensure temporal order: x1 < x2 < x3
- Format: <points x1="start" x2="peak" x3="end" alt="wave_type">wave_type</points>
"""
    
    return f"{temporal_context}\n{user_prompt}\n\n{precision_instructions}"

def precision_inference(model, processor, conversation, num_attempts=3):
    """
    Run inference with precision optimization
    
    Args:
        model: Loaded VL model
        processor: Model processor
        conversation: Conversation history
        num_attempts: Number of inference attempts for consensus
    """
    
    # Apply temporal context to the last user message
    if conversation and conversation[-1].get('role') == 'user':
        last_content = conversation[-1]['content']
        for item in last_content:
            if item.get('type') == 'text':
                item['text'] = create_precision_prompt(item['text'])
                break
    
    # Process vision inputs with coordinate preservation
    image_inputs, video_inputs = process_vision_info_with_precision(conversation)
    
    # Apply chat template
    prompt = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    
    # Prepare inputs
    inputs = processor(
        text=[prompt],
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt",
        padding=False,
        do_resize=False  # Critical: preserve coordinates
    ).to(model.device)
    
    responses = []
    coordinates_list = []
    
    # Multiple inference attempts for coordinate consensus
    for attempt in range(num_attempts):
        # Precision-optimized generation parameters
        generation_kwargs = {
            "max_new_tokens": 256,
            "temperature": 0.05 + (attempt * 0.02),  # Slightly vary temperature
            "do_sample": True,
            "top_p": 0.85,
            "top_k": 40,
            "repetition_penalty": 1.03,
            "no_repeat_ngram_size": 3,
        }
        
        with torch.no_grad():
            output_ids = model.generate(**inputs, **generation_kwargs)
        
        response = processor.batch_decode(
            output_ids[:, inputs['input_ids'].shape[-1]:],
            skip_special_tokens=True
        )[0]
        
        responses.append(response)
        
        # Extract coordinates for consensus
        coords = extract_coordinates(response)
        if coords:
            coordinates_list.append(coords)
        
        print(f"🔍 Attempt {attempt+1}: {len(coords)} coordinates extracted")
    
    # Coordinate consensus analysis
    if coordinates_list and len(coordinates_list) >= 2:
        print(f"\n📊 COORDINATE CONSENSUS ANALYSIS:")
        for i, coords in enumerate(coordinates_list):
            print(f"   Attempt {i+1}: {coords}")
        
        # Calculate average coordinates for consensus
        if all(len(coords) == len(coordinates_list[0]) for coords in coordinates_list):
            avg_coords = []
            for i in range(len(coordinates_list[0])):
                avg_val = sum(coord_set[i] for coord_set in coordinates_list) / len(coordinates_list)
                avg_coords.append(round(avg_val, 1))
            
            print(f"   Consensus: {avg_coords}")
            
            # Find response closest to consensus
            best_response_idx = 0
            min_deviation = float('inf')
            
            for i, coords in enumerate(coordinates_list):
                deviation = sum(abs(coords[j] - avg_coords[j]) for j in range(len(coords)))
                if deviation < min_deviation:
                    min_deviation = deviation
                    best_response_idx = i
            
            print(f"   Best match: Attempt {best_response_idx+1} (deviation: {min_deviation:.1f})")
            return responses[best_response_idx], avg_coords
    
    # Return first response if no consensus possible
    return responses[0], coordinates_list[0] if coordinates_list else []

def analyze_coordinate_accuracy(pred_coords, gt_coords):
    """Analyze coordinate accuracy vs ground truth"""
    if not pred_coords or not gt_coords or len(pred_coords) != len(gt_coords):
        return None
    
    shifts = [abs(p - g) for p, g in zip(pred_coords, gt_coords)]
    avg_shift = sum(shifts) / len(shifts)
    max_shift = max(shifts)
    
    # Convert to temporal accuracy
    temporal_shifts_ms = [shift * 2 for shift in shifts]  # Each pixel = 2ms
    avg_temporal_shift = avg_shift * 2
    
    return {
        'pixel_shifts': shifts,
        'avg_pixel_shift': avg_shift,
        'max_pixel_shift': max_shift,
        'temporal_shifts_ms': temporal_shifts_ms,
        'avg_temporal_shift_ms': avg_temporal_shift,
        'sub_pixel_accuracy': avg_shift < 1.0,
        'millisecond_accuracy': avg_temporal_shift < 4.0  # Within 2 pixels = 4ms
    }

def main():
    parser = argparse.ArgumentParser(description="Precision ECG coordinate inference")
    parser.add_argument("--model_path", required=True, help="Path to trained model")
    parser.add_argument("--data_file", required=True, help="Test data JSON file")
    parser.add_argument("--image_folder", required=True, help="Folder containing ECG images")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to test")
    parser.add_argument("--attempts", type=int, default=3, help="Inference attempts per sample")
    
    args = parser.parse_args()
    
    # Load model
    print("🔧 Loading precision-trained model...")
    model_name = get_model_name_from_path(args.model_path)
    processor, model = load_pretrained_model(
        model_base="Qwen/Qwen2.5-VL-3B-Instruct",
        model_path=args.model_path,
        device_map="cuda",
        model_name=model_name,
        use_flash_attn=True
    )
    
    # Load test data
    with open(args.data_file, 'r') as f:
        dataset = json.load(f)
    
    # Test samples
    test_samples = random.sample(dataset, min(args.num_samples, len(dataset)))
    
    print(f"\n🎯 TESTING COORDINATE PRECISION")
    print("=" * 60)
    
    accuracy_results = []
    
    for i, sample in enumerate(test_samples):
        print(f"\n📊 Sample {i+1}/{len(test_samples)}")
        print(f"Image: {os.path.basename(sample['image'])}")
        
        # Prepare conversation
        image_path = os.path.join(args.image_folder, os.path.basename(sample["image"]))
        
        conversations = sample.get('conversations', [])
        if not conversations:
            continue
            
        # Create test conversation (first user query)
        user_msg = conversations[0]
        test_conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": user_msg["value"].replace("<image>", "").strip()}
                ]
            }
        ]
        
        # Get ground truth coordinates
        gt_response = conversations[1]["value"] if len(conversations) > 1 else ""
        gt_coords = extract_coordinates(gt_response)
        
        # Run precision inference
        response, pred_coords = precision_inference(model, processor, test_conversation, args.attempts)
        
        print(f"Ground Truth: {gt_coords}")
        print(f"Predicted:    {pred_coords}")
        print(f"Response:     {response[:100]}...")
        
        # Analyze accuracy
        if gt_coords and pred_coords:
            accuracy = analyze_coordinate_accuracy(pred_coords, gt_coords)
            if accuracy:
                accuracy_results.append(accuracy)
                print(f"🎯 Accuracy Analysis:")
                print(f"   Avg pixel shift: {accuracy['avg_pixel_shift']:.1f}")
                print(f"   Avg temporal shift: {accuracy['avg_temporal_shift_ms']:.1f}ms")
                print(f"   Sub-pixel accuracy: {accuracy['sub_pixel_accuracy']}")
                print(f"   Millisecond accuracy: {accuracy['millisecond_accuracy']}")
    
    # Overall results
    if accuracy_results:
        print(f"\n🏆 OVERALL PRECISION RESULTS")
        print("=" * 40)
        avg_pixel_accuracy = sum(r['avg_pixel_shift'] for r in accuracy_results) / len(accuracy_results)
        avg_temporal_accuracy = sum(r['avg_temporal_shift_ms'] for r in accuracy_results) / len(accuracy_results)
        sub_pixel_rate = sum(r['sub_pixel_accuracy'] for r in accuracy_results) / len(accuracy_results)
        millisecond_rate = sum(r['millisecond_accuracy'] for r in accuracy_results) / len(accuracy_results)
        
        print(f"Average pixel accuracy: {avg_pixel_accuracy:.1f} pixels")
        print(f"Average temporal accuracy: {avg_temporal_accuracy:.1f}ms")
        print(f"Sub-pixel accuracy rate: {sub_pixel_rate:.1%}")
        print(f"Millisecond accuracy rate: {millisecond_rate:.1%}")
        
        if avg_pixel_accuracy < 2.0:
            print("✅ Excellent coordinate precision achieved!")
        elif avg_pixel_accuracy < 5.0:
            print("✅ Good coordinate precision achieved!")
        else:
            print("⚠️ Coordinate precision needs improvement")

if __name__ == "__main__":
    main() 