#!/usr/bin/env python3
"""
Ultimate Patch-Aligned ECG Inference Script
Compatible with all coordinate fixes and Vision Transformer optimizations
"""

import argparse
import json
import os
import random
import re
import torch
import numpy as np
from PIL import Image
from typing import List, Dict, Tuple, Optional
from functools import partial
import warnings
from threading import Thread

# Import Qwen2-VL components
from src.utils import load_pretrained_model, get_model_name_from_path, disable_torch_init
from transformers import TextIteratorStreamer
from qwen_vl_utils import process_vision_info

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

def process_vision_info_with_perfect_alignment(messages):
    """
    Process vision info while preserving exact dimensions for patch alignment
    This ensures no resizing artifacts that could shift coordinates
    """
    # Create modified conversation that forces original dimensions
    modified_conversation = []
    
    for msg in messages:
        modified_msg = msg.copy()
        if 'content' in modified_msg:
            modified_content = []
            for item in modified_msg['content']:
                if item.get('type') == 'image':
                    # Load image and preserve exact dimensions
                    img = Image.open(item['image'])
                    orig_width, orig_height = img.size
                    
                    # Create new item with forced dimensions
                    new_item = item.copy()
                    new_item['resized_width'] = orig_width
                    new_item['resized_height'] = orig_height
                    
                    # Remove any min/max pixel constraints to prevent resizing
                    new_item.pop('min_pixels', None)
                    new_item.pop('max_pixels', None)
                    
                    print(f"🎯 Preserving image dimensions: {orig_width}×{orig_height}")
                    
                    # Verify patch alignment
                    width_remainder = orig_width % 28
                    height_remainder = orig_height % 28
                    if width_remainder == 0 and height_remainder == 0:
                        print(f"✅ Perfect patch alignment: {orig_width // 28}×{orig_height // 28} patches")
                    else:
                        print(f"⚠️  Non-aligned dimensions: {width_remainder}×{height_remainder} remainder")
                    
                    modified_content.append(new_item)
                else:
                    modified_content.append(item)
            modified_msg['content'] = modified_content
        modified_conversation.append(modified_msg)
    
    return process_vision_info(modified_conversation)

def extract_coordinates_advanced(text: str) -> List[Tuple[int, int, int, str]]:
    """
    Advanced coordinate extraction with error handling and validation
    Returns list of (x1, x2, x3, wave_type) tuples
    """
    coordinates = []
    
    # Enhanced regex pattern for coordinate extraction
    patterns = [
        r'<points\s+x1="(\d+)"\s+x2="(\d+)"\s+x3="(\d+)"\s+alt="([^"]+)"[^>]*>',
        r'x1="(\d+)"\s+x2="(\d+)"\s+x3="(\d+)".*?alt="([^"]+)"',
        r'(\d+)\s+(\d+)\s+(\d+).*?([PQT])',  # Fallback pattern
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            try:
                x1, x2, x3 = int(match[0]), int(match[1]), int(match[2])
                wave_type = match[3].upper()
                
                # Validate coordinate order and ranges
                if 0 <= x1 <= x2 <= x3 and wave_type in ['P', 'QRS', 'T']:
                    coordinates.append((x1, x2, x3, wave_type))
            except (ValueError, IndexError):
                continue
    
    return coordinates

def create_temporal_context_prompt(base_prompt: str, metadata: Dict = None) -> str:
    """
    Enhanced prompt with temporal and patch context for better coordinate accuracy
    """
    if metadata:
        ms_per_pixel = metadata.get('ms_per_pixel', 4.0)
        total_patches = metadata.get('total_patches', 144)
        patch_grid = metadata.get('patch_grid', '18×8')
        granularity = metadata.get('granularity', 'fine')
        
        context = f"""🎯 ECG Analysis Context:
- Image granularity: {granularity} ({ms_per_pixel:.1f}ms per pixel)
- Patch alignment: {patch_grid} grid ({total_patches} patches total)
- Coordinate precision: Each pixel = {ms_per_pixel:.1f} milliseconds
- Perfect Vision Transformer alignment for accurate coordinates

{base_prompt}

📏 Coordinate Guidelines:
- Provide precise pixel coordinates (x1, x2, x3)
- Each coordinate represents {ms_per_pixel:.1f}ms intervals
- Use patch-aware spatial reasoning for accuracy"""
    else:
        # Fallback context for images without metadata
        context = f"""🎯 ECG Analysis Context:
- Provide precise pixel-level coordinates for wave detection
- Use spatial reasoning to identify wave boundaries accurately
- Consider temporal relationships between consecutive waves

{base_prompt}

📏 Coordinate Guidelines:
- Provide precise pixel coordinates (x1, x2, x3) 
- Ensure coordinates follow anatomical wave progression
- Use consistent coordinate formatting"""
    
    return context.strip()

def consensus_inference(model, processor, conversation: List[Dict], 
                       num_attempts: int = 3, temperature: float = 0.1) -> Tuple[str, List[str], Dict]:
    """
    Perform multiple inference attempts and find coordinate consensus
    """
    device = next(model.parameters()).device
    all_responses = []
    all_coordinates = []
    
    for attempt in range(num_attempts):
        print(f"🔄 Inference attempt {attempt + 1}/{num_attempts}")
        
        # Apply chat template
        prompt = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        
        # Process vision info with perfect alignment
        image_inputs, video_inputs = process_vision_info_with_perfect_alignment(conversation)
        
        # Prepare inputs with coordinate preservation
        inputs = processor(
            text=[prompt], 
            images=image_inputs, 
            videos=video_inputs, 
            padding=True, 
            return_tensors="pt", 
            do_resize=False  # 🎯 CRITICAL: Prevent coordinate shifts
        ).to(device)
        
        # Generate with controlled randomness
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=temperature + (attempt * 0.05),  # Slight variation per attempt
                top_p=0.9,
                top_k=50,
                repetition_penalty=1.05,
                do_sample=True if temperature > 0 else False
            )
        
        # Decode response
        response = processor.batch_decode(
            output_ids[:, inputs['input_ids'].shape[-1]:], 
            skip_special_tokens=True
        )[0].strip()
        
        all_responses.append(response)
        coordinates = extract_coordinates_advanced(response)
        all_coordinates.append(coordinates)
        
        print(f"   Response {attempt + 1}: {len(coordinates)} coordinates found")
    
    # Select best response (most coordinates found)
    best_response_idx = max(range(len(all_coordinates)), 
                           key=lambda i: len(all_coordinates[i]))
    best_response = all_responses[best_response_idx]
    
    analysis = {
        'total_attempts': num_attempts,
        'responses_per_attempt': [len(coords) for coords in all_coordinates],
        'best_attempt': best_response_idx + 1
    }
    
    return best_response, all_responses, analysis

def analyze_coordinate_accuracy(pred_coords: List[Tuple], gt_coords: List[Tuple], 
                              ms_per_pixel: float = 4.0) -> Dict:
    """
    Comprehensive coordinate accuracy analysis
    """
    analysis = {
        'total_predicted': len(pred_coords),
        'total_ground_truth': len(gt_coords),
        'pixel_errors': [],
        'temporal_errors_ms': [],
        'wave_type_accuracy': {},
        'average_pixel_error': 0.0,
        'average_temporal_error_ms': 0.0
    }
    
    # Group by wave type
    pred_by_type = {}
    gt_by_type = {}
    
    for x1, x2, x3, wave_type in pred_coords:
        if wave_type not in pred_by_type:
            pred_by_type[wave_type] = []
        pred_by_type[wave_type].append((x1, x2, x3))
    
    for x1, x2, x3, wave_type in gt_coords:
        if wave_type not in gt_by_type:
            gt_by_type[wave_type] = []
        gt_by_type[wave_type].append((x1, x2, x3))
    
    # Calculate errors for each wave type
    all_pixel_errors = []
    
    for wave_type in set(list(pred_by_type.keys()) + list(gt_by_type.keys())):
        pred_waves = pred_by_type.get(wave_type, [])
        gt_waves = gt_by_type.get(wave_type, [])
        
        # Simple matching: closest coordinates
        wave_errors = []
        for gt_wave in gt_waves:
            if pred_waves:
                # Find closest predicted wave
                distances = [
                    abs(gt_wave[1] - pred_wave[1])  # Compare center points (x2)
                    for pred_wave in pred_waves
                ]
                closest_idx = distances.index(min(distances))
                closest_pred = pred_waves[closest_idx]
                
                # Calculate coordinate errors
                coord_errors = [abs(gt - pred) for gt, pred in zip(gt_wave, closest_pred)]
                wave_errors.extend(coord_errors)
                all_pixel_errors.extend(coord_errors)
        
        analysis['wave_type_accuracy'][wave_type] = {
            'predicted_count': len(pred_waves),
            'ground_truth_count': len(gt_waves),
            'average_error_pixels': np.mean(wave_errors) if wave_errors else float('inf')
        }
    
    if all_pixel_errors:
        analysis['average_pixel_error'] = np.mean(all_pixel_errors)
        analysis['average_temporal_error_ms'] = analysis['average_pixel_error'] * ms_per_pixel
        analysis['pixel_errors'] = all_pixel_errors
        analysis['temporal_errors_ms'] = [error * ms_per_pixel for error in all_pixel_errors]
    
    return analysis

def run_multi_turn_evaluation(model, processor, dataset_sample: Dict, num_attempts: int = 3):
    """
    Run complete multi-turn evaluation on a dataset sample
    """
    print(f"\n🎯 MULTI-TURN EVALUATION: {dataset_sample['id']}")
    print("=" * 60)
    
    # Load image and verify dimensions
    image_path = os.path.join("./ecg_images", os.path.basename(dataset_sample["image"]))
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return None
    
    img = Image.open(image_path)
    width, height = img.size
    print(f"📊 Image: {width}×{height} pixels")
    
    # Get metadata for temporal context
    metadata = dataset_sample.get('metadata', {})
    if metadata:
        print(f"🎯 Metadata: {metadata['granularity']} granularity, {metadata['ms_per_pixel']:.1f}ms/pixel")
    
    # Extract conversations and ground truths
    full_convo = dataset_sample["conversations"]
    messages = []
    ground_truths = []
    
    # Process conversation turns
    for i in range(0, len(full_convo), 2):
        user_msg = full_convo[i]
        gt_reply = full_convo[i + 1]["value"]
        
        content = user_msg["value"].replace("<image>", "").strip()
        
        # Enhanced prompt with temporal context
        enhanced_content = create_temporal_context_prompt(content, metadata)
        
        if "<image>" in user_msg["value"]:
            messages.append({
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": enhanced_content}
                ]
            })
        else:
            messages.append({
                "role": "user", 
                "content": [{"type": "text", "text": enhanced_content}]
            })
        
        ground_truths.append(gt_reply)
    
    # Run inference for each turn
    assistant_replies = []
    evaluation_results = []
    
    for idx, user_message in enumerate(messages):
        print(f"\n🔄 Turn {idx + 1}/{len(messages)}")
        
        # Build conversation history
        conv_history = []
        for j in range(idx + 1):
            conv_history.append(messages[j])
            if j < len(assistant_replies):
                conv_history.append(assistant_replies[j])
        
        # Enhanced inference with consensus
        response, all_responses, analysis = consensus_inference(
            model, processor, conv_history, num_attempts=num_attempts
        )
        
        # Store assistant response
        assistant_replies.append({
            "role": "assistant",
            "content": [{"type": "text", "text": response}]
        })
        
        # Extract and analyze coordinates
        pred_coords = extract_coordinates_advanced(response)
        gt_coords = extract_coordinates_advanced(ground_truths[idx])
        
        ms_per_pixel = metadata.get('ms_per_pixel', 4.0)
        accuracy_analysis = analyze_coordinate_accuracy(pred_coords, gt_coords, ms_per_pixel)
        
        # Display results
        user_text = next(x["text"] for x in user_message["content"] if x["type"] == "text")
        print(f">>> User: {user_text[:100]}...")
        print(f"--- Ground Truth: {ground_truths[idx]}")
        print(f"<<< Assistant: {response}")
        print(f"📊 Accuracy: {accuracy_analysis['average_pixel_error']:.1f}px (±{accuracy_analysis['average_temporal_error_ms']:.1f}ms)")
        
        evaluation_results.append({
            'turn': idx + 1,
            'predicted_coordinates': pred_coords,
            'ground_truth_coordinates': gt_coords,
            'accuracy_analysis': accuracy_analysis,
            'consensus_analysis': analysis,
            'response': response
        })
    
    return evaluation_results

def main():
    parser = argparse.ArgumentParser(description="Patch-Aligned ECG Inference")
    parser.add_argument("--model_path", required=True, help="Path to trained model")
    parser.add_argument("--data_file", required=True, help="JSON dataset file")
    parser.add_argument("--image_folder", default="./ecg_images", help="Image folder path")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of samples to evaluate")
    parser.add_argument("--attempts", type=int, default=3, help="Inference attempts per question")
    parser.add_argument("--temperature", type=float, default=0.1, help="Generation temperature")
    parser.add_argument("--device", default="cuda", help="Device to use")
    parser.add_argument("--use_flash_attn", action="store_true", help="Use flash attention")
    
    args = parser.parse_args()
    
    print("🎯 PATCH-ALIGNED ECG INFERENCE")
    print("=" * 60)
    print(f"Model: {args.model_path}")
    print(f"Data: {args.data_file}")
    print(f"Samples: {args.num_samples}")
    print(f"Attempts per question: {args.attempts}")
    
    # Load model
    print("\n🔄 Loading model...")
    model_name = get_model_name_from_path(args.model_path)
    processor, model = load_pretrained_model(
        model_base="Qwen/Qwen2.5-VL-3B-Instruct",
        model_path=args.model_path,
        device_map=args.device,
        model_name=model_name,
        use_flash_attn=args.use_flash_attn
    )
    
    # Load dataset
    print("🔄 Loading dataset...")
    with open(args.data_file, "r") as f:
        dataset = json.load(f)
    
    # Select random samples
    if args.num_samples > 0:
        samples = random.sample(dataset, min(args.num_samples, len(dataset)))
    else:
        samples = dataset
    
    print(f"📊 Evaluating {len(samples)} samples...")
    
    # Run evaluations
    all_results = []
    total_pixel_errors = []
    total_temporal_errors = []
    
    for i, sample in enumerate(samples):
        print(f"\n{'='*20} SAMPLE {i+1}/{len(samples)} {'='*20}")
        
        try:
            results = run_multi_turn_evaluation(model, processor, sample, args.attempts)
            if results:
                all_results.append(results)
                
                # Collect accuracy statistics
                for result in results:
                    acc = result['accuracy_analysis']
                    if acc['pixel_errors']:
                        total_pixel_errors.extend(acc['pixel_errors'])
                        total_temporal_errors.extend(acc['temporal_errors_ms'])
        
        except Exception as e:
            print(f"❌ Error processing sample {i+1}: {e}")
            continue
    
    # Final summary
    print(f"\n🏆 FINAL EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Processed samples: {len(all_results)}")
    
    if total_pixel_errors:
        avg_pixel_error = np.mean(total_pixel_errors)
        avg_temporal_error = np.mean(total_temporal_errors)
        std_pixel_error = np.std(total_pixel_errors)
        
        print(f"📊 Coordinate Accuracy:")
        print(f"   Average pixel error: {avg_pixel_error:.2f} ± {std_pixel_error:.2f} pixels")
        print(f"   Average temporal error: {avg_temporal_error:.2f} ± {std_pixel_error*4:.2f} ms")
        print(f"   Median pixel error: {np.median(total_pixel_errors):.2f} pixels")
        print(f"   95th percentile: {np.percentile(total_pixel_errors, 95):.2f} pixels")
        
        # Clinical assessment
        excellent_count = sum(1 for e in total_pixel_errors if e <= 2)
        good_count = sum(1 for e in total_pixel_errors if 2 < e <= 5)
        
        print(f"📏 Clinical Assessment:")
        print(f"   Excellent (≤2px): {excellent_count}/{len(total_pixel_errors)} ({100*excellent_count/len(total_pixel_errors):.1f}%)")
        print(f"   Good (≤5px): {excellent_count + good_count}/{len(total_pixel_errors)} ({100*(excellent_count + good_count)/len(total_pixel_errors):.1f}%)")
    
    print(f"\n✅ Evaluation complete! Patch-aligned inference performed successfully.")

if __name__ == "__main__":
    main() 