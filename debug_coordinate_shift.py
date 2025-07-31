#!/usr/bin/env python3
"""
Diagnostic script to identify coordinate shift issues in ECG segmentation training.
This script helps identify if image resizing is causing coordinate misalignment.
"""

import os
import json
from PIL import Image
from qwen_vl_utils import process_vision_info

def analyze_image_resizing(image_path, min_pixels=512*28*28, max_pixels=1280*28*28):
    """
    Analyze how an image gets resized during the training pipeline
    """
    # Load original image
    original_img = Image.open(image_path)
    orig_width, orig_height = original_img.size
    orig_pixels = orig_width * orig_height
    
    print(f"Original image: {orig_width} x {orig_height} = {orig_pixels:,} pixels")
    
    # Simulate the same processing as get_image_info()
    content = {
        "type": "image", 
        "image": image_path,
        "min_pixels": min_pixels,
        "max_pixels": max_pixels
    }
    
    messages = [
        {"role": "user", 
         "content": [content]
        }
    ]
    
    # This is what happens in get_image_info()
    image_input, _ = process_vision_info(messages)
    
    # Check if we can get the processed image dimensions
    # Note: This might not directly give us dimensions, but we can infer scaling
    
    print(f"Min pixels constraint: {min_pixels:,}")
    print(f"Max pixels constraint: {max_pixels:,}")
    
    # Calculate scaling factors
    if orig_pixels < min_pixels:
        scale_factor = (min_pixels / orig_pixels) ** 0.5
        new_width = int(orig_width * scale_factor)
        new_height = int(orig_height * scale_factor)
        print(f"Image likely UPSCALED by factor ~{scale_factor:.3f}")
        print(f"Estimated new size: {new_width} x {new_height}")
    elif orig_pixels > max_pixels:
        scale_factor = (max_pixels / orig_pixels) ** 0.5
        new_width = int(orig_width * scale_factor)
        new_height = int(orig_height * scale_factor)
        print(f"Image likely DOWNSCALED by factor ~{scale_factor:.3f}")
        print(f"Estimated new size: {new_width} x {new_height}")
    else:
        scale_factor = 1.0
        new_width, new_height = orig_width, orig_height
        print("Image likely unchanged (within pixel constraints)")
    
    return {
        'original_size': (orig_width, orig_height),
        'estimated_new_size': (new_width, new_height),
        'scale_factor': scale_factor,
        'resized': scale_factor != 1.0
    }

def analyze_coordinate_shift(original_coords, predicted_coords, scale_factor):
    """
    Analyze if coordinate shift matches expected scaling
    """
    print("\n=== COORDINATE ANALYSIS ===")
    
    # Parse coordinates from your example
    # Ground Truth: x1="206" x2="246" x3="260"
    # Assistant: x1="275" x2="304" x3="328"
    
    gt_coords = original_coords  # [206, 246, 260]
    pred_coords = predicted_coords  # [275, 304, 328]
    
    print(f"Ground truth coordinates: {gt_coords}")
    print(f"Predicted coordinates: {pred_coords}")
    
    # Calculate shifts
    shifts = [pred - gt for pred, gt in zip(pred_coords, gt_coords)]
    print(f"Coordinate shifts: {shifts}")
    
    # Calculate what coordinates should be if properly scaled
    if scale_factor != 1.0:
        scaled_gt = [int(coord * scale_factor) for coord in gt_coords]
        print(f"Expected coordinates after scaling by {scale_factor:.3f}: {scaled_gt}")
        
        # Check if predictions match scaled ground truth
        scaled_shifts = [pred - scaled for pred, scaled in zip(pred_coords, scaled_gt)]
        print(f"Shifts from scaled ground truth: {scaled_shifts}")
    
    return shifts

def main():
    print("ECG Coordinate Shift Diagnostic Tool")
    print("=" * 50)
    
    # You'll need to provide a sample image path
    sample_image_path = input("Enter path to a sample ECG image: ").strip()
    
    if not os.path.exists(sample_image_path):
        print(f"Error: Image not found at {sample_image_path}")
        return
    
    # Analyze the image resizing
    print(f"\nAnalyzing image: {sample_image_path}")
    result = analyze_image_resizing(sample_image_path)
    
    # Example coordinate analysis from your data
    print("\n" + "=" * 50)
    print("EXAMPLE FROM YOUR DATA:")
    
    # First P complex example
    gt_coords_p = [206, 246, 260]
    pred_coords_p = [275, 304, 328]
    
    print("P Complex coordinates:")
    shifts = analyze_coordinate_shift(gt_coords_p, pred_coords_p, result['scale_factor'])
    
    # Calculate average shift
    avg_shift = sum(shifts) / len(shifts)
    print(f"Average coordinate shift: {avg_shift:.1f} pixels")
    
    print("\n" + "=" * 50)
    print("DIAGNOSIS:")
    
    if result['resized']:
        print(f"✗ Images are being resized during training (scale factor: {result['scale_factor']:.3f})")
        print("✗ Ground truth coordinates don't match resized image dimensions")
        print("\nRECOMMENDED FIXES:")
        print("1. Scale your coordinate annotations by the resize factor")
        print("2. OR disable automatic resizing (see fix below)")
    else:
        print("✓ Images are not being resized")
        print("⚠ The coordinate shift might be due to other factors")
    
    print(f"\nIf your images have scale factor {result['scale_factor']:.3f}:")
    print("Your coordinates should be multiplied by this factor.")

if __name__ == "__main__":
    main() 